"""GPT Responses-API agents for CUA-Lite.

Self-contained agents that talk to OpenAI's Responses API via liteLLM.
They inherit from ``BaseAgent`` — no adapter, no processor, no generate_fn.

Desktop use relies on OpenAI's native ``{"type":"computer"}`` tool, which
produces ``computer_call`` output items. Mobile use has no Android computer
tool, so it uses provider-flat function tools for tap/swipe/etc.

Usage:
    from lite.agents.models.gpt.agent import GPTDesktopUseAgent

    agent = GPTDesktopUseAgent(model_id="gpt-5.5")
    result = await agent.sample(env, max_steps=20)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, ClassVar

from lite.agents.core.agent import (
    BaseAgent,
)
from lite.agents.core.agent.hooks import SampleHook
from lite.agents.core.agent.utils.image_markers import (
    mark_provider_visible_image_index,
    provider_visible_image_indices_from_marked_blocks,
    strip_provider_visible_image_index_marker,
)
from lite.agents.core.agent.utils.loop import (
    append_lite_rl_step,
    build_lite_rl_sample,
    dispatch_lite_step_hooks,
    execute_lite_turn,
    finalize_lite_sample,
    mark_steps_truncated,
    record_lite_env_result,
)
from lite.agents.core.agent.utils.mobile_finish import mobile_finish_guidance
from lite.agents.core.agent.utils.retry import acompletion_with_retry

# ``litellm`` is imported lazily inside _aresponses_with_retry, NOT here:
# importing it runs a blocking GitHub fetch of litellm's model-cost map, and an
# env main transitively imports this module (env → lite.agents → gpt agent), so
# a module-level import would freeze the env-server loop when the reaper's
# registry._import_all() runs. litellm is only needed when the agent calls the
# API. Tests patch the real ``litellm.aresponses`` (not this module's attribute).
from lite.agents.models.gpt.action_space import (
    GPTDesktopActionSpace,
    GPTDesktopGroundingPointActionSpace,
    GPTMobileActionSpace,
)
from lite.agents.models.gpt.utils.history import (
    _append_desktop_provider_feedback,
    _append_mobile_provider_feedback,
    _compress_computer_use_history,
    _strip_images,
)
from lite.agents.models.gpt.utils.image_io import (
    _DEFAULT_IMAGE_DETAIL,
    _call_api_with_actual_dim,
    _resize_for_api,
)
from lite.agents.models.gpt.utils.loop import (
    _append_gpt_terminal_tool_feedback,
)
from lite.agents.models.gpt.utils.parse import (
    GPTParsedOutput,
    parse_gpt_mobile_output_items_with_provenance,
    parse_output_items_with_provenance,
)
from lite.agents.models.gpt.utils.responses import (
    _TRUNCATION_INCOMPLETE_REASONS,
    _CacheStats,
    _log_trajectory_usage_summary,
    _log_usage_and_check_cache,
    _normalized_gpt_response,
    _raise_if_response_failed,
)
from lite.core import (
    LiteMessage,
    LiteRLSample,
    LiteRLStep,
    LiteSample,
)
from lite.core.messages.final import (
    pop_model_output_error,
)
from lite.core.metadata import LiteBaseMetadata, LiteCUAMetadata
from lite.core.tools.schemas import (
    tool_schema_name,
)
from lite.gym.base import LiteBaseEnv
from lite.utils.image import decode_image, normalize_resolution
from lite.utils.timer import timed

logger = logging.getLogger(__name__)

# Desktop use has no default system prompt: the native computer tool is enough
# for the model to understand its role. Override via ``system_prompt`` when a
# desktop run needs extra policy text.


def _provider_visible_image_indices_from_gpt_input_items(
    input_items: list[dict[str, Any]],
) -> tuple[int, ...]:
    """Indices of the images this Responses request actually sends the model.

    Walks the three GPT input shapes that can carry a screenshot — message
    content blocks, ``computer_call_output``, and ``function_call_output`` — and
    reads the marker each one carries. The values index ``LiteSample.images``.
    """
    image_blocks: list[dict[str, Any]] = []
    for item in input_items:
        content = item.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "input_image":
                    image_blocks.append(block)

        item_type = item.get("type")
        output = item.get("output")
        if (
            item_type == "computer_call_output"
            and isinstance(output, dict)
            and output.get("type") == "computer_screenshot"
        ):
            image_blocks.append(output)
        elif item_type == "function_call_output" and isinstance(output, list):
            for block in output:
                if isinstance(block, dict) and block.get("type") == "input_image":
                    image_blocks.append(block)
    return provider_visible_image_indices_from_marked_blocks(image_blocks)


def _strip_gpt_image_markers(input_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip provider-visible image-index markers from GPT Responses image blocks."""
    stripped_items: list[dict[str, Any]] = []
    for item in input_items:
        stripped_item = dict(item)

        content = item.get("content")
        if isinstance(content, list):
            stripped_item["content"] = [
                strip_provider_visible_image_index_marker(block)
                if isinstance(block, dict) and block.get("type") == "input_image"
                else block
                for block in content
            ]

        item_type = item.get("type")
        output = item.get("output")
        if (
            item_type == "computer_call_output"
            and isinstance(output, dict)
            and output.get("type") == "computer_screenshot"
        ):
            stripped_item["output"] = strip_provider_visible_image_index_marker(output)
        elif item_type == "function_call_output" and isinstance(output, list):
            stripped_item["output"] = [
                strip_provider_visible_image_index_marker(block)
                if isinstance(block, dict) and block.get("type") == "input_image"
                else block
                for block in output
            ]

        stripped_items.append(stripped_item)
    return stripped_items


# =============================================================================
# GPT Agent
# =============================================================================

GPT_API_KWARGS_DEFAULTS: dict[str, Any] = {
    # Reasoning depth, shared across the three API families so their rows sit
    # at a comparable setting. OpenAI's own default for gpt-5.5 is also
    # ``medium``; the grounding subclass drops to ``none`` (see its defaults).
    "reasoning_effort": "medium",
    # Surface concise reasoning summaries as Lite ``inline_reasoning`` content.
    "reasoning_summary": "concise",
    "truncation": "auto",
    # Chain turns through Responses API state by default. Set false only when
    # replaying full provider history yourself.
    "chain_previous_response": True,
    # Serializes function tools; native ``computer_call.actions`` can still batch
    # multiple GUI actions inside one provider call.
    "parallel_tool_calls": False,
}


@dataclass
class _GPTBaseAgent(BaseAgent):
    """Shared base for the GPT Responses-API agents (desktop + mobile).

    Not registered directly. It owns provider request plumbing, retry, common
    api_kwargs defaults, and shared prompt/reasoning setup. Concrete subclasses
    provide the tool surface, parser, and sampling loop for their environment.
    """

    # --- class-attr knobs (NOT dataclass fields) ---
    _API_KWARGS_DEFAULTS: ClassVar[dict[str, Any]] = GPT_API_KWARGS_DEFAULTS
    _RETRY_LOG_NAME: ClassVar[str | None] = None
    # Coordinate frame for click/tap parsing. Desktop uses provider-processed
    # image dims; fixed-frame subclasses parse against the sent frame.
    _normalize_by_actual_dims: ClassVar[bool] = True

    # --- shared dataclass fields, grouped by concern (base defaults; a subclass
    # may override any — mobile re-declares function_tool_strict, api_retry_max,
    # system_prompt). ---

    # Identity.
    model_id: str = "gpt-5.5"

    # API transport. ``__post_init__`` fills missing ``api_kwargs`` from the
    # subclass defaults; retry uses capped exponential backoff.
    api_kwargs: dict[str, Any] = field(default_factory=dict)
    api_key: str | None = None
    api_base: str | None = None
    api_retry_max: int = 24
    api_retry_base_delay: float = 2.0
    api_retry_max_delay: float = 60.0

    # Prompt (subclass default carries the dialect's base prompt; ``None`` =
    # rely on the native computer tool, as desktop does).
    system_prompt: str | None = None

    # I/O behavior. ``resolution`` controls the image sent to the provider.
    # Coordinate parsing uses either provider-processed dims (desktop) or sent
    # dims (mobile/grounding), selected by ``_normalize_by_actual_dims``.
    resolution: tuple[int, int] | None = None
    function_tool_strict: bool = False
    tool_execution_timeout_s: float | None = None

    # Framework plumbing. ``metadata`` default is installed in ``__post_init__``
    # so direct construction (unit tests) needs none. Unknown registry kwargs
    # should fail at the registry boundary instead of being silently accepted.
    metadata: LiteBaseMetadata | None = None

    def __post_init__(self) -> None:
        """Install empty metadata default + merge missing api_kwargs from
        ``_API_KWARGS_DEFAULTS``. YAML ``api_kwargs:`` with nothing under it
        loads as ``None``; treat that as an empty dict."""
        if self.metadata is None:
            self.metadata = LiteCUAMetadata()
        if self.api_kwargs is None:
            self.api_kwargs = {}
        for k, v in self._API_KWARGS_DEFAULTS.items():
            self.api_kwargs.setdefault(k, v)
        self.api_kwargs.setdefault("detail", _DEFAULT_IMAGE_DETAIL)
        # A bespoke API loop constructs no adapter, so it does not inherit
        # ``BaseAgentAdapter``'s coercion — same rule, one owner.
        self.resolution = normalize_resolution(self.resolution)

    def _cua_metadata(self) -> LiteCUAMetadata:
        if not isinstance(self.metadata, LiteCUAMetadata):
            raise TypeError(
                f"{type(self).__name__} requires LiteCUAMetadata for CUA tool "
                f"assembly; got {type(self.metadata).__name__}"
            )
        return self.metadata

    def _build_reasoning(self) -> dict[str, Any]:
        """Build the Responses reasoning config from ``api_kwargs``."""
        reasoning: dict[str, Any] = {}
        effort = self.api_kwargs.get("reasoning_effort")
        if effort:
            reasoning["effort"] = effort
        summary = self.api_kwargs.get("reasoning_summary")
        if summary:
            reasoning["summary"] = summary
        return reasoning

    def _detail(self) -> str:
        return self.api_kwargs.get("detail", _DEFAULT_IMAGE_DETAIL)

    def _effective_system_prompt(self) -> str | None:
        # GPTTeacherAgent overrides this to append its instruction. ``strip()``
        # drops a YAML ``|``-block trailing newline before sending.
        return (self.system_prompt or "").strip() or None

    def _extra_tool_names(self) -> frozenset[str]:
        """Names of env-supplied extra tools (``metadata.extra_tool_schemas``).
        The parser passes function_calls with these names through verbatim; every
        other call is an agent-declared action routed through the action space."""
        return frozenset(tool_schema_name(et) for et in self.metadata.extra_tool_schemas)

    def _declared_agent_tool_names(self) -> frozenset[str]:
        """Names of agent-declared function tools outside env extras.

        These route through the action space after native GUI calls and env
        extras.
        """
        return (
            frozenset(
                str(name)
                for tool in self._build_tools()
                if tool.get("type") == "function" and (name := tool.get("name"))
            )
            - self._extra_tool_names()
        )

    def _active_provider_tool_names(self, tools: list[dict[str, Any]]) -> frozenset[str]:
        """Top-level provider tool names declared in this request, minus env extras.

        Covers both GPT surfaces: provider-flat function tools (mobile,
        grounding) and the native ``computer`` tool (desktop).
        """
        names = {
            str(name)
            for tool in tools
            if tool.get("type") == "function" and (name := tool.get("name"))
        }
        if any(tool.get("type") == "computer" for tool in tools):
            names.add("computer")
        return frozenset(names) - self._extra_tool_names()

    def _responses_function_tool_is_strict(
        self,
        parameters: dict[str, Any],
    ) -> bool:
        """Whether one projected Responses function tool can opt into strict mode."""
        if not self.function_tool_strict:
            return False
        properties = parameters.get("properties")
        required = parameters.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            return False
        required_names = {name for name in required if isinstance(name, str)}
        return set(properties) <= required_names and parameters.get("additionalProperties") is False

    def _to_responses_function_tool(
        self,
        schema: dict[str, Any],
        *,
        close_parameters_for_strict: bool = False,
    ) -> dict[str, Any]:
        """Project a canonical Lite schema to OpenAI's Responses function-tool shape."""
        func = schema["function"]
        parameters = dict(func["parameters"])
        if (
            close_parameters_for_strict
            and self.function_tool_strict
            and isinstance(parameters.get("properties"), dict)
        ):
            parameters["additionalProperties"] = False
        tool: dict[str, Any] = {
            "type": "function",
            "name": func["name"],
            "description": func.get("description", ""),
            "parameters": parameters,
        }
        if self._responses_function_tool_is_strict(parameters):
            tool["strict"] = True
        return tool

    async def _finalize_sample(self, *, cache_stats, hooks, result, completed, env) -> None:
        """Shared ``sample()`` teardown (desktop + mobile): log the trajectory
        usage summary, fire ``on_complete`` hooks, and close the env. Runs in the
        ``finally`` of both loops."""
        _log_trajectory_usage_summary(cache_stats)
        await finalize_lite_sample(
            hooks=hooks,
            result=result,
            completed=completed,
            env=env,
            logger=logger,
        )

    async def _aresponses_with_retry(self, **api_kwargs: Any) -> Any:
        """Capped exp-backoff retry around ``litellm.aresponses`` (the shared
        ``acompletion_with_retry`` owns the backoff formula). Retry budget +
        log label come from this instance's fields / ``_RETRY_LOG_NAME``."""
        import litellm  # lazy — kept off the module-import path (see top-of-file note)

        extra = {} if self._RETRY_LOG_NAME is None else {"log_name": self._RETRY_LOG_NAME}
        return await acompletion_with_retry(
            litellm.aresponses,
            max_retries=int(self.api_retry_max),
            base_delay=float(self.api_retry_base_delay),
            max_delay=float(self.api_retry_max_delay),
            **extra,
            **api_kwargs,
        )

    def _build_request_kwargs(
        self,
        *,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        previous_response_id: str | None,
        instructions: str | None = None,
        prompt_cache_key: str | None = None,
    ) -> dict[str, Any]:
        """Assemble the kwargs dict for ``litellm.aresponses``.

        ``instructions`` is only populated when the caller chooses top-level
        system prompting. ``prompt_cache_key`` is sent through ``extra_body`` so
        chained turns stay cache-friendly without changing the litellm surface.
        """
        api_kwargs: dict[str, Any] = {
            "model": self.model_id,
            "input": input_items,
            "tools": tools if tools else None,
            "parallel_tool_calls": self.api_kwargs.get("parallel_tool_calls", False),
            "stream": False,
            "reasoning": self._build_reasoning(),
        }
        if "max_output_tokens" in self.api_kwargs:
            api_kwargs["max_output_tokens"] = self.api_kwargs["max_output_tokens"]
        if instructions:
            api_kwargs["instructions"] = instructions
        truncation = self.api_kwargs.get("truncation", "auto")
        if truncation is not None:
            api_kwargs["truncation"] = truncation
        if "temperature" in self.api_kwargs:
            api_kwargs["temperature"] = self.api_kwargs["temperature"]
        if "tool_choice" in self.api_kwargs:
            # `"required"` forces a tool call every turn (browsergym text-mode
            # needs the answer to flow through send_msg_to_user for WA's grader).
            api_kwargs["tool_choice"] = self.api_kwargs["tool_choice"]
        if previous_response_id:
            api_kwargs["previous_response_id"] = previous_response_id
        if prompt_cache_key:
            api_kwargs["extra_body"] = {"prompt_cache_key": prompt_cache_key}
        if self.api_key:
            api_kwargs["api_key"] = self.api_key
        if self.api_base:
            api_kwargs["api_base"] = self.api_base
        return api_kwargs


@dataclass
class GPTDesktopUseAgent(_GPTBaseAgent, key=r"gpt@(desktop|browser)@use"):
    """Self-contained GPT computer-use agent.

    Manages the OpenAI Responses API loop directly via liteLLM.
    No adapter, no processor, no generate_fn.

    Args:
        model_id: OpenAI model ID (e.g. "gpt-5.5").
        action_space: Action space for CUA-lite <-> GPT conversion.
        metadata: Env-supplied :class:`LiteBaseMetadata`. Reads of
            ``self.metadata.extra_tool_schemas`` / ``valid_actions`` / ``others.<key>``
            happen in the request-building paths below. ``__post_init__``
            installs an empty default ``LiteCUAMetadata()`` so direct construction
            (e.g. unit tests) doesn't require passing one.
        api_kwargs: ``{"max_output_tokens": int, "temperature": float, "reasoning_effort": str}``.
        api_key: OpenAI API key (uses OPENAI_API_KEY env var if None).
        api_base: Optional API base URL override.
        system_prompt: Optional system prompt override.
    """

    # Desktop action space.
    action_space: GPTDesktopActionSpace = field(default_factory=GPTDesktopActionSpace)
    # Native-computer-use history compaction (chain=False): keep the last N
    # call/output steps in full and summarize earlier steps. None disables it.
    # Desktop-only; mobile uses message-based history with no call/output pairing.
    full_history_size: int | None = None

    def _build_tools(self) -> list[dict[str, Any]]:
        """Build the tools list: native computer tool + env extra tools.

        ``valid_actions`` filters the native computer tool; env-declared
        function tools are unwrapped to the Responses function-tool shape.
        """
        metadata = self._cua_metadata()
        provider_tool_declarations: list[dict[str, Any]] = [{"type": "computer"}]
        if metadata.valid_actions is not None:
            provider_tool_declarations = type(
                self.action_space
            ).filter_tool_schemas_for_valid_actions(
                provider_tool_declarations,
                metadata.valid_actions,
            )
        provider_tool_declarations.extend(self._extra_function_tools())
        return provider_tool_declarations

    def _extra_function_tools(self) -> list[dict[str, Any]]:
        """Convert ``self.metadata.extra_tool_schemas`` (env-supplied function tools,
        e.g. ``report_infeasible`` from osworld_g) to OpenAI's Responses-API
        function-tool schema. Shared by use + grounding subclasses."""
        return [self._to_responses_function_tool(et) for et in self.metadata.extra_tool_schemas]

    # -- predict ---------------------------------------------------------------

    async def predict(self, sample: LiteSample) -> LiteMessage:
        # This direct API family drives rollouts through ``sample()``; single-step
        # ``predict()`` is unsupported, matching the mobile sibling.
        raise NotImplementedError(
            "GPTDesktopUseAgent.predict is not supported; use sample() for rollouts"
        )

    def _parse_output_items(
        self,
        output_items: list[dict[str, Any]],
        *,
        resolution: tuple[int, int],
        active_provider_tool_names: frozenset[str] | None = None,
        call_id_start: int = 0,
    ) -> GPTParsedOutput:
        """Parse Responses-API output items into the Lite message + provenance.

        ``GPTTeacherAgent`` overrides this parser to relabel controlled
        Thought/Action prose.

        ``resolution`` is the frame the model's coords are in: the fetched
        processed dims for desktop nav, or the sent frame for grounding.
        """
        return parse_output_items_with_provenance(
            output_items,
            action_space=self.action_space,
            resolution=resolution,
            extra_tool_names=self._extra_tool_names(),
            declared_agent_tool_names=self._declared_agent_tool_names(),
            active_provider_tool_names=active_provider_tool_names,
            call_id_start=call_id_start,
        )

    # -- sample ----------------------------------------------------------------

    async def sample(
        self,
        env: LiteBaseEnv,
        max_steps: int = 10_000,
        hooks: list[SampleHook] | None = None,
    ) -> LiteRLSample:
        """Run a full sampling loop on the environment.

        Each turn observes the env, calls the Responses API, converts provider
        output to Lite tool calls, executes them, then appends provider-shaped
        feedback for the next turn.
        """
        hooks = hooks or []
        result: LiteRLSample | None = None
        completed = False

        # Responses API conversation state.
        input_items: list[dict[str, Any]] = []
        previous_response_id: str | None = None
        instructions = self._effective_system_prompt()
        # Default: place the system prompt in the first input as a persistent
        # developer message. Set ``system_prompt_in_input: false`` to send it
        # through the top-level ``instructions`` field instead.
        system_in_input = bool(self.api_kwargs.get("system_prompt_in_input", True))
        cache_stats = _CacheStats()
        # Stable per-trajectory cache key for chained Responses turns.
        prompt_cache_key = f"cua-lite-{uuid.uuid4().hex}"

        try:
            # Initial env state.
            observation = await env.reset()
            image_bytes = observation.image
            instruction = observation.text
            if not instruction:
                raise RuntimeError("No instruction returned on env.reset()")

            trajectory = LiteSample(metadata=env.metadata)
            steps: list[LiteRLStep] = []
            episode_return = 0.0
            terminated = False
            truncated = False
            next_call_id = 0
            # Last frame sent to the API, with the ``trajectory.images`` index it
            # carries. Desktop provider feedback reuses the pair when the env
            # returns text feedback without a new image, so both must stay set
            # together — an unmarked resend would drop out of
            # ``LiteRLStep.image_indices``.
            sent_image_b64: str | None = None
            sent_image_index: int | None = None
            sent_w = sent_h = 0
            provider_visible_image_indices: list[int] = []
            observation_image_index: int | None = None

            for step in range(max_steps):
                # 1. Observe: decode/resize the current env frame for the API.
                if image_bytes:
                    if observation_image_index is None:
                        image = await asyncio.to_thread(decode_image, image_bytes)
                        trajectory.images.append(image)
                        sent_image_index = len(trajectory.images) - 1
                    else:
                        image = trajectory.images[observation_image_index]
                        sent_image_index = observation_image_index

                    sent_image_b64, sent_w, sent_h = await asyncio.to_thread(
                        _resize_for_api,
                        image_bytes,
                        self.resolution,
                    )
                else:
                    # No new frame: the previous ``sent_image_b64``/
                    # ``sent_image_index`` pair is still what the model sees.
                    image = None
                    if sent_image_b64 is None:
                        # Text-only obs mode: no native GUI coords should appear,
                        # but downstream normalization still needs nonzero dims.
                        sent_w, sent_h = self.resolution or (1, 1)

                tools = self._build_tools()
                active_provider_tool_names = self._active_provider_tool_names(tools)

                # First turn: seed provider and Lite histories with task input.
                if step == 0:
                    if instructions and "{w}" in instructions:
                        instructions = instructions.replace("{w}", str(sent_w)).replace(
                            "{h}", str(sent_h)
                        )
                    if system_in_input and instructions:
                        input_items.append({"role": "developer", "content": instructions})
                    user_content: list[dict[str, Any]] = [
                        {"type": "input_text", "text": instruction},
                    ]
                    traj_content: list[dict[str, Any]] = [
                        {"type": "text", "text": instruction},
                    ]
                    if sent_image_b64 is not None:
                        user_content.append(
                            mark_provider_visible_image_index(
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/png;base64,{sent_image_b64}",
                                    "detail": self._detail(),
                                },
                                sent_image_index,
                            )
                        )
                        if sent_image_index is not None:
                            traj_content.insert(0, {"type": "image", "index": sent_image_index})
                            provider_visible_image_indices.append(sent_image_index)
                    input_items.append({"role": "user", "content": user_content})
                    trajectory.messages.append({"role": "user", "content": traj_content})

                # 2. Request: compact client-managed history, then call the model.
                request_input = _compress_computer_use_history(input_items, self.full_history_size)
                use_chaining = self.api_kwargs.get("chain_previous_response", True)
                request_provider_visible_image_indices = (
                    _provider_visible_image_indices_from_gpt_input_items(request_input)
                )
                api_input = _strip_gpt_image_markers(request_input)
                if system_in_input:
                    turn_instructions = None
                else:
                    turn_instructions = (
                        instructions
                        if (step == 0 or not (use_chaining and previous_response_id))
                        else None
                    )
                api_kwargs = self._build_request_kwargs(
                    input_items=api_input,
                    tools=tools,
                    previous_response_id=previous_response_id if use_chaining else None,
                    instructions=turn_instructions,
                    prompt_cache_key=prompt_cache_key,
                )

                (response, actual_w, actual_h), predict_s = await timed(
                    _call_api_with_actual_dim(
                        self._aresponses_with_retry,
                        api_kwargs,
                        sent_w=sent_w,
                        sent_h=sent_h,
                        model_id=self.model_id,
                        api_base=self.api_base,
                        api_key=self.api_key,
                    )
                )
                if self._normalize_by_actual_dims:
                    norm_w, norm_h = actual_w, actual_h
                else:
                    if (actual_w, actual_h) != (sent_w, sent_h):
                        raise RuntimeError(
                            f"GPT API auto-downsampled the fixed-frame screenshot "
                            f"{sent_w}x{sent_h} → {actual_w}x{actual_h}; the system prompt "
                            f"declares ABSOLUTE pixels on the {sent_w}x{sent_h} frame, so "
                            f"coordinates would be wrong. Lower the sent resolution to fit "
                            f"the API's image budget."
                        )
                    norm_w, norm_h = sent_w, sent_h
                normalized_response = _normalized_gpt_response(response)
                _log_usage_and_check_cache(
                    normalized_response,
                    step,
                    chained=use_chaining,
                    stats=cache_stats,
                )

                _raise_if_response_failed(normalized_response)

                previous_response_id = normalized_response.response_id
                output_items = normalized_response.output_items

                # 3. Parse: provider output -> canonical Lite assistant message
                # plus the provider->canonical provenance the feedback step needs.
                parsed_output = self._parse_output_items(
                    output_items,
                    resolution=(norm_w, norm_h),
                    active_provider_tool_names=active_provider_tool_names,
                    call_id_start=next_call_id,
                )
                lite_message = parsed_output.message
                model_output_error = pop_model_output_error(lite_message)
                next_call_id += len(lite_message.get("tool_calls", []))

                prompt_for_log = json.dumps(
                    _strip_images(api_input),
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
                response_for_log = json.dumps(
                    output_items,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )

                trajectory.messages.append(lite_message)

                rl_step = append_lite_rl_step(
                    steps=steps,
                    prompt=prompt_for_log,
                    image_indices=(
                        tuple(provider_visible_image_indices)
                        if use_chaining
                        else request_provider_visible_image_indices
                    ),
                    response=response_for_log,
                    finish_reason=normalized_response.finish_reason,
                    truncation_finish_reasons=_TRUNCATION_INCOMPLETE_REASONS,
                )

                # with chaining on, API remembers context — clear and start
                # fresh. With chaining off, keep accumulating (truncation
                # handles bounds).
                if use_chaining:
                    input_items = []

                # 4. Act: execute Lite tool calls and update RL accounting. A
                # zero-tool-call final still steps the env, so hook actions and
                # persisted assistant calls diverge on that path.
                turn = await execute_lite_turn(
                    env=env,
                    trajectory=trajectory,
                    lite_message=lite_message,
                    model_output_error=model_output_error,
                    rl_step=rl_step,
                    tool_execution_timeout_s=self.tool_execution_timeout_s,
                )
                lite_message = turn.message
                step_result = turn.step_result

                episode_return, terminated, truncated = await record_lite_env_result(
                    trajectory=trajectory,
                    persisted_actions=turn.persisted_actions,
                    step_result=step_result,
                    rl_step=rl_step,
                    step=step,
                    max_steps=max_steps,
                    episode_return=episode_return,
                    append_terminal_tool_feedback=_append_gpt_terminal_tool_feedback,
                )

                await dispatch_lite_step_hooks(
                    hooks=hooks,
                    step_idx=step,
                    image=image,
                    lite_message=lite_message,
                    rl_step=rl_step,
                    step_result=step_result,
                    hook_actions=turn.hook_actions,
                    timings={"predict": predict_s, "act": turn.act_s},
                )

                if terminated or truncated:
                    break

                # 5. Feedback: append Responses-shaped outputs for next turn.
                (
                    image_bytes,
                    step_provider_visible_image_indices,
                ) = await _append_desktop_provider_feedback(
                    input_items=input_items,
                    output_items=output_items,
                    parsed_provider_calls=parsed_output.provider_calls,
                    trajectory=trajectory,
                    step_result=step_result,
                    lite_tool_calls=lite_message.get("tool_calls", []),
                    resolution=self.resolution,
                    sent_image_b64=sent_image_b64,
                    sent_image_index=sent_image_index,
                    detail=self._detail(),
                    use_chaining=use_chaining,
                    model_output_error=model_output_error,
                )
                observation_image_index = (
                    step_provider_visible_image_indices[-1]
                    if image_bytes and step_provider_visible_image_indices
                    else None
                )
                provider_visible_image_indices.extend(step_provider_visible_image_indices)
            else:
                truncated = True
                mark_steps_truncated(steps)

            result = build_lite_rl_sample(
                trajectory=trajectory,
                steps=steps,
                episode_return=episode_return,
                terminated=terminated,
                truncated=truncated,
            )
            completed = True
            return result
        finally:
            await self._finalize_sample(
                cache_stats=cache_stats,
                hooks=hooks,
                result=result,
                completed=completed,
                env=env,
            )


# =============================================================================
# GPT Desktop Grounding (Point) Agent
# =============================================================================
#
# Single-step click prediction for grounding evals. This path uses a
# click-shaped function tool instead of the native computer tool, so the model
# can only return the point requested by the task.

GPT_GROUNDING_SYSTEM_PROMPT = (
    "You are a GUI grounding agent. Given a screenshot and an instruction, locate the "
    "UI element the instruction refers to and click its center with one click action.\n\n"
    "Coordinates are ABSOLUTE pixels on the {w}×{h} screenshot, with (0,0) at the top-left."
)

GPT_GROUNDING_API_KWARGS_DEFAULTS: dict[str, Any] = {
    **GPT_API_KWARGS_DEFAULTS,
    "max_output_tokens": 4096,
    # gpt-5.5 reasoning levels, verified against the API:
    # none / minimal / low / medium / high / xhigh / max. Use ``none`` for
    # grounding — one click, no reasoning wanted.
    "reasoning_effort": "none",
    "parallel_tool_calls": False,
}


@dataclass
class GPTDesktopGroundingPointAgent(GPTDesktopUseAgent, key=r"gpt@(desktop|browser)@grounding\.point"):
    """GPT desktop grounding (single-step click prediction).

    Uses a focused grounding prompt, a point action space, absolute-pixel
    parsing on the sent frame, and grounding-specific api defaults.
    """

    # Grounding-specific api_kwargs (reasoning_effort=none, max_output_tokens=4096)
    # drive the inherited __post_init__ via the class-attr override.
    _API_KWARGS_DEFAULTS: ClassVar[dict[str, Any]] = GPT_GROUNDING_API_KWARGS_DEFAULTS
    # Grounding declares absolute pixels on the sent {w}×{h} frame, so parse
    # coordinates against the sent frame rather than provider-processed dims.
    _normalize_by_actual_dims: ClassVar[bool] = False
    action_space: GPTDesktopGroundingPointActionSpace = field(
        default_factory=GPTDesktopGroundingPointActionSpace
    )
    system_prompt: str | None = GPT_GROUNDING_SYSTEM_PROMPT

    def _build_tools(self) -> list[dict[str, Any]]:
        """Expose click-shaped function tools plus env extra tools."""
        metadata = self._cua_metadata()
        lite_action_schemas = type(self.action_space).get_tool_schemas()
        if metadata.valid_actions is not None:
            lite_action_schemas = type(self.action_space).filter_tool_schemas_for_valid_actions(
                lite_action_schemas,
                metadata.valid_actions,
            )
        provider_tool_declarations = [
            self._to_responses_function_tool(schema) for schema in lite_action_schemas
        ]
        return [*provider_tool_declarations, *self._extra_function_tools()]


# =============================================================================
# GPT Mobile Agent (Android)
# =============================================================================

# Mobile base system prompt. Tool descriptions own the individual tap/swipe
# mechanics; the system prompt owns the absolute-pixel coordinate frame. Finish
# guidance is appended only when the env declares response/terminate tools.
_MOBILE_SYSTEM_PROMPT = (
    "You are a GUI agent operating a mobile phone via screenshots. At each step "
    "you see a screenshot and call one tool to act on the screen.\n\n"
    "Coordinates are ABSOLUTE pixels on the {w}×{h} screenshot, with (0,0) at the "
    "top-left."
)


GPT_MOBILE_API_KWARGS_DEFAULTS: dict[str, Any] = {
    # Same provider knobs as desktop; mobile uses provider-flat function
    # schemas rather than the native computer tool.
    "reasoning_effort": "medium",
    # Reasoning summary → inline_reasoning (see desktop defaults).
    "reasoning_summary": "concise",
    # detail: set dynamically in __post_init__
    "chain_previous_response": True,
    "parallel_tool_calls": False,
}


@dataclass
class GPTMobileUseAgent(_GPTBaseAgent, key="gpt@mobile@use"):
    """GPT mobile GUI-use agent for AndroidWorld.

    Sibling to ``GPTDesktopUseAgent``.
    GPT's mobile path has no native computer tool (GPT Responses API's `environment`
    enum only covers linux/mac/windows/browser/ubuntu — not Android), so we
    send pure function tools describing tap/swipe/etc. semantics.

    Mobile-specific pieces are the action space, strict-tool eligibility,
    shorter retry budget, and coordinate-frame prompt.
    """

    _API_KWARGS_DEFAULTS: ClassVar[dict[str, Any]] = GPT_MOBILE_API_KWARGS_DEFAULTS
    _RETRY_LOG_NAME: ClassVar[str | None] = "litellm.aresponses (mobile)"

    action_space: GPTMobileActionSpace = field(default_factory=GPTMobileActionSpace)
    # The mobile tools are locally defined provider-flat function schemas.
    # Some have optional fields (e.g. long_press.duration), so the shared
    # Responses projection only marks schemas strict when every property is required.
    function_tool_strict: bool = True
    # Base prompt with ``{w}``/``{h}`` fields resolved per step-0 screenshot.
    # ``{w}``/``{h}`` are replaced with the sent screenshot dims at step 0.
    system_prompt: str | None = _MOBILE_SYSTEM_PROMPT
    # 12 retries ≈ 4–14 min total wait — shorter than desktop's 24; long enough
    # to ride out multi-minute Azure quota spikes without losing partial work.
    api_retry_max: int = 12

    def _system_prompt_for_mobile_request(self, sent_w: int, sent_h: int) -> str | None:
        prompt = self.system_prompt
        if prompt:
            prompt = prompt.replace("{w}", str(sent_w)).replace("{h}", str(sent_h))
        guidance = mobile_finish_guidance(self._extra_tool_names())
        if not guidance:
            return prompt
        if not prompt:
            return guidance
        return f"{prompt}\n\n{guidance}"

    def _build_tools(self) -> list[dict[str, Any]]:
        """Mobile tools: provider-flat `tap`/`swipe`/etc. functions. No computer tool."""
        provider_tool_declarations: list[dict[str, Any]] = []
        # GUI tools: ``valid_actions=None`` exposes the provider-flat set;
        # a list filters them; ``[]`` leaves only env extra tools.
        metadata = self._cua_metadata()
        lite_action_schemas = self.action_space.get_tool_schemas()
        if metadata.valid_actions is not None:
            lite_action_schemas = type(self.action_space).filter_tool_schemas_for_valid_actions(
                lite_action_schemas, metadata.valid_actions
            )
        # action_space schemas are canonical Lite declarations; project them to
        # GPT Responses' provider-flat function-tool shape.
        for fn in lite_action_schemas:
            provider_tool_declarations.append(
                self._to_responses_function_tool(fn, close_parameters_for_strict=True)
            )

        for et in self.metadata.extra_tool_schemas:
            provider_tool_declarations.append(self._to_responses_function_tool(et))

        return provider_tool_declarations

    async def sample(
        self,
        env: LiteBaseEnv,
        max_steps: int = 10_000,
        hooks: list[SampleHook] | None = None,
    ) -> LiteRLSample:
        """Run the mobile Responses-API sampling loop."""
        hooks = hooks or []
        result: LiteRLSample | None = None
        completed = False

        input_items: list[dict[str, Any]] = []
        previous_response_id: str | None = None
        cache_stats = _CacheStats()
        # Stable per-trajectory cache key for chained Responses turns.
        prompt_cache_key = f"cua-lite-{uuid.uuid4().hex}"

        try:
            # Initial env state.
            observation = await env.reset()
            image_bytes = observation.image
            if not image_bytes:
                raise RuntimeError("No image returned on env.reset()")

            instruction = observation.text
            if not instruction:
                raise RuntimeError("No instruction returned on env.reset()")

            trajectory = LiteSample(metadata=env.metadata)
            steps: list[LiteRLStep] = []
            episode_return = 0.0
            terminated = False
            truncated = False
            next_call_id = 0
            sys_prompt: str | None = None
            provider_visible_image_indices: list[int] = []
            observation_image_index: int | None = None
            # Same default as desktop: persist the system prompt as a developer
            # input item unless config opts back into top-level instructions.
            system_in_input = bool(self.api_kwargs.get("system_prompt_in_input", True))

            for step in range(max_steps):
                # 1. Observe: mobile always requires an image observation.
                if observation_image_index is None:
                    image = await asyncio.to_thread(decode_image, image_bytes)
                    trajectory.images.append(image)
                    sent_image_index = len(trajectory.images) - 1
                else:
                    image = trajectory.images[observation_image_index]
                    sent_image_index = observation_image_index

                sent_image_b64, sent_w, sent_h = await asyncio.to_thread(
                    _resize_for_api,
                    image_bytes,
                    self.resolution,
                )

                tools = self._build_tools()
                active_provider_tool_names = self._active_provider_tool_names(tools)

                # First turn: format the prompt once sent dimensions are known.
                if step == 0:
                    sys_prompt = self._system_prompt_for_mobile_request(sent_w, sent_h)
                    if system_in_input and sys_prompt:
                        input_items.append({"role": "developer", "content": sys_prompt})
                    input_items.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": instruction},
                                mark_provider_visible_image_index(
                                    {
                                        "type": "input_image",
                                        "image_url": f"data:image/png;base64,{sent_image_b64}",
                                        "detail": self._detail(),
                                    },
                                    sent_image_index,
                                ),
                            ],
                        }
                    )
                    trajectory.messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "index": sent_image_index},
                                {"type": "text", "text": instruction},
                            ],
                        }
                    )
                    provider_visible_image_indices.append(sent_image_index)

                # 2. Request: call the model through Responses API.
                use_chaining = self.api_kwargs.get("chain_previous_response", True)
                request_input = input_items
                request_provider_visible_image_indices = (
                    _provider_visible_image_indices_from_gpt_input_items(request_input)
                )
                api_input = _strip_gpt_image_markers(request_input)
                if system_in_input:
                    turn_instructions = None
                else:
                    turn_instructions = (
                        sys_prompt
                        if (step == 0 or not (use_chaining and previous_response_id))
                        else None
                    )
                api_kwargs = self._build_request_kwargs(
                    input_items=api_input,
                    tools=tools,
                    previous_response_id=previous_response_id if use_chaining else None,
                    instructions=turn_instructions,
                    prompt_cache_key=prompt_cache_key,
                )

                # Mobile prompts declare absolute pixels on the sent frame; fail
                # fast if the API reports that it processed a different size.
                (response, actual_w, actual_h), predict_s = await timed(
                    _call_api_with_actual_dim(
                        self._aresponses_with_retry,
                        api_kwargs,
                        sent_w=sent_w,
                        sent_h=sent_h,
                        model_id=self.model_id,
                        api_base=self.api_base,
                        api_key=self.api_key,
                    )
                )  # predict_s = LLM-call wall time (incl. retries)
                if (actual_w, actual_h) != (sent_w, sent_h):
                    raise RuntimeError(
                        f"GPT API auto-downsampled the mobile screenshot {sent_w}x{sent_h} → "
                        f"{actual_w}x{actual_h}; the system prompt declares ABSOLUTE pixels on "
                        f"the {sent_w}x{sent_h} frame, so tap coordinates would be wrong. "
                        f"Lower the sent resolution to fit the API's image budget."
                    )
                normalized_response = _normalized_gpt_response(response)
                _log_usage_and_check_cache(
                    normalized_response,
                    step,
                    chained=use_chaining,
                    stats=cache_stats,
                )

                _raise_if_response_failed(normalized_response)

                previous_response_id = normalized_response.response_id
                output_items = normalized_response.output_items

                # 3. Parse: provider output -> canonical Lite assistant message.
                parsed_output = parse_gpt_mobile_output_items_with_provenance(
                    output_items,
                    action_space=self.action_space,
                    resolution=(sent_w, sent_h),
                    extra_tool_names=self._extra_tool_names(),
                    declared_agent_tool_names=self._declared_agent_tool_names(),
                    active_provider_tool_names=active_provider_tool_names,
                    call_id_start=next_call_id,
                )
                lite_message = parsed_output.message
                model_output_error = pop_model_output_error(lite_message)
                next_call_id += len(lite_message.get("tool_calls", []))

                prompt_for_log = json.dumps(
                    _strip_images(api_input),
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
                response_for_log = json.dumps(
                    output_items,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )

                trajectory.messages.append(lite_message)

                rl_step = append_lite_rl_step(
                    steps=steps,
                    prompt=prompt_for_log,
                    image_indices=(
                        tuple(provider_visible_image_indices)
                        if use_chaining
                        else request_provider_visible_image_indices
                    ),
                    response=response_for_log,
                    finish_reason=normalized_response.finish_reason,
                    truncation_finish_reasons=_TRUNCATION_INCOMPLETE_REASONS,
                )

                if use_chaining:
                    input_items = []

                # 4. Act: execute Lite tool calls and update RL accounting. A
                # zero-tool-call final still steps the env, so hook actions and
                # persisted assistant calls diverge on that path.
                turn = await execute_lite_turn(
                    env=env,
                    trajectory=trajectory,
                    lite_message=lite_message,
                    model_output_error=model_output_error,
                    rl_step=rl_step,
                    tool_execution_timeout_s=self.tool_execution_timeout_s,
                )
                lite_message = turn.message
                step_result = turn.step_result

                episode_return, terminated, truncated = await record_lite_env_result(
                    trajectory=trajectory,
                    persisted_actions=turn.persisted_actions,
                    step_result=step_result,
                    rl_step=rl_step,
                    step=step,
                    max_steps=max_steps,
                    episode_return=episode_return,
                    append_terminal_tool_feedback=_append_gpt_terminal_tool_feedback,
                )

                await dispatch_lite_step_hooks(
                    hooks=hooks,
                    step_idx=step,
                    image=image,
                    lite_message=lite_message,
                    rl_step=rl_step,
                    step_result=step_result,
                    hook_actions=turn.hook_actions,
                    timings={"predict": predict_s, "act": turn.act_s},
                )

                if terminated or truncated:
                    break

                # 5. Feedback: append Responses-shaped outputs for next turn.
                (
                    image_bytes,
                    step_provider_visible_image_indices,
                ) = await _append_mobile_provider_feedback(
                    input_items=input_items,
                    output_items=output_items,
                    parsed_provider_calls=parsed_output.provider_calls,
                    trajectory=trajectory,
                    step_result=step_result,
                    lite_tool_calls=lite_message.get("tool_calls", []),
                    resolution=self.resolution,
                    detail=self._detail(),
                    use_chaining=use_chaining,
                )
                observation_image_index = (
                    step_provider_visible_image_indices[-1]
                    if step_provider_visible_image_indices
                    else None
                )
                provider_visible_image_indices.extend(step_provider_visible_image_indices)
            else:
                truncated = True
                mark_steps_truncated(steps)

            result = build_lite_rl_sample(
                trajectory=trajectory,
                steps=steps,
                episode_return=episode_return,
                terminated=terminated,
                truncated=truncated,
            )
            completed = True
            return result
        finally:
            await self._finalize_sample(
                cache_stats=cache_stats,
                hooks=hooks,
                result=result,
                completed=completed,
                env=env,
            )

    async def predict(self, sample: LiteSample) -> LiteMessage:
        raise NotImplementedError(
            "GPTMobileUseAgent.predict is not supported; use sample() for rollouts"
        )
