"""Claude computer-use agent for CUA-Lite.

Self-contained agent that talks directly to Anthropic's computer-use API via
liteLLM.  Inherits from ``BaseAgent`` — no adapter, no processor, no
generate_fn.  This file owns the model-family defaults, tool-surface builders,
and desktop/mobile sample loops; provider image sizing, history shaping, and
response parsing live in ``claude.utils``.

Usage:
    from lite.agents.models.claude.agent import ClaudeDesktopUseAgent

    agent = ClaudeDesktopUseAgent(model_id="claude-opus-4-6")
    result = await agent.sample(env, max_steps=20)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar

# ``litellm`` is imported lazily inside API-call methods so registry/env-server
# startup stays side-effect-free.
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
from lite.agents.models.claude.action_space import (
    ClaudeDesktopActionSpace,
    ClaudeDesktopGroundingPointActionSpace,
    ClaudeMobileActionSpace,
)
from lite.agents.models.claude.utils.history import (
    append_desktop_provider_feedback,
    append_mobile_provider_feedback,
    append_provider_assistant_message,
    filter_to_n_most_recent_images,
    inject_prompt_caching,
    strip_images,
)
from lite.agents.models.claude.utils.image_io import resize_for_claude_api
from lite.agents.models.claude.utils.loop import (
    _append_claude_terminal_tool_feedback,
)
from lite.agents.models.claude.utils.parse import (
    _claude_active_provider_tool_names,
    parse_mobile_response_with_provenance,
    parse_response_with_provenance,
)
from lite.agents.models.claude.utils.toolset import acompletion_with_computer_toolset
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

# =============================================================================
# Constants
# =============================================================================

# Model version -> computer tool version + beta flag
MODEL_TOOL_MAPPING = [
    {
        "pattern": r"(?:^|/)claude-opus-5$",
        "tool_version": "computer_toolset_20260801",
        "beta_flag": "",
    },
    {
        "pattern": r"(?:^|/)claude-(?:opus-4-(?:5|6|7|8)|sonnet-4-6)$",
        "tool_version": "computer_20251124",
        "beta_flag": "computer-use-2025-11-24",
    },
    {
        "pattern": r"(?:^|/)claude-(?:4|opus-4|sonnet-4|haiku-4)$",
        "tool_version": "computer_20250124",
        "beta_flag": "computer-use-2025-01-24",
    },
    {
        "pattern": r"(?:^|/)claude-3(?:\.|-)?7$",
        "tool_version": "computer_20250124",
        "beta_flag": "computer-use-2025-01-24",
    },
    {
        "pattern": r"(?:^|/)claude-3(?:\.|-)?5$",
        "tool_version": "computer_20241022",
        "beta_flag": "computer-use-2024-10-22",
    },
]

# No default system prompt — Claude API auto-generates a computer use prompt
# when Anthropic-schema tools are used. Override via system_prompt kwarg or config.

# =============================================================================
# Helpers
# =============================================================================


def _get_tool_config_for_model(
    model: str,
    *,
    tool_version: str | None = None,
    beta_flag: str | None = None,
) -> dict[str, str]:
    """Get the declared computer-use tool version and beta flag for a model."""
    if tool_version is not None or beta_flag is not None:
        if not tool_version or not beta_flag:
            raise ValueError(
                "Claude computer-use override requires both "
                "api_kwargs.computer_tool_version and api_kwargs.computer_use_beta_flag"
            )
        return {"tool_version": tool_version, "beta_flag": beta_flag}

    for mapping in MODEL_TOOL_MAPPING:
        if re.search(mapping["pattern"], model, re.IGNORECASE):
            return {"tool_version": mapping["tool_version"], "beta_flag": mapping["beta_flag"]}
    raise ValueError(
        f"Unsupported Claude model id for computer-use tool mapping: {model!r}. "
        "Add a MODEL_TOOL_MAPPING entry or pass both "
        "api_kwargs.computer_tool_version and api_kwargs.computer_use_beta_flag."
    )


def _model_rejects_temperature(model: str) -> bool:
    """Return whether the model rejects an explicit ``temperature`` param."""
    return bool(re.search(r"(?:^|/)claude-opus-(?:4-(?:7|8)|5)$", model, re.IGNORECASE))


def _extra_tool_names(metadata: LiteBaseMetadata) -> frozenset[str]:
    """Names of env-supplied extra tools (``metadata.extra_tool_schemas``). The
    parser passes function-calls with these names through verbatim."""
    return frozenset(tool_schema_name(et) for et in metadata.extra_tool_schemas)


def _provider_visible_image_indices_from_claude_messages(
    messages: list[dict[str, Any]],
) -> tuple[int, ...]:
    """Indices of the images this Claude request actually sends the model.

    Claude carries every screenshot as an ``image_url`` content block, so one
    pass over message content is enough. The values index ``LiteSample.images``.
    """
    image_blocks: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "image_url":
                continue
            image_blocks.append(block)
    return provider_visible_image_indices_from_marked_blocks(image_blocks)


def _strip_claude_image_markers(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip provider-visible image-index markers from Claude ``image_url`` blocks."""
    stripped_messages: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            stripped_messages.append(message)
            continue

        stripped_message = dict(message)
        stripped_message["content"] = [
            strip_provider_visible_image_index_marker(block)
            if isinstance(block, dict) and block.get("type") == "image_url"
            else block
            for block in content
        ]
        stripped_messages.append(stripped_message)
    return stripped_messages


def _step_log_payloads(messages: list[dict[str, Any]], response: Any) -> tuple[str, str]:
    """Serialize the redacted provider prompt and raw response for ``LiteRLStep``."""
    prompt_for_log = json.dumps(
        strip_images(messages),
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    response_for_log = json.dumps(
        response.model_dump(),
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    return prompt_for_log, response_for_log


# Chat-completions stop signals meaning the token budget ran out, not that the
# model deliberately finished. The sample loop records these as truncated steps.
_TRUNCATION_FINISH_REASONS = ("length", "max_tokens", "context_length_exceeded")


# =============================================================================
# Claude Agent
# =============================================================================

CLAUDE_API_KWARGS_DEFAULTS: dict[str, Any] = {
    "max_tokens": 4096,
    "thinking_budget": 2048,
    # Reasoning depth, sent as top-level ``output_config``. The API default is
    # ``high``; ``medium`` is the family-wide setting here so the three API
    # families sit at a comparable depth. Not routed through litellm's
    # ``reasoning_effort``: that maps to the deprecated
    # ``thinking:{type:"enabled"}`` form, which Claude 4.7+ rejects with 400.
    "effort": "medium",
    "prompt_caching": True,
    "cache_breakpoints": 4,
    "token_efficient_tools_beta": False,
}


@dataclass
class _ClaudeBaseAgent(BaseAgent):
    """Shared base for the bespoke Claude API-loop agents (desktop + mobile).

    NOT registered (no ``key=``) — an abstract intermediate that holds the
    fields and provider plumbing both siblings duplicated: the ``api_kwargs``
    merge, system-prompt assembly, the ``litellm.acompletion`` retry wrapper,
    and ``_apply_sampling_extras`` (tool_choice, thinking, image truncation,
    prompt caching). Subclasses supply the parts that genuinely diverge:
    ``action_space`` (type), ``_build_tools``, ``_build_beta_header`` (whose
    signature differs — desktop needs the computer-use tool config, mobile
    does not) and the ``sample`` / ``predict`` loop interiors.

    Mirrors :class:`lite.agents.models.gpt.agent._GPTBaseAgent`. Two class
    attributes drive the shared methods so subclasses customise via a
    one-liner instead of re-implementing:
      * ``_API_KWARGS_DEFAULTS`` — the canonical api_kwargs filled in by
        ``__post_init__`` (desktop vs. mobile vs. grounding defaults).
      * ``_RETRY_LOG_NAME`` — label for the retry wrapper's logs.
    """

    # --- class-attr knobs (NOT dataclass fields) ---
    _API_KWARGS_DEFAULTS: ClassVar[dict[str, Any]] = CLAUDE_API_KWARGS_DEFAULTS
    _RETRY_LOG_NAME: ClassVar[str] = "litellm.acompletion"
    #: Whether a request from this agent can carry SEVERAL images, which the
    #: provider holds to a stricter per-image ceiling (see
    #: ``image_io.effective_max_edge_px``). It is a class fact, not a per-turn
    #: one, because the sent size IS the computer-use coordinate frame: a
    #: ceiling that moved as history grew would shift the frame mid-episode.
    #: ``use`` loops accumulate history and are many-image from turn one;
    #: single-step grounding never is, and keeps the full single-image ceiling.
    _MANY_IMAGE_REQUESTS: ClassVar[bool] = True

    # --- shared dataclass fields (base defaults; a subclass may override any —
    # mobile re-declares ``system_prompt`` and ``resolution``). ---
    model_id: str = "claude-opus-4-6"
    # Starts empty; ``__post_init__`` fills it from ``_API_KWARGS_DEFAULTS``
    # (subclass-set) plus user config via ``setdefault``, so every call site
    # can trust the keys are populated.
    api_kwargs: dict[str, Any] = field(default_factory=dict)
    api_key: str | None = None
    api_base: str | None = None
    system_prompt: str | None = None
    # Agent-level behavior fields (not actual API params).
    # ``None`` means "choose the largest aspect-preserving image size that
    # avoids Anthropic server-side downsampling"; a tuple is an explicit fixed
    # image size override.
    resolution: tuple[int, int] | None = None
    only_n_most_recent_images: int | None = None
    image_truncation_threshold: int = 1
    system_prompt_suffix: str = ""
    api_retry_max: int = 3
    api_retry_base_delay: float = 0.5
    api_retry_max_delay: float = 60.0
    tool_execution_timeout_s: float | None = None
    # Env metadata (forwarded by lite.agents.factory.make).
    metadata: LiteBaseMetadata | None = None

    def __post_init__(self) -> None:
        """Merge missing kwargs from ``_API_KWARGS_DEFAULTS`` + install an
        empty metadata default.

        YAML configs with ``api_kwargs:`` and nothing under it load as None.
        """
        if self.metadata is None:
            self.metadata = LiteCUAMetadata()
        if self.api_kwargs is None:
            self.api_kwargs = {}
        for k, v in self._API_KWARGS_DEFAULTS.items():
            self.api_kwargs.setdefault(k, v)
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

    def _effective_system_prompt(self) -> str | None:
        """combine explicit system_prompt with system_prompt_suffix field."""
        base = self.system_prompt or ""
        suffix = self.system_prompt_suffix or ""
        combined = (base + ("\n" if base and suffix else "") + suffix).strip()
        return combined or None

    def _computer_tool_config(self) -> dict[str, str]:
        """Resolve the declared Anthropic computer-use tool surface."""
        return _get_tool_config_for_model(
            self.model_id,
            tool_version=self.api_kwargs.get("computer_tool_version"),
            beta_flag=self.api_kwargs.get("computer_use_beta_flag"),
        )

    @staticmethod
    def _to_litellm_wrapped_function_tool(schema: dict[str, Any]) -> dict[str, Any]:
        """Project a Lite schema to liteLLM's OpenAI-style tool envelope."""
        func = schema["function"]
        return {
            "type": "function",
            "function": {
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func["parameters"],
            },
        }

    async def _acompletion_with_retry(self, **api_kwargs: Any) -> Any:
        """Call the Claude transport with capped exp backoff + jitter retry.

        Delegates to the shared ``acompletion_with_retry`` (single source of
        truth for the backoff formula). Retries on any Exception up to
        ``api_retry_max`` attempts (total ``max + 1`` calls). Per-attempt sleep:
        ``min(api_retry_max_delay, api_retry_base_delay * 2**attempt)`` scaled by
        U(0.5, 1.5) jitter.
        """
        import litellm  # lazy — kept off the module-import path (see top-of-file note)

        completion = (
            acompletion_with_computer_toolset
            if any(
                tool.get("type") == "computer_toolset_20260801"
                for tool in api_kwargs.get("tools", [])
            )
            else litellm.acompletion
        )
        return await acompletion_with_retry(
            completion,
            max_retries=int(self.api_retry_max),
            base_delay=float(self.api_retry_base_delay),
            max_delay=float(self.api_retry_max_delay),
            log_name=self._RETRY_LOG_NAME,
            **api_kwargs,
        )

    async def _finalize_sample(self, *, hooks, result, completed, env) -> None:
        """Shared ``sample()`` teardown for desktop and mobile Claude loops."""
        await finalize_lite_sample(
            hooks=hooks,
            result=result,
            completed=completed,
            env=env,
            logger=logger,
        )

    def _apply_sampling_extras(self, api_kwargs: dict[str, Any]) -> None:
        """Apply explicit Claude request policy in-place before the API call."""
        # ``tool_choice="required"`` forces some tool call every turn.
        # Used by browsergym text-mode where the answer must flow through
        # ``send_msg_to_user(text=...)`` for WA's grader to find it.
        # AgentLab tool_use_agent does the same. Mirrors GPT's
        # ``api_kwargs.tool_choice`` plumbing.
        #
        # litellm validates ``tool_choice`` against the OpenAI spec before
        # translating to Anthropic's ``{"type": "any"}`` form server-side,
        # so accept the OpenAI-style string form (``"required"`` /
        # ``"auto"`` / ``"none"``) and pass dict forms (e.g.
        # ``{"type":"tool","name":...}``) through unchanged.
        forces_tool = False
        if "tool_choice" in self.api_kwargs:
            tc = self.api_kwargs["tool_choice"]
            api_kwargs["tool_choice"] = tc
            forces_tool = tc in ("required", "any") or (
                isinstance(tc, dict) and tc.get("type") in ("any", "tool")
            )

        # Reasoning depth. Top-level ``output_config`` is the only route that
        # reaches the API: litellm drops it from ``extra_body``, and its own
        # ``reasoning_effort`` param rewrites to the deprecated
        # ``thinking:{type:"enabled"}`` form that 4.7+ rejects.
        effort = self.api_kwargs.get("effort")
        if effort:
            api_kwargs["output_config"] = {"effort": effort}

        thinking_budget = self.api_kwargs.get("thinking_budget")
        if thinking_budget and forces_tool:
            raise ValueError(
                "Claude thinking_budget cannot be used when tool_choice forces tool use. "
                "Set api_kwargs.thinking_budget=0 or remove the forced tool_choice."
            )
        if thinking_budget:
            if _model_rejects_temperature(self.model_id):
                if "temperature" in self.api_kwargs:
                    raise ValueError(
                        f"{self.model_id} rejects explicit temperature. "
                        "Remove api_kwargs.temperature for adaptive Claude thinking."
                    )
                api_kwargs["thinking"] = {"type": "adaptive"}
            else:
                if api_kwargs["max_tokens"] <= thinking_budget:
                    raise ValueError(
                        "Claude max_tokens must be greater than thinking_budget. "
                        "Increase api_kwargs.max_tokens or lower api_kwargs.thinking_budget."
                    )
                if "temperature" in self.api_kwargs and self.api_kwargs["temperature"] != 1.0:
                    raise ValueError(
                        "Claude fixed-budget thinking requires temperature=1.0. "
                        "Remove api_kwargs.temperature or set it to 1.0."
                    )
                api_kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
                api_kwargs["temperature"] = 1.0
        elif _model_rejects_temperature(self.model_id) and "temperature" in self.api_kwargs:
            raise ValueError(
                f"{self.model_id} rejects explicit temperature. "
                "Remove api_kwargs.temperature for this Claude model."
            )

        # image history truncation (applied before caching so cache keys stay stable)
        only_n = self.only_n_most_recent_images
        if only_n is not None:
            threshold = int(self.image_truncation_threshold)
            api_kwargs["messages"] = filter_to_n_most_recent_images(
                api_kwargs["messages"],
                only_n,
                min_removal_threshold=threshold,
            )

        # rolling prompt caching — 1 breakpoint on system + rolling N on
        # most-recent user turns, capped by ``cache_breakpoints`` kwarg.
        if self.api_kwargs.get("prompt_caching"):
            cap = int(self.api_kwargs.get("cache_breakpoints", 4))
            if cap > 0:
                inject_prompt_caching(api_kwargs["messages"], cap=cap)


@dataclass
class ClaudeDesktopUseAgent(_ClaudeBaseAgent, key=r"claude@(desktop|browser)@use"):
    """Self-contained Claude computer-use agent.

    Uses the Anthropic SDK for native toolsets and LiteLLM for older tools. No adapter,
    no processor, no generate_fn — all logic is self-contained.

    Args:
        model_id: Claude model ID (e.g. "claude-opus-4-6", "claude-opus-4-7").
        action_space: Action space for CUA-lite <-> Claude conversion.
        metadata: Env-supplied :class:`LiteBaseMetadata`. Reads of
            ``self.metadata.extra_tool_schemas`` (function tools surfaced alongside
            the native ``computer_*`` tool, e.g. browsergym
            ``send_msg_to_user``), ``self.metadata.valid_actions``
            (``[]`` drops the native ``computer_*`` tool — text-only obs
            modes; ``None`` keeps the full native action set; non-empty
            lists keep the opaque native tool whole because Anthropic exposes
            no per-action request-schema filter), and
            ``self.metadata.others.<key>`` happen in the request-building
            paths below. ``__post_init__`` installs an empty default
            ``LiteCUAMetadata()`` so direct construction doesn't require one.
        api_kwargs: ``{"max_tokens": int, "temperature": float,
            "effort": str (low|medium|high|xhigh|max; sent as top-level
            ``output_config``), "thinking_budget": int, "prompt_caching": bool, "tool_choice":
            "required"|"auto"|"none"|dict, "cache_breakpoints": int,
            "token_efficient_tools_beta": bool, "computer_tool_version": str,
            "computer_use_beta_flag": str}``. ``temperature`` is not set by
            default. Adaptive-only Opus models reject explicit temperature;
            fixed-budget thinking requires ``temperature=1.0`` and
            ``max_tokens > thinking_budget``; forced ``tool_choice`` rejects
            positive ``thinking_budget`` before the API request.
        api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if None).
        api_base: Optional API base URL override (e.g. for a litellm proxy).
        system_prompt: Optional system prompt override.
        system_prompt_suffix: Appended to ``system_prompt`` (or stands alone
            if ``system_prompt`` is None).
        resolution: Optional fixed image size sent to Claude. Defaults to
            ``None`` so screenshots are resized only as much as Anthropic's
            vision limits require, preserving aspect ratio.
        only_n_most_recent_images / image_truncation_threshold: Optional
            image-history truncation (skips oldest images, keeps last N).
        api_retry_max / api_retry_base_delay: Exponential-backoff retry on
            ``litellm.acompletion`` failures.
        tool_execution_timeout_s: Optional safety-net wrap around
            ``env.step(...)``. Leave ``None`` when the env is already bounded by
            :class:`~lite.gym.wrappers.StepTimeoutWrapper`; set a numeric value
            only for envs that need an agent-local timeout.
    """

    # Desktop-specific field; shared runtime/config fields live on
    # ``_ClaudeBaseAgent``. ``metadata.valid_actions=[]`` drops the native
    # ``computer_*`` tool for text-only modes.
    action_space: ClaudeDesktopActionSpace = field(default_factory=ClaudeDesktopActionSpace)

    def _build_beta_header(self, tool_config: dict[str, str]) -> str:
        """Build the comma-separated anthropic-beta header value.

        Includes the selected computer-use beta, if required, plus optional
        prompt-caching and token-efficient-tools betas.
        """
        betas: list[str] = [tool_config["beta_flag"]] if tool_config["beta_flag"] else []
        if self.api_kwargs.get("prompt_caching"):
            # caching requires its own beta flag to actually activate
            # cache_control breakpoints server-side.
            betas.append("prompt-caching-2024-07-31")
        if self.api_kwargs.get("token_efficient_tools_beta"):
            betas.append("token-efficient-tools-2025-02-19")
        return ",".join(betas)

    def _build_tools(
        self,
        tool_config: dict[str, str],
        display_w: int,
        display_h: int,
    ) -> list[dict[str, Any]]:
        """Build the tools list: native computer tool + extra function tools.

        Mirrors GPT's pattern: build the native tool schema, run it through
        ``action_space.filter_tool_schemas_for_valid_actions`` (which honors ``valid_actions=[]``
        to drop the native tool entirely for text-only obs modes), then
        append env-supplied function tools.
        """
        metadata = self._cua_metadata()
        provider_tool_declarations: list[dict[str, Any]] = [
            self._native_computer_schema(tool_config, display_w, display_h)
        ]
        if metadata.valid_actions is not None:
            provider_tool_declarations = type(
                self.action_space
            ).filter_tool_schemas_for_valid_actions(
                provider_tool_declarations,
                metadata.valid_actions,
            )
        # Env-supplied extras include finish schemas when the env declares
        # response/terminate in metadata.extra_tool_schemas.
        provider_tool_declarations.extend(self._extra_function_tools())
        return provider_tool_declarations

    def _native_computer_schema(
        self,
        tool_config: dict[str, str],
        display_w: int,
        display_h: int,
    ) -> dict[str, Any]:
        """The Anthropic native computer tool schema.

        Opus 5 uses a native toolset; older computer tools use LiteLLM's wrapper
        with explicit display dimensions. Toolset coordinates follow the image.
        """
        if tool_config["tool_version"] == "computer_toolset_20260801":
            return {
                "type": "computer_toolset_20260801",
                "configs": {
                    "zoom": {"enabled": False},
                    "cursor_position": {"enabled": False},
                },
            }
        return {
            "type": tool_config["tool_version"],
            "function": {
                "name": "computer",
                "parameters": {
                    "display_height_px": display_h,
                    "display_width_px": display_w,
                    "display_number": 1,
                },
            },
        }

    def _to_anthropic_function_tool(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Project a Lite tool schema to Anthropic's flat function-tool schema."""
        func = schema["function"]
        return {
            "name": func["name"],
            "description": func.get("description", ""),
            "input_schema": func["parameters"],
        }

    def _extra_function_tools(self) -> list[dict[str, Any]]:
        """Convert ``self.metadata.extra_tool_schemas`` (env-supplied function tools,
        e.g. ``report_infeasible`` from osworld_g) to Anthropic's flat
        function-tool schema. Shared by use + grounding subclasses."""
        return [self._to_anthropic_function_tool(et) for et in self.metadata.extra_tool_schemas]

    # -- predict ---------------------------------------------------------------

    async def predict(self, sample: LiteSample) -> LiteMessage:
        # Production drives rollouts via ``sample()``; single-step ``predict()``
        # (from a stored LiteSample) has no caller. Stubbed like the mobile sibling
        # so the self-contained family is uniform ("rollout = sample()").
        raise NotImplementedError(
            "ClaudeDesktopUseAgent.predict is not supported; use sample() for rollouts"
        )

    # -- sample ----------------------------------------------------------------

    async def sample(
        self,
        env: LiteBaseEnv,
        max_steps: int = 10_000,
        hooks: list[SampleHook] | None = None,
    ) -> LiteRLSample:
        """Run a full sampling loop on the environment.

        Manages conversation in liteLLM format directly. Each step:
        1. Send obs to API — screenshot for vision modes; AXTree text
           (in ``instruction`` on turn 0, in tool_result on subsequent
           turns) for text-only modes; both for VWA mixed mode.
        2. Parse response -> CUA-lite tool_calls
        3. env.step(tool_calls) -> next obs
        """
        hooks = hooks or []
        result: LiteRLSample | None = None
        completed = False

        # liteLLM conversation (managed separately from CUA-lite messages)
        tool_config = self._computer_tool_config()
        completion_messages: list[dict[str, Any]] = []

        # System prompt (only if user provided one via config)
        sys_prompt = self._effective_system_prompt()
        if sys_prompt:
            completion_messages.append({"role": "system", "content": sys_prompt})

        try:
            # Reset env
            observation = await env.reset()
            image_bytes = observation.image
            # ``image_bytes`` may be empty for text-only obs modes
            # (e.g. browsergym AXTree-only). Image embedding is gated on
            # presence below; nothing to assert here. Mirrors GPT's
            # image-optional sample loop.

            instruction = observation.text
            if not instruction:
                raise RuntimeError("No instruction returned on env.reset()")

            # Transient LiteSample for in-loop trajectory state. Final
            # LiteRLSample is built at the end with processed_images +
            # per-step LiteRLStep records (Claude has no adapter, so
            # processed_images == raw screenshots).
            trajectory = LiteSample(metadata=env.metadata)
            steps: list[LiteRLStep] = []
            episode_return = 0.0
            terminated = False
            truncated = False
            next_call_id = 0
            observation_image_index: int | None = None

            resize_target = self.resolution
            for step in range(max_steps):
                # Decode + downsample screenshot (text-only obs: skip).
                if image_bytes:
                    if observation_image_index is None:
                        image = await asyncio.to_thread(decode_image, image_bytes)
                        trajectory.images.append(image)
                        current_sent_image_index = len(trajectory.images) - 1
                    else:
                        image = trajectory.images[observation_image_index]
                        current_sent_image_index = observation_image_index
                    orig_w, orig_h = image.size

                    sent_image_b64, sent_w, sent_h = await asyncio.to_thread(
                        resize_for_claude_api,
                        image_bytes,
                        self.model_id,
                        resize_target,
                        many_image=self._MANY_IMAGE_REQUESTS,
                    )
                    scale_x = orig_w / sent_w if sent_w else 1.0
                    scale_y = orig_h / sent_h if sent_h else 1.0
                else:
                    # No image: build placeholder display dims for the native
                    # computer schema before valid_actions filtering.
                    image = None
                    current_sent_image_index = None
                    sent_image_b64 = None
                    placeholder = resize_target or (1024, 768)
                    sent_w, sent_h = placeholder
                    orig_w, orig_h = placeholder
                    scale_x = scale_y = 1.0

                # Build tools (computer + extra)
                tools = self._build_tools(tool_config, sent_w, sent_h)
                active_provider_tool_names = _claude_active_provider_tool_names(
                    tools
                ) - _extra_tool_names(self.metadata)

                # Add to conversation
                if step == 0:
                    # First turn: user message with instruction (+ screenshot
                    # iff present). For text-only obs modes the env's textual
                    # observation (AXTree) is already in ``instruction`` from
                    # the reset observation text.
                    user_content: list[dict[str, Any]] = [
                        {"type": "text", "text": instruction},
                    ]
                    traj_content: list[dict[str, Any]] = [
                        {"type": "text", "text": instruction},
                    ]
                    if sent_image_b64 is not None:
                        user_content.append(
                            mark_provider_visible_image_index(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{sent_image_b64}"},
                                },
                                current_sent_image_index,
                            )
                        )
                        if current_sent_image_index is not None:
                            traj_content.insert(
                                0, {"type": "image", "index": current_sent_image_index}
                            )
                    completion_messages.append({"role": "user", "content": user_content})
                    trajectory.messages.append({"role": "user", "content": traj_content})
                else:
                    # Subsequent turns: tool result with screenshot (or text)
                    # appended after env.step in previous iteration — see below.
                    pass

                # Call API
                api_kwargs: dict[str, Any] = {
                    "model": self.model_id,
                    "messages": completion_messages,
                    "tools": tools,
                    "max_tokens": self.api_kwargs.get("max_tokens", 4096),
                    "stream": False,
                }
                beta_header = self._build_beta_header(tool_config)
                if beta_header:
                    api_kwargs["headers"] = {"anthropic-beta": beta_header}
                if "temperature" in self.api_kwargs:
                    api_kwargs["temperature"] = self.api_kwargs["temperature"]
                self._apply_sampling_extras(api_kwargs)
                request_provider_visible_image_indices = (
                    _provider_visible_image_indices_from_claude_messages(api_kwargs["messages"])
                )
                api_kwargs["messages"] = _strip_claude_image_markers(api_kwargs["messages"])
                if self.api_key:
                    api_kwargs["api_key"] = self.api_key
                if self.api_base:
                    api_kwargs["api_base"] = self.api_base

                response, predict_s = await timed(
                    self._acompletion_with_retry(**api_kwargs)
                )  # predict_s = LLM-call wall time

                # Parse response
                parsed_response = parse_response_with_provenance(
                    response,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    action_space=self.action_space,
                    resolution=(orig_w, orig_h),
                    extra_tool_names=_extra_tool_names(self.metadata),
                    active_provider_tool_names=active_provider_tool_names,
                    call_id_start=next_call_id,
                )
                lite_message = parsed_response.message
                model_output_error = pop_model_output_error(lite_message)
                next_call_id += len(lite_message.get("tool_calls", []))

                # Record step — full API input/output
                prompt_for_log, response_for_log = _step_log_payloads(
                    api_kwargs["messages"],
                    response,
                )

                # Persist only the canonical Lite assistant message. The provider
                # output stays in LiteRLStep.response as diagnostics, not as a
                # replay sidecar on the saved message.
                trajectory.messages.append(lite_message)

                # LiteRLStep binds only provider-visible images. A tool result may
                # preserve additional action-batch frames in ``trajectory.images``
                # that Claude did not receive in the next request.
                rl_step = append_lite_rl_step(
                    steps=steps,
                    prompt=prompt_for_log,
                    image_indices=request_provider_visible_image_indices,
                    response=response_for_log,
                    finish_reason=parsed_response.finish_reason,
                    truncation_finish_reasons=_TRUNCATION_FINISH_REASONS,
                )

                # 4. Act: replay assistant turn, execute Lite tool calls, and update
                # RL accounting. Replay content comes from the single parse pass
                # above and keeps the thinking signatures Anthropic requires.
                append_provider_assistant_message(
                    completion_messages,
                    replay_content=parsed_response.replay_content,
                    provider_tool_uses=parsed_response.provider_tool_uses,
                )
                # A zero-tool-call final still steps the env, so hook actions and
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
                    append_terminal_tool_feedback=_append_claude_terminal_tool_feedback,
                )

                # Usage info from Claude API is diagnostic-only (not
                # threaded into LiteRLStep — its surface is kept minimal).
                # If a hook needs usage, it can read from the response
                # object directly.

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

                # 5. Feedback: append Claude-shaped tool results for the next turn.
                image_bytes, observation_image_index = await append_desktop_provider_feedback(
                    completion_messages=completion_messages,
                    response=response,
                    trajectory=trajectory,
                    step_result=step_result,
                    lite_tool_calls=lite_message.get("tool_calls", []),
                    model_id=self.model_id,
                    resize_target=resize_target,
                    many_image=self._MANY_IMAGE_REQUESTS,
                    provider_tool_uses=parsed_response.provider_tool_uses,
                    provider_call_provenance=parsed_response.provider_call_provenance,
                    provider_errors=parsed_response.provider_errors,
                )
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
                hooks=hooks,
                result=result,
                completed=completed,
                env=env,
            )


# =============================================================================
# Claude Mobile Agent (Android)
# =============================================================================

_CLAUDE_MOBILE_SYSTEM_PROMPT = (
    "You are operating a mobile device to complete a user task. "
    "At each step you receive a screenshot and a list of visible UI "
    "elements with their pixel coordinates.\n\n"
    "Coordinates are ABSOLUTE pixels on the {w}x{h} screenshot sent with "
    "the current request, with (0,0) at the top-left."
)

CLAUDE_MOBILE_API_KWARGS_DEFAULTS: dict[str, Any] = {
    "max_tokens": 4096,
    "thinking_budget": 2048,
    "effort": "medium",
    "prompt_caching": True,
    # Inherited from desktop (references have these)
    "cache_breakpoints": 4,
    "token_efficient_tools_beta": False,
}


# =============================================================================
# Claude Desktop Grounding (Point) Agent
# =============================================================================
#
# Single-step click prediction for grounding evals (osworld_g, screenspot_pro).
# Subclasses ``ClaudeDesktopUseAgent`` and narrows four things:
#   1. action_space → ``ClaudeDesktopGroundingPointActionSpace``: maps the
#      model's ``left_click(coordinate=[x,y])`` → cua-lite ``point(coord)``.
#   2. Default ``system_prompt`` → focused grounding instruction so YAML
#      configs don't need to repeat it.
#   3. ``api_kwargs`` defaults: ``thinking_budget=0`` (no extended thinking),
#      ``effort=low`` (the family ships medium), and a smaller ``max_tokens``
#      budget — the task is one click, no reasoning needed.
#   4. ``_build_tools`` returns a single click-shaped function tool
#      (``left_click({coordinate: [x, y]})``) instead of the full
#      ``computer_*`` native tool. Drops the screenshot/key/scroll/etc.
#      action space the model would otherwise reach for.

CLAUDE_GROUNDING_SYSTEM_PROMPT = (
    "You are an expert in using electronic devices and interacting with graphic interfaces.\n"
    "You are asked to find the UI element in the given screenshot corresponding to a given\n"
    "instruction. The instruction may be a short label, an action verb, or a description —\n"
    "make your best guess about which element it refers to from the screenshot context.\n"
    "If the requested element does NOT exist in the screenshot, call ``report_infeasible``\n"
    "with a brief reason. Otherwise output exactly one click action at the pixel\n"
    "coordinates of that element via the available tool.\n"
    "Don't output any analysis. Don't ask for clarification."
)

CLAUDE_GROUNDING_API_KWARGS_DEFAULTS: dict[str, Any] = {
    **CLAUDE_API_KWARGS_DEFAULTS,
    "max_tokens": 1024,
    "thinking_budget": 0,  # disable extended thinking on grounding
    # One click, no reasoning wanted -- so grounding does NOT inherit the
    # family's ``medium``. ``low`` is the floor Claude's effort scale offers
    # (there is no ``none``), and effort shapes tool-call tokens too, which is
    # exactly the "examine first" preamble this agent exists to suppress.
    "effort": "low",
}


@dataclass
class ClaudeDesktopGroundingPointAgent(
    ClaudeDesktopUseAgent, key=r"claude@(desktop|browser)@grounding\.point"
):
    """Claude desktop grounding (single-step click prediction).

    Uses the desktop loop with a click-only function surface, the grounding
    prompt, and shorter no-thinking API defaults.
    """

    # Grounding-specific api_kwargs (max_tokens=1024, thinking_budget=0, and
    # effort=low instead of the family's medium) drive the inherited
    # ``__post_init__`` via the class-attr override; they extend
    # ``CLAUDE_API_KWARGS_DEFAULTS``, so nothing else needs re-declaring.
    _API_KWARGS_DEFAULTS: ClassVar[dict[str, Any]] = CLAUDE_GROUNDING_API_KWARGS_DEFAULTS
    # One screenshot, one click, no history — so the stricter many-image ceiling
    # does not apply and the full single-image resolution is kept. That matters
    # here more than anywhere: screenspot_pro is high-res by design.
    _MANY_IMAGE_REQUESTS: ClassVar[bool] = False
    action_space: ClaudeDesktopGroundingPointActionSpace = field(
        default_factory=ClaudeDesktopGroundingPointActionSpace
    )
    system_prompt: str | None = CLAUDE_GROUNDING_SYSTEM_PROMPT

    def _computer_tool_config(self) -> dict[str, str]:
        """Grounding uses function tools and needs no native computer version."""
        return {}

    def _build_tools(
        self,
        tool_config: dict[str, str],
        display_w: int,
        display_h: int,
    ) -> list[dict[str, Any]]:
        """Single click-shaped function tool + inherited extra tools.

        The primary tool's function name + arg names mirror Anthropic's
        ``computer_*`` ``left_click`` action so the model is fluent;
        only the wrapping is "function tool" instead of the native
        computer-use tool, which strips ``screenshot`` / ``key`` /
        ``scroll`` / ``mouse_move`` / etc. from the model's reachable
        action space. ``self._extra_function_tools()`` is reused from
        ``ClaudeDesktopUseAgent`` so extra-tool pass-through (e.g.
        ``report_infeasible`` for osworld_g refusal) stays in one place.
        """
        metadata = self._cua_metadata()
        lite_action_schemas = type(self.action_space).get_tool_schemas()
        if metadata.valid_actions is not None:
            lite_action_schemas = type(self.action_space).filter_tool_schemas_for_valid_actions(
                lite_action_schemas,
                metadata.valid_actions,
            )
        provider_tool_declarations: list[dict[str, Any]] = [
            self._to_litellm_wrapped_function_tool(schema) for schema in lite_action_schemas
        ]
        provider_tool_declarations.extend(self._extra_function_tools())
        return provider_tool_declarations

    def _build_beta_header(self, tool_config: dict[str, str]) -> str:
        """No computer-use beta needed when there's no native computer tool."""
        betas: list[str] = []
        if self.api_kwargs.get("prompt_caching"):
            betas.append("prompt-caching-2024-07-31")
        if self.api_kwargs.get("token_efficient_tools_beta"):
            betas.append("token-efficient-tools-2025-02-19")
        return ",".join(betas)


@dataclass
class ClaudeMobileUseAgent(_ClaudeBaseAgent, key="claude@mobile@use"):
    """Claude mobile computer-use agent for AndroidWorld.

    Sibling to ``ClaudeDesktopUseAgent``; both inherit the shared provider
    plumbing (api_kwargs merge, system-prompt combine, retry wrapper,
    ``_apply_sampling_extras``) from ``_ClaudeBaseAgent``. Mobile builds
    function-only tool schemas (no native ``computer_*`` tool) and routes all
    tool_use blocks through ``ClaudeMobileActionSpace``.
    """

    _API_KWARGS_DEFAULTS: ClassVar[dict[str, Any]] = CLAUDE_MOBILE_API_KWARGS_DEFAULTS
    _RETRY_LOG_NAME: ClassVar[str] = "litellm.acompletion (mobile)"

    action_space: ClaudeMobileActionSpace = field(default_factory=ClaudeMobileActionSpace)
    # Mobile base prompt lives in ``system_prompt`` (NOT the suffix) so a config
    # that sets ``system_prompt_suffix`` APPENDS rather than clobbering it —
    # mirrors the GPT mobile agent. The suffix stays "" for caller additions.
    system_prompt: str | None = _CLAUDE_MOBILE_SYSTEM_PROMPT
    resolution: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.api_kwargs and "computer_use_beta_enabled" in self.api_kwargs:
            raise ValueError(
                "Claude mobile uses function tools, not the native computer-use tool; "
                "remove api_kwargs.computer_use_beta_enabled."
            )
        super().__post_init__()

    def _build_beta_header(self) -> str | None:
        """mobile path does NOT activate computer-use beta; only caching /
        token-efficient tools when their kwargs are on. Returns None if no
        betas need to be sent (in that case the caller should omit the header).
        """
        betas: list[str] = []
        if self.api_kwargs.get("prompt_caching"):
            betas.append("prompt-caching-2024-07-31")
        if self.api_kwargs.get("token_efficient_tools_beta"):
            betas.append("token-efficient-tools-2025-02-19")
        return ",".join(betas) if betas else None

    def _build_tools(self) -> list[dict[str, Any]]:
        """Wrapped LiteLLM function tools from mobile actions plus extra tools.

        No native ``computer_*`` tool is emitted on the mobile path.
        """
        provider_tool_declarations: list[dict[str, Any]] = []
        # Mobile GUI tools, filtered by valid_actions by provider-flat tool name
        # (STANDARD contract: None ⇒ all; a list ⇒ only those; [] ⇒ none).
        metadata = self._cua_metadata()
        lite_action_schemas = self.action_space.get_tool_schemas()
        if metadata.valid_actions is not None:
            lite_action_schemas = type(self.action_space).filter_tool_schemas_for_valid_actions(
                lite_action_schemas, metadata.valid_actions
            )
        provider_tool_declarations.extend(
            self._to_litellm_wrapped_function_tool(schema) for schema in lite_action_schemas
        )
        # Extra function tools (rare on mobile but supported)
        for et in self.metadata.extra_tool_schemas:
            provider_tool_declarations.append(self._to_litellm_wrapped_function_tool(et))
        return provider_tool_declarations

    def _system_prompt_for_mobile_request(self, sent_w: int, sent_h: int) -> str | None:
        prompt = self._effective_system_prompt()
        if prompt:
            prompt = prompt.replace("{w}", str(sent_w)).replace("{h}", str(sent_h))
        guidance = mobile_finish_guidance(_extra_tool_names(self.metadata))
        if not guidance:
            return prompt
        if not prompt:
            return guidance
        return f"{prompt}\n\n{guidance}"

    # -- sample ----------------------------------------------------------------

    async def sample(
        self,
        env: LiteBaseEnv,
        max_steps: int = 10_000,
        hooks: list[SampleHook] | None = None,
    ) -> LiteRLSample:
        """Mobile sampling loop. Parallel to desktop sample() but:
        - builds function-only tools
        - parses all tool_use blocks through ClaudeMobileActionSpace
        - no pending_safety_checks / computer_call handling
        """
        hooks = hooks or []
        result: LiteRLSample | None = None
        completed = False

        completion_messages: list[dict[str, Any]] = []

        try:
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
            observation_image_index: int | None = None

            for step in range(max_steps):
                if observation_image_index is None:
                    image = await asyncio.to_thread(decode_image, image_bytes)
                    trajectory.images.append(image)
                    current_sent_image_index = len(trajectory.images) - 1
                else:
                    image = trajectory.images[observation_image_index]
                    current_sent_image_index = observation_image_index
                orig_w, orig_h = image.size

                sent_image_b64, sent_w, sent_h = await asyncio.to_thread(
                    resize_for_claude_api,
                    image_bytes,
                    self.model_id,
                    self.resolution,
                    many_image=self._MANY_IMAGE_REQUESTS,
                )
                scale_x = orig_w / sent_w if sent_w else 1.0
                scale_y = orig_h / sent_h if sent_h else 1.0

                tools = self._build_tools()
                active_provider_tool_names = _claude_active_provider_tool_names(
                    tools
                ) - _extra_tool_names(self.metadata)

                if step == 0:
                    sys_prompt = self._system_prompt_for_mobile_request(sent_w, sent_h)
                    if sys_prompt:
                        completion_messages.append({"role": "system", "content": sys_prompt})
                    completion_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": instruction},
                                mark_provider_visible_image_index(
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{sent_image_b64}"
                                        },
                                    },
                                    current_sent_image_index,
                                ),
                            ],
                        }
                    )
                    trajectory.messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "index": current_sent_image_index},
                                {"type": "text", "text": instruction},
                            ],
                        }
                    )

                api_kwargs: dict[str, Any] = {
                    "model": self.model_id,
                    "messages": completion_messages,
                    "tools": tools,
                    "max_tokens": self.api_kwargs.get("max_tokens", 4096),
                    "stream": False,
                }
                if "temperature" in self.api_kwargs:
                    api_kwargs["temperature"] = self.api_kwargs["temperature"]
                beta_header = self._build_beta_header()
                if beta_header:
                    api_kwargs["headers"] = {"anthropic-beta": beta_header}
                self._apply_sampling_extras(api_kwargs)
                request_provider_visible_image_indices = (
                    _provider_visible_image_indices_from_claude_messages(api_kwargs["messages"])
                )
                api_kwargs["messages"] = _strip_claude_image_markers(api_kwargs["messages"])
                if self.api_key:
                    api_kwargs["api_key"] = self.api_key
                if self.api_base:
                    api_kwargs["api_base"] = self.api_base

                response, predict_s = await timed(
                    self._acompletion_with_retry(**api_kwargs)
                )  # predict_s = LLM-call wall time

                parsed_response = parse_mobile_response_with_provenance(
                    response,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    action_space=self.action_space,
                    resolution=(orig_w, orig_h),
                    extra_tool_names=_extra_tool_names(self.metadata),
                    active_provider_tool_names=active_provider_tool_names,
                    call_id_start=next_call_id,
                )
                lite_message = parsed_response.message
                model_output_error = pop_model_output_error(lite_message)
                next_call_id += len(lite_message.get("tool_calls", []))

                prompt_for_log, response_for_log = _step_log_payloads(
                    api_kwargs["messages"],
                    response,
                )

                # Persist only the canonical Lite assistant message. The provider
                # output stays in LiteRLStep.response as diagnostics, not as a
                # replay sidecar on the saved message.
                trajectory.messages.append(lite_message)

                rl_step = append_lite_rl_step(
                    steps=steps,
                    prompt=prompt_for_log,
                    image_indices=request_provider_visible_image_indices,
                    response=response_for_log,
                    finish_reason=parsed_response.finish_reason,
                    truncation_finish_reasons=_TRUNCATION_FINISH_REASONS,
                )

                # 4. Act: replay assistant turn, execute Lite tool calls, and update
                # RL accounting. Mobile uses the same Claude replay/history helper
                # as desktop, but a function-only tool surface.
                append_provider_assistant_message(
                    completion_messages,
                    replay_content=parsed_response.replay_content,
                    provider_tool_uses=parsed_response.provider_tool_uses,
                )
                # A zero-tool-call final still steps the env, so hook actions and
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
                    append_terminal_tool_feedback=_append_claude_terminal_tool_feedback,
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

                # 5. Feedback: append Claude-shaped tool results for the next turn.
                image_bytes, observation_image_index = await append_mobile_provider_feedback(
                    completion_messages=completion_messages,
                    response=response,
                    trajectory=trajectory,
                    step_result=step_result,
                    lite_tool_calls=lite_message.get("tool_calls", []),
                    model_id=self.model_id,
                    resolution=self.resolution,
                    many_image=self._MANY_IMAGE_REQUESTS,
                    provider_tool_uses=parsed_response.provider_tool_uses,
                    provider_call_provenance=parsed_response.provider_call_provenance,
                    provider_errors=parsed_response.provider_errors,
                )
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
                hooks=hooks,
                result=result,
                completed=completed,
                env=env,
            )

    async def predict(self, sample: LiteSample) -> LiteMessage:
        # This direct API family drives rollouts through ``sample()``; single-step
        # ``predict()`` is unsupported, matching the desktop sibling.
        raise NotImplementedError(
            "ClaudeMobileUseAgent.predict is not supported; use sample() for rollouts"
        )
