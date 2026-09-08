"""
Qwen3VL Adapters

Provides Qwen3VL-specific adapters for desktop, browser, and mobile:

    Qwen3VLBaseAdapter (self-contained: tool_calls routing, chat-template
    │                   token parsing, image preprocessing, system-prompt
    │                   + tools-section assembly. Assistant content is
    │                   flattened to a single ``type: "text"`` part —
    │                   workflow-agnostic, no Thought:/Action: assumptions.
    │                   Registered under the constrained platform/task wildcard;
    │                   yaml-driven rollouts set ``agent_id: "qwen3_vl.base"``
    │                   to resolve here.
    │                   ``valid_actions: []`` + ``extra_tools`` in the
    │                   yaml fully override the default action surface.)
    ├── Qwen3VLDesktopGroundingPointAdapter   (env-eval grounding, trimmed schema; desktop+browser)
    ├── Qwen3VLDesktopGroundingActionAdapter  (SFT-replay grounding, full schema; desktop+browser)
    ├── Qwen3VLMobileGroundingPointAdapter    (mobile env-eval grounding)
    ├── Qwen3VLMobileGroundingActionAdapter   (mobile SFT-replay grounding)
    └── Qwen3VLUseAdapter (intermediate: adds the ``use`` wire
        │                         format — ``Thought:`` / ``Action:`` line
        │                         rendering + ``action_description`` /
        │                         ``inline_reasoning`` round-trip, with the
        │                         ``enable_inline_reasoning`` toggle. Decomposed
        │                         only; opaque/verbatim turns use a passthrough
        │                         adapter, not a branch here.)
        ├── Qwen3VLDesktopUseAdapter   (multi-turn ``use``; desktop+browser)
        └── Qwen3VLMobileUseAdapter    (mobile ``use``; inherits the default)

The ``Desktop`` adapters register under the
``r"qwen3_vl@(desktop|browser)@..."`` regex, so the same class is reachable
via both ``qwen3_vl@desktop@...`` and ``qwen3_vl@browser@...`` keys.

``grounding.point`` and ``grounding.action`` coexist: the former is the
env-side eval task type (trimmed click-only schema, single-step
``LitePointActionSpace.point``); the latter is the SFT-data preproc task
type (full action vocabulary, multi-action shape — ``lite/data/preproc/``
parquet replays land here). For pass-through (understanding,
grounding/bbox), use AsIsAdapter from base.

See ``lite/agents/models/README.md`` for model-family usage notes.

Usage:
    from lite.agents.models.qwen3_vl.adapter import (
        Qwen3VLDesktopGroundingPointAdapter,
        Qwen3VLDesktopGroundingActionAdapter,
        Qwen3VLDesktopUseAdapter,
        Qwen3VLMobileGroundingPointAdapter,
        Qwen3VLMobileGroundingActionAdapter,
        Qwen3VLMobileUseAdapter,
    )

"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import re
from typing import Any, ClassVar

from lite.agents.core.action_space import BaseActionSpace, assemble_tool_schemas
from lite.agents.core.adapter import (
    AgentAdapterRegistry,
    AsIsAdapter,
    BaseAgentAdapter,
)
from lite.agents.core.protocol.base import FullHistoryProtocol
from lite.agents.models.qwen3_vl.action_space import (
    Qwen3VLDesktopActionSpace,
    Qwen3VLDesktopGroundingPointActionSpace,
    Qwen3VLMobileActionSpace,
    Qwen3VLMobileGroundingPointActionSpace,
)
from lite.agents.models.qwen3_vl.protocol import Qwen3VLHistoryProtocol
from lite.agents.types import AgentMessage, AgentStep
from lite.core import (
    LiteCUAMetadata,
    LiteMessage,
    LiteSample,
)
from lite.core.messages import make_assistant_content
from lite.core.messages.final import mark_model_output_error
from lite.core.messages.turns import truncate_sample_to_turn
from lite.core.tools.calls import (
    tool_call_arguments,
    tool_call_name,
)
from lite.core.tools.extra_tools import (
    LiteAppLaunchToolSet,
    LiteFinishToolSet,
    open_app_names_from_metadata,
)
from lite.core.tools.schemas import (
    tool_schema_name,
    tool_schema_parameters,
    validate_extra_tool_schemas,
)
from lite.utils.image import smart_resize

logger = logging.getLogger(__name__)

# Qwen action values that end an episode. ``terminate`` here is a Qwen ACTION
# VALUE inside ``computer_use(action=...)``; it collides by string only with the
# Lite ``terminate`` extra-tool NAME. The mapping between the two layers is the
# action space's ``QWEN_ACTION_VALUE_TO_EXTRA_TOOL_NAMES`` table.
_QWEN_FINISH_ACTION_VALUES = frozenset({"answer", "terminate"})
_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"


def _strip_tool_calls(text: str) -> str:
    """Remove every ``<tool_call>`` span, however the model malformed it.

    One owner for the scrub, because two of them drifted: the copy that cleaned
    the published content handled a dropped opener and stray bare tags, while
    the copy that cleaned an unclosed-``<think>`` remainder handled only the
    paired block -- so raw tool-call JSON leaked into ``reasoning_content``.
    """
    open_tag = re.escape(_TOOL_CALL_OPEN)
    close_tag = re.escape(_TOOL_CALL_CLOSE)
    text = re.sub(rf"(?:{open_tag}\s*)+.*?{close_tag}", "", text, flags=re.DOTALL)
    text = re.sub(rf"\{{[^<]*\}}\s*{close_tag}", "", text)
    return text.replace(_TOOL_CALL_OPEN, "").replace(_TOOL_CALL_CLOSE, "")

# Qwen3VL vision constants. ``_smart_resize_image`` lives as a method on
# :class:`Qwen3VLBaseAdapter` (reads the per-subclass ``smart_resize_factor`` /
# ``smart_resize_max_pixels`` class attrs) so the Qwen2.5-VL subclass can
# override just those two constants (factor=28, smaller max_pixels) instead of
# re-implementing the resize body.
_QWEN3VL_FACTOR = 32
_QWEN3VL_MAX_PIXELS = 16 * 16 * 4 * 12800

# =============================================================================
# System Prompts
# =============================================================================

# Platform-agnostic response-format prompt for Qwen3-VL ``use`` adapters.
# This default is the 2-part Action + ``<tool_call>`` adaptation used when
# inline reasoning is disabled. Finish/answer guidance is not hard-coded
# here: it is rendered from the finish action values the wrapper still offers
# after the row/env's ``extra_tool_schemas`` gate. When a row exposes the
# schema-level ``answer`` action, the adapter translates it to
# ``response(text=...)`` so the env can forward it — see
# :meth:`Qwen3VLMobileActionSpace.convert_tool_calls_from_agent` and
# AndroidLab / AndroidWorld env wrappers.
#
# Subclasses SFT'd against a divergent prompt override ``system_prompt`` with a
# full rewrite rather than patching this one.

USE_SYSTEM_PROMPT = """# Response format

Response format for every step:
1) Action: a short imperative describing what to do in the UI.
2) A single <tool_call>...</tool_call> block.

Rules:
- Output exactly in the order: Action, <tool_call>.
- Be brief: one sentence for Action.
- Do not output anything else outside those parts."""

# Grounding-only system prompt. Single-step click; no Thought / Action prose.
# The model emits exactly ONE <tool_call> with a click coordinate. The
# trimmed schema lives on :class:`Qwen3VLDesktopGroundingPointActionSpace`
# (``computer_use`` with ``left_click``) or
# :class:`Qwen3VLMobileGroundingPointActionSpace` (``mobile_use`` with
# ``click``). Refusal — when the env exposes the ``report_infeasible`` extra
# tool — is the only other tool the model may emit; the env-side prompt /
# extra_tools schema teaches the model when to use it.
GROUNDING_POINT_SYSTEM_PROMPT = """# Response format

For each grounding instruction, return a single <tool_call>...</tool_call> block.

Rules:
- Output exactly one <tool_call> block, nothing else.
- Click the (x, y) pixel coordinate of the target element.
- Do not produce any prose, explanation, or extra tool calls."""

# =============================================================================
# Qwen3VL Base Adapter (shared logic)
# =============================================================================

@dataclasses.dataclass
class Qwen3VLBaseAdapter(
    BaseAgentAdapter,
    key=(
        r"qwen3_vl\.base"
        r"(@(desktop|browser|mobile)"
        r"@(use|understanding|grounding\.action|grounding\.point|grounding\.bbox))?"
    ),
):
    """
    Self-contained base adapter for the Qwen3VL chat-template format
    (desktop, browser, mobile).

    Decoupled from any specific workflow / system prompt. The base
    handles only the platform-invariant parts of the wire format:

      * ``computer_use`` / ``mobile_use`` tool-call routing through
        ``self.action_space``;
      * ``<think>`` and ``<tool_call>`` chat-template token parsing in
        :meth:`parse_raw_assistant_response`;
      * ``smart_resize`` image preprocessing per Qwen3VL vision
        constraints;
      * inline tools schema in the system message via
        :meth:`_build_tools_section`;
      * extra-tool collision check.

    Assistant-message content is normalized to a single
    ``[{"type": "text", "text": ...}]`` part: any text-bearing parts are
    concatenated by their ``text`` fields (the part ``type`` is
    ignored). ``convert_message_from_agent`` round-trips that text part
    verbatim. Workflow-specific structured content
    (``action_description`` / ``inline_reasoning``, ``Thought:`` /
    ``Action:`` line conventions, multi-part Memory/Progress blocks) is
    the responsibility of :class:`Qwen3VLUseAdapter` and below
    — base does NOT preset any of those.

    Subclasses must set ``action_space`` and ``protocol``.

    Env-supplied non-standard tools (e.g. browser nav back/goto) are read from
    ``self.metadata.extra_tool_schemas`` (the single source of truth, forwarded
    by make / export_sft) — NOT an adapter field.
    """

    # ─── Core (action surface + history shaping) ─────────────────────
    # ``BaseActionSpace`` is the workflow-agnostic default: empty
    # ``_SCHEMAS`` (no tools surfaced) + identity pass-through
    # ``convert_tool_calls_*`` (so any tool_call routes unchanged).
    # The base adapter does NOT presume any platform-specific action
    # space. BrowserGym (the primary base-flavor caller) passes
    # ``valid_actions: []`` + its own ``extra_tools`` to surface only
    # env-defined tools in the system message, and emits ``<tool_call>``
    # blocks whose ``function.name`` matches an extra_tool — so the
    # action_space conversion path is a no-op. Concrete subclasses
    # (use / Grounding leaves) override this via
    # ``dataclasses.field(default_factory=...)``.
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=BaseActionSpace
    )
    protocol: FullHistoryProtocol = dataclasses.field(
        default_factory=FullHistoryProtocol
    )

    # ─── General configuration ───────────────────────────────────────
    system_prompt: str | None = None
    # Some single-turn QA adapters use the Qwen chat-template parser but do not
    # want a model-facing function-call surface. When disabled, render_step keeps
    # the configured system prompt and leaves any final answer as plain text; the
    # shared agent loop still wraps that text into the internal response action.
    render_tools_section: bool = True
    # ``extra_tools`` and ``valid_actions`` are inherited as properties
    # on ``BaseAgentAdapter`` (reading from ``self.metadata``). Env is
    # the single source of truth — yaml overrides go through env, not
    # adapter kwargs.
    # Canonical extra tools this adapter's Qwen wrapper already renders as
    # ``action`` values, so they are never repeated as standalone schemas:
    # desktop ``{response, terminate}`` (``action=answer`` / ``action=terminate``),
    # mobile ``{response, terminate, open_app}`` (plus ``action=open``).
    natively_rendered_extra_tool_names: ClassVar[frozenset[str]] = frozenset()

    # ─── Qwen3-VL-specific knobs ─────────────────────────────────────
    # Whether to apply ``_smart_resize_image`` (round to 32-px multiples + cap
    # to ``max_pixels``) before sending images to the vision tower. ON for
    # desktop / browser (where screenshots can be 1920×1080 and need capping); OFF
    # for mobile, because the official Qwen3-VL mobile cookbook and MAI-UI
    # send raw screenshots — resizing introduces a small
    # coordinate-space shift that doesn't match the SFT distribution.
    smart_resize_enabled: bool = True
    # Per-subclass smart_resize knobs (read by ``_smart_resize_image``).
    # Qwen3-VL: factor=32 (patch 16 × merge 2), large pixel cap. The
    # Qwen2.5-VL subclass overrides these (factor=28, smaller max_pixels).
    smart_resize_factor: int = _QWEN3VL_FACTOR
    smart_resize_max_pixels: int = _QWEN3VL_MAX_PIXELS
    # Native ``<think>`` channel toggle. False (default) matches the
    # ``-Instruct`` checkpoints whose chat_template suppresses ``<think>``
    # by default and avoids inflating prefix tokens during eval. Set to
    # True for the ``-Thinking`` checkpoints (e.g. Qwen3-VL-8B-Thinking)
    # — :class:`Qwen3VLBaseAgent.build_generation_prompt` forwards this
    # value to ``processor.apply_chat_template(..., enable_thinking=...)``.
    enable_thinking: bool = False

    # ``system_prompt`` defaults are set per workflow subclass:
    # ``Qwen3VLUseAdapter`` pins it to ``USE_SYSTEM_PROMPT`` (which
    # elicits the ``Action: ...`` wire format it parses), grounding adapters pin
    # ``GROUNDING_POINT_SYSTEM_PROMPT``, and the base leaves it ``None`` so the
    # system message is just the tools schema. Extra-tool schema collision
    # checks live in BaseAgentAdapter (shared).

    def _tool_calls_to_agent_ordered(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Lite calls without moving standalone extra tools across action-batch calls."""
        self._require_standalone_tool_schemas(tool_calls)
        visible_tool_schemas = self._tool_schemas_for_tools_section()
        # An extra tool the wrapper advertises as an ``action`` value must be
        # rendered through the action space, not passed through standalone.
        natively_rendered = self._extra_tool_names_rendered_as_qwen_action_values(
            visible_tool_schemas
        )
        extra_names = {
            name for name in self.active_extra_tool_names()
            if name not in natively_rendered
        }
        out: list[dict[str, Any]] = []
        for tc in tool_calls:
            if self._admits_active_extra_tool_call(tc, allowed_names=extra_names):
                out.append({
                    "name": tool_call_name(tc),
                    "arguments": tool_call_arguments(tc),
                })
            else:
                out.extend(self.action_space.convert_tool_calls_to_agent([tc]))
        return out

    def _unwrap_standalone_extra_tool_call(
        self,
        agent_tool_call: dict[str, Any],
    ) -> dict[str, Any]:
        """Lift a standalone extra tool out of the Qwen wrapper's ``action`` slot.

        The model keeps the wrapper but names a STANDALONE tool as the action
        value — ``computer_use(action="goto", url=...)`` for browsergym's
        ``goto``. ``extra_tools`` gates tools and ``valid_actions`` gates
        actions, so that call belongs at the top level: nested as an
        action-batch child it is a canonical shape env ingress
        (``nested_extra_tool_action_batch_child_message``) and row validation
        (``lite/data/utils/rows.py``) both reject, so the action never runs.

        Only a call whose remaining arguments route to the extra's own schema is
        lifted, by the same key-shape predicate those two rejections use.
        Anything else stays a wrapper action and keeps the unknown-action
        feedback the action space already gives it.
        """
        if not isinstance(agent_tool_call, dict):
            return agent_tool_call
        arguments = agent_tool_call.get("arguments")
        if not isinstance(arguments, dict):
            return agent_tool_call
        wrapper_names = {
            tool_schema_name(schema) for schema in self.action_space.get_tool_schemas()
        }
        if agent_tool_call.get("name") not in wrapper_names:
            return agent_tool_call
        action = arguments.get("action")
        if not isinstance(action, str):
            return agent_tool_call
        lifted = {
            "name": action,
            "arguments": {k: v for k, v in arguments.items() if k != "action"},
        }
        if not self._matches_active_extra_tool_schema_keys(lifted):
            return agent_tool_call
        return {**agent_tool_call, **lifted}

    def _set_converted_tool_calls_from_agent(
        self,
        result: dict[str, Any],
        tool_calls: list[dict[str, Any]],
    ) -> None:
        converted = self._route_agent_tool_calls_to_lite(
            [self._unwrap_standalone_extra_tool_call(tc) for tc in tool_calls]
        )
        result["tool_calls"] = converted
        if tool_calls and not converted:
            mark_model_output_error(
                result,
                "tool call did not satisfy the active tool schema or native action grammar",
            )

    @staticmethod
    def _qwen_wrapper_action_values(schema: dict[str, Any]) -> list[str]:
        """The ``action`` enum a rendered Qwen wrapper schema still offers."""
        return (
            tool_schema_parameters(schema)
            .get("properties", {})
            .get("action", {})
            .get("enum", [])
        )

    def _extra_tool_names_rendered_as_qwen_action_values(
        self,
        tool_schemas: list[dict[str, Any]],
    ) -> frozenset[str]:
        """Extra tools the visible Qwen wrapper already offers as action values.

        Read straight off the rendered wrapper enum through the action space's
        ``QWEN_ACTION_VALUE_TO_EXTRA_TOOL_NAMES`` table, so an extra tool whose
        action value was pruned — or whose wrapper is gone entirely — keeps its
        standalone schema instead of losing every spelling.
        """
        if not self.natively_rendered_extra_tool_names:
            return frozenset()
        action_value_to_extra_tools = (
            type(self.action_space).QWEN_ACTION_VALUE_TO_EXTRA_TOOL_NAMES
        )
        qwen_native_tool_names = type(self.action_space).get_tool_names()
        rendered: set[str] = set()
        for schema in tool_schemas:
            if tool_schema_name(schema) not in qwen_native_tool_names:
                continue
            for action_value in self._qwen_wrapper_action_values(schema):
                rendered |= action_value_to_extra_tools.get(action_value, frozenset())
        return self.natively_rendered_extra_tool_names & rendered

    def _native_tool_names_with_only_qwen_finish_action_values(
        self,
        tool_schemas: list[dict[str, Any]],
    ) -> frozenset[str]:
        """Qwen wrapper tools left carrying nothing but finish action values.

        Example: ``valid_actions=[]`` + ``extra_tools=["terminate"]`` trims
        ``computer_use`` down to ``enum=["terminate"]``. Rendering a provider-native wrapper
        that can only end the episode is confusing, so the prompt drops it and
        shows the standalone ``terminate`` schema instead.
        """
        qwen_native_tool_names = type(self.action_space).get_tool_names()
        finish_only: set[str] = set()
        for schema in tool_schemas:
            name = tool_schema_name(schema)
            if name not in qwen_native_tool_names:
                continue
            action_values = self._qwen_wrapper_action_values(schema)
            if action_values and set(action_values) <= _QWEN_FINISH_ACTION_VALUES:
                finish_only.add(name)
        return frozenset(finish_only)

    def _with_native_open_app_hint(
        self,
        tool_schemas: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Attach the env's app catalog to Qwen mobile's ``action=open`` prose.

        Qwen mobile spells the canonical ``open_app`` extra tool as an action
        value, so the app-name enum the env ships on the standalone schema has
        nowhere else to land in the rendered prompt.
        """
        if "open_app" not in self.natively_rendered_extra_tool_names:
            return tool_schemas
        app_names = open_app_names_from_metadata(self.metadata)
        if not app_names:
            return tool_schemas

        qwen_native_tool_names = type(self.action_space).get_tool_names()
        result = copy.deepcopy(tool_schemas)
        apps_json = json.dumps(app_names, ensure_ascii=False, separators=(",", ":"))
        hint = (
            "For `action=open`, set `text` to the exact app name. "
            f"Available apps: {apps_json}."
        )
        for schema in result:
            if tool_schema_name(schema) not in qwen_native_tool_names:
                continue
            action_prop = (
                tool_schema_parameters(schema)
                .get("properties", {})
                .get("action", {})
            )
            if "open" not in action_prop.get("enum", []):
                continue
            description = action_prop.get("description", "")
            if hint not in description:
                action_prop["description"] = (
                    f"{description}\n{hint}" if description else hint
                )
        return result

    def _assemble_tool_schemas(self) -> list[dict[str, Any]]:
        """Assemble generic base-adapter tools without reading CUA-only fields."""
        if type(self.action_space) is BaseActionSpace:
            return assemble_tool_schemas(
                self.action_space,
                self.action_space.get_tool_schemas(),
                valid_actions=None,
                extra_tool_schemas=self.metadata.extra_tool_schemas,
            )
        return super()._assemble_tool_schemas()

    def _tool_schemas_for_tools_section(self) -> list[dict[str, Any]]:
        """Schemas visible in Qwen's rendered ``<tools>`` block.

        An extra tool is hidden from the standalone list exactly while the
        visible wrapper enum still spells it as an ``action`` value, so the
        prompt offers one spelling per tool. A wrapper reduced to finish-only
        action values is dropped instead, and its finish schemas stay
        standalone. Mobile ``open`` is not a finish value, so it stays native.
        """
        tool_schemas = self._assemble_tool_schemas()
        # The base pipeline owns only the ``valid_actions`` gate. Which Qwen action
        # values a sample's active extra tools leave reachable is Qwen policy, so the
        # family gate runs here rather than behind a hook on the base class.
        gate = getattr(
            type(self.action_space),
            "filter_qwen_action_values_for_active_extra_tools",
            None,
        )
        if gate is not None:
            tool_schemas = gate(tool_schemas, self.active_extra_tool_names())
        finish_only_wrappers = self._native_tool_names_with_only_qwen_finish_action_values(
            tool_schemas
        )
        if finish_only_wrappers:
            return self._with_native_open_app_hint([
                schema for schema in tool_schemas
                if tool_schema_name(schema) not in finish_only_wrappers
            ])
        hidden = self._extra_tool_names_rendered_as_qwen_action_values(tool_schemas)
        return self._with_native_open_app_hint([
            schema for schema in tool_schemas
            if tool_schema_name(schema) not in hidden
        ])

    def _build_tools_section(self, image_size: tuple[int, int] | None = None) -> str:
        """Format tool schemas into Qwen3VL chat_template's tools section text.

        If ``image_size`` is provided, substitutes any ``{display_width_px}`` /
        ``{display_height_px}`` placeholders in the tool descriptions with the
        actual resized image dims. Qwen3-VL itself emits no such placeholders
        (no-op), but the Qwen2.5-VL subclass passes ``image_size`` to inject the
        resolution declared in its ``computer_use`` schema.
        """
        tool_schemas = self._tool_schemas_for_tools_section()
        # Byte policy (this family owns it; there is no shared Qwen prompt
        # helper): one JSON object per line, ``json.dumps`` defaults, so
        # non-ASCII escapes as ``\uXXXX``. Fara reuses the same nested schema
        # shape but dumps with ``ensure_ascii=False`` in its own adapter.
        validate_extra_tool_schemas(
            tool_schemas,
            where="Qwen3VLBaseAdapter._build_tools_section.tool_schemas",
        )
        tools_json = "\n".join(json.dumps(schema) for schema in tool_schemas)
        if image_size is not None:
            W, H = image_size
            tools_json = (
                tools_json
                .replace("{display_width_px}", str(W))
                .replace("{display_height_px}", str(H))
            )
        return (
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>\n"
            f"{tools_json}\n"
            "</tools>\n\n"
            "For each function call, return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>"
        )

    def _smart_resize_image(self, img):
        """Resize a PIL Image using smart_resize to fit the vision constraints.

        Reads the per-subclass ``smart_resize_factor`` / ``smart_resize_max_pixels``
        class attrs (Qwen3-VL: 32 / large cap; Qwen2.5-VL overrides to 28 / smaller).
        """
        w, h = img.size
        new_h, new_w = smart_resize(
            height=h, width=w,
            factor=self.smart_resize_factor, max_pixels=self.smart_resize_max_pixels,
        )
        if (new_w, new_h) != (w, h):
            img = img.resize((new_w, new_h))
        return img

    def _process_image_after_target(self, img):
        """Smart-resize per Qwen3VL vision constraints (32-px multiples,
        capped pixel count). Disabled (identity) when
        ``smart_resize_enabled=False`` — set on mobile adapters whose
        reference implementations send raw screenshots.

        Stage-2 hook called after ``BaseAgentAdapter.process_image``'s
        exact-stretch target-resize (when ``self.resolution`` is set).
        """
        if not self.smart_resize_enabled:
            return img
        return self._smart_resize_image(img)

    def _finish_guidance_for_active_surface(self) -> str | None:
        """Finish guidance for Qwen-native folded finish tools."""
        if self.system_prompt != USE_SYSTEM_PROMPT:
            return None

        qwen_native_tool_names = type(self.action_space).get_tool_names()
        visible_actions: set[str] = set()
        for schema in self._tool_schemas_for_tools_section():
            if tool_schema_name(schema) not in qwen_native_tool_names:
                continue
            visible_actions.update(
                action_value
                for action_value in self._qwen_wrapper_action_values(schema)
                if action_value in _QWEN_FINISH_ACTION_VALUES
            )

        if {"answer", "terminate"} <= visible_actions:
            return (
                "- If finishing, use action=answer for a question-answering task, "
                "otherwise use action=terminate in the tool call."
            )
        if "answer" in visible_actions:
            return (
                "- If finishing a question-answering task, use action=answer "
                "in the tool call."
            )
        if "terminate" in visible_actions:
            return "- If finishing, use action=terminate in the tool call."
        return None

    def _system_prompt_for_active_surface(self) -> str | None:
        """System prompt text after sample-specific tool-surface gating."""
        guidance = self._finish_guidance_for_active_surface()
        if not self.system_prompt:
            return guidance
        if not guidance:
            return self.system_prompt
        return f"{self.system_prompt}\n{guidance}"

    def render_step(
        self,
        sample: LiteSample,
        k: int,
        processed,
        **kwargs,
    ) -> AgentStep:
        """Render turn ``k``: protocol on truncated history + system prompt
        with inline tools section.

        ``processed`` is the trajectory-level processed-image list (already
        through ``process_image``); not consumed here because messages
        keep ImageContent indices.
        """
        truncated = truncate_sample_to_turn(sample, k)
        messages = self.protocol.process_messages(truncated.messages)

        result_messages: list[AgentMessage] = []

        # System prompt with inline tools.
        parts: list[str] = []
        system_prompt = self._system_prompt_for_active_surface()
        if system_prompt:
            parts.append(system_prompt)
        if self.render_tools_section:
            parts.append(self._build_tools_section())
        if parts:
            system_text = "\n\n".join(parts)
            result_messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_text}],
            })

        for msg in messages:
            result_messages.append(self.convert_message_to_agent(msg))

        return result_messages

    def _convert_message_to_agent(
        self,
        message: LiteMessage,
        **kwargs,
    ) -> dict[str, Any]:
        """LiteMessage → AgentMessage (chat-template-friendly).

        Routes ``tool_calls`` through the action_space (extras pass
        through unchanged). For assistant content, picks ONLY the
        ``type: "text"`` parts and concatenates their ``text`` fields
        into a single ``[{"type": "text", "text": ...}]`` part. Other
        content types (``action_description``, ``inline_reasoning``,
        etc.) are workflow-specific and are NOT consumed by base —
        :class:`Qwen3VLUseAdapter` overrides this method to
        render them as ``Thought:`` / ``Action:`` lines.

        Top-level ``reasoning_content`` (Qwen3-VL's NATIVE ``<think>``
        channel) is preserved untouched — the chat_template's
        Thinking-mode branch wraps it as ``<think>...</think>``
        automatically.
        """
        result = copy.deepcopy(message)
        if result.get("role") != "assistant":
            return result
        if "tool_calls" in result:
            result["tool_calls"] = self._tool_calls_to_agent_ordered(result["tool_calls"])
        content = message.get("content") or []
        texts = [
            p["text"] for p in content
            if isinstance(p, dict) and p.get("type") == "text" and p.get("text")
        ]
        result["content"] = (
            [{"type": "text", "text": "\n".join(texts)}] if texts else []
        )
        return result

    def convert_message_from_agent(
        self,
        message: AgentMessage,
        **kwargs,
    ) -> LiteMessage:
        """AgentMessage → LiteMessage. Symmetric inverse of
        :meth:`convert_message_to_agent`. Operates on the
        chat-template-level wire format only — ``<think>`` and
        ``<tool_call>`` tokens were already extracted by
        :meth:`parse_raw_assistant_response` into top-level
        ``reasoning_content`` / ``tool_calls``.

        For role=assistant:
          * Tool calls: ``BaseAgentAdapter._route_agent_tool_calls_to_lite``
            (env ``extra_tool_schemas`` names pass through unchanged and win a
            name collision with the canonical GUI actions — e.g. webvoyager
            SoM ``click(index)``; everything else goes to
            ``action_space.convert_tool_calls_from_agent``). Works whether
            tool_calls came from parse_raw (local pipeline) or from an API agent
            that bypassed parse_raw.
          * Content text: kept verbatim as a single
            ``[{"type": "text", "text": <raw>}]`` part. System-prompt-
            level conventions (``Thought:`` / ``Action:`` extraction,
            multi-part Memory/Progress blocks) live on
            :class:`Qwen3VLUseAdapter`.
        """
        result = copy.deepcopy(message)
        if result.get("role") != "assistant":
            return result

        if "tool_calls" in result:
            self._set_converted_tool_calls_from_agent(result, result["tool_calls"])

        raw_text = ""
        for part in message.get("content") or []:
            if part.get("type") == "text" and part.get("text"):
                raw_text = part["text"]
                break
        result["content"] = (
            [{"type": "text", "text": raw_text}] if raw_text else []
        )
        return result

    @staticmethod
    def _extract_json_tool_calls(response: str) -> list[dict[str, Any]]:
        """Extract ``<tool_call>{json}</tool_call>`` blocks → agent-format dicts.

        Shared by Qwen3-VL and the Qwen2.5-VL subclass (whose chat_template uses
        the same JSON ``<tool_call>`` tag format). Two paths: normal
        (``<tool_call>`` open token present) and sglang-VLM fallback (open token
        id 151657 was stripped but ``</tool_call>`` remains as plain text).
        """
        def _flat_call(tc_json: Any) -> dict[str, Any] | None:
            if not isinstance(tc_json, dict):
                logger.warning("Ignoring non-object tool_call JSON: %s", tc_json)
                return None
            name = tc_json.get("name")
            arguments = tc_json.get("arguments")
            if not isinstance(name, str) or not isinstance(arguments, dict):
                logger.warning("Ignoring non-flat tool_call JSON: %s", tc_json)
                return None
            return {"name": name, "arguments": arguments}

        tool_calls: list[dict[str, Any]] = []
        if _TOOL_CALL_OPEN in response:
            open_tag = re.escape(_TOOL_CALL_OPEN)
            close_tag = re.escape(_TOOL_CALL_CLOSE)
            block_re = rf"(?:{open_tag}\s*)+(.*?)\s*{close_tag}"
            for m in re.finditer(block_re, response, re.DOTALL):
                body = m.group(1).strip()
                # Qwen3-VL sometimes repeats the opening delimiter before the
                # JSON payload. Treat only that duplicate delimiter as noise;
                # the remaining payload still must be valid flat JSON.
                while body.startswith(_TOOL_CALL_OPEN):
                    body = body[len(_TOOL_CALL_OPEN):].strip()
                try:
                    tc_json = json.loads(body)
                    if call := _flat_call(tc_json):
                        tool_calls.append(call)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse tool_call JSON: %s", body)
        elif _TOOL_CALL_CLOSE in response:
            for m in re.finditer(rf"(\{{[^<]*\}})\s*{re.escape(_TOOL_CALL_CLOSE)}", response):
                try:
                    tc_json = json.loads(m.group(1).strip())
                    if call := _flat_call(tc_json):
                        tool_calls.append(call)
                except json.JSONDecodeError:
                    continue
        return tool_calls

    def parse_raw_assistant_response(
        self,
        response: str,
        **kwargs,
    ) -> AgentMessage:
        """Parse raw model output into an ``AgentMessage`` by extracting
        chat-template-level tokens only — these are baked into the model's
        chat template and are invariant across system prompts:

          * ``<think>...</think>`` → top-level ``reasoning_content``
            (Qwen3-VL's NATIVE thinking channel; the same field the
            chat_template's Thinking-mode branch wraps on the way back).
          * ``<tool_call>{json}</tool_call>`` → top-level ``tool_calls``
            (via :meth:`_extract_json_tool_calls`), still in raw agent format.

        The prose remainder (after stripping those two tokens) goes
        verbatim into a single ``{"type": "text", "text": ...}`` content
        part. System-prompt-level conventions (``Thought:`` / ``Action:``
        lines, Memory/Progress multi-part blocks) are NOT parsed here —
        that's the contract of :meth:`convert_message_from_agent`.
        """
        result: AgentMessage = {"role": "assistant"}

        m = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        if m:
            result["reasoning_content"] = m.group(1).strip()
        elif "</think>" in response:
            result["reasoning_content"] = response.split("</think>", 1)[0].strip()

        tool_calls = self._extract_json_tool_calls(response)
        if tool_calls:
            result["tool_calls"] = tool_calls
        elif _TOOL_CALL_OPEN in response or _TOOL_CALL_CLOSE in response:
            mark_model_output_error(result, "malformed <tool_call> JSON")

        if "</think>" in response:
            clean = response.split("</think>", 1)[-1]
        elif "<think>" in _strip_tool_calls(response):
            # An unclosed opener: everything after it is reasoning, and must
            # not fall through into published content. Both the test and the
            # split run on the tool-call-scrubbed text, because a ``type``
            # payload may contain the literal characters ``<think>``.
            head, _, tail = _strip_tool_calls(response).partition("<think>")
            trailing = tail.strip()
            if trailing:
                result["reasoning_content"] = trailing
            clean = head
        else:
            clean = response
        clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL)
        clean = _strip_tool_calls(clean).strip()
        if clean:
            result["content"] = [{"type": "text", "text": clean}]

        return result

# =============================================================================
# ``use`` adapter (intermediate)
# =============================================================================

@dataclasses.dataclass
class Qwen3VLUseAdapter(Qwen3VLBaseAdapter):
    """Intermediate adapter that adds the ``use`` wire format on top of
    :class:`Qwen3VLBaseAdapter`. Concrete ``use`` adapters (desktop,
    browser, mobile) inherit from this class.

    Wire format:
      * 2-part: ``Action: <text>`` + ``<tool_call>{json}</tool_call>``
        (default — ``enable_inline_reasoning=False``).
      * 3-part: ``Thought: <text>`` + ``Action: <text>`` +
        ``<tool_call>{json}</tool_call>`` (``enable_inline_reasoning=True``).

    This adapter is the single *decomposed* use convention: structured
    ``action_description`` (+ optional ``inline_reasoning``) parts ↔ ``Action:``
    / ``Thought:`` wire lines. It does NOT carry an opaque/verbatim mode — a
    prompting convention that keeps the assistant turn whole (e.g. WebGym's
    Memory/Progress block) uses a text-passthrough adapter instead (the base
    adapter, or a passthrough adapter with this family's vision action space),
    selected via ``agent_id`` — never an if-branch here.

    Attributes:
        enable_inline_reasoning: If False (default), use the 2-part wire format
            and DROP any ``inline_reasoning`` parts. If True, surface
            ``inline_reasoning`` parts as ``Thought:`` body lines. The top-level
            ``reasoning_content`` slot is reserved for the NATIVE ``<think>``
            channel and is NEVER coerced into a ``Thought:`` line. If you flip
            ``enable_inline_reasoning=True`` you MUST also pass a ``system_prompt``
            that asks for the Thought line; the default ``USE_SYSTEM_PROMPT``
            only asks for ``Action:``.
            Parallel to :attr:`enable_thinking` (native ``<think>``).
        system_prompt: Defaults to ``USE_SYSTEM_PROMPT``. Pinned here
            because the ``Thought:`` / ``Action:`` wire format is pinned here —
            the prompt is what tells the model to emit the line prefixes that
            the methods below parse and render. Subclasses inherit it; override
            only if you toggle ``enable_inline_reasoning`` or swap the prompt.
    """

    enable_inline_reasoning: bool = False
    system_prompt: str | None = USE_SYSTEM_PROMPT

    def _convert_message_to_agent(
        self,
        message: LiteMessage,
        **kwargs,
    ) -> dict[str, Any]:
        """Render the structured ``action_description`` (+ optional
        ``inline_reasoning``) parts as ``Thought:`` / ``Action:`` wire lines.

        Tool-call routing + deepcopy + role check are handled by
        :meth:`Qwen3VLBaseAdapter._convert_message_to_agent`; we only override
        the assistant ``content`` field.
        """
        result = super()._convert_message_to_agent(message, **kwargs)
        if result.get("role") != "assistant":
            return result
        content = message.get("content") or []
        lines: list[str] = []
        if self.enable_inline_reasoning:
            lines += [
                f"Thought: {p['text']}" for p in content
                if p.get("type") == "inline_reasoning" and p.get("text")
            ]
        lines += [
            f"Action: {p['text']}" for p in content
            if p.get("type") == "action_description" and p.get("text")
        ]
        if lines:
            result["content"] = [{"type": "text", "text": "\n".join(lines)}]
        elif not result.get("tool_calls"):
            # Content-only terminal turns (the model's own "Done." final, or a
            # teacher trajectory's) are the canonical no-tool final path;
            # preserve super()'s plain ``text`` part verbatim instead of turning
            # them into an empty assistant target. Symmetric with
            # :meth:`convert_message_from_agent`, which leaves a no-tool-call
            # turn's prose as ``text`` rather than ``action_description``.
            pass
        else:
            result["content"] = []
        return result

    def convert_message_from_agent(
        self,
        message: AgentMessage,
        **kwargs,
    ) -> LiteMessage:
        """Parse the ``Action:`` line into ``action_description`` (the whole
        pre-tool-call prose when unprefixed — see the comment at the fallback
        for why nothing is trimmed) and, when :attr:`enable_inline_reasoning`,
        the ``Thought:`` line into ``inline_reasoning``.

        ``action_description`` is the "narration accompanying an action"
        channel, so the retag applies ONLY to a turn that actually carries
        ``tool_calls``. A turn with no tool call is either a deliberate
        text-oriented termination or a parse failure; both keep their prose as
        plain ``TextContent`` so
        :func:`lite.core.messages.final.no_tool_call_final_text` can see it
        (a retag here made every genuine text final of every ``@use`` adapter
        look like empty model output to the shared loop). The parse-failure case
        is still separated upstream — the loop checks the transient
        ``model_output_error`` marker BEFORE the text-final check.

        Tool-call routing + deepcopy are handled by
        :meth:`Qwen3VLBaseAdapter.convert_message_from_agent` (super); we only
        re-build the ``content`` field with structured parts.
        """
        result = super().convert_message_from_agent(message, **kwargs)
        if result.get("role") != "assistant":
            return result
        if not result.get("tool_calls"):
            # super() already normalized content to a single plain ``text`` part.
            return result

        raw_text = ""
        for part in result.get("content") or []:
            if part.get("type") == "text" and part.get("text"):
                raw_text = part["text"]
                break

        parts: list[dict[str, Any]] = []
        if raw_text:
            inline_reasoning = ""
            if self.enable_inline_reasoning:
                # Stop at the ``\nAction:`` line that begins the action body, not
                # the first ``\n`` — the inline reasoning is multi-line (matches
                # the to-agent render above, which emits the full body verbatim
                # as ``Thought: <text>``). ``\Z`` (not ``$``) is the fallback so a
                # multi-line body is not clipped at an interior line end.
                m = re.search(r"Thought:\s*(.*?)(?:\n(?=Action:)|\Z)", raw_text, re.DOTALL)
                if m:
                    inline_reasoning = m.group(1).strip()
            # Capture the action body from the LAST ``Action:`` line to
            # end-of-string (symmetric with Thought, so a multi-line
            # action_description round-trips intact instead of being truncated at
            # the first ``\n``). The leading ``.*\n`` is greedy under DOTALL, so it
            # consumes any ``Action:``-prefixed line nested inside the thought
            # body and anchors on the FINAL one — the real action.
            m = re.search(r"(?:.*\n)?Action:\s*(.*)\Z", raw_text, re.DOTALL)
            if m:
                action_text = m.group(1).strip()
            else:
                # No ``Action:`` marker: the WHOLE reply is the action
                # description. Keeping only its first line used to drop the
                # model's reasoning -- 17% of tool-call turns, a median of 232
                # characters each -- and this teacher runs with thinking OFF, so
                # that prose IS its reasoning. Nothing at runtime wanted the
                # truncation either: same-family replay re-renders
                # ``raw_response.text`` verbatim, so it only ever cost the
                # durable record, which is exactly what a cross-family student
                # is trained on.
                action_text = raw_text.strip()
            parts = make_assistant_content(
                inline_reasoning=inline_reasoning, action_description=action_text,
            )
        result["content"] = parts
        return result

# =============================================================================
# Desktop + Browser Adapters
# =============================================================================
# Desktop and browser share one adapter class per task type — same action
# space, protocol, and system prompt. The (desktop|browser) regex on each
# class's key registers the same body under both ``qwen3_vl@desktop@...``
# and ``qwen3_vl@browser@...``. Mobile keeps its own classes because the
# action space and protocol differ.

@dataclasses.dataclass
class Qwen3VLDesktopGroundingActionAdapter(
    Qwen3VLBaseAdapter, key=r"qwen3_vl@(desktop|browser)@grounding\.action"
):
    """Desktop+browser grounding/action: full action vocabulary, single turn.

    Currently NOT routed by any env (env eval moved to ``grounding.point``;
    see :class:`Qwen3VLDesktopGroundingPointAdapter`). Kept available for
    SFT-data replay and offline rendering of historical grounding.action
    parquets, where the family-native ``computer_use`` XML wire format is
    needed (full action enum, smart-resize images, ``<tool_call>`` tags).
    Single-step content is structured tool_calls; the assistant text part
    flows through base's text-flatten path — no ``Action:`` / ``Thought:``
    ``use`` wire format.
    """
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=Qwen3VLDesktopActionSpace
    )
    protocol: FullHistoryProtocol = dataclasses.field(
        default_factory=FullHistoryProtocol
    )
    natively_rendered_extra_tool_names: ClassVar[frozenset[str]] = LiteFinishToolSet.get_tool_names()


@dataclasses.dataclass
class Qwen3VLDesktopGroundingPointAdapter(
    Qwen3VLBaseAdapter, key=r"qwen3_vl@(desktop|browser)@grounding\.point"
):
    """Desktop+browser grounding (single-step click).

    Uses the trimmed :class:`Qwen3VLDesktopGroundingPointActionSpace`
    (``computer_use`` with only ``left_click`` + ``terminate``) plus the
    minimal :data:`GROUNDING_POINT_SYSTEM_PROMPT`. Single turn, full history.
    The agent emits exactly one tool_call; the action_space round-trips
    ``computer_use(left_click, coord)`` to cua-lite ``point(coord)``.
    """
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=Qwen3VLDesktopGroundingPointActionSpace
    )
    metadata: LiteCUAMetadata = dataclasses.field(
        default_factory=lambda: LiteCUAMetadata(
            dims=(
                LiteCUAMetadata.Platform.DESKTOP.value,
                LiteCUAMetadata.TaskType.GROUNDING_POINT.value,
            )
        )
    )
    protocol: FullHistoryProtocol = dataclasses.field(
        default_factory=FullHistoryProtocol
    )
    system_prompt: str | None = GROUNDING_POINT_SYSTEM_PROMPT


@dataclasses.dataclass
class Qwen3VLDesktopUseAdapter(
    Qwen3VLUseAdapter, key=r"qwen3_vl@(desktop|browser)@use"
):
    """Desktop+browser ``use`` (multi-step rollout): summarized history with default action set
    and wire format / system_prompt inherited from
    :class:`Qwen3VLUseAdapter`. One class for both platforms — browser nav
    (goto/back/...) is an env extra_tool, not a per-platform action space.
    The ``computer_use`` desktop tool + canonical ``answer`` serve
    both; env extra_tools pass through the base converter's identity ``else``.
    """
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=Qwen3VLDesktopActionSpace
    )
    protocol: Qwen3VLHistoryProtocol = dataclasses.field(
        default_factory=lambda: Qwen3VLHistoryProtocol(full_history_size=4)
    )
    natively_rendered_extra_tool_names: ClassVar[frozenset[str]] = LiteFinishToolSet.get_tool_names()


# =============================================================================
# Mobile Adapters
# =============================================================================

@dataclasses.dataclass
class Qwen3VLMobileGroundingActionAdapter(Qwen3VLBaseAdapter, key="qwen3_vl@mobile@grounding.action"):
    """Mobile grounding/action: full mobile action vocabulary, single turn.

    Kept for SFT replay (env eval routes to ``grounding.point``).
    See :class:`Qwen3VLDesktopGroundingActionAdapter` for the rationale.
    """
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=Qwen3VLMobileActionSpace
    )
    protocol: FullHistoryProtocol = dataclasses.field(
        default_factory=FullHistoryProtocol
    )
    natively_rendered_extra_tool_names: ClassVar[frozenset[str]] = (
        LiteFinishToolSet.get_tool_names() | LiteAppLaunchToolSet.get_tool_names()
    )


@dataclasses.dataclass
class Qwen3VLMobileGroundingPointAdapter(Qwen3VLBaseAdapter, key="qwen3_vl@mobile@grounding.point"):
    """Mobile grounding (single-step click).

    Uses the same trimmed ``computer_use`` schema + grounding prompt as
    desktop — single-step click is platform-invariant for grounding eval.
    The mobile-specific :class:`Qwen3VLMobileActionSpace` is only relevant
    for use; grounding goes through the dedicated point harness.
    """
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=Qwen3VLMobileGroundingPointActionSpace
    )
    metadata: LiteCUAMetadata = dataclasses.field(
        default_factory=lambda: LiteCUAMetadata(
            dims=(
                LiteCUAMetadata.Platform.MOBILE.value,
                LiteCUAMetadata.TaskType.GROUNDING_POINT.value,
            )
        )
    )
    protocol: FullHistoryProtocol = dataclasses.field(
        default_factory=FullHistoryProtocol
    )
    system_prompt: str | None = GROUNDING_POINT_SYSTEM_PROMPT

@dataclasses.dataclass
class Qwen3VLMobileUseAdapter(Qwen3VLUseAdapter, key="qwen3_vl@mobile@use"):
    """Mobile ``use`` adapter for Qwen3-VL and Qwen3-VL mobile fine-tunes.

    Uses the mobile ``mobile_use`` tool with ``999×999`` declared coordinate
    space (pass-through, no rescale) and the canonical
    :class:`Qwen3VLHistoryProtocol` history-summary layout (same as desktop).
    The cookbook's mobile-specific ``Step N: ...; Step N+1: ...; `` layout is
    intentionally not implemented or registered by default.

    Default ``enable_inline_reasoning=False`` (inherited from
    :class:`Qwen3VLUseAdapter`) → 2-part Action + ``<tool_call>`` wire format,
    matching the desktop adapter. The mobile cookbook advertises the 3-part
    Thought + Action + ``<tool_call>`` format; callers that want that format
    should pass ``enable_inline_reasoning=True`` plus a ``system_prompt`` that
    asks for the ``Thought:`` line.

    Reference:
        ${CUA_LITE_REFERENCES_ROOT}/Qwen3-VL/cookbooks/mobile_agent.ipynb
    """
    # ─── Core ────────────────────────────────────────────────────────
    action_space: BaseActionSpace = dataclasses.field(
        default_factory=Qwen3VLMobileActionSpace
    )
    # Mobile uses the canonical Qwen3VLHistoryProtocol — same as desktop. The
    # cookbook-specific single-bubble "Task progress" layout is intentionally
    # not registered.
    protocol: Qwen3VLHistoryProtocol = dataclasses.field(
        default_factory=lambda: Qwen3VLHistoryProtocol(full_history_size=4)
    )
    # ─── Class-specific knobs ────────────────────────────────────────
    # (system_prompt + enable_inline_reasoning inherited from Qwen3VLUseAdapter)
    # Mobile reference implementations (Qwen3-VL cookbook, MAI-UI)
    # all send raw screenshots without resizing. AndroidWorld emulator
    # screenshots are ~720×1520 (1.1M pixels) — well under the desktop cap, so
    # _smart_resize_image would only round dimensions to 32-px multiples
    # (e.g. 720×1520 → 704×1536), which shifts the model's learned coordinate
    # space by ~16px on each axis. Disabling resize keeps the SFT distribution
    # intact.
    smart_resize_enabled: bool = False
    natively_rendered_extra_tool_names: ClassVar[frozenset[str]] = (
        LiteFinishToolSet.get_tool_names() | LiteAppLaunchToolSet.get_tool_names()
    )

# =============================================================================
# Pass-through Adapters (understanding, bbox, point)
# =============================================================================

# These don't need transformation, register AsIsAdapter for matching patterns.
# Pattern matches: qwen3_vl@desktop@understanding, qwen3_vl@mobile@grounding.bbox, etc.
AgentAdapterRegistry.register(r"qwen3_vl@(desktop|browser|mobile)@understanding", AsIsAdapter)
AgentAdapterRegistry.register(r"qwen3_vl@(desktop|browser|mobile)@grounding\.bbox", AsIsAdapter)
# ``grounding.action`` is now served by the concrete adapter classes
# above — ``Qwen3VLDesktopGroundingActionAdapter`` (registered under
# ``r"qwen3_vl@(desktop|browser)@grounding.action"`` so it covers both
# platforms) and ``Qwen3VLMobileGroundingActionAdapter``. Used for
# SFT-data replay under the family-native XML wire format. Eval-side
# ``grounding.point`` is served by the parallel GroundingPointAdapter
# trimmed-schema chain.
