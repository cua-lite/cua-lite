"""
Step-GUI (GELab-Zero) Adapter

Provides the Step-GUI mobile adapter for the ``use`` task type:

    STEPGUIMobileBaseAdapter (shared logic)
    └── STEPGUIMobileUseAdapter   (mobile, StepGUIHistoryProtocol)

Reference: ${CUA_LITE_REFERENCES_ROOT}/gelab-zero/copilot_tools/parser_0920_summary.py
(Parser0920Summary class).

Notable design points:

1. **No function calling / ``<tool_call>`` JSON.** Unlike MAI-UI and Qwen3VL, the
   gelab-zero reference uses plain ``openai.ChatCompletion.create()`` with NO
   ``tools=`` parameter (ask_llm_v2.py:90-101). The model outputs a
   **tab-separated key:value** string, NOT JSON tool calls. Consequently,
   there is no ``_build_tools_section`` and no ``tools=`` to
   ``apply_chat_template``. The action space is baked into
   ``STEP_GUI_MOBILE_SYS_PROMPT`` as Chinese examples.

2. **``<THINK>`` tags are PLAIN BPE TEXT (uppercase).** The model emits
   ``<THINK>...</THINK>`` as part of its SFT distribution. This differs from
   both MAI-UI's ``<thinking>`` (lowercase with ``ing``) and Qwen3's special
   token ``<think>`` (id 151667). The parser tolerates case/typo variations
   (``<think>``, ``<TINK>``, spacing), matching the reference
   ``Parser0920Summary.str2action()`` (line 264).

3. **Tab-separated action format.** The model output after ``</THINK>`` is
   ``explain:...\taction:CLICK\tpoint:x,y\tsummary:...`` — tab-delimited
   key:value pairs. The adapter parses this into a ``mobile_use`` tool call
   dict for the cua-lite data plane, and serializes back to the same format
   when rendering history. A turn carrying N actions is N such records, one
   per line (the reference ``action2str`` terminates each action with
   ``\n``): **newline is the record separator, tab the field separator**.

4. **Model-generated ``summary`` field.** The model emits a ``summary:`` field
   at each step that serves as compressed history for the next turn. The
   adapter parses it into a ``HistorySummaryContent`` part inside
   ``content``; ``StepGUIHistoryProtocol`` reads it from there when
   building the next prompt.

Usage:
    from lite.agents.models.step_gui.adapter import STEPGUIMobileUseAdapter

    adapter = STEPGUIMobileUseAdapter()
    agent_sample = adapter.unroll(sample)
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import re
from collections import OrderedDict
from typing import Any, ClassVar

from lite.agents.core.action_space import BaseActionSpace
from lite.agents.core.adapter import BaseAgentAdapter
from lite.agents.models.step_gui.action_space import STEPGUIMobileActionSpace
from lite.agents.models.step_gui.protocol import StepGUIHistoryProtocol
from lite.agents.types import AgentMessage, AgentStep
from lite.core import (
    LiteMessage,
    LiteSample,
)
from lite.core.messages import get_inline_reasoning, make_assistant_content
from lite.core.messages.final import MODEL_OUTPUT_ERROR_KEY, mark_model_output_error
from lite.core.messages.turns import truncate_sample_to_turn

logger = logging.getLogger(__name__)


# =============================================================================
# System prompt — based on the gelab-zero reference
# (parser_0920_summary.py:23-52, task_define_prompt), with INFO aligned to
# cua-lite's terminal ``response`` semantics.
# =============================================================================

# DO NOT modify this string without checking it against the upstream
# ${CUA_LITE_REFERENCES_ROOT}/gelab-zero/copilot_tools/parser_0920_summary.py
# — every byte was part of the SFT distribution. That is also why no gate ever
# narrows it: neither the sample's active extra tools nor ``valid_actions``
# may delete a numbered row. Withholding a trained row pushes the model off the
# distribution it was fine-tuned on, and reachability is env ingress's to
# answer — a call to an inactive tool comes back as model-visible feedback
# keyed to its call id.
#
# The one local override is INFO's description: upstream spells it "询问用户问题
# 或详细信息", but here INFO lowers to canonical ``response``.
STEP_GUI_MOBILE_SYS_PROMPT = """你是一个手机 GUI-Agent 操作专家，你需要根据用户下发的任务、手机屏幕截图和交互操作的历史记录，借助既定的动作空间与手机进行交互，从而完成用户的任务。
请牢记，手机屏幕坐标系以左上角为原点，x轴向右，y轴向下，取值范围均为 0-1000。

# 行动原则：

1. 你需要明确记录自己上一次的action，如果是滑动，不能超过5次。
2. 你需要严格遵循用户的指令，如果你和用户进行过对话，需要更遵守最后一轮的指令

# Action Space:

在 Android 手机的场景下，你的动作空间包含以下9类操作，所有输出都必须遵守对应的参数要求：
1. CLICK：点击手机屏幕坐标，需包含点击的坐标位置 point。
例如：action:CLICK\tpoint:x,y
2. TYPE：在手机输入框中输入文字，需包含输入内容 value、输入框的位置 point。
例如：action:TYPE\tvalue:输入内容\tpoint:x,y
3. COMPLETE：任务完成后向用户报告结果，需包含报告的内容 value。
例如：action:COMPLETE\treturn:完成任务后向用户报告的内容
4. WAIT：等待指定时长，需包含等待时间 value（秒）。
例如：action:WAIT\tvalue:等待时间
5. AWAKE：唤醒指定应用，需包含唤醒的应用名称 value。
例如：action:AWAKE\tvalue:应用名称
6. INFO：向用户提交最终回答，需包含回答内容 value。
例如：action:INFO\tvalue:最终回答内容
7. ABORT：终止当前任务，仅在当前任务无法继续执行时使用，需包含 value 说明原因。
例如：action:ABORT\tvalue:终止任务的原因
8. SLIDE：在手机屏幕上滑动，滑动的方向不限，需包含起点 point1 和终点 point2。
例如：action:SLIDE\tpoint1:x1,y1\tpoint2:x2,y2
9. LONGPRESS：长按手机屏幕坐标，需包含长按的坐标位置 point。
例如：action:LONGPRESS\tpoint:x,y
"""


# =============================================================================
# Step-GUI Mobile Base Adapter
# =============================================================================

@dataclasses.dataclass
class STEPGUIMobileBaseAdapter(BaseAgentAdapter):
    """
    Base adapter for Step-GUI (GELab-Zero) mobile.

    Responsibilities:
      - Build a system message containing ``STEP_GUI_MOBILE_SYS_PROMPT``.
        No ``_build_tools_section`` — the model was SFT'd with the action space
        baked into the system prompt as Chinese examples, not via function-calling
        schema injection. The reference ``ask_llm_v2.py`` calls
        ``ChatCompletion.create()`` with no ``tools=`` parameter.
      - Apply the protocol (default: StepGUIHistoryProtocol).
      - Translate cua-lite mobile tool_calls <-> Step-GUI ``mobile_use`` shape
        via ``STEPGUIMobileActionSpace``.
      - Parse raw model output (``<THINK>`` + tab-separated key:value) into a
        cua-lite assistant message: ``<THINK>`` text becomes an
        ``InlineReasoningContent`` part (non-native CoT), ``explain:`` becomes
        ``ActionDescriptionContent``, ``summary:`` becomes
        ``HistorySummaryContent``, and the action fields become a
        ``mobile_use`` tool_call.
    """

    system_prompt: str | None = None

    # -- sample <-> agent ----------------------------------------------------

    def render_step(
        self,
        sample: LiteSample,
        k: int,
        processed,
        **kwargs,
    ) -> AgentStep:
        """Render turn ``k`` in Step-GUI format.

        Steps:
          1. Truncate sample to turn ``k``.
          2. Apply protocol (history windowing / summarization).
          3. Prepend the Step-GUI system prompt, whole — it is SFT text and no
             gate narrows it (see ``STEP_GUI_MOBILE_SYS_PROMPT``).
          4. Translate each remaining message via convert_message_to_agent.
        """
        truncated = truncate_sample_to_turn(sample, k)
        messages = self.protocol.process_messages(truncated.messages)

        result_messages: list[AgentMessage] = []
        if self.system_prompt:
            result_messages.append({
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            })

        for msg in messages:
            result_messages.append(self.convert_message_to_agent(msg))

        return result_messages

    # -- per-message conversion ----------------------------------------------

    def _convert_message_to_agent(
        self,
        message: LiteMessage,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert a single cua-lite message to Step-GUI's wire format.

        For assistants: fold reasoning + content + tool_calls into the
        SFT-trained ``<THINK>...</THINK>\\nexplain:...\\taction:...\\t...``
        plain text and drop the structured ``tool_calls`` / ``reasoning_content``
        fields. The Qwen2-VL chat_template would otherwise (a) silently drop
        ``reasoning_content``, and (b) render ``tool_calls`` as Qwen JSON
        ``<tool_call>{json}</tool_call>`` which is off-distribution for StepGUI.
        Other roles pass through (deep-copied).

        Selective extract: reads the ORIGINAL message's content parts, picks
        ``inline_reasoning`` / ``action_description`` / ``history_summary``
        kinds, and rebuilds a fresh single-text content list.
        """
        result = copy.deepcopy(message)
        if result.get("role") != "assistant":
            return result

        thinking = get_inline_reasoning(message)
        tool_calls = self.action_space.convert_tool_calls_to_agent(
            message.get("tool_calls") or []
        )
        explain_parts: list[str] = []
        summary_parts: list[str] = []
        for part in message.get("content") or []:
            kind = part["type"]
            if kind == "action_description" and part["text"]:
                explain_parts.append(part["text"])
            elif kind == "history_summary" and part["text"]:
                summary_parts.append(part["text"])
        explain = "".join(explain_parts)
        summary = "".join(summary_parts)

        # Build the wire-format text. ``format_agent_tool_call_as_wire_text``
        # injects ``summary`` in the canonical ``action:X\tsummary:S\t<fields>``
        # slot (matches SFT distribution byte-for-byte); attach to first
        # tool_call only so multi-call messages don't repeat it.
        blocks: list[str] = []
        if thinking:
            blocks.append(f"<THINK> {thinking} </THINK>")
        for i, tc in enumerate(tool_calls):
            tc_text = self.format_agent_tool_call_as_wire_text(
                tc, summary=summary if i == 0 else "",
            )
            if i == 0 and explain:
                tc_text = f"explain:{explain}\t{tc_text}"
            blocks.append(tc_text)
        if not tool_calls and explain:
            blocks.append(explain)

        had_tool_calls = bool(message.get("tool_calls"))
        full_text = "\n".join(blocks)
        result.pop("reasoning_content", None)
        result.pop("tool_calls", None)
        if full_text:
            result["content"] = [{"type": "text", "text": full_text}]
        elif not had_tool_calls:
            # Content-only final turn: no action to narrate. Keep ONLY the plain
            # ``text`` parts -- the canonical "Done." this policy writes -- so the
            # turn does not become an empty SFT target. Non-``text`` parts still
            # drop: this format has no prose slot for them, and the data layer
            # guarantees a no-tool-call final carries exactly one ``text`` part.
            result["content"] = [
                p for p in (message.get("content") or [])
                if isinstance(p, dict) and p.get("type") == "text" and p.get("text")
            ]
        else:
            result["content"] = []
        return result

    def convert_message_from_agent(
        self,
        message: AgentMessage,
        **kwargs,
    ) -> LiteMessage:
        """Convert AgentMessage → LiteMessage. Step-GUI's entire wire
        format is SFT/system-prompt-defined (``<THINK>...</THINK>`` +
        tab-separated ``explain:/action:/point:/summary:/...`` KV line).
        There are no chat-template tokens, so all parsing lives here.

        Mirrors ``Parser0920Summary.str2action()``
        (parser_0920_summary.py:255-313). Tolerances (matching the
        reference parser):
          * Normalizes ``<TINK>`` / ``<think>`` / ``< THINK >`` variants.
          * Handles missing ``<THINK>`` tags (entire response treated as kv).
          * Parses point coordinates in both ``x,y`` and ``x y`` format.
        """
        result = copy.deepcopy(message)
        if result.get("role") != "assistant":
            return result

        raw_text = ""
        for part in message.get("content") or []:
            if part.get("type") == "text" and part.get("text"):
                raw_text = part["text"]
                break

        if not raw_text:
            if "tool_calls" in result:
                result["tool_calls"] = (
                    self._route_agent_tool_calls_to_lite(result["tool_calls"])
                )
            return result

        text = raw_text.strip()
        text = (
            text
            .replace("<TINK>", "<THINK>").replace("</TINK>", "</THINK>")
            .replace("<think>", "<THINK>").replace("</think>", "</THINK>")
        )
        text = re.sub(
            r"<\s*/?THINK\s*>",
            lambda m: "<THINK>" if "/" not in m.group() else "</THINK>",
            text,
            flags=re.IGNORECASE,
        )
        try:
            cot_part = text.split("<THINK>")[1].split("</THINK>")[0].strip()
            kv_part = text.split("</THINK>")[1].strip()
        except IndexError:
            logger.warning("Missing <THINK> tags, treating entire response as kv")
            kv_part = text
            cot_part = ""

        records = self._parse_kv_records(kv_part)
        if self._records_are_doneish_final(records):
            tool_calls = []
        else:
            agent_tool_calls = [
                tc for rec in records
                if (tc := self._action_dict_to_tool_call(rec)) is not None
            ]
            tool_calls = (
                self._route_agent_tool_calls_to_lite(agent_tool_calls)
                if agent_tool_calls else []
            )
        if tool_calls:
            result["tool_calls"] = tool_calls
        else:
            result.pop("tool_calls", None)
            if self._ACTION_MARKER in kv_part and not self._records_are_doneish_final(records):
                # The model reached for the action grammar but nothing usable
                # came out -- signal a terminal parse-failure final instead of
                # letting the shared loop read this as a clean content-only final.
                mark_model_output_error(result, "malformed action:... record")

        if not result.get("tool_calls") and MODEL_OUTPUT_ERROR_KEY not in result:
            result["content"] = [{"type": "text", "text": raw_text}]
            return result

        # ``explain`` / ``summary`` are emitted on the FIRST record only
        # (see ``_convert_message_to_agent``); accept them from whichever
        # record carries them so a scrambled turn still round-trips.
        result["content"] = make_assistant_content(
            inline_reasoning=cot_part,
            action_description=next(
                (r["explain"] for r in records if r.get("explain")), ""
            ),
            history_summary=next(
                (r["summary"] for r in records if r.get("summary")), ""
            ),
        )
        return result

    # -- raw response parser -------------------------------------------------

    def parse_raw_assistant_response(
        self,
        response: str,
        **kwargs,
    ) -> AgentMessage:
        """Wrap raw Step-GUI output verbatim as an ``AgentMessage``.
        Step-GUI has no chat-template tokens — ``<THINK>`` and the
        tab-separated KV line are both system-prompt-defined — so all
        parsing happens in :meth:`convert_message_from_agent`.
        """
        return {"role": "assistant", "content": [{"type": "text", "text": response}]}

    # -- internal parsing helpers -------------------------------------------

    # Grammar marker: the model attempted an action record. Used to tell a
    # malformed action from legitimate prose (which stays a terminal final).
    _ACTION_MARKER: ClassVar[str] = "action:"
    _DONEISH_FINAL_ACTIONS: ClassVar[frozenset[str]] = frozenset({
        "DONE",
        "FINISH",
        "FINISHED",
    })

    @classmethod
    def _parse_kv_records(cls, kv_part: str) -> list[dict[str, Any]]:
        """Split the post-``</THINK>`` blob into one action dict per record.

        Records are **newline**-delimited, fields **tab**-delimited: the
        reference ``Parser0920Summary.action2str`` emits
        ``<THINK> cot </THINK>\\n`` + ``"\\t".join(kvs)`` + ``"\\n"`` per action
        (parser_0920_summary.py:245), and ``_convert_message_to_agent`` joins
        one such record per tool call with ``"\\n"`` (see ``full_text`` above).
        Blank lines and trailing whitespace are ignored.
        """
        records: list[dict[str, Any]] = []
        for line in kv_part.splitlines():
            line = line.strip()
            if not line:
                continue
            if action := cls._parse_kv_to_action(line):
                records.append(action)
        return records

    @staticmethod
    def _parse_kv_to_action(kv_part: str) -> dict[str, Any]:
        """Parse ONE tab-separated key:value record into an action dict.

        Mirrors ``Parser0920Summary.str2action()`` (lines 276-313).
        """
        action: dict[str, Any] = OrderedDict()

        kvs = [kv.strip() for kv in kv_part.split("\t") if kv.strip()]

        for kv in kvs:
            if ":" not in kv:
                continue

            key = kv.split(":", 1)[0].strip()
            value = kv.split(":", 1)[1].strip()

            if key == "action":
                action["action"] = value
            elif key == "summary":
                action["summary"] = value
            elif key == "return":
                action["return"] = value
            elif "point" in key:
                # Parse point format: "x,y" or "x y"
                try:
                    coords = value.replace(",", " ").split()
                    if len(coords) < 2:
                        logger.warning("Expected 2 coordinates for %s, got: %s", key, value)
                        continue
                    x, y = int(coords[0]), int(coords[1])
                    action[key] = [x, y]
                except (ValueError, IndexError):
                    logger.warning("Failed to parse point '%s' for key '%s'", value, key)
                    continue
            else:
                action[key] = value

        return action

    @classmethod
    def _records_are_doneish_final(cls, records: list[dict[str, Any]]) -> bool:
        action_records = [r for r in records if "action" in r]
        if not action_records:
            return False
        allowed_keys = {"action", "explain", "summary"}
        for record in action_records:
            action = str(record.get("action") or "").strip().upper().rstrip(".。")
            if action not in cls._DONEISH_FINAL_ACTIONS:
                return False
            if set(record) - allowed_keys:
                return False
        return True

    @staticmethod
    def _action_dict_to_tool_call(action_dict: dict[str, Any]) -> dict[str, Any] | None:
        """Convert a parsed action dict to a mobile_use tool call.

        Maps reference field names to the ``STEPGUIMobileActionSpace``
        ``mobile_use`` parameter names:
          - action → action
          - point / point1 / point2 → same
          - value → value
          - return → return_value

        ``summary`` is intentionally NOT carried in ``tool_call.arguments`` —
        it is trajectory metadata (rolling history summary), not an action
        parameter. The parser stores it in ``HistorySummaryContent`` inside
        ``content`` instead.
        """
        action_type = action_dict.get("action")
        if not action_type:
            return None

        # Normalize action type to uppercase (model may emit mixed case)
        action_type = action_type.upper()

        args: dict[str, Any] = {"action": action_type}

        if "value" in action_dict:
            args["value"] = action_dict["value"]
        if "point" in action_dict:
            args["point"] = action_dict["point"]
        if "point1" in action_dict:
            args["point1"] = action_dict["point1"]
        if "point2" in action_dict:
            args["point2"] = action_dict["point2"]
        if "return" in action_dict:
            args["return_value"] = action_dict["return"]

        return {"name": "mobile_use", "arguments": args}

    # -- text format rendering (for history) ---------------------------------

    # Per-action field emission allowlist, matching reference
    # ``Parser0920Summary.action2action`` (parser_0920_summary.py:107-213).
    # The base OrderedDict carries ``{cot, explain, action, summary}``; each
    # branch appends a subset of action-specific keys. ``action2str`` then
    # iterates the dict (skipping ``cot``), which is how we know the on-wire
    # order the model was SFT'd to emit: ``explain, action, summary, <fields>``.
    # cua-lite handles ``explain`` / THINK in ``_convert_message_to_agent``, so
    # the renderer only emits the args-side portion: ``action, summary, <fields>``.
    _EMIT_FIELDS: ClassVar[dict[str, tuple[str, ...]]] = {
        "CLICK":     ("point",),
        "LONGPRESS": ("point",),
        "TYPE":      ("value",),          # ref drops point (parser_0920_summary.py:142)
        "SLIDE":     ("point1", "point2"),
        "AWAKE":     ("value",),
        "INFO":      ("value",),
        "WAIT":      ("value",),
        "COMPLETE":  ("return_value",),   # rendered as "return:..."
        "ABORT":     (),                  # ref drops value (parser_0920_summary.py:188-189)
    }

    @staticmethod
    def _render_value(value: Any) -> str:
        if isinstance(value, list):
            return ",".join(str(v) for v in value)
        if isinstance(value, (bool, int, float)):
            return str(value)
        return str(value).replace("\n", "").replace("\t", "").strip()

    def format_agent_tool_call_as_wire_text(
        self,
        agent_tool_call: dict[str, Any],
        summary: str = "",
    ) -> str:
        """Render one Step-GUI agent projection as the tab-separated wire text.

        Input shape is part of the contract: this takes the family's bare
        ``{"name": "mobile_use", "arguments": {...}}`` projection — what
        ``self.action_space.convert_tool_calls_to_agent`` returns — never a
        canonical Lite call, which raises ``KeyError`` here. The renderer lives
        on the adapter because this text is the SFT wire format the adapter
        folds assistant turns into (``convert_message_to_agent``); the name is
        deliberately NOT the action space's ``format_tool_call_as_text``, which
        stays canonical-shaped and which Step-GUI's action space does not
        override.

        Emits fields in the reference ``action2str`` order (verified against
        ``Parser0920Summary.action2str``):

            action:XXX \t [summary:...] \t <per-action allowlist>

        ``summary`` is **not** stored in ``args`` (it is trajectory metadata,
        not an action parameter; the parser keeps it as
        ``HistorySummaryContent`` in the message). Pass it explicitly when
        re-rendering so it lands in the canonical position right after
        ``action:`` — ``_convert_message_to_agent`` does this when folding
        an assistant message back into the SFT wire format.
        """
        args = agent_tool_call["arguments"]
        action_type = (args.get("action") or "").upper()

        parts: list[str] = []
        if "action" in args:
            parts.append(f"action:{args['action']}")
        if summary:
            parts.append(f"summary:{self._render_value(summary)}")

        if action_type in self._EMIT_FIELDS:
            for key in self._EMIT_FIELDS[action_type]:
                if key in args:
                    wire_key = "return" if key == "return_value" else key
                    parts.append(f"{wire_key}:{self._render_value(args[key])}")
        else:
            for key, value in args.items():
                if key == "action":
                    continue
                wire_key = "return" if key == "return_value" else key
                parts.append(f"{wire_key}:{self._render_value(value)}")

        return "\t".join(parts)


# =============================================================================
# ``use`` adapter
# =============================================================================

@dataclasses.dataclass
class STEPGUIMobileUseAdapter(
    STEPGUIMobileBaseAdapter, key="step_gui@mobile@use",
):
    """Step-GUI mobile, ``use`` mode (multi-step rollout).

    Default protocol: :class:`StepGUIHistoryProtocol` — matches the reference
    ``make_status_prompt`` / ``env2messages4ask`` Chinese prompt format
    byte-for-byte: system prompt contains the action space (Chinese), and
    each turn's user message is rebuilt as

        已知用户指令为：{task}指令结束
        已知已经执行过的历史动作如下：{rolling_summary}
        当前手机屏幕截图如下：
        <image>
        ...<Chinese format instructions including <THINK> example>

    The rolling ``summary`` field from the previous assistant response acts
    as compressed history — no per-step action list, no image window.
    """

    action_space: BaseActionSpace = dataclasses.field(
        default_factory=STEPGUIMobileActionSpace
    )
    protocol: StepGUIHistoryProtocol = dataclasses.field(
        default_factory=StepGUIHistoryProtocol
    )
    system_prompt: str | None = STEP_GUI_MOBILE_SYS_PROMPT


__all__ = [
    "STEP_GUI_MOBILE_SYS_PROMPT",
    "STEPGUIMobileBaseAdapter",
    "STEPGUIMobileUseAdapter",
]
