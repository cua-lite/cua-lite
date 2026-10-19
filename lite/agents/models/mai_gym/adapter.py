"""
MAI-Gym Adapters

``MAIGymMobileUseAdapter`` (mobile ``use``) reproduces
``MAIGymMobileUseWindowsWisePolicy`` and ``MAIGymDesktopUseAdapter`` (desktop
``use``) reproduces ``MAIGymComputerUsePolicy`` with its window option — the
harnesses MAI-Gym-9B was trained and evaluated with. The desktop wire is
described on :class:`MAIGymDesktopUseAdapter`; the mobile wire below.

Reference: ``MAI-Gym/src/mai_gym/agents/implementations/mai_gym_mobile_use*.py``,
``MAI-Gym/src/mai_gym/agents/parsing/mai_gym_mobile_use.py`` and
``MAI-Gym/src/mai_gym/agents/prompts/mai_gym_mobile_use_windows_wise_system.j2``.

Wire format:

1. **Native ``<think>``.** The checkpoint is a Qwen3.5-9B fine-tune with the
   stock Qwen3.5 chat template, so the generation prompt opens ``<think>\\n``
   (``enable_thinking=True``) and the reasoning lands in top-level
   ``reasoning_content``. The agent renders with ``preserve_thinking=True``, so
   history turns keep their reasoning, matching the paper's
   ``token_native_history`` (sampled tokens replayed verbatim).
2. **JSON ``<tool_call>`` body**, ``{"name": "mobile_use", "arguments":
   {"action_type": ...}}``, prompted by the system text only: no ``tools=`` is
   passed to the chat template. Like upstream ``parse_action_response`` only
   the FIRST ``<tool_call>`` block counts, and a reply without a parseable
   block is malformed output that takes the shared parse-failure final.
3. **System prompt** is the window-wise template byte-for-byte, with one
   deliberate substitution: the "Available Apps" list is the env's own
   ``open_app`` enum rather than upstream's per-platform list, because env
   ingress validates ``open_app`` against that enum (upstream's ``files`` /
   ``railway12306`` would be rejected). With no ``open_app`` enum both app lines
   are omitted. Non-native active extras render through the template's
   ``## MCP Tools`` slot.
4. **Images** are sent at native resolution (upstream has no client resize;
   coordinates are relative [0, 999], so the serving processor's own rounding
   cannot shift them).

Usage:
    from lite.agents.models.mai_gym.adapter import MAIGymMobileUseAdapter

    adapter = MAIGymMobileUseAdapter()
    agent_sample = adapter.unroll(sample)
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import json
import re
from typing import Any, ClassVar

from lite.agents.core.action_space import BaseActionSpace
from lite.agents.core.adapter import BaseAgentAdapter
from lite.agents.models.mai_gym.action_space import (
    COMPUTER_ASK_USER_TOOL_NAME,
    COMPUTER_TERMINATE_TOOL_NAME,
    COMPUTER_WAIT_TOOL_NAME,
    MAI_GYM_NATIVE_TOOL_NAME,
    PYAUTOGUI_TOOL_NAME,
    SAFE_PYAUTOGUI_FUNCS,
    MAIGymDesktopActionSpace,
    MAIGymMobileActionSpace,
    render_desktop_block,
    validate_pyautogui_code,
)
from lite.agents.models.mai_gym.protocol import (
    MAIGymDesktopHistoryProtocol,
    MAIGymHistoryProtocol,
)
from lite.agents.types import AgentMessage, AgentStep
from lite.core import LiteMessage, LiteSample
from lite.core.messages.final import mark_model_output_error
from lite.core.messages.turns import truncate_sample_to_turn
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.extra_tools import (
    BASH_TOOL_NAME,
    LiteAppLaunchToolSet,
    LiteFinishToolSet,
    open_app_names_from_metadata,
)
from lite.core.tools.schemas import tool_schema_name

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_THINK_CLOSE = "</think>"


# =============================================================================
# System prompt — mai_gym_mobile_use_windows_wise_system.j2
# =============================================================================
#
# The template splits into the head (through the Action Space block), the
# optional ``{% if tools %}`` MCP block and the ``## Note`` footer. Upstream
# renders the apps with ``json.dumps(apps, ensure_ascii=False)`` and MCP tools
# with ``json.dumps(tools, ensure_ascii=False)``, so this module does the same.

_MAI_GYM_MOBILE_SYS_PROMPT_HEAD = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
For each function call, return the thinking process in <think> </think> tags, and a json object with function name and arguments within <tool_call></tool_call> XML tags:
```
<think>
...
</think>
<tool_call>
{"name": "mobile_use", "arguments": <args-json-object>}
</tool_call>
```

## Action Space
{"action_type": "click", "coordinate": [x, y]}
{"action_type": "double_click", "coordinate": [x, y]}
{"action_type": "long_press", "coordinate": [x, y]}
{"action_type": "type", "text": ""}
{"action_type": "open", "text": "app_name"} # app_name must be chosen from the available apps listed in the Note section.
{"action_type": "drag", "start_coordinate": [x1, y1], "end_coordinate": [x2, y2]}
{"action_type": "system_button", "button": "button_name"} # Options: back, home, menu, enter
{"action_type": "wait", "waiting_time": "xxx"} # waiting_time denotes the wait time in seconds.
{"action_type": "ask_user", "text": "xxx"} # you can ask user for more information to complete the task.
{"action_type": "terminate", "answer": "xxx", "status": "success or failed"}

"""

_MAI_GYM_MOBILE_SYS_PROMPT_MCP = """
## MCP Tools
You are also provided with MCP tools, you can use them to complete the task.
{tools_json}

If you want to use MCP tools, you must output as the following format:
```
<think>
...
</think>
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
```
"""

_MAI_GYM_MOBILE_SYS_PROMPT_APPS_LINES = (
    "- Available Apps: `{apps_json}`.\n"
    "- You should use the `open` action to open the app as possible as you can,"
    " because it is the fast way to open the app.\n"
)

_MAI_GYM_MOBILE_SYS_PROMPT_FOOTER = (
    "- You must follow the Action Space strictly, and return the correct json"
    " object within <think> </think> and <tool_call></tool_call> XML tags."
)


def render_mai_gym_mobile_sys_prompt(
    apps: list[str] | None,
    mcp_tool_schemas: list[dict[str, Any]] | None = None,
) -> str:
    """Render the window-wise system prompt for ``apps`` and MCP extras.

    Byte-equal to upstream ``render_system_prompt(apps, tools=...)`` whenever
    ``apps`` is non-empty; with no apps the two "Available Apps" lines drop.
    """
    parts = [_MAI_GYM_MOBILE_SYS_PROMPT_HEAD]
    if mcp_tool_schemas:
        parts.append(_MAI_GYM_MOBILE_SYS_PROMPT_MCP.format(
            tools_json=json.dumps(mcp_tool_schemas, ensure_ascii=False)
        ))
    parts.append("\n## Note\n")
    if apps:
        parts.append(_MAI_GYM_MOBILE_SYS_PROMPT_APPS_LINES.format(
            apps_json=json.dumps(list(apps), ensure_ascii=False)
        ))
    parts.append(_MAI_GYM_MOBILE_SYS_PROMPT_FOOTER)
    return "".join(parts)


# =============================================================================
# Shared adapter base
# =============================================================================


@dataclasses.dataclass
class MAIGymBaseAdapter(BaseAgentAdapter):
    """Shared MAI-Gym fields and the system-prompt-then-protocol render.

    ``system_prompt`` defaults to ``None`` so each platform's ``__post_init__``
    renders it from the env metadata; a yaml ``agent_kwargs.system_prompt``
    pins it instead.
    """

    system_prompt: str | None = None
    # Forwarded to ``apply_chat_template`` by ``MAIGymBaseAgent``.
    # MAI-Gym-9B always reasons before acting.
    enable_thinking: bool = True

    def render_step(
        self,
        sample: LiteSample,
        k: int,
        processed,
        **kwargs,
    ) -> AgentStep:
        """Render turn ``k``: system prompt, then the protocol's upstream layout."""
        truncated = truncate_sample_to_turn(sample, k)
        messages = self.protocol.process_messages(truncated.messages)
        result_messages: list[AgentMessage] = [{
            "role": "system",
            "content": [{"type": "text", "text": self.system_prompt}],
        }]
        for msg in messages:
            result_messages.append(self.convert_message_to_agent(msg))
        return result_messages

    def convert_message_from_agent(
        self,
        message: AgentMessage,
        **kwargs,
    ) -> LiteMessage:
        """AgentMessage -> LiteMessage; tool calls via the shared extra-tool router."""
        result = copy.deepcopy(message)
        if result.get("role") != "assistant" or "tool_calls" not in result:
            return result
        converted = self._route_agent_tool_calls_to_lite(result["tool_calls"])
        if result["tool_calls"] and not converted:
            mark_model_output_error(result, "tool call did not convert to a canonical Lite call")
        result["tool_calls"] = converted
        return result


# =============================================================================
# Mobile use adapter
# =============================================================================


@dataclasses.dataclass
class MAIGymMobileUseAdapter(MAIGymBaseAdapter, key="mai_gym@mobile@use"):
    """Adapter for MAI-Gym-9B mobile ``use`` (multi-step rollout).

    Default protocol: ``MAIGymHistoryProtocol(window_size=10, overlap=2)`` —
    the upstream window-wise harness. The default system prompt lists the
    env's ``open_app`` enum as the available apps.
    """

    action_space: BaseActionSpace = dataclasses.field(
        default_factory=MAIGymMobileActionSpace
    )
    protocol: MAIGymHistoryProtocol = dataclasses.field(
        default_factory=MAIGymHistoryProtocol
    )
    # Extras the Action Space block already spells natively (``open`` /
    # ``ask_user`` / ``terminate``); they never repeat in the MCP block.
    natively_rendered_extra_tool_names: ClassVar[frozenset[str]] = (
        LiteAppLaunchToolSet.get_tool_names()
        | LiteFinishToolSet.get_tool_names()
        | frozenset({"ask_user"})
    )

    def __post_init__(self):
        super().__post_init__()
        if self.system_prompt is None:
            self.system_prompt = render_mai_gym_mobile_sys_prompt(
                open_app_names_from_metadata(self.metadata),
                self._mcp_extra_tool_schemas(),
            )

    def _mcp_extra_tool_names(self) -> frozenset[str]:
        return self.active_extra_tool_names() - self.natively_rendered_extra_tool_names

    def _mcp_extra_tool_schemas(self) -> list[dict[str, Any]]:
        mcp_names = self._mcp_extra_tool_names()
        return [
            schema for schema in self.metadata.extra_tool_schemas
            if tool_schema_name(schema) in mcp_names
        ]

    # -- sample <-> agent ----------------------------------------------------

    def _convert_message_to_agent(
        self,
        message: LiteMessage,
        **kwargs,
    ) -> dict[str, Any]:
        """Render an assistant turn as text + inline JSON ``<tool_call>`` blocks.

        Top-level ``reasoning_content`` stays in place; the chat template
        rebuilds the ``<think>`` block from it under ``preserve_thinking``.
        Tool calls render inline (not through the template's ``tool_calls``
        branch, which would emit Qwen3.5 XML) with upstream's
        ``json.dumps`` defaults.
        """
        result = copy.deepcopy(message)
        if result.get("role") != "assistant":
            return result

        mcp_names = self._mcp_extra_tool_names()
        agent_calls: list[dict[str, Any]] = []
        for tc in result.pop("tool_calls", None) or []:
            if self._admits_active_extra_tool_call(tc, allowed_names=mcp_names):
                agent_calls.append(
                    {"name": tool_call_name(tc), "arguments": tool_call_arguments(tc)}
                )
            else:
                agent_calls.extend(self.action_space.convert_tool_calls_to_agent([tc]))

        lines = [
            p["text"] for p in message.get("content") or []
            if isinstance(p, dict) and p.get("type") == "text" and p.get("text")
        ]
        lines += [
            "<tool_call>\n"
            + json.dumps({"name": tc["name"], "arguments": tc["arguments"]}, ensure_ascii=False)
            + "\n</tool_call>"
            for tc in agent_calls
        ]
        result["content"] = [{"type": "text", "text": "\n".join(lines)}] if lines else []
        return result

    def parse_raw_assistant_response(
        self,
        response: str,
        **kwargs,
    ) -> AgentMessage:
        """Split native reasoning, the first ``<tool_call>`` JSON, and prose.

        The generation prompt already opened ``<think>``, so the reply is
        ``reasoning </think> answer``; a leading ``<think>`` is tolerated.
        """
        result: AgentMessage = {"role": "assistant"}
        answer = response
        if _THINK_CLOSE in response:
            reasoning, answer = response.split(_THINK_CLOSE, 1)
            reasoning = reasoning.split("<think>", 1)[-1].strip()
            if reasoning:
                result["reasoning_content"] = reasoning

        match = _TOOL_CALL_RE.search(answer)
        body = None
        if match is not None:
            try:
                body = json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                body = None
        if isinstance(body, dict) and isinstance(body.get("arguments"), dict):
            name = body.get("name")
            result["tool_calls"] = [{
                "name": name if isinstance(name, str) and name else MAI_GYM_NATIVE_TOOL_NAME,
                "arguments": body["arguments"],
            }]
        elif match is not None:
            mark_model_output_error(result, "malformed <tool_call> JSON")
        else:
            mark_model_output_error(result, "No <tool_call> block found")

        clean = _TOOL_CALL_RE.sub("", answer).strip()
        if clean:
            result["content"] = [{"type": "text", "text": clean}]
        return result


# =============================================================================
# Desktop system prompt — mai_gym_computer_use_system.j2
# =============================================================================
#
# Rendered as upstream ``_render_system_prompt`` with ``range_scale=1000`` and
# ``enable_ask_user=False`` (both ask_user placeholders render empty, leaving
# their blank lines). When the sample offers no ``bash`` tool every Bash
# passage is cut — the sole deviation, because telling the model to prefer a
# tool the env rejects only burns steps. Each cut is an exact substring.

_MAI_GYM_DESKTOP_SYS_PROMPT = """You are the MAI-Gym computer-use agent, capable of GUI operations and command-line interactions. You are given a task instruction, the current screenshot, and, when applicable, output from the previous Bash command. Complete the task through a sequence of actions. The computer password is {password}.

## GUI actions

Use direct `pyautogui` calls for graphical applications. Supported functions include:

- `pyautogui.click(x, y)`
- `pyautogui.rightClick(x, y)`
- `pyautogui.doubleClick(x, y)`
- `pyautogui.tripleClick(x, y)`
- `pyautogui.moveTo(x, y)`
- `pyautogui.dragTo(x, y, duration=0.5)`
- `pyautogui.typewrite(text)`
- `pyautogui.press(key)`
- `pyautogui.keyDown(key)` / `pyautogui.keyUp(key)`
- `pyautogui.hotkey(key1, key2, ...)`
- `pyautogui.scroll(clicks, x, y)`

Coordinates must be in the range `[0, 1000]` on both axes. They are mapped to the actual screen. `(500, 500)` is the screen center.

## Bash actions

Put the raw shell script directly inside `<bash>...</bash>`. Everything between the tags is passed verbatim:

```text
<bash>
python3 - <<'PY'
print("quotes are preserved verbatim")
PY
</bash>
```

Never type shell commands into a terminal with `pyautogui.typewrite`. Do not wrap Bash in `<tool_call>`, and do not call `computer.bash(...)`.

## Flow-control actions

- `computer.wait()` pauses execution.
- `computer.terminate(status="success")` completes the task successfully.
- `computer.terminate(status="failure")` reports failure.


## Response format

Emit one or more actions. Each GUI or flow-control action uses its own block:

```text
<think>brief reasoning</think>
<tool_call>pyautogui.click(835, 420)</tool_call>
```

Bash is the only exception and uses its own raw `<bash>...</bash>` block.

The following flow-control calls are available:

- `computer.wait()`
- `computer.terminate(status="success"|"failure", answer="optional answer")`


Actions execute from top to bottom. Adjacent GUI actions may be emitted together. A later action in the same response cannot observe the output of an earlier Bash action; when later work depends on Bash output, emit the Bash action alone and inspect its output on the next step.

## Operating rules

- Prefer Bash when it is safer and more direct than GUI interaction.
- Read exact values from the screen; do not guess.
- Preserve the task's requested operation, target, value, and output format.
- Save or confirm GUI edits before terminating.
- Do not declare failure after one attempt. Try reasonable alternatives and wait for slow applications.
- For multi-application tasks, finish and save one application's work before switching.
- Emit only supported actions and never place draft actions inside the reasoning section."""

_NO_BASH_CUTS: tuple[tuple[str, str], ...] = (
    (
        " capable of GUI operations and command-line interactions. You are given a task"
        " instruction, the current screenshot, and, when applicable, output from the"
        " previous Bash command.",
        " capable of GUI operations. You are given a task instruction and the current"
        " screenshot.",
    ),
    (
        _MAI_GYM_DESKTOP_SYS_PROMPT[
            _MAI_GYM_DESKTOP_SYS_PROMPT.index("## Bash actions"):
            _MAI_GYM_DESKTOP_SYS_PROMPT.index("## Flow-control actions")
        ],
        "",
    ),
    ("Bash is the only exception and uses its own raw `<bash>...</bash>` block.\n\n", ""),
    (
        " A later action in the same response cannot observe the output of an earlier"
        " Bash action; when later work depends on Bash output, emit the Bash action"
        " alone and inspect its output on the next step.",
        "",
    ),
    ("- Prefer Bash when it is safer and more direct than GUI interaction.\n", ""),
)


def render_mai_gym_desktop_sys_prompt(password: str, *, bash: bool) -> str:
    """Upstream computer-use system prompt; Bash passages cut when ``bash`` is off."""
    prompt = _MAI_GYM_DESKTOP_SYS_PROMPT.replace("{password}", password or "(not configured)")
    if not bash:
        for old, new in _NO_BASH_CUTS:
            prompt = prompt.replace(old, new)
    return prompt


# Upstream ``MAIGymComputerUsePolicy._retry_feedback``: the user turn that follows
# an unparseable reply on a parse retry. The Bash option is cut, as in the
# system prompt, when the sample offers no ``bash`` tool.
_MAI_GYM_DESKTOP_RETRY_FEEDBACK = (
    "Your previous response could not be parsed into an executable action.\n"
    "Use raw <bash>...</bash>, or one supported pyautogui/wait/terminate "
    "call inside <tool_call>...</tool_call>. Never call computer.bash.\n\n"
    "Unparseable previous response excerpt:\n"
)
_NO_BASH_RETRY_CUT = ("Use raw <bash>...</bash>, or one supported", "Use one supported")
_RETRY_EXCERPT_CHARS = 2000


def render_mai_gym_desktop_retry_feedback(response: str, *, bash: bool) -> str:
    """Upstream parse-retry feedback quoting the first 2000 chars of ``response``."""
    excerpt = response.strip()
    if len(excerpt) > _RETRY_EXCERPT_CHARS:
        excerpt = excerpt[:_RETRY_EXCERPT_CHARS] + "\n... [truncated]"
    feedback = _MAI_GYM_DESKTOP_RETRY_FEEDBACK
    if not bash:
        feedback = feedback.replace(*_NO_BASH_RETRY_CUT)
    return feedback + excerpt


# =============================================================================
# Desktop reply parsing — port of parsing/mai_gym_computer_use.py
# =============================================================================

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_THINKING_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE)
_BASH_RE = re.compile(r"<bash\s*>(.*?)</bash\s*>", re.DOTALL | re.IGNORECASE)
_ACTION_BLOCK_RE = re.compile(
    r"<bash\s*>.*?</bash\s*>|<tool_call[^>]*>.*?</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_TOOL_BODY_RE = re.compile(r"^<tool_call[^>]*>(.*?)</tool_call>$", re.DOTALL | re.IGNORECASE)
_TOOL_CALL_BEFORE_BASH_RE = re.compile(r"<tool_call[^>]*>\s*(?=<bash\s*>)", re.IGNORECASE)
_TOOL_CALL_AFTER_BASH_RE = re.compile(r"(</bash\s*>)\s*</tool_call>", re.IGNORECASE)


class _FatalBlock(Exception):
    """A block that fails the whole reply (upstream ``action_type: FAIL``)."""


def _call_string_argument(code: str, argument_name: str) -> str | None:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == argument_name and isinstance(keyword.value, ast.Constant):
                    value = keyword.value.value
                    return value if isinstance(value, str) else None
    return None


def _terminate_call(status: Any, answer: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {"status": "success" if status == "success" else "failure"}
    if isinstance(answer, str) and answer:
        arguments["answer"] = answer
    return {"name": COMPUTER_TERMINATE_TOOL_NAME, "arguments": arguments}


def _ask_user_call(question: Any) -> dict[str, Any]:
    return {"name": COMPUTER_ASK_USER_TOOL_NAME, "arguments": {"question": str(question or "")}}


def _json_tool_block(code: str) -> dict[str, Any] | None:
    """Upstream ``_parse_json_tool_call``: a JSON-bodied ``<tool_call>``."""
    if not code.lstrip().startswith("{"):
        return None
    try:
        payload = json.loads(code)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    name = str(payload.get("name") or payload.get("tool") or "")
    parameters = payload.get("parameters", payload.get("arguments", {}))
    merged = {**payload, **(parameters if isinstance(parameters, dict) else {})}
    if name in ("computer.wait", "wait"):
        return {"name": COMPUTER_WAIT_TOOL_NAME, "arguments": {}}
    if name in ("computer.terminate", "terminate"):
        status = str(merged.get("status", "success")).lower()
        return _terminate_call(status, merged.get("answer"))
    if name in ("computer.ask_user", "ask_user"):
        return _ask_user_call(merged.get("question"))
    func = name.removeprefix("pyautogui.")
    if func not in SAFE_PYAUTOGUI_FUNCS:
        return None
    args = merged.get("args", [])
    ignored = {"name", "tool", "parameters", "arguments", "args"}
    rendered = [repr(value) for value in (args if isinstance(args, list) else [])]
    rendered += [f"{key}={value!r}" for key, value in merged.items() if key not in ignored]
    code = f"pyautogui.{func}({', '.join(rendered)})"
    return {"name": PYAUTOGUI_TOOL_NAME, "arguments": {"code": code}}


def _tool_block(code: str) -> dict[str, Any] | None:
    """Upstream ``_parse_tool_block``; ``None`` skips the block silently."""
    json_call = _json_tool_block(code)
    if json_call is not None:
        return json_call
    lowered = code.lower()
    if "computer.bash" in lowered:
        raise _FatalBlock("legacy_computer_bash_is_disabled")
    if "computer.wait" in lowered:
        return {"name": COMPUTER_WAIT_TOOL_NAME, "arguments": {}}
    if "computer.terminate" in lowered:
        return _terminate_call(
            _call_string_argument(code, "status"), _call_string_argument(code, "answer")
        )
    if "computer.ask_user" in lowered:
        return _ask_user_call(_call_string_argument(code, "question"))
    if not validate_pyautogui_code(code):
        return None
    return {"name": PYAUTOGUI_TOOL_NAME, "arguments": {"code": code}}


def parse_mai_gym_desktop_reply(
    response: str,
    *,
    enable_ask_user: bool = False,
) -> tuple[str, list[dict[str, Any]], str, str | None]:
    """``(reasoning, agent calls, history text, error)`` for one desktop reply.

    Port of upstream ``parse_action_response``. ``history text`` is upstream's
    ``history_action`` — the accepted action blocks verbatim, newline-joined —
    which becomes this turn's content. Unparseable blocks are skipped; the
    reply fails only when no block survives or one block is fatal.
    """
    original = response or ""
    lowered = original.lower()
    if "</think>" in lowered and "<think>" not in lowered.split("</think>", 1)[0]:
        original = "<think>" + original
    reasoning_parts = [m.group(1).strip() for m in _THINK_RE.finditer(original)]
    reasoning_parts += [m.group(1).strip() for m in _THINKING_RE.finditer(original)]
    reasoning = "\n".join(part for part in reasoning_parts if part)

    action_content = _THINKING_RE.sub("", _THINK_RE.sub("", original))
    action_content = _TOOL_CALL_BEFORE_BASH_RE.sub("", action_content)
    action_content = _TOOL_CALL_AFTER_BASH_RE.sub(r"\1", action_content)

    calls: list[dict[str, Any]] = []
    history_blocks: list[str] = []
    for match in _ACTION_BLOCK_RE.finditer(action_content):
        block = match.group(0).strip()
        bash = _BASH_RE.fullmatch(block)
        if bash:
            calls.append({"name": BASH_TOOL_NAME, "arguments": {"command": bash.group(1).strip()}})
            history_blocks.append(block)
            continue
        tool = _TOOL_BODY_RE.match(block)
        if tool is None:
            continue
        try:
            call = _tool_block(tool.group(1).strip())
        except _FatalBlock as exc:
            return reasoning, [], "\n".join(history_blocks), str(exc)
        if call is not None:
            calls.append(call)
            history_blocks.append(block)

    history = "\n".join(history_blocks)
    if not calls:
        return reasoning, [], history, "no_executable_action"
    if not enable_ask_user and any(c["name"] == COMPUTER_ASK_USER_TOOL_NAME for c in calls):
        return reasoning, [], history, "ask_user_is_disabled"
    return reasoning, calls, history, None


# =============================================================================
# Desktop use adapter
# =============================================================================


@dataclasses.dataclass
class MAIGymDesktopUseAdapter(MAIGymBaseAdapter, key="mai_gym@desktop@use"):
    """Adapter for MAI-Gym-9B desktop ``use`` (computer use).

    Wire format:

    * native ``<think>`` reasoning, then one or more action blocks run top to
      bottom in one step: ``<tool_call>pyautogui.click(835, 420)</tool_call>``
      (coordinates in [0, 1000]), ``<tool_call>computer.wait()</tool_call>``,
      ``<tool_call>computer.terminate(status="success")</tool_call>`` or a raw
      ``<bash>...</bash>`` script;
    * the system prompt is upstream's computer-use template with the sudo
      ``password`` (yaml ``agent_kwargs.password``); Bash passages drop when
      the sample offers no ``bash`` tool;
    * history (``MAIGymDesktopHistoryProtocol``) is one user message per step
      — instruction on the first, ``## Bash Output`` when present, then the
      screenshot or ``<image_collapsed>`` — and each assistant turn replays
      its reply verbatim (the paper's ``token_native_history``);
    * screenshots go at native resolution (upstream projects [0, 1000] onto
      the real screen size).

    ask_user stays disabled, as upstream's default: an ``ask_user`` block makes
    the reply unparseable.
    """

    action_space: BaseActionSpace = dataclasses.field(
        default_factory=MAIGymDesktopActionSpace
    )
    protocol: MAIGymDesktopHistoryProtocol = dataclasses.field(
        default_factory=MAIGymDesktopHistoryProtocol
    )
    # The guest's sudo password, rendered into the system prompt.
    password: str = ""

    def __post_init__(self):
        super().__post_init__()
        if self.system_prompt is None:
            self.system_prompt = render_mai_gym_desktop_sys_prompt(
                self.password,
                bash=BASH_TOOL_NAME in self.active_extra_tool_names(),
            )

    def render_parse_retry_feedback(self, response: str) -> str:
        """The correction user turn that follows an unparseable ``response``."""
        return render_mai_gym_desktop_retry_feedback(
            response,
            bash=BASH_TOOL_NAME in self.active_extra_tool_names(),
        )

    def _convert_message_to_agent(
        self,
        message: LiteMessage,
        **kwargs,
    ) -> dict[str, Any]:
        """Canonical fallback render: reasoning plus re-rendered action blocks.

        Same-adapter turns replay their reply verbatim through the
        ``raw_response`` sidecar; this path serves turns produced elsewhere.
        """
        result = copy.deepcopy(message)
        if result.get("role") != "assistant":
            return result
        tool_calls = result.pop("tool_calls", None) or []
        lines = [
            render_desktop_block(call)
            for call in self.action_space.convert_tool_calls_to_agent(tool_calls)
        ]
        if not lines:
            lines = [
                p["text"] for p in message.get("content") or []
                if isinstance(p, dict) and p.get("type") == "text" and p.get("text")
            ]
        result["content"] = [{"type": "text", "text": "\n".join(lines)}] if lines else []
        return result

    def parse_raw_assistant_response(
        self,
        response: str,
        **kwargs,
    ) -> AgentMessage:
        """Parse one desktop reply; content is the accepted action blocks.

        A reply with no accepted block keeps its visible prose (the text after
        ``</think>``) as content, so the parse-failure final still shows what
        the model said.
        """
        reasoning, calls, history, error = parse_mai_gym_desktop_reply(response)
        result: AgentMessage = {"role": "assistant"}
        if reasoning:
            result["reasoning_content"] = reasoning
        if calls:
            result["tool_calls"] = calls
        if error is not None:
            mark_model_output_error(result, error)
        text = history or response.split(_THINK_CLOSE)[-1].strip()
        if text:
            result["content"] = [{"type": "text", "text": text}]
        return result


__all__ = [
    "MAIGymBaseAdapter",
    "MAIGymDesktopUseAdapter",
    "MAIGymMobileUseAdapter",
    "parse_mai_gym_desktop_reply",
    "render_mai_gym_desktop_retry_feedback",
    "render_mai_gym_desktop_sys_prompt",
    "render_mai_gym_mobile_sys_prompt",
]
