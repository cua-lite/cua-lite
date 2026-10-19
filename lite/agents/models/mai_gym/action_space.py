"""
MAI-Gym Action Space

Wraps every mobile action in a single ``mobile_use`` tool keyed by
``action_type``, matching the format MAI-Gym-9B was trained on.

Reference: ``MAI-Gym/src/mai_gym/agents/prompts/mai_gym_mobile_use_windows_wise_system.j2``
(action JSON examples) and ``MAI-Gym/src/mai_gym/agents/parsing/mai_gym_mobile_use.py``
(``convert_action``).

Key differences from :class:`MAIUIMobileActionSpace` (same Tongyi-MAI lineage):

- The action key is ``action_type``, not ``action``.
- ``double_click`` exists and maps to the canonical ``tap(clicks=2)``.
- There is NO ``swipe``: the model scrolls with ``drag``. Canonical ``swipe``
  and ``drag`` both render as ``drag``; every mobile env in this repo except
  mobilegym executes the two identically.
- ``system_button(menu)`` lowers to Home, exactly like upstream
  ``convert_action`` (``"menu": "navigate_home"``).
- ``terminate`` carries an optional ``answer``. Upstream turns any terminate
  that carries an answer into a terminal answer and ignores its ``status``;
  here that is ``response(text)``. Without an answer it is
  ``terminate(status)``.
- ``wait`` takes ``waiting_time`` (seconds, emitted as a string).
- ``ask_user`` is a native value; it projects to the env-owned standalone
  ``ask_user`` tool.

Coordinates are absolute [0, 999] on the wire (upstream scales by
``x / 999 * width``); conversion to cua-lite's normalized [0, 1000] happens
here. Malformed native actions raise :class:`ModelToolCallParseError`, the
terminal parse failure that mirrors upstream's invalid-action ->
``terminate(failure)`` path.

Usage:
    from lite.agents.models.mai_gym.action_space import MAIGymMobileActionSpace

    space = MAIGymMobileActionSpace()
    tc = MAIGymMobileActionSpace.mobile_use(action_type="click", coordinate=[500, 500])
"""

from __future__ import annotations

import ast
import dataclasses
import json
from typing import Any, Literal

from lite.agents.core.action_space import BaseActionSpace
from lite.agents.core.action_space.base import LiteDesktopActionSpace, LiteMobileActionSpace
from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.agents.core.action_space.utils.geometry import (
    compact_number,
    required_coord,
    required_model_text,
)
from lite.core.tools.action_space import merge_adjacent_lite_action_batches
from lite.core.tools.action_space.keys import normalize_keys
from lite.core.tools.calls import make_tool_call, tool_call_arguments, tool_call_name
from lite.core.tools.extra_tools import LiteFinishToolSet
from lite.core.tools.schemas import tool

# The single provider-native wrapper tool MAI-Gym-9B was trained on.
MAI_GYM_NATIVE_TOOL_NAME = "mobile_use"

# Upstream ``_scaled_point``: ``int(float(v) / 999 * width)``.
_SCALE_FACTOR = 999


def _scale_to_mai_gym(pts: list[int]) -> list[int]:
    return [int(round(p * _SCALE_FACTOR / 1000)) for p in pts]


def _required_from_mai_gym(c: Any, *, name: str = "coordinate") -> list[int]:
    """Required MAI-Gym [0, 999] point -> cua-lite [0, 1000]."""
    return [
        int(round(p * 1000 / _SCALE_FACTOR))
        for p in required_coord(c, dimensions=2, name=name)
    ]


_BUTTON_LITE_TO_MAI_GYM = {"Back": "back", "Home": "home", "Enter": "enter", "Menu": "menu"}
# Upstream ``convert_action`` maps ``menu`` to ``navigate_home``.
_BUTTON_MAI_GYM_TO_LITE = {"back": "Back", "home": "Home", "enter": "Enter", "menu": "Home"}

_STATUS_LITE_TO_MAI_GYM = {"success": "success", "failure": "failed"}
_STATUS_MAI_GYM_TO_LITE = {"success": "success", "failed": "failure", "failure": "failure"}


@dataclasses.dataclass
class MAIGymMobileActionSpace(BaseActionSpace, key="mai_gym@mobile"):
    """MAI-Gym ``mobile_use`` action space.

    The model-visible action vocabulary is the ``## Action Space`` block of the
    adapter's system prompt (trained text), so the schema below documents the
    wire and is never rendered into the prompt.

    Actions:
        click, double_click, long_press, type, open, drag, system_button,
        wait, ask_user, terminate.
    """

    platform: str = "mobile"

    @staticmethod
    @tool(
        action_type="The action to perform.",
        coordinate="(x, y): point for click/double_click/long_press. Range [0, 999].",
        start_coordinate="(x1, y1): drag start in [0, 999]. Required only by `action_type=drag`.",
        end_coordinate="(x2, y2): drag end in [0, 999]. Required only by `action_type=drag`.",
        text="Required only by `action_type=type`, `action_type=open`, and `action_type=ask_user`.",
        button="back, home, menu, or enter. Required only by `action_type=system_button`.",
        waiting_time="Seconds to wait. Used only by `action_type=wait`.",
        answer="Final answer. Used only by `action_type=terminate`.",
        status="Task status. Used only by `action_type=terminate`.",
    )
    def mobile_use(
        action_type: Literal[
            "click", "double_click", "long_press", "type", "open",
            "drag", "system_button", "wait", "ask_user", "terminate",
        ],
        coordinate: list[int] | None = None,
        start_coordinate: list[int] | None = None,
        end_coordinate: list[int] | None = None,
        text: str | None = None,
        button: Literal["back", "home", "menu", "enter"] | None = None,
        waiting_time: str | None = None,
        answer: str | None = None,
        status: Literal["success", "failed"] | None = None,
    ) -> dict[str, Any]:
        """Use a touchscreen to interact with a mobile device."""
        arguments = {
            "action_type": action_type,
            "coordinate": coordinate,
            "start_coordinate": start_coordinate,
            "end_coordinate": end_coordinate,
            "text": text,
            "button": button,
            "waiting_time": waiting_time,
            "answer": answer,
            "status": status,
        }
        return make_tool_call(
            MAI_GYM_NATIVE_TOOL_NAME,
            {k: v for k, v in arguments.items() if v is not None},
        )

    # -------------------------------------------------------------------------
    # Tool call conversion: cua-lite -> MAI-Gym
    # -------------------------------------------------------------------------

    def _convert_single_to_agent(self, tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        name = tool_call_name(tool_call)
        args = tool_call_arguments(tool_call)

        if name == "mobile":
            result = []
            for child in args["actions"]:
                child_args = {k: v for k, v in child.items() if k != "action"}
                child_call = make_tool_call(child["action"], child_args)
                result.extend(self._convert_single_to_agent(child_call))
            return result

        if name == "tap":
            clicks = args.get("clicks", 1)
            return [MAIGymMobileActionSpace.mobile_use(
                action_type="double_click" if clicks == 2 else "click",
                coordinate=_scale_to_mai_gym(required_coord(args.get("coordinate"), dimensions=2)),
            )["function"]]

        if name == "long_press":
            # The wire has no press duration; upstream long-presses for the env default.
            return [MAIGymMobileActionSpace.mobile_use(
                action_type="long_press",
                coordinate=_scale_to_mai_gym(required_coord(args.get("coordinate"), dimensions=2)),
            )["function"]]

        if name in ("swipe", "drag"):
            start = required_coord(
                args.get("start_coordinate"), dimensions=2, name="start_coordinate"
            )
            end = required_coord(args.get("coordinate"), dimensions=2)
            return [MAIGymMobileActionSpace.mobile_use(
                action_type="drag",
                start_coordinate=_scale_to_mai_gym(start),
                end_coordinate=_scale_to_mai_gym(end),
            )["function"]]

        if name == "type":
            return [MAIGymMobileActionSpace.mobile_use(
                action_type="type",
                text=args.get("text", ""),
            )["function"]]

        if name == "open_app":
            return [MAIGymMobileActionSpace.mobile_use(
                action_type="open",
                text=args.get("app_name", ""),
            )["function"]]

        if name == "system_button":
            button = args.get("button", "")
            if button not in _BUTTON_LITE_TO_MAI_GYM:
                raise ValueError(
                    f"MAI-Gym has no native system button for {button!r}; "
                    f"supported: {sorted(_BUTTON_LITE_TO_MAI_GYM)}"
                )
            return [MAIGymMobileActionSpace.mobile_use(
                action_type="system_button",
                button=_BUTTON_LITE_TO_MAI_GYM[button],
            )["function"]]

        if name == "wait":
            return [MAIGymMobileActionSpace.mobile_use(
                action_type="wait",
                waiting_time=str(compact_number(args.get("duration", 1.0))),
            )["function"]]

        if name == "ask_user":
            return [MAIGymMobileActionSpace.mobile_use(
                action_type="ask_user",
                text=args.get("text", ""),
            )["function"]]

        if name == "response":
            return [MAIGymMobileActionSpace.mobile_use(
                action_type="terminate",
                answer=args.get("text", ""),
                status="success",
            )["function"]]

        if name == "terminate":
            status = args.get("status", "success")
            return [MAIGymMobileActionSpace.mobile_use(
                action_type="terminate",
                status=_STATUS_LITE_TO_MAI_GYM.get(status, status),
            )["function"]]

        if name in LiteMobileActionSpace.get_action_names():
            # ``pinch`` / ``screenshot`` have no MAI-Gym action; dropping them
            # would silently lose the action from the wire.
            raise ValueError(f"MAI-Gym cannot render canonical tool {name!r}")

        # Anything else is an env extra tool: bare function-call agent wire.
        return super()._convert_single_to_agent(tool_call)

    # -------------------------------------------------------------------------
    # Tool call conversion: MAI-Gym -> cua-lite
    # -------------------------------------------------------------------------

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Convert MAI-Gym ``mobile_use`` calls back to cua-lite format.

        Like upstream ``parse_action_response``, dispatch reads
        ``arguments.action_type`` (falling back to ``action``) and ignores the
        tool name. A call with neither key is a standalone tool and keeps its
        own wire name, so env ingress answers it with model-visible feedback.
        """
        result: list[dict[str, Any]] = []
        for tc in agent_tool_calls:
            result.extend(self._convert_single_from_agent(tc["name"], tc["arguments"]))
        return merge_adjacent_lite_action_batches(result)

    def _convert_single_from_agent(
        self,
        name: str,
        args: dict[str, Any],
    ) -> list[dict[str, Any]]:
        action = args.get("action_type", args.get("action"))
        if action is None:
            return [make_tool_call(name, args)]

        if action in ("click", "double_click"):
            return [LiteMobileActionSpace.tap(
                coordinate=_required_from_mai_gym(args.get("coordinate")),
                clicks=2 if action == "double_click" else 1,
            )]

        if action == "long_press":
            return [LiteMobileActionSpace.long_press(
                coordinate=_required_from_mai_gym(args.get("coordinate")),
            )]

        if action == "drag":
            return [LiteMobileActionSpace.drag(
                start_coordinate=_required_from_mai_gym(
                    args.get("start_coordinate"), name="start_coordinate"
                ),
                coordinate=_required_from_mai_gym(
                    args.get("end_coordinate"), name="end_coordinate"
                ),
            )]

        if action == "type":
            return [LiteMobileActionSpace.type(text=required_model_text(args, action=action))]

        if action == "open":
            return [make_tool_call("open_app", {"app_name": args.get("text", "")})]

        if action == "system_button":
            button = str(args.get("button", "")).lower()
            if button not in _BUTTON_MAI_GYM_TO_LITE:
                raise ModelToolCallParseError(f"system_button: unknown button {button!r}")
            return [LiteMobileActionSpace.system_button(button=_BUTTON_MAI_GYM_TO_LITE[button])]

        if action == "wait":
            try:
                duration = float(args["waiting_time"])
            except (KeyError, TypeError, ValueError):
                duration = 1.0
            return [LiteMobileActionSpace.wait(duration=duration)]

        if action == "ask_user":
            return [make_tool_call("ask_user", {"text": args.get("text", "")})]

        if action == "terminate":
            if args.get("answer") is not None:
                return [LiteFinishToolSet.response(text=str(args["answer"]))]
            status = args.get("status", "success")
            return [LiteFinishToolSet.terminate(status=_STATUS_MAI_GYM_TO_LITE.get(status, status))]

        if action == "answer":
            # Upstream ``convert_action`` still accepts the legacy answer action.
            answer = args.get("answer")
            if answer is None:
                answer = args.get("text", "")
            return [LiteFinishToolSet.response(text=str(answer))]

        raise ModelToolCallParseError(f"Unknown action: {action}")


# =============================================================================
# Desktop (computer use)
# =============================================================================
#
# Reference: ``MAI-Gym/src/mai_gym/agents/parsing/mai_gym_computer_use.py``.
#
# The desktop wire has no JSON schema: each ``<tool_call>`` block holds literal
# ``pyautogui.*`` calls (coordinates in [0, 1000]) or a ``computer.*``
# flow-control call, and ``<bash>...</bash>`` holds a raw shell script. The
# adapter splits a reply into those blocks; this space converts one block's
# agent-format call to canonical Lite calls:
#
#     {"name": "pyautogui", "arguments": {"code": "pyautogui.click(1, 2)"}}
#     {"name": "bash", "arguments": {"command": "ls"}}
#     {"name": "computer.wait", "arguments": {}}
#     {"name": "computer.terminate", "arguments": {"status": ..., "answer"?: ...}}
#     {"name": "computer.ask_user", "arguments": {"question": ...}}
#
# Upstream's [0, 1000] grid is cua-lite's normalized grid, so coordinates pass
# through (rounded to ints).

PYAUTOGUI_TOOL_NAME = "pyautogui"
COMPUTER_WAIT_TOOL_NAME = "computer.wait"
COMPUTER_TERMINATE_TOOL_NAME = "computer.terminate"
COMPUTER_ASK_USER_TOOL_NAME = "computer.ask_user"

#: Upstream ``_SAFE_PYAUTOGUI_FUNCS``: the only calls a block may contain.
SAFE_PYAUTOGUI_FUNCS = frozenset({
    "click", "leftClick", "rightClick", "middleClick", "doubleClick", "tripleClick",
    "moveTo", "dragTo", "typewrite", "write", "press", "hotkey", "keyDown", "keyUp",
    "scroll", "hscroll", "vscroll", "mouseDown", "mouseUp",
})

# ``computer.wait()`` carries no duration; upstream sends the env's bare WAIT.
_COMPUTER_WAIT_SECONDS = 1.0

_CLICK_FUNCS = {
    "click": ("left", 1), "leftClick": ("left", 1), "rightClick": ("right", 1),
    "middleClick": ("middle", 1), "doubleClick": ("left", 2), "tripleClick": ("left", 3),
}
# Positional layout of the pyautogui signatures this parser reads.
_POSITIONAL = {
    "click": ("x", "y", "clicks", "interval", "button"),
    "leftClick": ("x", "y"), "rightClick": ("x", "y"), "middleClick": ("x", "y"),
    "doubleClick": ("x", "y", "interval", "button"),
    "tripleClick": ("x", "y", "interval", "button"),
    "moveTo": ("x", "y", "duration"),
    "dragTo": ("x", "y", "duration", "tween", "button"),
    "mouseDown": ("x", "y", "button"), "mouseUp": ("x", "y", "button"),
    "typewrite": ("message", "interval"), "write": ("message", "interval"),
    "press": ("keys", "presses", "interval"),
    "keyDown": ("key",), "keyUp": ("key",),
    "scroll": ("clicks", "x", "y"), "vscroll": ("clicks", "x", "y"),
    "hscroll": ("clicks", "x", "y"),
}
# Canonical key token -> pyautogui key name, where they differ.
_KEY_LITE_TO_PYAUTOGUI = {"meta": "win", "kp_enter": "enter"}


def _literal_value(node: ast.expr) -> bool:
    try:
        value = ast.literal_eval(node)
    except (SyntaxError, ValueError, TypeError):
        return False
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    return isinstance(value, (list, tuple)) and all(
        item is None or isinstance(item, (str, int, float, bool)) for item in value
    )


def validate_pyautogui_code(code: str) -> bool:
    """Upstream ``validate_pyautogui_code``: only direct safe calls with literal args."""
    try:
        tree = ast.parse(code.strip(), mode="exec")
    except (SyntaxError, ValueError):
        return False
    if not tree.body:
        return False
    for statement in tree.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            return False
        call = statement.value
        if not isinstance(call.func, ast.Attribute) or not isinstance(call.func.value, ast.Name):
            return False
        if call.func.value.id != "pyautogui" or call.func.attr not in SAFE_PYAUTOGUI_FUNCS:
            return False
        if any(keyword.arg is None for keyword in call.keywords):
            return False
        if not all(_literal_value(arg) for arg in call.args):
            return False
        if not all(_literal_value(keyword.value) for keyword in call.keywords):
            return False
    return True


def _model_keys(raw: Any, func: str) -> list[str]:
    tokens = raw if isinstance(raw, (list, tuple)) else [raw]
    try:
        return normalize_keys([str(token) for token in tokens])
    except (TypeError, ValueError) as exc:
        raise ModelToolCallParseError(f"pyautogui.{func}: {exc}") from exc


def _number(value: Any, what: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ModelToolCallParseError(f"{what} must be numeric; got {value!r}") from exc


def _point(bound: dict[str, Any], func: str, *, required: bool) -> list[int] | None:
    x, y = bound.get("x"), bound.get("y")
    if x is None and y is None and not required:
        return None
    if x is None or y is None:
        raise ModelToolCallParseError(f"pyautogui.{func} needs both x and y")
    return [int(round(_number(x, "x"))), int(round(_number(y, "y")))]


def _pyautogui_call_to_lite(call: ast.Call) -> list[dict[str, Any]]:
    """One validated ``pyautogui.<func>(...)`` call -> canonical computer calls."""
    func = call.func.attr
    args = [ast.literal_eval(arg) for arg in call.args]
    bound: dict[str, Any] = dict(zip(_POSITIONAL.get(func, ()), args))
    bound.update({kw.arg: ast.literal_eval(kw.value) for kw in call.keywords})

    if func in _CLICK_FUNCS:
        button, clicks = _CLICK_FUNCS[func]
        if func == "click":
            clicks = max(1, min(3, int(_number(bound.get("clicks", 1), "clicks"))))
        return [LiteDesktopActionSpace.click(
            coordinate=_point(bound, func, required=False),
            button=bound.get("button", button),
            clicks=clicks,
        )]
    if func == "moveTo":
        return [LiteDesktopActionSpace.mouse_move(coordinate=_point(bound, func, required=True))]
    if func == "dragTo":
        # No start point: pyautogui drags from the current cursor, as the
        # canonical drag does when ``start_coordinate`` is omitted.
        return [LiteDesktopActionSpace.drag(
            coordinate=_point(bound, func, required=True),
            button=bound.get("button", "left"),
        )]
    if func in ("mouseDown", "mouseUp"):
        out: list[dict[str, Any]] = []
        point = _point(bound, func, required=False)
        if point is not None:
            out.append(LiteDesktopActionSpace.mouse_move(coordinate=point))
        press = LiteDesktopActionSpace.mouse_down if func == "mouseDown" else LiteDesktopActionSpace.mouse_up
        out.append(press(button=bound.get("button", "left")))
        return out
    if func in ("typewrite", "write"):
        message = bound.get("message", "")
        if isinstance(message, (list, tuple)):
            # ``typewrite([...])`` presses each named key in turn.
            return [LiteDesktopActionSpace.key(keys=[key]) for key in _model_keys(message, func)]
        return [LiteDesktopActionSpace.type(text=str(message))]
    if func == "press":
        keys = _model_keys(bound.get("keys"), func)
        presses = max(1, int(_number(bound.get("presses", 1), "presses")))
        return [LiteDesktopActionSpace.key(keys=[key]) for _ in range(presses) for key in keys]
    if func == "hotkey":
        return [LiteDesktopActionSpace.key(keys=_model_keys(args, func))]
    if func in ("keyDown", "keyUp"):
        keys = _model_keys(bound.get("key"), func)
        hold = LiteDesktopActionSpace.key_down if func == "keyDown" else LiteDesktopActionSpace.key_up
        return [hold(keys=keys)]
    if func in ("scroll", "vscroll", "hscroll"):
        clicks = int(_number(bound.get("clicks", 0), "clicks"))
        if clicks == 0:
            return []
        if func == "hscroll":
            direction = "right" if clicks > 0 else "left"
        else:
            direction = "up" if clicks > 0 else "down"
        return [LiteDesktopActionSpace.scroll(
            direction=direction,
            amount=abs(clicks),
            coordinate=_point(bound, func, required=False),
        )]
    raise ModelToolCallParseError(f"unsupported GUI call: pyautogui.{func}")


def pyautogui_code_to_lite(code: str) -> list[dict[str, Any]]:
    """A validated pyautogui block -> canonical computer calls, in order."""
    if not validate_pyautogui_code(code):
        raise ModelToolCallParseError(f"not a supported pyautogui block: {code!r}")
    result: list[dict[str, Any]] = []
    for statement in ast.parse(code.strip()).body:
        result.extend(_pyautogui_call_to_lite(statement.value))
    return result


def _pyautogui_key(key: str) -> str:
    return _KEY_LITE_TO_PYAUTOGUI.get(key, key)


def _lite_child_to_pyautogui(name: str, args: dict[str, Any]) -> str:
    """One canonical computer child action -> pyautogui source (render direction)."""
    point = args.get("coordinate")
    xy = f"{point[0]}, {point[1]}" if point is not None else ""
    button = args.get("button", "left")
    if name == "click":
        clicks = args.get("clicks", 1)
        if button == "left" and clicks in (2, 3):
            return f"pyautogui.{'doubleClick' if clicks == 2 else 'tripleClick'}({xy})"
        if clicks == 1 and button in ("right", "middle"):
            return f"pyautogui.{button}Click({xy})"
        extra = [f"clicks={clicks}"] if clicks != 1 else []
        extra += [f"button={button!r}"] if button != "left" else []
        return f"pyautogui.click({', '.join([xy, *extra] if xy else extra)})"
    if name == "mouse_move":
        return f"pyautogui.moveTo({xy})"
    if name == "drag":
        button_arg = f", button={button!r}" if button != "left" else ""
        drag = f"pyautogui.dragTo({xy}, duration=0.5{button_arg})"
        start = args.get("start_coordinate")
        return f"pyautogui.moveTo({start[0]}, {start[1]})\n{drag}" if start is not None else drag
    if name in ("mouse_down", "mouse_up"):
        func = "mouseDown" if name == "mouse_down" else "mouseUp"
        return f"pyautogui.{func}(button={button!r})"
    if name == "type":
        return f"pyautogui.typewrite({args.get('text', '')!r})"
    if name == "key":
        keys = [_pyautogui_key(key) for key in args.get("keys", [])]
        if len(keys) == 1:
            return f"pyautogui.press({keys[0]!r})"
        return f"pyautogui.hotkey({', '.join(repr(key) for key in keys)})"
    if name in ("key_down", "key_up"):
        func = "keyDown" if name == "key_down" else "keyUp"
        return "\n".join(f"pyautogui.{func}({_pyautogui_key(key)!r})" for key in args.get("keys", []))
    if name == "scroll":
        amount = int(args.get("amount", 0))
        direction = args.get("direction", "down")
        func = "hscroll" if direction in ("left", "right") else "scroll"
        clicks = amount if direction in ("up", "right") else -amount
        return f"pyautogui.{func}({', '.join([str(clicks), xy] if xy else [str(clicks)])})"
    raise ValueError(f"MAI-Gym cannot render canonical computer action {name!r}")


@dataclasses.dataclass
class MAIGymDesktopActionSpace(BaseActionSpace, key="mai_gym@desktop"):
    """MAI-Gym computer-use space: pyautogui / ``<bash>`` / ``computer.*`` blocks.

    The model-visible vocabulary is the system prompt (trained text), so no
    tool schema is declared.
    """

    platform: str = "desktop"

    @classmethod
    def get_tool_schemas(cls, include: list[str] | None = None) -> list[dict[str, Any]]:
        return []

    @classmethod
    def get_declared_action_schema_names(cls) -> frozenset[str]:
        """Native call names the system prompt orders: ``pyautogui.*`` and ``computer.wait``.

        The wire carries no tool schema, so the names are declared here rather
        than derived from ``_SCHEMAS``.
        """
        return SAFE_PYAUTOGUI_FUNCS | {"wait"}

    def _convert_single_to_agent(self, tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        name = tool_call_name(tool_call)
        args = tool_call_arguments(tool_call)
        if name == "computer":
            result = []
            for child in args["actions"]:
                child_name = child["action"]
                child_args = {k: v for k, v in child.items() if k != "action"}
                if child_name == "wait":
                    result.append({"name": COMPUTER_WAIT_TOOL_NAME, "arguments": {}})
                    continue
                result.append({
                    "name": PYAUTOGUI_TOOL_NAME,
                    "arguments": {"code": _lite_child_to_pyautogui(child_name, child_args)},
                })
            return result
        if name == "bash":
            return [{"name": "bash", "arguments": {"command": args.get("command", "")}}]
        if name == "response":
            return [{
                "name": COMPUTER_TERMINATE_TOOL_NAME,
                "arguments": {"status": "success", "answer": args.get("text", "")},
            }]
        if name == "terminate":
            return [{
                "name": COMPUTER_TERMINATE_TOOL_NAME,
                "arguments": {"status": args.get("status", "success")},
            }]
        if name == "ask_user":
            return [{
                "name": COMPUTER_ASK_USER_TOOL_NAME,
                "arguments": {"question": args.get("text", "")},
            }]
        # Any other name is an env extra tool: bare function-call agent wire,
        # which ``render_desktop_block`` renders as a JSON ``<tool_call>``.
        return super()._convert_single_to_agent(tool_call)

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Convert parsed desktop blocks to canonical calls, preserving order."""
        result: list[dict[str, Any]] = []
        for tc in agent_tool_calls:
            name, args = tc["name"], tc["arguments"]
            if name == PYAUTOGUI_TOOL_NAME:
                result.extend(pyautogui_code_to_lite(args["code"]))
            elif name == COMPUTER_WAIT_TOOL_NAME:
                result.append(LiteDesktopActionSpace.wait(duration=_COMPUTER_WAIT_SECONDS))
            elif name == COMPUTER_TERMINATE_TOOL_NAME:
                # Upstream ``finished`` keys success on ``status``; an answer
                # rides only on a successful finish (osworld reads a failure
                # finish as FAIL, the infeasible verdict).
                success = args.get("status") == "success"
                if success and isinstance(args.get("answer"), str) and args["answer"]:
                    result.append(LiteFinishToolSet.response(text=args["answer"]))
                else:
                    status = "success" if success else "failure"
                    result.append(LiteFinishToolSet.terminate(status=status))
            elif name == COMPUTER_ASK_USER_TOOL_NAME:
                result.append(make_tool_call("ask_user", {"text": args.get("question", "")}))
            else:
                # ``bash`` and anything else: a standalone tool under its own
                # name; env ingress answers an inactive one with feedback.
                result.append(make_tool_call(name, args))
        return merge_adjacent_lite_action_batches(result)


def render_desktop_block(agent_call: dict[str, Any]) -> str:
    """Render one agent-format desktop call as its wire block."""
    name, args = agent_call["name"], agent_call["arguments"]
    if name == PYAUTOGUI_TOOL_NAME:
        return f"<tool_call>{args['code']}</tool_call>"
    if name == "bash":
        return f"<bash>\n{args.get('command', '')}\n</bash>"
    if name == COMPUTER_WAIT_TOOL_NAME:
        return "<tool_call>computer.wait()</tool_call>"
    if name == COMPUTER_TERMINATE_TOOL_NAME:
        parts = [f"status={json.dumps(args.get('status', 'success'))}"]
        if args.get("answer"):
            parts.append(f"answer={json.dumps(args['answer'], ensure_ascii=False)}")
        return f"<tool_call>computer.terminate({', '.join(parts)})</tool_call>"
    if name == COMPUTER_ASK_USER_TOOL_NAME:
        question = json.dumps(args.get("question", ""), ensure_ascii=False)
        return f"<tool_call>computer.ask_user(question={question})</tool_call>"
    return "<tool_call>" + json.dumps({"name": name, "arguments": args}, ensure_ascii=False) + "</tool_call>"


__all__ = [
    "COMPUTER_ASK_USER_TOOL_NAME",
    "COMPUTER_TERMINATE_TOOL_NAME",
    "COMPUTER_WAIT_TOOL_NAME",
    "MAI_GYM_NATIVE_TOOL_NAME",
    "MAIGymDesktopActionSpace",
    "MAIGymMobileActionSpace",
    "PYAUTOGUI_TOOL_NAME",
    "SAFE_PYAUTOGUI_FUNCS",
    "pyautogui_code_to_lite",
    "render_desktop_block",
    "validate_pyautogui_code",
]
