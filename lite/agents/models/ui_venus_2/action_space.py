"""
UI-Venus-2 Action Spaces (Computer / Browser / Mobile / Grounding)

UI-Venus-2 answers with ``<think>…</think><action>…</action>`` and puts exactly
one capitalized call inside ``<action>``. Coordinates are bare parenthesised
pairs normalized to the screen — the prompts spell the inclusive maximum as
999 and the harnesses divide by 999 (computer) or 1000 (mobile / browser), a
sub-pixel difference from cua-lite's canonical [0, 1000], so every surface here
is an identity coordinate frame.

Unlike UI-Venus-1.5, the three ``use`` surfaces have genuinely DIFFERENT
grammars, so each gets its own registry key rather than sharing one under a
``(desktop|browser)`` regex:

* ``ui_venus_2@desktop`` — the Computer grammar
  (``${CUA_LITE_REFERENCES_ROOT}/UI-Venus@UI-Venus-2/models/computer/computer_example.py``).
  The richest of the three: separate verbs for double / triple / right / middle
  click, half-press key and mouse verbs, and ``Sequence(actions=[...])``, which
  is this family's spelling of cua-lite's ``computer`` action batch.
* ``ui_venus_2@browser`` — the Browser grammar (``models/browser/venus_browser.py``):
  a point-and-direction ``Scroll``, ``Launch(url=)``, plus three verbs with no
  canonical counterpart (``GetUrl`` / ``TakeNote`` / ``SelectOption``).
* ``ui_venus_2@mobile`` — the Android grammar (``models/mobile/mobile_example.py``
  and ``Venus_framework/Venus_framework_mobile/processor/ui_venus_2_processor.py``):
  two-endpoint ``Swipe`` / ``Drag``, device buttons, and a separate ``Answer``
  verb for the reply channel.
* ``ui_venus_2@(desktop|browser|mobile)@point`` — grounding, whose wire is not a
  call at all but a bare ``[x,y]`` list with ``[-1,-1]`` as the trained
  infeasible marker (``models/grounding/ui_venus2_gd.py``).

Usage:
    from lite.agents.models.ui_venus_2.action_space import UIVenus2DesktopActionSpace
"""

from __future__ import annotations

import ast
import dataclasses
import logging
import re
from typing import Any, Literal

from lite.agents.core.action_space.base import (
    BaseActionSpace,
    LiteDesktopActionSpace,
    LiteMobileActionSpace,
    LitePointActionSpace,
)
from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.agents.core.action_space.utils.geometry import (
    PIXELS_PER_CLICK,
    required_coord,
)
from lite.agents.core.action_space.utils.grounding_point import (
    convert_non_point_call_for_grounding_space,
)
from lite.core.tools.action_space import (
    LiteDesktopActionSet,
    LiteMobileActionSet,
    make_lite_action_batch_call,
    merge_adjacent_lite_action_batches,
)
from lite.core.tools.calls import (
    make_tool_call,
    tool_call_arguments,
    tool_call_name,
)
from lite.core.tools.extra_tools import (
    LiteAppLaunchToolSet,
    LiteBrowserNavToolSet,
    LiteFinishToolSet,
)
from lite.core.tools.schemas import tool

logger = logging.getLogger(__name__)

#: Seconds a bare ``Wait()`` means. The wire carries no duration; the Android
#: harness is the only upstream code that assigns one
#: (``normalize_action``: ``result["duration"] = 1000`` milliseconds), so the
#: whole family reads ``Wait()`` as one second.
WAIT_SECONDS = 1.0

#: This family's native name for the ``Sequence`` open-loop action batch, which
#: is what cua-lite's ``computer`` batch projects to on the Computer surface.
SEQUENCE_NATIVE_NAME = "Sequence"

#: Argument names whose value renders as a bare ``(x, y)`` pair.
_COORD_ARG_NAMES = frozenset({"box", "point", "start", "end"})

# Canonical Lite catalogs are owned by the core action sets; each surface can
# express only a subset, so only the convertible parse outputs are wrapped back
# into ``computer`` / ``mobile``.
_DESKTOP_UNSUPPORTED_ACTIONS = frozenset({"cursor_position", "hold_key", "screenshot"})
_BROWSER_UNSUPPORTED_ACTIONS = frozenset({
    "cursor_position",
    "hold_key",
    "key_down",
    "key_up",
    "mouse_down",
    "mouse_up",
    "screenshot",
})
_MOBILE_UNSUPPORTED_ACTIONS = frozenset({"pinch"})

_DESKTOP_ACTIONS = LiteDesktopActionSet.get_action_names() - _DESKTOP_UNSUPPORTED_ACTIONS
_BROWSER_ACTIONS = LiteDesktopActionSet.get_action_names() - _BROWSER_UNSUPPORTED_ACTIONS
_MOBILE_ACTIONS = LiteMobileActionSet.get_action_names() - _MOBILE_UNSUPPORTED_ACTIONS


def _wrap_action_call(
    call: dict[str, Any], wrapper: str, actions: frozenset[str],
) -> dict[str, Any]:
    name = tool_call_name(call)
    if name not in actions:
        return call
    return make_lite_action_batch_call(wrapper, call)


# =============================================================================
# Wire text: render + parse
# =============================================================================

def format_tool_call_as_text(agent_tool_call: dict[str, Any]) -> str:
    """Render one UI-Venus-2 agent projection as the text inside ``<action>``.

    Input shape is part of the contract: this takes the family's BARE
    ``{name, arguments}`` projection — what ``convert_tool_calls_to_agent``
    returns — not a canonical Lite call, which raises ``KeyError`` here.

    ``None``-valued arguments are dropped: every optional argument in this
    grammar is spelled by omission (``Click()`` acts at the cursor), and the
    model never emits ``key=None``.

    Returns:
        ``Click(box=(512, 300))``, ``Hotkey(keys=['ctrl', 'c'])``, or a nested
        ``Sequence(actions=[Click(box=(1, 2)), Type(content='x')])``.
    """
    name = agent_tool_call["name"]
    args = agent_tool_call["arguments"]

    if name == SEQUENCE_NATIVE_NAME:
        children = ", ".join(format_tool_call_as_text(child) for child in args["actions"])
        return f"{SEQUENCE_NATIVE_NAME}(actions=[{children}])"

    parts: list[str] = []
    for key, value in args.items():
        if value is None:
            continue
        if key in _COORD_ARG_NAMES and isinstance(value, (list, tuple)):
            # EVERY component is rendered, never just the first two: this wire's
            # points are (x, y), so a longer list is malformed, and printing a
            # well-formed-looking prefix would put a point back into model
            # context that the parser had already rejected.
            parts.append(f"{key}=({', '.join(str(c) for c in value)})")
        elif isinstance(value, (list, tuple)):
            parts.append(f"{key}=[{', '.join(repr(str(item)) for item in value)}]")
        else:
            parts.append(f"{key}={value!r}")

    return f"{name}({', '.join(parts)})"


class _ActionSyntaxError(ValueError):
    """The ``<action>`` body is not a call this grammar can carry."""


def _literal(node: ast.AST) -> Any:
    """Literal-only argument evaluation — never executes model text."""
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError) as exc:
        raise _ActionSyntaxError("action arguments must be literals") from exc
    return list(value) if isinstance(value, tuple) else value


def _parse_call_node(node: ast.AST, *, allow_sequence: bool) -> dict[str, Any]:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise _ActionSyntaxError("action must be a direct function call")
    if node.args:
        raise _ActionSyntaxError("positional arguments are not allowed")

    keywords: dict[str, ast.AST] = {}
    for keyword in node.keywords:
        if keyword.arg is None or keyword.arg in keywords:
            raise _ActionSyntaxError("invalid action arguments")
        keywords[keyword.arg] = keyword.value

    if node.func.id == SEQUENCE_NATIVE_NAME:
        # Upstream ``_parse_call`` rejects a nested Sequence outright; so does
        # this parser, because a nested batch has no canonical shape either.
        if not allow_sequence:
            raise _ActionSyntaxError("nested Sequence is not allowed")
        actions_node = keywords.get("actions")
        if set(keywords) != {"actions"} or not isinstance(actions_node, ast.List):
            raise _ActionSyntaxError("Sequence requires actions=[...]")
        return {
            "name": SEQUENCE_NATIVE_NAME,
            "arguments": {
                "actions": [
                    _parse_call_node(child, allow_sequence=False)
                    for child in actions_node.elts
                ],
            },
        }

    return {
        "name": node.func.id,
        "arguments": {key: _literal(value) for key, value in keywords.items()},
    }


def parse_action_text(action_str: str) -> dict[str, Any] | None:
    """Parse one ``<action>`` body into a bare ``{name, arguments}`` projection.

    Returns ``None`` when the text is not a call at all, so the caller can tell
    "the model wrote prose" from "the model wrote a broken call".
    """
    text = action_str.strip()
    if not text:
        return None
    try:
        node = ast.parse(text, mode="eval").body
    except (SyntaxError, ValueError, RecursionError):
        return _parse_truncated_action_text(text)
    try:
        return _parse_call_node(node, allow_sequence=True)
    except _ActionSyntaxError:
        return None


#: A truncated call still starts with ``Name(``; the model runs out of tokens
#: mid-argument often enough (long ``Type(content=…)`` bodies) that dropping the
#: whole step would cost a turn the trajectory cannot get back.
_TRUNCATED_CALL_RE = re.compile(r"^([A-Za-z_]\w*)\s*\((.*)$", re.DOTALL)
_TRUNCATED_COORD_RE = re.compile(r"(\w+)\s*=\s*\(\s*(-?\d+)\s*,\s*(-?\d+)")
_TRUNCATED_STR_RE = re.compile(r"(\w+)\s*=\s*'([^']*)'?")


def _parse_truncated_action_text(text: str) -> dict[str, Any] | None:
    match = _TRUNCATED_CALL_RE.match(text)
    if not match or match.group(1) == SEQUENCE_NATIVE_NAME:
        return None
    name, rest = match.group(1), match.group(2)
    arguments: dict[str, Any] = {}
    for coord in _TRUNCATED_COORD_RE.finditer(rest):
        arguments[coord.group(1)] = [int(coord.group(2)), int(coord.group(3))]
    for string in _TRUNCATED_STR_RE.finditer(rest):
        arguments.setdefault(string.group(1), string.group(2))
    if not arguments:
        # Nothing survived the truncation. Returning the bare name would be
        # actively harmful on this grammar, where every coordinate argument is
        # OPTIONAL: ``Click(box=(1`` would become ``Click()``, a click at the
        # cursor -- a different action the model never asked for. Report a
        # failed parse instead, which the adapter turns into model-visible
        # feedback.
        return None
    return {"name": name, "arguments": arguments}


# =============================================================================
# Shared scroll geometry
# =============================================================================
# The Computer grammar's ``Swipe(amount=N, axis=…)`` counts SCREEN PIXELS while
# cua-lite's canonical ``scroll`` counts wheel clicks, so both directions go
# through the one shared convention (:data:`PIXELS_PER_CLICK`). Sign is the
# grammar's own: "vertical positive scrolls up and negative scrolls down;
# horizontal positive scrolls right and negative scrolls left".
_SWIPE_AXIS_BY_DIRECTION = {
    "up": ("vertical", 1),
    "down": ("vertical", -1),
    "right": ("horizontal", 1),
    "left": ("horizontal", -1),
}
_SWIPE_DIRECTION_BY_AXIS = {
    ("vertical", 1): "up",
    ("vertical", -1): "down",
    ("horizontal", 1): "right",
    ("horizontal", -1): "left",
}


# =============================================================================
# UI-Venus-2 Computer (desktop) action space
# =============================================================================

@dataclasses.dataclass
class UIVenus2DesktopActionSpace(BaseActionSpace, key="ui_venus_2@desktop"):
    """UI-Venus-2's desktop operating-system grammar.

    Every row of the Computer system prompt is declared here as a ``@tool``
    method so the schema names, the declaration tables, and the renderer all
    agree on one spelling. The prompt itself is SFT text and is never rendered
    from these schemas — see :mod:`lite.agents.models.ui_venus_2.adapter`.
    """

    platform: str = "desktop"

    # -------------------------------------------------------------------------
    # Pointer
    # -------------------------------------------------------------------------
    # ``box`` is OPTIONAL on the click verbs: omitting it acts at the current
    # cursor position, which canonical ``click(coordinate=None)`` also spells.

    @staticmethod
    @tool(box="(x, y) coordinates; omit to act at the cursor.")
    def Click(box: list[int] | None = None) -> dict[str, Any]:
        """Perform a left-click."""
        return make_tool_call("Click", {"box": box})

    @staticmethod
    @tool(box="(x, y) coordinates; omit to act at the cursor.")
    def DoubleClick(box: list[int] | None = None) -> dict[str, Any]:
        """Perform a double-click, which selects a word in text."""
        return make_tool_call("DoubleClick", {"box": box})

    @staticmethod
    @tool(box="(x, y) coordinates; omit to act at the cursor.")
    def TripleClick(box: list[int] | None = None) -> dict[str, Any]:
        """Perform a triple-click, which selects a line."""
        return make_tool_call("TripleClick", {"box": box})

    @staticmethod
    @tool(box="(x, y) coordinates; omit to act at the cursor.")
    def RightClick(box: list[int] | None = None) -> dict[str, Any]:
        """Perform a right-click to open a context menu."""
        return make_tool_call("RightClick", {"box": box})

    @staticmethod
    @tool(box="(x, y) coordinates; omit to act at the cursor.")
    def MiddleClick(box: list[int] | None = None) -> dict[str, Any]:
        """Perform a middle-click."""
        return make_tool_call("MiddleClick", {"box": box})

    @staticmethod
    @tool(box="(x, y) coordinates to move the cursor to.")
    def Hover(box: list[int]) -> dict[str, Any]:
        """Move the cursor to the coordinates without clicking."""
        return make_tool_call("Hover", {"box": box})

    @staticmethod
    @tool(
        end="Ending (x, y) coordinates.",
        start="Starting (x, y) coordinates; omit to start at the cursor.",
    )
    def Drag(end: list[int], start: list[int] | None = None) -> dict[str, Any]:
        """Drag to the end coordinates with a fixed 0.5-second drag."""
        return make_tool_call("Drag", {"end": end, "start": start})

    @staticmethod
    @tool(
        amount="Signed scroll magnitude in pixels, -4096 to 4096.",
        axis="The scroll axis.",
    )
    def Swipe(amount: int, axis: Literal["vertical", "horizontal"]) -> dict[str, Any]:
        """Scroll at the current cursor position."""
        return make_tool_call("Swipe", {"amount": amount, "axis": axis})

    # -------------------------------------------------------------------------
    # Keyboard
    # -------------------------------------------------------------------------

    @staticmethod
    @tool(content="The text to type. Each newline presses Enter.")
    def Type(content: str) -> dict[str, Any]:
        """Type text into the focused field."""
        return make_tool_call("Type", {"content": content})

    @staticmethod
    @tool(
        keys="Key names to press together, e.g. ['ctrl', 'c'].",
        repeat="How many times to press the shortcut, 1 to 128.",
    )
    def Hotkey(keys: list[str], repeat: int | None = None) -> dict[str, Any]:
        """Press the listed keys as a keyboard shortcut."""
        return make_tool_call("Hotkey", {"keys": keys, "repeat": repeat})

    @staticmethod
    @tool(keys="Key names to press and hold.")
    def KeyDown(keys: list[str]) -> dict[str, Any]:
        """Press keys and keep them held across later actions."""
        return make_tool_call("KeyDown", {"keys": keys})

    @staticmethod
    @tool(keys="Key names to release.")
    def KeyUp(keys: list[str]) -> dict[str, Any]:
        """Release keys that were previously held."""
        return make_tool_call("KeyUp", {"keys": keys})

    # -------------------------------------------------------------------------
    # Half-press mouse
    # -------------------------------------------------------------------------

    @staticmethod
    @tool(box="(x, y) coordinates to move to first; omit to press at the cursor.")
    def MouseDown(box: list[int] | None = None) -> dict[str, Any]:
        """Press and hold the left mouse button across later actions."""
        return make_tool_call("MouseDown", {"box": box})

    @staticmethod
    @tool(box="(x, y) coordinates to move to first; omit to release at the cursor.")
    def MouseUp(box: list[int] | None = None) -> dict[str, Any]:
        """Release the left mouse button."""
        return make_tool_call("MouseUp", {"box": box})

    # -------------------------------------------------------------------------
    # Batch + terminal
    # -------------------------------------------------------------------------

    @staticmethod
    @tool(actions="2 to 32 non-Sequence actions to run open-loop, in order.")
    def Sequence(actions: list[dict[str, Any]]) -> dict[str, Any]:
        """Execute several actions in order as one open-loop model turn."""
        return make_tool_call("Sequence", {"actions": actions})

    @staticmethod
    @tool()
    def Wait() -> dict[str, Any]:
        """Wait for the current page, animation, or content to finish loading."""
        return make_tool_call("Wait")

    @staticmethod
    @tool(content="Details about how the task was completed.")
    def Finished(content: str | None = None) -> dict[str, Any]:
        """Mark the task as completed successfully."""
        return make_tool_call("Finished", {"content": content})

    @staticmethod
    @tool(content="Why the task cannot continue, or what is needed.")
    def CallUser(content: str | None = None) -> dict[str, Any]:
        """Request user takeover, or report that the task cannot be completed."""
        return make_tool_call("CallUser", {"content": content})

    # -------------------------------------------------------------------------
    # Native action / extra-tool declaration
    # -------------------------------------------------------------------------
    # UI-Venus-2 emits FLAT schemas, so the action layer and the extra-tool layer
    # are indistinguishable from class-visible data: the generated schemas for
    # ``Wait`` (an ACTION) and ``Finished`` (a finish TOOL) are identical modulo
    # name and description. Both are therefore DECLARED here.
    #
    # ``Sequence`` is absent from both tables on purpose: it is the batch
    # ENVELOPE, not an action or a tool, and it converts to and from the
    # canonical ``computer`` action-batch tool rather than to any single Lite
    # action name.

    LITE_ACTION_NAME_TO_UI_VENUS_2_PROVIDER_FLAT_TOOL_NAMES = {
        "click": ["Click", "DoubleClick", "TripleClick", "RightClick", "MiddleClick"],
        "mouse_move": ["Hover"],
        "drag": ["Drag"],
        "scroll": ["Swipe"],
        "type": ["Type"],
        "key": ["Hotkey"],
        "key_down": ["KeyDown"],
        "key_up": ["KeyUp"],
        "mouse_down": ["MouseDown"],
        "mouse_up": ["MouseUp"],
        "wait": ["Wait"],
    }
    #: ``Finished(content='x')`` parses to ``response`` and bare ``Finished()``
    #: to ``terminate``, so the entry spells BOTH — it must survive whenever
    #: EITHER is active, or the surface loses its only success verb. ``CallUser``
    #: is the failure/takeover channel, which lowers to a failed ``terminate``.
    UI_VENUS_2_PROVIDER_FLAT_TOOL_NAME_TO_EXTRA_TOOL_NAMES = {}

    # -------------------------------------------------------------------------
    # Tool call conversion
    # -------------------------------------------------------------------------

    def convert_tool_calls_to_agent(
        self,
        tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Canonical Lite → the Computer projection, folding a turn into ``Sequence``.

        The fold spans the WHOLE turn, not one canonical call: a turn that ends
        in a terminal reaches here as two Lite calls (a ``computer`` batch plus a
        standalone ``terminate``), and the grammar carries exactly one action per
        ``<action>`` block. ``Sequence`` is how it spells "several actions in one
        turn", and it accepts a trailing ``Finished`` / ``CallUser`` — which is
        the same ordering the canonical turn already has. One projection stays a
        bare call, because ``Sequence`` requires at least two actions.
        """
        result: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            result.extend(self._convert_single_to_agent(tool_call))
        if len(result) <= 1:
            return result
        return [{
            "name": SEQUENCE_NATIVE_NAME,
            "arguments": {"actions": result},
        }]

    def _convert_single_to_agent(self, tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        name = tool_call_name(tool_call)
        args = tool_call_arguments(tool_call)

        if name == "computer":
            result: list[dict[str, Any]] = []
            for child in args["actions"]:
                action = child["action"]
                action_args = {k: v for k, v in child.items() if k != "action"}
                result.extend(self._convert_single_to_agent(make_tool_call(action, action_args)))
            return result

        if name == "click":
            verb = _DESKTOP_CLICK_VERBS.get((args.get("button", "left"), args.get("clicks", 1)))
            if verb is None:
                raise ValueError(
                    f"UI-Venus-2 desktop cannot render click(button="
                    f"{args.get('button', 'left')!r}, clicks={args.get('clicks', 1)})"
                )
            verb_fn = getattr(UIVenus2DesktopActionSpace, verb)
            return [verb_fn(box=args.get("coordinate"))["function"]]

        if name == "mouse_move":
            return [UIVenus2DesktopActionSpace.Hover(box=args.get("coordinate"))["function"]]

        if name == "drag":
            if args.get("button", "left") != "left":
                raise ValueError(
                    "UI-Venus-2 desktop cannot render a non-left drag; its Drag "
                    "is left-button only"
                )
            return [UIVenus2DesktopActionSpace.Drag(
                end=args.get("coordinate"),
                start=args.get("start_coordinate"),
            )["function"]]

        if name == "scroll":
            axis_sign = _SWIPE_AXIS_BY_DIRECTION.get(args.get("direction", ""))
            if axis_sign is None:
                raise ValueError(
                    f"UI-Venus-2 desktop cannot render scroll(direction={args.get('direction')!r})"
                )
            axis, sign = axis_sign
            # ``coordinate`` has no wire slot: Swipe always scrolls at the
            # cursor. Emitting a Hover first would invent a second action the
            # model never asked for, so the anchor is dropped instead.
            return [UIVenus2DesktopActionSpace.Swipe(
                amount=sign * int(args.get("amount", 1)) * PIXELS_PER_CLICK,
                axis=axis,
            )["function"]]

        if name == "type":
            text = args.get("text", "")
            if args.get("press_enter"):
                text = f"{text}\n"
            return [UIVenus2DesktopActionSpace.Type(content=text)["function"]]

        if name == "key":
            return [UIVenus2DesktopActionSpace.Hotkey(keys=list(args.get("keys", [])))["function"]]

        if name == "key_down":
            return [UIVenus2DesktopActionSpace.KeyDown(keys=list(args.get("keys", [])))["function"]]

        if name == "key_up":
            return [UIVenus2DesktopActionSpace.KeyUp(keys=list(args.get("keys", [])))["function"]]

        if name in ("mouse_down", "mouse_up"):
            if args.get("button", "left") != "left":
                raise ValueError(
                    f"UI-Venus-2 desktop cannot render {name}(button="
                    f"{args.get('button')!r}); MouseDown/MouseUp are left-button only"
                )
            verb = "MouseDown" if name == "mouse_down" else "MouseUp"
            return [getattr(UIVenus2DesktopActionSpace, verb)()["function"]]

        if name == "wait":
            return [UIVenus2DesktopActionSpace.Wait()["function"]]

        if name == "terminate":
            if args.get("status") == "failure":
                return [UIVenus2DesktopActionSpace.CallUser(content=args.get("reason"))["function"]]
            return [UIVenus2DesktopActionSpace.Finished()["function"]]

        if name == "response":
            return [UIVenus2DesktopActionSpace.Finished(content=args.get("text", ""))["function"]]

        if name in LiteDesktopActionSet.get_action_names():
            raise ValueError(f"UI-Venus-2 desktop cannot render canonical tool {name!r}")
        logger.warning("Unknown CUA-lite action for UI-Venus-2 desktop: %s(%s)", name, args)
        return [{"name": name, "arguments": args}]

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Computer calls → canonical Lite, unfolding ``Sequence`` into the batch."""
        result: list[dict[str, Any]] = []
        for agent_tool_call in agent_tool_calls:
            for call in self._convert_single_from_agent(agent_tool_call):
                result.append(_wrap_action_call(call, "computer", _DESKTOP_ACTIONS))
        return merge_adjacent_lite_action_batches(result)

    def _convert_single_from_agent(self, agent_tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        name = agent_tool_call["name"]
        args = agent_tool_call["arguments"]

        if name == SEQUENCE_NATIVE_NAME:
            # Children are already bare projections; ``merge_adjacent_lite_action_batches``
            # in the caller re-joins the GUI ones into one ``computer`` batch and
            # leaves a trailing standalone tool (``Finished``) beside it.
            return [
                call
                for child in args.get("actions", [])
                for call in self._convert_single_from_agent(child)
            ]

        if name in _DESKTOP_CLICK_VERB_TO_CANONICAL:
            button, clicks = _DESKTOP_CLICK_VERB_TO_CANONICAL[name]
            return [LiteDesktopActionSpace.click(
                coordinate=_optional_point(args, "box"),
                button=button,
                clicks=clicks,
            )]

        if name == "Hover":
            return [LiteDesktopActionSpace.mouse_move(
                coordinate=required_coord(args.get("box"), dimensions=2, name="box"),
            )]

        if name == "Drag":
            return [LiteDesktopActionSpace.drag(
                coordinate=required_coord(args.get("end"), dimensions=2, name="end"),
                start_coordinate=_optional_point(args, "start"),
            )]

        if name == "Swipe":
            return [_swipe_to_canonical_scroll(args)]

        if name == "Type":
            content = args.get("content", "")
            if isinstance(content, str) and content.endswith("\n"):
                return [LiteDesktopActionSpace.type(text=content[:-1], press_enter=True)]
            return [LiteDesktopActionSpace.type(text=content)]

        if name in ("Hotkey", "KeyDown", "KeyUp"):
            keys = _require_keys(name, args.get("keys"))
            if name == "KeyDown":
                return [LiteDesktopActionSpace.key_down(keys=keys)]
            if name == "KeyUp":
                return [LiteDesktopActionSpace.key_up(keys=keys)]
            # ``repeat=N`` has no canonical carrier, so it fans out to N presses
            # rather than silently collapsing to one.
            repeat = args.get("repeat")
            times = repeat if isinstance(repeat, int) and repeat > 1 else 1
            return [LiteDesktopActionSpace.key(keys=keys) for _ in range(times)]

        if name in ("MouseDown", "MouseUp"):
            verb = (
                LiteDesktopActionSpace.mouse_down
                if name == "MouseDown"
                else LiteDesktopActionSpace.mouse_up
            )
            box = _optional_point(args, "box")
            # The optional ``box`` is a move THEN a press/release; canonical
            # mouse_down/up carry no coordinate, so the move becomes its own call.
            move = [LiteDesktopActionSpace.mouse_move(coordinate=box)] if box else []
            return [*move, verb(button="left")]

        if name == "Wait":
            return [LiteDesktopActionSpace.wait(duration=WAIT_SECONDS)]

        if name == "Finished":
            content = args.get("content")
            if content:
                return [LiteFinishToolSet.response(text=content)]
            return [LiteFinishToolSet.terminate(status="success")]

        if name == "CallUser":
            # ``reason`` is optional, so an empty CallUser stays reasonless
            # rather than gaining an invented string -- that keeps
            # ``terminate(status="failure")`` an exact round trip.
            return [LiteFinishToolSet.terminate(
                status="failure",
                reason=args.get("content") or None,
            )]

        if name not in LiteDesktopActionSet.get_action_names():
            logger.warning("Unknown UI-Venus-2 desktop action: %s(%s)", name, args)
        return [make_tool_call(name, args)]

    def format_tool_call_as_text(self, agent_tool_call: dict[str, Any]) -> str:
        """Render this family's bare ``{name, arguments}`` projection as text.

        Narrows the base contract on purpose: the input is what
        :meth:`convert_tool_calls_to_agent` produced, never a canonical Lite
        call. ``format_tool_calls_as_text`` inherits that choice element-wise.
        """
        return format_tool_call_as_text(agent_tool_call)


#: ``(button, clicks)`` → the Computer verb that spells it, and back.
_DESKTOP_CLICK_VERBS = {
    ("left", 1): "Click",
    ("left", 2): "DoubleClick",
    ("left", 3): "TripleClick",
    ("right", 1): "RightClick",
    ("middle", 1): "MiddleClick",
}
_DESKTOP_CLICK_VERB_TO_CANONICAL = {
    verb: key for key, verb in _DESKTOP_CLICK_VERBS.items()
}


def _optional_point(args: dict[str, Any], key: str) -> list[int] | None:
    """``None`` when the point argument is ABSENT, parsed when it is present.

    Not :func:`optional_coord`, which maps a malformed value onto the same
    ``None``. On this grammar an omitted point MEANS "act at the current cursor
    position", so treating a broken coordinate as absent would silently turn
    ``Click(box=(1,2,3))`` into a click somewhere else entirely. Absence is the
    only thing that yields ``None`` here; a present-but-broken value raises.
    """
    if args.get(key) is None:
        return None
    return required_coord(args[key], dimensions=2, name=key)


def _require_keys(name: str, keys: Any) -> list[str]:
    """Model-emitted ``keys`` as a non-empty list of names.

    Raises the action-space parse error, NOT a bare ``ValueError``: this is the
    parse boundary for untrusted model output, and only
    :class:`ModelToolCallParseError` is turned into a terminal parse-failure
    final by ``AdapterBasedAgent._parse_generation_response``. A bare
    ``ValueError`` here would propagate as a crash.
    """
    if isinstance(keys, str):
        keys = [keys]
    if not isinstance(keys, (list, tuple)) or not keys:
        raise ModelToolCallParseError(
            f"UI-Venus-2 {name} requires a non-empty keys list; got {keys!r}"
        )
    return [str(key) for key in keys]


def _swipe_to_canonical_scroll(args: dict[str, Any]) -> dict[str, Any]:
    """``Swipe(amount=±pixels, axis=…)`` → canonical ``scroll(direction, amount)``.

    The wire carries no anchor point, so the canonical call carries no
    ``coordinate``; the magnitude comes back through the shared
    pixels-per-wheel-click convention, floored at one click so a small
    pixel amount never becomes a no-op scroll.

    ``amount`` is REQUIRED: it is the only carrier of the DIRECTION on this
    wire, so defaulting it silently picks "up"/"right". A truncated
    ``Swipe(axis='vertical', amount=-12`` would scroll UP one click when the
    model asked to scroll DOWN twelve. Same rule, same reason, as
    ``geometry.required_scroll_pixels`` on the Qwen surfaces.

    Raises:
        ModelToolCallParseError: ``amount`` is missing or not an integer, or
            ``axis`` is not one of the two the grammar spells. All are
            model-chosen, so they are the parse boundary's error, not a bare
            ``ValueError``.
    """
    axis = args.get("axis", "vertical")
    if args.get("amount") is None:
        raise ModelToolCallParseError(
            "UI-Venus-2 Swipe requires an amount; it carries the scroll "
            "direction, so there is no safe default."
        )
    try:
        amount = int(args["amount"])
    except (TypeError, ValueError) as exc:
        raise ModelToolCallParseError(
            f"UI-Venus-2 Swipe requires an integer amount; got {args.get('amount')!r}"
        ) from exc
    sign = 1 if amount >= 0 else -1
    direction = _SWIPE_DIRECTION_BY_AXIS.get((axis, sign))
    if direction is None:
        raise ModelToolCallParseError(
            f"UI-Venus-2 Swipe axis must be vertical or horizontal; got {axis!r}"
        )
    return LiteDesktopActionSpace.scroll(
        direction=direction,
        amount=max(1, round(abs(amount) / PIXELS_PER_CLICK)),
    )


def _single_wire_action(
    projections: list[dict[str, Any]],
    *,
    surface: str,
) -> list[dict[str, Any]]:
    """Enforce this grammar's one-action-per-turn rule.

    The Computer grammar has ``Sequence`` for a multi-action turn; the browser
    and mobile grammars have nothing, and their upstream parsers accept exactly
    one ``<action>`` block. Rendering several actions into one block would emit
    text no UI-Venus-2 harness can read back, so it fails loudly instead.
    """
    if len(projections) <= 1:
        return projections
    raise ValueError(
        f"{surface} cannot render {len(projections)} actions in one turn: the "
        "grammar carries one action per <action> block and has no Sequence"
    )


# =============================================================================
# UI-Venus-2 Browser action space
# =============================================================================

@dataclasses.dataclass
class UIVenus2BrowserActionSpace(BaseActionSpace, key="ui_venus_2@browser"):
    """UI-Venus-2's browser grammar.

    Carries the canonical DESKTOP action vocabulary (cua-lite's browser platform
    uses desktop coordinate verbs) plus the browser-nav extra tools. Three rows
    have no canonical counterpart at all — ``GetUrl``, ``TakeNote`` and
    ``SelectOption`` — and pass through by name so the env answers them, the
    same treatment Fara gives ``pause_and_memorize_fact``.
    """

    platform: str = "browser"

    @staticmethod
    @tool(point="(x, y) coordinates of the tap target.")
    def Click(point: list[int]) -> dict[str, Any]:
        """Perform a tap at the specified screen coordinate."""
        return make_tool_call("Click", {"point": point})

    @staticmethod
    @tool(point="(x, y) coordinates of the double-tap target.")
    def DoubleClick(point: list[int]) -> dict[str, Any]:
        """Perform a double tap at the specified screen coordinate."""
        return make_tool_call("DoubleClick", {"point": point})

    @staticmethod
    @tool(point="(x, y) coordinates to hover over.")
    def Hover(point: list[int]) -> dict[str, Any]:
        """Hover to reveal tooltips or dropdown menus."""
        return make_tool_call("Hover", {"point": point})

    @staticmethod
    @tool(
        point="(x, y) coordinates of the long-press target.",
        duration="Hold time in seconds.",
    )
    def LongPress(point: list[int], duration: float | None = None) -> dict[str, Any]:
        """Press and hold the specified screen coordinate."""
        return make_tool_call("LongPress", {"point": point, "duration": duration})

    @staticmethod
    @tool(
        start="Starting (x, y) coordinates.",
        end="Ending (x, y) coordinates.",
    )
    def Drag(start: list[int], end: list[int]) -> dict[str, Any]:
        """Long-press at the start coordinate, then drag to the end coordinate."""
        return make_tool_call("Drag", {"start": start, "end": end})

    @staticmethod
    @tool(
        point="(x, y) coordinates to scroll at.",
        direction="The direction to scroll.",
    )
    def Scroll(
        direction: Literal["up", "down", "left", "right"],
        point: list[int] | None = None,
    ) -> dict[str, Any]:
        """Scroll the page at the given coordinate."""
        return make_tool_call("Scroll", {"point": point, "direction": direction})

    @staticmethod
    @tool(content="The text to type.")
    def Type(content: str) -> dict[str, Any]:
        """Enter text into the currently active input field."""
        return make_tool_call("Type", {"content": content})

    @staticmethod
    @tool(keys="Key names to press together, e.g. ['ctrl', 'c'].")
    def Hotkey(keys: list[str]) -> dict[str, Any]:
        """Press a combination of keys."""
        return make_tool_call("Hotkey", {"keys": keys})

    @staticmethod
    @tool(url="The URL to open.")
    def Launch(url: str) -> dict[str, Any]:
        """Navigate to the target URL."""
        return make_tool_call("Launch", {"url": url})

    @staticmethod
    @tool()
    def PressBack() -> dict[str, Any]:
        """Return to the previous page."""
        return make_tool_call("PressBack")

    @staticmethod
    @tool()
    def PressHome() -> dict[str, Any]:
        """Press the Home key to scroll to the top of the current page."""
        return make_tool_call("PressHome")

    @staticmethod
    @tool()
    def PressEnter() -> dict[str, Any]:
        """Perform an Enter key action."""
        return make_tool_call("PressEnter")

    @staticmethod
    @tool()
    def GetUrl() -> dict[str, Any]:
        """Get the URL of the current browser tab."""
        return make_tool_call("GetUrl")

    @staticmethod
    @tool(index="Zero-based index into the native select option list.")
    def SelectOption(index: int) -> dict[str, Any]:
        """Choose an option from the last clicked native HTML select element."""
        return make_tool_call("SelectOption", {"index": index})

    @staticmethod
    @tool(content="The fact to remember.")
    def TakeNote(content: str) -> dict[str, Any]:
        """Record important screenshot information to avoid forgetting it."""
        return make_tool_call("TakeNote", {"content": content})

    @staticmethod
    @tool()
    def Wait() -> dict[str, Any]:
        """Wait for the current page, animation, or content to finish loading."""
        return make_tool_call("Wait")

    @staticmethod
    @tool(content="Details about how the task was completed.")
    def Finished(content: str | None = None) -> dict[str, Any]:
        """Mark the task as completed and report the execution status."""
        return make_tool_call("Finished", {"content": content})

    @staticmethod
    @tool(content="Why the task cannot continue, or what is needed.")
    def CallUser(content: str | None = None) -> dict[str, Any]:
        """Request user takeover or additional information."""
        return make_tool_call("CallUser", {"content": content})

    # -------------------------------------------------------------------------
    # Native action / extra-tool declaration
    # -------------------------------------------------------------------------
    # Flat schemas — see :class:`UIVenus2DesktopActionSpace` for why these are
    # declared rather than derived. ``GetUrl`` / ``TakeNote`` / ``SelectOption``
    # appear in NEITHER table: they name no canonical action and no canonical
    # standalone tool, and both directions pass them through untouched.

    LITE_ACTION_NAME_TO_UI_VENUS_2_PROVIDER_FLAT_TOOL_NAMES = {
        "click": ["Click", "DoubleClick"],
        "mouse_move": ["Hover"],
        "drag": ["Drag"],
        "scroll": ["Scroll"],
        "type": ["Type"],
        "key": ["Hotkey", "PressHome", "PressEnter"],
        "wait": ["Wait"],
    }
    UI_VENUS_2_PROVIDER_FLAT_TOOL_NAME_TO_EXTRA_TOOL_NAMES = {
        "Launch": frozenset({"goto"}),
        "PressBack": frozenset({"back"}),
    }

    # -------------------------------------------------------------------------
    # Tool call conversion
    # -------------------------------------------------------------------------

    def _convert_single_to_agent(self, tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        name = tool_call_name(tool_call)
        args = tool_call_arguments(tool_call)

        if name == "computer":
            result: list[dict[str, Any]] = []
            for child in args["actions"]:
                action = child["action"]
                action_args = {k: v for k, v in child.items() if k != "action"}
                result.extend(self._convert_single_to_agent(make_tool_call(action, action_args)))
            return result

        if name == "click":
            button = args.get("button", "left")
            clicks = args.get("clicks", 1)
            if button != "left" or clicks not in (1, 2):
                raise ValueError(
                    f"UI-Venus-2 browser cannot render click(button={button!r}, "
                    f"clicks={clicks}): its grammar has only Click and DoubleClick"
                )
            if args.get("coordinate") is None:
                # Canonical ``click`` allows a bare click-at-cursor, but this
                # grammar's Click REQUIRES ``point=``. Rendering it anyway emits
                # ``Click()``, which this class's own parser then rejects with
                # "point is required" -- wire text we cannot read back. Fail
                # here, the way the ``drag`` branch below already does.
                raise ValueError(
                    "UI-Venus-2 browser cannot render click() without a "
                    "coordinate: its Click requires point=(x, y)"
                )
            verb = "DoubleClick" if clicks == 2 else "Click"
            verb_fn = getattr(UIVenus2BrowserActionSpace, verb)
            return [verb_fn(point=args["coordinate"])["function"]]

        if name == "mouse_move":
            return [UIVenus2BrowserActionSpace.Hover(point=args.get("coordinate"))["function"]]

        if name == "drag":
            start = args.get("start_coordinate")
            if start is None:
                raise ValueError(
                    "UI-Venus-2 browser cannot render drag without start_coordinate: "
                    "its Drag spells both endpoints"
                )
            return [UIVenus2BrowserActionSpace.Drag(
                start=start, end=args.get("coordinate"),
            )["function"]]

        if name == "scroll":
            return [UIVenus2BrowserActionSpace.Scroll(
                direction=args.get("direction", "down"),
                point=args.get("coordinate"),
            )["function"]]

        if name == "type":
            if args.get("press_enter"):
                raise ValueError(
                    "UI-Venus-2 browser cannot render type(press_enter=True): its "
                    "Type does not submit and the grammar allows one action per turn"
                )
            return [UIVenus2BrowserActionSpace.Type(content=args.get("text", ""))["function"]]

        if name == "key":
            return [UIVenus2BrowserActionSpace.Hotkey(keys=list(args.get("keys", [])))["function"]]

        if name == "wait":
            return [UIVenus2BrowserActionSpace.Wait()["function"]]

        if name == "terminate":
            if args.get("status") == "failure":
                return [UIVenus2BrowserActionSpace.CallUser(content=args.get("reason"))["function"]]
            return [UIVenus2BrowserActionSpace.Finished()["function"]]

        if name == "response":
            return [UIVenus2BrowserActionSpace.Finished(content=args.get("text", ""))["function"]]

        if name == "goto":
            return [UIVenus2BrowserActionSpace.Launch(url=args.get("url", ""))["function"]]

        if name == "back":
            return [UIVenus2BrowserActionSpace.PressBack()["function"]]

        if name in LiteDesktopActionSet.get_action_names():
            raise ValueError(f"UI-Venus-2 browser cannot render canonical tool {name!r}")
        logger.warning("Unknown CUA-lite action for UI-Venus-2 browser: %s(%s)", name, args)
        return [{"name": name, "arguments": args}]

    def convert_tool_calls_to_agent(
        self,
        tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Canonical Lite → the Browser projection, one action per turn."""
        result: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            result.extend(self._convert_single_to_agent(tool_call))
        return _single_wire_action(result, surface="UI-Venus-2 browser")

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Browser calls → canonical Lite calls."""
        result: list[dict[str, Any]] = []
        for agent_tool_call in agent_tool_calls:
            for call in self._convert_single_from_agent(agent_tool_call):
                result.append(_wrap_action_call(call, "computer", _BROWSER_ACTIONS))
        return merge_adjacent_lite_action_batches(result)

    def _convert_single_from_agent(self, agent_tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        name = agent_tool_call["name"]
        args = agent_tool_call["arguments"]

        if name in ("Click", "DoubleClick"):
            return [LiteDesktopActionSpace.click(
                coordinate=required_coord(args.get("point"), dimensions=2, name="point"),
                clicks=2 if name == "DoubleClick" else 1,
            )]

        if name == "Hover":
            return [LiteDesktopActionSpace.mouse_move(
                coordinate=required_coord(args.get("point"), dimensions=2, name="point"),
            )]

        if name == "Drag":
            return [LiteDesktopActionSpace.drag(
                start_coordinate=required_coord(args.get("start"), dimensions=2, name="start"),
                coordinate=required_coord(args.get("end"), dimensions=2, name="end"),
            )]

        if name == "Scroll":
            # The wire carries a direction and an optional anchor but no
            # magnitude, so parse fixes the canonical default.
            return [LiteDesktopActionSpace.scroll(
                direction=args.get("direction", "down"),
                amount=5,
                coordinate=_optional_point(args, "point"),
            )]

        if name == "Type":
            return [LiteDesktopActionSpace.type(text=args.get("content", ""))]

        if name == "Hotkey":
            return [LiteDesktopActionSpace.key(keys=_require_keys(name, args.get("keys")))]

        if name == "PressEnter":
            return [LiteDesktopActionSpace.key(keys=["enter"])]

        if name == "PressHome":
            # The browser prompt spells this "Press the Home key to scroll to
            # the top of the current page" — a keystroke, not a device button.
            return [LiteDesktopActionSpace.key(keys=["home"])]

        if name == "Wait":
            return [LiteDesktopActionSpace.wait(duration=WAIT_SECONDS)]

        if name == "Finished":
            content = args.get("content")
            if content:
                return [LiteFinishToolSet.response(text=content)]
            return [LiteFinishToolSet.terminate(status="success")]

        if name == "CallUser":
            # ``reason`` is optional, so an empty CallUser stays reasonless
            # rather than gaining an invented string -- that keeps
            # ``terminate(status="failure")`` an exact round trip.
            return [LiteFinishToolSet.terminate(
                status="failure",
                reason=args.get("content") or None,
            )]

        if name == "Launch":
            return [LiteBrowserNavToolSet.goto(url=args.get("url", ""))]

        if name == "PressBack":
            return [LiteBrowserNavToolSet.back()]

        # ``GetUrl`` / ``TakeNote`` / ``SelectOption`` / ``LongPress`` name no
        # canonical verb. Provider output is kept intact; env feedback owns
        # unknown executable names.
        if name not in LiteDesktopActionSet.get_action_names():
            logger.warning("Unknown UI-Venus-2 browser action: %s(%s)", name, args)
        return [make_tool_call(name, args)]

    def format_tool_call_as_text(self, agent_tool_call: dict[str, Any]) -> str:
        """Render this family's bare ``{name, arguments}`` projection as text."""
        return format_tool_call_as_text(agent_tool_call)


# =============================================================================
# UI-Venus-2 Mobile action space
# =============================================================================

@dataclasses.dataclass
class UIVenus2MobileActionSpace(BaseActionSpace, key="ui_venus_2@mobile"):
    """UI-Venus-2's Android grammar.

    Two verbs are unique to this surface: ``Answer`` is the reply channel
    (canonical ``response``), distinct from ``Finished``, and ``GetScreenshot``
    is a real device action rather than an implicit one.
    """

    platform: str = "mobile"

    @staticmethod
    @tool(point="(x, y) coordinates of the tap target.")
    def Click(point: list[int]) -> dict[str, Any]:
        """Perform a tap at the specified screen coordinate."""
        return make_tool_call("Click", {"point": point})

    @staticmethod
    @tool(point="(x, y) coordinates of the double-tap target.")
    def DoubleClick(point: list[int]) -> dict[str, Any]:
        """Perform a double tap at the specified screen coordinate."""
        return make_tool_call("DoubleClick", {"point": point})

    @staticmethod
    @tool(point="(x, y) coordinates of the long-press target.")
    def LongPress(point: list[int]) -> dict[str, Any]:
        """Long-press to trigger options such as copy, forward or delete."""
        return make_tool_call("LongPress", {"point": point})

    @staticmethod
    @tool(
        start="Starting (x, y) coordinates.",
        end="Ending (x, y) coordinates.",
    )
    def Swipe(start: list[int], end: list[int]) -> dict[str, Any]:
        """Swipe from the start coordinate to the end coordinate."""
        return make_tool_call("Swipe", {"start": start, "end": end})

    @staticmethod
    @tool(
        start="Starting (x, y) coordinates.",
        end="Ending (x, y) coordinates.",
    )
    def Drag(start: list[int], end: list[int]) -> dict[str, Any]:
        """Long-press at the start coordinate for a few seconds, then drag."""
        return make_tool_call("Drag", {"start": start, "end": end})

    @staticmethod
    @tool(content="The text to type.")
    def Type(content: str) -> dict[str, Any]:
        """Enter text into the currently active input field."""
        return make_tool_call("Type", {"content": content})

    @staticmethod
    @tool(app="Name of the app to launch.")
    def LaunchApp(app: str) -> dict[str, Any]:
        """Launch the target app when it is not visible on screen."""
        return make_tool_call("LaunchApp", {"app": app})

    @staticmethod
    @tool()
    def PressBack() -> dict[str, Any]:
        """Return to the previous screen."""
        return make_tool_call("PressBack")

    @staticmethod
    @tool()
    def PressHome() -> dict[str, Any]:
        """Return to the system home screen."""
        return make_tool_call("PressHome")

    @staticmethod
    @tool()
    def PressEnter() -> dict[str, Any]:
        """Perform an Enter key action."""
        return make_tool_call("PressEnter")

    @staticmethod
    @tool()
    def PressRecent() -> dict[str, Any]:
        """Open the system recent-apps screen."""
        return make_tool_call("PressRecent")

    @staticmethod
    @tool()
    def GetScreenshot() -> dict[str, Any]:
        """Take a screenshot and save it to the device photo album."""
        return make_tool_call("GetScreenshot")

    @staticmethod
    @tool()
    def Wait() -> dict[str, Any]:
        """Wait for the current page, animation, or content to finish loading."""
        return make_tool_call("Wait")

    @staticmethod
    @tool(content="The answer to show the user.")
    def Answer(content: str) -> dict[str, Any]:
        """Answer the user's question as requested."""
        return make_tool_call("Answer", {"content": content})

    @staticmethod
    @tool(content="Details about how the task was completed.")
    def Finished(content: str | None = None) -> dict[str, Any]:
        """Mark the task as completed and report the execution status."""
        return make_tool_call("Finished", {"content": content})

    @staticmethod
    @tool(content="Why the task cannot continue, or what is needed.")
    def CallUser(content: str | None = None) -> dict[str, Any]:
        """Request user takeover or additional information."""
        return make_tool_call("CallUser", {"content": content})

    # -------------------------------------------------------------------------
    # Native action / extra-tool declaration
    # -------------------------------------------------------------------------
    # Flat schemas — see :class:`UIVenus2DesktopActionSpace`.

    LITE_ACTION_NAME_TO_UI_VENUS_2_PROVIDER_FLAT_TOOL_NAMES = {
        "tap": ["Click", "DoubleClick"],
        "long_press": ["LongPress"],
        "swipe": ["Swipe"],
        "drag": ["Drag"],
        "type": ["Type"],
        "system_button": ["PressBack", "PressHome", "PressEnter", "PressRecent"],
        "screenshot": ["GetScreenshot"],
        "wait": ["Wait"],
    }
    #: Mobile is the one surface with a DEDICATED answer verb, so ``Finished``
    #: keeps a single meaning here: task complete.
    UI_VENUS_2_PROVIDER_FLAT_TOOL_NAME_TO_EXTRA_TOOL_NAMES = {
        "LaunchApp": frozenset({"open_app"}),
    }

    # -------------------------------------------------------------------------
    # Tool call conversion
    # -------------------------------------------------------------------------

    def _convert_single_to_agent(self, tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        name = tool_call_name(tool_call)
        args = tool_call_arguments(tool_call)

        if name == "mobile":
            result: list[dict[str, Any]] = []
            for child in args["actions"]:
                action = child["action"]
                action_args = {k: v for k, v in child.items() if k != "action"}
                result.extend(self._convert_single_to_agent(make_tool_call(action, action_args)))
            return result

        if name == "tap":
            clicks = args.get("clicks", 1)
            if clicks not in (1, 2):
                raise ValueError(
                    f"UI-Venus-2 mobile cannot render tap(clicks={clicks}): its "
                    "grammar spells only single and double tap"
                )
            verb = "DoubleClick" if clicks == 2 else "Click"
            verb_fn = getattr(UIVenus2MobileActionSpace, verb)
            return [verb_fn(point=args.get("coordinate"))["function"]]

        if name == "long_press":
            # ``duration`` has no wire slot: LongPress takes only a point.
            return [UIVenus2MobileActionSpace.LongPress(point=args.get("coordinate"))["function"]]

        if name in ("swipe", "drag"):
            verb = "Swipe" if name == "swipe" else "Drag"
            return [getattr(UIVenus2MobileActionSpace, verb)(
                start=args.get("start_coordinate"),
                end=args.get("coordinate"),
            )["function"]]

        if name == "type":
            return [UIVenus2MobileActionSpace.Type(content=args.get("text", ""))["function"]]

        if name == "system_button":
            button = args.get("button", "")
            verb = _MOBILE_SYSTEM_BUTTON_TO_NATIVE.get(button)
            if verb is None:
                raise ValueError(
                    f"UI-Venus-2 mobile cannot render system_button {button!r}; "
                    f"expected one of {sorted(_MOBILE_SYSTEM_BUTTON_TO_NATIVE)}"
                )
            return [getattr(UIVenus2MobileActionSpace, verb)()["function"]]

        if name == "screenshot":
            return [UIVenus2MobileActionSpace.GetScreenshot()["function"]]

        if name == "wait":
            return [UIVenus2MobileActionSpace.Wait()["function"]]

        if name == "terminate":
            if args.get("status") == "failure":
                return [UIVenus2MobileActionSpace.CallUser(content=args.get("reason"))["function"]]
            return [UIVenus2MobileActionSpace.Finished()["function"]]

        if name == "response":
            return [UIVenus2MobileActionSpace.Answer(content=args.get("text", ""))["function"]]

        if name == "open_app":
            return [UIVenus2MobileActionSpace.LaunchApp(app=args.get("app_name", ""))["function"]]

        if name in LiteMobileActionSet.get_action_names():
            raise ValueError(f"UI-Venus-2 mobile cannot render canonical tool {name!r}")
        logger.warning("Unknown CUA-lite mobile action for UI-Venus-2: %s(%s)", name, args)
        return [{"name": name, "arguments": args}]

    def convert_tool_calls_to_agent(
        self,
        tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Canonical Lite → the Mobile projection, one action per turn."""
        result: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            result.extend(self._convert_single_to_agent(tool_call))
        return _single_wire_action(result, surface="UI-Venus-2 mobile")

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Mobile calls → canonical Lite calls."""
        result: list[dict[str, Any]] = []
        for agent_tool_call in agent_tool_calls:
            for call in self._convert_single_from_agent(agent_tool_call):
                result.append(_wrap_action_call(call, "mobile", _MOBILE_ACTIONS))
        return merge_adjacent_lite_action_batches(result)

    def _convert_single_from_agent(self, agent_tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        name = agent_tool_call["name"]
        args = agent_tool_call["arguments"]

        if name in ("Click", "DoubleClick"):
            return [LiteMobileActionSpace.tap(
                coordinate=required_coord(args.get("point"), dimensions=2, name="point"),
                clicks=2 if name == "DoubleClick" else 1,
            )]

        if name == "LongPress":
            return [LiteMobileActionSpace.long_press(
                coordinate=required_coord(args.get("point"), dimensions=2, name="point"),
            )]

        if name in ("Swipe", "Drag"):
            verb = LiteMobileActionSpace.swipe if name == "Swipe" else LiteMobileActionSpace.drag
            return [verb(
                start_coordinate=required_coord(args.get("start"), dimensions=2, name="start"),
                coordinate=required_coord(args.get("end"), dimensions=2, name="end"),
            )]

        if name == "Type":
            return [LiteMobileActionSpace.type(text=args.get("content", ""))]

        if name in _MOBILE_NATIVE_TO_SYSTEM_BUTTON:
            return [LiteMobileActionSpace.system_button(
                button=_MOBILE_NATIVE_TO_SYSTEM_BUTTON[name],
            )]

        if name == "GetScreenshot":
            return [LiteMobileActionSpace.screenshot()]

        if name == "Wait":
            return [LiteMobileActionSpace.wait(duration=WAIT_SECONDS)]

        if name == "Answer":
            return [LiteFinishToolSet.response(text=args.get("content", ""))]

        if name == "Finished":
            return [LiteFinishToolSet.terminate(status="success")]

        if name == "CallUser":
            # ``reason`` is optional, so an empty CallUser stays reasonless
            # rather than gaining an invented string -- that keeps
            # ``terminate(status="failure")`` an exact round trip.
            return [LiteFinishToolSet.terminate(
                status="failure",
                reason=args.get("content") or None,
            )]

        if name == "LaunchApp":
            return [LiteAppLaunchToolSet.open_app(app_name=args.get("app") or "")]

        if name not in LiteMobileActionSet.get_action_names():
            logger.warning("Unknown UI-Venus-2 mobile action: %s(%s)", name, args)
        return [make_tool_call(name, args)]

    def format_tool_call_as_text(self, agent_tool_call: dict[str, Any]) -> str:
        """Render this family's bare ``{name, arguments}`` projection as text."""
        return format_tool_call_as_text(agent_tool_call)


#: The four device buttons UI-Venus-2's mobile grammar spells. ``Menu`` is
#: canonical but has no row here, so rendering it raises.
_MOBILE_SYSTEM_BUTTON_TO_NATIVE = {
    "Back": "PressBack",
    "Home": "PressHome",
    "Enter": "PressEnter",
    "Recent": "PressRecent",
}
_MOBILE_NATIVE_TO_SYSTEM_BUTTON = {
    native: button for button, native in _MOBILE_SYSTEM_BUTTON_TO_NATIVE.items()
}


# =============================================================================
# UI-Venus-2 Grounding (single-step point) action space
# =============================================================================
#
# The grounding wire is NOT a call: the checkpoint answers with a bare ``[x,y]``
# list normalized to 1000, or the trained refusal marker ``[-1,-1]``
# (``models/grounding/ui_venus2_gd.py``). The adapter owns that text; this class
# only routes it to and from the canonical grounding vocabulary.

#: Synthetic native name for the ``[x,y]`` answer. The grounding prompt
#: advertises no tools at all, so this spelling exists only here and in the
#: grounding adapter's text codec.
GROUNDING_POINT_NATIVE_NAME = "point"


@dataclasses.dataclass
class UIVenus2GroundingPointActionSpace(
    BaseActionSpace, key=r"ui_venus_2@(desktop|browser|mobile)@point",
):
    """UI-Venus-2 grounding surface, shared by every platform.

    Upstream ships ONE grounding harness — ``eval_multi_benchmark.py`` runs
    ScreenSpot-Pro, OSWorld-G and the mobile splits of VenusBench-GD through the
    same prompt — so one regex key serves all three platforms.

    Coordinate frame is identity: the checkpoint normalizes against 1000, which
    is cua-lite's canonical [0, 1000].
    """

    platform: str | None = None
    _LITE_FORMAT_ACTION_NAME = next(iter(LitePointActionSpace.get_action_names()))

    LITE_ACTION_NAME_TO_UI_VENUS_2_PROVIDER_FLAT_TOOL_NAMES = {
        "point": [GROUNDING_POINT_NATIVE_NAME],
    }

    @classmethod
    def get_tool_schemas(cls, include: list[str] | None = None) -> list[dict[str, Any]]:
        """No schemas: the ``[x,y]`` output format lives entirely in the prompt."""
        return []

    def convert_tool_calls_to_agent(
        self,
        tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """cua-lite ``point(coord)`` → synthetic ``point(box=coord)``.

        ``report_infeasible`` passes through as a bare projection; the adapter
        renders it as the ``[-1,-1]`` marker.
        """
        result: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            if tool_call_name(tool_call) == "point":
                coordinate = tool_call_arguments(tool_call)["coordinate"]
                result.append({
                    "name": GROUNDING_POINT_NATIVE_NAME,
                    "arguments": {"box": list(coordinate)},
                })
            else:
                result.extend(convert_non_point_call_for_grounding_space(
                    tool_call, surface="UI-Venus-2 grounding (point) action_space",
                ))
        return result

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Synthetic ``point(box=coord)`` → cua-lite ``point(coord)``.

        Anything else — ``report_infeasible``, an env extra, an off-schema name
        — passes through as a canonical standalone call, the same answer every
        other ``@point`` space gives.
        """
        result: list[dict[str, Any]] = []
        for agent_tool_call in agent_tool_calls:
            name = agent_tool_call["name"]
            args = agent_tool_call["arguments"]
            if name == GROUNDING_POINT_NATIVE_NAME:
                result.append(LitePointActionSpace.point(
                    coordinate=required_coord(args.get("box"), dimensions=2, name="box"),
                ))
            else:
                result.append(make_tool_call(name, args))
        return result


__all__ = [
    "GROUNDING_POINT_NATIVE_NAME",
    "SEQUENCE_NATIVE_NAME",
    "WAIT_SECONDS",
    "UIVenus2BrowserActionSpace",
    "UIVenus2DesktopActionSpace",
    "UIVenus2GroundingPointActionSpace",
    "UIVenus2MobileActionSpace",
    "format_tool_call_as_text",
    "parse_action_text",
]
