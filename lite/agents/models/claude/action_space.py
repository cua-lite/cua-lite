"""
Claude action spaces for CUA-Lite desktop and mobile agents.

Desktop maps between CUA-lite normalized [0, 1000] coordinates and Claude's
pixel-based native computer tool actions. Mobile exposes provider-flat function tools
with absolute pixel coordinates.

Claude computer tool actions use absolute pixel coordinates.
CUA-lite uses normalized [0, 1000] coordinates.

Conversion example:
    from lite.agents.models.claude.action_space import ClaudeDesktopActionSpace

    action_space = ClaudeDesktopActionSpace()

    # Convert Claude tool_use -> CUA-lite
    claude_actions = [{"action": "left_click", "coordinate": [512, 384]}]
    lite_actions = action_space.convert_tool_calls_from_agent(
        claude_actions,
        resolution=(1024, 768),
    )

    # Convert CUA-lite -> Claude tool_use action dicts
    claude_actions = action_space.convert_tool_calls_to_agent(lite_actions, resolution=(1024, 768))

Provider tool assembly lives in ``ClaudeDesktopUseAgent`` because Anthropic's
native computer-use schema is selected by model version.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import math
from typing import Any, Literal

from lite.agents.core.action_space import BaseActionSpace
from lite.agents.core.action_space.base import (
    LiteDesktopActionSpace,
    LiteMobileActionSpace,
    LitePointActionSpace,
)
from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.agents.core.action_space.utils.grounding_point import (
    convert_non_point_call_for_grounding_space,
)
from lite.core.tools.action_space import (
    merge_adjacent_lite_action_batches,
    normalize_keys,
    pixel_to_norm,
)
from lite.core.tools.action_space.batches import LITE_MOBILE_ACTION_BATCH_TOOL_NAME
from lite.core.tools.action_space.duration import duration_description
from lite.core.tools.action_space.geometry import strict_norm_to_pixel
from lite.core.tools.calls import (
    make_tool_call,
    tool_call_arguments,
    tool_call_name,
)
from lite.core.tools.schemas import (
    tool,
    tool_call_satisfies_schema,
    tool_schema_parameters,
)

logger = logging.getLogger(__name__)

# Every ``(button, clicks)`` pair the Claude computer wire can actually spell,
# mapped to the native action name. Each native click action pins BOTH facts at
# once — ``middle_click`` is single by definition, ``triple_click`` is left by
# definition — so the four combinations outside this table (right/middle with 2
# or 3 clicks) have no faithful spelling. Emitting the nearest one silently
# changes the action, which is the bug this table exists to prevent.
# Keep in lockstep with the parse side in ``_convert_single_from_agent``.
_CLICK_WIRE_ACTIONS: dict[tuple[str, int], str] = {
    ("left", 1): "left_click",
    ("left", 2): "double_click",
    ("left", 3): "triple_click",
    ("right", 1): "right_click",
    ("middle", 1): "middle_click",
}

_CLAUDE_MOBILE_COORD_KEYS_BY_ACTION: dict[str, tuple[str, ...]] = {
    "tap": ("coordinate",),
    "long_press": ("coordinate",),
    "pinch": ("coordinate",),
    "swipe": ("start_coordinate", "coordinate"),
    "drag": ("start_coordinate", "coordinate"),
}


def _is_coordinate_number(value: Any) -> bool:
    """True for a finite model-supplied number.

    ``json.loads`` parses ``NaN``/``Infinity`` into Python floats, so the tool
    arguments this parser reads can carry a non-finite coordinate. ``int(nan)``
    raises ``ValueError`` and ``int(inf)`` raises ``OverflowError``, neither of
    which is the owned parse error, so a non-finite value is rejected here as
    malformed model output — matching ``action_space.utils.geometry``'s finite
    check used by the other model families.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        # ``json.loads`` also spells arbitrary-precision integer literals, and
        # one too large for a float cannot reach a pixel surface either.
        return False


def _is_coordinate_pair(value: Any) -> bool:
    """True for exactly two finite numbers — the only shape ``[x, y]`` spells.

    A longer list is malformed model output, not a point carrying spare values:
    reading its first two entries would turn a wrong-shaped box into a
    valid-looking tap.
    """
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(_is_coordinate_number(item) for item in value)
    )


def _require_number(value: Any, *, action_type: str, key: str) -> int | float:
    """A model-supplied numeric action argument, or the owned parse error.

    ``int()``/``float()`` on raw model JSON raises a bare ``TypeError`` for
    ``null`` and a bare ``ValueError`` for a non-numeric string. Both are
    malformed model output, so the parse boundary names them here instead of
    leaving callers to classify an incidental exception type.
    """
    if not _is_coordinate_number(value):
        raise ModelToolCallParseError(f"{action_type} requires numeric {key}")
    return value


# =============================================================================
# Coordinate Helpers
# =============================================================================


def _require_resolution(resolution: tuple[int, int] | None) -> tuple[int, int]:
    """Require ``resolution=(w, h)`` — the frame dims the model saw. Claude's
    pixel-coord actions only round-trip correctly when (w, h) matches that frame;
    a silent fallback (e.g. ``(1024, 768)``) would mis-scale every coordinate when
    the real frame differs. The sample loop always passes it, so this should never
    be None in production (mirrors GPT's ``_require_resolution``)."""
    if resolution is None:
        raise ValueError(
            "resolution=(w, h) is required for Claude coordinate conversion "
            "(the model's frame dims); got None"
        )
    return resolution


# =============================================================================
# Claude Desktop Action Space
# =============================================================================


@dataclasses.dataclass
class ClaudeDesktopActionSpace(BaseActionSpace, key=r"claude@(desktop|browser)"):
    """
    Claude computer-use action space for desktop interactions.

    Claude uses absolute pixel coordinates. CUA-lite uses normalized [0, 1000].
    Coordinate conversion requires passing ``resolution=(width, height)`` to
    ``convert_tool_calls_to_agent()`` and ``convert_tool_calls_from_agent()``.

    Claude tool actions:
        left_click, right_click, middle_click, double_click,
        type, key, scroll, left_click_drag, mouse_move,
        screenshot, wait, left_mouse_down, left_mouse_up.
    """

    platform: str = "desktop"

    @classmethod
    def get_tool_schemas(cls, include: list[str] | None = None) -> list[dict[str, Any]]:
        """Claude desktop uses Anthropic's native computer tool, not function schemas.

        This action space is the native-computer ↔ Lite conversion surface.
        Provider tool assembly happens in ``ClaudeDesktopUseAgent._build_tools``.
        """
        return []

    @classmethod
    def filter_tool_schemas_for_valid_actions(
        cls,
        schemas: list[dict[str, Any]],
        valid_actions: list[str],
    ) -> list[dict[str, Any]]:
        """Gate Anthropic's opaque versioned ``computer_*`` native schemas.

        Claude's native computer schema is selected by the provider-facing
        agent and has no request-side per-action enum, so valid-actions can
        only keep or drop the whole native tool. Grounding subclasses still
        delegate their provider-flat click schema to the normal
        action-space-owned named-schema filter.
        """
        result: list[dict[str, Any]] = []
        for schema in schemas:
            if str(schema.get("type", "")).startswith("computer_"):
                if valid_actions:
                    result.append(copy.deepcopy(schema))
                continue
            result.extend(
                super().filter_tool_schemas_for_valid_actions(
                    [schema],
                    valid_actions,
                )
            )
        return result

    # -------------------------------------------------------------------------
    # Mouse Click Actions
    # -------------------------------------------------------------------------

    @staticmethod
    @tool(coordinate="[x, y] pixel coordinates.")
    def left_click(coordinate: list[int]) -> dict[str, Any]:
        """Click the left mouse button at the specified coordinates."""
        return make_tool_call("left_click", {"coordinate": coordinate})

    @staticmethod
    @tool(coordinate="[x, y] pixel coordinates.")
    def right_click(coordinate: list[int]) -> dict[str, Any]:
        """Click the right mouse button at the specified coordinates."""
        return make_tool_call("right_click", {"coordinate": coordinate})

    @staticmethod
    @tool(coordinate="[x, y] pixel coordinates.")
    def middle_click(coordinate: list[int]) -> dict[str, Any]:
        """Click the middle mouse button at the specified coordinates."""
        return make_tool_call("middle_click", {"coordinate": coordinate})

    @staticmethod
    @tool(coordinate="[x, y] pixel coordinates.")
    def double_click(coordinate: list[int]) -> dict[str, Any]:
        """Double-click the left mouse button at the specified coordinates."""
        return make_tool_call("double_click", {"coordinate": coordinate})

    # -------------------------------------------------------------------------
    # Keyboard Actions
    # -------------------------------------------------------------------------

    @staticmethod
    @tool(text="The text to type.")
    def type(text: str) -> dict[str, Any]:
        """Type text content."""
        return make_tool_call("type", {"text": text})

    @staticmethod
    @tool(text="Keys joined by '+', e.g. 'ctrl+c'.")
    def key(text: str) -> dict[str, Any]:
        """Press a keyboard key combination."""
        return make_tool_call("key", {"text": text})

    # -------------------------------------------------------------------------
    # Scroll
    # -------------------------------------------------------------------------

    @staticmethod
    @tool(
        coordinate="[x, y] pixel coordinates to scroll at.",
        scroll_direction="The direction to scroll.",
        scroll_amount="Number of scroll units.",
    )
    def scroll(
        coordinate: list[int],
        scroll_direction: Literal["up", "down", "left", "right"] = "down",
        scroll_amount: int = 3,
    ) -> dict[str, Any]:
        """Scroll in the specified direction at the given coordinates."""
        return make_tool_call(
            "scroll",
            {
                "coordinate": coordinate,
                "scroll_direction": scroll_direction,
                "scroll_amount": scroll_amount,
            },
        )

    # -------------------------------------------------------------------------
    # Drag & Move
    # -------------------------------------------------------------------------

    @staticmethod
    @tool(
        start_coordinate="Starting [x, y] pixel coordinates.",
        end_coordinate="Ending [x, y] pixel coordinates.",
    )
    def left_click_drag(start_coordinate: list[int], end_coordinate: list[int]) -> dict[str, Any]:
        """Drag from start coordinates to end coordinates."""
        return make_tool_call(
            "left_click_drag",
            {
                "start_coordinate": start_coordinate,
                "end_coordinate": end_coordinate,
            },
        )

    @staticmethod
    @tool(coordinate="[x, y] pixel coordinates.")
    def mouse_move(coordinate: list[int]) -> dict[str, Any]:
        """Move the mouse cursor to specified coordinates."""
        return make_tool_call("mouse_move", {"coordinate": coordinate})

    # -------------------------------------------------------------------------
    # Mouse Down/Up
    # -------------------------------------------------------------------------

    @staticmethod
    @tool(coordinate="[x, y] pixel coordinates.")
    def left_mouse_down(coordinate: list[int] | None = None) -> dict[str, Any]:
        """Press the left mouse button down without releasing."""
        return make_tool_call("left_mouse_down", {"coordinate": coordinate})

    @staticmethod
    @tool(coordinate="[x, y] pixel coordinates.")
    def left_mouse_up(coordinate: list[int] | None = None) -> dict[str, Any]:
        """Release the left mouse button."""
        return make_tool_call("left_mouse_up", {"coordinate": coordinate})

    # -------------------------------------------------------------------------
    # Utility Actions
    # -------------------------------------------------------------------------

    @staticmethod
    @tool()
    def screenshot() -> dict[str, Any]:
        """Take a screenshot."""
        return make_tool_call("screenshot")

    @staticmethod
    @tool()
    def wait() -> dict[str, Any]:
        """Wait for the screen to update."""
        return make_tool_call("wait")

    # -------------------------------------------------------------------------
    # Tool Call Conversion: Claude -> CUA-lite
    # -------------------------------------------------------------------------

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Convert Claude computer tool actions to CUA-lite format.

        Args:
            agent_tool_calls: List of Claude tool_use dicts.
                Each has ``{"action": "left_click", "coordinate": [x, y], ...}``.
            **kwargs: Must include ``resolution=(width, height)`` for coordinate conversion.
        """
        resolution = kwargs.get("resolution")
        result = []
        for tc in agent_tool_calls:
            result.extend(
                self._convert_single_from_agent(
                    tc,
                    resolution,
                )
            )
        return merge_adjacent_lite_action_batches(result)

    def _convert_single_from_agent(
        self,
        action: dict[str, Any],
        resolution: tuple[int, int] | None,
    ) -> list[dict[str, Any]]:
        """Convert a single Claude action to CUA-lite format."""
        action_type = action.get("action", "")

        def _coord(key: str = "coordinate", *, required: bool = False) -> list[int] | None:
            coord = action.get(key)
            if coord is not None:
                if not _is_coordinate_pair(coord):
                    raise ModelToolCallParseError(f"{action_type} requires valid {key}")
                w, h = _require_resolution(resolution)
                return pixel_to_norm(coord[0], coord[1], w, h)
            if required:
                raise ModelToolCallParseError(f"{action_type} requires {key}")
            return None

        def _keys_arg(key: str) -> list[str]:
            value = action.get(key, "")
            if not isinstance(value, str):
                raise ModelToolCallParseError(f"{action_type} requires string {key}")
            try:
                return normalize_keys(value)
            except ValueError as exc:
                raise ModelToolCallParseError(str(exc)) from exc

        def _mouse_action(call: dict[str, Any]) -> list[dict[str, Any]]:
            if "text" not in action or action["text"] == "":
                return [call]
            keys = _keys_arg("text")
            if not set(keys) <= {"shift", "ctrl", "alt", "meta"}:
                raise ModelToolCallParseError(f"{action_type} requires modifier keys in text")
            return [
                LiteDesktopActionSpace.key_down(keys=keys),
                call,
                LiteDesktopActionSpace.key_up(keys=keys),
            ]

        if action_type in ("left_click", "click"):
            return _mouse_action(LiteDesktopActionSpace.click(coordinate=_coord(required=True)))

        elif action_type == "right_click":
            return _mouse_action(
                LiteDesktopActionSpace.click(coordinate=_coord(required=True), button="right")
            )

        elif action_type == "middle_click":
            return _mouse_action(
                LiteDesktopActionSpace.click(coordinate=_coord(required=True), button="middle")
            )

        elif action_type == "double_click":
            return _mouse_action(
                LiteDesktopActionSpace.click(coordinate=_coord(required=True), clicks=2)
            )

        elif action_type in ("type", "type_text"):
            return [LiteDesktopActionSpace.type(text=action.get("text", ""))]

        elif action_type in ("key", "keypress", "hotkey"):
            keys = _keys_arg("text")
            repeat = action.get("repeat", 1)
            if type(repeat) is not int or not 1 <= repeat <= 100:
                raise ModelToolCallParseError("key requires repeat between 1 and 100")
            return [LiteDesktopActionSpace.key(keys=keys) for _ in range(repeat)]

        elif action_type == "scroll":
            coord = _coord()
            direction = action.get("scroll_direction", "down")
            amount = int(
                _require_number(
                    action.get("scroll_amount", 3),
                    action_type=action_type,
                    key="scroll_amount",
                )
            )
            return _mouse_action(
                LiteDesktopActionSpace.scroll(
                    direction=direction,
                    amount=amount,
                    coordinate=coord,
                )
            )

        elif action_type in ("left_click_drag", "drag"):
            start = _coord("start_coordinate")
            end = _coord("end_coordinate")
            # If only coordinate is present (not start/end), treat as end
            if not end:
                end = _coord("coordinate")
            if end is None:
                raise ModelToolCallParseError(
                    f"{action_type} requires end_coordinate or coordinate"
                )
            return _mouse_action(
                LiteDesktopActionSpace.drag(
                    coordinate=end,
                    start_coordinate=start,
                )
            )

        elif action_type in ("mouse_move", "move_cursor", "move"):
            return [LiteDesktopActionSpace.mouse_move(coordinate=_coord(required=True))]

        elif action_type == "screenshot":
            return [LiteDesktopActionSpace.screenshot()]

        elif action_type == "triple_click":
            return _mouse_action(
                LiteDesktopActionSpace.click(coordinate=_coord(required=True), clicks=3)
            )

        elif action_type == "wait":
            duration = _require_number(
                action.get("duration", 1.0),
                action_type=action_type,
                key="duration",
            )
            return [LiteDesktopActionSpace.wait(duration=float(duration))]

        elif action_type == "hold_key":
            keys = _keys_arg("text" if "text" in action else "key")
            duration = _require_number(
                action.get("duration", 1.0),
                action_type=action_type,
                key="duration",
            )
            return [LiteDesktopActionSpace.hold_key(keys=keys, duration=float(duration))]

        elif action_type == "left_mouse_down":
            result = []
            coord = _coord()
            if coord:
                result.append(LiteDesktopActionSpace.mouse_move(coordinate=coord))
            result.append(LiteDesktopActionSpace.mouse_down())
            return result

        elif action_type == "left_mouse_up":
            result = []
            coord = _coord()
            if coord:
                result.append(LiteDesktopActionSpace.mouse_move(coordinate=coord))
            result.append(LiteDesktopActionSpace.mouse_up())
            return result

        else:
            message = f"unknown Claude native computer action: {action_type or '<unknown>'}"
            raise ModelToolCallParseError(message)

    # -------------------------------------------------------------------------
    # Tool Call Conversion: CUA-lite -> Claude
    # -------------------------------------------------------------------------

    def convert_tool_calls_to_agent(
        self,
        tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Convert CUA-lite tool calls to Claude computer tool actions.

        Args:
            tool_calls: CUA-lite tool call dicts.
            **kwargs: Must include ``resolution=(width, height)`` for coordinate conversion.
        """
        resolution = kwargs.get("resolution")
        result = []
        for tc in tool_calls:
            converted = self._convert_single_to_agent(tc, resolution)
            result.extend(converted)
        return result

    def _convert_single_to_agent(
        self,
        tool_call: dict[str, Any],
        resolution: tuple[int, int] | None,
    ) -> list[dict[str, Any]]:
        """Convert a single CUA-lite tool call to Claude action format."""
        name = tool_call_name(tool_call)
        args = tool_call_arguments(tool_call)
        w, h = _require_resolution(resolution)

        def _coord(key: str = "coordinate", *, required: bool = False) -> list[int] | None:
            coord = args.get(key)
            if coord is None:
                if required:
                    raise ValueError(f"{name} requires {key}")
                return None
            if (
                not isinstance(coord, (list, tuple))
                or len(coord) < 2
                or not _is_coordinate_number(coord[0])
                or not _is_coordinate_number(coord[1])
            ):
                raise ValueError(f"{name} requires numeric {key}")
            return list(strict_norm_to_pixel(coord, w, h, clamp=False))

        if name == "computer":
            result: list[dict[str, Any]] = []
            for child in args["actions"]:
                action = child["action"]
                action_args = {k: v for k, v in child.items() if k != "action"}
                result.extend(
                    self._convert_single_to_agent(
                        make_tool_call(action, action_args),
                        resolution,
                    )
                )
            return result

        if name == "click":
            coord = _coord(required=True)
            button = args.get("button", "left")
            clicks = args.get("clicks", 1)
            action = _CLICK_WIRE_ACTIONS.get((button, clicks))
            if action is None:
                spellable = ", ".join(f"{b}x{c}" for b, c in sorted(_CLICK_WIRE_ACTIONS))
                raise ValueError(
                    f"Claude cannot express click(button={button!r}, clicks={clicks}): "
                    f"the Claude computer wire spells only {spellable}"
                )
            return [{"action": action, "coordinate": coord}]

        elif name == "type":
            return [{"action": "type", "text": args.get("text", "")}]

        elif name == "key":
            keys = args.get("keys", [])
            return [{"action": "key", "text": "+".join(keys)}]

        elif name == "scroll":
            return [
                {
                    "action": "scroll",
                    "coordinate": _coord(required=True),
                    "scroll_direction": args.get("direction", "down"),
                    "scroll_amount": args.get("amount", 3),
                }
            ]

        elif name == "drag":
            end = _coord("coordinate", required=True)
            start = _coord("start_coordinate", required=True)
            return [
                {
                    "action": "left_click_drag",
                    "start_coordinate": start,
                    "end_coordinate": end,
                }
            ]

        elif name == "mouse_move":
            return [{"action": "mouse_move", "coordinate": _coord(required=True)}]

        elif name == "mouse_down":
            return [{"action": "left_mouse_down"}]

        elif name == "mouse_up":
            return [{"action": "left_mouse_up"}]

        elif name == "hold_key":
            keys = args.get("keys", [])
            out: dict[str, Any] = {"action": "hold_key", "key": "+".join(keys)}
            if args.get("duration") is not None:
                out["duration"] = float(args["duration"])
            return [out]

        elif name in ("key_down", "key_up"):
            # The Claude computer wire has NO half-press verb: ``key`` presses AND
            # releases, ``hold_key`` holds then releases. An action the wire cannot
            # spell raises rather than being rewritten or dropped, which would leave
            # a held modifier unreleased. ``ClaudeMobileActionSpace``'s wire spells
            # both, so it is unaffected.
            raise ValueError(
                f"Claude desktop cannot render canonical tool {name!r}: the "
                "Claude computer wire has no half-press verb (use 'key' for a "
                "full press or 'hold_key' for a timed hold)"
            )

        elif name == "screenshot":
            return [{"action": "screenshot"}]

        elif name == "wait":
            return [{"action": "wait", "duration": float(args.get("duration", 1.0))}]

        else:
            return [{"name": name, "arguments": copy.deepcopy(args)}]


# =============================================================================
# Claude Desktop Grounding Point Action Space
# =============================================================================


@dataclasses.dataclass
class ClaudeDesktopGroundingPointActionSpace(
    ClaudeDesktopActionSpace, key=r"claude@(desktop|browser)@point"
):
    """Claude desktop grounding (single-step click).

    Registered under the action-space ``@point`` format axis. The
    agent/adapter task key remains ``@grounding.point``.

    Wire format keeps Claude's familiar ``left_click(coordinate=[x, y])``
    shape — that's what the API model is most fluent in (matches the
    Anthropic ``computer_*`` tool's ``left_click`` action). The
    cua-lite side speaks the canonical grounding shape
    (:class:`LitePointActionSpace.point`); the conversion paths are
    overridden so the agent's output is mapped from native ``left_click``
    → lite ``point``.
    """

    _PROVIDER_FLAT_GROUNDING_TOOL_NAME = "left_click"
    _LITE_FORMAT_ACTION_NAME = next(iter(LitePointActionSpace.get_action_names()))

    @classmethod
    def _build_grounding_schema(cls) -> dict[str, Any] | None:
        """The grounding ``left_click`` schema, built from ``_SCHEMAS`` directly.

        Reads the declaration table rather than ``get_tool_schema``: the two
        public accessors both delegate here, so any back-edge between them
        would be infinite recursion.
        """
        name = cls._PROVIDER_FLAT_GROUNDING_TOOL_NAME
        if name not in cls._SCHEMAS:
            return None
        schema = copy.deepcopy(cls._SCHEMAS[name])
        function = schema["function"]
        function["description"] = (
            "Click the left mouse button at the specified pixel coordinates "
            "of the target UI element."
        )
        coordinate = (
            tool_schema_parameters(schema).setdefault("properties", {}).setdefault("coordinate", {})
        )
        coordinate.update(
            {
                "description": ("[x, y] pixel coordinates of the click target on the screenshot."),
                "minItems": 2,
                "maxItems": 2,
            }
        )
        return schema

    @classmethod
    def get_tool_schemas(cls, include: list[str] | None = None) -> list[dict[str, Any]]:
        """The single Claude-provider ``left_click`` schema used for grounding."""
        if include is not None and cls._PROVIDER_FLAT_GROUNDING_TOOL_NAME not in set(include):
            return []
        schema = cls._build_grounding_schema()
        return [] if schema is None else [schema]

    @classmethod
    def get_declared_action_schema_names(cls) -> frozenset[str]:
        return frozenset({cls._PROVIDER_FLAT_GROUNDING_TOOL_NAME})

    def _convert_single_from_agent(
        self,
        action: dict[str, Any],
        resolution: tuple[int, int] | None,
    ) -> list[dict[str, Any]]:
        """Map Claude's left_click → lite ``point`` (single coordinate prediction)."""
        action_type = action.get("action", "")
        w, h = _require_resolution(resolution)
        coord = action.get("coordinate")
        if action_type in ("left_click", "click"):
            if not _is_coordinate_pair(coord):
                raise ModelToolCallParseError(f"{action_type} requires valid coordinate")
            norm = pixel_to_norm(coord[0], coord[1], w, h)
            return [LitePointActionSpace.point(coordinate=norm)]
        message = f"unknown Claude desktop grounding action: {action_type or '<unknown>'}"
        raise ModelToolCallParseError(message)

    def convert_tool_calls_to_agent(
        self,
        tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """cua-lite ``point(coord)`` → Claude ``left_click(coordinate=coord)``.

        Inverse for replay. Coordinates round-trip via norm→pixel using
        ``resolution=(w, h)``.
        """
        resolution = _require_resolution(kwargs.get("resolution"))
        w, h = resolution
        out: list[dict[str, Any]] = []
        for tc in tool_calls:
            name = tool_call_name(tc)
            args = tool_call_arguments(tc)
            if name == "point":
                coord = args.get("coordinate")
                if coord is None:
                    raise ValueError("point requires coordinate")
                if (
                    not isinstance(coord, (list, tuple))
                    or len(coord) < 2
                    or not _is_coordinate_number(coord[0])
                    or not _is_coordinate_number(coord[1])
                ):
                    raise ValueError("point requires numeric coordinate")
                px = list(strict_norm_to_pixel(coord, w, h, clamp=False))
                out.append({"action": "left_click", "coordinate": px})
            else:
                out.extend(
                    convert_non_point_call_for_grounding_space(
                        tc,
                        surface="Claude desktop grounding (point) action_space",
                    )
                )
        return out


# =============================================================================
# Claude Mobile Action Space (Android)
# =============================================================================


@dataclasses.dataclass
class ClaudeMobileActionSpace(BaseActionSpace, key="claude@mobile"):
    """Claude mobile action space for Android interactions.

    Unlike desktop where Claude has a native ``computer_*`` tool type,
    mobile actions are provider-flat function tools (``tap``, ``swipe``,
    ``type``, etc.). Coordinates are absolute pixels on the current request's
    screenshot frame, origin top-left.

    Canonical Lite/parquet still stores GUI actions as one or more
    ``mobile(actions=[...])`` calls. ``open_app``/``response``/``terminate`` and
    other env extras remain standalone top-level function tools.
    """

    platform: str = "mobile"
    _MOBILE_ACTION_BATCH_TOOL_NAME = LITE_MOBILE_ACTION_BATCH_TOOL_NAME

    # -------------------------------------------------------------------------
    # Mobile tool schemas (model-facing; descriptions must anchor semantics)
    # -------------------------------------------------------------------------

    @staticmethod
    @tool(
        coordinate="[x, y] pixel coordinates on the current screenshot, origin top-left.",
        clicks="Number of taps: 1=single, 2=double.",
    )
    def tap(coordinate: list[int], clicks: Literal[1, 2] = 1) -> dict[str, Any]:
        """Single tap at the given pixel coordinate on an Android device.
        Use for most UI interactions (buttons, list items, text fields).
        Prefer over long_press unless you need a context menu.
        """
        return make_tool_call("tap", {"coordinate": coordinate, "clicks": clicks})

    @staticmethod
    @tool(
        coordinate="[x, y] pixel coordinates on the current screenshot.",
        duration=duration_description("Press duration in seconds (default 1.0)."),
    )
    def long_press(
        coordinate: list[int],
        duration: float | None = None,
    ) -> dict[str, Any]:
        """Long-press at the given pixel coordinate on an Android device.
        Use ONLY for drag-initiation, context menus, or explicit long-press UI.
        """
        return make_tool_call(
            "long_press",
            {"coordinate": coordinate, "duration": duration},
        )

    @staticmethod
    @tool(
        start_coordinate="[x, y] pixel coordinates where the swipe starts on the screenshot.",
        coordinate="[x, y] pixel coordinates where the swipe ends on the screenshot.",
    )
    def swipe(start_coordinate: list[int], coordinate: list[int]) -> dict[str, Any]:
        """Swipe from start to end on an Android device.
        Use for scrolling lists, dismissing notifications, or dragging widgets.
        Both coordinates are absolute pixels on the current screenshot.
        """
        return make_tool_call(
            "swipe",
            {
                "start_coordinate": start_coordinate,
                "coordinate": coordinate,
            },
        )

    @staticmethod
    @tool(
        start_coordinate="[x, y] pixel coordinates where the drag starts on the screenshot.",
        coordinate="[x, y] pixel coordinates where the drag ends on the screenshot.",
    )
    def drag(start_coordinate: list[int], coordinate: list[int]) -> dict[str, Any]:
        """Drag from start to end on an Android device.
        Use for moving items, rearranging widgets, or slider adjustments.
        Unlike swipe, drag holds the touch down throughout the motion.
        """
        return make_tool_call(
            "drag",
            {
                "start_coordinate": start_coordinate,
                "coordinate": coordinate,
            },
        )

    @staticmethod
    @tool(
        coordinate="[x, y] center pixel coordinate on the current screenshot.",
        direction="Pinch direction: 'in' to zoom out, 'out' to zoom in.",
        amount="Pinch distance as percentage of screen (10-50). Default 25.",
    )
    def pinch(
        coordinate: list[int],
        direction: Literal["in", "out"],
        amount: int = 25,
    ) -> dict[str, Any]:
        """Pinch at the given center point."""
        return make_tool_call(
            "pinch",
            {
                "coordinate": coordinate,
                "direction": direction,
                "amount": amount,
            },
        )

    @staticmethod
    @tool(text="Text to type into the currently focused input field.")
    def type(text: str) -> dict[str, Any]:
        """Type text into the currently-focused Android input field.
        Ensure the field is focused (tap it first) before typing.
        """
        return make_tool_call("type", {"text": text})

    @staticmethod
    @tool(button="Which system button to press.")
    def system_button(
        button: Literal["Home", "Back", "Enter", "Menu", "Recent"],
    ) -> dict[str, Any]:
        """Press an Android system button: Home, Back, Enter, Menu, or Recent."""
        return make_tool_call("system_button", {"button": button})

    # NOTE: open_app is NOT a native mobile action — it's an env extra_tool
    # (make_open_app_tool, which carries the env's app catalog as an enum). See
    # env_kwargs.extra_tools. Keeping it out of the declared action schemas
    # avoids colliding with the schema-backed standalone extra tool.

    @staticmethod
    @tool(duration=duration_description("Time in seconds to wait."))
    def wait(duration: float) -> dict[str, Any]:
        """Pause the Android agent loop for a specified duration.
        Use to let apps finish loading after a navigation.
        """
        return make_tool_call("wait", {"duration": duration})

    @staticmethod
    @tool()
    def screenshot() -> dict[str, Any]:
        """Capture a fresh screenshot of the Android screen.
        The image will be sent back to you on the next turn.
        """
        return make_tool_call("screenshot")

    # -------------------------------------------------------------------------
    # Round-trip conversion (pixel ↔ CUA-lite normalized [0, 1000])
    # -------------------------------------------------------------------------

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Claude provider-flat mobile GUI calls → canonical Lite mobile calls.

        Adjacent provider GUI calls are one GUI turn and normalize to one
        canonical ``mobile(actions=[...])`` call. Standalone extra tools remain
        separate and are never crossed by the merge.
        """
        resolution = kwargs.get("resolution")
        result: list[dict[str, Any]] = []
        native_tool_names = self.get_tool_names()
        for tc in agent_tool_calls:
            if tc.get("name") == self._MOBILE_ACTION_BATCH_TOOL_NAME:
                message = f"unknown Claude mobile action: {self._MOBILE_ACTION_BATCH_TOOL_NAME}"
                raise ModelToolCallParseError(message)
            name = tc.get("name")
            if name in native_tool_names:
                arguments = tc.get("arguments")
                if not isinstance(arguments, dict):
                    raise ModelToolCallParseError(
                        f"malformed Claude mobile arguments for {name}: expected object"
                    )
                result.extend(
                    self._convert_single_from_agent(
                        {"action": name, **arguments},
                        resolution,
                    )
                )
                continue
            if name:
                result.append(make_tool_call(tc["name"], tc.get("arguments", {})))
                continue
            result.extend(
                self._convert_single_from_agent(
                    tc,
                    resolution,
                )
            )
        return merge_adjacent_lite_action_batches(result)

    def _convert_single_from_agent(
        self,
        action: dict[str, Any],
        resolution: tuple[int, int] | None,
    ) -> list[dict[str, Any]]:
        action_type = action.get("action", "")
        if action_type in type(self).get_tool_names():
            self._validate_native_action(
                str(action_type),
                {k: v for k, v in action.items() if k != "action"},
            )
        w, h = _require_resolution(resolution)

        def _coord(key: str = "coordinate") -> list[int] | None:
            coord = action.get(key)
            if _is_coordinate_pair(coord):
                return pixel_to_norm(coord[0], coord[1], w, h)
            return None

        def _require_coord(action_name: str, coord: list[int] | None) -> list[int]:
            if coord is None:
                raise ModelToolCallParseError(
                    f"malformed Claude mobile arguments for {action_name}: missing coordinate"
                )
            return coord

        if action_type == "tap":
            return [
                LiteMobileActionSpace.tap(
                    coordinate=_require_coord("tap", _coord()),
                    clicks=int(action.get("clicks", 1)),
                )
            ]
        if action_type == "drag":
            return [
                LiteMobileActionSpace.drag(
                    start_coordinate=_require_coord("drag", _coord("start_coordinate")),
                    coordinate=_require_coord("drag", _coord("coordinate")),
                )
            ]
        if action_type == "pinch":
            return [
                LiteMobileActionSpace.pinch(
                    coordinate=_require_coord("pinch", _coord()),
                    direction=action.get("direction", "in"),
                    amount=int(action.get("amount", 25)),
                )
            ]
        if action_type == "long_press":
            return [
                LiteMobileActionSpace.long_press(
                    coordinate=_require_coord("long_press", _coord()),
                    duration=action.get("duration"),
                )
            ]
        if action_type == "swipe":
            return [
                LiteMobileActionSpace.swipe(
                    start_coordinate=_require_coord("swipe", _coord("start_coordinate")),
                    coordinate=_require_coord("swipe", _coord("coordinate")),
                )
            ]
        if action_type == "type":
            return [LiteMobileActionSpace.type(text=action.get("text", ""))]
        if action_type == "system_button":
            return [LiteMobileActionSpace.system_button(button=action.get("button", "Home"))]
        if action_type == "wait":
            return [LiteMobileActionSpace.wait(duration=float(action.get("duration", 1.0)))]
        if action_type == "screenshot":
            return [LiteMobileActionSpace.screenshot()]

        message = f"unknown Claude mobile action: {action_type or '<unknown>'}"
        raise ModelToolCallParseError(message)

    @classmethod
    def _validate_native_action(cls, name: str, arguments: dict[str, Any]) -> None:
        """Gate model-emitted mobile arguments before they reach the converters."""
        schema = cls.get_tool_schema(name)
        if schema is None:
            return
        call = make_tool_call(name, arguments)
        if not tool_call_satisfies_schema(call, schema):
            raise ModelToolCallParseError(f"malformed Claude mobile arguments for {name}")
        for coord_key in _CLAUDE_MOBILE_COORD_KEYS_BY_ACTION.get(name, ()):
            coord = arguments.get(coord_key)
            if not _is_coordinate_pair(coord):
                raise ModelToolCallParseError(f"malformed Claude mobile arguments for {name}")

    def convert_tool_calls_to_agent(
        self,
        tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Canonical Lite mobile calls → Claude provider-flat mobile GUI calls."""
        resolution = kwargs.get("resolution")
        result: list[dict[str, Any]] = []
        for tc in merge_adjacent_lite_action_batches(tool_calls):
            result.extend(self._convert_single_to_agent(tc, resolution))
        return result

    def _convert_single_to_agent(
        self,
        tool_call: dict[str, Any],
        resolution: tuple[int, int] | None,
    ) -> list[dict[str, Any]]:
        name = tool_call_name(tool_call)
        args = tool_call_arguments(tool_call)
        if name == self._MOBILE_ACTION_BATCH_TOOL_NAME:
            calls: list[dict[str, Any]] = []
            for child in args["actions"]:
                action = child["action"]
                action_args = {k: v for k, v in child.items() if k != "action"}
                agent_action = self._convert_action_to_agent_action(
                    action,
                    action_args,
                    resolution,
                )
                agent_name = agent_action.pop("action")
                calls.append(
                    {
                        "name": agent_name,
                        "arguments": {
                            k: copy.deepcopy(v) for k, v in agent_action.items() if v is not None
                        },
                    }
                )
            return calls

        if name in type(self).get_declared_action_schema_names():
            agent_action = self._convert_action_to_agent_action(name, args, resolution)
            agent_name = agent_action.pop("action")
            return [
                {
                    "name": agent_name,
                    "arguments": {
                        k: copy.deepcopy(v) for k, v in agent_action.items() if v is not None
                    },
                }
            ]

        return [
            {
                "name": name,
                "arguments": {k: copy.deepcopy(v) for k, v in args.items() if v is not None},
            }
        ]

    def _convert_action_to_agent_action(
        self,
        name: str,
        args: dict[str, Any],
        resolution: tuple[int, int] | None,
    ) -> dict[str, Any]:
        """Convert one Lite mobile child action to a Claude mobile action item."""

        def _coord(key: str = "coordinate", *, required: bool = False) -> list[int] | None:
            coord = args.get(key)
            if coord is None:
                if required:
                    raise ValueError(f"malformed Claude mobile arguments for {name}: missing {key}")
                return None
            if (
                not isinstance(coord, (list, tuple))
                or len(coord) < 2
                or not _is_coordinate_number(coord[0])
                or not _is_coordinate_number(coord[1])
            ):
                raise ValueError(f"malformed Claude mobile arguments for {name}: invalid {key}")
            w, h = _require_resolution(resolution)
            return list(strict_norm_to_pixel(coord, w, h, clamp=False))

        if name == "tap":
            return {
                "action": "tap",
                "coordinate": _coord(required=True),
                "clicks": int(args.get("clicks", 1)),
            }
        if name == "long_press":
            out: dict[str, Any] = {
                "action": "long_press",
                "coordinate": _coord(required=True),
            }
            if args.get("duration") is not None:
                out["duration"] = args["duration"]
            return out
        if name == "swipe":
            return {
                "action": "swipe",
                "start_coordinate": _coord("start_coordinate", required=True),
                "coordinate": _coord("coordinate", required=True),
            }
        if name == "drag":
            return {
                "action": "drag",
                "start_coordinate": _coord("start_coordinate", required=True),
                "coordinate": _coord("coordinate", required=True),
            }
        if name == "pinch":
            return {
                "action": "pinch",
                "coordinate": _coord(required=True),
                "direction": args.get("direction", "in"),
                "amount": int(args.get("amount", 25)),
            }
        if name == "type":
            return {"action": "type", "text": args.get("text", "")}
        if name == "system_button":
            return {"action": "system_button", "button": args.get("button", "Home")}
        if name == "wait":
            return {"action": "wait", "duration": float(args.get("duration", 1.0))}
        if name == "screenshot":
            return {"action": "screenshot"}

        logger.warning("Unknown CUA-lite mobile action for Claude: %s(%s)", name, args)
        return {"action": name, **args}
