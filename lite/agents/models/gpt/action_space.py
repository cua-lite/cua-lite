"""
GPT Computer-Use Action Space (Desktop)

Maps between CUA-lite normalized [0, 1000] coordinates and OpenAI's pixel-based
computer tool actions (GA ``{"type":"computer"}`` → ``computer_call`` items).

OpenAI computer tool actions use absolute pixel coordinates with separate x, y fields.
CUA-lite uses normalized [0, 1000] coordinates with coordinate arrays.

Usage:
    from lite.agents.models.gpt.action_space import GPTDesktopActionSpace

    action_space = GPTDesktopActionSpace()

    # Convert OpenAI action -> CUA-lite
    gpt_action = {"type": "click", "x": 512, "y": 384}
    lite_actions = action_space.convert_tool_calls_from_agent([gpt_action], resolution=(1024, 768))

    # Convert CUA-lite -> OpenAI action
    lite_action = LiteDesktopActionSpace.click(coordinate=[500, 500])
    gpt_actions = action_space.convert_tool_calls_to_agent([lite_action], resolution=(1024, 768))
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import math
from typing import Any, Literal

from lite.agents.core.action_space.base import (
    BaseActionSpace,
    LiteDesktopActionSpace,
    LiteMobileActionSpace,
    LitePointActionSpace,
)
from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.agents.core.action_space.utils.geometry import PIXELS_PER_CLICK
from lite.agents.core.action_space.utils.grounding_point import (
    convert_non_point_call_for_grounding_space,
)
from lite.core.tools.action_space import (
    merge_adjacent_lite_action_batches,
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

# =============================================================================
# Coordinate Helpers
# =============================================================================

# GPT native computer tool outputs pixel deltas for scroll; CUA-lite scroll
# amount is in clicks. The px-per-click conversion is the shared cross-family
# convention owned by ``lite.agents.core.action_space.utils``.

# Every ``(button, clicks)`` pair the OpenAI computer wire can actually spell,
# mapped to the native action type. ``click`` carries a ``button`` but is always
# ONE click; ``double_click``/``right_click`` are fixed shapes with no button or
# count field; there is no triple-click action at all. The lite
# ``click`` schema is wider (3 buttons x 3 counts), so five of its nine
# combinations have no faithful wire spelling — emitting the nearest one
# silently changes the action, which is the bug this table exists to prevent.
# Keep in lockstep with the parse side in ``_convert_single_from_agent``.
_CLICK_WIRE_TYPES: dict[tuple[str, int], str] = {
    ("left", 1): "click",
    ("left", 2): "double_click",
    ("right", 1): "right_click",
    ("middle", 1): "click",
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


def _require_resolution(resolution: tuple[int, int] | None) -> tuple[int, int]:
    """Require ``resolution=(w, h)`` — the actual frame dims the model saw.

    GPT's pixel-coord actions only round-trip correctly when (w, h) matches
    the frame the model saw. A silent fallback to some hardcoded default
    (e.g. ``(1024, 768)``) would mis-scale every click whenever the API
    processed the image at a different dim — which can happen any time the
    API auto-resizes a sent image past its detail-level limits.

    The agent passes API-echoed processed dims from
    ``_call_api_with_actual_dim``, which degrades to the sent frame (with a
    warning) when the echo lookup does not answer, and for text-only/no-image
    requests where no provider image block exists to echo dims at all.
    """
    if resolution is None:
        raise ValueError(
            "GPT action_space coord conversion requires resolution=(w, h) — "
            "the actual frame dims the model saw."
        )
    return resolution


# =============================================================================
# GPT Desktop Action Space
# =============================================================================


@dataclasses.dataclass
class GPTDesktopActionSpace(BaseActionSpace, key=r"gpt@(desktop|browser)"):
    """
    GPT computer-use action space for desktop interactions.

    OpenAI uses separate x, y fields for coordinates (not arrays). The GA native
    ``{"type":"computer"}`` tool emits ``computer_call`` items; env-supplied
    function tools emit ``function_call`` items.

    Computer action types: click, double_click, right_click, type, keypress,
    scroll, move, drag, screenshot, wait.
    """

    platform: str = "desktop"

    @classmethod
    def get_tool_schemas(cls, include: list[str] | None = None) -> list[dict[str, Any]]:
        """GPT desktop uses OpenAI's native computer tool, not function schemas.

        This action space is the native-computer ↔ Lite conversion surface.
        Provider tool assembly happens in ``GPTDesktopUseAgent._build_tools``.
        """
        return []

    @classmethod
    def filter_tool_schemas_for_valid_actions(
        cls,
        schemas: list[dict[str, Any]],
        valid_actions: list[str],
    ) -> list[dict[str, Any]]:
        """Gate OpenAI's opaque native ``{"type": "computer"}`` schema.

        The OpenAI schema exposes no request-side enum for individual computer
        actions, so the only valid-action decision is whole-tool visibility:
        an empty whitelist drops it, and any non-empty whitelist keeps it whole.
        Grounding subclasses still delegate their provider-flat click schema to
        the normal action-space-owned named-schema filter.
        """
        result: list[dict[str, Any]] = []
        for schema in schemas:
            if str(schema.get("type", "")) == "computer":
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
    @tool(x="X pixel coordinate.", y="Y pixel coordinate.")
    def click(x: int, y: int) -> dict[str, Any]:
        """Click the left mouse button at the specified coordinates."""
        return make_tool_call("click", {"x": x, "y": y})

    @staticmethod
    @tool(x="X pixel coordinate.", y="Y pixel coordinate.")
    def double_click(x: int, y: int) -> dict[str, Any]:
        """Double-click at the specified coordinates."""
        return make_tool_call("double_click", {"x": x, "y": y})

    @staticmethod
    @tool(x="X pixel coordinate.", y="Y pixel coordinate.")
    def right_click(x: int, y: int) -> dict[str, Any]:
        """Right-click at the specified coordinates."""
        return make_tool_call("right_click", {"x": x, "y": y})

    # -------------------------------------------------------------------------
    # Keyboard Actions
    # -------------------------------------------------------------------------

    @staticmethod
    @tool(text="The text to type.")
    def type(text: str) -> dict[str, Any]:
        """Type text content."""
        return make_tool_call("type", {"text": text})

    @staticmethod
    @tool(keys="List of keys to press together, e.g. ['ctrl', 'c'].")
    def keypress(keys: list[str]) -> dict[str, Any]:
        """Press a keyboard key combination."""
        return make_tool_call("keypress", {"keys": keys})

    # -------------------------------------------------------------------------
    # Scroll
    # -------------------------------------------------------------------------

    @staticmethod
    @tool(
        x="X pixel coordinate to scroll at.",
        y="Y pixel coordinate to scroll at.",
        scroll_x="Horizontal scroll amount. Positive=right, negative=left.",
        scroll_y="Vertical scroll amount. Positive=down, negative=up.",
    )
    def scroll(x: int, y: int, scroll_x: int = 0, scroll_y: int = 0) -> dict[str, Any]:
        """Scroll at the specified coordinates."""
        return make_tool_call(
            "scroll",
            {"x": x, "y": y, "scroll_x": scroll_x, "scroll_y": scroll_y},
        )

    # -------------------------------------------------------------------------
    # Move & Drag
    # -------------------------------------------------------------------------

    @staticmethod
    @tool(x="X pixel coordinate.", y="Y pixel coordinate.")
    def move(x: int, y: int) -> dict[str, Any]:
        """Move the mouse cursor to specified coordinates."""
        return make_tool_call("move", {"x": x, "y": y})

    @staticmethod
    @tool(
        start_x="Starting X pixel coordinate.",
        start_y="Starting Y pixel coordinate.",
        end_x="Ending X pixel coordinate.",
        end_y="Ending Y pixel coordinate.",
    )
    def drag(start_x: int, start_y: int, end_x: int, end_y: int) -> dict[str, Any]:
        """Drag from start coordinates to end coordinates."""
        return make_tool_call(
            "drag",
            {
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
            },
        )

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
    # Tool Call Conversion: OpenAI -> CUA-lite
    # -------------------------------------------------------------------------

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Convert OpenAI computer tool actions to CUA-lite format.

        Accepts both native ``computer_call`` format and function ``function_call`` format.

        Args:
            agent_tool_calls: List of OpenAI action dicts. Each is either:
                - Native: ``{"type": "click", "x": 100, "y": 200}``
                - Function: ``{"action": "click", "x": 100, "y": 200}``
            **kwargs: Must include ``resolution=(width, height)`` for coordinate conversion.
        """
        resolution = kwargs.get("resolution")
        result = []
        for tc in agent_tool_calls:
            converted = self._convert_single_from_agent(
                tc,
                resolution,
            )
            result.extend(converted)
        return merge_adjacent_lite_action_batches(result)

    def _convert_single_from_agent(
        self,
        action: dict[str, Any],
        resolution: tuple[int, int] | None,
    ) -> list[dict[str, Any]]:
        """Convert a single OpenAI action to CUA-lite format."""
        # Normalize: both native and function formats use "type" or "action" for the action name
        action_type = action.get("type") or action.get("action", "")

        def _coord(xk: str = "x", yk: str = "y", *, required: bool = False) -> list[int] | None:
            x_val = action.get(xk)
            y_val = action.get(yk)
            if x_val is not None or y_val is not None:
                if not _is_coordinate_number(x_val) or not _is_coordinate_number(y_val):
                    raise ModelToolCallParseError(
                        f"{action_type} requires numeric {xk}/{yk} coordinates"
                    )
                w, h = _require_resolution(resolution)
                return pixel_to_norm(int(x_val), int(y_val), w, h)
            if required:
                raise ModelToolCallParseError(f"{action_type} requires {xk}/{yk} coordinates")
            return None

        if action_type == "click":
            coord = _coord(required=True)
            button = action.get("button", "left")
            if button == "right":
                return [LiteDesktopActionSpace.click(coordinate=coord, button="right")]
            elif button == "middle":
                return [LiteDesktopActionSpace.click(coordinate=coord, button="middle")]
            return [LiteDesktopActionSpace.click(coordinate=coord)]

        elif action_type == "double_click":
            return [LiteDesktopActionSpace.click(coordinate=_coord(required=True), clicks=2)]

        elif action_type == "right_click":
            return [LiteDesktopActionSpace.click(coordinate=_coord(required=True), button="right")]

        elif action_type == "type":
            # GPT spells "press Enter after typing" as a trailing newline inside
            # ``text`` — the same spelling canonical uses, so this is 1:1. The
            # env hands ``text`` straight to ``computer.interface.type_text``,
            # which sends the newline as a discrete Return keypress.
            return [LiteDesktopActionSpace.type(text=action.get("text", ""))]

        elif action_type == "keypress":
            keys = action.get("keys", [])
            return [LiteDesktopActionSpace.key(keys=keys)]

        elif action_type == "scroll":
            coord = _coord()

            def _scroll_delta(*keys: str) -> int:
                for key in keys:
                    value = action.get(key)
                    if value is not None:
                        if not _is_coordinate_number(value):
                            raise ModelToolCallParseError(f"scroll requires numeric {key}")
                        return int(value)
                return 0

            scroll_x = _scroll_delta("scroll_x", "scrollX")
            scroll_y = _scroll_delta("scroll_y", "scrollY")
            # Convert pixel deltas to click units
            if abs(scroll_x) > abs(scroll_y):
                direction = "right" if scroll_x > 0 else "left"
                amount = max(1, round(abs(scroll_x) / PIXELS_PER_CLICK))
            else:
                direction = "down" if scroll_y >= 0 else "up"
                amount = max(1, round(abs(scroll_y) / PIXELS_PER_CLICK)) if scroll_y != 0 else 3
            return [
                LiteDesktopActionSpace.scroll(
                    direction=direction,
                    amount=amount,
                    coordinate=coord,
                )
            ]

        elif action_type in ("move", "mouse_move"):
            return [LiteDesktopActionSpace.mouse_move(coordinate=_coord(required=True))]

        elif action_type == "drag":
            # Handle both separate fields and path formats
            path = action.get("path")
            if path and isinstance(path, list) and len(path) >= 2:
                if not isinstance(path[0], dict) or not isinstance(path[-1], dict):
                    raise ModelToolCallParseError("drag path requires coordinate objects")
                start_x, start_y = path[0].get("x"), path[0].get("y")
                end_x, end_y = path[-1].get("x"), path[-1].get("y")
            else:
                start_x = action.get("start_x")
                start_y = action.get("start_y")
                end_x = action.get("end_x", action.get("x"))
                end_y = action.get("end_y", action.get("y"))
            if start_x is None or start_y is None or end_x is None or end_y is None:
                raise ModelToolCallParseError("drag requires start and end coordinates")
            for key, value in (
                ("start_x", start_x),
                ("start_y", start_y),
                ("end_x", end_x),
                ("end_y", end_y),
            ):
                if not _is_coordinate_number(value):
                    raise ModelToolCallParseError(f"drag requires numeric {key}")
            w, h = _require_resolution(resolution)
            start = pixel_to_norm(int(start_x), int(start_y), w, h)
            end = pixel_to_norm(int(end_x), int(end_y), w, h)
            return [
                LiteDesktopActionSpace.drag(
                    coordinate=end,
                    start_coordinate=start,
                )
            ]

        elif action_type == "screenshot":
            return [LiteDesktopActionSpace.screenshot()]

        elif action_type == "wait":
            return [LiteDesktopActionSpace.wait(duration=1)]

        else:
            message = f"unknown GPT native computer action: {action_type or '<unknown>'}"
            raise ModelToolCallParseError(message)

    # -------------------------------------------------------------------------
    # Tool Call Conversion: CUA-lite -> OpenAI
    # -------------------------------------------------------------------------

    def convert_tool_calls_to_agent(
        self,
        tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Convert CUA-lite tool calls to OpenAI computer action format.

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
        """Convert a single CUA-lite tool call to OpenAI action format."""
        name = tool_call_name(tool_call)
        args = tool_call_arguments(tool_call)
        w, h = _require_resolution(resolution)

        def _to_pixel(key: str = "coordinate", *, required: bool = False) -> tuple[int, int] | None:
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
            return strict_norm_to_pixel(coord, w, h, clamp=False)

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
            px = _to_pixel(required=True)
            button = args.get("button", "left")
            clicks = args.get("clicks", 1)
            action_type = _CLICK_WIRE_TYPES.get((button, clicks))
            if action_type is None:
                spellable = ", ".join(f"{b}x{c}" for b, c in sorted(_CLICK_WIRE_TYPES))
                raise ValueError(
                    f"GPT cannot express click(button={button!r}, clicks={clicks}): "
                    f"the OpenAI computer wire spells only {spellable}"
                )
            result: dict[str, Any] = {"type": action_type}
            if px:
                result["x"], result["y"] = px
            # ``click`` carries the button on the wire; the dedicated
            # right_click/double_click types already encode it. Only the
            # non-default button needs spelling out, and without it a middle
            # click re-parsed as a left click.
            if action_type == "click" and button != "left":
                result["button"] = button
            return [result]

        elif name == "type":
            return [{"type": "type", "text": args.get("text", "")}]

        elif name == "key":
            return [{"type": "keypress", "keys": args.get("keys", [])}]

        elif name == "scroll":
            direction = args.get("direction", "down")
            amount = args.get("amount", 3)
            px = _to_pixel(required=True)
            scroll_x, scroll_y = 0, 0
            if direction == "down":
                scroll_y = amount * PIXELS_PER_CLICK
            elif direction == "up":
                scroll_y = -amount * PIXELS_PER_CLICK
            elif direction == "right":
                scroll_x = amount * PIXELS_PER_CLICK
            elif direction == "left":
                scroll_x = -amount * PIXELS_PER_CLICK
            result = {"type": "scroll", "scroll_x": scroll_x, "scroll_y": scroll_y}
            result["x"], result["y"] = px
            return [result]

        elif name == "drag":
            end_px = _to_pixel("coordinate", required=True)
            start_px = _to_pixel("start_coordinate", required=True)
            sx, sy = start_px
            ex, ey = end_px
            return [{"type": "drag", "start_x": sx, "start_y": sy, "end_x": ex, "end_y": ey}]

        elif name == "mouse_move":
            x, y = _to_pixel(required=True)
            return [{"type": "move", "x": x, "y": y}]

        elif name == "screenshot":
            return [{"type": "screenshot"}]

        elif name == "wait":
            return [{"type": "wait"}]

        elif name in ("key_down", "key_up"):
            # The OpenAI computer wire has NO half-press verb -- ``keypress``
            # presses AND releases. An action the wire cannot spell raises rather
            # than being dropped, which would leave a held modifier unreleased.
            # ``GPTMobileActionSpace``'s wire spells both, so it is unaffected.
            raise ValueError(
                f"GPT desktop cannot render canonical tool {name!r}: the OpenAI "
                "computer wire has no half-press verb (use 'key' for a full "
                "press)"
            )

        else:
            return [{"name": name, "arguments": copy.deepcopy(args)}]


# =============================================================================
# GPT Desktop Grounding Point Action Space
# =============================================================================


@dataclasses.dataclass
class GPTDesktopGroundingPointActionSpace(GPTDesktopActionSpace, key=r"gpt@(desktop|browser)@point"):
    """GPT desktop grounding (single-step click).

    Registered under the action-space ``@point`` format axis. The
    agent/adapter task key remains ``@grounding.point``.

    Wire format keeps GPT's familiar ``click(x=..., y=...)`` shape — that's
    what the API model is most fluent in (matches OpenAI's native
    ``computer`` tool's click action). The cua-lite side speaks the
    canonical grounding shape (:class:`LitePointActionSpace.point`); the
    conversion paths are overridden so the agent's output is mapped from
    native ``click`` → lite ``point``.
    """

    _PROVIDER_FLAT_GROUNDING_TOOL_NAME = "click"
    _LITE_FORMAT_ACTION_NAME = next(iter(LitePointActionSpace.get_action_names()))

    @classmethod
    def _build_grounding_schema(cls) -> dict[str, Any] | None:
        """The grounding ``click`` schema, built from ``_SCHEMAS`` directly.

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
        parameters = tool_schema_parameters(schema)
        parameters["additionalProperties"] = False
        properties = parameters.get("properties", {})
        if "x" in properties:
            properties["x"]["description"] = "X pixel coordinate of the click target."
        if "y" in properties:
            properties["y"]["description"] = "Y pixel coordinate of the click target."
        return schema

    @classmethod
    def get_tool_schemas(cls, include: list[str] | None = None) -> list[dict[str, Any]]:
        """The single GPT-provider ``click`` schema used for grounding."""
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
        """Map GPT's click → lite ``point`` (single coordinate prediction)."""
        action_type = action.get("type") or action.get("action", "")
        w, h = _require_resolution(resolution)
        x_val = action.get("x")
        y_val = action.get("y")
        if action_type == "click":
            if not _is_coordinate_number(x_val) or not _is_coordinate_number(y_val):
                raise ModelToolCallParseError("click requires numeric x/y coordinates")
            norm = pixel_to_norm(int(x_val), int(y_val), w, h)
            return [LitePointActionSpace.point(coordinate=norm)]
        message = f"unknown GPT desktop grounding action: {action_type or '<unknown>'}"
        raise ModelToolCallParseError(message)

    def convert_tool_calls_to_agent(
        self,
        tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """cua-lite ``point(coord)`` → GPT ``click(x=..., y=...)``.

        Inverse for replay. Coordinates round-trip via norm→pixel using
        ``resolution=(w, h)``.
        """
        w, h = _require_resolution(kwargs.get("resolution"))
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
                px = strict_norm_to_pixel(coord, w, h, clamp=False)
                out.append({"type": "click", "x": px[0], "y": px[1]})
            else:
                out.extend(
                    convert_non_point_call_for_grounding_space(
                        tc,
                        surface="GPT desktop grounding (point) action_space",
                    )
                )
        return out


# =============================================================================
# GPT Mobile Action Space (Android)
# =============================================================================


@dataclasses.dataclass
class GPTMobileActionSpace(BaseActionSpace, key="gpt@mobile"):
    """GPT mobile action space for Android interactions.

    Unlike desktop where GPT has a native ``{"type":"computer"}`` tool type,
    mobile actions are provider-flat function tools such as ``tap`` and ``swipe``.

    The provider-facing actions use separate x/y pixel fields; conversion maps
    them to Lite's canonical ``mobile.actions[]`` children with normalized
    ``coordinate: [x, y]`` values:
        ``tap(x=540, y=1200)``
        ``swipe(start_x=..., start_y=..., end_x=..., end_y=...)``

    Mobile tool set mirrors ``LiteMobileActionSpace``; the only wire
    difference is pixel coordinates instead of normalized [0, 1000] coordinates.
    (open_app/response/terminate are env extra tools when active.)
    """

    platform: str = "mobile"
    _MOBILE_ACTION_BATCH_TOOL_NAME = LITE_MOBILE_ACTION_BATCH_TOOL_NAME

    # -------------------------------------------------------------------------
    # Mobile tool schemas (model-facing; descriptions must anchor semantics)
    # -------------------------------------------------------------------------

    @staticmethod
    @tool(
        x="X pixel coordinate (0 = left edge of screen).",
        y="Y pixel coordinate (0 = top edge of screen).",
        clicks="Number of taps: 1=single, 2=double.",
    )
    def tap(x: int, y: int, clicks: Literal[1, 2] = 1) -> dict[str, Any]:
        """Tap the screen at pixel coordinates (x, y).
        Aim for the CENTER of the target element.
        """
        return make_tool_call("tap", {"x": x, "y": y, "clicks": clicks})

    @staticmethod
    @tool(
        x="X pixel coordinate (0 = left edge of screen).",
        y="Y pixel coordinate (0 = top edge of screen).",
        duration=duration_description("Hold duration in seconds (default 1.0)."),
    )
    def long_press(x: int, y: int, duration: float | None = None) -> dict[str, Any]:
        """Long-press the screen. Use for context menus or drag initiation."""
        return make_tool_call(
            "long_press",
            {"x": x, "y": y, "duration": duration},
        )

    @staticmethod
    @tool(
        start_x="Start X pixel coordinate.",
        start_y="Start Y pixel coordinate.",
        end_x="End X pixel coordinate.",
        end_y="End Y pixel coordinate.",
    )
    def swipe(start_x: int, start_y: int, end_x: int, end_y: int) -> dict[str, Any]:
        """Swipe from (start_x, start_y) to (end_x, end_y).
        To scroll DOWN: swipe from lower y to upper y (e.g. start_y=1800, end_y=600).
        To scroll UP: swipe from upper y to lower y.
        """
        return make_tool_call(
            "swipe",
            {
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
            },
        )

    @staticmethod
    @tool(
        start_x="Start X pixel coordinate.",
        start_y="Start Y pixel coordinate.",
        end_x="End X pixel coordinate.",
        end_y="End Y pixel coordinate.",
    )
    def drag(start_x: int, start_y: int, end_x: int, end_y: int) -> dict[str, Any]:
        """Drag from start to end (holds touch down throughout). Use for
        moving items, sliders, or rearranging widgets.
        """
        return make_tool_call(
            "drag",
            {
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
            },
        )

    @staticmethod
    @tool(
        x="Center X pixel coordinate.",
        y="Center Y pixel coordinate.",
        direction="Pinch direction: 'in' to zoom out, 'out' to zoom in.",
        amount="Pinch distance as percentage of screen (10-50). Default 25.",
    )
    def pinch(
        x: int,
        y: int,
        direction: Literal["in", "out"],
        amount: int = 25,
    ) -> dict[str, Any]:
        """Pinch at the given center point."""
        return make_tool_call(
            "pinch",
            {"x": x, "y": y, "direction": direction, "amount": amount},
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
    # the config's env_kwargs.extra_tools. Keeping it out of the declared action
    # schemas avoids colliding with the schema-backed standalone extra tool.

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
        """Mobile GPT provider-flat GUI calls → canonical Lite mobile batches.

        Adjacent provider GUI calls are one GUI turn and normalize to one
        canonical ``mobile(actions=[...])`` call. Standalone extra tools remain
        separate and are never crossed by the merge.
        """
        resolution = kwargs.get("resolution")
        result: list[dict[str, Any]] = []
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
        name = action.get("name")
        if name == self._MOBILE_ACTION_BATCH_TOOL_NAME:
            message = f"unknown GPT mobile action: {self._MOBILE_ACTION_BATCH_TOOL_NAME}"
            raise ModelToolCallParseError(message)
        if name in type(self).get_tool_names():
            arguments = action.get("arguments")
            if not isinstance(arguments, dict):
                raise ModelToolCallParseError(
                    f"malformed GPT mobile arguments for {name}: expected object"
                )
            self._validate_native_action(name, arguments)
            return self._convert_single_from_agent(
                {"action": name, **arguments},
                resolution,
            )
        if name:
            return [make_tool_call(name, action.get("arguments", {}))]

        action_type = action.get("action") or action.get("type", "")
        if action_type in type(self).get_tool_names():
            self._validate_native_action(
                str(action_type),
                {k: v for k, v in action.items() if k not in {"action", "type"}},
            )
        w, h = _require_resolution(resolution)

        def _xy(x_key: str = "x", y_key: str = "y") -> list[int] | None:
            x = action.get(x_key)
            y = action.get(y_key)
            if x is not None and y is not None:
                return pixel_to_norm(x, y, w, h)
            return None

        def _require_xy(action_name: str, coord: list[int] | None) -> list[int]:
            if coord is None:
                raise ModelToolCallParseError(
                    f"malformed GPT mobile arguments for {action_name}: missing coordinate"
                )
            return coord

        if action_type == "tap":
            return [
                LiteMobileActionSpace.tap(
                    coordinate=_require_xy("tap", _xy()),
                    clicks=int(action.get("clicks", 1)),
                )
            ]
        if action_type == "drag":
            return [
                LiteMobileActionSpace.drag(
                    start_coordinate=_require_xy("drag", _xy("start_x", "start_y")),
                    coordinate=_require_xy("drag", _xy("end_x", "end_y")),
                )
            ]
        if action_type == "pinch":
            return [
                LiteMobileActionSpace.pinch(
                    coordinate=_require_xy("pinch", _xy()),
                    direction=action.get("direction", "in"),
                    amount=int(action.get("amount", 25)),
                )
            ]
        if action_type == "long_press":
            return [
                LiteMobileActionSpace.long_press(
                    coordinate=_require_xy("long_press", _xy()),
                    duration=action.get("duration"),
                )
            ]
        if action_type == "swipe":
            return [
                LiteMobileActionSpace.swipe(
                    start_coordinate=_require_xy("swipe", _xy("start_x", "start_y")),
                    coordinate=_require_xy("swipe", _xy("end_x", "end_y")),
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

        message = f"unknown GPT mobile action: {action_type or '<unknown>'}"
        raise ModelToolCallParseError(message)

    @classmethod
    def _validate_native_action(cls, name: str, arguments: dict[str, Any]) -> None:
        """Gate model-emitted mobile arguments before they reach the converters."""
        schema = cls.get_tool_schema(name)
        if schema is None:
            return
        if not tool_call_satisfies_schema(make_tool_call(name, arguments), schema):
            raise ModelToolCallParseError(f"malformed GPT mobile arguments for {name}")

    def convert_tool_calls_to_agent(
        self,
        tool_calls: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Canonical Lite mobile calls → provider-flat GPT mobile function calls."""
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

        if name == "mobile":
            children = [
                (child["action"], {k: v for k, v in child.items() if k != "action"})
                for child in args["actions"]
            ]
        elif name in type(self).get_declared_action_schema_names():
            children = [(name, args)]
        else:
            return [{"name": name, "arguments": args}]

        calls: list[dict[str, Any]] = []
        for child_name, child_args in children:
            agent_action = self._convert_action_to_agent_action(
                child_name,
                child_args,
                resolution,
            )
            agent_name = agent_action.pop("action")
            calls.append({"name": agent_name, "arguments": agent_action})
        return calls

    def _convert_action_to_agent_action(
        self,
        name: str,
        args: dict[str, Any],
        resolution: tuple[int, int] | None,
    ) -> dict[str, Any]:
        """Convert one Lite mobile child action to GPT mobile action args."""

        def _xy(key: str = "coordinate", *, required: bool = False) -> tuple[int, int] | None:
            coord = args.get(key)
            if coord is None:
                if required:
                    raise ValueError(f"malformed GPT mobile arguments for {name}: missing {key}")
                return None
            if (
                not isinstance(coord, (list, tuple))
                or len(coord) < 2
                or not _is_coordinate_number(coord[0])
                or not _is_coordinate_number(coord[1])
            ):
                raise ValueError(f"malformed GPT mobile arguments for {name}: invalid {key}")
            w, h = _require_resolution(resolution)
            return strict_norm_to_pixel(coord, w, h, clamp=False)

        if name == "tap":
            xy = _xy(required=True)
            return {
                "action": "tap",
                "x": xy[0],
                "y": xy[1],
                "clicks": int(args.get("clicks", 1)),
            }
        if name == "long_press":
            xy = _xy(required=True)
            out: dict[str, Any] = {"action": "long_press", "x": xy[0], "y": xy[1]}
            if args.get("duration") is not None:
                out["duration"] = args["duration"]
            return out
        if name == "swipe":
            start = _xy("start_coordinate", required=True)
            end = _xy("coordinate", required=True)
            return {
                "action": "swipe",
                "start_x": start[0],
                "start_y": start[1],
                "end_x": end[0],
                "end_y": end[1],
            }
        if name == "drag":
            start = _xy("start_coordinate", required=True)
            end = _xy("coordinate", required=True)
            return {
                "action": "drag",
                "start_x": start[0],
                "start_y": start[1],
                "end_x": end[0],
                "end_y": end[1],
            }
        if name == "pinch":
            xy = _xy(required=True)
            return {
                "action": "pinch",
                "x": xy[0],
                "y": xy[1],
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

        logger.warning("Unknown CUA-lite mobile action for GPT: %s(%s)", name, args)
        return {"action": name, **args}
