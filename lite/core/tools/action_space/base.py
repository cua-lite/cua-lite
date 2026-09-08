"""The canonical Lite action sets — exactly what ``env_kwargs.valid_actions`` selects from.

The classes are the source of truth for the ACTION layer; the module-level
readers at the bottom are READ off them at import time and never re-declare a
name. They exist because each answers something no single class accessor can:
the union over all four sets (:data:`LITE_VALID_ACTION_NAMES`,
:func:`is_lite_action_name_or_action_batch_tool_name`), the action-batch tool ->
child-action map (:func:`lite_action_names_for_action_batch_tool`), or a
per-``LiteCUAMetadata`` selection among the sets
(:func:`lite_action_set_tool_names_for_metadata`,
:func:`lite_builtin_tool_names_for_metadata`).

Run:
    uv run pytest tests/core/tools/action_space/test_tool_catalog.py -q
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.action_space.batches import (
    LITE_COMPUTER_ACTION_BATCH_TOOL_NAME,
    LITE_MOBILE_ACTION_BATCH_TOOL_NAME,
    filter_action_batch_schema,
    make_lite_action_batch_call,
    make_lite_action_batch_schema,
)
from lite.core.tools.action_space.duration import duration_description
from lite.core.tools.action_space.keys import is_canonical_key_token, normalize_keys
from lite.core.tools.schemas import BaseTools, install_tool_schemas, tool


def _normalize_canonical_keys(keys: list[str] | str) -> list[str]:
    if keys == []:
        raise ValueError("keys must not be empty")
    normalized = normalize_keys(keys)
    for key in normalized:
        if not is_canonical_key_token(key):
            raise ValueError(
                f"keys: unknown key token {key!r}; use one literal printable "
                "character or a canonical special key"
            )
    return normalized


class BaseActionSet(BaseTools):
    """The ACTION layer: a tool set whose members ``valid_actions`` gates.

    Adds exactly one accessor to the tool-layer root — the canonical action
    vocabulary this set carries. No invariant binds it to the tool layer:
    an action-batch set carries many actions and emits 1 top-level tool, and
    that divergence is what "action-batch" means.
    """

    @classmethod
    def get_action_names(cls) -> frozenset[str]:
        """Canonical action names this set CARRIES (never what it emits).

        Answers the action question independently of the tool question: an
        action-batch set emits one top-level tool but carries every declared
        action.
        """
        return frozenset(cls._SCHEMAS)


class _ActionBatchActionSet(BaseActionSet):
    """Abstract action set whose members ride inside an action-batch call.

    A GUI action is not advertised at top level in the canonical Lite surface:
    the model sees one action-batch tool (``computer`` / ``mobile``) whose
    ``arguments.actions`` array carries child actions.

    ``_ACTION_BATCH_TOOL_NAME`` and ``_ACTION_BATCH_DESCRIPTION`` stay UNPREFIXED
    on purpose: they are owner-internal protected attrs of this class, not part
    of the ``LITE_*`` public action-batch vocabulary. A broad ``ACTION_BATCH``
    grep used as closure evidence should treat these two as a documented
    exception, not as unmigrated public symbols.
    """

    _ACTION_BATCH_TOOL_NAME: ClassVar[str]
    _ACTION_BATCH_DESCRIPTION: ClassVar[str]

    @classmethod
    def _wrap_action(cls, child_call: dict[str, Any]) -> dict[str, Any]:
        return make_lite_action_batch_call(cls._ACTION_BATCH_TOOL_NAME, child_call)

    @classmethod
    def get_tool_schemas(cls, include: list[str] | None = None) -> list[dict[str, Any]]:
        """The single action-batch schema, narrowed to the ``include`` children.

        **Here ``include=`` stays a FILTER**, where
        :meth:`BaseTools.get_tool_schemas` on unbatched tool sets makes it a
        SELECTOR that raises.
        The difference is the layer, not taste: on an action-batch set
        ``include=`` ranges over the ACTION layer (``get_action_names()``,
        the child actions), which is deliberately DISJOINT from the tool layer
        this set emits — ``LiteDesktopActionSet.get_tool_schemas(include=["computer"])``
        names the action-batch tool and matches no child, which is a layer
        crossing rather than a typo. Filtering to nothing drops the
        action-batch tool entirely (``[]``), which is how ``valid_actions: []``
        strips the GUI tool.
        """
        schema = make_lite_action_batch_schema(
            action_batch_tool_name=cls._ACTION_BATCH_TOOL_NAME,
            description=cls._ACTION_BATCH_DESCRIPTION,
            child_schemas=cls._SCHEMAS,
            include=include,
        )
        return [] if schema is None else [schema]

    @classmethod
    def filter_child_action_enum(
        cls,
        schemas: list[dict[str, Any]],
        valid_actions: list[str],
    ) -> list[dict[str, Any]]:
        """Filter child action enum entries inside this action-batch schema."""
        return filter_action_batch_schema(
            schemas,
            action_batch_tool_name=cls._ACTION_BATCH_TOOL_NAME,
            description=cls._ACTION_BATCH_DESCRIPTION,
            child_schemas=cls._SCHEMAS,
            valid_actions=valid_actions,
        )


@install_tool_schemas
class LiteDesktopActionSet(_ActionBatchActionSet):
    """Core-owned canonical desktop GUI actions, carried by ``computer``."""

    _ACTION_BATCH_TOOL_NAME: ClassVar[str] = LITE_COMPUTER_ACTION_BATCH_TOOL_NAME
    _ACTION_BATCH_DESCRIPTION: ClassVar[str] = (
        "Perform one or more desktop GUI actions. Each item in actions is a "
        "single action desktop action."
    )

    @staticmethod
    @tool(keys="list of keys to press.")
    def key(keys: list[str]) -> dict[str, Any]:
        """On a desktop, press and release keys."""
        return LiteDesktopActionSet._wrap_action(
            BaseTools._make_call(
                "key",
                keys=_normalize_canonical_keys(keys),
            )
        )

    @staticmethod
    @tool(keys="list of keys to press down.")
    def key_down(keys: list[str]) -> dict[str, Any]:
        """On a desktop, press keys down without releasing them."""
        return LiteDesktopActionSet._wrap_action(
            BaseTools._make_call(
                "key_down",
                keys=_normalize_canonical_keys(keys),
            )
        )

    @staticmethod
    @tool(keys="list of keys to release.")
    def key_up(keys: list[str]) -> dict[str, Any]:
        """On a desktop, release keys that were previously pressed down."""
        return LiteDesktopActionSet._wrap_action(
            BaseTools._make_call(
                "key_up",
                keys=_normalize_canonical_keys(keys),
            )
        )

    @staticmethod
    @tool(
        text=(
            "The text content to type. End it with a newline to press Enter "
            "after typing, e.g. to submit a search box."
        ),
    )
    def type(text: str) -> dict[str, Any]:
        """On a desktop, type text content into the currently focused input field.

        A TRAILING newline in ``text`` is an Enter press -- that is the one
        spelling for "submit", so there is no separate "press enter" argument.
        A transport that types character by character executes every real
        newline as Return; one that sets a field value directly honours only the
        trailing one and projects it onto its own flag at its own boundary.

        NOT every provider wire can carry this spelling. A wire that delimits a
        parameter body with newlines and strips the surrounding whitespace --
        the Qwen XML ``<parameter=text>`` block is the one in this repo -- cannot
        tell a payload newline from a delimiter, so a trailing newline does not
        survive a round trip through it in either direction. Those families
        spell "submit" as ``type`` followed by ``key(["enter"])`` instead, which
        does round-trip; see ``lite/agents/models/qwen3_5/adapter.py``.
        """
        return LiteDesktopActionSet._wrap_action(
            BaseTools._make_call("type", text=text)
        )

    @staticmethod
    @tool(
        keys="list of keys to hold down.",
        duration=duration_description("Duration in seconds."),
    )
    def hold_key(keys: list[str], duration: float) -> dict[str, Any]:
        """On a desktop, hold keys down for a specified duration."""
        return LiteDesktopActionSet._wrap_action(
            BaseTools._make_call(
                "hold_key",
                keys=_normalize_canonical_keys(keys),
                duration=duration,
            )
        )

    @staticmethod
    @tool(coordinate="(x, y) coordinates normalized to [0, 1000].")
    def mouse_move(coordinate: list[int]) -> dict[str, Any]:
        """On a desktop, move the mouse cursor to specified coordinates."""
        return LiteDesktopActionSet._wrap_action(
            BaseTools._make_call("mouse_move", coordinate=coordinate)
        )

    @staticmethod
    @tool(
        coordinate="(x, y) coordinates normalized to [0, 1000].",
        button="Mouse button to click.",
        clicks="Number of clicks: 1=single, 2=double, 3=triple.",
    )
    def click(
        coordinate: list[int] | None = None,
        button: Literal["left", "right", "middle"] = "left",
        clicks: Literal[1, 2, 3] = 1,
    ) -> dict[str, Any]:
        """On a desktop, perform mouse click at specified coordinates."""
        return LiteDesktopActionSet._wrap_action(
            BaseTools._make_call(
                "click",
                LiteDesktopActionSet._DEFAULTS.get("click", {}),
                coordinate=coordinate,
                button=button,
                clicks=clicks,
            )
        )

    @staticmethod
    @tool(
        coordinate="Ending (x, y) coordinates, normalized to [0, 1000].",
        start_coordinate="Starting (x, y) coordinates, normalized to [0, 1000].",
        button="Mouse button to hold while dragging.",
    )
    def drag(
        coordinate: list[int],
        start_coordinate: list[int] | None = None,
        button: Literal["left", "right", "middle"] = "left",
    ) -> dict[str, Any]:
        """On a desktop, drag the mouse from start to end coordinates."""
        return LiteDesktopActionSet._wrap_action(
            BaseTools._make_call(
                "drag",
                LiteDesktopActionSet._DEFAULTS.get("drag", {}),
                coordinate=coordinate,
                start_coordinate=start_coordinate,
                button=button,
            )
        )

    @staticmethod
    @tool(button="Mouse button to press.")
    def mouse_down(button: Literal["left", "right", "middle"] = "left") -> dict[str, Any]:
        """On a desktop, press the mouse button without releasing."""
        return LiteDesktopActionSet._wrap_action(
            BaseTools._make_call(
                "mouse_down",
                LiteDesktopActionSet._DEFAULTS.get("mouse_down", {}),
                button=button,
            )
        )

    @staticmethod
    @tool(button="Mouse button to release.")
    def mouse_up(button: Literal["left", "right", "middle"] = "left") -> dict[str, Any]:
        """On a desktop, release the mouse button."""
        return LiteDesktopActionSet._wrap_action(
            BaseTools._make_call(
                "mouse_up",
                LiteDesktopActionSet._DEFAULTS.get("mouse_up", {}),
                button=button,
            )
        )

    @staticmethod
    @tool(
        direction="The direction to scroll.",
        amount="Number of scroll units.",
        coordinate="(x, y) coordinates. If provided, cursor moves here before scrolling.",
    )
    def scroll(
        direction: Literal["up", "down", "left", "right"],
        amount: int,
        coordinate: list[int] | None = None,
    ) -> dict[str, Any]:
        """On a desktop, scroll in a specified direction by a specified amount."""
        return LiteDesktopActionSet._wrap_action(
            BaseTools._make_call(
                "scroll",
                coordinate=coordinate,
                direction=direction,
                amount=amount,
            )
        )

    @staticmethod
    @tool(duration=duration_description("Time in seconds to wait."))
    def wait(duration: float) -> dict[str, Any]:
        """On a desktop, pause execution for a specified duration."""
        return LiteDesktopActionSet._wrap_action(BaseTools._make_call("wait", duration=duration))

    @staticmethod
    @tool()
    def screenshot() -> dict[str, Any]:
        """On a desktop, take a screenshot."""
        return LiteDesktopActionSet._wrap_action(BaseTools._make_call("screenshot"))

    @staticmethod
    @tool()
    def cursor_position() -> dict[str, Any]:
        """On a desktop, get the current cursor position."""
        return LiteDesktopActionSet._wrap_action(BaseTools._make_call("cursor_position"))


LITE_DESKTOP_KEY_ACTION_NAMES: frozenset[str] = frozenset({
    "key",
    "key_down",
    "key_up",
    "hold_key",
})


@install_tool_schemas
class LiteMobileActionSet(_ActionBatchActionSet):
    """Core-owned canonical mobile GUI actions, carried by ``mobile``."""

    _ACTION_BATCH_TOOL_NAME: ClassVar[str] = LITE_MOBILE_ACTION_BATCH_TOOL_NAME
    _ACTION_BATCH_DESCRIPTION: ClassVar[str] = (
        "Perform one or more mobile GUI actions. Each item in actions is a "
        "single action mobile action."
    )

    @staticmethod
    @tool(
        coordinate="(x, y) coordinates normalized to [0, 1000].",
        clicks="Number of taps: 1=single, 2=double.",
    )
    def tap(
        coordinate: list[int],
        clicks: Literal[1, 2] = 1,
    ) -> dict[str, Any]:
        """On a mobile device, perform a tap at the specified coordinates."""
        return LiteMobileActionSet._wrap_action(
            BaseTools._make_call("tap", coordinate=coordinate, clicks=clicks)
        )

    @staticmethod
    @tool(
        coordinate="(x, y) coordinates normalized to [0, 1000].",
        duration=duration_description("Duration of the press in seconds."),
    )
    def long_press(
        coordinate: list[int],
        duration: float | None = None,
    ) -> dict[str, Any]:
        """On a mobile device, perform a long press action at the specified coordinates."""
        return LiteMobileActionSet._wrap_action(
            BaseTools._make_call("long_press", coordinate=coordinate, duration=duration)
        )

    @staticmethod
    @tool(text="The text content to type.")
    def type(text: str) -> dict[str, Any]:
        """On a mobile device, type text content into the currently focused input field."""
        return LiteMobileActionSet._wrap_action(BaseTools._make_call("type", text=text))

    @staticmethod
    @tool(
        start_coordinate="Starting (x, y) coordinates, normalized to [0, 1000].",
        coordinate="Ending (x, y) coordinates, normalized to [0, 1000].",
    )
    def swipe(start_coordinate: list[int], coordinate: list[int]) -> dict[str, Any]:
        """On a mobile device, perform a swipe gesture from a start point to an end point."""
        return LiteMobileActionSet._wrap_action(
            BaseTools._make_call(
                "swipe",
                start_coordinate=start_coordinate,
                coordinate=coordinate,
            )
        )

    @staticmethod
    @tool(
        start_coordinate="Starting (x, y) coordinates, normalized to [0, 1000].",
        coordinate="Ending (x, y) coordinates, normalized to [0, 1000].",
    )
    def drag(start_coordinate: list[int], coordinate: list[int]) -> dict[str, Any]:
        """On mobile, long-press at a start point and move to an end point."""
        return LiteMobileActionSet._wrap_action(
            BaseTools._make_call(
                "drag",
                start_coordinate=start_coordinate,
                coordinate=coordinate,
            )
        )

    @staticmethod
    @tool(
        coordinate="(x, y) center coordinates normalized to [0, 1000].",
        direction="Pinch direction: 'in' to zoom out, 'out' to zoom in.",
        amount="Pinch distance as percentage of screen (10-50). Default 25.",
    )
    def pinch(
        coordinate: list[int],
        direction: Literal["in", "out"],
        amount: int = 25,
    ) -> dict[str, Any]:
        """On a mobile device, perform a pinch gesture (two-finger zoom) at the specified center."""
        return LiteMobileActionSet._wrap_action(
            BaseTools._make_call(
                "pinch",
                coordinate=coordinate,
                direction=direction,
                amount=amount,
            )
        )

    @staticmethod
    @tool(button="The system button to press.")
    def system_button(
        button: Literal["Home", "Back", "Enter", "Menu", "Recent"],
    ) -> dict[str, Any]:
        """On a mobile device, press a system button (Home, Back, Enter, Menu, or Recent)."""
        return LiteMobileActionSet._wrap_action(
            BaseTools._make_call("system_button", button=button)
        )

    @staticmethod
    @tool(duration=duration_description("Time in seconds to wait."))
    def wait(duration: float) -> dict[str, Any]:
        """On a mobile device, pause execution for a specified duration."""
        return LiteMobileActionSet._wrap_action(BaseTools._make_call("wait", duration=duration))

    @staticmethod
    @tool()
    def screenshot() -> dict[str, Any]:
        """On a mobile device, take a screenshot."""
        return LiteMobileActionSet._wrap_action(BaseTools._make_call("screenshot"))


@install_tool_schemas
class LitePointActionSet(BaseActionSet):
    """Core-owned canonical point-grounding action surface."""

    @staticmethod
    @tool(
        coordinate=(
            "(x, y) coordinates representing the center of the object, "
            "normalized to [0, 1000]."
        ),
    )
    def point(coordinate: list[int]) -> dict[str, Any]:
        """Locate a specific object on the screen by returning its center coordinates."""
        return BaseTools._make_call("point", coordinate=coordinate)


@install_tool_schemas
class LiteBBoxActionSet(BaseActionSet):
    """Core-owned canonical bbox-grounding action surface."""

    @staticmethod
    @tool(
        coordinate=(
            "(x_min, y_min, x_max, y_max) defining the bounding box. "
            "Normalized to [0, 1000]."
        ),
    )
    def bbox(coordinate: list[int]) -> dict[str, Any]:
        """Locate a specific object on the screen by returning its bounding box coordinates."""
        return BaseTools._make_call("bbox", coordinate=coordinate)


#: The four core action sets ``valid_actions`` may select from.
ACTION_SETS: tuple[type[BaseActionSet], ...] = (
    LiteDesktopActionSet,
    LiteMobileActionSet,
    LitePointActionSet,
    LiteBBoxActionSet,
)


#: Every action name ``env_kwargs.valid_actions`` may contain.
LITE_VALID_ACTION_NAMES: frozenset[str] = frozenset().union(
    *(tools.get_action_names() for tools in ACTION_SETS)
)


#: Top-level tool names emitted by action sets: ``computer/mobile/point/bbox``.
LITE_ACTION_SET_TOOL_NAMES: frozenset[str] = frozenset().union(
    *(tools.get_tool_names() for tools in ACTION_SETS)
)


#: Action-batch tool names: ``computer`` and ``mobile``.
LITE_ACTION_BATCH_TOOL_NAMES: frozenset[str] = frozenset().union(
    *(tools.get_tool_names() for tools in (LiteDesktopActionSet, LiteMobileActionSet))
)


def is_lite_action_name_or_action_batch_tool_name(name: str) -> bool:
    """True for an action name or an action-batch tool name."""
    return name in LITE_VALID_ACTION_NAMES or name in LITE_ACTION_BATCH_TOOL_NAMES


#: Desktop/mobile action names in schema declaration order.
LITE_DESKTOP_ACTION_NAMES_IN_SCHEMA_ORDER: tuple[str, ...] = tuple(LiteDesktopActionSet._SCHEMAS)
LITE_MOBILE_ACTION_NAMES_IN_SCHEMA_ORDER: tuple[str, ...] = tuple(LiteMobileActionSet._SCHEMAS)


def lite_action_names_by_action_batch_tool() -> dict[str, frozenset[str]]:
    """Map each action-batch tool name to the action names allowed inside it."""
    return {
        tools._ACTION_BATCH_TOOL_NAME: tools.get_action_names()
        for tools in (LiteDesktopActionSet, LiteMobileActionSet)
    }


def lite_action_names_for_action_batch_tool(
    action_batch_tool_name: str,
) -> frozenset[str] | None:
    """Action names allowed inside ``action_batch_tool_name``, or ``None`` on a miss.

    ``None`` and not ``frozenset()``, because the two answers are different
    facts and one caller must tell a miss from a real action-batch tool with an
    invalid child action.
    """
    return lite_action_names_by_action_batch_tool().get(action_batch_tool_name)


def lite_action_batch_tool_name_for_platform(platform: str) -> str | None:
    """The action-batch tool for a platform, or ``None`` when unbatched."""
    if platform in {LiteCUAMetadata.Platform.DESKTOP, LiteCUAMetadata.Platform.BROWSER}:
        return LITE_COMPUTER_ACTION_BATCH_TOOL_NAME
    if platform == LiteCUAMetadata.Platform.MOBILE:
        return LITE_MOBILE_ACTION_BATCH_TOOL_NAME
    return None


def _lite_action_batch_tool_names_for_platform(platform: str) -> frozenset[str]:
    action_batch_tool = lite_action_batch_tool_name_for_platform(platform)
    return (
        frozenset({action_batch_tool})
        if action_batch_tool is not None
        else frozenset()
    )


def lite_action_set_tool_names_for_metadata(metadata: LiteCUAMetadata) -> frozenset[str]:
    """Action-set tool names active for this metadata surface."""
    if metadata.task_type == LiteCUAMetadata.TaskType.GROUNDING_POINT:
        return LitePointActionSet.get_tool_names()
    if metadata.task_type == LiteCUAMetadata.TaskType.GROUNDING_BBOX:
        return LiteBBoxActionSet.get_tool_names()
    if metadata.task_type != LiteCUAMetadata.TaskType.USE:
        return frozenset()
    return _lite_action_batch_tool_names_for_platform(metadata.platform)


def lite_builtin_tool_names_for_metadata(metadata: LiteCUAMetadata) -> frozenset[str]:
    """Tool-call names valid without ``metadata.extra_tool_schemas``."""
    if metadata.task_type == LiteCUAMetadata.TaskType.GROUNDING_ACTION:
        return _lite_action_batch_tool_names_for_platform(metadata.platform)
    return lite_action_set_tool_names_for_metadata(metadata)


__all__ = [
    "LiteDesktopActionSet",
    "LiteMobileActionSet",
    "LitePointActionSet",
    "LiteBBoxActionSet",
    "LITE_VALID_ACTION_NAMES",
    "LITE_ACTION_SET_TOOL_NAMES",
    "LITE_ACTION_BATCH_TOOL_NAMES",
    "LITE_DESKTOP_ACTION_NAMES_IN_SCHEMA_ORDER",
    "LITE_DESKTOP_KEY_ACTION_NAMES",
    "LITE_MOBILE_ACTION_NAMES_IN_SCHEMA_ORDER",
    "lite_action_names_by_action_batch_tool",
    "lite_action_names_for_action_batch_tool",
    "lite_action_batch_tool_name_for_platform",
    "is_lite_action_name_or_action_batch_tool_name",
    "lite_action_set_tool_names_for_metadata",
    "lite_builtin_tool_names_for_metadata",
]
