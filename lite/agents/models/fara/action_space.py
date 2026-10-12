"""
Fara-1.0 Action Space

Wire format: a single ``computer_use`` tool with an ``action`` parameter —
mirrors the reference ``fara/src/fara/_prompts.py::FaraComputerUse``
byte-for-byte (11-action web-browsing enum). Fara is a Qwen2.5-VL fine-tune,
so this shares Qwen2.5-VL's conventions:

  * The schema description contains the placeholder
    ``{display_width_px}x{display_height_px}``; the adapter substitutes the
    actual smart-resized image dims (factor=28) per render.
  * Coordinates ARE the resized-image pixels — no normalization here. The
    adapter rescales between cua-lite [0, 1000] and pixel-in-resized using
    ``self._current_image_size`` cached during ``render_step``.

Fara adds four web-native actions on top of the desktop verbs:
``visit_url`` / ``web_search`` / ``history_back`` / ``pause_and_memorize_fact``.
``visit_url`` ↔ cua-lite ``goto`` and ``history_back`` ↔ ``back`` map onto the
canonical browser-nav verbs (:class:`LiteBrowserNavToolSet`); ``web_search`` becomes a
``goto`` of the Bing results URL (matching the reference, which visits Bing);
``pause_and_memorize_fact`` has no browser effect and passes through by name
(a benign no-op on the env — the fact lives in the assistant thoughts).

Usage::

    from lite.agents.models.fara.action_space import FaraDesktopActionSpace
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Collection
from typing import Any, ClassVar, Literal
from urllib.parse import quote_plus

from lite.agents.core.action_space import BaseActionSpace
from lite.agents.core.action_space.base import (
    LiteDesktopActionSpace,
    LitePointActionSpace,
)
from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.agents.core.action_space.utils.geometry import (
    PIXELS_PER_CLICK,
    compact_number,
    optional_coord,
    required_coord,
    required_scroll_pixels,
    scroll_clicks,
    scroll_wire_magnitude,
)
from lite.agents.core.action_space.utils.grounding_point import (
    convert_non_point_call_for_grounding_space,
)
from lite.agents.core.action_space.utils.unknown_wrapper_action import unknown_wrapper_action_batch
from lite.agents.core.action_space.utils.wrapper_enum import filter_wrapper_action_enum
from lite.core.tools.action_space import (
    LITE_COMPUTER_ACTION_BATCH_TOOL_NAME,
    make_lite_action_batch_call,
    merge_adjacent_lite_action_batches,
)
from lite.core.tools.action_space.duration import duration_description
from lite.core.tools.calls import (
    make_tool_call,
    tool_call_arguments,
    tool_call_name,
)
from lite.core.tools.extra_tools import (
    LiteFinishToolSet,
    extra_tool_name_and_arguments_are_admitted,
)
from lite.core.tools.schemas import tool

logger = logging.getLogger(__name__)

# Fara scroll is page-based (the reference maps any positive ``pixels`` to a
# single page_up and any negative to a single page_down). We keep the shared
# pixels-per-wheel-click convention owned by
# ``lite.agents.core.action_space.utils`` so a cua-lite ``scroll(amount=N)``
# round-trips through a sensible pixel magnitude.

#: Fara's single provider-native wrapper tool. Its ``action`` enum carries the
#: whole Fara surface, so every schema gate below prunes that one enum.
_FARA_NATIVE_TOOL_NAME = "computer_use"


def _fara_action_values(cls) -> frozenset[str]:
    """Every Fara ``action`` value ``cls``'s from-agent dispatch understands.

    The wrapped GUI values plus the native web/finish values that spell
    standalone canonical tools — i.e. exactly the ``computer_use`` action enum
    the model was trained on. Read when provider output drops the wrapper and
    uses an action value as the tool name, so that flat output converts through
    the same branches as nested output.
    """
    return frozenset(
        action_value
        for action_values in cls.LITE_ACTION_NAME_TO_FARA_ACTION_VALUES.values()
        for action_value in action_values
    ) | frozenset(cls.FARA_ACTION_VALUE_TO_EXTRA_TOOL_NAMES)


def _wrap_desktop_action_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    action_names = LiteDesktopActionSpace.get_action_names()
    for call in calls:
        name = tool_call_name(call)
        if name in action_names:
            out.append(make_lite_action_batch_call(LITE_COMPUTER_ACTION_BATCH_TOOL_NAME, call))
            continue
        out.append(call)
    return merge_adjacent_lite_action_batches(out)


# =============================================================================
# Fara Desktop / Browser Action Space
# =============================================================================

@dataclasses.dataclass
class FaraDesktopActionSpace(BaseActionSpace, key=r"fara@(desktop|browser)"):
    """Fara-1.0 action space — single ``computer_use`` tool.

    Matches the reference ``FaraComputerUse`` (``_prompts.py``): 11 actions
    (``key``, ``type``, ``mouse_move``, ``left_click``, ``scroll``,
    ``visit_url``, ``web_search``, ``history_back``, ``pause_and_memorize_fact``,
    ``wait``, ``terminate``), with the ``type`` input-text key args
    (``press_enter`` / ``delete_existing_text``) included — the reference agent
    runs with ``include_input_text_key_args=True``.

    The screen-resolution sentence is templated against
    ``{display_width_px}x{display_height_px}``; the adapter substitutes the
    resized image dims per render. Coordinate space is **pixels in the
    smart-resized image**; the adapter rescales to/from cua-lite [0, 1000].
    """


    #: Fara's wire is real image pixels; its reference reads only the SIGN
    #: (``fara_agent.py:534``), so the magnitude cannot change what executes.
    SCROLL_WIRE_UNIT: ClassVar[int] = PIXELS_PER_CLICK

    platform: str = "desktop"

    _ACTION_DESCRIPTION = (
        "The action to perform. The available actions are:\n"
        "* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order. Includes \"Enter\", \"Alt\", \"Shift\", \"Tab\", \"Control\", \"Backspace\", \"Delete\", \"Escape\", \"ArrowUp\", \"ArrowDown\", \"ArrowLeft\", \"ArrowRight\", \"PageDown\", \"PageUp\", \"Shift\", etc.\n"
        "* `type`: Type a string of text on the keyboard.\n"
        "* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n"
        "* `left_click`: Click the left mouse button.\n"
        "* `scroll`: Performs a scroll of the mouse scroll wheel.\n"
        "* `visit_url`: Visit a specified URL.\n"
        "* `web_search`: Perform a web search with a specified query.\n"
        "* `history_back`: Go back to the previous page in the browser history.\n"
        "* `pause_and_memorize_fact`: Pause and memorize a fact for future reference.\n"
        "* `wait`: Wait specified seconds for the change to happen.\n"
        "* `terminate`: Terminate the current task and report its completion status."
    )

    @staticmethod
    @tool(
        action=_ACTION_DESCRIPTION,
        keys="Required only by `action=key`.",
        text="Required only by `action=type`.",
        press_enter="Whether to press the Enter key after typing. Required only by `action=type`.",
        delete_existing_text="Whether to delete existing text before typing. Required only by `action=type`.",
        coordinate="(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=left_click`, `action=mouse_move`, and `action=type`.",
        pixels="The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.",
        url="The URL to visit. Required only by `action=visit_url`.",
        query="The query to search for. Required only by `action=web_search`.",
        fact="The fact to remember for the future. Required only by `action=pause_and_memorize_fact`.",
        time=duration_description("The seconds to wait. Required only by `action=wait`."),
        status="The status of the task. Required only by `action=terminate`.",
    )
    def computer_use(
        action: Literal[
            "key", "type", "mouse_move", "left_click", "scroll",
            "visit_url", "web_search", "history_back",
            "pause_and_memorize_fact", "wait", "terminate",
        ],
        keys: list[str] | None = None,
        text: str | None = None,
        press_enter: bool | None = None,
        delete_existing_text: bool | None = None,
        coordinate: list[int] | None = None,
        pixels: int | None = None,
        url: str | None = None,
        query: str | None = None,
        fact: str | None = None,
        time: float | None = None,
        status: Literal["success", "failure"] | None = None,
    ) -> dict[str, Any]:
        """Use a mouse and keyboard to interact with a computer, and take screenshots.

        * This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.
        * Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.
        * The screen's resolution is {display_width_px}x{display_height_px}.
        * Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
        * If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.
        * Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.
        * When a separate scrollable container prominently overlays the webpage, if you want to scroll within it, you typically need to mouse_move() over it first and then scroll().
        * If a popup window appears that you want to close, if left_click() on the 'X' or close button doesn't work, try key(keys=['Escape']) to close it.
        * On some search bars, when you type(), you may need to press_enter=False and instead separately call left_click() on the search button to submit the search query. This is especially true of search bars that have auto-suggest popups for e.g. locations
        * For calendar widgets, you usually need to left_click() on arrows to move between months and left_click() on dates to select them; type() is not typically used to input dates there.
        """
        return make_tool_call(
            _FARA_NATIVE_TOOL_NAME,
            {
                "action": action,
                "keys": keys,
                "text": text,
                "press_enter": press_enter,
                "delete_existing_text": delete_existing_text,
                "coordinate": coordinate,
                "pixels": pixels,
                "url": url,
                "query": query,
                "fact": fact,
                "time": time,
                "status": status,
            },
        )

    # -------------------------------------------------------------------------
    # Action filtering
    # -------------------------------------------------------------------------

    # CUA-lite GUI action name → the Fara action values it maps to.
    LITE_ACTION_NAME_TO_FARA_ACTION_VALUES = {
        "click": ["left_click"],
        "type": ["type"], "key": ["key"], "mouse_move": ["mouse_move"],
        "scroll": ["scroll"], "wait": ["wait"],
    }
    # Fara's native web/finish action values are the wire spelling of canonical
    # STANDALONE extra tools; they are scoped by
    # ``filter_fara_action_values_for_active_extra_tools`` from
    # ``metadata.extra_tool_schemas``, never by ``valid_actions``.
    #
    # The extra-tool name is the one ``_convert_single_from_agent`` PRODUCES.
    # Hence ``web_search -> goto``:
    # the native action navigates to a Bing results page, so it parses to
    # ``goto(url=...)`` and is usable exactly when ``goto`` is active (the
    # ``to_agent`` direction additionally accepts standalone native ``web_search``,
    # but no env offers that schema).
    FARA_ACTION_VALUE_TO_EXTRA_TOOL_NAMES = {
        "visit_url": frozenset({"goto"}),
        "web_search": frozenset({"goto"}),
        "history_back": frozenset({"back"}),
        "pause_and_memorize_fact": frozenset({"pause_and_memorize_fact"}),
        "terminate": frozenset({"terminate"}),
    }

    @classmethod
    def filter_tool_schemas_for_valid_actions(
        cls,
        schemas: list[dict[str, Any]],
        valid_actions: list[str],
    ) -> list[dict[str, Any]]:
        """Keep the Fara action values a ``valid_actions`` gate leaves reachable.

        GUI-only gate: every native web/finish action value stays, because the
        active-extra gate below owns those.
        """
        allowed = {
            value
            for name in valid_actions
            for value in cls.LITE_ACTION_NAME_TO_FARA_ACTION_VALUES.get(name) or ()
        }
        allowed.update(cls.FARA_ACTION_VALUE_TO_EXTRA_TOOL_NAMES)
        return filter_wrapper_action_enum(
            schemas,
            wrapper_tool_name=_FARA_NATIVE_TOOL_NAME,
            keep_action_value=lambda value: value in allowed,
        )

    @classmethod
    def filter_fara_action_values_for_active_extra_tools(
        cls,
        schemas: list[dict[str, Any]],
        active_extra_tool_names: Collection[str],
    ) -> list[dict[str, Any]]:
        """Drop native action values whose canonical extra tool is not offered.

        Orthogonal to the ``valid_actions`` gate: this one is scoped by the
        env's active standalone extra tools, and touches only the action values
        in :attr:`FARA_ACTION_VALUE_TO_EXTRA_TOOL_NAMES`.
        """
        active = set(active_extra_tool_names)
        inactive = {
            value
            for value, extra_tool_names in cls.FARA_ACTION_VALUE_TO_EXTRA_TOOL_NAMES.items()
            if extra_tool_names.isdisjoint(active)
        }
        if not inactive:
            return list(schemas)
        return filter_wrapper_action_enum(
            schemas,
            wrapper_tool_name=_FARA_NATIVE_TOOL_NAME,
            keep_action_value=lambda value: value not in inactive,
        )

    # -------------------------------------------------------------------------
    # Tool Call Conversion (cua-lite ↔ agent format; coords pass through;
    # the adapter handles [0,1000] ↔ pixel rescaling).
    # -------------------------------------------------------------------------

    def _convert_single_to_agent(self, tool_call: dict[str, Any]) -> list[dict[str, Any]]:
        name = tool_call_name(tool_call)
        args = tool_call_arguments(tool_call)

        results = []

        if name == "computer":
            for child in args["actions"]:
                action = child["action"]
                action_args = {k: v for k, v in child.items() if k != "action"}
                results.extend(self._convert_single_to_agent(make_tool_call(action, action_args)))

        elif name == "click":
            # Fara's action enum has ``left_click`` and no right/middle/double
            # counterpart, so ``button`` and ``clicks`` have no carrier: rewriting
            # them onto a plain left click re-parses as a DIFFERENT action than
            # the caller asked for. Raise rather than silently substitute.
            button = args.get("button", "left")
            clicks = args.get("clicks", 1)
            if button != "left" or clicks != 1:
                raise ValueError(
                    f"Fara cannot render click(button={button!r}, clicks={clicks}): "
                    "its action enum has only 'left_click' and carries neither a "
                    "button nor a repeat count"
                )
            results.append(FaraDesktopActionSpace.computer_use(
                action="left_click",
                coordinate=args.get("coordinate"),
            )["function"])

        elif name == "type":
            # Canonical's trailing newline is Fara's ``press_enter``; without
            # this the Enter would be dropped on the way back to the model.
            text = args.get("text", "")
            type_kwargs: dict[str, Any] = {}
            if text.endswith("\n"):
                text, type_kwargs["press_enter"] = text[:-1], True
            results.append(FaraDesktopActionSpace.computer_use(
                action="type",
                text=text,
                **type_kwargs,
            )["function"])

        elif name == "key":
            results.append(FaraDesktopActionSpace.computer_use(
                action="key",
                keys=args.get("keys", []),
            )["function"])

        elif name == "mouse_move":
            results.append(FaraDesktopActionSpace.computer_use(
                action="mouse_move",
                coordinate=args.get("coordinate"),
            )["function"])

        elif name == "scroll":
            direction = args.get("direction", "down")
            amount = args.get("amount", 3)
            if direction not in ("up", "down"):
                # Fara's only scroll carrier is the signed scalar ``pixels``, whose
                # single axis is vertical -- the parse side reads the sign back as
                # ``up``/``down`` and nothing else. There is no faithful spelling
                # of a horizontal scroll, so raise rather than silently substitute
                # the vertical axis.
                raise ValueError(
                    f"Fara cannot render scroll(direction={direction!r}): its "
                    "action enum has no 'hscroll' and 'pixels' carries the "
                    "vertical axis only"
                )
            # Fara ``pixels``: positive = up, negative = down (see reference
            # execute_action).
            # ``int(round(...))``: the schema type is integer, so a float
            # ``amount`` from source data must not leak a float onto the wire.
            scroll_pixels = scroll_wire_magnitude(int(round(amount)),
                                                  wire_unit=type(self).SCROLL_WIRE_UNIT)
            if direction == "down":
                scroll_pixels = -scroll_pixels
            results.append(FaraDesktopActionSpace.computer_use(
                action="scroll",
                pixels=scroll_pixels,
            )["function"])

        elif name == "goto":
            results.append(FaraDesktopActionSpace.computer_use(
                action="visit_url",
                url=args.get("url", ""),
            )["function"])

        elif name == "back":
            results.append(FaraDesktopActionSpace.computer_use(
                action="history_back",
            )["function"])

        elif name == "web_search":
            results.append(FaraDesktopActionSpace.computer_use(
                action="web_search",
                query=args.get("query", ""),
            )["function"])

        elif name == "pause_and_memorize_fact":
            results.append(FaraDesktopActionSpace.computer_use(
                action="pause_and_memorize_fact",
                fact=args.get("fact", ""),
            )["function"])

        elif name == "wait":
            results.append(FaraDesktopActionSpace.computer_use(
                action="wait",
                time=compact_number(args.get("duration", 3)),
            )["function"])

        elif name == "terminate":
            results.append(FaraDesktopActionSpace.computer_use(
                action="terminate",
                status=args.get("status", "success"),
            )["function"])

        elif name == "response":
            raise ValueError("Fara cannot render canonical tool 'response'")

        elif name in LiteDesktopActionSpace.get_action_names():
            # A canonical GUI action with NO branch above has no Fara enum entry
            # (``key_down``/``key_up``/``hold_key``/``drag``/``mouse_down``/
            # ``mouse_up``/``screenshot``/``cursor_position``). Warning and
            # returning ``[]`` deleted the action from the wire: the round trip
            # came back EMPTY, so the caller could not tell an unrenderable
            # action from a rendered one. Raise, as the ``ui_tars`` grammars do.
            raise ValueError(
                f"Fara cannot render canonical tool {name!r}: its action enum has "
                "no counterpart, and dropping it silently loses the action"
            )

        else:
            logger.warning("Unsupported CUA-lite action for Fara: %s(%s)", name, args)

        return results

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[dict[str, Any]],
        *,
        active_extra_tool_names: set[str] | None = None,
        active_extra_tool_schemas: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Parse ``computer_use`` calls back to canonical Lite calls.

        Unknown-action policy: a ``computer_use`` action this family cannot
        express becomes an INVALID ``computer`` action-batch naming the model's
        own action value, so env ingress rejects it and the model sees the
        rejection. Non-wrapper tool names stay by-name calls for env ingress to
        judge.

        Two subcases fail differently on purpose: a wire-level ``response``
        raises, because Fara has no answer channel and swallowing an answer is
        worse than a loud parse failure; a ``response`` embedded in
        ``computer_use`` is dropped, so an off-schema wrapper payload can never
        masquerade as a submitted answer.
        """
        result = []
        for tc in agent_tool_calls:
            result.extend(_wrap_desktop_action_calls(self._convert_single_from_agent(
                tc,
                active_extra_tool_names=active_extra_tool_names,
                active_extra_tool_schemas=active_extra_tool_schemas,
            )))
        return merge_adjacent_lite_action_batches(result)

    def _convert_single_from_agent(
        self,
        agent_tool_call: dict[str, Any],
        *,
        active_extra_tool_names: set[str] | None = None,
        active_extra_tool_schemas: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        name = agent_tool_call["name"]
        args = agent_tool_call["arguments"]
        action = args.get("action", "")

        if name == "response":
            # Fara has no native answer channel, so a wire-level ``response``
            # cannot be turned into a canonical call this family could ever
            # render back (see ``_convert_single_to_agent``). Fail loudly
            # instead of swallowing it.
            raise ModelToolCallParseError("Fara cannot parse wire tool 'response'")

        if name != _FARA_NATIVE_TOOL_NAME:
            admitted = extra_tool_name_and_arguments_are_admitted(
                name,
                args,
                active_extra_tool_names=active_extra_tool_names,
                active_extra_tool_schemas=active_extra_tool_schemas,
            )
            # The model dropped the ``computer_use`` wrapper and used the native
            # action value AS the tool name, e.g. ``left_click(coordinate=...)``.
            # Read it as that action so flat output converts through the same
            # branches as nested output — ``visit_url``/``web_search``/
            # ``history_back``/``terminate`` included. An ACTIVE env extra
            # outranks a colliding native value: browsergym's
            # ``scroll(delta_y=...)`` is the env's tool, not Fara's page gesture.
            if not action and not admitted and name in _fara_action_values(type(self)):
                action = name
            elif not action or admitted:
                # A standalone tool (env extra, custom tool): its Lite name IS its
                # wire name.
                return [make_tool_call(name, args)]
            # Otherwise ``action`` is set under a non-wrapper name: the model kept
            # the native nesting but got the tool name wrong; the dispatch claims
            # it, and the tail guard turns an unknown one into a by-name call.

        if action == "left_click":
            return [LiteDesktopActionSpace.click(
                coordinate=required_coord(args.get("coordinate"), dimensions=2),
            )]
        if action == "mouse_move":
            return [LiteDesktopActionSpace.mouse_move(
                coordinate=required_coord(args.get("coordinate"), dimensions=2),
            )]
        if action == "type":
            # Fara's ``type`` carries a ``coordinate`` (click the target field
            # first), plus ``press_enter`` / ``delete_existing_text`` — see
            # ``FaraComputerUse`` (_prompts.py) and the reference ``fill_coords``
            # (click coord → clear → type → optional Enter). cua-lite ``type``
            # only types into the already-focused element, so decompose into:
            #   click(coordinate)  → focuses the field AND draws the crosshair
            #   type(text)         → types into it
            # ``press_enter`` becomes canonical's trailing newline, which every
            # transport executes as Return; the container clears existing text
            # itself, so ``delete_existing_text`` needs no separate mapping.
            out: list[dict[str, Any]] = []
            focus_coordinate = optional_coord(args.get("coordinate"), dimensions=2)
            if focus_coordinate is not None:
                out.append(LiteDesktopActionSpace.click(coordinate=focus_coordinate))
            text = args.get("text", "")
            if args.get("press_enter"):
                text = f"{text}\n"
            out.append(LiteDesktopActionSpace.type(text=text))
            return out
        if action == "key":
            raw_keys = args.get("keys", [])
            return [LiteDesktopActionSpace.key(keys=raw_keys)]
        if action == "scroll":
            # ``pixels`` is REQUIRED — it is the only carrier of the direction.
            # See ``required_scroll_pixels`` for why a default is never right.
            scroll_pixels = required_scroll_pixels(args, action)
            direction = "down" if scroll_pixels < 0 else "up"
            amount = scroll_clicks(scroll_pixels)
            return [LiteDesktopActionSpace.scroll(direction=direction, amount=amount)]
        if action == "wait":
            return [LiteDesktopActionSpace.wait(duration=float(args.get("time", 3)))]
        if action == "terminate":
            return [LiteFinishToolSet.terminate(status=args.get("status", "success"))]
        # Web-native Fara actions → canonical browser-nav verbs / by-name passthrough.
        if action == "visit_url":
            return [make_tool_call("goto", {"url": args.get("url", "")})]
        if action == "history_back":
            return [make_tool_call("back")]
        if action == "web_search":
            # Fara's ``web_search`` navigates to a Bing results page (reference
            # ``fara_agent.py``: ``visit_page("https://www.bing.com/search?q=...")``).
            # cua-lite has no ``web_search`` verb and the browser envs don't execute
            # one (→ "unknown action web_search" no-op), so map it to ``goto`` of
            # the same Bing URL — which every browser env executes.
            url = f"https://www.bing.com/search?q={quote_plus(args.get('query', ''))}&FORM=QBLH"
            return [make_tool_call("goto", {"url": url})]
        if action == "pause_and_memorize_fact":
            return [
                make_tool_call(
                    "pause_and_memorize_fact",
                    {"fact": args.get("fact", "")},
                )
            ]
        if name != _FARA_NATIVE_TOOL_NAME:
            # A non-wrapper tool name that merely carried an ``action`` argument
            # naming nothing Fara knows: it is not a Fara GUI action, so it stays
            # a by-name call for env ingress to judge — same as the non-wrapper
            # branch at the top of this method.
            return [make_tool_call(name, args)]

        logger.warning("Unknown Fara computer_use action: %s(%s)", action, args)
        return [unknown_wrapper_action_batch(LITE_COMPUTER_ACTION_BATCH_TOOL_NAME, args)]


# =============================================================================
# Fara Grounding (point) — trimmed single-step click
# =============================================================================

@dataclasses.dataclass
class FaraDesktopGroundingPointActionSpace(BaseActionSpace, key=r"fara@(desktop|browser)@point"):
    """Trimmed ``computer_use`` schema: ``left_click`` only (grounding eval).

    Mirrors :class:`Qwen2_5VLDesktopGroundingPointActionSpace` but pulls in the
    Fara screen-resolution wording. Coordinate space is pixel-in-resized; the
    adapter rescales to / from cua-lite [0, 1000]. Fara has no native grounding
    mode (it is a web-browsing agent) — this exists so single-step click
    benchmarks (ScreenSpot-Pro) can route through the Fara family.
    """

    platform: str = "desktop"

    _ACTION_DESCRIPTION = (
        "The action to perform. The available actions are:\n"
        "* `left_click`: Click the left mouse button at the (x, y) pixel coordinate of the target element on the screen."
    )

    @staticmethod
    @tool(
        action=_ACTION_DESCRIPTION,
        coordinate="(x, y): The x (pixels from the left edge) and y (pixels from the top edge) pixel coordinate of the target element. Required by `action=left_click`.",
    )
    def computer_use(
        action: Literal["left_click"],
        coordinate: list[int] | None = None,
    ) -> dict[str, Any]:
        """Use a mouse to click on the target element in the screenshot.

        * The screen's resolution is {display_width_px}x{display_height_px}.
        * Click with the cursor tip centered on the target; avoid edges unless the instruction asks for an edge.
        * Do not use other tools. Only ``left_click`` is available.
        """
        return make_tool_call(
            _FARA_NATIVE_TOOL_NAME,
            {"action": action, "coordinate": coordinate},
        )

    LITE_ACTION_NAME_TO_FARA_ACTION_VALUES = {"point": ["left_click"]}

    @classmethod
    def filter_tool_schemas_for_valid_actions(
        cls,
        schemas: list[dict[str, Any]],
        valid_actions: list[str],
    ) -> list[dict[str, Any]]:
        """Keep the trimmed ``left_click`` schema only while ``point`` is valid.

        This surface advertises no native standalone tools, so there is no
        active-extra gate: ``left_click`` is the whole enum.
        """
        allowed = {
            value
            for name in valid_actions
            for value in cls.LITE_ACTION_NAME_TO_FARA_ACTION_VALUES.get(name) or ()
        }
        return filter_wrapper_action_enum(
            schemas,
            wrapper_tool_name=_FARA_NATIVE_TOOL_NAME,
            keep_action_value=lambda value: value in allowed,
        )

    def convert_tool_calls_to_agent(
        self, tool_calls: list[dict[str, Any]], **kwargs,
    ) -> list[dict[str, Any]]:
        result = []
        for tc in tool_calls:
            name = tool_call_name(tc)
            args = tool_call_arguments(tc)
            if name == "point":
                result.append(
                    FaraDesktopGroundingPointActionSpace.computer_use(
                        action="left_click",
                        coordinate=args.get("coordinate"),
                    )["function"]
                )
            else:
                result.extend(convert_non_point_call_for_grounding_space(
                    tc, surface="Fara grounding (point) action_space",
                ))
        return result

    def convert_tool_calls_from_agent(
        self, agent_tool_calls: list[dict[str, Any]], **kwargs,
    ) -> list[dict[str, Any]]:
        result = []
        for tc in agent_tool_calls:
            name = tc["name"]
            args = tc["arguments"]
            action = args.get("action", "")
            if name != _FARA_NATIVE_TOOL_NAME:
                result.append(make_tool_call(name, args))
                continue
            if action == "left_click":
                result.append(LitePointActionSpace.point(
                    coordinate=required_coord(args.get("coordinate"), dimensions=2),
                ))
            else:
                logger.warning("Fara grounding (point) dropping off-schema action %s(%s)", action, args)
        return result
