"""
Gemini action space for CUA-Lite desktop and browser agents.

Gemini's ``computerUse`` is an opaque server-side tool: the client never
declares an action schema, so this space is purely a projection surface between
the provider's flat function calls and canonical Lite calls. Provider tool
assembly lives in ``GeminiDesktopUseAgent``.

Two properties make this family simpler than the Claude/GPT desktop spaces:

* **Coordinates are identity.** Gemini emits flat ``x``/``y`` integers on the
  same 0-1000 grid canonical Lite uses (both of Google's reference clients
  denormalize with ``x / 1000 * width``), so there is no pixel domain to cross
  and ``resolution`` is accepted and ignored. Canonical ``MAX_NORM`` is
  inclusive, so 1000 is a legal canonical value the wire cannot carry; the
  render leg clamps it to 999. **Neither behaviour is falsifiable by the
  round-trip suite** (its identity frame is 1000x1000 and its cases use
  mid-range coordinates), so both are documented here rather than tested.
* **The wire is provider-flat.** ``{"name": ..., "arguments": {...}}`` on both
  legs; Gemini has no nested ``action`` field like the Anthropic/OpenAI native
  computer tools.

Conversion example::

    from lite.agents.models.gemini.action_space import GeminiDesktopActionSpace

    action_space = GeminiDesktopActionSpace()

    gemini_calls = [{"name": "click", "arguments": {"x": 500, "y": 300}}]
    lite_calls = action_space.convert_tool_calls_from_agent(gemini_calls)
    gemini_calls = action_space.convert_tool_calls_to_agent(lite_calls)
"""

from __future__ import annotations

import copy
import dataclasses
import math
from typing import Any

from lite.agents.core.action_space import BaseActionSpace
from lite.agents.core.action_space.base import (
    LiteDesktopActionSpace,
    LiteMobileActionSpace,
)
from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.agents.core.action_space.utils.geometry import (
    PIXELS_PER_CLICK,
    required_model_text,
)
from lite.core.tools.action_space import merge_adjacent_lite_action_batches
from lite.core.tools.action_space.geometry import MAX_NORM
from lite.core.tools.calls import make_tool_call, tool_call_arguments, tool_call_name

# =============================================================================
# Server-side verb vocabulary
# =============================================================================
#
# Enumerated against the live API, not the docs: excluding a verb the
# environment does not offer is rejected server-side, which makes
# ``excludedPredefinedFunctions`` an exact oracle. The control that makes it
# conclusive is ``navigate`` — accepted under ENVIRONMENT_BROWSER, rejected
# under ENVIRONMENT_DESKTOP with the same request shape.
#
# Do NOT declare these as ``@tool`` methods. ``_SCHEMAS`` has exactly two
# readers -- ``get_tool_schemas()`` (overridden to ``[]`` here) and
# ``get_declared_action_schema_names()`` (not used for admission; see
# ``visible_predefined_verbs``) -- so declarations would have no consumer and
# would become a second representation of this same set.

#: Gemini verb -> the canonical actions parsing it can produce.
#:
#: Doubles as the ``valid_actions`` gate: a verb stays visible only when every
#: canonical action it can produce is allowed. ``mouse_down``/``mouse_up`` carry
#: two because canonical ``mouse_down`` has no coordinate, so the wire position
#: becomes a separate ``mouse_move``.
_DESKTOP_VERB_TO_LITE_ACTIONS: dict[str, tuple[str, ...]] = {
    "click": ("click",),
    "double_click": ("click",),
    "triple_click": ("click",),
    "middle_click": ("click",),
    "right_click": ("click",),
    "move": ("mouse_move",),
    "mouse_down": ("mouse_move", "mouse_down"),
    "mouse_up": ("mouse_move", "mouse_up"),
    "type": ("type",),
    "drag_and_drop": ("drag",),
    "press_key": ("key",),
    "hotkey": ("key",),
    "key_down": ("key_down",),
    "key_up": ("key_up",),
    "scroll": ("scroll",),
    "take_screenshot": ("screenshot",),
    "wait": ("wait",),
}

#: ENVIRONMENT_BROWSER offers these three on top of the desktop set.
#:
#: They have no canonical action: browser navigation is an env extra tool
#: spelled ``goto``/``back``/``forward`` (``LiteBrowserNavToolSet``), not
#: ``navigate``. So
#: they are always excluded and the parse leg never sees them.
_BROWSER_ONLY_VERBS = frozenset({"navigate", "go_back", "go_forward"})

ENVIRONMENT_DESKTOP = "ENVIRONMENT_DESKTOP"
ENVIRONMENT_BROWSER = "ENVIRONMENT_BROWSER"

#: Every ``(button, clicks)`` pair the Gemini desktop wire can spell. Each verb
#: pins both facts at once -- ``middle_click`` is single by definition,
#: ``triple_click`` is left by definition -- so the combinations outside this
#: table have no faithful spelling and must raise rather than round down.
_CLICK_WIRE_ACTIONS: dict[tuple[str, int], str] = {
    ("left", 1): "click",
    ("left", 2): "double_click",
    ("left", 3): "triple_click",
    ("right", 1): "right_click",
    ("middle", 1): "middle_click",
}

#: Canonical ``[0, MAX_NORM]`` is inclusive; the wire is 0-999. The far edge is
#: the one canonical value the wire cannot carry, so the render leg clamps it to
#: the last addressable cell -- the same policy ``strict_norm_to_pixel(clamp=True)``
#: applies in pixel space.
_GEMINI_MAX_COORD = MAX_NORM - 1


# =============================================================================
# Argument helpers
# =============================================================================


def _is_number(value: Any) -> bool:
    """True for a finite model-supplied number.

    ``json.loads`` parses ``NaN``/``Infinity`` into Python floats, and an
    arbitrary-precision integer literal into an int too large for a float.
    Neither can address a screen, and ``int()`` on them raises a bare
    ``ValueError``/``OverflowError`` rather than the owned parse error.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _require_number(args: dict[str, Any], key: str, *, verb: str) -> int | float:
    if not _is_number(args.get(key)):
        raise ModelToolCallParseError(f"{verb} requires numeric {key}")
    return args[key]


def _require_xy(args: dict[str, Any], *, verb: str, x: str = "x", y: str = "y") -> list[int]:
    """Read a flat ``x``/``y`` pair.

    Gemini spells coordinates as two scalar fields, not an ``[x, y]`` array, and
    they are already on the canonical grid -- no projection, no ``resolution``.
    """
    return [
        int(_require_number(args, x, verb=verb)),
        int(_require_number(args, y, verb=verb)),
    ]


def _clamp(coordinate: list[int]) -> list[int]:
    """Canonical -> wire. See ``_GEMINI_MAX_COORD``."""
    return [min(int(value), _GEMINI_MAX_COORD) for value in coordinate]


def _require_clamped(args: dict[str, Any], key: str, *, name: str) -> list[int]:
    """Clamp a coordinate the Gemini wire requires but canonical leaves optional.

    ``click``/``scroll`` may omit ``coordinate`` and ``drag`` may omit
    ``start_coordinate``; every Gemini verb spells its coordinates explicitly.
    Raising keeps the substitution unrepresentable -- inventing a default here
    would write a position the caller never chose back out as canonical fact.
    Same message shape as the Claude and GPT desktop spaces, though those also
    validate that the pair is numeric; this leg consumes canonical calls from
    trusted producers, not model output, so a malformed pair surfaces as the
    raw unpack/int error rather than an owned one.
    """
    coordinate = args.get(key)
    if coordinate is None:
        raise ValueError(f"{name} requires {key}")
    return _clamp(coordinate)


def _flat(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """One provider-flat wire call."""
    return {"name": name, "arguments": arguments}


# =============================================================================
# Gemini Desktop / Browser Action Space
# =============================================================================


@dataclasses.dataclass
class GeminiDesktopActionSpace(BaseActionSpace, key=r"gemini@(desktop|browser)"):
    """Projection between Gemini's native desktop/browser verbs and canonical Lite.

    One class serves both platforms because the server offers the same verbs
    plus three browser-only navigation ones; ``platform`` is passed to every
    policy method rather than read off the instance, because ``AgentRegistry``
    does not back-fill ``self.platform`` the way ``ActionSpaceRegistry`` does.
    """

    platform: str = "desktop"

    # -- tool layer ------------------------------------------------------------

    @classmethod
    def get_tool_schemas(cls, include: list[str] | None = None) -> list[dict[str, Any]]:
        """Gemini desktop uses the native ``computerUse`` tool, not function schemas.

        This space is the native <-> Lite conversion surface. Provider tool
        assembly happens in ``GeminiDesktopUseAgent._build_tools``.
        """
        return []

    # -- provider tool policy --------------------------------------------------

    @staticmethod
    def environment_for_platform(*, platform: str) -> str:
        """The ``ComputerUse.environment`` value this platform runs against."""
        return ENVIRONMENT_BROWSER if platform == "browser" else ENVIRONMENT_DESKTOP

    @classmethod
    def predefined_verbs(cls, *, platform: str) -> frozenset[str]:
        """The verbs the server offers for this platform."""
        desktop = frozenset(_DESKTOP_VERB_TO_LITE_ACTIONS)
        return desktop | _BROWSER_ONLY_VERBS if platform == "browser" else desktop

    @classmethod
    def excluded_predefined_functions(
        cls,
        *,
        platform: str,
        valid_actions: list[str] | None,
    ) -> frozenset[str]:
        """Verbs to strip from ``computerUse`` for this request.

        Two sources: the browser-only navigation verbs (no canonical home) and
        the ``valid_actions`` gate. The result is intersected with the offered
        set, which is what makes "excluded a verb this environment does not
        have" -- an opaque server-side rejection -- unrepresentable.
        """
        offered = cls.predefined_verbs(platform=platform)
        excluded = set(_BROWSER_ONLY_VERBS)
        if valid_actions is not None:
            allowed = set(valid_actions)
            excluded |= {
                verb
                for verb, lite_actions in _DESKTOP_VERB_TO_LITE_ACTIONS.items()
                if not set(lite_actions) <= allowed
            }
        return frozenset(excluded) & offered

    @classmethod
    def visible_predefined_verbs(
        cls,
        *,
        platform: str,
        valid_actions: list[str] | None,
    ) -> frozenset[str]:
        """Admission set: what the model may actually emit this request.

        NOT ``get_declared_action_schema_names()`` -- this space declares no
        ``@tool`` methods, and a static set could not see the exclusion anyway.
        """
        return cls.predefined_verbs(platform=platform) - cls.excluded_predefined_functions(
            platform=platform, valid_actions=valid_actions
        )

    # -- parse: Gemini -> canonical -------------------------------------------

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[Any],
        *,
        resolution: tuple[int, int] | None = None,
        active_extra_tool_names: set[str] | None = None,
        active_extra_tool_schemas: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Provider-flat calls -> canonical Lite calls.

        ``resolution`` is accepted and ignored: coordinates are already on the
        canonical grid.
        """
        result: list[dict[str, Any]] = []
        for call in agent_tool_calls:
            result.extend(self._convert_single_from_agent(call))
        return merge_adjacent_lite_action_batches(result)

    def _convert_single_from_agent(self, call: dict[str, Any]) -> list[dict[str, Any]]:
        verb = call.get("name", "")
        args = call.get("arguments") or {}

        if verb in ("click", "double_click", "triple_click"):
            clicks = {"click": 1, "double_click": 2, "triple_click": 3}[verb]
            coordinate = _require_xy(args, verb=verb)
            if clicks == 1:
                return [LiteDesktopActionSpace.click(coordinate=coordinate)]
            return [LiteDesktopActionSpace.click(coordinate=coordinate, clicks=clicks)]
        if verb in ("right_click", "middle_click"):
            button = "right" if verb == "right_click" else "middle"
            return [
                LiteDesktopActionSpace.click(coordinate=_require_xy(args, verb=verb), button=button)
            ]
        if verb == "move":
            return [LiteDesktopActionSpace.mouse_move(coordinate=_require_xy(args, verb=verb))]
        if verb in ("mouse_down", "mouse_up"):
            # Canonical mouse_down/up carry a button and no coordinate, so the
            # wire position becomes its own mouse_move. Gemini's x/y are
            # required, so unlike Claude's `left_mouse_down` this is unconditional.
            coordinate = _require_xy(args, verb=verb)
            press = (
                LiteDesktopActionSpace.mouse_down()
                if verb == "mouse_down"
                else LiteDesktopActionSpace.mouse_up()
            )
            return [LiteDesktopActionSpace.mouse_move(coordinate=coordinate), press]
        if verb == "type":
            # Gemini has a dedicated ``press_enter`` field; canonical spells the
            # same intent as a trailing newline, which every transport executes
            # as Return. ``press_enter=False`` and an absent field both mean "do
            # not submit", so both leave the text alone.
            text = args.get("text", "")
            if args.get("press_enter"):
                text = f"{text}\n"
            return [LiteDesktopActionSpace.type(text=text)]
        if verb == "drag_and_drop":
            # Gemini's (start_x, start_y) is the START; canonical `coordinate`
            # is the END and `start_coordinate` the start. Swapping them is
            # silent -- no shape test can catch it.
            return [
                LiteDesktopActionSpace.drag(
                    start_coordinate=_require_xy(args, verb=verb, x="start_x", y="start_y"),
                    coordinate=_require_xy(args, verb=verb, x="end_x", y="end_y"),
                )
            ]
        if verb == "press_key":
            # Gemini spells W3C KeyboardEvent.key names ("Control", "Escape",
            # "AudioVolumeUp"). Pass the raw string to the shared key owner so
            # source-ingress chord strings split there instead of becoming list
            # tokens such as ["ctrl+s"].
            key = args.get("key", "")
            if key and not isinstance(key, str):
                raise ModelToolCallParseError("press_key requires string key")
            return [LiteDesktopActionSpace.key(keys=key)]
        if verb == "hotkey":
            keys = args.get("keys")
            if not isinstance(keys, list):
                raise ModelToolCallParseError("hotkey requires a list of keys")
            return [LiteDesktopActionSpace.key(keys=keys)]
        if verb in ("key_down", "key_up"):
            # Wire field is singular; pass the raw string through the shared key
            # owner before canonical storage.
            keys = args.get("key", "")
            if keys and not isinstance(keys, str):
                raise ModelToolCallParseError(f"{verb} requires string key")
            factory = (
                LiteDesktopActionSpace.key_down
                if verb == "key_down"
                else LiteDesktopActionSpace.key_up
            )
            return [factory(keys=keys)]
        if verb == "scroll":
            direction = args.get("direction")
            if direction not in ("up", "down", "left", "right"):
                raise ModelToolCallParseError(f"scroll requires a direction, got {direction!r}")
            # Despite the name, magnitude_in_pixels is a distance on the same
            # 0-1000 grid as the coordinates (documented range 0-999, and the
            # model emits 500 for "half a screen"). PIXELS_PER_CLICK is itself
            # calibrated on that grid, so the division is dimensionally sound.
            magnitude = args.get("magnitude_in_pixels")
            magnitude = (
                300
                if magnitude is None
                else int(_require_number(args, "magnitude_in_pixels", verb=verb))
            )
            return [
                LiteDesktopActionSpace.scroll(
                    direction=direction,
                    amount=max(1, round(magnitude / PIXELS_PER_CLICK)),
                    coordinate=_require_xy(args, verb=verb),
                )
            ]
        if verb == "take_screenshot":
            return [LiteDesktopActionSpace.screenshot()]
        if verb == "wait":
            seconds = args.get("seconds")
            duration = (
                1.0 if seconds is None else float(_require_number(args, "seconds", verb=verb))
            )
            return [LiteDesktopActionSpace.wait(duration=duration)]

        raise ModelToolCallParseError(f"unknown Gemini desktop action: {verb or '<unknown>'}")

    # -- render: canonical -> Gemini -------------------------------------------

    def convert_tool_calls_to_agent(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        resolution: tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        result: list[Any] = []
        for call in tool_calls:
            result.extend(self._convert_single_to_agent(call))
        return result

    def _convert_single_to_agent(self, tool_call: dict[str, Any]) -> list[Any]:
        name = tool_call_name(tool_call)
        args = tool_call_arguments(tool_call)

        if name == "computer":
            # Canonical action batch: unwrap and recurse per child.
            out: list[Any] = []
            for child in args.get("actions", []):
                child_args = {k: v for k, v in child.items() if k != "action"}
                out.extend(
                    self._convert_single_to_agent(make_tool_call(child["action"], child_args))
                )
            return out

        if name == "click":
            button = args.get("button", "left")
            clicks = args.get("clicks", 1)
            verb = _CLICK_WIRE_ACTIONS.get((button, clicks))
            if verb is None:
                spellable = ", ".join(f"{b}x{c}" for b, c in sorted(_CLICK_WIRE_ACTIONS))
                raise ValueError(
                    f"Gemini cannot express click(button={button!r}, clicks={clicks}): "
                    f"the Gemini desktop wire spells only {spellable}"
                )
            x, y = _require_clamped(args, "coordinate", name="click")
            return [_flat(verb, {"x": x, "y": y})]
        if name == "mouse_move":
            x, y = _clamp(args["coordinate"])
            return [_flat("move", {"x": x, "y": y})]
        if name == "drag":
            button = args.get("button", "left")
            if button != "left":
                raise ValueError(
                    f"Gemini cannot express drag(button={button!r}): "
                    "drag_and_drop carries no button"
                )
            sx, sy = _require_clamped(args, "start_coordinate", name="drag")
            ex, ey = _clamp(args["coordinate"])
            return [
                _flat(
                    "drag_and_drop",
                    {"start_x": sx, "start_y": sy, "end_x": ex, "end_y": ey},
                )
            ]
        if name == "type":
            # Inverse of the from-agent mapping: canonical's trailing newline is
            # Gemini's ``press_enter``. Emit the field only when it is true -- a
            # literal False would add a key the model never sent.
            text = args.get("text", "")
            wire: dict[str, Any] = {"text": text}
            if text.endswith("\n"):
                wire["text"] = text[:-1]
                wire["press_enter"] = True
            return [_flat("type", wire)]
        if name == "key":
            keys = args.get("keys", [])
            if len(keys) == 1:
                return [_flat("press_key", {"key": keys[0]})]
            return [_flat("hotkey", {"keys": list(keys)})]
        if name in ("key_down", "key_up"):
            # Canonical takes a list, the wire is singular: fan out.
            return [_flat(name, {"key": key}) for key in args.get("keys", [])]
        if name == "scroll":
            x, y = _require_clamped(args, "coordinate", name="scroll")
            return [
                _flat(
                    "scroll",
                    {
                        "x": x,
                        "y": y,
                        "direction": args["direction"],
                        # Same 0-999 wire range as the coordinates, so a large
                        # canonical ``amount`` clamps rather than overflowing it.
                        "magnitude_in_pixels": min(
                            args["amount"] * PIXELS_PER_CLICK, _GEMINI_MAX_COORD
                        ),
                    },
                )
            ]
        if name == "wait":
            return [_flat("wait", {"seconds": args.get("duration", 1.0)})]
        if name == "screenshot":
            return [_flat("take_screenshot", {})]

        # Named branches for the canonical actions the wire cannot spell. These
        # come BEFORE the extra-tool pass-through so the error names the missing
        # verb, instead of leaking through and failing on re-parse.
        if name == "hold_key":
            raise ValueError(
                "Gemini cannot express hold_key: the desktop wire has "
                "key_down/key_up but no timed hold verb"
            )
        if name in ("mouse_down", "mouse_up"):
            raise ValueError(
                f"Gemini cannot express {name}(button=...): the wire's {name} "
                "requires x/y and carries no button"
            )
        if name == "cursor_position":
            raise ValueError("Gemini cannot express cursor_position: the wire has no query verb")

        # Env extra tools pass through by name.
        return [_flat(name, copy.deepcopy(args))]


# =============================================================================
# Gemini Mobile Action Space
# =============================================================================

ENVIRONMENT_MOBILE = "ENVIRONMENT_MOBILE"

#: The 10 verbs ENVIRONMENT_MOBILE offers, enumerated against the live API.
#:
#: Semantics come from Google's own ADB bridge (the Android quickstart), which
#: is the definition of what each verb does on a device:
#:   * ``go_back``      -> ``keyevent 4``  = the Android BACK key, not history-back
#:   * ``press_key``    -> a 5-entry keymap that is exactly canonical system_button
#:   * ``drag_and_drop``-> ``input swipe ... 300`` = a 300ms SWIPE, not a long-press drag
_MOBILE_VERB_TO_LITE_ACTIONS: dict[str, tuple[str, ...]] = {
    "click": ("tap",),
    "long_press": ("long_press",),
    "drag_and_drop": ("swipe",),
    "go_back": ("system_button",),
    "press_key": ("system_button",),
    "type": ("type", "system_button"),  # press_enter expands (see below)
    "wait": ("wait",),
    "take_screenshot": ("screenshot",),
    # No canonical home; always excluded. ``open_app`` is supplied by the env
    # as an extra tool whose enum carries the real app catalog.
    "open_app": (),
    "list_apps": (),
}

#: Always excluded, for different reasons:
#:   * ``list_apps``: no canonical action, no env handler anywhere -- keeping it
#:     guarantees a wasted turn.
#:   * ``open_app``: the builtin takes an Android PACKAGE name (``monkey -p``)
#:     while the env's enum carries DISPLAY names ("Chrome", "WeChat"). Same
#:     name, incompatible domain -- sending both would put two conflicting
#:     ``open_app`` declarations in one request.
_MOBILE_ALWAYS_EXCLUDED = frozenset({"open_app", "list_apps"})

#: Android keyevent name -> canonical system button.
#:
#: This is a durable mobile input vocabulary fact, not Gemini-only policy.
#: Keep every entry pinned here until the Android keyevent mapping has a shared
#: owner.
_GEMINI_KEY_TO_SYSTEM_BUTTON = {
    "back": "Back",
    "home": "Home",
    "menu": "Menu",
    "enter": "Enter",
    "dpad_center": "Enter",
    "app_switch": "Recent",
    "recent": "Recent",
    "recents": "Recent",
}


@dataclasses.dataclass
class GeminiMobileActionSpace(BaseActionSpace, key="gemini@mobile"):
    """Projection between Gemini's native mobile verbs and canonical Lite.

    Nearly 1:1 -- this is the payoff of reaching the native mobile action space
    instead of steering a desktop one with custom function tools.
    """

    platform: str = "mobile"

    @classmethod
    def get_tool_schemas(cls, include: list[str] | None = None) -> list[dict[str, Any]]:
        """Mobile uses the native ``computerUse`` tool, not function schemas."""
        return []

    @staticmethod
    def environment_for_platform(*, platform: str) -> str:
        return ENVIRONMENT_MOBILE

    @classmethod
    def predefined_verbs(cls, *, platform: str) -> frozenset[str]:
        return frozenset(_MOBILE_VERB_TO_LITE_ACTIONS)

    @classmethod
    def excluded_predefined_functions(
        cls, *, platform: str, valid_actions: list[str] | None
    ) -> frozenset[str]:
        excluded = set(_MOBILE_ALWAYS_EXCLUDED)
        if valid_actions is not None:
            allowed = set(valid_actions)
            excluded |= {
                verb
                for verb, lite_actions in _MOBILE_VERB_TO_LITE_ACTIONS.items()
                if lite_actions and not set(lite_actions) <= allowed
            }
        return frozenset(excluded) & cls.predefined_verbs(platform=platform)

    @classmethod
    def visible_predefined_verbs(
        cls, *, platform: str, valid_actions: list[str] | None
    ) -> frozenset[str]:
        return cls.predefined_verbs(platform=platform) - cls.excluded_predefined_functions(
            platform=platform, valid_actions=valid_actions
        )

    # -- parse -----------------------------------------------------------------

    def convert_tool_calls_from_agent(
        self,
        agent_tool_calls: list[Any],
        *,
        resolution: tuple[int, int] | None = None,
        active_extra_tool_names: set[str] | None = None,
        active_extra_tool_schemas: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for call in agent_tool_calls:
            result.extend(self._convert_single_from_agent(call))
        return merge_adjacent_lite_action_batches(result)

    def _convert_single_from_agent(self, call: dict[str, Any]) -> list[dict[str, Any]]:
        verb = call.get("name", "")
        args = call.get("arguments") or {}

        if verb == "click":
            return [LiteMobileActionSpace.tap(coordinate=_require_xy(args, verb=verb))]
        if verb == "long_press":
            seconds = args.get("seconds")
            coordinate = _require_xy(args, verb=verb)
            if seconds is None:
                return [LiteMobileActionSpace.long_press(coordinate=coordinate)]
            return [
                LiteMobileActionSpace.long_press(
                    coordinate=coordinate,
                    duration=float(_require_number(args, "seconds", verb=verb)),
                )
            ]
        if verb == "drag_and_drop":
            # 300ms input swipe -> canonical swipe. NOT canonical drag, whose
            # docstring is "long-press at a start point and move to an end point".
            return [
                LiteMobileActionSpace.swipe(
                    start_coordinate=_require_xy(args, verb=verb, x="start_x", y="start_y"),
                    coordinate=_require_xy(args, verb=verb, x="end_x", y="end_y"),
                )
            ]
        if verb == "go_back":
            return [LiteMobileActionSpace.system_button(button="Back")]
        if verb == "press_key":
            key = str(args.get("key", "")).lower()
            button = _GEMINI_KEY_TO_SYSTEM_BUTTON.get(key)
            if button is None:
                raise ModelToolCallParseError(
                    f"unknown Gemini mobile key {args.get('key')!r}: canonical mobile "
                    "has no key action, only system_button"
                )
            return [LiteMobileActionSpace.system_button(button=button)]
        if verb == "type":
            # Canonical mobile ``type`` has no Enter spelling at all -- unlike
            # desktop, where a trailing newline is the Return keypress, mobile
            # envs read neither. Expanding is the only faithful mapping.
            out = [LiteMobileActionSpace.type(text=required_model_text(args, action=verb))]
            if args.get("press_enter"):
                out.append(LiteMobileActionSpace.system_button(button="Enter"))
            return out
        if verb == "wait":
            seconds = args.get("seconds")
            duration = (
                1.0 if seconds is None else float(_require_number(args, "seconds", verb=verb))
            )
            return [LiteMobileActionSpace.wait(duration=duration)]
        if verb == "take_screenshot":
            return [LiteMobileActionSpace.screenshot()]

        raise ModelToolCallParseError(f"unknown Gemini mobile action: {verb or '<unknown>'}")

    # -- render ----------------------------------------------------------------

    def convert_tool_calls_to_agent(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        resolution: tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        result: list[Any] = []
        for call in tool_calls:
            result.extend(self._convert_single_to_agent(call))
        return result

    def _convert_single_to_agent(self, tool_call: dict[str, Any]) -> list[Any]:
        name = tool_call_name(tool_call)
        args = tool_call_arguments(tool_call)

        if name == "mobile":
            out: list[Any] = []
            for child in args.get("actions", []):
                child_args = {k: v for k, v in child.items() if k != "action"}
                out.extend(
                    self._convert_single_to_agent(make_tool_call(child["action"], child_args))
                )
            return out

        if name == "tap":
            clicks = args.get("clicks", 1)
            if clicks != 1:
                raise ValueError(
                    f"Gemini mobile cannot express tap(clicks={clicks}): the wire has "
                    "only 'click' and carries no repeat count"
                )
            x, y = _clamp(args["coordinate"])
            return [_flat("click", {"x": x, "y": y})]
        if name == "long_press":
            x, y = _clamp(args["coordinate"])
            wire: dict[str, Any] = {"x": x, "y": y}
            if args.get("duration") is not None:
                wire["seconds"] = args["duration"]
            return [_flat("long_press", wire)]
        if name == "swipe":
            sx, sy = _clamp(args["start_coordinate"])
            ex, ey = _clamp(args["coordinate"])
            return [
                _flat(
                    "drag_and_drop",
                    {"start_x": sx, "start_y": sy, "end_x": ex, "end_y": ey},
                )
            ]
        if name == "type":
            return [_flat("type", {"text": args.get("text", "")})]
        if name == "system_button":
            button = args.get("button")
            if button == "Back":
                return [_flat("go_back", {})]
            wire_key = {"Home": "home", "Enter": "enter", "Menu": "menu", "Recent": "app_switch"}
            if button not in wire_key:
                raise ValueError(f"Gemini mobile cannot express system_button({button!r})")
            return [_flat("press_key", {"key": wire_key[button]})]
        if name == "wait":
            return [_flat("wait", {"seconds": args.get("duration", 1.0)})]
        if name == "screenshot":
            return [_flat("take_screenshot", {})]

        if name == "drag":
            raise ValueError(
                "Gemini mobile cannot express drag: its drag_and_drop is a 300ms "
                "swipe, not a long-press drag"
            )
        if name == "pinch":
            raise ValueError("Gemini mobile cannot express pinch: no pinch verb")

        return [_flat(name, copy.deepcopy(args))]


__all__ = [
    "ENVIRONMENT_BROWSER",
    "ENVIRONMENT_DESKTOP",
    "ENVIRONMENT_MOBILE",
    "GeminiDesktopActionSpace",
    "GeminiMobileActionSpace",
]
