"""Per-backend keyboard-key projection.

The env-side, EXECUTION-path half of key handling: it takes canonical Lite key
tokens produced by :mod:`lite.core.tools.action_space.keys` (``normalize_keys``,
applied once by the shared key vocabulary owner) and projects
them to a specific backend's spelling (Playwright / xdotool / pyautogui / pynput
/ Selenium). Canonical Lite key tokens are lowercase named keys plus literal
printable glyphs. Each ``to_*`` RAISES (loud) on a token it cannot map — never a
silent no-op (the bug class). The backend names themselves are the one
``Backend`` vocabulary owned by this projector.

The canonical Lite key-token vocabulary half lives in
:mod:`lite.core.tools.action_space.keys` (light, shared) so the model's
action-space build path carries no env dependency; the backend projection lives
here because it is an env-execution concern. Import canonical normalization from
core directly; this module intentionally exposes projection only.

Run / smoke-test:
    uv run python -c "from lite.gym.utils.backend.keys import translate_keys as t; \
        print(t(['ctrl','c'],'xdotool'), t(['ctrl','a'],'playwright'))"
"""
from __future__ import annotations

from typing import Literal

from lite.core.tools.action_space.keys import is_canonical_key_token

Backend = Literal["playwright", "xdotool", "pyautogui", "pynput", "selenium"]


# ---------------------------------------------------------------------------
# per-backend projection tables (Lite canonical → backend spelling).
# ---------------------------------------------------------------------------

# --- Playwright (browsergym, webgym, captcha) ---
_PLAYWRIGHT: dict[str, str] = {
    "ctrl": "Control", "alt": "Alt", "shift": "Shift", "meta": "Meta",
    "enter": "Enter", "tab": "Tab", "space": "Space",
    "backspace": "Backspace", "delete": "Delete", "esc": "Escape",
    "up": "ArrowUp", "down": "ArrowDown", "left": "ArrowLeft", "right": "ArrowRight",
    "home": "Home", "end": "End", "pageup": "PageUp", "pagedown": "PageDown",
    "insert": "Insert", "capslock": "CapsLock",
    # media / volume keys (Playwright key names).
    "volumeup": "AudioVolumeUp", "volumedown": "AudioVolumeDown",
    "volumemute": "AudioVolumeMute", "playpause": "MediaPlayPause",
    "nexttrack": "MediaTrackNext", "prevtrack": "MediaTrackPrevious",
    # Playwright resolves canonical single-character punctuation directly;
    # there is NO "Divide"/"Backslash" *named* key for "/" / "\").
    **{f"f{i}": f"F{i}" for i in range(1, 25)},
}

# --- xdotool (lite.osworld / sandbox-family) — copied verbatim from the existing
# exec_stdio/server.py _KEYSYM + _PUNCT so xdotool projection is byte-equal.
_XDOTOOL: dict[str, str] = {
    "ctrl": "ctrl", "alt": "alt", "shift": "shift", "meta": "super",
    "enter": "Return", "tab": "Tab", "space": "space",
    "backspace": "BackSpace", "delete": "Delete", "esc": "Escape",
    "up": "Up", "down": "Down", "left": "Left", "right": "Right",
    "home": "Home", "end": "End", "pageup": "Prior", "pagedown": "Next",
    "insert": "Insert", "capslock": "Caps_Lock",
    "menu": "Menu", "printscreen": "Print", "clear": "Clear", "kp_enter": "KP_Enter",
    # media / volume keys (X keysyms).
    "volumeup": "XF86AudioRaiseVolume", "volumedown": "XF86AudioLowerVolume",
    "volumemute": "XF86AudioMute", "playpause": "XF86AudioPlay",
    "nexttrack": "XF86AudioNext", "prevtrack": "XF86AudioPrev",
    **{f"f{i}": f"F{i}" for i in range(1, 25)},
}
_XDOTOOL_PUNCT: dict[str, str] = {
    ",": "comma", ".": "period", "/": "slash", "\\": "backslash",
    ";": "semicolon", "'": "apostrophe", "`": "grave",
    "[": "bracketleft", "]": "bracketright", "-": "minus", "+": "plus", "=": "equal",
}

# --- pyautogui (osworld umbrella) ---
_PYAUTOGUI: dict[str, str] = {
    "ctrl": "ctrl", "alt": "alt", "shift": "shift",
    "meta": "win",  # pyautogui KEYBOARD_KEYS has no "meta"
    "enter": "enter", "tab": "tab", "space": "space",
    "backspace": "backspace", "delete": "delete", "esc": "esc",
    "up": "up", "down": "down", "left": "left", "right": "right",
    "home": "home", "end": "end", "pageup": "pageup", "pagedown": "pagedown",
    "insert": "insert", "capslock": "capslock",
    "printscreen": "printscreen", "clear": "clear",
    # media / volume keys (pyautogui KEY_NAMES).
    "volumeup": "volumeup", "volumedown": "volumedown", "volumemute": "volumemute",
    "playpause": "playpause", "nexttrack": "nexttrack", "prevtrack": "prevtrack",
    **{f"f{i}": f"f{i}" for i in range(1, 25)},
}
# pyautogui.KEYBOARD_KEYS is display-coupled (no DISPLAY headless); mirror the
# package's KEY_NAMES set (pyautogui/__init__.py, v0.9.54) for projection-output
# validation only. This is not Lite's canonical allowlist; _require_canonical()
# rejects whitespace/control tokens before this backend capability mirror runs.
_PYAUTOGUI_VALID: frozenset[str] = frozenset(
    list("\t\n\r ")
    + [chr(c) for c in range(0x21, 0x7F)]  # all printable ASCII glyphs ! .. ~
    + ["accept", "add", "alt", "altleft", "altright", "apps", "backspace",
       "browserback", "browserfavorites", "browserforward", "browserhome",
       "browserrefresh", "browsersearch", "browserstop", "capslock", "clear",
       "convert", "ctrl", "ctrlleft", "ctrlright", "decimal", "del", "delete",
       "divide", "down", "end", "enter", "esc", "escape", "execute", "final",
       "fn", "hanguel", "hangul", "hanja", "help", "home", "insert", "junja",
       "kana", "kanji", "launchapp1", "launchapp2", "launchmail",
       "launchmediaselect", "left", "modechange", "multiply", "nexttrack",
       "nonconvert", "numlock", "pagedown", "pageup", "pause", "pgdn", "pgup",
       "playpause", "prevtrack", "print", "printscreen", "prntscrn", "prtsc",
       "prtscr", "return", "right", "scrolllock", "select", "separator",
       "shift", "shiftleft", "shiftright", "sleep", "space", "stop", "subtract",
       "tab", "up", "volumedown", "volumemute", "volumeup", "win", "winleft",
       "winright", "yen", "command", "option", "optionleft", "optionright"]
    + [f"f{i}" for i in range(1, 25)]
    + [f"num{i}" for i in range(10)]
)


def _require_canonical(keys: list[str]) -> None:
    """The env projections expect canonical Lite key tokens.

    That means lowercase named keys plus literal printable glyphs, normalized
    once by the shared key vocabulary owner. This cheap defensive
    assert fails loud if an un-normalized key leaks through a direct projector
    call.

    It guards DIRECT ``to_*`` callers only. The single production entry,
    ``model_inputs.project_model_keys``, pre-enforces the same core predicate
    (see the CONTRACT in its docstring), so this raise CANNOT reach env feedback
    and the feedback layer must not carry a rule for its wording."""
    for k in keys:
        if not is_canonical_key_token(k):
            raise ValueError(
                f"keys: expected Lite canonical key tokens (normalize at the "
                f"action factory, not the env); got {k!r}")


def _project(
    keys: list[str],
    table: dict[str, str],
    *,
    backend: str,
    what: str,
) -> list[str]:
    """The shared table-lookup projection: Lite canonical token → ``table``
    spelling, a printable non-whitespace glyph passed through as itself, and
    anything else RAISES.

    Used by the three backends whose rule is exactly that (playwright / pynput /
    selenium). ``to_xdotool`` (extra punctuation table + an ``isalnum`` branch)
    and ``to_pyautogui`` (validity-SET membership, not table membership) have
    genuinely different rules and stay hand-written.

    ``backend`` and ``what`` reproduce each backend's own raise wording
    BYTE-FOR-BYTE, because that wording is a live contract:
    :func:`lite.gym.utils.feedback.errors.model_visible_error_detail`
    string-matches ``"no <what> for"`` to turn the raise into model-safe
    guidance, and an unrecognised wording leaks the backend name into the
    prompt. Change a wording here only together with that translator (and its
    generative guard in ``tests/gym/utils/feedback/test_feedback_errors.py``)."""
    _require_canonical(keys)
    out: list[str] = []
    for k in keys:
        if k in table:
            out.append(table[k])
        elif len(k) == 1:
            out.append(k)  # backend resolves a single printable char ("a","/","#")
        else:
            raise ValueError(f"keys.to_{backend}: no {what} for {k!r}")
    return out


def to_playwright(keys: list[str]) -> list[str]:
    """Project canonical Lite key tokens to Playwright wire key names."""
    return _project(keys, _PLAYWRIGHT, backend="playwright", what="Playwright key")


def to_xdotool(keys: list[str]) -> list[str]:
    """Project canonical Lite key tokens to xdotool X keysyms."""
    _require_canonical(keys)
    out: list[str] = []
    for k in keys:
        if k in _XDOTOOL:
            out.append(_XDOTOOL[k])
        elif len(k) == 1:
            if k.isalnum():
                out.append(k)  # letter/digit is a valid keysym name as-is
            elif k in _XDOTOOL_PUNCT:
                out.append(_XDOTOOL_PUNCT[k])
            else:
                raise ValueError(f"keys.to_xdotool: no X keysym for {k!r}")
        else:
            raise ValueError(f"keys.to_xdotool: no X keysym for {k!r}")
    return out


def to_pyautogui(keys: list[str]) -> list[str]:
    """Project canonical Lite key tokens to pyautogui wire key names."""
    _require_canonical(keys)
    out: list[str] = []
    for k in keys:
        tok = _PYAUTOGUI.get(k, k)  # specials map; single chars pass through
        if tok not in _PYAUTOGUI_VALID:
            raise ValueError(
                f"keys.to_pyautogui: {k!r}->{tok!r} not in pyautogui.KEYBOARD_KEYS")
        out.append(tok)
    return out


# --- pynput (cua computer-server keyboard: trycua/cua-xfce native + cua.sandbox
# both terminate at `getattr(pynput.keyboard.Key, tok)`, with a len==1 token
# passed through as a literal char). Spellings differ from pyautogui: UNDERSCORES
# (page_up / page_down / caps_lock / print_screen) and `cmd` (not win/meta);
# pynput's Key enum has f1..f20 only. ---
_PYNPUT: dict[str, str] = {
    "ctrl": "ctrl", "alt": "alt", "shift": "shift", "meta": "cmd",
    "enter": "enter", "tab": "tab", "space": "space",
    "backspace": "backspace", "delete": "delete", "esc": "esc",
    "up": "up", "down": "down", "left": "left", "right": "right",
    "home": "home", "end": "end", "pageup": "page_up", "pagedown": "page_down",
    "insert": "insert", "capslock": "caps_lock",
    "menu": "menu", "printscreen": "print_screen",
    # media / volume keys (pynput Key attribute names).
    "volumeup": "media_volume_up", "volumedown": "media_volume_down",
    "volumemute": "media_volume_mute", "playpause": "media_play_pause",
    "nexttrack": "media_next", "prevtrack": "media_previous",
    # Canonical single-character punctuation passes through to pynput.
    **{f"f{i}": f"f{i}" for i in range(1, 21)},   # pynput Key has f1..f20 only
}


def to_pynput(keys: list[str]) -> list[str]:
    """Project canonical Lite key tokens to pynput ``Key`` attribute names (the cua
    computer-server keyboard vocabulary; a single printable char passes through
    as a literal char to pynput). Asserts canonical input, raises on an
    unmappable token (never silent)."""
    return _project(keys, _PYNPUT, backend="pynput", what="pynput key")


# --- Selenium / W3C WebDriver (webharbor.webvoyager: the container drives
# `ActionChains.key_down/key_up/send_keys`, and `_selenium_key` passes any token
# it does not recognize straight through — so an unprojected token is TYPED AS
# LITERAL TEXT, the worst failure mode of the four backends). The values are the
# W3C WebDriver "normalised key value" code points (U+E000..U+E03D) that
# `selenium.webdriver.common.keys.Keys` exposes by name; they are hardcoded
# because selenium is a CONTAINER dependency and is not installed host-side.
# WebDriver models no capslock, menu, printscreen, numpad-enter or media keys,
# and only F1..F12 — those RAISE rather than silently type their own name. ---
_SELENIUM: dict[str, str] = {
    "ctrl": "\ue009", "alt": "\ue00a", "shift": "\ue008", "meta": "\ue03d",
    "enter": "\ue007", "tab": "\ue004", "space": "\ue00d",
    "backspace": "\ue003", "delete": "\ue017", "esc": "\ue00c",
    "up": "\ue013", "down": "\ue015", "left": "\ue012", "right": "\ue014",
    "home": "\ue011", "end": "\ue010", "pageup": "\ue00e", "pagedown": "\ue00f",
    "insert": "\ue016", "clear": "\ue005",
    # Canonical single-character punctuation passes through; send_keys types it.
    **{f"f{i}": chr(0xE030 + i) for i in range(1, 13)},  # F1=U+E031 .. F12=U+E03C
}


def to_selenium(keys: list[str]) -> list[str]:
    """Project canonical Lite key tokens to Selenium/W3C WebDriver key code points
    (the vocabulary ``ActionChains`` speaks; a single printable char passes
    through as itself — ``send_keys`` types it directly). Asserts canonical
    input, raises on an unmappable token — never silent, and never a token that
    would be typed as literal text."""
    return _project(keys, _SELENIUM, backend="selenium", what="WebDriver key")


_PROJECTORS = {
    "playwright": to_playwright,
    "xdotool": to_xdotool,
    "pyautogui": to_pyautogui,
    "pynput": to_pynput,
    "selenium": to_selenium,
}


def translate_keys(keys: list[str], backend: Backend) -> list[str]:
    """Project ALREADY-canonical keys to *backend*; raises on an unmappable token.
    (Call ``normalize_keys`` first if the input is not yet canonical.)"""
    return _PROJECTORS[backend](keys)
