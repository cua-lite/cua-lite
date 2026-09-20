"""Canonical key-name normalization for Lite action surfaces."""

from __future__ import annotations

KEY_NAME_ALIASES: dict[str, str] = {
    "ctrl": "ctrl", "control": "ctrl", "ctl": "ctrl",
    "ctrlleft": "ctrl", "ctrlright": "ctrl",
    "controlleft": "ctrl", "controlright": "ctrl",
    "alt": "alt", "option": "alt", "opt": "alt",
    "altleft": "alt", "altright": "alt", "optionleft": "alt", "optionright": "alt",
    "shift": "shift", "shiftleft": "shift", "shiftright": "shift",
    "meta": "meta", "cmd": "meta", "command": "meta",
    "super": "meta", "win": "meta", "windows": "meta",
    "cmdleft": "meta", "cmdright": "meta",
    "commandleft": "meta", "commandright": "meta",
    "metaleft": "meta", "metaright": "meta",
    "superleft": "meta", "superright": "meta",
    "winleft": "meta", "winright": "meta",
    "windowsleft": "meta", "windowsright": "meta",
    "osleft": "meta", "osright": "meta",
    "enter": "enter", "return": "enter",
    "tab": "tab",
    "space": "space", "spacebar": "space",
    "backspace": "backspace", "bksp": "backspace",
    "delete": "delete", "del": "delete",
    "esc": "esc", "escape": "esc",
    "up": "up", "arrowup": "up", "arrow_up": "up",
    "down": "down", "arrowdown": "down", "arrow_down": "down",
    "left": "left", "arrowleft": "left", "arrow_left": "left",
    "right": "right", "arrowright": "right", "arrow_right": "right",
    "home": "home", "end": "end",
    "pageup": "pageup", "page_up": "pageup", "pgup": "pageup",
    "pagedown": "pagedown", "page_down": "pagedown", "pgdn": "pagedown",
    "insert": "insert", "ins": "insert",
    "capslock": "capslock", "caps_lock": "capslock",
    "menu": "menu", "contextmenu": "menu",
    "printscreen": "printscreen", "print_screen": "printscreen",
    "prtsc": "printscreen", "prtscr": "printscreen", "prntscrn": "printscreen",
    "clear": "clear",
    "kp_enter": "kp_enter", "numpad_enter": "kp_enter",
    "comma": ",",
    "period": ".", "dot": ".",
    "slash": "/", "forwardslash": "/", "forward_slash": "/",
    "backslash": "\\", "back_slash": "\\",
    "semicolon": ";",
    "apostrophe": "'", "singlequote": "'", "single_quote": "'",
    "grave": "`", "backtick": "`",
    "bracketleft": "[", "leftbracket": "[", "left_bracket": "[",
    "bracketright": "]", "rightbracket": "]", "right_bracket": "]",
    "minus": "-", "plus": "+", "equal": "=",
    "+": "+",
    "volumeup": "volumeup", "volume_up": "volumeup",
    "audiovolumeup": "volumeup", "audio_volume_up": "volumeup",
    "mediavolumeup": "volumeup", "media_volume_up": "volumeup",
    "xf86audioraisevolume": "volumeup",
    "volumedown": "volumedown", "volume_down": "volumedown",
    "audiovolumedown": "volumedown", "audio_volume_down": "volumedown",
    "mediavolumedown": "volumedown", "media_volume_down": "volumedown",
    "xf86audiolowervolume": "volumedown",
    "volumemute": "volumemute", "volume_mute": "volumemute",
    "audiomute": "volumemute", "audio_mute": "volumemute",
    "audiovolumemute": "volumemute", "audio_volume_mute": "volumemute",
    "mediavolumemute": "volumemute", "media_volume_mute": "volumemute",
    "xf86audiomute": "volumemute",
    "playpause": "playpause", "mediaplaypause": "playpause",
    "xf86audioplay": "playpause", "media_play_pause": "playpause",
    "nexttrack": "nexttrack", "xf86audionext": "nexttrack",
    "mediatracknext": "nexttrack", "media_track_next": "nexttrack",
    "prevtrack": "prevtrack", "xf86audioprev": "prevtrack",
    "mediatrackprevious": "prevtrack", "media_track_previous": "prevtrack",
}

CANONICAL_SPECIAL_KEY_NAMES: frozenset[str] = (
    frozenset(value for value in KEY_NAME_ALIASES.values() if len(value) != 1)
    | frozenset(f"f{i}" for i in range(1, 25))
)
_CONTROL_CHARS = frozenset(chr(i) for i in range(32)) | frozenset({chr(127)})
_VISIBLE_ASCII_GLYPHS = frozenset(chr(i) for i in range(33, 127))


def canonical_special_keys() -> frozenset[str]:
    """Return the canonical named key vocabulary excluding literal characters."""
    return CANONICAL_SPECIAL_KEY_NAMES


def is_canonical_key_token(token: object) -> bool:
    """Return whether *token* is already in Lite's stored key vocabulary."""
    if not isinstance(token, str) or not token:
        return False
    if token != token.lower() or ("+" in token and token != "+"):
        return False
    if len(token) == 1:
        return token in _VISIBLE_ASCII_GLYPHS
    return token in CANONICAL_SPECIAL_KEY_NAMES


def _clean_raw_token(tok: str) -> str:
    if any(ch in _CONTROL_CHARS for ch in tok):
        raise ValueError(f"keys: invalid control character in {tok!r}")
    clean = tok.strip()
    if not clean:
        raise ValueError("keys: empty or whitespace-only token is not allowed")
    return clean


def _normalize_token(tok: str) -> str:
    low = _clean_raw_token(tok).lower()
    if low in KEY_NAME_ALIASES:
        return KEY_NAME_ALIASES[low]
    if low.startswith("f") and low[1:].isdigit() and 1 <= int(low[1:]) <= 24:
        return low
    if len(low) == 1 and low in _VISIBLE_ASCII_GLYPHS:
        return low
    raise ValueError(f"keys: unknown key token {tok!r}")


def _split_plus_chord(value: str) -> list[str]:
    """Split a ``+``-joined chord while preserving literal plus glyphs."""
    clean = _clean_raw_token(value)
    if clean == "+":
        return ["+"]
    if clean.startswith("+"):
        raise ValueError(f"keys: invalid leading '+' chord syntax in {value!r}")
    if clean.endswith("+"):
        if not clean.endswith("++") or clean.endswith("+++"):
            raise ValueError(f"keys: invalid trailing '+' chord syntax in {value!r}")
        head = clean[:-2]
        parts = [part.strip() for part in head.split("+")]
        if not parts or any(not part for part in parts):
            raise ValueError(f"keys: invalid '+' chord syntax in {value!r}")
        return parts + ["+"]
    parts = [part.strip() for part in clean.split("+")]
    if any(not part for part in parts):
        raise ValueError(f"keys: invalid '+' chord syntax in {value!r}")
    return parts


def normalize_keys(keys: list[str] | str) -> list[str]:
    """Any model spelling to canonical Lite key tokens.

    Canonical Lite key tokens are lowercase named keys plus literal printable
    glyphs.
    """
    if isinstance(keys, str):
        return [_normalize_token(token) for token in _split_plus_chord(keys)]
    if not isinstance(keys, list):
        raise TypeError(f"keys: expected list[str] or str, got {type(keys).__name__}")
    raw: list[str] = []
    for k in keys:
        if not isinstance(k, str):
            raise TypeError(f"keys: expected str tokens, got {type(k).__name__}")
        raw.append(k)
    if not raw:
        raise ValueError("keys must not be empty")
    return [_normalize_token(token) for token in raw]


__all__ = ["canonical_special_keys", "is_canonical_key_token", "normalize_keys"]
