"""Cross-backend key projection coverage.

The guard that was missing when the browsergym scroll/key bugs shipped: feed the
full canonical vocabulary through every backend projection and assert each yields
a valid backend token, and that unmappable tokens RAISE (loud, never silent).

    uv run pytest tests/gym/utils/backend/test_keys.py
"""
from __future__ import annotations

from typing import get_args

import pytest

from lite.core.tools.action_space.keys import canonical_special_keys, normalize_keys
from lite.gym.utils.backend import keys as K
from lite.gym.utils.backend.keys import _PYAUTOGUI_VALID

# Backend valid-token sets (ground truth).
_LETTERS_DIGITS = list("abcdefghijklmnopqrstuvwxyz0123456789")
_PUNCT_GLYPHS = list(",./\\;'`[]-+=")

_PW_VALID = set(K._PLAYWRIGHT.values()) | set(_LETTERS_DIGITS) | set(_PUNCT_GLYPHS)
_XDO_VALID = set(K._XDOTOOL.values()) | set(_LETTERS_DIGITS) | set(K._XDOTOOL_PUNCT.values())
_PYNPUT_VALID = set(K._PYNPUT.values()) | set(_LETTERS_DIGITS) | set(_PUNCT_GLYPHS)
# pynput's Key enum has f1..f20 only (no f21-f24) and no clear/kp_enter → map-or-RAISE.
_PYNPUT_UNSUPPORTED = {f"f{i}" for i in range(21, 25)}

# Desktop-only specials: present in the xdotool vocabulary but with no universal
# browser/pyautogui equivalent (verified: Playwright's layout has no PrintScreen/
# ContextMenu/Clear/NumpadEnter; pyautogui has no menu/kp_enter). Those backends
# must map-or-RAISE-loud (never silent); they never appear in browser rollouts.
_DESKTOP_ONLY = {"menu", "printscreen", "clear", "kp_enter"}

# Vocabulary valid across ALL backends (common specials + literal chars).
_COMMON = (
    sorted(canonical_special_keys() - _DESKTOP_ONLY)
    + _LETTERS_DIGITS
    + _PUNCT_GLYPHS
)


def test_gym_key_helper_does_not_reexport_private_core_vocabulary() -> None:
    assert not hasattr(K, "CANONICAL_SPECIAL_KEY_NAMES")
    assert not hasattr(K, "normalize_keys")


def test_backend_vocabulary_is_declared_exactly_once() -> None:
    # ONE Backend vocabulary, owned by the backend projector. The model-input
    # validator imports it; core owns only Lite key-token normalization.
    import lite.core.tools.action_space.keys as core_keys
    from lite.gym.utils.backend import model_inputs

    assert not hasattr(core_keys, "Backend")
    assert model_inputs.Backend is K.Backend
    assert not hasattr(K, "_Backend")
    assert not hasattr(model_inputs, "KeyBackend")
    # Every modelled backend has a projector — a name in the vocabulary with no
    # entry in _PROJECTORS is a KeyError waiting at runtime.
    assert set(get_args(K.Backend)) == set(K._PROJECTORS)


@pytest.mark.parametrize("k", _COMMON)
def test_playwright_projection_valid(k):
    for tok in K.to_playwright([k]):
        assert tok in _PW_VALID, f"{k!r}->{tok!r} not a valid Playwright key"


@pytest.mark.parametrize("k", _COMMON + sorted(_DESKTOP_ONLY))
def test_xdotool_projection_valid(k):  # xdotool maps the full vocabulary
    for tok in K.to_xdotool([k]):
        assert tok in _XDO_VALID, f"{k!r}->{tok!r} not a valid X keysym"


@pytest.mark.parametrize("k", _COMMON)
def test_pyautogui_projection_valid(k):
    for tok in K.to_pyautogui([k]):
        assert tok in _PYAUTOGUI_VALID, f"{k!r}->{tok!r} not in KEYBOARD_KEYS"


@pytest.mark.parametrize("k", [c for c in _COMMON if c not in _PYNPUT_UNSUPPORTED])
def test_pynput_projection_valid(k):  # cua computer-server keyboard vocabulary
    for tok in K.to_pynput([k]):
        assert tok in _PYNPUT_VALID, f"{k!r}->{tok!r} not a valid pynput key"


@pytest.mark.parametrize("k", sorted(_PYNPUT_UNSUPPORTED))
def test_pynput_unsupported_raises_never_silent(k):  # f21-f24 not in pynput.Key
    with pytest.raises(ValueError):
        K.to_pynput([k])


# --- Selenium / W3C WebDriver (webvoyager's container) ---
# Pinned against the W3C WebDriver "normalised key value" table, which is what
# `selenium.webdriver.common.keys.Keys` exposes by name. Selenium is a CONTAINER
# dependency (not installed host-side), so the spec code points are the ground
# truth here rather than an import of the package.
_WEBDRIVER_SPEC = {
    "backspace": "\ue003", "tab": "\ue004", "clear": "\ue005", "enter": "\ue007",
    "shift": "\ue008", "ctrl": "\ue009", "alt": "\ue00a", "esc": "\ue00c",
    "space": "\ue00d", "pageup": "\ue00e", "pagedown": "\ue00f", "end": "\ue010",
    "home": "\ue011", "left": "\ue012", "up": "\ue013", "right": "\ue014",
    "down": "\ue015", "insert": "\ue016", "delete": "\ue017", "meta": "\ue03d",
    "f1": "\ue031", "f5": "\ue035", "f12": "\ue03c",
}


@pytest.mark.parametrize("canon,code", sorted(_WEBDRIVER_SPEC.items()))
def test_selenium_projection_matches_webdriver_spec(canon, code):
    assert K.to_selenium([canon]) == [code]


# WebDriver models none of these, and `_selenium_key` passes an unknown token
# through — so an unprojected token would be TYPED AS LITERAL TEXT. They must
# RAISE instead (the worst silent-failure mode of the four backends).
@pytest.mark.parametrize("k", [
    "capslock", "menu", "printscreen", "kp_enter",
    "volumeup", "volumedown", "volumemute", "playpause", "nexttrack", "prevtrack",
    "f13", "f24",
])
def test_selenium_unmodelled_key_raises_never_typed_as_text(k):
    with pytest.raises(ValueError):
        K.to_selenium([k])


@pytest.mark.parametrize("k", sorted(canonical_special_keys()))
def test_selenium_maps_or_raises_never_silently_invalid(k):
    # Contract: every canonical special either projects to a WebDriver code point
    # or raises — it may never come back as a bare word Selenium would type.
    try:
        toks = K.to_selenium([k])
    except ValueError:
        return
    for tok in toks:
        assert len(tok) == 1, f"{k!r}->{tok!r} is not a single key value"
        assert "\ue000" <= tok <= "\ue05d" or tok.isprintable()


def test_selenium_literal_chars_pass_through():
    assert K.to_selenium(["ctrl", "a"]) == ["\ue009", "a"]
    assert K.to_selenium(["-", "+", "="]) == ["-", "+", "="]
    assert K.to_selenium([","]) == [","]


@pytest.mark.parametrize(
    "backend,valid",
    [("playwright", _PW_VALID), ("pyautogui", _PYAUTOGUI_VALID), ("pynput", _PYNPUT_VALID)],
)
@pytest.mark.parametrize("k", sorted(_DESKTOP_ONLY))
def test_desktop_only_special_maps_or_raises_never_silent(backend, valid, k):
    # Contract: a desktop-only special either projects to a VALID backend token
    # or RAISES loudly — it must NEVER return a silently-invalid token.
    try:
        toks = K.translate_keys([k], backend)
    except ValueError:
        return  # loud failure is acceptable
    assert all(t in valid for t in toks), f"{k!r}->{toks} silently invalid in {backend}"


def test_lone_plus_glyph_projects_to_each_backend():
    assert K.to_xdotool(["+"]) == ["plus"]
    assert K.to_pyautogui(["+"]) == ["+"]
    assert K.to_playwright(["+"]) == ["+"]


def test_canonical_key_chord_projects_per_backend():
    assert K.to_xdotool(["ctrl", "o"]) == ["ctrl", "o"]
    assert K.to_playwright(["ctrl", "a"]) == ["Control", "a"]


@pytest.mark.parametrize("backend", ["playwright", "xdotool", "pyautogui", "pynput", "selenium"])
@pytest.mark.parametrize("bad", [
    "Ctrl", "A", "ctrl+a", "ArrowDown",
    "plus", "minus", "equal", "comma",
    "", " ", "\n", "\t", "\r", "\x1b", "\x00",
])
def test_projection_rejects_noncanonical_input(backend, bad):
    # Envs must receive canonical Lite key tokens: lowercase named keys plus
    # literal printable glyphs (normalized once at the LiteDesktopActionSpace.key()
    # factory). A non-canonical token — uppercase or an unsplit "+" chord, raw
    # alias, or non-printing glyph -- fails LOUD at the projection, never a silent
    # no-op.
    with pytest.raises(ValueError):
        K.translate_keys([bad], backend)


# --- loud failure on unmappable token (never silent) ---
@pytest.mark.parametrize("backend", ["playwright", "xdotool", "pyautogui", "pynput", "selenium"])
def test_bad_key_raises(backend):
    with pytest.raises(ValueError):
        K.translate_keys(["definitely_not_a_key"], backend)


def test_meta_per_backend():
    assert K.to_playwright(["meta"]) == ["Meta"]
    assert K.to_xdotool(["meta"]) == ["super"]      # GNOME Super
    assert K.to_pyautogui(["meta"]) == ["win"]      # pyautogui has no "meta"
    assert K.to_pynput(["meta"]) == ["cmd"]         # pynput's super key is Key.cmd


def test_media_keys_project_per_backend():
    assert K.to_xdotool(["volumeup"]) == ["XF86AudioRaiseVolume"]
    assert K.to_playwright(["volumeup"]) == ["AudioVolumeUp"]
    assert K.to_pyautogui(["volumeup"]) == ["volumeup"]
    assert K.to_pynput(["volumeup"]) == ["media_volume_up"]
    # play/pause + mute round out the set
    assert K.to_xdotool(["playpause"]) == ["XF86AudioPlay"]
    assert K.to_xdotool(["volumemute"]) == ["XF86AudioMute"]


def test_pynput_underscore_spellings():  # pynput.Key uses underscores + literal glyphs
    assert K.to_pynput(["pageup", "pagedown"]) == ["page_up", "page_down"]
    assert K.to_pynput(["capslock", "printscreen"]) == ["caps_lock", "print_screen"]
    assert K.to_pynput(["-", "+", "="]) == ["-", "+", "="]
    assert K.to_pynput(["ctrl", "s"]) == ["ctrl", "s"]   # common combo unchanged


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("xdotool", ["plus", "minus", "equal", "comma"]),
        ("playwright", ["+", "-", "=", ","]),
        ("pyautogui", ["+", "-", "=", ","]),
        ("pynput", ["+", "-", "=", ","]),
        ("selenium", ["+", "-", "=", ","]),
    ],
)
def test_punctuation_backends(backend, expected):
    assert K.translate_keys(["+", "-", "=", ","], backend) == expected
    assert K.to_xdotool(["-"]) == ["minus"]
    assert K.to_xdotool(["="]) == ["equal"]
    assert K.to_xdotool(["/"]) == ["slash"]
    assert K.to_playwright(["/"]) == ["/"]           # NOT "Divide" (numpad)
    assert K.to_playwright(["\\"]) == ["\\"]          # NOT "Backslash" (numpad)


def test_named_punctuation_hotkey_normalizes_before_xdotool_projection():
    # Regression: Qwen3.5 emitted Ctrl+comma for VS Code settings. The action
    # factory must canonicalize named punctuation before xdotool projection.
    keys = normalize_keys(["ctrl", "comma"])
    assert keys == ["ctrl", ","]
    assert K.to_xdotool(keys) == ["ctrl", "comma"]


def test_fkeys_all_backends():
    assert K.to_playwright(["f5"]) == ["F5"]
    assert K.to_xdotool(["f5"]) == ["F5"]
    assert K.to_pyautogui(["f5"]) == ["f5"]


def test_key_chord_roundtrip():
    assert K.to_playwright(["ctrl", "a"]) == ["Control", "a"]
    assert K.to_xdotool(["ctrl", "a"]) == ["ctrl", "a"]
    assert K.to_pyautogui(["ctrl", "a"]) == ["ctrl", "a"]


# --- REGRESSION GUARD: to_xdotool is byte-equal to the existing _norm_key, so
# migrating lite.osworld to the shared util changes nothing for that backend. ---
def test_xdotool_matches_existing_norm_key():
    # every name in the live _KEYSYM vocabulary + a sample of letters/punct
    from lite.gym.sandbox.exec_stdio.server import _KEYSYM, _PUNCT, _norm_key
    tokens = (
        list(_KEYSYM.keys())
        + list(_PUNCT.keys())
        + list("aZ0/.,;'`[]-+=")
        + ["Ctrl", "ENTER", "ArrowDown", "ctrl+a", "ctrl++"]  # casing/alias/chord forms
    )
    for tok in tokens:
        normalized = normalize_keys(tok)
        expected = [_norm_key(t) for t in normalized]
        # normalize (chokepoint) + to_xdotool (env projection) together reproduce
        # the old _norm_key behavior exactly → zero lite.osworld regression.
        got = K.to_xdotool(normalized)
        assert got == expected, f"{tok!r}: {got} != {expected}"
