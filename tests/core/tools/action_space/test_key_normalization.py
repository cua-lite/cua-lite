"""Regression pins for action-space key normalization."""

from __future__ import annotations

from pathlib import Path

import pytest

import lite.core.tools.action_space as core_action_space
import lite.core.tools.action_space.keys as key_owner
from lite.agents.core.action_space.base import LiteDesktopActionSpace
from lite.core.tools.action_space import normalize_keys
from lite.core.tools.action_space.keys import (
    canonical_special_keys,
    is_canonical_key_token,
)
from lite.core.tools.action_space.keys import (
    normalize_keys as owner_normalize_keys,
)
from lite.core.tools.calls import tool_call_arguments


def _only_action(tool_call: dict) -> dict:
    actions = tool_call_arguments(tool_call)["actions"]
    assert len(actions) == 1
    return actions[0]


def test_desktop_key_builders_normalize_common_model_spellings() -> None:
    assert _only_action(LiteDesktopActionSpace.key("ctrl+s"))["keys"] == [
        "ctrl",
        "s",
    ]
    assert _only_action(LiteDesktopActionSpace.key(["Control", "C"]))["keys"] == [
        "ctrl",
        "c",
    ]
    assert _only_action(LiteDesktopActionSpace.key_down("Command+Shift+P"))["keys"] == [
        "meta",
        "shift",
        "p",
    ]
    assert _only_action(LiteDesktopActionSpace.hold_key(["PageDown"], 0.25))["keys"] == [
        "pagedown",
    ]
    assert _only_action(LiteDesktopActionSpace.key("ctrl++"))["keys"] == [
        "ctrl",
        "+",
    ]
    assert _only_action(LiteDesktopActionSpace.key(["ctrl", "left"]))["keys"] == [
        "ctrl",
        "left",
    ]
    with pytest.raises(ValueError):
        LiteDesktopActionSpace.key(["ctrl", "", ""])


@pytest.mark.parametrize("action_name", ["key", "key_down", "key_up", "hold_key"])
def test_desktop_key_builders_reject_empty_key_lists(action_name: str) -> None:
    action = getattr(LiteDesktopActionSpace, action_name)
    args = ([], 0.25) if action_name == "hold_key" else ([],)

    with pytest.raises(ValueError, match="keys must not be empty"):
        action(*args)


def test_key_normalization_owner_and_public_exports_are_intentional() -> None:
    assert normalize_keys is owner_normalize_keys
    assert "ctrl" in canonical_special_keys()
    assert all(len(key) != 1 for key in canonical_special_keys())
    assert "canonical_special_keys" in key_owner.__all__
    assert "is_canonical_key_token" in key_owner.__all__

    for public_module in (core_action_space,):
        assert "canonical_special_keys" not in public_module.__all__
        assert "is_canonical_key_token" not in public_module.__all__
        assert "Backend" not in public_module.__all__
        assert "CANONICAL_SPECIAL_KEY_NAMES" not in public_module.__all__
    assert "Backend" not in key_owner.__all__
    assert "CANONICAL_SPECIAL_KEY_NAMES" not in key_owner.__all__
    assert not hasattr(key_owner, "Backend")

    for public_package in (core_action_space,):
        assert not hasattr(public_package, "canonical_special_keys")
        assert not hasattr(public_package, "is_canonical_key_token")
        assert not hasattr(public_package, "Backend")
        assert not hasattr(public_package, "CANONICAL_SPECIAL_KEY_NAMES")


def test_agent_key_normalization_leaf_is_deleted() -> None:
    repo = Path(__file__).resolve().parents[4]
    assert not (repo / "lite" / "agents" / "core" / "action_space" / "utils" / "keys.py").exists()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("+", ["+"]),
        ("ctrl+s", ["ctrl", "s"]),
        ("ctrl++", ["ctrl", "+"]),
        ("ctrl+-", ["ctrl", "-"]),
        ("ctrl+=", ["ctrl", "="]),
        ("ctrl+plus", ["ctrl", "+"]),
        ("ctrl+minus", ["ctrl", "-"]),
        ("ctrl+equal", ["ctrl", "="]),
        ("ctrl+shift+t", ["ctrl", "shift", "t"]),
    ],
)
def test_raw_plus_chord_grammar_accepts_explicit_forms(raw: str, expected: list[str]) -> None:
    assert normalize_keys(raw) == expected


@pytest.mark.parametrize("raw", ["ctrl+", "++", "+++", "+ctrl", "ctrl+++"])
def test_raw_plus_chord_grammar_rejects_ambiguous_forms(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_keys(raw)


@pytest.mark.parametrize("raw", [["ctrl", "", ""], ["ctrl", "", "shift"], ["", ""]])
def test_pre_split_plus_chord_artifacts_are_not_generic_normalizer_inputs(raw: list[str]) -> None:
    with pytest.raises(ValueError):
        normalize_keys(raw)


def test_key_list_boundaries_are_canonical_boundaries() -> None:
    assert normalize_keys(["ctrl", "+"]) == ["ctrl", "+"]
    assert normalize_keys(["ctrl", "left"]) == ["ctrl", "left"]
    assert normalize_keys(["alt", "left"]) == ["alt", "left"]
    assert normalize_keys(["ctrl", "shift", "left"]) == ["ctrl", "shift", "left"]
    with pytest.raises(ValueError, match="unknown key token"):
        normalize_keys(["ctrl+s"])
    with pytest.raises(ValueError, match="unknown key token"):
        normalize_keys(["ctrl++"])
    with pytest.raises(ValueError, match="unknown key token"):
        normalize_keys(["ctrl+plus"])
    with pytest.raises(ValueError, match="unknown key token"):
        normalize_keys(["page", "down"])
    with pytest.raises(ValueError, match="unknown key token"):
        normalize_keys(["ctrl+s", "enter"])
    assert normalize_keys(["page_down"]) == ["pagedown"]
    assert normalize_keys(["pageup"]) == ["pageup"]


@pytest.mark.parametrize(
    "bad",
    [
        "ctrl left",
        "ctrl-left",
        "alt left",
        "ctrl -",
        "arrow up",
        "arrow-up",
        "arrow down",
        "arrow-down",
        "arrow left",
        "arrow-left",
        "arrow right",
        "arrow-right",
        "page up",
        "page-up",
        "page down",
        "page-down",
        "caps lock",
        "caps-lock",
        "print screen",
        "print-screen",
        "foo",
        "é",
    ],
)
def test_key_normalization_does_not_alias_unknown_or_phrase_tokens(bad: str) -> None:
    with pytest.raises(ValueError, match="unknown key token"):
        normalize_keys([bad])


@pytest.mark.parametrize(
    ("alias", "canon"),
    [
        ("ControlLeft", "ctrl"),
        ("ControlRight", "ctrl"),
        ("CtrlLeft", "ctrl"),
        ("CtrlRight", "ctrl"),
        ("AltLeft", "alt"),
        ("AltRight", "alt"),
        ("OptionLeft", "alt"),
        ("OptionRight", "alt"),
        ("ShiftLeft", "shift"),
        ("ShiftRight", "shift"),
        ("CmdLeft", "meta"),
        ("CmdRight", "meta"),
        ("CommandLeft", "meta"),
        ("CommandRight", "meta"),
        ("MetaLeft", "meta"),
        ("MetaRight", "meta"),
        ("SuperLeft", "meta"),
        ("SuperRight", "meta"),
        ("WindowsLeft", "meta"),
        ("WindowsRight", "meta"),
        ("OsLeft", "meta"),
        ("OsRight", "meta"),
    ],
)
def test_side_modifier_aliases_are_explicit(alias: str, canon: str) -> None:
    assert normalize_keys([alias]) == [canon]


@pytest.mark.parametrize(
    ("alias", "canon"),
    [
        ("+", "+"),
        ("-", "-"),
        ("=", "="),
        ("comma", ","),
        ("period", "."),
        ("dot", "."),
        ("slash", "/"),
        ("backslash", "\\"),
        ("semicolon", ";"),
        ("apostrophe", "'"),
        ("singlequote", "'"),
        ("grave", "`"),
        ("backtick", "`"),
        ("bracketleft", "["),
        ("bracketright", "]"),
        ("plus", "+"),
        ("minus", "-"),
        ("equal", "="),
    ],
)
def test_punctuation_aliases_normalize_to_glyphs(alias: str, canon: str) -> None:
    assert normalize_keys([alias]) == [canon]
    assert normalize_keys([canon]) == [canon]
    assert normalize_keys([canon, canon]) == [canon, canon]


@pytest.mark.parametrize(
    ("alias", "canon"),
    [
        ("print_screen", "printscreen"),
        ("prtsc", "printscreen"),
        ("prtscr", "printscreen"),
        ("prntscrn", "printscreen"),
    ],
)
def test_print_screen_source_aliases_are_explicit(alias: str, canon: str) -> None:
    assert normalize_keys([alias]) == [canon]


@pytest.mark.parametrize(
    "token",
    ["ctrl", "enter", "space", "pagedown", "f24", "a", "0", "+", "-", "=", ",", "\\"],
)
def test_canonical_key_predicate_accepts_stored_tokens(token: str) -> None:
    assert is_canonical_key_token(token)


@pytest.mark.parametrize(
    "token",
    [
        None,
        1,
        ("ctrl",),
        "",
        " ",
        "\n",
        "\t",
        "\r",
        "\x1b",
        "\x00",
        "Ctrl",
        "A",
        "ctrl+a",
        "++",
        "plus",
        "minus",
        "equal",
        "comma",
        "return",
        "arrowdown",
        "ctrl left",
        "definitely_not_a_key",
        "é",
    ],
)
def test_canonical_key_predicate_rejects_raw_or_invalid_tokens(token: object) -> None:
    assert not is_canonical_key_token(token)


@pytest.mark.parametrize(
    ("alias", "canon"),
    [
        ("volumeup", "volumeup"),
        ("audiovolumeup", "volumeup"),
        ("xf86audioraisevolume", "volumeup"),
        ("AudioVolumeUp", "volumeup"),
        ("media_volume_up", "volumeup"),
        ("audio_volume_up", "volumeup"),
        ("media_volume_down", "volumedown"),
        ("audio_volume_down", "volumedown"),
        ("media_volume_mute", "volumemute"),
        ("audio_volume_mute", "volumemute"),
        ("media_play_pause", "playpause"),
        ("media_track_next", "nexttrack"),
        ("media_track_previous", "prevtrack"),
    ],
)
def test_media_key_source_aliases_are_explicit(alias: str, canon: str) -> None:
    assert normalize_keys([alias]) == [canon]


@pytest.mark.parametrize(
    "bad",
    [
        "print",
        "mute",
        "audioplay",
        "audionext",
        "audioprev",
        "audioraisevolume",
        "audiolowervolume",
        "audiovolup",
        "audio_vol_up",
        "audiovoldown",
        "audio_vol_down",
    ],
)
def test_key_normalization_rejects_unproven_convenience_aliases(bad: str) -> None:
    with pytest.raises(ValueError, match="unknown key token"):
        normalize_keys([bad])


def test_key_normalization_rejects_non_string_shapes() -> None:
    for value in [None, ("ctrl", "c"), {"keys": ["ctrl"]}, {"ctrl"}, b"ctrl"]:
        with pytest.raises(TypeError):
            normalize_keys(value)  # type: ignore[arg-type]

    for value in [1, True]:
        with pytest.raises(TypeError):
            normalize_keys(["ctrl", value])  # type: ignore[list-item]


@pytest.mark.parametrize("bad", ["", " ", "\n", "\t", "\r", "\x1b", "\x00"])
def test_key_normalization_rejects_empty_whitespace_and_control_tokens(bad: str) -> None:
    with pytest.raises(ValueError):
        normalize_keys([bad])
    with pytest.raises(ValueError):
        normalize_keys(bad)


def test_key_normalization_rejects_empty_key_lists() -> None:
    with pytest.raises(ValueError, match="keys must not be empty"):
        normalize_keys([])
