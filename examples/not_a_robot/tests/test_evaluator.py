"""Offline checks for visible-heading and access-gate classification."""

from __future__ import annotations

import pytest

from examples.not_a_robot.evaluator import classify_page

GAME_URL = "https://neal.fun/not-a-robot/"


@pytest.mark.parametrize(
    "text",
    [
        "Level 1: Checkbox\nI'm not a robot",
        "Level 1: Checkbox\nVerify you are human",
        "Level 1: Checkbox\nProve that you are not a robot",
    ],
)
def test_synthetic_robot_wording_is_not_an_access_gate(text: str) -> None:
    state = classify_page(url=GAME_URL, title="I'm not a robot", text=text)
    assert state.status == "in_progress"
    assert state.level == 1
    assert state.label == "Checkbox"


@pytest.mark.parametrize("level", [1, 17, 48])
def test_unique_visible_heading(level: int) -> None:
    state = classify_page(
        url=GAME_URL,
        title="I'm not a robot",
        text=f"Neal.fun\nLevel {level}: Visible task\nTask instructions",
    )
    assert state.level == level
    assert state.label == "Visible task"
    assert state.status == "in_progress"
    assert f"Level {level}" in state.evidence


@pytest.mark.parametrize(
    "gate",
    [
        {"http_status": 403},
        {"cf_mitigated": "challenge"},
        {"cf_mitigated": " Challenge "},
        {"title": "Just a moment..."},
        {"title": "Attention Required! | Cloudflare"},
        {"title": "Access denied | Cloudflare"},
        {"url": "https://neal.fun/cdn-cgi/challenge-platform/h/g/orchestrate/chl_page/v1"},
        {"text": "Level 1: Checkbox\nCloudflare\nVerify you are human"},
        {"text": "Level 1: Checkbox\nCloudflare\nSorry, you have been blocked"},
    ],
)
def test_explicit_access_gate_takes_precedence_over_level_heading(gate: dict) -> None:
    inputs = {"url": GAME_URL, "title": "I'm not a robot", "text": "Level 1: Checkbox"}
    inputs.update(gate)
    state = classify_page(**inputs)
    assert state.status == "access_blocked"
    assert state.level is None
    assert state.label is None
    assert state.evidence


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Congratulations! You won!",
        "Level 0: Out of range",
        "Level 49: Out of range",
        "Level 1: Checkbox\nLevel 2: Keyboard",
        "Level 1: Checkbox\nLevel 1: Different task",
        "Instructions mention Level 1: Checkbox within prose",
    ],
)
def test_missing_ambiguous_or_out_of_scope_headings_are_unrecognized(text: str) -> None:
    state = classify_page(url=GAME_URL, title="I'm not a robot", text=text)
    assert state.status == "unrecognized"
    assert state.level is None
    assert state.label is None


def test_identical_repeated_heading_is_unambiguous() -> None:
    state = classify_page(
        url=GAME_URL,
        title="I'm not a robot",
        text="Level 1: Checkbox\nLevel 1: Checkbox",
    )
    assert state.level == 1
    assert state.label == "Checkbox"


def test_heading_without_label_does_not_consume_instructions() -> None:
    state = classify_page(
        url=GAME_URL,
        title="I'm not a robot",
        text="Level 1:\nClick the checkbox below.",
    )
    assert state.level == 1
    assert state.label is None


def test_crlf_whitespace_and_case() -> None:
    state = classify_page(
        url=GAME_URL,
        title="I'm not a robot",
        text="Introduction\r\n  level 17 :  Keyboard and drag  \r\nInstructions",
    )
    assert state.level == 17
    assert state.label == "Keyboard and drag"
