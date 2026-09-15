"""Incremental captured instances, kept separate from default references and scores."""

from __future__ import annotations

import json
from urllib.parse import parse_qs, urlsplit

import pytest

from examples.not_a_robot import codex_bridge, codex_smoke
from examples.not_a_robot.env import NotARobotEnv
from examples.not_a_robot.local_tasks import CATALOG, task_reference
from examples.not_a_robot.reference_assets import REFERENCE_CATALOG
from examples.not_a_robot.tests.test_first10 import tic_board
from examples.not_a_robot.tests.test_local_tasks import click, gui
from examples.not_a_robot.tests.test_local_tasks import local_env as local_env
from lite import gym

INCREMENTAL_GRID = (
    "PCJYGSFLYT",
    "FFIWSNMENV",
    "PLSIWDSUCL",
    "BSSOGOTHUB",
    "JOEWVEOWFE",
    "PMYJDFPOLH",
    "LDFFRGSESM",
    "GNMWEKIBOL",
    "FFFCRAGSTO",
    "BPMTONNGAY",
)
INCREMENTAL_SELECTED = {26, 36, 46, 56, 66, 76, 86, 96, 77, 75, 74}


@pytest.mark.live
async def test_default_word_search_keeps_its_original_full_grid(local_env):
    env = await local_env("neal_07")
    expected = (
        "UILIBEWESD",
        "NPVBVFGBUK",
        "YGVOOLKOYW",
        "KBIRHOYNDD",
        "CSUSGUASGP",
        "LHFAPCWLWO",
        "KITIYOCWPN",
        "PMLUBJTFEJ",
        "JNWIYDVSRA",
        "IBUGUBIKEK",
    )
    assert "".join(await env.unwrapped._page.locator(".neal-tile").all_text_contents()) == "".join(
        expected
    )
    assert env.unwrapped.state.reference_instance == "default"


@pytest.mark.parametrize("task", ["neal_07", "neal_08"])
def test_instance_has_separate_evidence_and_metadata(task):
    original = task_reference(task)
    selected = task_reference(task, "incremental")
    assert original["supplement_sha256"] == REFERENCE_CATALOG["archives"]["first10"]
    assert selected["supplement_sha256"] == REFERENCE_CATALOG["archives"]["incremental"]
    assert selected["completion_basis"] == "visible_main_site_transition"
    assert not selected["official_completion_message_recorded"]
    assert selected["policy_mode"] == "dom_assisted"
    env = gym.make(f"visual_tasks@{task}", reference_instance="incremental", seed=13)
    metadata = env.metadata.others
    assert metadata["reference"] == selected
    assert metadata["reference_instance"] == "incremental"
    assert metadata["seed"] == 13 and metadata["game_version"] == CATALOG["version"]
    assert env.unwrapped._page is None


@pytest.mark.parametrize("task,instance", [("neal_06", "incremental"), ("neal_07", "typo")])
def test_unknown_instance_rejected_before_browser(task, instance):
    with pytest.raises(ValueError, match="Unknown reference_instance"):
        gym.make(f"visual_tasks@{task}", reference_instance=instance)


@pytest.mark.parametrize("mode,campaign", [("live", False), ("fixture", False), ("local", True)])
def test_nondefault_instance_cannot_silently_enter_other_modes(mode, campaign):
    with pytest.raises(ValueError, match="independent local task"):
        NotARobotEnv(
            mode=mode, target_level=None, local_campaign=campaign, reference_instance="incremental"
        )


@pytest.mark.parametrize("module,target", [(codex_smoke, "--tasks"), (codex_bridge, "--task")])
def test_cli_selects_instance_and_rejects_invalid_pairs(module, target, tmp_path):
    base = [
        "--artifact-root",
        str(tmp_path / "new-run"),
        "--browser-executable",
        "/chrome",
        "--model",
        "gpt-6-astra",
        "--reasoning-effort",
        "xhigh",
    ]
    args = module._parse_args(
        [*base, target, "neal_07", "--reference-instance", "incremental", "--seed", "13"]
    )
    assert args.seed == 13 and args.reference_instance == "incremental"
    for invalid in (
        [target, "neal_06", "--reference-instance", "incremental"],
        ["--campaign", "--reference-instance", "incremental"],
        [target, "neal_07", "--seed", "-1"],
    ):
        with pytest.raises(SystemExit):
            module._parse_args([*base, *invalid])


@pytest.mark.live
async def test_incremental_word_search_shared_cell_and_provenance(local_env):
    env = await local_env("neal_07", reference_instance="incremental")
    raw, page = env.unwrapped, env.unwrapped._page
    assert parse_qs(urlsplit(page.url).query)["instance"] == ["incremental"]
    assert raw.state.reference_instance == "incremental"
    tiles = page.locator(".neal-tile")
    assert "".join(await tiles.all_text_contents()) == "".join(INCREMENTAL_GRID)
    for index in sorted(INCREMENTAL_SELECTED):
        await click(env, tiles.nth(index))
    assert raw.state.progress == 11
    # The shared I in BIKE and STOPSIGN is one selected tile, not two toggles.
    await click(env, tiles.nth(76))
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert not result.terminated and raw.state.progress == 10 and raw.state.mistakes == 1
    await click(env, tiles.nth(76))
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert result.terminated and result.reward == 1
    attempt = raw.attempt_dir
    await env.reset()
    assert raw.state.reference_instance == "incremental" and raw.state.progress == 0
    assert "".join(await raw._page.locator(".neal-tile").all_text_contents()) == "".join(
        INCREMENTAL_GRID
    )
    manifest = json.loads((attempt / "manifest.json").read_text())
    assert manifest["recording_complete"] and manifest["data"]["cleanup_complete"]
    assert manifest["outcome"] == "success"


@pytest.mark.live
@pytest.mark.parametrize(
    "instance,answer,other",
    [
        ("default", "867V 309", "JHB007"),
        ("incremental", "JHB007", "867V 309"),
    ],
)
async def test_plate_instances_do_not_share_images_or_answers(local_env, instance, answer, other):
    env = await local_env("neal_08", reference_instance=instance)
    raw, page = env.unwrapped, env.unwrapped._page
    image = page.locator(".neal-plate-crop img")
    expected = (
        "level08_incremental_reference.jpg"
        if instance == "incremental"
        else "level08_reference.jpg"
    )
    assert (await image.get_attribute("src")).endswith(expected)
    await page.screenshot(path=str(raw.attempt_dir / "plate-initial.png"))
    assert await page.locator("#neal-answer").input_value() == ""
    await click(env, page.locator("#neal-answer"))
    result = await gui(env, [{"action": "type", "text": other, "press_enter": True}])
    assert not result.terminated and raw.state.mistakes == 1
    # Exact matching is the conservative local rule, not evidence that the
    # original site rejected every other spelling or whitespace variation.
    variant = "JHB 007" if instance == "incremental" else "867V309"
    result = await gui(
        env,
        [
            {"action": "key", "keys": ["ctrl", "a"]},
            {"action": "type", "text": variant, "press_enter": True},
        ],
    )
    assert not result.terminated and raw.state.mistakes == 2
    result = await gui(
        env,
        [
            {"action": "key", "keys": ["ctrl", "a"]},
            {"action": "type", "text": answer, "press_enter": True},
        ],
    )
    assert result.terminated and result.reward == 1


@pytest.mark.live
async def test_tic_policy_priority_ties_and_immutability(local_env):
    env = await local_env("neal_06")
    page = env.unwrapped._page
    for board, expected in [
        (["X", "", "", "", "", "", "", "", ""], [1, 2, 3, 4, 5, 6, 7, 8]),
        (["O", "O", "", "X", "X", "", "", "", ""], [2]),
        (["X", "", "", "", "X", "", "", "O", ""], [8]),
        (["X", "", "", "", "X", "", "X", "O", "O"], [2, 3]),
    ]:
        result = await page.evaluate(
            "board => ({candidates: ticTacToeCandidates(board), board})", board
        )
        assert result == {"candidates": expected, "board": board}


@pytest.mark.live
async def test_tic_seed_reset_repeats_reply_and_refresh_clears_marks(local_env):
    env = await local_env("neal_06", seed=0)
    boards = []
    for _ in range(2):
        page = env.unwrapped._page
        await click(env, page.locator(".neal-tile").nth(4))
        assert env.unwrapped.state.progress == 0  # Occupied cell cannot be overwritten.
        await click(env, page.get_by_role("button", name="Refresh challenge"))
        assert await tic_board(page) == ("",) * 9
        await click(env, page.locator(".neal-tile").nth(0))
        await gui(env, [{"action": "wait", "duration": 0.45}])
        board = await tic_board(page)
        assert board[0] == "X" and board.count("O") == 1
        boards.append(board)
        result = await click(env, page.get_by_role("button", name="Verify", exact=True))
        assert not result.terminated and result.reward is None
        await env.reset()
    assert boards[0] == boards[1]


@pytest.mark.live
@pytest.mark.parametrize("moves,reason", [([0, 1], "opponent_win"), ([0, 6, 5, 7], "draw_not_win")])
async def test_local_seed_zero_loss_and_draw_never_award_success(local_env, moves, reason):
    """Local fixed-seed regression paths, not original-site reply replay."""
    env = await local_env("neal_06", seed=0)
    raw, page = env.unwrapped, env.unwrapped._page
    for index in moves:
        await click(env, page.locator(".neal-tile").nth(index))
        await gui(env, [{"action": "wait", "duration": 0.45}])
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert not result.terminated and result.reward is None
    assert raw.outcome == "in_progress" and raw.state.status == "in_progress"
    events = [
        json.loads(line) for line in (raw.attempt_dir / "events.jsonl").read_text().splitlines()
    ]
    rejections = [
        row["data"]
        for row in events
        if row["type"] == "game_event" and row["data"]["kind"] == "rejected"
    ]
    assert rejections[-1]["reason"] == reason
