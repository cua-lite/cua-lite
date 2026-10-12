"""Source tolerance boundaries reached through real recursive-grid clicks.

The existing 31-cell reference mask drives a GUI test oracle, not model play.
Every subdivision and final selection toggle uses the canonical input path;
neither evaluator state nor a completion event is written by these tests.
"""

from __future__ import annotations

import pytest

from examples.not_a_robot.tests.test_first10 import RECURSIVE_ROWS
from examples.not_a_robot.tests.test_local_tasks import click, gui
from examples.not_a_robot.tests.test_local_tasks import local_env as local_env

pytestmark = pytest.mark.live
REFERENCE_CELLS = tuple(
    (row, column) for row, columns in RECURSIVE_ROWS.items() for column in columns
)
EXTRA_CELLS = ((0, 0), (0, 1), (0, 2))


async def click_recursive_cells(env, cells):
    """Plan quadtree subdivisions, then send only genuine screen-coordinate clicks."""
    page = env.unwrapped._page
    photo = await page.locator(".neal-recursive-board").bounding_box()
    assert photo
    screen_width, screen_height = env.unwrapped.display_resolution
    leaves = [(0, 0, 16)]
    actions, split_depths = [], []
    for row, column in cells:
        coordinate = [
            (photo["x"] + (column + 0.5) / 16 * photo["width"]) * 1000 / screen_width,
            (photo["y"] + (row + 0.5) / 16 * photo["height"]) * 1000 / screen_height,
        ]
        while True:
            leaf = next(
                (x, y, size)
                for x, y, size in leaves
                if x <= column < x + size and y <= row < y + size
            )
            actions.append({"action": "click", "coordinate": coordinate})
            x, y, size = leaf
            if size == 1:
                break
            split_depths.append(4 - (size.bit_length() - 1))
            leaves.remove(leaf)
            half = size // 2
            leaves.extend(
                [(x, y, half), (x + half, y, half), (x, y + half, half), (x + half, y + half, half)]
            )
    for offset in range(0, len(actions), 8):
        result = await gui(env, actions[offset : offset + 8])
        assert not result.terminated and not result.truncated
    events = await page.evaluate("window.syntheticTask.snapshot().events")
    assert [event["depth"] for event in events if event["kind"] == "region_split"] == split_depths
    selections = [event for event in events if event["kind"] == "selection_changed"]
    assert len(selections) == len(cells)
    assert all(event["depth"] == 4 and event["width"] == 1 / 16 for event in selections)
    assert await page.locator(".neal-recursive-leaf").count() == len(leaves)


@pytest.mark.parametrize(
    "missing,extra,empty,accepted",
    [
        pytest.param(0, 0, False, True, id="exact"),
        pytest.param(1, 0, False, True, id="missing-one"),
        pytest.param(2, 0, False, True, id="missing-two"),
        pytest.param(0, 1, False, True, id="extra-one"),
        pytest.param(0, 2, False, True, id="extra-two"),
        pytest.param(1, 1, False, True, id="missing-one-extra-one"),
        pytest.param(2, 1, False, False, id="total-three-errors"),
        pytest.param(3, 0, False, False, id="missing-three"),
        pytest.param(0, 3, False, False, id="extra-three"),
        pytest.param(0, 0, True, False, id="empty"),
    ],
)
async def test_recursive_source_symmetric_difference_tolerance(
    local_env, missing, extra, empty, accepted
):
    env = await local_env("neal_09")
    raw, page = env.unwrapped, env.unwrapped._page
    assert len(REFERENCE_CELLS) == 31 and not set(EXTRA_CELLS) & set(REFERENCE_CELLS)
    # Select the reference first, then really toggle omitted terminal cells off.
    cells = () if empty else REFERENCE_CELLS + REFERENCE_CELLS[:missing] + EXTRA_CELLS[:extra]
    await click_recursive_cells(env, cells)
    expected = set() if empty else set(REFERENCE_CELLS[missing:]) | set(EXTRA_CELLS[:extra])
    visible_cells = await page.locator('.neal-recursive-leaf[aria-pressed="true"]').evaluate_all(
        "nodes => nodes.map(n=>[Math.round(parseFloat(n.style.top)*16/100),"
        "Math.round(parseFloat(n.style.left)*16/100)])"
    )
    assert {tuple(cell) for cell in visible_cells} == expected
    assert raw.state.progress == len(expected)
    assert len(expected ^ set(REFERENCE_CELLS)) == (31 if empty else missing + extra)
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert result.terminated is accepted
    assert not result.truncated
    assert result.reward == (1 if accepted else None)
    assert raw.state.status == ("success" if accepted else "in_progress")
    assert raw.state.mistakes == (0 if accepted else 1)
