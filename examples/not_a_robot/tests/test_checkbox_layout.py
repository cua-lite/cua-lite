"""Source checkbox geometry observed through an owned real browser.

DOM bounds are a scripted layout oracle, not model input. Interaction checks use
canonical GUI actions; neither evaluator state nor original scripts are injected.
"""

from __future__ import annotations

import hashlib

import pytest

from examples.not_a_robot.tests.test_expansion_grids import scroll_click
from examples.not_a_robot.tests.test_local_tasks import gui
from examples.not_a_robot.tests.test_local_tasks import local_env as local_env

pytestmark = pytest.mark.live
LABELS_SHA256 = "588eb2cd3ffd67fb956f0e77e34274925b2b5fcd69ebf195998bb461562dade3"


async def checkbox_geometry(page, selector):
    """Observe card and visible child bounds without mutating scroll or layout."""
    return await page.locator(selector).evaluate_all(
        "nodes => nodes.map(node => {const box = el => el.getBoundingClientRect().toJSON(); "
        "const caption=node.querySelector('.neal-checkbox-caption'); "
        "const wrapper=node.querySelector('.neal-checkbox-mark-container'); "
        "const wrapperCss=getComputedStyle(wrapper); "
        "return {card:box(node),caption:box(caption), "
        "mark:box(node.querySelector('.neal-checkbox-mark')), "
        "wrapper:box(wrapper),wrapperMargins:[wrapperCss.marginLeft,wrapperCss.marginRight], "
        "logo:box(node.querySelector('.neal-recaptcha-mark')), "
        "svg:box(node.querySelector('.neal-recaptcha-mark svg')), "
        "lineHeight:getComputedStyle(caption).lineHeight}})"
    )


def assert_card_content_fits(cards):
    for item in cards:
        card, caption, mark, logo = (item[key] for key in ("card", "caption", "mark", "logo"))
        assert card["height"] == pytest.approx(76, abs=0.1)
        assert mark["right"] <= caption["left"] + 0.1
        assert caption["right"] <= logo["left"] + 0.1
        for child in (caption, mark, logo):
            assert card["left"] - 0.1 <= child["left"] < child["right"] <= card["right"] + 0.1
            assert card["top"] - 0.1 <= child["top"] < child["bottom"] <= card["bottom"] + 0.1
        assert item["svg"]["width"] == pytest.approx(32, abs=0.1)
        assert item["svg"]["height"] == pytest.approx(32, abs=0.1)
        assert item["wrapper"]["width"] == pytest.approx(26, abs=0.1)
        assert item["wrapperMargins"] == ["14px", "14px"]
        assert mark["width"] == mark["height"] == 27
        assert item["lineHeight"] == "normal"


@pytest.mark.parametrize(
    "width,columns",
    [
        (375, 1),
        (700, 1),
        (701, 2),
        (1060, 2),
        (1061, 3),
        (1280, 3),
        (1400, 3),
        (1401, 4),
        (1600, 4),
    ],
)
async def test_statement_source_grid_columns(local_env, width, columns):
    env = await local_env("neal_14", display_resolution=(width, 800))
    page = env.unwrapped._page
    cards = await checkbox_geometry(page, ".neal-statement-card")
    boxes = [item["card"] for item in cards]
    await page.screenshot(path=str(env.unwrapped.attempt_dir / "statement-grid.png"))
    assert len(boxes) == 56
    assert sum(abs(box["y"] - boxes[0]["y"]) < 0.1 for box in boxes) == columns
    document = await page.evaluate(
        "({viewport:innerWidth,client:document.documentElement.clientWidth,"
        "scrollWidth:document.documentElement.scrollWidth,"
        "scrollHeight:document.documentElement.scrollHeight})"
    )
    assert document["viewport"] == width
    assert document["scrollWidth"] == document["client"]
    root = await page.locator(".neal-statements").evaluate(
        "node => {const r=node.getBoundingClientRect(); const css=getComputedStyle(node); "
        "return {box:r.toJSON(),padding:css.padding,gap:css.gap,boxSizing:css.boxSizing,"
        "overflowY:css.overflowY,scrollTop:node.scrollTop,"
        "clientHeight:node.clientHeight,scrollHeight:node.scrollHeight}}"
    )
    assert root["box"]["x"] == root["box"]["y"] == 0
    assert root["box"]["width"] == pytest.approx(min(document["client"], 1500), abs=0.1)
    assert root["padding"] == "13px" and root["gap"] == "10px"
    assert root["boxSizing"] == "border-box"
    assert root["overflowY"] == "visible" and root["scrollTop"] == 0
    assert root["clientHeight"] == root["scrollHeight"]
    expected_width = (root["box"]["width"] - 26 - 10 * (columns - 1)) / columns
    for index, box in enumerate(boxes):
        assert box["width"] == pytest.approx(expected_width, abs=0.1)
        assert box["x"] == pytest.approx(13 + (index % columns) * (expected_width + 10), abs=0.1)
        assert box["y"] == pytest.approx(13 + (index // columns) * 86, abs=0.1)
    assert root["box"]["height"] == pytest.approx(boxes[-1]["bottom"] + 13, abs=0.1)
    assert document["scrollHeight"] > 800
    labels = await page.get_by_role("checkbox").evaluate_all(
        "nodes => nodes.map(node => node.getAttribute('aria-label'))"
    )
    # Fixed source 1096 text literals in DOM order, not a hash of production code.
    assert hashlib.sha256("\n".join(labels).encode()).hexdigest() == LABELS_SHA256
    assert labels[33] == "I'm not a robot"
    assert_card_content_fits(cards)


@pytest.mark.parametrize("width", [375, 700, 1600])
async def test_single_checkbox_uses_embed_origin_not_main_site_margin(local_env, width):
    env = await local_env("neal_01", display_resolution=(width, 800))
    page = env.unwrapped._page
    cards = await checkbox_geometry(page, ".neal-checkbox-card")
    assert len(cards) == 1
    card = cards[0]["card"]
    await page.screenshot(path=str(env.unwrapped.attempt_dir / "single-checkbox.png"))
    assert card["x"] == card["y"] == 0
    assert card["width"] == pytest.approx(315, abs=0.1)
    assert_card_content_fits(cards)


async def test_statements_use_document_scroll_and_correct_card_remains_clickable(local_env):
    env = await local_env("neal_14", display_resolution=(375, 800))
    raw, page = env.unwrapped, env.unwrapped._page
    correct = page.get_by_role("checkbox", name="I'm not a robot", exact=True)
    initial = await correct.bounding_box()
    assert initial["y"] > 800
    result = await scroll_click(env, correct)
    assert not result.terminated and not result.truncated
    assert await page.evaluate("window.scrollY") > 0
    assert await page.locator(".neal-statements").evaluate("node => node.scrollTop") == 0
    await page.screenshot(path=str(raw.attempt_dir / "document-scrolled-click.png"))
    result = await gui(env, [{"action": "wait", "duration": 2}])
    assert result.terminated and not result.truncated and result.reward == 1
    assert raw.state.mistakes == 0
    events = await page.evaluate("window.syntheticTask.snapshot().events")
    loading = [event for event in events if event["kind"] == "checkbox_loading"]
    assert len(loading) == 1 and loading[0]["index"] == 33
