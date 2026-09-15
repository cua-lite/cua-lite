"""Visible chart feedback, sampled through real GUI input and browser pixels.

The test clock advances ordinary price ticks. Expected geometry is calculated
from the displayed quotes, not game state or a replacement price generator.
These are local source-derived regressions, not original-site or model runs.
"""

from __future__ import annotations

import math
from io import BytesIO

import pytest
from PIL import Image

from examples.not_a_robot.tests.test_motion_expansion import market, press
from examples.not_a_robot.tests.test_motion_expansion import motion_page as motion_page

pytestmark = pytest.mark.live
GREEN = (0, 212, 170)
RED = (255, 107, 107)


async def chart_image(page, path):
    """Keep the actual screenshot; map backing-canvas geometry to screen pixels."""
    chart = page.locator("canvas")
    box = await chart.bounding_box()
    assert box
    width, height = await chart.evaluate("node => [node.width, node.height]")
    image = Image.open(BytesIO(await page.screenshot(clip=box, path=str(path)))).convert("RGB")
    return image, width, height


def point(prices, index):
    low, high = 0.97 * min(prices), 1.03 * max(prices)
    return index / (len(prices) - 1), 1 - (prices[index] - low) / (high - low)


def color_count(image, box, color, tolerance=45):
    """Count screenshot colors in a normalized region, allowing edge antialiasing."""
    left, top, right, bottom = box
    crop = image.crop(
        (
            max(0, math.floor(left * image.width)),
            max(0, math.floor(top * image.height)),
            min(image.width, math.ceil(right * image.width)),
            min(image.height, math.ceil(bottom * image.height)),
        )
    )
    return sum(
        max(abs(component - target) for component, target in zip(pixel, color)) <= tolerance
        for pixel in crop.getdata()
    )


def assert_color_near(image, x, y, color, *, tolerance=45):
    dx, dy = 3 / image.width, 3 / image.height
    assert color_count(image, (x - dx, y - dy, x + dx, y + dy), color, tolerance) >= 2, (
        x,
        y,
        color,
    )


async def assert_no_history_fill(page):
    """A lone quote has no historical curve/area in the left three quarters."""
    green_pixels = await page.locator("canvas").evaluate(
        "c => { const pixels = c.getContext('2d')"
        ".getImageData(0,0,Math.floor(c.width*.75),c.height).data;"
        "let green=0; for(let i=0;i<pixels.length;i+=4)"
        "if(pixels[i+3]>0 && pixels[i+1]>pixels[i]+20 && pixels[i+1]>pixels[i+2]+5)"
        "green++; return green; }"
    )
    assert green_pixels == 0


async def test_market_single_sample_shows_no_invented_history(motion_page, tmp_path):
    page = await motion_page(28)
    image, _, _ = await chart_image(page, tmp_path / "single-first-sample.png")
    assert_color_near(image, 1, 0.5, GREEN)
    await assert_no_history_fill(page)


@pytest.mark.parametrize("samples", [2, 8, 30])
async def test_market_dynamic_range_and_full_width(motion_page, tmp_path, samples):
    page = await motion_page(28)
    prices = [(await market(page))[0]]
    for _ in range(samples - 1):
        await page.clock.run_for(500)
        prices.append((await market(page))[0])
    image, _, _ = await chart_image(page, tmp_path / f"samples-{samples}.png")
    # Interior segment midpoints avoid the latest-price label and axis text.
    for index in sorted({0, (samples - 2) // 2, samples - 2}):
        x0, y0 = point(prices, index)
        x1, y1 = point(prices, index + 1)
        assert_color_near(image, (x0 + x1) / 2, (y0 + y1) / 2, GREEN)
    assert_color_near(image, *point(prices, samples - 1), GREEN)


async def test_market_trade_markers_stay_on_the_traded_sample(motion_page, tmp_path):
    page = await motion_page(28)
    prices = [(await market(page))[0]]
    await page.clock.run_for(500)
    prices.append((await market(page))[0])
    await press(page, "Buy one share")
    assert (await market(page))[2] == 1
    await press(page, "Sell one share")
    assert (await market(page))[1:] == (500.0, 0, 0.0)
    await page.clock.run_for(500)
    prices.append((await market(page))[0])
    await press(page, "Buy one share")
    assert (await market(page))[1:3] == (500.0 - prices[-1], 1)
    await page.clock.run_for(500)
    prices.append((await market(page))[0])
    image, _, _ = await chart_image(page, tmp_path / "markers-four-samples.png")
    for index, color in [(1, RED), (2, GREEN)]:
        x, y = point(prices, index)
        assert_color_near(image, x, y, color)
        # A 3-backing-pixel stroke can split across two CSS pixels: measured
        # green (4,165,134), against neighboring fill (14,35,31), differs by 47.
        assert_color_near(image, x, (y + 1) / 2, color, tolerance=50)
    # The last-price dot no longer covers the same-tick buy/sell sample.
    # Source drawing order leaves that historical dot and line red.
    await page.clock.run_for(500)
    prices.append((await market(page))[0])
    moved, _, _ = await chart_image(page, tmp_path / "markers-five-samples.png")
    for index, color in [(1, RED), (2, GREEN)]:
        x, y = point(prices, index)
        assert_color_near(moved, x, y, color)
        assert_color_near(moved, x, (y + 1) / 2, color, tolerance=50)
    assert color_count(moved, (0.32, 0, 0.35, 1), RED) == 0
    assert await page.locator(".neal-market-portfolio").inner_text() == (
        f"Portfolio value ${prices[-1]:.2f}"
    )


async def test_market_axis_labels_use_twenty_and_forty_dollar_steps(motion_page, tmp_path):
    page = await motion_page(28)
    prices = [(await market(page))[0]]
    checked = set()
    for _ in range(120):
        await page.clock.run_for(500)
        prices = [*prices[-29:], (await market(page))[0]]
        low, high = 0.97 * min(prices), 1.03 * max(prices)
        step = 20 if high - low <= 100 else 40
        ticks = list(range(math.ceil(low / step) * step, math.floor(high / step) * step + 1, step))
        if step in checked or len(ticks) < 2:
            continue
        image, width, height = await chart_image(page, tmp_path / f"axis-step-{step}.png")
        for value in ticks:
            y = 1 - (value - low) / (high - low)
            # Axis glyphs are light neutral pixels, distinct from the green plot
            # and its fill. The source label is right-aligned at backing x=50.
            region = image.crop(
                (
                    0,
                    max(0, math.floor((y - 11 / height) * image.height)),
                    math.ceil(51 / width * image.width),
                    min(image.height, math.ceil((y + 11 / height) * image.height)),
                )
            )
            glyphs = sum(
                95 < min(pixel) < 240 and max(pixel) - min(pixel) < 20 for pixel in region.getdata()
            )
            assert glyphs >= 4, (step, value, glyphs)
            # Inspect actual canvas pixels as well, excluding its CSS grid.
            # A price-axis line adds red/alpha above the green fill or transparent
            # canvas. Isolated glyphs/curve crossings cannot cover many columns.
            center = round(y * height)
            if 4 <= center < height - 4:
                pixels = await page.locator("canvas").evaluate(
                    "(c, y) => Array.from(c.getContext('2d')"
                    ".getImageData(60,y-4,c.width-60,9).data)",
                    center,
                )
                grid = Image.frombytes("RGBA", (width - 60, 9), bytes(pixels))
                raised_columns = sum(
                    max(grid.getpixel((x, row))[0] for row in range(2, 7))
                    > max(grid.getpixel((x, 0))[0], grid.getpixel((x, 8))[0]) + 20
                    for x in range(width - 60)
                )
                assert raised_columns > (width - 60) / 5, (step, value, raised_columns)
        checked.add(step)
        if checked == {20, 40}:
            break
    assert checked == {20, 40}


async def test_market_latest_point_has_white_edge_and_visible_price_label(motion_page, tmp_path):
    page = await motion_page(28)
    prices = [(await market(page))[0]]
    for sample in range(1, 4):
        await page.clock.run_for(500)
        prices.append((await market(page))[0])
        image, width, height = await chart_image(page, tmp_path / f"latest-{sample}.png")
        _, y = point(prices, len(prices) - 1)
        assert_color_near(image, 1, y, GREEN)
        # The point is clipped at the right border; its left semicircle remains visible.
        assert (
            color_count(
                image,
                (1 - 11 / width, y - 11 / height, 1, y + 11 / height),
                (255, 255, 255),
                tolerance=75,
            )
            >= 1
        )
        label_y = y * height - 30
        if label_y - 16 < 20:
            label_y = y * height + 40
        label_box = (
            1 - 140 / width,
            (label_y - 16) / height,
            1 - 16 / width,
            (label_y + 16) / height,
        )
        assert color_count(image, label_box, GREEN) >= 6
        assert color_count(image, label_box, (0, 0, 0), tolerance=30) >= 15


async def test_market_rolling_markers_and_refresh_leave_no_old_feedback(motion_page, tmp_path):
    page = await motion_page(28)
    # Record a red sample at t=0. No hidden ledger or history writes are used.
    await press(page, "Buy one share")
    await press(page, "Sell one share")
    await page.clock.run_for(29 * 500)
    image, _, _ = await chart_image(page, tmp_path / "oldest-marker-at-thirty.png")
    assert color_count(image, (0, 0, 0.015, 1), RED) >= 3
    await page.clock.run_for(500)
    rolled, _, _ = await chart_image(page, tmp_path / "oldest-marker-expired.png")
    assert color_count(rolled, (0, 0, 1, 1), RED) == 0
    await press(page, "Buy one share")
    assert (await market(page))[2] == 1
    await press(page, "Refresh challenge")
    assert (await market(page))[1:] == (500.0, 0, 0.0)
    assert await page.locator(".neal-market-portfolio").inner_text() == "Portfolio value $0.00"
    assert await page.get_by_role("button", name="Buy one share", exact=True).is_disabled()
    cleared, _, _ = await chart_image(page, tmp_path / "refreshed.png")
    assert color_count(cleared, (0, 0, 1, 1), GREEN) == 0
    assert color_count(cleared, (0, 0, 1, 1), RED) == 0
    await page.clock.run_for(499)
    assert await page.get_by_role("button", name="Buy one share", exact=True).is_disabled()
    await page.clock.run_for(1)
    assert await page.get_by_role("button", name="Buy one share", exact=True).is_enabled()
    restarted, _, _ = await chart_image(page, tmp_path / "first-sample-after-refresh.png")
    assert_color_near(restarted, 1, 0.5, GREEN)
    await assert_no_history_fill(page)
