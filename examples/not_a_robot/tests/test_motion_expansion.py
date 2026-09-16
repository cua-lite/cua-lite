"""Owned-browser UI oracles for seven authored motion renderers.

Playwright's test clock advances timers without changing game state. The
production renderers still use real RAF, performance.now(), and timers. All
challenge inputs are trusted mouse/keyboard actions, never evaluator writes.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image, ImageChops
from playwright.async_api import async_playwright

LOCAL = Path(__file__).parents[1] / "local"
EPOCH = datetime(2026, 9, 15, tzinfo=UTC)


@pytest.fixture
async def motion_page():
    async with async_playwright() as playwright:
        executable = os.environ.get("NEAL_BROWSER_EXECUTABLE")
        browser = await playwright.chromium.launch(executable_path=executable, headless=True)
        pages = []

        async def create(level: int, *, clock: bool = True):
            page = await browser.new_page(viewport={"width": 800, "height": 850})
            pages.append(page)
            await page.set_content("<!doctype html><html><body><main></main></body></html>")
            await page.add_style_tag(path=str(LOCAL / "neal.css"))
            if clock:
                await page.clock.install(time=EPOCH)
                await page.clock.pause_at(EPOCH)
            await page.add_script_tag(path=str(LOCAL / "neal.js"))
            await page.add_script_tag(path=str(LOCAL / "neal_motion.js"))
            task = next(
                item
                for item in json.loads((LOCAL / "tasks.json").read_text())["tasks"]
                if item["id"] == f"neal_{level:02}"
            )
            await page.evaluate(
                "task => window.renderNealTask({task,seed:17,version:'test',"
                "referenceInstance:'source-derived'})",
                task,
            )
            return page

        yield create
        for page in pages:
            await page.close()
        await browser.close()


async def snapshot(page):
    return await page.evaluate("window.syntheticTask.snapshot()")


async def press(page, name: str):
    """Use low-level mouse input so a paused clock needs no stability polling."""
    box = await page.get_by_role("button", name=name, exact=True).bounding_box()
    assert box
    await page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)


async def stroke(page, selector: str, points: list[tuple[float, float]]):
    box = await page.locator(selector).bounding_box()
    assert box
    x, y = points[0]
    await page.mouse.move(box["x"] + x * box["width"], box["y"] + y * box["height"])
    await page.mouse.down()
    for x, y in points[1:]:
        await page.mouse.move(box["x"] + x * box["width"], box["y"] + y * box["height"])
    await page.mouse.up()


@pytest.mark.live
async def test_circle_continuous_path_scoring_failure_and_refresh(motion_page):
    page = await motion_page(17)
    await press(page, "Verify")
    assert (await snapshot(page))["mistakes"] == 1
    await stroke(page, ".neal-circle-canvas", [(0.5, 0.5), (0.52, 0.5), (0.55, 0.5)])
    assert "farther" in await page.locator(".neal-circle-score").inner_text()
    # A partial high-quality arc resets to zero on release.
    arc = [
        (0.5 + 0.3 * math.cos(a), 0.5 + 0.3 * math.sin(a))
        for a in [i * math.pi / 80 for i in range(30)]
    ]
    await stroke(page, ".neal-circle-canvas", arc)
    assert (await snapshot(page))["progress"] == 0
    await press(page, "Refresh challenge")
    circle = [
        (0.5 + 0.3 * math.cos(i * math.tau / 120), 0.5 + 0.3 * math.sin(i * math.tau / 120))
        for i in range(122)
    ]
    await stroke(page, ".neal-circle-canvas", circle)
    assert (await snapshot(page))["progress"] >= 94
    # Refresh preserves the displayed best ratio, but not a valid current score.
    await press(page, "Refresh challenge")
    assert "Best 99." in await page.locator(".neal-circle-score").inner_text()
    await press(page, "Verify")
    assert (await snapshot(page))["status"] == "in_progress"
    await stroke(page, ".neal-circle-canvas", circle)
    await press(page, "Verify")
    assert (await snapshot(page))["status"] == "success"


@pytest.mark.live
async def test_creativity_counts_tools_last_color_and_clear_retention(motion_page):
    page = await motion_page(25)
    await press(page, "Verify")
    for index in range(10):
        if index in (1, 2):
            await press(page, "Spray" if index == 1 else "Pencil")
        await stroke(
            page,
            ".neal-creativity-canvas",
            [(0.1, 0.15 + index * 0.06), (0.7, 0.15 + index * 0.06)],
        )
    await press(page, "Verify")
    assert (await snapshot(page))["status"] == "in_progress"
    await page.get_by_label("Drawing color").fill("#ff0000")
    await stroke(page, ".neal-creativity-canvas", [(0.15, 0.85), (0.8, 0.85)])
    await page.get_by_label("Drawing color").fill("#000000")
    await stroke(page, ".neal-creativity-canvas", [(0.2, 0.9), (0.8, 0.9)])
    await press(page, "Verify")
    assert (await snapshot(page))["status"] == "in_progress"
    await press(page, "Refresh challenge")
    assert (await snapshot(page))["progress"] == 12
    # Pixel read is a visible-canvas oracle, not an answer/state mutation.
    assert await page.locator("canvas").evaluate(
        "c=>Array.from(c.getContext('2d').getImageData(200,200,1,1).data)"
    ) == [255, 255, 255, 255]
    await page.get_by_label("Drawing color").fill("#00aa66")
    await stroke(page, ".neal-creativity-canvas", [(0.2, 0.4), (0.8, 0.4)])
    await press(page, "Verify")
    assert (await snapshot(page))["status"] == "success"


async def market(page):
    price = int(re.search(r"\$(\d+)", await page.locator(".neal-market-price").inner_text())[1])
    values = re.search(
        r"Cash \$(-?[\d.]+) · Shares (\d+) · Realized profit \$(-?[\d.]+)",
        await page.locator(".neal-market-account").inner_text(),
    )
    return price, float(values[1]), int(values[2]), float(values[3])


@pytest.mark.live
async def test_market_continuous_prices_inventory_reset_and_profitable_trades(motion_page):
    page = await motion_page(28)
    first = await market(page)
    assert first[1:] == (500.0, 0, 0.0)
    await press(page, "Verify")
    assert (await snapshot(page))["mistakes"] == 1
    await page.clock.run_for(500)
    assert (await market(page))[0] != first[0]
    await press(page, "Buy one share")
    assert (await market(page))[2] == 1
    await press(page, "Refresh challenge")
    assert (await market(page))[1:] == (500.0, 0, 0.0)
    # Reading the visible price and portfolio is an ordinary trading strategy.
    # Virtual timer advancement changes no price function or account variable.
    for _ in range(2000):
        await page.clock.run_for(500)
        price, cash, shares, profit = await market(page)
        if price <= 220:
            for _ in range(min(100, int(cash // price))):
                await press(page, "Buy one share")
        elif price >= 365:
            for _ in range(shares):
                await press(page, "Sell one share")
        _, cash, _, profit = await market(page)
        if cash >= 2500 or profit >= 2500:
            break
    else:
        pytest.fail("Visible-price trading strategy did not reach the cash/profit target")
    await press(page, "Verify")
    assert (await snapshot(page))["status"] == "success"


async def visible_ball_cup(page):
    for index in range(3):
        if await page.locator(".neal-cup-ball").nth(index).is_visible():
            return index + 1
    return None


@pytest.mark.live
async def test_cups_three_real_shuffle_rounds_wrong_guess_and_refresh(motion_page):
    page = await motion_page(35)
    await page.clock.run_for(600)
    ball = await visible_ball_cup(page)
    assert ball in (1, 2, 3)
    await press(page, "Verify")
    assert (await snapshot(page))["mistakes"] == 1
    initial = await page.locator(".neal-cup").evaluate_all("nodes=>nodes.map(n=>n.style.transform)")
    await page.clock.run_for(6700)
    assert await page.locator(".neal-cups-board").get_attribute("data-phase") == "choose"
    assert await visible_ball_cup(page) is None
    assert (
        await page.locator(".neal-cup").evaluate_all("nodes=>nodes.map(n=>n.style.transform)")
        != initial
    )
    await press(page, f"Cup {ball % 3 + 1}")
    assert (await snapshot(page))["mistakes"] == 2
    await page.clock.run_for(3100)
    assert await visible_ball_cup(page) == ball
    await press(page, "Refresh challenge")
    assert (await snapshot(page))["progress"] == 0
    for round_number, duration in enumerate([6800, 6600, 10000], start=1):
        await page.clock.run_for(duration)
        assert await page.locator(".neal-cups-board").get_attribute("data-phase") == "choose"
        await press(page, f"Cup {ball}")
        assert (await snapshot(page))["progress"] == round_number
    await press(page, "Verify")
    assert (await snapshot(page))["status"] == "success"


@pytest.mark.live
async def test_grave_brush_distance_bouquet_candle_delayed_gate_and_noop_refresh(motion_page):
    page = await motion_page(41)
    assert await page.get_by_role("button", name="Verify", exact=True).is_disabled()
    await press(page, "Light candle")
    for number in range(1, 5):
        await press(page, f"Pick flower {number}")
    assert (await snapshot(page))["progress"] == 4
    await press(page, "Refresh challenge")
    assert (await snapshot(page))["progress"] == 4
    assert await page.get_by_role("button", name="Verify", exact=True).is_disabled()
    await press(page, "Clean the gravestone")
    brush = await page.locator(".neal-mourn-brush").bounding_box()
    assert brush
    x, y = brush["x"] + 35, brush["y"] + 30
    await page.mouse.move(x, y)
    await page.mouse.down()
    # A short motion alone is not enough, and returning adds actual distance.
    await page.mouse.move(x + 10, y - 10)
    await page.mouse.up()
    assert "is clean" not in await page.locator(".neal-mourn-status").inner_text()
    brush = await page.locator(".neal-mourn-brush").bounding_box()
    x, y = brush["x"] + 35, brush["y"] + 30
    await page.mouse.move(x, y)
    await page.mouse.down()
    for index in range(12):
        await page.mouse.move(
            x + (170 if index % 2 == 0 else -50), y - 180 + (index % 3) * 35, steps=4
        )
    await page.mouse.up()
    assert "is clean" in await page.locator(".neal-mourn-status").inner_text()
    await press(page, "Back to grave")
    await page.clock.run_for(1100)
    assert await page.get_by_role("button", name="Verify", exact=True).is_disabled()
    box = await page.locator(".neal-mourn-bouquet").bounding_box()
    assert box
    await page.mouse.move(box["x"] + 32, box["y"] + 32)
    await page.mouse.down()
    await page.mouse.move(box["x"] - 68, box["y"] - 68, steps=10)
    await page.mouse.up()
    assert "Remembering" in await page.locator(".neal-mourn-status").inner_text()
    await page.clock.run_for(3499)
    assert await page.get_by_role("button", name="Verify", exact=True).is_disabled()
    await page.clock.run_for(1)
    assert await page.get_by_role("button", name="Verify", exact=True).is_enabled()
    await press(page, "Verify")
    assert (await snapshot(page))["status"] == "success"


@pytest.mark.live
@pytest.mark.parametrize("level", [15, 26])
async def test_parking_held_physics_failure_collision_and_refresh(motion_page, level):
    page = await motion_page(level)
    initial = await page.locator(".neal-parking-car").get_attribute("style")
    await press(page, "Verify")
    assert (await snapshot(page))["mistakes"] == 1
    await page.keyboard.down("ArrowUp")
    await page.clock.run_for(1600)
    await page.keyboard.up("ArrowUp")
    assert await page.locator(".neal-parking-car").get_attribute("style") != initial
    await press(page, "Refresh challenge")
    assert await page.locator(".neal-parking-car").get_attribute("style") == initial
    assert (await snapshot(page))["progress"] == 0
    # Holding forward reaches a real obstacle, followed by a current-stage reset.
    await page.keyboard.down("ArrowUp")
    for _ in range(120):
        await page.clock.run_for(100)
        if await page.locator(".neal-parking-board").get_attribute("data-phase") == "collision":
            break
    await page.keyboard.up("ArrowUp")
    assert await page.locator(".neal-parking-board").get_attribute("data-phase") == "collision"
    await page.clock.run_for(2000)
    assert await page.locator(".neal-parking-board").get_attribute("data-phase") == "driving"
    assert await page.locator(".neal-parking-car").get_attribute("style") == initial


async def car_pose(page):
    """Read only the position and rotation actually rendered on the car."""
    return await page.locator(".neal-parking-car").evaluate(
        "n=>[parseFloat(n.style.left),parseFloat(n.style.top),parseFloat(n.style.transform.slice(7))]"
    )


async def front_wheel_angles(page):
    """Read rendered wheel rotations, not the parking evaluator's steering state."""
    return await page.locator(".neal-parking-front-wheel").evaluate_all(
        "nodes=>nodes.map(n=>{const m=new DOMMatrix(getComputedStyle(n).transform);"
        "return Math.atan2(m.b,m.a)})"
    )


@pytest.mark.live
@pytest.mark.parametrize("level", [15, 26])
async def test_parking_stationary_front_wheels_show_steering_and_release(
    motion_page, level, tmp_path
):
    page = await motion_page(level)
    pose = await car_pose(page)
    wheels = page.locator(".neal-parking-front-wheel")
    assert await wheels.count() == 2
    geometry = await wheels.evaluate_all(
        "nodes=>nodes.map(n=>{const s=getComputedStyle(n),p=n.parentElement;"
        "return {w:n.offsetWidth,h:n.offsetHeight,"
        "x:parseFloat(s.left)+p.clientLeft+n.offsetWidth/2,"
        "y:parseFloat(s.top)+p.clientTop+n.offsetHeight/2}})"
    )
    for index, wheel in enumerate(geometry):
        assert (wheel["w"], wheel["h"]) == (16, 6)
        assert wheel["x"] == pytest.approx(63, abs=0.02)
        assert await wheels.nth(index).is_visible()
    assert sorted(wheel["y"] for wheel in geometry) == pytest.approx(
        [25 - 50 / 3, 25 + 50 / 3], abs=0.02
    )
    clip = await page.locator(".neal-parking-car").bounding_box()
    assert clip
    neutral = Image.open(
        BytesIO(await page.screenshot(clip=clip, path=str(tmp_path / "neutral.png")))
    ).convert("RGB")
    # Actual rendered wheel-center pixels must be dark, not merely present in DOM.
    for index in range(2):
        box = await wheels.nth(index).bounding_box()
        assert box
        pixel = neutral.getpixel(
            (
                round(box["x"] + box["width"] / 2 - clip["x"]),
                round(box["y"] + box["height"] / 2 - clip["y"]),
            )
        )
        assert max(pixel) < 64, pixel
    for key, direction in [("ArrowLeft", -1), ("ArrowRight", 1)]:
        await page.keyboard.down(key)
        await page.clock.run_for(320)
        assert await car_pose(page) == pose
        assert await front_wheel_angles(page) == pytest.approx([direction * math.pi / 4] * 2)
        turned = Image.open(
            BytesIO(await page.screenshot(clip=clip, path=str(tmp_path / f"{key}.png")))
        ).convert("RGB")
        assert ImageChops.difference(neutral, turned).getbbox() is not None
        await page.keyboard.up(key)
        await page.clock.run_for(400)
        assert await car_pose(page) == pose
        assert await front_wheel_angles(page) == pytest.approx([0, 0])
        centered = Image.open(
            BytesIO(await page.screenshot(clip=clip, path=str(tmp_path / f"{key}-released.png")))
        ).convert("RGB")
        assert ImageChops.difference(neutral, centered).getbbox() is None


@pytest.mark.live
@pytest.mark.parametrize("level", [15, 26])
async def test_parking_refresh_and_collision_reset_preserve_held_key(motion_page, level):
    page = await motion_page(level)
    initial = await car_pose(page)
    await page.keyboard.down("ArrowUp")
    await page.clock.run_for(200)
    assert (await car_pose(page))[1] < initial[1]
    await press(page, "Refresh challenge")
    assert await car_pose(page) == initial
    # No second keydown: the physical key remains held across manual Refresh.
    await page.clock.run_for(200)
    assert (await car_pose(page))[1] < initial[1]
    await page.keyboard.up("ArrowUp")
    await press(page, "Refresh challenge")
    await page.keyboard.down("ArrowUp")
    for _ in range(120):
        await page.clock.run_for(100)
        if await page.locator(".neal-parking-board").get_attribute("data-phase") == "collision":
            break
    else:
        pytest.fail("Forward driving did not reach the existing obstacle")
    assert (await snapshot(page))["mistakes"] == 1
    # Keep the key held through the genuine collision timer, then observe motion.
    await page.clock.run_for(2200)
    assert await page.locator(".neal-parking-board").get_attribute("data-phase") == "driving"
    after_reset = await car_pose(page)
    assert after_reset[1] < initial[1]
    await page.clock.run_for(200)
    assert (await car_pose(page))[1] < after_reset[1]
    await page.keyboard.up("ArrowUp")


class ParkingDriver:
    def __init__(self, page):
        self.page = page
        self.held = set()

    async def advance(self, keys, ms=48):
        requested = set(keys)
        for key in self.held - requested:
            await self.page.keyboard.up(key)
        for key in requested - self.held:
            await self.page.keyboard.down(key)
        self.held = requested
        await self.page.clock.run_for(ms)
        pose = await car_pose(self.page)
        assert (
            await self.page.locator(".neal-parking-board").get_attribute("data-phase")
            != "collision"
        ), pose
        return pose

    async def stop(self):
        for key in self.held:
            await self.page.keyboard.up(key)
        self.held.clear()

    async def straight(self, axis, boundary, heading):
        for _ in range(250):
            pose = await car_pose(self.page)
            coordinate = pose[axis]
            if (math.cos(heading) if axis == 0 else math.sin(heading)) * (
                boundary - coordinate
            ) <= 0:
                return pose
            error = (heading - pose[2] + math.pi) % math.tau - math.pi
            steer = "d" if error > 0.025 else "a" if error < -0.025 else ""
            await self.advance("w" + steer, 16 if boundary < 20 else 32)
        pytest.fail(f"Driving segment exceeded budget: {await car_pose(self.page)}")

    async def turn(self, heading, direction):
        for _ in range(180):
            pose = await car_pose(self.page)
            error = (heading - pose[2] + math.pi) % math.tau - math.pi
            if (direction == "d" and error < 0.15) or (direction == "a" and error > -0.15):
                # Steering has to return to center; turning is not instantaneous.
                for _ in range(6):
                    await self.advance("w", 32)
                return await car_pose(self.page)
            await self.advance("w" + direction, 32)
        pytest.fail("Steering did not reach heading")


@pytest.mark.live
@pytest.mark.parametrize("level", [15, 26])
async def test_parking_stage_transition_preserves_held_steering(motion_page, level):
    """Synthetic-clock GUI regression, not a screenshot-only model success."""
    page = await motion_page(level)
    driver = ParkingDriver(page)
    if level == 15:
        await driver.straight(1, 164, -math.pi / 2)
        await driver.turn(0, "d")
        await driver.straight(0, 300, 0)
    else:
        await driver.straight(1, 260, -math.pi / 2)
        await driver.turn(-math.pi, "a")
        await driver.straight(0, 138, -math.pi)
        await driver.turn(-math.pi / 2, "d")
        await driver.straight(1, 13, -math.pi / 2)
    await driver.stop()
    await page.keyboard.down("ArrowLeft")
    await press(page, "Verify")
    assert (await snapshot(page))["progress"] == 1
    assert (await snapshot(page))["status"] == "in_progress"
    initial = await car_pose(page)
    assert initial == [10, 50 if level == 15 else 300, 0]
    await page.clock.run_for(160)
    assert await car_pose(page) == initial
    angles = await front_wheel_angles(page)
    assert len(angles) == 2 and all(angle < -0.3 for angle in angles)
    await page.keyboard.up("ArrowLeft")
    await page.clock.run_for(400)
    assert await front_wheel_angles(page) == pytest.approx([0, 0])


@pytest.mark.live
async def test_motion_uses_wall_clock_without_an_input_or_test_clock(motion_page):
    page = await motion_page(15, clock=False)
    before = await car_pose(page)
    await page.keyboard.down("ArrowUp")
    await asyncio.sleep(0.3)
    after = await car_pose(page)
    await page.keyboard.up("ArrowUp")
    assert after[1] < before[1]
    # The price process also continues during model/tool idle time.
    market_page = await motion_page(28, clock=False)
    price = (await market(market_page))[0]
    observed = {price}
    for _ in range(3):
        await asyncio.sleep(0.55)
        observed.add((await market(market_page))[0])
    assert len(observed) > 1


@pytest.mark.live
async def test_parking_level15_both_stages_by_real_driving(motion_page):
    """DOM-guided trusted keyboard input with a test clock, not model gameplay."""
    page = await motion_page(15)
    driver = ParkingDriver(page)
    await driver.straight(1, 164, -math.pi / 2)
    await driver.turn(0, "d")
    await driver.straight(0, 300, 0)
    await driver.stop()
    pose = await car_pose(page)
    assert 295 <= pose[0] <= 315 and 102 <= pose[1] <= 112, pose
    await press(page, "Verify")
    assert (await snapshot(page))["progress"] == 1
    assert (await snapshot(page))["status"] == "in_progress"
    await page.clock.run_for(2200)  # Let the visible cross-traffic clear the entry.
    await driver.straight(0, 120, 0)
    await driver.turn(math.pi / 2, "d")
    await driver.straight(1, 264, math.pi / 2)
    await driver.turn(0, "a")
    await driver.straight(0, 285, 0)
    await driver.stop()
    pose = await car_pose(page)
    assert 275 <= pose[0] <= 295 and 295 <= pose[1] <= 305, pose
    await press(page, "Verify")
    assert (await snapshot(page))["status"] == "success"


@pytest.mark.live
async def test_parking_level26_both_stages_by_real_driving(motion_page):
    """DOM-guided trusted keyboard input with a test clock, not model gameplay."""
    page = await motion_page(26)
    driver = ParkingDriver(page)
    await driver.straight(1, 260, -math.pi / 2)
    await driver.turn(-math.pi, "a")
    await driver.straight(0, 138, -math.pi)
    await driver.turn(-math.pi / 2, "d")
    await driver.straight(1, 13, -math.pi / 2)
    await driver.stop()
    pose = await car_pose(page)
    assert 75 <= pose[0] <= 105 and 0 <= pose[1] <= 15, pose
    await press(page, "Verify")
    assert (await snapshot(page))["progress"] == 1
    # Current-stage refresh must not return to the first parking layout.
    await press(page, "Refresh challenge")
    assert (await snapshot(page))["progress"] == 1
    assert (await car_pose(page))[:2] == [10, 300]
    # Travel around the end of the long horizontal wall before heading north.
    await page.clock.run_for(2200)
    await driver.straight(0, 250, 0)
    await driver.turn(-math.pi / 2, "a")
    await driver.straight(1, 170, -math.pi / 2)
    await driver.turn(-math.pi, "a")
    await driver.straight(0, 170, -math.pi)
    await driver.turn(-math.pi / 2, "d")
    await driver.straight(1, 14.9, -math.pi / 2)
    await driver.stop()
    pose = await car_pose(page)
    assert 75 <= pose[0] <= 105 and 0 <= pose[1] <= 15, pose
    await press(page, "Verify")
    assert (await snapshot(page))["status"] == "success"
