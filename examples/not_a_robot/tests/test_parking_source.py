"""Trusted-input parking regressions derived from source module 511.

The source stores each keyboard key separately and ORs aliases with virtual
controls in updateCar. These tests use a paused test clock and visible DOM
geometry as privileged oracles, not screenshot-only model evaluation. They do
not assign car state, synthesize input events, or inject a completion result.
"""

from __future__ import annotations

import math

import pytest

from examples.not_a_robot.tests.test_motion_expansion import car_pose, front_wheel_angles
from examples.not_a_robot.tests.test_motion_expansion import motion_page as motion_page


@pytest.mark.live
@pytest.mark.parametrize("level", [15, 26])
@pytest.mark.parametrize("released", ["a", "ArrowLeft"])
async def test_steering_alias_release_preserves_other_held_key(
    motion_page, level, released, tmp_path
):
    page = await motion_page(level)
    initial = await car_pose(page)
    try:
        await page.keyboard.down("a")
        await page.keyboard.down("ArrowLeft")
        await page.clock.run_for(320)
        assert await front_wheel_angles(page) == pytest.approx([-math.pi / 4] * 2)
        await page.screenshot(path=str(tmp_path / "both-keys-held.png"))

        await page.keyboard.up(released)
        await page.clock.run_for(400)
        await page.screenshot(path=str(tmp_path / "one-key-still-held.png"))
        assert await car_pose(page) == initial
        assert await front_wheel_angles(page) == pytest.approx([-math.pi / 4] * 2)

        remaining = "ArrowLeft" if released == "a" else "a"
        await page.keyboard.up(remaining)
        await page.clock.run_for(400)
        assert await car_pose(page) == initial
        assert await front_wheel_angles(page) == pytest.approx([0, 0])
    finally:
        await page.keyboard.up("a")
        await page.keyboard.up("ArrowLeft")


@pytest.mark.live
@pytest.mark.parametrize("level", [15, 26])
@pytest.mark.parametrize("released", ["keyboard", "pointer"])
async def test_steering_keyboard_and_pointer_have_independent_release(
    motion_page, level, released, tmp_path
):
    page = await motion_page(level)
    initial = await car_pose(page)
    box = await page.get_by_role("button", name="Steer left", exact=True).bounding_box()
    assert box
    try:
        await page.keyboard.down("a")
        await page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        await page.mouse.down()
        await page.clock.run_for(320)
        assert await front_wheel_angles(page) == pytest.approx([-math.pi / 4] * 2)

        if released == "keyboard":
            await page.keyboard.up("a")
        else:
            await page.mouse.up()
        await page.clock.run_for(400)
        await page.screenshot(path=str(tmp_path / "one-input-still-held.png"))
        assert await car_pose(page) == initial
        assert await front_wheel_angles(page) == pytest.approx([-math.pi / 4] * 2)

        if released == "keyboard":
            await page.mouse.up()
        else:
            await page.keyboard.up("a")
        await page.clock.run_for(400)
        assert await front_wheel_angles(page) == pytest.approx([0, 0])
    finally:
        await page.keyboard.up("a")
        await page.mouse.up()


@pytest.mark.live
@pytest.mark.parametrize("level", [15, 26])
@pytest.mark.parametrize("released", ["w", "ArrowUp"])
async def test_throttle_alias_release_matches_one_continuously_held_key(
    motion_page, level, released, tmp_path
):
    reference = await motion_page(level)
    remaining = "ArrowUp" if released == "w" else "w"
    try:
        await reference.keyboard.down(remaining)
        await reference.clock.run_for(160)
        expected_before_release = await car_pose(reference)
        await reference.clock.run_for(320)
        expected = await car_pose(reference)
        await reference.screenshot(path=str(tmp_path / "single-key-reference.png"))
        assert (
            await reference.locator(".neal-parking-board").get_attribute("data-phase") == "driving"
        )
    finally:
        await reference.keyboard.up(remaining)

    # Finish the reference input before creating another page: focus/blur is
    # not part of the alias-release condition under test.
    page = await motion_page(level)
    initial = await car_pose(page)
    try:
        await page.keyboard.down("w")
        await page.keyboard.down("ArrowUp")
        await page.clock.run_for(160)
        assert await car_pose(page) == pytest.approx(expected_before_release)

        await page.keyboard.up(released)
        await page.clock.run_for(320)
        await page.screenshot(path=str(tmp_path / "throttle-alias-held.png"))
        pose = await car_pose(page)
        assert expected[1] < initial[1]
        assert pose == pytest.approx(expected, abs=1e-8)
        assert await page.locator(".neal-parking-board").get_attribute("data-phase") == "driving"
    finally:
        await page.keyboard.up("w")
        await page.keyboard.up("ArrowUp")
