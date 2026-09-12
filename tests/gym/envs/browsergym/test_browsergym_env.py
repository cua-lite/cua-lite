"""BrowserGymEnv fake lifecycle, stepping, metadata, and observation text tests."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("browsergym.core", reason="browsergym not installed")

from lite.core.tools import make_tool_call
from lite.core.tools.results import TOOL_RESULT_ERROR_SECTION_HEADER
from lite.core.tools.schemas import tool_schema_name
from lite.gym.envs.browsergym.main import (
    BrowserGymConfig,
    BrowserGymEnv,
    _tools_for_subsets,
)
from tests.gym.envs.browsergym._support import (
    _BROWSERGYM_T2_MODE_CONFIGS,
    _decode_png_size,
    _fake_bgym_step_obs,
    _make_fake,
    _make_t2_browsergym_fake,
    _png_bytes,
)

# ---------------------------------------------------------------------------
# BrowserGymEnv: fake env lifecycle + obs builders
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fake_reset_returns_screenshot():
    env = _make_fake()
    try:
        obs = await env.reset()
        assert obs.image
        raw = obs.image
        assert raw[:4] == b"\x89PNG"
        # ``reset`` returns a bare ``LiteEnvObservation`` — reward/terminated/
        # truncated belong to ``LiteEnvStepResult`` and are asserted by the step
        # tests below.
        assert obs.text is not None
    finally:
        await env.close()


def test_encode_screenshot_paints_env_tracked_cursor(monkeypatch):
    env = _make_fake()
    env._cursor_x = 12.4
    env._cursor_y = 56.6
    calls: list[tuple[int, int]] = []

    def _paint(png: bytes, x: int, y: int) -> bytes:
        assert png.startswith(b"\x89PNG")
        calls.append((x, y))
        return b"painted"

    monkeypatch.setattr("lite.gym.envs.browsergym.main.overlay_cursor_px", _paint)

    obs = {"screenshot": _png_bytes(), "extra_element_properties": None}

    # Writing the tracked coordinate is NOT enough to license a paint: until a
    # real pointer move established it, the position is unknown and the frame
    # must stay raw.
    assert env._encode_screenshot(obs).startswith(b"\x89PNG")
    assert calls == []

    # A coord action really does move the Playwright pointer there.
    assert env._to_bgym_code("mouse_move", {"coordinate": [10, 79]}) is not None
    env._cursor_x = 12.4
    env._cursor_y = 56.6

    assert env._encode_screenshot(obs) == b"painted"
    assert calls == [(12, 57)]


def _sprite_box_pixels(png: bytes, x: int, y: int) -> list[tuple[int, int, int]]:
    """RGB pixels of the sprite-sized (16x24) box whose top-left is the tip."""
    from PIL import Image

    img = Image.open(io.BytesIO(png)).convert("RGB")
    w, h = img.size
    return [
        img.getpixel((px, py)) for px in range(x, min(x + 16, w)) for py in range(y, min(y + 24, h))
    ]


@pytest.mark.asyncio
async def test_reset_frame_draws_no_cursor_at_the_origin():
    """PIXEL-level, no mock: the reset frame must NOT carry a cursor at (0, 0).

    (0, 0) is the "never moved" sentinel, not a position any pointer was ever
    observed at. Painting a high-contrast arrow there puts a false visual anchor
    over the top-left of the page content on the FIRST frame of every episode,
    which measurably shifts a VL model's normalized output. The pointer is
    parked at the viewport centre instead — and really moved there — so that is
    where the sprite belongs.
    """
    env = _make_fake()  # 500x320 viewport, cursor on, coord subset
    obs = await env.reset()

    backdrop = (200, 200, 200)
    assert _sprite_box_pixels(obs.image, 0, 0) == [backdrop] * (16 * 24)

    centre = _sprite_box_pixels(obs.image, 250, 160)
    assert any(px != backdrop for px in centre), "no cursor painted at the centre"


@pytest.mark.asyncio
async def test_frame_before_reset_composites_nothing_anywhere():
    """A constructed-but-not-reset env has established no pointer position, so
    the guard must be False and NOTHING may be composited — not at the origin,
    not at the centre. This is the ``_cursor_position_known`` lie in its purest
    form: "rendering is enabled" must never be read as "position is known"."""
    env = _make_fake()

    assert env._cursor_position_known is False
    raw = _png_bytes(500, 320, color=(200, 200, 200))

    assert env._overlay_cursor(raw) == raw

    await env.reset()
    assert env._cursor_position_known is True


@pytest.mark.asyncio
async def test_position_is_not_known_until_the_parking_move_actually_lands(monkeypatch):
    """The flag must track the MOVE, not the knob.

    ``_cursor_position_known = self._cursor_rendering_enabled`` at the TOP of
    reset is the original lie in its subtlest form: with the coordinate already
    defaulted to the centre the painted frame looks right, yet the flag claims
    the pointer is parked before ``page.mouse.move`` has been issued at all. If
    browser creation fails, no pointer was ever placed — the flag must stay
    False and the next frame must composite nothing.
    """
    from lite.gym.envs.browsergym import main as bg_main

    env = _make_fake()
    env._use_fake = False  # take the real reset path (browser creation)
    monkeypatch.setattr(bg_main, "_check_env_vars", lambda benchmark: None)
    monkeypatch.setattr(
        env,
        "_run_in_thread",
        AsyncMock(side_effect=RuntimeError("browser boom")),
    )

    with pytest.raises(RuntimeError, match="browser boom"):
        await env.reset()

    assert env._cursor_position_known is False
    raw = _png_bytes(500, 320, color=(200, 200, 200))
    assert env._overlay_cursor(raw) == raw


def test_encode_screenshot_respects_cursor_false(monkeypatch):
    env = _make_fake(cursor=False)
    calls: list[tuple[int, int]] = []

    def _paint(png: bytes, x: int, y: int) -> bytes:
        calls.append((x, y))
        return b"painted"

    monkeypatch.setattr("lite.gym.envs.browsergym.main.overlay_cursor_px", _paint)

    out = env._encode_screenshot(
        {
            "screenshot": _png_bytes(),
            "extra_element_properties": None,
        }
    )

    assert out.startswith(b"\x89PNG")
    assert calls == []


def test_native_bid_som_mode_does_not_overlay_stale_cursor(monkeypatch):
    env = _make_fake(
        benchmark="webarena",
        bgym_task_id="webarena.1",
        action_subsets=("webarena",),
        use_som=True,
        valid_actions=[],
        extra_tools=[],
    )
    env._cursor_x = 400
    env._cursor_y = 200
    calls: list[tuple[int, int]] = []

    def _paint(png: bytes, x: int, y: int) -> bytes:
        calls.append((x, y))
        return b"painted"

    monkeypatch.setattr("lite.gym.envs.browsergym.main.overlay_cursor_px", _paint)

    out = env._encode_screenshot(
        {
            "screenshot": _png_bytes(),
            "extra_element_properties": None,
        }
    )

    assert out.startswith(b"\x89PNG")
    assert env._cursor_rendering_enabled is False
    assert calls == []


def test_native_pointer_action_invalidates_cursor_until_next_coord_action(monkeypatch):
    env = _make_fake(action_subsets=("coord", "miniwob_all"))
    calls: list[tuple[int, int]] = []

    def _paint(png: bytes, x: int, y: int) -> bytes:
        calls.append((x, y))
        return b"painted"

    monkeypatch.setattr("lite.gym.envs.browsergym.main.overlay_cursor_px", _paint)

    assert env._cursor_rendering_enabled is True
    assert env._to_bgym_code("click", {"bid": "a47"}) == "click('a47')"
    assert env._cursor_position_known is False
    assert env._overlay_cursor(_png_bytes()).startswith(b"\x89PNG")
    assert calls == []

    assert env._to_bgym_code("mouse_move", {"coordinate": [200, 100]}) == "mouse_move(100.0, 32.0)"
    assert env._cursor_position_known is True
    assert env._overlay_cursor(_png_bytes()) == b"painted"
    assert calls == [(100, 32)]


@pytest.mark.asyncio
async def test_dead_backend_failfast_truncates():
    """L2: N CONSECUTIVE steps where the BrowserGym step
    returned no observation (last_obs is None = dead WA/VWA stack / miniwob)
    must TRUNCATE loudly, not grind to max_steps as a silent reward-0."""
    from unittest.mock import AsyncMock

    from lite.gym.envs.browsergym import isolation

    N = isolation._POOL_UNREACHABLE_FAILED_STEPS
    assert N >= 2
    env = _make_fake(
        max_steps=50, use_screenshot=False
    )  # text-only: isolate L2 counter from screenshot encoding
    try:
        await env.reset()
        env._execute_bgym_action = AsyncMock(return_value=None)  # dead backend
        last = None
        truncs = []
        for _ in range(N):
            last = await env.step(
                [
                    make_tool_call("click", {"coordinate": [10, 10]}),
                ]
            )
            truncs.append(last.truncated)
        assert truncs[:-1] == [False] * (N - 1), truncs
        assert truncs[-1] is True, "should truncate on the Nth consecutive no-obs step"
        assert last.info.get("pool_unreachable") is True
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_noop_steps_do_not_trip_dead_backend_failfast():
    """REGRESSION: a no-op step (wait/screenshot/cursor_position) never calls the
    BrowserGym backend, so its ``last_obs is None`` must NOT count toward the
    dead-backend fail-fast — a run of `wait`s on a HEALTHY env must not truncate.
    (The earlier version keyed off `last_obs is None` directly and would falsely
    truncate after N consecutive no-op steps.)"""
    from lite.gym.envs.browsergym import isolation

    N = isolation._POOL_UNREACHABLE_FAILED_STEPS
    env = _make_fake(max_steps=50, use_screenshot=False)
    try:
        await env.reset()
        any_trunc = False
        for _ in range(2 * N + 1):  # well past the threshold — must still not trip
            r = await env.step([make_tool_call("wait")])
            any_trunc = any_trunc or r.truncated
        assert not any_trunc, "no-op (wait) steps must not trip the dead-backend fail-fast"
    finally:
        await env.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("duration", [-1, "NaN", "Infinity", 31])
async def test_bad_wait_duration_returns_current_feedback_without_backend(duration):
    env = _make_fake(max_steps=50, use_screenshot=False)
    try:
        await env.reset()
        env._execute_bgym_action = AsyncMock()

        r = await env.step(
            [
                make_tool_call("wait", {"duration": duration}, call_id="call_wait"),
            ]
        )

        assert r.terminated is False
        assert r.truncated is False
        assert r.results[0].tool_call_id == "call_wait"
        assert r.results[0].error.startswith("invalid arguments for wait: wait.duration")
        assert r.results[0].metadata == {"is_error": True}
        env._execute_bgym_action.assert_not_awaited()
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_env_step_accepts_nested_tool_calls_and_returns_plural_images():
    env = _make_fake(max_steps=50)
    screenshot = _png_bytes(5, 5, (0, 0, 255))
    try:
        await env.reset()
        env._execute_bgym_action = AsyncMock(
            return_value=_fake_bgym_step_obs(screenshot=screenshot)
        )

        result = await env.step(
            [
                make_tool_call("click", {"coordinate": [10, 10]}, call_id="call_click"),
            ]
        )

        env._execute_bgym_action.assert_awaited_once()
        assert result.results[0].tool_call_id == "call_click"
        assert result.results[0].images[-1] == env._overlay_cursor(screenshot)
        assert result.results[0].error is None
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_action_batch_returns_one_frame_per_executed_action():
    """N executed actions → N frames, in action order, each its own capture.

    BrowserGym captures the page inside its own ``env.step``, so each executed
    action already owns a distinct frame; the step loop must stop discarding all
    but the last. Distinctness is asserted, not just the count: re-emitting one
    cached frame N times would satisfy the count while carrying no information.
    """
    from unittest.mock import AsyncMock

    env = _make_fake(max_steps=50, cursor=False)
    # Distinct SIZES, so ORDER survives whatever re-encoding the render does.
    frames = [_png_bytes(w, w, (255, 0, 0)) for w in (4, 5, 6)]
    try:
        await env.reset()
        env._execute_bgym_action = AsyncMock(
            side_effect=[_fake_bgym_step_obs(screenshot=f) for f in frames]
        )

        result = await env.step(
            [
                make_tool_call(
                    "computer",
                    {
                        "actions": [
                            {"action": "click", "coordinate": [10, 10]},
                            {"action": "click", "coordinate": [20, 20]},
                            {"action": "click", "coordinate": [30, 30]},
                        ]
                    },
                    call_id="call_gui",
                ),
            ]
        )

        images = result.results[0].images
        assert env._execute_bgym_action.await_count == 3
        assert len(images) == 3
        assert len(set(images)) == 3
        assert [_decode_png_size(png) for png in images] == [(4, 4), (5, 5), (6, 6)]
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_read_only_actions_earn_a_frame_read_off_the_page():
    """``screenshot``/``cursor_position``/``wait`` reach no backend, so they own
    no obs -- but they DID execute, and the frame count must never depend on
    WHAT the actions were. Their frame is read straight off the Playwright page,
    which is why it needs no ``env.step`` and moves no step counter.
    """
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    env = _make_fake(max_steps=50, cursor=False)
    try:
        await env.reset()
        env._execute_bgym_action = AsyncMock()
        # Only the None-guard reads ``_env`` on this path; the capture itself is
        # stubbed, since a fake env has no Playwright page to read.
        env._env = SimpleNamespace()
        env._take_screenshot = AsyncMock(
            side_effect=[_png_bytes(w, w, (0, 0, 255)) for w in (7, 8, 9)]
        )

        result = await env.step(
            [
                make_tool_call(
                    "computer",
                    {
                        "actions": [
                            {"action": "screenshot"},
                            {"action": "cursor_position"},
                            {"action": "wait", "duration": 1},
                        ]
                    },
                    call_id="call_gui",
                ),
            ]
        )

        env._execute_bgym_action.assert_not_awaited()
        assert env._take_screenshot.await_count == 3
        assert [_decode_png_size(png) for png in result.results[0].images] == [
            (7, 7),
            (8, 8),
            (9, 9),
        ]
    finally:
        env._env = None
        await env.close()


@pytest.mark.asyncio
async def test_step_that_executed_nothing_returns_one_frame():
    """Every call REJECTED before execution: no action earned a frame -- there
    is no screen state after an action that never ran. The turn still owes
    exactly one current observation, not N duplicates."""
    from unittest.mock import AsyncMock

    env = _make_fake(max_steps=50, cursor=False)
    try:
        await env.reset()
        env._execute_bgym_action = AsyncMock()

        result = await env.step(
            [
                make_tool_call(
                    "computer",
                    {
                        "actions": [
                            {"action": "no_such_action"},
                            {"action": "another_bogus_action"},
                        ]
                    },
                    call_id="call_gui",
                ),
            ]
        )

        env._execute_bgym_action.assert_not_awaited()
        assert len(result.results[0].images) == 1
    finally:
        await env.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "match"),
    [
        (
            {"name": "wait", "arguments": {}},
            "env.step expects canonical Lite tool calls",
        ),
        (
            {"call_id": "call_wait", "name": "wait", "arguments": {}},
            "env.step expects canonical Lite tool calls",
        ),
        (
            {"id": "call_wait", "name": "wait", "arguments": {}},
            r"action has noncanonical outer keys \['arguments', 'name'\]",
        ),
    ],
    ids=["bare-agent-wire", "old-call-id-bare", "id-plus-bare"],
)
async def test_env_step_rejects_bare_runtime_tool_call_shapes(
    action: dict[str, Any],
    match: str,
):
    env = _make_fake(max_steps=50, use_screenshot=False)
    try:
        await env.reset()
        env._execute_bgym_action = AsyncMock()

        with pytest.raises(TypeError, match=match):
            await env.step([action])

        env._execute_bgym_action.assert_not_awaited()
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_max_step_truncation_pairs_current_result():
    env = _make_fake(max_steps=1, use_screenshot=False)
    try:
        await env.reset()

        async def _fake_execute(_code: str) -> dict[str, Any]:
            return {
                "reward": 0.0,
                "last_action_error": "",
                "open_pages_urls": ("http://miniwob/",),
                "open_pages_titles": ("miniwob",),
                "active_page_index": [0],
            }

        env._execute_bgym_action = AsyncMock(side_effect=_fake_execute)

        r = await env.step(
            [
                make_tool_call("click", {"coordinate": [10, 10]}, call_id="call_click"),
            ]
        )

        assert r.terminated is False
        assert r.truncated is True
        assert r.results[0].tool_call_id == "call_click"
        assert r.results[0].text
        assert r.results[0].images == []
        assert r.results[0].error is None
        env._execute_bgym_action.assert_awaited_once()
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_terminal_tool_does_not_pair_later_unexecuted_sibling():
    env = _make_fake(max_steps=50, use_screenshot=False, extra_tools=["terminate"])
    try:
        await env.reset()

        async def _fake_execute(_code: str) -> dict[str, Any]:
            env._terminated = True
            return {
                "reward": 1.0,
                "last_action_error": "",
                "open_pages_urls": ("http://miniwob/",),
                "open_pages_titles": ("miniwob",),
                "active_page_index": [0],
            }

        env._execute_bgym_action = AsyncMock(side_effect=_fake_execute)

        r = await env.step(
            [
                make_tool_call("terminate", {"status": "success"}, call_id="call_done"),
                make_tool_call("click", {"coordinate": [10, 10]}, call_id="call_click"),
            ]
        )

        assert r.terminated is True
        # A terminal call gets NO tool result: it ended the episode, so there is
        # no next decision for an observation to inform, and
        # ``devs/migration/verify.py`` refuses a tool result for a terminal call.
        # The point this test defends survives: the later UNEXECUTED sibling
        # (``call_click``) is still not paired either.
        assert r.results == []
        env._execute_bgym_action.assert_awaited_once()
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_model_action_error_does_not_drop_later_valid_sibling():
    env = _make_fake(max_steps=50, use_screenshot=False)
    try:
        await env.reset()

        async def _fake_execute(_code: str) -> dict[str, Any]:
            return {
                "reward": 0.0,
                "last_action_error": "",
                "open_pages_urls": ("http://miniwob/",),
                "open_pages_titles": ("miniwob",),
                "active_page_index": [0],
            }

        env._execute_bgym_action = AsyncMock(side_effect=_fake_execute)

        r = await env.step(
            [
                make_tool_call("drag", {"start_coordinate": [0, 0]}, call_id="call_bad_drag"),
                make_tool_call("click", {"coordinate": [10, 10]}, call_id="call_click"),
            ]
        )

        assert r.terminated is False
        assert r.truncated is False
        assert [result.tool_call_id for result in r.results] == [
            "call_bad_drag",
            "call_click",
        ]
        assert r.results[0].error
        assert r.results[0].error.startswith("invalid arguments for drag:")
        assert r.results[0].metadata == {"is_error": True}
        assert r.results[1].error is None
        assert r.results[1].text
        env._execute_bgym_action.assert_awaited_once()
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_dead_backend_counter_resets_on_real_obs():
    """A logical action error (obs WITH last_action_error, last_obs NOT None) or
    a recovered backend resets the consecutive-failure counter, so a transient
    blip never trips the fail-fast."""
    from unittest.mock import AsyncMock

    from lite.gym.envs.browsergym import isolation

    N = isolation._POOL_UNREACHABLE_FAILED_STEPS
    env = _make_fake(max_steps=50, use_screenshot=False)
    try:
        await env.reset()
        good = {"reward": 0, "screenshot": None, "last_action_error": ""}
        # N-1 dead, one real obs (resets), then N-1 dead → never N-in-a-row.
        seq = [None] * (N - 1) + [good] + [None] * (N - 1)
        env._execute_bgym_action = AsyncMock(side_effect=seq)
        any_trunc = False
        for _ in range(len(seq)):
            r = await env.step(
                [
                    make_tool_call("click", {"coordinate": [10, 10]}),
                ]
            )
            any_trunc = any_trunc or r.truncated
        assert not any_trunc, "a recovered/blip backend must NOT trip the fail-fast"
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_fake_metadata():
    env = _make_fake()
    try:
        await env.reset()
        m = env.metadata
        assert m.platform == "browser"
        assert m.task_type == "use"
        assert m.others["benchmark"] == "miniwob"
        # Identity is framework-injected by gym.make; direct construction
        # carries NO task_id (same-source contract, metadata contract).
        assert "task_id" not in m.others
        assert m.others["bgym_task_id"] == "miniwob.click-dialog"
        assert m.others["viewport"] == (500, 320)
        # miniwob has no WA/VWA task facts: the builder must not attach them.
        for key in ("sites", "llm_as_a_judge", "mutating", "depends_on", "conflict_keys"):
            assert key not in m.others, key
        # valid_actions serves the BrowserGymConfig default (None = no filter).
        assert m.valid_actions is None
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_fake_metadata_extra_tools_default():
    """The shipped default (``extra_tools: null``) exposes the WHOLE catalog.

    Default subsets are coord/vision mode, so the browser platform uses the
    native desktop-coordinate computer_use surface. BrowserGym's chat/infeas
    actions surface as canonical ``response``/``terminate``, and nav is
    advertised as canonical extra_tools so WebGym SFT data transfers. The coord
    preset never surfaces.
    """
    env = _make_fake()
    try:
        await env.reset()
        names = {tool_schema_name(t) for t in env.metadata.extra_tool_schemas}
        assert {"goto", "back", "forward", "new_tab", "switch_tab", "close_tab"} <= names
        # WA/VWA information-seeking tasks ANSWER through send_msg_to_user; it is
        # surfaced under its canonical name. Suppressing it by default made every
        # such task structurally unanswerable.
        assert {"response", "terminate"} <= names
        assert not (names & {"send_msg_to_user", "report_infeasible"})
        assert not any(n.startswith(("mouse_", "keyboard_")) for n in names)
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_configless_registered_miniwob_metadata_matches_the_live_env():
    import lite.gym as gym
    from lite.gym.registry import _specs

    try:
        task_ids = gym.registry.task_ids("browsergym.miniwob")
    except Exception as exc:
        pytest.skip(f"browsergym.miniwob unavailable: {exc}")
    first = next((ids[0] for ids in task_ids.values() if ids), None)
    if first is None:
        pytest.skip("browsergym.miniwob registered no tasks")

    spec = _specs[f"browsergym.miniwob@{first}"]
    env = spec.entry_point(**spec.kwargs, use_fake=True)
    try:
        registered = gym.registry.task_metadata("browsergym.miniwob", first)
        assert registered is not None
        # Same-source contract: the registered copy is byte-identical to what
        # default construction produces — and under the shipped
        # ``extra_tools: null`` that is the whole action_subsets catalog, not [].
        assert {tool_schema_name(s) for s in registered.extra_tool_schemas} >= {
            "response",
            "terminate",
        }
        assert env.metadata.extra_tool_schemas == registered.extra_tool_schemas
    finally:
        await env.close()


def test_browsergym_catalog_declares_page_context_toggle():
    import lite.gym as gym

    try:
        supported = gym.registry.env_supported_kwargs("browsergym.miniwob")
    except Exception as exc:
        pytest.skip(f"browsergym.miniwob unavailable: {exc}")

    assert "include_page_context_text" in supported


def test_browsergym_default_disables_page_context_text():
    import lite.gym.envs.browsergym.main as m

    assert m.CFG.env_kwargs["include_page_context_text"] is False
    assert m._INCLUDE_PAGE_CONTEXT_TEXT is False
    assert BrowserGymConfig(
        bgym_task_id="miniwob.click-dialog",
        benchmark="miniwob",
    ).include_page_context_text is False


@pytest.mark.asyncio
async def test_metadata_overrides_via_action_subsets():
    """yaml-style override: action_subsets=['webarena'] should re-derive tools at
    metadata-time (no re-registration needed)."""
    env = _make_fake(
        action_subsets=("webarena",),
        extra_tools=["click", "fill", "terminate"],
    )
    try:
        await env.reset()
        names = {tool_schema_name(t) for t in env.metadata.extra_tool_schemas}
        assert "click" in names  # bid-style click
        assert "fill" in names
        assert "terminate" in names
        assert names == {"click", "fill", "terminate"}
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_text_bid_mode_metadata_suppresses_action_and_surfaces_bid_tools():
    env = _make_fake(
        use_screenshot=False,
        use_ax_tree=True,
        action_subsets=("webarena",),
        valid_actions=[],
        extra_tools=["click", "fill", "response", "terminate"],
    )
    try:
        await env.reset()
        assert env._config.use_screenshot is False
        assert env._config.use_ax_tree is True
        assert env.metadata.valid_actions == []
        assert [tool_schema_name(s) for s in env.metadata.extra_tool_schemas] == [
            "click",
            "fill",
            "response",
            "terminate",
        ]
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_som_bid_mode_keeps_dom_extraction_and_bid_tool_surface():
    env = _make_fake(
        use_som=True,
        skip_dom_extraction=False,
        action_subsets=("webarena",),
        valid_actions=[],
        extra_tools=["click"],
    )
    try:
        await env.reset()
        assert env._config.use_som is True
        assert env._config.skip_dom_extraction is False
        assert env.metadata.valid_actions == []
        assert [tool_schema_name(s) for s in env.metadata.extra_tool_schemas] == ["click"]
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_visualwebarena_mode_surfaces_upload_file_as_active_extra():
    env = _make_fake(
        action_subsets=("visualwebarena",),
        valid_actions=[],
        extra_tools=["upload_file"],
    )
    try:
        await env.reset()
        assert env.metadata.valid_actions == []
        assert [tool_schema_name(s) for s in env.metadata.extra_tool_schemas] == ["upload_file"]
    finally:
        await env.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", sorted(_BROWSERGYM_T2_MODE_CONFIGS))
async def test_t2_browsergym_modes_use_deterministic_error_carriers(mode: str):
    env = _make_t2_browsergym_fake(mode)
    expects_image = _BROWSERGYM_T2_MODE_CONFIGS[mode].get("use_screenshot", True)
    try:
        await env.reset()
        env._execute_bgym_action = AsyncMock()

        unknown = await env.step(
            [
                make_tool_call("foo", {}, call_id="call_unknown"),
            ]
        )
        assert unknown.results[0].tool_call_id == "call_unknown"
        assert unknown.results[0].error == "unknown tool: foo"
        assert unknown.results[0].text is None
        assert unknown.results[0].images == []
        assert unknown.results[0].metadata == {"is_error": True}

        inactive = await env.step(
            [
                make_tool_call("hover", {"bid": "a47"}, call_id="call_hover"),
            ]
        )
        assert inactive.results[0].tool_call_id == "call_hover"
        assert inactive.results[0].error == "hover is not available in this task."
        assert inactive.results[0].text
        assert "hover is not available in this task." not in inactive.results[0].text
        if expects_image:
            assert inactive.results[0].images[-1][:4] == b"\x89PNG"
        else:
            assert inactive.results[0].images == []
        assert inactive.results[0].metadata == {"is_error": True}
        env._execute_bgym_action.assert_not_awaited()
    finally:
        await env.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["webarena/som", "visualwebarena/som"])
async def test_t2_som_bid_tool_name_collision_keeps_coordinate_click_on_action_gate(
    mode: str,
):
    env = _make_t2_browsergym_fake(mode)
    try:
        await env.reset()
        env._execute_bgym_action = AsyncMock()

        result = await env.step(
            [
                make_tool_call("click", {"coordinate": [10, 10]}, call_id="call_click"),
            ]
        )

        assert result.results[0].tool_call_id == "call_click"
        assert result.results[0].error == (
            "invalid action: click; choose an available action for this task"
        )
        assert result.results[0].images[-1][:4] == b"\x89PNG"
        assert result.results[0].text
        assert "click is not available in this task." not in result.results[0].text
        assert result.results[0].metadata == {"is_error": True}
        env._execute_bgym_action.assert_not_awaited()
    finally:
        await env.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected_image"),
    [
        ("visualwebarena/mixed", False),
        ("visualwebarena/som", True),
    ],
)
async def test_t2_visualwebarena_upload_file_binds_result_to_current_payload(
    mode: str,
    expected_image: bool,
):
    env = _make_t2_browsergym_fake(mode)
    screenshot = _png_bytes(5, 5, (0, 0, 255))
    try:
        await env.reset()
        env._execute_bgym_action = AsyncMock(
            return_value=_fake_bgym_step_obs(
                screenshot=screenshot,
                url="http://visualwebarena.fake/",
            )
        )

        result = await env.step(
            [
                make_tool_call(
                    "upload_file",
                    {"bid": "a47", "file": ["receipt.pdf", "/tmp/photo.jpg"]},
                    call_id="call_upload",
                ),
            ]
        )

        assert result.terminated is False
        assert result.truncated is False
        env._execute_bgym_action.assert_awaited_once()
        assert env._execute_bgym_action.await_args.args[0] == (
            "upload_file('a47', ['receipt.pdf', '/tmp/photo.jpg'])"
        )
        assert len(result.results) == 1
        tool_result = result.results[0]
        assert tool_result.tool_call_id == "call_upload"
        assert tool_result.error is None
        assert tool_result.text
        if expected_image:
            assert tool_result.images[-1] == env._overlay_cursor(screenshot)
        else:
            assert tool_result.images == []
        assert (tool_result.metadata or {}).get("is_error") is not True
    finally:
        await env.close()


def test_t2_visualwebarena_goal_image_indices_stay_metadata_ordered_before_page():
    from lite.agents.extensions.browsergym.goal_image import (
        _goal_image_indices,
        _persist_goal_images,
        splice_goal_images,
    )

    message = {
        "role": "user",
        "content": [
            {"type": "image", "index": 2},
            {"type": "image", "index": 1},
            {"type": "image", "index": 0},
            {"type": "text", "text": "goal\n## AXTree:\nbody"},
            {"type": "metadata", "data": {"goal_image_indices": [2, 1]}},
        ],
    }

    assert _goal_image_indices([message]) == [2, 1]
    _persist_goal_images(message, [2, 1])

    image_indices = [part["index"] for part in message["content"] if part.get("type") == "image"]
    assert image_indices == [2, 1, 0]

    rendered = splice_goal_images(
        [message],
        [{"role": "user", "content": list(message["content"])}],
    )
    assert rendered[0]["content"][0]["text"].startswith("Task reference image")
    assert rendered[0]["content"][3]["text"] == "Current screenshot:"


def test_display_resolution_rejected():
    """browsergym does NOT accept display_resolution: its render size is task-fixed
    (we never pass a viewport override to gym.make), so viewport_width/height is
    only the coord-match constant. Accepting display_resolution would silently
    mismatch the real render → misclicks; it's a hard error instead."""
    config = BrowserGymConfig(bgym_task_id="miniwob.click-dialog", benchmark="miniwob")
    with pytest.raises(ValueError, match="does not accept display_resolution"):
        BrowserGymEnv(config=config, use_fake=True, display_resolution=[1920, 1080])


@pytest.mark.asyncio
async def test_extra_tools_empty_returns_empty():
    config = BrowserGymConfig(bgym_task_id="miniwob.click-dialog", benchmark="miniwob")
    env = BrowserGymEnv(config=config, use_fake=True, extra_tools=[])
    try:
        await env.reset()
        assert env.metadata.extra_tool_schemas == []
    finally:
        await env.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("response", {"text": "OK"}),
        ("terminate", {"status": "success"}),
    ],
)
async def test_inactive_terminal_tools_do_not_submit_on_direct_env(
    name: str,
    arguments: dict[str, Any],
):
    config = BrowserGymConfig(
        bgym_task_id="miniwob.click-dialog",
        benchmark="miniwob",
        use_screenshot=False,
    )
    env = BrowserGymEnv(config=config, use_fake=True, extra_tools=[])
    try:
        await env.reset()
        env._execute_bgym_action = AsyncMock()

        r = await env.step(
            [
                make_tool_call(name, arguments, call_id=f"call_{name}"),
            ]
        )

        assert r.terminated is False
        assert r.truncated is False
        assert r.results[0].tool_call_id == f"call_{name}"
        assert r.results[0].text
        assert TOOL_RESULT_ERROR_SECTION_HEADER not in r.results[0].text
        assert f"{name} is not available in this task." not in r.results[0].text
        assert r.results[0].error == f"{name} is not available in this task."
        assert r.results[0].metadata == {"is_error": True}
        env._execute_bgym_action.assert_not_awaited()
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_inactive_bid_tool_does_not_execute_on_direct_env():
    config = BrowserGymConfig(
        bgym_task_id="miniwob.click-dialog",
        benchmark="miniwob",
        action_subsets=["bid"],
        use_screenshot=False,
    )
    env = BrowserGymEnv(config=config, use_fake=True, extra_tools=[])
    try:
        await env.reset()
        env._execute_bgym_action = AsyncMock()

        r = await env.step(
            [
                make_tool_call("click", {"bid": "a47"}, call_id="call_bid"),
            ]
        )

        assert r.terminated is False
        assert r.truncated is False
        assert r.results[0].tool_call_id == "call_bid"
        assert r.results[0].error == "click is not available in this task."
        assert r.results[0].metadata == {"is_error": True}
        env._execute_bgym_action.assert_not_awaited()
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_unknown_foo_is_error_only_but_inactive_known_tool_keeps_current_carrier():
    config = BrowserGymConfig(
        bgym_task_id="miniwob.click-dialog",
        benchmark="miniwob",
        action_subsets=["bid"],
        use_screenshot=False,
    )
    env = BrowserGymEnv(config=config, use_fake=True, extra_tools=[])
    try:
        await env.reset()
        env._execute_bgym_action = AsyncMock()

        unknown = await env.step(
            [
                make_tool_call("foo", {}, call_id="call_foo"),
            ]
        )
        assert unknown.results[0].tool_call_id == "call_foo"
        assert unknown.results[0].error == "unknown tool: foo"
        assert unknown.results[0].text is None
        assert unknown.results[0].images == []
        assert unknown.results[0].metadata == {"is_error": True}

        inactive_known = await env.step(
            [
                make_tool_call("click", {"bid": "a47"}, call_id="call_bid"),
            ]
        )
        assert inactive_known.results[0].tool_call_id == "call_bid"
        assert inactive_known.results[0].error == "click is not available in this task."
        assert inactive_known.results[0].text
        assert inactive_known.results[0].metadata == {"is_error": True}
        env._execute_bgym_action.assert_not_awaited()
    finally:
        await env.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("tap", {"coordinate": [500, 500]}),
        ("pinch", {"coordinate": [500, 500], "direction": "in", "amount": 25}),
    ],
)
async def test_known_unsupported_mobile_action_keeps_current_carrier(
    name: str,
    arguments: dict[str, Any],
):
    config = BrowserGymConfig(
        bgym_task_id="miniwob.click-dialog",
        benchmark="miniwob",
        use_screenshot=False,
    )
    env = BrowserGymEnv(config=config, use_fake=True, extra_tools=[])
    try:
        await env.reset()
        env._execute_bgym_action = AsyncMock()

        r = await env.step(
            [
                make_tool_call(name, arguments, call_id="call_mobile"),
            ]
        )

        assert r.results[0].tool_call_id == "call_mobile"
        assert r.results[0].error == f"unsupported action: {name}"
        assert r.results[0].text
        assert "BrowserGym Fake" not in r.results[0].text
        assert "http://browsergym.fake/" not in r.results[0].text
        assert r.results[0].images == []
        assert r.results[0].metadata == {
            "open_pages_urls": ["http://browsergym.fake/"],
            "open_pages_titles": ["BrowserGym Fake"],
            "active_page_index": 0,
            "is_error": True,
        }
        env._execute_bgym_action.assert_not_awaited()
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_direct_valid_actions_empty_rejects_lite_action_batch():
    config = BrowserGymConfig(
        bgym_task_id="miniwob.click-dialog",
        benchmark="miniwob",
        use_screenshot=False,
        valid_actions=[],
    )
    env = BrowserGymEnv(config=config, use_fake=True)
    try:
        await env.reset()
        env._execute_bgym_action = AsyncMock()

        r = await env.step(
            [
                make_tool_call(
                    "computer",
                    {"actions": [{"action": "click", "coordinate": [500, 500]}]},
                    call_id="call_action",
                ),
            ]
        )

        assert r.terminated is False
        assert r.truncated is False
        assert r.results[0].tool_call_id == "call_action"
        assert r.results[0].text is not None
        assert r.results[0].error == (
            "invalid action: click; choose an available action for this task"
        )
        assert r.results[0].metadata == {"is_error": True}
        env._execute_bgym_action.assert_not_awaited()
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_valid_actions_empty_keeps_axtree_text_feedback():
    config = BrowserGymConfig(
        bgym_task_id="miniwob.click-dialog",
        benchmark="miniwob",
        use_screenshot=False,
        valid_actions=[],
    )
    env = BrowserGymEnv(config=config, use_fake=True)
    try:
        await env.reset()
        env._execute_bgym_action = AsyncMock()

        r = await env.step(
            [
                make_tool_call(
                    "computer",
                    {"actions": [{"action": "click", "coordinate": [500, 500]}]},
                    call_id="call_action",
                ),
            ]
        )

        assert r.results[0].tool_call_id == "call_action"
        assert r.results[0].error == (
            "invalid action: click; choose an available action for this task"
        )
        assert r.results[0].metadata == {"is_error": True}
        assert r.results[0].text
        assert r.results[0].images == []
        assert "invalid action: click" not in r.results[0].text
        env._execute_bgym_action.assert_not_awaited()
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_coordinate_error_preserves_image_text_and_error_carriers():
    config = BrowserGymConfig(
        bgym_task_id="miniwob.click-dialog",
        benchmark="miniwob",
        valid_actions=[],
    )
    env = BrowserGymEnv(config=config, use_fake=True)
    try:
        await env.reset()
        env._execute_bgym_action = AsyncMock()

        r = await env.step(
            [
                make_tool_call(
                    "computer",
                    {"actions": [{"action": "click", "coordinate": [10, 10]}]},
                    call_id="call_action",
                ),
            ]
        )

        assert r.results[0].tool_call_id == "call_action"
        assert r.results[0].images[-1][:4] == b"\x89PNG"
        assert r.results[0].text
        assert r.results[0].error == (
            "invalid action: click; choose an available action for this task"
        )
        assert "invalid action: click" not in r.results[0].text
        assert r.results[0].metadata == {"is_error": True}
        env._execute_bgym_action.assert_not_awaited()
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_backend_last_action_error_sets_tool_result_error():
    env = _make_fake(max_steps=50, use_screenshot=False)
    try:
        await env.reset()
        env._execute_bgym_action = AsyncMock(
            return_value={
                "reward": 0,
                "screenshot": None,
                "last_action_error": "timeout exceeded",
            }
        )

        r = await env.step(
            [
                make_tool_call(
                    "computer",
                    {"actions": [{"action": "click", "coordinate": [500, 500]}]},
                    call_id="call_action",
                ),
            ]
        )

        assert r.results[0].tool_call_id == "call_action"
        assert r.results[0].error == "click failed: execution failed"
        assert r.results[0].text
        assert TOOL_RESULT_ERROR_SECTION_HEADER not in r.results[0].text
        assert "timeout exceeded" not in r.results[0].text
        assert r.results[0].metadata == {"is_error": True}
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_page_context_disabled_keeps_backend_action_error_without_page_text():
    env = _make_fake(
        max_steps=50,
        include_page_context_text=False,
        use_screenshot=True,
        use_ax_tree=True,
        use_html=True,
    )
    screenshot = _png_bytes(5, 5, (0, 0, 255))
    try:
        await env.reset()
        env._execute_bgym_action = AsyncMock(
            return_value={
                **_fake_bgym_step_obs(
                    screenshot=screenshot,
                    url="https://metadata-leak.test/",
                ),
                "open_pages_titles": ("Metadata Leak",),
                "last_action_error": "timeout exceeded",
            }
        )

        r = await env.step(
            [
                make_tool_call(
                    "computer",
                    {"actions": [{"action": "click", "coordinate": [500, 500]}]},
                    call_id="call_action",
                ),
            ]
        )

        result = r.results[0]
        assert result.tool_call_id == "call_action"
        assert result.images[-1] == env._overlay_cursor(screenshot)
        assert result.text is None
        assert result.error == "click failed: execution failed"
        assert result.metadata == {
            "open_pages_urls": ["https://metadata-leak.test/"],
            "open_pages_titles": ["Metadata Leak"],
            "active_page_index": 0,
            "is_error": True,
        }
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_backend_execution_exception_sets_tool_result_error():
    env = _make_fake(max_steps=50, use_screenshot=False)
    try:
        await env.reset()

        async def _fake_execute(_code: str):
            env._last_action_execution_error = "playwright target closed"
            return None

        env._execute_bgym_action = AsyncMock(side_effect=_fake_execute)

        r = await env.step(
            [
                make_tool_call(
                    "computer",
                    {"actions": [{"action": "click", "coordinate": [500, 500]}]},
                    call_id="call_action",
                ),
            ]
        )

        assert r.results[0].tool_call_id == "call_action"
        assert r.results[0].error == "click failed: execution failed"
        assert "playwright target closed" not in r.results[0].error
        assert r.results[0].metadata == {"is_error": True}
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_malformed_drag_returns_current_feedback_without_execution():
    env = _make_fake(max_steps=50, use_screenshot=False)
    try:
        await env.reset()
        env._execute_bgym_action = AsyncMock()

        r = await env.step(
            [
                make_tool_call(
                    "computer",
                    {"actions": [{"action": "drag", "start_coordinate": [0, 0]}]},
                    call_id="call_drag",
                ),
            ]
        )

        env._execute_bgym_action.assert_not_awaited()
        assert r.results[0].tool_call_id == "call_drag"
        assert r.results[0].text
        assert TOOL_RESULT_ERROR_SECTION_HEADER not in r.results[0].text
        assert r.results[0].error
        assert r.results[0].error.startswith("invalid arguments for drag:")
        assert "coordinate" in r.results[0].error
        assert r.results[0].metadata == {"is_error": True}
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_direct_valid_actions_empty_keeps_active_bid_extra():
    config = BrowserGymConfig(
        bgym_task_id="miniwob.click-dialog",
        benchmark="miniwob",
        action_subsets=["bid"],
        use_screenshot=False,
        valid_actions=[],
    )
    env = BrowserGymEnv(config=config, use_fake=True, extra_tools=["click"])
    try:
        await env.reset()

        async def _fake_execute(code: str) -> dict[str, Any]:
            return {
                "reward": 0.0,
                "last_action_error": "",
                "open_pages_urls": ("http://miniwob/",),
                "open_pages_titles": ("miniwob",),
                "active_page_index": [0],
            }

        env._execute_bgym_action = AsyncMock(side_effect=_fake_execute)

        r = await env.step(
            [
                make_tool_call("click", {"bid": "a47"}, call_id="call_bid"),
            ]
        )

        assert r.terminated is False
        assert r.truncated is False
        env._execute_bgym_action.assert_awaited_once()
        assert env._execute_bgym_action.await_args.args[0] == "click('a47')"
        assert r.results[0].tool_call_id == "call_bid"
        assert (r.results[0].metadata or {}).get("is_error") is not True
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_active_upload_file_accepts_list_and_executes_when_selected():
    config = BrowserGymConfig(
        bgym_task_id="miniwob.click-dialog",
        benchmark="miniwob",
        action_subsets=["visualwebarena"],
        use_screenshot=False,
        valid_actions=[],
    )
    env = BrowserGymEnv(config=config, use_fake=True, extra_tools=["upload_file"])
    try:
        await env.reset()

        async def _fake_execute(code: str) -> dict[str, Any]:
            return {
                "reward": 0.0,
                "last_action_error": "",
                "open_pages_urls": ("http://vwa/",),
                "open_pages_titles": ("vwa",),
                "active_page_index": [0],
            }

        env._execute_bgym_action = AsyncMock(side_effect=_fake_execute)

        r = await env.step(
            [
                make_tool_call(
                    "upload_file",
                    {"bid": "a47", "file": ["receipt.pdf", "/tmp/photo.jpg"]},
                    call_id="call_upload",
                ),
            ]
        )

        assert r.terminated is False
        assert r.truncated is False
        env._execute_bgym_action.assert_awaited_once()
        assert env._execute_bgym_action.await_args.args[0] == (
            "upload_file('a47', ['receipt.pdf', '/tmp/photo.jpg'])"
        )
        assert r.results[0].tool_call_id == "call_upload"
        assert (r.results[0].metadata or {}).get("is_error") is not True
    finally:
        await env.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", [[], ["click"]])
async def test_bid_click_reaches_backend_only_when_selected(selected: list[str]):
    """A bid-mode ``click`` executes iff ``extra_tools`` selected it.

    Runtime validation is env-owned. ``click`` is also a desktop GUI action
    name, so the env must route by name+args shape: ``click(bid=...)`` reaches
    BrowserGym only when that native extra is active.
    """
    config = BrowserGymConfig(
        bgym_task_id="miniwob.click-dialog",
        benchmark="miniwob",
        action_subsets=["bid"],
        use_screenshot=False,
    )
    env = BrowserGymEnv(config=config, use_fake=True, extra_tools=selected)
    try:
        await env.reset()

        async def _fake_execute(code: str) -> dict[str, Any]:
            return {
                "reward": 0.0,
                "last_action_error": "",
                "open_pages_urls": ("http://miniwob/",),
                "open_pages_titles": ("miniwob",),
                "active_page_index": [0],
            }

        env._execute_bgym_action = AsyncMock(side_effect=_fake_execute)

        r = await env.step(
            [
                make_tool_call("click", {"bid": "a47"}, call_id="call_bid"),
            ]
        )

        assert r.terminated is False
        assert r.results[0].tool_call_id == "call_bid"
        if selected:
            env._execute_bgym_action.assert_awaited_once()
            assert env._execute_bgym_action.await_args.args[0] == "click('a47')"
            assert (r.results[0].metadata or {}).get("is_error") is not True
        else:
            env._execute_bgym_action.assert_not_awaited()
            assert r.results[0].error == "click is not available in this task."
            assert r.results[0].metadata == {"is_error": True}
            assert r.results[0].text
    finally:
        await env.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("response", {"text": "OK"}),
        ("terminate", {"status": "failure", "reason": "blocked"}),
    ],
)
async def test_active_canonical_terminal_tools_execute_with_canonical_schema(
    name: str,
    arguments: dict[str, Any],
):
    config = BrowserGymConfig(
        bgym_task_id="miniwob.click-dialog",
        benchmark="miniwob",
        use_screenshot=False,
    )
    env = BrowserGymEnv(config=config, use_fake=True, extra_tools=["response", "terminate"])
    try:
        await env.reset()
        schema_names = {tool_schema_name(s) for s in env.metadata.extra_tool_schemas}
        assert {"response", "terminate"} <= schema_names
        assert not (schema_names & {"answer", "send_msg_to_user", "report_infeasible"})

        async def fake_execute(_code: str) -> dict[str, Any]:
            env._terminated = True
            return {
                "reward": 1.0,
                "open_pages_urls": (),
                "open_pages_titles": (),
                "active_page_index": [0],
            }

        env._execute_bgym_action = AsyncMock(side_effect=fake_execute)

        r = await env.step(
            [
                make_tool_call(name, arguments, call_id=f"call_{name}"),
            ]
        )

        assert r.terminated is True
        assert r.truncated is False
        # A terminal call gets NO tool result (see above); what this test defends
        # is that the CANONICAL name still executes its bgym code and leaves the
        # schema set untouched, asserted below.
        assert r.results == []
        env._execute_bgym_action.assert_awaited_once()
        executed = r.info["executed_actions"][0]["call"]
        if name == "terminate":
            assert executed.startswith("report_infeasible(")
        else:
            assert executed.startswith("send_msg_to_user(")
        schema_names_after = {tool_schema_name(s) for s in env.metadata.extra_tool_schemas}
        assert schema_names_after == schema_names
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_raw_answer_is_an_undeclared_name_not_a_send_msg_alias():
    """``answer`` is NOT a name this env executes.

    The families whose prompts spell ``answer`` (``qwen3_vl`` / ``qwen3_5``) map it
    to canonical ``response`` in ``convert_tool_calls_from_agent``, so the env
    never sees the raw name. A bare ``answer`` is an undeclared name like any
    other: it must NOT submit and must NOT terminate."""
    config = BrowserGymConfig(
        bgym_task_id="miniwob.click-dialog",
        benchmark="miniwob",
        use_screenshot=False,
    )
    env = BrowserGymEnv(config=config, use_fake=True, extra_tools=["response"])
    try:
        await env.reset()
        schema_names = {tool_schema_name(s) for s in env.metadata.extra_tool_schemas}
        assert "response" in schema_names
        assert "answer" not in schema_names

        env._execute_bgym_action = AsyncMock()

        r = await env.step(
            [
                make_tool_call("answer", {"text": "OK"}, call_id="call_answer"),
            ]
        )

        assert r.terminated is False
        assert r.truncated is False
        assert len(r.results) == 1
        assert r.results[0].tool_call_id == "call_answer"
        assert r.results[0].error == "unknown tool: answer"
        assert r.results[0].metadata == {"is_error": True}
        env._execute_bgym_action.assert_not_awaited()
        assert r.info["executed_actions"] == [
            {"call": "noop", "args": {"name": "answer", "reason": "unknown action"}},
        ]
    finally:
        await env.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("send_msg_to_user", {"message": "OK"}),
        ("report_infeasible", {"reason": "blocked"}),
        ("go_back", {}),
        ("go_forward", {}),
        ("tab_focus", {"index": 0}),
        ("tab_close", {}),
    ],
)
async def test_noncanonical_tool_names_return_unsupported_without_submitting(
    name: str,
    arguments: dict[str, Any],
):
    config = BrowserGymConfig(
        bgym_task_id="miniwob.click-dialog",
        benchmark="miniwob",
        include_page_context_text=True,
        use_screenshot=False,
    )
    env = BrowserGymEnv(config=config, use_fake=True, extra_tools=["response", "terminate"])
    try:
        await env.reset()
        schema_names = {tool_schema_name(s) for s in env.metadata.extra_tool_schemas}
        assert {"response", "terminate"} <= schema_names
        assert not (schema_names & {"answer", "send_msg_to_user", "report_infeasible"})
        env._execute_bgym_action = AsyncMock()
        call_id = f"call_{name}"

        r = await env.step(
            [
                make_tool_call(name, arguments, call_id=call_id),
            ]
        )

        assert r.terminated is False
        assert r.truncated is False
        assert r.results[0].tool_call_id == call_id
        assert r.results[0].text
        assert "AXTree" in r.results[0].text or "Currently open tabs" in r.results[0].text
        assert r.results[0].error == f"{name} is not available in this task."
        assert r.results[0].metadata == {"is_error": True}
        env._execute_bgym_action.assert_not_awaited()
        schema_names_after = {tool_schema_name(s) for s in env.metadata.extra_tool_schemas}
        assert schema_names_after == schema_names
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_page_context_disabled_keeps_unsupported_tool_error_without_page_text():
    config = BrowserGymConfig(
        bgym_task_id="miniwob.click-dialog",
        benchmark="miniwob",
        include_page_context_text=False,
    )
    env = BrowserGymEnv(config=config, use_fake=True, extra_tools=["response", "terminate"])
    try:
        await env.reset()
        env._execute_bgym_action = AsyncMock()

        r = await env.step(
            [
                make_tool_call("tab_focus", {"index": 0}, call_id="call-tab-focus"),
            ]
        )

        result = r.results[0]
        assert result.tool_call_id == "call-tab-focus"
        assert result.images
        assert result.text is None
        assert result.error == "tab_focus is not available in this task."
        assert result.metadata == {
            "open_pages_urls": ["http://browsergym.fake/"],
            "open_pages_titles": ["BrowserGym Fake"],
            "active_page_index": 0,
            "is_error": True,
        }
        env._execute_bgym_action.assert_not_awaited()
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_page_context_disabled_keeps_page_context_in_reset_metadata():
    config = BrowserGymConfig(
        bgym_task_id="miniwob.click-dialog",
        benchmark="miniwob",
        include_page_context_text=False,
    )
    env = BrowserGymEnv(config=config, use_fake=True)
    try:
        obs = await env.reset()

        assert obs.text == "Fake task instruction for testing"
        assert obs.metadata == {
            "open_pages_urls": ["http://browsergym.fake/"],
            "open_pages_titles": ["BrowserGym Fake"],
            "active_page_index": 0,
        }
    finally:
        await env.close()


def test_browsergym_yaml_terminal_prompts_use_canonical_names():
    root = Path(__file__).resolve().parents[4]
    config_root = root / "scripts" / "configs"
    stale = (
        "Use send_msg_to_user",
        "When sending the final answer with send_msg_to_user",
    )
    offenders: list[str] = []
    for path in sorted(config_root.glob("**/browsergym*/*.yaml")):
        text = path.read_text(encoding="utf-8")
        for needle in stale:
            if needle in text:
                offenders.append(f"{path.relative_to(root)}: {needle}")
    assert not offenders


@pytest.mark.asyncio
async def test_extra_tools_subset_surfaces_only_those():
    # The default action_subsets include `nav`, which advertises canonical
    # goto/back; a subset request must surface ONLY the named tools, preserving
    # caller/YAML order so golden surfaces do not drift by catalog order.
    config = BrowserGymConfig(bgym_task_id="miniwob.click-dialog", benchmark="miniwob")
    requested = ["back", "goto"]
    env = BrowserGymEnv(config=config, use_fake=True, extra_tools=requested)
    try:
        await env.reset()
        names = [tool_schema_name(t) for t in env.metadata.extra_tool_schemas]
        assert names == requested
    finally:
        await env.close()


def test_extra_tools_unknown_name_raises():
    config = BrowserGymConfig(bgym_task_id="miniwob.click-dialog", benchmark="miniwob")
    with pytest.raises(ValueError, match="unknown extra_tools"):
        BrowserGymEnv(config=config, use_fake=True, extra_tools=["totally_fake"])


@pytest.mark.asyncio
async def test_extra_tools_none_exposes_the_whole_derived_catalog():
    """``None`` (the shipped default) == naming every derivable tool.

    The nav block is emitted in ``LiteBrowserNavToolSet`` DECLARATION order, not in the
    order BrowserGym's ``action_set`` iterates:
    ``goto, back, forward, new_tab, switch_tab, close_tab``. It is
    model-visible and it is not a bug; do not restore BrowserGym's iteration
    order.

    See tests/gym/envs/browsergym/test_browsergym_tool_schema.py for the full tri-state
    regression pin.
    """
    config = BrowserGymConfig(bgym_task_id="miniwob.click-dialog", benchmark="miniwob")
    env = BrowserGymEnv(config=config, use_fake=True, extra_tools=None)
    try:
        await env.reset()
        assert [tool_schema_name(s) for s in env.metadata.extra_tool_schemas] == [
            "response",
            "terminate",
            "goto",
            "back",
            "forward",
            "new_tab",
            "switch_tab",
            "close_tab",
        ]
    finally:
        await env.close()

    derivable = [
        tool_schema_name(s)
        for s in _tools_for_subsets(
            BrowserGymConfig(
                bgym_task_id="miniwob.click-dialog",
                benchmark="miniwob",
            ).action_subsets
        )
    ]
    env = BrowserGymEnv(config=config, use_fake=True, extra_tools=derivable)
    try:
        await env.reset()
        # Naming the derived catalog explicitly reproduces it exactly: the
        # ``[names]`` branch preserves the CALLER's order, and ``derivable`` is
        # now itself in ``LiteBrowserNavToolSet`` declaration order.
        assert [tool_schema_name(s) for s in env.metadata.extra_tool_schemas] == [
            "response",
            "terminate",
            "goto",
            "back",
            "forward",
            "new_tab",
            "switch_tab",
            "close_tab",
        ]
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_close_idempotent():
    env = _make_fake()
    await env.reset()
    await env.close()
    # Double close should also be safe (was an issue earlier).
    await env.close()


def test_unknown_config_field_raises_typeerror():
    """Strict-typo behavior: unknown env_kwargs from yaml raise TypeError."""
    cfg = BrowserGymConfig(bgym_task_id="miniwob.click-dialog", benchmark="miniwob")
    with pytest.raises(TypeError):
        BrowserGymEnv(config=cfg, use_fake=True, totally_unknown_field=42)


class TestBuildObsText:
    """``BrowserGymEnv._build_obs_text`` composes prefix + AXTree + HTML + focused-element."""

    def test_empty_returns_none(self):
        env = _make_fake()
        assert env._build_obs_text(None, prefix=None) is None
        assert env._build_obs_text({}, prefix=None) is None

    def test_prefix_only(self):
        env = _make_fake()
        out = env._build_obs_text({}, prefix="hello")
        assert out == "hello"

    def test_page_context_disabled_keeps_prefix_only(self):
        env = _make_fake(
            include_page_context_text=False,
            use_ax_tree=True,
            use_html=True,
            use_focused_element=True,
        )
        obs = {
            "open_pages_urls": ["https://example.com/"],
            "open_pages_titles": ["Example"],
            "active_page_index": [0],
            "url": "https://example.com/",
            "focused_element_bid": "a47",
            "axtree_object": {"role": "RootWebArea", "name": "root", "nodeId": "1"},
            "dom_object": {"any": "html"},
            "extra_element_properties": {},
        }
        with (
            patch("browsergym.utils.obs.flatten_axtree_to_str") as ax_fn,
            patch("browsergym.utils.obs.flatten_dom_to_str") as html_fn,
        ):
            out = env._build_obs_text(obs, prefix="goal text")
            no_prefix = env._build_obs_text(obs, prefix=None)

        assert out == "goal text"
        assert no_prefix is None
        assert not ax_fn.called
        assert not html_fn.called

        metadata = env._build_page_context_metadata(obs)
        assert metadata == {
            "open_pages_urls": ["https://example.com/"],
            "open_pages_titles": ["Example"],
            "active_page_index": 0,
            "url": "https://example.com/",
            "focused_element_bid": "a47",
        }

    def test_focused_element_inline(self):
        env = _make_fake(include_page_context_text=True, use_focused_element=True)
        out = env._build_obs_text({"focused_element_bid": "a47"}, prefix=None)
        assert out is not None
        assert "## Focused element:" in out
        assert "bid='a47'" in out

    def test_focused_element_disabled(self):
        env = _make_fake(use_focused_element=False)
        out = env._build_obs_text({"focused_element_bid": "a47"}, prefix=None)
        # Without the flag, focused-element block is suppressed.
        assert out is None

    def test_focused_element_no_bid(self):
        env = _make_fake(include_page_context_text=True, use_focused_element=True)
        out = env._build_obs_text({"focused_element_bid": ""}, prefix=None)
        # Empty bid → block skipped (no "## Focused element:" emitted).
        assert out is None

    def test_tabs_block_rendered(self):
        env = _make_fake(include_page_context_text=True)
        obs = {
            "open_pages_urls": ["https://google.com/", "http://localhost:7770/"],
            "open_pages_titles": ["Google", "WA Shopping"],
            "active_page_index": [0],
        }
        out = env._build_obs_text(obs, prefix=None)
        assert out is not None
        assert "## Currently open tabs:" in out
        assert "Tab 0 (active tab):" in out
        assert "Title: Google" in out
        assert "Tab 1:" in out
        assert "URL: http://localhost:7770/" in out

    def test_tabs_block_active_index_int_or_array(self):
        env = _make_fake(include_page_context_text=True)
        # numpy-style: active is single-element array
        obs1 = {"open_pages_urls": ["a"], "open_pages_titles": ["t"], "active_page_index": [0]}
        # plain int variant
        obs2 = {"open_pages_urls": ["a"], "open_pages_titles": ["t"], "active_page_index": 0}
        for obs in (obs1, obs2):
            out = env._build_obs_text(obs, prefix=None)
            assert out is not None
            assert "Tab 0 (active tab):" in out

    def test_no_tabs_block_when_obs_lacks_them(self):
        env = _make_fake(include_page_context_text=True)
        out = env._build_obs_text({"focused_element_bid": "a1"}, prefix=None)
        # No open_pages_urls/titles → no tabs section
        if out:
            assert "## Currently open tabs:" not in out

    def test_prefix_with_focused_block(self):
        env = _make_fake(include_page_context_text=True, use_focused_element=True)
        out = env._build_obs_text(
            {"focused_element_bid": "a47"},
            prefix=f"{TOOL_RESULT_ERROR_SECTION_HEADER}\nclick: timed out",
        )
        assert out is not None
        # Header + body present, then focused block follows.
        assert TOOL_RESULT_ERROR_SECTION_HEADER in out
        assert "click: timed out" in out
        assert "## Focused element:" in out


class TestObsFlagsTakeEffect:
    """Each ObsFlags knob in BrowserGymConfig (`use_ax_tree`, `use_html`,
    `use_focused_element`, `use_error_logs`, `extract_visible_tag`,
    `extract_clickable_tag`, `extract_coords`, `filter_visible_elements_only`,
    `action_subsets`) must actually drive ``_build_obs_text`` /
    ``_tools_for_subsets`` output. The yaml configs ship these explicitly so
    a regression that silently ignores them would damage every rollout.
    """

    # Minimal mock obs: a stub ``axtree_object`` that ``flatten_axtree_to_str``
    # can render. We don't care about content — we care that the flag plumbing
    # actually CALLS the BrowserGym renderer (we verify the AXTree section
    # appears or doesn't), and that tabs / focused / error sections respect
    # their flags.
    @staticmethod
    def _stub_obs_with_axtree() -> dict:
        # Fake AXTree object: a minimal valid root node. ``flatten_axtree_to_str``
        # accepts any dict with the expected shape.
        return {
            "axtree_object": {"role": "RootWebArea", "name": "root", "nodeId": "1"},
            "extra_element_properties": {},
            "focused_element_bid": "a1",
            "open_pages_urls": ["https://example.com/"],
            "open_pages_titles": ["Example"],
            "active_page_index": [0],
        }

    def test_use_ax_tree_true_calls_renderer(self):
        # When use_ax_tree=True AND axtree_object provided, env calls
        # browsergym's flatten_axtree_to_str + emits a ## AXTree: section.
        with patch(
            "browsergym.utils.obs.flatten_axtree_to_str", return_value="MOCKED AXTREE BODY"
        ) as fn:
            env = _make_fake(include_page_context_text=True, use_ax_tree=True)
            out = env._build_obs_text(self._stub_obs_with_axtree(), prefix=None)
        assert out is not None
        assert "## AXTree:" in out
        assert "MOCKED AXTREE BODY" in out
        assert fn.called

    def test_use_ax_tree_false_skips_renderer(self):
        # Flag off → renderer NOT called, section NOT emitted.
        with patch("browsergym.utils.obs.flatten_axtree_to_str") as fn:
            env = _make_fake(use_ax_tree=False)
            out = env._build_obs_text(self._stub_obs_with_axtree(), prefix=None)
        assert not fn.called
        if out:
            assert "## AXTree:" not in out

    def test_tabs_block_precedes_axtree_section(self):
        # AgentLab convention: tabs context comes BEFORE the page tree so the
        # model sees "which tab am I on" before parsing the (often huge) tree.
        with patch("browsergym.utils.obs.flatten_axtree_to_str", return_value="ax body"):
            env = _make_fake(include_page_context_text=True, use_ax_tree=True)
            out = env._build_obs_text(self._stub_obs_with_axtree(), prefix=None)
        assert out is not None
        assert "## Currently open tabs:" in out
        assert "## AXTree:" in out
        assert out.index("## Currently open tabs:") < out.index("## AXTree:")

    def test_use_html_true_calls_renderer(self):
        with patch(
            "browsergym.utils.obs.flatten_dom_to_str", return_value="MOCKED HTML BODY"
        ) as fn:
            env = _make_fake(
                include_page_context_text=True,
                use_html=True,
                use_ax_tree=False,
            )
            obs = {"dom_object": {"any": "html"}, "extra_element_properties": {}}
            out = env._build_obs_text(obs, prefix=None)
        assert out is not None
        assert "## HTML:" in out
        assert "MOCKED HTML BODY" in out
        assert fn.called

    def test_use_html_false_skips_renderer(self):
        with patch("browsergym.utils.obs.flatten_dom_to_str") as fn:
            env = _make_fake(use_html=False)
            obs = {"dom_object": {"some": "html"}}
            out = env._build_obs_text(obs, prefix=None)
        assert not fn.called
        if out:
            assert "## HTML:" not in out

    def test_extract_visible_clickable_flags_propagate(self):
        # `extract_visible_tag` / `extract_clickable_tag` are passed verbatim
        # to flatten_axtree_to_str — verify the kwargs reach it.
        with patch("browsergym.utils.obs.flatten_axtree_to_str", return_value="ax") as fn:
            env = _make_fake(
                include_page_context_text=True,
                use_ax_tree=True,
                extract_visible_tag=False,
                extract_clickable_tag=True,
                filter_visible_elements_only=True,
            )
            env._build_obs_text(self._stub_obs_with_axtree(), prefix=None)
        kwargs = fn.call_args.kwargs
        assert kwargs["with_visible"] is False
        assert kwargs["with_clickable"] is True
        assert kwargs["filter_visible_only"] is True

    def test_extract_coords_string_drives_renderer_kwargs(self):
        # `extract_coords` literal must select the right pair of bool kwargs.
        for value, expect_center, expect_box in [
            ("False", False, False),
            ("center", True, False),
            ("box", False, True),
        ]:
            with patch("browsergym.utils.obs.flatten_axtree_to_str", return_value="ax") as fn:
                env = _make_fake(
                    include_page_context_text=True,
                    use_ax_tree=True,
                    extract_coords=value,
                )
                env._build_obs_text(self._stub_obs_with_axtree(), prefix=None)
            kwargs = fn.call_args.kwargs
            assert kwargs["with_center_coords"] is expect_center, value
            assert kwargs["with_bounding_box_coords"] is expect_box, value

    def test_extract_coords_string_literal(self):
        # `extract_coords` is a Literal["False","center","box"] STRING — not bool.
        # Verify the field type passes through.
        cfg = BrowserGymConfig(
            bgym_task_id="miniwob.click-dialog",
            benchmark="miniwob",
            extract_coords="center",
        )
        assert cfg.extract_coords == "center"
        # And boolean comparison would silently mis-route — verify equality is by string.
        assert cfg.extract_coords != True  # noqa: E712

    def test_filter_visible_elements_only_propagates(self):
        # Plumb-through: the field is set on config and passed through.
        cfg = BrowserGymConfig(
            bgym_task_id="miniwob.click-dialog",
            benchmark="miniwob",
            filter_visible_elements_only=True,
        )
        assert cfg.filter_visible_elements_only is True

    def test_action_subsets_drives_metadata_extra_tools(self):
        """Yaml `env_kwargs.action_subsets` re-derives the selectable catalog.

        `extra_tools` remains the explicit selector; selecting every derived
        name should surface the full bid/VWA catalog.
        """
        for subsets, expected_count, must_include in [
            (("bid",), 11, "drag_and_drop"),
            (("visualwebarena",), 15, "upload_file"),
        ]:
            requested = [tool_schema_name(tool) for tool in _tools_for_subsets(subsets)]
            env = _make_fake(action_subsets=subsets, extra_tools=requested)
            try:
                names = {tool_schema_name(t) for t in env.metadata.extra_tool_schemas}
                assert len(names) == expected_count, (
                    f"subsets={subsets}: expected {expected_count} tools, got {len(names)}: {names}"
                )
                assert must_include in names, (
                    f"subsets={subsets}: expected '{must_include}' in tools"
                )
            finally:
                # _make_fake creates an executor; clean up.
                env._executor.shutdown(wait=False)

    def test_max_steps_enforced(self):
        # max_steps from env_kwargs (yaml) is plumbed onto BrowserGymEnv.
        env = _make_fake(max_steps=7)
        assert env._max_steps == 7
        env._executor.shutdown(wait=False)

    def test_error_section_via_labeled_prefix(self):
        # ``BrowserGymEnv.step`` wraps the action error in a labeled section
        # before passing it as the ``prefix=`` arg to ``_build_obs_text``.
        # Protocol re-parses via ``_extract_section_after`` instead of a
        # fragile substring heuristic.
        env = _make_fake()
        out = env._build_obs_text(
            {"focused_element_bid": "a1"},
            prefix=f"{TOOL_RESULT_ERROR_SECTION_HEADER}\nelement not visible",
        )
        assert out is not None
        assert TOOL_RESULT_ERROR_SECTION_HEADER in out
        assert "element not visible" in out

    def test_no_error_section_when_prefix_none(self):
        # Empty error → ``BrowserGymEnv.step`` passes ``prefix=None``;
        # ``_build_obs_text`` must not synthesize a stray error section.
        env = _make_fake(include_page_context_text=True, use_focused_element=True)
        out = env._build_obs_text({"focused_element_bid": "a1"}, prefix=None)
        if out:
            assert TOOL_RESULT_ERROR_SECTION_HEADER not in out
