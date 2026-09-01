"""Tests for the androidworld CUA-Lite gym environment.

Uses a fake environment mock — no real emulator needed.
Requires the androidworld package to be installed.

Run:
    uv run pytest tests/gym/envs/androidworld/test_androidworld.py -v
"""

from __future__ import annotations

import pytest

import lite.gym.envs.androidworld.main as M
from lite.core.messages.final import make_no_tool_call_final_actions
from lite.core.tools import make_tool_call
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.schemas import tool_schema_name, tool_schema_parameters
from lite.gym.envs.androidworld.main import (
    AndroidWorldEnv,
    AndroidWorldTaskConfig,
    _FakeTaskEval,
    _make_env,
)


def _make_fake(max_steps: int = 50, **kwargs) -> AndroidWorldEnv:
    """Create a fake androidworld env for testing (no emulator needed)."""
    return _make_env(
        task_class=_FakeTaskEval,
        config=AndroidWorldTaskConfig(
            use_fake=True,
            instruction_template="Interact with the fake Android device (testing mode).",
        ),
        max_steps=max_steps,
        **kwargs,
    )


def _tool_schema(env: AndroidWorldEnv, name: str) -> dict:
    return next(
        schema
        for schema in env.metadata.extra_tool_schemas
        if tool_schema_name(schema) == name
    )


def _mobile_action_schema_enum(metadata) -> list[str]:
    from lite.agents.core.action_space.base import LiteMobileActionSpace

    schemas = LiteMobileActionSpace.get_tool_schemas()
    if metadata.valid_actions is not None:
        schemas = LiteMobileActionSpace.filter_child_action_enum(
            schemas, metadata.valid_actions,
        )
    mobile = next(s for s in schemas if tool_schema_name(s) == "mobile")
    action_properties = tool_schema_parameters(mobile)["properties"]["actions"]["items"][
        "properties"
    ]["action"]
    return action_properties["enum"]


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

def test_task_registration_from_metadata():
    """If androidworld is importable, tasks should be registered."""
    import lite.gym.envs.androidworld.main  # noqa: F401
    from lite.gym.registry import _splits

    if "androidworld" in _splits:
        all_tasks = []
        for split_tasks in _splits["androidworld"].values():
            all_tasks.extend(split_tasks)
        assert len(all_tasks) > 0


def test_services_ensure_uses_dependency_preflight(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(M, "_ensure_services", lambda env_id: calls.append(env_id))

    svc = M.AndroidWorldServices()
    svc.ensure("androidworld")

    assert calls == ["androidworld"]


def test_services_health_uses_cached_dependency_preflight(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(M, "_HEALTH_CHECK", lambda env_id: calls.append(env_id))

    svc = M.AndroidWorldServices()
    svc.health("androidworld")

    assert calls == ["androidworld"]


@pytest.mark.asyncio
async def test_destroy_backend_refuses_replacement_when_removal_is_unconfirmed():
    from lite.gym.errors import CapacityExhausted

    class _Container:
        name = "still-running-emulator"

        def destroy(self):
            return False

    env = _make_fake()
    env._current_container = _Container()
    with pytest.raises(CapacityExhausted, match="refusing to start a replacement"):
        await env.destroy_backend()
    assert env._current_container.name == "still-running-emulator"
    with pytest.raises(CapacityExhausted, match="refusing to start a replacement"):
        await env.boot()

# ---------------------------------------------------------------------------
# Sync tests
# ---------------------------------------------------------------------------

def test_make_fake_env():
    """Factory should return an env with mobile platform."""
    e = _make_fake(max_steps=3)
    assert e.metadata.platform == "mobile"
    assert e.metadata.task_type == "use"

def test_metadata():
    """metadata should have correct platform."""
    e = _make_fake(max_steps=5)
    assert e.metadata.platform == "mobile"


def test_open_app_catalog_metadata_matches_schema_enum():
    env = _make_fake(max_steps=5, extra_tools=["open_app"])

    assert env.metadata.others["apps"] == M._ANDROID_WORLD_APPS
    assert (
        tool_schema_parameters(
            _tool_schema(env, "open_app"),
        )["properties"]["app_name"]["enum"]
        == env.metadata.others["apps"]
    )


def test_default_mobile_schema_does_not_advertise_pinch():
    env = _make_fake(max_steps=5)
    enum = _mobile_action_schema_enum(env.metadata)
    assert "pinch" not in enum
    assert "system_button" in enum


def test_androidworld_swipe_duration_is_forwarded_to_adb_rpc():
    env = _make_fake(max_steps=5)
    env._screen_w, env._screen_h = 1080, 2400
    calls = []

    class _Rpc:
        def post(self, path, body):
            calls.append((path, body))
            return {}

    class _Backend:
        _rpc = _Rpc()

    env._env = _Backend()

    out = env._exec_swipe_adb({
        "start_coordinate": [250, 250],
        "coordinate": [750, 750],
        "duration": 1.25,
    })

    assert calls == [(
        "/env/exec_swipe",
        {"sx": 270, "sy": 600, "ex": 810, "ey": 1800, "duration_ms": 1250},
    )]
    assert out["args"]["duration_ms"] == 1250


def test_androidworld_json_action_durations_are_validate_only():
    env = _make_fake(max_steps=5)
    env._screen_w, env._screen_h = 1080, 2400

    assert env._translate_action(
        "long_press",
        {"coordinate": [500, 500], "duration": 2.0},
    ) == {"action_type": "long_press", "x": 540, "y": 1200}
    assert env._translate_action("wait", {"duration": 2.0}) == {
        "action_type": "wait"
    }


def test_androidworld_bare_double_tap_is_not_an_action_name():
    """``double_tap`` is a JSONAction *action_type*, never a cua-lite action.

    The shared gate cannot reject it
    (``is_lite_action_name_or_action_batch_tool_name("double_tap")`` is False), so
    ``unsupported_env_action_message`` permits the name. A dispatch arm for it would
    execute an undeclared action beside the canonical ``tap(clicks=2)``.
    """
    from lite.gym.utils.feedback.surface import android_supported_actions

    env = _make_fake(max_steps=5)
    env._screen_w, env._screen_h = 1080, 2400

    assert "double_tap" not in android_supported_actions()
    assert env._translate_action("double_tap", {"coordinate": [500, 500]}) is None
    # ...but the canonical spelling still emits the upstream action_type.
    assert env._translate_action("tap", {"coordinate": [500, 500], "clicks": 2}) == {
        "action_type": "double_tap", "x": 540, "y": 1200,
    }


def test_androidworld_translation_preserves_no_clamp_max_edge():
    env = _make_fake(max_steps=5)
    env._screen_w, env._screen_h = 1080, 2400

    assert env._translate_action("tap", {"coordinate": [1000, 1000]}) == {
        "action_type": "click",
        "x": 1080,
        "y": 2400,
    }


# ---------------------------------------------------------------------------
# Async tests — full lifecycle with fake env
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_lifecycle():
    """End-to-end: reset, actions, termination."""
    env = _make_fake(max_steps=50, extra_tools=["open_app", "terminate"])
    try:
        # === RESET ===
        obs0 = await env.reset()
        assert obs0.image
        raw = obs0.image
        assert raw[:4] == b"\x89PNG", "Screenshot should be valid PNG"

        # === TAP ===
        r1 = await env.step([
            make_tool_call("tap", {"coordinate": [500, 300]}),
        ])
        assert r1.results[0].images
        assert r1.info["executed_actions"][0]["call"] == "tap"

        # === LONG PRESS ===
        r2 = await env.step([
            make_tool_call("long_press", {"coordinate": [200, 800]}),
        ])
        assert r2.info["executed_actions"][0]["call"] == "long_press"

        # === SWIPE ===
        r3 = await env.step([
            make_tool_call("swipe", {
                "start_coordinate": [500, 700],
                "coordinate": [500, 300],
            }),
        ])
        assert r3.info["executed_actions"][0]["call"] == "swipe"

        # === TYPE ===
        r4 = await env.step([
            make_tool_call("type", {"text": "hello world"}),
        ])
        assert r4.info["executed_actions"][0]["call"] == "type"

        # === OPEN APP ===
        r5 = await env.step([
            make_tool_call("open_app", {"app_name": "Chrome"}),
        ])
        assert r5.info["executed_actions"][0]["call"] == "open_app"

        # === SYSTEM BUTTON HOME ===
        r6 = await env.step([
            make_tool_call("system_button", {"button": "Home"}),
        ])
        assert r6.info["executed_actions"][0]["call"] == "system_button"
        assert r6.info["executed_actions"][0]["args"]["button"] == "Home"

        # === SYSTEM BUTTON BACK ===
        r7 = await env.step([
            make_tool_call("system_button", {"button": "Back"}),
        ])
        assert r7.info["executed_actions"][0]["call"] == "system_button"
        assert r7.info["executed_actions"][0]["args"]["button"] == "Back"

        # === WAIT ===
        r8 = await env.step([
            make_tool_call("wait"),
        ])
        assert r8.info["executed_actions"][0]["call"] == "wait"

        # === SCREENSHOT (noop) ===
        r9 = await env.step([
            make_tool_call("screenshot"),
        ])
        assert r9.info["executed_actions"][0]["call"] == "screenshot"
        assert r9.results[0].images

        # === MULTIPLE ACTIONS PER STEP ===
        r10 = await env.step([
            make_tool_call("tap", {"coordinate": [100, 100]}),
            make_tool_call("tap", {"coordinate": [900, 900]}),
        ])
        assert len(r10.info["executed_actions"]) == 2

        # === TERMINATE ===
        r_term = await env.step([
            make_tool_call("terminate", {"status": "success"}),
        ])
        assert r_term.terminated is True
        # Fake task always returns 1.0
        assert r_term.reward == 1.0

    finally:
        await env.close()

@pytest.mark.asyncio
async def test_truncation():
    """Environment should truncate after max_steps."""
    env = _make_fake(max_steps=2)
    try:
        await env.reset()
        r1 = await env.step([
            make_tool_call("tap", {"coordinate": [500, 500]}),
        ])
        assert not r1.truncated

        r2 = await env.step([
            make_tool_call("tap", {"coordinate": [500, 500]}),
        ])
        assert r2.truncated
        # Truncation → reward is 0.0 (matching androidworld: agent must
        # signal done for reward to count)
        assert r2.reward == 0.0
    finally:
        await env.close()

@pytest.mark.asyncio
async def test_reset_reuse():
    """reset() should reuse the existing env connection."""
    env = _make_fake(max_steps=10)
    try:
        obs1 = await env.reset()
        assert obs1.image

        await env.step([
            make_tool_call("tap", {"coordinate": [500, 500]}),
        ])

        obs2 = await env.reset()
        assert obs2.image
    finally:
        await env.close()

@pytest.mark.asyncio
async def test_response_terminates():
    """'response' action should terminate like 'terminate'."""
    env = _make_fake(max_steps=10, extra_tools=["response"])
    try:
        await env.reset()
        r = await env.step([
            make_tool_call("response", {"text": "done"}, call_id="call_response"),
        ])
        assert r.terminated is True
        # A terminal call gets NO tool result: it ended the episode, so there
        # is no next decision for an observation to inform, and
        # ``devs/migration/verify.py`` refuses a tool result for a terminal call.
        assert r.results == []
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_content_only_final_text_uses_internal_response_no_tool_result():
    env = _make_fake(max_steps=10)
    try:
        await env.reset()

        actions = make_no_tool_call_final_actions("final text")
        result = await env.step(actions)

        assert tool_call_name(actions[0]) == "response"
        assert tool_call_arguments(actions[0]) == {"text": "final text"}
        assert result.terminated is True
        assert result.results == []
        assert result.reward == 1.0
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_step_rejects_flat_lite_boundary_call():
    env = _make_fake(max_steps=10)
    try:
        with pytest.raises(TypeError, match="env.step expects canonical Lite tool calls"):
            await env.step([
                {"name": "tap", "arguments": {"coordinate": [500, 500]}},
            ])
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_active_open_app_executes_with_current_image_when_enabled():
    env = _make_fake(max_steps=10, extra_tools=["open_app"])
    try:
        await env.reset()

        result = await env.step([
            make_tool_call("open_app", {"app_name": "Chrome"}, call_id="call_open"),
        ])

        assert result.info["executed_actions"][0]["call"] == "open_app"
        assert result.results[0].tool_call_id == "call_open"
        assert result.results[0].images
        assert result.results[0].metadata is None
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_response_side_effect_failure_returns_error_with_current_obs():
    class _FailingInteractionCacheEnv(M._FakeAsyncEnv):
        @property
        def interaction_cache(self):
            return None

        @interaction_cache.setter
        def interaction_cache(self, _value):
            raise RuntimeError("cache write failed")

    env = _make_fake(max_steps=10, extra_tools=["response"])
    try:
        await env.reset()
        env._env = _FailingInteractionCacheEnv()

        result = await env.step([
            make_tool_call("response", {"text": "done"}, call_id="call_response"),
        ])

        assert result.terminated is True
        assert result.results[0].tool_call_id == "call_response"
        assert result.results[0].images
        assert result.results[0].error == "response failed: execution failed"
        assert "cache write failed" not in result.results[0].error
        assert result.results[0].metadata == {"is_error": True}
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_inactive_open_app_returns_error_only_feedback():
    env = _make_fake(max_steps=10)
    try:
        await env.reset()

        result = await env.step([
            make_tool_call("open_app", {"app_name": "Chrome"}, call_id="call_open"),
        ])

        assert result.info["executed_actions"] == [{
            "call": "noop",
            "args": {"name": "open_app", "reason": "inactive extra tool"},
        }]
        assert result.results[0].tool_call_id == "call_open"
        assert result.results[0].metadata == {"is_error": True}
        assert result.results[0].error == "open_app is not available in this task."
        assert result.results[0].images == []
        assert env._step_count == 1
        assert result.truncated is False
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_invalid_open_app_is_rejected_before_dispatch(monkeypatch):
    env = _make_fake(max_steps=10, extra_tools=["open_app"])
    try:
        await env.reset()

        async def fail_dispatch(_name, _args):
            raise AssertionError("invalid app should not dispatch")

        monkeypatch.setattr(env, "_dispatch_action", fail_dispatch)

        result = await env.step([
            make_tool_call(
                "open_app",
                {"app_name": "Not A Real App"},
                call_id="call_open",
            ),
        ])

        assert result.info["executed_actions"] == []
        assert result.results[0].tool_call_id == "call_open"
        assert result.results[0].metadata == {"is_error": True}
        assert result.results[0].error.startswith("invalid arguments for open_app: ")
        assert "Not A Real App" in result.results[0].error
        assert result.results[0].images
        assert env._step_count == 1
        assert result.truncated is False
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_unknown_standalone_tool_returns_error_only():
    env = _make_fake(max_steps=1)
    try:
        await env.reset()

        result = await env.step([
            make_tool_call("foo", {}, call_id="call_foo"),
        ])

        assert result.results[0].tool_call_id == "call_foo"
        assert result.results[0].metadata == {"is_error": True}
        assert result.results[0].error == "unknown tool: foo"
        assert result.results[0].images == []
        assert result.results[0].text is None
        assert result.info["executed_actions"] == [{
            "call": "noop",
            "args": {"name": "foo", "reason": "unknown tool"},
        }]
        assert env._step_count == 1
        assert result.truncated is True
        assert result.reward == 0.0
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_unsupported_system_button_returns_model_error_with_current_obs():
    env = _make_fake(max_steps=10, post_action_delay=0.0)
    try:
        await env.reset()
        # Fake mode records actions before backend translation. Flip only the
        # dispatch branch so the test exercises AndroidWorld's button mapper
        # without needing a real emulator.
        env._config.use_fake = False

        result = await env.step([
            make_tool_call(
                "system_button",
                {"button": "Bogus"},
                call_id="call_bogus",
            ),
        ])

        assert result.info["executed_actions"][0]["call"] == "noop"
        assert result.results[0].tool_call_id == "call_bogus"
        assert result.results[0].metadata == {"is_error": True}
        assert result.results[0].error.startswith("invalid arguments for system_button: ")
        assert "unknown button 'Bogus'" in result.results[0].error
        assert result.results[0].images
        assert env._step_count == 1
        assert result.truncated is False
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_recent_system_button_executes_raw_keyevent(monkeypatch):
    env = _make_fake(max_steps=10, post_action_delay=0.0)
    try:
        await env.reset()
        env._config.use_fake = False

        def fake_keyevent(button: str) -> dict:
            return {
                "call": "adb.input.keyevent",
                "args": {"button": button, "keycode": 187},
            }

        monkeypatch.setattr(env, "_exec_system_button_adb", fake_keyevent)

        result = await env.step([
            make_tool_call(
                "system_button",
                {"button": "Recent"},
                call_id="call_recent",
            ),
        ])

        assert result.info["executed_actions"] == [{
            "call": "adb.input.keyevent",
            "args": {"button": "Recent", "keycode": 187},
        }]
        assert result.results[0].tool_call_id == "call_recent"
        assert result.results[0].metadata != {"is_error": True}
        assert result.results[0].images
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_model_action_error_uses_canonical_error_text(monkeypatch):
    env = _make_fake(max_steps=10)
    try:
        await env.reset()

        async def fake_dispatch(_name, _args):
            return [{
                "call": "noop",
                "args": {
                    "name": "tap",
                    "reason": "coordinate must be a list",
                    "is_error": True,
                },
            }]

        monkeypatch.setattr(env, "_dispatch_action", fake_dispatch)

        result = await env.step([
            make_tool_call("tap", {"coordinate": "bad"}, call_id="call_bad"),
        ])

        assert result.results[0].tool_call_id == "call_bad"
        assert result.results[0].metadata == {"is_error": True}
        assert result.results[0].error == (
            "invalid arguments for tap: coordinate must be a list"
        )
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_execution_failure_returns_error_with_current_obs(monkeypatch):
    env = _make_fake(max_steps=10)
    try:
        await env.reset()

        async def fake_dispatch(_name, _args):
            raise RuntimeError("device rpc failed")

        monkeypatch.setattr(env, "_dispatch_action", fake_dispatch)

        result = await env.step([
            make_tool_call("tap", {"coordinate": [500, 500]}, call_id="call_tap"),
        ])

        assert result.terminated is False
        assert result.results[0].tool_call_id == "call_tap"
        assert result.results[0].metadata == {"is_error": True}
        assert result.results[0].error == "tap failed: execution failed"
        assert "device rpc failed" not in result.results[0].error
        assert result.results[0].images
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_malformed_tap_coordinate_returns_error_with_current_obs():
    env = _make_fake(max_steps=10)
    try:
        await env.reset()
        env._config.use_fake = False

        result = await env.step([
            make_tool_call("tap", {}, call_id="call_tap"),
        ])

        assert result.terminated is False
        assert result.results[0].tool_call_id == "call_tap"
        assert result.results[0].metadata == {"is_error": True}
        assert result.results[0].error == (
            "invalid arguments for tap: coordinate is required"
        )
        assert result.results[0].images
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_unsupported_action_returns_error_only_feedback():
    env = _make_fake(max_steps=10)
    try:
        await env.reset()

        result = await env.step([
            make_tool_call("pinch", {
                "coordinate": [500, 500],
                "direction": "in",
                "amount": 25,
            }, call_id="call_pinch"),
        ])

        assert result.info["executed_actions"] == [{
            "call": "noop",
            "args": {"name": "pinch", "reason": "unsupported action: pinch"},
        }]
        assert result.results[0].tool_call_id == "call_pinch"
        assert result.results[0].metadata == {"is_error": True}
        assert result.results[0].error == "unsupported action: pinch"
        assert result.results[0].images == []
    finally:
        await env.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("name,args,expected", [
    (
        "long_press",
        {"coordinate": [500, 500], "duration": 0},
        "long_press.duration must be greater than 0",
    ),
    (
        "swipe",
        {
            "start_coordinate": [500, 700],
            "coordinate": [500, 300],
            "duration": "slow",
        },
        "swipe.duration must be a finite number",
    ),
    (
        "swipe",
        {
            "start_coordinate": [500, 700],
            "coordinate": [500, 300],
            "duration": float("inf"),
        },
        "swipe.duration must be finite",
    ),
    (
        "wait",
        {"duration": -1},
        "wait.duration must be non-negative",
    ),
    (
        "wait",
        {"duration": float("nan")},
        "wait.duration must be finite",
    ),
    (
        "wait",
        {"duration": 31},
        "wait.duration must be <= 30",
    ),
    (
        "long_press",
        {"coordinate": [500, 500], "duration": 6},
        "long_press.duration must be <= 5",
    ),
])
async def test_bad_duration_returns_model_error_with_current_obs(
    name, args, expected,
):
    env = _make_fake(max_steps=10, post_action_delay=0.0)
    try:
        await env.reset()
        env._config.use_fake = False

        result = await env.step([
            make_tool_call(name, args, call_id=f"call_{name}"),
        ])

        assert result.info["executed_actions"][0]["call"] == "noop"
        assert result.results[0].tool_call_id == f"call_{name}"
        assert result.results[0].metadata == {"is_error": True}
        assert result.results[0].error == f"invalid arguments for {name}: {expected}"
        assert result.results[0].images
    finally:
        await env.close()


# ---------------------------------------------------------------------------
# Seed determinism tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_seed_deterministic():
    """Same seed should produce identical task params across resets."""
    params = []
    for _ in range(2):
        env = _make_fake(max_steps=5, seed=42)
        try:
            await env.reset()
            params.append(env._config.metadata.get("task_params"))
        finally:
            await env.close()
    assert params[0] == params[1], f"seed=42 produced different params: {params}"


@pytest.mark.asyncio
async def test_no_seed_random():
    """Without seed, consecutive resets should (almost certainly) differ."""
    params = []
    env = _make_fake(max_steps=5)
    try:
        for _ in range(5):
            await env.reset()
            params.append(env._config.metadata.get("task_params"))
    finally:
        await env.close()
    # At least 2 out of 5 should differ (probability of all same ≈ 0)
    assert len(set(str(p) for p in params)) >= 2, f"No-seed produced identical params: {params}"


@pytest.mark.asyncio
async def test_seed_does_not_pollute_global_random():
    """Seeded env should not affect unseeded env created afterwards."""
    # Run seeded env
    env1 = _make_fake(max_steps=5, seed=42)
    try:
        await env1.reset()
    finally:
        await env1.close()

    # Run two unseeded envs — they should differ
    params = []
    for _ in range(2):
        env = _make_fake(max_steps=5)
        try:
            await env.reset()
            params.append(env._config.metadata.get("task_params"))
        finally:
            await env.close()
    # Not guaranteed to differ in 2 runs, but fake_value is 0-1M so collision is ~1e-6
    # Use str comparison since dicts may have same structure but different values
    assert str(params[0]) != str(params[1]), (
        f"Unseeded envs after seeded env produced same params: {params}"
    )


def test_display_resolution_rejected():
    """androidworld does NOT accept display_resolution: the AVD render size is
    emulator-fixed (not controllable) and coords derive from the native
    screenshot. Accepting it would silently mismatch the real render → misclicks,
    so construction must be a hard error (parity with browsergym's rejection).

    Enforced by explicit constructor/bind signatures: ``display_resolution`` is
    in neither signature, so Python parameter binding TypeErrors."""
    import inspect
    assert "display_resolution" not in inspect.signature(
        AndroidWorldEnv.__init__).parameters
    assert "display_resolution" not in inspect.signature(
        AndroidWorldEnv.bind).parameters
    with pytest.raises(TypeError, match="display_resolution"):
        AndroidWorldEnv(
            config=AndroidWorldTaskConfig(),
            display_resolution=[1920, 1080],
        )


# ── one frame per executed action ───────────────────────────────────────────

def _install_distinct_frames(env: AndroidWorldEnv) -> list[int]:
    """Make ``get_state`` return a DIFFERENT screen on every call.

    The fake env's constant black frame cannot tell "N real captures" from
    "one cached frame emitted N times", which is the failure this guards.
    """
    from types import SimpleNamespace

    import numpy as np

    shots: list[int] = []

    def get_state(wait_to_stabilize: bool = False):
        shots.append(len(shots) + 1)
        return SimpleNamespace(
            pixels=np.full((4, 4, 3), len(shots) * 10, dtype=np.uint8),
            ui_elements=[],
            forest=None,
        )

    env._env.get_state = get_state
    return shots


@pytest.mark.asyncio
async def test_action_batch_returns_one_distinct_frame_per_action():
    """An N-action batch returns N frames, one per action, in action order."""
    env = _make_fake(max_steps=10)
    try:
        await env.reset()
        shots = _install_distinct_frames(env)

        result = await env.step([
            make_tool_call(
                "mobile",
                {"actions": [
                    {"action": "tap", "coordinate": [100, 100]},
                    {"action": "tap", "coordinate": [200, 200]},
                    {"action": "tap", "coordinate": [300, 300]},
                ]},
                call_id="call_batch",
            ),
        ])

        images = result.results[0].images
        assert len(images) == 3, "one frame per executed action"
        assert len(set(images)) == 3, "frames must be real per-action captures"
        assert len(shots) == 3, "one get_state per action, none spare"
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_read_only_screenshot_action_still_gets_its_own_frame():
    """No exception list: ``screenshot`` executes and therefore owes a frame,
    so the frame count stays a pure function of how many actions ran."""
    env = _make_fake(max_steps=10)
    try:
        await env.reset()
        _install_distinct_frames(env)

        result = await env.step([
            make_tool_call(
                "mobile",
                {"actions": [
                    {"action": "tap", "coordinate": [100, 100]},
                    {"action": "screenshot"},
                ]},
                call_id="call_batch",
            ),
        ])

        images = result.results[0].images
        assert len(images) == 2
        assert len(set(images)) == 2
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_zero_executed_actions_still_returns_one_frame():
    """Nothing ran, but the turn still owes the model a current observation."""
    env = _make_fake(max_steps=10)
    try:
        await env.reset()
        shots = _install_distinct_frames(env)

        result = await env.step([
            make_tool_call("mobile", {"actions": []}, call_id="call_empty"),
        ])

        assert len(result.results[0].images) == 1
        assert len(shots) == 1
    finally:
        await env.close()
