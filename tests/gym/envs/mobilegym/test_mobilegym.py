"""Tests for the MobileGym CUA-Lite gym environment (container-only, Model B).

The host imports ZERO ``bench_env``: task metadata is read from the static
manifest (``lite/gym/envs/mobilegym/data/tasks.json``) and the container
resolves the live task class from the ``task_id``. These tests target the host
side only — the thin RPC client :class:`RemoteMobileGymEnv`, the manifest-driven
registration, and the config plumbing. The in-container browser pool + task
lifecycle live in ``docker/server.py`` and are exercised by the live container,
not here.

Run:
    uv run python -m pytest tests/gym/envs/mobilegym/test_mobilegym.py -v
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

from lite.core.messages.final import make_no_tool_call_final_actions
from lite.core.tools import make_tool_call
from lite.core.tools.schemas import tool_schema_name, tool_schema_parameters

# ---------------------------------------------------------------------------
# Import-time tests (no browser, no container)
# ---------------------------------------------------------------------------


def test_import_main():
    """main.py imports successfully and exposes the RPC client + manifest loader."""
    from lite.gym.envs.mobilegym.main import RemoteMobileGymEnv, _load_tasks_json

    assert RemoteMobileGymEnv is not None
    tasks = _load_tasks_json()
    eval_ids = [tid for tid, m in tasks.items() if m["split"] == "eval"]
    train_ids = [tid for tid, m in tasks.items() if m["split"] == "train"]
    assert len(eval_ids) == 256
    assert len(train_ids) == 160


@pytest.fixture
def _direct_mode():
    import os

    old = os.environ.pop("CUA_LITE_ENV_SERVER_URL", None)
    try:
        yield
    finally:
        if old is not None:
            os.environ["CUA_LITE_ENV_SERVER_URL"] = old


def test_task_registration(_direct_mode):
    import lite.gym as gym

    ids = gym.registry.task_ids("mobilegym")
    assert len(ids["eval"]) == 256
    assert len(ids["train"]) == 160


def test_gym_make_constructs_env(monkeypatch, _direct_mode):
    """gym.make spawns the shared container (ensure). Mock ensure + point
    MOBILEGYM_RPC_URL at a stub so no docker is touched; the entry_point must
    still construct a RemoteMobileGymEnv with the right manifest metadata."""
    import lite.gym as gym
    from lite.gym.envs.mobilegym.main import MobileGymContainerServices

    monkeypatch.setattr(MobileGymContainerServices, "ensure", lambda self, env_id: None)
    monkeypatch.setenv("MOBILEGYM_RPC_URL", "http://localhost:65535")

    env = gym.make("mobilegym@bilibili.OpenRankingTask", max_steps=3)
    assert type(env).__name__ in ("RemoteMobileGymEnv", "LiteEnvClient") or hasattr(env, "metadata")
    assert env.metadata.platform == "mobile"
    assert env.metadata.task_type == "use"
    assert env.metadata.others["task_id"] == "bilibili.OpenRankingTask"


def test_container_shutdown_evicts_singleton_caches(monkeypatch):
    import importlib

    import lite.gym.envs.mobilegym.main as m

    registry_mod = importlib.import_module("lite.gym.registry")
    env_id = "mobilegym"
    monkeypatch.setattr(m, "_services_started", {env_id})
    monkeypatch.setenv("MOBILEGYM_RPC_URL", "http://localhost:7666")
    registry_mod._services_started.add(env_id)
    monkeypatch.setattr(
        "lite.gym.remote.reaper.docker_rm_f",
        lambda *args, **kwargs: 1,
    )

    m.MobileGymContainerServices().shutdown(env_id, SimpleNamespace(server_port=1234))

    assert env_id not in m._services_started
    assert env_id not in registry_mod._services_started
    assert "MOBILEGYM_RPC_URL" not in os.environ


def test_metadata_fields(_direct_mode):
    import lite.gym as gym
    from lite.gym.registry import _specs

    gym.registry.task_ids("mobilegym")

    key = "mobilegym@crossapp_life.RailwayEarliestGTrainToWechat"
    spec = _specs[key]
    m = spec.metadata
    assert m.platform == "mobile"
    assert m.task_type == "use"
    others = m.others
    assert others["difficulty"] == "L4"
    assert "railway12306" in others["task_apps"]  # per-task apps this cross-app task involves
    assert "wechat" in others["task_apps"]


# ---------------------------------------------------------------------------
# Tunable config defaults + overrides (host yaml → CFG → -e into the container)
# ---------------------------------------------------------------------------


@pytest.fixture
def _reimport_mobilegym_main():
    import importlib
    import os

    from lite.gym.utils import config as env_config

    mod_name = "lite.gym.envs.mobilegym.main"
    sys.modules.pop(mod_name, None)
    yield
    # Restore a CLEAN module for later (non-fixture) readers. ``monkeypatch`` tears
    # down AFTER this fixture (reverse param order), so a test's ``MOBILEGYM_CONFIG``
    # override is still live here — drop it + the lru_cache before re-importing, else
    # the module-level ``CFG`` gets pinned to the override (e.g. a ``headless`` key)
    # and pollutes tests like ``test_every_config_env_kwarg_is_a_constructor_param``.
    sys.modules.pop(mod_name, None)
    os.environ.pop("MOBILEGYM_CONFIG", None)
    env_config.load.cache_clear()
    importlib.import_module(mod_name)


def test_default_contexts_per_browser_is_8(monkeypatch, _reimport_mobilegym_main):
    """Chromium rendering note section 2: Chromium serializes rendering across pages; 6-8 is the
    cleanly-handled range. The host reads it from the yaml and forwards it to the
    container via -e MOBILEGYM_CONTEXTS_PER_BROWSER."""
    import importlib

    monkeypatch.delenv("MOBILEGYM_CONTEXTS_PER_BROWSER", raising=False)
    main = importlib.import_module("lite.gym.envs.mobilegym.main")
    assert main._CONTEXTS_PER_BROWSER == 8


def test_config_override_replaces_whole_config(monkeypatch, tmp_path, _reimport_mobilegym_main):
    """Per-key MOBILEGYM_* env-vars are retired; override the WHOLE config
    via MOBILEGYM_CONFIG=<path> (resolved by env_config.load, lru_cached)."""
    import importlib

    from lite.gym.utils import config as env_config

    override = tmp_path / "override.yaml"
    override.write_text(
        "env_var_prefix: MOBILEGYM\n"
        "env_kwargs: {max_steps: null, post_action_delay: 0.8, eval_mode: text, "
        "reward_shaping: false, "
        "display_resolution: [1080, 2400], dpr: 3.0, seed: null, extra_tools: [], "
        "headless: true}\n"
        "server_kwargs: {max_browsers: 0, contexts_per_browser: 4, "
        "idle_browser_ttl_s: 300.0, pool_wait_timeout_s: 60.0, "
        "launch_timeout_s: 60.0, rm_timeout_s: 60.0, port: 4173, url: ''}\n"
    )
    monkeypatch.setenv("MOBILEGYM_CONFIG", str(override))
    env_config.load.cache_clear()
    try:
        main = importlib.import_module("lite.gym.envs.mobilegym.main")
        assert main._CONTEXTS_PER_BROWSER == 4
    finally:
        env_config.load.cache_clear()


def test_default_max_browsers_uses_yaml_pin(monkeypatch, _reimport_mobilegym_main):
    """The bundled default pins the pool high for large rollout hosts.

    ``max_browsers: 0`` still means auto-derive; the shipped default is a
    deliberate non-zero pin that should not be clamped by the derivation path.
    """
    import importlib

    main = importlib.import_module("lite.gym.envs.mobilegym.main")
    assert main._MAX_BROWSERS == 128


def test_config_override_auto_derives_max_browsers(monkeypatch, tmp_path, _reimport_mobilegym_main):
    """``max_browsers: 0`` is HostCapacity-derived and clamped to [4, 32]."""
    import importlib

    from lite.gym.utils import config as env_config

    override = tmp_path / "override.yaml"
    override.write_text(
        "env_var_prefix: MOBILEGYM\n"
        "env_kwargs: {max_steps: null, post_action_delay: 0.8, eval_mode: text, "
        "reward_shaping: false, "
        "display_resolution: [1080, 2400], dpr: 3.0, seed: null, extra_tools: [], "
        "headless: true}\n"
        "server_kwargs: {max_browsers: 0, contexts_per_browser: 8, "
        "idle_browser_ttl_s: 300.0, pool_wait_timeout_s: 60.0, "
        "launch_timeout_s: 60.0, rm_timeout_s: 60.0, port: 4173, url: ''}\n"
    )
    monkeypatch.setenv("MOBILEGYM_CONFIG", str(override))
    env_config.load.cache_clear()
    try:
        main = importlib.import_module("lite.gym.envs.mobilegym.main")
        assert 4 <= main._MAX_BROWSERS <= 32
    finally:
        env_config.load.cache_clear()


def test_config_override_pins_max_browsers(monkeypatch, tmp_path, _reimport_mobilegym_main):
    """A non-zero ``max_browsers`` in the override config pins the cap
    (0 = auto-derive); per-key MOBILEGYM_MAX_BROWSERS is retired."""
    import importlib

    from lite.gym.utils import config as env_config

    override = tmp_path / "override.yaml"
    override.write_text(
        "env_var_prefix: MOBILEGYM\n"
        "env_kwargs: {max_steps: null, post_action_delay: 0.8, eval_mode: text, "
        "reward_shaping: false, "
        "display_resolution: [1080, 2400], dpr: 3.0, seed: null, extra_tools: [], "
        "headless: true}\n"
        "server_kwargs: {max_browsers: 7, contexts_per_browser: 8, "
        "idle_browser_ttl_s: 300.0, pool_wait_timeout_s: 60.0, "
        "launch_timeout_s: 60.0, rm_timeout_s: 60.0, port: 4173, url: ''}\n"
    )
    monkeypatch.setenv("MOBILEGYM_CONFIG", str(override))
    env_config.load.cache_clear()
    try:
        main = importlib.import_module("lite.gym.envs.mobilegym.main")
        assert main._MAX_BROWSERS == 7
    finally:
        env_config.load.cache_clear()


def test_default_idle_browser_ttl_is_300s(monkeypatch, _reimport_mobilegym_main):
    import importlib

    monkeypatch.delenv("MOBILEGYM_IDLE_BROWSER_TTL_S", raising=False)
    main = importlib.import_module("lite.gym.envs.mobilegym.main")
    assert main._IDLE_BROWSER_TTL_S == 300.0


def test_default_launch_timeout_is_60s(monkeypatch, _reimport_mobilegym_main):
    import importlib

    monkeypatch.delenv("MOBILEGYM_LAUNCH_TIMEOUT_S", raising=False)
    main = importlib.import_module("lite.gym.envs.mobilegym.main")
    assert main._LAUNCH_TIMEOUT_S == 60.0


# ---------------------------------------------------------------------------
# RemoteMobileGymEnv — construct from manifest-style kwargs (NOT a task_cls)
# ---------------------------------------------------------------------------


def _remote(task_id, *, answer_fields, base_max_steps, **kwargs):
    """Build a RemoteMobileGymEnv with manifest-derived fields (the shape
    _make_remote_env forwards from data/tasks.json)."""
    from lite.gym.envs.mobilegym.main import RemoteMobileGymEnv

    defaults = dict(
        difficulty="L2",
        scope="S1",
        objective="operate",
        composition="atomic",
        capabilities=["operate"],
        task_apps=["clock"],
        answer_fields=answer_fields,
        base_max_steps=base_max_steps,
    )
    defaults.update(kwargs)
    return RemoteMobileGymEnv(task_id=task_id, **defaults)


def _seed_current_observation(env) -> None:
    env._last_observation_image = b"current-mobile-png"
    env._last_observation_text = "current mobile instruction"


#: What the container returns for a step whose calls were all rejected host-side.
_FRESH_PNG = b"fresh-mobile-png"


def _expect_host_rejected_step(**overrides):
    """``_post`` double for a step whose calls were ALL rejected host-side.

    The host still POSTs ``/step`` with an EMPTY action list, and that empty
    list is the assertion that matters -- it proves the rejected call never
    reached the backend for execution, which is what these tests are about.

    The RPC itself must happen: ``inst.step_count`` in docker/server.py is the
    only step counter AND the only place the evaluator runs, so a host-side
    early return desynced the budget and left a truncating step with
    ``reward=None`` -- a finished trajectory scored as nothing.
    """
    import base64

    def _post(path, body):
        if path != "/step":
            raise AssertionError(f"unexpected RPC call: {path} {body}")
        assert body["actions"] == [], "a host-rejected action must not be forwarded to the backend"
        return _step_resp(screenshots_b64=[base64.b64encode(_FRESH_PNG).decode()], **overrides)

    return _post


def _tool_schema(env, name: str) -> dict:
    return next(s for s in env.metadata.extra_tool_schemas if tool_schema_name(s) == name)


def _mobile_action_schema_enum(metadata) -> list[str]:
    from lite.agents.core.action_space.base import LiteMobileActionSpace

    schemas = LiteMobileActionSpace.get_tool_schemas()
    if metadata.valid_actions is not None:
        schemas = LiteMobileActionSpace.filter_child_action_enum(
            schemas,
            metadata.valid_actions,
        )
    mobile = next(s for s in schemas if tool_schema_name(s) == "mobile")
    action_schema = tool_schema_parameters(mobile)["properties"]["actions"]["items"]
    return action_schema["properties"]["action"]["enum"]


@pytest.mark.parametrize(
    "name,args,expected",
    [
        (
            "long_press",
            {"coordinate": [500, 500], "duration": 0},
            "long_press.duration must be greater than 0",
        ),
        (
            "long_press",
            {"coordinate": [500, 500], "duration": 6},
            "long_press.duration must be <= 5",
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
            "drag",
            {
                "start_coordinate": [500, 700],
                "coordinate": [500, 300],
                "duration": "slow",
            },
            "drag.duration must be a finite number",
        ),
        (
            "wait",
            {"duration": float("nan")},
            "wait.duration must be finite",
        ),
        (
            "wait",
            {"duration": -1},
            "wait.duration must be non-negative",
        ),
        (
            "wait",
            {"duration": 31},
            "wait.duration must be <= 30",
        ),
    ],
)
def test_mobilegym_container_duration_validation_mirrors_shared_caps(
    monkeypatch,
    name,
    args,
    expected,
):
    server = _import_mobilegym_container_server(monkeypatch)

    with pytest.raises(ValueError) as exc:
        server._translate_action(name, args)

    assert str(exc.value) == expected


def test_mobilegym_container_duration_caps_match_host_table_by_value(monkeypatch):
    """The in-container copy must agree with the host table on VALUES.

    The parametrized behavioural test above pins the boundaries indirectly (via
    error strings), so it is the only one of the three container mirrors with
    real value coverage. Assert the table directly too, so all three containers
    are guarded the same way.
    """
    from lite.core.tools.action_space.duration import (
        ACTION_SCHEMA_DURATION_CAPS_SECONDS,
    )
    from lite.gym.utils.backend.model_inputs import DEFAULT_MODEL_DURATION_CAP_SECONDS

    server = _import_mobilegym_container_server(monkeypatch)

    assert server._MODEL_DURATION_CAPS_SECONDS == dict(ACTION_SCHEMA_DURATION_CAPS_SECONDS)
    assert server._DEFAULT_MODEL_DURATION_CAP_SECONDS == DEFAULT_MODEL_DURATION_CAP_SECONDS


def test_mobilegym_container_positive_duration_contract(monkeypatch):
    server = _import_mobilegym_container_server(monkeypatch)

    long_press = server._translate_action(
        "long_press",
        {"coordinate": [100, 200], "duration": 1.25},
    )
    assert long_press.action_type == server.ActionType.LONG_PRESS
    assert long_press.data == {"point": [100, 200], "duration": 1250}

    wait = server._translate_action("wait", {"duration": 2.5})
    assert wait.action_type == server.ActionType.WAIT
    assert wait.data == {"value": 2.5}

    swipe = server._translate_action(
        "swipe",
        {
            "start_coordinate": [100, 200],
            "coordinate": [300, 400],
            "duration": 1.25,
        },
    )
    assert swipe.action_type == server.ActionType.SWIPE
    assert swipe.data == {
        "point1": [100, 200],
        "point2": [300, 400],
    }
    assert "duration" not in swipe.data

    drag = server._translate_action(
        "drag",
        {
            "start_coordinate": [100, 200],
            "coordinate": [300, 400],
            "duration": 1.25,
        },
    )
    assert drag.action_type == server.ActionType.DRAG
    assert drag.data == {
        "point1": [100, 200],
        "point2": [300, 400],
    }
    assert "duration" not in drag.data


def test_eval_mode_default_is_text():
    env = _remote("clock.CountAlarms", answer_fields=True, base_max_steps=15)
    assert env._eval_mode == "text"


def test_text_mode_skips_answersheet_bonus_for_answer_fields_task():
    """A task that declares answer_fields gets NO +15 bonus in text mode
    (mirrors upstream where +15 is gated on eval_mode == 'grounded')."""
    env = _remote(
        "clock.CountAlarms",
        answer_fields=True,
        base_max_steps=15,
        max_steps=15,
        eval_mode="text",
    )
    assert env._max_steps == 15
    assert env.max_steps == 15


def test_grounded_mode_adds_answersheet_bonus_for_answer_fields_task():
    env = _remote(
        "clock.CountAlarms",
        answer_fields=True,
        base_max_steps=15,
        max_steps=15,
        eval_mode="grounded",
    )
    # Raw stored budget is the caller's value; the +15 grounded bonus is
    # derived at read time by the ``max_steps`` property.
    assert env._max_steps == 15
    assert env.max_steps == 15 + 15


def test_grounded_mode_no_bonus_for_task_without_answer_fields():
    """Non-AnswerSheet tasks (answer_fields False) get no bonus even in
    grounded mode — mirrors upstream's `and task.answer_fields` guard."""
    env = _remote(
        "clock.AddAlarm",
        answer_fields=False,
        base_max_steps=30,
        max_steps=20,
        eval_mode="grounded",
    )
    assert env._max_steps == 20
    assert env.max_steps == 20


def test_max_steps_null_falls_back_to_manifest_base():
    """max_steps=None → the property uses the manifest-supplied base_max_steps."""
    env = _remote(
        "clock.AddAlarm",
        answer_fields=False,
        base_max_steps=30,
        max_steps=None,
        eval_mode="text",
    )
    assert env.max_steps == 30
    # grounded but no answer_fields → still just the base
    env2 = _remote(
        "clock.AddAlarm",
        answer_fields=False,
        base_max_steps=30,
        max_steps=None,
        eval_mode="grounded",
    )
    assert env2.max_steps == 30


def test_every_config_env_kwarg_is_a_constructor_param():
    """Systemic guard: every overridable ``env_kwargs`` key in default.yaml must
    map to a RemoteMobileGymEnv ``__init__`` parameter. Otherwise a per-rollout
    override (gym.make(..., <key>=...)) raises TypeError (the ``dpr`` bug) or is
    silently dropped (the ``headless`` bug). This is the host→container config
    contract: nothing advertised as a knob may be unwired."""
    import inspect

    from lite.gym.envs.mobilegym.main import CFG, RemoteMobileGymEnv

    params = set(inspect.signature(RemoteMobileGymEnv.__init__).parameters)
    for key in CFG.env_kwargs:
        assert key in params, (
            f"env_kwarg '{key}' has no RemoteMobileGymEnv.__init__ param — a "
            f"per-run override would crash or be ignored. Add the param + forward it."
        )


def test_reset_body_forwards_overridable_budget_and_viewport():
    """reset() must send the host-resolved max_steps (override + grounded bonus)
    and the per-instance dpr to the container — else the container recomputes /
    uses its own default and the override is lost (the max_steps/dpr bugs)."""
    import asyncio

    env = _remote(
        "clock.CountAlarms",
        answer_fields=True,
        base_max_steps=15,
        max_steps=20,
        eval_mode="grounded",
        dpr=2.5,
    )
    captured = {}

    def _fake_post(path, body):
        captured["path"] = path
        captured["body"] = body
        return {
            "instance_id": "iid",
            "screenshot_b64": "",
            "instruction": "",
            "max_steps": body["max_steps"],
        }

    env._post = _fake_post
    asyncio.run(env.reset())
    assert captured["path"] == "/reset"
    # grounded + answer_fields → 20 + 15 bonus, forwarded verbatim
    assert captured["body"]["max_steps"] == 35
    assert captured["body"]["dpr"] == 2.5


def test_response_step_terminates_with_canonical_call():
    """Canonical response calls end the host-side episode even when text-mode
    MobileGym ANSWER itself returns terminated=False."""
    import asyncio

    env = _remote(
        "clock.CountAlarms",
        answer_fields=True,
        base_max_steps=15,
        extra_tools=["response"],
    )
    env._instance_id = "iid"
    action = make_tool_call("response", {"text": "done"}, call_id="call_response")
    captured = {}

    def _fake_post(path, body):
        captured["path"] = path
        captured["body"] = body
        return {
            "screenshots_b64": [""],
            "reward": 1.0,
            "terminated": False,
            "truncated": False,
            "executed": [{"call": "ANSWER", "args": {"text": "done"}}],
        }

    env._post = _fake_post
    result = asyncio.run(env.step([action]))

    assert captured["path"] == "/step"
    assert captured["body"]["actions"] == [
        {
            "call_id": "call_response",
            "name": "response",
            "arguments": {"text": "done"},
        }
    ]
    assert result.terminated is True
    assert result.reward == 1.0
    # A terminal call gets NO tool result: it ended the episode, so there is no
    # next decision for an observation to inform, and
    # ``devs/migration/verify.py`` refuses a tool result for a terminal call.
    # Note the container reported ``terminated: False`` -- ``agent_finished``
    # is what ends this episode, and it is the same fact the exclusion reads.
    assert result.results == []


def test_step_rejects_flat_lite_boundary_call():
    import asyncio

    env = _remote("clock.AddAlarm", answer_fields=False, base_max_steps=30)
    env._post = lambda _path, _body: pytest.fail("flat boundary call reached RPC")

    with pytest.raises(TypeError, match="env.step expects canonical Lite tool calls"):
        asyncio.run(
            env.step(
                [
                    {"name": "tap", "arguments": {"coordinate": [500, 500]}},
                ]
            )
        )


def test_content_only_final_text_uses_internal_response_no_tool_result():
    import asyncio

    env = _remote(
        "clock.CountAlarms",
        answer_fields=True,
        base_max_steps=15,
    )
    env._instance_id = "iid"
    captured = {}

    def _fake_post(path, body):
        captured["path"] = path
        captured["body"] = body
        return {
            "screenshots_b64": [""],
            "reward": 1.0,
            "terminated": False,
            "truncated": False,
            "executed": [{"call": "ANSWER", "args": {"text": "final text"}}],
            "action_errors": [],
        }

    env._post = _fake_post
    actions = make_no_tool_call_final_actions("final text")
    result = asyncio.run(env.step(actions))

    assert captured["path"] == "/step"
    assert captured["body"]["actions"][0]["name"] == "response"
    assert captured["body"]["actions"][0]["arguments"] == {"text": "final text"}
    assert "call_id" not in captured["body"]["actions"][0]
    assert result.terminated is True
    assert result.results == []
    assert result.reward == 1.0


def test_open_app_catalog_metadata_matches_schema_enum():
    from lite.gym.envs.mobilegym.main import _MOBILEGYM_APPS

    env = _remote(
        "clock.AddAlarm",
        answer_fields=False,
        base_max_steps=30,
        extra_tools=["open_app"],
    )

    assert env.metadata.others["apps"] == _MOBILEGYM_APPS
    assert (
        tool_schema_parameters(_tool_schema(env, "open_app"))["properties"]["app_name"]["enum"]
        == env.metadata.others["apps"]
    )


def test_default_mobile_schema_does_not_advertise_pinch():
    env = _remote("clock.AddAlarm", answer_fields=False, base_max_steps=30)
    enum = _mobile_action_schema_enum(env.metadata)
    assert "pinch" not in enum
    assert "system_button" in enum


def test_active_open_app_executes_when_enabled():
    import asyncio

    env = _remote(
        "clock.AddAlarm",
        answer_fields=False,
        base_max_steps=30,
        extra_tools=["open_app"],
    )
    env._instance_id = "iid"
    action = make_tool_call("open_app", {"app_name": "Settings"}, call_id="call_open")
    captured = {}

    def _fake_post(path, body):
        captured["path"] = path
        captured["body"] = body
        return _step_resp(
            screenshots_b64=[""],
            executed=[{"call": "AWAKE", "args": {"app_name": "Settings"}}],
        )

    env._post = _fake_post
    result = asyncio.run(env.step([action]))

    assert captured["path"] == "/step"
    assert captured["body"]["actions"] == [
        {
            "call_id": "call_open",
            "name": "open_app",
            "arguments": {"app_name": "Settings"},
        }
    ]
    assert result.results[0].tool_call_id == "call_open"
    assert result.results[0].images
    assert result.results[0].metadata != {"is_error": True}


def test_inactive_open_app_returns_error_only_feedback():
    import asyncio

    env = _remote("clock.AddAlarm", answer_fields=False, base_max_steps=30)
    env._instance_id = "iid"
    _seed_current_observation(env)

    env._post = _expect_host_rejected_step()
    result = asyncio.run(
        env.step([make_tool_call("open_app", {"app_name": "Settings"}, call_id="call_open")])
    )

    assert result.results[0].tool_call_id == "call_open"
    assert result.results[0].metadata == {"is_error": True}
    assert result.results[0].error == "open_app is not available in this task."
    assert result.results[0].images == []


def test_invalid_open_app_is_rejected_before_rpc_dispatch():
    import asyncio

    env = _remote(
        "clock.AddAlarm",
        answer_fields=False,
        base_max_steps=30,
        extra_tools=["open_app"],
    )
    env._instance_id = "iid"
    _seed_current_observation(env)

    env._post = _expect_host_rejected_step()
    result = asyncio.run(
        env.step([make_tool_call("open_app", {"app_name": "Not A Real App"}, call_id="call_open")])
    )

    assert result.results[0].tool_call_id == "call_open"
    assert result.results[0].metadata == {"is_error": True}
    assert result.results[0].error.startswith("invalid arguments for open_app: ")
    assert "Not A Real App" in result.results[0].error
    assert result.results[0].images[-1] == _FRESH_PNG
    assert result.truncated is False
    assert result.reward is None


# ---------------------------------------------------------------------------
# Bad model action → PAIRED per-call feedback (never dropped, never a crash)
# ---------------------------------------------------------------------------


def _step_resp(**overrides):
    resp = {
        "screenshots_b64": [""],
        "reward": None,
        "terminated": False,
        "truncated": False,
        "executed": [],
        "action_errors": [],
    }
    resp.update(overrides)
    return resp


def _import_mobilegym_container_server(monkeypatch):
    import importlib
    import types

    server_mod = "lite.gym.envs.mobilegym.docker.server"
    sys.modules.pop(server_mod, None)

    class _Action:
        def __init__(self, action_type, data):
            self.action_type = action_type
            self.data = data

    class _ActionType:
        CLICK = "CLICK"
        DOUBLE_TAP = "DOUBLE_TAP"
        LONG_PRESS = "LONG_PRESS"
        TYPE = "TYPE"
        SWIPE = "SWIPE"
        DRAG = "DRAG"
        AWAKE = "AWAKE"
        BACK = "BACK"
        HOME = "HOME"
        ENTER = "ENTER"
        RECENT = "RECENT"
        WAIT = "WAIT"
        COMPLETE = "COMPLETE"
        ABORT = "ABORT"
        ANSWER = "ANSWER"

    class _Observation:
        def __init__(self, screenshot_bytes=b"mobile-png", route=None, state=None):
            self._screenshot_bytes = screenshot_bytes
            self.route = route
            self.state = state

        def get_screenshot_bytes(self):
            return self._screenshot_bytes

    class _MobileGymEnv:
        @staticmethod
        def get_launch_args(_headless):
            return {"args": []}

    class _JudgeResult:
        @staticmethod
        def error(_message):
            return SimpleNamespace(progress=0.0)

    modules = {
        "bench_env": types.ModuleType("bench_env"),
        "bench_env.env": types.ModuleType("bench_env.env"),
        "bench_env.env.base": types.ModuleType("bench_env.env.base"),
        "bench_env.env.mobile_gym": types.ModuleType("bench_env.env.mobile_gym"),
        "bench_env.task": types.ModuleType("bench_env.task"),
        "bench_env.task.base": types.ModuleType("bench_env.task.base"),
        "bench_env.task.judge": types.ModuleType("bench_env.task.judge"),
        "bench_env.task.registry": types.ModuleType("bench_env.task.registry"),
    }
    modules["bench_env.env.base"].Action = _Action
    modules["bench_env.env.base"].ActionType = _ActionType
    modules["bench_env.env.base"].Observation = _Observation
    modules["bench_env.env.mobile_gym"].MobileGymEnv = _MobileGymEnv
    modules["bench_env.task.base"].BaseTask = object
    modules["bench_env.task.judge"].JudgeInput = SimpleNamespace
    modules["bench_env.task.judge"].JudgeResult = _JudgeResult
    modules["bench_env.task.registry"].TaskRegistry = lambda: SimpleNamespace()
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    server = importlib.import_module(server_mod)
    monkeypatch.delitem(sys.modules, server_mod, raising=False)
    return server


def test_container_step_malformed_action_envelopes_return_typed_action_errors(monkeypatch):
    from fastapi.testclient import TestClient

    server = _import_mobilegym_container_server(monkeypatch)

    class _Env:
        def __init__(self):
            self.executed = []

        async def step(self, action):
            self.executed.append(action)

        async def get_observation(self):
            return SimpleNamespace(get_screenshot_bytes=lambda: b"mobile-png")

    iid = "iid-malformed-mobile-action-envelope"
    env = _Env()
    inst = server._Instance(
        env=env,
        slot={},
        task=SimpleNamespace(),
        task_id="clock.AddAlarm",
        eval_mode="text",
        max_steps=5,
        init_obs=SimpleNamespace(),
    )
    monkeypatch.setitem(server._instances, iid, inst)

    client = TestClient(server.app)
    r = client.post(
        "/step",
        json={
            "instance_id": iid,
            "actions": [
                {"call_id": "missing-name", "arguments": {}},
                {"name": "wait", "call_id": "bad-args", "arguments": ["bad"]},
                "bad-action",
                {"name": "pinch", "call_id": "unsupported", "arguments": {}},
            ],
        },
    )
    assert r.status_code == 200, r.text
    resp = r.json()

    assert env.executed == []
    assert resp["executed"] == []
    assert resp["terminated"] is False
    assert resp["truncated"] is False
    assert [record["index"] for record in resp["action_errors"]] == [0, 1, 2, 3]
    assert [record["kind"] for record in resp["action_errors"]] == [
        "model_action",
        "model_action",
        "model_action",
        "unsupported_action",
    ]
    assert [record.get("call_id") for record in resp["action_errors"]] == [
        "missing-name",
        "bad-args",
        None,
        "unsupported",
    ]
    assert [record["name"] for record in resp["action_errors"]] == [
        "<invalid>",
        "wait",
        "<invalid>",
        "pinch",
    ]
    assert resp["action_errors"][3]["message"] == "unsupported action: pinch"
    assert (
        "Lite tool_call.arguments must be a dict, got list" in (resp["action_errors"][1]["message"])
    )
    assert inst.step_count == 1

    r = client.post(
        "/step",
        json={
            "instance_id": iid,
            "actions": {"name": "wait", "arguments": {}},
        },
    )

    assert r.status_code == 200, r.text
    resp = r.json()
    assert resp["action_errors"] == [
        {
            "index": 0,
            "kind": "model_action",
            "name": "<invalid>",
            "error": "body.actions must be a list, got dict",
            "message": "<invalid>: body.actions must be a list, got dict",
        }
    ]
    assert inst.step_count == 2

    r = client.post("/step", json={"instance_id": "unknown-iid", "actions": []})
    assert r.status_code == 404, r.text


def test_bad_action_arguments_pair_to_originating_call_id():
    """The container reports the offending action by index; the host must turn
    it into a LiteToolResult with is_error on the ORIGINATING call_id. Before
    this, the container logged a warning and the host dropped it entirely — the
    agent saw a normal screenshot and could not tell the action failed."""
    import asyncio

    env = _remote(
        "clock.AddAlarm",
        answer_fields=False,
        base_max_steps=30,
        extra_tools=["response"],
    )
    env._instance_id = "iid"
    actions = [
        make_tool_call(
            "mobile",
            {"actions": [{"action": "tap", "coordinate": [10, 10]}]},
            call_id="call-a",
        ),
        make_tool_call(
            "mobile",
            {"actions": [{"action": "system_button", "button": "Nope"}]},
            call_id="call-b",
        ),
    ]

    def _fake_post(path, body):
        assert path == "/step"
        # index 1 == the system_button action unpacked from ``call-b``.
        return _step_resp(
            action_errors=[
                {
                    "index": 1,
                    "kind": "model_action",
                    "name": "system_button",
                    "error": "system_button: unknown button 'Nope'; expected one of "
                    "['Back', 'Enter', 'Home', 'Menu', 'Recent']",
                    "message": "system_button: system_button: unknown button 'Nope'; "
                    "expected one of ['Back', 'Enter', 'Home', 'Menu', 'Recent']",
                },
            ]
        )

    env._post = _fake_post
    result = asyncio.run(env.step(actions))

    by_id = {r.tool_call_id: r for r in result.results}
    assert by_id["call-a"].metadata is None or not (by_id["call-a"].metadata or {}).get("is_error")
    assert by_id["call-b"].metadata["is_error"] is True
    assert by_id["call-b"].text is None
    assert "unknown button 'Nope'" in by_id["call-b"].error


def test_container_action_error_index_maps_after_host_side_filtering():
    """The container reports indexes in the filtered list the host actually
    sent, not in the original model-emitted list. A bad direct action after an
    inactive extra tool must still pair to its own call_id."""
    import asyncio

    env = _remote("clock.AddAlarm", answer_fields=False, base_max_steps=30)
    env._instance_id = "iid"
    actions = [
        make_tool_call("open_app", {"app_name": "Settings"}, call_id="call-open"),
        make_tool_call("wait", {"duration": -1}, call_id="call-wait"),
    ]
    captured = {}

    def _fake_post(path, body):
        captured["path"] = path
        captured["body"] = body
        assert body["actions"] == [
            {"name": "wait", "call_id": "call-wait", "arguments": {"duration": -1}},
        ]
        return _step_resp(
            action_errors=[
                {
                    "index": 0,
                    "kind": "model_action",
                    "name": "wait",
                    "error": "wait.duration must be non-negative",
                    "message": "wait: wait.duration must be non-negative",
                }
            ]
        )

    env._post = _fake_post
    result = asyncio.run(env.step(actions))

    by_id = {r.tool_call_id: r for r in result.results}
    assert captured["path"] == "/step"
    assert by_id["call-open"].error == "open_app is not available in this task."
    assert by_id["call-open"].metadata == {"is_error": True}
    assert by_id["call-wait"].metadata == {"is_error": True}
    assert by_id["call-wait"].error == (
        "invalid arguments for wait: wait.duration must be non-negative"
    )


def test_container_unsupported_action_error_returns_error_only_feedback():
    import asyncio

    env = _remote("clock.AddAlarm", answer_fields=False, base_max_steps=30)
    env._instance_id = "iid"

    env._post = lambda path, body: _step_resp(
        action_errors=[
            {
                "index": 0,
                "kind": "unsupported_action",
                "name": "wait",
                "error": "unsupported action: wait",
                "message": "unsupported action: wait",
            }
        ]
    )
    result = asyncio.run(env.step([make_tool_call("wait", {"duration": 0}, call_id="call-wait")]))

    assert result.results[0].tool_call_id == "call-wait"
    assert result.results[0].metadata == {"is_error": True}
    assert result.results[0].error == "unsupported action: wait"
    assert result.results[0].images == []
    assert result.results[0].text is None


def test_tool_execution_error_pairs_without_model_argument_prefix():
    import asyncio

    env = _remote(
        "clock.AddAlarm",
        answer_fields=False,
        base_max_steps=30,
        extra_tools=["response"],
    )
    env._instance_id = "iid"
    actions = [make_tool_call("response", {"text": "done"}, call_id="call-response")]

    env._post = lambda path, body: _step_resp(
        action_errors=[
            {
                "index": 0,
                "kind": "tool_execution",
                "name": "response",
                "error": "response side effect failed: answer rejected",
                "message": "response: response side effect failed: answer rejected",
            }
        ],
    )
    result = asyncio.run(env.step(actions))

    assert result.terminated is True
    assert result.results[0].tool_call_id == "call-response"
    assert result.results[0].metadata["is_error"] is True
    assert result.results[0].error == "response failed: execution failed"
    assert "side effect" not in result.results[0].error
    assert not result.results[0].error.startswith("invalid arguments for")


def test_unsupported_action_pairs_to_originating_call_id():
    """A action with no MobileGym mapping is reported as the canonical
    ``unsupported action: <name>`` on the call that emitted it."""
    import asyncio

    env = _remote("clock.AddAlarm", answer_fields=False, base_max_steps=30)
    env._instance_id = "iid"
    _seed_current_observation(env)
    actions = [
        make_tool_call(
            "mobile",
            {"actions": [{"action": "pinch", "coordinate": [500, 500]}]},
            call_id="call-x",
        )
    ]

    env._post = _expect_host_rejected_step()
    result = asyncio.run(env.step(actions))

    assert [r.tool_call_id for r in result.results] == ["call-x"]
    assert result.results[0].images == []
    assert result.results[0].error == "unsupported action: pinch"
    assert result.results[0].metadata["is_error"] is True


def test_unsupported_action_is_rejected_host_side_with_error_only_feedback():
    """Unimplemented mobile actions must not rely on a backend no-op.

    The host owns the feedback and forwards an EMPTY action list, so nothing is
    executed -- but the RPC still happens, because the container owns the step
    counter and the evaluator.
    """
    import asyncio

    env = _remote("clock.AddAlarm", answer_fields=False, base_max_steps=30)
    env._instance_id = "iid"
    _seed_current_observation(env)
    actions = [
        make_tool_call(
            "mobile",
            {
                "actions": [
                    {
                        "action": "pinch",
                        "coordinate": [500, 500],
                        "direction": "in",
                        "amount": 25,
                    }
                ]
            },
            call_id="call-x",
        )
    ]

    env._post = _expect_host_rejected_step()
    result = asyncio.run(env.step(actions))

    assert result.info["executed_actions"] == []
    assert [r.tool_call_id for r in result.results] == ["call-x"]
    assert result.results[0].error == "unsupported action: pinch"
    assert result.results[0].metadata["is_error"] is True
    assert result.results[0].images == []


def test_fully_filtered_final_step_is_still_scored():
    """A step whose only action was filtered host-side must STILL be scored.

    ``pinch`` is not in ``android_supported_actions()``, so this batch empties
    out host-side. If the host short-circuits instead of POSTing ``/step``, the
    container never advances ``inst.step_count`` and never runs its evaluator --
    the episode ends ``truncated`` with ``reward=None`` and a complete
    trajectory is discarded. mobilegym has no host-side evaluator to fall back
    on, so reaching the container is the only way to score.
    """
    import asyncio
    import base64

    env = _remote("clock.AddAlarm", answer_fields=False, base_max_steps=1)
    env._instance_id = "iid"
    _seed_current_observation(env)
    posted: list[dict] = []

    def _post(path, body):
        assert path == "/step"
        posted.append(body)
        # The container is on its LAST turn: it counts this step and evaluates.
        return _step_resp(
            screenshots_b64=[base64.b64encode(_FRESH_PNG).decode()],
            reward=1.0,
            truncated=True,
        )

    env._post = _post
    result = asyncio.run(
        env.step(
            [
                make_tool_call(
                    "mobile",
                    {"actions": [{"action": "pinch", "coordinate": [500, 500]}]},
                    call_id="call-x",
                )
            ]
        )
    )

    assert posted == [{"instance_id": "iid", "actions": []}], (
        "the filtered step must still reach the container -- with nothing to execute"
    )
    assert result.truncated is True
    assert result.reward == 1.0, "a truncating step returning reward=None throws the episode away"
    assert result.results[0].error == "unsupported action: pinch"


def test_container_counts_and_scores_a_step_with_no_actions(monkeypatch):
    """The container side of the same contract.

    An empty action list is what the host sends when it filtered everything.
    The container must treat it as a consumed turn -- advance ``step_count``,
    truncate when the budget is gone, and run the evaluator.
    """
    from fastapi.testclient import TestClient

    server = _import_mobilegym_container_server(monkeypatch)

    class _Env:
        def __init__(self):
            self.executed = []

        async def step(self, action):
            self.executed.append(action)

        async def get_observation(self):
            return SimpleNamespace(get_screenshot_bytes=lambda: b"mobile-png")

    async def _fake_evaluate(inst, terminated):
        return 1.0

    monkeypatch.setattr(server, "_evaluate", _fake_evaluate)

    iid = "iid-empty-actions"
    env = _Env()
    inst = server._Instance(
        env=env,
        slot={},
        task=SimpleNamespace(),
        task_id="clock.AddAlarm",
        eval_mode="text",
        max_steps=1,
        init_obs=SimpleNamespace(),
    )
    monkeypatch.setitem(server._instances, iid, inst)

    client = TestClient(server.app)
    r = client.post(
        "/step",
        json={
            "instance_id": iid,
            "actions": [],
        },
    )
    assert r.status_code == 200, r.text
    resp = r.json()

    assert env.executed == [], "nothing to execute -- that is the point"
    assert inst.step_count == 1, "an empty step is still a consumed turn"
    assert resp["truncated"] is True
    assert resp["reward"] == 1.0, "the truncating step must be evaluated"


def test_recent_system_button_pairs_to_originating_call_id():
    import asyncio

    env = _remote("clock.AddAlarm", answer_fields=False, base_max_steps=30)
    env._instance_id = "iid"
    actions = [
        make_tool_call(
            "mobile",
            {"actions": [{"action": "system_button", "button": "Recent"}]},
            call_id="call-recent",
        )
    ]

    captured = {}

    def _fake_post(path, body):
        captured["path"] = path
        captured["body"] = body
        return _step_resp(
            executed=[
                {
                    "call": "RECENT",
                    "args": {"button": "Recent"},
                }
            ],
        )

    env._post = _fake_post
    result = asyncio.run(env.step(actions))

    assert captured["path"] == "/step"
    assert captured["body"]["actions"] == [
        {"name": "system_button", "arguments": {"button": "Recent"}}
    ]
    assert result.results[0].tool_call_id == "call-recent"
    assert result.results[0].metadata is None or not result.results[0].metadata.get("is_error")
    assert result.results[0].error is None


def test_invalid_valid_action_error_hides_valid_actions_list_from_model_visible_text():
    import asyncio

    env = _remote(
        "clock.AddAlarm",
        answer_fields=False,
        base_max_steps=30,
        valid_actions=["swipe"],
    )
    env._instance_id = "iid"
    _seed_current_observation(env)

    env._post = _expect_host_rejected_step()
    result = asyncio.run(
        env.step(
            [
                make_tool_call(
                    "mobile",
                    {"actions": [{"action": "tap", "coordinate": [10, 20]}]},
                    call_id="call_tap",
                )
            ]
        )
    )

    assert result.results[0].tool_call_id == "call_tap"
    assert result.results[0].metadata == {"is_error": True}
    assert result.results[0].error == (
        "invalid action: tap; choose an available action for this task"
    )
    assert "valid_actions" not in result.results[0].error
    assert "['swipe']" not in result.results[0].error


def test_malformed_tap_action_error_returns_current_image():
    import asyncio

    env = _remote("clock.AddAlarm", answer_fields=False, base_max_steps=30)
    env._instance_id = "iid"

    def _fake_post(path, body):
        assert path == "/step"
        assert body["actions"] == [
            {
                "call_id": "call_tap",
                "name": "tap",
                "arguments": {},
            }
        ]
        return _step_resp(
            screenshots_b64=[""],
            action_errors=[
                {
                    "index": 0,
                    "kind": "model_action",
                    "name": "tap",
                    "error": "malformed normalized coordinate: None",
                    "message": "tap: malformed normalized coordinate: None",
                }
            ],
        )

    env._post = _fake_post
    result = asyncio.run(env.step([make_tool_call("tap", {}, call_id="call_tap")]))

    assert result.results[0].tool_call_id == "call_tap"
    assert result.results[0].metadata == {"is_error": True}
    assert result.results[0].error == ("invalid arguments for tap: coordinate is required")
    assert result.results[0].images


def test_unknown_standalone_tool_returns_error_only():
    import asyncio

    env = _remote("clock.AddAlarm", answer_fields=False, base_max_steps=30)
    env._instance_id = "iid"

    env._post = _expect_host_rejected_step()
    result = asyncio.run(env.step([make_tool_call("foo", {}, call_id="call_foo")]))

    assert result.results[0].tool_call_id == "call_foo"
    assert result.results[0].metadata == {"is_error": True}
    assert result.results[0].error == "unknown tool: foo"
    assert result.results[0].images == []
    assert result.results[0].text is None
    assert result.truncated is False
    assert result.reward is None


def test_container_infra_failure_still_raises():
    """A dead container / RPC failure is INFRA: it must keep hard-raising out of
    step(), never be downgraded into agent-visible action feedback."""
    import asyncio

    env = _remote("clock.AddAlarm", answer_fields=False, base_max_steps=30)
    env._instance_id = "iid"

    def _fake_post(path, body):
        raise RuntimeError("mobilegym container /step returned 500: boom")

    env._post = _fake_post
    with pytest.raises(RuntimeError, match="returned 500"):
        asyncio.run(
            env.step(
                [
                    make_tool_call(
                        "mobile",
                        {"actions": [{"action": "tap", "coordinate": [10, 10]}]},
                        call_id="call-x",
                    )
                ]
            )
        )


def test_metadata_shape_from_manifest_fields():
    """metadata is a pure function of the stored manifest fields (no task_cls)."""
    from lite.gym.envs.mobilegym.main import _MOBILEGYM_APPS

    env = _remote(
        "crossapp_life.RailwayEarliestGTrainToWechat",
        answer_fields=False,
        base_max_steps=45,
        difficulty="L4",
        scope="S2",
        objective="operate",
        composition="transfer",
        capabilities=["operate", "search"],
        task_apps=["railway12306", "wechat"],
    )
    m = env.metadata
    assert m.platform == "mobile"
    assert m.task_type == "use"
    # task_id is framework-injected identity (gym.make / registry key) —
    # a directly-constructed env never writes it (same-source contract).
    assert "task_id" not in m.others
    assert m.others["task_name"] == "RailwayEarliestGTrainToWechat"
    assert m.others["suite"] == "crossapp_life"
    assert m.others["difficulty"] == "L4"
    assert m.others["scope"] == "S2"
    assert m.others["composition"] == "transfer"
    # Recovered by the ONE builder: these registration-side fields now show
    # up on live metadata too (they were dropped by the old dual-source
    # property).
    assert m.others["capabilities"] == ["operate", "search"]
    assert m.others["task_apps"] == ["railway12306", "wechat"]
    assert "needs_answer_sheet" not in m.others  # answer_fields=False
    assert m.others["apps"] == _MOBILEGYM_APPS
    assert m.others["max_steps"] == env.max_steps


# ---------------------------------------------------------------------------
# Registration: split / seed / max_steps base / needs_answer_sheet
# ---------------------------------------------------------------------------


def test_registry_max_steps_is_text_mode_base(_direct_mode):
    """metadata.others['max_steps'] should be the text-mode base (no +15) so
    that grounded callers must opt in via eval_mode."""
    import lite.gym as gym
    from lite.gym.registry import _specs

    gym.registry.task_ids("mobilegym")
    # clock.CountAlarms: manifest max_steps=15, answer_fields True.
    spec = _specs["mobilegym@clock.CountAlarms"]
    assert spec.metadata.others["max_steps"] == 15
    assert spec.metadata.others["needs_answer_sheet"] is True


def test_no_needs_answer_sheet_for_non_answer_task(_direct_mode):
    import lite.gym as gym
    from lite.gym.registry import _specs

    gym.registry.task_ids("mobilegym")
    spec = _specs["mobilegym@clock.AddAlarm"]
    assert "needs_answer_sheet" not in spec.metadata.others


def test_eval_split_registers_seed_42(_direct_mode):
    """eval-split specs carry seed=42 as a default kwarg; train-split omit it."""
    import lite.gym as gym
    from lite.gym.registry import _specs

    gym.registry.task_ids("mobilegym")
    eval_spec = _specs["mobilegym@clock.CountAlarms"]  # eval split
    assert eval_spec.kwargs.get("seed") == 42
    train_spec = _specs["mobilegym@clock.AddAlarm"]  # train split
    assert "seed" not in train_spec.kwargs


def test_difficulty_distribution(_direct_mode):
    import lite.gym as gym
    from lite.gym.registry import _specs, _splits

    gym.registry.task_ids("mobilegym")
    eval_task_ids = set(_splits.get("mobilegym", {}).get("eval", []))
    counts: dict[str, int] = {}
    for key in (k for k in _specs if k.startswith("mobilegym@")):
        tid = key.split("@", 1)[1]
        if tid not in eval_task_ids:
            continue
        d = _specs[key].metadata.others["difficulty"]
        counts[d] = counts.get(d, 0) + 1
    assert sum(counts.values()) == 256
    assert set(counts) <= {"L1", "L2", "L3", "L4"}


# ---------------------------------------------------------------------------
# Per-action result frames
#
# mobilegym POSTs the WHOLE batch to the container in one ``/step``, so the
# action loop -- and therefore the only place that can see the screen BETWEEN
# two actions of one batch -- lives in ``docker/server.py``. The container
# returns ``screenshots_b64`` (one frame per executed action, in action order)
# and the host turns it into the per-action ``images`` of the tool result.
# ---------------------------------------------------------------------------


def _distinct_frames(prefix: str, count: int) -> list[str]:
    import base64

    return [base64.b64encode(f"{prefix}-{index}".encode()).decode() for index in range(count)]


def test_action_batch_result_carries_one_image_per_executed_action():
    """N executed actions -> N DISTINCT frames on the batch's single result.

    Distinctness is the assertion that matters: re-emitting one frame N times
    would satisfy a length check while carrying no information at all.
    """
    import asyncio
    import base64

    frames = _distinct_frames("mobile-frame", 3)
    env = _remote("clock.AddAlarm", answer_fields=False, base_max_steps=30)
    env._instance_id = "iid"

    def _fake_post(path, body):
        assert path == "/step"
        assert [action["name"] for action in body["actions"]] == ["tap", "tap", "tap"]
        return _step_resp(screenshots_b64=frames)

    env._post = _fake_post
    result = asyncio.run(
        env.step(
            [
                make_tool_call(
                    "mobile",
                    {
                        "actions": [
                            {"action": "tap", "coordinate": [100, 100]},
                            {"action": "tap", "coordinate": [200, 200]},
                            {"action": "tap", "coordinate": [300, 300]},
                        ]
                    },
                    call_id="call_batch",
                )
            ]
        )
    )

    assert len(result.results) == 1
    images = list(result.results[0].images)
    assert images == [base64.b64decode(frame) for frame in frames]
    assert len(set(images)) == 3


def test_a_host_rejected_slot_is_re_interleaved_in_slot_order(monkeypatch):
    """R3 for a container-batched env: N slots in, N frames out.

    A host-rejected action never reaches the container, so no frame comes back
    for it. The host cannot capture mid-batch either -- the container runs the
    whole batch in ONE RPC -- so the rejected slot repeats the frame of the
    action before it, re-interleaved at its own position. The repeat is honest:
    that slot changed nothing.
    """
    import asyncio

    env = _remote("clock.AddAlarm", answer_fields=False, base_max_steps=30)
    env._instance_id = "iid"
    frames = _distinct_frames("mg", 2)
    env._post = lambda path, body: _step_resp(screenshots_b64=frames)

    result = asyncio.run(
        env.step(
            [
                make_tool_call(
                    "mobile",
                    {
                        "actions": [
                            {"action": "tap", "coordinate": [10, 20]},
                            {"action": "bogus_action"},
                            {"action": "tap", "coordinate": [30, 40]},
                        ]
                    },
                    call_id="call_batch",
                )
            ]
        )
    )

    # three slots, three frames; the middle one repeats its predecessor
    images = result.results[0].images
    assert len(images) == 3
    assert images[1] == images[0]
    assert "bogus_action" in (result.results[0].error or "")


def test_container_step_returns_one_frame_per_executed_action(monkeypatch):
    """The container-side half: the frame count follows the ACTION count.

    Also pins the other end of the contract: a batch that executed nothing
    still owes the model exactly one observation.
    """
    import base64

    from fastapi.testclient import TestClient

    server = _import_mobilegym_container_server(monkeypatch)
    frames = iter([b"mg-1", b"mg-2", b"mg-3", b"mg-4"])

    class _Env:
        async def step(self, action):
            pass

        async def get_observation(self):
            return SimpleNamespace(get_screenshot_bytes=lambda b=next(frames): b)

    iid = "iid-per-action-frames"
    inst = server._Instance(
        env=_Env(),
        slot={},
        task=SimpleNamespace(),
        task_id="clock.AddAlarm",
        eval_mode="text",
        max_steps=5,
        init_obs=SimpleNamespace(),
    )
    monkeypatch.setitem(server._instances, iid, inst)

    client = TestClient(server.app)
    r = client.post(
        "/step",
        json={
            "instance_id": iid,
            "actions": [
                {"name": "tap", "call_id": "c1", "arguments": {"coordinate": [100, 100]}},
                {"name": "tap", "call_id": "c2", "arguments": {"coordinate": [200, 200]}},
                {"name": "tap", "call_id": "c3", "arguments": {"coordinate": [300, 300]}},
            ],
        },
    )
    assert r.status_code == 200, r.text
    resp = r.json()
    assert [base64.b64decode(f) for f in resp["screenshots_b64"]] == [
        b"mg-1",
        b"mg-2",
        b"mg-3",
    ]

    r = client.post("/step", json={"instance_id": iid, "actions": []})
    assert r.status_code == 200, r.text
    resp = r.json()
    assert [base64.b64decode(f) for f in resp["screenshots_b64"]] == [b"mg-4"]


# ---------------------------------------------------------------------------
# reward_shaping (env_kwarg) — MobileGym's Success Rate vs the RL-only shaped
# variant. These pin the properties that make the shaped branch safe to train
# on; without them a regression here is silent (it only moves a number).
# ---------------------------------------------------------------------------


def _judge(*, success: bool, clean: bool, progress: float, judge_error: str | None = None):
    return SimpleNamespace(
        success=success, clean=clean, progress=progress, judge_error=judge_error,
    )


async def _score(monkeypatch, *, judge, terminated: bool, reward_shaping: bool) -> float:
    """Drive server._evaluate with a stubbed judge and no browser."""
    server = _import_mobilegym_container_server(monkeypatch)

    inst = server._Instance(
        env=SimpleNamespace(
            get_state=_async_return({}),
            get_route=_async_return({}),
            get_observation=_async_return(
                SimpleNamespace(get_screenshot_bytes=lambda: b"png")
            ),
        ),
        slot={},
        task=SimpleNamespace(apps=None, evaluate=lambda _inp: judge, answer_fields=None),
        task_id="clock.AddAlarm",
        eval_mode="text",
        max_steps=5,
        init_obs=SimpleNamespace(),
        reward_shaping=reward_shaping,
    )
    return await server._evaluate(inst, terminated)


def _async_return(value):
    async def _f(*_a, **_kw):
        return value
    return _f


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("success", "clean", "terminated", "expected"),
    [
        (True, True, True, 1.0),    # all three hold -> the SR numerator
        (True, True, False, 0.0),   # ran out of steps: upstream scores 0
        (True, False, True, 0.0),   # goal met but state dirtied
        (False, True, True, 0.0),   # goal not met
    ],
)
async def test_default_reward_is_the_benchmark_success_rate(
    monkeypatch, success, clean, terminated, expected
):
    """Default (reward_shaping=False) reproduces upstream's SR definition:
    COMPLETE and judge.success and judge.clean. progress is high on purpose --
    it must not leak into the score."""
    got = await _score(
        monkeypatch,
        judge=_judge(success=success, clean=clean, progress=0.99),
        terminated=terminated,
        reward_shaping=False,
    )
    assert got == expected


@pytest.mark.asyncio
async def test_shaped_reward_never_reaches_a_real_success(monkeypatch):
    """The argmax invariant: every shaped (SR=0) outcome must rank strictly below
    a genuine success, or RL learns to farm progress by skipping terminate or
    dirtying state -- neither of which `progress` penalizes."""
    server = _import_mobilegym_container_server(monkeypatch)

    success_score = await _score(
        monkeypatch, judge=_judge(success=True, clean=True, progress=1.0),
        terminated=True, reward_shaping=True,
    )
    assert success_score == 1.0
    # The two ways to have progress==1.0 while the SR is 0.
    for terminated, clean in ((False, True), (True, False)):
        shaped = await _score(
            monkeypatch, judge=_judge(success=True, clean=clean, progress=1.0),
            terminated=terminated, reward_shaping=True,
        )
        assert 0.0 < shaped < success_score
        assert shaped == pytest.approx(server._SHAPED_REWARD_CEILING)


@pytest.mark.asyncio
async def test_shaped_reward_is_monotone_in_progress(monkeypatch):
    prev = -1.0
    for progress in (0.0, 0.25, 0.5, 0.75, 1.0):
        got = await _score(
            monkeypatch, judge=_judge(success=False, clean=True, progress=progress),
            terminated=True, reward_shaping=True,
        )
        assert got > prev
        prev = got


@pytest.mark.asyncio
async def test_judge_error_scores_zero_even_when_shaping(monkeypatch):
    """A broken judge must not be paid partial credit: JudgeResult.error()
    returns progress=0.0, and shaping must not invent a score around it."""
    got = await _score(
        monkeypatch, judge=_judge(success=False, clean=True, progress=0.0, judge_error="boom"),
        terminated=True, reward_shaping=True,
    )
    assert got == 0.0


def _instance_for_step(server, monkeypatch, *, iid, judge, reward_shaping=False):
    """A live _Instance wired to a fake device so /step can be driven end to end."""
    class _Env:
        def __init__(self):
            self.executed = []

        async def step(self, action):
            self.executed.append(action)

        async def get_observation(self):
            return SimpleNamespace(get_screenshot_bytes=lambda: b"mobile-png")

        async def get_state(self, required_apps=None):
            return {}

        async def get_route(self):
            return {}

    env = _Env()
    inst = server._Instance(
        env=env,
        slot={},
        task=SimpleNamespace(apps=None, evaluate=lambda _inp: judge, answer_fields=None),
        task_id="clock.AddAlarm",
        eval_mode="text",
        max_steps=30,
        init_obs=SimpleNamespace(),
        reward_shaping=reward_shaping,
    )
    monkeypatch.setitem(server._instances, iid, inst)
    return env, inst


@pytest.mark.parametrize(
    ("status", "expected"),
    [("success", 1.0), ("failure", 0.0)],
)
def test_terminate_status_gates_the_success_rate(monkeypatch, status, expected):
    """Driven through /step so the COMPLETE-vs-ABORT branch actually runs.

    `terminate(status="failure")` steps ABORT, and upstream's SR requires
    stop_reason == COMPLETE (bench_env/runner/base.py::EpisodeResult.success), so a
    self-declared failure scores 0 even with every goal met. `terminated` alone
    cannot gate it -- it is True on both branches, which is exactly the bug this
    pins. Asserting on _evaluate's boolean argument instead would not catch it.
    """
    from fastapi.testclient import TestClient

    server = _import_mobilegym_container_server(monkeypatch)
    iid = f"iid-terminate-{status}"
    _instance_for_step(
        server, monkeypatch, iid=iid,
        judge=_judge(success=True, clean=True, progress=1.0),
    )
    r = TestClient(server.app).post("/step", json={
        "instance_id": iid,
        "actions": [{"name": "terminate", "arguments": {"status": status}}],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["terminated"] is True
    assert body["reward"] == expected


def test_reward_shaping_defaults_to_off_everywhere(monkeypatch):
    """These three defaults must agree on False, or eval silently scores shaped
    runs: the env yaml, the host ctor, and the container's _Instance field. The
    fourth (the /reset body fallback) is pinned by
    test_container_reset_reads_reward_shaping_off_the_body."""
    from lite.gym.envs.mobilegym import main as mg

    assert mg._REWARD_SHAPING is False, "configs/default.yaml"
    server = _import_mobilegym_container_server(monkeypatch)
    inst = server._Instance(
        env=SimpleNamespace(), slot={}, task=SimpleNamespace(), task_id="t",
        eval_mode="text", max_steps=5, init_obs=SimpleNamespace(),
    )
    assert inst.reward_shaping is False, "_Instance field default"
    import inspect
    sig = inspect.signature(mg.RemoteMobileGymEnv.__init__)
    assert sig.parameters["reward_shaping"].default is False, "host ctor default"


@pytest.mark.parametrize("shaping", [False, True])
def test_reset_body_forwards_reward_shaping(shaping):
    """The env_kwarg has to reach the container, where the judge runs.

    Dropping it from the body is silent in the worst way: the training yaml says
    `reward_shaping: true`, the container falls back to its `False` default, and RL
    trains on the sparse binary reward with nothing logged. Asserting the host-side
    attribute would not catch that -- only the wire body does.
    """
    import asyncio

    env = _remote(
        "clock.AddAlarm", answer_fields=False, base_max_steps=15,
        reward_shaping=shaping,
    )
    captured = {}

    def _fake_post(path, body):
        captured["body"] = body
        return {
            "instance_id": "iid", "screenshot_b64": "", "instruction": "",
            "max_steps": body["max_steps"],
        }

    env._post = _fake_post
    asyncio.run(env.reset())
    assert captured["body"]["reward_shaping"] is shaping


@pytest.mark.parametrize("shaping", [False, True])
def test_container_reset_reads_reward_shaping_off_the_body(monkeypatch, shaping):
    """The other half of the wire: /reset must put the body flag on the _Instance.

    The host can send `reward_shaping: true` correctly and the container still
    drop it on the floor, in which case every episode is scored with the sparse
    Success Rate while the training yaml says otherwise -- silently, since the
    default is a legal value. Only reading back the stored instance catches that.
    """
    from fastapi.testclient import TestClient

    server = _import_mobilegym_container_server(monkeypatch)

    class _Task:
        description = "task"
        answer_fields = None

        def __init__(self, _seed=None):
            pass

        async def setup(self, _env):
            return SimpleNamespace(get_screenshot_bytes=lambda: b"mobile-png")

    class _Env:
        def __init__(self, **_kwargs):
            pass

        async def start(self):
            pass

    async def _acquire():
        return {"browser": object(), "n_ctx": 0}

    monkeypatch.setattr(server, "TaskRegistry",
                        lambda: SimpleNamespace(get_by_id=lambda _tid: _Task))
    monkeypatch.setattr(server, "MobileGymEnv", _Env)
    monkeypatch.setattr(server, "_acquire_browser", _acquire)

    r = TestClient(server.app).post("/reset", json={
        "task_id": "clock.AddAlarm", "max_steps": 15, "reward_shaping": shaping,
    })
    assert r.status_code == 200, r.text
    inst = server._instances[r.json()["instance_id"]]
    assert inst.reward_shaping is shaping

    # An older host that predates the flag sends no key at all; the fallback must
    # be the eval-safe value, or an unflagged run would score shaped by accident.
    r = TestClient(server.app).post("/reset", json={
        "task_id": "clock.AddAlarm", "max_steps": 15,
    })
    assert r.status_code == 200, r.text
    assert server._instances[r.json()["instance_id"]].reward_shaping is False


@pytest.mark.parametrize(
    ("judge_kwargs", "expected"),
    [
        ({"success": True, "clean": True}, 1.0),
        ({"success": True, "clean": False}, 0.0),
        ({"success": False, "clean": True}, 0.0),
    ],
)
def test_text_mode_response_is_scored_as_a_completed_episode(
    monkeypatch, judge_kwargs, expected
):
    """`response` ending a text-mode episode is the DEFAULT eval scoring path.

    It is also this env's one deliberate deviation from upstream, which returns
    done=False from AnswerHandler and lets its own runner end the episode
    (bench_env/env/mobile_gym.py) -- there is no such outer runner here, so the
    COMPLETE signal has to land in /step. Flipping that flag scores every Q&A
    eval task 0 with nothing in the logs, so pin the reward, not just `terminated`.
    """
    from fastapi.testclient import TestClient

    server = _import_mobilegym_container_server(monkeypatch)
    iid = f"iid-response-{judge_kwargs['success']}-{judge_kwargs['clean']}"
    _instance_for_step(
        server, monkeypatch, iid=iid,
        judge=_judge(progress=1.0, **judge_kwargs),
    )
    r = TestClient(server.app).post("/step", json={
        "instance_id": iid,
        "actions": [{"name": "response", "arguments": {"text": "42"}}],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["terminated"] is True
    assert body["reward"] == expected
