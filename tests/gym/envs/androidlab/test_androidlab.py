"""Tests for the androidlab CUA-Lite gym environment.

Unit tests mock ``env._controller`` (the host-side ``_RemoteController``
proxy that talks to the in-container ``_CmdController``) via a
MagicMock — no Docker required. Lifecycle tests require the
``cua-lite/androidlab:latest`` image + Docker:
  - uv run --no-sync bash lite/gym/envs/androidlab/scripts/install.sh
  - set ANDROID_LAB_DOCKER=1

Run:
    uv run pytest tests/gym/envs/androidlab/test_androidlab.py -v
"""

from __future__ import annotations

import os
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import lite.gym as gym
import lite.gym.envs.androidlab.main as M
from lite.core.messages.final import make_no_tool_call_final_actions
from lite.core.tools import make_tool_call
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.schemas import tool_schema_name, tool_schema_parameters
from lite.gym.envs.androidlab.main import (
    _BOUNDS_RE,
    AndroidLabEnv,
    AndroidLabTaskConfig,
    _downscale_png,
    _make_env,
    _rescale_tree_bounds,
)


def _tc(name: str, arguments: dict | None = None) -> dict:
    return make_tool_call(name, arguments or {})


def _tool_schema(env: AndroidLabEnv, name: str) -> dict:
    return next(s for s in env.metadata.extra_tool_schemas if tool_schema_name(s) == name)


def _mobile_action_schema_enum(metadata) -> list[str]:
    from lite.agents.core.action_space.base import LiteMobileActionSpace

    schemas = LiteMobileActionSpace.get_tool_schemas()
    if metadata.valid_actions is not None:
        schemas = LiteMobileActionSpace.filter_child_action_enum(
            schemas, metadata.valid_actions,
        )
    mobile = next(s for s in schemas if tool_schema_name(s) == "mobile")
    action_schema = tool_schema_parameters(mobile)["properties"]["actions"]["items"]
    return action_schema["properties"]["action"]["enum"]


# ---------------------------------------------------------------------------
# Registry tests (sync, no env creation needed)
# ---------------------------------------------------------------------------

def test_task_registration():
    """Walking the vendored YAMLs should register all 138 tasks."""
    import lite.gym.envs.androidlab.main  # noqa: F401
    from lite.gym.registry import _splits

    eval_tasks = _splits.get("androidlab", {}).get("eval", [])
    # Reference ships 138 tasks across 9 apps; allow mild fluctuation across
    # upstream revisions but assert the order of magnitude.
    assert len(eval_tasks) >= 130, f"expected ~138 tasks, got {len(eval_tasks)}"

    # Every registered task_id must carry app + judge metadata (needed for
    # reward computation); missing either silently produces reward=0. Read the
    # registered config table directly — gym.make() would trigger ensure_services
    # (a container image-freshness check) that a pure metadata test shouldn't need.
    from lite.gym.envs.androidlab.main import _TASK_CONFIGS
    for task_id in eval_tasks[:10]:
        cfg: AndroidLabTaskConfig = _TASK_CONFIGS[task_id]
        assert cfg.task_id == task_id
        assert cfg.app, f"task {task_id} missing app name"
        # Production tasks identify the judge by name (resolved inside
        # the container's /task/load). Host-side `judge_class` is a
        # test-only backdoor and is None for real registrations.
        assert cfg.judge_class_name, f"task {task_id} missing judge_class_name"
        assert cfg.app_module_name, f"task {task_id} missing app_module_name"


def test_services_health_uses_cached_dependency_preflight(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(M, "_HEALTH_CHECK", lambda env_id: calls.append(env_id))

    svc = M.AndroidLabServices()
    svc.health("androidlab")

    assert calls == ["androidlab"]


def test_registry_covers_all_apps():
    """All 9 apps (calendar, clock, ...) should have at least one task."""
    import lite.gym.envs.androidlab.main  # noqa: F401
    from lite.gym.registry import _splits

    eval_tasks = _splits.get("androidlab", {}).get("eval", [])
    apps = {tid.rsplit("_", 1)[0] for tid in eval_tasks}
    expected = {"bluecoins", "calendar", "cantook", "clock", "contacts",
                "map", "pimusic", "setting", "zoom"}
    assert expected.issubset(apps), f"missing apps: {expected - apps}"


def test_make_env_factory():
    """_make_env should build an AndroidLabEnv with the right metadata."""
    cfg = AndroidLabTaskConfig(task_id="test_1", app="TestApp", instruction="do X")
    env = _make_env(config=cfg, max_steps=7)
    assert isinstance(env, AndroidLabEnv)
    assert env.metadata.platform == "mobile"
    assert env.metadata.task_type == "use"
    assert env._max_steps == 7


def test_open_app_catalog_metadata_matches_schema_enum():
    env = _make_env(
        config=AndroidLabTaskConfig(task_id="test_1"),
        max_steps=7,
        extra_tools=["open_app"],
    )

    assert env.metadata.others["apps"] == M._ANDROID_LAB_APPS
    assert (
        tool_schema_parameters(_tool_schema(env, "open_app"))["properties"]["app_name"]["enum"]
        == env.metadata.others["apps"]
    )


def test_default_mobile_schema_does_not_advertise_pinch():
    env = _make_env(config=AndroidLabTaskConfig(task_id="test_1"), max_steps=7)
    enum = _mobile_action_schema_enum(env.metadata)
    assert "pinch" not in enum
    assert "system_button" in enum


def test_make_env_overrides_dont_mutate_config():
    """Overrides passed via **kwargs must not mutate the shared base config."""
    cfg = AndroidLabTaskConfig(task_id="t1", metadata={"source": "yaml"})
    _ = _make_env(config=cfg, adb_query="adb shell getprop")
    # Shared config's metadata must stay untouched — otherwise concurrent
    # resets would see each other's overrides.
    assert cfg.metadata == {"source": "yaml"}


def test_bind_accepts_host_side_soft_kwargs():
    """Bind compatibility: bind() must accept the host-side knobs
    (post_action_delay / observation_text / screenshot_max_dim) as soft
    kwargs and write any non-None value onto self._config.

    Previously these lived on the config but not in bind()'s signature, so
    explicit bind calls with any of them raised TypeError.
    """
    cfg = AndroidLabTaskConfig(task_id="t1")
    env = _make_env(config=cfg)
    baseline = (
        env._config.post_action_delay,
        env._config.observation_text,
        env._config.screenshot_max_dim,
    )

    # Must NOT raise TypeError (the regression).
    env.bind(
        post_action_delay=0.0,
        observation_text="a11y_list:norm",
        screenshot_max_dim=512,
    )
    assert env._config.post_action_delay == 0.0
    assert env._config.observation_text == "a11y_list:norm"
    assert env._config.screenshot_max_dim == 512

    # None (the default, == absent on the cold path) keeps the baked value:
    # byte-equal cold/warm guarantee.
    env2 = _make_env(config=AndroidLabTaskConfig(task_id="t2"))
    env2.bind()
    assert (
        env2._config.post_action_delay,
        env2._config.observation_text,
        env2._config.screenshot_max_dim,
    ) == baseline


def test_host_side_kwargs_are_bind_owned():
    """The three host-side knobs are bind-owned soft state.

    Construction accepts them explicitly only to forward into bind(); the image
    shape remains controlled by config/image.
    """
    import inspect
    ctor_params = set(inspect.signature(AndroidLabEnv.__init__).parameters)
    bind_params = set(inspect.signature(AndroidLabEnv.bind).parameters)
    for knob in ("post_action_delay", "observation_text", "screenshot_max_dim"):
        assert knob in ctor_params
        assert knob in bind_params


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------

class TestBoundsRegex:
    """_BOUNDS_RE parses UIAutomator's [x1,y1][x2,y2] format."""

    def test_basic_match(self):
        m = _BOUNDS_RE.match("[10,20][30,40]")
        assert m is not None
        assert tuple(map(int, m.groups())) == (10, 20, 30, 40)

    def test_large_values(self):
        m = _BOUNDS_RE.match("[1440,3120][0,0]")
        assert m is not None
        assert tuple(map(int, m.groups())) == (1440, 3120, 0, 0)

    def test_no_match(self):
        assert _BOUNDS_RE.match("not bounds") is None
        assert _BOUNDS_RE.match("[10, 20][30, 40]") is None  # spaces not tolerated


class TestRescaleTreeBounds:
    """_rescale_tree_bounds walks a compressed JSON tree and maps every bounds
    corner through the ONE a11y point conversion it is handed (``_a11y_point``),
    so the tree view and the flat list cannot drift into different spaces."""

    @staticmethod
    def _scale(sx: float, sy: float):
        return lambda x, y: (int(round(x * sx)), int(round(y * sy)))

    def test_rescales_single_node(self):
        tree = {"root": {"bounds": "[100,200][300,400]", "children": {}}}
        out = _rescale_tree_bounds(tree, self._scale(0.5, 0.25))
        assert out["root"]["bounds"] == "[50,50][150,100]"

    def test_nested_tree(self):
        tree = {
            "a": {
                "bounds": "[0,0][1000,1000]",
                "kids": {
                    "b": {"bounds": "[100,200][300,400]"},
                },
            },
        }
        out = _rescale_tree_bounds(tree, self._scale(0.1, 0.1))
        assert out["a"]["bounds"] == "[0,0][100,100]"
        assert out["a"]["kids"]["b"]["bounds"] == "[10,20][30,40]"

    def test_rounds_not_truncates(self):
        """The conversion rounds (``pixel_to_norm``), matching _to_pixels, so
        there is no off-by-one at the far edge under :norm."""
        # [1,1][3,3] × 0.5 → round(0.5/0.5/1.5/1.5); truncation would give
        # [0,0][1,1], but Python's banker's round gives [0,0][2,2].
        tree = {"x": {"bounds": "[1,1][3,3]"}}
        out = _rescale_tree_bounds(tree, self._scale(0.5, 0.5))
        assert out["x"]["bounds"] == "[0,0][2,2]"

    def test_preserves_non_bounds_keys(self):
        tree = {"root": {"bounds": "[0,0][10,10]", "text": "Hello"}}
        out = _rescale_tree_bounds(tree, self._scale(2.0, 2.0))
        assert out["root"]["text"] == "Hello"
        assert out["root"]["bounds"] == "[0,0][20,20]"

    def test_returns_copy_not_mutation(self):
        """Judge needs the native-pixel tree intact; rescale must not mutate."""
        tree = {"root": {"bounds": "[0,0][100,100]"}}
        _ = _rescale_tree_bounds(tree, self._scale(0.5, 0.5))
        assert tree["root"]["bounds"] == "[0,0][100,100]"

    def test_list_nodes(self):
        tree = {"root": [{"bounds": "[0,0][10,10]"}, {"bounds": "[20,20][30,30]"}]}
        out = _rescale_tree_bounds(tree, self._scale(2.0, 2.0))
        assert out["root"][0]["bounds"] == "[0,0][20,20]"
        assert out["root"][1]["bounds"] == "[40,40][60,60]"

    def test_passthrough_non_containers(self):
        convert = self._scale(0.5, 0.5)
        assert _rescale_tree_bounds("literal", convert) == "literal"
        assert _rescale_tree_bounds(42, convert) == 42
        assert _rescale_tree_bounds(None, convert) is None

    def test_norm_tree_bounds_invert_to_pixels(self):
        """The tree goes through the same conversion as the flat list: a corner
        emitted under :norm maps back onto the native pixel it came from."""
        env = _make_unit_env()
        env._screen_w, env._screen_h = 1440, 3120
        tree = {"root": {"bounds": "[0,0][1440,3120]", "mid": {"bounds": "[720,1560][720,1560]"}}}
        out = _rescale_tree_bounds(
            tree, lambda x, y: env._a11y_point(x, y, "norm")
        )
        assert out["root"]["bounds"] == "[0,0][1000,1000]"
        assert out["root"]["mid"]["bounds"] == "[500,500][500,500]"
        assert env._to_pixels([500, 500]) == (720, 1560)


class TestDownscalePng:
    """_downscale_png caps the longer side while preserving the PNG header."""

    @staticmethod
    def _make_png(w: int, h: int) -> bytes:
        from PIL import Image
        buf = BytesIO()
        Image.new("RGB", (w, h), (128, 128, 128)).save(buf, format="PNG")
        return buf.getvalue()

    def test_downscales_large(self):
        from PIL import Image
        big = self._make_png(2000, 1000)
        raw = _downscale_png(big, max_dim=500)
        assert raw[:4] == b"\x89PNG"
        img = Image.open(BytesIO(raw))
        assert max(img.size) <= 500

    def test_passthrough_small(self):
        """Images under max_dim should be encoded without re-sampling (content preserved)."""
        from PIL import Image
        small = self._make_png(100, 50)
        raw = _downscale_png(small, max_dim=500)
        img = Image.open(BytesIO(raw))
        assert img.size == (100, 50)

    def test_invalid_bytes_fallback(self):
        """On any failure, fall back to the raw bytes — observation must not error."""
        out = _downscale_png(b"not-a-png", max_dim=500)
        assert out == b"not-a-png"


# ---------------------------------------------------------------------------
# Env method tests — minimal instance, no controller needed
# ---------------------------------------------------------------------------

def _make_unit_env(native=(1440, 3120), **cfg_kwargs) -> AndroidLabEnv:
    """Build an env with a minimal config, no container or controller bound.
    ``native`` sets ``_screen_w/h`` (device-native size, normally from ``wm size``
    at reset). There is no display_resolution — coords derive from native."""
    extra_tools = cfg_kwargs.pop("extra_tools", None)
    cfg = AndroidLabTaskConfig(
        task_id=cfg_kwargs.pop("task_id", "unit_test_1"),
        app=cfg_kwargs.pop("app", "UnitApp"),
        instruction=cfg_kwargs.pop("instruction", "do unit test"),
        **cfg_kwargs,
    )
    env_kwargs = {"max_steps": 10}
    if extra_tools is not None:
        env_kwargs["extra_tools"] = extra_tools
    env = AndroidLabEnv(config=cfg, **env_kwargs)
    env._screen_w, env._screen_h = native
    return env


class TestCoordinateScaling:
    def test_to_pixels_midpoint(self):
        env = _make_unit_env()
        assert env._to_pixels([500, 500]) == (720, 1560)

    def test_to_pixels_origin(self):
        env = _make_unit_env()
        assert env._to_pixels([0, 0]) == (0, 0)

    def test_to_pixels_max(self):
        env = _make_unit_env()
        assert env._to_pixels([1000, 1000]) == (1440, 3120)

    def test_to_pixels_none_raises(self):
        env = _make_unit_env()
        with pytest.raises(ValueError, match="malformed normalized coordinate: None"):
            env._to_pixels(None)

    def test_a11y_point_pixel_mode_is_the_space_to_pixels_outputs(self):
        """pixel mode is IDENTITY: a11y bboxes are already native pixels, the
        surface ``_to_pixels`` projects ONTO — no display_resolution scaling."""
        env = _make_unit_env()
        env._screen_w, env._screen_h = 1440, 3120
        assert env._a11y_point(720, 1560, "pixel") == (720, 1560)

    def test_a11y_point_norm_mode_inverts_to_pixels(self):
        """norm mode is the exact inverse of the conversion the ACTION side
        applies, so a centre the model echoes back lands on the element. Emitting
        native pixels here instead would send the click off-screen."""
        env = _make_unit_env()
        env._screen_w, env._screen_h = 1440, 3120
        centre = env._a11y_point(720, 1560, "norm")
        assert centre == (500, 500)
        assert env._to_pixels(list(centre)) == (720, 1560)


class TestPreferAcXml:
    """map.me and pimusic need the XMLParser a11y service, not uiautomator dump."""

    @pytest.mark.parametrize("task_id,expected", [
        ("map_1", True),
        ("map_15", True),
        ("pimusic_1", True),
        ("pimusic_12", True),
        ("calendar_1", False),
        ("clock_7", False),
        ("bluecoins_11", False),
        ("contacts_3", False),
    ])
    def test_prefer_ac_xml(self, task_id, expected):
        env = _make_unit_env(task_id=task_id)
        assert env._prefer_ac_xml() is expected


class TestComputeReward:
    def test_no_judge_hit_zero(self):
        env = _make_unit_env()
        assert env._compute_reward() == 0.0

    def test_judge_complete_true(self):
        env = _make_unit_env()
        env._best_judge = {"judge_page": True, "complete": True}
        assert env._compute_reward() == 1.0

    def test_judge_page_only_not_complete(self):
        """judge_page=True but complete=False → reward 0 (partial progress)."""
        env = _make_unit_env()
        env._best_judge = {"judge_page": True, "complete": False}
        assert env._compute_reward() == 0.0


# ---------------------------------------------------------------------------
# Judge integration tests — regression guards for 6 silent-zero bugs
# ---------------------------------------------------------------------------

class _FakeStatefulJudge:
    """Mimics reference's stateful judges (origin_bill, edit_started_correctly).

    Counts how often judge() is called and tracks whether it received a dict
    (required by find_subtrees_of_parents_with_key) — both invariants have
    been broken by bugs before.
    """
    instances_created = 0

    def __init__(self):
        type(self).instances_created += 1
        self.call_count = 0
        self.last_input_type: type | None = None
        self.flag = False  # stateful — flips true on first call with "hit"

    def judge(self, xml_compressed_tree, line):
        self.call_count += 1
        self.last_input_type = type(xml_compressed_tree)
        # Stateful: once flag is set, every subsequent call returns complete=True.
        if isinstance(xml_compressed_tree, dict) and xml_compressed_tree.get("hit"):
            self.flag = True
        return {"judge_page": True, "complete": self.flag}


def _reset_judge_counters():
    _FakeStatefulJudge.instances_created = 0


def test_judge_receives_dict_not_string():
    """Regression: _get_obs_text must json.loads before handoff (bug #1)."""
    _reset_judge_counters()
    env = _make_unit_env(judge_class=_FakeStatefulJudge)
    env._run_judge({"hit": False}, {"parsed_action": {}})
    assert isinstance(env._judge, _FakeStatefulJudge)
    assert env._judge.last_input_type is dict, (
        "judge() must be called with a dict; JSON strings cause "
        "find_subtrees_of_parents_with_key to silently return []"
    )


def test_judge_instance_reused_across_steps():
    """Regression: one judge per episode, NOT one per step (bug #2).

    ~16 reference judges carry state on self.<flag>; recreating per step
    resets the flag and the task never registers as complete.
    """
    _reset_judge_counters()
    env = _make_unit_env(judge_class=_FakeStatefulJudge)
    # 5 judge() calls — one judge instance, five calls.
    for _ in range(5):
        env._run_judge({"hit": False}, {"parsed_action": {}})
    assert _FakeStatefulJudge.instances_created == 1
    assert env._judge.call_count == 5


def test_judge_state_survives_across_steps():
    """Stateful flag set on step N must persist through step N+1..M."""
    _reset_judge_counters()
    env = _make_unit_env(judge_class=_FakeStatefulJudge)
    # Step 1: set the flag
    env._run_judge({"hit": True}, {"parsed_action": {}})
    # Step 2..4: no hit, but the flag should stay set → complete=True
    env._run_judge({"hit": False}, {"parsed_action": {}})
    env._run_judge({"hit": False}, {"parsed_action": {}})
    assert env._judge.flag is True
    assert env._compute_reward() == 1.0


def test_best_judge_only_caches_judge_page_true():
    """best_judge should not be updated when judge_page=False."""
    class _PageOnlyJudge:
        def __init__(self):
            self._calls = 0

        def judge(self, tree, line):
            self._calls += 1
            # Alternate: step 1 page=False, step 2 page=True+complete=True, step 3 page=False
            if self._calls == 1:
                return {"judge_page": False}
            if self._calls == 2:
                return {"judge_page": True, "complete": True}
            return {"judge_page": False}

    env = _make_unit_env(judge_class=_PageOnlyJudge)
    env._run_judge({"t": 1}, {})
    assert env._best_judge is None
    env._run_judge({"t": 2}, {})
    assert env._best_judge is not None
    assert env._best_judge["complete"] is True
    env._run_judge({"t": 3}, {})
    # Not overwritten by the later page=False step.
    assert env._best_judge["complete"] is True


def test_judge_errors_propagate():
    """Evaluator failures must reach the rollout error handler."""
    class _BustedJudge:
        def judge(self, tree, line):
            raise ValueError("upstream judge bug")

    env = _make_unit_env(judge_class=_BustedJudge)
    with pytest.raises(ValueError, match="upstream judge bug"):
        env._run_judge({"t": 1}, {})
    assert env._best_judge is None


def test_judge_skips_on_no_xml():
    """compressed_xml=None (dump failed) → silent skip, no exception."""
    _reset_judge_counters()
    env = _make_unit_env(judge_class=_FakeStatefulJudge)
    env._run_judge(None, {})
    assert env._judge.call_count == 0


def test_judge_notimplemented_propagates():
    """An abstract judge cannot provide a valid score."""
    class _AbstractJudge:
        def judge(self, tree, line):
            raise NotImplementedError

    env = _make_unit_env(judge_class=_AbstractJudge)
    with pytest.raises(NotImplementedError):
        env._run_judge({"t": 1}, {})
    assert env._best_judge is None


@pytest.fixture
def llm_judge_client(monkeypatch):
    # The pinned dependency lives in the AndroidLab image, not the host venv.
    pytest.importorskip("android_lab.evaluation.task")
    monkeypatch.setenv("AZURE_API_BASE", "https://judge.example.test")
    monkeypatch.setenv("AZURE_API_KEY", "test-key")
    client = MagicMock()
    monkeypatch.setattr("openai.AzureOpenAI", lambda **kwargs: client)
    return client.chat.completions.create


@pytest.mark.parametrize("text,finish_reason,expected", [
    ("[True]", "stop", True),
    ("[False]", "stop", False),
    ("", "stop", None),
    ("[True] [False]", "stop", None),
    ("[True]", "length", None),
])
def test_llm_judge_response_validation(llm_judge_client, text, finish_reason, expected):
    from android_lab.evaluation.task import SingleTask

    llm_judge_client.return_value = SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish_reason, message=SimpleNamespace(content=text),
    )])
    judge = SingleTask()
    if expected is None:
        with pytest.raises(ValueError):
            judge._llm_judge("question", "answer", "ground truth")
    else:
        assert judge._llm_judge("question", "answer", "ground truth") is expected
    assert llm_judge_client.call_args.kwargs["model"] == "gpt-4o-mini"
    assert llm_judge_client.call_args.kwargs["max_tokens"] == 16
    assert llm_judge_client.call_args.kwargs["temperature"] == 0.0


def test_llm_judge_requires_credentials(monkeypatch):
    task = pytest.importorskip("android_lab.evaluation.task")
    for name in ("AZURE_API_BASE", "AZURE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="no credentials"):
        task.SingleTask()._llm_judge("question", "answer", "ground truth")


@pytest.mark.parametrize("task_id", [14, 16, 17, 22])
def test_setting_judge_api_failure_propagates(llm_judge_client, task_id):
    import httpx
    import openai
    from android_lab.evaluation.tasks.setting import setting

    llm_judge_client.side_effect = openai.AuthenticationError(
        "judge authentication failed",
        response=httpx.Response(401, request=httpx.Request("POST", "https://judge.example.test")),
        body=None,
    )
    judge = getattr(setting, f"SingleTask_Setting_{task_id}")()
    judge.judge_page = lambda tree: True
    line = {
        "target": "question",
        "parsed_action": {"action": "finish", "kwargs": {"message": "answer"}},
        "command": {
            "adb shell 'getprop persist.sys.timezone'": "America/Los_Angeles",
            "adb shell getprop ro.build.version.release": "13",
            "adb shell settings get global airplane_mode_on": "1",
        },
    }
    with pytest.raises(openai.AuthenticationError, match="judge authentication failed"):
        judge.judge({}, line)


# ---------------------------------------------------------------------------
# Action dispatch tests — mock DockerExecAndroidController
# ---------------------------------------------------------------------------

def _env_with_mock_controller() -> tuple[AndroidLabEnv, MagicMock]:
    env = _make_unit_env()
    ctrl = MagicMock()
    env._controller = ctrl
    return env, ctrl


def _env_for_step(monkeypatch, extra_tools=None) -> tuple[AndroidLabEnv, list[dict]]:
    env = _make_unit_env(post_action_delay=0.0, extra_tools=extra_tools)
    env._step_count = 0
    env._max_steps = 10
    judge_lines: list[dict] = []

    monkeypatch.setattr(
        env,
        "_observe_via_rpc",
        lambda: {
            "screenshot": b"\x89PNG",
            "obs_text": "state",
            "compressed_xml": {"hit": True},
            "current_activity": "UnitActivity",
        },
    )

    def fake_run_judge(compressed_xml, line):
        judge_lines.append(line)
        env._best_judge = {"judge_page": True, "complete": True}

    monkeypatch.setattr(env, "_run_judge", fake_run_judge)
    return env, judge_lines


@pytest.mark.asyncio
async def test_judge_rpc_failure_records_error_for_resume(monkeypatch, tmp_path):
    import lite.agents.factory as agent_factory
    from lite.infer.rollout import run_rollout

    env, _ = _env_for_step(monkeypatch, extra_tools=["response"])
    monkeypatch.setattr(env, "_run_judge", AndroidLabEnv._run_judge.__get__(env))
    env._judge_loaded = True
    env._rpc = MagicMock()
    env._rpc.post.side_effect = RuntimeError("server /task/judge returned 500")

    async def sample(env, hooks):
        result = await env.step([_tc("response", {"text": "final answer"})])
        return SimpleNamespace(
            steps=[], episode_return=result.reward,
            terminated=result.terminated, truncated=result.truncated,
        )

    monkeypatch.setattr(env, "close", AsyncMock())
    monkeypatch.setattr(gym, "make", lambda *args, **kwargs: env)
    monkeypatch.setattr(
        agent_factory, "make", lambda *args, **kwargs: SimpleNamespace(sample=sample),
    )
    all_done, log_root = await run_rollout(
        model_id="claude-opus-5", model_path="claude-opus-5", env_id="androidlab",
        agent_kwargs={}, env_kwargs={}, seed=1, concurrency=1,
        log_root=tmp_path, task_id="setting_14", save_data=False,
    )
    sample_dir = log_root / "task" / "setting_14" / "sample_00"
    assert not all_done
    assert "/task/judge returned 500" in (sample_dir / "error.txt").read_text()
    assert not (sample_dir / "summary.json").exists()
    assert env._best_judge is None


@pytest.mark.asyncio
async def test_execution_failure_returns_error_with_current_obs(monkeypatch):
    env, _ = _env_for_step(monkeypatch)

    def fail_dispatch(_name, _args):
        raise RuntimeError("adb tap failed")

    monkeypatch.setattr(env, "_dispatch_action", fail_dispatch)

    result = await env.step([
        make_tool_call("tap", {"coordinate": [500, 500]}, call_id="call_tap")
    ])

    assert result.terminated is False
    assert result.results[0].tool_call_id == "call_tap"
    assert result.results[0].metadata == {"is_error": True}
    assert result.results[0].error == "tap failed: execution failed"
    assert result.results[0].images[-1] == b"\x89PNG"
    assert result.results[0].text == "state"


@pytest.mark.asyncio
async def test_malformed_tap_coordinate_returns_error_with_current_obs(monkeypatch):
    env, _ = _env_for_step(monkeypatch)
    env._controller = MagicMock()

    result = await env.step([make_tool_call("tap", {}, call_id="call_tap")])

    assert result.terminated is False
    assert result.results[0].tool_call_id == "call_tap"
    assert result.results[0].metadata == {"is_error": True}
    assert result.results[0].error == (
        "invalid arguments for tap: coordinate is required"
    )
    assert result.results[0].images[-1] == b"\x89PNG"
    assert result.results[0].text == "state"


@pytest.mark.asyncio
async def test_active_open_app_executes_when_enabled(monkeypatch):
    env, _judge_lines = _env_for_step(monkeypatch, extra_tools=["open_app"])
    ctrl = MagicMock()
    env._controller = ctrl

    result = await env.step([
        make_tool_call("open_app", {"app_name": "Settings"}, call_id="call_open")
    ])

    ctrl.launch_app.assert_called_once_with("Settings")
    assert result.info["executed_actions"] == [{
        "call": "open_app",
        "args": {"app_name": "Settings"},
    }]
    assert result.results[0].tool_call_id == "call_open"
    assert result.results[0].images[-1] == b"\x89PNG"
    assert result.results[0].text == "state"
    assert result.results[0].metadata != {"is_error": True}


@pytest.mark.asyncio
async def test_inactive_open_app_returns_error_only_feedback(monkeypatch):
    env, _judge_lines = _env_for_step(monkeypatch)

    result = await env.step([
        make_tool_call("open_app", {"app_name": "Settings"}, call_id="call_open")
    ])

    assert result.info["executed_actions"] == [{
        "call": "noop",
        "args": {"name": "open_app", "reason": "inactive extra tool"},
    }]
    assert result.results[0].tool_call_id == "call_open"
    assert result.results[0].metadata == {"is_error": True}
    assert result.results[0].error == "open_app is not available in this task."
    assert result.results[0].images == []
    assert result.results[0].text is None
    assert env._step_count == 1
    assert result.info["step"] == 1
    assert result.truncated is False


@pytest.mark.asyncio
async def test_invalid_open_app_is_rejected_before_dispatch(monkeypatch):
    env, _judge_lines = _env_for_step(monkeypatch, extra_tools=["open_app"])

    def fail_dispatch(_name, _args):
        raise AssertionError("invalid app should not dispatch")

    monkeypatch.setattr(env, "_dispatch_action", fail_dispatch)

    result = await env.step([
        make_tool_call("open_app", {"app_name": "Not A Real App"}, call_id="call_open")
    ])

    assert result.info["executed_actions"] == []
    assert result.results[0].tool_call_id == "call_open"
    assert result.results[0].metadata == {"is_error": True}
    assert result.results[0].error.startswith("invalid arguments for open_app: ")
    assert "Not A Real App" in result.results[0].error
    assert result.results[0].images[-1] == b"\x89PNG"
    assert env._step_count == 1
    assert result.info["step"] == 1
    assert result.truncated is False


@pytest.mark.asyncio
async def test_unknown_standalone_tool_returns_error_only(monkeypatch):
    env, judge_lines = _env_for_step(monkeypatch)
    env._max_steps = 1

    result = await env.step([make_tool_call("foo", {}, call_id="call_foo")])

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
    assert result.info["step"] == 1
    assert result.truncated is True
    assert len(judge_lines) == 1


@pytest.mark.asyncio
async def test_step_rejects_flat_lite_boundary_call(monkeypatch):
    env, _judge_lines = _env_for_step(monkeypatch)

    with pytest.raises(TypeError, match="env.step expects canonical Lite tool calls"):
        await env.step([
            {"name": "tap", "arguments": {"coordinate": [500, 500]}},
        ])


class TestDispatchAction:
    def test_tap(self):
        env, ctrl = _env_with_mock_controller()
        out = env._dispatch_action("tap", {"coordinate": [500, 500]})
        ctrl.tap.assert_called_once_with(720, 1560)
        assert out["call"] == "tap"
        assert out["args"] == {"x": 720, "y": 1560}

    def test_long_press(self):
        env, ctrl = _env_with_mock_controller()
        out = env._dispatch_action("long_press", {"coordinate": [250, 250], "duration": 2.0})
        ctrl.long_press.assert_called_once_with(360, 780, duration_ms=2000)
        assert out["args"]["duration_ms"] == 2000

    def test_swipe(self):
        env, ctrl = _env_with_mock_controller()
        env._dispatch_action("swipe", {
            "start_coordinate": [500, 700],
            "coordinate": [500, 300],
        })
        ctrl.swipe_precise.assert_called_once()
        (start, end), kwargs = ctrl.swipe_precise.call_args[0], ctrl.swipe_precise.call_args[1]
        assert start == (720, 2184)
        assert end == (720, 936)
        assert kwargs == {"duration_ms": 400}

    def test_drag_uses_precise_swipe(self):
        env, ctrl = _env_with_mock_controller()
        out = env._dispatch_action("drag", {
            "start_coordinate": [500, 700],
            "coordinate": [500, 300],
            "duration": 1.25,
        })
        ctrl.swipe_precise.assert_called_once()
        (start, end), kwargs = ctrl.swipe_precise.call_args[0], ctrl.swipe_precise.call_args[1]
        assert start == (720, 2184)
        assert end == (720, 936)
        assert kwargs == {"duration_ms": 1250}
        assert out["call"] == "drag"
        assert out["args"]["duration_ms"] == 1250

    def test_type_sends_text_then_enter(self):
        """Reference's TextOnlyExecutor submits after typing (search boxes)."""
        env, ctrl = _env_with_mock_controller()
        env._dispatch_action("type", {"text": "hello"})
        ctrl.text.assert_called_once_with("hello")
        ctrl.enter.assert_called_once_with()

    def test_system_button_home(self):
        env, ctrl = _env_with_mock_controller()
        env._dispatch_action("system_button", {"button": "Home"})
        ctrl.home.assert_called_once()

    def test_system_button_back(self):
        env, ctrl = _env_with_mock_controller()
        env._dispatch_action("system_button", {"button": "Back"})
        ctrl.back.assert_called_once()

    def test_system_button_recent(self):
        env, ctrl = _env_with_mock_controller()
        env._dispatch_action("system_button", {"button": "Recent"})
        ctrl.recent.assert_called_once()

    def test_system_button_unknown(self):
        env, ctrl = _env_with_mock_controller()
        with pytest.raises(ValueError, match="unknown button 'Bogus'"):
            env._dispatch_action("system_button", {"button": "Bogus"})
        ctrl.home.assert_not_called()
        ctrl.back.assert_not_called()

    def test_wait(self, monkeypatch):
        env, _ = _env_with_mock_controller()
        sleeps: list[float] = []
        monkeypatch.setattr(M.time, "sleep", lambda duration: sleeps.append(duration))

        out = env._dispatch_action("wait", {"duration": 0.25})

        assert out["call"] == "wait"
        assert out["args"]["duration_s"] == 0.25
        assert sleeps == [0.25]

    def test_screenshot_is_noop(self):
        """screenshot is captured after every step anyway; action itself does nothing."""
        env, ctrl = _env_with_mock_controller()
        out = env._dispatch_action("screenshot", {})
        assert out["call"] == "screenshot_noop"
        ctrl.assert_not_called()  # no adb traffic

    def test_unknown_action_noop(self):
        env, _ = _env_with_mock_controller()
        out = env._dispatch_action("klingon_greet", {})
        assert out["call"] == "noop"


@pytest.mark.asyncio
async def test_step_terminate_uses_canonical_call_and_finish_payload(monkeypatch):
    env, judge_lines = _env_for_step(monkeypatch, extra_tools=["terminate"])

    r = await env.step([_tc("terminate", {"status": "success", "reason": "done"})])

    assert r.terminated is True
    assert r.reward == 1.0
    assert r.info["executed_actions"] == [
        {"call": "terminate", "args": {"status": "success", "reason": "done"}}
    ]
    assert judge_lines[0]["parsed_action"] == {
        "operation": "finish",
        "action": "finish",
        "kwargs": {"message": "done"},
    }


@pytest.mark.asyncio
async def test_step_response_uses_canonical_call_and_answer_message(monkeypatch):
    env, judge_lines = _env_for_step(monkeypatch, extra_tools=["response"])

    r = await env.step([
        make_tool_call("response", {"text": "final answer"}, call_id="call_response")
    ])

    assert r.terminated is True
    assert r.reward == 1.0
    assert r.info["executed_actions"] == [{"call": "response", "args": {"text": "final answer"}}]
    # A terminal call gets NO tool result: it ended the episode, so there is no
    # next decision for an observation to inform, and
    # ``devs/migration/verify.py`` refuses a tool result for a terminal call.
    assert r.results == []
    assert judge_lines[0]["parsed_action"] == {
        "operation": "finish",
        "action": "finish",
        "kwargs": {"message": "final answer"},
    }


@pytest.mark.asyncio
async def test_content_only_final_text_uses_internal_response_no_tool_result(monkeypatch):
    env, judge_lines = _env_for_step(monkeypatch)

    actions = make_no_tool_call_final_actions("final answer")
    result = await env.step(actions)

    assert tool_call_name(actions[0]) == "response"
    assert tool_call_arguments(actions[0]) == {"text": "final answer"}
    assert result.terminated is True
    assert result.results == []
    assert result.reward == 1.0
    assert result.info["executed_actions"] == [
        {"call": "response", "args": {"text": "final answer"}}
    ]
    assert judge_lines[0]["parsed_action"] == {
        "operation": "finish",
        "action": "finish",
        "kwargs": {"message": "final answer"},
    }


@pytest.mark.asyncio
async def test_step_unsupported_action_returns_error_only_feedback(monkeypatch):
    env, _judge_lines = _env_for_step(monkeypatch)

    r = await env.step([
        make_tool_call(
            "pinch",
            {
                "coordinate": [500, 500],
                "direction": "in",
                "amount": 25,
            },
            call_id="call_pinch",
        )
    ])

    assert r.info["executed_actions"] == [{
        "call": "noop",
        "args": {"name": "pinch", "reason": "unsupported action: pinch"},
    }]
    assert r.results[0].tool_call_id == "call_pinch"
    assert r.results[0].metadata == {"is_error": True}
    assert r.results[0].error == "unsupported action: pinch"
    assert r.results[0].images == []
    assert r.results[0].text is None
    assert env._step_count == 1
    assert r.info["step"] == 1
    assert r.truncated is False


@pytest.mark.asyncio
async def test_step_unknown_system_button_returns_model_error_with_current_obs(monkeypatch):
    env, _judge_lines = _env_for_step(monkeypatch)
    env._controller = MagicMock()

    r = await env.step([
        make_tool_call("system_button", {"button": "Bogus"}, call_id="call_bogus")
    ])

    assert r.info["executed_actions"][0]["call"] == "noop"
    assert r.results[0].tool_call_id == "call_bogus"
    assert r.results[0].metadata == {"is_error": True}
    assert r.results[0].error.startswith("invalid arguments for system_button: ")
    assert "unknown button 'Bogus'" in r.results[0].error
    assert r.results[0].images[-1] == b"\x89PNG"
    assert env._step_count == 1
    assert r.info["step"] == 1
    assert r.truncated is False


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
    monkeypatch, name, args, expected,
):
    env, _judge_lines = _env_for_step(monkeypatch)
    env._controller = MagicMock()

    result = await env.step([make_tool_call(name, args, call_id=f"call_{name}")])

    assert result.info["executed_actions"][0]["call"] == "noop"
    assert result.results[0].tool_call_id == f"call_{name}"
    assert result.results[0].metadata == {"is_error": True}
    assert result.results[0].error == f"invalid arguments for {name}: {expected}"
    assert result.results[0].images[-1] == b"\x89PNG"
    env._controller.long_press.assert_not_called()
    env._controller.swipe_precise.assert_not_called()


# ---------------------------------------------------------------------------
# Lifecycle tests — require Docker, skipped by default
# ---------------------------------------------------------------------------

_run_docker = os.environ.get("ANDROID_LAB_DOCKER", "") == "1"


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _run_docker,
    reason="Set ANDROID_LAB_DOCKER=1 to run Docker lifecycle tests",
)
async def test_reset_returns_screenshot():
    """reset() should spawn a container, load snapshot, return a valid PNG."""
    from lite.gym.registry import _splits
    task_id = _splits["androidlab"]["eval"][0]
    env = gym.make(f"androidlab@{task_id}", max_steps=3)
    try:
        obs = await env.reset()
        assert obs.image
        raw = obs.image
        assert raw[:4] == b"\x89PNG"
    finally:
        await env.close()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _run_docker,
    reason="Set ANDROID_LAB_DOCKER=1 to run Docker lifecycle tests",
)
async def test_step_tap_returns_new_screenshot():
    from lite.gym.registry import _splits
    task_id = _splits["androidlab"]["eval"][0]
    env = gym.make(f"androidlab@{task_id}", max_steps=3)
    try:
        await env.reset()
        r = await env.step([
            _tc("tap", {"coordinate": [500, 500]}),
        ])
        assert r.results[0].images
        assert r.results[0].images[-1][:4] == b"\x89PNG"
        assert r.info["executed_actions"][0]["call"] == "tap"
    finally:
        await env.close()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _run_docker,
    reason="Set ANDROID_LAB_DOCKER=1 to run Docker lifecycle tests",
)
async def test_terminate_triggers_judge():
    from lite.gym.registry import _splits
    task_id = _splits["androidlab"]["eval"][0]
    env = gym.make(f"androidlab@{task_id}", max_steps=3)
    try:
        await env.reset()
        r = await env.step([
            _tc("terminate", {"status": "success"}),
        ])
        assert r.terminated is True
        assert isinstance(r.reward, float)
    finally:
        await env.close()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _run_docker,
    reason="Set ANDROID_LAB_DOCKER=1 to run Docker lifecycle tests",
)
async def test_truncation_returns_reward():
    from lite.gym.registry import _splits
    task_id = _splits["androidlab"]["eval"][0]
    env = gym.make(f"androidlab@{task_id}", max_steps=1)
    try:
        await env.reset()
        r = await env.step([
            _tc("tap", {"coordinate": [500, 500]}),
        ])
        assert r.truncated is True
        assert r.terminated is False
        assert isinstance(r.reward, float)
    finally:
        await env.close()


def test_display_resolution_rejected():
    """androidlab does NOT accept display_resolution: the AVD render size is
    emulator-fixed (not controllable); clicks map [0,1000] → native and a11y
    coords derive from native. Accepting it would silently mismatch the real
    render → misclicks, so construction must be a hard error (parity with
    browsergym).

    Enforced by explicit constructor/bind signatures: ``display_resolution`` is
    in neither signature, so Python parameter binding TypeErrors."""
    import inspect
    assert "display_resolution" not in inspect.signature(
        AndroidLabEnv.__init__).parameters
    assert "display_resolution" not in inspect.signature(
        AndroidLabEnv.bind).parameters
    with pytest.raises(TypeError, match="display_resolution"):
        AndroidLabEnv(
            config=AndroidLabTaskConfig(task_id="t1"),
            display_resolution=[1920, 1080],
        )


# ── one frame per executed action ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_action_batch_returns_one_distinct_frame_per_action(monkeypatch):
    """An N-action batch returns N frames, one per action, in action order.

    Frames must be REAL per-action captures: the stubs below hand back a
    different payload on every call, so emitting one cached frame N times
    (numerically N frames, zero new information) fails here. The final frame
    comes from the batched observation, which returns the frame and the XML in
    one RPC so the image and ``obs_text`` describe the same instant.
    """
    env, _ = _env_for_step(monkeypatch)
    env._controller = MagicMock()

    shots: list[int] = []

    def fake_screenshot():
        shots.append(len(shots) + 1)
        return f"frame-{len(shots)}".encode()

    monkeypatch.setattr(env, "_get_screenshot_png", fake_screenshot)

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
    assert images == [b"frame-1", b"frame-2", b"\x89PNG"], (
        "per-action captures in action order; the last frame is the batched "
        "observation's, which carries obs_text with it"
    )
    assert len(shots) == 2, "the trailing observation supplies the last frame"


@pytest.mark.asyncio
async def test_zero_executed_actions_still_returns_one_frame(monkeypatch):
    """Nothing ran, but the turn still owes the model a current observation."""
    env, _ = _env_for_step(monkeypatch)
    env._controller = MagicMock()
    monkeypatch.setattr(
        env, "_get_screenshot_png", lambda: pytest.fail("no action executed"),
    )

    result = await env.step([
        make_tool_call("mobile", {"actions": []}, call_id="call_empty"),
    ])

    assert result.results[0].images == [b"\x89PNG"]
