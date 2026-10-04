"""browsergym.timewarp: registration, metadata, UI-version config, the scoped
TimeWarp container lifecycle (docker mocked), and a live oracle check.

Run:
  uv run pytest tests/gym/envs/browsergym/test_browsergym_timewarp.py
  uv run pytest -m live tests/gym/envs/browsergym/test_browsergym_timewarp.py  # needs the image
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("browsergym.core", reason="browsergym not installed")

from lite.gym.envs.browsergym import main as bg
from lite.gym.errors import CapacityExhausted, EnvDepsMissingError

try:
    _CONFIGS = bg._timewarp_task_configs()
except FileNotFoundError:
    _CONFIGS = {}

needs_timewarp = pytest.mark.skipif(
    not _CONFIGS,
    reason="browsergym-timewarp not installed (install.sh timewarp)",
)

_SUBSETS = ("coord", "chat", "infeas", "nav", "tab")


# ---------------------------------------------------------------------------
# Registration + static task facts
# ---------------------------------------------------------------------------


@needs_timewarp
class TestRegistration:

    def test_split_matches_upstream_browsergym_metadata(self):
        """Upstream ``timewarp.csv``: tasks 1-103 are test (registered "eval"),
        104-231 — the ones carrying a human-refined plan — are train."""
        import lite.gym as gym

        splits = gym.registry.task_ids("browsergym.timewarp")
        assert set(splits) == {"eval", "train"}
        assert splits["eval"] == [str(i) for i in range(1, 104)]
        assert splits["train"] == [str(i) for i in range(104, 232)]

    def test_registry_probe_does_not_import_the_package(self):
        """``browsergym.timewarp.__init__`` downloads nltk data when missing, so
        reading the catalog must not import it (fresh interpreter: other tests
        in this process may already have imported it)."""
        code = (
            "import sys\n"
            "import lite.gym.envs.browsergym.main as m\n"
            "assert len(m._timewarp_task_configs()) == 231\n"
            "assert 'browsergym.timewarp' not in sys.modules\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True, timeout=120)

    def test_viewport_matches_upstream_task(self):
        assert bg._BENCHMARK_VIEWPORTS["timewarp"] == (1280, 720)

    def test_metadata_carries_sites_judge_and_version(self, monkeypatch):
        monkeypatch.delenv("TIMEWARP_VERSION", raising=False)
        md = bg.BrowserGymEnv._task_metadata(
            "timewarp", "timewarp.84", action_subsets=_SUBSETS, viewport=(1280, 720),
        )
        assert md.dims == ("browser", "use")
        assert md.others["sites"] == ["news", "webshop", "wiki"]
        assert md.others["timewarp_version"] == bg._TIMEWARP_VERSION_DEFAULT
        assert "llm_as_a_judge" not in md.others
        # Read-only Q&A: no shared-backend isolation facts.
        assert "mutating" not in md.others

        judged = bg.BrowserGymEnv._task_metadata(
            "timewarp", "timewarp.32", action_subsets=_SUBSETS, viewport=(1280, 720),
        )
        assert judged.others["llm_as_a_judge"] is True

    def test_finish_and_nav_tools_are_advertised(self):
        md = bg.BrowserGymEnv._task_metadata(
            "timewarp", "timewarp.1", action_subsets=_SUBSETS, viewport=(1280, 720),
        )
        names = {schema["function"]["name"] for schema in md.extra_tool_schemas}
        assert {"response", "terminate", "goto", "back", "new_tab", "switch_tab"} <= names


class TestVersion:

    def test_env_var_overrides_yaml_default(self, monkeypatch):
        monkeypatch.setenv("TIMEWARP_VERSION", "4")
        assert bg._timewarp_version() == 4

    def test_yaml_default(self, monkeypatch):
        monkeypatch.delenv("TIMEWARP_VERSION", raising=False)
        assert bg._timewarp_version() == bg._TIMEWARP_VERSION_DEFAULT == 1

    @pytest.mark.parametrize("bad", ["0", "7", "v3", ""])
    def test_rejects_unknown_version(self, monkeypatch, bad):
        monkeypatch.setenv("TIMEWARP_VERSION", bad)
        with pytest.raises(ValueError, match="1-6"):
            bg._timewarp_version()


# ---------------------------------------------------------------------------
# Container lifecycle (docker mocked)
# ---------------------------------------------------------------------------


def _docker_port_result(returncode: int, stdout: str):
    return SimpleNamespace(returncode=returncode, stdout=stdout)


class TestPublishedPorts:

    def test_reads_each_site_port(self, monkeypatch):
        monkeypatch.setattr(bg, "_resolve_env_server_port", lambda: "30100")
        calls = []

        def fake_run(argv, **_kw):
            calls.append(argv)
            container_port = argv[-1].split("/")[0]
            host = {"5000": 6203, "5001": 6204, "5002": 6205}[container_port]
            return _docker_port_result(0, f"127.0.0.1:{host}\n")

        monkeypatch.setattr(bg.subprocess, "run", fake_run)
        assert bg._timewarp_published_ports() == {
            "TIMEWARP_WIKI_PORT": 6203,
            "TIMEWARP_NEWS_PORT": 6204,
            "TIMEWARP_WEBSHOP_PORT": 6205,
        }
        assert all(argv[:3] == ["docker", "port", "timewarp-30100"] for argv in calls)

    def test_none_when_container_absent(self, monkeypatch):
        monkeypatch.setattr(bg.subprocess, "run", lambda *a, **k: _docker_port_result(1, ""))
        assert bg._timewarp_published_ports() is None

    def test_none_when_docker_unavailable(self, monkeypatch):
        def boom(*_a, **_k):
            raise OSError("docker: not found")

        monkeypatch.setattr(bg.subprocess, "run", boom)
        assert bg._timewarp_published_ports() is None


class TestEnsureTimewarp:
    """``_ensure_timewarp``: start.sh gets the version + host ports, and TW_*
    always follow what the running container publishes."""

    PUBLISHED = {"TIMEWARP_WIKI_PORT": 6203, "TIMEWARP_NEWS_PORT": 6204,
                 "TIMEWARP_WEBSHOP_PORT": 6205}

    @pytest.fixture
    def harness(self, monkeypatch):
        for var in ("TIMEWARP_VERSION", "TIMEWARP_WIKI_PORT", "TIMEWARP_NEWS_PORT",
                    "TIMEWARP_WEBSHOP_PORT", "TW_WIKI", "TW_NEWS", "TW_WEBSHOP"):
            monkeypatch.delenv(var, raising=False)
        import lite.gym.utils.backend.docker as docker_mod
        import lite.gym.utils.backend.freshness as freshness_mod
        import lite.gym.utils.backend.ports as ports_mod

        monkeypatch.setattr(freshness_mod, "image_for", lambda env_id: env_id)
        monkeypatch.setattr(docker_mod, "require_image_present", lambda image: None)
        state = SimpleNamespace(
            published=[dict(self.PUBLISHED)],  # successive _timewarp_published_ports results
            allocated=[],
            start_env=None,
            start_error=None,
            up=True,
        )

        def published():
            return state.published.pop(0) if len(state.published) > 1 else state.published[0]

        def allocate(n, range_start, range_end):
            state.allocated.append((n, range_start, range_end))
            return list(range(range_start, range_start + n))

        def run(argv, **kw):
            assert argv[-1] == "timewarp" and argv[1].endswith("scripts/start.sh")
            state.start_env = kw["env"]
            if state.start_error is not None:
                raise state.start_error
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(bg, "_timewarp_published_ports", published)
        monkeypatch.setattr(ports_mod, "allocate_ports", allocate)
        monkeypatch.setattr(bg.subprocess, "run", run)
        monkeypatch.setattr(bg, "_timewarp_up", lambda: state.up)
        return state

    def test_adopts_running_container_ports(self, harness):
        bg._ensure_timewarp()
        assert harness.allocated == []
        assert harness.start_env["TIMEWARP_VERSION"] == "1"
        assert harness.start_env["TIMEWARP_WIKI_PORT"] == "6203"
        assert harness.start_env["TIMEWARP_WEBSHOP_PORT"] == "6205"
        assert os.environ["TW_WIKI"] == "http://localhost:6203"
        assert os.environ["TW_NEWS"] == "http://localhost:6204"
        assert os.environ["TW_WEBSHOP"] == "http://localhost:6205/abc"

    def test_allocates_from_band_when_no_container(self, harness, monkeypatch):
        monkeypatch.setenv("TIMEWARP_VERSION", "5")
        harness.published = [None, dict(self.PUBLISHED)]
        bg._ensure_timewarp()
        lo, hi = bg._TIMEWARP_PORT_RANGE
        assert harness.allocated == [(3, lo, hi + 1)]
        assert harness.start_env["TIMEWARP_VERSION"] == "5"
        assert [harness.start_env[v] for v, *_ in bg._TIMEWARP_SITES] == [
            str(lo), str(lo + 1), str(lo + 2)
        ]
        # TW_* come from the container afterwards, not from the request.
        assert os.environ["TW_WIKI"] == "http://localhost:6203"

    def test_operator_port_pin_wins(self, harness, monkeypatch):
        monkeypatch.setenv("TIMEWARP_NEWS_PORT", "6250")
        harness.published = [None, dict(self.PUBLISHED)]
        bg._ensure_timewarp()
        assert harness.start_env["TIMEWARP_NEWS_PORT"] == "6250"
        assert harness.allocated[0][0] == 2

    def test_start_failure_is_deps_missing(self, harness):
        harness.start_error = subprocess.CalledProcessError(
            1, ["bash"], stderr="error: cua-lite/browsergym.timewarp:latest not built",
        )
        with pytest.raises(EnvDepsMissingError, match="not built"):
            bg._ensure_timewarp()

    def test_still_loading_is_retriable(self, harness):
        harness.up = False
        with pytest.raises(CapacityExhausted, match="warming"):
            bg._ensure_timewarp()

    def test_ensure_services_dispatches_and_caches(self, monkeypatch):
        calls = []
        monkeypatch.setattr(bg, "_ensure_timewarp", lambda: calls.append(1))
        monkeypatch.setattr(bg, "_services_started", set())
        bg._ensure_services("browsergym.timewarp")
        bg._ensure_services("browsergym.timewarp")
        assert calls == [1]


class TestServices:

    def test_service_up_probes_the_sites(self, monkeypatch):
        monkeypatch.setattr(bg, "_timewarp_up", lambda: True)
        assert bg._service_up("timewarp") is True
        monkeypatch.setattr(bg, "_timewarp_up", lambda: False)
        assert bg._service_up("timewarp") is False

    def test_sites_down_without_urls(self, monkeypatch):
        for var in ("TW_WIKI", "TW_NEWS", "TW_WEBSHOP"):
            monkeypatch.delenv(var, raising=False)
        assert bg._timewarp_up() is False

    def test_health(self, monkeypatch):
        svc = bg.BrowserGymServices()
        monkeypatch.setattr(bg, "_timewarp_up", lambda: True)
        svc.health("browsergym.timewarp")
        monkeypatch.setattr(bg, "_timewarp_up", lambda: False)
        with pytest.raises(EnvDepsMissingError, match="timewarp"):
            svc.health("browsergym.timewarp")

    @pytest.mark.parametrize("hook", ["shutdown", "reap"])
    def test_lifecycle_removes_only_this_scopes_container(self, monkeypatch, hook):
        import lite.gym.utils.backend.docker as docker_mod

        removed: list[str] = []
        monkeypatch.setattr(
            docker_mod, "docker_rm_f",
            lambda name, *, timeout, label="docker": removed.append(name) or 0,
        )
        monkeypatch.setattr(bg, "_classifieds_compose_down_scoped", lambda sp: False)
        monkeypatch.setattr(bg, "_wa_stop_homepage_scoped", lambda hp: False)
        scope = SimpleNamespace(server_port=30111)
        svc = bg.BrowserGymServices()
        if hook == "shutdown":
            svc.shutdown("browsergym", scope)
        else:
            svc.reap("browsergym", scope, set(), boot=True)
        assert "timewarp-30111" in removed
        assert not any(n.startswith("timewarp-") and n != "timewarp-30111" for n in removed)


# ---------------------------------------------------------------------------
# Reset-time judge check
# ---------------------------------------------------------------------------


def _has_punkt() -> bool:
    try:
        import nltk

        nltk.data.find("tokenizers/punkt_tab")
        return True
    except (ImportError, LookupError):
        return False


@needs_timewarp
@pytest.mark.skipif(not _has_punkt(), reason="nltk punkt_tab missing (install.sh timewarp)")
class TestJudgeCheck:

    async def test_llm_judged_task_fails_at_reset_without_a_judge(self, monkeypatch):
        for var in ("OPENAI_API_KEY", "TW_JUDGE", "TW_JUDGE_MODEL", "TW_JUDGE_BASE_URL"):
            monkeypatch.delenv(var, raising=False)
        for var in ("TW_WIKI", "TW_NEWS", "TW_WEBSHOP"):
            monkeypatch.setenv(var, "http://localhost:1")
        env = bg.BrowserGymEnv(
            config=bg.BrowserGymConfig(
                bgym_task_id="timewarp.32", benchmark="timewarp",
                viewport_width=1280, viewport_height=720,
            ),
            max_steps=3,
        )
        try:
            with pytest.raises(EnvDepsMissingError, match="judge"):
                await env.reset()
            assert env._env is None, "must fail before any browser is launched"
        finally:
            await env.close()


# ---------------------------------------------------------------------------
# Live: real container + real browser, scripted (oracle) answers
# ---------------------------------------------------------------------------


@needs_timewarp
@pytest.mark.live
class TestTimeWarpLive:
    """Needs the ``cua-lite/browsergym.timewarp`` image and Playwright Chromium.
    Answers with each task's own gold (``fuzzy_match``) and a wrong answer, so the
    deterministic verifiers are exercised end-to-end through the wrapper."""

    async def _episode(self, task: str, answer: str):
        import lite.gym as gym
        from lite.core.tools import make_tool_call

        env = gym.make(f"browsergym.timewarp@{task}", max_steps=5)
        try:
            obs = await env.reset()
            assert obs.image[:4] == b"\x89PNG"
            assert os.environ["TW_WIKI"] in obs.text and os.environ["TW_WEBSHOP"] in obs.text
            result = await env.step([
                make_tool_call("response", {"text": answer}, call_id="call_0000"),
            ])
            assert result.terminated
            return result.reward
        finally:
            await env.close()

    # One task per site plus a three-site one; all reject a non-answer.
    @pytest.mark.parametrize("task", ["1", "33", "54", "84"])
    async def test_gold_answer_scores_and_wrong_answer_does_not(self, task):
        gold = _CONFIGS[f"timewarp.{task}"]["eval"]["reference_answers"]["fuzzy_match"]
        assert await self._episode(task, gold) == 1.0
        assert await self._episode(task, "I could not find it.") == 0.0
