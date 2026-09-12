"""An unscoreable episode must never be spelled the same way as a reward-0 one.

``0.0`` is a training signal: it says the agent tried and failed. The osworld evaluator used
to reach it from four other places — a metric past its deadline, a dead metric child, a gold
file that never downloaded, a container that stopped answering. None of those is anything the
policy did, and none of them was recorded anywhere in production (the ``{"error": ...}`` detail is
built only under ``debug=True``, which rollout never sets).

These pin the split. The agent's failures still score 0.0; everything else raises a typed
:class:`~lite.gym.errors.LiteGymError`, which rollout writes to the sample's ``error.txt``
and drops from the denominator (lite/infer/rollout.py:1562-1578).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lite.gym.envs.lite.osworld.src.eval import runner
from lite.gym.errors import EnvBlocked, is_retryable

_FIXTURES = "tests.gym.envs.lite.osworld.metric_fixtures"


def _command_result(stdout="", stderr="", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def _computer(run_command):
    return SimpleNamespace(interface=SimpleNamespace(run_command=run_command))


# ---------------------------------------------------------------------------
# The metric side
# ---------------------------------------------------------------------------

@pytest.fixture
def metrics_from_fixtures(monkeypatch):
    monkeypatch.setattr(runner, "_METRIC_MODULES", (_FIXTURES,))
    monkeypatch.setattr(runner, "_metric_ctx_cached", None)


@pytest.fixture
def stub_getters(monkeypatch):
    """Take the getters out of the picture so a test drives the metric alone."""

    def _use(result):
        async def _result(_computer, _config, _cache_dir):
            return result

        async def _expected(_computer, _config, _cache_dir, _result=None):
            return None

        monkeypatch.setattr(runner, "_get_result", _result)
        monkeypatch.setattr(runner, "_get_expected", _expected)

    return _use


@pytest.mark.asyncio
async def test_a_metric_timeout_does_not_become_reward_zero(
    monkeypatch, metrics_from_fixtures, stub_getters
):
    """The headline case: a metric killed at its deadline told us nothing about the agent."""
    monkeypatch.setattr(runner, "_METRIC_TIMEOUT_S", 2.0)
    stub_getters(600.0)  # seconds for the ``spin`` fixture — far past the deadline

    with pytest.raises(EnvBlocked) as excinfo:
        await runner.evaluate_osworld_task(None, {"_postconfig_done": True, "func": "spin"})

    assert "spin" in str(excinfo.value)
    assert "MetricTimeout" in str(excinfo.value)


@pytest.mark.asyncio
async def test_a_metric_that_raises_is_still_the_agents_zero(
    monkeypatch, metrics_from_fixtures, stub_getters
):
    """The deliberate other half: the metric RAN and rejected the artifacts.

    That is a verdict on what the episode produced, so it keeps scoring 0.0 — the episode
    stays in the denominator. Without this the fix above would void every corrupt-artifact
    task instead of failing it.
    """
    monkeypatch.setattr(runner, "_METRIC_TIMEOUT_S", 30.0)
    stub_getters(None)

    score = await runner.evaluate_osworld_task(
        None, {"_postconfig_done": True, "func": "explode"}
    )

    assert score == 0.0


@pytest.mark.asyncio
async def test_a_missing_result_artifact_is_still_the_agents_zero(
    monkeypatch, metrics_from_fixtures
):
    """Upstream parity: ``DesktopEnv.evaluate`` catches exactly ``FileNotFoundError``."""

    async def _result(_computer, _config, _cache_dir):
        raise FileNotFoundError("/home/user/Desktop/never-saved.xlsx")

    monkeypatch.setattr(runner, "_get_result", _result)

    score = await runner.evaluate_osworld_task(
        None, {"_postconfig_done": True, "func": "unit_score"}
    )

    assert score == 0.0


@pytest.mark.asyncio
async def test_a_void_episode_is_not_retried_and_carries_its_reason(
    monkeypatch, metrics_from_fixtures, stub_getters
):
    """The contract rollout reads: non-retryable (drop the sample, do not re-run the task),
    and the reason travels in the message, not in a debug-only dict."""
    monkeypatch.setattr(runner, "_METRIC_TIMEOUT_S", 2.0)
    stub_getters(600.0)

    with pytest.raises(EnvBlocked) as excinfo:
        await runner.evaluate_osworld_task(None, {"_postconfig_done": True, "func": "spin"})

    assert is_retryable(excinfo.value) is False
    assert excinfo.value.to_payload()["error_type"] == "EnvBlocked"
    assert "could not score" in excinfo.value.to_payload()["what"]


# ---------------------------------------------------------------------------
# The getter side
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_gold_that_will_not_download_voids_the_episode(monkeypatch, tmp_path):
    """It used to return ``None``: the metric then scored a gold nobody ever fetched."""
    monkeypatch.setattr(runner, "_DOWNLOAD_TOTAL_BUDGET_S", 0.5)
    monkeypatch.setattr(runner, "_DOWNLOAD_TIMEOUT_S", 0.5)

    with pytest.raises(EnvBlocked, match="gold file"):
        await runner._get_expected(
            None,
            {"type": "cloud_file", "path": "http://127.0.0.1:1/gold.xlsx", "dest": "gold.xlsx"},
            str(tmp_path),
        )


@pytest.mark.asyncio
async def test_a_container_that_stops_answering_is_not_a_missing_file(tmp_path):
    """``None`` from ``_download_from_container`` means "the container answered and had no
    bytes" — the agent's artifact is absent. A dead session must not borrow that spelling."""

    class _Iface:
        async def read_bytes(self, path):
            raise RuntimeError("exec-stdio session is gone")

        async def run_command(self, cmd):
            raise RuntimeError("exec-stdio session is gone")

    with pytest.raises(RuntimeError, match="session is gone"):
        await runner._download_from_container(
            SimpleNamespace(interface=_Iface()), "/home/user/out.xlsx", str(tmp_path)
        )


@pytest.mark.asyncio
async def test_a_command_that_ran_and_complained_is_still_scored(tmp_path):
    """The other half: a program that ran and failed IS evidence, and the rules judge it."""
    computer = _computer(
        lambda cmd: _fake_async(
            _command_result(stderr="ModuleNotFoundError: No module named 'x'", returncode=1)
        )
    )

    stderr = await runner._get_result(
        computer,
        {"type": "vm_command_error", "command": ["python", "-c", "import x"]},
        str(tmp_path),
    )

    assert "ModuleNotFoundError" in stderr


async def _fake_async(value):
    return value


@pytest.mark.asyncio
async def test_scoring_a_task_twice_does_not_grow_its_getter_lists(
    monkeypatch, metrics_from_fixtures, stub_getters
):
    """``options_list += ...`` extended the task's OWN evaluator dict in place."""
    stub_getters(None)
    evaluator = {
        "_postconfig_done": True,
        "func": ["unit_score", "unit_score"],
        "result": [{}, {}],
        "expected": [{}, {}],
        "options": [{"value": 1.0}],
    }

    await runner.evaluate_osworld_task(None, evaluator)
    await runner.evaluate_osworld_task(None, evaluator)

    assert evaluator["options"] == [{"value": 1.0}]


