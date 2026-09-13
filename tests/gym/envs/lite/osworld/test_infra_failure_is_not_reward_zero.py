"""OSWorld evaluator infrastructure failures must not enter reward as 0.0."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from lite.gym.errors import EnvBlocked


class _Response:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size=-1):
        payload, self._payload = self._payload, b""
        return payload


@pytest.mark.asyncio
async def test_container_missing_file_returns_none(tmp_path):
    from lite.gym.envs.lite.osworld.src.eval.runner import _download_from_container

    class _Iface:
        async def read_bytes(self, path):
            raise RuntimeError("read_bytes failed")

        async def run_command(self, cmd):
            return SimpleNamespace(stdout="", stderr="missing", returncode=1)

    class _Comp:
        interface = _Iface()

    assert await _download_from_container(_Comp(), "/tmp/missing.txt", str(tmp_path)) is None


@pytest.mark.asyncio
async def test_container_transport_failure_raises_env_blocked(tmp_path):
    from lite.gym.envs.lite.osworld.src.eval.runner import _download_from_container

    class _Iface:
        async def read_bytes(self, path):
            raise RuntimeError("read_bytes failed")

        async def run_command(self, cmd):
            raise RuntimeError("session closed")

    class _Comp:
        interface = _Iface()

    with pytest.raises(EnvBlocked, match="evaluator file could not be read"):
        await _download_from_container(_Comp(), "/tmp/result.txt", str(tmp_path))


@pytest.mark.asyncio
async def test_container_base64_decode_failure_raises_env_blocked(tmp_path):
    from lite.gym.envs.lite.osworld.src.eval.runner import _download_from_container

    class _Iface:
        async def read_bytes(self, path):
            raise RuntimeError("read_bytes failed")

        async def run_command(self, cmd):
            return SimpleNamespace(stdout="not-base64!!!", stderr="", returncode=0)

    class _Comp:
        interface = _Iface()

    with pytest.raises(EnvBlocked, match="evaluator file could not be decoded"):
        await _download_from_container(_Comp(), "/tmp/result.txt", str(tmp_path))


def test_download_url_uses_timeout_and_writes_nonempty_payload(monkeypatch, tmp_path):
    from lite.gym.envs.lite.osworld.src.eval import runner

    timeouts = []

    def fake_urlopen(_url, *, timeout):
        timeouts.append(timeout)
        return _Response(b"payload")

    monkeypatch.setattr(runner.urllib.request, "urlopen", fake_urlopen)

    result = runner._download_url("https://example.test/file.txt", str(tmp_path))

    assert result is not None
    assert Path(result).read_bytes() == b"payload"
    assert timeouts == [runner._DOWNLOAD_TIMEOUT_S]


def test_download_url_retry_exhaustion_raises_env_blocked(monkeypatch, tmp_path):
    from lite.gym.envs.lite.osworld.src.eval import runner

    times = iter([0.0, 0.0, 1.0])

    def fake_urlopen(_url, *, timeout):
        raise TimeoutError("socket timeout")

    monkeypatch.setattr(runner.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(runner, "_DOWNLOAD_TOTAL_BUDGET_S", 0.1)

    with pytest.raises(EnvBlocked, match="evaluator file"):
        runner._download_url("https://example.test/missing.txt", str(tmp_path))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config", "expected_name"),
    [
        (
            {"type": "cloud_file", "path": "https://example.test/result.txt", "dest": "result.txt"},
            "result.txt",
        ),
        (
            {
                "type": "cloud_file",
                "path": ["https://example.test/result.txt"],
                "dest": ["result.txt"],
                "gives": [0],
            },
            "result.txt",
        ),
    ],
)
async def test_result_cloud_file_download_runs_off_event_loop(
    monkeypatch, tmp_path, config, expected_name
):
    from lite.gym.envs.lite.osworld.src.eval import runner

    main_thread = threading.get_ident()
    download_threads = []

    def fake_download(_url, cache_dir, dest=""):
        download_threads.append(threading.get_ident())
        local = Path(cache_dir) / (dest or "result.txt")
        local.write_text("payload", encoding="utf-8")
        return str(local)

    monkeypatch.setattr(runner, "_download_url", fake_download)

    local = await runner._get_result(None, config, str(tmp_path))

    assert Path(local).name == expected_name
    assert download_threads
    assert all(thread_id != main_thread for thread_id in download_threads)


@pytest.mark.asyncio
async def test_result_file_not_found_scores_zero(monkeypatch):
    from lite.gym.envs.lite.osworld.src.eval import runner

    async def missing_result(_computer, _config, _cache_dir):
        raise FileNotFoundError("missing result")

    async def unused_expected(_computer, _config, _cache_dir):
        raise AssertionError("expected getter should not run")

    monkeypatch.setattr(runner, "_get_result", missing_result)
    monkeypatch.setattr(runner, "_get_expected", unused_expected)

    score, detail = await runner.evaluate_osworld_task(
        None,
        {"_postconfig_done": True, "func": "missing_metric", "result": {}, "expected": {}},
        debug=True,
    )

    assert score == 0.0
    assert detail["details"] == [
        {"func": "missing_metric", "error": "missing result", "score": 0.0}
    ]


@pytest.mark.asyncio
async def test_getter_infrastructure_failure_raises_env_blocked(monkeypatch):
    from lite.gym.envs.lite.osworld.src.eval import runner

    async def fake_result(_computer, _config, _cache_dir):
        return "result"

    async def broken_expected(_computer, _config, _cache_dir):
        raise RuntimeError("gold could not be read")

    monkeypatch.setattr(runner, "_get_result", fake_result)
    monkeypatch.setattr(runner, "_get_expected", broken_expected)

    with pytest.raises(EnvBlocked, match="expected getter failed"):
        await runner.evaluate_osworld_task(
            None,
            {"_postconfig_done": True, "func": "any_metric", "result": {}, "expected": {}},
        )


@pytest.mark.asyncio
async def test_result_getter_infrastructure_failure_raises_env_blocked(monkeypatch):
    from lite.gym.envs.lite.osworld.src.eval import runner

    async def broken_result(_computer, _config, _cache_dir):
        raise RuntimeError("container stopped answering")

    monkeypatch.setattr(runner, "_get_result", broken_result)

    with pytest.raises(EnvBlocked, match="result getter failed"):
        await runner.evaluate_osworld_task(
            None,
            {"_postconfig_done": True, "func": "any_metric", "result": {}, "expected": {}},
        )


@pytest.mark.asyncio
async def test_env_blocked_from_result_getter_bubbles(monkeypatch):
    from lite.gym.envs.lite.osworld.src.eval import runner

    blocked = EnvBlocked(what="download transport failed")

    async def blocked_result(_computer, _config, _cache_dir):
        raise blocked

    monkeypatch.setattr(runner, "_get_result", blocked_result)

    with pytest.raises(EnvBlocked) as exc_info:
        await runner.evaluate_osworld_task(
            None,
            {"_postconfig_done": True, "func": "any_metric", "result": {}, "expected": {}},
        )
    assert exc_info.value is blocked


@pytest.mark.asyncio
async def test_metric_exception_scores_zero_but_env_blocked_bubbles(monkeypatch):
    from lite.gym.envs.lite.osworld.src.eval import metrics as custom_metrics
    from lite.gym.envs.lite.osworld.src.eval import runner

    async def fake_result(_computer, _config, _cache_dir):
        return "result"

    async def fake_expected(_computer, _config, _cache_dir):
        return "expected"

    def broken_metric(_result, _expected):
        raise ValueError("metric crashed")

    blocked = EnvBlocked(what="metric transport failed")

    def blocked_metric(_result, _expected):
        raise blocked

    monkeypatch.setattr(runner, "_get_result", fake_result)
    monkeypatch.setattr(runner, "_get_expected", fake_expected)
    monkeypatch.setattr(custom_metrics, "_broken_metric", broken_metric, raising=False)
    monkeypatch.setattr(custom_metrics, "_blocked_metric", blocked_metric, raising=False)

    score, detail = await runner.evaluate_osworld_task(
        None,
        {
            "_postconfig_done": True,
            "func": "_broken_metric",
            "result": {},
            "expected": {},
        },
        debug=True,
    )

    assert score == 0.0
    assert detail["details"] == [
        {"func": "_broken_metric", "error": "metric crashed", "score": 0.0}
    ]
    with pytest.raises(EnvBlocked) as exc_info:
        await runner.evaluate_osworld_task(
            None,
            {
                "_postconfig_done": True,
                "func": "_blocked_metric",
                "result": {},
                "expected": {},
            },
        )
    assert exc_info.value is blocked
