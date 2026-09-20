"""ScaleCUA evaluator shim and metric-call tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lite.gym.envs.lite.scalecua.src.osworld import judges
from lite.gym.envs.lite.scalecua.src.osworld import verify as scalecua_verify
from lite.gym.errors import EnvBlocked


class _FakeInterface:
    def __init__(
        self,
        stdout: str | list[str] = "stdout",
        files: dict[str, bytes] | None = None,
    ):
        self.commands: list[str] = []
        self.command_calls: list[dict[str, object]] = []
        self.hotkeys: list[tuple[str, ...]] = []
        self.typed_text: list[str] = []
        self.stdout = stdout
        self.files = files or {}

    async def read_bytes(self, path: str) -> bytes:
        if path in self.files:
            return self.files[path]
        return f"bytes:{path}".encode()

    async def screenshot(self) -> bytes:
        return b"png"

    async def get_screen_size(self):
        return {"width": 800, "height": 600}

    async def run_command(self, command: str, timeout=None):
        self.commands.append(command)
        self.command_calls.append({"command": command, "timeout": timeout})
        if isinstance(self.stdout, list):
            stdout = self.stdout.pop(0) if self.stdout else ""
        else:
            stdout = self.stdout
        return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    async def hotkey(self, *keys: str):
        self.hotkeys.append(tuple(keys))

    async def type_text(self, text: str):
        self.typed_text.append(text)


class _FakeComputer:
    def __init__(
        self,
        stdout: str = "stdout",
        files: dict[str, bytes] | None = None,
    ):
        self.interface = _FakeInterface(stdout=stdout, files=files)


@pytest.mark.asyncio
async def test_scalecua_overlay_getter_receives_desktop_env_adapter(tmp_path):
    env = judges.make_eval_env(_FakeComputer(), str(tmp_path))

    def getter(eval_env, config):
        result = eval_env.controller.run_bash_script("echo ok")
        return {
            "cache_dir": eval_env.cache_dir,
            "desktop": eval_env.controller.get_vm_desktop_path(),
            "file": eval_env.controller.get_file(config["path"]),
            "stdout": result["stdout"],
            "output": result["output"],
        }

    out = await judges.call_overlay_getter(
        getter,
        env,
        {"type": "custom", "path": "/home/user/Desktop/a.txt"},
        str(tmp_path),
    )
    assert out["cache_dir"] == str(tmp_path)
    assert out["desktop"] == "/home/user/Desktop"
    assert out["file"] == b"bytes:/home/user/Desktop/a.txt"
    assert out["stdout"] == "stdout"
    assert out["output"] == "stdout"


@pytest.mark.asyncio
async def test_scalecua_eval_env_shim_exposes_vm_platform_and_machine(tmp_path):
    env = judges.make_eval_env(_FakeComputer(), str(tmp_path))

    assert env.vm_platform == "Linux"
    assert env.vm_machine == "Linux"


@pytest.mark.asyncio
async def test_scalecua_controller_run_bash_script_accepts_argv_list(tmp_path):
    computer = _FakeComputer(stdout="0")
    env = judges.make_eval_env(computer, str(tmp_path))

    def getter(eval_env, config):
        return eval_env.controller.run_bash_script(
            ["pgrep", "-x", "soffice.bin"],
            timeout=5,
        )["output"]

    out = await judges.call_overlay_getter(
        getter,
        env,
        {"type": "custom"},
        str(tmp_path),
    )

    assert out == "0"
    assert computer.interface.commands == ["bash -lc 'pgrep -x soffice.bin'"]


@pytest.mark.asyncio
async def test_scalecua_volume_settings_getter_repairs_pgrep_x_false_negative(tmp_path):
    computer = _FakeComputer(stdout="")
    env = judges.make_eval_env(computer, str(tmp_path))

    def getter(eval_env, config):
        assert eval_env.cache_dir == str(tmp_path)
        assert config["type"] == "custom"
        return {"volume_output": "Volume: front-left: 65536 / 100%", "settings_open": False}

    getter.__name__ = "get_volume_settings_state__dummy"

    out = await judges.call_overlay_getter(
        getter,
        env,
        {"type": "custom"},
        str(tmp_path),
    )

    assert out["settings_open"] is True
    assert len(computer.interface.commands) == 1
    command = computer.interface.commands[0]
    assert "wmctrl -lx" in command
    assert "pgrep -f" in command
    assert "pgrep -x gnome-control-center" not in command


@pytest.mark.asyncio
async def test_scalecua_volume_settings_repair_is_narrow(tmp_path):
    computer = _FakeComputer(stdout="")
    env = judges.make_eval_env(computer, str(tmp_path))

    def getter(eval_env, config):
        return {"settings_open": False}

    getter.__name__ = "get_other_settings_state__dummy"

    out = await judges.call_overlay_getter(
        getter,
        env,
        {"type": "custom"},
        str(tmp_path),
    )

    assert out == {"settings_open": False}
    assert computer.interface.commands == []


@pytest.mark.asyncio
async def test_scalecua_volume_terminal_getter_uses_window_state_not_pgrep(tmp_path):
    computer = _FakeComputer(stdout="terminal-open\n")
    env = judges.make_eval_env(computer, str(tmp_path))

    def getter(eval_env, config):
        assert config["type"] == "custom"
        return {"volume_output": "Volume: front-left: 65536 / 100%", "terminal_open": False}

    getter.__name__ = "get_volume_terminal_state__dummy"

    out = await judges.call_overlay_getter(
        getter,
        env,
        {"type": "custom"},
        str(tmp_path),
    )

    assert out["terminal_open"] is True
    assert len(computer.interface.commands) == 1
    command = computer.interface.commands[0]
    assert "wmctrl -lx" in command
    assert "xdotool search --onlyvisible --class" in command
    assert "pgrep gnome-terminal" not in command


@pytest.mark.asyncio
async def test_scalecua_volume_terminal_getter_does_not_trust_returncode_only(tmp_path):
    computer = _FakeComputer(stdout="")
    env = judges.make_eval_env(computer, str(tmp_path))

    def getter(eval_env, config):
        return {"volume_output": "Volume: front-left: 65536 / 100%", "terminal_open": True}

    getter.__name__ = "get_volume_terminal_state__dummy"

    out = await judges.call_overlay_getter(
        getter,
        env,
        {"type": "custom"},
        str(tmp_path),
    )

    assert out["terminal_open"] is False
    assert len(computer.interface.commands) == 1


@pytest.mark.asyncio
async def test_scalecua_volume_terminal_repair_uses_result_type_when_name_is_plain(tmp_path):
    computer = _FakeComputer(stdout="")
    env = judges.make_eval_env(computer, str(tmp_path))

    def getter(eval_env, config):
        return {"volume_output": "Volume: front-left: 65536 / 100%", "terminal_open": True}

    getter.__name__ = "get_volume_terminal_state"

    out = await judges.call_overlay_getter(
        getter,
        env,
        {"type": "volume_terminal_state"},
        str(tmp_path),
    )

    assert out["terminal_open"] is False
    assert len(computer.interface.commands) == 1


@pytest.mark.asyncio
async def test_scalecua_volume_terminal_repair_is_narrow(tmp_path):
    computer = _FakeComputer(stdout="")
    env = judges.make_eval_env(computer, str(tmp_path))

    def getter(eval_env, config):
        return {"terminal_open": False}

    getter.__name__ = "get_other_terminal_state__dummy"

    out = await judges.call_overlay_getter(
        getter,
        env,
        {"type": "custom"},
        str(tmp_path),
    )

    assert out == {"terminal_open": False}
    assert computer.interface.commands == []


@pytest.mark.asyncio
async def test_scalecua_metric_injects_env_and_does_not_swallow_typeerror(tmp_path):
    env = judges.make_eval_env(_FakeComputer(), str(tmp_path))

    def metric(result, expected, env):
        assert env.cache_dir == str(tmp_path)
        return result == expected

    assert await judges.call_metric(metric, env, "ok", "ok", {}) is True

    def bad_getter(eval_env, config):
        raise TypeError("body failure")

    with pytest.raises(TypeError, match="body failure"):
        await judges.call_overlay_getter(bad_getter, env, {"type": "bad"}, str(tmp_path))


@pytest.mark.asyncio
async def test_scalecua_exact_match_metric_does_not_leak_injected_env(tmp_path):
    env = judges.make_eval_env(_FakeComputer(), str(tmp_path))
    metric = judges.resolve_metric("exact_match", "train")

    assert await judges.call_metric(metric, env, "true", {"expected": "true"}, {}) == 1.0


@pytest.mark.asyncio
async def test_scalecua_url_pattern_metric_does_not_leak_injected_env(tmp_path):
    env = judges.make_eval_env(_FakeComputer(), str(tmp_path))
    metric = judges.resolve_metric("is_expected_url_pattern_match", "train")

    assert (
        await judges.call_metric(
            metric,
            env,
            {"url": "https://mileageplustravel.united.com/explore/car-rentals"},
            {"expected": ["united.com/en/us/book/cars"]},
            {},
        )
        == 1.0
    )


@pytest.mark.asyncio
async def test_scalecua_execute_python_command_keeps_pyautogui_prefix(tmp_path):
    computer = _FakeComputer(stdout="")
    env = judges.make_eval_env(computer, str(tmp_path))

    def getter(eval_env, config):
        return eval_env.controller.execute_python_command("pyautogui.hotkey('ctrl', 's')")

    await judges.call_overlay_getter(getter, env, {"type": "custom"}, str(tmp_path))
    assert "pyautogui.FAILSAFE = False" in computer.interface.commands[-1]


@pytest.mark.asyncio
async def test_scalecua_evaluate_uses_official_score_aggregation(monkeypatch):
    async def fake_get_result(eval_env, config, cache_dir, runtime_split):
        return config["score"]

    async def fake_get_expected(eval_env, config, cache_dir, runtime_split):
        return "expected"

    monkeypatch.setattr(scalecua_verify, "_get_result", fake_get_result)
    monkeypatch.setattr(scalecua_verify, "_get_expected", fake_get_expected)
    monkeypatch.setattr(
        judges,
        "resolve_metric",
        lambda name, runtime_split: lambda result, expected=None: result,
    )

    single = await scalecua_verify.evaluate_scalecua_task(
        _FakeComputer(),
        {"func": "score_metric", "result": {"score": 0.25}, "expected": {}},
        runtime_split="train",
    )
    and_score = await scalecua_verify.evaluate_scalecua_task(
        _FakeComputer(),
        {
            "func": ["score_metric", "score_metric"],
            "result": [{"score": 1.0}, {"score": 0.5}],
            "expected": [{}, {}],
            "conj": "and",
        },
        runtime_split="train",
    )
    or_score = await scalecua_verify.evaluate_scalecua_task(
        _FakeComputer(),
        {
            "func": ["score_metric", "score_metric"],
            "result": [{"score": 0.25}, {"score": 0.5}],
            "expected": [{}, {}],
            "conj": "or",
        },
        runtime_split="train",
    )

    assert single == 0.25
    assert and_score == 0.75
    assert or_score == 0.5


@pytest.mark.asyncio
async def test_scalecua_evaluate_does_not_mutate_short_options_list(monkeypatch):
    async def fake_get_result(eval_env, config, cache_dir, runtime_split):
        return config["score"]

    async def fake_get_expected(eval_env, config, cache_dir, runtime_split):
        return "expected"

    monkeypatch.setattr(scalecua_verify, "_get_result", fake_get_result)
    monkeypatch.setattr(scalecua_verify, "_get_expected", fake_get_expected)
    monkeypatch.setattr(
        judges,
        "resolve_metric",
        lambda name, runtime_split: lambda result, expected=None: result,
    )

    evaluator = {
        "func": ["score_metric", "score_metric"],
        "result": [{"score": 1.0}, {"score": 0.5}],
        "expected": [{}, {}],
        "options": [{}],
        "conj": "and",
    }

    assert await scalecua_verify.evaluate_scalecua_task(
        _FakeComputer(),
        evaluator,
        runtime_split="train",
    ) == 0.75
    assert evaluator["options"] == [{}]


@pytest.mark.asyncio
@pytest.mark.parametrize("short_key", ["result", "expected"])
async def test_scalecua_short_result_or_expected_list_is_env_blocked(short_key):
    from lite.gym.errors import EnvBlocked

    evaluator = {
        "func": ["score_metric", "score_metric"],
        "result": [{}, {}],
        "expected": [{}, {}],
    }
    evaluator[short_key] = [{}]

    with pytest.raises(EnvBlocked, match=f"{short_key} list is shorter than func list"):
        await scalecua_verify.evaluate_scalecua_task(
            _FakeComputer(),
            evaluator,
            runtime_split="train",
        )


@pytest.mark.asyncio
async def test_scalecua_task_removes_owned_cache_dir(monkeypatch, tmp_path):
    owned = tmp_path / "owned"

    async def fake_evaluate(
        _computer,
        _evaluator,
        *,
        runtime_split,
        cache_dir,
        run_postconfig,
        pre_postconfig_state,
        reference_sources,
        debug,
    ):
        Path(cache_dir, "artifact.txt").write_text("payload", encoding="utf-8")
        return 0.5

    monkeypatch.setattr(scalecua_verify.tempfile, "mkdtemp", lambda prefix: str(owned))
    monkeypatch.setattr(scalecua_verify, "_evaluate_scalecua_task", fake_evaluate)

    assert (
        await scalecua_verify.evaluate_scalecua_task(
            _FakeComputer(),
            {"func": "score_metric"},
            runtime_split="train",
        )
        == 0.5
    )
    assert not owned.exists()


@pytest.mark.asyncio
async def test_scalecua_task_leaves_caller_cache_dir(monkeypatch, tmp_path):
    caller = tmp_path / "caller"
    caller.mkdir()

    async def fake_evaluate(
        _computer,
        _evaluator,
        *,
        runtime_split,
        cache_dir,
        run_postconfig,
        pre_postconfig_state,
        reference_sources,
        debug,
    ):
        Path(cache_dir, "artifact.txt").write_text("payload", encoding="utf-8")
        return 0.5

    monkeypatch.setattr(scalecua_verify, "_evaluate_scalecua_task", fake_evaluate)

    assert (
        await scalecua_verify.evaluate_scalecua_task(
            _FakeComputer(),
            {"func": "score_metric"},
            runtime_split="train",
            cache_dir=str(caller),
        )
        == 0.5
    )
    assert (caller / "artifact.txt").read_text(encoding="utf-8") == "payload"


@pytest.mark.asyncio
async def test_scalecua_task_keeps_owned_cache_dir_in_debug(monkeypatch, tmp_path):
    owned = tmp_path / "debug-owned"

    async def fake_evaluate(
        _computer,
        _evaluator,
        *,
        runtime_split,
        cache_dir,
        run_postconfig,
        pre_postconfig_state,
        reference_sources,
        debug,
    ):
        Path(cache_dir, "artifact.txt").write_text("payload", encoding="utf-8")
        return 0.5, {"cache_dir": cache_dir}

    monkeypatch.setattr(scalecua_verify.tempfile, "mkdtemp", lambda prefix: str(owned))
    monkeypatch.setattr(scalecua_verify, "_evaluate_scalecua_task", fake_evaluate)

    assert await scalecua_verify.evaluate_scalecua_task(
        _FakeComputer(),
        {"func": "score_metric"},
        runtime_split="train",
        debug=True,
    ) == (0.5, {"cache_dir": str(owned)})
    assert (owned / "artifact.txt").read_text(encoding="utf-8") == "payload"


@pytest.mark.asyncio
async def test_scalecua_final_fn_removes_owned_postconfig_cache_dir(monkeypatch, tmp_path):
    owned = tmp_path / "final-owned"
    task = SimpleNamespace(
        metadata={
            "evaluator": {"func": "score_metric", "postconfig": [{"type": "noop"}]},
            "scalecua": {"runtime_split": "train"},
        }
    )

    async def fake_capture(_computer, _evaluator):
        return "pre-state"

    async def fake_postconfig(_computer, evaluator, cache_dir):
        Path(cache_dir, "postconfig.txt").write_text("payload", encoding="utf-8")
        evaluator["_postconfig_done"] = True

    async def fake_evaluate_scalecua_task(
        _computer,
        _evaluator,
        *,
        runtime_split,
        cache_dir,
        run_postconfig,
        pre_postconfig_state,
        reference_sources,
        debug,
    ):
        assert run_postconfig is False
        assert pre_postconfig_state == "pre-state"
        Path(cache_dir, "score.txt").write_text("payload", encoding="utf-8")
        return 0.75

    monkeypatch.setattr(scalecua_verify.tempfile, "mkdtemp", lambda prefix: str(owned))
    monkeypatch.setattr(scalecua_verify, "_capture_pre_postconfig_state", fake_capture)
    monkeypatch.setattr(scalecua_verify, "_run_postconfig", fake_postconfig)
    monkeypatch.setattr(
        scalecua_verify,
        "evaluate_scalecua_task",
        fake_evaluate_scalecua_task,
    )

    assert await scalecua_verify.evaluate_final_fn(task, _FakeComputer()) == 0.75
    assert not owned.exists()


@pytest.mark.asyncio
async def test_scalecua_final_fn_keeps_owned_postconfig_cache_dir_in_debug(
    monkeypatch, tmp_path
):
    owned = tmp_path / "final-debug-owned"
    task = SimpleNamespace(
        metadata={
            "evaluator": {"func": "score_metric", "postconfig": [{"type": "noop"}]},
            "scalecua": {"runtime_split": "train"},
        }
    )

    async def fake_capture(_computer, _evaluator):
        return None

    async def fake_postconfig(_computer, evaluator, cache_dir):
        Path(cache_dir, "postconfig.txt").write_text("payload", encoding="utf-8")
        evaluator["_postconfig_done"] = True

    async def fake_evaluate_scalecua_task(
        _computer,
        _evaluator,
        *,
        runtime_split,
        cache_dir,
        run_postconfig,
        pre_postconfig_state,
        reference_sources,
        debug,
    ):
        return 0.75, {"cache_dir": cache_dir}

    monkeypatch.setattr(scalecua_verify.tempfile, "mkdtemp", lambda prefix: str(owned))
    monkeypatch.setattr(scalecua_verify, "_capture_pre_postconfig_state", fake_capture)
    monkeypatch.setattr(scalecua_verify, "_run_postconfig", fake_postconfig)
    monkeypatch.setattr(
        scalecua_verify,
        "evaluate_scalecua_task",
        fake_evaluate_scalecua_task,
    )

    assert await scalecua_verify.evaluate_final_fn(
        task, _FakeComputer(), debug=True,
    ) == (0.75, {"cache_dir": str(owned)})
    assert (owned / "postconfig.txt").read_text(encoding="utf-8") == "payload"


@pytest.mark.asyncio
async def test_scalecua_env_blocked_from_getter_bubbles(monkeypatch):
    blocked = EnvBlocked(what="download transport failed")

    async def blocked_result(eval_env, config, cache_dir, runtime_split):
        raise blocked

    monkeypatch.setattr(scalecua_verify, "_get_result", blocked_result)

    with pytest.raises(EnvBlocked) as exc_info:
        await scalecua_verify.evaluate_scalecua_task(
            _FakeComputer(),
            {"func": "score_metric", "result": {}, "expected": {}},
            runtime_split="train",
        )
    assert exc_info.value is blocked


@pytest.mark.asyncio
async def test_scalecua_result_getter_infrastructure_failure_raises_env_blocked(
    monkeypatch,
):
    async def broken_result(eval_env, config, cache_dir, runtime_split):
        raise RuntimeError("container stopped answering")

    monkeypatch.setattr(scalecua_verify, "_get_result", broken_result)

    with pytest.raises(EnvBlocked, match="result getter failed"):
        await scalecua_verify.evaluate_scalecua_task(
            _FakeComputer(),
            {"func": "score_metric", "result": {}, "expected": {}},
            runtime_split="train",
        )


@pytest.mark.asyncio
async def test_scalecua_expected_getter_infrastructure_failure_raises_env_blocked(
    monkeypatch,
):
    async def fake_get_result(eval_env, config, cache_dir, runtime_split):
        return "result"

    async def broken_expected(eval_env, config, cache_dir, runtime_split):
        raise RuntimeError("gold could not be read")

    monkeypatch.setattr(scalecua_verify, "_get_result", fake_get_result)
    monkeypatch.setattr(scalecua_verify, "_get_expected", broken_expected)

    with pytest.raises(EnvBlocked, match="expected getter failed"):
        await scalecua_verify.evaluate_scalecua_task(
            _FakeComputer(),
            {"func": "score_metric", "result": {}, "expected": {}},
            runtime_split="train",
        )


@pytest.mark.asyncio
async def test_scalecua_metric_exception_still_scores_zero(monkeypatch):
    async def fake_get_result(eval_env, config, cache_dir, runtime_split):
        return "result"

    async def fake_get_expected(eval_env, config, cache_dir, runtime_split):
        return "expected"

    def broken_metric(result, expected):
        raise ValueError("metric crashed")

    monkeypatch.setattr(scalecua_verify, "_get_result", fake_get_result)
    monkeypatch.setattr(scalecua_verify, "_get_expected", fake_get_expected)
    monkeypatch.setattr(judges, "resolve_metric", lambda name, runtime_split: broken_metric)

    score, detail = await scalecua_verify.evaluate_scalecua_task(
        _FakeComputer(),
        {"func": "score_metric", "result": {}, "expected": {}},
        runtime_split="train",
        debug=True,
    )

    assert score == 0.0
    assert detail["details"] == [
        {"func": "score_metric", "error": "metric crashed", "score": 0.0}
    ]


@pytest.mark.asyncio
async def test_scalecua_metric_env_blocked_bubbles(monkeypatch):
    async def fake_get_result(eval_env, config, cache_dir, runtime_split):
        return "result"

    async def fake_get_expected(eval_env, config, cache_dir, runtime_split):
        return "expected"

    blocked = EnvBlocked(what="metric transport failed")

    def blocked_metric(result, expected):
        raise blocked

    monkeypatch.setattr(scalecua_verify, "_get_result", fake_get_result)
    monkeypatch.setattr(scalecua_verify, "_get_expected", fake_get_expected)
    monkeypatch.setattr(judges, "resolve_metric", lambda name, runtime_split: blocked_metric)

    with pytest.raises(EnvBlocked) as exc_info:
        await scalecua_verify.evaluate_scalecua_task(
            _FakeComputer(),
            {"func": "score_metric", "result": {}, "expected": {}},
            runtime_split="train",
        )
    assert exc_info.value is blocked


@pytest.mark.asyncio
async def test_scalecua_postconfig_failure_raises_env_blocked(monkeypatch):
    async def broken_postconfig(computer, evaluator, cache_dir):
        raise RuntimeError("save command failed")

    async def should_not_evaluate(*args, **kwargs):
        raise AssertionError("postconfig failure should block evaluation")

    monkeypatch.setattr(scalecua_verify, "_run_postconfig", broken_postconfig)
    monkeypatch.setattr(scalecua_verify, "evaluate_scalecua_task", should_not_evaluate)
    task = SimpleNamespace(
        metadata={
            "evaluator": {
                "func": "score_metric",
                "postconfig": [{"type": "sleep", "parameters": {"seconds": 0}}],
            },
            "scalecua": {"runtime_split": "train"},
        }
    )

    with pytest.raises(EnvBlocked, match="postconfig failed"):
        await scalecua_verify.evaluate_final_fn(task, _FakeComputer())


@pytest.mark.asyncio
async def test_scalecua_task_postconfig_failure_raises_env_blocked(monkeypatch):
    async def broken_postconfig(computer, evaluator, cache_dir):
        raise RuntimeError("save command failed")

    monkeypatch.setattr(scalecua_verify, "_run_postconfig", broken_postconfig)

    with pytest.raises(EnvBlocked, match="postconfig failed"):
        await scalecua_verify.evaluate_scalecua_task(
            _FakeComputer(),
            {
                "func": "score_metric",
                "result": {},
                "expected": {},
                "postconfig": [{"type": "sleep", "parameters": {"seconds": 0}}],
            },
            runtime_split="train",
        )


class _ReadFailureInterface(_FakeInterface):
    def __init__(self, exc: BaseException):
        super().__init__()
        self.exc = exc

    async def read_bytes(self, path: str) -> bytes:
        raise self.exc


@pytest.mark.asyncio
async def test_scalecua_controller_missing_file_returns_none(tmp_path):
    env = judges.make_eval_env(
        SimpleNamespace(interface=_ReadFailureInterface(FileNotFoundError("missing"))),
        str(tmp_path),
    )

    assert await asyncio.to_thread(
        env.controller.get_file,
        "/home/user/Desktop/missing.txt",
    ) is None


@pytest.mark.asyncio
async def test_scalecua_controller_file_transport_failure_raises_env_blocked(tmp_path):
    env = judges.make_eval_env(
        SimpleNamespace(interface=_ReadFailureInterface(RuntimeError("exec pipe closed"))),
        str(tmp_path),
    )

    with pytest.raises(EnvBlocked, match="file could not be read"):
        await asyncio.to_thread(
            env.controller.get_file,
            "/home/user/Desktop/result.txt",
        )


@pytest.mark.asyncio
async def test_scalecua_direct_file_materialization_transport_failure_is_env_blocked(
    tmp_path,
):
    env = judges.make_eval_env(
        SimpleNamespace(interface=_ReadFailureInterface(RuntimeError("exec pipe closed"))),
        str(tmp_path),
    )

    with pytest.raises(EnvBlocked, match="file could not be materialized"):
        await judges._download_vm_path(env, "/home/user/Desktop/result.xlsx", str(tmp_path))


def test_scalecua_response_raise_for_status_is_env_blocked():
    response = judges._ResponseShim(status_code=599, text="connection refused")

    with pytest.raises(EnvBlocked, match="HTTP request failed"):
        response.raise_for_status()


@pytest.mark.asyncio
async def test_scalecua_request_in_container_transport_599_is_env_blocked(tmp_path):
    computer = _FakeComputer(
        stdout=json.dumps(
            {
                "status_code": 599,
                "headers": {},
                "text": "connection refused",
                "content_b64": "",
            }
        )
    )
    env = judges.make_eval_env(computer, str(tmp_path))

    with pytest.raises(EnvBlocked, match="HTTP request transport failed"):
        await asyncio.to_thread(
            env.request_in_container,
            "GET",
            "http://lite-scalecua-vm:5000/state",
        )


def test_scalecua_score_coercion_clamps_generated_metric_noise():
    assert scalecua_verify._coerce_score(0.25) == 0.25
    assert scalecua_verify._coerce_score(0.9999999999999999) == 1.0
    assert scalecua_verify._coerce_score(1.0555555555555556) == 1.0
    assert scalecua_verify._coerce_score(-0.1) == 0.0


@pytest.mark.asyncio
async def test_scalecua_clickboard_metric_uses_official_config_first_order(
    monkeypatch,
):
    async def fake_get_result(eval_env, config, cache_dir, runtime_split):
        return "/home/user/Data3/List3\n"

    async def fake_get_expected(eval_env, config, cache_dir, runtime_split):
        return {"expected": "/home/user/Data3/List3"}

    def is_in_vm_clickboard(config, terminal_output):
        assert config == {"expected": "/home/user/Data3/List3"}
        assert terminal_output == "/home/user/Data3/List3\n"
        return 1.0

    monkeypatch.setattr(scalecua_verify, "_get_result", fake_get_result)
    monkeypatch.setattr(scalecua_verify, "_get_expected", fake_get_expected)
    monkeypatch.setattr(
        judges,
        "resolve_metric",
        lambda name, runtime_split: is_in_vm_clickboard,
    )

    score = await scalecua_verify.evaluate_scalecua_task(
        _FakeComputer(),
        {
            "func": "is_in_vm_clickboard",
            "result": {
                "type": "vm_command_line",
                "command": "xsel --clipboard --output",
            },
            "expected": {
                "type": "rule",
                "rules": {"expected": "/home/user/Data3/List3"},
            },
        },
        runtime_split="rl",
    )

    assert score == 1.0


@pytest.mark.asyncio
async def test_scalecua_vm_file_converts_to_bytes_for_annotated_metrics(monkeypatch, tmp_path):
    result_path = tmp_path / "result.xlsx"
    result_path.write_bytes(b"xlsx-bytes")

    async def fake_get_result(eval_env, config, cache_dir, runtime_split):
        return str(result_path)

    async def fake_get_expected(eval_env, config, cache_dir, runtime_split):
        return {"ok": True}

    def metric(result: bytes, expected):
        assert result == b"xlsx-bytes"
        return 1.0

    monkeypatch.setattr(scalecua_verify, "_get_result", fake_get_result)
    monkeypatch.setattr(scalecua_verify, "_get_expected", fake_get_expected)
    monkeypatch.setattr(judges, "resolve_metric", lambda name, runtime_split: metric)

    score = await scalecua_verify.evaluate_scalecua_task(
        _FakeComputer(),
        {
            "func": "bytes_metric",
            "result": {
                "type": "vm_file",
                "path": "/home/user/Documents/result.xlsx",
                "dest": "result.xlsx",
            },
            "expected": {"type": "rule", "rules": {"ok": True}},
        },
        runtime_split="train",
    )

    assert score == 1.0


def test_scalecua_vm_file_converts_to_text_for_known_generated_prefs_metrics(tmp_path):
    prefs = 'user_pref("mail.identity.id1.auto_quote", false);\n'
    result_path = tmp_path / "prefs.js"
    result_path.write_text(prefs, encoding="utf-8")

    def check_tb_dual_auto_quote_reply__07795e258bd2a0b78003c553ab0b53c5(
        result,
        expected,
    ):
        return 1.0

    def ordinary_metric(result, expected):
        return 1.0

    converted = scalecua_verify._prepare_metric_result(
        check_tb_dual_auto_quote_reply__07795e258bd2a0b78003c553ab0b53c5,
        str(result_path),
        {"type": "vm_file"},
    )
    unchanged = scalecua_verify._prepare_metric_result(
        ordinary_metric,
        str(result_path),
        {"type": "vm_file"},
    )

    assert converted == prefs
    assert unchanged == str(result_path)


def test_scalecua_thunderbird_prefs_accepts_generated_rule_method_aliases(tmp_path):
    prefs = "\n".join(
        [
            'user_pref("mail.identity.id1.attach_vcard", true);',
            'user_pref("mail.identity.id1.useremail", "anonym-x2024@outlook.com");',
            'user_pref("layout.spellcheckDefault", 2);',
            'user_pref("mail.smtpservers", "smtp1,smtp2");',
            'user_pref("mail.server.server1.type", "none");',
        ]
    )
    result_path = tmp_path / "prefs.js"
    result_path.write_text(prefs, encoding="utf-8")
    metric = judges.resolve_metric("check_thunderbird_prefs", "train")

    score = metric(
        str(result_path),
        {
            "expect": {
                "mail.identity.id1.attach_vcard": {"method": "==", "ref": True},
                "mail.identity.id1.useremail": {
                    "method": "literal",
                    "ref": "anonym-x2024@outlook.com",
                },
                "layout.spellcheckDefault": {"method": "in", "ref": [1, 2]},
                "mail.smtpservers": {"method": "contains", "ref": "smtp"},
            },
            "unexpect": {
                "mail.server.server1.type": {"method": "in", "ref": ["imap", "pop3"]},
            },
        },
    )

    assert score == 1.0


@pytest.mark.asyncio
async def test_scalecua_thunderbird_prefs_metric_does_not_leak_injected_env(tmp_path):
    env = judges.make_eval_env(_FakeComputer(), str(tmp_path))
    metric = judges.resolve_metric("check_thunderbird_prefs", "train")
    expected = {
        "expect": {
            "mail.server.default.autosync_offline_stores": {
                "method": "eq",
                "ref": True,
            }
        }
    }

    good_path = tmp_path / "good-prefs.js"
    good_path.write_text(
        'user_pref("mail.server.default.autosync_offline_stores", true);\n',
        encoding="utf-8",
    )
    wrong_path = tmp_path / "wrong-prefs.js"
    wrong_path.write_text(
        'user_pref("mail.server.default.autosync_offline_stores", false);\n',
        encoding="utf-8",
    )

    assert await judges.call_metric(metric, env, str(good_path), expected, {}) == 1.0
    assert await judges.call_metric(metric, env, str(wrong_path), expected, {}) == 0.0


@pytest.mark.asyncio
async def test_scalecua_generated_thunderbird_prefs_metric_does_not_leak_injected_env(
    tmp_path,
):
    env = judges.make_eval_env(_FakeComputer(), str(tmp_path))
    metric = judges.resolve_metric(
        "check_thunderbird_prefs__f0f94c7477cc672daff3cf5ec91ccac4",
        "train",
    )
    expected = {
        "expect": {
            "mail.check_all_imap_folders_for_new": {"method": "eq", "ref": True},
            "mail.server.default.check_new_mail": {"method": "eq", "ref": True},
            "mail.server.default.check_time": {"method": "eq", "ref": 5},
        }
    }

    good_path = tmp_path / "good-generated-prefs.js"
    good_path.write_text(
        "\n".join(
            [
                'user_pref("mail.check_all_imap_folders_for_new", true);',
                'user_pref("mail.server.default.check_new_mail", true);',
                'user_pref("mail.server.default.check_time", 5);',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    wrong_path = tmp_path / "wrong-generated-prefs.js"
    wrong_path.write_text(
        "\n".join(
            [
                'user_pref("mail.check_all_imap_folders_for_new", true);',
                'user_pref("mail.server.default.check_new_mail", true);',
                'user_pref("mail.server.default.check_time", 15);',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert await judges.call_metric(metric, env, str(good_path), expected, {}) == 1.0
    assert await judges.call_metric(metric, env, str(wrong_path), expected, {}) == 0.0
