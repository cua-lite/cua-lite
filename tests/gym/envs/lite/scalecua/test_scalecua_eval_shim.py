"""ScaleCUA evaluator shim and metric-call tests."""

from __future__ import annotations

import os
import tempfile
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
async def test_scalecua_eval_scratch_dir_is_removed_by_whoever_created_it(
    monkeypatch, tmp_path
):
    """Every episode used to leave a ``/tmp/scalecua_eval_*`` dir behind forever.

    Cleanup follows OWNERSHIP: a dir this function created is removed on the way
    out, a caller-supplied one is left alone (``evaluate_final_fn`` keeps using
    its own after the call and removes it itself), and a ``debug`` run keeps the
    artifacts so a failure can still be inspected.
    """
    created: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def tracking_mkdtemp(*args, **kwargs):
        # Rooted under tmp_path so an artifact this test deliberately keeps is
        # still cleaned up by pytest.
        path = real_mkdtemp(*args, **{**kwargs, "dir": str(tmp_path)})
        created.append(path)
        return path

    monkeypatch.setattr(tempfile, "mkdtemp", tracking_mkdtemp)

    async def fake_get_result(eval_env, config, cache_dir, runtime_split):
        return "result"

    async def fake_get_expected(eval_env, config, cache_dir, runtime_split):
        return "result"

    monkeypatch.setattr(scalecua_verify, "_get_result", fake_get_result)
    monkeypatch.setattr(scalecua_verify, "_get_expected", fake_get_expected)
    monkeypatch.setattr(
        judges,
        "resolve_metric",
        lambda name, runtime_split: lambda result, expected=None: 1.0,
    )
    evaluator = {"func": "score_metric", "result": {}, "expected": {}}

    assert await scalecua_verify.evaluate_scalecua_task(
        _FakeComputer(), dict(evaluator), runtime_split="train"
    ) == 1.0
    assert len(created) == 1
    assert not os.path.exists(created[0])

    # A caller-supplied dir belongs to the caller.
    caller_dir = tmp_path / "caller"
    caller_dir.mkdir()
    assert await scalecua_verify.evaluate_scalecua_task(
        _FakeComputer(), dict(evaluator), runtime_split="train", cache_dir=str(caller_dir)
    ) == 1.0
    assert len(created) == 1
    assert caller_dir.is_dir()

    # --debug keeps what it collected.
    await scalecua_verify.evaluate_scalecua_task(
        _FakeComputer(), dict(evaluator), runtime_split="train", debug=True
    )
    assert len(created) == 2
    assert os.path.isdir(created[1])


@pytest.mark.asyncio
async def test_scalecua_final_fn_removes_the_postconfig_scratch_dir(monkeypatch, tmp_path):
    """``evaluate_final_fn`` owns the dir it opens for postconfig, on every exit."""
    created: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def tracking_mkdtemp(*args, **kwargs):
        # Rooted under tmp_path so an artifact this test deliberately keeps is
        # still cleaned up by pytest.
        path = real_mkdtemp(*args, **{**kwargs, "dir": str(tmp_path)})
        created.append(path)
        return path

    monkeypatch.setattr(tempfile, "mkdtemp", tracking_mkdtemp)

    seen_cache_dirs: list[str | None] = []

    async def fake_run_postconfig(computer, evaluator, cache_dir):
        return None

    async def fake_evaluate(computer, evaluator, *, cache_dir=None, **kwargs):
        seen_cache_dirs.append(cache_dir)
        return 1.0

    monkeypatch.setattr(scalecua_verify, "_run_postconfig", fake_run_postconfig)
    monkeypatch.setattr(scalecua_verify, "evaluate_scalecua_task", fake_evaluate)
    task = SimpleNamespace(
        metadata={
            "evaluator": {"func": "rule", "postconfig": [{"type": "launch"}]},
            "scalecua": {"runtime_split": "train"},
        }
    )

    assert await scalecua_verify.evaluate_final_fn(task, None) == 1.0

    assert len(created) == 1
    assert seen_cache_dirs == [created[0]]
    assert not os.path.exists(created[0])

    # The metric path raising must not turn the scratch dir into a leak.
    async def boom(*args, **kwargs):
        raise RuntimeError("metric exploded")

    monkeypatch.setattr(scalecua_verify, "evaluate_scalecua_task", boom)
    with pytest.raises(RuntimeError):
        await scalecua_verify.evaluate_final_fn(task, None)

    assert len(created) == 2
    assert not os.path.exists(created[1])


@pytest.mark.parametrize("failing_getter", ["_get_result", "_get_expected"])
@pytest.mark.asyncio
async def test_scalecua_unscoreable_episode_does_not_become_a_score(
    monkeypatch, failing_getter
):
    """An env-side failure must leave the loop as a VOID EPISODE, not as a reward.

    The getters reach into ``base_runner``, which raises ``EnvBlocked`` when it cannot
    read the artifact at all. ``EnvBlocked`` is the drop-the-trajectory signal
    (lite/gym/errors.py): rollout writes the reason and excludes the sample from the
    denominator. Degrading it to 0.0 instead would teach the model that infrastructure
    failures are its own fault -- on the path GRPO trains on.
    """

    async def blocked(eval_env, config, cache_dir, runtime_split):
        raise EnvBlocked(what="osworld container stopped answering")

    async def fine(eval_env, config, cache_dir, runtime_split):
        return "artifact"

    monkeypatch.setattr(scalecua_verify, "_get_result", fine)
    monkeypatch.setattr(scalecua_verify, "_get_expected", fine)
    monkeypatch.setattr(scalecua_verify, failing_getter, blocked)
    monkeypatch.setattr(
        judges,
        "resolve_metric",
        lambda name, runtime_split: lambda result, expected=None: 1.0,
    )

    with pytest.raises(EnvBlocked):
        await scalecua_verify.evaluate_scalecua_task(
            _FakeComputer(),
            {"func": "score_metric", "result": {}, "expected": {}},
            runtime_split="train",
        )


@pytest.mark.asyncio
async def test_scalecua_metric_that_ran_and_threw_still_scores_zero(monkeypatch):
    """The other half of the split: a judge that looked at the artifacts and threw.

    That IS a verdict, however crude, so it degrades to 0.0 as before -- the generated
    overlay metrics that call helpers this overlay never defines must not start voiding
    episodes.
    """

    async def fine(eval_env, config, cache_dir, runtime_split):
        return "artifact"

    def exploding_metric(result, expected=None):
        raise NameError("_verify_single_check__7767eef2")

    monkeypatch.setattr(scalecua_verify, "_get_result", fine)
    monkeypatch.setattr(scalecua_verify, "_get_expected", fine)
    monkeypatch.setattr(
        judges, "resolve_metric", lambda name, runtime_split: exploding_metric
    )

    score = await scalecua_verify.evaluate_scalecua_task(
        _FakeComputer(),
        {"func": "score_metric", "result": {}, "expected": {}},
        runtime_split="train",
    )

    assert score == 0.0


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


async def _always_isolatable(*_a) -> bool:
    return True


@pytest.mark.asyncio
async def test_a_single_positional_metric_reaches_the_child_with_the_same_arity(monkeypatch):
    """The in-thread path drops ``expected`` for a 1-positional metric; the child must too.

    The child calls ``fn(result, expected)`` whenever ``expected`` is not None, so sending
    one of these in unchanged kills it on arity -- and a ``TypeError`` comes back as
    ``MetricRaised``, which the caller scores as the agent's 0.0.
    """
    from lite.gym.envs.lite.osworld.src.eval import runner as osworld_runner

    seen = {}

    async def _record(fn_name, result_data, expected_data, opts):
        seen["expected"] = expected_data
        return 1.0

    one_positional = lambda only: 1.0  # noqa: E731
    monkeypatch.setattr(osworld_runner, "_run_metric_isolated", _record)
    monkeypatch.setattr(osworld_runner, "_lookup_metric", lambda _n: one_positional)

    score = await judges.call_metric(
        one_positional,
        SimpleNamespace(run_sync=lambda fn, *a, **k: fn(*a, **k)),
        "/result",
        "/gold",
        {},
        fn_name="one_positional",
    )
    assert score == 1.0
    assert seen["expected"] is None


@pytest.mark.asyncio
async def test_isolatable_is_identity_not_name(monkeypatch):
    """The child resolves by NAME, so the classifier has to compare the FUNCTION.

    A generated overlay shard that shadows an upstream metric name keeps the same
    ``fn_name``; sending it to the child would score the upstream function instead.
    """
    from lite.gym.envs.lite.osworld.src.eval import runner as osworld_runner

    upstream = lambda a, b=None: 1.0  # noqa: E731
    overlay = lambda a, b=None: 0.0  # noqa: E731
    monkeypatch.setattr(osworld_runner, "_lookup_metric", lambda _n: upstream)

    assert await judges._is_isolatable_metric("m", upstream) is True
    assert await judges._is_isolatable_metric("m", overlay) is False


@pytest.mark.asyncio
async def test_an_overlay_shadowed_metric_is_never_sent_to_the_child(monkeypatch):
    """End to end: the classifier says no, so ``_run_metric_isolated`` is not reached."""
    from lite.gym.envs.lite.osworld.src.eval import runner as osworld_runner

    async def _must_not_run(*_a, **_k):
        raise AssertionError("an overlay-shadowed metric reached the child")

    monkeypatch.setattr(osworld_runner, "_run_metric_isolated", _must_not_run)
    monkeypatch.setattr(osworld_runner, "_lookup_metric", lambda _n: (lambda a, b=None: 1.0))

    overlay = lambda a, b=None: 0.5  # noqa: E731
    score = await judges.call_metric(
        overlay,
        SimpleNamespace(run_sync=lambda fn, *a, **k: fn(*a, **k)),
        "/a",
        "/b",
        {},
        fn_name="shadowed",
    )
    assert score == 0.5


@pytest.mark.asyncio
async def test_scalecua_isolated_metric_timeout_is_not_a_score(monkeypatch):
    """A metric killed on its deadline never produced a verdict, so it must not score.

    Routing env-free metrics into the osworld metric child added two failure modes that
    the in-thread path never had -- a deadline kill and a dead child. Neither is a
    statement about the episode, and the caller's generic handler would degrade both to
    0.0 and train GRPO on an infrastructure failure.
    """
    from lite.gym.envs.lite.osworld.src.eval import runner as osworld_runner

    async def _timeout(*_a, **_k):
        raise osworld_runner.MetricTimeout("spin exceeded 90s")

    monkeypatch.setattr(osworld_runner, "_run_metric_isolated", _timeout)
    monkeypatch.setattr(judges, "_is_isolatable_metric", _always_isolatable)

    with pytest.raises(EnvBlocked):
        await judges.call_metric(
            lambda a, b=None: 1.0,
            SimpleNamespace(run_sync=lambda fn, *a, **k: fn(*a, **k)),
            "/a",
            "/b",
            {},
            fn_name="spin",
        )


@pytest.mark.asyncio
async def test_scalecua_isolated_metric_that_threw_is_still_the_agents_zero(monkeypatch):
    """The other half: a metric that RAN and threw is a verdict and stays a 0.0."""
    from lite.gym.envs.lite.osworld.src.eval import runner as osworld_runner

    async def _raised(*_a, **_k):
        raise osworld_runner.MetricRaised("ValueError: bad artifact")

    monkeypatch.setattr(osworld_runner, "_run_metric_isolated", _raised)
    monkeypatch.setattr(judges, "_is_isolatable_metric", _always_isolatable)

    with pytest.raises(osworld_runner.MetricRaised):
        await judges.call_metric(
            lambda a, b=None: 1.0,
            SimpleNamespace(run_sync=lambda fn, *a, **k: fn(*a, **k)),
            "/a",
            "/b",
            {},
            fn_name="boom",
        )
