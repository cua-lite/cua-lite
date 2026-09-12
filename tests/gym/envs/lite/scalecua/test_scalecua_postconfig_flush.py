"""ScaleCUA postconfig and config-flush behavior tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lite.gym.envs.lite.osworld.src.gen.eval import postconfig as osworld_postconfig
from lite.gym.envs.lite.scalecua.src.osworld import judges
from lite.gym.envs.lite.scalecua.src.osworld import verify as scalecua_verify


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


def test_scalecua_libreoffice_registry_getters_trigger_config_flush():
    # Every registry-config writer family (hashed result_type -> base) must gate on.
    for base in (
        "writer_default_font",
        "writer_default_font_size",
        "writer_heading_font",
        "writer_heading_font_size",
        "writer_list_font",
        "writer_caption_font",
        "writer_index_font",
        "writer_font_combo",
        "writer_user_data",
        "writer_multiple_settings",
        "writer_autosave_setting",
        "writer_zoom_level",
        "writer_page_margin",
        "writer_paragraph_spacing",
    ):
        result_type = base + "__deadbeefdeadbeefdeadbeefdeadbeef"
        assert scalecua_verify._needs_libreoffice_config_flush(
            result_type, {"type": result_type}
        ), base


def test_scalecua_libreoffice_flush_marker_catches_xcu_path():
    # File-marker anchor: fires on the registry path even for an unknown result_type.
    cfg = {
        "type": "vm_file__novel_hash",
        "path": "/home/user/.config/libreoffice/4/user/registrymodifications.xcu",
    }
    assert scalecua_verify._needs_libreoffice_config_flush(cfg["type"], cfg)


def test_scalecua_libreoffice_flush_excludes_document_getters():
    # .docx/.odt document getters read the saved file, not the registry -> no flush.
    for base, path in (
        ("writer_line_spacing", "/home/user/report.docx"),
        ("writer_bold_text", "/home/user/report.odt"),
        ("odt_font_sizes", "/home/user/report.odt"),
    ):
        result_type = base + "__cafebabecafebabecafebabecafebabe"
        assert not scalecua_verify._needs_libreoffice_config_flush(
            result_type, {"type": result_type, "path": path}
        ), base


def test_scalecua_libreoffice_flush_is_idempotent():
    import asyncio

    class _Iface:
        def __init__(self):
            self.calls = 0

        async def run_command(self, *a, **k):
            self.calls += 1

    class _Env:
        pass

    env = _Env()
    env.computer = type("C", (), {"interface": _Iface()})()

    asyncio.run(scalecua_verify._flush_libreoffice(env))
    asyncio.run(scalecua_verify._flush_libreoffice(env))

    assert env.computer.interface.calls == 1
    assert getattr(env, "_libreoffice_config_flushed", False) is True
    stats = scalecua_verify._flush_stats_snapshot(env)
    assert stats["libreoffice"]["executed"] == 1
    assert stats["libreoffice"]["skipped_already_flushed"] == 1


def test_scalecua_thunderbird_flush_decider_selects_thunderbird_result_types():
    assert scalecua_verify._needs_thunderbird_config_flush(
        "check_tb_triple_prefs__abc", {"type": "check_tb_triple_prefs__abc"}
    )
    assert scalecua_verify._needs_thunderbird_config_flush(
        "vm_file",
        {"type": "vm_file", "path": "/home/user/.thunderbird/x.default/prefs.js"},
    )
    assert scalecua_verify._needs_thunderbird_config_flush(
        "content_from_vm_file",
        {"src": "/home/user/.thunderbird/p/ImapMail/x/msgFilterRules.dat"},
    )
    # no-op for unrelated / other-app result types
    assert not scalecua_verify._needs_thunderbird_config_flush(
        "gimp_config_file", {"type": "gimp_config_file"}
    )
    assert not scalecua_verify._needs_thunderbird_config_flush("vlc_config", {"type": "vlc_config"})
    assert not scalecua_verify._needs_thunderbird_config_flush("bookmarks", {})


def test_scalecua_vlc_flush_decider_selects_vlc_config_but_not_playing_info():
    assert scalecua_verify._needs_vlc_config_flush("vlc_config", {"type": "vlc_config"})
    assert scalecua_verify._needs_vlc_config_flush(
        "vm_file", {"type": "vm_file", "path": "/home/user/.config/vlc/vlcrc"}
    )
    # vlc_playing_info reads the HTTP interface — flushing (quitting) would break
    # it, so the decider must NOT fire.
    assert not scalecua_verify._needs_vlc_config_flush(
        "vlc_playing_info", {"type": "vlc_playing_info", "dest": "vlc_status.xml"}
    )
    assert not scalecua_verify._needs_vlc_config_flush("gimp_config_file", {})
    assert not scalecua_verify._needs_vlc_config_flush("bookmarks", {})


@pytest.mark.asyncio
async def test_scalecua_thunderbird_flush_is_graceful_and_noop_safe():
    eval_env = SimpleNamespace(computer=_FakeComputer())
    await scalecua_verify._flush_thunderbird(eval_env)
    assert len(eval_env.computer.interface.commands) == 1
    cmd = eval_env.computer.interface.commands[0]
    # graceful: gated on a real window (sent_quit), ctrl+q + SIGTERM, never kill -9
    assert "sent_quit" in cmd and "ctrl+q" in cmd
    assert "pkill -TERM thunderbird" in cmd
    assert "-9" not in cmd
    # idempotent: a second call (e.g. multiple thunderbird result types) no-ops
    await scalecua_verify._flush_thunderbird(eval_env)
    assert len(eval_env.computer.interface.commands) == 1


@pytest.mark.asyncio
async def test_scalecua_vlc_flush_is_graceful_and_noop_safe():
    eval_env = SimpleNamespace(computer=_FakeComputer())
    await scalecua_verify._flush_vlc(eval_env)
    assert len(eval_env.computer.interface.commands) == 1
    cmd = eval_env.computer.interface.commands[0]
    assert "sent_quit" in cmd and "ctrl+q" in cmd
    assert "pkill -TERM vlc" in cmd
    assert "-9" not in cmd
    # #154 RC-FN-1a: a lingering modal (Simple Preferences / privacy prompt)
    # swallows ctrl+q so the [qt] keys never get written. The flush must commit +
    # close any open VLC dialog (Return on the Preferences window) BEFORE quitting,
    # and it must do so ahead of the ctrl+q so the main window's save path runs.
    assert "Preferences" in cmd and "Return" in cmd
    assert cmd.index("Preferences") < cmd.index("ctrl+q")
    await scalecua_verify._flush_vlc(eval_env)
    assert len(eval_env.computer.interface.commands) == 1


@pytest.mark.asyncio
async def test_scalecua_get_result_records_flush_fired_and_execution_counts(
    tmp_path,
    monkeypatch,
):
    async def fake_base_get_result(computer, config, cache_dir):
        return "ok"

    monkeypatch.setattr(scalecua_verify.base_runner, "_get_result", fake_base_get_result)
    eval_env = judges.make_eval_env(_FakeComputer(), str(tmp_path))
    tb_config = {
        "type": "vm_file",
        "path": "/home/user/.thunderbird/x.default/prefs.js",
        "dest": "prefs.js",
    }
    vlc_config = {
        "type": "vm_file",
        "path": "/home/user/.config/vlc/vlcrc",
        "dest": "vlcrc",
    }

    await scalecua_verify._get_result(eval_env, tb_config, str(tmp_path), "rl")
    await scalecua_verify._get_result(eval_env, tb_config, str(tmp_path), "rl")
    await scalecua_verify._get_result(eval_env, vlc_config, str(tmp_path), "rl")
    await scalecua_verify._get_result(eval_env, vlc_config, str(tmp_path), "rl")
    await scalecua_verify._get_result(
        eval_env,
        {"type": "vlc_playing_info", "dest": "vlc_status.xml"},
        str(tmp_path),
        "rl",
    )

    stats = scalecua_verify._flush_stats_snapshot(eval_env)
    assert stats["thunderbird"]["needed"] == 2
    assert stats["thunderbird"]["executed"] == 1
    assert stats["thunderbird"]["skipped_already_flushed"] == 1
    assert stats["vlc"]["needed"] == 2
    assert stats["vlc"]["executed"] == 1
    assert stats["vlc"]["skipped_already_flushed"] == 1
    flush_calls = [
        call
        for call in eval_env.computer.interface.command_calls
        if "xdotool search --class" in str(call["command"])
    ]
    assert len(flush_calls) == 2


@pytest.mark.asyncio
async def test_scalecua_evaluate_debug_includes_flush_fired_counters(
    tmp_path,
    monkeypatch,
):
    async def fake_base_get_result(computer, config, cache_dir):
        return "ok"

    async def fake_get_expected(eval_env, config, cache_dir, runtime_split):
        return "ok"

    async def fake_call_metric(metric_fn, eval_env, result, expected, opts, fn_name=None):
        return 1.0

    monkeypatch.setattr(scalecua_verify.base_runner, "_get_result", fake_base_get_result)
    monkeypatch.setattr(scalecua_verify, "_get_expected", fake_get_expected)
    monkeypatch.setattr(judges, "resolve_metric", lambda name, split: object())
    monkeypatch.setattr(judges, "call_metric", fake_call_metric)
    evaluator = {
        "func": ["metric_a", "metric_b"],
        "conj": "and",
        "result": [
            {
                "type": "vm_file",
                "path": "/home/user/.thunderbird/x.default/prefs.js",
                "dest": "prefs.js",
            },
            {
                "type": "vm_file",
                "path": "/home/user/.config/vlc/vlcrc",
                "dest": "vlcrc",
            },
        ],
        "expected": [{}, {}],
    }

    reward, debug = await scalecua_verify.evaluate_scalecua_task(
        _FakeComputer(),
        evaluator,
        runtime_split="rl",
        cache_dir=str(tmp_path),
        run_postconfig=False,
        debug=True,
    )

    assert reward == 1.0
    assert debug["scores"] == [1.0, 1.0]
    assert debug["flush_fired_counts"] == {"thunderbird": 1, "vlc": 1}
    assert debug["flush_counters"] == {
        "thunderbird_flush_fired": 1,
        "vlc_flush_fired": 1,
    }
    assert debug["flush_stats"]["thunderbird"]["executed"] == 1
    assert debug["flush_stats"]["vlc"]["executed"] == 1


def test_scalecua_postconfig_upgrades_vlc_pkill_to_graceful_quit():
    out = osworld_postconfig.normalize_postconfig(
        [
            {"type": "execute", "parameters": {"command": "pkill vlc", "shell": True}},
            {"type": "sleep", "parameters": {"seconds": 1}},
        ],
        "vlc",
    )
    upgraded = out[0]["parameters"]["command"]
    assert "pkill vlc" != upgraded
    assert "ctrl+q" in upgraded and "pkill -TERM vlc" in upgraded and "-9" not in upgraded
    # non-kill steps preserved
    assert out[1] == {"type": "sleep", "parameters": {"seconds": 1}}


def test_scalecua_postconfig_upgrades_vlc_killall_9_form():
    out = osworld_postconfig.normalize_postconfig(
        [{"type": "execute", "parameters": {"command": ["killall", "-9", "vlc"], "shell": True}}],
        "vlc",
    )
    assert "ctrl+q" in out[0]["parameters"]["command"]


def test_scalecua_postconfig_leaves_non_vlc_kill_untouched():
    steps = [{"type": "execute", "parameters": {"command": "pkill firefox", "shell": True}}]
    assert osworld_postconfig.normalize_postconfig(steps, "chrome") == steps


async def test_scalecua_flush_is_best_effort_and_never_fails_reward():
    # #154: a config-flush that hangs/times out under load must NOT fail the reward
    # (_flush_vlc's run_command TimeoutExpired propagated to reward 0.0). The
    # _best_effort_flush decorator swallows Exception, keeps the success path
    # intact, and lets asyncio.CancelledError (BaseException) still propagate.
    import asyncio

    import pytest

    from lite.gym.envs.lite.scalecua.src.osworld import verify as V

    ran = []

    @V._best_effort_flush
    async def _raises(eval_env):
        raise TimeoutError("flush hung under load")

    @V._best_effort_flush
    async def _ok(eval_env):
        ran.append("ran")
        return "done"

    @V._best_effort_flush
    async def _cancel(eval_env):
        raise asyncio.CancelledError()

    assert await _raises(None) is None  # swallowed -> reward not failed
    assert await _ok(None) == "done" and ran == ["ran"]  # success path unchanged
    with pytest.raises(asyncio.CancelledError):  # cancellation still propagates
        await _cancel(None)
