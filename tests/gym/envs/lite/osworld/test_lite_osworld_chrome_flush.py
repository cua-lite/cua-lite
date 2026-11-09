"""Unit tests for the osworld Chrome read-before-flush gate (#154 read-before-flush).

The osworld eval runner reads on-disk Chrome state (Preferences / Bookmarks /
History) which Chrome commits lazily (in-memory until a commit timer / clean
shutdown). `_needs_chrome_profile_flush` whitelists ONLY the on-disk result_types
(+ a profile-dir file-marker) so a per-getter graceful-quit flush persists the
agent's state before the read, while the live CDP/accessibility readers (which a
quit would break) are excluded. Mirrors the LibreOffice flush gate.

Run: uv run pytest tests/gym/envs/lite/osworld/test_lite_osworld_chrome_flush.py
"""

from __future__ import annotations

import asyncio
import shlex
import subprocess

import pytest

from lite.gym.envs.lite.osworld.src.eval import runner


def test_chrome_ondisk_result_types_trigger_flush():
    # Every whitelisted on-disk Chrome getter (hashed result_type -> base) gates on.
    for base in (
        "enable_do_not_track",
        "check_dns_prefetch",
        "enable_safe_browsing",
        "data_delete_automacally",
        "new_startup_page",
        "default_search_engine",
        "chrome_font_size",
        "profile_name",
        "chrome_color_scheme",
        "bookmarks",
        "cookie_data",
        "history",
        "find_unpacked_extension_path",
        "find_installed_extension_name",
    ):
        rt = base + "__deadbeefdeadbeefdeadbeefdeadbeef"
        assert runner._needs_chrome_profile_flush(rt, {"type": rt}), base


def test_chrome_flush_file_marker_catches_profile_path():
    cfg = {"type": "vm_file__novel", "path": "/home/user/chrome-data/Default/Preferences"}
    assert runner._needs_chrome_profile_flush(cfg["type"], cfg)


def test_chrome_flush_excludes_live_readers():
    # Live CDP / accessibility-tree readers must NOT flush (a quit breaks them).
    for base in (
        "active_url",
        "active_url_from_accessTree",
        "active_tab_info",
        "open_tabs_info",
        "active_tab_url_parse",
        "active_tab_html_parse",
        "chrome_appearance_mode_ui",
    ):
        rt = base + "__cafebabecafebabecafebabecafebabe"
        assert not runner._needs_chrome_profile_flush(rt, {"type": rt}), base


def test_chrome_flush_is_a_noop_when_no_chrome(monkeypatch):
    # The flush body reduces to a bare sync when no chrome window exists; it must
    # never fabricate reward — here we just assert it runs one controller command.
    calls = []

    class _Iface:
        async def run_command(self, *a, **k):
            calls.append((a, k))

    class _Computer:
        interface = _Iface()

    asyncio.run(runner._flush_chrome_profile(_Computer()))
    assert len(calls) == 1


_ZOMBIE_ONLY_PGREP = """#!/bin/sh
# The only matching process is a zombie: match unless a runstate filter (-r)
# that leaves out Z is given.
while [ $# -gt 0 ]; do
  if [ "$1" = "-r" ]; then case "$2" in *Z*) exit 0 ;; *) exit 1 ;; esac; fi
  shift
done
exit 0
"""


@pytest.mark.parametrize("flush", ["_flush_chrome_profile", "_flush_libreoffice_config"])
def test_flush_wait_loop_ignores_zombie_process(tmp_path, flush):
    # The in-container server never reaps the apps it launches, so after the app
    # exits it stays a zombie under the same name. The wait loop must treat that as
    # "exited" and stop at once instead of signalling a dead process 30 times.
    captured = []

    class _Iface:
        async def run_command(self, command, **_kw):
            captured.append(command)

    class _Computer:
        interface = _Iface()

    asyncio.run(getattr(runner, flush)(_Computer()))
    script = shlex.split(captured[0])[-1]  # "bash -lc '<script>'" -> <script>

    stubs = tmp_path / "bin"
    stubs.mkdir()
    pkill_log = tmp_path / "pkill.log"
    for name, body in {
        "pgrep": _ZOMBIE_ONLY_PGREP,
        "pkill": f'#!/bin/sh\necho "$*" >> {pkill_log}\n',
        "seq": "#!/bin/sh\ni=$1; while [ $i -le $2 ]; do echo $i; i=$((i+1)); done\n",
        "sleep": "#!/bin/sh\n", "sync": "#!/bin/sh\n",
        "xdotool": "#!/bin/sh\n", "python3": "#!/bin/sh\ncat >/dev/null\n",
    }.items():
        (stubs / name).write_text(body)
        (stubs / name).chmod(0o755)
    # Non-login bash with only the stubs on PATH: nothing here can signal a real process.
    subprocess.run(["/bin/bash", "-c", script], env={"PATH": str(stubs)}, check=True, timeout=30)
    assert not pkill_log.exists(), pkill_log.read_text()


def test_osworld_flush_is_best_effort_and_never_fails_reward():
    # #154 (cross-env mirror of scalecua): an osworld config-flush that hangs/
    # times out under load must NOT fail the reward. The _best_effort_flush
    # decorator swallows Exception, keeps the success path intact, and lets
    # asyncio.CancelledError (BaseException) still propagate.
    import asyncio

    import pytest

    from lite.gym.envs.lite.osworld.src.eval import runner as R

    ran = []

    @R._best_effort_flush
    async def _raises(computer):
        raise TimeoutError("flush hung under load")

    @R._best_effort_flush
    async def _ok(computer):
        ran.append("ran")
        return "done"

    @R._best_effort_flush
    async def _cancel(computer):
        raise asyncio.CancelledError()

    assert asyncio.run(_raises(None)) is None  # swallowed -> reward not failed
    assert asyncio.run(_ok(None)) == "done" and ran == ["ran"]
    with pytest.raises(asyncio.CancelledError):  # cancellation still propagates
        asyncio.run(_cancel(None))
