"""Unit tests for lite.cuagym (CUA-Gym browser + desktop on one container env).

These need no container — the container-spawning rollout (reset → setup_fn) is
covered by the batch-consistency harness. Here we test the pure helpers and that
both backends register under the single ``lite.cuagym`` env_id and dispatch to
the right per-task image/platform.

Run:
    uv run --no-sync python -m pytest tests/gym/envs/lite/cuagym/test_cuagym.py -v
"""

from __future__ import annotations

import ast
import inspect
import io
import json
import re
import subprocess
import tarfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import zstandard

from lite.gym.envs.lite.cuagym import main as M
from lite.gym.envs.lite.cuagym.scripts.utils import (
    import_desktop_tasks,
    import_web_tasks,
)
from lite.gym.envs.lite.cuagym.src.browser import scripts as browser
from lite.gym.envs.lite.cuagym.src.desktop import scripts as desktop
from lite.gym.envs.lite.cuagym.src.utils import (
    container,
    dataset,
    display,
    runtime,
)
from lite.gym.envs.lite.cuagym.src.utils.reward import REWARD_RE, parse_reward
from lite.gym.errors import (
    CapacityExhausted,
    CuaGymTaskError,
    EnvDepsMissingError,
    EnvDesktopCrashed,
    is_retryable,
)
from lite.gym.sandbox.base import SandboxBaseEnv
from lite.gym.types import LiteEnvObservation

# --- shared reward parsing -------------------------------------------------


def _require_fresh_task_cache() -> None:
    try:
        M._register_tasks()
    except EnvDepsMissingError as exc:
        pytest.skip(str(exc))


def test_reward_parse_case_insensitive_last_wins():
    # CUA-Gym reward.py output is inconsistent: "REWARD: x" and "reward:x" both parse.
    assert parse_reward("diag\nREWARD: 0.5\n") == 0.5
    assert parse_reward("x\nreward:1.0\n") == 1.0
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        parse_reward("REWARD: -1")
    assert parse_reward("REWARD: 0.2\nREWARD: 0.9") == 0.9  # last match wins
    with pytest.raises(ValueError, match="REWARD sentinel"):
        parse_reward("no score here")
    assert REWARD_RE.findall("REWARD: 1.0") == ["1.0"]


def test_desktop_uno_scripts_use_system_python():
    from lite.gym.envs.lite.cuagym.src.desktop.scripts import (
        _python_command_for_source,
        _python_for_source,
    )

    assert _python_for_source("import uno\n") == "/opt/env/uno-venv/bin/python"
    assert _python_for_source("from unohelper import Base\n") == "/opt/env/uno-venv/bin/python"
    assert _python_for_source("import gi\n") == "/opt/env/uno-venv/bin/python"
    assert (
        _python_for_source("from gi.repository import Gio\n")
        == "/opt/env/uno-venv/bin/python"
    )
    assert _python_for_source("import fitz\n") == "/opt/env/venv/bin/python"
    # The interpreter runs as the desktop user on the env venv (no GUI-drop shim
    # on PATH now). UNO sources keep a PATH prefix so bare-`python3` children stay
    # on the UNO ABI; env-venv sources are just the bare interpreter.
    assert _python_command_for_source("import uno\n") == (
        "PATH=/opt/env/uno-venv/bin:$PATH /opt/env/uno-venv/bin/python"
    )
    assert _python_command_for_source("import fitz\n") == "/opt/env/venv/bin/python"


def test_install_imports_tasks_from_the_current_checkout():
    repo = Path(__file__).resolve().parents[5]
    script = (
        repo / "lite/gym/envs/lite/cuagym/scripts/install.sh"
    ).read_text()

    assert script.index('cd "$REPO_ROOT"') < script.index("DEFAULT_IMAGE=")
    assert (
        "python -m lite.gym.envs.lite.cuagym.scripts.utils.import_tasks "
        "--backend web"
    ) in script
    assert (
        "python -m lite.gym.envs.lite.cuagym.scripts.utils.import_tasks "
        "--backend desktop"
    ) in script
    assert 'python "$SCRIPT_DIR/utils/import_tasks.py"' not in script


# --- browser setup helpers -------------------------------------------------


def test_browser_referenced_apps_and_materialize():
    src = "BASE='__CUA_GYM_INSTAGRAM_URL__'\nH='__CUA_GYM_GITHUB_HOST__'"
    assert browser._referenced_apps(src) == ["github", "instagram"]
    out = browser._materialize(src)
    assert f"http://127.0.0.1:{browser._port_for('instagram')}" in out
    assert f"127.0.0.1:{browser._port_for('github')}" in out  # HOST → no scheme
    assert "__CUA_GYM_" not in out


def test_browser_bundle_render_errors_are_nonretryable_task_failures(tmp_path):
    source = tmp_path / "initial_setup.py"
    source.write_text("print('__CUA_GYM_UNKNOWN__')\n")

    with pytest.raises(CuaGymTaskError) as raised:
        browser._load_bundle_script(source, [], phase="setup")

    assert raised.value.phase == "setup"
    assert raised.value.kind == "invalid_bundle"


def test_browser_port_deterministic_in_range():
    assert browser._port_for("instagram") == browser._port_for("instagram")
    assert 20000 <= browser._port_for("instagram") < 40000
    with pytest.raises(ValueError, match="unsupported CUA-Gym mock app"):
        browser._port_for("not_in_the_pinned_mock_hub")


def test_browser_known_mock_ports_are_unique():
    apps_file = M._DIR / ".cache" / "web" / "lite.cuagym_tasks" / "apps.txt"
    if not apps_file.exists():
        pytest.skip(
            "mock app list not imported "
            "(run lite/gym/envs/lite/cuagym/scripts/install.sh provision)"
        )
    apps = [
        line.strip().removesuffix("_mock")
        for line in apps_file.read_text().splitlines()
        if line.strip()
    ]
    ports = [browser._port_for(app) for app in apps]
    assert len(ports) == len(set(ports))


@pytest.mark.asyncio
async def test_missing_upstream_mock_is_a_terminal_task_failure():
    class Interface:
        async def run_command(self, _command):
            return SimpleNamespace(returncode=66, stdout="", stderr="")

    with pytest.raises(CuaGymTaskError) as raised:
        await browser._start_mock(SimpleNamespace(interface=Interface()), "github")

    assert raised.value.kind == "missing_mock"
    assert is_retryable(raised.value) is False


def test_baked_mock_list_covers_every_buildable_app():
    _require_fresh_task_cache()
    root = M._DIR / ".cache" / "web" / "lite.cuagym_tasks"
    catalog = root / "train.jsonl"
    apps_file = root / "apps.txt"
    intentionally_unbaked = set()
    rows = [json.loads(line) for line in catalog.read_text().splitlines()]
    needed = {
        app
        for row in rows
        for app in row["metadata"]["others"]["apps"]
        if app not in intentionally_unbaked
    }
    baked = {
        line.strip().removesuffix("_mock")
        for line in apps_file.read_text().splitlines()
        if line.strip()
    }
    assert needed <= baked
    assert intentionally_unbaked.isdisjoint(baked)


def test_browser_mock_suffix_stripped():
    assert browser._referenced_apps("__CUA_GYM_AMAZON_MOCK_URL__") == ["amazon"]


def test_browser_dynamic_mock_template_uses_task_apps():
    src = (
        "def fetch(mock_name):\n"
        "    return f'__CUA_GYM_MOCK_APP_URL_TEMPLATE__/go?sid=x'\n"
    )
    out = browser._materialize(src, ["jira", "gmail"])
    assert "__CUA_GYM_" not in out
    namespace = {}
    exec(out, namespace)
    assert namespace["fetch"]("jira") == (
        f"http://127.0.0.1:{browser._port_for('jira')}/go?sid=x"
    )


def test_browser_dynamic_mock_template_uses_actual_function_argument():
    src = (
        "def fetch_mock(name):\n"
        "    return f'__CUA_GYM_MOCK_APP_URL_TEMPLATE__/go?sid=x'\n"
    )
    namespace = {}
    exec(browser._materialize(src, ["jira"]), namespace)
    assert namespace["fetch_mock"]("jira") == (
        f"http://127.0.0.1:{browser._port_for('jira')}/go?sid=x"
    )


@pytest.mark.asyncio
async def test_cross_app_setup_opens_missing_tabs_and_restores_landing(monkeypatch):
    commands = []

    async def run(_computer, command, phase, **kwargs):
        commands.append((command, phase, kwargs))
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(browser, "_run_task_command", run)
    await browser._open_missing_app_tabs(
        SimpleNamespace(),
        ["twitter", "slack", "notion"],
    )

    command, phase, kwargs = commands[0]
    assert phase == "cross-app tabs"
    assert kwargs["error_type"] is CuaGymTaskError
    assert f":{browser._port_for('twitter')}/?sid=$sid" in command
    assert f":{browser._port_for('slack')}/?sid=$sid" in command
    assert f":{browser._port_for('notion')}/?sid=$sid" in command
    assert 'for _ in "${urls[@]}"' in command
    assert "/opt/env/bin/xdotool key ctrl+shift+Tab" in command
    assert "ctrl+1" not in command
    assert subprocess.run(
        ["bash", "-n"],
        input=command,
        text=True,
        capture_output=True,
    ).returncode == 0


@pytest.mark.asyncio
async def test_single_app_setup_does_not_open_extra_tabs(monkeypatch):
    async def fail(*_args):
        raise AssertionError("single-app setup should not run a tab command")

    monkeypatch.setattr(browser, "_run_task_command", fail)
    await browser._open_missing_app_tabs(SimpleNamespace(), ["twitter"])


def test_browser_dev_host_rewrite_picks_the_app_owning_the_hardcoded_port():
    # 5efc831f-…-a5f2991e478b is slack+github and hardcodes the author's dev box on
    # slack's upstream port (8047); sorted `apps[0]` would send it to github's mock.
    src = "URL = 'http://172.17.46.46:8047/?sid=x'\n"
    out = browser._materialize(src, ["github", "slack"])
    assert f"http://127.0.0.1:{browser._port_for('slack')}/?sid=x" in out
    assert str(browser._port_for("github")) not in out


def test_browser_dev_host_rewrite_falls_back_only_when_one_app_is_named():
    # Unknown upstream port + a single app: the URL can only have meant that mock.
    src = "URL = 'http://172.17.46.46:9999/'\n"
    assert f"http://127.0.0.1:{browser._port_for('jira')}/" in browser._materialize(src, ["jira"])
    # …but with several apps any pick is a coin flip, so refuse instead of silently
    # dialling the wrong mock (a deterministic REWARD: 0.0).
    with pytest.raises(ValueError, match="cannot attribute"):
        browser._materialize(src, ["jira", "slack"])


def test_browser_upstream_ports_are_unique_and_known():
    assert len(set(browser._UPSTREAM_PORTS.values())) == len(browser._UPSTREAM_PORTS)
    for app in browser._UPSTREAM_PORTS:
        browser._port_for(app)  # every entry must be a baked mock


# --- desktop setup helpers -------------------------------------------------


def test_desktop_doc_kinds():
    assert desktop._DOC_KINDS == {"pptx", "docx", "xlsx"}


@pytest.mark.parametrize("source", [
    # the bundles' own helper / an inlined Popen / a desktop opener
    "launch_gui('gimp x.xcf')\n",
    "subprocess.Popen(['soffice', '--calc', p])\n",
    "os.system('xdg-open /home/user/a.pdf')\n",
    # …and the .sh form none of those can see: a backgrounded GUI binary.
    'code "$WORKSPACE" &\n',
    'code "$WORKSPACE" &> /dev/null &\n',
    '    DISPLAY=:0 nohup soffice --calc "$F" >/dev/null 2>&1 &\n',
    '  ( DISPLAY=:0 gimp "${XCF}" || true ) &\n',
    'libreoffice --calc "$X" --norestore &\n',
    'DISPLAY=:0 nautilus /home/user/Testing &\n',
    'setsid /usr/bin/vlc "$VIDEO" &\n',
    # a `\`-continued launch: the `&` lands on a later physical line
    'code "$WS" \\\n  --new-window \\\n  "$FILE" &\n',
    # --- TAIL forms: bash only needs `&` to TERMINATE the command, so an
    # `&[ \t]*$` anchor (which demands `&` be the line's last token) misses all
    # of these. Each one is a REAL launch, and each was measured in the pinned
    # corpus: `& disown` 59 bundles, `( … & )` 67, trailing `#` 9, a following
    # command 1. Relaxing the gate on them turns a dead GUI session into a
    # silent reward-0 trajectory instead of a diagnosable no_task_window.
    'code "$WORKSPACE" & disown\n',
    'DISPLAY=:0 nohup nautilus "${WORKDIR}" >/dev/null 2>&1 & disown\n',
    'DISPLAY=:0 libreoffice --writer "$D/report.docx" & disown || true\n',
    '( libreoffice --impress "${PPT_FILE}" & ) >/dev/null 2>&1 || true\n',
    '(libreoffice --writer "${POLICY_DOC}" >/dev/null 2>&1 &) || true\n',
    '( DISPLAY=:0 gnome-terminal --working-directory="$D" & disown ) || true\n',
    'DISPLAY=:0 gimp "${XCF_FILE}" &       # opens in background\n',
    'libreoffice --impress "$PPTX" &>/dev/null &   LO_IMPRESS_PID=$!\n',
    # a launch whose URL argument itself contains `&` — the trailing `&` still wins
    'DISPLAY=:0 google-chrome "https://x.test/s?type=all&q=pruning" &\n',
    # --- HEAD forms. The old `^…<literal binary>` anchor missed all of these, and
    # 32 pinned bundles launch an app through one of them. Each shape below is
    # copied from the bundle named beside it.
    #
    # a VARIABLE head resolved from a `command -v` probe (12 bundles, the
    # dominant class) — a48cd6f4, 9fb584e0, b4ba374e, c3e3d977, d41fd845, …
    'LIBRE_BIN="$(command -v libreoffice || command -v soffice)"\n'
    'DISPLAY=:0 "$LIBRE_BIN" --writer "$ODT_FILE" >/dev/null 2>&1 &\n',
    'if command -v libreoffice &>/dev/null; then LO="libreoffice"; fi\n'
    'DISPLAY=:0 "$LO" --calc "$SPREADSHEET" & disown\n',
    'find_browser() { command -v google-chrome 2>/dev/null || true; }\n'
    'BROWSER_BIN="$(find_browser)"\n'
    'DISPLAY=:0 "$BROWSER_BIN" --new-window "$URL" >/dev/null 2>&1 &\n',
    'TERMINALS=(\n  "gnome-terminal"\n  "xterm"\n)\n'
    'for term in "${TERMINALS[@]}"; do\n'
    '  ( DISPLAY=:0 "$term" --working-directory="$HOME" >/dev/null 2>&1 & ) || true\n'
    'done\n',
    'VSCODE_CMD="code"        # VS Code CLI executable\n'
    '$VSCODE_CMD "$WORKSPACE" &\n',
    # a `#` in a URL FRAGMENT — the old body was `[^\n#]*`, so it stopped dead
    # before the `&` (933eb164)
    'DISPLAY=:0 google-chrome --new-window "https://x.test/docs/#install" & disown\n',
    # absolute-path wrappers — the old list only accepted BARE nohup/env
    # (d07c963a, cd1deed1, fe1e8f73)
    '/usr/bin/nohup soffice --calc /tmp/m/customers.ods >/dev/null 2>&1 &\n',
    '/usr/bin/env nohup soffice --writer "/home/user/case.docx" >/dev/null 2>&1 &\n',
    '/usr/bin/env DISPLAY=:0 libreoffice --writer "$DOC_PATH" >/dev/null 2>&1 &\n',
    # detached with NO `&` at all — the old pattern structurally required one
    # (f7ec9ac8)
    'DISPLAY=:0 setsid -f chromium --new-window "$DOC_URL" 2>/dev/null || true\n',
    # a head that is not at `^`: after `then`, after `&&`, after a pipeline
    # (d31881ea, c14350a9, c3246db5, cec4b066)
    'if command -v google-chrome >/dev/null; then google-chrome --new-window "$t" & disown\nfi\n',
    'command -v gnome-terminal &>/dev/null && \\\n'
    '  DISPLAY=:0 gnome-terminal --working-directory="$D" & disown || true\n',
    'command -v nautilus >/dev/null 2>&1 && ( nautilus "$WORKDIR" >/dev/null 2>&1 & )\n',
    "echo \"pw\" | sudo -S -u user DISPLAY=:0 chromium --new-window 'https://x.test' &\n",
    # the real command inside a `-c` STRING (7dfe791d, 89689a4d, e5d0f6ef)
    'setsid bash -c "DISPLAY=:0 chromium --password-store=basic \'file://$WS/n.txt\'" &\n',
    'su - user -c "DISPLAY=:0 nautilus /home/user/Webpack &" 2>/dev/null || true\n',
    '/usr/bin/nohup bash -c "sleep 1; DISPLAY=:0 gimp \\"$JPEG\\" &" >/dev/null 2>&1 &\n',
    # `a || b &` backgrounds the whole AND-OR list, so `a` is backgrounded too
    # (c496fcb9)
    'nohup vlc --play-and-pause "$MP4" >/dev/null 2>&1 || true &\n',
    # a backgrounded subshell GROUP whose launch is inside it (fb3346cc)
    '(\n  if command -v google-chrome >/dev/null 2>&1; then exec google-chrome "$P"\n'
    '  fi\n) >/dev/null 2>&1 &\n',
])
def test_desktop_gui_gate_sees_every_launch_form(source):
    assert desktop._script_launches_gui(source) is True


@pytest.mark.parametrize("source", [
    # Naming an app is not launching it — these are what `subprocess.run(...)`
    # actually does in this corpus, and gating on the call form would hard-fail
    # 61 pure file-seeding bundles.
    "subprocess.run(['pkill', '-f', 'soffice'], capture_output=True)\n",
    "subprocess.run(['code', '--install-extension', 'ms-python.python'])\n",
    "subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'reportlab'])\n",
    'echo "now open libreoffice calc yourself" &\n',
    'python3 /home/user/seed_workbook.py &\n',
    # batch conversion raises no window
    'soffice --headless --convert-to pdf "$DOC" &\n',
    'libreoffice --calc "$X" --invisible --minimized &\n',
    # `&&` is a chain, not a background job
    'mkdir -p /home/user/ws && code --version\n',
    'pkill -f gimp && rm -rf /home/user/.config/GIMP\n',
    # …and neither is a `&` INSIDE an argument. This is the false positive the
    # tail's delimiter lookahead exists to prevent (2 bundles in the corpus pass
    # a query string to a FOREGROUND browser); a false positive here hard-fails
    # a legitimate file-seeding task, which is worse than the hole it closes.
    'google-chrome "https://x.test/search?searchtype=all&query=pruning"\n',
    'code --list-extensions 2>&1 | sort > /home/user/exts.txt\n',
    # the pure file-seeding shape the whole relaxation exists for: build the
    # files, tell the AGENT to open the app.
    'set -euo pipefail\n'
    'mkdir -p /home/user/Documents/q3\n'
    'printf "a,b\\n1,2\\n" > /home/user/Documents/q3/sales.csv\n'
    'echo "Open LibreOffice Calc and chart sales.csv"\n',
    # --- the DECOYS the corpus deliberately plants. A false positive here
    # hard-fails a legitimate file-seeding task, which is worse than the hole a
    # wider anchor closes, so each one must stay negative for a STRUCTURAL
    # reason, not a lucky anchor.
    #
    # `exec -a` supplies a FAKE argv[0]; the real command is `sleep`
    # (ca775249, ec425996)
    "bash -c 'exec -a soffice.bin sleep 1800' &   # 30-minute sleep\n",
    'bash -c "exec -a \'libreoffice --writer ${DOC}\' sleep infinity" & PID=$!\n',
    # a launch fragment inside an instructions HERE-DOC is DATA, not a command
    # (e275acf0), and so is a launcher script the setup WRITES for later use
    # (a3de98fb — a real false positive of the old anchor)
    "cat > /home/user/README.md <<'EOF'\n"
    '• finish by opening VS Code with `code "$WORKSPACE" &`\n'
    'EOF\n',
    "cat > \"$WS/launch.sh\" << 'LAUNCH_EOF'\n"
    '#!/bin/bash\nDISPLAY=:0 chromium --new-window "$URL" > /tmp/c.log 2>&1 &\n'
    'LAUNCH_EOF\nchmod +x "$WS/launch.sh"\n',
    # a STUB the script writes over the real binary's name and then runs
    # (bb9c8e1d): `/usr/local/bin/vlc` here is a `sleep 3600`, not VLC
    'STUB_VLC="/usr/local/bin/vlc"\n'
    "tee \"$STUB_VLC\" >/dev/null <<'EOS'\n#!/bin/bash\nsleep 3600\nEOS\n"
    'chmod +x "$STUB_VLC"\n'
    'nohup "$STUB_VLC" "$MKV_FILE" >/dev/null 2>&1 & disown\n',
    # prose must never enter the variable table
    'MSG="Open libreoffice yourself when ready"\necho "$MSG" &\n',
    # a `command -v` probe is a LOOKUP, not a launch
    'command -v libreoffice >/dev/null 2>&1 &\n',
])
def test_desktop_gui_gate_ignores_non_launch_mentions(source):
    assert desktop._script_launches_gui(source) is False


#: An INDEPENDENT over-approximation of "this line could be a GUI launch".
#:
#: The previous oracle here shared ``_GUI_BACKGROUND_RE``'s HEAD anchor verbatim
#: and only varied its tail, so it was blind to exactly the bug it was meant to
#: guard: the head missed 32 launching bundles while ``sum(relaxed) == 283`` kept
#: passing. This shares nothing with the detector — no head anchor, no wrapper
#: list, no tokenizer. It is a deliberately CRUDE co-occurrence test (a GUI name
#: OR a variable head, plus a job-control marker, on one logical line), which
#: over-approximates: every real launch line is in it. The test then requires
#: that nothing it flags survives in the RELAXED population except a pinned,
#: individually-explained residue, so a head regression shows up as a new residue
#: entry naming the bundle rather than as a silently shrinking count.
_SUSPECT_GUI_RE = re.compile(
    r"(?<![\w./-])("
    + "|".join(sorted(desktop._GUI_BINARIES, key=len, reverse=True))
    + r")(?![\w-])"
)
_SUSPECT_JOB_RE = re.compile(r"(?<![&|>])&(?![&>])|\bdisown\b|\bsetsid\b[ \t]+-f\b")
_SUSPECT_VAR_HEAD_RE = re.compile(
    r"(?:^|[;&|(]|\bthen\b)[ \t]*(?:[A-Za-z_]\w*=\S*[ \t]+)*[\"']?\$\{?\w+\}?[\"']?[ \t]"
)

#: The only lines the scan above may still flag inside the relaxed population.
#: Every one is a decoy, verified by reading the bundle: two `exec -a` fake-argv0
#: sleeps, a launch fragment quoted in an instructions here-doc, and a launcher
#: script the setup WRITES for the agent to run later (that last one, a3de98fb,
#: was a genuine FALSE POSITIVE of the old head anchor — it hard-failed a
#: perfectly good file-seeding task on every rollout).
_RELAXED_SUSPECT_RESIDUE = {
    ("a3de98fb-8302-54fc-bdba-ef712e324309", "chromium"),
    ("ca775249-73a0-5716-816a-4e509100d6d0", "soffice"),
    ("e275acf0-d236-599e-93b8-c71385182477", "code"),
    ("ec425996-4516-5576-892c-1953b1f705bf", "libreoffice"),
    ("ec425996-4516-5576-892c-1953b1f705bf", "gedit"),
}


def test_desktop_gui_gate_arms_every_backgrounded_launch_in_the_corpus():
    """Pin the relaxed population, and prove it holds no un-armed launch.

    The old guard asserted ``relaxed/scripts < 0.10``, which could not catch this
    bug class at all. What replaces it is two assertions that fail LOUDLY and
    name names: an independent suspicion scan whose residue inside the relaxed
    population must be exactly the known decoys, and the exact relaxed count with
    its app_type breakdown.
    """
    catalog = M._DIR / ".cache" / "desktop" / "lite.cuagym_desktop_tasks" / "train.jsonl"
    if not catalog.exists():
        pytest.skip(
            "desktop task cache not imported "
            "(run lite/gym/envs/lite/cuagym/scripts/install.sh provision)"
        )
    scripts = 0
    relaxed = Counter()
    suspects = set()
    for line in catalog.read_text().splitlines():
        metadata = json.loads(line)["metadata"]
        if metadata["setup_kind"] in desktop._DOC_KINDS:
            continue
        scripts += 1
        setup = Path(metadata["setup"])
        source = setup.read_text(errors="replace")
        if desktop._script_launches_gui(source):
            continue
        if desktop._open_paths(setup.parent / "task.json"):
            continue
        relaxed[(metadata.get("others") or {}).get("app_type", "?")] += 1
        names_probed = _SUSPECT_GUI_RE.search(source) is not None
        for physical in re.sub(r"\\\n[ \t]*", " ", source).splitlines():
            text = physical.strip()
            if text.startswith("#") or not _SUSPECT_JOB_RE.search(text):
                continue
            if "--headless" in text or "--invisible" in text:
                continue
            named = _SUSPECT_GUI_RE.search(text)
            if named:
                suspects.add((setup.parent.name, named.group(1)))
            elif names_probed and _SUSPECT_VAR_HEAD_RE.search(text):
                suspects.add((setup.parent.name, text[:60]))

    assert suspects == _RELAXED_SUSPECT_RESIDUE
    # What is left is genuine file-seeding: the script builds the CSVs/workspace
    # and the instruction tells the AGENT to open the app. 251/8704 = 2.9% on the
    # all-script-kind basis (222 of them non-excluded). It was 283 while the head
    # anchor was line-anchored and literal-only.
    assert (scripts, sum(relaxed.values())) == (8704, 251)
    assert dict(relaxed) == {
        "multi_apps": 107, "pdf": 133, "vscode": 10, "libreoffice_calc": 1,
    }


def test_desktop_task_json_helpers_cover_downloads_and_open(tmp_path):
    (tmp_path / "seed.docx").write_bytes(b"seed")
    task_json = tmp_path / "task.json"
    task_json.write_text(json.dumps({
        "config": [
            {
                "type": "download",
                "parameters": {
                    "files": [
                        {"url": "./seed.docx", "path": "/home/user/a.docx"},
                        {"url": "https://example.test/x", "path": "/home/user/x"},
                    ]
                },
            },
            {"type": "open", "parameters": {"path": "/home/user/a.docx"}},
        ]
    }))
    assert desktop._local_downloads(task_json) == [
        (tmp_path / "seed.docx", "/home/user/a.docx")
    ]
    assert desktop._open_paths(task_json) == ["/home/user/a.docx"]
    assert desktop._opener_for("/home/user/a.docx") == "soffice"


@pytest.mark.asyncio
async def test_desktop_open_steps_dispatch_office_vs_generic(tmp_path):
    task_json = tmp_path / "task.json"
    task_json.write_text(json.dumps({
        "config": [
            {"type": "open", "parameters": {"path": "/home/user/report.docx"}},
            {"type": "open", "parameters": {"path": "/home/user/readme.txt"}},
            {"type": "download", "parameters": {"path": "/home/user/ignored.pdf"}},
        ]
    }))
    commands = []

    class Interface:
        async def run_command(self, command, **_kwargs):
            commands.append(command)

    class Computer:
        interface = Interface()

    assert await desktop._apply_open_steps(Computer(), task_json) is True
    assert "setsid soffice /home/user/report.docx" in commands[0]
    assert "setsid xdg-open /home/user/readme.txt" in commands[1]
    assert len(commands) == 2


@pytest.mark.asyncio
async def test_desktop_postconfig_executes_save_then_sleep(monkeypatch):
    commands = []
    sleeps = []

    class Interface:
        async def run_command(self, command, **_kwargs):
            commands.append(command)
            return SimpleNamespace(stdout="", stderr="", returncode=0)

    class Computer:
        interface = Interface()

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(desktop.asyncio, "sleep", fake_sleep)
    await desktop._run_postconfig(
        Computer(),
        [
            {
                "type": "execute",
                "parameters": {
                    "command": ["python", "-c", "import pyautogui; pyautogui.hotkey('ctrl','s')"]
                },
            },
            {"type": "sleep", "parameters": {"seconds": 0.5}},
        ],
    )
    assert "pyautogui.hotkey" in commands[0]
    assert sleeps == [0.5]


@pytest.mark.asyncio
async def test_desktop_invalid_postconfig_is_nonretryable_task_failure():
    with pytest.raises(CuaGymTaskError) as raised:
        await desktop._run_postconfig(
            SimpleNamespace(),
            [{"type": "unknown", "parameters": {}}],
        )

    assert raised.value.phase == "setup"
    assert raised.value.kind == "invalid_bundle"


def test_command_failures_raise_instead_of_scoring_zero():
    failed = SimpleNamespace(stdout="", stderr="boom", returncode=2)
    with pytest.raises(CuaGymTaskError, match="setup failed"):
        browser._raise_for_command(
            failed,
            "setup",
            error_type=CuaGymTaskError,
        )
    with pytest.raises(CuaGymTaskError, match="reward failed"):
        desktop._raise_for_command(failed, "reward")


@pytest.mark.asyncio
@pytest.mark.parametrize("module", [desktop, browser])
async def test_evaluate_final_returns_reward_with_rubric_lines(module, tmp_path, monkeypatch):
    """The declared `(reward, eval_info)` tuple is what `SandboxBaseEnv.step`
    unpacks into `info["eval"]` — and cuagym returns it unconditionally (no `debug`
    arg reaches a 2-arg evaluate_final_fn)."""
    reward_py = tmp_path / "reward.py"
    reward_py.write_text("print('REWARD: 0.4')\n")
    stdout = "PASS: header (0.2)\nFAIL: totals (0.0)\nREWARD: 0.4\n"

    async def run(*_args, **_kwargs):
        return SimpleNamespace(stdout=stdout, stderr="", returncode=0)

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(module, "write_text", noop)
    monkeypatch.setattr(module, "_run_task_command", run)
    if module is browser:
        monkeypatch.setattr(module, "_ensure_mocks", noop)
    task = SimpleNamespace(metadata={"reward": str(reward_py), "postconfig": []})

    result = await module.evaluate_final_fn(task, SimpleNamespace())
    assert result == (0.4, {"reward_stdout": stdout.splitlines()})


@pytest.mark.asyncio
async def test_container_write_bytes_prefers_exec_stdio_rpc():
    calls = []

    class Interface:
        async def write_bytes(self, path, data):
            calls.append((path, data))

        async def run_command(self, command):
            raise AssertionError(command)

    class Computer:
        interface = Interface()

    await container.write_bytes(Computer(), b"payload", "/tmp/payload.bin")
    # cuagym writes are user-facing seed/script files; the server runs as the
    # desktop user, so the RPC lands them owned by that user (the /home/user
    # invariant) — no run_as needed.
    assert calls == [("/tmp/payload.bin", b"payload")]


@pytest.mark.asyncio
async def test_display_readiness_uses_explicit_rollout_timeouts():
    calls = []

    class Interface:
        async def run_command(self, command, **kwargs):
            calls.append((command, kwargs))
            return SimpleNamespace(stdout="", stderr="", returncode=0)

    await display.bridge_display(SimpleNamespace(interface=Interface()))

    assert calls[0][1]["timeout"] == 180
    assert calls[1][1]["timeout"] == 30
    # Third leg: setup's baseline window snapshot needs the WM to have published
    # its client list, which gnome-ready does not promise.
    assert calls[2][1]["timeout"] == 40
    assert "wmctrl -l" in calls[2][0]


@pytest.mark.asyncio
async def test_bridge_display_fails_loudly_when_no_client_list_appears():
    class Interface:
        async def run_command(self, command, **kwargs):
            rc = 1 if "wmctrl -l" in command else 0
            return SimpleNamespace(stdout="", stderr="Cannot open display.", returncode=rc)

    with pytest.raises(RuntimeError, match="published no client list"):
        await display.bridge_display(SimpleNamespace(interface=Interface()))


@pytest.mark.asyncio
async def test_window_ids_retries_a_listing_that_lost_a_window_mid_enumeration():
    # wmctrl reads _NET_CLIENT_LIST and then queries each window it names, so a
    # window closing between those steps fails the whole call. That is a moving
    # desktop, not a broken connection, so the snapshot is simply retaken.
    results = [
        SimpleNamespace(stdout="", stderr="BadWindow (invalid Window parameter)", returncode=1),
        SimpleNamespace(stdout="0x01c00003 -1 host one\n", stderr="", returncode=0),
    ]
    calls = []

    class Interface:
        async def run_command(self, command):
            calls.append(command)
            return results.pop(0)

    ids = await runtime.window_ids(SimpleNamespace(interface=Interface()))

    assert ids == {"0x01c00003"}
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_window_ids_raises_with_the_last_error_once_the_budget_is_spent():
    # Distinct stderr per attempt: the contract is that the LAST one is reported,
    # and identical text cannot tell that apart from reporting the first. The
    # attempt count is a LITERAL -- comparing it against `_LIST_ATTEMPTS` would
    # compare the loop against the constant it reads and pin nothing.
    stderrs = ["first attempt", "second attempt", "Cannot open display."]
    calls = []

    class Interface:
        async def run_command(self, command):
            calls.append(command)
            return SimpleNamespace(
                stdout="", stderr=stderrs[len(calls) - 1], returncode=1,
            )

    with pytest.raises(CuaGymTaskError, match="Cannot open display.") as excinfo:
        await runtime.window_ids(SimpleNamespace(interface=Interface()))

    assert "first attempt" not in str(excinfo.value)
    assert len(calls) == 3


def test_bundle_refresh_keeps_old_task_until_archive_is_complete(tmp_path):
    dest = tmp_path / "tasks"
    old = dest / "t1"
    old.mkdir(parents=True)
    (old / "value").write_text("old")
    archive = tmp_path / "tasks.tar.zst"
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        payload = b"new"
        info = tarfile.TarInfo("other/value")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    archive.write_bytes(zstandard.ZstdCompressor().compress(raw.getvalue()))

    with pytest.raises(RuntimeError, match="missing 1 selected ids"):
        dataset.extract_bundles(archive, {"t1"}, dest, refresh=True)

    assert (old / "value").read_text() == "old"


def test_bundle_refresh_rolls_back_if_publish_fails(tmp_path, monkeypatch):
    dest = tmp_path / "tasks"
    old = dest / "t1"
    old.mkdir(parents=True)
    (old / "value").write_text("old")
    archive = tmp_path / "tasks.tar.zst"
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        payload = b"new"
        info = tarfile.TarInfo("t1/value")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    archive.write_bytes(zstandard.ZstdCompressor().compress(raw.getvalue()))
    real_replace = dataset.os.replace
    failed = False

    def fail_publish(src, target):
        nonlocal failed
        if (
            Path(target) == dest
            and ".extracting-" in Path(src).name
            and not failed
        ):
            failed = True
            raise OSError("publish failed")
        return real_replace(src, target)

    monkeypatch.setattr(dataset.os, "replace", fail_publish)
    with pytest.raises(OSError, match="publish failed"):
        dataset.extract_bundles(archive, {"t1"}, dest, refresh=True)

    assert (dest / "t1" / "value").read_text() == "old"


def test_bundle_root_is_one_fixed_cache_directory(tmp_path):
    assert dataset.bundle_root(tmp_path) == tmp_path / "bundles"


def test_extract_bundles_rejects_parent_directory_escape(tmp_path):
    archive = tmp_path / "tasks.tar.zst"
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        payload = b"escaped"
        info = tarfile.TarInfo("t1/../../escape")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    archive.write_bytes(zstandard.ZstdCompressor().compress(raw.getvalue()))

    with pytest.raises(tarfile.OutsideDestinationError):
        dataset.extract_bundles(archive, {"t1"}, tmp_path / "tasks")

    assert not (tmp_path / "escape").exists()


def test_extract_bundles_rejects_symlink_escape(tmp_path):
    archive = tmp_path / "tasks.tar.zst"
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        info = tarfile.TarInfo("t1/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../escape"
        tar.addfile(info)
    archive.write_bytes(zstandard.ZstdCompressor().compress(raw.getvalue()))

    with pytest.raises(tarfile.LinkOutsideDestinationError):
        dataset.extract_bundles(archive, {"t1"}, tmp_path / "tasks")

    assert not (tmp_path / "escape").exists()


def test_extract_tar_zst_rejects_absolute_path(tmp_path):
    escaped = tmp_path / "escape"
    dest = tmp_path / "assets"
    archive = tmp_path / "assets.tar.zst"
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        payload = b"escaped"
        info = tarfile.TarInfo(str(escaped))
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    archive.write_bytes(zstandard.ZstdCompressor().compress(raw.getvalue()))

    dataset.extract_tar_zst(archive, dest)

    assert not escaped.exists()
    sanitized = dest / escaped.relative_to(escaped.anchor)
    assert sanitized.read_bytes() == b"escaped"


def test_atomic_catalog_writer_and_validator(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [{"task_id": "t1", "instruction": "do it", "metadata": {}}]
    dataset.write_jsonl_atomic(path, rows)
    assert dataset.validate_catalog(path) == 1
    assert not list(tmp_path.glob(".train.jsonl.*"))


@pytest.mark.parametrize("module", [import_desktop_tasks, import_web_tasks])
def test_cuagym_importers_require_fresh_revision_and_digest(
    tmp_path,
    monkeypatch,
    module,
):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    revision = tmp_path / ".asset_revision"
    digest = tmp_path / ".asset_digest"
    monkeypatch.setattr(module, "TASKS_DIR", tasks_dir)
    monkeypatch.setattr(module, "REVISION_FILE", revision)
    monkeypatch.setattr(module, "DIGEST_FILE", digest)
    monkeypatch.setattr(module.dataset, "asset_identity", lambda: "asset-v1")
    monkeypatch.setattr(module.dataset, "task_cache_digest", lambda _root: "digest-v1")

    assert not module._cache_is_fresh()
    revision.write_text("asset-v1\n")
    assert not module._cache_is_fresh()
    digest.write_text("old-digest\n")
    assert not module._cache_is_fresh()
    digest.write_text("digest-v1\n")
    assert module._cache_is_fresh()

    def broken_digest(_root):
        raise OSError("unreadable")

    monkeypatch.setattr(module.dataset, "task_cache_digest", broken_digest)
    assert not module._cache_is_fresh()


@pytest.mark.parametrize(
    "contents",
    ["", "{\n", '{"task_id":"t1","instruction":"x","metadata":{}}\n'
     '{"task_id":"t1","instruction":"y","metadata":{}}\n'],
)
def test_catalog_validator_rejects_empty_malformed_or_duplicate(tmp_path, contents):
    path = tmp_path / "train.jsonl"
    path.write_text(contents)
    with pytest.raises(RuntimeError):
        dataset.validate_catalog(path)


def test_every_exclude_reason_is_used_by_the_pinned_catalogs():
    """The EXCLUDE_REASONS docstring quotes a count; keep it honest. A category
    defined but never counted is how it drifted to 362 (it omitted the 41
    `broken_reward:no_sentinel` rows defined two lines below it)."""
    catalogs = [
        M._DIR / ".cache" / "desktop" / "lite.cuagym_desktop_tasks" / "train.jsonl",
        M._DIR / ".cache" / "web" / "lite.cuagym_tasks" / "train.jsonl",
    ]
    _require_fresh_task_cache()
    rows = [
        json.loads(line)
        for path in catalogs
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    tagged = [
        reason
        for row in rows
        if (reason := (row["metadata"].get("others") or {}).get("exclude_reason"))
    ]
    assert set(tagged) <= set(dataset.EXCLUDE_REASONS)
    assert (len(rows), len(tagged)) == (10910, 513)
    assert Counter(tagged) == Counter({
        "broken_reward:empty": 152,
        "broken_mock:blank_render": 81,
        "broken_reward:no_sentinel": 42,
        "broken_reward:syntax_error": 26,
        # Structural, from dataset.reward_defect: the reward opens a `*_golden.*`
        # only its authoring run had, while setup builds the same artifact without
        # the suffix -- so os.path.exists is False on every trajectory and the
        # script returns a SILENT 0.0 no matter what the agent did.
        "broken_reward:missing_golden": 19,
        "broken_setup:unsatisfiable_gate": 1,
        "broken_setup:external_dependency": 8,
        "broken_setup:wrong_backend": 1,
        "broken_setup:missing_seed_file": 2,
        "broken_setup:syntax_error": 1,
        "broken_setup:no_task_window": 1,
        "broken_reward:instruction_mismatch": 178,
        "broken_task:empty_instruction": 1,
    })


def test_no_catalog_row_states_an_empty_instruction_without_a_reason():
    """A row whose upstream task.json has no instruction cannot be attempted: the
    agent raises ``No instruction returned on env.reset()`` after a full container
    boot. The importer annotates it mechanically, so `guard_excluded` refuses it at
    setup and the standard `exclude_reason` filter never selects it."""
    assert dataset.instruction_defect("") == "broken_task:empty_instruction"
    assert dataset.instruction_defect(" \n\t ") == "broken_task:empty_instruction"
    assert dataset.instruction_defect("Open the report and fix the total.") is None

    _require_fresh_task_cache()
    for path in (
        M._DIR / ".cache" / "desktop" / "lite.cuagym_desktop_tasks" / "train.jsonl",
        M._DIR / ".cache" / "web" / "lite.cuagym_tasks" / "train.jsonl",
    ):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["instruction"].strip():
                continue
            assert row["metadata"]["others"].get("exclude_reason") == (
                "broken_task:empty_instruction"
            ), row["task_id"]


def test_curated_desktop_reward_instruction_mismatch_is_exact(tmp_path):
    reward = tmp_path / "reward.py"
    reward.write_text("print('REWARD: 1.0')\n")

    targets = {
        "0cf1927d-92d8-5848-86a7-ad911ed6df32",
        "71284cbb-428c-5297-ab5c-2381ff47fb7b",
        "93256f83-a331-5216-a7ed-7af0a1a04c0a",
        "fa44cbc6-979b-5ddb-bc46-6ef0ad6c414c",
        "3b3af31c-46bb-5707-9742-0f6d2cb96991",
        "3cda6a52-988a-5710-8fb1-baf9f8cba495",
        "85f4ed1b-f1ff-5da4-a8f5-662daaf43a6d",
        "9d28ae95-4a52-5689-9a5b-be572dc3b8fb",
        "ba26902d-be86-5c37-8355-93bf9f940d3c",
        "fe5b3914-5301-5717-9488-ee718f1fc1bb",
        "69ef92b8-eb83-542a-a1d4-daa6076f45f6",
        "f72d146d-50a9-59bb-8b74-ede7265d1275",
        "f77f100f-24c6-5995-a327-7f9c1fd52bc6",
        "8ad56749-de8b-52a2-a15e-4dc9add0ddcb",
        "ab46f49b-8b90-599b-a4f3-fd1a61d2f297",
        "f3adc962-25ca-5ce5-8e27-0a297f4292d2",
        "b953f35e-8bba-57f9-8d0c-162ece93e6c4",
        "6908abb2-b325-5977-944b-07bce78e45a9",
        "7012f478-fc57-5d56-9baa-055703fc1209",
        "e9af319c-4dd9-5b9e-80a1-d1f2543970fd",
        "b14f9d17-34e0-57c4-8dda-cb5ca4d4c945",
        "7772a692-491a-5119-a4d9-da106befa0ec",
        "9b7b2fa4-76b7-5d72-af13-f47d28c73e74",
        "8c046751-8722-5be6-9f53-818f0a931a3d",
        "ff5ae91b-b80f-5eff-87b9-311dd2bdaa0b",
        "68b3f608-6635-5562-a454-980f71d71b75",
        "fc1598e2-bc9d-5e9e-972b-dcd14dffcde5",
        "fa097d51-e80d-5c62-98bd-41e09fe01055",
        "30cd6e7f-7455-58ee-9ae4-9fef84ea7448",
        "4b3b41f2-1c2a-58b1-9ebf-034c5316c00f",
        "0ca36934-08ab-5087-9b15-7a937ea473c7",
        "fc9d9886-1d4a-5a6f-a3d3-f9a63b4f773b",
        "c30e9cf1-c374-53f7-aacf-7f6d7a56975b",
        "b599432c-1233-5142-9bdb-290b9296cc50",
        "d768dee5-079b-5805-b4ea-47888bc03318",
        "53d36690-34c0-5091-8b45-acfa0474cb08",
        "b09925d7-3fc5-518c-b715-342b16acb812",
        "d4eae0d6-b21d-548b-b2ad-140eddd06bfb",
        "569b9ce9-642c-5bbc-8639-9f36deff0d6f",
        "cc12909d-ee5a-502f-839f-557dca9ce9d4",
        "986205c2-879a-528f-9d03-f4a96722ddd2",
        "682daab8-a3e6-5241-9c25-2a9a7f3edc48",
        "298a795a-a725-5b1f-9291-509a67c29b56",
        "10a0eae5-ba1b-5487-9663-6f5c5d9e62f3",
        "a69bb3c1-0054-5743-a4ad-b76c562ab68f",
        "ce39f74c-619a-58c0-a32c-4255372b727c",
        "27f41643-9f77-524a-8ee0-eac537dc7190",
        "0647deaa-6a13-59c0-bd0a-9a928864b993",
        "9ab5869f-6312-5a99-89cf-8ce84199cd4e",
        "88cc3568-6078-5ba6-b18a-35ba74fcd864",
        "c877b513-be19-5449-bccd-849215b5aa90",
    }
    for target in targets:
        assert import_desktop_tasks._exclude_reason(target, reward) == (
            "broken_reward:instruction_mismatch"
        )
    assert import_desktop_tasks._exclude_reason(
        "0cf1927d-92d8-5848-86a7-ad911ed6df33",
        reward,
    ) is None

    empty = tmp_path / "empty_reward.py"
    empty.write_text("")
    assert import_desktop_tasks._exclude_reason("not-curated", empty) == (
        "broken_reward:empty"
    )


def test_curated_desktop_external_setup_dependency_is_exact(tmp_path):
    reward = tmp_path / "reward.py"
    reward.write_text("print('REWARD: 1.0')\n")

    targets = {
        "c3ef8fcc-f312-531b-a857-acb953d0afb3",
        "1a92f790-8527-509c-80cf-c37373d7effe",
        "77e22e90-e016-5bfa-a06c-484efcc4ef7a",
        "b4ba374e-f593-567a-9abb-22461e8b0689",
        "a093712a-9e99-5dbe-a25e-c40025fda4df",
        "5d90b9b5-6539-543b-b19b-1fb20e4eebe3",
        "5aa9bb0b-2a92-54aa-b4fa-5ffa83ccae69",
        "6e0a27f8-f39e-5ded-ba3d-848c6db975cd",
    }
    for target in targets:
        assert import_desktop_tasks._exclude_reason(
            target,
            reward,
        ) == "broken_setup:external_dependency"
    assert import_desktop_tasks._exclude_reason(
        "c3ef8fcc-f312-531b-a857-acb953d0afb4",
        reward,
    ) is None


def test_curated_desktop_missing_seed_file_setup_is_exact(tmp_path):
    reward = tmp_path / "reward.py"
    reward.write_text("print('REWARD: 1.0')\n")

    for target in (
        "77a46ae0-2a04-5537-8408-bd7e87c67274",
        "a119da79-d2cb-5019-b5cd-4ff00c32d1b5",
    ):
        assert import_desktop_tasks._exclude_reason(
            target,
            reward,
        ) == "broken_setup:missing_seed_file"
    assert import_desktop_tasks._exclude_reason(
        "77a46ae0-2a04-5537-8408-bd7e87c67275",
        reward,
    ) is None


def test_curated_desktop_setup_syntax_error_is_exact(tmp_path):
    reward = tmp_path / "reward.py"
    reward.write_text("print('REWARD: 1.0')\n")

    assert import_desktop_tasks._exclude_reason(
        "fe1e8f73-5c46-56f9-8fae-debd3cc49918",
        reward,
    ) == "broken_setup:syntax_error"
    assert import_desktop_tasks._exclude_reason(
        "fe1e8f73-5c46-56f9-8fae-debd3cc49919",
        reward,
    ) is None


def test_curated_desktop_no_task_window_setup_is_exact(tmp_path):
    reward = tmp_path / "reward.py"
    reward.write_text("print('REWARD: 1.0')\n")

    assert import_desktop_tasks._exclude_reason(
        "918c2c55-2474-57db-ae86-a6a06f2e8b76",
        reward,
    ) == "broken_setup:no_task_window"
    assert import_desktop_tasks._exclude_reason(
        "918c2c55-2474-57db-ae86-a6a06f2e8b77",
        reward,
    ) is None


def test_curated_desktop_reward_no_sentinel_is_exact(tmp_path):
    reward = tmp_path / "reward.py"
    reward.write_text("print('REWARD: 1.0')\n")

    assert import_desktop_tasks._exclude_reason(
        "b3668f33-b8a2-5a52-b737-0ca4cbe6296f",
        reward,
    ) == "broken_reward:no_sentinel"
    assert import_desktop_tasks._exclude_reason(
        "b3668f33-b8a2-5a52-b737-0ca4cbe6296e",
        reward,
    ) is None


def test_curated_google_drive_mock_blank_render_is_exact(tmp_path):
    reward = tmp_path / "reward.py"
    reward.write_text("print('REWARD: 1.0')\n")

    task_id = next(
        task_id
        for task_id, finding in dataset.validation_excludes().items()
        if finding.get("app") == "google_drive"
    )
    assert import_web_tasks._exclude_reason(
        ["google_drive"], reward, task_id=task_id
    ) == (
        "broken_mock:blank_render"
    )


def test_curated_web_wrong_backend_is_exact(tmp_path):
    reward = tmp_path / "reward.py"
    reward.write_text("print('REWARD: 1.0')\n")

    assert import_web_tasks._exclude_reason(
        ["google_sheets"],
        reward,
        task_id="9bbdfe1c-8098-5771-9cf0-a526a705c266",
    ) == "broken_setup:wrong_backend"
    assert import_web_tasks._exclude_reason(
        ["google_sheets"],
        reward,
        task_id="9bbdfe1c-8098-5771-9cf0-a526a705c267",
    ) is None
    assert import_web_tasks._exclude_reason(["gmail", "google_drive"], reward) is None
    assert import_web_tasks._exclude_reason(["gmail"], reward) is None


def test_cuagym_vlc_hidden_xspf_requirement_is_excluded():
    catalog = M._DIR / ".cache" / "desktop" / "lite.cuagym_desktop_tasks" / "train.jsonl"
    if not catalog.exists():
        pytest.skip(
            "task cache not imported "
            "(run lite/gym/envs/lite/cuagym/scripts/install.sh provision)"
        )

    target = "0cf1927d-92d8-5848-86a7-ad911ed6df32"
    rows = [json.loads(line) for line in catalog.read_text().splitlines() if line.strip()]
    row = next(row for row in rows if row["task_id"] == target)

    assert "XSPF" not in row["instruction"]
    assert (row["metadata"].get("others") or {}).get("exclude_reason") == (
        "broken_reward:instruction_mismatch"
    )


def test_cuagym_calc_reward_instruction_mismatches_are_excluded():
    catalog = M._DIR / ".cache" / "desktop" / "lite.cuagym_desktop_tasks" / "train.jsonl"
    if not catalog.exists():
        pytest.skip(
            "task cache not imported "
            "(run lite/gym/envs/lite/cuagym/scripts/install.sh provision)"
        )

    rows = {
        row["task_id"]: row
        for row in (
            json.loads(line) for line in catalog.read_text().splitlines() if line.strip()
        )
    }

    ranking = rows["682daab8-a3e6-5241-9c25-2a9a7f3edc48"]
    assert "ranking.txt" in ranking["instruction"]
    assert "header" not in ranking["instruction"].lower()
    assert ranking["metadata"]["others"]["exclude_reason"] == (
        "broken_reward:instruction_mismatch"
    )

    dashboard = rows["298a795a-a725-5b1f-9291-509a67c29b56"]
    assert "project dashboard" in dashboard["instruction"]
    assert "data validations" not in dashboard["instruction"].lower()
    assert "Hours: Estimated vs Actual" not in dashboard["instruction"]
    assert dashboard["metadata"]["others"]["exclude_reason"] == (
        "broken_reward:instruction_mismatch"
    )


def test_b22_cuagym_reward_instruction_mismatches_are_excluded():
    catalog = M._DIR / ".cache" / "desktop" / "lite.cuagym_desktop_tasks" / "train.jsonl"
    if not catalog.exists():
        pytest.skip(
            "task cache not imported "
            "(run lite/gym/envs/lite/cuagym/scripts/install.sh provision)"
        )

    rows = {
        row["task_id"]: row
        for row in (
            json.loads(line) for line in catalog.read_text().splitlines() if line.strip()
        )
    }

    devcontainer = rows["ce39f74c-619a-58c0-a32c-4255372b727c"]
    assert "postCreateCommand" in devcontainer["instruction"]
    assert '"image"' not in devcontainer["instruction"]
    assert devcontainer["metadata"]["others"]["exclude_reason"] == (
        "broken_reward:instruction_mismatch"
    )

    snapshot = rows["27f41643-9f77-524a-8ee0-eac537dc7190"]
    assert "insert that snapshot image" in snapshot["instruction"]
    assert "inline" not in snapshot["instruction"].lower()
    assert snapshot["metadata"]["others"]["exclude_reason"] == (
        "broken_reward:instruction_mismatch"
    )

    pdf = rows["0647deaa-6a13-59c0-bd0a-9a928864b993"]
    assert "consistent page sizes" in pdf["instruction"]
    assert "links found" not in pdf["instruction"].lower()
    assert pdf["metadata"]["others"]["exclude_reason"] == (
        "broken_reward:instruction_mismatch"
    )


def test_b23_cuagym_reward_instruction_mismatches_are_excluded():
    catalog = M._DIR / ".cache" / "desktop" / "lite.cuagym_desktop_tasks" / "train.jsonl"
    if not catalog.exists():
        pytest.skip(
            "task cache not imported "
            "(run lite/gym/envs/lite/cuagym/scripts/install.sh provision)"
        )

    rows = {
        row["task_id"]: row
        for row in (
            json.loads(line) for line in catalog.read_text().splitlines() if line.strip()
        )
    }

    account_mask = rows["9ab5869f-6312-5a99-89cf-8ce84199cd4e"]
    assert "Replace the middle 4 characters" in account_mask["instruction"]
    assert "REPLACE" not in account_mask["instruction"]
    assert account_mask["metadata"]["others"]["exclude_reason"] == (
        "broken_reward:instruction_mismatch"
    )

    portfolio = rows["88cc3568-6078-5ba6-b18a-35ba74fcd864"]
    assert "pikepdf" in portfolio["instruction"]
    assert "size" not in portfolio["instruction"].lower()
    assert portfolio["metadata"]["others"]["exclude_reason"] == (
        "broken_reward:instruction_mismatch"
    )

    two_column = rows["c877b513-be19-5449-bccd-849215b5aa90"]
    assert "Summary" in two_column["instruction"]
    assert "Q4" not in two_column["instruction"]
    assert "NPS" not in two_column["instruction"]
    assert two_column["metadata"]["others"]["exclude_reason"] == (
        "broken_reward:instruction_mismatch"
    )


@pytest.mark.parametrize(
    ("target", "instruction_needle", "app_type", "app_family", "exclude_reason"),
    [
        (
            "0f32736b-ccdf-50c3-94cf-51c3a278a198",
            "VSCode editor settings",
            "vscode",
            "desktop",
            None,
        ),
        (
            "ab46f49b-8b90-599b-a4f3-fd1a61d2f297",
            "map_sprite.png",
            "multi_apps",
            "other",
            "broken_reward:instruction_mismatch",
        ),
        (
            "78030638-08ea-5d6e-991c-87cf4a25a6d9",
            "signed_contract.pdf",
            "pdf",
            "desktop",
            None,
        ),
        (
            "577cd0c1-b76f-581e-a628-ad46c8e321bc",
            "multi-language PDF invoice generator",
            "pdf",
            "desktop",
            None,
        ),
        (
            "ad645b23-87a6-58b2-9914-755a2af12ab2",
            "Flappy Bird clone",
            "vscode",
            "desktop",
            None,
        ),
        (
            "c30e9cf1-c374-53f7-aacf-7f6d7a56975b",
            "maze_game",
            "multi_apps",
            "other",
            "broken_reward:instruction_mismatch",
        ),
        (
            "377a0ccb-6013-5f8e-9631-ca06461b21ed",
            "product_cutout_gimp.png",
            "multi_apps",
            "other",
            None,
        ),
        (
            "cf9cf22d-f111-5898-9f16-1f2451865827",
            "_config.yml",
            "vscode",
            "desktop",
            None,
        ),
        (
            "10a0eae5-ba1b-5487-9663-6f5c5d9e62f3",
            "Configure Java import organization",
            "vscode",
            "desktop",
            "broken_reward:instruction_mismatch",
        ),
        (
            "7b5b471d-b4a0-51d3-99cd-1d0a54a9d322",
            "renovation_showcase.odp",
            "libreoffice_impress",
            "desktop_office",
            None,
        ),
    ],
)
def test_cuagym_desktop_metadata_overrides_are_applied(
    target,
    instruction_needle,
    app_type,
    app_family,
    exclude_reason,
):
    catalog = M._DIR / ".cache" / "desktop" / "lite.cuagym_desktop_tasks" / "train.jsonl"
    if not catalog.exists():
        pytest.skip(
            "task cache not imported "
            "(run lite/gym/envs/lite/cuagym/scripts/install.sh provision)"
        )

    rows = [json.loads(line) for line in catalog.read_text().splitlines() if line.strip()]
    row = next(row for row in rows if row["task_id"] == target)
    others = row["metadata"]["others"]

    assert instruction_needle in row["instruction"]
    assert others["app_type"] == app_type
    assert others["app_family"] == app_family
    if exclude_reason is None:
        assert "exclude_reason" not in others
    else:
        assert others["exclude_reason"] == exclude_reason


# --- unified registration --------------------------------------------------


def test_single_env_id_and_one_image():
    assert M._ENV_ID == "lite.cuagym"
    assert [M._COMPUTER_CONFIG["image"]] == ["cua-lite/lite.cuagym:latest"]


def test_cuagym_dockerfile_private_osworld_base_hook():
    repo = Path(__file__).resolve().parents[5]
    dockerfile = repo / "lite/gym/envs/lite/cuagym/docker/Dockerfile"
    install = repo / "lite/gym/envs/lite/cuagym/scripts/install.sh"
    docker_text = dockerfile.read_text()
    docker_lines = docker_text.splitlines()
    install_text = install.read_text()

    assert "ARG BASE_IMAGE=cua-lite/lite.osworld:latest" in docker_text
    assert "FROM ${BASE_IMAGE} AS xcftools-builder" in docker_text
    assert docker_lines.count("FROM ${BASE_IMAGE}") == 1
    assert 'IMAGE="${LITE_CUAGYM_IMAGE:-$DEFAULT_IMAGE}"' in install_text
    assert 'BASE="${LITE_CUAGYM_BASE_IMAGE:-cua-lite/lite.osworld:latest}"' in install_text
    assert '--build-arg "BASE_IMAGE=$BASE"' in install_text
    assert 'if [ "$BASE" = "cua-lite/lite.osworld:latest" ]; then' in install_text
    assert (
        'if [ "$BASE" = "cua-lite/lite.osworld:latest" ] && '
        'image_is_fresh "$IMAGE" lite.cuagym; then'
    ) in install_text
    assert (
        'private base $BASE requested; rebuilding $IMAGE to avoid stale parent layers'
    ) in install_text
    assert 'docker image inspect "$BASE" >/dev/null' in install_text


def test_cuagym_vscode_wrapper_scrubs_backend_path_and_covers_absolute_code():
    repo = Path(__file__).resolve().parents[5]
    dockerfile = repo / "lite/gym/envs/lite/cuagym/docker/Dockerfile"
    text = dockerfile.read_text()

    assert (
        "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        in text
    )
    assert (
        'exec /usr/share/code/bin/code --no-sandbox --disable-workspace-trust "$@"'
        in text
    )
    assert "rm -f /usr/bin/code" in text
    assert "install -m 0755 /usr/local/bin/code /usr/bin/code" in text


def test_bind_refreshes_backend_callbacks():
    # ``platform`` mirrors the real SandboxTaskConfig field: bind() resolves
    # env_kwargs.valid_actions against the task's GUI vocabulary, so a fake
    # task without it is not a faithful stand-in (and these two fakes are
    # named for exactly that browser/desktop split).
    browser_task = SimpleNamespace(
        computer={},
        platform="browser",
        setup_fn=object(),
        evaluate_step_fn=object(),
        evaluate_final_fn=object(),
    )
    desktop_task = SimpleNamespace(
        computer={},
        platform="desktop",
        setup_fn=object(),
        evaluate_step_fn=object(),
        evaluate_final_fn=object(),
    )
    env = object.__new__(M._CuaGymEnv)
    env._env_id = "lite.cuagym"
    env._display_resolution = (1920, 1080)
    env._computer_config = {}

    env.bind(browser_task)
    env.bind(desktop_task)

    assert env._setup_fn is desktop_task.setup_fn
    assert env._evaluate_step_fn is desktop_task.evaluate_step_fn
    assert env._evaluate_final_fn is desktop_task.evaluate_final_fn


def test_cuagym_waits_for_child_image_settings():
    from lite.gym.envs.lite.cuagym.src.utils.runtime import READY_MARKER

    assert READY_MARKER == "/tmp/lite-cuagym-ready"
    assert M._CuaGymEnv.READY_MARKER == READY_MARKER
    dockerfile = (
        Path(__file__).parents[5]  # tests/gym/envs/lite/cuagym/<file> -> repo root
        / "lite/gym/envs/lite/cuagym/docker/Dockerfile"
    ).read_text()
    assert f"touch {READY_MARKER}" in dockerfile


@pytest.mark.asyncio
async def test_evaluate_terminal_step_uses_canonical_terminate(monkeypatch):
    from lite.core.tools import make_tool_call

    seen = {}

    async def step(env, actions):
        seen["actions"] = actions
        seen["task"] = env._task
        return SimpleNamespace(terminated=True, truncated=False, reward=0.25)

    monkeypatch.setattr(SandboxBaseEnv, "step", step)
    task = SimpleNamespace(task_id="task")
    reward = await runtime.evaluate_terminal_step(
        task,
        SimpleNamespace(),
        evaluate_final_fn=object(),
        timeout=1.0,
    )

    assert reward == 0.25
    assert seen["task"] is task
    assert seen["actions"] == [make_tool_call("terminate", {})]


@pytest.mark.asyncio
async def test_repeated_reset_uses_a_fresh_container(monkeypatch):
    first = SimpleNamespace(name="first")
    second = SimpleNamespace(name="second")
    computers = iter((first, second))
    closed = []

    async def reset_base(env):
        env._computer = next(computers)
        return LiteEnvObservation(image=b"image")

    async def release(env):
        closed.append(env._computer)
        env._computer = None
        env._container_name = None

    async def boot(env):
        if env._computer is None:
            env._computer = SimpleNamespace(name="booted")

    async def healthy_pre_setup(_computer):
        return None

    async def healthy_post_setup(_computer, *, screenshot):
        assert screenshot == b"image"

    monkeypatch.setattr(SandboxBaseEnv, "reset", reset_base)
    monkeypatch.setattr(M._CuaGymEnv, "_release_failed_attempt", release)
    monkeypatch.setattr(M._CuaGymEnv, "boot", boot)
    monkeypatch.setattr(M, "validate_pre_setup_runtime", healthy_pre_setup)
    monkeypatch.setattr(M, "validate_post_setup_runtime", healthy_post_setup)

    env = object.__new__(M._CuaGymEnv)
    env._owns_computer = True
    env._container_override = None
    env._computer = None
    env._container_name = None
    # object.__new__ skips __init__ — plant the two attrs it would set
    # (_recycle_count/_recycle_first_done are class defaults on EnvServerPoolable)
    env._max_resets_per_container = 0
    env._cuagym_episode_started = False

    await env.reset()
    await env.reset()

    assert closed == [first]
    assert env._computer is second
    assert env._recycle_first_done is True


@pytest.mark.asyncio
async def test_destroy_backend_does_not_later_delete_injected_container(monkeypatch):
    stopped = []
    removed = []

    class Computer:
        async def stop(self):
            stopped.append("session")

    async def docker_rm(name, *, timeout=None, label=None):
        removed.append((name, timeout, label))

    monkeypatch.setattr("lite.gym.sandbox.base.docker_rm_f_async", docker_rm)

    env = object.__new__(M._CuaGymEnv)
    env._owns_computer = False
    env._container_override = SimpleNamespace(name="external")
    env._computer = Computer()
    env._container_name = "external-container"
    env._cuagym_episode_started = True

    await env.destroy_backend()
    await SandboxBaseEnv.close(env)

    assert stopped == ["session"]
    assert removed == []
    assert env._container_name is None
    assert env._owns_computer is True


@pytest.mark.asyncio
async def test_reset_rejects_empty_initial_screenshot(monkeypatch):
    computer = SimpleNamespace()
    released = []

    async def reset_base(env):
        env._computer = computer
        return LiteEnvObservation(image=b"")

    async def boot(_env):
        return None

    async def healthy_pre_setup(_computer):
        return None

    async def release(env):
        released.append(env._computer)
        env._computer = None

    monkeypatch.setattr(SandboxBaseEnv, "reset", reset_base)
    monkeypatch.setattr(M._CuaGymEnv, "boot", boot)
    monkeypatch.setattr(M, "validate_pre_setup_runtime", healthy_pre_setup)
    monkeypatch.setattr(M._CuaGymEnv, "_release_failed_attempt", release)

    env = object.__new__(M._CuaGymEnv)
    env._owns_computer = True
    env._computer = computer
    env._container_name = None
    # object.__new__ skips __init__ — plant the two attrs it would set
    env._max_resets_per_container = 0
    env._cuagym_episode_started = False

    with pytest.raises(EnvDesktopCrashed, match="screenshot/desktop health"):
        await env.reset()

    assert released == [computer]
    assert env._computer is None


@pytest.mark.asyncio
async def test_reset_failure_releases_half_configured_container(monkeypatch):
    released = []

    async def reset_base(_env):
        raise EnvDepsMissingError("task setup failed", "install", "readme")

    async def discard(env):
        released.append(env._computer)
        env._computer = None

    async def healthy_pre_setup(_computer):
        return None

    monkeypatch.setattr(SandboxBaseEnv, "reset", reset_base)
    monkeypatch.setattr(M._CuaGymEnv, "destroy_backend", discard)
    monkeypatch.setattr(M, "validate_pre_setup_runtime", healthy_pre_setup)

    computer = SimpleNamespace()
    env = object.__new__(M._CuaGymEnv)
    env._owns_computer = True
    env._computer = computer
    env._container_name = None
    # object.__new__ skips __init__ — plant the two attrs it would set
    env._max_resets_per_container = 0
    env._cuagym_episode_started = False

    with pytest.raises(EnvDepsMissingError):
        await env.reset()

    assert released == [computer]
    assert env._computer is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"image": "custom"}, "does not accept image overrides"),
        (
            {"display_resolution": (1280, 720)},
            "does not accept display_resolution overrides",
        ),
    ],
)
def test_baked_overrides_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        M._CuaGymEnv(**kwargs)


def test_missing_or_empty_task_cache_has_actionable_lazy_registration_error(
    tmp_path, monkeypatch
):
    calls = []
    web_root = tmp_path / "web"
    desktop_root = tmp_path / "desktop"
    monkeypatch.setattr(M, "_catalog_registered", False)
    monkeypatch.setattr(M, "_WEB_ROOT", web_root)
    monkeypatch.setattr(M, "_DESKTOP_ROOT", desktop_root)
    monkeypatch.setattr(M, "_WEB_JSONL", web_root / "train.jsonl")
    monkeypatch.setattr(M, "_DESKTOP_JSONL", desktop_root / "train.jsonl")
    monkeypatch.setattr(
        M,
        "register_jsonl_tasks",
        lambda path, **kwargs: calls.append((path, kwargs["platform"])) or 0,
    )
    with pytest.raises(EnvDepsMissingError, match="task cache is missing"):
        M.CuaGymServices().register_tasks("lite.cuagym")

    for root in (web_root, desktop_root):
        root.mkdir(parents=True)
    for path in (M._WEB_JSONL, M._DESKTOP_JSONL):
        path.write_text("")
    for root in (web_root, desktop_root):
        (root / ".asset_revision").write_text(dataset.asset_identity() + "\n")
        (root / ".asset_digest").write_text(dataset.task_cache_digest(root) + "\n")
    with pytest.raises(EnvDepsMissingError, match="task cache is invalid"):
        M.CuaGymServices().register_tasks("lite.cuagym")
    assert calls == []


def test_stale_task_cache_asset_stamp_has_actionable_lazy_registration_error(
    tmp_path, monkeypatch
):
    web_root = tmp_path / "web"
    desktop_root = tmp_path / "desktop"
    for root in (web_root, desktop_root):
        root.mkdir(parents=True)
        (root / "train.jsonl").write_text("")
        (root / ".asset_revision").write_text("stale\n")
    monkeypatch.setattr(M, "_catalog_registered", False)
    monkeypatch.setattr(M, "_WEB_ROOT", web_root)
    monkeypatch.setattr(M, "_DESKTOP_ROOT", desktop_root)
    monkeypatch.setattr(M, "_WEB_JSONL", web_root / "train.jsonl")
    monkeypatch.setattr(M, "_DESKTOP_JSONL", desktop_root / "train.jsonl")

    with pytest.raises(EnvDepsMissingError, match="stale or invalid"):
        M.CuaGymServices().register_tasks("lite.cuagym")


def test_stale_task_cache_digest_has_actionable_lazy_registration_error(
    tmp_path, monkeypatch
):
    web_root = tmp_path / "web"
    desktop_root = tmp_path / "desktop"
    for root in (web_root, desktop_root):
        root.mkdir(parents=True)
        (root / "train.jsonl").write_text("")
        (root / ".asset_revision").write_text(dataset.asset_identity() + "\n")
        (root / ".asset_digest").write_text("stale\n")
    monkeypatch.setattr(M, "_catalog_registered", False)
    monkeypatch.setattr(M, "_WEB_ROOT", web_root)
    monkeypatch.setattr(M, "_DESKTOP_ROOT", desktop_root)
    monkeypatch.setattr(M, "_WEB_JSONL", web_root / "train.jsonl")
    monkeypatch.setattr(M, "_DESKTOP_JSONL", desktop_root / "train.jsonl")

    with pytest.raises(EnvDepsMissingError, match="stale or invalid"):
        M.CuaGymServices().register_tasks("lite.cuagym")


def _local_cuagym_cache_is_fresh() -> bool:
    if not (M._WEB_JSONL.is_file() and M._DESKTOP_JSONL.is_file()):
        return False
    expected_asset = dataset.asset_identity()
    for root in (M._WEB_ROOT, M._DESKTOP_ROOT):
        try:
            actual_asset = (root / ".asset_revision").read_text().strip()
            actual_digest = (root / ".asset_digest").read_text().strip()
            expected_digest = dataset.task_cache_digest(root)
        except (OSError, ValueError):
            return False
        if actual_asset != expected_asset or actual_digest != expected_digest:
            return False
    return True


@pytest.mark.skipif(
    not _local_cuagym_cache_is_fresh(),
    reason=(
        "task caches not imported or stale "
        "(run lite/gym/envs/lite/cuagym/scripts/install.sh provision)"
    ),
)
def test_both_backends_register_under_one_env_id():
    import lite.gym as gym

    assert "lite.cuagym" in gym.registry.env_ids()
    tasks = gym.registry.task_ids("lite.cuagym", split="train")
    web_count = dataset.validate_catalog(M._WEB_JSONL)
    desktop_count = dataset.validate_catalog(M._DESKTOP_JSONL)
    assert (web_count, desktop_count) == (1505, 9405)
    assert len(tasks) == web_count + desktop_count == 10910
    # Upstream web rows register first as browser tasks, desktop after -> both Lite platforms
    # present under the single env_id (metadata lookup needs no container/image).
    plats = {
        str(gym.registry.task_metadata("lite.cuagym", t).platform) for t in (tasks[0], tasks[-1])
    }
    assert plats == {"browser", "desktop"}

    from lite.gym.sandbox import lookup_task

    browser_task = lookup_task("lite.cuagym", tasks[0])
    desktop_task = lookup_task("lite.cuagym", tasks[-1])
    assert browser_task.max_steps == desktop_task.max_steps == M._MAX_STEPS == 30
    # Registered setup is the backend's own body behind ``guard_excluded`` (the
    # excluded-row refusal); the evaluators are unwrapped.
    assert browser_task.setup_fn.__wrapped__ is browser.setup_fn
    assert browser_task.evaluate_final_fn is browser.evaluate_final_fn
    assert desktop_task.setup_fn.__wrapped__ is desktop.setup_fn
    assert desktop_task.evaluate_final_fn is desktop.evaluate_final_fn


# --- excluded-row refusal (parity with lite.scalecua) ----------------------


async def test_setup_refuses_an_excluded_row_before_the_backend_body_runs():
    """(a) The guard sits in the setup path, where the task metadata IS visible:
    ``task.metadata['others']['exclude_reason']``. Mirrors lite.scalecua's
    ``setup_fn`` guard (phase="setup", kind="excluded_task"), so the two sibling
    envs refuse an excluded row identically instead of one dying at reset and the
    other burning a whole episode."""
    ran = []

    async def backend_setup(task, computer):
        ran.append(task)

    task = SimpleNamespace(
        task_id="cuagym_excluded",
        metadata={"others": {"exclude_reason": "broken_reward:empty"}},
    )
    with pytest.raises(CuaGymTaskError, match="broken_reward:empty") as raised:
        await M.guard_excluded(backend_setup)(task, SimpleNamespace())

    assert (raised.value.phase, raised.value.kind) == ("setup", "excluded_task")
    assert ran == []  # refused BEFORE any container work
    # Both registered backends carry it — one registration site, two setup_fns.
    for guarded, body in (
        (M.guard_excluded(browser.setup_fn), browser.setup_fn),
        (M.guard_excluded(desktop.setup_fn), desktop.setup_fn),
    ):
        assert guarded.__wrapped__ is body
        with pytest.raises(CuaGymTaskError, match="is excluded"):
            await guarded(task, SimpleNamespace())


async def test_excluded_refusal_is_an_env_error_not_a_zero_reward_sample(monkeypatch):
    """(b) The refusal must reach the caller as a SETUP/ENV failure, never as a
    reward-0 trajectory — that distinction is the whole point of failing fast.

    ``SandboxBaseEnv.reset`` wraps setup in ``except (RuntimeError, TimeoutError)
    -> CapacityExhausted`` (a RETRYABLE 503). ``CuaGymTaskError`` is a
    ``LiteGymError``, so it escapes that clause untouched and stays terminal.
    """
    async def booted(_self):
        return None

    monkeypatch.setattr(SandboxBaseEnv, "boot", booted)

    env = object.__new__(SandboxBaseEnv)
    env._task = SimpleNamespace(
        task_id="cuagym_excluded",
        instruction="do it",
        max_steps=30,
        metadata={"others": {"exclude_reason": "broken_mock:blank_render"}},
    )
    env._setup_fn = M.guard_excluded(browser.setup_fn)
    env._computer = SimpleNamespace()
    env._max_steps = None

    with pytest.raises(CuaGymTaskError) as raised:
        await env.reset()

    assert raised.value.kind == "excluded_task"
    assert not isinstance(raised.value, (RuntimeError, TimeoutError, CapacityExhausted))
    assert is_retryable(raised.value) is False  # terminal: a re-run refuses identically


async def test_non_excluded_task_is_untouched_by_the_guard():
    """(c) A runnable row must be completely unaffected: same args, same body,
    no error — including rows whose ``others`` exists without an exclude_reason
    and rows that carry no ``others`` at all."""
    seen = []

    async def backend_setup(task, computer):
        seen.append((task, computer))

    computer = SimpleNamespace()
    for metadata in (
        {"others": {"apps": ["instagram"]}},
        {"others": {}},
        {},
    ):
        task = SimpleNamespace(task_id="cuagym_runnable", metadata=metadata)
        assert await M.guard_excluded(backend_setup)(task, computer) is None

    assert [t.metadata for t, _ in seen] == [
        {"others": {"apps": ["instagram"]}}, {"others": {}}, {},
    ]
    assert all(c is computer for _, c in seen)


def test_cuagym_constructor_wires_cap(monkeypatch):
    """cuagym's constructor calls the docker-gated ``_check_images``, so the
    non-live suite otherwise never runs the REAL constructor — every reset
    test builds via ``object.__new__`` and PLANTS
    ``_max_resets_per_container``/``_cuagym_episode_started``, which would mask a
    dropped cap-wiring line (green suite; a real constructed env only trips
    ``reset_with_recycle``'s ``cap is None`` -> RuntimeError at runtime / the live
    smoke). Monkeypatch the image check out and run the real constructor so a
    regression fails HERE. cuaworld's dep-light twin is already behaviorally
    pinned by its own constructor tests."""
    monkeypatch.setattr(M, "_check_images", lambda: None)
    env = M._CuaGymEnv()
    assert env._max_resets_per_container == 0
    assert env._cuagym_episode_started is False


# --- the in-guest LLM judge (`/tmp/reward_judge.py`) -----------------------


def _judge_namespace(**overrides):
    """Exec the guest module exactly as it lands in the container."""
    config = {"model": "gpt-4o", "base_url": None, "api_key": "k",
              "max_retries": 1, "timeout": 5.0}
    config.update(overrides)
    namespace: dict = {}
    exec(compile(runtime.reward_judge_source(config), "reward_judge", "exec"), namespace)
    return namespace


def test_reward_judge_module_is_valid_guest_python_with_config_baked_in():
    source = runtime.reward_judge_source(
        {"model": "openai/x", "base_url": None, "api_key": None,
         "max_retries": 3, "timeout": 180.0}
    )
    # `repr`, not `json.dumps` — JSON's `null` is a NameError in Python source,
    # and an unset base_url/api_key is the common case.
    assert "__CUA_GYM_JUDGE_CONFIG__" not in source
    assert "'base_url': None" in source
    namespace = _judge_namespace()
    # The API the bundles actually call: three parameters, exactly these names.
    # An AST census of every pinned bundle finds 52 call sites in 46 files, all
    # three passed BY KEYWORD at every one, zero positional args, zero extra
    # kwargs — so there is nothing for a `**kwargs` sponge to absorb.
    assert list(
        inspect.signature(namespace["call_llm_judge"]).parameters
    ) == ["task_instruction", "success_criteria", "state_excerpt"]


def test_judge_config_validation_and_redacted_smoke_output(monkeypatch):
    for prefix in ("LITE_CUAGYM_JUDGE_", "VLM_"):
        for suffix in ("MODEL", "BASE_URL", "API_KEY", "MAX_RETRIES", "TIMEOUT"):
            monkeypatch.delenv(prefix + suffix, raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    monkeypatch.setenv("LITE_CUAGYM_JUDGE_MODEL", "openai/")
    with pytest.raises(ValueError, match="empty"):
        runtime.judge_config()

    monkeypatch.setenv("LITE_CUAGYM_JUDGE_MODEL", "openai/gpt-test")
    monkeypatch.setenv("LITE_CUAGYM_JUDGE_MAX_RETRIES", "0")
    with pytest.raises(ValueError, match="MAX_RETRIES"):
        runtime.judge_config()

    monkeypatch.setenv("LITE_CUAGYM_JUDGE_MAX_RETRIES", "1")
    monkeypatch.setenv("LITE_CUAGYM_JUDGE_TIMEOUT", "nan")
    with pytest.raises(ValueError, match="TIMEOUT"):
        runtime.judge_config()

    monkeypatch.setenv("LITE_CUAGYM_JUDGE_TIMEOUT", "60")
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-secret")
    assert runtime.judge_config()["model"] == "gpt-test"
    assert runtime.redacted_judge_config()["api_key"] == "<redacted>"


@pytest.mark.parametrize("reply,expected", [
    ('{"score": 0.75}', 0.75),
    ('prose {"score": 1, "reason": "ok"} more prose', 1.0),
    ("```json\n{\"score\": 0.5}\n```", 0.5),
    ('{"passed": true}', 1.0),
    ('{"score": 80}', 0.8),          # a 0-100 scale, rescaled
    ("Score: 0.4", 0.4),
    ("85%", 0.85),
    ("8/10", 0.8),
    ("3/5", 0.6),
    ("0.9", 0.9),
    ("yes", 1.0),
    ("no", 0.0),
])
def test_reward_judge_reads_the_score_shapes_a_model_actually_emits(reply, expected):
    assert _judge_namespace()["_score_from"](reply) == pytest.approx(expected)


def test_reward_judge_unreadable_reply_scores_zero_but_an_outage_raises():
    """The two ways the judge lets a caller down are NOT the same answer.

    A reply we cannot read is still an answer, and it awards no evidence of
    quality -> 0.0. A provider outage is no answer at all, and it must reach the
    caller's `except`: all 46 live call sites wrap the judge precisely to fall
    back on a deterministic keyword check, so returning 0.0 there would walk past
    that handler and silently delete the consolation credit it grants.
    """
    namespace = _judge_namespace()
    namespace["_ask"] = lambda _prompt: "I would rather not say."
    assert namespace["call_llm_judge"](
        task_instruction="t", success_criteria="s", state_excerpt="e"
    ) == 0.0

    def dead(_prompt):
        raise namespace["JudgeUnavailable"]("URLError: connection refused")

    namespace["_ask"] = dead
    with pytest.raises(namespace["JudgeUnavailable"]):
        namespace["call_llm_judge"](
            task_instruction="t", success_criteria="s", state_excerpt="e"
        )


def test_reward_judge_provider_error_redacts_api_key(monkeypatch):
    namespace = _judge_namespace(api_key="unit-test-secret")

    def urlopen(_request, timeout):
        raise RuntimeError("api_key=unit-test-secret transport down")

    monkeypatch.setattr(namespace["urllib"].request, "urlopen", urlopen)
    with pytest.raises(namespace["JudgeUnavailable"]) as excinfo:
        namespace["_ask"]("prompt")

    message = str(excinfo.value)
    assert "transport down" in message
    assert "unit-test-secret" not in message


def test_reward_judge_endpoint_follows_the_configured_base_url():
    assert _judge_namespace()["_endpoint"]() == (
        "https://api.openai.com/v1/chat/completions"
    )
    assert _judge_namespace(base_url="http://gw.test/v1/")["_endpoint"]() == (
        "http://gw.test/v1/chat/completions"
    )


def test_reward_judge_wire_payload_uses_gpt5_compatible_completion_budget(monkeypatch):
    namespace = _judge_namespace()
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"{\\"score\\": 1}"}}]}'

    def urlopen(request, timeout):
        seen["body"] = json.loads(request.data.decode())
        seen["timeout"] = timeout
        return Response()

    monkeypatch.setattr(namespace["urllib"].request, "urlopen", urlopen)

    assert namespace["call_llm_judge"]("task", "criteria", "state") == 1.0
    assert seen["timeout"] == 5.0
    assert seen["body"]["max_completion_tokens"] == 512
    assert "max_tokens" not in seen["body"]
    assert "temperature" not in seen["body"]


@pytest.mark.asyncio
async def test_install_reward_judge_writes_the_module_upstream_imports():
    written = {}

    class Interface:
        async def write_bytes(self, path, data):
            written[path] = data.decode()

    await runtime.install_reward_judge(SimpleNamespace(interface=Interface()))
    assert set(written) == {"/tmp/reward_judge.py"}
    assert "def call_llm_judge(" in written["/tmp/reward_judge.py"]


def test_bundle_needs_judge_matches_every_pinned_importer_and_nothing_else():
    """Pin the population the shim exists for, on the LIVE (non-excluded) basis.

    46 bundles on disk import it; 2 are excluded, leaving 44 live — 42 web
    tasks and 2 desktop tasks. Every importer is
    reached by the substring gate,
    whether it writes `from reward_judge import …` or
    `importlib.import_module('reward_judge')`.
    """
    catalogs = {
        "web": M._DIR / ".cache" / "web" / "lite.cuagym_tasks" / "train.jsonl",
        "desktop": (
            M._DIR / ".cache" / "desktop" / "lite.cuagym_desktop_tasks" / "train.jsonl"
        ),
    }
    _require_fresh_task_cache()
    on_disk = Counter()
    live = Counter()
    call_sites = 0
    for platform, catalog in catalogs.items():
        for line in catalog.read_text().splitlines():
            row = json.loads(line)
            source = Path(row["metadata"]["reward"]).read_text(errors="replace")
            if not runtime.bundle_needs_judge(source):
                # The gate must not fire on a bundle that never imports it.
                assert "call_llm_judge" not in source
                continue
            on_disk[platform] += 1
            if not (row["metadata"].get("others") or {}).get("exclude_reason"):
                live[platform] += 1
            call_sites += sum(
                1
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.Call)
                and getattr(node.func, "id", getattr(node.func, "attr", None))
                in ("call_llm_judge", "_call_llm_judge")
            )
    assert (sum(on_disk.values()), dict(live)) == (46, {"web": 42, "desktop": 2})
    assert call_sites == 52
