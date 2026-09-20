"""Shared action dispatch for config, postconfig, and oracle.

All three use the same ``{"type": ..., "parameters": {...}}`` format.

Principle: types with Flask server endpoints are forwarded directly via curl.
Types without endpoints (download, sleep, chrome_open/close_tabs,
update_browse_history) and CUA low-level actions use computer.interface.

Usage:
    from lite.gym.envs.lite.osworld.src.utils.dispatch import dispatch_action, dispatch_actions
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shlex
from pathlib import Path

from lite.core.tools.action_space.keys import normalize_keys
from lite.gym.envs.lite.osworld.src.utils.assets import asset_root
from lite.gym.utils.backend.model_inputs import project_model_keys

logger = logging.getLogger(__name__)

_SERVER = "http://localhost:5000"
CHROME_DATA_DIR = "/home/user/chrome-data"
_EXECUTE_GUI_LAUNCHERS = (
    "gnome-terminal",
    "nautilus",
    "nemo",
    "thunar",
    "gedit",
    "mousepad",
)


def _cache_output_path(cache_dir: str, name: str, *, label: str = "output") -> Path:
    """Resolve a host-side dispatch artifact under ``cache_dir``."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"{label} must be a non-empty relative path")
    raw = Path(name)
    if not raw.parts:
        raise ValueError(f"{label} must be a non-empty relative path")
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"{label} must stay inside cache_dir: {name!r}")
    cache_root = Path(cache_dir).resolve()
    path = (cache_root / raw).resolve()
    if path != cache_root and cache_root not in path.parents:
        raise ValueError(f"{label} must stay inside cache_dir: {name!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

# Runtime root for ``host_push`` synth assets. Resolved by the shared
# ``asset_root()`` (HF-downloaded ``.cache/`` → in-repo ``data/`` fallback) so
# runtime + codegen agree on one location. host_push steps store a checkout-
# RELATIVE ``host_path`` (portable across checkouts); it is resolved against
# this root here. An absolute ``host_path`` (legacy) is used as-is.
_SYNTH_ASSET_ROOT = asset_root()

# Template variables replaced in execute commands, matching OSWorld's
# SetupController.replace_screen_env_in_command().
_TEMPLATE_VARS = {
    "{CLIENT_PASSWORD}": "user",
    "{SCREEN_WIDTH}": "1920",
    "{SCREEN_HEIGHT}": "1080",
    "{SCREEN_WIDTH_HALF}": "960",
    "{SCREEN_HEIGHT_HALF}": "540",
}

def _replace_templates(obj):
    """Replace OSWorld template variables in a string or list of strings."""
    if isinstance(obj, str):
        for k, v in _TEMPLATE_VARS.items():
            obj = obj.replace(k, v)
        return obj
    if isinstance(obj, list):
        return [_replace_templates(x) for x in obj]
    return obj


def _execute_uses_ctrl_alt_t(cmd_str_for_check: str) -> bool:
    """Detect pyautogui Ctrl+Alt+T terminal launches in setup commands."""
    lower = cmd_str_for_check.lower()
    return "ctrl" in lower and "alt" in lower and "'t'" in cmd_str_for_check


def _execute_launches_gnome_terminal(cmd_str_for_check: str) -> bool:
    return "gnome-terminal" in cmd_str_for_check


def _execute_launches_gui(cmd_str_for_check: str) -> bool:
    """Whether the execute/command branch should run post-dispatch GUI raise."""
    return (
        any(gui in cmd_str_for_check for gui in _EXECUTE_GUI_LAUNCHERS)
        or _execute_uses_ctrl_alt_t(cmd_str_for_check)
        or _execute_launches_gnome_terminal(cmd_str_for_check)
    )


def _update_browse_history_command(history_entries: list[dict]) -> str:
    entries_json = shlex.quote(json.dumps(history_entries))
    return r"""set -e
export LITE_OSWORLD_HISTORY_ENTRIES=__LITE_OSWORLD_HISTORY_ENTRIES__
if ! python3 - <<'PY'
import os, sqlite3, sys
hist = "/home/user/chrome-data/Default/History"
try:
    conn = sqlite3.connect(hist)
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    conn.close()
except Exception:
    sys.exit(1)
sys.exit(0 if {"urls", "visits", "meta"} <= tables else 1)
PY
then
    rm -f /home/user/chrome-data/Default/History \
          /home/user/chrome-data/Default/History-journal \
          /home/user/chrome-data/Default/History-wal \
          /home/user/chrome-data/Default/History-shm
    mkdir -p /home/user/chrome-data/Default
    bootstrap_dir="$(mktemp -d /tmp/lite-osworld-history-bootstrap.XXXXXX)"
    setsid google-chrome --user-data-dir="$bootstrap_dir" \
        --no-first-run --no-default-browser-check about:blank \
        >/tmp/lite-osworld-history-bootstrap.log 2>&1 &
    bootstrap_pid="$!"
    sleep 5
    kill -- "-$bootstrap_pid" 2>/dev/null || true
    sleep 1
    kill -9 -- "-$bootstrap_pid" 2>/dev/null || true
    wait "$bootstrap_pid" 2>/dev/null || true
    if [ -f "$bootstrap_dir/Default/History" ]; then
        cp "$bootstrap_dir/Default/History" /home/user/chrome-data/Default/History
    fi
    rm -rf "$bootstrap_dir"
    sleep 2
fi
python3 - <<'PY'
import json, os, sqlite3
from collections import defaultdict
from datetime import datetime, timedelta

entries = json.loads(os.environ["LITE_OSWORLD_HISTORY_ENTRIES"])
hist = "/home/user/chrome-data/Default/History"
os.makedirs(os.path.dirname(hist), exist_ok=True)
conn = sqlite3.connect(hist)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS urls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url LONGVARCHAR,
    title LONGVARCHAR DEFAULT '',
    visit_count INTEGER DEFAULT 0 NOT NULL,
    typed_count INTEGER DEFAULT 0 NOT NULL,
    last_visit_time INTEGER NOT NULL,
    hidden INTEGER DEFAULT 0 NOT NULL
)''')
c.execute('''CREATE TABLE IF NOT EXISTS visits (
    id INTEGER PRIMARY KEY,
    url INTEGER NOT NULL,
    visit_time INTEGER NOT NULL,
    from_visit INTEGER DEFAULT 0,
    transition INTEGER DEFAULT 0 NOT NULL,
    segment_id INTEGER DEFAULT 0,
    visit_duration INTEGER DEFAULT 0 NOT NULL
)''')

prepared = []
base_time = datetime.now()
epoch = datetime(1601, 1, 1)
for e in entries:
    from_now = e.get("visit_time_from_now_in_seconds", 0)
    vt = base_time - timedelta(seconds=from_now)
    chrome_ts = int((vt - epoch).total_seconds() * 1000000)
    prepared.append({
        "url": e["url"],
        "title": e.get("title", ""),
        "chrome_ts": chrome_ts,
    })

target_urls = sorted({e["url"] for e in prepared})
if target_urls:
    url_placeholders = ",".join("?" for _ in target_urls)
    old_url_ids = [
        row[0] for row in c.execute(
            f"SELECT id FROM urls WHERE url IN ({url_placeholders})",
            target_urls,
        ).fetchall()
    ]
    if old_url_ids:
        id_placeholders = ",".join("?" for _ in old_url_ids)
        c.execute(f"DELETE FROM visits WHERE url IN ({id_placeholders})", old_url_ids)
        c.execute(f"DELETE FROM urls WHERE id IN ({id_placeholders})", old_url_ids)

by_url = defaultdict(list)
for e in prepared:
    by_url[e["url"]].append(e)

for url, url_entries in by_url.items():
    last = max(url_entries, key=lambda e: e["chrome_ts"])
    c.execute(
        "INSERT INTO urls "
        "(url,title,visit_count,typed_count,last_visit_time,hidden) "
        "VALUES (?,?,?,?,?,0)",
        (url, last["title"], len(url_entries), 0, last["chrome_ts"]),
    )
    uid = c.lastrowid
    for e in url_entries:
        c.execute(
            "INSERT INTO visits (url,visit_time,from_visit,transition,segment_id,visit_duration) "
            "VALUES (?,?,0,805306368,0,0)",
            (uid, e["chrome_ts"]),
        )
conn.commit()
try:
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.commit()
except Exception:
    pass
conn.close()
print("history seeded:", len(entries))
PY
""".replace("__LITE_OSWORLD_HISTORY_ENTRIES__", entries_json)


def _compare_urls(url1: str, url2: str) -> bool:
    """Compare two URLs with normalization, matching OSWorld's compare_urls()."""
    from urllib.parse import urlparse
    if not url1 or not url2:
        return url1 == url2

    def _normalize(url: str) -> str:
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        p = urlparse(url)
        host = p.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        path = p.path.rstrip("/")
        return f"{host}{path}{('?' + p.query) if p.query else ''}"

    return _normalize(url1) == _normalize(url2)


async def _flask_post(computer, endpoint: str, data: dict) -> str:
    """POST to Flask server endpoint, return raw response."""
    payload = shlex.quote(json.dumps(data))
    r = await computer.interface.run_command(
        f"curl -s -X POST {_SERVER}{endpoint} "
        f"-H 'Content-Type: application/json' -d {payload}"
    )
    return r.stdout


# Allowlist of container paths that host→container file push is permitted to
# write. Anything outside this set indicates a template bug; we fail loudly
# rather than silently writing somewhere unsafe.
_DST_PREFIX_ALLOWLIST = ("/home/user/", "/tmp/", "/var/tmp/")


async def _fix_thunderbird_profile(computer) -> None:
    """Fix Thunderbird profile selection after extracting an OSWorld tarball.

    The tarball's installs.ini references the OSWorld VM's install ID, which
    doesn't match the container's Thunderbird.  Thunderbird falls back to the
    profile marked Default=1 (an empty stub).  Fix: rewrite profiles.ini so
    the *-default-release profile (the one with actual mail data) is Default=1.
    """
    r = await computer.interface.run_command(
        "cat /home/user/.thunderbird/profiles.ini 2>/dev/null"
    )
    if not r.stdout.strip():
        return

    lines = r.stdout.splitlines()
    sections: list[list[str]] = []
    cur: list[str] = []
    for line in lines:
        if line.startswith("[") and cur:
            sections.append(cur)
            cur = []
        cur.append(line)
    if cur:
        sections.append(cur)

    # Find the section whose Path contains "default-release" and set Default=1;
    # remove Default=1 from all other Profile sections. Capture that section's
    # Path= value so the donation-tab suppression below writes to the profile
    # Thunderbird will actually load, instead of a hardcoded guess.
    release_path: str | None = None
    for sec in sections:
        header = sec[0] if sec else ""
        if not header.startswith("[Profile"):
            continue
        has_release = any("default-release" in l for l in sec)
        # Remove existing Default line
        sec[:] = [l for l in sec if not l.startswith("Default=")]
        if has_release:
            sec.append("Default=1")
            for l in sec:
                if l.startswith("Path="):
                    release_path = l[len("Path="):].strip()

    new_ini = "\n".join(l for sec in sections for l in sec) + "\n"
    escaped = new_ini.replace("'", "'\\''")
    await computer.interface.run_command(
        f"printf '%s' '{escaped}' > /home/user/.thunderbird/profiles.ini"
    )
    logger.info("Fixed Thunderbird profiles.ini: default-release set as Default")

    if release_path is None:
        logger.warning(
            "Thunderbird profiles.ini: no default-release Path found; "
            "skipping donation-tab suppression"
        )
        return

    # Suppress the first-run/upgrade donation tab so Thunderbird opens to the inbox.
    # Write to user.js only: Thunderbird re-applies user.js on every startup,
    # layered on top of prefs.js, so these prefs take effect regardless of
    # whatever prefs.js already contains — a separate prefs.js write is redundant.
    _tb_profile = f"/home/user/.thunderbird/{release_path}"
    _donation_prefs = [
        'user_pref("mailnews.start_page.enabled", false);',
        'user_pref("mailnews.start_page_override.mstone", "ignore");',
        'user_pref("app.donation.eoy.version.viewed", 99999);',
        'user_pref("app.donation.eoy.version.last_viewed", 99999);',
        'user_pref("app.update.checkInstallTime", false);',
        # TB 140+: donation tab is opened via the in-app notification system.
        # Disabling it prevents the EOY appeal tab from appearing at all.
        'user_pref("mail.inappnotifications.enabled", false);',
    ]
    for pref in _donation_prefs:
        await computer.interface.run_command(
            f"echo '{pref}' >> {_tb_profile}/user.js"
        )
    logger.info("Wrote user.js to suppress Thunderbird start/donation pages")


async def dispatch_actions(
    computer,
    actions: list[dict],
    cache_dir: str = "/tmp",
) -> None:
    """Execute a list of actions sequentially."""
    for i, action in enumerate(actions):
        try:
            await dispatch_action(computer, action, cache_dir=cache_dir)
        except Exception as e:
            logger.warning("Action %d (%s) failed: %s", i, action.get("type", "?"), e)


async def dispatch_action(
    computer,
    action: dict,
    cache_dir: str = "/tmp",
):
    """Execute a single action.

    Types with Flask endpoints are forwarded directly.
    Types without endpoints use computer.interface.
    """
    t = action.get("type", "")
    p = action.get("parameters", {})

    # ===================================================================
    # Types forwarded to Flask server endpoints
    # ===================================================================

    # --- execute/command via exec-stdio run_command ---
    # The legacy OSWorld ``shell`` flag is not a routing boundary here: strings
    # and argv lists are both normalized to the command line sent to run_command.
    if t in ("execute", "command"):
        cmd = _replace_templates(p.get("command", ""))
        if isinstance(cmd, str):
            # Chrome profile redirect: OSWorld tasks write to /home/user/.config/google-chrome
            # but we launch Chrome with --user-data-dir=/home/user/chrome-data.
            # Rewrite so jq/python3 setup steps actually affect the profile Chrome reads.
            cmd = cmd.replace("/home/user/.config/google-chrome", CHROME_DATA_DIR)
            # Relative-path setup scripts (os/ tasks): "chmod +x setup.sh" / "./setup.sh" /
            # "bash ./setup.sh" reference a file downloaded with a relative path.
            # The download fix puts the file at /home/user/setup.sh; rewrite the
            # shell commands to reference the absolute path so they don't look in
            # /home/user (CWD of the Flask server) where the file doesn't exist.
            cmd = cmd.replace("chmod +x setup.sh", "chmod +x /home/user/setup.sh")
            cmd = cmd.replace("./setup.sh", "/home/user/setup.sh")
            cmd = cmd.replace("bash setup.sh", "bash /home/user/setup.sh")
            # (An upstream setup that hardcodes the QEMU-VM systemd session paths
            # DBUS_SESSION_BUS_ADDRESS=…/run/user/1000/bus / XDG_RUNTIME_DIR=/run/user/1000
            # needs no rewrite: the exec-stdio server already runs as the desktop
            # user (uid 1000), so those hardcoded /run/user/1000 paths match the
            # container's real session paths.)
        # When a setup step launches a GUI app (gnome-terminal, nautilus, etc.)
        # via execute (not launch), the new window may lose focus to an existing
        # app window.  Raise and focus it after a brief settle delay.
        cmd_str_for_check = cmd if isinstance(cmd, str) else " ".join(str(c) for c in cmd)
        _launches_gui = _execute_launches_gui(cmd_str_for_check)
        # Some OSWorld setups use pyautogui Ctrl+Alt+T to open a terminal.
        # Our GNOME Docker session doesn't bind this shortcut by default. Detect
        # it and launch gnome-terminal explicitly as a fallback.
        _uses_ctrl_alt_t = _execute_uses_ctrl_alt_t(cmd_str_for_check)
        # A plain `gnome-terminal ...` setup step hits the same wall: gnome-terminal
        # is a thin D-Bus client and can't activate gnome-terminal-server from the
        # Flask /execute env ("connection is closed"), so its window never appears and
        # the raise-below finds nothing. Detect it so the block below spawns the
        # terminal itself (with the server pre-started) instead of only raising it.
        _launches_gnome_terminal = _execute_launches_gnome_terminal(cmd_str_for_check)
        # The whole command runs as the desktop user (the exec-stdio server's
        # identity), so the session/GUI/office tools it shells out to (gnome-terminal,
        # gsettings/dconf, soffice, ...) run in the user session directly and the files
        # it writes are born user-owned. Apps that geteuid()-check and ABORT under root
        # (thunderbird, vlc) therefore need no special handling; Chrome/Chromium are
        # likewise fine because every launch passes `--no-sandbox` plus an explicit
        # `--user-data-dir`.
        # A11y eval scripts (AT-SPI probes: show-thunderbird-attachments.py + its
        # `cssselect` dep) `import pyatspi`, whose gi/C-ext bindings are installed ONLY
        # for the SYSTEM python (Debian python3-pyatspi), not the /opt/env venv that the
        # exec-stdio server's PATH (server.py) resolves a bare `python`/`pip` to. So a
        # bare `python <a11y-script>` hits the venv → ModuleNotFoundError → empty stdout
        # → false-negative (osworld_thunderbird_d38192b0: agent attached the file on
        # screen, reward 0). Pin the a11y script AND the pip that installs its deps to
        # /usr/bin/python3 so they run under system python — exactly as they do on the
        # osworld VM, where this task passes. TARGETED to the a11y-script family (matched
        # by the script name / its cssselect dep) so the venv-dependent pyautogui /
        # PyMuPDF / Pillow / openpyxl eval commands (which need the venv's pip-installed
        # packages) are left untouched.
        _a11y_str = cmd if isinstance(cmd, str) else " ".join(str(x) for x in cmd)
        if "show-thunderbird-attachments" in _a11y_str or ("cssselect" in _a11y_str and "pip" in _a11y_str):
            _toks = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
            if _toks and _toks[0] in ("pip", "pip3"):
                cmd = ["/usr/bin/python3", "-m", "pip"] + _toks[1:]
            elif _toks and _toks[0] in ("python", "python3"):
                cmd = ["/usr/bin/python3"] + _toks[1:]
        cmd_run = cmd if isinstance(cmd, str) else shlex.join(cmd)
        r = await computer.interface.run_command(cmd_run)
        resp = {"output": r.stdout, "error": r.stderr, "returncode": r.returncode}
        stdout_file = p.get("stdout", "")
        if stdout_file:
            with _cache_output_path(cache_dir, stdout_file, label="stdout").open("w") as f:
                f.write(resp.get("output", ""))
        stderr_file = p.get("stderr", "")
        if stderr_file:
            with _cache_output_path(cache_dir, stderr_file, label="stderr").open("w") as f:
                f.write(resp.get("error", ""))
        # Raise the newly-launched GUI window to the foreground.
        # Wait up to 5 s for the window to appear (it may fork and return fast).
        if _launches_gui:
            await asyncio.sleep(3)
            # Ctrl+Alt+T, or a plain `gnome-terminal` setup step: spawn the terminal
            # ourselves. gnome-terminal can't D-Bus-activate its server from this env,
            # so pre-start gnome-terminal-server on the desktop session bus + DISPLAY=:1
            # first, then launch a maximized window (matching the VM snapshot). Preserve
            # the command's --working-directory. Skip if a terminal is already open.
            if _uses_ctrl_alt_t or _launches_gnome_terminal:
                _wd = ""
                try:
                    _toks = shlex.split(cmd_str_for_check)
                    for _i, _tok in enumerate(_toks):
                        if _tok.startswith("--working-directory="):
                            _wd = _tok.split("=", 1)[1]
                        elif _tok == "--working-directory" and _i + 1 < len(_toks):
                            _wd = _toks[_i + 1]
                except ValueError:
                    pass
                _wd_flag = f"--working-directory={shlex.quote(_wd)}" if _wd else ""
                # A bare `gnome-terminal` runs as the desktop user (the server's
                # identity) with the session bus available and D-Bus-activates its own
                # gnome-terminal-server — so no pre-start needed. Guard on whether a
                # terminal window already exists (xdotool on the shared :1), else
                # launch one.
                _guard = await computer.interface.run_command(
                    "DISPLAY=:1 xdotool search --class 'terminal' 2>/dev/null | grep -q .",
                )
                if _guard.returncode != 0:
                    await computer.interface.run_command(
                        f"DISPLAY=:1 nohup gnome-terminal --maximize {_wd_flag} "
                        ">/dev/null 2>&1 & disown",
                    )
                await asyncio.sleep(2)
            # Activate the terminal window. `xdotool` is env-vendored (resolved via
            # the server's PATH); it drives the live user X session on DISPLAY=:1 as
            # the desktop user (load-bearing, verified). wmctrl (system) also acts on :1.
            await computer.interface.run_command(
                # Try xdotool class search first (robust across title variants),
                # then wmctrl title search, then activate the most-recently-opened window.
                "export DISPLAY=:1; "
                "xdotool search --class 'terminal' | tail -1 | xargs -r xdotool windowactivate --sync 2>/dev/null || "
                "wmctrl -a terminal 2>/dev/null || wmctrl -a Terminal 2>/dev/null || "
                "wmctrl -l | awk 'END{print $1}' | xargs -r wmctrl -ia 2>/dev/null || true",
            )
        return resp

    # --- launch → POST /setup/launch ---
    elif t == "launch":
        cmd = p.get("command", "")
        shell = p.get("shell", False)
        if isinstance(cmd, str) and not shell and len(cmd.split()) > 1:
            cmd = cmd.split()
        if isinstance(cmd, list):
            cmd_str = " ".join(cmd)
            # Chrome needs special flags for our container
            if "google-chrome" in cmd_str:
                if "--no-sandbox" not in cmd:
                    idx = next(i for i, c in enumerate(cmd) if "google-chrome" in c)
                    cmd.insert(idx + 1, "--no-sandbox")
                if "--user-data-dir" not in cmd_str:
                    cmd.append(f"--user-data-dir={CHROME_DATA_DIR}")
                if "--remote-allow-origins" not in cmd_str:
                    cmd.append("--remote-allow-origins=*")
                # Suppress info bars that distract the agent:
                # --test-type: hides "unsupported flag --no-sandbox" warning
                # --no-default-browser-check: hides "set as default browser" bar
                if "--test-type" not in cmd_str:
                    cmd.append("--test-type")
                if "--no-default-browser-check" not in cmd_str:
                    cmd.append("--no-default-browser-check")
                # --password-store=basic: use plaintext password storage
                # instead of gnome-keyring, which pops up a "Choose password
                # for new keyring" dialog under GNOME that blocks Chrome.
                if "--password-store" not in cmd_str:
                    cmd.append("--password-store=basic")
                # --disable-dev-shm-usage: Docker containers have a default
                # /dev/shm of only 64MB; Chrome renderer processes crash
                # ("Aw, Snap!") when they exhaust it on heavy pages.
                # This flag makes Chrome use /tmp instead.
                if "--disable-dev-shm-usage" not in cmd_str:
                    cmd.append("--disable-dev-shm-usage")
                # Start maximized to match OSWorld QEMU VM snapshot state.
                if "--start-maximized" not in cmd_str:
                    cmd.append("--start-maximized")
            # VS Code needs --no-sandbox in Docker.
            # Also disable workspace trust dialog: OSWorld QEMU VMs have trust
            # disabled; in Docker the dialog blocks extension commands during eval.
            # Accept absolute paths (e.g. "/usr/bin/code") too —
            # synth/vs_code.py:_vs_code_preopen_steps uses the absolute-path
            # form, which the check below matches so the workspace-trust dialog
            # is dismissed for those ~30 vs_code tasks.
            if cmd and (cmd[0] in ("code", "code-oss") or cmd[0].endswith("/code") or cmd[0].endswith("/code-oss")):
                if "--no-sandbox" not in cmd:
                    cmd.insert(1, "--no-sandbox")
                await computer.interface.run_command(
                    "mkdir -p /home/user/.config/Code/User && "
                    "python3 -c \""
                    "import json, os; "
                    "p = '/home/user/.config/Code/User/settings.json'; "
                    "s = json.load(open(p)) if os.path.exists(p) else {}; "
                    "s['security.workspace.trust.enabled'] = False; "
                    "json.dump(s, open(p, 'w'), indent=2)"
                    "\""
                )
            # Thunderbird: fix profile selection so it uses the tarball profile,
            # and close any donation/what's-new tabs after launch.
            if "thunderbird" in cmd_str:
                await _fix_thunderbird_profile(computer)
            # VLC: suppress first-run privacy dialog by pre-writing vlcrc.
            # The file requires [qt] section header; just appending the key
            # without a section causes VLC to ignore it.
            # Only write if vlcrc doesn't exist — oracle may have already
            # written a full vlcrc with task-specific settings; overwriting it
            # would erase those changes and cause evaluators to see defaults.
            if "vlc" in cmd_str and "vlc" in (cmd[0] if cmd else ""):
                await computer.interface.run_command(
                    "mkdir -p /home/user/.config/vlc && "
                    "{ test -f /home/user/.config/vlc/vlcrc || printf '[qt]\\nqt-privacy-ask=0\\none-instance=1\\none-instance-when-started-from-file=1\\n[core]\\nextraintf=http\\n[http]\\nhttp-host=localhost\\nhttp-port=8080\\nhttp-password=password\\n' > /home/user/.config/vlc/vlcrc; }"
                )
        # VLC with shell=true (string cmd): also suppress the privacy dialog.
        # The block above only runs for list commands, so handle string cmd here.
        if isinstance(cmd, str) and "vlc" in cmd:
            await computer.interface.run_command(
                "mkdir -p /home/user/.config/vlc && "
                "{ test -f /home/user/.config/vlc/vlcrc || printf '[qt]\\nqt-privacy-ask=0\\none-instance=1\\none-instance-when-started-from-file=1\\n[core]\\nextraintf=http\\n[http]\\nhttp-host=localhost\\nhttp-port=8080\\nhttp-password=password\\n' > /home/user/.config/vlc/vlcrc; }"
            )
        await _flask_post(computer, "/setup/launch", {
            "command": cmd, "shell": shell,
        })
        await asyncio.sleep(2)

        _cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd) if isinstance(cmd, list) else ""

        # Thunderbird post-launch: wait extra for TB to finish loading.
        if "thunderbird" in _cmd_str:
            await asyncio.sleep(3)

        # Maximize GUI app windows to match OSWorld QEMU VM snapshot state
        # (most apps start maximized in the VM). Chrome uses --start-maximized
        # above; other apps need wmctrl after the window appears.
        # NOTE: VLC is intentionally EXCLUDED — the osworld reference sizes the VLC
        # window to the media's native resolution and floats it (verified via
        # head-to-head); force-maximizing it diverged (upscaled small clips to
        # full-screen, and hid VLC entirely behind a fullscreen PDF in multi-app
        # tasks), shifting the agent's click coordinates.
        _GUI_APPS = ("libreoffice", "soffice", "gimp", "thunderbird", "code", "gedit", "nautilus", "evince")
        if "gimp" in _cmd_str:
            # GIMP shows a splash + ICC-profile dialog for a few seconds after
            # launch, so a one-shot `wmctrl -r :ACTIVE:` maximizes that transient,
            # not the image window. Poll (~10s) for the image window by its stable
            # WM role (gimp-image-window persists empty→loaded, while the window
            # *title* changes once an image opens), then maximize it by id.
            await computer.interface.run_command(
                "for i in $(seq 1 20); do "
                "  for w in $(xdotool search --class gimp 2>/dev/null); do "
                "    if xprop -id \"$w\" WM_WINDOW_ROLE 2>/dev/null | grep -q gimp-image-window; then "
                "      wmctrl -i -r \"$w\" -b add,maximized_vert,maximized_horz; exit 0; "
                "    fi; "
                "  done; sleep 0.5; "
                "done; true"
            )
        elif any(app in _cmd_str for app in _GUI_APPS):
            await computer.interface.run_command(
                "wmctrl -r :ACTIVE: -b add,maximized_vert,maximized_horz 2>/dev/null || true"
            )

    # --- open → POST /setup/open_file ---
    elif t == "open":
        path = p.get("path", "")
        if path:
            # Flask server CWD is /home/user. Relative paths would resolve to
            # /home/user/... which may not exist. Anchor them at the user home so the
            # file-existence check succeeds.
            if not path.startswith("/"):
                path = f"/home/user/{path}"
            await _flask_post(computer, "/setup/open_file", {"path": path})
            # open_file only activates (focuses) the window — does not maximize.
            # In OSWorld QEMU, the snapshot already has apps maximized; xdg-open
            # just opens the file into the existing maximized window.  In
            # lite.osworld (Docker), LibreOffice/Evince starts fresh, so we must
            # maximize the resulting window ourselves.
            await computer.interface.run_command(
                "wmctrl -r :ACTIVE: -b add,maximized_vert,maximized_horz 2>/dev/null || true"
            )

    # --- activate_window → POST /setup/activate_window ---
    elif t == "activate_window":
        await _flask_post(computer, "/setup/activate_window", {
            "window_name": p.get("window_name", ""),
            "strict": p.get("strict", False),
            "by_class": p.get("by_class", False),
        })

    # --- close_window → POST /setup/close_window ---
    elif t == "close_window":
        await _flask_post(computer, "/setup/close_window", {
            "window_name": p.get("window_name", ""),
            "strict": p.get("strict", False),
            "by_class": p.get("by_class", False),
        })

    # ===================================================================
    # Types without Flask endpoints — use computer.interface
    # ===================================================================

    # --- download (container-internal wget, faster than host→upload) ---
    elif t == "download":
        for f in p.get("files", []):
            url, path = f.get("url", ""), f.get("path", "")
            if url and path:
                # The computer server's CWD is /home/user. Relative paths silently
                # fail at mkdir. Anchor them at the user home so mkdir + cp succeed.
                if not path.startswith("/"):
                    path = f"/home/user/{path}"
                cache_name = hashlib.md5(url.encode()).hexdigest()[:12] + "_" + os.path.basename(url).split("?")[0]
                cache_path = f"/tmp/osworld_cache/{cache_name}"
                # Retry loop: up to 5 attempts with exponential backoff (3s, 6s, 12s,
                # 24s). HuggingFace CDN occasionally drops connections or returns
                # transient errors; silent wget failure left files missing, causing
                # compare_references(empty,empty)=1 false trivial-passes.
                await computer.interface.run_command(
                    f"mkdir -p /tmp/osworld_cache $(dirname '{path}') && "
                    f"if [ -f '{cache_path}' ] && [ -s '{cache_path}' ]; then cp '{cache_path}' '{path}'; "
                    f"else "
                    f"  ok=0; delay=3; "
                    f"  for attempt in 1 2 3 4 5; do "
                    f"    wget -q --tries=1 --timeout=60 -O '{cache_path}.tmp' '{url}' && "
                    f"    [ -s '{cache_path}.tmp' ] && mv '{cache_path}.tmp' '{cache_path}' && ok=1 && break; "
                    f"    rm -f '{cache_path}.tmp'; "
                    f"    sleep $delay; delay=$((delay * 2)); "
                    f"  done; "
                    f"  [ $ok -eq 1 ] && cp '{cache_path}' '{path}'; "
                    f"fi"
                )

    # --- sleep (client-side) ---
    elif t == "sleep":
        await asyncio.sleep(p.get("seconds", 1))

    # --- chrome_open_tabs (CDP HTTP API, no Flask endpoint) ---
    elif t == "chrome_open_tabs":
        urls_to_open = p.get("urls_to_open", [])
        if not urls_to_open:
            return

        # Retry connecting to CDP (matches OSWorld's 15-attempt loop). curl -m 2 so a
        # hung/refused socket fails fast instead of stalling the preamble on Docker cold-start.
        tabs_json = None
        for attempt in range(15):
            r = await computer.interface.run_command("curl -s -m 2 http://localhost:1337/json")
            if r.stdout.strip().startswith("["):
                tabs_json = r.stdout
                break
            if attempt > 0:
                logger.warning("CDP not ready (attempt %d/15)", attempt + 1)
            await asyncio.sleep(2)

        if not tabs_json:
            logger.error("Failed to connect to CDP after 15 attempts")
            return

        # Remember default tab to close later
        try:
            tabs = json.loads(tabs_json)
            default_tab_id = next((t["id"] for t in tabs if t.get("type") == "page"), None)
        except (json.JSONDecodeError, TypeError):
            default_tab_id = None

        _BLANK = ("about:blank", "chrome://newtab/", "chrome://new-tab-page/", "")
        for i, url in enumerate(urls_to_open):
            # Snapshot existing tab ids so we can identify THE new tab below (not merely that
            # the count grew — count alone would let setup return before the tab navigated).
            r0 = await computer.interface.run_command("curl -s -m 2 http://localhost:1337/json")
            try:
                before_ids = {p.get("id") for p in json.loads(r0.stdout) if p.get("type") == "page"}
            except (json.JSONDecodeError, TypeError):
                before_ids = set()
            await computer.interface.run_command(
                f"curl -s -m 2 -X PUT 'http://localhost:1337/json/new?{url}'"
            )
            # Wait until the NEW tab has actually navigated: its URL matches the requested one
            # (normalized via _compare_urls, so http/https + www + trailing-slash match) OR it
            # committed to any non-blank page (a redirect to a different host). This mirrors
            # upstream v1's playwright `page.goto` (returns on the load event, follows redirects)
            # instead of returning the instant the tab merely exists. Bounded at 30s so a hung
            # load can't stall setup; the old raw substring match never matched a redirected/
            # normalized URL and burned the full 60s per URL.
            for _ in range(30):
                r = await computer.interface.run_command("curl -s -m 2 http://localhost:1337/json")
                try:
                    pages = [p for p in json.loads(r.stdout) if p.get("type") == "page"]
                    new = [p for p in pages if p.get("id") not in before_ids]
                    if new and (_compare_urls(url, new[0].get("url", ""))
                                or new[0].get("url", "") not in _BLANK):
                        break
                except (json.JSONDecodeError, TypeError):
                    pass
                await asyncio.sleep(1)
            logger.info("Opened tab %d: %s", i + 1, url)

            # Close default blank tab after first new tab (matches OSWorld)
            if i == 0 and default_tab_id:
                await computer.interface.run_command(
                    f"curl -s http://localhost:1337/json/close/{default_tab_id}"
                )

    # --- chrome_close_tabs (CDP HTTP API, no Flask endpoint) ---
    elif t == "chrome_close_tabs":
        urls_to_close = p.get("urls_to_close", [])
        if not urls_to_close:
            return

        await asyncio.sleep(5)  # Wait for Chrome (matches OSWorld)

        r = await computer.interface.run_command("curl -s http://localhost:1337/json")
        try:
            pages = json.loads(r.stdout)
        except (json.JSONDecodeError, TypeError):
            logger.error("Failed to get CDP tabs for close")
            return

        for url_to_close in urls_to_close:
            for page in pages:
                if page.get("type") != "page":
                    continue
                if _compare_urls(page.get("url", ""), url_to_close):
                    await computer.interface.run_command(
                        f"curl -s http://localhost:1337/json/close/{page['id']}"
                    )
                    pages.remove(page)
                    logger.info("Closed tab: %s", url_to_close)
                    break

    # --- update_browse_history (no Flask endpoint, container SQLite) ---
    elif t == "update_browse_history":
        # Support both "history" key (current format) and "entries" key (legacy format).
        # Also support "days_ago" as an alias for visit_time_from_now_in_seconds.
        history_entries = p.get("history", p.get("entries", []))
        # Normalize: "days_ago" → "visit_time_from_now_in_seconds"
        history_entries = [
            dict(e, visit_time_from_now_in_seconds=e["days_ago"] * 86400)
            if "days_ago" in e and "visit_time_from_now_in_seconds" not in e
            else e
            for e in history_entries
        ]
        if history_entries:
            return await computer.interface.run_command(_update_browse_history_command(history_entries))

    # ===================================================================
    # CUA low-level actions (for oracle GUI operations)
    # ===================================================================

    elif t == "left_click":
        x, y = p.get("coordinate", [0, 0])
        await computer.interface.left_click(x, y)

    elif t == "right_click":
        x, y = p.get("coordinate", [0, 0])
        await computer.interface.right_click(x, y)

    elif t == "double_click":
        x, y = p.get("coordinate", [0, 0])
        await computer.interface.double_click(x, y)

    elif t == "type_text":
        await computer.interface.type_text(p.get("text", ""))

    elif t == "key":
        # ``keys`` is canonical Lite payload. ``key`` is a source/config chord
        # string and is the only field normalized here.
        if "keys" in p:
            keys = project_model_keys(p["keys"], action_name="key", backend="xdotool")
        elif "key" in p:
            chord = p["key"]
            if not isinstance(chord, str):
                raise ValueError("key.key must be a string")
            keys = project_model_keys(
                normalize_keys(chord),
                action_name="key",
                backend="xdotool",
            )
        else:
            keys = project_model_keys([], action_name="key", backend="xdotool")
        await computer.interface.hotkey(*keys)

    elif t == "scroll":
        direction = p.get("direction", "down")
        amount = p.get("amount", 3)
        if direction == "down":
            await computer.interface.scroll_down(clicks=amount)
        elif direction == "up":
            await computer.interface.scroll_up(clicks=amount)

    elif t == "mouse_move":
        x, y = p.get("coordinate", [0, 0])
        await computer.interface.move_cursor(x, y)

    elif t == "screenshot":
        pass  # no-op, caller handles screenshots

    elif t == "wait":
        await asyncio.sleep(p.get("duration", 5.0))

    # --- host_push (read host files → write_bytes into container) ---
    # Lightweight per-task asset staging for the host-heredoc synth design
    # (devs/envs/lite.osworld/synth/AGENTS.md). Used to copy a small set of
    # in-repo assets (lite/.../data/assets/synth/...) directly into the
    # container so synth source-file builders don't need to network-fetch.
    #
    # Wire format:
    #   {"type": "host_push", "parameters": {"files": [
    #       {"host_path": "/abs/host/path", "dst": "/home/user/Desktop/foo.jpg"},
    #       ...
    #   ]}}
    elif t == "host_push":
        for f in p.get("files", []):
            host_path = f.get("host_path", "")
            dst = f.get("dst", "")
            if not host_path or not dst:
                continue
            assert dst.startswith(_DST_PREFIX_ALLOWLIST), (
                f"host_push dst not in allowlist: {dst!r}"
            )
            # host_path is checkout-relative (portable); resolve against this
            # checkout's asset root. Absolute (legacy) paths are used as-is.
            src_path = Path(host_path)
            if not src_path.is_absolute():
                src_path = _SYNTH_ASSET_ROOT / host_path
            parent = dst.rsplit("/", 1)[0]
            if parent:
                await computer.interface.run_command(
                    f"mkdir -p '{parent}'"
                )
            with open(src_path, "rb") as fh:
                payload = fh.read()
            # The server runs as the desktop user, so the synth asset lands
            # owned by that user (its ~/Desktop) — an in-place overwrite by the
            # agent (same user) succeeds. Essential winnability on 489 synth.train
            # rows whose task overwrites the seeded file.
            await computer.interface.write_bytes(dst, payload)

    else:
        logger.warning("Unknown action type: %s", t)
