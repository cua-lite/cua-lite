"""Live isolation checks for a built lite.cuagym image."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid

import pytest

pytestmark = pytest.mark.live

_IMAGE = os.environ.get("LITE_CUAGYM_TEST_IMAGE", "cua-lite/lite.cuagym:latest")

# Single-sourced from the parent so the child cannot drift behind it.
from tests.gym.envs.lite.osworld.test_lite_osworld_agent_parity import (  # noqa: E402
    CLI_FORBIDDEN as _PARENT_CLI_FORBIDDEN,
)


def _image_present() -> bool:
    return shutil.which("docker") is not None and subprocess.run(
        ["docker", "image", "inspect", _IMAGE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


@pytest.fixture(scope="module")
def probe() -> dict:
    if not _image_present():
        pytest.skip(f"{_IMAGE} is not built")

    snippet = r"""
import hashlib
import importlib
import json
import os
import re
import subprocess

agent_dirs = ("/usr/local/bin", "/usr/bin", "/bin")

def on_agent_path(command):
    return any(os.path.exists(os.path.join(root, command)) for root in agent_dirs)

def imports(interpreter, modules):
    source = "; ".join(f"import {module}" for module in modules)
    return subprocess.run([interpreter, "-c", source]).returncode == 0

def digest(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()

def user_scrubbed(command):
    return subprocess.run(
        [
            "/usr/sbin/runuser",
            "-u",
            "user",
            "--",
            "env",
            "-i",
            "HOME=/home/user",
            "USER=user",
            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "bash",
            "-lc",
            command,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

def setup_session_which(command):
    return subprocess.run(
        [
            "bash",
            "-lc",
            f"PATH=/opt/env/venv/bin:/opt/env/bin:$PATH command -v {command}",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

def setup_session_python(command):
    return subprocess.run(
        [
            "bash",
            "-lc",
            "PATH=/opt/env/venv/bin:/opt/env/bin:$PATH python3 - <<'PY'\n"
            f"{command}\n"
            "PY",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

def script_lines(path):
    lines = []
    current = ""
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            current = f"{current} {line}".strip() if current else line
            if current.endswith("\\"):
                current = current[:-1].rstrip()
                continue
            lines.append(current)
            current = ""
    if current:
        lines.append(current)
    return lines

def script_execs(path, target):
    quoted = r"['\"]?" + re.escape(target) + r"['\"]?"
    pattern = re.compile(r"^exec\s+" + quoted + r"(?:\s|$)")
    return any(pattern.search(line) for line in script_lines(path))

def script_contains(path, needles):
    text = "\n".join(script_lines(path))
    return all(needle in text for needle in needles)

print("PARITY_JSON:" + json.dumps({
    "agent_python_version": subprocess.check_output(
        ["/usr/bin/python3", "--version"],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip(),
    "agent_imports_fitz": imports("/usr/bin/python3", ["fitz"]),
    "env_imports": imports(
        "/opt/env/venv/bin/python",
        [
            "fitz", "pandas", "playwright", "reportlab",
            "openpyxl", "docx", "pptx", "odf",
            "pip", "python_docx", "python_pptx",
        ],
    ),
    "setup_python3": setup_session_which("python3"),
    "setup_pip": setup_session_which("pip"),
    "setup_pip3": setup_session_which("pip3"),
    "setup_google_chrome": setup_session_which("google-chrome"),
    "setup_pip_user_clean": setup_session_python(
        "import json, pip\n"
        "print(json.dumps(pip._clean_install_args("
        "['--user', '--quiet', '--disable-pip-version-check', "
        "'--break-system-packages', '--root-user-action', 'ignore', "
        "'--no-cache-dir', '--constraint', 'constraints.txt', "
        "'--timeout', '40', '--retries=2', 'pip', 'openpyxl'])))"
    ),
    "setup_pip_main_version": setup_session_python(
        "import pip, sys\nsys.exit(pip.main(['--version']))"
    ) == "pip shim for /opt/env/venv",
    "setup_pip_target_detection": setup_session_python(
        "import json, pip\n"
        "print(json.dumps({\n"
        "'self_upgrade': pip._has_install_target("
        "pip._clean_install_args(['--quiet', '--upgrade', 'pip'])),\n"
        "'constraints_only': pip._has_install_target("
        "pip._clean_install_args(['--constraint', 'constraints.txt', 'pip'])),\n"
        "'requirements': pip._has_install_target("
        "pip._clean_install_args(['-r', 'requirements.txt'])),\n"
        "'package': pip._has_install_target(pip._clean_install_args(['openpyxl'])),\n"
        "}))"
    ),
    "uno_imports": imports(
        "/opt/env/uno-venv/bin/python", ["openpyxl", "docx", "odf", "uno"]
    ),
    "xcf_on_agent_path": on_agent_path("xcf2png") or on_agent_path("xcf2pnm"),
    "xcf_env_present": all(
        os.access(path, os.X_OK)
        for path in ("/opt/env/bin/xcf2png", "/opt/env/bin/xcf2pnm")
    ),
    "base_env_clis_off_agent_path": (
        not on_agent_path("jq") and not on_agent_path("xdotool")
    ),
    # The parent's own forbidden set, not a two-name sample of it. This image adds apt
    # RUNs AFTER the parent's hide step, and the parent warns that any later install can
    # re-register a hidden CLI on /usr/bin; an openjdk point upgrade dragged in by an
    # unrelated `apt-get install` did exactly that and put the whole java family back on
    # the agent PATH. Probing only jq/xdotool could not see it.
    "parent_forbidden_clis_leaked": sorted(
        name for name in __PARENT_CLI_FORBIDDEN__ if on_agent_path(name)
    ),
    "base_env_clis_present": all(
        os.access(path, os.X_OK)
        for path in ("/opt/env/bin/jq", "/opt/env/bin/xdotool")
    ),
    "agent_scrubbed_path": user_scrubbed("printf '%s' \"$PATH\""),
    "agent_scrubbed_chrome": user_scrubbed("command -v google-chrome || true"),
    "agent_scrubbed_jq": user_scrubbed("command -v jq || true"),
    "agent_scrubbed_xdotool": user_scrubbed("command -v xdotool || true"),
    "agent_task_clis_present": all(
        on_agent_path(command)
        for command in (
            "node", "npm", "rustc", "cargo",
            "ssh", "scp", "sftp", "ssh-keygen",
        )
    ),
    "base_chrome_present": os.access("/usr/local/bin/google-chrome", os.X_OK),
    "env_chrome_present": os.access("/opt/env/bin/google-chrome", os.X_OK),
    "chrome_wrappers_distinct": (
        digest("/usr/local/bin/google-chrome") != digest("/opt/env/bin/google-chrome")
    ),
    "env_chrome_forwarder_exec": script_execs(
        "/opt/env/bin/google-chrome",
        "/usr/local/bin/google-chrome",
    ),
    "base_chrome_execs_real_binary": script_execs(
        "/usr/local/bin/google-chrome",
        "/opt/google/chrome/chrome",
    ),
    "base_chrome_not_recursive": (
        not script_execs("/usr/local/bin/google-chrome", "/usr/local/bin/google-chrome")
        and not script_execs("/usr/local/bin/google-chrome", "/opt/env/bin/google-chrome")
    ),
    "base_chrome_cuagym_flags": script_contains(
        "/usr/local/bin/google-chrome",
        [
            "--remote-debugging-port=9222",
            "--start-maximized",
            "--user-data-dir=/tmp/chrome-profile",
            "$size_flags",
        ],
    ),
    "osworld_server_disabled": not os.path.exists(
        "/etc/supervisor/conf.d/50-osworld-server.conf"
    ),
}))
"""
    result = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "/usr/bin/python3", _IMAGE, "-c",
         snippet.replace("__PARENT_CLI_FORBIDDEN__", json.dumps(list(_PARENT_CLI_FORBIDDEN)))],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"image probe failed:\n{result.stdout}\n{result.stderr}"
    line = next(
        value for value in result.stdout.splitlines() if value.startswith("PARITY_JSON:")
    )
    return json.loads(line.removeprefix("PARITY_JSON:"))


def test_python_dependencies_stay_env_facing(probe: dict) -> None:
    assert probe["agent_python_version"].startswith("Python 3.10.")
    assert not probe["agent_imports_fitz"]
    assert probe["env_imports"]
    assert probe["setup_python3"] == "/opt/env/venv/bin/python3"
    assert probe["setup_pip"] == "/opt/env/venv/bin/pip"
    assert probe["setup_pip3"] == "/opt/env/venv/bin/pip3"
    assert json.loads(probe["setup_pip_user_clean"]) == [
        "--quiet",
        "--no-cache",
        "--constraints",
        "constraints.txt",
        "openpyxl",
    ]
    assert probe["setup_pip_main_version"]
    assert json.loads(probe["setup_pip_target_detection"]) == {
        "self_upgrade": False,
        "constraints_only": False,
        "requirements": True,
        "package": True,
    }
    assert probe["uno_imports"]


def test_reward_only_xcftools_stay_env_facing(probe: dict) -> None:
    assert not probe["xcf_on_agent_path"]
    assert probe["xcf_env_present"]


def test_osworld_base_env_clis_stay_env_facing(probe: dict) -> None:
    assert probe["base_env_clis_off_agent_path"]
    assert probe["base_env_clis_present"]
    assert probe["agent_scrubbed_path"] == (
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    )
    assert probe["agent_scrubbed_chrome"] == "/usr/local/bin/google-chrome"
    assert probe["agent_scrubbed_jq"] == ""
    assert probe["agent_scrubbed_xdotool"] == ""


def test_official_task_clis_stay_agent_facing(probe: dict) -> None:
    assert probe["agent_task_clis_present"]


def test_web_chrome_paths_share_cuagym_wrapper_semantics(probe: dict) -> None:
    assert probe["base_chrome_present"]
    assert probe["env_chrome_present"]
    assert probe["setup_google_chrome"] == "/opt/env/bin/google-chrome"
    assert probe["chrome_wrappers_distinct"]
    assert probe["env_chrome_forwarder_exec"]
    assert probe["base_chrome_execs_real_binary"]
    assert probe["base_chrome_not_recursive"]
    assert probe["base_chrome_cuagym_flags"]


def test_unused_osworld_http_server_is_disabled(probe: dict) -> None:
    assert probe["osworld_server_disabled"]


@pytest.fixture(scope="module")
def booted_desktop() -> dict[str, str]:
    if not _image_present():
        pytest.skip(f"{_IMAGE} is not built")

    name = f"lite-cuagym-parity-{uuid.uuid4().hex[:12]}"
    started = subprocess.run(
        [
            "docker",
            "run",
            "--detach",
            "--name",
            name,
            # match the env's configs/default.yaml (8GB); a GEGL/office op
            # peaks past 4g and the container is SIGKILLed mid-test.
            "--memory",
            "8g",
            "--cpus",
            "2",
            "-e",
            "VNC_RESOLUTION=1920x1080",
            _IMAGE,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert started.returncode == 0, started.stderr

    try:
        deadline = time.monotonic() + 180
        last = subprocess.CompletedProcess([], 1, "", "desktop probe did not run")
        while time.monotonic() < deadline:
            last = subprocess.run(
                [
                    "docker",
                    "exec",
                    name,
                    "bash",
                    "-lc",
                    (
                        "test -f /tmp/gnome-ready && "
                        "test -f /tmp/lite-cuagym-ready && "
                        "pgrep -u user -f gnome-shell >/dev/null"
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if last.returncode == 0:
                break
            time.sleep(2)
        else:
            logs = subprocess.run(
                ["docker", "logs", name],
                capture_output=True,
                text=True,
                timeout=30,
            )
            pytest.fail(
                "lite.cuagym desktop did not become ready:\n"
                f"{last.stdout}\n{last.stderr}\n{logs.stdout}\n{logs.stderr}"
            )

        server = subprocess.run(
            [
                "docker",
                "exec",
                name,
                "bash",
                "-lc",
                (
                    "pgrep -af '[s]tart-osworld-server|[d]esktop_env.server' || true; "
                    "ss -ltn '( sport = :5000 )' | tail -n +2"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        theme = subprocess.run(
            [
                "docker",
                "exec",
                name,
                "bash",
                "-lc",
                (
                    'bus="$(cat /tmp/dbus-session-bus-address)"; '
                    "runuser -u user -- env HOME=/home/user DISPLAY=:1 "
                    'DBUS_SESSION_BUS_ADDRESS="$bus" '
                    "gsettings get org.gnome.desktop.interface gtk-theme; "
                    "runuser -u user -- env HOME=/home/user DISPLAY=:1 "
                    'DBUS_SESSION_BUS_ADDRESS="$bus" '
                    "gsettings get org.gnome.desktop.interface color-scheme"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        theme_lines = theme.stdout.splitlines()
        yield {
            "gnome_ready": "true",
            "cuagym_settings_ready": "true",
            "osworld_server": server.stdout.strip(),
            "gtk_theme": theme_lines[0].strip() if theme_lines else "",
            "color_scheme": theme_lines[1].strip() if len(theme_lines) > 1 else "",
        }
    finally:
        subprocess.run(
            ["docker", "rm", "--force", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )


def test_child_image_boots_gnome_and_inherits_base_appearance(
    booted_desktop: dict[str, str],
) -> None:
    assert booted_desktop["gnome_ready"] == "true"
    assert booted_desktop["cuagym_settings_ready"] == "true"
    assert booted_desktop["gtk_theme"] == "'Yaru'"
    assert booted_desktop["color_scheme"] == "'default'"


def test_child_image_does_not_start_osworld_http_server(
    booted_desktop: dict[str, str],
) -> None:
    assert not booted_desktop["osworld_server"]


def test_parent_hidden_clis_stay_off_the_agent_path(probe):
    """No apt RUN in this image may re-register a CLI the parent hid.

    lite.osworld ends with a relocate+remove step and states it "MUST stay after every
    apt install, or a later install could re-register the CLI on /usr/bin and re-leak it
    onto the agent PATH". This image installs after that step, so it owns re-closing the
    door. It stayed shut by luck until a dependency upgrade re-ran the JRE's
    `update-alternatives --install`.
    """
    assert probe["parent_forbidden_clis_leaked"] == [], (
        "these CLIs are hidden in lite.osworld but visible to the agent here: "
        f"{probe['parent_forbidden_clis_leaked']}. An apt RUN after the parent's hide "
        "step re-registered them; re-run the /usr/lib/jvm symlink walk (or the "
        "relocate helper) after this image's LAST apt-touching RUN."
    )
