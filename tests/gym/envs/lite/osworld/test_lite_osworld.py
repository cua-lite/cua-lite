"""Tests for the production-facing lite.gym.envs.lite.osworld surface.

Dev validation helpers live under their owning ``devs/**/tests`` directories.
This file stays with the Lite OSWorld env implementation and groups only the
runtime, conversion, evaluator, and catalog contracts owned by that env.

Coverage groups:
- TestDispatchHelpers, TestDispatchActionTypes — dispatch.py wiring
- TestSetup                                    — setup.py
- TestConvertOSWorld                           — eval gen (per-domain ORACLES)
- TestEvalGetters, TestEvalMetricCalling       — eval-side helpers
- TestVerifyInit                               — verify package wiring
- TestInfeasibleHandling                       — agent-side terminate paths
- TestDataIntegrity                            — schema gate (eval.jsonl only)
- TestJsonlContract                            — split-uniform contract gates
                                                 (schema + byte-lock + det)
- TestEvalPull                                 — read_bytes RPC download

Usage:
    uv run pytest tests/gym/envs/lite/osworld/test_lite_osworld.py -n 0

Inside Slime container:
    pytest tests/gym/envs/lite/osworld/test_lite_osworld.py -n 0
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from lite.core.tools.calls import tool_call_arguments, tool_call_id, tool_call_name
from lite.core.tools.schemas import tool_schema_name
from lite.gym.envs.lite.osworld import exclude_reasons

#: Repo root. Everything this file reads off disk is anchored here — a CWD-relative
#: `Path("lite/gym/...")` silently resolves to nothing outside the repo root, which
#: turns a source-scanning gate into a `FileNotFoundError` (or, worse, a vacuous
#: pass) depending on where pytest was invoked from.
_REPO = Path(__file__).resolve().parents[5]

_DATA_DIR = _REPO / "lite/gym/envs/lite/osworld/data"

# The task catalogs are generated artifacts (gitignored, pinned by
# catalog.lock.json) — catalog-gated tests must skip on a fresh checkout.
_requires_catalogs = pytest.mark.skipif(
    not all(
        (_DATA_DIR / name).is_file()
        for name in ("eval.jsonl", "train.synth.jsonl", "train.perturb.jsonl")
    ),
    reason=(
        "osworld catalogs not generated — run "
        "`bash lite/gym/envs/lite/osworld/scripts/utils/tasks.sh generate` "
        "(or `bash lite/gym/envs/lite/osworld/scripts/install.sh provision`) first"
    ),
)


def _assert_tag_dropped_safely(script_text: str, tag_expr: str) -> None:
    """An install script must drop ``tag_expr`` before rebuilding it, and must never
    force it. ``docker image rm -f`` on an image a co-tenant's container references
    detaches that container from its image, so the shared ``image_rm`` helper
    (lite/gym/scripts/image_build.sh) is the only sanctioned removal. Asserted as the
    property — helper + tag — rather than a whole command string, so the next
    refactor of the surrounding redirection does not fail here.
    """
    assert f'image_rm "{tag_expr}"' in script_text, (
        f"{tag_expr} is never dropped before the rebuild, so a rebuild can ride the old label"
    )
    forced = re.findall(r"docker (?:image rm|rmi)\b[^\n]*\s-f\b", script_text)
    assert not forced, f"force-removed images orphan co-tenant containers: {forced}"


# =========================================================================
# Dispatch tests (unit — no Docker)
# =========================================================================


class TestDispatchHelpers:
    """Test dispatch.py helper functions."""

    def test_replace_templates_string(self):
        from lite.gym.envs.lite.osworld.src.utils.dispatch import _replace_templates

        # OSWorld VM user/password = "user" (not "cua" — that's the docker
        # container's `cua` group; the in-VM credential is "user").
        assert _replace_templates("{CLIENT_PASSWORD}") == "user"
        assert _replace_templates("{SCREEN_WIDTH}") == "1920"
        assert _replace_templates("{SCREEN_HEIGHT}") == "1080"
        assert _replace_templates("{SCREEN_WIDTH_HALF}") == "960"
        assert _replace_templates("{SCREEN_HEIGHT_HALF}") == "540"
        assert (
            _replace_templates("echo {CLIENT_PASSWORD} {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
            == "echo user 1920x1080"
        )

    def test_replace_templates_list(self):
        from lite.gym.envs.lite.osworld.src.utils.dispatch import _replace_templates

        result = _replace_templates(["python3", "-c", "print({SCREEN_WIDTH})"])
        assert result == ["python3", "-c", "print(1920)"]

    def test_replace_templates_passthrough(self):
        from lite.gym.envs.lite.osworld.src.utils.dispatch import _replace_templates

        assert _replace_templates(42) == 42
        assert _replace_templates("no templates here") == "no templates here"

    def test_dispatch_has_no_local_chord_splitter(self):
        # dispatch.py used to carry its own copy of core's _split_plus_chord on
        # the (wrong) premise that env code may not import the projector. The
        # owner is lite.core; a missed call site must surface as an ImportError.
        from lite.gym.envs.lite.osworld.src.utils import dispatch

        assert not hasattr(dispatch, "_split_plus_chord")

    def test_postconfig_chord_strings_route_through_the_core_projector(self):
        # postconfig / oracle chord strings are HAND-WRITTEN, so they are not
        # pre-normalized: normalize_keys splits supported chord strings while
        # preserving explicit literal plus keys, then to_xdotool resolves the
        # keysym. Every shipped config spelling must land byte-identical to what
        # the container's _norm_key produced before, and an unmappable token must
        # RAISE rather than ship silently. Leading-plus chords such as "+a" are
        # ambiguous with the chord separator and must be rejected.
        from lite.core.tools.action_space.keys import normalize_keys
        from lite.gym.utils.backend.model_inputs import project_model_keys

        def wire(chord: str) -> list[str]:
            return project_model_keys(normalize_keys(chord), action_name="key", backend="xdotool")

        assert wire("ctrl+s") == ["ctrl", "s"]
        assert wire("ctrl++") == ["ctrl", "plus"]  # literal plus glyph survives
        assert wire("+") == ["plus"]
        with pytest.raises(ValueError, match=r"invalid leading '\+' chord syntax"):
            wire("+a")
        assert wire("Return") == ["Return"]  # alias-folded, then projected
        assert wire("ctrl+End") == ["ctrl", "End"]
        assert wire("F6") == ["F6"]
        with pytest.raises(ValueError):
            wire("definitely_not_a_key")

    def test_postconfig_bad_keys_field_does_not_mask_valid_key_field(self):
        from lite.gym.envs.lite.osworld.src.gen.common import VS_CODE_SAVE_POSTCONFIG
        from lite.gym.envs.lite.osworld.src.gen.eval.postconfig import normalize_postconfig

        postconfig = [
            {
                "type": "activate_window",
                "parameters": {"window_name": "Visual Studio Code"},
            },
            {
                "type": "key",
                "parameters": {"keys": ["ctrl+s"], "key": "ctrl+s"},
            },
        ]

        assert normalize_postconfig(postconfig, "vs_code") == VS_CODE_SAVE_POSTCONFIG

    @pytest.mark.asyncio
    async def test_dispatch_key_field_is_source_chord_and_keys_field_is_canonical_list(self):
        from lite.gym.envs.lite.osworld.src.utils.dispatch import dispatch_action

        class _Interface:
            def __init__(self):
                self.hotkeys: list[tuple[str, ...]] = []

            async def hotkey(self, *keys):
                self.hotkeys.append(tuple(keys))

        computer = SimpleNamespace(interface=_Interface())

        await dispatch_action(computer, {"type": "key", "parameters": {"key": "ctrl++"}})
        await dispatch_action(computer, {"type": "key", "parameters": {"keys": ["ctrl", "+"]}})

        assert computer.interface.hotkeys == [("ctrl", "plus"), ("ctrl", "plus")]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("parameters", "match"),
        [
            ({"keys": "ctrl+s"}, "key.keys must be a list of strings, not a string"),
            ({"keys": ["ctrl+s"]}, "split chords into separate keys"),
            ({"keys": []}, "key.keys must not be empty"),
            ({"key": ""}, "empty or whitespace-only token"),
            ({}, "key.keys must not be empty"),
        ],
    )
    async def test_dispatch_rejects_malformed_key_payloads(self, parameters: dict, match: str):
        from lite.gym.envs.lite.osworld.src.utils.dispatch import dispatch_action

        computer = SimpleNamespace(interface=SimpleNamespace(hotkey=lambda *keys: None))

        with pytest.raises(ValueError, match=match):
            await dispatch_action(computer, {"type": "key", "parameters": parameters})

    def test_update_browse_history_command_bootstraps_chrome_schema(self):
        from lite.gym.envs.lite.osworld.src.utils.dispatch import _update_browse_history_command

        command = _update_browse_history_command(
            [
                {
                    "url": "https://www.bbc.co.uk/",
                    "title": "BBC",
                    "visit_time_from_now_in_seconds": 1500,
                }
            ]
        )

        assert "LITE_OSWORLD_HISTORY_ENTRIES=" in command
        assert '{"urls", "visits", "meta"} <= tables' in command
        assert "mktemp -d /tmp/lite-osworld-history-bootstrap.XXXXXX" in command
        assert 'google-chrome --user-data-dir="$bootstrap_dir"' in command
        assert "INSERT INTO urls" in command
        assert "DELETE FROM visits WHERE url IN" in command
        assert "killall -9 -q chrome" not in command
        assert "pkill" not in command
        # No post-hoc chown: the exec-stdio server runs AS the desktop user, so the
        # chrome-data this command seeds is born user-owned.
        assert "chown -R user:user /home/user/chrome-data" not in command

    def test_update_browse_history_seed_is_idempotent_with_duplicate_urls(
        self, tmp_path, monkeypatch
    ):
        import sqlite3

        from lite.gym.envs.lite.osworld.src.utils.dispatch import _update_browse_history_command

        entries = [
            {
                "url": "https://example.com/a",
                "title": "Example A",
                "visit_time_from_now_in_seconds": 60,
            },
            {
                "url": "https://example.com/a",
                "title": "Example A Again",
                "visit_time_from_now_in_seconds": 120,
            },
            {
                "url": "https://example.com/b",
                "title": "Example B",
                "visit_time_from_now_in_seconds": 180,
            },
        ]
        command = _update_browse_history_command(entries)
        script = command.split("fi\npython3 - <<'PY'\n", 1)[1].split("\nPY", 1)[0]
        hist = tmp_path / "chrome-data" / "Default" / "History"
        script = script.replace(
            'hist = "/home/user/chrome-data/Default/History"',
            f"hist = {str(hist)!r}",
        )
        monkeypatch.setenv("LITE_OSWORLD_HISTORY_ENTRIES", json.dumps(entries))

        exec(script, {})
        exec(script, {})

        conn = sqlite3.connect(hist)
        try:
            urls = dict(conn.execute("SELECT url, visit_count FROM urls").fetchall())
            visits = conn.execute("SELECT COUNT(*) FROM visits").fetchone()[0]
        finally:
            conn.close()

        assert urls == {"https://example.com/a": 2, "https://example.com/b": 1}
        assert visits == 3

    def test_compare_urls_basic(self):
        from lite.gym.envs.lite.osworld.src.utils.dispatch import _compare_urls

        assert _compare_urls("https://www.google.com", "https://google.com")
        assert _compare_urls("http://google.com/", "https://google.com")
        assert _compare_urls(
            "https://google.com/search?q=test", "http://www.google.com/search?q=test"
        )

    def test_compare_urls_different(self):
        from lite.gym.envs.lite.osworld.src.utils.dispatch import _compare_urls

        assert not _compare_urls("https://google.com", "https://bing.com")
        assert not _compare_urls("https://google.com/a", "https://google.com/b")

    def test_compare_urls_empty(self):
        from lite.gym.envs.lite.osworld.src.utils.dispatch import _compare_urls

        assert not _compare_urls("", "https://google.com")
        assert not _compare_urls("https://google.com", "")
        assert _compare_urls("", "")


class TestDispatchActionTypes:
    """Verify dispatch_action handles all expected types without error."""

    def test_all_flask_types_recognized(self):
        """execute, command, launch, open, activate_window, close_window should not warn."""
        from lite.gym.envs.lite.osworld.src.utils.dispatch import dispatch_action

        # Just verify the function exists and is async
        assert inspect.iscoroutinefunction(dispatch_action)

    def test_all_client_types_recognized(self):
        """download, sleep, chrome_open_tabs, chrome_close_tabs, update_browse_history."""
        from lite.gym.envs.lite.osworld.src.utils.dispatch import dispatch_action

        assert inspect.iscoroutinefunction(dispatch_action)

    def test_all_cua_types_recognized(self):
        """left_click, right_click, double_click, type_text, key, scroll, mouse_move."""
        from lite.gym.envs.lite.osworld.src.utils.dispatch import dispatch_action

        assert inspect.iscoroutinefunction(dispatch_action)

    def test_chrome_flags_injection(self):
        """Chrome launch should inject container CDP flags."""
        # This tests the logic path, not actual execution
        from lite.gym.envs.lite.osworld.src.utils.dispatch import CHROME_DATA_DIR

        cmd = ["google-chrome", "--remote-debugging-port=1337"]
        cmd_str = " ".join(cmd)
        # Simulate flag injection
        if "--no-sandbox" not in cmd:
            idx = next(i for i, c in enumerate(cmd) if "google-chrome" in c)
            cmd.insert(idx + 1, "--no-sandbox")
        if "--user-data-dir" not in cmd_str:
            cmd.append(f"--user-data-dir={CHROME_DATA_DIR}")
        if "--remote-allow-origins" not in cmd_str:
            cmd.append("--remote-allow-origins=*")
        assert "--no-sandbox" in cmd
        assert any("--user-data-dir" in c for c in cmd)
        assert "--remote-allow-origins=*" in cmd
        assert cmd[0] == "google-chrome"
        assert cmd[1] == "--no-sandbox"


# =========================================================================
# Setup tests
# =========================================================================


class TestSetup:
    """Tests for setup.py."""

    def test_setup_fn_is_async(self):
        from lite.gym.envs.lite.osworld.src.utils.setup import setup_fn

        assert inspect.iscoroutinefunction(setup_fn)

    def test_setup_via_commands_is_async(self):
        from lite.gym.envs.lite.osworld.src.utils.setup import _setup_via_commands

        assert inspect.iscoroutinefunction(_setup_via_commands)

    def test_timedatectl_shim_is_image_artifact(self):
        repo = Path(__file__).resolve().parents[5]
        shim = repo / "lite/gym/envs/lite/osworld/docker/bin/timedatectl"
        dockerfile = repo / "lite/gym/envs/lite/osworld/docker/Dockerfile"

        script = shim.read_text()
        docker = dockerfile.read_text()

        assert script.startswith("#!/usr/bin/env bash")
        assert "Time zone:" in script
        assert "NTP service:" in script
        assert "show)" in script
        assert "--property=NTP" in script
        assert "printf 'NTP=%s\\n' \"$(ntp_value)\"" in script
        assert "set-timezone)" in script
        assert "set-ntp)" in script
        assert ("COPY --chmod=0755 docker/bin/timedatectl /usr/local/bin/timedatectl") in docker
        # The three shims share one consolidated osworld sudoers drop-in.
        assert "/etc/sudoers.d/99-osworld-shims" in docker
        assert "NOPASSWD: /usr/local/bin/timedatectl" in docker
        assert "ENV TZ=" not in docker
        assert "ln -sf /usr/share/zoneinfo/Asia/Hong_Kong /etc/localtime" in docker
        assert "echo Asia/Hong_Kong > /etc/timezone" in docker

    def test_powerprofilesctl_shim_is_user_visible_image_artifact(self):
        repo = Path(__file__).resolve().parents[5]
        shim = repo / "lite/gym/envs/lite/osworld/docker/bin/powerprofilesctl"
        dockerfile = repo / "lite/gym/envs/lite/osworld/docker/Dockerfile"

        script = shim.read_text()
        docker = dockerfile.read_text()

        assert script.startswith("#!/usr/bin/env bash")
        assert "power-saver|balanced|performance" in script
        assert "read_profile" in script
        assert "set)" in script
        assert "list)" in script
        assert (
            "COPY --chmod=0755 docker/bin/powerprofilesctl /usr/local/bin/powerprofilesctl"
        ) in docker
        assert "/etc/sudoers.d/99-osworld-shims" in docker
        assert "NOPASSWD: /usr/local/bin/powerprofilesctl" in docker
        assert "/opt/env/bin/powerprofilesctl" not in docker

    def test_gsettings_large_text_shim_is_user_visible_image_artifact(self):
        repo = Path(__file__).resolve().parents[5]
        shim = repo / "lite/gym/envs/lite/osworld/docker/bin/gsettings"
        dockerfile = repo / "lite/gym/envs/lite/osworld/docker/Dockerfile"

        script = shim.read_text()
        docker = dockerfile.read_text()

        assert script.startswith("#!/usr/bin/env bash")
        assert "org.gnome.desktop.a11y.interface" in script
        assert "large-text" in script
        assert "text-scaling-factor" in script
        assert "gsettings_large_text" not in script
        assert "range)" in script
        assert "writable)" in script
        assert "list-recursively" in script
        assert 'exec "$REAL_GSETTINGS" "$@"' in script
        assert "bool=" not in script
        assert "sudo -n" not in script
        assert "COPY --chmod=0755 docker/bin/gsettings /usr/local/bin/gsettings" in docker
        assert "/etc/sudoers.d/99-user-gsettings" not in docker

    def test_hostname_shim_is_user_visible_image_artifact(self):
        repo = Path(__file__).resolve().parents[5]
        shim = repo / "lite/gym/envs/lite/osworld/docker/bin/hostname"
        dockerfile = repo / "lite/gym/envs/lite/osworld/docker/Dockerfile"

        script = shim.read_text()
        docker = dockerfile.read_text()

        assert script.startswith("#!/usr/bin/env bash")
        assert "REAL_HOSTNAME=/usr/bin/hostname" in script
        assert 'HOSTNAME_FILE="$STATE_DIR/hostname"' in script
        assert 'exec "$REAL_HOSTNAME" "$@"' in script
        assert 'printf \'%s\\n\' "$1" > "$HOSTNAME_FILE"' in script
        assert "COPY --chmod=0755 docker/bin/hostname /usr/local/bin/hostname" in docker
        assert "/etc/sudoers.d/99-osworld-shims" in docker
        assert "NOPASSWD: /usr/local/bin/hostname" in docker
        assert "/opt/env/bin/hostname" not in docker

    def test_osworld_image_artifacts_do_not_couple_to_scalecua(self):
        repo = Path(__file__).resolve().parents[5]
        dockerfile = repo / "lite/gym/envs/lite/osworld/docker/Dockerfile"
        docker = dockerfile.read_text()
        forbidden = ("scalecua", "scale-cua", "SCALE-CUA", "lite.scalecua", "rl_")
        artifacts = {
            "docker/bin/gsettings": "/usr/local/bin/gsettings",
            "docker/bin/hostname": "/usr/local/bin/hostname",
            "docker/bin/powerprofilesctl": "/usr/local/bin/powerprofilesctl",
            "docker/bin/timedatectl": "/usr/local/bin/timedatectl",
            "docker/desktop/bin/bwrap": "/usr/bin/bwrap",
            "docker/desktop/dconf/local.d/03-shell-dock": (
                "/etc/dconf/db/local.d/03-shell-dock"
            ),
            "docker/desktop/dconf/local.d/04-file-chooser": (
                "/etc/dconf/db/local.d/04-file-chooser"
            ),
            "docker/desktop/dconf/local.d/05-desktop-icons": (
                "/etc/dconf/db/local.d/05-desktop-icons"
            ),
            "docker/desktop/dconf/local.d/06-osworld-parity": (
                "/etc/dconf/db/local.d/06-osworld-parity"
            ),
            "docker/desktop/gtk/bookmarks": "/home/user/.config/gtk-3.0/bookmarks",
            "docker/fonts/conf.d/49-cjk-aliases.conf": "/etc/fonts/conf.d/49-cjk-aliases.conf",
            "docker/fonts/google-fonts.txt": "/tmp/osworld-google-fonts.txt",
            "docker/runtime/apply-osworld-settings.sh": "/usr/local/bin/apply-osworld-settings.sh",
            "docker/server/start-osworld-server.sh": "/usr/local/bin/start-osworld-server.sh",
            "docker/software/chrome/bin/google-chrome": "/usr/local/bin/google-chrome",
            "docker/software/libreoffice/registrymodifications.xcu": (
                "/usr/local/share/osworld/libreoffice/registrymodifications.xcu"
            ),
            "docker/software/mime/mimeapps.list": "/home/user/.config/mimeapps.list",
            "docker/software/thunderbird/policies.json": (
                "/usr/lib/thunderbird/distribution/policies.json"
            ),
            "docker/software/vscode/settings.json": "/home/user/.config/Code/User/settings.json",
            "docker/supervisor/36-apply-osworld-settings.conf": (
                "/etc/supervisor/conf.d/36-apply-osworld-settings.conf"
            ),
            "docker/supervisor/50-osworld-server.conf": (
                "/etc/supervisor/conf.d/50-osworld-server.conf"
            ),
        }
        executable_artifacts = {
            "docker/bin/gsettings",
            "docker/bin/hostname",
            "docker/bin/powerprofilesctl",
            "docker/bin/timedatectl",
            "docker/desktop/bin/bwrap",
            "docker/runtime/apply-osworld-settings.sh",
            "docker/server/start-osworld-server.sh",
            "docker/software/chrome/bin/google-chrome",
        }

        for rel, target in artifacts.items():
            text = (repo / "lite/gym/envs/lite/osworld" / rel).read_text()
            assert not any(token in text for token in forbidden), rel
            copy = (
                f"COPY --chmod=0755 {rel} {target}"
                if rel in executable_artifacts
                else f"COPY {rel} {target}"
            )
            assert copy in docker
            assert f"/opt/env/bin/{Path(rel).name}" not in docker

        assert "COPY --chmod=0755 docker/software/vscode/bin/code /usr/local/bin/code" in docker

        runtime = (
            repo / "lite/gym/envs/lite/osworld/docker/runtime/apply-osworld-settings.sh"
        ).read_text()
        libreoffice_profile = (
            repo
            / "lite/gym/envs/lite/osworld/docker/software/libreoffice/registrymodifications.xcu"
        ).read_text()
        # The Help-suppression keys now live solely in the baked, canonical xcu.
        # The runtime script no longer inlines them; it restores the baked profile
        # on every boot via cp (eval re-inits the LibreOffice profile).
        assert "cp -f /usr/local/share/osworld/libreoffice/registrymodifications.xcu" in runtime
        for key in ("HelpUserAssistance", "HelpAgentEnabled"):
            assert key in libreoffice_profile

        assert "cat > /usr/local/bin/start-osworld-server.sh" not in docker
        assert "cat > /usr/local/bin/google-chrome" not in docker
        assert "cat > /usr/bin/bwrap" not in docker
        assert "cat > /usr/local/bin/apply-osworld-settings.sh" not in docker
        assert "cat > /etc/fonts/conf.d/49-cjk-aliases.conf" not in docker
        assert "cat > /etc/dconf/db/local.d/" not in docker
        assert "cat > /etc/supervisor/conf.d/" not in docker
        assert "COPY docker/server/ /home/user/server/" not in docker
        assert "COPY docker/server/main.py /home/user/server/" in docker

    def test_vscode_settings_baseline_has_compat_keys(self):
        repo = Path(__file__).resolve().parents[5]
        settings_path = repo / "lite/gym/envs/lite/osworld/docker/software/vscode/settings.json"
        settings = json.loads(settings_path.read_text())

        # Subset, not equality: these keys are the VM-parity contract (6c662b0f
        # merged the VM's 3 workspace-trust keys so lite is not noisier than the
        # VM on the trust prompt). An unrelated harmless key added later is not
        # a regression — dropping one of these, or flipping its value, is.
        compat_keys = {
            "workbench.startupEditor": "none",
            "telemetry.telemetryLevel": "off",
            "update.mode": "none",
            "extensions.autoCheckUpdates": False,
            "chat.commandCenter.enabled": False,
            "workbench.welcomePage.walkthroughs.openOnInstall": False,
            "security.workspace.trust.enabled": False,
            "security.workspace.trust.startupPrompt": "never",
            "security.workspace.trust.emptyWindow": False,
        }
        assert compat_keys.items() <= settings.items(), (
            "vscode settings.json lost a compat key/value: "
            f"{sorted(set(compat_keys.items()) - set(settings.items()))}"
        )

    def test_vscode_wrapper_scrubs_backend_path_and_covers_absolute_code(self):
        repo = Path(__file__).resolve().parents[5]
        dockerfile = repo / "lite/gym/envs/lite/osworld/docker/Dockerfile"
        wrapper = repo / "lite/gym/envs/lite/osworld/docker/software/vscode/bin/code"
        docker = dockerfile.read_text()
        script = wrapper.read_text()

        assert "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" in script
        assert 'exec /usr/share/code/bin/code --no-sandbox "$@"' in script
        assert ("COPY --chmod=0755 docker/software/vscode/bin/code /usr/local/bin/code") in docker
        assert "rm -f /usr/bin/code" in docker
        assert "install -m 0755 /usr/local/bin/code /usr/bin/code" in docker
        assert "s#Exec=/usr/share/code/code#Exec=/usr/bin/code#" in docker

    def test_install_script_uses_latest_image_tags(self):
        repo = Path(__file__).resolve().parents[5]
        script = repo / "lite/gym/envs/lite/osworld/scripts/install.sh"
        dockerfile = repo / "lite/gym/envs/lite/osworld/docker/Dockerfile"

        text = script.read_text()
        docker = dockerfile.read_text()

        assert 'BASE_TAG="cua-lite/sandbox.linux:latest"' in text
        assert 'ADDITIVE_TAG="cua-lite/lite.osworld:latest"' in text
        assert 'IMAGE="$ADDITIVE_TAG"' in text
        assert 'REPO_ROOT="$(cd "$OSWORLD_DIR/../../../../.." && pwd)"' in text
        assert '"$REPO_ROOT[gym]"' in text
        assert "guard_non_latest_images" not in text
        assert "LITE_REQUIRE_NON_LATEST_IMAGES" not in text
        assert "LITE_SCALECUA_VALIDATION" not in text
        assert "LITE_OSWORLD_BASE_IMAGE" not in text
        assert "LITE_SANDBOX_IMAGE" not in text
        assert "LITE_OSWORLD_IMAGE" not in text
        assert '--build-arg "BASE_IMAGE=$BASE_IMAGE"' not in text
        _assert_tag_dropped_safely(text, "$ADDITIVE_TAG")
        assert "ARG BASE_TAG=latest" in docker
        assert "FROM cua-lite/sandbox.linux:${BASE_TAG}" in docker
        assert "ARG BASE_IMAGE" not in docker

    def test_evaluator_dependencies_stay_env_facing_without_pip_wrapper(self):
        repo = Path(__file__).resolve().parents[5]
        dockerfile = repo / "lite/gym/envs/lite/osworld/docker/Dockerfile"
        pip_wrapper = repo / "lite/gym/envs/lite/osworld/docker/bin/pip"

        docker = dockerfile.read_text()

        assert "/opt/env/bin/uv pip install --python /opt/env/venv/bin/python" in docker
        assert "pysrt==1.1.2" in docker
        assert not pip_wrapper.exists()
        assert "COPY docker/bin/pip" not in docker
        assert "/usr/local/bin/pip" not in docker

    def test_sandbox_install_script_uses_latest_image_tag(self):
        repo = Path(__file__).resolve().parents[5]
        script = repo / "lite/gym/sandbox/scripts/install.sh"

        text = script.read_text()

        assert 'IMAGE="cua-lite/sandbox.linux:latest"' in text
        assert "LITE_SANDBOX_IMAGE" not in text
        assert "LITE_SCALECUA_VALIDATION" not in text
        assert "non-latest" not in text
        _assert_tag_dropped_safely(text, "$IMAGE")

    @pytest.mark.asyncio
    async def test_setup_fn_waits_for_flask_before_task_setup(self):
        from lite.gym.envs.lite.osworld.src.utils.setup import setup_fn

        class _Interface:
            def __init__(self):
                self.commands: list[str] = []

            async def run_command(self, command, timeout=None):
                self.commands.append(command)
                if "http://localhost:5000/setup/execute" in command:
                    return SimpleNamespace(stdout="200", returncode=0)
                return SimpleNamespace(stdout="", returncode=0)

        computer = SimpleNamespace(interface=_Interface())
        task = SimpleNamespace(metadata={})

        await setup_fn(task, computer)

        commands = computer.interface.commands
        assert "http://localhost:5000/setup/execute" in commands[0]
        # B1 removed the `sudo pkill` sudoers-grant block (no-sudo isolation
        # contract). The revert-to-default-user refactor also DELETED the
        # `.setup-marker` touch and the git `safe.directory` run_command — the
        # server runs as the desktop user, so there is nothing to pre-arrange.
        assert not any("99-user-pkill" in c for c in commands)
        assert not any(".setup-marker" in c for c in commands)
        assert not any("safe.directory" in c for c in commands)
        # The LibreOffice first-run info-bar suppression (setup.xcu write) must
        # run, and must run AFTER the flask-wait — asserted by ordering, not by
        # index, so inserting another setup command is not a false regression.
        flask_idxs = [
            i for i, c in enumerate(commands) if "http://localhost:5000/setup/execute" in c
        ]
        xcu_idxs = [i for i, c in enumerate(commands) if "setup.xcu" in c]
        assert len(xcu_idxs) == 1, f"expected one setup.xcu write, got {xcu_idxs}"
        assert xcu_idxs[0] > flask_idxs[-1], (
            f"setup.xcu write at {xcu_idxs[0]} precedes the flask-wait {flask_idxs}"
        )


# =========================================================================
# ConvertOSWorld tests
# =========================================================================


class TestConvertOSWorld:
    """Tests for generate/eval (OSWorld task conversion)."""

    def test_rewrite(self):
        from lite.gym.envs.lite.osworld.src.gen.eval.__main__ import _rewrite

        # /home/user/* paths pass through unchanged (the OSWorld VM user is `user`).
        assert _rewrite("/home/user/Desktop") == "/home/user/Desktop"
        assert _rewrite("/tmp/file") == "/tmp/file"
        # {CLIENT_PASSWORD} maps to the OSWorld VM user/password = "user".
        assert _rewrite("{CLIENT_PASSWORD}") == "user"
        assert _rewrite("{SCREEN_WIDTH}") == "1920"
        assert _rewrite("{SCREEN_HEIGHT}") == "1080"
        assert _rewrite("{SCREEN_WIDTH_HALF}") == "960"
        assert _rewrite("{SCREEN_HEIGHT_HALF}") == "540"

    def test_rewrite_nested(self):
        from lite.gym.envs.lite.osworld.src.gen.eval.__main__ import _rewrite

        data = {
            "path": "/home/user/doc.txt",
            "files": ["/home/user/a.xlsx"],
            "cred": "{CLIENT_PASSWORD}",
        }
        result = _rewrite(data)
        assert result["path"] == "/home/user/doc.txt"
        assert result["files"][0] == "/home/user/a.xlsx"
        assert result["cred"] == "user"

    def test_rewrite_passthrough_non_strings(self):
        from lite.gym.envs.lite.osworld.src.gen.eval.__main__ import _rewrite

        assert _rewrite(42) == 42
        assert _rewrite(None) is None
        assert _rewrite(True) is True
        assert _rewrite([1, 2, "no_template"]) == [1, 2, "no_template"]

    def test_convert_task_infeasible(self):
        from lite.gym.envs.lite.osworld.src.gen.eval.__main__ import convert_task

        # `infeasible` is DERIVED from the upstream evaluator, not a hardcoded list:
        # evaluator.func == "infeasible" ⇒ exclude_reason "infeasible".
        task = {
            "id": "u-1",
            "instruction": "do X",
            "evaluator": {"func": "infeasible"},
            "config": [],
        }
        result = convert_task(task, "os", {})
        assert result["metadata"]["others"]["exclude_reason"] == "infeasible"
        # "infeasible" in a func LIST also counts.
        task_list = {
            "id": "u-2",
            "instruction": "do X",
            "evaluator": {"func": ["check", "infeasible"]},
            "config": [],
        }
        assert (
            convert_task(task_list, "os", {})["metadata"]["others"]["exclude_reason"]
            == "infeasible"
        )
        # A real evaluator func is NOT infeasible (env-difference tasks stay feasible).
        task_ok = {
            "id": "u-3",
            "instruction": "do Y",
            "evaluator": {"func": "check_list"},
            "config": [],
        }
        assert not convert_task(task_ok, "os", {})["metadata"]["others"].get("exclude_reason")

    def test_convert_task_googledrive(self):
        from lite.gym.envs.lite.osworld.src.gen.eval.__main__ import (
            _GOOGLE_AUTH_TASK_IDS,
            convert_task,
        )

        oid = next(iter(_GOOGLE_AUTH_TASK_IDS))
        task = {
            "id": oid,
            "instruction": "upload to drive",
            "config": [{"type": "googledrive", "parameters": {}}],
            "evaluator": {"func": "check"},
        }
        result = convert_task(task, "multi_apps", {})
        assert result["metadata"]["others"]["exclude_reason"] == "google_auth"

    def test_convert_task_basic(self):
        from lite.gym.envs.lite.osworld.src.gen.eval.__main__ import convert_task

        task = {
            "id": "abc12345678",
            "instruction": "do something",
            "config": [{"type": "execute", "parameters": {"command": "echo test"}}],
            "evaluator": {"func": "exact_match", "result": {}, "expected": {}},
        }
        result = convert_task(task, "os", {})
        assert result["task_id"] == "osworld_os_abc12345"
        assert result["metadata"]["others"]["domain"] == "os"
        assert result["metadata"]["others"]["oracle_actions"] == []
        assert "oracle_verified" not in result["metadata"]["others"]
        assert "oracle_actions" not in result["metadata"]
        assert "oracle_after_postconfig" not in result["metadata"]

    def test_convert_task_with_oracle_actions(self):
        """ORACLES['actions'] populates metadata.others oracle fields.

        ``oracle_actions`` is the only fact: there is no derived
        ``oracle_verified`` duplicate, which only this generator ever wrote.
        """
        from lite.gym.envs.lite.osworld.src.gen.eval.__main__ import convert_task

        task = {
            "id": "abc12345678",
            "instruction": "do something",
            "config": [],
            "evaluator": {"func": "exact_match"},
        }
        oracles = {
            "abc12345678": {
                "actions": [{"type": "execute", "parameters": {"command": "ls"}}],
                "after_postconfig": True,
            }
        }
        result = convert_task(task, "os", oracles)
        assert result["metadata"]["others"]["oracle_actions"] == oracles["abc12345678"]["actions"]
        assert "oracle_verified" not in result["metadata"]["others"]
        assert result["metadata"]["others"]["oracle_after_postconfig"] is True

    def test_convert_task_with_evaluator_override(self):
        """ORACLES['evaluator'] replaces the upstream evaluator after _rewrite."""
        from lite.gym.envs.lite.osworld.src.gen.eval.__main__ import convert_task

        task = {
            "id": "abc12345678",
            "instruction": "do something",
            "config": [],
            "evaluator": {
                "func": "exact_match",
                "result": {"type": "vm_file", "path": "/upstream"},
            },
        }
        override = {"func": "diff_text_file", "result": {"type": "vm_file", "path": "/curated"}}
        oracles = {"abc12345678": {"evaluator": override}}
        result = convert_task(task, "os", oracles)
        assert result["metadata"]["evaluator"] == override

    def test_convert_task_with_canonical_exclude_reason(self):
        """ORACLES['exclude_reason'] must be canonical before it reaches catalogs."""
        from lite.gym.envs.lite.osworld.src.gen.eval.__main__ import convert_task

        task = {
            "id": "abc12345678",
            "instruction": "do something",
            "config": [],
            "evaluator": {"func": "exact_match"},
        }
        reason = "upstream_live_site_drift"
        oracles = {"abc12345678": {"exclude_reason": reason}}
        result = convert_task(task, "os", oracles)
        assert result["metadata"]["others"]["exclude_reason"] == reason

        bad_oracles = {"abc12345678": {"exclude_reason": "block: this is hand-curated"}}
        with pytest.raises(ValueError):
            convert_task(task, "os", bad_oracles)

    def test_multi_apps_2c1ebcd7_excluded_for_upstream_eval_bug(self):
        """2c1ebcd7 is excluded because the generated eval row can score a no-op."""
        from lite.gym.envs.lite.osworld.src.gen.eval.multi_apps import ORACLES

        entry = ORACLES["2c1ebcd7-9c6d-4c9a-afad-900e381ecd5e"]
        assert entry["exclude_reason"] == "upstream_generated_eval_bug"

    def test_load_oracles(self):
        """_load_oracles imports the per-domain ORACLES dict by name."""
        from lite.gym.envs.lite.osworld.src.gen.eval.__main__ import _load_oracles

        oracles = _load_oracles("chrome")
        assert isinstance(oracles, dict) and len(oracles) > 0
        # Every key must be a 36-char OSWorld UUID
        for tid in oracles:
            assert len(tid) == 36 and tid.count("-") == 4


# =========================================================================
# Eval pull tests (unit — no Docker)
# =========================================================================
#
# Tests for _download_from_container.
# Uses computer.interface.read_bytes RPC — symmetric with synth_push's
# write_bytes, no Flask :5000 host mapping needed.


class TestEvalPull:
    """Unit tests for _download_from_container over read_bytes RPC."""

    @pytest.fixture
    def tmp_cache(self, tmp_path):
        return str(tmp_path)

    @pytest.fixture
    def fake_computer_ok(self):
        """Container handle whose read_bytes returns file content."""

        class _Iface:
            async def read_bytes(self, path):
                return b"file-content"

            async def run_command(self, cmd):
                class _R:
                    stdout = ""

                return _R()

        class _Comp:
            interface = _Iface()

        return _Comp()

    @pytest.fixture
    def fake_computer_read_bytes_fail(self):
        """Container handle whose read_bytes raises; base64 fallback returns content."""

        class _R:
            import base64 as _b64

            stdout = _b64.b64encode(b"fallback-content").decode()

        class _Iface:
            async def read_bytes(self, path):
                raise RuntimeError("not implemented")

            async def run_command(self, cmd):
                return _R()

        class _Comp:
            interface = _Iface()

        return _Comp()

    @pytest.fixture
    def fake_computer_both_fail(self):
        """Container handle where read_bytes raises and run_command returns empty."""

        class _R:
            stdout = ""

        class _Iface:
            async def read_bytes(self, path):
                raise RuntimeError("not implemented")

            async def run_command(self, cmd):
                return _R()

        class _Comp:
            interface = _Iface()

        return _Comp()

    @pytest.mark.asyncio
    async def test_read_bytes_primary_path(self, fake_computer_ok, tmp_cache):
        """Happy path: read_bytes returns bytes → file written, path returned."""
        from lite.gym.envs.lite.osworld.src.eval.runner import (
            _download_from_container,
        )

        result = await _download_from_container(
            fake_computer_ok, "/home/user/Desktop/report.pdf", tmp_cache
        )
        assert result is not None
        assert result.endswith("report.pdf")
        assert Path(result).read_bytes() == b"file-content"

    @pytest.mark.asyncio
    async def test_read_bytes_empty_path_returns_none(self, fake_computer_ok, tmp_cache):
        """Empty remote_path → None without touching the interface."""
        from lite.gym.envs.lite.osworld.src.eval.runner import (
            _download_from_container,
        )

        result = await _download_from_container(fake_computer_ok, "", tmp_cache)
        assert result is None

    @pytest.mark.asyncio
    async def test_base64_fallback_when_read_bytes_fails(
        self, fake_computer_read_bytes_fail, tmp_cache
    ):
        """read_bytes raises → fallback base64 via run_command succeeds."""
        from lite.gym.envs.lite.osworld.src.eval.runner import (
            _download_from_container,
        )

        result = await _download_from_container(
            fake_computer_read_bytes_fail, "/tmp/some.xlsx", tmp_cache
        )
        assert result is not None
        assert Path(result).read_bytes() == b"fallback-content"

    @pytest.mark.asyncio
    async def test_returns_none_when_both_paths_fail(self, fake_computer_both_fail, tmp_cache):
        """Both read_bytes and base64 fail → None returned (no exception raised)."""
        from lite.gym.envs.lite.osworld.src.eval.runner import (
            _download_from_container,
        )

        result = await _download_from_container(
            fake_computer_both_fail, "/tmp/missing.xlsx", tmp_cache
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_pre_cmd_executed_before_read(self, tmp_cache):
        """pre_cmd is run before the read_bytes call."""
        calls: list[str] = []

        class _R:
            stdout = ""

        class _Iface:
            async def run_command(self, cmd):
                calls.append(cmd)
                return _R()

            async def read_bytes(self, path):
                calls.append(f"read:{path}")
                return b"data"

        class _Comp:
            interface = _Iface()

        from lite.gym.envs.lite.osworld.src.eval.runner import (
            _download_from_container,
        )

        await _download_from_container(
            _Comp(),
            "/tmp/wallpaper.png",
            tmp_cache,
            pre_cmd="curl -s http://localhost:5000/wallpaper -o /tmp/wallpaper.png",
        )
        assert calls[0].startswith("curl")
        assert calls[1] == "read:/tmp/wallpaper.png"

    @pytest.mark.asyncio
    async def test_container_download_empty_basename_rejected_before_write(
        self, fake_computer_ok, tmp_cache
    ):
        from lite.gym.envs.lite.osworld.src.eval.runner import (
            _download_from_container,
        )

        with pytest.raises(ValueError, match="download filename"):
            await _download_from_container(fake_computer_ok, "/tmp/", tmp_cache)

    def test_download_url_rejects_dest_outside_cache(self, tmp_path):
        from lite.gym.envs.lite.osworld.src.eval.runner import _download_url

        for dest in ("../dst", "/tmp/dst"):
            with pytest.raises(ValueError, match="cache_dir"):
                _download_url("https://example.test/file.txt", str(tmp_path), dest=dest)
        assert not (tmp_path.parent / "dst").exists()

    def test_download_url_allows_nested_dest_inside_cache(self, monkeypatch, tmp_path):
        from lite.gym.envs.lite.osworld.src.eval import runner

        def fake_urlretrieve(_url, filename):
            Path(filename).write_bytes(b"payload")

        monkeypatch.setattr(runner.urllib.request, "urlretrieve", fake_urlretrieve)

        result = runner._download_url(
            "https://example.test/file.txt",
            str(tmp_path),
            dest="nested/file.txt",
        )

        assert result == str(tmp_path / "nested/file.txt")
        assert (tmp_path / "nested/file.txt").read_bytes() == b"payload"


# =========================================================================
# Eval getter tests (unit — no Docker)
# =========================================================================


class TestEvalGetters:
    """Tests for osworld_eval.py getter/extractor logic."""

    @pytest.mark.asyncio
    async def test_vlc_playing_info_tries_official_password(self, tmp_path):
        from lite.gym.envs.lite.osworld.src.eval.runner import _get_result

        class _Interface:
            def __init__(self):
                self.commands = []

            async def run_command(self, command, timeout=None):
                self.commands.append(command)
                if "--user :password" in command:
                    return SimpleNamespace(stdout="<root><state>playing</state></root>")
                return SimpleNamespace(stdout="")

        computer = SimpleNamespace(interface=_Interface())

        out = await _get_result(
            computer,
            {"type": "vlc_playing_info", "dest": "status.xml"},
            str(tmp_path),
        )

        assert out == str(tmp_path / "status.xml")
        assert (tmp_path / "status.xml").read_text() == "<root><state>playing</state></root>"
        assert computer.interface.commands == [
            "curl -s --user :password http://localhost:8080/requests/status.xml",
        ]

    def test_extract_chrome_pref_enable_do_not_track(self):
        from lite.gym.envs.lite.osworld.src.eval.runner import _extract_chrome_pref

        assert _extract_chrome_pref("enable_do_not_track", {"enable_do_not_track": True}) == "true"
        assert (
            _extract_chrome_pref("enable_do_not_track", {"enable_do_not_track": False}) == "false"
        )
        assert _extract_chrome_pref("enable_do_not_track", {}) == "false"

    def test_extract_chrome_pref_enable_safe_browsing(self):
        from lite.gym.envs.lite.osworld.src.eval.runner import _extract_chrome_pref

        # enhanced OR enabled
        assert (
            _extract_chrome_pref("enable_safe_browsing", {"safebrowsing": {"enhanced": True}})
            == "true"
        )
        assert (
            _extract_chrome_pref("enable_safe_browsing", {"safebrowsing": {"enabled": True}})
            == "true"
        )
        assert (
            _extract_chrome_pref(
                "enable_safe_browsing", {"safebrowsing": {"enabled": False, "enhanced": False}}
            )
            == "false"
        )
        assert _extract_chrome_pref("enable_safe_browsing", {}) == "false"

    def test_extract_chrome_pref_data_delete(self):
        from lite.gym.envs.lite.osworld.src.eval.runner import _extract_chrome_pref

        # The real Chrome "delete data on close" UI writes
        # profile.default_content_setting_values.cookies = 4 (clear-on-exit),
        # live-confirmed on Chrome 149. The old browsing_data_lifetime
        # key is an enterprise policy the UI never writes.
        assert (
            _extract_chrome_pref(
                "data_delete_automacally",
                {"profile": {"default_content_setting_values": {"cookies": 4}}},
            )
            == "true"
        )
        # any non-4 value (or absent) → "false"
        assert (
            _extract_chrome_pref(
                "data_delete_automacally",
                {"profile": {"default_content_setting_values": {"cookies": 1}}},
            )
            == "false"
        )
        assert _extract_chrome_pref("data_delete_automacally", {}) == "false"
        # the old enterprise-policy key alone must NOT pass
        assert (
            _extract_chrome_pref(
                "data_delete_automacally",
                {"browser": {"clear_data": {"browsing_data_lifetime": {"enabled": True}}}},
            )
            == "false"
        )

    def test_extract_chrome_pref_new_startup_page(self):
        from lite.gym.envs.lite.osworld.src.eval.runner import _extract_chrome_pref

        # restore_on_startup == 5 → "true" (positive proof; OSWorld semantics).
        assert (
            _extract_chrome_pref("new_startup_page", {"session": {"restore_on_startup": 5}})
            == "true"
        )
        # No session key → "false" (closes a loophole that previously
        # auto-passed composite Chrome+X tasks where Chrome never launched).
        assert _extract_chrome_pref("new_startup_page", {}) == "false"
        # restore_on_startup != 5 → "false"
        assert (
            _extract_chrome_pref("new_startup_page", {"session": {"restore_on_startup": 1}})
            == "false"
        )

    def test_extract_chrome_pref_default_search_engine(self):
        from lite.gym.envs.lite.osworld.src.eval.runner import _extract_chrome_pref

        prefs = {"default_search_provider_data": {"template_url_data": {"short_name": "Bing"}}}
        assert _extract_chrome_pref("default_search_engine", prefs) == "Bing"
        # Fallback to keyword
        prefs2 = {
            "default_search_provider_data": {"template_url_data": {"keyword": "duckduckgo.com"}}
        }
        assert _extract_chrome_pref("default_search_engine", prefs2) == "duckduckgo.com"
        # Default
        assert _extract_chrome_pref("default_search_engine", {}) == "Google"

    def test_extract_chrome_pref_font_size(self):
        from lite.gym.envs.lite.osworld.src.eval.runner import _extract_chrome_pref

        # Returns full webkit.webprefs dict (matches OSWorld)
        prefs = {
            "webkit": {
                "webprefs": {
                    "default_font_size": 24,
                    "default_fixed_font_size": 14,
                    "minimum_font_size": 10,
                }
            }
        }
        result = _extract_chrome_pref("chrome_font_size", prefs)
        assert isinstance(result, dict)
        assert result["default_font_size"] == 24
        assert result["default_fixed_font_size"] == 14
        # Default when missing
        result2 = _extract_chrome_pref("chrome_font_size", {})
        assert result2["default_font_size"] == 16

    def test_extract_chrome_pref_profile_name(self):
        from lite.gym.envs.lite.osworld.src.eval.runner import _extract_chrome_pref

        assert _extract_chrome_pref("profile_name", {"profile": {"name": "Work"}}) == "Work"
        assert _extract_chrome_pref("profile_name", {}) == ""

    def test_extract_chrome_pref_color_scheme(self):
        from lite.gym.envs.lite.osworld.src.eval.runner import _extract_chrome_pref

        # color_scheme integer
        assert (
            _extract_chrome_pref("chrome_color_scheme", {"browser": {"theme": {"color_scheme": 1}}})
            == "light"
        )
        assert (
            _extract_chrome_pref("chrome_color_scheme", {"browser": {"theme": {"color_scheme": 2}}})
            == "dark"
        )
        assert (
            _extract_chrome_pref("chrome_color_scheme", {"browser": {"theme": {"color_scheme": 0}}})
            == "system"
        )
        # #155 chrome_93eabf48: canonical `color_scheme` takes priority; `color_scheme2`
        # is a stale mirror left by perturbation setup, used ONLY as a fallback when
        # canonical is absent. (Old behavior read the mirror first -> false-negative
        # when the UI wrote canonical=light but left a stale mirror=dark.)
        assert (
            _extract_chrome_pref(
                "chrome_color_scheme",
                {"browser": {"theme": {"color_scheme": 0, "color_scheme2": 2}}},
            )
            == "system"
        )
        # fallback: only the mirror present -> read it (no regression for older writes)
        assert (
            _extract_chrome_pref(
                "chrome_color_scheme", {"browser": {"theme": {"color_scheme2": 1}}}
            )
            == "light"
        )
        # system_theme fallback
        assert (
            _extract_chrome_pref(
                "chrome_color_scheme", {"extensions": {"theme": {"system_theme": 0}}}
            )
            == "light"
        )
        # Default
        assert _extract_chrome_pref("chrome_color_scheme", {}) == "system"

    def test_extract_chrome_pref_dns_prefetch(self):
        from lite.gym.envs.lite.osworld.src.eval.runner import _extract_chrome_pref

        assert (
            _extract_chrome_pref("check_dns_prefetch", {"dns_prefetching": {"enabled": True}})
            == "true"
        )
        assert (
            _extract_chrome_pref("check_dns_prefetch", {"dns_prefetching": {"enabled": False}})
            == "false"
        )

    @pytest.mark.asyncio
    async def test_get_expected_rule(self):
        from lite.gym.envs.lite.osworld.src.eval.runner import _get_expected

        result = await _get_expected(None, {"type": "rule", "rules": {"expected": "true"}}, "/tmp")
        assert result == {"expected": "true"}

    @pytest.mark.asyncio
    async def test_get_expected_list(self):
        from lite.gym.envs.lite.osworld.src.eval.runner import _get_expected

        result = await _get_expected(None, {"type": "list", "list": [1, 2, 3]}, "/tmp")
        assert result == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_get_expected_empty(self):
        from lite.gym.envs.lite.osworld.src.eval.runner import _get_expected

        assert await _get_expected(None, {}, "/tmp") is None
        assert await _get_expected(None, None, "/tmp") is None

    @pytest.mark.asyncio
    async def test_get_expected_vm_file_does_not_double_normalize(self, monkeypatch, tmp_path):
        from lite.gym.envs.lite.osworld.src.eval import runner

        normalized = []

        async def fake_get_result(computer, config, cache_dir):
            assert config["type"] == "vm_file"
            return str(tmp_path / "expected.pptx")

        async def fake_normalize(computer, local_path):
            normalized.append(local_path)

        monkeypatch.setattr(runner, "_get_result", fake_get_result)
        monkeypatch.setattr(runner, "_lo_normalize_pptx", fake_normalize)

        out = await runner._get_expected(
            SimpleNamespace(),
            {"type": "vm_file", "path": "/tmp/expected.pptx"},
            str(tmp_path),
        )

        assert out == str(tmp_path / "expected.pptx")
        assert normalized == []

    def test_ensure_csv_exports(self):
        """CSV export from xlsx should create per-sheet CSV files."""
        import os
        import tempfile

        from lite.gym.envs.lite.osworld.src.eval.runner import _ensure_csv_exports

        try:
            import openpyxl
        except ImportError:
            pytest.skip("openpyxl not installed")
        # Create a minimal xlsx
        cache = tempfile.mkdtemp()
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws.append(["A", "B"])
        ws.append([1, 2])
        xlsx_path = os.path.join(cache, "test.xlsx")
        wb.save(xlsx_path)
        wb.close()
        # Generate CSV
        _ensure_csv_exports(xlsx_path, cache)
        csv_path = os.path.join(cache, "test-Sheet1.csv")
        assert os.path.exists(csv_path)
        with open(csv_path) as f:
            content = f.read()
        assert "A,B" in content
        assert "1,2" in content

    def test_vlc_http_password_is_canonical_not_split(self):
        """PREVENTIVE (generalization guard): every VLC HTTP-interface writer must
        seed the password the getter authenticates with. The upstream getter
        desktop_env/evaluators/getters/vlc.py hardcodes ``password = 'password'``
        (no fallback -> 401 -> reward 0); the scalecua override
        ``_get_vlc_playing_info`` tries 'password' before 'vlc'. So 'password'
        works for EVERY getter and 'vlc' 401s the upstream one. A writer seeding
        'vlc' false-fails a correct trajectory ONLY in a real rollout (oracle
        replay pre-seeds its own matching password, masking it) -- the exact
        class this guards.
        """
        root = Path(__file__).resolve().parents[5] / "lite/gym/envs/lite/osworld/src"
        writers = [
            root / "utils/dispatch.py",
            root / "gen/train/synth/vlc.py",
            root / "gen/train/perturb/vlc.py",
            root / "gen/eval/vlc.py",
        ]
        for f in writers:
            text = f.read_text()
            assert "http-password=vlc" not in text, f"{f.name}: vlcrc seeds vlc-only password"
            assert "--http-password vlc " not in text, (
                f"{f.name}: launches vlc with vlc-only password"
            )
            if "http-password" in text:
                assert "http-password=password" in text or "--http-password password" in text, (
                    f"{f.name}: vlc http-password must be 'password' to match the getter"
                )

    @_requires_catalogs
    def test_sheet_print_tasks_use_explicit_libreoffice_csv_sidecars(self):
        """OSWorld sheet_print parity comes from task postconfig, not getter synthesis."""
        rows = []
        data = Path(__file__).resolve().parents[5] / "lite/gym/envs/lite/osworld/data/eval.jsonl"

        for line in data.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            evaluator = row.get("evaluator")
            if not isinstance(evaluator, dict):
                evaluator = row.get("metadata", {}).get("evaluator")
            if not isinstance(evaluator, dict) or evaluator.get("func") != "compare_table":
                continue
            rules = (evaluator.get("options") or {}).get("rules") or []
            if any(isinstance(rule, dict) and rule.get("type") == "sheet_print" for rule in rules):
                rows.append((row["task_id"], evaluator))

        assert len(rows) == 7
        for task_id, evaluator in rows:
            postconfig = evaluator.get("postconfig") or []
            result = evaluator.get("result") or {}
            expected = evaluator.get("expected") or {}
            result_paths = result.get("path") or []
            expected_paths = expected.get("path") or []

            assert any(
                "convert-to" in json.dumps(step) and "csv" in json.dumps(step).lower()
                for step in postconfig
            ), task_id
            assert any(str(path).endswith(".csv") for path in result_paths), task_id
            assert any(str(path).endswith(".csv") for path in expected_paths), task_id

    def test_download_url_empty(self):
        from lite.gym.envs.lite.osworld.src.eval.runner import _download_url

        assert _download_url("", "/tmp") is None

    def test_download_url_with_dest(self):
        import tempfile

        from lite.gym.envs.lite.osworld.src.eval.runner import _download_url

        cache = tempfile.mkdtemp()
        # _download_url with dest uses the dest filename
        # (we can't test actual download without network, just test the path logic)
        result = _download_url("", cache, dest="test.txt")
        assert result is None  # empty URL returns None


class TestEvalMetricCalling:
    """Test that metric calling handles single-arg and multi-arg metrics."""

    def test_single_arg_metric_detection(self):
        """Metrics with 1 required param should be called with just result_data."""
        import inspect

        # Simulate a single-arg metric
        def fake_metric(result):
            return 1.0

        sig = inspect.signature(fake_metric)
        n_params = len(
            [
                p
                for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind
                in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
        )
        assert n_params == 1

    def test_multi_arg_metric_detection(self):
        """Metrics with 2+ required params should be called with result + expected."""
        import inspect

        def fake_metric(result, expected, **kwargs):
            return 1.0

        sig = inspect.signature(fake_metric)
        n_params = len(
            [
                p
                for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind
                in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
        )
        assert n_params == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("conj", "scores", "expected"),
        [
            ("and", [0.25], 0.25),
            ("and", [1.0, 0.5], 0.75),
            ("and", [1.0, 0.0], 0.0),
            ("or", [0.25, 0.5], 0.5),
            ("or", [1.0, 0.5], 1.0),
            ("or", [1.0, 1.2], 1.2),
        ],
    )
    async def test_eval_aggregation_preserves_osworld_partial_credit(
        self, monkeypatch, conj, scores, expected
    ):
        """OSWorld aggregation keeps raw partial scores; it is not thresholded."""
        from lite.gym.envs.lite.osworld.src.eval import metrics as custom_metrics
        from lite.gym.envs.lite.osworld.src.eval import runner

        async def fake_result(_computer, _config, _cache_dir):
            return "result"

        async def fake_expected(_computer, _config, _cache_dir):
            return "expected"

        def unit_score(_result, _expected, *, value):
            return value

        monkeypatch.setattr(runner, "_get_result", fake_result)
        monkeypatch.setattr(runner, "_get_expected", fake_expected)
        monkeypatch.setattr(custom_metrics, "_unit_score", unit_score, raising=False)

        evaluator = {
            "_postconfig_done": True,
            "conj": conj,
            "func": ["_unit_score"] * len(scores),
            "result": [{}] * len(scores),
            "expected": [{}] * len(scores),
            "options": [{"value": score} for score in scores],
        }

        assert await runner.evaluate_osworld_task(None, evaluator) == expected


# =========================================================================
# Verify init
# =========================================================================


class TestVerifyInit:
    """Tests for verify/__init__.py."""

    def test_evaluate_final_fn_exists(self):
        from lite.gym.envs.lite.osworld.src.utils.verify import evaluate_final_fn

        assert inspect.iscoroutinefunction(evaluate_final_fn)

    def test_evaluate_final_fn_accepts_actions(self):
        """evaluate_final_fn should accept (task, computer, actions) signature."""
        from lite.gym.envs.lite.osworld.src.utils.verify import evaluate_final_fn

        sig = inspect.signature(evaluate_final_fn)
        params = list(sig.parameters.keys())
        assert params == ["task", "computer", "actions", "debug"]


class TestInfeasibleHandling:
    """Tests for infeasible task handling."""

    def test_lite_osworld_env_has_extra_tools(self):
        from lite.gym.envs.lite.osworld.main import LiteOsworldTools

        assert LiteOsworldTools.get_tool_names() == frozenset({"report_infeasible"})
        assert (
            tool_schema_name(LiteOsworldTools.get_tool_schema("report_infeasible"))
            == "report_infeasible"
        )

    def test_lite_osworld_env_metadata_exposes_extra_tools(self):
        pytest.importorskip("desktop_env", reason="required for lite.osworld evaluators")
        from lite.gym.envs.lite.osworld.main import LiteOsworldEnv, LiteOsworldTools
        from lite.gym.sandbox.types import SandboxTaskConfig

        declared_schemas = LiteOsworldTools.get_tool_schemas()
        task = SandboxTaskConfig(
            task_id="test",
            instruction="test",
            platform="desktop",
            computer={},
            max_steps=1,
            metadata={"evaluator": {"func": "infeasible"}},
            extra_tool_schemas=declared_schemas,
        )
        # Yaml must explicitly opt in (default ``[]``)
        env = LiteOsworldEnv(task=task, extra_tools=["report_infeasible"])
        assert env.metadata.extra_tool_schemas == declared_schemas

    @staticmethod
    def _unbooted_env(extra_tools: list[str]):
        from lite.gym.envs.lite.osworld.main import LiteOsworldEnv
        from lite.gym.sandbox.types import SandboxTaskConfig

        env = LiteOsworldEnv.__new__(LiteOsworldEnv)
        env._task = SandboxTaskConfig(
            task_id="test",
            instruction="test",
            platform="desktop",
            computer={},
            max_steps=1,
            metadata={"others": {}},
        )
        env._extra_tool_schemas = LiteOsworldEnv.extra_tool_schemas(extra_tools)
        env._expose_oracle = False
        env.env_id = None
        env.task_id = None
        return env

    class _FakeInterface:
        def __init__(self, image: bytes = b"screenshot"):
            self.image = image
            self.calls: list[tuple] = []

        async def get_screen_size(self):
            return {"width": 1000, "height": 1000}

        async def screenshot(self):
            self.calls.append(("screenshot",))
            return self.image

        async def left_click(self, x, y):
            self.calls.append(("left_click", x, y))

    def _runnable_env(
        self,
        extra_tools: list[str],
        *,
        image: bytes = b"screenshot",
        max_steps: int = 10,
        evaluate_step_fn=None,
        evaluate_final_fn=None,
    ):
        env = self._unbooted_env(extra_tools)
        interface = self._FakeInterface(image=image)
        env._computer = SimpleNamespace(interface=interface)
        env._display_resolution = (1000, 1000)
        env._post_action_delay = 0.0
        env._max_steps = max_steps
        env._step_count = 0
        env._evaluate_step_fn = evaluate_step_fn
        env._evaluate_final_fn = evaluate_final_fn
        env._debug = False
        return env, interface

    @pytest.mark.asyncio
    async def test_release_rows_active_known_tool(self):
        from lite.core.tools import make_tool_call

        env, interface = self._runnable_env([])

        result = await env.step(
            [
                make_tool_call(
                    "click",
                    {"coordinate": [500, 500]},
                    call_id="active_known_tool",
                )
            ]
        )

        assert ("left_click", 500, 500) in interface.calls
        assert result.terminated is False
        assert result.truncated is False
        assert result.results[0].tool_call_id == "active_known_tool"
        assert result.results[0].images[-1] == b"screenshot"
        assert result.results[0].text is None
        assert result.results[0].error is None
        assert result.results[0].metadata is None

    @pytest.mark.asyncio
    async def test_t2_release_rows_malformed_known_action(self):
        from lite.core.tools import make_tool_call

        env, interface = self._runnable_env([])

        result = await env.step(
            [
                make_tool_call(
                    "click",
                    {"coordinate": [None, None]},
                    call_id="malformed_known_action",
                )
            ]
        )

        assert not any(call[0] == "left_click" for call in interface.calls)
        assert result.terminated is False
        assert result.truncated is False
        assert result.results[0].tool_call_id == "malformed_known_action"
        assert result.results[0].images[-1] == b"screenshot"
        assert result.results[0].text is None
        assert result.results[0].error == (
            "invalid arguments for click: arguments could not be interpreted"
        )
        assert result.results[0].metadata == {"is_error": True}

    @pytest.mark.asyncio
    async def test_malformed_pairable_envelope_delegates_to_shared_ingress(self):
        env, interface = self._runnable_env([])
        action = {
            "id": "bad_env",
            "type": "function",
            "function": {"arguments": {}},
        }

        result = await env.step([action])

        assert result.terminated is False
        assert result.truncated is False
        assert result.results[0].tool_call_id == "bad_env"
        assert result.results[0].images[-1] == b"screenshot"
        assert result.results[0].text is None
        assert result.results[0].error == (
            "invalid tool call: tool_call.function.name must be a non-empty string"
        )
        assert result.results[0].metadata == {"is_error": True}
        assert interface.calls == [("screenshot",)]

    @pytest.mark.asyncio
    async def test_t2_release_rows_literal_unknown_tool(self):
        from lite.core.tools import make_tool_call

        env, interface = self._runnable_env([])

        result = await env.step([make_tool_call("foo", {}, call_id="literal_unknown_tool")])

        assert interface.calls == [("screenshot",)]
        assert result.terminated is False
        assert result.truncated is False
        assert result.results[0].tool_call_id == "literal_unknown_tool"
        assert result.results[0].images == []
        assert result.results[0].text is None
        assert result.results[0].error == "unknown tool: foo"
        assert result.results[0].metadata == {"is_error": True}

    @pytest.mark.asyncio
    async def test_t2_release_rows_content_only_final_text(self):
        from lite.core.messages.final import make_no_tool_call_final_actions

        seen_actions: list[dict] = []

        async def evaluate_final(_task, _computer, actions, _debug):
            seen_actions.extend(actions)
            return 0.75

        env, interface = self._runnable_env([], evaluate_final_fn=evaluate_final)

        actions = make_no_tool_call_final_actions("final text")
        result = await env.step(actions)

        assert tool_call_name(actions[0]) == "response"
        assert tool_call_arguments(actions[0]) == {"text": "final text"}
        assert tool_call_id(actions[0]) is None
        assert interface.calls == [("screenshot",)]
        assert [action["name"] for action in seen_actions] == ["response"]
        assert result.terminated is True
        assert result.truncated is False
        assert result.reward == 0.75
        assert result.results == []

    @pytest.mark.asyncio
    async def test_t2_release_rows_image_data_binding(self):
        from lite.core.tools import make_tool_call

        env, _interface = self._runnable_env(
            [],
            image=b"post-action-bound-frame",
        )

        result = await env.step(
            [
                make_tool_call(
                    "click",
                    {"coordinate": [500, 500]},
                    call_id="image_data_binding",
                )
            ]
        )

        assert result.results[0].tool_call_id == "image_data_binding"
        assert result.results[0].images[-1] == b"post-action-bound-frame"
        assert result.results[0].images[-1] != b"reset-frame"
        assert result.results[0].error is None
        assert result.results[0].metadata is None

    @pytest.mark.asyncio
    async def test_t2_release_rows_response_terminal_tool(self):
        from lite.core.tools import make_tool_call

        seen_actions: list[dict] = []

        async def evaluate_final(_task, _computer, actions, _debug):
            seen_actions.extend(actions)
            return 1.0

        env, interface = self._runnable_env(
            ["response"],
            evaluate_final_fn=evaluate_final,
        )

        result = await env.step(
            [
                make_tool_call(
                    "response",
                    {"text": "done"},
                    call_id="response_terminal_tool",
                )
            ]
        )

        assert interface.calls == [("screenshot",)]
        assert [action["name"] for action in seen_actions] == ["response"]
        assert result.terminated is True
        assert result.truncated is False
        assert result.reward == 1.0
        # A terminal call gets NO tool result: it ended the episode, so there
        # is no next decision for an observation to inform, and
        # ``devs/migration/verify.py`` refuses a tool result for a terminal call.
        assert result.results == []

    @pytest.mark.asyncio
    async def test_report_infeasible_without_extra_tool_is_unsupported(
        self,
        monkeypatch,
    ):
        from lite.core.tools import make_tool_call
        from lite.gym.sandbox import SandboxBaseEnv

        env, interface = self._runnable_env([])

        async def fake_base_step(self, actions):
            raise AssertionError("local report_infeasible rejection must not touch backend")

        monkeypatch.setattr(SandboxBaseEnv, "step", fake_base_step)

        result = await env.step(
            [
                make_tool_call(
                    "report_infeasible",
                    {"reason": "missing"},
                    call_id="call_report",
                )
            ]
        )

        assert env.metadata.extra_tool_schemas == []
        assert result.terminated is False
        assert len(result.results) == 1
        assert result.results[0].tool_call_id == "call_report"
        assert result.results[0].images[-1] == b"screenshot"
        assert result.results[0].text is None
        assert result.results[0].error == ("report_infeasible is not available in this task.")
        assert result.results[0].metadata == {"is_error": True}
        assert interface.calls == [("screenshot",)]

    @pytest.mark.asyncio
    async def test_unsupported_report_infeasible_rejection_counts_step_and_final_eval(
        self,
        monkeypatch,
    ):
        from lite.core.tools import make_tool_call
        from lite.gym.sandbox import SandboxBaseEnv

        seen_actions: list[dict] = []

        async def evaluate_final(_task, _computer, actions, _debug):
            seen_actions.extend(actions)
            return 0.25, {"phase": "final"}

        env, interface = self._runnable_env(
            [],
            max_steps=1,
            evaluate_final_fn=evaluate_final,
        )

        async def fake_base_step(self, actions):
            raise AssertionError("local report_infeasible rejection must not touch backend")

        monkeypatch.setattr(SandboxBaseEnv, "step", fake_base_step)

        result = await env.step(
            [
                make_tool_call(
                    "report_infeasible",
                    {"reason": "missing"},
                    call_id="call_report",
                )
            ]
        )

        assert env._step_count == 1
        assert result.terminated is False
        assert result.truncated is True
        assert result.reward == 0.25
        assert result.info["stop_reason"] == "max_steps"
        assert result.info["eval"] == {"phase": "final"}
        assert result.info["executed_actions"] == [
            {
                "call": "noop",
                "args": {
                    "name": "report_infeasible",
                    "reason": "inactive extra tool",
                },
            }
        ]
        assert seen_actions == []
        assert result.results[0].tool_call_id == "call_report"
        assert result.results[0].images[-1] == b"screenshot"
        assert result.results[0].error == ("report_infeasible is not available in this task.")
        assert result.results[0].metadata == {"is_error": True}
        assert interface.calls == [("screenshot",)]

    @pytest.mark.asyncio
    async def test_report_infeasible_without_call_id_does_not_touch_backend(
        self,
        monkeypatch,
    ):
        from lite.core.tools import make_tool_call
        from lite.gym.sandbox import SandboxBaseEnv

        env = self._unbooted_env([])

        async def fake_base_step(self, actions):
            raise AssertionError("unpaired local report_infeasible must not touch backend")

        monkeypatch.setattr(SandboxBaseEnv, "step", fake_base_step)

        result = await env.step(
            [
                make_tool_call("report_infeasible", {"reason": "missing"}),
            ]
        )

        assert result.terminated is False
        assert result.truncated is False
        assert result.results == []
        assert env._step_count == 1

    @pytest.mark.asyncio
    async def test_active_report_infeasible_maps_to_local_internal_terminate(
        self,
    ):
        from lite.core.messages.final import ENV_INTERNAL_TERMINATE_REASON
        from lite.core.tools import make_tool_call

        seen_eval_actions = []

        async def evaluate_final(_task, _computer, actions, _debug):
            seen_eval_actions.extend(actions)
            return 1.0

        env, interface = self._runnable_env(
            ["report_infeasible"],
            evaluate_final_fn=evaluate_final,
        )

        result = await env.step(
            [
                make_tool_call(
                    "report_infeasible",
                    {"reason": "missing"},
                    call_id="call_report",
                )
            ]
        )

        expected_action = {
            "name": "terminate",
            "arguments": {"status": "failure", "reason": "missing"},
            "_internal_stop_reason": ENV_INTERNAL_TERMINATE_REASON,
            "_result_call_id": "call_report",
        }
        assert seen_eval_actions == [expected_action]
        assert [tool_schema_name(schema) for schema in env.metadata.extra_tool_schemas] == [
            "report_infeasible"
        ]
        assert result.terminated is True
        assert result.reward == 1.0
        assert [item.tool_call_id for item in result.results] == ["call_report"]
        assert result.results[0].images[-1] == b"screenshot"
        assert result.results[0].text is None
        assert interface.calls == [("screenshot",)]

    @pytest.mark.asyncio
    async def test_active_report_infeasible_delegated_remap_uses_side_table(
        self,
        monkeypatch,
    ):
        from lite.core.messages.final import ENV_INTERNAL_TERMINATE_REASON
        from lite.core.tools import make_tool_call
        from lite.core.tools.results import LiteToolResult
        from lite.gym.sandbox import SandboxBaseEnv
        from lite.gym.types import LiteEnvStepResult
        from lite.gym.utils.feedback.ingress import make_internal_terminate_action

        env, interface = self._runnable_env(["report_infeasible"])
        seen_actions = []

        async def fake_base_step(self, actions):
            seen_actions.extend(actions)
            return LiteEnvStepResult(
                terminated=True,
                results=[LiteToolResult(tool_call_id="call_click", images=[b"shot"])],
            )

        monkeypatch.setattr(SandboxBaseEnv, "step", fake_base_step)

        result = await env.step(
            [
                make_tool_call(
                    "report_infeasible",
                    {"reason": "missing"},
                    call_id="call_report",
                ),
                make_tool_call(
                    "click",
                    {"coordinate": [500, 500]},
                    call_id="call_click",
                ),
            ]
        )

        expected_action = make_internal_terminate_action(
            status="failure",
            reason="missing",
            internal_reason=ENV_INTERNAL_TERMINATE_REASON,
            result_call_id="call_report",
        )
        assert seen_actions == [
            expected_action,
            make_tool_call(
                "click",
                {"coordinate": [500, 500]},
                call_id="call_click",
            ),
        ]
        assert seen_actions[0]["_result_call_id"] == "call_report"
        assert interface.calls == []
        assert [item.tool_call_id for item in result.results] == [
            "call_report",
            "call_click",
        ]
        assert result.results[0].images[-1] == b"shot"
        assert result.results[0].text is None
        assert result.results[0].error is None

    @pytest.mark.asyncio
    async def test_active_report_infeasible_schema_error_not_remapped(
        self,
        monkeypatch,
    ):
        from lite.core.tools import make_tool_call
        from lite.gym.sandbox import SandboxBaseEnv

        env, interface = self._runnable_env(["report_infeasible"])

        async def fake_base_step(self, actions):
            raise AssertionError("malformed report_infeasible must not touch backend")

        monkeypatch.setattr(SandboxBaseEnv, "step", fake_base_step)

        result = await env.step(
            [
                make_tool_call(
                    "report_infeasible",
                    {},
                    call_id="call_report",
                )
            ]
        )

        assert len(result.results) == 1
        assert result.results[0].tool_call_id == "call_report"
        assert result.results[0].images[-1] == b"screenshot"
        assert result.results[0].text is None
        assert result.results[0].metadata == {"is_error": True}
        assert result.results[0].error == (
            "invalid arguments for report_infeasible: "
            "report_infeasible.arguments.reason is required"
        )
        assert interface.calls == [("screenshot",)]

    @pytest.mark.asyncio
    async def test_malformed_report_infeasible_rejection_counts_step_and_step_eval(
        self,
        monkeypatch,
    ):
        from lite.core.tools import make_tool_call
        from lite.gym.sandbox import SandboxBaseEnv

        seen_actions: list[dict] = []

        async def evaluate_step(_task, _computer, actions, _debug):
            seen_actions.extend(actions)
            return 0.5, {"phase": "step"}

        env, interface = self._runnable_env(
            ["report_infeasible"],
            evaluate_step_fn=evaluate_step,
        )

        async def fake_base_step(self, actions):
            raise AssertionError("malformed report_infeasible must not touch backend")

        monkeypatch.setattr(SandboxBaseEnv, "step", fake_base_step)

        result = await env.step(
            [
                make_tool_call(
                    "report_infeasible",
                    {},
                    call_id="call_report",
                )
            ]
        )

        assert env._step_count == 1
        assert result.terminated is False
        assert result.truncated is False
        assert result.reward == 0.5
        assert result.info["eval"] == {"phase": "step"}
        assert result.info["executed_actions"] == [
            {
                "call": "noop",
                "args": {
                    "name": "report_infeasible",
                    "reason": (
                        "invalid arguments for report_infeasible: "
                        "report_infeasible.arguments.reason is required"
                    ),
                },
            }
        ]
        assert seen_actions == []
        assert result.results[0].tool_call_id == "call_report"
        assert result.results[0].images[-1] == b"screenshot"
        assert result.results[0].text is None
        assert result.results[0].metadata == {"is_error": True}
        assert result.results[0].error == (
            "invalid arguments for report_infeasible: "
            "report_infeasible.arguments.reason is required"
        )
        assert interface.calls == [("screenshot",)]

    @pytest.mark.asyncio
    async def test_batched_gui_call_is_delegated_without_result_rebuild(self, monkeypatch):
        from lite.core.tools import make_tool_call
        from lite.core.tools.results import LiteToolResult
        from lite.gym.sandbox import SandboxBaseEnv
        from lite.gym.types import LiteEnvStepResult

        env = self._unbooted_env([])
        seen_actions = []

        async def fake_base_step(self, actions):
            seen_actions.extend(actions)
            return LiteEnvStepResult(
                results=[
                    LiteToolResult(
                        tool_call_id=tool_call_id(actions[0]),
                        images=[b"shot"],
                        text="current desktop observation",
                        metadata={"is_error": True},
                        error="invalid arguments for key: bad key",
                    )
                ],
            )

        monkeypatch.setattr(SandboxBaseEnv, "step", fake_base_step)

        result = await env.step(
            [
                make_tool_call(
                    "computer",
                    {"actions": [{"action": "key", "keys": ["not_a_real_key"]}]},
                    call_id="gui_0",
                )
            ]
        )

        assert seen_actions == [
            make_tool_call(
                "computer",
                {"actions": [{"action": "key", "keys": ["not_a_real_key"]}]},
                call_id="gui_0",
            )
        ]
        assert len(result.results) == 1
        assert result.results[0].tool_call_id == "gui_0"
        assert result.results[0].images[-1] == b"shot"
        assert result.results[0].text == "current desktop observation"
        assert result.results[0].metadata == {"is_error": True}
        assert result.results[0].error == "invalid arguments for key: bad key"

    @pytest.mark.asyncio
    async def test_screenshot_call_at_env_max_steps_returns_paired_current_result(self):
        from lite.core.tools import make_tool_call

        class FakeInterface:
            async def get_screen_size(self):
                return {"width": 1920, "height": 1080}

            async def screenshot(self):
                return b"fresh-screen"

        async def fake_evaluate_final(_task, _computer, actions, _debug):
            seen_eval_actions.extend(actions)
            return 0.0, {"eval_probe": "final"}

        seen_eval_actions: list[dict] = []
        env = self._unbooted_env([])
        env._computer = SimpleNamespace(interface=FakeInterface())
        env._display_resolution = (1920, 1080)
        env._post_action_delay = 0.0
        env._max_steps = 1
        env._step_count = 0
        env._evaluate_step_fn = None
        env._evaluate_final_fn = fake_evaluate_final
        env._debug = False

        result = await env.step(
            [
                make_tool_call(
                    "computer",
                    {"actions": [{"action": "screenshot"}]},
                    call_id="call_screen",
                )
            ]
        )

        assert result.terminated is False
        assert result.truncated is True
        assert result.reward == 0.0
        assert result.info["executed_actions"] == []
        assert result.info["eval"] == {"eval_probe": "final"}
        assert seen_eval_actions == [{"name": "screenshot", "arguments": {}}]
        assert len(result.results) == 1
        tool_result = result.results[0]
        assert tool_result.tool_call_id == "call_screen"
        assert tool_result.images[-1] == b"fresh-screen"
        assert tool_result.text is None
        assert tool_result.metadata is None
        assert tool_result.error is None

    @pytest.mark.asyncio
    async def test_unsupported_report_infeasible_preserves_delegated_current_carrier(
        self,
        monkeypatch,
    ):
        from lite.core.tools import make_tool_call
        from lite.core.tools.results import LiteToolResult
        from lite.gym.sandbox import SandboxBaseEnv
        from lite.gym.types import LiteEnvStepResult

        env = self._unbooted_env([])
        seen_actions = []

        async def fake_base_step(self, actions):
            seen_actions.extend(actions)
            return LiteEnvStepResult(
                results=[
                    LiteToolResult(
                        tool_call_id="call_click",
                        images=[b"shot"],
                    )
                ]
            )

        monkeypatch.setattr(SandboxBaseEnv, "step", fake_base_step)

        result = await env.step(
            [
                make_tool_call(
                    "report_infeasible",
                    {"reason": "missing"},
                    call_id="call_report",
                ),
                make_tool_call(
                    "click",
                    {"coordinate": [500, 500]},
                    call_id="call_click",
                ),
            ]
        )

        assert seen_actions == [
            make_tool_call(
                "click",
                {"coordinate": [500, 500]},
                call_id="call_click",
            ),
        ]
        by_id = {tool_result.tool_call_id: tool_result for tool_result in result.results}
        assert set(by_id) == {"call_click", "call_report"}
        assert by_id["call_click"].images[-1] == b"shot"
        assert by_id["call_click"].text is None
        assert by_id["call_report"].images[-1] == b"shot"
        assert by_id["call_report"].text is None
        assert by_id["call_report"].metadata == {"is_error": True}
        assert by_id["call_report"].error == ("report_infeasible is not available in this task.")

    @pytest.mark.asyncio
    async def test_unsupported_report_infeasible_with_bash_text_takes_current_image(
        self,
        monkeypatch,
    ):
        from lite.core.tools import make_tool_call
        from lite.core.tools.results import LiteToolResult
        from lite.gym.sandbox import SandboxBaseEnv
        from lite.gym.types import LiteEnvStepResult

        env, interface = self._runnable_env([], image=b"current-shot")
        seen_actions = []

        async def fake_base_step(self, actions):
            seen_actions.extend(actions)
            return LiteEnvStepResult(
                results=[
                    LiteToolResult(
                        tool_call_id="call_bash",
                        text="pwd output",
                        metadata={"returncode": 0},
                    )
                ]
            )

        monkeypatch.setattr(SandboxBaseEnv, "step", fake_base_step)

        result = await env.step(
            [
                make_tool_call(
                    "report_infeasible",
                    {"reason": "missing"},
                    call_id="call_report",
                ),
                make_tool_call(
                    "bash",
                    {"command": "pwd"},
                    call_id="call_bash",
                ),
            ]
        )

        assert seen_actions == [
            make_tool_call(
                "bash",
                {"command": "pwd"},
                call_id="call_bash",
            )
        ]
        assert all("_result_call_id" not in action for action in seen_actions)
        assert interface.calls == [("screenshot",)]
        by_id = {tool_result.tool_call_id: tool_result for tool_result in result.results}
        assert set(by_id) == {"call_bash", "call_report"}
        assert by_id["call_bash"].text == "pwd output"
        assert by_id["call_bash"].images == []
        assert by_id["call_report"].images[-1] == b"current-shot"
        assert by_id["call_report"].text is None
        assert by_id["call_report"].metadata == {"is_error": True}
        assert by_id["call_report"].error == ("report_infeasible is not available in this task.")

    @pytest.mark.asyncio
    async def test_malformed_report_infeasible_preserves_delegated_current_carrier(
        self,
        monkeypatch,
    ):
        from lite.core.tools import make_tool_call
        from lite.core.tools.results import LiteToolResult
        from lite.gym.sandbox import SandboxBaseEnv
        from lite.gym.types import LiteEnvStepResult

        env = self._unbooted_env(["report_infeasible"])

        async def fake_base_step(self, actions):
            return LiteEnvStepResult(
                results=[
                    LiteToolResult(
                        tool_call_id="call_click",
                        images=[b"shot"],
                    )
                ]
            )

        monkeypatch.setattr(SandboxBaseEnv, "step", fake_base_step)

        result = await env.step(
            [
                make_tool_call("report_infeasible", {}, call_id="call_report"),
                make_tool_call("click", {"coordinate": [500, 500]}, call_id="call_click"),
            ]
        )

        by_id = {tool_result.tool_call_id: tool_result for tool_result in result.results}
        assert set(by_id) == {"call_click", "call_report"}
        assert by_id["call_report"].images[-1] == b"shot"
        assert by_id["call_report"].text is None
        assert by_id["call_report"].metadata == {"is_error": True}
        assert by_id["call_report"].error == (
            "invalid arguments for report_infeasible: "
            "report_infeasible.arguments.reason is required"
        )

    @pytest.mark.asyncio
    async def test_infeasible_report_infeasible_reward(self):
        """Agent uses report_infeasible on infeasible task → reward 1.0."""
        from lite.gym.envs.lite.osworld.src.utils.verify import evaluate_final_fn
        from lite.gym.sandbox.types import SandboxTaskConfig

        task = SandboxTaskConfig(
            task_id="test",
            instruction="test",
            platform="desktop",
            computer={},
            max_steps=1,
            metadata={"evaluator": {"func": "infeasible"}},
        )
        actions = [{"name": "report_infeasible", "arguments": {"reason": "no app"}}]
        reward = await evaluate_final_fn(task, None, actions)
        assert reward == 1.0

    @pytest.mark.asyncio
    async def test_infeasible_env_internal_report_infeasible_debug_reward(self):
        """Step 0.4: env-internal report_infeasible scores infeasible."""
        from lite.gym.envs.lite.osworld.src.utils.verify import evaluate_final_fn
        from lite.gym.sandbox.types import SandboxTaskConfig

        task = SandboxTaskConfig(
            task_id="test",
            instruction="test",
            platform="desktop",
            computer={},
            max_steps=1,
            metadata={"evaluator": {"func": "infeasible"}},
        )
        actions = [{"name": "report_infeasible", "arguments": {"reason": "no app"}}]

        assert await evaluate_final_fn(task, None, actions) == 1.0
        assert await evaluate_final_fn(task, None, actions, debug=True) == (
            1.0,
            {"infeasible": True},
        )

    @pytest.mark.asyncio
    async def test_infeasible_canonical_report_infeasible_debug_reward(self):
        """Nested canonical report_infeasible scores infeasible."""
        from lite.core.tools import make_tool_call
        from lite.gym.envs.lite.osworld.src.utils.verify import evaluate_final_fn
        from lite.gym.sandbox.types import SandboxTaskConfig

        task = SandboxTaskConfig(
            task_id="test",
            instruction="test",
            platform="desktop",
            computer={},
            max_steps=1,
            metadata={"evaluator": {"func": "infeasible"}},
        )
        actions = [
            make_tool_call(
                "report_infeasible",
                {"reason": "no app"},
                call_id="call_report",
            )
        ]

        assert await evaluate_final_fn(task, None, actions) == 1.0
        assert await evaluate_final_fn(task, None, actions, debug=True) == (
            1.0,
            {"infeasible": True},
        )

    @pytest.mark.asyncio
    async def test_infeasible_terminate_failure_reward(self):
        """Agent uses terminate(status=failure) on infeasible task → reward 1.0."""
        from lite.gym.envs.lite.osworld.src.utils.verify import evaluate_final_fn
        from lite.gym.sandbox.types import SandboxTaskConfig

        task = SandboxTaskConfig(
            task_id="test",
            instruction="test",
            platform="desktop",
            computer={},
            max_steps=1,
            metadata={"evaluator": {"func": "infeasible"}},
        )
        actions = [{"name": "terminate", "arguments": {"status": "failure"}}]
        reward = await evaluate_final_fn(task, None, actions)
        assert reward == 1.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "malformed",
        [
            pytest.param({"name": "terminate"}, id="no-arguments-key"),
            pytest.param({"arguments": {"status": "failure"}}, id="no-name-key"),
            pytest.param({"name": "terminate", "arguments": None}, id="arguments-not-a-dict"),
            pytest.param({"name": 7, "arguments": {}}, id="name-not-a-str"),
            pytest.param(
                {"name": "response", "arguments": {"text": 7}}, id="response-text-not-a-str"
            ),
            pytest.param("terminate", id="not-a-dict-at-all"),
            pytest.param(
                {"function": {"name": "terminate", "arguments": "{}"}},
                id="noncanonical-tool-call",
            ),
        ],
    )
    async def test_infeasible_reward_scores_malformed_actions_zero_never_raises(self, malformed):
        """R18 — a malformed final action must SCORE 0.0, not error the episode.

        This replaces a test that pinned the ``KeyError`` from the old bare
        ``action["name"]`` / ``action["arguments"]`` reads. Pinning the raise
        enforced the defect: ``evaluate_final_fn`` is the reward boundary, so a
        raise removes the task from the eval DENOMINATOR rather than giving it
        the ``0.0`` an unrecognized final action has earned.

        None of these shapes can reach the reward on the rollout path — see the
        producer contract on ``SandboxBaseEnv._finalize_step_result`` — which is
        exactly why this gate exists: without it nothing pins the reader's
        totality and the next refactor re-introduces the raise.
        """
        from lite.gym.envs.lite.osworld.src.utils.verify import evaluate_final_fn
        from lite.gym.sandbox.types import SandboxTaskConfig

        task = SandboxTaskConfig(
            task_id="test",
            instruction="test",
            platform="desktop",
            computer={},
            max_steps=1,
            metadata={"evaluator": {"func": "infeasible"}},
        )

        assert await evaluate_final_fn(task, None, [malformed]) == 0.0
        assert await evaluate_final_fn(task, None, [malformed], debug=True) == (
            0.0,
            {"infeasible": False},
        )

    @pytest.mark.asyncio
    async def test_infeasible_reward_still_scores_a_wellformed_action_after_a_malformed_one(
        self,
    ):
        """Totality must not swallow a correct report that follows junk."""
        from lite.gym.envs.lite.osworld.src.utils.verify import evaluate_final_fn
        from lite.gym.sandbox.types import SandboxTaskConfig

        task = SandboxTaskConfig(
            task_id="test",
            instruction="test",
            platform="desktop",
            computer={},
            max_steps=1,
            metadata={"evaluator": {"func": "infeasible"}},
        )
        actions = [
            {"name": "terminate"},
            {"name": "terminate", "arguments": {"status": "failure"}},
        ]

        assert await evaluate_final_fn(task, None, actions) == 1.0

    def test_infeasible_reader_does_not_decode_json_string_arguments(self):
        """The osworld reader is total but NARROWER than the ScaleCUA twin.

        ``lite/gym/envs/lite/scalecua/src/osworld/verify.py::_action_call``
        decodes an OpenAI-style ``arguments`` JSON STRING because that env's
        oracle/replay harnesses under ``devs/envs/lite.scalecua/validate/`` call
        ``evaluate_final_fn`` directly with provider-shaped actions. lite.osworld
        has no such caller — both harnesses under
        ``devs/envs/lite.osworld/validate/`` pass ``actions=None`` — so decoding
        here would be tolerance with no producer behind it.
        """
        from lite.gym.envs.lite.osworld.src.utils.verify import _action_call

        assert _action_call({"name": "terminate", "arguments": '{"status": "failure"}'}) == ("", {})

    @pytest.mark.asyncio
    async def test_infeasible_response_text_reward(self):
        """Agent uses response([INFEASIBLE]) on infeasible task → reward 1.0."""
        from lite.gym.envs.lite.osworld.src.utils.verify import evaluate_final_fn
        from lite.gym.sandbox.types import SandboxTaskConfig

        task = SandboxTaskConfig(
            task_id="test",
            instruction="test",
            platform="desktop",
            computer={},
            max_steps=1,
            metadata={"evaluator": {"func": "infeasible"}},
        )
        actions = [{"name": "response", "arguments": {"text": "This task is [INFEASIBLE]"}}]
        reward = await evaluate_final_fn(task, None, actions)
        assert reward == 1.0

    @pytest.mark.asyncio
    async def test_infeasible_no_report_reward(self):
        """Agent doesn't report infeasible on infeasible task → reward 0.0."""
        from lite.gym.envs.lite.osworld.src.utils.verify import evaluate_final_fn
        from lite.gym.sandbox.types import SandboxTaskConfig

        task = SandboxTaskConfig(
            task_id="test",
            instruction="test",
            platform="desktop",
            computer={},
            max_steps=1,
            metadata={"evaluator": {"func": "infeasible"}},
        )
        actions = [{"name": "terminate", "arguments": {}}]
        reward = await evaluate_final_fn(task, None, actions)
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_infeasible_no_actions_reward(self):
        """No actions on infeasible task → reward 0.0."""
        from lite.gym.envs.lite.osworld.src.utils.verify import evaluate_final_fn
        from lite.gym.sandbox.types import SandboxTaskConfig

        task = SandboxTaskConfig(
            task_id="test",
            instruction="test",
            platform="desktop",
            computer={},
            max_steps=1,
            metadata={"evaluator": {"func": "infeasible"}},
        )
        reward = await evaluate_final_fn(task, None, None)
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_normal_task_report_infeasible_no_reward(self):
        """Agent reports infeasible on normal task → should not get reward 1.0."""
        pytest.importorskip("desktop_env", reason="required for lite.osworld evaluators")
        from lite.gym.envs.lite.osworld.src.utils.verify import evaluate_final_fn
        from lite.gym.sandbox.types import SandboxTaskConfig

        task = SandboxTaskConfig(
            task_id="test",
            instruction="test",
            platform="desktop",
            computer={},
            max_steps=1,
            metadata={"evaluator": {"func": "check_include_exclude"}},
        )
        actions = [{"name": "report_infeasible", "arguments": {"reason": "fake"}}]
        # Normal task eval needs computer — just verify it doesn't return 1.0 from infeasible logic
        # (it will fail/return 0.0 because no computer, but won't return 1.0)
        reward = await evaluate_final_fn(task, None, actions)
        assert reward != 1.0


# =========================================================================
# Parquet data integrity
# =========================================================================


@_requires_catalogs
class TestDataIntegrity:
    """Tests for eval.jsonl data format (SandboxTaskDataRow schema)."""

    @pytest.fixture
    def rows(self):
        """Load eval.jsonl as a list of SandboxTaskDataRow dicts."""
        out = []
        with open(_DATA_DIR / "eval.jsonl") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def test_required_top_level_keys(self, rows):
        for row in rows:
            assert "task_id" in row
            assert "instruction" in row
            assert "metadata" in row

    def test_task_id_format(self, rows):
        for row in rows:
            assert row["task_id"].startswith("osworld_"), f"Bad task_id: {row['task_id']}"

    def test_metadata_structure(self, rows):
        for row in rows:
            m = row["metadata"]
            assert "others" in m
            assert "config" in m
            assert "evaluator" in m
            others = m["others"]
            assert "domain" in others
            assert "oracle_actions" in others
            assert isinstance(others["oracle_actions"], list)
            assert "oracle_actions" not in m
            assert "oracle_after_postconfig" not in m

    def test_oracle_actions_format(self, rows):
        """All oracle_actions should be structured {type, parameters} dicts."""
        for row in rows:
            for action in row["metadata"]["others"].get("oracle_actions", []):
                assert "type" in action, f"Missing type in oracle_action: {row['task_id']}"
                assert "parameters" in action, (
                    f"Missing parameters in oracle_action: {row['task_id']}"
                )

    def test_evaluator_structure(self, rows):
        for row in rows:
            ev = row["metadata"].get("evaluator", {})
            assert "func" in ev, f"Missing func in evaluator: {row['task_id']}"

    def test_no_derived_oracle_verified_flag(self, rows):
        """``oracle_actions`` is the single fact — no derived duplicate of it.

        A ``bool(oracle_actions)`` mirror was written by the eval generator
        only; the synth/perturb generators never wrote it, so a filter on it
        silently matched nothing on the train splits.
        """
        for row in rows:
            assert "oracle_verified" not in row["metadata"]["others"], (
                f"derived oracle flag is back: {row['task_id']}"
            )

    def test_excluded_tasks_have_reason(self, rows):
        """Tasks with exclude_reason should have a valid reason string."""
        for row in rows:
            reason = row["metadata"]["others"].get("exclude_reason")
            if reason:
                assert exclude_reasons.validate(reason) == reason

    @pytest.mark.parametrize(
        "catalog_name",
        ("eval.jsonl", "train.synth.jsonl", "train.perturb.jsonl"),
    )
    @_requires_catalogs
    def test_exclude_reason_is_canonical_in_every_catalog_row(self, catalog_name):
        path = _DATA_DIR / catalog_name
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            reason = row["metadata"]["others"].get("exclude_reason")
            if reason:
                assert exclude_reasons.validate(reason) == reason, (
                    f"{path}:{line_number}: {row['task_id']} has {reason!r}"
                )

    def test_config_action_format(self, rows):
        """All config steps should have type and parameters."""
        for row in rows:
            for step in row["metadata"].get("config", []):
                assert "type" in step, f"Missing type in config step: {row['task_id']}"
                assert "parameters" in step, f"Missing parameters in config step: {row['task_id']}"


# =========================================================================
# Split-uniform JSONL contract gates
# =========================================================================
#
# These gates fire across all three splits (eval, train.synth, train.perturb).
# They are the project's anti-destruction guarantee: a 4-day refactor cannot
# silently regress curated artifacts (oracle_actions, eval.jsonl bytes,
# dispatch contract) without one of these turning red at code review.

_ROUTED_ACTION_TYPES = {"execute", "command", "launch"}


def _join_command(command) -> str:
    if isinstance(command, str):
        return command
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    return "" if command is None else str(command)


async def _no_sleep(_seconds: float) -> None:
    """Skip dispatch_action's GUI-raise settle delays (3s + 2s per call)."""


async def _dispatched_commands(dispatch_mod, action_type: str, params: dict) -> list[str]:
    """Run the REAL ``dispatch_action`` and return every command it issued."""
    issued: list[str] = []

    class _Result:
        stdout = ""
        stderr = ""
        returncode = 0

    class _Interface:
        async def run_command(self, command, *args, **kwargs):
            issued.append(command)
            return _Result()

    await dispatch_mod.dispatch_action(
        SimpleNamespace(interface=_Interface()),
        {"type": action_type, "parameters": params},
    )
    return issued


def _dispatch_command_to_string(action_type: str, params: dict) -> str:
    """Mirror dispatch_action's command normalization before routing checks."""
    from lite.gym.envs.lite.osworld.src.utils import dispatch as dispatch_mod

    if action_type == "launch":
        command = params.get("command", "")
        if isinstance(command, str) and not params.get("shell", False) and len(command.split()) > 1:
            command = command.split()
        return _join_command(command)

    command = dispatch_mod._replace_templates(params.get("command", ""))
    if isinstance(command, str):
        command = command.replace(
            "/home/user/.config/google-chrome",
            dispatch_mod.CHROME_DATA_DIR,
        )
        command = command.replace("chmod +x setup.sh", "chmod +x /home/user/setup.sh")
        command = command.replace("./setup.sh", "/home/user/setup.sh")
        command = command.replace("bash setup.sh", "bash /home/user/setup.sh")
    return _join_command(command)


#: Any spelling of the QEMU-VM session paths. A reintroduced rewrite has to name
#: one of these SOMEWHERE in a string literal to do its job, whichever function,
#: operator or regex it is dressed up as — that is what makes the scan below
#: semantic rather than a match on one historical spelling.
_SESSION_PATH_TOKENS = (
    "/run/user",
    "XDG_RUNTIME_DIR",
    "DBUS_SESSION_BUS_ADDRESS",
    "runtime-",
    "dbus-session",
)

_DISPATCH_SRC = _REPO / "lite/gym/envs/lite/osworld/src/utils/dispatch.py"


def _string_literals(node) -> list[tuple[int, str]]:
    """Every ``str`` constant under ``node``, EXCLUDING docstrings.

    ``#`` comments are invisible to ``ast``, which is what this wants — dispatch.py
    explains the passthrough at length and that prose must not trip the scan. But a
    docstring is an ``ast.Constant`` like any other string, so "prose is safe" only
    held for the ``#`` form: the moment someone documented the same contract in a
    module or function docstring — the natural place to put it — this scan would
    report "the passthrough contract may have been reverted" about the very
    documentation of that contract, and the obvious fix would be to delete the
    documentation. Drop the docstring node explicitly so prose is safe in BOTH forms.
    """
    docstrings = {
        id(body[0].value)
        for child in ast.walk(node)
        if isinstance(child, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and (body := child.body)
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    }
    return [
        (child.lineno, child.value)
        for child in ast.walk(node)
        if isinstance(child, ast.Constant)
        and isinstance(child.value, str)
        and id(child) not in docstrings
    ]


def _replace_sources(node) -> set[str]:
    """The literal FIRST argument of every ``x.replace(old, new)`` / ``re.sub``
    under ``node`` — i.e. the set of inputs this code rewrites."""
    sources: set[str] = set()
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr in ("replace", "sub")
            and child.args
            and isinstance(child.args[0], ast.Constant)
            and isinstance(child.args[0].value, str)
        ):
            sources.add(child.args[0].value)
    return sources


def _dispatch_execute_branch() -> ast.If:
    """``dispatch_action``'s ``if t in ("execute", "command"):`` body."""
    tree = ast.parse(_DISPATCH_SRC.read_text())
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "dispatch_action"
    )
    return next(
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "t in ('execute', 'command')"
    )


def test_dispatch_declares_no_session_path_rewrite_in_any_spelling():
    """Semantic anti-drift for the session-path passthrough.

    ``0ccad92c`` added a session-path rewrite to dispatch.py AND to this file's
    local mirror; the revert (``f756eb9f``) deleted the production rewrite but not
    the mirror's copy, so the behavioural test below stayed green while certifying
    behaviour that no longer existed. The first fix for that asserted two VERBATIM
    strings (``"/tmp/dbus-session-bus-address"``, ``"${XDG_RUNTIME_DIR:-"``) were
    absent — which mutation testing showed catches only a byte-identical
    reintroduction. An equivalent rewrite spelled differently
    (``cmd.replace("/run/user/1000", "/tmp/runtime-user")``, an ``re.sub`` on
    ``XDG_RUNTIME_DIR=\\S+``, an f-string prefix, ...) sailed through.

    So assert the property instead: dispatch.py contains NO string literal naming a
    session path at all. Any rewrite must name one to find/emit it, and ``ast``
    ignores comments, so the file's prose explanation of the passthrough is free to
    keep saying ``/run/user/1000``.
    """
    tree = ast.parse(_DISPATCH_SRC.read_text())
    offenders = [
        (lineno, value[:120], token)
        for lineno, value in _string_literals(tree)
        for token in _SESSION_PATH_TOKENS
        if token in value
    ]
    assert offenders == [], (
        "dispatch.py names a session path in executable code; the passthrough "
        f"contract may have been reverted: {offenders}"
    )


def test_dispatch_mirror_rewrites_exactly_what_production_rewrites():
    """``_dispatch_command_to_string`` is a MIRROR, and a mirror is only honest
    while it matches its original — the failure mode that let the stale
    session-path rewrite survive above. Compare the two rewrite SETS (the literal
    left-hand side of every ``.replace``), so adding, dropping or retargeting a
    rewrite in production turns this red instead of silently diverging.
    """
    production = _replace_sources(_dispatch_execute_branch())
    mirror_tree = ast.parse(Path(__file__).resolve().read_text())
    mirror_fn = next(
        node
        for node in ast.walk(mirror_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_dispatch_command_to_string"
    )
    assert _replace_sources(mirror_fn) == production
    # And it is a non-empty set, so an accidentally-empty extraction cannot pass.
    assert "/home/user/.config/google-chrome" in production


@pytest.mark.asyncio
async def test_dispatch_action_leaves_hardcoded_session_paths_untouched(monkeypatch):
    """The behavioural half, driven through the REAL ``dispatch_action``.

    An upstream setup that hardcodes the QEMU-VM session paths
    (``XDG_RUNTIME_DIR=/run/user/1000``, ``DBUS_SESSION_BUS_ADDRESS=unix:path=
    /run/user/1000/bus``) must reach the container UNCHANGED: the base image now
    creates ``/run/user/<uid>`` for real, owned by the desktop user, with
    ``/tmp/runtime-<user>`` kept as a compat symlink, so the corpus's assumption is
    simply true. Exercises the audio-oracle path (pactl/pulseaudio --start) and the
    gnome-terminal path (which also takes the GUI-raise branch).

    This used to call this file's local mirror, so production could diverge without
    the test noticing. It now asserts on the command string production actually
    hands to ``interface.run_command``.
    """
    from lite.gym.envs.lite.osworld.src.utils import dispatch as dispatch_mod

    monkeypatch.setattr(dispatch_mod.asyncio, "sleep", _no_sleep)

    audio = await _dispatched_commands(
        dispatch_mod,
        "execute",
        {
            "command": "export XDG_RUNTIME_DIR=/run/user/1000; "
            "pulseaudio --start 2>/dev/null || true; "
            "pactl set-sink-mute @DEFAULT_SINK@ 0"
        },
    )
    assert "XDG_RUNTIME_DIR=/run/user/1000" in audio[0]

    dbus = await _dispatched_commands(
        dispatch_mod,
        "execute",
        {"command": "DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus' gnome-terminal"},
    )
    assert "unix:path=/run/user/1000/bus" in dbus[0]

    # The image really does provide that path (this is what makes passthrough correct).
    dockerfile = (_REPO / "lite/gym/sandbox/docker/Dockerfile.linux").read_text()
    assert 'mkdir -p "/run/user/${_uid}"' in dockerfile
    assert 'ln -sfn "/run/user/${_uid}" "/tmp/runtime-${USER}"' in dockerfile


@pytest.mark.asyncio
async def test_dispatch_execute_outputs_are_cache_bounded(tmp_path):
    from lite.gym.envs.lite.osworld.src.utils import dispatch as dispatch_mod

    class _Interface:
        async def run_command(self, command, *args, **kwargs):
            return SimpleNamespace(stdout="out", stderr="err", returncode=0)

    computer = SimpleNamespace(interface=_Interface())

    await dispatch_mod.dispatch_action(
        computer,
        {
            "type": "execute",
            "parameters": {
                "command": "echo ok",
                "stdout": "nested/stdout.txt",
                "stderr": "nested/stderr.txt",
            },
        },
        cache_dir=str(tmp_path),
    )
    assert (tmp_path / "nested/stdout.txt").read_text() == "out"
    assert (tmp_path / "nested/stderr.txt").read_text() == "err"

    with pytest.raises(ValueError, match="cache_dir"):
        await dispatch_mod.dispatch_action(
            computer,
            {
                "type": "execute",
                "parameters": {"command": "echo ok", "stdout": "../stdout.txt"},
            },
            cache_dir=str(tmp_path),
        )
    assert not (tmp_path.parent / "stdout.txt").exists()


def test_live_smoke_attaches_as_the_shipped_exec_user():
    """The lite.osworld live smokes must drive the SHIPPED session identity.

    ``5e29d622`` pinned ``exec_user="root"`` at three ``attach`` sites in
    ``test_lite_osworld_live_smoke.py``, so screenshot / input / clipboard / VS Code ran
    through a root session that still had ``HOME=/home/user`` — the state-poisoning
    shape ``f756eb9f`` reverted, and the one thing a smoke test must not do. The
    smoke module is marked ``live``, so nothing in the default suite would have
    caught it; this gate is deliberately plain source + attribute inspection so it
    runs everywhere.
    """
    from lite.gym.sandbox.base import SandboxBaseEnv
    from lite.gym.sandbox.exec_stdio.client import attach

    assert SandboxBaseEnv.EXEC_USER == "user"
    assert inspect.signature(attach).parameters["exec_user"].default == "user"
    smoke = (Path(__file__).resolve().parent / "test_lite_osworld_live_smoke.py").read_text()
    tree = ast.parse(smoke)
    overrides = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "attach"
        and any(kw.arg == "exec_user" for kw in node.keywords)
    ]
    assert overrides == [], (
        "the osworld live smoke overrides exec_user at lines "
        f"{overrides}; it must exercise the shipped desktop-user session"
    )


def test_getter_normalizer_applies_setup_symmetric_rewrites():
    """The eval-getter path (runner._normalize_osworld_command_config) must apply
    the SAME value-rewrites as the setup/oracle dispatch — OSWorld template vars
    and the chrome-profile redirect — exercised by calling the REAL functions (not
    this file's dispatch reimpl), so a future change to any shared rewrite is caught
    here instead of silently drifting from the getter path.
    """
    from lite.gym.envs.lite.osworld.src.eval.runner import (
        _normalize_osworld_command_config,
    )

    # template var → replaced (setup symmetry; {CLIENT_PASSWORD} → the sudo pw)
    out, _ = _normalize_osworld_command_config("echo {CLIENT_PASSWORD} | sudo -S true", True)
    assert "{CLIENT_PASSWORD}" not in out and "user" in out

    # chrome default profile path → the --user-data-dir Chrome actually uses
    out, _ = _normalize_osworld_command_config(
        "cat /home/user/.config/google-chrome/Default/Preferences", True
    )
    assert "/home/user/chrome-data/Default/Preferences" in out

    # a plain user-home getter (no template / chrome / session token) is unchanged
    assert _normalize_osworld_command_config("cat /home/user/foo.txt", True) == (
        "cat /home/user/foo.txt",
        True,
    )


def _chrome_wal_checkpoint_script() -> str:
    """The ``python3`` heredoc ``_flush_chrome_profile`` ships into the container."""
    from lite.gym.envs.lite.osworld.src.eval.runner import _flush_chrome_profile

    source = inspect.getsource(_flush_chrome_profile)
    after_marker = source.split("python3 - <<'PYEOF'", 1)[1]
    # drop the rest of the `python3 ...` line (its redirections) before the heredoc
    return after_marker.split("\n", 1)[1].split("PYEOF", 1)[0]


def _run_chrome_wal_checkpoint(profiles: list[Path]) -> None:
    """Run that heredoc HERE, with its hardcoded ``/home/user/...`` profile tuple
    retargeted at throwaway dirs. Executing production's own source is the point:
    a paraphrase in the test could not have caught the bug this pins."""
    tree = ast.parse(_chrome_wal_checkpoint_script())
    loop = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "profile"
    )
    loop.iter = ast.parse(repr([str(p) for p in profiles])).body[0].value
    ast.fix_missing_locations(tree)
    exec(compile(tree, "<chrome-wal-checkpoint>", "exec"), {})  # noqa: S102


def test_chrome_wal_checkpoint_never_fabricates_a_missing_store():
    """The checkpoint must not CREATE a profile store it was meant to flush.

    ``sqlite3.connect(path)`` opens rw-CREATE, so the original bare connect wrote a
    0-byte ``Cookies`` / ``History`` / ``Web Data`` / ``Login Data`` into any profile
    dir that exists — and ``/home/user/chrome-data/Default`` always exists. A getter
    that checks existence, or a cookie-deletion evaluator, then reads an empty table
    instead of seeing "absent", i.e. it can pass for the WRONG REASON.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        empty = Path(tmp) / "Default"
        empty.mkdir()
        _run_chrome_wal_checkpoint([empty, Path(tmp) / "does-not-exist"])
        assert sorted(p.name for p in empty.iterdir()) == []


def test_chrome_wal_checkpoint_still_truncates_a_real_wal_sidecar():
    """...and the non-creating open must not have disarmed the flush itself: an
    existing WAL-mode store is still checkpointed (its ``-wal`` sidecar truncated to
    0 bytes) with its rows intact. This is the false-negative the checkpoint exists
    for — a UI-driven "Delete browsing data" lands in the sidecar while the getters
    read the main ``.db`` (osworld_chrome_7b6c7e24).
    """
    import sqlite3
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        profile = Path(tmp) / "Default"
        profile.mkdir()
        store = profile / "History"
        held = sqlite3.connect(store)
        try:
            held.execute("PRAGMA journal_mode=WAL")
            held.execute("CREATE TABLE urls (url TEXT)")
            held.execute("INSERT INTO urls VALUES ('https://amazon.com')")
            held.commit()
            sidecar = store.with_name("History-wal")
            assert sidecar.stat().st_size > 0

            _run_chrome_wal_checkpoint([profile])

            assert sidecar.stat().st_size == 0
            assert held.execute("SELECT url FROM urls").fetchall() == [("https://amazon.com",)]
        finally:
            held.close()


def _iter_command_actions(value, *, path: str = "$"):
    if isinstance(value, dict):
        action_type = value.get("type")
        params = value.get("parameters")
        if (
            isinstance(action_type, str)
            and action_type in _ROUTED_ACTION_TYPES
            and isinstance(params, dict)
            and "command" in params
        ):
            yield path, value
        for key, child in value.items():
            yield from _iter_command_actions(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_command_actions(child, path=f"{path}[{index}]")


def test_eval_exclusions_match_upstream_parity_except_confirmed_quarantine():
    """lite.osworld eval excludes upstream ``infeasible``/``google_auth`` plus
    the two task-level quarantines confirmed by oracle validation.

    Keep this narrow: a new per-domain ORACLES ``exclude_reason`` must name a
    specific task id and canonical reason here, otherwise upstream parity has
    regressed silently.
    """
    from lite.gym.envs.lite.osworld.src.gen.eval import (
        chrome,
        gimp,
        libreoffice_calc,
        libreoffice_impress,
        libreoffice_writer,
        multi_apps,
        thunderbird,
        vlc,
        vs_code,
    )
    from lite.gym.envs.lite.osworld.src.gen.eval import (
        os as os_oracles,
    )

    allowed = {
        (
            "lite.gym.envs.lite.osworld.src.gen.eval.chrome",
            "2888b4e6-5b47-4b57-8bf5-c73827890774",
        ): "upstream_live_site_drift",
        (
            "lite.gym.envs.lite.osworld.src.gen.eval.chrome",
            "b4f95342-463e-4179-8c3f-193cd7241fb2",
        ): "trivial_pass:color_precheck",
        (
            "lite.gym.envs.lite.osworld.src.gen.eval.multi_apps",
            "2c1ebcd7-9c6d-4c9a-afad-900e381ecd5e",
        ): "upstream_generated_eval_bug",
        (
            "lite.gym.envs.lite.osworld.src.gen.eval.multi_apps",
            "36037439-2044-4b50-b9d1-875b5a332143",
        ): "upstream_live_site_drift",
    }

    seen = {}
    for mod in (
        chrome,
        multi_apps,
        os_oracles,
        thunderbird,
        vlc,
        libreoffice_calc,
        libreoffice_impress,
        libreoffice_writer,
        gimp,
        vs_code,
    ):
        for tid, entry in mod.ORACLES.items():
            reason = entry.get("exclude_reason")
            if reason:
                seen[(mod.__name__, tid)] = reason

    assert seen == allowed


# Closed action-type set. Every step in `metadata.config` and
# `metadata.others.oracle_actions` across every split must use one of these.
# Adding a new type is a deliberate dispatcher change — bump this set
# in the same commit.
_ALLOWED_ACTION_TYPES = frozenset(
    {
        "execute",
        "command",
        "launch",
        "open",
        "activate_window",
        "close_window",
        "download",
        "sleep",
        "chrome_open_tabs",
        "chrome_close_tabs",
        "update_browse_history",
        "googledrive",
        "login",
        "synth_push",
        "host_push",
        "left_click",
        "right_click",
        "double_click",
        "type_text",
        "key",
        "scroll",
        "mouse_move",
        "screenshot",
        "wait",
    }
)


class TestJsonlContract:
    """Schema + reproducibility + determinism gates for all 3 splits."""

    def test_eval_catalog_lock_scored_counts_are_public_owner(self):
        """Tracked lock owns the Lite.OSWorld public scored-count prose."""
        entry = json.loads((_DATA_DIR / "catalog.lock.json").read_text())["splits"]["eval"]

        assert entry["rows"] == 369
        assert entry["excluded_rows"] == 41
        assert entry["scored_rows"] == 328
        assert entry["exclude_reasons"] == {
            "google_auth": 8,
            "infeasible": 29,
            "trivial_pass:color_precheck": 1,
            "upstream_generated_eval_bug": 1,
            "upstream_live_site_drift": 2,
        }

    @_requires_catalogs
    @pytest.mark.parametrize("split", ["eval.jsonl", "train.synth.jsonl", "train.perturb.jsonl"])
    def test_every_row_has_oracle_and_evaluator(self, split):
        """Schema gate. Every row carries oracle_actions / evaluator / config,
        and every action.type is in the closed allowlist."""
        with open(_DATA_DIR / split) as f:
            for ln, line in enumerate(f, 1):
                r = json.loads(line)
                tid = r["task_id"]
                m = r["metadata"]
                others = m["others"]
                assert "oracle_actions" in others, (
                    f"{split}:{ln} {tid} missing others.oracle_actions"
                )
                assert isinstance(others["oracle_actions"], list), (
                    f"{split}:{ln} {tid} oracle_actions not a list"
                )
                assert "oracle_actions" not in m, (
                    f"{split}:{ln} {tid} has legacy top-level oracle_actions"
                )
                assert "oracle_after_postconfig" not in m, (
                    f"{split}:{ln} {tid} has legacy top-level oracle_after_postconfig"
                )
                assert m.get("evaluator"), f"{split}:{ln} {tid} missing evaluator"
                assert m.get("config") is not None, f"{split}:{ln} {tid} missing config"
                for s in m["config"] + others["oracle_actions"]:
                    t = s.get("type")
                    assert t in _ALLOWED_ACTION_TYPES, (
                        f"{split}:{ln} {tid} unknown action type {t!r}"
                    )

    @_requires_catalogs
    def test_catalog_lock_matches_generated_jsonl(self):
        """Reproducibility gate for eval/train generated catalogs."""
        lock = json.loads((_DATA_DIR / "catalog.lock.json").read_text())
        for split, entry in lock["splits"].items():
            path = _DATA_DIR / entry["path"]
            data = path.read_bytes()
            rows = 0
            excluded_rows = 0
            reason_counts: dict[str, int] = {}
            for line in data.splitlines():
                if not line.strip():
                    continue
                rows += 1
                row = json.loads(line)
                reason = (
                    row.get("metadata", {})
                    .get("others", {})
                    .get("exclude_reason")
                )
                if reason:
                    exclude_reasons.validate(reason)
                    excluded_rows += 1
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
            actual = hashlib.sha256(data).hexdigest()
            assert rows == entry["rows"], f"{split} row count changed"
            assert excluded_rows == entry["excluded_rows"], (
                f"{split} excluded row count changed"
            )
            assert rows - excluded_rows == entry["scored_rows"], (
                f"{split} scored row count changed"
            )
            assert dict(sorted(reason_counts.items())) == entry["exclude_reasons"], (
                f"{split} exclude_reason breakdown changed"
            )
            assert actual == entry["sha256"], (
                f"{entry['path']} bytes changed!\n"
                f"  pinned : {entry['sha256']}\n"
                f"  actual : {actual}\n"
                "  If intentional: regenerate JSONL and run "
                "scripts/utils/tasks.sh refresh-lock."
            )

    @_requires_catalogs
    def test_gimp_image_op_export_postconfig_matches_result_filename(self):
        """GIMP image-op postconfig must export to evaluator.result.path."""
        with open(_DATA_DIR / "train.perturb.jsonl") as f:
            for line in f:
                row = json.loads(line)
                task_id = row["task_id"]
                if not task_id.startswith("perturb_osworld_gimp_"):
                    continue
                evaluator = row["metadata"]["evaluator"]
                if evaluator.get("func") != "check_structure_sim":
                    continue
                postconfig = evaluator.get("postconfig") or []
                result_path = evaluator.get("result", {}).get("path", "")
                if not result_path:
                    continue
                result_stem = Path(result_path).stem
                write_targets: list[str] = []
                for step in postconfig:
                    cmd = step.get("parameters", {}).get("command")
                    cmd_text = " ".join(cmd) if isinstance(cmd, list) else str(cmd or "")
                    write_targets.extend(
                        match.group(2)
                        for match in re.finditer(
                            r"pyautogui\.write\((['\"])(.*?)\1\)",
                            cmd_text,
                        )
                    )
                if write_targets:
                    assert write_targets == [result_stem], (
                        f"{task_id} exports {write_targets} but evaluator reads {result_path}"
                    )

    @_requires_catalogs
    def test_writer_spacing_long_quote_variants_target_audited_ordinals(self):
        """Long-quote Writer spacing rows must target an audited ordinal.

        The generator emits the content-resolved ``long_idxs[<ord_pos>]`` oracle
        ONLY for the long-quote instruction variants, and only for bases with an
        audited ``_SPACING_LONG_QUOTE_LIMITS`` entry — ``ord_pos`` must stay
        below that limit or the oracle ``sys.exit(42)``s on a paragraph the base
        does not have (Writer spacing bug: 66399b0d's "first paragraph contains a long
        quotation" wording pointed at a 9-word body line).

        The current corpus emits ZERO such rows (``long_quote_limit`` gates them
        out for every base but 66399b0d, which never drew the variant), so the
        per-row properties below are latent. The count pin at the end is the
        part that bites today: any regeneration that brings long-quote rows back
        fails here and forces the ordinal audit to be re-reviewed.
        """
        import lite.gym.envs.lite.osworld.src.gen.train.perturb.libreoffice_writer as W

        # Re-derive the phrases the generator tags its long-quote variants with,
        # so a rename there breaks this test loudly instead of quietly making
        # the row scan below match nothing.
        phrases = (
            "long quotation",
            "long block quote",
            "long block of quoted material",
        )
        tagged = {v for v in W._SPACING_VARIANTS if any(p in v for p in phrases)}
        assert tagged and tagged == set(W._SPACING_LONG_QUOTE_VARIANTS), (
            "long-quote variant tagging changed — update `phrases` here"
        )

        long_idx_rows = 0
        long_quote_rows = 0
        with open(_DATA_DIR / "train.perturb.jsonl") as f:
            for line in f:
                row = json.loads(line)
                task_id = row["task_id"]
                if not task_id.startswith("perturb_osworld_libreoffice_writer_"):
                    continue
                config_text = "\n".join(
                    str(step.get("parameters", {}).get("command", ""))
                    for step in row["metadata"].get("config", [])
                )
                # `_effective_spacing` is the spacing oracle's preamble — it
                # scopes this to spacing rows, so a non-spacing op that happens
                # to mention a quotation cannot false-positive here.
                if "_effective_spacing" not in config_text:
                    continue
                ordinals = [
                    int(m.group(1)) for m in re.finditer(r"long_idxs\[(\d+)\]", config_text)
                ]
                is_long_quote = any(p in row["instruction"] for p in phrases)
                long_idx_rows += bool(ordinals)
                long_quote_rows += bool(is_long_quote)
                # Wording and oracle must agree: the long-quote wording asserts
                # the target *is* a long quotation, which only the content-
                # resolved oracle guarantees; conversely a content-resolved
                # oracle without that wording targets the wrong paragraph.
                assert bool(ordinals) == is_long_quote, (
                    f"{task_id}: long_idxs={ordinals} but long-quote wording={is_long_quote}"
                )
                short = task_id.split("_")[4]
                limit = W._SPACING_LONG_QUOTE_LIMITS.get(short)
                for ord_pos in ordinals:
                    assert limit is not None and ord_pos < limit, (
                        f"{task_id} targets long_idxs[{ord_pos}] but base "
                        f"{short} has an audited long-quote limit of {limit}"
                    )

        assert (long_idx_rows, long_quote_rows) == (0, 0), (
            "the corpus regained long-quote Writer spacing rows "
            f"(long_idxs={long_idx_rows}, long-quote instructions="
            f"{long_quote_rows}). The per-row assertions above just went live — "
            "re-review the audited limits, then update this pin."
        )

    def test_malformed_catalog_lock_defers_during_import_path(self, monkeypatch, tmp_path):
        from lite.gym.envs.lite.osworld import main as M
        from lite.gym.errors import EnvDepsMissingError

        bad_lock = tmp_path / "catalog.lock.json"
        bad_lock.write_text('{"splits": []}')
        monkeypatch.setattr(M, "_CATALOG_LOCK", bad_lock)

        M._register_tasks(raise_if_none=False)
        with pytest.raises(EnvDepsMissingError, match="invalid lite.osworld catalog lock"):
            M._register_tasks()

    def test_malformed_catalog_entry_defers_during_import_path(self, monkeypatch, tmp_path):
        from lite.gym.envs.lite.osworld import main as M
        from lite.gym.errors import EnvDepsMissingError

        bad_lock = tmp_path / "catalog.lock.json"
        bad_lock.write_text(
            json.dumps(
                {
                    "splits": {
                        "eval": {
                            "exclude_reasons": {},
                            "excluded_rows": 0,
                            "path": 1,
                            "rows": 0,
                            "scored_rows": 0,
                            "sha256": "x",
                        },
                        "train.synth": {
                            "exclude_reasons": {},
                            "excluded_rows": 0,
                            "path": "missing.jsonl",
                            "rows": 0,
                            "scored_rows": 0,
                            "sha256": "x",
                        },
                        "train.perturb": {
                            "exclude_reasons": {},
                            "excluded_rows": 0,
                            "path": "missing.jsonl",
                            "rows": 0,
                            "scored_rows": 0,
                            "sha256": "x",
                        },
                    }
                }
            )
        )
        monkeypatch.setattr(M, "_CATALOG_LOCK", bad_lock)

        M._register_tasks(raise_if_none=False)
        with pytest.raises(EnvDepsMissingError, match="bad path"):
            M._register_tasks()

    def test_docker_image_preflight_uses_selected_tag(self, monkeypatch):
        from lite.gym.envs.lite.osworld import main as M

        seen = {}

        def fake_image_for(env_id, tag=None):
            seen["env_id"] = env_id
            seen["tag"] = tag
            return object()

        monkeypatch.setattr("lite.gym.utils.backend.freshness.image_for", fake_image_for)
        monkeypatch.setattr(
            "lite.gym.utils.backend.docker.require_image_present",
            lambda image: seen.setdefault("checked", image),
        )

        M._check_docker_image("cua-lite/lite.osworld:mine")
        assert seen["env_id"] == "lite.osworld"
        assert seen["tag"] == "cua-lite/lite.osworld:mine"
        assert "checked" in seen

    def test_flat_image_override_updates_constructor_computer_config(self, monkeypatch):
        from lite.gym.envs.lite.osworld import main as M

        monkeypatch.setattr(M, "_check_desktop_env", lambda: None)

        env = M.LiteOsworldEnv(image="cua-lite/lite.osworld:mine")

        assert env._image == "cua-lite/lite.osworld:mine"
        assert env._computer_config["image"] == "cua-lite/lite.osworld:mine"
        assert M._COMPUTER_CONFIG["image"] == "cua-lite/lite.osworld:latest"

    def test_train_jsonl_idempotent(self):
        """Idempotency gate. Running the generate scripts reproduces both train JSONL files.

        Calls _generate_synth() and _generate_perturb() from the train __main__ module
        and compares task_ids + serialized JSON to the committed files. Fails if the
        generator produces a different set or order of tasks than what is committed.

        Data-gated: the synth generators stage files from the asset bundle
        (HF ``cua-lite/lite.osworld-assets``, seeded per-worktree by the env's
        install.sh) — skip on unseeded checkouts instead of hard-failing.
        """
        # Probe via src/utils/assets (safe): importing anything under gen/train/
        # synth executes module-level ``_stage_asset`` calls and hard-fails on
        # an unseeded checkout before any skip could fire.
        from lite.gym.envs.lite.osworld.src.utils.assets import asset_root

        root = asset_root()
        manifest = root / "MANIFEST.csv"
        seeded = manifest.is_file()
        if seeded:
            # Every manifest-listed asset must exist; a partial pulled cache can
            # have a manifest while still missing files needed by generated tasks.
            for line in manifest.read_text().splitlines()[1:]:
                rel = line.split(",")[0].strip()
                if rel and not rel.startswith("#") and not (root / rel).is_file():
                    seeded = False
                    break
        if not seeded:
            pytest.skip(f"lite.osworld synth asset bundle not seeded at {root}")

        from lite.gym.envs.lite.osworld.src.gen.train.__main__ import (
            _generate_perturb,
            _generate_synth,
        )

        for split, gen_fn, args in [
            ("train.synth.jsonl", _generate_synth, (None,)),
            ("train.perturb.jsonl", _generate_perturb, (None,)),
        ]:
            committed: list[dict] = []
            with open(_DATA_DIR / split) as f:
                for line in f:
                    committed.append(json.loads(line))

            generated = gen_fn(*args)

            committed_ids = [r["task_id"] for r in committed]
            generated_ids = [r["task_id"] for r in generated]

            missing = sorted(set(committed_ids) - set(generated_ids))
            extra = sorted(set(generated_ids) - set(committed_ids))
            assert not missing and not extra, (
                f"{split}: task_id mismatch\n"
                f"  missing from generator : {missing[:5]}\n"
                f"  extra from generator   : {extra[:5]}\n"
                f"  If intentional: regenerate JSONL and update sha256 in the same commit."
            )

            assert committed_ids == generated_ids, (
                f"{split}: task_id order differs between committed file and generator output.\n"
                f"  If intentional: regenerate JSONL and update sha256 in the same commit."
            )

            for c, g in zip(committed, generated):
                assert json.dumps(c, ensure_ascii=False) == json.dumps(g, ensure_ascii=False), (
                    f"{split}: content mismatch for task {c['task_id']}.\n"
                    f"  If intentional: regenerate JSONL and update sha256 in the same commit."
                )

    def test_synth_chrome_exclude_reason_passthrough(self):
        """Chrome Param.exclude_reason must survive template params into row metadata.

        The full idempotency test catches committed JSONL drift, but this focused
        check protects the lightweight adapter path that emits known live-site
        drift rows from the Chrome synth generator.
        """
        import lite.gym.envs.lite.osworld.src.gen.train.synth.chrome as chrome
        from lite.gym.envs.lite.osworld.src.gen.train.synth._utils import make_synth_row

        # Task-agnostic: find ANY chrome synth param that emits an
        # `upstream_live_site_drift` exclude_reason and assert it survives into row
        # metadata. (The rust navigate_to_target_url task this originally hard-coded
        # was un-excluded by the #155 #27 canonical-URL fix; coupling the mechanism
        # check to a specific task is brittle against future un-exclusions.)
        found = None
        for ft in chrome.FILE_TASKS:
            template = chrome._to_synth_template(ft)
            for i in range(10):
                try:
                    params = template.param_fn(i)
                except Exception:
                    continue
                if (
                    isinstance(params, dict)
                    and params.get("exclude_reason") == "upstream_live_site_drift"
                ):
                    found = (template, i, params)
                    break
            if found:
                break
        assert found is not None, "expected at least one chrome synth live-site-drift param"
        template, i, params = found
        row = make_synth_row(template, i, params)
        assert row["metadata"]["others"]["exclude_reason"] == "upstream_live_site_drift"


# =========================================================================
# Per-domain ORACLES sources
# =========================================================================
#
# After step 1, eval oracle_actions live in ``scripts/generate/eval/<domain>.py``
# instead of being hand-edited in eval.jsonl. These tests cover the extraction:
# every domain module loads, every key in each module is a 36-char OSWorld
# UUID, and the union of ORACLES entries reconstructs every populated
# oracle_actions list in the current eval.jsonl.

_OSWORLD_DOMAINS = (
    "chrome",
    "gimp",
    "libreoffice_calc",
    "libreoffice_impress",
    "libreoffice_writer",
    "multi_apps",
    "os",
    "thunderbird",
    "vlc",
    "vs_code",
)


class TestEvalOraclesSource:
    """Per-domain ORACLES dicts under scripts/generate/eval/<domain>.py."""

    @pytest.mark.parametrize("domain", _OSWORLD_DOMAINS)
    def test_domain_module_loads(self, domain):
        import importlib

        mod = importlib.import_module(f"lite.gym.envs.lite.osworld.src.gen.eval.{domain}")
        oracles = mod.ORACLES
        assert isinstance(oracles, dict), f"{domain}.ORACLES is not a dict"

    @pytest.mark.parametrize("domain", _OSWORLD_DOMAINS)
    def test_domain_keys_are_osworld_uuids(self, domain):
        import importlib

        mod = importlib.import_module(f"lite.gym.envs.lite.osworld.src.gen.eval.{domain}")
        for k in mod.ORACLES:
            assert isinstance(k, str) and len(k) == 36 and k.count("-") == 4, (
                f"{domain}.ORACLES key not a UUID: {k!r}"
            )

    @pytest.mark.parametrize("domain", _OSWORLD_DOMAINS)
    def test_domain_entries_have_known_keys(self, domain):
        """Each entry uses only the documented ORACLES schema keys."""
        import importlib

        mod = importlib.import_module(f"lite.gym.envs.lite.osworld.src.gen.eval.{domain}")
        allowed = {
            "actions",
            "after_postconfig",
            "exclude_reason",
            "evaluator",
            "evaluator_options",
            "config_append",
            "config_prepend",
            "config_override",
        }
        for tid, entry in mod.ORACLES.items():
            extra = set(entry) - allowed
            assert not extra, f"{domain}.ORACLES[{tid}] has unknown keys: {extra}"

    @_requires_catalogs
    def test_oracles_reconstruct_eval_jsonl(self):
        """Every raw generated curated field is faithfully captured in ORACLES.

        Walks each eval.jsonl row, finds the per-domain ORACLES entry, and
        proves the extraction is lossless across all curated channels:
        ``actions``, ``after_postconfig``, canonical ``exclude_reason``,
        and ``evaluator`` overrides. Exclude reasons added by the post-gen
        quarantine script are checked against that script's exact target set.
        """
        import importlib

        per_domain: dict[str, dict] = {}
        for d in _OSWORLD_DOMAINS:
            mod = importlib.import_module(f"lite.gym.envs.lite.osworld.src.gen.eval.{d}")
            per_domain[d] = mod.ORACLES

        with open(_DATA_DIR / "eval.jsonl") as f:
            for line in f:
                r = json.loads(line)
                m = r["metadata"]
                domain = m["others"]["domain"]
                oid = m["osworld_id"]
                entry = per_domain[domain].get(oid, {})
                actions = m["others"].get("oracle_actions") or []
                if actions:
                    assert entry.get("actions") == actions, (
                        f"oracle_actions mismatch for {domain}/{oid}: "
                        f"eval.jsonl has {len(actions)} steps, ORACLES has "
                        f"{len(entry.get('actions', []))} steps"
                    )
                if m["others"].get("oracle_after_postconfig"):
                    assert entry.get("after_postconfig") is True, (
                        f"after_postconfig mismatch for {domain}/{oid}"
                    )
                er = m["others"].get("exclude_reason")
                if er and er not in {"infeasible", "google_auth"}:
                    exclude_reasons.validate(er)
                    # Every eval exclusion lives in its ORACLES source (folded;
                    # no post-generation quarantine step) -> the generator entry
                    # carries the exclude_reason directly.
                    assert entry.get("exclude_reason") == er, (
                        f"exclude_reason mismatch for {domain}/{oid}"
                    )

    @_requires_catalogs
    def test_convert_all_byte_identical_to_eval_jsonl(self):
        """Run the eval generator end-to-end and assert byte equality.

        Catches drift in: ORACLES → convert_task → post-generation quarantine
        → JSON serialization. Faster than waiting for the byte-lock test to
        fail at commit time. Skipped if the upstream OSWorld task package isn't
        installed (e.g. CI without the optional `lite[osworld]` extras).
        """
        try:
            from lite.gym.envs.lite.osworld.src.gen.eval.__main__ import (
                _find_examples_dir,
                convert_all,
            )
        except ImportError:
            pytest.skip("eval gen module not importable")
        try:
            examples_dir = _find_examples_dir()
        except FileNotFoundError:
            pytest.skip("OSWorld examples package not installed")

        # No post-generation quarantine step: every eval exclusion is folded
        # into its ORACLES source (e.g. chrome fc6d8143 exclude_reason), so raw
        # convert_all() reproduces eval.jsonl byte-for-byte. The reproducible
        # pipeline does NOT depend on devs/separation-wip/.
        tasks = convert_all(examples_dir)
        regen_lines = [json.dumps(t, ensure_ascii=False) + "\n" for t in tasks]
        regen_bytes = "".join(regen_lines).encode()
        actual_bytes = (_DATA_DIR / "eval.jsonl").read_bytes()
        assert regen_bytes == actual_bytes, (
            "convert_all() output does not match eval.jsonl byte-for-byte. "
            "Either the generator drifted, an ORACLES entry diverged, or "
            "the upstream OSWorld task JSONs changed. First differing line "
            f"= {next((
                i
                for i, (a, b) in enumerate(
                    zip(regen_bytes.splitlines(), actual_bytes.splitlines())
                )
                if a != b
            ), 'EOF')}"
        )


class TestLibreOfficeConfigFlush:
    """#154 read-before-flush -> LibreOffice registry config (osworld intra-env).

    LibreOffice commits registrymodifications.xcu only on a clean shutdown, so a
    bare vm_file download of that path while soffice.bin is still running reads a
    stale xcu -> false-negative. runner._get_result must flush (graceful ctrl+q
    -> SIGTERM soffice.bin, NEVER -9 -> sync) BEFORE the download. Pure-unit: a
    fake computer whose read_bytes returns the STALE xcu until a soffice.bin
    flush command runs, then the FRESH xcu (mirroring the clean-shutdown commit).
    """

    @staticmethod
    def _fake(stale: bytes, fresh: bytes):
        state = {"flushed": False, "commands": []}

        class _Iface:
            async def run_command(self, command, timeout=None):
                state["commands"].append(command)
                if "soffice.bin" in command:
                    state["flushed"] = True
                return SimpleNamespace(stdout="", stderr="", returncode=0)

            async def read_bytes(self, path, offset=0, length=None):
                return fresh if state["flushed"] else stale

        return SimpleNamespace(interface=_Iface()), state

    @pytest.mark.asyncio
    async def test_registry_read_flushes_before_download(self, tmp_path):
        """0->1: without the flush the download is the STALE xcu; the fix
        SIGTERMs soffice.bin BEFORE read_bytes, so the FRESH xcu is returned."""
        from lite.gym.envs.lite.osworld.src.eval.runner import _get_result

        fresh = b"<xcu>font=Arial</xcu>"
        computer, state = self._fake(b"<xcu></xcu>", fresh)
        local = await _get_result(
            computer,
            {
                "type": "vm_file",
                "path": "/home/user/.config/libreoffice/4/user/registrymodifications.xcu",
                "dest": "registrymodifications.xcu",
            },
            str(tmp_path),
        )
        assert state["flushed"] is True
        assert Path(local).read_bytes() == fresh
        joined = "\n".join(state["commands"])
        # NEVER SIGKILL soffice (that skips the registry-save handler).
        assert "pkill -TERM soffice.bin" in joined
        assert not any(
            bad in joined for bad in ("pkill -9", "kill -9", "killall -9", "-KILL", "SIGKILL")
        )

    @pytest.mark.asyncio
    async def test_non_libreoffice_vm_file_does_not_flush(self, tmp_path):
        """Negative direction: a plain (non-config) download must NOT flush."""
        from lite.gym.envs.lite.osworld.src.eval.runner import _get_result

        computer, state = self._fake(b"DOC", b"DOC")
        await _get_result(
            computer,
            {"type": "vm_file", "path": "/home/user/notes.txt", "dest": "notes.txt"},
            str(tmp_path),
        )
        assert state["flushed"] is False
        assert not any("soffice" in c for c in state["commands"])

    def test_mentions_libreoffice_config_file_gate(self):
        """File-marker gate covers the 4 osworld registrymodifications tasks and
        excludes .docx/.odt/.xlsx document reads (no-overfit anchor)."""
        from lite.gym.envs.lite.osworld.src.eval.runner import (
            _mentions_libreoffice_config_file,
        )

        # The exact result config shared by all 4 osworld LO-registry tasks
        # (check_presenter_console_disable / check_auto_saving_time /
        # check_left_panel / find_default_font).
        assert _mentions_libreoffice_config_file(
            {
                "type": "vm_file",
                "path": "/home/user/.config/libreoffice/4/user/registrymodifications.xcu",
                "dest": "registrymodifications.xcu",
            }
        )
        # Document getters read the saved file, not the registry -> no flush.
        assert not _mentions_libreoffice_config_file(
            {"type": "vm_file", "path": "/home/user/report.docx"}
        )
        assert not _mentions_libreoffice_config_file(
            {"type": "vm_file", "path": "/home/user/report.odt"}
        )
        assert not _mentions_libreoffice_config_file(
            {"type": "vm_file", "path": "/home/user/sheet.xlsx"}
        )
