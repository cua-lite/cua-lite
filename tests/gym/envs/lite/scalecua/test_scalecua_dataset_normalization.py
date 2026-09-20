"""ScaleCUA dataset normalization and import-repair tests."""

from __future__ import annotations

import json
import struct
import subprocess

import pytest

from lite.gym.envs.lite.osworld.src.gen.common import LO_SAVE_POSTCONFIG
from lite.gym.envs.lite.scalecua.src.osworld import verify as scalecua_verify
from lite.gym.envs.lite.scalecua.src.utils import dataset


def test_scalecua_official_parity_types_do_not_route_to_base_runner():
    high_risk_result_types = {
        "active_url_from_accessTree",
        "active_tab_info",
        "active_tab_url_parse",
        "active_tab_html_parse",
        "open_tabs_info",
        "url_dashPart",
        "vlc_playing_info",
        "new_startup_page",
        "data_delete_automacally",
        "enable_do_not_track",
        "vm_file",
    }

    assert not high_risk_result_types & scalecua_verify.BASE_RUNNER_RESULT_TYPES
    assert high_risk_result_types <= scalecua_verify.SCALECUA_LOCAL_RESULT_TYPES
    assert "vm_file" not in scalecua_verify.BASE_RUNNER_EXPECTED_TYPES
    assert "vm_file" in scalecua_verify.SCALECUA_LOCAL_EXPECTED_TYPES


def test_scalecua_hoists_generated_ignore_list_order_rule_flag():
    raw_rules = {
        "expected": {
            "modelList": ["iphone-15", "iphone-14", "iphone-13"],
            "ignore_list_order": True,
        }
    }

    fixed = scalecua_verify._normalize_scalecua_expected_rules(
        "check_direct_json_object", raw_rules
    )

    assert fixed == {
        "expected": {"modelList": ["iphone-15", "iphone-14", "iphone-13"]},
        "ignore_list_order": True,
    }
    assert raw_rules["expected"]["ignore_list_order"] is True


def test_scalecua_import_reuses_lite_osworld_lo_save_postconfig(tmp_path):
    payload = {
        "evaluator": {
            "postconfig": [
                {
                    "type": "activate_window",
                    "parameters": {
                        "strict": True,
                        "window_name": "214_9.pptx - LibreOffice Impress",
                    },
                },
                {"type": "sleep", "parameters": {"seconds": 0.5}},
                {
                    "type": "execute",
                    "parameters": {
                        "command": [
                            "python",
                            "-c",
                            "import pyautogui; pyautogui.hotkey('ctrl', 's')",
                        ]
                    },
                },
                {"type": "sleep", "parameters": {"seconds": 0.5}},
            ]
        }
    }

    normalized, unsupported = dataset._normalize_runtime_payload(
        payload,
        runtime_split="train",
        source_domain="libreoffice_impress",
        context=dataset._ImportContext(snapshot=tmp_path, broken_metrics=frozenset()),
    )

    assert unsupported == []
    postconfig = normalized["evaluator"]["postconfig"]
    assert postconfig[0] == {
        "type": "activate_window",
        "parameters": {"window_name": "LibreOffice"},
    }
    postconfig_text = json.dumps(postconfig)
    assert "ctrl+s" in postconfig_text
    # The slimmed LO-save postconfig is pure keystrokes: ctrl+s, then the File→Save
    # menubar accelerator (alt+f, s) to rescue a ctrl+s swallowed by a focused inner
    # widget — no literal dialog-title/button text.
    assert "alt+f" in postconfig_text
    assert "214_9.pptx - LibreOffice Impress" not in postconfig_text


@pytest.mark.parametrize(
    "key_params",
    [
        {"key": "ctrl+s"},
        {"keys": ["ctrl", "s"]},
    ],
)
def test_scalecua_import_detects_lo_save_postconfig_key_shapes(
    tmp_path,
    key_params,
):
    payload = {
        "evaluator": {
            "postconfig": [
                {
                    "type": "activate_window",
                    "parameters": {"window_name": "LibreOffice Writer"},
                },
                {"type": "key", "parameters": key_params},
                {"type": "sleep", "parameters": {"seconds": 0.5}},
            ]
        }
    }

    normalized, unsupported = dataset._normalize_runtime_payload(
        payload,
        runtime_split="train",
        source_domain="libreoffice_writer",
        context=dataset._ImportContext(snapshot=tmp_path, broken_metrics=frozenset()),
    )

    assert unsupported == []
    assert normalized["evaluator"]["postconfig"] == LO_SAVE_POSTCONFIG


def test_scalecua_import_source_preserves_key_list_boundaries(tmp_path):
    normalized, unsupported = dataset._normalize_runtime_payload(
        {
            "evaluator": {
                "postconfig": [
                    {"type": "key", "parameters": {"keys": ["ctrl", "s"]}},
                ]
            }
        },
        runtime_split="train",
        source_domain="vs_code",
        context=dataset._ImportContext(snapshot=tmp_path, broken_metrics=frozenset()),
    )

    assert unsupported == []
    assert normalized["evaluator"]["postconfig"] == [
        {"type": "key", "parameters": {"keys": ["ctrl", "s"]}},
    ]


def test_scalecua_import_does_not_repair_key_press_singleton_chords(tmp_path):
    normalized, unsupported = dataset._normalize_runtime_payload(
        {
            "evaluator": {
                "postconfig": [
                    {"type": "key_press", "parameters": {"keys": ["enter"]}},
                    {"type": "key_press", "parameters": {"keys": ["ctrl+s"]}},
                ]
            }
        },
        runtime_split="train",
        source_domain="vs_code",
        context=dataset._ImportContext(snapshot=tmp_path, broken_metrics=frozenset()),
    )

    assert unsupported == []
    assert normalized["evaluator"]["postconfig"] == [
        {"type": "key", "parameters": {"key": "enter"}},
        {"type": "key", "parameters": {"keys": ["ctrl+s"]}},
    ]


@pytest.mark.parametrize(
    (
        "source_rel",
        "source_domain",
        "payload",
        "expected_postconfig",
    ),
    [
        (
            "osworld/generated_tasks/vs_code/26150609-0da3-4a7d-8868-0faf9c5f01bb_task_verify_23.json",
            "vs_code",
            {
                "id": "26150609-0da3-4a7d-8868-0faf9c5f01bb",
                "instruction": "Save the file.",
                "evaluator": {
                    "postconfig": [
                        {"type": "key", "parameters": {"keys": ["ctrl+s"]}},
                        {"type": "sleep", "parameters": {"seconds": 0.5}},
                    ]
                },
            },
            [
                {"type": "key", "parameters": {"key": "ctrl+s"}},
                {"type": "sleep", "parameters": {"seconds": 0.5}},
            ],
        ),
        (
            "osworld/generated_tasks/gimp/a746add2-cab0-4740-ac36-c3769d9bfb46_task_verify_79.json",
            "gimp",
            {
                "id": "a746add2-cab0-4740-ac36-c3769d9bfb46",
                "instruction": "Quit GIMP cleanly.",
                "evaluator": {
                    "postconfig": [
                        {"type": "sleep", "parameters": {"seconds": 2}},
                        {"type": "key_press", "parameters": {"keys": ["ctrl+q"]}},
                        {"type": "sleep", "parameters": {"seconds": 3}},
                    ]
                },
            },
            [
                {"type": "sleep", "parameters": {"seconds": 2}},
                {"type": "key", "parameters": {"key": "ctrl+q"}},
                {"type": "sleep", "parameters": {"seconds": 3}},
            ],
        ),
    ],
)
def test_scalecua_import_repairs_known_upstream_chord_key_list_sources(
    tmp_path,
    source_rel,
    source_domain,
    payload,
    expected_postconfig,
):
    context = dataset._ImportContext(snapshot=tmp_path, broken_metrics=frozenset())
    source_path = tmp_path / "hf_snapshot" / source_rel

    row = dataset._row_from_payload(
        payload,
        runtime_split="train",
        source_name="generated_tasks",
        source_domain=source_domain,
        source_path=source_path,
        inherited_exclusion=None,
        context=context,
    )

    assert row["metadata"]["evaluator"]["postconfig"] == expected_postconfig
    assert context.normalization_notes["repair_known_upstream_chord_key_list"] == 1


def test_scalecua_import_repairs_known_upstream_expected_text_literal_newlines(
    tmp_path,
):
    source_rel = (
        "osworld/generated_tasks/vs_code/"
        "f918266a-b3e0-4914-865d-4faa564f1aef_task_verify_60.json"
    )
    payload = {
        "id": "f918266a-b3e0-4914-865d-4faa564f1aef",
        "instruction": "Run calculator.py and save sorted output.",
        "evaluator": {
            "func": "compare_text_output__d398cd07",
            "expected": {
                "type": "rule",
                "rules": {"expected_text": "11\\n12\\n22\\n25\\n34\\n64\\n90\\n"},
            },
            "result": {
                "type": "vm_file",
                "path": "/home/user/Desktop/log.txt",
                "dest": "log.txt",
            },
            "postconfig": [],
        },
    }
    context = dataset._ImportContext(snapshot=tmp_path, broken_metrics=frozenset())

    row = dataset._row_from_payload(
        payload,
        runtime_split="train",
        source_name="generated_tasks",
        source_domain="vs_code",
        source_path=tmp_path / "hf_snapshot" / source_rel,
        inherited_exclusion=None,
        context=context,
    )

    rules = row["metadata"]["evaluator"]["expected"]["rules"]
    assert rules["expected_text"] == "11\n12\n22\n25\n34\n64\n90\n"
    assert context.normalization_notes["repair_known_upstream_expected_text"] == 1


def test_scalecua_import_preserves_legacy_top_level_action_parameters(tmp_path):
    payload = {
        "config": [
            {
                "type": "execute",
                "command": ["code", "--install-extension", "/home/user/test.vsix"],
            }
        ],
        "evaluator": {
            "postconfig": [
                {"type": "key", "key": "ctrl+s"},
                {"type": "sleep", "seconds": 2},
            ]
        },
    }

    context = dataset._ImportContext(snapshot=tmp_path, broken_metrics=frozenset())
    normalized, unsupported = dataset._normalize_runtime_payload(
        payload,
        runtime_split="rl",
        source_domain="chrome",
        context=context,
    )

    assert unsupported == []
    assert normalized["config"] == [
        {
            "type": "execute",
            "parameters": {"command": ["code", "--install-extension", "/home/user/test.vsix"]},
        }
    ]
    assert normalized["evaluator"]["postconfig"] == [
        {"type": "key", "parameters": {"key": "ctrl+s"}},
        {"type": "sleep", "parameters": {"seconds": 2}},
    ]
    assert context.normalization_notes["legacy_top_level_action_parameters"] == 3


def test_scalecua_import_reuses_lite_osworld_gimp_export_postconfig(tmp_path):
    payload = {
        "evaluator": {
            "postconfig": [
                {
                    "type": "execute",
                    "parameters": {
                        "command": [
                            "python3",
                            "-c",
                            'import pyautogui; pyautogui.hotkey(["shift", "ctrl", "e"]);',
                        ]
                    },
                },
                {"type": "sleep", "parameters": {"seconds": 0.5}},
                {
                    "type": "execute",
                    "parameters": {
                        "command": [
                            "python3",
                            "-c",
                            (
                                "import pyautogui;"
                                "pyautogui.write('Triangle_Blue');"
                                "pyautogui.press(['enter'])"
                            ),
                        ]
                    },
                },
            ],
            "result": {
                "type": "vm_file",
                "path": "/home/user/Desktop/Triangle_Blue.png",
            },
        }
    }

    normalized, unsupported = dataset._normalize_runtime_payload(
        payload,
        runtime_split="train",
        source_domain="gimp",
        context=dataset._ImportContext(snapshot=tmp_path, broken_metrics=frozenset()),
    )

    assert unsupported == []
    postconfig = normalized["evaluator"]["postconfig"]
    postconfig_text = json.dumps(postconfig)
    assert postconfig[0] == {
        "type": "activate_window",
        "parameters": {"window_name": "Gimp", "by_class": True},
    }
    assert {"type": "key", "parameters": {"key": "shift+ctrl+e"}} in postconfig
    assert "/home/user/Desktop/Triangle_Blue.png" in postconfig_text
    assert "pyautogui.hotkey" not in postconfig_text
    assert "already exists" in postconfig_text


def test_scalecua_import_uses_gimp_expected_target_path_for_export_postconfig(tmp_path):
    payload = {
        "evaluator": {
            "postconfig": [
                {
                    "type": "execute",
                    "parameters": {
                        "command": [
                            "python3",
                            "-c",
                            'import pyautogui; pyautogui.hotkey(["shift", "ctrl", "e"]);',
                        ]
                    },
                },
                {"type": "sleep", "parameters": {"seconds": 0.5}},
                {
                    "type": "execute",
                    "parameters": {
                        "command": [
                            "python3",
                            "-c",
                            (
                                "import pyautogui;"
                                "pyautogui.write('red_background_with_object');"
                                "pyautogui.press(['enter'])"
                            ),
                        ]
                    },
                },
            ],
            "result": {"type": "rule", "rules": {}},
            "expected": {
                "type": "rule",
                "rules": {
                    "src_path": "/home/user/Desktop/white_background_with_object.png",
                    "tgt_path": "/home/user/Desktop/red_background_with_object.png",
                },
            },
        }
    }

    normalized, unsupported = dataset._normalize_runtime_payload(
        payload,
        runtime_split="train",
        source_domain="gimp",
        context=dataset._ImportContext(snapshot=tmp_path, broken_metrics=frozenset()),
    )

    assert unsupported == []
    postconfig = normalized["evaluator"]["postconfig"]
    postconfig_text = json.dumps(postconfig)
    assert {"type": "key", "parameters": {"key": "shift+ctrl+e"}} in postconfig
    assert "/home/user/Desktop/red_background_with_object.png" in postconfig_text
    assert "pyautogui.hotkey" not in postconfig_text
    assert "red_background_with_object');" not in postconfig_text


def test_scalecua_import_normalizes_vscode_builtin_theme_aliases(tmp_path):
    payload = {
        "evaluator": {
            "expected": {
                "type": "rule",
                "rules": {
                    "expected": {
                        "workbench.colorTheme": "Light+ (default light)",
                        "editor.fontSize": 14,
                    }
                },
            }
        }
    }
    context = dataset._ImportContext(snapshot=tmp_path, broken_metrics=frozenset())

    normalized, unsupported = dataset._normalize_runtime_payload(
        payload,
        runtime_split="train",
        source_domain="vs_code",
        context=context,
    )

    assert unsupported == []
    expected = normalized["evaluator"]["expected"]["rules"]["expected"]
    assert expected == {
        "workbench.colorTheme": "Default Light+",
        "editor.fontSize": 14,
    }
    assert context.normalization_notes["vscode_theme_expected_alias"] == 1


def test_scalecua_runtime_normalizes_vscode_builtin_theme_aliases():
    expected = {
        "expected": {
            "workbench.colorTheme": "Dark+ (default dark)",
            "editor.wordWrap": "on",
        }
    }

    normalized = scalecua_verify._normalize_scalecua_expected_rules(
        "check_json_settings",
        expected,
    )

    assert normalized == {
        "expected": {
            "workbench.colorTheme": "Default Dark+",
            "editor.wordWrap": "on",
        }
    }


def test_scalecua_import_does_not_replace_non_export_gimp_postconfig(tmp_path):
    payload = {
        "evaluator": {
            "postconfig": [
                {
                    "type": "execute",
                    "parameters": {
                        "command": [
                            "python3",
                            "-c",
                            'import pyautogui; pyautogui.hotkey(["ctrl", "q"]);',
                        ]
                    },
                }
            ],
            "result": {
                "type": "vm_file",
                "path": "/home/user/.config/GIMP/2.10/gimprc",
            },
        }
    }

    normalized, unsupported = dataset._normalize_runtime_payload(
        payload,
        runtime_split="train",
        source_domain="gimp",
        context=dataset._ImportContext(snapshot=tmp_path, broken_metrics=frozenset()),
    )

    assert unsupported == []
    postconfig = normalized["evaluator"]["postconfig"]
    postconfig_text = json.dumps(postconfig)
    assert "pyautogui.hotkey" in postconfig_text
    assert "shift+ctrl+e" not in postconfig_text


def test_scalecua_row_uses_canonical_domain_but_stable_source_task_id(tmp_path):
    payload = {
        "id": "2c1ebcd7-9c6d-4c9a-afad-900e381ecd5e",
        "instruction": "Bold all headings in the document.",
        "config": [],
        "evaluator": {},
    }

    row = dataset._row_from_payload(
        payload,
        runtime_split="train",
        source_name="generated_tasks",
        source_domain="libreoffice_calc",
        canonical_domain="multi_apps",
        source_path=(tmp_path / "2c1ebcd7-9c6d-4c9a-afad-900e381ecd5e_task_verify_20.json"),
        inherited_exclusion=None,
        context=dataset._ImportContext(snapshot=tmp_path, broken_metrics=frozenset()),
    )

    assert row["task_id"] == (
        "scalecua_osworld_train_libreoffice_calc_"
        "2c1ebcd7_9c6d_4c9a_afad_900e381ecd5e_task_verify_20"
    )
    assert row["metadata"]["others"]["domain"] == "multi_apps"
    assert row["metadata"]["others"]["source_domain"] == "libreoffice_calc"


def test_scalecua_drops_stale_a462_setup_and_keeps_runnable_rl_variants(tmp_path):
    assert len(dataset.INSTRUCTION_SETUP_MISMATCH_TASK_IDS) == 4
    stale_setup = {
        "type": "execute",
        "parameters": {
            "command": "echo {CLIENT_PASSWORD} | sudo -S su - charles",
            "shell": True,
        },
    }
    focus_click = {
        "type": "execute",
        "parameters": {
            "command": [
                "python",
                "-c",
                "import pyautogui; pyautogui.click({SCREEN_WIDTH_HALF}, {SCREEN_HEIGHT_HALF})",
            ],
        },
    }
    payload = {
        "id": "a462a795-fdc7-4b23-b689-e8b6df786b78",
        "instruction": "Change the hostname of this machine to `dev-workstation`.",
        "config": [stale_setup, focus_click],
        "evaluator": {"func": "exact_match"},
    }
    context = dataset._ImportContext(snapshot=tmp_path, broken_metrics=frozenset())

    normalized, unsupported = dataset._normalize_runtime_payload(
        payload,
        runtime_split="rl",
        source_domain="os",
        context=context,
    )

    assert unsupported == []
    assert normalized["config"] == [focus_click]
    assert context.normalization_notes["drop_stale_rl_a462_charles_setup"] == 1
    assert (
        dataset._exclude_reason(
            payload,
            task_id=(
                "scalecua_osworld_train_os_a462a795_fdc7_4b23_b689_e8b6df786b78_task_verify_23"
            ),
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        == "instruction_setup_mismatch"
    )
    assert (
        dataset._exclude_reason(
            payload,
            task_id=("scalecua_osworld_rl_os_a462a795_fdc7_4b23_b689_e8b6df786b78_traj_verify_0"),
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="rl",
            broken_metrics=frozenset(),
        )
        == "upstream_generated_eval_bug"
    )
    assert (
        dataset._exclude_reason(
            payload,
            task_id=("scalecua_osworld_rl_os_a462a795_fdc7_4b23_b689_e8b6df786b78_traj_verify_2"),
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="rl",
            broken_metrics=frozenset(),
        )
        is None
    )


def test_scalecua_repairs_root_home_test1_setup_without_excluding_rl(tmp_path):
    payload = {
        "id": "5812b315-e7bd-4265-b51f-863c02174c28",
        "instruction": (
            'Please create an SSH user named "charles" with password '
            '"Ex@mpleP@55w0rd!" on Ubuntu who is only allowed to access the '
            'folder "/home/test1".'
        ),
        "config": [
            {
                "type": "execute",
                "parameters": {"command": "mkdir /home/test1", "shell": True},
            }
        ],
        "evaluator": {"func": "exact_match"},
    }
    context = dataset._ImportContext(snapshot=tmp_path, broken_metrics=frozenset())

    normalized, unsupported = dataset._normalize_runtime_payload(
        payload,
        runtime_split="rl",
        source_domain="os",
        context=context,
    )

    assert unsupported == []
    assert normalized["config"] == [
        {
            "type": "execute",
            "parameters": {
                "command": ("printf '%s\\n' {CLIENT_PASSWORD} | sudo -S mkdir -p /home/test1"),
                "shell": True,
            },
        }
    ]
    assert context.normalization_notes["repair_root_home_test1_setup"] == 1
    assert (
        dataset._exclude_reason(
            payload,
            task_id=("scalecua_osworld_rl_os_5812b315_e7bd_4265_b51f_863c02174c28_traj_verify_0"),
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="rl",
            broken_metrics=frozenset(),
        )
        is None
    )


def test_scalecua_normalizes_bare_update_desktop_database_setup(tmp_path):
    payload = {
        "id": "48c46dc7-fe04-4505-ade7-723cba1aa6f6",
        "instruction": "Open GitHub homepage and Python documentation in separate Chrome tabs.",
        "config": [
            {
                "type": "execute",
                "parameters": {"command": ["update-desktop-database"]},
            }
        ],
        "evaluator": {"func": "exact_match"},
    }
    context = dataset._ImportContext(snapshot=tmp_path, broken_metrics=frozenset())

    normalized, unsupported = dataset._normalize_runtime_payload(
        payload,
        runtime_split="rl",
        source_domain="multi_apps",
        context=context,
    )

    assert unsupported == []
    assert normalized["config"] == [
        {
            "type": "execute",
            "parameters": {
                "command": (
                    "mkdir -p /home/user/.local/share/applications && "
                    "update-desktop-database /home/user/.local/share/applications "
                    "2>/dev/null || true"
                ),
                "shell": True,
            },
        }
    ]
    assert context.normalization_notes["normalize_update_desktop_database"] == 1


def test_scalecua_rewrites_external_placeholder_download_to_local_png(tmp_path):
    out_path = tmp_path / "photo.png"
    payload = {
        "id": "e19bd559-633b-4b02-940f-d946248f088e",
        "instruction": "Resize the desktop photo to 800x600 pixels while maintaining aspect ratio.",
        "config": [
            {
                "type": "download",
                "parameters": {
                    "files": [
                        {
                            "path": str(out_path),
                            "url": dataset.PLACEHOLDER_1024_URL,
                        }
                    ]
                },
            }
        ],
        "evaluator": {"func": "check_image_size__e19bd559"},
    }
    context = dataset._ImportContext(snapshot=tmp_path, broken_metrics=frozenset())

    normalized, unsupported = dataset._normalize_runtime_payload(
        payload,
        runtime_split="train",
        source_domain="gimp",
        context=context,
    )

    assert unsupported == []
    assert normalized["config"][0]["type"] == "execute"
    assert dataset.PLACEHOLDER_1024_URL not in normalized["config"][0]["parameters"]["command"]
    assert context.normalization_notes["local_placeholder_image_download"] == 1

    subprocess.run(
        normalized["config"][0]["parameters"]["command"],
        shell=True,
        check=True,
    )
    data = out_path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack(">II", data[16:24]) == (1024, 768)


def test_scalecua_repairs_literal_newline_heredoc_setup():
    # Upstream generated_tasks defect: a `cat > file << 'PYEOF'\ndef ...` setup was
    # emitted with a LITERAL \n after the delimiter, so the heredoc never closes.
    # Restoring real newlines is a safe, exact repair (the command is a pure write).
    broken = {
        "shell": True,
        "command": (
            "cat > /home/user/Desktop/calculator.py << 'PYEOF'\\ndef f():\\n    return 1\\nPYEOF"
        ),
    }
    repaired, changed = dataset._repair_upstream_heredoc_command(broken)
    assert changed
    cmd = repaired["command"]
    assert "\\n" not in cmd and "\n" in cmd
    import re as _re

    assert _re.search(r"(?m)^PYEOF\s*$", cmd)  # delimiter now closes on its own line
    # argv-list form is repaired element-wise.
    broken_list = {
        "command": ["/bin/bash", "-c", "cat > /tmp/a << 'E'\\nhi\\nE"],
    }
    repaired_list, changed_list = dataset._repair_upstream_heredoc_command(broken_list)
    assert changed_list and "\n" in repaired_list["command"][2]
    # A legitimate `printf '%s\n'` (literal \n is intentional, no heredoc) is untouched.
    good = {"shell": True, "command": "printf '%s\\n' '11' '12' > /home/user/a.txt"}
    unchanged, changed_none = dataset._repair_upstream_heredoc_command(good)
    assert not changed_none and unchanged is good
