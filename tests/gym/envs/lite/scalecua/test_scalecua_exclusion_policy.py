"""ScaleCUA dataset exclusion policy tests."""

from __future__ import annotations

import copy

from lite.gym.envs.lite.scalecua.src.utils import dataset


def test_scalecua_excludes_tell_explain_gimp_action_history_mismatch():
    evaluator = {
        "func": "check_include_exclude",
        "result": {
            "type": "vm_command_line",
            "command": "cat /home/user/.config/GIMP/2.10/action-history",
        },
        "expected": {
            "type": "rule",
            "rules": {"include": ["filters-lens-distortion"]},
        },
    }

    tell_payload = {
        "instruction": "Tell me how to bring up the Lens Distortion filter dialog box.",
        "evaluator": evaluator,
    }
    show_payload = {
        "instruction": "Show me how to access the Lens Distortion filter dialog.",
        "evaluator": evaluator,
    }
    guide_payload = {
        "instruction": "Guide me to access the Gaussian Blur filter window.",
        "evaluator": evaluator,
    }
    instruct_payload = {
        "instruction": "Instruct me on launching the Brightness-Contrast adjustment panel.",
        "evaluator": evaluator,
    }
    action_payload = {
        "instruction": "Help me open up the Lens Distortion filter dialog.",
        "evaluator": evaluator,
    }

    assert (
        dataset._exclude_reason(
            tell_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        == "instruction_eval_mismatch"
    )
    assert (
        dataset._exclude_reason(
            show_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        == "instruction_eval_mismatch"
    )
    assert (
        dataset._exclude_reason(
            guide_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        == "instruction_eval_mismatch"
    )
    assert (
        dataset._exclude_reason(
            instruct_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        == "instruction_eval_mismatch"
    )
    assert (
        dataset._exclude_reason(
            action_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        is None
    )


def test_scalecua_excludes_exact_instruction_eval_mismatch_rows():
    exact_ids = {
        "scalecua_osworld_train_chrome_386dbd0e_0241_4a0a_b6a2_6704fba26b1c_task_verify_17",
        "scalecua_osworld_train_chrome_f0b971a1_6831_4b9b_a50e_22a6e47f45ba_task_verify_17",
        "scalecua_osworld_train_chrome_f0b971a1_6831_4b9b_a50e_22a6e47f45ba_task_verify_26",
        "scalecua_osworld_train_chrome_f3977615_2b45_4ac5_8bba_80c17dbe2a37_task_verify_2",
        "scalecua_osworld_train_chrome_f3977615_2b45_4ac5_8bba_80c17dbe2a37_task_verify_10",
        "scalecua_osworld_train_chrome_f3977615_2b45_4ac5_8bba_80c17dbe2a37_task_verify_47",
        "scalecua_osworld_train_chrome_f3977615_2b45_4ac5_8bba_80c17dbe2a37_task_verify_66",
        "scalecua_osworld_train_chrome_f3977615_2b45_4ac5_8bba_80c17dbe2a37_task_verify_91",
        "scalecua_osworld_train_chrome_f5d96daf_83a8_4c86_9686_bada31fc66ab_task_verify_8",
        "scalecua_osworld_train_chrome_f5d96daf_83a8_4c86_9686_bada31fc66ab_task_verify_14",
        "scalecua_osworld_train_chrome_f5d96daf_83a8_4c86_9686_bada31fc66ab_task_verify_17",
        "scalecua_osworld_train_chrome_f5d96daf_83a8_4c86_9686_bada31fc66ab_task_verify_22",
        "scalecua_osworld_train_chrome_f5d96daf_83a8_4c86_9686_bada31fc66ab_task_verify_51",
        "scalecua_osworld_train_chrome_f8369178_fafe_40c2_adc4_b9b08a125456_task_verify_37",
        "scalecua_osworld_train_chrome_f8369178_fafe_40c2_adc4_b9b08a125456_task_verify_38",
        "scalecua_osworld_train_chrome_f8369178_fafe_40c2_adc4_b9b08a125456_task_verify_41",
        "scalecua_osworld_train_chrome_f8369178_fafe_40c2_adc4_b9b08a125456_task_verify_42",
        "scalecua_osworld_train_chrome_f8369178_fafe_40c2_adc4_b9b08a125456_task_verify_46",
        "scalecua_osworld_train_chrome_f8369178_fafe_40c2_adc4_b9b08a125456_task_verify_47",
        "scalecua_osworld_train_chrome_f8369178_fafe_40c2_adc4_b9b08a125456_task_verify_50",
        "scalecua_osworld_train_chrome_f8369178_fafe_40c2_adc4_b9b08a125456_task_verify_51",
        "scalecua_osworld_train_chrome_f8369178_fafe_40c2_adc4_b9b08a125456_task_verify_54",
        "scalecua_osworld_train_chrome_f8369178_fafe_40c2_adc4_b9b08a125456_task_verify_55",
        "scalecua_osworld_train_chrome_35253b65_1c19_4304_8aa4_6884b8218fc0_task_verify_0",
        "scalecua_osworld_train_chrome_35253b65_1c19_4304_8aa4_6884b8218fc0_task_verify_1",
        "scalecua_osworld_train_chrome_35253b65_1c19_4304_8aa4_6884b8218fc0_task_verify_12",
        "scalecua_osworld_train_chrome_35253b65_1c19_4304_8aa4_6884b8218fc0_task_verify_23",
        "scalecua_osworld_train_chrome_58565672_7bfe_48ab_b828_db349231de6b_task_verify_22",
        "scalecua_osworld_train_chrome_58565672_7bfe_48ab_b828_db349231de6b_task_verify_23",
        "scalecua_osworld_train_chrome_58565672_7bfe_48ab_b828_db349231de6b_task_verify_24",
        "scalecua_osworld_train_chrome_58565672_7bfe_48ab_b828_db349231de6b_task_verify_25",
        "scalecua_osworld_train_gimp_d52d6308_ec58_42b7_a2c9_de80e4837b2b_task_verify_18",
        "scalecua_osworld_train_gimp_d52d6308_ec58_42b7_a2c9_de80e4837b2b_task_verify_19",
        "scalecua_osworld_train_gimp_d52d6308_ec58_42b7_a2c9_de80e4837b2b_task_verify_20",
        "scalecua_osworld_train_libreoffice_impress_358aa0a7_6677_453f_ae35_e440f004c31e_task_verify_26",
        "scalecua_osworld_train_libreoffice_impress_ef9d12bd_bcee_4ba0_a40e_918400f43ddf_task_verify_26",
        "scalecua_osworld_train_libreoffice_impress_ef9d12bd_bcee_4ba0_a40e_918400f43ddf_task_verify_28",
        "scalecua_osworld_train_libreoffice_writer_0e47de2a_32e0_456c_a366_8c607ef7a9d2_task_verify_53",
        "scalecua_osworld_train_libreoffice_writer_2b9493d7_49b8_493a_a71b_56cd1f4d6908_task_verify_3",
        "scalecua_osworld_train_multi_apps_82e3c869_49f6_4305_a7ce_f3e64a0618e7_task_verify_53",
        "scalecua_osworld_train_vlc_8ba5ae7a_5ae5_4eab_9fcc_5dd4fe3abf89_task_verify_32",
        "scalecua_osworld_train_os_13584542_872b_42d8_b299_866967b5c3ef_task_verify_14",
        "scalecua_osworld_train_os_13584542_872b_42d8_b299_866967b5c3ef_task_verify_15",
        "scalecua_osworld_train_os_13584542_872b_42d8_b299_866967b5c3ef_task_verify_16",
        "scalecua_osworld_train_os_13584542_872b_42d8_b299_866967b5c3ef_task_verify_22",
        "scalecua_osworld_train_os_13584542_872b_42d8_b299_866967b5c3ef_task_verify_24",
        "scalecua_osworld_train_os_13584542_872b_42d8_b299_866967b5c3ef_task_verify_25",
        "scalecua_osworld_train_os_13584542_872b_42d8_b299_866967b5c3ef_task_verify_27",
        "scalecua_osworld_train_os_13584542_872b_42d8_b299_866967b5c3ef_task_verify_28",
        "scalecua_osworld_train_vs_code_0ed39f63_6049_43d4_ba4d_5fa2fe04a951_task_verify_15",
        "scalecua_osworld_train_vs_code_982d12a5_beab_424f_8d38_d2a48429e511_task_verify_5",
    }
    assert dataset.INSTRUCTION_EVAL_MISMATCH_TASK_IDS == exact_ids

    payload = {
        "instruction": "Configure VLC so the hidden setting matches my preference.",
        "evaluator": {
            "func": "check_vlc_config",
            "result": {"type": "vlc_config"},
            "expected": {"type": "rule", "rules": {"expected": 1}},
        },
    }

    for task_id in exact_ids:
        assert (
            dataset._exclude_reason(
                payload,
                task_id=task_id,
                inherited_exclusion=None,
                unsupported=[],
                runtime_split="train",
                broken_metrics=frozenset(),
            )
            == "instruction_eval_mismatch"
        )
    assert (
        dataset._exclude_reason(
            payload,
            task_id=(
                "scalecua_osworld_train_chrome_386dbd0e_0241_4a0a_b6a2_6704fba26b1c_task_verify_18"
            ),
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        is None
    )


def test_scalecua_excludes_exact_thunderbird_gmail_auth_gap():
    assert len(dataset.THUNDERBIRD_GMAIL_AUTH_TASK_IDS) == 18
    payload = {
        "instruction": "Enter Gmail login details into Thunderbird.",
        "config": [],
        "evaluator": {"func": "check_csv"},
    }

    for task_id in dataset.THUNDERBIRD_GMAIL_AUTH_TASK_IDS:
        assert (
            dataset._exclude_reason(
                payload,
                task_id=task_id,
                inherited_exclusion=None,
                unsupported=[],
                runtime_split="train",
                broken_metrics=frozenset(),
            )
            == "google_auth"
        )
    assert (
        dataset._exclude_reason(
            payload,
            task_id=(
                "scalecua_osworld_train_thunderbird_15c3b339_88f7_4a86_ab16_"
                "e71c58dcb01e_task_verify_17"
            ),
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        is None
    )


def test_scalecua_excludes_exact_chrome_webstore_live_site_rows():
    assert len(dataset.CHROME_WEBSTORE_LIVE_SITE_TASK_IDS) == 42
    payload = {
        "id": "873cafdd-a581-47f6-8b33-b9696ddb7b05",
        "instruction": "Open the Chrome Web Store detail page for an extension.",
        "evaluator": {
            "func": "is_expected_url_pattern_match",
            "result": {"type": "active_url_from_accessTree"},
            "expected": {
                "type": "rule",
                "rules": {"expected": ["chromewebstore\\.google\\.com/detail"]},
            },
        },
    }

    for task_id in dataset.CHROME_WEBSTORE_LIVE_SITE_TASK_IDS:
        split = "rl" if "_rl_" in task_id else "train"
        assert (
            dataset._exclude_reason(
                payload,
                task_id=task_id,
                inherited_exclusion=None,
                unsupported=[],
                runtime_split=split,
                broken_metrics=frozenset(),
            )
            == "upstream_live_site_drift"
        )


def test_scalecua_excludes_exact_visual_audit_live_site_drift_rows():
    assert dataset.UPSTREAM_LIVE_SITE_DRIFT_TASK_IDS == {
        "scalecua_osworld_rl_chrome_7b6c7e24_c58a_49fc_a5bb_d57b80e5b4c3_traj_verify_2",
        "scalecua_osworld_rl_chrome_9f3f70fc_5afc_4958_a7b7_3bb4fcb01805_traj_verify_1",
        "scalecua_osworld_rl_chrome_47543840_672a_467d_80df_8f7c3b9788c9_traj_verify_0",
        "scalecua_osworld_rl_chrome_47543840_672a_467d_80df_8f7c3b9788c9_traj_verify_1",
        "scalecua_osworld_rl_chrome_47543840_672a_467d_80df_8f7c3b9788c9_traj_verify_2",
        "scalecua_osworld_rl_chrome_b4f95342_463e_4179_8c3f_193cd7241fb2_traj_verify_3",
        "scalecua_osworld_rl_chrome_b4f95342_463e_4179_8c3f_193cd7241fb2_traj_verify_4",
        "scalecua_osworld_rl_chrome_b7895e80_f4d1_4648_bee0_4eb45a6f1fa8_traj_verify_0",
        "scalecua_osworld_rl_chrome_b7895e80_f4d1_4648_bee0_4eb45a6f1fa8_traj_verify_2",
        "scalecua_osworld_rl_chrome_da46d875_6b82_4681_9284_653b0c7ae241_traj_verify_2",
        "scalecua_osworld_rl_chrome_f0b971a1_6831_4b9b_a50e_22a6e47f45ba_traj_verify_1",
        "scalecua_osworld_rl_chrome_f5d96daf_83a8_4c86_9686_bada31fc66ab_traj_verify_0",
        "scalecua_osworld_rl_chrome_f5d96daf_83a8_4c86_9686_bada31fc66ab_traj_verify_1",
        "scalecua_osworld_rl_chrome_f5d96daf_83a8_4c86_9686_bada31fc66ab_traj_verify_2",
        "scalecua_osworld_train_libreoffice_calc_36037439_2044_4b50_b9d1_875b5a332143_task_verify_20",
        "scalecua_osworld_train_libreoffice_calc_36037439_2044_4b50_b9d1_875b5a332143_task_verify_24",
        "scalecua_osworld_train_libreoffice_calc_36037439_2044_4b50_b9d1_875b5a332143_task_verify_35",
        "scalecua_osworld_train_chrome_06fe7178_4491_4589_810f_2e2bc9502122_task_verify_2",
        "scalecua_osworld_train_chrome_06fe7178_4491_4589_810f_2e2bc9502122_task_verify_34",
        "scalecua_osworld_train_chrome_121ba48f_9e17_48ce_9bc6_a4fb17a7ebba_task_verify_2",
        "scalecua_osworld_train_chrome_121ba48f_9e17_48ce_9bc6_a4fb17a7ebba_task_verify_3",
        "scalecua_osworld_train_chrome_121ba48f_9e17_48ce_9bc6_a4fb17a7ebba_task_verify_4",
        "scalecua_osworld_train_chrome_121ba48f_9e17_48ce_9bc6_a4fb17a7ebba_task_verify_5",
        "scalecua_osworld_train_chrome_121ba48f_9e17_48ce_9bc6_a4fb17a7ebba_task_verify_6",
        "scalecua_osworld_train_chrome_121ba48f_9e17_48ce_9bc6_a4fb17a7ebba_task_verify_7",
        "scalecua_osworld_train_chrome_121ba48f_9e17_48ce_9bc6_a4fb17a7ebba_task_verify_8",
        "scalecua_osworld_train_chrome_121ba48f_9e17_48ce_9bc6_a4fb17a7ebba_task_verify_9",
        "scalecua_osworld_train_chrome_121ba48f_9e17_48ce_9bc6_a4fb17a7ebba_task_verify_10",
        "scalecua_osworld_train_chrome_121ba48f_9e17_48ce_9bc6_a4fb17a7ebba_task_verify_11",
        "scalecua_osworld_train_chrome_121ba48f_9e17_48ce_9bc6_a4fb17a7ebba_task_verify_13",
        "scalecua_osworld_train_chrome_121ba48f_9e17_48ce_9bc6_a4fb17a7ebba_task_verify_14",
        "scalecua_osworld_train_chrome_121ba48f_9e17_48ce_9bc6_a4fb17a7ebba_task_verify_15",
        "scalecua_osworld_train_chrome_121ba48f_9e17_48ce_9bc6_a4fb17a7ebba_task_verify_16",
        "scalecua_osworld_train_chrome_121ba48f_9e17_48ce_9bc6_a4fb17a7ebba_task_verify_26",
        "scalecua_osworld_train_chrome_121ba48f_9e17_48ce_9bc6_a4fb17a7ebba_task_verify_27",
        "scalecua_osworld_train_chrome_121ba48f_9e17_48ce_9bc6_a4fb17a7ebba_task_verify_28",
        "scalecua_osworld_train_chrome_121ba48f_9e17_48ce_9bc6_a4fb17a7ebba_task_verify_29",
        "scalecua_osworld_train_chrome_121ba48f_9e17_48ce_9bc6_a4fb17a7ebba_task_verify_30",
        "scalecua_osworld_train_chrome_121ba48f_9e17_48ce_9bc6_a4fb17a7ebba_task_verify_31",
        "scalecua_osworld_train_chrome_121ba48f_9e17_48ce_9bc6_a4fb17a7ebba_task_verify_32",
        "scalecua_osworld_train_chrome_121ba48f_9e17_48ce_9bc6_a4fb17a7ebba_task_verify_33",
        "scalecua_osworld_train_chrome_121ba48f_9e17_48ce_9bc6_a4fb17a7ebba_task_verify_37",
        "scalecua_osworld_train_chrome_121ba48f_9e17_48ce_9bc6_a4fb17a7ebba_task_verify_38",
        "scalecua_osworld_train_chrome_121ba48f_9e17_48ce_9bc6_a4fb17a7ebba_task_verify_39",
        "scalecua_osworld_train_chrome_121ba48f_9e17_48ce_9bc6_a4fb17a7ebba_task_verify_40",
        "scalecua_osworld_train_chrome_a728a36e_8bf1_4bb6_9a03_ef039a5233f0_task_verify_57",
        "scalecua_osworld_train_chrome_b4f95342_463e_4179_8c3f_193cd7241fb2_task_verify_42",
        "scalecua_osworld_train_chrome_b4f95342_463e_4179_8c3f_193cd7241fb2_task_verify_45",
        "scalecua_osworld_train_chrome_f3b19d1e_2d48_44e9_b4e1_defcae1a0197_task_verify_39",
        "scalecua_osworld_train_chrome_f3b19d1e_2d48_44e9_b4e1_defcae1a0197_task_verify_52",
        "scalecua_osworld_train_chrome_f3b19d1e_2d48_44e9_b4e1_defcae1a0197_task_verify_64",
        "scalecua_osworld_train_chrome_f3b19d1e_2d48_44e9_b4e1_defcae1a0197_task_verify_71",
        "scalecua_osworld_train_chrome_9f3f70fc_5afc_4958_a7b7_3bb4fcb01805_task_verify_37",
        "scalecua_osworld_train_chrome_9f3f70fc_5afc_4958_a7b7_3bb4fcb01805_task_verify_16",
        "scalecua_osworld_train_chrome_82279c77_8fc6_46f6_9622_3ba96f61b477_task_verify_6",
        "scalecua_osworld_train_chrome_82279c77_8fc6_46f6_9622_3ba96f61b477_task_verify_7",
        "scalecua_osworld_train_chrome_82279c77_8fc6_46f6_9622_3ba96f61b477_task_verify_8",
        "scalecua_osworld_train_chrome_82279c77_8fc6_46f6_9622_3ba96f61b477_task_verify_9",
        "scalecua_osworld_train_chrome_82279c77_8fc6_46f6_9622_3ba96f61b477_task_verify_10",
        "scalecua_osworld_train_chrome_82279c77_8fc6_46f6_9622_3ba96f61b477_task_verify_11",
        "scalecua_osworld_train_chrome_a728a36e_8bf1_4bb6_9a03_ef039a5233f0_task_verify_39",
        "scalecua_osworld_train_chrome_a728a36e_8bf1_4bb6_9a03_ef039a5233f0_task_verify_21",
        "scalecua_osworld_train_chrome_a728a36e_8bf1_4bb6_9a03_ef039a5233f0_task_verify_22",
        "scalecua_osworld_train_chrome_a728a36e_8bf1_4bb6_9a03_ef039a5233f0_task_verify_24",
        "scalecua_osworld_train_chrome_a728a36e_8bf1_4bb6_9a03_ef039a5233f0_task_verify_25",
        "scalecua_osworld_train_chrome_a96b564e_dbe9_42c3_9ccf_b4498073438a_task_verify_39",
        "scalecua_osworld_train_chrome_c1fa57f3_c3db_4596_8f09_020701085416_task_verify_23",
        "scalecua_osworld_train_chrome_c1fa57f3_c3db_4596_8f09_020701085416_task_verify_26",
        "scalecua_osworld_train_chrome_c1fa57f3_c3db_4596_8f09_020701085416_task_verify_56",
        "scalecua_osworld_train_chrome_f3b19d1e_2d48_44e9_b4e1_defcae1a0197_task_verify_19",
        "scalecua_osworld_train_chrome_f3b19d1e_2d48_44e9_b4e1_defcae1a0197_task_verify_23",
        "scalecua_osworld_train_chrome_f3b19d1e_2d48_44e9_b4e1_defcae1a0197_task_verify_35",
        "scalecua_osworld_train_chrome_f3b19d1e_2d48_44e9_b4e1_defcae1a0197_task_verify_38",
        "scalecua_osworld_train_chrome_f3b19d1e_2d48_44e9_b4e1_defcae1a0197_task_verify_40",
        "scalecua_osworld_train_chrome_f3b19d1e_2d48_44e9_b4e1_defcae1a0197_task_verify_66",
        "scalecua_osworld_train_chrome_b4f95342_463e_4179_8c3f_193cd7241fb2_task_verify_46",
        "scalecua_osworld_train_chrome_f0b971a1_6831_4b9b_a50e_22a6e47f45ba_task_verify_53",
    }
    payload = {"instruction": "Open the live web page.", "evaluator": {}}

    for task_id in dataset.UPSTREAM_LIVE_SITE_DRIFT_TASK_IDS:
        runtime_split = "rl" if "_rl_" in task_id else "train"
        assert (
            dataset._exclude_reason(
                payload,
                task_id=task_id,
                inherited_exclusion=None,
                unsupported=[],
                runtime_split=runtime_split,
                broken_metrics=frozenset(),
            )
            == "upstream_live_site_drift"
        )
    for task_id in (
        "scalecua_osworld_train_chrome_121ba48f_9e17_48ce_9bc6_a4fb17a7ebba_task_verify_0",
        "scalecua_osworld_train_chrome_121ba48f_9e17_48ce_9bc6_a4fb17a7ebba_task_verify_17",
        "scalecua_osworld_train_libreoffice_calc_36037439_2044_4b50_b9d1_875b5a332143_task_verify_21",
    ):
        assert (
            dataset._exclude_reason(
                payload,
                task_id=task_id,
                inherited_exclusion=None,
                unsupported=[],
                runtime_split="train",
                broken_metrics=frozenset(),
            )
            is None
        )


def test_scalecua_excludes_missing_instruction_asset_url_without_overfiltering():
    missing_url_payload = {
        "instruction": (
            "Fetch the image from the provided URL, rotate it 90 degrees clockwise "
            "using GIMP, and save it as 'rotated.jpeg' on the Desktop."
        ),
        "config": [
            {
                "type": "execute",
                "parameters": {
                    "command": [
                        "python",
                        "-c",
                        'import pyautogui; pyautogui.hotkey("ctrl", "alt", "t")',
                    ]
                },
            }
        ],
        "evaluator": {
            "func": "check_image_orientation__hash",
            "result": {"type": "image_dimensions__hash"},
            "expected": {"type": "rule", "rules": {"is_portrait": True}},
        },
    }
    explicit_instruction_url_payload = dict(
        missing_url_payload,
        instruction=(
            "Fetch the image from https://example.com/bird.jpeg, rotate it 90 "
            "degrees clockwise using GIMP, and save it as 'rotated.jpeg'."
        ),
    )
    supplied_link_payload = dict(
        missing_url_payload,
        instruction=(
            "Get the image via the supplied link, reduce its file size below "
            "500KB in GIMP, and export it to the Desktop as 'downsized.jpeg'."
        ),
    )
    setup_url_payload = dict(
        missing_url_payload,
        config=[
            {
                "type": "chrome_open_tabs",
                "parameters": {"urls_to_open": ["https://example.com/buttons"]},
            }
        ],
    )
    email_link_payload = {
        "instruction": (
            "In Thunderbird, go to the Bills folder, choose the latest email, "
            "and open its primary link in Chrome as the focused tab."
        ),
        "config": [],
        "evaluator": {
            "func": "is_expected_active_tab",
            "result": {"type": "active_url_from_accessTree"},
            "expected": {"type": "rule", "rules": {"url": "https://example.com"}},
        },
    }

    assert (
        dataset._exclude_reason(
            missing_url_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        == "instruction_setup_mismatch"
    )
    assert (
        dataset._exclude_reason(
            supplied_link_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        == "instruction_setup_mismatch"
    )
    assert (
        dataset._exclude_reason(
            explicit_instruction_url_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        is None
    )
    assert (
        dataset._exclude_reason(
            setup_url_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        is None
    )
    assert (
        dataset._exclude_reason(
            email_link_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        is None
    )


def test_scalecua_excludes_missing_author_results_reference_asset():
    missing_reference_payload = {
        "instruction": "Resize the image from slide two to 800x600 pixels.",
        "evaluator": {
            "func": "check_image_resized__8000c7e00c88e8061974ceb3ccc555df",
            "result": {
                "type": "ppt_slide_image__8000c7e00c88e8061974ceb3ccc555df",
                "path": "/home/user/Desktop/background_resized.png",
            },
            "expected": {
                "type": "rule",
                "rules": {
                    "target_width": 800,
                    "target_height": 600,
                    "reference_path": (
                        "/home/lvbowen/project/AutoGen/results/task_verify_per5_1221/"
                        "run_20251221_151547/reference_slide2_image.png"
                    ),
                },
            },
        },
    }
    vm_reference_payload = copy.deepcopy(missing_reference_payload)
    vm_reference_payload["evaluator"]["expected"]["rules"]["reference_path"] = (
        "/tmp/reference_slide2.png"
    )
    author_cache_payload = copy.deepcopy(missing_reference_payload)
    rules = author_cache_payload["evaluator"]["expected"]["rules"]
    rules.pop("reference_path")
    rules["original_path"] = (
        "/home/lvbowen/project/AutoGen/src/envs/osworld_env/cache/"
        "e8172110-ec08-421b-a6f5-842e6451911f/character.png"
    )

    assert (
        dataset._exclude_reason(
            missing_reference_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        == "missing_reference_asset"
    )
    assert (
        dataset._exclude_reason(
            vm_reference_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        is None
    )
    assert (
        dataset._exclude_reason(
            author_cache_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="rl",
            broken_metrics=frozenset(),
        )
        is None
    )


def test_scalecua_excludes_known_upstream_generated_eval_bug_exactly():
    buggy_task_ids = {
        "scalecua_osworld_rl_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_traj_verify_2",
        "scalecua_osworld_rl_vs_code_53ad5833_3455_407b_bbc6_45b4c79ab8fb_traj_verify_4",
        "scalecua_osworld_rl_multi_apps_2b9493d7_49b8_493a_a71b_56cd1f4d6908_traj_verify_3",
        "scalecua_osworld_rl_os_a462a795_fdc7_4b23_b689_e8b6df786b78_traj_verify_0",
        "scalecua_osworld_rl_os_a462a795_fdc7_4b23_b689_e8b6df786b78_traj_verify_1",
        "scalecua_osworld_rl_os_b3d4a89c_53f2_4d6b_8b6a_541fb5d205fa_traj_verify_0",
        "scalecua_osworld_rl_os_b3d4a89c_53f2_4d6b_8b6a_541fb5d205fa_traj_verify_1",
        "scalecua_osworld_rl_os_b3d4a89c_53f2_4d6b_8b6a_541fb5d205fa_traj_verify_2",
        "scalecua_osworld_rl_os_94d95f96_9699_4208_98ba_3c3119edf9c2_traj_verify_3",
        "scalecua_osworld_rl_libreoffice_writer_0a0faba3_5580_44df_965d_f562a99b291c_traj_verify_6",
        "scalecua_osworld_rl_libreoffice_writer_e1fc0df3_c8b9_4ee7_864c_d0b590d3aa56_traj_verify_2",
        "scalecua_osworld_rl_vs_code_971cbb5b_3cbf_4ff7_9e24_b5c84fcebfa6_traj_verify_5",
        "scalecua_osworld_rl_chrome_788b3701_3ec9_4b67_b679_418bfa726c22_traj_verify_0",
        "scalecua_osworld_rl_libreoffice_impress_c59742c0_4323_4b9d_8a02_723c251deaa0_traj_verify_3",
        "scalecua_osworld_rl_libreoffice_impress_5d901039_a89c_4bfb_967b_bf66f4df075e_traj_verify_0",
        "scalecua_osworld_rl_libreoffice_impress_9ec204e4_f0a3_42f8_8458_b772a6797cab_traj_verify_1",
        "scalecua_osworld_rl_libreoffice_impress_af23762e_2bfd_4a1d_aada_20fa8de9ce07_traj_verify_2",
        "scalecua_osworld_rl_libreoffice_impress_ed43c15f_00cb_4054_9c95_62c880865d68_traj_verify_1",
        "scalecua_osworld_rl_libreoffice_impress_ef9d12bd_bcee_4ba0_a40e_918400f43ddf_traj_verify_1",
        "scalecua_osworld_rl_libreoffice_impress_ef9d12bd_bcee_4ba0_a40e_918400f43ddf_traj_verify_2",
        "scalecua_osworld_rl_libreoffice_impress_ef9d12bd_bcee_4ba0_a40e_918400f43ddf_traj_verify_4",
        "scalecua_osworld_rl_libreoffice_impress_ef9d12bd_bcee_4ba0_a40e_918400f43ddf_traj_verify_5",
        "scalecua_osworld_rl_libreoffice_impress_550ce7e7_747b_495f_b122_acdc4d0b8e54_traj_verify_4",
        "scalecua_osworld_rl_libreoffice_impress_550ce7e7_747b_495f_b122_acdc4d0b8e54_traj_verify_5",
        "scalecua_osworld_rl_libreoffice_impress_550ce7e7_747b_495f_b122_acdc4d0b8e54_traj_verify_6",
        "scalecua_osworld_rl_libreoffice_impress_af2d657a_e6b3_4c6a_9f67_9e3ed015974c_traj_verify_3",
        "scalecua_osworld_rl_libreoffice_calc_37608790_6147_45d0_9f20_1137bb35703d_traj_verify_3",
        "scalecua_osworld_rl_libreoffice_calc_ecb0df7a_4e8d_4a03_b162_053391d3afaf_traj_verify_0",
        "scalecua_osworld_rl_libreoffice_calc_1954cced_e748_45c4_9c26_9855b97fbc5e_traj_verify_5",
        "scalecua_osworld_rl_libreoffice_calc_1de60575_bb6e_4c3d_9e6a_2fa699f9f197_traj_verify_4",
        "scalecua_osworld_rl_libreoffice_calc_3a7c8185_25c1_4941_bd7b_96e823c9f21f_traj_verify_0",
        "scalecua_osworld_rl_libreoffice_calc_3a7c8185_25c1_4941_bd7b_96e823c9f21f_traj_verify_1",
        "scalecua_osworld_rl_libreoffice_calc_51b11269_2ca8_4b2a_9163_f21758420e78_traj_verify_2",
        "scalecua_osworld_rl_libreoffice_calc_7a4e4bc8_922c_4c84_865c_25ba34136be1_traj_verify_2",
        "scalecua_osworld_train_libreoffice_impress_eb303e01_261e_4972_8c07_c9b4e7a4922a_task_verify_14",
        "scalecua_osworld_train_libreoffice_impress_eb303e01_261e_4972_8c07_c9b4e7a4922a_task_verify_16",
        "scalecua_osworld_train_libreoffice_calc_1de60575_bb6e_4c3d_9e6a_2fa699f9f197_task_verify_43",
        "scalecua_osworld_train_libreoffice_impress_bf4e9888_f10f_47af_8dba_76413038b73c_task_verify_43",
        "scalecua_osworld_rl_libreoffice_writer_4bcb1253_a636_4df4_8cb0_a35c04dfef31_traj_verify_6",
        "scalecua_osworld_rl_libreoffice_writer_6f81754e_285d_4ce0_b59e_af7edb02d108_traj_verify_5",
        "scalecua_osworld_rl_libreoffice_writer_936321ce_5236_426a_9a20_e0e3c5dc536f_traj_verify_5",
        "scalecua_osworld_rl_libreoffice_writer_adf5e2c3_64c7_4644_b7b6_d2f0167927e7_traj_verify_3",
        "scalecua_osworld_rl_libreoffice_writer_b21acd93_60fd_4127_8a43_2f5178f4a830_traj_verify_3",
        "scalecua_osworld_rl_libreoffice_writer_b21acd93_60fd_4127_8a43_2f5178f4a830_traj_verify_4",
        "scalecua_osworld_rl_libreoffice_writer_b21acd93_60fd_4127_8a43_2f5178f4a830_traj_verify_5",
        "scalecua_osworld_rl_libreoffice_writer_b21acd93_60fd_4127_8a43_2f5178f4a830_traj_verify_6",
        "scalecua_osworld_rl_libreoffice_writer_b21acd93_60fd_4127_8a43_2f5178f4a830_traj_verify_7",
        "scalecua_osworld_rl_libreoffice_writer_e246f6d8_78d7_44ac_b668_fcf47946cb50_traj_verify_0",
        "scalecua_osworld_rl_libreoffice_writer_8df7e444_8e06_4f93_8a1a_c5c974269d82_traj_verify_1",
        "scalecua_osworld_rl_gimp_734d6579_c07d_47a8_9ae2_13339795476b_traj_verify_0",
        "scalecua_osworld_rl_multi_apps_415ef462_bed3_493a_ac36_ca8c6d23bf1b_traj_verify_2",
        "scalecua_osworld_rl_multi_apps_47f7c0ce_a5fb_4100_a5e6_65cd0e7429e5_traj_verify_4",
        "scalecua_osworld_rl_multi_apps_716a6079_22da_47f1_ba73_c9d58f986a38_traj_verify_2",
        "scalecua_osworld_rl_os_3ce045a0_877b_42aa_8d2c_b4a863336ab8_traj_verify_6",
        "scalecua_osworld_rl_os_3f05f3b9_29ba_4b6b_95aa_2204697ffc06_traj_verify_1",
        "scalecua_osworld_rl_thunderbird_9bc3cc16_074a_45ac_9bdc_b2a362e1daf3_traj_verify_2",
        "scalecua_osworld_rl_thunderbird_15c3b339_88f7_4a86_ab16_e71c58dcb01e_traj_verify_2",
        "scalecua_osworld_rl_thunderbird_15c3b339_88f7_4a86_ab16_e71c58dcb01e_traj_verify_4",
        "scalecua_osworld_rl_thunderbird_15c3b339_88f7_4a86_ab16_e71c58dcb01e_traj_verify_6",
        "scalecua_osworld_rl_thunderbird_7b1e1ff9_bb85_49be_b01d_d6424be18cd0_traj_verify_0",
        "scalecua_osworld_rl_thunderbird_7b1e1ff9_bb85_49be_b01d_d6424be18cd0_traj_verify_1",
        "scalecua_osworld_rl_thunderbird_7b1e1ff9_bb85_49be_b01d_d6424be18cd0_traj_verify_5",
        "scalecua_osworld_train_thunderbird_dd84e895_72fd_4023_a336_97689ded257c_task_verify_2",
        "scalecua_osworld_train_thunderbird_dd84e895_72fd_4023_a336_97689ded257c_task_verify_3",
        "scalecua_osworld_train_thunderbird_dd84e895_72fd_4023_a336_97689ded257c_task_verify_34",
        "scalecua_osworld_train_thunderbird_dd84e895_72fd_4023_a336_97689ded257c_task_verify_35",
        "scalecua_osworld_rl_vs_code_53ad5833_3455_407b_bbc6_45b4c79ab8fb_traj_verify_6",
        "scalecua_osworld_train_vs_code_53ad5833_3455_407b_bbc6_45b4c79ab8fb_task_verify_0",
        "scalecua_osworld_train_vs_code_53ad5833_3455_407b_bbc6_45b4c79ab8fb_task_verify_1",
        "scalecua_osworld_train_vs_code_53ad5833_3455_407b_bbc6_45b4c79ab8fb_task_verify_12",
        "scalecua_osworld_train_vs_code_53ad5833_3455_407b_bbc6_45b4c79ab8fb_task_verify_23",
        "scalecua_osworld_train_vs_code_53ad5833_3455_407b_bbc6_45b4c79ab8fb_task_verify_30",
        "scalecua_osworld_train_vs_code_53ad5833_3455_407b_bbc6_45b4c79ab8fb_task_verify_31",
        "scalecua_osworld_train_vs_code_53ad5833_3455_407b_bbc6_45b4c79ab8fb_task_verify_32",
        "scalecua_osworld_train_vs_code_53ad5833_3455_407b_bbc6_45b4c79ab8fb_task_verify_33",
        "scalecua_osworld_train_vs_code_53ad5833_3455_407b_bbc6_45b4c79ab8fb_task_verify_36",
        "scalecua_osworld_train_vs_code_53ad5833_3455_407b_bbc6_45b4c79ab8fb_task_verify_37",
        "scalecua_osworld_train_vs_code_53ad5833_3455_407b_bbc6_45b4c79ab8fb_task_verify_48",
        "scalecua_osworld_rl_os_23393935_50c7_4a86_aeea_2b78fd089c5c_traj_verify_1",
        "scalecua_osworld_train_libreoffice_writer_0a0faba3_5580_44df_965d_f562a99b291c_task_verify_39",
        "scalecua_osworld_train_libreoffice_writer_0a0faba3_5580_44df_965d_f562a99b291c_task_verify_40",
        "scalecua_osworld_train_libreoffice_writer_0a0faba3_5580_44df_965d_f562a99b291c_task_verify_41",
        "scalecua_osworld_train_libreoffice_writer_0a0faba3_5580_44df_965d_f562a99b291c_task_verify_42",
        "scalecua_osworld_train_libreoffice_writer_e528b65e_1107_4b8c_8988_490e4fece599_task_verify_57",
        "scalecua_osworld_train_os_6f56bf42_85b8_4fbb_8e06_6c44960184ba_task_verify_43",
        "scalecua_osworld_train_os_6f56bf42_85b8_4fbb_8e06_6c44960184ba_task_verify_44",
        "scalecua_osworld_train_os_6f56bf42_85b8_4fbb_8e06_6c44960184ba_task_verify_46",
        "scalecua_osworld_train_os_6f56bf42_85b8_4fbb_8e06_6c44960184ba_task_verify_47",
        "scalecua_osworld_rl_os_5c1075ca_bb34_46a3_a7a0_029bd7463e79_traj_verify_1",
        "scalecua_osworld_train_libreoffice_calc_eb03d19a_b88d_4de4_8a64_ca0ac66f426b_task_verify_41",
        "scalecua_osworld_train_libreoffice_calc_04d9aeaf_7bed_4024_bedb_e10e6f00eb7f_task_verify_33",
        "scalecua_osworld_train_libreoffice_calc_12382c62_0cd1_4bf2_bdc8_1d20bf9b2371_task_verify_45",
        "scalecua_osworld_train_libreoffice_calc_1f18aa87_af6f_41ef_9853_cdb8f32ebdea_task_verify_36",
        "scalecua_osworld_train_libreoffice_calc_2c1ebcd7_9c6d_4c9a_afad_900e381ecd5e_task_verify_10",
        "scalecua_osworld_train_libreoffice_calc_2c1ebcd7_9c6d_4c9a_afad_900e381ecd5e_task_verify_20",
        "scalecua_osworld_train_libreoffice_calc_4188d3a4_077d_46b7_9c86_23e1a036f6c1_task_verify_31",
        "scalecua_osworld_train_libreoffice_calc_42e0a640_4f19_4b28_973d_729602b5a4a7_task_verify_24",
        "scalecua_osworld_train_libreoffice_calc_42e0a640_4f19_4b28_973d_729602b5a4a7_task_verify_34",
        "scalecua_osworld_train_libreoffice_calc_42e0a640_4f19_4b28_973d_729602b5a4a7_task_verify_36",
        "scalecua_osworld_train_libreoffice_calc_7e429b8d_a3f0_4ed0_9b58_08957d00b127_task_verify_2",
        "scalecua_osworld_train_libreoffice_calc_81c425f5_78f3_4771_afd6_3d2973825947_task_verify_22",
        "scalecua_osworld_train_libreoffice_calc_881deb30_9549_4583_a841_8270c65f2a17_task_verify_71",
        "scalecua_osworld_train_libreoffice_calc_f9584479_3d0d_4c79_affa_9ad7afdd8850_task_verify_10",
        "scalecua_osworld_train_libreoffice_calc_f9584479_3d0d_4c79_affa_9ad7afdd8850_task_verify_12",
        "scalecua_osworld_train_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_task_verify_39",
        "scalecua_osworld_train_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_task_verify_40",
        "scalecua_osworld_train_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_task_verify_41",
        "scalecua_osworld_train_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_task_verify_42",
        "scalecua_osworld_train_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_task_verify_62",
        "scalecua_osworld_train_chrome_788b3701_3ec9_4b67_b679_418bfa726c22_task_verify_6",
        "scalecua_osworld_train_chrome_788b3701_3ec9_4b67_b679_418bfa726c22_task_verify_19",
        "scalecua_osworld_train_chrome_12086550_11c0_466b_b367_1d9e75b3910e_task_verify_6",
        "scalecua_osworld_train_chrome_12086550_11c0_466b_b367_1d9e75b3910e_task_verify_28",
        "scalecua_osworld_train_chrome_386dbd0e_0241_4a0a_b6a2_6704fba26b1c_task_verify_11",
        "scalecua_osworld_train_chrome_386dbd0e_0241_4a0a_b6a2_6704fba26b1c_task_verify_8",
        "scalecua_osworld_train_chrome_386dbd0e_0241_4a0a_b6a2_6704fba26b1c_task_verify_9",
        "scalecua_osworld_train_chrome_48c46dc7_fe04_4505_ade7_723cba1aa6f6_task_verify_1",
        "scalecua_osworld_train_chrome_480bcfea_d68f_4aaa_a0a9_2589ef319381_task_verify_21",
        "scalecua_osworld_train_chrome_480bcfea_d68f_4aaa_a0a9_2589ef319381_task_verify_2",
        "scalecua_osworld_train_chrome_480bcfea_d68f_4aaa_a0a9_2589ef319381_task_verify_15",
        "scalecua_osworld_train_chrome_480bcfea_d68f_4aaa_a0a9_2589ef319381_task_verify_39",
        "scalecua_osworld_train_chrome_fc6d8143_9452_4171_9459_7f515143419a_task_verify_8",
        "scalecua_osworld_train_chrome_fc6d8143_9452_4171_9459_7f515143419a_task_verify_25",
        "scalecua_osworld_train_chrome_2ae9ba84_3a0d_4d4c_8338_3a1478dc5fe3_task_verify_8",
        "scalecua_osworld_train_chrome_2ae9ba84_3a0d_4d4c_8338_3a1478dc5fe3_task_verify_9",
        "scalecua_osworld_train_chrome_2ae9ba84_3a0d_4d4c_8338_3a1478dc5fe3_task_verify_10",
        "scalecua_osworld_train_chrome_2ae9ba84_3a0d_4d4c_8338_3a1478dc5fe3_task_verify_11",
        "scalecua_osworld_train_chrome_99146c54_4f37_4ab8_9327_5f3291665e1e_task_verify_22",
        "scalecua_osworld_train_chrome_bb5e4c0d_f964_439c_97b6_bdb9747de3f4_task_verify_40",
        "scalecua_osworld_rl_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_traj_verify_4",
        "scalecua_osworld_rl_chrome_3720f614_37fd_4d04_8a6b_76f54f8c222d_traj_verify_4",
        "scalecua_osworld_train_libreoffice_calc_01b269ae_2111_4a07_81fd_3fcd711993b0_task_verify_3",
        "scalecua_osworld_train_libreoffice_calc_0a2e43bf_b26c_4631_a966_af9dfa12c9e5_task_verify_28",
        "scalecua_osworld_train_libreoffice_calc_0326d92d_d218_48a8_9ca1_981cd6d064c7_task_verify_37",
        "scalecua_osworld_train_libreoffice_calc_3a7c8185_25c1_4941_bd7b_96e823c9f21f_task_verify_22",
        "scalecua_osworld_train_libreoffice_calc_51719eea_10bc_4246_a428_ac7c433dd4b3_task_verify_57",
        "scalecua_osworld_train_libreoffice_calc_aa3a8974_2e85_438b_b29e_a64df44deb4b_task_verify_64",
        "scalecua_osworld_train_chrome_99146c54_4f37_4ab8_9327_5f3291665e1e_task_verify_40",
        "scalecua_osworld_train_os_5c1075ca_bb34_46a3_a7a0_029bd7463e79_task_verify_33",
        "scalecua_osworld_train_os_5c1075ca_bb34_46a3_a7a0_029bd7463e79_task_verify_51",
        "scalecua_osworld_train_multi_apps_a74b607e_6bb5_4ea8_8a7c_5d97c7bbcd2a_task_verify_54",
        "scalecua_osworld_train_gimp_06ca5602_62ca_47f6_ad4f_da151cde54cc_task_verify_11",
        "scalecua_osworld_train_gimp_06ca5602_62ca_47f6_ad4f_da151cde54cc_task_verify_16",
        "scalecua_osworld_train_gimp_7767eef2_56a3_4cea_8c9f_48c070c7d65b_task_verify_17",
        "scalecua_osworld_train_gimp_7767eef2_56a3_4cea_8c9f_48c070c7d65b_task_verify_19",
        "scalecua_osworld_train_gimp_7767eef2_56a3_4cea_8c9f_48c070c7d65b_task_verify_20",
        "scalecua_osworld_train_gimp_7767eef2_56a3_4cea_8c9f_48c070c7d65b_task_verify_25",
        "scalecua_osworld_train_gimp_7767eef2_56a3_4cea_8c9f_48c070c7d65b_task_verify_35",
        "scalecua_osworld_train_gimp_d52d6308_ec58_42b7_a2c9_de80e4837b2b_task_verify_31",
        "scalecua_osworld_train_gimp_d52d6308_ec58_42b7_a2c9_de80e4837b2b_task_verify_47",
        "scalecua_osworld_train_gimp_d52d6308_ec58_42b7_a2c9_de80e4837b2b_task_verify_54",
        "scalecua_osworld_train_gimp_d52d6308_ec58_42b7_a2c9_de80e4837b2b_task_verify_55",
        "scalecua_osworld_train_gimp_d52d6308_ec58_42b7_a2c9_de80e4837b2b_task_verify_56",
        "scalecua_osworld_train_gimp_d52d6308_ec58_42b7_a2c9_de80e4837b2b_task_verify_69",
        "scalecua_osworld_train_gimp_d52d6308_ec58_42b7_a2c9_de80e4837b2b_task_verify_75",
        "scalecua_osworld_train_gimp_d52d6308_ec58_42b7_a2c9_de80e4837b2b_task_verify_77",
        "scalecua_osworld_train_gimp_d52d6308_ec58_42b7_a2c9_de80e4837b2b_task_verify_78",
        "scalecua_osworld_train_gimp_e8172110_ec08_421b_a6f5_842e6451911f_task_verify_64",
        "scalecua_osworld_train_gimp_227d2f97_562b_4ccb_ae47_a5ec9e142fbb_task_verify_43",
        "scalecua_osworld_train_gimp_227d2f97_562b_4ccb_ae47_a5ec9e142fbb_task_verify_47",
        "scalecua_osworld_train_gimp_7b7617bd_57cc_468e_9c91_40c4ec2bcb3d_task_verify_30",
        "scalecua_osworld_train_gimp_7b7617bd_57cc_468e_9c91_40c4ec2bcb3d_task_verify_38",
        "scalecua_osworld_train_gimp_b148e375_fe0b_4bec_90e7_38632b0d73c2_task_verify_0",
        "scalecua_osworld_train_gimp_b148e375_fe0b_4bec_90e7_38632b0d73c2_task_verify_1",
        "scalecua_osworld_train_gimp_b148e375_fe0b_4bec_90e7_38632b0d73c2_task_verify_8",
        "scalecua_osworld_train_gimp_b148e375_fe0b_4bec_90e7_38632b0d73c2_task_verify_9",
        "scalecua_osworld_train_gimp_b148e375_fe0b_4bec_90e7_38632b0d73c2_task_verify_16",
        "scalecua_osworld_train_gimp_b148e375_fe0b_4bec_90e7_38632b0d73c2_task_verify_17",
        "scalecua_osworld_train_gimp_b148e375_fe0b_4bec_90e7_38632b0d73c2_task_verify_28",
        "scalecua_osworld_train_gimp_b148e375_fe0b_4bec_90e7_38632b0d73c2_task_verify_29",
        "scalecua_osworld_train_gimp_b148e375_fe0b_4bec_90e7_38632b0d73c2_task_verify_30",
        "scalecua_osworld_train_gimp_b148e375_fe0b_4bec_90e7_38632b0d73c2_task_verify_31",
        "scalecua_osworld_train_gimp_b148e375_fe0b_4bec_90e7_38632b0d73c2_task_verify_32",
        "scalecua_osworld_train_gimp_b148e375_fe0b_4bec_90e7_38632b0d73c2_task_verify_33",
        "scalecua_osworld_train_gimp_b148e375_fe0b_4bec_90e7_38632b0d73c2_task_verify_34",
        "scalecua_osworld_train_gimp_b148e375_fe0b_4bec_90e7_38632b0d73c2_task_verify_35",
        "scalecua_osworld_train_libreoffice_impress_841b50aa_df53_47bd_a73a_22d3a9f73160_task_verify_16",
        "scalecua_osworld_train_libreoffice_impress_9cf05d24_6bd9_4dae_8967_f67d88f5d38a_task_verify_47",
        "scalecua_osworld_train_libreoffice_impress_9cf05d24_6bd9_4dae_8967_f67d88f5d38a_task_verify_55",
        "scalecua_osworld_train_vlc_778efd0a_153f_4842_9214_f05fc176b877_task_verify_26",
        "scalecua_osworld_train_libreoffice_writer_936321ce_5236_426a_9a20_e0e3c5dc536f_task_verify_3",
        "scalecua_osworld_train_libreoffice_writer_72b810ef_4156_4d09_8f08_a0cf57e7cefe_task_verify_45",
        "scalecua_osworld_train_libreoffice_writer_6a33f9b9_0a56_4844_9c3f_96ec3ffb3ba2_task_verify_0",
        "scalecua_osworld_train_libreoffice_writer_0b17a146_2934_46c7_8727_73ff6b6483e8_task_verify_7",
        "scalecua_osworld_train_libreoffice_writer_ecc2413d_8a48_416e_a3a2_d30106ca36cb_task_verify_67",
        "scalecua_osworld_train_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_task_verify_34",
        "scalecua_osworld_train_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_task_verify_45",
        "scalecua_osworld_train_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_task_verify_52",
        "scalecua_osworld_train_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_task_verify_53",
        "scalecua_osworld_train_chrome_f3b19d1e_2d48_44e9_b4e1_defcae1a0197_task_verify_86",
        "scalecua_osworld_train_chrome_f8cfa149_d1c1_4215_8dac_4a0932bad3c2_task_verify_52",
        "scalecua_osworld_train_libreoffice_calc_a82b78bb_7fde_4cb3_94a4_035baf10bcf0_task_verify_48",
        # TRAIN no-op/trivial precheck passes (fresh state already satisfies the check).
        "scalecua_osworld_train_chrome_215dfd39_f493_4bc3_a027_8a97d72c61bf_task_verify_21",
        "scalecua_osworld_train_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_task_verify_17",
        "scalecua_osworld_train_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_task_verify_21",
        "scalecua_osworld_train_multi_apps_9f3bb592_209d_43bc_bb47_d77d9df56504_task_verify_17",
        # TRAIN rows whose upstream gold oracle_actions under-achieve the metric on replay.
        "scalecua_osworld_train_libreoffice_impress_73c99fb9_f828_43ce_b87a_01dc07faa224_task_verify_0",
        "scalecua_osworld_train_libreoffice_writer_66399b0d_8fda_4618_95c4_bfc6191617e9_task_verify_0",
        "scalecua_osworld_train_libreoffice_writer_00fa164e_2612_4439_992e_157d019a8436_task_verify_10",
        "scalecua_osworld_train_multi_apps_5bc63fb9_276a_4439_a7c1_9dc76401737f_task_verify_0",
        "scalecua_osworld_train_multi_apps_da922383_bfa4_4cd3_bbad_6bebab3d7742_task_verify_46",
        "scalecua_osworld_train_os_4d117223_a354_47fb_8b45_62ab1390a95f_task_verify_21",
        "scalecua_osworld_train_vs_code_df67aebb_fb3a_44fd_b75b_51b6012df509_task_verify_0",
        # RL genuine upstream eval defects (non-dbus, deterministic): vlc gold caps at
        # reward 0.5; chrome no-op precheck trivially passes.
        "scalecua_osworld_rl_vlc_59f21cfb_0120_4326_b255_a5b827b38967_traj_verify_3",
        "scalecua_osworld_rl_vlc_59f21cfb_0120_4326_b255_a5b827b38967_traj_verify_6",
        "scalecua_osworld_rl_vlc_59f21cfb_0120_4326_b255_a5b827b38967_traj_verify_7",
        "scalecua_osworld_rl_chrome_f3977615_2b45_4ac5_8bba_80c17dbe2a37_traj_verify_1",
    }
    assert dataset.UPSTREAM_GENERATED_EVAL_BUG_TASK_IDS == buggy_task_ids

    payload = {
        "instruction": (
            "Sort the table B2:F5 based on the Marks in row 5, from highest "
            "to lowest, while preserving row groupings."
        ),
        "evaluator": {
            "func": "check_xlsx_sorted_by_column__447f5fcf01e57a045588cce9783df3a1",
            "result": {
                "type": "xlsx_table_data__447f5fcf01e57a045588cce9783df3a1",
                "path": "/home/user/Students_Class_Subject_Marks.xlsx",
                "sheet": 0,
                "range": "B2:F5",
            },
            "expected": {
                "type": "rule",
                "rules": {"descending": True, "row_idx": 3},
            },
        },
    }

    for task_id in buggy_task_ids:
        assert (
            dataset._exclude_reason(
                payload,
                task_id=task_id,
                inherited_exclusion=None,
                unsupported=[],
                runtime_split="train",
                broken_metrics=frozenset(),
            )
            == "upstream_generated_eval_bug"
        )
    assert (
        dataset._exclude_reason(
            payload,
            task_id=(
                "scalecua_osworld_train_libreoffice_impress_eb303e01_261e_"
                "4972_8c07_c9b4e7a4922a_task_verify_15"
            ),
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        is None
    )
    assert (
        dataset._exclude_reason(
            payload,
            task_id=(
                "scalecua_osworld_train_thunderbird_dd84e895_72fd_4023_"
                "a336_97689ded257c_task_verify_4"
            ),
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        is None
    )
    assert (
        dataset._exclude_reason(
            payload,
            task_id=(
                "scalecua_osworld_train_libreoffice_calc_"
                "2bd59342_0664_4ccb_ba87_79379096cc08_task_verify_55"
            ),
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        == "infeasible"
    )
    assert (
        dataset._exclude_reason(
            payload,
            task_id=(
                "scalecua_osworld_train_libreoffice_calc_"
                "eb03d19a_b88d_4de4_8a64_ca0ac66f426b_task_verify_42"
            ),
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        is None
    )
    assert (
        dataset._exclude_reason(
            payload,
            task_id=(
                "scalecua_osworld_train_os_6f56bf42_85b8_4fbb_8e06_6c44960184ba_task_verify_45"
            ),
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        is None
    )


def test_scalecua_excludes_uncompilable_python_heredoc_oracle_script():
    # Upstream generated_tasks defect: gold oracle script's `import` line is left
    # flush while the body kept an indent -> IndentationError on line 2. The task
    # cannot run, so it is filtered as upstream_generated_eval_bug.
    malformed = {
        "oracle_actions": [
            {
                "type": "execute",
                "parameters": {
                    "shell": True,
                    "command": (
                        "python3 - <<'PY'\nimport os, fitz\n"
                        "        p = '/home/user/Documents/a.pdf'\n"
                        "        os.makedirs(os.path.dirname(p), exist_ok=True)\nPY"
                    ),
                },
            }
        ]
    }
    assert dataset._has_uncompilable_python_heredoc(malformed, runtime_split="train")
    assert (
        dataset._exclude_reason(
            malformed,
            task_id="scalecua_osworld_train_chrome_deadbeef_task_verify_0",
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        == "upstream_generated_eval_bug"
    )
    # A well-formed python heredoc is NOT flagged.
    well_formed = {
        "oracle_actions": [
            {
                "type": "execute",
                "parameters": {
                    "shell": True,
                    "command": "python3 - <<'PY'\nimport os\np = '/home/user/a'\nPY",
                },
            }
        ]
    }
    assert not dataset._has_uncompilable_python_heredoc(well_formed, runtime_split="train")
    # A heredoc carrying an unsubstituted {PLACEHOLDER} (valid at runtime, only a
    # SyntaxError before substitution) must NOT be mistaken for a defect.
    templated = {
        "oracle_actions": [
            {
                "type": "execute",
                "parameters": {
                    "shell": True,
                    "command": "python3 - <<'PY'\nx = '{TARGET_PATH\np = 1\nPY",
                },
            }
        ]
    }
    assert not dataset._has_uncompilable_python_heredoc(templated, runtime_split="train")


def test_scalecua_upstream_generated_eval_bug_does_not_blanket_extension_load_family():
    extension_load_payload = {
        "instruction": "Load the unpacked Chrome extension from the Desktop.",
        "evaluator": {
            "func": "is_in_list",
            "result": {"type": "find_unpacked_extension_path"},
            "expected": {
                "type": "rule",
                "rules": {"expected": "/home/user/Desktop/helloExtension"},
            },
        },
    }
    negative_controls = [
        (
            "scalecua_osworld_train_chrome_6766f2b8_8a72_417f_a9e5_56fcaa735837_task_verify_76",
            "train",
        ),
        (
            "scalecua_osworld_rl_chrome_6766f2b8_8a72_417f_a9e5_56fcaa735837_traj_verify_4",
            "rl",
        ),
        (
            "scalecua_osworld_train_multi_apps_a74b607e_6bb5_4ea8_8a7c_5d97c7bbcd2a_task_verify_90",
            "train",
        ),
        (
            "scalecua_osworld_rl_multi_apps_a74b607e_6bb5_4ea8_8a7c_5d97c7bbcd2a_traj_verify_1",
            "rl",
        ),
    ]

    assert not dataset.UPSTREAM_GENERATED_EVAL_BUG_TASK_IDS.intersection(
        task_id for task_id, _ in negative_controls
    )
    for task_id, runtime_split in negative_controls:
        assert (
            dataset._exclude_reason(
                extension_load_payload,
                task_id=task_id,
                inherited_exclusion=None,
                unsupported=[],
                runtime_split=runtime_split,
                broken_metrics=frozenset(),
            )
            != "upstream_generated_eval_bug"
        )


def test_scalecua_excludes_exact_missing_proxy_social_tab_rows(tmp_path):
    payload = {
        "instruction": "Reopen the tab that was most recently closed.",
        "config": [],
        "evaluator": {
            "func": "is_expected_tabs",
            "result": {"type": "open_tabs_info"},
            "expected": {
                "type": "rule",
                "rules": {
                    "type": "url",
                    "urls": [
                        "https://www.reddit.com",
                        "https://www.twitter.com",
                        "https://www.facebook.com",
                    ],
                },
            },
        },
    }

    for index in range(30, 34):
        assert (
            dataset._exclude_reason(
                payload,
                task_id=(
                    "scalecua_osworld_train_chrome_"
                    f"06fe7178_4491_4589_810f_2e2bc9502122_task_verify_{index}"
                ),
                inherited_exclusion=None,
                unsupported=[],
                runtime_split="train",
                broken_metrics=frozenset(),
            )
            == "proxy_required"
        )
    assert (
        dataset._exclude_reason(
            payload,
            task_id=(
                "scalecua_osworld_train_chrome_06fe7178_4491_4589_810f_2e2bc9502122_task_verify_34"
            ),
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        == "upstream_live_site_drift"
    )
    row = dataset._row_from_payload(
        payload,
        runtime_split="train",
        source_name="generated_tasks",
        source_domain="chrome",
        source_path=(tmp_path / "06fe7178-4491-4589-810f-2e2bc9502122_task_verify_30.json"),
        inherited_exclusion=None,
        context=dataset._ImportContext(snapshot=tmp_path, broken_metrics=frozenset()),
    )

    assert row["metadata"]["others"]["proxy"] is True
    assert row["metadata"]["others"]["exclude_reason"] == "proxy_required"


def test_scalecua_excludes_generated_babycenter_exact_url_live_site_drift():
    payload = {
        "source": "Mind2Web",
        "instruction": "Find baby names that are similar to liam",
        "evaluator": {
            "func": "is_expected_active_tab",
            "result": {
                "type": "active_url_from_accessTree",
                "goto_prefix": "https://www.",
            },
            "expected": {
                "type": "rule",
                "rules": {
                    "type": "url",
                    "url": "https://www.babycenter.com/baby-names/details/liam-5051",
                },
            },
        },
    }
    regex_payload = {
        "source": "Mind2Web",
        "instruction": "Look up baby name details for Liam on BabyCenter.",
        "evaluator": {
            "func": "is_expected_url_pattern_match",
            "result": {
                "type": "active_url_from_accessTree",
                "goto_prefix": "https://www.",
            },
            "expected": {
                "type": "rule",
                "rules": {"expected": ["babycenter\\.com/baby-names/details/liam"]},
            },
        },
    }

    assert (
        dataset._exclude_reason(
            payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        == "upstream_live_site_drift"
    )
    assert (
        dataset._exclude_reason(
            regex_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="rl",
            broken_metrics=frozenset(),
        )
        is None
    )


def test_scalecua_excludes_macys_live_url_filter_drift_without_overfiltering():
    macys_filter_payload = {
        "id": "2888b4e6-5b47-4b57-8bf5-c73827890774",
        "instruction": (
            "Display all men's extra-large short-sleeve shirts with a discount of 40% or more."
        ),
        "evaluator": {
            "func": "check_direct_json_object",
            "result": {
                "type": "url_path_parse",
                "goto_prefix": "https://www.",
                "parse_keys": [
                    "mens_clothing",
                    "shirts",
                    "Men_regular_size_t",
                    "Price_discount_range",
                    "short_sleeve",
                ],
            },
            "expected": {
                "type": "rule",
                "rules": {
                    "expected": {
                        "mens_clothing": True,
                        "shirts": True,
                        "Men_regular_size_t": "XL",
                        "Price_discount_range": "40_PERCENT_ off & more",
                        "short_sleeve": True,
                    }
                },
            },
        },
    }
    macys_rl_browse_payload = {
        "id": "2888b4e6-5b47-4b57-8bf5-c73827890774",
        "instruction": "Navigate to the men's shirts section on Macy's website.",
        "evaluator": {
            "func": "is_expected_url_pattern_match",
            "result": {
                "type": "active_url_from_accessTree",
                "goto_prefix": "https://www.",
            },
            "expected": {
                "type": "rule",
                "rules": {"expected": ["macys\\.com.*shop/mens.*shirts"]},
            },
        },
    }
    macys_bookmark_payload = {
        "id": "2888b4e6-5b47-4b57-8bf5-c73827890774",
        "instruction": "Add the Macy's homepage to the bookmarks bar.",
        "evaluator": {
            "func": "check_macys_bookmarked__hash",
            "result": {"type": "bookmarks"},
            "expected": {
                "type": "rule",
                "rules": {"target_url": "macys.com"},
            },
        },
    }
    macys_rl_filter_payload = {
        "id": "2888b4e6-5b47-4b57-8bf5-c73827890774",
        "instruction": "Filter men's shirts on Macy's to show short-sleeve options in size L.",
        "evaluator": {
            "func": "check_url_filters__c943b11f811330084211b7bb4efc17c3",
            "result": {
                "type": "active_url_from_accessTree",
                "goto_prefix": "https://www.",
            },
            "expected": {
                "type": "rule",
                "rules": {"short_sleeve": True, "size_l": True},
            },
        },
    }
    unrelated_downloads_payload = {
        "id": "2888b4e6-5b47-4b57-8bf5-c73827890774",
        "instruction": "Open Chrome's downloads page.",
        "evaluator": {
            "func": "is_expected_active_tab_approximate",
            "result": {"type": "active_tab_info"},
            "expected": {
                "type": "rule",
                "rules": {"type": "url", "url": "chrome://downloads"},
            },
        },
    }

    assert (
        dataset._exclude_reason(
            macys_filter_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        == "upstream_live_site_drift"
    )
    assert (
        dataset._exclude_reason(
            macys_rl_browse_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="rl",
            broken_metrics=frozenset(),
        )
        == "upstream_live_site_drift"
    )
    assert (
        dataset._exclude_reason(
            macys_rl_filter_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="rl",
            broken_metrics=frozenset(),
        )
        == "upstream_live_site_drift"
    )

    proxied = dict(macys_filter_payload, proxy=True)
    assert (
        dataset._exclude_reason(
            proxied,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        == "proxy_required"
    )
    assert (
        dataset._exclude_reason(
            macys_bookmark_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="rl",
            broken_metrics=frozenset(),
        )
        is None
    )
    assert (
        dataset._exclude_reason(
            unrelated_downloads_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="rl",
            broken_metrics=frozenset(),
        )
        is None
    )


def test_scalecua_excludes_doubleclick_cookie_live_site_drift_exactly():
    payload = {
        "id": "7b6c7e24-c58a-49fc-a5bb-d57b80e5b4c3",
        "instruction": (
            "Remove all cookies stored by doubleclick.net to stop ad tracking from Amazon pages."
        ),
        "evaluator": {
            "func": "is_cookie_deleted",
            "result": {"type": "cookie_data", "dest": "Cookies"},
            "expected": {
                "type": "rule",
                "rules": {"type": "domains", "domains": [".doubleclick.net"]},
            },
        },
    }

    assert (
        dataset._exclude_reason(
            payload,
            task_id=(
                "scalecua_osworld_rl_chrome_7b6c7e24_c58a_49fc_a5bb_d57b80e5b4c3_traj_verify_2"
            ),
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="rl",
            broken_metrics=frozenset(),
        )
        == "upstream_live_site_drift"
    )
    assert (
        dataset._exclude_reason(
            payload,
            task_id=(
                "scalecua_osworld_rl_chrome_7b6c7e24_c58a_49fc_a5bb_d57b80e5b4c3_traj_verify_1"
            ),
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="rl",
            broken_metrics=frozenset(),
        )
        is None
    )


def test_scalecua_excludes_rl_url_live_site_drift_exactly():
    payload = {
        "id": "live-url",
        "instruction": "Open the requested live website route.",
        "evaluator": {
            "func": "is_expected_url_pattern_match",
            "result": {"type": "active_url_from_accessTree"},
            "expected": {"type": "rule", "rules": {"expected": "/historical-route"}},
        },
    }
    drift_task_ids = {
        "scalecua_osworld_rl_chrome_9f3f70fc_5afc_4958_a7b7_3bb4fcb01805_traj_verify_1",
        "scalecua_osworld_rl_chrome_f0b971a1_6831_4b9b_a50e_22a6e47f45ba_traj_verify_1",
        "scalecua_osworld_rl_chrome_f5d96daf_83a8_4c86_9686_bada31fc66ab_traj_verify_0",
    }

    for task_id in drift_task_ids:
        assert (
            dataset._exclude_reason(
                payload,
                task_id=task_id,
                inherited_exclusion=None,
                unsupported=[],
                runtime_split="rl",
                broken_metrics=frozenset(),
            )
            == "upstream_live_site_drift"
        )
    assert (
        dataset._exclude_reason(
            payload,
            task_id=(
                "scalecua_osworld_rl_chrome_f0b971a1_6831_4b9b_a50e_22a6e47f45ba_traj_verify_0"
            ),
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="rl",
            broken_metrics=frozenset(),
        )
        is None
    )


def test_scalecua_excludes_doj_forms_component_id_drift_without_overfiltering():
    doj_antitrust_payload = {
        "id": "9f935cce-0a9f-435f-8007-817732bfc0a5",
        "instruction": "Locate and access the list of Antitrust Division forms.",
        "evaluator": {
            "func": "is_expected_url_pattern_match",
            "result": {
                "type": "active_url_from_accessTree",
                "goto_prefix": "https://www.",
            },
            "expected": {
                "type": "rule",
                "rules": {"expected": ["forms\\?title=&field_component_target_id=401"]},
            },
        },
    }
    doj_generic_payload = {
        "id": "9f935cce-0a9f-435f-8007-817732bfc0a5",
        "instruction": "Navigate to the DOJ publications page.",
        "evaluator": {
            "func": "is_expected_url_pattern_match",
            "result": {"type": "active_url_from_accessTree"},
            "expected": {
                "type": "rule",
                "rules": {"expected": ["justice\\.gov/publications"]},
            },
        },
    }

    assert (
        dataset._exclude_reason(
            doj_antitrust_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        == "upstream_live_site_drift"
    )
    assert (
        dataset._exclude_reason(
            dict(doj_antitrust_payload, proxy=True),
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        == "proxy_required"
    )
    assert (
        dataset._exclude_reason(
            doj_generic_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        is None
    )


def test_scalecua_excludes_dmv_old_path_drift_without_overfiltering():
    dmv_old_path_payload = {
        "id": "a728a36e-8bf1-4bb6-9a03-ef039a5233f0",
        "instruction": "Visit the Virginia DMV ID card application requirements page.",
        "evaluator": {
            "func": "is_expected_url_pattern_match",
            "result": {
                "type": "active_url_from_accessTree",
                "goto_prefix": "https://www.",
            },
            "expected": {
                "type": "rule",
                "rules": {
                    "expected": ["^https://(www\\.)?dmv\\.virginia\\.gov/licenses-ids/id/applying"]
                },
            },
        },
    }
    dmv_current_exact_payload = {
        "id": "a728a36e-8bf1-4bb6-9a03-ef039a5233f0",
        "instruction": "Open the DMV office locations page.",
        "evaluator": {
            "func": "is_expected_active_tab",
            "result": {"type": "active_url_from_accessTree"},
            "expected": {
                "type": "rule",
                "rules": {"type": "url", "url": "https://www.dmv.virginia.gov/locations"},
            },
        },
    }
    dmv_current_regex_payload = {
        "id": "a728a36e-8bf1-4bb6-9a03-ef039a5233f0",
        "instruction": "Open the driver license eligibility page.",
        "evaluator": {
            "func": "is_expected_url_pattern_match",
            "result": {"type": "active_url_from_accessTree"},
            "expected": {
                "type": "rule",
                "rules": {
                    "expected": [
                        "^https://(www\\.)?dmv\\.virginia\\.gov/licenses-ids/license/applying/eligibility"
                    ]
                },
            },
        },
    }

    assert (
        dataset._exclude_reason(
            dmv_old_path_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        == "upstream_live_site_drift"
    )
    assert (
        dataset._exclude_reason(
            dmv_current_exact_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        is None
    )
    assert (
        dataset._exclude_reason(
            dmv_current_regex_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="rl",
            broken_metrics=frozenset(),
        )
        is None
    )


def test_scalecua_excludes_flightaware_category_redirect_drift():
    flightaware_category_payload = {
        "id": "a96b564e-dbe9-42c3-9ccf-b4498073438a",
        "instruction": "Open FlightAware Feature Requests or Suggestions.",
        "evaluator": {
            "func": "is_expected_url_pattern_match",
            "result": {
                "type": "active_tab_info",
                "goto_prefix": "https://www.",
            },
            "expected": {
                "type": "rule",
                "rules": {
                    "expected": [
                        "https://discussions\\.flightaware\\.com/c/(feature-requests|suggestions|ideas)/"
                    ]
                },
            },
        },
    }
    flightaware_latest_payload = {
        "id": "a96b564e-dbe9-42c3-9ccf-b4498073438a",
        "instruction": "Open the latest FlightAware discussions.",
        "evaluator": {
            "func": "is_expected_url_pattern_match",
            "result": {"type": "active_url_from_accessTree"},
            "expected": {
                "type": "rule",
                "rules": {"expected": ["https://discussions\\.flightaware\\.com/latest"]},
            },
        },
    }

    assert (
        dataset._exclude_reason(
            flightaware_category_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        == "upstream_live_site_drift"
    )
    assert (
        dataset._exclude_reason(
            dict(flightaware_category_payload, proxy=True),
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        == "proxy_required"
    )
    assert (
        dataset._exclude_reason(
            flightaware_latest_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="rl",
            broken_metrics=frozenset(),
        )
        is None
    )


def test_scalecua_excludes_united_special_needs_path_drift_without_overfiltering():
    united_special_needs_payload = {
        "id": "c1fa57f3-c3db-4596-8f09-020701085416",
        "instruction": "Navigate to United Airlines' special assistance services webpage.",
        "evaluator": {
            "func": "is_expected_url_pattern_match",
            "result": {
                "type": "active_tab_info",
                "goto_prefix": "https://www.",
            },
            "expected": {
                "type": "rule",
                "rules": {"expected": ["united.com/en/us/fly/travel/special-needs"]},
            },
        },
    }
    united_contact_payload = {
        "id": "c1fa57f3-c3db-4596-8f09-020701085416",
        "instruction": "Open the United customer service contact page.",
        "evaluator": {
            "func": "is_expected_url_pattern_match",
            "result": {"type": "active_tab_info"},
            "expected": {
                "type": "rule",
                "rules": {"expected": ["united.com/en/us/contact"]},
            },
        },
    }

    assert (
        dataset._exclude_reason(
            united_special_needs_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        == "upstream_live_site_drift"
    )
    assert (
        dataset._exclude_reason(
            dict(united_special_needs_payload, proxy=True),
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        == "proxy_required"
    )
    assert (
        dataset._exclude_reason(
            united_contact_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        is None
    )


def test_scalecua_excludes_generated_shenzhen_address_lookup_drift_only_for_train():
    generated_lookup_payload = {
        "id": "7ff48d5b-2df2-49da-b500-a5150ffc7f18",
        "instruction": (
            "请查询宝安区内三个全天开放的自助签注办理终端地址，并以中文填写到当前编辑的Word文档中。"
        ),
        "evaluator": {
            "func": "check_docx_addresses__fda9f993c465f67855b1fa54901e999b",
            "result": {
                "type": "vm_file",
                "path": "/home/user/Desktop/AllLocations.docx",
                "dest": "AllLocations.docx",
            },
            "expected": {
                "type": "rule",
                "rules": {
                    "addresses": [
                        "深圳市宝安区新安街道创业路1004号",
                        "深圳市宝安区宝城街道新安二路37号",
                        "深圳市宝安区西乡街道宝源路1053号",
                    ],
                    "count": 3,
                },
            },
        },
    }
    rl_static_edit_payload = {
        "id": "7ff48d5b-2df2-49da-b500-a5150ffc7f18",
        "instruction": (
            "Type the title '深圳福田区地址列表' at the top of the open Word document and save it."
        ),
        "evaluator": {
            "func": "check_docx_title__b98b6fc5ce8ebe003cf7eaeacc71d919",
            "result": {
                "type": "docx_text_content__b98b6fc5ce8ebe003cf7eaeacc71d919",
                "path": "/home/user/Desktop/AllLocations.docx",
            },
            "expected": {
                "type": "rule",
                "rules": {"expected_title": "深圳福田区地址列表"},
            },
        },
    }
    min_count_only_lookup_payload = copy.deepcopy(generated_lookup_payload)
    min_count_only_lookup_payload["instruction"] = (
        "I'm a student living in Futian District, Shenzhen. Please find the "
        "addresses of 3 libraries and save them in Chinese to this Word document."
    )
    min_count_only_lookup_payload["evaluator"]["expected"]["rules"] = {"min_count": 3}

    assert (
        dataset._exclude_reason(
            generated_lookup_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        == "upstream_live_site_drift"
    )
    assert (
        dataset._exclude_reason(
            min_count_only_lookup_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="train",
            broken_metrics=frozenset(),
        )
        == "upstream_live_site_drift"
    )
    assert (
        dataset._exclude_reason(
            rl_static_edit_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="rl",
            broken_metrics=frozenset(),
        )
        is None
    )
    assert (
        dataset._exclude_reason(
            generated_lookup_payload,
            inherited_exclusion=None,
            unsupported=[],
            runtime_split="eval",
            broken_metrics=frozenset(),
        )
        is None
    )
