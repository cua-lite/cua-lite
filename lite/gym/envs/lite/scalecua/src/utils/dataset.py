"""Import and validate ScaleCUA OSWorld task catalogs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from lite.gym.envs.lite.osworld import exclude_reasons
from lite.gym.envs.lite.osworld.src.gen.common import gimp_export_as_postconfig
from lite.gym.envs.lite.osworld.src.gen.eval.postconfig import (
    normalize_postconfig as _normalize_osworld_postconfig,
)
from lite.gym.envs.lite.scalecua.src.gen.eval.judge_helpers import (
    patch_judge_functions as _patch_judge_functions,
)
from lite.gym.envs.lite.scalecua.src.utils import assets

RUNTIME_SPLITS = ("train", "rl")
SOURCE_TO_RUNTIME = {
    "generated_tasks": "train",
    "rl_tasks": "rl",
}
RUNTIME_TO_SOURCE = {v: k for k, v in SOURCE_TO_RUNTIME.items()}

EXPECTED_COUNTS = {
    "train": 20289,
    "rl": 2049,
}
EXPECTED_DOMAIN_COUNTS = {
    "train": {
        "chrome": 2143,
        "gimp": 1092,
        "libreoffice_calc": 2705,
        "libreoffice_impress": 2704,
        "libreoffice_writer": 1485,
        "multi_apps": 5250,
        "os": 1479,
        "thunderbird": 827,
        "vlc": 1138,
        "vs_code": 1466,
    },
    "rl": {
        "chrome": 246,
        "gimp": 175,
        "libreoffice_calc": 250,
        "libreoffice_impress": 250,
        "libreoffice_writer": 172,
        "multi_apps": 380,
        "os": 180,
        "thunderbird": 117,
        "vlc": 135,
        "vs_code": 144,
    },
}
EXPECTED_SOURCE_DOMAIN_COUNTS = {
    "train": {
        "chrome": 3492,
        "gimp": 1386,
        "libreoffice_calc": 4224,
        "libreoffice_impress": 2958,
        "libreoffice_writer": 1864,
        "multi_apps": 677,
        "os": 1898,
        "thunderbird": 1136,
        "vlc": 915,
        "vs_code": 1739,
    },
    "rl": {
        "chrome": 300,
        "gimp": 188,
        "libreoffice_calc": 328,
        "libreoffice_impress": 262,
        "libreoffice_writer": 193,
        "multi_apps": 148,
        "os": 201,
        "thunderbird": 132,
        "vlc": 132,
        "vs_code": 165,
    },
}

FILE_CACHE_MAIN = "xlangai/ubuntu_osworld_file_cache/resolve/main"
FILE_CACHE_PINNED = (
    "xlangai/ubuntu_osworld_file_cache/resolve/"
    + assets.FILE_CACHE_REVISION
)
# Deterministic re-point for live-fetch download URLs that flake in oracle
# validation. The Mind2Web NeurIPS'23 D&B PDF on papers.nips.cc is bot-flaky;
# the sibling downloads in the same task (libreoffice_calc_b5062e3e) already use
# arxiv.org, so re-point it to the paper's arXiv mirror (2306.06070) for a
# deterministic fetch matching the siblings.
LIVE_URL_REWRITES = {
    "https://papers.nips.cc/paper_files/paper/2023/file/"
    "5950bf290a1570ea401bf98882128160-Paper-Datasets_and_Benchmarks.pdf":
        "https://arxiv.org/pdf/2306.06070.pdf",
}
GIMP_ACTION_HISTORY = "/home/user/.config/GIMP/2.10/action-history"
PLACEHOLDER_1024_URL = "https://via.placeholder.com/1024x768.png"
URL_RE = re.compile(r"https?://", re.IGNORECASE)
MISSING_ASSET_URL_REFERENCE_RE = re.compile(
    r"\bprovided\s+url\b|\bfrom\s+the\s+link\b|\bsupplied\s+link\b",
    re.IGNORECASE,
)
AUTHOR_RESULTS_PREFIX = "/home/lvbowen/project/AutoGen/results/"
BABYCENTER_EXACT_NAME_URL_RE = re.compile(
    r"^https?://(?:www\.)?babycenter\.com/baby-names/details/[a-z0-9-]+-\d+/?$",
    re.IGNORECASE,
)
MACYS_LIVE_SITE_DRIFT_OSWORLD_ID = "2888b4e6-5b47-4b57-8bf5-c73827890774"
DOJ_FORMS_COMPONENT_DRIFT_OSWORLD_ID = "9f935cce-0a9f-435f-8007-817732bfc0a5"
DMV_LIVE_SITE_DRIFT_OSWORLD_ID = "a728a36e-8bf1-4bb6-9a03-ef039a5233f0"
FLIGHTAWARE_CATEGORY_DRIFT_OSWORLD_ID = "a96b564e-dbe9-42c3-9ccf-b4498073438a"
DELTA_AWARD_LIVE_SITE_DRIFT_OSWORLD_ID = "6c4c23a1-42a4-43cc-9db1-2f86ff3738cc"
UNITED_SPECIAL_NEEDS_DRIFT_OSWORLD_ID = "c1fa57f3-c3db-4596-8f09-020701085416"
SHENZHEN_ADDRESS_LOOKUP_DRIFT_OSWORLD_ID = "7ff48d5b-2df2-49da-b500-a5150ffc7f18"
A462_USER_SWITCH_OSWORLD_ID = "a462a795-fdc7-4b23-b689-e8b6df786b78"
ROOT_HOME_TEST1_OSWORLD_ID = "5812b315-e7bd-4265-b51f-863c02174c28"
UPSTREAM_CHORD_KEY_LIST_REPAIRS = {
    (
        "osworld/generated_tasks/gimp/"
        "a746add2-cab0-4740-ac36-c3769d9bfb46_task_verify_79.json"
    ): (1, "key_press", "ctrl+q"),
    (
        "osworld/generated_tasks/vs_code/"
        "26150609-0da3-4a7d-8868-0faf9c5f01bb_task_verify_23.json"
    ): (0, "key", "ctrl+s"),
}
UPSTREAM_EXPECTED_TEXT_REPAIRS = {
    (
        "osworld/generated_tasks/vs_code/"
        "f918266a-b3e0-4914-865d-4faa564f1aef_task_verify_60.json"
    ): (
        "compare_text_output__d398cd07",
        "11\\n12\\n22\\n25\\n34\\n64\\n90\\n",
        "11\n12\n22\n25\n34\n64\n90\n",
    ),
}
THUNDERBIRD_GMAIL_AUTH_TASK_IDS = {
    *(
        f"scalecua_osworld_train_thunderbird_15c3b339_88f7_4a86_ab16_"
        f"e71c58dcb01e_task_verify_{index}"
        for index in (8, 9, 10, 11, 13, 14, 15, 16, 34, 35, 36, 37, 40)
    ),
    *(
        f"scalecua_osworld_train_thunderbird_dfac9ee8_9bc4_4cdc_b465_"
        f"4a4bfcd2f397_task_verify_{index}"
        for index in (34, 36, 37, 39, 42)
    ),
}
CHROME_WEBSTORE_LIVE_SITE_TASK_IDS = {
    *(
        f"scalecua_osworld_train_multi_apps_873cafdd_a581_47f6_8b33_"
        f"b9696ddb7b05_task_verify_{index}"
        for index in range(39)
    ),
    *(
        f"scalecua_osworld_rl_multi_apps_873cafdd_a581_47f6_8b33_"
        f"b9696ddb7b05_traj_verify_{index}"
        for index in range(3)
    ),
}
MISSING_DEPENDENCY_IMAGEMAGICK_TASK_IDS = {
    *(
        f"scalecua_osworld_train_gimp_d68204bf_11c1_4b13_b48b_"
        f"d303c73d4bf6_task_verify_{index}"
        for index in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 23, 34, 45, 56, 57, 58, 59)
    ),
}
MISSING_DEPENDENCY_JAVA_TASK_IDS = {
    # Official generated NLPSOLVER row requires LibreOffice Java extension
    # support. The current lite.osworld image has OpenJDK headless, but lacks
    # libreoffice-java-common/libreoffice-nlpsolver, and the agent user cannot
    # install them at runtime.
    "scalecua_osworld_train_libreoffice_writer_e1fc0df3_c8b9_4ee7_864c_d0b590d3aa56_task_verify_24",
}
UPSTREAM_LIVE_SITE_DRIFT_TASK_IDS = {
    # Official RL cookie-deletion row depends on the live Amazon pages creating
    # a doubleclick.net cookie during setup. Current Chrome/Amazon state does
    # not materialize that cookie, so a no-op already satisfies the evaluator.
    "scalecua_osworld_rl_chrome_7b6c7e24_c58a_49fc_a5bb_d57b80e5b4c3_traj_verify_2",
    # Official RL URL rows target historical live routes that now redirect to
    # different canonical paths, while the evaluator still requires the old
    # path fragment.
    "scalecua_osworld_rl_chrome_9f3f70fc_5afc_4958_a7b7_3bb4fcb01805_traj_verify_1",
    # Official RL Budget.com rows now hit a live-site access restriction before
    # the expected reservation/filter URL can be reached. Keep this exact so
    # unrelated Office rows that mention budget data remain runnable.
    "scalecua_osworld_rl_chrome_47543840_672a_467d_80df_8f7c3b9788c9_traj_verify_0",
    "scalecua_osworld_rl_chrome_47543840_672a_467d_80df_8f7c3b9788c9_traj_verify_1",
    "scalecua_osworld_rl_chrome_47543840_672a_467d_80df_8f7c3b9788c9_traj_verify_2",
    # Official RL live-site rows depend on current Recreation.gov,
    # TripAdvisor, and Microsoft Bookings client state. Current pages no longer
    # expose the historical title/HTML/form state required by the evaluator.
    "scalecua_osworld_rl_chrome_b4f95342_463e_4179_8c3f_193cd7241fb2_traj_verify_3",
    "scalecua_osworld_rl_chrome_b4f95342_463e_4179_8c3f_193cd7241fb2_traj_verify_4",
    "scalecua_osworld_rl_chrome_b7895e80_f4d1_4648_bee0_4eb45a6f1fa8_traj_verify_0",
    "scalecua_osworld_rl_chrome_b7895e80_f4d1_4648_bee0_4eb45a6f1fa8_traj_verify_2",
    "scalecua_osworld_rl_chrome_da46d875_6b82_4681_9284_653b0c7ae241_traj_verify_2",
    "scalecua_osworld_rl_chrome_f0b971a1_6831_4b9b_a50e_22a6e47f45ba_traj_verify_1",
    "scalecua_osworld_rl_chrome_f5d96daf_83a8_4c86_9686_bada31fc66ab_traj_verify_0",
    "scalecua_osworld_rl_chrome_f5d96daf_83a8_4c86_9686_bada31fc66ab_traj_verify_1",
    "scalecua_osworld_rl_chrome_f5d96daf_83a8_4c86_9686_bada31fc66ab_traj_verify_2",
    # Official generated Google Scholar rows expect a live profile URL with a
    # historical `user=` token. Current Scholar anti-bot/live state makes these
    # exact URL-pattern checks nondeterministic; keep exact and leave adjacent
    # Scholar rows runnable unless separately proven drifted.
    "scalecua_osworld_train_libreoffice_calc_36037439_2044_4b50_b9d1_875b5a332143_task_verify_20",
    "scalecua_osworld_train_libreoffice_calc_36037439_2044_4b50_b9d1_875b5a332143_task_verify_24",
    "scalecua_osworld_train_libreoffice_calc_36037439_2044_4b50_b9d1_875b5a332143_task_verify_35",
    "scalecua_osworld_train_chrome_06fe7178_4491_4589_810f_2e2bc9502122_task_verify_34",
    "scalecua_osworld_train_chrome_a728a36e_8bf1_4bb6_9a03_ef039a5233f0_task_verify_57",
    "scalecua_osworld_train_chrome_b4f95342_463e_4179_8c3f_193cd7241fb2_task_verify_42",
    "scalecua_osworld_train_chrome_b4f95342_463e_4179_8c3f_193cd7241fb2_task_verify_45",
    "scalecua_osworld_train_chrome_f3b19d1e_2d48_44e9_b4e1_defcae1a0197_task_verify_39",
    "scalecua_osworld_train_chrome_f3b19d1e_2d48_44e9_b4e1_defcae1a0197_task_verify_52",
    "scalecua_osworld_train_chrome_f3b19d1e_2d48_44e9_b4e1_defcae1a0197_task_verify_64",
    "scalecua_osworld_train_chrome_f3b19d1e_2d48_44e9_b4e1_defcae1a0197_task_verify_71",
    "scalecua_osworld_train_chrome_9f3f70fc_5afc_4958_a7b7_3bb4fcb01805_task_verify_37",
    # Official generated NBA Store row now reaches the live site's Access
    # Denied edge before the women's Mitchell & Ness jacket filters can be
    # inspected.
    "scalecua_osworld_train_chrome_9f3f70fc_5afc_4958_a7b7_3bb4fcb01805_task_verify_16",
    # Official generated Cars.com rows depend on current Cars.com URL params.
    # The 60-mile rows ask for a distance no longer exposed by the live site,
    # while the 02101/40mi hybrid rows now hit live-site unavailability/fallback
    # behavior before the generated Cars.com URL-param evaluator can be
    # satisfied deterministically.
    "scalecua_osworld_train_chrome_82279c77_8fc6_46f6_9622_3ba96f61b477_task_verify_6",
    "scalecua_osworld_train_chrome_82279c77_8fc6_46f6_9622_3ba96f61b477_task_verify_7",
    "scalecua_osworld_train_chrome_82279c77_8fc6_46f6_9622_3ba96f61b477_task_verify_8",
    "scalecua_osworld_train_chrome_82279c77_8fc6_46f6_9622_3ba96f61b477_task_verify_9",
    "scalecua_osworld_train_chrome_82279c77_8fc6_46f6_9622_3ba96f61b477_task_verify_10",
    "scalecua_osworld_train_chrome_82279c77_8fc6_46f6_9622_3ba96f61b477_task_verify_11",
    # Official generated Virginia DMV row expects a URL path containing
    # `/fees`, but the live vehicle registration fees page is currently served
    # at `/vehicles/registration/exemp-disc-chart`.
    "scalecua_osworld_train_chrome_a728a36e_8bf1_4bb6_9a03_ef039a5233f0_task_verify_39",
    # Official generated Virginia DMV Real ID rows require the historical
    # `https://www.dmv.virginia.gov/licenses-ids/real-id` exact URL. The live
    # site now serves the same page canonically without `www`, while neighboring
    # regex rows that already allow optional `www` remain runnable.
    "scalecua_osworld_train_chrome_a728a36e_8bf1_4bb6_9a03_ef039a5233f0_task_verify_21",
    "scalecua_osworld_train_chrome_a728a36e_8bf1_4bb6_9a03_ef039a5233f0_task_verify_22",
    "scalecua_osworld_train_chrome_a728a36e_8bf1_4bb6_9a03_ef039a5233f0_task_verify_24",
    "scalecua_osworld_train_chrome_a728a36e_8bf1_4bb6_9a03_ef039a5233f0_task_verify_25",
    # Official generated reopen-tab row expects historical naked domains, but
    # current live targets redirect to canonical pages (`about.gitlab.com`,
    # `stackoverflow.com/questions`) that fail the strict open-tabs comparison.
    "scalecua_osworld_train_chrome_06fe7178_4491_4589_810f_2e2bc9502122_task_verify_2",
    # Official generated URL rows hard-code historical United, FlightAware, and
    # Ticketek URL shapes. Current live pages route equivalent user-facing
    # content through new help-center paths, in-page tabs, or canonical map
    # routes.
    "scalecua_osworld_train_chrome_a96b564e_dbe9_42c3_9ccf_b4498073438a_task_verify_39",
    "scalecua_osworld_train_chrome_c1fa57f3_c3db_4596_8f09_020701085416_task_verify_23",
    "scalecua_osworld_train_chrome_c1fa57f3_c3db_4596_8f09_020701085416_task_verify_26",
    "scalecua_osworld_train_chrome_c1fa57f3_c3db_4596_8f09_020701085416_task_verify_56",
    "scalecua_osworld_train_chrome_f3b19d1e_2d48_44e9_b4e1_defcae1a0197_task_verify_19",
    "scalecua_osworld_train_chrome_f3b19d1e_2d48_44e9_b4e1_defcae1a0197_task_verify_23",
    "scalecua_osworld_train_chrome_f3b19d1e_2d48_44e9_b4e1_defcae1a0197_task_verify_35",
    # Official generated Ticketek/Biletix help-center row expects an active-tab
    # URL/title containing both "Accessible" and "FAQ". The current help center
    # routes the correct accessible-seating information to an Accessibility
    # article under `/articles/...Accessibility-Everything-You-Need-to-Know`
    # without an FAQ route/title token.
    "scalecua_osworld_train_chrome_f3b19d1e_2d48_44e9_b4e1_defcae1a0197_task_verify_38",
    "scalecua_osworld_train_chrome_f3b19d1e_2d48_44e9_b4e1_defcae1a0197_task_verify_40",
    "scalecua_osworld_train_chrome_f3b19d1e_2d48_44e9_b4e1_defcae1a0197_task_verify_66",
    # Official generated Steam cart rows target free games, free/integrated DLC,
    # soundtrack download/license flows, or removed/expired event items. Current
    # Steam does not expose these as anonymous shopping-cart products, while the
    # evaluator still requires their names to appear in `/cart/` page HTML.
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
    # Current Recreation.gov no longer exposes historical availability data for
    # this Glacier/Two Medicine row, so the evaluator's "Next Available" target
    # cannot be reached by a correct browser trajectory.
    "scalecua_osworld_train_chrome_b4f95342_463e_4179_8c3f_193cd7241fb2_task_verify_46",
    # The historical NFL playoff-picture target now requires a stable archived
    # or alternate bracket page; the generated evaluator still hard-codes the
    # stale nfl.com route.
    "scalecua_osworld_train_chrome_f0b971a1_6831_4b9b_a50e_22a6e47f45ba_task_verify_53",
}
INSTRUCTION_SETUP_MISMATCH_TASK_IDS = {
    *(
        f"scalecua_osworld_train_os_a462a795_fdc7_4b23_b689_"
        f"e8b6df786b78_task_verify_{index}"
        for index in (0, 1, 12, 23)
    ),
}
FLAKE_CHROME_GUI_EXTENSION_LOAD_TASK_IDS = {
    # Official RL Chrome unpacked-extension rows whose oracle replay is
    # nondeterministic: loading the GUI unpacked extension intermittently fails
    # to register in `chrome://extensions` before the evaluator reads state.
    # Quarantined as a flake pending a runtime fix rather than a task defect.
    *(
        f"scalecua_osworld_rl_chrome_6766f2b8_8a72_417f_a9e5_"
        f"56fcaa735837_traj_verify_{index}"
        for index in range(5)
    ),
}
FLAKE_CHROME_SECURE_PREFS_MAC_TASK_IDS = {
    # Official RL Chrome rows that mutate Secure Preferences-protected settings.
    # The protected-prefs HMAC is seeded for a macOS profile, so the Linux
    # runtime intermittently rejects the write and the evaluator flakes.
    # Quarantined as a flake pending a runtime fix rather than a task defect.
    *(
        f"scalecua_osworld_rl_chrome_9656a811_9b5b_4ddf_99c7_"
        f"5117bcef0626_traj_verify_{index}"
        for index in range(6)
    ),
    *(
        f"scalecua_osworld_rl_chrome_99146c54_4f37_4ab8_9327_"
        f"5f3291665e1e_traj_verify_{index}"
        for index in range(6)
    ),
}
PROXY_REQUIRED_TASK_IDS = {
    # Official generated reopen-tab rows whose setup/eval target live social
    # sites but whose source JSON omitted the proxy flag. Keep this exact: the
    # same OSWorld id also has runnable non-social rows and proxy-flagged rows.
    "scalecua_osworld_train_chrome_06fe7178_4491_4589_810f_2e2bc9502122_task_verify_30",
    "scalecua_osworld_train_chrome_06fe7178_4491_4589_810f_2e2bc9502122_task_verify_31",
    "scalecua_osworld_train_chrome_06fe7178_4491_4589_810f_2e2bc9502122_task_verify_32",
    "scalecua_osworld_train_chrome_06fe7178_4491_4589_810f_2e2bc9502122_task_verify_33",
}
PROXY_REQUIRED_EXCLUDE_ONLY_TASK_IDS = {
    # Official RL Chrome rows whose setup/eval reach live social endpoints that
    # need the proxy, but whose source payload carries no proxy flag. Unlike
    # PROXY_REQUIRED_TASK_IDS these only tag the exclude_reason (`proxy_required`)
    # without asserting a `proxy: True` runtime flag on the row.
    *(
        f"scalecua_osworld_rl_chrome_b070486d_e161_459b_aa2b_"
        f"ef442d973b92_traj_verify_{index}"
        for index in range(3)
    ),
}
INSTRUCTION_EVAL_MISMATCH_TASK_IDS = {
    # Official generated rows whose instruction asks for a how-to/list/compare
    # answer, while the generated evaluator requires mutating hidden app/browser
    # state. Keep exact; neighboring rows from the same source ids may be valid
    # state-changing desktop tasks.
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
    # Official generated bookmark rows ask only to add/save a bookmark, while
    # the generated evaluator specifically requires the URL on the bookmarks
    # bar. Chrome's default star flow can validly save under All Bookmarks, so
    # these rows are instruction/eval mismatches rather than runtime failures.
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
    # Official generated terminal-profile rows ask for a how-to answer
    # ("Guide/Explain/Show me how...") while their evaluators require mutating
    # GNOME Terminal profile state.
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
INFEASIBLE_TASK_IDS = {
    # Same generated sparkline/inline chart family as neighboring rows already
    # marked infeasible upstream; this leaked row uses a concrete evaluator
    # even though LibreOffice cannot create Excel sparklines.
    "scalecua_osworld_train_libreoffice_calc_2bd59342_0664_4ccb_ba87_79379096cc08_task_verify_55",
    # Official RL GIMP rows in the 38f48d40 family whose task requires a GIMP
    # capability the runtime cannot provide; the sibling rows (verify_0..2) are
    # already tagged infeasible upstream, while these leaked rows carry concrete
    # evaluators.
    *(
        f"scalecua_osworld_rl_gimp_38f48d40_764e_4e77_a7cf_"
        f"51dfce880291_traj_verify_{index}"
        for index in (3, 4, 5, 6)
    ),
}
UPSTREAM_GENERATED_EVAL_BUG_TASK_IDS = {
    # Official ScaleCUA RL rows whose no-op precheck already returns reward 1.0
    # in the fresh OSWorld desktop state. These are upstream initial-state/eval
    # mismatches, so they must be filtered instead of counted as valid oracle
    # coverage.
    "scalecua_osworld_rl_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_traj_verify_2",
    "scalecua_osworld_rl_vs_code_53ad5833_3455_407b_bbc6_45b4c79ab8fb_traj_verify_4",
    # Official ScaleCUA RL composite getter captures `ls` stdout into
    # backup_exists, producing `backup_exists:/path\nyes` while the metric
    # requires the literal substring `backup_exists:yes`. A correct solution can
    # satisfy only the process half, so the task is capped at reward 0.5.
    "scalecua_osworld_rl_multi_apps_2b9493d7_49b8_493a_a71b_56cd1f4d6908_traj_verify_3",
    # Official generated OS rows use exact_match against stdout from
    # user-facing CLIs that normally end with a newline (`hostname`,
    # `powerprofilesctl get`, and `gsettings get ...`). The task target can be
    # reached, but the generated expected strings omit the trailing newline.
    "scalecua_osworld_rl_os_a462a795_fdc7_4b23_b689_e8b6df786b78_traj_verify_0",
    "scalecua_osworld_rl_os_a462a795_fdc7_4b23_b689_e8b6df786b78_traj_verify_1",
    "scalecua_osworld_rl_os_b3d4a89c_53f2_4d6b_8b6a_541fb5d205fa_traj_verify_0",
    "scalecua_osworld_rl_os_b3d4a89c_53f2_4d6b_8b6a_541fb5d205fa_traj_verify_1",
    "scalecua_osworld_rl_os_b3d4a89c_53f2_4d6b_8b6a_541fb5d205fa_traj_verify_2",
    # Official generated rows whose baseline desktop/app state already matches
    # the target state, so the no-op precheck returns reward 1.0.
    "scalecua_osworld_rl_os_94d95f96_9699_4208_98ba_3c3119edf9c2_traj_verify_3",
    "scalecua_osworld_rl_libreoffice_writer_e1fc0df3_c8b9_4ee7_864c_d0b590d3aa56_traj_verify_2",
    "scalecua_osworld_rl_vs_code_971cbb5b_3cbf_4ff7_9e24_b5c84fcebfa6_traj_verify_5",
    # Official RL rows whose no-op precheck already earns partial or full
    # reward in the fresh imported state. They cannot become strict oracle
    # fixtures because the no-op-negative gate fails before replay.
    "scalecua_osworld_rl_chrome_788b3701_3ec9_4b67_b679_418bfa726c22_traj_verify_0",
    "scalecua_osworld_rl_libreoffice_impress_c59742c0_4323_4b9d_8a02_723c251deaa0_traj_verify_3",
    "scalecua_osworld_rl_libreoffice_impress_5d901039_a89c_4bfb_967b_bf66f4df075e_traj_verify_0",
    "scalecua_osworld_rl_libreoffice_impress_9ec204e4_f0a3_42f8_8458_b772a6797cab_traj_verify_1",
    "scalecua_osworld_rl_libreoffice_impress_af23762e_2bfd_4a1d_aada_20fa8de9ce07_traj_verify_2",
    "scalecua_osworld_rl_libreoffice_impress_ed43c15f_00cb_4054_9c95_62c880865d68_traj_verify_1",
    # Official RL Impress UI-state rows from `ef9d12bd` are already partially
    # or fully satisfied immediately after setup, so strict no-op-negative
    # oracle validation cannot produce meaningful evidence.
    "scalecua_osworld_rl_libreoffice_impress_ef9d12bd_bcee_4ba0_a40e_918400f43ddf_traj_verify_1",
    "scalecua_osworld_rl_libreoffice_impress_ef9d12bd_bcee_4ba0_a40e_918400f43ddf_traj_verify_2",
    "scalecua_osworld_rl_libreoffice_impress_ef9d12bd_bcee_4ba0_a40e_918400f43ddf_traj_verify_4",
    "scalecua_osworld_rl_libreoffice_impress_ef9d12bd_bcee_4ba0_a40e_918400f43ddf_traj_verify_5",
    # Official RL Impress generated metrics have code defects: the `550ce7e7`
    # rows read non-public python-pptx Font.strike state, while `af2d657a`
    # imports PP_PLACEHOLDER from pptx.util where it does not exist.
    "scalecua_osworld_rl_libreoffice_impress_550ce7e7_747b_495f_b122_acdc4d0b8e54_traj_verify_4",
    "scalecua_osworld_rl_libreoffice_impress_550ce7e7_747b_495f_b122_acdc4d0b8e54_traj_verify_5",
    "scalecua_osworld_rl_libreoffice_impress_550ce7e7_747b_495f_b122_acdc4d0b8e54_traj_verify_6",
    "scalecua_osworld_rl_libreoffice_impress_af2d657a_e6b3_4c6a_9f67_9e3ed015974c_traj_verify_3",
    "scalecua_osworld_rl_libreoffice_writer_8df7e444_8e06_4f93_8a1a_c5c974269d82_traj_verify_1",
    "scalecua_osworld_rl_libreoffice_calc_37608790_6147_45d0_9f20_1137bb35703d_traj_verify_3",
    "scalecua_osworld_rl_libreoffice_calc_ecb0df7a_4e8d_4a03_b162_053391d3afaf_traj_verify_0",
    # Remaining RL Calc tail rows whose official generated metrics award
    # partial reward to an untouched workbook, so they fail the strict no-op
    # gate before any oracle action can run.
    "scalecua_osworld_rl_libreoffice_calc_1954cced_e748_45c4_9c26_9855b97fbc5e_traj_verify_5",
    "scalecua_osworld_rl_libreoffice_calc_1de60575_bb6e_4c3d_9e6a_2fa699f9f197_traj_verify_4",
    "scalecua_osworld_rl_libreoffice_calc_3a7c8185_25c1_4941_bd7b_96e823c9f21f_traj_verify_0",
    "scalecua_osworld_rl_libreoffice_calc_3a7c8185_25c1_4941_bd7b_96e823c9f21f_traj_verify_1",
    "scalecua_osworld_rl_libreoffice_calc_51b11269_2ca8_4b2a_9163_f21758420e78_traj_verify_2",
    "scalecua_osworld_rl_libreoffice_calc_7a4e4bc8_922c_4c84_865c_25ba34136be1_traj_verify_2",
    # Official generated exact FP rows: lax notes-count,
    # partial top-N, and image-present checks accept states that do not satisfy
    # the generated instruction. Keep exact; neighboring variants in the same
    # source families are not proven broken.
    "scalecua_osworld_train_libreoffice_impress_eb303e01_261e_4972_8c07_c9b4e7a4922a_task_verify_14",
    "scalecua_osworld_train_libreoffice_impress_eb303e01_261e_4972_8c07_c9b4e7a4922a_task_verify_16",
    "scalecua_osworld_train_libreoffice_calc_1de60575_bb6e_4c3d_9e6a_2fa699f9f197_task_verify_43",
    "scalecua_osworld_train_libreoffice_impress_bf4e9888_f10f_47af_8dba_76413038b73c_task_verify_43",
    # Remaining RL Writer rows from current_missing_52 whose official metrics
    # award nonzero reward to untouched documents in a fresh desktop state
    # (validated precheck rewards: 0.25, 0.3, 0.4803921568627451, 0.5, 0.8).
    # They cannot be strict oracle fixtures because the no-op-negative gate
    # fails before replay.
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
    "scalecua_osworld_rl_gimp_734d6579_c07d_47a8_9ae2_13339795476b_traj_verify_0",
    "scalecua_osworld_rl_multi_apps_415ef462_bed3_493a_ac36_ca8c6d23bf1b_traj_verify_2",
    "scalecua_osworld_rl_multi_apps_47f7c0ce_a5fb_4100_a5e6_65cd0e7429e5_traj_verify_4",
    "scalecua_osworld_rl_multi_apps_716a6079_22da_47f1_ba73_c9d58f986a38_traj_verify_2",
    "scalecua_osworld_rl_os_3ce045a0_877b_42aa_8d2c_b4a863336ab8_traj_verify_6",
    "scalecua_osworld_rl_os_3f05f3b9_29ba_4b6b_95aa_2204697ffc06_traj_verify_1",
    "scalecua_osworld_rl_thunderbird_9bc3cc16_074a_45ac_9bdc_b2a362e1daf3_traj_verify_2",
    # Remaining exact Thunderbird UI/generated-eval tail rows from
    # thunderbird_tail_probe*_20260716. No-op reward is 0.0, but strict
    # replays cannot reach reward 1.0 because the generated checks require
    # password text, exact IMAP host strings, or page-tab selectors that do not
    # materialize in Thunderbird's accessibility tree for the visible state.
    "scalecua_osworld_rl_thunderbird_15c3b339_88f7_4a86_ab16_e71c58dcb01e_traj_verify_2",
    "scalecua_osworld_rl_thunderbird_15c3b339_88f7_4a86_ab16_e71c58dcb01e_traj_verify_4",
    "scalecua_osworld_rl_thunderbird_15c3b339_88f7_4a86_ab16_e71c58dcb01e_traj_verify_6",
    "scalecua_osworld_rl_thunderbird_7b1e1ff9_bb85_49be_b01d_d6424be18cd0_traj_verify_0",
    "scalecua_osworld_rl_thunderbird_7b1e1ff9_bb85_49be_b01d_d6424be18cd0_traj_verify_1",
    "scalecua_osworld_rl_thunderbird_7b1e1ff9_bb85_49be_b01d_d6424be18cd0_traj_verify_5",
    # Official generated Thunderbird rows ask to star every message in the
    # Articles folder, but the imported profile has no folderID 15 messages.
    # The SQL therefore passes on no-op via count(starred)=0 == count(all)=0.
    "scalecua_osworld_train_thunderbird_dd84e895_72fd_4023_a336_97689ded257c_task_verify_2",
    "scalecua_osworld_train_thunderbird_dd84e895_72fd_4023_a336_97689ded257c_task_verify_3",
    "scalecua_osworld_train_thunderbird_dd84e895_72fd_4023_a336_97689ded257c_task_verify_34",
    "scalecua_osworld_train_thunderbird_dd84e895_72fd_4023_a336_97689ded257c_task_verify_35",
    # Official generated VS Code active-file getter parses window titles only
    # when the title part still contains a second " - ". A correct Code window
    # titled "main.py - Visual Studio Code" therefore yields active_file=None.
    "scalecua_osworld_rl_vs_code_53ad5833_3455_407b_bbc6_45b4c79ab8fb_traj_verify_6",
    # Official generated VS Code open-file rows call non-existent eval extension
    # commands (`GetOpenFile` / `OpenFile`) and wait for `/home/user/OpenFile.txt`.
    # The bundled official vscodeEvalExtension for this source id only registers
    # OpenProject, GetColorTheme, and GetBreakPoint, so even visually correct
    # editor states cannot produce the expected result file.
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
    # Official generated Writer tab-stop metric compares UNO enum strings
    # directly. The getter returns "RIGHT (2)" for a correct right tab stop,
    # while the metric accepts only "RIGHT" or "END", making reward 1.0
    # unreachable without changing official evaluator semantics.
    "scalecua_osworld_rl_libreoffice_writer_0a0faba3_5580_44df_965d_f562a99b291c_traj_verify_6",
    # Official generated Writer email-bold rows assume the source document
    # contains `service@blcup.com`, but the actual DOCX text is
    # `Support Contact: service blcup.com` and contains no regex-valid email.
    "scalecua_osworld_train_libreoffice_writer_0a0faba3_5580_44df_965d_f562a99b291c_task_verify_39",
    "scalecua_osworld_train_libreoffice_writer_0a0faba3_5580_44df_965d_f562a99b291c_task_verify_40",
    "scalecua_osworld_train_libreoffice_writer_0a0faba3_5580_44df_965d_f562a99b291c_task_verify_41",
    "scalecua_osworld_train_libreoffice_writer_0a0faba3_5580_44df_965d_f562a99b291c_task_verify_42",
    # Official generated Writer paragraph-index row asks to italicize the second
    # paragraph, but the source DOCX has paragraph 0 as the title, paragraph 1 as
    # the first body paragraph, and paragraph 2 as the second body paragraph. The
    # generated evaluator hard-codes `para_idx=1`, so a visually correct second
    # body paragraph edit is scored against the wrong paragraph.
    "scalecua_osworld_train_libreoffice_writer_e528b65e_1107_4b8c_8988_490e4fece599_task_verify_57",
    # Official generated OS image-copy metric walks dictionary keys instead of
    # child filenames; validated artifacts contained all expected image files
    # but still scored 0.0.
    "scalecua_osworld_rl_os_23393935_50c7_4a86_aeea_2b78fd089c5c_traj_verify_1",
    # Official generated OS rows change the instruction/expected result to
    # `report.txt` in `archive*` directories, but their postconfig still
    # downloads the original `eval.sh` that checks `dir*/file1`.
    "scalecua_osworld_train_os_6f56bf42_85b8_4fbb_8e06_6c44960184ba_task_verify_43",
    "scalecua_osworld_train_os_6f56bf42_85b8_4fbb_8e06_6c44960184ba_task_verify_44",
    "scalecua_osworld_train_os_6f56bf42_85b8_4fbb_8e06_6c44960184ba_task_verify_46",
    "scalecua_osworld_train_os_6f56bf42_85b8_4fbb_8e06_6c44960184ba_task_verify_47",
    # Official generated OS notebook-delete row asks for failed notebooks to
    # remain while excluding "d.ipynb"; the substring metric necessarily matches
    # "d.ipynb" inside every "*_failed.ipynb" include, so reward 1.0 is
    # unreachable under the official evaluator.
    "scalecua_osworld_rl_os_5c1075ca_bb34_46a3_a7a0_029bd7463e79_traj_verify_1",
    # Official generated evaluator sorts B2:F5 by row 5, but the range includes
    # the row-label cell ("Marks") and the generated metric compares it with
    # numeric mark values.
    "scalecua_osworld_train_libreoffice_calc_eb03d19a_b88d_4de4_8a64_ca0ac66f426b_task_verify_41",
    # Official generated Calc rows whose task instructions and generated
    # evaluator assumptions disagree: sheet names, required label text, header
    # cells, percentage normalization, or fixed output cell coordinates.
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
    # Official generated Chrome-source row operating on tally_book.xlsx has
    # impossible expected cached formula values: the source workbook has
    # B2*C2=2307.132 and B6*C6=16184.96, while generated expected values are
    # 2306.592 and 16184.0.
    "scalecua_osworld_train_chrome_788b3701_3ec9_4b67_b679_418bfa726c22_task_verify_6",
    "scalecua_osworld_train_chrome_788b3701_3ec9_4b67_b679_418bfa726c22_task_verify_19",
    # Official generated Chrome rows with evaluator semantics that are stricter
    # or invalid relative to the task text/schema: privacy/security menu
    # navigation accepts only chrome://settings/privacy, VLC global shortcut
    # accepts only vlcrc global-key-stop, and bookmark rules omit the metric's
    # required type field.
    "scalecua_osworld_train_chrome_12086550_11c0_466b_b367_1d9e75b3910e_task_verify_28",
    "scalecua_osworld_train_chrome_386dbd0e_0241_4a0a_b6a2_6704fba26b1c_task_verify_11",
    "scalecua_osworld_train_chrome_386dbd0e_0241_4a0a_b6a2_6704fba26b1c_task_verify_8",
    "scalecua_osworld_train_chrome_386dbd0e_0241_4a0a_b6a2_6704fba26b1c_task_verify_9",
    "scalecua_osworld_train_chrome_48c46dc7_fe04_4505_ade7_723cba1aa6f6_task_verify_1",
    # Official generated Chrome flag row asks to disable historical Chrome
    # refresh flags. Current Chrome does not expose those flags, and the
    # generated metric treats missing flags as already disabled, so no-op can
    # score 1.0.
    "scalecua_osworld_train_chrome_480bcfea_d68f_4aaa_a0a9_2589ef319381_task_verify_21",
    # Official generated Chrome flag row expects removed/renamed Chrome flag
    # ids (`tab-groups`, `tab-groups-collapse`) while the runtime Chrome exposes
    # the current Tab Groups flags under different ids.
    "scalecua_osworld_train_chrome_480bcfea_d68f_4aaa_a0a9_2589ef319381_task_verify_2",
    "scalecua_osworld_train_chrome_480bcfea_d68f_4aaa_a0a9_2589ef319381_task_verify_15",
    "scalecua_osworld_train_chrome_480bcfea_d68f_4aaa_a0a9_2589ef319381_task_verify_39",
    "scalecua_osworld_train_chrome_fc6d8143_9452_4171_9459_7f515143419a_task_verify_8",
    "scalecua_osworld_train_chrome_fc6d8143_9452_4171_9459_7f515143419a_task_verify_25",
    # Official generated Chrome language rows ask for a preferred-language
    # change, but the getter checks Chrome's app locale in Local State. The UI
    # can be visibly correct while app_locale remains unchanged.
    "scalecua_osworld_train_chrome_2ae9ba84_3a0d_4d4c_8338_3a1478dc5fe3_task_verify_8",
    "scalecua_osworld_train_chrome_2ae9ba84_3a0d_4d4c_8338_3a1478dc5fe3_task_verify_9",
    "scalecua_osworld_train_chrome_2ae9ba84_3a0d_4d4c_8338_3a1478dc5fe3_task_verify_10",
    "scalecua_osworld_train_chrome_2ae9ba84_3a0d_4d4c_8338_3a1478dc5fe3_task_verify_11",
    "scalecua_osworld_train_chrome_99146c54_4f37_4ab8_9327_5f3291665e1e_task_verify_22",
    "scalecua_osworld_train_chrome_bb5e4c0d_f964_439c_97b6_bdb9747de3f4_task_verify_40",
    "scalecua_osworld_rl_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_traj_verify_4",
    "scalecua_osworld_rl_chrome_3720f614_37fd_4d04_8a6b_76f54f8c222d_traj_verify_4",
    # Official generated Calc row asks for distinct students per level, but
    # expected G2:G5 values are subject-row counts instead.
    "scalecua_osworld_train_libreoffice_calc_01b269ae_2111_4a07_81fd_3fcd711993b0_task_verify_3",
    # Official generated Calc sorted-rep row has the right expected order but
    # impossible synthetic expected row data (three 1000/999/... values per
    # rep) while the source workbook contains six monthly sales columns.
    "scalecua_osworld_train_libreoffice_calc_0a2e43bf_b26c_4631_a966_af9dfa12c9e5_task_verify_28",
    # Official generated Calc average-sales row has expected averages that do
    # not match the source SalesRep.xlsx data for several representatives, so a
    # correct sheet/chart is capped at partial reward.
    "scalecua_osworld_train_libreoffice_calc_0326d92d_d218_48a8_9ca1_981cd6d064c7_task_verify_37",
    # Official generated Calc date-sort row hard-codes the wrong top-row
    # website after sorting by Date Time descending.
    "scalecua_osworld_train_libreoffice_calc_3a7c8185_25c1_4941_bd7b_96e823c9f21f_task_verify_22",
    # Official generated Calc transaction-count row is ambiguous: a correct
    # summary with Retail/Wholesale counts satisfies the summary check, while
    # the generated per-row "Transaction Count" semantics remain inconsistent.
    "scalecua_osworld_train_libreoffice_calc_51719eea_10bc_4246_a428_ac7c433dd4b3_task_verify_57",
    # Official generated Calc empty-row cleanup evaluator compares an
    # impossible absolute row count instead of checking that only the blank
    # separator rows were removed.
    "scalecua_osworld_train_libreoffice_calc_aa3a8974_2e85_438b_b29e_a64df44deb4b_task_verify_64",
    # Official generated OS rows reuse the source eval.sh contract while the
    # generated instruction changes the destination/pattern: the eval still
    # checks the original fails/failed-notebook shape instead of notebooks or
    # failures.
    "scalecua_osworld_train_os_5c1075ca_bb34_46a3_a7a0_029bd7463e79_task_verify_33",
    "scalecua_osworld_train_os_5c1075ca_bb34_46a3_a7a0_029bd7463e79_task_verify_51",
    # Official generated extension row expects "Hello Extension", but the
    # bundled unpacked extension manifest name is "Hello Extensions".
    "scalecua_osworld_train_multi_apps_a74b607e_6bb5_4ea8_8a7c_5d97c7bbcd2a_task_verify_54",
    # Official generated Chrome Safe Browsing row is already satisfied in the
    # baseline profile / weak getter state, so a no-action trajectory returns
    # reward 1.0.
    "scalecua_osworld_train_chrome_99146c54_4f37_4ab8_9327_5f3291665e1e_task_verify_40",
    # Official generated GIMP rows whose evaluator checks hidden persistence or
    # config state that is inconsistent with the visible task state, plus rows
    # with generated-code defects such as missing private helpers or undefined
    # local variables.
    "scalecua_osworld_train_gimp_06ca5602_62ca_47f6_ad4f_da151cde54cc_task_verify_11",
    "scalecua_osworld_train_gimp_06ca5602_62ca_47f6_ad4f_da151cde54cc_task_verify_16",
    "scalecua_osworld_train_gimp_7767eef2_56a3_4cea_8c9f_48c070c7d65b_task_verify_17",
    "scalecua_osworld_train_gimp_7767eef2_56a3_4cea_8c9f_48c070c7d65b_task_verify_19",
    "scalecua_osworld_train_gimp_7767eef2_56a3_4cea_8c9f_48c070c7d65b_task_verify_20",
    "scalecua_osworld_train_gimp_7767eef2_56a3_4cea_8c9f_48c070c7d65b_task_verify_25",
    "scalecua_osworld_train_gimp_7767eef2_56a3_4cea_8c9f_48c070c7d65b_task_verify_35",
    "scalecua_osworld_train_gimp_d52d6308_ec58_42b7_a2c9_de80e4837b2b_task_verify_31",
    "scalecua_osworld_train_gimp_d52d6308_ec58_42b7_a2c9_de80e4837b2b_task_verify_54",
    "scalecua_osworld_train_gimp_d52d6308_ec58_42b7_a2c9_de80e4837b2b_task_verify_55",
    "scalecua_osworld_train_gimp_d52d6308_ec58_42b7_a2c9_de80e4837b2b_task_verify_56",
    "scalecua_osworld_train_gimp_d52d6308_ec58_42b7_a2c9_de80e4837b2b_task_verify_69",
    "scalecua_osworld_train_gimp_d52d6308_ec58_42b7_a2c9_de80e4837b2b_task_verify_75",
    "scalecua_osworld_train_gimp_d52d6308_ec58_42b7_a2c9_de80e4837b2b_task_verify_77",
    "scalecua_osworld_train_gimp_d52d6308_ec58_42b7_a2c9_de80e4837b2b_task_verify_78",
    "scalecua_osworld_train_gimp_d52d6308_ec58_42b7_a2c9_de80e4837b2b_task_verify_47",
    "scalecua_osworld_train_gimp_e8172110_ec08_421b_a6f5_842e6451911f_task_verify_64",
    # Official generated GIMP rows with mismatched evaluator semantics:
    # compare_docx_images is called with one result file and a rule dict, GIMP
    # config parsing cannot handle quoted/nested values, and layer creation rows
    # check the layer-new-name preference instead of the actual document layer.
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
    # Official generated Impress rows whose visible slide instructions do not
    # match generated evaluator targets: speaker notes instead of visible text
    # or 0-based slide indices for "slide 1".
    "scalecua_osworld_train_libreoffice_impress_841b50aa_df53_47bd_a73a_22d3a9f73160_task_verify_16",
    "scalecua_osworld_train_libreoffice_impress_9cf05d24_6bd9_4dae_8967_f67d88f5d38a_task_verify_47",
    "scalecua_osworld_train_libreoffice_impress_9cf05d24_6bd9_4dae_8967_f67d88f5d38a_task_verify_55",
    # Official generated VLC-source row operates on an Impress deck, but expects
    # a 7-slide deck after duplicating slide 2. The fixture deck has 16 slides,
    # so a correct duplicate produces 17 slides and only the text-duplicate
    # half can pass.
    "scalecua_osworld_train_vlc_778efd0a_153f_4842_9214_f05fc176b877_task_verify_26",
    # Official generated docx metrics for these rows are stricter than the
    # task semantics: exact whitespace in split table cells, explicit run-level
    # font size instead of effective style, missing generated helper functions,
    # or requiring a whole underlined phrase to live in one run.
    "scalecua_osworld_train_libreoffice_writer_936321ce_5236_426a_9a20_e0e3c5dc536f_task_verify_3",
    "scalecua_osworld_train_libreoffice_writer_72b810ef_4156_4d09_8f08_a0cf57e7cefe_task_verify_45",
    "scalecua_osworld_train_libreoffice_writer_6a33f9b9_0a56_4844_9c3f_96ec3ffb3ba2_task_verify_0",
    "scalecua_osworld_train_libreoffice_writer_0b17a146_2934_46c7_8727_73ff6b6483e8_task_verify_7",
    # Official generated Writer rows with broken or semantically incomplete
    # checks: contains_page_break returns 0 when expected and actual counts are
    # both zero.
    "scalecua_osworld_train_libreoffice_writer_ecc2413d_8a48_416e_a3a2_d30106ca36cb_task_verify_67",
    # Official generated Chrome New Tab startup rows do not set up the
    # advertised negative state first. The fresh Chrome state already opens the
    # New Tab page, so these rows are initial-state/eval bugs rather than
    # useful rollout tasks.
    "scalecua_osworld_train_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_task_verify_34",
    "scalecua_osworld_train_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_task_verify_45",
    "scalecua_osworld_train_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_task_verify_52",
    "scalecua_osworld_train_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_task_verify_53",
    # Official generated Chrome rows whose evaluator asks for the New Tab
    # startup state while the instruction asks to restore previous tabs.
    "scalecua_osworld_train_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_task_verify_39",
    "scalecua_osworld_train_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_task_verify_40",
    "scalecua_osworld_train_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_task_verify_41",
    "scalecua_osworld_train_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_task_verify_42",
    # Official generated Chrome rows whose evaluator URL regex or postconfig
    # active-tab assumptions are inconsistent with the visible correct state.
    "scalecua_osworld_train_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_task_verify_62",
    "scalecua_osworld_train_chrome_12086550_11c0_466b_b367_1d9e75b3910e_task_verify_6",
    "scalecua_osworld_train_chrome_f3b19d1e_2d48_44e9_b4e1_defcae1a0197_task_verify_86",
    "scalecua_osworld_train_chrome_f8cfa149_d1c1_4215_8dac_4a0932bad3c2_task_verify_52",
    "scalecua_osworld_train_libreoffice_calc_a82b78bb_7fde_4cb3_94a4_035baf10bcf0_task_verify_48",
    # TRAIN rows surfaced by the V4 oracle sweep as no-op/trivial precheck passes:
    # the fresh desktop state already satisfies the check (precheck reward 1.0 with
    # `executed_actions: []`), so they cannot serve as strict oracle fixtures. Same
    # upstream initial-state/eval-mismatch class as the RL rows above.
    "scalecua_osworld_train_chrome_215dfd39_f493_4bc3_a027_8a97d72c61bf_task_verify_21",
    "scalecua_osworld_train_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_task_verify_17",
    "scalecua_osworld_train_chrome_030eeff7_b492_4218_b312_701ec99ee0cc_task_verify_21",
    "scalecua_osworld_train_multi_apps_9f3bb592_209d_43bc_bb47_d77d9df56504_task_verify_17",
    # TRAIN rows whose upstream gold `oracle_actions` deterministically under-achieve
    # the generated metric on replay (validated rewards 0.0 / 0.3 / 0.5 / 0.927): the
    # gold trajectory omits sub-steps the multi-check reward requires, so the oracle
    # cannot self-validate. Diverse domains, no shared mechanism (transport-invariant,
    # not a runtime regression) — the upstream gold/eval is the defect.
    "scalecua_osworld_train_libreoffice_impress_73c99fb9_f828_43ce_b87a_01dc07faa224_task_verify_0",
    "scalecua_osworld_train_libreoffice_writer_66399b0d_8fda_4618_95c4_bfc6191617e9_task_verify_0",
    "scalecua_osworld_train_libreoffice_writer_00fa164e_2612_4439_992e_157d019a8436_task_verify_10",
    "scalecua_osworld_train_multi_apps_5bc63fb9_276a_4439_a7c1_9dc76401737f_task_verify_0",
    "scalecua_osworld_train_multi_apps_da922383_bfa4_4cd3_bbad_6bebab3d7742_task_verify_46",
    "scalecua_osworld_train_os_4d117223_a354_47fb_8b45_62ab1390a95f_task_verify_21",
    "scalecua_osworld_train_vs_code_df67aebb_fb3a_44fd_b75b_51b6012df509_task_verify_0",
    # RL rows surfaced by the V4 sweep as genuine upstream eval defects (verified
    # NOT dbus/gsettings-dependent, deterministic on replay — distinct from the
    # /run/user/1000/bus cluster). vlc `59f21cfb` gold `oracle_actions` cap at reward
    # 0.5 across all 3 verify points (check_vlc_paused under-achieved); chrome
    # `f3977615` no-op precheck already scores 1.0 (check_vlc_systray trivial pass).
    "scalecua_osworld_rl_vlc_59f21cfb_0120_4326_b255_a5b827b38967_traj_verify_3",
    "scalecua_osworld_rl_vlc_59f21cfb_0120_4326_b255_a5b827b38967_traj_verify_6",
    "scalecua_osworld_rl_vlc_59f21cfb_0120_4326_b255_a5b827b38967_traj_verify_7",
    "scalecua_osworld_rl_chrome_f3977615_2b45_4ac5_8bba_80c17dbe2a37_traj_verify_1",
}
DMV_LIVE_SITE_DRIFT_FRAGMENTS = (
    "licenses-ids/id/applying",
    "licenses-ids/license/applying/fees",
    "licenses-ids/license/applying/documents",
    "licenses-ids/license/renew",
    "licenses-ids/license/renewing",
    "licenses-ids/learners-permit",
    "licenses-ids/permit",
    "vehicles/registrations",
    "vehicles/registration/renew",
    "vehicles/titles",
    "vehicles/title/transfer",
    "drivers/transcript",
    "new.*resident",
    "licenses-ids/license/(applying/tests|testing)",
    "licenses-ids/(license|drivers-ed)/(testing|tests|knowledge-test)",
    "vehicles/(titles|title-transfer",
)
SUPPORTED_ACTIONS = {
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
    "left_click",
    "right_click",
    "double_click",
    "type_text",
    "key",
    "scroll",
    "mouse_move",
    "screenshot",
    "wait",
    "host_push",
}
UNSUPPORTED_AUTH_ACTIONS = {"googledrive", "login"}
LEGACY_ACTION_PARAMETER_KEYS = {
    "activate_window": ("window_name", "strict", "by_class"),
    "chrome_open_tabs": ("urls_to_open", "url"),
    "command": ("command", "shell", "timeout", "cwd", "env"),
    "download": ("url", "path", "filename", "dest", "dest_path"),
    "execute": ("command", "shell", "timeout", "cwd", "env"),
    "key": ("key", "keys"),
    "launch": ("command", "args", "shell", "cwd", "env"),
    "open": ("path", "file_path", "url"),
    "sleep": ("seconds", "duration"),
    "type_text": ("text", "interval"),
    "wait": ("seconds", "duration"),
}


def catalog_root() -> Path:
    return assets.CATALOG_DIR


def catalog_path(split: str, root: Path | None = None) -> Path:
    if split not in RUNTIME_SPLITS:
        raise ValueError(f"unknown lite.scalecua split: {split!r}")
    return (root or catalog_root()) / f"{split}.jsonl"


def oracle_fixture_path(split: str, oracle_dir: Path | None = None) -> Path:
    if split not in RUNTIME_SPLITS:
        raise ValueError(f"unknown lite.scalecua split: {split!r}")
    return (oracle_dir or assets.ENV_DIR / "data" / "oracle") / f"{split}.jsonl"


def _load_oracle_fixtures(
    split: str,
    *,
    oracle_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    fixture_path = oracle_fixture_path(split, oracle_dir)
    if not fixture_path.is_file():
        return {}

    oracle_by_task: dict[str, dict[str, Any]] = {}
    for line_number, fixture in iter_jsonl(fixture_path):
        fixture_split = fixture.get("split")
        task_id = fixture.get("task_id")
        if fixture_split != split:
            raise RuntimeError(
                f"{fixture_path}:{line_number}: fixture split {fixture_split!r} "
                f"does not match {split!r}"
            )
        if not isinstance(task_id, str) or not task_id:
            raise RuntimeError(f"{fixture_path}:{line_number}: missing task_id")
        if task_id in oracle_by_task:
            raise RuntimeError(f"{fixture_path}:{line_number}: duplicate task_id {task_id}")
        oracle_by_task[task_id] = fixture
    return oracle_by_task


def _apply_oracle_fixtures(
    split: str,
    rows: list[dict[str, Any]],
    *,
    oracle_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Embed curated oracle fixtures into final generated task catalogs."""

    oracle_by_task = _load_oracle_fixtures(split, oracle_dir=oracle_dir)
    if not oracle_by_task:
        return rows

    merged_rows: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    for row in rows:
        task_id = row["task_id"]
        seen_task_ids.add(task_id)
        merged = copy.deepcopy(row)
        metadata = merged.setdefault("metadata", {})
        metadata.pop("oracle_actions", None)
        metadata.pop("oracle_after_postconfig", None)
        others = metadata.setdefault("others", {})
        others.setdefault("oracle_actions", [])
        others.setdefault("oracle_after_postconfig", False)
        fixture = oracle_by_task.get(task_id)
        if fixture is not None:
            if others.get("exclude_reason"):
                # Excluded rows never run; injecting oracle actions into them
                # silently hides a dead fixture. Fail so the fixture set and
                # the exclusion lists cannot drift apart.
                raise RuntimeError(
                    f"oracle fixture targets excluded {split} task {task_id}: "
                    f"{others['exclude_reason']}"
                )
            others["oracle_actions"] = copy.deepcopy(fixture.get("oracle_actions") or [])
            if fixture.get("oracle_after_postconfig") is not None:
                others["oracle_after_postconfig"] = fixture["oracle_after_postconfig"]
        merged_rows.append(merged)

    missing = sorted(set(oracle_by_task) - seen_task_ids)
    if missing:
        preview = ", ".join(missing[:5])
        source = oracle_fixture_path(split, oracle_dir)
        raise RuntimeError(
            f"{source}: {len(missing)} fixture task(s) are absent from "
            f"{split} catalog: {preview}"
        )
    return merged_rows


def registration_catalog_path(
    split: str,
    *,
    root: Path | None = None,
    oracle_dir: Path | None = None,
) -> Path:
    """Return the final generated catalog used for task registration."""

    del oracle_dir
    return catalog_path(split, root)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def iter_jsonl(path: Path):
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    yield line_number, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def validate_catalog(path: Path, *, expected_split: str | None = None) -> int:
    seen: set[str] = set()
    count = 0
    for line_number, row in iter_jsonl(path):
        try:
            task_id = row["task_id"]
            instruction = row["instruction"]
            metadata = row["metadata"]
        except KeyError as exc:
            raise RuntimeError(f"{path}:{line_number}: missing key {exc}") from exc
        if not isinstance(task_id, str) or not task_id:
            raise RuntimeError(f"{path}:{line_number}: task_id must be non-empty")
        if task_id in seen:
            raise RuntimeError(f"{path}:{line_number}: duplicate task_id {task_id}")
        if not isinstance(instruction, str) or not instruction:
            raise RuntimeError(f"{path}:{line_number}: instruction must be non-empty")
        if not isinstance(metadata, dict):
            raise RuntimeError(f"{path}:{line_number}: metadata must be an object")
        others = metadata.get("others")
        if not isinstance(others, dict):
            raise RuntimeError(f"{path}:{line_number}: metadata.others must be an object")
        if expected_split and others.get("source_split") != RUNTIME_TO_SOURCE[expected_split]:
            raise RuntimeError(
                f"{path}:{line_number}: source_split={others.get('source_split')!r} "
                f"does not match {expected_split}"
            )
        reason = others.get("exclude_reason")
        if reason is not None:
            try:
                exclude_reasons.validate(reason)
            except ValueError as exc:
                raise RuntimeError(
                    f"{path}:{line_number}: invalid exclude_reason {reason!r}: {exc}"
                ) from exc
            if not isinstance(reason, str) or not reason:
                raise RuntimeError(
                    f"{path}:{line_number}: exclude_reason must be omitted or non-empty string"
                )
        if reason is None:
            text = json.dumps(row, ensure_ascii=False)
            if FILE_CACHE_MAIN in text:
                raise RuntimeError(f"{path}:{line_number}: runnable row contains resolve/main URL")
        bad_key_paths = _chord_string_key_list_paths(row)
        if bad_key_paths:
            raise RuntimeError(
                f"{path}:{line_number}: key list contains chord string; "
                f"use a raw key string or tokenized list: {bad_key_paths[0]}"
            )
        seen.add(task_id)
        count += 1
    if count == 0:
        raise RuntimeError(f"{path}: catalog is empty")
    return count


def validate_all(root: Path | None = None, *, strict_counts: bool = False) -> dict[str, Any]:
    root = root or catalog_root()
    report: dict[str, Any] = {"splits": {}}
    all_ids: set[str] = set()
    for split in RUNTIME_SPLITS:
        path = catalog_path(split, root)
        if not path.is_file():
            raise RuntimeError(f"missing catalog: {path}")
        count = validate_catalog(path, expected_split=split)
        domain_counts = Counter()
        source_domain_counts = Counter()
        excluded = Counter()
        for _, row in iter_jsonl(path):
            others = row["metadata"]["others"]
            domain_counts[str(others.get("domain", ""))] += 1
            source_domain_counts[str(others.get("source_domain", ""))] += 1
            reason = others.get("exclude_reason")
            if reason:
                excluded[reason] += 1
            task_id = row["task_id"]
            if task_id in all_ids:
                raise RuntimeError(f"duplicate task_id across splits: {task_id}")
            all_ids.add(task_id)
        if strict_counts:
            if count != EXPECTED_COUNTS[split]:
                raise RuntimeError(f"{split}: count {count} != {EXPECTED_COUNTS[split]}")
            if dict(domain_counts) != EXPECTED_DOMAIN_COUNTS[split]:
                raise RuntimeError(
                    f"{split}: domain counts {dict(domain_counts)} != "
                    f"{EXPECTED_DOMAIN_COUNTS[split]}"
                )
            if dict(source_domain_counts) != EXPECTED_SOURCE_DOMAIN_COUNTS[split]:
                raise RuntimeError(
                    f"{split}: source domain counts {dict(source_domain_counts)} != "
                    f"{EXPECTED_SOURCE_DOMAIN_COUNTS[split]}"
                )
        report["splits"][split] = {
            "rows": count,
            "domain_counts": dict(sorted(domain_counts.items())),
            "source_domain_counts": dict(sorted(source_domain_counts.items())),
            "excluded_count_by_reason": dict(sorted(excluded.items())),
            "runnable": count - sum(excluded.values()),
        }
    return report


def oracle_fixture_identity(oracle_dir: Path | None = None) -> str:
    """Stable identity for checked-in oracle fixture sources."""

    root = oracle_dir or assets.ENV_DIR / "data" / "oracle"
    digest = hashlib.sha256()
    count = 0
    if root.is_dir():
        for path in sorted(root.glob("*.jsonl")):
            digest.update(path.name.encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
            count += 1
    return json.dumps(
        {
            "path": "lite/gym/envs/lite/scalecua/data/oracle",
            "files": count,
            "sha256": digest.hexdigest(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def osworld_eval_domain_map_identity() -> dict[str, Any]:
    """Identity of the ``osworld_id -> domain`` map this env reads out of the
    lite.osworld eval catalog.

    Pins the PROJECTION, not the file's bytes: the bytes are lite.osworld's own
    lock to own, and pinning them here made an edit that left this mapping
    untouched read as a stale catalog — which registers zero tasks.
    """

    mapping = _load_osworld_eval_domains()
    payload = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
    return {
        "source": "lite/gym/envs/lite/osworld/data/eval.jsonl",
        "entries": len(mapping),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def catalog_lock_data(root: Path | None = None) -> dict[str, Any]:
    root = root or catalog_root()
    splits: dict[str, dict[str, Any]] = {}
    for split in RUNTIME_SPLITS:
        path = catalog_path(split, root)
        data = path.read_bytes()
        splits[split] = {
            "path": f"{split}.jsonl",
            "rows": sum(1 for line in data.splitlines() if line.strip()),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    return {
        "version": 1,
        "generated": True,
        "sources": {
            "generator": "scripts/utils/tasks.sh",
            "asset_identity": assets.asset_identity(),
            "oracle_fixture_identity": oracle_fixture_identity(),
            "osworld_eval_domain_map_identity": osworld_eval_domain_map_identity(),
        },
        "splits": splits,
    }


def catalog_lock_path() -> Path:
    return assets.CATALOG_DIR / "catalog.lock.json"


def write_catalog_lock() -> None:
    path = catalog_lock_path()
    path.write_text(
        json.dumps(catalog_lock_data(), indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def validate_catalog_lock() -> None:
    path = catalog_lock_path()
    if not path.is_file():
        raise RuntimeError(f"missing catalog lock: {path}")
    expected = catalog_lock_data()
    actual = json.loads(path.read_text(encoding="utf-8"))
    if actual != expected:
        raise RuntimeError(
            "lite.scalecua catalog.lock.json is stale; "
            "run lite/gym/envs/lite/scalecua/scripts/utils/tasks.sh generate"
        )


def validate_runtime_cache() -> None:
    """Validate non-catalog runtime material produced by ``tasks.sh generate``."""

    root = assets.CACHE_DIR
    expected_identity = assets.asset_identity()
    if not root.is_dir():
        raise RuntimeError(f"missing ScaleCUA runtime cache: {root}")
    for marker in (".complete", ".asset_identity", "import_report.json"):
        path = root / marker
        if not path.exists():
            raise RuntimeError(f"missing ScaleCUA runtime cache marker: {path}")
    actual_identity = (root / ".asset_identity").read_text(encoding="utf-8").strip()
    if actual_identity != expected_identity:
        raise RuntimeError(
            "lite.scalecua runtime cache is stale; "
            "run lite/gym/envs/lite/scalecua/scripts/utils/tasks.sh generate"
        )

    snapshot = root / "hf_snapshot"
    for marker in (".complete", ".asset_identity"):
        path = snapshot / marker
        if not path.exists():
            raise RuntimeError(f"missing ScaleCUA HF snapshot marker: {path}")
    snapshot_identity = (snapshot / ".asset_identity").read_text(encoding="utf-8").strip()
    if snapshot_identity != expected_identity:
        raise RuntimeError(
            "lite.scalecua HF snapshot cache is stale; "
            "run lite/gym/envs/lite/scalecua/scripts/utils/tasks.sh generate"
        )

    for split in ("train", "rl"):
        for module in ("getters.py", "metrics.py"):
            path = root / "judge_functions" / split / module
            if not path.is_file():
                raise RuntimeError(f"missing ScaleCUA judge overlay: {path}")


def import_all(*, force_download: bool = False) -> dict[str, Any]:
    live_cache = assets.CACHE_DIR
    staging = assets.STAGING_DIR
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        live_cache.mkdir(parents=True, exist_ok=True)
        snapshot = assets.ensure_hf_snapshot(
            live_cache / "hf_snapshot", force_download=force_download
        )
        _materialize_judges(snapshot, staging)
        judge_root = staging / "judge_functions"
        # Harden the pulled read-helpers in place before
        # catalog validation, so the fix is re-applied on every .cache pull and
        # is never a manual .cache edit.
        import_report_judge_patch = _patch_judge_functions(judge_root)
        broken_metrics = _metrics_calling_undefined_helpers(judge_root)
        rows_by_split: dict[str, list[dict[str, Any]]] = {
            split: [] for split in RUNTIME_SPLITS
        }
        import_report: dict[str, Any] = {
            "asset_snapshot": assets.asset_snapshot(),
            "asset_identity": assets.asset_identity(),
            "splits": {},
            "action_type_counts_before": {},
            "action_type_counts_after": {},
            "excluded_count_by_reason": {},
            "url_rewrite_count": 0,
        }
        context = _ImportContext(snapshot=snapshot, broken_metrics=broken_metrics)
        for source_name, runtime_split in (
            ("generated_tasks", "train"),
            ("rl_tasks", "rl"),
        ):
            rows_by_split[runtime_split] = _import_hf_split(
                snapshot=snapshot,
                source_name=source_name,
                runtime_split=runtime_split,
                context=context,
            )

        for split, rows in rows_by_split.items():
            rows = _apply_oracle_fixtures(split, rows)
            rows_by_split[split] = rows
            write_jsonl_atomic(catalog_path(split, staging), rows)
            import_report["splits"][split] = _summarize_rows(rows)

        import_report["action_type_counts_before"] = {
            k: dict(sorted(v.items())) for k, v in sorted(context.action_before.items())
        }
        import_report["action_type_counts_after"] = {
            k: dict(sorted(v.items())) for k, v in sorted(context.action_after.items())
        }
        import_report["excluded_count_by_reason"] = dict(
            sorted(context.excluded_count_by_reason.items())
        )
        import_report["url_rewrite_count"] = context.url_rewrite_count
        import_report["judge_helper_patch"] = {
            "patched": len(import_report_judge_patch["patched"]),
            "already_patched": len(import_report_judge_patch["already_patched"]),
        }
        import_report["broken_metrics"] = sorted(context.broken_metrics)
        import_report["normalization_notes"] = dict(sorted(context.normalization_notes.items()))
        (staging / "import_report.json").write_text(
            json.dumps(import_report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        )
        (staging / ".asset_identity").write_text(assets.asset_identity() + "\n")
        validate_all(staging, strict_counts=True)
        target_judges = live_cache / "judge_functions"
        if target_judges.exists():
            shutil.rmtree(target_judges)
        os.replace(staging / "judge_functions", target_judges)
        os.replace(staging / "import_report.json", live_cache / "import_report.json")
        os.replace(staging / ".asset_identity", live_cache / ".asset_identity")
        (live_cache / ".complete").touch()
        for split in RUNTIME_SPLITS:
            (live_cache / f"{split}.jsonl").unlink(missing_ok=True)
            (live_cache / f"osworld_{split}.jsonl").unlink(missing_ok=True)
        for retired_split in ("eval", "eval_full"):
            (live_cache / f"{retired_split}.jsonl").unlink(missing_ok=True)
            (live_cache / f"osworld_{retired_split}.jsonl").unlink(missing_ok=True)
            (assets.CATALOG_DIR / f"{retired_split}.jsonl").unlink(missing_ok=True)
        for split, rows in rows_by_split.items():
            write_jsonl_atomic(catalog_path(split), rows)
        # VALIDATE, never write. Writing the lock here made the very next
        # `validate_catalog_lock()` vacuous (it checked the file it had just
        # produced), so `install.sh` silently rewrote the TRACKED lock to match
        # whatever it had generated. That is how 23611d76 shipped a lock whose
        # `splits.rl.sha256` predated its own oracle-fixture edit: on a host
        # whose rl.jsonl was also stale, the pair matched and the lock CERTIFIED
        # the stale catalog as fresh — 4 os-domain rl rows then ran without the
        # sudo-priming fix that commit existed to deliver, with no warning.
        # Refreshing the lock is a deliberate act: `tasks.sh refresh-lock`
        # (matching lite.osworld, whose `generate` likewise never writes it).
        validate_catalog_lock()
        validate_runtime_cache()
        return import_report
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


class _ImportContext:
    def __init__(self, *, snapshot: Path, broken_metrics: frozenset[str]):
        self.snapshot = snapshot
        self.broken_metrics = broken_metrics
        self.action_before: dict[str, Counter] = defaultdict(Counter)
        self.action_after: dict[str, Counter] = defaultdict(Counter)
        self.excluded_count_by_reason: Counter = Counter()
        self.normalization_notes: Counter = Counter()
        self.url_rewrite_count = 0


def _materialize_judges(snapshot: Path, staging: Path) -> None:
    dst_root = staging / "judge_functions"
    mapping = {"generated_tasks": "train", "rl_tasks": "rl"}
    for source, runtime in mapping.items():
        src = snapshot / "osworld" / "judge_functions" / source
        dst = dst_root / runtime
        if not src.is_dir():
            raise RuntimeError(f"missing ScaleCUA judge overlay: {src}")
        shutil.copytree(src, dst)


def _load_osworld_eval_domains() -> dict[str, str]:
    out: dict[str, str] = {}
    for _, row in iter_jsonl(assets.OSWORLD_EVAL_JSONL):
        md = row.get("metadata") or {}
        osworld_id = md.get("osworld_id")
        domain = (md.get("others") or {}).get("domain")
        if isinstance(osworld_id, str) and isinstance(domain, str) and domain:
            out[osworld_id] = domain
    return out


def _import_hf_split(
    *,
    snapshot: Path,
    source_name: str,
    runtime_split: str,
    context: _ImportContext,
) -> list[dict[str, Any]]:
    root = snapshot / "osworld" / source_name
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        raise RuntimeError(f"missing ScaleCUA source split: {root}")
    canonical_domains = _load_osworld_eval_domains()
    for path in sorted(root.glob("*/*.json")):
        source_domain = path.parent.name
        payload = _read_json(path)
        osworld_id = payload.get("id")
        canonical_domain = (
            canonical_domains.get(osworld_id, source_domain)
            if isinstance(osworld_id, str)
            else source_domain
        )
        rows.append(
            _row_from_payload(
                payload,
                runtime_split=runtime_split,
                source_name=source_name,
                source_domain=source_domain,
                canonical_domain=canonical_domain,
                source_path=path,
                inherited_exclusion=None,
                context=context,
            )
        )
    return rows

def _row_from_payload(
    payload: dict[str, Any],
    *,
    runtime_split: str,
    source_name: str,
    source_domain: str,
    source_path: Path,
    inherited_exclusion: str | None,
    context: _ImportContext,
    canonical_domain: str | None = None,
) -> dict[str, Any]:
    canonical_domain = canonical_domain or source_domain
    task_id = _task_id(payload, runtime_split, source_domain, source_path)
    payload = _rewrite_urls(copy.deepcopy(payload), context=context)
    _repair_known_upstream_chord_key_list(
        payload,
        source_path=source_path,
        context=context,
    )
    _repair_known_upstream_expected_text(
        payload,
        source_path=source_path,
        context=context,
    )
    instruction = payload.get("instruction")
    if not isinstance(instruction, str) or not instruction:
        raise RuntimeError(f"{source_path}: missing instruction")
    normalized, unsupported = _normalize_runtime_payload(
        payload,
        runtime_split=runtime_split,
        source_domain=source_domain,
        context=context,
    )
    others: dict[str, Any] = {
        "domain": canonical_domain,
        "source_split": source_name,
        "source_domain": source_domain,
        "oracle_actions": normalized.get("oracle_actions", []),
        "oracle_after_postconfig": bool(normalized.get("oracle_after_postconfig", False)),
    }
    proxy = bool(
        payload.get("proxy")
        or (payload.get("metadata") or {}).get("proxy")
        or _has_proxy_required_task_id(task_id=task_id, runtime_split=runtime_split)
    )
    if proxy:
        others["proxy"] = True
    exclude_reason = _exclude_reason(
        payload,
        task_id=task_id,
        inherited_exclusion=inherited_exclusion,
        unsupported=unsupported,
        runtime_split=runtime_split,
        broken_metrics=context.broken_metrics,
    )
    if exclude_reason:
        exclude_reason = exclude_reasons.validate(exclude_reason)
        others["exclude_reason"] = exclude_reason
        context.excluded_count_by_reason[exclude_reason] += 1
    metadata = {
        "others": others,
        "osworld_id": payload.get("id"),
        "source_split": source_name,
        "source_domain": source_domain,
        "scalecua": {
            "source_path": _stable_source_path(source_path),
            "source_split": source_name,
            "runtime_split": runtime_split,
            "new_id": payload.get("new_id"),
            "snapshot": payload.get("snapshot"),
            "related_apps": payload.get("related_apps", []),
        },
        "config": normalized.get("config", []),
        "evaluator": normalized.get("evaluator", {}),
    }
    return {
        "task_id": task_id,
        "instruction": instruction,
        "max_steps": 30,
        "metadata": metadata,
    }


def _exclude_reason(
    payload: dict[str, Any],
    *,
    task_id: str | None = None,
    inherited_exclusion: str | None,
    unsupported: list[str],
    runtime_split: str,
    broken_metrics: frozenset[str],
) -> str | None:
    del inherited_exclusion
    actions = _action_types(payload)
    if actions & UNSUPPORTED_AUTH_ACTIONS:
        return "google_auth"
    if _has_thunderbird_gmail_auth_gap(task_id=task_id, runtime_split=runtime_split):
        return "google_auth"
    if _has_missing_dependency_java(task_id=task_id, runtime_split=runtime_split):
        return "missing_dependency:java"
    if _has_missing_dependency_imagemagick(task_id=task_id, runtime_split=runtime_split):
        return "missing_dependency:imagemagick"
    flake = _flake_reason(task_id=task_id, runtime_split=runtime_split)
    if flake:
        return flake
    raw_reason = (payload.get("metadata") or {}).get("exclude_reason") or payload.get(
        "exclude_reason"
    )
    if isinstance(raw_reason, str) and raw_reason:
        return exclude_reasons.validate(raw_reason)
    evaluator = payload.get("evaluator") or {}
    if evaluator.get("func") == "infeasible" or (payload.get("metadata") or {}).get(
        "infeasible"
    ):
        return "infeasible"
    if task_id in INFEASIBLE_TASK_IDS:
        return "infeasible"
    if _has_missing_instruction_asset_url(payload, runtime_split=runtime_split):
        return "instruction_setup_mismatch"
    if _has_instruction_setup_mismatch_task_id(
        task_id=task_id,
        runtime_split=runtime_split,
    ):
        return "instruction_setup_mismatch"
    if _has_instruction_eval_mismatch(payload, task_id=task_id, runtime_split=runtime_split):
        return "instruction_eval_mismatch"
    if _has_missing_author_results_reference_asset(payload, runtime_split=runtime_split):
        return "missing_reference_asset"
    if _has_upstream_generated_eval_bug(task_id=task_id, runtime_split=runtime_split):
        return "upstream_generated_eval_bug"
    if _has_uncompilable_python_heredoc(payload, runtime_split=runtime_split):
        return "upstream_generated_eval_bug"
    if _has_metric_with_undefined_helper(
        payload,
        runtime_split=runtime_split,
        broken_metrics=broken_metrics,
    ):
        return "upstream_generated_eval_bug"
    if unsupported:
        first = unsupported[0]
        if first == "unsupported_asset_url" or first.startswith("unsupported_schema:"):
            return first
        return f"unsupported_setup:{first}"
    if (
        payload.get("proxy")
        or (payload.get("metadata") or {}).get("proxy")
        or _has_proxy_required_task_id(task_id=task_id, runtime_split=runtime_split)
        or _has_proxy_required_exclude_only(task_id=task_id, runtime_split=runtime_split)
    ):
        return "proxy_required"
    if _has_upstream_live_site_drift(
        payload,
        task_id=task_id,
        runtime_split=runtime_split,
    ):
        return "upstream_live_site_drift"
    return None


def _has_upstream_generated_eval_bug(
    *,
    task_id: str | None,
    runtime_split: str,
) -> bool:
    if runtime_split not in {"train", "rl"} or not task_id:
        return False
    return task_id in UPSTREAM_GENERATED_EVAL_BUG_TASK_IDS


def _iter_payload_command_strings(payload: Any):
    """Yield every ``parameters.command`` string anywhere in a task payload —
    commands live under top-level action lists AND nested under
    ``metadata.config`` / ``metadata.evaluator.postconfig`` / ``metadata.others``.
    argv-list commands yield each string element."""
    if isinstance(payload, dict):
        params = payload.get("parameters")
        if isinstance(params, dict) and "command" in params:
            command = params["command"]
            if isinstance(command, str):
                yield command
            elif isinstance(command, list):
                for element in command:
                    if isinstance(element, str):
                        yield element
        for value in payload.values():
            yield from _iter_payload_command_strings(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _iter_payload_command_strings(value)


# Matches a shell heredoc body: `<<'PY'\n<body>\nPY`. Used to compile-check the
# python source an upstream oracle types into `python3 - <<'PY' … PY`.
_PY_HEREDOC_BODY_RE = re.compile(r"<<\s*'?([A-Za-z_]\w*)'?\s*\n(.*?)\n\1\b", re.S)


def _has_uncompilable_python_heredoc(
    payload: dict[str, Any],
    *,
    runtime_split: str,
) -> bool:
    """True iff a python heredoc in the task's actions has a body that raises
    ``IndentationError`` — the upstream ``generated_tasks`` defect where a gold
    oracle script's ``import`` line was left flush-left while the rest of the body
    kept an indent (``import os\\n        p = …`` → ``unexpected indent`` on line 2).
    The script fails deterministically as authored, independent of exec user, so the
    task is filtered as ``upstream_generated_eval_bug`` rather than counted as
    coverage. Restricted to ``IndentationError`` (not every ``SyntaxError``) so that
    heredocs carrying unsubstituted ``{PLACEHOLDER}`` templates — which are valid at
    runtime — are never mistaken for defects."""
    if runtime_split not in {"train", "rl"}:
        return False
    for command in _iter_payload_command_strings(payload):
        if "python" not in command or "<<" not in command:
            continue
        for match in _PY_HEREDOC_BODY_RE.finditer(command):
            preceding = command[: match.start()].split("\n")[-1]
            if "python" not in preceding:
                continue
            try:
                compile(match.group(2), "<oracle>", "exec")
            except IndentationError:
                return True
            except SyntaxError:
                continue
    return False


def _metrics_calling_undefined_helpers(judge_root: Path) -> frozenset[str]:
    """Generated metric functions that call a helper this overlay never defines.

    The generated shards were split from a larger source and some kept calls to
    module-level helpers that did not come with them; ``judges`` injects 20 such
    names back, but not these. The call raises ``NameError`` at scoring time, and
    ``verify.evaluate_scalecua_task`` degrades any metric exception to 0.0 -- so
    the row scores zero no matter what the agent did, and (before that handler
    started logging) left no trace to tell it apart from a hard task.

    Derived, not an id list: a future overlay refresh that fixes or breaks more
    shards is picked up without anyone re-curating.
    """
    import ast
    import builtins

    from lite.gym.envs.lite.scalecua.src.osworld import judges

    if not judge_root.is_dir():
        raise RuntimeError(f"no ScaleCUA judge overlay under {judge_root}")
    roots = [judge_root / split for split in RUNTIME_SPLITS if (judge_root / split).is_dir()]
    if not roots:
        raise RuntimeError(f"no ScaleCUA judge overlay under {judge_root}")
    paths = sorted(pth for r in roots for pth in r.rglob("verigen_metrics/*.py"))
    if not paths:
        raise RuntimeError(f"no ScaleCUA metric overlay under {judge_root}")
    trees: dict[Path, ast.Module] = {}
    # judges injects a fixed set of helpers back into every shard; those names
    # resolve at runtime even though the shard never defines them.
    defined: set[str] = set(judges._INJECTED_HELPER_NAMES)
    for path in paths:
        try:
            tree = ast.parse(path.read_text(errors="replace"), str(path))
        except SyntaxError:
            continue
        trees[path] = tree
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    defined.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined.add(target.id)
    known = defined | set(dir(builtins))
    broken: set[str] = set()
    for tree in trees.values():
        for fn in tree.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            local = {arg.arg for arg in fn.args.args}
            local |= {
                c.name for c in ast.walk(fn)
                if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            local |= {
                t.id for n in ast.walk(fn) if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name)
            }
            for node in ast.walk(fn):
                # Only CALL sites: a bare Name load can be a comprehension target or
                # a nested-scope binding, and counting those over-reports wildly.
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id not in known
                    and node.func.id not in local
                ):
                    broken.add(fn.name)
    return frozenset(broken)


def _has_metric_with_undefined_helper(
    payload: dict[str, Any],
    *,
    runtime_split: str,
    broken_metrics: frozenset[str],
) -> bool:
    """True iff the row is scored by a metric that cannot run (see above)."""
    if runtime_split not in {"train", "rl"}:
        return False
    if not broken_metrics:
        return False
    blob = json.dumps(payload)
    return any(f'"{name}"' in blob for name in broken_metrics)


def _has_thunderbird_gmail_auth_gap(
    *,
    task_id: str | None,
    runtime_split: str,
) -> bool:
    if runtime_split != "train" or not task_id:
        return False
    return task_id in THUNDERBIRD_GMAIL_AUTH_TASK_IDS


def _has_missing_dependency_imagemagick(
    *,
    task_id: str | None,
    runtime_split: str,
) -> bool:
    if runtime_split != "train" or not task_id:
        return False
    return task_id in MISSING_DEPENDENCY_IMAGEMAGICK_TASK_IDS


def _has_missing_dependency_java(
    *,
    task_id: str | None,
    runtime_split: str,
) -> bool:
    if runtime_split != "train" or not task_id:
        return False
    return task_id in MISSING_DEPENDENCY_JAVA_TASK_IDS


def _has_instruction_setup_mismatch_task_id(
    *,
    task_id: str | None,
    runtime_split: str,
) -> bool:
    if runtime_split not in {"train", "rl"} or not task_id:
        return False
    return task_id in INSTRUCTION_SETUP_MISMATCH_TASK_IDS


def _flake_reason(
    *,
    task_id: str | None,
    runtime_split: str,
) -> str | None:
    if runtime_split not in {"train", "rl"} or not task_id:
        return None
    if task_id in FLAKE_CHROME_GUI_EXTENSION_LOAD_TASK_IDS:
        return "flake:chrome_gui_extension_load"
    if task_id in FLAKE_CHROME_SECURE_PREFS_MAC_TASK_IDS:
        return "flake:chrome_secure_prefs_mac"
    return None


def _has_proxy_required_task_id(
    *,
    task_id: str | None,
    runtime_split: str,
) -> bool:
    if runtime_split not in {"train", "rl"} or not task_id:
        return False
    return task_id in PROXY_REQUIRED_TASK_IDS


def _has_proxy_required_exclude_only(
    *,
    task_id: str | None,
    runtime_split: str,
) -> bool:
    if runtime_split not in {"train", "rl"} or not task_id:
        return False
    return task_id in PROXY_REQUIRED_EXCLUDE_ONLY_TASK_IDS


def _has_upstream_live_site_drift(
    payload: dict[str, Any],
    *,
    task_id: str | None = None,
    runtime_split: str,
) -> bool:
    if (
        isinstance(task_id, str)
        and runtime_split in {"train", "rl"}
        and (
            task_id in CHROME_WEBSTORE_LIVE_SITE_TASK_IDS
            or task_id in UPSTREAM_LIVE_SITE_DRIFT_TASK_IDS
        )
    ):
        return True
    return _has_babycenter_exact_url_live_site_drift(
        payload,
        runtime_split=runtime_split,
    ) or _has_macys_live_site_drift(
        payload,
        runtime_split=runtime_split,
    ) or _has_doj_forms_component_drift(
        payload,
        runtime_split=runtime_split,
    ) or _has_dmv_live_site_drift(
        payload,
        runtime_split=runtime_split,
    ) or _has_flightaware_category_drift(
        payload,
        runtime_split=runtime_split,
    ) or _has_delta_award_live_site_drift(
        payload,
        runtime_split=runtime_split,
    ) or _has_united_special_needs_drift(
        payload,
        runtime_split=runtime_split,
    ) or _has_shenzhen_address_lookup_drift(
        payload,
        runtime_split=runtime_split,
    )


def _has_missing_author_results_reference_asset(
    payload: dict[str, Any],
    *,
    runtime_split: str,
) -> bool:
    if runtime_split not in {"train", "rl"}:
        return False
    evaluator = payload.get("evaluator") or {}
    for value in _iter_evaluator_key_values(evaluator, "reference_path"):
        if isinstance(value, str) and value.startswith(AUTHOR_RESULTS_PREFIX):
            return True
    return False


def _iter_evaluator_key_values(obj: Any, key_name: str):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == key_name:
                yield value
            yield from _iter_evaluator_key_values(value, key_name)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_evaluator_key_values(item, key_name)


def _has_babycenter_exact_url_live_site_drift(
    payload: dict[str, Any],
    *,
    runtime_split: str,
) -> bool:
    if runtime_split != "train":
        return False
    evaluator = payload.get("evaluator") or {}
    if evaluator.get("func") != "is_expected_active_tab":
        return False
    result = evaluator.get("result") or {}
    if not isinstance(result, dict) or result.get("type") != "active_url_from_accessTree":
        return False
    expected = evaluator.get("expected") or {}
    rules = expected.get("rules") if isinstance(expected, dict) else {}
    url = rules.get("url") if isinstance(rules, dict) else None
    if not isinstance(url, str) or not BABYCENTER_EXACT_NAME_URL_RE.match(url.strip()):
        return False
    source = str(payload.get("source") or "").lower()
    source_split = str((payload.get("metadata") or {}).get("source_split") or "").lower()
    source_path = str((payload.get("metadata") or {}).get("source_path") or "").lower()
    return (
        source == "mind2web"
        or source_split == "generated_tasks"
        or "generated_tasks/chrome/59155008-fe71-45ec-8a8f-dc35497b6aa8" in source_path
    )


def _has_macys_live_site_drift(
    payload: dict[str, Any],
    *,
    runtime_split: str,
) -> bool:
    if runtime_split not in {"train", "rl"}:
        return False
    if str(payload.get("id") or "") != MACYS_LIVE_SITE_DRIFT_OSWORLD_ID:
        return False
    evaluator = payload.get("evaluator") or {}
    result = evaluator.get("result") or {}
    results = result if isinstance(result, list) else [result]
    result_types = {
        str(config.get("type") or "")
        for config in results
        if isinstance(config, dict)
    }
    if any(t == "url_path_parse" or t.startswith("macys_url_parse") for t in result_types):
        return True
    if "active_url_from_accessTree" not in result_types:
        return False
    funcs = evaluator.get("func")
    func_names = funcs if isinstance(funcs, list) else [funcs]
    func_text = " ".join(str(name or "") for name in func_names).lower()
    expected_text = json.dumps(evaluator.get("expected") or {}, ensure_ascii=False).lower()
    if "check_url_filters" in func_text:
        return True
    return (
        "macys" in expected_text
        and (
            "is_expected_url_pattern_match" in func_text
        )
    )


def _has_doj_forms_component_drift(
    payload: dict[str, Any],
    *,
    runtime_split: str,
) -> bool:
    if runtime_split not in {"train", "rl"}:
        return False
    if str(payload.get("id") or "") != DOJ_FORMS_COMPONENT_DRIFT_OSWORLD_ID:
        return False
    evaluator = payload.get("evaluator") or {}
    expected_text = " ".join(_expected_rule_texts(evaluator)).lower()
    return "field_component_target_id=" in expected_text


def _has_dmv_live_site_drift(
    payload: dict[str, Any],
    *,
    runtime_split: str,
) -> bool:
    if runtime_split not in {"train", "rl"}:
        return False
    if str(payload.get("id") or "") != DMV_LIVE_SITE_DRIFT_OSWORLD_ID:
        return False
    expected_text = " ".join(
        _normalize_expected_text(text)
        for text in _expected_rule_texts(payload.get("evaluator") or {})
    )
    return any(fragment in expected_text for fragment in DMV_LIVE_SITE_DRIFT_FRAGMENTS)


def _has_flightaware_category_drift(
    payload: dict[str, Any],
    *,
    runtime_split: str,
) -> bool:
    if runtime_split not in {"train", "rl"}:
        return False
    if str(payload.get("id") or "") != FLIGHTAWARE_CATEGORY_DRIFT_OSWORLD_ID:
        return False
    expected_text = " ".join(
        _normalize_expected_text(text)
        for text in _expected_rule_texts(payload.get("evaluator") or {})
    )
    return "discussions.flightaware.com/c/" in expected_text


def _has_delta_award_live_site_drift(
    payload: dict[str, Any],
    *,
    runtime_split: str,
) -> bool:
    if runtime_split not in {"train", "rl"}:
        return False
    if str(payload.get("id") or "") != DELTA_AWARD_LIVE_SITE_DRIFT_OSWORLD_ID:
        return False
    evaluator = payload.get("evaluator") or {}
    result = evaluator.get("result") or {}
    result_type = str(result.get("type") or "")
    if runtime_split == "train":
        result_text = _normalize_expected_text(json.dumps(result, ensure_ascii=False))
        return (
            result_type == "active_tab_html_parse"
            and "mach-flight-context-info" in result_text
        )
    return result_type.startswith("delta_")


def _has_united_special_needs_drift(
    payload: dict[str, Any],
    *,
    runtime_split: str,
) -> bool:
    if runtime_split not in {"train", "rl"}:
        return False
    if str(payload.get("id") or "") != UNITED_SPECIAL_NEEDS_DRIFT_OSWORLD_ID:
        return False
    expected_text = " ".join(
        _normalize_expected_text(text)
        for text in _expected_rule_texts(payload.get("evaluator") or {})
    )
    return "united.com/en/us/fly/travel/special-needs" in expected_text


def _has_shenzhen_address_lookup_drift(
    payload: dict[str, Any],
    *,
    runtime_split: str,
) -> bool:
    if runtime_split != "train":
        return False
    if str(payload.get("id") or "") != SHENZHEN_ADDRESS_LOOKUP_DRIFT_OSWORLD_ID:
        return False
    evaluator = payload.get("evaluator") or {}
    result = evaluator.get("result") or {}
    if not isinstance(result, dict):
        return False
    if result.get("type") != "vm_file":
        return False
    return result.get("path") == "/home/user/Desktop/AllLocations.docx"


def _expected_rule_texts(evaluator: dict[str, Any]) -> list[str]:
    expected = evaluator.get("expected") if isinstance(evaluator, dict) else None
    configs = expected if isinstance(expected, list) else [expected]
    out: list[str] = []
    for config in configs:
        if not isinstance(config, dict):
            continue
        rules = config.get("rules")
        if not isinstance(rules, dict):
            continue
        for key in ("url", "target_url"):
            value = rules.get(key)
            if isinstance(value, str):
                out.append(value)
        for key in ("urls", "expected"):
            value = rules.get(key)
            if isinstance(value, str):
                out.append(value)
            elif isinstance(value, list):
                out.extend(item for item in value if isinstance(item, str))
    return out


def _normalize_expected_text(text: str) -> str:
    return (
        text.lower()
        .replace("\\.", ".")
        .replace("\\/", "/")
        .replace("\\?", "?")
    )


def _has_missing_instruction_asset_url(
    payload: dict[str, Any],
    *,
    runtime_split: str,
) -> bool:
    if runtime_split not in {"train", "rl"}:
        return False
    instruction = payload.get("instruction")
    if not isinstance(instruction, str):
        return False
    if URL_RE.search(instruction):
        return False
    if not MISSING_ASSET_URL_REFERENCE_RE.search(instruction):
        return False
    # Official ScaleCUA evaluation passes only the instruction to the agent and
    # executes setup solely from config. If neither contains the referenced URL,
    # the task is underspecified rather than a migration/setup failure.
    return not _contains_url(payload.get("config")) and not _contains_url(
        payload.get("setup")
    )


def _contains_url(value: Any) -> bool:
    if isinstance(value, str):
        return bool(URL_RE.search(value))
    if isinstance(value, list):
        return any(_contains_url(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_url(item) for item in value.values())
    return False


def _has_instruction_eval_mismatch(
    payload: dict[str, Any],
    *,
    task_id: str | None = None,
    runtime_split: str,
) -> bool:
    if (
        runtime_split in {"train", "rl"}
        and task_id
        and task_id in INSTRUCTION_EVAL_MISMATCH_TASK_IDS
    ):
        return True
    instruction = payload.get("instruction")
    if not isinstance(instruction, str):
        return False
    normalized = re.sub(r"\s+", " ", instruction.strip().lower())
    if not normalized.startswith(
        (
            "tell me how",
            "tell me the steps",
            "explain how",
            "explain the steps",
            "guide me",
            "show me how",
            "instruct me",
        )
    ):
        return False
    evaluator = payload.get("evaluator") or {}
    result = evaluator.get("result", {})
    results = result if isinstance(result, list) else [result]
    return any(
        isinstance(config, dict)
        and config.get("type") == "vm_command_line"
        and GIMP_ACTION_HISTORY in str(config.get("command") or "")
        for config in results
    )


def _normalize_runtime_payload(
    payload: dict[str, Any],
    *,
    runtime_split: str,
    source_domain: str,
    context: _ImportContext,
) -> tuple[dict[str, Any], list[str]]:
    unsupported: list[str] = []

    def normalize_action_list(actions: Any, phase: str) -> list[dict[str, Any]]:
        if actions is None:
            return []
        if not isinstance(actions, list):
            unsupported.append("unsupported_schema:action_list")
            return []
        out: list[dict[str, Any]] = []
        for action in actions:
            if not isinstance(action, dict):
                unsupported.append("unsupported_schema:action_object")
                continue
            if _is_stale_rl_a462_charles_setup_action(
                payload,
                action,
                phase=phase,
                runtime_split=runtime_split,
            ):
                context.normalization_notes["drop_stale_rl_a462_charles_setup"] += 1
                continue
            placeholder_action = _repair_external_placeholder_download(action, phase=phase)
            if placeholder_action is not None:
                action = placeholder_action
                context.normalization_notes["local_placeholder_image_download"] += 1
            repaired_action = _repair_root_home_test1_setup_action(
                payload,
                action,
                phase=phase,
            )
            if repaired_action is not None:
                action = repaired_action
                context.normalization_notes["repair_root_home_test1_setup"] += 1
            if "type" not in action:
                if phase == "evaluator.postconfig":
                    unsupported.append("unsupported_schema:evaluator_postconfig_query_config")
                else:
                    unsupported.append("unsupported_schema:action_type_missing")
                continue
            normalized = _normalize_action(
                action,
                phase=phase,
                runtime_split=runtime_split,
                context=context,
            )
            if normalized is None:
                unsupported.append(str(action.get("type")))
                continue
            if normalized == "unsupported_asset_url":
                unsupported.append("unsupported_asset_url")
                continue
            out.extend(normalized)
        return out

    evaluator = copy.deepcopy(payload.get("evaluator") or {})
    evaluator_postconfig = normalize_action_list(
        evaluator.get("postconfig", []), "evaluator.postconfig"
    )
    top_level_postconfig = normalize_action_list(
        payload.get("postconfig", []), "postconfig"
    )
    evaluator["postconfig"] = _normalize_osworld_postconfig(
        evaluator_postconfig + top_level_postconfig,
        source_domain,
    )
    evaluator["postconfig"] = _normalize_scalecua_gimp_export_postconfig(
        evaluator,
        source_domain=source_domain,
        context=context,
    )
    evaluator = _normalize_vscode_theme_expected_aliases(evaluator, context=context)
    return {
        "config": normalize_action_list(payload.get("config", []), "config"),
        "evaluator": evaluator,
        "oracle_actions": normalize_action_list(
            payload.get("oracle_actions", []), "oracle_actions"
        ),
        "oracle_after_postconfig": bool(payload.get("oracle_after_postconfig", False)),
    }, unsupported


def _normalize_scalecua_gimp_export_postconfig(
    evaluator: dict[str, Any],
    *,
    source_domain: str,
    context: _ImportContext,
) -> list[dict[str, Any]]:
    postconfig = evaluator.get("postconfig")
    if source_domain != "gimp" or not isinstance(postconfig, list):
        return postconfig if isinstance(postconfig, list) else []
    if not _looks_like_gimp_export_postconfig(postconfig):
        return postconfig
    out_path = _first_gimp_export_result_path(
        evaluator.get("result")
    ) or _first_gimp_export_expected_path(evaluator.get("expected"))
    if not out_path:
        return postconfig
    context.normalization_notes["gimp_export_full_path_postconfig"] += 1
    return gimp_export_as_postconfig(out_path)


def _looks_like_gimp_export_postconfig(postconfig: list[dict[str, Any]]) -> bool:
    text = json.dumps(postconfig, sort_keys=True).lower()
    return (
        "shift" in text
        and "ctrl" in text
        and "e" in text
        and "pyautogui.write" in text
    )


def _first_gimp_export_result_path(result: Any) -> str | None:
    for item in result if isinstance(result, list) else [result]:
        if not isinstance(item, dict):
            continue
        path = item.get("path") or item.get("dest")
        normalized = _normalize_gimp_export_path(path)
        if normalized:
            return normalized
    return None


def _first_gimp_export_expected_path(expected: Any) -> str | None:
    for item in expected if isinstance(expected, list) else [expected]:
        if not isinstance(item, dict):
            continue
        rules = item.get("rules")
        if not isinstance(rules, dict):
            continue
        for key in ("tgt_path", "target_path", "output_path", "path", "dest"):
            normalized = _normalize_gimp_export_path(rules.get(key))
            if normalized:
                return normalized
    return None


def _normalize_gimp_export_path(path: Any) -> str | None:
    if not isinstance(path, str) or not path:
        return None
    if not re.search(r"\.(png|jpe?g|gif|bmp|tiff?|webp)$", path, re.IGNORECASE):
        return None
    if path.startswith("/"):
        return path
    return f"/home/user/Desktop/{path}"


VSCODE_THEME_VALUE_ALIASES = {
    "Light+ (default light)": "Default Light+",
    "Dark+ (default dark)": "Default Dark+",
}


def _normalize_vscode_theme_expected_aliases(
    evaluator: dict[str, Any],
    *,
    context: _ImportContext,
) -> dict[str, Any]:
    replacements = 0

    def visit(value: Any) -> None:
        nonlocal replacements
        if isinstance(value, dict):
            for key, item in list(value.items()):
                if (
                    key == "workbench.colorTheme"
                    and isinstance(item, str)
                    and item in VSCODE_THEME_VALUE_ALIASES
                ):
                    value[key] = VSCODE_THEME_VALUE_ALIASES[item]
                    replacements += 1
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(evaluator.get("expected"))
    if replacements:
        context.normalization_notes["vscode_theme_expected_alias"] += replacements
    return evaluator


def _is_stale_rl_a462_charles_setup_action(
    payload: dict[str, Any],
    action: dict[str, Any],
    *,
    phase: str,
    runtime_split: str,
) -> bool:
    if phase != "config" or runtime_split != "rl":
        return False
    if payload.get("id") != A462_USER_SWITCH_OSWORLD_ID:
        return False
    params = action.get("parameters") or {}
    command = params.get("command")
    if not isinstance(command, str):
        return False
    normalized = re.sub(r"\s+", " ", command.lower())
    return "su - charles" in normalized


def _repair_external_placeholder_download(
    action: dict[str, Any],
    *,
    phase: str,
) -> dict[str, Any] | None:
    if phase != "config" or action.get("type") != "download":
        return None
    params = action.get("parameters") or {}
    files = params.get("files")
    if not isinstance(files, list) or not files:
        return None
    paths: list[str] = []
    for item in files:
        if not isinstance(item, dict) or item.get("url") != PLACEHOLDER_1024_URL:
            return None
        path = item.get("path")
        if not isinstance(path, str) or not path:
            return None
        paths.append(path if path.startswith("/") else f"/home/user/{path}")
    return {
        "type": "execute",
        "parameters": {
            "command": _placeholder_png_command(paths),
            "shell": True,
        },
    }


def _placeholder_png_command(paths: list[str]) -> str:
    code = (
        "import json, os, struct, zlib\n"
        f"paths = {json.dumps(paths)}\n"
        "w, h = 1024, 768\n"
        "def chunk(tag, data):\n"
        "    return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)\n"
        "rows = []\n"
        "for y in range(h):\n"
        "    row = bytearray([0])\n"
        "    for x in range(w):\n"
        "        row.extend((x * 255 // (w - 1), y * 255 // (h - 1), 160))\n"
        "    rows.append(bytes(row))\n"
        "png = b'\\x89PNG\\r\\n\\x1a\\n'\n"
        "png += chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))\n"
        "png += chunk(b'IDAT', zlib.compress(b''.join(rows), 6))\n"
        "png += chunk(b'IEND', b'')\n"
        "for path in paths:\n"
        "    os.makedirs(os.path.dirname(path), exist_ok=True)\n"
        "    with open(path, 'wb') as f:\n"
        "        f.write(png)\n"
    )
    return _python_heredoc(code)


def _repair_root_home_test1_setup_action(
    payload: dict[str, Any],
    action: dict[str, Any],
    *,
    phase: str,
) -> dict[str, Any] | None:
    if phase != "config":
        return None
    if payload.get("id") != ROOT_HOME_TEST1_OSWORLD_ID:
        return None
    if action.get("type") not in {"execute", "command"}:
        return None
    params = action.get("parameters") or {}
    command = params.get("command")
    if not isinstance(command, str):
        return None
    normalized = re.sub(r"\s+", " ", command.strip())
    if normalized not in {"mkdir /home/test1", "mkdir -p /home/test1"}:
        return None
    repaired = copy.deepcopy(action)
    repaired_params = copy.deepcopy(params)
    repaired_params["command"] = (
        "printf '%s\\n' {CLIENT_PASSWORD} | sudo -S mkdir -p /home/test1"
    )
    repaired_params["shell"] = True
    repaired["parameters"] = repaired_params
    return repaired


def _legacy_action_parameters(action: dict[str, Any]) -> dict[str, Any]:
    raw_parameters = action.get("parameters")
    parameters = copy.deepcopy(raw_parameters) if isinstance(raw_parameters, dict) else {}
    action_type = action.get("type")
    if not isinstance(action_type, str):
        return parameters
    for key in LEGACY_ACTION_PARAMETER_KEYS.get(action_type, ()):
        if key in action and key not in parameters:
            parameters[key] = copy.deepcopy(action[key])
    return parameters


# Matches a heredoc whose delimiter is immediately followed by a LITERAL `\n`/`\t`
# (JSON `\\n`) instead of a real newline — the upstream `generated_tasks` defect
# where a `cat > file << 'PYEOF'\ndef …` setup was emitted with escaped newlines, so
# the delimiter never closes and the file-write fails with a bash heredoc syntax
# error. These commands are pure heredoc file-writes, so restoring real newlines/tabs
# is a safe, exact repair.
_LITERAL_NL_HEREDOC_RE = re.compile(r"<<\s*'?\w+'?\\[nt]")


def _repair_upstream_heredoc_command(
    params: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Repair the literal-``\\n`` heredoc defect in an action's ``command`` (str or
    argv-list). Returns ``(params, repaired)`` — a copy with real newlines/tabs when
    the malformed shape is present, else the original params untouched."""

    def _fix(command: str) -> str:
        if isinstance(command, str) and _LITERAL_NL_HEREDOC_RE.search(command):
            return command.replace("\\n", "\n").replace("\\t", "\t")
        return command

    command = params.get("command")
    if isinstance(command, str):
        fixed = _fix(command)
        if fixed != command:
            return {**params, "command": fixed}, True
    elif isinstance(command, list):
        fixed_list = [_fix(x) if isinstance(x, str) else x for x in command]
        if fixed_list != command:
            return {**params, "command": fixed_list}, True
    return params, False


def _normalize_action(
    action: Any,
    *,
    phase: str,
    runtime_split: str,
    context: _ImportContext,
) -> list[dict[str, Any]] | str | None:
    if not isinstance(action, dict):
        return None
    t = action.get("type")
    if not isinstance(t, str):
        return None
    context.action_before[runtime_split][t] += 1
    p = _legacy_action_parameters(action)
    if p != (action.get("parameters") or {}):
        context.normalization_notes["legacy_top_level_action_parameters"] += 1
    p, repaired_heredoc = _repair_upstream_heredoc_command(p)
    if repaired_heredoc:
        context.normalization_notes["repair_heredoc_literal_newline"] += 1
    out: list[dict[str, Any]]
    if t in SUPPORTED_ACTIONS:
        desktop_database_params = _normalize_update_desktop_database_params(p)
        if desktop_database_params is not None and phase != "oracle_actions":
            p = desktop_database_params
            context.normalization_notes["normalize_update_desktop_database"] += 1
        out = [{"type": t, "parameters": p}]
    elif t == "key_press":
        keys = p.get("keys") or p.get("key")
        if isinstance(keys, str):
            p = {"key": keys}
        elif (
            isinstance(keys, list)
            and len(keys) == 1
            and isinstance(keys[0], str)
            and "+" not in keys[0]
        ):
            p = {"key": keys[0]}
        else:
            p = {"keys": keys}
        out = [{"type": "key", "parameters": p}]
    elif t in {"open_url", "chrome_open_url"}:
        url = p.get("url") or action.get("url")
        out = [{"type": "chrome_open_tabs", "parameters": {"urls_to_open": [url]}}] if url else []
    elif t in {"focus_window", "focus_app", "activate_app"}:
        name = p.get("window_name") or p.get("app_name") or p.get("name")
        out = [{"type": "activate_window", "parameters": {"window_name": name or ""}}]
    elif t in {"kill_all", "close_chrome"}:
        keyword = p.get("keyword") or "chrome"
        out = [
            {
                "type": "execute",
                "parameters": {
                    "command": f"pkill -f {json.dumps(str(keyword))} || true",
                    "shell": True,
                },
            }
        ]
    elif t in {"close_all_applications", "close_all_apps", "close_all_windows"}:
        out = [
            {
                "type": "execute",
                "parameters": {
                    "command": (
                        "pkill -f 'google-chrome|chrome|libreoffice|soffice|"
                        "thunderbird|vlc|gimp|code|evince' || true"
                    ),
                    "shell": True,
                },
            }
        ]
    elif t == "close_all_windows_except":
        keep = "|".join(re.escape(str(x)) for x in p.get("keep_windows", []))
        if not keep:
            return None
        out = [
            {
                "type": "execute",
                "parameters": {
                    "command": (
                        "wmctrl -lx | awk "
                        + json.dumps(f"'$0 !~ /{keep}/ {{print $1}}'")
                        + " | xargs -r -n1 wmctrl -ic || true"
                    ),
                    "shell": True,
                },
            }
        ]
    elif t == "chrome_close_tabs_except":
        keep = p.get("keep_urls", [])
        script = (
            "python3 <<'PY'\n"
            "import json, urllib.request\n"
            f"keep={json.dumps(keep)}\n"
            "try:\n"
            "    tabs=json.load(urllib.request.urlopen('http://localhost:1337/json', timeout=2))\n"
            "except Exception:\n"
            "    tabs=[]\n"
            "for tab in tabs:\n"
            "    url=tab.get('url','')\n"
            "    if tab.get('type')=='page' and not any(k in url for k in keep):\n"
            "        try: urllib.request.urlopen('http://localhost:1337/json/close/'+tab['id'], timeout=2)\n"
            "        except Exception: pass\n"
            "PY"
        )
        out = [{"type": "execute", "parameters": {"command": script, "shell": True}}]
    elif t == "chrome_open_file":
        path = p.get("file_path") or p.get("path")
        url = f"file://{path}" if path else ""
        out = [{"type": "chrome_open_tabs", "parameters": {"urls_to_open": [url]}}] if url else []
    elif t == "copyfile":
        src = p.get("source") or p.get("src")
        dst = p.get("dest") or p.get("dst")
        if not (isinstance(src, str) and isinstance(dst, str) and src.startswith("/") and dst.startswith("/")):
            return None
        out = [
            {
                "type": "execute",
                "parameters": {
                    "command": f"mkdir -p {sh_quote(str(Path(dst).parent))} && cp {sh_quote(src)} {sh_quote(dst)}",
                    "shell": True,
                },
            }
        ]
    elif t in {"python", "execute_python"}:
        code = p.get("code") or p.get("script")
        if not isinstance(code, str):
            return None
        out = [
            {
                "type": "execute",
                "parameters": {"command": _python_heredoc(code), "shell": True},
            }
        ]
    elif t in {"upload_file", "copyfile_from_host_to_guest"}:
        host = p.get("src") or p.get("local_path")
        dst = p.get("dest") or p.get("remote_path")
        resolved = _resolve_host_file(host)
        if not (resolved and dst):
            return "unsupported_asset_url"
        if not str(dst).startswith("/"):
            dst = f"/home/user/{dst}"
        out = [
            {
                "type": "host_push",
                "parameters": {"files": [{"host_path": str(resolved), "dst": str(dst)}]},
            }
        ]
    elif t == "active_setting":
        context.normalization_notes["active_setting_noop"] += 1
        out = [{"type": "wait", "parameters": {"duration": 0.1}}]
    elif t == "get_tabs_info":
        context.normalization_notes["get_tabs_info_noop"] += 1
        out = [{"type": "wait", "parameters": {"duration": 0.1}}]
    elif t == "copyfile_from_guest_to_host":
        # Tier-2/3 revival: redundant postconfig copy — the evaluator's
        # `vm_file multi:true` result getter already pulls main.py (+siblings)
        # guest->host, so the extra copy is a no-op for scoring.
        context.normalization_notes["copyfile_from_guest_to_host_noop"] += 1
        out = [{"type": "wait", "parameters": {"duration": 0.1}}]
    elif t == "close_all_libreoffice":
        # Tier-2/3 revival: scoped equivalent of close_all_applications. Default
        # pkill signal is SIGTERM -> LibreOffice quits gracefully and flushes
        # registrymodifications.xcu before the getter reads it.
        out = [
            {
                "type": "execute",
                "parameters": {
                    "command": "pkill -f 'soffice.bin|libreoffice|soffice' || true",
                    "shell": True,
                },
            }
        ]
    elif t == "navigate_to_chrome_extensions":
        # Tier-2/3 revival: cosmetic "ensure UI is ready" nav (nested open_url
        # chrome://extensions); scoring is a disk folder-exists check independent
        # of the browser UI, so no-op it.
        context.normalization_notes["navigate_to_chrome_extensions_noop"] += 1
        out = [{"type": "wait", "parameters": {"duration": 0.1}}]
    else:
        return None
    if not out:
        return None
    for item in out:
        context.action_after[runtime_split][item["type"]] += 1
    return out


def _chord_string_key_list_paths(value: Any, prefix: str = "row") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        params = value.get("parameters")
        if value.get("type") == "key" and isinstance(params, dict):
            keys = params.get("keys")
            if isinstance(keys, list):
                for index, token in enumerate(keys):
                    if isinstance(token, str) and "+" in token and token != "+":
                        paths.append(f"{prefix}.parameters.keys[{index}]")
        for key, child in value.items():
            paths.extend(_chord_string_key_list_paths(child, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_chord_string_key_list_paths(child, f"{prefix}[{index}]"))
    return paths


def _repair_known_upstream_chord_key_list(
    payload: dict[str, Any],
    *,
    source_path: Path,
    context: _ImportContext,
) -> None:
    repair = UPSTREAM_CHORD_KEY_LIST_REPAIRS.get(_stable_source_path(source_path))
    if repair is None:
        return
    action_index, action_type, chord = repair

    changed_message = f"{source_path}: known upstream key-list repair target changed"
    evaluator = payload.get("evaluator")
    if not isinstance(evaluator, dict):
        raise RuntimeError(changed_message)
    postconfig = evaluator.get("postconfig")
    if not isinstance(postconfig, list) or action_index >= len(postconfig):
        raise RuntimeError(changed_message)
    action = postconfig[action_index]
    if not isinstance(action, dict) or action.get("type") != action_type:
        raise RuntimeError(changed_message)
    if action.get("parameters") != {"keys": [chord]}:
        raise RuntimeError(changed_message)

    action["parameters"] = {"key": chord}
    context.normalization_notes["repair_known_upstream_chord_key_list"] += 1


def _repair_known_upstream_expected_text(
    payload: dict[str, Any],
    *,
    source_path: Path,
    context: _ImportContext,
) -> None:
    repair = UPSTREAM_EXPECTED_TEXT_REPAIRS.get(_stable_source_path(source_path))
    if repair is None:
        return
    expected_func, malformed, corrected = repair

    changed_message = f"{source_path}: known upstream expected-text repair target changed"
    evaluator = payload.get("evaluator")
    if not isinstance(evaluator, dict) or evaluator.get("func") != expected_func:
        raise RuntimeError(changed_message)
    expected = evaluator.get("expected")
    rules = expected.get("rules") if isinstance(expected, dict) else None
    if not isinstance(rules, dict) or rules.get("expected_text") != malformed:
        raise RuntimeError(changed_message)

    rules["expected_text"] = corrected
    context.normalization_notes["repair_known_upstream_expected_text"] += 1


def _normalize_update_desktop_database_params(params: dict[str, Any]) -> dict[str, Any] | None:
    command = params.get("command")
    if isinstance(command, str):
        is_bare_command = command.strip() == "update-desktop-database"
    elif (
        isinstance(command, list)
        and len(command) == 1
        and isinstance(command[0], str)
    ):
        is_bare_command = command[0].strip() == "update-desktop-database"
    else:
        is_bare_command = False
    if not is_bare_command:
        return None
    normalized = copy.deepcopy(params)
    normalized["command"] = (
        "mkdir -p /home/user/.local/share/applications && "
        "update-desktop-database /home/user/.local/share/applications "
        "2>/dev/null || true"
    )
    normalized["shell"] = True
    return normalized


def _rewrite_urls(obj: Any, *, context: _ImportContext) -> Any:
    if isinstance(obj, str):
        if obj in LIVE_URL_REWRITES:
            context.url_rewrite_count += 1
            return LIVE_URL_REWRITES[obj]
        if FILE_CACHE_MAIN in obj:
            context.url_rewrite_count += obj.count(FILE_CACHE_MAIN)
            return obj.replace(FILE_CACHE_MAIN, FILE_CACHE_PINNED)
        return obj
    if isinstance(obj, list):
        return [_rewrite_urls(x, context=context) for x in obj]
    if isinstance(obj, dict):
        return {k: _rewrite_urls(v, context=context) for k, v in obj.items()}
    return obj


def _task_id(
    payload: dict[str, Any],
    runtime_split: str,
    source_domain: str,
    source_path: Path,
) -> str:
    raw = source_path.stem
    if not isinstance(raw, str) or not raw:
        raise RuntimeError(f"payload missing id/new_id for split {runtime_split}")
    return f"scalecua_osworld_{runtime_split}_{source_domain}_{_slug(raw)}"


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def _action_types(payload: dict[str, Any]) -> set[str]:
    out: set[str] = set()

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            t = x.get("type")
            if isinstance(t, str):
                out.add(t)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    for key in ("config", "postconfig", "oracle_actions"):
        walk(payload.get(key))
    walk((payload.get("evaluator") or {}).get("postconfig"))
    return out


def _resolve_host_file(path_value: Any) -> Path | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    p = Path(path_value)
    snapshot = assets.CACHE_DIR / "hf_snapshot"
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
        if snapshot.is_dir():
            candidates.extend(snapshot.rglob(p.name))
    else:
        candidates.extend(
            [
                assets.REPO_ROOT / path_value,
                snapshot / path_value,
                snapshot / "osworld" / path_value,
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _python_heredoc(code: str) -> str:
    return "python3 <<'PY'\n" + code.rstrip() + "\nPY"


def sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(f"failed to read {path}: {exc}") from exc


def _relpath(path: Path) -> str:
    try:
        return path.relative_to(assets.REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _stable_source_path(path: Path) -> str:
    parts = path.parts
    if "hf_snapshot" in parts:
        index = parts.index("hf_snapshot")
        return Path(*parts[index + 1:]).as_posix()
    return _relpath(path)


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    domains = Counter()
    excluded = Counter()
    proxy_count = 0
    for row in rows:
        others = row["metadata"]["others"]
        domains[str(others.get("domain", ""))] += 1
        if others.get("proxy"):
            proxy_count += 1
        reason = others.get("exclude_reason")
        if reason:
            excluded[reason] += 1
    return {
        "rows": len(rows),
        "domain_counts": dict(sorted(domains.items())),
        "excluded_count_by_reason": dict(sorted(excluded.items())),
        "runnable": len(rows) - sum(excluded.values()),
        "proxy_true_count": proxy_count,
    }


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--strict-counts", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(validate_all(strict_counts=args.strict_counts), indent=2, sort_keys=True))
        return
    report = import_all(force_download=args.force_download)
    print(json.dumps(report["splits"], indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
