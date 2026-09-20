"""Import upstream CUA-Gym desktop tasks into the ``lite.cuagym`` catalog.

Downloads the dataset (via utils.dataset), extracts every ``platform ==
desktop`` bundle plus upstream rows with missing ``platform`` into
``.cache/desktop/lite.cuagym_desktop_tasks/bundles/<id>/``, and writes
``train.jsonl`` (one CUABenchTaskDataRow per task) for main.py's
register_jsonl_tasks. setup_fn / evaluate_final_fn read the bundle's
official initial_setup + reward scripts via the paths in each row's metadata.

Imports two setup kinds: script-type (setup_kind py/sh) and document-seed
(pptx/docx/xlsx — a seed document opened via the task.json download/open step).
Every selected upstream row is registered. Broken setup or reward logic remains
an upstream task-level failure and is isolated by the rollout driver.

Run (via install.sh, or standalone):
    uv run python \
      lite/gym/envs/lite/cuagym/scripts/utils/import_tasks.py \
      --backend desktop --refresh
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from lite.gym.envs.lite.cuagym.src.utils import dataset

ENV_DIR = Path(__file__).resolve().parents[2]
CACHE = ENV_DIR / ".cache" / "desktop"
TASKS_DIR = CACHE / "lite.cuagym_desktop_tasks"
REPORT = TASKS_DIR / "import_report.json"
REVISION_FILE = TASKS_DIR / ".asset_revision"
DIGEST_FILE = TASKS_DIR / ".asset_digest"
_SCRIPT_KINDS = {"py", "sh"}
_DOC_KINDS = {"pptx", "docx", "xlsx"}  # setup is a seed document + task.json download/open

#: Curated: setups whose gate NO branch can satisfy (the mechanical `reward_defect`
#: check cannot see these — the reward is fine, the setup never runs).
_UNSATISFIABLE_SETUP_TASK_IDS = {
    # initial_setup.sh verifies with
    #   code --list-extensions | grep -q "^ritwickdey.LiveServer$"
    # but VS Code prints the canonical id LOWERCASED, so the online branch fails
    # even after a successful `--install-extension`; the offline fallback writes a
    # stub whose package.json is publisher=ritwickdey name=live-server, i.e. id
    # `ritwickdey.live-server`, which cannot match that anchored grep either.
    # Under `set -euo pipefail` the script exits 1 on both paths.
    "7d7ee7fd-9d4b-557e-a25f-5c3ad294cdc7",
}

_EXTERNAL_SETUP_DEPENDENCY_TASK_IDS = {
    # initial_setup.py runs `npm install --save-dev eslint@7.32.0 prettier@2.8.8`
    # during reset. The bundle should have staged these packages in the pinned
    # image/cache, not fetched them during task setup.
    "c3ef8fcc-f312-531b-a857-acb953d0afb3",
    # Multi-app reset failures. These setup scripts try to install packages
    # during reset instead of using pinned/baked assets, so the agent never reaches
    # turn_00 when the installer path is unavailable or mis-specified:
    # * math grading verification calls `sys.executable -m pip install ...` under
    #   /opt/env/venv, whose no-pip shape is an intentional runtime contract;
    # * EPUB and frozen-Writer setups map package names to invalid module names
    #   (`python-docx` -> `python_docx`, `python-pptx` -> `python_pptx`) and then
    #   call `python3 -m pip install ...`;
    # * OCR setup hard-codes `echo "password" | sudo -S apt-get ...` and depends
    #   on apt OCR packages at reset time.
    # * PDF setup probes `importlib.import_module("Pillow")` instead of `PIL`,
    #   then tries `pip install --user Pillow` during reset.
    # * teaser-deck setup live-installs unpinned Pillow, then calls the
    #   removed `ImageDraw.textsize` API during reset before turn_00.
    # * Calc setup asks LibreOffice for the invalid headless conversion
    #   filter `ods:"Calc8"`; LibreOffice exits 0 but writes no `.ods`, so the
    #   stated starting spreadsheet never exists.
    "1a92f790-8527-509c-80cf-c37373d7effe",
    "77e22e90-e016-5bfa-a06c-484efcc4ef7a",
    "b4ba374e-f593-567a-9abb-22461e8b0689",
    "a093712a-9e99-5dbe-a25e-c40025fda4df",
    "5d90b9b5-6539-543b-b19b-1fb20e4eebe3",
    "5aa9bb0b-2a92-54aa-b4fa-5ffa83ccae69",
    "6e0a27f8-f39e-5ded-ba3d-848c6db975cd",
}

_MISSING_SEED_FILE_SETUP_TASK_IDS = {
    # task.json asks the agent to update /home/user/Dev/api_reference.md, and
    # reward.py grades that path. initial_setup.sh creates only
    # api_docs_standard.md, so the stated input document is absent at turn_00.
    "77a46ae0-2a04-5537-8408-bd7e87c67274",
    # PDF setup: task asks to convert Desktop/blueprint.dwg, but setup
    # creates only a placeholder target blueprint.pdf and never seeds the DWG.
    "a119da79-d2cb-5019-b5cd-4ff00c32d1b5",
}

_SETUP_SYNTAX_ERROR_TASK_IDS = {
    # initial_setup.sh has an `if ! python3 - <<PY 2>/dev/null; then ... PY then`
    # duplicate `then`, so reset aborts before turn_00.
    "fe1e8f73-5c46-56f9-8fae-debd3cc49918",
}

_NO_TASK_WINDOW_SETUP_TASK_IDS = {
    # VSCode setup installs/disables extensions and launches Code,
    # but the desktop window gate consistently saw no task window after setup.
    "918c2c55-2474-57db-ae86-a6a06f2e8b76",
}

_REWARD_NO_SENTINEL_TASK_IDS = {
    # reward.py defines verify_apa_block_quotes(), which can print REWARD, but
    # never calls it at top level.
    "b3668f33-b8a2-5a52-b737-0ca4cbe6296f",
}

# Curated: reward and instruction disagree in ways visible from saved trajectories.
# Keep the upstream bundles unchanged and annotate the catalog rows instead:
# * VLC playlist task: instruction names no file format; reward accepts only XSPF.
# * West-LA tennis task: reward de-duplicates city/ZIP fragments, not street addresses.
# * Python Makefile task: instruction asks for coverage <80% failure, reward accepts only
#   literal `--cov-fail-under=80`, not an equivalent variable-expanded threshold.
# * VS Code call-stack task: instruction asks to view the call stack; reward also
#   requires a hidden `debug.uxstate == "simple"` workspace state.
# * Calc energy YoY task: instruction does not require adding two summary rows;
#   reward accepts only B6:E7 and rejects the natural per-row E/F-column layout.
# * Impress onboarding task: task says save `Onboarding.pptx` on Desktop, but
#   reward prefers the pre-seeded one-slide `/home/user/impress_wf_032.pptx`.
# * Writer quiz booklet task: task names only `instructor_test_booklet.odt`;
#   reward checks `/home/user/`, rejecting a correct `~/Documents/` save.
# * Writer envelope task: the visible envelope addresses are in Writer frames/text
#   boxes, but reward reads normal DOCX paragraphs and sees an empty envelope.
# * Calc tax lookup task: instruction asks for a lookup formula using named ranges,
#   while reward requires the literal `VLOOKUP` function and rejects equivalent
#   `LOOKUP` output using the same named ranges.
# * Writer thesis task: instruction asks to merge chapter files, but setup/reward
#   seed and grade an unrelated Heading 2 style document.
# * Calc changeover task: instruction asks for calculated duration/lost-production
#   formulas, while reward accepts only exact string forms and rejects equivalent
#   `ROUND((F-E)*1440,0)` and `I*K/60`.
# * GIF task: instruction explicitly says to use VLC and GIMP; setup pre-extracts
#   frames and reward checks only a GIF artifact, so terminal ffmpeg/convert passes.
# * Impress divider task: reward checks transitions by physical slide part number
#   after insertion, not by the logical sldId order it uses for the other divider
#   components.
# * Placeholder/GIMP task: instruction requires a terminal fetch from
#   via.placeholder.com followed by GIMP conversion, but setup pre-creates the
#   image and reward only checks the final PNG artifact.
# * GIMP map-sprite cleanup task: instruction requires a GIMP fuzzy-select
#   workflow while reward accepts a Python-generated artifact as full success.
# * Impress golden-path task: setup opens the `_t.pptx` deck while reward
#   hard-codes `_t_golden.pptx`.
# * Researcher-data Writer task: instruction references missing context for the
#   researcher details that the reward expects.
# * Michelin Calc task: instruction does not specify the exact output path/format,
#   while reward path-gates only `/home/user/Desktop/nyc_michelin.ods`.
# * Currency-format Calc task: a visually correct `$#,##0.00` format is rejected
#   because the OOXML number format string stores an escaped dollar sign.
# * VS Code extract-method task: reward regex extracts/matches the wrong method
#   body and under-scores a plausible `buildQueryString` implementation.
# * Python PEP list task: prompt asks for referenced PEP numbers/titles, while
#   reward requires a long "What's New" wording instead of the PEP page title.
# * Impress slide-16 task: setup opens `_and_.pptx`, reward hard-codes
#   `_and__golden.pptx`.
# * Impress table-row task: prompt uses a Writer pagination concept on PPTX;
#   reward checks a non-standard DrawingML `a:noSplit` predicate.
# * Writer memo task: prompt asks for memo header/bullet summaries; reward grades
#   a References/Bibliography section that setup explicitly omits.
# * Writer endnote task: prompt asks for `civil_war_thesis.docx`, but setup seeds
#   unrelated `writer_acad_039.docx` content and reward grades research-question
#   prefixes.
# * Impress PDF-export task: prompt names only `CloudSync_Proposal.pdf`, while
#   reward hard-codes `/home/user/Desktop/CloudSync_Proposal.pdf`.
# * Impress org chart task: prompt names hierarchy/connectors only, while reward
#   withholds credit for hidden 12pt text-box font.
# * Writer privacy-policy comparison task: setup/reward are for a legal-memo
#   footnote insertion task, not privacy-policy tracked comparison.
# * VS Code C++ sim task: reward name-gates `RTOS|Scheduler` and exact
#   `test_gpio.cpp`/`test_uart.cpp` filenames not required by the prompt.
# * VS Code compound launch task: reward rejects modern `debugpy` program
#   launch and `pwa-chrome` despite the compound correctly launching Flask and
#   Chrome.
# * Writer hyphenation task: prompt tells the agent to add literal
#   `bio-in-for-mat-ics`, while reward accepts only LibreOffice's internal
#   `bio=in=for=mat=ics` encoding.
# * VS Code TODO task: prompt claims the Python file uses `// TODO` comments,
#   but setup seeds `# TODO` lines and reward accepts only the hybrid
#   `# /* TODO: ... */` form.
# * Maze game task: prompt permits a semantic fix by translating direction
#   strings in `main.py`, but reward requires changing `Player.move`'s
#   signature to one non-dx/dy parameter.
# * Calc/Writer/Impress/VS Code pins: trajectory audits found more semantic
#   reward false negatives and one missing golden deck path. These are task-local
#   evaluator/spec defects, not CUA-Lite runtime failures.
# * Calc GPA ranking task: prompt asks for names and GPAs in `ranking.txt`, but
#   reward rejects a natural header line via an exact 12-line gate.
# * Calc dashboard task: prompt asks only for dependencies, progress bars, and a
#   burndown chart, while reward withholds most credit for hidden title/header
#   styling, dropdown validation, and exact chart-title details.
# * VS Code Java import task: upstream marks it Calc and the prompt asks for
#   import organization behavior, while reward checks hidden Java setting keys.
# * VS Code logpoint task: a visible logpoint can be rejected because reward
#   reads only the persisted `debug.breakpoints` workspace-state key.
# * VS Code devcontainer task: prompt asks for postCreateCommand only, while
#   reward requires hidden `name` and `image` fields.
# * VLC snapshot/Writer task: the visible embedded snapshot is stored as a
#   non-inline image, but reward only credits inline shapes.
# * PDF QA report task: prompt asks for natural checks, while reward's regexes
#   require unstated numeric wording for text/page-size/link sections.
# * Calc account-mask task: prompt asks for masked values, while reward requires
#   the literal `REPLACE` formula and rejects equivalent `LEFT`/`RIGHT` formulas.
# * PDF portfolio task: prompt asks for pikepdf attachments, while reward
#   withholds credit if a compressed valid portfolio is smaller than source bytes.
# * two-column PDF task: prompt names only Summary/Key Metrics layout, while
#   reward requires hidden Q4/satisfaction/revenue/growth/customer/NPS content.
_REWARD_INSTRUCTION_MISMATCH_TASK_IDS = {
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

_METADATA_OVERRIDES = {
    # Upstream app_type says libreoffice_calc, but the bundle setup/reward are
    # VS Code editor settings. Keep the task runnable; correct only local
    # sampling metadata so app_type-stratified rollout does not pull it as Calc.
    "0f32736b-ccdf-50c3-94cf-51c3a278a198": {
        "app_type": "vscode",
        "app_family": "desktop",
    },
    # Upstream marks this as Calc, but the task opens GIMP/Python image cleanup
    # assets. It is also excluded as a reward/spec mismatch.
    "ab46f49b-8b90-599b-a4f3-fd1a61d2f297": {
        "app_type": "multi_apps",
        "app_family": "other",
    },
    # Upstream marks this PDF-signing task as Calc; keep it collectable but move
    # it out of Calc-stratified sampling.
    "78030638-08ea-5d6e-991c-87cf4a25a6d9": {
        "app_type": "pdf",
        "app_family": "desktop",
    },
    # Upstream marks this invoice/PDF generation task as Calc; keep it
    # collectable, but do not let Calc-stratified sampling pull it.
    "577cd0c1-b76f-581e-a628-ad46c8e321bc": {
        "app_type": "pdf",
        "app_family": "desktop",
    },
    # Upstream marks this Python/VSCode Flappy Bird debugging task as Calc.
    "ad645b23-87a6-58b2-9914-755a2af12ab2": {
        "app_type": "vscode",
        "app_family": "desktop",
    },
    # Upstream marks this Python/gedit maze debugging task as Calc.
    "c30e9cf1-c374-53f7-aacf-7f6d7a56975b": {
        "app_type": "multi_apps",
        "app_family": "other",
    },
    # Upstream marks this GIMP/OpenCV product cutout task as Calc.
    "377a0ccb-6013-5f8e-9631-ca06461b21ed": {
        "app_type": "multi_apps",
        "app_family": "other",
    },
    # Upstream marks this Jekyll/VS Code config-edit task as Calc.
    "cf9cf22d-f111-5898-9f16-1f2451865827": {
        "app_type": "vscode",
        "app_family": "desktop",
    },
    # Upstream marks this Java import-organization settings task as Calc.
    "10a0eae5-ba1b-5487-9663-6f5c5d9e62f3": {
        "app_type": "vscode",
        "app_family": "desktop",
    },
    # Upstream marks this Impress/PDF renovation showcase task as Calc; keep it
    # collectable, but do not let Calc-stratified sampling pull it.
    "7b5b471d-b4a0-51d3-99cd-1d0a54a9d322": {
        "app_type": "libreoffice_impress",
        "app_family": "desktop_office",
    },
}


def _exclude_reason(tid: str, reward: Path) -> str | None:
    if tid in _UNSATISFIABLE_SETUP_TASK_IDS:
        return "broken_setup:unsatisfiable_gate"
    if tid in _EXTERNAL_SETUP_DEPENDENCY_TASK_IDS:
        return "broken_setup:external_dependency"
    if tid in _MISSING_SEED_FILE_SETUP_TASK_IDS:
        return "broken_setup:missing_seed_file"
    if tid in _SETUP_SYNTAX_ERROR_TASK_IDS:
        return "broken_setup:syntax_error"
    if tid in _NO_TASK_WINDOW_SETUP_TASK_IDS:
        return "broken_setup:no_task_window"
    if tid in _REWARD_NO_SENTINEL_TASK_IDS:
        return "broken_reward:no_sentinel"
    if tid in _REWARD_INSTRUCTION_MISMATCH_TASK_IDS:
        return "broken_reward:instruction_mismatch"
    if reason := dataset.validation_reason(tid):
        return reason
    return dataset.reward_defect(reward)


def _missing(value: object) -> bool:
    return value is None or value != value


def _cell(value: object) -> str:
    return "" if _missing(value) else str(value)


def _source_platform(value: object) -> str:
    return "missing" if _missing(value) else str(value)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="replace extracted task bundles with the current upstream archive",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="force Hugging Face snapshot revalidation before importing",
    )
    return parser.parse_args()


def _cache_is_fresh() -> bool:
    cached_revision = (
        REVISION_FILE.read_text().strip() if REVISION_FILE.is_file() else ""
    )
    if cached_revision != dataset.asset_identity():
        return False
    cached_digest = DIGEST_FILE.read_text().strip() if DIGEST_FILE.is_file() else ""
    if not cached_digest:
        return False
    try:
        return cached_digest == dataset.task_cache_digest(TASKS_DIR)
    except (OSError, ValueError):
        return False


def main() -> None:
    args = _args()
    parquet, tar = dataset.download(CACHE, force_download=args.force_download)
    df = dataset.read_tasks(parquet)
    desk = df[(df["platform"] == "desktop") | df["platform"].isna()]
    ids = {str(tid) for tid in desk["id"]}
    bundles = dataset.bundle_root(TASKS_DIR)
    asset_stamp = dataset.asset_identity()
    refresh = args.refresh or not _cache_is_fresh()
    selected = ids if refresh else {tid for tid in ids if not (bundles / tid).exists()}
    verb = "refreshing" if refresh else "extracting missing"
    print(
        f"[import] {len(ids)} desktop/no-platform tasks in dataset; "
        f"{verb} {len(selected)} bundles ..."
    )
    if selected:
        dataset.extract_bundles(tar, selected, bundles, refresh=refresh)

    rows = []

    for _, r in desk.iterrows():
        tid = str(r["id"])
        kind = str(r["setup_kind"])
        bundle = bundles / tid
        reward, tj = bundle / "reward.py", bundle / "task.json"
        if kind in _SCRIPT_KINDS:
            setup = bundle / ("initial_setup.sh" if kind == "sh" else "initial_setup.py")
        elif kind in _DOC_KINDS:
            setup = bundle / f"initial_setup.{kind}"
        else:
            raise RuntimeError(f"{tid}: unsupported upstream setup_kind={kind!r}")
        if not (setup.exists() and reward.exists() and tj.exists()):
            raise RuntimeError(f"{tid}: upstream desktop bundle is incomplete")
        try:
            with tj.open() as stream:
                tjd = json.load(stream)
            instr = tjd["instruction"]
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError) as exc:
            raise RuntimeError(
                f"{tid}: upstream desktop task.json is unreadable: {exc}"
            ) from exc
        md = {
            "setup": str(setup),
            "reward": str(reward),
            "setup_kind": kind,
            "postconfig": (tjd.get("evaluator") or {}).get("postconfig", []),
            "others": {
                "app_type": _METADATA_OVERRIDES.get(tid, {}).get(
                    "app_type", _cell(r["app_type"])
                ),
                "app_family": _METADATA_OVERRIDES.get(tid, {}).get(
                    "app_family", _cell(r.get("app_family", ""))
                ),
                "difficulty": _cell(r.get("difficulty", "")),
                "source_platform": _source_platform(r.get("platform", "")),
                # Annotate (never drop) rows that cannot run, plus curated
                # pinned reward/spec mismatches. See dataset.EXCLUDE_REASONS.
                **(
                    {"exclude_reason": _reason}
                    if (_reason := dataset.instruction_defect(instr)
                        or _exclude_reason(tid, reward))
                    else {}
                ),
            },
        }
        if kind in _DOC_KINDS:
            # Document-seed task: setup = upload the seed doc to the path in the
            # task.json download step, then open it. Record that dest.
            dest = None
            for step in tjd.get("config", []):
                if step.get("type") == "download":
                    files = step.get("parameters", {}).get("files", [])
                    if files:
                        dest = files[0].get("path")
            if not dest:
                raise RuntimeError(
                    f"{tid}: document task has no upstream download destination"
                )
            md["doc_dest"] = dest
        row = {"task_id": tid, "instruction": instr, "max_steps": 30, "metadata": md}
        rows.append(row)

    catalog = TASKS_DIR / "train.jsonl"
    dataset.write_jsonl_atomic(catalog, rows)
    print(f"[import] wrote {len(rows)} desktop tasks -> {catalog}")
    by_app = dict(Counter(r["metadata"]["others"]["app_type"] for r in rows))
    by_source_platform = dict(Counter(r["metadata"]["others"]["source_platform"] for r in rows))
    print("[import] by app:", by_app)
    print("[import] by source platform:", by_source_platform)
    report = {
        "source_repo": dataset.REPO,
        "asset_snapshot": dataset.asset_snapshot(),
        "platform": "desktop+missing_platform",
        "counts": dataset.row_count_summary(rows),
        "upstream_rows": len(desk),
        "registered": len(rows),
        "by_app": by_app,
        "by_source_platform": by_source_platform,
        "parquet": str(parquet),
        "archive": str(tar),
        "refreshed_bundles": bool(refresh),
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    REVISION_FILE.write_text(asset_stamp + "\n")
    DIGEST_FILE.write_text(dataset.task_cache_digest(TASKS_DIR) + "\n")
    lock = dataset.maybe_write_catalog_count_lock({
        "desktop": catalog,
        "web": ENV_DIR / ".cache" / "web" / "lite.cuagym_tasks" / "train.jsonl",
    })
    if lock is not None:
        print(f"[import] refreshed count lock -> {dataset.CATALOG_LOCK_PATH}")
    print(f"[import] wrote report -> {REPORT}")


if __name__ == "__main__":
    main()
