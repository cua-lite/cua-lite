"""Import upstream CUA-Gym web tasks into the ``lite.cuagym`` catalog.

Downloads the dataset (via utils.dataset), extracts the bundles for every
``platform in {web, cross_app}`` task into
``.cache/web/lite.cuagym_tasks/bundles/<id>/``, and writes the complete
upstream selection to ``train.jsonl`` for registration. Also writes ``apps.txt``
(mocks that can be built unchanged) and
``import_report.json``.

Run (via install.sh, or standalone):
    uv run python \
      lite/gym/envs/lite/cuagym/scripts/utils/import_tasks.py \
      --backend web --refresh
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from lite.gym.envs.lite.cuagym.src.utils import dataset

ENV_DIR = Path(__file__).resolve().parents[2]
CACHE = ENV_DIR / ".cache" / "web"
HUB = CACHE / "cua-gym-hub"
TASKS_DIR = CACHE / "lite.cuagym_tasks"
REPORT = TASKS_DIR / "import_report.json"
REVISION_FILE = TASKS_DIR / ".asset_revision"
DIGEST_FILE = TASKS_DIR / ".asset_digest"
_WRONG_BACKEND_TASK_IDS = {
    # Upstream labels this row as web/google_sheets, but initial_setup.py opens
    # LibreOffice Calc and never presents the declared web mock in Chrome.
    "9bbdfe1c-8098-5771-9cf0-a526a705c266",
}


def _referenced_script_apps(*sources: str) -> set[str]:
    apps: set[str] = set()
    for source in sources:
        apps.update(
            name.lower().removesuffix("_mock")
            for name in re.findall(
                r"__CUA_GYM_([A-Z0-9_]+)_(?:URL|HOST)__",
                source,
            )
        )
    return apps


def _exclude_reason(
    required_apps: list[str] | tuple[str, ...],
    reward: Path,
    *,
    task_id: str = "",
) -> str | None:
    if task_id in _WRONG_BACKEND_TASK_IDS:
        return "broken_setup:wrong_backend"
    if reason := dataset.validation_reason(task_id):
        return reason
    return dataset.reward_defect(reward)


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
    # web = single mock; cross_app = several mocks, same env (mock + browser).
    web = df[df["platform"].isin(["web", "cross_app"])]
    ids = {str(tid) for tid in web["id"]}
    bundles = dataset.bundle_root(TASKS_DIR)
    asset_stamp = dataset.asset_identity()
    refresh = args.refresh or not _cache_is_fresh()
    selected = ids if refresh else {tid for tid in ids if not (bundles / tid).exists()}
    verb = "refreshing" if refresh else "extracting missing"
    print(f"[import] {len(ids)} web+cross_app tasks in dataset; {verb} {len(selected)} bundles ...")
    if selected:
        dataset.extract_bundles(tar, selected, bundles, refresh=refresh)

    index = []

    for _, row in web.iterrows():
        tid = str(row["id"])
        # app_type is one mock (web) or a comma-list (cross_app). Landing app =
        # the first; apps = all (agent goto targets).
        parts = [
            p.strip().removesuffix("_mock") for p in str(row["app_type"]).split(",") if p.strip()
        ]
        app = parts[0] if parts else ""
        bundle = bundles / tid
        setup, reward, tj = bundle / "initial_setup.py", bundle / "reward.py", bundle / "task.json"
        if not (setup.exists() and reward.exists() and tj.exists()):
            raise RuntimeError(f"{tid}: upstream web bundle is incomplete")
        try:
            setup_source = setup.read_text()
            reward_source = reward.read_text()
            with tj.open() as stream:
                instr = json.load(stream)["instruction"]
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError) as exc:
            raise RuntimeError(f"{tid}: upstream web bundle is unreadable: {exc}") from exc
        mock_parts = {
            name
            for name in parts
            if (HUB / "websites" / f"{name}_mock").is_dir()
        }
        required_apps = sorted(
            mock_parts | _referenced_script_apps(setup_source, reward_source)
        )
        # Annotate (never drop) rows that cannot run.
        reason = dataset.instruction_defect(instr) or _exclude_reason(
            required_apps, reward, task_id=tid
        )
        task = {
            "task_id": tid,
            "instruction": instr,
            "max_steps": 30,
            "metadata": {
                "setup": str(setup),
                "reward": str(reward),
                "others": {
                    "app": app,
                    "apps": required_apps,
                    "source": "lite.cuagym",
                    **({"exclude_reason": reason} if reason else {}),
                },
            },
        }
        index.append(task)

    # train.jsonl — one CUABenchTaskDataRow per task for main.py's
    # register_jsonl_tasks (setup_fn / evaluate_final_fn read metadata.setup /
    # metadata.reward to run the official scripts inside the container).
    catalog = TASKS_DIR / "train.jsonl"
    dataset.write_jsonl_atomic(catalog, index)
    # apps to bake into the image = every app referenced by any setup/reward
    # script (including cross-referenced mocks, _mock suffix stripped), restricted
    # to those with a mock dir in the Hub. install.sh stages these into the image.
    ref = set()
    for t in index:
        ref.update(t["metadata"]["others"]["apps"])
        for f in (t["metadata"]["setup"], t["metadata"]["reward"]):
            ref.update(_referenced_script_apps(Path(f).read_text()))
    apps = sorted(
        a
        for a in ref
        if (HUB / "websites" / f"{a}_mock").is_dir()
    )
    (TASKS_DIR / "apps.txt").write_text("\n".join(f"{a}_mock" for a in apps) + "\n")
    print(f"[import] wrote {len(index)} tasks across {len(apps)} apps")
    print(f"[import] published catalog -> {catalog}")
    print(f"[import] apps: {apps}")
    by_app = dict(Counter(t["metadata"]["others"]["app"] for t in index))
    report = {
        "source_repo": dataset.REPO,
        "asset_snapshot": dataset.asset_snapshot(),
        "platform": "web+cross_app",
        "counts": dataset.row_count_summary(index),
        "upstream_rows": len(web),
        "registered": len(index),
        "by_app": by_app,
        "apps_baked": apps,
        "parquet": str(parquet),
        "archive": str(tar),
        "refreshed_bundles": bool(refresh),
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    REVISION_FILE.write_text(asset_stamp + "\n")
    DIGEST_FILE.write_text(dataset.task_cache_digest(TASKS_DIR) + "\n")
    lock = dataset.maybe_write_catalog_count_lock({
        "desktop": ENV_DIR / ".cache" / "desktop" / "lite.cuagym_desktop_tasks" / "train.jsonl",
        "web": catalog,
    })
    if lock is not None:
        print(f"[import] refreshed count lock -> {dataset.CATALOG_LOCK_PATH}")
    print(f"[import] wrote report -> {REPORT}")


if __name__ == "__main__":
    main()
