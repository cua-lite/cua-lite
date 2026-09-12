"""Shared CUA-Gym asset access — used by both the browser and desktop backends.

Both backends ingest the SAME mirrored asset release: a ``tasks.parquet`` index,
a ``cua_gym_tasks_v1.tar.zst`` of per-task bundles, and a clean CUA-Gym-Hub
source snapshot for the web mocks. The default mirror is owned by ``cua-lite``
so installs do not depend on moving upstream GitHub branches.
They differ only in which ``platform`` rows they keep (web/cross_app vs desktop),
so the download + stream-extract lives here and each backend's ``import_tasks``
filters the table and walks its own bundles.

Run: not directly — imported by ``scripts/utils/import_*_tasks.py``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tarfile
import tempfile
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path

import zstandard
from huggingface_hub import hf_hub_download, snapshot_download

from lite.gym.utils.config.manifest import load_asset_lock

_ENV_DIR = Path(__file__).resolve().parents[2]
_PIN = load_asset_lock(_ENV_DIR)
REPO = _PIN.repo
REVISION = _PIN.revision
_PARQUET = _PIN.component_path("task_table")
_TAR = _PIN.component_path("task_bundles")
_HUB_TAR = _PIN.component_path("mock_hub")
CATALOG_LOCK_PATH = _ENV_DIR / "data" / "catalog.lock.json"
VALIDATION_EXCLUDES_PATH = _ENV_DIR / "data" / "validation_excludes.json"


def asset_snapshot() -> dict[str, str]:
    return {
        "repo": REPO,
        "revision": REVISION,
        "task_table": _PARQUET,
        "task_bundles": _TAR,
        "mock_hub": _HUB_TAR,
    }


def asset_identity() -> str:
    """Stable stamp for every lock field that controls imported task bytes."""
    return json.dumps(asset_snapshot(), sort_keys=True, separators=(",", ":"))


def load_catalog_lock() -> dict:
    """Load the tracked count lock for imported CUA-Gym catalogs."""
    return json.loads(CATALOG_LOCK_PATH.read_text())


def row_count_summary(rows: Iterable[dict]) -> dict:
    """Summarize registered / excluded rows using the canonical reason vocabulary."""
    total = 0
    reasons: Counter[str] = Counter()
    for row in rows:
        total += 1
        reason = (
            row.get("metadata", {})
            .get("others", {})
            .get("exclude_reason")
        )
        if reason:
            if reason not in EXCLUDE_REASONS:
                raise RuntimeError(
                    f"unknown exclude_reason {reason!r} "
                    f"(known: {sorted(EXCLUDE_REASONS)})"
                )
            reasons[reason] += 1
    excluded = sum(reasons.values())
    return {
        "rows": total,
        "excluded_rows": excluded,
        "collectable_rows": total - excluded,
        "exclude_reasons": dict(sorted(reasons.items())),
    }


def catalog_count_summary(path: Path) -> dict:
    """Read a generated JSONL catalog and summarize its registered/excluded rows."""
    with path.open() as stream:
        return row_count_summary(
            json.loads(line) for line in stream if line.strip()
        )


def catalog_count_lock(backends: Mapping[str, Path]) -> dict:
    """Build the tracked count lock shape from generated backend catalogs."""
    backend_counts = {
        name: catalog_count_summary(path)
        for name, path in sorted(backends.items())
    }
    reasons: Counter[str] = Counter()
    rows = 0
    excluded = 0
    collectable = 0
    for summary in backend_counts.values():
        rows += summary["rows"]
        excluded += summary["excluded_rows"]
        collectable += summary["collectable_rows"]
        reasons.update(summary["exclude_reasons"])
    return {
        "version": 1,
        "generated": True,
        "sources": {
            "generator": "scripts/utils/import_tasks.py",
            "asset_identity": asset_identity(),
        },
        "splits": {
            "train": {
                "rows": rows,
                "excluded_rows": excluded,
                "collectable_rows": collectable,
                "exclude_reasons": dict(sorted(reasons.items())),
                "backends": backend_counts,
            }
        },
    }


def write_catalog_count_lock(backends: Mapping[str, Path]) -> dict:
    """Refresh ``data/catalog.lock.json`` after importing both backend catalogs."""
    lock = catalog_count_lock(backends)
    CATALOG_LOCK_PATH.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    return lock


def maybe_write_catalog_count_lock(backends: Mapping[str, Path]) -> dict | None:
    """Refresh the count lock when both backend catalogs are present."""
    if not all(path.is_file() for path in backends.values()):
        return None
    return write_catalog_count_lock(backends)


def validate_catalog_count_lock(backends: Mapping[str, Path]) -> None:
    """Validate generated backend catalogs against the tracked count lock."""
    actual = catalog_count_lock(backends)
    expected = load_catalog_lock()
    if actual != expected:
        raise RuntimeError(
            f"{CATALOG_LOCK_PATH}: imported catalog counts do not match lock"
        )


def task_cache_digest(root: Path) -> str:
    """Content digest for an imported CUA-Gym task cache.

    The revision stamp says which locked HF asset release the cache came from;
    this digest catches local/cache mutation after import without making the
    Docker image freshness hash depend on runtime-only task bundles.
    """
    from lite.gym.utils.backend.freshness import tree_content_hash

    markers = {".asset_revision", ".asset_digest"}

    def skip(_base: Path, path: Path) -> bool:
        return path.name in markers

    return tree_content_hash(root, skip=skip)


def bundle_root(root: Path) -> Path:
    """Single task-bundle cache owned by the current locked asset release."""
    return root / "bundles"


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    """Publish a complete JSONL catalog with one same-filesystem rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


#: Closed vocabulary for ``metadata.others.exclude_reason``, env-local (lite.osworld's
#: REGISTRY has no home for these categories, and lite.cuaworld likewise keeps its own).
#:
#: These rows stay REGISTERED — the env's contract is "all pinned upstream rows remain
#: registered; do not silently drop" (devs/envs/lite.cuagym/UPSTREAM_ISSUES.md
#: and devs/envs/lite.cuagym/AGENTS.md). What the contract forbids is dropping rows and patching
#: upstream bundles; ANNOTATING them is what this env already does for trajectories
#: (devs/data/lite.cuagym/AGENTS.md: "keeps every trajectory and tags quality gates in
#: metadata.others.exclude_reason … nothing is physically dropped"). Consumers opt out
#: with the same one-liner every other env uses:
#:     --filter "lambda m: not m.others.get('exclude_reason')"
#:
#: Every category here is a pinned upstream defect that makes the row unusable as
#: a default training signal. Most are UNRUNNABLE BY CONSTRUCTION: a rollout burns
#: a full episode and can only end in a task-level error, which is indistinguishable
#: from a genuine agent failure in an FN/TN analysis. The curated
#: ``broken_reward:instruction_mismatch`` class covers a reward/spec mismatch that
#: runs but silently penalizes an otherwise plausible completion. Measured on the
#: pinned snapshot: 513 of 10910
#: rows (4.70%) — 152 broken_reward:empty, 81 broken_mock:blank_render, 42
#: broken_reward:no_sentinel (26 desktop + 16 web), 26 broken_reward:syntax_error,
#: 19 broken_reward:missing_golden,
#: 1 broken_setup:unsatisfiable_gate, 8 broken_setup:external_dependency,
#: 1 broken_setup:wrong_backend, 2 broken_setup:missing_seed_file,
#: 1 broken_setup:syntax_error, 1 broken_setup:no_task_window, 178
#: broken_reward:instruction_mismatch, 1 broken_task:empty_instruction.
EXCLUDE_REASONS: dict[str, str] = {
    "broken_mock:blank_render": (
        "pinned mock builds but its browser root renders empty, so setup cannot "
        "present a usable task page"
    ),
    "broken_reward:empty": (
        "upstream reward.py is whitespace-only — nothing can print the REWARD sentinel, "
        "so evaluate_final_fn always raises kind='no_reward'"
    ),
    "broken_reward:syntax_error": (
        "upstream reward.py does not compile under the container's py3.12"
    ),
    "broken_reward:no_sentinel": (
        "upstream reward.py has no reachable top-level path that emits the REWARD "
        "sentinel, so every trajectory raises kind='no_reward'"
    ),
    "broken_reward:instruction_mismatch": (
        "upstream reward.py and task.json disagree on a material success "
        "criterion, making the reward unsafe as a default training signal"
    ),
    "broken_reward:missing_golden": (
        "upstream reward.py compares against a `*_golden.*` artifact that only its "
        "authoring run had; setup builds the same file WITHOUT the suffix, so the "
        "comparison target never exists and every trajectory scores a silent zero"
    ),
    "broken_reward:nonzero_baseline": (
        "a live reset + immediate terminate scores above zero before the agent acts"
    ),
    "broken_reward:live_validation_error": (
        "a live reset succeeds, but the official reward fails during no-op validation"
    ),
    "broken_setup:unsatisfiable_gate": (
        "initial_setup.sh gate-aborts on a condition no branch can satisfy"
    ),
    "broken_setup:external_dependency": (
        "initial_setup depends on a live external package/service rather than "
        "pinned in-image assets, so reset can fail before the agent acts"
    ),
    "broken_setup:wrong_backend": (
        "upstream row is registered under the browser backend but its setup launches "
        "a desktop application without presenting the declared web mock"
    ),
    "broken_setup:missing_seed_file": (
        "task.json asks the agent to update a seed file that initial_setup never "
        "creates, so the starting state is not the stated task"
    ),
    "broken_setup:syntax_error": (
        "initial_setup has a shell/Python syntax error and aborts before the "
        "agent can act"
    ),
    "broken_setup:no_task_window": (
        "setup is expected to launch a task GUI, but no usable task window appears"
    ),
    "broken_setup:live_validation_error": (
        "a live no-op validation cannot complete reset before the agent acts"
    ),
    "broken_task:empty_instruction": (
        "upstream task.json carries no instruction, so reset has no prompt to hand the "
        "agent and the episode can only end in a task-level error"
    ),
}


@lru_cache(maxsize=1)
def validation_excludes() -> dict[str, dict[str, str]]:
    """Load the revision-pinned validation findings consumed by both importers."""
    raw = json.loads(VALIDATION_EXCLUDES_PATH.read_text())
    meta = raw.get("_meta") or {}
    if meta.get("asset_revision") != REVISION:
        raise RuntimeError(
            f"{VALIDATION_EXCLUDES_PATH} targets asset revision "
            f"{meta.get('asset_revision')!r}, expected {REVISION!r}"
        )
    findings = {key: value for key, value in raw.items() if key != "_meta"}
    for task_id, finding in findings.items():
        reason = finding.get("reason")
        if reason not in EXCLUDE_REASONS:
            raise RuntimeError(
                f"{VALIDATION_EXCLUDES_PATH}: {task_id} has unknown reason {reason!r}"
            )
    if meta.get("total") != len(findings):
        raise RuntimeError(
            f"{VALIDATION_EXCLUDES_PATH}: _meta.total does not match findings"
        )
    return findings


def validation_reason(task_id: str) -> str | None:
    finding = validation_excludes().get(task_id)
    return finding["reason"] if finding else None


#: Does the source contain a statement that can EMIT the reward sentinel?
#:
#: Two traps this has to thread, both found by adversarially auditing the naive
#: forms. (a) A case-SENSITIVE `"REWARD:" in source` test wrongly condemns 4
#: runnable tasks that print `reward:{score}` — `REWARD_RE` in src/utils/reward.py
#: is `re.IGNORECASE` on purpose, because upstream is inconsistent about casing.
#: (b) A merely case-INSENSITIVE test wrongly ABSOLVES 14 genuinely-broken bundles
#: (12 web, 2 desktop) whose docstring merely NAMES the sentinel while the code only
#: ever prints a bare score — the 12 web ones all open `Reward: <task summary>`
#: (e.g. `Reward: Verify duplicate item creation on Monday Team Projects…`), the 2
#: desktop ones spell out the contract (`REWARD: {score}`). So: match
#: case-insensitively, but only on a line that also writes output.
_EMITS_SENTINEL_RE = re.compile(
    r"^(?!\s*#).*\b(?:print|write|stdout)\b.*reward\s*:", re.IGNORECASE | re.MULTILINE
)


def instruction_defect(instruction: str) -> str | None:
    """``broken_task:empty_instruction`` when a bundle's task.json states no task, else None.

    Mechanical, like :func:`reward_defect`: an empty prompt is a property of the pinned
    upstream row, so a future asset revision that carries more of them is annotated
    without anyone curating an id list.
    """
    return None if instruction.strip() else "broken_task:empty_instruction"


def reward_defect(reward_path: Path) -> str | None:
    """``broken_reward:*`` when a bundle's reward can never produce a score, else None."""
    source = reward_path.read_text(errors="replace")
    if not source.strip():
        return "broken_reward:empty"
    try:
        compile(source, str(reward_path), "exec")
    except SyntaxError:
        return "broken_reward:syntax_error"
    if not _EMITS_SENTINEL_RE.search(source):
        # The reward is the ONLY thing that can print the sentinel — there is no
        # harness wrapper — so a script that can never emit it raises
        # CuaGymTaskError(kind="no_reward") on every trajectory. These rows compile
        # fine, so the two checks above miss them.
        return "broken_reward:no_sentinel"
    if _reads_an_unbuilt_golden(source, reward_path.parent):
        return "broken_reward:missing_golden"
    return None


#: ``/home/user/<name>_golden.<ext>`` — the upstream generator renamed the artifact for
#: its solution run and the reward script kept that path.
_GOLDEN_PATH_RE = re.compile(r"""['"](/home/user/[^'"\n]{2,200}_golden\.[A-Za-z0-9]{2,5})['"]""")


def _reads_an_unbuilt_golden(source: str, bundle: Path) -> bool:
    """True when the reward compares against a ``*_golden.*`` nothing ever creates.

    Setup writes ``X.ext`` and the reward opens ``X_golden.ext``, so
    ``os.path.exists`` is False on every trajectory and the script returns 0.0 —
    SILENTLY, which is what makes it worse than a crash: the row looks like an agent
    that earned nothing, and a perfect trajectory is published as a confident zero.

    Structural, not an id list, so a future asset revision carrying more of them is
    annotated without anyone curating.
    """
    setups = sorted(bundle.glob("initial_setup.*"))
    if not setups:
        return False
    setup = setups[0].read_text(errors="replace")
    for golden in set(_GOLDEN_PATH_RE.findall(source)):
        if golden in setup:
            continue  # setup really does build it
        name = golden.rsplit("/", 1)[-1]
        pattern = (
            r"(save|write|copy|move|to_excel|to_csv)\([^)]{0,120}"
            + re.escape(name)
        )
        if re.search(pattern, source):
            continue  # the reward builds its own reference
        if golden.replace("_golden.", ".") in setup:
            return True  # setup builds the same artifact WITHOUT the suffix
    return False


def catalog_task_ids(
    path: Path, *, required_metadata_paths: tuple[str, ...] = ()
) -> set[str]:
    """Fully parse a non-empty task catalog before registry mutation begins."""
    seen: set[str] = set()
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                task_id = row["task_id"]
                row["instruction"]
                metadata = row["metadata"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise RuntimeError(f"{path}:{line_number}: invalid task row: {exc}") from exc
            if not isinstance(task_id, str) or not task_id:
                raise RuntimeError(f"{path}:{line_number}: task_id must be non-empty")
            if task_id in seen:
                raise RuntimeError(f"{path}:{line_number}: duplicate task_id {task_id!r}")
            if not isinstance(metadata, dict):
                raise RuntimeError(f"{path}:{line_number}: metadata must be an object")
            reason = (metadata.get("others") or {}).get("exclude_reason")
            if reason is not None and reason not in EXCLUDE_REASONS:
                raise RuntimeError(
                    f"{path}:{line_number}: unknown exclude_reason {reason!r} "
                    f"(known: {sorted(EXCLUDE_REASONS)})"
                )
            for field in required_metadata_paths:
                value = metadata.get(field)
                if not isinstance(value, str) or not Path(value).is_file():
                    raise RuntimeError(
                        f"{path}:{line_number}: metadata.{field} is missing: {value!r}"
                    )
            seen.add(task_id)
    if not seen:
        raise RuntimeError(f"{path}: task catalog is empty")
    return seen


def validate_catalog(
    path: Path, *, required_metadata_paths: tuple[str, ...] = ()
) -> int:
    return len(
        catalog_task_ids(path, required_metadata_paths=required_metadata_paths)
    )


def download(cache_dir: Path, *, force_download: bool = False) -> tuple[Path, Path]:
    """Download the task index + bundle archive into ``cache_dir/dataset``.

    Returns ``(parquet_path, tar_path)``. Idempotent unless ``force_download`` is
    set, in which case Hugging Face revalidates the local snapshot.
    """
    dest = cache_dir / "dataset"
    parquet = hf_hub_download(
        REPO,
        _PARQUET,
        repo_type="dataset",
        revision=REVISION,
        local_dir=str(dest),
        force_download=force_download,
    )
    tar = hf_hub_download(
        REPO,
        _TAR,
        repo_type="dataset",
        revision=REVISION,
        local_dir=str(dest),
        force_download=force_download,
    )
    return Path(parquet), Path(tar)


def download_hub(cache_dir: Path, *, force_download: bool = False) -> Path:
    """Download the mirrored clean CUA-Gym-Hub source tarball."""
    dest = cache_dir / "dataset"
    tar = hf_hub_download(
        REPO,
        _HUB_TAR,
        repo_type="dataset",
        revision=REVISION,
        local_dir=str(dest),
        force_download=force_download,
    )
    return Path(tar)


def snapshot(cache_dir: Path, *, force_download: bool = False) -> Path:
    """Download the full CUA-Gym assets mirror for local inspection."""
    dest = cache_dir / "dataset"
    return Path(
        snapshot_download(
            repo_id=REPO,
            repo_type="dataset",
            revision=REVISION,
            local_dir=str(dest),
            force_download=force_download,
        )
    )


def read_tasks(parquet: Path):
    """Load the full task table as a pandas DataFrame (each backend filters by
    ``platform`` itself)."""
    import pyarrow.parquet as pq

    return pq.read_table(parquet).to_pandas()


def extract_bundles(tar: Path, ids: set[str], dest: Path, *, refresh: bool = False) -> None:
    """Stream-extract just the bundle dirs whose top-level name is in ``ids``
    into ``dest`` (skipping AppleDouble ``._`` entries). Streaming keeps the
    multi-GB archive off disk twice. With ``refresh=True``, each source bundle is
    replaced atomically after the complete archive selection is staged. Runtime
    catalogs refer to the extracted bundles directly.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dest.name}.extracting-", dir=dest.parent))
    try:
        with open(tar, "rb") as fh:
            reader = zstandard.ZstdDecompressor().stream_reader(fh)
            with tarfile.open(fileobj=reader, mode="r|") as archive:
                for member in archive:
                    top = member.name.split("/")[0]
                    if (
                        top in ids
                        and "/._" not in member.name
                        and not Path(member.name).name.startswith("._")
                    ):
                        archive.extract(member, staging, filter="data")
        missing = sorted(task_id for task_id in ids if not (staging / task_id).is_dir())
        if missing:
            raise RuntimeError(
                f"task bundle archive is missing {len(missing)} selected ids: {missing[:5]}"
            )
        if refresh:
            backup = dest.parent / f".{dest.name}.previous-{uuid.uuid4().hex}"
            if dest.exists():
                os.replace(dest, backup)
            try:
                os.replace(staging, dest)
            except Exception:
                if backup.exists() and not dest.exists():
                    os.replace(backup, dest)
                raise
            shutil.rmtree(backup, ignore_errors=True)
            return

        dest.mkdir(parents=True, exist_ok=True)
        for task_id in sorted(ids):
            target = dest / task_id
            if target.exists():
                continue
            incoming = staging / task_id
            os.replace(incoming, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def extract_tar_zst(tar: Path, dest: Path, *, refresh: bool = False) -> None:
    """Extract a whole ``.tar.zst`` archive into ``dest``."""
    if refresh:
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    with open(tar, "rb") as fh:
        reader = zstandard.ZstdDecompressor().stream_reader(fh)
        with tarfile.open(fileobj=reader, mode="r|") as t:
            t.extractall(dest, filter="data")
