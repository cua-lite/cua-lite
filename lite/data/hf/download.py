"""Pull a cua-lite HuggingFace dataset back to the canonical local layout.

This is the inverse of :mod:`lite.data.hf.upload`. Given a published
``cua-lite/<Name>`` repo, it:

1. Snapshots the repo via ``huggingface_hub.snapshot_download``.
2. Walks the snapshot, merging per-cohort shard groups.
3. For every row, extracts each embedded image's bytes back into a
   content-addressed image store (``<out>/images/<hash[:2]>/<hash>.<ext>``).
4. Rewrites the row so ``images`` is a list of relative paths into the
   image store; embedded bytes are dropped.
5. Writes the rewritten rows to the canonical non-sharded local path
   (``<plat>/<task_type>/<split>[/<variant>].parquet``).

Downloaded rows reconstruct the canonical staging layout. A partition the hub
stored without a variant is named after its task_type, so a downloaded tree can
differ from a freshly preprocessed one in that name alone. The same
``export_sft`` invocation works against either source.

**Contract — read before adding a validator here.** Rows leaving ``download``
are *layout*-canonical (HF sharding reshaped to the local partition layout,
embedded image bytes extracted into the ``ImageStore``) but **not** necessarily
*content*-canonical. See :func:`download_dataset` for why, and for which gate
owns content instead.

CLI:

    python -m lite.data.hf.download <DatasetName>
    python -m lite.data.hf.download <DatasetName> --out <dir>
    python -m lite.data.hf.download <DatasetName> --revision <sha>
"""

from __future__ import annotations

import argparse
import collections
import itertools
import logging
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import filter_repo_objects

from lite.data.hf.fold import FOLD_COL, unfold_rows
from lite.data.staging import (
    CANONICAL_SPLITS,
    ORG,
    ImageStore,
    coerce_messages,
    coerce_meta,
    dataset_root,
    image_rel_prefix,
    parse_partition_path,
    partition_path,
    prepare_output_dir,
    write_partition,
)

log = logging.getLogger("hf.download")

_SHARD_RE = re.compile(r"^shard-\d+-of-\d+\.parquet$")

# ---------------------------------------------------------------------------
# Image extension detection (HF Image feature serves bytes + path; we set
# path=None at upload time so we have to read the magic bytes here).
# ---------------------------------------------------------------------------

def _image_ext(blob: bytes) -> str:
    if blob.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if blob.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if blob.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if len(blob) >= 12 and blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return "webp"
    if blob[:2] == b"BM":
        return "bmp"
    return "bin"

# ---------------------------------------------------------------------------
# Snapshot walking
# ---------------------------------------------------------------------------

def _is_shard(path: Path) -> bool:
    return bool(_SHARD_RE.match(path.name))

def _classify_parquets(
    repo_root: Path, allow_patterns: list[str] | None = None
) -> tuple[
    dict[Path, list[Path]],   # shard groups: parent_dir -> [shard files]
    list[Path],               # standalone parquets
]:
    """Group shard files by their parent dir; collect non-shard parquets.

    *allow_patterns* bounds the WALK, not just the fetch. ``snapshot_download``
    populates a shared, persistent HF cache dir and then returns that whole dir
    — including shards pulled by *earlier* runs under *different* patterns. So
    with a warm cache the snapshot dir is a superset of what was requested, and
    walking it unfiltered silently drags cohorts outside the pattern into the
    output. Re-applying the patterns here (with hf_hub's own matcher, so the
    walk and the fetch agree exactly) makes ``--allow-patterns`` mean the same
    thing on a cold and a warm cache.
    """
    paths = sorted(repo_root.rglob("*.parquet"))
    if allow_patterns is not None:
        rels = [p.relative_to(repo_root).as_posix() for p in paths]
        kept = set(filter_repo_objects(rels, allow_patterns=allow_patterns))
        paths = [p for p, rel in zip(paths, rels) if rel in kept]
    shard_groups: dict[Path, list[Path]] = collections.defaultdict(list)
    standalone: list[Path] = []
    for p in paths:
        if _is_shard(p):
            shard_groups[p.parent].append(p)
        else:
            standalone.append(p)
    return shard_groups, standalone

def _hf_path_to_canonical(rel: Path) -> tuple[str, str, str, str | None] | None:
    """Map an HF in-repo path to ``(platform, task_type, split, variant_or_None)``.

    Handles every layout permutation upload may emit:

    * ``<plat>/<tt>/<split>.parquet`` (collapsed unsharded)
    * ``<plat>/<tt>/<split>/<variant>.parquet`` (multi-variant unsharded)
    * ``<plat>/<tt>/<split>/shard-*.parquet`` (collapsed sharded)
    * ``<plat>/<tt>/<split>/<variant>/shard-*.parquet`` (multi-variant sharded)
    """
    parts = rel.parts
    fname = parts[-1]
    if not fname.endswith(".parquet"):
        return None

    # Sharded forms: leaf is shard-NNNNN-of-NNNNN.parquet. The second path
    # component is the task_type literal verbatim (no transform).
    if _is_shard(Path(fname)):
        if len(parts) == 4:
            platform, task_type, split, _ = parts
            if split in CANONICAL_SPLITS:
                return platform, task_type, split, None
        if len(parts) == 5:
            platform, task_type, split, variant, _ = parts
            if split in CANONICAL_SPLITS:
                return platform, task_type, split, variant
        return None

    # Unsharded forms — reuse the canonical parser
    parsed = parse_partition_path(rel)
    if parsed is None:
        return None
    platform, task_type, split, variant, _collapsed = parsed
    return platform, task_type, split, variant

# ---------------------------------------------------------------------------
# Row rewrite
# ---------------------------------------------------------------------------

def _rewrite_rows(rows: list[dict], store: ImageStore) -> tuple[list[dict], int]:
    """Drop ``images``-bytes; replace with rel-path strings into *store*.

    Returns ``(rewritten_rows, count_of_newly_written_image_files)``.
    """
    n_new_before = store.count()
    out: list[dict] = []
    for row in rows:
        imgs = row.get("images") or []
        new_paths: list[str] = []
        for img in imgs:
            blob = img.get("bytes") if isinstance(img, dict) else None
            if blob is None:
                raise ValueError("expected HF Image dict with bytes; got " + repr(type(img)))
            ext = _image_ext(blob)
            new_paths.append(store.put(blob, ext=ext))
        new_row = dict(row)
        new_row["images"] = new_paths
        new_row["messages"] = coerce_messages(row["messages"])
        new_row["metadata"] = coerce_meta(row.get("metadata"))
        out.append(new_row)
    return out, store.count() - n_new_before

def _maybe_unfold(rows: list[dict]) -> list[dict]:
    """Expand image-dedup folded rows back to one row per instruction.

    Folded cohorts (grounding/understanding) carry a ``_folded`` column; the
    canonical local format has one instruction per row, so reverse the upload
    side's :func:`fold_rows` here before the rows hit the image-store rewrite.
    """
    if rows and FOLD_COL in rows[0]:
        return unfold_rows(rows)
    return rows

def _read_rows(parquet_path: Path) -> list[dict]:
    pf = pq.ParquetFile(parquet_path)
    names = pf.schema_arrow.names
    if "images" not in names or "messages" not in names:
        log.warning("skip (missing required columns): %s", parquet_path)
        return []
    rows: list[dict] = []
    for batch in pf.iter_batches(batch_size=64):
        rows.extend(batch.to_pylist())
    return rows

# ---------------------------------------------------------------------------
# --allow-patterns handling
# ---------------------------------------------------------------------------
#
# ``--allow-patterns`` is forwarded to ``huggingface_hub.snapshot_download``,
# which matches with **fnmatch glob**, NOT regex. A regex-alternation like
# ``(desktop|browser|mobile)/grounding.action/**`` (which ``export_sft
# --data-paths`` *does* accept) is taken literally by glob and matches 0 files,
# so the pull silently produces an empty dataset. Two guards below:
#   * ``_expand_alternations`` gives ``--allow-patterns`` the same readable
#     ``(a|b|c)`` alternation as ``--data-paths`` — expanded to the cartesian
#     product of plain globs, so ``.``/``*``/``**`` keep glob semantics and
#     alternation-free patterns preserve existing plain-glob behavior.
#   * ``_assert_patterns_match`` fails loud when a pattern matches 0 repo files,
#     so a typo can't silently yield an empty pull.

def _expand_alternations(patterns: list[str] | str | None) -> list[str] | None:
    """Expand shell-glob alternation groups ``(a|b|c)`` into concrete globs.

    ``"(desktop|browser)/grounding.action/**"`` →
    ``["desktop/grounding.action/**", "browser/grounding.action/**"]``.
    Patterns without ``(...)`` are returned
    verbatim, so plain globs (``"*/grounding.action/**"``) are unaffected.
    """
    if patterns is None:
        return None
    if isinstance(patterns, str):
        patterns = [patterns]
    out: list[str] = []
    for pat in patterns:
        # Only groups containing a ``|`` are treated as alternations, so literal
        # parentheses (e.g. ``file_(1).parquet``) are preserved rather than
        # stripped. re.split with a capture group yields
        # [literal, alt, literal, alt, ...]: even indices are literal spans, odd
        # indices are the captured ``a|b`` alternations.
        parts = re.split(r"\(([^()|]*\|[^()]*)\)", pat)
        choices = [
            [seg] if i % 2 == 0 else seg.split("|")
            for i, seg in enumerate(parts)
        ]
        out.extend("".join(combo) for combo in itertools.product(*choices))
    return out

def _assert_patterns_match(
    repo_id: str, patterns: list[str], *, revision: str | None
) -> None:
    """Raise if none of *patterns* (fnmatch globs) match any repo file.

    Prevents ``snapshot_download`` from silently pulling 0 files (rc=0) when a
    pattern is mistyped or uses an unsupported syntax.
    """
    files = HfApi().list_repo_files(repo_id, repo_type="dataset", revision=revision)
    # Use huggingface_hub's OWN filter (the exact function snapshot_download applies
    # allow_patterns with) so this guard predicts the pull EXACTLY. A hand-rolled
    # ``fnmatch`` loop misses hf_hub's ``_add_wildcard_to_directories`` step (a
    # trailing-slash ``dir/`` pattern → ``dir/*``), and would false-raise on a
    # ``mobile/grounding.point/`` directory pattern that snapshot_download honors.
    matched = list(filter_repo_objects(files, allow_patterns=patterns))
    if not matched:
        raise ValueError(
            f"--allow-patterns {patterns} matched 0 of {len(files)} files in "
            f"{repo_id}. Patterns are shell globs (fnmatch), not regex — use e.g. "
            "'*/grounding.action/**' or 'desktop/grounding.action/**' "
            "(alternation '(desktop|browser)/...' is accepted and expanded to globs)."
        )
    log.info("--allow-patterns matched %d/%d repo files", len(matched), len(files))


def _assert_patterns_match_snapshot(snapshot_dir: Path, patterns: list[str]) -> None:
    rels = [
        path.relative_to(snapshot_dir).as_posix()
        for path in snapshot_dir.rglob("*")
        if path.is_file()
    ]
    matched = list(filter_repo_objects(rels, allow_patterns=patterns))
    if not matched:
        raise ValueError(
            f"--allow-patterns {patterns} matched 0 of {len(rels)} files in "
            f"local snapshot {snapshot_dir}. Patterns are shell globs (fnmatch), "
            "not regex."
        )

# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def download_dataset(
    name: str,
    *,
    out_dir: Path | None = None,
    org: str = ORG,
    revision: str | None = None,
    snapshot_dir: Path | None = None,
    allow_patterns: list[str] | str | None = None,
    overwrite: bool = False,
    max_workers: int = 16,
) -> Path:
    """Pull ``<org>/<name>`` and rewrite to canonical local layout.

    **Output contract: layout-canonical, NOT content-canonical.** This function
    reshapes *storage* — HF shard groups merged into the local partition layout,
    embedded image bytes extracted into the ``ImageStore``, ``messages`` /
    ``metadata`` coerced from their transport shape (see ``coerce_messages`` /
    ``coerce_meta``, which normalize shape only: provider tool-call envelopes
    stay envelopes, invalid image refs stay invalid). It deliberately does **not**
    assert that row *content* matches the current LiteSample schema, and must
    not be made to.

    **Why no content gate belongs here.** ``download`` is the only entry point
    whose input is *by definition* possibly-unmigrated: it reads historical data
    already published on HF, which may be in any past format. The publication
    gate belongs to canonical producers: ``stage`` takes a local log-root (fresh
    rollout, or post-migration) and gates content with
    ``validate_canonical_rows``; fresh preproc publication paths run the same
    validator before upload. ``upload`` then only transports rows that already
    reached the canonical local layout. Adding a content gate here instead
    demands that data be repaired before it can be fetched *for repair* — which
    makes step 1 of ``devs/migration`` (pull the old rows) unrunnable on exactly
    the rows migration exists to fix.

    *max_workers* is the shard fetch concurrency (``snapshot_download``'s own
    default is 8; these repos are hundreds of shards, so the default here is
    higher).

    *allow_patterns* restricts BOTH the fetch and the walk to matching paths —
    useful to pull a single cohort (``mobile/grounding.point/**``) instead of the
    full repo. Shell-glob syntax (fnmatch); a ``(desktop|browser|mobile)/...``
    alternation is accepted and expanded to globs. A pattern that matches 0 repo
    files raises (no silent empty pull).

    Returns the resolved output directory.
    """
    repo_id = f"{org}/{name}"
    allow_patterns = _expand_alternations(allow_patterns)
    if snapshot_dir is None:
        if allow_patterns is not None:
            _assert_patterns_match(repo_id, allow_patterns, revision=revision)
        log.info("snapshot_download %s (allow_patterns=%s)", repo_id, allow_patterns)
        snapshot_dir = Path(snapshot_download(
            repo_id=repo_id, repo_type="dataset", revision=revision,
            allow_patterns=allow_patterns, max_workers=max_workers,
        ))
    else:
        snapshot_dir = Path(snapshot_dir)
        if allow_patterns is not None:
            _assert_patterns_match_snapshot(snapshot_dir, allow_patterns)

    out = prepare_output_dir(
        out_dir or dataset_root(name),
        overwrite=overwrite,
        label="download output directory",
        protected_roots=(snapshot_dir,),
    )
    store = ImageStore(out / "images", rel_prefix=image_rel_prefix(name))

    shard_groups, standalone = _classify_parquets(snapshot_dir, allow_patterns)

    # Map from canonical key → row buffer
    buffers: dict[tuple[str, str, str, str | None], list[dict]] = collections.defaultdict(list)
    n_new_total = 0
    n_rows_total = 0

    # 1. shard groups: merge sibling shards into one row stream
    for shard_dir, shard_files in sorted(shard_groups.items()):
        rel = shard_dir.relative_to(snapshot_dir) / shard_files[0].name
        canon = _hf_path_to_canonical(rel)
        if canon is None:
            log.warning("unrecognized layout (sharded), skipping: %s", rel)
            continue
        merged: list[dict] = []
        for sf in sorted(shard_files):
            merged.extend(_read_rows(sf))
        if not merged:
            continue
        merged = _maybe_unfold(merged)
        rewritten, n_new = _rewrite_rows(merged, store)
        buffers[canon].extend(rewritten)
        n_new_total += n_new
        n_rows_total += len(rewritten)
        log.info(
            "%s (%d shards merged): %d rows, %d new images",
            "/".join(canon[:3]) + ("/" + canon[3] if canon[3] else ""),
            len(shard_files), len(rewritten), n_new,
        )

    # 2. standalone parquets — defensive, hub_upload always shards but small
    # repos may not.
    for parquet in standalone:
        rel = parquet.relative_to(snapshot_dir)
        canon = _hf_path_to_canonical(rel)
        if canon is None:
            continue
        rows = _read_rows(parquet)
        if not rows:
            continue
        rows = _maybe_unfold(rows)
        rewritten, n_new = _rewrite_rows(rows, store)
        buffers[canon].extend(rewritten)
        n_new_total += n_new
        n_rows_total += len(rewritten)
        log.info("%s: %d rows, %d new images", rel, len(rewritten), n_new)

    # 3. write each merged buffer to its canonical local path, always EXPANDED
    # so the reconstructed tree matches what ``flush_buffers`` writes. A hub
    # partition stored collapsed carries no variant; name it the way
    # ``collect_stats_from_disk`` already names one, rather than passing "" --
    # that produced a `<split>/.parquet` dotfile, which ``iter_partitions``
    # skips, so the whole split vanished from stats and from any re-upload.
    # A hub partition stored under the older collapsed spelling carries no
    # variant. Naming it after its task_type matches what ``collect_stats_from_disk``
    # does, but that name is also a real variant in this repo (``VARIANT = "use"``
    # under task_type ``use``), so a repo holding BOTH spellings for one split
    # would map two buffers onto one file and lose whichever wrote first. That
    # repo is already corrupt -- it is what an interrupted push leaves behind --
    # so say so instead of silently picking a winner.
    destinations: dict[Path, tuple] = {}
    for key in buffers:
        platform, task_type, split, variant = key
        dest = partition_path(
            out, platform=platform, task_type=task_type, split=split,
            variant=variant or task_type.split(".")[-1],
        )
        if dest in destinations:
            raise ValueError(
                f"{repo_id} stores {platform}/{task_type}/{split} in both the "
                f"collapsed and the expanded layout; {destinations[dest]} and {key} "
                f"both resolve to {dest}. Re-push the dataset so one layout wins, "
                "then download again."
            )
        destinations[dest] = key

    for (platform, task_type, split, variant), rows in buffers.items():
        path = partition_path(
            out,
            platform=platform,
            task_type=task_type,
            split=split,
            variant=variant or task_type.split(".")[-1],
        )
        write_partition(rows, path)

    log.info(
        "done: %d rows, %d newly-stored images under %s",
        n_rows_total, n_new_total, out,
    )
    return out

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("name", help="dataset repo name (e.g. ScaleCUA)")
    p.add_argument("--org", default=ORG, help=f"HF org (default: {ORG})")
    p.add_argument("--out", type=Path, default=None,
                   help=f"output dir; defaults to $CUA_LITE_DATASETS_ROOT/{ORG}/<name>")
    p.add_argument("--revision", default=None, help="HF revision (branch / sha / tag)")
    p.add_argument("--snapshot-dir", type=Path, default=None,
                   help="reuse an existing snapshot dir instead of re-downloading")
    p.add_argument("--allow-patterns", action="append", default=None,
                   help="restrict the snapshot AND the walk to matching paths (so a warm "
                        "HF cache cannot drag in cohorts you did not ask for); "
                        "repeat for multiple "
                        "patterns. Shell globs (fnmatch), NOT regex — but a "
                        "'(desktop|browser|mobile)/...' alternation is expanded to globs. "
                        "A pattern matching 0 files errors (no silent empty pull). "
                        "Examples: 'mobile/grounding.point/**', '*/grounding.action/**', "
                        "'(desktop|browser)/grounding.action/**'")
    p.add_argument("--overwrite", action="store_true",
                   help="replace an existing non-empty output dir. Without this, download "
                        "requires a fresh --out so old cohorts cannot survive a subset rerun.")
    p.add_argument("--max-workers", type=int, default=16,
                   help="shard fetch concurrency (default: 16; snapshot_download's own "
                        "default is 8, which is slow for the several-hundred-shard repos)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    download_dataset(
        args.name,
        out_dir=args.out,
        org=args.org,
        revision=args.revision,
        snapshot_dir=args.snapshot_dir,
        allow_patterns=args.allow_patterns,
        overwrite=args.overwrite,
        max_workers=args.max_workers,
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
