"""Push a staged cua-lite dataset to ``cua-lite/<Name>`` on HuggingFace.

Input is a directory in the canonical local layout (see
:mod:`lite.data.staging`):

    <staging>/
        images/<hash[:2]>/<hash>.<ext>
        <platform>/<task_type>/<split>[/<variant>].parquet

``upload`` is a transport step: staging/preproc producers own row validation,
including canonical tool/message shape and coordinate bounds. This module assumes
that stage/preproc already produced the canonical local layout, then embeds image
bytes and pushes it.

This tool:

1. Recomputes ``DatasetStats`` from the on-disk parquets + image store.
2. Renders the canonical README via :mod:`lite.data.hf.card` (reading
   ``repo.json`` from the dataset's preproc directory).
3. For every partition, materializes a HF parquet with embedded image
   bytes:
     - reads rows
     - resolves each image rel-path to bytes from the local image store
     - casts the ``images`` column to ``Sequence(Image())``
     - shards by image-count to keep parquets ≤ HF viewer's 300 MB row
       group limit (``write_page_index=True`` and a small ``batch_size``
       give us deterministic row-group sizing)
4. Batches uploads into ``files_per_commit``-sized HF commits (HF caps
   at 128 commits/hour).
5. Cleans up orphan files left over from prior pushes whose layout
   differed.

CLI:

    python -m lite.data.hf.upload <DatasetName> --preproc-dir <path>
    python -m lite.data.hf.upload <DatasetName> --staging <path> --preproc-dir <path>
    python -m lite.data.hf.upload <DatasetName> --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from datasets import Dataset, Image, Sequence
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, create_repo
from huggingface_hub.errors import HfHubHTTPError

from lite.data.hf.card import load_repo_json, render_card
from lite.data.hf.fold import fold_rows, should_fold
from lite.data.staging import (
    ORG,
    ImageStore,
    collect_stats_from_disk,
    dataset_root,
    image_rel_prefix,
    iter_parquet_rows,
    iter_partitions,
    partition_path,
    write_stats,
)

log = logging.getLogger("hf.upload")

_COMMIT_MAX_RETRIES = 6
_COMMIT_BASE_DELAY = 60  # seconds

def _commit_with_retry(api: HfApi, **kwargs) -> None:
    """Call ``api.create_commit`` with retry + exponential backoff for 429."""
    for attempt in range(_COMMIT_MAX_RETRIES):
        try:
            api.create_commit(**kwargs)
            return
        except Exception as exc:
            if "429" not in str(exc) and "Too Many Requests" not in str(exc):
                raise
            if attempt == _COMMIT_MAX_RETRIES - 1:
                raise
            delay = _COMMIT_BASE_DELAY * (2 ** attempt)
            log.warning("429 rate-limited; retrying in %ds (attempt %d/%d)", delay, attempt + 1, _COMMIT_MAX_RETRIES)
            time.sleep(delay)

# ---------------------------------------------------------------------------
# Sharding
# ---------------------------------------------------------------------------

def _build_shards(rows: list[dict], *, shard_rows: int, shard_images: int) -> list[list[dict]]:
    """Split *rows* into shards bounded by row count and image count.

    The image-count bound keeps each parquet's binary payload well under
    Arrow's 2 GB-per-chunk offset limit and HF viewer's ~300 MB row-group
    cap (multi-image trajectory rows otherwise blow past either).
    """
    shards: list[list[dict]] = []
    cur: list[dict] = []
    cur_imgs = 0
    for row in rows:
        n = len(row.get("images") or [])
        if cur and (len(cur) >= shard_rows or cur_imgs + n > shard_images):
            shards.append(cur)
            cur, cur_imgs = [], 0
        cur.append(row)
        cur_imgs += n
    if cur:
        shards.append(cur)
    return shards

def _rows_to_dataset(rows: list[dict], store: ImageStore) -> Dataset:
    """Build a Dataset whose ``images`` column carries embedded bytes.

    Resolves each row's path-based ``images`` list against the local image
    store. Casting to ``Sequence(Image())`` stamps the HF Image feature in
    the parquet metadata so the Hub viewer renders thumbnails.
    """
    out = []
    for row in rows:
        new = dict(row)
        new["images"] = [
            {"bytes": store.path_of(p).read_bytes(), "path": None}
            for p in row.get("images") or []
        ]
        out.append(new)
    ds = Dataset.from_list(out)
    ds = ds.cast_column("images", Sequence(Image()))
    return ds

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Push driver
# ---------------------------------------------------------------------------

def _existing_repo_files(api: HfApi, repo_id: str) -> set[str]:
    try:
        return set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    except Exception:
        return set()

def push_dataset(
    name: str,
    *,
    staging_dir: Path,
    preproc_dir: Path,
    org: str = ORG,
    private: bool = False,
    dry_run: bool = False,
    shard_rows: int = 200,
    shard_images: int = 200,
    files_per_commit: int = 50,
    skip_existing: bool = False,
) -> None:
    """Push ``staging_dir`` to ``<org>/<name>``.

    Reads ``preproc_dir/repo.json`` for description / license / citation
    metadata used in the README.
    """
    repo_id = f"{org}/{name}"
    api = HfApi()

    # 1. recompute stats and render README from current on-disk state
    stats = collect_stats_from_disk(staging_dir)
    repo_meta = load_repo_json(preproc_dir)
    # ``repo.json`` is static per-dataset metadata; how THIS staging run drew the
    # train/validation carve is a property of the run, so the adapter wrote it to
    # ``<staging>/split.json`` (``SourceStaging._record_split_policy``). Fold it in
    # so it reaches the card, which is the only place a consumer of the published
    # dataset would look -- the rollout path bakes its equivalent sentence into
    # ``extra_notes`` directly, in ``lite.data.hf.stage``, and has no split.json.
    split_json = staging_dir / "split.json"
    if split_json.is_file():
        policy = json.loads(split_json.read_text())["split_policy"]
        repo_meta["extra_notes"] = f"{repo_meta['extra_notes']}\n\n{policy}".strip()
    readme = render_card(name=name, repo=repo_meta, stats=stats, org=org)
    (staging_dir / "README.md").write_text(readme)
    write_stats(staging_dir, stats)

    # 2. set up CA-store reader for image bytes resolution
    store = ImageStore(staging_dir / "images", rel_prefix=image_rel_prefix(name))

    if not dry_run:
        log.info("create_repo %s (exist_ok=True)", repo_id)
        create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)
    existing = _existing_repo_files(api, repo_id) if (skip_existing and not dry_run) else set()

    push_tmp = staging_dir / ".push_tmp"
    push_tmp.mkdir(parents=True, exist_ok=True)

    # Pending batch: (local_path, path_in_repo, unlink_after).
    batch: list[tuple[Path, str, bool]] = []
    commit_idx = [0]
    planned_paths: set[str] = set()

    def flush_batch() -> None:
        if not batch or dry_run:
            batch.clear()
            return
        commit_idx[0] += 1
        ops = [CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(lp)) for lp, rel, _ in batch]
        log.info("create_commit #%d: %d files", commit_idx[0], len(ops))
        _commit_with_retry(
            api,
            repo_id=repo_id,
            repo_type="dataset",
            operations=ops,
            commit_message=f"Upload {name} batch {commit_idx[0]}",
        )
        for lp, _, unlink_after in batch:
            if not unlink_after:
                continue
            try:
                lp.unlink()
            except FileNotFoundError:
                pass
        batch.clear()

    def enqueue(local_path: Path, path_in_repo: str, *, unlink_after: bool = True) -> None:
        planned_paths.add(path_in_repo)
        if skip_existing and path_in_repo in existing:
            log.info("skip (exists) %s", path_in_repo)
            if unlink_after:
                try:
                    local_path.unlink()
                except FileNotFoundError:
                    pass
            return
        batch.append((local_path, path_in_repo, unlink_after))
        if len(batch) >= files_per_commit:
            flush_batch()

    # 3. Retire the previous layout BEFORE adding anything. A repo published
    # under the old collapsed spelling stores shards at
    # ``<plat>/<tt>/<split>/shard-*`` (four components); this layout writes
    # ``<plat>/<tt>/<split>/<variant>/shard-*`` (five). The card globs every
    # depth, so leaving the old files in place while the new ones land makes
    # ``load_dataset`` return both copies for the whole push -- hours, on a
    # dataset with thousands of shards -- and a push that dies midway leaves it
    # that way. Deleting first costs a partial dataset during the window, which
    # is visibly incomplete rather than quietly doubled.
    if not dry_run:
        try:
            existing = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
        except Exception:
            existing = set()
        superseded = sorted(
            f for f in existing
            if len(f.split("/")) == 4
            and f.split("/")[-1].startswith("shard-")
            and f.endswith(".parquet")
        )
        for i in range(0, len(superseded), files_per_commit):
            chunk = superseded[i:i + files_per_commit]
            commit_idx[0] += 1
            log.info("retire previous layout, commit #%d: %d files",
                     commit_idx[0], len(chunk))
            _commit_with_retry(
                api,
                repo_id=repo_id,
                repo_type="dataset",
                operations=[CommitOperationDelete(path_in_repo=f) for f in chunk],
                commit_message=f"Retire {len(chunk)} file(s) from the previous layout",
            )

    # 4. parquet shards
    for platform, task_type, split, variant, parquet_path in iter_partitions(staging_dir):
        rows = list(iter_parquet_rows(parquet_path))
        if not rows:
            continue
        # Image-dedup: grounding/understanding cohorts are single-image-per-row
        # and many rows share the same screenshot. Fold same-image rows into one
        # (image embedded once) — a pure transport optimization the download
        # side reverses. ~12x smaller for SeeClick-style grounding.
        if should_fold(task_type):
            n_before = len(rows)
            rows = fold_rows(rows)
            log.info("fold %s/%s/%s%s: %d rows → %d (image-dedup %.1fx)",
                     platform, task_type, split, f"/{variant}" if variant else "",
                     n_before, len(rows), n_before / max(1, len(rows)))
        shards = _build_shards(rows, shard_rows=shard_rows, shard_images=shard_images)
        n_shards = len(shards)

        for si, chunk in enumerate(shards):
            out_path = partition_path(
                Path(""),
                platform=platform,
                task_type=task_type,
                split=split,
                variant=variant or task_type.split(".")[-1],
                shard_idx=si,
                shard_total=n_shards,
            )
            shard_in_repo = str(out_path)

            if skip_existing and shard_in_repo in existing:
                log.info("skip (exists) %s", shard_in_repo)
                planned_paths.add(shard_in_repo)
                continue

            log.info(
                "[%s] shard %d/%d rows=%d → %s",
                "dry" if dry_run else "stage",
                si + 1, n_shards, len(chunk), shard_in_repo,
            )
            if dry_run:
                planned_paths.add(shard_in_repo)
                continue

            ds = _rows_to_dataset(chunk, store)
            tmp_out = push_tmp / shard_in_repo
            tmp_out.parent.mkdir(parents=True, exist_ok=True)
            # HF dataset viewer wants 100-300 MB uncompressed per row group.
            # The datasets library targets 100 MB/group via its batch_size
            # heuristic, but the heuristic under-counts for bimodal image
            # bytes (Aguvis observed 409 MB/group at default), so pin
            # batch_size=50 explicitly. write_page_index=True bypasses the
            # row-group ceiling for trajectory rows.
            ds.to_parquet(str(tmp_out), batch_size=50, write_page_index=True)
            enqueue(tmp_out, shard_in_repo)

    # 5. README + stats.json LAST: the card states row counts, so publishing it
    # before the shards land would advertise numbers the repo does not yet hold.
    for fname in ("README.md", "stats.json"):
        fp = staging_dir / fname
        if fp.is_file():
            batch.append((fp, fname, False))
            planned_paths.add(fname)

    flush_batch()

    # 6. orphan cleanup — remove repo files that prior pushes wrote but
    # this layout no longer references.
    if not dry_run:
        try:
            current = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
        except Exception:
            current = set()
        protected = {".gitattributes"}
        orphans = sorted(current - planned_paths - protected)
        if orphans:
            for i in range(0, len(orphans), files_per_commit):
                chunk = orphans[i:i + files_per_commit]
                commit_idx[0] += 1
                ops = [CommitOperationDelete(path_in_repo=p) for p in chunk]
                log.info("orphan cleanup commit #%d: %d files", commit_idx[0], len(ops))
                _commit_with_retry(
                    api,
                    repo_id=repo_id,
                    repo_type="dataset",
                    operations=ops,
                    commit_message=f"Remove {len(ops)} stale file(s) from prior push",
                )

    if push_tmp.exists() and not dry_run:
        import shutil as _sh
        _sh.rmtree(push_tmp, ignore_errors=True)

    if dry_run:
        log.info(
            "dry-run complete: planned upload for https://huggingface.co/datasets/%s",
            repo_id,
        )
    else:
        log.info("done: https://huggingface.co/datasets/%s", repo_id)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="dataset repo name (e.g. ScaleCUA)")
    p.add_argument("--org", default=ORG, help=f"HF org (default: {ORG})")
    p.add_argument("--staging", type=Path, default=None,
                   help=f"staging dir; defaults to $CUA_LITE_DATASETS_ROOT/{ORG}/<name>")
    p.add_argument("--preproc-dir", type=Path, default=None,
                   help="dataset's preproc directory (contains repo.json); "
                        "defaults to lite/data/preproc/<lower(name)>/ "
                        "with hyphens/underscores normalized")
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--shard-rows", type=int, default=200,
                   help="max rows per pushed parquet shard")
    p.add_argument("--shard-images", type=int, default=200,
                   help="max embedded images per shard — keeps each parquet "
                        "under ~300 MB so HF viewer doesn't reject them")
    p.add_argument("--files-per-commit", type=int, default=50,
                   help="batch this many files per HF commit "
                        "(HF rate limit: 128 commits/hour)")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip files already present in the repo; default re-uploads stable shard paths")
    p.add_argument("--tag", nargs="?", const="", default="",
                   help="HF git tag on the pushed commit for versioning. DEFAULT = current "
                        "cua-lite git short commit id (so the dataset revision is pinnable to "
                        "the producing code). Pass --tag <name> to override; --tag NONE to skip.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    staging = (args.staging or dataset_root(args.name)).resolve()
    if not staging.is_dir():
        log.error("staging dir missing: %s", staging)
        return 2

    preproc = args.preproc_dir
    if preproc is None:
        # Default lookup: lite/data/preproc/<name>/, allowing hyphens to
        # match underscored dirs (GUI-360 → gui360).
        candidates = [
            args.name.lower(),
            args.name.lower().replace("-", "_"),
            args.name.lower().replace("-", ""),
        ]
        here = Path(__file__).parent.parent / "preproc"
        for c in candidates:
            if (here / c / "repo.json").is_file():
                preproc = here / c
                break
        # Fallback: a self-describing staging dir (e.g. rollout-staged datasets
        # via lite.data.hf.stage write their own repo.json — no preproc dir).
        if preproc is None and (staging / "repo.json").is_file():
            preproc = staging
        if preproc is None:
            log.error(
                "could not auto-locate preproc dir for %s; pass --preproc-dir "
                "(or stage with lite.data.hf.stage, which writes repo.json)",
                args.name,
            )
            return 2
    preproc = preproc.resolve()
    if not (preproc / "repo.json").is_file():
        log.error("repo.json missing under %s", preproc)
        return 2

    push_dataset(
        args.name,
        staging_dir=staging,
        preproc_dir=preproc,
        org=args.org,
        private=args.private,
        dry_run=args.dry_run,
        shard_rows=args.shard_rows,
        shard_images=args.shard_images,
        files_per_commit=args.files_per_commit,
        skip_existing=args.skip_existing,
    )

    # Version tag on the pushed commit. Default = current cua-lite git short
    # commit id, so the dataset revision is pinnable to the code that produced
    # it (download --revision <tag>). Override with --tag <name>; --tag NONE skips.
    if not args.dry_run and args.tag.upper() != "NONE":
        tag = args.tag
        if tag == "":
            from lite.utils.git import run_git
            # Default HF tag = clean current short HEAD. This is not the
            # rollout provenance helper (which can append ``-dirty``), and raw
            # preproc rows do not necessarily carry ``others.commit``; freeze
            # the producing tree before publishing.
            tag = run_git("rev-parse --short HEAD")
            if not tag:
                log.warning("could not resolve git commit for default tag; skipping tag")
        if tag:
            repo_id = f"{args.org}/{args.name}"
            api = HfApi()
            # A re-publish under the same tag name is the NORMAL case (the default
            # tag is the git short HEAD, so publishing twice from one commit reuses
            # it). ``exist_ok=True`` would leave the tag on the superseded revision
            # while still logging success, which silently breaks the one thing the
            # tag is for: pinning THIS data. Move it instead.
            try:
                api.create_tag(repo_id, tag=tag, repo_type="dataset")
                log.info("created HF tag '%s' on %s", tag, repo_id)
            except HfHubHTTPError as exc:
                if exc.response.status_code != 409:
                    raise
                # 409 == the tag exists. The Hub has no move, so this is
                # delete-then-create, and the window between them is the one way
                # this can end with NO tag: say so with the superseded revision in
                # hand, because that is what an operator needs to put it back.
                superseded = api.dataset_info(repo_id, revision=tag).sha
                api.delete_tag(repo_id, tag=tag, repo_type="dataset")
                try:
                    api.create_tag(repo_id, tag=tag, repo_type="dataset")
                except Exception as recreate_failed:
                    raise RuntimeError(
                        f"tag {tag!r} on {repo_id} was deleted but could not be "
                        f"recreated, so the repo now has NO {tag!r}; the data is "
                        f"pushed and reachable on main. It previously pointed at "
                        f"{superseded}."
                    ) from recreate_failed
                log.info(
                    "moved HF tag '%s' on %s off %s onto the new commit",
                    tag, repo_id, superseded[:12],
                )
    return 0

if __name__ == "__main__":
    sys.exit(main())
