"""Stage agent rollout trajectories into the canonical cua-lite dataset layout.

Turns one or more rollout log-roots (``scripts/rollout.py`` output) into the
SAME canonical layout that ``lite/data/preproc/<dataset>/`` adapters produce::

    ${CUA_LITE_DATASETS_ROOT}/cua-lite/<Name>/
      images/<hash[:2]>/<hash>.<ext>           # content-addressed image store
      <platform>/<task_type>/<split>/<variant>.parquet   # rows: images(rel), messages, metadata

so the published artifact is indistinguishable from a preproc dataset and the
existing tooling consumes it UNCHANGED:

    lite.data.hf.upload   (canonical → HF, embeds bytes + shards)
    lite.data.hf.download (HF → canonical)
    lite.train.export.export_sft (canonical → model-ready SFT parquet)

This staging step is a canonical producer: it emits the local dataset layout and
delegates publication to ``lite.data.hf.upload``. It does NOT filter, strip
no-ops, or drop footguns by default; omit ``--filter`` and every row is kept.
Do success/footgun/captcha filtering upstream in the data-collection cleaner,
then point ``--log-roots`` at the cleaner output. Direct raw ``export_sft``
remains the local same-family fast path. Staging normalizes private runtime
sidecars before publish, so canonical rows export the same way after local stage
or upload/download.

Image paths inside a rollout parquet are resolved from the trajectory path's
ancestors, then the project root and current working directory. Images are
ingested in row order; message ``{"type":"image","index":N}`` references are
never reindexed or renumbered.

**Row validation is the publication gate, and it is fatal for the whole run.**
Every row goes through ``validate_canonical_rows`` before it is buffered, including
tool/message shape, metadata, image references, tool-surface validity, pairing,
and normalized coordinate bounds. One malformed row aborts the run before any
partition parquet, ``stats.json``, or ``repo.json`` is written. The failed attempt
may have already created the output directory or stored images from earlier valid
rows; rerun into a fresh directory or pass ``--overwrite``.

What to do when the gate fires. The message names the failing
``trajectory.parquet`` and the message index, and the fault is a PRODUCER bug in
whatever wrote that row: re-collect, or drop the row upstream in the cleaner and
stage the cleaner's output. ``devs/migration/run.py`` repairs old published
dataset rows; it is not a repair path for rollout log-roots.

Run:
    uv run python -m lite.data.hf.stage \
        --log-roots .logs/rollout/gpt55_tier2 .logs/rollout/gpt55_tier3 \
        --name WebGymGPT
    # then: uv run python -m lite.data.hf.upload WebGymGPT --org cua-lite
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from lite.core.messages.final import (
    CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY,
    STOP_REASON_INFO_KEY,
)
from lite.core.metadata import LiteCUAMetadata, metadata_from_dict
from lite.core.samples import PERSISTED_FINAL_STOP_REASONS
from lite.core.utils.filters import parse_filter
from lite.data.hf.card import load_repo_json
from lite.data.staging import (
    CANONICAL_SPLITS,
    ImageStore,
    coerce_messages,
    coerce_meta,
    collect_stats,
    dataset_root,
    flush_buffers,
    hash_split,
    image_rel_prefix,
    prepare_output_dir,
    resolve_artifact_path,
    to_plain,
    write_stats,
)
from lite.data.utils.rows import validate_canonical_rows


# No default row predicate: stage is a general tool and stages EVERY row by
# default. Filtering (success/footgun/captcha) is done upstream in the
# data-collection cleaner (webgym: ``devs/data/webgym/filter.py``). ``--filter``
# (any ``lambda m: …`` on LiteBaseMetadata) remains available for ad-hoc staging.
def _apply_stop_reason_publication_policy(metadata: dict[str, Any]) -> None:
    """Drop ``others.stop_reason`` unless it is on the durable publication allowlist.

    Routine ``content_only_final``/``empty`` finals stay internal runtime labels;
    only reasons in :data:`PERSISTED_FINAL_STOP_REASONS` (e.g. ``parse_failure``)
    survive into published metadata.
    """
    others = metadata.setdefault("others", {})
    stop_reason = others.get(STOP_REASON_INFO_KEY)
    if not isinstance(stop_reason, str) or stop_reason not in PERSISTED_FINAL_STOP_REASONS:
        others.pop(STOP_REASON_INFO_KEY, None)


def stage(
    log_roots: list[Path],
    *,
    name: str,
    out_dir: Path,
    filter_expr: str | None,
    val_frac: float = 0.0,
    seed: int = 42,
    config_names: list[str] | None = None,
    repo_dir: Path | None = None,
    description: str = "",
    original_urls: list[str] | None = None,
    license: str | None = None,
    citation: str = "",
    overwrite: bool = False,
) -> None:
    # ``config_names`` (the --config-names override) maps 1:1 to ``log_roots``:
    # each root's rows are tagged with that label as the partition ``variant``,
    # so each label gets its own ``<plat>/<tt>/<split>/<label>.parquet`` for the
    # card's label-scoped glob, and the dataset card names HF configs VERBATIM
    # after the labels (repo.json ``config_name_override``). ``None`` ⇒ a single
    # ``"rollout"`` variant under the derived ``<plat>.<tt>`` cohort config —
    # what preproc / no-flag staging use.
    override = config_names is not None
    if override:
        assert len(config_names) == len(log_roots), "config_names must be 1:1 with log_roots"
    root_variants = config_names if override else ["rollout"] * len(log_roots)

    out_dir = prepare_output_dir(
        out_dir,
        overwrite=overwrite,
        label="stage output directory",
        protected_roots=tuple(Path(root) for root in log_roots),
    )
    store = ImageStore(out_dir / "images", rel_prefix=image_rel_prefix(name))
    filter_fn = parse_filter(filter_expr) if filter_expr else None
    buffers: dict[tuple, list[dict]] = {}
    n_seen = n_kept = n_dropped = n_noimg = 0

    traj_files: list[tuple[Path, str]] = []  # (trajectory.parquet, variant)
    for root, root_variant in zip(log_roots, root_variants):
        for tp in sorted(Path(root).rglob("trajectory.parquet")):
            traj_files.append((tp, root_variant))
    if not traj_files:
        raise SystemExit(f"no trajectory.parquet under {[str(r) for r in log_roots]}")
    print(f"staging {len(traj_files)} trajectories from {len(log_roots)} log-root(s) → {out_dir}")

    # Ingest a row's images through ``ImageStore.put_many``: each image is read three
    # times (PIL verify, sha256, copy), and there are hundreds of thousands.
    # ``ImageStore`` documents ``put`` as thread-safe and its writes as idempotent, and
    # the rel path it returns is a content hash, so concurrency cannot change what a
    # SUCCESSFUL row records: ``put_many`` preserves order and every append below stays
    # on this thread, so row order and parquet bytes match the serial version. The one
    # difference is the abort path — a row with a corrupt image submits its whole list
    # before raising, so more images reach the store than a serial run would have put
    # there. That is dead weight in a run that fails anyway, never a wrong row.
    for tp, variant in traj_files:
        tp = tp.resolve()
        df = pd.read_parquet(tp)
        for _, row in df.iterrows():
            n_seen += 1
            md = coerce_meta(row["metadata"])
            if "split" in md:
                raise ValueError(
                    "metadata.split must not be present; split lives in the partition path"
                )
            lite_meta = metadata_from_dict(md)
            if not isinstance(lite_meta, LiteCUAMetadata):
                raise ValueError(
                    f"{tp}: lite.data.hf.stage only supports CUA metadata; "
                    "generic publish layout is not supported by this plan"
                )
            md = lite_meta.to_dict()
            others = md["others"]
            if not isinstance(others, dict):
                raise ValueError("metadata.others must be a dict")
            task_id = others.get("task_id")
            if not task_id:
                raise ValueError(f"{tp}: metadata.others.task_id is required")
            task_id = str(task_id)
            if filter_fn is not None and not filter_fn(lite_meta):
                n_dropped += 1
                continue
            platform = lite_meta.platform.value
            task_type = lite_meta.task_type.value
            messages = coerce_messages(row["messages"])
            _apply_stop_reason_publication_policy(md)
            # Drop private runtime sidecars before publishing/distillation.
            for m in messages:
                if isinstance(m, dict):
                    m.pop("raw_response", None)
                    m.pop(CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY, None)

            imgs = to_plain(row["images"]) or []
            if not imgs:
                n_noimg += 1
            # Message ``{"type":"image","index":N}`` references are never reindexed.
            new_imgs = store.put_many(
                [resolve_artifact_path(p, anchor_path=tp) for p in imgs]
            )

            # ``others["split"]`` is a transient routing hint, never row content: a
            # log-root reconstructed by ``lite.data.hf.unstage`` records the partition
            # each row came from, so a re-stage reproduces the publisher's carve
            # instead of re-drawing it at THIS run's --val-frac. Consuming the hint
            # means POPPING it -- the split is encoded by the partition path, and
            # nothing downstream would catch the hint leaking into a published row.
            recorded_split = others.pop("split", None)
            if recorded_split is not None:
                if (not isinstance(recorded_split, str)
                        or recorded_split not in CANONICAL_SPLITS):
                    allowed = ", ".join(CANONICAL_SPLITS)
                    raise ValueError(
                        f"{tp}: metadata.others.split must be one of {{{allowed}}}; "
                        f"got {recorded_split!r}"
                    )
                split = recorded_split
            else:
                split = hash_split(task_id, val_frac=val_frac, seed=seed)

            out_row = {
                "images": new_imgs,
                "messages": messages,
                "metadata": md,
            }
            validate_canonical_rows([out_row], str(tp))
            buffers.setdefault((platform, task_type, split, variant), []).append(out_row)
            n_kept += 1
        del df

    flush_buffers(out_dir, buffers)
    stats = collect_stats(buffers, store)
    stats.rows_in = n_seen
    stats.rows_dropped = n_dropped
    write_stats(out_dir, stats)

    # Self-describing repo.json so `lite.data.hf.upload` can render the dataset
    # card WITHOUT a preproc dir (rollout-staged datasets have none). upload
    # falls back to <staging>/repo.json when --preproc-dir is absent.
    repo_json = out_dir / "repo.json"
    if not repo_json.exists():
        # Static per-dataset facts (upstream links, citation, license) are owned by the
        # route's checked-in repo.json -- the SAME file shape `lite/data/preproc/<dataset>/`
        # publishes, read through its owner. The run-derived half below is owned by this run.
        # Explicit flags win over the file so an ad-hoc stage can still override.
        meta = load_repo_json(repo_dir) if repo_dir else {}
        repo_payload = {
            "description": description or meta.get("description") or (
                f"{name}: agent rollout trajectories staged into the canonical "
                "cua-lite layout for SFT distillation."
            ),
            "original_urls": list(original_urls or meta.get("original_urls") or []),
            "license": license or meta.get("license") or "other",
            "citation": citation or meta.get("citation") or "",
            # No row carries a split marker, so hash_split's two inputs are the
            # only way a consumer can reproduce or audit the carve. Record them.
            "extra_notes": (
                "Staged via `lite.data.hf.stage` from rollout log-roots: "
                + ", ".join(str(r) for r in log_roots)
                + f" (row filter: {filter_expr or 'none'}; "
                + f"split: hash_split on task_id with val_frac={val_frac}, seed={seed})."
            ),
        }
        # Only stamp the override flag when --config-names was used, so a
        # default/preproc repo.json is byte-identical to before (card.render_card
        # reads repo.get("config_name_override", False)).
        if override:
            repo_payload["config_name_override"] = True
        repo_json.write_text(json.dumps(repo_payload, indent=2))

    print(
        f"done: seen={n_seen} kept={n_kept} dropped_by_filter={n_dropped} "
        f"(filter={filter_expr!r}; rows_with_no_image={n_noimg}) | "
        f"unique_images={stats.unique_images} store_bytes={stats.image_store_bytes}"
    )
    for (plat, tt, split, var), rows in sorted(buffers.items()):
        print(f"  {plat}/{tt}/{split} [{var}]: {len(rows)} rows")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--log-roots",
        nargs="+",
        required=True,
        type=Path,
        help="one or more rollout log-roots to absorb into a single dataset",
    )
    ap.add_argument("--name", required=True, help="dataset name → cua-lite/<name>")
    ap.add_argument("--out", type=Path, default=None,
                    help="output staging dir (default: $CUA_LITE_DATASETS_ROOT/cua-lite/<name>)")
    ap.add_argument("--filter", dest="filter_expr", default=None,
                    help="Optional Python lambda on Lite metadata to KEEP rows (same semantics as "
                         "export_sft --filter), e.g. "
                         "\"lambda m: (m.others.get('episode_return') or 0)>=1.0\". "
                         "DEFAULT: no filter — stage ALL rows. Do success/footgun "
                         "filtering upstream "
                         "in the cleaner (webgym: devs/data/webgym/filter.py).")
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="validation fraction (hash split on task_id). DEFAULT 0 = ALL train: "
                         "rollout-collected data is training data; evaluate on held-out ENV tasks "
                         "(scripts/rollout.py --splits eval), not on a carved-out parquet split. "
                         "Set e.g. 0.02 only if you really want an in-dataset validation slice.")
    ap.add_argument("--seed", type=int, default=42, help="split hash seed")
    cn = ap.add_mutually_exclusive_group()
    cn.add_argument("--config-names", nargs="+", default=None,
                    help="OVERRIDE the HF config_name(s), one per --log-roots (1:1, may "
                         "repeat to merge roots): the label becomes the verbatim HF "
                         "config_name (NOT the derived <platform>.<task_type>). e.g. "
                         "`--config-names synth perturb`. Pass a fully-dotted name "
                         "(`desktop.use.synth`) if you want that exact spelling.")
    cn.add_argument("--config-name", default=None,
                    help="OVERRIDE the HF config_name with a single label broadcast to all "
                         "--log-roots. Shorthand for `--config-names X X ...`.")
    ap.add_argument("--repo-dir", type=Path, default=None,
                    help="directory owning this dataset's `repo.json` (description, "
                         "original_urls, license, citation) — the same file `hf.upload "
                         "--preproc-dir` reads. Published rollout routes keep it at "
                         "`devs/data/<route>/`, so the route runbook and the migration runbook "
                         "cannot drift. The per-field flags below "
                         "(--description/--original-urls/--license/--citation) override it.")
    ap.add_argument("--description", default="", help="dataset card description (repo.json)")
    ap.add_argument("--original-urls", nargs="+", default=None,
                    help="upstream source URL(s) the rollouts derive from — the task/judge repo, "
                         "dataset, and paper (repo.json `original_urls`, rendered as the card's "
                         "`## Origin` list). e.g. `--original-urls "
                         "https://github.com/THUDM/SCALE-CUA https://arxiv.org/abs/2607.11185`")
    ap.add_argument("--license", default=None, help="dataset license (repo.json; default 'other')")
    ap.add_argument("--citation", default="", help="dataset citation (repo.json)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing non-empty output staging dir. Without this, stage "
                         "requires a fresh --out so stale partitions/images cannot "
                         "survive a rerun.")
    args = ap.parse_args()

    # Resolve per-log-root config_name override (None ⇒ today's cohort-derived naming).
    config_names: list[str] | None = None
    if args.config_names is not None:
        if len(args.config_names) != len(args.log_roots):
            ap.error(
                f"--config-names takes one label per --log-roots "
                f"({len(args.log_roots)} roots given, {len(args.config_names)} labels)"
            )
        config_names = list(args.config_names)
    elif args.config_name is not None:
        config_names = [args.config_name] * len(args.log_roots)

    out_dir = args.out or dataset_root(args.name)
    stage(
        args.log_roots,
        name=args.name,
        out_dir=Path(out_dir),
        filter_expr=args.filter_expr,
        val_frac=args.val_frac,
        seed=args.seed,
        config_names=config_names,
        repo_dir=args.repo_dir,
        description=args.description,
        original_urls=args.original_urls,
        license=args.license,
        citation=args.citation,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
