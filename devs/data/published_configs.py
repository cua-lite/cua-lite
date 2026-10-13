"""List what a published dataset ALREADY holds, so a re-stage can list all of it.

``lite.data.hf.upload`` sweeps: ``orphans = current - planned``. A stage that
omits a published config therefore DELETES its shards and drops it from the card,
and ``--dry-run`` does not report the orphan set. So the input to "add a config"
is not what you remember publishing — it is what the repo actually has.

Point this at a root that ``lite.data.hf.download`` wrote (it merges HF shards
back into the canonical ``<plat>/<tt>/<split>/<config>.parquet`` layout) and it
prints one ``config<TAB>dir`` line per published config -- the second column is the
DIRECTORY to unstage into, not a split: a config spanning train+validation prints
``all``, because one ``unstage`` call pulls both::

    uv run python -m lite.data.hf.download Lite.OSWorld --org "$HF_ORG" \
      --out "${READBACK_ROOT}/cua-lite/Lite.OSWorld"
    uv run python devs/data/published_configs.py "${READBACK_ROOT}/cua-lite/Lite.OSWorld"

``--stage-args REC`` instead prints the ``--log-roots`` / ``--config-names`` pair
that re-stages every one of them from roots ``unstage`` rebuilt under ``REC``,
which is the part that is easy to get wrong by hand.

A dataset staged WITHOUT ``--config-names`` carries no variant labels, so the listing
reports the on-disk ``rollout`` variant. That is the STEM, NOT a config name: the card
derives ``<platform>.<task_type>`` config names from it. Passing ``rollout`` back to
``stage --config-names`` keeps the shard paths identical (so the orphan sweep deletes
nothing) but re-renders the card as a single config literally called ``rollout``, and
the derived cohort names stop being addressable by ``load_dataset(repo, name)``
(:func:`lite.data.hf.card._build_configs_override`). The tool warns on stderr for the
shape it can detect; pick the real labels deliberately.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lite.data.staging import split_of_partition_file

#: ``stage`` writes this fixed variant when no ``--config-names`` was given
#: (``lite/data/hf/stage.py``), so seeing it means the repo publishes DERIVED
#: ``<platform>.<task_type>`` config names rather than variant labels.
DERIVED_VARIANT = "rollout"


def published_partitions(dataset: Path) -> list[tuple[str, tuple[str, ...]]]:
    """``(variant, splits)`` for the partition parquet under *dataset*, one row per variant.

    The variant is the parquet stem — ``stage`` writes each ``--config-names`` label
    as its own ``<plat>/<tt>/<split>/<label>.parquet``.

    One row per VARIANT, not per file and not per split, because that is the unit
    ``unstage`` works in: it selects parquet by config name alone and never by split
    (``lite/data/hf/unstage.py``), so one call already pulls every cohort and every
    split of a variant into one log-root. Emitting a variant twice makes the caller
    unstage it twice — the second call dies on the non-empty split dir — and, if the
    printed flags were used as-is, stages the same rows twice under one label.
    """
    splits: dict[str, set[str]] = {}
    for path in sorted(dataset.rglob("*.parquet")):
        rel = path.relative_to(dataset)
        if "images" in rel.parts:  # the CA image store holds no parquet; scope to the
            continue                # dataset, or a root under any dir named `images` skips all
        split = split_of_partition_file(rel)
        if split is None:
            raise SystemExit(f"{path} is not a canonical partition under {dataset}")
        splits.setdefault(path.stem, set()).add(split)
    if not splits:
        raise SystemExit(f"no partition parquet under {dataset}")
    return [(variant, tuple(sorted(s))) for variant, s in sorted(splits.items())]


def unstage_dir(splits: tuple[str, ...]) -> str:
    """The one directory name ``unstage --splits`` should write this variant into.

    ``--splits`` names a directory, not a filter: every row of the variant lands in
    it whatever its own carve, and ``stage`` re-reads that carve per row from
    ``others["split"]``. So a variant spanning both splits needs ONE call, and the
    name only has to be stable — ``all`` says so out loud rather than mislabelling
    the validation rows ``train``.
    """
    return splits[0] if len(splits) == 1 else "all"


def derived_cohorts(dataset: Path) -> list[str]:
    """The ``<platform>.<task_type>`` config names a DERIVED-mode repo publishes.

    Detected by every variant being the literal ``rollout`` that ``stage`` writes
    when no ``--config-names`` was given. That is sufficient but NOT necessary: a
    ``lite/data/preproc`` dataset is also derived-mode yet names its variants
    ``use``/``point``/``bbox``, and nothing on disk distinguishes those from override
    labels — ``download`` does not copy ``repo.json``, which holds the authoritative
    ``config_name_override``. So an empty result is not proof of override mode; the
    dataset card is. A mixed root returns empty too: one non-``rollout`` variant
    proves ``--config-names`` was used.
    """
    variants = {variant for variant, _ in published_partitions(dataset)}
    if variants != {DERIVED_VARIANT}:
        return []
    cohorts = set()
    for path in sorted(dataset.rglob(f"{DERIVED_VARIANT}.parquet")):
        rel = path.relative_to(dataset).parts
        if "images" in rel:
            continue
        if len(rel) >= 3:
            cohorts.add(f"{rel[0]}.{rel[1]}")
    return sorted(cohorts)


def _stage_args(partitions: list[tuple[str, tuple[str, ...]]], rec: str) -> str:
    """The two 1:1 flag lists that re-stage every published config from ``REC``.

    ``unstage --log-root REC/<config> --splits <split>`` writes ``REC/<config>/<split>``,
    so that is the root each config re-enters ``stage`` under.
    """
    roots = [f'"{rec}/{c}/{unstage_dir(sp)}"' for c, sp in partitions]
    names = [c for c, _ in partitions]
    lines = [f"  --log-roots {roots[0]} \\"]
    lines += [f"              {root} \\" for root in roots[1:]]
    lines += [f"  --config-names {names[0]} \\"]
    lines += [f"                 {name} \\" for name in names[1:]]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("dataset", type=Path, help="a downloaded canonical dataset root")
    ap.add_argument("--stage-args", metavar="REC",
                    help="print --log-roots/--config-names re-staging from roots under REC")
    args = ap.parse_args()

    dataset = args.dataset.resolve()
    partitions = published_partitions(dataset)
    cohorts = derived_cohorts(dataset)
    if cohorts:
        print(
            f"WARNING: {dataset} was staged WITHOUT --config-names, so what it\n"
            f"  publishes is the derived cohort config(s): {', '.join(cohorts)}\n"
            f"  '{DERIVED_VARIANT}' below is the on-disk variant, NOT a config label.\n"
            f"  Passing it to stage --config-names re-renders the card as one config\n"
            f"  called '{DERIVED_VARIANT}'; the shards stay put, but the derived names\n"
            f"  stop being addressable. Choose the real labels deliberately, and say\n"
            f"  so in the runbook.",
            file=sys.stderr,
        )
    if args.stage_args:
        print(
            "NOTE: these roots hold published rows as-is. Rows published BEFORE a\n"
            "  teacher's post-collection step existed still carry the pre-step shape -- for\n"
            "  gpt5_5 that is `inline_reasoning`, so run its `## Internalize Reasoning` on the\n"
            "  rebuilt root and stage the `.think` sibling instead of the root below. The pass\n"
            "  is idempotent, so it costs nothing when the rows are already canonical.",
            file=sys.stderr,
        )
        print(_stage_args(partitions, args.stage_args.rstrip("/")))
    else:
        for config, sp in partitions:
            print(f"{config}\t{unstage_dir(sp)}")


if __name__ == "__main__":
    main()
