"""Tests for reading back what a published dataset already holds.

Run:
    uv run --extra dev pytest devs/data/tests/test_published_configs.py -q
"""

from __future__ import annotations

import pandas as pd
import pytest

from devs.data.published_configs import (
    _stage_args,
    derived_cohorts,
    published_partitions,
    unstage_dir,
)


def _dataset(tmp_path, partitions: list[str]):
    """A canonical dataset tree: <plat>/<tt>/<split>/<config>.parquet."""
    for rel in partitions:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"images": [], "messages": "[]", "metadata": "{}"}]).to_parquet(path)
    return tmp_path


def test_every_published_config_is_listed_with_its_split(tmp_path):
    dataset = _dataset(tmp_path, [
        "desktop/use/train/desktop.use.synth.gpt5_5.parquet",
        "desktop/use/train/desktop.use.synth.qwen3_8_27b.parquet",
        "desktop/use/validation/desktop.use.synth.gpt5_5.parquet",
    ])
    assert published_partitions(dataset) == [
        ("desktop.use.synth.gpt5_5", ("train", "validation")),
        ("desktop.use.synth.qwen3_8_27b", ("train",)),
    ]


def test_a_config_spanning_two_splits_is_one_root_not_two(tmp_path):
    """`unstage` selects parquet by config name and never by split, so one call already
    pulls both carves into one log-root. Emitting the config twice would unstage it twice
    (the second call dies on the non-empty dir) and, if the flags were used, stage every
    row twice under one label."""
    dataset = _dataset(tmp_path, [
        "desktop/use/train/desktop.use.rl.parquet",
        "desktop/use/validation/desktop.use.rl.parquet",
    ])
    assert published_partitions(dataset) == [("desktop.use.rl", ("train", "validation"))]
    assert _stage_args(published_partitions(dataset), "$REC").splitlines() == [
        '  --log-roots "$REC/desktop.use.rl/all" \\',
        "  --config-names desktop.use.rl \\",
    ]


def test_unstage_dir_names_the_single_split_but_not_a_wrong_one():
    assert unstage_dir(("train",)) == "train"
    assert unstage_dir(("validation",)) == "validation"
    # mislabelling the validation rows `train` would be silently wrong in the log-root
    assert unstage_dir(("train", "validation")) == "all"


def test_one_config_spanning_two_cohorts_is_listed_once(tmp_path):
    """A variant is not tied to one `<plat>/<tt>` tree; `unstage` reads them all into
    ONE log-root. Listing the pair twice makes the caller unstage it twice -- the second
    call dies on the non-empty split dir -- and, if the printed flags were used, stages
    the same rows twice."""
    dataset = _dataset(tmp_path, [
        "browser/use/train/rollout.parquet",
        "desktop/use/train/rollout.parquet",
    ])
    assert published_partitions(dataset) == [("rollout", ("train",))]


def test_a_derived_mode_repo_reports_the_cohorts_it_actually_publishes(tmp_path):
    """Staged without --config-names, the variant stem is the literal `rollout` and the
    published config names are the derived cohorts. Re-staging under `--config-names
    rollout` leaves the shards where they are but re-renders the card as one config
    called `rollout`, so the derived names stop being addressable."""
    dataset = _dataset(tmp_path, [
        "browser/use/train/rollout.parquet",
        "desktop/use/train/rollout.parquet",
    ])
    assert derived_cohorts(dataset) == ["browser.use", "desktop.use"]


def test_an_override_mode_repo_reports_no_derived_cohorts(tmp_path):
    dataset = _dataset(tmp_path, ["desktop/use/train/desktop.use.rl.gpt5_5.parquet"])
    assert derived_cohorts(dataset) == []


def test_a_mixed_root_claims_no_derived_cohorts(tmp_path):
    """One non-`rollout` variant proves --config-names was used, so "staged WITHOUT it"
    would be a false statement -- and the cohort names it would print are not config
    names at all."""
    dataset = _dataset(tmp_path, [
        "browser/use/train/rollout.parquet",
        "desktop/use/train/desktop.use.gpt5_5.parquet",
    ])
    assert derived_cohorts(dataset) == []


def test_the_image_store_is_not_a_partition(tmp_path):
    """Missing this would feed `stage` a config named after an image shard."""
    dataset = _dataset(tmp_path, ["desktop/use/train/desktop.use.rl.parquet"])
    stray = dataset / "images" / "aa" / "aa.parquet"
    stray.parent.mkdir(parents=True)
    pd.DataFrame([{"x": 1}]).to_parquet(stray)
    assert published_partitions(dataset) == [("desktop.use.rl", ("train",))]


def test_a_dataset_staged_without_config_names_reports_its_derived_variant(tmp_path):
    dataset = _dataset(tmp_path, ["browser/use/train/rollout.parquet"])
    assert published_partitions(dataset) == [("rollout", ("train",))]


def test_a_non_canonical_parquet_is_refused_rather_than_guessed(tmp_path):
    """Silently skipping it would drop a config from the re-stage, so upload
    would sweep its published shards away."""
    dataset = _dataset(tmp_path, ["desktop/use/train/desktop.use.rl.parquet"])
    partition = dataset / "desktop/use/train/desktop.use.rl.parquet"
    (dataset / "loose.parquet").write_bytes(partition.read_bytes())
    with pytest.raises(SystemExit, match="not a canonical partition"):
        published_partitions(dataset)


def test_an_empty_dataset_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="no partition parquet"):
        published_partitions(tmp_path)


def test_stage_args_pairs_each_rebuilt_root_with_its_own_label():
    """`stage` maps --log-roots to --config-names 1:1 and in order; a shifted
    pair republishes every config under a neighbour's name."""
    args = _stage_args(
        [("desktop.use.rl", ("train",)), ("desktop.use.eval", ("validation",))], "$REC"
    )
    assert args.splitlines() == [
        '  --log-roots "$REC/desktop.use.rl/train" \\',
        '              "$REC/desktop.use.eval/validation" \\',
        "  --config-names desktop.use.rl \\",
        "                 desktop.use.eval \\",
    ]
