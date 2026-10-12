"""Tests for the gpt5_5 pre-stage pass that canonicalizes reasoning.

Run:
    uv run --extra dev pytest devs/data/tests/test_internalize_cot.py -q
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from devs.data import internalize_cot
from devs.data.internalize_cot import _process_file, move_inline_to_reasoning


def _assistant(*content: dict, **extra) -> dict:
    return {"role": "assistant", "content": list(content), **extra}


def test_inline_reasoning_moves_into_the_canonical_field():
    messages = [_assistant(
        {"type": "inline_reasoning", "text": "first"},
        {"type": "text", "text": "visible"},
        raw_response={"provider": "payload"},
    )]
    assert move_inline_to_reasoning(messages) == 1
    assert messages[0]["reasoning_content"] == "first"
    assert messages[0]["content"] == [{"type": "text", "text": "visible"}]
    # the saved payload describes the pre-move message, so it cannot survive it
    assert "raw_response" not in messages[0]


def test_several_inline_parts_join_and_an_existing_field_is_kept():
    messages = [_assistant(
        {"type": "inline_reasoning", "text": "b"},
        {"type": "inline_reasoning", "text": "c"},
        reasoning_content="a",
    )]
    assert move_inline_to_reasoning(messages) == 1
    assert messages[0]["reasoning_content"] == "a\nb\nc"


def test_a_teacher_with_no_inline_reasoning_is_a_no_op():
    """`qwen3_8_27b` runs thinking off; running this on its rows must change nothing."""
    messages = [_assistant({"type": "text", "text": "visible"}, raw_response={"keep": 1})]
    before = json.dumps(messages, sort_keys=True)
    assert move_inline_to_reasoning(messages) == 0
    assert json.dumps(messages, sort_keys=True) == before


def _write_root(root: Path, images: list[str], *, as_json_string: bool) -> Path:
    """One trajectory parquet, laid out the way `filter.py` writes its output."""
    src = root / "task_a" / "sample_00" / "trajectory.parquet"
    src.parent.mkdir(parents=True)
    for rel in images:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (2, 2)).save(path)
    messages = [_assistant({"type": "inline_reasoning", "text": "why"})]
    pd.DataFrame([{
        "images": images,
        "messages": json.dumps(messages) if as_json_string else messages,
        "metadata": json.dumps({"task_id": "task_a"}),
    }]).to_parquet(src, index=False)
    return src


def test_relative_image_refs_are_repointed_at_the_original_files(tmp_path):
    """The output root holds parquet only, so a copied relative ref would dangle.

    It dangles silently: nothing in the row is wrong, and `hf.stage` only trips
    over it at image ingest, well after the row passed canonical validation.
    """
    rel = ["task_a/sample_00/images/000000.png"]
    src = _write_root(tmp_path / "annotated", rel, as_json_string=False)
    _process_file(src, tmp_path / "annotated.think" / "task_a" / "sample_00" / "trajectory.parquet")

    out = pd.read_parquet(tmp_path / "annotated.think/task_a/sample_00/trajectory.parquet")
    written = list(out["images"][0])
    assert written == [str(tmp_path / "annotated" / rel[0])]
    assert Path(written[0]).exists()


def test_absolute_image_refs_pass_through(tmp_path):
    absolute = [str(tmp_path / "elsewhere" / "shot.png")]
    Path(absolute[0]).parent.mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(absolute[0])
    src = _write_root(tmp_path / "annotated", [], as_json_string=False)
    pd.DataFrame([{
        "images": absolute,
        "messages": [_assistant({"type": "inline_reasoning", "text": "why"})],
        "metadata": json.dumps({"task_id": "task_a"}),
    }]).to_parquet(src, index=False)

    dst = tmp_path / "out.parquet"
    _process_file(src, dst)
    assert list(pd.read_parquet(dst)["images"][0]) == absolute


@pytest.mark.parametrize("as_json_string", [True, False])
def test_the_messages_encoding_survives_the_round_trip(tmp_path, as_json_string):
    """A JSON-string column stays a string; a list<struct> column stays a list."""
    src = _write_root(
        tmp_path / "annotated",
        ["task_a/sample_00/images/000000.png"],
        as_json_string=as_json_string,
    )
    dst = tmp_path / "out.parquet"
    assert _process_file(src, dst) == (1, 1)

    value = pd.read_parquet(dst)["messages"][0]
    assert isinstance(value, str) is as_json_string
    messages = json.loads(value) if as_json_string else value
    assert messages[0]["reasoning_content"] == "why"


# --- main(): the guard layer, previously untested end to end ---------------------

def _run(argv: list[str]) -> Exception | None:
    """Invoke the CLI the way an operator does; return the exception it refused with."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            internalize_cot.main(argv)
        except (SystemExit, OSError, ValueError) as exc:
            return exc
    return None


def test_main_refuses_a_dirty_output_directory_without_overwrite(tmp_path):
    """Every sibling in the chain (filter.py / stage / unstage / download) refuses one.

    A dirty root MERGES: re-running the upstream filter with tighter flags drops
    trajectories, but their old `.think/<task>/sample_NN/` dirs survive, and `stage`
    rglobs `trajectory.parquet` — so the dropped rows would get published.
    """
    src = tmp_path / "in"
    _write_root(src, ["task_a/sample_00/images/000000.png"], as_json_string=False)
    out = tmp_path / "out"
    assert _run(["--in", str(src), "--out", str(out)]) is None
    assert isinstance(_run(["--in", str(src), "--out", str(out)]), FileExistsError)
    assert _run(["--in", str(src), "--out", str(out), "--overwrite"]) is None


def test_main_refuses_to_rewrite_a_single_file_in_place(tmp_path):
    """`df.to_parquet` is not atomic, so writing onto the input loses it on a crash."""
    src = tmp_path / "in"
    parquet = _write_root(src, [], as_json_string=False)
    assert isinstance(_run(["--in", str(parquet), "--out", str(parquet)]), ValueError)


def test_main_refuses_an_existing_output_file_without_overwrite(tmp_path):
    """File mode has no directory to prepare, so it needs its own clobber guard."""
    src = tmp_path / "in"
    parquet = _write_root(src, [], as_json_string=False)
    dst = tmp_path / "out.parquet"
    dst.write_bytes(parquet.read_bytes())
    assert isinstance(_run(["--in", str(parquet), "--out", str(dst)]), FileExistsError)
    assert _run(["--in", str(parquet), "--out", str(dst), "--overwrite"]) is None
