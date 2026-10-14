"""Normalize a teacher's reasoning into the canonical ``reasoning_content`` field.

``reasoning_content`` is a first-class field on ``LiteAssistantMessage``, beside
``content`` and ``tool_calls``. ``inline_reasoning`` is a content PART, and it
records something narrower: that the reasoning arrived inline in the visible
text, because the teacher was prompted for a ``Thought:`` line. The GPT-5.5
teacher is prompted that way; a teacher sampled with ``enable_thinking`` writes
``reasoning_content`` directly. Same fact, two shapes — this moves the prompted
one to the field the other already uses.

This script rewrites a dataset (or rollout log-root) in place-to-``--out``:
per assistant turn it moves every ``inline_reasoning`` part's text into
``reasoning_content`` (joined by newline), drops those content parts, and pops
``raw_response`` because the saved provider payload no longer matches the
mutated message. The ``metadata`` column, the other message FIELD (``tool_calls``)
and every surviving content PART (``action_description``, ``text``, ``image``, ...) are
preserved untouched, and the ``messages`` encoding is preserved (a JSON-string
column stays a string; a list<struct> column stays a list).

``images`` keeps pointing at the SAME files, rewritten to absolute paths: the
output holds parquet only, so a relative ref would dangle against the new root
(see :func:`_absolutize_images`). The output root is a transient pre-stage
artifact on one machine, consumed by ``hf.stage`` right after.

Where this runs: the ``gpt5_5`` teacher's last step before ``hf.stage``, so the
PUBLISHED rows already carry reasoning in the canonical
:attr:`LiteAssistantMessage.reasoning_content` field. One vocabulary: a teacher
sampled with ``enable_thinking`` writes that field natively, and prompted
``Thought:`` teachers land there too instead of forcing every consumer to know
which shape a given config produced. For a teacher that never emits reasoning
(``qwen3_8_27b`` writes prose into ``action_description``) the MOVE finds nothing
and the pass reports ``0 assistant turns`` -- but it still writes the output tree:
``df.to_parquet(dst)`` in :func:`_process_file` runs whatever the move found.

Both export recipes then read one published root:
``*.reasoning.yaml`` (``enable_thinking: true``) renders the reasoning, and a
thinking-off config strips it at the model boundary (see
``lite/train/export/sft_tokenize.py``) — verified byte-identical to exporting
from a non-internalized root.

Run it on the annotated log-root, between the filter pass and ``hf.stage``:
    uv run python devs/data/internalize_cot.py \
        --in  ".data/rollout/<env>/gpt5_5/$COMMIT/<platform>/train_annotated" \
        --out ".data/rollout/<env>/gpt5_5/$COMMIT/<platform>/train_annotated.think"
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lite.data.staging import (
    coerce_image_paths,
    prepare_output_dir,
    resolve_artifact_path,
)


def _plain(obj: Any) -> Any:
    """numpy arrays → lists / scalars, recursively (so message dicts are mutable)."""
    if isinstance(obj, np.ndarray):
        return [_plain(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(x) for x in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def move_inline_to_reasoning(messages: list[dict]) -> int:
    """Mutate ``messages`` in place; return the number of assistant turns changed."""
    changed = 0
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        content = m.get("content") or []
        inline = [
            p["text"] for p in content
            if isinstance(p, dict) and p.get("type") == "inline_reasoning" and p.get("text")
        ]
        if not inline:
            continue
        text = "\n".join(inline)
        existing = m.get("reasoning_content") or ""
        m["reasoning_content"] = f"{existing}\n{text}".strip() if existing else text
        m["content"] = [
            p for p in content
            if not (isinstance(p, dict) and p.get("type") == "inline_reasoning")
        ]
        m.pop("raw_response", None)  # saved provider payload no longer matches the mutated message
        changed += 1
    return changed


def _absolutize_images(images: Any, *, anchor: Path) -> Any:
    """Repoint a row's image refs at the ORIGINAL files, as absolute paths.

    The output root holds parquet only — no per-trajectory ``images/`` dirs — so a
    RELATIVE ref copied verbatim would be resolved against the new root, find
    nothing, and fall back to ``cwd``. ``hf.stage`` only notices at image ingest,
    long after the row passed validation, and dies with ``CorruptImageError``.

    Resolving here, against the source parquet, is the same rule ``hf.stage``
    applies (:func:`lite.data.staging.resolve_artifact_path`), so the staged bytes
    are the ones the input root pointed at. Already-absolute refs pass through.
    """
    was_str = isinstance(images, str)
    resolved = [
        str(resolve_artifact_path(path, anchor_path=anchor)) for path in coerce_image_paths(images)
    ]
    return json.dumps(resolved) if was_str else resolved


def _process_file(src: Path, dst: Path) -> tuple[int, int]:
    df = pd.read_parquet(src)
    new_messages: list[Any] = []
    n_rows = n_turns = 0
    for _, row in df.iterrows():
        raw = row["messages"]
        was_str = isinstance(raw, str)
        msgs = json.loads(raw) if was_str else _plain(list(raw))
        n_turns += move_inline_to_reasoning(msgs)
        n_rows += 1
        new_messages.append(json.dumps(msgs) if was_str else msgs)
    df["messages"] = new_messages
    if "images" in df.columns:
        df["images"] = [_absolutize_images(v, anchor=src) for v in df["images"]]
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    return n_rows, n_turns


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--in", dest="src", required=True, help="input parquet file or directory")
    ap.add_argument("--out", dest="dst", required=True, help="output (mirrors input structure)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace a NON-EMPTY --out directory, or an existing --out file")
    args = ap.parse_args(argv)

    src = Path(args.src)
    dst = Path(args.dst)
    files = [src] if src.is_file() else sorted(src.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no *.parquet under {src}")
    if src.is_file():
        # Single-file mode has no directory to prepare, so guard the two ways it can
        # destroy data: `df.to_parquet` is not atomic, so writing onto the input loses it
        # on a mid-write crash, and an existing --out is otherwise clobbered in silence.
        # Same exception types as ``prepare_output_dir`` below, so file and directory
        # mode fail the same way for the same mistake.
        if dst.resolve() == src.resolve():
            raise ValueError(f"--out must differ from --in ({src}); this rewrites in place")
        if dst.exists() and not args.overwrite:
            raise FileExistsError(
                f"internalize output file {dst} already exists; pass --overwrite to replace it"
            )
    else:
        # A dirty root merges instead of replacing: re-running the upstream filter with
        # tighter flags drops trajectories, but their old .think/<task>/sample_NN/ dirs
        # survive here, and `stage` rglobs trajectory.parquet -- so the dropped rows get
        # published. Same fresh-output contract as filter.py / stage / unstage / download.
        prepare_output_dir(
            dst,
            overwrite=args.overwrite,
            label="internalize output directory",
            protected_roots=(src,),
        )

    tot_rows = tot_turns = 0
    for f in files:
        out = dst if src.is_file() else dst / f.relative_to(src)
        n_rows, n_turns = _process_file(f, out)
        tot_rows += n_rows
        tot_turns += n_turns
        gc.collect()
    print(
        f"done: {len(files)} parquet(s), {tot_rows} trajectories, "
        f"moved inline_reasoning → reasoning_content on {tot_turns} assistant turns → {dst}"
    )


if __name__ == "__main__":
    main()
