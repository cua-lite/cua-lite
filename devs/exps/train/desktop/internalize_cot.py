"""Move prompted ``inline_reasoning`` into the native ``reasoning_content`` channel.

The GPT-5.5 teacher stores its Thought as an ``inline_reasoning`` content part
(prompted pseudo-reasoning). To distill into Qwen3.5's NATIVE ``<think>`` channel
(``enable_thinking: true``, no custom Thought:/Action: system prompt), the same
reasoning must live in the assistant message's top-level ``reasoning_content``
field — which the Qwen3.5 chat template renders as ``<think>...</think>``.

This script rewrites a dataset (or rollout log-root) in place-to-``--out``:
per assistant turn it moves every ``inline_reasoning`` part's text into
``reasoning_content`` (joined by newline), drops those content parts, and pops
``raw_response`` because the saved provider payload no longer matches the
mutated message. All
other columns (``images``, ``metadata``) and message fields (``action_description``,
``tool_calls``) are preserved untouched. The ``messages`` encoding is preserved
(a JSON-string column stays a string; a list<struct> column stays a list).

Run (on a downloaded desktop.use dataset, then export with enable_thinking):
    uv run python devs/exps/train/desktop/internalize_cot.py \
        --in  "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.ScaleCUA" \
        --out "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.ScaleCUA.think"
    # then: export_sft --config .../desktop.use.default.reasoning.yaml (enable_thinking: true)
    #                  --data-paths "${CUA_LITE_DATASETS_ROOT}/cua-lite/Lite.ScaleCUA.think" ...
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


def _process_file(src: Path, dst: Path) -> tuple[int, int]:
    df = pd.read_parquet(src)
    new_messages: list[Any] = []
    n_msgs = n_turns = 0
    for _, row in df.iterrows():
        raw = row["messages"]
        was_str = isinstance(raw, str)
        msgs = json.loads(raw) if was_str else _plain(list(raw))
        n_turns += move_inline_to_reasoning(msgs)
        n_msgs += 1
        new_messages.append(json.dumps(msgs) if was_str else msgs)
    df = df.copy()
    df["messages"] = new_messages
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    return n_msgs, n_turns


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--in", dest="src", required=True, help="input parquet file or directory")
    ap.add_argument("--out", dest="dst", required=True, help="output (mirrors input structure)")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    files = [src] if src.is_file() else sorted(src.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no *.parquet under {src}")

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
