"""Browser SFT transfer filter for the WebGym -> WebVoyager no-goto cell.

This is experiment-local on purpose. The generic WebGym data cleaner keeps its
published filtering contract; this script makes the narrower projection needed
by `/SFT.md`:

* drop trajectories that used any top-level `goto`;
* drop trajectories whose task instruction exactly matches the WebVoyager eval
  manifest, after removing WebGym's reset wrapper and normalizing whitespace;
* replace historical WebGym extra-tool metadata with the eval surface
  `back,response`;
* remove WebGym's collected `Initial website: ...` reset wrapper because the
  shared train/eval config intentionally omits `instruction_template`;
* copy row images beside the output parquet so `export_sft --image-root <out>`
  is self-contained.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from devs.data.utils import compact_row_images, rebase_images_for_output  # noqa: E402
from lite.core.tools.calls import tool_call_name  # noqa: E402
from lite.core.tools.extra_tools import LiteBrowserNavToolSet, LiteFinishToolSet  # noqa: E402
from lite.data.staging import (  # noqa: E402
    coerce_messages,
    coerce_meta,
    prepare_output_dir,
    write_partition,
)

_INITIAL_WEBSITE_MARKER = "\n\nInitial website:"
_ANSWER_PROMPT_MARKER = "\n\nWhen you have found"
_DEFAULT_WEBVOYAGER_TASK_MANIFEST = (
    _REPO_ROOT / "lite/gym/envs/webharbor/webvoyager/data/tasks.json"
)


def _input_parquets(src_root: Path) -> list[Path]:
    if src_root.is_file():
        if src_root.suffix != ".parquet":
            raise SystemExit(f"input file must be a parquet: {src_root}")
        return [src_root]
    parquets = sorted(src_root.rglob("*.parquet"))
    if not parquets:
        raise SystemExit(f"no parquet files under {src_root}")
    return parquets


def _has_top_level_goto(messages: list[dict[str, Any]]) -> bool:
    return any(
        tool_call_name(tool_call) == "goto"
        for message in messages
        if message.get("role") == "assistant"
        for tool_call in message.get("tool_calls") or []
    )


def _strip_task_boilerplate(text: str) -> str:
    for marker in (_INITIAL_WEBSITE_MARKER, _ANSWER_PROMPT_MARKER):
        marker_idx = text.lower().find(marker.lower())
        if marker_idx >= 0:
            text = text[:marker_idx]
    return text.rstrip()


def _normalize_task_instruction(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _first_user_instruction(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            return ""
        text = "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
        return _normalize_task_instruction(_strip_task_boilerplate(text))
    return ""


def _load_webvoyager_instruction_blacklist(path: Path) -> set[str]:
    if not path.is_file():
        raise SystemExit(f"WebVoyager task manifest not found: {path}")
    with path.open(encoding="utf-8") as f:
        tasks = json.load(f)
    instructions = {
        _normalize_task_instruction(task["instruction"])
        for task in tasks.values()
        if isinstance(task, dict) and isinstance(task.get("instruction"), str)
    }
    if not instructions:
        raise SystemExit(f"WebVoyager task manifest has no instructions: {path}")
    return instructions


def _strip_webgym_instruction_template(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    out: list[dict[str, Any]] = []
    stripped = False
    saw_first_user = False
    for message in messages:
        if saw_first_user or message.get("role") != "user":
            out.append(message)
            continue
        saw_first_user = True
        content = message.get("content")
        if not isinstance(content, list):
            out.append(message)
            continue
        parts: list[Any] = []
        changed = False
        for part in content:
            if not changed and isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    stripped_text = _strip_task_boilerplate(text)
                    if stripped_text != text:
                        part = {**part, "text": stripped_text}
                        changed = True
                        stripped = True
            parts.append(part)
        out.append({**message, "content": parts} if changed else message)
    return out, stripped


def _nogoto_extra_tool_schemas() -> list[dict[str, Any]]:
    back = LiteBrowserNavToolSet.get_tool_schema("back")
    response = LiteFinishToolSet.get_tool_schema("response")
    if back is None or response is None:
        raise RuntimeError("missing required no-goto browser extra tool schema")
    return [back, response]


def _project_metadata(metadata: Any) -> dict[str, Any]:
    out = dict(coerce_meta(metadata))
    out["extra_tool_schemas"] = _nogoto_extra_tool_schemas()
    return out


def _process_file(
    src: Path,
    dst: Path,
    output_root: Path,
    webvoyager_instruction_blacklist: set[str],
) -> tuple[int, int, int, int, bool]:
    df = pd.read_parquet(src)
    kept_indices: list[int] = []
    messages_out: list[list[dict[str, Any]]] = []
    metadata_out: list[dict[str, Any]] = []
    dropped_goto = 0
    dropped_webvoyager_instruction = 0
    stripped_instruction = 0

    for idx, row in df.iterrows():
        messages = coerce_messages(row["messages"])
        if _has_top_level_goto(messages):
            dropped_goto += 1
            continue
        metadata = _project_metadata(row["metadata"])
        if _first_user_instruction(messages) in webvoyager_instruction_blacklist:
            dropped_webvoyager_instruction += 1
            continue
        messages, stripped = _strip_webgym_instruction_template(messages)
        stripped_instruction += int(stripped)
        kept_indices.append(idx)
        messages_out.append(messages)
        metadata_out.append(metadata)

    if not kept_indices:
        return len(df), dropped_goto, dropped_webvoyager_instruction, stripped_instruction, False

    out = df.iloc[kept_indices].copy()
    out["messages"] = messages_out
    out["metadata"] = metadata_out
    if "images" in out.columns:
        compacted_images, compacted_messages = [], []
        for row_images, row_messages in zip(out["images"], out["messages"], strict=True):
            images, messages = compact_row_images(row_images, row_messages)
            compacted_images.append(images)
            compacted_messages.append(messages)
        out["images"] = compacted_images
        out["messages"] = compacted_messages
        out["images"] = rebase_images_for_output(
            out,
            source_parquet=src,
            output_parquet=dst,
            image_path_root=output_root,
        )
    write_partition(out.to_dict("records"), dst)
    return len(df), dropped_goto, dropped_webvoyager_instruction, stripped_instruction, True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="HF-downloaded WebGym root or parquet")
    parser.add_argument("--out", required=True, help="output root for the no-goto transfer dataset")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing non-empty output root",
    )
    parser.add_argument(
        "--webvoyager-task-manifest",
        default=str(_DEFAULT_WEBVOYAGER_TASK_MANIFEST),
        help="WebHarbor WebVoyager tasks.json whose exact instructions are excluded",
    )
    args = parser.parse_args()

    src_root = Path(args.input)
    out_root = prepare_output_dir(
        args.out,
        overwrite=args.overwrite,
        label="browser SFT filter output root",
        protected_roots=(src_root,),
    )
    src_files = _input_parquets(src_root)
    webvoyager_instruction_blacklist = _load_webvoyager_instruction_blacklist(
        Path(args.webvoyager_task_manifest)
    )

    total_rows = total_goto = total_webvoyager_instruction = total_instruction = wrote_files = 0
    for src in src_files:
        rel = src.relative_to(src_root) if src_root.is_dir() else Path(src.name)
        rows, goto_rows, webvoyager_instruction_rows, instruction_rows, wrote = _process_file(
            src, out_root / rel, out_root, webvoyager_instruction_blacklist
        )
        total_rows += rows
        total_goto += goto_rows
        total_webvoyager_instruction += webvoyager_instruction_rows
        total_instruction += instruction_rows
        wrote_files += int(wrote)

    kept = total_rows - total_goto - total_webvoyager_instruction
    print(
        f"done: rows={total_rows} kept={kept} dropped_goto={total_goto} "
        f"stripped_instruction={total_instruction} "
        f"dropped_webvoyager_instruction={total_webvoyager_instruction} "
        f"webvoyager_blacklist={len(webvoyager_instruction_blacklist)} "
        f"wrote={wrote_files}/{len(src_files)} "
        f"parquet file(s) -> {out_root}"
    )


if __name__ == "__main__":
    main()
