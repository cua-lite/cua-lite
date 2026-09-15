"""Tests for the browser WebGym -> WebVoyager SFT transfer filter."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.calls import make_tool_call, tool_call_name
from lite.core.tools.extra_tools import LiteBrowserNavToolSet, LiteFinishToolSet
from lite.data.staging import coerce_image_paths, coerce_messages, coerce_meta, write_partition

_spec = importlib.util.spec_from_file_location(
    "browser_sft_filter",
    Path(__file__).resolve().parents[1] / "utils" / "filter.py",
)
assert _spec is not None and _spec.loader is not None
browser_filter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(browser_filter)


def _write_webvoyager_manifest(path: Path, *instructions: str) -> None:
    path.write_text(
        json.dumps(
            {
                f"site.{i}": {"instruction": instruction}
                for i, instruction in enumerate(instructions)
            }
        )
    )


def _text(text: str) -> dict:
    return {"type": "text", "text": text}


def _user(*parts: dict) -> dict:
    return {"role": "user", "content": list(parts)}


def _asst(*calls: dict) -> dict:
    return {"role": "assistant", "tool_calls": list(calls)}


def _call(name: str, **args) -> dict:
    return make_tool_call(name, args, call_id=f"call_{name}")


def test_browser_filter_projects_webgym_rows_to_webvoyager_nogoto_surface(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    src_root = tmp_path / "downloaded" / "cua-lite" / "WebGym"
    parquet = src_root / "browser" / "use" / "train" / "browser.use.gpt5_5.parquet"
    image = src_root / "images" / "00" / "000000.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (1, 1), color=(1, 2, 3)).save(image)

    source_surface = [
        LiteFinishToolSet.get_tool_schema("response"),
        LiteBrowserNavToolSet.get_tool_schema("goto"),
    ]
    rows = [
        {
            "images": ["cua-lite/WebGym/images/00/000000.png"],
            "messages": [
                _user(
                    {"type": "image", "index": 0},
                    _text(
                        "Find the price.\n\n"
                        "Initial website: https://example.com\n\n"
                        "When you have found the answer, submit it with the response action."
                    ),
                ),
                _asst(_call("response", text="$10")),
            ],
            "metadata": LiteCUAMetadata(
                dims=("browser", "use"),
                extra_tool_schemas=source_surface,
                others={"episode_return": 1.0},
            ).to_dict(),
        },
        {
            "images": ["cua-lite/WebGym/images/00/000000.png"],
            "messages": [
                _user(_text("Find the price.")),
                _asst(_call("goto", url="https://example.com")),
            ],
            "metadata": LiteCUAMetadata(
                dims=("browser", "use"),
                extra_tool_schemas=source_surface,
                others={"episode_return": 1.0},
            ).to_dict(),
        },
    ]
    write_partition(rows, parquet)

    out_root = tmp_path / "nogoto"
    manifest = tmp_path / "webvoyager_tasks.json"
    _write_webvoyager_manifest(manifest, "Unrelated WebVoyager task.")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "filter.py",
            "--input",
            str(src_root),
            "--out",
            str(out_root),
            "--webvoyager-task-manifest",
            str(manifest),
            "--overwrite",
        ],
    )
    browser_filter.main()

    captured = capsys.readouterr()
    assert "rows=2 kept=1 dropped_goto=1 stripped_instruction=1" in captured.out
    out = out_root / "browser" / "use" / "train" / "browser.use.gpt5_5.parquet"
    df = pd.read_parquet(out)
    assert len(df) == 1
    row = df.iloc[0]
    assert coerce_image_paths(row["images"]) == ["browser/use/train/images/000000.png"]
    assert (out.parent / "images" / "000000.png").read_bytes() == image.read_bytes()

    messages = coerce_messages(row["messages"])
    first_text = messages[0]["content"][1]["text"]
    assert first_text == "Find the price."
    assert not any(
        tool_call_name(tool_call) == "goto"
        for message in messages if message.get("role") == "assistant"
        for tool_call in message.get("tool_calls") or []
    )
    names = [
        schema["function"]["name"]
        for schema in coerce_meta(row["metadata"]).get("extra_tool_schemas") or []
    ]
    assert names == ["back", "response"]


def test_browser_filter_drops_webvoyager_instruction_matches(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    src_root = tmp_path / "downloaded" / "cua-lite" / "WebGym"
    parquet = src_root / "browser" / "use" / "train" / "browser.use.gpt5_5.parquet"
    image = src_root / "images" / "00" / "000000.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (1, 1), color=(1, 2, 3)).save(image)

    leaked_instruction = "Browse for a compact air fryer on Amazon."
    write_partition([
        {
            "images": ["cua-lite/WebGym/images/00/000000.png"],
            "messages": [
                _user(
                    {"type": "image", "index": 0},
                    _text(
                        f"{leaked_instruction}\n\n"
                        "Initial website: https://amazon.com\n\n"
                        "When you have found the answer, submit it with the response action."
                    ),
                ),
                _asst(_call("response", text="Air fryer")),
            ],
            "metadata": LiteCUAMetadata(
                dims=("browser", "use"),
                extra_tool_schemas=[LiteFinishToolSet.get_tool_schema("response")],
                others={"episode_return": 1.0},
            ).to_dict(),
        },
        {
            "images": ["cua-lite/WebGym/images/00/000000.png"],
            "messages": [
                _user({"type": "image", "index": 0}, _text("Find the price.")),
                _asst(_call("response", text="$10")),
            ],
            "metadata": LiteCUAMetadata(
                dims=("browser", "use"),
                extra_tool_schemas=[LiteFinishToolSet.get_tool_schema("response")],
                others={"episode_return": 1.0},
            ).to_dict(),
        },
    ], parquet)

    out_root = tmp_path / "nogoto"
    manifest = tmp_path / "webvoyager_tasks.json"
    _write_webvoyager_manifest(manifest, "Browse   for a compact air fryer on Amazon.")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "filter.py",
            "--input",
            str(src_root),
            "--out",
            str(out_root),
            "--webvoyager-task-manifest",
            str(manifest),
            "--overwrite",
        ],
    )
    browser_filter.main()

    captured = capsys.readouterr()
    assert "rows=2 kept=1 dropped_goto=0" in captured.out
    assert "dropped_webvoyager_instruction=1" in captured.out
    out = out_root / "browser" / "use" / "train" / "browser.use.gpt5_5.parquet"
    df = pd.read_parquet(out)
    assert len(df) == 1
    assert coerce_messages(df.iloc[0]["messages"])[0]["content"][1]["text"] == "Find the price."


def test_browser_filter_compacts_referenced_images_before_rebase(
    tmp_path,
    monkeypatch,
) -> None:
    src_root = tmp_path / "downloaded" / "cua-lite" / "WebGym"
    parquet = src_root / "browser" / "use" / "train" / "browser.use.gpt5_5.parquet"
    image_0 = src_root / "images" / "00" / "000000.png"
    image_1 = src_root / "images" / "00" / "000001.png"
    image_0.parent.mkdir(parents=True)
    Image.new("RGB", (1, 1), color=(1, 2, 3)).save(image_0)
    Image.new("RGB", (1, 1), color=(4, 5, 6)).save(image_1)

    write_partition([
        {
            "images": [
                "cua-lite/WebGym/images/00/000000.png",
                "cua-lite/WebGym/images/00/000001.png",
            ],
            "messages": [
                _user({"type": "image", "index": 1}, _text("Find the price.")),
                _asst(_call("response", text="$10")),
            ],
            "metadata": LiteCUAMetadata(
                dims=("browser", "use"),
                extra_tool_schemas=[LiteFinishToolSet.get_tool_schema("response")],
                others={"episode_return": 1.0},
            ).to_dict(),
        },
    ], parquet)

    out_root = tmp_path / "nogoto"
    manifest = tmp_path / "webvoyager_tasks.json"
    _write_webvoyager_manifest(manifest, "Unrelated WebVoyager task.")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "filter.py",
            "--input",
            str(src_root),
            "--out",
            str(out_root),
            "--webvoyager-task-manifest",
            str(manifest),
            "--overwrite",
        ],
    )
    browser_filter.main()

    out = out_root / "browser" / "use" / "train" / "browser.use.gpt5_5.parquet"
    row = pd.read_parquet(out).iloc[0]
    messages = coerce_messages(row["messages"])
    assert messages[0]["content"][0]["index"] == 0
    assert coerce_image_paths(row["images"]) == ["browser/use/train/images/000000.png"]
    assert (out.parent / "images" / "000000.png").read_bytes() == image_1.read_bytes()
