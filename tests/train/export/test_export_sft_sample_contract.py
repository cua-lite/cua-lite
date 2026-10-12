"""``--sample N`` is a contract on the OUTPUT parquet, not on the converter input.

Rows can disappear *after* they are sampled: under ``--no-strict`` a row whose
actions the student cannot render, or whose image is unreadable, is logged and
dropped. Capping before conversion and converting once would therefore
under-deliver by a dataset-dependent amount -- the asymmetry a paired
teacher-comparison must not inherit. These tests pin the guarantee.
"""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

import lite.train.export.export_sft as export_sft
from lite.core import LiteCUAMetadata


class _TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [1] * len(text.split())


class _TinyProcessor:
    image_token = "<|image_pad|>"
    tokenizer = _TinyTokenizer()

    def apply_chat_template(self, messages, *, tokenize=False,
                            add_generation_prompt=False, enable_thinking=False):
        del tokenize, enable_thinking
        chunks: list[str] = []
        for message in messages:
            chunks.append(f"<{message.get('role')}>")
            for part in message.get("content") or []:
                if part.get("type") == "image":
                    chunks.append("<|image_pad|>")
                elif part.get("type") == "text" and part.get("text"):
                    chunks.append(part["text"])
        if add_generation_prompt:
            chunks.append("<assistant>")
        return "\n".join(chunks)


def _write_pool(tmp_path, n_rows: int):
    """A pool of ``n_rows`` canonical rows; ``others.idx`` identifies each."""
    img_path = tmp_path / "frame0.png"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(img_path)
    rows = []
    for i in range(n_rows):
        md = LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={"env_id": "osworld", "idx": i, "split": "train"},
        ).to_dict()
        rows.append({
            "images": [str(img_path)],
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": f"task {i}"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
            ],
            "metadata": json.dumps(md),
        })
    table = pa.Table.from_pylist(rows)
    # ``--data-paths`` takes DIRECTORIES; the parent dir names the canonical split.
    pool_dir = tmp_path / "pool" / "train"
    pool_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, pool_dir / "data.parquet")
    return pool_dir.parent


def _install_adapter(monkeypatch, *, fail_predicate):
    """Adapter whose ``unroll`` raises for rows matching ``fail_predicate``."""
    def fake_get(adapter_key, **kwargs):
        del adapter_key
        meta = kwargs.get("metadata")

        def unroll(_sample):
            if fail_predicate(meta.others.get("idx")):
                raise ValueError("cannot render canonical tool 'key_down'")
            return SimpleNamespace(processed_images=[], steps=[[
                {"role": "user", "content": [{"type": "text", "text": "obs"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "act"}]},
            ]])

        return SimpleNamespace(enable_thinking=False, unroll=unroll)

    monkeypatch.setattr(export_sft, "register_all", lambda: None)
    monkeypatch.setattr(export_sft.AgentAdapterRegistry, "get", staticmethod(fake_get))
    monkeypatch.setattr(export_sft, "_get_processor", lambda _model_id: _TinyProcessor())


def _run_main(monkeypatch, pool, out, *, sample, extra=()):
    argv = [
        "export_sft",
        "--agent-id", "qwen3_vl",
        "--model-id", "local-model",
        "--data-paths", str(pool),
        "--num-proc", "1",
        "--sample", str(sample),
        "--seed", "42",
        "-o", str(out),
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    export_sft.main()


def test_sample_is_honoured_on_the_output_when_rows_drop(tmp_path, monkeypatch):
    """Every 3rd row fails to convert; --sample 10 must still write 10 rows."""
    pool = _write_pool(tmp_path, 60)
    _install_adapter(monkeypatch, fail_predicate=lambda i: i % 3 == 0)
    out = tmp_path / "out.parquet"
    _run_main(monkeypatch, pool, out, sample=10, extra=("--no-strict",))

    assert pq.read_table(out).num_rows == 10


def test_sample_tops_up_across_several_rounds(tmp_path, monkeypatch):
    """A harsh drop rate needs more than one top-up; the quota still lands."""
    pool = _write_pool(tmp_path, 200)
    _install_adapter(monkeypatch, fail_predicate=lambda i: i % 5 != 0)  # 80% drop
    out = tmp_path / "out.parquet"
    _run_main(monkeypatch, pool, out, sample=20, extra=("--no-strict",))

    assert pq.read_table(out).num_rows == 20


def test_sample_falls_short_only_when_the_pool_runs_out(tmp_path, monkeypatch):
    """No silent padding: an exhausted pool yields what it can, not N."""
    pool = _write_pool(tmp_path, 30)
    _install_adapter(monkeypatch, fail_predicate=lambda i: i >= 12)  # only 12 usable
    out = tmp_path / "out.parquet"
    _run_main(monkeypatch, pool, out, sample=25, extra=("--no-strict",))

    assert pq.read_table(out).num_rows == 12


def test_strict_mode_still_converts_exactly_the_sample_once(tmp_path, monkeypatch):
    """With nothing dropping, the quota is met in round 1 (old behaviour)."""
    pool = _write_pool(tmp_path, 40)
    _install_adapter(monkeypatch, fail_predicate=lambda i: False)
    out = tmp_path / "out.parquet"
    _run_main(monkeypatch, pool, out, sample=15)

    assert pq.read_table(out).num_rows == 15


def test_sample_larger_than_pool_exports_everything(tmp_path, monkeypatch):
    pool = _write_pool(tmp_path, 7)
    _install_adapter(monkeypatch, fail_predicate=lambda i: False)
    out = tmp_path / "out.parquet"
    _run_main(monkeypatch, pool, out, sample=99)

    assert pq.read_table(out).num_rows == 7
