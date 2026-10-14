"""CLI: export CUA-Lite datasets to model-ready SFT parquet.

Loads images from disk, runs ``adapter.unroll(sample)`` to produce a
trajectory-level :class:`AgentSample`, **tokenizes each step** (prompt/response
boundary via the model's chat template), and writes parquet with one row per
trajectory. Images are PNG-encoded once per trajectory.

Tokenizing here (not at train time) means the exported ``steps`` are real
:class:`LiteRLStep` records: ``rollout_sft`` reads them straight into the same
``build_segment_samples`` path GRPO uses — no re-render, so the
``enable_thinking`` generation-prefix is frozen at export (read off
``adapter.enable_thinking``) and can't drift from how the data was rendered.
This is why ``--model-id`` is required (it loads the processor for the chat
template + tokenizer).

Input: canonical/preproc parquets or raw rollout log roots (same
``images``/``messages``/``metadata`` schema). Pass ``--image-root`` whenever rows
contain relative image paths; raw rollout exports usually use the repo root
(``--image-root .``), while canonical HF layouts use the dataset root. A canonical
layout also carries its train/validation carve in the partition path, so only
``--splits`` (default ``train``) is exported — pointing ``--data-paths`` at a
dataset root does NOT sweep the held-out shards into training data.

Output parquet schema:
    processed_images : large_list[large_binary?]
                                      — PNG bytes for model-visible images;
                                        ``None`` placeholders for stored
                                        screenshots that no step references
    steps            : list[struct]   — list of serialized LiteRLStep (see
                                        lite.train.export.sft_tokenize.serialize_rl_step):
                                        {prompt, image_indices, response,
                                         response_tokens, reward, status,
                                         prompt_tokens}. ``image_indices`` is an
                                        ordered per-step view into
                                        ``processed_images``: the Nth index
                                        supplies the Nth processor-owned image
                                        slot in ``prompt``. ``status`` is
                                        ``completed`` per step, except the last
                                        step of a row whose
                                        ``metadata.others.truncated`` records a
                                        truncated episode.
    metadata         : string         — JSON Lite metadata provenance from the
                                        input row; not consumed by the SFT
                                        training harness

Failed rows (corrupt image, missing file, adapter error) fail fast by default;
pass --no-strict to log and drop them and continue.

Usage:
    uv run python -m lite.train.export.export_sft \
      --config scripts/configs/qwen3_vl/recipes/sft/default.yaml \
      --model-id Qwen/Qwen3-VL-8B-Instruct \
      --data-paths "${CUA_LITE_DATASETS_ROOT}/cua-lite/ScaleCUA" \
      --image-root "${CUA_LITE_DATASETS_ROOT}" \
      --head 10 -o /tmp/test.parquet
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
from pathlib import Path

from PIL import Image

from lite.agents.bootstrap import register_all
from lite.agents.core.adapter import (
    AgentAdapterRegistry,
    tool_surface_agent_kwarg_names,
)
from lite.agents.core.agent.utils.loop import mark_steps_truncated
from lite.core.metadata import LiteBaseMetadata, LiteGenericMetadata, metadata_from_dict
from lite.core.samples import LiteSample
from lite.core.utils.filters import parse_filter
from lite.data.load import discover_files_under_paths, load_file_as_dataset
from lite.data.staging import coerce_image_paths, coerce_messages, coerce_meta
from lite.train.export.sft_tokenize import agent_step_to_rl_step, serialize_rl_step
from lite.utils.config import load_config
from lite.utils.image import load_images
from lite.utils.registry import compose_key

logger = logging.getLogger(__name__)

# Generation-only knobs consumed by rollout/local serving. Offline SFT export
# re-renders saved trajectories and must not forward them as adapter kwargs.
_EXPORT_IGNORED_AGENT_KWARGS = frozenset({"sampling_kwargs"})

# Per-worker processor cache. ``Dataset.map(num_proc=N)`` forks N workers; each
# lazy-loads the processor once (keyed by model_id) rather than re-loading per row.
_PROCESSOR_CACHE: dict[str, object] = {}


def _get_processor(model_id: str):
    proc = _PROCESSOR_CACHE.get(model_id)
    if proc is None:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        _PROCESSOR_CACHE[model_id] = proc
    return proc


def _pil_to_bytes(img: Image.Image | None) -> bytes | None:
    if img is None:
        return None
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _adapter_kwargs_for_export(agent_kwargs: dict | None) -> dict:
    """Return adapter construction kwargs for offline export.

    ``sampling_kwargs`` configures generation in rollout/local serving; export
    has no generation step, so keep that rollout-only key out of adapter
    construction while preserving every other key for adapter fail-loud checks.
    """
    akw = dict(agent_kwargs or {})
    for key in _EXPORT_IGNORED_AGENT_KWARGS:
        akw.pop(key, None)
    return akw


def _output_metadata_json(metadata: LiteBaseMetadata) -> str:
    return json.dumps(metadata.to_dict())


def _metadata_parse_error_json() -> str:
    return _output_metadata_json(
        LiteGenericMetadata(
            dims=("export_error",),
            others={"reason": "metadata_parse_failed"},
        )
    )


def _convert_sample(
    raw: dict, *, agent_id: str, agent_kwargs: dict, model_id: str,
    image_root: str | None, strict: bool,
) -> dict:
    """Convert one raw parquet row into one ``AgentSample`` row.

    1. Load images from paths → PIL.
    2. ``adapter.unroll(sample)`` → :class:`AgentSample` (trajectory shape).
    3. Tokenize each AgentStep → :class:`LiteRLStep` (generic RL step contract:
       rendered ``prompt`` + ordered ``image_indices`` + ``response`` /
       ``response_tokens``; ``enable_thinking`` read off the adapter). A step
       with no assistant target is a partial turn with nothing to supervise and
       is dropped; a row left with no step at all fails.
    4. Serialize PIL images to PNG bytes + LiteRLSteps to structs for parquet.

    Always returns the same key set ``{_error, processed_images, steps,
    metadata}`` (``_error`` is ``""`` on success) so HF ``Dataset.map`` infers a
    stable schema: it locks the schema to the first row and never unions keys, so
    a key present on only some rows would either silently vanish or crash the
    Arrow writer with ``KeyError``. ``strict=True`` re-raises the original error
    immediately (fail fast); ``strict=False`` returns the sentinel for the caller
    to filter out.
    """
    output_metadata: str | None = None
    try:
        raw_images = coerce_image_paths(raw.get("images", []))
        messages = coerce_messages(raw["messages"])
        metadata = coerce_meta(raw.get("metadata"))
        lite_metadata = metadata_from_dict(metadata)
        output_metadata = _output_metadata_json(lite_metadata)
        adapter_key = compose_key(agent_id, *lite_metadata.dims)
        images = load_images(raw_images, image_root=image_root)
        akw = _adapter_kwargs_for_export(agent_kwargs)
        surface_overrides = tool_surface_agent_kwarg_names(akw)
        if surface_overrides:
            raise TypeError(
                f"tool-surface settings provided via agent_kwargs: {sorted(surface_overrides)}; "
                "pass resolved surface via Lite task metadata"
            )

        # Env surface (extra_tool_schemas / valid_actions / others) is fixed at
        # ROLLOUT time and saved in the parquet. Replay it verbatim by forwarding
        # the WHOLE metadata object — exactly as rollout's make does
        # (lite/agents/factory.py) — so the rendered SFT prompt matches the
        # surface the data was generated under (same-source). The SFT config only
        # contributes agent-side RENDERING choices via agent_kwargs (resolution,
        # protocol_kwargs, system_prompt); it must NOT override the env surface.
        #
        # These surface fields belong to Lite task metadata, not adapter kwargs. Keep
        # them on metadata so nav extra tools and valid-action trimming replay
        # exactly as they did during rollout.
        akw["metadata"] = lite_metadata
        register_all()
        adapter = AgentAdapterRegistry.get(adapter_key, **akw)

        sample = LiteSample.from_dict(
            {"images": images, "messages": messages, "metadata": metadata}
        )
        agent_sample = adapter.unroll(sample)

        # Tokenize each AgentStep into a LiteRLStep here (offline) so the train
        # path never re-renders. ``enable_thinking`` comes from the adapter so
        # the chat-template generation-prefix matches how the data was rendered.
        processor = _get_processor(model_id)
        enable_thinking = bool(getattr(adapter, "enable_thinking", False))
        rl_steps = []
        for step in agent_sample.steps:
            rl_step = agent_step_to_rl_step(step, processor, enable_thinking)
            if rl_step is None:
                # A partial turn: no assistant target, so nothing to supervise.
                # This is ordinary rollout output, not corruption — a terminal
                # turn persists its tool feedback AFTER the assistant action
                # (lite/agents/core/agent/base.py), so a saved trajectory usually
                # ends on a ``role:"tool"`` observation. Drop it and keep every
                # supervisable step.
                continue
            rl_steps.append(rl_step)
        if not rl_steps:
            # Nothing to train on. Route it through this function's row-failure
            # contract (--strict raises, --no-strict logs and drops) rather than
            # writing an image-carrying row whose only effect downstream is the
            # zero-gradient dummy branch in lite/train/rollout/sft.py.
            raise ValueError(
                "adapter.unroll() produced no supervisable SFT step: the trajectory "
                "has no assistant target"
            )

        # Trajectory outcome. Each tokenized step is COMPLETED on its own — the
        # row holds no per-turn finish_reason — but a truncated episode truncates
        # its LAST step. Replay the row's own env feedback
        # (``metadata.others.truncated``, written by the trajectory logger) so
        # the segmenter's max-severity status is the real one, through the same
        # projection the online loops apply. Offline demonstration rows carry no
        # env feedback and stay COMPLETED.
        if lite_metadata.others.get("truncated") is True:
            mark_steps_truncated(rl_steps)

        return {
            "_error": "",
            "processed_images": [_pil_to_bytes(img) for img in agent_sample.processed_images],
            "steps": [serialize_rl_step(s) for s in rl_steps],
            "metadata": output_metadata,
        }
    except Exception as e:
        if strict:
            raise  # fail fast — propagate the original error out of .map()
        logger.warning("Skipping row: %s", e)
        if output_metadata is None:
            output_metadata = _metadata_parse_error_json()
        return {
            "_error": str(e),
            "processed_images": [],
            "steps": [],
            "metadata": output_metadata,
        }


def _output_features(metadata_feature=None):
    """Return the explicit Arrow contract for model-ready SFT rows."""
    from datasets import Features, LargeList, List, Value

    return Features({
        "_error": Value("string"),
        "processed_images": LargeList(Value("large_binary")),
        "steps": List({
            "prompt": Value("string"),
            "image_indices": List(Value("int64")),
            "response": Value("string"),
            "response_tokens": List(Value("int64")),
            "reward": Value("float64"),
            "status": Value("string"),
            "prompt_tokens": List(Value("int64")),
        }),
        "metadata": Value("string"),
    })


def _write_model_ready_parquet(dataset, output: Path, *, row_group_size,
                               writer_batch_size) -> None:
    """Write model-ready rows straight from the Arrow table.

    Preserves the explicit large-list schema and avoids materializing the
    multi-gigabyte image payload back into Python just to re-infer it. A lazy
    selection (``filter``/``select`` leave an indices mapping that
    ``dataset.data.table`` ignores) is flattened first — otherwise the dropped
    rows would be written back into the parquet.
    """
    import pyarrow.parquet as pq

    if dataset._indices is not None:
        dataset = dataset.flatten_indices(
            writer_batch_size=writer_batch_size, features=dataset.features)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        dataset.data.table,
        output,
        compression="snappy",
        row_group_size=row_group_size,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export CUA-Lite datasets to model-ready SFT parquet (v2 schema)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None,
                        help="YAML config with agent_id, agent_kwargs, data_paths. Prefer the "
                             "rollout config for the env the data came from "
                             "(scripts/configs/<agent>/default/<env>.yaml); env_kwargs is "
                             "ignored (the env surface is replayed from each row's metadata)")
    parser.add_argument("--agent-id", default=None,
                        help="Agent ID (e.g. qwen3_vl, cua-lite) (overrides config)")
    parser.add_argument("--model-id", default=None,
                        help="HF model id / path for the processor (chat template + "
                             "tokenizer) used to tokenize each step. Required — steps are "
                             "exported as tokenized LiteRLStep records. Must match the model "
                             "you'll TRAIN. Reads from the host HF cache (same as rollout); "
                             "no download needed if cached.")
    parser.add_argument("--data-paths", nargs="+", default=None,
                        help="Absolute paths to parquet directories (overrides config)")
    parser.add_argument("--splits", nargs="+", default=["train"],
                        help="Canonical dataset splits to export. A canonical dataset carves "
                             "train/validation into the partition PATH, so pointing "
                             "--data-paths at the dataset root would otherwise train on the "
                             "held-out shards with nothing in the row to notice it. Files "
                             "that name no canonical split (raw rollout trajectory.parquet) "
                             "are unaffected. Pass 'train validation' to export the whole set.")
    parser.add_argument("--image-root", default=None,
                        help="Prepended to relative image paths, e.g. repo root for raw "
                             "rollouts or dataset root for canonical HF layouts")
    parser.add_argument("--head", type=int, default=None,
                        help="Keep first N rows after all input parquets are pooled")
    parser.add_argument("--sample", type=int, default=None,
                        help="Write N rows (seeded by --seed). Rows dropped by conversion are "
                             "topped up from the same shuffled order, so N is a guarantee on "
                             "the OUTPUT, not on what was fed to the converter. Fewer only "
                             "when the pool runs out. Re-running a previous export with the "
                             "same seed therefore yields a superset-prefix of its rows, not "
                             "the identical set.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global RNG seed for --sample (default 42 for reproducibility).")
    parser.add_argument("--filter", default=None, dest="filter_expr",
                        help="Python lambda on Lite task metadata to filter rows. "
                             "Same syntax as rollout --filter. "
                             "E.g. \"lambda m: (m.others.get('episode_return') or 0) >= 1.0\" "
                             "or \"lambda m: not m.others.get('exclude_reason')\"")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True,
                        help="Fail fast on the first row that fails to convert (corrupt image, "
                             "missing file, adapter error). With --no-strict, such rows are "
                             "logged and dropped and conversion continues. Default: --strict.")
    parser.add_argument("--row-group-size", type=int, default=None,
                        help="Parquet row group size (rows per group); None for default.")
    parser.add_argument("--num-proc", type=int, default=16,
                        help="Process pool size for parallel row conversion via "
                             "Dataset.map(num_proc=N). Default 16 (image decode + "
                             "smart_resize are CPU-bound; ~3-8× faster than "
                             "sequential on typical rollout exports). Set to 1 "
                             "for deterministic sequential behavior.")
    parser.add_argument("--map-writer-batch-size", type=int, default=32,
                        help="Rows per Dataset.map Arrow writer batch. Keep this "
                             "small to bound worker memory for image-heavy exports.")
    parser.add_argument("-o", "--output", required=True, help="Output parquet path")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    agent_id = args.agent_id or cfg.get("agent_id")
    model_id = args.model_id  # CLI-only: the model/checkpoint isn't part of the agent recipe
    paths = args.data_paths or [os.path.expandvars(p) for p in cfg.get("data_paths", [])]
    agent_kwargs = _adapter_kwargs_for_export(cfg.get("agent_kwargs", {}))
    image_root = args.image_root

    if not agent_id:
        parser.error("--agent-id is required (or set agent_id in config)")
    if not model_id:
        parser.error("--model-id is required — steps are tokenized at export")
    if not paths:
        parser.error("--data-paths is required (or set data_paths in config)")
    if args.sample is not None and args.sample < 1:
        parser.error("--sample must be >= 1; a zero-row export has no output to write")

    files = discover_files_under_paths(paths, splits=args.splits)
    if not files:
        print(f"No data found for agent_id={agent_id}, paths={paths}, splits={args.splits}")
        return

    filter_fn = parse_filter(args.filter_expr) if args.filter_expr else None
    map_fn_kwargs = dict(
        agent_id=agent_id, agent_kwargs=agent_kwargs, model_id=model_id,
        image_root=image_root, strict=args.strict,
    )

    import pyarrow as pa
    from datasets import Dataset, concatenate_datasets

    # === STEP 1: Pool all files into one Dataset ===
    # ``num_proc`` only parallelizes across rows of a single ``.map`` /
    # ``.filter`` call. Per-file rollout trajectory.parquets often have
    # 1 row each (so per-file map can't parallelize), AND inferred Arrow
    # schemas can differ across files (metadata struct shapes vary).
    #
    # We stream per-file Arrow tables and concatenate them on the Arrow side
    # with ``pa.concat_tables(..., promote_options="permissive")``, which
    # reconciles the per-file schema drift (a field missing in one file becomes
    # nulls there) and adds missing columns. The Arrow path keeps memory bounded
    # by avoiding a Python-dict materialization of every row.
    tables: list[pa.Table] = []
    for file_path, rel_path in files:
        ds = load_file_as_dataset(file_path)
        if len(ds) == 0:
            continue
        # Combine chunks so each file contributes one clean InMemoryTable —
        # concat_tables then sees plain pa.Table schemas to promote.
        tables.append(ds.data.table.combine_chunks())
        print(f"  loaded {rel_path}: {len(ds)} rows")
    if not tables:
        print(f"No data found for agent_id={agent_id}, paths={paths}")
        return

    combined_table = pa.concat_tables(tables, promote_options="permissive")
    combined_in = Dataset(combined_table)
    del tables, combined_table  # free intermediate per-file tables

    # === STEP 2: Filter (single .filter call across all rows) ===
    n_filtered = 0
    if filter_fn is not None:
        before = len(combined_in)
        def _filter_row(row: dict) -> bool:
            return filter_fn(metadata_from_dict(coerce_meta(row.get("metadata"))))

        combined_in = combined_in.filter(
            _filter_row,
            num_proc=args.num_proc if len(combined_in) >= args.num_proc else 1,
            desc="filter rows",
        )
        n_filtered = before - len(combined_in)
        print(f"Filter ({args.filter_expr!r}): {n_filtered} rows excluded")

    # === STEP 3: --head cap before mapping ===
    if args.head is not None and len(combined_in) > args.head:
        combined_in = combined_in.select(range(args.head))

    if len(combined_in) == 0:
        print(f"No rows remain after filter for agent_id={agent_id}")
        return

    # === STEP 3b/4: convert until --sample rows SURVIVE conversion ===
    #
    # ``--sample N`` is a contract on the OUTPUT. Under ``--no-strict`` a row
    # can be dropped after it is sampled, so sampling N up front would
    # under-deliver by a teacher-dependent amount -- an asymmetry a paired
    # comparison must not inherit. Shuffle once (``--seed``), then walk a cursor
    # over that fixed order in rounds, converting only as far as needed. Under
    # ``--strict`` a failure raises, so round 1 always fills the quota and this
    # degenerates to the single ``.map`` it replaces.
    shuffled = combined_in
    if args.sample is not None and len(combined_in) > args.sample:
        shuffled = combined_in.shuffle(seed=args.seed)

    target = args.sample if args.sample is not None else len(shuffled)
    target = min(target, len(shuffled))

    converted_parts = []
    n_kept = 0
    n_dropped = 0
    cursor = 0
    round_no = 0
    while n_kept < target and cursor < len(shuffled):
        round_no += 1
        need = target - n_kept
        if round_no == 1:
            take = need
        else:
            # Size the top-up from the observed survival rate, with a margin so
            # a third round is rare. A round that kept nothing falls back to
            # ``need`` so the cursor always advances.
            attempted = cursor
            rate = n_kept / attempted if attempted else 0.0
            take = int(need / rate * 1.1) + 1 if rate > 0 else need
        take = min(take, len(shuffled) - cursor)

        chunk = shuffled.select(range(cursor, cursor + take))
        cursor += take
        if round_no == 1:
            print(f"Converting {take} rows with num_proc={args.num_proc}...")
        else:
            print(f"Top-up round {round_no}: converting {take} more rows "
                  f"({n_kept}/{target} kept so far)")

        # Declare 64-bit image offsets at the first Arrow writer boundary. The
        # smaller writer batch is then only a worker-memory bound, not a
        # correctness workaround for list<binary>'s 2 GiB ceiling.
        out = chunk.map(
            _convert_sample,
            num_proc=args.num_proc if len(chunk) >= args.num_proc else 1,
            fn_kwargs=map_fn_kwargs,
            remove_columns=chunk.column_names,
            desc="convert",
            writer_batch_size=args.map_writer_batch_size,
            features=_output_features(),
        )

        # In --strict mode failures already raised above. In --no-strict mode,
        # drop the rows that failed conversion (corrupt images, missing files,
        # unrenderable actions). ``_error`` is present on every row
        # (success = ""), so the filter is stable.
        if not args.strict:
            before = len(out)
            out = out.filter(
                lambda row: not row.get("_error"),
                writer_batch_size=args.map_writer_batch_size,
            )
            n_dropped += before - len(out)

        n_kept += len(out)
        converted_parts.append(out)

    if n_dropped > 0:
        print(f"Skipped {n_dropped} rows due to conversion errors")

    combined_out = (converted_parts[0] if len(converted_parts) == 1
                    else concatenate_datasets(converted_parts))
    if len(combined_out) > target:
        combined_out = combined_out.select(range(target))
    if args.sample is not None and len(combined_out) < args.sample:
        print(f"Only {len(combined_out)} rows convertible; --sample "
              f"{args.sample} exceeds what the pool can supply")

    # ``_error`` is a transient filter marker — drop it so it never reaches the parquet.
    combined_out = combined_out.remove_columns("_error")

    output = Path(args.output)
    _write_model_ready_parquet(
        combined_out, output,
        row_group_size=args.row_group_size,
        writer_batch_size=args.map_writer_batch_size,
    )
    print(f"Wrote {len(combined_out)} trajectory rows to {output}")


if __name__ == "__main__":
    main()
