---
description:
globs: lite/data/preproc/ui_genie_agent/**
alwaysApply: false
---

# UI-Genie-Agent Dataset Preprocessing Guide

Technical specification for the UI-Genie-Agent (`HanXiao1999/UI-Genie-Agent-16k`)
adapter. For how to run it, see [`README.md`](/lite/data/preproc/ui_genie_agent/README.md).
For the cross-dataset contract, see [`lite/data/preproc/AGENTS.md`](/lite/data/preproc/AGENTS.md).

## 1. Overview

UI-Genie-Agent is a set of mobile/Android multi-step agent trajectories. The
adapter emits a single `use` cohort, with each of the two source subsets
carried as a **variant**:

- **ui_genie** → `mobile@use`, variant `ui_genie` — `ui_genie_agent_16k.jsonl`,
  screenshots shipped in the HF repo (`data/screenshots/<uid>/screenshot-<n>.png`).
- **amex** → `mobile@use`, variant `amex` — `AMEX_Agent_34K.jsonl`, a
  re-annotation whose screenshots come from [`Yuxiang007/AMEX`](https://huggingface.co/datasets/Yuxiang007/AMEX)
  (`AMEX/screenshot/<uid>-<n>.png`), fetched + unpacked separately.

Each JSONL line is one *step*: `messages` = [system (carries `"resolution is
WxH"`), user (carries `"The user query: <goal>"`), assistant (one `<tool_call>`
wrapping a `mobile_use` action dict)] plus `images=[path]`. Steps are grouped
into trajectories by `uid` (parsed from the image path) and sorted by step index.
Splits are assigned by hashing `metadata.others.id` (per-trajectory unique;
2,000-row val cap per partition; no upstream-split label).

## 1a. Two annotation modalities ship under one filename

`ui_genie_agent_16k.jsonl` is **not one corpus**. It interleaves two annotation
modalities whose system prompts are disjoint, and only one of them is expressible
in the canonical `use` vocabulary. `step_resolution` in
[`use.py`](/lite/data/preproc/ui_genie_agent/use.py) is the single place a
trajectory is assigned to one, and it decides the modality **before** reading the
resolution — see that docstring for the measurement.

| | **point** (published) | **set-of-mark** (skipped) |
|---|---|---|
| prompt says | `The screen's resolution is WxH` | `The interactive UI elements on the screenshot are labeled with numeric tags starting from 1` |
| resolution declared | yes | **no, by design** — there is nothing to normalize |
| coordinate payload | `coordinate: [x, y]` in the declared frame | `som: <int>`, a numeric element tag |
| tag → coordinate map | n/a | **nowhere in the record** — the tags exist only as pixels burned into the screenshot |
| screenshots | clean device captures | overlaid with numbered label boxes |
| `ui_genie` steps | 14,282 of 16,698 | **2,416 of 16,698 (14.5%)** |
| `ui_genie` trajectories | 1,791 of 2,208 | **417 of 2,208 (18.9%)** |
| `amex` | 35,088 of 35,088 | **0** |

The two sets coincide exactly, in both directions and at both granularities:
every step lacking a resolution carries the SOM marker, none of the 14,282
resolution-declaring steps does, and the modality is uniform within a trajectory
(0 mixed). So the 417 are **one bucket, not a mixture**: a `malformed_resolution`
residual exists as `step_resolution`'s third outcome and measures **0** on
today's snapshot of both subsets.

**Why the SOM variant is unpublishable rather than merely unimplemented.** Its
action's argument is an element *tag*, not a point, so it cannot become
`tap(coordinate=…)`; the only thing that resolves a tag to a location is OCR of
the overlay, i.e. a re-annotation, not an adapter change. And the observation
itself is off-contract: the images carry numbered boxes no lite `mobile` env ever
renders, so publishing them would train a policy to expect labels that will not
be there. Both halves would have to be fixed, and the second one cannot be. 385
of the 417 have at least one `som`-referenced action; the remaining 32 are
single-step `terminate`-only trajectories — zero-action episodes over a
SOM-overlaid screenshot, so unpublishable on the observation alone.

Deriving the missing frame does not help here and was checked rather than assumed:
`smart_resize(image_h, image_w, factor=28, max_px=768*28*28)` reproduces the
declared resolution on **1,791 of 1,791** declaring trajectories (so the declared
`WxH` is the Qwen2-VL token grid, not the image size — a separate cross-dataset
question about `others.resolution`, not this adapter's). It is inapplicable to the
417 regardless: **0 of their 2,416 steps carry a `coordinate` key at all.**

Reproduce every count above with
[`use.py`](/lite/data/preproc/ui_genie_agent/use.py)'s own printed ledger:

```bash
uv run python lite/data/preproc/ui_genie_agent/use.py --subset ui_genie
# ui_genie: 2208 trajectories read (16698 step records) → 1747 rows, 461 skipped
# ui_genie skip reasons (trajectories): {'som_annotation_variant': 417,
#   'coordinate_outside_resolution': 33, 'truncated_terminal_text': 9,
#   'non_contiguous_step_numbers': 2}
```

**Related files:**

| Type | File |
|------|------|
| Parent specification | [`lite/data/preproc/AGENTS.md`](/lite/data/preproc/AGENTS.md) |
| Adapter shared helpers | [`utils.py`](/lite/data/preproc/ui_genie_agent/utils.py) (staging wiring + OOB check) |
| Preprocessing script | [`use.py`](/lite/data/preproc/ui_genie_agent/use.py) |
| Shell scripts | [`scripts/download_raw_data.sh`](/lite/data/preproc/ui_genie_agent/scripts/download_raw_data.sh) (download UI-Genie + AMEX), [`scripts/process_raw_data.sh`](/lite/data/preproc/ui_genie_agent/scripts/process_raw_data.sh) (verify layout, merge/unzip/symlink AMEX), [`scripts/process_data.sh`](/lite/data/preproc/ui_genie_agent/scripts/process_data.sh) (run Python pipeline) |
| HF repo metadata | [`repo.json`](/lite/data/preproc/ui_genie_agent/repo.json) |
| Tool definitions | [`lite/core/tools/action_space/base.py`](/lite/core/tools/action_space/base.py) (`LiteMobileActionSet`) |

**Source data locations** (under `${CUA_LITE_RAW_DATASETS_ROOT}/`):

| Variant | Annotations (JSONL) | Screenshots |
|---------|--------------------|-------------|
| ui_genie | `HanXiao1999/UI-Genie-Agent-16k/ui_genie_agent_16k.jsonl` | `HanXiao1999/UI-Genie-Agent-16k/data/screenshots/<uid>/screenshot-<n>.png` |
| amex | `HanXiao1999/UI-Genie-Agent-16k/AMEX_Agent_34K.jsonl` | `HanXiao1999/UI-Genie-Agent-16k/AMEX/screenshot/<uid>-<n>.png` (symlink → `Yuxiang007/AMEX/screenshot/`) |

`process_raw_data.sh` builds the AMEX symlink from the merged + unzipped
`Yuxiang007/AMEX` archive.

## 2. Output Directory Structure

```
${CUA_LITE_DATASETS_ROOT}/cua-lite/UI-Genie-Agent/
├── images/<hash[:2]>/<hash>.<ext>                       # content-addressed image store
└── mobile/
    └── use/
        └── <split>/
            ├── ui_genie.parquet                          # ui_genie subset
            └── amex.parquet                              # amex subset
```

`<split>` ∈ {`train`, `validation`}. Because the cohort is **multi-variant**
(two subsets), the variant is a real filename: `<split>/<variant>.parquet`.

## 3. Data Statistics

Regenerate after a full run:

```bash
uv run python -c "
from pathlib import Path
import pyarrow.parquet as pq
from lite.data.staging import iter_partitions
root = Path('${CUA_LITE_DATASETS_ROOT}/cua-lite/UI-Genie-Agent')
for *_, f in iter_partitions(root):
    print(f'{f.relative_to(root)}: {pq.ParquetFile(f).metadata.num_rows} rows')
print('images:', sum(1 for p in (root/'images').rglob('*') if p.is_file()))
"
```

### Skip ledger

Every drop is trajectory-granular and one trajectory yields at most one row, so
**the trajectory is the unit in which a run reconciles**, and `process_subset`
closes the ledger before returning `(rows, n_trajectories, n_step_records, skips)`:

```
n_trajectories == len(rows) + sum(skips.values())
```

with any remainder folded into `skips["unaccounted"]`, so a `raise` added later
without its own bucket surfaces by name. The per-reason counts print **every
run**, not only under `--verbose`, and the step-record count is printed as a
separate, labelled denominator — a count of steps and a count of trajectories are
different claims. The ledger key is `SkipTrajectoryError`'s required first
positional argument, so a drop nobody can count is unconstructible.

| Ledger key | Cause | `ui_genie` | `amex` |
|---|---|--:|--:|
| `som_annotation_variant` | the set-of-mark modality (§1a) — no coordinate frame, tags not points | **417** | 0 |
| `coordinate_outside_resolution` | source coordinate outside the declared frame (before clamping) | 33 | 1 |
| `duplicate_step_numbers` | two records claim the same logical `Task progress` index | 0 | 0 |
| `non_contiguous_step_numbers` | a logical decision record is missing inside the trajectory | 2 | 19 |
| `empty_text` | `type` / `open` with blank text | 0 | 3 |
| `truncated_terminal_text` | successful `action_info` ends with whitespace, including inside wrapping quotes; 18 source values are cut off mid-phrase, but 9 belong to the earlier SOM bucket | 9 | 0 |
| `malformed_resolution` | a point-modality prompt with no parseable `resolution is WxH` | 0 | 0 |
| `no_instruction` | `The user query:` absent | 0 | 0 |
| `malformed_messages`, `malformed_tool_call` | source record shape | 0 | 0 |
| `post_terminal_steps` | a later logical record follows `terminate`; reject instead of silently truncating | 0 | 0 |
| `missing_action_field`, `unmapped_action`, `key_action`, `unsupported_system_button`, `unsupported_terminate_status`, `swipe_without_endpoints`, `malformed_duration`, `non_positive_resolution`, `no_usable_steps` | action mapping refusals | 0 | 0 |
| `image_absent_on_host` | the step screenshot is not on this host — **host lacking data**, not the adapter refusing it | 0 | 0 |
| `image_corrupt_on_host` | unreadable image at staging time — likewise host-side | 0 | 0 |
| `oob_coordinate` | OOB after normalization (`has_oob_coordinate`; rare — coords are clamped to [0, 1000]) | 0 | 0 |
| | **total skipped / read** | **461 / 2,208** | **23 / 2,981** |

Trajectories that lack a final `terminate` but end on an executable action now
publish that action as the EOF label instead of entering the skip ledger. Some
SOM-modality trajectories also lack a `terminate`, but the modality remains the
more fundamental disqualifier and is named first.

## 4. Source Data Format

Each JSONL line (one step):

```json
{
  "messages": [
    {"role": "system", "content": "... The screen resolution is 1080x2400 ..."},
    {"role": "user", "content": "... The user query: Open settings and turn on wifi\nTask progress ..."},
    {"role": "assistant", "content": "<tool_call>{\"name\": \"mobile_use\", \"arguments\": {\"action\": \"click\", \"coordinate\": [540, 1200], \"action_desc\": \"tap the Settings icon\"}}</tool_call>"}
  ],
  "images": ["data/screenshots/<uid>/screenshot-0.png"]
}
```

- **uid + step index** are parsed from `images[0]`: `ui_genie` →
  `.../<uid>/screenshot-<n>.png`; `amex` → `.../<uid>-<n>.png`.
- **resolution** is regex-matched from the system prompt (`resolution is WxH`),
  but only after `step_resolution` has established that the prompt is the point
  modality at all — see §1a and §4a.
- **instruction** is the text after `The user query:` (up to `\nTask progress`),
  taken from the first step.
- coordinates are **pixels in the per-step screen resolution**, normalized to
  [0, 1000] and clamped (`_norm_xy`).
- `action_desc` → `action_description` content part (via [`make_assistant_content`](/lite/core/messages/content.py)).

## 4a. Logical steps are not screenshot filename numbers

The filename suffix locates an image; it is not the action index. AMEX commonly
jumps from filenames such as 1 to 4 while `Task progress` advances normally by
one. Conversely, six valid trajectories reuse one filename for two consecutive
logical states (five UI-Genie, one AMEX). Sorting or validating by filename would
therefore either corrupt or discard valid data.

The adapter instead uses the largest `StepN:` in accumulated `Task progress`
(no history means logical step 0). Across the full sources this yields zero
logical duplicates. It finds four UI-Genie and 19 AMEX trajectories with a real
interior `+2` jump. Two UI-Genie cases are SOM and are rejected under that more
fundamental reason; the other two and all 19 AMEX cases are rejected as
`non_contiguous_step_numbers` rather than pairing an action with a screenshot
that follows an unobserved decision.

Nothing in the published artifact records either fact today. In particular a
dataset `stats.json` cannot: for a preproc dataset it is written by
[`lite/data/hf/upload.py`](/lite/data/hf/upload.py) from `collect_stats_from_disk`,
which never assigns `rows_dropped`, so the `0` there is the `DatasetStats` field
default. The only producer that ever sets it is
[`lite/data/hf/stage.py`](/lite/data/hf/stage.py)'s `filter_fn`, on the rollout
log-root path a preproc dataset never traverses. Read the adapter's own printed
ledger instead.

## 5. Action Space & Mapping

Mobile action space ([`LiteMobileActionSet`](/lite/core/tools/action_space/base.py)),
keyed off the `mobile_use` `arguments.action`:

| Source `action` / payload | → tool_call |
|---|---|
| `click {coordinate}` | `tap(coordinate=C)` |
| `long_press {coordinate, time?}` | `long_press(coordinate=C, duration=time?)` |
| `swipe {coordinate, coordinate2}` | `swipe(start_coordinate=C1, coordinate=C2)` |
| `swipe {direction: up/down/left/right}` | `swipe(...)` with synthetic [0,1000] endpoints (`DIRECTION_SWIPES`) |
| `type {text}` | `type(text=...)` |
| `system_button {button ∈ Back/Home/Enter/Menu/Recent}` | `system_button(button=...)` |
| `open {text}` | `open_app(app_name=text)` |
| `wait {time?}` | `wait(duration=time?)` (default 1.0) |
| `terminate {status ∈ success/failure, action_info?}` | terminal outcome; not persisted as a tool call, and `action_info` becomes the final turn's text when it is the task's answer (see below) |
| `key` (ADB keyevent) | **skip trajectory** (`key_action`) |
| any other action | **skip trajectory** (`unmapped_action`) |

`args_to_tool_call` has **no `som` branch**, deliberately: a `som`-referenced
action is reachable only from a set-of-mark prompt, which `step_resolution`
already skipped as `som_annotation_variant`. Measured exhaustively — all 1,768
`som`-bearing steps sit under a SOM prompt and 0 of the 14,282 point-modality
steps carry the key — so a branch there would restate a classification already
made, and it was the pair of them racing that produced the mis-attribution §1a
describes: the resolution check ran first, so `"uses SOM reference"` fired **zero**
times and 417 trajectories were logged as malformed source prompts instead.

**Terminator.** A source `terminate` step is dropped from `messages`; no `terminate`
tool_call is persisted in `use` rows, and non-success status/reason payloads move to
`metadata.others` via `terminate_outcome_others`. The episode ends on the content-only
assistant final, whose **text is the task's answer when the terminate authored one**.

`ui_genie` hangs all source-authored terminal text off `terminate.action_info`.
The adapter preserves every complete, non-empty successful value as the content-only final.
It does not guess from English wording whether the text is an answer or a completion
note: the old regex both deleted real answers and misclassified product names such as
“Find My Device”. A value ending in whitespace (including immediately inside wrapping
quotes) is rejected as `truncated_terminal_text`; all 18 such source values end mid-phrase,
with 9 already rejected earlier as set-of-mark trajectories.
Empty values still fall back to `Done.`; failure text remains outcome metadata rather
than a successful final answer.

`amex` never authors `action_info` (2,816/2,816 terminates carry `{status}` only), so
that variant always ends on `Done.`.

## 6. Processing Workflow

1. Per subset: locate the JSONL and assert the screenshot dir exists
   (`image_dir_check_rel`); raise if absent.
2. Stream the JSONL, parse `(uid, step)` from `images[0]`, group records by uid
   (`group_records_by_uid`). Invalid JSON, non-object rows, malformed `images`,
   and unrecognized image paths raise with the source file and line number;
   they are never silently omitted before the trajectory ledger starts.
3. Per uid (`build_trajectory`): derive the logical index from accumulated `Task progress`, reject duplicates/interior gaps, and sort by that index;
   parse instruction + first-step resolution.
4. Per step: parse per-step resolution + `<tool_call>` args, map action to a
   tool_call (coords normalized to [0,1000]), resolve the screenshot. Attach an
   `action_description` content part when `action_desc` is non-empty.
5. Consume the source `terminate` into the final policy: pop it, then append the
   content-only final — its text is `terminal_answer_text(action_info)`
   when the successful source authored non-empty text, else the `Done.` marker.
   Build user/assistant messages.
6. `has_oob_coordinate` over the whole trajectory → drop if any OOB.
7. `stage_entry` hashes images into the content-addressed store, normalizes
   tagged CUA metadata, assigns a split (hash on `id`, 2,000-row val cap), keyed
   by `(metadata.dims[0], metadata.dims[1], split, variant)`; `flush_buffers` writes one parquet
   per partition.

## 7. Error Handling

| Error type | Behavior |
|------------|----------|
| Malformed JSONL envelope / image path | fail loud with file and line number before grouping |
| Set-of-mark modality | `SkipTrajectoryError("som_annotation_variant", …)` from `step_resolution`, before any coordinate is read |
| Any other per-step / per-trajectory validation failure above | `SkipTrajectoryError(<ledger key>, …)` → skip whole trajectory, counted under that key |
| OOB coordinate | drop whole trajectory via `has_oob_coordinate`, counted `oob_coordinate` |
| Corrupt image at staging | skip whole trajectory, counted `image_corrupt_on_host` |
| Missing JSONL / screenshot dir for a requested subset | raise `FileNotFoundError` (fail loud) |

Every key is listed with its measured count in the §3 ledger table.

## 8. Output Format

**This worked example is one of the 316** (§4a): its source steps are `[1,2,3,4,5]`
and `screenshot-0.png` does not exist, which is why the episode opens on the Timer
tab and its first action is *"Switch to the Alarm tab **from the Timer tab**"* —
the `open`-the-clock-app step that preceded it is missing from the source. Kept as
the example precisely because it is typical, but read it knowing that.

Its frame was re-verified against pixels, not just arithmetic: the source coordinate
`[50, 1067]` in the declared `504x1148` grid normalizes to `[99, 929]`, which
de-normalizes on the real `1080x2400` screenshot to px `(107, 2230)` — dead centre
on the *Alarm* tab icon, matching the action description.

`use` row (real row `ui_genie_16k_00388e23`, two middle steps elided at `…`;
goal text and image hashes truncated). Actions are nested in a `mobile`
batch wrapper, each post-action screenshot is a `role:"tool"` message keyed by
the batch's `id`, and no `terminate` tool_call is persisted — the episode
ends on a content-only final. This goal asks a question, so that final's text is the
answer the source hung off `terminate` as `action_info` (`"The new alarm is set for
07:00."`) rather than the `Done.` marker.

The block below is the **whole** row, regenerated from the raw trajectory and fed
back through `validate_canonical_rows`; only image hashes and the goal text are
abbreviated at `…`. No step is elided: an earlier version replaced two middle
`assistant`/`role:"tool"` pairs with a bare `"…"` **string** inside `messages`,
which the gate rejects outright (`role None is outside the Lite role vocabulary`) —
`messages` elements are objects, never prose placeholders.

```json
{
  "images": [
    "cua-lite/UI-Genie-Agent/images/f9/f91881b8e8…png",
    "cua-lite/UI-Genie-Agent/images/6e/6ed5535eee…png",
    "cua-lite/UI-Genie-Agent/images/3c/3c278c0dc7…png",
    "cua-lite/UI-Genie-Agent/images/54/54644f148d…png",
    "cua-lite/UI-Genie-Agent/images/f9/f9b7f486c8…png"
  ],
  "messages": [
    {"role": "user", "content": [{"type": "image", "index": 0}, {"type": "text", "text": "You should use clock to complete the following task: What time is the new al…"}]},
    {"role": "assistant",
     "tool_calls": [{"id": "call_0000", "type": "function", "function": {"name": "mobile", "arguments": {"actions": [{"action": "tap", "coordinate": [99, 929], "clicks": 1}]}}}],
     "content": [{"type": "action_description", "text": "Switch to the Alarm tab from the Timer tab."}]},
    {"role": "tool", "tool_call_id": "call_0000", "content": [{"type": "image", "index": 1}]},
    {"role": "assistant",
     "tool_calls": [{"id": "call_0001", "type": "function", "function": {"name": "mobile", "arguments": {"actions": [{"action": "tap", "coordinate": [500, 815], "clicks": 1}]}}}],
     "content": [{"type": "action_description", "text": "Add a new alarm by clicking the plus button."}]},
    {"role": "tool", "tool_call_id": "call_0001", "content": [{"type": "image", "index": 2}]},
    {"role": "assistant",
     "tool_calls": [{"id": "call_0002", "type": "function", "function": {"name": "mobile", "arguments": {"actions": [{"action": "tap", "coordinate": [377, 660], "clicks": 1}]}}}],
     "content": [{"type": "action_description", "text": "Set the hour to 7 in the alarm time selection interface."}]},
    {"role": "tool", "tool_call_id": "call_0002", "content": [{"type": "image", "index": 3}]},
    {"role": "assistant",
     "tool_calls": [{"id": "call_0003", "type": "function", "function": {"name": "mobile", "arguments": {"actions": [{"action": "tap", "coordinate": [776, 760], "clicks": 1}]}}}],
     "content": [{"type": "action_description", "text": "Confirm the selected alarm time of 07:00 by clicking OK."}]},
    {"role": "tool", "tool_call_id": "call_0003", "content": [{"type": "image", "index": 4}]},
    {"role": "assistant", "content": [{"type": "text", "text": "The new alarm is set for 07:00."}]}
  ],
  "metadata": {
    "metadata_kind": "cua", "dims": ["mobile", "use"],
    "extra_tool_schemas": [], "valid_actions": null,
    "others": {"id": "ui_genie_16k_00388e23", "resolution": [504, 1148], "os": "android",
               "source": "HanXiao1999/UI-Genie-Agent-16k", "source_id": "00388e23"}
  }
}
```

A non-success source terminator is preserved as `others.terminate_status`
(plus `others.terminate_reason` when the source authored non-blank text); a
success terminator records nothing there, since the final turn already carries it —
as the answer when the task asked for one, else as the `Done.` marker. (In practice
both subsets are 100% `status="success"`: 2,176/2,176 and 2,816/2,816.)

The `amex` variant is identical except `others.id` is prefixed `ui_genie_amex_`.
