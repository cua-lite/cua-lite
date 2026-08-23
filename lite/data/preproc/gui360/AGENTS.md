# GUI-360 Dataset Preprocessing Guide

Technical specification for the GUI-360 (`vyokky/GUI-360`) adapter. It conforms
to the cua-lite preprocessing contract in
[`/lite/data/preproc/AGENTS.md`](/lite/data/preproc/AGENTS.md); for how to run
the pipeline see [`/lite/data/preproc/gui360/README.md`](/lite/data/preproc/gui360/README.md).
The reference implementation for shared conventions is
[`/lite/data/preproc/scalecua/AGENTS.md`](/lite/data/preproc/scalecua/AGENTS.md).

GUI-360 is a large-scale dataset of computer-using-agent trajectories on
**Windows Microsoft-Office** apps (Word / Excel / PowerPoint), collected with the
Microsoft UFO framework. Every sample is therefore CUA platform dim `"desktop"`,
`os="windows"`, `source="vyokky/GUI-360"`. See the
[paper](https://arxiv.org/abs/2511.04307).

## 1. Overview

### Related Files

| Type | File |
|------|------|
| Parent specification | [`/lite/data/preproc/AGENTS.md`](/lite/data/preproc/AGENTS.md) |
| Action-space definitions | [`/lite/core/tools/action_space/base.py`](/lite/core/tools/action_space/base.py) (`LiteDesktopActionSet`) |
| Shared helpers | [`utils.py`](/lite/data/preproc/gui360/utils.py) (image store, splitter, `stage_entry`, `has_oob_coordinate`, `iter_json_array`, `normalize_coordinate`, `VK_KEY_MAP`) |
| Preprocessing scripts | [`grounding-point.py`](/lite/data/preproc/gui360/grounding-point.py), [`understanding.py`](/lite/data/preproc/gui360/understanding.py), [`use.py`](/lite/data/preproc/gui360/use.py) |
| Shell scripts | [`scripts/download_raw_data.sh`](/lite/data/preproc/gui360/scripts/download_raw_data.sh), [`scripts/process_raw_data.sh`](/lite/data/preproc/gui360/scripts/process_raw_data.sh), [`scripts/process_data.sh`](/lite/data/preproc/gui360/scripts/process_data.sh) |
| HF repo metadata | [`repo.json`](/lite/data/preproc/gui360/repo.json) |

### Source Data Locations

The GUI-360 step record is tagged with the tasks it can generate
(`tags: [grounding, screen_parsing, action_prediction]`), so the upstream
`processed_data/` subsets are just per-step *views* of the **successful** raw
`train/` trajectories. We consume each task at its most useful granularity:

| cua-lite cohort | Source | Path (under `${CUA_LITE_RAW_DATASETS_ROOT}/vyokky/GUI-360/`) |
|---|---|---|
| `desktop/use` | raw `train/` trajectories | `train/data/{excel,word,ppt}/{in_app,search,online,wikihow}/success/*.jsonl` + `train/image.tar.gz` |
| `desktop/grounding.point` | processed `grounding_resize` | `processed_data/grounding_resize/{training_data.json,images/}` |
| `desktop/understanding` (variant `screen_parsing`) | processed `screen_parsing_train_resize` | `processed_data/screen_parsing_train_resize/{training_data.json,images/}` |

**Deliberately excluded** (see [§2 of the parent contract](/lite/data/preproc/AGENTS.md) and the [decisions](#decisions-recap) below):

| Excluded | Why |
|---|---|
| `processed_data/action_prediction_train_resize` | Single-step action prediction — exactly what unfolding `use` produces. Redundant. |
| `processed_data/action_prediction_train_resize_a11y` | Byte-identical prompts to the non-a11y subset; the only difference is float vs int coordinates. Images ship as a 20 GB tar that is never extracted. Adds nothing for our `image+instruction` format. |
| `test/` | This is GUI-360-Bench (held-out benchmark) — training on it would contaminate the benchmark. |
| `fail/` | Failed trajectories; the paper reserves these for error-analysis / RL, **not** SFT. |

Raw trajectory screenshots are native **1040×736**; the processed subsets are
resized (grounding 1040×736, screen_parsing 1036×728). Coordinates are
normalized per-image from the actual decoded size (read via PIL), so the small
resolution differences are handled automatically.

## 2. Output Directory Structure

```
${CUA_LITE_DATASETS_ROOT}/cua-lite/GUI-360/
├── images/<hash[:2]>/<hash>.png            # content-addressed image store
└── desktop/
    ├── use/<split>/<variant>.parquet                 # multi-step trajectories
    ├── grounding.point/<split>/<variant>.parquet     # intent -> element coordinate
    └── understanding/<split>/<variant>.parquet        # screen parsing (one variant)
```

`<split>` ∈ {`train`, `validation`}, assigned by the hash-based
[`SplitAssigner`](/lite/data/staging.py) (2 000-decision validation cap per cohort;
no upstream split label inside the `train` set we consume).

## 3. Data Statistics

| Cohort | Input records | Output rows | Notes |
|---|---|---|---|
| `grounding.point` | 79,487 | 79,426 | 61 true out-of-bounds points dropped |
| `understanding` | 97,351 | 97,335 | 16 screens had no retained visible control |
| `use` | 13,750 train trajectories | 10,482 | 2,376 Office-API, 842 incomplete/empty/other, and 50 out-of-bounds dropped |

`use` drop/skip reasons from the full source run (regenerate exact numbers from
the run summary printed by `use.py`):

- **Office-API trajectory drop** — 2,376 / 13,750 (17.3%); contains an Office-API action (`select_table_range`, `set_font`, `insert_excel_table`, `insert_table`, `select_paragraph`, `select_table`, `save_as`, `set_background_color`, `table2markdown`, `reorder_columns`, …) with no faithful coordinate/UI equivalent.
- **per-step skips** within kept trajectories — no-op "sub-task complete" steps (empty action) and `summary` steps (textual state report); neither changes screen state.
- **incomplete** — `evaluation.complete != "yes"` or final step not `OVERALL_FINISH`, or < 1 real action step.

Reproduce counts:

```bash
# Per-cohort row counts
uv run python -c "
from pathlib import Path; from datasets import load_dataset
from lite.data.staging import iter_partitions
root = Path('${CUA_LITE_DATASETS_ROOT}/cua-lite/GUI-360')
for *_, f in iter_partitions(root):
    print(f.relative_to(root), len(load_dataset('parquet', data_files=str(f), split='train')))
"
# Use drop/keep tallies are printed by use.py at the end of its run.
```

## 4. Source Data Format

### 4a. Processed `grounding_resize` record

```json
{
  "id": "excel_4s_1_2",
  "images": ["images\\excel_4s_1/action_step4.png"],
  "conversation": [
    {"from": "human", "value": "<image>\nYou are a helpful assistant. ... The instruction is:\nTo use the ATAN function as requested, I will enter the formula =ATAN(1) into cell H4, which is currently empty and selected. ...\n\nOutput the coordinate of the element you will operate within <coordinate></coordinate> tag:\n<coordinate> [x, y] </coordinate>\n"},
    {"from": "gpt", "value": "<coordinate> [630, 241] </coordinate>"}
  ]
}
```

The "instruction" is the agent's step **thought** (a description of the target
element / intended action). The image path's first component uses a Windows
backslash (`images\...`) — normalized to `/` before resolving. Coordinates are
absolute pixels on the 1040×736 image.

### 4b. Processed `screen_parsing_train_resize` record

```json
{
  "id": "excel_bing_search_excel_4s_1_2",
  "images": ["images/excel_4s_1/action_step4.png"],
  "conversation": [
    {"from": "human", "value": "<image>\n\nYou are an expert in screen parsing ... output ... control_text ... control_rect [left, top, right, bottom] ..."},
    {"from": "gpt", "value": "[{\"control_type\": \"Button\", \"control_rect\": [62, 699, 94, 720], \"control_text\": \"Macro Recording Not Recording\", \"source\": \"uia\", \"label\": 1}, ... 426 more ...]"}
  ]
}
```

The answer is a JSON array (≈ 50 KB, up to 500 controls) from the Windows UIA
accessibility tree. It carries three fields the prompt never requests
(`control_type`, `source="uia"`, `label`); coordinates are pixels on 1036×728 —
but ≈ 4.7% of them do **not** fit that box, because UIA recorded them on the
pre-resize 1040×736 screenshot (see
[clip or drop](#out-of-bounds-control_rect--clip-or-drop)).

### 4c. Raw `train` trajectory step (one JSONL line per step)

`ui_tree` / `control_infos` (50–80 KB each) shown truncated; everything else
verbatim:

```json
{
  "execution_id": "excel_4s_1",
  "app_domain": "excel",
  "request": "Enable editing and use the ATAN function in Excel to calculate the arctangent of a number.",
  "step_id": 2,
  "total_steps": 5,
  "evaluation": {"complete": "yes", "reason": "...", "sub_scores": {...}},
  "step": {
    "screenshot_clean": "success/excel_4s_1/action_step4.png",
    "screenshot_annotated": "success/excel_4s_1/action_step4_annotated.png",
    "subtask": "In the Excel file ... use the ATAN function in an empty cell (e.g., H4) ...",
    "observation": "The current screenshot shows ...",
    "thought": "To use the ATAN function as requested, I will enter the formula =ATAN(1) into cell H4 ...",
    "action": {
      "action_type": "GUI",
      "control_text": "Formula Bar",
      "function": "type",
      "args": {"text": "=ATAN(1)", "clear_current_text": true},
      "coordinate_x": 630.0, "coordinate_y": 241.5,
      "rectangle": {"left": 227, "top": 218, "right": 1028, "bottom": 259}
    },
    "status": "CONTINUE",
    "tags": ["grounding", "screen_parsing", "action_prediction"],
    "ui_tree": "[truncated, ~60 KB]",
    "control_infos": {"uia_controls_info": "[truncated, 427 controls]"}
  }
}
```

A trajectory terminates with a final step `status: "OVERALL_FINISH"` and an empty
API action (`function: "", args: {}`). Mid-trajectory `status: "FINISH"` steps
with an empty action are "sub-task complete" no-ops.

## 5. Action Space & Mapping

GUI-360 uses a Windows-Office action space. Only the **GUI** actions map to the
cua-lite desktop action space; Office-**API** actions have no GUI equivalent.
Coordinates come from `action.coordinate_x/coordinate_y` (fallback `args.x/args.y`)
and are normalized to `[0, 1000]` against the decoded image size.

| GUI-360 action | → cua-lite desktop |
|---|---|
| `click {x,y,button,double}` | `click(coordinate, button, clicks=2 if double else 1)` |
| `double_click_input` / `double_click_on_coordinates` | `click(coordinate, clicks=2)` |
| `type {text=...}` | `click(coordinate)` + optional `key(keys=["ctrl", "a"])` when clearing + `type(text=...)` |
| `type {keys=SendKeys}` | ordered `key` / `key_down` / `key_up` / `type`; preserves repeats, literals, modifiers, and modifier groups |
| `set_focus` | `click(coordinate)` |
| `select_text` | Office API with no faithful computer equivalent → drop trajectory |
| `drag {start_x,start_y,end_x,end_y,button,key_hold?}` | optional `key_down` → `drag` → `key_up` |
| `wheel_mouse_input {wheel_dist}` | `scroll(coordinate, direction=up if dist>0 else down, amount=abs(dist))` |
| empty action with terminal `OVERALL_FINISH` | content-only assistant `Done.` |
| empty action mid-trajectory / `summary` | step skipped (no screen change) |
| any Office-API action | **trajectory dropped** (`use`) |

### `{VK_*}` key-token mapping ([`utils.py:VK_KEY_MAP`](/lite/data/preproc/gui360/utils.py))

`{VK_CONTROL}→ctrl`, `{VK_SHIFT}→shift`, `{VK_MENU}→alt`,
`{ENTER}`/`{VK_RETURN}→enter`, `{TAB}→tab`, `{VK_ESCAPE}→esc`,
`{VK_BACK}→backspace`, `{VK_DELETE}→delete`, arrows / `home` / `end` /
`pageup` / `pagedown`, `f1`–`f12`, `{VK_SPACE}→space`. A `{VK_*}c` chord →
`["ctrl", "c"]`; a standalone modifier token such as `{VK_MENU}` remains a
standalone `key(keys=["alt"])` action rather than a dangling chord. Unknown `VK_`
tokens raise `SkipTrajectory` (so new codes are
caught rather than silently dropped); single-character virtual-key spellings
such as the source's successful `{VK_P}` action map to that literal character.
Bare `+`, `^`, and `%` inside ordinary text are literal punctuation, while a
leading compact SendKeys expression such as `^a` remains a modifier chord.
For an all-symbol compact chord, the final symbol is the target: `^+` is
Ctrl+Plus and `^+%` is Ctrl+Shift+Percent.
Whitespace after a leading modifier is the Space key, so the source's
`{VK_CONTROL} {SPACE}` and `{VK_SHIFT} ` spellings become Ctrl+Space and
Shift+Space rather than a fabricated `plus` key. GUI-360's source `type` is target-aware,
while canonical `type` writes only to the currently focused field, so a source
coordinate is emitted as a focus click. `clear_current_text` is preserved as
`Ctrl+A` after that click and before typing. Drag `key_hold` is preserved with
`key_down`/`key_up`; the source duration has no canonical drag field.
Keys are emitted through
`LiteDesktopActionSet.key(...)`, so the stored tokens are lowercase named keys
or literal printable glyphs regardless of the intermediate spelling — see the
[Lite key vocabulary](/lite/data/preproc/AGENTS.md#keyboard-keys--lowercase-named-keys-plus-printable-glyphs).

## 6. Processing Workflow

1. **grounding.point** — stream `grounding_resize/training_data.json`
   ([`iter_json_array`](/lite/data/preproc/gui360/utils.py)); extract the
   instruction (the `thought` between `The instruction is:` and `Output the
   coordinate`), parse `<coordinate>[x,y]</coordinate>`, normalize against the
   decoded image size, emit one `point(coordinate)` tool call.
2. **understanding** — stream `screen_parsing/training_data.json`; parse the
   controls JSON, trim each to `{control_text, control_rect}` with `control_rect`
   [clipped to the image then normalized](#out-of-bounds-control_rect--clip-or-drop) to `[0, 1000]`, drop
   `control_type/source/label`, and emit the trimmed list as the assistant's
   plain `text`. The user prompt is rewritten so it is consistent with the
   normalized-coordinate answer.
3. **use** — for each `train/.../success/*.jsonl` trajectory: keep only
   `complete == "yes"` trajectories ending in `OVERALL_FINISH`; map each GUI
   step's action, skipping no-op/`summary` steps and dropping the whole
   trajectory on any Office-API action; resolve each step's `screenshot_clean`
   under `train/image/<app>/<category>/`; build a first
   user(image + goal) turn followed by assistant(`inline_reasoning`=thought,
   `action_description`=subtask, `tool_calls`=one `computer` batch) turns, each
   answered by a `role:"tool"` message that carries the batch's `id` as `tool_call_id` and the
   next screenshot (`finalize_use_messages`); the terminal `OVERALL_FINISH`
   becomes a content-only `Done.` final. Asserts
   `len(images) == n_assistant_turns`.
4. All paths run each row through `stage_entry` (hash images into the store,
   fill metadata defaults, assign split) and drop rows whose tool-call
   coordinates fall outside `[0, 1000]` via `has_oob_coordinate`.

### Out-of-bounds `control_rect` — clip or drop

`control_rect` values are UIA bounds captured on the **1040×736** screenshot and
shipped against the **1036×728** resize, so a large minority of them do not fit
the image they annotate. This is upstream noise plus a little upstream truth, and
the adapter separates the two **geometrically**, never by refusing the record:

* **clip** the rect to the image, and
* **drop** the control when the clipped rect has no area — its intersection with
  the screen is empty, so it is not one of the controls "visible in this
  screenshot" that the prompt asks for.

Both counts are printed by `understanding.py` on **every** run (not only under
`--verbose`). Measured over 2,400 records / 801,796 controls sampled at 20 byte
offsets of `training_data.json` — re-derive rather than quoting these:

| population | count | share |
|---|---|---|
| controls outside the image | 37,818 | 4.72% |
| …of which overflow by exactly **2 px** (the resize rounding) | 24,898 | 65.8% of the above |
| …of which have **empty** intersection with the image | 322 | 0.85% of the above |
| in-bounds zero-area `[8, 8, 8, 8]` UIA phantoms (also dropped) | 98 | 0.012% of all controls |
| records carrying ≥1 out-of-bounds rect | 1,648 / 2,400 | 68.7% |
| records left with **no** control after dropping | 0 | — |

**A pixel tolerance cannot be the rule**: the clip and drop populations overlap
across the whole overflow range (3,875 clippable vs 70 off-screen at 6–20 px;
215 vs 100 at 51–200 px; smallest off-screen overflow 20 px, largest 2,453 px).
The intersection test separates them exactly, which is why the rule is geometric.

Clipping is also what makes the answer obey its own prompt: the un-clipped
normalized rects put **37,818** values outside the `[0, 1000]` range the prompt
promises, and clipping puts that at **0**. Against the pre-clip behaviour, on
1,000 real records / 449,601 controls: 427,146 rects byte-identical, 22,278
altered (4.96%, and 18,153 of those by ≤2 normalized units), 177 dropped
(0.039%), 0 `control_text` changes, and prompt / `images` / `metadata`
identical on all 1,000.

## 7. Error Handling

| Condition | Behavior |
|---|---|
| Missing / unresolvable image | grounding/understanding: skip the record (counted); use: skip the whole **trajectory** |
| Corrupt / unreadable image | skip the affected record/trajectory and continue; print the count |
| Malformed use JSONL | fail loud with source path and line number |
| Malformed/truncated understanding JSON array | fail loud with source path |
| Unparseable `<coordinate>` / instruction (grounding) | `raise ValueError` (malformed annotation that exists) |
| `control_rect` that is not four numbers (screen_parsing) | `raise ValueError` (0 occurrences in 801,796 measured controls) |
| `control_rect` overflowing the image (screen_parsing) | clip to the image, count it |
| `control_rect` with no on-screen area (screen_parsing) | drop just that control, count it |
| Empty controls list (screen_parsing) | skip the record (counted) |
| Every control off-screen (screen_parsing) | skip the record (counted) |
| Office-API action (`use`) | skip the whole trajectory (counted) |
| no-op / `summary` step (`use`) | skip just that step |
| Unknown `{VK_*}` token (`use`) | `SkipTrajectory` (skip trajectory, counted) |
| `complete != "yes"` / not `OVERALL_FINISH` / < 1 action step | skip the trajectory |
| Tool-call coordinate outside `[0, 1000]` | drop the row (`use`: the whole trajectory) |

## 8. Output Format

Actions are **never** emitted as bare top-level calls: every step's
actions are nested inside a `computer` batch wrapper (`platform="desktop"`), and
each post-action screenshot comes back as a `role:"tool"` message carrying the
`tool_call_id` of the call it answers. See the canonical schema in
[§ Use of the parent contract](/lite/data/preproc/AGENTS.md#3-use-multi-step-rollout-tasks).

Every block in this section was regenerated from the raw record and fed back
through `validate_canonical_rows`; only reasoning text and image hashes are
abbreviated at `…`. Note the `use` block's counts: **2 actions, 3 images** — each
assistant call `id` is answered by a `role:"tool"` result before the content-only final. An
earlier version showed 2 images and left `call_0001` unanswered, which the gate
**rejects**.

A `use` record (real row `excel_3_1089`; reasoning text and image hashes truncated):

```json
{
  "images": [
    "cua-lite/GUI-360/images/dd/ddadcd0fb2bc72601…png",
    "cua-lite/GUI-360/images/82/82d14a914cbede9b7…png",
    "cua-lite/GUI-360/images/76/766afeb6080865…png"
  ],
  "messages": [
    {"role": "user", "content": [
      {"type": "image", "index": 0},
      {"type": "text", "text": "Insert the copyright symbol (©) in the Excel spreadsheet."}
    ]},
    {"role": "assistant",
     "content": [
       {"type": "inline_reasoning", "text": "To insert the copyright symbol (©) into the Excel spreadsheet, I need to place the curso…"},
       {"type": "action_description", "text": "Insert the copyright symbol (©) somewhere in the 'm365_help_doc_310002327' Excel spreads…"}
     ],
     "tool_calls": [
       {"id": "call_0000",
        "type": "function",
        "function": {"name": "computer", "arguments": {"actions": [
          {"action": "click", "coordinate": [487, 295]},
          {"action": "type", "text": "©"}
        ]}}}
     ]},
    {"role": "tool", "tool_call_id": "call_0000", "content": [{"type": "image", "index": 1}]},
    {"role": "assistant",
     "content": [
       {"type": "inline_reasoning", "text": "Given that the copyright symbol has been successfully inserted and is visible within the…"},
       {"type": "action_description", "text": "Insert the copyright symbol (©) somewhere in the 'm365_help_doc_310002327' Excel spreads…"}
     ],
     "tool_calls": [
       {"id": "call_0001",
        "type": "function",
        "function": {
          "name": "computer",
          "arguments": {"actions": [{"action": "click", "coordinate": [178, 43]}]}
        }}
     ]},
    {"role": "tool", "tool_call_id": "call_0001", "content": [{"type": "image", "index": 2}]},
    {"role": "assistant", "content": [{"type": "text", "text": "Done."}]}
  ],
  "metadata": {
    "metadata_kind": "cua",
    "dims": ["desktop", "use"],
    "extra_tool_schemas": [],
    "valid_actions": null,
    "others": {
      "id": "excel_3_1089",
      "resolution": [1040, 736],
      "os": "windows",
      "source": "vyokky/GUI-360",
      "source_id": "excel_online"
    }
  }
}
```

Note the first step: a source `type` action with a coordinate expands to
`click` + `type` because canonical `type` has no target coordinate. When
`clear_current_text=true`, the same `computer` wrapper carries `click`,
`Ctrl+A`, then `type`.

A `grounding.point` record:

```json
{
  "images": ["cua-lite/GUI-360/images/6b/6bfbeb67df74a8267…png"],
  "messages": [
    {"role": "user", "content": [
      {"type": "image", "index": 0},
      {"type": "text", "text": "To use the ATAN function as requested, I will enter the formula =ATAN(1) into cell H4, w…"}
    ]},
    {"role": "assistant", "tool_calls": [
      {"id": "call_0000", "type": "function", "function": {"name": "point", "arguments": {"coordinate": [606, 327]}}}
    ]}
  ],
  "metadata": {"metadata_kind": "cua", "dims": ["desktop", "grounding.point"], "extra_tool_schemas": [], "valid_actions": null,
    "others": {"id": "excel_4s_1_2", "resolution": [1040, 736], "os": "windows", "source": "vyokky/GUI-360", "source_id": "grounding_resize"}}
}
```

An `understanding` (screen_parsing) record:

```json
{
  "images": ["cua-lite/GUI-360/images/12/129086b2c0ea90112…png"],
  "messages": [
    {"role": "user", "content": [
      {"type": "image", "index": 0},
      {"type": "text", "text": "List all interactive controls (anything that can be clicked, typed into, or selected) visible in this screenshot. For each control, give its visible text and bounding box.\n\nRespond with a JS…"}
    ]},
    {"role": "assistant", "content": [
      {"type": "text", "text": "[{\"control_text\": \"Macro Recording Not Recording\", \"control_rect\": [60, 960, 91, 989]}, ...]"}
    ]}
  ],
  "metadata": {"metadata_kind": "cua", "dims": ["desktop", "understanding"], "extra_tool_schemas": [], "valid_actions": null,
    "others": {"id": "excel_bing_search_excel_4s_1_2", "resolution": [1036, 728], "os": "windows", "source": "vyokky/GUI-360", "source_id": "screen_parsing_train_resize"}}
}
```

## Decisions recap

- **action_prediction dropped** — it is the single-step view of the same
  successful `train` trajectories that `use` reconstructs in full;
  keeping both would duplicate the supervision.
- **grounding.point kept** — a genuinely different task (intent → a single point)
  that `use` does not produce; sourced from the clean, validated
  `grounding_resize` subset.
- **screen_parsing → understanding, normalized + trimmed** — a unique
  full-screen UIA element-listing task; rects clipped to the image and normalized
  to `[0, 1000]`, only `{control_text, control_rect}` kept so the answer matches
  the prompt. Controls with no on-screen area are dropped and counted; see
  [above](#out-of-bounds-control_rect--clip-or-drop) for why the rule is
  geometric rather than a pixel tolerance, and why it is not a `raise`.
- **use from `train` only**; `test` (benchmark) and `fail` (paper-excluded
  from SFT) are left out.
