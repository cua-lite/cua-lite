---
description: 
globs: lite/data/preproc/opencua/**
alwaysApply: false
---

# OpenCUA (AgentNet) Dataset Preprocessing Guide

## Overview

This document describes the preprocessing specifications for the AgentNet dataset (xlangai/AgentNet). The preprocessed data follows the canonical format defined in [`lite/data/preproc/AGENTS.md`](/lite/data/preproc/AGENTS.md) and is written to the `cua-lite/OpenCUA/` canonical layout via [`lite/data/staging`](/lite/data/staging.py).

For the ScaleCUA dataset guide (the reference implementation), see [`lite/data/preproc/scalecua/AGENTS.md`](/lite/data/preproc/scalecua/AGENTS.md).

**Related Files:**

| Type | File |
|------|------|
| Parent specification | [`lite/data/preproc/AGENTS.md`](/lite/data/preproc/AGENTS.md) |
| Adapter shared helpers | [`utils.py`](/lite/data/preproc/opencua/utils.py) |
| Tool definitions | [`lite/core/tools/action_space/base.py`](/lite/core/tools/action_space/base.py) (`LiteDesktopActionSet`) |
| Preprocessing script | [`use.py`](/lite/data/preproc/opencua/use.py) |
| HF repo metadata | [`repo.json`](/lite/data/preproc/opencua/repo.json) |
| Shell scripts | [`scripts/download_raw_data.sh`](/lite/data/preproc/opencua/scripts/download_raw_data.sh) (download), [`scripts/process_raw_data.sh`](/lite/data/preproc/opencua/scripts/process_raw_data.sh) (merge+extract images), [`scripts/process_data.sh`](/lite/data/preproc/opencua/scripts/process_data.sh) (run Python pipeline) |

**Source Data Locations:**

| Dataset | JSONL Path | Images Path |
|---------|------------|-------------|
| Ubuntu | `${CUA_LITE_RAW_DATASETS_ROOT}/xlangai/AgentNet/agentnet_ubuntu_5k.jsonl` | `${CUA_LITE_RAW_DATASETS_ROOT}/xlangai/AgentNet/ubuntu_images/` |
| Windows/Mac | `${CUA_LITE_RAW_DATASETS_ROOT}/xlangai/AgentNet/agentnet_win_mac_18k.jsonl` | `${CUA_LITE_RAW_DATASETS_ROOT}/xlangai/AgentNet/win_mac_images/` |

Both subsets are joined against a third file from the same snapshot,
`${CUA_LITE_RAW_DATASETS_ROOT}/xlangai/AgentNet/meta_data_merged.jsonl`, keyed on
the same `task_id`. It is where `metadata.others.os` comes from — see
[Step 2](#step-2-resolve-images-and-read-the-os).

---

## Output Directory Structure

Output follows the canonical cua-lite layout:

```
${CUA_LITE_DATASETS_ROOT}/cua-lite/OpenCUA/
├── images/<hash[:2]>/<hash>.png        # content-addressed image store
└── desktop/
    └── use/
        ├── train/
        │   ├── ubuntu.parquet          # Ubuntu trajectories
        │   └── win_mac.parquet         # Windows/Mac trajectories
        └── validation/
            ├── ubuntu.parquet
            └── win_mac.parquet
```

`ubuntu` and `win_mac` are **variants** — semantically distinct sub-sources (different OS families) kept separately countable in stats and independently addressable.

---

## Data Statistics

| Subset | Records | Images (total) | Avg steps/traj |
|--------|---------|----------------|----------------|
| Ubuntu | 5,000 | 82,448 | 16.5 |
| Windows/Mac | 17,625 | 339,005 | 19.2 |
| **Total** | **22,625** | **421,453** | **18.6** |

A trajectory is dropped **whole** or kept whole — never repaired. Three rules do
all the dropping, and each is stated by the boundary that decides it rather than
by a count, because a count here rots the moment the snapshot or the adapter
moves (see [Error Handling](#error-handling) for where each one raises):

| Skip rule | The boundary that decides it |
|-----------|------------------------------|
| `value.code` does not parse | Any `AgentNetCodeParseError` — a malformed `pyautogui.press` `keys` arg, or a source coordinate outside the normalized `[0, 1]` range, which `_norm01_to_0_1000` refuses rather than clamps — skips the whole trajectory. Dropping just the offending step is not an option: its screenshot is the next step's `role:"tool"` result. |
| Missing / unresolvable image | One bad image skips the trajectory; `images` must stay 1:1 with `traj`. |
| Out-of-bound coordinate | `has_oob_coordinate` drops the staged entry in `main()`, which prints its own count. Distinct from the range check above. On the 2026-08-16 full run it fires for **no** record: the one out-of-range Ubuntu trajectory is already refused at parse time. |

**The counts belong to the source snapshot, not to this document.** Every run
prints `Processed: N, Skipped: M` per subset (per-trajectory reasons under
`--verbose`) plus the out-of-bound line, and the [Reproduction](#reproduction)
block below counts the published rows. A trajectory with no terminal
`terminate` is not a skip case: its final executable `computer` action is kept as
the EOF SFT label, with no synthetic tool result and no structural `Done.` behind
it.

### Quality field distributions (among kept rows)

Counted from a full run against the public-release key contract (4,999 Ubuntu
and 17,337 Windows/Mac rows). The release restricts stored key glyphs to visible
ASCII, so 43 previously published Windows/Mac trajectories containing source
spellings such as `å`, `√`, `ç`, or `ł` are now rejected. These counts move with
the source snapshot and skip rules — re-derive them from the output rather than
trust them:

| Field | Ubuntu | Windows/Mac |
|-------|--------|-------------|
| `task_completed=True` | 2,315 | 13,539 |
| `task_completed=False` | 357 | 3,798 |
| `task_completed=null` | 2,327 | 0 |
| `alignment_score` mean | 9.22 (2,672 present) | 8.93 |
| `alignment_score ≤ 3` | 59 | 326 |
| `efficiency_score` mean | 7.47 (2,672 present) | 7.60 |
| `efficiency_score ≤ 3` | 36 | 560 |
| `domain` | 12 categories | absent (all `null`) |

These fields are preserved in `metadata.others` for downstream filtering — no quality-based filtering is applied during preprocessing.

### Reproduction

```bash
uv run python -c "
import json, pyarrow.parquet as pq
from pathlib import Path
from lite.data.staging import iter_partitions
root = Path('${CUA_LITE_DATASETS_ROOT}/cua-lite/OpenCUA')
for *_, f in iter_partitions(root):
    t = pq.read_table(f)
    meta = json.loads(t.to_pydict()['metadata'][0])
    print(f'{f.relative_to(root)}: {t.num_rows} rows, os={meta[\"others\"][\"os\"]}')
"
```

---

## Source Data Format

Each JSONL line contains a complete trajectory record with the following fields:

### Record-level fields

| Field | Type | Mapped to | Notes |
|-------|------|-----------|-------|
| `task_id` | `str` | `metadata.others.source_id`; base for `metadata.others.id` | Upstream id; the first occurrence keeps the base id and later occurrences get `_record_<line_no>`, so ids stay unique without changing under `--head` |
| `instruction` | `str` | `messages[0]` user text | Task instruction |
| `traj` | `list[step]` | `messages` + `images` | See step-level fields below |
| `domain` | `str \| null` | `metadata.others.domain` | Ubuntu: app name (Chrome, VScode, Gimp, …); win_mac: absent |
| `task_completed` | `bool \| null` | `metadata.others.task_completed` | Whether the agent completed the task |
| `alignment_score` | `int 0-10 \| null` | `metadata.others.alignment_score` | How well agent actions aligned with the goal |
| `efficiency_score` | `int 0-10 \| null` | `metadata.others.efficiency_score` | How efficiently the agent completed the task |
| `task_difficulty` | `int \| null` | `metadata.others.task_difficulty` | Annotator-assigned difficulty |
| `natural_language_task` | `str` | ✕ dropped | Paraphrase of `instruction` |
| `actual_task` | `str` | ✕ dropped | Detailed task description |
| `reason` | `str` | ✕ dropped | Annotator's scoring rationale |

**`domain` values (Ubuntu only):** Chrome, VScode, libreoffice_impress, OS, Gimp, libreoffice_writer, MultiApp, Thunderbird, libreoffice_calc, infeasible, VLC, error_correction. The win_mac subset does not have this field.

### Step-level fields (`traj[i]`)

| Field | Type | Mapped to | Notes |
|-------|------|-----------|-------|
| `image` | `str` | `images[]` (via ImageStore) | Screenshot filename, hashed into CA store |
| `value.thought` | `str` | assistant content `inline_reasoning` | Agent's reasoning about current state |
| `value.action` | `str` | assistant content `action_description` | Human-readable action description |
| `value.code` | `str` | assistant `tool_calls[]` (AST parsed) | pyautogui/computer API call string |
| `value.observation` | `str` | assistant `metadata.data.opencua_step.observation` | Environment state description |
| `value.reflection` | `str` | assistant `metadata.data.opencua_step.reflection` | Agent's reflection on previous action |
| `value.last_step_correct` | `bool` | assistant `metadata.data.opencua_step.last_step_correct` | Annotator label |
| `value.last_step_redundant` | `bool` | assistant `metadata.data.opencua_step.last_step_redundant` | Annotator label |
| `index` | `int` | ✕ dropped | Step ordinal (replaced by `enumerate`) |

**Step sidecars**: `observation`, `reflection`, `last_step_correct`, and
`last_step_redundant` are preserved in one standard assistant metadata content
part:
`{"type": "metadata", "data": {"opencua_step": {...}}}`. Metadata content is a
documented canonical side-channel and is stripped before model rendering, so the
sidecars stay available for analysis without expanding the assistant message
schema or inventing new content types.

### Full raw record example (step 0 unabridged)

```json
{
  "task_id": "0030dc52-2a4a-4c0e-895b-48284c200efe",
  "instruction": "Open the Pikachu picture on the desktop using GIMP, and then select the \"Venetian Blinds\" filter in the animation.",
  "task_completed": false,
  "alignment_score": 6,
  "efficiency_score": 9,
  "task_difficulty": 4,
  "reason": "The agent successfully completed the first part of the task by efficiently opening GIMP, loading the pikachu.jpeg file from the desktop, and navigating to the Animation filters menu. All steps were logical and executed without redundancy. However, the task could not be completed because the 'Venetian Blinds' filter does not exist in GIMP's Animation category. The agent correctly identified this limitation and appropriately terminated the task rather than continuing futilely.",
  "natural_language_task": "Open the image editing software GIMP, load the pikachu.jpeg file from the desktop, then navigate to the Filters menu and look for a 'Venetian Blinds' filter in the Animation section to apply it to the image.",
  "actual_task": "Launch GIMP application, open the pikachu.jpeg file from the desktop using the file dialog, navigate to Filters > Animation menu to locate the Venetian Blinds filter, and discover that this specific filter is not available in GIMP's Animation category.",
  "domain": "infeasible",
  "traj": [
    {
      "index": 0,
      "image": "30a6f01b-9daf-4107-a755-a4a602d0de8e.png",
      "value": {
        "thought": "I need to start working towards the goal of opening the Pikachu picture on the desktop using GIMP and then applying the \"Venetian Blinds\" filter. Looking at the desktop, I can see there's a file called \"pikachu.jpeg\" which appears to be the Pikachu picture I need to open. To accomplish this task, I first need to open GIMP. I can see various application icons in the left taskbar, and I should look for GIMP among them. The logical first step would be to launch GIMP, which I can do by clicking on the GIMP icon if it's visible in the taskbar, or I might need to access it through the applications menu. Looking at the taskbar, I can see what appears to be an image editing application icon that could be GIMP. The next action should be to click on this application icon to launch GIMP, which will then allow me to open the Pikachu image file and proceed with applying the Venetian Blinds filter.",
        "action": "Click on the GIMP application icon in the left taskbar to launch the image editing software.",
        "code": "pyautogui.click(x=0.018, y=0.508)",
        "observation": "The current screen shows a desktop environment with a blue background featuring a large Pikachu character and the Pokémon logo. On the left side, there is a vertical taskbar containing various application icons and file shortcuts. The desktop displays multiple files and folders including xlsx files, text documents, images, and applications. Notable items visible include \"charizard.jpeg\", \"pikachu.jpeg\", \"Untitled 1.txt\", \"writer\", \"Chinese Writer.xlsx\", and various other files. There's also a \"Kpop Girl Groups\" folder and several application icons like LibreOffice, VLC media player, and others. The taskbar on the left contains system applications including what appears to be a terminal icon, file manager, and other system utilities. The overall layout suggests this is a Linux-based desktop environment, likely GNOME, with the Activities overview or similar interface active.",
        "reflection": "The last action successfully launched GIMP (GNU Image Manipulation Program) as evidenced by the application window now being open in the center of the screen. The GIMP interface is clearly visible with its characteristic dark gray theme, showing the main canvas area, toolbox on the left with various editing tools, and panels on the right including brush selection and layer options. The application title bar shows 'GNU Image Manipulation Program' confirming that GIMP has been properly launched. The desktop background with the Pokemon logo and Pikachu character is still visible behind the GIMP window, and I can still see the desktop files including the 'pikachu.jpeg' file that needs to be opened next. The taskbar on the left shows that GIMP is now the active application. This step was successful in opening the image editing software needed to complete the task.",
        "last_step_correct": true,
        "last_step_redundant": false
      }
    }
  ]
}
```

---

## Action Space & Mapping

### Source Action Space (pyautogui/computer API)

```python
# Mouse actions
pyautogui.click(x=0.5, y=0.5)           # Left click at normalized coords
pyautogui.rightClick(x=0.5, y=0.5)      # Right click
pyautogui.middleClick(x=0.5, y=0.5)     # Middle click
pyautogui.doubleClick(x=0.5, y=0.5)     # Double click
pyautogui.tripleClick(x=0.5, y=0.5)     # Triple click
pyautogui.moveTo(x=0.5, y=0.5)          # Move cursor
pyautogui.dragTo(x=0.5, y=0.5, button="left")  # Drag to position

# Scroll actions
pyautogui.scroll(3)                      # Scroll up (positive)
pyautogui.scroll(-3)                     # Scroll down (negative)
pyautogui.hscroll(3)                     # Horizontal scroll

# Keyboard actions
pyautogui.hotkey(["ctrl", "c"])          # Key combination
pyautogui.press("enter")                 # Single key press
pyautogui.write(message="hello")         # Type text
pyautogui.typewrite(message="hello")     # Type text (alias)

# Utility actions
computer.wait(2)                         # Wait seconds
computer.terminate(status="success")     # End task
```

### Mapping

Source coordinates use normalized values in `[0, 1]` range. Target coordinates are scaled to `[0, 1000]` range.

| Source Action | Target Action |
|--------------|---------------|
| `pyautogui.click(x=0.5, y=0.5)` | `click(coordinate=[500, 500])` |
| `pyautogui.rightClick(x, y)` | `click(coordinate=[...], button="right")` |
| `pyautogui.middleClick(x, y)` | `click(coordinate=[...], button="middle")` |
| `pyautogui.doubleClick(x, y)` | `click(coordinate=[...], clicks=2)` |
| `pyautogui.tripleClick(x, y)` | `click(coordinate=[...], clicks=3)` |
| `pyautogui.moveTo(x, y)` | `mouse_move(coordinate=[...])` |
| `pyautogui.dragTo(x, y, button)` | `drag(coordinate=[...], button=...)` |
| `pyautogui.scroll(n)` | `scroll(direction="up"/"down", amount=abs(n))` |
| `pyautogui.hscroll(n)` | `scroll(direction="left"/"right", amount=abs(n))` |
| `pyautogui.hotkey([keys])` | `key(keys=[...])` |
| `pyautogui.press(key, presses=N)` / `press(keys=key, ...)` | N ordered `key(keys=[normalized_token])` actions; raw aliases normalize before storage |
| `pyautogui.write(message=text)` | `type(text=text)` |
| `computer.wait(t)` | `wait(duration=t)` |
| `computer.terminate(status=s)` | dropped from `messages`; non-success `s` recorded as `others.terminate_status` |

`hotkey`/`press` validate every source token before routing through
`LiteDesktopActionSet.key(...)`: only one literal printable character or a
lowercase named key is publishable. This prevents schema-valid strings such as
`"ac"` or `"8000"` from surviving until a backend rejects them. Stored keys are
lowercase named keys or literal printable glyphs — see the
[Lite key vocabulary](/lite/data/preproc/AGENTS.md#keyboard-keys--lowercase-named-keys-plus-printable-glyphs).
AgentNet's one observed `num*` spelling is normalized source-locally to the
literal `*`: its successful trajectory and following screenshot both show that
the numeric-keypad multiplication operator was entered. The alias is not added
to the shared key vocabulary because it is not a PyAutoGUI key name outside
this source dialect.

Every action in this table renders back to OpenCUA's own wire: the whole emitted
vocabulary round-trips through the upstream OpenCUA desktop action-space text
format. Both scroll axes are covered — `pyautogui.scroll` carries the vertical
one and `pyautogui.hscroll` the horizontal one — so a
`scroll(direction="left"|"right")` from `hscroll` is not a dead end.

This repo hosts the OpenCUA *dataset* preprocessor only; the OpenCUA model
family was retired, so there is no in-repo action space to replay the table
against and the round-trip claim above is an upstream-format statement with no
local verifier. What IS pinned locally is this preprocessor's own output
contract, in `tests/data/preproc/opencua/test_preproc_opencua_tool_io_contract.py`.

---

## Processing Workflow

### Step 1: Parse Source JSONL

Read each line from the source JSONL files (`agentnet_ubuntu_5k.jsonl`, `agentnet_win_mac_18k.jsonl`) and parse the trajectory record.

### Step 2: Resolve Images and Read the OS

- Resolve each image path via `resolve_path(rel_path, "CUA_LITE_RAW_DATASETS_ROOT")`. If **any** image in a trajectory is missing or invalid, skip the **whole trajectory**. Assert `len(images) == len(traj)` (one image per step).
- **OS**: read, never inferred. `load_os_by_task_id()` loads
  `meta_data_merged.jsonl` once per run and maps AgentNet's own `system` field to
  the canonical `others.os` vocabulary:

| Source `system` | `others.os` |
|-----------------|-------------|
| `Windows` | `"windows"` |
| `Darwin` | `"macos"` |
| `Ubuntu` | `"ubuntu"` |

  Both lookups are total, so an unmapped `system` spelling or an uncovered
  `task_id` raises instead of publishing a guess — and `others.os` is therefore
  never `null` for this dataset. The `win_mac` subset genuinely mixes two systems
  and its records carry **no other signal** that separates them, which is why the
  sidecar is joined rather than approximated from the instruction text or the
  image path. The `ubuntu`/`win_mac` split of the source files plays no part in
  the OS value.

### Step 3: Validate and Convert Actions

For each trajectory step:
1. Parse the `code` field using Python AST
2. Convert pyautogui/computer calls to CUA-lite tool calls (see [Action Space & Mapping](#action-space--mapping))
3. Normalize coordinates from `[0, 1]` to `[0, 1000]`

### Step 4: Build Messages

Create multi-turn conversation format:

```
msg[0]  user:      [image(idx=0), text(instruction)]
msg[1]  assistant: [inline_reasoning, action_description, metadata(opencua_step)]
                   + tool_calls=[computer(actions=[...])] (id=call_0000)
msg[2]  tool:      tool_call_id=call_0000, content=[image(idx=1)]
msg[3]  assistant: [inline_reasoning, action_description, metadata(opencua_step)]
                   + tool_calls=[computer(actions=[...])] (id=call_0001)
msg[4]  tool:      tool_call_id=call_0001, content=[image(idx=2)]
  ...
final assistant:
  - content=[{"type": "text", "text": "Done."}] when terminal cleanup leaves no
    executable EOF action
  - or tool_calls=[computer(actions=[...])] when a final executable action remains
```

- First user message = screenshot + task instruction
- Every subsequent screenshot is a `role:"tool"` message keyed by the `tool_call_id`
  of the `computer` batch it answers — **not** a `role:"user"` turn. The rewrite
  happens in [`finalize_use_messages`](/lite/data/utils/messages.py).
- Actions are always nested in a `computer` batch (`arguments.actions`),
  never emitted as bare top-level calls.
- Every assistant message has:
  - **Training-active content**: `inline_reasoning` (from `thought`), `action_description` (from `action`), `tool_calls` (from `code`)
  - **Metadata sidecar content**: one optional `metadata` part containing `data.opencua_step` with `observation`, `reflection`, `last_step_correct`, and `last_step_redundant`
- A source terminate is never persisted as a tool_call. If popping it leaves no
  executable EOF action, the trajectory ends on the content-only `Done.` final; a
  non-success status is preserved as `metadata.others.terminate_status` /
  `terminate_reason`. If there is no source terminate, or executable calls remain
  after terminal cleanup, the final executable action stays as the EOF SFT label.
- **No `reasoning_content`** — AgentNet is not a native-thinking model. The `thought` field is a structured SFT protocol, mapped to `inline_reasoning`

### Step 5: Stage Entry

For each well-formed trajectory:
1. Check for out-of-bound coordinates (`has_oob_coordinate`) — drop if any
2. Hash images into `ImageStore` via `stage_entry()`
3. Assign train/validation split via `SplitAssigner`
4. Buffer by `(metadata.dims[0], metadata.dims[1], split, variant)` key

### Step 6: Flush Buffers

Write all partition buffers to canonical parquet paths via `staging.flush_buffers()`.

---

## Error Handling

Following the guidelines in [`lite/data/preproc/AGENTS.md`](/lite/data/preproc/AGENTS.md).

| Error Type | Behavior |
|------------|----------|
| Missing image file (any image in trajectory) | Skip whole trajectory with `SkipTrajectoryError` |
| Corrupt/unreadable image | Skip whole trajectory at staging, count it, and continue |
| JSON parse error | Raise `ValueError` |
| Invalid code format | Skip trajectory with `SkipTrajectoryError` |
| Terminal final action | Drop the source `terminate` turn and append structural `Done.` when no executable calls remain; do not synthesize schema-less terminate |
| **No** terminal final action | Keep the final executable action as the EOF SFT label; do not append synthetic `Done.` |
| Out-of-bound coordinates | Drop trajectory (counted, printed at end) |
| Ambiguous/malformed data | Skip trajectory with `SkipTrajectoryError` |

---

## Output Format

See [`lite/data/preproc/AGENTS.md`](/lite/data/preproc/AGENTS.md) for the complete output format specification.

Actions are **never** bare top-level calls: every step's actions are
nested inside a `computer` batch wrapper (`platform="desktop"`), and each
post-action screenshot is a `role:"tool"` message carrying the `tool_call_id` of
the call it answers — never a `role:"user"` turn. The per-step sidecars ride
alongside on the assistant turn as one standard `metadata` content part under
`data.opencua_step`.

Example output record (real row `agentnet_ubuntu_0211fa90-…`; long text and
image hashes truncated):

```json
{
  "images": [
    "cua-lite/OpenCUA/images/57/57646f0adf26efda4…png",
    "cua-lite/OpenCUA/images/f5/f57aef34512f84a64…png",
    "cua-lite/OpenCUA/images/ce/ce0e3b11fe585f2de…png"
  ],
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "index": 0},
        {"type": "text", "text": "Open \"Eating Grapes\" on the desktop using writer"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "inline_reasoning", "text": "I need to open \"Eating Grapes\" using Writer, but I'm currently looking at a …"},
        {"type": "action_description", "text": "Click on the LibreOffice Writer icon in the left taskbar to launch the Write…"},
        {"type": "metadata", "data": {"opencua_step": {
          "observation": "The current screen shows a file browser dialog window titled \"Browse for mor…",
          "reflection": "The last action successfully launched LibreOffice Writer. The visual change …",
          "last_step_correct": true,
          "last_step_redundant": false
        }}}
      ],
      "tool_calls": [
        {
          "id": "call_0000",
          "type": "function",
          "function": {
            "name": "computer",
            "arguments": {"actions": [{"action": "click", "coordinate": [23, 442]}]}
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_0000",
      "content": [{"type": "image", "index": 1}]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "inline_reasoning", "text": "LibreOffice Writer has successfully launched and is now displaying a blank d…"},
        {"type": "action_description", "text": "Press Ctrl+O to open the file browser dialog for opening an existing documen…"},
        {"type": "metadata", "data": {"opencua_step": {
          "observation": "LibreOffice Writer is now successfully open and active, displaying a blank d…",
          "reflection": "The Ctrl+O keyboard shortcut was successful in opening the file browser dial…",
          "last_step_correct": true,
          "last_step_redundant": false
        }}}
      ],
      "tool_calls": [
        {
          "id": "call_0001",
          "type": "function",
          "function": {
            "name": "computer",
            "arguments": {"actions": [{"action": "key", "keys": ["ctrl", "o"]}]}
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_0001",
      "content": [{"type": "image", "index": 2}]
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "Done."}]
    }
  ],
  "metadata": {
    "metadata_kind": "cua",
    "dims": ["desktop", "use"],
    "extra_tool_schemas": [],
    "valid_actions": null,
    "others": {
      "id": "agentnet_ubuntu_0211fa90-6c77-48bc-b560-7a56093101ed",
      "resolution": [1920, 1080],
      "os": "ubuntu",
      "source": "xlangai/AgentNet",
      "source_id": "0211fa90-6c77-48bc-b560-7a56093101ed",
      "domain": "infeasible",
      "task_completed": false,
      "alignment_score": 7,
      "efficiency_score": 9,
      "task_difficulty": 3,
      "terminate_status": "failure"
    }
  }
}
```

Note `others.terminate_status`: a `use` row never persists a `terminate`
tool_call. This example ends on the content-only `Done.` final because terminal
cleanup leaves no executable EOF action, and a **non-success** source terminate
is preserved in `metadata.others` via
[`terminate_outcome_others`](/lite/data/utils/messages.py) (with
`terminate_reason` when the source authored non-blank text). A success terminate
records nothing — absence of the key means "not a self-reported failure".
