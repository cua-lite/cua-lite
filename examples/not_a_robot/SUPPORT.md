# Local Neal task support

The local catalog contains **46 Neal task paths**: 25 preserved captured-instance
paths plus 21 source-derived local mechanic variants. Levels **42 and 45 remain
unimplemented** because their dialogue backend is unavailable. Level 39 remains
limited to its unavailable-camera branch. Some spatial and media variants differ
materially from the original. See [LOCAL_EXPANSION.md](/examples/not_a_robot/LOCAL_EXPANSION.md).
An ordered campaign controller is available, but there is no completed 1-to-48
campaign result: it cannot advance beyond missing level 42.
See [CAMPAIGN.md](/examples/not_a_robot/CAMPAIGN.md) for its explicit mode and limits.
Eight older authored exercises are separate and are not
counted as reconstructed Neal levels.

This runs entirely in the existing CUA-Lite example. There is no new remote
service, database, GPU requirement or dependency on access to the Neal website.
Each evaluation owns a loopback listener, Playwright browser and trajectory
directory. A container can package that same runtime later; a verified container
image and production multi-user env-server integration are not implemented yet.

Level 44 bundles a pinned official Stockfish 17 Lite JS/WASM pair and chess.js
1.4.0. The browser runs the engine in a same-origin Worker, offline, at Skill Level
10 with a one-second search budget. A missing or failed engine is an explicit
error; no substitute opponent is selected. The server permits same-origin Workers
and WebAssembly compilation, not arbitrary JavaScript evaluation or external
connections. See [dependency licenses and provenance](/examples/not_a_robot/local/vendor/spatial/PROVENANCE.md).
The original chess-rule version and deployed WASM byte identity remain unverified.
Fatal local runtime errors produce `outcome="infra_error"` and a truncated attempt,
which is excluded from model success/failure evaluation even when its archive and
cleanup are complete. A genuine completed White checkmate does not require a
further engine reply before Verify can accept it.

## Private import and preview

Run from the repository root with an interpreter containing
[the example requirements](/examples/not_a_robot/requirements.txt):

```bash
uv run --no-project --python /path/to/python python \
  -m examples.not_a_robot.reference_assets \
  /path/to/neal_reference_first10_supplement_20260909_030500.zip

uv run --no-project --python /path/to/python python \
  -m examples.not_a_robot.reference_assets \
  /path/to/neal_levels31_48_reference_20260910_040601.zip

uv run --no-project --python /path/to/python python \
  -m examples.not_a_robot.reference_assets \
  /path/to/neal_all48_incremental_20260910_01.zip

uv run --no-project --python /path/to/python python \
  -m examples.not_a_robot.reference_assets \
  /path/to/module_1070.js

uv run --no-project --python /path/to/python python \
  -m examples.not_a_robot.local_tasks --port 8765
```

Open `http://127.0.0.1:8765/` on that host. Examples:
`/?task=neal_21&seed=0`, `/?task=neal_24&seed=0`, `/?task=neal_46&seed=0`.
For a remote host, use SSH port forwarding through your normal authorized route:

```bash
ssh -N -L 8765:127.0.0.1:8765 user@the-host-running-the-preview
```

Then open your computer's `http://127.0.0.1:8765/`. The preview binds only to
loopback, has no authentication layer and must not be exposed as a public server.
Ctrl+C stops this owned preview. Evaluations do not need the preview running.

The first ZIP imports 15 files; the second imports 49; the incremental ZIP adds
one level-eight reference screenshot. The second ZIP's basename says
31–48, but its reviewed tree also includes levels 11–30. The
[manifest](/examples/not_a_robot/reference_manifest.json) pins exact archive and
member hashes. Only 65 allowlisted images are imported, including one captured
inkblot for level 20, nine face images for level 37 and the level-47 cover.
Arbitrary ZIP paths,
capture logs, scripts and bundled executables are not extracted or served.
Different existing user bytes and symlinks are refused, not overwritten.
Artwork remains private and Git-ignored because redistribution rights are unknown.
Importing only one reviewed ZIP does not prevent unrelated already-imported tasks
from running; a task needing absent images fails with an explicit boot error.

The fourth input is the pinned, locally held analysis module, not a downloaded
script to execute. The importer derives only `level47_chart.json`, verifies its
SHA256 and preserves all 336 notes in source order. Neither the JavaScript module
nor other source files are copied or served. The chart is the 66th allowlisted
asset and stays private and Git-ignored. A missing chart produces an explicit
level-47 boot error, not a replacement generated chart. The original song and
video are still required for original-media fidelity.

Version 0.6.0 adds explicit `incremental` instances for 07 and 08, not two new
levels. Preview them with `/?task=neal_07&instance=incremental&seed=0` or
`/?task=neal_08&instance=incremental&seed=0`. Existing links and the campaign
retain the `default` instances. For `gym.make`, pass
`reference_instance="incremental"`; for the bounded Codex launcher, add
`--reference-instance incremental --tasks neal_07 neal_08`. Unknown instance/task
pairs and nondefault campaign instances are rejected. The seed is independent
of instance selection. See [INCREMENTAL.md](/examples/not_a_robot/INCREMENTAL.md)
for source limits and the evidence-based level-six opponent refinement.

## First expansion: twelve interactive tasks

| Levels | Added behavior | Important limits |
| --- | --- | --- |
| 11 | Responsive 25-by-25 Waldo image grid, source-derived two-required-cell predicate with at most one extra cell, refresh | Captured image; no original-site selection-variant replay; shared card chrome and checkmark artwork remain approximate |
| 12, 13, 31 | Matched-image grids, toggle selection, exact Verify, refresh | Fixed captured instances; refresh variant coverage and original-site boundary replay remain incomplete |
| 29 | Nine fixed source images, one combined selection error allowed, forbidden fourth image always rejected | Game-specific labels, not factual classification; original-site boundary replay and original selection artwork unverified |
| 14 | All 56 fixed statement cards; source-derived responsive 4/3/2/1-column embed grid and document scrolling; whole-card clicks, 700 ms green check/1600 ms completion and persistent 800 ms red-cross feedback | Authored branding/mark/spinner artwork, font fallback, missing original audio and local terminal timer cleanup; no original-site pixel-equivalence claim |
| 21 | Inventory/held stacks, right-click single placement, material-consuming recipes, pickaxe verification | Slot generalization and stack merge/swap are local inferences; not an exact inventory-engine recovery |
| 24 | Four successive visual-exam stages with live input/selection and Verify | Same captured instance, static recorded blur, untested normalization and color variants |
| 30 | Adjacent-empty-cell sliding puzzle; verify actual solved image | Captured initial permutation, inferred animation timing; no fixed action-count shortcut |
| 33 | Captured logo-glyph image with live input/Submit | Original capture quality retained; case/space normalization unverified |
| 34 | Rendered mathematical expressions and full numerical-order verification | Host font rendering, deselection and reset differ in unmeasured ways |
| 46 | Full-height 14×57 image grid, real scrolling, all eight source floor cells required with at most two extras | Original-site boundary replay and original selection artwork unverified |

Levels 12, 18, 24 and 33 use CSS crops of unchanged reference screenshots; the old
answers, surrounding capture UI and parent page are not shown in the game.
Downloaded assets from unrelated instances are not paired with these answers.
Level 29's labels are fictional game criteria, not factual claims about souls.
No task contacts a real CAPTCHA service or bypasses an access control.

## Dynamic and unavailable-device expansion

- 18: independent per-cell photo replacement, actual timers, pending/incorrect
  submission handling and no-target final verification; not a 30-click counter.
- 22: nine distinct moving/catchable ducks, individual return animations and
  all-caught verification; not nine arbitrary clicks. Motion continues during
  inference. Speed, hitboxes and animation parameters are explicitly inferred.
- 39: only the observed unavailable-camera Verify path; zero device requests,
  no face collection and no expression-recognition claim.

See [DYNAMIC.md](/examples/not_a_robot/DYNAMIC.md) for exact fixed local parameters,
reference boundaries, required supplements and continuous-input calibration.

`captured_instance` means an evidence-bounded reconstruction, not exact original
parity. Metadata records the attempt, source ZIP and remaining limitations.
Environment reset restores the initial instance; in-game Refresh follows each
task's rules. Levels 05/07 generate new arrangements, while level 06 clears the
board for X to start. `seed` does not select a recovered original random seed.
See [first-ten limits](/examples/not_a_robot/FIRST10.md)
and the [all-48 evidence and plan](/examples/not_a_robot/ALL48_PLAN.md).

## CUA-Lite and screenshot-only attempts

```python
from examples.not_a_robot import registration
from lite import gym

env = gym.make(
    "visual_tasks@neal_21",
    seed=0,
    artifact_root="/path/to/new-artifacts",
    browser_executable="/path/to/chrome",
)
observation = await env.reset()
# Send canonical computer calls with env.step(...).
await env.close()
```

For existing entrypoints, set
`CUA_LITE_REGISTRATION_MODULES=examples.not_a_robot.registration`.
For a small real Codex smoke, use existing CLI authentication and explicitly
select tasks (omitting `--tasks` attempts all 46 implemented entries):

```bash
uv run --no-project --python /path/to/python python \
  -m examples.not_a_robot.codex_smoke --tasks neal_21 neal_34 \
  --artifact-root /path/to/a-new-run-directory \
  --browser-executable /path/to/chrome \
  --max-seconds 300 --max-steps 100
```

The artifact root must not exist. Each task gets a fresh isolated client with
only `get_observation`, `computer` and `finish`. No DOM, source, evaluator answers,
shell or web tools are exposed to that client. The existing default requests
`gpt-6-astra` and `xhigh`; access must already work through the installed CLI.
No global authentication or configuration is changed.

`get_observation` and `computer` optionally accept
`sequence={"duration_seconds": 5, "interval_seconds": 0.1}`. A sequence returns
ordered full-viewport screenshots with actual capture timestamps. It can start
with the first observation or immediately after an action batch. Each additional
screenshot is an ordinary environment step; budget, time, terminal and campaign
boundaries still apply. A sequence is bounded to 10 seconds, 64 frames and 8 MiB
of PNG bytes, and cannot be combined with a magnified `region`. Late sampling
skips missed slots instead of claiming to reconstruct them. Check its stop reason
and returned frame count; nominal spacing is not a capture-time guarantee.

Trajectories archive screenshots/crops and hashes, public action summaries,
requested and executed mouse/keyboard primitives, timestamps, task/stage events,
rejections/retries and finalized outcome/cleanup. Private chain-of-thought is not
requested or reconstructed. JSONL plus content-addressed images is the current
format; canonical training export is separate work.

Scripted browser tests use known references and visible control geometry. They
check the environment, not a model's vision ability. Actual Codex success,
recording completeness and original-site fidelity are separate acceptance gates.
See [validation results](/examples/not_a_robot/VALIDATION.md).

## What follows

Keep building the mechanisms supported by evidence while missing captures arrive.
Replacement images and moving-object interaction now have captured-instance
implementations. Driving, drawing, rhythm, 3D scenes, dialogue judges and media each need their own
remaining rules/assets. Do not substitute arbitrary click counts or guessed
answers for those mechanisms. The full scope and supplemental capture requests
remain in [ALL48_PLAN.md](/examples/not_a_robot/ALL48_PLAN.md).
