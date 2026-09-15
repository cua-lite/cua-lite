# Reference-based first ten tasks

This document covers the first ten **synthetic game** reconstructions from the
supplied Neal captures. For the expanded catalog, see
[current support](/examples/not_a_robot/SUPPORT.md). The experiment does not
operate real CAPTCHA services, bypass access gates,
contain original game source, or implement all 48 levels. All changes are in
`examples/not_a_robot/`, reusing CUA-Lite registration, `gym.make`, canonical GUI
actions and episode recording. Core and Slime behavior are unchanged. The eight
older local exercises remain separate and are not counted as reconstructions.

## Import and run

Use an existing Python environment with this example's requirements and a
compatible Playwright Chromium. No training stack or GPU is required. Original
artwork is excluded from Git and must be imported privately:

```bash
uv run --no-project --python /path/to/python python \
  -m examples.not_a_robot.reference_assets /path/to/neal_reference_first10_supplement_20260909_030500.zip

uv run --no-project --python /path/to/python python \
  -m examples.not_a_robot.local_tasks --port 8765
```

Preview `http://127.0.0.1:8765/?task=neal_02&seed=0` on that host. A remote
host's loopback is not your laptop's localhost. The preview process is optional:
each environment reset owns its own listener, browser and recording directory.

```python
from examples.not_a_robot import registration
from lite import gym

env = gym.make(
    "visual_tasks@neal_02",
    seed=0,
    artifact_root="/path/to/new-artifacts",
    browser_executable="/path/to/chrome",
)
observation = await env.reset()
# Send canonical computer calls through env.step(...).
await env.close()
```

For existing entrypoints, set
`CUA_LITE_REGISTRATION_MODULES=examples.not_a_robot.registration`. The ten keys
are `visual_tasks@neal_01` through `visual_tasks@neal_10`. This increment supports
direct local environments, not production env-server workers, multi-user pooling
or a verified container image. A later container can package the same Python
runtime/browser and start the existing listener; it need not duplicate a service.

## Evidence and fidelity limits

Base archive SHA256:
`d48803abd8a7838745b97aad84e68f5a3eed28165f38f493d3876ea12ebac1e0`.
Supplement SHA256:
`a2900366ef7ef7ea18312b607d62b65e003290ff97a20d4318b00e5f740323d2`.
The supplement has 199 screenshots, 14 standalone artwork files and nine
recorded official-embed completion messages: levels 1–5 and 7–10. It contains no
continuous video or level-six winning completion.

The later incremental archive adds a visible level-six refresh-to-win flow and
second captured instances for levels seven and eight. These are visible outcomes,
not saved official completion messages. See
[incremental evidence and usage](/examples/not_a_robot/INCREMENTAL.md).

The importer verifies the archive and each allowlisted file, never extracts
arbitrary members, and refuses to overwrite different user bytes. Its 15 files
are the 14 artwork files plus the level-eight successful instance's screenshot.
CSS displays only the car photo from that screenshot, with live HTML input and
Submit controls. The older base photo depicts a different car and is never
paired with this new instance's accepted answer. Redistribution rights are
unknown: access to the archive does not confer an artwork license.

| Key | Implemented observed mechanism | Unverified or inferred behavior |
| --- | --- | --- |
| `neal_01` | Whole-card click, source-derived 700 ms green check and 1600 ms completion | Authored logo/mark animation, local terminal timer cleanup, missing original audio |
| `neal_02` | One original photo, 4×4 grid, inset/check selection, captured accepted set | Other photos and signpost boundary rules |
| `neal_03` | Whole-raster distorted text, interference curves, input/Submit | Authored deformation and timing; audio unavailable; broader normalization |
| `neal_04` | Nine original photos and source-defined required/optional labels with one combined selection error allowed | Game labels are not a general classifier; original-site boundary replay and exact selection artwork unverified |
| `neal_05` | Captured initial orientation, source-derived random Refresh, clockwise turns, 200 ms ease-out transition and modulo-360 verification | Fixed initial mount, local seeded stream, clipped tile artwork and missing original audio |
| `neal_06` | Source-derived opponent, 100 ms center opening, 450 ms replies, empty-board/X-first refresh and X-win Verify | Local seeded RNG, safe timer cancellation, authored marks/controls; original animation/audio and full runtime parity unverified |
| `neal_07` | Two captured initial boards plus source-derived eight-direction, maximum-overlap Refresh generation | Fixed initial mount, local seeded stream, authored grid styling and missing original audio |
| `neal_08` | Two separately selected car photos, real input/Submit, instance-specific strings with source-derived ASCII space/hyphen removal | Screenshot-derived images; original-site input-variant replay remains unavailable |
| `neal_09` | Quadtree split, depth-four selection and source-derived two-error tolerance around 31 positive leaves | Other photos, authored split animation and missing original audio |
| `neal_10` | Source-derived spawn timers, separate selection/hit state and at least five current hits then Verify; original sprite frames | Local RNG, off-grid source index, safe refresh cleanup; original sounds/pop animation and full runtime replay unavailable |

The mole PNG is a 400×200 two-frame sprite sheet (normal left, hit right), not
a single normal-state image. Both frames are used; the earlier capture review's
claim that hit artwork was missing was incorrect.

Catalog 0.7.10 uses level four's source module 1125 predicate. Required images
are carrot, onion, corn and potato; eggplant is optional. Missing required items
and wrong extra items share a total error budget of one. Thus the captured
three-item success remains valid but is not the only accepted set. These are
the game's labels, including the Mr. Potato Head image, not botanical claims.
The image order and historical capture provenance remain unchanged.

Level eight removes only ordinary spaces (`U+0020`) and ASCII hyphens (`U+002D`)
before a case-sensitive comparison. The default instance compares with `867V309`;
the incremental instance compares with `JHB007`. Nonbreaking spaces and Unicode
hyphens remain significant, and level three still compares its input exactly.
This rule is derived from reviewed source modules 1094 and 414 in the authored
spec bundle with SHA256
`2513409ca66c9f2ee15b3845bc66ec1c0064bedf6757d95f81d3c05f865455d4`.
It does not establish new original-site input trials or change historical
capture outcomes. Catalog version 0.7.1 identifies this input-rule refinement.

`reference.fidelity="captured_instance"` identifies this limited evidence scope,
not pixel-perfect or generator-level parity. Limitations remain in task metadata.
The fixed captured challenges are not an original random distribution. `seed`
affects local dynamics, not a recovered original seed. Refresh preserves episode
mistakes/events. Levels 05 and 07 generate new local-seeded arrangements using
source-derived algorithms; tic-tac-toe refresh clears the board for X to start
instead of repeating the initial O opening. Other fixed-instance refresh paths
retain their documented limits. Arial/Georgia may
resolve to host fallback fonts; browser/font versions matter for visual parity.

Catalog 0.7.9 derives the tic-tac-toe opponent from source module 1121. The first
board starts empty and O opens in the center after 100 ms. Replies after X moves
wait 450 ms. With legal cells remaining, every reply first consumes one random
draw. After at least one refresh, a draw below 0.4 selects uniformly from empty
cells using a second draw. Otherwise the opponent takes the first O win, first
X block, center, first corner in order 0/2/6/8, or first empty cell. Winning-line
priority follows the source's row/column/diagonal order, not empty-cell order.

Refresh clears the board, makes X start, increments the game count and continues
the local seeded stream; environment reset restarts both seed and game count.
Pending timers are cancelled on refresh/shutdown, intentionally not reproducing
the source's stale-callback race. Only a real X line followed by Verify succeeds;
losses and draws do not. No recorded move sequence or winning seed is injected.
Source-rule recovery does not establish original random-stream or audiovisual
parity. Historical results remain attached to their frozen earlier runtime.

Catalog 0.7.10 also derives mole behavior from modules 1127 and 379. The first
mole appears immediately; each next spawn is scheduled after `500 + 2000*r`
milliseconds, with a separate random draw for its location. Lifetime is fixed
when spawned: 1500, 1000, 750, 625 or 600 ms at zero through four current hits.
The literal source pool is 1–16 while the grid renders 0–15, so cell zero never
spawns a mole and source index 16 is off-grid. This discrepancy is preserved,
not silently corrected or claimed as an independently observed original-site run.

Selection is independent of a hit: an empty click toggles its checkbox without
awarding a hit or mistake, and clicking a hit again removes it. Preselection
must be toggled off and on to hit a newly appearing mole. At five hits the source
cancels only the latest hide and next-spawn timers, leaving other active moles
and earlier hide callbacks intact. A remaining mole can produce a sixth hit;
Verify accepts at least five current hits. Cancelling a hit can disable Verify
without restarting spawns. Refresh restarts play and clears all owned timers,
an explicit safe-lifecycle difference from the original's stale callbacks.

Time continues during inference; the source windows are not stretched for
Astra. The local random stream, authored sprite layout, missing 110 ms pop
animation and missing original sounds remain separate fidelity limits.

## Engineering tests versus model attempts

Use an authenticated Codex CLI with access to the requested model. The launcher
defaults to `gpt-6-astra` with `xhigh`. Always select tasks for a bounded smoke:
omitting `--tasks` selects all 46 registered Neal paths (25 captured-instance
paths and 21 local variants), not only ten. Level 39 remains unavailable-camera
handling only; registration does not establish full gameplay coverage.

```bash
uv run --no-project --python /path/to/python python \
  -m examples.not_a_robot.codex_smoke --tasks neal_01 neal_02 \
  --artifact-root /path/to/a-new-run-directory \
  --browser-executable /path/to/chrome \
  --model gpt-6-astra --reasoning-effort xhigh \
  --max-seconds 600 --max-steps 150
```

The artifact root must not exist. Each task has a fresh solver workspace and
browser. The client uses a read-only shell sandbox with shell, web, apps and
multi-agent tools disabled. Only the three reviewed `local_game` MCP tools are
enabled and explicitly approved for unattended operation. This permission is
limited to the owned local game, not a global approval bypass. MCP server code
runs outside the client shell sandbox; this is not a hostile-code OS sandbox.
No user configuration or credentials are rewritten. Model access must already
work through the installed CLI authentication.

Browser regressions use known reference inputs and visible DOM geometry as
scripted test oracles, executing normal canonical GUI actions. They test the
environment, not vision-model accuracy:

```bash
NEAL_BROWSER_EXECUTABLE=/path/to/chrome PYTHONDONTWRITEBYTECODE=1 \
  uv run --no-project --python /path/to/python python -m pytest \
  -p no:cacheprovider -o addopts= -m 'not stress' examples/not_a_robot/tests
```

The Codex MCP bridge offers only `get_observation`, `computer`, and `finish`.
The generic prompt permits screenshot-grounded retries, not blind answer enumeration.
The bridge uses no additional post-action settle sleep; explicit waits remain
available and each child action still has an archived screenshot. Feedback includes
remaining time/steps, screenshot acquisition start/end times and image age when
feedback is constructed. The same controller deadline governs feedback and
execution. Cached terminal responses retain their original measurement timestamp.
Animations continue during inference, so a low age in feedback does not guarantee that
the next model action will still match the visible dynamic state. Keyboard guidance
uses canonical lowercase tokens such as `["ctrl", "a"]`.
`get_observation` optionally accepts a normalized `[left, top, right, bottom]`
`region` and returns both the full screenshot and a magnified crop. This is a
nearest-neighbor enlargement of those same pixels (up to 4x, longest edge at most
1600 pixels), not DOM access or extra image detail. Actions always use the full
screenshot's coordinates. The crop, its source image hash, acquisition interval,
rectangle and next decision link are archived. This is needed because browser
zoom shortcuts do not change the page scale in the tested headless Chromium.
It returns screenshots and public feedback, never DOM, selectors, game source,
target sets or evaluation-only state. Supplied public action summaries, requested
actions, actual GUI primitives, observations, timestamps, errors, retries and
outcomes are archived. Private chain-of-thought is not requested or fabricated.

The outer runner saves exact CLI arguments, JSONL events, CLI errors and selected
model/effort evidence from its matching new thread's `turn_context`, without
exporting private reasoning. Bridge requested-model flags alone are not proof
of selected client configuration; neither is client evidence provider attestation.

Each attempt retains `events.jsonl`, content-addressed PNGs and a verified
`manifest.json`. Task success, recording completeness, CLI completion and cleanup
are distinct facts. An agent success claim cannot award reward. Failed/time-limited
attempts remain recorded, and retries get new directories. These archives are
not yet a canonical training-ready LiteSample/Parquet export.

The checked runs and their interpretation are recorded in
[the validation report](/examples/not_a_robot/VALIDATION.md).

The bridge finalizes the episode and closes its owned resources **before**
replying to `finish`; the client may immediately terminate its MCP process after
that reply. Repeated close/finish calls cannot append to a finalized episode.
An abruptly killed process can still leave an incomplete archive, which must not
be reported as a fully evaluated attempt.

## Remaining reference work

The new visible level-six win resolves the missing normal-UI flow, but a saved
official completion event, multiple opponent games and timing remain useful.
Other gaps include continuous level-three/ten videos, rotation/split transitions,
refresh variants and same-instance accepted/rejected text comparisons. The local
campaign controller and expanded task coverage are documented in
[SUPPORT.md](/examples/not_a_robot/SUPPORT.md); all 48 are not implemented.
