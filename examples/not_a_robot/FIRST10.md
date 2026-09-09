# Reference-based first ten tasks

This experiment reimplements ten **synthetic game** tasks from the supplied
Neal captures. It does not operate real CAPTCHA services, bypass access gates,
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

The importer verifies the archive and each allowlisted file, never extracts
arbitrary members, and refuses to overwrite different user bytes. Its 15 files
are the 14 artwork files plus the level-eight successful instance's screenshot.
CSS displays only the car photo from that screenshot, with live HTML input and
Submit controls. The older base photo depicts a different car and is never
paired with this new instance's accepted answer. Redistribution rights are
unknown: access to the archive does not confer an artwork license.

| Key | Implemented observed mechanism | Unverified or inferred behavior |
| --- | --- | --- |
| `neal_01` | Checkbox/label click, loading, green check | Authored logo vector, precise click boundary, 650 ms loading |
| `neal_02` | One original photo, 4×4 grid, inset/check selection, captured accepted set | Other photos and signpost boundary rules |
| `neal_03` | Whole-raster distorted text, interference curves, input/Submit | Authored deformation and timing; audio unavailable; broader normalization |
| `neal_04` | Nine original photos and captured vegetable selection | Other image sets and classification rules |
| `neal_05` | Click-to-rotate tiles, captured initial orientation and accepted assembly | 160 ms transition and other initial arrangements |
| `neal_06` | O opens center, legal turns, both observed draw sequences, Verify/refresh | Opponent candidate, 350 ms reply; original winning route unverified |
| `neal_07` | Captured 10×10 letters and individual word-cell selection | Crossing words and random placement |
| `neal_08` | Successful instance's car photo, real input/Submit | Screenshot-derived image; broader whitespace/case behavior |
| `neal_09` | Quadtree split, depth-four selection toggles, captured 31-cell accepted set | Other photos, boundary tolerance and transition timing |
| `neal_10` | Real-time moles, original normal/hit sprite frames, five distinct hits then Verify | Spawn/lifetime timing, penalties and universal threshold |

The mole PNG is a 400×200 two-frame sprite sheet (normal left, hit right), not
a single normal-state image. Both frames are used; the earlier capture review's
claim that hit artwork was missing was incorrect.

`reference.fidelity="captured_instance"` identifies this limited evidence scope,
not pixel-perfect or generator-level parity. Limitations remain in task metadata.
The fixed captured challenges are not an original random distribution. `seed`
affects local dynamics, not a recovered original seed. Refresh restarts the
same captured instance and preserves episode mistakes/events. Arial/Georgia may
resolve to host fallback fonts; browser/font versions matter for visual parity.

The tic-tac-toe candidate takes an immediate win, otherwise blocks an immediate
X win, otherwise selects the first empty corner `[0,2,6,8]`, then the first empty
edge `[1,3,5,7]`. It reproduces both recorded games but is not a recovered original
algorithm. Do not weaken it to manufacture model success. Draws are not success;
the missing original winning route remains a fidelity gap despite green tests.
Exhaustive enumeration of the implemented legal turns gives **zero X-winning
paths**, 80 O-winning terminal paths and 14 draws. This is a partial reconstruction,
not a solvable benchmark task: a model's non-success here is not evidence of model
inability. Original normal-UI winning evidence is required to finish this level.

Moles spawn at inferred 850–1349 ms intervals and remain active for 1200–1999 ms.
Time continues during inference: stale screenshots may cause genuine missed
clicks. The windows are not stretched for Astra. Five distinct hits are the
captured instance's local rule, not a universal claim about the original game.

## Engineering tests versus model attempts

Use an authenticated Codex CLI with access to the requested model. The launcher
defaults to `gpt-6-astra` with `xhigh`; omit `--tasks` to attempt all ten:

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

Highest priority: a genuinely winning normal-UI level-six flow with its official
completion message. Next: continuous level-three/ten videos, rotation/split
transitions, refresh variants and clear rejected submissions. These independent
instances do not reproduce the main site's sequential 1–10 campaign. Levels
11–48 need their own evidence and implementation.
