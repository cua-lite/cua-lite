# Not a Robot experiment

## Reference-based local levels

The local catalog contains 25 implemented captured-instance task paths:
levels 01–14, 18, 21–22, 24, 29–31, 33–34, 39 and 46. Level 39 covers only the
recorded unavailable-camera branch, not expression recognition.
They are separate from the eight earlier
authored exercises below. See the
[current support and launch instructions](/examples/not_a_robot/SUPPORT.md),
[first-ten evidence](/examples/not_a_robot/FIRST10.md), and
[remaining all-48 plan](/examples/not_a_robot/ALL48_PLAN.md).
Original artwork is imported privately with `reference_assets.py`, not included
in Git. Local tasks contact no external site.

An experimental [local campaign controller](/examples/not_a_robot/CAMPAIGN.md)
preserves one browser and trajectory across ordered levels. It stops at missing
implementations; it is not a completed all-48 game or campaign result.

Level-six opponent behavior remains inferred; new supplied captures show a normal
refresh-to-win route, and the seeded local candidate fits the observed replies.
Explicit second instances for 07/08 are documented in
[INCREMENTAL.md](/examples/not_a_robot/INCREMENTAL.md); defaults stay unchanged.
Dynamic timing and unobserved matching boundaries remain approximate. These are
explicit fidelity limits, not claims of complete original-site parity.

## Self-hosted tasks

Alongside the reference reconstructions, `local/` contains five original exercises and three locally authored
adaptations of Neal-style game mechanics. It is **not** the official game,
an exact replica, or an implementation of the full 48-level inventory.
The original exercises are shape matching (`click`), code entry (`input`), block placement
(`drag`), ordered number selection (`sequence`), and a moving target (`moving`).
All instructions, scripts, styles, and shapes are bundled locally.

### Neal-inspired mechanic adaptations

| Local task key | Reference mechanic | Deliberate local differences |
| --- | --- | --- |
| `visual_tasks@checkbox` | Level 1: Checkbox | Locally drawn checkbox, no vendor logo/loading flow; completes one independent episode. |
| `visual_tasks@stop_signs` | Level 2: Stop Signs | Nine separately drawn SVG scenes, four STOP signs, seeded arrangement; not the original photo/crop grid. |
| `visual_tasks@wiggles` | Level 3: Wiggles | Seeded six-letter uppercase code with independent CSS motion; not the original GIFs, strings or timing. |

The starting mechanic is visible in the [official game listing](https://neal.fun/not-a-robot/).
Descriptions for [Stop Signs](https://www.followchain.org/level-2-im-not-a-robot/)
and [Wiggles](https://www.followchain.org/level-3-im-not-a-robot/) are public
walkthrough references, not official specifications or proof of live parity.
These earlier adaptations do not import official game code or artwork. Each
adapted task carries a public `reference` annotation with
`fidelity="mechanic_adaptation"` in catalog and trajectory metadata.

The checkbox succeeds only when activated, not when its neighboring caption is
clicked. Stop Signs requires the exact selection on Verify; empty, incomplete,
or extra selections fail without revealing which tiles are wrong, and can be
edited. Wiggles requires the exact uppercase code; incorrect entries are
retryable. These are local benchmark rules, not claims about every detail of
the corresponding Neal levels. The checkbox layout is fixed; seeds affect the
grid and animated-letter tasks.

### Preview and run

Preview the pages from this checkout with an outside-repository interpreter:

```bash
uv run --no-project --python /path/to/python python \
  -m examples.not_a_robot.local_tasks --port 8765
```

Open `http://127.0.0.1:8765/` on the machine running that command. The gallery
selects the task and seed; a direct URL is `/?task=drag&seed=17`. The listener
binds only to loopback and serves allowlisted bundled files and verified private
reference assets when imported. It has no login,
database, external asset requests, or directory listing. Ctrl+C closes it.
On a remote cluster, its loopback URL is not your laptop's localhost.

In addition to the three keys above, the original environments are
`visual_tasks@click`, `visual_tasks@input`, `visual_tasks@drag`,
`visual_tasks@sequence`, and `visual_tasks@moving`. For a new adapted task:

```bash
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
uv run --no-project --python /path/to/python python \
  -m examples.not_a_robot.smoke --mode local --task stop_signs --seed 17 \
  --artifact-root /path/to/new-artifacts --browser-executable /path/to/chrome
```

Each reset owns a new loopback server, browser context, and attempt archive.
The adapter starts and stops these resources; no separate preview process or
container daemon is required. Agent workers use the same registration-module
environment variable described below. Production env-server workers and a
dedicated container image are not implemented by this increment.

The game owns a narrow status bridge containing task identity, seed/version,
progress, mistakes, outcome, elapsed time, and gameplay events. Only the
evaluator reads this bridge. The agent gets screenshots, visible instructions,
and canonical computer actions, never hidden solution values or JS execution.
A model's `terminate(success)` alone still receives reward zero. Success or
task failure terminates the episode; environment budgets truncate it. Incorrect
submissions are retryable. The moving target fails 20 seconds after
Start and keeps running while the model is thinking.

Seeds reproduce generated codes, layouts and animation parameters at the same
viewport/browser version, not the exact frame of a real-time animation,
wall-clock interaction timing, or arbitrary pixel-perfect replays. The gallery's
restart control is hidden in agent runs. Recordings retain game asset hashes,
browser version, seed, clock policy, gameplay mistakes, screenshots and executed
inputs alongside existing controller events. The Codex MCP controller supports
separate screenshot-only attempts; training-format export remains separate work.

Run the bounded browser tests using the existing external runtime and a new
temporary artifact directory. These tests contact only their owned localhost
servers; `live` here means a real browser, not the Neal website:

```bash
NEAL_BROWSER_EXECUTABLE=/path/to/chrome PYTHONDONTWRITEBYTECODE=1 \
  uv run --no-project --python /path/to/python python -m pytest \
  -p no:cacheprovider -o addopts= -m 'not stress' \
  --basetemp=/path/to/new-test-artifacts examples/not_a_robot/tests
```

Browser integration tests use visible DOM controls as scripted test oracles.
They are environment tests, not evidence of vision-model task accuracy. The
stdin bridge is available for separate screenshot-only interaction tests.

## Existing Neal and fixture modes

This out-of-tree environment uses CUA-Lite's existing `gym.make`, metadata,
canonical `computer` actions, feedback and reset/step contracts. It does not
modify core environments, rollout, or training. No new HTTP server is required.

## Remote and fixture scope

- `not_a_robot@level_001`: fresh browser, level-one start, success only upon an
  observed transition to level two.
- `not_a_robot_campaign@full_game`: fresh browser, sequential interaction and
  progress recording. Full-game completion grading is **not implemented**;
  `terminate(status="success")` is only an agent claim, never an evaluator reward.
- `mode="fixture"`: a local two-stage synthetic control, not Neal's game. It
  tests checkbox, keyboard and drag interaction without network access.
- `catalog.py` reserves 48 remote level entries. Remote levels 2–48 do **not**
  have independent reset or verified graders. This remote inventory is separate
  from the implemented `visual_tasks@neal_*` local catalog described above.

For live mode, only the synthetic Neal game is in scope. External network origins,
off-game navigations, Cloudflare challenge assets, service workers and
WebSockets are blocked. Explicit access gates stop interaction, including a
fresh check before each mouse/keyboard primitive. This is a conservative browser
policy, not an OS-level network sandbox. It may block required game assets;
do not weaken it to bypass an access control. A container/network policy is
needed before treating arbitrary page code as securely isolated.

## Run a small direct-mode smoke

From this source checkout, use an **outside-repository** Python 3.12 environment.
Install `requirements.txt` there with `uv pip install --python /path/to/python`.
Install a compatible Chromium with `python -m playwright install --with-deps chromium`
inside an authorized compute/container environment, or provide an existing
Chromium executable. Do not install the training stack for this smoke.

```bash
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1
uv run --no-project --python /path/to/python python \
  -m examples.not_a_robot.smoke \
  --mode fixture --campaign \
  --artifact-root /path/to/new-artifacts \
  --browser-executable /path/to/chrome
```

The process prints an image path. View that image, then send one JSON line on
stdin. Coordinates are normalized to 0–1000, not pixels:

```json
{"controller":"codex","decision_summary":"Click the visible checkbox","actions":[{"action":"click","coordinate":[500,355]}]}
```

Follow subsequent screenshots instead of reusing fixture coordinates on the
live game. Send `{"finish":true}` to stop. Use `--mode live` for the single
allowed website; a real access gate ends the run with exit code 2. Infrastructure
errors and budget truncations exit 1. Exit 0 means orderly controller completion,
**not** necessarily task success; inspect `manifest.json.outcome`.

For an existing CUA-Lite agent, set
`CUA_LITE_REGISTRATION_MODULES=examples.not_a_robot.registration` and select one
of the keys above. Use direct mode: env-server workers and cross-worker recovery
are not supported by this prototype. Production model-adapter integration and
automated API rollouts remain untested.

## What is recorded

Every reset creates a new UUID attempt directory; retries never overwrite it:

```text
attempt-id/
  events.jsonl       # ordered, full-length, fsynced events
  images/<sha>.png   # raw, model-visible and returned observation images
  manifest.json     # written only after event/image integrity verification
```

Events include UTC and monotonic timestamps, canonical requested actions,
individual executed mouse/keyboard primitives, visible evaluator state,
errors, rejected actions, controller retries and outcome. Image references have
SHA256 and byte size. The smoke bridge also records the exact stdin request,
public decision summary, input image and returned feedback. Private model
reasoning is unavailable; no hidden reasoning or absent API responses are
fabricated. General agent callers must separately retain provider responses
and model-call errors; the environment cannot observe those itself.

`recording_complete` means the archive is internally intact, not that the task
succeeded or that every intermediate rendered frame was captured. Abrupt process
death can leave partial events without a manifest. This event archive is not yet
a training-ready `LiteSample`/Parquet export. Existing CUA-Lite agent rollouts can
produce their usual artifacts separately, but that path needs its own smoke test.

## Further work

For self-hosted work, continue the remaining 23 tasks and campaign according to
[the reference audit and implementation plan](/examples/not_a_robot/ALL48_PLAN.md),
with per-task tests and bounded model attempts. The 48 remote placeholders are
not completed local tasks and must not be counted as such.

If resuming the remote experiment, validate permitted live access first, then
verify a few representative levels,
implement evidence-based evaluators, test a vision-model adapter, and add a
canonical export. Independent resets for later levels need an authorized local
game version or another verified reset mechanism; browser storage alone cannot
restore live JavaScript state. A dedicated container image, reproducible local
game version and remote env-server lifecycle belong to later increments.
