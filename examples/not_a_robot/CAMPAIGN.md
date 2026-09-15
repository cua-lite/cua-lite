# Local sequential campaign

The experimental controller can now attempt the local levels **1 through 48 in
order**, using one browser, one global budget and one trajectory. This is not a
claim that all 48 pages exist or that a full campaign has been completed. The
catalog contains 25 captured-instance paths and 21 local mechanic variants,
with level 39 limited to its unavailable-camera branch. The first missing
sequential task is level 42; level 45 is also unavailable. The controller never
skips missing tasks. Local variant completion is not original-site completion.
See [variant limits](/examples/not_a_robot/LOCAL_EXPANSION.md).

## Run

Import the private assets and install the existing runtime as described in
[SUPPORT.md](/examples/not_a_robot/SUPPORT.md). No additional server, model
credentials or container are introduced by this controller. Existing Codex CLI
access is needed for a real model attempt.

```bash
uv run --no-project --python /path/to/python python \
  -m examples.not_a_robot.codex_smoke --campaign \
  --artifact-root /path/to/a-new-campaign-run \
  --browser-executable /path/to/chrome \
  --max-seconds 120 --max-steps 12
```

This is a bounded diagnostic budget, not an expected all-48 completion budget.
Do not use `--tasks` together with `--campaign`. Without either flag, the existing
smoke launcher still attempts every implemented reference task independently.
Each campaign run starts at 01; there is no skip-to-level or shortened campaign
success option. The underlying MCP bridge accepts `--campaign` instead of
`--task`, with the same three screenshot/GUI/finish tools and existing isolation.

For a custom controller:

```python
from examples.not_a_robot import registration
from lite import gym

env = gym.make(
    "visual_tasks_campaign@full_game",
    browser_executable="/path/to/chrome",
    artifact_root="/path/to/campaign-artifacts",
    seed=0,
    max_steps=200,
    max_seconds=900,
)
try:
    observation = await env.reset()
    # Send canonical computer calls to env.step(...), observing each result.
finally:
    await env.close()
```

`visual_tasks@neal_NN` retains independent task entry. The separate
`not_a_robot_campaign@full_game` key is the older **remote** prototype, not this
local campaign. The static preview gallery remains independent-task navigation;
it does not itself drive the sequential controller. The local environment owns
campaign navigation because it must save terminal observations before unloading
each page. No second JavaScript transition authority is added.

## Boundary and outcome contract

- A task's own evaluator must report success. The controller first records its
  final screenshot, state and remaining page events, then advances exactly one
  position in the fixed 48-task order.
- It keeps the same browser, context, resource ID, server, recorder, step counter
  and elapsed-time budget. The same seed is passed to each stage; this is a local
  dynamics seed, not a recovered original-site random seed.
- Pending actions in the current batch do not execute on the new page. Public
  feedback reports the boundary and number of unexecuted actions. The next model
  response must use the new screenshot; old coordinates are not replayed.
- Owned held mouse buttons and keys are released on the old page before
  navigation. These lifecycle releases are recorded without attributing them to
  a new model decision. Real-time tasks continue during inference as before.
- A timer can complete a stage while the model is thinking. A subsequent
  observation-only call also records the final page and advances.
- `finish` stops a partial campaign as `agent_stopped`, including when the
  current page has just completed. It does not navigate or turn a model's claim
  into full-game success.
- A missing next implementation produces `unsupported_task`, truncation and
  reward 0. The displayed page can still have successful **per-task** state;
  that does not make the **campaign** successful. The launcher flags
  `blocked_by_missing_task=true`, excludes it from evaluated model outcomes and
  returns a nonzero exit code even when recording and cleanup are complete.
- Global budget exhaustion is truncation, a task failure terminates the campaign,
  and navigation/observation failures remain infrastructure errors. Only ordered
  completion of all 48 stages earns campaign reward 1.
  A dependency failure first seen in the next stage's initial observation also
  truncates that same call with reward 0, preserving the completed prefix.
- A reset finalizes the old archive and starts a new campaign at 01. In-page
  refresh/retry preserves the current stage's original task behavior and events.

## Trajectory evidence

The existing JSONL sequence, UTC timestamps and episode elapsed time remain
monotonic across navigation. `game_event` and source observations carry task IDs;
page-local event sequences and elapsed times restart on each page.
`campaign_stage_started` includes that stage's reference metadata.
`campaign_stage_completed` links its successful state to its terminal screenshot.
`campaign_boundary` records the pending task and exact unexecuted action tail;
`campaign_end` records the completed prefix and final campaign outcome.

The bridge returns the current screenshot, an optional crop or ordered sequence
of real screenshots, and public progress/budget/error information. Final old-stage
frames remain in the archive; `controller_result` identifies images prepared for
the response, and `model_decision` links back to the preceding prepared frames.
`controller_response_sent` records a successful transport write and flush while
the archive is open. Neither event proves model receipt or understanding.
No evaluator targets, source-reading tool or private chain-of-thought is exposed.

Tests include real 01→02 transitions and a scripted GUI traversal of 01→14 that
injects a missing level 15 and stops with reward 0. Level 15 is implemented in the
current catalog; this fixture is not evidence of reaching the real missing 42.
The traversal uses known test answers and the local level-six opponent; it is not
a model or original-site success. Mocked asynchronous-error and final-controller
boundary checks remain separate. None establishes gameplay through levels 15–48.
See [VALIDATION.md](/examples/not_a_robot/VALIDATION.md) for
actual test/model outcomes and [ALL48_PLAN.md](/examples/not_a_robot/ALL48_PLAN.md)
for the remaining full-scope gates.
