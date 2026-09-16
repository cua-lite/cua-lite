# Not a Robot

A local synthetic-game environment for CUA-Lite: **46 Neal task paths**
(25 captured-instance reconstructions and 21 local variants), plus eight
separate authored exercises. This is not the official game or a complete
original-site replica. Levels **42/45 are unavailable**; level **39 supports
only the unavailable-camera branch**, with no camera access or face recognition.
Original artwork and some media are not distributed.

See [REFERENCE.md](/examples/not_a_robot/REFERENCE.md) for coverage, asset
provenance, instance rules, protocol semantics and remaining fidelity limits.
Local preview needs no GPU, Docker, model credentials or remote service.

## Install the browser runtime

Run from the repository root with `uv` and an outside-repository Python 3.12
environment. Replace `/path/to/python` with that environment's interpreter:

```bash
uv pip install --python /path/to/python -r examples/not_a_robot/requirements.txt
uv run --no-project --python /path/to/python python -m playwright install chromium
```

On Linux, replace `install chromium` above with `install --with-deps chromium`
only if system libraries are needed and installing them is authorized.
An existing compatible Chromium/Chrome/Edge executable is also supported.
No training stack is needed. Level 44's pinned chess.js and Stockfish JS/WASM
run locally in a browser Worker; retain their
[licenses and provenance](/examples/not_a_robot/local/vendor/spatial/PROVENANCE.md).

## Import references and preview

Authored exercises work without private references. For tasks that need them,
obtain the reviewed inputs separately and import each required input with:

```bash
uv run --no-project --python /path/to/python python \
  -m examples.not_a_robot.reference_assets /path/to/reviewed-input
```

| Reviewed input | Purpose |
| --- | --- |
| `neal_reference_first10_supplement_20260909_030500.zip` | First-ten reference images |
| `neal_levels31_48_reference_20260910_040601.zip` | Additional images, including levels 11–30 |
| `neal_all48_incremental_20260910_01.zip` | Alternate level-eight image |
| `module_1070.js` | Derive the level-47 chart as data, without executing JavaScript |

The [manifest](/examples/not_a_robot/reference_manifest.json) pins input/member
hashes. Imports remain private and Git-ignored; other existing bytes are not
overwritten. Missing required assets produce an error, not substitute answers.
Neither these inputs nor the private importer confer redistribution rights.

```bash
uv run --no-project --python /path/to/python python \
  -m examples.not_a_robot.local_tasks --port 8765
```

Open `http://127.0.0.1:8765/` on that host. The gallery selects independent tasks;
`/?task=drag&seed=17` runs an authored exercise without private images.
The listener is loopback-only and has no authentication. Do not expose it publicly.
For a remote host, forward it with
`ssh -N -L 8765:127.0.0.1:8765 user@host`, then open your local URL.
Ctrl+C stops the preview. Evaluations start their own server and do not need it.

## Select a mode

| Mode | Entry | Scope |
| --- | --- | --- |
| Independent local Neal task | `visual_tasks@neal_NN` | Registered tasks only; no 42/45 |
| Authored exercise | `visual_tasks@<name>` for `click`, `input`, `drag`, `sequence`, `moving`, `checkbox`, `stop_signs`, `wiggles` | Separate local mechanics, not captured instances |
| Local campaign | `visual_tasks_campaign@full_game` | Ordered from 01; stops at missing 42, never skips |
| Fixture or remote prototype | `not_a_robot@level_001`, `not_a_robot_campaign@full_game` | `mode="fixture"` is synthetic; `mode="live"` is restricted website access |

The remote catalog's later-level placeholders are not independently resettable
verified graders. Real access gates stop interaction and must not be bypassed.
These modes use direct owned-browser environments, not a production multi-user
env-server or a verified container image.

Local `seed` controls local randomness, not the original website's random stream.
Only 07/08 accept `reference_instance="incremental"` or `?instance=incremental`.
Environment reset restores the initial instance; in-game Refresh follows each
task's rules. The local campaign uses default instances.

## Run a screenshot-only model attempt

Use an installed, authenticated Codex CLI with access to the requested model.
The launcher defaults to `gpt-6-astra` / `xhigh`. Choose explicit tasks and a
**new, nonexistent artifact directory**:

```bash
uv run --no-project --python /path/to/python python \
  -m examples.not_a_robot.codex_smoke --tasks neal_01 neal_34 \
  --artifact-root /path/to/new-run --browser-executable /path/to/chrome \
  --max-seconds 300 --max-steps 100 --record-video
```

Each task gets an isolated client and browser. Only `get_observation`, `computer`
and `finish` are exposed, not DOM, source, shell or web tools. Authentication and
global settings are not rewritten. Omit `--record-video` to disable recording.
Omitting both task-selection flags attempts all **46** tasks independently.

For a bounded local campaign diagnostic:

```bash
uv run --no-project --python /path/to/python python \
  -m examples.not_a_robot.codex_smoke --campaign \
  --artifact-root /path/to/new-campaign --browser-executable /path/to/chrome \
  --max-seconds 120 --max-steps 12
```

`--campaign` and `--tasks` are mutually exclusive. This small budget is not an
all-48 completion budget. The campaign keeps one browser, trajectory and global
budget; a successful stage does not imply a successful campaign.

For a manual JSONL controller without model credentials, use:

```bash
uv run --no-project --python /path/to/python python \
  -m examples.not_a_robot.smoke --mode fixture --campaign \
  --artifact-root /path/to/new-fixture --browser-executable /path/to/chrome
```

View its printed screenshot path, then send an `actions` array on stdin, e.g.
`{"actions":[{"action":"screenshot"}]}`. Coordinates are normalized to 0–1000;
`{"finish":true}` stops. For an independent local task use
`--mode local --task drag` without `--campaign`. The `smoke` entrypoint's
`--campaign` is for fixture/live mode, unlike `codex_smoke`'s local campaign.

## Use the environment directly

```python
from examples.not_a_robot import registration
from lite import gym

env = gym.make(
    "visual_tasks@neal_01",
    seed=0,
    artifact_root="/path/to/new-artifacts",
    browser_executable="/path/to/chrome",
    record_video=False,
)
try:
    observation = await env.reset()
    # Send canonical computer calls through env.step(...).
finally:
    await env.close()
```

For existing CUA-Lite entrypoints set
`CUA_LITE_REGISTRATION_MODULES=examples.not_a_robot.registration`.
Use `visual_tasks_campaign@full_game` for a local campaign.

## Test and inspect results

Install `pytest` and `pytest-asyncio` into the same external environment for tests.
Use a fresh temporary directory. This command includes real-browser tests:

```bash
NEAL_BROWSER_EXECUTABLE=/path/to/chrome PYTHONDONTWRITEBYTECODE=1 \
  uv run --no-project --python /path/to/python python -m pytest \
  -p no:cacheprovider -o addopts= -m 'not stress' \
  --basetemp=/path/to/new-test-artifacts examples/not_a_robot/tests
```

In PowerShell, set `$env:NEAL_BROWSER_EXECUTABLE` and
`$env:PYTHONDONTWRITEBYTECODE` first, and put the command on one line.
Browser tests need the referenced private assets. GUI oracles and virtual-clock
checks are not screenshot-model results or original-site parity evidence.

Each attempt saves `events.jsonl`, content-addressed `images/*.png` and a final
`manifest.json`; optional silent WebM is under `videos/`. Inspect the recorded
outcome, not just the process exit code or an agent's success claim.
`recording_complete` covers event/PNG integrity, independently of game success.
`video.status="saved"` means finalized and hashed, not decoded or synchronized;
video has no audio track and is not supplied as model input.

Fatal runtime errors are `infra_error`, excluded from model success/failure
evaluation. Sequence observations are successive screenshots, not native video
or audio; real-time games continue during inference. Detailed outcome, sequence,
delivery and campaign contracts are in [REFERENCE.md](/examples/not_a_robot/REFERENCE.md).
Canonical training export and complete original-media/campaign validation remain
outside this example's completed scope.
