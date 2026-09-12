# Lite.CUAGym

`--env-id` `lite.cuagym`

`lite.cuagym` runs [CUA-Gym](https://github.com/xlang-ai/CUA-Gym) tasks on the
CUA-Lite local Docker/Sandbox runtime.

CUA-Gym upstream provides executable task bundles: an instruction, setup assets,
and a `reward.py` evaluator. This adapter preserves those upstream task
semantics while replacing the original execution infrastructure with the
standard CUA-Lite `lite/` runtime:

| upstream family | upstream execution shape | CUA-Lite execution |
|---|---|---|
| `web` / `cross_app` | CUA-Gym-Hub mock websites plus a browser | mock websites served in-container; agent controls real Chrome by desktop coordinate actions |
| `desktop` | GUI desktop apps such as LibreOffice, VSCode, PDF viewers, VLC, GIMP | same task bundles on the `lite.osworld` desktop substrate |
| missing `platform` rows | upstream table does not label the platform | imported through the desktop backend when the bundle has the same desktop-shaped setup/evaluator structure |

The env lives under `lite/gym/envs/lite/` because it is a CUA-Lite local
container environment. It is not an `inv` env: rollout does not talk to an
external hosted service. Each rollout owns one local exec-stdio Sandbox
container.

## Quick Start

Choose one install path:

```bash
# One-time repo bootstrap; rerun the env install script after any bare uv sync.
uv sync --all-extras

# Source path: provision pinned tasks/assets and build the image locally if missing/stale.
uv run --no-sync bash lite/gym/envs/lite/cuagym/scripts/install.sh

# Or, published-image path: use the matching image, then provision pinned tasks/assets.
# uv run --no-sync bash lite/gym/envs/lite/cuagym/scripts/install.sh pull

# Optional lifecycle helpers:
# uv run --no-sync bash lite/gym/envs/lite/cuagym/scripts/install.sh status      # read-only freshness/assets check
# uv run --no-sync bash lite/gym/envs/lite/cuagym/scripts/install.sh provision   # host import deps + tasks/assets only, no image acquisition
# uv run --no-sync bash lite/gym/envs/lite/cuagym/scripts/install.sh rebuild     # force a fresh rebuild
```

Then run tasks through the normal CUA-Lite interfaces:

```python
import lite.gym as gym

task_ids = gym.registry.task_ids("lite.cuagym", split="train")
env = gym.make(f"lite.cuagym@{task_ids[0]}")
```

For GPT rollout — always filter the unrunnable rows before sampling (see
[Excluded Rows](#excluded-rows)):

```bash
uv run python scripts/rollout.py \
  --model-id gpt-5.5 \
  --env-id lite.cuagym \
  --head 5 \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  --config-path scripts/configs/gpt/default/lite.cuagym.yaml \
  --log-root .data/rollouts/lite.cuagym_smoke
```

The install script does not start an env-server and it does not configure model
credentials. The local source-build path may bootstrap Node 20 with `fnm` under
the current user if no Node 20 runtime is installed; `pull` avoids local mock
builds when a matching image is published. The default GPT judge path reads `OPENAI_API_KEY`; set
`OPENAI_BASE_URL` only for a custom endpoint. The small text-only reward-judge
population falls back to those vars. Set `LITE_CUAGYM_JUDGE_*` for
judge-specific model/base URL/API key/retry/timeout; `VLM_*` are compatibility
aliases with lower precedence.

For teacher-data collection:

```bash
uv run python scripts/rollout.py \
  --model-id gpt-5.5 \
  --env-id lite.cuagym \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  --config-path scripts/configs/gpt/recipes/collect/lite.cuagym.yaml \
  --log-root .data/rollouts/lite.cuagym_collect
```

## Registered Tasks

The pinned upstream release contains 1,505 web/cross-app rows and 9,405
desktop/missing-platform rows. All 10,910 rows are registered under the single
`train` split:

```python
len(gym.registry.task_ids("lite.cuagym", split="train"))
```

No row is ever dropped from the registry, and the corpus is not exhaustively
pre-validated task by task. Known tagged defects are refused before agent steps;
new setup/mock/reward runtime failures raise typed task errors during rollout.
Reward/spec mismatches are annotated because they can otherwise return misleading
scores. CUA-Lite's rollout drivers record failed samples and continue other
tasks instead of aborting the batch. Known upstream issues are documented in
[/devs/envs/lite.cuagym/UPSTREAM_ISSUES.md](/devs/envs/lite.cuagym/UPSTREAM_ISSUES.md);
upstream task logic is not patched.

## Excluded Rows

513 of the 10,910 registered rows (4.70%) are unusable as default training
signals and are *annotated* — never removed — with
`metadata.others.exclude_reason`, drawn from the closed `EXCLUDE_REASONS`
vocabulary in
[`src/utils/dataset.py`](/lite/gym/envs/lite/cuagym/src/utils/dataset.py):

| `exclude_reason` | rows | category |
|---|---:|---|
| `broken_reward:empty` | 152 | broken reward script — `reward.py` is whitespace-only, so nothing can print the sentinel |
| `broken_mock:blank_render` | 81 | broken upstream mocks — 44 Google Drive and 37 Uber Eats homepage rows render an empty browser root |
| `broken_reward:no_sentinel` | 42 | broken reward script — compiles, but no reachable top-level path can emit `REWARD:` (16 web + 26 desktop) |
| `broken_reward:syntax_error` | 26 | broken reward script — `reward.py` does not compile under the container's Python 3.12 |
| `broken_reward:missing_golden` | 19 | broken reward script — compares against a `*_golden.*` artifact that setup never creates |
| `broken_reward:instruction_mismatch` | 178 | broken reward/spec pair — `task.json` and `reward.py` disagree on a material success criterion; 128 come from the revision-pinned duplicate-bundle audit |
| `broken_setup:unsatisfiable_gate` | 1 | broken setup script — `initial_setup.sh` gate-aborts on a condition no branch can satisfy |
| `broken_setup:external_dependency` | 8 | broken setup script — reset depends on a live external package/service or invalid external CLI conversion instead of pinned assets |
| `broken_setup:wrong_backend` | 1 | broken setup/catalog pair — a web-labelled row launches a desktop app, so web page health cannot succeed |
| `broken_setup:missing_seed_file` | 2 | broken setup script — the instruction/reward target seed file is never created |
| `broken_setup:syntax_error` | 1 | broken setup script — setup has a syntax error and aborts before the agent can act |
| `broken_setup:no_task_window` | 1 | broken setup script — setup is expected to launch a task GUI, but no usable task window appears |
| `broken_task:empty_instruction` | 1 | broken upstream row — `task.json` states no instruction, so reset has no prompt to hand the agent |

81 are broken upstream mocks, 417 are broken reward/spec pairs, 14 are broken
setups, and 1 states no task at all. Per runtime side, 98 of the 1,505 upstream
web/cross-app rows and 415 of the 9,405 desktop-shaped rows are tagged, leaving
**10,397 default-collectable rows**.

The revision-pinned findings live in
[`data/validation_excludes.json`](/lite/gym/envs/lite/cuagym/data/validation_excludes.json);
the generated count lock lives in
[`data/catalog.lock.json`](/lite/gym/envs/lite/cuagym/data/catalog.lock.json).
Audit them offline or run a review-sized live no-op sample with:

```bash
uv run python lite/gym/envs/lite/cuagym/scripts/utils/validation_sweep.py
uv run python lite/gym/envs/lite/cuagym/scripts/utils/validation_sweep.py \
  --live --limit 20 --concurrency 4
```

The full live-eligible run (10,397 rows in the pinned snapshot) is intentionally
explicit: `--live --all --write`.

The registry does not filter them for you: a tagged task remains registered.
`guard_excluded` refuses it at setup with `CuaGymTaskError(kind="excluded_task")`
so it does not produce a reward-0 trajectory, but an unfiltered rollout still
burns a container reset per row and pollutes the batch with terminal setup
errors. Every rollout, export, and scoring pass must therefore pass the standard
one-liner:

```bash
--filter "lambda m: not m.others.get('exclude_reason')"
```

The yaml config cannot express this — `lite/infer/rollout.py` reads only
`env_id`, `env_kwargs`, `agent_kwargs`, and `agent_id` from the config file, so
the CLI flag is the only mechanism.

## Runtime Architecture

`lite.cuagym` uses one Docker image, `cua-lite/lite.cuagym:latest`, for both
browser and desktop-shaped tasks. The browser side comes from upstream
`web`/`cross_app` rows. The image extends `cua-lite/lite.osworld:latest` with
additive CUA-Gym dependencies:

| layer | purpose |
|---|---|
| `lite.osworld` base | Xvnc desktop, Sandbox exec-stdio server, Chrome, LibreOffice, VSCode, PDF viewer, VLC, GIMP-compatible desktop runtime |
| desktop additions | Python libraries and small CLI/runtime shims needed by upstream desktop setup/reward scripts |
| browser additions | Node 20, built CUA-Gym-Hub mock apps under `/opt/mocks`, shared mock `node_modules`, Chrome wrapper for a maximized first frame |

Web mocks run inside the task container using the original CUA-Gym
setup/reward flow.

Node/npm, Rust/Cargo, and OpenSSH clients stay on the normal agent PATH because
official desktop tasks explicitly ask the agent to use them. Reward-only tools
such as `xcf2png` live under `/opt/env/bin`, which exec-stdio prepends only for
environment commands.

Task metadata selects the runtime side at registration time:

| runtime side | setup path | evaluator path | task max steps | container cap |
|---|---|---|---:|---|
| browser (upstream `web` / `cross_app`) | start referenced mocks, materialize `__CUA_GYM_<APP>_(URL\|HOST)__`, run upstream `initial_setup.py`; expose any additional cross-app targets as Chrome tabs | run upstream `reward.py`, parse final `REWARD:` | 30 | 8 GB / 2 CPU |
| desktop-shaped | run upstream `initial_setup.py`/`.sh`, or upload the seed `docx`/`pptx`/`xlsx` and open it | run upstream `reward.py`, parse final `REWARD:` | 30 | 8 GB / 2 CPU |

The container cap follows the `lite.osworld` default because both `lite.cuagym`
backends run on the same desktop substrate.

The upstream scripts often assume `DISPLAY=:0`; the CUA-Lite container desktop
is `:1`, so the adapter bridges the X socket before running setup/evaluation.
No upstream task bundle is edited during rollout.

## Upstream Assets

Runtime materials are mirrored in the CUA-Lite Hugging Face assets collection so
installs do not depend on a moving GitHub branch:

- [🤗 `cua-lite/lite.cuagym-assets`](https://huggingface.co/datasets/cua-lite/lite.cuagym-assets)
- locked revision: `025b0ab5f89c8f492850dd8536398bf60cb14b45`

The revision and component paths are recorded in
[`data/assets.lock.yaml`](/lite/gym/envs/lite/cuagym/data/assets.lock.yaml). The mirror contains:

| asset | role |
|---|---|
| `data/tasks.parquet` | upstream task table: ids, platform labels, app type, setup kind, difficulty, and metadata used by importers |
| `artifacts/cua_gym_tasks_v1.tar.zst` | per-task bundles extracted into ignored local caches |
| `artifacts/cua_gym_hub.tar.zst` | clean CUA-Gym-Hub source snapshot for mock websites |

Each task bundle may contain:

| file | meaning |
|---|---|
| `task.json` | instruction plus upstream `config` steps such as file download/open actions |
| `initial_setup.py` or `initial_setup.sh` | upstream setup script for script-style tasks |
| `initial_setup.docx` / `.pptx` / `.xlsx` | seed document for document-style desktop tasks |
| `reward.py` | upstream evaluator; CUA-Lite parses the final `REWARD:` line |
| additional files | seed data referenced by setup or reward scripts |

Local generated caches are intentionally ignored by git:

- `.cache/web/lite.cuagym_tasks/`
- `.cache/web/cua-gym-hub/`
- `.cache/desktop/lite.cuagym_desktop_tasks/`

Each task cache keeps the current asset release in one fixed `bundles/`
directory and one runtime catalog, `train.jsonl`. `.asset_revision` records the
locked HF release; `.asset_digest` records the imported cache contents so local
cache mutations fail during registration. A refresh atomically replaces
the whole bundle directory instead of retaining revision-named generations.
Importers atomically replace the catalog, and registration fully parses it
before mutating the registry.

## Repository Layout

```
lite/gym/envs/lite/cuagym/
├── main.py          # registers browser + desktop-shaped tasks under env_id lite.cuagym
├── src/             # runtime source: utils, browser setup/eval, desktop setup/eval
├── data/            # pinned asset lock
├── docker/          # image overlay on top of lite.osworld
├── scripts/         # shell lifecycle entrypoints
├── scripts/utils/   # Python task import tools
└── .cache/          # ignored install.sh material: web, desktop, image context
```

## Development Notes

For normal use, prefer `install.sh pull` or plain `install.sh`. The `assets`
verb force-refreshes the pinned HF mirror for development; normal users should
use `provision`, `pull`, or plain `install.sh`. `rebuild` forces the local mock
dist cache and Docker image to be rebuilt.

```bash
uv run --no-sync bash lite/gym/envs/lite/cuagym/scripts/install.sh provision
uv run --no-sync bash lite/gym/envs/lite/cuagym/scripts/install.sh assets
uv run --no-sync bash lite/gym/envs/lite/cuagym/scripts/install.sh rebuild
uv run --no-sync bash lite/gym/envs/lite/cuagym/scripts/install.sh status
```

PR and release checks use representative rollout samples across the browser
side, upstream cross-app rows, and desktop application families. They verify
that task import, container setup, GUI interaction, and reward execution work
end to end without claiming that every upstream task is healthy.

```bash
uv run python scripts/rollout.py \
  --model-id gpt-5.5 \
  --env-id lite.cuagym \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  --config-path scripts/configs/gpt/default/lite.cuagym.yaml \
  --head 5
```

## Citation

Please cite the upstream work alongside [CUA-Lite](/README.md#citation).

```bibtex
@misc{wang2026cuagymscalingverifiabletraining,
  title         = {CUA-Gym: Scaling Verifiable Training Environments and Tasks for Computer-Use Agents},
  author        = {Bowen Wang and Dunjie Lu and Junli Wang and Tianyi Bai and Shixuan Liu and Zhipeng Zhang and Haiquan Wang and Hao Hu and Tianbao Xie and Shuai Bai and Dayiheng Liu and Que Shen and Junyang Lin and Tao Yu},
  year          = {2026},
  eprint        = {2605.25624},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2605.25624}
}
```
