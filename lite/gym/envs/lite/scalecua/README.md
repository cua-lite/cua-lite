# Lite.ScaleCUA

`--env-id` `lite.scalecua`

`lite.scalecua` runs the OSWorld subset of
[ScaleCUA](https://github.com/THUDM/SCALE-CUA) on the standard CUA-Lite local
desktop runtime.

ScaleCUA upstream provides OSWorld-shaped task JSON, setup actions, postconfig
steps, and judge functions. This adapter preserves those task semantics while
reusing the existing `lite.osworld` desktop substrate:

| upstream source | runtime split | CUA-Lite execution |
|---|---|---|
| HF `osworld/generated_tasks` | `train` | task setup/eval on the configured `lite.osworld` runtime image with ScaleCUA judge overlays |
| HF `osworld/rl_tasks` | `rl` | curated RL tasks on the same desktop runtime |

`lite.scalecua` intentionally exposes only training-oriented ScaleCUA splits.
For evaluation, use `lite.osworld`'s canonical `eval` split directly.

The env does not migrate the ScaleCUA trainer, controller, VM runtime, VeriGen
runtime, or non-OSWorld suites.

## Quick Start

Install from the repo root:

```bash
uv sync

# Choose one install path:
# Source path: install lite.osworld locally, then provision ScaleCUA catalogs and judge overlays.
uv run --no-sync bash lite/gym/envs/lite/scalecua/scripts/install.sh

# Or, published-image path: pull lite.osworld first, then provision ScaleCUA catalogs and judge overlays.
# uv run --no-sync bash lite/gym/envs/lite/scalecua/scripts/install.sh pull

# Optional lifecycle helpers:
# uv run --no-sync bash lite/gym/envs/lite/scalecua/scripts/install.sh status      # read-only freshness/catalog check
# uv run --no-sync bash lite/gym/envs/lite/scalecua/scripts/install.sh provision   # host deps + ScaleCUA catalogs/judge overlays only
# uv run --no-sync bash lite/gym/envs/lite/scalecua/scripts/install.sh rebuild     # refresh ScaleCUA catalogs; lite.osworld rebuilds only if stale
```

The default install path first delegates to
`lite/gym/envs/lite/osworld/scripts/install.sh` so `desktop_env` and the
configured `lite.osworld` image are present, then imports the pinned ScaleCUA
catalogs and judge overlays.

Then run tasks through normal CUA-Lite interfaces:

```python
import lite.gym as gym

task_ids = gym.registry.task_ids("lite.scalecua", split="rl")
env = gym.make(f"lite.scalecua@{task_ids[0]}")
```

For GPT rollout, always filter unsupported rows before sampling:

```bash
uv run python scripts/rollout.py \
  --model-id gpt-5.5 \
  --env-id lite.scalecua \
  --splits rl \
  --head 1 \
  --filter "lambda m: not m.others.get('exclude_reason')" \
  --config-path scripts/configs/gpt/default/lite.scalecua.yaml \
  --log-root .data/rollouts/lite.scalecua_smoke
```

## Registered Tasks

Runtime split names are exactly `train` and `rl`.

| split | rows | source |
|---|---:|---|
| `train` | 20,289 | HF `extreme1228/ScaleCUA/osworld/generated_tasks` |
| `rl` | 2,049 | HF `extreme1228/ScaleCUA/osworld/rl_tasks` |

All runnable export/rollout commands must use:

```bash
--filter "lambda m: not m.others.get('exclude_reason')"
```

Runnable rows omit `metadata.others.exclude_reason`; unsupported rows set it to
a non-empty string. `metadata.others.domain` is aligned with
`lite.osworld/data/eval.jsonl` where the same OSWorld id exists, otherwise it
falls back to the upstream ScaleCUA source directory.

## Runtime Architecture

`lite.scalecua` does not define a Dockerfile and does not publish a
`cua-lite/lite.scalecua` image. Every task runs on an explicit `lite.osworld`
image tag. This env defaults to:

```text
cua-lite/lite.osworld:latest
```

That base image supplies the GNOME/Xvnc desktop, exec-stdio transport, Chrome,
LibreOffice, GIMP, VLC, Thunderbird, VS Code, and the OSWorld Flask server.
`lite.scalecua` adds only task registration, setup/eval adapters, and judge
overlay resolution.

| component | behavior |
|---|---|
| setup | run the shared OSWorld preamble, then strict-dispatch normalized ScaleCUA `config` actions |
| eval | run ScaleCUA `postconfig`, resolve getters scalecua-local first, then the `lite.osworld` base runner, then the generated `train`/`rl` overlay; metrics resolve overlay → `lite.osworld` → upstream |
| registry | registers generated `data/{train,rl}.jsonl` after catalog-lock validation |
| env-server | registered as a DEDICATED desktop env using the same immutable backend-shape kwargs as `lite.osworld` |

## Upstream Assets

Pinned task material is recorded in
[`data/assets.lock.yaml`](/lite/gym/envs/lite/scalecua/data/assets.lock.yaml).
Final runtime catalogs live in `data/` and are guarded by
`data/catalog.lock.json`. Bulky upstream material and judge overlays live at
the repo root:

```text
.cache/lite.scalecua_tasks/
```

Expected generated files:

| file | meaning |
|---|---|
| `data/train.jsonl` | imported generated task catalog with embedded oracle fixtures |
| `data/rl.jsonl` | imported RL task catalog with embedded oracle fixtures |
| `data/catalog.lock.json` | generated catalog row/hash/source lock |
| `.cache/lite.scalecua_tasks/hf_snapshot/` | pinned upstream ScaleCUA snapshot |
| `.cache/lite.scalecua_tasks/judge_functions/` | ScaleCUA generated/RL judge overlays |
| `.cache/lite.scalecua_tasks/.asset_identity` | installed asset identity |
| `.cache/lite.scalecua_tasks/import_report.json` | counts, exclusions, proxy/auth flags, and judge coverage |

Do not commit HF snapshots, judge staging, rollout logs, or exported parquet
files under this env directory. Generated runtime catalogs are maintained by
`scripts/utils/tasks.sh` and checked by `data/catalog.lock.json`.

## Repository Layout

```
lite/gym/envs/lite/scalecua/
├── main.py            # registers lite.scalecua
├── src/
│   ├── osworld/       # setup/eval adapters and judge overlay loader
│   └── utils/         # asset lock, HF/local import, catalog validation
├── configs/           # default env kwargs
├── data/              # asset/catalog locks, oracle fixtures, generated runtime catalogs
└── scripts/           # install/provision lifecycle
```

Development runbooks, validation logs, and migration notes live under
[`devs/envs/lite.scalecua/`](/devs/envs/lite.scalecua/).

## Development Notes

For normal use, run plain `install.sh`. Development runbooks live under
[`devs/envs/lite.scalecua/`](/devs/envs/lite.scalecua/).

ScaleCUA reuses the normal `lite.osworld:latest` image lifecycle. The
`lite.scalecua` install script does not expose fixed-tag image overrides.
`import` remains a compatibility alias for `provision`, but user-facing docs
should use `provision`.

`status` is read-only. `pull` delegates to `lite.osworld`'s `pull` first so the
base image, host deps, OSWorld assets, and OSWorld catalogs are present, then
provisions ScaleCUA task catalogs so the env is runnable.

Validation details, large-run gates, and audit criteria are maintained in
`devs/envs/lite.scalecua/`.

## Citation

Please cite the upstream work alongside [CUA-Lite](/README.md#citation).

```bibtex
@misc{lv2026scalecuascalingcomputeruse,
  title         = {SCALECUA: Scaling Computer Use Agents with Verifiable Task Synthesis and Efficient Online RL},
  author        = {Bowen Lv and Xiao Liu and Yanyu Ren and Hanyu Lai and Bohao Jing and Hanchen Zhang and Yanxiao Zhao and Shuntian Yao and Jie Tang and Yuxiao Dong},
  year          = {2026},
  eprint        = {2607.11185},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2607.11185}
}
```
