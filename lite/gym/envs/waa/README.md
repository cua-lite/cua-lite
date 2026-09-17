# WindowsAgentArena

`--env-id` `waa`

CUA-Lite wrapper for Microsoft's [WindowsAgentArena](https://github.com/microsoft/WindowsAgentArena) (WAA). 154 Windows 11 tasks plus 154 no-context setup variants, via `gym.make("waa@<task_id>")` with `LiteDesktopActionSpace`. Each `reset()` boots a disposable Windows-11 QEMU/KVM VM and drives WAA's original setup + evaluator implementation through a bridge. See [docs/envs.md](/docs/envs.md) for the env contract.

## Setup

> **KVM required** — `/dev/kvm` must be readable and writable by Docker. Host prerequisites: Linux x86_64, Docker, `uv`, ≥8 GB RAM, and 80 GiB free disk.

```bash
# Recommended path: pull the source-matched runner + prepared qcow2 when published;
# otherwise build/prep locally. Also fetches task assets and builds the ready snapshot.
uv run --no-sync bash lite/gym/envs/waa/scripts/install.sh

# Equivalent explicit fast path, shown only for spelling.
# uv run --no-sync bash lite/gym/envs/waa/scripts/install.sh pull

# Source-only path: force the runner + qcow2 from source, ignoring published artifacts.
# uv run --no-sync bash lite/gym/envs/waa/scripts/install.sh rebuild      # force the runner + qcow2 from source

# Optional lifecycle helper:
# uv run --no-sync bash lite/gym/envs/waa/scripts/install.sh status       # print runner / ISO / qcow2 / asset status
```

The Windows ISO is Microsoft's freely downloadable, time-limited evaluation image, used under Microsoft's evaluation license. Paths, resources, and env-wide `gym.make` defaults come from [`configs/default.yaml`](/lite/gym/envs/waa/configs/default.yaml) — override the whole file with `WAA_CONFIG=<abs-path | name-under-configs/>`. `uv run --no-sync bash lite/gym/envs/waa/scripts/uninstall.sh` removes the images, ISO, qcow2, task assets, and runtime state.

The runner follows CUA-Lite's standard GHCR lifecycle. A matching `pull` skips
the ~1h local Windows install by adopting the prepared disk image for this
checkout, then rebuilds the host-specific ready snapshot locally.

**Env-server mode (recommended)** — launch [`scripts/serve_env.py`](/scripts/serve_env.py); clients only set `CUA_LITE_ENV_SERVER_URL`:

```bash
uv run python scripts/serve_env.py   # serves all envs on :30100
```

**Direct mode** — `gym.make` boots a fresh VM itself:

```python
import asyncio, lite.gym as gym
env = gym.make("waa@<task_id>", max_steps=15)
asyncio.run(env.reset())
```

Every episode runs on a **fresh, disposable VM** so guest state never leaks across tasks. Env-server mode centralizes admission, ownership cleanup, and retry/recovery; it no longer preboots spare VMs. When a ready snapshot is present each VM boot is a ~15-50s **restore** instead of a ~60-90s cold boot (see below).

### Snapshot restore (fast by default)

Cold-booting Windows dominates episode wall-clock, so `install.sh` builds a **ready snapshot** at the end of a fresh install (a one-time ~90s: cold-boot a VM, `migrate`-save its RAM+device state — ~4 GB, shared read-only by every VM). Once it exists, **every VM restore uses it automatically** (~5x faster boot; there is no on/off knob). It degrades gracefully to cold boot if the bundle is missing or was built for a different base disk.

```bash
# Rebuild it manually (install.sh does this for you; rerun after install.sh regenerates
# the base qcow2; install.sh always attempts this, non-fatal if it fails):
uv run python lite/gym/envs/waa/scripts/utils/prepare_snapshot.py

# Serve normally; each env-server instance uses the ready snapshot automatically:
uv run python scripts/serve_env.py --env-ids waa
```

<details>
<summary>Setup notes — lifecycle & cleanup</summary>

```bash
uv run --no-sync bash lite/gym/envs/waa/scripts/install.sh status   # runner / ISO / qcow2 / assets
uv run --no-sync bash lite/gym/envs/waa/scripts/cleanup.sh          # clean leaked containers + orphan overlays
uv run --no-sync bash lite/gym/envs/waa/scripts/uninstall.sh        # remove images / ISO / qcow2 / assets / runtime
```

Set `SESSION_ID` for session-scoped cleanup. Without it, `cleanup.sh` warns and removes all WAA runtime containers owned by this checkout's naming convention.

</details>

## Quick Start

```python
import asyncio
import lite.gym as gym
from lite.core.tools import make_tool_call

async def main():
    splits = gym.registry.task_ids("waa")   # {"eval": [...], "eval_noctxt": [...]}
    task_id = splits["eval"][0]

    env = gym.make(f"waa@{task_id}", max_steps=15)
    result = await env.reset()
    print(result.text)

    result = await env.step([
        make_tool_call("computer", {"actions": [
            {"action": "click", "coordinate": [500, 500]}
        ]}, call_id="call_0000"),
    ])
    print(f"reward={result.reward}, terminated={result.terminated}")
    await env.close()

asyncio.run(main())
```

## Configuration

Tunable defaults live in [`configs/default.yaml`](/lite/gym/envs/waa/configs/default.yaml) — `env_kwargs` (per-instance), `server_kwargs` (per-deployment infra), and `make_kwargs` (env-wide `gym.make` defaults), read via `env_config.load`. Swap the whole file with `WAA_CONFIG=<abs-path | name-under-configs/>`. See [the env config contract](/docs/envs.md).

`make_kwargs.cursor` defaults to `true` because WAA screenshots arrive without a guest cursor. WAA tracks the dispatched cursor and composites the shared Linux cursor sprite in its capture path; pass `cursor=False` for raw screenshot parity checks.

| Key | Default | Why |
|---|---|---|
| `env_kwargs.max_steps` | `15` | Step budget per episode. |
| `env_kwargs.base_disk` | `~/.cache/cua-lite/waa/images/windows11-waa.qcow2` | Prepared Windows qcow2 (installer output). |
| `env_kwargs.assets_dir` | `~/.cache/cua-lite/waa/assets` | Content-addressed task-asset cache. |
| `make_kwargs.cursor` | `true` | Env-owned cursor compositing on returned screenshots. |
| `server_kwargs.snapshot_dir` | `~/.cache/cua-lite/waa/snapshot` | Where `prepare_snapshot.py` writes the ready snapshot. When present it is used automatically (~5x faster VM boot); see [Snapshot restore](#snapshot-restore-fast-by-default). |
| `server_kwargs.vcpus` / `memory_gb` | `8` / `8` | Per-VM QEMU resources. |
| `server_kwargs.ready_timeout_s` | `900.0` | Windows-VM boot + readiness timeout. |
| `server_kwargs.runtime_root` | `~/.cache/cua-lite/waa/runtime` | Overlay/slot root used for cleanup. |

## Available Tasks

```python
import lite.gym as gym
print(gym.registry.task_ids("waa"))
# {"eval": [...154 tasks...], "eval_noctxt": [...154 "noctxt_"-prefixed variants...]}
```

| Split | Tasks | Scored by the standard recipe | Filter |
|---|---|---|---|
| `eval` | 154 | 138 (13 `infeasible` + 3 `block:` dropped) | `--filter "lambda m: not m.others.get('exclude_reason')"` |
| `eval_noctxt` | 154 | 138 (13 `infeasible` + 3 `block:` dropped) | same |

The `eval` split contains WAA's 154 published tasks. `eval_noctxt` is **not** a separate data partition (nor a train/eval mirror) — it is WAA's alternate no-context variant of those tasks (task ids prefixed `noctxt_`). In the pinned catalog, setup differs for 40 task pairs and is identical for 114; instructions differ for 17 pairs. The two are exposed as separate splits so `--splits eval_noctxt` selects the no-context benchmark directly.

Thirteen tasks in each split use WAA's original `infeasible` evaluator — valid scored tasks, not env failures — flagged by `env.metadata.others["exclude_reason"] == "infeasible"`. A further three variants per split are flagged with a `block:` reason: their evaluator returns reward `1` on the prepared setup state with no agent action (Details-view Explorer, an accepted-absent cookie, a pre-set time zone, an already-satisfied file ordering — six variants across the two splits, split-specific), mirroring `lite.osworld`'s hand-curated `block:` exclusions. The standard CUA-Lite rollout recipe filters both and evaluates 138 tasks per split; see [`docs/eval.md`](/docs/eval.md#windowsagentarena).

Task definitions derive from WindowsAgentArena and retain its MIT license; see `data/NOTICE`.

## Evaluation

Runs **only at episode end** (`terminate` / `response` / `max_steps`) via WAA's original evaluator for the task, which inspects real Windows guest state and returns a reward in `[0, 1]`. Most evaluators are binary, while similarity-based evaluators can return partial credit. `terminate(failure)` and a `response` containing `[INFEASIBLE]` submit `FAIL` to the evaluator; otherwise `DONE`. The integration preserves WAA's upstream setup and evaluator code unchanged — WAA does not publish golden action scripts, so the `trajectory` strings in task JSON are metadata, not solutions.

<details>
<summary>Architecture & background</summary>

```
lite/gym/envs/waa/
├── main.py                 # env class, service registration, bridge client
├── qemu.py                 # host-side QEMU/KVM container lifecycle + qcow2 overlay slots
├── configs/default.yaml    # tunable defaults (env_kwargs + server_kwargs + make_kwargs.cursor)
├── data/                   # tasks.json (154 eval + 154 eval_noctxt) + assets.json (SHA256 lock) + NOTICE (upstream MIT)
├── docker/                 # runner image (py3.9): bridge.py (host↔guest bridge :5050) + Dockerfile + entrypoint.sh + patches/
│   └── prep/               # Windows-guest disk prep: Dockerfile + on-logon.ps1 (baked into the qcow2)
└── scripts/                # install.sh / uninstall.sh / cleanup.sh (lifecycle) + utils/ (helpers)
```

Three Python runtimes cooperate; the host talks **only** to a bridge on container port `5050`:

- CUA-Lite + env-server run on the repo's **Python 3.12** environment.
- The QEMU runner image uses **Python 3.9** for WAA's original client / setup / evaluators.
- The prepared Windows guest runs WAA's **Python 3.10** Flask server on port 5000.

The bridge invokes WAA's `DesktopEnv`, which controls the guest via the WAA guest HTTP server (`20.20.20.21:5000`, `/screenshot`), a Chrome/Edge CDP proxy (`20.20.20.21:9222`), and QEMU QMP (`127.0.0.1:7200` inside the runner). The runner pins WAA commit `6d39ed88c545a0d40a7a02e39b928e278df7332b`; it excludes agent/model deps but keeps the published setup + evaluator deps.

</details>

<details>
<summary>Windows image provenance</summary>

WAA's unattended installation does not create a blank VM — its OEM scripts install the benchmark applications, Python 3.10, and the guest control server on first boot. Windows Setup performs full shutdowns between installation phases, so the preparation script automatically boots the same raw disk up to six times, waits for WAA's clean-shutdown marker, then converts `data.img` to a local qcow2:

```bash
bash lite/gym/envs/waa/scripts/utils/prepare_image.sh \
  /path/to/Win11_Enterprise_Evaluation.iso
# default output: ~/.cache/cua-lite/waa/images/windows11-waa.qcow2
```

`install.sh` invokes this automatically. By default it builds a thin local prep image on top of the upstream image; the patch layer fixes the archived LibreOffice installer URL, pins `pywin32==311` (312 removed the `mfc140u.dll` that WAA's `win32ui` import needs), starts the guest server through the explicit Python 3.10 executable, completes LibreOffice Writer's first GUI launch before sealing, and applies small guest-HTTP-server compat fixes. It validates the import before registering the server startup task and does not change WAA's guest Python version.

- `WAA_PREP_BASE_IMAGE` — private mirror of the upstream image (patches still applied).
- `WAA_PREP_IMAGE` — bypass the patch build, use an already-patched prep image.
- `WAA_PREP_NOVNC_PORT` — pin the prep VM's noVNC host port (default: an ephemeral port on 127.0.0.1, so parallel preps don't clash; find it with `docker port <prep-container> 8006`).
- `WAA_KEEP_RAW=1` — keep the raw `data.img` after conversion.

The upstream default pins digest `sha256:869afffe384f0e0356dbfcdf78486611e31d62e692544cbfd9a6c08606f07440` (published 2024-11-18), matching the pinned WAA source revision. The final qcow2 is moved into place atomically with `windows11-waa.qcow2.provenance.json` (ISO checksum, prep-source checksum, prep image identity, size). Concurrent builds of the same output are rejected.

</details>

<details>
<summary>Task assets, catalog & all-task verification</summary>

The JSON catalog holds setup operations + evaluator definitions, not the large task files. Across the 154 standard tasks, setup uses 82 input-file references and evaluators use 48 expected-file references; after URL dedup the catalog references 121 GitHub assets + 2 Google Drive assets. `install.sh` pulls them from cua-lite's HF mirror (`cua-lite/waa-assets`, revision-pinned; the upstream GitHub/Drive URLs stay a per-asset fallback) into a content-addressed cache, verifies every SHA256 against `data/assets.json`, and the runner serves them locally — before handing a task to WAA the bridge rewrites only matching upstream asset URLs to that local server, so setup/evaluation never depend on external hosts. Upstream tasks hard-code the `C:\Users\Docker` profile, so use the prepared WAA image (not a generic Windows image).

Regenerate the checked-in catalog from the pinned checkout:

```bash
lite/gym/envs/waa/scripts/utils/sync_tasks.py --waa-root ../WindowsAgentArena
```

Verification is **standalone** (not part of `install.sh`, which only prepares
resources) — run the checks directly:

```bash
# Registration sanity + unit/live tests
uv run python -c "import lite.gym as gym; print(len(gym.registry.task_ids('waa', split='eval')), len(gym.registry.task_ids('waa', split='eval_noctxt')))"
WAA_DOCKER=1 uv run pytest tests/gym/envs/waa/test_waa.py -m live -n0

# One task through the public gym.make API (quick post-install sanity):
uv run python lite/gym/envs/waa/scripts/utils/smoke_test.py \
  ~/.cache/cua-lite/waa/images/windows11-waa.qcow2

# Every task variant's setup + evaluator in a fresh VM (validates upstream
# setup/screenshot/evaluator; does NOT perform tasks or require reward 1):
uv run python lite/gym/envs/waa/scripts/utils/verify_all_tasks.py \
  --base-disk ~/.cache/cua-lite/waa/images/windows11-waa.qcow2 \
  --concurrency 1 --attempts 2 --resume
```

The 308 published variants contain 194 distinct setup/evaluator combinations, so
identical standard/no-context pairs share one VM run; results are resumable under
`.data/waa/task-smoke` (gitignored).

</details>

## Citation

Please cite the upstream work alongside [CUA-Lite](/README.md#citation).

```bibtex
@inproceedings{bonatti2025windows,
  author    = {Rogerio Bonatti and Dan Zhao and Francesco Bonacci and Dillon Dupont and Sara Abdali and Yinheng Li and Yadong Lu and Justin Wagle and Kazuhito Koishida and Arthur Bucker and Lawrence Keunho Jang and Zheng Hui},
  title     = {Windows Agent Arena: Evaluating Multi-Modal {OS} Agents at Scale},
  booktitle = {Forty-second International Conference on Machine Learning, {ICML} 2025, Vancouver, BC, Canada, July 13-19, 2025},
  series    = {Proceedings of Machine Learning Research},
  volume    = {267},
  publisher = {{PMLR} / OpenReview.net},
  year      = {2025},
  url       = {https://proceedings.mlr.press/v267/bonatti25a.html}
}
```
