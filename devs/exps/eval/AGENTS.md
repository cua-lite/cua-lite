# Eval campaigns

Standardized layout for repeatable, version-pinned eval runs across all envs in the cua-lite matrix. Each `<env>/` subdir produces one committed snapshot per **(commit, run_id)** pair.

This README is **self-contained**: the matrix, the discipline, and the snapshot template all live here. An agent given (this README) + (an env name) + (optional model subset) should be able to run the campaign end-to-end and write `<env>/logs/<commit-ts>_<commit>/<run_id>.md`.

---

## Layout

```
devs/exps/eval/
├── AGENTS.md                          # this file — workflow spec + matrix + snapshot template
├── <env>/                             # lite.osworld | osworld | androidlab | androidworld | osworld_g | screenspot_pro
│   ├── run.sh                         # per-model invocation
│   ├── CHANGELOG                      # commit-pair diffs that affect this env's numbers
│   └── logs/
│       └── <commit-ts>_<commit>/
│           ├── <run_id>.md            # one snapshot per run; multiple per commit allowed
│           └── ...
└── ...

.exps/eval/<env>/<commit-ts>_<commit>/<run_id>/<slug>/      # raw rollout artifacts — gitignored
```

| Token | How resolved | Example |
|---|---|---|
| `<commit-ts>_<commit>` | `git log -1 --date=format:'%Y-%m-%dT%H-%M' --format=%cd HEAD` + `_` + `git rev-parse --short HEAD`. The short-sha width is Git-configured; runners match `[0-9a-f]+` and let `git log <sha>..HEAD` validate reuse candidates instead of hardcoding seven characters. | `2026-08-08T23-56_81cdfa408` |
| `<run_id>` | `$EVAL_RUN_ID` — **optional**, falls back to highest-numbered `run_<N>[_<label>]` under this commit dir (resume-to-latest), or `run_0` if none exist. Schema: `run_<N>` default, optionally `run_<N>_<label>` to name a campaign's intent (mop-up, variance check, hyperparam tweak). See [Run id contract](#run-id-contract-eval_run_id) below — **read it before launching anything**. | `run_0`, `run_1_mopup`, `run_2_temp1.0` |
| `<slug>` | `<model-id>` with `/` → `_`; browsergym runners append `__<EVAL_CONFIG_ID>` when that variable is set, so different config variants can share one `<run_id>` without resuming each other. | `Qwen_Qwen3-VL-8B-Instruct`, `Qwen_Qwen3.5-9B__think_on` |

Same `(commit, run_id, model)` → resume; for browsergym runners, `EVAL_CONFIG_ID` is also part of the artifact slug when set. Bumping any key component starts a fresh artifact dir.

> **Why `<commit-ts>_<commit>`?** Sortable chronologically (sha alone is opaque). Computed from `HEAD`'s committer time (not the wall clock), so the path stays stable across re-runs of the same commit — only `<run_id>` advances when you start a new campaign at the same commit.

---

## Env-server prerequisite

Before running any `./<env>/run.sh`, an env-server must be reachable AND expose the `env_id` you'll use. Probe the URL you plan to use:

```bash
curl -sf -H "Authorization: Bearer $CUA_LITE_ENV_SERVER_TOKEN" \
  "$CUA_LITE_ENV_SERVER_URL/envs/$ENV_ID" | jq .
# → {"available": true,  "n_tasks": N, ...}   ready
# → {"available": false, "error": "..."}      env setup missing on server host
# → HTTP 404                                  env_id not registered on this server
```

No env-server reachable? Start a local one with [`serve_env.py`](/scripts/serve_env.py) -- its host must already have the env's deps installed per the env's README. Scope it to the envs you intend to run:

```bash
uv run python scripts/serve_env.py --port 30100 --env-ids "$ENV_ID"
HOST_IP=$(hostname -I | awk '{print $1}')
export CUA_LITE_ENV_SERVER_URL=http://${HOST_IP}:30100
export CUA_LITE_ENV_SERVER_TOKEN=$(whoami)   # any string; passthrough mode keys by sha256(token)[:6]
```

Singleton-backed envs with slow shared services, including WebArena and VisualWebArena, must be started with `--warm-singleton` so the backend is available before rollouts begin:

```bash
uv run python scripts/serve_env.py \
  --port 30100 \
  --env-ids browsergym.webarena \
  --warm-singleton
```

Do not set `--max-live-envs` in the default eval runbook. The server derives its normal admission cap from host capacity; use `--max-live-envs` only as an explicit advanced override for a constrained repro, and record why it was needed.

Once exported, `LiteEnvClient` reads `CUA_LITE_ENV_SERVER_URL` / `_TOKEN` at `gym.make` time and routes every env interaction over HTTP — `run.sh` doesn't have to know. Env containers (`lite-env-*`, qemu emulators, `osworld` VM-in-Docker) spawn on the env-server's host. Only the leak-survey + cleanup paths change with env-server location (same host vs SSH vs HTTP `DELETE /instances`).

See [Env-server](/docs/envs.md#env-server) for details.

---

## ⚠️ Commit-first discipline + pipeline-aware path reuse

**Always `git commit` before launching a campaign.** Pipeline files (see list below) must be clean — `run.sh` enforces this with a hard pre-flight: any uncommitted edit to those paths and the script aborts. The point of the commit-keyed layout is **idempotent reproduction** — same pipeline state + same `run.sh` line → same numbers every time. A mid-campaign code edit to pipeline files would split the campaign across two effective commits but only one ends up recorded.

But — and this is what makes the layout livable — **doc-only commits don't fragment paths**. `run.sh` resolves the commit dir by *pipeline state*, not by HEAD: it finds the most recent existing `<commit-ts>_<commit>/` whose pipeline state matches HEAD's (no pipeline-relevant edits between them) and reuses that dir. Push a new CHANGELOG entry, edit a snapshot markdown, write a new section in this README — none of that bumps the path key. Only changes to pipeline files do.

**Pipeline files** (the set that, when changed, advances the path key):
- `lite/agents/` — adapter, agent, protocol, registry
- `lite/gym/envs/<env>/`, `lite/gym/utils/`, `lite/gym/remote/`, and
  top-level `lite/gym/{base,types,registry,factory,services,wrappers}.py`
- `lite/agents/factory.py` (model registry)
- `scripts/serve_env.py`, `devs/exps/eval/utils/runtime_mode.sh`, and
  `devs/exps/eval/utils/campaign_dir.sh`
- `scripts/rollout.py` (inference entry)
- `scripts/configs/*/default/<env>.yaml` plus env-specific variants matched by
  the runner, such as `<env>*.yaml` or `<env>/*.yaml` (per-env rollout configs)

Anything else — README/CHANGELOG/snapshots, training-only code, other-env configs — is doc-equivalent for this env.

**If you patch a pipeline file mid-campaign**: stop, commit, then either restart at the new commit (the campaign is now at a new pipeline state — pure restart, fresh dir) or accept the kickoff hash as approximate and add a `code drifted mid-campaign` note to the snapshot header (see the `2026-04-27T23-21_828ea14` grounding snapshots for the bad example we want to avoid).

---

## Run id contract (`EVAL_RUN_ID`)

**The rule, in one sentence:** `EVAL_RUN_ID` is the campaign identifier; **one campaign = one `EVAL_RUN_ID`, used by every model in it AND every failure-restart of every model in it, regardless of how many days it spans**. Bump only to start a brand-new campaign at the same commit.

This matters because the failure-restart pattern relies on it:

```
.exps/eval/<env>/<commit-ts>_<commit>/<run_id>/<slug>/
                                      └────────────────┘
                                      this is scripts/rollout.py --log-root.
                                      Same path → resume; different path → start over.
```

If you change `EVAL_RUN_ID` mid-campaign, the new run.sh invocation writes to a *different* `--log-root` and re-runs every task from scratch, **including the ones that already finished**. That's almost never what you want.

For browsergym config comparisons inside one report, keep `EVAL_RUN_ID` fixed and set a literal `EVAL_CONFIG_ID` per variant. For thinking comparisons, explicitly setting `EVAL_ENABLE_THINKING=false` or `true` automatically defaults `EVAL_CONFIG_ID` to `think_off` or `think_on`; you may still override it. These produce sibling artifact dirs under the same run and can be summarized together in one `run_2_textonly.md`.

**Schema** (what to put in `EVAL_RUN_ID`):

- Default: `run_<N>` (e.g. `run_0`, `run_1`, `run_2`) — incrementing per fresh campaign at the same commit.
- With label: `run_<N>_<label>` (e.g. `run_1_mopup`, `run_2_temp1.0`) — name the *intent* of the campaign in the label. Free-form. Useful for mop-up rounds, variance re-runs, hyperparam tweaks.

**Auto-resume default**: `EVAL_RUN_ID` left unset, `run.sh` picks the highest existing `run_<N>[_<label>]` under the current commit dir; falls back to `run_0` for the first campaign at this commit. The intended workflow:
- **Failure-restart** (the common case): just re-run `./run.sh <model>` with no `EVAL_RUN_ID` — auto-resume picks up the same dir, re-runs only the missing tasks.
- **Fresh new campaign at same commit**: explicitly `export EVAL_RUN_ID=run_<max+1>` (or `run_<max+1>_<label>`).

### What "one campaign" means

- ✅ Day-1 starts model A, model B, model C; some fail; you restart them later same day or next day. **All same `run_<N>`** (auto-resume handles this for free).
- ✅ A model crashes mid-eval; you `kill -9` it and rerun the exact same `run.sh` line tomorrow. **Auto-resume** picks the same dir.
- ✅ Mop-up rounds (re-running models with `num_valid < num_tasks`) are part of the same campaign — auto-resume.
- ❌ You ran a campaign last week, now want to re-eval the same commit (e.g. for variance / re-check). **New campaign**: `export EVAL_RUN_ID=run_<max+1>` to start a fresh dir.

### Discipline for fresh campaigns

- **Bump explicitly**, with a fixed string literal — never a function call:
  ```bash
  export EVAL_RUN_ID="run_1"                # ← fixed; intent is clear
  # NOT:
  export EVAL_RUN_ID="run_$(some-counter)"  # ← don't compute it on the fly
  ```
- **Recovery across sessions.** If your agent session ends and a future session needs to start a *fresh* campaign at the same commit, list `ls .exps/eval/<env>/<commit-ts>_<commit>/` to find the existing `run_<N>` set, then export `run_<max+1>`. (For pure resume, just don't export anything — auto-resume does the right thing.)
- `run.sh` safety:
  1. `EVAL_RUN_ID` unset → auto-resolved to highest existing `run_<N>` (resume-friendly default).
  2. `EVAL_RUN_ID` set to a *fresh* value while other `run_<N>` already exist under the same commit dir → run.sh prints a warning + 5-second sleep with the existing values, so you can Ctrl-C if you actually meant to resume (just unset).

---

## Matrix — which agent on which env

"Agentic" envs (multi-step, episodic):

| Model | lite.osworld | osworld | androidlab | androidworld |
|---|---|---|---|---|
| `Qwen/Qwen3-VL-{2,4,8,32}B-{Instruct,Thinking}` | ✅ | ✅ | ✅ | ✅ |
| `Qwen/Qwen3.5-{2,4,9,27}B` | ✅ | ✅ | ✅ | ✅ |
| `ByteDance-Seed/UI-TARS-7B-DPO` | ✅ | ✅ | ✅ | ✅ |
| `ByteDance-Seed/UI-TARS-1.5-7B` | ✅ | ✅ | ✅ | ✅ |
| `meituan/EvoCUA-8B-20260105` | ✅ | ✅ | — | — |
| `Tongyi-MAI/MAI-UI-{2,8}B` | — | — | ✅ | ✅ |
| `stepfun-ai/GELab-Zero-4B-preview` | — | — | ✅ | ✅ |

"Grounding" envs (single-step click prediction; smaller model roster — OS-agentic-only models like GELab-Zero are out of scope here):

| Model | osworld_g | screenspot_pro |
|---|---|---|
| `Qwen/Qwen3-VL-{2,4,8,32}B-{Instruct,Thinking}` | ✅ | ✅ |
| `Qwen/Qwen3.5-{2,4,9,27}B` | ✅ | ✅ |
| `ByteDance-Seed/UI-TARS-7B-DPO` | ✅ | ✅ |
| `ByteDance-Seed/UI-TARS-1.5-7B` | ✅ | ✅ |
| `meituan/EvoCUA-8B-20260105` | ✅ | ✅ |
| `Tongyi-MAI/MAI-UI-{2,8}B` | ✅ | ✅ |

Default `tp_size` is 1 except `Qwen3-VL-32B` (tp=2) and `Qwen3.5-27B` (tp=4). `Qwen/Qwen3-VL-*-Thinking` uses the same default-mode command as `Instruct`; browsergym runners keep thinking disabled unless `EVAL_ENABLE_THINKING=true` is set. For MiniWoB/WebArena `text_only.yaml` and VisualWebArena `mixed.yaml` thinking-on runs, the runners append the fixed override `--agent-kwargs '{"enable_thinking": true, "sampling_kwargs": {"max_new_tokens": 4096}}'`. Use one explicit `EVAL_RUN_ID` for a paired comparison report; explicitly set `EVAL_ENABLE_THINKING=false` / `true` to get separate `think_off` / `think_on` artifact dirs under that same run. Do not compute `EVAL_RUN_ID` from a shell template. On OOM, double — see [OOM escalation](#oom-escalation).

**Task counts per env** (used to determine "fully finished" in the snapshot):

| Env | Tasks | Notes |
|---|---|---|
| `lite.osworld` | 332 | of 369; 37 dropped by the `exclude_reason` filter |
| `osworld` | 325 | of 369; 44 dropped by the `exclude_reason` filter |
| `androidlab` | 138 | no filter |
| `androidworld` | 116 | no filter |
| `osworld_g` | 510 | of 564; 54 `refusal` tasks dropped by the `exclude_reason` filter |
| `screenspot_pro` | 1581 | no filter |

---

## Running

> **GPU IDs are host-specific.** Use only the user-given allowed range; never hardcode. See [GPU reservation discipline](#gpu-reservation-discipline) for watchdog rules.

```bash
# from repo root.
# Default workflow — auto-resume to latest run_<N> at this commit (failure-restart):
CUDA_VISIBLE_DEVICES=<gpus> ./devs/exps/eval/<env>/run.sh <model-id>

# Or start a fresh campaign at the same commit (see Run id contract above):
export EVAL_RUN_ID="run_1"                    # bump past existing run_0 / run_1 / ...
CUDA_VISIBLE_DEVICES=<gpus> ./devs/exps/eval/<env>/run.sh <model-id>

# illustrative — substitute your own GPUs
CUDA_VISIBLE_DEVICES=0       ./devs/exps/eval/lite.osworld/run.sh Qwen/Qwen3-VL-8B-Instruct       # tp=1
CUDA_VISIBLE_DEVICES=0,1     ./devs/exps/eval/osworld/run.sh Qwen/Qwen3-VL-32B-Instruct           # tp=2
CUDA_VISIBLE_DEVICES=0,1,2,3 ./devs/exps/eval/androidworld/run.sh Qwen/Qwen3.5-27B               # tp=4
```

`run.sh` resolves `<commit-ts>_<commit>` and `<run_id>` itself, looks up the rollout-config YAML by model family, sets `HF_HUB_OFFLINE=1`, and routes `--log-root` to `.exps/eval/<env>/<commit-ts>_<commit>/<run_id>/<slug>/`. Browsergym runners append `__<EVAL_CONFIG_ID>` to `<slug>` when set. tp_size is inferred from the GPU count in `CUDA_VISIBLE_DEVICES` (1, 2, or 4).

Pre-req — cache model weights locally if missing:

```bash
hf download <model-id>
```

**Failure-restart is the core mop-up mechanism.** A single `run.sh` invocation often does not bring `num_valid` to `num_tasks` — sglang stalls, env-VM blips, network hiccups, OOMs, all leave gaps. Re-running the same `run.sh` line with the same `EVAL_RUN_ID` and, for browsergym variants, the same `EVAL_CONFIG_ID`, is the cure: `scripts/rollout.py` / `lite.infer.rollout.get_pending` reads existing per-task summaries from `--log-root` and skips finished tasks, retrying only the missing ones. **Repeat the invocation until `num_valid == num_tasks`** (or you accept a residual env-flaky set — see workflow step 3).

---

## GPU reservation discipline

The user provides the **allowed GPU range** (e.g. "use GPUs 0-3"). The agent must:

1. **Hold every idle GPU in the allowed range under a watchdog at all times.** Zero idle window — even a few seconds is enough for another tenant on the host to grab the GPU.
2. **Never release back to the host pool.** When a job finishes, the watchdog must be re-launched on its GPUs immediately. The allowed range belongs to us until the user says otherwise.
3. **Never use any GPU outside the allowed range**, regardless of how empty `nvidia-smi` shows it.

If no allowed range was given, **ask the user** before launching anything — don't guess.

### Pattern

- **One watchdog per GPU at the start.** Releasing GPU `k` only kills its own watchdog; the others keep holding. This is the only pattern that's gap-free under per-GPU release.
- **Job needs a strict subset of the allowed range**: kill only that subset's watchdogs, run the job, re-launch their watchdogs the instant the job exits.
- **Job needs the whole allowed range** (e.g. tp=4 on a 4-GPU reservation): release all, run, re-hold the moment the job exits — minimize the gap.

### Watchdog launcher

One invocation per GPU (substitute the GPU id):

```bash
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python .claude/skills/watchdog/watchdog.py \
  > /tmp/gpu_watchdog_0.log 2>&1 &
```

After every campaign — including aborts, errors, or stop-and-debug pauses — **the allowed range must end up fully held**. Don't leave it un-held overnight or while you context-switch.

---

## OOM escalation

If sglang OOMs on launch, **double `tp_size` and the GPU count** until it fits or you hit `tp=4` (the cap on a 4-GPU host):

```
tp=1 → tp=2: CUDA_VISIBLE_DEVICES=0,1     ./<env>/run.sh ...
tp=2 → tp=4: CUDA_VISIBLE_DEVICES=0,1,2,3 ./<env>/run.sh ...
```

Still OOM at tp=4? Levers: lower `--concurrency` (default 16 → 4), lower `--rollout-max-response-len` in the YAML, or trim `history_n` / `image_max` via `--agent-kwargs '{"protocol_kwargs": {"history_n": 50, "image_max": 10}}'`.

**Qwen3.5 family** uses `history_n=100, image_max=20` → prompts balloon to tens of thousands of tokens. The 9B often needs tp=2 on long-task envs; 27B is already tp=4.

**Qwen3.5 is hybrid mamba+attention.** sglang's radix cache only applies to attention layers, so each turn's mamba SSM state is recomputed from scratch — Qwen3.5 on `osworld` typically runs 2–3× slower than equivalent-size Qwen3-VL even when GPU memory fits comfortably. Don't read this as "stuck"; check sample-summary mtime to confirm forward progress (see [GPU reservation discipline](#gpu-reservation-discipline) below).

### Known Qwen3.5 mobile failures (and fixes wired into `run.sh`)

Two failure modes confirmed at `2026-04-28T21-31_785df232`:

1. **sglang OOM at tp=1 on androidworld** for `{4,9}B` (mamba state + KV peak under `history_n=100` exceeds `mem_fraction_static=0.79`). For 4B/androidworld specifically, even tp=2/4 OOM'd via fragmentation — `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is the silver bullet.
2. **`EnvTimeoutError: step() timed out`** on androidworld/androidlab. The framework default is 120s (`lite/gym/registry.py` `_WRAPPER_KWARG_DEFAULTS["step_timeout"]`); env-wide `make_kwargs` can override it (`lite.osworld`=120s, `osworld`=180s, android = no step override). Qwen3.5 mobile inference under contention can exceed the framework default.

Fixes baked into `devs/exps/eval/<env>/run.sh`:
- Long-step runners set explicit timeout overrides where needed (usually
  `--env-kwargs '{"step_timeout": 180}'`; `mobileworld` uses `240`, and envs
  with env-wide `make_kwargs` may rely on their config instead).
- androidworld: `--concurrency 4`. 4B retries also need `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (set in `run_one.sh`-style wrappers).
- androidlab: `--concurrency 8`. 4B/9B should launch with 2 GPUs (sglang dp=2).
- 27B (any env): tp=2 pinned in `lite/agents/factory.py`.

If still OOM at tp=4: lower `mem_fraction_static` to 0.65 (needs sglang flag passthrough — not currently exposed in run.sh).

### Known docker timeouts — androidlab under high host load

`docker run --rm -d --device /dev/kvm` has a 180-second hardcoded timeout in [`lite/gym/envs/androidlab/container.py`](/lite/gym/envs/androidlab/container.py) (`AndroidLabContainer.start`). Under load_avg ≫ NCPU (e.g. concurrent training rollouts), `docker run` exceeds 180s → `subprocess.TimeoutExpired` → task retries.

Mitigations: fewer concurrent android jobs, lower `--concurrency`, or wait out the contending workload.

### In-process leak guards (atexit / acquire-time reaper)

These guards run inside the env-server process (where envs actually live), not in the eval client. The eval client (`scripts/rollout.py` via `LiteEnvClient`) owns *sessions*, not containers — when the client exits cleanly, it `DELETE /instances/<id>` for each session, and the env-server then `docker rm -f -v`'s the underlying container. SIGKILL of the client still leaves orphan envs on the server; SIGKILL of the env-server itself is what the next-startup zombie reaper catches.

Concretely, on the env-server host:

- **androidworld** — `AndroidWorldContainer.destroy()` ([`lite/gym/envs/androidworld/container.py`](/lite/gym/envs/androidworld/container.py)) is registered via `atexit` for every live container. On normal env-server exit / unhandled exceptions / Ctrl-C the container is `docker rm -f -v`'d and its host-side API port reservation released.
- **lite.osworld / osworld_g / screenspot_pro** (Sandbox family) — `lite/gym/sandbox/base.py` registers an `atexit` hook that `docker rm -f`'s any container in `_LIVE_CONTAINERS`. Each `reset()` adds, each `close()` removes.
- **osworld** — `OSWorldContainer._register()` enrolls each VM-in-Docker container in `LiteContainerBase`'s process-wide atexit backstop; `destroy()` removes the container and releases the reserved API port.
- **env-server cross-cutting** — on every fresh start, [`recover_all`](/lite/gym/remote/recovery.py) calls each env's `EnvServices.reap(boot=True)`; docker-backed envs reap via [`ContainerReaper`](/lite/gym/remote/reaper.py), which does a `docker ps -a --filter name=^lite-env-<port>-...-<env_id>-` sweep (scoped by the env-server's `server_port` so two env-servers on the same host don't reap each other's containers) and `docker rm -f`'s anything left from a previous lifetime. Idle envs are reaped per the `--idle-ttl-sec` flag.

For SIGKILL of the env-server, leftovers are caught by the next-startup reaper. For other exits, the in-process + lifespan guards handle it. From the eval client side, `DELETE /instances?session_id=...&env_id=...` is the explicit kill-switch when you need to drop all of *your* `SESSION_ID`'s sessions without waiting for idle TTL.

---

## Leak survey + cleanup (between every run)

> ⚠️ **Scope cleanup to your own `SESSION_ID`. Never kill what you didn't start.** Multi-tenant hosts may have concurrent `lite.slime-*` / `lite-env-*` / ray / sglang processes you don't own — leave them alone. No-arg `/cleanup` is safest (this session only); named-arg scopes (`/cleanup lite.osworld`, etc.) widen — read [`.claude/skills/cleanup/SKILL.md`](/.claude/skills/cleanup/SKILL.md) for per-scope semantics. Notable footguns: `pkill`-pattern envs (`androidworld` / `webgym`) are **host-global on the host**, safe only inside the slime container; `/cleanup osworld` with `SESSION_ID` unset removes **every** `-osworld-` container on the daemon, co-tenants' included. CLAUDE.md's rule applies: **never** stop, kill, or touch a Docker container you didn't start.

Aborted rollouts (OOM, Ctrl-C, sglang stall, VM crash) leave orphan docker containers (including `osworld`'s VM-in-Docker) and qemu emulators **on the env-server host**, plus orphan sglang / rollout python on the client (`run.sh`) host. They don't free RAM/CPU on their own and silently starve the next run.

Survey before / after each run, and every 10-15 min during long ones — both layers:

```bash
# Client-side (where `./run.sh` was invoked):
echo "=== gpu ==="; nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
echo "=== orphan rollout python ==="; ps -eo pid,etime,cmd | grep -E 'rollout/local\.py' | grep -v grep || echo "none"

# Env-server-side (works from anywhere with the URL+TOKEN — no SSH to the env host needed):
echo "=== live env sessions ==="; \
  curl -s -H "Authorization: Bearer ${CUA_LITE_ENV_SERVER_TOKEN}" \
       "${CUA_LITE_ENV_SERVER_URL}/instances" | jq .

# Only if you have shell on the env host (same host as Slime, or SSH'd in):
docker ps --filter "name=lite-env-" --format '{{.Names}} ({{.Status}})'
docker ps --filter "name=-osworld-" --format '{{.Names}} ({{.Status}})'   # osworld VM-in-Docker
pgrep -af "qemu-system.*lite_avd" | grep -v grep || echo "none"
```

If anything shows up that isn't from your active run, the **portable, host-agnostic** path is the env-server's bulk-close endpoint — works the same whether the env host is local or remote:

```bash
curl -X DELETE \
  "${CUA_LITE_ENV_SERVER_URL}/instances?session_id=${SESSION_ID:?set SESSION_ID}&env_id=<env>" \
  -H "Authorization: Bearer ${CUA_LITE_ENV_SERVER_TOKEN}"
```

Tier-1 `cleanup.sh` scripts still exist and are still useful — but they only do anything when invoked **on the env host**, since that's where the containers live:

| Scope | Targets | Where to run |
|---|---|---|
| `/cleanup lite.osworld` | `lite-env-<sid>-lite.osworld-*` docker containers | env host |
| `/cleanup osworld` | `*-<sid>-osworld-*` VM-in-Docker containers | env host |
| `/cleanup androidlab` | `lite-env-<sid>-androidlab-*` docker containers | env host |
| `/cleanup androidworld` | leaked qemu emulators + AVD locks | env host |
| `/cleanup all` | sweep everything (skips `lite.slime-*` training containers) | env host |

**OSWorld cleanup is local docker.** [`lite/gym/envs/osworld/scripts/cleanup.sh`](/lite/gym/envs/osworld/scripts/cleanup.sh) is a `docker ps -aq --filter name=…-osworld-` sweep on the env host — `osworld` runs each trajectory in a locally-managed VM-in-Docker container ([README](/lite/gym/envs/osworld/README.md)). **Footgun**: with `SESSION_ID` unset the script's filter is bare `-osworld-`, i.e. every osworld container on the daemon including co-tenants' — set `SESSION_ID` to stay session-scoped. `lite.osworld` is a separate, lighter GNOME-container env whose `-lite.osworld-` names the `-osworld-` filter does not match; use `/cleanup lite.osworld` or `DELETE /instances?session_id=...&env_id=lite.osworld` for it.

---

## Reading utilization

A single `nvidia-smi` snapshot at 0% does **not** mean a slot is dead — agent rollouts alternate sglang prefill/decode bursts (99% util) with multi-second env-wait gaps (0% util) where 16 docker/qemu workers wait on the env. Before declaring a slot stuck:

1. Sample 3-5 readings of `utilization.gpu` over ~10s — if you see any ≥ 50% reading, it's working.
2. Or check forward progress: `find <log-root>/eval -name "*.json" -mmin -5 | wc -l` — non-zero means tasks are being written.
3. **System-level resource pressure** (the silent killers, `nvidia-smi` won't show them):
   ```bash
   free -g | head -3                   # available RAM (should stay > 50 GiB on a 1.5 TiB host)
   top -bn1 | head -3                  # load avg vs cpu count
   df -h /tmp .exps/                   # disk: /tmp (sglang spills), .exps/ (per-task PNGs accumulate)
   docker ps --filter "name=lite-env-" --format '{{.Names}}' | wc -l   # rollout fan-out
   ```
   Sustained low `available` (< 20 GiB) or load_avg ≫ NCPU sets up the **Linux OOM killer** to SIGKILL sglang/rollout without warning — the most common silent failure mode for "process just disappeared".

What does mean stuck: GPU memory dropped from ~70+ GB to a few GB (sglang exited but rollout still alive) **and** sglang stdout shows `Gracefully exiting... Remaining number of requests N` looping. See [the deadlock recipe in the agent workflow](#agent-workflow-checklist).

---

## Snapshot template

Path: `<env>/logs/<commit-ts>_<commit>/<run_id>.md`.

This file is the **running progress log**, not just a wrap-up artifact. **Update it after each model finishes (or partially-finishes)**: append the row to `Results`, refresh `Last updated`, commit. The same `(commit, run_id)` markdown lives across the whole campaign — one file, edited multiple times. Required fields only; omit a field rather than restate a default.

```markdown
# <env> @ <commit-ts>_<commit> · <run_id>

- **Commit**: `<short sha>` — `<subject>`
- **Host / GPUs**: `<hostname>` / `<e.g. 0-3>`
- **Artifacts**: `.exps/eval/<env>/<commit-ts>_<commit>/<run_id>/`
- **Started**: `<YYYY-MM-DD HH:MM TZ>`
- **Last updated**: `<YYYY-MM-DD HH:MM TZ>` ← bump on every edit
- **Notes**: anything non-default — non-16 concurrency, sglang stalls, mop-up rounds, raw-artifact location override

## Results

| Model | Finished | Mean episode return |
|---|---|---|
| `<model-id>` | <nf>/<nt> | <mer> |

## Highlights

- short interpretive bullets — call out partials, scaling trends, surprising ranking, etc.

## Breakdown   ← only for grounding envs (osworld_g, screenspot_pro)

<!-- 1. uv run python devs/exps/eval/utils/reaggregate_breakdown.py \
        --axes <comma-separated others keys> \
        .exps/eval/<env>/<commit-ts>_<commit>/<run_id>/
     2. paste output of: uv run python devs/exps/eval/utils/render_breakdown.py
        .exps/eval/<env>/<commit-ts>_<commit>/<run_id>/ -->
```

**Include every matrix-applicable model in the Results table** — fully finished, partial, AND never-started. The reader needs to see at a glance what was attempted, what wasn't, and where the data is incomplete. Fill cells per this rule:

| State | Finished cell | MER cell | Visual marker |
|---|---|---|---|
| fully finished (`nf == nt`) | `<nt>/<nt>` | actual `<mer>` | none — plain row |
| partial (`0 < nf < nt`) | `<nf>/<nt>` | actual MER over those `nf` samples | `⚠️` + bold-italic on both metric cells |
| not started (no rollout ever launched, or summary missing entirely) | `0/<nt>` | `—` | `⚠️` + bold-italic on both metric cells |

Examples:

```
| `<model-id>`   | 332/332       | 0.2553        |        ← fully finished
| ⚠️ `<model-id>` | _**163/332**_ | _**0.4233**_  |        ← partial
| ⚠️ `<model-id>` | _**0/332**_   | _**—**_       |        ← not started (incl. user-skipped, infrastructure-blocked)
```

**Row order**: sort by mean episode return **descending**. Put every `⚠️` row (partial + not-started) in a separate block at the **bottom** of the table, also internally sorted by MER descending (not-started rows with no MER go last among the `⚠️` block).

Add a one-liner in Highlights for each `⚠️` row explaining why it's partial / not started (e.g. user halted, HF-only backend, OOM unresolved).

`Mean episode return` = `stats.mean_episode_return` from `<commit-ts>_<commit>/<run_id>/<slug>/summary.json`. For binary-reward envs this equals success rate. Extract:

```bash
python3 -c "
import json, glob
for f in sorted(glob.glob('.exps/eval/<env>/<commit-ts>_<commit>/<run_id>/*/summary.json')):
    d = json.load(open(f))['stats']
    slug = f.split('/')[-2]
    print(f'  {slug:42s}  {d[\"num_valid\"]}/{d[\"num_tasks\"]}  {d[\"mean_episode_return\"]:.4f}')
"
```

The `<run_id>` is implicit in the path (snapshot lives at `logs/<commit-ts>_<commit>/<run_id>.md`); no need to repeat in the body.

### Breakdown tables (grounding envs only)

`breakdown.json` is **always** post-hoc — the rollout itself does not write it. The eval-side scripts (`devs/exps/eval/utils/`) reproduce per-task `metadata.json` from the env code, then aggregate.

Two-step flow per snapshot:

```bash
# 1. Aggregate. --axes picks which others keys to pivot on (the env publishes
#    factual fields under others; choosing axes is an analysis decision).
#    Typical axes:
#      osworld_g     → paper_category,box_type,GUI_types
#      screenspot_pro → group,ui_type,application
uv run python devs/exps/eval/utils/reaggregate_breakdown.py \
    --axes paper_category,box_type,GUI_types \
    .exps/eval/<env>/<commit-ts>_<commit>/<run_id>/

# 2. Render the multi-model markdown table for paste-in.
uv run python devs/exps/eval/utils/render_breakdown.py \
    .exps/eval/<env>/<commit-ts>_<commit>/<run_id>/

# Cross-product (e.g. screenspot_pro paper-canonical 12-cell `group × ui_type`).
uv run python devs/exps/eval/utils/render_breakdown.py --cross "group,ui_type" \
    .exps/eval/<env>/<commit-ts>_<commit>/<run_id>/
```

Skip very wide axes when pasting (osworld_g `GUI_types`: 33 cols; screenspot_pro `application`: 26 cols) — keep them in `breakdown.json` for re-render but leave them out of the snapshot. Refresh the breakdown section on every snapshot edit (same cadence as `Results` / `Last updated`).

`reaggregate_breakdown.py` writes a `metadata.json` next to each `summary.json` from current env code (`task_id` → `LiteBaseMetadata.others`). Use `--force-metadata` after the env adds a new field that you want to pivot on (re-renders all metadata.json files in place).

`--axes` is typed against the `others` keys the run's metadata actually declares: a misspelled axis **exits non-zero** naming the declared keys, instead of quietly pivoting on the remaining axes and reporting a smaller denominator as a result. (In `--sweep` mode the typing is done once against the union across all visited runs, since a sweep deliberately spans envs with different `others` keys.)

### Carry-forward convention

When a campaign at a new commit re-runs **only a subset of the matrix**, results from a prior commit for the un-rerun models stay valid as long as the pipeline change between the two commits doesn't affect them (e.g. `1bdef611` Qwen3.5 mobile alias fix doesn't change Qwen3-VL / UI-TARS / MAI-UI behavior). In that case, **migrate the prior rows into the new snapshot rather than maintaining cross-commit views**:

- Snapshot table: prefix the carried row with `📌` and suffix the MER cell with `(@<prior-short-sha>)`. Sort the merged table by MER as usual.
- Add a one-liner under `**Notes**:` naming the prior commit and naming the pipeline change that justifies the carry-forward (so reviewers can verify it's safe).
- Artifact tree (`.exps/eval/<env>/<new-commit-dir>/run_0/`): copy the carried-over `<slug>/` dirs from the prior commit's `run_0/` so plot/aggregator scripts (which read one snapshot dir) Just Work.

This keeps each snapshot.md a complete view of "results at this commit's pipeline state" without forcing every campaign to re-run the full matrix or every tool to walk multiple commits. If a pipeline change affecting *all* models lands, drop the carry-forward and re-run the matrix from scratch.

---

## CHANGELOG conventions — `<env>/CHANGELOG`

CHANGELOG is **commit-scoped**, not run-scoped — it tracks code-level deltas that change the eval's numbers, regardless of how many times each commit was run. Newest on top.

Record only changes that **plausibly affect that env's numbers**:
- agent code: `lite/agents/factory.py`
- env code under `lite/gym/envs/`. The actual path per env:
  - `lite.osworld` → `lite/gym/envs/lite/osworld/**` + `lite/gym/sandbox/**`
  - `osworld` → `lite/gym/envs/osworld/**` + `lite/gym/sandbox/**`
  - `androidlab` → `lite/gym/envs/androidlab/**`
  - `androidworld` → `lite/gym/envs/androidworld/**`
- rollout configs referenced by this env: `scripts/configs/<agent>/default/<env>.yaml`
- sampling parameters, `history_n` / `image_max`, filter expression
- dataset / split changes
- HF model revisions if a known-incompatible bump landed

Skip pure docs / unrelated training infra / cosmetic refactors.

```markdown
## <new_short> ← <prev_short>  (YYYY-MM-DD)

- one-liner about what changed and why it matters here
```

The very first entry in a new `CHANGELOG` has no predecessor; just write `## <new_short>  (YYYY-MM-DD)` and a one-liner noting it's the initial snapshot.

---

## Agent workflow checklist

For an agent given **(this README) + `<env>` + (optional model subset)**:

1. **Pre-flight**
   - `cd` to repo root.
   - **Commit first**: any code change you intend to evaluate must already
     be committed before you launch a campaign — `git diff` and `git diff
     --staged` should both be empty (untracked artifact dirs are fine).
     The artifact path is `.exps/eval/<env>/<commit-ts>_<commit>/<run_id>/<slug>/`,
     and `<commit-ts>_<commit>` is captured ONCE at kickoff. If you patch code mid-
     campaign (e.g. a bug fix, a prompt template tweak), the in-flight
     rollouts produced by later code paths are *misattributed* to the
     kickoff commit. Either commit + restart the campaign, or accept
     the kickoff hash as approximate and add a "code drifted mid-
     campaign" note to the snapshot's header (see the
     `2026-04-27T23-21_828ea14` grounding snapshots for an example of the
     latter, which we now want to avoid).
   - Resolve `<commit-ts>_<commit>` = `git log -1 --date=format:'%Y-%m-%dT%H-%M' --format=%cd HEAD` + `_` + `git rev-parse --short HEAD`; the short-sha width is not fixed at seven characters. (`run.sh` does this for you through `utils/campaign_dir.sh`.)
   - Pick `<run_id>`. **For a fresh campaign**: `export EVAL_RUN_ID=run_<N>` (literal string, choose `run_0` for the first campaign at this commit, `run_<max+1>` for a subsequent fresh campaign). **For failure-restart / mop-up**: leave `EVAL_RUN_ID` unset — `run.sh` auto-resumes to the latest existing `run_<N>`. See [Run id contract](#run-id-contract-eval_run_id). If recovering a session and you genuinely want a *new* fresh campaign, `ls .exps/eval/<env>/<commit-ts>_<commit>/` to see existing `run_<N>` set, then export `run_<max+1>`.
   - Identify the model list: matrix subset for `<env>`, filtered by the optional user-specified subset.
   - Confirm the allowed GPU range with the user (ask if not given). Launch one watchdog per GPU in that range. See [GPU reservation discipline](#gpu-reservation-discipline).
   - Run the leak survey; invoke `/cleanup` if anything stale shows up.
   - **Initialize the progress log** at `<env>/logs/<commit-ts>_<commit>/<run_id>.md` from the snapshot template. Fill the header (commit / host+GPUs / artifact path / Started timestamp). `git add` it; commit at the next model boundary.

2. **For each model — fixed-log-dir failure-restart loop**

   The contract: same `(commit, run_id, model)` → same `--log-root` → resume; for browsergym config variants, `EVAL_CONFIG_ID` is part of the slug and must also match. Re-launch the **identical** `run.sh` line until `num_valid == num_tasks`. With `EVAL_RUN_ID` left unset, auto-resume points at the same `run_<N>` every restart (see [Run id contract](#run-id-contract-eval_run_id)) — only export `EVAL_RUN_ID` when you intend to start a *fresh* campaign.

   - Pick the right number of GPUs from the allowed range (1 / 2 / 4 per the model's tp).
   - Release watchdog on those GPUs only; launch via `./<env>/run.sh <model-id>`; re-hold immediately when it exits.
   - On OOM, double tp_size (and GPU count) and retry — auto-resume keeps the same `run_<N>` so completed tasks are still skipped.
   - **For runs > 1h, poll every 15-25 min** — distinguish "still working" from "stuck" via the sanity checks in [GPU reservation discipline → Reading utilization](#reading-utilization).
   - **Graceful-exit deadlock recipe** (common — esp. Qwen3.5/Qwen3-VL near end-of-run):
     1. Diagnose: sglang stdout shows `Gracefully exiting... Remaining number of requests N` looping; GPU memory dropped from 70+ GB to a few GB.
     2. `pkill -KILL -f "sglang.launch_server.*<model-name>"` — frees the GPU.
     3. The rollout python often still hangs even after sglang dies — `kill -KILL <rollout_pid>` it explicitly.
     4. Verify: `nvidia-smi` shows the slot's GPU back to 0 MiB; no orphan `sglang::scheduler` / `sglang::detokenizer` processes left (`pgrep -f sglang`).
     5. Re-run the same `run.sh` line — completed task summaries on disk are skipped, only the unfinished tasks re-execute.
   - **External-SIGKILL recipe** (`subprocess.CalledProcessError: ... died with <Signals.SIGKILL: 9>`): usually Linux OOM killer or another tenant's `pkill` on the host. Re-check [Reading utilization point 3](#reading-utilization) for resource pressure first; if `available` RAM was low or load was high, lower `--concurrency` (32 → 16 → 8) before re-launching. **Don't `/watchdog hold` between for-loop / mop-up retries** — the watchdog will saturate VRAM and OOM the next attempt's sglang at startup; hold only after the entire eval round exits.
   - After each model, run a leak survey; cleanup if non-empty.
   - **When the model finishes** (cleanly or terminal failure): append its row to `Results` (with the `<nf>/<nt>`, MER, `⚠️` marker if partial), refresh `Last updated`, commit. Don't wait for the campaign to finish — every model boundary is a commit point.

3. **Mop-up loop** (per env, same `(commit, run_id)`)

   The goal is **every model at `<nt>/<nt>`**. Run this loop until convergence:

   ```text
   while any model has num_valid < num_tasks:
       for each such model:
           re-launch ./<env>/run.sh <model-id>     # auto-resume → same --log-root
           wait for it to exit (clean or sglang-deadlock-killed-manually)
       if no model gained tasks vs the previous iteration:
           break    # remaining gaps are env-flaky tasks that won't recover
   ```

   - Each re-launch only re-attempts the missing tasks (resume via `--log-root`); finished tasks are skipped.
   - Plan for 2-3 iterations on average. Some models may need more if env-flaky.
   - Stop only when an iteration produces zero gains — those last few tasks are env-bugs, not modelling.
   - Models that plateau below `<nt>/<nt>` after convergence are recorded in the Results table marked `⚠️` per the snapshot template, with a one-liner in Highlights stating "env-flaky; mop-up plateaued at `<nf>/<nt>`".

4. **Wrap-up**
   - Final `/cleanup <env>` (for `osworld` campaigns this sweeps the VM-in-Docker containers — see the cleanup table above).
   - Verify the allowed GPU range is fully held by watchdogs (one per GPU). Re-launch any missing.
   - Extract numbers from `.exps/eval/<env>/<commit-ts>_<commit>/<run_id>/*/summary.json`; write `<env>/logs/<commit-ts>_<commit>/<run_id>.md` per the template above.
   - For grounding envs (`osworld_g`, `screenspot_pro`) — first run `reaggregate_breakdown.py --axes <...>` to produce `breakdown.json`, then `render_breakdown.py` to render the `## Breakdown` markdown for paste-in. See [Breakdown tables](#breakdown-tables-grounding-envs-only).
   - If this is the first run at this commit and the diff vs the previous catalogued commit is non-trivial, append a `CHANGELOG` entry.

---

## Cross-references

- [`docs/envs.md`](/docs/envs.md) — env architecture (direct vs server mode)
- [`docs/slime.md`](/docs/slime.md) — Slime container build / launch / init
- [`docs/eval.md`](/docs/eval.md) — rollout entry, per-env prereq tables
- [`devs/exps/train/desktop/README.md`](/devs/exps/train/desktop/README.md) — sibling training campaign
