# WebGym Teacher-Data Pipeline

This directory owns WebGym teacher-data collection and the WebGym trajectory
filter used before staging.

Collect clean `gpt-5.5` demonstrations on the webgym **train** split to distill web-agent skills
into a Qwen student (**target 8B/32B**, so multi-step *research* is learnable — not only a 2B).
The dataset must cover **all tiers' skills**: easy/medium tasks **don't abuse goto** (read the page
you're on), and complex tasks that genuinely need it **do search/navigate** — the
"minimal-necessary goto" principle.

WebGym publishes trajectories from ONE teacher. Collection and filtering live in
that teacher's own runbook; everything below the cleaned log roots is
dataset-level.

| Teacher | Runbook | Model |
|---|---|---|
| `gpt5_5` | [`gpt5_5/AGENTS.md`](/devs/data/webgym/gpt5_5/AGENTS.md) | `gpt-5.5` (API) |

The handoff between the teacher runbook and this one is exactly:

    .data/rollout/webgym/gpt5_5/$COMMIT/{d1..d7,popular}_clean

This doc is the converged, scale-ready pipeline. Exploration rollouts that produced it live in
`.logs/rollout/webgym_explore/` (transient); the real collection writes to `.data/` (curated).

## Pipeline (3 components, validated)

1. **Teacher prompt — balanced**
   ([/scripts/configs/gpt/recipes/collect/webgym.yaml](/scripts/configs/gpt/recipes/collect/webgym.yaml)):
   per teacher — see [Prompt Design](/devs/data/webgym/gpt5_5/AGENTS.md#prompt-design).
2. **Filter — the single filtering step** (`filter.py`): see [Shared Filter](#shared-filter).
3. **Distribution — 25 / 60 / 15** (easy / medium / hard, site-start only): see
   [Collection Targets](#collection-targets).

## Collection Targets

**Tier distribution — per-tier `--sample` 500 / 1000 / 2000 (easy / medium / hard).**

The collection knob is a **flat per-tier `--sample`** (see Attempt sizing in
[`gpt5_5/AGENTS.md`](/devs/data/webgym/gpt5_5/AGENTS.md)): easy d1–3 → 500 each, medium d4–6 →
1000 each, hard d7 → 2000. This weights attempts toward mid/high **because that's where
distillation actually adds value** — a base Qwen3-VL-8B already does easy well but cracks on
medium/hard (natural train split is 79 / 19 / 2 easy/medium/hard, so the raw pool is the opposite
shape). The realized clean mix is ~33 / 52 / 15 over the difficulty tiers — easy ran higher
and hard lower than an even target, because easy yield is ~2× hard's and the d7 pool capped at
~1,392 (below `--sample 2000`):

- **easy — low headroom, keep only a floor.** Base 8B on eval-easy ≈ 62% raw / ~81% reachable
  — comparable to the gpt teacher. easy = "read the visible page", a skill the base already has, so
  SFT on more easy mostly wastes budget. We keep a floor as a **grounding/format anchor** (eval is
  71% easy, so the student must stay fluent there) — not zero. (`--sample 500`/tier; realized 771
  clean, higher than intended because easy yield is highest.)
- **medium — highest value × largest pool → the bulk.** Paired eval on identical tasks
  (same seed/budget): **gpt 83% vs base 8B 33%** (preliminary) — a big gap. The base's navigation is
  broken (re-goto's its own homepage, loops, jumps to blank google), exactly what the teacher fixes.
  Medium's task pool is huge (51k site-start), so this is the **safe place to scale**.
- **hard — high headroom, sample the full d7 site-start pool.** Highest per-task gain. We set
  `--sample 2000` to take the whole d7 site-start pool, but it **caps at ~1,392** (the pool is
  smaller than 2000); d7 site-start clean-yield is **measured 25%** (this batch) → **350 clean** at
  n=1. NOTE: this is `difficulty==7` ONLY — d8+ not yet collected; to add the harder-than-7 skills,
  run a separate `difficulty>=8` pass (see "Within hard" below and
  [Scale to 5k](#scale-to-5k--additive)).
  - **Future (when a `difficulty>=8` pass exists): OVERSAMPLE d8+ before staging.** This batch is
    d7-only, so nothing to oversample yet. Once d8+ is collected: the >=7 pool decays hard with
    difficulty (~59/20/10/5/6% for d7/d8/d9/d10/d11+) and success-rate also falls (d7 ~40% → d10
    ~15%), so a random pick is ~⅔ d7. To keep the harder skills represented, choose the cleaned
    rows/roots explicitly: **take all available d8+ clean demos first, then fill with d7** before
    invoking `stage` (best-effort — d10+ is genuinely scarce, "cover what exists" not "force a
    quota"). Difficulty histogram of any tier:
    `uv run python devs/data/webgym/quality_check.py` (via metadata), or group clean
    `episode_return>=1.0` trajectories by `metadata.others.difficulty`.

**Collect site-start only, all tiers** (exclude search-engine start pages). Search-start tasks open
on google.com/bing.com whose **results page is blank in the headless browser** (see
[Search & navigation policy](#search--navigation-policy)) → ~0 clean demos
(the hardest browsecomp riddles, unsolvable here). There are **no ddg-start tasks** in the dataset,
so "exclude {google,bing}-start" = "site-start only". Composition (train): medium **92% site-start**
/ 8% google·bing-blank; hard **45% site-start (3,007)** / 55% google·bing-blank. Kept site-start
demos still search via **ddg** when a task needs it (ddg works; see the policy section).

| band | difficulties (site-start only) | `--sample`/tier | realized clean | share of 3,251 |
|------|------|------:|---------------:|------:|
| easy    | d1–3 | 500  | 771   | 24% |
| medium  | d4–6 | 1000 | 1,202 | 37% |
| hard    | d7   | 2000 | 350   | 11% |
| popular | curated pool | full (~2,102) | 928 | 29% |
| **total** | | | **3,251** | |

(Realized this batch `a2cad60b`, per the teacher runbook's Attempt sizing table. Over the difficulty
tiers only that's ~33 / 52 / 15 easy/medium/hard — hard came in under an even target because the d7
pool capped at ~1,392 and its yield is the lowest (25%). Scale to 5,000 is additive — see
[Scale to 5k](#scale-to-5k--additive).)

## Shared Filter

`filter.py` is the single filtering step:
`--drop-failed --drop-loops --drop-serp-only --drop-captcha --drop-unsubmitted --drop-illposed-task`.
**ALL filtering lives here** (`stage` no longer applies any default filter). `--drop-failed` is the
success filter (`episode_return>=1.0`, the predicate that used to be `stage`'s hidden default).
`serp_only` drops trajectories that **search but never click through to a real page**
(snippet-scraping / SERP bouncing — degenerate for any model size) while **keeping productive
research** (search → click a result → read, however many steps — the learnable complex-task skill).
`--drop-captcha` drops bot-verification-wall trajectories; `--drop-unsubmitted` drops trajectories
whose answer was never sent via a top-level non-empty `response` call; `--drop-illposed-task` drops
degenerate underspecified instructions (≥3 unfilled "a specific …" slots — a data-gen bug; best
applied pre-rollout). No-op stripping drops a turn but no picture, so the output row is
**compacted**: `compact_row_images` ([devs/data/utils.py](/devs/data/utils.py)) drops the images
nothing references any more and renumbers
the survivors `0..N-1` (image list and every `{"type":"image","index": N}` part in one step),
and when an old pre-`role:"tool"` leading observation is dropped, carried goal content preserves
text, metadata, and earlier indexed reference image parts while dropping the stale observation image.
For a **2B** target, swap to the stricter `--drop-search-flail` (also drops ≥2-engine /
≥3-search chains).

*Search reality (validated):* from the cluster's datacenter egress IP the search **engines**
(DuckDuckGo/Google/Bing) IP-block the search **request** (results error with "anomaly"/reCAPTCHA;
the homepage still loads), so blocked search attempts become dead-ends. We still **keep productive
search** and only drop `serp_only` — a blanket `--drop-search-goto` would kill the learnable skill
and is NOT used for 8B/32B; the durable fix is infra (clean egress IP / proxy), not filtering. Most
search/block/unsubmitted/ill-posed cases score 0 and are already removed by `--drop-failed`; the
new flags are explicit guards + pre-rollout hygiene.

## Search & navigation policy

The dataset must teach **both** "stay on the page" (easy/medium) **and** "search/navigate when a
task truly needs it" (complex). The split between a *good* and a *bad* search demo is **not** how
many searches — it's whether the agent **clicks through to a real page** or just reads SERP snippets:

- **KEEP (productive research):** `search → click a result → read the page` (→ optionally refine &
  repeat). This is the complex-task skill; an 8B/32B learns it. Measured: 18/18 medium and 23/24
  hard search-trajectories click through.
- **DROP (`serp_only`):** searches but **never clicks through** — answers from snippets, or bounces
  the same query across engines (e.g. `126677`: "oriolesband.com" on Google→Bing→DuckDuckGo, 0
  clicks, then guesses — while ignoring the open page that had the answer). Degenerate for any size.
- **DROP (`loops`):** ≥3 consecutive identical actions (stalls).

Why one search engine is enough: **Google/Bing return a blank results page in the webgym browser**
(confirmed by screenshot) — so the agent was *forced* to bounce to DuckDuckGo. The prompt now tells
it to search on DuckDuckGo directly, which removes the bounce — validated: medium = **20/20
search demos single-engine ddg, 0 bounce** (the prior prompt bounced google→bing→ddg). `serp_only` still catches any
residual snippet-scraping.

The older blanket drops **F1** (any search goto) / **F2** (cross-domain goto) / **F6** (search-start
task) are **OFF** — they kill legitimate research. They remain in `filter.py` for a
single-site-grounded dataset. `--drop-search-flail` is the stricter 2B-target variant of the
search filter.

## Complete Workflow

Run from the repository root. Pipeline: collect → filter → stage → upload/download →
`export_sft`. Freeze the code revision, task set, prompt, and log root for each batch; a resume
must use the identical command.

### 1. Install And Configure

**Run through the env server (default).** Build the webgym image once (`install.sh build`), then
launch [`scripts/serve_env.py`](/scripts/serve_env.py) once and point the collector at it via
`CUA_LITE_ENV_SERVER_URL`. The env-server starts the shared OmniBoxes backend container on first
use and owns instance lease/release with a drift-reaper + TTL + session scoping, so a crashed or
stalled stream can't leak browser workers — steadier than the in-process **direct** path (which is
the fallback when `CUA_LITE_ENV_SERVER_URL` is unset).

```bash
# --- host ---

# step 0: build the webgym image once, then start a durable env-server.
#   - JUDGE CREDS (load-bearing): the VLM reward judge needs OpenAI creds IN THE
#     SERVER's environment. Without them webgym reset() fails 500
#     (EnvDepsMissingError) — and since --drop-failed in the teacher runbook keys
#     on `episode_return` (which IS that judge's reward), no creds ⇒ no clean data.
#     OPENAI_BASE_URL is optional; set it only for a custom endpoint.
#   - WEBGYM_INSTANCES sizes the OmniBoxes browser pool: set >=64 for production
#     collection; omit to auto-size from host RAM/CPU (capped at 128). The server
#     starts the shared backend container on first use and sets WEBGYM_MASTER_URL
#     itself. Launch detached (setsid/nohup) so it outlives the shell. Any free
#     port works (30100 here); point CUA_LITE_ENV_SERVER_URL at it.
uv run --no-sync bash lite/gym/envs/webgym/scripts/install.sh build
export OPENAI_API_KEY=...                  # + export OPENAI_BASE_URL=... for a custom endpoint
# webgym's server imports install-added host judge deps, so serve_env is one of
# the explicit `--no-sync` exceptions.
setsid bash -c 'WEBGYM_INSTANCES=64 uv run --no-sync python scripts/serve_env.py \
  --port 30100 --env-ids webgym >> serve_env.out 2>&1' &
HOST_IP=$(hostname -I | awk '{print $1}')
export CUA_LITE_ENV_SERVER_URL=http://${HOST_IP}:30100

# VERSION this rollout batch by a cua-lite commit id, PINNED ONCE here at batch start — the
# commit whose recipe (collect.yaml + filter.py + sft config) this batch uses. Every log-root
# lives under .../gpt5_5/$COMMIT/, so one batch = one dir. Set it as a FIXED literal and reuse it
# for the whole batch (incl. resumes) — do NOT re-derive from HEAD per command: a multi-day
# collection will see unrelated intermediate commits, and the batch must keep its original id.
# This same id is the HF tag at upload (`upload --tag $COMMIT`), so local dir == HF revision ==
# producing commit (`download --revision $COMMIT` pulls exactly this version).
COMMIT=a2cad60b   # ← pin to your batch's recipe commit (e.g. `git rev-parse --short HEAD` once, then freeze)
```

### 2. Collect And Filter (per teacher)

Run [`gpt5_5/AGENTS.md`](/devs/data/webgym/gpt5_5/AGENTS.md). It ends with the cleaned log roots
under `.data/rollout/webgym/gpt5_5/$COMMIT/`, which are the only input staging takes.

Filtering happens THERE, once: `filter.py` is the only filtering step and `stage` below applies no
predicate of its own.

### 3. Stage, Upload Transport, And Download

> **Upload is a declarative full sync, not an append.** It plans the whole repo
> from the LOCAL staging dir and deletes everything else: `orphans = current -
> planned_paths - {.gitattributes}` are committed as deletions, and the rendered
> README (which defines the HF configs) is rebuilt from local stats alone.
> Staging only some cleaned roots and uploading would therefore DELETE the
> published shards of every root left out. There is no flag that disables the
> sweep, and `--skip-existing` does not protect anything (it only skips
> re-uploading files this run already plans). **Every stage must list every
> cleaned log root.**
>
> `--dry-run` does NOT report the orphan set — the whole sweep, including its
> logging, sits behind `if not dry_run`. A dry run only prints the paths it would
> push. The real pre-flight is to diff those planned paths against
> `HfApi().list_repo_files(repo_id=..., repo_type="dataset")` yourself.

To ADD trajectories to an already-published dataset, every stage must still list
EVERYTHING already published — the sweep above deletes whatever this stage
does not plan. Which means two cases, and only one needs `unstage`:

**You still have the published rows' annotated log roots** (the usual case —
under `.data/rollout/webgym/gpt5_5/$COMMIT/`). Nothing to reconstruct: run the single
stage below listing every root, and upload. Skip the rest of this block.

**Those roots are gone** (a different machine, or the tree was cleaned).
Rebuild them from the published repo first. `unstage` writes a rollout LOG-ROOT (not a staging layout), and it must be run **once per
published config** into its own directory — `stage` maps log-roots to config names 1:1, and one call
that pours several configs into one directory cannot be relabelled afterwards. WebGym stages without
`--config-names`, so the repo holds a SINGLE derived cohort config and one `unstage` call reads all
of it. `stage` also refuses a non-empty output dir (and with `--overwrite` deletes it), so there is
no "append into the same directory" path:

```bash
# 1. pull the published repo, then unstage it back into a rollout log-root
uv run python -m lite.data.hf.download WebGym --org "$HF_ORG" \
  --out "${READBACK_ROOT}/cua-lite/WebGym"
uv run python -m lite.data.hf.unstage \
  --dataset "${READBACK_ROOT}/cua-lite/WebGym" --splits train \
  --log-root ".data/rollout/webgym/gpt5_5/$COMMIT/published_clean"
```

Then run the stage below with `published_clean/train` substituted for the
per-tier `*_clean` roots it replaces (`stage` walks a root with
`rglob("trajectory.parquet")`, so the reconstruction enters as one root), plus
every newly cleaned root.

Provenance note: after an unstage→re-stage cycle the card's `## Notes` names the RECONSTRUCTED
log-roots, not the original rollout roots.

```bash
# --- host ---
export CUA_LITE_DATASETS_ROOT="$PWD/.data/huggingface"
READBACK_ROOT="$PWD/.data/huggingface-readback"
COMMIT=a2cad60b   # same pinned batch id as §1 (the log-root version)

# step 3: stage all cleaned roots → canonical cua-lite/WebGym (NO filter — filtering was done in
#   the teacher runbook's step 2; stage has no hidden success predicate; content-addresses images;
#   writes repo.json).
#   Record the final seen/kept/dropped_by_filter line and per-config row lines:
#   this is the row-content validation gate.
#   --log-roots takes many roots → ONE
#   dataset, so growing the corpus = append more cleaned roots here (see Scale to 5k).
uv run python -m lite.data.hf.stage \
  --log-roots .data/rollout/webgym/gpt5_5/$COMMIT/d1_clean .data/rollout/webgym/gpt5_5/$COMMIT/d2_clean \
              .data/rollout/webgym/gpt5_5/$COMMIT/d3_clean .data/rollout/webgym/gpt5_5/$COMMIT/d4_clean \
              .data/rollout/webgym/gpt5_5/$COMMIT/d5_clean .data/rollout/webgym/gpt5_5/$COMMIT/d6_clean \
              .data/rollout/webgym/gpt5_5/$COMMIT/d7_clean .data/rollout/webgym/gpt5_5/$COMMIT/popular_clean \
  --name WebGym \
  --repo-dir devs/data/webgym

# step 4: upload to a private smoke repo. Upload is transport only: it packages,
#   pushes, and tags the staged tree; it does not replace stage validation.
#   --tag defaults to the current HEAD short commit; pass --tag $COMMIT explicitly
#   so the HF revision matches the pinned batch id. Use the release org only for
#   the approved final publish.
: "${HF_ORG:?set HF_ORG to your Hub user/org for the private smoke repo}"
uv run python -m lite.data.hf.upload WebGym --org "$HF_ORG" --private --tag "$COMMIT"

# step 5: read back the pinned revision and run a small conversion smoke (below).
uv run python -m lite.data.hf.download WebGym \
  --org "$HF_ORG" \
  --revision "$COMMIT" \
  --out "${READBACK_ROOT}/cua-lite/WebGym"
```

### 4. Export SFT Parquet

A small conversion smoke on the read-back revision:

```bash
uv run python -m lite.train.export.export_sft \
  --config scripts/configs/qwen3_5/default/webgym.yaml \
  --model-id Qwen/Qwen3.5-9B \
  --data-paths "${READBACK_ROOT}/cua-lite/WebGym" \
  --image-root "${READBACK_ROOT}" \
  --head 10 \
  --num-proc 1 \
  -o .data/sft/qwen3_5/webgym-smoke/train.parquet
```

Then distill + eval per the **Consumer** flow in
[/docs/examples/rollout_to_hf.md](/docs/examples/rollout_to_hf.md#consumer--train-from-the-hub)
(`download → export_sft → run_sft`, then base-vs-SFT on the held-out webgym eval split).

## Bookkeeping

- **Dedup:** persist consumed `task_id`s across runs; exclude the **eval split** to prevent
  leakage. Collection uses `seed 1` (smoke/exploration used `seed 0`); still dedup post-hoc.
- **n_samples/task = 1** for all tiers including the curated `popular` pool (`--group-size 1` —
  coverage over diversity).
- **Paths:** curated collection under `.data/rollout/webgym/gpt5_5/<commit>/` — one dir per rollout
  BATCH, named by a cua-lite commit pinned once at batch start (= the HF tag at upload; see §1).
  Transient exploration under `.logs/rollout/webgym_explore/`. Keep `<commit>/d<d>_clean` until staged.

## Scale to 5k — additive

The run writes per-tier `.data/rollout/webgym/gpt5_5/<commit>/d<d>` dirs; the published dataset is
`stage --log-roots <all <commit>/d<d>_clean dirs>` (many roots → one dataset). To grow the corpus
WITHIN a batch, collect more tasks into the same `<commit>/` dirs and re-stage; a new recipe ⇒ pin a
new `<commit>/` at the next batch start + a new HF tag. **Dedup is collection-side, not staging-side**:
`stage` does NOT dedup — it keeps every trajectory row (so e.g. the tier↔popular overlap lands as
multiple rows; `hash_split(task_id)` only guarantees all rows of
one task share a split, no train/eval leakage). To avoid re-collecting a task, persist consumed
`task_id`s and exclude them at rollout time (see Bookkeeping). Extend toward 5,000 (e.g. easy 1,250 /
medium 3,000 / hard 750). **Hard at 750 is the hard part**: at the measured ~25% d7 yield that's
~3,000 attempts, but the d7-only site-start pool is just ~1,392 (the full `difficulty>=7` pool is
~3,007 incl. d8+) → needs a `difficulty>=8` pass and/or `n_samples=2` (duplicate tasks, lower
diversity), or accept fewer hard. Medium (3,000 of a 51k pool) scales freely. Re-confirm hard yield
before committing to 750 — if it stays ~10–15%, cap hard lower and let medium absorb the rest.

## Current status

- Canonical row validation now requires nested Lite tool calls (`id/type/function`) and hard-fails
  null or missing `function.arguments`; legacy flat/provider-envelope repair belongs under
  `devs/migration`, not `lite/data`.
