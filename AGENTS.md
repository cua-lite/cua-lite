# AGENTS.md

This is the repo-local operating guide for AI coding agents. Keep this file as
the short execution contract: it is self-contained, and must not depend on
scratch planning documents, which come and go.

## Non-Negotiables

- Read the surrounding code before editing. Reuse the owner module and the local
  file-family style instead of inventing a parallel path.
- Do not `git add` or commit unless the user explicitly asks.
- Do not modify `slime/` unless the user explicitly asks.
- Do not pull top-level `README.md` or `docs/**` into readability/cleanup slices.
  Touch them only when the user directly scopes those files.
- Do not modify `.venv/`, `__pycache__/`, `.pytest_cache/`, or other generated
  caches.
- Do not use any memory system. Do not read from or write to a memory directory.
- Never stop, kill, exec into, or otherwise interfere with a Docker container or
  server you did not start. Clean up only resources owned by your session/token
  scope.
- Markdown links to repo files should use repo-root absolute paths, such as
  `/pyproject.toml`, so they work from any GitHub page.

## Engineering North Star

Use these code-shape rules before choosing a local fix. They are meant to keep a
small contract cleanup from growing into new aliases, hidden behavior, or a
general loop that knows too much about one downstream case.

### One Owner, One Vocabulary

Each durable fact should have one owning module and one representation. If two
shapes express the same fact, fix the producer or shared contract instead of
adding another branch.

Do not create distributed aliases for one concept. This includes near-synonym
names, helpers that only rename an operation, batch-tool / GUI-batch aliases for
action-batch, and native-wrapper vocabulary outside provider/model-family
projections.

Do not keep old spellings alive through import re-exports, wrapper functions,
test-only aliases, or facade shims unless the old path is documented public API;
otherwise update callers directly.

When naming a concept, choose the layer first and keep that spelling everywhere:
canonical Lite contracts, provider projections, env/server transport, and
debug/log artifacts are different layers. Do not let one layer's vocabulary leak
into another layer unless the code is explicitly projecting between them.

### No Implicit Backward Compatibility

Do not preserve backward compatibility for internal contracts, old helper names,
pre-refactor tests, or unpublished data shapes. Once a contract is chosen, make
the repo use that contract directly. Update callers, fixtures, and tests instead
of adding compatibility branches, fallback readers, or defensive "old shape"
handling.

Legacy repair is allowed only in the explicit migration/preproc owner for
already-published external data, and it must end before normal runtime,
training, rollout, stage, or export readers. Do not add compatibility shims to
general loops, adapters, or core facades unless the old path is documented public
API and the user explicitly scopes support for it. Treat unrequested backward
compatibility as over-defense: it keeps invalid states representable and spreads
the old contract into new code.

### Boundary Ownership

Keep general layers general. Shared runtime, core, and adapter code must not
carry downstream-specific escape hatches for one model family, env, dataset, or
rollout quirk. Put specific behavior at the owning boundary, then call that
owner from the general loop.

General loops should be downstream-agnostic. They may call a narrow owner hook
only when current production callers prove the hook owns a real shared boundary
decision; otherwise keep simple local logic direct. Loops should not contain
concrete model-family names, env IDs, dataset names, provider incident
workarounds, or special cases for one rollout source. Future or near-term users
are not evidence.

### Proportional Design

Simple changes should stay direct. A small mechanical migration should read like
a small mechanical migration: direct field access, direct builder calls, and
local control flow are often clearer than extra helper layers.

Before adding a helper, answer all three questions:

1. Does it own a real policy, boundary transformation, or repeated nontrivial
   logic?
2. Is it used by more than one real downstream owner, not just one call site or
   one test?
3. Is its name more revealing than the code it replaces?

Add a helper only when all three answers are yes. If any answer is no, keep the
logic local. Do not add helpers whose only effect is to rename, unpack/repack,
forward one call, or hide a one-line selection such as `indices[-1:]`, a direct
constructor call, or `call["function"]` access.

Large pure-addition diffs are suspicious for migrations and cleanup. When the
task is mostly `flat -> nested`, readability, or owner cleanup, expect many
edits to be direct replacements, deletions, or small local restructures. New
frameworks, compatibility branches, and broad helper surfaces need explicit
justification.

Net simplification is evidence that the owner and contract likely became
clearer, not a score to optimize. Do not delete structure, names, tests, or
comments just to reduce diff size if they still carry real policy, boundary
meaning, or user orientation.

### Helper Placement

Helper placement follows ownership, not convenience.

Before creating a helper module, name the semantic owner. If the answer is only
"two files need it," keep looking; shared call sites do not define ownership.

- Pure path, formatting, collection, or serialization helpers with no runtime
  policy may live in `lite/utils`.
- Cross-layer Lite contracts belong under `lite/core`; narrow non-tutorial
  helpers may live under `lite/core/utils/*`.
- Remote env HTTP policy belongs under `lite/gym/remote`.
- Provider-independent agent runtime behavior belongs under `lite/agents/core`.
- Provider-independent agent-loop helpers belong under
  `lite/agents/core/agent/utils`.
- Logger/debug/artifact decoration helpers should stay under the logger or agent
  utility area, not beside primary runtime abstractions.
- Model-family-specific behavior belongs under
  `lite/agents/models/<family>/utils/*`.
- Local entrypoint edge helpers should stay local unless several entrypoints
  share the same domain contract.
- `lite/agents/models/<family>/` should expose only `action_space.py`,
  `adapter.py`, `agent.py`, `protocol.py`, `__init__.py`, and `utils/*`.
- Core first-level files are tutorial surfaces. Keep primary abstractions at
  first level; move narrow helpers into a named utility package only when that
  makes the first read clearer.

## Contract Hygiene

Use this section when reviewing or repairing a split contract.

### Finding Test

A contract finding needs both parts: two representable shapes for the same fact,
and a real consumer cost caused by that split. "Nothing checks X" is not enough.

### Repair Order

Use this ladder in order:

1. Make the invalid state unrepresentable.
2. Fix the producer that creates the invalid state.
3. Add a guard only when levels 1 and 2 are impossible and the failure would
   otherwise be silent.

### Guards And Over-Defense

Apply these checks before writing `assert`, `validate`, `guard`, or
`bounds-check` code. Do not add guards, fallbacks, broad exception catches, shape
sniffers, or compatibility branches just because a bad input is imaginable.

- Prove the bad state is reachable.
- Prove downstream would not already fail loudly through a crash, an existing
  test, or a visibly wrong loss curve.
- Prove the failure would otherwise be silent or misleading.
- Put the check at the narrowest owner, not a downstream caller that can only
  guess what went wrong.
- Prefer domain constraints over pairwise consistency checks. A field's legal
  values belong at construction; comparing two independently derived values is
  last-resort guard work.
- Treat an env rejection as silent only when no `result_call_id` reaches the
  model. If the rejection is model-visible, record or propagate it instead of
  adding a hidden guard.
- If a boundary already validates the contract, downstream code should use the
  contract directly. If a producer can be fixed, fix the producer instead of
  teaching every consumer to survive both shapes.
- Defensive code is appropriate at real boundaries: user CLI/config input,
  external APIs, provider responses, file I/O, env/server transport, and
  published data ingestion. Even there, prefer one clear boundary check and one
  clear error over parallel compatibility logic.

Canonical row/response validation has named owners. `stage.py` is the
canonical row-content gate before upload/training datasets, and rollout
`--debug` plus `lite.infer.debug.log_contract` is the debug/log response gate.
Do not turn `upload.py`, `export_sft`, collect filters, or ordinary training
readers into second row validators. They may fail on inputs they cannot read,
but they should not grow broad canonical-shape gates, old-shape repair, or
strict response validation that belongs to stage/debug.

### Rework Smells

- Treat escape-hatch flags/env vars and heuristic comparisons that hard-fail as
  disqualifying smells for guard-shaped fixes. They prove or strongly suggest
  the proposed guard rejects legitimate flows.
- If a contract cleanup is mostly additions, re-check whether the producer or
  shared contract was fixed instead of adding another consumer path.
- Monkeypatches, compatibility shims, and parallel logic usually mean the owner
  was not fixed. Apply `Single Owner, Single Vocabulary` unless the old path is
  documented public API.

## Readability

Use this section for readability-only work. Semantic changes need their own
scope and tests.

### Preserve Behavior

Readability work must not change behavior: no prompt/tool-surface drift, no
training or rollout semantic change, no hidden compatibility shim. If cleanup
needs behavior to change, make it a separately scoped semantic fix with tests.

For `scripts/train/run_*.sh`, executable-line movement needs before/after
fake-run equivalence for launch argv, driver env, runtime-env JSON, service
order, checkpoint/dump paths, entrypoint, and submit semantics.

### Product And Entry Surfaces

CUA-Lite is an external open-source project. Public entry files, root metadata,
package facades, model families, and package-local READMEs are product surfaces.
They should keep the user-facing command/import, required inputs, important env
vars, main control flow, and implementation handoff visible near the top.

- Optimize the files users and contributors touch first: launchers, serving
  scripts, rollout/training entrypoints, public facades, root metadata,
  package-local READMEs, and model `agent.py` / `adapter.py` /
  `action_space.py` / `protocol.py` files.
- A user-facing entry should show purpose, minimal command/import, required
  inputs and important env vars, visible control flow, and the specific
  function/module called next.
- CLI files should put `_parse_args()` near the top because flags are the
  user-facing contract. The main flow should read as `parse -> resolve ->
  validate -> run`, or the local equivalent.
- Keep sibling entrypoints mechanically comparable: same section order, banner
  style, validation style, and option formatting unless their lifecycle truly
  differs.
- Headers, examples, and first-screen comments are part of the product. Preserve
  useful orientation; remove or rewrite dated incident notes in hot paths.
- Training launchers must keep operator-facing overrides, launch parameter
  arrays, and the final `ray job submit` command visible. Extract only repeated
  validation or formatting that does not hide the submitted command.
- `__init__.py` files are entry surfaces only when the package intentionally
  exposes a facade. Public re-export and lazy facades should define `__all__`;
  empty or docstring-only package markers do not need it. Keep package
  initializers import-light unless eager import is part of the public contract.

### Loop Readability

Loops are product surfaces when they define rollout, training, provider, or env
lifecycle behavior. Keep their state machines readable:

- In loops that cross provider, env, trajectory, or persistence boundaries, put
  short phase comments in the loop body before each major transition: prepare
  input, call provider, parse output, step env, record trajectory, and prepare
  next turn.
- Use short comments for non-obvious invariants: which image the model saw, why
  a result is model-visible, when provider history differs from Lite storage,
  and what state is carried across turns.
- Keep parse errors, provenance, terminal accounting, and history mutation in
  separate visible blocks; avoid updating the same state from both success and
  exception branches.
- Apply Proportional Design inside loops: inline one-branch decisions; extract
  lifecycle code only after multiple model families share the same transition.

### Comments and Docstrings

- Comments and docstrings should state current behavior and active policy, not a
  changelog.
- A wrong docstring is worse than no docstring. State only behavior that can be
  checked against code, tests, or owning config.
- Before closing a comment/docstring cleanup slice, scan touched public files for
  stale intent markers such as `TODO`, `FIXME`, `HACK`, `stale`, `historical`,
  `incident`, `temporary`, dated notes, and `DO NOT`; keep only current policy.
- Add docstrings to public entrypoints, public facades, registry leaf classes,
  and non-obvious helpers that a reader is likely to navigate through. Do not
  mechanically add docstrings to every private helper, test helper, generated
  file, or empty `__init__.py`.
- If a docstring describes prompt/tool surface, env lifecycle, persistence,
  retries, schema shape, or training semantics, verify the code path before
  editing and run a targeted test or import/compile check afterward.

## Python Style

These rules apply to first-party root code. Do not churn vendored code, Docker
patches, generated files, or `slime/` to satisfy them.

- Use modern type hints: `dict`, `list`, `tuple`, `set`, `type`, and `X | None`
  rather than `Dict`, `List`, or `Optional`.
- Use `from __future__ import annotations` for new or nontrivial first-party
  modules.
- Formatting, import sorting, and line length are governed by
  [pyproject.toml](/pyproject.toml) and the configured tools.

## Running Code

- Outside Slime Docker, use `uv run python` for Python scripts. Inside Slime
  Docker or other configured containers, use the interpreter expected by that
  container.
- Shell-script invocation is script-specific. Follow the script header or the
  relevant docs; do not generalize env installer/lifecycle rules to every shell
  script.
- For GPU work, inspect available devices with the host-appropriate tool
  (`nvidia-smi` on NVIDIA hosts) and set `CUDA_VISIBLE_DEVICES` when needed. Do
  not hardcode CPU or GPU counts into repo guidance.
- Stop training/eval jobs gracefully before escalating to hard kills. Hard-killed
  accelerator processes can leave device state wedged until reset.
- For long-running work, use the available background/session runner and report
  progress periodically. Do not block the foreground on multi-minute jobs.
- For large data analysis (`.pt`, `.jsonl`, `.parquet`, images, etc.), process
  files one at a time and release memory after extracting the needed values.

## Environments and Rollout

- Local in-process rollout is the default low-friction path when the env docs say
  it is supported.
- Prefer the env-server for Slime training, multi-tenant hosts, heavy-env /
  light-agent splits, crash isolation, and coordinated large rollout runs.
- For coordinated rollout work, the coordinator owns the env-server lifecycle and
  provides `CUA_LITE_ENV_SERVER_URL` plus `CUA_LITE_ENV_SERVER_TOKEN`. Subagents
  may inspect logs/artifacts or query an already-running server when given those
  values, but must not start, stop, or restart env-servers.
- To audit cleanup/reaping, compare server-owned instances and scoped Docker
  resources using the naming rules in `lite/gym/utils/config/naming.py`.
  `/host_status` is useful for server identity and health; it is not the cleanup
  source of truth.
- After a run, clean up only resources owned by your server/session/token scope.
  Never remove co-tenant containers or servers.

## Slime

- Slime requires Docker. See [docs/slime.md](/docs/slime.md).
- Verify backend assumptions in the current training launcher before changing
  Slime or launch code.

## Testing

- Default pytest parallelism and marker filters live in
  [pyproject.toml](/pyproject.toml). On the normal host/dev venv path, use
  `uv run pytest` for the default suite; inside Slime or another configured
  container, use that container's expected pytest/Python invocation.
- The default is `-n 32`, already in `addopts`. Do not use `-n auto`. Override
  with an explicit `-n <workers>` only when the host is smaller than that, and
  use `-n 0` for a serial run when debugging a single test. Do NOT reach for
  `-p no:xdist`: it unloads the plugin that owns `-n`, so the `addopts` default
  becomes an unrecognized argument and pytest refuses to start.
- Live and stress tests are excluded by default through pytest markers. Opt in
  explicitly with marker selection when the required env-server, Docker daemon,
  credentials, or other live resource is available.
- Run targeted tests first. Run the relevant broader suite when Python behavior,
  shared contracts, prompt/tool surfaces, training, rollout, or public entrypoints
  change.

## Subagents

Use subagents for broad audits, sweeps, and verification work. Split them by
concept rather than directory, and keep each assignment bounded.

- Use several small, uniform tasks instead of one whole-repo mega-task.
- Subagents are read-only by default. They may edit only when given an explicit
  scoped implementation assignment with bounded files. Unscoped or audit
  subagents may read files and run checks only; they must not edit, `git add`, or
  commit.
- The main agent remains responsible for the result: independently verify
  material subagent findings in the main workspace before reporting or acting on
  them, especially deletions, zero-caller claims, and compatibility claims.
- Failed, timed-out, or incomplete subagent runs do not count as review. Rerun or
  replace them if their coverage is required; otherwise report the missing
  coverage.

## Git and Worktree

- The worktree may be dirty. Do not revert unrelated or user-authored changes.
  If a precise revert is requested, revert only the requested change.
- Before making an explicitly requested commit that modifies Python behavior,
  ensure the relevant tests pass first. If tests fail, report the failures
  instead of committing.
