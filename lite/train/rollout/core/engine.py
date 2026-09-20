"""Async rollout driver + batch assembly for CUA-Lite RL training (GRPO, REINFORCE, etc.).

Drives multi-turn agent rollouts in the env (``generate``), orchestrates the
concurrent rollout / eval loops with cooperative abort (``generate_rollout``),
and assembles the surviving trajectories into slime's train_data dict
(``bucket_trajectories`` / ``flatten_and_align``). The pure steps→Sample-span
packing lives in :mod:`lite.train.rollout.core.segmenter` (imported here);
prefix-extensional protocols can pack adjacent turns into one multi-span
Sample, while windowed/Markov protocols fall back to length-1 samples.

Key design decisions (vs geo3k unfold rollout):
  - Uses agent.sample() directly: the adapter handles multi-turn prompt
    construction, so we don't need manual context accumulation.
  - The agent returns a generic RL step stream: each step supplies rendered
    prompt text, ordered image_indices, response text, and response tokens; the
    segmenter owns prompt tokenization and radix packing.
  - Custom generate_fn: calls SGLang HTTP with return_logprob=True using "text"
    key (not "input_ids") to avoid image placeholder pre-expansion bug.
  - Env-based reward: set in generate(), skips slime RM pipeline.
  - Async envs: CUA-lite envs are async (Docker containers + browsers).

Internal module — not a slime entry point. The per-algorithm shims
(``lite.train.rollout.{grpo,reinforce,dagger,sft}``) bind ``generate`` /
``generate_rollout`` from here via ``core/__init__.py``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import statistics
from argparse import Namespace
from collections.abc import Callable
from typing import Any

from packaging.version import parse
from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.rollout.sglang_rollout import GenerateState, generate_and_rm
from slime.utils.async_utils import run
from slime.utils.http_utils import get, post
from slime.utils.misc import load_function
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample
from tqdm import tqdm

import lite.gym as gym
from lite.agents.bootstrap import register_all
from lite.agents.core.adapter import tool_surface_agent_kwarg_names
from lite.agents.models import AgentRegistry
from lite.utils.registry import compose_key, split_key
from lite.gym import finalize_env_kwargs
from lite.gym.errors import CapacityExhausted
from lite.gym.errors import failure_category as gym_failure_category
from lite.gym.wrappers import EnvTimeoutError
from lite.train.rollout.core.segmenter import build_segment_samples
from lite.utils.config import deep_merge
from lite.utils.timer import Timer

logger = logging.getLogger(__name__)


# Stall-detection timeout for ``asyncio.wait`` calls in the rollout / eval
# loops below. If no task completes within this many seconds, the entire
# rollout is treated as stalled (typically a single hung env container that
# never returns from a synchronous call) — pending tasks are cancelled and
# the function raises so the outer training launcher can resume from the latest
# checkpoint instead of hanging forever. Override with ``ROLLOUT_STALL_TIMEOUT_S``;
# default 1800 s (30 min) — comfortably above the slowest legitimate
# lite.osworld episode (~6 min) and androidworld episode (~10 min).
_ROLLOUT_STALL_TIMEOUT_S = float(
    os.environ.get("ROLLOUT_STALL_TIMEOUT_S", "1800")
)


# Module-level per-group seed cache. Populated lazily on first sample of a
# group in generate(); cleared at the start of each _generate_rollout_async()
# iteration so the next rollout_id redraws fresh seeds. Keyed by
# ``sample.group_index`` (slime's per-prompt group identifier, set by the
# standard ``RolloutGroupGenerator`` data source) — NOT by ``env_key``,
# since a single env_key can back multiple groups within one iteration when
# the same prompt is sampled more than once (or when rollout_batch_size
# exceeds the unique task count). All siblings within a group share the
# same env initial state (task_params / noise) while distinct groups draw
# independent seeds. Shared by GRPO and REINFORCE (both import generate /
# generate_rollout from this module). Eval samples are NEVER re-seeded
# here — see generate() for the guard.
_group_seeds: dict[int, int] = {}

# Set by ``_generate_rollout_async`` at the top of each iteration so
# ``generate()`` can derive seeds from ``(rollout_id, group/index)``
# using a dedicated ``random.Random``. That keeps seeds independent of
# asyncio scheduling order. ``None`` outside an active rollout iteration.
_current_rollout_id: int | None = None

# Count of entirely-empty trajectories (all turns ``tokens=[]``) in the most
# recent ``_generate_rollout_async`` call, captured BEFORE ``_drop_empty_turns``
# erases them from the flat sample list (which it must, to keep slime's
# sample-count view in sync with the convert output — see _drop_empty_turns).
# ``bucket_trajectories`` reads this to attribute those failures to
# ``n_errored`` (a real env/judge failure that produced a Sample) rather than
# ``n_missing`` (a framework-level drop that produced nothing). Mirrors the
# ``_current_rollout_id`` / ``_group_seeds`` module-state pattern: generate →
# convert is strictly sequential and 1:1 per rollout, so a single int is safe.
_last_rollout_n_errored: int = 0

# Env-side capacity backpressure. Read by ``_classify_failure`` below and again
# to suppress its traceback. ``lite.train.filters`` drops the group on the same
# value but hardcodes its own copy on purpose: slime loads that filter by dotted
# path, so it must import without this module's deps (ray / sglang_router).
CAPACITY_FAILURE_REASON = "capacity_503"


def _count_empty_trajs(grp) -> int:
    """Number of entirely-empty trajectories in one rollout group.

    Pairs with ``_drop_empty_turns`` (same nesting selector): a trajectory is
    "empty" iff every one of its turn-samples has ``tokens == []`` — exactly
    the trajectories ``_drop_empty_turns`` collapses to ``[]`` (so
    ``_regroup_trajectories`` can't reconstruct them). Module-level so the
    multi-turn nested branch is unit-testable.

    Nested (multi-turn agent): ``grp`` is a list of trajectories, each a list
    of turn-samples. Flat (single-turn): ``grp`` is a list of single-sample
    trajectories.
    """
    if grp and isinstance(grp[0], list):
        return sum(1 for traj in grp if not any(s.tokens for s in traj))
    return sum(1 for s in grp if not s.tokens)


# ---------------------------------------------------------------------------
# Env registry (module-level)
# ---------------------------------------------------------------------------

_active_envs: set = set()

def _register_env(env):
    _active_envs.add(env)

def _unregister_env(env):
    _active_envs.discard(env)

# ---------------------------------------------------------------------------
# Abort signal
# ---------------------------------------------------------------------------

class AbortError(Exception):
    """Raised by generate_fn when the rollout is aborted."""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _empty_sample(
    original_sample: Sample,
    failure_reason: str | None = None,
    retryable: bool = True,
) -> Sample:
    """Return a minimal failed sample when the env errors out.

    Args:
        original_sample: Sample whose group_index / prompt / label are
            preserved so slime's pipeline can still attribute the
            failure to its group.
        failure_reason: Optional stable reason — one of ``"capacity_503"``,
            ``"reset_timeout"``, ``"step_timeout"``, ``"task_crash"``,
            ``"aborted"``, or ``None`` for unclassified.
            Stored under ``metadata["lite_failure_reason"]``. Its consumer is
            ``lite.train.filters.drop_capacity_failed_groups``, selected by dotted
            string via ``--dynamic-sampling-filter-path`` (see ``load_function``
            below) — NOT by any Python import, so a grep for callers will not find
            it. ``grpo.py``'s ``DROP_ZERO_STD_GROUP`` is a *different* mechanism,
            not a replacement: it swaps zero-std groups for dummies to skip
            compute without re-drawing, and its gradient-identity holds only
            under the conditions its own ``drop_safe`` guard enumerates.
        retryable: Whether re-running this trajectory might recover — the cached
            result of ``is_retryable(exc)`` evaluated where the exception was live
            (``generate``'s except). Stored under ``metadata["lite_error_retryable"]``
            so the eval retry loop (which no longer has the exception) can gate on
            it. Defaults ``True``; only the exception path computes it (no-turns /
            aborted are transient by construction).
    """
    metadata = dict(original_sample.metadata or {})
    if failure_reason is not None:
        metadata["lite_failure_reason"] = failure_reason
    metadata["lite_error_retryable"] = retryable
    return Sample(
        group_index=original_sample.group_index,
        index=original_sample.index,
        group_id=original_sample.index,
        prompt=original_sample.prompt,
        tokens=[],
        loss_mask=[],
        rollout_log_probs=[],
        response_length=0,
        response="",
        label=original_sample.label,
        reward=0.0,
        status=Sample.Status.FAILED,
        metadata=metadata,
    )


def _classify_failure(exc: BaseException) -> str:
    """Categorize a generate-time exception into a stable failure label."""
    category = gym_failure_category(exc)
    if category is not None:
        return category.value

    if isinstance(exc, CapacityExhausted):
        return CAPACITY_FAILURE_REASON
    if isinstance(exc, EnvTimeoutError):
        # Distinguish reset-timeout (heavier — usually env-boot path)
        # from step-timeout (lighter — usually intra-episode hang) so
        # operators can see which is dominant in the typed counters.
        return f"{exc.phase}_timeout"
    # Anything else: most often a real env-side crash (None deref, missing
    # data, agent producing malformed actions). Distinct from the
    # capacity / timeout categories so P2's metrics don't conflate them.
    return "task_crash"

def dummy_sample(original_sample: Sample) -> Sample:
    """Create a dummy sample that passes through slime's pipeline without
    affecting training (loss_mask=[0] → zero gradient).

    Used to maintain batch-size alignment when real samples are dropped.

    tokens must have at least 2 elements (1 prompt + 1 response) because
    slime computes ``prompt_length = total_length - response_length`` and
    pads loss_mask with ``(prompt_length - 1, 1)``.  A single-token sample
    would give prompt_length=0 → negative padding → crash.
    """
    return Sample(
        group_index=original_sample.group_index,
        index=original_sample.index,
        group_id=original_sample.index,
        prompt=original_sample.prompt,
        tokens=[0, 0],  # 1 prompt token + 1 response token
        loss_mask=[0],
        rollout_log_probs=[0.0],
        response_length=1,
        response="",
        label=original_sample.label,
        reward=0.0,
        status=Sample.Status.FAILED,
        metadata=dict(original_sample.metadata or {}),
    )


def synthetic_dummy_sample() -> Sample:
    """A from-scratch zero-gradient dummy needing NO source sample — for the
    empty/all-failed batch guards. ``index=0`` so a later
    ``flatten_and_align`` pad (``base = max(index)+1``) never sees a ``None``."""
    return dummy_sample(Sample(group_index=0, index=0, group_id=0))

# ---------------------------------------------------------------------------
# Generate with cooperative abort + env registry
# ---------------------------------------------------------------------------

async def generate(args: Any, sample: Sample, sampling_params: dict,
                   evaluation: bool = False, *, relabel_fn=None) -> list[Sample]:
    """CUA-Lite multi-turn rollout using agent.sample().

    Returns a list of segment Samples. A segment may cover one completed step
    or several prefix-extensional steps packed by ``build_segment_samples``.

    ``relabel_fn`` (DAgger, default None): an awaitable
    ``relabel_fn(args, rl_sample, sample, agent, agent_key) -> rl_sample`` applied
    AFTER ``agent.sample()`` and BEFORE ``build_segment_samples`` — it swaps each
    step's SFT *target* to a teacher's generated action (the student's executed
    actions / context history are untouched). Skipped when ``evaluation`` (eval only
    needs the student's env success; relabeling there is wasted and would corrupt the
    eval). ``relabel_fn=None`` ⇒ byte-identical to the non-DAgger path.
    """
    if getattr(args, "partial_rollout", False):
        raise ValueError("Partial rollout is not supported for unfold rollouts.")

    t = Timer()

    state = GenerateState(args)
    tokenizer = state.tokenizer
    processor = state.processor
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    sample.metadata = sample.metadata or {}
    env_key = sample.metadata["env_key"]

    # One wide try/except wraps env creation, metadata access, agent lookup,
    # and agent.sample(). Any normal ``Exception`` here becomes an
    # ``_empty_sample`` and is counted as ``n_trajs_errored`` downstream.
    #
    # ``asyncio.CancelledError`` inherits from ``BaseException`` (not
    # ``Exception``), so it still propagates — cancellation by slime's
    # rollout controller remains a legitimate ``n_trajs_missing``.
    env = None
    try:
        # Env kwargs precedence: prompt_data row > args (yaml is merged
        # into args upstream by slime). Per-key DEEP merge (matches
        # rollout's ``_resolve_env_kwargs``) — args supplies the run-wide
        # baseline; prompt_data ``metadata.env_kwargs`` (optional) overrides
        # per row, winning only at the named nested leaf. Final normalization
        # (strip Nones left by pyarrow roundtrip, ``resolution`` → tuple) is
        # delegated to the shared :func:`finalize_env_kwargs` so rollout and
        # train pipelines stay byte-identical at gym.make.
        env_kwargs = finalize_env_kwargs(
            deep_merge(
                getattr(args, "env_kwargs", None) or {},
                sample.metadata.get("env_kwargs") or {},
            )
        )

        split = sample.metadata.get("split", "")

        if not getattr(args, "group_shared_seed", True) and split == "eval":
            logger.warning("group_shared_seed=false has no effect on eval splits "
                           "(they use registered deterministic seeds, e.g. seed=42).")

        # Training rollout: one shared env seed per group per rollout_id.
        # Keyed by ``sample.group_index`` (slime's per-prompt group identifier
        # set in data_source.py) — siblings of the same group get the same
        # cached seed → identical env initial state (task_params / noise),
        # so GRPO's group baseline is well-defined (mathematically required —
        # otherwise advantage = R - group_mean conflates policy stochasticity
        # with initial-state variance). Distinct groups (even backed by the
        # same env_key) draw independent seeds so cross-group exploration is
        # preserved. Cache cleared at the start of each _generate_rollout_async()
        # call, so next rollout_id redraws.
        #
        # Default ON — set `group_shared_seed: false` in the custom config yaml
        # to disable (e.g. for A/B ablation). Also skipped when:
        #   (a) eval split — its registered seed (e.g. seed=42) is authoritative
        #   (b) user already pinned a seed via env_kwargs
        #   (c) the env does not declare ``seed`` as an accepted soft kwarg
        #       (registry capability gate — envs that ignore it must not be
        #       handed one).
        if (
            getattr(args, "group_shared_seed", True)
            and split != "eval"
            and "seed" not in env_kwargs
            and gym.registry.env_supports_kwarg(split_key(env_key)[0], "seed")
        ):
            # Derive deterministically from (rollout_id, group_index) so the
            # same group in the same iteration gets the same seed regardless
            # of which async task first hits this branch.
            # ``_current_rollout_id`` is always set by
            # ``_generate_rollout_async`` before this branch runs (the only
            # non-eval entry point).
            assert _current_rollout_id is not None, (
                "non-eval generate() called outside _generate_rollout_async()"
            )
            if sample.group_index not in _group_seeds:
                _group_seeds[sample.group_index] = random.Random(
                    f"{_current_rollout_id}:{sample.group_index}"
                ).randint(0, 2**31 - 1)
            env_kwargs["seed"] = _group_seeds[sample.group_index]

        # Sampling seed — the per-MEMBER-DISTINCT twin of the per-group-SHARED env
        # seed above. Same derivation and same base (``_current_rollout_id``); only
        # the IDENTITY flips from the shared ``group_index`` to the globally-unique
        # ``sample.index`` (data_source.py), so a group's siblings sample DIFFERENTLY
        # — the variance GRPO's group baseline measures — while ``(rollout_id, index)``
        # reproduces it across runs. Always on (reproducibility is a baseline, not
        # gated): a caller that already pinned ``sampling_seed`` wins, and eval keeps
        # its own path. NOTE: this fixes the sampling DRAW; BIT-EXACT reproduction
        # additionally needs sglang deterministic inference (continuous-batching logit
        # jitter) — an orthogonal, perf-costly toggle left to the caller.
        if split != "eval" and "sampling_seed" not in sampling_params:
            assert _current_rollout_id is not None, (
                "non-eval generate() called outside _generate_rollout_async()"
            )
            sampling_params = {
                **sampling_params,
                "sampling_seed": random.Random(
                    f"{_current_rollout_id}:{sample.index}"
                ).randint(0, 2**31 - 1),
            }

        agent_kwargs = dict(getattr(args, "agent_kwargs", None) or {})
        # ``sampling_kwargs`` is a generate_fn concern, never an adapter kwarg —
        # the same split ``lite.infer.serving.extract_sampling_kwargs`` makes on
        # the rollout.py path. In TRAINING the trainer owns sampling entirely
        # (``--rollout-temperature`` / ``--eval-temperature``), which is what the
        # shipped configs' own comment says, so drop it rather than honor it.
        # Without this the adapter's strict unknown-kwarg check rejects every
        # config that carries the key — i.e. every config shared with rollout.py.
        agent_kwargs.pop("sampling_kwargs", None)
        surface_overrides = tool_surface_agent_kwarg_names(agent_kwargs)
        if surface_overrides:
            raise ValueError(
                f"training agent_kwargs contains tool-surface settings "
                f"{sorted(surface_overrides)}; pass resolved surface via env "
                "metadata, not agent_kwargs"
            )

        # Create env + register for abort tracking.
        # ``asyncio.to_thread`` offloads ``gym.make`` because in direct
        # mode it can do synchronous heavy work (``docker run`` for
        # lite.osworld / Sandbox, ``ensure_services`` start.sh for
        # browsergym/webgym) that would otherwise block this rollout
        # worker's event loop and starve sibling coroutines. In server
        # mode ``gym.make`` is just a ``LiteEnvClient`` constructor and
        # the ``to_thread`` wrap is near-free — so the same call site
        # works for both modes without API changes.
        env = await asyncio.to_thread(gym.make, env_key, **env_kwargs)
        t.mark("setup")
        _register_env(env)

        # Derive agent_key: per-sample override > config-based derivation.
        # ``sample.metadata`` is Slime/control metadata; the fallback route uses
        # the live env's Lite task metadata dims after env construction.
        metadata = env.metadata
        agent_key = sample.metadata.get("agent_key")
        if not agent_key:
            agent_id = getattr(args, "agent_id", None)
            if not agent_id:
                raise ValueError(
                    "Cannot derive agent_key: set `agent_id` in the training custom "
                    "config (--custom-config-path), or `agent_key` per prompt_data row."
                )
            agent_key = compose_key(agent_id, *metadata.dims)

        _generate_fn_timings: list[Timer] = []

        # Per-trajectory PNG-encode cache. K-window protocols share K-1
        # PIL refs across consecutive turns; without the cache each image
        # is PNG+base64-encoded T times. Keyed by id(pil) since the
        # trajectory owns the PIL objects (lifetime ≥ generate()), so id
        # is stable.
        _png_cache: dict[int, str] = {}

        async def _encode_one(img):
            key = id(img)
            cached = _png_cache.get(key)
            if cached is not None:
                return cached
            encoded = await asyncio.to_thread(encode_image_for_rollout_engine, img)
            _png_cache[key] = encoded
            return encoded

        async def generate_fn(*, prompt, images, **kwargs):
            """Custom generate_fn: SGLang HTTP with return_logprob=True."""
            if state.aborted:
                raise AbortError()

            gt = Timer()
            if images:
                image_data = list(await asyncio.gather(*[_encode_one(img) for img in images]))
            else:
                image_data = []
            gt.mark("encode")

            payload: dict = {
                "text": prompt,  # "text" avoids SGLang image placeholder pre-expansion bug
                "sampling_params": sampling_params,
                "return_logprob": True,
            }
            if image_data:
                payload["image_data"] = image_data

            output = await post(url, payload)
            gt.mark("infer")
            _generate_fn_timings.append(gt)
            response: str = output["text"]

            # Extract response token IDs and log probs from SGLang output
            meta_info = output.get("meta_info", {})
            if "output_token_logprobs" in meta_info:
                response_tokens = [item[1] for item in meta_info["output_token_logprobs"]]
                response_log_probs = [item[0] for item in meta_info["output_token_logprobs"]]
            else:
                response_tokens, response_log_probs = [], []

            # Strip trailing EOS/stop tokens.
            # SGLang HTTP with no_stop_trim=True includes the stop token
            # (e.g. <|im_end|>) in the response. If kept, apply_chat_template
            # adds ANOTHER <|im_end|>, causing double <|im_end|> in the next
            # turn's prompt.
            eos_id = tokenizer.eos_token_id
            while response_tokens and response_tokens[-1] == eos_id:
                response_tokens.pop()
                response_log_probs.pop()
                if response.endswith(tokenizer.eos_token):
                    response = response[:-len(tokenizer.eos_token)]

            finish_reason: str = meta_info.get("finish_reason", {}).get("type", "stop")

            return {
                "response": response,
                "response_tokens": response_tokens,
                "response_log_probs": response_log_probs,
                "finish_reason": finish_reason,
            }

        # Create agent (pass through optional overrides from custom config).
        # Forward the WHOLE env metadata object (extra_tool_schemas / valid_actions
        # / others) — exactly as make does (lite/agents/factory.py). These
        # are Lite task metadata fields, not adapter fields: passing them as loose kwargs
        # silently drops them (BaseAgentAdapter._apply_kwargs → unrouted), which
        # would render the RL student with no nav tools + an untrimmed enum,
        # mismatching the SFT surface. Same fix as export_sft.py.
        register_all()
        agent = AgentRegistry.get(
            agent_key,
            processor=processor,
            generate_fn=generate_fn,
            metadata=metadata,
            **agent_kwargs,
        )

        # ---- Run agent.sample(), then build Samples from rl_sample.steps ---
        # Reset/step timeouts raise EnvTimeoutError → caught below → _empty_sample.
        rl_sample = await agent.sample(env)
        t.mark("sample")

        # DAgger: swap each step's SFT target to the teacher's generated action
        # (on the student's own ``rl_sample`` — its executed actions + context
        # history stay untouched). Before segmenting, since ``build_segment_samples``
        # consumes ``step.response_tokens``. Skipped at eval (relabel_fn gated by the
        # caller; ``not evaluation`` belt-and-suspenders). None ⇒ no-op for GRPO/SFT.
        if relabel_fn is not None and not evaluation:
            rl_sample = await relabel_fn(args, rl_sample, sample, agent, agent_key)
            t.mark("relabel")

        # ``build_segment_samples`` runs ``processor`` (Qwen2VL fast HF
        # processor) per-step — pure CPU, blocking. Without to_thread, the
        # call blocks the asyncio event loop and serializes build phases
        # across all concurrent rollout tasks. ``processor`` releases the
        # GIL during numpy/torch ops, so the thread-pool path gives real
        # cross-trajectory parallelism (default executor has ~32 workers,
        # matching ENV_CONCURRENCY=32).
        turn_samples = await asyncio.to_thread(
            build_segment_samples,
            rl_sample, sample, processor, tokenizer, state,
        )
        t.mark("build")

        logger.info("⏱ generate %s: %s n_calls=%d", env_key, t, len(_generate_fn_timings))

        return turn_samples if turn_samples else [_empty_sample(sample)]

    except AbortError:
        logger.warning("Aborted generate() for %s", env_key)
        return [_empty_sample(sample, failure_reason="aborted")]

    except Exception as e:
        from lite.gym.errors import is_retryable
        reason = _classify_failure(e)
        # capacity_503 is expected backpressure under multi-tenant load —
        # don't drown logs in tracebacks for it (the env-server already
        # logs the typed 503 at INFO). Other categories get exc_info for
        # the operator.
        log_with_traceback = reason != CAPACITY_FAILURE_REASON
        logger.warning(
            "Error in generate() for %s [reason=%s]", env_key, reason,
            exc_info=log_with_traceback,
        )
        # Cache the retry verdict on the empty sample: the eval retry loop runs
        # after this exception is gone, so it reads metadata["lite_error_retryable"].
        return [_empty_sample(sample, failure_reason=reason, retryable=is_retryable(e))]

    finally:
        # env may be None if gym.make() raised before assignment.
        if env is not None:
            _unregister_env(env)
            # Ensure env is closed even if agent.sample() didn't reach its finally block
            try:
                await asyncio.wait_for(env.close(), timeout=60.0)
            except TimeoutError:
                logger.warning("env.close() timed out after 60s in generate() finally block")
            except Exception as e:
                # Server-mode DELETE failures (env-server unreachable mid-call)
                # land here. Don't crash the rollout, but make the failure
                # visible — silent swallow used to leave parked instances
                # holding pool slots indefinitely.
                logger.warning("env.close() failed in generate() finally: %s", e)

# ---------------------------------------------------------------------------
# Abort with env cleanup
# ---------------------------------------------------------------------------

async def _abort(args: Namespace, rollout_id: int) -> list[list[Sample]]:
    """Extended abort: clean up registered envs, then do standard SGLang abort."""
    aborted_samples: list[list[Sample]] = []

    state = GenerateState(args)
    if state.aborted:
        # Already aborted, just drain pending tasks. Bounded by stall timeout
        # so a hung pending task can't trap us here forever.
        while state.pendings:
            done, state.pendings = await asyncio.wait(
                state.pendings,
                return_when=asyncio.FIRST_COMPLETED,
                timeout=_ROLLOUT_STALL_TIMEOUT_S,
            )
            if not done:
                logger.warning(
                    "abort drain stalled after %ss; cancelling %d hung pending(s) and giving up",
                    _ROLLOUT_STALL_TIMEOUT_S, len(state.pendings),
                )
                for t in state.pendings:
                    t.cancel()
                state.pendings = set()
                break
        return aborted_samples

    state.aborted = True

    # ---- Actively clean up all registered envs ----------------------------
    # CUA-lite envs are async — close them in parallel. ``_active_envs`` is
    # a bare set; ``list()`` snapshots atomically under asyncio's single
    # event loop, but new envs can still register *during* the gather
    # below (a generate() that already passed _register_env but hasn't yet
    # hit AbortError in its generate_fn). Loop up to 3 passes to drain
    # any stragglers — bounded so a leak can't trap abort here forever.
    for _pass in range(3):
        snapshot = list(_active_envs)
        if not snapshot:
            break
        close_tasks = [_safe_close_env(env) for env in snapshot]
        try:
            await asyncio.wait_for(asyncio.gather(*close_tasks), timeout=120.0)
        except TimeoutError:
            logger.warning("Abort env cleanup timed out after 120s, %d envs may be orphaned",
                           len(close_tasks))
            break  # don't loop on a stuck close
    if _active_envs:
        logger.warning("Abort env cleanup left %d env(s) registered after 3 passes",
                       len(_active_envs))

    # ---- Standard SGLang abort (cancel pending inference requests) ---------
    try:
        import sglang_router

        if parse(sglang_router.__version__) <= parse("0.2.1") or args.use_slime_router:
            response = await get(
                f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers"
            )
            urls = response["urls"]
        else:
            response = await get(
                f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers"
            )
            urls = [worker["url"] for worker in response["workers"]]

        logger.info("Abort request for %s", urls)
        abort_tasks = [
            post(f"{url}/abort_request", {"abort_all": True}) for url in urls
        ]
        abort_results = await asyncio.gather(*abort_tasks, return_exceptions=True)
        for url, result in zip(urls, abort_results, strict=False):
            if isinstance(result, Exception):
                logger.warning("Failed to abort worker at %s: %s", url, result)
    except Exception:
        logger.warning("Failed to abort SGLang workers", exc_info=True)

    # Wait for all pending tasks to finish. Bounded by stall timeout so a
    # hung pending task (e.g. env stuck in a sync call that ignores cancel)
    # can't keep abort spinning forever.
    while state.pendings:
        done, state.pendings = await asyncio.wait(
            state.pendings,
            return_when=asyncio.FIRST_COMPLETED,
            timeout=_ROLLOUT_STALL_TIMEOUT_S,
        )
        if not done:
            logger.warning(
                "abort drain stalled after %ss; cancelling %d hung pending(s) and giving up",
                _ROLLOUT_STALL_TIMEOUT_S, len(state.pendings),
            )
            for t in state.pendings:
                t.cancel()
            state.pendings = set()
            break

    return aborted_samples

async def _safe_close_env(env):
    """Close an env (with 60s timeout) and unregister it from
    ``_active_envs``. Unregister always runs — even on close failure —
    so the surrounding :func:`_abort` snapshot loop terminates instead
    of re-closing the same env on the next pass. ``_unregister_env``
    uses ``set.discard``, so a later ``generate()`` finally that also
    unregisters is a safe no-op."""
    try:
        await asyncio.wait_for(env.close(), timeout=60.0)
    except TimeoutError:
        logger.warning("env.close() timed out after 60s: %s", env)
    except Exception:
        logger.warning("Failed to close env during abort: %s", env, exc_info=True)
    finally:
        _unregister_env(env)

# ---------------------------------------------------------------------------
# Custom _generate_rollout_async
# ---------------------------------------------------------------------------

async def _generate_rollout_async(
    args: Namespace,
    rollout_id: int,
    data_source: Callable[[int], list[list[Sample]]],
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    """Async rollout loop with cooperative local abort handling."""
    if not args.rollout_global_dataset:
        raise ValueError("online rollout requires rollout_global_dataset=True")

    # New rollout iteration → drop cached per-group seeds so each group in
    # this iteration draws a fresh seed on first encounter (and next
    # iteration redraws again). No-op when group_shared_seed is off.
    _group_seeds.clear()
    global _current_rollout_id
    _current_rollout_id = rollout_id

    # Env-server lifecycle and admission are server-owned; rollout workers
    # do not run any background backend refresh.

    state = GenerateState(args)

    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path)
        if args.dynamic_sampling_filter_path is not None
        else None
    )

    metric_gatherer = MetricGatherer()
    target_data_size = args.rollout_batch_size

    data = []
    all_data = []
    do_print = True
    pbar = tqdm(
        total=target_data_size * args.n_samples_per_prompt,
        desc="Rollout generation",
    )

    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            samples = data_source(args.over_sampling_batch_size)
            state.submit_generate_tasks(samples)

        done, state.pendings = await asyncio.wait(
            state.pendings,
            return_when=asyncio.FIRST_COMPLETED,
            timeout=_ROLLOUT_STALL_TIMEOUT_S,
        )
        if not done:
            # No task progressed within _ROLLOUT_STALL_TIMEOUT_S — typically a
            # single hung env container blocking everyone in a synchronous
            # call (e.g. gimp eval, container API server never coming up).
            # Cancel pending tasks and raise so the surrounding retry loop
            # can resume from the latest checkpoint.
            for t in state.pendings:
                t.cancel()
            raise RuntimeError(
                f"Rollout stall: no task completed in "
                f"{_ROLLOUT_STALL_TIMEOUT_S}s; cancelled "
                f"{len(state.pendings)} pending task(s). "
                "Increase ROLLOUT_STALL_TIMEOUT_S if legitimate episodes "
                "exceed this; otherwise the surrounding retry loop will "
                "resume from the latest checkpoint."
            )
        for task in done:
            group = task.result()

            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                logger.info(
                    "First rollout sample: prompt=%s, response=%s, reward=%s",
                    str(sample.prompt)[:100],
                    sample.response[:100] if sample.response else "",
                    sample.reward,
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            all_data.append(group)

            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(
                    reason=dynamic_filter_output.reason
                )
                state.remaining_batch_size -= 1
                continue

            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

    pbar.close()

    # Use local abort with env cleanup
    aborted_samples = await _abort(args, rollout_id)

    assert len(data) == args.rollout_batch_size, (
        f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    )

    # Sort by sample index for deterministic ordering
    def _sort_key(group):
        first = group[0][0] if isinstance(group[0], list) else group[0]
        return first.index

    data = sorted(data, key=_sort_key)
    all_data = sorted(all_data, key=_sort_key)

    state.reset()

    if args.rollout_sample_filter_path is not None:
        filter_func = load_function(args.rollout_sample_filter_path)
        filter_func(args, data)

    if args.rollout_all_samples_process_path is not None:
        process_func = load_function(args.rollout_all_samples_process_path)
        process_func(args, all_data, data_source)

    # Drop empty-token turn samples (env crashes / timeouts within a turn)
    # in place, preserving the nested structure. Reason: slime v0.3.0 sees the
    # rollout output twice — first the raw nested ``data`` (validation /
    # sample accounting in ``slime/ray/rollout.py``), then the rows our
    # ``convert_samples_to_train_data`` emits (which ``build_dp_schedule``
    # schedules by group). ``_filter_errored_rollouts`` inside convert drops
    # these same empty turns, so without this pre-filter slime's sample count
    # and the convert output diverge by however many turns failed this
    # rollout. Filtering here keeps both views identical.
    #
    # Bookkeeping (corrected): the drop erases all-empty trajectories from the
    # flat list, so ``_regroup_trajectories`` can't reconstruct them and
    # ``_filter_errored_rollouts`` reports 0. Left alone they'd all be mislabeled
    # ``n_missing`` (framework drop) instead of ``n_errored`` (env/judge failure
    # that DID produce a Sample). So count them HERE, pre-drop, and stash the
    # count on ``_last_rollout_n_errored``; ``bucket_trajectories`` restores the
    # correct split. ``n_valid`` is identical either way — only the failure
    # sub-split is affected.
    def _drop_empty_turns(grp):
        if grp and isinstance(grp[0], list):
            return [[s for s in roll if s.tokens] for roll in grp]
        return [s for s in grp if s.tokens]

    # Count entirely-empty trajectories BEFORE the drop below erases them, so
    # bucket_trajectories can label them n_errored (env/judge failure) not
    # n_missing (framework drop). See _count_empty_trajs / bucket_trajectories.
    global _last_rollout_n_errored
    _last_rollout_n_errored = sum(_count_empty_trajs(g) for g in data)

    n_before = sum(len(roll) if isinstance(roll, list) else 1
                   for g in data for roll in (g if g and isinstance(g[0], list) else [g]))
    data = [_drop_empty_turns(g) for g in data]
    n_after = sum(len(roll) if isinstance(roll, list) else 1
                  for g in data for roll in (g if g and isinstance(g[0], list) else [g]))
    if n_before != n_after:
        logger.info(
            "Pre-return filter: dropped %d empty-token turn samples (%d -> %d) "
            "to keep slime's dyn_gbs view in sync with convert output",
            n_before - n_after, n_before, n_after,
        )

    # All-failed guard: a 100%-failed batch (env-server down / all containers crash) collapses
    # every trajectory to ``[]`` above, so ``data`` becomes all-empty. slime's unguarded
    # flatten (``_get_rollout_data``: ``while isinstance(data[0], list): chain.from_iterable``)
    # would then reduce to ``[]`` and IndexError on ``data[0]`` — BEFORE convert's all-errored
    # fallback ever runs. Inject one synthetic dummy group so the flatten sees a non-empty leaf
    # and the step becomes a clean zero-gradient no-op (must be fixed here, not by guarding
    # slime's flatten — that just relocates the crash to convert→{}).
    if not any(roll for g in data for roll in g):
        logger.warning(
            "All trajectories empty (full-batch failure) — injecting 1 dummy group so "
            "slime produces a clean zero-gradient no-op step instead of crashing."
        )
        data = [[[synthetic_dummy_sample()]]]

    return (
        RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()),
        aborted_samples,
    )

# ---------------------------------------------------------------------------
# Custom eval rollout (works around slime bug with list[Sample] returns)
# ---------------------------------------------------------------------------

async def _eval_rollout(
    args: Namespace, rollout_id: int,
) -> tuple[RolloutFnEvalOutput, list]:
    """Custom eval rollout that handles multi-turn generate() returns.

    slime's eval_rollout_single_dataset has a bug: it does ``sample.prompt``
    on the result of generate_and_rm(), but our generate() returns
    list[Sample] (not a single Sample). We fix this by flattening the result
    before logging.
    """
    import copy

    from slime.utils.data import Dataset
    from slime.utils.processing_utils import load_processor, load_tokenizer

    # Env-server lifecycle and admission are server-owned; eval workers
    # do not run any background backend refresh.

    results = {}
    # Tokenizer + processor depend only on args.hf_checkpoint (constant across
    # eval datasets) — load once, not once per dataset (the VLM processor is
    # multi-MB to construct).
    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
    # ``or []`` kept deliberately, on PARITY grounds only: this line is
    # byte-identical to ``slime/slime/rollout/sglang_rollout.py`` (``eval_rollout``),
    # the function this one forks, and that tree is off-limits to us — so the
    # twin has to stay diffable line-for-line. Unlike the ``rl_sample.steps``
    # reads above (``LiteRLSample.steps`` is ours and is a
    # ``field(default_factory=list)``), ``args`` is a slime-owned ``Namespace``
    # this file cannot pin: whether ``eval_datasets`` is absent, ``None``, or a
    # list is decided by ``slime.utils.arguments.slime_validate_args``.
    for dataset_cfg in getattr(args, "eval_datasets", []) or []:
        dataset = Dataset(
            path=dataset_cfg.path,
            tokenizer=tokenizer,
            processor=processor,
            max_length=getattr(args, "eval_max_prompt_len", None),
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            multimodal_keys=args.multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        )

        base_sampling_params = dict(
            temperature=dataset_cfg.temperature,
            top_p=dataset_cfg.top_p,
            top_k=dataset_cfg.top_k,
            max_new_tokens=dataset_cfg.max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )

        # Build index → original prompt_sample mapping for retries.
        prompt_samples_by_index: dict[int, Any] = {}
        tasks = []
        sample_index = 0
        for prompt_sample in dataset.samples:
            for j in range(dataset_cfg.n_samples_per_eval_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.index = sample_index
                sample.metadata = dataset_cfg.inject_metadata(
                    getattr(sample, "metadata", None)
                )
                prompt_samples_by_index[sample_index] = prompt_sample
                sample_index += 1
                tasks.append(
                    asyncio.create_task(
                        generate_and_rm(
                            args, sample,
                            sampling_params=base_sampling_params.copy(),
                            evaluation=True,
                        )
                    )
                )

        n_trajs_expected = len(tasks)

        # Collect results with inline retry: when a task produces no
        # token-carrying turn at all (env crash, abort, or a trajectory that
        # segmented to nothing), immediately re-queue it at the back of the
        # pending set instead of waiting for a separate retry round. This avoids
        # wasting an extra batch of concurrency slots. A model-side parse failure
        # is NOT errored — it carries tokens and is scored at whatever the env
        # awarded, see below.
        max_eval_retries = getattr(args, "max_eval_retries", 3)
        retry_counts: dict[int, int] = {}   # index → retries so far
        data = []
        remaining: set[asyncio.Task] = set(tasks)
        pbar = tqdm(total=n_trajs_expected, desc=f"Eval {dataset_cfg.name}")
        while remaining:
            done, remaining = await asyncio.wait(
                remaining,
                return_when=asyncio.FIRST_COMPLETED,
                timeout=_ROLLOUT_STALL_TIMEOUT_S,
            )
            if not done:
                # Same stall guard as _generate_rollout_async — a single hung
                # eval task can otherwise wedge the entire eval phase.
                for t in remaining:
                    t.cancel()
                raise RuntimeError(
                    f"Eval stall: no task completed in "
                    f"{_ROLLOUT_STALL_TIMEOUT_S}s; cancelled "
                    f"{len(remaining)} pending eval task(s)."
                )
            for fut in done:
                try:
                    result = fut.result()
                except Exception:
                    logger.exception("Eval task raised unexpectedly")
                    pbar.update(1)
                    continue
                samples = result if isinstance(result, list) else [result]
                if not samples:
                    pbar.update(1)
                    continue
                idx = samples[0].index
                valid_turns = [s for s in samples if s.tokens]
                # Only re-queue errored trajectories whose error is retryable.
                # is_retryable was evaluated at catch time (the exception is gone
                # here) and cached on the empty sample's metadata; a non-retryable
                # error (e.g. a bot-blocked site) re-runs identically, so retrying
                # only burns rollouts. The True default is still reachable: slime's
                # abort short-circuit returns the original prompt Sample without
                # entering generate(), so it carries no cached verdict.
                retryable = samples[0].metadata.get("lite_error_retryable", True)
                if (not valid_turns and retryable
                        and retry_counts.get(idx, 0) < max_eval_retries):
                    retry_counts[idx] = retry_counts.get(idx, 0) + 1
                    logger.info(
                        "Eval %s: index %d errored, retry %d/%d",
                        dataset_cfg.name, idx,
                        retry_counts[idx], max_eval_retries,
                    )
                    sample = copy.deepcopy(prompt_samples_by_index[idx])
                    sample.index = idx
                    sample.metadata = dataset_cfg.inject_metadata(
                        getattr(sample, "metadata", None)
                    )
                    remaining.add(
                        asyncio.create_task(
                            generate_and_rm(
                                args, sample,
                                sampling_params=base_sampling_params.copy(),
                                evaluation=True,
                            )
                        )
                    )
                else:
                    data.extend(samples)
                    pbar.update(1)
        pbar.close()
        n_trajs_retried = len(retry_counts)

        # Per-trajectory episode return + truncated flag, keyed by ``s.index``
        # so the emitted ``rewards`` list stays contiguous-by-prompt (slime's
        # ``compute_pass_rate`` reshapes ``rewards`` to
        # ``(num_groups, n_samples_per_eval_prompt)`` assuming the allocation
        # order from ``sample_index += 1`` above). Failed rollouts get 0.0 /
        # False at their own index position. ``_regroup_trajectories`` returns
        # ``as_completed`` order, so we re-sort by ``s.index`` to restore
        # contiguous-by-prompt ordering.
        rollouts_per_traj = _regroup_trajectories(data)
        rollouts_per_traj.sort(key=lambda r: r[0].index if r else 0)

        # 4 mutually-exclusive trajectory buckets (symmetric with the train
        # rollout/* metrics). ``n_trajs_missing`` = tasks slime launched that
        # never returned any Sample at all: Ray actor death, controller cancel,
        # or cancellation outside ``generate()``'s normal Exception path.
        n_trajs_missing = max(0, n_trajs_expected - len(rollouts_per_traj))

        # FAILED-status turns are KEPT here, matching the train path. Under
        # slime's generic ``Sample.Status`` a FAILED turn may be partial output.
        # Among the TOKEN-CARRYING samples that reach this loop the only producer
        # is the parse-failure final (``agent/utils/final.py``) — the model could
        # not drive the tool surface, which is a task failure, not an env error.
        # (``_empty_sample`` / ``dummy_sample`` also set FAILED but carry no
        # tokens, so ``s.tokens`` already excludes them.) Excluding parse failures
        # would drop them from the eval denominator and bias every reported return
        # upward. Caveat: ``examples/grounding``'s adapter also writes FAILED and
        # fabricates ``reward=0.0`` without stepping the env, so for that producer
        # the 0 is structural, not measured.
        episode_returns: list[float] = []
        truncated_per_rollout: list[bool] = []
        # Prompt id per rollout: ``index // n`` recovers the prompt because eval
        # allocates n_samples_per_eval_prompt contiguous indices per prompt (loop
        # above). Robust to dropped samples, unlike a positional reshape. Used for
        # the clustered env-noise std/stderr below.
        n_eval = max(dataset_cfg.n_samples_per_eval_prompt, 1)
        prompt_ids: list[int] = []
        is_valid: list[bool] = []   # parallel to episode_returns: did this rollout score?
        n_trajs_errored = 0
        for r in rollouts_per_traj:
            # ``if r else -1`` mirrors the empty-group guard in the sort above; a
            # (degenerate) empty rollout lands in a sentinel bucket, never IndexErrors.
            prompt_ids.append((r[0].index // n_eval) if r else -1)
            valid_turns = [s for s in r if s.tokens]
            if not valid_turns:
                # errored: no token-carrying turn at all (env crash or abort).
                # Keep a dense 0.0 at this index — slime's compute_pass_rate reshapes
                # the returned rewards list positionally — but mark it invalid so OUR
                # stats exclude it.
                episode_returns.append(0.0)
                truncated_per_rollout.append(False)
                is_valid.append(False)
                n_trajs_errored += 1
                continue
            er = (valid_turns[-1].metadata.get("others") or {}).get("episode_return")
            episode_returns.append(float(er) if er is not None else 0.0)
            truncated_per_rollout.append(
                any(s.status == Sample.Status.TRUNCATED for s in valid_turns)
            )
            is_valid.append(True)

        reward_key = args.eval_reward_key or args.reward_key
        if reward_key:
            episode_returns = [r[reward_key] if isinstance(r, dict) else r for r in episode_returns]

        # ``episode_returns`` stays DENSE (one entry per rollout, 0.0 for errored) for
        # slime's positional ``compute_pass_rate`` reshape. OUR metrics are computed
        # over the VALID-only view so env failures never dilute them — UNLIKE slime's
        # bare ``eval/{ds}`` (sum/len over the dense list), which stays diluted and is
        # NOT what return_mean equals anymore.
        n_trajs_valid = n_trajs_expected - n_trajs_missing - n_trajs_errored
        valid_returns = [er for er, ok in zip(episode_returns, is_valid) if ok]
        valid_prompt_ids = [pid for pid, ok in zip(prompt_ids, is_valid) if ok]

        # Aligned eval-mean trio (return_mean / return_std / return_stderr), all
        # VALID-only. return_std / return_stderr measure ENVIRONMENT-REPLICATION
        # noise: how much the SAME task's reward varies when re-run (the env is
        # stochastic even at greedy temp=0). Computed WITHIN task (variance over a
        # prompt's n samples), then pooled — NOT across tasks (that would conflate
        # task-difficulty spread with run-to-run noise). With n_samples_per_eval_prompt
        # =1 there is no replication → NaN (genuinely no error to measure, not 0; bump
        # N_SAMPLES_PER_EVAL_PROMPT≥2 to get a real bar).
        #   return_std    = sqrt(mean_p Var_i(r_{p,i}))   # typical per-task env std
        #   return_stderr = return_std / sqrt(N_valid)
        return_mean = sum(valid_returns) / max(n_trajs_valid, 1)
        by_prompt: dict[int, list[float]] = {}
        for pid, er in zip(valid_prompt_ids, valid_returns):
            by_prompt.setdefault(pid, []).append(er)
        within_vars = [statistics.variance(v) for v in by_prompt.values() if len(v) >= 2]
        if n_eval >= 2 and within_vars and valid_returns:
            return_std = (sum(within_vars) / len(within_vars)) ** 0.5
            return_stderr = return_std / (len(valid_returns) ** 0.5)
        else:
            return_std = return_stderr = float("nan")

        # nonzero_return_rate: valid-only by construction (errored are dense 0.0s,
        # never > 0; denominator is n_trajs_valid).
        # `r > 0` mirrors the threshold used in convert_samples_to_train_data
        # (grpo.py). Binary 0/1 rewards → equivalent to `r >= 1.0`;
        # continuous rewards → counts any positive return (NOT a true success
        # rate on partial-credit envs — hence ``nonzero_return_rate``).
        nonzero_return_rate = sum(1 for r in episode_returns if r > 0) / max(n_trajs_valid, 1)
        n_truncated = sum(truncated_per_rollout)

        # return_mean is the number to read: slime also logs a bare
        # ``eval/{ds}`` (rollout.py), but that one averages over the DENSE list,
        # so every errored rollout dilutes it as a 0.0. Printing the valid-only
        # mean next to the valid/errored counts means a diluted comparison is
        # visible in the log itself, not only in wandb.
        logger.info(
            "Eval %s: return_mean=%.4f (std=%.4f stderr=%.4f), %d valid / %d errored"
            " / %d missing / %d expected (retried %d), nonzero_return_rate=%.2f",
            dataset_cfg.name, return_mean, return_std, return_stderr,
            n_trajs_valid, n_trajs_errored, n_trajs_missing,
            n_trajs_expected, n_trajs_retried, nonzero_return_rate,
        )

        # Symmetric with train rollout/* metrics. Avoid emitting
        # ``truncated_ratio`` — slime already emits a per-turn one from
        # compute_metrics_from_samples (slime/ray/rollout.py:1210) under the
        # same key, which would overwrite ours. Use n_truncated (count)
        # instead; slime also emits per-rollout ``eval/{ds}-truncated_ratio``
        # (with a dash) from the ``truncated`` list we return below.
        prefix = f"eval/{dataset_cfg.name}"
        safe_wandb_log({
            f"{prefix}/return_mean": return_mean,
            f"{prefix}/return_std": return_std,
            f"{prefix}/return_stderr": return_stderr,
            f"{prefix}/nonzero_return_rate": nonzero_return_rate,
            f"{prefix}/n_trajs_valid": n_trajs_valid,
            f"{prefix}/n_trajs_errored": n_trajs_errored,
            f"{prefix}/n_trajs_missing": n_trajs_missing,
            f"{prefix}/n_trajs_expected": n_trajs_expected,
            f"{prefix}/n_trajs_retried": n_trajs_retried,
            f"{prefix}/n_truncated": n_truncated,
        })

        # ``samples`` flattened from the sorted ``rollouts_per_traj`` — gives
        # trajectories in ``s.index`` order (outer) with segments in
        # ``turn_range[0]`` order (inner, from ``_regroup_trajectories``). Any
        # custom eval logger (or ``custom_eval_rollout_log_function_path``)
        # that assumes index-sorted samples keeps working. Slime's own default
        # metrics are order-invariant.
        eval_samples = [s for r in rollouts_per_traj for s in r]
        # Strip the lazy multimodal payload on eval-side Samples: the expand
        # hook only fires from training's ``get_batch``; eval inference goes
        # through SGLang and never reads the field, so leaving it hanging
        # only pollutes debug dumps and inflates pickling cost on eval logs.
        for s in eval_samples:
            s.multimodal_lazy_payloads = None
        results[dataset_cfg.name] = {
            "rewards": episode_returns,
            "truncated": truncated_per_rollout,
            "samples": eval_samples,
        }

    return RolloutFnEvalOutput(data=results), []

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """Entry point. Uses custom eval (slime's eval_rollout_single_dataset has a
    bug: it does ``sample.prompt`` on the result of generate_and_rm, but our
    generate() returns list[Sample], not Sample)."""
    if not args.rollout_global_dataset:
        raise ValueError("online rollout requires rollout_global_dataset=True")
    if evaluation:
        output, _ = run(_eval_rollout(args, rollout_id))
        return output

    output, aborted_samples = run(
        _generate_rollout_async(args, rollout_id, data_source.get_samples)
    )
    data_source.add_samples(aborted_samples)
    return output

# ---------------------------------------------------------------------------
# Shared helpers for convert_samples_to_train_data
# ---------------------------------------------------------------------------

def _regroup_trajectories(samples) -> list[list[Sample]]:
    """Reconstruct per-trajectory grouping from flattened Sample list.

    Slime flattens our nested list[list[list[Sample]]] into list[Sample]
    before calling convert_samples_to_train_data. We re-group by
    (group_index, index) to recover the trajectory structure.
    """
    if isinstance(samples[0], Sample):
        rollout_map: dict[tuple, list[Sample]] = {}
        for s in samples:
            rollout_map.setdefault((s.group_index, s.index), []).append(s)
        # Sort by ``turn_range[0]`` (segment's first turn index, set by
        # ``_emit_rl_sample``). Falls back to 0 for any sample without
        # turn_range — order between such samples is undefined.
        return [
            sorted(turns, key=lambda s: s.metadata.get("turn_range", (0, 0))[0])
            for turns in rollout_map.values()
        ]
    return list(samples)


def _filter_errored_rollouts(
    rollouts: list[list[Sample]],
) -> tuple[list[list[Sample]], int]:
    """Drop rollouts where ALL turns have empty tokens (env crashes, timeouts).

    Within valid rollouts, individual errored turns (empty tokens) are also
    removed — they carry no training signal. If every rollout failed, returns
    N dummies (loss_mask=[0]) so slime's pipeline still gets a batch for a
    no-op gradient step.

    Train path only. Eval applies the same ``s.tokens`` predicate but has
    index-ordering and dense-reward requirements that make a shared helper
    awkward — see the inline loop in ``_eval_rollout``.

    Returns (valid_rollouts, n_errored). ``n_errored`` is always the
    true count of dropped rollouts (not diluted by dummy injection).
    """
    valid: list[list[Sample]] = []
    for rollout in rollouts:
        valid_turns = [s for s in rollout if s.tokens]
        if valid_turns:
            valid.append(valid_turns)

    # Compute n_errored BEFORE potential dummy injection — otherwise the
    # "all failed" case would report 0 errored (dummies bring len(valid)
    # back up to len(rollouts)), hiding the failure signal from metrics.
    n_errored = len(rollouts) - len(valid)

    if not valid:
        logger.warning("All %d rollouts errored — using dummy samples", len(rollouts))
        # Each dummy inherits its ORIGINAL rollout's template (not a single
        # shared template) so ``group_index`` diversity is preserved —
        # otherwise GRPO ``group_map`` collapses N dummies into 1 fake group,
        # and ``rollout/n_groups`` / ``rollout/mixed_group_ratio`` metrics lie.
        # Gradient is still zero (loss_mask=[0]); this only affects metrics.
        valid = [[dummy_sample(r[0])] for r in rollouts]

    if n_errored > 0:
        logger.info("Dropped %d/%d errored rollouts", n_errored, len(rollouts))

    return valid, n_errored


def get_episode_returns(rollouts: list[list[Sample]]) -> list[float]:
    """Extract ``metadata.others.episode_return`` from each rollout."""
    returns: list[float] = []
    for rollout in rollouts:
        episode_return = (
            (rollout[-1].metadata.get("others") or {}).get("episode_return")
            if rollout
            else None
        )
        returns.append(float(episode_return) if episode_return is not None else 0.0)
    return returns


def bucket_trajectories(
    samples, n_expected: int,
) -> tuple[list[list[Sample]], int, int, int, int]:
    """Regroup + filter + classify trajectories into 4 mutually-exclusive buckets.

    Composes ``_regroup_trajectories`` and ``_filter_errored_rollouts`` and
    emits the quota-accounting metrics that ``convert_samples_to_train_data``
    reports to wandb. The 4 buckets sum to ``n_expected``:

        n_valid      — usable for training (has non-empty tokens).
        n_errored — produced a Sample but no usable turn (all turns
                       tokens=[]) — an env/judge failure caught inside
                       generate()'s try/except → ``_empty_sample``. These are
                       stripped pre-regroup by ``_drop_empty_turns``, so the
                       count is recovered from ``_last_rollout_n_errored``
                       (captured before the drop) rather than from
                       ``_filter_errored_rollouts`` (which now sees ~0).
        n_missing    — produced NO Sample at all — a framework-level drop:
                       Ray actor died, slime controller cancel,
                       asyncio.CancelledError, or an exception thrown above
                       generate()'s outer try/except.
        n_expected   — launch quota. Caller supplies it; train path uses
                       ``rollout_batch_size × n_samples_per_prompt``.

    Train path only — ``_filter_errored_rollouts``'s dummy-padding behavior
    is wrong for eval (see its docstring).

    Returns: (rollouts, n_valid, n_errored, n_missing, n_expected)
        where ``rollouts`` is the post-filter list. A full-batch failure may
        contain the synthetic dummy group inserted before convert so slime can
        complete a zero-gradient step; use ``n_errored``/``n_missing`` for
        failure accounting rather than inferring it from ``len(rollouts)``.
    """
    rollouts = _regroup_trajectories(samples)
    n_missing = max(0, n_expected - len(rollouts))
    rollouts, n_errored = _filter_errored_rollouts(rollouts)
    # ``_drop_empty_turns`` (in _generate_rollout_async) stripped the all-empty
    # trajectories before regroup, so they fell into ``n_missing`` above and
    # ``_filter_errored_rollouts`` reports ~0. Move that pre-drop count back into
    # ``n_errored`` (an env/judge failure that DID produce a Sample) so
    # ``n_missing`` means only genuine framework-level drops. ``n_valid`` is
    # unchanged (= post-filter trajectory count).
    n_errored_predrop = min(n_missing, _last_rollout_n_errored)
    n_missing -= n_errored_predrop
    n_errored += n_errored_predrop
    n_valid = n_expected - n_missing - n_errored
    logger.info(
        "Rollouts: %d valid / %d errored / %d missing / %d expected",
        n_valid, n_errored, n_missing, n_expected,
    )
    return rollouts, n_valid, n_errored, n_missing, n_expected


def safe_wandb_log(metrics: dict) -> None:
    """Log to wandb if a run is active; silently no-op otherwise.

    Uses ``commit=False`` so callers don't race with slime's own step commit.
    """
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, commit=False)
    except Exception:
        pass


def flatten_and_align(rollouts: list[list[Sample]], args: Any) -> dict:
    """Pad trajectories to the launched count, flatten, assemble train_data.

    Steps:
      1. Group-level alignment: pad the trajectory list up to the launched
         count (``rollout_batch_size * n_samples_per_prompt``) with
         zero-gradient dummy groups, so slime's ``build_dp_schedule`` sees
         ``num_groups == global_batch_size * num_steps_per_rollout``.
      2. Flatten rollouts (groups of segments) into a single list of Samples.
         (Static-path bshd segment alignment — doc §6 M2 — is slime's job:
         ``pad_static_groups`` at the convert boundary in
         ``_split_train_data_by_dp`` pads every group with zero-loss dummy
         rows, covering native and custom converters alike.)
      3. (no length reconciliation — see the note at the assembly site)
      4. Assemble train_data, including per-trajectory ``group_ids`` and
         ``group_mask_sums`` for slime's per-group loss reducer.
    """
    # (1) Group-level alignment. v0.3.0 schedules micro-batches by GROUP:
    # ``build_dp_schedule`` groups samples by ``group_id`` and floors
    # ``num_steps = num_groups // global_batch_size``, dropping any trailing
    # groups (and asserting ``num_groups >= global_batch_size``). Missing /
    # all-errored trajectories were dropped upstream (regroup + filter), so
    # ``len(rollouts)`` can be below the launched count. Refill the gap with
    # zero-gradient dummy groups (``loss_mask=[0]``) carrying fresh,
    # collision-free ``group_id``s so each occupies its own group and
    # ``num_groups`` lands exactly on the launched count. This also makes
    # ``raw_reward`` span the full ``[rollout_batch_size, n_samples_per_prompt]``
    # grid that slime's pass-rate logging reshapes against.
    n_expected = args.rollout_batch_size * args.n_samples_per_prompt
    if 0 < len(rollouts) < n_expected:
        base = max((s.index for r in rollouts for s in r if s.index is not None), default=-1) + 1
        template = rollouts[0][0]
        for k in range(n_expected - len(rollouts)):
            dummy = dummy_sample(template)
            dummy.index = base + k
            dummy.group_id = base + k
            # ``dummy_sample`` copies the template's metadata verbatim, including
            # its ``metadata.others.episode_return``. The template is a SURVIVING
            # trajectory, so this could be a non-zero reward. Force the dummy's
            # return to 0.0 so ``raw_reward`` below stays honest.
            dummy.metadata = dict(dummy.metadata or {})
            dummy.metadata["others"] = {
                **(dummy.metadata.get("others") or {}),
                "episode_return": 0.0,
            }
            rollouts.append([dummy])

    # Static-path (bshd) segment alignment is handled downstream by slime's
    # ``pad_static_groups`` after this custom converter returns. Keep the
    # per-rollout reward/group metadata stable here; do not pad trajectories in
    # this function.
    # Per-rollout raw_reward (NOT per-turn). Slime's ``log_passrate``
    # (megatron_utils/data.py) reshapes this to
    # ``[rollout_batch_size, n_samples_per_prompt]`` — one entry per rollout.
    # All turns of the same rollout share ``metadata.others.episode_return``,
    # so we take one value per rollout; padded dummies contribute 0.0.
    raw_reward_per_rollout = [
        (rollout[0].metadata.get("others") or {}).get("episode_return", 0.0)
        for rollout in rollouts
    ]

    flat: list[Sample] = [s for rollout in rollouts for s in rollout]

    # No reconciliation of response_length / rollout_log_probs / loss_mask: every
    # ``Sample`` constructor in ``lite/train`` allocates all three from ONE
    # response_length local, and nothing between them and here touches a length.

    # Per-trajectory aggregation id (slime falls back to ``index`` when unset).
    group_ids = [s.group_id if s.group_id is not None else s.index for s in flat]

    # Assemble train_data dict
    train_data: dict = {
        "tokens": [s.tokens for s in flat],
        "response_lengths": [s.response_length for s in flat],
        "rewards": [s.reward for s in flat],
        "raw_reward": raw_reward_per_rollout,
        "truncated": [1 if s.status == Sample.Status.TRUNCATED else 0 for s in flat],
        "sample_indices": [s.index for s in flat],
        "loss_masks": [s.loss_mask for s in flat],
        "group_ids": group_ids,
    }

    # Per-trajectory total mask, broadcast per-segment, so slime's loss reducer
    # divides each segment's contribution by its trajectory's total mask -> one
    # token-weighted mean per trajectory (exactly packing-invariant and
    # length-independent). This is the slime-native replacement for the old
    # ``adv / n_turns`` hand-normalization; mirrors slime's own
    # ``_convert_samples_to_train_data`` (slime/ray/rollout.py:713-719) which our
    # custom converter bypasses. PAIRING INVARIANT: emit this only together with
    # ``reward = raw advantage`` in convert_samples_to_train_data.
    mask_sums = [sum(m) for m in train_data["loss_masks"]]
    group_total_mask: dict[int, int] = {}
    for gid, ms in zip(group_ids, mask_sums, strict=True):
        group_total_mask[gid] = group_total_mask.get(gid, 0) + ms

    # CUA_LITE_NORM_BY_TURNS=1 switches the per-trajectory denominator from
    # "total masked tokens" to "turn count, scaled by the batch's mean tokens per
    # turn". Ablation knob for the multi-turn setting — default is unchanged.
    #
    # Why it matters here: with the token denominator, a trajectory that writes
    # FEWER tokens per turn raises its own per-token gradient (A / N), so the
    # objective rewards compressing each turn's output. Measured on 8-task
    # train=eval: response length fell 80% while turn count rose 56% with no
    # accuracy gain. Scaling by the batch mean keeps ``sum(denoms)`` equal to
    # ``sum(mask_sums)``, so only the RELATIVE weighting changes, not the
    # gradient scale — otherwise this would confound normalization with a ~300x
    # effective LR change.
    #
    # Still packing-invariant: ``_segment_steps`` packs WHOLE turns (its unit is
    # the turn index), so a turn never spans segments and the per-trajectory turn
    # count is independent of how segments are packed.
    if os.environ.get("CUA_LITE_NORM_BY_TURNS") == "1":
        group_turns: dict[int, int] = {}
        for gid, s in zip(group_ids, flat, strict=True):
            n_turns = ((s.metadata or {}).get("n_turns")) or 1
            group_turns[gid] = max(group_turns.get(gid, 0), int(n_turns))
        total_tokens = sum(group_total_mask.values())
        total_turns = sum(group_turns.values()) or 1
        mean_tokens_per_turn = total_tokens / total_turns
        train_data["group_mask_sums"] = [
            max(group_turns[gid] * mean_tokens_per_turn, 1.0) for gid in group_ids
        ]
    else:
        train_data["group_mask_sums"] = [group_total_mask[gid] for gid in group_ids]

    if flat[0].rollout_log_probs is not None:
        train_data["rollout_log_probs"] = [s.rollout_log_probs for s in flat]

    if any(s.multimodal_train_inputs is not None for s in flat):
        train_data["multimodal_train_inputs"] = [s.multimodal_train_inputs for s in flat]

    # Lazy multimodal payloads. The trainer-side expand hook materializes
    # these into multimodal_train_inputs once per RL iter, inside
    # actor._get_rollout_data via slime.utils.data.materialize_lazy_payloads.
    # Mutually exclusive with multimodal_train_inputs at the per-Sample
    # level (enforced in materialize_lazy_payloads).
    if any(s.multimodal_lazy_payloads is not None for s in flat):
        train_data["multimodal_lazy_payloads"] = [s.multimodal_lazy_payloads for s in flat]

    return train_data
