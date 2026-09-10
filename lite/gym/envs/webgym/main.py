"""WebGym — CUA-Lite gym wrapper for WebGym OmniBoxes HTTP API.

Wraps WebGym's distributed Playwright-based browser automation platform as a
CUA-Lite gym environment using coordinate-based GUI actions ([0, 1000]
normalized coordinates) plus standalone terminal ``response``/``terminate`` tools.
Browser nav (goto, back, ...) is opt-in via ``env_kwargs.extra_tools`` (canonical
names; schemas resolved by ``LiteBrowserNavToolSet.get_tool_schemas(include=)``), executed by
``step`` per canonical name.

This wrapper is a thin HTTP client (``WebGymClient``) over WebGym's OmniBoxes
master. The backend is container-only: the whole OmniBoxes pool (redis + master
+ node + M instance-server Chromiums) runs inside ONE ``cua-lite/webgym`` container
per env-server, brought up automatically by ``WebGymContainerServices.ensure``,
which publishes the master port into ``WEBGYM_MASTER_URL``. See
lite/gym/envs/webgym/README.md.

Prerequisites:
  - uv run --no-sync bash lite/gym/envs/webgym/scripts/install.sh   # build cua-lite/webgym:latest
  - docker available on the host

Usage:
    # The container is brought up on demand by gym.make / the env-server.
    uv run python -c "
    import asyncio, lite.gym as gym
    async def main():
        print(gym.registry.task_ids('webgym'))
        env = gym.make('webgym@0', max_steps=10)
        obs = await env.reset()
        print(f'Instruction: {obs.text}')
        await env.close()
    asyncio.run(main())
    "
"""

from __future__ import annotations

import asyncio
import dataclasses
import io
import logging
import os
import time
import warnings
from pathlib import Path
from typing import Any, ClassVar

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.action_space import LITE_DESKTOP_KEY_ACTION_NAMES
from lite.core.tools.action_space.geometry import strict_norm_to_pixel
from lite.core.tools.calls import (
    EnvAction,
    RuntimeEnvAction,
    runtime_internal_stop_reason,
    runtime_rejected_reason,
)
from lite.core.tools.extra_tools import LiteBrowserNavToolSet, LiteFinishToolSet
from lite.core.tools.schemas import BaseTools, tool_schema_name
from lite.gym.base import LiteBaseEnv
from lite.gym.errors import (
    CapacityExhausted,
    EnvBlocked,
    EnvDepsMissingError,
    TrueInfraFailure,
)
from lite.gym.registry import invalidate_services, register, registry
from lite.gym.remote.reaper import SingletonContainerServices
from lite.gym.services import register_services
from lite.gym.types import (
    EXECUTED_ACTIONS_INFO_KEY,
    LiteEnvObservation,
    LiteEnvStepResult,
    LiteExecutedAction,
)
from lite.gym.utils import config as env_config
from lite.gym.utils.backend.model_inputs import (
    coerce_model_duration,
    project_model_keys,
)
from lite.gym.utils.feedback.errors import (
    MODEL_ACTION_ERROR_TYPES,
    ModelVisibleErrorDetail,
    ToolErrorFeedback,
    append_feedback,
    current_feedback,
    error_only_feedback,
    record_model_action_error,
    record_tool_execution_error,
    unavailable_action_message,
    unknown_tool_message,
    unsupported_action_message,
)
from lite.gym.utils.feedback.ingress import (
    classify_standalone_tool_call,
    invalid_action_message,
    is_lite_action_name_or_action_batch_tool_name,
    prepare_env_tool_calls,
)
from lite.gym.utils.feedback.results import (
    build_tool_results_from_decisions,
    ordered_tool_call_ids,
)
from lite.gym.utils.feedback.surface import (
    copy_valid_actions,
    resolve_extra_tools,
    resolve_valid_actions,
)
from lite.utils.config import deep_merge

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency check
# ---------------------------------------------------------------------------

try:
    import httpx
except ImportError:
    raise EnvDepsMissingError(
        what="httpx package not installed (required by webgym)",
        install="uv run --no-sync bash lite/gym/envs/webgym/scripts/install.sh",
        see="lite/gym/envs/webgym/README.md",
    )

# ---------------------------------------------------------------------------
# Blank screenshot detection (imported from webgym package)
# ---------------------------------------------------------------------------

try:
    from webgym.misc import is_white_image
except ImportError:
    def is_white_image(img: Any) -> bool:  # type: ignore[misc]
        """Fallback: treat all images as non-blank when webgym is unavailable."""
        return False

ENV_DIR = str(Path(__file__).parent)
CFG = env_config.load(ENV_DIR)


# ============================================================================
# Config defaults — every value below is read once from configs/default.yaml
# via env_config.load(ENV_DIR). Swap the whole file at startup with
# WEBGYM_CONFIG=<abs-path | bundled-name>. A rollout's env_kwargs still
# override per run; these are only registration defaults.
# ============================================================================
# --- env_kwargs (per-instance) ---
# max_steps slot: per-difficulty budget tables (train/eval) replace the scalar max_steps.
_MAX_STEPS_TRAIN = {int(k): v for k, v in CFG.env_kwargs["max_steps_train"].items()}
_MAX_STEPS_EVAL = {int(k): v for k, v in CFG.env_kwargs["max_steps_eval"].items()}
# step_timeout now lives in CFG.make_kwargs (env-wide make default), applied via
# registry.set_env_make_kwargs in _load_and_register() — no longer a per-task register arg.
_POST_ACTION_DELAY = CFG.env_kwargs["post_action_delay"]
_VIEWPORT = tuple(CFG.env_kwargs["viewport"])
_GOBACK_SKIP_THRESHOLD = CFG.env_kwargs["goback_skip_threshold"]
_GOBACK_TERMINATE_THRESHOLD = CFG.env_kwargs["goback_terminate_threshold"]
_EVAL_CONFIG = CFG.env_kwargs["eval_config"]
# Markers the webgym Evaluator embeds in a per-criterion response when the OpenAI
# CALL for that criterion FAILS (e.g. a 429 rate-limit): it CATCHES the exception,
# appends "[Criterion X ...] Error: <exc>" to the responses, and scores that
# criterion 0 (evaluator.py:241-250 / 309-313). That silently turns an infra
# failure into a FALSE reward=0 — poisoning collected data. We detect the marker
# after the judge returns and RAISE, so the trajectory ERRs and is retried
# (--max-attempts) instead of committing a corrupted reward. See _judge_incomplete_reason.
_JUDGE_CALL_ERROR_MARKERS = ("] Error: ", "Error code:", "Too Many Requests", "RateLimitError")
# ⚠ Defines the GUI action enum. Terminal/nav tools live in the extra-tool
# set and are exposed only when env_kwargs.extra_tools selects them.


_VALID_ACTIONS = resolve_valid_actions(
    CFG.env_kwargs["valid_actions"], env_name="webgym", platform="browser",
)
_EXTRA_TOOLS = CFG.env_kwargs["extra_tools"]   # opt-in standalone tools
# --- server_kwargs (per-deployment) ---
_MASTER_PORT = CFG.server_kwargs["master_port"]
# "" in yaml → derive from the (auto-allocated) master port; a non-empty value pins a remote master.
_MASTER_URL = CFG.server_kwargs["master_url"] or f"http://localhost:{_MASTER_PORT}"
_API_KEY = CFG.server_kwargs["api_key"]
_HF_REPO_ID = CFG.server_kwargs["hf_repo_id"]
_INSTANCE_LIFETIME_MINS = CFG.server_kwargs["instance_lifetime_mins"]
_HTTP_TIMEOUT = CFG.server_kwargs["http_timeout"]
_MAX_RETRIES = CFG.server_kwargs["max_retries"]
_NAV_MAX_RETRIES = CFG.server_kwargs["nav_max_retries"]
_RETRY_BACKOFF_BASE = CFG.server_kwargs["retry_backoff_base"]
_SEM_ALLOCATE = CFG.server_kwargs["sem_allocate"]
_SEM_EXECUTE = CFG.server_kwargs["sem_execute"]
_SEM_SCREENSHOT = CFG.server_kwargs["sem_screenshot"]
_SEM_NAVIGATE = CFG.server_kwargs["sem_navigate"]
# VLM-judge concurrency cap. The terminal evaluate() runs the (synchronous)
# webgym Evaluator, which fires ~20+ OpenAI vision calls; running it off the event
# loop via asyncio.to_thread (so it can't freeze peers' browser steps) means many
# judges could otherwise fire at once and storm the Azure deployment into 429s
# (whose client-side retry-backoff then stacks a terminal step past step_timeout).
# This semaphore bounds concurrent judge runs across all envs on the server. Tune
# to the judge deployment's RPM/TPM: lower if 429s persist, raise if it has slack.
_SEM_JUDGE = CFG.server_kwargs["sem_judge"]
# VLM-judge submission filtering: batch ALL frames into ONE multi-image request
# instead of the stock per-image fan-out (~N separate calls). The judge's latency is
# dominated by call COUNT, not per-call speed — the stock fan-out makes ~N submission
# calls + criteria ≈ 30 sequential calls/eval (p50 ~116s on gpt-4.1, brushing the
# 300s step_timeout). Batching collapses the fan-out to 1 (like the agent reads many
# images in one call), so the judge does ~9 calls (~35s). Default TRUE because the
# gpt-4.1 judge needs it to stay under step_timeout at concurrency. Set false for the
# stock per-image behavior. NOTE: batching judges frames jointly vs in isolation — a
# reward-semantics difference vs stock (A/B follow-up). See _batched_judge_submission.
_JUDGE_BATCH_SUBMISSION = CFG.server_kwargs["judge_batch_submission"]
# Max frames per batched submission call. The batched filter emits a per-image YES/NO
# line for EVERY frame, and long structured output degrades (the model drops/misorders
# lines past ~30-40 → parse-miss → None → guard ERR). It's ALSO bounded by the provider's
# ~50-image/request limit (stock caps its criterion call at 48). So chunk: eval tiers
# (max_steps_eval 30/50/70) and a fully-stepped trajectory split into ceil(n/30) calls
# rather than one oversized request that 400s deterministically. 30 keeps the common
# eval-30 tier a single call; train (<=35) stays 1-2 calls. See _batched_judge_submission.
_JUDGE_SUBMISSION_MAX_IMAGES_PER_CALL = 30
# VLM-judge OpenAI client retry budget. At higher judge concurrency gpt-4.1 starts
# returning 429s; the openai SDK default (2 retries) absorbs most, and the rest get
# swallowed by the Evaluator → caught by _judge_incomplete_reason → trajectory ERR.
# Raising the client's retries (it honors Azure's Retry-After) absorbs those residual
# 429s so the eval SUCCEEDS instead of ERRing. Off the event loop + bounded by
# step_timeout, so a generous value is safe. <0 = SDK default. See _get_judge_evaluator_cls.
_JUDGE_OPENAI_MAX_RETRIES = CFG.server_kwargs["judge_openai_max_retries"]
_BLANK_SCREENSHOT_MAX_RETRIES = CFG.server_kwargs["blank_screenshot_max_retries"]
_BLANK_SCREENSHOT_WAIT = CFG.server_kwargs["blank_screenshot_wait"]
# L2 fail-fast: after this many CONSECUTIVE steps that could only
# produce a fallback (stale/blank) screenshot, treat the pool as unreachable and
# truncate the trajectory loudly — rather than grind every remaining step through
# a teacher turn on a dead pool (which silently completes as episode_return=0 and
# keeps count/throughput looking normal). A real screenshot resets the counter,
# so a transient single-instance blip never trips it; only a sustained outage
# does. Conservative (>1) so normal one-off fallbacks degrade gracefully. yaml-
# driven like the other tunables (0 disables); change via configs/default.yaml.
_POOL_UNREACHABLE_FALLBACK_STEPS = CFG.server_kwargs["pool_unreachable_fallback_steps"]
_MIN_SCREENSHOT_SIZE = CFG.server_kwargs["min_screenshot_size"]
_WAIT_FOR_CONTENT_TIMEOUT = CFG.server_kwargs["wait_for_content_timeout"]
_WAIT_FOR_CONTENT_INTERVAL = CFG.server_kwargs["wait_for_content_interval"]
# ============================================================================

# Tool surface: the grounded recipe (default) advertises native
# click/type/scroll/key/wait GUI verbs in ``valid_actions``. Browser nav
# (``goto``/``back``/...) and terminal tools (``response``/``terminate``) are
# opt-in via ``env_kwargs.extra_tools`` (canonical names); ``step`` below
# executes any offered tool by canonical name.
#
# ``metadata.valid_actions`` defaults to ``_VALID_ACTIONS`` (yaml-sourced, then
# non-GUI names stripped — the grounded GUI interaction verbs; no nav and no
# finish tools). It gates Qwen's ``computer_use`` enum; gpt/claude's black-box
# native tool can't be per-action filtered, so grounded entries are inert there
# (advisory only — their @browser ``filter_child_action_enum`` logs at debug, no warning).
#
# DO NOT override this to ``valid_actions: ["response"]`` in any teacher config:
# response/terminate are terminal extra tools. Keeping them out of
# ``valid_actions`` prevents SFT exports from treating finish as GUI action enum
# entries.

# Extra tools webgym can actually EXECUTE. OmniBoxes is single-tab and exposes
# only back/goto for nav. Canonical response/terminate are resolved centrally
# from LiteFinishToolSet when selected by env_kwargs.extra_tools.
_WEBGYM_NAV_TOOLS: tuple[str, ...] = ("goto", "back")
_WEBGYM_UNSUPPORTED_ACTIONS = (
    frozenset({"screenshot", "cursor_position", "mouse_down", "mouse_up"})
    | (LITE_DESKTOP_KEY_ACTION_NAMES - {"key"})
)

class WebgymTools(BaseTools):
    """What webgym declares beyond the GUI surface: browser nav (goto/back)."""

    _SCHEMAS: ClassVar[dict[str, dict[str, Any]]] = {
        tool_schema_name(schema): schema
        for schema in LiteBrowserNavToolSet.get_tool_schemas(include=list(_WEBGYM_NAV_TOOLS))
    }


#: Finish tools cannot live in an env's own set, so the union is not optional.
_KNOWN_STANDALONE_TOOL_NAMES = WebgymTools.get_tool_names() | LiteFinishToolSet.get_tool_names()

#: Builder-side no-override default, resolved ONCE at import (the builder
#: runs per registered task — ~293k for webgym).
_DEFAULT_EXTRA_TOOL_SCHEMAS = resolve_extra_tools(_EXTRA_TOOLS, tools=WebgymTools, env_name="webgym")


def _page_feedback_text(
    page_title: str | None,
    url: str | None,
    obs_text: str | None = None,
) -> str:
    if obs_text:
        return obs_text
    parts = []
    if page_title:
        parts.append(f"Current page title: {page_title}")
    if url:
        parts.append(f"Current URL: {url}")
    return "\n".join(parts) or "Current page context is unchanged."


# ---------------------------------------------------------------------------
# Network error classification (#4)
# ---------------------------------------------------------------------------

def _is_retryable(exc: Exception) -> bool:
    """Return True if *exc* is a transient network/server error worth retrying.

    Read only what the raiser typed -- httpx already classifies its own transport
    failures, so these three groups ARE the transient set:

    * ``TimeoutException`` — Read/Write/Connect/PoolTimeout;
    * ``NetworkError`` — Connect/Read/Write/CloseError, i.e. the raw socket
      failures (TLS handshake as ``ConnectError``, peer reset as ``ReadError``);
    * ``RemoteProtocolError`` — the peer broke the protocol mid-response.

    Deliberately excluded: ``LocalProtocolError`` (our own malformed request, a
    re-send fails identically), ``UnsupportedProtocol`` / ``ProxyError``
    (configuration, not weather), and every ``HTTPStatusError`` except 503.
    """
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)):
        return True
    # 503 Service Unavailable from server
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 503:
        return True
    return False


def _reports_instance_dead(response: httpx.Response) -> bool:
    """True if the master's error body carries the typed ``instance_dead`` flag.

    A closed browser/page/context is a PERMANENT failure for this instance, so
    ``execute()`` must fail fast instead of burning 3 attempts + backoff, and the
    trajectory must end instead of grinding every remaining step (and GPT turn)
    against the corpse.

    The flag is produced in-container by ``docker/patches/playwright_instance.py``
    (typed ``TargetClosedError`` / closed-page state, never a message match) and
    survives both remaining hops — see those patches' docstrings. Read the WHOLE
    body, not the truncated log slice: the flag is the LAST key the master emits,
    so it falls outside ``r.text[:200]``.

    Polarity is deliberately conservative: a non-JSON body, or one without the
    key, is NOT a recognised death and falls through to the ordinary 5xx retry.
    Declaring a live browser dead would end the episode, so "unknown" must mean
    "keep retrying". ``.get`` is right here — this is another process's HTTP
    response, a true system boundary, not an internal path.
    """
    try:
        payload = response.json()
    except ValueError:
        return False
    return isinstance(payload, dict) and payload.get("instance_dead") is True

# ---------------------------------------------------------------------------
# VLM-judge submission batching
# ---------------------------------------------------------------------------
# The webgym Evaluator filters submission relevance one OpenAI call PER frame (a
# 32-wide fan-out). On gpt-4.1 that's ~30 sequential calls/eval → the judge brushes
# the 300s step_timeout (p50 ~116s, observed 19% timeouts at conc-32/sample-48).
# A subclass batches those per-frame calls into ONE multi-image request (the agent's
# call shape), cutting the count ~N x. Clean OOP extension, no .venv edits / no
# monkey-patching. Gated by _JUDGE_BATCH_SUBMISSION.

_judge_evaluator_cls: type | None = None


def _is_content_policy_error(e: BaseException) -> bool:
    """True for an Azure/OpenAI content-safety refusal (HTTP 400 content_policy_violation).
    These are DETERMINISTIC — the same image is always rejected — so unlike a 429/5xx/timeout
    a retry can NEVER recover them. The submission filter treats such a frame as excluded
    (NO/irrelevant) rather than failing the whole eval: it can only LOWER reward (drop one
    frame of evidence), never fabricate a pass, so the no-silent-corruption invariant holds
    while a content-flagged screenshot stops being a hard rollout-failure bottleneck."""
    if getattr(e, "code", None) == "content_policy_violation":
        return True
    s = str(e).lower()
    return (
        "content_policy_violation" in s
        or "content safety" in s
        or "content management policy" in s
    )


def _read_b64(path: str) -> str:
    with open(path, "rb") as fh:
        import base64
        return base64.b64encode(fh.read()).decode("utf-8")


def _batched_judge_submission(evaluator: Any, trajectory: list[dict]) -> None:
    """Batched replacement for ``Evaluator.judge_submission_images``: judge frame
    relevance in MULTI-image OpenAI requests (per-image YES/NO) instead of the stock
    N-separate-calls fan-out. Mutates ``trajectory[i].reward.submit /
    .submission_judgment`` in place exactly like the stock method (the last frame is
    always force-included). A failed call OR a missing/unparseable per-image decision
    leaves ``submission_judgment=None`` so ``_judge_incomplete_reason`` raises — never
    a silent wrong reward.

    Frames are CHUNKED at _JUDGE_SUBMISSION_MAX_IMAGES_PER_CALL: one oversized request
    (eval trajectories reach ~70 frames) both degrades per-image parse reliability and
    can exceed the provider's ~50-image/request limit → a deterministic 400 that no
    retry can fix. Each chunk is one multi-image call; a chunk that hits the content
    filter falls back to per-image so only the offending frame is excluded."""
    keypoint_client, keypoint_model = evaluator._get_client_and_model(
        evaluator.TASK_KEYPOINT_DETECTION
    )
    task = trajectory[0]["observation"].task
    task_name = task.task_name
    key_points = "\n".join(
        f"- {r['description'] if isinstance(r, dict) else r}" for r in task.evaluator_reference
    )
    steps_with_images = [
        (i, s) for i, s in enumerate(trajectory)
        if s.get("observation") and hasattr(s["observation"], "image_path")
    ]
    if not steps_with_images:
        return
    last_image_index = steps_with_images[-1][0]
    steps_to_judge = steps_with_images[:-1]  # last frame is always force-included

    from webgym.data.components import Reward

    def _set(i: int, submit: bool, judgment: str | None) -> None:
        rw = trajectory[i].get("reward")
        if rw is None:
            trajectory[i]["reward"] = Reward(
                reward=0, evaluation="", submit=submit, submission_judgment=judgment
            )
        else:
            rw.submit = submit
            rw.submission_judgment = judgment

    system_msg = (
        "You are an expert evaluator deciding, for EACH screenshot, whether it "
        "contains information relevant to completing a task.\n"
        "- YES if the image shows ANY task-related content (actions, progress, "
        "search results, tool usage, errors, blocking screens).\n"
        "- NO only if completely irrelevant (generic homepage, unrelated page, blank).\n"
        "- When in doubt, answer YES."
    )
    cap = _JUDGE_SUBMISSION_MAX_IMAGES_PER_CALL
    for c0 in range(0, len(steps_to_judge), cap):
        _judge_submission_chunk(
            keypoint_client, keypoint_model, system_msg, task_name, key_points,
            steps_to_judge[c0:c0 + cap], _set,
        )

    _set(last_image_index, True, "Last step is always included for evaluation")


def _judge_submission_chunk(
    client: Any, model: str, system_msg: str, task_name: str, key_points: str,
    chunk: list[tuple[int, dict]], set_fn: Any,
) -> None:
    """Judge ONE chunk of <=cap frames in a single multi-image call. ``chunk`` is a list
    of ``(trajectory_index, step)``; labels are re-based to ``Image 0..len(chunk)-1`` per
    chunk and mapped back to the trajectory index via ``set_fn``."""
    import re as _re

    n = len(chunk)
    instr = (
        f"**Task**: {task_name}\n\n**Key Points for Task Completion**:\n{key_points}\n\n"
        f"You are shown {n} screenshots, labeled 'Image 0' .. 'Image {n - 1}'. For EACH "
        f"image, decide YES or NO. Respond with EXACTLY one line per image, in order, "
        f"nothing else:\nImage 0: YES|NO\nImage 1: YES|NO\n... up to Image {n - 1}."
    )
    content: list[dict] = [{"type": "text", "text": instr}]
    for k, (_i, step) in enumerate(chunk):
        content.append({"type": "text", "text": f"Image {k}:"})
        content.append(
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{_read_b64(step['observation'].image_path)}", "detail": "high"}}
        )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": content},
    ]
    try:
        completion = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=24 + 12 * n, temperature=0.0, top_p=0.95, stream=False,
        )
        resp = completion.choices[0].message.content or ""
        for k, (i, _step) in enumerate(chunk):
            mt = _re.search(rf"Image\s*{k}\b\s*[:\-]\s*\**\s*(YES|NO)", resp, _re.IGNORECASE)
            if mt:
                decision = mt.group(1).upper()
                set_fn(i, decision == "YES", f"[batched] Image {k}: {decision}")
            else:
                set_fn(i, False, None)  # missing decision -> guard raises (no silent verdict)
    except Exception as e:  # noqa: BLE001 — surface as None judgments -> guard raises
        if _is_content_policy_error(e):
            # ONE frame tripped Azure's content filter → the WHOLE multi-image chunk
            # 400s (all-or-nothing). Re-judge per-image so only the OFFENDING frame is
            # affected: a deterministic content-policy refusal excludes that frame
            # (NO + a real verdict, so the guard passes — it can only lower reward,
            # never fabricate a pass), while every other frame is judged normally. A
            # *transient* per-image error still leaves judgment=None → guard raises → retry.
            print(f"Image chunk hit Azure content filter; re-judging per-image to isolate: {e}")
            for i, step in chunk:
                msgs = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": [
                        {"type": "text", "text": (
                            f"**Task**: {task_name}\n\n**Key Points for Task Completion**:\n{key_points}\n\n"
                            "Does this screenshot contain task-relevant information? Respond with "
                            "EXACTLY one line, nothing else:\nDecision: YES|NO"
                        )},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{_read_b64(step['observation'].image_path)}", "detail": "high"}},
                    ]},
                ]
                try:
                    completion = client.chat.completions.create(
                        model=model, messages=msgs,
                        max_tokens=36, temperature=0.0, top_p=0.95, stream=False,
                    )
                    r1 = completion.choices[0].message.content or ""
                    m1 = _re.search(r"Decision\s*[:\-]\s*\**\s*(YES|NO)", r1, _re.IGNORECASE)
                    if m1:
                        d = m1.group(1).upper()
                        set_fn(i, d == "YES", f"[per-image] Decision: {d}")
                    else:
                        set_fn(i, False, None)  # parse miss -> guard raises (retry)
                except Exception as e2:  # noqa: BLE001
                    if _is_content_policy_error(e2):
                        print(f"⚠️ Frame at step {i} EXCLUDED: Azure content filter refused the image ({e2})")
                        set_fn(i, False, "[content-filtered] excluded: Azure content_policy_violation (not a relevance signal)")
                    else:
                        set_fn(i, False, None)  # transient -> guard raises (retry)
        else:
            print(f"Error judging image chunk (call failed): {e}")
            for i, _step in chunk:
                set_fn(i, False, None)


def _get_judge_evaluator_cls() -> type:
    """webgym Evaluator subclass whose ``judge_submission_images`` batches the per-frame
    fan-out into ONE call when _JUDGE_BATCH_SUBMISSION is on (else stock per-image).
    Built lazily — webgym is an optional import."""
    global _judge_evaluator_cls
    if _judge_evaluator_cls is None:
        from openai import OpenAI
        from webgym.models.evaluator import Evaluator

        class _BatchingEvaluator(Evaluator):
            def judge_submission_images(self, trajectory: list[dict]) -> None:
                if _JUDGE_BATCH_SUBMISSION:
                    _batched_judge_submission(self, trajectory)
                else:
                    super().judge_submission_images(trajectory)

            def _get_client_for_config(self, config: dict) -> Any:
                # Same as stock, but give the client a larger retry budget so 429s are
                # absorbed (Retry-After backoff) instead of swallowed → ERR. See
                # _JUDGE_OPENAI_MAX_RETRIES.
                api_key_env_var = config["openai_api_key_env_var"]
                base_url = config.get("base_url")
                cache_key = (api_key_env_var, base_url)
                if cache_key not in self._clients:
                    if api_key_env_var not in os.environ:
                        raise ValueError(f"Environment variable {api_key_env_var} not found")
                    kw: dict[str, Any] = {"api_key": os.environ[api_key_env_var]}
                    if base_url:
                        kw["base_url"] = base_url
                    if _JUDGE_OPENAI_MAX_RETRIES is not None and _JUDGE_OPENAI_MAX_RETRIES >= 0:
                        kw["max_retries"] = _JUDGE_OPENAI_MAX_RETRIES
                    self._clients[cache_key] = OpenAI(**kw)
                return self._clients[cache_key]

        _judge_evaluator_cls = _BatchingEvaluator
    return _judge_evaluator_cls


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

class WebGymClient:
    """Async HTTP client for WebGym OmniBoxes Master server."""

    # Class-level backpressure semaphores (#13) — shared across all instances
    # so that concurrent WebGymEnv instances collectively respect the limits.
    _sem_allocate: asyncio.Semaphore | None = None
    _sem_execute: asyncio.Semaphore | None = None
    _sem_screenshot: asyncio.Semaphore | None = None
    _sem_navigate: asyncio.Semaphore | None = None
    # VLM-judge concurrency cap (terminal evaluate()); see _SEM_JUDGE.
    _sem_judge: asyncio.Semaphore | None = None

    @classmethod
    def _init_semaphores(cls) -> None:
        """Lazily create semaphores (must be called inside a running event loop)."""
        if cls._sem_allocate is None:
            cls._sem_allocate = asyncio.Semaphore(_SEM_ALLOCATE)
            cls._sem_execute = asyncio.Semaphore(_SEM_EXECUTE)
            cls._sem_screenshot = asyncio.Semaphore(_SEM_SCREENSHOT)
            cls._sem_navigate = asyncio.Semaphore(_SEM_NAVIGATE)
            cls._sem_judge = asyncio.Semaphore(_SEM_JUDGE)

    def __init__(self, master_url: str, api_key: str):
        self._client = httpx.AsyncClient(
            base_url=master_url,
            headers={"x-api-key": api_key},
            timeout=httpx.Timeout(_HTTP_TIMEOUT),
        )
        self._init_semaphores()
        # Per-operation metrics (#18)
        self._metrics: dict[str, dict[str, int]] = {
            op: {"total": 0, "completed": 0, "failed": 0, "retried": 0}
            for op in ("allocate", "execute", "screenshot", "metadata", "release", "get_page_metadata")
        }
        # Navigation error log (#19)
        self._failed_urls: list[str] = []

    async def get_instance(self, retries: int = 20, backoff: float = 3.0) -> dict:
        """Lease a browser instance from the pool.

        Retries on 503 (no available instances) with exponential backoff,
        because the master server's available count lags behind the node's
        actual state (updated every ~10s by the health check worker), and
        because under concurrency a request should WAIT for a peer's lease
        to free rather than bounce — waiting grabs the instance the moment
        one is released (max throughput), whereas bouncing to the
        env-server's 503 + Retry-After path would idle the request for the
        full Retry-After window even if the pool freed up sooner.

        Holding the env-server L2 admission slot during the wait is safe:
        the L2 in-flight cap (host vCPU x4) vastly exceeds the OmniBoxes
        pool size (``WEBGYM_INSTANCES``), so waiting leasers cannot starve
        admission. On genuine, sustained saturation the bounded retry
        budget still expires and raises ``CapacityExhausted`` (-> 503 +
        Retry-After) as the safety valve, uniform with other bounded-pool
        envs. Throughput is bounded only by the pool size, which is sized
        once at deploy time (``deploy.py N`` / ``WEBGYM_INSTANCES``).

        Returns:
            {"instance_id": "uuid:port", "node": "hash"}
        """
        async with self._sem_allocate:
            self._metrics["allocate"]["total"] += 1
            last_exc: Exception | None = None
            for attempt in range(retries):
                try:
                    r = await self._client.post(
                        "/get", params={"lifetime_mins": _INSTANCE_LIFETIME_MINS}
                    )
                except Exception as e:
                    # Network-level error (ReadError, ConnectError, etc.)
                    last_exc = e
                    if _is_retryable(e) and attempt < retries - 1:
                        self._metrics["allocate"]["retried"] += 1
                        wait = backoff * (1.2 ** attempt)
                        logger.warning("Network error on get_instance (attempt %d/%d): %s, retrying in %.1fs",
                                       attempt + 1, retries, e, wait)
                        await asyncio.sleep(wait)
                        continue
                    raise
                if r.status_code == 503:
                    if attempt < retries - 1:
                        self._metrics["allocate"]["retried"] += 1
                        wait = backoff * (1.2 ** attempt)
                        logger.debug(
                            "No instances available, retrying in %.1fs "
                            "(attempt %d/%d)", wait, attempt + 1, retries,
                        )
                        await asyncio.sleep(wait)
                        continue
                    # Exhausted: translate the terminal upstream 503 into a
                    # typed CapacityExhausted so the env-server's exception
                    # handler can map it to 503 + Retry-After uniformly with
                    # other bounded-pool envs. Without this translation, the
                    # raw httpx.HTTPStatusError would bubble up as a 500 from
                    # /reset and the client would treat it as fatal.
                    self._metrics["allocate"]["failed"] += 1
                    raise CapacityExhausted(
                        what=(
                            f"webgym OmniBoxes pool full after {retries} "
                            f"retries (no instance available); WEBGYM_INSTANCES "
                            f"is the upstream cap"
                        ),
                        retry_after_s=60.0,
                    )
                r.raise_for_status()
                self._metrics["allocate"]["completed"] += 1
                return r.json()
            self._metrics["allocate"]["failed"] += 1
            if last_exc is not None:
                raise last_exc
            r.raise_for_status()  # unreachable, but for type checker
            return {}

    async def execute(self, instance: dict, command: dict) -> dict:
        """Execute a browser command on an instance with retry (#2).

        Args:
            instance: {"instance_id": ..., "node": ...} from get_instance()
            command: {"command_name": {args}} e.g. {"click_coords": {"x": 500, "y": 300}}

        Returns:
            Response JSON (usually {"status": "success"})

        Raises:
            httpx.HTTPStatusError only for non-retryable 4xx errors.
        """
        async with self._sem_execute:
            self._metrics["execute"]["total"] += 1
            last_exc: Exception | None = None
            for attempt in range(_MAX_RETRIES):
                try:
                    payload = {**instance, **command}
                    r = await self._client.post("/execute", json=payload)
                    if r.status_code >= 500:
                        body = r.text[:200]
                        # Dead browser/page/context is PERMANENT for this instance:
                        # don't burn the remaining retries+backoff against a corpse —
                        # fail fast and signal the caller to end the trajectory.
                        if _reports_instance_dead(r):
                            self._metrics["execute"]["failed"] += 1
                            logger.warning("Instance browser closed on execute (non-retryable): %s", body)
                            return {"status": "error", "message": body, "instance_dead": True}
                        logger.warning("Server error %d on execute (attempt %d/%d): %s",
                                       r.status_code, attempt + 1, _MAX_RETRIES, body)
                        if attempt < _MAX_RETRIES - 1:
                            self._metrics["execute"]["retried"] += 1
                            await asyncio.sleep(_RETRY_BACKOFF_BASE * (2 ** attempt))
                            continue
                        self._metrics["execute"]["failed"] += 1
                        return {"status": "error", "message": body}
                    r.raise_for_status()
                    self._metrics["execute"]["completed"] += 1
                    return r.json()
                except Exception as e:
                    last_exc = e
                    if _is_retryable(e) and attempt < _MAX_RETRIES - 1:
                        self._metrics["execute"]["retried"] += 1
                        logger.warning("Retryable error on execute (attempt %d/%d): %s",
                                       attempt + 1, _MAX_RETRIES, e)
                        await asyncio.sleep(_RETRY_BACKOFF_BASE * (2 ** attempt))
                        continue
                    # Non-retryable or last attempt
                    if not _is_retryable(e):
                        raise
                    break
            # Persistent failure — return error dict to keep episode alive
            self._metrics["execute"]["failed"] += 1
            logger.warning("Execute failed after %d retries: %s", _MAX_RETRIES, last_exc)
            return {"status": "error", "message": str(last_exc)[:200]}

    async def screenshot(self, instance: dict, mode: str = "coordinates", cursor: bool = True) -> bytes:
        """Capture a screenshot of the instance viewport using streaming (#14).

        Args:
            instance: Instance dict from get_instance()
            mode: "coordinates" (plain) or "set_of_marks" (annotated)
            cursor: Whether to draw the shared cursor sprite at capture time.

        Returns:
            Raw PNG bytes

        Raises:
            httpx.HTTPStatusError on 4xx. 5xx errors raise after logging.
        """
        async with self._sem_screenshot:
            self._metrics["screenshot"]["total"] += 1
            try:
                chunks: list[bytes] = []
                async with self._client.stream(
                    "GET",
                    "/screenshot",
                    params={
                        **instance,
                        "interaction_mode": mode,
                        "cursor": "1" if cursor else "0",
                    },
                ) as r:
                    if r.status_code >= 500:
                        await r.aread()
                        logger.warning("Server error %d on screenshot: %s",
                                       r.status_code, r.text[:200])
                        r.raise_for_status()  # raises, caught by outer except
                    r.raise_for_status()  # 4xx errors
                    async for chunk in r.aiter_bytes():
                        chunks.append(chunk)
                self._metrics["screenshot"]["completed"] += 1
                return b"".join(chunks)
            except Exception:
                self._metrics["screenshot"]["failed"] += 1
                raise

    async def get_metadata(self, instance: dict) -> dict:
        """Get viewport dimensions.

        Returns:
            {"width": 1280, "height": 720}
        """
        self._metrics["metadata"]["total"] += 1
        try:
            r = await self._client.get("/metadata", params=instance)
            r.raise_for_status()
            self._metrics["metadata"]["completed"] += 1
            return r.json()
        except Exception:
            self._metrics["metadata"]["failed"] += 1
            raise

    async def get_page_metadata(self, instance: dict) -> dict:
        """Get current page metadata (title, url) via execute (#12).

        Returns:
            {"title": str, "url": str}
        """
        self._metrics["get_page_metadata"]["total"] += 1
        try:
            result = await self.execute(instance, {"get_page_metadata": {}})
            if result.get("status") == "error":
                self._metrics["get_page_metadata"]["failed"] += 1
                return {"title": "", "url": ""}
            title = result.get("title", "")
            url = result.get("url", "")
            self._metrics["get_page_metadata"]["completed"] += 1
            return {"title": title, "url": url}
        except Exception as e:
            self._metrics["get_page_metadata"]["failed"] += 1
            logger.warning("get_page_metadata failed: %s", e)
            return {"title": "", "url": ""}

    async def reset_instance(self, instance: dict) -> None:
        """Release an instance back to the pool."""
        self._metrics["release"]["total"] += 1
        try:
            r = await self._client.post("/reset", params=instance)
            r.raise_for_status()
            self._metrics["release"]["completed"] += 1
        except Exception:
            self._metrics["release"]["failed"] += 1
            raise

    async def info(self) -> dict:
        """Get server status (node capacities, availability)."""
        r = await self._client.get("/info")
        r.raise_for_status()
        return r.json()

    async def close(self) -> None:
        """Close the HTTP client. Log metrics summary (#18) and failed URLs (#19)."""
        # Log metrics summary
        for op, counts in self._metrics.items():
            if counts["total"] > 0:
                logger.info("WebGymClient metrics [%s]: %s", op, counts)
        # Log failed URLs
        if self._failed_urls:
            logger.warning("WebGymClient failed URLs (%d): %s",
                           len(self._failed_urls), self._failed_urls[:20])
        await self._client.aclose()

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class WebGymEnv(LiteBaseEnv):
    """CUA-Lite wrapper for WebGym OmniBoxes environments.

    Communicates with the WebGym OmniBoxes HTTP API to control Playwright
    browser instances. Actions use [0, 1000] normalized coordinates converted
    to pixels. The grounded recipe (default) advertises GUI verbs through
    ``valid_actions``. Browser nav (``goto``/``back``/…) and terminal
    ``response``/``terminate`` are **opt-in** via ``env_kwargs.extra_tools``;
    ``step`` executes any offered tool by its canonical name.
    """

    EXTRA_TOOLS: ClassVar[type[BaseTools]] = WebgymTools

    def __init__(
        self,
        *,
        task: dict,
        # User-facing tunables (§6c canonical order: max_steps → post_action_delay →
        # … → eval_config → valid_actions → extra_tools). All yaml-sourced.
        max_steps: int | None = None,
        post_action_delay: float = _POST_ACTION_DELAY,
        viewport: tuple[int, int] = _VIEWPORT,
        goback_skip_threshold: int = _GOBACK_SKIP_THRESHOLD,
        goback_terminate_threshold: int = _GOBACK_TERMINATE_THRESHOLD,
        eval_config: dict | None = None,
        valid_actions: list[str] | None = _VALID_ACTIONS,
        extra_tools: list[str] | None = _EXTRA_TOOLS,
        cursor: bool = True,
        # ── internal / registration ──
        split: str = "eval",
        skip_eval: bool = False,
        # When the episode-end judge finds the live site bot-blocked the agent
        # (``info.blocking.is_blocked``), raise ``EnvBlocked`` so the episode is
        # treated as a void env-error (excluded from the eval denominator,
        # dropped from GRPO) — never a reward-0 model failure. Default on; set
        # False to keep blocked tasks scored as reward-0 (e.g. block-rate study).
        block_is_error: bool = True,
        instruction_template: str | None = None,
        **kwargs: Any,
    ):
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"WebGymEnv got unexpected env kwargs: {unknown}")
        # Per-run eval_config (env_kwargs) overlays the registration default.
        self._eval_config = deep_merge(_EVAL_CONFIG, eval_config) if eval_config else dict(_EVAL_CONFIG)
        # dict(...): `task` is the shared registration-side row — copy,
        # don't alias.
        self._task = dict(task)
        # Infra / secrets are NEVER __init__ params: exactly one
        # source each — env-var, else yaml default.
        self._master_url = os.environ.get("WEBGYM_MASTER_URL") or _MASTER_URL
        self._api_key = os.environ.get("WEBGYM_API_KEY", _API_KEY)
        diff = int(task.get("difficulty", 2))
        steps_map = _MAX_STEPS_TRAIN if split == "train" else _MAX_STEPS_EVAL
        default_steps = 15 if diff <= 3 else (25 if diff <= 6 else 35) if split == "train" \
            else 30 if diff <= 3 else (50 if diff <= 6 else 70)
        self._max_steps = max_steps or steps_map.get(diff, default_steps)
        self._post_action_delay = post_action_delay
        self._cursor = cursor
        self._skip_eval = skip_eval
        self._block_is_error = block_is_error
        self._instruction_template = instruction_template
        self._step_count = 0
        self._terminated = False
        self._client: WebGymClient | None = None
        self._instance: dict | None = None
        # Set when execute() reports the browser/page died (permanent). Ends the
        # trajectory early instead of grinding every remaining step + GPT turn
        # against a dead instance. Reset per-episode in reset().
        self._instance_dead: bool = False
        # Viewport is a POOL-LEVEL setting (browsers launch once at container boot —
        # see WebGymContainerServices.ensure forwarding -e WEBGYM_VIEWPORT), NOT
        # per-episode. The host stores it observationally and re-adopts the
        # container-reported size from get_metadata() at reset().
        self._viewport: tuple[int, int] = tuple(viewport)
        self._goback_skip_threshold = goback_skip_threshold
        self._goback_terminate_threshold = goback_terminate_threshold
        # ONE frame per step (plus reset's), indexed against _actions /
        # _observations by _evaluate to build the judge's trajectory. A step that
        # captures a frame per action still contributes exactly its LAST frame
        # here, so the three lists stay index-aligned.
        self._screenshots: list[bytes] = []  # for evaluation
        # The most recent captured frame, whether or not it entered the per-step
        # record above. This is what a failed capture falls back to.
        self._last_frame: bytes | None = None
        self._actions: list[str] = []  # action strings per step, for evaluation
        self._observations: list[str] = []  # observation text per step (page changed/didn't change)
        self._agent_response: str | None = None
        # Track last click position for fill_coords (used by type action)
        self._last_click_x: float = 0.0
        self._last_click_y: float = 0.0
        # Goback-from-homepage protection (#6)
        self._goback_from_homepage_count: int = 0
        self._homepage_url: str = ""
        # Navigation failure tracking (#17)
        self._is_fallback_screenshot: bool = False
        # Current page URL for observation text (#16)
        self._current_page_url: str = ""
        # Grounded GUI interaction subset. Unconditional assignment: the
        # signature default (yaml-sourced ``_VALID_ACTIONS``) is the single
        # source of truth for "omitted", so an EXPLICIT ``valid_actions: null``
        # keeps its shared meaning (no filtering) instead of silently snapping
        # back to the yaml subset.
        self._valid_actions = resolve_valid_actions(
            valid_actions, env_name="webgym", platform="browser",
        )
        # Resolve + validate via the same tool set the builder default uses
        # (registered == no-override live by construction).
        self._extra_tool_schemas = type(self).extra_tool_schemas(extra_tools)

    @staticmethod
    def _task_metadata(task: dict[str, Any]) -> LiteCUAMetadata:
        """Same-source metadata builder.
        valid_actions = the yaml grounded GUI subset (narrows qwen's advertised enum);
        extra tools are exactly the yaml-selected standalone tool schemas;
        viewport's DEFAULT is a yaml fact (the live env re-adopts the
        container-reported size at reset)."""
        return LiteCUAMetadata(
            dims=("browser", "use"),
            extra_tool_schemas=list(_DEFAULT_EXTRA_TOOL_SCHEMAS),
            valid_actions=copy_valid_actions(_VALID_ACTIONS),
            others={
                "website": str(task.get("website", "")),
                "domain": str(task.get("domain", "")),
                "subdomain": str(task.get("subdomain", "")),
                "difficulty": int(task.get("difficulty", 2)),
                "viewport": _VIEWPORT,
            },
        )

    def _runtime_metadata(self) -> LiteCUAMetadata:
        # env_kwargs amendments: valid_actions / extra_tools overrides + the
        # live viewport (env_kwarg default, re-adopted from the container-
        # reported size at reset()).
        md = self._task_metadata(self._task)
        return dataclasses.replace(
            md,
            valid_actions=self._valid_actions,
            extra_tool_schemas=self._extra_tool_schemas,
            others={**md.others, "viewport": self._viewport},
        )

    async def reset(self) -> LiteEnvObservation:
        if not self._skip_eval:
            self._require_eval_deps()

        # Clean up previous instance
        if self._instance is not None:
            await self._safe_release()

        self._step_count = 0
        self._terminated = False
        self._screenshots = []
        self._last_frame = None
        self._actions = []
        self._observations = []
        self._agent_response = None
        self._last_click_x = 0.0
        self._last_click_y = 0.0
        self._goback_from_homepage_count = 0
        self._is_fallback_screenshot = False
        self._current_page_url = ""
        self._instance_dead = False
        self._consecutive_fallback_steps = 0  # L2 dead-pool fail-fast counter

        # Create client if needed
        if self._client is None:
            self._client = WebGymClient(self._master_url, self._api_key)

        # Lease a browser instance
        try:
            self._instance = await self._client.get_instance()
        except httpx.ConnectError as e:
            # The OmniBoxes master is brought up by WebGymContainerServices.ensure
            # BEFORE reset; a ConnectError here means it is still WARMING (cold
            # boot / recycle), which is RECOVERABLE. Raise the typed transient so
            # the env-server maps it to 503 + Retry-After and the client retries —
            # consistent with get_instance's pool-exhaustion path above and with
            # browsergym/mobilegym. (A genuinely missing/unbuilt image is caught
            # earlier by ensure() as EnvDepsMissingError, never reaching here.)
            raise CapacityExhausted.warming(
                what=(
                    f"webgym OmniBoxes master at {self._master_url} not reachable "
                    f"yet (container warming up): {e}"
                ),
            ) from e
        logger.debug("Leased instance %s", self._instance.get("instance_id"))

        # Try-finally instance protection (#10): release on any error after allocation
        try:
            # Get actual viewport dimensions
            try:
                meta = await self._client.get_metadata(self._instance)
                self._viewport = (meta.get("width", 1280), meta.get("height", 720))
            except Exception as e:
                logger.warning("Failed to get viewport metadata, using default %s: %s",
                               _VIEWPORT, e)
                self._viewport = _VIEWPORT

            # Navigate to task website with retry (#3, #5)
            website = self._task.get("website", "")
            if website:
                if not website.startswith("http"):
                    website = f"https://{website}"
                self._homepage_url = website
                await self._navigate_with_retry(website)

                # Wait-for-content (#5): poll page metadata until title is non-empty
                await self._wait_for_content()

            # Initial screenshot + page metadata are independent node round-trips
            # — issue them in parallel (mirrors step()'s asyncio.gather) instead of
            # serially, saving one round-trip off the reset critical path.
            screenshot, page_meta = await asyncio.gather(
                self._take_screenshot(),
                self._client.get_page_metadata(self._instance),
            )
            # Slot 0 of the evaluator record (see _screenshots).
            self._screenshots.append(screenshot)
            self._current_page_url = page_meta.get("url", "")
            instruction = self._task.get("task_name", "")
            if self._instruction_template:
                instruction = self._instruction_template.format(
                    instruction=instruction,
                    website=website or "",
                )
            obs_metadata = {
                "page_title": page_meta.get("title", ""),
                "url": page_meta.get("url", ""),
            }
            return LiteEnvObservation(
                image=screenshot, text=instruction, metadata=obs_metadata,
            )
        except Exception:
            # Release instance on any unexpected error during reset
            await self._safe_release()
            raise

    async def step(self, actions: list[RuntimeEnvAction]) -> LiteEnvStepResult:
        input_actions = actions
        result_call_ids = ordered_tool_call_ids(input_actions)
        metadata = self.metadata
        actions_with_result_ids, ingress_errors = prepare_env_tool_calls(
            actions,
            metadata,
        )
        # Try-finally instance protection (#11)
        try:
            return await self._step_inner(
                actions_with_result_ids,
                input_actions,
                ingress_errors,
                result_call_ids,
            )
        except EnvBlocked:
            # Intentional void-episode signal (site blocked the agent), NOT a
            # crash — re-raise quietly. No early _safe_release: the episode is
            # over and the agent's env.close() releases the instance normally.
            raise
        except Exception as e:
            logger.error("Unexpected error in step(), releasing instance: %s", e)
            await self._safe_release()
            raise

    async def _step_inner(
        self,
        actions: list[tuple[EnvAction, str | None]],
        input_actions: list[RuntimeEnvAction],
        ingress_errors: dict[str, ToolErrorFeedback] | None = None,
        result_call_ids: list[str | None] | None = None,
    ) -> LiteEnvStepResult:
        """Core step logic, wrapped by step() with try-finally protection."""
        terminated = False
        # Model-emitted calls that ENDED the episode. They get no continuation
        # observation: devs/migration/verify.py forbids a tool result for a
        # terminal public model call. Env-private terminal calls have no public
        # id; wrapper-injected finishes can carry the intercepted model call's
        # result id, and that intercepted call must still be answered.
        terminal_call_ids: set[str] = set()
        step_action_strs: list[str] = []
        executed_actions: list[LiteExecutedAction] = []
        action_errors: dict[str, ToolErrorFeedback] = dict(ingress_errors or {})
        self._is_fallback_screenshot = False
        metadata = self.metadata
        extra_tool_schemas = metadata.extra_tool_schemas
        # One frame PER EXECUTED ACTION, in action order.
        step_screenshots: list[bytes] = []

        async def settle_and_capture() -> None:
            """Let the UI settle, then capture the frame this action earned.

            The delay must come BEFORE the capture or the frame records the
            pre-settle page. Every action that reached the browser captures,
            with no exception list, so the frame count is a pure function of how
            many actions ran.
            """
            if self._post_action_delay > 0:
                await asyncio.sleep(self._post_action_delay)
            step_screenshots.append(await self._take_screenshot())

        for action, result_call_id in actions:
            name = action["name"]
            args = action["arguments"]

            if rejected_reason := runtime_rejected_reason(action):
                if result_call_id:
                    append_feedback(
                        action_errors, result_call_id, current_feedback(rejected_reason),
                    )
                executed_actions.append({
                    "call": "noop",
                    "args": {"name": name, "reason": rejected_reason},
                })
                continue

            tool_availability = classify_standalone_tool_call(
                action, _KNOWN_STANDALONE_TOOL_NAMES, extra_tool_schemas,
            )
            if tool_availability == "inactive":
                if result_call_id:
                    action_errors[result_call_id] = current_feedback(
                        unavailable_action_message(name)
                    )
                executed_actions.append({
                    "call": "noop",
                    "args": {"name": name, "reason": "inactive extra tool"},
                })
                continue
            if tool_availability == "unknown":
                if result_call_id:
                    action_errors[result_call_id] = error_only_feedback(
                        unknown_tool_message(name)
                    )
                executed_actions.append({
                    "call": "noop",
                    "args": {"name": name, "reason": "unknown extra tool"},
                })
                continue

            invalid_action = invalid_action_message(action, metadata.valid_actions)
            if invalid_action:
                if result_call_id:
                    action_errors[result_call_id] = current_feedback(invalid_action)
                executed_actions.append({
                    "call": "noop",
                    "args": {"name": name, "reason": invalid_action},
                })
                continue

            if name == "terminate":
                step_action_strs.append(f"terminate({args})")
                terminated = True
                if result_call_id and not runtime_internal_stop_reason(action):
                    terminal_call_ids.add(result_call_id)
                break

            if name == "response":
                self._agent_response = args.get("text", "")
                step_action_strs.append(f"response(text={self._agent_response!r})")
                terminated = True
                if result_call_id and not runtime_internal_stop_reason(action):
                    terminal_call_ids.add(result_call_id)
                break

            # Sleep command local execution (#9, reference context_manager.py:260-262)
            if name == "wait":
                try:
                    duration = coerce_model_duration(
                        args.get("duration", args.get("time", 2.0)),
                        action_name="wait",
                    )
                except MODEL_ACTION_ERROR_TYPES as e:
                    record_model_action_error(action_errors, result_call_id, e, action_name=name)
                    executed_actions.append({"call": "noop", "args": {"name": name, "reason": str(e)}})
                    break
                step_action_strs.append(f"wait(duration={duration})")
                await asyncio.sleep(duration)
                await settle_and_capture()
                continue

            # Goback-from-homepage protection (#6)
            if name == "back":
                goback_result = self._handle_goback(step_action_strs)
                if goback_result == "skip":
                    continue
                if goback_result == "terminate":
                    terminated = True
                    break
                # goback_result == "execute" — fall through to normal execution
            else:
                # Non-back action resets the goback counter
                self._goback_from_homepage_count = 0

            # Translate and execute action
            step_action_strs.append(f"{name}({args})")
            try:
                command = self._translate_action(name, args)
            except MODEL_ACTION_ERROR_TYPES as e:
                record_model_action_error(action_errors, result_call_id, e, action_name=name)
                executed_actions.append({"call": "noop", "args": {"name": name, "reason": str(e)}})
                break
            if command is not None:
                # Record the WebGym command for executed_actions
                cmd_name = next(iter(command))
                executed_actions.append({"call": cmd_name, "args": command[cmd_name]})
                # Navigation commands get retry logic (#3)
                if name == "goto":
                    url = args.get("url", "")
                    error = await self._navigate_with_retry(url)
                    if error:
                        record_tool_execution_error(
                            action_errors,
                            result_call_id,
                            ModelVisibleErrorDetail(error),
                            action_name=name,
                        )
                else:
                    try:
                        result = await self._client.execute(self._instance, command)
                        if isinstance(result, dict) and result.get("instance_dead"):
                            # Browser died — remaining steps (and their GPT turns)
                            # are hopeless on this instance. Stop now; end the
                            # trajectory below instead of grinding to max_steps.
                            self._instance_dead = True
                            record_tool_execution_error(
                                action_errors,
                                result_call_id,
                                result.get("message") or "instance browser closed",
                                action_name=name,
                            )
                            executed_actions.append({
                                "call": "noop",
                                "args": {
                                    "name": name,
                                    "reason": "instance browser closed",
                                },
                            })
                            break
                        if isinstance(result, dict) and result.get("status") == "error":
                            record_tool_execution_error(
                                action_errors,
                                result_call_id,
                                result.get("message") or f"{name} failed",
                                action_name=name,
                            )
                    except Exception as e:
                        logger.warning("Action execution failed: %s — action: %s", e, name)
                        record_tool_execution_error(
                            action_errors, result_call_id, e, action_name=name
                        )
                        executed_actions.append({
                            "call": "noop",
                            "args": {"name": name, "reason": str(e)},
                        })
                # The command reached the browser (the ``instance_dead`` arm
                # above is the one exit that did not, and it breaks). An
                # execution error still earns a frame: the page may have changed
                # before the failure, and the model needs to see what it left.
                await settle_and_capture()
            elif name != "wait":
                if result_call_id:
                    feedback_fn = (
                        current_feedback if is_lite_action_name_or_action_batch_tool_name(name) else error_only_feedback
                    )
                    message = (
                        unsupported_action_message(name)
                        if is_lite_action_name_or_action_batch_tool_name(name)
                        else unknown_tool_message(name)
                    )
                    action_errors[result_call_id] = feedback_fn(message)
                executed_actions.append({"call": "noop", "args": {"name": name, "reason": "unknown action"}})

        self._actions.append("; ".join(step_action_strs) if step_action_strs else "noop")

        self._step_count += 1
        # End the episode on (a) step budget, or (b) a dead browser instance —
        # the latter can't recover on the same lease, so continuing only wastes
        # GPT turns on a corpse (the trajectory fails eval either way).
        truncated = not terminated and (
            self._step_count >= self._max_steps or self._instance_dead
        )

        # Each executed action already settled and captured its own frame inside
        # the loop. A step that executed nothing (empty batch, terminal-only
        # call, every call rejected) still owes the model one current
        # observation, so take the frame the loop never reached.
        if not step_screenshots:
            await settle_and_capture()
        # Only the LAST frame enters the per-step evaluator record, keeping
        # _screenshots index-aligned with _actions / _observations. Appended
        # before the change detection below, which compares this step's frame
        # against the previous step's.
        self._screenshots.append(step_screenshots[-1])

        if self._client is not None and self._instance is not None:
            page_meta = await self._client.get_page_metadata(self._instance)
            self._current_page_url = page_meta.get("url", "")
        else:
            page_meta = {"title": "", "url": ""}

        # Fail-fast on a site that keeps rendering blank: count CONSECUTIVE steps
        # whose capture succeeded but returned an all-white page. One is a
        # transient render race; N in a row means the site is not going to paint
        # for this agent, and every further step burns a teacher turn on a
        # trajectory that will score 0. Truncate so the budget is not wasted. A
        # real frame resets the counter. (A dead POOL no longer reaches here —
        # capture failures raise TrueInfraFailure in ``_take_screenshot``.)
        if self._is_fallback_screenshot:
            self._consecutive_fallback_steps += 1
            if (_POOL_UNREACHABLE_FALLBACK_STEPS > 0
                    and self._consecutive_fallback_steps >= _POOL_UNREACHABLE_FALLBACK_STEPS
                    and not truncated and not terminated):
                logger.error(
                    "webgym: %d consecutive all-white page renders — the site is not "
                    "painting for this agent; truncating trajectory to stop burning "
                    "teacher budget (master=%s)",
                    self._consecutive_fallback_steps, self._master_url,
                )
                truncated = True
        else:
            self._consecutive_fallback_steps = 0

        # Shorten URL if too long (reference async_webgym.py:1540-1546)
        url_display = self._current_page_url or "Unknown"
        if url_display != "Unknown" and len(url_display) > 60:
            try:
                import urllib.parse
                parsed = urllib.parse.urlparse(url_display)
                url_display = (f"{parsed.netloc}{parsed.path[:30]}..."
                               if len(parsed.path) > 30
                               else f"{parsed.netloc}{parsed.path}")
            except Exception:
                url_display = url_display[:60] + "..."

        # Detect if last action was a navigation (for observation text)
        last_action_is_navigate = False
        last_navigate_url = ""
        if step_action_strs:
            last_str = step_action_strs[-1]
            if last_str.startswith("goto("):
                last_action_is_navigate = True
                last_navigate_url = self._current_page_url

        # Generate observation text (reference async_webgym.py:1521-1581)
        obs_text = None

        # Blank-page fallback: the capture SUCCEEDED but the page rendered all
        # white, so the previous frame was re-appended and the last two frames are
        # identical by construction. Change detection has no evidence here and must
        # not run. Capture FAILURES never reach this branch — they raise
        # TrueInfraFailure in ``_take_screenshot`` and void the trajectory.
        if self._is_fallback_screenshot:
            if last_action_is_navigate and last_navigate_url:
                obs_text = (
                    f"Navigation failed: The website '{last_navigate_url}' returned a blank page "
                    "and is not accessible. The screenshot shows the previous page before the "
                    "failed navigation. Please try navigating to a different website or use a "
                    f"different approach to complete the task. Current URL: {url_display}"
                )
            else:
                obs_text = (
                    "The page rendered blank after the action above, so the image shown is the "
                    "previous page. The action may still have taken effect — the blank render "
                    "says nothing either way. Consider navigating elsewhere or trying a "
                    f"different approach. The URL of the webpage after executing the action: "
                    f"{url_display}"
                )
        elif len(self._screenshots) >= 2:
            images_identical = self._screenshots[-2] == self._screenshots[-1]
            if images_identical:
                if last_action_is_navigate and last_navigate_url:
                    obs_text = (
                        f"Navigation failed: The website '{last_navigate_url}' is not accessible "
                        "or does not exist. The page did not change. Please try navigating to a "
                        "different website or use a different approach to complete the task. "
                        f"Current URL: {url_display}"
                    )
                else:
                    obs_text = (
                        "After the action above is executed by the environment, the webpage did "
                        "not change (this means the last action is not effective). "
                        f"The URL of the webpage after executing the action: {url_display}"
                    )
            else:
                obs_text = (
                    "After the action above is executed by the environment, the webpage changed "
                    "(this means the last action was effective). "
                    f"The URL of the webpage after executing the action: {url_display}"
                )

        self._observations.append(obs_text or "")

        # Evaluate at episode end
        reward = None
        eval_info = {}
        if terminated or truncated:
            reward, eval_info = await self._evaluate()
            # Live site bot-blocked the agent → void episode (env's fault, not the
            # model's). Raise EnvBlocked so it rides the shared error path: out of
            # the eval denominator, dropped from GRPO. Terminal by construction
            # (is_blocked is only set at terminate/truncate). See errors.EnvBlocked.
            if self._block_is_error and eval_info.get("blocking", {}).get("is_blocked"):
                raise EnvBlocked(what="webgym: live site blocked the agent (is_blocked) — episode void")

        self._terminated = terminated
        info = {EXECUTED_ACTIONS_INFO_KEY: executed_actions, **eval_info}
        obs_metadata = {
            "page_title": page_meta.get("title", ""),
            "url": url_display,
        }
        result_text = _page_feedback_text(page_meta.get("title", ""), url_display, obs_text)
        return build_tool_results_from_decisions(
            LiteEnvStepResult(
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            ),
            ordered_call_ids=result_call_ids or [],
            continue_call_ids=[
                call_id for call_id in result_call_ids or []
                if call_id not in terminal_call_ids
            ],
            images=step_screenshots,
            text=result_text,
            metadata=obs_metadata,
            feedback=action_errors,
        )

    async def close(self) -> None:
        await self._safe_release()
        if self._client is not None:
            await self._client.close()
            self._client = None

    # -----------------------------------------------------------------------
    # Internal: navigation with retry (#3)
    # -----------------------------------------------------------------------

    async def _navigate_with_retry(self, url: str) -> str | None:
        """Navigate to *url* with graceful degradation.

        Acquires the navigate semaphore only around the actual visit_page call so
        retry backoff does not occupy a scarce navigation slot.

        Network-level retries are handled by WebGymClient.execute(). This method
        adds one extra retry on logical error (status=error) with backoff, matching
        reference navigate_with_retries() which does up to max_retries attempts.
        """
        last_error = "navigation failed"
        for attempt in range(_NAV_MAX_RETRIES):
            try:
                async with self._client._sem_navigate:
                    result = await self._client.execute(
                        self._instance, {"visit_page": {"url": url}}
                    )
                if result.get("status") != "error":
                    return None  # success
                # Dead browser — permanent for this instance; stop retrying the
                # nav and let the step loop end the trajectory.
                if result.get("instance_dead"):
                    self._instance_dead = True
                    logger.warning("Navigation to %s aborted: instance browser closed", url)
                    return result.get("message") or "instance browser closed"
                # Logical error (block page / cert error / hung domcontentloaded):
                # PERMANENT for this URL — execute() already exhausted its transient
                # network/500 retries, so an outer retry just re-pays the full goto
                # timeout. With nav_max_retries=1 this loop ends here (fail-fast).
                if attempt < _NAV_MAX_RETRIES - 1:
                    logger.warning("Navigation error for %s (attempt %d/%d), retrying",
                                   url, attempt + 1, _NAV_MAX_RETRIES)
                    await asyncio.sleep(_RETRY_BACKOFF_BASE * (2 ** attempt))
                    continue
            except Exception as e:
                logger.warning("Navigation exception for %s: %s", url, e)
                last_error = str(e)
                break  # Network errors already retried in execute(); don't double-retry
            else:
                last_error = result.get("message") or "navigation failed"

        # Persistent failure — graceful degradation (don't crash)
        logger.warning("Navigation to %s failed, continuing with current page", url)
        if self._client is not None:
            self._client._failed_urls.append(url)
        return last_error

    # -----------------------------------------------------------------------
    # Internal: wait-for-content (#5)
    # -----------------------------------------------------------------------

    async def _wait_for_content(self) -> None:
        """Poll get_page_metadata until page has a non-empty title, up to timeout."""
        if self._client is None or self._instance is None:
            return
        deadline = time.monotonic() + _WAIT_FOR_CONTENT_TIMEOUT
        while time.monotonic() < deadline:
            meta = await self._client.get_page_metadata(self._instance)
            if meta.get("title"):
                return
            await asyncio.sleep(_WAIT_FOR_CONTENT_INTERVAL)
        logger.debug("Wait-for-content timed out after %.1fs", _WAIT_FOR_CONTENT_TIMEOUT)

    # -----------------------------------------------------------------------
    # Internal: goback-from-homepage protection (#6)
    # -----------------------------------------------------------------------

    def _handle_goback(self, step_action_strs: list[str]) -> str:
        """Handle back action with homepage protection.

        Uses cached _current_page_url (updated after each step's screenshot)
        instead of making an HTTP call, to avoid latency on every back action.

        Returns:
            "execute" — proceed with normal back execution
            "skip" — skip this action (already at homepage)
            "terminate" — terminate the episode
        """
        # Check if we're at the homepage using cached URL
        at_homepage = False
        if self._homepage_url and self._current_page_url:
            at_homepage = self._current_page_url.rstrip("/") == self._homepage_url.rstrip("/")

        if not at_homepage:
            self._goback_from_homepage_count = 0
            return "execute"

        self._goback_from_homepage_count += 1

        if self._goback_from_homepage_count >= self._goback_terminate_threshold:
            step_action_strs.append("back() [terminated: repeated back from homepage]")
            logger.info("Terminating episode: %d consecutive back actions from homepage",
                        self._goback_from_homepage_count)
            return "terminate"

        if self._goback_from_homepage_count >= self._goback_skip_threshold:
            step_action_strs.append("back() [skipped: already at homepage]")
            logger.debug("Skipping back action: already at homepage (count=%d)",
                         self._goback_from_homepage_count)
            return "skip"

        # First back from homepage — execute normally (redirects to homepage)
        return "execute"

    # -----------------------------------------------------------------------
    # Internal: action translation
    # -----------------------------------------------------------------------

    def _translate_action(self, name: str, args: dict[str, Any]) -> dict | None:
        """Translate a CUA-Lite action to a WebGym HTTP command."""
        vw, vh = self._viewport

        def _px(coord: list[int | float]) -> tuple[int, int]:
            """Convert [0, 1000] normalized coords to clamped pixel coords."""
            return strict_norm_to_pixel(coord, vw, vh, clamp=True)

        if name == "click":
            coord = args.get("coordinate", [500, 500])
            x, y = _px(coord)
            self._last_click_x, self._last_click_y = x, y
            return {"click_coords": {"x": x, "y": y}}

        if name == "type":
            text = args.get("text", "")
            coord_raw = args.get("coordinate")
            if coord_raw is not None:
                x, y = _px(coord_raw)
            else:
                x, y = self._last_click_x, self._last_click_y
            # fill_coords does click + type + optional enter in one call. The
            # container's ``press_enter`` is THIS TRANSPORT's spelling of the
            # trailing newline canonical uses; strip it here so the flag the
            # container already honours stays the single carrier of "submit".
            # Only a trailing newline submits -- an interior one is part of the
            # value, and Playwright's ``keyboard.type`` presses Enter for it.
            press_enter = text.endswith("\n")
            if press_enter:
                text = text[:-1]
            return {"fill_coords": {
                "x": x,
                "y": y,
                "value": text,
                "press_enter": press_enter,
                "delete_existing": True,
            }}

        if name == "key":
            keys = project_model_keys(
                args.get("keys", []),
                action_name=name,
                backend="playwright",
            )
            # Keys arrive as canonical Lite key tokens: lowercase named keys plus
            # literal printable glyphs. project_model_keys(backend="playwright")
            # already resolved them to final Playwright wire key names, rejecting
            # an empty list — the container's keypress just runs them.
            return {"keypress": {"keys": keys}}

        if name == "scroll":
            coord = args.get("coordinate")
            direction = args.get("direction", "down")
            amount = int(args.get("amount", 3))
            # Convert scroll clicks to pixels (~100px per click)
            px_amount = amount * 100
            if coord:
                x, y = _px(coord)
                # Element-specific scroll (WebGym: hover_and_scroll_coords).
                # Pass `amount` (wheel clicks) so the element scroll honors the
                # agent's magnitude instead of a fixed one-notch step (matches the
                # page_down/page_up branch which scales by amount).
                return {"hover_and_scroll_coords": {
                    "x": x, "y": y, "direction": direction, "amount": amount
                }}
            else:
                # Full page scroll with pixel amount
                if direction in ("up", "left"):
                    return {"page_up": {"amount": px_amount}}
                else:
                    return {"page_down": {"amount": px_amount}}

        if name == "mouse_move":
            coord = args.get("coordinate", [500, 500])
            x, y = _px(coord)
            return {"hover_coords": {"x": x, "y": y}}

        if name == "drag":
            # WebGym has no native drag: it can only click the end point, so
            # `start_coordinate` (from cursor or otherwise) has no effect here —
            # nothing to align to "drag from current cursor".
            end = args["coordinate"]
            ex, ey = _px(end)
            return {"click_coords": {"x": ex, "y": ey}}

        if name == "back":
            return {"back": {}}

        if name == "goto":
            url = args.get("url", "")
            return {"visit_page": {"url": url}}

        if name in _WEBGYM_UNSUPPORTED_ACTIONS:
            return None  # no-op or not supported

        logger.warning("Unknown action: %s(%s), skipping", name, args)
        return None

    # -----------------------------------------------------------------------
    # Internal: screenshot (with blank detection + retry + size validation)
    # -----------------------------------------------------------------------

    async def _take_screenshot(self) -> bytes:
        """Capture the current page as PNG bytes, or void the trajectory.

        Raises ``TrueInfraFailure`` (retryable) when the env owed an observation
        and could not deliver one: a non-PNG body, a truncated frame, an
        undecodable PNG, or a raised exception, each after
        ``blank_screenshot_max_retries`` attempts. Those are env faults, not model
        faults, so the trajectory is void and re-run rather than scored 0.

        The ONE non-raising failure is an all-white page that captured fine: the
        site's own render, so the previous frame is re-served and the model is
        told (see ``_fallback_to_previous_screenshot``).

        Records the frame as :attr:`_last_frame` but NOT in
        :attr:`_screenshots`: a step captures one frame per action and only its
        last frame belongs in the per-step evaluator record, so the caller owns
        that append.
        """
        if self._client is None or self._instance is None:
            raise RuntimeError("Cannot take screenshot: WebGym client/instance is None")

        for attempt in range(_BLANK_SCREENSHOT_MAX_RETRIES):
            try:
                png_bytes = await self._client.screenshot(
                    self._instance,
                    mode="coordinates",
                    cursor=self._cursor,
                )

                # Validate PNG magic bytes
                if not png_bytes or png_bytes[:4] != b"\x89PNG":
                    logger.warning("Invalid screenshot data (%d bytes, starts with %r)",
                                   len(png_bytes), png_bytes[:20])
                    if attempt < _BLANK_SCREENSHOT_MAX_RETRIES - 1:
                        await asyncio.sleep(_BLANK_SCREENSHOT_WAIT)
                        continue
                    raise TrueInfraFailure(
                        f"webgym: screenshot API returned non-PNG data "
                        f"({len(png_bytes)} bytes) after "
                        f"{_BLANK_SCREENSHOT_MAX_RETRIES} attempts"
                    )

                # Screenshot file size validation (#8)
                if len(png_bytes) <= _MIN_SCREENSHOT_SIZE:
                    logger.warning("Screenshot too small (%d bytes), retrying (attempt %d/%d)",
                                   len(png_bytes), attempt + 1, _BLANK_SCREENSHOT_MAX_RETRIES)
                    if attempt < _BLANK_SCREENSHOT_MAX_RETRIES - 1:
                        await asyncio.sleep(_BLANK_SCREENSHOT_WAIT)
                        continue
                    raise TrueInfraFailure(
                        f"webgym: screenshot truncated ({len(png_bytes)} bytes) after "
                        f"{_BLANK_SCREENSHOT_MAX_RETRIES} attempts"
                    )

                # Blank screenshot detection (#1)
                try:
                    def _check_blank(data):
                        from PIL import Image
                        return is_white_image(Image.open(io.BytesIO(data)))
                    if await asyncio.to_thread(_check_blank, png_bytes):
                        logger.warning("Blank screenshot detected, retrying (attempt %d/%d)",
                                       attempt + 1, _BLANK_SCREENSHOT_MAX_RETRIES)
                        if attempt < _BLANK_SCREENSHOT_MAX_RETRIES - 1:
                            await asyncio.sleep(_BLANK_SCREENSHOT_WAIT)
                            continue
                        return await asyncio.to_thread(self._fallback_to_previous_screenshot)
                except ImportError:
                    pass  # PIL not available, skip blank detection
                except Exception as e:
                    # Corrupted PNG (UnidentifiedImageError, ValueError, etc.)
                    logger.warning("Failed to decode screenshot for blank detection: %s", e)
                    # Treat as invalid — retry
                    if attempt < _BLANK_SCREENSHOT_MAX_RETRIES - 1:
                        await asyncio.sleep(_BLANK_SCREENSHOT_WAIT)
                        continue
                    raise TrueInfraFailure(
                        f"webgym: screenshot PNG undecodable after "
                        f"{_BLANK_SCREENSHOT_MAX_RETRIES} attempts"
                    )

                self._last_frame = png_bytes
                return png_bytes

            except TrueInfraFailure:
                raise
            except Exception as e:
                logger.warning("Screenshot failed (attempt %d/%d): %s",
                               attempt + 1, _BLANK_SCREENSHOT_MAX_RETRIES, e)
                if attempt < _BLANK_SCREENSHOT_MAX_RETRIES - 1:
                    await asyncio.sleep(_BLANK_SCREENSHOT_WAIT)
                    continue
                raise TrueInfraFailure(
                    f"webgym: screenshot capture failed after "
                    f"{_BLANK_SCREENSHOT_MAX_RETRIES} attempts: {e}"
                ) from e

        raise TrueInfraFailure(
            f"webgym: screenshot capture exhausted "
            f"{_BLANK_SCREENSHOT_MAX_RETRIES} attempts without a frame"
        )

    def _fallback_to_previous_screenshot(self) -> bytes:
        """Re-serve the last frame for a page that legitimately rendered blank.

        ONE caller: the all-white detection above, after its retries. That is a
        property of the SITE, not of the env — the capture succeeded, the page
        really is blank — so the episode continues and the model is told so
        (see the ``_is_fallback_screenshot`` branch in ``step``, which explains
        the blank page and suggests navigating elsewhere).

        Capture FAILURES do not come here: a non-PNG body, a truncated frame, an
        undecodable PNG, or a raised exception are all ``TrueInfraFailure`` —
        the env owed an observation and did not produce one, so the trajectory
        is void and retryable rather than a model failure scored 0. Re-serving a
        stale frame there made the model act on a page it was no longer looking
        at, which read downstream as a repetitive-action loop.

        ``_is_fallback_screenshot`` also suppresses change detection: the last
        two frames are identical by construction here, and that is not evidence
        of an ineffective action.
        """
        self._is_fallback_screenshot = True
        if self._last_frame is not None:
            logger.info("Using previous screenshot as fallback")
            return self._last_frame
        # No previous screenshot (first call during reset) — generate blank placeholder
        logger.warning("No previous screenshot available, generating blank placeholder")
        from PIL import Image
        w, h = self._viewport
        img = Image.new("RGB", (w, h), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        blank_bytes = buf.getvalue()
        self._last_frame = blank_bytes
        return blank_bytes

    # -----------------------------------------------------------------------
    # Internal: evaluation
    # -----------------------------------------------------------------------

    def _build_eval_config(self) -> dict:
        """Build openai_config dict for the evaluator.

        Merge priority (later wins): config yaml < env vars < eval_config kwarg.
        API endpoint priority: WEBGYM_EVAL_BASE_URL > OPENAI_BASE_URL.

        Only the standard OpenAI/LiteLLM API surface is used (a plain OpenAI client
        against an OpenAI-compatible /v1 endpoint). Azure is reached the same way by
        pointing OPENAI_BASE_URL at Azure's OpenAI-compatible /openai/v1 surface
        (Bearer auth with the Azure key) — no Azure-specific code path.
        """
        # Start from the resolved eval_config (registration default + per-run overlay)
        config = dict(self._eval_config)

        # Env var overrides
        if os.environ.get("WEBGYM_EVAL_MODEL"):
            config["model"] = os.environ["WEBGYM_EVAL_MODEL"]
        config.setdefault("openai_api_key_env_var",
                          os.environ.get("WEBGYM_EVAL_API_KEY_VAR", "OPENAI_API_KEY"))

        base_url = os.environ.get("WEBGYM_EVAL_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        if base_url:
            config["base_url"] = base_url
        return config

    def _require_eval_deps(self) -> None:
        """Fail loud + fast if the VLM judge can't run (called at reset when
        ``skip_eval`` is off). The host-side judge is pip-installed by
        ``scripts/install.sh``; the API key is a deployment credential. We trust
        that protocol and import directly — no ``WEBGYM_REFERENCE_PATH`` sys.path
        fallback (it shadows the pinned, installed judge with a stray checkout)
        and no silent ``rewards=0.0`` degradation (which would masquerade a broken
        setup as a model that scored zero). To run without evaluation, pass
        ``skip_eval=True`` — the single explicit opt-out."""
        try:
            import webgym.models.evaluator  # noqa: F401
        except ImportError as e:
            raise EnvDepsMissingError(
                what="WebGym VLM judge (webgym package) not importable",
                install="uv run --no-sync bash lite/gym/envs/webgym/scripts/install.sh",
                see="lite/gym/envs/webgym/README.md",
            ) from e
        api_key_var = os.environ.get("WEBGYM_EVAL_API_KEY_VAR", "OPENAI_API_KEY")
        if not os.environ.get(api_key_var):
            raise EnvDepsMissingError(
                what=f"WebGym VLM judge needs {api_key_var} (+ OPENAI_BASE_URL for a proxy / Azure /openai/v1 surface)",
                install=f"export {api_key_var}=...   # or pass skip_eval=True to run without evaluation",
                see="lite/gym/envs/webgym/README.md",
            )

    @staticmethod
    def _write_temp_pngs(pngs: list[bytes]) -> list[str]:
        """Write screenshots to temp .png files and return their paths. Sync file
        I/O — call via asyncio.to_thread so it doesn't block the event loop (and
        freeze peer trajectories) during a terminal eval."""
        import tempfile
        paths: list[str] = []
        for b in pngs:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(b)
                paths.append(f.name)
        return paths

    @staticmethod
    def _judge_incomplete_reason(
        evaluation_text: Any, trajectory: list[dict]
    ) -> str | None:
        """Detect — from the judge's STRUCTURED return value — any reward-affecting
        sub-call that FAILED (e.g. a 429). The webgym Evaluator swallows such errors
        and returns a degraded reward=0, which would silently poison data; we surface
        them so the caller RAISES (→ trajectory ERR → retried) instead of trusting a
        non-verdict reward. Returns a reason string if incomplete, else None. Covers
        every path that influences the reward — read off the public return value, no
        global state / monkey-patching:

          - Criterion B / A / reference-answer: the Evaluator embeds
            "[Criterion X ...] Error: <exc>" / "[Reference Answer Evaluation] Error:
            <exc>" in ``evaluation_text`` on a call error (_JUDGE_CALL_ERROR_MARKERS).
            (We do NOT flag a *missing* reference-answer entry: the Evaluator only
            runs that block when reference_answer is set AND all rubrics passed —
            evaluator.py:320 — so its absence is a legitimate skip, not an error.)
          - Submission-image filtering: ``judge_single_image`` returns judgment=None
            on a failed call (defaulting submit=False), which silently drops a frame
            from Criterion B — detected as a non-last image step with judgment None.
        """
        responses = evaluation_text if isinstance(evaluation_text, list) else [evaluation_text]
        # Criterion B / A / reference-answer call error markers.
        for r in responses:
            s = str(r)
            if any(marker in s for marker in _JUDGE_CALL_ERROR_MARKERS):
                return f"criterion call errored: {s[:160]}"
        # Submission-image call error (judgment=None on a non-last image step).
        img_steps = [i for i, st in enumerate(trajectory) if st.get("observation") is not None]
        for i in img_steps[:-1]:  # last step is always force-included, never judged
            rw = trajectory[i].get("reward")
            if rw is not None and getattr(rw, "submission_judgment", "") is None:
                return f"submission-image judgment failed at step {i} (call errored)"
        return None

    async def _evaluate(self) -> tuple[float, dict]:
        """Evaluate the episode using WebGym's VLM evaluator.

        Returns 0.0 only on the two legitimate paths: skip_eval=True, or the
        agent gave no answer (truncated). A judge RUNTIME error is re-raised for
        rollout retry; the missing-key / uninstalled-judge case already fails
        loud at reset() via _require_eval_deps — never a silent 0.0.

        Matches reference async_webgym.py:1688-1725:
        - Agent answered -> full evaluation (get_verifiable_reward)
        - Agent didn't answer (truncated) -> only blocking detection, reward=0

        Returns:
            (reward, info_dict) where info_dict contains evaluation details.
        """
        if self._skip_eval:
            return 0.0, {}
        try:
            if self._agent_response:
                return await self._vlm_evaluate()
            else:
                # Agent truncated without answering — only check blocking
                return await self._blocking_only_evaluate()
        except Exception as e:
            # A judge RUNTIME error (rate-limit / timeout / content-filter / network) must NOT be
            # silently scored 0.0 — that would drop real successes as if they failed and quietly
            # corrupt the collected data. Re-raise so the rollout marks the task errored and retries
            # it (--max-attempts). The missing-key / setup case is already caught fail-fast in
            # reset() via _require_eval_deps(); the legitimate "agent gave no answer" path returns 0
            # normally (not via this except).
            logger.error("WebGym VLM evaluation errored (propagating for retry): %s", e)
            raise

    async def _blocking_only_evaluate(self) -> tuple[float, dict]:
        """Check if the website blocked the agent (no-answer path).

        When the agent runs out of steps without providing an answer,
        we skip the full Criterion B/A evaluation (which would fail anyway
        on an empty response) and only run blocking detection.
        """
        from webgym.data.components import (
            Action as WGAction,
        )
        from webgym.data.components import (
            Observation as WGObservation,
        )
        from webgym.data.components import (
            Response as WGResponse,
        )
        from webgym.data.components import (
            Reward as WGReward,
        )
        from webgym.data.components import (
            Task as WGTask,
        )

        # Task metadata field mapping (#20): try reference_answer first, then definite_answer
        reference_answer = self._task.get("reference_answer") or self._task.get("definite_answer", "")

        task = WGTask(
            task_name=self._task.get("task_name", ""),
            domain=self._task.get("domain", ""),
            subdomain=self._task.get("subdomain", ""),
            website=self._task.get("website", ""),
            difficulty=self._task.get("difficulty", 2),
            evaluator_reference=self._task["evaluator_reference"],
            reference_answer=reference_answer,
        )

        # Build minimal trajectory for blocking detection.
        temp_files = await asyncio.to_thread(self._write_temp_pngs, self._screenshots)
        trajectory: list[dict] = []
        for i, path in enumerate(temp_files):
            action_str = self._actions[i] if i < len(self._actions) else "noop"
            obs_text = self._observations[i] if i < len(self._observations) else ""
            trajectory.append({
                "observation": WGObservation(task=task, image_path=path, ac_tree="",
                                             page_metadata={"title": "", "url": ""}),
                "action": WGAction(action={"key": "noop", "arguments": {}},
                                   action_string=action_str),
                "response": WGResponse(raw_response=action_str,
                                       answering_tokens={"observation": obs_text, "action": action_str},
                                       raw_prompt=""),
                "reward": WGReward(reward=0, evaluation="", is_blocked=False, submit=True),
            })

        if not trajectory:
            return 0.0, {}

        class _SimpleConversationBuilder:
            def summarize_trajectory(self, traj):
                lines = []
                for j, step in enumerate(traj):
                    act = step.get("action")
                    resp = step.get("response")
                    act_str = act.action_string if act else "noop"
                    obs = resp.answering_tokens.get("observation", "") if resp else ""
                    lines.append(f"Step {j}: {act_str} | Observation: {obs}")
                return "\n".join(lines) if lines else "No trajectory available"
            def build_conversation(self, task_str, traj, current_observation, **kwargs):
                return []
            def get_conversation_type(self):
                return "multi-turn"

        evaluator = _get_judge_evaluator_cls()(
            openai_config=self._build_eval_config(),
            conversation_builder=_SimpleConversationBuilder(),
            max_retries=1,
            verbose=False,
        )

        try:
            # The webgym Evaluator is SYNCHRONOUS and fires many OpenAI vision
            # calls — run it off the event loop (to_thread) so it can't freeze
            # peers' browser steps, and gate concurrent judges (sem_judge) so they
            # don't storm the deployment into 429s. See _SEM_JUDGE.
            WebGymClient._init_semaphores()
            async with WebGymClient._sem_judge:
                is_blocked = await asyncio.to_thread(evaluator.check_if_blocked, trajectory)
            # This is the NO-ANSWER path: reward is 0 regardless (the agent submitted
            # nothing — a genuine "not completed" verdict, not a judge failure). The
            # blocking check only sets an info LABEL, not the reward, so a failed
            # blocking call cannot poison the reward; no throw needed here. (The
            # reward-bearing path, _vlm_evaluate, is guarded by _judge_incomplete_reason.)
            logger.info("Blocking-only eval (no agent response): is_blocked=%s", is_blocked)

            if is_blocked:
                eval_text = "Website blocked the agent - detected by verifier"
            else:
                eval_text = "Task incomplete - no answer provided"

            info = {
                "agent_response": None,
                "actions": list(self._actions),
                "num_steps": len(self._actions),
                "blocking": {"is_blocked": is_blocked},
                "eval_skipped": True,
                "eval_skip_reason": "no_agent_response",
                "eval_text": eval_text,
                "eval_model": self._eval_config.get("model", "gpt-4.1"),
            }
            return 0.0, info
        finally:
            for path in temp_files:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    async def _vlm_evaluate(self) -> tuple[float, dict]:
        """Run WebGym's VLM-based evaluation, aligned with the reference.

        Uses evaluator.get_verifiable_reward(trajectory) which returns
        (reward, evaluation_text, is_blocked). Trajectory must contain
        observation, action, response, and reward objects per step.

        Returns:
            (reward, info_dict) with detailed intermediate results for debugging.
        """
        from webgym.data.components import (
            Action as WGAction,
        )
        from webgym.data.components import (
            Observation as WGObservation,
        )
        from webgym.data.components import (
            Response as WGResponse,
        )
        from webgym.data.components import (
            Reward as WGReward,
        )
        from webgym.data.components import (
            Task as WGTask,
        )

        # Build Task object
        evaluator_reference = _normalize_evaluator_reference(
            self._task["evaluator_reference"]
        )
        # Task metadata field mapping (#20): try reference_answer first, then definite_answer
        reference_answer = self._task.get("reference_answer") or self._task.get("definite_answer", "")

        task = WGTask(
            task_name=self._task.get("task_name", ""),
            domain=self._task.get("domain", ""),
            subdomain=self._task.get("subdomain", ""),
            website=self._task.get("website", ""),
            difficulty=self._task.get("difficulty", 2),
            evaluator_reference=evaluator_reference,
            reference_answer=reference_answer,
        )

        # Build trajectory with all 4 required fields per step:
        #   observation, action, response, reward
        # The evaluator accesses:
        #   trajectory[-1]['action'].action['arguments']['content']  (agent answer)
        #   trajectory[-1]['observation'].task                       (Task object)
        #   conversation_builder.summarize_trajectory(trajectory)    (text summary)
        temp_files = await asyncio.to_thread(self._write_temp_pngs, self._screenshots)
        trajectory: list[dict] = []

        for i, path in enumerate(temp_files):
            obs = WGObservation(
                task=task,
                image_path=path,
                ac_tree="",
                page_metadata={"title": "", "url": ""},
            )

            action_str = self._actions[i] if i < len(self._actions) else "noop"
            # The evaluator reads the agent answer from the last step's action:
            #   trajectory[-1]['action'].action['arguments']['content']
            action = WGAction(
                action={"key": "answer", "arguments": {"content": self._agent_response or ""}},
                action_string=action_str,
            )

            obs_text = self._observations[i] if i < len(self._observations) else ""
            response = WGResponse(
                raw_response=action_str,
                answering_tokens={"observation": obs_text, "action": action_str},
                raw_prompt="",
            )

            reward = WGReward(reward=0, evaluation="", is_blocked=False, submit=True)

            trajectory.append({
                "observation": obs,
                "action": action,
                "response": response,
                "reward": reward,
            })

        if not trajectory:
            return 0.0, {}

        # Simple conversation_builder that implements summarize_trajectory()
        class _SimpleConversationBuilder:
            def summarize_trajectory(self, traj: list[dict]) -> str:
                lines = []
                for j, step in enumerate(traj):
                    act = step.get("action")
                    resp = step.get("response")
                    act_str = act.action_string if act else "noop"
                    obs = resp.answering_tokens.get("observation", "") if resp else ""
                    lines.append(f"Step {j}: {act_str} | Observation: {obs}")
                return "\n".join(lines) if lines else "No trajectory available"

            def build_conversation(self, task_str, traj, current_observation, **kwargs):
                return []

            def get_conversation_type(self):
                return "multi-turn"

        # Run evaluator
        evaluator = _get_judge_evaluator_cls()(
            openai_config=self._build_eval_config(),
            conversation_builder=_SimpleConversationBuilder(),
            max_retries=1,
            verbose=False,
        )

        try:
            # The webgym Evaluator is SYNCHRONOUS and fires ~20+ OpenAI vision
            # calls — run it off the event loop (to_thread) so a 60-80s judge can't
            # freeze every peer trajectory's browser steps, and gate concurrent
            # judges (sem_judge) so they don't storm the deployment into 429s whose
            # retry-backoff would stack a terminal step past step_timeout. See _SEM_JUDGE.
            WebGymClient._init_semaphores()
            async with WebGymClient._sem_judge:
                reward_value, evaluation_text, is_blocked = await asyncio.to_thread(
                    evaluator.get_verifiable_reward, trajectory
                )
            # No silent rewards: the Evaluator swallows a failed sub-call (e.g. 429)
            # and returns a degraded reward=0. _judge_incomplete_reason reads that off
            # the structured result (criterion markers / missing reference verdict /
            # None submission judgments); if any reward-affecting call failed, the
            # reward is NOT a real verdict — raise so the trajectory ERRs and is
            # retried (--max-attempts) rather than poisoning the dataset.
            reason = self._judge_incomplete_reason(evaluation_text, trajectory)
            if reason is not None:
                raise RuntimeError(f"VLM judge did not complete — refusing to score: {reason}")
            logger.info(
                "VLM eval: reward=%d, is_blocked=%s, eval=%s",
                reward_value, is_blocked, str(evaluation_text)[:200],
            )

            # --- Collect per-step image filtering results ---
            # (evaluator modifies trajectory in-place during judge_submission_images)
            image_filter_results = []
            num_submitted = 0
            for i, step in enumerate(trajectory):
                rw = step.get("reward")
                if rw is not None and hasattr(rw, "submit"):
                    submitted = bool(rw.submit)
                    if submitted:
                        num_submitted += 1
                    image_filter_results.append({
                        "step": i,
                        "submit": submitted,
                        "judgment": getattr(rw, "submission_judgment", None),
                    })

            # --- Parse evaluation_text into structured per-criterion results ---
            criterion_b_result = None
            criterion_a_results = []
            reference_answer_result = None
            raw_responses = evaluation_text if isinstance(evaluation_text, list) else [evaluation_text]

            for resp_text in raw_responses:
                resp_str = str(resp_text)
                # Match evaluator's _extract_verification_response logic exactly:
                if "Verdict:" in resp_str:
                    verdict_part = resp_str.split("Verdict:")[-1].split("\n")[0]
                    passed = "NOT SUCCESS" not in verdict_part
                else:
                    passed = "NOT SUCCESS" not in resp_str

                if resp_str.startswith("[Criterion B"):
                    criterion_b_result = {"passed": passed, "response": resp_str}
                elif resp_str.startswith("[Criterion A"):
                    criterion_a_results.append({"passed": passed, "response": resp_str})
                elif resp_str.startswith("[Reference Answer"):
                    reference_answer_result = {"passed": passed, "response": resp_str}

            # --- Extract rubrics ---
            rubrics = self._task["evaluator_reference"]
            rubric_texts = [
                r["description"] if isinstance(r, dict) else r for r in rubrics
            ]

            info = {
                # Task metadata
                "task": {
                    "task_name": self._task.get("task_name", ""),
                    "website": self._task.get("website", ""),
                    "difficulty": self._task.get("difficulty", ""),
                    "rubrics": rubric_texts,
                    "reference_answer": reference_answer,
                },
                # Agent output
                "agent_response": self._agent_response,
                "actions": list(self._actions),
                "num_steps": len(self._actions),
                # Step 0: Image filtering
                "image_filtering": {
                    "num_screenshots": len(self._screenshots),
                    "num_submitted": num_submitted,
                    "per_step": image_filter_results,
                },
                # Step 1: Blocking detection
                "blocking": {
                    "is_blocked": is_blocked,
                },
                # Step 2: Criterion B (anti-hallucination)
                "criterion_b": criterion_b_result,
                # Step 3: Criterion A (per-fact verification)
                # Empty list means skipped (Criterion B failed) or no rubrics.
                # Shorter than rubrics means early-exited on first failure.
                "criterion_a": criterion_a_results,
                "criterion_a_skipped": (
                    criterion_b_result is not None
                    and not criterion_b_result["passed"]
                    and len(criterion_a_results) == 0
                ),
                # Step 4: Reference answer check (None if not applicable)
                "reference_answer_check": reference_answer_result,
                # Eval config
                "eval_model": self._eval_config.get("model", "gpt-4.1"),
            }
            return float(reward_value), info
        finally:
            # Clean up temp screenshot files
            for path in temp_files:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    # -----------------------------------------------------------------------
    # Internal: cleanup
    # -----------------------------------------------------------------------

    async def _safe_release(self) -> None:
        """Release the instance back to the pool, ignoring errors.

        Uses asyncio.shield to prevent CancelledError from aborting the
        release — this is critical when StepTimeoutWrapper cancels step/reset
        and the agent's finally block calls env.close(). Without shield, the
        release HTTP call gets cancelled and the instance leaks.
        If shield also fails (e.g. event loop shutting down), falls back to
        a synchronous release in a daemon thread.
        """
        if self._client is not None and self._instance is not None:
            instance = self._instance
            self._instance = None  # Clear first to prevent double-release
            try:
                await asyncio.shield(self._client.reset_instance(instance))
                logger.debug("Released instance %s", instance.get("instance_id"))
            except asyncio.CancelledError:
                # shield was cancelled — fall back to sync release in background thread
                logger.warning("Release cancelled, spawning background cleanup for %s",
                               instance.get("instance_id"))
                self._background_release(instance)
            except Exception as e:
                logger.warning("Failed to release instance %s: %s",
                               instance.get("instance_id"), e)

    def _background_release(self, instance: dict) -> None:
        """Synchronous fallback release in a daemon thread."""
        import threading

        import requests
        url = self._master_url
        api_key = self._api_key

        def _release():
            try:
                requests.post(
                    f"{url}/reset", params=instance,
                    headers={"x-api-key": api_key}, timeout=30,
                )
                logger.info("Background release succeeded for %s", instance.get("instance_id"))
            except Exception as e:
                logger.warning("Background release failed for %s: %s",
                               instance.get("instance_id"), e)

        t = threading.Thread(target=_release, daemon=True)
        t.start()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _normalize_evaluator_reference(raw: Any) -> list[dict]:
    """Normalize evaluator_reference to list of {"description": str} dicts.

    Matches reference async_webgym.py:251-277 normalization:
    - str -> [{"description": str}]
    - list of dicts with 'facts' field -> flatten fact groups into individual items
    - list of dicts with 'description' -> keep as-is
    - list of strings -> convert to list of dicts
    - other -> stringify
    """
    import json as _json

    if isinstance(raw, str):
        # Could be a JSON string or plain text
        try:
            parsed = _json.loads(raw)
            if isinstance(parsed, list):
                return _normalize_evaluator_reference(parsed)
        except (ValueError, TypeError):
            pass
        return [{"description": raw, "difficulty": 1}]

    if isinstance(raw, list):
        if not raw:
            return []
        if isinstance(raw[0], dict):
            # Fact-group format: flatten into individual rubric items
            if "facts" in raw[0]:
                flattened = []
                for group in raw:
                    group_desc = group.get("description", "")
                    group_id = group.get("id", "")
                    for fact in group.get("facts", []):
                        fact_with_context = f"[Group {group_id}: {group_desc}] {fact}"
                        flattened.append({"description": fact_with_context, "difficulty": 1})
                return flattened
            # Already in correct format (list of dicts with 'description')
            return raw
        if isinstance(raw[0], str):
            return [{"description": s, "difficulty": 1} for s in raw]

    return [{"description": str(raw), "difficulty": 1}]


def _ensure_data() -> bool:
    """Ensure the HuggingFace dataset is available, auto-downloading if needed."""
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(_HF_REPO_ID, repo_type="dataset", local_files_only=True)
        return True
    except Exception:
        pass

    try:
        logger.info("WebGym data not cached. Downloading from HuggingFace (%s) ...", _HF_REPO_ID)
        path = snapshot_download(_HF_REPO_ID, repo_type="dataset")
        logger.info("WebGym data cached at: %s", path)
        return True
    except Exception as e:
        warnings.warn(
            f"Failed to download WebGym data: {e}\n"
            "  Tasks will not be registered.\n"
            "  See: lite/gym/envs/webgym/README.md",
            stacklevel=3,
        )
        return False

#: Set once :func:`_load_and_register` has registered the catalog, so the lazy
#: hook (``WebGymContainerServices.register_tasks``) is idempotent across the
#: catalog-probe and make() paths.
_tasks_registered = False


def _load_and_register() -> None:
    """Load tasks from HuggingFace and register them in the gym registry.

    Lazy: fired by ``WebGymContainerServices.register_tasks`` on the first catalog
    probe (``task_ids``) or ``make()`` — NOT at module import. ``load_dataset`` is
    ~18s, so paying it at import would tax every webgym startup (rollout / eval /
    env-server) AND every pytest collection. Idempotent; re-tries if the dataset
    isn't present yet (mirrors the static-dataset envs)."""
    global _tasks_registered
    if _tasks_registered:
        return
    if not _ensure_data():
        return

    # Env-wide make defaults declared once here, sourced from default.yaml
    # make_kwargs. Per-task register() kwargs / gym.make() still override.
    registry.set_env_make_kwargs("webgym", CFG.make_kwargs)

    from datasets import load_dataset

    n_registered = 0
    for hf_split, cua_split in [("train", "train"), ("test", "eval")]:
        ds = load_dataset(_HF_REPO_ID, split=hf_split)
        for i, row in enumerate(ds):
            task_id = str(row.get("task_id", i))

            task_data = dict(row)
            # evaluator_reference is REQUIRED upstream (webgym Task.__init__ takes it
            # positionally, no default; the judge reads it unguarded). Index directly
            # so a malformed dataset row fails loud here at the load boundary rather
            # than silently scoring 0 later with empty rubrics.
            task_data["evaluator_reference"] = _normalize_evaluator_reference(
                task_data["evaluator_reference"]
            )

            register(
                key=f"webgym@{task_id}",
                entry_point=lambda *, _task=task_data, _split=cua_split, **kw: WebGymEnv(task=_task, split=_split, **kw),
                split=cua_split,
                # Same-source contract: registered copy == the env's
                # builder output. task_id is auto-injected by registry.register();
                # env_id + task_id are injected into persisted parquet by the logger.
                metadata=WebGymEnv._task_metadata(task_data),
                # step_timeout is an env-wide default (CFG.make_kwargs, applied via
                # set_env_make_kwargs above) — not repeated per task here. A per-task
                # or per-subset override would still go here as a register() kwarg.
            )
            n_registered += 1

    _tasks_registered = True
    logger.info("Registered %d webgym tasks", n_registered)


# ---------------------------------------------------------------------------
# Container lifecycle helpers (relocated from the former host process-pool
# section; reused by the container Services below)
# ---------------------------------------------------------------------------
#
# webgym is container-only: the whole OmniBoxes tree runs
# inside ONE ``cua-lite/webgym`` container per env-server, fronted by the host
# ``WebGymEnv``/``WebGymClient`` HTTP client. See
# :func:`lite.gym.registry.ensure_services` for the lifecycle protocol.

import urllib.error  # noqa: E402
import urllib.request  # noqa: E402

from lite.gym.utils.backend.docker import docker_rm_f, docker_run  # noqa: E402
from lite.gym.utils.backend.freshness import image_for  # noqa: E402
from lite.gym.utils.backend.wait import wait_until_ready  # noqa: E402
from lite.gym.utils.config.naming import container_name  # noqa: E402

# Env-ids already brought up this process; ``WebGymContainerServices.ensure``
# is idempotent against it (one shared container per env-server).
_wg_services_started: set[str] = set()


# ── Containerized variant ─────────────────────────────────
# ONE cua-lite/webgym container per env-server (SINGLETON — the browsergym-WA shape):
# the whole OmniBoxes tree runs inside, only the master port is published, and
# resource management is one atomic `docker rm -f`. The host WebGymEnv/WebGymClient
# are UNCHANGED — they just point WEBGYM_MASTER_URL at the container's published
# port. This is the ONLY backend (the host process-pool path was removed).
_WEBGYM_IMAGE = image_for("webgym")


def _webgym_container_name(server_port: int | None = None) -> str:
    """``webgym-<server_port>`` (or ``webgym-d<pid>`` in direct mode) — the
    shared-singleton scheme; see :func:`lite.gym.utils.config.naming.container_name`."""
    return container_name("webgym", server_port)


def _webgym_master_capacity_ok(url: str) -> bool:
    """True iff the OmniBoxes master at ``url`` serves AND total capacity>0 — the
    L3 check (a 0-instance shell master answers ``/info`` but can't serve)."""
    if not url:
        return False
    import json as _json
    try:
        req = urllib.request.Request(
            url.rstrip("/") + "/info",
            headers={"x-api-key": os.environ.get("WEBGYM_API_KEY", "default_key")},
        )
        with urllib.request.urlopen(req, timeout=3) as r:
            d = _json.loads(r.read())
        nodes = d.get("nodes") or []
        return bool(nodes) and sum(int(n.get("capacity", 0)) for n in nodes) > 0
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return False


_WEBGYM_RM_TIMEOUT_S = float(os.environ.get("CUA_LITE_WEBGYM_RM_TIMEOUT_S", str(CFG.server_kwargs["rm_timeout_s"])))


def _webgym_docker_rm_f(name: str) -> int:
    """``docker rm -f <name>`` bounded by ``_WEBGYM_RM_TIMEOUT_S``; see
    :func:`lite.gym.utils.backend.docker.docker_rm_f`."""
    return docker_rm_f(name, timeout=_WEBGYM_RM_TIMEOUT_S, label="webgym")


class WebGymContainerServices(SingletonContainerServices):
    """Container variant of webgym's env-server capability.

    SINGLETON (fixed shared container per env-server): ``ensure`` brings up ONE
    ``cua-lite/webgym`` container (whole OmniBoxes pool inside) and points
    ``WEBGYM_MASTER_URL`` at its published port. ``shutdown``/``reap(boot=True)`` (both
    inherited from :class:`SingletonContainerServices`) ``docker rm -f`` this server's
    container, named by :meth:`container_name`, scoped by ``server_port`` (→ ``d<pid>`` in
    direct mode)."""

    rm_timeout_s = _WEBGYM_RM_TIMEOUT_S
    rm_label = "webgym"

    def container_name(self, scope) -> str:
        return _webgym_container_name(scope.server_port)

    def register_tasks(self, env_id: str) -> None:
        # Register the task catalog (HF load) lazily — cheap relative to ensure(),
        # and fired by a bare task_ids() probe WITHOUT booting the container.
        _load_and_register()

    def ensure(self, env_id: str) -> None:
        if env_id in _wg_services_started:
            return
        name = _webgym_container_name()
        # WEBGYM_MASTER_URL is an output of this method. Do not treat an
        # ambient value from a previous pytest/process/container lifetime as a
        # reusable backend: that bypasses docker_run's image freshness gate and
        # can silently hit an old OmniBoxes tree after patch changes.
        os.environ.pop("WEBGYM_MASTER_URL", None)
        _webgym_docker_rm_f(name)                    # clear a dead-but-present same-name container
        from lite.gym.envs.webgym.pool_sizing import derive_pool_size_from_env
        from lite.gym.utils.backend.ports import allocate_ports
        # Published master port from webgym's documented fallback band 7700-7799
        # (backend.ports port-map), NOT the default 20000-20999 cb-noVNC band.
        # flock-safe.
        host_port = allocate_ports(n=1, range_start=7700, range_end=7800)[0]
        m = derive_pool_size_from_env().instances
        mem = f"{max(8, m * 6 // 10)}g"              # generous OOM headroom
        # Viewport is POOL-LEVEL: the pool launches all browsers once at boot, so the
        # configured size is forwarded to the container as -e WEBGYM_VIEWPORT=WxH (read
        # by docker/patches/playwright_instance.py) rather than set per-episode. The
        # host then adopts the container-reported size at reset().
        vw, vh = _VIEWPORT
        logger.info("webgym container: docker run %s (M=%d, host_port=%d, mem=%s, viewport=%dx%d)",
                    name, m, host_port, mem, vw, vh)
        docker_run(
            name, _WEBGYM_IMAGE, mem=mem, port=(host_port, 7000),
            env={
                "WEBGYM_API_KEY": os.environ.get("WEBGYM_API_KEY", "default_key"),
                "WEBGYM_INSTANCES": str(m),
                "WEBGYM_VIEWPORT": f"{vw}x{vh}",
            },
        )
        new_url = f"http://localhost:{host_port}"
        os.environ["WEBGYM_MASTER_URL"] = new_url    # WebGymEnv reads this at construction
        # M browsers can take a while to warm.
        if wait_until_ready(lambda: _webgym_master_capacity_ok(new_url), timeout=300, interval=2):
            _wg_services_started.add(env_id)
            logger.info("webgym container ready: %s", new_url)
            return
        raise EnvDepsMissingError(
            what=f"webgym container {name} started but capacity>0 not reached within 300s",
            install="check `docker logs %s` and lite/gym/envs/webgym/scripts/install.sh" % name,
            see="lite/gym/envs/webgym/README.md",
        )

    def health(self, env_id: str) -> None:
        if not _webgym_master_capacity_ok(os.environ.get("WEBGYM_MASTER_URL", "")):
            raise EnvDepsMissingError(
                what="webgym container not serving (capacity 0 / unreachable)",
                install="uv run --no-sync bash lite/gym/envs/webgym/scripts/install.sh",
                see="lite/gym/envs/webgym/README.md",
            )

    def _evict_cache(self, env_id: str) -> None:
        _wg_services_started.discard(env_id)
        os.environ.pop("WEBGYM_MASTER_URL", None)
        invalidate_services(env_id)

    def reap(self, env_id: str, scope, in_use: set[str], *, boot: bool = False) -> int:
        try:
            return super().reap(env_id, scope, in_use, boot=boot)
        finally:
            if boot:
                self._evict_cache(env_id)

    def shutdown(self, env_id: str, scope) -> None:
        try:
            super().shutdown(env_id, scope)
        finally:
            self._evict_cache(env_id)

    # live_ids inherited from SingletonContainerServices: SINGLETON backends have
    # no per-instance reconcile view.


register_services("webgym", WebGymContainerServices())
from lite.gym.services import BackendFamily, register_family  # noqa: E402

register_family("webgym", BackendFamily.SINGLETON)
