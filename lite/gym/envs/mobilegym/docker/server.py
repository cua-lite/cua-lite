"""In-container FastAPI RPC server for cua-lite/mobilegym.

Runs the **full mobilegym task lifecycle** in-container — the logic was ported
from the former host env (removed): its reset/step/close/_evaluate/
_evaluate_grounded/_translate_action methods. That lifecycle uses ONLY bench_env
types (``Action``/``ActionType``/``JudgeInput``) + pure Python, and the
container already vendors bench_env — so the whole loop (task setup → action
translate → step → evaluate → teardown) lives here and the RPC wire carries ONLY
primitives (screenshots_b64:list[str], reward:float|None, terminated/truncated:bool,
instruction:str, executed:list). The host becomes a thin JSON client with ZERO
bench_env import (NO pickle, unlike androidworld).

Also hosts the Chromium browser POOL that used to live in the former host env's
``_browsers`` module state: one shared Chromium per ``_CONTEXTS_PER_BROWSER``
contexts, up to ``_MAX_BROWSERS`` browsers, with an idle-reaper (the
load-bearing every-tick idle compaction, re-homed in-container ⚠️).

Endpoints (JSON in/out, no pickle):
  GET  /healthz                     → {ok, browsers, instances, contexts_in_use}
  POST /reset  {task_id, seed, eval_mode, reward_shaping, physical_size, dpr,
               delay, max_steps?, app_ids?}
                                    → {instance_id, screenshot_b64, instruction, max_steps}
  POST /step   {instance_id, actions}
                                    → {screenshots_b64, reward, terminated,
                                       truncated, executed, action_errors}
                                    screenshots_b64: ONE frame per executed action, in action
                                      order. ``/step`` returns no singular spelling.
                                    action_errors: [{index, kind, name, error, message, call_id?}]
                                      — per-action MODEL feedback the host pairs back to the
                                      originating call_id (additive key; an older host ignores it).
  POST /close  {instance_id}        → {ok}

Run by entrypoint.sh: uvicorn server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import base64
import logging
import math
import os
import random
import time
import uuid
from dataclasses import dataclass
from typing import Any

import uvicorn
from bench_env.env.base import Action, ActionType, Observation
from bench_env.env.mobile_gym import MobileGymEnv
from bench_env.task.base import BaseTask
from bench_env.task.judge import JudgeInput, JudgeResult
from bench_env.task.registry import TaskRegistry
from fastapi import FastAPI, HTTPException, Request
from playwright.async_api import async_playwright

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mobilegym-server")


# ---------------------------------------------------------------------------
# Config. The yaml (configs/default.yaml server_kwargs) is the SINGLE SOURCE OF
# TRUTH: the host reads it via env_config.load -> CFG and passes the values in as
# `-e MOBILEGYM_*` at `docker run` (MobileGymContainerServices.ensure). This
# container has no access to lite.gym/env_config, so os.environ is the only channel
# in; the literal defaults below are PURE FALLBACKS (kept in sync with the yaml
# defaults) used only if a value wasn't passed — never an independent config source.
# ---------------------------------------------------------------------------
_VITE_URL = f"http://localhost:{os.environ.get('MOBILEGYM_VITE_PORT', '4173')}"
_MAX_BROWSERS = int(os.environ.get("MOBILEGYM_MAX_BROWSERS", "16"))
_CONTEXTS_PER_BROWSER = int(os.environ.get("MOBILEGYM_CONTEXTS_PER_BROWSER", "8"))
_IDLE_BROWSER_TTL_S = float(os.environ.get("MOBILEGYM_IDLE_BROWSER_TTL_S", "300"))
# Bound a wedged Chromium launch (defends against the documented
# inotify-exhaustion hang; the host forwards its yaml value via
# MOBILEGYM_LAUNCH_TIMEOUT_S). Launch runs outside the pool lock so a hang can't
# deadlock the pool, but without this it would stall one /reset until the host
# client's 180s urlopen timeout.
_LAUNCH_TIMEOUT_S = float(os.environ.get("MOBILEGYM_LAUNCH_TIMEOUT_S", "60"))
_DIFFICULTY_MAX_STEPS = {"L1": 15, "L2": 30, "L3": 45, "L4": 60}
#: Ceiling for the shaped branch (reward_shaping=true). Any value < 1.0 keeps the
#: argmax on genuine success; 0.5 leaves a wide margin over the best partial run.
_SHAPED_REWARD_CEILING = 0.5

#: Model-caused action errors — mirrors
#: ``lite/gym/utils/feedback/errors.py::MODEL_ACTION_ERROR_TYPES`` (this
#: container cannot import ``lite.gym``).
#: These become PER-CALL agent feedback in the ``/step`` response. Everything
#: else (playwright/browser death, timeouts, RPC loss) is INFRA and keeps
#: propagating out of ``/step`` as a 500 so the rollout errors instead of
#: silently telling the model its action was bad.
_MODEL_ACTION_ERROR_TYPES = (ValueError, TypeError, IndexError, KeyError)
_DEFAULT_MODEL_DURATION_CAP_SECONDS = 30.0
_MODEL_DURATION_CAPS_SECONDS = {
    "wait": 30.0,
    "hold_key": 5.0,
    "long_press": 5.0,
    "swipe": 5.0,
    "drag": 5.0,
}


class _UnsupportedAction(Exception):
    """The action has no MobileGym backend mapping (agent-visible, not infra)."""


app = FastAPI(title="cua-lite mobilegym in-container server")


def _resolve_task_max_steps(task_cls: type | None) -> int:
    """Text-mode base budget for a task class — explicit ``cls.max_steps`` else
    difficulty default else 30 (same rule the host manifest dump applies)."""
    raw = getattr(task_cls, "max_steps", None)
    if raw is not None:
        return int(raw)
    return _DIFFICULTY_MAX_STEPS.get(getattr(task_cls, "difficulty", "L1"), 30)


# ---------------------------------------------------------------------------
# Browser pool
# ---------------------------------------------------------------------------
_pw = None                       # shared Playwright
_pw_lock = asyncio.Lock()
_browsers: list[dict] = []       # [{"browser", "n_ctx", "last_used"}]
_pool_lock = asyncio.Lock()


async def _ensure_pw():
    global _pw
    if _pw is None:
        async with _pw_lock:
            if _pw is None:
                _pw = await async_playwright().start()
    return _pw


async def _acquire_browser() -> dict:
    """Reserve a browser slot. Reuse one with spare context room (and a live
    Chromium); else reserve a NEW slot under the cap and launch OUTSIDE the
    lock; else 503 (pool full — host retries).

    Audit fixes vs the prior version:
      (a) ``chromium.launch()`` runs OUTSIDE ``_pool_lock`` (reserve a
          placeholder slot under the lock, launch after release) so a slow
          launch doesn't block every other acquire/release.
      (b) zombie-slot eviction — a dead browser (``is_connected()`` False with
          no in-flight contexts) is dropped instead of being handed out.
    """
    while True:
        async with _pool_lock:
            # Evict dead, idle slots so we never hand out a dead browser.
            for b in list(_browsers):
                if b.get("browser") is not None and b["n_ctx"] == 0:
                    try:
                        if not b["browser"].is_connected():
                            _browsers.remove(b)
                            logger.warning("evicted dead idle browser slot")
                    except Exception:
                        _browsers.remove(b)
            # Reuse a live browser with spare room.
            for b in _browsers:
                if b.get("browser") is None:
                    continue  # another acquire is mid-launch on this slot
                if b["n_ctx"] < _CONTEXTS_PER_BROWSER:
                    try:
                        alive = b["browser"].is_connected()
                    except Exception:
                        alive = False
                    if not alive:
                        continue
                    b["n_ctx"] += 1
                    b["last_used"] = time.monotonic()
                    return b
            # Reserve a new slot (placeholder) if under cap, launch outside lock.
            if len(_browsers) < _MAX_BROWSERS:
                slot: dict = {"browser": None, "n_ctx": 1, "last_used": time.monotonic()}
                _browsers.append(slot)
                break
            raise HTTPException(status_code=503, detail="mobilegym pool full")

    # --- launch OUTSIDE the pool lock (audit fix a) ---
    try:
        pw = await _ensure_pw()
        launch_args = MobileGymEnv.get_launch_args(True)  # headless
        args = list(launch_args.get("args") or [])
        if "--no-sandbox" not in args:  # container requirement (audit)
            args.append("--no-sandbox")
        launch_args["args"] = args
        browser = await asyncio.wait_for(
            pw.chromium.launch(**launch_args), timeout=_LAUNCH_TIMEOUT_S,
        )
    except Exception:
        async with _pool_lock:
            if slot in _browsers:
                _browsers.remove(slot)
        raise
    async with _pool_lock:
        slot["browser"] = browser
    logger.info("spawned browser (total=%d, cap=%d)", len(_browsers), _MAX_BROWSERS)
    return slot


async def _release_browser(slot: dict) -> None:
    async with _pool_lock:
        slot["n_ctx"] = max(0, slot["n_ctx"] - 1)
        slot["last_used"] = time.monotonic()


async def _close_idle_loop():
    """Reap browsers idle (0 contexts) past the TTL — the host-side idle
    compaction, re-homed in-container ⚠️. Also evicts dead
    idle browsers. Without it the pool only ever grows."""
    while True:
        await asyncio.sleep(30)
        now = time.monotonic()
        async with _pool_lock:
            keep = []
            for b in _browsers:
                if b.get("browser") is None:
                    keep.append(b)  # mid-launch placeholder
                    continue
                dead = False
                try:
                    dead = not b["browser"].is_connected()
                except Exception:
                    dead = True
                idle_expired = b["n_ctx"] == 0 and (now - b["last_used"]) > _IDLE_BROWSER_TTL_S
                if b["n_ctx"] == 0 and (dead or idle_expired):
                    try:
                        await b["browser"].close()
                        logger.info("reaped browser (dead=%s idle>%.0fs)",
                                    dead, _IDLE_BROWSER_TTL_S)
                    except Exception as e:
                        logger.warning("idle reap failed: %s", e)
                else:
                    keep.append(b)
            _browsers[:] = keep


# ---------------------------------------------------------------------------
# Per-instance task lifecycle (ported from the former host env, removed)
# ---------------------------------------------------------------------------
@dataclass
class _Instance:
    env: MobileGymEnv
    slot: dict
    task: BaseTask
    task_id: str
    eval_mode: str
    max_steps: int
    init_obs: Observation
    #: Defaults False so an instance built without it scores the benchmark metric.
    #: Only RL opts in (see _evaluate and configs/default.yaml).
    reward_shaping: bool = False
    agent_answer: str | None = None
    step_count: int = 0


_instances: dict[str, _Instance] = {}


def _raw_action_name(action: Any) -> str:
    if isinstance(action, dict):
        name = action.get("name")
        if isinstance(name, str) and name:
            return name
    return "<invalid>"


def _read_flat_action(action: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(action, dict):
        raise TypeError(f"Lite action must be an object, got {type(action).__name__}")
    name = action.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Lite action.name must be a non-empty string")
    args = action.get("arguments")
    if not isinstance(args, dict):
        raise TypeError(
            f"Lite tool_call.arguments must be a dict, got {type(args).__name__}"
        )
    return name, args


def _append_action_error(
    action_errors: list[dict[str, Any]],
    *,
    index: int,
    kind: str,
    name: str,
    error: Any,
    action: Any = None,
    message: str | None = None,
) -> None:
    detail = str(error)
    record = {
        "index": index,
        "kind": kind,
        "name": name,
        "error": detail,
        "message": message or f"{name}: {detail}",
    }
    if isinstance(action, dict) and action.get("call_id"):
        record["call_id"] = action["call_id"]
    action_errors.append(record)


def _duration_cap_seconds(action_name: str) -> float:
    return _MODEL_DURATION_CAPS_SECONDS.get(
        action_name,
        _DEFAULT_MODEL_DURATION_CAP_SECONDS,
    )


def _coerce_duration_number(raw: Any, action_name: str) -> float:
    label = f"{action_name}.duration"
    if isinstance(raw, bool):
        raise ValueError(f"{label} must be a finite number, got bool")
    if isinstance(raw, (int, float)):
        duration = float(raw)
    elif isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            raise ValueError(f"{label} must be a finite number")
        try:
            duration = float(raw)
        except ValueError as e:
            raise ValueError(f"{label} must be a finite number") from e
    else:
        raise ValueError(
            f"{label} must be a finite number, got {type(raw).__name__}"
        )
    if not math.isfinite(duration):
        raise ValueError(f"{label} must be finite")
    cap = _duration_cap_seconds(action_name)
    if duration > cap:
        raise ValueError(f"{label} must be <= {cap:g}")
    return duration


def _duration_seconds(
    args: dict[str, Any],
    action_name: str,
    *,
    default: float,
    allow_zero: bool = False,
) -> float:
    raw = args.get("duration", default)
    if raw is None:
        raw = default
    duration = _coerce_duration_number(raw, action_name)
    if duration < 0 or (duration == 0 and not allow_zero):
        comparator = "non-negative" if allow_zero else "greater than 0"
        raise ValueError(f"{action_name}.duration must be {comparator}")
    return duration


def launchable_app_names() -> list[str]:
    """Every app name ``open_app`` can actually launch, by MobileGym's own two rules:
    an exact ``APP_NAME_MAP`` key, or a string whose lowercase matches an appId.

    Reported by ``/healthz`` so the host's advertised ``open_app`` enum can be checked
    against what this device can resolve. The host enum is a deliberate SUBSET (24 of
    these 95 — it omits Chinese aliases and snake_case appIds), so the check is
    ``host_enum <= launchable_apps``, not set equality.
    """
    name_map = MobileGymEnv.APP_NAME_MAP
    names = set(name_map)
    names.update(str(app_id) for app_id in name_map.values())
    return sorted(names)


def _checked_app_name(app_name: str) -> str:
    """``app_name`` verbatim if this device can launch it, else raise so the MODEL is told.

    bench_env's ``_open_app`` silently ``return``s on a name it cannot resolve (the screen
    simply does not change, and even its log is gated behind ``verbose``), so an
    unresolvable name would burn the rest of the episode invisibly. ``_translate_action``'s
    caller catches ``ValueError`` as a ``model_action`` error, so raising sends the valid
    list back to the model instead.

    Deliberately validates WITHOUT rewriting: the two rules below are exactly the ones
    bench_env then applies itself, so translating here would only replace the model's own
    word with an arbitrary alias picked by dict order (every appId became a Chinese key —
    ``"bilibili"`` → ``"哔哩哔哩"``) for no behavioural gain.
    """
    name_map = MobileGymEnv.APP_NAME_MAP
    if app_name in name_map:
        return app_name
    if str(app_name).lower() in {str(app_id).lower() for app_id in name_map.values()}:
        return app_name
    raise ValueError(
        f"open_app: {app_name!r} is not an app this device can launch. "
        f"Launchable: {launchable_app_names()}"
    )


def _translate_action(name: str, args: dict[str, Any]) -> Action | None:
    """LiteMobileActionSpace action → bench_env Action. Verbatim port of
    the former host env's _translate_action (removed).

    ``None`` means "deliberate no-op" (nothing to execute, nothing to report).
    Model mistakes RAISE so ``/step`` can report them back to the agent:
    :class:`_UnsupportedAction` for an action this backend cannot express,
    and the usual ``ValueError``/``TypeError``/... for bad arguments.
    """
    if name == "tap":
        clicks = args.get("clicks", 1)
        at = ActionType.DOUBLE_TAP if clicks >= 2 else ActionType.CLICK
        return Action(at, {"point": args.get("coordinate", [500, 500])})
    if name == "long_press":
        duration_ms = int(
            _duration_seconds(args, "long_press", default=0.8) * 1000
        )
        return Action(ActionType.LONG_PRESS,
                      {"point": args.get("coordinate", [500, 500]), "duration": duration_ms})
    if name == "type":
        return Action(ActionType.TYPE, {"value": args.get("text", "")})
    if name == "swipe":
        if "duration" in args:
            # bench_env SWIPE has no duration field; validate the model value
            # here so malformed durations still become action_errors.
            _duration_seconds(args, "swipe", default=0.4)
        return Action(ActionType.SWIPE, {
            "point1": args.get("start_coordinate", [500, 500]),
            "point2": args.get("coordinate", [500, 500]),
        })
    if name == "drag":
        if "duration" in args:
            # bench_env DRAG has no duration field; keep duration validate-only.
            _duration_seconds(args, "drag", default=0.4)
        return Action(ActionType.DRAG, {
            "point1": args.get("start_coordinate", [500, 500]),
            "point2": args.get("coordinate", [500, 500]),
        })
    if name == "open_app":
        return Action(ActionType.AWAKE, {"value": _checked_app_name(args.get("app_name", ""))})
    if name == "system_button":
        menu_action = getattr(ActionType, "MENU", ActionType.RECENT)
        _map = {
            "Back": ActionType.BACK, "Home": ActionType.HOME,
            "Enter": ActionType.ENTER, "Menu": menu_action,
            "Recent": ActionType.RECENT,
        }
        button = args.get("button", "")
        at = _map.get(button)
        if at is None:
            raise ValueError(
                f"system_button: unknown button {button!r}; expected one of {sorted(_map)}"
            )
        return Action(at, {})
    if name == "wait":
        return Action(ActionType.WAIT, {
            "value": _duration_seconds(
                args, "wait", default=1.0, allow_zero=True,
            )
        })
    if name in ("screenshot", "terminate", "response", "noop"):
        return None
    raise _UnsupportedAction(name)


def _evaluate_grounded(task: BaseTask, judge_input: JudgeInput) -> JudgeResult:
    """Grounded eval path for AnswerSheet tasks — verbatim port of host
    the former host env's _evaluate_grounded (removed)."""
    sheet_state = judge_input.apps.get("answer_sheet", {})
    has_grounded = hasattr(task, "get_expected_response")
    try:
        from bench_env.task.common_tasks import AnswerTask as _AT
    except ImportError:
        _AT = BaseTask
    _cg_definer = next(
        (c for c in type(task).__mro__ if "check_goals" in c.__dict__), BaseTask
    )
    has_custom_cg = _cg_definer not in (BaseTask, _AT)

    if has_grounded and not has_custom_cg:
        from bench_env.task.common_tasks import build_grounded_checks
        try:
            checks = build_grounded_checks(task, judge_input, sheet_state)
        except Exception as err:
            return JudgeResult.error(f"build_grounded_checks() raised: {err}")
        return task._evaluate_with_checks(judge_input, checks)

    answers = sheet_state.get("answers", {})
    submitted = sheet_state.get("submitted", False)
    if answers and submitted:
        parts = []
        for k in sorted(answers, key=lambda x: int(x)):
            v = answers[k]
            if isinstance(v, list):
                parts.extend(str(item) for item in v)
            else:
                parts.append(str(v))
        judge_input = JudgeInput(
            init_obs=judge_input.init_obs, last_obs=judge_input.last_obs,
            answer=", ".join(parts),
        )
    else:
        judge_input = JudgeInput(
            init_obs=judge_input.init_obs, last_obs=judge_input.last_obs, answer=None,
        )
    return task.evaluate(judge_input)


async def _evaluate(inst: _Instance, terminated: bool) -> float:
    """Run the task judge — port of the former host env's _evaluate (removed).

    Returns MobileGym's official Success Rate as 1.0/0.0, which upstream defines
    (bench_env/runner/base.py::EpisodeResult.success) as all three of:
    ``stop_reason == COMPLETE`` and ``judge.success`` (goal achieved) and
    ``judge.clean`` (no unexpected state changes) -- the last two being
    ``JudgeResult.passed`` (bench_env/task/judge.py). ``terminated`` is this env's
    COMPLETE signal: an episode that merely ran out of steps scores 0 upstream even
    if the goal happened to be met, so it must gate the score here too.

    This deliberately does NOT return ``result.progress`` (the fraction of
    check_goals that passed). progress >= success always, so reporting it as THE
    score made this env read high against MobileGym's published numbers. It is a
    separate, coarser signal and upstream reports it separately.

    0.0 on any exception (the silent path; logged).
    """
    if inst.task is None or inst.init_obs is None:
        return 0.0
    try:
        apps = list(inst.task.apps) if inst.task.apps else None
        final_state = await inst.env.get_state(required_apps=apps) or {}
        final_route = await inst.env.get_route()
        final_screenshot = await inst.env.get_observation()
        final_obs = Observation(
            screenshot_bytes=final_screenshot.get_screenshot_bytes(),
            route=final_route or {},
            state=final_state,
        )
        judge_input = JudgeInput(
            init_obs=inst.init_obs, last_obs=final_obs, answer=inst.agent_answer,
        )
        if inst.eval_mode == "grounded" and getattr(inst.task, "answer_fields", None):
            result = _evaluate_grounded(inst.task, judge_input)
        else:
            result = inst.task.evaluate(judge_input)
        if result.judge_error:
            # JudgeResult.error() RETURNS (bench_env/task/judge.py::error), it does
            # not raise, so the except below never sees it and the score is a plain
            # 0.0 -- indistinguishable from an agent that achieved nothing. Log it so
            # a broken judge is attributable instead of silently counted as failure.
            logger.error(
                "Judge errored for %s (scored 0, NOT an agent failure): %s",
                inst.task_id, result.judge_error,
            )
        if terminated and result.success and result.clean:
            return 1.0
        if not inst.reward_shaping:
            return 0.0
        # Shaping (RL only, same contract as screenspot_pro/osworld_g: partial
        # credit ONLY where the metric is 0, and strictly below 1.0 so a truly
        # successful episode always outranks a partial one and the argmax is
        # unchanged). The explicit 0.5 cap is what those two do not need -- their
        # shaped value is 1-dist/max_dist, which only reaches 1.0 at a point that
        # already scored 1.0. Here the cap is load-bearing: `progress` is the fraction of
        # check_goals that passed and looks at NEITHER `clean` NOR COMPLETE
        # (bench_env/task/base.py computes it before either is known), so
        # paying it at face value would let a policy max the reward by hitting
        # every goal while dirtying state or never calling terminate.
        return _SHAPED_REWARD_CEILING * float(result.progress)
    except Exception as e:
        logger.error("Evaluation failed for %s: %s", inst.task_id, e)
        return 0.0


# ---------------------------------------------------------------------------
# Lifecycle background task
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def _startup():
    asyncio.create_task(_close_idle_loop())
    logger.info("mobilegym server up: vite=%s max_browsers=%d contexts/browser=%d",
                _VITE_URL, _MAX_BROWSERS, _CONTEXTS_PER_BROWSER)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "launchable_apps": launchable_app_names(),
        "browsers": len(_browsers),
        "instances": len(_instances),
        "contexts_in_use": sum(b["n_ctx"] for b in _browsers),
    }


@app.post("/reset")
async def reset(request: Request) -> dict[str, Any]:
    """Ported from the former host env's reset (removed): acquire a browser,
    start an env, instantiate + setup the task, inject AnswerSheet (grounded),
    return the initial screenshot + instruction + max_steps."""
    body = await request.json()
    task_id = body["task_id"]
    seed = body.get("seed")
    eval_mode = body.get("eval_mode", "text")
    reward_shaping = bool(body.get("reward_shaping", False))
    physical_size = tuple(body.get("physical_size", [1080, 2400]))
    dpr = float(body.get("dpr", 3.0))
    delay = float(body.get("delay", 0.8))

    try:
        task_cls = TaskRegistry().get_by_id(task_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"unknown task_id {task_id!r}: {e}")

    slot = await _acquire_browser()
    env = None
    try:
        env = MobileGymEnv(
            url=_VITE_URL, browser=slot["browser"],
            physical_size=physical_size, device_scale_factor=dpr,
            headless=True, delay_after_action=delay, verbose=False,
        )
        await env.start()

        task = task_cls(_seed=seed)
        # Template-variant selection (ported from the former host env, removed).
        n_templates = len(getattr(task_cls, "templates", []) or [])
        if n_templates > 1:
            if seed is not None:
                task._template_index = random.Random(
                    f"tpl:{seed}:{task_id}"
                ).randrange(n_templates)
            else:
                task._template_index = random.Random().randrange(n_templates)

        init_obs = await task.setup(env)

        # AnswerSheet injection — grounded mode only (ported from the former host env, removed).
        if eval_mode == "grounded" and getattr(task, "answer_fields", None):
            fields = task._resolve_answer_fields()
            question = task._resolve_answer_question() or task.description
            await env.set_state({"apps": {"answer_sheet": {
                "question": question,
                "hint": getattr(task, "answer_hint", None),
                "fields": fields,
                "answers": {},
                "submitted": False,
            }}}, deep=True, reload=False)
            task.task_name = (
                task.description + " 然后打开 答题卡 APP 在里面回答问题并提交"
            )

        # Budget: prefer the host-supplied value (RemoteMobileGymEnv.max_steps,
        # which already folds in any env_kwargs override + the grounded bonus) so
        # a config/CLI max_steps override actually caps the episode. Fall back to
        # recomputing it (text-mode base + grounded AnswerSheet bonus) only when
        # the host didn't send one (older client).
        override = body.get("max_steps")
        if override is not None:
            max_steps = int(override)
        else:
            bonus = 15 if (eval_mode == "grounded" and getattr(task_cls, "answer_fields", None)) else 0
            max_steps = _resolve_task_max_steps(task_cls) + bonus

        screenshot_b64 = base64.b64encode(init_obs.get_screenshot_bytes()).decode()
    except HTTPException:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass
        await _release_browser(slot)
        raise
    except Exception:
        if env is not None:  # audit fix c: close env before releasing slot
            try:
                await env.close()
            except Exception:
                pass
        await _release_browser(slot)
        raise

    iid = uuid.uuid4().hex
    _instances[iid] = _Instance(
        env=env, slot=slot, task=task, task_id=task_id, eval_mode=eval_mode,
        reward_shaping=reward_shaping,
        max_steps=max_steps, init_obs=init_obs,
    )
    return {
        "instance_id": iid,
        "screenshot_b64": screenshot_b64,
        "instruction": task.description,
        "max_steps": max_steps,
    }


async def _capture_frame_b64(inst: _Instance) -> str:
    """One result frame (base64 PNG) of the device's CURRENT screen.

    ``MobileGymEnv`` was constructed with ``delay_after_action``, so by the time
    ``env.step()`` returns the UI has already settled and the frame is a
    post-action state, not a mid-transition one.
    """
    obs = await inst.env.get_observation()
    return base64.b64encode(obs.get_screenshot_bytes()).decode()


@app.post("/step")
async def step(request: Request) -> dict[str, Any]:
    """Ported from the former host env's step (removed): translate + execute a
    batch of Lite tool calls, handle terminate/response, evaluate on
    termination/truncation, return screenshots + reward + flags + executed."""
    try:
        body = await request.json()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"invalid JSON body: {e}") from e
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=400,
            detail=f"step body must be an object, got {type(body).__name__}",
        )
    iid = body.get("instance_id")
    if not iid:
        raise HTTPException(status_code=400, detail="missing instance_id")
    iid = str(iid)
    inst = _instances.get(iid)
    if inst is None:
        raise HTTPException(status_code=404, detail=f"unknown instance {iid}")

    terminated = False
    # Whether the episode ended via COMPLETE (vs ABORT). Only COMPLETE counts
    # toward the Success Rate; see the terminate branch below and _evaluate.
    completed = False
    executed: list[dict[str, Any]] = []
    # One result frame per EXECUTED action, in action order. The host ships the
    # whole batch here, so this loop -- not the host -- is the only place that
    # can see the screen between two actions of one batch.
    frames_b64: list[str] = []
    # Per-action model feedback, keyed by the INDEX of the offending entry in the
    # ``actions`` list the host sent — the host pairs it back to the originating
    # canonical ``call_id``. Purely ADDITIVE to the response (an older host just
    # ignores the key), so the wire stays backward-tolerant.
    action_errors: list[dict[str, Any]] = []
    raw_actions = body.get("actions", [])
    if isinstance(raw_actions, list):
        actions = raw_actions
    else:
        _append_action_error(
            action_errors,
            index=0,
            kind="model_action",
            name="<invalid>",
            error=f"body.actions must be a list, got {type(raw_actions).__name__}",
            action=raw_actions,
        )
        actions = []

    for index, action in enumerate(actions):
        try:
            name, args = _read_flat_action(action)
        except _MODEL_ACTION_ERROR_TYPES as e:
            logger.warning("invalid mobilegym action envelope for %s: %s", iid, e)
            _append_action_error(
                action_errors,
                index=index,
                kind="model_action",
                name=_raw_action_name(action),
                error=e,
                action=action,
            )
            continue

        if name == "terminate":
            terminated = True
            status = args.get("status", "success")
            if status == "success":
                completed = True
                mgym_act = Action(ActionType.COMPLETE, {"return": ""})
            else:
                # ABORT, not COMPLETE. Upstream's Success Rate requires
                # stop_reason == COMPLETE (bench_env/runner/base.py::
                # EpisodeResult.success), so a self-declared failure scores 0 there
                # even with every goal met -- ``terminated`` alone cannot gate the
                # score, since it is True for both branches.
                completed = False
                mgym_act = Action(ActionType.ABORT, {"value": args.get("reason", "")})
            await inst.env.step(mgym_act)
            # ``terminate`` really drives the device here (COMPLETE/ABORT), so it
            # is an executed action and owes its frame -- which also keeps the
            # last frame of a terminating batch the POST-terminate screen, as it
            # has always been.
            frames_b64.append(await _capture_frame_b64(inst))
            break

        if name == "response":
            inst.agent_answer = args.get("text", "")
            try:
                await inst.env.step(Action(ActionType.ANSWER, {"value": inst.agent_answer}))
            except Exception as e:
                message = f"response side effect failed: {e}"
                logger.warning("response action failed for %s: %s", iid, e)
                _append_action_error(
                    action_errors,
                    index=index,
                    kind="tool_execution",
                    name=name,
                    error=message,
                    action=action,
                    message=f"{name}: {message}",
                )
                executed.append({
                    "call": "noop",
                    "args": {"name": name, "reason": message, "is_error": True},
                })
            else:
                # Success only, and OUTSIDE the try: a failure while capturing is
                # not the answer's fault and must not be reported as one.
                executed.append({"call": "ANSWER", "args": args})
                frames_b64.append(await _capture_frame_b64(inst))
            if inst.eval_mode == "text":
                # Answering IS how a text task finishes here, so this counts as
                # COMPLETE for scoring. Upstream's AnswerHandler returns done=False
                # and leaves stop_reason unset (bench_env/env/mobile_gym.py::
                # AnswerHandler) because its runner ends the episode itself; this
                # env has no such outer runner, so the completion signal lands here.
                # Only ABORT (terminate status!=success) is excluded from the SR.
                terminated = True
                completed = True
                break
            continue

        try:
            mgym_act = _translate_action(name, args)
        except _UnsupportedAction:
            logger.warning("Unknown mobilegym action: %s(%s)", name, args)
            _append_action_error(
                action_errors,
                index=index,
                kind="unsupported_action",
                name=name,
                error=f"unsupported action: {name}",
                action=action,
                message=f"unsupported action: {name}",
            )
            continue
        except _MODEL_ACTION_ERROR_TYPES as e:
            _append_action_error(
                action_errors,
                index=index,
                kind="model_action",
                name=name,
                error=e,
                action=action,
            )
            continue
        if mgym_act is None:
            # ``screenshot``/``noop``: a dispatch no-op still RAN, and the frame
            # contract has no read-only exception list -- the count depends on
            # how many actions ran, never on what they were.
            frames_b64.append(await _capture_frame_b64(inst))
            continue
        try:
            await inst.env.step(mgym_act)
        except _MODEL_ACTION_ERROR_TYPES as e:
            # Backend rejected the model's arguments (bad point, bad value).
            # Infra failures are NOT in the tuple and still propagate → 500.
            _append_action_error(
                action_errors,
                index=index,
                kind="model_action",
                name=name,
                error=e,
                action=action,
            )
            continue
        executed.append({"call": name, "args": args})
        frames_b64.append(await _capture_frame_b64(inst))

    if not frames_b64:
        # Nothing executed (empty batch, or every call rejected). The turn still
        # owes the model exactly one observation.
        frames_b64.append(await _capture_frame_b64(inst))

    inst.step_count += 1
    truncated = not terminated and (inst.step_count >= inst.max_steps)

    reward = None
    if terminated or truncated:
        reward = await _evaluate(inst, terminated and completed)

    return {
        "screenshots_b64": frames_b64,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "executed": executed,
        "action_errors": action_errors,
    }


@app.post("/close")
async def close(request: Request) -> dict[str, Any]:
    """Ported from the former host env's close (removed): teardown task, close
    env (releases its own context+page), release the browser slot."""
    body = await request.json()
    iid = body.get("instance_id", "")
    inst = _instances.pop(iid, None)
    if inst is not None:
        if inst.task is not None:
            try:
                inst.task.teardown(inst.env)
            except Exception:
                pass
        try:
            await inst.env.close()
        except Exception as e:
            logger.warning("env close failed: %s", e)
        await _release_browser(inst.slot)
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("MOBILEGYM_RPC_PORT", "8000")))
