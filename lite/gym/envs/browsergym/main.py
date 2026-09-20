"""BrowserGym — CUA-Lite gym wrapper for BrowserGym benchmarks.

Wraps BrowserGym's web benchmarks (MiniWoB, WebArena, VisualWebArena) as
CUA-Lite gym environments using coordinate-based actions only, exposed
through LiteBrowserActionSpace ([0, 1000] coordinates).

BrowserGym provides the Playwright browser, task setup, and evaluation;
this wrapper translates CUA-Lite actions to Playwright calls and captures
screenshots as raw PNG bytes.

Prerequisites:
  - uv run --no-sync bash lite/gym/envs/browsergym/scripts/install.sh <benchmark>
  - For MiniWoB: set MINIWOB_URL env var
  - For WebArena: Docker services running + WA_* env vars
  - For VisualWebArena: Docker services + VWA_* env vars

Config: ``configs/default.yaml`` declares the uniform env kwargs that should be
visible in the registry catalog; per-benchmark task defaults such as step budget
are still table-baked at registration. The same file also owns the env-server
isolation knobs read in ``isolation.py`` via ``env_config.load(ENV_DIR)``.
Launch the env-server with ``BROWSERGYM_CONFIG=isolation`` (bundled
``configs/isolation.yaml``) to engage strict shared-backend isolation.

Usage:
    uv run python -c "
    import lite.gym as gym
    print(gym.registry.task_ids('browsergym.miniwob'))
    "
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import dataclasses
import functools
import json
import logging
import os
import subprocess
import threading
import types
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, NamedTuple

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.action_space import LiteDesktopActionSet
from lite.core.tools.calls import (
    EnvAction,
    RuntimeEnvAction,
    runtime_rejected_reason,
)
from lite.core.tools.extra_tools import LiteBrowserNavToolSet, LiteFinishToolSet
from lite.core.tools.schemas import make_tool_schema, tool_schema_name
from lite.gym.base import LiteBaseEnv
from lite.gym.envs.browsergym import isolation
from lite.gym.registry import register, registry
from lite.gym.remote.conflict import OTHERS_CONFLICT_KEYS, OTHERS_MUTATING
from lite.gym.services import BackendFamily, EnvServices, register_family, register_services
from lite.gym.types import (
    EXECUTED_ACTIONS_INFO_KEY,
    LiteEnvObservation,
    LiteEnvStepResult,
    LiteExecutedAction,
)
from lite.gym.utils import config as env_config
from lite.gym.utils.backend.coordinate import norm_to_pixel
from lite.gym.utils.backend.model_inputs import (
    coerce_model_duration,
    project_model_keys,
)
from lite.gym.utils.feedback.errors import (
    MODEL_ACTION_ERROR_TYPES,
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
    is_active_extra_tool_call,
    is_lite_action_name_or_action_batch_tool_name,
    prepare_env_tool_calls,
)
from lite.gym.utils.feedback.results import (
    build_tool_results_from_decisions,
    ordered_tool_call_ids,
)
from lite.gym.utils.feedback.surface import resolve_valid_actions
from lite.gym.wrappers import overlay_cursor_px
from lite.utils.image import encode_png
from lite.utils.path import project_root

logger = logging.getLogger(__name__)


_DOM_DEPENDENT_ACTION_SUBSETS = frozenset({
    "bid",
    "webarena",
    "visualwebarena",
    "miniwob_all",
})


def _is_lite_action_call(name: str, args: dict[str, Any]) -> bool:
    """True for BrowserGym's Lite-coordinate GUI path.

    Some BrowserGym native/BID tools share names with Lite GUI tools
    (`click`, `scroll`). Those argument shapes must still require an active
    extra schema.
    """
    if name not in LiteDesktopActionSet.get_action_names():
        return False
    if name == "click" and "bid" in args:
        return False
    if name == "scroll" and ("delta_x" in args or "delta_y" in args):
        return False
    return True


def _is_browsergym_standalone_tool_call(action: EnvAction) -> bool:
    name = action["name"]
    return (
        name in LiteDesktopActionSet.get_action_names()
        and not _is_lite_action_call(name, action["arguments"])
    )


def _patch_lightweight_obs() -> None:
    """Idempotently monkeypatch ``BrowserEnv._get_obs`` for an opt-in vision
    fast-path. Mirrors the in-repo runtime-patch convention (androidworld's
    ``_patch_adb_controller_for_fast_hang_recovery``; lite/osworld's
    ``get_rule_relativeTime``): save original, wrap, idempotent ``_lite_patched``
    guard. Reassigns the attribute on the imported module IN MEMORY — does NOT
    touch any ``.venv`` file (so it survives venv rebuilds + lives in git).

    Upstream ``_get_obs`` UNCONDITIONALLY runs ``_pre_extract`` (bid marking) +
    ``extract_dom_snapshot`` + ``extract_merged_axtree`` +
    ``extract_dom_extra_properties`` every step — all O(page nodes), computed in
    chromium over CDP — even when the agent consumes only the screenshot. On big
    WA/VWA pages that IS the 22-32s env.step bottleneck. When an env instance
    carries ``_cua_lite_skip_dom=True`` (set from
    ``BrowserGymConfig.skip_dom_extraction``), the patched method returns ONLY the
    screenshot + cheap page metadata (empty placeholders for the heavy fields),
    dropping env.step to ~1-2s. Otherwise it delegates to the original verbatim
    (zero behavior change — default path). SAFE ONLY for coordinate-only action
    spaces: ``_pre_extract`` is skipped so elements are unmarked and bid actions
    can't resolve.
    """
    import browsergym.core.env as _bgym_env

    if getattr(_bgym_env.BrowserEnv._get_obs, "_lite_patched", False):
        return  # idempotent — already patched this process
    _orig_get_obs = _bgym_env.BrowserEnv._get_obs

    def _get_obs(self):
        if not getattr(self, "_cua_lite_skip_dom", False):
            return _orig_get_obs(self)  # unchanged upstream path
        # Vision fast-path: skip the O(nodes) DOM/AXTree/extra-properties +
        # bid-marking; keep only the screenshot + cheap page metadata. Empty
        # placeholders keep the obs schema intact for downstream consumers.
        import copy
        import time as _time

        import numpy as np

        return {
            "chat_messages": tuple(copy.deepcopy(self.chat.messages)),
            "goal": _bgym_env._try_to_extract_legacy_goal(self.goal_object),
            "goal_object": tuple(copy.deepcopy(self.goal_object)),
            "open_pages_urls": tuple(p.url for p in self.context.pages),
            "open_pages_titles": tuple(p.title() for p in self.context.pages),
            "active_page_index": np.asarray([self.context.pages.index(self.page)]),
            "url": self.page.url,
            "screenshot": _bgym_env.extract_screenshot(self.page),
            "dom_object": {},
            "axtree_object": {},
            "extra_element_properties": {},
            "focused_element_bid": None,
            "last_action": self.last_action,
            "last_action_error": self.last_action_error,
            "elapsed_time": np.asarray([_time.time() - self.start_time]),
        }

    _get_obs._lite_patched = True
    _bgym_env.BrowserEnv._get_obs = _get_obs
    logger.info("browsergym _get_obs patched: opt-in lightweight vision obs available")

# ============================================================================
# Config. Read once at import via env_config.load.
# Swap the WHOLE file at startup with BROWSERGYM_CONFIG=<abs-path|bundled-name>
# (e.g. BROWSERGYM_CONFIG=isolation → configs/isolation.yaml). A rollout's
# env_kwargs still override per run. isolation.py loads the SAME cached CFG.
# ENV_DIR is the browsergym PACKAGE dir (this file lives directly under it).
# ============================================================================
ENV_DIR = str(Path(__file__).parent)
CFG = env_config.load(ENV_DIR)
# --- env_kwargs (per-instance) — these ARE the __init__ defaults below. browsergym
#     registers MULTIPLE benchmarks with differing budgets, so the per-benchmark
#     knob (max_steps) is `null` here and hardcoded at registration (10/30); the
#     uniform knobs carry their real default. ---
_MAX_STEPS = CFG.env_kwargs["max_steps"]                  # null → per-benchmark registration value
_POST_ACTION_DELAY = CFG.env_kwargs["post_action_delay"]  # 0.0 (synchronous render)
_SEED = CFG.env_kwargs["seed"]                            # null → unseeded bare construction
_EXTRA_TOOLS = CFG.env_kwargs["extra_tools"]              # null default → all action_subset tools; [] → none; [names] → subset
_INCLUDE_PAGE_CONTEXT_TEXT = bool(CFG.env_kwargs["include_page_context_text"])
# --- server_kwargs (per-deployment) — auxiliary-service host ports. SINGLETON
#     (one env-server ↔ one shared WA/VWA stack + miniwob singleton): each is a
#     PREFERRED port, auto-reallocated if busy (see _auto_pick_webarena_ports /
#     _pick_miniwob_port). ---
_SK = CFG.server_kwargs
_MINIWOB_PORT_DEFAULT = _SK["miniwob_port"]
_SHOPPING_PORT_DEFAULT = _SK["shopping_port"]
_SHOPPING_ADMIN_PORT_DEFAULT = _SK["shopping_admin_port"]
_REDDIT_PORT_DEFAULT = _SK["reddit_port"]
_GITLAB_PORT_DEFAULT = _SK["gitlab_port"]
_WIKIPEDIA_PORT_DEFAULT = _SK["wikipedia_port"]
_CLASSIFIEDS_PORT_DEFAULT = _SK["classifieds_port"]
_HOMEPAGE_PORT_DEFAULT = _SK["homepage_port"]
_MAP_PORT_DEFAULT = _SK["map_port"]
_WA_FALLBACK_START, _WA_FALLBACK_END = _SK["wa_fallback_port_range"]
_MINIWOB_PORT_RANGE: tuple[int, int] = tuple(_SK["miniwob_port_range"])  # type: ignore[assignment]
_WIKIPEDIA_PATH = _SK["wikipedia_path"]
_CLASSIFIEDS_RESET_TOKEN = _SK["classifieds_reset_token"]
_LLM_JUDGE_MODEL = _SK["llm_judge_model"]
# ============================================================================

# Re-export so the env-server's restore dispatch finds the hook at the
# conventional ``<env>.main.restore_backend`` (see lite/gym/remote/conflict.py
# restore_backend_dispatch). Policy lives in isolation.py.
restore_backend = isolation.restore_backend

# ---------------------------------------------------------------------------
# Optional dependency check
# ---------------------------------------------------------------------------

try:
    import browsergym.core  # noqa: F401
except ImportError:
    from lite.gym.errors import EnvDepsMissingError
    raise EnvDepsMissingError(
        what="browsergym package not installed",
        install="uv run --no-sync bash lite/gym/envs/browsergym/scripts/install.sh <benchmark>",
        see="lite/gym/envs/browsergym/README.md",
    )

# Each BrowserGymEnv gets its own single-thread executor so that its
# Playwright instance stays in one thread. Multiple envs can run in
# parallel because each has an independent executor + Playwright — there is
# NO global op-lock serializing browser steps (see _ThreadLocalPlaywright).
# The one cross-thread guard that remains is a NARROW lock around the lazy
# ``sync_playwright().start()`` (the node-driver subprocess spawn) — see
# ``_pw_start_lock`` below; it never touches per-step browser ops.
_thread_local = threading.local()

# Serializes ONLY the one-time, per-thread ``sync_playwright().start()`` across
# threads. The global op-lock this design removed had exactly one load-bearing
# job: stopping concurrent ``start()`` calls (each spawns the node driver
# subprocess) from racing — uvloop raises "Racing with another loop to spawn a
# process" when two loops spawn at once. Today uvloop is not the global asyncio
# policy (uvicorn's loop_factory path), so the race is dormant — but uvloop IS an
# installed dep, so we keep this narrow guard rather than rely on that invariant.
# start() runs once per thread (then cached in _thread_local.pw), so this lock is
# contended only on the cold-start wave and NEVER serializes per-step ops.
_pw_start_lock = threading.Lock()


class _ThreadLocalPlaywright:
    """Proxy installed as ``browsergym.core._PLAYWRIGHT`` so every attribute access
    resolves to the CURRENT thread's Playwright instance.

    Each env runs in its own single-thread executor with its own ``sync_playwright``
    (its browser/context/page stay thread-confined), so routing ``_PLAYWRIGHT`` per
    thread here removes the **global lock** that used to wrap every browser op. That
    lock serialized all concurrent envs — making ``env.step`` scale as ~N×(single
    step) under concurrency N (e.g. ~30 s at N=8). Lazily starts a thread's instance
    on first use, under ``_pw_start_lock`` so concurrent first-starts can't race the
    driver-subprocess spawn (the global lock's only real job). (NB: dunder lookups
    like ``bool()`` bypass ``__getattr__``, so the proxy reads truthy —
    ``_get_global_playwright``'s ``if not _PLAYWRIGHT`` won't replace it.)"""
    def __getattr__(self, name):
        pw = getattr(_thread_local, "pw", None)
        if pw is None:
            import playwright.sync_api
            # Narrow lock: guards ONLY the subprocess-spawning start(), not the
            # returned object's per-step ops. Each thread runs this once.
            with _pw_start_lock:
                pw = playwright.sync_api.sync_playwright().start()
            _thread_local.pw = pw
        return getattr(pw, name)


_pw_proxy = _ThreadLocalPlaywright()


def _install_pw_proxy() -> None:
    """Install the per-thread Playwright proxy as BrowserGym's global (idempotent)."""
    import browsergym.core as bgym_core
    if bgym_core._PLAYWRIGHT is not _pw_proxy:
        bgym_core._PLAYWRIGHT = _pw_proxy


def _with_playwright(fn):
    """Run *fn* in the caller's (env-dedicated) thread with BrowserGym's ``_PLAYWRIGHT``
    routed per-thread via the proxy. **No global lock** — concurrent envs' browser ops
    run in parallel, each on its own thread + its own Playwright instance."""
    _install_pw_proxy()
    return fn()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Per-bench paper-aligned action subsets match BrowserGym's
# `DEFAULT_HIGHLEVEL_ACTION_SET_ARGS`, which AgentLab's GenericAgent
# `set_benchmark()` overrides `agent.action_set` with at runtime — see upstream
# AgentLab `generic_agent.py`.
# The `webarena` / `visualwebarena` / `miniwob_all` keys live in
# BrowserGym's ACTION_SUBSETS dict and resolve to curated bid-style action
# sets per benchmark — NOT just the bare "bid" subset that
# `agent_as_annotators/modeling.py:321` ostensibly sets.
#
# Verified action counts per preset (introspected from
# `HighLevelActionSet.action_set`):
#   webarena (15):        noop, scroll, keyboard_press, click, fill, hover,
#                         tab_focus, new_tab, go_back, go_forward, goto,
#                         tab_close, select_option, send_msg_to_user,
#                         report_infeasible
#   visualwebarena (16):  webarena + upload_file
#   miniwob_all (11):     noop + mouse_move, mouse_click, mouse_dblclick,
#                         mouse_down, mouse_up, scroll, click,
#                         keyboard_press, keyboard_type, fill


class _BgymParam(NamedTuple):
    """One introspected BrowserGym action parameter.

    ``schema`` is the parameter's FULL JSON-schema fragment, not a bare type
    name: BrowserGym annotates its most constrained parameters with
    ``Literal[...]`` (``click.button``), ``list[Literal[...]]``
    (``click.modifiers``) and unions (``select_option.options``,
    ``upload_file.file``), none of which a single type string can hold. The
    fragment is what carries the enum through to the emitted tool schema.
    """

    name: str
    schema: dict[str, Any]
    required: bool


# Module-level cache: action name → ordered tuple of `_BgymParam`.
# Populated lazily by introspecting `browsergym.core.action.functions`.
_BGYM_PARAM_INFO_CACHE: dict[str, tuple[_BgymParam, ...] | None] = {}

# Scalar Python annotations → JSON-schema type names.
_JSON_SCALAR_TYPES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _json_schema_for_annotation(ann: Any) -> dict[str, Any]:
    """Compile one Python annotation into a JSON-schema fragment.

    Handles the four shapes BrowserGym 0.14.3 actually uses:

    * scalars (``bid: str``, ``x: float``) → ``{"type": ...}``;
    * ``Literal["left", "middle", "right"]`` (``click.button`` and the four
      ``mouse_*`` buttons) → ``{"type": "string", "enum": [...]}`` — the model
      is told the three legal values instead of guessing at a free string;
    * ``list[Literal[...]]`` (``click.modifiers`` / ``dblclick.modifiers``) →
      an array whose ``items`` carry the element enum. JSON-schema requires
      ``items`` on arrays anyway (the OpenAI function-tool API rejects a bare
      ``"array"``), so a bare ``list`` annotation still falls back to
      ``{"type": "string"}`` items;
    * ``str | list[str]`` (``select_option.options``, ``upload_file.file``) →
      ``oneOf``, so the multi-select / multi-file form is advertised rather
      than collapsing to the single-value string.

    Anything else degrades to ``{"type": "string"}`` (let the model pick).
    """
    import typing

    if ann in _JSON_SCALAR_TYPES:
        return {"type": _JSON_SCALAR_TYPES[ann]}
    origin = typing.get_origin(ann)
    if origin is Literal:
        values = typing.get_args(ann)
        return {"type": _JSON_SCALAR_TYPES.get(type(values[0]), "string"), "enum": list(values)}
    if origin is list or ann is list:
        args = typing.get_args(ann)
        items = _json_schema_for_annotation(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": items}
    if origin in (types.UnionType, typing.Union):
        return {"oneOf": [_json_schema_for_annotation(a) for a in typing.get_args(ann)]}
    return {"type": "string"}


def _bgym_param_info(name: str) -> tuple[_BgymParam, ...] | None:
    """Introspect a BrowserGym action function by name.

    `HighLevelAction` no longer carries a `.function` attribute (only
    `signature`/`description`/`examples` strings) but the underlying
    Python functions live in `browsergym.core.action.functions`, where
    `getattr(F, name)` returns the callable. We use `inspect.signature`
    on it to derive parameter order, JSON-schema fragment, and required-ness.

    Returns None for unknown action names; result is cached per call.
    """
    if name in _BGYM_PARAM_INFO_CACHE:
        return _BGYM_PARAM_INFO_CACHE[name]
    try:
        import inspect

        import browsergym.core.action.functions as F

        fn = getattr(F, name, None)
        if fn is None or not callable(fn):
            _BGYM_PARAM_INFO_CACHE[name] = None
            return None
        sig = inspect.signature(fn)
        info: list[_BgymParam] = []
        for p in sig.parameters.values():
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            info.append(
                _BgymParam(
                    p.name,
                    _json_schema_for_annotation(p.annotation),
                    p.default is inspect.Parameter.empty,
                )
            )
        result = tuple(info)
        _BGYM_PARAM_INFO_CACHE[name] = result
        return result
    except Exception:
        _BGYM_PARAM_INFO_CACHE[name] = None
        return None


def _tool_schema_from_signature(name: str, description: str = "") -> dict | None:
    """Derive an OpenAI-style tool schema for a BrowserGym action by name.

    Returns None if the action name isn't a known BrowserGym function, so the
    caller can skip it cleanly. Every parameter fragment comes from
    ``_json_schema_for_annotation`` — there are no per-action special cases,
    so ``select_option.options`` and ``upload_file.file`` get the same union
    treatment from the same rule.
    """
    info = _bgym_param_info(name)
    if info is None:
        return None
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    for param in info:
        # deepcopy: ``info`` is the module-level cache, so every emitted schema
        # must own its fragments (metadata copies are handed to callers).
        properties[param.name] = copy.deepcopy(param.schema)
        if param.required:
            required.append(param.name)
    return make_tool_schema(
        name,
        description=description.strip(),
        parameters={
            "type": "object",
            "properties": properties,
            "required": required,
        },
    )


# BrowserGym nav action names → canonical CUA-lite nav names. In coord/vision
# mode nav is advertised as canonical ``extra_tools`` sourced from
# ``LiteBrowserNavToolSet.get_tool_schemas(include=)`` (identical surface to webgym
# → SFT transfer works);
# ``_to_bgym_code`` remaps them back at execution.
_BGYM_NAV_TO_CANONICAL: dict[str, str] = {
    "goto": "goto", "go_back": "back", "go_forward": "forward",
    "new_tab": "new_tab", "tab_focus": "switch_tab", "tab_close": "close_tab",
}

# BrowserGym terminal action names → the canonical finish tool each is surfaced
# as. Advertising them under the canonical name is what lets ``step`` reject a
# raw ``send_msg_to_user``/``report_infeasible`` as a noncanonical input tool —
# which is also why their descriptions cannot reuse BG's examples verbatim.
_BGYM_FINISH_TO_CANONICAL: dict[str, str] = {
    "send_msg_to_user": "response",
    "report_infeasible": "terminate",
}


def _canonical_finish_example(example: str, canonical: str) -> str:
    """Re-render a BrowserGym finish example as a call to the CANONICAL tool.

    BG's examples are single-argument source snippets naming BG's own function
    (``send_msg_to_user('…')`` / ``report_infeasible('…')``). Copied verbatim
    onto a schema advertised as ``response``/``terminate``, they give the model a
    worked example of the one call guaranteed to be rejected, so the payload is
    lifted out and re-rendered with the canonical name and the ARGUMENT names the
    schema actually declares.

    An example we cannot parse is passed through unchanged: a slightly-off
    example is a much smaller harm than dropping BG's reference text.
    """
    stripped = example.strip()
    payload = stripped
    if "(" in stripped and stripped.endswith(")"):
        payload = stripped[stripped.index("(") + 1 : -1].strip()
    if canonical == "response":
        return f"response(text={payload})"
    return f"terminate(status='failure', reason={payload})"


def _tools_for_subsets(subsets: tuple[str, ...] | list[str]) -> list[dict]:
    """Auto-derive tool schemas by introspecting BrowserGym's HighLevelActionSet.

    Pass `subsets=("webarena",)` etc. — same keys the per-bench
    DEFAULT_HIGHLEVEL_ACTION_SET_ARGS uses. Returns one tool schema per
    action in the resulting `action_set.action_set` dict, deduped by name.
    Skips `noop` (always present, useless to expose) and any action whose
    underlying function we can't introspect.

    Vision-pipeline filter: when `"coord"` is in `subsets`, the agent drives the
    cua-lite browser action space (native desktop-coordinate ``computer_use``:
    click/scroll/type/key).
    So in coord mode we:
      * drop BrowserGym's coord preset (mouse_*/keyboard_*) — the native tool;
      * keep nav as CANONICAL extra_tools from
        ``LiteBrowserNavToolSet.get_tool_schemas(include=)`` (``goto``/``back``/...) —
        identical to webgym so a student SFT'd on webgym transfers
        (north-star #2); execution remaps via ``_to_bgym_code``;
        they are emitted in ``LiteBrowserNavToolSet`` DECLARATION order, not in the
        order BrowserGym's ``action_set`` happens to iterate;
      * map browsergym ``send_msg_to_user``/``report_infeasible`` onto
        canonical ``response``/``terminate`` where surfaced.
    Non-coord (bid/AXTree) mode is unchanged — bgym names used natively.
    """
    if not subsets:
        return []
    try:
        from browsergym.core.action.highlevel import ACTION_SUBSETS, HighLevelActionSet
    except ImportError:
        return []

    try:
        hl = HighLevelActionSet(
            subsets=list(subsets),
            multiaction=False,
            strict=False,
        )
    except (ValueError, KeyError):
        return []

    is_coord = "coord" in subsets
    # In coord mode the native browser coordinate tool covers the coord preset,
    # so drop those bgym duplicates. BrowserGym finish/chat actions are remapped
    # to explicit extra tool schemas below.
    drop: set[str] = set()
    if is_coord:
        try:
            drop = {fn.__name__ for fn in ACTION_SUBSETS.get("coord", [])}
        except Exception:
            drop = set()

    seen: set[str] = set()
    out: list[dict] = []
    canonical_nav: list[str] = []
    for name, action in hl.action_set.items():
        if name in seen or name == "noop" or name in drop:
            continue
        seen.add(name)
        if name in _BGYM_NAV_TO_CANONICAL:
            # Defer to canonical schemas (single source of truth) below.
            canonical_nav.append(_BGYM_NAV_TO_CANONICAL[name])
            continue
        if name in _BGYM_FINISH_TO_CANONICAL:
            # Same description recipe as the generic branch below (BG's own
            # ``describe()`` text: one-liner + examples) plus the two edits that
            # follow from advertising these under a CANONICAL name:
            #   * the examples are re-rendered by ``_canonical_finish_example``.
            #     Verbatim they read ``Examples: send_msg_to_user('…')`` — a
            #     worked example naming a function the model cannot call, since
            #     ``step`` rejects the bgym name as a noncanonical input tool.
            #   * " Calling this ENDS the episode." is appended HERE, not in the
            #     generic branch, because this branch returns early and is now
            #     the only place a terminal action is described. BG's terse
            #     one-liner doesn't surface the termination semantics —
            #     observed misuse on GPT-5.4 + WA.
            canonical = _BGYM_FINISH_TO_CANONICAL[name]
            schema = LiteFinishToolSet.get_tool_schema(canonical)
            extra_description = action.description or ""
            if action.examples:
                extra_description += " Examples: " + "; ".join(
                    _canonical_finish_example(example, canonical)
                    for example in action.examples
                )
            extra_description += " Calling this ENDS the episode."
            function = schema["function"]
            function["description"] = (
                function.get("description", "") + " " + extra_description
            ).strip()
            out.append(schema)
            continue
        # ``description`` mirrors BG's own ``HighLevelActionSet.describe()``
        # output (description + examples) so our tool surface matches the
        # reference. The terminal actions never reach here — the early-returning
        # branch above owns them, including the "ENDS the episode" hint.
        description = action.description or ""
        if action.examples:
            description += " Examples: " + "; ".join(action.examples)
        schema = _tool_schema_from_signature(name, description)
        if schema is not None:
            out.append(schema)
    if canonical_nav:
        out.extend(LiteBrowserNavToolSet.get_tool_schemas(include=canonical_nav))
    return out


def _extra_tool_schemas_for_subsets(
    subsets: tuple[str, ...] | list[str],
    extra_tools: list[str] | None,
) -> list[dict]:
    """Select standalone tool schemas out of the ``subsets``-derived catalog.

    browsergym's tri-state (the SHIPPED default is ``None`` — see
    ``configs/default.yaml``), deliberately different from the shared
    ``resolve_extra_tools`` two-state in ``lite/gym/utils/feedback/surface.py``:

      * ``None``    → the WHOLE ``action_subsets``-derived catalog. The catalog
        is not a fixed menu here, it is *derived* from the BrowserGym action
        subsets the benchmark registered, so "everything the action space
        offers" is the only default that tracks a re-registered subset. It is
        also what makes ``response`` (→ ``send_msg_to_user``) reachable: WA/VWA
        information-seeking tasks ANSWER through that call, so suppressing it
        by default makes every one of them structurally unanswerable.
      * ``[]``      → nothing (an operator opting the surface off).
      * ``[names]`` → only those, in the given order.
    """
    if extra_tools is None:
        return _tools_for_subsets(subsets)
    names = list(extra_tools)
    if not names:
        return []
    catalog = _tools_for_subsets(subsets)
    by_name = {tool_schema_name(schema): schema for schema in catalog}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(
            f"browsergym: unknown extra_tools {sorted(set(unknown))}; available for "
            f"action_subsets {tuple(subsets)}: {sorted(by_name)}"
        )
    seen: set[str] = set()
    out: list[dict] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        out.append(by_name[name])
    return out


# Canonical browser navigation action names (``LiteBrowserNavToolSet``) → BrowserGym's
# own function names, applied once at the env boundary so a model using the
# canonical ``@browser`` vocabulary (e.g. a Qwen3-VL student SFT'd on webgym)
# evaluates on browsergym with no action-space mismatch. ``goto``/``new_tab``
# already match bgym; ``response``/``terminate`` are handled in ``step`` (→
# send_msg_to_user / report_infeasible). Derived from the advertise-time map
# (single source of truth) by dropping the identity entries (goto/new_tab) — a
# rename in one map can't silently desync the other.
_CANONICAL_TO_BGYM: dict[str, str] = {
    canon: bgym for bgym, canon in _BGYM_NAV_TO_CANONICAL.items() if bgym != canon
}

_NONCANONICAL_INPUT_TOOL_NAMES = frozenset(
    set(_BGYM_NAV_TO_CANONICAL) - set(_BGYM_NAV_TO_CANONICAL.values())
    ) | frozenset(_BGYM_FINISH_TO_CANONICAL)

_BGYM_POINTER_VALID_ACTION_NAMES = frozenset({
    "click",
    "fill",
    "hover",
    "select_option",
    "dblclick",
    "press",
    "focus",
    "clear",
    "drag_and_drop",
    "upload_file",
})


@functools.cache
def _known_browsergym_tool_names() -> frozenset[str]:
    """BrowserGym-local standalone names the env may withhold.

    Materialized rather than left a predicate: the shared classifier takes a
    NAME SET, one kind of answer for every env. Resolved LAZILY (and cached)
    because ``browsergym`` must not be imported at module import — the same
    discipline ``envs/lite/osworld/main.py`` documents. Membership is derived
    with the exact rule the old predicate used, so it is unchanged name for
    name: anything ``_bgym_param_info`` can introspect off
    ``browsergym.core.action.functions``, plus the canonical nav aliases and
    the finish tools.
    """
    try:
        import browsergym.core.action.functions as F
    except ImportError:
        bgym_names: frozenset[str] = frozenset()
    else:
        bgym_names = frozenset(
            name for name in dir(F)
            if not name.startswith("_") and _bgym_param_info(name) is not None
        )
    return frozenset(_CANONICAL_TO_BGYM) | LiteFinishToolSet.get_tool_names() | bgym_names


def _format_bgym_call(name: str, args: dict[str, Any]) -> str | None:
    """Format a tool_call as positional BrowserGym Python code.

    Uses sig-derived parameter order from `_bgym_param_info` to emit
    `name(arg1, arg2, ...)` matching `HighLevelActionSet.example_action`
    output (single-quoted strings via repr). Returns None if `name` isn't
    a known BrowserGym action OR the first param isn't in args (lets the
    caller fall through to coord-mode handling for shape-shared names
    like `click`, `scroll` where args distinguish bid-style vs coord-style).
    """
    info = _bgym_param_info(name)
    if info is None:
        return None
    if info and info[0].name not in args:
        # First param missing → likely a different action shape (e.g.
        # cua-lite scroll(direction=..., amount=...) vs bgym scroll(delta_x, delta_y)).
        return None
    parts: list[str] = []
    for param in info:
        if param.name not in args:
            break  # caller relying on Python defaults for the tail
        parts.append(repr(args[param.name]))
    return f"{name}({', '.join(parts)})"

@dataclass
class BrowserGymConfig:
    """Configuration for a BrowserGym environment instance.

    Two parameter groups, both modeled on AgentLab's `GenericAgent`
    surface so that yaml configs can pick exact A3-paper / VL-text /
    vision-coord settings without env-side knob proliferation:

    - Group A (ObsFlags pass-through): the subset of
      `dynamic_prompting.ObsFlags` actually wired into
      `_build_obs_text` and BrowserGym's obs preprocessor. Defaults
      preserve today's screenshot-only behavior.
    - Group B (HighLevelActionSetArgs pass-through): drives which
      BrowserGym action functions exist + how `action_set.to_python_code`
      validates. Defaults preserve today's coord+chat+infeas+nav+tab
      action set.

    Protocol-side knobs (use_thinking / use_concrete_example / hints / etc.)
    live on the protocol and are not exposed here.

    See `lite/gym/envs/browsergym/README.md` (Action modes section) and
    `lite/agents/extensions/browsergym/protocol.py` module docstring for the
    rebuild-per-turn vs chat-style protocol rationale.
    """

    # BrowserGym task ID (gymnasium format, e.g. "miniwob.click-dialog")
    bgym_task_id: str = ""
    # Benchmark name (e.g. "miniwob", "webarena")
    benchmark: str = ""
    # Viewport size (pixels) — determines coordinate conversion
    viewport_width: int = 1920
    viewport_height: int = 1080
    # Whether to run browser headless
    headless: bool = True
    # Slow-mo for Playwright (ms)
    slow_mo: int | None = None
    # Playwright timeout (ms) — default 30s for Docker-hosted services
    timeout: int = 30000
    # Which DOM elements BrowserGym marks with bids. BrowserGym's default
    # preserves today's behavior; MiniWoB bid-only configs can opt into "all"
    # so SVG children (e.g. <text>) become targetable.
    tags_to_mark: Literal["all", "standard_html"] = "standard_html"
    # Additional task kwargs passed to BrowserGym (escape hatch for any
    # browsergym task param without a dedicated field). Merged UNDER the flat
    # hint fields below in ``_task_kwargs`` (explicit keys here win).
    task_kwargs: dict[str, Any] = field(default_factory=dict)

    # ─── WebArena/VWA closed-world goal hints — flat yaml knobs ───────────────
    # Forwarded into the browsergym task at make-time (``_task_kwargs``),
    # NOT baked at registration, so a config flips them per run like any ObsFlag
    # (``env_kwargs: {with_homepage_hint: false}``). Benchmark-guarded: WA accepts
    # both, VWA accepts only ``with_na_hint`` (its task __init__ has no homepage
    # param — passing it TypeErrors), miniwob accepts neither.
    #
    # with_homepage_hint (default OFF — browsergym-faithful; opt in per-run via
    # ``env_kwargs: {with_homepage_hint: true}``): appends the $WA_HOMEPAGE
    # pointer — the homepage lists every site + /password.html creds — which is
    # what the ORIGINAL WebArena p_cot prompt carries. When ON it is the fix for
    # weak models wandering to PUBLIC urls (github.com/reddit.com) → validate()'s
    # unauthorized-url check instant-fails any tab whose netloc ∉ the self-hosted
    # site set; start.sh renders the homepage links to the live auto-picked ports
    # so the hint is actionable. Registration default stays OFF to match the
    # upstream browsergym default; the webarena eval configs flip it ON per-run
    # (env_kwargs.with_homepage_hint: true, all 9 yaml) to match the WebArena
    # p_cot baseline.
    #
    # with_na_hint (default OFF): would append "answer N/A if impossible". Left
    # off because its text is phrased for the ORIGINAL WebArena TEXT stop-action
    # (`stop [N/A]`), but our action space is tool calls — N/A must go through
    # report_infeasible(...) / send_msg_to_user("N/A"). A weak model reads the
    # note literally and emits the bare text "N/A" → no tool call → routed to
    # noop "unknown action" → episode never terminates, burning the whole step
    # budget (observed: tasks 103/349 looped bare-"N/A" for 27-30 turns →
    # truncation, reward 0). report_infeasible() still requires active
    # canonical terminate schema before execution.
    with_homepage_hint: bool = False
    with_na_hint: bool = False

    # ─── Group A — BrowserGym ObsFlags pass-through ──────────────────────────
    # False keeps browser vision+coord rollouts screenshot-only: task
    # instructions and action errors still render, but tab URL/title, AXTree,
    # HTML, and focused-element page context do not.
    include_page_context_text: bool = _INCLUDE_PAGE_CONTEXT_TEXT

    # Subset of AgentLab `dynamic_prompting.ObsFlags` actually consumed by
    # `_build_obs_text` and the BrowserGym observation pipeline. Defaults
    # keep screenshot-first browser defaults (use_screenshot=True, AXTree/HTML
    # flags off, page-context text off). Set use_screenshot=False +
    # use_ax_tree=True to get the agent-as-annotators / paper observation shape.
    #
    # Fields not listed here (use_history, use_action_history,
    # use_think_history, use_past_error_logs, use_diff, use_som, html_type)
    # are AgentLab-specific knobs we don't currently honor — note that
    # ``use_error_logs`` IS read by BrowserGym's obs preprocessor (controls
    # whether ``last_action_error`` surfaces in obs); we then route the
    # error string through ``BrowserGymEnv.step``'s ``prefix=`` arg.
    use_html: bool = False
    use_ax_tree: bool = False
    use_focused_element: bool = False
    use_error_logs: bool = False
    use_screenshot: bool = True
    # Set-of-Marks: overlay numbered bid boxes onto the per-step screenshot (via
    # BrowserGym's ``overlay_som`` + the obs ``extra_element_properties`` bboxes),
    # so a VISION model reads the bid off the image and emits BID actions
    # (``click('a23')``) — image input, bid output. Requires the DOM extraction
    # (the marks need ``set_of_marks`` + ``bbox`` per bid), so a SoM config MUST
    # keep ``skip_dom_extraction=False``. Pair with ``use_screenshot=true,
    # use_ax_tree=false`` + a bid action space (``action_subsets:[<bench>],
    # valid_actions:[]``). Default False = raw (un-marked) screenshot.
    use_som: bool = False
    extract_visible_tag: bool = True
    extract_clickable_tag: bool = True
    # NOTE: literal string, not bool — AgentLab `dynamic_prompting.py`
    # `class ObsFlags` declares `Literal["False","center","box"]`. Comparison
    # is `==` not bool truth, so the str matters. yaml callers must quote
    # it: `extract_coords: "False"`.
    extract_coords: Literal["False", "center", "box"] = "False"
    filter_visible_elements_only: bool = False
    # Vision fast-path: skip BrowserGym's per-step DOM-snapshot + merged-AXTree +
    # per-node extra-properties + element bid-marking (all O(page nodes), computed
    # in chromium over CDP) — upstream ``_get_obs`` runs them UNCONDITIONALLY even
    # when the agent only consumes the screenshot, which is the 22-32s env.step
    # bottleneck on big WA/VWA pages. A module-level monkeypatch
    # (``_patch_lightweight_obs``) short-circuits ``_get_obs`` to extract only the
    # screenshot + cheap page metadata, dropping env.step to ~1-2s.
    #
    # Explicit, INDEPENDENT opt-in (deliberately NOT auto-derived from
    # ``use_ax_tree``): kept decoupled so a future config can mix obs/action modes
    # freely — e.g. set-of-marks vision (``use_ax_tree=False`` but bids drawn on the
    # screenshot, which still needs the DOM bid-marking). True is SAFE ONLY for a
    # coordinate-only action space (skipping ``_pre_extract`` leaves elements
    # UNMARKED, so bid actions can't resolve). Default False = byte-identical to
    # upstream. Set ``skip_dom_extraction: true`` in pure-coordinate vision configs.
    skip_dom_extraction: bool = False

    # ─── Group B — HighLevelActionSetArgs pass-through ───────────────────────
    # Modeled on `browsergym.experiments.benchmark.HighLevelActionSetArgs`.
    # Defaults preserve coord-based action space + chat + infeas + nav + tab
    # so the screenshot-only default obs has a usable matching action space.
    action_subsets: tuple[str, ...] = ("coord", "chat", "infeas", "nav", "tab")
    multiaction: bool = False
    strict: bool = False
    # True matches the reference eval preset (DEFAULT_HIGHLEVEL_ACTION_SET_ARGS
    # sets retry_with_force=True for EVERY benchmark): on a Playwright
    # TimeoutError the action retries with force=True instead of failing the
    # step. Strictly recovery-only (can't fail a passing action) — keeps WA/VWA
    # scores comparable to the paper on obscured/animating Magento/gitlab elems.
    retry_with_force: bool = True
    demo_mode: Literal["off", "default", "all_blue", "only_visible_elements"] | None = None

    # Surfaced verbatim on ``env.metadata.valid_actions`` so the agent
    # adapter can filter / suppress its standard action enum. Set
    # explicitly in yaml (``env_kwargs.valid_actions: []``) when running
    # text-only / bid-only modes that should drop the LiteBrowser coord-
    # action wrapper — keeps the suppress signal visible at the yaml
    # surface instead of relying on env-side magic.
    #   None  → no filter (full coord+bid+chat+infeas+nav+tab surface)
    #   []    → suppress LiteBrowser coord wrapper entirely
    valid_actions: list[str] | None = None

# ---------------------------------------------------------------------------
# Default viewports per benchmark
# ---------------------------------------------------------------------------

# Fixed task seed registered for every task (forwarded as a ``register(seed=…)``
# kwarg → ``BrowserGymEnv(seed=…)``; see ``BrowserGymEnv.__init__``). 0 matches the
# BrowserGym reference's webarena/visualwebarena ``fixed_seeds=[0]``; for miniwob the
# reference randomizes, so any fixed value works — 0 keeps a single anchor across all
# three benchmarks. Override per-task via ``env_kwargs.seed``.
_DEFAULT_TASK_SEED = 0

_BENCHMARK_VIEWPORTS: dict[str, tuple[int, int]] = {
    # The SCREENSHOT-pixel space (= task viewport × deviceScaleFactor), NOT the
    # task's page viewport. ``_px`` converts CUA-Lite [0,1000] → these pixels,
    # then ``mouse_click`` routes through BrowserGym's ``map_coordinates`` which
    # divides by the same scale_factor to recover true page coords. So this must
    # equal round(task_viewport × scale_factor) for clicks to land correctly.
    #
    # miniwob: task viewport 332×214 (miniwob/base.py:41) × scale_factor 1.5
    #          (base.py:144 / core/observation.py) = 498×321 exactly (both
    #          integers — no rounding). DO NOT "fix" this to 332×214 — that
    #          breaks the ÷1.5 round-trip.
    # webarena/vwa: task viewport 1280×720, scale_factor 1.0 → (1280,720).
    #          DO NOT "unify" these to 1920×1080 — the task hard-codes a 1280×720
    #          viewport (webarena/task.py, visualwebarena/task.py) and we pass NO
    #          viewport override, so the screenshot stays 1280×720; denormalizing
    #          [0,1000] against 1920×1080 would scale every click ~1.5× off-page.
    "miniwob": (498, 321),
    "webarena": (1280, 720),
    "visualwebarena": (1280, 720),
}

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class BrowserGymEnv(LiteBaseEnv):
    """CUA-Lite wrapper for BrowserGym environments.

    Bridges BrowserGym's synchronous Playwright-based API to CUA-Lite's async
    interface using asyncio.to_thread(). Actions use LiteBrowserActionSpace
    with [0, 1000] normalized coordinates converted to pixel coordinates.
    """

    def __init__(
        self,
        *,
        config: BrowserGymConfig,
        max_steps: int | None = _MAX_STEPS,
        post_action_delay: float = _POST_ACTION_DELAY,
        seed: int | None = _SEED,
        extra_tools: list[str] | None = _EXTRA_TOOLS,
        cursor: bool = True,
        # ── internal ──
        use_fake: bool = False,
        **config_overrides: Any,
    ):
        # Yaml env_kwargs that name `BrowserGymConfig` fields override the
        # pre-built config (registered per-benchmark in `_register_benchmark`).
        # Lets a yaml flip e.g. `use_ax_tree: true, action_subsets: [bid]`
        # without re-registering. Unknown field names raise TypeError from
        # dataclasses.replace — the desirable strict-typo behavior.
        if config_overrides:
            overrides = dict(config_overrides)
            # NO display_resolution: browsergym's render size is FIXED by the task
            # (webarena/vwa hard-code 1280×720; miniwob via its own viewport) — we
            # never pass a viewport override to gym.make, so viewport_width/height
            # is only the coord-MATCH constant (= the task's screenshot size, set
            # per-benchmark in _register_benchmark). Accepting display_resolution
            # would silently mismatch the real render → off-by-scale misclicks.
            if "display_resolution" in overrides:
                raise ValueError(
                    "browsergym does not accept display_resolution: the render size "
                    "is task-fixed (not controllable); viewport_width/height is the "
                    "coord-match constant set per benchmark. See docs/envs.md."
                )
            config = dataclasses.replace(config, **overrides)
        self._config = config
        dom_dependent_subsets = sorted(
            set(self._config.action_subsets) & _DOM_DEPENDENT_ACTION_SUBSETS
        )
        if self._config.skip_dom_extraction and (
            dom_dependent_subsets or self._config.use_som
        ):
            raise ValueError(
                "browsergym skip_dom_extraction is only valid for coordinate-only "
                "vision configs; bid/SoM modes need DOM/BID extraction "
                f"(action_subsets={dom_dependent_subsets}, "
                f"use_som={self._config.use_som})."
            )
        # Deterministic task seed, fixed at registration (see ``_DEFAULT_TASK_SEED``).
        # Threaded into ``env.reset(seed=...)`` → BrowserGym's ``task_entrypoint(seed=...)``,
        # the single source of task randomness (core/env.py sets ``np_random=None``).
        # Registration supplies the fixed value via the ``register(..., seed=N)`` kwarg
        # — mirrors the cua-lite eval convention (androidworld/mobilegym store
        # ``self._seed`` from a registered seed kwarg). For webarena/visualwebarena the
        # task is a fixed config so the seed is inert (singleton ``random.choice``,
        # reference uses ``fixed_seeds=[0]``); it only selects the variant for miniwob,
        # whose reference randomizes per-repeat — fixing it here makes miniwob eval
        # reproducible. ``None`` → unseeded (legacy ``RandomState(None)``); yaml may
        # override via ``env_kwargs.seed``.
        self._seed = seed
        # max_steps is hardcoded per-benchmark at registration (10/30), so via
        # gym.make it is always a concrete int; None only on bare construction
        # (yaml `max_steps: null` default) → no step-count truncation (line below
        # guards None). Multi-benchmark → null + registration hardcode.
        self._max_steps = max_steps
        self._post_action_delay = post_action_delay
        self._cursor = cursor
        # BrowserGym BID/SoM native actions move by DOM/BID, not by Lite pixel
        # coordinates. Unless the coord subset is active, there is no reliable
        # env-owned pixel cursor to paint, so keep the screenshot raw.
        self._cursor_rendering_enabled = bool(cursor) and "coord" in set(self._config.action_subsets)
        # "Rendering is enabled" is NOT "we know where the pointer is" — the two
        # were conflated here and at reset, which painted an arrow at (0, 0), a
        # coordinate no pointer was ever placed at. Nothing has been established
        # before reset(), so the position is unknown and nothing may be painted.
        self._cursor_position_known = False
        self._use_fake = use_fake
        # extra_tools surface — browsergym's tri-state (see
        # `_extra_tool_schemas_for_subsets`):
        #   None   → ALL action_subset-derived tools (the shipped default)
        #   []     → none
        #   names  → explicit subset of that catalog
        # None is KEPT as None (not normalised to []) — the two are distinct.
        self._extra_tools = None if extra_tools is None else list(extra_tools)
        # Fail fast on unknown names, through the SAME resolver `_task_metadata`
        # uses (other envs validate by calling resolve_extra_tools in __init__ —
        # match that, so there is exactly one validator and one error string).
        _extra_tool_schemas_for_subsets(self._config.action_subsets, self._extra_tools)
        self._step_count = 0
        self._terminated = False
        # L2 dead-backend fail-fast counter + Bug-A last-good-obs cache — init
        # here too (not only in reset()) because the fake/test reset path
        # early-returns before reset()'s init.
        self._consecutive_failed_steps = 0
        self._last_obs: dict[str, Any] | None = None
        self._last_action_execution_error: str | None = None
        self._env: Any = None  # BrowserGym gymnasium env
        self._instruction: str = ""
        # Track cursor position for mouse_down/mouse_up (BrowserGym needs coords).
        # Meaningless until reset() parks the real pointer — guarded by
        # ``_cursor_position_known``, which is False until then.
        self._cursor_x: float = 0.0
        self._cursor_y: float = 0.0
        # Per-instance single-thread executor — each env gets its own Playwright
        # instance in its own thread, enabling true parallelism across envs.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="bgym"
        )
        self._closed = False

    @staticmethod
    def _task_metadata(
        benchmark: str,
        bgym_task_id: str,
        *,
        action_subsets: tuple[str, ...] | list[str],
        viewport: tuple[int, int],
        extra_tools: list[str] | None = _EXTRA_TOOLS,
    ) -> LiteCUAMetadata:
        """Same-source metadata builder:
        registration passes its loop locals; the live instance passes from
        ``self._config`` (see ``_runtime_metadata``).

        * ``extra_tool_schemas`` — ``extra_tools`` selects out of the catalog
          ``action_subsets`` derives via ``_tools_for_subsets``
          (introspects the live BrowserGym ``HighLevelActionSet``;
          mouse_*/keyboard_* never surface as tools). Tri-state: ``None`` (the
          bundled default) exposes the WHOLE catalog, ``[]`` suppresses the
          surface, a list selects only those names. It is applied HERE rather
          than only on the live instance so the registered copy is
          byte-identical to what default construction produces. A yaml
          ``env_kwargs.extra_tools`` re-selects at make time through
          ``_runtime_metadata``.
        * ``valid_actions`` — the ``BrowserGymConfig.valid_actions`` default
          (None = no filter); explicit env_kwargs surface amended live.
        * ``others`` — the task record: benchmark/bgym_task_id/viewport plus the
          STATIC WA/VWA task facts (``sites``/``llm_as_a_judge``/``mutating``/
          ``depends_on`` — see ``_wa_vwa_task_facts``; callers compose their own
          runtime filters, the env never bakes a deploy-dependent
          exclude_reason) and the shared-backend isolation gate's
          ``conflict_keys`` (server-mode-only opt-in contract, see
          lite/gym/remote/conflict.py; keyed OFF by default → () → gate no-op,
          populated under BROWSERGYM_CONFIG=isolation — policy in isolation.py).
        """
        others: dict[str, Any] = {
            "benchmark": benchmark,
            "bgym_task_id": bgym_task_id,
            "viewport": tuple(viewport),
        }
        sites, llm_judge, mutating = _wa_vwa_task_facts(benchmark, bgym_task_id)
        if sites:
            others["sites"] = sites
        if llm_judge:
            others["llm_as_a_judge"] = True
        # ``mutating`` is a STATIC task fact (see _wa_vwa_task_facts), surfaced
        # UNCONDITIONALLY for every WA/VWA task so an operator can split the eval
        # into a residue-immune read pass and a write pass — regardless of whether
        # the env-server runs strict isolation. (Non-WA/VWA → no shared writable
        # backend → key omitted; absent reads as falsy = non-mutating.)
        if benchmark in ("webarena", "visualwebarena"):
            others[OTHERS_MUTATING] = mutating
            # ``depends_on`` (BrowserGym's curated run-order parents) lets the
            # rollout dispatch writers in topological order so the suite needs
            # ONE reset at the start instead of one per writer. Static fact;
            # absent when the optional dependency metadata isn't installed.
            deps = isolation.task_depends_on(benchmark, bgym_task_id)
            if deps:
                # list(...): task_depends_on returns the module-cached list —
                # sever the registered/live others from it.
                others["depends_on"] = list(deps)
        conflict_keys, _gate_mutating = isolation.conflict_keys_and_mutating(
            benchmark, bgym_task_id
        )
        if conflict_keys:
            others[OTHERS_CONFLICT_KEYS] = conflict_keys
        return LiteCUAMetadata(
            dims=("browser", "use"),
            extra_tool_schemas=_extra_tool_schemas_for_subsets(
                action_subsets, extra_tools
            ),
            valid_actions=None,  # BrowserGymConfig.valid_actions default
            others=others,
        )

    def _runtime_metadata(self) -> LiteCUAMetadata:
        md = self._task_metadata(
            self._config.benchmark,
            self._config.bgym_task_id,
            action_subsets=self._config.action_subsets,
            viewport=(self._config.viewport_width, self._config.viewport_height),
            # extra_tools name filter: None → the whole catalog; [] suppresses
            # extras; [names] selects a subset.
            extra_tools=self._extra_tools,
        )
        # env_kwargs amendment:
        #   * valid_actions — explicit env_kwargs surface (see
        #     ``BrowserGymConfig.valid_actions``): text-only / bid-only yamls set
        #     ``env_kwargs.valid_actions: []`` to suppress the adapter's
        #     LiteBrowser coord-action wrapper.
        return dataclasses.replace(
            md,
            valid_actions=resolve_valid_actions(
                self._config.valid_actions,
                env_name="browsergym",
                platform="browser",
            ),
        )

    async def _run_in_thread(self, fn, *args):
        """Run a sync function in this env's dedicated Playwright thread."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, fn, *args)

    async def reset(self) -> LiteEnvObservation:
        if self._env is not None:
            await self._run_in_thread(self._env.close)
            self._env = None

        self._step_count = 0
        self._terminated = False
        self._consecutive_failed_steps = 0  # L2 dead-backend fail-fast counter
        # INVARIANT: what we composite is the REAL pointer position, or we
        # composite nothing. Playwright exposes no mouse-position getter
        # (``playwright._impl._input.Mouse`` is a write-only channel: move/down/
        # up/click/wheel, no state), so we ESTABLISH the position instead of
        # guessing it — ``_create_and_reset`` issues a real ``page.mouse.move``
        # to ``_center_px()`` below, and only then may ``_cursor_position_known``
        # become True. Until that move lands the pointer is nowhere we can name,
        # so the flag stays False and ``_overlay_cursor`` draws nothing.
        self._cursor_x, self._cursor_y = self._center_px()
        self._cursor_position_known = False

        if self._use_fake:
            return self._fake_reset()

        _check_env_vars(self._config.benchmark)

        def _create_and_reset():
            import gymnasium as gym
            from browsergym.core.action.highlevel import HighLevelActionSet

            # We emit bare Python calls (e.g. `mouse_click(18.2, 38.9, "left")`) in
            # `_to_bgym_code`. BrowserGym's default `execute_python_code` has none
            # of the action-function names in scope, so passing them straight
            # through raises NameError. Route them through
            # `HighLevelActionSet.to_python_code`, whose `python_includes` prepends
            # the required imports + function definitions before exec.
            #
            # Subsets are driven by Group B (BrowserGymConfig.action_subsets) —
            # default ("coord","chat","infeas","nav","tab") matches today's
            # behavior; A3 paper / VL text-only configs override to ("bid",).
            #   coord  — mouse_click / move / scroll / keyboard_*
            #   bid    — click(bid) / fill(bid, value) / keyboard_press / etc.
            #   chat   — send_msg_to_user (response)
            #   infeas — report_infeasible (terminate with status=failure)
            #   nav    — canonical back / goto (we expose these as extra_tools)
            #   tab    — new_tab / switch_tab / close_tab (WA/VWA/AB)
            action_set = HighLevelActionSet(
                subsets=list(self._config.action_subsets),
                multiaction=self._config.multiaction,
                strict=self._config.strict,
                retry_with_force=self._config.retry_with_force,
                demo_mode=self._config.demo_mode,
            )

            # Forward the flat hint knobs into the browsergym task, benchmark-
            # guarded by which params each task __init__ accepts (WA: both; VWA:
            # na_hint only — no homepage param; miniwob: neither). Explicit
            # ``task_kwargs`` entries win over the flat-field derived ones.
            _hint_kwargs: dict[str, Any] = {}
            if self._config.benchmark == "webarena":
                # Install the judge-model override before WA's evaluators import
                # helper_functions (binds the judge symbol by value). WA has no
                # heavy deferred import like VWA, but its llm_fuzzy_match judge
                # needs the same _LLM_JUDGE_MODEL override — otherwise a pure-WA
                # run silently falls back to upstream's hard-coded gpt-4-1106.
                _install_judge_model_override()
                _hint_kwargs = {
                    "with_homepage_hint": self._config.with_homepage_hint,
                    "with_na_hint": self._config.with_na_hint,
                }
            elif self._config.benchmark == "visualwebarena":
                # Deferred (torch-free import contract): the heavy
                # ``browsergym.visualwebarena`` import is done now, on first VWA
                # reset, NOT at module load — it both populates BrowserGym's
                # gymnasium registry (so the ``gym.make`` below resolves) and
                # installs the judge-model override. Idempotent.
                _ensure_visualwebarena_imported()
                _hint_kwargs = {"with_na_hint": self._config.with_na_hint}
            _task_kwargs = {**_hint_kwargs, **self._config.task_kwargs}

            env = _with_playwright(lambda: gym.make(
                f"browsergym/{self._config.bgym_task_id}",
                headless=self._config.headless,
                slow_mo=self._config.slow_mo,
                timeout=self._config.timeout,
                tags_to_mark=self._config.tags_to_mark,
                action_mapping=action_set.to_python_code,
                task_kwargs=_task_kwargs,
                # Skip gymnasium's per-step PassiveEnvChecker: it re-validates the
                # obs against the space every step (pure overhead for a vetted env)
                # and warns on the lightweight vision obs (empty axtree placeholder,
                # see skip_dom_extraction). We read obs fields directly, not via the
                # space.
                disable_env_checker=True,
            ))
            # Vision fast-path: skip the O(nodes) DOM/AXTree obs extraction when
            # explicitly opted in (``skip_dom_extraction``, decoupled from obs mode
            # so set-of-marks-style configs stay possible). Patch is idempotent + a
            # no-op unless the per-env flag is set.
            if self._config.skip_dom_extraction:
                _patch_lightweight_obs()
                env.unwrapped._cua_lite_skip_dom = True
            # Fixed seed (registration-time, see ``self._seed``) → the task
            # entrypoint's only randomness source. None preserves legacy unseeded
            # behavior (RandomState(None)). Makes each task_id reproducible.
            obs, info = _with_playwright(lambda: env.reset(seed=self._seed))
            # Make the default TRUE rather than assumed: park the real pointer at
            # the viewport centre. This is the coordinate the retired
            # CursorOverlayWrapper painted (normalized 500,500), so the frame
            # still looks like the reference run — the difference is that the
            # browser's pointer is now actually there instead of being asserted
            # to be.
            #
            # Gated on cursor rendering: a mouse move dispatches a real mousemove
            # in the page (:hover CSS + JS pointer handlers fire on whatever sits
            # at the centre), so configs that paint no cursor (bid/SoM) must not
            # pay that perturbation for nothing — they stay untouched.
            #
            # Ordering note: BrowserGym captured obs["screenshot"] at the end of
            # env.reset(), microseconds BEFORE this move, so any hover styling the
            # move induces first appears on the next frame. The COORDINATE we
            # composite is nonetheless the live pointer's real position; only the
            # hover repaint lags by one frame. Re-extracting the obs here to close
            # that gap would cost a second full DOM+AXTree extraction per episode.
            if self._cursor_rendering_enabled:
                cx, cy = self._center_px()
                _with_playwright(lambda: env.unwrapped.page.mouse.move(cx, cy))
            return env, obs, info

        try:
            env, obs, info = await self._run_in_thread(_create_and_reset)
        except Exception as e:
            # A backend (gitlab / reddit / classifieds) can still flap or restart
            # AFTER ensure_services' all-sites gate passed (e.g. gitlab's sidekiq
            # recycling during its long warmup), so this task's reset goto may still
            # hit a cold service. A connection-level error during the reset goto is
            # TRANSIENT (service warming), not a task failure: raise a retriable 503
            # so the client retries /reset until it's HTTP-ready.
            if _is_backend_warming_error(e):
                from lite.gym.errors import CapacityExhausted
                _evict_services_cache(self._config.benchmark)
                raise CapacityExhausted(
                    what=(
                        f"browsergym.{self._config.benchmark} backend not "
                        f"HTTP-ready during reset (warming up): {str(e)[:160]}"
                    ),
                    retry_after_s=20.0,
                    layer="env_internal",
                )
            raise
        # The move above landed (the thread returned without raising), so the
        # tracked centre is now a fact about the live page, not a guess.
        self._cursor_position_known = self._cursor_rendering_enabled
        self._env = env
        # No-op and rejected actions reuse the current page rendering, so keep
        # the latest real BrowserGym observation available after reset.
        self._last_obs = obs
        self._instruction = _extract_instruction(obs)
        # Screenshot only when requested. Default (use_screenshot=True) keeps
        # today's behavior; text-only configs (`use_screenshot=False, use_ax_tree=True`)
        # null the PNG and rely on the AXTree text from _build_obs_text.
        screenshot = (
            await asyncio.to_thread(
                self._encode_screenshot,
                obs,
            )
            if self._config.use_screenshot else None
        )
        # VWA goal image(s) — task-defining input ("find THIS product"), not the
        # per-turn page screenshot. They ride the ordinary reset metadata as
        # base64 PNGs, so the same value crosses the env-server wire frame and
        # the in-process path unchanged. The generic loop passes the metadata
        # through untouched; the ``visualwebarena.goal_image`` agent decodes them
        # once at turn 0 and re-shows them on EVERY turn, in both text+goal
        # (mixed: screenshot=None) and vision (screenshot=page) modes. A task may
        # carry several goal images (``config["image"]`` is a list) — all are
        # transported. See ``lite/agents/extensions/browsergym/goal_image.py``.
        goal_images_b64 = _extract_goal_images_b64(obs)
        metadata = self._build_page_context_metadata(obs)
        if goal_images_b64:
            metadata["goal_images_b64"] = goal_images_b64
        text = await asyncio.to_thread(self._build_obs_text, obs, self._instruction)
        return LiteEnvObservation(
            image=screenshot, text=text, metadata=metadata or None,
        )

    async def _render_feedback_frame(
        self, render_obs: dict[str, Any] | None,
    ) -> bytes | None:
        """The one frame a turn that executed nothing still owes the model.

        Per-action frames come from the obs each executed action returned; this
        is only the fallback for a turn that produced none.
        """
        if not self._config.use_screenshot:
            return None
        if render_obs is not None:
            return await asyncio.to_thread(self._encode_screenshot, render_obs)
        if self._env is not None:
            return await self._take_screenshot()
        return None

    async def _render_feedback_text(
        self,
        render_obs: dict[str, Any] | None,
        action_error_section: str | None,
    ) -> str | None:
        """The turn's page-context text — one per turn, never per action."""
        text = await asyncio.to_thread(
            self._build_obs_text,
            render_obs,
            action_error_section,
        )
        if not self._config.use_screenshot and not text:
            # Text-only config with nothing to say: the model would otherwise
            # get an empty result and no hint that the action produced nothing.
            text = (
                "Action execution failed (no observation returned). "
                "Continue with a different approach."
            )
        return text

    async def step(self, actions: list[RuntimeEnvAction]) -> LiteEnvStepResult:
        input_actions = actions
        result_call_ids = ordered_tool_call_ids(input_actions)
        metadata = self.metadata
        actions, ingress_errors = prepare_env_tool_calls(
            actions,
            metadata,
        )
        terminated = False
        reward = None
        last_obs = None
        last_result_call_id: str | None = None
        last_result_action_name: str | None = None
        self._last_action_execution_error = None
        current_result_call_ids: set[str] = set()
        # Did this step actually invoke the BrowserGym backend? Only an EXECUTED
        # action's None result signals a dead backend (L2). A no-op step
        # (wait/screenshot/cursor_position/unknown) never calls the backend, so
        # its `last_obs is None` must NOT count toward the dead-backend counter.
        bgym_action_attempted = False
        executed_actions: list[LiteExecutedAction] = []
        action_errors: dict[str, ToolErrorFeedback] = dict(ingress_errors)
        extra_tool_schemas = metadata.extra_tool_schemas
        # One frame PER EXECUTED ACTION, in action order. BrowserGym captures the
        # page inside its own ``env.step``, so each executed action already owns a
        # distinct frame in the obs it returned -- this loop just stops discarding
        # every frame but the last. ``wait``/``screenshot``/``cursor_position``
        # never reach the backend and so own no obs; they read the page directly
        # instead, because the frame count must not depend on WHAT the actions
        # were. Actions REJECTED before execution still get no frame -- there is
        # no screen state after an action that never ran.
        step_screenshots: list[bytes] = []

        async def record_rejected_frame() -> None:
            """One frame for a slot that never reached the backend.

            Gated on the SAME policy an executed slot obeys: under
            ``use_screenshot: false`` an executed GUI action produces no frame
            either, so a rejected one must not produce more than its siblings.
            """
            if self._config.use_screenshot and self._env is not None:
                step_screenshots.append(await self._take_screenshot())

        async def record_action_frame(obs: dict[str, Any] | None) -> None:
            if obs is not None and self._config.use_screenshot:
                step_screenshots.append(
                    await asyncio.to_thread(self._encode_screenshot, obs)
                )

        for action, result_call_id in actions:
            name = action["name"]
            args = action["arguments"]

            # A fault ingress already attributed to the MODEL. It must not be
            # dispatched -- without this arm an ingress-rejected child would run.
            # It is a GUI slot, so it earns the current observation and its own
            # frame (R2a + R3); the frame repeats the screen it did not change.
            if rejected_reason := runtime_rejected_reason(action):
                if result_call_id:
                    append_feedback(
                        action_errors, result_call_id, current_feedback(rejected_reason),
                    )
                executed_actions.append({
                    "call": "noop",
                    "args": {"name": name, "reason": rejected_reason},
                })
                await record_rejected_frame()
                continue

            if name in _NONCANONICAL_INPUT_TOOL_NAMES:
                if result_call_id:
                    action_errors[result_call_id] = current_feedback(
                        unavailable_action_message(name)
                    )
                executed_actions.append({
                    "call": "noop",
                    "args": {"name": name, "reason": "noncanonical input tool"},
                })
                continue

            tool_availability = classify_standalone_tool_call(
                action,
                _known_browsergym_tool_names(),
                extra_tool_schemas,
                is_standalone_action_tool=_is_browsergym_standalone_tool_call,
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

            if (
                not _is_lite_action_call(name, args)
                and name in LiteDesktopActionSet.get_action_names()
                and not is_active_extra_tool_call(action, extra_tool_schemas)
            ):
                if result_call_id:
                    action_errors[result_call_id] = current_feedback(
                        unsupported_action_message(name)
                    )
                executed_actions.append({
                    "call": "noop",
                    "args": {"name": name, "reason": "inactive extra tool"},
                })
                continue

            if _is_lite_action_call(name, args):
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
                terminated = True
                status = args.get("status", "success")
                reason = args.get("reason", "")
                if status == "failure" and reason:
                    code = f'report_infeasible("{_escape(reason)}")'
                else:
                    code = f'send_msg_to_user("{_escape(reason or "Task completed")}")'
                last_obs = await self._execute_bgym_action(code)
                await record_action_frame(last_obs)
                last_result_call_id = result_call_id
                last_result_action_name = name
                bgym_action_attempted = True
                # A MODEL-emitted call that ENDED the episode gets no
                # continuation observation: devs/migration/verify.py forbids a
                # tool result for a terminal call. An INTERNAL finish call has no
                # ``call_id`` -- the loop-detect wrapper's ``terminate`` carries
                # the intercepted NON-finish model call's id as
                # ``_result_call_id`` -- and that call must still be answered.
                if result_call_id and not action.get("call_id"):
                    current_result_call_ids.add(result_call_id)
                executed_actions.append({"call": code})
                break

            if name == "response":
                terminated = True
                text = args.get("text", "")
                code = f'send_msg_to_user("{_escape(text)}")'
                last_obs = await self._execute_bgym_action(code)
                await record_action_frame(last_obs)
                last_result_call_id = result_call_id
                last_result_action_name = name
                bgym_action_attempted = True
                if result_call_id and not action.get("call_id"):
                    current_result_call_ids.add(result_call_id)
                executed_actions.append({"call": code})
                break

            # Convert CUA-Lite action to BrowserGym Python code
            try:
                code = self._to_bgym_code(name, args)
            except MODEL_ACTION_ERROR_TYPES as e:
                record_model_action_error(
                    action_errors,
                    result_call_id,
                    e,
                    action_name=name,
                )
                executed_actions.append({"call": "noop", "args": {"name": name, "reason": str(e)}})
                continue
            if code:
                last_obs = await self._execute_bgym_action(code)
                await record_action_frame(last_obs)
                last_result_call_id = result_call_id
                last_result_action_name = name
                bgym_action_attempted = True
                if result_call_id:
                    current_result_call_ids.add(result_call_id)
                executed_actions.append({"call": code})
            elif name in ("wait", "screenshot", "cursor_position"):
                # No bgym code, so no obs -- but these DID execute and owe a
                # frame. Read the page directly: no ``env.step``, so no step
                # counter or reward side effect. ``_render_feedback_frame``
                # already reads the page this way for a turn that executed
                # nothing. The frame carries no SoM overlay (that is derived
                # from an obs), which is visible only when ``use_som`` is on
                # and one of these is the LAST action of a batch.
                if self._config.use_screenshot and self._env is not None:
                    step_screenshots.append(await self._take_screenshot())
                if result_call_id:
                    current_result_call_ids.add(result_call_id)
            elif name not in ("wait", "screenshot", "cursor_position"):
                if result_call_id:
                    feedback_fn = (
                        current_feedback
                        if is_lite_action_name_or_action_batch_tool_name(name)
                        else error_only_feedback
                    )
                    message = (
                        unsupported_action_message(name)
                        if is_lite_action_name_or_action_batch_tool_name(name)
                        else unknown_tool_message(name)
                    )
                    action_errors[result_call_id] = feedback_fn(message)
                executed_actions.append({"call": "noop", "args": {"name": name, "reason": "unknown action"}})

        self._step_count += 1

        # L2 dead-backend fail-fast: ``last_obs is None``
        # means the BrowserGym step itself failed to return an observation — a
        # dead WA/VWA stack / unreachable miniwob / crashed page. (A logical
        # action error is DIFFERENT: it returns an obs WITH last_action_error, so
        # last_obs is not None — this counter never trips on bad actions.) One is
        # a transient blip; N CONSECUTIVE means the backend is gone and every
        # further step just burns a model turn on a stale screenshot for a
        # silent reward-0. Truncate loudly. A real observation resets the count.
        _ff = isolation._POOL_UNREACHABLE_FAILED_STEPS
        if bgym_action_attempted:
            # Only a step that actually called the backend tells us its health.
            if last_obs is None:
                self._consecutive_failed_steps += 1   # executed an action, got no obs → backend down
            else:
                self._consecutive_failed_steps = 0     # got a real obs → backend alive
        # else: a no-op step (wait/screenshot/cursor/unknown) is NEUTRAL — it
        # neither proves the backend dead nor alive, so the counter is unchanged.
        _pool_unreachable = _ff > 0 and self._consecutive_failed_steps >= _ff

        # No-op and rejected actions do not advance BrowserGym, so render from
        # the cached page while reward and action-error state still read the real
        # step result.
        render_obs = last_obs
        if last_obs is None and not bgym_action_attempted and self._last_obs is not None:
            render_obs = self._last_obs
        elif last_obs is not None:
            self._last_obs = last_obs   # refresh the cache from a real step

        # Extract reward from last BrowserGym step result
        if last_obs is not None:
            reward = last_obs.get("reward")
        elif bgym_action_attempted and self._last_action_execution_error:
            record_tool_execution_error(
                action_errors,
                last_result_call_id,
                self._last_action_execution_error,
                action_name=last_result_action_name,
            )

        # Surface BrowserGym's last_action_error so the agent knows when an
        # action was rejected (e.g. element not visible, navigation aborted).
        # Empty string in upstream means "no error", which we map to None.
        # Store it in LiteToolResult.error; the shared projection boundary adds
        # the model-visible ``## Error from previous action:`` label exactly
        # once. Env observation text remains page context.
        action_error: str | None = None
        if last_obs is not None:
            err = last_obs.get("last_action_error")
            if err:
                action_error = str(err).strip() or None
                # Match AgentLab's `Error` element (dynamic_prompting.py): cap the
                # Playwright ``Call log:`` to its first 10 lines so an
                # obscured-element / timeout error doesn't flood the prompt with a
                # 100+-line retry log.
                if action_error and "Call log:" in action_error:
                    head, logs = action_error.split("Call log:", 1)
                    logs = "\n".join(logs.split("\n")[:10])
                    action_error = f"{head}Call log:\n{logs}".strip()
                record_tool_execution_error(
                    action_errors,
                    last_result_call_id,
                    action_error,
                    action_name=last_result_action_name,
                )
        # Let the page settle, then render this turn's payload. FRAMES are
        # per-action and already captured above; TEXT is per-turn, always
        # rendered here from ``render_obs``.
        if self._post_action_delay > 0:
            await asyncio.sleep(self._post_action_delay)
        # A turn where no action reached the backend (no-op step, dead backend,
        # everything rejected) captured nothing, and still owes the model one
        # current observation. It renders from ``render_obs`` (the cache: the
        # last-good obs on a no-op/unknown step, else this step's real obs), NOT
        # ``last_obs``. The cached obs' screenshot comes from BrowserGym's
        # ``extract_screenshot`` (CDP ``Page.captureScreenshot`` at
        # ``deviceScaleFactor``), so it matches the per-benchmark
        # screenshot-pixel viewport (miniwob 498×321 = 332×214 × 1.5). The
        # ``_take_screenshot`` fallback uses Playwright ``page.screenshot()``
        # which IGNORES ``_bgym_scale_factor`` → CSS-pixel 332×214, a DIFFERENT
        # frame. Feeding that on a no-op step made absolute-pixel agents
        # (GPT/Claude computer-use) report coords in 332×214 while ``_px``
        # denormalized against 498×321 → every click undershot ~1.5× and miniwob
        # reward collapsed. Only fall back to ``_take_screenshot`` when there is
        # NO cached obs at all (dead backend / pre-first-step), where a
        # mismatched frame is moot.
        if not step_screenshots:
            fallback_frame = await self._render_feedback_frame(render_obs)
            if fallback_frame is not None:
                step_screenshots.append(fallback_frame)
        text = await self._render_feedback_text(render_obs, None)
        metadata = self._build_page_context_metadata(render_obs)

        # OR with self._terminated so natural BrowserGym termination via
        # native bid actions (send_msg_to_user, report_infeasible — set
        # bg_terminated=True inside _execute_bgym_action) propagates as
        # terminated=True even when the local switch wasn't hit.
        final_terminated = terminated or self._terminated
        self._terminated = final_terminated
        if _pool_unreachable and not final_terminated:
            logger.error(
                "browsergym(%s): %d consecutive steps with no observation — "
                "backend appears UNREACHABLE; truncating trajectory instead of "
                "grinding to max_steps as a silent reward-0",
                self._config.benchmark, self._consecutive_failed_steps,
            )
        truncated = not final_terminated and (
            (self._max_steps is not None and self._step_count >= self._max_steps)
            or _pool_unreachable
        )
        return build_tool_results_from_decisions(
            LiteEnvStepResult(
                reward=reward if (final_terminated or truncated) else None,
                terminated=final_terminated,
                truncated=truncated,
                info={EXECUTED_ACTIONS_INFO_KEY: executed_actions,
                      "pool_unreachable": _pool_unreachable},
            ),
            ordered_call_ids=result_call_ids,
            continue_call_ids=current_result_call_ids,
            images=step_screenshots,
            text=text,
            metadata=metadata or None,
            feedback=action_errors,
        )

    async def close(self) -> None:
        # Idempotent: the executor is single-use and gone after the first close.
        if self._closed:
            return
        self._closed = True

        # Tear down the bgym env AND this thread's Playwright driver (a node
        # subprocess) on the worker thread, in one task, before discarding the
        # executor. browsergym has NO env-server reaper, so a driver left
        # unstopped leaks for the whole process lifetime (direct mode) — there
        # is no backstop. max_workers=1 ⇒ this runs on the same thread that
        # start()'d pw (sync Playwright objects are thread-affine).
        def _teardown():
            # Independent guards: a failing env.close() must NOT skip pw.stop()
            # — leaking pw leaks a node subprocess + chromium for the whole
            # process lifetime (browsergym has NO env-server reaper backstop).
            if self._env is not None:
                try:
                    _with_playwright(lambda: self._env.close())
                except Exception as e:
                    logger.warning("BrowserGym env.close failed: %s", e)
            pw = getattr(_thread_local, "pw", None)
            if pw is not None:
                pw.stop()
                _thread_local.pw = None
        try:
            await self._run_in_thread(_teardown)
        except Exception as e:
            logger.warning("BrowserGym close failed: %s", e)
        self._env = None
        self._executor.shutdown(wait=False)

    # -----------------------------------------------------------------------
    # Internal: action execution
    # -----------------------------------------------------------------------

    async def _execute_bgym_action(self, code: str) -> dict[str, Any] | None:
        """Execute a BrowserGym action string and return merged obs+reward dict."""
        if self._env is None:
            self._last_action_execution_error = "BrowserGym env is None"
            return None
        self._last_action_execution_error = None
        try:
            def _step():
                return _with_playwright(lambda: self._env.step(code))
            obs, reward, bg_terminated, bg_truncated, info = await self._run_in_thread(_step)
            if bg_terminated:
                self._terminated = True
            result = dict(obs)
            result["reward"] = reward
            result["bg_terminated"] = bg_terminated
            return result
        except Exception as e:
            logger.warning("BrowserGym action failed: %s — code: %s", e, code[:200])
            self._last_action_execution_error = str(e)
            return None

    async def _take_screenshot(self) -> bytes:
        """Take a screenshot directly from the Playwright page.

        Raises on failure so the caller sees the real error instead of a
        generic "No screenshot returned".
        """
        if self._env is None:
            raise RuntimeError("Cannot take screenshot: BrowserGym env is None")
        def _screenshot():
            return _with_playwright(lambda: self._env.unwrapped.page.screenshot())
        raw = await self._run_in_thread(_screenshot)
        return await asyncio.to_thread(self._overlay_cursor, raw)

    def _encode_screenshot(self, obs: dict[str, Any]) -> bytes:
        png = _encode_screenshot_maybe_som(
            obs.get("screenshot"),
            obs.get("extra_element_properties"),
            self._config.use_som,
        )
        return self._overlay_cursor(png)

    def _center_px(self) -> tuple[float, float]:
        """Viewport centre in CSS pixels — the pointer's parking spot at reset.

        Same point the retired CursorOverlayWrapper painted (normalized 500,500
        through ``norm_to_pixel``), and the same pixel basis ``_px`` maps model
        coordinates onto, so the parked pointer and every later action share one
        coordinate space.
        """
        return (self._config.viewport_width / 2.0, self._config.viewport_height / 2.0)

    def _overlay_cursor(self, png: bytes) -> bytes:
        """INVARIANT: the composited coordinate is the real pointer position, or
        nothing is composited — ``_cursor_position_known`` is True only after a
        real ``page.mouse.move`` (reset's parking move, or a coord action) put
        the pointer at ``(_cursor_x, _cursor_y)``, and False whenever the pointer
        went somewhere only the DOM knows (BID/SoM actions)."""
        if not self._cursor or not self._cursor_rendering_enabled or not self._cursor_position_known:
            return png
        return overlay_cursor_px(
            png,
            int(round(self._cursor_x)),
            int(round(self._cursor_y)),
        )

    # -----------------------------------------------------------------------
    # Internal: build LiteEnvObservation from BrowserGym obs dict
    # -----------------------------------------------------------------------

    def _build_obs_text(self, obs_dict: dict[str, Any] | None, prefix: str | None) -> str | None:
        """Compose the textual observation seen by the agent.

        Combines (in this order, omitting empty parts):
          - prefix: task instruction on reset, or BrowserGym `last_action_error` on step.
          - AXTree text (if use_ax_tree=True), via BrowserGym's flatten_axtree_to_str
            with the same flag plumbing AgentLab uses (extract_visible_tag, etc.).
          - Pruned-HTML text (if use_html=True), via flatten_dom_to_str.

        Returns None if everything is empty.

        Sync (CPU-bound — flatten_axtree_to_str on a 75K-token shopping_admin
        AXTree is 0.3-1.5s). Callers MUST offload via ``asyncio.to_thread`` so
        this doesn't block the event loop and serialize all concurrent envs.
        """
        cfg = self._config
        parts: list[str] = []
        if prefix:
            parts.append(prefix)

        if not cfg.include_page_context_text:
            return "\n\n".join(parts) or None

        # AgentLab-aligned ``## Currently open tabs:`` block — surfaces the
        # tab list so the model knows which ``switch_tab(N)`` index to call
        # for multi-tab benches (WebArena / VisualWebArena can spawn
        # additional tabs via ``new_tab`` or task setup).
        # Mirrors `agentlab.agents.dynamic_prompting.Tabs` (dynamic_prompting.py
        # in upstream AgentLab) byte-for-byte: ``Tab N (active tab):\n  Title: ...\n  URL: ...``.
        if obs_dict is not None:
            urls = obs_dict.get("open_pages_urls")
            titles = obs_dict.get("open_pages_titles")
            active = obs_dict.get("active_page_index")
            if urls is not None and titles is not None:
                # active is a numpy array of length 1; coerce to int.
                try:
                    active_idx = int(active[0]) if hasattr(active, "__getitem__") else int(active)
                except (TypeError, ValueError, IndexError):
                    active_idx = 0
                tab_lines = ["## Currently open tabs:"]
                for i, (url, title) in enumerate(zip(urls, titles)):
                    suffix = " (active tab)" if i == active_idx else ""
                    tab_lines.append(f"Tab {i}{suffix}:\n    Title: {title}\n    URL: {url}")
                parts.append("\n".join(tab_lines))

        if obs_dict is not None and cfg.use_ax_tree and obs_dict.get("axtree_object") is not None:
            from browsergym.utils.obs import flatten_axtree_to_str
            try:
                axtree_txt = flatten_axtree_to_str(
                    obs_dict["axtree_object"],
                    extra_properties=obs_dict.get("extra_element_properties"),
                    with_visible=cfg.extract_visible_tag,
                    with_clickable=cfg.extract_clickable_tag,
                    with_center_coords=(cfg.extract_coords == "center"),
                    with_bounding_box_coords=(cfg.extract_coords == "box"),
                    # SVG elements such as <text> are merged into AXTree as
                    # bid-bearing generic parents whose StaticText children hold
                    # the visible label. If tags_to_mark="all", keep those
                    # generic nodes so bid-only MiniWoB can target them.
                    skip_generic=(cfg.tags_to_mark != "all"),
                    filter_visible_only=cfg.filter_visible_elements_only,
                )
                parts.append(f"## AXTree:\n{axtree_txt}")
            except Exception as e:
                logger.warning("flatten_axtree_to_str failed: %s", e)

        if obs_dict is not None and cfg.use_html and obs_dict.get("dom_object") is not None:
            from browsergym.utils.obs import flatten_dom_to_str, prune_html
            try:
                # Match the reference (demo_agent / AgentLab `html_type="pruned_html"`):
                # prune the flattened DOM (strip script/style/link/br + unwrap
                # bid-only structural div/span/p) so the agent sees the same HTML
                # the paper does, not the raw, noisier flattened DOM.
                dom_txt = prune_html(flatten_dom_to_str(
                    obs_dict["dom_object"],
                    extra_properties=obs_dict.get("extra_element_properties"),
                    with_visible=cfg.extract_visible_tag,
                    with_clickable=cfg.extract_clickable_tag,
                    with_center_coords=(cfg.extract_coords == "center"),
                    with_bounding_box_coords=(cfg.extract_coords == "box"),
                    filter_visible_only=cfg.filter_visible_elements_only,
                ))
                parts.append(f"## HTML:\n{dom_txt}")
            except Exception as e:
                logger.warning("flatten_dom_to_str failed: %s", e)

        # Mirror AgentLab's `## Focused element:\nbid='X'` block — paper-aligned
        # framing that helps the model track which input it just typed into.
        # Previously surfaced only via metadata, but chat templates strip that
        # (qwen3_vl/agent.py:build_generation_prompt). Render inline as text so
        # the model actually sees it.
        if obs_dict is not None and cfg.use_focused_element:
            bid = obs_dict.get("focused_element_bid")
            if bid:
                parts.append(f"## Focused element:\nbid='{bid}'")

        return "\n\n".join(parts) if parts else None

    def _build_page_context_metadata(self, obs_dict: dict[str, Any] | None) -> dict[str, Any]:
        """Preserve cheap page context for logs when prompt text is suppressed."""
        if self._config.include_page_context_text or obs_dict is None:
            return {}

        metadata: dict[str, Any] = {}
        urls = obs_dict.get("open_pages_urls")
        titles = obs_dict.get("open_pages_titles")
        if urls is not None:
            metadata["open_pages_urls"] = list(urls)
        if titles is not None:
            metadata["open_pages_titles"] = list(titles)

        active = obs_dict.get("active_page_index")
        if active is not None:
            try:
                metadata["active_page_index"] = (
                    int(active[0]) if hasattr(active, "__getitem__") else int(active)
                )
            except (TypeError, ValueError, IndexError):
                pass

        url = obs_dict.get("url")
        if url:
            metadata["url"] = str(url)
        bid = obs_dict.get("focused_element_bid")
        if bid:
            metadata["focused_element_bid"] = str(bid)
        return metadata

    # -----------------------------------------------------------------------
    # Internal: CUA-Lite → BrowserGym action translation
    # -----------------------------------------------------------------------

    def _to_bgym_code(self, name: str, args: dict[str, Any]) -> str | None:
        """Translate a LiteBrowserActionSpace action to BrowserGym Python code.

        Dispatches across two action paths (see ``Action modes`` section of
        ``lite/gym/envs/browsergym/README.md``):
          (1) coord (vision pipelines): name="click" with {coordinate}
              → mouse_click(x, y); cua-lite-side scroll/move/drag/key
              shapes get pixel-mapped here.
          (2) bid (text+AXTree pipelines): name="click" with {bid}
              → click('a47'). All native BrowserGym actions (scroll,
              fill, hover, select_option, goto, switch_tab, ...,
              report_infeasible) flow through `_format_bgym_call`,
              which uses the function's true signature so we never
              drift from BrowserGym's API.

        Distinguished by argument shape: bgym `click` requires `bid`,
        cua-lite `click` carries `coordinate`; bgym `scroll` takes
        `delta_x`, cua-lite `scroll` takes `direction`/`amount`. The
        generic dispatcher returns None when the first sig param is
        absent so the coord branch below catches it.
        """
        vw = self._config.viewport_width
        vh = self._config.viewport_height

        # Canonical browser nav (back/forward/switch_tab/close_tab) → bgym names,
        # so a model speaking the canonical @browser vocabulary round-trips here.
        # goto/new_tab already match; response/terminate
        # handled in step(). Arg names for the aliased nav verbs already match bgym
        # (go_back/go_forward/tab_close take none; tab_focus takes an index).
        name = _CANONICAL_TO_BGYM.get(name, name)

        def _px(coord: list[int | float]) -> tuple[float, float]:
            """Convert [0, 1000] normalized coords to SUB-PIXEL floats
            (Playwright accepts them; as_float is this env's contract)."""
            return norm_to_pixel(coord, vw, vh, as_float=True, clamp=False,
                                 on_malformed="raise")

        # ─── path 2: native BrowserGym dispatch ──────────────────────────
        # Auto-derived from inspect.signature on `browsergym.core.action.functions`.
        # Covers click(bid), fill, hover, select_option, goto, go_back,
        # go_forward, tab_focus, new_tab, tab_close, send_msg_to_user,
        # report_infeasible, keyboard_press, scroll(dx,dy), press, dblclick,
        # focus, clear, drag_and_drop, upload_file — i.e. everything across
        # the bid / webarena / visualwebarena / miniwob_all presets.
        # Returns None when args don't fit (e.g. cua-lite-shape
        # `click(coordinate=...)` lacks `bid`), letting coord branch run.
        bgym_code = _format_bgym_call(name, args)
        if bgym_code is not None:
            if name in _BGYM_POINTER_VALID_ACTION_NAMES:
                self._cursor_position_known = False
            return bgym_code

        # ─── path 1: coord-mode (LiteBrowserActionSpace) ─────────────────
        if name == "click":
            coord = args.get("coordinate", [500, 500])
            x, y = _px(coord)
            button = args.get("button", "left")
            clicks = args.get("clicks", 1)
            self._cursor_x, self._cursor_y = x, y
            self._cursor_position_known = True
            if clicks >= 2:
                return f'mouse_dblclick({x}, {y}, "{button}")'
            return f'mouse_click({x}, {y}, "{button}")'

        if name == "mouse_move":
            coord = args.get("coordinate", [500, 500])
            x, y = _px(coord)
            self._cursor_x, self._cursor_y = x, y
            self._cursor_position_known = True
            return f"mouse_move({x}, {y})"

        if name == "mouse_down":
            button = args.get("button", "left")
            x, y = self._cursor_x, self._cursor_y
            return f'mouse_down({x}, {y}, "{button}")'

        if name == "mouse_up":
            button = args.get("button", "left")
            x, y = self._cursor_x, self._cursor_y
            return f'mouse_up({x}, {y}, "{button}")'

        if name == "drag":
            end = args["coordinate"]
            tx, ty = _px(end)
            start = args.get("start_coordinate")
            # start_coordinate optional → drag from the tracked cursor (updated by
            # click/mouse_move), matching SandboxBaseEnv "drag from current cursor".
            fx, fy = _px(start) if start else (self._cursor_x, self._cursor_y)
            self._cursor_x, self._cursor_y = tx, ty
            self._cursor_position_known = True
            return f"mouse_drag_and_drop({fx}, {fy}, {tx}, {ty})"

        if name == "scroll":
            coord = args.get("coordinate")
            direction = args.get("direction", "down")
            amount = args.get("amount", 3)
            scroll_px = amount * 100  # approximate pixels per scroll unit
            if direction == "up":
                dx, dy = 0, -scroll_px
            elif direction == "left":
                dx, dy = -scroll_px, 0
            elif direction == "right":
                dx, dy = scroll_px, 0
            else:  # "down" (default)
                dx, dy = 0, scroll_px
            # BrowserGym's coord subset exposes ONLY `scroll_at(x, y, dx, dy)`
            # (wheel at a point) — there is NO bare `scroll` in it, and the action
            # set is built with multiaction=False so a `mouse_move\nscroll` pair is
            # rejected. Emit the single-action `scroll_at`, anchored at the model's
            # coord or the tracked cursor when none is given. (Emitting bare
            # `scroll` raised NameError → swallowed → scrolling was a silent no-op
            # in every default (coord) config.)
            if coord:
                x, y = _px(coord)
                self._cursor_x, self._cursor_y = x, y
                self._cursor_position_known = True
            else:
                x, y = self._cursor_x, self._cursor_y
            return f"scroll_at({x}, {y}, {dx}, {dy})"

        if name == "type":
            text = args.get("text", "")
            return f'keyboard_type("{_escape(text)}")'

        if name == "key":
            keys = project_model_keys(
                args.get("keys", []),
                action_name=name,
                backend="playwright",
            )
            # project_model_keys(backend="playwright") already resolved them to
            # Playwright key names (raises loudly on a bad key), and rejects an
            # empty list -- so ``keys`` is non-empty here by construction.
            key_str = "+".join(keys)
            return f'keyboard_press("{_escape(key_str)}")'

        if name == "key_down":
            keys = project_model_keys(
                args.get("keys", []),
                action_name=name,
                backend="playwright",
            )
            return "\n".join(f'keyboard_down("{_escape(pk)}")' for pk in keys)

        if name == "key_up":
            keys = project_model_keys(
                args.get("keys", []),
                action_name=name,
                backend="playwright",
            )
            return "\n".join(f'keyboard_up("{_escape(pk)}")' for pk in keys)

        if name == "back":
            return "go_back()"

        if name == "wait":
            coerce_model_duration(args.get("duration", 1.0), action_name="wait")
            return None  # no bgym code; the loop frames it directly

        if name in ("screenshot", "cursor_position"):
            return None  # no bgym code; the loop frames it directly

        logger.warning("Unknown action: %s(%s), skipping", name, args)
        return None

    # -----------------------------------------------------------------------
    # Fake env for testing
    # -----------------------------------------------------------------------

    def _fake_reset(self) -> LiteEnvObservation:
        """Return a fake observation for testing without a real browser."""
        from PIL import Image
        # Stand in for the real reset's parking move: there is no Playwright
        # mouse here, so assert the post-move state the fake browser represents —
        # pointer at the viewport centre, position established. reset() already
        # set _cursor_x/_cursor_y; only the "known" bit is ours to flip.
        self._cursor_position_known = self._cursor_rendering_enabled
        img = Image.new("RGB", (self._config.viewport_width, self._config.viewport_height), (200, 200, 200))
        screenshot = self._overlay_cursor(encode_png(img))
        self._instruction = "Fake task instruction for testing"
        self._last_obs = {
            "goal": self._instruction,
            "screenshot": screenshot,
            "open_pages_urls": ("http://browsergym.fake/",),
            "open_pages_titles": ("BrowserGym Fake",),
            "active_page_index": [0],
            "last_action_error": "",
        }
        metadata = self._build_page_context_metadata(self._last_obs)
        return LiteEnvObservation(
            image=screenshot,
            text=self._instruction,
            metadata=metadata or None,
        )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_README = "lite/gym/envs/browsergym/README.md"

# Default URLs matching the ports baked into WebArena's published Docker images
# and scripts/start.sh. `setdefault` in _check_env_vars means users on localhost
# with default ports don't need to source start.sh to set MINIWOB_URL / WA_* /
# VWA_*; remote / custom-port users set the vars themselves and setdefault
# no-ops. See scripts/start.sh "Overriding ports" README section.
# Default port → URL is yaml-sourced (server_kwargs): each port is a
# CFG constant (the PREFERRED port, auto-reallocated if busy). miniwob's port is
# NOT 8080 (webgym's OmniBoxes node defaults to 8080 — a latent collision); the
# yaml reserves miniwob_port: 7560. See _ensure_miniwob_singleton.
_DEFAULT_ENV_VARS: dict[str, dict[str, str]] = {
    "miniwob": {"MINIWOB_URL": f"http://localhost:{_MINIWOB_PORT_DEFAULT}/miniwob/"},
    "webarena": {
        "WA_SHOPPING": f"http://localhost:{_SHOPPING_PORT_DEFAULT}/",
        "WA_SHOPPING_ADMIN": f"http://localhost:{_SHOPPING_ADMIN_PORT_DEFAULT}/admin",
        "WA_REDDIT": f"http://localhost:{_REDDIT_PORT_DEFAULT}",
        "WA_GITLAB": f"http://localhost:{_GITLAB_PORT_DEFAULT}",
        "WA_WIKIPEDIA": f"http://localhost:{_WIKIPEDIA_PORT_DEFAULT}{_WIKIPEDIA_PATH}",
        "WA_MAP": f"http://localhost:{_MAP_PORT_DEFAULT}",
        "WA_HOMEPAGE": f"http://localhost:{_HOMEPAGE_PORT_DEFAULT}",
    },
    "visualwebarena": {
        "VWA_SHOPPING": f"http://localhost:{_SHOPPING_PORT_DEFAULT}/",
        "VWA_REDDIT": f"http://localhost:{_REDDIT_PORT_DEFAULT}",
        "VWA_WIKIPEDIA": f"http://localhost:{_WIKIPEDIA_PORT_DEFAULT}{_WIKIPEDIA_PATH}",
        "VWA_HOMEPAGE": f"http://localhost:{_HOMEPAGE_PORT_DEFAULT}",
        "VWA_CLASSIFIEDS": f"http://localhost:{_CLASSIFIEDS_PORT_DEFAULT}",
        "VWA_CLASSIFIEDS_RESET_TOKEN": _CLASSIFIEDS_RESET_TOKEN,
    },
}

_REQUIRED_ENV_VARS: dict[str, list[str]] = {
    bench: list(defaults.keys()) for bench, defaults in _DEFAULT_ENV_VARS.items()
}
# WA/VWA score their ``fuzzy_match`` tasks with an LLM judge (``llm_judge_model``
# via OPENAI_API_KEY + OPENAI_BASE_URL). Require the key up-front so a missing judge
# HARD-FAILS at reset (the task ERRs and is retried via --max-attempts) instead of
# silently scoring those tasks 0.0 mid-run — consistent with webgym's judge, which
# also raises rather than commit a corrupted reward. The var name is hard-coded (not
# the configurable ``*_EVAL_API_KEY_VAR`` peers use) because upstream's vendored
# ``openai_utils`` binds ``os.environ["OPENAI_API_KEY"]`` by name.
_REQUIRED_ENV_VARS["webarena"].append("OPENAI_API_KEY")
_REQUIRED_ENV_VARS["visualwebarena"].append("OPENAI_API_KEY")

def _check_env_vars(benchmark: str) -> None:
    """Raise a clear error if required env vars for a benchmark are missing."""
    import os
    for k, v in _DEFAULT_ENV_VARS.get(benchmark, {}).items():
        os.environ.setdefault(k, v)
    required = _REQUIRED_ENV_VARS.get(benchmark, [])
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        raise EnvironmentError(
            f"Missing environment variables for {benchmark}: {', '.join(missing)}\n"
            f"  See setup instructions: {_README}"
        )


# ---------------------------------------------------------------------------
# ensure_services(env_id) — registry lifecycle hook
# ---------------------------------------------------------------------------
#
# Auto-launch the auxiliary service each browsergym sub-env needs so
# clients (and ``gym.make`` callers in direct mode) don't have to
# remember ``source scripts/start.sh``. See
# :func:`lite.gym.registry.ensure_services` for the protocol.
#
# Per-benchmark services:
#   * miniwob          — host-wide shared HTTP server on ``MINIWOB_PORT``
#                        (default 7560) serving ``miniwob-plusplus/miniwob/html``
#   * webarena         — Shopping / Shopping Admin / Forum / GitLab /
#                        Wikipedia docker containers
#   * visualwebarena   — webarena services + Classifieds
# Footprint is fixed per service (not per env instance).

_BG_SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
_BG_REPO_ROOT = project_root()  # repo root (was parents[4]; marker-found, move-proof)

# Set of (benchmark) already brought up in this process. ``ensure_services``
# below short-circuits on hit.
_services_started: set[str] = set()


def _evict_services_cache(benchmark: str) -> None:
    """Self-heal seam: a service that DIED mid-run (not merely cold) would
    otherwise 503 forever. TWO caches short-circuit ensure — this module's
    ``_services_started`` AND the registry-level ensure cache (which runs
    FIRST, so evicting only the local one is dead code). Clearing both makes
    the next create re-enter the idempotent start.sh."""
    _services_started.discard(benchmark)
    # NB: must import the MODULE — ``from lite.gym import registry`` resolves
    # to the re-exported Registry INSTANCE, which has no invalidate_services.
    from lite.gym.registry import invalidate_services
    invalidate_services(f"browsergym.{benchmark}")

# Per-benchmark probes — each (url, needle) pair must return 200 AND
# contain the needle substring in the body. Needle-matching distinguishes
# the real service from a port-squatter (e.g. Apache returning 200 from
# someone else's deploy on the same port). For services where any 2xx
# response is sufficient (Gitlab login page, shopping landing), needle
# can be empty — status-only check.
#: Per-benchmark probes used by ``ensure_services`` and the per-env
#: import-time sanity check. Scoped to the SHARED service that EVERY
#: task of that benchmark needs to function — not the union of every
#: sub-service. Per-task failures (e.g. a webarena task that requires
#: the Map service which isn't auto-started) surface naturally at
#: ``reset()`` time with the real error from browsergym/playwright;
#: we don't want to block the entire benchmark on a sub-service that
#: only some tasks need.
#:
#: Each entry is ``(url, needle)``: ``needle != ""`` means the response
#: body must literally contain that substring (defeats port-squatters
#: that happen to return 200). ``needle == ""`` accepts any 2xx (used
#: when the legitimate service returns a generic landing page that's
#: hard to fingerprint cheaply).
_PROBE_SPECS: dict[str, list[tuple[str, str]]] = {
    "miniwob": [
        # Every miniwob task page references ``../core/core.js`` (the
        # framework's task driver). An Apache / nginx port-squatter
        # returning a directory listing or its own index.html won't have
        # this substring. Mirrors ``_miniwob_alive`` in start.sh.
        ("http://localhost:{MINIWOB_PORT}/miniwob/ascending-numbers.html",
         "core/core.js"),
    ],
    # webarena / visualwebarena are NOT keyed here: their readiness is the full
    # local-stack check in ``_wa_stack_up``: a benchmark is ready iff ALL
    # its served local sites are up, incl the slow gitlab, so ``available: true`` can
    # never lie while a site is still booting). See the ``_service_up`` dispatch below.
}

#: WA/VWA shared local sites probed by :func:`_wa_stack_up` — (name, PORT env var,
#: default port, needle | None, acceptable HTTP codes). gitlab serves a **302** redirect
#: to /users/sign_in when ready (and 5xx/conn-refused while booting), so it accepts
#: ``{200, 302}``; shopping keeps its storefront needle as a port-squatter guard. The
#: PORT env vars carry the auto-picked ports (``_auto_pick_webarena_ports``). Classifieds
#: (VWA-only) and external Map are NOT here — classifieds stays a separate
#: ``_classifieds_up`` gate (callers), Map is never gated (map tasks filtered upstream).
_WA_SITE_SPECS: list[tuple[str, str, int, str | None, tuple[int, ...]]] = [
    ("shopping", "SHOPPING_PORT", _SHOPPING_PORT_DEFAULT, "One Stop Market", (200,)),
    ("shopping_admin", "SHOPPING_ADMIN_PORT", _SHOPPING_ADMIN_PORT_DEFAULT, None, (200, 302)),
    ("forum", "REDDIT_PORT", _REDDIT_PORT_DEFAULT, None, (200,)),
    ("gitlab", "GITLAB_PORT", _GITLAB_PORT_DEFAULT, None, (200, 302)),
    ("wikipedia", "WIKIPEDIA_PORT", _WIKIPEDIA_PORT_DEFAULT, None, (200,)),
]


def _wa_site_up(url: str, needle: str | None, ok_codes: tuple[int, ...]) -> bool:
    """One WA site is up iff it answers with an acceptable code (and, if given, the
    needle is in the body). ``_NoRedirectHandler`` turns a 3xx into an ``HTTPError``,
    so gitlab's ready-state 302 surfaces there — accept it only when ``ok_codes``
    allows and no needle is required."""
    import urllib.error
    import urllib.request
    opener = urllib.request.build_opener(_NoRedirectHandler())
    try:
        with opener.open(urllib.request.Request(url), timeout=3) as r:
            if r.status not in ok_codes:
                return False
            if needle:
                body = r.read(8192).decode("utf-8", errors="replace")
                if needle not in body:
                    return False
            return True
    except urllib.error.HTTPError as e:  # redirect (via _NoRedirectHandler) or 4xx/5xx
        return needle is None and e.code in ok_codes
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def _wa_stack_up(benchmark: str) -> bool:
    """All of the WA/VWA stack's **local** sites reachable. gitlab is the slow long
    pole (~5-15 min); gating on it is the readiness contract — ``available`` must
    not flip true until every site a task could hit is actually serving."""
    import os as _os
    for _name, port_env, default, needle, ok_codes in _WA_SITE_SPECS:
        port = _os.environ.get(port_env, str(default))
        if not _wa_site_up(f"http://localhost:{port}/", needle, ok_codes):
            return False
    return True


def _wa_down_sites(benchmark: str) -> list[str]:
    """Names of the WA/VWA local sites NOT yet serving — for a clear ``warming: …``
    503 message so a rollout's error.txt says *which* site is cold (almost always
    ``gitlab``, the slow long pole) instead of a vague timeout."""
    import os as _os
    down = [
        name
        for name, port_env, default, needle, ok_codes in _WA_SITE_SPECS
        if not _wa_site_up(
            f"http://localhost:{_os.environ.get(port_env, str(default))}/", needle, ok_codes
        )
    ]
    if benchmark == "visualwebarena" and not _classifieds_up():
        down.append("classifieds")
    return down


def _service_up(benchmark: str) -> bool:
    """Readiness probe for a benchmark's backend(s). WA/VWA → the full local-stack
    check (:func:`_wa_stack_up`); miniwob → its single needle probe. ``True`` for an
    unknown/probe-less benchmark."""
    if benchmark in ("webarena", "visualwebarena"):
        return _wa_stack_up(benchmark)
    specs = _PROBE_SPECS.get(benchmark, [])
    if not specs:
        return True
    import os as _os
    import urllib.error
    import urllib.request
    for url_template, needle in specs:
        url = url_template.format(
            # 7560 = miniwob shared-singleton preferred port (see
            # _ensure_miniwob_singleton); only used if MINIWOB_PORT is unset.
            MINIWOB_PORT=_os.environ.get("MINIWOB_PORT", str(_MINIWOB_PORT_DEFAULT)),
        )
        try:
            req = urllib.request.Request(url)
            # Don't auto-follow redirects: a 301 from Apache → /miniwob
            # (without trailing slash) would otherwise be silently
            # accepted as success.
            opener = urllib.request.build_opener(_NoRedirectHandler())
            with opener.open(req, timeout=3) as r:
                if r.status != 200:
                    return False
                if needle:
                    body = r.read(8192).decode("utf-8", errors="replace")
                    if needle not in body:
                        return False
        except (urllib.error.URLError, OSError, TimeoutError):
            return False
    return True


def _homepage_up() -> bool:
    """True iff the WebArena homepage Flask answers with start.sh's
    ``_HOMEPAGE_PROBE_NEEDLE`` ("WebArena").

    The homepage is in no ``_PROBE_SPECS`` row and start.sh launches it LAST
    (after the slow WA/map tail), so a cold boot whose 5 shared sites turn
    ready between ensure retries used to early-return with the homepage
    never launched → every ``__HOMEPAGE__``-goal VWA task and
    ``with_homepage_hint`` run failed as endless retriable 503s. Gating the
    benchmark on this probe keeps the ensure retry loop re-entering the
    idempotent start.sh until the Flask is actually serving."""
    import urllib.request
    port = os.environ.get("HOMEPAGE_PORT", str(_HOMEPAGE_PORT_DEFAULT))
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/", timeout=3) as r:
            return r.status == 200 and b"WebArena" in r.read(65536)
    except Exception:
        return False


def _classifieds_up() -> bool:
    """True if THIS env-server's Classifieds web container answers HTTP.

    Classifieds is deliberately NOT in ``_PROBE_SPECS["visualwebarena"]`` (we
    don't fold it into ``_service_up``'s shared-site gate — see ``_wa_stack_up``), but
    it DOES still need to be *started* for the classifieds tasks. ``_service_up``
    gates on the shared WA sites only, so once those are warm the early-return used to
    skip start.sh entirely → ``start_classifieds`` never ran → the classifieds
    compose stack was never created → every classifieds reset hit
    ``ERR_CONNECTION_REFUSED`` and the client retried until its deadline. This
    probe lets ``_ensure_services`` keep (re-)running start.sh until classifieds
    is actually up, WITHOUT making it a hard gate (a still-cold classifieds
    surfaces as a retriable 503, not a benchmark-blocking 501).

    Probes the auto-picked CLASSIFIEDS_PORT (mirrors start.sh's published port).
    Any 2xx/3xx is alive — the osclass landing 302-redirects to /index.php.
    """
    import os as _os
    import urllib.error
    import urllib.request
    port = _os.environ.get("CLASSIFIEDS_PORT", str(_CLASSIFIEDS_PORT_DEFAULT))
    try:
        with urllib.request.urlopen(
            f"http://localhost:{port}/", timeout=3
        ) as r:
            return 200 <= r.status < 400
    except urllib.error.HTTPError as e:
        # urlopen FOLLOWS redirects by default, so osclass's 302 normally resolves
        # to a 200 in the try above; this branch only catches a 3xx that urlopen
        # surfaced as an error (e.g. no Location) — still counts as "alive".
        return 300 <= e.code < 400
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """urlopen handler that turns redirects into a urllib HTTPError so
    ``_service_up`` treats them as failures (the redirector is probably
    a different service squatting the port)."""
    def http_error_301(self, req, fp, code, msg, headers):  # type: ignore[override]
        return None
    http_error_302 = http_error_301
    http_error_303 = http_error_301
    http_error_307 = http_error_301
    http_error_308 = http_error_301


def _benchmark_of(env_id: str) -> str | None:
    """``browsergym.miniwob`` → ``miniwob``; umbrella ``browsergym`` → None."""
    parts = env_id.split(".", 1)
    if len(parts) != 2 or parts[0] != "browsergym":
        return None
    return parts[1]


# ── miniwob: a host-wide SHARED SINGLETON with conflict-avoidance ────────────
# miniwob serves READ-ONLY static HTML, so — unlike the per-env-server WA/VWA
# docker stacks — every env-server on the host shares ONE instance. Two design
# goals that must BOTH hold (an earlier version had only one each and was buggy):
#
#   (1) CONFLICT-AVOIDANCE: the port is auto-picked if the preferred one is
#       held by a foreign service — like webarena's _auto_pick. Preferred port
#       is 7560 (a dedicated reservation), NOT 8080: webgym's OmniBoxes node
#       ALSO defaults to 8080, a latent collision a fixed 8080 would re-trigger.
#   (2) SINGLETON / NO LEAK: all env-servers converge on the SAME instance via a
#       host-wide registry file (flock-serialised). Without it, each server that
#       auto-picked independently spawned its OWN http.server and — because
#       miniwob is started nohup+disown (PPid==1 even when live, so no orphan
#       sweep can tell live from leaked) — they accumulated until the whole
#       range exhausted. With the registry, a second server READS the live
#       port and REUSES it; only when the registered one is dead does the next
#       server re-pick + re-record. At most one miniwob ever exists.
#
# No env-server ever KILLS miniwob (BrowserGymServices.shutdown/reap skip it) —
# that would cross-kill a co-resident server's shared backend. The only teardown
# is the manual host-level scripts/cleanup.sh.
# (_MINIWOB_PORT_DEFAULT / _MINIWOB_PORT_RANGE are yaml-sourced in the CFG block
#  at the top of this module — miniwob_port / miniwob_port_range.)


def _miniwob_url(port: int | str) -> str:
    return f"http://localhost:{port}/miniwob/"


def _miniwob_registry_paths() -> tuple[Path, Path]:
    """(registry json, lock file) under the shared .tmp dir every env-server +
    slime container sees (same inode as the port-reservation file)."""
    from lite.gym.utils.backend.ports import _SHARED_TMP
    return (_SHARED_TMP / "miniwob-singleton.json",
            _SHARED_TMP / "miniwob-singleton.lock")


@contextmanager
def _miniwob_singleton_lock():
    """flock the host-wide miniwob registry so concurrent env-servers serialise
    the reuse-or-start decision and converge on one instance."""
    import fcntl
    _, lock_path = _miniwob_registry_paths()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch(exist_ok=True)
    fd = open(lock_path)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


def _read_miniwob_registry() -> int | None:
    reg_path, _ = _miniwob_registry_paths()
    try:
        return int(json.loads(reg_path.read_text())["port"])
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _write_miniwob_registry(port: int) -> None:
    reg_path, _ = _miniwob_registry_paths()
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    reg_path.write_text(json.dumps({"port": port}))


def _miniwob_alive_on(port: int) -> bool:
    """True iff a REAL miniwob (not a foreign port-squatter) serves on ``port``
    — reuses ``_service_up``'s needle-checked probe by pointing it at ``port``."""
    import os as _os
    saved = _os.environ.get("MINIWOB_PORT")
    _os.environ["MINIWOB_PORT"] = str(port)
    try:
        return _service_up("miniwob")
    finally:
        if saved is None:
            _os.environ.pop("MINIWOB_PORT", None)
        else:
            _os.environ["MINIWOB_PORT"] = saved


def _set_miniwob_env(port: int) -> None:
    import os as _os
    _os.environ["MINIWOB_PORT"] = str(port)
    _os.environ["MINIWOB_URL"] = _miniwob_url(port)


def _pick_miniwob_port() -> int:
    """Pick the shared singleton's port: adopt a live miniwob already on the
    preferred port (e.g. an orphan whose registry entry was lost — never spawn a
    duplicate beside it), else the preferred port if free, else the first free
    port in the conflict-avoidance range."""
    from lite.gym.utils.backend.ports import _is_port_free
    lo, hi = _MINIWOB_PORT_RANGE
    if _miniwob_alive_on(_MINIWOB_PORT_DEFAULT) or _is_port_free(_MINIWOB_PORT_DEFAULT):
        return _MINIWOB_PORT_DEFAULT
    for p in range(lo, hi + 1):
        if _is_port_free(p):
            return p
    from lite.gym.errors import EnvDepsMissingError
    raise EnvDepsMissingError(
        what=f"no free miniwob port in [{lo},{hi}] — too many foreign services",
        install="free a port in that range, or set MINIWOB_PORT=<free-port>",
        see="lite/gym/envs/browsergym/README.md",
    )


def _run_miniwob_start_sh() -> None:
    """Invoke start.sh to bring up miniwob on the already-resolved MINIWOB_PORT.
    Raises EnvDepsMissingError on failure (mirrors the generic ensure path)."""
    start_sh = _BG_SCRIPTS_DIR / "start.sh"
    try:
        subprocess.run(
            ["bash", str(start_sh), "miniwob"],
            cwd=str(_BG_REPO_ROOT),
            capture_output=True, text=True, timeout=180, check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        from lite.gym.errors import EnvDepsMissingError
        stderr = getattr(e, "stderr", "") or ""
        raise EnvDepsMissingError(
            what=f"browsergym.miniwob HTTP server failed to start: {stderr.strip()[:200]}",
            install="uv run --no-sync bash lite/gym/envs/browsergym/scripts/install.sh miniwob",
            see="lite/gym/envs/browsergym/README.md",
        )


def _ensure_miniwob_singleton() -> None:
    """Bring up (or reuse) THE host-wide shared miniwob singleton. flock-
    serialised so concurrent env-servers converge on one instance."""
    import os as _os
    # Operator override: a pinned MINIWOB_PORT means operator-owned lifecycle on
    # a private corpus — honour it verbatim, no registry, no auto-pick.
    if _os.environ.get("MINIWOB_PORT"):
        _os.environ.setdefault("MINIWOB_URL", _miniwob_url(_os.environ["MINIWOB_PORT"]))
        if not _service_up("miniwob"):
            _run_miniwob_start_sh()
        return
    with _miniwob_singleton_lock():
        # 1. Reuse a live registered singleton (the common multi-server path).
        reg = _read_miniwob_registry()
        if reg is not None:
            _set_miniwob_env(reg)
            if _service_up("miniwob"):
                return
        # 2. None live → pick a free port (conflict-avoidance) and start it
        #    under the lock so a racing second server can't start a duplicate.
        port = _pick_miniwob_port()
        _set_miniwob_env(port)
        if not _service_up("miniwob"):
            _run_miniwob_start_sh()
            if not _service_up("miniwob"):
                from lite.gym.errors import EnvDepsMissingError
                raise EnvDepsMissingError(
                    what=f"browsergym.miniwob started on port {port} but probe failed",
                    install=(
                        "uv run --no-sync bash "
                        "lite/gym/envs/browsergym/scripts/install.sh miniwob"
                    ),
                    see="lite/gym/envs/browsergym/README.md",
                )
        # 3. Record so the next env-server reuses THIS instance (no fan-out).
        _write_miniwob_registry(port)


# ── WebArena / VisualWebArena per-env-server port isolation ──────────────────
# Each WA host port busy on its yaml default is reallocated into the yaml fallback
# range (_WA_FALLBACK_START/END, sourced in the CFG block at the top), so a second
# env-server's WA stack lands on its own ports and never collides with the first's.
# (miniwob, by contrast, is a fixed-port host singleton with its own range — see
# _ensure_miniwob_singleton.)
#
# One row per WA host port that start.sh publishes. ``port_var`` is the env var
# start.sh reads (``${SHOPPING_PORT:-7770}`` etc.); ``default`` is the yaml
# PREFERRED port (CFG constant); ``url_vars`` are the in-process base_url env vars
# (WA_* / VWA_*) the Python client reads from ``_DEFAULT_ENV_VARS`` via
# ``_check_env_vars``'s ``setdefault``. When a port is reallocated we MUST set
# those url_vars in ``os.environ`` so ``setdefault`` no-ops onto the real (picked)
# port — start.sh's own ``export WA_*`` runs in a subprocess and never reaches this
# Python process, so the static table would otherwise leave the client pointing at
# the wrong port. ``url`` templates over the resolved port.
_WA_PORT_PLAN: list[dict] = [
    {
        "port_var": "SHOPPING_PORT", "default": _SHOPPING_PORT_DEFAULT,
        "container_base": "shopping",
        "url_vars": {
            "WA_SHOPPING": "http://localhost:{port}/",
            "VWA_SHOPPING": "http://localhost:{port}/",
        },
    },
    {
        "port_var": "SHOPPING_ADMIN_PORT", "default": _SHOPPING_ADMIN_PORT_DEFAULT,
        "container_base": "shopping_admin",
        "url_vars": {"WA_SHOPPING_ADMIN": "http://localhost:{port}/admin"},
    },
    {
        "port_var": "REDDIT_PORT", "default": _REDDIT_PORT_DEFAULT,
        "container_base": "forum",
        "url_vars": {
            "WA_REDDIT": "http://localhost:{port}",
            "VWA_REDDIT": "http://localhost:{port}",
        },
    },
    {
        "port_var": "GITLAB_PORT", "default": _GITLAB_PORT_DEFAULT,
        "container_base": "gitlab",
        "url_vars": {"WA_GITLAB": "http://localhost:{port}"},
    },
    {
        "port_var": "WIKIPEDIA_PORT", "default": _WIKIPEDIA_PORT_DEFAULT,
        "container_base": "wikipedia",
        "url_vars": {
            "WA_WIKIPEDIA": "http://localhost:{port}" + _WIKIPEDIA_PATH,
            "VWA_WIKIPEDIA": "http://localhost:{port}" + _WIKIPEDIA_PATH,
        },
    },
    {
        "port_var": "CLASSIFIEDS_PORT", "default": _CLASSIFIEDS_PORT_DEFAULT,
        "container_base": "classifieds",
        "url_vars": {"VWA_CLASSIFIEDS": "http://localhost:{port}"},
    },
    {
        "port_var": "HOMEPAGE_PORT", "default": _HOMEPAGE_PORT_DEFAULT,
        "probe_adopt": True,   # host Flask, no container name — adopt iff it answers
        "url_vars": {
            "WA_HOMEPAGE": "http://localhost:{port}",
            "VWA_HOMEPAGE": "http://localhost:{port}",
        },
    },
    {
        "port_var": "MAP_PORT", "default": _MAP_PORT_DEFAULT,
        "url_vars": {"WA_MAP": "http://localhost:{port}"},
        # Host-wide SHARED singleton (one OSM stack per host; fixed compose project
        # ``openstreetmap-website``, reused via start.sh ``_map_alive``). Unlike the
        # per-env-server WA app-services, its port must NOT be reallocated: a busy
        # default means the shared map is already up → reuse it, not spin up a
        # duplicate ~200 GB stack on a different port (which would also recreate the
        # shared container off the port other env-servers point at). Override via MAP_PORT.
        "singleton": True,
    },
]


def _own_stack_published_port(row: dict) -> int | None:
    """Host port THIS scope's own ``<container_base>-<scope>`` container
    currently publishes, or None (not running / no container row / probe
    error → fail closed to the normal free/busy logic).

    This is the rediscovery primitive for the adopt-don't-reallocate family:
    a previous direct-mode run may have booted the stack on AUTO-PICKED
    ports (the defaults were busy then). Checking only the default port
    (the original 89af1f42 fix) misses that stack entirely — the re-run
    would pick a THIRD port while start.sh's ``container_running`` guard
    skips creation, and the readiness probe never converges (the sibling
    of the adopted-default warming deadlock). ``docker port`` on the
    scope-owned NAME finds the stack wherever it landed."""
    base = row.get("container_base")
    if base is None:
        return None
    name = f"{base}-{_resolve_env_server_port()}"
    try:
        r = subprocess.run(
            ["docker", "port", name],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return None  # not running (or docker hiccup) — fail closed
        for line in r.stdout.splitlines():
            if "->" in line and ":" in line:
                return int(line.rsplit(":", 1)[1])
    except Exception:
        return None
    return None


def _default_held_by_own_stack(row: dict, port: int) -> bool:
    """Homepage-row adoption gate (host Flask — no container name, so no
    ownership evidence; container rows use :func:`_own_stack_published_port`
    instead). Fails closed (False → auto-pick) on any probe error so foreign
    services are never adopted by accident."""
    if row.get("probe_adopt"):
        # The homepage is a host Flask — no container name, so there is NO
        # ownership evidence. Adopt only in DIRECT mode (scope "default":
        # one stack per host by convention) and only when the body carries
        # start.sh's ``_HOMEPAGE_PROBE_NEEDLE`` ("WebArena") — a bare 200
        # could be any foreign service, and under an env-server scope a
        # sibling server's homepage is indistinguishable from ours (adopting
        # it would let OUR shutdown pkill THEIR Flask). Fail closed → the
        # caller auto-picks a fresh port.
        if _resolve_env_server_port() != "default":
            return False
        import urllib.request
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/", timeout=3) as resp:
                return resp.status == 200 and b"WebArena" in resp.read(65536)
        except Exception:
            return False
    return False


def _auto_pick_webarena_ports() -> None:
    """Pick free host ports for the WebArena/VisualWebArena stack so two env
    servers' WA stacks coexist on one host.

    For each WA host port (see :data:`_WA_PORT_PLAN`): honour an operator-set
    ``*_PORT`` env var verbatim; else keep the published-image default when it is
    free; else ask the shared ``port`` allocator for a free port in
    :data:`_WA_FALLBACK_START`–:data:`_WA_FALLBACK_END` (flock-protected, so a
    co-resident env-server's auto-pick can't race onto the same port).

    The chosen port is exported BOTH as the ``*_PORT`` env var start.sh reads AND
    (crucially — easy to miss) as the WA_*/VWA_* **base_url** env vars the Python
    client reads. start.sh's own ``export WA_*`` runs in a subprocess and never
    reaches this process, so without setting the URL vars here the client would
    fall back to ``_DEFAULT_ENV_VARS``'s :data:`_DEFAULT_ENV_VARS` (the default
    port) via ``_check_env_vars``'s ``setdefault`` and point at the wrong stack.
    We set ``os.environ[WA_*]`` directly so ``setdefault`` no-ops onto the real
    value (the static table stays fallback-only). Never overrides an operator-set
    URL var.
    """
    import os as _os

    from lite.gym.errors import CapacityExhausted
    from lite.gym.utils.backend.ports import _is_port_free, allocate_ports

    def _set_urls(row: dict, port: int) -> None:
        for url_var, tmpl in row["url_vars"].items():
            if url_var in _os.environ:
                continue  # operator override — never clobber
            _os.environ[url_var] = tmpl.format(port=port)

    for row in _WA_PORT_PLAN:
        port_var = row["port_var"]
        if port_var in _os.environ:
            # Operator pinned the host port; mirror it into the URL vars so the
            # client matches the subprocess.
            _set_urls(row, int(_os.environ[port_var]))
            continue
        default = row["default"]
        if row.get("singleton"):
            # Host-wide shared singleton (e.g. map/OSM): pin the fixed default +
            # url and NEVER reallocate. A busy default is the shared instance
            # itself (which start.sh reuses); auto-picking a new port would spin
            # up / recreate a duplicate stack off the port other env-servers use.
            _os.environ[port_var] = str(default)
            _set_urls(row, default)
            continue
        own_port = _own_stack_published_port(row)
        if own_port is not None:
            # THIS scope's own container is already running — wherever its
            # port landed (default OR a previous run's auto-pick). Adopt it;
            # start.sh's container_running guard will reuse the container.
            _os.environ[port_var] = str(own_port)
            _set_urls(row, own_port)
            logger.info(
                "webarena: %s — own %s-%s already publishes port %d; adopting",
                port_var, row["container_base"], _resolve_env_server_port(),
                own_port,
            )
            continue
        if _is_port_free(default):
            # Default free → keep it. CRITICAL: pin it into os.environ[port_var]
            # so this function is IDEMPOTENT across calls. ensure_services may
            # run multiple times for one stack boot (the 503 cold-boot retry
            # loop re-enters ensure on every client retry). Without pinning, a
            # default that is free NOW but busy on the NEXT call — e.g. our own
            # just-started container now holds it — would be re-auto-picked to a
            # DIFFERENT port, drifting away from the already-running container →
            # probe/container port mismatch → ensure never converges (infinite
            # 503 until the client deadline). Pinning makes the port stable for
            # the process lifetime, matching the auto-picked branch below.
            _os.environ[port_var] = str(default)
            _set_urls(row, default)
            continue
        if _default_held_by_own_stack(row, default):
            # Homepage row only (container rows were handled by the
            # rediscovery above): the busy default answers with the
            # WebArena needle in direct mode → adopt it, exactly what
            # start.sh's _homepage_alive does. A foreign 200 (no needle)
            # or an env-server scope falls through to auto-pick.
            _os.environ[port_var] = str(default)
            _set_urls(row, default)
            logger.info(
                "webarena: %s default %d already served by this scope's own "
                "stack — adopting", port_var, default,
            )
            continue
        try:
            [p] = allocate_ports(
                n=1, range_start=_WA_FALLBACK_START, range_end=_WA_FALLBACK_END + 1,
            )
        except CapacityExhausted as e:
            logger.warning(
                "webarena: auto-pick for %s failed (%s); letting start.sh fail",
                port_var, e,
            )
            continue
        _os.environ[port_var] = str(p)
        _set_urls(row, p)
        logger.info(
            "webarena: %s default %d busy; auto-picked %d", port_var, default, p,
        )


def _resolve_env_server_port() -> str:
    """Per-env-server scope id for container naming. The host env-server sets
    ``CUA_LITE_ENV_SERVER_PORT`` (see ``lite.gym.remote.server``); in direct mode
    (no env-server) it's unset, so fall back to a stable ``"default"`` token —
    matching start.sh / cleanup.sh's ``${CUA_LITE_ENV_SERVER_PORT:-default}``."""
    from lite.gym.utils.backend.ports import env_server_scope
    return env_server_scope(default="default")


# WA service base names start.sh suffixes with the env-server scope. Used by
# BrowserGymServices.shutdown/reap to docker stop/rm THIS env-server's WA set.
_WA_CONTAINER_BASES: tuple[str, ...] = (
    "shopping", "shopping_admin", "forum", "gitlab", "wikipedia",
)


def _ensure_services(env_id: str) -> None:
    """Registry hook: bring up this benchmark's auxiliary service.

    Idempotent. Probe → start.sh → re-probe → raise EnvDepsMissingError
    if still down (operator hasn't run install.sh, port conflict, etc.).

    Caller (``lite.gym.registry.ensure_services``) holds a process-wide
    lock so multiple gym.make races collapse to one start attempt.
    """
    benchmark = _benchmark_of(env_id)
    if benchmark is None:
        return  # umbrella import, nothing to do
    if benchmark in _services_started:
        return
    if benchmark == "miniwob":
        # miniwob is a host-wide shared singleton with its own flock-serialised
        # reuse-or-start path (registry + conflict-avoiding port pick). Handle it
        # entirely here and short-circuit the generic start.sh flow below.
        _ensure_miniwob_singleton()
        _services_started.add(benchmark)
        return
    if benchmark in ("webarena", "visualwebarena"):
        # Auto-pick free host ports for the WA/VWA stack + set the in-process
        # WA_*/VWA_* base_urls so this env-server's Python client and its start.sh
        # subprocess agree, and a co-resident env-server's stack stays disjoint.
        _auto_pick_webarena_ports()
    # ``_service_up`` gates on the shared WA sites (shopping/admin/forum/gitlab/wiki).
    # For visualwebarena that is NOT sufficient to short-circuit: Classifieds is a
    # separate compose stack that start.sh's ``start_classifieds`` brings up, and
    # if the WA stack is already warm (shared-site probe passes) the early-return
    # below would skip start.sh entirely → classifieds is never created → every
    # classifieds reset gets ERR_CONNECTION_REFUSED forever. So for VWA, also
    # require classifieds to be answering HTTP before we declare the benchmark
    # "up". start.sh is idempotent (its WA per-service + ``classifieds_all_up``
    # guards leave already-running containers alone), so re-entering it just to
    # finish bringing up classifieds is safe.
    if _service_up(benchmark) and (
        benchmark != "visualwebarena" or _classifieds_up()
    ) and (
        benchmark not in ("webarena", "visualwebarena") or _homepage_up()
    ):
        _services_started.add(benchmark)
        return

    # Service down → try ``scripts/start.sh <benchmark>``. Sourcing into
    # a child shell is fine — we only need the side effect (containers
    # / HTTP server started); env-var exports back into the Python
    # process are handled by _check_env_vars's setdefault from the
    # static ``_DEFAULT_ENV_VARS`` table.
    logger.info("ensure_services(%s): probing failed, running start.sh", env_id)
    start_sh = _BG_SCRIPTS_DIR / "start.sh"
    try:
        subprocess.run(
            ["bash", str(start_sh), benchmark],
            cwd=str(_BG_REPO_ROOT),
            capture_output=True, text=True, timeout=180, check=True,
        )
    except subprocess.TimeoutExpired:
        # start.sh was STILL bringing the stack up when our 180s budget killed
        # it — the containers it launched (``docker run -d``) keep booting. That
        # is TRANSIENT, not missing deps: a retriable 503 lets the client retry
        # while the (heavy, cold) Magento/gitlab finish. The per-service
        # idempotent start.sh won't tear them down on the retry. (Genuine
        # deps-missing surfaces as CalledProcessError below, → 501.)
        from lite.gym.errors import CapacityExhausted
        down = _wa_down_sites(benchmark)
        raise CapacityExhausted(
            what=(
                f"browsergym.{benchmark} warming: {', '.join(down) or 'backend'} still "
                f"booting (start.sh exceeded its 180s budget; gitlab ~5-15 min). Retry."
            ),
            retry_after_s=30.0,
            layer="env_internal",
        )
    except subprocess.CalledProcessError as e:
        from lite.gym.errors import EnvDepsMissingError
        stderr = getattr(e, "stderr", "") or ""
        raise EnvDepsMissingError(
            what=(
                f"browsergym.{benchmark} auxiliary service failed to "
                f"start: {stderr.strip()[:200]}"
            ),
            install=(
                "uv run --no-sync bash "
                f"lite/gym/envs/browsergym/scripts/install.sh {benchmark}"
            ),
            see="lite/gym/envs/browsergym/README.md",
        )

    # Re-probe. start.sh launches the containers with `docker run -d` and
    # returns fast, but the Magento apps (shopping / shopping_admin) need
    # ~1-3 min more to become HTTP-ready. That is a TRANSIENT "warming up"
    # state, NOT missing deps — so surface it as a retriable 503
    # (CapacityExhausted) instead of a terminal 501 (EnvDepsMissingError).
    # The client's eager-create loop honours Retry-After and waits the stack
    # out (up to _EAGER_CREATE_DEADLINE_S=600s), so cold boot is foolproof:
    # the user just runs the rollout, no manual start.sh pre-warm needed.
    # Genuine deps-missing still raises 501 via the start.sh-failed branch above.
    if not _service_up(benchmark) or (
        benchmark == "visualwebarena" and not _classifieds_up()
    ) or (
        benchmark in ("webarena", "visualwebarena") and not _homepage_up()
    ):
        # Same composite gate as the early-return above — without the
        # classifieds half, VWA could be marked started while the separate
        # classifieds compose stack is still down (or lost its DB-init race).
        from lite.gym.errors import CapacityExhausted
        down = _wa_down_sites(benchmark)
        if benchmark == "visualwebarena" and not _classifieds_up():
            down = [*down, "classifieds"]
        if benchmark in ("webarena", "visualwebarena") and not _homepage_up():
            down = [*down, "homepage"]
        raise CapacityExhausted(
            what=(
                f"browsergym.{benchmark} warming: {', '.join(down) or 'backend'} not "
                f"serving yet (gitlab is the slow long pole, ~5-15 min on a busy host). "
                f"This is normal on a fresh server — retry; the site converges."
            ),
            retry_after_s=30.0,
            layer="env_internal",
        )
    _services_started.add(benchmark)
    logger.info("ensure_services(%s): up", env_id)


# Connection-level playwright errors that mean "a backend service is still
# cold-booting" (gitlab :8023 / reddit / classifieds / shopping), as opposed to
# a genuine task/logic failure. ensure_services gates on all local sites, but a
# backend can still flap/restart after the gate (e.g. gitlab), so a task's reset
# goto can still hit one of these. They are TRANSIENT — converted to a retriable
# 503 so the client retries until the service is HTTP-ready (symmetric with the
# create-side warming 503).
_BACKEND_WARMING_NET_ERRORS: tuple[str, ...] = (
    "ERR_CONNECTION_REFUSED", "ERR_CONNECTION_RESET", "ERR_CONNECTION_TIMED_OUT",
    "ERR_CONNECTION_CLOSED", "ERR_EMPTY_RESPONSE", "ERR_SOCKET_NOT_CONNECTED",
    "ERR_ADDRESS_UNREACHABLE", "ERR_NAME_NOT_RESOLVED",
)


def _is_backend_warming_error(exc: BaseException) -> bool:
    s = str(exc)
    return any(m in s for m in _BACKEND_WARMING_NET_ERRORS)


def _encode_screenshot(obs: Any) -> bytes:
    """Convert a BrowserGym screenshot (numpy array H×W×3) to PNG bytes.

    Raises on failure so the caller sees the real error.
    """
    import numpy as np
    from PIL import Image

    if obs is None:
        raise RuntimeError("BrowserGym returned None screenshot")
    if isinstance(obs, np.ndarray):
        return encode_png(Image.fromarray(obs))
    if isinstance(obs, (bytes, bytearray)):
        return encode_png(bytes(obs))
    raise RuntimeError(f"Unexpected screenshot type: {type(obs)}")


def _encode_screenshot_maybe_som(
    screenshot: Any, extra_properties: dict | None, use_som: bool
) -> bytes:
    """PNG-encode a BrowserGym screenshot, optionally overlaying Set-of-Marks.

    When ``use_som`` and ``extra_properties`` are present, draws numbered bid
    boxes onto the screenshot via BrowserGym's ``overlay_som`` (dashed bbox + the
    ``bid`` as a label per element flagged ``set_of_marks``) so a vision model can
    read the bid off the image and emit ``click('<bid>')``. Falls back to the raw
    screenshot when ``use_som`` is off or no extra-properties are available (e.g.
    a dead-backend step, or ``skip_dom_extraction`` left them empty — which would
    be a misconfiguration for SoM).
    """
    if not use_som or not extra_properties:
        return _encode_screenshot(screenshot)
    import numpy as np
    from browsergym.utils.obs import overlay_som
    from PIL import Image

    # overlay_som returns a numpy RGBA array (it converts its PIL canvas back via
    # ``np.array(img)``); fold the alpha onto white → RGB PNG for the model.
    arr = overlay_som(np.asarray(screenshot), extra_properties)
    img = Image.fromarray(arr).convert("RGB")
    return encode_png(img)


def _extract_instruction(obs: dict[str, Any]) -> str:
    """Extract the task instruction from a BrowserGym observation."""
    goal_object = obs.get("goal_object") or []
    has_goal_image = any(
        isinstance(msg, dict) and msg.get("type") == "image_url"
        for msg in goal_object
    )
    if has_goal_image:
        parts = []
        for msg in goal_object:
            if not isinstance(msg, dict) or msg.get("type") != "text":
                continue
            text = str(msg.get("text") or "")
            stripped = text.strip()
            if (
                stripped.startswith("Input image ")
                and " below" in stripped
                and ("local path:" in stripped or "url:" in stripped or stripped.endswith(" below"))
            ):
                continue
            if stripped.startswith("WARNING: This goal cannot be converted to a text-only goal format."):
                continue
            parts.append(text)
        return "\n".join(parts)

    goal = obs.get("goal", "")
    if goal:
        return goal

    if goal_object:
        parts = []
        for msg in goal_object:
            if isinstance(msg, dict) and msg.get("type") == "text":
                parts.append(msg["text"])
        if parts:
            return "\n".join(parts)

    return ""


def _extract_goal_images_b64(obs: dict[str, Any]) -> list[str]:
    """Extract ALL goal images from a BrowserGym ``goal_object`` as base64 PNGs.

    BrowserGym builds VWA's ``goal_object`` as a list of multimodal parts:
    ``[{"type":"text",...}, {"type":"text","text":"Input image 1/N..."}, {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}, ...]``.
    A task may carry 0, 1, or several ``image_url`` parts (VWA ``config["image"]``
    is null / a single URL / a list of URLs). The url is always a data URI per
    ``browsergym/visualwebarena/task.py:_build_goal``'s ``pil_image_to_data_uri``.

    Returns the base64 PNG bodies (no ``data:image/...;base64,`` prefix), in
    ``goal_object`` order. Empty list if the goal has no image. ``reset()``
    surfaces these under ``metadata["goal_images_b64"]``; the
    ``visualwebarena.goal_image`` agent decodes them and shows them every turn.
    See ``goal_image.py``.
    """
    images: list[str] = []
    for msg in obs.get("goal_object") or []:
        if not isinstance(msg, dict):
            continue
        if msg.get("type") != "image_url":
            continue
        url = (msg.get("image_url") or {}).get("url") or ""
        if not url.startswith("data:"):
            continue
        # data:image/png;base64,XXXX → keep XXXX (decoded later via decode_image).
        comma = url.find(",")
        if comma == -1:
            continue
        images.append(url[comma + 1:])
    return images

def _escape(text: str) -> str:
    """Escape string for embedding in Python code (between double quotes)."""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")


# ---------------------------------------------------------------------------
# Task registration — enumerate BrowserGym benchmarks and register tasks
# ---------------------------------------------------------------------------

def _wa_vwa_task_facts(benchmark: str, bgym_task_id: str) -> tuple[list[str], bool, bool]:
    """Return STATIC, hardware-independent facts about a WA/VWA task, surfaced
    into ``metadata.others`` so callers compose their OWN runtime filters:

      * ``sites`` (``list[str]``) — the WebArena sites the task touches
        (``"shopping"``, ``"map"``, ``"gitlab"``, ``"reddit"``, ...). Map runs by
        default (``install.sh`` installs the ~200GB OSM assets, ``start.sh`` serves
        them); only if you built with ``WEBARENA_INSTALL_MAP=0`` skip map tasks via
        ``--filter "lambda m: 'map' not in m.others.get('sites', [])"``.
      * ``llm_as_a_judge`` (``bool``) — True iff the evaluator uses
        ``fuzzy_match`` (string_match with an LLM scoring the agent's NL answer;
        reference WA hardcodes ``model="gpt-4-1106-preview"``, so without that
        endpoint the eval crashes). Filter
        ``--filter "lambda m: not m.others.get('llm_as_a_judge')"``.
      * ``mutating`` (``bool``) — True iff the task WRITES persisted shared-backend
        state (``isolation.is_mutating``: a ``program_html`` / ``page_image_query``
        eval that re-navigates to a SPECIFIC url to verify persistence, a
        ``require_reset`` flag, or a write-verb intent). NON-mutating tasks are
        read-only Q&A — residue-immune and safe to run at full concurrency; the
        mutating ones dirty the shared stack, so an operator runs them in a
        separate write pass — sequential, in ``depends_on`` order, after one
        baseline reset (the order-based strict flow in docs/eval.md). Split the
        eval: reads ``--filter "lambda m: not m.others.get('mutating')"``;
        writes ``--filter "lambda m: m.others.get('mutating')"``.
        Exposed UNCONDITIONALLY (a static task fact) — distinct from the
        conflict-gate machinery (``conflict_keys``), which is only populated
        when the env-server runs the strict-isolation config.

    These are properties of the TASK (from its raw json), not of the running
    deployment. We deliberately do NOT probe service reachability here (no
    ``map_unreachable`` / ``classifieds_unreachable`` baked ``exclude_reason``):
    reachability is a MUTABLE, deploy/hardware-dependent state — baking it at
    import (and persisting it into exported parquet) would wrongly freeze a
    transient value. The env stays generic; reachability is the caller's runtime
    concern (and a cold-booting service is now retried via 503, not pre-excluded).

    Tasks not found in the raw json (e.g. train split, ID drift) → ([], False, False).
    """
    if benchmark not in ("webarena", "visualwebarena"):
        return [], False, False
    t = isolation.task_raw_config(benchmark, bgym_task_id)
    if t is None:
        return [], False, False
    sites = sorted(t.get("sites") or [])
    e = t.get("eval", {}) or {}
    llm_judge = "fuzzy_match" in (e.get("reference_answers") or {})
    mutating = isolation.is_mutating(t)
    return sites, llm_judge, mutating


def _register_benchmark(
    benchmark: str,
    bgym_task_ids: list[str],
    split: str = "eval",
    viewport: tuple[int, int] | None = None,
    # Multi-benchmark env → the concrete budget is HARDCODED at each
    # call site below (miniwob 10; webarena/visualwebarena 30). The signature
    # default is the yaml constant (null) so an un-passed call inherits "derive",
    # never an arbitrary literal.
    max_steps: int | None = _MAX_STEPS,
    task_kwargs: dict[str, Any] | None = None,
    action_subsets: tuple[str, ...] = ("coord", "chat", "infeas", "nav", "tab"),
    seed: int | None = _DEFAULT_TASK_SEED,
) -> None:
    """Register all tasks from a BrowserGym benchmark.

    `action_subsets` drives the BrowserGym action set + the tool schemas
    surfaced to the agent (via `_tools_for_subsets` in `metadata`). yaml
    overrides of `action_subsets` cleanly re-derive the tool list at
    metadata-access time.
    """
    vw, vh = viewport or _BENCHMARK_VIEWPORTS.get(benchmark, (1280, 720))

    for bgym_tid in bgym_task_ids:
        config = BrowserGymConfig(
            bgym_task_id=bgym_tid,
            benchmark=benchmark,
            viewport_width=vw,
            viewport_height=vh,
            task_kwargs=task_kwargs or {},
            action_subsets=action_subsets,
        )
        # CUA-Lite task ID: browsergym.<benchmark>@<task_name>
        task_name = bgym_tid.split(".", 1)[-1] if "." in bgym_tid else bgym_tid
        cua_id = f"browsergym.{benchmark}@{task_name}"
        # Env-wide make() defaults from default.yaml make_kwargs. Keyed
        # per-benchmark env_id; idempotent (last-wins).
        registry.set_env_make_kwargs(cua_id.split("@", 1)[0], CFG.make_kwargs)
        # Fixed task seed forwarded as a registered kwarg → ``BrowserGymEnv(seed=…)``
        # → ``env.reset(seed=…)``. Mirrors the cua-lite eval convention
        # (androidworld/mobilegym register a fixed ``seed``); only attach when set
        # so ``seed=None`` registrations stay unseeded. yaml may override via
        # ``env_kwargs.seed``.
        reg_kwargs: dict[str, Any] = {}
        if seed is not None:
            reg_kwargs["seed"] = seed
        register(
            key=cua_id,
            entry_point=lambda *, cfg=config, ms=max_steps, **kw: BrowserGymEnv(config=cfg, max_steps=kw.pop("max_steps", ms), **kw),
            split=split,
            # Same-source contract: registered copy == the env's builder
            # output — the registered subsets' tool schemas + the STATIC task
            # facts (sites / llm_as_a_judge / mutating / depends_on /
            # conflict_keys — semantics documented on ``_task_metadata``). At
            # runtime the builder recomputes from the (possibly yaml-overridden)
            # config; the env-server's conflict gate reads the REGISTERED copy.
            metadata=BrowserGymEnv._task_metadata(
                benchmark,
                bgym_tid,
                action_subsets=action_subsets,
                viewport=(vw, vh),
            ),
            **reg_kwargs,
        )

#: Set once :func:`_load_tasks` has registered the benchmark catalog, so the lazy
#: hook (``BrowserGymServices.register_tasks``) is idempotent across leaf env_ids
#: and the catalog-probe / make() paths.
_tasks_registered = False


def _load_tasks() -> None:
    """Discover and register BrowserGym benchmark tasks.

    Lazy: fired by ``BrowserGymServices.register_tasks`` on the first catalog
    probe (``task_ids``) or ``make()`` for ANY browsergym leaf — NOT at module
    import. It registers ALL benchmarks (~21s: miniwob + WA + VWA discovery), so
    paying it at import would tax every startup (rollout / eval / env-server) AND
    every pytest collection. Idempotent: the first call registers every leaf;
    later leaf probes find them already in ``_splits``."""
    global _tasks_registered
    if _tasks_registered:
        return

    # MiniWoB is single-page → only nav tools (back/goto), no tab. Other three
    # are multi-tab benchmarks → include "tab" in the action subset so
    # canonical new_tab/switch_tab/close_tab are exposed.
    _SUBSETS_NO_TAB = ("coord", "chat", "infeas", "nav")
    _SUBSETS_WITH_TAB = ("coord", "chat", "infeas", "nav", "tab")

    # --- MiniWoB ---
    try:
        import browsergym.miniwob
        task_ids = [
            task_cls.get_task_id()
            for task_cls in browsergym.miniwob.ALL_MINIWOB_TASKS
        ]
        _register_benchmark(
            "miniwob",
            task_ids,
            split="eval",
            max_steps=10,
            action_subsets=_SUBSETS_NO_TAB,
        )
        logger.debug("Registered %d MiniWoB tasks", len(task_ids))
    except ImportError:
        logger.debug("Skipping MiniWoB (browsergym.miniwob not installed)")
    except Exception as e:
        logger.warning("Failed to register MiniWoB tasks: %s", e)

    # --- WebArena ---
    try:
        import browsergym.webarena
        task_ids = browsergym.webarena.ALL_WEBARENA_TASK_IDS
        _register_benchmark(
            "webarena",
            task_ids,
            split="eval",
            max_steps=30,
            action_subsets=_SUBSETS_WITH_TAB,
            # Closed-world goal hints (with_homepage_hint / with_na_hint, both
            # default OFF — browsergym-faithful) are flat ``BrowserGymConfig``
            # fields — see their definition for
            # the full rationale — forwarded into the task per-benchmark in the
            # ``gym.make`` path. A config can flip them via env_kwargs; no longer
            # baked here.
        )
        logger.debug("Registered %d WebArena tasks", len(task_ids))
    except ImportError:
        logger.debug("Skipping WebArena (browsergym.webarena not installed)")
    except Exception as e:
        logger.warning("Failed to register WebArena tasks: %s", e)

    # --- VisualWebArena ---
    # IMPORTANT: do NOT ``import browsergym.visualwebarena`` here. Its package
    # ``__init__`` imports the ``visualwebarena`` submodule whose body pulls in
    # ``torch`` (transitively), which would make merely importing this module
    # drag a multi-hundred-MB ML stack into every process — breaking the gym
    # self-containment contract (tests/gym/test_self_containment.py). The task
    # ids are a pure function of ``visualwebarena.config.TASK_IDS`` (range), so
    # we read them WITHOUT triggering the heavy package init, and defer the
    # actual ``import browsergym.visualwebarena`` (which populates BrowserGym's
    # own gymnasium registry) to the first VWA reset via
    # :func:`_ensure_visualwebarena_imported`.
    try:
        task_ids = _vwa_config_task_ids()
        _register_benchmark(
            "visualwebarena",
            task_ids,
            split="eval",
            max_steps=30,
            action_subsets=_SUBSETS_WITH_TAB,
            # Goal hints are flat ``BrowserGymConfig`` fields now (see their
            # definition). VWA's task.__init__ accepts ONLY ``with_na_hint`` (no
            # ``with_homepage_hint`` param — visualwebarena/task.py:_build_goal
            # has no homepage note), so the gym.make path forwards only
            # ``with_na_hint`` for this benchmark; the homepage portal is still
            # served + reachable, only the goal-string hint is omitted.
        )
        logger.debug("Registered %d VisualWebArena tasks", len(task_ids))
    except FileNotFoundError:
        logger.debug("Skipping VisualWebArena (browsergym.visualwebarena not installed)")
    except Exception as e:
        logger.warning("Failed to register VisualWebArena tasks: %s", e)

    _tasks_registered = True


def _vwa_config_task_ids() -> list[str]:
    """VWA task ids (``visualwebarena.<i>``) without importing the heavy package.

    Loads only ``browsergym/visualwebarena/config.py`` in isolation — a pure
    ``TASK_IDS = range(...)`` constant table — so we never run the package
    ``__init__`` (which pulls ``torch`` via the ``task`` submodule). The ids are
    identical to ``browsergym.visualwebarena.ALL_VISUALWEBARENA_TASK_IDS``,
    which is built as ``f"visualwebarena.{i}" for i in config.TASK_IDS``.
    Raises ``FileNotFoundError`` when browsergym is not installed (mapped to the
    "skip benchmark" path by the caller).
    """
    import importlib.util

    # Locate the package DIR via the torch-free ``browsergym`` namespace package
    # only — ``find_spec("browsergym.visualwebarena.config")`` would import the
    # ``browsergym.visualwebarena`` PARENT to resolve the submodule, which pulls
    # torch (the very thing we defer). Then exec ``config.py`` in isolation.
    bgym_spec = importlib.util.find_spec("browsergym")
    if bgym_spec is None or not bgym_spec.submodule_search_locations:
        raise FileNotFoundError("browsergym not installed")
    config_path = (
        Path(next(iter(bgym_spec.submodule_search_locations)))
        / "visualwebarena"
        / "config.py"
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"browsergym.visualwebarena.config not found: {config_path}")
    spec = importlib.util.spec_from_file_location("_vwa_config", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [f"visualwebarena.{i}" for i in mod.TASK_IDS]


_VWA_IMPORTED = False
_VWA_IMPORT_LOCK = threading.Lock()
_JUDGE_OVERRIDE_LOCK = threading.Lock()
# One shared BLIP-2 caption model per process (see _patch_vwa_captioning): keyed
# by (model_name, device, dtype), guarded by the lock so the (thread-unsafe) load
# happens exactly once regardless of reset concurrency.
_CAPTIONING_LOCK = threading.Lock()
_CAPTIONING_FN_CACHE: dict[tuple[str, str, str], Any] = {}


def _ensure_visualwebarena_imported() -> None:
    """Eagerly import ``browsergym.visualwebarena`` on the first VWA reset.

    Deferred out of :func:`_load_tasks` so merely importing this module stays
    torch-free (the gym self-containment contract). Idempotent. Two effects the
    actual VWA ``gym.make`` needs:

    1. The package ``__init__`` registers every ``visualwebarena.<i>`` task into
       BrowserGym's OWN gymnasium registry (``register_task``), which
       ``gym.make("browsergym/visualwebarena.<i>")`` resolves against.
    2. The judge-model override: replace the model the upstream WA *and* VWA
       ``llm_fuzzy_match`` evaluators HARD-CODE ("gpt-4-1106-preview") with the
       configured ``_LLM_JUDGE_MODEL``, so the judge model is one config knob
       while ROUTING stays universal (the vendored plain ``OpenAI`` client reads
       the standard OPENAI_BASE_URL / OPENAI_API_KEY). We wrap the symbol in
       EACH package's ``openai_utils`` BEFORE the first reset imports
       ``helper_functions``, which binds it by value.
    """
    global _VWA_IMPORTED
    # Fast path: no lock once imported. Otherwise serialize under the module
    # lock with a double check — this runs on each env's worker thread (from
    # reset), so without the lock two threads can both pass the gate and both
    # run the heavy import + monkeypatch concurrently.
    if _VWA_IMPORTED:
        return
    with _VWA_IMPORT_LOCK:
        if _VWA_IMPORTED:
            return
        _import_visualwebarena_locked()


def _import_visualwebarena_locked() -> None:
    """The heavy VWA import + judge-model monkeypatch, run once under
    ``_VWA_IMPORT_LOCK``. See :func:`_ensure_visualwebarena_imported`."""
    global _VWA_IMPORTED

    # VWA's ``setup()`` (run at every env.reset) imports the upstream
    # ``visualwebarena.evaluation_harness.evaluators`` → ``helper_functions`` →
    # ``llms.providers.openai_utils``, whose MODULE BODY does
    # ``OpenAI(api_key=os.environ["OPENAI_API_KEY"])`` (openai_utils.py:15). With
    # no key set this raises ``KeyError: 'OPENAI_API_KEY'`` at import, so EVERY
    # VWA reset 500s — even the ~majority of tasks whose evaluator never calls
    # OpenAI (string / URL / page-state checks). Seed a placeholder so the import
    # succeeds; the operator overrides it with a REAL key (env-var = the right
    # home for a secret, per docs/envs.md §Config) only for the
    # LLM-fuzzy-match / image-caption evaluator subset.
    os.environ.setdefault("OPENAI_API_KEY", "cua-lite-vwa-import-placeholder")
    import browsergym.visualwebarena  # noqa: F401 — populates BrowserGym registry
    _install_judge_model_override()
    _install_vwa_captioning_hook()
    _VWA_IMPORTED = True


def _install_vwa_captioning_hook() -> None:
    """Arrange for the caption-model cache (:func:`_patch_vwa_captioning`) to be
    installed at the first moment it is safe.

    ``image_utils`` is only importable once ``DATASET`` + the site env vars exist,
    and those are set by ``VisualWebArenaInstance.__init__`` (instance.py:28) — which
    browsergym runs during ``env.reset`` (``self.task = task_entrypoint(...)``, whose
    VWA ``__init__`` builds the instance), NOT at ``gym.make`` and NOT at package
    import. So wrap that ``__init__`` to install the caption cache right AFTER it has
    set the env vars — still before the same reset's ``setup()`` calls
    ``get_captioning_fn``. Installed once under ``_VWA_IMPORT_LOCK`` (the class patch
    is single-threaded); the per-instance ``_patch_vwa_captioning`` it triggers is
    itself idempotent + race-safe. Idempotent via the ``_cua_lite_caption_hook``
    marker on the wrapped ``__init__``."""
    import functools as _ft

    from browsergym.visualwebarena.instance import VisualWebArenaInstance

    _orig_init = VisualWebArenaInstance.__init__
    if getattr(_orig_init, "_cua_lite_caption_hook", False):
        return

    @_ft.wraps(_orig_init)
    def _init(self, *a, _orig_init=_orig_init, **kw):
        _orig_init(self, *a, **kw)  # sets DATASET + SHOPPING/REDDIT/... env vars
        _patch_vwa_captioning()

    _init._cua_lite_caption_hook = True
    VisualWebArenaInstance.__init__ = _init


def _patch_vwa_captioning() -> None:
    """Make VWA's per-reset BLIP-2 caption-model load process-global + one-time.

    Upstream ``get_captioning_fn`` (``visualwebarena.evaluation_harness.image_utils``)
    calls ``Blip2ForConditionalGeneration.from_pretrained(...)`` then ``.to(device)``
    on EVERY ``task.setup()`` — i.e. every env.reset. Two problems at concurrency>1:

    1. ``from_pretrained`` is NOT thread-safe. For a large model it builds the
       module on the ``meta`` device and materializes weights in place; two threads
       interleaving in that critical section leave one model's params on ``meta``,
       so the following ``captioning_model.to(device)`` raises
       "Cannot copy out of meta tensor; ... use torch.nn.Module.to_empty()".
    2. It reloads a ~4 GB model from scratch on every reset — pure waste even once
       the crash is gone.

    Fix: cache the built ``caption_images`` closure per (model_name, device, dtype)
    and serialize the (racy) load behind ``_CAPTIONING_LOCK``, so the model is
    loaded exactly once per process regardless of reset concurrency. Inference on
    the shared model is read-only, so concurrent ``caption_images`` calls are safe.
    Idempotent (the ``_cua_lite_cached`` marker); ``task.setup`` re-imports the
    symbol from ``image_utils`` on every call, so patching the source module
    attribute is enough for later resets to pick it up.

    MUST be called only AFTER a ``VisualWebArenaInstance`` has been constructed:
    importing ``image_utils`` pulls in ``visualwebarena.browser_env.envs``, whose
    module body reads ``os.environ["DATASET"]`` — set by the instance ``__init__``,
    NOT at ``browsergym.visualwebarena`` import time. That is exactly why this runs
    from the instance-``__init__`` hook (:func:`_install_vwa_captioning_hook`), not
    at package-import time. The install is race-safe: the wrapper closes over the
    module-level cache + lock, so concurrent first-reset installers produce
    equivalent wrappers regardless of which assignment wins."""
    import functools as _ft

    import visualwebarena.evaluation_harness.image_utils as _iu

    _orig = _iu.get_captioning_fn
    if getattr(_orig, "_cua_lite_cached", False):
        return

    @_ft.wraps(_orig)
    def _cached_get_captioning_fn(
        device, dtype, model_name: str = "Salesforce/blip2-flan-t5-xl", _orig=_orig
    ):
        key = (model_name, str(device), str(dtype))
        with _CAPTIONING_LOCK:
            fn = _CAPTIONING_FN_CACHE.get(key)
            if fn is None:
                fn = _orig(device, dtype, model_name=model_name)
                _CAPTIONING_FN_CACHE[key] = fn
        return fn

    _cached_get_captioning_fn._cua_lite_cached = True
    _iu.get_captioning_fn = _cached_get_captioning_fn


def _install_judge_model_override() -> None:
    """Force the upstream WA *and* VWA ``llm_fuzzy_match`` judge to use the
    configured ``_LLM_JUDGE_MODEL`` instead of the value they HARD-CODE
    ("gpt-4-1106-preview"), so the judge model is one config knob while ROUTING
    stays universal (the vendored plain ``OpenAI`` client reads the standard
    OPENAI_BASE_URL / OPENAI_API_KEY).

    Wraps EACH package's ``openai_utils.generate_from_openai_chat_completion``
    BEFORE the first reset imports ``helper_functions`` (which binds it by value).
    Called from BOTH reset paths — the WA branch (which has no heavy deferred
    import) and the VWA branch (via :func:`_ensure_visualwebarena_imported`) — so
    a pure-WebArena run gets the override too, not just runs that touch VWA.
    Idempotent + per-package (the ``_cua_lite_model_override`` marker) and
    thread-safe (``_JUDGE_OVERRIDE_LOCK``); a package not importable yet (e.g.
    ``visualwebarena`` on a pure-WA run) is skipped and retried on a later call."""
    import functools as _ft
    import importlib as _il
    with _JUDGE_OVERRIDE_LOCK:
        for _judge_pkg in ("webarena", "visualwebarena"):
            try:
                _ou = _il.import_module(f"{_judge_pkg}.llms.providers.openai_utils")
            except Exception as _e:  # WA needs WA_* env / VWA the placeholder key
                logger.debug("judge-model override: %s skipped (%s)", _judge_pkg, _e)
                continue
            _orig = getattr(_ou, "generate_from_openai_chat_completion", None)
            if _orig is None or getattr(_orig, "_cua_lite_model_override", False):
                continue

            @_ft.wraps(_orig)
            def _judge(*a, _orig=_orig, **kw):
                kw["model"] = _LLM_JUDGE_MODEL
                return _orig(*a, **kw)

            _judge._cua_lite_model_override = True
            _ou.generate_from_openai_chat_completion = _judge


def _wa_scoped_containers(server_port: str) -> list[str]:
    """The WA container names THIS env-server owns (``<svc>-<server_port>``),
    including the classifieds compose stack's two pinned-name containers."""
    names = [f"{base}-{server_port}" for base in _WA_CONTAINER_BASES]
    names += [f"classifieds-{server_port}", f"classifieds_db-{server_port}"]
    return names


def _wa_stop_scoped(server_port: str) -> int:
    """``docker rm -f -v`` every WA container scoped to *server_port*.
    Returns the count actually removed. Never touches a bare/global name, so a
    co-resident env-server's stack is untouched. ``-v`` reaps only the
    image-baked anonymous (Magento/MySQL) volumes; the NAMED ``classifieds_db``
    volume is untouched here and torn down by
    :func:`_classifieds_compose_down_scoped` (``compose down -v``)."""
    from lite.gym.utils.backend.docker import docker_rm_f
    removed = 0
    for name in _wa_scoped_containers(server_port):
        # `docker rm -f` is stop+rm in one; skip the noise of a separate stop.
        removed += docker_rm_f(name, timeout=60.0, label="browsergym.wa")
    return removed


def _classifieds_compose_down_scoped(server_port: str) -> bool:
    """Tear down THIS server's classifieds compose stack INCLUDING its named
    MySQL data volume — the residue ``_wa_stop_scoped``'s ``docker rm -f``
    (container names only) leaves behind.

    Plain container removal keeps the ``*_db_data`` volume, so the next run would
    re-create the web/db containers against a DIRTY database and silently
    mis-score VWA classifieds. Mirrors ``cleanup.sh``'s ``stop_classifieds``:
    ``docker compose -p classifieds-<port> down -v`` from the per-scope
    extraction dir (compose reclaims the volume + network BY PROJECT, so the
    generated volume name need not be known here). Scoped by the
    ``classifieds-<port>`` project so a co-tenant is never touched; a no-op when
    the per-scope dir is absent (classifieds never built on this server). Returns
    True iff it ran the teardown."""
    # Resolve the cache the SAME way start.sh does (its
    # ``${BROWSERGYM_CACHE:-$ENV_DIR/.cache}`` default): the env-server python
    # process does NOT inherit start.sh's subprocess export, so reading
    # os.environ alone would miss the default location and silently skip the
    # volume cleanup whenever the operator didn't export BROWSERGYM_CACHE.
    cache = os.environ.get("BROWSERGYM_CACHE") or os.path.join(ENV_DIR, ".cache")
    scope_dir = os.path.join(cache, f"classifieds_docker_compose-{server_port}")
    if not os.path.isdir(scope_dir):
        return False
    import subprocess as _sp
    _sp.run(
        ["docker", "compose", "-p", f"classifieds-{server_port}",
         "down", "-v", "--remove-orphans"],
        cwd=scope_dir, capture_output=True, text=True,
    )
    return True


def _wa_stop_homepage_scoped(homepage_port: str | None) -> bool:
    """Kill THIS env-server's homepage Flask (``python -m flask run --port
    <homepage_port>``, launched by start.sh). The launcher's argv carries the
    port but NOT ``app.py`` (FLASK_APP is an env var), so match by the SCOPED
    port — a co-resident env-server's homepage (different port) is never killed.
    No-op when the port is unknown (e.g. a miniwob-only server never started one).
    Returns True iff a process was matched + killed."""
    if not homepage_port:
        return False
    import subprocess as _sp
    r = _sp.run(
        ["pkill", "-9", "-f", rf"flask run .*--port {homepage_port}([^0-9]|$)"],
        capture_output=True, text=True,
    )
    return r.returncode == 0


# NOTE: there is intentionally NO ``_miniwob_kill_owned`` here anymore. miniwob
# is a host-level SHARED singleton (see _ensure_miniwob_singleton): no env-server
# owns or kills it, so killing it from shutdown/reap would cross-kill a
# co-resident server's live miniwob. The only place that stops miniwob is the
# manual host teardown ``scripts/cleanup.sh`` (bash), never the env-server.


class BrowserGymServices(EnvServices):
    """Env-server capability for browsergym: bring up each benchmark's auxiliary
    service (miniwob HTTP server; WA/VWA docker stacks) and tear down THIS
    env-server's own resources at shutdown / restart. Registered under the
    ``browsergym`` umbrella; ``ensure`` receives the leaf env_id and dispatches
    per-benchmark.

    LIFECYCLE MODEL (per-env-server isolation): the WA/VWA stack is a FIXED,
    long-lived resource (NOT per-episode), so there is NO periodic orphan
    reaping. Everything is scoped by the env-server port (``scope.server_port``
    or ``"default"`` in direct mode) so one env-server NEVER touches another's
    containers:

      * ``shutdown`` (normal exit, AFTER every per-instance close) → docker
        stop/rm THIS env-server's WA container set. miniwob is NOT touched (it
        is a host-shared singleton — see _ensure_miniwob_singleton).
      * ``reap(boot=True)`` (one-shot restart recovery, before serving) → clean
        THIS env-server's OWN leftover WA containers, in case a prior run died
        abnormally. miniwob is NOT touched. ``reap`` on a STEADY tick is a NO-OP
        (the WA stack is a fixed resource — reaping it would kill the live
        backend).
      * ``live_ids`` → ``None`` (fail-closed): the WA stack is shared/long-lived
        and has no per-instance external_resource_id to reconcile, so the
        framework must NOT per-instance-reap it.

    NOTE: shared-backend isolation (conflict_keys / mutating / restore_backend)
    is intentionally NOT yet on this object: it stays on metadata.others + the
    module ``restore_backend`` until it can be validated against live WA/VWA infra
    (the conflict gate is correctness-critical and not exercisable in unit
    tests)."""

    def register_tasks(self, env_id: str) -> None:
        # Register the benchmark catalog (miniwob + WA + VWA discovery) lazily —
        # cheap relative to ensure() (which boots the miniwob HTTP / WA docker
        # backends), and fired by a bare task_ids() probe WITHOUT booting them.
        _load_tasks()

    def ensure(self, env_id: str) -> None:
        _ensure_services(env_id)

    def health(self, env_id: str) -> None:
        # Cheap per-benchmark readiness probe so ``GET /envs/<leaf>`` tells the truth
        # (without this, the no-op default reported ``available: true`` while the WA/VWA
        # stack was still cold). Probe-only, NEVER boots. Mirrors
        # ``_ensure_services``'s up-gate: all shared local sites (_service_up) + classifieds for VWA.
        benchmark = _benchmark_of(env_id)
        if benchmark is None:
            return  # bare umbrella has no single backend to probe
        up = _service_up(benchmark) and (
            benchmark != "visualwebarena" or _classifieds_up()
        ) and (
            benchmark not in ("webarena", "visualwebarena") or _homepage_up()
        )
        if not up:
            from lite.gym.errors import EnvDepsMissingError
            raise EnvDepsMissingError(
                what=f"browsergym {benchmark} backend not serving (stack cold/unreachable)",
                install=f"bash lite/gym/envs/browsergym/scripts/start.sh {benchmark}",
                see="lite/gym/envs/browsergym/README.md",
            )

    def shutdown(self, env_id: str, scope) -> None:
        # Normal env-server shutdown: bring THIS env-server's WA stack down +
        # kill its owned miniwob HTTP server. Scoped by server_port so a
        # co-resident env-server's stack survives. ``env_id`` is the umbrella
        # ``browsergym`` here; we tear down all of THIS run's browsergym
        # resources regardless of which leaf benchmarks were used.
        server_port = str(scope.server_port) if scope.server_port is not None else "default"
        n = _wa_stop_scoped(server_port)
        # Also reap THIS server's homepage Flask (a host PROCESS, not a container —
        # _wa_stop_scoped only handles containers). It is scoped by its auto-picked
        # HOMEPAGE_PORT (set into os.environ by _auto_pick_webarena_ports at ensure).
        # Without this the homepage leaks on every shutdown (cleanup.sh's reaper
        # alone is the manual path; the env-server must clean its own).
        hp = os.environ.get("HOMEPAGE_PORT")
        killed_hp = _wa_stop_homepage_scoped(hp)
        # NOTE: do NOT kill miniwob here. It is a host-level SHARED singleton
        # (see _ensure_miniwob_singleton) that co-resident env-servers reuse;
        # killing it would tear down another live server's miniwob mid-episode.
        # A true host teardown is scripts/cleanup.sh, not per-server shutdown.
        if n or killed_hp:
            logger.info("browsergym shutdown: removed %d WA containers%s (scope %s)",
                        n, f" + homepage Flask (port {hp})" if killed_hp else "", server_port)

    # live_ids inherited from base (→ None: "no external world to reconcile"). The WA/VWA
    # stack is a shared, long-lived resource with no per-instance external_resource_id, and
    # BackendFamily.SINGLETON is gated out of steady reconcile — so this is
    # never called and the framework never per-instance-reaps the shared stack mid-episode.
    def reap(self, env_id: str, scope, in_use: set[str], *, boot: bool = False) -> int:
        # NO periodic reaping — the WA stack is a FIXED, long-lived resource. On a
        # steady tick this MUST be a no-op (reaping would kill the live backend).
        # Only the one-shot framework-supplied boot recovery cleans THIS env
        # server's OWN leftover containers (a prior run that died abnormally),
        # scoped by server_port so a co-tenant's stack is never touched.
        if not boot:
            return 0
        server_port = str(scope.server_port) if scope.server_port is not None else "default"
        n = _wa_stop_scoped(server_port)
        # Reclaim the classifieds compose DB VOLUME too: _wa_stop_scoped's
        # docker-rm-by-name leaves the *_db_data volume behind, so without this a
        # restart would re-create classifieds against a DIRTY DB (silent VWA
        # mis-score). Boot-only — this is where "restart = clean baseline" is made
        # true for VWA, without slowing the graceful-shutdown path.
        cf_vol = _classifieds_compose_down_scoped(server_port)
        # Also reap a leftover homepage Flask from a prior abnormally-died run
        # (scoped by HOMEPAGE_PORT). See shutdown().
        killed_hp = _wa_stop_homepage_scoped(os.environ.get("HOMEPAGE_PORT"))
        # NOTE: do NOT reap miniwob here either — it is a host-level SHARED
        # singleton (see _ensure_miniwob_singleton + shutdown). Boot-recovery on
        # one server must not kill the miniwob a co-resident server is serving.
        if n or killed_hp or cf_vol:
            logger.info("browsergym boot-reap: cleaned %d leftover WA containers%s%s (scope %s)",
                        n, " + homepage Flask" if killed_hp else "",
                        " + classifieds volume" if cf_vol else "", server_port)
        return n


register_services("browsergym", BrowserGymServices())

register_family("browsergym", BackendFamily.SINGLETON)
