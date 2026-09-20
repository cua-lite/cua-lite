"""lite.osworld — OSWorld-aligned CUA eval environment on Docker (GNOME-Shell desktop).

Host↔container interaction rides the exec-stdio transport
(``lite/gym/sandbox/exec_stdio/``) — one persistent ``docker exec -i`` JSON-lines
session per container, no published agent port. The container runs ONE in-image
server:
  - OSWorld Flask server (:5000) — setup dispatch + eval getters (/execute, /accessibility)
    Bound on the container's loopback; reached from the host by ``curl``'d
    inside ``computer.interface.run_command``.

Usage:
    uv run python - <<'PY'
    import lite.gym as gym
    print(len(gym.registry.task_ids("lite.osworld", split="eval")))
    PY

Agent-facing vs env-facing CLI surface (north-star: ``lite.osworld == osworld``)
--------------------------------------------------------------------------------
The container's CLI surface is split into two tiers, and the SINGLE test for
which tier a tool belongs in is: **does the osworld VM's agent surface have it?**
Both tiers exist to make the container's *agent-observable* world identical to
the VM's — the split is a PARITY mechanism, not (primarily) a security one.

* **agent-facing** = ``/usr/bin`` (on the desktop user's login PATH). Must equal
  the VM's agent surface *at that task*: the stock-Ubuntu baseline PLUS whatever
  the task's own ``setup`` installs. Verified against a live VM: the baseline
  ships NONE of jq / xsel / xclip / pandoc / pdftk / ImageMagick / xdotool /
  sysstat — every such tool is **absent until a task's ``apt install`` brings it
  in** (landing on ``/usr/bin``, agent-facing). So:
    - Do NOT pre-bake these into the image agent-facing (a permanent surface the
      VM lacks → an agent could ``which jq`` and distinguish lite from the VM in a
      task that never installs jq). Install them **on-demand**: let each task's
      setup ``apt install`` run for real. Runtime apt works in the container
      (verified: a task installing ``sysstat`` succeeds).
    - COROLLARY: a task that *uses* a CLI must *install* it in setup — exactly
      like the VM. Never rely on a pre-baked agent-facing copy. (A task whose only
      jq use is a ``[ -f x ] && echo … || jq …`` fallback is fine: on a fresh
      profile the echo branch runs and jq is never invoked, matching the VM.)

* **env-facing** = ``/opt/env/bin`` (+ ``/opt/env/venv``), off the agent's login
  PATH. Reserved for tools the VM's agent surface does NOT have and that only the
  harness / eval machinery needs. Verified harness-only (VM baseline lacks all of
  them): ``xdotool`` (op_input drives X; the VM uses pyautogui and has no xdotool),
  ``ffmpeg`` + ImageMagick ``import``/``convert`` (op_screenshot capture +
  fallbacks, ``exec_stdio/server.py``), and the ``/opt/env/venv`` eval python.
  Hiding these makes the agent's world match the VM (which lacks them) — hiding is
  PARITY, not over-isolation. Enforcement is **PATH-based (soft)**, NOT a uid wall:
  the exec-stdio server runs *as the desktop user* and prepends
  ``/opt/env/venv/bin:/opt/env/bin`` to its command PATH (``exec_stdio/server.py``
  module-level), so ``op_run_command`` (setup/eval) resolves them by bare name; the
  agent's interactive *login* PATH is ``/usr/bin`` only, so a GUI terminal never
  surfaces them by name. ``/opt/env`` is ``a+rX`` (user-readable, no write) — the
  separation keeps harness tools off the agent's *observed* surface, not behind a
  permission boundary.

Why no double-install is needed. ``op_run_command``'s PATH is a SUPERSET —
``/opt/env/venv/bin:/opt/env/bin:`` + the inherited ``…:/usr/bin:…``
(``exec_stdio/server.py`` prepends). So a tool at ``/usr/bin`` is reachable by BOTH
the agent (login PATH = /usr/bin) AND setup/eval (server PATH superset). Therefore
setup/eval ``apt install`` → ``/usr/bin`` once is enough; the server finds it there.
``/opt/env`` holds only the harness-only tools kept off the agent's login PATH.

Image comparators (``src/eval/metrics.py``) are pure-Python (PIL / skimage /
python-docx / openpyxl); they do NOT shell out to ``convert``/``identify``. The
only eval getter that shells to a CLI is the clipboard read (``xsel --clipboard
--output`` per upstream ``basic_os.is_in_vm_clickboard``), and the sole task using
it installs ``xsel`` itself — so on-demand install preserves eval correctness.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, ClassVar

from lite.core.messages.final import ENV_INTERNAL_TERMINATE_REASON
from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.calls import (
    RuntimeEnvAction,
    tool_call_name,
)
from lite.core.tools.extra_tools import LiteFinishToolSet, make_report_infeasible_tool
from lite.core.tools.results import LiteToolResult, make_tool_result
from lite.core.tools.schemas import BaseTools
from lite.gym.envs.lite.osworld import exclude_reasons
from lite.gym.errors import EnvDepsMissingError
from lite.gym.registry import registry
from lite.gym.sandbox import SandboxBaseEnv, SandboxTaskConfig, register_jsonl_tasks
from lite.gym.types import (
    LiteEnvObservation,
    LiteEnvStepResult,
    LiteExecutedAction,
)
from lite.gym.utils import config as env_config
from lite.gym.utils.feedback.errors import (
    ToolErrorFeedback,
    current_feedback,
    unavailable_action_message,
)
from lite.gym.utils.feedback.ingress import (
    classify_standalone_tool_call,
    make_internal_terminate_action,
    prepare_env_tool_calls,
)
from lite.gym.utils.feedback.results import (
    ordered_tool_call_ids,
)
from lite.gym.utils.feedback.surface import merge_extra_tool_schemas

logger = logging.getLogger(__name__)

ENV_DIR = str(Path(__file__).parent)
CFG = env_config.load(ENV_DIR)

# ============================================================================
# Config defaults — every value below is read once from configs/default.yaml
# via env_config.load(ENV_DIR). Swap the whole file at startup with
# LITE_OSWORLD_CONFIG=<abs-path | bundled-name>. A rollout's env_kwargs still
# override per run; these are only registration defaults.
# ============================================================================
# --- env_kwargs (per-instance) ---
_MAX_STEPS = CFG.env_kwargs["max_steps"]
# step_timeout lives in CFG.make_kwargs (env-wide make default), applied via
# registry.set_env_make_kwargs below — not a per-task register arg.
_POST_ACTION_DELAY = CFG.env_kwargs["post_action_delay"]
_NOISE = CFG.env_kwargs["noise"]
_SEED = CFG.env_kwargs["seed"]
_EXTRA_TOOLS = CFG.env_kwargs["extra_tools"]
_IMAGE = CFG.env_kwargs["computer"]["image"]
# Oracle-only keys in ``metadata.others``: the ground-truth solution
# (``oracle_actions`` / ``oracle_trajectory``) + its postconfig flag. These are
# needed ONLY by the oracle-validation harness (devs/envs/*/validate/oracle),
# never by rollout/SFT. They are also free-form nested structures PyArrow can't
# infer a stable struct schema for (e.g. ScaleCUA's ``oracle_actions[].parameters.command``
# is a shell str in one action and an argv list in another → ``ArrowInvalid:
# cannot mix list and non-list`` on trajectory.parquet save, dropping the whole
# row). So they are STRIPPED from the exposed metadata by default and re-added
# only when the ``expose_oracle`` env_kwarg is set (see bind / _runtime_metadata).
# Inherited by lite.scalecua (subclasses LiteOsworldEnv without overriding these).
_ORACLE_METADATA_KEYS = ("oracle_actions", "oracle_trajectory", "oracle_after_postconfig")
# Composed computer_config (reused 3-5×). ``display`` is the "WxH" STRING the
# container shape expects, built from the yaml display_resolution list.
# ``timeout`` is the boot-readiness ceiling — how long the DockerProvisioner
# polls /tmp/gnome-ready over ``docker exec`` (heavy concurrent boots compete
# for host NVMe IOPS); matches the framework ``reset_timeout``.
_c = CFG.env_kwargs["computer"]
_COMPUTER_CONFIG = {
    "image": _IMAGE,
    "display": f"{_c['display_resolution'][0]}x{_c['display_resolution'][1]}",
    "memory": _c["memory"],
    "cpu": _c["cpu"],
    "timeout": _c["timeout"],
    # Pin the guest hostname to the OSWorld reference VM's so terminal prompts/titles
    # and `hostname`/`uname -n` read `user@user-virtual-machine` (the DockerProvisioner
    # passes this as `docker run --hostname`; a Dockerfile can't set it in a container).
    "hostname": "user-virtual-machine",
}
# server_kwargs: none — reaper/pool/admission are framework-owned.
# ============================================================================

_DATA_DIR = Path(__file__).parent / "data"
_README = "lite/gym/envs/lite/osworld/README.md"
_CATALOG_LOCK = _DATA_DIR / "catalog.lock.json"


def _catalog_dep_error(what: str):
    from lite.gym.errors import EnvDepsMissingError

    return EnvDepsMissingError(
        what=what,
        install=(
            "uv run --no-sync bash "
            "lite/gym/envs/lite/osworld/scripts/install.sh"
        ),
        see=_README,
    )


def _catalog_lock_entries() -> dict[str, Any]:
    if not _CATALOG_LOCK.is_file():
        raise _catalog_dep_error(f"missing lite.osworld catalog lock: {_CATALOG_LOCK}")
    try:
        lock = json.loads(_CATALOG_LOCK.read_text())
        splits = lock["splits"]
        if not isinstance(splits, dict):
            raise ValueError("catalog lock 'splits' must be an object")
        return splits
    except Exception as exc:
        raise _catalog_dep_error(f"invalid lite.osworld catalog lock: {exc}") from exc


def _check_catalog_lock(splits: set[str] | None = None) -> None:
    """Verify generated OSWorld catalogs before registration."""
    entries = _catalog_lock_entries()
    wanted = splits or set(entries)
    for split in wanted:
        entry = entries.get(split)
        if entry is None:
            raise _catalog_dep_error(f"missing lite.osworld {split} entry in catalog lock")
        _check_catalog_entry(split, entry)


def _check_catalog_entry(split: str, entry: dict[str, Any]) -> None:
    if not isinstance(entry, dict):
        raise _catalog_dep_error(f"invalid lite.osworld {split} catalog lock entry")
    try:
        rel_path = entry["path"]
        expected_rows = entry["rows"]
        expected_sha = entry["sha256"]
        expected_excluded = entry["excluded_rows"]
        expected_scored = entry["scored_rows"]
        expected_reasons = entry["exclude_reasons"]
    except KeyError as exc:
        raise _catalog_dep_error(
            f"invalid lite.osworld {split} catalog lock entry: missing {exc.args[0]}"
        ) from exc
    if (
        not isinstance(rel_path, str)
        or not rel_path
        or Path(rel_path).is_absolute()
    ):
        raise _catalog_dep_error(
            f"invalid lite.osworld {split} catalog lock entry: bad path"
        )
    if not isinstance(expected_rows, int) or isinstance(expected_rows, bool):
        raise _catalog_dep_error(
            f"invalid lite.osworld {split} catalog lock entry: bad rows"
        )
    if not isinstance(expected_excluded, int) or isinstance(expected_excluded, bool):
        raise _catalog_dep_error(
            f"invalid lite.osworld {split} catalog lock entry: bad excluded_rows"
        )
    if not isinstance(expected_scored, int) or isinstance(expected_scored, bool):
        raise _catalog_dep_error(
            f"invalid lite.osworld {split} catalog lock entry: bad scored_rows"
        )
    if (
        expected_excluded < 0
        or expected_scored < 0
        or expected_excluded + expected_scored != expected_rows
    ):
        raise _catalog_dep_error(
            f"invalid lite.osworld {split} catalog lock entry: inconsistent counts"
        )
    if (
        not isinstance(expected_reasons, dict)
        or any(not isinstance(k, str) for k in expected_reasons)
        or any(not isinstance(v, int) or isinstance(v, bool) or v < 0
               for v in expected_reasons.values())
    ):
        raise _catalog_dep_error(
            f"invalid lite.osworld {split} catalog lock entry: bad exclude_reasons"
        )
    for reason in expected_reasons:
        try:
            exclude_reasons.validate(reason)
        except ValueError as exc:
            raise _catalog_dep_error(
                f"invalid lite.osworld {split} catalog lock entry: bad exclude_reasons"
            ) from exc
    if sum(expected_reasons.values()) != expected_excluded:
        raise _catalog_dep_error(
            f"invalid lite.osworld {split} catalog lock entry: inconsistent counts"
        )
    if not isinstance(expected_sha, str) or not expected_sha:
        raise _catalog_dep_error(
            f"invalid lite.osworld {split} catalog lock entry: bad sha256"
        )
    path = _DATA_DIR / rel_path
    if not path.is_file():
        raise _catalog_dep_error(f"missing lite.osworld {split} catalog: {path}")
    data = path.read_bytes()
    rows, excluded, reasons = _summarize_catalog_counts(data, path=path)
    digest = hashlib.sha256(data).hexdigest()
    scored = rows - excluded
    if (
        rows != expected_rows
        or excluded != expected_excluded
        or scored != expected_scored
        or reasons != expected_reasons
        or digest != expected_sha
    ):
        raise _catalog_dep_error(
            "stale lite.osworld "
            f"{split} catalog: run scripts/utils/tasks.sh generate && "
            "scripts/utils/tasks.sh check; if the generator change is intentional, "
            "run scripts/utils/tasks.sh refresh-lock in a reviewed commit"
        )


def _summarize_catalog_counts(
    data: bytes,
    *,
    path: Path,
) -> tuple[int, int, dict[str, int]]:
    rows = 0
    excluded = 0
    reasons: dict[str, int] = {}
    for line_number, line in enumerate(data.splitlines(), 1):
        if not line.strip():
            continue
        rows += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _catalog_dep_error(
                f"{path}:{line_number}: invalid JSON: {exc}"
            ) from exc
        reason = (
            row.get("metadata", {})
            .get("others", {})
            .get("exclude_reason")
        )
        if reason:
            try:
                exclude_reasons.validate(reason)
            except ValueError as exc:
                raise _catalog_dep_error(
                    f"{path}:{line_number}: invalid exclude_reason {reason!r}: {exc}"
                ) from exc
            excluded += 1
            reasons[reason] = reasons.get(reason, 0) + 1
    return rows, excluded, dict(sorted(reasons.items()))


def _check_docker_image(tag: str | None = None) -> None:
    """Verify the Docker image exists; raise ``EnvDepsMissingError`` if not.

    Thin wrapper around the shared
    :func:`lite.gym.utils.backend.docker.require_image_present` so direct-mode
    rollouts and the server-mode ``ensure_services`` hook share one
    install-hint format.
    """
    from lite.gym.utils.backend.docker import require_image_present
    from lite.gym.utils.backend.freshness import image_for
    require_image_present(image_for("lite.osworld", tag=tag or _IMAGE))


def _check_desktop_env() -> None:
    """Verify upstream OSWorld ``desktop_env`` is importable.

    Evaluators in ``src/eval/metrics.py`` delegate to ``desktop_env``; this
    check exists precisely so a missing install fails loudly HERE instead of
    every task silently scoring 0 at eval time.

    Probe at ``gym.make`` time (via ``__init__`` for direct-mode and
    ``ensure_services`` for env-server startup), **not** at module load
    — task-metadata enumeration via ``gym.registry.task_ids`` must work
    without this runtime dep so lightweight tooling (e.g. dataset export
    scripts) can run on minimal installs.
    """
    try:
        import desktop_env  # noqa: F401
    except ImportError:
        from lite.gym.errors import EnvDepsMissingError
        raise EnvDepsMissingError(
            what=(
                "OSWorld (desktop_env) package not installed — required for "
                "lite.osworld evaluators"
            ),
            install="uv run --no-sync bash lite/gym/envs/lite/osworld/scripts/install.sh",
            see=_README,
        )

class LiteOsworldTools(BaseTools):
    """What lite.osworld declares beyond the GUI surface: ``report_infeasible``.

    ``bash`` is NOT here and never will be: it is granted by the sandbox
    TRANSPORT through ``SandboxBaseEnv.extra_tool_schemas``'s ``executable``
    set, not declared by any one env.
    """

    _SCHEMAS: ClassVar[dict[str, dict[str, Any]]] = {
        "report_infeasible": make_report_infeasible_tool(),
    }


#: The local report_infeasible classification path only — deliberately WITHOUT
#: ``bash`` (the family-wide union that carries it is
#: ``SandboxBaseEnv.known_standalone_tool_names()``). Finish tools cannot live
#: in an env's own set, so the union is not optional.
_KNOWN_STANDALONE_TOOL_NAMES = (
    LiteOsworldTools.get_tool_names() | LiteFinishToolSet.get_tool_names()
)


from lite.gym.envs.lite.osworld.src.utils.setup import setup_fn  # noqa: E402
from lite.gym.envs.lite.osworld.src.utils.verify import evaluate_final_fn  # noqa: E402


def _ensure_services(env_id: str) -> None:
    """env-server startup hook: verify runtime prerequisites are in place
    before accepting traffic. Mirror of android_{lab,world}'s same-named
    hook. Either failure surfaces as ``EnvDepsMissingError`` → HTTP 501
    on the first ``POST /instances`` and ``available: false`` on
    ``GET /envs``.

    Two checks: the Docker image (for ``gym.make`` to spawn containers)
    and the ``desktop_env`` Python package (for evaluators in
    ``src/eval/metrics.py``). Both must pass before the env is reachable.
    """
    _check_desktop_env()
    _check_docker_image()


class LiteOsworldEnv(SandboxBaseEnv):
    """lite.osworld env with infeasible task handling + optional noise injection.

    Extends SandboxBaseEnv to handle report_infeasible, terminate(status=failure),
    and response([INFEASIBLE]) — matching OSWorld's infeasible task logic.

    Pass ``noise=True`` via ``gym.make(..., noise=True)`` to enable per-reset
    visual noise injection. Noise candidates are read from ``task.metadata["noise"]``
    when present (train tasks); eval tasks fall back to NOISE_CANDIDATES[domain].
    Default is off.
    """

    #: env-owned extras; ``bash`` + the finish tools come from the base resolver.
    EXTRA_TOOLS: ClassVar[type[BaseTools]] = LiteOsworldTools

    def __init__(
        self,
        task: Any = None,
        *,
        display_resolution: tuple[int, int] = (1920, 1080),
        image: str | None = None,
        vnc_port: int | None = None,
        env_id: str | None = "lite.osworld",
        setup_fn: Any = None,
        evaluate_step_fn: Any = None,
        evaluate_final_fn: Any = None,
        container: str | None = None,
        computer_config: dict[str, Any] | None = None,
        post_action_delay: float = 1.0,
        debug: bool = False,
        max_steps: int | None = None,
        noise: bool = _NOISE,
        seed: int | None = _SEED,
        valid_actions: list[str] | None = None,
        extra_tools: list[str] | None = _EXTRA_TOOLS,
        expose_oracle: bool = False,
        domain_overrides: dict[str, dict] | None = None,
    ) -> None:
        """Construct a lite.osworld env.

        Defaults ``computer_config`` + ``env_id``, runs the desktop-env prereq
        probe, then initializes sandbox constructor state before binding soft
        kwargs.

        The probe is cheap (one ``import`` attempt) and surfaces a
        missing prereq before any container spawn — direct-mode
        catches it here; server-mode catches the same via
        ``ensure_services``.
        """
        _check_desktop_env()
        if computer_config is None:
            computer_config = _COMPUTER_CONFIG
        self._set_constructor_state(
            display_resolution=display_resolution,
            image=image,
            vnc_port=vnc_port,
            env_id=env_id,
            setup_fn=setup_fn,
            evaluate_step_fn=evaluate_step_fn,
            evaluate_final_fn=evaluate_final_fn,
            container=container,
            computer_config=computer_config,
            post_action_delay=post_action_delay,
            debug=debug,
        )
        self.bind(
            task,
            max_steps=max_steps,
            noise=noise,
            seed=seed,
            valid_actions=valid_actions,
            extra_tools=extra_tools,
            expose_oracle=expose_oracle,
            domain_overrides=domain_overrides,
        )

    def bind(
        self,
        task: Any = None,
        *,
        max_steps: int | None = None,
        noise: bool = _NOISE,
        seed: int | None = _SEED,
        valid_actions: list[str] | None = None,
        extra_tools: list[str] | None = _EXTRA_TOOLS,
        expose_oracle: bool = False,
        # ── internal / registration ──
        domain_overrides: dict[str, dict] | None = None,
    ) -> None:
        """Bind a task (object or task_id) and apply soft kwargs.

        Accepts EITHER a :class:`SandboxTaskConfig` (direct-mode
        tests, registry factories) OR a ``task_id`` string (compatibility/tests);
        the latter is resolved via the env's ``lookup_task`` registry.

        All soft kwargs are assigned UNCONDITIONALLY — the defaults
        in this signature are the SINGLE source of truth for what
        each soft kwarg defaults to when absent. Direct and server construction
        both route through this method, so the body always runs with the caller's
        exact value (None included) and the two modes produce byte-equal state.
        ``seed`` field-name
        remap (stored as ``_noise_seed``) and ``extra_tools``
        tool-set resolution live here, next to the field they touch.

        Unknown kwargs raise ``TypeError`` automatically — the
        compatibility bind paths thus fail fast at this boundary if env_kwargs
        the env doesn't understand are passed.
        """
        # task=None or task="" is the taskless seed shape (construction always
        # invokes bind once, even with no kwargs); skip the task-registry lookup
        # and let super().bind handle the None-task branch.
        if isinstance(task, str) and task:
            from lite.gym.sandbox import lookup_task
            env_id = self._env_id or "lite.osworld"
            task = lookup_task(env_id, task)
        elif task == "":
            task = None
        # Unconditional assignment — see method docstring.
        self._noise = noise
        # Scoped to this env's noise rng only — does NOT call
        # random.seed() globally. None → each reset draws fresh
        # randomness from time_ns().
        self._noise_seed = seed
        # ``extra_tools`` resolution is the BASE resolver's job (it grants the
        # canonical ``bash`` slice, which no env tool set may declare); this env
        # only contributes ``EXTRA_TOOLS``.
        # Opt-in re-exposure of oracle-only metadata keys (default off, so
        # rollout/SFT trajectories never carry the ground-truth solution and
        # never trip the PyArrow struct-inference crash — see _ORACLE_METADATA_KEYS).
        self._expose_oracle = expose_oracle
        self._domain_overrides = domain_overrides
        super().bind(task, max_steps=max_steps,
                     valid_actions=valid_actions, extra_tools=extra_tools)
        # Apply domain-override env_kwargs (e.g. ``max_steps=30`` for
        # ``multi_apps``) AFTER the task is bound so the override
        # key can look up by task metadata. Each override maps to
        # ``self._<key>`` directly — these are baseline-instance
        # fields known to this env class.
        if self._domain_overrides and task is not None:
            task_domain = task.metadata.get("others", {}).get("domain", "")
            for k, v in self._domain_overrides.get(task_domain, {}).items():
                setattr(self, f"_{k}", v)

    @staticmethod
    def _task_metadata(task: SandboxTaskConfig) -> LiteCUAMetadata:
        """Same-source metadata builder: the
        sandbox builder + extra_tool_schemas mirroring bind()'s default
        resolution, so the yaml default flows to BOTH sides. Oracle-only keys
        (see ``_ORACLE_METADATA_KEYS``) are STRIPPED here — the default exposed
        metadata carries no ground-truth solution; ``_runtime_metadata`` re-adds
        it only under the ``expose_oracle`` env_kwarg."""
        md = SandboxBaseEnv._task_metadata(task)
        return dataclasses.replace(
            md,
            # CONCATENATE, don't replace: the base builder's slice (the task's
            # own extra_tool_schemas) must survive alongside the yaml default.
            extra_tool_schemas=merge_extra_tool_schemas(
                md.extra_tool_schemas,
                LiteOsworldEnv.extra_tool_schemas(_EXTRA_TOOLS),
            ),
            others={k: v for k, v in md.others.items() if k not in _ORACLE_METADATA_KEYS},
        )

    def _runtime_metadata(self) -> LiteCUAMetadata:
        # env_kwargs amendment on top of the shared builder: extra_tool_schemas
        # is resolved from the ``extra_tools`` env_kwarg at bind (no-override
        # default matches the builder above). ``expose_oracle`` re-adds the
        # oracle-only keys (stripped by _task_metadata) from the raw task
        # config — used by the oracle-validation harness only.
        md = super()._runtime_metadata()
        others = dict(md.others)
        if self._expose_oracle and self._task is not None:
            raw = self._task.metadata.get("others", {})
            for k in _ORACLE_METADATA_KEYS:
                if k in raw:
                    others[k] = copy.deepcopy(raw[k])
        # ``super()._runtime_metadata()`` already merged the env-kwarg extra
        # tools. This override only owns optional oracle metadata exposure.
        return dataclasses.replace(md, others=others)

    async def step(self, actions: list[RuntimeEnvAction]) -> LiteEnvStepResult:
        input_actions = actions
        metadata = self.metadata
        ordered_call_ids = ordered_tool_call_ids(input_actions)
        remapped_actions: list[RuntimeEnvAction] = []
        report_accepted_actions: list[dict[str, Any]] = []
        report_result_call_ids: list[str | None] = []
        local_feedback: dict[str, ToolErrorFeedback] = {}
        local_executed_actions: list[LiteExecutedAction] = []
        handled_local_report = False

        for action, original_call_id in zip(input_actions, ordered_call_ids, strict=True):
            try:
                name = tool_call_name(action)
            except (KeyError, TypeError):
                remapped_actions.append(action)
                continue
            if name == "report_infeasible":
                prepared_actions, ingress_errors = prepare_env_tool_calls(
                    [action],
                    metadata,
                )
                local_feedback.update(ingress_errors)
                if not prepared_actions:
                    handled_local_report = True
                    reason = "invalid report_infeasible call"
                    if original_call_id and original_call_id in ingress_errors:
                        reason = ingress_errors[original_call_id].message
                    local_executed_actions.append({
                        "call": "noop",
                        "args": {"name": name, "reason": reason},
                    })
                    continue
                prepared, result_call_id = prepared_actions[0]
                tool_availability = classify_standalone_tool_call(
                    prepared,
                    _KNOWN_STANDALONE_TOOL_NAMES,
                    metadata.extra_tool_schemas,
                )
                if tool_availability == "inactive":
                    handled_local_report = True
                    if result_call_id:
                        local_feedback[result_call_id] = current_feedback(
                            unavailable_action_message("report_infeasible")
                        )
                    local_executed_actions.append({
                        "call": "noop",
                        "args": {
                            "name": name,
                            "reason": "inactive extra tool",
                        },
                    })
                    continue
                args = prepared["arguments"]
                action = make_internal_terminate_action(
                    status="failure",
                    reason=str(args.get("reason", "")),
                    internal_reason=ENV_INTERNAL_TERMINATE_REASON,
                    result_call_id=result_call_id,
                )
                prepared_terminate, terminate_errors = prepare_env_tool_calls(
                    [action],
                    metadata,
                )
                local_feedback.update(terminate_errors)
                if prepared_terminate:
                    report_accepted_actions.append(prepared_terminate[0][0])
                report_result_call_ids.append(result_call_id)
                remapped_actions.append(action)
                continue
            remapped_actions.append(action)

        if (local_feedback or handled_local_report) and not remapped_actions:
            continue_call_ids = [call_id for call_id in ordered_call_ids if call_id]
            current_image = None
            computer = getattr(self, "_computer", None)
            if (
                computer is not None
                and (
                    continue_call_ids
                    or any(feedback.carrier == "current" for feedback in local_feedback.values())
                )
            ):
                from lite.gym.sandbox.base import _take_screenshot

                current_image = await _take_screenshot(computer)
            return await self._finalize_step_result(
                ordered_call_ids=ordered_call_ids,
                continue_call_ids=continue_call_ids,
                images=[] if current_image is None else [current_image],
                accepted_actions=[],
                executed_actions=local_executed_actions,
                feedback=local_feedback,
            )

        if report_result_call_ids and len(remapped_actions) == len(report_result_call_ids):
            current_image = None
            computer = getattr(self, "_computer", None)
            if (
                computer is not None
                and (
                    any(call_id for call_id in ordered_call_ids)
                    or any(feedback.carrier == "current" for feedback in local_feedback.values())
                )
            ):
                from lite.gym.sandbox.base import _take_screenshot

                current_image = await _take_screenshot(computer)
            return await self._finalize_step_result(
                ordered_call_ids=ordered_call_ids,
                continue_call_ids=ordered_call_ids,
                images=[] if current_image is None else [current_image],
                accepted_actions=report_accepted_actions[:1],
                executed_actions=local_executed_actions,
                feedback=local_feedback,
                terminated=True,
                stop_reason="terminate",
            )

        result = await super().step(remapped_actions)
        if local_feedback or report_result_call_ids:
            delegated_results = list(result.results or [])
            delegated = {
                tool_result.tool_call_id: tool_result
                for tool_result in delegated_results
                if tool_result.tool_call_id is not None
            }
            current_image = None
            for tool_result in delegated_results:
                if tool_result.images:
                    current_image = tool_result.images[-1]
                    break
            needs_local_current_image = any(
                feedback.carrier == "current"
                for feedback in local_feedback.values()
            )
            consumed_unpaired_result_indexes: set[int] = set()
            for result_call_id in report_result_call_ids:
                if not result_call_id or result_call_id in delegated:
                    continue
                for index, tool_result in enumerate(delegated_results):
                    if index in consumed_unpaired_result_indexes:
                        continue
                    if tool_result.tool_call_id is not None:
                        continue
                    delegated[result_call_id] = make_tool_result(
                        tool_call_id=result_call_id,
                        images=tool_result.images,
                        text=tool_result.text,
                        metadata=tool_result.metadata,
                        error=tool_result.error,
                    )
                    consumed_unpaired_result_indexes.add(index)
                    if current_image is None and tool_result.images:
                        current_image = tool_result.images[-1]
                    break
            missing_report_call_ids = [
                result_call_id
                for result_call_id in report_result_call_ids
                if result_call_id
                and result_call_id not in delegated
                and result_call_id not in local_feedback
            ]
            if (
                current_image is None
                and (needs_local_current_image or missing_report_call_ids)
            ):
                computer = getattr(self, "_computer", None)
                if computer is not None:
                    from lite.gym.sandbox.base import _take_screenshot

                    current_image = await _take_screenshot(computer)
            for result_call_id in missing_report_call_ids:
                delegated[result_call_id] = make_tool_result(
                    tool_call_id=result_call_id,
                    images=[] if current_image is None else [current_image],
                )
            used_call_ids: set[str] = set()
            ordered_results: list[LiteToolResult] = []
            for call_id in ordered_call_ids:
                if call_id is None:
                    continue
                if call_id in local_feedback:
                    feedback = local_feedback[call_id]
                    if feedback.carrier == "current":
                        ordered_results.append(make_tool_result(
                            tool_call_id=call_id,
                            images=[] if current_image is None else [current_image],
                            metadata={"is_error": True},
                            error=feedback.message,
                        ))
                    else:
                        ordered_results.append(make_tool_result(
                            tool_call_id=call_id,
                            metadata={"is_error": True},
                            error=feedback.message,
                        ))
                    used_call_ids.add(call_id)
                elif call_id in delegated:
                    ordered_results.append(delegated[call_id])
                    used_call_ids.add(call_id)
            for index, tool_result in enumerate(delegated_results):
                call_id = tool_result.tool_call_id
                if call_id is None and index in consumed_unpaired_result_indexes:
                    continue
                if call_id is None or call_id not in used_call_ids:
                    ordered_results.append(tool_result)
            result.results = ordered_results
        return result

    async def reset(self) -> LiteEnvObservation:
        image = _IMAGE
        if self._computer_config is not None:
            image = str(self._computer_config.get("image", _IMAGE))
        await asyncio.to_thread(_check_docker_image, image)
        from lite.gym.errors import EnvDesktopCrashed
        # Cold-vs-existing signal for the post-boot liveness check below:
        # ``self._computer`` is None until ``SandboxBaseEnv.boot()`` (called
        # inside ``super().reset()``) first acquires it, and boot() is
        # idempotent — a no-op once acquired. So "None right now" means
        # THIS reset() call is the one that will perform the actual docker
        # run + DockerProvisioner._wait_ready(/tmp/gnome-ready) poll; "not
        # None" means a container was already attached before this call, usually
        # from a prior reset() on this same instance — i.e. not a fresh cold boot.
        was_warm = self._computer is not None
        # No boot-failure diagnostic read here: every raising path out of
        # ``SandboxBaseEnv.boot`` runs ``_attempt_boot_computer``'s ``except
        # BaseException: await self._release_failed_attempt()``, which ``docker rm -f``s
        # the container and nulls ``_container_name`` before the exception reaches this
        # frame — there is nothing left to exec into. The live GNOME-crash signal is the
        # post-boot ``_desktop_alive()`` probe below.
        result = await super().reset()
        # Post-boot desktop liveness. The one-shot gnome-ready watchdog
        # (lite/gym/sandbox/docker/Dockerfile.linux: wait-for-gnome-ready.sh) writes
        # /tmp/gnome-ready the instant gnome-shell FIRST comes up, then exits — so a
        # session that comes up and then crashes a moment later (host inotify
        # exhaustion under concurrent boots) slips past it. super().reset() then
        # returns a "successful" screenshot of the "Oh no, something has gone wrong"
        # GNOME fallback; the agent flails on a dead desktop and the task SILENTLY
        # scores reward=0. Re-verify gnome-shell is STILL up; if not, raise the typed
        # EnvDesktopCrashed (retryable) so the message propagates to the client's
        # error.txt and the rollout re-runs on a fresh container instead of
        # writing a poisoned reward-0 summary.
        #
        # Worth the extra exec round-trip when a container is already attached
        # (``was_warm`` is the legacy local flag name): the desktop may have sat
        # idle long enough for gnome-shell to crash after the ready marker was
        # written. ALSO on the injected-container attach
        # path (``not self._owns_computer``): ``boot()`` took the
        # ``_container_override`` branch → ``attach()`` (session start + ping
        # only), which never ran the gnome-ready poll, so this is the sole
        # desktop-liveness confirmation there. On a fresh COLD boot
        # (owned + ``not was_warm``), ``super().reset()`` above just ran
        # ``DockerProvisioner._wait_ready`` seconds ago, confirming gnome-shell
        # up — re-checking would be a redundant exec-transport round-trip +
        # in-container pgrep spawn on every reset for no additional signal.
        if (was_warm or not self._owns_computer) and not await self._desktop_alive():
            raise EnvDesktopCrashed(
                "lite.osworld desktop crashed post-boot: the GNOME-Shell session "
                "is down ('Oh no, something has gone wrong' fallback — typically "
                "host inotify exhaustion under concurrent boots). Container is "
                "unusable; retrying on a fresh one."
            )
        if self._noise:
            await self._inject_noise()
            # Brief pause so window manager completes any resize/move before
            # the screenshot is taken (compositing is async).
            await asyncio.sleep(1.0)
            # Retake screenshot so the returned observation reflects the noise state.
            from lite.gym.sandbox.base import _take_screenshot
            screenshot = await _take_screenshot(self._computer)
            result = LiteEnvObservation(
                image=screenshot,
                text=result.text,
            )
        # No ownership handover — the server runs as the desktop user, so every setup
        # step (config / settings.json / _inject_noise) creates user-owned files by
        # construction. (A task's own `sudo` into $HOME is a data-side concern.)
        return result

    async def _desktop_alive(self) -> bool:
        """True if the GNOME-Shell session is healthy.

        Mirrors the boot watchdog's own definition of "ready"
        (lite/gym/sandbox/docker/Dockerfile.linux: gnome-shell running, no
        gnome-session-failed fail-whale) but runs AFTER boot, to catch a session
        that came up then crashed — the "Oh no, something has gone wrong"
        fallback. A probe failure returns True: an exec hiccup must not be
        misread as a desktop crash.
        """
        if not self._computer:
            return True
        try:
            # gnome-shell is the WM + compositor + top panel + dash in one process, so
            # its liveness IS the desktop's — and the session's RequiredComponents are
            # trimmed to org.gnome.Shell (base Dockerfile), so nothing else can trip the
            # "Oh no" fallback: gnome-shell down IS the crash. Use ``pgrep -x`` (exact
            # process-NAME match), NOT ``pgrep -f``: -f scans full cmdlines, including
            # the bash that ``run_command`` spawns to run THIS check — a ``-f`` naming
            # the crash process would self-match this very command and always say "down".
            res = await self._computer.interface.run_command(
                "pgrep -x gnome-shell >/dev/null && echo OK || echo DOWN"
            )
            return "OK" in (getattr(res, "stdout", "") or "")
        except Exception:
            return True

    async def _inject_noise(self) -> None:
        """Inject visual noise per metadata.noise config.

        Noise types:
          mouse_jitter     — move cursor to a random position
          background_app   — launch xeyes or xclock
          fake_file        — create a dummy file on the Desktop
          notification     — pop up a desktop notification
          window_resize    — resize the active window to a random fraction of the screen
          window_move      — move the active window to a random position
          open_terminal    — open a terminal window (partially covers the app)
          desktop_files    — scatter several fake files on ~/Desktop
          extra_tab        — open extra browser tabs in Chrome (if running)

        If the task has no per-task noise config (e.g. eval tasks), falls back to
        NOISE_CANDIDATES[domain] so that ``noise=True`` works uniformly across
        eval and train tasks without modifying the task JSON.
        """
        noise_cfg = self._task.metadata.get("noise")
        if not noise_cfg:
            # Fallback: derive domain-appropriate noise from NOISE_CANDIDATES
            from lite.gym.envs.lite.osworld.src.gen.common import NOISE_CANDIDATES
            domain = (self._task.metadata.get("others") or {}).get("domain", "")
            noise_cfg = NOISE_CANDIDATES.get(domain)
        if not noise_cfg or not self._computer:
            return
        candidates = noise_cfg.get("candidates", [])
        max_apply = noise_cfg.get("max_apply", 2)
        if not candidates:
            return

        import random
        import time
        rng = random.Random(self._noise_seed if self._noise_seed is not None else time.time_ns())
        n = rng.randint(1, max_apply)  # at least 1 noise (was 0–max)
        chosen = rng.sample(candidates, min(n, len(candidates)))

        run = self._computer.interface.run_command
        for noise_type in chosen:
            try:
                if noise_type == "mouse_jitter":
                    x, y = rng.randint(50, 1870), rng.randint(50, 1030)
                    await run(f"xdotool mousemove {x} {y}")

                elif noise_type == "background_app":
                    app = rng.choice(["xeyes", "xclock"])
                    await run(
                        f"bash -c 'nohup {app} &>/dev/null & disown'")

                elif noise_type == "fake_file":
                    await run(
                        f"touch ~/Desktop/note_{rng.randint(0, 999)}.txt")

                elif noise_type == "notification":
                    await run(
                        f'notify-send "info" "{rng.randint(0, 1000)}"')

                elif noise_type in ("window_resize", "window_move"):
                    # Pick a sensible (w, h, x, y) placement: left/right half, top/bottom half,
                    # or slightly-smaller-than-full. Min size ≥ 960×720 so dialogs fit.
                    # Desktop is 1920×1080 (env_kwargs.display_resolution).
                    placements = [
                        (960, 1080,    0,   0),  # left half
                        (960, 1080,  960,   0),  # right half
                        (1920, 540,    0,   0),  # top half
                        (1920, 540,    0, 540),  # bottom half
                        (1700, 950,  100,  50),  # near-full, slight inset
                        (1600, 900,  160,  90),  # near-full, larger inset
                        (1500, 850,    0,   0),  # 1500×850 anchored top-left
                        # 1500×850 anchored bottom-right.
                        (1500, 850,  420, 230),
                    ]
                    w, h, x, y = rng.choice(placements)
                    wid = (await run("xdotool getactivewindow")).stdout.strip()
                    await run(f"wmctrl -ir {wid} -b remove,maximized_vert,maximized_horz")
                    await asyncio.sleep(1.0)
                    await run(f"xdotool windowsize {wid} {w} {h}")
                    await run(f"xdotool windowmove {wid} {x} {y}")

                elif noise_type == "open_terminal":
                    await run(
                        "bash -c 'nohup gnome-terminal &>/dev/null & disown'")

                elif noise_type == "desktop_files":
                    exts = ["txt", "pdf", "csv", "log", "bak"]
                    names = ["report", "backup", "data", "notes", "draft",
                             "temp", "old", "copy", "archive", "meeting"]
                    for _ in range(rng.randint(2, 5)):
                        fname = f"{rng.choice(names)}_{rng.randint(0,99)}.{rng.choice(exts)}"
                        await run(f"touch ~/Desktop/{fname}")

                elif noise_type == "extra_tab":
                    urls = [
                        "https://example.com",
                        "about:blank",
                        "https://www.wikipedia.org",
                    ]
                    url = rng.choice(urls)
                    await run(
                        f"bash -c 'nohup xdg-open \"{url}\" &>/dev/null & disown'")

            except Exception:
                pass  # noise is best-effort


# NOTE: _check_docker_image() is called in LiteOsworldEnv.reset(), not here.
# Import-time check would block other lite-* envs (e.g. lite.demo) on machines
# where cua-lite/lite.osworld:latest is not built.

# Env-wide make defaults declared once here, sourced from default.yaml
# make_kwargs. Per-task register() kwargs / gym.make() still override.
registry.set_env_make_kwargs("lite.osworld", CFG.make_kwargs)

_REGISTERED_CATALOG_KEYS: set[tuple[str, str]] = set()


def _register_jsonl_catalog(jsonl_path: Path, *, split: str) -> None:
    key = (str(jsonl_path), split)
    if key in _REGISTERED_CATALOG_KEYS:
        return
    register_jsonl_tasks(
        jsonl_path=jsonl_path,
        env_id="lite.osworld",
        split=split,
        computer=_COMPUTER_CONFIG,
        setup_fn=setup_fn,
        evaluate_final_fn=evaluate_final_fn,
        env_class=LiteOsworldEnv,
        factory_kwargs={"post_action_delay": _POST_ACTION_DELAY},
        # Registered metadata carries bind()'s no-override resolution
        # (LiteOsworldEnv._task_metadata calls the same
        # LiteOsworldEnv.extra_tool_schemas(_EXTRA_TOOLS) bind() resolves through);
        # tools are opt-in via env_kwargs.extra_tools (same-source contract; the
        # full catalog has no registered-side consumers).
        default_max_steps=_MAX_STEPS,
    )
    _REGISTERED_CATALOG_KEYS.add(key)


def _register_tasks(*, raise_if_none: bool = True) -> None:
    """Register every locally valid OSWorld split without import-time blast radius."""

    try:
        entries = _catalog_lock_entries()
    except EnvDepsMissingError as exc:
        if raise_if_none:
            raise
        logger.warning("lite.osworld catalog registration deferred: %s", exc.what)
        return

    errors: list[EnvDepsMissingError] = []
    valid_catalogs = 0

    def _try(lock_split: str, register_split: str) -> None:
        nonlocal valid_catalogs
        entry = entries.get(lock_split)
        if entry is None:
            errors.append(_catalog_dep_error(
                f"missing lite.osworld {lock_split} entry in catalog lock"
            ))
            return
        try:
            _check_catalog_entry(lock_split, entry)
            valid_catalogs += 1
            _register_jsonl_catalog(_DATA_DIR / entry["path"], split=register_split)
        except EnvDepsMissingError as exc:
            errors.append(exc)
            logger.warning(
                "lite.osworld %s catalog registration deferred: %s",
                lock_split,
                exc.what,
            )

    _try("eval", "eval")
    for _lock_split in ("train.synth", "train.perturb"):
        _try(_lock_split, _lock_split)
        _try(_lock_split, "train")

    if valid_catalogs == 0 and raise_if_none:
        if errors:
            raise errors[0]
        raise _catalog_dep_error("no lite.osworld catalogs registered")


_register_tasks(raise_if_none=False)

from lite.gym.remote.reaper import ContainerServices  # noqa: E402
from lite.gym.services import register_services  # noqa: E402


class LiteOsworldServices(ContainerServices):
    """Env-server capabilities for lite.osworld: docker-image presence check
    (``ensure``) + per-container reconciliation (``live_ids``/``reap`` inherited
    from :class:`ContainerServices`)."""

    def ensure(self, env_id: str) -> None:
        _ensure_services(env_id)

    def register_tasks(self, env_id: str) -> None:
        _register_tasks()


register_services("lite.osworld", LiteOsworldServices())
from lite.gym.services import BackendFamily, register_family  # noqa: E402

register_family("lite.osworld", BackendFamily.DEDICATED)
