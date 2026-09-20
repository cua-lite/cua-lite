"""
SandboxBaseEnv — Base Computer Use benchmark environment.

Self-contained task environment with a gym-like reset/step/close interface.
Host↔container transport is **exec-stdio** (lite/gym/sandbox/exec_stdio/):
one persistent ``docker exec -i`` JSON-lines session per container, talking
to a lean in-container server (xdotool/ffmpeg/shell).

Lifecycle is constructor/soft aware: construction sets immutable post-boot
fields (container config, callbacks, display resolution, etc.) and ``bind``
stamps task identity + soft kwargs.
``boot()`` is idempotent and task-independent (reads only constructor fields).
``reset()`` composes the two. The env-server cold-constructs the caller's real
key through ``gym.make``. Direct-mode ``EnvCls(task=t, **kwargs)`` partitions
kwargs at the env class and ends up running constructor-state setup + ``bind``
under the hood.
"""

from __future__ import annotations

import asyncio
import atexit
import dataclasses
import inspect
import logging
import threading
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any, ClassVar

from lite.core.messages.final import (
    MAX_STEPS_STOP_REASON,
    STOP_REASON_INFO_KEY,
)
from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.action_space import LITE_DESKTOP_KEY_ACTION_NAMES, LiteDesktopActionSet
from lite.core.tools.calls import (
    EnvAction,
    RuntimeEnvAction,
    runtime_rejected_reason,
)
from lite.core.tools.extra_tools import BASH_TOOL_NAME, LiteFinishToolSet
from lite.core.tools.results import LiteToolResult, make_tool_result
from lite.core.tools.schemas import tool_schema_parameters
from lite.gym.errors import (
    CapacityExhausted,
    PairableModelActionError,
    TrueInfraFailure,
)
from lite.gym.remote.admission import docker_create_slot_async
from lite.gym.sandbox.exec_stdio import DockerProvisioner, attach
from lite.gym.sandbox.types import SandboxTaskConfig
from lite.gym.services import EnvServerPoolable, EnvServerResource
from lite.gym.types import (
    EXECUTED_ACTIONS_INFO_KEY,
    LiteEnvObservation,
    LiteEnvStepResult,
    LiteExecutedAction,
)
from lite.gym.utils.backend.coordinate import norm_to_pixel
from lite.gym.utils.backend.docker import docker_rm_f, docker_rm_f_async
from lite.gym.utils.backend.model_inputs import (
    coerce_model_duration,
    project_model_keys,
)
from lite.gym.utils.backend.reaper import reap
from lite.gym.utils.config.naming import format_container_name
from lite.gym.utils.feedback.errors import (
    MODEL_ACTION_ERROR_TYPES,
    ToolErrorFeedback,
    append_feedback,
    current_feedback,
    record_batch_abort,
    record_model_action_error,
    record_tool_execution_error,
)
from lite.gym.utils.feedback.ingress import (
    invalid_action_message,
    prepare_env_tool_calls,
    standalone_tool_call_feedback_with_reason,
    unsupported_env_action_message,
)
from lite.gym.utils.feedback.results import (
    build_tool_results_from_decisions,
    ordered_tool_call_ids,
)
from lite.gym.utils.feedback.surface import (
    merge_extra_tool_schemas,
    resolve_extra_tools,
    resolve_valid_actions,
)

logger = logging.getLogger(__name__)

# Track auto-named containers so atexit can reap them on hard exit
# (SIGKILL/OOM bypass `close()` — `--rm` doesn't help because the
# container's main process is `sleep 86400`, not the rollout). Cleared
# entries are removed by close().
_LIVE_CONTAINERS: set[str] = set()
_LIVE_LOCK = threading.Lock()

# Actions that are provably READ-ONLY: they inspect the desktop but cannot
# mutate it, so the post-action settle delay is pure latency for them. Kept
# deliberately minimal — ``screenshot`` (a dispatch no-op that still
# earns its own frame) and ``cursor_position`` (a get_cursor_position read). NOT
# included: ``mouse_move`` (hover can change the UI) and every state-changing
# action, which must still settle before the next action / screenshot.
_READ_ONLY_ACTIONS = frozenset({"screenshot", "cursor_position"})

# The GUI vocabulary this exec-stdio desktop backend can actually execute —
# DERIVED from the catalog (never hand-listed), exactly matching the branches of
# ``_dispatch_desktop_action``. ``step`` gates on it BEFORE dispatch so a
# foreign-platform action (``tap``/``swipe``) or a grounding tool
# (``point``/``bbox``) — which ingress lets through, since this env family does
# not run the top-level GUI check — becomes model-visible ``unsupported action:
# <name>`` feedback instead of reaching the ladder. With that gate in front,
# ANY name still reaching the ladder's fallthrough is a dispatcher bug, so the
# ladder raises (see the derive-then-raise contract in
# ``lite/gym/utils/feedback/surface.py:resolve_valid_actions``).
_SUPPORTED_ACTIONS = LiteDesktopActionSet.get_action_names()


#: Mouse-button vocabulary, derived from the tool schema. ``make_lite_action_batch_schema``
#: merges children with ``setdefault``, so the enum comes from the first child that
#: declares one (``click``); a button added only to ``drag``/``mouse_down`` would not
#: show up here.
_MOUSE_BUTTONS: frozenset[str] = frozenset(
    tool_schema_parameters(LiteDesktopActionSet.get_tool_schemas()[0])["properties"]["actions"]
    ["items"]["properties"]["button"]["enum"]
)


def _required_model_arg(args: dict[str, Any], field: str, action_name: str) -> Any:
    """Read an argument the canonical action declares with NO default.

    ``LiteDesktopActionSet`` gives ``type.text``, ``scroll.direction``,
    ``scroll.amount``, ``wait.duration`` and ``hold_key.duration`` no default,
    and env ingress checks envelope shape only — ``prepare_env_tool_calls``
    passes ``{"action": "type"}`` through with ``arguments == {}``. So this
    dispatcher is where an absent required argument is caught, exactly as
    :func:`project_model_keys` catches an absent ``keys``. Substituting a value
    here would type nothing / scroll in an invented direction and still hand the
    model a normal post-action screenshot, which it cannot tell from success.
    ``ValueError`` is in ``MODEL_ACTION_ERROR_TYPES``, so ``step`` turns this
    into model-visible feedback keyed to the call id.
    """
    if field not in args:
        raise ValueError(f"{action_name}.{field} is required")
    return args[field]


def _model_button(args: dict[str, Any], action_name: str) -> str:
    button = args.get("button", "left")
    if button not in _MOUSE_BUTTONS:
        raise PairableModelActionError(
            f"{action_name}: unsupported button {button!r}; "
            f"expected one of {sorted(_MOUSE_BUTTONS)}"
        )
    return button


def _is_transient_boot_error(exc: Exception) -> bool:
    """``docker run``/exec transients worth one fresh boot attempt: a lingering
    ``--name`` from a just-reaped container, dockerd contention timeouts, or
    (when noVNC publishing is opted in) a host-port bind race — both the
    rootful ("port is already allocated") and RootlessKit ("address already in
    use") spellings, plus exec-stdio's own connection-level transients."""
    msg = str(exc).lower()
    return any(s in msg for s in (
        "is already in use by container",
        "port is already allocated", "address already in use",
        "timed out", "timeout", "connection reset",
        "exec-stdio ping transient",
    ))


#: Budget for removing a sandbox container. DERIVED from the two numbers that
#: already bracket it, not picked:
#:
#: * a FLOOR from the daemon: ``docker rm -f`` waits for the container's exit
#:   event and gives up at ~12 s (measured), after which it reports a refusal that
#:   the shared helper re-issues. A budget below that guarantees the client
#:   abandons the CLI before the daemon can say what happened — which is exactly
#:   what the **10** this replaces (at all three sites here) did. Measured worst
#:   case for refusal + successful re-issue: 20.5 s.
#: * a CEILING from the caller: the rollout engine allows the whole
#:   ``env.close()`` 60 s (``asyncio.wait_for(env.close(), timeout=60.0)``, twice
#:   in ``lite/train/rollout/core/engine.py``), and ``close()`` spends up to 30 s
#:   stopping the exec-stdio session first. Removal is the LAST thing ``close()``
#:   does, so it is the step an outer cancel lands on — and a ``CancelledError``
#:   is not an ``Exception``, so it escapes ``docker_rm_f_async``'s reporting
#:   entirely. The two sequential budgets must therefore SUM to the ceiling:
#:   30 + 30 = 60.
#:
#: Every other removal path in the tree declares 60 (``lite/gym/container.py``,
#: ``lite/gym/remote/reaper.py``, browsergym's WA sweep, the four env
#: ``rm_timeout_s`` configs); the sandbox was the only outlier, and the only one
#: with a caller-imposed ceiling that forbids 60.
_RM_TIMEOUT_S = 30.0


def _rm_f(name: str) -> None:
    docker_rm_f(name, timeout=_RM_TIMEOUT_S, label="sandbox")


@atexit.register
def _reap_live_containers() -> None:
    reap(_LIVE_CONTAINERS, _LIVE_LOCK, _rm_f, clear=True)


class SandboxBaseEnv(EnvServerPoolable, EnvServerResource):
    """Desktop environment. step() executes LiteDesktopActionSpace actions."""

    #: Boot-readiness probe: a file the image touches when its desktop stack is
    #: up, polled over ``docker exec`` (NOT a server handshake). Both sandbox-family
    #: images (cua-lite/lite.osworld, cua-lite/sandbox.linux) write /tmp/gnome-ready via the
    #: same one-shot watchdog. ``None`` = exec answering is ready enough.
    READY_MARKER: str | None = "/tmp/gnome-ready"

    #: In-container uid the persistent ``docker exec`` backend runs as. Default
    #: ``user`` — the harness runs as the env's desktop user, so setup/getter/reward
    #: files are user-owned by construction (no ownership handover) and ``/opt/env``
    #: (user-readable) is reached directly. An env whose upstream hooks self-drop via
    #: ``su - <user> -c`` overrides this to ``root`` (passwordless ``su`` works only
    #: from root) — cuaworld does this (``_CUAWorldEnv.EXEC_USER = "root"``).
    EXEC_USER: ClassVar[str] = "user"
    AGENT_USER: ClassVar[str] = "user"

    #: Env-owned ``env_kwargs.extra_tools`` — a ``BaseTools`` subclass, declared
    #: exactly the way every other tool set in the tree is. Subclasses set
    #: ``EXTRA_TOOLS`` (inherited from :class:`~lite.gym.base.LiteBaseEnv`) to
    #: their own set instead of re-implementing the resolution in ``bind``; the
    #: two overrides below add what is true of the whole FAMILY rather than of
    #: any one env in it — the canonical finish tools and ``bash``.

    @classmethod
    def extra_tool_schemas(
        cls,
        selection: list[str] | None = None,
        **ctor_args: Any,
    ) -> list[dict[str, Any]]:
        """The class-declared set, plus the family's ``bash``.

        ``bash`` is granted through ``executable`` — the orthogonal half of the
        contract — rather than declared in any env's tool set, because it is a
        property of the sandbox TRANSPORT: the whole family is a plain desktop
        container reached over ``docker exec``, so the agent shell always lands
        on the machine the agent is looking at. An env whose desktop lives in a
        nested QEMU VM inherits neither this override nor the grant.
        """
        return resolve_extra_tools(
            selection,
            tools=cls.EXTRA_TOOLS,
            env_name=cls.__name__,
            executable=cls.EXTRA_TOOLS.get_tool_names() | {BASH_TOOL_NAME},
        )

    @classmethod
    def known_standalone_tool_names(cls) -> frozenset[str]:
        """Declared extras + finish + the family's ``bash``.

        Resolved per CLASS, not per ``step()``: ``bash`` is the only name this
        family adds to the shared union, and it never varies within a run.
        """
        return super().known_standalone_tool_names() | {BASH_TOOL_NAME}

    #: Per-run resolution of ``env_kwargs.extra_tools`` (stamped by :meth:`bind`).
    #: Class-level default so a metadata read on a hand-built instance (e.g.
    #: cuagym's ``evaluate_terminal_step`` shim, which ``object.__new__``s this
    #: class and never binds) stays safe. NEVER mutated in place.
    _extra_tool_schemas: ClassVar[list[dict[str, Any]]] = []

    #: Per-run resolution of ``env_kwargs.valid_actions`` (stamped by
    #: :meth:`bind`). Same class-level-default rationale as
    #: ``_extra_tool_schemas`` above — and ``None`` is exactly what "omitted"
    #: means in the shared contract (no filtering, full native GUI surface),
    #: so a hand-built instance reads the same answer a real bind would give.
    _valid_actions: ClassVar[list[str] | None] = None

    def __init__(
        self,
        task: SandboxTaskConfig | None = None,
        *,
        # User-facing constructor state.
        display_resolution: tuple[int, int] = (1920, 1080),
        image: str | None = None,
        vnc_port: int | None = None,
        # Framework-injected constructor state (set by the registry's entry_point
        # factory; user does NOT pass these directly).
        env_id: str | None = None,
        setup_fn: Callable | None = None,
        evaluate_step_fn: Callable | None = None,
        evaluate_final_fn: Callable | None = None,
        container: str | None = None,
        computer_config: dict[str, Any] | None = None,
        post_action_delay: float = 1.0,
        debug: bool = False,
        # Soft state.
        max_steps: int | None = None,
        seed: int | None = None,
        valid_actions: list[str] | None = None,
        extra_tools: list[str] | None = None,
    ) -> None:
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
            seed=seed,
            valid_actions=valid_actions,
            extra_tools=extra_tools,
        )

    def _set_constructor_state(
        self,
        *,
        display_resolution: tuple[int, int] = (1920, 1080),
        image: str | None = None,
        vnc_port: int | None = None,
        env_id: str | None = None,
        setup_fn: Callable | None = None,
        evaluate_step_fn: Callable | None = None,
        evaluate_final_fn: Callable | None = None,
        container: str | None = None,
        computer_config: dict[str, Any] | None = None,
        post_action_delay: float = 1.0,
        debug: bool = False,
    ) -> None:
        """Set fields that are immutable after construction.

        MUST NOT touch any per-task field — those are owned by :meth:`bind`.
        """
        # Framework-injected constructor state
        self._env_id = env_id
        self._setup_fn = setup_fn
        self._evaluate_step_fn = evaluate_step_fn
        self._evaluate_final_fn = evaluate_final_fn
        # User-facing constructor state
        self._display_resolution = tuple(display_resolution)
        self._post_action_delay = post_action_delay
        self._debug = debug
        self._image = image
        self._vnc_port = vnc_port

        # ``container=`` injects a PRE-CREATED container by name: boot()
        # attaches the exec-stdio session (no docker run), close() never
        # removes it — the injector owns the container's lifecycle. A
        # container name is all docker exec needs to drive it.
        self._computer: Any | None = None  # a _ContainerHandle after boot()
        self._container_override = container
        self._owns_computer = container is None
        self._container_name: str | None = None  # set during boot()

        # ``_computer_config`` is BAKED (the container's shape — image,
        # display, resources — is fixed at docker run time and cannot
        # change for the life of the container). Subclasses (e.g.
        # lite.osworld) default this so boot can stay task-independent; if not
        # pre-stamped, the first :meth:`bind` call's ``task.computer`` populates
        # it.
        if computer_config is not None:
            display = f"{self._display_resolution[0]}x{self._display_resolution[1]}"
            self._computer_config: dict[str, Any] | None = {
                **computer_config, "display": display,
            }
            if image:
                self._computer_config["image"] = image
        else:
            self._computer_config = None

    @staticmethod
    def _task_metadata(task: SandboxTaskConfig) -> LiteCUAMetadata:
        """Same-source metadata builder for the sandbox family. Copy, don't
        alias — callers must not get a mutable reference into the task config."""
        return LiteCUAMetadata(
            dims=(task.platform, "use"),
            # [] not None: extra tools are opt-in, and subclasses that resolve
            # env_kwargs.extra_tools (e.g. lite.osworld) serve [] with no
            # override — the registered copy must say the same.
            extra_tool_schemas=task.extra_tool_schemas or [],
            others={**task.metadata.get("others", {})},
        )

    def _runtime_metadata(self) -> LiteCUAMetadata:
        # Read (via the base ``metadata`` property) by ``LiteEnvClient`` after
        # construction. Normal task-bearing direct/server construction sets
        # ``self._task`` via ``__init__`` so this works immediately. The guard
        # below only trips on explicit no-task construction / hand-built test
        # instances.
        if self._task is None:
            raise RuntimeError(
                f"{type(self).__name__}.metadata read before bind(); "
                "no-task compatibility instances have no task identity"
            )
        # The env_kwargs merge MUST live here, not in ``_task_metadata``: that
        # builder is a @staticmethod over the task alone and cannot see the
        # per-run ``extra_tools`` opt-in. CONCATENATE (never replace) so the
        # task's own slice and the base resolver's slice (``bash`` + finish
        # tools, [] unless this run opted in) both survive.
        md = self._task_metadata(self._task)
        return dataclasses.replace(
            md,
            valid_actions=self._valid_actions,
            extra_tool_schemas=merge_extra_tool_schemas(
                md.extra_tool_schemas, self._extra_tool_schemas,
            ),
        )

    def bind(
        self, task: SandboxTaskConfig | None = None,
        *, max_steps: int | None = None, seed: int | None = None,
        valid_actions: list[str] | None = None,
        extra_tools: list[str] | None = None,
    ) -> None:
        """Bind (or re-bind) a task and apply soft kwargs. Cheap —
        in-memory only; does NOT touch the booted container (its
        display resolution + image are baked at boot, immutable here).

        **Single entry point for soft state.** Direct mode and server cold
        construction both call the concrete env's explicit ``__init__``; this
        class sets constructor state, then calls ``bind`` exactly once.
        Subclasses with additional soft kwargs MUST
        extend the signature with their own named params and call
        ``super().bind(task, max_steps=max_steps)`` once — see
        ``LiteOsworldEnv.bind`` for the pattern.

        ``seed`` is a UNIVERSAL harness-injected soft kwarg (``group_shared_seed`` /
        ``--env-kwargs seed=...``), so the BASE signature accepts it — deterministic sandbox
        tasks ignore it; subclasses that use it (``LiteOsworldEnv`` → ``_noise_seed``) override
        and consume it. An env using this base ``bind`` directly (e.g. ``lite.demo``) must stay
        callable with the harness's standard soft kwargs.

        **Unconditional assignment**: signature defaults are the SINGLE
        source of truth for what each soft kwarg defaults to. Inside
        the body, every soft field gets assigned exactly once per
        call — never ``if x is not None: self._x = x``. Because direct and server
        paths route here, the body always runs with the caller's exact value
        (None included), so the two modes produce byte-equal state.

        ``task=None`` is the no-task default used when construction seeds
        soft defaults before a task is bound. In that case the body still runs fully —
        ``self._task`` ends up ``None`` and ``self._max_steps`` takes
        the explicit ``max_steps`` kwarg. ``setup_fn`` work that
        needs a task happens later at :meth:`reset`.
        """
        # Sandbox-specific: stamp display resolution onto the task's
        # computer config so setup_fn(task, computer) sees the right
        # ``display`` field.
        resolved_task = task
        if task is not None and task.computer:
            resolved_task = task.with_display(
                f"{self._display_resolution[0]}x{self._display_resolution[1]}"
            )
            # Cold-fallback: subclass didn't pre-stamp computer_config
            # in constructor state setup (e.g. ``register_tasks(...)`` path
            # builds a bare SandboxBaseEnv whose task carries its own
            # ``computer`` config). Seed once on first bind so
            # ``boot()`` has a computer config. Legacy direct construction
            # pre-stamps via the subclass default in
            # constructor (see ``LiteOsworldEnv``'s
            # ``computer_config=_COMPUTER_CONFIG`` default), so the
            # second-bind / re-bind path skips this branch — the
            # booted container's ``display`` / ``image`` are immutable
            # at ``docker run`` time and we MUST NOT rewrite them.
            if self._computer_config is None:
                self._computer_config = resolved_task.computer
        self._task = resolved_task
        # Unconditional: stash the caller's pinned budget; reset() /
        # subclasses can lazily fall back to ``task.max_steps`` when
        # this is None. Single field, single source of truth.
        self._max_steps = max_steps
        # ``seed`` is a UNIVERSAL harness-injected soft kwarg (group_shared_seed /
        # --env-kwargs seed=...). Deterministic sandbox tasks ignore it, but the BASE bind
        # must accept it so every Sandbox env stays callable with the harness's standard soft
        # kwargs — lite.demo uses this base bind directly. Subclasses that DO consume it (e.g.
        # ``LiteOsworldEnv`` → ``_noise_seed``) override bind. Stashed for completeness.
        self._seed = seed
        # ``valid_actions`` is a SOFT env_kwarg like ``extra_tools`` — it constrains
        # the advertised GUI action enum only, changes no container shape, and
        # is re-applied on every bind. Runtime invalid/unsupported feedback is
        # handled by this env's ``step`` path. Resolution goes through the same
        # helper every other env uses: ``None`` = full GUI surface, ``[]`` =
        # deliberately none, and a typo raises here.
        self._valid_actions = resolve_valid_actions(
            valid_actions,
            env_name=self._env_id or type(self).__name__,
            # Every sandbox task is desktop/browser (both map to the desktop
            # action vocabulary); a legacy no-task bind has no task to ask.
            platform=resolved_task.platform if resolved_task is not None else "desktop",
        )
        # ``extra_tools`` is a SOFT env_kwarg — it is applied at bind time here
        # and the agent shell is opened lazily post-boot, so it changes no
        # container shape: a bash and a non-bash run boot the same backend.
        self._extra_tool_schemas = type(self).extra_tool_schemas(extra_tools)

    @property
    def external_resource_id(self) -> str | None:
        # Full docker container name, set during boot() (booted or
        # injected). ``None`` between gym.make and the first successful
        # boot/attach.
        return self._container_name

    async def boot(self) -> None:
        """Acquire the container + exec-stdio handle. Idempotent.
        Task-independent: reads :attr:`_computer_config` (set in
        ``__init__`` or by ``bind()``), NOT ``self._task``.

        Called implicitly by ``reset()`` on normal direct/server paths.
        Transient errors (dockerd contention, lingering
        ``--name``, port-bind races when noVNC is opted in) trigger a
        bounded retry; a genuinely full host fails fast.
        """
        if self._computer is not None:
            return  # idempotent

        # Injected pre-created container: attach the exec-stdio session
        # only — no docker run, no ownership transfer. boot() is still
        # called so the lifecycle is symmetric with the booted path.
        if self._container_override:
            self._container_name = self._container_override
            self._computer = await attach(
                self._container_name,
                exec_user=type(self).EXEC_USER,
                agent_user=type(self).AGENT_USER,
            )
            return

        if self._computer_config is None:
            raise RuntimeError(
                f"{type(self).__name__}.boot() called before bind() or "
                "computer_config; nothing to acquire"
            )

        _MAX_BOOT_RETRIES = 3
        for attempt in range(_MAX_BOOT_RETRIES):
            try:
                await self._attempt_boot_computer()
                return  # success
            except RuntimeError as e:
                is_last = attempt == _MAX_BOOT_RETRIES - 1
                if is_last or not _is_transient_boot_error(e):
                    raise
                logger.warning(
                    "Boot transient on attempt %d, retrying: %s",
                    attempt + 1, e,
                )
                await self._release_failed_attempt()
                # Exponential-ish backoff: gives the daemon room when the
                # transient is contention-shaped (1s, 2s, 3s).
                await asyncio.sleep(1 + attempt)

    async def reset(self) -> LiteEnvObservation:
        """Ensure booted + task bound, run setup, take initial screenshot.

        Lazily calls :meth:`boot` and asserts a task is bound via
        :meth:`bind`.
        """
        await self.boot()
        if self._task is None:
            raise RuntimeError(
                f"{type(self).__name__}.reset() called before bind(); "
                "a task must be bound first"
            )

        self._terminated = False
        self._step_count = 0
        # Lazy max_steps fallback: bind stashes the caller-pinned
        # budget (or None); reset fills in from the task's default
        # iff still unset. Single source of truth — bind never
        # touches ``task.max_steps`` itself, so direct and server paths
        # resolve identically here.
        if self._max_steps is None:
            self._max_steps = self._task.max_steps

        # Post-boot steps (setup_fn / screenshot) can fail under burst
        # with transient ``RuntimeError`` / ``TimeoutError`` from the
        # exec-stdio session (e.g. a still-warming desktop returning a
        # blank screenshot, or a config step racing the Flask server's
        # bind). Boot itself succeeded so ``_attempt_boot_computer``'s
        # cleanup didn't fire — the broken half-configured env stays
        # assigned and the next reset() crashes again on the same
        # broken state. Tear down here too so the client's 503 retry
        # rebuilds from scratch.
        try:
            if self._setup_fn:
                await _call_fn(self._setup_fn, self._task, self._computer)

            screenshot = await _take_screenshot(self._computer)
        except (RuntimeError, TimeoutError) as e:
            await self._release_failed_attempt()
            # Only the exception TYPE in the public ``what`` — the
            # original message text often includes phrases that
            # diagnostic tools and tests grep for. Wrapping it here
            # as a string would falsely trip those substring checks;
            # the real diagnostic is preserved via ``__cause__``
            # (``raise ... from e``) for operator log inspection.
            raise CapacityExhausted(
                f"post-boot setup transient ({type(e).__name__})",
                retry_after_s=30,
                layer="env_internal",
            ) from e
        return LiteEnvObservation(image=screenshot, text=self._task.instruction)

    async def _attempt_boot_computer(self) -> None:
        """One boot attempt: ``docker run`` via :class:`DockerProvisioner`,
        then attach an exec-stdio session. Raises on any failure; the
        caller (:meth:`boot`) decides whether to retry based on
        :func:`_is_transient_boot_error`.

        Reads :attr:`_computer_config` (NOT ``self._task.computer``), keeping
        boot task-independent. Task identity (``self._task.task_id``) is used
        purely for the container name label — falls back to ``None`` for
        explicit no-task construction / hand-built test instances.

        ``docker run`` is wrapped in :func:`docker_create_slot_async` (the
        admission semaphore) so concurrent creates don't thrash the daemon —
        under an env-server only; direct mode (no ``env_id``) takes no slot.
        Slot acquisition timeout surfaces as :exc:`CapacityExhausted` →
        client 503 + Retry-After.

        Any exception (transient ``RuntimeError``, ``CapacityExhausted``
        from the admission semaphore, :exc:`asyncio.CancelledError` from
        request teardown — anything from ``BaseException`` on down)
        triggers :meth:`_release_failed_attempt` so:

        * the half-created docker container, if it got that far, is
          force-removed
        * any opened exec-stdio session is closed
        * ``_LIVE_CONTAINERS`` membership is dropped
        * ``self._container_name`` is nulled so the drift reaper no
          longer sees a stale ``external_resource_id``
        """
        import uuid

        from lite.gym.utils.config.identity import EnvIdentity

        assert self._computer_config is not None, (
            "boot() invariant violated: _computer_config is None"
        )

        try:
            # ``identity`` is set by ``registry.make`` when running under
            # an env-server; absent in direct mode (falls back to
            # ``$SESSION_ID`` then ``"local"``).
            identity = getattr(self, "identity", None) or EnvIdentity()
            cfg = dict(self._computer_config)
            if self._image:
                # Family-neutral boot-image override (env-kwarg): boot a
                # DERIVED image without editing the env's configured default.
                cfg["image"] = self._image

            # task_id is None on explicit no-task construction / hand-built
            # test instances;
            # format_container_name omits the segment cleanly.
            task_id_segment = self._task.task_id if self._task is not None else None
            name = format_container_name(
                env_id=self._env_id,
                task_id=task_id_segment,
                suffix=uuid.uuid4().hex[:6],
                session_id=identity.resolved_session_id(),
                token_hash=identity.token_hash,
                server_port=identity.server_port,
            )
            # Stash before the docker-creating call so
            # ``_release_failed_attempt`` can clean up regardless of
            # which step raises.
            self._container_name = name
            with _LIVE_LOCK:
                _LIVE_CONTAINERS.add(name)
            logger.info("Auto-assigned container name: %s", name)
            image_tag = cfg.get("image")
            if image_tag and self._env_id:
                from lite.gym.errors import EnvDepsMissingError
                from lite.gym.utils.backend.freshness import (
                    UnknownImageFreshnessProvider,
                    image_for,
                )

                try:
                    def _ensure_image_runnable() -> None:
                        image_for(self._env_id, tag=str(image_tag)).ensure_runnable()

                    await asyncio.to_thread(_ensure_image_runnable)
                except UnknownImageFreshnessProvider:
                    # Generic sandbox tasks may point at non-CUA-Lite images.
                    pass
                except (KeyError, OSError, TypeError, ValueError) as exc:
                    raise EnvDepsMissingError(
                        what=(
                            f"{self._env_id} image freshness inputs are unavailable "
                            f"or invalid ({type(exc).__name__})"
                        ),
                        install="run the environment install.sh for this env",
                        see="docs/envs.md#image-build-and-freshness",
                    ) from exc

            prov = DockerProvisioner(
                name, cfg, vnc_port=self._vnc_port,
                ready_marker=self.READY_MARKER,
                # the task config's ``timeout`` IS the boot-readiness ceiling
                # (osworld sets 600s for heavy concurrent boots); don't let the
                # provisioner's 120s default override it.
                ready_timeout=float(cfg.get("timeout") or 120.0))

            try:
                # Admission semaphore: bound concurrent ``docker run`` so the
                # daemon doesn't thrash. Bounded wait; slot timeout raises
                # CapacityExhausted → 503 + Retry-After.
                #
                # SKIPPED IN DIRECT MODE (no env_id), and that is a standing
                # decision, not an oversight. Direct mode has no 503-retry client,
                # so a slot timeout here becomes a hard task failure instead of
                # back-pressure; a direct-mode caller bounds its own fan-out
                # (the validators' ``--concurrency``, rollout's worker count) and
                # that is where the bound belongs.
                if self._env_id:
                    async with docker_create_slot_async():
                        await prov.run()
                else:
                    await prov.run()
                self._computer = await attach(
                    name,
                    exec_user=type(self).EXEC_USER,
                    agent_user=type(self).AGENT_USER,
                )
                logger.info("Booted %s (exec-stdio attached)", name)
            except TimeoutError as e:
                # The DockerProvisioner readiness wait can time out
                # under burst even with admission throttling — the
                # /tmp/gnome-ready marker lags when the host's NVMe is
                # saturated. Treat it as transient admission pressure
                # rather than a hard 500: the client's 503 retry
                # budget usually finds a thinner-queue window.
                # ``_release_failed_attempt`` (via the outer
                # ``except BaseException``) tears down the half-created
                # container so the retry rebuilds cleanly.
                raise CapacityExhausted(
                    f"container boot transient ({type(e).__name__})",
                    retry_after_s=30,
                    layer="env_internal",
                ) from e
        # KEEP the bare ``BaseException``: it is a cleanup-and-RE-RAISE, not a
        # swallow. ``BaseException`` (not ``Exception``) is required because
        # ``asyncio.CancelledError`` — the common case when the client
        # disconnects mid-boot — inherits from ``BaseException``; narrowing to
        # ``Exception`` would leak the half-created container on every cancel.
        # Nothing is suppressed: the original exception propagates unchanged.
        except BaseException:
            await self._release_failed_attempt()
            raise

    async def _release_failed_attempt(self) -> None:
        """Tear down the half-created container before the next retry.
        Best-effort: docker rm timeouts / errors are swallowed because
        the retry will provision a fresh name anyway.

        Also nulls ``self._container_name`` so the
        :func:`external_resource_id` property does not report a
        container name with no docker container behind it — without
        this, the drift reaper sees the stale name on its next sweep,
        flags it as a ghost, and triggers a false-positive
        ``state.envs.pop`` that races whatever subsequent boot attempt
        actually succeeded.
        """
        # Close the exec-stdio session FIRST (if it opened) so the
        # ``docker exec`` pipe doesn't hold a reference into a
        # container we're about to ``docker rm -f``.
        if self._computer is not None:
            try:
                await asyncio.wait_for(self._computer.stop(), timeout=10.0)
            except Exception:
                pass
            self._computer = None
        if self._container_name:
            await docker_rm_f_async(self._container_name, timeout=_RM_TIMEOUT_S, label="sandbox")
            with _LIVE_LOCK:
                _LIVE_CONTAINERS.discard(self._container_name)
        self._container_name = None

    async def step(self, actions: list[RuntimeEnvAction]) -> LiteEnvStepResult:
        """Execute actions and take screenshot."""
        input_actions = actions
        metadata = self.metadata
        result_call_ids = ordered_tool_call_ids(input_actions)
        actions_with_result_ids, ingress_errors = prepare_env_tool_calls(
            actions,
            metadata,
            defer_extra_schema_validation_for={BASH_TOOL_NAME},
        )
        screen_w, screen_h = await _get_screen_size(
            self._computer, fallback=self._display_resolution,
        )
        terminated = False
        # Model-emitted terminal calls get no continuation observation.
        # Internal loop-detect finishes carry only ``_result_call_id`` through
        # ingress, so they still receive their paired result.
        terminal_call_ids: set[str] = set()
        executed_actions: list[LiteExecutedAction] = []
        accepted_actions: list[EnvAction] = []
        step_screenshots: list[bytes] = []
        action_errors: dict[str, ToolErrorFeedback] = dict(ingress_errors)
        bash_results: dict[str, LiteToolResult] = {}
        known_standalone_tools = type(self).known_standalone_tool_names()
        stop_reason: str | None = None

        for index, (action, result_call_id) in enumerate(actions_with_result_ids):
            name = action["name"]
            args = action["arguments"]

            # A fault env ingress already attributed to the MODEL: a name this
            # batch tool does not carry, a standalone extra nested inside it, or
            # a name the task config withholds. It is never dispatched, so it
            # changed nothing -- but it still occupies one slot of the batch the
            # model emitted, so it still owes one frame and one model-visible
            # reason. That frame repeats the previous screen by construction:
            # the honest record of an action that did not land, and what keeps
            # the frame count equal to the action count.
            if rejected_reason := runtime_rejected_reason(action):
                if result_call_id:
                    # APPEND, never assign: a batch no longer stops at the first
                    # rejected action, so several children can fail onto the ONE
                    # call id they share. Assigning would show the model only the
                    # last one, and it would re-emit the others verbatim forever.
                    append_feedback(
                        action_errors,
                        result_call_id,
                        current_feedback(rejected_reason),
                    )
                executed_actions.append({
                    "call": "noop",
                    "args": {"name": name, "reason": rejected_reason},
                })
                step_screenshots.append(await _take_screenshot(self._computer))
                continue

            tool_feedback, noop_reason = standalone_tool_call_feedback_with_reason(
                action, known_standalone_tools, metadata.extra_tool_schemas,
            )
            if tool_feedback is not None:
                if result_call_id:
                    action_errors[result_call_id] = tool_feedback
                executed_actions.append({
                    "call": "noop",
                    "args": {
                        "name": name,
                        "reason": noop_reason,
                    },
                })
                continue

            if name == "bash":
                unexpected = sorted(set(args) - {"command"})
                if unexpected:
                    reason = f"bash.arguments got unexpected keys: {unexpected}"
                    record_model_action_error(
                        action_errors,
                        result_call_id,
                        ValueError(reason),
                        carrier="error_only",
                        action_name=name,
                    )
                    executed_actions.append({
                        "call": "noop",
                        "args": {
                            "name": name,
                            "reason": reason,
                        },
                    })
                    continue
                command = args.get("command")
                if not isinstance(command, str):
                    record_model_action_error(
                        action_errors,
                        result_call_id,
                        ValueError("bash.arguments.command must be a string"),
                        carrier="error_only",
                        action_name=name,
                    )
                    executed_actions.append({
                        "call": "noop",
                        "args": {
                            "name": name,
                            "reason": "bash.arguments.command must be a string",
                        },
                    })
                    continue
                try:
                    shell_result = await self._computer.agent_shell.run(command)
                    text = shell_result.output
                    if shell_result.returncode:
                        text = f"{text}\n[exit {shell_result.returncode}]"
                    if result_call_id:
                        bash_results[result_call_id] = make_tool_result(
                            tool_call_id=result_call_id,
                            text=text,
                            metadata={"returncode": shell_result.returncode},
                        )
                    accepted_actions.append(action)
                    executed_actions.append({
                        "call": "agent_shell.run",
                        "args": {
                            "command": command,
                            "returncode": shell_result.returncode,
                        },
                    })
                except Exception as e:
                    record_tool_execution_error(
                        action_errors,
                        result_call_id,
                        e,
                        carrier="error_only",
                        action_name=name,
                    )
                    executed_actions.append({
                        "call": "noop",
                        "args": {"name": name, "reason": str(e)},
                    })
                continue

            if name in LiteFinishToolSet.get_tool_names():
                accepted_actions.append(action)
                terminated = True
                stop_reason = name
                if action.get("call_id"):
                    terminal_call_ids.add(action["call_id"])
                logger.info("Agent terminated (%s): %s", name, args)
                break

            invalid_action = invalid_action_message(action, metadata.valid_actions)
            if invalid_action:
                if result_call_id:
                    # R2(a): a GUI action the model got wrong still earns the
                    # current observation, and APPEND because several children of
                    # one batch share this call id.
                    append_feedback(
                        action_errors, result_call_id, current_feedback(invalid_action),
                    )
                executed_actions.append({
                    "call": "noop",
                    "args": {"name": name, "reason": invalid_action},
                })
                step_screenshots.append(await _take_screenshot(self._computer))
                continue

            unsupported_action = unsupported_env_action_message(
                name, _SUPPORTED_ACTIONS,
            )
            if unsupported_action:
                if result_call_id:
                    # R2(a): a GUI action the model got wrong still earns the
                    # current observation, and APPEND because several children of
                    # one batch share this call id.
                    append_feedback(
                        action_errors, result_call_id, current_feedback(unsupported_action),
                    )
                executed_actions.append({
                    "call": "noop",
                    "args": {"name": name, "reason": unsupported_action},
                })
                step_screenshots.append(await _take_screenshot(self._computer))
                continue

            try:
                calls = await _dispatch_desktop_action(
                    self._computer, name, args, screen_w, screen_h
                )
            except MODEL_ACTION_ERROR_TYPES as e:
                record_model_action_error(
                    action_errors,
                    result_call_id,
                    e,
                    action_name=name,
                )
                executed_actions.append({
                    "call": "noop",
                    "args": {"name": name, "reason": str(e)},
                })
                # POLICY: a MODEL fault costs this action and nothing more. Bad
                # arguments or an off-enum value mean the action never reached
                # the screen, so the state the tail was chosen against is exactly
                # the state that is still on screen -- there is nothing to
                # protect the tail from. The model gets the reason as text and a
                # frame for this slot, then the batch continues. Contrast the
                # interface-failure arm below: THERE the action may have half
                # executed and the screen is unknown, which is a different fault
                # with a different answer.
                step_screenshots.append(await _take_screenshot(self._computer))
                continue
            executed_actions.extend(calls)
            # ANY noop among the records, not just a lone one: a partially executed
            # multi-record action (hold_key, drag, multi-key key_down/up, coordinate
            # scroll) leaves its real records BEFORE the downgrade.
            if any(c.get("call") == "noop" for c in calls):
                # The only noop the dispatcher can emit is the interface-boundary
                # downgrade (container died / exec pipe broke); an unhandled action
                # name raises instead. Same sequence policy as the arm above: abort
                # the step. Not appended to ``accepted_actions`` either -- that list
                # is forwarded verbatim to ``evaluate_final_fn``, which must not be
                # told the agent performed an action that failed at the interface.
                # The reason lives on the noop record, not on ``calls[0]``, which for
                # multi-record actions is the first real record. It reaches the LOG,
                # not the model: ``_tool_execution_visible_detail`` replaces a plain
                # ``str`` reason with "execution failed".
                noop = next(c for c in calls if c.get("call") == "noop")
                record_tool_execution_error(
                    action_errors,
                    result_call_id,
                    noop["args"]["reason"],
                    action_name=name,
                )
                record_batch_abort(
                    action_errors,
                    result_call_id,
                    actions_with_result_ids[index + 1:],
                )
                break
            accepted_actions.append(action)

            # Sleep AFTER every state-changing action (not just the last one)
            # so that rollouts which pass multiple actions in a single env.step
            # give the UI time to settle between each action, matching the
            # timing of multi-turn single-step rollouts. Skip the
            # delay for provably-no-op READ-ONLY actions (screenshot /
            # cursor_position): they cannot change the UI, so there is nothing
            # to settle and the delay would be pure latency.
            if self._post_action_delay > 0 and name not in _READ_ONLY_ACTIONS:
                await asyncio.sleep(self._post_action_delay)

            # One frame PER EXECUTED ACTION, captured after that action settled.
            # ``_READ_ONLY_ACTIONS`` only suppresses the settle delay above; it
            # must NOT suppress the capture, so the frame count is a pure
            # function of how many actions ran and never depends on what they
            # were. An aborted batch (``break`` arms above) captured a frame for
            # each action that DID run, which is the honest record.
            step_screenshots.append(await _take_screenshot(self._computer))

        if not step_screenshots:
            # Nothing executed (empty batch, or every call rejected at ingress).
            # The turn still owes the model a current observation, so take the
            # one frame the loop never reached.
            step_screenshots.append(await _take_screenshot(self._computer))

        return await self._finalize_step_result(
            ordered_call_ids=result_call_ids,
            continue_call_ids=[
                call_id for call_id in result_call_ids
                if call_id not in terminal_call_ids
            ],
            images=step_screenshots,
            accepted_actions=accepted_actions,
            executed_actions=executed_actions,
            feedback=action_errors,
            explicit_results=bash_results,
            terminated=terminated,
            stop_reason=stop_reason,
        )

    async def _finalize_step_result(
        self,
        *,
        ordered_call_ids: Sequence[str | None],
        continue_call_ids: Collection[str | None] | None,
        images: Sequence[bytes],
        accepted_actions: Sequence[EnvAction],
        executed_actions: Sequence[LiteExecutedAction],
        feedback: Mapping[str, ToolErrorFeedback] | None = None,
        explicit_results: Mapping[str, LiteToolResult] | None = None,
        terminated: bool = False,
        stop_reason: str | None = None,
    ) -> LiteEnvStepResult:
        """Apply shared step accounting, max-step truncation, eval, and results.

        CONTRACT — ``accepted_actions``: every element is an env-internal bare
        action, ``{"name": str, "arguments": dict}``. Callers must only append actions that
        survived ``lite.gym.utils.feedback.ingress.prepare_env_tool_calls``,
        which is what rejects noncanonical Lite calls and malformed payloads; the
        ``step`` above reads ``action["name"]`` / ``action["arguments"]`` bare
        off exactly these values.

        This matters beyond ``step``: ``accepted_actions`` is forwarded verbatim
        to the task's ``evaluate_final_fn`` below, so it is the shape every
        env's reward function reads. A ``setup_fn``-style helper that wanted to
        synthesize a terminal action should build the canonical call with
        ``make_internal_terminate_action`` and then project it through
        ``prepare_env_tool_calls`` before passing it here.
        """
        self._step_count = getattr(self, "_step_count", 0) + 1
        max_steps = getattr(self, "_max_steps", None)
        truncated = not terminated and (
            max_steps is not None and self._step_count >= max_steps
        )
        if truncated:
            stop_reason = MAX_STEPS_STOP_REASON

        reward = None
        eval_info: dict[str, Any] = {}
        done = terminated or truncated
        eval_fn = (
            getattr(self, "_evaluate_final_fn", None)
            if done
            else getattr(self, "_evaluate_step_fn", None)
        )
        if eval_fn is not None:
            result = await _call_fn(
                eval_fn,
                getattr(self, "_task", None),
                getattr(self, "_computer", None),
                list(accepted_actions),
                getattr(self, "_debug", False),
            )
            if isinstance(result, tuple):
                reward, eval_info = result
                eval_info = eval_info or {}
            else:
                reward = result

        info: dict[str, Any] = {EXECUTED_ACTIONS_INFO_KEY: list(executed_actions)}
        if stop_reason is not None:
            info[STOP_REASON_INFO_KEY] = stop_reason
        if eval_info:
            info["eval"] = eval_info
        return build_tool_results_from_decisions(
            LiteEnvStepResult(
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            ),
            ordered_call_ids=ordered_call_ids,
            continue_call_ids=continue_call_ids,
            images=list(images),
            feedback=feedback,
            explicit_results=explicit_results,
        )

    async def close(self) -> None:
        """Close the exec-stdio session; remove the container only if we own it.

        The session (our ``docker exec`` pipe) is OURS even on an injected
        container, so it is always closed; the CONTAINER belongs to whoever
        created it — injected ones are left running for their owner.
        """
        if self._computer is not None:
            try:
                await asyncio.wait_for(self._computer.stop(), timeout=30.0)
            except TimeoutError:
                logger.warning("exec-stdio session close timed out after 30s")
            except Exception as e:
                logger.warning("exec-stdio session cleanup error: %s", e)
            self._computer = None

        if self._owns_computer and self._container_name:
            await docker_rm_f_async(self._container_name, timeout=_RM_TIMEOUT_S, label="sandbox")
            with _LIVE_LOCK:
                _LIVE_CONTAINERS.discard(self._container_name)
            self._container_name = None

# =============================================================================
# Internal Helpers
# =============================================================================

async def _call_fn(fn: Callable, *args: Any) -> Any:
    """Call a function (sync or async) with as many args as it accepts."""
    sig = inspect.signature(fn)
    # Count positional parameters (exclude **kwargs)
    n_params = len([p for p in sig.parameters.values()
                    if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                  inspect.Parameter.POSITIONAL_OR_KEYWORD)])
    result = fn(*args[:n_params])
    if inspect.isawaitable(result):
        return await result
    return result

async def _get_screen_size(
    computer: Any, *, fallback: tuple[int, int],
) -> tuple[int, int]:
    """Get screen dimensions from the container handle's interface.

    Results are cached on the handle after the first successful call,
    since the display resolution is baked at ``docker run`` time
    (``VNC_RESOLUTION``) and immutable for the container's lifetime.

    ``fallback`` is the env's OWN configured ``display_resolution`` — the
    value the container was booted with — so a transient probe failure keeps
    coordinate scaling correct. (A magic ``(1920, 1080)`` here would silently
    mis-scale EVERY click on a non-1080p display while the rollout looked
    successful.)
    """
    cached = getattr(computer, "_lite_screen_size", None)
    if cached is not None:
        return cached

    try:
        # ExecStdioInterface.get_screen_size always returns {"width", "height"}.
        raw = await computer.interface.get_screen_size()
        size = (raw["width"], raw["height"])
        computer._lite_screen_size = size  # type: ignore[attr-defined]
        return size
    except Exception as e:
        logger.warning("Could not get screen size: %s. Falling back to the "
                       "configured display_resolution %s.", e, fallback)
    return fallback

async def _take_screenshot(computer: Any) -> bytes:
    """Take a screenshot and return raw PNG **bytes**.

    ``ExecStdioInterface.screenshot`` already returns PNG bytes; the old redundant
    ``b64decode``→``b64encode`` pair is gone. Raises on
    failure so the caller sees the real error (e.g. container crash).
    """
    return await computer.interface.screenshot()

async def _dispatch_desktop_action(
    computer: Any,
    name: str,
    args: dict[str, Any],
    screen_w: int,
    screen_h: int,
) -> list[LiteExecutedAction]:
    """
    Dispatch a cua-lite tool call to ``computer.interface`` (the
    ``ExecStdioInterface`` on the exec-stdio container handle).

    Converts [0,1000] normalized coordinates → absolute pixels.
    Covers all LiteDesktopActionSpace actions.

    Returns a list of dicts recording each low-level interface call made,
    e.g. ``[{"call": "left_click", "args": {"x": 960, "y": 540}}]``.

    Raises:
        TrueInfraFailure: ``name`` has no branch in the ladder below. ``step``
          already rejected every model-reachable non-desktop name against the
          DERIVED :data:`_SUPPORTED_ACTIONS`, so this is a dispatcher gap
          (or a caller bypassing the gate) — an internal bug, per the note
          below, and never a recorded noop.
    """
    # The noop-downgrade below must cover ONLY the interface-call boundary
    # (container/exec-pipe failures) — never bugs in this dispatch ladder
    # itself: a bad arg / KeyError here must raise, not be silently recorded
    # as a noop and then screenshotted + eval'd as if the action ran.
    interface = _InterfaceBoundary(computer.interface)
    calls: list[LiteExecutedAction] = []

    def _to_pixel(coord: list[int] | None) -> tuple[int, int]:
        # Canonical round+clamp (parity-verified): a model emitting
        # 1000 lands at w-1 (on-canvas), not w.
        try:
            return norm_to_pixel(coord, screen_w, screen_h, on_malformed="raise")
        except (TypeError, ValueError, IndexError):
            raise ValueError("arguments could not be interpreted") from None

    def _record(call_name: str, **kwargs: Any) -> None:
        calls.append({"call": f"computer.interface.{call_name}", "args": kwargs})

    # Resolve keyboard keys → xdotool keysyms HERE, BEFORE the try/except below
    # (which records any error as a recorded noop). Keys arrive as canonical Lite
    # key tokens: lowercase named keys plus literal printable glyphs (normalized
    # once at LiteDesktopActionSpace.key()); ``backend="xdotool"`` makes
    # project_model_keys pick the xdotool dialect, asserting canonical input and
    # raising LOUD on a contract violation instead of a silent no-op.
    # ``keys`` is REQUIRED with no default (``LiteDesktopActionSet.key(keys: list[str])``),
    # and env ingress does not check argument presence, so a model emitting
    # ``{"action": "key"}`` used to arrive here as ``[]`` and press nothing while the
    # model got a normal post-action screenshot. Not passing ``allow_empty`` is what
    # makes the missing/empty list a ValueError, which ``step``'s
    # ``except MODEL_ACTION_ERROR_TYPES`` turns into model-visible feedback.
    _xkeys = (
        project_model_keys(
            args.get("keys", []),
            action_name=name,
            backend="xdotool",
        )
        if name in LITE_DESKTOP_KEY_ACTION_NAMES
        else []
    )

    try:
        # Mouse Actions
        if name == "click":
            x, y = _to_pixel(args.get("coordinate"))
            button = _model_button(args, name)
            clicks = args.get("clicks", 1)

            if clicks >= 2:
                # One ``--repeat`` call, not double_click + (clicks-2) singles: only
                # ``--repeat`` guarantees the presses land inside the X server's
                # double-click interval.
                await interface.multi_click(x, y, button=button, clicks=clicks)
                _record("multi_click", x=x, y=y, button=button, clicks=clicks)
            elif button == "right":
                await interface.right_click(x, y)
                _record("right_click", x=x, y=y)
            elif button == "middle":
                await interface.mouse_down(x, y, button="middle")
                _record("mouse_down", x=x, y=y, button="middle")
                await interface.mouse_up(x, y, button="middle")
                _record("mouse_up", x=x, y=y, button="middle")
            else:
                await interface.left_click(x, y)
                _record("left_click", x=x, y=y)

        elif name == "mouse_move":
            x, y = _to_pixel(args.get("coordinate"))
            await interface.move_cursor(x, y)
            _record("move_cursor", x=x, y=y)

        elif name == "mouse_down":
            button = _model_button(args, name)
            coord = args.get("coordinate")
            if coord is not None:
                x, y = _to_pixel(coord)
                await interface.mouse_down(x, y, button=button)
                _record("mouse_down", x=x, y=y, button=button)
            else:
                await interface.mouse_down(button=button)
                _record("mouse_down", button=button)

        elif name == "mouse_up":
            button = _model_button(args, name)
            coord = args.get("coordinate")
            if coord is not None:
                x, y = _to_pixel(coord)
                await interface.mouse_up(x, y, button=button)
                _record("mouse_up", x=x, y=y, button=button)
            else:
                await interface.mouse_up(button=button)
                _record("mouse_up", button=button)

        elif name == "drag":
            start = args.get("start_coordinate")
            end = args.get("coordinate")
            button = _model_button(args, name)
            if start is not None:
                sx, sy = _to_pixel(start)
            else:
                # Cursor already at start (e.g. from a preceding mouse_move).
                # ExecStdioInterface.get_cursor_position always returns
                # {"x": int, "y": int}.
                pos = await interface.get_cursor_position()
                sx, sy = pos["x"], pos["y"]
            ex, ey = _to_pixel(end)
            path = [(sx, sy), (ex, ey)]
            await interface.drag(path, button=button)
            _record("drag", path=path, button=button)

        elif name == "scroll":
            direction = _required_model_arg(args, "direction", name)
            amount = _required_model_arg(args, "amount", name)
            coord = args.get("coordinate")

            if coord is not None:
                x, y = _to_pixel(coord)
                await interface.move_cursor(x, y)
                _record("move_cursor", x=x, y=y)

            if direction == "down":
                await interface.scroll_down(clicks=amount)
                _record("scroll_down", clicks=amount)
            elif direction == "up":
                await interface.scroll_up(clicks=amount)
                _record("scroll_up", clicks=amount)
            elif direction in ("left", "right"):
                delta = -amount if direction == "left" else amount
                await interface.scroll(delta, 0)
                _record("scroll", delta_x=delta, delta_y=0)

        elif name == "cursor_position":
            pos = await interface.get_cursor_position()
            _record("get_cursor_position", result=str(pos))

        # Keyboard Actions
        elif name == "type":
            text = _required_model_arg(args, "text", name)
            # A real newline inside ``text`` is an Enter press: ``type_text``
            # splits on it and sends a discrete Return keypress, because
            # ``xdotool type`` would map it to Linefeed/Ctrl-J and GTK widgets
            # drop that silently (lite/gym/sandbox/exec_stdio/server.py).
            await interface.type_text(text)
            _record("type_text", text=text)

        elif name == "key":
            keys = _xkeys  # pre-resolved xdotool keysyms (canonical → backend)
            await interface.hotkey(*keys)
            _record("hotkey", keys=keys)

        elif name == "key_down":
            keys = _xkeys
            for k in keys:
                await interface.key_down(k)
                _record("key_down", key=k)

        elif name == "key_up":
            keys = _xkeys
            for k in reversed(keys):
                await interface.key_up(k)
                _record("key_up", key=k)

        elif name == "hold_key":
            keys = _xkeys
            duration = coerce_model_duration(
                _required_model_arg(args, "duration", name),
                action_name="hold_key",
            )
            # A key that went DOWN must come back UP on every exit path, or the
            # modifier stays physically held on the X server and every later action
            # in the rollout runs with e.g. ctrl down. ``pressed`` (not ``keys``) is
            # released, so a key that never went down is not spuriously released.
            pressed: list[str] = []
            completed = False
            try:
                for k in keys:
                    await interface.key_down(k)
                    pressed.append(k)
                    _record("key_down", key=k)
                await asyncio.sleep(duration)
                _record("sleep", duration=duration)
                completed = True
            finally:
                # Release ALL of them even if one release fails, and surface the
                # first release failure ONLY when the body completed normally: on
                # the cancellation path (``StepTimeoutWrapper`` cancels the sleep)
                # raising here would replace the ``CancelledError`` and lose the
                # cancellation. Keys are released either way.
                release_error: _InterfaceCallFailed | None = None
                for k in reversed(pressed):
                    try:
                        await interface.key_up(k)
                        _record("key_up", key=k)
                    except _InterfaceCallFailed as e:
                        release_error = release_error or e
                if completed and release_error is not None:
                    raise release_error

        # Utility Actions
        elif name == "wait":
            duration = coerce_model_duration(
                _required_model_arg(args, "duration", name),
                action_name="wait",
            )
            await asyncio.sleep(duration)
            _record("sleep", duration=duration)

        elif name == "screenshot":
            pass  # dispatch no-op; the loop still captures this action's frame

        else:
            # NOT a recorded noop: ``step`` already gated on the DERIVED
            # ``_SUPPORTED_ACTIONS``, so a name arriving here is a gap in
            # this ladder (or a caller bypassing the gate), i.e. exactly the
            # "bug in this dispatch ladder itself" the module docstring above
            # says must raise rather than be screenshotted + eval'd as if the
            # action ran. INFRA_FAILURE, not a model action error: the model
            # cannot cause it, so it must not be fed back as agent feedback.
            raise TrueInfraFailure(
                f"desktop dispatch has no branch for action {name!r} "
                f"(args={args!r}); this backend executes "
                f"{sorted(_SUPPORTED_ACTIONS)}"
            )

    except _InterfaceCallFailed as e:
        # Interface-boundary failure (container died, exec pipe broke, call
        # timed out): degrade to a recorded noop so the step can still
        # screenshot + surface the state. Dispatch-ladder bugs propagate.
        logger.error("Action %s failed at the interface: %s", name, e.cause)
        calls.append({"call": "noop", "args": {"name": name, "reason": str(e.cause)}})

    return calls


class _InterfaceCallFailed(Exception):
    """An ``interface.<method>()`` call raised — the ONE failure class the
    action dispatcher downgrades to a recorded noop."""

    def __init__(self, method: str, cause: Exception):
        super().__init__(f"{method}: {cause}")
        self.method = method
        self.cause = cause


class _InterfaceBoundary:
    """Thin proxy marking the interface-call boundary: every method call is
    forwarded, and any exception it raises is wrapped in
    :class:`_InterfaceCallFailed` so the dispatcher's catch cannot swallow
    dispatch-ladder bugs."""

    def __init__(self, iface: Any):
        self._iface = iface

    def __getattr__(self, method: str) -> Any:
        fn = getattr(self._iface, method)

        async def call(*args: Any, **kwargs: Any) -> Any:
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                raise _InterfaceCallFailed(method, e) from e

        return call
