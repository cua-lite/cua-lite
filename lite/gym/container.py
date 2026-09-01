"""LiteContainerBase — the ONE contract for a DEDICATED per-episode container handle.

Run: not directly. Subclassed by the env container adapters (osworld,
osworld_2, androidworld, androidlab, mobileworld).

Layer 3 of the framework: the concrete
container *handle* under the capability layer. The base does two jobs:

1. **Declares the interface** the framework types against — ``name`` /
   ``api_port`` (subclass dataclass fields), ``base_url`` / ``start()``
   (abstract: host/port shape and boot pipeline genuinely differ per env).
2. **Provides the invariants every env used to hand-roll** (and drift on):
   the process-wide tracked registry + atexit backstop, normalized port
   ownership (``_ports_owned``), and a template-method :meth:`destroy` that
   is idempotent, never raises, and always runs its cleanup tail.

Subclasses override :meth:`_pre_destroy` ONLY — never :meth:`destroy` itself
(overriding the template would silently lose its guarantees; the ``release``
alias dispatches to the template, which dispatches to your hook).

A new container env therefore writes: a ``@dataclass`` subclass with its
fields, ``start()``, and ``base_url`` — teardown, registry, atexit and port
release are inherited and cannot be copied wrong.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

from lite.gym.errors import CapacityExhausted
from lite.gym.utils.backend.docker import docker_rm_f
from lite.gym.utils.backend.ports import release_ports
from lite.gym.utils.backend.reaper import reap

logger = logging.getLogger(__name__)


# ── Process-wide registry + atexit backstop (ONE for all container types) ────
# Every LiteContainerBase self-registers at the top of its boot path (see
# `_register`'s docstring for the ordering contract) and de-registers when
# destroy() completes; the atexit hook reaps whatever is still tracked on
# hard interpreter exit (Ctrl-C / unhandled exception; SIGKILL still leaks —
# that's the drift reaper's job, server-side).
_TRACKED: list[LiteContainerBase] = []
_TRACKED_LOCK = threading.Lock()


@atexit.register
def _cleanup_tracked_on_exit() -> None:
    reap(_TRACKED, _TRACKED_LOCK, lambda c: c.destroy())


class LiteContainerBase(ABC):
    """Contract + shared teardown for a per-episode DEDICATED container.

    Mixin ABC, deliberately NOT a ``@dataclass``: the concrete containers are
    dataclasses and provide the fields the base's methods read. Dataclass
    semantics require the *fields* to stay declared on each subclass (a
    non-dataclass base's bare annotation is not collected as a field); the
    base carries the annotations for typing plus the shared behavior.
    """

    # ── Subclass (dataclass) fields the base types against ──────────────────
    #: Docker container name (the reaper-visible ``external_resource_id``).
    name: str
    #: Host-side port the in-container API server is published on.
    api_port: int
    #: Host ports reserved via ``lite.gym.utils.backend.ports`` — released (once) by
    #: destroy(). Declare on the subclass as
    #: ``_ports_owned: tuple[int, ...] = field(default=(), repr=False)``.
    _ports_owned: tuple[int, ...]

    # ── Teardown knobs (same spelling as SingletonContainerServices) ────────
    #: ``docker rm -f -v`` bound; a wedged daemon can't hang destroy() longer.
    rm_timeout_s: ClassVar[float] = 60.0
    #: Log-prefix for the rm-timeout warning (set to the env id).
    rm_label: ClassVar[str] = "container"

    # ── Env-specific surface ─────────────────────────────────────────────────
    @property
    @abstractmethod
    def base_url(self) -> str:
        """Host-side URL of the in-container API server (the single channel
        through which the host drives the container)."""

    @abstractmethod
    def start(self) -> None:
        """``docker run`` + env-specific boot pipeline + readiness wait.

        Contract: call :meth:`_register` FIRST — before the container-creating
        call — so the whole (possibly multi-minute) boot window is covered by
        the atexit backstop (a KeyboardInterrupt mid-boot is a BaseException
        that retry loops' ``except Exception`` won't catch)."""

    # ── Registry (base-owned; envs declare zero registry state) ─────────────
    def _register(self) -> None:
        with _TRACKED_LOCK:
            if self not in _TRACKED:
                _TRACKED.append(self)

    def _deregister(self) -> None:
        # Discard-not-remove: a second destroy() (release alias, or
        # reaper-after-close) must be a true no-op.
        with _TRACKED_LOCK:
            try:
                _TRACKED.remove(self)
            except ValueError:
                pass

    # ── Teardown (template method) ───────────────────────────────────────────
    def _pre_destroy(self) -> None:
        """Hook: runs before the ``docker rm``. Default no-op. Override for
        diagnostic probes (androidworld's postmortem dump). Exceptions are
        swallowed — a diagnostic hook must never block teardown."""

    def _release_ports(self) -> None:
        """Release ``_ports_owned`` (once). Swallows internally and clears
        ownership after release, so a double-destroy can never re-release a
        port another process may have reused since."""
        ports = tuple(self._ports_owned)
        if not ports:
            return
        try:
            release_ports(*ports)
        except Exception:
            pass
        self._ports_owned = ()

    def destroy(self) -> None:
        """``docker rm -f -v`` + release ports + de-register. Idempotent,
        never raises — the rm is routed through :func:`docker_rm_f` (which
        swallows TimeoutExpired and already-gone), so the cleanup tail ALWAYS
        runs; a raw ``subprocess.run(timeout=…)`` here would raise on a
        wedged daemon and leak the port reservation + registry entry.

        Subclasses must NOT override this — override :meth:`_pre_destroy`.
        """
        try:
            self._pre_destroy()
        except Exception:
            pass  # diagnostic hook must never block teardown
        docker_rm_f(self.name, timeout=self.rm_timeout_s, label=self.rm_label)
        self._release_ports()
        self._deregister()


def boot_with_retry(
    build: Callable[[], LiteContainerBase],
    *,
    start: Callable[[LiteContainerBase], None],
    attempt_gate: Callable[[], Any] | None = None,
    retry_barrier: Callable[[LiteContainerBase], None] | None = None,
    max_attempts: int = 2,
    label: str = "container",
) -> LiteContainerBase:
    """THE construct→boot→retry orchestration for DEDICATED acquires.

    ``build()`` returns a fresh UN-started container (name assigned, ports
    reserved, any pre-reap done); ``start(c)`` boots it (all callers wrap
    ``c.start()`` to hold ``docker_create_slot()`` across the boot);
    ``attempt_gate()`` returns a context manager held across the
    WHOLE attempt including failure cleanup (mobileworld's ``_SPAWN_SEMA``).
    Classification, single-sourced:

    * ``RuntimeError`` | ``CapacityExhausted`` → transient boot failure
      (docker-run timeout, boot deadline, warming): destroy + retry with a
      FRESH ``build()`` (new port + name — lower contention next attempt).
      On the last attempt the final error re-raises (a warming
      ``CapacityExhausted`` → 503 + Retry-After → client retry).
    * anything else (docker daemon down, image missing) → non-transient:
      destroy + raise immediately, no retry.

    ``retry_barrier`` is an optional env-owned check after failed-attempt
    cleanup and before constructing the next attempt.
    """
    last: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        with (attempt_gate() if attempt_gate is not None else contextlib.nullcontext()):
            c = build()
            try:
                start(c)
                return c
            except (RuntimeError, CapacityExhausted) as e:
                last = e
                logger.warning("%s acquire attempt %d/%d failed: %s",
                               label, attempt, max_attempts, e)
                c.destroy()
                if retry_barrier is not None and attempt < max_attempts:
                    retry_barrier(c)
            except Exception:
                c.destroy()
                raise
    assert last is not None
    raise last


def spawn_background_destroy(
    cf: Any,
    *,
    on_result: Callable[[Any], None],
    on_failure: Callable[[BaseException], None] | None = None,
    on_done: Callable[[], None] | None = None,
    timeout: float | None = None,
    thread_name: str = "bg-destroy",
) -> None:
    """Daemon-thread shell for reaping a cancelled in-flight boot —
    the "a boot is still running when reset/close arrives" dance, previously
    re-implemented per env with big warning comments.

    Waits for the ``concurrent.futures`` future (optionally bounded), then
    dispatches: ``on_result(result)`` on success, ``on_failure(exc)`` on any
    failure/timeout, and ``on_done()`` ALWAYS afterwards (the conditional
    handle-nulling tail). Callback exceptions are logged, never raised —
    this thread must survive event-loop shutdown and die quietly.

    **HARD RULE (the shared-attr hazard):** callbacks must capture the
    dying container/handle as a LOCAL or closure argument — never read it
    back off a shared attr like ``_current_container`` at destroy time; the
    next acquire reassigns that attr, and a shared read would destroy the
    NEW container. (Reading a ``_pending_*`` attr that only the boot path
    writes — mobileworld's failure branch — is the documented exception.)
    """

    def _bg() -> None:
        try:
            res = cf.result(timeout) if timeout is not None else cf.result()
        except BaseException as e:
            if on_failure is not None:
                try:
                    on_failure(e)
                except Exception as e2:
                    logger.warning("%s: on_failure callback failed: %s", thread_name, e2)
        else:
            try:
                on_result(res)
            except Exception as e2:
                logger.warning("%s: on_result callback failed: %s", thread_name, e2)
        if on_done is not None:
            try:
                on_done()
            except Exception as e2:
                logger.warning("%s: on_done callback failed: %s", thread_name, e2)

    threading.Thread(target=_bg, daemon=True, name=thread_name).start()
