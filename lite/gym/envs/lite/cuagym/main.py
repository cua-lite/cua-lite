"""lite.cuagym — CUA-Gym browser + desktop tasks as ONE Docker-container gym env.

Both CUA-Gym task families now run on the same substrate — one Sandbox
exec-stdio Docker container per rollout, driven by coordinate actions — so
they're a single env_id. They differ only by per-task setup/reward, carried on
each task:

  - upstream web / cross_app rows: the CUA-Gym-Hub mock is served by
    ``vite preview`` and the agent drives a real Chrome. (src/browser/scripts.py)
  - desktop: explicit desktop rows plus upstream rows with missing platform
    labels, including multi-app bundles, on the lite.osworld desktop.
    (src/desktop/scripts.py)

ONE image ``cua-lite/lite.cuagym`` serves both — their extras (browser mocks + Node,
desktop's doc/PDF libs) are additive and non-conflicting, so there's no reason to split it.
Each task keeps its official ``initial_setup`` + ``reward.py`` semantics; the browser
backend materializes mock URL placeholders and both platforms provide their own
``setup_fn``/``evaluate_final_fn`` transport.
The registered count is derived from the complete pinned upstream catalogs: every
pinned row is registered, and rows that are unusable as default training signals
because of pinned upstream defects are ANNOTATED with
``metadata.others.exclude_reason`` rather than dropped. Callers
filter them (``--filter "lambda m: not m.others.get('exclude_reason')"``); an
unfiltered one is refused at setup by ``guard_excluded`` below.

Setup (one image + both task sets):
    uv run --no-sync bash lite/gym/envs/lite/cuagym/scripts/install.sh

Usage:
    import lite.gym as gym
    gym.registry.task_ids("lite.cuagym", split="train")   # browser + desktop
    gym.make("lite.cuagym@<task_uuid>")                    # backend picked by task
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from lite.gym.envs.lite.cuagym.src.browser import scripts as browser_scripts
from lite.gym.envs.lite.cuagym.src.desktop import scripts as desktop_scripts
from lite.gym.envs.lite.cuagym.src.utils.runtime import (
    CFG,
    READY_MARKER,
    desktop_alive,
    desktop_failure_diagnostic,
    validate_post_setup_runtime,
    validate_pre_setup_runtime,
)
from lite.gym.envs.lite.cuagym.src.utils.runtime import (
    COMPUTER_CONFIG as _COMPUTER_CONFIG,
)
from lite.gym.envs.lite.cuagym.src.utils.runtime import (
    ENV_KWARGS as _EK,
)
from lite.gym.errors import CuaGymTaskError, EnvDepsMissingError
from lite.gym.sandbox import SandboxBaseEnv, register_jsonl_tasks

logger = logging.getLogger(__name__)

_ENV_ID = "lite.cuagym"
_DIR = Path(__file__).parent
_MAX_STEPS = _EK["max_steps"]
_POST_ACTION_DELAY = _EK["post_action_delay"]
_IMAGE = _COMPUTER_CONFIG["image"]
#: gym-anything contract: fresh runner per episode = Layer-B cap 0;
#: canonical telling in configs/default.yaml.
_MAX_RESETS_PER_CONTAINER = CFG.server_kwargs["max_resets_per_container"]

#: ``(SandboxTaskConfig, container handle) -> None`` — the backend setup contract.
_SetupFn = Callable[[Any, Any], Awaitable[None]]

_CACHE_ROOT = _DIR / ".cache"
_WEB_ROOT = _CACHE_ROOT / "web" / "lite.cuagym_tasks"
_DESKTOP_ROOT = _CACHE_ROOT / "desktop" / "lite.cuagym_desktop_tasks"
_WEB_JSONL = _WEB_ROOT / "train.jsonl"
_DESKTOP_JSONL = _DESKTOP_ROOT / "train.jsonl"


def _computer() -> dict:
    # exec-stdio transport: SandboxBaseEnv waits for GNOME readiness over
    # docker exec (no published computer-server port).
    return dict(_COMPUTER_CONFIG)


async def _read_gnome_failed_marker(container_name: str | None) -> str | None:
    if not container_name:
        return None
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", container_name, "cat", "/tmp/gnome-failed",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        if proc.returncode == 0:
            return stdout.decode("utf-8", errors="replace").strip() or None
    except Exception:
        pass
    return None


class _CuaGymEnv(SandboxBaseEnv):
    """Env class for both lite.cuagym backends. CUA-Gym tasks are deterministic —
    fixed official setup + reward scripts, no domain randomization — so the
    rollout framework's ``seed`` soft-kwarg (injected per group when
    ``group_shared_seed`` is on) is accepted and ignored: there is nothing
    stochastic to seed. The framework contract requires each concrete env to
    subclass ``bind`` with an explicit signature naming the soft kwargs it
    accepts (cf. ``LiteOsworldEnv.bind``). This override keeps that soft-kwarg
    explicit for CUA-Gym even though the base sandbox implementation only stores it.
    """

    READY_MARKER = READY_MARKER

    def __init__(
        self,
        task=None,
        *,
        display_resolution: tuple[int, int] = (1920, 1080),
        image: str | None = None,
        vnc_port: int | None = None,
        env_id: str | None = _ENV_ID,
        setup_fn: Any = None,
        evaluate_step_fn: Any = None,
        evaluate_final_fn: Any = None,
        container: str | None = None,
        computer_config: dict[str, Any] | None = None,
        post_action_delay: float = 1.0,
        debug: bool = False,
        seed: int | None = None,
        max_steps: int | None = None,
        valid_actions: list[str] | None = None,
        extra_tools: list[str] | None = None,
    ) -> None:
        resolution = tuple(display_resolution)
        if image not in (None, _IMAGE):
            raise ValueError(
                "lite.cuagym does not accept image overrides because setup and "
                "reward paths target its pinned runtime image"
            )
        if resolution != (1920, 1080):
            raise ValueError(
                "lite.cuagym does not accept display_resolution overrides because "
                "upstream tasks target the 1920x1080 desktop contract"
            )
        _check_images()
        if computer_config is None:
            computer_config = _COMPUTER_CONFIG
        self._set_constructor_state(
            display_resolution=resolution,
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
        #: Reuse cap consumed by ``reset_with_recycle`` (0 — see default.yaml).
        self._max_resets_per_container = _MAX_RESETS_PER_CONTAINER
        #: True once THIS backend has served (started) an episode — the
        #: pristine criterion (cleared when the backend is destroyed).
        self._cuagym_episode_started = False
        self.bind(
            task,
            seed=seed,
            max_steps=max_steps,
            valid_actions=valid_actions,
            extra_tools=extra_tools,
        )

    def bind(self, task=None, *, seed: int | None = None, max_steps: int | None = None,
             valid_actions: list[str] | None = None,
             extra_tools: list[str] | None = None) -> None:
        if isinstance(task, str) and task:
            from lite.gym.sandbox import lookup_task
            task = lookup_task(self._env_id or _ENV_ID, task)
        elif task == "":
            task = None
        # ``extra_tools`` is forwarded verbatim: lite.cuagym owns no env-local
        # tool set, so the base resolver serves it (``bash`` + finish tools).
        super().bind(task, max_steps=max_steps, seed=seed,
                     valid_actions=valid_actions, extra_tools=extra_tools)
        # Refresh task-owned callbacks after bind; browser and desktop tasks share
        # this env class but use different setup/evaluators.
        self._setup_fn = task.setup_fn if task is not None else None
        self._evaluate_step_fn = (
            task.evaluate_step_fn if task is not None else None
        )
        self._evaluate_final_fn = (
            task.evaluate_final_fn if task is not None else None
        )

    async def reset(self):
        from lite.gym.errors import CapacityExhausted, EnvDesktopCrashed

        try:
            # Framework-owned fresh-runner recycle (cap 0) —
            # replaces the hand-rolled episode discard this env carried
            # before its Layer-B migration. Recycle owns boot.
            await self.reset_with_recycle()
            assert self._computer is not None
            await validate_pre_setup_runtime(self._computer)
            result = await super().reset()
        except CapacityExhausted:
            await self.destroy_backend()
            raise
        except Exception as e:
            diag = None
            computer = getattr(self, "_computer", None)
            if computer is not None:
                try:
                    diag = await desktop_failure_diagnostic(computer)
                except Exception:
                    pass
            if not diag:
                diag = await _read_gnome_failed_marker(
                    getattr(self, "_container_name", None)
                )
            await self.destroy_backend()
            if diag:
                raise EnvDesktopCrashed(
                    "lite.cuagym reset failed; GNOME boot diagnostic from container:\n"
                    f"{diag}\n\n"
                    f"Original error: {type(e).__name__}: {e}"
                ) from e
            raise
        try:
            await validate_post_setup_runtime(
                self._computer,
                screenshot=result.image,
            )
        except Exception as e:
            await self.destroy_backend()
            raise EnvDesktopCrashed(
                "lite.cuagym post-reset screenshot/desktop health probe failed; "
                "retry on a fresh container."
            ) from e
        self._cuagym_episode_started = True
        return result

    def backend_alive(self) -> bool:
        return self._computer is not None

    async def reset_to_pristine(self) -> bool:
        # No rollback primitive — but a runner that never served an episode
        # (the first reset before any episode work — the only moment the
        # framework consults pristine under cap 0) IS pristine. One that served
        # → False → destroy + reboot.
        return not self._cuagym_episode_started

    async def destroy_backend(self) -> None:
        """Layer-B destroy hook: drop this episode's container
        without deleting an explicitly injected one."""
        self._cuagym_episode_started = False
        if self._owns_computer:
            await self._release_failed_attempt()
            return
        await SandboxBaseEnv.close(self)
        self._container_name = None
        self._container_override = None
        self._owns_computer = True

    async def _desktop_alive(self) -> bool:
        if not getattr(self, "_computer", None):
            return True
        return await desktop_alive(self._computer)


def guard_excluded(setup_fn: _SetupFn) -> _SetupFn:
    """Wrap a backend ``setup_fn`` so an excluded row is refused FAST, at setup.

    Same guard, same error semantics as the sibling env — cf.
    ``lite/gym/envs/lite/scalecua/src/osworld/setup.py`` (``ScaleCuaTaskError``,
    ``phase="setup"``, ``kind="excluded_task"``). It lives here rather than inline
    in ``src/{browser,desktop}/scripts.py`` only because lite.cuagym has TWO backend
    ``setup_fn``s and one registration site; the wrapper is applied where the
    catalogs are registered, so it is the ``setup_fn`` every task carries.

    513 of the 10910 pinned rows carry ``metadata.others.exclude_reason``
    (closed vocabulary: ``src/utils/dataset.py::EXCLUDE_REASONS``). Those rows stay
    REGISTERED — this env never silently drops a row — but each is either
    unrunnable by construction or a confirmed pinned reward/spec mismatch, so
    without this guard an unfiltered run burns a full episode and only then
    raises a task-level error or returns a misleading 0, which an FN/TN analysis
    cannot tell apart from a genuine agent failure.

    ``CuaGymTaskError`` is a :class:`~lite.gym.errors.LiteGymError`, which is what
    makes the refusal an ENV/setup error rather than a reward-0 trajectory:
    ``SandboxBaseEnv.reset`` only converts ``RuntimeError``/``TimeoutError`` raised
    by setup into a retryable ``CapacityExhausted``, so this propagates out of
    ``env.reset()`` and out of ``agent.sample`` untouched, and ``is_retryable``
    is False (terminal).
    The rollout worker records it as ``error=<traceback>``, which
    ``rebuild_results`` excludes from ``valid`` instead of averaging in as a 0.
    """

    @functools.wraps(setup_fn)
    async def _setup_fn(task, computer) -> None:
        reason = (task.metadata.get("others") or {}).get("exclude_reason")
        if reason:
            raise CuaGymTaskError(
                f"lite.cuagym task {task.task_id} is excluded: {reason}",
                phase="setup",
                kind="excluded_task",
            )
        await setup_fn(task, computer)

    return _setup_fn


_TASK_COMPUTER = _computer()
_catalog_registered = False


def _register_tasks() -> None:
    global _catalog_registered
    if _catalog_registered:
        return
    missing = [
        label
        for label, path in (
            ("web cache", _WEB_JSONL),
            ("desktop cache", _DESKTOP_JSONL),
        )
        if not path.is_file()
    ]
    if missing:
        raise EnvDepsMissingError(
            what=(
                "lite.cuagym task cache is missing: "
                + ", ".join(missing)
            ),
            install=(
                "uv run --no-sync bash "
                "lite/gym/envs/lite/cuagym/scripts/install.sh pull"
            ),
            see="lite/gym/envs/lite/cuagym/README.md",
        )
    from lite.gym.envs.lite.cuagym.src.utils import dataset

    expected_asset = dataset.asset_identity()
    stale_stamps: list[str] = []
    for label, root in (("web", _WEB_ROOT), ("desktop", _DESKTOP_ROOT)):
        stamp = root / ".asset_revision"
        digest_stamp = root / ".asset_digest"
        try:
            actual_asset = stamp.read_text().strip() if stamp.is_file() else ""
        except OSError as exc:
            raise EnvDepsMissingError(
                what=f"lite.cuagym {label} task cache stamp is unreadable",
                install=(
                    "uv run --no-sync bash "
                    "lite/gym/envs/lite/cuagym/scripts/install.sh provision"
                ),
                see="lite/gym/envs/lite/cuagym/README.md",
            ) from exc
        if actual_asset != expected_asset:
            stale_stamps.append(label)
            continue
        try:
            actual_digest = (
                digest_stamp.read_text().strip() if digest_stamp.is_file() else ""
            )
            expected_digest = dataset.task_cache_digest(root)
        except (OSError, ValueError) as exc:
            raise EnvDepsMissingError(
                what=(
                    f"lite.cuagym {label} task cache content digest "
                    f"could not be validated ({type(exc).__name__})"
                ),
                install=(
                    "uv run --no-sync bash "
                    "lite/gym/envs/lite/cuagym/scripts/install.sh provision"
                ),
                see="lite/gym/envs/lite/cuagym/README.md",
            ) from exc
        if actual_digest != expected_digest:
            stale_stamps.append(label)
    if stale_stamps:
        raise EnvDepsMissingError(
            what=(
                "lite.cuagym task cache is stale or invalid: "
                + ", ".join(f"{label} cache" for label in stale_stamps)
            ),
            install=(
                "uv run --no-sync bash "
                "lite/gym/envs/lite/cuagym/scripts/install.sh provision"
            ),
            see="lite/gym/envs/lite/cuagym/README.md",
        )

    try:
        for path in (_WEB_JSONL, _DESKTOP_JSONL):
            dataset.validate_catalog(
                path,
                required_metadata_paths=("setup", "reward"),
            )
        dataset.validate_catalog_count_lock({
            "desktop": _DESKTOP_JSONL,
            "web": _WEB_JSONL,
        })
    except (
        RuntimeError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
    ) as exc:
        raise EnvDepsMissingError(
            what=(
                "lite.cuagym task cache is invalid "
                f"({type(exc).__name__})"
            ),
            install=(
                "uv run --no-sync bash "
                "lite/gym/envs/lite/cuagym/scripts/install.sh provision"
            ),
            see="lite/gym/envs/lite/cuagym/README.md",
        ) from exc

    common = {
        "env_id": _ENV_ID,
        "split": "train",
        "computer": _TASK_COMPUTER,
        "factory_kwargs": {
            "computer_config": _TASK_COMPUTER,
            "post_action_delay": _POST_ACTION_DELAY,
        },
        "env_class": _CuaGymEnv,
        "default_max_steps": _MAX_STEPS,
    }
    register_jsonl_tasks(
        _WEB_JSONL,
        **common,
        setup_fn=guard_excluded(browser_scripts.setup_fn),
        evaluate_final_fn=browser_scripts.evaluate_final_fn,
        platform="browser",
    )
    register_jsonl_tasks(
        _DESKTOP_JSONL,
        **common,
        setup_fn=guard_excluded(desktop_scripts.setup_fn),
        evaluate_final_fn=desktop_scripts.evaluate_final_fn,
        platform="desktop",
    )
    _catalog_registered = True

def _check_images() -> None:
    """Verify the docker image is built (one image serves both backends)."""
    from lite.gym.utils.backend.docker import require_image_present
    from lite.gym.utils.backend.freshness import image_for

    require_image_present(image_for(_ENV_ID, tag=_IMAGE))


def ensure_services(env_id: str) -> None:
    """env-server startup hook: verify the docker image(s) are built."""
    _check_images()


from lite.gym.registry import registry  # noqa: E402
from lite.gym.remote.reaper import ContainerServices  # noqa: E402
from lite.gym.services import BackendFamily, register_family, register_services  # noqa: E402


class CuaGymServices(ContainerServices):
    """Env-server capabilities for lite.cuagym: image/deps ensure plus normal
    per-container reconciliation inherited from ContainerServices."""

    def register_tasks(self, env_id: str) -> None:
        _register_tasks()

    def ensure(self, env_id: str) -> None:
        ensure_services(env_id)


register_services(_ENV_ID, CuaGymServices())
register_family(_ENV_ID, BackendFamily.DEDICATED)
registry.set_env_make_kwargs(_ENV_ID, CFG.make_kwargs)
if _WEB_JSONL.is_file() and _DESKTOP_JSONL.is_file():
    try:
        _register_tasks()
    except EnvDepsMissingError as exc:
        logger.warning("lite.cuagym catalog registration deferred: %s", exc.what)
