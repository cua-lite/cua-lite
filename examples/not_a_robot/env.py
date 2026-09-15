"""Direct Playwright environment for original local tasks and the Neal experiment.

The policy uses screenshots and canonical Lite actions. Original local tasks
have independent seeded resets and page-owned status grading. Local campaign
mode retains one browser/archive across the ordered 48 tasks and stops at missing
implementations without awarding completion. Remote Neal mode
reads visible text only for evaluation, supports independent reset for level
one, and has no unverified final-game campaign success heuristic.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.calls import RuntimeEnvAction
from lite.core.tools.extra_tools import LiteFinishToolSet
from lite.gym.base import LiteBaseEnv
from lite.gym.services import EnvServerResource
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult
from lite.gym.utils.backend.coordinate import norm_to_pixel
from lite.gym.utils.backend.model_inputs import coerce_model_duration, project_model_keys
from lite.gym.utils.feedback.errors import (
    error_only_feedback,
    record_model_action_error,
    record_tool_execution_error,
)
from lite.gym.utils.feedback.ingress import (
    invalid_action_message,
    prepare_env_tool_calls,
    standalone_tool_call_feedback,
)
from lite.gym.utils.feedback.results import build_tool_results_from_decisions, ordered_tool_call_ids
from lite.gym.wrappers import overlay_cursor_px

from .catalog import GAME_URL, LEVELS, VALID_ACTIONS
from .evaluator import GameState, classify_page
from .local_tasks import (
    ASSETS,
    CATALOG,
    LOCAL_TASKS,
    LocalTaskServer,
    LocalTaskState,
    task_reference,
)
from .recorder import EventRecorder


class NealAccessBlocked(RuntimeError):
    """The origin returned a real access gate; interaction must stop."""


# Direct-mode resources only: do not claim cross-worker recovery support.
LIVE_ENVS: dict[str, NotARobotEnv] = {}
LOCAL_CAMPAIGN_TASKS = tuple(f"neal_{row['level_id']:02d}" for row in LEVELS)


class NotARobotEnv(LiteBaseEnv, EnvServerResource):
    """Own one browser and one append-only episode archive per reset."""

    def __init__(
        self,
        *,
        target_level: int | None = 1,
        mode: str = "live",
        artifact_root: str = ".logs/not_a_robot",
        browser_executable: str | None = None,
        display_resolution: tuple[int, int] = (1280, 800),
        headless: bool = True,
        max_steps: int = 20,
        max_seconds: float = 180,
        post_action_delay: float = 0.2,
        cursor: bool = True,
        extra_tools: list[str] | None = None,
        local_task: str | None = None,
        local_campaign: bool = False,
        seed: int = 0,
        reference_instance: str = "default",
    ):
        if target_level not in (1, None):
            raise NotImplementedError("Only level one has an independent reset; use campaign mode.")
        if mode not in ("live", "fixture", "local"):
            raise ValueError("mode must be live, fixture or local")
        if mode == "local":
            if local_campaign and local_task is not None:
                raise ValueError("local_campaign and local_task are mutually exclusive")
            if (not local_campaign and local_task not in LOCAL_TASKS) or target_level is not None:
                raise ValueError("Local mode requires a known local_task and target_level=None")
            if type(seed) is not int or not 0 <= seed <= 4294967295:
                raise ValueError("seed must be an integer from 0 through 4294967295")
            if not local_campaign:
                task_reference(local_task, reference_instance)
        elif local_task is not None or local_campaign:
            raise ValueError("local_task and local_campaign are only supported in local mode")
        if reference_instance != "default" and (mode != "local" or local_campaign):
            raise ValueError("reference_instance requires an independent local task")
        if max_steps <= 0 or max_seconds <= 0 or post_action_delay < 0:
            raise ValueError("budgets must be positive and settle delay nonnegative")
        if os.environ.get("CUA_LITE_ENV_SERVER_PORT"):
            raise RuntimeError(
                "The Neal prototype supports direct mode only, not env-server workers."
            )
        self.target_level = target_level
        self.mode = mode
        self.local_task = local_task
        self.local_campaign = local_campaign
        self._completed_tasks: list[str] = []
        self.seed = seed
        self.reference_instance = reference_instance
        self._local_server: LocalTaskServer | None = None
        self._local_url: str | None = None
        self._game_event_sequence = 0
        self.artifact_root = Path(artifact_root)
        self.browser_executable = browser_executable
        self.display_resolution = tuple(display_resolution)
        self.headless = headless
        self.max_steps = max_steps
        self.max_seconds = max_seconds
        self.post_action_delay = post_action_delay
        self.cursor = cursor
        self._extra_tools = ["terminate"] if extra_tools is None else extra_tools
        self._playwright = self._browser = self._context = self._page = None
        self.recorder: EventRecorder | None = None
        self.attempt_dir: Path | None = None
        self.last_observation: dict[str, Any] | None = None
        self.state: GameState | LocalTaskState | None = None
        self.outcome = "not_started"
        self._terminal = False
        self._steps = 0
        self._started = 0.0
        self._resource_id = None
        self._cursor_xy = (0, 0)
        self._active_call_id = None
        self._scope_violation = None
        self._main_status = None
        self._main_gate = None
        self._pressed_keys: set[str] = set()
        self._pressed_buttons: set[str] = set()

    @property
    def current_local_task(self) -> str | None:
        """The task selected for navigation; state.task_id owns the displayed page."""
        if self.local_campaign:
            return LOCAL_CAMPAIGN_TASKS[
                min(len(self._completed_tasks), len(LOCAL_CAMPAIGN_TASKS) - 1)
            ]
        return self.local_task

    @staticmethod
    def task_metadata(
        target_level: int | None,
        mode: str = "live",
        extra_tools=None,
        local_task: str | None = None,
        seed: int = 0,
        local_campaign: bool = False,
        reference_instance: str = "default",
    ):
        provenance = {
            "source": "neal_not_a_robot",
            "target_level": target_level,
            "execution_mode": mode,
            "synthetic_only": True,
        }
        if mode == "local":
            provenance = {
                "source": "original_visual_tasks",
                "execution_mode": "local",
                "synthetic_only": True,
                "local_task": local_task,
                "seed": seed,
                "reference_instance": reference_instance,
                "game_version": CATALOG["version"],
                "clock": "real_time",
            }
            if local_campaign:
                provenance.update(
                    source="neal_reference_reconstruction",
                    local_campaign=True,
                    ordered_tasks=list(LOCAL_CAMPAIGN_TASKS),
                    seed_scope="local_dynamics_not_original_challenge_seed",
                )
                del provenance["local_task"]
            elif "reference" in LOCAL_TASKS[local_task]:
                provenance["reference"] = dict(task_reference(local_task, reference_instance))
                if provenance["reference"]["fidelity"] == "captured_instance":
                    provenance["source"] = "neal_reference_reconstruction"
                    provenance["seed_scope"] = "local_dynamics_not_original_challenge_seed"
        return LiteCUAMetadata(
            dims=("browser", "use"),
            valid_actions=list(VALID_ACTIONS),
            extra_tool_schemas=NotARobotEnv.extra_tool_schemas(
                ["terminate"] if extra_tools is None else extra_tools
            ),
            others=provenance,
        )

    def _runtime_metadata(self) -> LiteCUAMetadata:
        return self.task_metadata(
            self.target_level,
            self.mode,
            self._extra_tools,
            self.local_task,
            self.seed,
            self.local_campaign,
            self.reference_instance,
        )

    @property
    def external_resource_id(self) -> str | None:
        return self._resource_id

    async def _route(self, route) -> None:
        request = route.request
        parsed = urlsplit(request.url)
        if self.mode == "local":
            allowed = (
                self._local_server is not None
                and f"{parsed.scheme}://{parsed.netloc}" == self._local_server.origin
                and parsed.path in ASSETS
            )
            if request.is_navigation_request():
                allowed = allowed and request.url == self._local_url
            if allowed:
                await route.continue_()
                return
        allowed = (
            self.mode == "live"
            and parsed.scheme == "https"
            and parsed.hostname == "neal.fun"
            and parsed.port in (None, 443)
            and not parsed.path.startswith("/cdn-cgi/")
        )
        if request.is_navigation_request():
            allowed = allowed and parsed.path.rstrip("/") == "/not-a-robot"
        if allowed:
            await route.continue_()
            return
        self.recorder.emit(
            "request_blocked", url=request.url, navigation=request.is_navigation_request()
        )
        if request.is_navigation_request() and request.frame == self._page.main_frame:
            self._scope_violation = request.url
        await route.abort("blockedbyclient")

    async def _block_socket(self, socket) -> None:
        self.recorder.emit("websocket_blocked", url=socket.url)
        await socket.close()

    def _response(self, response) -> None:
        request = response.request
        if request.is_navigation_request() and request.frame == self._page.main_frame:
            self._main_status = response.status
            self._main_gate = response.headers.get("cf-mitigated")

    async def reset(self) -> LiteEnvObservation:
        await self.close()
        self._steps = 0
        self._terminal = False
        self.outcome = "in_progress"
        self.state = None
        self.last_observation = None
        self._game_event_sequence = 0
        self._completed_tasks = []
        self._pressed_keys.clear()
        self._pressed_buttons.clear()
        self._scope_violation = self._main_status = self._main_gate = None
        self._resource_id = uuid.uuid4().hex
        self.attempt_dir = (self.artifact_root / self._resource_id).resolve()
        self.recorder = EventRecorder(
            self.attempt_dir,
            {
                "env": self.metadata.to_dict(),
                "game_url": "owned_loopback" if self.mode == "local" else GAME_URL,
                "controller": "external",
                "viewport": list(self.display_resolution),
                "browser_executable": self.browser_executable,
                "headless": self.headless,
            },
        )
        LIVE_ENVS[self._resource_id] = self
        self._started = time.monotonic()
        try:
            if self.mode == "local":
                self._local_server = LocalTaskServer(
                    require_reference=LOCAL_TASKS[self.current_local_task]
                    .get("reference", {})
                    .get("fidelity")
                    == "captured_instance"
                )
                self._local_url = (
                    self._local_server.origin
                    + "/?"
                    + urlencode(
                        {
                            "task": self.current_local_task,
                            "seed": self.seed,
                            "instance": self.reference_instance,
                            "runner": 1,
                        }
                    )
                )
                self.recorder.emit(
                    "local_game_started",
                    url=self._local_url,
                    asset_hashes=self._local_server.asset_hashes,
                    clock="real_time",
                )
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=self.headless,
                executable_path=self.browser_executable,
            )
            w, h = self.display_resolution
            self._context = await self._browser.new_context(
                viewport={"width": w, "height": h},
                device_scale_factor=1,
                locale="en-US",
                timezone_id="UTC",
                service_workers="block",
            )
            await self._context.route("**/*", self._route)
            await self._context.route_web_socket("**/*", self._block_socket)
            self._page = await self._context.new_page()
            self._page.set_default_timeout(10_000)
            self._page.on("response", self._response)
            self._page.on(
                "pageerror", lambda error: self.recorder.emit("page_error", error=str(error))
            )
            self.recorder.emit("browser_started", version=self._browser.version)
            if self.mode == "local":
                await self._open_local_task()
            elif self.mode == "fixture":
                await self._page.set_content(Path(__file__).with_name("fixture.html").read_text())
            else:
                await self._page.goto(GAME_URL, wait_until="domcontentloaded", timeout=30_000)
            self._cursor_xy = (0, 0)
            # Observe access gates before sending any mouse or keyboard input.
            image = await self._observe("reset")
            if self.state.status == "access_blocked":
                self.outcome = "access_blocked"
                raise NealAccessBlocked(self.state.evidence)
            if self.mode == "local":
                if (
                    self.state.status != "in_progress"
                    or self.state.task_id != self.current_local_task
                    or self.state.seed != self.seed
                    or self.state.reference_instance != self.reference_instance
                    or self.state.version != CATALOG["version"]
                ):
                    raise RuntimeError(f"Local task reset returned unexpected state: {self.state}")
                return LiteEnvObservation(
                    image=image,
                    text=(
                        "Complete the local reconstruction in order from level 1 through 48. "
                        "Read each page's visible instructions. A completed level advances "
                        "automatically after an observation. Remaining actions in that batch "
                        "are not executed; inspect the new screenshot before acting. Missing "
                        "implementations stop the attempt without campaign success."
                        if self.local_campaign
                        else LOCAL_TASKS[self.local_task]["instruction"]
                    ),
                )
            if self.state.status != "in_progress" or self.state.level != 1:
                self.outcome = "unrecognized_start"
                raise RuntimeError(f"Expected verified level-one start, got {self.state}")
            instruction = (
                "Complete level 1 by interacting with the visible game."
                if self.target_level == 1
                else "Play the synthetic game through its levels using the visible instructions."
            )
            return LiteEnvObservation(image=image, text=instruction)
        except BaseException as error:
            if self.outcome == "in_progress":
                self.outcome = "infra_error"
            self.recorder.emit("error", phase="reset", error=repr(error))
            await self.close()
            raise

    async def _open_local_task(self) -> None:
        """Navigate the owned browser to one catalog task and reset page event offsets."""
        self._local_url = (
            self._local_server.origin
            + "/?"
            + urlencode(
                {
                    "task": self.current_local_task,
                    "seed": self.seed,
                    "instance": self.reference_instance,
                    "runner": 1,
                }
            )
        )
        self._game_event_sequence = 0
        await self._page.goto(self._local_url, wait_until="networkidle", timeout=10_000)
        await self._page.wait_for_function("() => window.syntheticTask !== undefined")
        await self._read_local_state()
        if (
            self.state.task_id != self.current_local_task
            or self.state.status != "in_progress"
            or self.state.version != CATALOG["version"]
            or self.state.seed != self.seed
            or self.state.reference_instance != self.reference_instance
        ):
            raise RuntimeError(f"Unexpected local page after navigation: {self.state}")
        if self.local_campaign:
            self.recorder.emit(
                "campaign_stage_started",
                task_id=self.state.task_id,
                completed_tasks=list(self._completed_tasks),
                reference=LOCAL_TASKS[self.state.task_id]["reference"],
                url=self._local_url,
            )

    async def _read_local_state(self) -> None:
        """Read only the original game's status and append unseen gameplay events."""
        try:
            snapshot = await self._page.evaluate("window.syntheticTask.snapshot()")
            events = snapshot.pop("events")
            self.state = LocalTaskState(**snapshot)
            if self.state.status not in ("in_progress", "success", "failure"):
                raise RuntimeError(f"Invalid local task status: {self.state.status}")
            for event in events[self._game_event_sequence :]:
                self.recorder.emit("game_event", task_id=self.state.task_id, **event)
            self._game_event_sequence = len(events)
        except asyncio.CancelledError:
            # The controller owns deadline/cancellation classification. A cancelled
            # observation cannot be resumed as another active environment step.
            self._terminal = True
            raise
        except BaseException as error:
            self.outcome, self._terminal = "infra_error", True
            self.recorder.emit("error", phase="local_state", error=repr(error))
            raise

    async def _observe(self, phase: str) -> bytes:
        try:
            if self.mode == "local":
                await self._read_local_state()
            task_context = {"task_id": self.state.task_id} if self.mode == "local" else {}
            capture_started = time.time()
            raw = await self._page.screenshot(type="png", timeout=10_000)
            capture_completed = time.time()
            if self.mode == "local":
                previous_status = self.state.status
                await self._read_local_state()
                if previous_status != self.state.status:
                    # A real-time deadline may expire during screenshot capture.
                    self.recorder.image(
                        raw,
                        phase=phase,
                        variant="before_terminal_transition",
                        capture_started_unix_s=capture_started,
                        capture_completed_unix_s=capture_completed,
                        **task_context,
                    )
                    capture_started = time.time()
                    raw = await self._page.screenshot(type="png", timeout=10_000)
                    capture_completed = time.time()
            self.recorder.image(
                raw,
                phase=phase,
                variant="raw",
                viewport=list(self.display_resolution),
                capture_started_unix_s=capture_started,
                capture_completed_unix_s=capture_completed,
                **task_context,
            )
            if self.mode != "local":
                title = await self._page.title()
                text = await self._page.locator("body").inner_text(timeout=5_000)
        except asyncio.CancelledError:
            self._terminal = True
            raise
        except BaseException as error:
            self.outcome, self._terminal = "infra_error", True
            self.recorder.emit("error", phase=phase, error=repr(error))
            raise
        if self.mode != "local":
            self.state = classify_page(
                url=self._page.url,
                title=title,
                text=text,
                http_status=self._main_status,
                cf_mitigated=self._main_gate,
            )
        model_image = (
            overlay_cursor_px(raw, *(round(value) for value in self._cursor_xy))
            if self.cursor
            else raw
        )
        self.last_observation = self.recorder.image(
            model_image,
            phase=phase,
            variant="model_visible",
            viewport=list(self.display_resolution),
            capture_started_unix_s=capture_started,
            capture_completed_unix_s=capture_completed,
            **task_context,
        )
        self.recorder.emit("task_state", state=dataclasses.asdict(self.state), url=self._page.url)
        return model_image

    async def _primitive(self, call: str, **args) -> None:
        if self._scope_violation:
            self.outcome = "scope_blocked"
            raise RuntimeError("Navigation outside the synthetic game was blocked.")
        if call != "wait" and self.mode != "local":
            # A real access gate can appear while the controller is thinking,
            # or between events of a compound action. Never interact with it.
            state = classify_page(
                url=self._page.url,
                title=await self._page.title(),
                text=await self._page.locator("body").inner_text(timeout=5_000),
                http_status=self._main_status,
                cf_mitigated=self._main_gate,
            )
            if state.status == "access_blocked":
                self.state, self.outcome = state, "access_blocked"
                raise NealAccessBlocked(state.evidence)
        event = self.recorder.emit(
            "input_started",
            call=call,
            args=args,
            tool_call_id=self._active_call_id,
            **({"task_id": self.state.task_id} if self.mode == "local" else {}),
        )
        try:
            if call == "wait":
                await asyncio.sleep(args["seconds"])
            else:
                device, method = call.split(".")
                await getattr(getattr(self._page, device), method)(**args)
                if device == "keyboard" and method in ("down", "up"):
                    if method == "down":
                        self._pressed_keys.add(args["key"])
                    else:
                        self._pressed_keys.discard(args["key"])
                if device == "mouse" and method in ("down", "up"):
                    if method == "down":
                        self._pressed_buttons.add(args["button"])
                    else:
                        self._pressed_buttons.discard(args["button"])
                if device == "mouse" and "x" in args:
                    self._cursor_xy = (args["x"], args["y"])
        except BaseException as error:
            self.recorder.emit(
                "input_error",
                parent_sequence=event["sequence"],
                error=repr(error),
                execution_uncertain=True,
            )
            raise
        self.recorder.emit("input_completed", parent_sequence=event["sequence"])

    async def _dispatch(self, name: str, args: dict) -> None:
        w, h = self.display_resolution
        coord = args.get("coordinate")
        if name == "mouse_move" and coord is None:
            raise ValueError(f"{name} requires coordinate")
        if name in ("wait", "hold_key"):
            duration = coerce_model_duration(args.get("duration", 1), action_name=name)
            if duration > 10:
                raise ValueError("This experimental environment limits waits/holds to 10 seconds.")
        if name in ("click", "mouse_move", "mouse_down", "mouse_up", "scroll"):
            x, y = self._cursor_xy
            if coord is not None:
                x, y = norm_to_pixel(coord, w, h, on_malformed="raise")
                await self._primitive("mouse.move", x=x, y=y)
            if name == "click":
                await self._primitive(
                    "mouse.click",
                    x=x,
                    y=y,
                    button=args.get("button", "left"),
                    click_count=args.get("clicks", 1),
                )
            elif name in ("mouse_down", "mouse_up"):
                await self._primitive(
                    "mouse." + name.removeprefix("mouse_"), button=args.get("button", "left")
                )
            elif name == "scroll":
                amount = float(args.get("amount", 3)) * 100
                direction = args.get("direction", "down")
                await self._primitive(
                    "mouse.wheel",
                    delta_x=amount * ({"left": -1, "right": 1}.get(direction, 0)),
                    delta_y=amount * ({"up": -1, "down": 1}.get(direction, 0)),
                )
        elif name == "drag":
            start = args.get("start_coordinate")
            sx, sy = (
                norm_to_pixel(start, w, h, on_malformed="raise")
                if start is not None
                else self._cursor_xy
            )
            ex, ey = norm_to_pixel(coord, w, h, on_malformed="raise")
            await self._primitive("mouse.move", x=sx, y=sy)
            button = args.get("button", "left")
            await self._primitive("mouse.down", button=button)
            try:
                for index in range(1, 21):
                    await self._primitive(
                        "mouse.move", x=sx + (ex - sx) * index / 20, y=sy + (ey - sy) * index / 20
                    )
                    await self._primitive("wait", seconds=0.02)
            finally:
                await self._primitive("mouse.up", button=button)
        elif name == "type":
            await self._primitive("keyboard.type", text=args["text"])
            if args.get("press_enter"):
                await self._primitive("keyboard.press", key="Enter")
        elif name in ("key", "key_down", "key_up", "hold_key"):
            keys = project_model_keys(args.get("keys", []), action_name=name, backend="playwright")
            if name == "key":
                await self._primitive("keyboard.press", key="+".join(keys))
            elif name == "key_up":
                for key in reversed(keys):
                    await self._primitive("keyboard.up", key=key)
            else:
                for key in keys:
                    await self._primitive("keyboard.down", key=key)
                if name == "hold_key":
                    try:
                        await self._primitive("wait", seconds=duration)
                    finally:
                        for key in reversed(keys):
                            await self._primitive("keyboard.up", key=key)
        elif name == "wait":
            await self._primitive("wait", seconds=duration)
        elif name != "screenshot":
            raise ValueError(f"Unsupported action {name}")

    async def step(self, actions: list[RuntimeEnvAction]) -> LiteEnvStepResult:
        if self._terminal or self._page is None:
            raise RuntimeError("Episode is not active")
        self.recorder.emit("actions_requested", actions=actions, turn=self._steps)
        ids = ordered_tool_call_ids(actions)
        prepared, feedback = prepare_env_tool_calls(actions, self.metadata)
        frames = []
        executed = []
        terminal_ids = set()
        self._steps += 1
        terminated = truncated = False
        stop_requested = False
        next_action_index = 0
        campaign_notice = None
        unexecuted_count = 0
        for index, (action, call_id) in enumerate(prepared):
            next_action_index = index
            name, args = action["name"], action["arguments"]
            self._active_call_id = call_id
            invalid = standalone_tool_call_feedback(
                action,
                type(self).known_standalone_tool_names(),
                self.metadata.extra_tool_schemas,
            )
            if invalid:
                if call_id:
                    feedback[call_id] = invalid
                next_action_index = index + 1
                continue
            stop_requested = name in LiteFinishToolSet.get_tool_names()
            if self.mode == "local":
                await self._read_local_state()
                if self.state.status in ("success", "failure"):
                    terminated = True
                    if name in LiteFinishToolSet.get_tool_names() and action.get("call_id"):
                        terminal_ids.add(action["call_id"])
                    if stop_requested:
                        next_action_index = index + 1
                    break
            if name in LiteFinishToolSet.get_tool_names():
                self.outcome = "agent_stopped"
                terminated = True
                if action.get("call_id"):
                    terminal_ids.add(action["call_id"])
                break
            reason = invalid_action_message(action, self.metadata.valid_actions)
            if reason:
                if call_id:
                    feedback[call_id] = error_only_feedback(reason)
                next_action_index = index + 1
                continue
            if time.monotonic() - self._started >= self.max_seconds:
                self.outcome, truncated = "timeout", True
                break
            try:
                await self._dispatch(name, args)
            except (ValueError, TypeError, KeyError) as error:
                self.recorder.emit("action_rejected", error=repr(error), tool_call_id=call_id)
                record_model_action_error(feedback, call_id, error, action_name=name)
                break
            except NealAccessBlocked as error:
                self.recorder.emit("access_blocked", evidence=str(error), tool_call_id=call_id)
                self.outcome, truncated = "access_blocked", True
                break
            except Exception as error:
                self.recorder.emit("error", phase="step", error=repr(error), tool_call_id=call_id)
                self.outcome, truncated = "infra_error", True
                record_tool_execution_error(feedback, call_id, error, action_name=name)
                break
            executed.append({"call": name, "args": args})
            next_action_index = index + 1
            await asyncio.sleep(self.post_action_delay)
            frames.append(await self._observe("post_action"))
            if self.state.status == "access_blocked":
                self.outcome, truncated = "access_blocked", True
                break
            if self._scope_violation:
                self.outcome, truncated = "scope_blocked", True
                break
            if self.mode == "local" and self.state.status in ("success", "failure"):
                terminated = True
                break
            if self.mode != "local" and self.target_level == 1 and self.state.level == 2:
                self.outcome, terminated = "success", True
                break
        if not frames:
            frames.append(await self._observe("step_end"))
        if self.state.status == "access_blocked":
            self.outcome, terminated, truncated = "access_blocked", False, True
        elif self._scope_violation:
            self.outcome, terminated, truncated = "scope_blocked", False, True
        elif self.mode == "local" and self.state.status in ("success", "failure"):
            # A page's success is not an episode success until campaign grading.
            # Keep this true even if the final-frame capture is cancelled.
            self.outcome = (
                "in_progress"
                if self.local_campaign and self.state.status == "success"
                else self.state.status
            )
            terminated, truncated = True, False
            if self.local_campaign and self.state.status == "success":
                # Archive the final page before navigating. Its game event sequence
                # restarts on the next page, but the episode sequence never does.
                frames.append(await self._observe("campaign_stage_end"))
                self._completed_tasks.append(self.state.task_id)
                self.recorder.emit(
                    "campaign_stage_completed",
                    state=dataclasses.asdict(self.state),
                    observation=self.last_observation,
                    completed_tasks=list(self._completed_tasks),
                )
                terminated = truncated = False
                if len(self._completed_tasks) == len(LOCAL_CAMPAIGN_TASKS):
                    self.outcome, terminated = "success", True
                elif stop_requested:
                    self.outcome, terminated = "agent_stopped", True
                elif self.current_local_task not in LOCAL_TASKS:
                    self.outcome, truncated = "unsupported_task", True
                elif (
                    self._steps >= self.max_steps
                    or time.monotonic() - self._started >= self.max_seconds
                ):
                    self.outcome, truncated = "budget_exhausted", True
                else:
                    self.outcome = "in_progress"
                    self._active_call_id = None
                    try:
                        # Playwright keeps device state across navigation. Release only
                        # our held inputs on the old page; never carry them to a new task.
                        for button in sorted(self._pressed_buttons):
                            await self._primitive("mouse.up", button=button)
                        for key in sorted(self._pressed_keys):
                            await self._primitive("keyboard.up", key=key)
                        await self._open_local_task()
                        frames.append(await self._observe("campaign_stage_start"))
                    except asyncio.CancelledError:
                        self._terminal = True
                        raise
                    except BaseException as error:
                        self.outcome, self._terminal = "infra_error", True
                        self.recorder.emit("error", phase="campaign_transition", error=repr(error))
                        raise
                campaign_notice = (
                    "The level completed. Remaining actions in this batch were not executed. "
                    + (
                        "Inspect the new level's screenshot before acting."
                        if not terminated and not truncated
                        else f"The campaign stopped with outcome {self.outcome}."
                    )
                )
                unexecuted_count = len(prepared) - next_action_index
                self.recorder.emit(
                    "campaign_boundary",
                    outcome=self.outcome,
                    next_task=(self.current_local_task if self.outcome != "success" else None),
                    unexecuted_actions=[action for action, _ in prepared[next_action_index:]],
                )
        if (
            not terminated
            and not truncated
            and (
                self._steps >= self.max_steps
                or time.monotonic() - self._started >= self.max_seconds
            )
        ):
            self.outcome, truncated = "budget_exhausted", True
        self._terminal = terminated or truncated
        reward = (1.0 if self.outcome == "success" else 0.0) if self._terminal else None
        self.recorder.emit(
            "step_result",
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            outcome=self.outcome,
            state=dataclasses.asdict(self.state),
        )
        return build_tool_results_from_decisions(
            LiteEnvStepResult(
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info={
                    "outcome": self.outcome,
                    "state": dataclasses.asdict(self.state),
                    "executed_actions": executed,
                    **(
                        {
                            "campaign": {
                                "completed_tasks": list(self._completed_tasks),
                                "displayed_task": self.state.task_id,
                                "total_tasks": len(LOCAL_CAMPAIGN_TASKS),
                                "next_task": (
                                    self.current_local_task
                                    if self.outcome == "unsupported_task"
                                    else None
                                ),
                                "notice": campaign_notice,
                                "unexecuted_actions": unexecuted_count,
                            }
                        }
                        if self.local_campaign
                        else {}
                    ),
                },
            ),
            ordered_call_ids=ids,
            continue_call_ids=[i for i in ids if i not in terminal_ids],
            images=frames,
            feedback=feedback,
            text=campaign_notice,
        )

    async def close(self) -> None:
        cleanup_errors = []
        try:
            if self._browser is not None:
                await self._browser.close()
        except Exception as error:
            cleanup_errors.append(repr(error))
        finally:
            self._browser = self._context = self._page = None
            if self._playwright is not None:
                try:
                    await self._playwright.stop()
                except Exception as error:
                    cleanup_errors.append(repr(error))
                self._playwright = None
            LIVE_ENVS.pop(self._resource_id, None)
            self._resource_id = None
            if self._local_server is not None:
                try:
                    self._local_server.close()
                except Exception as error:
                    cleanup_errors.append(repr(error))
                self._local_server = None
                self._local_url = None
        if self.recorder is not None:
            if self.outcome == "in_progress":
                self.outcome = "aborted"
            recorder = self.recorder
            if self.local_campaign:
                recorder.emit(
                    "campaign_end",
                    outcome=self.outcome,
                    completed_tasks=list(self._completed_tasks),
                    total_tasks=len(LOCAL_CAMPAIGN_TASKS),
                    displayed_task=self.state.task_id if self.state is not None else None,
                )
            self.recorder = None
            recorder.finalize(
                self.outcome, cleanup_errors=cleanup_errors, cleanup_complete=not cleanup_errors
            )
