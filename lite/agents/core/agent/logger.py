"""Lightweight trajectory logger for AdapterBasedAgent.

Implements ``SampleHook`` to save a resumable summary, optional LiteSample
parquet row, optional media, and per-turn debug artifacts alongside
``agent.sample()``. Per-turn artifact names are an internal debug layout; read
them through ``lite.infer.debug`` helpers instead of depending on this module's
docstring as a stable interface.

Usage:
    traj_logger = TrajectoryLogger(".logs/runs/my_run")
    lite_rl_sample = await agent.sample(env, hooks=[traj_logger])

Run:
    uv run python -m lite.agents.core.agent.logger
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageDraw, ImageFont

from lite.agents.core.agent.hooks import SampleHook, SampleStepData
from lite.agents.core.agent.utils.annotations import (
    action_inspection_records,
    annotation_coordinate,
    coordinate_annotation_records,
)
from lite.core.messages.final import STOP_REASON_INFO_KEY
from lite.core.messages.image_refs import validate_image_references
from lite.core.messages.selectors import instruction_text
from lite.core.samples import PERSISTED_FINAL_STOP_REASONS
from lite.core.tools.action_space.geometry import strict_norm_to_pixel
from lite.gym.types import EXECUTED_ACTIONS_INFO_KEY
from lite.utils.path import project_root

if TYPE_CHECKING:
    # Avoid import cycle: lite.agents.core.agent imports lite.agents.core.agent.hooks which
    # this file depends on via SampleHook. ``from __future__ import
    # annotations`` makes the runtime cost zero.
    from lite.core import (
        LiteRLSample,
    )

logger = logging.getLogger(__name__)
ENV_RESULT_IMAGES_DIR = "env_result_images"
ROLLOUT_METADATA_OTHER_KEYS = frozenset({
    "env_id",
    "task_id",
    "episode_return",
    "terminated",
    "truncated",
})


_LONG_STRING_DISPLAY_LIMIT = 500
_LONG_STRING_DISPLAY_HEAD = 200


def _render_long_string(text: str) -> str:
    """Truncate a long string for debug display. Shared by JSON artifact
    sanitization and the prompt-image action-label overlay so both apply the
    same long-string rendering policy instead of two ad hoc truncations."""
    if len(text) > _LONG_STRING_DISPLAY_LIMIT:
        return text[:_LONG_STRING_DISPLAY_HEAD] + f"... [{len(text)} chars total]"
    return text


def sanitize_trajectory_json(data: Any) -> Any:
    """Recursively replace image_url values and long base64 strings."""
    if isinstance(data, dict):
        out = {}
        for key, value in data.items():
            if key == "image_url":
                out[key] = "[omitted]"
            else:
                out[key] = sanitize_trajectory_json(value)
        return out
    elif isinstance(data, list):
        return [sanitize_trajectory_json(item) for item in data]
    elif isinstance(data, bytes | bytearray | memoryview):
        return f"[omitted {len(data)} bytes]"
    elif isinstance(data, str):
        # Likely base64 or very long content — truncate in JSON
        return _render_long_string(data)
    return data


def _safe_artifact_component(value: object) -> str:
    text = str(value or "unpaired")
    return "".join(
        ch
        if ("a" <= ch <= "z" or "A" <= ch <= "Z" or "0" <= ch <= "9" or ch in "._-")
        else "_"
        for ch in text
    )


def _save_env_result_images(
    turn_dir: Path,
    result_idx: int,
    result: Any,
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    images = list(result.images or [])
    if not images:
        return refs

    image_dir = turn_dir / ENV_RESULT_IMAGES_DIR
    image_dir.mkdir(exist_ok=True)
    safe_call_id = _safe_artifact_component(result.tool_call_id or f"result_{result_idx:04d}")
    for image_idx, image_bytes in enumerate(images):
        name = f"{result_idx:04d}_{image_idx:04d}_from_{safe_call_id}.png"
        rel_path = Path(ENV_RESULT_IMAGES_DIR) / name
        path = turn_dir / rel_path
        path.write_bytes(image_bytes)
        refs.append({
            "path": str(rel_path),
            "source": ENV_RESULT_IMAGES_DIR,
            "bytes": len(image_bytes),
            "sha1": hashlib.sha1(image_bytes).hexdigest(),
        })
    return refs


def _tool_results_for_log(turn_dir: Path, results: list[Any]) -> list[dict[str, Any]]:
    logged_results: list[dict[str, Any]] = []
    for result_idx, result in enumerate(results):
        logged_results.append(
            {
                "tool_call_id": result.tool_call_id,
                "images": _save_env_result_images(turn_dir, result_idx, result),
                "text": result.text,
                "metadata": result.metadata,
                "error": result.error,
            }
        )
    return logged_results


def build_trajectory_summary(
    *,
    n_turns: int,
    episode_return: float,
    terminated: bool,
    truncated: bool,
    stop_reason: str | None = None,
    duration_seconds: float | None = None,
    timing: dict | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build the shared ``summary.json`` payload.

    Optional keys are omitted when ``None`` so success summaries stay compact
    and ``data.get("error")`` separates valid records from errored records.
    """
    summary: dict[str, Any] = {
        "n_turns": n_turns,
        "episode_return": episode_return,
        "terminated": terminated,
        "truncated": truncated,
    }
    if duration_seconds is not None:
        summary["duration_seconds"] = duration_seconds
    # ``stop_reason`` is caller-filtered to exceptional durable reasons. Routine
    # no-tool finals stay visible in the per-turn result payload only.
    if stop_reason is not None:
        summary[STOP_REASON_INFO_KEY] = stop_reason
    if timing is not None:
        summary["timing"] = timing
    if error is not None:
        summary["error"] = error
    return summary

def _draw_crosshair(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    """Draw a red crosshair at (x, y) on an ImageDraw object."""
    size, width, color = 20, 2, "red"
    draw.line([(x - size, y), (x + size, y)], fill=color, width=width)
    draw.line([(x, y - size), (x, y + size)], fill=color, width=width)
    draw.ellipse([(x - 3, y - 3), (x + 3, y + 3)], fill=color)

def _wrap_text(
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    draw: ImageDraw.ImageDraw,
) -> str:
    """Word-wrap text to fit within max_width pixels."""
    words = text.split()
    lines: list[str] = []
    current_line = ""
    for word in words:
        test = f"{current_line} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current_line = test
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return "\n".join(lines)


def _draw_action_label(
    draw: ImageDraw.ImageDraw,
    image_size: tuple[int, int],
    actions_text: str,
) -> None:
    """Draw action list label at the bottom-left of the image."""
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    padding = 8
    w, h = image_size

    # Measure text bounding box
    bbox = draw.textbbox((0, 0), actions_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Background rectangle at bottom-left
    x0, y0 = 0, h - text_h - padding * 2
    x1, y1 = text_w + padding * 2, h
    draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, 180))
    draw.text((x0 + padding, y0 + padding), actions_text, fill="white", font=font)


def _image_content_sha1(image: Image.Image) -> str:
    normalized = image.convert("RGBA")
    h = hashlib.sha1()
    h.update(f"{normalized.size[0]}x{normalized.size[1]}:{normalized.mode}:".encode("ascii"))
    h.update(normalized.tobytes())
    return h.hexdigest()


class TrajectoryLogger(SampleHook):
    """Saves trajectory data to a turn-based directory.

    Extends ``SampleHook``: pass as a hook to
    ``agent.sample(hooks=[traj_logger])`` for automatic lifecycle management.

    Args:
        log_dir: Directory to save artifacts.
        save_data: Save turn directories + trajectory.parquet. When False,
            only summary.json is saved. The summary is still the rollout resume
            gate for a resolved sample.
        debug_artifacts: Save redundant turn-local prompt image caches and
            action annotations. Canonical parquet/media data does not depend on
            these caches.
        save_video: Save trajectory.mp4 (requires ffmpeg). Defaults to False.
        save_gif: Save trajectory.gif (PIL, no ffmpeg; downscaled). Defaults to False.
        render_instruction_banner: Burn the task instruction into the top of each
            mp4/gif frame. Defaults to True.
    """

    def __init__(
        self,
        log_dir: str | Path,
        *,
        save_data: bool = True,
        save_video: bool = False,
        save_gif: bool = False,
        debug_artifacts: bool = False,
        render_instruction_banner: bool = True,
        env_id: str | None = None,
        task_id: str | None = None,
        provenance: dict[str, Any] | None = None,
    ):
        self.log_dir = Path(log_dir)
        self.save_data = save_data
        self.save_video = save_video
        self.save_gif = save_gif
        self.debug_artifacts = debug_artifacts
        self.render_instruction_banner = render_instruction_banner
        # Rollout identity + provenance for every persisted trajectory.parquet
        # row. These stay in ``metadata.others`` as queryable row context
        # (env_id / task_id / model_id / agent_id / config_path / commit /
        # command), which the log-root-level ``run_info.txt`` cannot provide
        # once rows are staged/merged or the root accumulates resumes.
        self.env_id = env_id
        self.task_id = task_id
        self.provenance = provenance or {}
        self._step_timings: list[dict[str, float]] = []  # per-turn latency breakdowns
        self._media_frames: list[Image.Image] = []
        self._media_timeline_incomplete = False
        self._next_observation_source = "reset"
        self._final_stop_reason: str | None = None
        # Project root for computing relative image paths in trajectory.parquet —
        # the worktree this Python install came from. Found by marker (pyproject.toml
        # + lite/), NOT by __file__-depth counting, which silently broke when this
        # module moved deeper. Reading ``CUA_LITE_ROOT`` from the environment was
        # worktree-unsafe — a value inherited from another worktree's shell would
        # silently redirect every relative path.
        self._project_root = project_root()
        self._start_time = time.monotonic()

    def _save_json(self, path: Path, data: Any) -> None:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def _observation_image_name(
        self,
        step_idx: int,
        image_indices: tuple[int, ...],
        current_image_index: int | None = None,
    ) -> str:
        if current_image_index is not None:
            image_idx = current_image_index
        else:
            image_idx = image_indices[-1] if image_indices else step_idx
        source = self._next_observation_source
        suffix = "reset" if source == "reset" else f"from_{source}"
        return f"{image_idx:04d}_{suffix}.png"

    def _remember_next_observation_source(self, step_result: Any | None) -> None:
        if step_result is None:
            return
        for result in step_result.results:
            if result.images and result.tool_call_id:
                self._next_observation_source = str(result.tool_call_id)
                return

    def _append_media_frames(
        self,
        screenshot: Image.Image | None,
        step_result: Any | None,
        *,
        has_actions: bool,
    ) -> None:
        frames: list[Image.Image] = []
        if has_actions and screenshot is not None:
            frames.append(screenshot.copy())
        if step_result is None:
            self._media_frames.extend(frames)
            return
        for result in step_result.results:
            for image_bytes in result.images or []:
                try:
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        frames.append(image.convert("RGB"))
                except Exception as e:
                    self._media_timeline_incomplete = True
                    logger.warning(
                        "Failed to decode env result image for media; "
                        "falling back to stored trajectory images: %s",
                        e,
                    )
                    return
        self._media_frames.extend(frames)

    def on_step(self, data: SampleStepData) -> None:
        """Save turn artifacts and log to console."""
        step_idx = data.step_idx
        predict_result = data.predict_result
        step_result = data.step_result
        screenshot = data.image
        actions = data.actions
        inspected_actions = action_inspection_records(actions)
        env_info = step_result.info if step_result else {}
        stop_reason = env_info.get(STOP_REASON_INFO_KEY) if isinstance(env_info, dict) else None
        # Keep only durable final-stop classes for run summaries / row metadata.
        # Routine content-only / empty finals remain in the turn-local env info.
        if isinstance(stop_reason, str) and stop_reason in PERSISTED_FINAL_STOP_REASONS:
            self._final_stop_reason = str(stop_reason)

        if self.save_data:
            # Four digits preserve lexicographic order for long episodes.
            turn_dir = self.log_dir / f"turn_{step_idx:04d}"
            turn_dir.mkdir(parents=True, exist_ok=True)

            image_name = self._observation_image_name(
                step_idx,
                predict_result.step.image_indices,
                data.current_image_index,
            )

            # Redundant prompt image cache for debug inspection only.
            if self.debug_artifacts and screenshot is not None:
                image_dir = turn_dir / "prompt_images"
                image_dir.mkdir(exist_ok=True)
                screenshot.save(image_dir / image_name)

            # Turn-local debug artifacts; use lite.infer.debug readers for layout-sensitive tools.
            (turn_dir / "01_prompt.txt").write_text(predict_result.step.prompt, encoding="utf-8")

            (turn_dir / "02_response.txt").write_text(
                predict_result.step.response,
                encoding="utf-8",
            )

            actions_data: dict[str, Any] = {
                "agent_message": predict_result.agent_message,
                "lite_message": predict_result.lite_message,
                EXECUTED_ACTIONS_INFO_KEY: env_info.get(EXECUTED_ACTIONS_INFO_KEY, []),
            }

            logged_results = _tool_results_for_log(
                turn_dir,
                step_result.results if step_result else [],
            )

            results_data: dict[str, Any] = {
                "reward": step_result.reward if step_result else None,
                "terminated": step_result.terminated if step_result else False,
                "truncated": step_result.truncated if step_result else False,
                "results": logged_results,
                "info": env_info,
            }
            self._save_json(
                turn_dir / "03_actions.json",
                sanitize_trajectory_json(actions_data),
            )
            self._save_json(
                turn_dir / "04_results.json",
                sanitize_trajectory_json(results_data),
            )

            if data.timings:
                self._save_json(turn_dir / "05_timing.json", sanitize_trajectory_json(data.timings))

            # Optional coordinate overlay for the same prompt image debug cache.
            debug_coordinate_actions = coordinate_annotation_records(actions)
            if self.debug_artifacts and screenshot is not None and debug_coordinate_actions:
                annotated_dir = turn_dir / "prompt_images_annotated"
                annotated_dir.mkdir(exist_ok=True)
                self._save_prompt_image_debug_overlay(
                    screenshot,
                    inspected_actions,
                    annotated_dir / image_name,
                    debug_coordinate_actions=debug_coordinate_actions,
                )

        self._remember_next_observation_source(step_result)

        # Accumulate per-turn timings for the summary aggregate (always, even when
        # save_data is off — the aggregate is cheap and broadly useful).
        if data.timings:
            self._step_timings.append(dict(data.timings))

        if self.save_video or self.save_gif:
            self._append_media_frames(
                screenshot,
                step_result,
                has_actions=bool(actions),
            )

        # Console log (always)
        action_names = [tc["name"] for tc in inspected_actions]
        logger.info(
            "step %d: actions=%s reward=%s",
            step_idx, action_names, step_result.reward if step_result else None,
        )

    def _save_prompt_image_debug_overlay(
        self,
        screenshot: Image.Image,
        tool_calls: list[dict],
        out_path: Path,
        *,
        debug_coordinate_actions: list[dict],
    ) -> None:
        """Write a prompt-image debug copy with action labels and crosshairs.

        ``tool_calls`` are the inspection records of every executed action and
        carry the label list; ``debug_coordinate_actions`` is the drawable
        subset. Both come from ``utils.annotations``, which also resolves each
        record's point, so this renderer only projects and draws.
        """
        try:
            image = screenshot.copy().convert("RGBA")
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            w, h = image.size

            for tc in debug_coordinate_actions:
                px, py = strict_norm_to_pixel(
                    annotation_coordinate(tc["arguments"]),
                    w,
                    h,
                    clamp=True,
                )
                _draw_crosshair(draw, px, py)

            # Build action list label
            action_lines = []
            for i, tc in enumerate(tool_calls):
                name = tc["name"]
                args = tc["arguments"]
                args_str = ", ".join(
                    f"{k}={_render_long_string(str(v))}" for k, v in args.items()
                )
                action_lines.append(f"[{i}] {name}({args_str})")

            _draw_action_label(draw, (w, h), "\n".join(action_lines))

            # Composite and save
            composited = Image.alpha_composite(image, overlay)
            buf = io.BytesIO()
            composited.convert("RGB").save(buf, format="PNG")
            out_path.write_bytes(buf.getvalue())
        except Exception as e:
            logger.debug("Failed to write prompt image debug overlay: %s", e)

    def _media_timeline_frames(self, lite_rl_sample: LiteRLSample) -> list[Image.Image]:
        if self._media_frames and not self._media_timeline_incomplete:
            return self._dedup_adjacent_frames(self._media_frames)

        images = lite_rl_sample.lite_sample.images
        validate_image_references(lite_rl_sample.lite_sample.messages, images)
        return self._dedup_adjacent_frames(image.copy() for image in images)

    @staticmethod
    def _dedup_adjacent_frames(frames_in: Iterable[Image.Image]) -> list[Image.Image]:
        frames: list[Image.Image] = []
        previous_hash: str | None = None
        for image in frames_in:
            image_hash = _image_content_sha1(image)
            if frames and image_hash == previous_hash:
                continue
            frames.append(image.copy())
            previous_hash = image_hash
        return frames

    def on_complete(self, lite_rl_sample: LiteRLSample | None) -> None:
        """Save summary JSON and trajectory parquet (if enabled).

        ``summary.json`` marks a resolved attempt for rollout resume. Successful
        logger completions write it even when ``save_data=False``; rollout code
        separately owns terminal error summaries. ``lite_rl_sample is None``
        means the sample failed before that resolved boundary, so this hook must
        leave the summary absent and let retry/resume pick it up again.
        """
        if lite_rl_sample is None:
            return
        duration = time.monotonic() - self._start_time
        # Aggregate per-turn timing breakdown (predict=LLM, act=env.step, …) over
        # all turns. Unlike duration_seconds (which includes concurrency queue-wait
        # — the logger clock starts before the rollout semaphore), these sum only
        # the in-loop phase times, so they isolate where execution actually goes.
        timing_agg = None
        if self._step_timings:
            keys = {k for t in self._step_timings for k in t}
            timing_agg = {
                f"{k}_seconds_total": round(sum(t.get(k, 0.0) for t in self._step_timings), 2)
                for k in sorted(keys)
            }
            timing_agg["n_timed_turns"] = len(self._step_timings)
        summary = build_trajectory_summary(
            n_turns=len(lite_rl_sample.steps),
            episode_return=lite_rl_sample.episode_return,
            terminated=lite_rl_sample.terminated,
            truncated=lite_rl_sample.truncated,
            stop_reason=self._final_stop_reason,
            duration_seconds=round(duration, 2),
            timing=timing_agg,
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._save_json(self.log_dir / "summary.json", sanitize_trajectory_json(summary))

        if self.save_data:
            self._save_trajectory_parquet(lite_rl_sample)

        frames = (
            self._media_timeline_frames(lite_rl_sample)
            if (self.save_video or self.save_gif)
            else []
        )
        if frames:
            instruction = (
                instruction_text(lite_rl_sample.lite_sample.messages)
                if self.render_instruction_banner
                else ""
            )
            frames = self._banner_frames(frames, instruction)   # prepare once, reuse for mp4 + gif
            if self.save_video:
                self._save_video(frames)
            if self.save_gif:
                self._save_gif(frames)
        logger.info("trajectory saved to %s (%.1fs)", self.log_dir, duration)

    def _banner_frames(
        self,
        source_frames: list[Image.Image],
        instruction: str = "",
    ) -> list[Image.Image]:
        """RGB frames with a semi-transparent instruction banner + step counter
        burned in at the top. Shared by the mp4 / gif exporters."""
        frames = [img.convert("RGB") for img in source_frames]
        if not instruction:
            return frames

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except OSError:
            font = ImageFont.load_default()

        for i, frame in enumerate(frames):
            overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            w, _ = frame.size
            padding = 8

            # Word-wrap instruction to fit frame width
            wrapped = _wrap_text(instruction, font, w - padding * 2, draw)
            bbox = draw.textbbox((0, 0), wrapped, font=font)
            text_h = bbox[3] - bbox[1]

            # Draw banner background + text
            draw.rectangle([0, 0, w, text_h + padding * 2], fill=(0, 0, 0, 180))
            draw.text((padding, padding), wrapped, fill="white", font=font)

            # Also draw step counter at top-right
            step_text = f"{i + 1}/{len(frames)}"
            step_bbox = draw.textbbox((0, 0), step_text, font=font)
            step_w = step_bbox[2] - step_bbox[0]
            draw.text(
                (w - step_w - padding, padding),
                step_text,
                fill=(200, 200, 200, 255),
                font=font,
            )

            frames[i] = Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")
        return frames

    def _save_video(self, frames: list[Image.Image]) -> None:
        """Save prepared frames as mp4 via ffmpeg (1 fps)."""
        import subprocess
        import tempfile

        video_path = self.log_dir / "trajectory.mp4"
        w, h = frames[0].size
        # Ensure even dimensions for yuv420p
        w_out = w if w % 2 == 0 else w - 1
        h_out = h if h % 2 == 0 else h - 1
        cmd = [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", "1", "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", f"scale={w_out}:{h_out}",
            "-movflags", "faststart", str(video_path),
        ]
        # stderr goes to a file, never a pipe: libx264 emits more progress output
        # than a 64 KiB pipe buffer holds, and nothing here can drain it while the
        # frames are being written -- ffmpeg would block in write() and this
        # thread in wait(), stranding the rollout slot for the rest of the run.
        with tempfile.TemporaryFile() as stderr_file:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=stderr_file)
            for frame in frames:
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            proc.wait()
            if proc.returncode != 0:
                stderr_file.seek(0)
                raise RuntimeError(f"ffmpeg failed: {stderr_file.read().decode()}")
        logger.info("video saved to %s (%d frames)", video_path, len(frames))

    def _save_gif(self, frames: list[Image.Image], *, max_width: int = 960,
                  duration_ms: int = 1100) -> None:
        """Save prepared frames as an animated gif (PIL — no ffmpeg dependency).

        Frames are downscaled to ``max_width``, quantized with per-frame adaptive
        palettes and dithering, and the final frame is held longer so the end
        state is visible. The full-resolution record remains ``trajectory.mp4``
        and the per-turn PNG artifacts.
        """
        gif_path = self.log_dir / "trajectory.gif"
        scaled = []
        for f in frames:
            w, h = f.size
            if w > max_width:
                f = f.resize((max_width, max(1, round(h * max_width / w))), Image.LANCZOS)
            scaled.append(f.convert("RGB"))
        quantized = [
            f.quantize(colors=256, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG)
            for f in scaled
        ]
        durations = [duration_ms] * len(quantized)
        durations[-1] = duration_ms * 2          # hold the last frame
        quantized[0].save(
            gif_path, save_all=True, append_images=quantized[1:],
            duration=durations, loop=0, optimize=True,
        )
        logger.info("gif saved to %s (%d frames)", gif_path, len(quantized))

    def _save_trajectory_parquet(self, lite_rl_sample: LiteRLSample) -> None:
        """Save the LiteRLSample's LiteSample-shape view as trajectory.parquet
        (relative image paths). On-disk schema:
        ``{images: list[str_path], messages: list[dict], metadata: dict}``.

        Env identity/outcome fields (``env_id``, ``task_id``,
        ``episode_return``, ``terminated``, ``truncated``), run provenance, and
        domain/run extras stay under ``metadata.others`` so rows remain
        queryable without a sidecar.
        """
        from lite.utils.parquet import write_records_to_parquet

        validate_image_references(
            lite_rl_sample.lite_sample.messages,
            lite_rl_sample.lite_sample.images,
        )

        # Store the raw LiteSample images in their own stable sequence. Turn
        # directories are debugging artifacts and are not a reliable data source
        # once role:tool results can add images outside the next user turn.
        image_paths = []
        image_dir = self.log_dir / "images"
        if lite_rl_sample.lite_sample.images:
            image_dir.mkdir(parents=True, exist_ok=True)
        for image_idx, image in enumerate(lite_rl_sample.lite_sample.images):
            abs_path = (image_dir / f"{image_idx:06d}.png").resolve()
            image.save(abs_path)
            try:
                stored_path = str(abs_path.relative_to(self._project_root))
            except ValueError:
                stored_path = str(abs_path)
            image_paths.append(stored_path)

        metadata = lite_rl_sample.lite_sample.metadata.to_dict()

        metadata = {
            key: value
            for key, value in metadata.items()
            if key not in ROLLOUT_METADATA_OTHER_KEYS
        }
        source_others = dict(metadata.get("others") or {})
        source_others.pop(STOP_REASON_INFO_KEY, None)
        provenance = {
            key: value
            for key, value in self.provenance.items()
            if key not in ROLLOUT_METADATA_OTHER_KEYS
        }
        provenance.pop(STOP_REASON_INFO_KEY, None)
        final_stop_reason = (
            {STOP_REASON_INFO_KEY: self._final_stop_reason}
            if (
                isinstance(self._final_stop_reason, str)
                and self._final_stop_reason in PERSISTED_FINAL_STOP_REASONS
            )
            else {}
        )

        env_id = self.env_id or source_others.get("env_id")
        task_id = self.task_id or source_others.get("task_id")
        metadata = {
            **metadata,
            "others": {
                **provenance,
                **source_others,
                **({"env_id": env_id} if env_id else {}),
                **({"task_id": task_id} if task_id else {}),
                "episode_return": lite_rl_sample.episode_return,
                "terminated": lite_rl_sample.terminated,
                "truncated": lite_rl_sample.truncated,
                **final_stop_reason,
            },
        }

        record = {
            "images": image_paths,
            "messages": lite_rl_sample.lite_sample.messages,
            "metadata": metadata,
        }
        write_records_to_parquet(
            [record],
            self.log_dir / "trajectory.parquet",
            json_fields=("messages", "metadata"),
        )

if __name__ == "__main__":
    print("TrajectoryLogger")
    print("=" * 40)
    print("Usage: TrajectoryLogger(log_dir) — extends SampleHook")
