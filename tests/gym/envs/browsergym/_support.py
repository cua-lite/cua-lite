"""Shared BrowserGym test helpers."""

from __future__ import annotations

import io
from typing import Any

import pytest

pytest.importorskip("browsergym.core", reason="browsergym not installed")

from lite.gym.envs.browsergym.main import (
    BrowserGymConfig,
    BrowserGymEnv,
)

# ---------------------------------------------------------------------------
# Helpers (test-side)
# ---------------------------------------------------------------------------

def _make_fake(max_steps: int = 10, **cfg_overrides: Any) -> BrowserGymEnv:
    """Create a fake env for testing without a real browser.

    ``extra_tools`` is a ``BrowserGymEnv.__init__`` argument (the standalone
    extra-tool SELECTOR, resolved against the catalog ``action_subsets``
    derives; omitted/None → the whole catalog, [] → none, [names] → subset),
    NOT a ``BrowserGymConfig`` field — route it to the env, and let
    everything else be a config override.
    """
    env_kwargs = {
        k: cfg_overrides.pop(k)
        for k in ("extra_tools", "cursor")
        if k in cfg_overrides
    }
    config_kwargs = {
        "bgym_task_id": "miniwob.click-dialog",
        "benchmark": "miniwob",
        "viewport_width": 500,
        "viewport_height": 320,
        **cfg_overrides,
    }
    config = BrowserGymConfig(**config_kwargs)
    return BrowserGymEnv(config=config, max_steps=max_steps, use_fake=True, **env_kwargs)


_BROWSERGYM_T2_MODE_CONFIGS: dict[str, dict[str, Any]] = {
    "miniwob/default": {
        "benchmark": "miniwob",
        "bgym_task_id": "miniwob.click-dialog",
        "action_subsets": ("coord", "chat", "infeas", "nav"),
        "extra_tools": ["response", "terminate", "goto", "back", "forward"],
        "include_page_context_text": False,
        "use_screenshot": True,
        "use_ax_tree": False,
        "skip_dom_extraction": True,
    },
    "miniwob/text": {
        "benchmark": "miniwob",
        "bgym_task_id": "miniwob.click-dialog",
        "action_subsets": ("bid", "chat", "infeas"),
        "extra_tools": ["click", "fill", "response", "terminate"],
        "include_page_context_text": True,
        "use_screenshot": False,
        "use_ax_tree": True,
        "valid_actions": [],
    },
    "webarena/default": {
        "benchmark": "webarena",
        "bgym_task_id": "webarena.0",
        "action_subsets": ("coord", "chat", "infeas", "nav", "tab"),
        "extra_tools": ["response", "terminate", "goto", "back", "forward"],
        "include_page_context_text": False,
        "use_screenshot": True,
        "use_ax_tree": False,
        "skip_dom_extraction": True,
    },
    "webarena/text_only": {
        "benchmark": "webarena",
        "bgym_task_id": "webarena.0",
        "action_subsets": ("webarena",),
        "extra_tools": ["click", "fill", "response", "terminate"],
        "include_page_context_text": True,
        "use_screenshot": False,
        "use_ax_tree": True,
        "valid_actions": [],
    },
    "webarena/som": {
        "benchmark": "webarena",
        "bgym_task_id": "webarena.0",
        "action_subsets": ("webarena",),
        "extra_tools": ["click", "fill", "response", "terminate"],
        "include_page_context_text": True,
        "use_screenshot": True,
        "use_ax_tree": False,
        "use_som": True,
        "skip_dom_extraction": False,
        "valid_actions": [],
    },
    "visualwebarena/goal_image": {
        "benchmark": "visualwebarena",
        "bgym_task_id": "visualwebarena.5",
        "action_subsets": ("coord", "chat", "infeas", "nav", "tab"),
        "extra_tools": ["response", "terminate", "goto", "back", "forward"],
        "include_page_context_text": False,
        "use_screenshot": True,
        "use_ax_tree": False,
        "skip_dom_extraction": True,
    },
    "visualwebarena/mixed": {
        "benchmark": "visualwebarena",
        "bgym_task_id": "visualwebarena.5",
        "action_subsets": ("visualwebarena",),
        "extra_tools": ["click", "fill", "upload_file", "response", "terminate"],
        "include_page_context_text": True,
        "use_screenshot": False,
        "use_ax_tree": True,
        "valid_actions": [],
    },
    "visualwebarena/som": {
        "benchmark": "visualwebarena",
        "bgym_task_id": "visualwebarena.5",
        "action_subsets": ("visualwebarena",),
        "extra_tools": ["click", "fill", "upload_file", "response", "terminate"],
        "include_page_context_text": True,
        "use_screenshot": True,
        "use_ax_tree": False,
        "use_som": True,
        "skip_dom_extraction": False,
        "valid_actions": [],
    },
}


def _make_t2_browsergym_fake(mode: str, **overrides: Any) -> BrowserGymEnv:
    cfg = {**_BROWSERGYM_T2_MODE_CONFIGS[mode], **overrides}
    return _make_fake(**cfg)


def _png_bytes(width: int = 4, height: int = 4, color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    """Generate a small valid PNG; helpful for goal-image / screenshot tests."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def _decode_png_size(png: bytes) -> tuple[int, int]:
    from PIL import Image
    return Image.open(io.BytesIO(png)).size


def _fake_bgym_step_obs(
    *,
    screenshot: bytes | None = None,
    url: str = "http://browsergym.fake/",
) -> dict[str, Any]:
    return {
        "reward": 0.0,
        "screenshot": screenshot,
        "last_action_error": "",
        "open_pages_urls": (url,),
        "open_pages_titles": ("BrowserGym Fake",),
        "active_page_index": [0],
    }


__all__ = [
    "_BROWSERGYM_T2_MODE_CONFIGS",
    "_decode_png_size",
    "_fake_bgym_step_obs",
    "_make_fake",
    "_make_t2_browsergym_fake",
    "_png_bytes",
]
