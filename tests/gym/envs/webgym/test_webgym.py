"""Tests for the WebGym CUA-Lite gym environment (OmniBoxes HTTP wrapper).

Requires:
  - pip install httpx  (or uv sync)
  - For live tests (`-m live`): the cua-lite/webgym image built (install.sh) + a
    docker daemon — gym.make brings the container up via WebGymContainerServices.

Run:
    uv run python -m pytest tests/gym/envs/webgym/test_webgym.py        # unit only
    uv run python -m pytest tests/gym/envs/webgym/test_webgym.py -m live  # + live
    uv run python -m pytest tests/gym/envs/webgym/test_webgym.py -m live \
        -k cursor_toggle_and_lease_cleanup
    uv run pytest tests/gym/remote/test_direct_server_parity_matrix.py -q
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lite.core.tools import make_tool_call
from lite.core.tools.schemas import tool_schema_name
from lite.gym.envs.webgym.main import (
    WebGymClient,
    WebGymEnv,
)
from lite.gym.types import EXECUTED_ACTIONS_INFO_KEY
from lite.gym.utils.feedback.ingress import make_internal_terminate_action

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_TASK = {
    "task_id": "0",
    "task_name": "Find the release date of Oppenheimer on IMDb",
    "domain": "Entertainment",
    "subdomain": "Films & TV Shows",
    "website": "imdb.com",
    "difficulty": 2,
    "evaluator_reference": [{"description": "release date is July 21, 2023"}],
    "definite_answer": "July 21, 2023",
}


# Generate a mock PNG that exceeds _MIN_SCREENSHOT_SIZE (1000 bytes) and passes
# WebGym's blank-screenshot validation (non-white, random pixels).
def _make_test_png(seed: int = 42) -> bytes:
    import io
    import random as _rng

    from PIL import Image as _Img

    _rng.seed(seed)
    img = _Img.new("RGB", (50, 50))
    img.putdata(
        [(_rng.randint(0, 255), _rng.randint(0, 255), _rng.randint(0, 255)) for _ in range(2500)]
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_TINY_PNG = _make_test_png()
_OTHER_PNG = _make_test_png(seed=7)


def _decode_png_rgba(png: bytes):
    assert png[:4] == b"\x89PNG"
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(png))
    img.load()
    return img.convert("RGBA")


def _assert_shared_cursor_sprite_at(
    baseline_png: bytes,
    cursor_png: bytes,
    *,
    x: int,
    y: int,
) -> dict[str, int]:
    """Assert ``cursor_png`` has the shared 16x24 cursor over ``baseline_png``.

    WebGym composites the same sprite in the container-side capture path. This
    intentionally uses a tolerant pixel comparison so it remains useful for live
    browser screenshots as well as the unit fake.
    """
    from lite.gym.wrappers import load_cursor_sprite

    baseline = _decode_png_rgba(baseline_png)
    cursor = _decode_png_rgba(cursor_png)
    assert cursor.size == baseline.size

    sprite = load_cursor_sprite(height=24).convert("RGBA")
    assert sprite.size == (16, 24)
    assert 0 <= x < baseline.width
    assert 0 <= y < baseline.height

    w = min(sprite.width, baseline.width - x)
    h = min(sprite.height, baseline.height - y)
    assert (w, h) == sprite.size

    baseline_roi = baseline.crop((x, y, x + w, y + h))
    cursor_roi = cursor.crop((x, y, x + w, y + h))
    expected = baseline_roi.copy()
    expected.alpha_composite(sprite.crop((0, 0, w, h)))

    opaque_total = 0
    opaque_changed = 0
    opaque_close = 0
    alpha_total = 0
    alpha_changed = 0
    transparent_changed = 0
    for py in range(h):
        for px in range(w):
            alpha = sprite.getpixel((px, py))[3]
            before = baseline_roi.getpixel((px, py))[:3]
            after = cursor_roi.getpixel((px, py))[:3]
            want = expected.getpixel((px, py))[:3]
            changed = max(abs(after[i] - before[i]) for i in range(3))
            close = max(abs(after[i] - want[i]) for i in range(3))
            if alpha:
                alpha_total += 1
                if changed > 18:
                    alpha_changed += 1
            else:
                if changed > 18:
                    transparent_changed += 1
            if alpha >= 240:
                opaque_total += 1
                if changed > 18:
                    opaque_changed += 1
                if close <= 60:
                    opaque_close += 1

    assert opaque_total >= 80
    assert alpha_total >= 200
    assert opaque_changed >= 35, {
        "opaque_changed": opaque_changed,
        "opaque_total": opaque_total,
    }
    assert opaque_close >= 35, {
        "opaque_close": opaque_close,
        "opaque_total": opaque_total,
    }
    assert alpha_changed >= 75, {
        "alpha_changed": alpha_changed,
        "alpha_total": alpha_total,
    }
    assert transparent_changed <= 12, {
        "transparent_changed": transparent_changed,
    }
    return {
        "opaque_changed": opaque_changed,
        "opaque_close": opaque_close,
        "alpha_changed": alpha_changed,
        "transparent_changed": transparent_changed,
    }


def _unwrap_webgym_env(env) -> WebGymEnv:
    return env._env if hasattr(env, "_env") else env


@pytest.fixture(autouse=True)
def _webgym_env_vars():
    """Snapshot + restore WEBGYM_MASTER_URL / WEBGYM_API_KEY so the unit
    fixtures' infra/secrets (read from env-vars inside __init__)
    never leak into sibling tests. Set them here so _make_env / the metadata
    test don't write bare os.environ entries with no teardown."""
    saved = {k: os.environ.get(k) for k in ("WEBGYM_MASTER_URL", "WEBGYM_API_KEY")}
    os.environ["WEBGYM_MASTER_URL"] = "http://localhost:7000"
    os.environ["WEBGYM_API_KEY"] = "test_key"
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _make_mock_client() -> AsyncMock:
    """Create a mock WebGymClient for unit testing."""
    client = AsyncMock(spec=WebGymClient)
    client.get_instance.return_value = {"instance_id": "test-uuid:9001", "node": "test-node"}
    client.get_metadata.return_value = {"width": 1280, "height": 768}
    client.screenshot.return_value = _TINY_PNG
    client.execute.return_value = {"status": "success"}
    client.get_page_metadata.return_value = {"title": "Test Page", "url": "https://imdb.com"}
    client.reset_instance.return_value = None
    client.close.return_value = None
    client._failed_urls = []
    return client


def _make_env(
    task: dict | None = None,
    max_steps: int = 10,
    extra_tools: list[str] | None = None,
    valid_actions: list[str] | None = None,
    cursor: bool = True,
) -> WebGymEnv:
    """Create a WebGymEnv for unit testing.

    Infra/secrets (master_url, api_key) are no longer __init__ params — they're
    read from env-vars inside __init__. The autouse
    ``_webgym_env_vars`` fixture sets them (with teardown), so the unit env never
    falls back to the real yaml default and never leaks into sibling tests.
    """
    return WebGymEnv(
        task=task or _SAMPLE_TASK,
        max_steps=max_steps,
        skip_eval=True,
        extra_tools=extra_tools,
        valid_actions=valid_actions
        if valid_actions is not None
        else ["click", "type", "key", "scroll", "wait"],
        cursor=cursor,
    )


def _load_webgym_controller_patch(monkeypatch, path=None):
    """Load the vendored controller with tiny dependency stubs.

    The controller patch is copied into the WebGym container at build time; these
    tests execute the cursor-state helpers directly without requiring OmniBoxes or
    Playwright to be installed host-side.

    ``path`` overrides the module file, so a caller can load a copy staged next to
    a stub ``_page_script.js`` and exercise the real ``__init__`` (upstream ships
    that asset, the repo does not).
    """
    import sys
    import types

    timeout_error = type("TimeoutError", (Exception,), {})
    target_closed_error = type("TargetClosedError", (Exception,), {})
    playwright_error = type("PlaywrightError", (Exception,), {})

    modules = {
        "playwright": types.ModuleType("playwright"),
        "playwright._impl": types.ModuleType("playwright._impl"),
        "playwright._impl._errors": types.ModuleType("playwright._impl._errors"),
        "playwright.async_api": types.ModuleType("playwright.async_api"),
        "omniboxes": types.ModuleType("omniboxes"),
        "omniboxes.node": types.ModuleType("omniboxes.node"),
        "omniboxes.node.instances": types.ModuleType("omniboxes.node.instances"),
        "omniboxes.node.instances._types": types.ModuleType("omniboxes.node.instances._types"),
    }
    for name in (
        "playwright",
        "playwright._impl",
        "omniboxes",
        "omniboxes.node",
        "omniboxes.node.instances",
    ):
        modules[name].__path__ = []
    errors = modules["playwright._impl._errors"]
    errors.Error = playwright_error
    errors.TimeoutError = timeout_error
    errors.TargetClosedError = target_closed_error
    async_api = modules["playwright.async_api"]
    async_api.Download = type("Download", (), {})
    async_api.Page = type("Page", (), {})
    async_api.TimeoutError = timeout_error
    types_mod = modules["omniboxes.node.instances._types"]
    types_mod.InteractiveRegion = dict
    types_mod.VisualViewport = dict
    types_mod.interactiveregion_from_dict = lambda d: d
    types_mod.visualviewport_from_dict = lambda d: d
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    path = path or (
        Path(__file__).resolve().parents[4]
        / "lite/gym/envs/webgym/docker/patches/_playwright_controller.py"
    )
    spec = importlib.util.spec_from_file_location("_webgym_controller_patch_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeMouse:
    def __init__(self, *, fail_click: bool = False, fail_move: bool = False):
        self.fail_click = fail_click
        self.fail_move = fail_move
        self.moves: list[tuple[float, float]] = []
        self.clicks: list[tuple[float, float]] = []
        self.wheels: list[tuple[int, int]] = []

    async def move(self, x, y, steps=None):
        if self.fail_move:
            raise RuntimeError("move failed")
        self.moves.append((x, y))

    async def click(self, x, y, delay=None):
        if self.fail_click:
            raise RuntimeError("click failed")
        self.clicks.append((x, y))

    async def wheel(self, dx, dy):
        self.wheels.append((dx, dy))


class _FakeKeyboard:
    async def press(self, _key):
        return None

    async def type(self, _value, delay=None):
        return None


class _FakeTarget:
    def __init__(self, box: dict[str, float]):
        self._box = box
        self.first = self

    async def wait_for(self, timeout=None):
        return None

    async def scroll_into_view_if_needed(self):
        return None

    async def bounding_box(self):
        return dict(self._box)

    async def evaluate(self, *_args, **_kwargs):
        return None


class _FakeExpectEvent:
    def __init__(self, timeout_error):
        self._timeout_error = timeout_error

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc, _tb):
        return False

    @property
    def value(self):
        async def _raise_timeout():
            raise self._timeout_error("no popup")

        return _raise_timeout()


class _FakePage:
    url = "about:blank"

    def __init__(
        self,
        module,
        *,
        fail_click: bool = False,
        fail_move: bool = False,
        screenshot_bytes: bytes = _TINY_PNG,
    ):
        self.mouse = _FakeMouse(fail_click=fail_click, fail_move=fail_move)
        self.keyboard = _FakeKeyboard()
        self.target = _FakeTarget({"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0})
        self._timeout_error = module.TimeoutError
        self.marked = False
        self.evaluations = []
        self.screenshot_bytes = screenshot_bytes
        self.screenshot_calls = []

    async def evaluate(self, *args, **_kwargs):
        self.evaluations.append(args)
        return None

    async def screenshot(self, *args, **kwargs):
        self.screenshot_calls.append((args, kwargs))
        return self.screenshot_bytes

    async def wait_for_selector(self, *_args, **_kwargs):
        return None

    def locator(self, _selector):
        return self.target

    def expect_event(self, *_args, **_kwargs):
        return _FakeExpectEvent(self._timeout_error)

    async def wait_for_load_state(self, *args, **_kwargs):
        return None

    async def wait_for_timeout(self, *_args, **_kwargs):
        return None


def _controller_instance(module, *, animate_actions: bool = False):
    controller = module.PlaywrightController.__new__(module.PlaywrightController)
    controller.animate_actions = animate_actions
    controller.downloads_folder = None
    controller.viewport_width = 1280
    controller.viewport_height = 720
    controller.single_tab_mode = True
    controller._sleep_after_action = 0
    controller._timeout_load = 1
    controller.last_cursor_position = (0.0, 0.0)

    async def _ensure_page_ready(_page):
        return None

    controller._ensure_page_ready = _ensure_page_ready
    controller._mark_action = lambda page: setattr(page, "marked", True)
    return controller


def _assert_webgym_current_carrier(
    tool_result,
    *,
    call_id: str,
    image: bytes = _TINY_PNG,
    error: str | None = None,
) -> None:
    assert tool_result.tool_call_id == call_id
    assert tool_result.images[-1] == image
    assert "https://imdb.com" in (tool_result.text or "")
    assert tool_result.error == error
    expected_metadata = {
        "page_title": "Test Page",
        "url": "https://imdb.com",
    }
    if error is not None:
        expected_metadata["is_error"] = True
    assert tool_result.metadata == expected_metadata


def _assert_error_only_carrier(tool_result, *, call_id: str, error: str) -> None:
    assert tool_result.tool_call_id == call_id
    assert tool_result.images == []
    assert tool_result.text is None
    assert tool_result.error == error
    assert tool_result.metadata == {"is_error": True}


# ---------------------------------------------------------------------------
# Unit tests: action translation
# ---------------------------------------------------------------------------


class TestActionTranslation:
    """Test CUA-Lite -> WebGym HTTP command translation."""

    def _translate(self, name: str, args: dict) -> dict | None:
        env = _make_env()
        env._viewport = (1280, 768)
        return env._translate_action(name, args)

    def test_click(self):
        cmd = self._translate("click", {"coordinate": [500, 500]})
        assert cmd == {"click_coords": {"x": 640, "y": 384}}

    def test_click_corner(self):
        cmd = self._translate("click", {"coordinate": [0, 0]})
        assert cmd == {"click_coords": {"x": 0, "y": 0}}

    def test_click_max(self):
        cmd = self._translate("click", {"coordinate": [1000, 1000]})
        # Clamped to vw-1, vh-1
        assert cmd == {"click_coords": {"x": 1279, "y": 767}}

    def test_click_tracks_position(self):
        env = _make_env()
        env._viewport = (1280, 768)
        env._translate_action("click", {"coordinate": [500, 500]})
        assert env._last_click_x == 640
        assert env._last_click_y == 384

    def test_type_uses_fill_coords(self):
        env = _make_env()
        env._viewport = (1280, 768)
        # Simulate a prior click
        env._last_click_x = 640
        env._last_click_y = 384
        cmd = env._translate_action("type", {"text": "hello world"})
        assert cmd == {
            "fill_coords": {
                "x": 640,
                "y": 384,
                "value": "hello world",
                "press_enter": False,
                "delete_existing": True,
            }
        }

    def test_type_projects_a_trailing_newline_onto_the_container_flag(self):
        """The model owns whether to submit; webgym must not override it.

        Canonical spells "submit" as a trailing newline inside ``text``. The
        container's own wire keeps a ``press_enter`` flag, so the host strips
        the newline and sets the flag -- rather than letting the newline ride
        along, which would leave the value one character long and depend on the
        container's typing path. Only a TRAILING newline submits; an interior
        one is part of the value.
        """
        env = _make_env()
        env._viewport = (1280, 768)
        env._last_click_x, env._last_click_y = 640, 384
        for text, value, submits in (
            ("q\n", "q", True),
            ("q", "q", False),
            ("a\nb", "a\nb", False),
            ("", "", False),
        ):
            cmd = env._translate_action("type", {"text": text})
            assert cmd["fill_coords"]["value"] == value, text
            assert cmd["fill_coords"]["press_enter"] is submits, text

    def test_type_with_coordinate(self):
        """Type action with explicit coordinate uses it instead of last click."""
        env = _make_env()
        env._viewport = (1280, 768)
        env._last_click_x = 0
        env._last_click_y = 0
        cmd = env._translate_action("type", {"text": "query", "coordinate": [500, 500]})
        assert cmd == {
            "fill_coords": {
                "x": 640,
                "y": 384,
                "value": "query",
                "press_enter": False,
                "delete_existing": True,
            }
        }

    def test_key_basic(self):
        # Keys arrive as canonical named/glyph tokens; the webgym host resolves
        # them to final Playwright names for the container.
        cmd = self._translate("key", {"keys": ["enter"]})
        assert cmd == {"keypress": {"keys": ["Enter"]}}

    def test_key_combo(self):
        cmd = self._translate("key", {"keys": ["ctrl", "c"]})
        assert cmd == {"keypress": {"keys": ["Control", "c"]}}

    def test_key_glyphs(self):
        cmd = self._translate("key", {"keys": ["+", "-", "="]})
        assert cmd == {"keypress": {"keys": ["+", "-", "="]}}

    def test_key_mapping(self):
        """Verify canonical → Playwright key-name projection."""
        cmd = self._translate("key", {"keys": ["esc"]})
        assert cmd == {"keypress": {"keys": ["Escape"]}}
        cmd = self._translate("key", {"keys": ["backspace"]})
        assert cmd == {"keypress": {"keys": ["Backspace"]}}

    def test_key_empty_raises_model_visible_error(self):
        # An empty ``keys`` used to translate to ``None``; the container's
        # ``if keys:`` then pressed nothing and the model got a normal screenshot
        # for a keypress that never happened. ``keys`` is required with no default.
        with pytest.raises(ValueError, match="key.keys must not be empty"):
            self._translate("key", {"keys": []})

    @pytest.mark.parametrize(
        ("keys", "expected"),
        [
            (["plus"], "unknown key token 'plus'"),
            ([" "], "unknown key token ' '"),
        ],
    )
    def test_key_rejects_noncanonical_key_tokens(self, keys, expected):
        with pytest.raises(ValueError, match=expected):
            self._translate("key", {"keys": keys})

    def test_scroll_with_coords(self):
        # The element-anchored scroll must carry `amount` (regression guard: it
        # used to be dropped, so element scrolls ignored the agent's magnitude).
        cmd = self._translate(
            "scroll",
            {
                "coordinate": [500, 500],
                "direction": "down",
                "amount": 3,
            },
        )
        assert cmd == {
            "hover_and_scroll_coords": {
                "x": 640,
                "y": 384,
                "direction": "down",
                "amount": 3,
            }
        }

    def test_scroll_no_coords_down(self):
        cmd = self._translate("scroll", {"direction": "down", "amount": 2})
        assert cmd == {"page_down": {"amount": 200}}

    def test_scroll_no_coords_up(self):
        cmd = self._translate("scroll", {"direction": "up", "amount": 3})
        assert cmd == {"page_up": {"amount": 300}}

    def test_mouse_move(self):
        cmd = self._translate("mouse_move", {"coordinate": [250, 750]})
        assert cmd == {"hover_coords": {"x": 320, "y": 576}}

    def test_back(self):
        cmd = self._translate("back", {})
        assert cmd == {"back": {}}

    def test_goto(self):
        cmd = self._translate("goto", {"url": "https://example.com"})
        assert cmd == {"visit_page": {"url": "https://example.com"}}

    def test_drag_fallback(self):
        cmd = self._translate(
            "drag",
            {
                "start_coordinate": [100, 100],
                "coordinate": [900, 900],
            },
        )
        # Falls back to click at end position
        assert "click_coords" in cmd

    def test_drag_requires_coordinate(self):
        with pytest.raises(KeyError, match="coordinate"):
            self._translate("drag", {"start_coordinate": [100, 100]})

    def test_wait_noop(self):
        # wait is handled in _step_inner before _translate_action;
        # _translate_action returns None (skip)
        assert self._translate("wait", {"duration": 1.0}) is None

    def test_screenshot_noop(self):
        assert self._translate("screenshot", {}) is None

    def test_unknown_action_noop(self):
        # Unknown actions are skipped (no execution)
        assert self._translate("unknown_action", {}) is None


def test_playwright_instance_cursor_wire_bool_parses_false_values():
    import ast

    path = (
        Path(__file__).resolve().parents[4]
        / "lite/gym/envs/webgym/docker/patches/playwright_instance.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_coerce_cursor_bool"
    )
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"Any": object}
    exec(compile(module, str(path), "exec"), ns)
    coerce = ns["_coerce_cursor_bool"]

    for value in (False, 0, 0.0, "0", "false", "False", " no ", "off", ""):
        assert coerce(value) is False
    for value in (True, 1, "1", "true", "yes", None):
        assert coerce(value) is True


def test_instance_server_screenshot_route_forwards_cursor_static():
    path = (
        Path(__file__).resolve().parents[4]
        / "lite/gym/envs/webgym/docker/patches/instance_server.py"
    )
    source = path.read_text(encoding="utf-8")
    assert "cursor: bool = True" in source
    assert "cursor=cursor" in source


def test_node_server_screenshot_route_forwards_cursor_static():
    path = (
        Path(__file__).resolve().parents[4] / "lite/gym/envs/webgym/docker/patches/node_server.py"
    )
    source = path.read_text(encoding="utf-8")
    assert "cursor: bool = True" in source
    assert "'cursor': '1' if cursor else '0'" in source


def test_master_screenshot_route_forwards_cursor_static():
    path = (
        Path(__file__).resolve().parents[4] / "lite/gym/envs/webgym/docker/patches/master_server.py"
    )
    source = path.read_text(encoding="utf-8")
    assert "cursor: bool = True" in source
    assert '"cursor": "1" if cursor else "0"' in source


def test_webgym_dockerfile_copies_cursor_route_patches_static():
    root = Path(__file__).resolve().parents[4] / "lite/gym/envs/webgym/docker"
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    copies = {
        "master_server.py": "/opt/omniboxes/omniboxes/master/server.py",
        "instance_server.py": "/opt/omniboxes/omniboxes/node/instance_server.py",
    }
    assert "COPY patches/ /tmp/webgym-patches/" in dockerfile
    for patch_name, destination in copies.items():
        assert (root / "patches" / patch_name).is_file()
        assert f"/tmp/webgym-patches/{patch_name}" in dockerfile
        assert destination in dockerfile


def test_controller_seeds_cursor_at_viewport_centre(monkeypatch, tmp_path):
    """Session-start pointer origin = viewport CENTRE, not top-left.

    The retired ``CursorOverlayWrapper`` seeded ``(500, 500)`` normalized (=
    centre) for every env; a real desktop warps the pointer to screen centre at
    session start and the browser inherits it. ``(0, 0)`` is the "never moved"
    sentinel, so a top-left seed made webgym's turn-0 frame disagree with both
    the baseline and its sibling envs.

    Goes through the real ``__init__`` (staged next to a stub ``_page_script.js``,
    which upstream ships inside the container but the repo does not), so the
    assertion covers the shipped constructor rather than a hand-built object.
    """
    staged = tmp_path / "_playwright_controller.py"
    source = (
        Path(__file__).resolve().parents[4]
        / "lite/gym/envs/webgym/docker/patches/_playwright_controller.py"
    )
    staged.write_bytes(source.read_bytes())
    (tmp_path / "_page_script.js").write_text("", encoding="utf-8")

    module = _load_webgym_controller_patch(monkeypatch, path=staged)
    controller = module.PlaywrightController(viewport_width=1280, viewport_height=720)

    assert controller.last_cursor_position == (640.0, 360.0)

    wide = module.PlaywrightController(viewport_width=1920, viewport_height=1080)

    assert wide.last_cursor_position == (960.0, 540.0)


@pytest.mark.asyncio
async def test_controller_updates_last_cursor_position_for_pointer_actions(monkeypatch):
    module = _load_webgym_controller_patch(monkeypatch)

    async def _sleep(_seconds):
        return None

    monkeypatch.setattr(module.asyncio, "sleep", _sleep)
    controller = _controller_instance(module)
    page = _FakePage(module)

    await controller.click_coords(page, 111.0, 222.0)
    assert controller.last_cursor_position == (111.0, 222.0)

    await controller.fill_coords(page, 333.0, 444.0, "query")
    assert controller.last_cursor_position == (333.0, 444.0)

    await controller.click_id(page, "bid-1")
    assert controller.last_cursor_position == (25.0, 40.0)

    await controller.fill_id(page, "bid-1", "query")
    assert controller.last_cursor_position == (25.0, 40.0)

    await controller.hover_id(page, "bid-1")
    assert controller.last_cursor_position == (25.0, 40.0)

    await controller.page_down(page, amount=200)
    assert controller.last_cursor_position == (10, 10)
    await controller.page_up(page, amount=200)
    assert controller.last_cursor_position == (10, 10)


@pytest.mark.asyncio
async def test_controller_animation_cleanup_runs_on_action_errors(monkeypatch):
    module = _load_webgym_controller_patch(monkeypatch)

    async def _sleep(_seconds):
        return None

    monkeypatch.setattr(module.asyncio, "sleep", _sleep)

    click_controller = _controller_instance(module, animate_actions=True)
    click_page = _FakePage(module, fail_click=True)
    click_controller.add_cursor_box = AsyncMock()

    async def _click_gradual(_page, _sx, _sy, ex, ey):
        click_controller._set_last_cursor_position(ex, ey)

    click_controller.gradual_cursor_animation = AsyncMock(side_effect=_click_gradual)
    click_controller._safe_remove_cursor_box = AsyncMock()

    with pytest.raises(RuntimeError, match="click failed"):
        await click_controller.click_id(click_page, "bid-1")
    click_controller._safe_remove_cursor_box.assert_awaited_once_with(click_page, "bid-1")

    hover_controller = _controller_instance(module, animate_actions=True)
    hover_page = _FakePage(module, fail_move=True)
    hover_controller.add_cursor_box = AsyncMock()

    async def _hover_gradual(_page, _sx, _sy, ex, ey):
        hover_controller._set_last_cursor_position(ex, ey)

    hover_controller.gradual_cursor_animation = AsyncMock(side_effect=_hover_gradual)
    hover_controller._safe_remove_cursor_box = AsyncMock()

    with pytest.raises(RuntimeError, match="move failed"):
        await hover_controller.hover_id(hover_page, "bid-1")
    hover_controller._safe_remove_cursor_box.assert_awaited_once_with(hover_page, "bid-1")


@pytest.mark.asyncio
async def test_controller_screenshot_cursor_toggle_composites_shared_sprite(monkeypatch):
    module = _load_webgym_controller_patch(monkeypatch)
    controller = _controller_instance(module)
    controller.viewport_width = 50
    controller.viewport_height = 50
    controller.last_cursor_position = (8.0, 9.0)
    page = _FakePage(module, screenshot_bytes=_TINY_PNG)

    without_cursor = await controller.get_screenshot(page, cursor=False)
    with_cursor = await controller.get_screenshot(page, cursor=True)

    assert without_cursor == _TINY_PNG
    assert with_cursor[:4] == b"\x89PNG"
    assert with_cursor != _TINY_PNG
    _assert_shared_cursor_sprite_at(_TINY_PNG, with_cursor, x=8, y=9)
    assert len(page.screenshot_calls) == 2
    assert page.evaluations == []


# Key-name mapping moved to the canonical chokepoint + shared lite/gym/utils/backend/keys.py
# (canonical Lite key tokens -> Playwright wire key names); its coverage lives in
# tests/gym/utils/backend/test_keys.py.

# ---------------------------------------------------------------------------
# Unit tests: env lifecycle with mocked client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reset_leases_instance():
    """reset() should lease an instance and navigate to the website."""
    env = _make_env()
    env._client = _make_mock_client()

    obs = await env.reset()
    assert obs.image
    raw = obs.image
    assert raw[:4] == b"\x89PNG"

    env._client.get_instance.assert_called_once()
    env._client.execute.assert_called_once()
    call_args = env._client.execute.call_args
    assert "visit_page" in call_args[0][1]

    await env.close()


@pytest.mark.asyncio
async def test_reset_releases_previous_instance():
    """reset() should release the previous instance if one exists."""
    env = _make_env()
    env._client = _make_mock_client()

    await env.reset()
    # Reset again — should release the first instance
    await env.reset()

    env._client.reset_instance.assert_called_once()
    await env.close()


@pytest.mark.asyncio
async def test_metadata():
    """metadata should reflect the task after construction."""
    env = _make_env()
    m = env.metadata
    assert m.platform == "browser"
    assert m.task_type == "use"
    assert m.others["website"] == "imdb.com"
    assert m.others["domain"] == "Entertainment"


@pytest.mark.asyncio
@pytest.mark.parametrize("cursor", [True, False])
async def test_take_screenshot_forwards_cursor_make_kwarg(cursor):
    env = _make_env(cursor=cursor)
    client = _make_mock_client()
    env._client = client
    env._instance = {"instance_id": "test-uuid:9001", "node": "test-node"}

    screenshot = await env._take_screenshot()

    assert screenshot == _TINY_PNG
    client.screenshot.assert_awaited_once_with(
        env._instance,
        mode="coordinates",
        cursor=cursor,
    )


@pytest.mark.asyncio
async def test_metadata_extra_tool_schemas_roundtrip():
    """Selected extra-tools must surface in metadata.extra_tool_schemas AND
    survive the server↔client to_dict/from_dict round-trip. This is the live-boundary
    the offline export path implicitly relies on; a regression here silently strips the
    selected tools from the agent prompt (the symptom seen when a rollout runs without the
    extra_tools env_kwarg). Finish tools are extra schemas, not GUI valid_actions;
    default construction surfaces no standalone extras."""
    env = WebGymEnv(
        task=_SAMPLE_TASK, max_steps=10, skip_eval=True, extra_tools=["goto", "back", "response"]
    )
    names = [tool_schema_name(t) for t in env.metadata.extra_tool_schemas]
    assert names == ["goto", "back", "response"]
    assert all(t["type"] == "function" for t in env.metadata.extra_tool_schemas)

    # server→client round-trip (to_dict/from_dict) must preserve them
    rt = type(env.metadata).from_dict(env.metadata.to_dict())
    rt_names = [tool_schema_name(t) for t in rt.extra_tool_schemas]
    assert rt_names == ["goto", "back", "response"]

    # Default construction surfaces no extra schemas.
    default_env = _make_env()
    default_names = [tool_schema_name(t) for t in default_env.metadata.extra_tool_schemas]
    assert default_names == []
    assert "response" not in default_env.metadata.valid_actions
    assert "terminate" not in default_env.metadata.valid_actions


@pytest.mark.asyncio
async def test_step_click():
    """step() should translate and execute a click action."""
    env = _make_env()
    env._client = _make_mock_client()
    await env.reset()

    r = await env.step(
        [
            make_tool_call("click", {"coordinate": [500, 500]}, call_id="call_click_1"),
        ]
    )
    assert r.results[0].images
    assert not r.terminated
    assert not r.truncated
    assert len(r.results) == 1
    assert r.results[0].tool_call_id == "call_click_1"

    # Should have called execute twice: once for visit_page (reset) + once for click
    assert env._client.execute.call_count == 2
    await env.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"name": "click", "arguments": {"coordinate": [500, 500]}},
        {"function": {"name": "click", "arguments": {"coordinate": [500, 500]}}},
    ],
)
async def test_step_rejects_noncanonical_tool_payloads(payload):
    """step() consumes canonical Lite calls, not provider-bare payloads."""
    env = _make_env()
    env._client = _make_mock_client()
    await env.reset()

    with pytest.raises(TypeError, match="canonical Lite tool calls"):
        await env.step([payload])
    await env.close()


@pytest.mark.asyncio
async def test_malformed_tool_call_with_call_id_returns_current_feedback():
    """Pairable malformed Lite calls are model feedback, not rollout crashes."""
    env = _make_env()
    env._client = _make_mock_client()
    await env.reset()
    env._client.execute.reset_mock()

    r = await env.step(
        [
            {
                "id": "call_bad_click",
                "type": "function",
                "function": {
                    "name": "click",
                    "arguments": ["not", "an", "object"],
                },
            },
        ]
    )

    assert len(r.results) == 1
    _assert_webgym_current_carrier(
        r.results[0],
        call_id="call_bad_click",
        error="invalid tool call: tool_call.function.arguments must be an object, got list",
    )
    assert env._step_count == 1
    env._client.execute.assert_not_called()
    await env.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("duration", [-1, "NaN", "Infinity", 31])
async def test_bad_wait_duration_returns_current_feedback_without_sleep(
    monkeypatch,
    duration,
):
    import lite.gym.envs.webgym.main as webgym_main

    env = _make_env()
    client = _make_mock_client()
    env._client = client
    await env.reset()
    client.execute.reset_mock()

    async def _sleep(_seconds):
        raise AssertionError("bad wait duration reached asyncio.sleep")

    monkeypatch.setattr(webgym_main.asyncio, "sleep", _sleep)

    r = await env.step(
        [
            make_tool_call(
                "wait",
                {"duration": duration},
                call_id="call_wait",
            ),
        ]
    )
    await env.close()

    assert len(r.results) == 1
    _assert_webgym_current_carrier(
        r.results[0],
        call_id="call_wait",
        error=r.results[0].error,
    )
    assert r.results[0].error.startswith("invalid arguments for wait: wait.duration")
    client.execute.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["bash", "foo"])
async def test_unknown_tool_returns_error_only_feedback(name):
    """Literal unknown standalone tools return call_id-paired error-only feedback."""
    env = _make_env()
    env._client = _make_mock_client()
    await env.reset()
    env._client.execute.reset_mock()

    r = await env.step(
        [
            make_tool_call(name, {"command": "pwd"}, call_id=f"call_{name}"),
        ]
    )

    assert len(r.results) == 1
    _assert_error_only_carrier(
        r.results[0],
        call_id=f"call_{name}",
        error=f"unknown tool: {name}",
    )
    assert env._step_count == 1
    assert r.terminated is False
    assert r.truncated is False
    env._client.execute.assert_not_called()
    await env.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("back", {}),
        ("goto", {"url": "https://example.com"}),
        ("response", {"text": "July 21, 2023"}),
        ("terminate", {"status": "success"}),
    ],
)
async def test_inactive_known_extra_returns_current_feedback(name, args):
    """BrowserGym/WebGym keep current page feedback for inactive nav/finish tools."""
    env = _make_env(extra_tools=[])
    env._client = _make_mock_client()
    await env.reset()
    env._client.execute.reset_mock()

    r = await env.step(
        [
            make_tool_call(name, args, call_id=f"call_{name}"),
        ]
    )

    assert len(r.results) == 1
    _assert_webgym_current_carrier(
        r.results[0],
        call_id=f"call_{name}",
        error=f"{name} is not available in this task.",
    )
    assert env._step_count == 1
    assert r.terminated is False
    assert r.truncated is False
    env._client.execute.assert_not_called()
    await env.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "args"),
    [
        ("screenshot", {}),
        ("cursor_position", {}),
        ("mouse_down", {"button": "left"}),
        ("mouse_up", {"button": "left"}),
        ("key_down", {"keys": ["shift"]}),
        ("key_up", {"keys": ["shift"]}),
        ("hold_key", {"keys": ["shift"], "duration": 0.1}),
    ],
)
async def test_unsupported_known_actions_return_current_feedback(action, args):
    """WebGym must not silently drop GUI actions it recognizes but cannot
    execute against OmniBoxes."""
    env = WebGymEnv(
        task=_SAMPLE_TASK,
        max_steps=10,
        skip_eval=True,
        valid_actions=None,
        extra_tools=[],
    )
    env._client = _make_mock_client()
    await env.reset()
    env._client.execute.reset_mock()

    r = await env.step(
        [
            make_tool_call(
                "computer",
                {"actions": [{"action": action, **args}]},
                call_id="call_action",
            ),
        ]
    )

    env._client.execute.assert_not_called()
    assert len(r.results) == 1
    _assert_webgym_current_carrier(
        r.results[0],
        call_id="call_action",
        error=f"unsupported action: {action}",
    )
    assert env._step_count == 1
    await env.close()


@pytest.mark.asyncio
async def test_malformed_drag_returns_current_feedback_without_execution():
    env = WebGymEnv(
        task=_SAMPLE_TASK,
        max_steps=10,
        skip_eval=True,
        valid_actions=None,
    )
    env._client = _make_mock_client()
    await env.reset()
    env._client.execute.reset_mock()

    r = await env.step(
        [
            make_tool_call("drag", {"start_coordinate": [100, 100]}, call_id="call_drag"),
        ]
    )

    env._client.execute.assert_not_called()
    assert len(r.results) == 1
    _assert_webgym_current_carrier(
        r.results[0],
        call_id="call_drag",
        error="invalid arguments for drag: 'coordinate'",
    )
    assert r.results[0].error
    assert r.results[0].error.startswith("invalid arguments for drag:")
    assert "coordinate" in r.results[0].error
    await env.close()


@pytest.mark.asyncio
async def test_successful_step_returns_current_page_text_and_image_carrier():
    env = _make_env()
    env._client = _make_mock_client()
    await env.reset()
    env._client.screenshot.return_value = _OTHER_PNG

    r = await env.step(
        [
            make_tool_call("click", {"coordinate": [500, 500]}, call_id="call_click_carrier"),
        ]
    )

    assert len(r.results) == 1
    _assert_webgym_current_carrier(
        r.results[0],
        call_id="call_click_carrier",
        image=_OTHER_PNG,
    )
    assert "webpage changed" in (r.results[0].text or "")
    assert env._step_count == 1
    assert r.terminated is False
    assert r.truncated is False
    await env.close()


@pytest.mark.asyncio
async def test_action_batch_returns_one_frame_per_executed_action():
    """N executed actions → N frames, in action order, each its own capture.

    Distinctness is asserted, not just the count: re-emitting one cached frame N
    times would satisfy the count while carrying no information at all. The
    per-step evaluator record (``_screenshots``, indexed against ``_actions`` /
    ``_observations``) still takes exactly one frame per step.
    """
    env = _make_env()
    env._client = _make_mock_client()
    await env.reset()

    frames = [_make_test_png(seed=s) for s in (11, 12, 13)]
    env._client.screenshot.side_effect = list(frames)

    r = await env.step(
        [
            make_tool_call(
                "computer",
                {
                    "actions": [
                        {"action": "click", "coordinate": [100, 100]},
                        {"action": "click", "coordinate": [200, 200]},
                        {"action": "click", "coordinate": [300, 300]},
                    ]
                },
                call_id="call_gui",
            ),
        ]
    )

    assert r.results[0].tool_call_id == "call_gui"
    assert r.results[0].images == frames
    assert len(set(r.results[0].images)) == 3
    # Evaluator record: reset's frame + this step's LAST frame, nothing else.
    assert env._screenshots == [_TINY_PNG, frames[-1]]
    assert len(env._actions) == len(env._observations) == len(env._screenshots) - 1
    await env.close()


@pytest.mark.asyncio
async def test_step_with_no_executed_action_still_returns_one_frame():
    """Zero executed actions still owes the model one current observation."""
    env = _make_env()
    env._client = _make_mock_client()
    await env.reset()
    env._client.screenshot.return_value = _OTHER_PNG

    # ``drag`` is outside this env's valid_actions, so it is rejected before
    # anything reaches the browser and the loop captures nothing.
    r = await env.step(
        [
            make_tool_call("drag", {"coordinate": [1, 2]}, call_id="call_bad"),
        ]
    )

    assert r.results[0].images == [_OTHER_PNG]
    assert env._screenshots == [_TINY_PNG, _OTHER_PNG]
    await env.close()


@pytest.mark.asyncio
async def test_step_terminate():
    """terminate action should end the episode."""
    env = _make_env(extra_tools=["terminate"])
    env._client = _make_mock_client()
    await env.reset()

    r = await env.step(
        [
            make_tool_call("terminate", {"status": "success"}, call_id="call_done_1"),
        ]
    )
    assert r.terminated is True
    assert r.reward is not None
    # A terminal call gets NO tool result: it ended the episode, so there is no
    # next decision for an observation to inform, and
    # ``devs/migration/verify.py`` refuses a tool result for a terminal call.
    assert r.results == []
    assert env._step_count == 1
    await env.close()


@pytest.mark.asyncio
async def test_internal_terminate_result_id_gets_current_feedback():
    """Env-private finish calls answer the intercepted model call id."""
    env = _make_env(extra_tools=["terminate"])
    env._client = _make_mock_client()
    await env.reset()

    r = await env.step(
        [
            make_internal_terminate_action(result_call_id="call_intercepted"),
        ]
    )

    assert r.terminated is True
    assert len(r.results) == 1
    _assert_webgym_current_carrier(r.results[0], call_id="call_intercepted")
    await env.close()


@pytest.mark.asyncio
async def test_dead_pool_failfast_truncates_after_consecutive_fallbacks():
    """L2: N CONSECUTIVE fallback screenshots (a dead/unreachable
    pool) must TRUNCATE the trajectory — not grind to max_steps burning a teacher
    turn per step on a dead pool while count/throughput look normal."""
    from lite.gym.envs.webgym.main import _POOL_UNREACHABLE_FALLBACK_STEPS as N

    assert N >= 2
    env = _make_env(max_steps=50)  # high cap so truncation is from L2, not budget
    env._client = _make_mock_client()
    await env.reset()

    async def _always_fallback():
        env._is_fallback_screenshot = True  # every screenshot degrades to stale
        return "STALE_B64"

    env._take_screenshot = _always_fallback

    truncs = []
    for _ in range(N):
        r = await env.step(
            [
                make_tool_call("click", {"coordinate": [10, 10]}),
            ]
        )
        truncs.append(r.truncated)
    # First N-1 steps degrade gracefully; the Nth trips the dead-pool fail-fast.
    assert truncs[:-1] == [False] * (N - 1), truncs
    assert truncs[-1] is True, "should truncate on the Nth consecutive fallback"
    await env.close()


@pytest.mark.asyncio
async def test_fallback_counter_resets_on_real_screenshot():
    """A RECOVERING blip must NOT trip the dead-pool fail-fast: a single real
    screenshot resets the consecutive-fallback counter, so 'N-1 bad, 1 good,
    N-1 bad' never reaches N-in-a-row."""
    from lite.gym.envs.webgym.main import _POOL_UNREACHABLE_FALLBACK_STEPS as N

    env = _make_env(max_steps=50)
    env._client = _make_mock_client()
    await env.reset()

    # good screenshot exactly at index N-1 (between two runs of N-1 fallbacks)
    state = {"i": 0}
    good_at = N - 1

    async def _intermittent():
        i = state["i"]
        state["i"] += 1
        if i == good_at:
            env._is_fallback_screenshot = False
            return "REAL_B64"
        env._is_fallback_screenshot = True
        return "STALE_B64"

    env._take_screenshot = _intermittent

    any_trunc = False
    for _ in range(2 * (N - 1) + 1):
        r = await env.step(
            [
                make_tool_call("click", {"coordinate": [10, 10]}),
            ]
        )
        any_trunc = any_trunc or r.truncated
    assert not any_trunc, "a recovered blip must NOT trip the dead-pool fail-fast"
    await env.close()


@pytest.mark.asyncio
async def test_fallback_screenshot_reports_capture_failure_not_ineffective_action():
    """Two identical consecutive frames mean different things, and the model-visible
    text must say which one happened.

    A fallback re-appends the previous frame, so the frames are identical BY
    CONSTRUCTION and change detection has no evidence: the step must be described
    as an env capture failure. A real capture that happens to match the previous
    frame IS evidence, and only that step may be described as an ineffective
    action. Both steps below end with ``_screenshots[-2] == _screenshots[-1]``.
    """
    env = _make_env(max_steps=10)
    env._client = _make_mock_client()
    await env.reset()

    # Step 1: the env could not capture this step's page. Drive the real owner so
    # the flag and the duplicated frame come from production code.
    real_take_screenshot = env._take_screenshot

    async def _capture_failure():
        return env._fallback_to_previous_screenshot()

    env._take_screenshot = _capture_failure

    fallback = await env.step(
        [
            make_tool_call("click", {"coordinate": [10, 10]}, call_id="call_fallback"),
        ]
    )
    fallback_text = fallback.results[0].text or ""
    assert env._screenshots[-2] == env._screenshots[-1]
    assert "could not capture a new screenshot" in fallback_text
    assert "not effective" not in fallback_text
    assert "did not change" not in fallback_text
    # The capture-failure prose is env feedback keyed to the call it answers, not
    # logger decoration: it rides the result the model reads for THIS call id.
    assert fallback.results[0].tool_call_id == "call_fallback"
    # ...and the step still carries a frame, so logger media keeps a screenshot
    # for the turn. It is the re-appended previous frame by construction.
    assert fallback.results[0].images == [env._screenshots[-1]]

    # Step 2: a real capture that matches the previous frame — the mock client
    # returns the same PNG every call — is genuine change-detection evidence.
    env._take_screenshot = real_take_screenshot
    unchanged = await env.step(
        [
            make_tool_call("click", {"coordinate": [10, 10]}, call_id="call_unchanged"),
        ]
    )
    unchanged_text = unchanged.results[0].text or ""
    assert env._screenshots[-2] == env._screenshots[-1]
    assert "the last action is not effective" in unchanged_text
    assert "could not capture a new screenshot" not in unchanged_text
    assert unchanged.results[0].tool_call_id == "call_unchanged"
    assert unchanged.results[0].images == [env._screenshots[-1]]

    await env.close()


@pytest.mark.asyncio
async def test_step_response():
    """response action should end the episode and store the response."""
    env = _make_env(extra_tools=["response"])
    env._client = _make_mock_client()
    await env.reset()

    r = await env.step(
        [
            make_tool_call("response", {"text": "July 21, 2023"}, call_id="call_done_1"),
        ]
    )
    assert r.terminated is True
    assert env._agent_response == "July 21, 2023"
    # A terminal call gets NO tool result: it ended the episode, so there is no
    # next decision for an observation to inform, and
    # ``devs/migration/verify.py`` refuses a tool result for a terminal call.
    assert r.results == []
    assert env._step_count == 1
    await env.close()


@pytest.mark.asyncio
async def test_truncation_at_max_steps():
    """Episode should truncate after max_steps."""
    env = _make_env(max_steps=2)
    env._client = _make_mock_client()
    await env.reset()

    r1 = await env.step(
        [
            make_tool_call("click", {"coordinate": [500, 500]}),
        ]
    )
    assert not r1.truncated
    assert env._step_count == 1

    r2 = await env.step(
        [
            make_tool_call("click", {"coordinate": [500, 500]}, call_id="call_step_2"),
        ]
    )
    assert r2.truncated
    assert not r2.terminated
    assert r2.reward is not None
    assert len(r2.results) == 1
    _assert_webgym_current_carrier(r2.results[0], call_id="call_step_2")
    assert env._step_count == 2
    await env.close()


@pytest.mark.asyncio
async def test_close_releases_instance():
    """close() should release the instance and close the client."""
    env = _make_env()
    mock_client = _make_mock_client()
    env._client = mock_client
    await env.reset()
    await env.close()

    mock_client.reset_instance.assert_called_once()
    mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_double_close_safe():
    """close() twice should not error."""
    env = _make_env()
    mock_client = _make_mock_client()
    env._client = mock_client
    await env.reset()
    await env.close()
    await env.close()  # Should not raise (client is None now)


@pytest.mark.asyncio
async def test_action_failure_continues():
    """A failed action execution should not crash the episode."""
    env = _make_env()
    env._client = _make_mock_client()
    env._client.execute.side_effect = [
        {"status": "success"},  # visit_page in reset
        Exception("command failed"),  # click in step
    ]
    await env.reset()

    r = await env.step(
        [
            make_tool_call("click", {"coordinate": [500, 500]}),
        ]
    )
    # Should still return a screenshot even though the action failed
    assert r.results[0].images
    await env.close()


@pytest.mark.asyncio
async def test_screenshot_failure_raises_retryable_infra_error():
    """A capture failure is the env owing an observation and not delivering it.

    It must surface as ``TrueInfraFailure`` (retryable) so the trajectory is
    void and re-run, NOT be papered over by re-serving the previous frame. The
    stale frame made the model act on a page it was no longer looking at, which
    read downstream as a repetitive-action loop and scored 0 as a MODEL failure.
    """
    from lite.gym.errors import TrueInfraFailure, is_retryable

    env = _make_env()
    env._client = _make_mock_client()
    env._client.screenshot.side_effect = Exception("screenshot failed")
    try:
        with pytest.raises(TrueInfraFailure) as excinfo:
            await env.reset()
        assert is_retryable(excinfo.value)
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_multiple_actions_per_step():
    """Multiple actions in a single step should all execute."""
    env = _make_env()
    env._client = _make_mock_client()
    await env.reset()

    r = await env.step(
        [
            make_tool_call("click", {"coordinate": [500, 300]}),
            make_tool_call("type", {"text": "hello"}),
            make_tool_call("key", {"keys": ["enter"]}),
        ]
    )
    assert r.results[0].images
    # reset's visit_page + 3 actions = 4 calls
    assert env._client.execute.call_count == 4
    await env.close()


# ---------------------------------------------------------------------------
# Extra tools: back / goto
# ---------------------------------------------------------------------------


class TestExtraToolsTranslation:
    """back/goto are nav extra_tools (opt-in via env_kwargs.extra_tools);
    the env executes them by canonical name. These tests cover that translation
    regardless of whether they're advertised."""

    def _translate(self, name: str, args: dict) -> dict | None:
        env = _make_env()
        env._viewport = (1280, 768)
        return env._translate_action(name, args)

    def test_back(self):
        assert self._translate("back", {}) == {"back": {}}

    def test_goto(self):
        assert self._translate("goto", {"url": "https://example.com"}) == {
            "visit_page": {"url": "https://example.com"},
        }

    def test_default_advertises_no_nav_extra_tools(self):
        # Grounded recipe (default): standalone extras are opt-in.
        env = WebGymEnv(task={"task_id": "t", "difficulty": 2})
        names = [tool_schema_name(t) for t in env.metadata.extra_tool_schemas]
        assert names == []

    def test_extra_tools_opt_in_advertised_as_canonical_schemas(self):
        # Nav/finish are opt-in via env_kwargs.extra_tools (canonical names).
        env = WebGymEnv(
            task={"task_id": "t", "difficulty": 2}, extra_tools=["back", "goto", "response"]
        )
        names = [tool_schema_name(t) for t in env.metadata.extra_tool_schemas]
        assert names == ["back", "goto", "response"]

    def test_unknown_extra_tool_raises(self):
        # A typo'd extra_tool must fail loudly at the config boundary.
        import pytest

        with pytest.raises(ValueError, match="unknown extra_tools"):
            WebGymEnv(task={"task_id": "t", "difficulty": 2}, extra_tools=["nonsense"])

    def test_every_config_env_kwarg_is_wired(self):
        """Systemic guard (mirrors mobilegym): every env_kwargs key in default.yaml
        must be an explicit WebGymEnv.__init__ param, OR a known registration-layer
        key resolved before __init__. WebGymEnv.__init__ has **kwargs that SILENTLY
        swallows unknowns, so an unwired knob (the viewport / goback_* SHAPE-B bugs)
        becomes a no-op override. This converts reviewer-vigilance into a failing test."""
        import inspect

        from lite.gym.envs.webgym.main import CFG, WebGymEnv

        params = set(inspect.signature(WebGymEnv.__init__).parameters)
        # Keys resolved at the module/registration layer, NOT as __init__ params:
        #   max_steps_train/eval → per-difficulty tables collapsed to `max_steps`;
        #   step_timeout → applied by the registry wrapper (wall-clock cap).
        registration_layer = {"max_steps_train", "max_steps_eval", "step_timeout"}
        for key in CFG.env_kwargs:
            assert key in params or key in registration_layer, (
                f"env_kwarg '{key}' is neither a WebGymEnv.__init__ param nor a known "
                f"registration-layer key — a per-run override would vanish into **kwargs."
            )

    def test_default_valid_actions_is_grounded_subset(self):
        env = WebGymEnv(task={"task_id": "t", "difficulty": 2})
        # Grounded GUI interaction verbs only. Nav and terminal tools are not
        # valid_actions.
        assert env.metadata.valid_actions == [
            "click",
            "type",
            "key",
            "scroll",
            "wait",
        ]
        assert "response" not in env.metadata.valid_actions
        assert "terminate" not in env.metadata.valid_actions
        assert "goto" not in env.metadata.valid_actions
        assert "back" not in env.metadata.valid_actions
        extra_names = [tool_schema_name(t) for t in env.metadata.extra_tool_schemas]
        assert extra_names == []

    def test_valid_actions_rejects_standalone_tools(self):
        """Standalone tools belong to ``extra_tools``, never the GUI enum.

        These names used to be dropped SILENTLY, which turned a wrong yaml into
        a quietly smaller action enum (and ``["terminate"]`` alone into ``[]``,
        i.e. no GUI tool at all). They now fail at the config boundary.
        """
        with pytest.raises(ValueError) as excinfo:
            WebGymEnv(
                task={"task_id": "t", "difficulty": 2},
                valid_actions=[
                    "click",
                    "response",
                    "terminate",
                    "goto",
                    "back",
                    "open_app",
                    "ask_user",
                ],
            )
        message = str(excinfo.value)
        for name in ("response", "terminate", "goto", "back", "open_app", "ask_user"):
            assert name in message
        assert "extra_tools" in message

    def test_valid_actions_action_subset_is_verbatim(self):
        env = WebGymEnv(
            task={"task_id": "t", "difficulty": 2},
            valid_actions=["click", "scroll"],
        )
        assert env.metadata.valid_actions == ["click", "scroll"]


def test_container_ensure_publishes_allocated_master_url(monkeypatch):
    """The runtime master URL is the allocated host port, not server_kwargs.master_port."""
    import lite.gym.envs.webgym.main as webgym_main
    import lite.gym.envs.webgym.pool_sizing as pool_sizing
    import lite.gym.utils.backend.ports as port_utils

    calls: dict[str, object] = {}
    monkeypatch.delenv("WEBGYM_MASTER_URL", raising=False)
    monkeypatch.setattr(webgym_main, "_wg_services_started", set())
    monkeypatch.setattr(
        webgym_main,
        "_webgym_docker_rm_f",
        lambda name: calls.setdefault("rm", name),
    )
    monkeypatch.setattr(port_utils, "allocate_ports", lambda **_kw: [7777])
    monkeypatch.setattr(
        pool_sizing,
        "derive_pool_size_from_env",
        lambda: pool_sizing.PoolSize(instances=13, nodes=1),
    )

    def _docker_run(name, image, *, mem, port, env):
        calls["docker_run"] = {
            "name": name,
            "image": image,
            "mem": mem,
            "port": port,
            "env": env,
        }

    monkeypatch.setattr(webgym_main, "docker_run", _docker_run)
    monkeypatch.setattr(
        webgym_main,
        "_webgym_master_capacity_ok",
        lambda url: url == "http://localhost:7777",
    )
    monkeypatch.setattr(
        webgym_main,
        "wait_until_ready",
        lambda predicate, timeout, interval: predicate(),
    )

    webgym_main.WebGymContainerServices().ensure("webgym")

    assert os.environ["WEBGYM_MASTER_URL"] == "http://localhost:7777"
    assert calls["docker_run"]["port"] == (7777, 7000)
    assert calls["docker_run"]["env"]["WEBGYM_API_KEY"] == "test_key"
    assert calls["docker_run"]["env"]["WEBGYM_INSTANCES"] == "13"


def test_container_ensure_ignores_ambient_master_url(monkeypatch):
    """A stale WEBGYM_MASTER_URL must not bypass the scoped fresh container launch."""
    import lite.gym.envs.webgym.main as webgym_main
    import lite.gym.envs.webgym.pool_sizing as pool_sizing
    import lite.gym.utils.backend.ports as port_utils

    calls: dict[str, object] = {}
    monkeypatch.setenv("WEBGYM_MASTER_URL", "http://localhost:7000")
    monkeypatch.setattr(webgym_main, "_wg_services_started", set())
    monkeypatch.setattr(
        webgym_main,
        "_webgym_docker_rm_f",
        lambda name: calls.setdefault("rm", name),
    )
    monkeypatch.setattr(port_utils, "allocate_ports", lambda **_kw: [7777])
    monkeypatch.setattr(
        pool_sizing,
        "derive_pool_size_from_env",
        lambda: pool_sizing.PoolSize(instances=2, nodes=1),
    )

    def _docker_run(name, image, *, mem, port, env):
        calls["docker_run"] = {
            "name": name,
            "image": image,
            "mem": mem,
            "port": port,
            "env": env,
        }

    monkeypatch.setattr(webgym_main, "docker_run", _docker_run)
    monkeypatch.setattr(webgym_main, "_webgym_master_capacity_ok", lambda _url: True)
    monkeypatch.setattr(
        webgym_main,
        "wait_until_ready",
        lambda predicate, timeout, interval: predicate(),
    )

    webgym_main.WebGymContainerServices().ensure("webgym")

    assert calls["docker_run"]["port"] == (7777, 7000)
    assert os.environ["WEBGYM_MASTER_URL"] == "http://localhost:7777"


def test_container_boot_reap_evicts_singleton_caches(monkeypatch):
    import importlib

    import lite.gym.envs.webgym.main as webgym_main

    registry_mod = importlib.import_module("lite.gym.registry")
    env_id = "webgym"
    monkeypatch.setattr(webgym_main, "_wg_services_started", {env_id})
    monkeypatch.setenv("WEBGYM_MASTER_URL", "http://localhost:7777")
    registry_mod._services_started.add(env_id)
    monkeypatch.setattr(
        "lite.gym.remote.reaper.docker_rm_f",
        lambda *args, **kwargs: 1,
    )

    removed = webgym_main.WebGymContainerServices().reap(
        env_id, SimpleNamespace(server_port=1234), set(), boot=True
    )

    assert removed == 1
    assert env_id not in webgym_main._wg_services_started
    assert env_id not in registry_mod._services_started
    assert "WEBGYM_MASTER_URL" not in os.environ


def test_container_health_failure_points_to_webgym_setup(monkeypatch):
    """B8 setup-health guard: a non-serving master is a setup blocker with
    operator-facing README/install guidance, not a silent rollout failure."""
    import lite.gym.envs.webgym.main as webgym_main
    from lite.gym.errors import EnvDepsMissingError

    monkeypatch.setenv("WEBGYM_MASTER_URL", "http://localhost:7777")
    monkeypatch.setattr(webgym_main, "_webgym_master_capacity_ok", lambda _url: False)

    with pytest.raises(EnvDepsMissingError) as excinfo:
        webgym_main.WebGymContainerServices().health("webgym")

    err = excinfo.value
    assert err.what == "webgym container not serving (capacity 0 / unreachable)"
    assert err.install == "uv run --no-sync bash lite/gym/envs/webgym/scripts/install.sh"
    assert err.see == "lite/gym/envs/webgym/README.md"


def test_readme_documents_setup_health_and_lifecycle_smoke():
    """B8 README-backed setup: docs must expose status, env-server, and direct
    create/reset/step/close smoke paths with config-file-owned infra defaults."""
    repo_root = Path(__file__).resolve().parents[4]
    readme = (repo_root / "lite/gym/envs/webgym/README.md").read_text()

    for expected in (
        "scripts/install.sh status",
        "host-side evaluator needs the `webgym` package importable",
        "OPENAI_API_KEY",
        "skip_eval=True",
        "uv run --no-sync python scripts/serve_env.py --env-ids webgym",
        "CUA_LITE_ENV_SERVER_URL",
        "CUA_LITE_ENV_SERVER_TOKEN",
        'gym.make("webgym@<task_id>"',
        "await env.reset()",
        "await env.step(",
        "await env.close()",
        "master URL **auto-allocate**",
        "configs/default.yaml",
        "WEBGYM_CONFIG",
        "cursor_toggle_and_lease_cleanup",
        "pixel-region shared cursor sprite assertion",
        "polls `/info` until the lease is released",
        "tests/gym/remote/test_direct_server_parity_matrix.py",
    ):
        assert expected in readme


def test_pool_sizing_ignores_stale_webgym_instances_env(monkeypatch):
    """WEBGYM_INSTANCES is the internal docker-run value; yaml/config owns sizing."""
    import lite.gym.envs.webgym.pool_sizing as pool_sizing
    from lite.gym.utils import config as env_config
    from lite.gym.utils.server import capacity as capacity_utils

    monkeypatch.setenv("WEBGYM_INSTANCES", "999")
    monkeypatch.setenv("WEBGYM_NODES", "99")
    monkeypatch.setattr(
        env_config,
        "load",
        lambda _env_dir: SimpleNamespace(server_kwargs={"instances": 11, "target_concurrency": 0}),
    )
    monkeypatch.setattr(
        capacity_utils,
        "cached_host_capacity",
        lambda: SimpleNamespace(vcpu=4, ram_total_gb=16.0),
    )

    size = pool_sizing.derive_pool_size_from_env()

    assert size.instances == 11
    assert size.nodes == 1


@pytest.mark.asyncio
async def test_back_executes_via_step():
    """back action should translate and execute through env.step()."""
    env = _make_env(extra_tools=["back"])
    env._client = _make_mock_client()
    await env.reset()

    await env.step([make_tool_call("back")])
    # reset's visit_page + back = 2 execute calls
    assert env._client.execute.call_count == 2
    last_call = env._client.execute.call_args
    assert last_call[0][1] == {"back": {}}
    await env.close()


@pytest.mark.asyncio
async def test_goto_executes_via_step():
    """goto action should translate and execute through env.step()."""
    env = _make_env(extra_tools=["goto"])
    env._client = _make_mock_client()
    await env.reset()

    await env.step(
        [
            make_tool_call("goto", {"url": "https://example.com"}),
        ]
    )
    assert env._client.execute.call_count == 2
    last_call = env._client.execute.call_args
    assert last_call[0][1] == {"visit_page": {"url": "https://example.com"}}
    await env.close()


@pytest.mark.asyncio
async def test_execute_status_error_sets_tool_result_error():
    env = _make_env(valid_actions=["click", "type", "key", "scroll", "wait"])
    client = _make_mock_client()
    env._client = client
    await env.reset()
    client.execute.reset_mock()
    client.execute.return_value = {"status": "error", "message": "backend rejected click"}

    r = await env.step(
        [
            make_tool_call("click", {"coordinate": [500, 500]}, call_id="call_click"),
        ]
    )
    await env.close()

    assert r.results[0].tool_call_id == "call_click"
    assert r.results[0].images[-1] == _TINY_PNG
    assert r.results[0].error == "click failed: execution failed"
    assert "backend rejected click" not in r.results[0].error
    assert r.results[0].metadata == {
        "page_title": "Test Page",
        "url": "https://imdb.com",
        "is_error": True,
    }
    assert env._step_count == 1


@pytest.mark.asyncio
async def test_goto_status_error_sets_tool_result_error():
    env = _make_env(extra_tools=["goto"])
    client = _make_mock_client()
    client._sem_navigate = asyncio.Semaphore(1)
    env._client = client
    await env.reset()
    client.execute.reset_mock()
    client.execute.return_value = {"status": "error", "message": "navigation blocked"}

    r = await env.step(
        [
            make_tool_call("goto", {"url": "https://blocked.example"}, call_id="call_goto"),
        ]
    )
    await env.close()

    assert r.results[0].tool_call_id == "call_goto"
    assert r.results[0].images[-1] == _TINY_PNG
    assert r.results[0].error == "goto failed: navigation blocked"
    assert r.results[0].metadata == {
        "page_title": "Test Page",
        "url": "https://imdb.com",
        "is_error": True,
    }
    assert env._step_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("child", "extra_tools"),
    [
        ({"action": "goto", "url": "https://example.com/nested"}, ["goto"]),
        ({"action": "response", "text": "done"}, ["response"]),
    ],
)
async def test_nested_active_extra_tool_child_obeys_ingress_rejection(
    child: dict[str, str],
    extra_tools: list[str],
):
    env = _make_env(extra_tools=extra_tools)
    client = _make_mock_client()
    env._client = client
    await env.reset()
    client.execute.reset_mock()

    r = await env.step([
        make_tool_call(
            "computer",
            {"actions": [child]},
            call_id="call_batch",
        )
    ])
    await env.close()

    name = child["action"]
    expected = (
        f"invalid action: {name}; "
        f"computer.actions cannot contain {name}"
    )
    client.execute.assert_not_called()
    assert r.terminated is False
    _assert_webgym_current_carrier(
        r.results[0],
        call_id="call_batch",
        error=expected,
    )
    assert r.info[EXECUTED_ACTIONS_INFO_KEY] == [
        {"call": "noop", "args": {"name": name, "reason": expected}}
    ]


@pytest.mark.asyncio
async def test_valid_actions_rejection_keeps_current_observation_feedback():
    env = _make_env(valid_actions=["click", "type", "key", "scroll", "wait"])
    client = _make_mock_client()
    env._client = client

    await env.reset()
    result = await env.step(
        [
            make_tool_call(
                "computer",
                {"actions": [{"action": "mouse_move", "coordinate": [500, 500]}]},
                call_id="call_0",
            )
        ]
    )
    await env.close()

    client.execute.assert_called_once()
    assert client.execute.call_args[0][1] == {"visit_page": {"url": "https://imdb.com"}}
    assert len(result.results) == 1
    _assert_webgym_current_carrier(
        result.results[0],
        call_id="call_0",
        error=("invalid action: mouse_move; choose an available action for this task"),
    )
    assert "invalid action" not in (result.results[0].text or "")
    assert env._step_count == 1


# ---------------------------------------------------------------------------
# Task registration tests
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_webgym_tasks_registered(self):
        """WebGym tasks should be discoverable via registry."""
        import lite.gym as gym

        result = gym.registry.task_ids("webgym")
        if isinstance(result, dict):
            task_ids = [tid for ids in result.values() for tid in ids]
        else:
            task_ids = result
        assert len(task_ids) > 100, f"Expected many WebGym tasks, got {len(task_ids)}"

    @pytest.mark.live  # gym.make brings up the real webgym container (128-instance
    # pool, ~5 min) via ensure() — same infra as TestWebGymLive.
    def test_make_creates_env(self):
        """gym.make() should create a WebGymEnv."""
        import lite.gym as gym
        from lite.gym.errors import EnvDepsMissingError

        result = gym.registry.task_ids("webgym")
        if isinstance(result, dict):
            task_ids = [tid for ids in result.values() for tid in ids]
        else:
            task_ids = result
        first_id = task_ids[0]
        try:
            env = gym.make(f"webgym@{first_id}")
        except EnvDepsMissingError as e:
            # OmniBoxes (the webgym backend) is an optional install; CI / dev
            # venvs without it should not be a hard failure here.
            pytest.skip(f"webgym deps unavailable: {e}")
        assert isinstance(env._env if hasattr(env, "_env") else env, (WebGymEnv, object))
        assert env.metadata.platform == "browser"


# ---------------------------------------------------------------------------
# Live tests (need a real cua-lite/webgym container — brought up by gym.make via
# WebGymContainerServices.ensure). Gated behind the `live` marker (opt in with
# `pytest -m live`), NOT a WEBGYM_MASTER_URL env probe: webgym is container-only,
# so WEBGYM_MASTER_URL is an OUTPUT that `ensure` writes (and the unit fixtures
# above set it too) — it is no longer a "server already running" signal.
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestWebGymLive:
    """Integration tests against a real cua-lite/webgym container (opt in: `pytest -m live`)."""

    def _make_live(
        self,
        max_steps: int = 10,
        extra_tools: list[str] | None = None,
        cursor: bool = True,
        skip_eval: bool = False,
    ) -> WebGymEnv:
        import lite.gym as gym

        result = gym.registry.task_ids("webgym")
        if isinstance(result, dict):
            task_ids = [tid for ids in result.values() for tid in ids]
        else:
            task_ids = result
        kwargs = {"max_steps": max_steps, "cursor": cursor, "skip_eval": skip_eval}
        if extra_tools is not None:
            kwargs["extra_tools"] = extra_tools
        return gym.make(f"webgym@{task_ids[0]}", **kwargs)

    @pytest.mark.asyncio
    async def test_cursor_toggle_and_lease_cleanup(self):
        def _leased_slots(info: dict) -> int:
            if "nodes" in info:
                return sum(
                    int(node.get("capacity", 0)) - int(node.get("available", 0))
                    for node in info.get("nodes", []) or []
                )
            if "in_use" in info:
                return len(info.get("in_use", []) or [])
            return int(info.get("capacity", 0)) - int(info.get("available", 0))

        def _in_use_ids(info: dict) -> set[str]:
            ids = set(info.get("in_use", []) or [])
            for node in info.get("nodes", []) or []:
                ids.update(node.get("in_use", []) or [])
            return ids

        env = self._make_live(max_steps=1, cursor=True, skip_eval=True)
        live_env = _unwrap_webgym_env(env)
        probe: WebGymClient | None = None
        try:
            obs = await env.reset()
            assert obs.image
            assert live_env._client is not None
            assert live_env._instance is not None
            leased_id = live_env._instance["instance_id"]
            info = await live_env._client.info()
            assert leased_id in _in_use_ids(info) or _leased_slots(info) >= 1
            result = await live_env._client.execute(
                live_env._instance,
                {"click_coords": {"x": 100, "y": 100}},
            )
            assert result.get("status") == "success"

            without_cursor_before = await live_env._client.screenshot(
                live_env._instance,
                mode="coordinates",
                cursor=False,
            )
            with_cursor = await live_env._client.screenshot(
                live_env._instance,
                mode="coordinates",
                cursor=True,
            )
            without_cursor_after = await live_env._client.screenshot(
                live_env._instance,
                mode="coordinates",
                cursor=False,
            )
            assert with_cursor[:4] == b"\x89PNG"
            assert without_cursor_before[:4] == b"\x89PNG"
            assert without_cursor_after[:4] == b"\x89PNG"
            assert with_cursor != without_cursor_before
            assert with_cursor != without_cursor_after
            before_score = _assert_shared_cursor_sprite_at(
                without_cursor_before,
                with_cursor,
                x=100,
                y=100,
            )
            after_score = _assert_shared_cursor_sprite_at(
                without_cursor_after,
                with_cursor,
                x=100,
                y=100,
            )
            assert before_score["opaque_close"] >= 35
            assert after_score["opaque_close"] >= 35

            master_url, api_key = live_env._master_url, live_env._api_key
            await env.close()
            probe = WebGymClient(master_url, api_key)
            for _ in range(20):
                info = await probe.info()
                if leased_id not in _in_use_ids(info) and _leased_slots(info) == 0:
                    break
                await asyncio.sleep(0.5)
            else:
                pytest.fail(f"webgym lease was not released: {leased_id}; info={info}")
        finally:
            if probe is not None:
                await probe.close()
            await env.close()
            from lite.gym.envs.webgym.main import WebGymContainerServices
            from lite.gym.remote.scope import ServerScope

            await asyncio.to_thread(WebGymContainerServices().shutdown, "webgym", ServerScope())

    @pytest.mark.asyncio
    async def test_reset_returns_screenshot(self):
        env = self._make_live()
        try:
            obs = await env.reset()
            assert obs.image
            raw = obs.image
            assert raw[:4] == b"\x89PNG"
            assert obs.text
        finally:
            await env.close()

    @pytest.mark.asyncio
    async def test_click_action(self):
        env = self._make_live()
        try:
            await env.reset()
            r = await env.step(
                [
                    make_tool_call("click", {"coordinate": [500, 500]}),
                ]
            )
            assert r.results[0].images
            assert not r.terminated
        finally:
            await env.close()

    @pytest.mark.asyncio
    async def test_type_action(self):
        env = self._make_live()
        try:
            await env.reset()
            r = await env.step(
                [
                    make_tool_call("click", {"coordinate": [500, 300]}),
                    make_tool_call("type", {"text": "test query"}),
                ]
            )
            assert r.results[0].images
        finally:
            await env.close()

    @pytest.mark.asyncio
    async def test_scroll_action(self):
        env = self._make_live()
        try:
            await env.reset()
            r = await env.step(
                [
                    make_tool_call(
                        "scroll",
                        {
                            "coordinate": [500, 500],
                            "direction": "down",
                            "amount": 3,
                        },
                    ),
                ]
            )
            assert r.results[0].images
        finally:
            await env.close()

    @pytest.mark.asyncio
    async def test_back_action(self):
        env = self._make_live(extra_tools=["back"])
        try:
            await env.reset()
            r = await env.step(
                [
                    make_tool_call("back"),
                ]
            )
            assert r.results[0].images
        finally:
            await env.close()

    @pytest.mark.asyncio
    async def test_goto_action(self):
        env = self._make_live(extra_tools=["goto"])
        try:
            await env.reset()
            r = await env.step(
                [
                    make_tool_call("goto", {"url": "https://example.com"}),
                ]
            )
            assert r.results[0].images
        finally:
            await env.close()

    @pytest.mark.asyncio
    async def test_terminate(self):
        env = self._make_live(extra_tools=["terminate"])
        try:
            await env.reset()
            r = await env.step(
                [
                    make_tool_call("terminate", {"status": "success"}),
                ]
            )
            assert r.terminated is True
        finally:
            await env.close()

    @pytest.mark.asyncio
    async def test_response(self):
        env = self._make_live(extra_tools=["response"])
        try:
            await env.reset()
            r = await env.step(
                [
                    make_tool_call("response", {"text": "test answer"}),
                ]
            )
            assert r.terminated is True
        finally:
            await env.close()

    @pytest.mark.asyncio
    async def test_full_episode(self):
        """Run a complete episode: reset → multiple steps → terminate."""
        env = self._make_live(max_steps=5, extra_tools=["response"])
        try:
            obs = await env.reset()
            assert obs.image

            # Click somewhere
            r = await env.step(
                [
                    make_tool_call("click", {"coordinate": [500, 300]}),
                ]
            )
            assert not r.terminated

            # type something
            r = await env.step(
                [
                    make_tool_call("type", {"text": "test"}),
                ]
            )
            assert not r.terminated

            # Scroll
            r = await env.step(
                [
                    make_tool_call(
                        "scroll",
                        {
                            "coordinate": [500, 500],
                            "direction": "down",
                            "amount": 2,
                        },
                    ),
                ]
            )
            assert not r.terminated

            # Terminate
            r = await env.step(
                [
                    make_tool_call("response", {"text": "answer"}),
                ]
            )
            assert r.terminated
        finally:
            await env.close()

    @pytest.mark.asyncio
    async def test_concurrent_instances(self):
        """Multiple env instances should work concurrently."""
        import lite.gym as gym

        result = gym.registry.task_ids("webgym")
        if isinstance(result, dict):
            task_ids = [tid for ids in result.values() for tid in ids]
        else:
            task_ids = result

        async def _run_episode(task_id: str) -> bool:
            env = gym.make(f"webgym@{task_id}", max_steps=3)
            try:
                obs = await env.reset()
                assert obs.image
                r = await env.step(
                    [
                        make_tool_call("click", {"coordinate": [500, 500]}),
                    ]
                )
                assert r.results[0].images
                await env.close()
                return True
            except Exception:
                await env.close()
                raise

        # Run 4 episodes concurrently
        results = await asyncio.gather(
            *[_run_episode(task_ids[i]) for i in range(min(4, len(task_ids)))],
            return_exceptions=True,
        )
        successes = [r for r in results if r is True]
        assert len(successes) >= 3, f"Expected >=3 successes, got {len(successes)}: {results}"


class TestJudgeCompletenessGuard:
    """No silent rewards: the vendored webgym Evaluator swallows a failed sub-call
    (e.g. a 429 rate-limit) and returns a degraded reward=0. _judge_incomplete_reason
    must surface every reward-affecting failure from the structured return value so
    the caller raises (→ ERR → retry) instead of committing a false reward. Guards
    against re-introducing the silent-corruption regression."""

    class _R:
        def __init__(self, judgment):
            self.submission_judgment = judgment

    @staticmethod
    def _traj(judgments):
        # one trajectory step per judgment; the LAST is the force-included frame.
        return [
            {"observation": object(), "reward": TestJudgeCompletenessGuard._R(j)} for j in judgments
        ]

    _CLEAN = ["[Criterion B ...] Verdict: SUCCESS", "[Criterion A - Fact 1] Verdict: SUCCESS"]

    def test_clean_eval_passes(self):
        assert (
            WebGymEnv._judge_incomplete_reason(self._CLEAN, self._traj(["ok", "ok", "last"]))
            is None
        )

    def test_criterion_call_error_detected(self):
        bad = ["[Criterion B - Anti-Hallucination] Error: Error code: 429 - Too Many Requests"]
        assert WebGymEnv._judge_incomplete_reason(bad, self._traj(["ok", "x"])) is not None

    def test_criterion_a_error_detected(self):
        bad = ["[Criterion B ...] Verdict: SUCCESS", "[Criterion A - Fact 2] Error: RateLimitError"]
        assert WebGymEnv._judge_incomplete_reason(bad, self._traj(["ok", "x"])) is not None

    def test_reference_answer_error_detected(self):
        # A reference-answer CALL error appends "[Reference Answer Evaluation] Error:"
        # → caught by the "] Error:" marker.
        bad = self._CLEAN + ["[Reference Answer Evaluation] Error: Error code: 429"]
        assert WebGymEnv._judge_incomplete_reason(bad, self._traj(["ok", "x"])) is not None

    def test_reference_answer_skip_passes(self):
        # The Evaluator legitimately SKIPS the reference block (evaluator.py:320) when
        # a rubric failed, so NO "[Reference Answer]" entry appears — that's not an
        # error and must NOT be flagged (regression guard for the false-positive fix).
        not_all_pass = [
            "[Criterion B ...] Verdict: SUCCESS",
            "[Criterion A - Fact 1] Verdict: NOT SUCCESS",
        ]
        assert WebGymEnv._judge_incomplete_reason(not_all_pass, self._traj(["ok", "x"])) is None

    def test_submission_image_error_detected(self):
        # judge_single_image returns judgment=None on a failed call (non-last step).
        assert (
            WebGymEnv._judge_incomplete_reason(self._CLEAN, self._traj([None, "ok", "last"]))
            is not None
        )

    def test_none_judgment_on_last_step_is_ok(self):
        # The last frame is force-included and never judged, so None there is benign.
        assert (
            WebGymEnv._judge_incomplete_reason(self._CLEAN, self._traj(["ok", "ok", None])) is None
        )


# _batched_judge_submission lazily imports the external `webgym` judge
# package (an install.sh-only dep that `uv sync` evicts); method-level (NOT
# class-level): test_is_content_policy_error_distinguishes_transient in the
# same class is webgym-free and must keep running.
needs_webgym = pytest.mark.skipif(
    importlib.util.find_spec("webgym") is None,
    reason="webgym pkg not installed (webgym install.sh)",
)


class TestBatchedJudgeSubmission:
    """The batched submission filter (gpt-4.1 latency fix): all frames judged in ONE
    multi-image call. Verify per-image parse + that a failed/missing decision leaves
    judgment=None (so _judge_incomplete_reason raises — no silent verdict)."""

    class _FakeClient:
        def __init__(self, resp):
            self._resp = resp

        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        def create(self, **kw):
            class _M:
                def __init__(s, c):
                    s.message = type("m", (), {"content": c})

            return type("R", (), {"choices": [_M(self._resp)]})

    class _FakeEval:
        TASK_KEYPOINT_DETECTION = "keypoint_detection"

        def __init__(self, resp):
            self._c = TestBatchedJudgeSubmission._FakeClient(resp)

        def _get_client_and_model(self, t):
            return self._c, "gpt-4.1"

    @staticmethod
    def _traj(n):
        import tempfile

        from PIL import Image

        tr = []
        for _ in range(n):
            f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            Image.new("RGB", (8, 8)).save(f.name)
            obs = type(
                "O",
                (),
                {
                    "image_path": f.name,
                    "task": type("T", (), {"task_name": "t", "evaluator_reference": ["r1"]})(),
                },
            )()
            tr.append({"observation": obs, "reward": None})
        return tr

    @needs_webgym
    def test_batched_parse_sets_submit_flags(self):
        from lite.gym.envs.webgym.main import _batched_judge_submission

        tr = self._traj(3)  # 2 judged + 1 force-included last
        _batched_judge_submission(self._FakeEval("Image 0: YES\nImage 1: NO"), tr)
        assert tr[0]["reward"].submit is True
        assert tr[1]["reward"].submit is False
        assert tr[2]["reward"].submit is True  # last always included

    @needs_webgym
    def test_missing_decision_leaves_none(self):
        from lite.gym.envs.webgym.main import _batched_judge_submission

        tr = self._traj(3)
        _batched_judge_submission(self._FakeEval("Image 0: YES"), tr)  # no decision for Image 1
        assert tr[1]["reward"].submission_judgment is None
        assert WebGymEnv._judge_incomplete_reason(["[Criterion B] SUCCESS"], tr) is not None

    # --- content-policy handling: deterministic Azure 400 excludes ONE frame, not the eval ---

    class _CPError(Exception):
        code = "content_policy_violation"

    class _FallbackClient:
        """Raises content-policy on the multi-image BATCH call (forcing the per-image
        fallback); on per-image calls returns YES except for ``bad_idx`` which again
        raises content-policy (the genuinely-flagged frame)."""

        def __init__(self, bad_idx):
            self._bad, self._n = bad_idx, 0

        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        def create(self, **kw):
            content = kw["messages"][-1]["content"]
            if sum(1 for c in content if c.get("type") == "image_url") > 1:
                raise TestBatchedJudgeSubmission._CPError(
                    "Error code: 400 content_policy_violation"
                )  # batch
            idx, self._n = self._n, self._n + 1
            if idx == self._bad:
                raise TestBatchedJudgeSubmission._CPError("content_policy_violation")
            return type(
                "R",
                (),
                {
                    "choices": [
                        type("M", (), {"message": type("m", (), {"content": "Decision: YES"})})
                    ]
                },
            )

    class _RaiseClient:
        def __init__(self, exc):
            self._exc = exc

        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        def create(self, **kw):
            raise self._exc

    def test_is_retryable_reads_the_httpx_type_not_the_message(self):
        """Transience is the exception's TYPE, never a substring of its text.

        httpx already classifies its own transport failures (a TLS handshake
        failure is a ``ConnectError``, a peer reset a ``ReadError``), so an
        error whose message merely mentions SSL or a reset — with a type that
        says "your request was wrong" — must not buy three retries.
        """
        import httpx

        from lite.gym.envs.webgym.main import _is_retryable

        request = httpx.Request("POST", "http://x/execute")
        assert _is_retryable(httpx.ReadTimeout("t", request=request)) is True
        assert _is_retryable(httpx.PoolTimeout("t", request=request)) is True
        assert _is_retryable(httpx.ConnectError("ssl handshake", request=request)) is True
        assert _is_retryable(httpx.WriteError("connection reset", request=request)) is True
        assert _is_retryable(httpx.RemoteProtocolError("half-closed", request=request)) is True
        assert (
            _is_retryable(
                httpx.HTTPStatusError(
                    "503",
                    request=request,
                    response=httpx.Response(503, request=request),
                )
            )
            is True
        )

        assert _is_retryable(httpx.LocalProtocolError("we sent garbage")) is False
        assert (
            _is_retryable(
                httpx.HTTPStatusError(
                    "400",
                    request=request,
                    response=httpx.Response(400, request=request),
                )
            )
            is False
        )
        assert _is_retryable(ValueError("ssl: connection reset by peer")) is False

    def test_is_content_policy_error_distinguishes_transient(self):
        from lite.gym.envs.webgym.main import _is_content_policy_error

        assert _is_content_policy_error(self._CPError("x")) is True
        assert (
            _is_content_policy_error(Exception("Error code: 400 - content_policy_violation"))
            is True
        )
        assert _is_content_policy_error(Exception("Error code: 429 - Too Many Requests")) is False

    @needs_webgym
    def test_content_policy_frame_excluded_via_per_image_fallback(self):
        from lite.gym.envs.webgym.main import _batched_judge_submission

        tr = self._traj(3)  # steps 0,1 judged; 2 force-included
        ev = self._FakeEval("unused")
        ev._c = self._FallbackClient(bad_idx=1)  # frame at step 1 is content-flagged
        _batched_judge_submission(ev, tr)
        # offending frame EXCLUDED with a real (non-None) verdict → guard passes
        assert tr[1]["reward"].submit is False
        assert tr[1]["reward"].submission_judgment is not None
        assert "content-filtered" in tr[1]["reward"].submission_judgment
        # the other frame is judged normally; last always included
        assert tr[0]["reward"].submit is True
        assert tr[2]["reward"].submit is True
        # eval completes — content filter is NOT a hard rollout-failure bottleneck
        assert WebGymEnv._judge_incomplete_reason(["[Criterion B] SUCCESS"], tr) is None

    @needs_webgym
    def test_transient_batch_error_still_raises(self):
        # A NON-content-policy batch failure (e.g. 429) must STILL leave judgment=None so
        # the guard raises → trajectory ERR → retry (transient errors are recoverable).
        from lite.gym.envs.webgym.main import _batched_judge_submission

        tr = self._traj(3)
        ev = self._FakeEval("unused")
        ev._c = self._RaiseClient(Exception("Error code: 429 - Too Many Requests"))
        _batched_judge_submission(ev, tr)
        assert tr[0]["reward"].submission_judgment is None
        assert tr[1]["reward"].submission_judgment is None
        assert WebGymEnv._judge_incomplete_reason(["[Criterion B] SUCCESS"], tr) is not None

    # --- chunking: long (eval) trajectories must NOT go in one oversized request ---

    class _ChunkRecordingClient:
        """Records the max images seen in any single call; returns one YES line per image
        for whatever chunk it's given, so every frame parses."""

        def __init__(self):
            self.max_images_per_call = 0
            self.calls = 0

        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        def create(self, **kw):
            imgs = sum(1 for c in kw["messages"][-1]["content"] if c.get("type") == "image_url")
            self.max_images_per_call = max(self.max_images_per_call, imgs)
            self.calls += 1
            resp = "\n".join(f"Image {k}: YES" for k in range(imgs))
            return type(
                "R", (), {"choices": [type("M", (), {"message": type("m", (), {"content": resp})})]}
            )

    @needs_webgym
    def test_long_trajectory_chunks_under_cap_and_judges_all(self):
        # Regression for D1: an eval-length trajectory (>cap frames) must be split into
        # multiple <=cap calls, NOT one oversized request that 400s deterministically.
        from lite.gym.envs.webgym.main import (
            _JUDGE_SUBMISSION_MAX_IMAGES_PER_CALL as CAP,
        )
        from lite.gym.envs.webgym.main import (
            _batched_judge_submission,
        )

        n = CAP + 5  # forces >1 chunk
        tr = self._traj(n)
        ev = self._FakeEval("unused")
        rec = self._ChunkRecordingClient()
        ev._c = rec
        _batched_judge_submission(ev, tr)
        assert rec.max_images_per_call <= CAP  # never exceed the per-call image cap
        assert rec.calls >= 2  # actually chunked
        # EVERY non-last frame judged (non-None) — no silent None that would trip the guard
        for i in range(n - 1):
            assert tr[i]["reward"].submission_judgment is not None
        assert tr[n - 1]["reward"].submit is True  # last force-included
        assert WebGymEnv._judge_incomplete_reason(["[Criterion B] SUCCESS"], tr) is None


# ---------------------------------------------------------------------------
# The ``instance_dead`` chain: a dead browser must be reported, not guessed.
#
# A crashed page/context is PERMANENT for that instance, so the host must
# fail-fast rather than burn every remaining step against a corpse. The fact
# travels as a typed flag over three hops (``playwright_instance.py`` →
# ``node/server.py`` → ``master/server.py``) and the host reads it; it used to
# be re-derived on the host by substring-matching the error prose, which broke
# on any upstream rewording.
#
# The tests below cover the two ends that a host-only suite CAN cover:
#   * hop 3 must not re-flatten the body back into a bare string, and
#   * given the body hop 3 really produces, the host must decide correctly.
# Neither can prove the flag crossed the wire in a live container — only an
# in-container probe can, and the coordinator ran that separately.
# ---------------------------------------------------------------------------

_MASTER_PATCH = (
    Path(__file__).resolve().parents[4] / "lite/gym/envs/webgym/docker/patches/master_server.py"
)

# What ``node/server.py`` forwards verbatim from a dead instance (hop 2's output
# = hop 3's input). Real text, so the fixture cannot quietly become unrealistic.
_NODE_DEAD_BODY = {
    "detail": {
        "message": ("Page.set_viewport_size: Target page, context or browser has been closed"),
        "instance_dead": True,
    }
}
_NODE_ALIVE_BODY = {
    "detail": {
        "message": "Page.click: Timeout 30000ms exceeded waiting for selector '#nope'",
        "instance_dead": False,
    }
}


class _FakeUpstreamResponse:
    """Stand-in for the ``requests.Response`` the master gets from a node.

    Only ``.text`` / ``.json()`` are touched by ``_execute_error_body``, and
    ``.json()`` really parses ``.text`` (raising ``ValueError`` on non-JSON, as
    requests does) so the test does not get to define away the parse.
    """

    def __init__(self, text: str) -> None:
        self.text = text

    def json(self):
        import json

        return json.loads(self.text)


def _load_master_execute_error_body():
    """Exec the REAL ``_execute_error_body`` out of the master patch source.

    The patch imports fastapi/requests/omniboxes, none of which exist on the
    host, so the two top-level nodes it needs are lifted out by AST and executed
    alone — same technique as ``_coerce_cursor_bool`` above. This keeps the test
    bound to the shipped source instead of a host-side copy of its logic.
    """
    import ast

    tree = ast.parse(_MASTER_PATCH.read_text(encoding="utf-8"))
    wanted = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "_INSTANCE_DEAD_KEY" for t in node.targets)
        )
        or (isinstance(node, ast.FunctionDef) and node.name == "_execute_error_body")
    ]
    assert len(wanted) == 2, (
        "master patch must still define _INSTANCE_DEAD_KEY and _execute_error_body; "
        f"found {[getattr(n, 'name', 'assign') for n in wanted]}"
    )
    module = ast.Module(body=wanted, type_ignores=[])
    ast.fix_missing_locations(module)
    ns: dict = {}
    exec(compile(module, str(_MASTER_PATCH), "exec"), ns)
    assert ns["_INSTANCE_DEAD_KEY"] == "instance_dead", "wire key renamed"
    return ns["_execute_error_body"]


def test_master_execute_error_body_never_flattens_instance_dead():
    """Hop 3 must keep ``instance_dead`` a TOP-LEVEL key of its error body.

    Regression guard for the exact upstream shape this patch replaced —
    ``{"status": "error", "message": f"...: {response.text}"}`` — which threw the
    typed fact away and left the host pattern-matching prose. A future edit that
    collapses the body back to message-only fails here on the missing key, and one
    that keeps the key but stops reading the node's nested ``detail`` fails on the
    dead case.
    """
    import json

    error_body = _load_master_execute_error_body()

    dead = error_body(_FakeUpstreamResponse(json.dumps(_NODE_DEAD_BODY)))
    assert dead["instance_dead"] is True
    assert dead["status"] == "error"
    # The prose survives for logs, but it is no longer what the host reads.
    assert "has been closed" in dead["message"]

    alive = error_body(_FakeUpstreamResponse(json.dumps(_NODE_ALIVE_BODY)))
    assert "instance_dead" in alive, "the key must be present even when False"
    assert alive["instance_dead"] is False

    # A future producer that stops nesting under FastAPI's ``detail`` envelope.
    top = error_body(_FakeUpstreamResponse(json.dumps({"instance_dead": True})))
    assert top["instance_dead"] is True

    # Non-JSON (proxy / uvicorn HTML error page): key present, "not known dead".
    opaque = error_body(_FakeUpstreamResponse("<html>502 Bad Gateway</html>"))
    assert opaque["instance_dead"] is False


def _master_500(body: dict):
    """An httpx transport that answers every ``/execute`` with a 500 *body*."""
    import httpx

    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if isinstance(body, dict):
            return httpx.Response(500, json=body)
        return httpx.Response(500, text=body)

    return httpx.MockTransport(handler), calls


async def _execute_against(monkeypatch, body):
    """Drive the real ``WebGymClient.execute`` retry loop against a 500 *body*."""
    import httpx

    from lite.gym.envs.webgym import main as webgym_main

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(webgym_main.asyncio, "sleep", _no_sleep)

    transport, calls = _master_500(body)
    client = WebGymClient("http://master.invalid", "k")
    await client._client.aclose()
    client._client = httpx.AsyncClient(base_url="http://master.invalid", transport=transport)
    try:
        result = await client.execute({"instance_id": "i", "node": "n"}, {"click_coords": {}})
    finally:
        await client._client.aclose()
    return result, len(calls)


@pytest.mark.asyncio
async def test_execute_fails_fast_on_the_masters_instance_dead_flag(monkeypatch):
    """The host reads the flag hop 3 actually sends — one attempt, then stop.

    The 500 body is BUILT BY the master patch's own ``_execute_error_body``, so
    this cannot pass against a body shape the master no longer produces.
    """
    import json

    error_body = _load_master_execute_error_body()
    body = error_body(_FakeUpstreamResponse(json.dumps(_NODE_DEAD_BODY)))

    result, attempts = await _execute_against(monkeypatch, body)

    assert result["instance_dead"] is True
    assert result["status"] == "error"
    assert attempts == 1, "a dead browser must not be retried"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        # Same env transport shape as the dead case, flag False: the DISCRIMINATING
        # leg — a predicate that answered True unconditionally would still pass the test
        # above, and this one catches it.
        "flag_false",
        # Unparseable body: "unknown" must fall to the CONSERVATIVE side (retry a
        # possibly-alive instance), never to a fabricated death that ends the
        # episode. This is also the leg that fails if anyone reintroduces a
        # substring-match fallback keyed on the prose.
        "non_json",
        # JSON, but no flag at all (a 5xx from something other than the master).
        "no_flag",
    ],
)
async def test_execute_retries_when_the_master_does_not_report_death(monkeypatch, case):
    import json

    from lite.gym.envs.webgym.main import _MAX_RETRIES

    if case == "flag_false":
        error_body = _load_master_execute_error_body()
        body = error_body(_FakeUpstreamResponse(json.dumps(_NODE_ALIVE_BODY)))
        assert body["instance_dead"] is False  # the fixture is the shape we think
    elif case == "non_json":
        body = "Target page, context or browser has been closed"
    else:
        body = {"status": "error", "message": "boom"}

    result, attempts = await _execute_against(monkeypatch, body)

    assert not result.get("instance_dead"), f"{case}: must not be reported dead"
    assert attempts == _MAX_RETRIES, f"{case}: must use the ordinary retry path"
