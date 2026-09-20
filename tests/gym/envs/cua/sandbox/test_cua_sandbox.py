"""Unit tests for cua.sandbox (CuaSandboxEnv) — no real `cua-sandbox` install needed.

Runs via a fake `cua_sandbox` module injected into sys.modules, so it exercises the
action translation + reset/step/close lifecycle without a cloud sandbox.
Live end-to-end coverage lives in test_cua_sandbox_live.py (pytest.mark.live).

    uv run pytest tests/gym/envs/cua/sandbox/test_cua_sandbox.py
"""

from __future__ import annotations

import io
import sys
import types

import pytest
from PIL import Image

from lite.core import LiteCUAMetadata
from lite.core.messages.final import make_no_tool_call_final_actions
from lite.core.tools import make_tool_call
from lite.core.tools.schemas import tool_schema_name
from lite.gym.envs.cua.sandbox import CuaSandboxEnv
from lite.gym.errors import (
    EnvDepsMissingError,
    FailureCategory,
    TrueInfraFailure,
    failure_category,
)

_CUA_SANDBOX_RELEASE_LIKE_EXTRA_TOOLS = [
    pytest.param("gpt", [], id="gpt-release-extra-tools-empty"),
    pytest.param("qwen3_5", ["terminate"], id="qwen3_5-release-extra-tools-terminate"),
    pytest.param("qwen3_vl", ["terminate"], id="qwen3_vl-release-extra-tools-terminate"),
]


def _real_png(w: int = 1000, h: int = 1000, color: tuple = (255, 255, 255)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


# A real, decodable PNG matching the fake sandbox's get_dimensions() (1000×1000),
# so cursor compositing (default-on for cua.sandbox) has a valid frame.
_PNG = _real_png()


# --- fake `cua` module ------------------------------------------------------
class _FakeMouse:
    def __init__(self):
        self.calls: list[tuple] = []

    async def click(self, x, y, button="left"):
        self.calls.append(("click", x, y, button))

    async def double_click(self, x, y):
        self.calls.append(("double_click", x, y))

    async def right_click(self, x, y):
        self.calls.append(("right_click", x, y))

    async def move(self, x, y):
        self.calls.append(("move", x, y))

    async def drag(self, sx, sy, ex, ey, button="left"):
        self.calls.append(("drag", sx, sy, ex, ey))

    async def scroll(self, x, y, scroll_x=0, scroll_y=0):
        self.calls.append(("scroll", x, y, scroll_x, scroll_y))

    async def mouse_down(self, x, y, button="left"):
        self.calls.append(("mouse_down", x, y, button))

    async def mouse_up(self, x, y, button="left"):
        self.calls.append(("mouse_up", x, y, button))


class _FakeKeyboard:
    def __init__(self):
        self.calls: list[tuple] = []

    async def type(self, text):
        self.calls.append(("type", text))

    async def keypress(self, keys):
        self.calls.append(("keypress", keys))

    async def key_down(self, key):
        self.calls.append(("key_down", key))

    async def key_up(self, key):
        self.calls.append(("key_up", key))


class _FakeMobile:
    def __init__(self):
        self.calls: list[tuple] = []

    async def tap(self, x, y):
        self.calls.append(("tap", x, y))

    async def double_tap(self, x, y):
        self.calls.append(("double_tap", x, y))

    async def long_press(self, x, y, duration_ms=1000):
        self.calls.append(("long_press", x, y, duration_ms))

    async def type_text(self, text):
        self.calls.append(("type_text", text))

    async def swipe(self, x1, y1, x2, y2, duration_ms=300):
        self.calls.append(("swipe", x1, y1, x2, y2))

    async def pinch_in(self, cx, cy, spread=300, duration_ms=400):
        self.calls.append(("pinch_in", cx, cy))

    async def pinch_out(self, cx, cy, spread=300, duration_ms=400):
        self.calls.append(("pinch_out", cx, cy))

    async def home(self):
        self.calls.append(("home",))

    async def back(self):
        self.calls.append(("back",))

    async def recents(self):
        self.calls.append(("recents",))

    async def key(self, keycode):
        self.calls.append(("key", keycode))


class _FakeSb:
    def __init__(self):
        self.mouse = _FakeMouse()
        self.keyboard = _FakeKeyboard()
        self.mobile = _FakeMobile()

    async def get_dimensions(self):
        return (1000, 1000)

    async def screenshot(self):
        return _PNG


def _mouse_calls_after_reset(env):
    """Desktop mouse calls with reset()'s pointer-parking move stripped.

    reset() issues a real ``mouse.move`` to screen centre so the composited
    cursor coordinate is a FACT rather than an assumption; asserting it here
    keeps the per-action assertions about the action, and fails loudly if the
    parking move is ever dropped.
    """
    calls = env._sb.mouse.calls
    assert calls[:1] == [("move", 500, 500)], (
        f"reset() must park the real pointer at screen centre; got {calls[:1]}"
    )
    return calls[1:]


class _FakeCM:
    def __init__(self, sb):
        self.sb = sb
        self.entered = self.exited = False

    async def __aenter__(self):
        self.entered = True
        return self.sb

    async def __aexit__(self, *exc):
        self.exited = True


class _FakeSandbox:
    last_cm: _FakeCM | None = None
    last_kwargs: dict = {}

    @staticmethod
    def ephemeral(image, **kwargs):
        _FakeSandbox.last_kwargs = {"image": image, **kwargs}
        cm = _FakeCM(_FakeSb())
        _FakeSandbox.last_cm = cm
        return cm


class _FakeImage:
    @staticmethod
    def linux(**kw):
        return ("linux", kw)

    @staticmethod
    def macos(**kw):
        return ("macos", kw)

    @staticmethod
    def windows(**kw):
        return ("windows", kw)

    @staticmethod
    def android(**kw):
        return ("android", kw)


@pytest.fixture
def fake_cua(monkeypatch):
    mod = types.ModuleType("cua_sandbox")
    mod.Sandbox = _FakeSandbox
    mod.Image = _FakeImage
    monkeypatch.setitem(sys.modules, "cua_sandbox", mod)
    _FakeSandbox.last_cm = None
    return mod


def _act(name, **args):
    return {"name": name, "arguments": args}


def _action_call(wrapper, *actions, call_id: str | None = None):
    """One canonical top-level GUI batch call carrying ``actions``.

    cua.sandbox is never ``gym.make``-d, so its own step() enforces the
    top-level surface: bare actions are NOT a legal model emission, the
    ``computer``/``mobile`` wrapper is.
    """
    return make_tool_call(
        wrapper,
        {
            "actions": [{"action": p["name"], **p["arguments"]} for p in actions],
        },
        call_id=call_id,
    )


def _assert_current_image_result(result, *, call_id: str, image: bytes) -> None:
    assert result.results[0].tool_call_id == call_id
    assert result.results[0].images[-1] == image
    assert result.results[0].text is None


def _assert_error_only_result(result, *, call_id: str, error: str) -> None:
    assert result.results[0].tool_call_id == call_id
    assert result.results[0].images == []
    assert result.results[0].text is None
    assert result.results[0].error == error
    assert result.results[0].metadata == {"is_error": True}


def _assert_cursor_at(png: bytes, x: int, y: int) -> None:
    img = Image.open(io.BytesIO(png)).convert("RGB")
    w, h = img.size
    box = [
        img.getpixel((px, py)) for px in range(x, min(x + 16, w)) for py in range(y, min(y + 24, h))
    ]
    assert any(pixel != (255, 255, 255) for pixel in box)
    assert img.getpixel((0, 0)) == (255, 255, 255)


# --- metadata (pure, no cua) ------------------------------------------------
def test_metadata_platform_desktop():
    assert CuaSandboxEnv("hi").metadata.platform == LiteCUAMetadata.Platform.DESKTOP


def test_metadata_platform_mobile():
    assert (
        CuaSandboxEnv("hi", platform="android").metadata.platform == LiteCUAMetadata.Platform.MOBILE
    )


def test_default_config_exposes_terminate():
    names = {
        tool_schema_name(schema)
        for schema in CuaSandboxEnv("hi").metadata.extra_tool_schemas
    }
    assert names == {"terminate"}


def test_direct_sandbox_is_not_a_registered_backend_family():
    """cua.sandbox is a direct wrapper, not a gym-registered env-server family."""
    from lite.gym.services import EnvServerPoolable, EnvServerResource, family_of

    env = CuaSandboxEnv("hi")

    assert not isinstance(env, EnvServerPoolable)
    assert not isinstance(env, EnvServerResource)
    assert family_of("cua.sandbox") is None


def test_accepts_seed_but_rejects_typos():
    # the standard factory/rollout.py passes `seed` to EVERY env; must construct
    # without TypeError (regression: CLI rollout was broken on `seed`). A typo'd
    # env_kwarg must still raise (strict, like cua.bench + browsergym — no **kwargs).
    env = CuaSandboxEnv("hi", seed=42)
    assert env.metadata.platform == LiteCUAMetadata.Platform.DESKTOP
    with pytest.raises(TypeError):
        CuaSandboxEnv("hi", max_step=3)  # typo → not silently swallowed


# --- deps missing (real: cua not installed) ---------------------------------
async def test_reset_raises_when_cua_missing(monkeypatch):
    # simulate the SDK not installed (None in sys.modules → `import cua_sandbox` raises
    # ImportError), so the test is robust regardless of what's in the venv.
    monkeypatch.setitem(sys.modules, "cua_sandbox", None)
    with pytest.raises(EnvDepsMissingError):
        await CuaSandboxEnv("hi").reset()


# --- lifecycle with fake cua ------------------------------------------------
async def test_reset_returns_instruction_and_screenshot(fake_cua):
    # cursor=False → the sandbox frame is passed through as raw bytes.
    env = CuaSandboxEnv("do the thing", cursor=False)
    obs = await env.reset()
    assert obs.text == "do the thing"  # reset text == instruction
    assert obs.image == _PNG
    # linux default → container image
    assert _FakeSandbox.last_kwargs["image"] == ("linux", {"kind": "container"})
    await env.close()


async def test_cursor_composites_at_tracked_pixel(fake_cua):
    # cursor defaults ON for cua.sandbox (config make_kwargs) — the Cua
    # frame has no host cursor, so the env composites one at its tracked pixel
    # position. reset() starts the cursor at screen-center (500, 500) on the
    # 1000×1000 fake; the returned frame must be a valid PNG that DIFFERS from the
    # raw white frame, with a non-white pixel at the cursor tip.
    env = CuaSandboxEnv("x")
    assert env._cursor is True  # sourced from configs/default.yaml
    obs = await env.reset()
    out = obs.image
    assert out != _PNG  # cursor was composited in
    img = Image.open(io.BytesIO(out)).convert("RGB")
    assert img.size == (1000, 1000)  # frame geometry preserved
    # The 16×24 sprite is pasted with its top-left at the cursor (500, 500); some
    # pixel in that box must be non-white (the arrow), while a far corner stays white.
    box = [img.getpixel((x, y)) for x in range(500, 516) for y in range(500, 524)]
    assert any(px != (255, 255, 255) for px in box)
    assert img.getpixel((0, 0)) == (255, 255, 255)
    await env.close()


async def test_step_click_maps_normalized_to_pixels(fake_cua):
    env = CuaSandboxEnv("x", post_action_delay=0)  # no settle sleep in tests
    await env.reset()  # 1000x1000 fake screen
    r = await env.step([_action_call("computer", _act("click", coordinate=[500, 250]))])
    assert _mouse_calls_after_reset(env) == [("click", 500, 250, "left")]
    assert r.reward is None and not r.terminated


async def test_scroll_updates_cursor_and_returned_overlay(fake_cua):
    env = CuaSandboxEnv("x", post_action_delay=0)
    await env.reset()

    r = await env.step(
        [
            _action_call(
                "computer",
                _act("scroll", coordinate=[250, 750], direction="down", amount=4),
                call_id="scroll0",
            ),
        ]
    )

    assert _mouse_calls_after_reset(env) == [("scroll", 250, 750, 0, -4)]
    assert env._cursor_px == (250, 750)
    assert r.results[0].tool_call_id == "scroll0"
    assert r.results[0].images[-1] != _PNG
    assert r.results[0].error is None
    _assert_cursor_at(r.results[0].images[-1], 250, 750)


@pytest.mark.parametrize("agent_family,extra_tools", _CUA_SANDBOX_RELEASE_LIKE_EXTRA_TOOLS)
async def test_sandbox_t2_release_like_active_known_tool(fake_cua, agent_family, extra_tools):
    env = CuaSandboxEnv("x", extra_tools=extra_tools, post_action_delay=0, cursor=False)
    await env.reset()

    call_id = f"{agent_family}_active_known_tool"
    r = await env.step(
        [
            _action_call("computer", _act("click", coordinate=[500, 250]), call_id=call_id),
        ]
    )

    assert env._sb.mouse.calls == [("click", 500, 250, "left")]
    assert not r.terminated and not r.truncated
    _assert_current_image_result(r, call_id=call_id, image=_PNG)
    assert r.results[0].error is None
    assert r.results[0].metadata is None


@pytest.mark.parametrize("agent_family,extra_tools", _CUA_SANDBOX_RELEASE_LIKE_EXTRA_TOOLS)
async def test_sandbox_t2_release_like_malformed_known_action(fake_cua, agent_family, extra_tools):
    env = CuaSandboxEnv("x", extra_tools=extra_tools, post_action_delay=0, cursor=False)
    await env.reset()

    call_id = f"{agent_family}_malformed_known_action"
    r = await env.step(
        [
            _action_call("computer", _act("click", coordinate=[None, None]), call_id=call_id),
        ]
    )

    assert env._sb.mouse.calls == []
    assert not r.terminated and not r.truncated
    _assert_current_image_result(r, call_id=call_id, image=_PNG)
    assert r.results[0].error == (
        "invalid arguments for click: coordinate values must be finite numbers"
    )
    assert r.results[0].metadata == {"is_error": True}


@pytest.mark.parametrize("agent_family,extra_tools", _CUA_SANDBOX_RELEASE_LIKE_EXTRA_TOOLS)
async def test_sandbox_t2_release_like_literal_unknown_tool(fake_cua, agent_family, extra_tools):
    env = CuaSandboxEnv("x", extra_tools=extra_tools, post_action_delay=0, cursor=False)
    await env.reset()

    call_id = f"{agent_family}_literal_unknown_tool"
    r = await env.step(
        [
            make_tool_call("foo", {}, call_id=call_id),
        ]
    )

    assert env._sb.mouse.calls == [] and env._sb.keyboard.calls == []
    assert not r.terminated and not r.truncated
    _assert_error_only_result(r, call_id=call_id, error="unknown tool: foo")


@pytest.mark.parametrize("agent_family,extra_tools", _CUA_SANDBOX_RELEASE_LIKE_EXTRA_TOOLS)
async def test_sandbox_t2_release_like_content_only_final_text(fake_cua, agent_family, extra_tools):
    env = CuaSandboxEnv("x", extra_tools=extra_tools, post_action_delay=0, cursor=False)
    await env.reset()

    r = await env.step(make_no_tool_call_final_actions(f"{agent_family} final text"))

    assert r.terminated and not r.truncated
    assert r.reward is None
    assert r.results == []
    assert env._sb.mouse.calls == [] and env._sb.keyboard.calls == []


@pytest.mark.parametrize("agent_family,extra_tools", _CUA_SANDBOX_RELEASE_LIKE_EXTRA_TOOLS)
async def test_sandbox_t2_release_like_image_data_binding(fake_cua, agent_family, extra_tools):
    env = CuaSandboxEnv("x", extra_tools=extra_tools, post_action_delay=0, cursor=False)
    reset_obs = await env.reset()
    bound_png = _real_png(1000, 1000, color=(240, 250, 255))

    async def screenshot_bound_frame():
        return bound_png

    env._sb.screenshot = screenshot_bound_frame
    call_id = f"{agent_family}_image_data_binding"
    r = await env.step(
        [
            _action_call("computer", _act("click", coordinate=[500, 250]), call_id=call_id),
        ]
    )

    assert env._sb.mouse.calls == [("click", 500, 250, "left")]
    assert bound_png != reset_obs.image
    assert not r.terminated and not r.truncated
    _assert_current_image_result(r, call_id=call_id, image=bound_png)
    assert r.results[0].error is None
    assert r.results[0].metadata is None


async def test_action_batch_returns_one_distinct_frame_per_action(fake_cua):
    """N executed actions → N frames, in action order, none a repeat of another.

    Distinct bytes are the point: repeating one cached frame N times would
    satisfy the count while carrying no new information. The third action is
    ``screenshot``: read-only actions get a frame too, so the frame count never
    depends on what the actions were.
    """
    env = CuaSandboxEnv("x", post_action_delay=0, cursor=False)
    await env.reset()
    frames = iter([b"frame-1", b"frame-2", b"frame-3"])

    async def next_frame():
        return next(frames)

    env._sb.screenshot = next_frame

    r = await env.step(
        [
            _action_call(
                "computer",
                _act("click", coordinate=[100, 100]),
                _act("type", text="hi"),
                _act("screenshot"),
                call_id="call_batch",
            ),
        ]
    )

    assert r.results[0].tool_call_id == "call_batch"
    assert r.results[0].images == [b"frame-1", b"frame-2", b"frame-3"]
    assert r.results[0].error is None


async def test_a_rejected_child_earns_a_frame_and_costs_no_sibling(fake_cua):
    """A slot the model got wrong owes a frame IN SLOT ORDER, and spares its siblings.

    ``bogus`` is not a name the batch tool carries. Ingress rejects it and
    forwards it anyway, so this env answers it per slot: one frame repeating the
    screen it did not change, plus a model-visible reason. The rejected slot sits
    BETWEEN two valid ones here, which is what pins the ordering -- a rejection
    filtered out before the frame loop would emit the two executed frames first
    and silently mis-attribute them.
    """
    env = CuaSandboxEnv("x", post_action_delay=0, cursor=False)
    await env.reset()
    frames = iter([b"frame-1", b"rejected-frame", b"frame-3"])

    async def next_frame():
        return next(frames)

    env._sb.screenshot = next_frame

    r = await env.step(
        [
            _action_call(
                "computer",
                _act("click", coordinate=[100, 100]),
                _act("bogus"),
                _act("type", text="hi"),
                call_id="call_batch",
            ),
        ]
    )

    assert r.results[0].images == [b"frame-1", b"rejected-frame", b"frame-3"]
    assert "bogus" in r.results[0].error


async def test_dispatch_backend_failure_raises_true_infra_failure(fake_cua):
    env = CuaSandboxEnv("x", post_action_delay=0, cursor=False)
    await env.reset()

    async def fail_dispatch(_action):
        raise RuntimeError("sandbox dispatch failed")

    env._dispatch = fail_dispatch
    with pytest.raises(
        TrueInfraFailure,
        match="cua.sandbox click backend failed: sandbox dispatch failed",
    ) as exc_info:
        await env.step(
            [
                _action_call(
                    "computer", _act("click", coordinate=[500, 250]), call_id="call_click"
                ),
            ]
        )

    assert failure_category(exc_info.value) is FailureCategory.INFRA_FAILURE


# --- unknown-action fallback: derive, then RAISE (never a silent no-op) ------
@pytest.mark.parametrize(
    ("platform", "name"),
    [
        ("linux", "bogus"),  # not in any vocabulary
        ("linux", "tap"),  # a real catalog action, wrong platform
        ("linux", "open_app"),  # a standalone extra, never a Sandbox call
        ("android", "click"),  # desktop action on the mobile ladder
        ("android", "mouse_move"),
    ],
)
async def test_dispatch_raises_on_action_outside_the_platform_vocabulary(
    fake_cua,
    platform,
    name,
):
    """An action name the ladder has no branch for must RAISE, not fall through
    to ``return False``.

    Regression for the unknown-action fallback: the ladders used to end with a
    bare ``return False``, so an unhandled name incremented the step, took a
    screenshot and continued — indistinguishable from a click that missed.
    Ingress (``prepare_env_tool_calls(validate_top_level_action=True)``) admits only
    this platform's catalog actions, so anything arriving here is a dispatcher
    bug, i.e. INFRA_FAILURE — NOT a model action error.
    """
    env = CuaSandboxEnv("x", platform=platform, post_action_delay=0, cursor=False)
    await env.reset()

    with pytest.raises(TrueInfraFailure, match=f"no branch for {name!r}") as exc_info:
        await env._dispatch(_act(name))

    assert failure_category(exc_info.value) is FailureCategory.INFRA_FAILURE
    # not a ValueError/TypeError family member → step() must NOT downgrade it
    # into per-call model feedback via MODEL_ACTION_ERROR_TYPES.
    assert not isinstance(exc_info.value, (ValueError, TypeError, IndexError, KeyError))
    await env.close()


@pytest.mark.parametrize(
    ("platform", "actions"),
    [
        ("linux", "desktop"),
        ("android", "mobile"),
    ],
)
async def test_dispatch_covers_every_catalog_action_for_its_platform(
    fake_cua,
    platform,
    actions,
):
    """The complement of the raise: every name the catalog admits for this
    platform is either executed or a DECLARED read-only no-op — so the new
    fallthrough can never fire on a legitimate model emission."""
    from lite.core.tools.action_space import LiteDesktopActionSet, LiteMobileActionSet

    names = (
        LiteDesktopActionSet.get_action_names()
        if actions == "desktop"
        else LiteMobileActionSet.get_action_names()
    )
    args_for = {
        "click": {"coordinate": [500, 500]},
        "tap": {"coordinate": [500, 500]},
        "long_press": {"coordinate": [500, 500], "duration": 0},
        "pinch": {"coordinate": [500, 500]},
        "swipe": {"start_coordinate": [1, 1], "coordinate": [9, 9]},
        "drag": {"start_coordinate": [1, 1], "coordinate": [9, 9]},
        "mouse_move": {"coordinate": [500, 500]},
        "type": {"text": "hi"},
        "key": {"keys": ["a"]},
        "key_down": {"keys": ["a"]},
        "key_up": {"keys": ["a"]},
        "hold_key": {"keys": ["a"], "duration": 0},
        "wait": {"duration": 0},
        "system_button": {"button": "Home"},
    }
    env = CuaSandboxEnv("x", platform=platform, post_action_delay=0, cursor=False)
    await env.reset()

    for name in sorted(names):
        assert await env._dispatch(_act(name, **args_for.get(name, {}))) is False

    await env.close()


async def test_malformed_coordinate_returns_error_with_current_screenshot(fake_cua):
    env = CuaSandboxEnv("x", post_action_delay=0, cursor=False)
    await env.reset()

    r = await env.step(
        [
            _action_call("computer", _act("click", coordinate=[None, None]), call_id="bad_click"),
        ]
    )

    assert env._sb.mouse.calls == []
    assert not r.terminated and not r.truncated
    assert r.results[0].tool_call_id == "bad_click"
    assert r.results[0].metadata == {"is_error": True}
    assert r.results[0].error == (
        "invalid arguments for click: coordinate values must be finite numbers"
    )
    assert r.results[0].images[-1] == _PNG
    assert r.results[0].text is None


@pytest.mark.parametrize(
    ("action", "expected_error"),
    [
        (
            _act("wait", duration=31),
            "invalid arguments for wait: wait.duration must be <= 30",
        ),
        (
            _act("hold_key", keys=["a"], duration=6),
            "invalid arguments for hold_key: hold_key.duration must be <= 5",
        ),
    ],
)
async def test_bad_desktop_duration_returns_error_with_current_screenshot(
    fake_cua,
    action,
    expected_error,
):
    env = CuaSandboxEnv("x", post_action_delay=0, cursor=False)
    await env.reset()

    r = await env.step(
        [
            _action_call("computer", action, call_id="bad_duration"),
        ]
    )

    assert env._sb.mouse.calls == []
    assert env._sb.keyboard.calls == []
    assert not r.terminated and not r.truncated
    assert r.results[0].tool_call_id == "bad_duration"
    assert r.results[0].metadata == {"is_error": True}
    assert r.results[0].error == expected_error
    assert r.results[0].images[-1] == _PNG
    assert r.results[0].text is None


async def test_unpairable_malformed_call_raises_before_fake_tool_feedback(fake_cua):
    env = CuaSandboxEnv("x", post_action_delay=0, cursor=False)
    await env.reset()

    with pytest.raises(TypeError, match="canonical Lite tool calls"):
        await env.step(
            [
                {"name": "computer", "arguments": ["not", "an", "object"]},
            ]
        )

    assert env._sb.mouse.calls == [] and env._sb.keyboard.calls == []


async def test_step_variants_and_keyboard(fake_cua):
    env = CuaSandboxEnv("x", post_action_delay=0)  # no settle sleep in tests
    await env.reset()
    await env.step(
        [
            _action_call(
                "computer",
                _act("click", coordinate=[100, 100], clicks=2),
                _act("click", coordinate=[200, 200], button="right"),
                _act("type", text="hello"),
                _act("key", keys=["ctrl", "c"]),
                _act("scroll", coordinate=[500, 500], direction="up", amount=5),
            )
        ]
    )
    assert ("double_click", 100, 100) in env._sb.mouse.calls
    assert ("right_click", 200, 200) in env._sb.mouse.calls
    assert ("scroll", 500, 500, 0, 5) in env._sb.mouse.calls  # up → scroll_y=+5, scroll_x=0
    assert ("type", "hello") in env._sb.keyboard.calls
    assert ("keypress", ["ctrl", "c"]) in env._sb.keyboard.calls


async def test_step_projects_canonical_key_glyphs(fake_cua):
    env = CuaSandboxEnv("x", post_action_delay=0, cursor=False)
    await env.reset()

    r = await env.step(
        [
            _action_call(
                "computer",
                _act("key", keys=["ctrl", "+"]),
                _act("key_down", keys=["-"]),
                _act("key_up", keys=["="]),
                _act("hold_key", keys=[","], duration=0),
                call_id="glyph_keys",
            )
        ]
    )

    assert r.results[0].error is None
    assert ("keypress", ["ctrl", "+"]) in env._sb.keyboard.calls
    assert ("key_down", "-") in env._sb.keyboard.calls
    assert ("key_up", "=") in env._sb.keyboard.calls
    assert ("key_down", ",") in env._sb.keyboard.calls
    assert ("key_up", ",") in env._sb.keyboard.calls


@pytest.mark.parametrize(
    ("keys", "expected_error"),
    [
        (["ctrl", "plus"], "invalid arguments for key: unknown key token 'plus'"),
        ([" "], "invalid arguments for key: unknown key token ' '"),
    ],
)
async def test_step_rejects_noncanonical_key_token_before_keyboard_dispatch(
    fake_cua,
    keys,
    expected_error,
):
    env = CuaSandboxEnv("x", post_action_delay=0, cursor=False)
    await env.reset()

    r = await env.step(
        [
            _action_call(
                "computer",
                _act("key", keys=keys),
                call_id="bad_key",
            )
        ]
    )

    assert env._sb.keyboard.calls == []
    assert not r.terminated and not r.truncated
    assert r.results[0].tool_call_id == "bad_key"
    assert r.results[0].metadata == {"is_error": True}
    assert r.results[0].error == expected_error


async def test_bare_top_level_action_is_rejected_not_dispatched(fake_cua):
    """cua.sandbox step() owns the top-level surface contract.

    A bare ``click`` is not on that surface (the action space is wrapper-native),
    so it must be rejected — not silently dispatched because it happens to look
    like an unpacked ``computer.actions[]`` child.
    """
    env = CuaSandboxEnv("x", post_action_delay=0, cursor=False)
    await env.reset()
    r = await env.step(
        [
            make_tool_call("click", {"coordinate": [500, 250]}, call_id="c0"),
        ]
    )
    assert env._sb.mouse.calls == []  # nothing reached the sandbox
    assert r.results[0].tool_call_id == "c0"
    assert r.results[0].text is None
    assert r.results[0].images[-1] == _PNG
    assert r.results[0].error == ("invalid action: click; choose an available action for this task")
    assert r.results[0].metadata == {"is_error": True}


async def test_wrong_platform_action_wrapper_is_rejected(fake_cua):
    """The desktop surface exposes ``computer`` only — a ``mobile`` batch is off-surface."""
    env = CuaSandboxEnv("x", post_action_delay=0, cursor=False)
    await env.reset()
    r = await env.step(
        [
            _action_call("mobile", _act("tap", coordinate=[100, 200]), call_id="m0"),
        ]
    )
    assert env._sb.mobile.calls == [] and env._sb.mouse.calls == []
    assert r.results[0].text is None
    assert r.results[0].images[-1] == _PNG
    assert r.results[0].error == (
        "invalid action: mobile; choose an available action for this task"
    )
    assert r.results[0].metadata == {"is_error": True}


async def test_unknown_extra_tool_is_not_dispatched(fake_cua):
    """A standalone tool absent from this env's declared surface is unknown."""
    env = CuaSandboxEnv("x", extra_tools=[], post_action_delay=0)
    await env.reset()
    r = await env.step(
        [
            make_tool_call("bash", {"command": "echo hi"}, call_id="b0"),
        ]
    )
    assert not r.terminated
    assert r.results[0].text is None
    assert r.results[0].images == []
    assert r.results[0].error == "unknown tool: bash"
    assert r.results[0].metadata == {"is_error": True}


async def test_inactive_finish_tool_returns_error_only_feedback(fake_cua):
    env = CuaSandboxEnv("x", extra_tools=[], post_action_delay=0, cursor=False)
    await env.reset()
    r = await env.step(
        [
            make_tool_call("response", {"text": "Done."}, call_id="r0"),
        ]
    )
    assert not r.terminated
    assert r.results[0].tool_call_id == "r0"
    assert r.results[0].text is None
    assert r.results[0].images == []
    assert r.results[0].error == "response is not available in this task."
    assert r.results[0].metadata == {"is_error": True}


async def test_terminate_ends_episode(fake_cua):
    env = CuaSandboxEnv("x", extra_tools=["terminate"], post_action_delay=0)
    await env.reset()
    r = await env.step([make_tool_call("terminate", {"status": "success"})])
    assert r.terminated is True and r.reward is None
    # terminate is NOT forwarded to the sandbox
    assert _mouse_calls_after_reset(env) == [] and env._sb.keyboard.calls == []


async def test_default_terminate_ends_episode(fake_cua):
    env = CuaSandboxEnv("x", post_action_delay=0)
    await env.reset()
    r = await env.step([make_tool_call("terminate", {"status": "success"})])
    assert r.terminated is True and r.reward is None
    assert _mouse_calls_after_reset(env) == [] and env._sb.keyboard.calls == []


async def test_terminate_with_call_id_returns_no_tool_result(fake_cua):
    env = CuaSandboxEnv("x", extra_tools=["terminate"], post_action_delay=0, cursor=False)
    await env.reset()
    r = await env.step(
        [
            make_tool_call("terminate", {"status": "success"}, call_id="term0"),
        ]
    )
    assert r.terminated is True and not r.truncated
    # A terminal call gets NO tool result: it ended the episode, so there is no
    # next decision for an observation to inform, and
    # ``devs/migration/verify.py`` refuses a tool result for a terminal call.
    # Same shape as the unstamped internal ``response`` above.
    assert r.results == []


async def test_response_with_call_id_returns_no_tool_result(fake_cua):
    env = CuaSandboxEnv("x", extra_tools=["response"], post_action_delay=0, cursor=False)
    await env.reset()
    r = await env.step(
        [
            make_tool_call("response", {"text": "Done."}, call_id="resp0"),
        ]
    )
    assert r.terminated is True and not r.truncated
    # A terminal call gets NO tool result: it ended the episode, so there is no
    # next decision for an observation to inform, and
    # ``devs/migration/verify.py`` refuses a tool result for a terminal call.
    # Same shape as the unstamped internal ``response`` above.
    assert r.results == []


async def test_terminal_action_stops_later_actions_in_same_step(fake_cua):
    env = CuaSandboxEnv("x", extra_tools=["terminate"], post_action_delay=0)
    await env.reset()
    r = await env.step(
        [
            make_tool_call("terminate", {"status": "success"}),
            _action_call("computer", _act("click", coordinate=[100, 100])),
        ]
    )

    assert r.terminated is True
    assert _mouse_calls_after_reset(env) == []


async def test_android_dispatches_mobile_actions(fake_cua):
    env = CuaSandboxEnv(
        "x", platform="android", post_action_delay=0
    )  # MOBILE action space; no settle in tests
    await env.reset()  # 1000x1000 fake screen
    await env.step(
        [
            _action_call(
                "mobile",
                _act("tap", coordinate=[100, 200]),
                _act("tap", coordinate=[100, 200], clicks=2),
                _act("long_press", coordinate=[300, 400], duration=2),
                _act("type", text="hi"),
                _act("swipe", start_coordinate=[0, 0], coordinate=[500, 500]),
                _act("pinch", coordinate=[500, 500], direction="in"),
                _act("system_button", button="Back"),
            )
        ]
    )
    c = env._sb.mobile.calls
    assert ("tap", 100, 200) in c and ("double_tap", 100, 200) in c
    assert ("long_press", 300, 400, 2000) in c  # duration 2s → 2000ms
    assert ("type_text", "hi") in c
    assert ("swipe", 0, 0, 500, 500) in c
    assert ("pinch_in", 500, 500) in c
    assert ("back",) in c
    # desktop interfaces untouched on android
    assert env._sb.mouse.calls == [] and env._sb.keyboard.calls == []


async def test_android_cursor_tracks_mobile_actions_and_overlays_post_step(fake_cua):
    env = CuaSandboxEnv("x", platform="android", post_action_delay=0)
    reset_obs = await env.reset()
    # Android is a touch surface: there is no pointer to park at reset, so
    # nothing may be composited until a touch establishes a real contact point.
    # Painting an arrow at screen centre here would be pure invention.
    assert reset_obs.image == _PNG
    assert env._sb.mouse.calls == []  # no pointer to park on android

    r = await env.step(
        [
            _action_call(
                "mobile",
                _act("tap", coordinate=[100, 200]),
                _act("long_press", coordinate=[300, 400], duration=1),
                _act("swipe", start_coordinate=[0, 0], coordinate=[700, 800]),
                _act("drag", start_coordinate=[700, 800], coordinate=[200, 300]),
                _act("pinch", coordinate=[500, 600], direction="out"),
                call_id="mobile0",
            ),
        ]
    )

    assert env._sb.mobile.calls == [
        ("tap", 100, 200),
        ("long_press", 300, 400, 1000),
        ("swipe", 0, 0, 700, 800),
        ("swipe", 700, 800, 200, 300),
        ("pinch_out", 500, 600),
    ]
    assert env._cursor_px == (500, 600)
    assert r.results[0].tool_call_id == "mobile0"
    assert r.results[0].images[-1] != _PNG
    assert r.results[0].error is None
    _assert_cursor_at(r.results[0].images[-1], 500, 600)


async def test_key_and_mouse_hold_actions(fake_cua):
    env = CuaSandboxEnv("x", post_action_delay=0)  # no settle sleep in tests
    await env.reset()  # 1000x1000 fake screen → cursor starts at (500, 500)
    await env.step(
        [
            _action_call(
                "computer",
                _act("key_down", keys=["ctrl", "shift"]),
                _act("key_up", keys=["ctrl", "shift"]),
                _act("hold_key", keys=["a"], duration=0),
                _act("mouse_move", coordinate=[300, 400]),  # cursor → (300, 400)
                _act("mouse_down", button="left"),
                _act("mouse_up", button="left"),
            )
        ]
    )
    kc, mc = env._sb.keyboard.calls, env._sb.mouse.calls
    assert ("key_down", "ctrl") in kc and ("key_down", "shift") in kc
    assert kc.index(("key_up", "shift")) < kc.index(("key_up", "ctrl"))  # released in reverse
    assert ("key_down", "a") in kc and ("key_up", "a") in kc  # hold_key press+release
    # mouse_down/up act at the current cursor (Lite carries no coordinate)
    assert ("mouse_down", 300, 400, "left") in mc and ("mouse_up", 300, 400, "left") in mc


async def test_drag_without_start_uses_cursor(fake_cua):
    env = CuaSandboxEnv("x", post_action_delay=0)  # no settle sleep in tests
    await env.reset()  # 1000x1000 fake screen → cursor starts at (500, 500)
    await env.step(
        [
            _action_call(
                "computer",
                _act("mouse_move", coordinate=[300, 400]),  # cursor → (300, 400)
                _act("drag", coordinate=[700, 800]),  # no start → from cursor
                _act("drag", start_coordinate=[100, 100], coordinate=[200, 200]),
            )
        ]
    )
    mc = env._sb.mouse.calls
    assert ("drag", 300, 400, 700, 800) in mc  # start-less drag begins at tracked cursor
    assert ("drag", 100, 100, 200, 200) in mc  # explicit start still honored


async def test_scroll_directions_map_to_correct_axis(fake_cua):
    env = CuaSandboxEnv("x", post_action_delay=0)  # no settle sleep in tests
    await env.reset()
    await env.step(
        [
            _action_call(
                "computer",
                _act("scroll", coordinate=[100, 100], direction="up", amount=4),
                _act("scroll", coordinate=[100, 100], direction="down", amount=4),
                _act("scroll", coordinate=[100, 100], direction="left", amount=4),
                _act("scroll", coordinate=[100, 100], direction="right", amount=4),
            )
        ]
    )
    calls = env._sb.mouse.calls
    assert ("scroll", 100, 100, 0, 4) in calls  # up    → scroll_y=+4
    assert ("scroll", 100, 100, 0, -4) in calls  # down  → scroll_y=-4
    assert ("scroll", 100, 100, -4, 0) in calls  # left  → scroll_x=-4 (horizontal, not vertical)
    assert ("scroll", 100, 100, 4, 0) in calls  # right → scroll_x=+4


async def test_max_steps_truncates(fake_cua):
    env = CuaSandboxEnv("x", max_steps=2, post_action_delay=0)  # no settle sleep in tests
    await env.reset()
    r1 = await env.step([_action_call("computer", _act("wait", duration=0))])
    assert not r1.truncated and not r1.terminated  # step 1 of 2
    r2 = await env.step([_action_call("computer", _act("wait", duration=0))])
    assert r2.truncated and not r2.terminated  # hit max_steps → truncated
    await env.close()


async def test_max_steps_truncation_returns_paired_screenshot(fake_cua):
    env = CuaSandboxEnv("x", max_steps=1, post_action_delay=0, cursor=False)
    await env.reset()
    r = await env.step(
        [
            _action_call("computer", _act("wait", duration=0), call_id="wait0"),
        ]
    )
    assert r.truncated and not r.terminated
    assert r.results[0].tool_call_id == "wait0"
    assert r.results[0].images[-1] == _PNG
    assert r.results[0].text is None
    assert r.results[0].error is None
    await env.close()


async def test_terminate_beats_truncation(fake_cua):
    # terminate on the max_steps-th step → terminated, NOT truncated
    env = CuaSandboxEnv(
        "x",
        max_steps=1,
        extra_tools=["terminate"],
        post_action_delay=0,
    )
    await env.reset()
    r = await env.step([make_tool_call("terminate", {"status": "success"})])
    assert r.terminated and not r.truncated
    await env.close()


async def test_close_tears_down_sandbox(fake_cua):
    env = CuaSandboxEnv("x")
    await env.reset()
    cm = _FakeSandbox.last_cm
    await env.close()
    assert cm.exited is True and env._sb is None


async def test_platform_selects_image_and_kwargs(fake_cua):
    env = CuaSandboxEnv(
        "x",
        platform="windows",
        image_kwargs={"version": "11"},
        sandbox_kwargs={"region": "us-west-2"},
    )
    await env.reset()
    assert _FakeSandbox.last_kwargs["image"] == ("windows", {"version": "11"})
    assert _FakeSandbox.last_kwargs["region"] == "us-west-2"
    await env.close()
