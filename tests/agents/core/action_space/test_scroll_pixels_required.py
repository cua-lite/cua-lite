"""A wrapper ``scroll`` with no ``pixels`` must ERROR, never default to "up".

THE DEFECT. Every wrapper family that spells a scroll as
``computer_use(action="scroll"|"hscroll", pixels=N)`` read it as
``int(float(args.get("pixels", 0)))`` and then ``direction = "down" if
scroll_pixels < 0 else "up"``. ``pixels`` is OPTIONAL in all five schemas
(``required`` is ``["action"]``), and ``0 < 0`` is false — so a scroll that
carried no ``pixels`` at all converted, *successfully*, to
``direction="up", amount=1``. Conversion success is not conversion correctness:
the turn produced a canonical action, so every conversion-rate metric scored it
green while the agent scrolled the wrong way with the wrong magnitude.

THE TRIGGER. ``webharbor.webvoyager`` is the only env in the tree that declares a
standalone ``scroll(down: bool, pages: number)`` extra tool
(``lite/gym/envs/webharbor/webvoyager/main.py``). If a model emits that env
tool's ``down``/``pages`` shape through a shared provider-native wrapper, the
shared wrapper must reject it instead of defaulting to a scroll-up action.

THE FIX and why this mechanism. ``lite.agents.core.action_space.utils
.required_scroll_pixels`` raises ``ValueError`` — the mechanism the family
converters ALREADY use to refuse a call they cannot interpret
(``FaraDesktopActionSpace._convert_single_from_agent`` raises for a wire-level
``response``), and a member of
``lite.gym.utils.feedback.errors.MODEL_ACTION_ERROR_TYPES``. It never escapes:
``BaseAgent.predict`` catches it
into ``model_output_error`` and the sampling loop treats the zero-call turn as a
terminal N3 response. ``test_missing_scroll_pixels_becomes_terminal_response_
not_wrong_scroll`` pins that end to end.

Hermetic: dict assertions on action spaces / adapters, plus one fake-env loop.
No model, no network, no real env.

Run:
    env -u CUA_LITE_ENV_SERVER_URL uv run pytest \
        tests/agents/core/action_space/test_scroll_pixels_required.py \
        -p no:cacheprovider -q
"""

from __future__ import annotations

import io
from typing import Any

import pytest
from PIL import Image

from lite.agents.core.action_space.utils.geometry import (
    compact_number,
    optional_coord,
    required_scroll_pixels,
)
from lite.agents.core.agent.base import AdapterBasedAgent
from lite.agents.models.evocua.action_space import EvoCUADesktopActionSpace
from lite.agents.models.fara.action_space import FaraDesktopActionSpace
from lite.agents.models.qwen2_5_vl.action_space import Qwen2_5VLDesktopActionSpace
from lite.agents.models.qwen3_5.action_space import Qwen3_5DesktopActionSpace
from lite.agents.models.qwen3_5.adapter import Qwen3_5DesktopUseAdapter
from lite.agents.models.qwen3_vl.action_space import Qwen3VLDesktopActionSpace
from lite.core import LiteCUAMetadata, LiteToolCall
from lite.core.tools.calls import tool_call_arguments, tool_call_id, tool_call_name
from lite.core.tools.results import LiteToolResult
from lite.core.tools.schemas import tool_schema_parameters
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult

# The five defect sites, one per (family, wrapper action). ``qwen3_5`` inherits
# its converter from ``qwen3_vl``; both are listed so the two do not drift.
_SCROLL_SPACES = [
    pytest.param(Qwen3VLDesktopActionSpace, id="qwen3_vl"),
    pytest.param(Qwen3_5DesktopActionSpace, id="qwen3_5"),
    pytest.param(Qwen2_5VLDesktopActionSpace, id="qwen2_5_vl"),
    pytest.param(FaraDesktopActionSpace, id="fara"),
    pytest.param(EvoCUADesktopActionSpace, id="evocua"),
]
# ``hscroll`` exists only in the Qwen3-VL / Qwen3.5 enum (verified by
# ``test_hscroll_is_only_in_the_qwen3_enums`` below).
_HSCROLL_SPACES = [
    pytest.param(Qwen3VLDesktopActionSpace, id="qwen3_vl"),
    pytest.param(Qwen3_5DesktopActionSpace, id="qwen3_5"),
]


def _convert(space_cls, args: dict[str, Any]) -> list[LiteToolCall]:
    return space_cls().convert_tool_calls_from_agent(
        [{"name": "computer_use", "arguments": args}]
    )


def _actions(calls: list[LiteToolCall]) -> list[dict[str, Any]]:
    assert len(calls) == 1 and tool_call_name(calls[0]) == "computer", calls
    return tool_call_arguments(calls[0])["actions"]


# ---------------------------------------------------------------------------
# The signal that must NOT regress: a real ``pixels`` still decides direction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("space_cls", _SCROLL_SPACES)
def test_negative_pixels_stay_down(space_cls) -> None:
    """The half of the contract the fix must not break. ``pixels=-300`` is the
    ordinary well-formed emission (1 513 of them in the corpus) and its sign is
    the direction: negative = down. A fix that made every scroll error, or that
    flipped the sign, would pass a "no longer scrolls up" assertion — so the
    correct-input case is pinned first."""
    assert _actions(_convert(space_cls, {"action": "scroll", "pixels": -300})) == [
        {"action": "scroll", "direction": "down", "amount": 3}
    ]


@pytest.mark.parametrize("space_cls", _SCROLL_SPACES)
def test_positive_pixels_stay_up(space_cls) -> None:
    """The mirror case: positive = up, magnitude preserved."""
    assert _actions(_convert(space_cls, {"action": "scroll", "pixels": 300})) == [
        {"action": "scroll", "direction": "up", "amount": 3}
    ]


@pytest.mark.parametrize("space_cls", _SCROLL_SPACES)
def test_string_float_pixels_still_parse(space_cls) -> None:
    """``pixels`` arrives as a string on the XML wire (``"-5.0"``). The
    ``int(float(...))`` coercion the sites carried must survive the refactor into
    the shared helper — otherwise a plain ``int("-5.0")`` would raise ValueError
    and turn 1 513 good scrolls into errors."""
    assert _actions(_convert(space_cls, {"action": "scroll", "pixels": "-5.0"})) == [
        {"action": "scroll", "direction": "down", "amount": 5}
    ]


# ---------------------------------------------------------------------------
# The defect: ``pixels`` absent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("space_cls", _SCROLL_SPACES)
def test_absent_pixels_is_an_error_not_a_default_up(space_cls) -> None:
    """PRE-FIX this returned
    ``[{"action": "scroll", "direction": "up", "amount": 1}]`` — a scroll the
    model never asked for, in the direction it is least likely to have wanted,
    reported as a successful conversion. It must raise instead."""
    with pytest.raises(ValueError, match="requires the 'pixels' argument"):
        _convert(space_cls, {"action": "scroll"})


@pytest.mark.parametrize("space_cls", _SCROLL_SPACES)
@pytest.mark.parametrize(
    "args",
    [
        pytest.param({"action": "scroll", "down": "true", "pages": "0.5"}, id="down=true"),
        pytest.param({"action": "scroll", "down": "false", "pages": "1"}, id="down=false"),
    ],
)
def test_the_webvoyager_down_pages_shape_is_an_error(space_cls, args: dict) -> None:
    """The env-specific ``down``/``pages`` shape must not be accepted by the
    shared wrapper action spaces.

    PRE-FIX both spellings — ``down=true`` AND ``down=false`` — converted to the
    SAME ``direction="up", amount=1``: the model's only direction bit was
    dropped on the floor. That is the tell that a default, not a parse, was
    producing the answer.

    Deliberately an ERROR and not a ``down``/``pages`` branch in the wrapper:
    ``scroll(down, pages)`` is ONE env's standalone-tool signature
    (``webharbor.webvoyager``, the only env in the tree that declares it), and
    teaching a SHARED GUI action space to speak it would bake a single env's
    wire format into all five families."""
    with pytest.raises(ValueError, match="requires the 'pixels' argument"):
        _convert(space_cls, args)


@pytest.mark.parametrize("space_cls", _HSCROLL_SPACES)
def test_absent_pixels_is_an_error_for_hscroll_too(space_cls) -> None:
    """``hscroll`` carried the identical default at
    ``qwen3_vl/action_space.py:448``, one axis over: absent ``pixels`` meant
    ``direction="right", amount=1``. Zero emissions measured (the corpus has no
    ``hscroll`` at all), but it is advertised in the Qwen3-VL / Qwen3.5 action
    enum — so it is reachable, and it must not be left as the one surviving
    copy of the pattern for the next family to inherit."""
    assert _actions(_convert(space_cls, {"action": "hscroll", "pixels": -300})) == [
        {"action": "scroll", "direction": "left", "amount": 3}
    ]
    with pytest.raises(ValueError, match="requires the 'pixels' argument"):
        _convert(space_cls, {"action": "hscroll"})


def test_missing_scroll_pixels_does_not_repr_other_huge_arguments() -> None:
    with pytest.raises(ValueError, match="requires the 'pixels' argument"):
        required_scroll_pixels({"down": 10 ** 5000}, "scroll")


@pytest.mark.parametrize("space_cls", _SCROLL_SPACES)
def test_non_numeric_pixels_still_raises_a_model_action_error(space_cls) -> None:
    """``pixels="soon"`` already raised (bare ``float()``); it must keep raising
    a ``ValueError`` — i.e. a member of ``MODEL_ACTION_ERROR_TYPES`` — rather
    than being "helpfully" defaulted by the same fix."""
    with pytest.raises(ValueError, match="numeric 'pixels'"):
        _convert(space_cls, {"action": "scroll", "pixels": "soon"})


@pytest.mark.parametrize("bad", ["1e400", "-1e400", "inf", "-inf"])
def test_nonfinite_scroll_pixels_raise_model_action_error(bad: str) -> None:
    """Overflowing float literals are ordinary model text, not exotic JSON.

    They must surface as the same ``ValueError`` class as any other malformed
    ``pixels`` value, never as a bare ``OverflowError`` past the model-action
    boundary.
    """
    with pytest.raises(ValueError, match="numeric 'pixels'"):
        required_scroll_pixels({"pixels": bad}, "scroll")


def test_huge_scroll_pixel_int_raises_model_action_error() -> None:
    with pytest.raises(ValueError, match="numeric 'pixels'"):
        required_scroll_pixels({"pixels": 10 ** 5000}, "scroll")


@pytest.mark.parametrize("bad", ["1e400", "-1e400", "inf", "-inf", "nan"])
def test_nonfinite_coordinates_are_rejected_without_overflow(bad: str) -> None:
    assert optional_coord([bad, 0]) is None
    assert optional_coord(f"[{bad},0]") is None


@pytest.mark.parametrize("bad", ["1e400", "-1e400", "inf", "-inf", "nan"])
def test_compact_number_leaves_nonfinite_values_raw(bad: str) -> None:
    assert compact_number(bad) == bad


def test_hscroll_is_only_in_the_qwen3_enums() -> None:
    """Justifies the split between ``_SCROLL_SPACES`` and ``_HSCROLL_SPACES``:
    there are FIVE defect sites, not ten, because only two families advertise a
    horizontal scroll at all."""
    have_hscroll = {
        cls.__name__
        for param in _SCROLL_SPACES
        for cls in [param.values[0]]
        if "hscroll"
        in tool_schema_parameters(cls().get_tool_schemas()[0])["properties"]["action"]["enum"]
    }
    assert have_hscroll == {
        Qwen3VLDesktopActionSpace.__name__, Qwen3_5DesktopActionSpace.__name__,
    }


@pytest.mark.parametrize("space_cls", _SCROLL_SPACES)
def test_pixels_is_optional_in_every_schema_which_is_why_this_is_reachable(
    space_cls,
) -> None:
    """The reachability argument, pinned rather than asserted in prose: every
    one of the five wrapper schemas declares ``pixels`` but requires only
    ``action``. Nothing upstream fills it in — the XML/JSON parsers pass through
    exactly the parameters the model emitted — so ANY of these families can be
    handed a directionless scroll, not just the one that was caught doing it."""
    params = tool_schema_parameters(space_cls().get_tool_schemas()[0])
    assert "pixels" in params["properties"]
    assert params["required"] == ["action"]


# ---------------------------------------------------------------------------
# Anti-drift: the silent-default pattern must not come back
# ---------------------------------------------------------------------------

_SOURCES = [
    pytest.param(Qwen3VLDesktopActionSpace, id="qwen3_vl"),
    pytest.param(Qwen2_5VLDesktopActionSpace, id="qwen2_5_vl"),
    pytest.param(FaraDesktopActionSpace, id="fara"),
    pytest.param(EvoCUADesktopActionSpace, id="evocua"),
]


@pytest.mark.parametrize("space_cls", _SOURCES)
def test_missing_pixels_has_no_silent_default_for_any_scroll_action(space_cls) -> None:
    """A missing ``pixels`` argument carries no direction and must raise for
    every scroll action the family advertises."""
    enum = tool_schema_parameters(space_cls().get_tool_schemas()[0])["properties"]["action"]["enum"]
    for action in sorted({"scroll", "hscroll"} & set(enum)):
        with pytest.raises(ValueError, match="requires the 'pixels' argument"):
            _convert(space_cls, {"action": action})


# ---------------------------------------------------------------------------
# End to end: the raise becomes a terminal runtime response, and is bounded
# ---------------------------------------------------------------------------

_RUNAWAY_RESPONSE = (
    "Action: Scroll down the page to view the ingredients.\n\n"
    "<tool_call>\n"
    "<function=computer_use>\n"
    "<parameter=action>\nscroll\n</parameter>\n"
    "<parameter=down>\ntrue\n</parameter>\n"
    "<parameter=pages>\n0.5\n</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (127, 127, 127)).save(buf, format="PNG")
    return buf.getvalue()


class _FakeProcessor:
    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        return f"<prompt {len(messages)} msgs>"


class _RecordingEnv:
    """Records every ``step`` so the test can prove no wrong scroll executed."""

    def __init__(self) -> None:
        self.metadata = LiteCUAMetadata()
        self.seen_actions: list[list[LiteToolCall]] = []

    async def reset(self) -> LiteEnvObservation:
        return LiteEnvObservation(image=_png_bytes(), text="find the recipe")

    async def step(self, actions: list[LiteToolCall]) -> LiteEnvStepResult:
        self.seen_actions.append(list(actions))
        return LiteEnvStepResult(
            reward=0.0,
            terminated=True,
            results=[LiteToolResult(tool_call_id=tool_call_id(a), images=[_png_bytes()])
                     for a in actions],
        )

    async def close(self) -> None:
        return None


def test_missing_scroll_pixels_becomes_terminal_response_not_wrong_scroll() -> None:
    """The contract in full, through the REAL Qwen3.5 adapter and the shared
    sampling loop, on the verbatim runaway response.

    A GUI tool call that cannot be interpreted must (1) never reach the env as a
    wrong action, (2) end as a terminal N3 response, and (3) not escape as an
    exception. PRE-FIX (1) failed: the env was handed ``computer(scroll up 1)``
    on every one of the 591 turns of ``allrecipes.31``."""
    import asyncio

    env = _RecordingEnv()

    async def _gen(**kwargs):
        return {"response": _RUNAWAY_RESPONSE}

    agent = AdapterBasedAgent(
        generate_fn=_gen,
        processor=_FakeProcessor(),
        adapter=Qwen3_5DesktopUseAdapter(
            metadata=LiteCUAMetadata(
                dims=(
                    LiteCUAMetadata.Platform.BROWSER,
                    LiteCUAMetadata.TaskType.USE,
                ),
            ),
        ),
    )
    rl = asyncio.run(agent.sample(env, max_steps=3))

    assert len(env.seen_actions) == 1
    (finish,) = env.seen_actions[0]
    assert tool_call_name(finish) == "response"
    assert tool_call_arguments(finish) == {"text": _RUNAWAY_RESPONSE}
    assert not any(
        tool_call_name(action) in {"computer", "mobile"}
        for turn in env.seen_actions
        for action in turn
    ), f"a scroll reached the env: {env.seen_actions}"
    assert rl.terminated is True
    assert rl.truncated is False


@pytest.mark.parametrize("space_cls", _SCROLL_SPACES)
@pytest.mark.parametrize("wire", [0, 0.0, "0", "-0.0"])
def test_a_zero_scroll_is_refused_like_a_missing_one(space_cls, wire) -> None:
    """The sign of ``pixels`` is the ONLY carrier of the direction, so zero is
    as unusable as absent. It used to fall through to the magnitude conversion,
    where ``max(1, ...)`` turned "scroll nothing" into a real one-click scroll
    UP -- an action the model never asked for, executed and recorded as if it
    had happened."""
    with pytest.raises(ValueError, match="must be nonzero"):
        _convert(space_cls, {"action": "scroll", "pixels": wire})
