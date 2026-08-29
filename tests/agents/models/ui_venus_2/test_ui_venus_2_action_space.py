"""UI-Venus-2 action spaces: wire text, per-surface grammar, and Sequence.

The generic ``parse(serialize(x)) == x`` sweep lives in
``tests/agents/core/action_space/test_round_trip_fidelity.py``; this file covers
what is specific to UI-Venus-2 — the three surfaces disagreeing on purpose, the
``Sequence`` batch envelope, and the rendered text itself.

Run:
    uv run pytest tests/agents/models/ui_venus_2 -p no:cacheprovider -q
"""

from __future__ import annotations

import pytest

from lite.agents.bootstrap import register_all
from lite.agents.core.action_space.base import (
    ActionSpaceRegistry,
    LiteDesktopActionSpace,
    LiteMobileActionSpace,
    LitePointActionSpace,
)
from lite.agents.models.ui_venus_2.action_space import (
    UIVenus2BrowserActionSpace,
    UIVenus2DesktopActionSpace,
    UIVenus2GroundingPointActionSpace,
    UIVenus2MobileActionSpace,
    format_tool_call_as_text,
    parse_action_text,
)
from lite.core.tools import make_tool_call
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.extra_tools import (
    LiteAppLaunchToolSet,
    LiteBrowserNavToolSet,
    LiteFinishToolSet,
)

register_all()


def _wire(space, call) -> str:
    return space.format_tool_calls_as_text(space.convert_tool_calls_to_agent([call]))


def _parse(space, text: str) -> list[dict]:
    return space.convert_tool_calls_from_agent([parse_action_text(text)])


# =============================================================================
# Registry keys
# =============================================================================
# The three ``use`` surfaces are separate registry keys, not one
# ``(desktop|browser)`` row: their grammars genuinely differ.

class TestRegistryKeys:
    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("ui_venus_2@desktop", UIVenus2DesktopActionSpace),
            ("ui_venus_2@browser", UIVenus2BrowserActionSpace),
            ("ui_venus_2@mobile", UIVenus2MobileActionSpace),
            ("ui_venus_2@desktop@point", UIVenus2GroundingPointActionSpace),
            ("ui_venus_2@browser@point", UIVenus2GroundingPointActionSpace),
            ("ui_venus_2@mobile@point", UIVenus2GroundingPointActionSpace),
        ],
    )
    def test_key_resolves_to_its_own_class(self, key, expected) -> None:
        assert ActionSpaceRegistry.get_class(key) is expected

    def test_desktop_and_browser_are_not_the_same_class(self) -> None:
        """The regression this guards: sharing one class would give the browser
        surface ``box=`` arguments and ``Sequence``, neither of which its
        grammar has."""
        assert (
            ActionSpaceRegistry.get_class("ui_venus_2@desktop")
            is not ActionSpaceRegistry.get_class("ui_venus_2@browser")
        )


# =============================================================================
# Wire text
# =============================================================================

class TestWireText:
    @pytest.mark.parametrize(
        ("projection", "expected"),
        [
            ({"name": "Click", "arguments": {"box": [512, 300]}}, "Click(box=(512, 300))"),
            ({"name": "Click", "arguments": {"box": None}}, "Click()"),
            ({"name": "Wait", "arguments": {}}, "Wait()"),
            (
                {"name": "Hotkey", "arguments": {"keys": ["ctrl", "c"], "repeat": None}},
                "Hotkey(keys=['ctrl', 'c'])",
            ),
            (
                {"name": "Swipe", "arguments": {"amount": -500, "axis": "vertical"}},
                "Swipe(amount=-500, axis='vertical')",
            ),
            (
                {"name": "Type", "arguments": {"content": "hello\n"}},
                "Type(content='hello\\n')",
            ),
        ],
    )
    def test_render(self, projection, expected) -> None:
        assert format_tool_call_as_text(projection) == expected

    def test_sequence_renders_nested_calls_not_dicts(self) -> None:
        text = format_tool_call_as_text({
            "name": "Sequence",
            "arguments": {"actions": [
                {"name": "Click", "arguments": {"box": [1, 2]}},
                {"name": "Hotkey", "arguments": {"keys": ["ctrl", "s"]}},
            ]},
        })
        assert text == "Sequence(actions=[Click(box=(1, 2)), Hotkey(keys=['ctrl', 's'])])"

    def test_render_takes_the_bare_projection_not_a_canonical_call(self) -> None:
        """The narrowed input contract: a canonical Lite call has no top-level
        ``name``, so it fails loudly instead of rendering nonsense."""
        with pytest.raises(KeyError):
            format_tool_call_as_text(LiteDesktopActionSpace.click(coordinate=[1, 2]))


class TestParseActionText:
    @pytest.mark.parametrize(
        "text",
        [
            "",
            "I will click the button",
            "Click(box=(1",                       # truncated, nothing recovered
            "Sequence(actions=[Sequence(actions=[Click(box=(1, 2))])])",  # nested
            "Click(*args)",                       # not a literal keyword call
        ],
    )
    def test_unparseable_text_is_none_not_a_guess(self, text) -> None:
        assert parse_action_text(text) is None

    def test_truncated_call_keeps_what_survived(self) -> None:
        assert parse_action_text("Type(content='hel") == {
            "name": "Type", "arguments": {"content": "hel"},
        }

    def test_tuples_become_lists(self) -> None:
        parsed = parse_action_text("Drag(end=(9, 9), start=(1, 1))")
        assert parsed == {"name": "Drag", "arguments": {"end": [9, 9], "start": [1, 1]}}


# =============================================================================
# Computer (desktop) surface
# =============================================================================

class TestDesktopSurface:
    space = UIVenus2DesktopActionSpace()

    @pytest.mark.parametrize(
        ("call", "text"),
        [
            (LiteDesktopActionSpace.click(coordinate=[5, 6]), "Click(box=(5, 6))"),
            (LiteDesktopActionSpace.click(coordinate=[5, 6], clicks=2), "DoubleClick(box=(5, 6))"),
            (LiteDesktopActionSpace.click(coordinate=[5, 6], clicks=3), "TripleClick(box=(5, 6))"),
            (
                LiteDesktopActionSpace.click(coordinate=[5, 6], button="right"),
                "RightClick(box=(5, 6))",
            ),
            (
                LiteDesktopActionSpace.click(coordinate=[5, 6], button="middle"),
                "MiddleClick(box=(5, 6))",
            ),
            (LiteDesktopActionSpace.mouse_move(coordinate=[1, 2]), "Hover(box=(1, 2))"),
            (LiteDesktopActionSpace.key_down(keys=["shift"]), "KeyDown(keys=['shift'])"),
            (LiteDesktopActionSpace.key_up(keys=["shift"]), "KeyUp(keys=['shift'])"),
            (LiteDesktopActionSpace.mouse_down(), "MouseDown()"),
            (LiteDesktopActionSpace.mouse_up(), "MouseUp()"),
        ],
    )
    def test_click_and_half_press_verbs(self, call, text) -> None:
        assert _wire(self.space, call) == text

    def test_scroll_becomes_a_signed_pixel_swipe(self) -> None:
        """``Swipe`` counts pixels and signs the direction: vertical positive is
        up, negative is down."""
        assert _wire(
            self.space, LiteDesktopActionSpace.scroll(direction="down", amount=3),
        ) == "Swipe(amount=-300, axis='vertical')"
        assert _wire(
            self.space, LiteDesktopActionSpace.scroll(direction="right", amount=2),
        ) == "Swipe(amount=200, axis='horizontal')"

    def test_type_carries_press_enter_as_a_trailing_newline(self) -> None:
        assert _wire(
            self.space, LiteDesktopActionSpace.type(text="hi", press_enter=True),
        ) == "Type(content='hi\\n')"
        parsed = _parse(self.space, "Type(content='hi\\n')")
        (child,) = tool_call_arguments(parsed[0])["actions"]
        assert child == {"action": "type", "text": "hi", "press_enter": True}

    def test_hotkey_repeat_fans_out_rather_than_collapsing(self) -> None:
        """``repeat`` has no canonical carrier. Dropping it would silently press
        the shortcut once where the model asked for three."""
        parsed = _parse(self.space, "Hotkey(keys=['ctrl', 'z'], repeat=3)")
        children = tool_call_arguments(parsed[0])["actions"]
        assert children == [{"action": "key", "keys": ["ctrl", "z"]}] * 3

    def test_mouse_down_with_a_box_moves_first(self) -> None:
        """Canonical ``mouse_down`` carries no coordinate, so the optional
        ``box`` becomes its own ``mouse_move``."""
        parsed = _parse(self.space, "MouseDown(box=(7, 8))")
        assert tool_call_arguments(parsed[0])["actions"] == [
            {"action": "mouse_move", "coordinate": [7, 8]},
            {"action": "mouse_down"},
        ]

    def test_finish_verbs(self) -> None:
        assert _wire(self.space, LiteFinishToolSet.terminate(status="success")) == "Finished()"
        assert _wire(
            self.space, LiteFinishToolSet.response(text="42"),
        ) == "Finished(content='42')"
        assert _wire(
            self.space, LiteFinishToolSet.terminate(status="failure", reason="stuck"),
        ) == "CallUser(content='stuck')"

    def test_bare_finished_is_terminate_and_content_is_response(self) -> None:
        assert tool_call_name(_parse(self.space, "Finished()")[0]) == "terminate"
        answered = _parse(self.space, "Finished(content='42')")[0]
        assert tool_call_name(answered) == "response"
        assert tool_call_arguments(answered)["text"] == "42"


class TestDesktopSequence:
    """``Sequence`` is this family's spelling of the ``computer`` action batch."""

    space = UIVenus2DesktopActionSpace()

    def test_multi_child_batch_folds_into_one_sequence(self) -> None:
        batch = make_tool_call("computer", {"actions": [
            {"action": "click", "coordinate": [1, 2]},
            {"action": "key", "keys": ["ctrl", "s"]},
        ]})
        assert _wire(self.space, batch) == (
            "Sequence(actions=[Click(box=(1, 2)), Hotkey(keys=['ctrl', 's'])])"
        )

    def test_single_child_batch_stays_a_bare_call(self) -> None:
        """``Sequence`` requires at least two actions, so a one-action turn must
        not be wrapped."""
        assert _wire(self.space, LiteDesktopActionSpace.click(coordinate=[1, 2])) == (
            "Click(box=(1, 2))"
        )

    def test_a_turn_that_ends_in_a_terminal_folds_across_both_lite_calls(self) -> None:
        """A finishing turn reaches the renderer as TWO Lite calls (a batch plus
        a standalone ``terminate``). One ``<action>`` block carries one action,
        and the grammar allows a trailing ``Finished``, so both go in one
        ``Sequence``."""
        calls = [
            make_tool_call("computer", {"actions": [{"action": "click", "coordinate": [1, 2]}]}),
            LiteFinishToolSet.terminate(status="success"),
        ]
        text = self.space.format_tool_calls_as_text(
            self.space.convert_tool_calls_to_agent(calls)
        )
        assert text == "Sequence(actions=[Click(box=(1, 2)), Finished()])"

    def test_sequence_unfolds_back_into_a_batch_plus_the_terminal(self) -> None:
        parsed = _parse(self.space, "Sequence(actions=[Click(box=(1, 2)), Finished()])")
        assert [tool_call_name(call) for call in parsed] == ["computer", "terminate"]
        assert tool_call_arguments(parsed[0])["actions"] == [
            {"action": "click", "coordinate": [1, 2]},
        ]

    def test_sequence_of_gui_actions_merges_into_one_batch(self) -> None:
        parsed = _parse(
            self.space, "Sequence(actions=[Click(box=(1, 2)), Type(content='x')])",
        )
        assert len(parsed) == 1
        assert tool_call_arguments(parsed[0])["actions"] == [
            {"action": "click", "coordinate": [1, 2]},
            {"action": "type", "text": "x"},
        ]


# =============================================================================
# Browser surface
# =============================================================================

class TestBrowserSurface:
    space = UIVenus2BrowserActionSpace()

    def test_points_are_spelled_point_not_box(self) -> None:
        assert _wire(self.space, LiteDesktopActionSpace.click(coordinate=[5, 6])) == (
            "Click(point=(5, 6))"
        )

    def test_scroll_keeps_the_anchor_and_the_direction(self) -> None:
        assert _wire(
            self.space,
            LiteDesktopActionSpace.scroll(direction="down", amount=3, coordinate=[4, 5]),
        ) == "Scroll(point=(4, 5), direction='down')"

    def test_browser_nav_extras(self) -> None:
        assert _wire(self.space, LiteBrowserNavToolSet.goto(url="https://x.dev")) == (
            "Launch(url='https://x.dev')"
        )
        assert _wire(self.space, LiteBrowserNavToolSet.back()) == "PressBack()"
        assert tool_call_name(_parse(self.space, "PressBack()")[0]) == "back"

    def test_press_keys_are_keystrokes_on_this_surface(self) -> None:
        """The browser prompt spells ``PressHome`` as "the Home key to scroll to
        the top", not a device button."""
        for text, key in [("PressEnter()", "enter"), ("PressHome()", "home")]:
            (child,) = tool_call_arguments(_parse(self.space, text)[0])["actions"]
            assert child == {"action": "key", "keys": [key]}

    @pytest.mark.parametrize(
        "text",
        ["GetUrl()", "TakeNote(content='the price is 12')", "SelectOption(index=3)"],
    )
    def test_verbs_with_no_canonical_counterpart_pass_through_by_name(self, text) -> None:
        """Consuming them would destroy a call the action space was never asked
        about; env feedback owns unknown executable names."""
        (call,) = _parse(self.space, text)
        assert tool_call_name(call) == text.split("(")[0]

    def test_submitting_type_is_refused_loudly(self) -> None:
        """The browser ``Type`` does not submit and the grammar allows one action
        per turn, so a canonical ``press_enter`` has nowhere to go."""
        with pytest.raises(ValueError, match="press_enter"):
            self.space.convert_tool_calls_to_agent(
                [LiteDesktopActionSpace.type(text="q", press_enter=True)]
            )

    def test_multi_action_turn_is_refused_loudly(self) -> None:
        """No ``Sequence`` here: rendering two actions into one ``<action>``
        block would emit text no UI-Venus-2 harness can read back."""
        batch = make_tool_call("computer", {"actions": [
            {"action": "click", "coordinate": [1, 2]},
            {"action": "type", "text": "x"},
        ]})
        with pytest.raises(ValueError, match="one action per <action> block"):
            self.space.convert_tool_calls_to_agent([batch])


# =============================================================================
# Mobile surface
# =============================================================================

class TestMobileSurface:
    space = UIVenus2MobileActionSpace()

    @pytest.mark.parametrize(
        ("call", "text"),
        [
            (LiteMobileActionSpace.tap(coordinate=[5, 6]), "Click(point=(5, 6))"),
            (
                LiteMobileActionSpace.tap(coordinate=[5, 6], clicks=2),
                "DoubleClick(point=(5, 6))",
            ),
            (
                LiteMobileActionSpace.swipe(start_coordinate=[1, 2], coordinate=[3, 4]),
                "Swipe(start=(1, 2), end=(3, 4))",
            ),
            (
                LiteMobileActionSpace.drag(start_coordinate=[1, 2], coordinate=[3, 4]),
                "Drag(start=(1, 2), end=(3, 4))",
            ),
            (LiteMobileActionSpace.system_button(button="Recent"), "PressRecent()"),
            (LiteMobileActionSpace.screenshot(), "GetScreenshot()"),
        ],
    )
    def test_grammar(self, call, text) -> None:
        assert _wire(self.space, call) == text

    def test_swipe_and_drag_stay_distinct(self) -> None:
        """Unlike the sibling families, this grammar has BOTH verbs, so a
        canonical ``drag`` is not degraded into a fling."""
        assert tool_call_arguments(_parse(self.space, "Drag(start=(1, 2), end=(3, 4))")[0])[
            "actions"
        ] == [{"action": "drag", "start_coordinate": [1, 2], "coordinate": [3, 4]}]

    def test_answer_is_the_reply_channel_and_finished_is_completion(self) -> None:
        """Mobile is the one surface with a dedicated ``Answer`` verb, so
        ``Finished`` keeps a single meaning here."""
        assert _wire(self.space, LiteFinishToolSet.response(text="42")) == (
            "Answer(content='42')"
        )
        assert _wire(self.space, LiteFinishToolSet.terminate(status="success")) == "Finished()"
        assert tool_call_name(_parse(self.space, "Answer(content='42')")[0]) == "response"
        assert tool_call_name(_parse(self.space, "Finished(content='done')")[0]) == "terminate"

    def test_launch_app(self) -> None:
        assert _wire(self.space, LiteAppLaunchToolSet.open_app(app_name="Settings")) == (
            "LaunchApp(app='Settings')"
        )
        opened = _parse(self.space, "LaunchApp(app='Settings')")[0]
        assert tool_call_name(opened) == "open_app"
        assert tool_call_arguments(opened)["app_name"] == "Settings"

    def test_menu_button_has_no_row_and_is_refused(self) -> None:
        with pytest.raises(ValueError, match="system_button"):
            self.space.convert_tool_calls_to_agent(
                [LiteMobileActionSpace.system_button(button="Menu")]
            )


class TestTerminalVerbsAcrossSurfaces:
    """The finish channels are the one vocabulary all three surfaces share."""

    @pytest.mark.parametrize(
        "space",
        [UIVenus2DesktopActionSpace(), UIVenus2BrowserActionSpace(), UIVenus2MobileActionSpace()],
    )
    @pytest.mark.parametrize(
        "call",
        [
            LiteFinishToolSet.terminate(status="success"),
            LiteFinishToolSet.terminate(status="failure"),
            LiteFinishToolSet.terminate(status="failure", reason="stuck"),
            LiteFinishToolSet.response(text="42"),
        ],
        ids=["success", "failure", "failure_with_reason", "answer"],
    )
    def test_finish_verbs_round_trip_exactly(self, space, call) -> None:
        """A reasonless ``CallUser`` must not gain an invented reason on the way
        back, or a plain failed ``terminate`` stops round-tripping."""
        wire = space.convert_tool_calls_to_agent([call])
        assert space.convert_tool_calls_from_agent(wire) == [call]


# =============================================================================
# Grounding surface
# =============================================================================

class TestGroundingSurface:
    space = UIVenus2GroundingPointActionSpace()

    def test_advertises_no_tool_schemas(self) -> None:
        """The ``[x,y]`` format lives entirely in the prompt."""
        assert self.space.get_tool_schemas() == []

    def test_point_round_trips(self) -> None:
        call = LitePointActionSpace.point(coordinate=[512, 300])
        wire = self.space.convert_tool_calls_to_agent([call])
        assert wire == [{"name": "point", "arguments": {"box": [512, 300]}}]
        assert self.space.convert_tool_calls_from_agent(wire) == [call]

    def test_env_extras_pass_through_both_ways(self) -> None:
        refusal = make_tool_call("report_infeasible", {"reason": "not on screen"})
        wire = self.space.convert_tool_calls_to_agent([refusal])
        assert wire == [{"name": "report_infeasible", "arguments": {"reason": "not on screen"}}]
        assert self.space.convert_tool_calls_from_agent(wire) == [refusal]

    def test_canonical_gui_actions_are_dropped_not_smuggled(self) -> None:
        assert self.space.convert_tool_calls_to_agent(
            [LiteDesktopActionSpace.click(coordinate=[1, 2])]
        ) == []
