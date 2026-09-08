"""
Tests for the Fara-1.0 action space (``FaraDesktopActionSpace``).

Covers:
  1. Registry resolution + the 11-action ``computer_use`` enum
  2. Tool-call creation
  3. Tool-call conversion cua-lite ↔ Fara (round-trip), including the four
     web-native actions (``visit_url`` ↔ ``goto``, ``history_back`` ↔ ``back``,
     ``web_search`` / ``pause_and_memorize_fact`` by-name passthrough)
  4. ``filter_tool_schemas_for_valid_actions``

Run:
    uv run pytest tests/agents/models/fara/test_fara_action_space.py -v
"""

from __future__ import annotations

import pytest

from lite.agents.core.action_space.base import (
    ActionSpaceRegistry,
    LiteDesktopActionSpace,
    LitePointActionSpace,
)
from lite.agents.core.action_space.errors import ModelToolCallParseError
from lite.agents.models.fara.action_space import (
    FaraDesktopActionSpace,
    FaraDesktopGroundingPointActionSpace,
)
from lite.core.tools import make_tool_call
from lite.core.tools.action_space import (
    lite_action_batch_child_name_errors,
    validate_lite_action_batch_structure,
)
from lite.core.tools.calls import tool_call_arguments, tool_call_name
from lite.core.tools.schemas import (
    tool_schema_name,
    tool_schema_parameters,
)

_FARA_ACTIONS = {
    "key",
    "type",
    "mouse_move",
    "left_click",
    "scroll",
    "visit_url",
    "web_search",
    "history_back",
    "pause_and_memorize_fact",
    "wait",
    "terminate",
}


# =============================================================================
# 1. Registry + schema
# =============================================================================


class TestRegistryAndSchema:
    def test_registry_key_desktop_and_browser(self):
        assert isinstance(ActionSpaceRegistry.get("fara@desktop"), FaraDesktopActionSpace)
        assert isinstance(ActionSpaceRegistry.get("fara@browser"), FaraDesktopActionSpace)

    def test_single_tool_named_computer_use(self):
        schemas = FaraDesktopActionSpace.get_tool_schemas()
        assert len(schemas) == 1
        assert tool_schema_name(schemas[0]) == "computer_use"

    def test_action_enum(self):
        schema = FaraDesktopActionSpace.get_tool_schemas()[0]
        props = tool_schema_parameters(schema)["properties"]
        assert set(props["action"]["enum"]) == _FARA_ACTIONS
        assert "action" in tool_schema_parameters(schema)["required"]

    def test_type_input_key_args_present(self):
        # Fara runs with include_input_text_key_args=True.
        props = tool_schema_parameters(FaraDesktopActionSpace.get_tool_schemas()[0])["properties"]
        assert "press_enter" in props
        assert "delete_existing_text" in props

    def test_resolution_placeholder_in_description(self):
        desc = FaraDesktopActionSpace.get_tool_schemas()[0]["function"]["description"]
        assert "{display_width_px}x{display_height_px}" in desc


# =============================================================================
# 2. Tool-call creation
# =============================================================================


class TestToolCalls:
    def test_left_click(self):
        tc = FaraDesktopActionSpace.computer_use(action="left_click", coordinate=[500, 300])
        assert tool_call_arguments(tc) == {"action": "left_click", "coordinate": [500, 300]}

    def test_visit_url(self):
        tc = FaraDesktopActionSpace.computer_use(action="visit_url", url="https://x.com")
        assert tool_call_arguments(tc) == {"action": "visit_url", "url": "https://x.com"}

    def test_none_args_omitted(self):
        tc = FaraDesktopActionSpace.computer_use(action="history_back")
        assert tool_call_arguments(tc) == {"action": "history_back"}


# =============================================================================
# 3. Conversion (cua-lite ↔ Fara)
# =============================================================================


class TestToAgent:
    def setup_method(self):
        self.space = FaraDesktopActionSpace()

    def test_click_to_left_click(self):
        out = self.space.convert_tool_calls_to_agent(
            [LiteDesktopActionSpace.click(coordinate=[5, 6])]
        )
        assert out[0]["arguments"]["action"] == "left_click"
        assert out[0]["arguments"]["coordinate"] == [5, 6]

    def test_type_renders_a_trailing_newline_as_press_enter(self):
        """Fara's wire has its own ``press_enter``; canonical spells the same
        thing as a trailing newline. Rendering must put it back, or replay hands
        the model a ``type`` that never submits. The key is emitted ONLY when it
        is true -- a literal False would add a field the model never sent."""
        submitting = self.space.convert_tool_calls_to_agent(
            [LiteDesktopActionSpace.type(text="shoes\n")]
        )
        assert submitting[0]["arguments"]["text"] == "shoes"
        assert submitting[0]["arguments"]["press_enter"] is True

        plain = self.space.convert_tool_calls_to_agent(
            [LiteDesktopActionSpace.type(text="shoes")]
        )
        assert plain[0]["arguments"]["text"] == "shoes"
        assert "press_enter" not in plain[0]["arguments"]

    def test_computer_batch_to_agent_unwraps(self):
        tc = make_tool_call(
            "computer",
            {
                "actions": [
                    {"action": "click", "coordinate": [5, 6]},
                    {"action": "type", "text": "hello"},
                ]
            },
        )
        out = self.space.convert_tool_calls_to_agent([tc])
        assert [r["arguments"]["action"] for r in out] == ["left_click", "type"]
        assert out[1]["arguments"]["text"] == "hello"

    def test_goto_to_visit_url(self):
        out = self.space.convert_tool_calls_to_agent(
            [make_tool_call("goto", {"url": "https://a.b"})]
        )
        assert out[0]["arguments"] == {"action": "visit_url", "url": "https://a.b"}

    def test_back_to_history_back(self):
        out = self.space.convert_tool_calls_to_agent([make_tool_call("back")])
        assert out[0]["arguments"] == {"action": "history_back"}

    @pytest.mark.parametrize("direction,exp_pixels", [("up", 300), ("down", -300)])
    def test_scroll_sign(self, direction, exp_pixels):
        out = self.space.convert_tool_calls_to_agent(
            [LiteDesktopActionSpace.scroll(direction=direction, amount=3)]
        )
        assert out[0]["arguments"]["pixels"] == exp_pixels

    def test_response_fails_loudly(self):
        with pytest.raises(ValueError, match="cannot render canonical tool 'response'"):
            self.space.convert_tool_calls_to_agent([make_tool_call("response", {"text": "42"})])


class TestFromAgent:
    def setup_method(self):
        self.space = FaraDesktopActionSpace()

    def _call(self, **args):
        return [{"name": "computer_use", "arguments": args}]

    def test_left_click_to_click(self):
        out = self.space.convert_tool_calls_from_agent(
            self._call(action="left_click", coordinate=[9, 9])
        )
        assert tool_call_name(out[0]) == "computer"
        assert tool_call_arguments(out[0])["actions"] == [{"action": "click", "coordinate": [9, 9]}]

    def test_visit_url_to_goto(self):
        out = self.space.convert_tool_calls_from_agent(
            self._call(action="visit_url", url="https://a.b")
        )
        assert out[0] == make_tool_call("goto", {"url": "https://a.b"})

    def test_history_back_to_back(self):
        out = self.space.convert_tool_calls_from_agent(self._call(action="history_back"))
        assert tool_call_name(out[0]) == "back"

    def test_web_search_maps_to_bing_goto(self):
        # Fara's web_search navigates to a Bing results page; cua-lite browser envs
        # have no web_search verb, so it must become goto(bing_url).
        out = self.space.convert_tool_calls_from_agent(
            self._call(action="web_search", query="paella recipe 4.5 stars")
        )
        assert tool_call_name(out[0]) == "goto"
        url = tool_call_arguments(out[0])["url"]
        assert url.startswith("https://www.bing.com/search?q=")
        assert "paella+recipe+4.5+stars" in url

    def test_pause_memorize_passthrough(self):
        out = self.space.convert_tool_calls_from_agent(
            self._call(action="pause_and_memorize_fact", fact="f")
        )
        assert out[0] == make_tool_call("pause_and_memorize_fact", {"fact": "f"})

    def test_key_string_normalized_to_list(self):
        # Adapter/action-space must normalize a bare "ctrl+f" string to a list.
        out = self.space.convert_tool_calls_from_agent(self._call(action="key", keys="ctrl+f"))
        assert tool_call_name(out[0]) == "computer"
        assert tool_call_arguments(out[0])["actions"] == [{"action": "key", "keys": ["ctrl", "f"]}]

    @pytest.mark.parametrize(
        "raw_keys,expected",
        [
            ("ctrl++", ["ctrl", "+"]),
            ("ctrl+-", ["ctrl", "-"]),
            ("ctrl+=", ["ctrl", "="]),
        ],
    )
    def test_key_string_chord_uses_core_normalizer(self, raw_keys, expected):
        out = self.space.convert_tool_calls_from_agent(self._call(action="key", keys=raw_keys))
        assert tool_call_name(out[0]) == "computer"
        assert tool_call_arguments(out[0])["actions"] == [{"action": "key", "keys": expected}]

    @pytest.mark.parametrize("raw_keys", ["ctrl left", "ctrl -"])
    def test_key_string_rejects_phrase_like_tokens(self, raw_keys):
        with pytest.raises(ValueError, match="unknown key token"):
            self.space.convert_tool_calls_from_agent(self._call(action="key", keys=raw_keys))

    def test_key_action_does_not_fallback_to_text_field(self):
        with pytest.raises(ValueError, match="keys must not be empty"):
            self.space.convert_tool_calls_from_agent(self._call(action="key", text="enter"))

    def test_type_with_coordinate_decomposes_to_click_then_type(self):
        # Fara's ``type`` carries a coordinate (focus the field first). Must emit
        # click(coordinate) — restores the crosshair — then type(text), with
        # Fara's ``press_enter`` spelled as canonical's trailing newline.
        out = self.space.convert_tool_calls_from_agent(
            self._call(action="type", coordinate=[694, 36], text="hi", press_enter=True)
        )
        assert [tool_call_name(tc) for tc in out] == ["computer"]
        assert tool_call_arguments(out[0])["actions"] == [
            {"action": "click", "coordinate": [694, 36]},
            {"action": "type", "text": "hi\n"},
        ]

    def test_type_without_coordinate_is_bare_type(self):
        out = self.space.convert_tool_calls_from_agent(self._call(action="type", text="hi"))
        assert [tool_call_name(tc) for tc in out] == ["computer"]
        assert tool_call_arguments(out[0])["actions"] == [{"action": "type", "text": "hi"}]

    def test_terminate(self):
        out = self.space.convert_tool_calls_from_agent(
            self._call(action="terminate", status="failure")
        )
        assert out[0] == make_tool_call("terminate", {"status": "failure"})

    def test_unknown_wrapped_action_becomes_invalid_action_batch(self):
        out = self.space.convert_tool_calls_from_agent(
            self._call(action="screen_record", path="/tmp/out.mp4")
        )

        assert len(out) == 1
        assert tool_call_name(out[0]) == "computer"
        assert tool_call_arguments(out[0]) == {
            "actions": [{"action": "screen_record", "path": "/tmp/out.mp4"}],
        }
        children, error = validate_lite_action_batch_structure(
            "computer",
            tool_call_arguments(out[0]),
        )
        assert len(children) == 1
        error = lite_action_batch_child_name_errors("computer", children).get(0)
        assert error is not None
        assert error.child_action_name == "screen_record"

    def test_non_wrapper_name_with_unknown_action_stays_a_by_name_call(self):
        """Only ``computer_use`` actions take the invalid-batch path. A
        non-wrapper name that merely carried an ``action`` argument is not a
        Fara GUI action, so it stays a by-name call for env ingress to judge —
        same as a non-wrapper name with no ``action`` at all."""
        out = self.space.convert_tool_calls_from_agent(
            [{"name": "report_problem", "arguments": {"action": "zzz", "reason": "x"}}]
        )
        assert out == [make_tool_call("report_problem", {"action": "zzz", "reason": "x"})]


# =============================================================================
# 4. filter_tool_schemas_for_valid_actions
# =============================================================================


class TestFilterToolSchemasForValidActions:
    def test_filter_click_preserves_native_web_and_finish(self):
        schemas = FaraDesktopActionSpace.get_tool_schemas()
        filt = FaraDesktopActionSpace.filter_tool_schemas_for_valid_actions(schemas, ["click"])
        assert set(tool_schema_parameters(filt[0])["properties"]["action"]["enum"]) == {
            "left_click",
            "visit_url",
            "web_search",
            "history_back",
            "pause_and_memorize_fact",
            "terminate",
        }

    def test_native_web_entries_follow_their_produced_canonical_tool(self):
        """``filter_fara_action_values_for_active_extra_tools`` gates each native
        web/finish action value by the canonical tool its parse PRODUCES — so
        ``goto`` alone opens both ``visit_url`` and ``web_search`` (both parse to
        ``goto``), while ``pause_and_memorize_fact`` needs its own schema."""
        schemas = FaraDesktopActionSpace.get_tool_schemas()

        def enum(active: set[str]) -> set[str]:
            return set(
                FaraDesktopActionSpace.filter_fara_action_values_for_active_extra_tools(
                    schemas, active
                )[0]["function"]["parameters"]["properties"]["action"]["enum"]
            ) & frozenset(FaraDesktopActionSpace.FARA_ACTION_VALUE_TO_EXTRA_TOOL_NAMES)

        assert enum({"goto"}) == {"visit_url", "web_search"}
        assert enum({"goto", "back", "terminate"}) == {
            "visit_url",
            "web_search",
            "history_back",
            "terminate",
        }
        assert enum({"pause_and_memorize_fact"}) == {"pause_and_memorize_fact"}

    def test_empty_valid_actions_keeps_only_native_semantic_entries(self):
        """``filter_tool_schemas_for_valid_actions`` gates GUI actions ONLY: with no GUI action
        left, the native action values survive (they are gated instead by
        ``filter_fara_action_values_for_active_extra_tools`` from
        ``extra_tool_schemas``)."""
        schemas = FaraDesktopActionSpace.get_tool_schemas()
        filt = FaraDesktopActionSpace.filter_tool_schemas_for_valid_actions(schemas, [])
        assert set(tool_schema_parameters(filt[0])["properties"]["action"]["enum"]) == (
            frozenset(FaraDesktopActionSpace.FARA_ACTION_VALUE_TO_EXTRA_TOOL_NAMES)
        )
        # ...and both gates together drop the tool when nothing is active.
        assert (
            FaraDesktopActionSpace.filter_fara_action_values_for_active_extra_tools(filt, set())
            == []
        )


# =============================================================================
# 5. Grounding (point) — trimmed left_click schema
# =============================================================================


class TestGroundingPoint:
    def test_registry_and_single_action(self):
        space = ActionSpaceRegistry.get("fara@desktop@point")
        assert isinstance(space, FaraDesktopGroundingPointActionSpace)
        schema = FaraDesktopGroundingPointActionSpace.get_tool_schemas()[0]
        assert tool_schema_parameters(schema)["properties"]["action"]["enum"] == ["left_click"]

    def test_valid_actions_gate_is_point_only(self):
        """The grounding surface has no native standalone tools, so ``point``
        alone decides whether the trimmed wrapper is advertised at all."""
        schemas = FaraDesktopGroundingPointActionSpace.get_tool_schemas()
        gate = FaraDesktopGroundingPointActionSpace.filter_tool_schemas_for_valid_actions
        assert gate(schemas, ["point"]) == schemas
        assert gate(schemas, []) == []
        assert gate(schemas, ["click", "type"]) == []

    def test_point_round_trip(self):
        space = FaraDesktopGroundingPointActionSpace()
        agent = space.convert_tool_calls_to_agent([LitePointActionSpace.point(coordinate=[7, 8])])
        assert agent[0]["arguments"] == {"action": "left_click", "coordinate": [7, 8]}
        back = space.convert_tool_calls_from_agent(agent)
        assert tool_call_name(back[0]) == "point"
        assert tool_call_arguments(back[0])["coordinate"] == [7, 8]


# =============================================================================
# 5. Malformed-but-recoverable provider output
# =============================================================================


class TestMalformedProviderOutput:
    def setup_method(self):
        self.space = FaraDesktopActionSpace()

    @pytest.mark.parametrize(
        "name,args",
        [
            ("left_click", {"coordinate": [500, 300]}),
            ("mouse_move", {"coordinate": [500, 300]}),
            ("key", {"keys": "ctrl+v"}),
            ("scroll", {"pixels": -300}),
            ("visit_url", {"url": "https://a.b"}),
            ("history_back", {}),
        ],
    )
    def test_flat_native_action_value_used_as_tool_name_converts(self, name, args):
        """Fara exposes its whole surface as ONE ``computer_use`` wrapper. Flat
        output — the wrapper dropped, the action VALUE used as the tool name —
        must run the same dispatch branch as nested output; otherwise real work
        is lost (``web_search`` builds a Bing ``goto`` URL, ``type`` fans out to
        click+type)."""
        flat = self.space.convert_tool_calls_from_agent([{"name": name, "arguments": args}])
        nested = self.space.convert_tool_calls_from_agent(
            [{"name": "computer_use", "arguments": {"action": name, **args}}]
        )
        assert flat == nested

    def test_wrong_wrapper_name_with_nested_action_converts(self):
        out = self.space.convert_tool_calls_from_agent(
            [{"name": "computer", "arguments": {"action": "left_click", "coordinate": [5, 6]}}]
        )
        assert tool_call_name(out[0]) == "computer"
        assert tool_call_arguments(out[0])["actions"] == [{"action": "click", "coordinate": [5, 6]}]

    def test_flat_terminate_becomes_canonical_terminate(self):
        out = self.space.convert_tool_calls_from_agent(
            [{"name": "terminate", "arguments": {"status": "failure"}}]
        )
        assert out == [make_tool_call("terminate", {"status": "failure"})]

    def test_flat_response_still_raises(self):
        """Fara has no answer channel; a wire-level ``response`` stays a loud
        parse failure rather than being swallowed."""
        with pytest.raises(ModelToolCallParseError):
            self.space.convert_tool_calls_from_agent(
                [{"name": "response", "arguments": {"text": "42"}}]
            )

    def test_non_native_name_stays_a_standalone_tool(self):
        out = self.space.convert_tool_calls_from_agent(
            [{"name": "summarize", "arguments": {"text": "hi"}}]
        )
        assert out == [make_tool_call("summarize", {"text": "hi"})]

    def test_active_extra_tool_outranks_a_colliding_native_action_value(self):
        """A browsergym ``scroll(delta_y=...)`` is the env's tool, not Fara's
        native page-gesture ``scroll``. Asserted on the single-call parse:
        ``_wrap_desktop_action_calls`` separately re-wraps any call whose NAME
        matches a canonical Lite action."""
        schema = {
            "type": "function",
            "function": {
                "name": "scroll",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {"delta_y": {"type": "number"}},
                    "required": ["delta_y"],
                },
            },
        }
        out = self.space._convert_single_from_agent(
            {"name": "scroll", "arguments": {"delta_y": 100}},
            active_extra_tool_names={"scroll"},
            active_extra_tool_schemas=[schema],
        )
        assert out == [make_tool_call("scroll", {"delta_y": 100})]


def test_fara_wire_response_parse_fails_loudly() -> None:
    """Fara has no native answer channel, so wire-level ``response`` fails loudly."""
    with pytest.raises(ValueError, match=r"^Fara cannot parse wire tool 'response'$"):
        FaraDesktopActionSpace().convert_tool_calls_from_agent(
            [{"name": "response", "arguments": {"text": "42"}}]
        )
