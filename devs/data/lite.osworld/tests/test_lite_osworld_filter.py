"""Unit tests for the shared Lite.OSWorld/Lite.CUAGym collection filter.

The basename is dataset-qualified on purpose. ``devs/data/*/`` dirs are not
importable packages (``lite.osworld`` is not even a legal module name), so
pytest names each test module by its bare basename -- two ``test_filter.py``
files here collide and ``pytest devs/data`` silently drops one of them with an
"import file mismatch" error. Keep every test basename under ``devs/`` unique.

Run: uv run pytest devs/data/lite.osworld/tests/test_lite_osworld_filter.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.calls import (
    make_tool_call,
    tool_call_arguments,
    tool_call_id,
    tool_call_name,
)
from lite.core.tools.schemas import tool_schema_name
from lite.data.staging import coerce_messages

_spec = importlib.util.spec_from_file_location(
    "lite_osworld_filter", Path(__file__).resolve().parents[1] / "filter.py"
)
assert _spec is not None and _spec.loader is not None
flt = importlib.util.module_from_spec(_spec)
# Register before exec: `filter.py` runs its annotate pass in a process pool, and
# a pool pickles its worker BY REFERENCE (module name + qualname). A module loaded
# from a path without this line is absent from ``sys.modules``, so the worker is
# unresolvable in the child and every ``main()`` test dies on PicklingError.
sys.modules[_spec.name] = flt
_spec.loader.exec_module(flt)

NOOP = frozenset(flt.DEFAULT_NOOP_ACTIONS)


# --- message builders -------------------------------------------------------
def _user(text="obs", img=True):
    content = [{"type": "text", "text": text}]
    if img:
        content.append({"type": "image", "image": "<png>"})
    return {"role": "user", "content": content, "tool_calls": []}


def _asst(*actions):
    """actions: (name, args_dict) tuples."""
    tcs = [
        make_tool_call(n, a, call_id=f"call_{i:04d}")
        for i, (n, a) in enumerate(actions)
    ]
    return {"role": "assistant", "content": [], "tool_calls": tcs}


def _call_names(messages):
    return [
        tool_call_name(tc)
        for m in messages
        if m["role"] == "assistant"
        for tc in m["tool_calls"]
    ]


def _schema_names(schemas):
    return [tool_schema_name(schema) for schema in schemas]


def _desktop_metadata(others: dict, **kwargs) -> dict:
    return LiteCUAMetadata(dims=("desktop", "use"), others=others, **kwargs).to_dict()


def _batched_computer(*actions):
    return make_tool_call(
        "computer",
        {"actions": [
            {"action": name, **args}
            for name, args in actions
        ]},
        call_id="call_0000",
    )


def _click(x=1, y=1):
    return ("click", {"coordinate": [x, y], "text": None, "keys": None, "start_coordinate": None})


def _key(*keys):
    return ("key", {"coordinate": None, "text": None, "keys": list(keys), "start_coordinate": None})


def _type(text):
    return ("type", {"coordinate": None, "text": text, "keys": None, "start_coordinate": None})


def _screenshot():
    return ("screenshot", {})


# --- strip_noop_actions -----------------------------------------------------
def test_strips_screenshot_and_drops_empty_turn_preserving_alternation():
    msgs = [
        _user("goal"), _asst(_click()),
        _user(), _asst(_screenshot()),          # no-op-only → drop turn + its obs
        _user(), _asst(_key("ctrl", "d")),
    ]
    out, n_strip, n_drop = flt.strip_noop_actions(msgs, NOOP)
    assert n_strip == 1 and n_drop == 1
    roles = [m["role"] for m in out]
    assert all(roles[i] != roles[i + 1] for i in range(len(roles) - 1)), "alternation broken"
    names = _call_names(out)
    assert "screenshot" not in names and names == ["click", "key"]


def test_strips_inline_screenshot_but_keeps_real_action_in_same_turn():
    msgs = [_user("goal"), _asst(_screenshot(), _click())]
    out, n_strip, n_drop = flt.strip_noop_actions(msgs, NOOP)
    assert n_strip == 1 and n_drop == 0
    assert _call_names(out) == ["click"]


def test_strips_noops_inside_batched_computer_but_keeps_real_actions():
    msgs = [
        _user("goal"),
        {"role": "assistant", "content": [], "tool_calls": [
            _batched_computer(_screenshot(), _click(), _screenshot()),
        ]},
    ]
    out, n_strip, n_drop = flt.strip_noop_actions(msgs, NOOP)
    assert n_strip == 2 and n_drop == 0
    tc = out[-1]["tool_calls"][0]
    assert tool_call_name(tc) == "computer"
    assert tool_call_arguments(tc)["actions"] == [
        {"action": "click", "coordinate": [1, 1]},
    ]


def test_strips_a_type_that_types_nothing():
    """An empty ``type`` executed nothing, so it is a no-op like ``screenshot``.

    It is also not something the model chose: the XML parser yields ``text=""``
    when a malformed reply drops the ``<parameter=text>`` opener, and the turn
    was then recorded as a successfully executed action that moved no state.
    """
    msgs = [
        _user("goal"),
        {"role": "assistant", "content": [], "tool_calls": [
            _batched_computer(("type", {"text": ""}), _click()),
        ]},
    ]
    out, n_strip, n_drop = flt.strip_noop_actions(msgs, NOOP)
    assert n_strip == 1 and n_drop == 0
    assert tool_call_arguments(out[-1]["tool_calls"][0])["actions"] == [
        {"action": "click", "coordinate": [1, 1]},
    ]


def test_a_type_that_types_something_is_never_stripped():
    """Only the EMPTY one is a no-op -- a text of ``"0"`` or a lone newline is
    real content the keyboard acts on."""
    for text in ("x", "0", "\n", " "):
        msgs = [
            _user("goal"),
            {"role": "assistant", "content": [], "tool_calls": [
                _batched_computer(("type", {"text": text})),
            ]},
        ]
        out, n_strip, n_drop = flt.strip_noop_actions(msgs, NOOP)
        assert (n_strip, n_drop) == (0, 0), text
        assert tool_call_arguments(out[-1]["tool_calls"][0])["actions"] == [
            {"action": "type", "text": text},
        ], text


def test_strip_noop_rewrite_drops_stale_raw_response():
    msgs = [
        _user("goal"),
        {
            "role": "assistant",
            "content": [],
            "tool_calls": [
                _batched_computer(_click(), _screenshot()),
            ],
            "raw_response": {
                "adapter_key": "qwen3_vl@desktop@use",
                "text": "old raw action",
            },
        },
    ]

    out, n_strip, n_drop = flt.strip_noop_actions(msgs, NOOP)

    assert (n_strip, n_drop) == (1, 0)
    assert "raw_response" not in out[1]


@pytest.mark.parametrize("name", ["response", "terminate", "goto", "open_app", "bash"])
def test_standalone_extra_rejected_inside_batched_computer(name):
    msgs = [
        _user("goal"),
        {"role": "assistant", "content": [], "tool_calls": [
            _batched_computer((name, {"text": "x"})),
        ]},
    ]
    with pytest.raises(ValueError, match="must not be nested"):
        flt.strip_noop_actions(msgs, NOOP)


def test_drops_all_noop_batched_computer_turn_preserving_alternation():
    msgs = [
        _user("goal"), _asst(_click()),
        _user(), {"role": "assistant", "content": [], "tool_calls": [
            _batched_computer(_screenshot(), ("wait", {})),
        ]},
        _user(), _asst(_key("ctrl", "d")),
    ]
    out, n_strip, n_drop = flt.strip_noop_actions(msgs, NOOP)
    assert n_strip == 2 and n_drop == 1
    assert [m["role"] for m in out] == ["user", "assistant", "user", "assistant"]
    assert _call_names(out) == ["click", "key"]


# --- bare Ctrl+S no-op save -------------------------------------------------
def test_strips_bare_ctrl_s_when_enabled():
    msgs = [
        _user("goal"), _asst(_type("=TRIM(B2)")),
        _user(), _asst(_key("ctrl", "s")),       # trailing no-op save
    ]
    out, n_strip, n_drop = flt.strip_noop_actions(msgs, NOOP, strip_noop_save=True)
    assert n_strip == 1 and n_drop == 1
    assert _call_names(out) == ["type"]


def test_keeps_ctrl_s_for_export_trajectory():
    # types a filename → genuine Save-As / export → must NOT strip the save
    msgs = [
        _user("goal"), _asst(_key("ctrl", "shift", "s")),
        _user(), _asst(_type("Movies_Export.csv")),
        _user(), _asst(_key("enter")),
    ]
    out, n_strip, _ = flt.strip_noop_actions(msgs, NOOP, strip_noop_save=True)
    names = _call_names(out)
    assert "key" in names and n_strip == 0, "export save must be preserved"


def test_bare_ctrl_s_not_stripped_when_disabled():
    msgs = [_user("goal"), _asst(_key("ctrl", "s"))]
    out, n_strip, _ = flt.strip_noop_actions(msgs, NOOP, strip_noop_save=False)
    assert n_strip == 0


# --- footguns ---------------------------------------------------------------
def test_footgun_loop_3_identical():
    a = _click(5, 5)
    msgs = [_user("g"), _asst(a, a, a)]   # 3 identical in one turn
    flags = flt._traj_footguns(msgs, drop_loops=True, drop_undo_storm=False, drop_no_submit=False)
    assert "loop" in flags


def test_footgun_loop_detects_repeated_batched_actions():
    a = _click(5, 5)
    msgs = [
        _user("g"),
        {"role": "assistant", "content": [], "tool_calls": [
            _batched_computer(a, a, a),
        ]},
    ]
    flags = flt._traj_footguns(msgs, drop_loops=True, drop_undo_storm=False, drop_no_submit=False)
    assert "loop" in flags


def test_footgun_undo_storm():
    msgs = [
        _user("g"),
        _asst(
            _key("ctrl", "z"),
            _key("ctrl", "z"),
            _key("ctrl", "z"),
            _key("ctrl", "z"),
        ),
    ]
    flags = flt._traj_footguns(
        msgs,
        drop_loops=False,
        drop_undo_storm=True,
        drop_no_submit=False,
        undo_storm_min=4,
    )
    assert "undo_storm" in flags
    # 3 undos is below the threshold
    msgs3 = [
        _user("g"),
        _asst(_key("ctrl", "z"), _key("ctrl", "z"), _key("ctrl", "z")),
    ]
    flags3 = flt._traj_footguns(
        msgs3,
        drop_loops=False,
        drop_undo_storm=True,
        drop_no_submit=False,
        undo_storm_min=4,
    )
    assert "undo_storm" not in flags3


def test_footgun_no_submit():
    msgs = [_user("g"), _asst(_click())]   # never terminates
    assert "no_submit" in flt._traj_footguns(msgs, False, False, drop_no_submit=True)
    msgs_ok = [_user("g"), _asst(_click()), _user(), _asst(("terminate", {}))]
    assert "no_submit" not in flt._traj_footguns(msgs_ok, False, False, drop_no_submit=True)


def test_footgun_no_submit_accepts_content_only_done_final():
    msgs = [_user("g"), _asst(_click()), _user(), _done_turn()]
    assert "no_submit" not in flt._traj_footguns(
        msgs, False, False, drop_no_submit=True
    )


def test_no_footguns_on_clean_trajectory():
    msgs = [
        _user("g"),
        _asst(_click()),
        _user(),
        _asst(_type("x")),
        _user(),
        _asst(("terminate", {})),
    ]
    assert flt._traj_footguns(msgs, True, True, True) == set()


@pytest.mark.parametrize("alias", ["answer", "done"])
def test_boundary_aliases_do_not_count_as_canonical_submit(alias):
    msgs = [_user("g"), _asst((alias, {}))]
    assert "no_submit" in flt._traj_footguns(
        msgs, False, False, drop_no_submit=True
    )


def test_exclude_reasons_emits_canonical_footgun_reasons():
    msgs = [_asst(_click()), _asst(_click()), _asst(_click())]

    assert flt._exclude_reasons(
        msgs,
        {"others": {"terminated": False}},
        check_loops=True,
        check_undo_storm=False,
        check_no_submit=True,
        undo_storm_min=4,
    ) == ["incomplete", "footgun:loop", "footgun:no_submit"]


def test_exclude_reasons_emits_undo_storm_footgun_reason():
    msgs = [_asst(_key("ctrl", "z")) for _ in range(4)]

    assert flt._exclude_reasons(
        msgs,
        {"others": {"terminated": True}},
        check_loops=False,
        check_undo_storm=True,
        check_no_submit=False,
        undo_storm_min=4,
    ) == ["footgun:undo_storm"]


# --- reward_vision_disagree soft tag ---------------------------------------
def test_reward_vision_disagree_true_when_reward0_vision_done():
    # RC-FN-8 (impress mono font) / RC-FN-13 (GIMP 400x400): pixel/format-strict
    # checker returns reward=0 while the agent visually completed the task.
    assert flt._reward_vision_disagree(
        {"others": {"episode_return": 0.0, "vision_done": True}}
    ) is True


def test_reward_vision_disagree_true_on_false_success():
    # The reverse disagreement: reward passed but the vision verdict says not done.
    assert flt._reward_vision_disagree(
        {"others": {"episode_return": 1.0, "vision_done": False}}
    ) is True


def test_reward_vision_disagree_false_when_agree():
    assert flt._reward_vision_disagree(
        {"others": {"episode_return": 1.0, "vision_done": True}}
    ) is False
    assert flt._reward_vision_disagree(
        {"others": {"episode_return": 0.0, "vision_done": False}}
    ) is False


def test_reward_vision_disagree_false_without_vision_verdict():
    # No vision judgement (key absent or non-bool) ⇒ no disagreement.
    assert flt._reward_vision_disagree({"others": {"episode_return": 0.0}}) is False
    assert flt._reward_vision_disagree(
        {"others": {"episode_return": 0.0, "vision_done": None}}
    ) is False
    assert flt._reward_vision_disagree(
        {"others": {"episode_return": 0.0, "vision_done": "yes"}}
    ) is False


def test_exclude_reasons_tags_reward_vision_disagree():
    # A reward=0 ∧ vision-done trajectory gets the soft tag; nothing else is triggered.
    msgs = [_user("g"), _asst(_click()), _user(), _asst(("terminate", {"status": "success"}))]
    assert flt._exclude_reasons(
        msgs,
        {"others": {"terminated": True, "episode_return": 0.0, "vision_done": True}},
        check_loops=True,
        check_undo_storm=True,
        check_no_submit=True,
        undo_storm_min=4,
    ) == ["reward_vision_disagree"]


def test_exclude_reasons_clean_trajectory_has_no_reward_vision_tag():
    # Reward and vision agree on success → clean, no exclude_reason at all.
    msgs = [_user("g"), _asst(_click()), _user(), _asst(("terminate", {"status": "success"}))]
    assert flt._exclude_reasons(
        msgs,
        {"others": {"terminated": True, "episode_return": 1.0, "vision_done": True}},
        check_loops=True,
        check_undo_storm=True,
        check_no_submit=True,
        undo_storm_min=4,
    ) == []


def test_validate_trajectory_reason_accepts_reward_vision_disagree():
    assert flt.validate_trajectory_reason("reward_vision_disagree") == "reward_vision_disagree"


def test_validate_trajectory_reason_rejects_invalid_details():
    for reason in (
        "footgun",
        "footgun:captcha",
        "footgun_loop",
        "footgun_undo_storm",
        "footgun_no_submit",
        "incomplete:anything",
        "reward_vision_disagree:anything",
    ):
        with pytest.raises(ValueError):
            flt.validate_trajectory_reason(reason)


# --- ensure_terminate_action ------------------------------------------------
def _done_turn():
    """The real final turn GPT-5.5 leaves: a text 'Done.' action_description, no tool_call."""
    return {"role": "assistant",
            "content": [{"type": "action_description", "text": "Done."}], "tool_calls": []}


def test_ensure_terminate_helper_appends_to_final_done_turn_when_opted_in():
    msgs = [_user("g"), _asst(_click()), _user(), _done_turn()]
    out, injected = flt.ensure_terminate_action(msgs)
    assert injected is True
    last = [m for m in out if m["role"] == "assistant"][-1]
    # terminate appended; the 'Done.' description is preserved on the same turn.
    assert last["content"][0]["text"] == "Done."
    assert [tool_call_name(tc) for tc in last["tool_calls"]] == ["terminate"]
    assert set(last["tool_calls"][0]) == {"id", "type", "function"}
    assert flt._args_of(last["tool_calls"][0]) == {"status": "success"}
    # alternation intact, no new turn added.
    roles = [m["role"] for m in out]
    assert all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))
    assert len(out) == len(msgs)


def test_ensure_terminate_idempotent_when_already_submitted():
    msgs = [_user("g"), _asst(_click()), _user(), _asst(("terminate", {"status": "success"}))]
    out, injected = flt.ensure_terminate_action(msgs)
    assert injected is False and out is msgs  # untouched


def test_ensure_terminate_does_not_append_to_real_action():
    msgs = [_user("g"), _asst(_key("ctrl", "s"))]
    out, injected = flt.ensure_terminate_action(msgs)
    assert injected is False
    assert _call_names([out[-1]]) == ["key"]


def test_ensure_terminate_does_not_mutate_input():
    import copy
    msgs = [_user("g"), _asst(_click()), _user(), _done_turn()]
    before = copy.deepcopy(msgs)
    flt.ensure_terminate_action(msgs)
    assert msgs == before


def test_annotate_metadata_adds_terminate_schema_when_terminate_injected():
    import json

    raw = json.dumps(LiteCUAMetadata(dims=("desktop", "use"), others={}).to_dict())
    out = flt._annotate_metadata(raw, [], injected_terminate=True)
    md = json.loads(out)
    names = _schema_names(md["extra_tool_schemas"])
    assert names == ["terminate"]
    assert "function" in md["extra_tool_schemas"][0]
    assert md["others"] == {}


def test_annotate_metadata_does_not_duplicate_existing_terminate_schema():
    md = LiteCUAMetadata(
        dims=("desktop", "use"),
        extra_tool_schemas=[flt.TERMINATE_SCHEMA],
        others={},
    ).to_dict()
    out = flt._annotate_metadata(md, [], injected_terminate=True)
    names = _schema_names(out["extra_tool_schemas"])
    assert names == ["terminate"]


# --- collapse_inline_reasoning -----------------------------------------------
def _asst_reasoning(text):
    return {"role": "assistant",
            "content": [{"type": "inline_reasoning", "text": text},
                        {"type": "action_description", "text": "Click."}],
            "tool_calls": [make_tool_call("click", {}, call_id="call_0000")]}


def test_collapse_inline_reasoning_flattens_to_one_line():
    msgs = [_user("g"), _asst_reasoning("**Header**\n\nLine one.\nLine two.  extra")]
    out, n = flt.collapse_inline_reasoning(msgs)
    assert n == 1
    assert out[1]["content"][0]["text"] == "**Header** Line one. Line two. extra"
    assert out[1]["content"][1]["text"] == "Click."          # action_description untouched
    assert "\n" in msgs[1]["content"][0]["text"]             # input not mutated


def test_collapse_inline_reasoning_rewrite_drops_stale_raw_response():
    msgs = [_user("g"), _asst_reasoning("line 1\nline 2")]
    msgs[1]["raw_response"] = {
        "adapter_key": "qwen3_vl@desktop@use",
        "text": "old raw reasoning",
    }

    out, n = flt.collapse_inline_reasoning(msgs)

    assert n == 1
    assert "raw_response" not in out[1]


def test_collapse_inline_reasoning_noop_on_clean_text():
    msgs = [_user("g"), _asst_reasoning("already one line")]
    out, n = flt.collapse_inline_reasoning(msgs)
    assert n == 0 and out[1]["content"][0]["text"] == "already one line"


# --- has_oob_coordinate: unconditional out-of-[0,1000] coordinate hard drop ---

def test_oob_coordinate_flagged():
    assert flt.has_oob_coordinate([_user("g"), _asst(_click(1500, 5))])   # x > 1000
    assert flt.has_oob_coordinate([_user("g"), _asst(_click(5, -3))])     # y < 0
    assert flt.has_oob_coordinate([
        _user("g"),
        _asst(("drag", {"start_coordinate": [-1, 5], "coordinate": [10, 10]})),
    ])
    assert flt.has_oob_coordinate([
        _user("g"),
        {"role": "assistant", "content": [], "tool_calls": [
            _batched_computer(_click(1, 1), _click(1001, 2)),
        ]},
    ])
    assert flt.has_oob_coordinate([
        _user("g"),
        {"role": "assistant", "content": [], "tool_calls": [
            _batched_computer(("drag", {"start_coordinate": [1, 1001], "coordinate": [10, 10]})),
        ]},
    ])
    assert flt.has_oob_coordinate([
        _user("g"),
        _asst(("click", {"coordinate": (1001, 5)})),
    ])


def test_in_range_coordinate_NOT_flagged():
    assert not flt.has_oob_coordinate([_user("g"), _asst(_click(0, 0), _click(1000, 1000))])


def test_non_click_action_NOT_flagged():
    # key / type carry coordinate=None → nothing to flag
    assert not flt.has_oob_coordinate([_user("g"), _asst(_key("ctrl", "s"), _type("hi"))])


def test_null_arguments_rejected_by_canonical_filter():
    msgs = [{"role": "assistant", "tool_calls": [
        {
            "id": "call_back",
            "type": "function",
            "function": {"name": "back", "arguments": None},
        },
    ]}]
    with pytest.raises(ValueError, match="arguments must be a dict"):
        flt.has_oob_coordinate(msgs)


def test_bare_agent_wire_call_rejected_by_canonical_filter():
    msgs = [{"role": "assistant", "tool_calls": [
        {"name": "back", "arguments": {}},
    ]}]

    with pytest.raises(KeyError, match="function"):
        flt.has_oob_coordinate(msgs)


# --- mandatory quality gates -----------------------------------------------
def _reasoning_turn(thought, action=""):
    content = [{"type": "inline_reasoning", "text": thought}]
    if action:
        content.append({"type": "action_description", "text": action})
    return {"role": "assistant", "content": content, "tool_calls": []}


def test_policy_tags_incomplete_trajectory():
    flags = flt._trajectory_policy_violations(
        [_user("g"), _done_turn()],
        {"others": {"terminated": False, "truncated": True}},
    )
    assert flags == {"incomplete"}


def test_policy_detects_dependency_install_in_terminal():
    msgs = [
        _user("g"),
        _asst(_key("ctrl", "alt", "t")),
        _user(),
        _asst(_type("sudo apt-get install -y imagemagick")),
        _user(),
        _done_turn(),
    ]
    flags = flt._trajectory_policy_violations(
        msgs, {"others": {"terminated": True, "truncated": False}}
    )
    assert "dependency_install" in flags


def test_policy_detects_dependency_install_inside_batched_computer():
    msgs = [
        _user("g"), {"role": "assistant", "content": [], "tool_calls": [
            _batched_computer(_key("ctrl", "alt", "t")),
        ]},
        _user(), {"role": "assistant", "content": [], "tool_calls": [
            _batched_computer(_type("sudo apt-get install -y imagemagick")),
        ]},
        _user(), _done_turn(),
    ]
    assert "dependency_install" in flt._trajectory_policy_violations(
        msgs, {"others": {"terminated": True, "truncated": False}}
    )


def test_policy_detects_chained_dependency_install():
    msgs = [
        _user("g"), _asst(_key("ctrl", "alt", "t")),
        _user(), _asst(_type("sudo apt-get update && sudo apt-get install -y imagemagick")),
        _user(), _done_turn(),
    ]
    assert "dependency_install" in flt._trajectory_policy_violations(
        msgs, {"others": {"terminated": True, "truncated": False}}
    )


def test_policy_detects_flags_before_install():
    # Flags BETWEEN the package manager and ``install`` exercise the
    # ``(?:-\S+\s+)*`` group — a raw-string ``-\\S+`` typo silently skipped
    # every one of these forms while the flags-after-install tests stayed green.
    for command in (
        "sudo apt-get -y install jq",
        "sudo apt -y install ffmpeg",
        "dnf -y install foo",
        "yum -y install foo",
        "sudo apt-get --yes -q install pandoc",
    ):
        msgs = [
            _user("g"), _asst(_key("ctrl", "alt", "t")),
            _user(), _asst(_type(command)),
            _user(), _done_turn(),
        ]
        assert "dependency_install" in flt._trajectory_policy_violations(
            msgs, {"others": {"terminated": True, "truncated": False}}
        ), command


def test_policy_tracks_terminal_from_assistant_description():
    install_turn = _asst(_type("sudo apt-get install -y qpdf"))
    install_turn["content"] = [
        {"type": "inline_reasoning", "text": "The terminal is open at the prompt."}
    ]
    msgs = [_user("Linearize this PDF"), install_turn, _user(), _done_turn()]
    assert "dependency_install" in flt._trajectory_policy_violations(
        msgs, {"others": {"terminated": True, "truncated": False}}
    )


def test_policy_detects_scripts_but_allows_simple_shell_composition():
    base = {"others": {"terminated": True, "truncated": False}}
    complex_msgs = [
        _user("g"), _asst(_key("ctrl", "alt", "t")),
        _user(), _asst(_type("python -c 'print(1)'")),
        _user(), _done_turn(),
    ]
    assert "complex_shell" in flt._trajectory_policy_violations(complex_msgs, base)

    simple_msgs = [
        _user("g"), _asst(_key("ctrl", "alt", "t")),
        _user(), _asst(_type("mkdir -p out && find . -type f | wc -l > out/count.txt")),
        _user(), _done_turn(),
    ]
    assert "complex_shell" not in flt._trajectory_policy_violations(simple_msgs, base)

    process_substitution = [
        _user("g"), _asst(_key("ctrl", "alt", "t")),
        _user(), _asst(_type("cat <(cat a)")),
        _user(), _done_turn(),
    ]
    assert "complex_shell" in flt._trajectory_policy_violations(
        process_substitution, base
    )


def test_policy_allows_multiple_simple_terminal_inputs():
    msgs = [_user("g"), _asst(_key("ctrl", "alt", "t"))]
    for text in ("pwd", "ls -la", "cd /tmp", "find . -type f"):
        msgs.extend([_user(), _asst(_type(text))])
    msgs.extend([_user(), _done_turn()])
    assert not flt._trajectory_policy_violations(
        msgs, {"others": {"terminated": True, "truncated": False}}
    )


def test_policy_does_not_treat_gui_text_as_terminal_input():
    msgs = [
        _user("Bookmark this page as Python Docs"),
        _asst(_type("Python Docs")),
        _user(),
        _done_turn(),
    ]
    assert not flt._trajectory_policy_violations(
        msgs, {"others": {"terminated": True, "truncated": False}}
    )


def test_ensure_terminate_helper_accepts_completed_final_thought_when_opted_in():
    msgs = [
        _user("g"),
        _reasoning_turn("The requested value is visibly applied; the task is complete.", "Done."),
    ]
    out, injected = flt.ensure_terminate_action(msgs)
    assert injected
    assert tool_call_name(out[-1]["tool_calls"][0]) == "terminate"


def test_ensure_terminate_helper_accepts_saved_final_thought_when_opted_in():
    msgs = [
        _user("g"),
        _reasoning_turn(
            "The requested column is visibly filled, and the workbook has been saved."
        ),
    ]
    _, injected = flt.ensure_terminate_action(msgs)
    assert injected


def test_metadata_reader_preserves_canonical_others_outcome_fields():
    metadata = _desktop_metadata(
        {"episode_return": 1.0, "terminated": True, "truncated": False}
    )

    out = flt._metadata({"metadata": metadata})

    assert out["others"] == metadata["others"]


def test_metadata_reader_does_not_lift_canonical_others_outcome_fields():
    metadata = _desktop_metadata(
        {"episode_return": 0.25, "terminated": False, "truncated": True}
    )

    out = flt._metadata({"metadata": metadata})

    assert "episode_return" not in out
    assert "terminated" not in out
    assert "truncated" not in out
    assert out["others"] == metadata["others"]


def _write_traj(root, name, msgs, others, *, images=None, metadata=None):
    from lite.data.staging import write_partition

    task_dir = root / "train" / name
    task_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "messages": msgs,
        "metadata": metadata or _desktop_metadata(others),
    }
    if images is not None:
        record["images"] = images
    write_partition([record], task_dir / "trajectory.parquet")


def test_process_file_accepts_json_string_messages_and_metadata(tmp_path):
    import json

    import pandas as pd

    from lite.data.staging import coerce_messages, coerce_meta

    log_root = tmp_path / "logs"
    task_dir = log_root / "train" / "task_json"
    task_dir.mkdir(parents=True)
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "goal"}]},
        _asst(("click", {"coordinate": [1, 2]})),
    ]
    metadata = _desktop_metadata({"episode_return": 1.0, "terminated": True})
    pd.DataFrame({
        "messages": [json.dumps(messages)],
        "metadata": [json.dumps(metadata)],
    }).to_parquet(task_dir / "trajectory.parquet", index=False)

    dst = tmp_path / "out" / "train" / "task_json" / "trajectory.parquet"
    *_, wrote = flt._process_file(
        task_dir / "trajectory.parquet",
        dst,
        noop=NOOP,
        strip_noop_save=False,
        output_root=tmp_path / "out",
        drop_loops=False,
        drop_undo_storm=False,
        drop_no_submit=False,
        undo_storm_min=4,
        ensure_terminate=False,
        collapse_reasoning=False,
    )

    assert wrote
    row = pd.read_parquet(dst).iloc[0]
    assert coerce_messages(row["messages"]) == messages
    assert coerce_meta(row["metadata"])["others"]["episode_return"] == 1.0


def test_process_file_preexisting_terminate_metadata_is_self_contained(tmp_path):
    import json

    import pandas as pd

    log_root = tmp_path / "logs"
    msgs = [_user("g"), _asst(_click()), _user(), _asst(("terminate", {"status": "success"}))]
    _write_traj(log_root, "task_done", msgs, {"terminated": True, "episode_return": 1.0})

    src = log_root / "train" / "task_done" / "trajectory.parquet"
    dst = tmp_path / "out" / "train" / "task_done" / "trajectory.parquet"
    _, _, n_terminate_injected, _, _, _, wrote = flt._process_file(
        src, dst, noop=NOOP, strip_noop_save=False, output_root=tmp_path / "out",
        drop_loops=False, drop_undo_storm=False,
        drop_no_submit=True, undo_storm_min=4, ensure_terminate=False,
        collapse_reasoning=False,
    )
    assert wrote
    assert n_terminate_injected == 0

    row = pd.read_parquet(dst).iloc[0]
    metadata = row["metadata"]
    md = json.loads(metadata) if isinstance(metadata, str) else flt.to_plain(metadata)
    names = _schema_names(md["extra_tool_schemas"])
    assert names == ["terminate"]
    assert "function" in md["extra_tool_schemas"][0]


def _kept_count(output):
    import re

    # ANNOTATE mode keeps every trajectory; parse the "all N trajectories kept" banner.
    match = re.search(r"all (\d+) trajectories kept", output)
    assert match, output
    return int(match.group(1))


def test_process_file_normalizes_the_content_only_final_to_a_text_part(tmp_path):
    """The final no-tool turn becomes ONE plain ``text`` part, unconditionally.

    ``_done_turn`` is the real GPT-5.5 shape: prose parked in an
    ``action_description`` on a turn with NO tool_calls. That field is by
    definition "narration accompanying an action", so it is invalid there, and
    ``no_tool_call_final_text`` ignores it -- which is what made such a row render
    an EMPTY SFT target. The filter now rewrites it to
    ``{"type": "text", "text": "Done."}``.

    Note the assertion on ``type``: checking only ``content[0]["text"] == "Done."``
    passes under BOTH the old and new policy, because the old ``action_description``
    part also carried the string "Done.". The part TYPE is the whole contract.
    """
    import json

    import pandas as pd

    log_root = tmp_path / "logs"
    msgs = [_user("g"), _asst(_click()), _user(), _done_turn()]
    _write_traj(log_root, "task_done", msgs, {"terminated": True, "episode_return": 1.0})

    src = log_root / "train" / "task_done" / "trajectory.parquet"
    dst = tmp_path / "out" / "train" / "task_done" / "trajectory.parquet"
    _, _, n_terminate_injected, _, _, _, wrote = flt._process_file(
        src, dst, noop=NOOP, strip_noop_save=False, output_root=tmp_path / "out",
        drop_loops=False, drop_undo_storm=False,
        drop_no_submit=False, undo_storm_min=4, ensure_terminate=False,
        collapse_reasoning=False,
    )
    assert wrote
    assert n_terminate_injected == 0

    row = pd.read_parquet(dst).iloc[0]
    last = [m for m in coerce_messages(row["messages"]) if m["role"] == "assistant"][-1]
    # Exactly one part, and it is a plain ``text`` -- NOT action_description.
    assert len(last["content"]) == 1
    assert last["content"][0]["type"] == "text"
    assert last["content"][0]["text"] == "Done."
    assert not last.get("tool_calls")

    metadata = row["metadata"]
    md = json.loads(metadata) if isinstance(metadata, str) else flt.to_plain(metadata)
    assert "exclude_reason" not in md["others"]
    names = _schema_names(md.get("extra_tool_schemas") or [])
    assert "terminate" not in names


def test_process_file_explicit_terminate_metadata_is_self_contained(tmp_path):
    import json

    import pandas as pd

    log_root = tmp_path / "logs"
    msgs = [_user("g"), _asst(_click()), _user(), _done_turn()]
    _write_traj(log_root, "task_done", msgs, {"terminated": True, "episode_return": 1.0})

    src = log_root / "train" / "task_done" / "trajectory.parquet"
    dst = tmp_path / "out" / "train" / "task_done" / "trajectory.parquet"
    *_, wrote = flt._process_file(
        src, dst, noop=NOOP, strip_noop_save=False, output_root=tmp_path / "out",
        drop_loops=False, drop_undo_storm=False,
        drop_no_submit=False, undo_storm_min=4, ensure_terminate=True,
        collapse_reasoning=False,
    )
    assert wrote

    row = pd.read_parquet(dst).iloc[0]
    last = [m for m in coerce_messages(row["messages"]) if m["role"] == "assistant"][-1]
    assert set(last["tool_calls"][0]) == {"id", "type", "function"}
    assert tool_call_name(last["tool_calls"][0]) == "terminate"

    metadata = row["metadata"]
    md = json.loads(metadata) if isinstance(metadata, str) else flt.to_plain(metadata)
    names = _schema_names(md["extra_tool_schemas"])
    assert names == ["terminate"]
    assert "function" in md["extra_tool_schemas"][0]
    assert "tool_io_transform" not in md["others"]


def test_dry_run_writes_nothing_and_matches_real_run(tmp_path, capsys, monkeypatch):
    import sys

    log_root = tmp_path / "logs"
    msgs = [_user("g"), _asst(_click()), _user(), _done_turn()]
    _write_traj(log_root, "task_pass", msgs, {"terminated": True, "episode_return": 1.0})
    _write_traj(log_root, "task_fail", msgs, {"terminated": True, "episode_return": 0.0})

    dry_out = tmp_path / "dry_out"
    monkeypatch.setattr(sys, "argv", [
        "filter.py", "--log-root", str(log_root), "--out", str(dry_out),
        "--dry-run", "--drop-failed",
    ])
    flt.main()
    dry_output = capsys.readouterr().out
    assert "DRY RUN" in dry_output
    assert "drop_failed=True" in dry_output
    assert not dry_out.exists(), "dry-run must not create the output root"

    real_out = tmp_path / "real_out"
    monkeypatch.setattr(sys, "argv", [
        "filter.py", "--log-root", str(log_root), "--out", str(real_out),
        "--drop-failed",
    ])
    flt.main()
    real_output = capsys.readouterr().out
    assert "DRY RUN" not in real_output
    assert "drop_failed=True" in real_output
    kept = _kept_count(real_output)
    assert kept == 2  # ANNOTATE keeps every trajectory; --drop-failed is a no-op.
    assert _kept_count(dry_output) == kept
    assert len(list(real_out.rglob("trajectory.parquet"))) == kept


def test_hard_drops_typed_opt_env(tmp_path, capsys, monkeypatch):
    """A trajectory whose agent TYPED a /opt/env/ path is HARD-DROPPED (physically
    absent from the output), not tagged — it leaked an env-only tool and is
    non-reproducible on the faithful guest. A clean sibling is kept; a mere reasoning
    mention of /opt/env is NOT a trigger (only ``type`` payloads are scanned)."""
    import sys

    log_root = tmp_path / "logs"
    clean = [_user("g"), _asst(_click()), _user(), _done_turn()]
    leak = [_user("g"), _asst(_type("/opt/env/bin/pandoc a.md -o a.docx")), _user(), _done_turn()]
    # reasoning-only mention must NOT trip the drop (agent never typed it into the env)
    reason_only = [
        {"role": "assistant",
         "content": [{"type": "inline_reasoning", "text": "avoid /opt/env/ tools"}],
         "tool_calls": [_asst(_click())["tool_calls"][0]]},
        _user(), _done_turn(),
    ]
    _write_traj(log_root, "task_clean", clean, {"terminated": True, "episode_return": 1.0})
    _write_traj(log_root, "task_leak", leak, {"terminated": True, "episode_return": 1.0})
    _write_traj(log_root, "task_reason", [_user("g")] + reason_only,
                {"terminated": True, "episode_return": 1.0})

    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["filter.py", "--log-root", str(log_root), "--out", str(out)])
    flt.main()
    output = capsys.readouterr().out

    assert "HARD-DROPPED 1 trajectories" in output
    kept = {p.parent.name for p in out.rglob("trajectory.parquet")}
    assert kept == {"task_clean", "task_reason"}, kept
    assert not (out / "train" / "task_leak").exists(), "leak sample must be absent"
    # summary.json must not be copied for the dropped sample
    assert not (out / "train" / "task_leak" / "summary.json").exists()


def test_hard_drops_oob_coordinate(tmp_path, capsys, monkeypatch):
    import sys

    log_root = tmp_path / "logs"
    clean = [_user("g"), _asst(_click()), _user(), _done_turn()]
    oob = [_user("g"), _asst(_click(1500, 5)), _user(), _done_turn()]
    _write_traj(log_root, "task_clean", clean, {"terminated": True, "episode_return": 1.0})
    _write_traj(log_root, "task_oob", oob, {"terminated": True, "episode_return": 1.0})

    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["filter.py", "--log-root", str(log_root), "--out", str(out)])
    flt.main()
    output = capsys.readouterr().out

    assert "1 with OOB coordinates" in output
    kept = {p.parent.name for p in out.rglob("trajectory.parquet")}
    assert kept == {"task_clean"}, kept
    assert not (out / "train" / "task_oob").exists()


def test_filter_refuses_stale_output_without_overwrite(tmp_path, monkeypatch):
    import sys

    log_root = tmp_path / "logs"
    clean = [_user("g"), _asst(_click()), _user(), _done_turn()]
    _write_traj(log_root, "task_clean", clean, {"terminated": True, "episode_return": 1.0})

    out = tmp_path / "out"
    out.mkdir()
    stale = out / "old" / "trajectory.parquet"
    stale.parent.mkdir()
    stale.write_text("old")

    monkeypatch.setattr(sys, "argv", ["filter.py", "--log-root", str(log_root), "--out", str(out)])
    with pytest.raises(SystemExit, match="--overwrite"):
        flt.main()

    monkeypatch.setattr(sys, "argv", [
        "filter.py", "--log-root", str(log_root), "--out", str(out), "--overwrite",
    ])
    flt.main()
    assert not stale.exists()
    kept = {p.parent.name for p in out.rglob("trajectory.parquet")}
    assert kept == {"task_clean"}, kept


def test_filter_rebases_images_so_filtered_root_stages_without_raw_root(
    tmp_path, monkeypatch,
):
    import shutil
    import sys

    import pandas as pd
    from PIL import Image

    from lite.data.hf.stage import stage
    from lite.utils.image import load_images

    raw_root = tmp_path / "raw"
    image = raw_root / "train" / "task_clean" / "trajectory_images" / "000000.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (1, 1), color=(1, 2, 3)).save(image)
    metadata = LiteCUAMetadata(
        dims=("desktop", "use"),
        extra_tool_schemas=[],
        valid_actions=None,
        others={"episode_return": 1.0, "terminated": True, "task_id": "task_clean"},
    ).to_dict()
    _write_traj(
        raw_root,
        "task_clean",
        [
            {"role": "user", "content": [
                {"type": "text", "text": "g"},
                {"type": "image", "index": 0},
            ], "tool_calls": []},
            {"role": "assistant", "content": [], "tool_calls": [
                _batched_computer(_click()),
            ]},
            {"role": "tool", "tool_call_id": "call_0000", "content": [
                {"type": "text", "text": "ok"},
            ]},
            _user(img=False),
            _done_turn(),
        ],
        {"terminated": True, "episode_return": 1.0},
        images=[str(image)],
        metadata=metadata,
    )

    filtered = tmp_path / "filtered"
    monkeypatch.setattr(sys, "argv", [
        "filter.py", "--log-root", str(raw_root), "--out", str(filtered),
    ])
    flt.main()

    filtered_parquet = filtered / "train" / "task_clean" / "trajectory.parquet"
    row = pd.read_parquet(filtered_parquet).to_dict("records")[0]
    assert list(row["images"]) == ["train/task_clean/images/000000.png"]
    assert isinstance(row["messages"], str)
    idx = coerce_messages(row["messages"])[0]["content"][1]["index"]
    assert idx == 0
    assert isinstance(idx, int)
    assert (filtered_parquet.parent / "images" / "000000.png").read_bytes() == image.read_bytes()
    assert load_images(list(row["images"]), image_root=str(filtered))[0].size == (1, 1)

    shutil.rmtree(raw_root)
    name = "LiteOSWorldFilteredPortable"
    staged = tmp_path / "staged" / "cua-lite" / name
    stage([filtered], name=name, out_dir=staged, filter_expr=None)
    assert list(staged.rglob("*.parquet"))


def test_filters_unstage_json_string_messages(tmp_path, capsys, monkeypatch):
    """``hf.unstage`` reconstructs ``messages`` as a single JSON string (the HF
    dataset encoding), not list<struct>. filter.py must decode it — else
    ``list(<str>)`` iterates the JSON char-by-char and the policy checks crash on
    ``'str'.get``. Regression for the cross-machine resume final merge
    (devs/data/lite.scalecua/AGENTS.md#6): the merged root mixes native rows with
    unstaged (string) rows. Parallels test_dry_run's list-encoded run — same clean
    trajectory, same keep decision, proving the two encodings behave identically."""
    import json
    import sys

    import pandas as pd

    log_root = tmp_path / "logs"
    task_dir = log_root / "train" / "task_unstaged"
    task_dir.mkdir(parents=True)
    msgs = [_user("g"), _asst(_click()), _user(), _done_turn()]
    pd.DataFrame({
        "messages": [json.dumps(msgs)],  # <-- string, as hf.unstage writes it
        "metadata": [json.dumps(_desktop_metadata({"terminated": True, "episode_return": 1.0}))],
    }).to_parquet(task_dir / "trajectory.parquet", index=False)

    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "filter.py", "--log-root", str(log_root), "--out", str(out),
    ])
    flt.main()  # must NOT raise AttributeError: 'str' object has no attribute 'get'
    assert _kept_count(capsys.readouterr().out) == 1


# =============================================================================
# role:"tool" result layout — the observation FOLLOWS the turn, paired by
# tool_call_id. Every fixture above uses the preceding-observation layout, so the
# branch these exercise had zero coverage: a sys.settrace run over this whole
# module hit the `out.pop()` path 3 times and the role:"tool" path 0 times.
# =============================================================================

def _rt_asst(call_id: str, *actions) -> dict:
    return {"role": "assistant", "content": [], "tool_calls": [
        make_tool_call("computer", {"actions": list(actions)}, call_id=call_id),
    ]}


def _rt_result(call_id: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": [{"type": "image", "index": 0}]}


def test_role_tool_all_noop_turn_drops_its_own_result_not_the_observation():
    """The paired result FOLLOWS, so dropping the turn must drop the result.

    Popping the preceding message instead (the preceding-observation behaviour) both
    orphans the real result and deletes the goal observation.
    """
    msgs = [
        _user("goal"),
        _rt_asst("c1", {"action": "screenshot"}, {"action": "wait"}),
        _rt_result("c1"),
        _rt_asst("c2", {"action": "click", "coordinate": [3, 4]}),
        _rt_result("c2"),
    ]
    out, n_strip, n_drop = flt.strip_noop_actions(msgs, NOOP)
    assert n_strip == 2 and n_drop == 1
    assert [m["role"] for m in out] == ["user", "assistant", "tool"]
    assert [m.get("tool_call_id") for m in out if m["role"] == "tool"] == ["c2"]
    # The goal observation survives — it is what the next turn actually saw.
    assert any(p.get("text") == "goal" for p in out[0]["content"])


def test_role_tool_leading_noop_turn_keeps_the_goal():
    """A trajectory OPENING on a no-op turn must not lose its instruction."""
    msgs = [
        _user("do the thing"),
        _rt_asst("c1", {"action": "wait"}),
        _rt_result("c1"),
        _rt_asst("c2", {"action": "click", "coordinate": [1, 1]}),
        _rt_result("c2"),
    ]
    out, _, n_drop = flt.strip_noop_actions(msgs, NOOP)
    assert n_drop == 1
    assert [m["role"] for m in out] == ["user", "assistant", "tool"]
    assert any(p.get("text") == "do the thing" for p in out[0]["content"])


def test_leading_noop_preserves_reference_image_parts_not_stale_screenshot():
    metadata = {"type": "metadata", "data": {"source": "reference-image"}}
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image", "index": 1},
                {"type": "image", "index": 0},
                {"type": "text", "text": "goal"},
                metadata,
            ],
        },
        _asst(_screenshot()),
        {"role": "user", "content": [{"type": "image", "index": 2}]},
        _asst(_click()),
    ]

    out, n_strip, n_drop = flt.strip_noop_actions(msgs, NOOP)

    assert n_strip == 1 and n_drop == 1
    assert out[0]["content"] == [
        {"type": "image", "index": 1},
        {"type": "text", "text": "goal"},
        metadata,
        {"type": "image", "index": 2},
    ]


def test_role_tool_partial_batch_keeps_the_turn_and_its_result():
    """Stripping SOME children must not touch the turn or its paired result."""
    msgs = [
        _user("goal"),
        _rt_asst("c1", {"action": "click", "coordinate": [1, 2]}, {"action": "screenshot"}),
        _rt_result("c1"),
    ]
    out, n_strip, n_drop = flt.strip_noop_actions(msgs, NOOP)
    assert n_strip == 1 and n_drop == 0
    assert [m["role"] for m in out] == ["user", "assistant", "tool"]
    assert tool_call_arguments(out[1]["tool_calls"][0])["actions"] == [
        {"action": "click", "coordinate": [1, 2]},
    ]


def test_role_tool_mixed_top_level_noop_drops_only_its_result():
    msgs = [
        _user("goal"),
        {
            "role": "assistant",
            "content": [],
            "tool_calls": [
                make_tool_call("screenshot", {}, call_id="c_noop"),
                make_tool_call(
                    "computer",
                    {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                    call_id="c_real",
                ),
            ],
        },
        _rt_result("c_noop"),
        _rt_result("c_real"),
    ]
    out, n_strip, n_drop = flt.strip_noop_actions(msgs, NOOP)
    assert n_strip == 1 and n_drop == 0
    live = {tool_call_id(tc) for m in out if m["role"] == "assistant" for tc in m["tool_calls"]}
    assert live == {"c_real"}
    assert [m.get("tool_call_id") for m in out if m["role"] == "tool"] == ["c_real"]


def test_role_tool_noop_only_batch_with_response_keeps_response_and_result():
    msgs = [
        _user("goal"),
        {
            "role": "assistant",
            "content": [],
            "tool_calls": [
                make_tool_call(
                    "computer",
                    {"actions": [{"action": "screenshot"}]},
                    call_id="call_computer",
                ),
                make_tool_call("response", {"text": "ok"}, call_id="call_response"),
            ],
        },
        _rt_result("call_computer"),
        _rt_result("call_response"),
    ]

    out, n_strip, n_drop = flt.strip_noop_actions(msgs, NOOP)

    assert (n_strip, n_drop) == (1, 0)
    assert [tool_call_id(tc) for m in out if m["role"] == "assistant" for tc in m["tool_calls"]] == [
        "call_response",
    ]
    assert [m.get("tool_call_id") for m in out if m["role"] == "tool"] == ["call_response"]


def test_result_layout_is_chosen_by_role_tool_presence_not_by_call_id():
    """Assistant call id is not the discriminator for result layout.

    Keying on "the assistant call has an id" routes a preceding-observation
    trajectory into the role:"tool" branch, leaving its observation un-popped and
    duplicated.
    """
    preceding_observation_with_ids = [
        _user("goal"),
        _rt_asst("c1", {"action": "wait"}),   # id present, but no role:"tool" anywhere
        _user(),
        _rt_asst("c2", {"action": "click", "coordinate": [1, 1]}),
    ]
    out, _, n_drop = flt.strip_noop_actions(preceding_observation_with_ids, NOOP)
    assert n_drop == 1
    assert [m["role"] for m in out] == ["user", "assistant"]


# =============================================================================
# Image compaction — dropping a TURN drops no PICTURE, so the filtered row would
# otherwise publish an image no message shows and leave a hole in the index
# sequence. compact_row_images (devs/data/utils.py) closes both, in one step.
# =============================================================================

def _colored_png(path, color):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1, 1), color=color).save(path)
    return path


def test_filter_compacts_the_orphan_image_a_dropped_noop_turn_leaves(tmp_path, monkeypatch):
    """End-to-end through ``flt.main()``: 4 turns, the MIDDLE one a no-op.

    The no-op turn and its paired result are dropped, so the picture that result
    showed is referenced by nothing. The written row must carry 3 images, not 4,
    with indices ``0..2`` — and each surviving reference must still resolve to
    the SAME picture (checked by pixel bytes, since the rebased file NAMES are
    positional and would look right either way).
    """
    import sys

    import pandas as pd

    raw_root = tmp_path / "raw"
    task_dir = raw_root / "train" / "task_noop"
    colors = [(10, 0, 0), (20, 0, 0), (30, 0, 0), (40, 0, 0)]
    images = [
        _colored_png(task_dir / "trajectory_images" / f"{i:06d}.png", color)
        for i, color in enumerate(colors)
    ]
    metadata = LiteCUAMetadata(
        dims=("desktop", "use"),
        extra_tool_schemas=[],
        valid_actions=None,
        others={"episode_return": 1.0, "terminated": True, "task_id": "task_noop"},
    ).to_dict()
    _write_traj(
        raw_root,
        "task_noop",
        [
            {"role": "user", "content": [
                {"type": "image", "index": 0},
                {"type": "text", "text": "g"},
            ], "tool_calls": []},
            _rt_asst("c0", {"action": "click", "coordinate": [1, 2]}),
            {"role": "tool", "tool_call_id": "c0", "content": [{"type": "image", "index": 1}]},
            _rt_asst("c1", {"action": "screenshot"}),          # <-- the no-op turn
            {"role": "tool", "tool_call_id": "c1", "content": [{"type": "image", "index": 2}]},
            _rt_asst("c2", {"action": "click", "coordinate": [3, 4]}),
            {"role": "tool", "tool_call_id": "c2", "content": [{"type": "image", "index": 3}]},
            _done_turn(),
        ],
        {"terminated": True, "episode_return": 1.0},
        images=[str(p) for p in images],
        metadata=metadata,
    )

    filtered = tmp_path / "filtered"
    monkeypatch.setattr(sys, "argv", [
        "filter.py", "--log-root", str(raw_root), "--out", str(filtered),
    ])
    flt.main()

    filtered_parquet = filtered / "train" / "task_noop" / "trajectory.parquet"
    row = pd.read_parquet(filtered_parquet).to_dict("records")[0]
    msgs = coerce_messages(row["messages"])
    indices = [
        part["index"]
        for m in msgs
        for part in (m.get("content") or [])
        if part.get("type") == "image"
    ]

    assert len(list(row["images"])) == 3
    assert indices == [0, 1, 2]
    assert sorted(set(indices)) == list(range(len(list(row["images"]))))
    # By CONTENT: the survivors are the goal screen, the first result, and the
    # last result — the no-op's screen (colors[2]) is gone, not merely renumbered.
    written = [(filtered / rel).read_bytes() for rel in row["images"]]
    assert written == [images[0].read_bytes(), images[1].read_bytes(), images[3].read_bytes()]
    assert images[2].read_bytes() not in written
