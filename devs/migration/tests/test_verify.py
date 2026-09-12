"""Tests for migration verify.py output invariants."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from lite.core import LiteCUAMetadata
from lite.core.tools.calls import (
    make_tool_call,
    tool_call_arguments,
    tool_call_id,
    tool_call_name,
)
from lite.core.tools.results import project_tool_result_text
from lite.core.tools.schemas import make_tool_schema, tool_schema_name
from lite.data.utils.rows import validate_canonical_rows

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_migration_module(filename: str):
    path = _PROJECT_ROOT / "devs" / "migration" / filename
    spec = importlib.util.spec_from_file_location(f"cua_lite_migration_{filename}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cua_metadata_dict(
    *,
    platform: str = "desktop",
    task_type: str = "use",
    valid_actions: list[str] | None = None,
    extra_tool_schemas: list[dict] | None = None,
) -> dict:
    return LiteCUAMetadata(
        dims=(platform, task_type),
        valid_actions=list(valid_actions or []),
        extra_tool_schemas=list(extra_tool_schemas or []),
    ).to_dict()


def _canonical_sample(
    tool_calls: list[dict],
    tool_messages: list[dict] | None = None,
    *,
    extra_tool_schemas: list[dict] | None = None,
    platform: str = "desktop",
) -> dict:
    return {
        "images": ["screen0.png"],
        "metadata": _cua_metadata_dict(
            platform=platform,
            valid_actions=["click"],
            extra_tool_schemas=list(extra_tool_schemas or []),
        ),
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "task"}]},
            {
                "role": "assistant",
                "content": [],
                "tool_calls": [_canonical_tool_call(call) for call in tool_calls],
            },
            *[_canonical_tool_message(message) for message in (tool_messages or [])],
        ],
    }


def _canonical_tool_call(call: dict) -> dict:
    return make_tool_call(
        call["name"],
        call.get("arguments") or {},
        call_id=call.get("call_id"),
    )


def _canonical_tool_message(message: dict) -> dict:
    if message.get("role") != "tool" or "tool_call_id" in message:
        return message
    out = dict(message)
    out["tool_call_id"] = out.pop("call_id")
    return out


def _tool_result(call_id: str, content: list[dict]) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _schema(name: str) -> dict:
    from lite.core.tools.extra_tools import (
        BASH_TOOL_NAME,
        LiteBrowserNavToolSet,
        LiteFinishToolSet,
        LiteShellToolSet,
        make_open_app_tool,
    )

    if name == "bash":
        return LiteShellToolSet.get_tool_schema(BASH_TOOL_NAME)
    if name == "open_app":
        return make_open_app_tool()
    if name in LiteFinishToolSet.get_tool_names():
        return LiteFinishToolSet.get_tool_schema(name)
    if name in LiteBrowserNavToolSet.get_tool_names():
        return LiteBrowserNavToolSet.get_tool_schema(name)
    return make_tool_schema(
        name,
        parameters={"type": "object", "properties": {}, "required": []},
    )


def test_verify_rejects_provider_envelope_leaks_and_swallowed_standalone():
    verify = _load_migration_module("verify.py")

    provider_envelope = _canonical_sample(
        [
            {"call_id": "call_0000", "name": "click", "arguments": {}},
        ]
    )
    provider_envelope["messages"][1]["tool_calls"] = [
        {"type": "function", "function": {"name": "click", "arguments": {}}},
    ]
    with pytest.raises(verify.VerificationError, match="missing non-empty id"):
        verify.verify_lite_sample(provider_envelope)

    with pytest.raises(verify.VerificationError, match="not valid for computer"):
        verify.verify_lite_sample(
            _canonical_sample(
                [
                    {
                        "call_id": "call_0000",
                        "name": "computer",
                        "arguments": {
                            "actions": [
                                {"action": "click", "coordinate": [10, 20]},
                                {"action": "terminate", "status": "success"},
                            ]
                        },
                    },
                ]
            )
        )


@pytest.mark.parametrize(
    "platform,tool_name,action,match",
    [
        (
            "desktop",
            "computer",
            {"action": "open_app", "app_name": "Settings"},
            "not valid for computer",
        ),
        (
            "desktop",
            "computer",
            {"action": "tap", "coordinate": [10, 20]},
            "not valid for computer",
        ),
        (
            "mobile",
            "mobile",
            {"action": "click", "coordinate": [10, 20]},
            "not valid for mobile",
        ),
    ],
)
def test_verify_rejects_batch_children_that_are_not_canonical_actions(
    platform: str,
    tool_name: str,
    action: dict,
    match: str,
):
    verify = _load_migration_module("verify.py")

    sample = _canonical_sample(
        [
            {
                "call_id": "call_0000",
                "name": tool_name,
                "arguments": {"actions": [action]},
            }
        ],
        platform=platform,
    )
    sample["metadata"]["valid_actions"] = ["tap"] if platform == "mobile" else ["click"]

    with pytest.raises(verify.VerificationError, match=match):
        verify.verify_lite_sample(sample)


@pytest.mark.parametrize(
    "platform,tool_name,actions",
    [
        (
            "desktop",
            "computer",
            [
                {"action": "click", "coordinate": [10, 20]},
                {"action": "type", "text": "hi"},
            ],
        ),
        (
            "mobile",
            "mobile",
            [
                {"action": "tap", "coordinate": [10, 20]},
                {
                    "action": "swipe",
                    "start_coordinate": [10, 200],
                    "coordinate": [10, 20],
                },
            ],
        ),
    ],
)
def test_verify_allows_action_batch_actions_for_matching_batch_type(
    platform: str,
    tool_name: str,
    actions: list[dict],
):
    verify = _load_migration_module("verify.py")

    sample = _canonical_sample(
        [
            {
                "call_id": "call_0000",
                "name": tool_name,
                "arguments": {"actions": actions},
            }
        ],
        [_tool_result("call_0000", [{"type": "image", "index": 0}])],
        platform=platform,
    )
    sample["metadata"]["valid_actions"] = [action["action"] for action in actions]

    parsed = verify.verify_lite_sample(sample)
    parsed_actions = tool_call_arguments(parsed["messages"][1]["tool_calls"][0])["actions"]
    assert [action["action"] for action in parsed_actions] == [
        action["action"] for action in actions
    ]


@pytest.mark.parametrize("bad_key", ["plus", "minus", "equal", "comma"])
def test_verify_rejects_noncanonical_key_tokens_in_action_batch(bad_key: str):
    verify = _load_migration_module("verify.py")

    sample = _canonical_sample(
        [
            {
                "call_id": "call_0000",
                "name": "computer",
                "arguments": {
                    "actions": [{"action": "key", "keys": ["ctrl", bad_key]}],
                },
            }
        ],
    )
    sample["metadata"]["valid_actions"] = ["key"]

    with pytest.raises(verify.VerificationError, match=f"noncanonical key token {bad_key!r}"):
        verify.verify_lite_sample(sample)


def test_verify_rejects_string_key_payload_in_top_level_grounding_action():
    verify = _load_migration_module("verify.py")

    sample = _canonical_sample(
        [{"call_id": "call_0000", "name": "key", "arguments": {"keys": "ctrl+s"}}],
    )
    sample["metadata"]["dims"][1] = "grounding.action"
    sample["metadata"]["valid_actions"] = None

    with pytest.raises(
        verify.VerificationError,
        match=r"function\.arguments\.keys must be a non-empty list\[str\]",
    ):
        verify.verify_lite_sample(sample)


def test_verify_rejects_noncanonical_key_tokens_in_top_level_grounding_action():
    verify = _load_migration_module("verify.py")

    sample = _canonical_sample(
        [{"call_id": "call_0000", "name": "key", "arguments": {"keys": ["ctrl", "plus"]}}],
    )
    sample["metadata"]["dims"][1] = "grounding.action"
    sample["metadata"]["valid_actions"] = None

    with pytest.raises(verify.VerificationError, match="noncanonical key token 'plus'"):
        verify.verify_lite_sample(sample)


def test_verify_rejects_non_string_key_token_in_top_level_grounding_action():
    verify = _load_migration_module("verify.py")

    sample = _canonical_sample(
        [{"call_id": "call_0000", "name": "key", "arguments": {"keys": ["ctrl", 1]}}],
    )
    sample["metadata"]["dims"][1] = "grounding.action"
    sample["metadata"]["valid_actions"] = None

    with pytest.raises(
        verify.VerificationError,
        match=r"function\.arguments\.keys\[1\] must be str, got int",
    ):
        verify.verify_lite_sample(sample)


@pytest.mark.parametrize(
    "platform,name,arguments",
    [
        ("desktop", "click", {"coordinate": [10, 20]}),
        ("mobile", "tap", {"coordinate": [10, 20]}),
        # ``browser`` maps to the ``computer`` wrapper too, so a top-level bare
        # action is non-canonical there as well.
        ("browser", "click", {"coordinate": [10, 20]}),
        ("browser", "scroll", {"direction": "down", "amount": 3}),
    ],
)
def test_verify_rejects_top_level_use_actions_even_with_tool_result(
    platform: str,
    name: str,
    arguments: dict,
):
    verify = _load_migration_module("verify.py")

    sample = _canonical_sample(
        [{"call_id": "call_0000", "name": name, "arguments": arguments}],
        [_tool_result("call_0000", [{"type": "image", "index": 0}])],
        platform=platform,
    )
    sample["metadata"]["valid_actions"] = [name]

    with pytest.raises(verify.VerificationError, match="top-level GUI action"):
        verify.verify_lite_sample(sample)


def test_verify_rejects_legacy_web_platform_output():
    verify = _load_migration_module("verify.py")
    sample = _canonical_sample(
        [
            {
                "call_id": "call_0000",
                "name": "computer",
                "arguments": {
                    "actions": [
                        {"action": "click", "coordinate": [10, 20]},
                    ]
                },
            },
        ],
        [_tool_result("call_0000", [{"type": "image", "index": 0}])],
    )
    sample["metadata"]["dims"][0] = "web"

    with pytest.raises(verify.VerificationError, match="must be 'browser', not 'web'"):
        verify.verify_lite_sample(sample)


def test_verify_rejects_image_result_for_bash_and_accepts_text_result():
    verify = _load_migration_module("verify.py")

    image_result = _canonical_sample(
        [{"call_id": "call_0000", "name": "bash", "arguments": {"command": "pwd"}}],
        [_tool_result("call_0000", [{"type": "image", "index": 0}])],
        extra_tool_schemas=[_schema("bash")],
    )
    with pytest.raises(verify.VerificationError, match="text-result-only"):
        verify.verify_lite_sample(image_result)

    text_result = _canonical_sample(
        [{"call_id": "call_0000", "name": "bash", "arguments": {"command": "pwd"}}],
        [_tool_result("call_0000", [{"type": "text", "text": "/tmp"}])],
        extra_tool_schemas=[_schema("bash")],
    )
    assert verify.verify_lite_sample(text_result)["messages"][2]["tool_call_id"] == "call_0000"


def test_verify_rejects_message_image_index_without_matching_image():
    verify = _load_migration_module("verify.py")

    sample = _canonical_sample(
        [
            {
                "call_id": "call_0000",
                "name": "computer",
                "arguments": {
                    "actions": [
                        {"action": "click", "coordinate": [10, 20]},
                    ]
                },
            },
        ],
        [_tool_result("call_0000", [{"type": "text", "text": "ok"}])],
    )
    sample["messages"][0]["content"] = [{"type": "image", "index": 1}]

    with pytest.raises(verify.VerificationError, match="out of range"):
        verify.verify_lite_sample(sample)


@pytest.mark.parametrize(
    "name",
    ["terminate", "response", "open_app", "goto", "bash", "ask_user", "report_infeasible", "done"],
)
def test_verify_rejects_standalone_extra_call_without_matching_schema(name: str):
    verify = _load_migration_module("verify.py")

    sample = _canonical_sample(
        [
            {"call_id": "call_0000", "name": name, "arguments": {}},
        ]
    )

    with pytest.raises(verify.VerificationError, match="missing from metadata.extra_tool_schemas"):
        verify.verify_lite_sample(sample)


def test_verify_rejects_extra_tool_invalid_arguments_even_with_error_feedback():
    verify = _load_migration_module("verify.py")
    schema = make_tool_schema(
        "search",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    sample = _canonical_sample(
        [{"call_id": "call_0000", "name": "search", "arguments": {"query": 3}}],
        [
            _tool_result(
                "call_0000",
                [{
                    "type": "text",
                    "text": project_tool_result_text(
                        "current observation",
                        "invalid arguments for search: search.arguments.query must be a string",
                    ),
                }],
            )
        ],
        extra_tool_schemas=[schema],
        platform="browser",
    )

    with pytest.raises(verify.VerificationError, match="arguments do not match"):
        verify.verify_lite_sample(sample)


@pytest.mark.parametrize(
    "tool_calls,extra_tool_schemas",
    [
        (
            [
                {
                    "call_id": "call_0000",
                    "name": "computer",
                    "arguments": {
                        "actions": [
                            {"action": "click", "coordinate": [10, 20]},
                        ]
                    },
                }
            ],
            [],
        ),
        (
            [{"call_id": "call_0000", "name": "goto", "arguments": {"url": "https://example.com"}}],
            [
                _schema("goto"),
            ],
        ),
    ],
)
def test_verify_rejects_screen_producing_call_without_tool_result(
    tool_calls: list[dict],
    extra_tool_schemas: list[dict],
):
    verify = _load_migration_module("verify.py")

    sample = _canonical_sample(tool_calls, extra_tool_schemas=extra_tool_schemas)
    # The rule only binds MID-episode: a later action turn (which does carry its
    # own result) makes the first turn's missing observation a real gap.
    sample["messages"] += [
        {
            "role": "assistant",
            "content": [],
            "tool_calls": [
                make_tool_call(
                    "computer",
                    {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                    call_id="call_0009",
                ),
            ],
        },
        _tool_result("call_0009", [{"type": "image", "index": 0}]),
    ]

    with pytest.raises(verify.VerificationError, match="missing a tool result"):
        verify.verify_lite_sample(sample)


@pytest.mark.parametrize(
    "tool_calls,extra_tool_schemas",
    [
        (
            [
                {
                    "call_id": "call_0000",
                    "name": "computer",
                    "arguments": {
                        "actions": [
                            {"action": "click", "coordinate": [10, 20]},
                        ]
                    },
                }
            ],
            [],
        ),
        (
            [
                {
                    "call_id": "call_0000",
                    "name": "computer",
                    "arguments": {
                        "actions": [
                            {"action": "click", "coordinate": [10, 20]},
                        ]
                    },
                },
                {"call_id": "call_0001", "name": "terminate", "arguments": {"status": "failure"}},
            ],
            [_schema("terminate")],
        ),
    ],
)
def test_verify_allows_screen_producing_call_in_the_final_turn(
    tool_calls: list[dict],
    extra_tool_schemas: list[dict],
):
    """An OLD published row folded its terminate onto the last action turn and
    kept no screenshot after it, so migration has no observation to pair.

    Migration verification and the canonical stage gate both accept EOF as an
    unobserved SFT target. Publish policy may still filter incomplete historical
    rows; see the terminal-turn note in `devs/migration/AGENTS.md`.
    """
    verify = _load_migration_module("verify.py")

    sample = _canonical_sample(tool_calls, extra_tool_schemas=extra_tool_schemas)

    parsed_calls = verify.verify_lite_sample(sample)["messages"][1]["tool_calls"]
    assert [
        (tool_call_id(call), tool_call_name(call), tool_call_arguments(call))
        for call in parsed_calls
    ] == [(call.get("call_id"), call["name"], call.get("arguments") or {}) for call in tool_calls]
    validate_canonical_rows([sample], "migration/final-eof")


@pytest.mark.parametrize("action_name", ["key", "key_down", "key_up", "hold_key"])
def test_verify_rejects_empty_key_lists_inside_action_batch(action_name: str):
    verify = _load_migration_module("verify.py")
    action = {"action": action_name, "keys": []}
    if action_name == "hold_key":
        action["duration"] = 0.25
    sample = _canonical_sample(
        [
            {
                "call_id": "call_0000",
                "name": "computer",
                "arguments": {"actions": [action]},
            },
        ],
    )

    with pytest.raises(verify.VerificationError, match=r"actions\[0\]\.keys must not be empty"):
        verify.verify_lite_sample(sample)


@pytest.mark.parametrize(
    "task_type,tool_name",
    [
        ("grounding.action", "click"),
        ("grounding.point", "point"),
        ("grounding.bbox", "bbox"),
    ],
)
def test_verify_does_not_require_observations_for_non_use_rows(task_type: str, tool_name: str):
    """Grounding/understanding rows are SFT labels, not rollout turns.

    An aguvis ``grounding.action`` row packs N independent single-step samples
    as ``[user, assistant] x N``; there are no observations by construction.
    """
    verify = _load_migration_module("verify.py")

    sample = _canonical_sample(
        [
            {"call_id": "call_0000", "name": tool_name, "arguments": {"coordinate": [10, 20]}},
        ]
    )
    sample["metadata"]["dims"][1] = task_type
    sample["messages"] += [
        {"role": "user", "content": [{"type": "text", "text": "task 2"}]},
        {
            "role": "assistant",
            "content": [],
            "tool_calls": [
                make_tool_call(tool_name, {"coordinate": [30, 40]}, call_id="call_0001"),
            ],
        },
    ]

    assert len(verify.verify_lite_sample(sample)["messages"]) == 4


def test_verify_allows_finish_call_without_tool_result_when_schema_backed():
    verify = _load_migration_module("verify.py")

    sample = _canonical_sample(
        [{"call_id": "call_0000", "name": "terminate", "arguments": {"status": "success"}}],
        extra_tool_schemas=[_schema("terminate")],
    )

    assert (
        tool_call_name(verify.verify_lite_sample(sample)["messages"][1]["tool_calls"][0])
        == "terminate"
    )


def test_verify_rejects_non_assistant_non_empty_tool_calls_but_tolerates_padding():
    verify = _load_migration_module("verify.py")
    sample = _canonical_sample(
        [
            {
                "call_id": "call_0000",
                "name": "computer",
                "arguments": {
                    "actions": [
                        {"action": "click", "coordinate": [10, 20]},
                    ]
                },
            },
        ],
        [_tool_result("call_0000", [{"type": "text", "text": "ok"}])],
    )

    padded = json.loads(json.dumps(sample))
    padded["messages"][0]["tool_calls"] = []
    padded["messages"][2]["tool_calls"] = None
    assert (
        tool_call_name(verify.verify_lite_sample(padded)["messages"][1]["tool_calls"][0])
        == "computer"
    )

    sample["messages"][0]["tool_calls"] = [
        {"call_id": "call_user", "name": "response", "arguments": {"text": "bad"}},
    ]
    with pytest.raises(verify.VerificationError, match="non-assistant"):
        verify.verify_lite_sample(sample)


def test_verify_raw_boundary_is_not_strict_canonical_publish_proof():
    verify = _load_migration_module("verify.py")
    sample = _canonical_sample(
        [
            {
                "call_id": "call_0000",
                "name": "computer",
                "arguments": {
                    "actions": [
                        {"action": "click", "coordinate": [10, 20]},
                    ]
                },
            },
        ],
        [_tool_result("call_0000", [{"type": "text", "text": "ok"}])],
    )

    verify.verify_lite_sample(sample)
    with pytest.raises(ValueError, match="trailing role:tool"):
        validate_canonical_rows([sample], "migration/raw-boundary")


def test_verify_allows_computer_then_terminate_with_one_computer_result():
    verify = _load_migration_module("verify.py")

    sample = _canonical_sample(
        [
            {
                "call_id": "call_0000",
                "name": "computer",
                "arguments": {
                    "actions": [
                        {"action": "click", "coordinate": [10, 20]},
                    ]
                },
            },
            {"call_id": "call_0001", "name": "terminate", "arguments": {"status": "success"}},
        ],
        [_tool_result("call_0000", [{"type": "image", "index": 0}])],
        extra_tool_schemas=[_schema("terminate")],
    )

    parsed = verify.verify_lite_sample(sample)
    assert [msg.get("tool_call_id") for msg in parsed["messages"] if msg["role"] == "tool"] == [
        "call_0000"
    ]


@pytest.mark.parametrize("filename", ["upgrade.py", "verify.py"])
def test_migration_action_catalogs_are_supersets_of_tool_surface(filename: str):
    """Migration is a raw-boundary reader, so it may ACCEPT more spellings than
    canonical emits — but never fewer.

    Today the two are equal: old rows already stored Lite action names
    (desktop ``double_click``/``hotkey``/``move``/``keypress`` are normalized by
    ``convert_tool_calls_from_agent`` before storage; mobile ``double_tap`` is
    not normalized by anything -- it is a name no family declares or emits, see
    the note on ``upgrade.DESKTOP_GUI_ACTIONS``), so there are no legacy input
    aliases. This pins the direction that actually breaks
    data: if ``lite.core.tools.action_space`` grows an action and migration does not
    learn it, that action stops batching and silently becomes a
    schema-less standalone tool.
    """
    from lite.core.tools.action_space import (
        LiteDesktopActionSet,
        LiteMobileActionSet,
        lite_action_names_by_action_batch_tool,
    )

    module = _load_migration_module(filename)

    assert LiteDesktopActionSet.get_action_names() <= set(module.DESKTOP_GUI_ACTIONS)
    assert LiteMobileActionSet.get_action_names() <= set(module.MOBILE_GUI_ACTIONS)
    assert lite_action_names_by_action_batch_tool() == module.ACTION_NAMES_BY_ACTION_BATCH_TOOL

    # Any extra spelling must be a documented accept-only legacy alias, never a
    # name migration is free to invent. There are none today.
    assert set(module.DESKTOP_GUI_ACTIONS) - LiteDesktopActionSet.get_action_names() == set()
    assert set(module.MOBILE_GUI_ACTIONS) - LiteMobileActionSet.get_action_names() == set()


@pytest.mark.parametrize("filename", ["upgrade.py", "verify.py"])
def test_migration_schema_free_names_keep_legacy_grounding_action_bare_calls(
    filename: str,
):
    """Old grounding.action labels are migration-only compatibility."""
    from lite.core.tools.action_space import lite_builtin_tool_names_for_metadata

    module = _load_migration_module(filename)
    desktop_meta = LiteCUAMetadata(dims=("desktop", "grounding.action")).to_dict()
    mobile_meta = LiteCUAMetadata(dims=("mobile", "grounding.action")).to_dict()

    assert "click" not in lite_builtin_tool_names_for_metadata(
        LiteCUAMetadata.from_dict(desktop_meta)
    )
    assert "tap" not in lite_builtin_tool_names_for_metadata(LiteCUAMetadata.from_dict(mobile_meta))
    assert {"computer", "click", "type"} <= module.schema_free_names(desktop_meta)
    assert {"mobile", "tap", "swipe"} <= module.schema_free_names(mobile_meta)


@pytest.mark.parametrize("filename", ["upgrade.py", "verify.py"])
def test_migration_finish_nav_and_action_batch_catalogs_use_shared_helpers(filename: str):
    """Canonical finish/nav/action-batch names come from the action-space catalog.

    Any compatibility-only names must stay separately pinned in migration code
    so they cannot silently widen the canonical catalogs.
    """
    from lite.core.tools.action_space import LITE_ACTION_BATCH_TOOL_NAMES
    from lite.core.tools.extra_tools import (
        BASH_TOOL_NAME,
        LiteBrowserNavToolSet,
        LiteFinishToolSet,
        make_open_app_tool,
    )

    module = _load_migration_module(filename)
    app_launch_name = tool_schema_name(make_open_app_tool())

    assert module.FINISH_TOOL_NAMES == LiteFinishToolSet.get_tool_names()
    assert module.BROWSER_NAV_TOOLS == LiteBrowserNavToolSet.get_tool_names()
    assert module.APP_LAUNCH_TOOL_NAME == app_launch_name
    assert module.APP_LAUNCH_TOOLS == frozenset({app_launch_name})
    assert module.NAV_TOOLS == LiteBrowserNavToolSet.get_tool_names() | frozenset({app_launch_name})
    assert module.LITE_ACTION_BATCH_TOOL_NAMES == LITE_ACTION_BATCH_TOOL_NAMES
    assert not hasattr(module, "ACTION_BATCH_TOOLS")
    assert not hasattr(module, "".join(("WEB", "_NAV_TOOLS")))

    if filename == "upgrade.py":
        assert module.MIGRATION_LOCAL_TEXT_RESULT_TOOLS == frozenset({BASH_TOOL_NAME, "ask_user"})
        for name in LiteFinishToolSet.get_tool_names():
            assert module._default_schema(name) == LiteFinishToolSet.get_tool_schema(name)
        assert module._default_schema(app_launch_name) == make_open_app_tool()
        for name in LiteBrowserNavToolSet.get_tool_names():
            assert module._default_schema(name) == LiteBrowserNavToolSet.get_tool_schema(name)
    else:
        assert module.TEXT_RESULT_ONLY_TOOLS == frozenset({BASH_TOOL_NAME, "ask_user"})
        assert module.MIGRATION_RESULTLESS_STANDALONE_TOOLS == frozenset(
            {
                "done",
                "report_infeasible",
                "send_msg_to_user",
            }
        )
        assert module.MIGRATION_RESULTLESS_STANDALONE_TOOLS.isdisjoint(
            LiteFinishToolSet.get_tool_names()
        )
        assert (
            module.TERMINAL_STANDALONE_TOOLS
            == LiteFinishToolSet.get_tool_names() | module.MIGRATION_RESULTLESS_STANDALONE_TOOLS
        )


def test_migration_upgrade_and_verify_share_one_action_catalog():
    """The upgrader and the verifier must not drift apart either."""
    upgrade = _load_migration_module("upgrade.py")
    verify = _load_migration_module("verify.py")

    assert set(upgrade.DESKTOP_GUI_ACTIONS) == set(verify.DESKTOP_GUI_ACTIONS)
    assert set(upgrade.MOBILE_GUI_ACTIONS) == set(verify.MOBILE_GUI_ACTIONS)
