"""Canonical and raw row publication validator tests."""

from __future__ import annotations

import json
from typing import Any

import pytest

from lite.core import LiteCUAMetadata, LiteGenericMetadata
from lite.core.errors import LiteContractError
from lite.core.messages.final import (
    CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY,
    MODEL_OUTPUT_ERROR_KEY,
)
from lite.core.tools.calls import make_tool_call
from lite.core.tools.extra_tools import LiteBrowserNavToolSet
from lite.core.tools.results import project_tool_result_text
from lite.core.tools.schemas import make_tool_schema
from lite.data.utils.rows import (
    validate_canonical_rows,
    validate_raw_rollout_rows,
)


def _tc(name: str, arguments: dict | None = None, *, call_id: str) -> dict:
    return make_tool_call(name, arguments, call_id=call_id)


@pytest.mark.parametrize("role", ["assistant", "system"])
def test_validate_canonical_rows_rejects_image_parts_on_unbound_roles(role):
    rows = [{
        "images": ["image.png"],
        "messages": [{"role": role, "content": [{"type": "image", "index": 0}]}],
        "metadata": LiteCUAMetadata(
            dims=("browser", "understanding"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    }]

    with pytest.raises(ValueError, match=f"role:{role} messages cannot carry"):
        validate_canonical_rows(rows, f"unit/{role}-image")


def test_validate_canonical_rows_rejects_standalone_call_without_schema():
    rows = [{
        "images": [],
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [_tc(
                    "goto",
                    {"url": "https://example.com"},
                    call_id="call_goto",
                )],
            },
            {
                "role": "tool",
                "tool_call_id": "call_goto",
                "content": [{"type": "text", "text": "ok"}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    }]

    with pytest.raises(ValueError, match="missing from metadata\\.extra_tool_schemas"):
        validate_canonical_rows(rows, "unit/missing-schema")


def _desktop_key_row(key: Any) -> dict:
    return {
        "images": ["screen.png"],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image", "index": 0}],
            },
            {
                "role": "assistant",
                "tool_calls": [_tc("key", {"keys": [key]}, call_id="call_key")],
            },
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
        ).to_dict(),
    }


def test_validate_canonical_rows_rejects_empty_standalone_key_list() -> None:
    row = _desktop_key_row("+")
    row["messages"][1]["tool_calls"][0]["function"]["arguments"]["keys"] = []

    with pytest.raises(ValueError, match="key.keys must not be empty"):
        validate_canonical_rows([row], "unit/empty-key-list")


@pytest.mark.parametrize(
    "key",
    [
        "ac",
        "CTRL",
        "plus",
        "minus",
        "equal",
        "comma",
        "ctrl+a",
        "",
        " ",
        "\n",
        "\t",
        "\r",
        "\x1b",
        "\x00",
    ],
)
def test_validate_canonical_rows_rejects_noncanonical_standalone_keys(key: Any) -> None:
    rows = [_desktop_key_row(key)]

    with pytest.raises(ValueError, match="noncanonical or unsupported key"):
        validate_canonical_rows(rows, "unit/bad-key")


@pytest.mark.parametrize("key", ["+", "-", "=", ","])
def test_validate_canonical_rows_accepts_standalone_key_glyphs_until_route_gate(
    key: str,
) -> None:
    with pytest.raises(ValueError, match="standalone but missing"):
        validate_canonical_rows([_desktop_key_row(key)], "unit/glyph-key")


def test_validate_canonical_rows_rejects_trailing_tool_result():
    """A canonical row may NOT end on an observation with no later decision,
    unless it records that the episode ended.

    A ``max_steps`` cutoff carries ``truncated: True`` -- and this row's
    ``others`` is empty, so nothing distinguishes "the env screenshotted, scored
    and reported the cutoff" from a capture that stopped mid-episode. The
    evidence-gated exception is pinned separately by
    ``..._allows_terminal_paired_tool_result_at_eof``, which exists only because
    this test rejects the unevidenced shape; the pair is meaningless if either
    half flips.

    This test was briefly renamed to ``..._accepts_...`` and its assertion
    dropped, to make a regressing validator pass. Three older tests in two other
    files pin the identical message list as REJECTED, so accepting it here would
    have made publication more permissive than the debug log gate.
    """
    rows = [{
        "images": [],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "go"}]},
            {
                "role": "assistant",
                "tool_calls": [_tc(
                    "computer",
                    {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                    call_id="call_0000",
                )],
            },
            {
                "role": "tool",
                "tool_call_id": "call_0000",
                "content": [{"type": "text", "text": "ok"}],
            },
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    }]

    with pytest.raises(ValueError, match="trailing role:tool result"):
        validate_canonical_rows(rows, "unit/trailing-tool")


@pytest.mark.parametrize("validate", [validate_canonical_rows, validate_raw_rollout_rows])
def test_validate_rows_reject_empty_tool_result_content(validate):
    rows = [{
        "images": [],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "go"}]},
            {
                "role": "assistant",
                "tool_calls": [_tc(
                    "computer",
                    {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                    call_id="call_0000",
                )],
            },
            {
                "role": "tool",
                "tool_call_id": "call_0000",
                "content": [],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    }]

    with pytest.raises(ValueError, match="role:tool content must be non-empty"):
        validate(rows, "unit/empty-tool-content")


def test_validate_canonical_rows_rejects_orphan_tool_result():
    rows = [{
        "images": [],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "go"}]},
            {
                "role": "tool",
                "tool_call_id": "missing",
                "content": [{"type": "text", "text": "ok"}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    }]

    with pytest.raises(ValueError, match="orphan role:tool"):
        validate_canonical_rows(rows, "unit/orphan-tool")


def test_validate_canonical_rows_rejects_pending_tool_result_before_final():
    rows = [{
        "images": [],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "go"}]},
            {
                "role": "assistant",
                "tool_calls": [_tc(
                    "computer",
                    {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                    call_id="call_0000",
                )],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    }]

    with pytest.raises(ValueError, match="before role:tool result"):
        validate_canonical_rows(rows, "unit/pending-tool")


def test_validate_canonical_rows_rejects_use_final_action_then_done_without_tool_result():
    rows = [{
        "images": [],
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [_tc("response", {"text": "Done."}, call_id="call_0000")],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[make_tool_schema(
                "response",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )],
            valid_actions=None,
            others={},
        ).to_dict(),
    }]

    with pytest.raises(ValueError, match="a finish call ends the row"):
        validate_canonical_rows(rows, "unit/final-action-done")


def test_validate_canonical_rows_allows_eof_tool_call_without_result():
    rows = [{
        "images": [],
        "messages": [{
            "role": "assistant",
            "tool_calls": [_tc(
                "computer",
                {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                call_id="call_0000",
            )],
        }],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    }]

    validate_canonical_rows(rows, "unit/eof-pending")


def test_validate_raw_rollout_rows_allows_eof_tool_call_without_result():
    rows = [{
        "images": [],
        "messages": [{
            "role": "assistant",
            "tool_calls": [_tc(
                "computer",
                {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                call_id="call_0000",
            )],
        }],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    }]

    validate_raw_rollout_rows(rows, "unit/raw-export-eof")


def test_validate_canonical_rows_allows_terminal_eof_schema_free_tool_without_result():
    rows = [{
        "images": [],
        "messages": [{
            "role": "assistant",
            "tool_calls": [_tc(
                "computer",
                {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                call_id="call_0000",
            )],
        }],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={"terminated": True},
        ).to_dict(),
    }]

    validate_canonical_rows(rows, "unit/terminal-eof")


def test_validate_canonical_rows_rejects_unpaired_tool_before_next_assistant_terminal():
    rows = [{
        "images": [],
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [_tc(
                    "computer",
                    {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                    call_id="call_0000",
                )],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={"terminated": True},
        ).to_dict(),
    }]

    with pytest.raises(ValueError, match="before role:tool result"):
        validate_canonical_rows(rows, "unit/mid-pending")


def test_validate_canonical_rows_allows_terminal_paired_tool_result_at_eof():
    rows = [{
        "images": [],
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [_tc(
                    "computer",
                    {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                    call_id="call_0000",
                )],
            },
            {
                "role": "tool",
                "tool_call_id": "call_0000",
                "content": [{"type": "text", "text": "done"}],
            },
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={"terminated": True},
        ).to_dict(),
    }]

    validate_canonical_rows(rows, "unit/terminal-paired-tool")


def test_validate_canonical_rows_allows_truncated_paired_tool_result_at_eof():
    rows = [{
        "images": [],
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [_tc(
                    "computer",
                    {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                    call_id="call_0000",
                )],
            },
            {
                "role": "tool",
                "tool_call_id": "call_0000",
                "content": [{"type": "text", "text": "done"}],
            },
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={"truncated": True},
        ).to_dict(),
    }]

    validate_canonical_rows(rows, "unit/nested-terminal-paired-tool")


@pytest.mark.parametrize(
    "name,arguments,schema",
    [
        (
            "response",
            {"text": "Done."},
            make_tool_schema(
                "response",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            ),
        ),
        (
            "terminate",
            {"status": "success"},
            make_tool_schema(
                "terminate",
                parameters={
                    "type": "object",
                    "properties": {"status": {"type": "string"}},
                    "required": ["status"],
                },
            ),
        ),
    ],
)
def test_validate_canonical_rows_allows_terminal_finish_tool_at_eof_without_result(
    name,
    arguments,
    schema,
):
    rows = [{
        "images": [],
        "messages": [{
            "role": "assistant",
            "tool_calls": [_tc(name, arguments, call_id="call_0000")],
        }],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[schema],
            valid_actions=None,
            others={"terminated": True},
        ).to_dict(),
    }]

    validate_canonical_rows(rows, f"unit/{name}-terminal-eof")


@pytest.mark.parametrize(
    "name,arguments,schema",
    [
        (
            "response",
            {"text": "Done."},
            make_tool_schema(
                "response",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            ),
        ),
        (
            "terminate",
            {"status": "success"},
            make_tool_schema(
                "terminate",
                parameters={
                    "type": "object",
                    "properties": {"status": {"type": "string"}},
                    "required": ["status"],
                },
            ),
        ),
    ],
)
def test_validate_canonical_rows_allows_schema_backed_finish_tool_at_eof_without_evidence(
    name,
    arguments,
    schema,
):
    rows = [{
        "images": [],
        "messages": [{
            "role": "assistant",
            "tool_calls": [_tc(name, arguments, call_id="call_0000")],
        }],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[schema],
            valid_actions=None,
            others={},
        ).to_dict(),
    }]

    validate_canonical_rows(rows, f"unit/{name}-terminal-eof-no-evidence")


@pytest.mark.parametrize(
    "validate",
    [validate_canonical_rows, validate_raw_rollout_rows],
    ids=["canonical", "raw"],
)
@pytest.mark.parametrize("position", ["eof", "midrow"])
def test_a_dialect_finish_spelling_in_tool_calls_is_rejected_under_both_contracts(
    validate,
    position,
):
    """The raw contract has no claim on a dialect finish spelling either.

    Position must not matter: at EOF the name used to be accepted and mid-row
    rejected, which is the same name judged two ways.
    """
    messages = [{
        "role": "assistant",
        "tool_calls": [_tc("answer", {"text": "Done."}, call_id="call_answer")],
    }]
    if position == "midrow":
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]}
        )
    rows = [{
        "images": [],
        "messages": messages,
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={"agent_id": "qwen3_vl"},
        ).to_dict(),
    }]

    with pytest.raises(ValueError, match="'answer' is dialect-only.*'response'"):
        validate(rows, f"unit/dialect-answer-{position}")


def _extra_tool_schema(
    name: str,
    *,
    properties: dict | None = None,
    required: list[str] | None = None,
) -> dict:
    return make_tool_schema(
        name,
        parameters={
            "type": "object",
            "properties": properties or {},
            "required": required or [],
        },
    )


@pytest.mark.parametrize(
    "name,arguments,platform,extra_tool_schemas",
    [
        (
            "computer",
            {"actions": [{"action": "click", "coordinate": [1, 2]}]},
            "desktop",
            [],
        ),
        (
            "goto",
            {"url": "https://example.com"},
            "browser",
            LiteBrowserNavToolSet.get_tool_schemas(include=["goto"]),
        ),
        (
            "bash",
            {"command": "pwd"},
            "desktop",
            [
                _extra_tool_schema(
                    "bash",
                    properties={"command": {"type": "string"}},
                    required=["command"],
                )
            ],
        ),
        (
            "lookup",
            {"query": "status"},
            "desktop",
            [
                _extra_tool_schema(
                    "lookup",
                    properties={"query": {"type": "string"}},
                    required=["query"],
                )
            ],
        ),
    ],
)
def test_validate_canonical_rows_allows_eof_unpaired_tool_calls(
    name,
    arguments,
    platform,
    extra_tool_schemas,
):
    rows = [{
        "images": [],
        "messages": [{
            "role": "assistant",
            "tool_calls": [_tc(name, arguments, call_id="call_0000")],
        }],
        "metadata": LiteCUAMetadata(
            dims=(platform, "use"),
            extra_tool_schemas=extra_tool_schemas,
            valid_actions=None,
            others={},
        ).to_dict(),
    }]

    validate_canonical_rows(rows, f"unit/{name}-eof-unpaired")


@pytest.mark.parametrize(
    "name,arguments,schema",
    [
        (
            "response",
            {"text": "Done."},
            _extra_tool_schema(
                "response",
                properties={"text": {"type": "string"}},
                required=["text"],
            ),
        ),
        (
            "terminate",
            {"status": "success"},
            _extra_tool_schema(
                "terminate",
                properties={"status": {"type": "string"}},
                required=["status"],
            ),
        ),
    ],
)
def test_validate_canonical_rows_rejects_assistant_turn_after_unpaired_terminal_finish(
    name,
    arguments,
    schema,
):
    rows = [{
        "images": [],
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [_tc(name, arguments, call_id="call_0000")],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[schema],
            valid_actions=None,
            others={"terminated": True},
        ).to_dict(),
    }]

    with pytest.raises(ValueError, match="a finish call ends the row"):
        validate_canonical_rows(rows, f"unit/{name}-nonterminal-finish")


def _understanding_row(
    *,
    assistant_message: dict,
    extra_tool_schemas: list[dict] | None = None,
) -> dict:
    return {
        "images": ["images/0000.png"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "What is shown?"},
                ],
            },
            assistant_message,
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "understanding"),
            extra_tool_schemas=extra_tool_schemas or [],
            valid_actions=None,
            others={},
        ).to_dict(),
    }


def test_validate_canonical_rows_allows_understanding_plain_text():
    row = _understanding_row(
        assistant_message={
            "role": "assistant",
            "content": [{"type": "text", "text": "A settings dialog."}],
        },
    )

    validate_canonical_rows([row], "unit/understanding-plain")


def test_validate_canonical_rows_rejects_schema_backed_understanding_tool_call():
    row = _understanding_row(
        assistant_message={
            "role": "assistant",
            "tool_calls": [_tc(
                "response",
                {"text": "A settings dialog."},
                call_id="call_response",
            )],
        },
        extra_tool_schemas=[
            _extra_tool_schema(
                "response",
                properties={"text": {"type": "string"}},
                required=["text"],
            )
        ],
    )

    with pytest.raises(ValueError, match="understanding rows are plain QA/caption"):
        validate_canonical_rows([row], "unit/understanding-tool-call")


def _grounding_label_row(
    *,
    task_type: str,
    platform: str,
    name: str,
    arguments: dict,
    extra_tool_schemas: list[dict] | None = None,
) -> dict:
    return {
        "images": ["images/0000.png"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": "label this target"},
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [_tc(name, arguments, call_id="call_label")],
            },
        ],
        "metadata": LiteCUAMetadata(
            dims=(platform, task_type),
            extra_tool_schemas=extra_tool_schemas or [],
            valid_actions=None,
            others={},
        ).to_dict(),
    }


@pytest.mark.parametrize(
    "task_type,platform,name,arguments,extra_tool_schemas",
    [
        (
            "grounding.action",
            "browser",
            "computer",
            {"actions": [{"action": "click", "coordinate": [1, 2]}]},
            [],
        ),
        ("grounding.point", "desktop", "point", {"coordinate": [1, 2]}, []),
        ("grounding.bbox", "desktop", "bbox", {"coordinate": [1, 2, 3, 4]}, []),
        (
            "grounding.action",
            "browser",
            "response",
            {"text": "No matching element."},
            [
                _extra_tool_schema(
                    "response",
                    properties={"text": {"type": "string"}},
                    required=["text"],
                )
            ],
        ),
    ],
)
def test_validate_canonical_rows_allows_grounding_label_calls_without_tool_result(
    task_type,
    platform,
    name,
    arguments,
    extra_tool_schemas,
):
    row = _grounding_label_row(
        task_type=task_type,
        platform=platform,
        name=name,
        arguments=arguments,
        extra_tool_schemas=extra_tool_schemas,
    )

    validate_canonical_rows([row], f"unit/{task_type}-label")


@pytest.mark.parametrize(
    "task_type,platform,name,arguments,extra_tool_schemas",
    [
        (
            "grounding.action",
            "browser",
            "computer",
            {"actions": [{"action": "click", "coordinate": [1, 2]}]},
            [],
        ),
        ("grounding.point", "desktop", "point", {"coordinate": [1, 2]}, []),
        ("grounding.bbox", "desktop", "bbox", {"coordinate": [1, 2, 3, 4]}, []),
        (
            "grounding.action",
            "browser",
            "response",
            {"text": "No matching element."},
            [
                _extra_tool_schema(
                    "response",
                    properties={"text": {"type": "string"}},
                    required=["text"],
                )
            ],
        ),
    ],
)
def test_validate_raw_rollout_rows_allows_grounding_label_calls_without_tool_result(
    task_type,
    platform,
    name,
    arguments,
    extra_tool_schemas,
):
    row = _grounding_label_row(
        task_type=task_type,
        platform=platform,
        name=name,
        arguments=arguments,
        extra_tool_schemas=extra_tool_schemas,
    )

    validate_raw_rollout_rows([row], f"unit/{task_type}-raw-label")


def test_validate_canonical_rows_rejects_fake_grounding_label_tool_result():
    row = _grounding_label_row(
        task_type="grounding.point",
        platform="desktop",
        name="point",
        arguments={"coordinate": [1, 2]},
    )
    row["messages"].append({
        "role": "tool",
        "tool_call_id": "call_label",
        "content": [{"type": "text", "text": "fake"}],
    })

    with pytest.raises(ValueError, match="orphan role:tool result"):
        validate_canonical_rows([row], "unit/grounding-fake-result")


def test_validate_raw_rollout_rows_rejects_fake_grounding_label_tool_result():
    row = _grounding_label_row(
        task_type="grounding.point",
        platform="desktop",
        name="point",
        arguments={"coordinate": [1, 2]},
    )
    row["messages"].append({
        "role": "tool",
        "tool_call_id": "call_label",
        "content": [{"type": "text", "text": "fake"}],
    })

    with pytest.raises(ValueError, match="orphan role:tool result"):
        validate_raw_rollout_rows([row], "unit/grounding-raw-fake-result")


_RESPONSE_SCHEMA = _extra_tool_schema(
    "response",
    properties={"text": {"type": "string"}},
    required=["text"],
)
_CLICK_CALL = _tc(
    "computer",
    {"actions": [{"action": "click", "coordinate": [1, 2]}]},
    call_id="call_click",
)


@pytest.mark.parametrize("validate", [validate_canonical_rows, validate_raw_rollout_rows])
def test_single_turn_row_may_continue_after_a_response_call(validate):
    """Grounding adapters emit `response` as a deliberate non-terminator.

    Treating it as one rejects every such label row, so those cohorts cannot publish.
    """
    row = _grounding_label_row(
        task_type="grounding.action",
        platform="browser",
        name="response",
        arguments={"text": "No matching element."},
        extra_tool_schemas=[_RESPONSE_SCHEMA],
    )
    row["messages"].append({"role": "assistant", "tool_calls": [_CLICK_CALL]})

    validate([row], "unit/grounding-response-then-action")


@pytest.mark.parametrize("validate", [validate_canonical_rows, validate_raw_rollout_rows])
def test_single_turn_row_rejects_env_feedback_on_its_response_call(validate):
    """A single-turn row answering its own ``response`` call is a producer bug.

    This is the ONE trailing-``role:"tool"`` shape the orphan raise cannot catch:
    ``response`` is a finish tool name, so ``_finish_call_ids`` admits the result
    as "unrequired but emitted" and the row survives to the trailing-result rule.
    ``is_rollout`` is False here and ``episode_ended`` False, so that rule is the
    only thing standing between this row and publication -- qualifying it with
    ``and is_rollout`` (a proposal that looks behaviour-neutral: it changes no
    verdict over the 2,967-row ``.logs`` corpus) would silently let this publish,
    against ``SINGLE_TURN_TASK_TYPES``'s own statement that such a row is a
    producer bug.
    """
    row = _grounding_label_row(
        task_type="grounding.point",
        platform="desktop",
        name="response",
        arguments={"text": "(10, 20)"},
        extra_tool_schemas=[_RESPONSE_SCHEMA],
    )
    row["messages"].append({
        "role": "tool",
        "tool_call_id": "call_label",
        "content": [{"type": "text", "text": "wrong target"}],
    })

    with pytest.raises(ValueError, match="trailing role:tool result"):
        validate([row], "unit/grounding-env-feedback-on-response")


@pytest.mark.parametrize("validate", [validate_canonical_rows, validate_raw_rollout_rows])
def test_use_row_still_ends_at_a_response_call(validate):
    """On a real episode a finish call is terminal; dropping that lets post-finish turns publish."""
    row = {
        "images": ["images/0000.png"],
        "messages": [
            {"role": "user", "content": [{"type": "image", "index": 0}]},
            {
                "role": "assistant",
                "tool_calls": [_tc("response", {"text": "Done."}, call_id="call_finish")],
            },
            {"role": "assistant", "tool_calls": [_CLICK_CALL]},
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[_RESPONSE_SCHEMA],
            valid_actions=None,
            others={"terminated": True},
        ).to_dict(),
    }

    with pytest.raises(ValueError, match="a finish call ends the row"):
        validate([row], "unit/use-response-then-action")


@pytest.mark.parametrize(
    "task_type,platform,name,arguments,feedback",
    [
        (
            "grounding.point",
            "desktop",
            "point",
            {"coordinate": [1, 2]},
            "terminal point feedback",
        ),
        (
            "grounding.action",
            "desktop",
            "computer",
            {"actions": [{"action": "click", "coordinate": [1, 2]}]},
            "## Error from previous action:\ntarget disappeared",
        ),
    ],
)
def test_validate_raw_rollout_rows_allows_terminal_grounding_env_tool_result_at_eof(
    task_type,
    platform,
    name,
    arguments,
    feedback,
):
    row = _grounding_label_row(
        task_type=task_type,
        platform=platform,
        name=name,
        arguments=arguments,
    )
    row["metadata"]["others"] = {
        "env_id": "test.env",
        "task_id": "task_0",
        "terminated": True,
    }
    row["messages"].append({
        "role": "tool",
        "tool_call_id": "call_label",
        "content": [{"type": "text", "text": feedback}],
    })

    validate_raw_rollout_rows([row], f"unit/{task_type}-terminal-feedback")


def test_validate_canonical_rows_rejects_terminal_grounding_env_tool_result_at_eof():
    row = _grounding_label_row(
        task_type="grounding.point",
        platform="desktop",
        name="point",
        arguments={"coordinate": [1, 2]},
    )
    row["metadata"]["others"] = {
        "env_id": "test.env",
        "task_id": "task_0",
        "terminated": True,
    }
    row["messages"].append({
        "role": "tool",
        "tool_call_id": "call_label",
        "content": [{"type": "text", "text": "terminal point feedback"}],
    })

    with pytest.raises(ValueError, match="orphan role:tool result"):
        validate_canonical_rows([row], "unit/grounding-publish-terminal-feedback")


@pytest.mark.parametrize(
    "name,arguments,schema,text",
    [
        (
            "response",
            {"text": "Done."},
            _extra_tool_schema(
                "response",
                properties={"text": {"type": "string"}},
                required=["text"],
            ),
            "Final answer submitted: Done.",
        ),
        (
            "terminate",
            {"status": "success"},
            _extra_tool_schema(
                "terminate",
                properties={"status": {"type": "string"}},
                required=["status"],
            ),
            "Task terminated: success",
        ),
    ],
)
def test_validate_canonical_rows_allows_paired_terminal_extra_tool_results(
    name,
    arguments,
    schema,
    text,
):
    rows = [{
        "images": [],
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [_tc(name, arguments, call_id="call_0000")],
            },
            {
                "role": "tool",
                "tool_call_id": "call_0000",
                "content": [{"type": "text", "text": text}],
            },
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[schema],
            valid_actions=None,
            others={"terminated": True},
        ).to_dict(),
    }]

    validate_canonical_rows(rows, f"unit/{name}-terminal-paired")


def test_validate_canonical_rows_allows_final_text_without_tool_result():
    rows = [{
        "images": [],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "task"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={"terminated": True},
        ).to_dict(),
    }]

    validate_canonical_rows(rows, "unit/final-text")


def test_validate_canonical_rows_rejects_synthetic_tool_result_after_final_text():
    rows = [{
        "images": [],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "task"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
            {
                "role": "tool",
                "tool_call_id": "parse_error_0",
                "content": [{"type": "text", "text": "unparseable output"}],
            },
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={"terminated": True},
        ).to_dict(),
    }]

    with pytest.raises(ValueError, match="orphan role:tool result"):
        validate_canonical_rows(rows, "unit/synthetic-unparseable-result")


def test_validate_raw_rollout_rows_allows_unknown_tool_error_only_result():
    rows = [{
        "images": [],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "task"}]},
            {
                "role": "assistant",
                "tool_calls": [_tc("foo", {}, call_id="foo_0")],
            },
            {
                "role": "tool",
                "tool_call_id": "foo_0",
                "content": [{
                    "type": "text",
                    "text": project_tool_result_text(None, "unknown tool: foo"),
                }],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    }]

    validate_raw_rollout_rows(rows, "unit/unknown-tool-error-only")


def _extra_tool_invalid_arguments_row(tool_result_text: str) -> dict:
    return {
        "images": ["screen.png"],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "task"}]},
            {
                "role": "assistant",
                "tool_calls": [_tc(
                    "click",
                    {"index": "[3]"},
                    call_id="call_click",
                )],
            },
            {
                "role": "tool",
                "tool_call_id": "call_click",
                "content": [
                    {"type": "image", "index": 0},
                    {"type": "text", "text": tool_result_text},
                    {"type": "metadata", "data": {"is_error": True}},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
        ],
        "metadata": LiteCUAMetadata(
            dims=("browser", "use"),
            extra_tool_schemas=[
                _extra_tool_schema(
                    "click",
                    properties={"index": {"type": "integer"}},
                    required=["index"],
                )
            ],
            valid_actions=None,
            others={},
        ).to_dict(),
    }


def test_validate_raw_rollout_rows_allows_extra_tool_invalid_arguments_feedback():
    row = _extra_tool_invalid_arguments_row(
        project_tool_result_text(
            "current observation",
            "invalid arguments for click: click.arguments.index must be an integer",
        )
    )

    validate_raw_rollout_rows([row], "unit/extra-tool-invalid-arguments")


def test_validate_raw_rollout_rows_rejects_extra_tool_invalid_arguments_without_feedback():
    row = _extra_tool_invalid_arguments_row(
        project_tool_result_text("current observation", "click failed: bad coordinate")
    )

    with pytest.raises(ValueError, match="arguments do not match"):
        validate_raw_rollout_rows([row], "unit/extra-tool-invalid-arguments-wrong-error")


def test_validate_raw_rollout_rows_rejects_invalid_arguments_for_other_tool():
    row = _extra_tool_invalid_arguments_row(
        project_tool_result_text(
            "current observation",
            "invalid arguments for type: type.arguments.text must be a string",
        )
    )

    with pytest.raises(ValueError, match="arguments do not match"):
        validate_raw_rollout_rows([row], "unit/extra-tool-invalid-arguments-wrong-tool")


def test_validate_raw_rollout_rows_rejects_invalid_arguments_with_multiple_text_parts():
    row = _extra_tool_invalid_arguments_row(
        project_tool_result_text(
            "current observation",
            "invalid arguments for click: click.arguments.index must be an integer",
        )
    )
    row["messages"][2]["content"].insert(
        2,
        {"type": "text", "text": "second text part"},
    )

    with pytest.raises(ValueError, match="arguments do not match"):
        validate_raw_rollout_rows([row], "unit/extra-tool-invalid-arguments-two-texts")


def test_validate_canonical_rows_rejects_extra_tool_invalid_arguments_feedback():
    row = _extra_tool_invalid_arguments_row(
        project_tool_result_text(
            "current observation",
            "invalid arguments for click: click.arguments.index must be an integer",
        )
    )

    with pytest.raises(ValueError, match="arguments do not match"):
        validate_canonical_rows([row], "unit/extra-tool-invalid-arguments-publish")


def test_validate_raw_rollout_rows_accepts_tagged_generic_metadata_json() -> None:
    metadata = LiteGenericMetadata(
        dims=("geo3k", "sft"),
        extra_tool_schemas=[_RESPONSE_SCHEMA],
        others={"source": "unit"},
    ).to_dict()
    row = {
        "images": [],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "solve"}]},
            {
                "role": "assistant",
                "tool_calls": [_tc(
                    "response",
                    {"text": "42"},
                    call_id="call_response",
                )],
            },
        ],
        "metadata": json.dumps(metadata),
    }

    validate_raw_rollout_rows([row], "unit/generic-raw")


def test_validate_canonical_rows_rejects_unknown_tool_error_only_result():
    rows = [{
        "images": [],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "task"}]},
            {
                "role": "assistant",
                "tool_calls": [_tc("foo", {}, call_id="foo_0")],
            },
            {
                "role": "tool",
                "tool_call_id": "foo_0",
                "content": [{
                    "type": "text",
                    "text": project_tool_result_text(None, "unknown tool: foo"),
                }],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    }]

    with pytest.raises(ValueError, match="standalone but missing"):
        validate_canonical_rows(rows, "unit/unknown-tool-error-only")


def test_validate_canonical_rows_rejects_unknown_tool_non_error_result():
    rows = [{
        "images": [],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "task"}]},
            {
                "role": "assistant",
                "tool_calls": [_tc("foo", {}, call_id="foo_0")],
            },
            {
                "role": "tool",
                "tool_call_id": "foo_0",
                "content": [{"type": "text", "text": "ok"}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    }]

    with pytest.raises(ValueError, match="standalone but missing"):
        validate_canonical_rows(rows, "unit/unknown-tool-non-error")


def test_validate_canonical_rows_allows_schema_shaped_action_named_extra_tool():
    rows = [{
        "images": [],
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [_tc("click", {"index": 3}, call_id="call_click")],
            },
            {
                "role": "tool",
                "tool_call_id": "call_click",
                "content": [{"type": "text", "text": "ok"}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
        ],
        "metadata": LiteCUAMetadata(
            dims=("browser", "use"),
            extra_tool_schemas=[make_tool_schema(
                "click",
                parameters={
                    "type": "object",
                    "properties": {"index": {"type": "integer"}},
                    "required": ["index"],
                },
            )],
            valid_actions=[],
            others={},
        ).to_dict(),
    }]

    validate_canonical_rows(rows, "unit/action-named-extra")


def test_validate_canonical_rows_rejects_nested_schema_shaped_action_named_extra_tool():
    rows = [{
        "images": [],
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [_tc(
                    "computer",
                    {"actions": [{"action": "click", "index": 3}]},
                    call_id="call_computer",
                )],
            },
            {
                "role": "tool",
                "tool_call_id": "call_computer",
                "content": [{"type": "text", "text": "ok"}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
        ],
        "metadata": LiteCUAMetadata(
            dims=("browser", "use"),
            extra_tool_schemas=[make_tool_schema(
                "click",
                parameters={
                    "type": "object",
                    "properties": {"index": {"type": "integer"}},
                    "required": ["index"],
                },
            )],
            valid_actions=None,
            others={},
        ).to_dict(),
    }]

    with pytest.raises(ValueError, match="must not nest standalone extra tool 'click'"):
        validate_canonical_rows(rows, "unit/nested-action-named-extra")


def test_validate_canonical_rows_rejects_action_named_extra_tool_wrong_shape():
    rows = [{
        "images": [],
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [_tc(
                    "click",
                    {"coordinate": [1, 2]},
                    call_id="call_click",
                )],
            },
            {
                "role": "tool",
                "tool_call_id": "call_click",
                "content": [{"type": "text", "text": "ok"}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
        ],
        "metadata": LiteCUAMetadata(
            dims=("browser", "use"),
            extra_tool_schemas=[make_tool_schema(
                "click",
                parameters={
                    "type": "object",
                    "properties": {"index": {"type": "integer"}},
                    "required": ["index"],
                },
            )],
            valid_actions=[],
            others={},
        ).to_dict(),
    }]

    with pytest.raises(ValueError, match="arguments do not match"):
        validate_canonical_rows(rows, "unit/action-named-extra-shape")


@pytest.mark.parametrize(
    "content",
    [
        [],
        [{"type": "inline_reasoning", "text": "done"}],
        [{"type": "action_description", "text": "Done."}],
        [{"type": "text", "text": ""}],
    ],
)
def test_validate_canonical_rows_rejects_degenerate_content_only_final(content):
    rows = [{
        "images": [],
        "messages": [{
            "role": "assistant",
            "content": content,
        }],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    }]

    with pytest.raises(ValueError, match="content-only final assistant turn"):
        validate_canonical_rows(rows, "unit/content-final")


@pytest.mark.parametrize(
    "content",
    [
        [{"type": "text", "text": "Done."}],
        [{"type": "inline_reasoning", "text": "thinking"}, {"type": "text", "text": "Done."}],
        [{"type": "text", "text": "Done."}, {"type": "text", "text": "extra"}],
    ],
)
def test_validate_canonical_rows_accepts_visible_content_only_final_text(content):
    rows = [{
        "images": [],
        "messages": [{
            "role": "assistant",
            "content": content,
        }],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    }]

    validate_canonical_rows(rows, "unit/content-final")


def test_validate_canonical_rows_rejects_mid_trajectory_content_only_assistant():
    rows = [{
        "images": [],
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "I will click next."}],
            },
            {
                "role": "assistant",
                "tool_calls": [_tc(
                    "computer",
                    {"actions": [{"action": "click", "coordinate": [1, 2]}]},
                    call_id="call_click",
                )],
            },
            {
                "role": "tool",
                "tool_call_id": "call_click",
                "content": [{"type": "text", "text": "ok"}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    }]

    with pytest.raises(ValueError, match="content-only assistant.*final message"):
        validate_canonical_rows(rows, "unit/mid-content-only")


def test_validate_canonical_rows_allows_content_only_attempt_with_user_feedback():
    rows = [{
        "images": [],
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "First answer"}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "incorrect; revise"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Done."}],
            },
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    }]

    validate_canonical_rows(rows, "unit/content-only-attempt-feedback")


def test_validate_canonical_rows_rejects_dangling_content_only_attempt_feedback():
    rows = [{
        "images": [],
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "First answer"}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "incorrect; revise"}],
            },
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    }]

    with pytest.raises(ValueError, match="continue to another assistant turn"):
        validate_canonical_rows(rows, "unit/content-only-attempt-dangling")


@pytest.mark.parametrize("content", [[], [{"type": "metadata", "data": {"source": "env"}}]])
def test_validate_canonical_rows_rejects_unrenderable_content_only_attempt_feedback(content):
    rows = [{
        "images": [],
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "First answer"}],
            },
            {
                "role": "user",
                "content": content,
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Done."}],
            },
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    }]

    with pytest.raises(ValueError, match="model-visible env feedback"):
        validate_canonical_rows(rows, "unit/content-only-attempt-unrenderable-feedback")


def test_validate_canonical_rows_rejects_empty_content_only_attempt():
    rows = [{
        "images": [],
        "messages": [
            {"role": "assistant", "content": []},
            {
                "role": "user",
                "content": [{"type": "text", "text": "incorrect; revise"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Done."}],
            },
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    }]

    with pytest.raises(ValueError, match="content-only response assistant turn"):
        validate_canonical_rows(rows, "unit/content-only-attempt-empty")


def test_validate_raw_rollout_rows_accepts_empty_live_content_only_final():
    rows = [{
        "images": [],
        "messages": [{
            "role": "assistant",
            "content": [],
        }],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={"terminated": True},
        ).to_dict(),
    }]

    validate_raw_rollout_rows(rows, "unit/raw-empty-final")


def _image_publish_row(index, *, images: list[str] | None = None) -> dict:
    return {
        "images": ["images/0000.png"] if images is None else images,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image", "index": index}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Done."}],
            },
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    }


def test_validate_canonical_rows_allows_orphan_images_without_remapping():
    rows = [_image_publish_row(0, images=["images/0000.png", "images/orphan.png"])]

    validate_canonical_rows(rows, "unit/image-indices")


@pytest.mark.parametrize("validate", [validate_canonical_rows, validate_raw_rollout_rows])
def test_validate_rows_require_plural_images_key_even_without_image_parts(validate):
    row = _image_publish_row(0, images=[])
    row["messages"][0]["content"] = [{"type": "text", "text": "No screenshot."}]
    del row["images"]

    with pytest.raises(ValueError, match="images is required"):
        validate([row], "unit/images-required")


@pytest.mark.parametrize("validate", [validate_canonical_rows, validate_raw_rollout_rows])
def test_validate_rows_reject_legacy_singular_image_key(validate):
    row = _image_publish_row(0, images=[])
    row["messages"][0]["content"] = [{"type": "text", "text": "No screenshot."}]
    row["image"] = "images/0000.png"

    with pytest.raises(ValueError, match=r"row\.image is retired; use images"):
        validate([row], "unit/images-required")


def test_validate_canonical_rows_preserves_plain_final_text():
    row = _image_publish_row(0)
    row["messages"][-1]["content"] = [
        {"type": "text", "text": "The file has been saved."},
    ]
    before = json.loads(json.dumps(row["messages"]))

    validate_canonical_rows([row], "unit/plain-final-text")

    assert row["messages"] == before


def test_validate_canonical_rows_rejects_raw_response_sidecar():
    row = _image_publish_row(0)
    row["messages"][-1]["raw_response"] = {
        "adapter_key": "qwen3_vl@desktop@use",
        "text": "Action: Done.",
    }

    with pytest.raises(ValueError, match="raw_response.*must not be published"):
        validate_canonical_rows([row], "unit/raw-response")


def test_validate_raw_rollout_rows_allows_raw_response_sidecar():
    row = _image_publish_row(0)
    row["messages"][-1]["raw_response"] = {
        "adapter_key": "qwen3_vl@desktop@use",
        "text": "Action: Done.",
    }

    validate_raw_rollout_rows([row], "unit/raw-response")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        (CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY, {"version": 1, "stop_reason": "text"}),
        (MODEL_OUTPUT_ERROR_KEY, "malformed tool call"),
    ],
)
def test_validate_canonical_rows_rejects_private_final_message_sidecars(
    key,
    value,
):
    row = _image_publish_row(0)
    row["messages"][-1][key] = value

    with pytest.raises(ValueError, match="private final-message sidecar"):
        validate_canonical_rows([row], f"unit/{key}")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        (CONTENT_ONLY_FINAL_DIAGNOSTIC_KEY, {"version": 1, "stop_reason": "text"}),
        (MODEL_OUTPUT_ERROR_KEY, "malformed tool call"),
    ],
)
def test_validate_raw_rollout_rows_allows_private_final_message_sidecars(
    key,
    value,
):
    row = _image_publish_row(0)
    row["messages"][-1][key] = value

    validate_raw_rollout_rows([row], f"unit/raw-{key}")


def test_validate_canonical_rows_rejects_content_only_final_metadata():
    row = _image_publish_row(0)
    row["metadata"]["others"]["content_only_final"] = {"stop_reason": "text"}

    with pytest.raises(ValueError, match="content_only_final.*must not be published"):
        validate_canonical_rows([row], "unit/content-final-metadata")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("env_id", "lite.osworld"),
        ("task_id", "task-001"),
        ("episode_return", 1.0),
        ("terminated", True),
        ("truncated", False),
    ],
)
def test_validate_canonical_rows_rejects_rollout_facts_at_metadata_top_level(
    key,
    value,
):
    row = _image_publish_row(0)
    row["metadata"][key] = value

    with pytest.raises(ValueError, match=f"unknown top-level keys.*{key}"):
        validate_canonical_rows([row], f"unit/top-level-{key}")


def test_validate_canonical_rows_keeps_rollout_facts_under_metadata_others():
    row = _image_publish_row(0)
    row["metadata"]["others"].update(
        {
            "env_id": "lite.osworld",
            "task_id": "task-001",
            "episode_return": 1.0,
            "terminated": True,
            "truncated": False,
        }
    )

    validate_canonical_rows([row], "unit/others-rollout-facts")


def test_validate_raw_rollout_rows_allows_unstage_split_hint():
    row = _image_publish_row(0)
    row["metadata"]["others"]["split"] = "train"

    validate_raw_rollout_rows([row], "unit/raw-others-split")


def test_validate_canonical_rows_rejects_dialect_only_answer_extra_tool_schema():
    row = _image_publish_row(0)
    row["metadata"]["extra_tool_schemas"] = [make_tool_schema("answer", parameters={})]

    with pytest.raises(ValueError, match="canonical 'response'"):
        validate_canonical_rows([row], "unit/answer-extra")


def test_validate_canonical_rows_rejects_dialect_only_answer_tool_call():
    row = {
        "images": [],
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [_tc("answer", {"text": "done"}, call_id="call_answer")],
            },
            {
                "role": "tool",
                "tool_call_id": "call_answer",
                "content": [{"type": "text", "text": "ok"}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
        ],
        "metadata": LiteCUAMetadata(
            dims=("desktop", "use"),
            extra_tool_schemas=[],
            valid_actions=None,
            others={},
        ).to_dict(),
    }

    with pytest.raises(ValueError, match="canonical 'response'"):
        validate_canonical_rows([row], "unit/answer-call")


@pytest.mark.parametrize("name", ["computer", "mobile", "point", "bbox"])
def test_validate_canonical_rows_rejects_top_level_action_extra_tool_schema(name):
    row = _image_publish_row(0)
    row["metadata"]["extra_tool_schemas"] = [make_tool_schema(name, parameters={})]

    with pytest.raises(ValueError, match="canonical top-level GUI"):
        validate_canonical_rows([row], f"unit/{name}-extra")


@pytest.mark.parametrize("index", [-1, True, "0", 1.5, 2])
def test_validate_canonical_rows_rejects_invalid_image_indices(index):
    rows = [_image_publish_row(index, images=["images/0000.png", "images/0001.png"])]

    with pytest.raises(LiteContractError, match="index.*(non-negative|out of range)"):
        validate_canonical_rows(rows, "unit/image-indices")


def _grounding_env_rollout_final() -> list[dict]:
    """The assistant turn a real ``osworld_g`` rollout produced on a length cap.

    Copied from what the REAL ``mai_ui`` adapter emits for
    ``.logs/famval/osworld_g/maiui8b/eval/0FOB4CLBT2-0``'s own model bytes cut at
    a max-token boundary: no ``tool_calls``, ``inline_reasoning`` only, so
    ``no_tool_call_final_text`` is empty.
    """
    return [
        {"role": "user", "content": [{"type": "image", "index": 0},
                                     {"type": "text", "text": "Open the filter function."}]},
        {"role": "assistant", "content": [{"type": "inline_reasoning", "text": "Thought: ..."}]},
    ]


def _grounding_env_metadata(others: dict) -> LiteCUAMetadata:
    return LiteCUAMetadata.from_dict(LiteCUAMetadata(
            dims=("desktop", "grounding.point"),
            extra_tool_schemas=[],
            valid_actions=["point"],
            others=dict(others),
        ).to_dict())
