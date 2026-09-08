"""
Cross-family tests for the ``grounding.action`` harness (SFT-data replay).

``grounding.action`` is the SFT-side task_type for the historical
multi-action grounding parquets (``lite/data/preproc/<source>/grounding-action.py``):
each record is a single-step user image + assistant tool_calls (potentially
multiple — ``click + type + key``). This is **distinct** from the env-side
``grounding.point`` harness (single-step click only, single tool_call).

Tests verify that every family's grounding.action adapter:

  1. Resolves from the registry to a concrete per-platform adapter.
  2. Uses ``FullHistoryProtocol`` (single-turn, k=1).
  3. Carries the family's full navigation action_space (NOT the trimmed
     grounding-point one) — multi-action SFT shape needs the full vocab.
  4. Round-trips an SFT-style multi-action sample without dropping calls.

Run:
    uv run pytest tests/agents/core/adapter/test_grounding_action.py -n 0
"""

from __future__ import annotations

import pytest
from agents._support.valid_actions_gating import agent_adapter_for

import lite.agents.models.evocua.adapter  # noqa: F401
import lite.agents.models.lite.adapter  # noqa: F401
import lite.agents.models.qwen3_5.adapter  # noqa: F401

# Trigger registration of every grounding.action adapter.
import lite.agents.models.qwen3_vl.adapter  # noqa: F401
import lite.agents.models.ui_tars.adapter  # noqa: F401
import lite.agents.models.ui_tars_15_v1.adapter  # noqa: F401
from lite.agents.core.adapter import AgentAdapterRegistry
from lite.core.tools.calls import make_tool_call

# ---------------------------------------------------------------------------
# Registry resolution — every (family, platform) lands at the per-platform
# concrete *GroundingActionAdapter* (NOT AsIsAdapter, NOT a *GroundingPoint*
# class; the env-eval ``:point`` chain is parallel and tested separately
# in ``test_grounding_point.py``).
# ---------------------------------------------------------------------------

GROUNDING_ACTION_KEYS = [
    "lite@desktop@grounding.action",
    "lite@browser@grounding.action",
    "lite@mobile@grounding.action",
    "qwen3_vl@desktop@grounding.action",
    "qwen3_vl@browser@grounding.action",
    "qwen3_vl@mobile@grounding.action",
    "qwen3_5@desktop@grounding.action",
    "qwen3_5@browser@grounding.action",
    "qwen3_5@mobile@grounding.action",
    "evocua@desktop@grounding.action",
    "evocua@browser@grounding.action",
    "ui_tars@desktop@grounding.action",
    "ui_tars@browser@grounding.action",
    "ui_tars@mobile@grounding.action",
    "ui_tars_15_v1@desktop@grounding.action",
    "ui_tars_15_v1@browser@grounding.action",
    "ui_tars_15_v1@mobile@grounding.action",
]


@pytest.mark.parametrize("key", GROUNDING_ACTION_KEYS)
def test_grounding_action_resolves_to_concrete_adapter(key):
    """The registry returns a concrete *GroundingActionAdapter* class — not
    AsIsAdapter (we removed the pattern-based AsIsAdapter registrations
    when restoring per-platform concrete classes).
    """
    adapter = AgentAdapterRegistry.get(key)
    name = type(adapter).__name__
    assert name.endswith("GroundingActionAdapter"), (
        f"{key} resolved to {name}; expected a *GroundingActionAdapter* class."
    )


@pytest.mark.parametrize("key", GROUNDING_ACTION_KEYS)
def test_grounding_action_uses_full_history_protocol(key):
    """Single-step SFT shape — protocol must be ``FullHistoryProtocol``."""
    adapter = AgentAdapterRegistry.get(key)
    assert type(adapter.protocol).__name__ == "FullHistoryProtocol", (
        f"{key} protocol = {type(adapter.protocol).__name__}, "
        "expected FullHistoryProtocol (single-step SFT replay)."
    )


# ---------------------------------------------------------------------------
# action_space integrity — must be the family's FULL navigation action_space
# (NOT the trimmed ``GroundingPointActionSpace``). SFT-data replay carries
# multi-action records (click + type + key + ...); a trimmed schema would
# drop everything but click.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "key,expected_space_class_endings",
    [
        # Family-native nav action_space class names — endswith match.
        ("qwen3_vl@desktop@grounding.action", "Qwen3VLDesktopActionSpace"),
        ("qwen3_vl@browser@grounding.action", "Qwen3VLDesktopActionSpace"),
        ("qwen3_vl@mobile@grounding.action", "Qwen3VLMobileActionSpace"),
        ("qwen3_5@desktop@grounding.action", "Qwen3_5DesktopActionSpace"),
        ("qwen3_5@browser@grounding.action", "Qwen3_5DesktopActionSpace"),
        ("qwen3_5@mobile@grounding.action", "Qwen3_5MobileActionSpace"),
        ("evocua@desktop@grounding.action", "EvoCUADesktopActionSpace"),
        ("evocua@browser@grounding.action", "EvoCUADesktopActionSpace"),
        ("ui_tars@desktop@grounding.action", "UITarsDesktopActionSpace"),
        ("ui_tars@browser@grounding.action", "UITarsDesktopActionSpace"),
        ("ui_tars@mobile@grounding.action", "UITarsMobileActionSpace"),
        ("ui_tars_15_v1@desktop@grounding.action", "UITars15V1DesktopActionSpace"),
        ("ui_tars_15_v1@browser@grounding.action", "UITars15V1DesktopActionSpace"),
        ("ui_tars_15_v1@mobile@grounding.action", "UITars15V1MobileActionSpace"),
        ("lite@desktop@grounding.action", "LiteDesktopActionSpace"),
        ("lite@browser@grounding.action", "LiteDesktopActionSpace"),
        ("lite@mobile@grounding.action", "LiteMobileActionSpace"),
    ],
)
def test_grounding_action_uses_full_nav_action_space(key, expected_space_class_endings):
    """The action_space is the FULL navigation action_space (not the
    trimmed grounding-point one). SFT replay carries multi-action records
    so the full vocabulary is needed.
    """
    adapter = AgentAdapterRegistry.get(key)
    actual = type(adapter.action_space).__name__
    assert actual == expected_space_class_endings, (
        f"{key} action_space = {actual!r}, expected {expected_space_class_endings!r}. "
        "GroundingAction adapters must NOT use the trimmed GroundingPointActionSpace."
    )


# ---------------------------------------------------------------------------
# Action / point coexistence — both keys resolve cleanly, to DIFFERENT
# adapters. A precedence regression (point key resolving to action class,
# or action key falling through to AsIsAdapter) would show up here.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "family,platforms",
    [
        ("qwen3_vl", ["desktop", "browser", "mobile"]),
        ("qwen3_5", ["desktop", "browser", "mobile"]),
        ("evocua", ["desktop", "browser"]),
        ("ui_tars", ["desktop", "browser", "mobile"]),
        ("ui_tars_15_v1", ["desktop", "browser", "mobile"]),
    ],
)
def test_grounding_action_and_point_are_distinct(family, platforms):
    """For each family that ships both ``grounding.action`` and
    ``grounding.point``: the two keys resolve to distinct adapter classes,
    with distinct action_spaces.
    """
    for plat in platforms:
        a_action = AgentAdapterRegistry.get(f"{family}@{plat}@grounding.action")
        a_point = AgentAdapterRegistry.get(f"{family}@{plat}@grounding.point")
        assert type(a_action) is not type(a_point), (
            f"{family}@{plat} action and point resolved to the same class "
            f"{type(a_action).__name__}"
        )
        assert type(a_action.action_space) is not type(a_point.action_space), (
            f"{family}@{plat} action and point share an action_space "
            f"{type(a_action.action_space).__name__}"
        )


@pytest.mark.parametrize(
    "platform,call",
    [
        (
            "desktop",
            make_tool_call(
                "computer",
                {"actions": [{"action": "click", "coordinate": [1, 2]}]},
            ),
        ),
        (
            "mobile",
            make_tool_call(
                "mobile",
                {"actions": [{"action": "tap", "coordinate": [1, 2]}]},
            ),
        ),
    ],
)
def test_grounding_action_calls_are_schema_free(platform, call) -> None:
    adapter = agent_adapter_for("as_is", platform, task_type="grounding.action")
    message = {"role": "assistant", "content": [], "tool_calls": [call]}

    assert adapter.convert_message_to_agent(message) == message


@pytest.mark.parametrize(
    "platform,call",
    [
        ("desktop", make_tool_call("click", {"coordinate": [1, 2]})),
        ("browser", make_tool_call("type", {"text": "hello"})),
        ("mobile", make_tool_call("tap", {"coordinate": [1, 2]})),
    ],
)
def test_grounding_action_renders_canonical_actions_without_availability_gate(
    platform,
    call,
) -> None:
    adapter = agent_adapter_for("as_is", platform, task_type="grounding.action")
    message = {"role": "assistant", "content": [], "tool_calls": [call]}

    assert adapter.convert_message_to_agent(message) == message


@pytest.mark.parametrize(
    "platform,call",
    [
        (
            "desktop",
            make_tool_call(
                "mobile",
                {"actions": [{"action": "tap", "coordinate": [1, 2]}]},
            ),
        ),
        (
            "mobile",
            make_tool_call(
                "computer",
                {"actions": [{"action": "click", "coordinate": [1, 2]}]},
            ),
        ),
    ],
)
def test_grounding_action_renders_wrong_platform_action_wrapper_without_availability_gate(
    platform,
    call,
) -> None:
    adapter = agent_adapter_for("as_is", platform, task_type="grounding.action")
    message = {"role": "assistant", "content": [], "tool_calls": [call]}

    assert adapter.convert_message_to_agent(message) == message


@pytest.mark.parametrize(
    "adapter_key,platform,call",
    [
        (
            "qwen3_vl@desktop@use",
            "desktop",
            make_tool_call(
                "mobile",
                {"actions": [{"action": "tap", "coordinate": [1, 2]}]},
            ),
        ),
        (
            "qwen3_vl@mobile@use",
            "mobile",
            make_tool_call(
                "computer",
                {"actions": [{"action": "click", "coordinate": [1, 2]}]},
            ),
        ),
    ],
)
def test_use_adapters_render_wrong_platform_action_wrapper_without_availability_gate(
    adapter_key,
    platform,
    call,
) -> None:
    adapter = agent_adapter_for(adapter_key, platform)
    message = {"role": "assistant", "content": [], "tool_calls": [call]}

    adapter.convert_message_to_agent(message)


def test_grounding_bbox_remains_task_local_schema_free() -> None:
    adapter = agent_adapter_for(
        "as_is",
        "desktop",
        task_type="grounding.bbox",
    )
    message = {
        "role": "assistant",
        "content": [],
        "tool_calls": [make_tool_call("bbox", {"coordinate": [1, 2, 3, 4]})],
    }

    assert adapter.convert_message_to_agent(message) == message
