"""Shared mobile prompt guidance for schema-backed finish tools."""

from __future__ import annotations

from collections.abc import Container


def mobile_finish_guidance(extra_tool_names: Container[str]) -> str | None:
    """Return completion guidance for active mobile finish tools."""
    has_response = "response" in extra_tool_names
    has_terminate = "terminate" in extra_tool_names
    if has_response and has_terminate:
        return (
            "When the task is done, call `terminate` (use `response` to return "
            "an answer the task asks for)."
        )
    if has_response:
        return "When the task asks for an answer, call `response` to return it."
    if has_terminate:
        return "When the task is done, call `terminate`."
    return None
