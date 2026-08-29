"""Explicit bootstrap for built-in agent registrations."""

from __future__ import annotations

import importlib

_REGISTRATION_MODULES = (
    "lite.agents.models.lite.adapter",
    "lite.agents.models.qwen3_vl.action_space",
    "lite.agents.models.qwen3_vl.protocol",
    "lite.agents.models.qwen3_vl.adapter",
    "lite.agents.models.qwen3_vl.agent",
    "lite.agents.models.qwen2_5_vl.action_space",
    "lite.agents.models.qwen2_5_vl.adapter",
    "lite.agents.models.qwen2_5_vl.agent",
    "lite.agents.models.fara.action_space",
    "lite.agents.models.fara.protocol",
    "lite.agents.models.fara.adapter",
    "lite.agents.models.fara.agent",
    "lite.agents.models.qwen3_5.action_space",
    "lite.agents.models.qwen3_5.protocol",
    "lite.agents.models.qwen3_5.adapter",
    "lite.agents.models.qwen3_5.agent",
    "lite.agents.models.qwen3_8.action_space",
    "lite.agents.models.qwen3_8.adapter",
    "lite.agents.models.qwen3_8.agent",
    "lite.agents.models.ui_tars.action_space",
    "lite.agents.models.ui_tars.protocol",
    "lite.agents.models.ui_tars.adapter",
    "lite.agents.models.ui_tars.agent",
    "lite.agents.models.ui_tars_15_v1.action_space",
    "lite.agents.models.ui_tars_15_v1.adapter",
    "lite.agents.models.ui_tars_15_v1.agent",
    "lite.agents.models.evocua.action_space",
    "lite.agents.models.evocua.adapter",
    "lite.agents.models.evocua.agent",
    "lite.agents.models.mai_ui.action_space",
    "lite.agents.models.mai_ui.protocol",
    "lite.agents.models.mai_ui.adapter",
    "lite.agents.models.mai_ui.agent",
    "lite.agents.models.ui_venus_2.action_space",
    "lite.agents.models.ui_venus_2.protocol",
    "lite.agents.models.ui_venus_2.adapter",
    "lite.agents.models.ui_venus_2.agent",
    "lite.agents.models.step_gui.action_space",
    "lite.agents.models.step_gui.protocol",
    "lite.agents.models.step_gui.adapter",
    "lite.agents.models.step_gui.agent",
    "lite.agents.extensions.browsergym.protocol",
    "lite.agents.extensions.browsergym.goal_image",
    "lite.agents.extensions.webharbor.webvoyager.protocol",
)

_OPTIONAL_REGISTRATION_MODULES = (
    "lite.agents.models.claude.action_space",
    "lite.agents.models.claude.agent",
    "lite.agents.models.gpt.action_space",
    "lite.agents.models.gpt.agent",
    "lite.agents.models.gemini.action_space",
    "lite.agents.models.gemini.agent",
    "lite.agents.extensions.teacher.agent",
)


def register_all() -> None:
    """Import built-in registration modules.

    Registries are populated by concrete classes' ``key=`` declarations. The
    imports are idempotent, so callers may safely invoke this more than once.
    Family package roots stay side-effect-free; this leaf-module list owns
    built-in registration order explicitly.
    """
    for module_name in _REGISTRATION_MODULES:
        importlib.import_module(module_name)
    for module_name in _OPTIONAL_REGISTRATION_MODULES:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            # API agents are optional when litellm is not installed. Keep real
            # internal import failures visible so registrations do not silently
            # disappear.
            if (e.name or "").split(".")[0] != "litellm":
                raise


__all__ = ["register_all"]
