"""OpenMobile-2 action space (``openmobile_2@mobile``).

OpenMobile-2 is a mobile GUI agent (9B fine-tuned from Qwen3.5, 27B from
Qwen3.6; both share the same XML tool_call chat_template). Its official
``qwen35_thought_session`` runtime advertises a single ``mobile_use`` tool
with an 8-action enum — the Qwen mobile enum minus ``open`` (no app-launch
action; apps are opened through the GUI) — and a 4-button
``system_button`` enum (no ``Recent``). Reference:
``OpenMobile2-Code/runtimes/prompts.py`` (the schema embedded in
``QWEN35_SESSION_SYSTEM_PROMPT_THOUGHT``) and
``OpenMobile2-Code/AndroidWorld/android_world/agents/utils.py``
(``qwen3vl_action_transform``).

Everything else inherits from :class:`Qwen3_5MobileActionSpace`:
coordinate pass-through (the model emits ``[0, 999]``; cua-lite's
``[0, 1000]`` normalized space differs by 0.1%, sub-pixel on real Android
resolutions), the ``left_click`` → ``click`` base-model leakage alias, and
the ``answer`` → ``response`` / ``terminate`` → ``terminate`` finish
mapping.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

from lite.agents.models.qwen3_5.action_space import Qwen3_5MobileActionSpace
from lite.core.tools.action_space.duration import duration_description
from lite.core.tools.calls import make_tool_call
from lite.core.tools.schemas import tool


@dataclasses.dataclass
class OpenMobile2MobileActionSpace(
    Qwen3_5MobileActionSpace,
    key="openmobile_2@mobile",
):
    """Mobile ``mobile_use`` action space for OpenMobile-2.

    Schema deltas from the Qwen3.5 parent, both taken from the official
    ``mobile_use`` schema:

      * ``action`` drops ``open`` — the official enum has no app-launch
        entry, so ``QWEN_ACTION_VALUE_TO_EXTRA_TOOL_NAMES`` drops the
        ``open_app`` folding too.
      * ``button`` drops ``Recent`` and uses the official one-line
        parameter description.

    Plus one conversion delta: the official transform lowers
    ``system_button(button="Menu")`` to a bare wait (a no-op — no keycode
    is pressed), so ``Menu`` is rewritten to the native ``wait`` call
    before the parent dispatch.
    """

    # Official action description: the Qwen mobile list minus the `open`
    # line (runtimes/prompts.py, mobile_use schema in
    # QWEN35_SESSION_SYSTEM_PROMPT_THOUGHT).
    _ACTION_DESCRIPTION = """The action to perform. The available actions are:
* `click`: Click the point on the screen with coordinate (x, y).
* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
* `type`: Input the specified text into the activated input box.
* `answer`: Output the answer.
* `system_button`: Press the system button.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status."""

    @staticmethod
    @tool(
        action=_ACTION_DESCRIPTION,
        coordinate="(x, y): The x and y coordinates. Required only by `action=click`, `action=long_press`, and `action=swipe`.",
        coordinate2="(x, y): The end coordinates. Required only by `action=swipe`.",
        text="Required only by `action=type` and `action=answer`.",
        time=duration_description(
            "The seconds to wait. Required only by `action=long_press` and `action=wait`."
        ),
        button="Required only by `action=system_button`.",
        status="Required only by `action=terminate`.",
    )
    def mobile_use(
        action: Literal[
            "click", "long_press", "swipe", "type",
            "answer", "system_button", "wait", "terminate",
        ],
        coordinate: list[int] | None = None,
        coordinate2: list[int] | None = None,
        text: str | None = None,
        time: float | None = None,
        button: Literal["Back", "Home", "Menu", "Enter"] | None = None,
        status: Literal["success", "failure"] | None = None,
    ) -> dict[str, Any]:
        """
        Use a touchscreen to interact with a mobile device, and take screenshots.

        * This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
        * Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
        * The screen's resolution is 999x999.
        * Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.
        """
        return make_tool_call(
            "mobile_use",
            {
                "action": action,
                "coordinate": coordinate,
                "coordinate2": coordinate2,
                "text": text,
                "time": time,
                "button": button,
                "status": status,
            },
        )

    # No ``open`` entry: the official enum has no app-launch action, so no
    # extra tool folds into it. ``answer`` / ``terminate`` keep folding the
    # finish tools, matching the official answer / terminate actions.
    QWEN_ACTION_VALUE_TO_EXTRA_TOOL_NAMES = {
        "answer": frozenset({"response"}),
        "terminate": frozenset({"terminate"}),
    }

    def _convert_single_from_agent(
        self,
        agent_tool_call: dict[str, Any],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        # Official qwen3vl_action_transform lowers system_button(Menu) to a
        # bare wait — a no-op, no keycode pressed. Rewrite both the nested
        # spelling and the wrapper-dropped flat one to the native wait call;
        # the parent's wait branch applies its default duration.
        args = agent_tool_call["arguments"]
        if (
            agent_tool_call["name"] in ("mobile_use", "system_button")
            and args.get("button") == "Menu"
            and args.get("action", "") in ("", "system_button")
        ):
            return super()._convert_single_from_agent(
                {"name": "mobile_use", "arguments": {"action": "wait"}},
                **kwargs,
            )
        return super()._convert_single_from_agent(agent_tool_call, **kwargs)


__all__ = ["OpenMobile2MobileActionSpace"]
