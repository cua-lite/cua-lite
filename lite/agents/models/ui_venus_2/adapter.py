"""
UI-Venus-2 Adapters (Computer / Browser / Mobile / Grounding)

    UIVenus2BaseAdapter (shared wire codec + prompt assembly)
    ├── UIVenus2DesktopUseAdapter          (desktop OS, Computer grammar)
    ├── UIVenus2BrowserUseAdapter          (browser, Browser grammar)
    ├── UIVenus2MobileUseAdapter           (Android, Mobile grammar)
    └── UIVenus2GroundingPointAdapter      (single-step point, all platforms)

Reference: ``${CUA_LITE_REFERENCES_ROOT}/UI-Venus`` @ branch ``UI-Venus-2``
  * computer  — ``models/computer/computer_example.py``
  * browser   — ``models/browser/venus_browser.py``
  * mobile    — ``models/mobile/mobile_example.py`` and
                ``Venus_framework/Venus_framework_mobile/processor/ui_venus_2_processor.py``
  * grounding — ``models/grounding/ui_venus2_gd.py``

Four facts drive everything below.

1. **The task lives in the system prompt.** Each surface has one SFT system
   prompt with the action rows baked in and a ``{user_task}`` slot at the end.
   No gate narrows it: neither the sample's active extra tools nor
   ``metadata.valid_actions`` may delete a row. Withholding a trained row moves
   the model off the distribution it was fine-tuned on, and reachability is env
   ingress's question to answer — a call to an inactive tool comes back as
   model-visible feedback keyed to its call id.

2. **One tagged block per turn:** ``<think>…</think>\\n<action>…</action>``.
   ``think`` is prompted CoT and becomes an ``InlineReasoningContent`` part.
   There is no ``<conclusion>`` (UI-Venus-1.5 had one). The chat template
   OPENS the think block in the generation prompt, so a live response carries
   only the CLOSING tag — see :func:`_reasoning_text` and
   :attr:`UIVenus2BaseAdapter.enable_thinking`, which is the single most
   load-bearing setting in this family.

3. **Full text history, newest-N screenshots.** Every completed turn's assistant
   text is replayed verbatim; only screenshots are evicted, and an evicted
   observation stays as an EMPTY user bubble so the user/assistant alternation
   the model was trained on is preserved. ``n_history_images`` counts PAST
   screenshots — the current one is always sent on top — matching upstream's
   ``N_IMG``. See :class:`~lite.agents.models.ui_venus_2.protocol.UIVenus2HistoryProtocol`.

4. **Identity coordinates.** The prompts spell the inclusive maximum as 999 and
   the harnesses divide by 999 (computer) or 1000 (browser / mobile) — a
   sub-pixel difference from cua-lite's canonical [0, 1000] — so no adapter here
   rescales coordinates.

Usage:
    from lite.agents.models.ui_venus_2.adapter import UIVenus2DesktopUseAdapter

    adapter = UIVenus2DesktopUseAdapter()
    agent_sample = adapter.unroll(sample)
"""

from __future__ import annotations

import copy
import dataclasses
import datetime
import logging
import re
from typing import Any

from lite.agents.core.action_space import BaseActionSpace
from lite.agents.core.adapter import BaseAgentAdapter
from lite.agents.core.protocol.base import FullHistoryProtocol
from lite.agents.models.ui_venus_2.action_space import (
    GROUNDING_POINT_NATIVE_NAME,
    UIVenus2BrowserActionSpace,
    UIVenus2DesktopActionSpace,
    UIVenus2GroundingPointActionSpace,
    UIVenus2MobileActionSpace,
    parse_action_text,
)
from lite.agents.models.ui_venus_2.protocol import UIVenus2HistoryProtocol
from lite.agents.types import AgentMessage, AgentStep
from lite.core import (
    LiteCUAMetadata,
    LiteMessage,
    LiteSample,
)
from lite.core.messages import (
    get_inline_reasoning,
    instruction_text,
    make_assistant_content,
    message_has_image,
)
from lite.core.messages.final import MODEL_OUTPUT_ERROR_KEY, mark_model_output_error
from lite.core.messages.turns import truncate_sample_to_turn
from lite.core.tools.calls import tool_call_name
from lite.utils.image import smart_resize

logger = logging.getLogger(__name__)

# =============================================================================
# Vision constants
# =============================================================================
# ``preprocessor_config.json`` on ``inclusionAI/UI-Venus-2-9B``: patch_size 16 x
# merge_size 2 -> a 32-px grid. The pixel band is the one the grounding client
# pins per image (``min_pixels`` 3136, ``max_pixels`` 12845056); the computer and
# mobile examples leave the server at its 16.78 MP checkpoint default, which
# makes a 4K screenshot's cost depend on the serving config instead of the run.
# Pinning the same band everywhere keeps a run reproducible; override it per row
# with ``agent_kwargs.smart_resize_max_pixels``.
SMART_RESIZE_FACTOR = 32
SMART_RESIZE_MIN_PIXELS = 3136
SMART_RESIZE_MAX_PIXELS = 12845056

#: Labels the computer and mobile harnesses put in front of each screenshot.
#: The browser harness sends none, so that adapter blanks both.
HISTORY_SCREENSHOT_LABEL = "History Screenshot:\n"
CURRENT_SCREENSHOT_LABEL = "Current Screenshot:\n"

# =============================================================================
# Prompts — verbatim SFT text, extracted from the reference sources
# =============================================================================
# DO NOT edit these strings without checking them against upstream: every byte
# was part of the SFT distribution. Two placeholder renames are the only local
# change, so the whole family substitutes one vocabulary: the browser prompt's
# ``{task}`` and the grounding prompt's ``{instruction}`` are both spelled
# ``{user_task}`` here.

COMPUTER_SYSTEM_PROMPT = r"""**You are a GUI Agent.**
Your role is to analyze the user's task, provide clear and accurate answers to their questions, and execute the task with precise actions on a desktop operating system. The password of the computer is {sudo_password}.

### Available Actions
You may execute one of the following functions. Coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).
- Click(box=(x1, y1)), or Click()
> Perform a left-click at `box`, or at the current cursor position when `box` is omitted.
- DoubleClick(box=(x1, y1)), or DoubleClick()
> Perform a double-click (selects a word in text). Use `box` to move first, or omit it to act at the current cursor position.
- TripleClick(box=(x1, y1)), or TripleClick()
> Perform a triple-click (selects a line or the content of a single-line input). Use `box` to move first, or omit it to act at the current cursor position.
- RightClick(box=(x1, y1)), or RightClick()
> Perform a right-click to open a context menu. Use `box` to move first, or omit it to act at the current cursor position.
- MiddleClick(box=(x1, y1)), or MiddleClick()
> Perform a middle-click (for example, open a link in a new tab). Use `box` to move first, or omit it to act at the current cursor position.
- Hover(box=(x1, y1))
> Move the cursor immediately to the coordinate WITHOUT clicking.
- Drag(end=(x2, y2), start=(x1, y1))
> Drag to `end` using a fixed 0.5-second drag. `start` is optional; omit it to begin from the current cursor position.
- Swipe(amount=-5, axis='vertical')
> Scroll at the current cursor position. `amount` is an integer from -4096 to 4096 and controls magnitude and direction: vertical positive scrolls up and negative scrolls down; horizontal positive scrolls right and negative scrolls left.
- Type(content='')
> Type the provided text into the focused field. Each `\n` presses Enter.
- Hotkey(keys=['ctrl', 'c'], repeat=1)
> Press 1 to 128 listed keys as a keyboard shortcut. Use `repeat=N` from 1 to 128 to press the shortcut N times.
- KeyDown(keys=['shift'])
> Press 1 to 128 listed keys in order and keep them held across later actions and model turns until a matching `KeyUp`.
- KeyUp(keys=['shift'])
> Release 1 to 128 listed keys in order.
- MouseDown(box=(x1, y1)), or MouseDown()
> Optionally move to `box`, then press and hold the left mouse button across later actions and model turns.
- MouseUp(box=(x1, y1)), or MouseUp()
> Optionally move to `box`, then release the left mouse button.
- Sequence(actions=[Click(box=(x1, y1)), Hotkey(keys=['ctrl', 's'])])
> Execute 2 to 32 actions in order as one open-loop model turn. Nested `Sequence` is not allowed, and `CallUser` or `Finished` may appear only as the final action.
- Wait()
> Wait for the current page, animation, or content to finish loading.
- CallUser(content='')
> Request user takeover or report failure when the task cannot be completed or additional information is required.
- Finished(content='')
> Mark the task as completed successfully and optionally report details in `content`.

### Instructions
- Make sure you understand the task goal to avoid wrong actions.
- Prefer one atomic action per turn. Use `Sequence` only when every child action is already known and no intermediate screenshot is needed; its children execute open-loop.
- `KeyDown`, `KeyUp`, `MouseDown`, and `MouseUp` preserve input state across turns. Release held input explicitly when it is no longer needed.
- Any `keys` list may contain at most 128 non-empty key names; each key name is limited to 1,024 characters.
- `Swipe` is the only scrolling action and always scrolls at the current cursor position.
- Make sure you carefully examine the current screenshot. Sometimes the summarized history might not be reliable, over-claiming some effects.
- To submit/search after typing into a field, end the text with a newline — `Type(content='query\n')` — which types the text and presses Enter in one action.
- To replace the existing content of an input field, use `TripleClick` to select it, then `Type` the new content.
- To open a submenu/dropdown, use `Hover` over the parent item to reveal it, then `Click` the desired entry.
- To use a context menu, `RightClick` the target to open it, then `Click` the desired entry.
- To hold a modifier during another action, use `KeyDown`, the target action, and `KeyUp`. Put them in one `Sequence` only when no intermediate screenshot is needed.
- After launching an app, running a command, downloading, or any slow operation, use `Wait()` to let it finish before continuing.
- To press a key or shortcut several times, use `repeat`, e.g. `Hotkey(keys=['down'], repeat=5)` or `Hotkey(keys=['ctrl', 'z'], repeat=3)`, instead of repeating the action.
- Consider exploring the screen by using the `Swipe` action to scroll and reveal additional content.
- Use `Hotkey` for keyboard shortcuts: copy (`ctrl+c`), paste (`ctrl+v`), save (`ctrl+s`), undo (`ctrl+z`), find (`ctrl+f`), etc.
- If the task cannot be completed or additional information is needed, use `CallUser`. Use `Finished` only after successful completion.

### Output Format
<think> your thinking process </think>
<action> the next action </action>

### User Task
{user_task}"""


BROWSER_SYSTEM_PROMPT = r"""**You are a GUI Browser Agent.**
Your task is to analyze a given user task, review current screenshot and previous actions, and determine the next action to complete the task.

### Available Actions
You may execute one of the following functions:
- Click(point=(x1,y1))
> Perform a tap action at the specified screen coordinate. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).
- Drag(start=(x1,y1), end=(x2,y2))
> Perform a drag action by long-pressing at the start coordinate for a few seconds and then dragging to the end coordinate. This is typically used for adjusting element layouts, moving sliders, solving slider captchas, etc. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).
- Scroll(point=(x1, y1), direction='up/down/left/right')
> Perform a scroll action on coordinate (x1, y1). This is typically used for scrolling to find content, switching tabs, pulling down the notification shade, etc. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999), scroll direction='up/down/left/right'.
- Type(content='')
> Enter the specified text into the currently active input field.
- Launch(url='')
> Launch the target url. Use this action when the target website is not currently visible on the screen.
- Wait()
> Wait for the current page, animation, or content to finish loading.
- GetUrl()
> Get the URL of the current browser tab. The URL is returned at the beginning of the next user message.
- Finished(content='')
> Mark the task as completed and inform the user of the task execution status.
- TakeNote(content='')
> Record important information from screenshots avoiding forgetting.
- CallUser(content='')
> Request user takeover or additional information when needed, for example, when there are multiple on-screen options that satisfy the requirement.
- LongPress(point=(x1,y1), duration=20)
> Press and hold the specified screen coordinate for `duration` seconds. `duration` must be a positive number and defaults to 20 when omitted. This can trigger additional options, such as copy, forward, or delete. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).
- PressBack()
> Return to the previous page.
- PressHome()
> Press the Home key to scroll to the top of the current page.
- PressEnter()
> Perform an Enter key action.
- Hover(point=(x1,y1))
> Perform a hover action at the specified screen coordinate. This can be used to reveal additional information or options, such as tooltips, dropdown menus, etc. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).
- DoubleClick(point=(x1,y1))
> Perform a double tap action at the specified screen coordinate. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).
- Hotkey(keys=('ctrl', 'c'))
> Press combination keys. Keys with comma and wrap each key in single quotes. Do not use more than 3 keys in one Hotkey action. Use `Hotkey(keys=('ctrl', 'tab'))` for the next browser tab or add `shift` for the previous tab.
- SelectOption(index=3)
> Choose an option from the last clicked native HTML select element. Use this only when the current user message provides an explicit native select option list. Do not use keyboard navigation hotkeys for native select dropdowns. For custom dropdowns or visible menu items in the screenshot, click the visible option directly.

### Instruction
- Today is {current_date}.
- Make sure you understand the task goal to avoid wrong actions.
- Make sure you carefully examine the the current screenshot. Sometimes the summarized history might not be reliable, over-claiming some effects.
- If additional information is needed during task execution, use `CallUser` to interact with the user.
- Consider exploring the screen by using the `Scroll` action with different directions to reveal additional content.
- Try to use simple language when searching.
- If you meet ERR_CONNECTION_CLOSED or 404 NOT FOUND error, please type the website key word in https://www.google.com to find the correct url.
- The official website of cryptpad is https://cryptpad.fr/ .
- Distinguish textbox from button: never `Type` into a button. If no textbox is visible, try clicking the search icon first — the input field may appear afterward.
- Strictly avoid repeating the same action when the webpage remains unchanged — you may have executed the wrong action. Continuous use of `Wait()` is also NOT allowed.

# Very Important
Take Notes:
- You are forgetful and will forget all information from the current screenshot before you scroll to next one. When you see important information(e.g. partial step info) for completing the task in the current screenshot, RECORD it using `TakeNote(content='...')` before you scrolling it down.
- The information needed for a task is often distributed across multiple pages. Even partial information should be taken note of — do not wait until all information is seen.
- Before you take `scroll` action, make sure you have taken notes for all important information in the current screenshot.

Apply Filters:
- If filters are available on the page, prioritize using filters for precise searching rather than using the search function for fuzzy searching.

### Output Format
<think> your thinking process </think>
<action> the next action </action>

### User Task
{user_task}
"""


MOBILE_SYSTEM_PROMPT = r"""**You are a GUI Agent.** Your role is to analyze the user's task, provide clear and accurate answers to their questions, and execute the task with precise actions.

### Available Actions
You may execute one of the following functions:

- Click(point=(x1, y1))
> Perform a tap action at the specified screen coordinate. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).

- Drag(start=(x1, y1), end=(x2, y2))
> Perform a drag action by long-pressing at the start coordinate for a few seconds and then dragging to the end coordinate. This is typically used for adjusting app layouts, moving sliders, solving slider captchas, etc. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).

- Swipe(start=(x1, y1), end=(x2, y2))
> Perform a swipe action by dragging from the start coordinate to the end coordinate. This is typically used for scrolling to find content, switching tabs, pulling down the notification shade, etc. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).

- DoubleClick(point=(x1, y1))
> Perform a double tap action at the specified screen coordinate. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).

- LongPress(point=(x1, y1))
> Perform a long-press action at the specified screen coordinate for a certain duration. This can be used to trigger additional options, such as copy, forward, delete, etc. Valid coordinates range from the top-left corner (0, 0) to the bottom-right corner (999, 999).

- Type(content='')
> Enter the specified text into the currently active input field.

- LaunchApp(app='')
> Launch the target app. Use this action when the target app is not currently visible on the screen.

- Wait()
> Wait for the current page, animation, or content to finish loading.

- CallUser(content='')
> Request user takeover or additional information when needed, for example, when there are multiple on-screen options that satisfy the requirement.

- GetScreenshot()
> Take a screenshot and save it to the device's photo album.

- PressBack()
> Return to the previous screen.

- PressHome()
> Return to the system home screen.

- PressEnter()
> Perform an Enter key action.

- PressRecent()
> Open the system recent apps screen.

- Answer(content='')
> Answer the user's questions as requested.

- Finished(content='')
> Mark the task as completed and inform the user of the task execution status.

### Instructions
- Make sure you understand the task goal to avoid wrong actions.
- Make sure you carefully examine the current screenshot. Sometimes the summarized history might not be reliable, over-claiming some effects.
- If additional information is needed during task execution, use `CallUser` to interact with the user.
- Consider exploring the screen by using the `Swipe` action with different directions to reveal additional content.
- To copy text: first select the exact text you want to copy, which usually also brings up the text selection bar, then click the `copy` button in bar.
- To paste text into a text box, first long press the text box, then usually the text selection bar will appear with a `paste` button in it.

### Output Format
<think> your thinking process </think>
<action> the next action </action>

### User Task
{user_task}"""


GROUNDING_USER_PROMPT = r"""Output the center point of the position corresponding to the following instruction: 
{user_task}. 

The output should just be the coordinates of a point, in the format [x,y]. Additionally, if the task is infeasible (e.g., the task is not related to the image), the output should be [-1,-1]."""

#: The grounding checkpoint's trained infeasible marker.
GROUNDING_INFEASIBLE_POINT = [-1, -1]
#: ``report_infeasible`` requires a reason and ``[-1,-1]`` carries none. The
#: marker is the whole message, so the reason names it rather than inventing
#: prose the model never wrote.
GROUNDING_INFEASIBLE_REASON = (
    "UI-Venus-2 answered with the [-1,-1] infeasible marker: the requested "
    "element is not present in the screenshot."
)

_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)
#: The template OPENS the think block in the generation prompt, so the model's
#: reasoning normally arrives with the CLOSING tag only.
_THINK_CLOSE_RE = re.compile(r"</think\s*>", re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think\s*>", re.IGNORECASE)
#: ``<action>`` is the SFT-trained grammar marker. Seeing it with nothing
#: parseable inside means the model ATTEMPTED an action and got the syntax
#: wrong, which must become a terminal parse-failure final rather than be
#: mistaken for a clean content-only final.
_ACTION_OPEN_RE = re.compile(r"<\s*action\s*>", re.IGNORECASE)
#: ``[x,y]`` / ``[x1,y1,x2,y2]`` anywhere in a grounding answer, mirroring
#: ``extract_coordinates_qwen35``'s bbox-before-point precedence.
_POINT_RE = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")
_BBOX_RE = re.compile(
    r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]"
)
_TWO_POINT_RE = re.compile(
    r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*,\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]"
)


def _action_parse_error(action_str: str) -> str:
    tail = f": {action_str!r}" if action_str else ""
    return (
        "malformed <action> block: expected exactly one block holding one call "
        "like Click(box=(x, y))" + tail
    )


def _first_text(message: AgentMessage) -> str:
    for part in message.get("content") or []:
        if part.get("type") == "text" and part.get("text"):
            return part["text"]
    return ""


def _group_has_image(group: list[LiteMessage]) -> bool:
    """Does this turn's observation block carry a screenshot?"""
    return any(message_has_image(message) for message in group)


def _reasoning_text(text: str) -> str:
    """The model's ``<think>`` content, however the tags ended up split.

    The generation prompt ends with an OPEN ``<think>\n`` (see
    :attr:`UIVenus2BaseAdapter.enable_thinking`), so a live response carries only
    the closing tag::

        thought about the screen</think>\n<action>Click(box=(1, 2))</action>

    Matching ``<think>(.*?)</think>`` would find nothing there and silently drop
    the whole reasoning trail from the replayed history. Everything before the
    first ``</think>`` is the reasoning; a leading ``<think>`` is stripped when
    the model does emit one (a re-rendered history turn always does).
    """
    match = _THINK_CLOSE_RE.search(text)
    if not match:
        return ""
    head = text[: match.start()]
    open_tag = list(_THINK_OPEN_RE.finditer(head))
    if open_tag:
        head = head[open_tag[-1].end():]
    return head.strip()


def _sole_action_body(text: str) -> str | None:
    """The one ``<action>`` body, or ``None`` when the turn has 0 or 2+ of them.

    Upstream ``parse_response`` raises on anything but exactly one block. Taking
    the first of several would execute an action the model did not single out,
    which is the failure this whole tag grammar exists to prevent.
    """
    blocks = _ACTION_RE.findall(text)
    return blocks[0].strip() if len(blocks) == 1 else None


# =============================================================================
# UI-Venus-2 Base Adapter
# =============================================================================

@dataclasses.dataclass
class UIVenus2BaseAdapter(BaseAgentAdapter):
    """Shared UI-Venus-2 wire codec and prompt assembly.

    Subclasses supply the system prompt, the action space, and (where the
    prompt has extra slots) the values for them.
    """

    #: SFT text with a ``{user_task}`` slot, and whatever extra slots the surface
    #: declares in :meth:`_system_prompt_fields`. Substituted whole; no gate edits
    #: it. This is the base's own ``system_prompt`` field rather than a second
    #: near-synonym, so the repo-wide prompt gates see these prompts.
    system_prompt: str | None = None
    history_screenshot_label: str = HISTORY_SCREENSHOT_LABEL
    current_screenshot_label: str = CURRENT_SCREENSHOT_LABEL
    #: Forwarded by the agent to ``apply_chat_template(enable_thinking=...)``.
    #:
    #: This is NOT the usual "separate reasoning channel" switch, and it must
    #: stay ``True``. The checkpoint's template branches like this::
    #:
    #:     {%- if enable_thinking is defined and enable_thinking is false %}
    #:         {{- '<think>\n\n</think>\n\n' }}   # CLOSED before generation
    #:     {%- else %}
    #:         {{- '<think>\n' }}                  # left OPEN for the model
    #:
    #: Every UI-Venus-2 prompt ends with ``### Output Format`` demanding
    #: ``<think> your thinking process </think>`` — so the ``else`` branch IS the
    #: trained distribution, and it is what the computer and mobile harnesses
    #: get (``call_model`` only sends ``chat_template_kwargs`` when thinking is
    #: on, so ``False`` never reaches the template). Closing the block instead
    #: leaves the model nowhere to reason: measured on lite.osworld, 95% of
    #: turns came back as a bare ``<action>``, and the score collapsed. (For
    #: scale, the checkpoint card reports 70.8 for this 9B on OSWorld-Verified,
    #: a different 361-task bench; the 80.5 in that table is the 27B.)
    #:
    #: The browser harness is the one upstream caller that really does send
    #: ``enable_thinking: false`` (``LLM_THINKING`` defaults to ``"false"``).
    #: That contradicts its own prompt's output format, so it is not the default
    #: here; set ``agent_kwargs.enable_thinking: false`` on a browser row to
    #: reproduce it exactly.
    enable_thinking: bool = True

    smart_resize_factor: int = SMART_RESIZE_FACTOR
    smart_resize_min_pixels: int = SMART_RESIZE_MIN_PIXELS
    smart_resize_max_pixels: int = SMART_RESIZE_MAX_PIXELS

    # -------------------------------------------------------------------------
    # Images
    # -------------------------------------------------------------------------

    def _process_image_after_target(self, img):
        """Stage-2 hook: smart_resize onto UI-Venus-2's 32-px grid + pixel band."""
        width, height = img.size
        new_h, new_w = smart_resize(
            height=height,
            width=width,
            factor=self.smart_resize_factor,
            min_pixels=self.smart_resize_min_pixels,
            max_pixels=self.smart_resize_max_pixels,
        )
        if (new_w, new_h) != (width, height):
            img = img.resize((new_w, new_h))
        return img

    # -------------------------------------------------------------------------
    # sample -> agent
    # -------------------------------------------------------------------------

    def render_step(
        self,
        sample: LiteSample,
        k: int,
        processed,
        **kwargs,
    ) -> AgentStep:
        """Render turn ``k`` as ``[system, (observation, assistant)*, observation]``.

        Matches ``build_messages`` on every UI-Venus-2 surface: the goal goes
        into the system prompt, each completed turn keeps its assistant text,
        and only the newest screenshots survive — an evicted observation stays
        as an empty user bubble so the alternation is unbroken.
        """
        truncated = truncate_sample_to_turn(sample, k)
        messages = self.protocol.process_messages(truncated.messages)

        # Group CONSECUTIVE observations into one bubble. An assistant always
        # separates turns, so a run of user/tool messages is exactly one turn's
        # observation block -- which upstream sends as a SINGLE user message.
        groups: list[list[LiteMessage]] = []
        rendered: list[AgentMessage | None] = []
        for message in messages:
            if message.get("role") == "assistant":
                rendered.append(self.convert_message_to_agent(message))
                continue
            if rendered and rendered[-1] is None:
                groups[-1].append(message)
            else:
                groups.append([message])
                rendered.append(None)

        # The "Current Screenshot" is the LAST group that actually carries one:
        # a trailing text-only tool result must not steal the label from the
        # screenshot the model is looking at.
        current_group = max(
            (index for index, group in enumerate(groups) if _group_has_image(group)),
            default=None,
        )

        result_messages: list[AgentMessage] = [{
            "role": "system",
            "content": [{"type": "text", "text": self._render_system_prompt(messages)}],
        }]
        group_index = 0
        for entry in rendered:
            if entry is not None:
                result_messages.append(entry)
                continue
            label = (
                self.current_screenshot_label
                if group_index == current_group
                else self.history_screenshot_label
            )
            result_messages.append(self._render_observation(groups[group_index], label))
            group_index += 1
        return result_messages

    def _render_system_prompt(self, messages: list[LiteMessage]) -> str:
        return (self.system_prompt or "").format(
            user_task=instruction_text(messages),
            **self._system_prompt_fields(),
        )

    def _system_prompt_fields(self) -> dict[str, Any]:
        """Extra ``{...}`` slots this surface's prompt carries beyond the task."""
        return {}

    def _render_observation(
        self,
        group: list[LiteMessage],
        label: str,
    ) -> AgentMessage:
        """One turn's observations as a single bubble: feedback, then label + image.

        Ordinary user text is dropped — it is the goal, which the system prompt
        already carries, and printing it twice is off-distribution.
        ``role:"tool"`` text is NOT ordinary text: it is per-call result/error
        feedback that no later screenshot re-reveals. It leads the bubble because
        that is where the browser harness puts it (``make_user`` builds
        ``[feedback, image]``, and ``GetUrl``'s result is documented as arriving
        "at the beginning of the next user message").
        """
        images: list[dict[str, Any]] = []
        tool_texts: list[dict[str, Any]] = []
        for message in group:
            content = message.get("content")
            # ``LiteUserMessage.content`` is ``str | list``; a bare str carries no
            # image, and its text is the goal, already in the system prompt.
            if not isinstance(content, list):
                continue
            is_tool = message.get("role") == "tool"
            for part in content:
                if part.get("type") == "image":
                    images.append(part)
                elif is_tool and part.get("type") == "text" and part.get("text"):
                    tool_texts.append(part)

        parts: list[dict[str, Any]] = list(tool_texts)
        if images:
            if label:
                parts.append({"type": "text", "text": label})
            parts.extend(images)
        # No image and no feedback: the screenshot was evicted by the image
        # window. Upstream still emits the (empty) user turn, so the
        # user/assistant alternation the model saw in training is preserved.
        return {"role": "user", "content": parts}

    # -------------------------------------------------------------------------
    # per-message conversion
    # -------------------------------------------------------------------------

    def _convert_message_to_agent(
        self,
        message: LiteMessage,
        **kwargs,
    ) -> dict[str, Any]:
        """Fold an assistant message into ``<think>…</think>\\n<action>…</action>``.

        Structured ``tool_calls`` and ``reasoning_content`` are dropped so the
        chat_template only sees the byte-exact wire text.
        """
        result = copy.deepcopy(message)
        if result.get("role") != "assistant":
            return result

        had_tool_calls = bool(message.get("tool_calls"))
        blocks: list[str] = []
        think = get_inline_reasoning(message)
        if think:
            blocks.append(f"<think>{think}</think>")
        if had_tool_calls:
            agent_tool_calls = self.action_space.convert_tool_calls_to_agent(
                message["tool_calls"]
            )
            action_text = self.action_space.format_tool_calls_as_text(agent_tool_calls)
            blocks.append(f"<action>{action_text}</action>")
        else:
            # A content-only final can now carry BOTH reasoning and answer text
            # (the parser splits them on ``</think>``). Its answer is the whole
            # point of the turn -- an SFT/DAgger target that renders the think
            # block and stops teaches the model to reason and never answer.
            blocks.extend(
                part["text"] for part in (message.get("content") or [])
                if isinstance(part, dict)
                and part.get("type") == "text"
                and part.get("text")
            )

        result.pop("tool_calls", None)
        result.pop("reasoning_content", None)
        if blocks:
            result["content"] = [{"type": "text", "text": "\n".join(blocks)}]
        else:
            result["content"] = []
        return result

    def convert_message_from_agent(
        self,
        message: AgentMessage,
        **kwargs,
    ) -> LiteMessage:
        """Parse the tagged wire text back into a cua-lite assistant message.

        UI-Venus-2's ``<think>`` / ``<action>`` tags are prompt-defined plain
        text, not chat-template sentinels, so all parsing lives here and
        :meth:`parse_raw_assistant_response` is a trivial text wrapper. A served
        endpoint that returns native ``reasoning_content`` has it spliced into a
        ``<think>`` block upstream before it ever reaches this parser, so there
        is one reasoning surface either way.
        """
        result = copy.deepcopy(message)
        if result.get("role") != "assistant":
            return result

        raw_text = _first_text(message)
        if not raw_text:
            # API path: no text, only structured calls to normalize.
            if "tool_calls" in result:
                result["tool_calls"] = self._route_agent_tool_calls_to_lite(
                    result["tool_calls"]
                )
            return result

        think = _reasoning_text(raw_text)
        action_str = _sole_action_body(raw_text)
        agent_tool_calls = self._parse_wire_action(action_str) if action_str else []
        # Routed LAST, not on the parsed projection: a block that parses as a
        # call but converts to nothing (an empty ``Sequence``) must stay a parse
        # failure instead of collapsing into a clean content-only final.
        tool_calls = (
            self._route_agent_tool_calls_to_lite(agent_tool_calls)
            if agent_tool_calls else []
        )

        if tool_calls:
            result["tool_calls"] = tool_calls
        elif "tool_calls" in result and not _ACTION_OPEN_RE.search(raw_text):
            result["tool_calls"] = self._route_agent_tool_calls_to_lite(
                result["tool_calls"]
            )
        else:
            result.pop("tool_calls", None)
            if _ACTION_OPEN_RE.search(raw_text):
                mark_model_output_error(result, _action_parse_error(action_str or ""))

        if not result.get("tool_calls") and MODEL_OUTPUT_ERROR_KEY not in result:
            close = list(_THINK_CLOSE_RE.finditer(raw_text))
            answer = raw_text[close[-1].end():] if close else raw_text
            if not answer.strip() and close:
                # Closed, then EOS: the answer is everything BEFORE the tag.
                # Splitting here would submit an empty string, and keeping the
                # raw text would ship the literal ``</think>`` to a grader that
                # string-matches it.
                answer, think = raw_text[: close[-1].start()], ""
            # Content-only final: this text becomes the model's ANSWER through
            # ``summarize_no_tool_call_final``, so it must not carry the CoT or
            # the literal ``</think>``. Answer-graded envs (AndroidWorld,
            # MobileWorld) string-match it. Same split qwen3_5 does.
            result["content"] = make_assistant_content(
                inline_reasoning=think,
                text=answer.strip(),
            )
            return result

        result["content"] = make_assistant_content(inline_reasoning=think)
        return result

    def _parse_wire_action(self, action_str: str) -> list[dict[str, Any]]:
        """The ``<action>`` body as bare ``{name, arguments}`` projections."""
        parsed = parse_action_text(action_str)
        if parsed is None:
            logger.warning("Failed to parse UI-Venus-2 action: %s", action_str)
            return []
        return [parsed]

    def parse_raw_assistant_response(
        self,
        response: str,
        **kwargs,
    ) -> AgentMessage:
        """Wrap raw UI-Venus-2 output verbatim; parsing lives in
        :meth:`convert_message_from_agent`.
        """
        return {"role": "assistant", "content": [{"type": "text", "text": response}]}


# =============================================================================
# ``use`` adapters
# =============================================================================
# Each platform gets its OWN key: unlike UI-Venus-1.5, the three ``use``
# grammars genuinely differ (``box=`` vs ``point=``, ``Swipe(amount, axis)`` vs
# ``Scroll(point, direction)`` vs ``Swipe(start, end)``), so there is no
# ``(desktop|browser)`` regex to share here.

@dataclasses.dataclass
class UIVenus2DesktopUseAdapter(UIVenus2BaseAdapter, key="ui_venus_2@desktop@use"):
    """Desktop OS ``use``: the Computer harness.

    The one surface with ``Sequence(actions=[...])``, which is this family's
    spelling of cua-lite's ``computer`` action batch — a canonical batch of two
    or more actions renders as one ``Sequence`` and parses straight back.
    """

    action_space: BaseActionSpace = dataclasses.field(
        default_factory=UIVenus2DesktopActionSpace
    )
    protocol: UIVenus2HistoryProtocol = dataclasses.field(
        default_factory=UIVenus2HistoryProtocol
    )
    system_prompt: str | None = COMPUTER_SYSTEM_PROMPT
    #: The Computer prompt states the machine's sudo password verbatim so the
    #: model can answer a privilege prompt. It is a PER-ENVIRONMENT secret, not a
    #: model property, and the envs disagree: the upstream OSWorld v1 image uses
    #: ``password`` (``lite/gym/envs/osworld/main.py::_CLIENT_PASSWORD``) while the
    #: lite sandbox's user AND password are both ``user``
    #: (``lite/gym/envs/lite/osworld/src/utils/dispatch.py::_TEMPLATE_VARS``).
    #: The default matches upstream's ``--sudo-password``; every shipped desktop
    #: row sets ``agent_kwargs.sudo_password`` explicitly so no row inherits a
    #: password its VM will reject.
    sudo_password: str = "password"

    def _system_prompt_fields(self) -> dict[str, Any]:
        return {"sudo_password": self.sudo_password}


@dataclasses.dataclass
class UIVenus2BrowserUseAdapter(UIVenus2BaseAdapter, key="ui_venus_2@browser@use"):
    """Browser ``use``: the Venus browser-plugin harness.

    Two upstream details are preserved here: the screenshot carries no label
    (``make_user`` sends the image bare), and the prompt is dated — ``Today
    is {current_date}`` — which upstream fills with the wall clock. A run whose
    prompt changes by the day is not reproducible, so ``current_date`` is a
    field: leave it ``None`` for upstream's behavior, or pin an ISO date in the
    row's yaml to freeze the prompt.
    """

    action_space: BaseActionSpace = dataclasses.field(
        default_factory=UIVenus2BrowserActionSpace
    )
    protocol: UIVenus2HistoryProtocol = dataclasses.field(
        default_factory=UIVenus2HistoryProtocol
    )
    system_prompt: str | None = BROWSER_SYSTEM_PROMPT
    history_screenshot_label: str = ""
    current_screenshot_label: str = ""
    current_date: str | None = None

    def _system_prompt_fields(self) -> dict[str, Any]:
        return {
            "current_date": self.current_date or datetime.date.today().isoformat(),
        }


@dataclasses.dataclass
class UIVenus2MobileUseAdapter(UIVenus2BaseAdapter, key="ui_venus_2@mobile@use"):
    """Android ``use``: the Venus mobile framework harness."""

    action_space: BaseActionSpace = dataclasses.field(
        default_factory=UIVenus2MobileActionSpace
    )
    protocol: UIVenus2HistoryProtocol = dataclasses.field(
        default_factory=UIVenus2HistoryProtocol
    )
    system_prompt: str | None = MOBILE_SYSTEM_PROMPT


# =============================================================================
# Grounding adapter (single-step point)
# =============================================================================

@dataclasses.dataclass
class UIVenus2GroundingPointAdapter(
    UIVenus2BaseAdapter,
    key=r"ui_venus_2@(desktop|browser|mobile)@grounding\.point",
):
    """Grounding (single-step point) for UI-Venus-2, all platforms.

    Upstream ships ONE grounding harness — ``eval_multi_benchmark.py`` scores
    ScreenSpot-Pro, OSWorld-G and the mobile splits of VenusBench-GD through the
    same prompt — so one regex-keyed class serves all three platforms.

    The wire is nothing like the ``use`` surfaces: there is no system message and
    no tag block, the answer is a bare ``[x,y]`` list, and ``[-1,-1]`` is the
    trained refusal, which lowers to the env's ``report_infeasible`` extra tool
    (the osworld_g / screenspot_pro convention).
    """

    action_space: BaseActionSpace = dataclasses.field(
        default_factory=UIVenus2GroundingPointActionSpace
    )
    metadata: LiteCUAMetadata = dataclasses.field(
        default_factory=lambda: LiteCUAMetadata(
            dims=(
                LiteCUAMetadata.Platform.DESKTOP.value,
                LiteCUAMetadata.TaskType.GROUNDING_POINT.value,
            )
        )
    )
    protocol: FullHistoryProtocol = dataclasses.field(
        default_factory=FullHistoryProtocol
    )
    #: The grounding client sends a user-only message, so there is no system
    #: prompt at all; ``user_prompt_template`` is the whole instruction surface.
    system_prompt: str | None = None
    user_prompt_template: str = GROUNDING_USER_PROMPT

    def render_step(
        self,
        sample: LiteSample,
        k: int,
        processed,
        **kwargs,
    ) -> AgentStep:
        """Render as ONE user message: image first, then the prompt.

        Matches ``Qwen35GroundModel._build_messages`` — no system message, and
        the image precedes the text (the ``use`` surfaces put text first).
        """
        truncated = truncate_sample_to_turn(sample, k)
        messages = self.protocol.process_messages(truncated.messages)

        target: LiteMessage | None = None
        body = messages
        if body and body[-1].get("role") == "assistant":
            target = body[-1]
            body = body[:-1]

        content: list[dict[str, Any]] = []
        for message in body:
            parts = message.get("content")
            if not isinstance(parts, list):
                continue
            content.extend(part for part in parts if part.get("type") == "image")
        content.append({
            "type": "text",
            "text": self.user_prompt_template.format(
                user_task=_strip_trailing_period(instruction_text(messages)),
            ),
        })

        result_messages: list[AgentMessage] = [{"role": "user", "content": content}]
        if target is not None:
            result_messages.append(self.convert_message_to_agent(target))
        return result_messages

    def _convert_message_to_agent(
        self,
        message: LiteMessage,
        **kwargs,
    ) -> dict[str, Any]:
        """Render an assistant turn as the bare ``[x,y]`` answer."""
        result = copy.deepcopy(message)
        if result.get("role") != "assistant":
            return result

        text = ""
        tool_calls = message.get("tool_calls") or []
        if any(tool_call_name(call) == "report_infeasible" for call in tool_calls):
            text = _render_point(GROUNDING_INFEASIBLE_POINT)
        else:
            for agent_call in self.action_space.convert_tool_calls_to_agent(tool_calls):
                if agent_call["name"] == GROUNDING_POINT_NATIVE_NAME:
                    text = _render_point(agent_call["arguments"]["box"])
                    break

        result.pop("tool_calls", None)
        result.pop("reasoning_content", None)
        result["content"] = [{"type": "text", "text": text}] if text else []
        return result

    def convert_message_from_agent(
        self,
        message: AgentMessage,
        **kwargs,
    ) -> LiteMessage:
        """Parse ``[x,y]`` / ``[x1,y1,x2,y2]`` / ``[-1,-1]`` into one call."""
        result = copy.deepcopy(message)
        if result.get("role") != "assistant":
            return result

        raw_text = _first_text(message)
        if not raw_text:
            if "tool_calls" in result:
                result["tool_calls"] = self._route_agent_tool_calls_to_lite(
                    result["tool_calls"]
                )
            return result

        agent_tool_call = _parse_grounding_answer(raw_text)
        if agent_tool_call is None:
            result.pop("tool_calls", None)
            result["content"] = [{"type": "text", "text": raw_text}]
            return result

        result["tool_calls"] = self._route_agent_tool_calls_to_lite([agent_tool_call])
        result["content"] = make_assistant_content()
        return result


def _strip_trailing_period(instruction: str) -> str:
    """Upstream drops a trailing ``.`` because the template supplies its own."""
    instruction = instruction.strip()
    return instruction[:-1] if instruction.endswith(".") else instruction


def _render_point(coordinate: list[int]) -> str:
    return f"[{int(coordinate[0])},{int(coordinate[1])}]"


def _parse_grounding_answer(raw_text: str) -> dict[str, Any] | None:
    """One bare projection for a grounding answer, or ``None`` for prose.

    Mirrors ``extract_coordinates_qwen35``: a four-number list and a pair of
    two-number lists are both boxes whose CENTER is the answer, and ``[-1,-1]``
    is the refusal, which becomes ``report_infeasible``.

    The scan starts after the LAST ``</think>``, never at the top of the
    response. :attr:`UIVenus2BaseAdapter.enable_thinking` leaves ``<think>``
    open in the generation prompt, so a grounding response is
    ``reasoning </think> answer``; the reasoning routinely names coordinates it
    goes on to reject, and may name ``[-1,-1]`` while arguing the target IS
    present. Searching the whole response takes whichever bracket comes first
    and silently answers with a rejected candidate.
    """
    # No ``</think>``: the block never closed (truncation, or a checkpoint run
    # with thinking off), so the whole text is the answer span. A blank tail
    # (closed, then EOS) is deliberately NOT special-cased -- falling back to
    # the reasoning would re-answer with the rejected candidate this scan exists
    # to skip, and unlike a ``use`` final there is no text the env could grade.
    # ``_THINK_CLOSE_RE`` also matches ``</think >`` and ``</THINK>``.
    close = list(_THINK_CLOSE_RE.finditer(raw_text))
    if close:
        raw_text = raw_text[close[-1].end():]
    box = _BBOX_RE.search(raw_text) or _TWO_POINT_RE.search(raw_text)
    if box:
        x1, y1, x2, y2 = (int(value) for value in box.groups())
        # ``extract_coordinates_qwen35`` tests the FIRST pair for the marker,
        # before averaging, so ``[-1,-1,5,5]`` is a refusal rather than a point
        # at the midpoint of a nonsense box.
        if [x1, y1] == GROUNDING_INFEASIBLE_POINT:
            return _report_infeasible_call()
        return _grounding_point_call([(x1 + x2) // 2, (y1 + y2) // 2])
    point = _POINT_RE.search(raw_text)
    if not point:
        logger.warning("Failed to parse UI-Venus-2 grounding answer: %s", raw_text)
        return None
    return _grounding_point_call([int(point.group(1)), int(point.group(2))])


def _report_infeasible_call() -> dict[str, Any]:
    return {
        "name": "report_infeasible",
        "arguments": {"reason": GROUNDING_INFEASIBLE_REASON},
    }


def _grounding_point_call(coordinate: list[int]) -> dict[str, Any]:
    if coordinate == GROUNDING_INFEASIBLE_POINT:
        return _report_infeasible_call()
    return {
        "name": GROUNDING_POINT_NATIVE_NAME,
        "arguments": {"box": coordinate},
    }


__all__ = [
    "BROWSER_SYSTEM_PROMPT",
    "COMPUTER_SYSTEM_PROMPT",
    "CURRENT_SCREENSHOT_LABEL",
    "GROUNDING_INFEASIBLE_POINT",
    "GROUNDING_INFEASIBLE_REASON",
    "GROUNDING_USER_PROMPT",
    "HISTORY_SCREENSHOT_LABEL",
    "MOBILE_SYSTEM_PROMPT",
    "UIVenus2BaseAdapter",
    "UIVenus2BrowserUseAdapter",
    "UIVenus2DesktopUseAdapter",
    "UIVenus2GroundingPointAdapter",
    "UIVenus2MobileUseAdapter",
]
