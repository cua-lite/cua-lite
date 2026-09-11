"""Qwen3.5 history protocols.

- :class:`Qwen3_5HistoryProtocol` (``qwen3_5.history``):
  Byte-faithful replica of the OSWorld Qwen3.5-VL reference flow
  (``${CUA_LITE_REFERENCES_ROOT}/OSWorld/mm_agents/qwen35vl_agent.py`` —
  :meth:`Qwen35VLAgent.predict`):

    * ``history_n`` most-recent turns kept as full user/assistant pairs
      (default 100).
    * Older turns are summarized as
      ``"Previous actions:\\nStep N: <action>\\n..."`` lines and injected
      into the first in-window user message. **No cap** on how many
      past actions make it into the summary — reference includes all
      ``range(0, start_step - 1)``.
    * Within the kept window, when the absolute turn count exceeds
      ``image_max`` the oldest ``fold_size`` screenshots are progressively
      replaced by a ``collapse_text`` placeholder. Default ``image_max=4``,
      ``fold_size=4``.

  Folding is computed stateless from the total turn count (reference
  carries it as a monotonic accumulator on the agent — same outcome).

  cua-lite's default keeps ordinary text beside image observations
  (``keep_text_with_images=True``), so image+text env feedback and projected
  tool-result errors remain visible through one renderer-owned text path.
  Configs that need the older pure-GUI screenshot-only reference should set
  ``keep_text_with_images=False`` explicitly.

  **Not** inheriting from ``Qwen3VLHistoryProtocol`` on purpose: keeping
  the Qwen3.5 contract self-contained makes the mapping to reference code
  obvious, and avoids the ``full_history_size`` / ``history_n`` mirror-sync
  bug that inheritance + ``setattr``-via-kwargs introduces.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable
from typing import Any

from lite.agents.core.protocol import TurnWindowProtocol
from lite.agents.core.protocol.window import (
    append_with_boundary_tool_projection,
    collapse_image_observation,
    evicted_tool_result_summary_text,
    filter_history_content,
)
from lite.core import (
    LiteMessage,
)


@dataclasses.dataclass
class Qwen3_5HistoryProtocol(TurnWindowProtocol, key="qwen3_5.history"):
    """Reference-aligned rolling window + image-fold protocol.

    Attributes:
        history_n: Most recent turns kept as full user/assistant pairs.
            Matches the reference ``--history_n`` CLI flag AND its default of
            100. Most envs never reach it, so for them the window is
            effectively "keep every turn". The ones that CAN reach it mostly
            declare their budget in env config or a per-task table rather than
            in the rollout yaml, so grepping ``max_steps`` understates the set:
            ``cua.bench`` kicad (200, still truncates), androidworld
            (``10 * complexity``; 9 of 116 tasks over 50, one at 120),
            mobilegym (L4 = 60, +15 grounded), and webgym EVAL at difficulty
            >= 7 (70). ``cua.bench`` workflows (100) stops truncating entirely
            -- at ``total_turns == history_n`` the window start is 0.
            A config that wants a tighter window sets one -- which is what the
            ``hN`` desktop profiles do.

            Not byte-faithful to the reference at the boundary: it keeps
            ``history_n + 1`` turns where we keep ``history_n`` (see
            ``tests/agents/models/qwen3_5/test_qwen3_5_history_protocol.py``).
            At 50 nothing ran long enough to reach it; at 100 ``cua.bench``
            kicad (200 steps) does. Screenshots are bounded separately by
            ``image_max``/``fold_size``, so a big history_n does NOT mean a big
            pixel budget.
        image_max: Cap on the number of un-folded screenshots in the
            rendered prompt. Matches reference ``image_max``. Default 4
            (reference: 20).
        fold_size: Increment by which the folded prefix grows each time
            it overflows ``image_max``. Matches reference ``fold_size``.
            Default 4 (reference: 10).
        collapse_text: Placeholder that replaces a folded image. Matches
            reference default verbatim.
        summary_template: Jinja-style template for the summary injected
            into the first kept user message. Matches the reference's
            ``instruction_prompt`` text.
        keep_text_with_images: If True, keep ordinary user text in windowed
            messages that also carry images. Text-only messages and role:"tool"
            result text are always preserved; this flag only controls ordinary
            image+text history compression. Set False only for explicit legacy
            pure-GUI reproduction configs.
        action_format_func: Optional ``(LiteMessage) -> str`` override
            for summary action lines. If None, uses the default
            ``"Step N: <action_description>"`` rendering.
    """

    history_n: int = 100
    image_max: int = 4
    fold_size: int = 4
    collapse_text: str = "This screenshot has been collapsed."
    summary_template: str = (
        "Please generate the next move according to the UI screenshot, "
        "instruction and previous actions.\n\n"
        "Instruction: {instruction}\n\n"
        "Previous actions:\n{previous_actions_str}"
    )
    keep_text_with_images: bool = True
    action_format_func: Callable[[LiteMessage], str] | None = None

    # -------------------------------------------------------------------------
    # Entry point
    # -------------------------------------------------------------------------

    def _select_messages(
        self,
        content: list[LiteMessage],
        turns: list[dict[str, Any]],
    ) -> list[LiteMessage]:
        """Reference-aligned multi-turn shaping.

        The surrounding peel/group/re-attach frame lives on
        :class:`TurnWindowProtocol`; this hook owns steps 1-3 below.

        1. Compute window + text summary (all pre-window actions).
        2. Compute folded prefix count from total turn count.
        3. Render kept turns with per-step fold decision.
        """
        if not turns:
            return list(content)
        else:
            kept_turns, summary_text, instruction = self._compute_summary_and_window(turns)
            total_turns = len(turns)
            # The fold budget bounds IMAGES, so it is spent by image-bearing
            # turns only and collapses exactly those. A text-only turn (a `bash`
            # result today) contributes no image, costs nothing, and is never
            # collapsed -- its output is one-shot information that no later
            # screenshot can re-reveal.
            #
            # GUI-only identity: every turn carries an image, so
            # ``image_turn_indices`` is ``range(total_turns)``,
            # ``_compute_folded_count`` receives ``total_turns`` exactly as
            # before, and ``collapsed_turns`` becomes ``{0 .. folded_count-1}``
            # -- which is precisely the old ``step_num <= folded_count`` test
            # rewritten in 0-based absolute indices. Byte-identical, by
            # construction rather than by luck; the render goldens pin it.
            image_indices = self.image_turn_indices(turns)
            folded_count = self._compute_folded_count(len(image_indices))
            collapsed_turns = frozenset(image_indices[:folded_count])
            # Absolute 1-based step of the first kept turn.
            window_first_step = total_turns - len(kept_turns) + 1
            return self._render_user_messages_with_folding(
                kept_turns, summary_text, instruction,
                window_first_step=window_first_step,
                collapsed_turns=collapsed_turns,
            )

    # -------------------------------------------------------------------------
    # Window / summary split
    # -------------------------------------------------------------------------

    def _compute_summary_and_window(
        self,
        turns: list[dict[str, LiteMessage | None]],
    ) -> tuple[list[dict[str, LiteMessage | None]], str, str]:
        """Split turns into (kept_window, summary_text, instruction).

        Window = last ``history_n`` turns.
        Summary = every action BEFORE the window, rendered as
        ``"Step N: <action>"`` lines joined by ``\\n``. No cap —
        reference's ``for i in range(0, start_step - 1)``.
        """
        num_turns = len(turns)
        window_start_idx = max(0, num_turns - self.history_n)

        instruction = self._extract_instruction(turns[0])

        summary_lines: list[str] = []
        for i in range(0, window_start_idx):
            turn = turns[i]
            if turn["assistant"]:
                line = self._format_action(turn["assistant"], i + 1)
                # A text tool's RESULT lands in the NEXT turn's observations
                # (observations precede the assistant that consumes them). It is
                # one-shot information -- no later screenshot re-reveals it -- so
                # carry it into the summary instead of dropping it with the
                # window. Empty for an image-bearing turn, so a GUI-only
                # trajectory renders exactly as before.
                result_text = self._tool_result_summary_text(
                    turns[i + 1] if i + 1 < len(turns) else None
                )
                if result_text:
                    line = f"{line}\n{result_text}"
                summary_lines.append(line)
        summary_text = "\n".join(summary_lines) if summary_lines else "None"

        kept_turns = turns[window_start_idx:]
        return kept_turns, summary_text, instruction

    # -------------------------------------------------------------------------
    # Image fold
    # -------------------------------------------------------------------------

    def _compute_folded_count(self, total_turns: int) -> int:
        """Number of oldest steps whose screenshots get replaced by
        ``collapse_text``. Stateless port of the reference's
        ``_update_folding_state`` accumulator.

        Reference behavior (equivalent fixed point)::

            k = 0
            while (total_turns - k) > image_max:
                k += fold_size
            if k > total_turns:
                k = total_turns

        Closed form: ``ceil((total_turns - image_max) / fold_size) * fold_size``
        when ``total_turns > image_max`` else 0, then capped at ``total_turns``.
        """
        if total_turns <= self.image_max:
            return 0
        overflow = total_turns - self.image_max
        folded = ((overflow + self.fold_size - 1) // self.fold_size) * self.fold_size
        return min(folded, total_turns)

    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------

    def _render_user_messages_with_folding(
        self,
        kept_turns: list[dict[str, LiteMessage | None]],
        summary_text: str,
        instruction: str,
        *,
        window_first_step: int,
        collapsed_turns: frozenset[int],
    ) -> list[LiteMessage]:
        """Render kept_turns with per-step image-fold logic.

        ``collapsed_turns`` holds ABSOLUTE 0-based indices of the turns whose
        image is replaced by ``collapse_text``. It is a set rather than a count
        because the folded prefix is now counted over image-bearing turns, which
        need not be contiguous once text-only turns are interleaved.
        """
        full_prompt = self.summary_template.format(
            instruction=instruction,
            previous_actions_str=summary_text,
        )

        result: list[LiteMessage] = []
        for i, turn in enumerate(kept_turns):
            step_num = window_first_step + i  # 1-based absolute
            is_collapsed = (step_num - 1) in collapsed_turns

            assistant_msg = turn["assistant"]
            observations: list[LiteMessage] = []

            for j, user_msg in enumerate(turn["observations"]):
                has_image = any(
                    c.get("type") == "image"
                    for c in user_msg.get("content", [])
                )
                if i == 0 and j == 0:
                    user_msg = copy.deepcopy(user_msg)
                    if is_collapsed:
                        # Reference: first kept turn collapsed → text-only user msg,
                        # no placeholder (the instruction prompt carries context).
                        user_msg["content"] = [
                            p for p in (user_msg.get("content") or [])
                            if p.get("type") != "image"
                        ]
                    self._inject_text(user_msg, full_prompt)
                elif is_collapsed and has_image:
                    user_msg = collapse_image_observation(
                        user_msg,
                        self.collapse_text,
                        strip_regular_image_text=not self.keep_text_with_images,
                    )
                else:
                    user_msg = filter_history_content(
                        user_msg,
                        strip_regular_image_text=not self.keep_text_with_images,
                    )
                observations.append(user_msg)

            append_with_boundary_tool_projection(result, observations)
            if assistant_msg:
                result.append(assistant_msg)

        return result

    # -------------------------------------------------------------------------
    # Helpers (inlined, no parent class dependency)
    # -------------------------------------------------------------------------

    def _tool_result_summary_text(
        self,
        turn: dict[str, LiteMessage | None] | None,
    ) -> str:
        """Text-tool output that survives when a result turn leaves history_n.

        This is Qwen3.5 history policy, not a base-protocol rule. Text-only
        role:"tool" results are one-shot outputs. Image-bearing results are
        summarized only when structured metadata or cua-lite's exact rendered
        error header marks them as execution-error feedback.
        """
        if turn is None:
            return ""
        return evicted_tool_result_summary_text(turn["observations"])
