"""Shared desktop quality-ANNOTATION pass — the single step before stage/export_sft.

Used by Lite.OSWorld, Lite.CUAGym, Lite.ScaleCUA, and Lite.CUAWorld, and by every
teacher within them: it runs on canonical Lite rows, after the adapter has already
projected each model family's wire format, so it carries no model-family branch.

Pipeline: collect (scripts/rollout.py) → **annotate** (this) → stage (lite.data.hf.stage)
→ upload; consumers download the published canonical dataset before ``export_sft``. Read
every ``trajectory.parquet`` under ``--log-root``, clean + tag each, and write them to
``--out``. Quality gates are recorded in ``metadata.others.exclude_reason`` (comma-joined;
the key is omitted when clean) and the consumer decides — with four hard-drop
exceptions: a trajectory whose agent **typed a ``/opt/env/`` path**, one with an
out-of-range GUI coordinate, one naming a tool the row never declared, and one
naming an action that does not exist. ``/opt/env`` is the env-only tool tree the
agent must not reach; the other three fail the staging row-format check and should
never enter the canonical dataset. All four are physically excluded rather than
left for a downstream threshold.
Everything else is kept. Downstream selects the training set with
``not m.others.get('exclude_reason') and (m.others.get('episode_return') or 0) > 0.5`` — the same
``exclude_reason`` idiom used for task-level exclusion.

Three layers, grounded in per-batch trajectory audits:

  1. STRIP no-op *actions* from inside a trajectory (keeps the trajectory, shortens it):
       --actions screenshot,wait   (default) — the agent is told never to use these (a
         screenshot is auto-taken every step), but it still emits them; they change no state
         and carry zero signal, so teaching the student to emit them is pure harm. Strip.
         For canonical action-batch calls, strip only the no-op child actions and keep the
         original batch id when any non-no-op child remains; remove the following
         tool result only when the whole call/turn is stripped.

     A bare Ctrl+S is NOT a no-op and is KEPT by default. Yes, this env auto-saves LibreOffice
     at eval exit (LO_SAVE_POSTCONFIG), so the 96% of LO trajectories that end on a bare Ctrl+S
     "waste" one step — but saving your work is a GOOD, generalizable habit that is redundant
     only because of THIS harness's quirk. Stripping it would overfit the student to the
     auto-save (teach it "don't save") for a one-step gain, so --strip-noop-save is OFF by
     default and NOT recommended — it exists only for ablations. When on, it still preserves
     Ctrl+S for EXPORT trajectories (a *.csv/.pdf/.pptx… filename typed, or Ctrl+Shift+S).

  2. TAG quality gates in ``metadata.others.exclude_reason`` (the trajectory is KEPT):
       incomplete           metadata.others.terminated != true.
       dependency_install   apt/pip/conda/snap/flatpak installs.
       complex_shell        a non-teachable terminal *operation* — OPERATION-driven, not
         structure-driven: for/while loops, multi-line blocks, ``;`` / ``&&`` / single ``|``
         of simple commands are KEPT (efficient repetition); tagged are ``$()`` / ``<()`` /
         backtick / heredoc, ``python -c`` / ``bash -c``, running or authoring code scripts,
         ``sed -i`` in-place edits, dotfile/.desktop authoring, and awk state machines.
       footgun:loop         >=3 consecutive identical (name,args) actions — enabled by
         --drop-loops (a genuine stall: re-clicking a dead launcher).
       footgun:undo_storm   >= N total Ctrl+Z (default 4) — enabled by --drop-undo-storm.
       footgun:no_submit    never issued a canonical terminate/response action or
         content-only completion text — enabled by --drop-no-submit.
       reward_vision_disagree  SOFT tag — the scalar checker reward and the multi-frame
         vision verdict (``metadata.others['vision_done']``) disagree about success. The canonical
         case: a pixel/format-strict checker returns reward=0 while the agent visually
         completed the task (impress mono-font applied, GIMP 400x400). It NEVER overwrites
         ``episode_return`` (so it creates no false pass) — a consumer down-weights/inspects.
     The raw reward is NOT a tag: ``metadata.others.episode_return`` is already available for the
     consumer to threshold. The ``--drop-*`` flag names are kept for CLI compatibility but
     ordinary quality gates are not dropped; ``--drop-failed`` has no effect.
     The closed vocabulary for these tags is ``TRAJECTORY_EXCLUDE_REASONS`` (issue
     #152) — a SEPARATE namespace from the task-level env catalogs; same
     ``category(:detail)?`` format grammar. The two vocabularies are each their own
     documentation: ``TRAJECTORY_EXCLUDE_REASONS`` below in this file, and the
     task-level registry in ``lite/gym/envs/lite/osworld/exclude_reasons.py``.

  3. NORMALIZE the content-only final turn to one clean ``text`` part. A final
     assistant turn with NO ``tool_calls`` has its content replaced wholesale by
     ``structural_final_message()`` — ``[{"type": "text", "text": "Done."}]`` — no
     matter what it held before (``inline_reasoning`` only, ``action_description``
     only, both, or anything else). This is unconditional and has no opt-out.

     Why unconditional: ``action_description`` is by definition "narration
     accompanying an action" (see ``no_tool_call_final_text``), so a turn with no
     action has nothing to narrate and the field is invalid there. Any conditional
     rule ("keep the prose when there is prose") re-creates the downstream question
     *does this turn have a trainable target?* — which is exactly the empty-SFT-target
     bug. Removing the branch removes the whole family. ``lite/data/preproc`` already
     ends every ``use`` row this way; this brings the collect side into line, and the
     shape now matches what the runtime emits for a no-tool-call turn
     (``response(text=...)`` via ``make_no_tool_call_final_actions``).

     A final turn that DOES carry tool_calls is untouched — that covers the
     ``response``-submitted answer of a QA task, ``terminate``, and any ordinary
     action. ``--ensure-terminate`` remains an explicit opt-in for derived artifacts
     that intentionally train a terminate tool; never append terminate to a turn that
     also performs an ordinary action, to an incomplete/failed trajectory, or to data
     whose terminal semantics are only structural.

Usage (recommended — keeps Ctrl+S):  # <teacher>/<commit> per the dataset's
                                     # per-teacher runbook, e.g.
                                     # devs/data/lite.osworld/gpt5_5/AGENTS.md
    uv run python devs/data/lite.osworld/filter.py \
        --log-root .data/rollout/lite.osworld/<teacher>/<commit>/train.synth \
        --out      .data/rollout/lite.osworld/<teacher>/<commit>/train.synth_annotated \
        --drop-loops --drop-undo-storm

Tests: uv run pytest devs/data/lite.osworld/tests/test_lite_osworld_filter.py
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from lite.core.tools import make_tool_call
from lite.core.tools.action_space import (
    LITE_ACTION_BATCH_TOOL_NAMES,
    action_coordinate_arguments_out_of_range,
    is_lite_action_name_or_action_batch_tool_name,
)
from lite.core.tools.calls import stamp_messages_tool_call_ids, tool_call_id, tool_call_name
from lite.core.tools.extra_tools import LiteFinishToolSet
from lite.core.tools.schemas import tool_schema_name
from lite.data.staging import (
    coerce_messages,
    coerce_meta,
    prepare_output_dir,
    write_partition,
)
from lite.data.utils.messages import (
    normalize_content_only_final,
    strip_raw_response_if_message_changed,
)

# ``devs`` is not an installed package (``pyproject.toml`` ships ``lite*`` only),
# and ``python <script>.py`` puts only the SCRIPT's directory on ``sys.path`` —
# so the repo root must be added before ``devs.data.utils`` can be imported.
# Depth is per-file: this file is ``<repo>/devs/data/lite.osworld/filter.py``, so
# the root is ``parents[3]``. (The cohort directory contains a dot and can never
# be a package name; only ``devs.data.utils`` has to be importable, not this dir.)
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from devs.data.utils import (  # noqa: E402  (needs _REPO_ROOT on sys.path)
    _action_name_args,
    _args_of,
    _iter_action_items,
    _with_args,
    carry_content_without_observation_images,
    compact_row_images,
    has_invalid_action_batch,
    rebase_images_for_output,
)

# Actions the agent is told never to use (no useful effect; a screenshot is auto-taken
# after every step). The env still records them when the model emits them anyway.
DEFAULT_NOOP_ACTIONS = ("screenshot", "wait")

# Canonical actions that signal task completion (a "finish"/submit) in a
# CUA-lite trajectory. Boundary aliases such as qwen ``answer`` or legacy web
# ``done`` must be normalized by devs/migration before this filter sees the row.
SUBMIT_ACTIONS = LiteFinishToolSet.get_tool_names()

# Canonical CUA-lite terminate schema (the ONE definition, from lite.core.tools.extra_tools).
# Used only by the explicit ``--ensure-terminate`` opt-in; default annotated data
# keeps the content-only final channel, normalized to ``Done.``, and needs no
# synthetic terminate schema.
TERMINATE_SCHEMA = LiteFinishToolSet.get_tool_schema("terminate")

# A typed string that names an output file → the trajectory is a genuine Save-As / export,
# so its Ctrl+S / Ctrl+Shift+S is the real deliverable and must NOT be stripped.
_EXPORT_NAME_RE = re.compile(r"\.(csv|pdf|pptx|docx|xlsx|ods|odt|odp|png|jpe?g|txt|html?)\b", re.I)

# Any run of whitespace (newlines included) → used to flatten inline_reasoning to one line.
_WS_RUN = re.compile(r"\s+")

_TERMINAL_CONTEXT_RE = re.compile(
    r"\b(?:integrated\s+)?terminal\b|"
    r"\bshell\s+(?:window|prompt|command|output)\b",
    re.I,
)
_TERMINAL_TASK_RE = re.compile(
    r"\b(?:terminal|command[- ]line|shell)\b|"
    r"\brun (?:the )?(?:following )?command\b",
    re.I,
)
_NON_TERMINAL_CONTEXT_RE = re.compile(
    r"\b(?:return to|switch to|focus|open|in)\s+(?:the\s+)?"
    r"(?:writer|calc|impress|gimp|chrome|browser|files|file manager|vscode|vs code|"
    r"thunderbird|vlc|settings|document|spreadsheet|presentation|editor)\b",
    re.I,
)
_SHELL_INPUT_RE = re.compile(
    r"^\s*(?:(?:env\s+)?[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*(?:sudo\s+)?(?:"
    r"apt(?:-get)?|dnf|yum|pip3?|npm|npx|yarn|uv|conda|"
    r"cd|pwd|ls|find|locate|which|whereis|stat|cat|head|tail|less|more|grep|rg|"
    r"sed|awk|cut|sort|uniq|wc|tr|echo|printf|tee|touch|mkdir|rmdir|cp|mv|rm|ln|"
    r"chmod|chown|curl|wget|ssh|scp|rsync|git|tar|zip|unzip|gzip|gunzip|7z|ps|"
    r"pgrep|pkill|kill|top|htop|ffmpeg|ffprobe|convert|magick|identify|pdftk|"
    r"pdfinfo|pdftotext|pandoc|libreoffice|soffice|xdg-open|gio|gsettings|dconf|"
    r"useradd|adduser|usermod|groupadd|passwd|chpasswd|systemctl|service|"
    r"python3?|node|bash|sh|zsh|perl|ruby|php|source|export|clear|exit|make|cmake|"
    r"pytest|cargo|go|java|javac|mvn|gradle|code"
    r")\b|^\s*(?:\./|/)[^\s]+",
    re.I,
)
_DEPENDENCY_INSTALL_RE = re.compile(
    r"(?:^|(?:&&|\|\||;|\n)\s*)(?:echo\s+\S+\s*\|\s*)?(?:sudo\s+)?(?:"
    r"apt(?:-get)?\s+(?:-\S+\s+)*(?:install|build-dep)\b|"
    r"dnf\s+(?:-\S+\s+)*install\b|yum\s+(?:-\S+\s+)*install\b|"
    r"pip3?\s+install\b|python3?\s+-m\s+pip\s+install\b|"
    r"npm\s+(?:install|i)\b|yarn\s+add\b|pnpm\s+(?:add|install)\b|"
    r"conda\s+install\b|mamba\s+install\b|uv\s+(?:add|pip\s+install)\b|"
    r"snap\s+install\b|flatpak\s+install\b|"
    r"(?:add-apt-repository|apt-add-repository)\b"
    r")",
    re.I,
)
# Operation-driven complexity (see :func:`_is_complex_shell`). Control STRUCTURE
# (for/while loops, multi-line sequences, ``;`` / ``&&`` / single ``|``) is NOT
# complexity — a loop or multi-line block of simple commands is efficient
# repetition, teachable to a small model. What is tagged is a non-intuitive
# *operation*: nested/substituted execution, authoring or running code, in-place
# editing of file internals, config/dotfile authoring, or an awk state machine.
#
#: Nested/substituted execution + heredoc — dangerous regardless of structure.
#: Checked on quote-stripped text so an awk/grep pattern like ``'a||b'`` or
#: ``'NR==1 || $5=="US"'`` (operator INSIDE a quoted program) does not trip it.
_SHELL_OP_RE = re.compile(r"\$\(|(?:<|>)\(|`|<<", re.I)
#: Invoking a general-purpose interpreter in ANY form (``python -c``,
#: ``python x.py``, ``python -m venv/unittest/py_compile``, ``python --version``,
#: ``./venv/bin/python …``, ``node``/``perl``/``ruby``) or running a shell script
#: (``bash -c``, ``bash x.sh``, ``./x.sh``). Running/authoring code is not a
#: screenshot-groundable desktop op. The interpreter must be the command
#: BASENAME (``(?:\S+/)?python`` then whitespace/end) so a path that merely
#: contains "python" (``~/my-python-project/run.pdf``) is not matched.
_INTERPRETER_RE = re.compile(
    r"^(?:sudo\s+)?(?:\S+/)?(?:python3?|node|nodejs|perl|ruby|php)(?=\s|$)|"
    r"^(?:sudo\s+)?(?:bash|sh|zsh)\s+-c\b|"
    r"^(?:sudo\s+)?(?:\S+/)?(?:bash|sh|zsh)\s+\S+\.sh\b|"
    r"^\./\S+\.(?:py|sh|js|rb|pl)\b",
    re.I,
)
#: ``sed -i`` in-place edit of file internals (regex surgery on a document).
#: NOT anchored — also catches ``find ... -exec sed -i`` / ``xargs sed -i``.
_SED_INPLACE_RE = re.compile(r"\bsed\s+(?:-\S+\s+)*-i\b", re.I)
#: ``bash -c`` / ``sh -c`` executing an inline shell program — NOT anchored, so
#: it is caught even nested inside ``find ... -exec sh -c '...'`` / ``xargs sh -c``.
_SHELL_C_RE = re.compile(r"\b(?:bash|sh|zsh)\s+-c\b", re.I)
#: awk program with control flow (``next`` / ``exit``) — a state machine, not a
#: simple column/row op like ``{s+=$1} END{print s}`` or ``NR==1``.
_AWK_STATE_RE = re.compile(r"\bawk\b.*\b(?:next|exit)\b", re.I | re.S)
#: Redirecting into a CODE file — authoring a script by another name.
_CODE_FILE_WRITE_RE = re.compile(
    r">\>?\s*\S+\.(?:py|sh|bash|zsh|js|ts|rb|pl|c|cpp|h|java)\b", re.I
)
#: Redirecting into a dotfile / config / ``.desktop`` — editing hidden settings.
_DOTFILE_WRITE_RE = re.compile(
    r">\>?\s*\S*(?:\.vimrc|\.bashrc|\.zshrc|\.profile|\.desktop|\.gitconfig)\b|"
    r">\>?\s*\S*/\.(?:config|local)/|>\>?\s*~?/\.\w+\b",
    re.I,
)
_COMPLETION_RE = re.compile(
    r"\b(?:done|completed?|finished|task is complete|request is complete|terminate)\b|"
    r"\b(?:has|have) been (?:saved|applied|created|updated|configured|set|removed|added)\b|"
    r"\b(?:is|are) now (?:saved|applied|created|updated|configured|set|removed|added)\b",
    re.I,
)


def _message_text(message: dict) -> str:
    return "\n".join(
        str(part.get("text") or "")
        for part in (message.get("content") or [])
        if isinstance(part, dict)
        and part.get("type") in {"text", "inline_reasoning", "action_description"}
    )


def _has_completion_signal(message: dict) -> bool:
    return bool(_COMPLETION_RE.search(_message_text(message)))


def _terminal_inputs(messages: list[dict]) -> list[str]:
    """Return command-like strings typed into Terminal.

    Requiring a command-like payload prevents stale surface state from counting normal GUI
    text after a window switch, and excludes the word ``terminal`` typed into an app launcher.
    """
    goal = next(
        (
            _message_text(message)
            for message in messages
            if message.get("role") == "user"
        ),
        "",
    )
    terminal_surface: bool | None = bool(_TERMINAL_TASK_RE.search(goal))
    inputs: list[str] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        description = _message_text(message)
        if _NON_TERMINAL_CONTEXT_RE.search(description):
            terminal_surface = False
        if _TERMINAL_CONTEXT_RE.search(description):
            terminal_surface = True
        for name, args in _iter_action_items(message):
            if name == "key":
                keys = {str(key).lower() for key in (args.get("keys") or [])}
                if {"ctrl", "alt", "t"} <= keys or (
                    "ctrl" in keys and keys & {"`", "grave", "backtick"}
                ):
                    terminal_surface = True
                elif "alt" in keys and "tab" in keys:
                    terminal_surface = None
                continue
            if name != "type":
                continue
            text = str(args.get("text") or "")
            if (
                text.strip()
                and terminal_surface is True
                and _SHELL_INPUT_RE.search(text)
            ):
                inputs.append(text)
    return inputs


def _strip_quotes(text: str) -> str:
    """Blank out SINGLE-quoted spans only — an awk/grep program ``'a || b'`` /
    ``'$(x)'`` is literal data, not a shell operator. Double quotes are left
    intact because ``"$(...)"`` DOES run command substitution, so a real nested
    execution inside double quotes must still be detected."""
    return re.sub(r"'[^']*'", "", text)


def _atom_commands(text: str) -> list[str]:
    """Split a terminal input into atomic commands across newlines / ``;`` /
    ``&&`` / ``||`` / ``|``, stripping loop & conditional keywords so the inner
    command is exposed (``for f in *; do cp "$f" out; done`` → ``cp "$f" out``).
    Control structure is transparent; only the inner *operations* are judged."""
    atoms: list[str] = []
    for part in re.split(r"\n|;|&&|\|\||\|", text):
        part = re.sub(
            r"^\s*(?:for\b.*?\bin\b.*|while\b.*?\bdo\b|until\b.*?\bdo\b|done|then|"
            r"fi|else|elif\b.*?\bthen\b|if\b|do|case\b.*?\bin\b|esac)\s*",
            "",
            part.strip(),
            flags=re.I,
        ).strip()
        if part:
            atoms.append(part)
    return atoms


def _is_complex_shell(text: str) -> bool:
    """True if the input contains a non-simple, non-teachable *operation*.

    Structure never counts: loops, multi-line blocks, ``;`` / ``&&`` / single
    ``|`` of simple commands are kept (efficient repetition). Dropped are
    operations a small GUI model can't ground from a screenshot: nested/
    substituted execution (``$()`` / ``<()`` / backtick / heredoc), inline
    interpreters (``python -c`` / ``bash -c``), running or authoring a code
    script, ``sed -i`` in-place file surgery, dotfile/config authoring, or an
    awk state machine."""
    stripped = _strip_quotes(text)
    if _SHELL_OP_RE.search(stripped):
        return True
    if _CODE_FILE_WRITE_RE.search(stripped) or _DOTFILE_WRITE_RE.search(stripped):
        return True
    if _SED_INPLACE_RE.search(stripped) or _AWK_STATE_RE.search(text):
        return True
    if _SHELL_C_RE.search(stripped):
        return True
    for atom in _atom_commands(text):
        if _INTERPRETER_RE.match(atom):
            return True
    return False


def _trajectory_policy_violations(
    messages: list[dict],
    metadata: dict[str, Any],
) -> set[str]:
    """Mandatory quality gates shared by lite.osworld and lite.cuagym staging."""
    found: set[str] = set()
    others = metadata.get("others") or {}
    if others.get("terminated") is not True:
        found.add("incomplete")

    terminal_inputs = _terminal_inputs(messages)
    if any(_DEPENDENCY_INSTALL_RE.search(text) for text in terminal_inputs):
        found.add("dependency_install")
    if any(_is_complex_shell(text) for text in terminal_inputs):
        found.add("complex_shell")
    return found


def has_oob_coordinate(messages: list[dict]) -> bool:
    """True if any tool-call carries a ``coordinate`` outside the normalized [0, 1000].

    Model edge over-prediction (a click predicted just past the screen) or screenshot-
    resolution corruption (#56). The rollout-pipeline analog of the preproc adapters'
    unconditional ``has_oob_coordinate`` drop: the whole trajectory is hard-dropped
    before staging. ``messages`` must already be ``to_plain``-ed."""
    for m in messages:
        if not isinstance(m, dict):
            continue
        for _, args in _iter_action_items(m):
            if action_coordinate_arguments_out_of_range(args):
                return True
    return False


def has_undeclared_tool_call(messages: list[dict], metadata: dict) -> bool:
    """True if a tool call names a tool the row never declared.

    The model occasionally hallucinates a tool NAME while emitting real arguments
    (``command(pixels=-3)`` where it meant ``computer(scroll)``). Staging rejects
    the row -- ``tool_call 'x' is standalone but missing from
    metadata.extra_tool_schemas`` -- so, like an OOB coordinate, it is hard-dropped
    here rather than left to fail the publish.

    "Known" is the canonical vocabulary plus what the row declares, NOT the gate's
    per-surface schema-free set: reproducing that set here would make this filter a
    second row validator, and an earlier version that hardcoded the action-batch
    names alone dropped every row calling a bare canonical ``click``.
    """
    declared = {
        tool_schema_name(t)
        for t in (metadata.get("extra_tool_schemas") or [])
    }
    # ``_annotate_metadata`` adds TERMINATE_SCHEMA for a row that CALLS terminate
    # without declaring it, so such a row is repaired, not publish-invalid. This
    # predicate runs before that repair, so it must not count the name the
    # repair guarantees -- dropping the row here deletes data the filter fixes.
    declared.add(tool_schema_name(TERMINATE_SCHEMA))
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            name = (tc.get("function") or {}).get("name")
            if is_lite_action_name_or_action_batch_tool_name(name):
                continue
            if name not in declared:
                return True
    return False


def _keys_of(tc: dict) -> list[str]:
    """Lower-cased key list for a ``key`` action (e.g. ['ctrl','s']); [] otherwise."""
    ks = _args_of(tc).get("keys")
    if isinstance(ks, list):
        return [str(k).lower() for k in ks]
    if isinstance(ks, str):
        return [ks.lower()]
    return []


def _keys_from_args(args: dict[str, Any]) -> list[str]:
    ks = args.get("keys")
    if isinstance(ks, list):
        return [str(k).lower() for k in ks]
    if isinstance(ks, str):
        return [ks.lower()]
    return []


def _is_bare_ctrl_s_action(name: str, args: dict[str, Any]) -> bool:
    return name == "key" and set(_keys_from_args(args)) == {"ctrl", "s"}


def _is_ctrl_z(tc: dict) -> bool:
    return tool_call_name(tc) == "key" and set(_keys_of(tc)) == {"ctrl", "z"}


def _is_export_traj(messages: list[dict]) -> bool:
    """True if the trajectory genuinely exports/saves-as a named file (types a filename with
    a known extension, or uses Ctrl+Shift+S) — then bare-Ctrl+S stripping is suppressed."""
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for name, args in _iter_action_items(m):
            if name == "type":
                t = args.get("text") or ""
                if isinstance(t, str) and _EXPORT_NAME_RE.search(t):
                    return True
            keys = _keys_from_args(args)
            if name == "key" and "shift" in keys and "s" in keys and "ctrl" in keys:
                return True
    return False


def _is_empty_type_action(action_name: str, action_args: dict) -> bool:
    """A ``type`` with no text typed nothing, so it is a no-op like ``wait``.

    Two producers put one here, and only one of them is still live. A malformed
    reply that dropped the ``<parameter=text>`` opener used to yield ``text=""``;
    that is now refused at the parse boundary (``geometry.required_model_text``),
    so it survives only in rows collected before that fix. What still arrives is
    an explicitly empty parameter body -- the model's own choice, and still a
    no-op. Either way the turn was recorded as a successfully executed action
    that moved no state, and keeping it teaches the student to emit typing
    actions that do nothing.
    """
    return action_name == "type" and not (action_args.get("text") or "")


def strip_noop_actions(
    messages: list[dict], noop: frozenset[str], strip_noop_save: bool = False,
) -> tuple[list[dict], int, int]:
    """Return (cleaned_messages, n_actions_stripped, n_turns_dropped).

    ``messages`` is a ``[user, assistant, tool, assistant, tool, ...]`` sequence (already
    ``to_plain``-ed) — the observation for turn N+1 is the ``role:"tool"`` result that FOLLOWS
    assistant turn N, paired by the assistant call's ``id`` and the result's
    ``tool_call_id``. For each assistant turn: drop no-op ``tool_calls`` or no-op children
    inside canonical action-batch calls (those whose action name is in ``noop``, plus bare Ctrl+S
    when ``strip_noop_save`` and the trajectory is not an export). If a batch still has any
    non-no-op child, preserve the wrapper and its id. If no tool call remains, drop the turn
    AND the ``role:"tool"`` results carrying those ids. Dropping the results (not the preceding
    observation) is what keeps the sequence valid: the no-op changed no state, so the
    observation already in place still matches what the next turn saw.

    Rows without ``role:"tool"`` results put the observation in a ``role:"user"`` message
    BEFORE the assistant turn; for those the preceding user is popped instead, and a leading
    goal is carried forward. That branch is selected by result layout, not by a flag — an
    assistant turn whose ids have matching ``role:"tool"`` results takes the first path.
    """
    save_ok = strip_noop_save and not _is_export_traj(messages)
    out: list[dict] = []
    n_stripped = n_dropped = 0
    carried: list[dict] = []
    # Result-layout discriminator: does this trajectory carry role:"tool" results at all?
    # Keying on "the assistant call has an id" is wrong: preceding-observation rows can
    # carry stamped ids too, so that test misroutes them and leaves the observation duplicated.
    has_tool_results = any(m.get("role") == "tool" for m in messages)
    # Assistant call ids whose turn was dropped; their role:"tool" results must go too.
    orphaned_call_ids: set[str] = set()
    for m in messages:
        if m.get("role") != "assistant":
            if m.get("role") == "tool" and m.get("tool_call_id") in orphaned_call_ids:
                orphaned_call_ids.discard(m.get("tool_call_id"))
                continue
            if m.get("role") == "user" and carried:
                m = dict(m)
                m["content"] = carried + list(m.get("content") or [])
                carried = []
            out.append(m)
            continue
        tcs = m.get("tool_calls") or []

        kept: list[dict] = []
        for tc in tcs:
            name = tool_call_name(tc)
            args = _args_of(tc)
            actions = args.get("actions")
            if name in LITE_ACTION_BATCH_TOOL_NAMES and isinstance(actions, list):
                kept_actions = []
                for action in actions:
                    action_name, action_args = _action_name_args(action, name)
                    if (
                        action_name in noop
                        or _is_empty_type_action(action_name, action_args)
                        or (save_ok and _is_bare_ctrl_s_action(action_name, action_args))
                    ):
                        n_stripped += 1
                    else:
                        kept_actions.append({"action": action_name, **action_args})
                if kept_actions:
                    kept.append(_with_args(tc, {**args, "actions": kept_actions}))
                elif has_tool_results and tool_call_id(tc):
                    orphaned_call_ids.add(tool_call_id(tc) or "")
                continue
            if name in noop or (save_ok and _is_bare_ctrl_s_action(name, args)):
                n_stripped += 1
                if has_tool_results and tool_call_id(tc):
                    orphaned_call_ids.add(tool_call_id(tc) or "")
                continue
            kept.append(tc)
        if not kept and tcs:
            if has_tool_results:
                # role:"tool" result layout: mark following results for removal and
                # leave the preceding observation in place.
                orphaned_call_ids |= {
                    call_id for tc in tcs if (call_id := tool_call_id(tc))
                }
            elif out and out[-1].get("role") == "user":
                # Preceding-observation layout: the observation precedes the turn.
                prev = out.pop()
                if not any(o.get("role") == "user" for o in out):
                    carried = carry_content_without_observation_images(
                        prev.get("content") or []
                    ) + carried
            n_dropped += 1
            continue
        updated = dict(m)
        updated["tool_calls"] = kept
        out.append(strip_raw_response_if_message_changed(m, updated))
    return out, n_stripped, n_dropped


def ensure_terminate_action(messages: list[dict]) -> tuple[list[dict], bool]:
    """Opt-in: append terminate only to a clearly completed final turn with no
    ordinary action.

    GPT-5.5 native computer-use finishes a desktop task by emitting a final text turn
    with NO tool_call. The default filter policy preserves that saved LiteSample
    shape. This helper is for derived artifacts that explicitly choose to teach a
    terminate tool and therefore also add ``TERMINATE_SCHEMA`` to metadata.

    Idempotent and does NOT mutate the input. Returns ``(messages, injected)``."""
    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") != "assistant":
            continue
        tcs = out[i].get("tool_calls") or []
        names = {tool_call_name(tc) for tc in tcs}
        if names & set(SUBMIT_ACTIONS):
            return messages, False  # already finishes explicitly — leave it
        if tcs or not _has_completion_signal(out[i]):
            return messages, False
        m = dict(out[i])
        m["tool_calls"] = [make_tool_call("terminate", {"status": "success"})]
        out[i] = m
        stamp_messages_tool_call_ids(out, preserve=True)
        return out, True
    return messages, False  # no assistant turn (should not happen for a real trajectory)


def collapse_inline_reasoning(messages: list[dict]) -> tuple[list[dict], int]:
    """Flatten each assistant turn's ``inline_reasoning`` into a single-line paragraph:
    collapse newlines + whitespace runs to single spaces. GPT-5.5 emits the reasoning
    as a bold header + blank line + body; the distilled student is trained to produce a
    one-line Thought, so we normalize it at filter time. Idempotent, does NOT mutate the
    input. Returns ``(messages, n_blocks_collapsed)``."""
    out: list[dict] = []
    n = 0
    for m in messages:
        parts = m.get("content") if m.get("role") == "assistant" else None
        if not parts:
            out.append(m)
            continue
        new_parts: list[Any] = []
        changed = False
        for p in parts:
            if (isinstance(p, dict) and p.get("type") == "inline_reasoning"
                    and isinstance(p.get("text"), str)):
                flat = _WS_RUN.sub(" ", p["text"]).strip()
                if flat != p["text"]:
                    p = {**p, "text": flat}
                    changed = True
                    n += 1
            new_parts.append(p)
        if changed:
            m = strip_raw_response_if_message_changed(m, {**m, "content": new_parts})
        out.append(m)
    return out, n


def _traj_footguns(
    messages: list[dict], drop_loops: bool, drop_undo_storm: bool,
    drop_no_submit: bool, undo_storm_min: int = 4,
) -> set[str]:
    """Footgun labels for a trajectory's assistant actions — patterns that hurt the distilled
    student even when the trajectory SUCCEEDED:

      * ``loop`` — >=3 consecutive identical ``(name, args)`` actions (a genuine stall: an
        undo storm, or re-clicking a dead launcher). 2-in-a-row is legitimate (double-click).
      * ``undo_storm`` — >= ``undo_storm_min`` total Ctrl+Z (a pixel-misplaced edit recovered
        by mashing Undo — messy even when it passes).
      * ``no_submit`` — never issued a terminate/response action or content-only
        completion text (ran out of steps).

    ``messages`` must already be ``to_plain``-ed."""
    found: set[str] = set()
    prev_key = None
    run = 1
    n_undo = 0
    submitted = False
    for m in messages:
        if m.get("role") != "assistant":
            continue
        if not (m.get("tool_calls") or []) and _has_completion_signal(m):
            submitted = True
        for name, args in _iter_action_items(m):
            if name in SUBMIT_ACTIONS:
                submitted = True
            if name == "key" and set(_keys_from_args(args)) == {"ctrl", "z"}:
                n_undo += 1
            key = (name, json.dumps(args, sort_keys=True, default=str))
            run = run + 1 if key == prev_key else 1
            if drop_loops and run >= 3:
                found.add("loop")
            prev_key = key
    if drop_undo_storm and n_undo >= undo_storm_min:
        found.add("undo_storm")
    if drop_no_submit and not submitted:
        found.add("no_submit")
    return found


def _metadata(row: Any) -> dict[str, Any]:
    return dict(coerce_meta(row["metadata"]) or {})


# --- Canonical TRAJECTORY-LEVEL exclude_reason vocabulary (issue #152) --------
# A property of a single ROLLOUT (not the task). Written comma-joined to
# metadata.others.exclude_reason by ``_exclude_reasons`` below; a clean trajectory
# omits the key. This is a SEPARATE namespace from the task-level env catalogs
# (lite/gym/envs/lite/osworld/exclude_reasons.py) — same ``category(:detail)?``
# format grammar, different closed set. The dict below IS the spec for this
# namespace; the task-level one is specified by that module.
#   entry := category(":" detail)?  ;  category=[a-z][a-z0-9_]* ; detail=[a-z0-9][a-z0-9_]*
# ``_exclude_reasons`` validates emitted values before writing them.
TRAJECTORY_EXCLUDE_REASONS: dict[str, str] = {
    "incomplete": "trajectory did not terminate (metadata.others.terminated != True)",
    "dependency_install": "typed an apt/pip/conda/snap/flatpak install into a terminal",
    "complex_shell": "non-teachable terminal op ($()/heredoc/sed -i/interpreter/awk-state)",
    "footgun": "a rollout footgun; detail in {loop, undo_storm, no_submit}",
    "reward_vision_disagree": "soft-tag: scalar reward and the vision judgement disagree",
}
TRAJECTORY_DETAIL_ALLOWED: dict[str, frozenset[str]] = {
    "footgun": frozenset({"loop", "undo_storm", "no_submit"}),
}
TRAJECTORY_DETAIL_REQUIRED: frozenset[str] = frozenset(TRAJECTORY_DETAIL_ALLOWED)
_TRAJ_ENTRY_RE = re.compile(
    r"\A(?P<category>[a-z][a-z0-9_]*)(?::(?P<detail>[a-z0-9][a-z0-9_]*))?\Z"
)


def validate_trajectory_reason(reason: str) -> str:
    m = _TRAJ_ENTRY_RE.match(reason or "")
    if not m:
        raise ValueError(f"trajectory exclude_reason {reason!r} not in closed vocabulary")
    category = m.group("category")
    detail = m.group("detail")
    if category not in TRAJECTORY_EXCLUDE_REASONS:
        raise ValueError(f"trajectory exclude_reason {reason!r} not in closed vocabulary")
    if category in TRAJECTORY_DETAIL_REQUIRED and not detail:
        raise ValueError(f"trajectory exclude_reason category {category!r} requires a :detail")
    if category not in TRAJECTORY_DETAIL_REQUIRED and detail:
        raise ValueError(
            f"trajectory exclude_reason category {category!r} does not accept a :detail"
        )
    if detail and detail not in TRAJECTORY_DETAIL_ALLOWED.get(category, frozenset()):
        raise ValueError(
            f"trajectory exclude_reason detail {detail!r} is not allowed for category {category!r}"
        )
    return reason


def _reward_vision_disagree(metadata: dict[str, Any]) -> bool:
    """True when the scalar checker reward and the multi-frame vision verdict disagree.

    ``metadata.others['vision_done']`` is the vision judge's boolean verdict (did the agent
    visually complete the task?); ``metadata.others.episode_return`` is the checker's scalar
    reward. Success is ``reward > 0.5`` (mirrors ``vision_gate._scalar_success``), and a
    disagreement is ``scalar_success != vision_done``.

    This backs the SOFT tag ``reward_vision_disagree``, which NEVER overwrites
    ``episode_return`` — it only lets a consumer down-weight or inspect the row, so it
    creates no false pass. The canonical G2-8 case is a pixel/format-strict checker that
    returns reward=0 while the agent DID complete the task visually
    (``vision_done=True``) — e.g. impress mono-font applied, GIMP resized to 400x400.
    The reverse (reward>0.5 but ``vision_done=False``, a false success) is equally a
    disagreement. No vision verdict (key absent or non-bool) ⇒ no disagreement, no tag."""
    others = metadata.get("others") or {}
    vision_done = others.get("vision_done")
    if not isinstance(vision_done, bool):
        return False
    try:
        scalar_success = float(others.get("episode_return", 0.0)) > 0.5
    except (TypeError, ValueError):
        scalar_success = False
    return scalar_success != vision_done


def _exclude_reasons(
    msgs: list[dict], metadata: dict[str, Any],
    check_loops: bool, check_undo_storm: bool, check_no_submit: bool,
    undo_storm_min: int,
) -> list[str]:
    """Ordered quality-exclusion tags for a trajectory (ANNOTATE, not drop).

    Written comma-joined to ``metadata.others.exclude_reason``; a clean
    trajectory omits the key entirely. Downstream filters with
    ``not m.others.get('exclude_reason')`` — identical to the task-level idiom.

    The raw reward is deliberately NOT a tag: ``metadata.others.episode_return`` is already
    the field for the consumer to threshold. The tags cover the gates that are NOT
    otherwise a field: ``incomplete`` (also derivable from ``metadata.others.terminated``,
    but tagged so a single ``exclude_reason`` check captures it), ``dependency_install``,
    ``complex_shell``, ``footgun:*``,
    and ``reward_vision_disagree`` (soft tag — scalar reward and
    the vision verdict disagree; NEVER overwrites ``episode_return``)."""
    reasons: list[str] = []
    reasons.extend(sorted(_trajectory_policy_violations(msgs, metadata)))
    footguns = _traj_footguns(
        msgs, check_loops, check_undo_storm, check_no_submit, undo_storm_min
    )
    reasons.extend(f"footgun:{fg}" for fg in ("loop", "undo_storm", "no_submit") if fg in footguns)
    if _reward_vision_disagree(metadata):
        reasons.append("reward_vision_disagree")
    return [validate_trajectory_reason(reason) for reason in reasons]


def _has_tool_call(messages: list[dict], name: str) -> bool:
    return any(
        tool_call_name(tc) == name
        for msg in messages
        if msg.get("role") == "assistant"
        for tc in (msg.get("tool_calls") or [])
    )


def _ensure_extra_tool_schema(md: dict[str, Any], schema: dict[str, Any]) -> None:
    schemas = list(md.get("extra_tool_schemas") or [])
    name = tool_schema_name(schema)
    if name and all(tool_schema_name(s) != name for s in schemas):
        schemas.append(copy.deepcopy(schema))
    md["extra_tool_schemas"] = schemas


def _annotate_metadata(md_raw: Any, reasons: list[str], *, injected_terminate: bool = False) -> Any:
    """Return ``md_raw`` with ``others.exclude_reason`` set to the comma-joined
    ``reasons`` (or the key removed when clean), preserving the original encoding:
    a JSON-string metadata (hf.unstage rows) stays a string; a struct/dict stays
    a dict. Per-file metadata is homogeneous, so the parquet column stays uniform."""
    was_str = isinstance(md_raw, str)
    md = coerce_meta(md_raw)
    md = dict(md or {})
    others = dict(md.get("others") or {})
    if injected_terminate:
        _ensure_extra_tool_schema(md, TERMINATE_SCHEMA)
    if reasons:
        others["exclude_reason"] = ",".join(reasons)
    else:
        others.pop("exclude_reason", None)
    md["others"] = others
    return json.dumps(md) if was_str else md


_OPT_ENV_RE = re.compile(r"/opt/env/")


def _typed_opt_env(messages: list[dict]) -> bool:
    """True if any assistant ``type`` action typed a ``/opt/env/`` absolute path.

    ``/opt/env`` is the env-only tool tree (env-vendored CLIs + the env venv) the
    agent is never meant to reach; after the agent/env separation it is
    root:700-locked and unreachable. A trajectory that reaches it by absolute path
    is a genuine env-tool LEAK and is **non-reproducible** on the faithful osworld
    guest (which has no ``/opt/env``). Such trajectories are **HARD-DROPPED** —
    physically removed from the annotated output, NOT merely tagged in
    ``exclude_reason`` — because the behavior it teaches cannot transfer.

    Only ``type`` payloads are scanned (what the agent actually typed into the
    env); reasoning text is ignored so a mere mention is not a false positive."""
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        for name, args in _iter_action_items(m):
            if name != "type":
                continue
            if _OPT_ENV_RE.search(str(args.get("text") or "")):
                return True
    return False


def _process_one(task: tuple) -> tuple:
    """Pool worker: unpack, run, and return the source path with the counts."""
    src, dst, noop, strip_noop_save, out_root, dl, du, dns, usm, et, cr, dry = task
    return src, _process_file(src, dst, noop, strip_noop_save, out_root,
                              dl, du, dns, usm, et, cr, dry)


def _process_file(
    src: Path, dst: Path, noop: frozenset[str], strip_noop_save: bool,
    output_root: Path,
    drop_loops: bool, drop_undo_storm: bool, drop_no_submit: bool,
    undo_storm_min: int, ensure_terminate: bool, collapse_reasoning: bool,
    dry_run: bool = False,
) -> tuple[int, int, int, int, int, Counter[str], bool]:
    # (stripped, dropped, terminate_injected, reasoning_collapsed,
    #  content_only_finals_normalized, exclude_reason_counts, wrote)
    """ANNOTATE mode — keep EVERY trajectory, tag quality gates in
    ``metadata.others.exclude_reason`` instead of dropping. Returns
    ``(n_stripped, n_turns_dropped, n_terminate_injected, n_reasoning_collapsed,
    n_content_only_finals_normalized, reason_counts, wrote_file)``.
    ``reason_counts['_trajectories']`` = number of trajectories carrying ANY
    exclude_reason; ``reason_counts['_total']`` = total trajectories. The
    ``drop_*`` flags gate which footgun checks CONTRIBUTE A TAG; they do not
    physically drop those tagged trajectories."""
    df = pd.read_parquet(src)
    tot_strip = tot_drop = tot_term = tot_collapse = tot_final_norm = 0
    reason_counts: Counter[str] = Counter()
    new_messages: list[Any] = []
    new_metadata: list[Any] = []
    kept_idx: list[int] = []
    for pos, (_, row) in enumerate(df.iterrows()):
        msgs = coerce_messages(row["messages"])
        # HARD DROP (not a tag): these trajectories either leak an env-only tool or
        # fail the staging row-format check, so remove them from the output.
        #
        # The invented-action-name check runs FIRST and must stay there. Every
        # other pass below walks the batch through ``_iter_action_items``, whose
        # ``_action_name_args`` RAISES on a child name that is not in the tool's
        # action set -- so a single such row aborts the whole log-root before any
        # later drop can remove it, taking every clean sibling with it.
        if has_invalid_action_batch(msgs):
            reason_counts["_dropped_invalid_action"] += 1
            continue
        if _typed_opt_env(msgs):
            reason_counts["_dropped_optenv"] += 1
            continue
        if has_oob_coordinate(msgs):
            reason_counts["_dropped_oob"] += 1
            continue
        metadata = _metadata(row)
        if has_undeclared_tool_call(msgs, metadata):
            reason_counts["_dropped_undeclared_tool"] += 1
            continue
        reasons = _exclude_reasons(
            msgs, metadata, drop_loops, drop_undo_storm, drop_no_submit, undo_storm_min
        )
        cleaned, ns, nd = strip_noop_actions(msgs, noop, strip_noop_save)
        if collapse_reasoning:
            cleaned, nc = collapse_inline_reasoning(cleaned)
            tot_collapse += nc
        # Unconditional, no flag: a no-tool-call final turn becomes one clean ``text``
        # part. Runs BEFORE ensure_terminate so the opt-in terminate decorates the
        # canonical shape. Side effect worth knowing: ``ensure_terminate_action``'s
        # ``_has_completion_signal`` guard is now always satisfied on that turn (the
        # text is "Done."), so for an opted-in, completed trajectory the injection
        # gate is effectively just ``metadata.others.terminated is True``.
        cleaned, normalized = normalize_content_only_final(cleaned)
        tot_final_norm += int(normalized)
        injected = False
        # Only explicit opt-in artifacts get a synthetic terminate(success), and only
        # for completed trajectories. Injecting one by default would change a
        # content-only final into a tool target and can fake success semantics.
        if ensure_terminate and (metadata.get("others") or {}).get("terminated") is True:
            cleaned, injected = ensure_terminate_action(cleaned)
            tot_term += int(injected)
        kept_idx.append(pos)
        new_messages.append(cleaned)
        new_metadata.append(_annotate_metadata(
            row["metadata"],
            reasons,
            injected_terminate=injected or _has_tool_call(cleaned, "terminate"),
        ))
        reason_counts["_total"] += 1
        if reasons:
            reason_counts["_trajectories"] += 1
            reason_counts.update(reasons)
        tot_strip += ns
        tot_drop += nd
    # All rows in this parquet were hard-dropped → write nothing, so the sample is
    # physically absent from the annotated output.
    if not new_messages:
        return (tot_strip, tot_drop, tot_term, tot_collapse, tot_final_norm, reason_counts, False)
    out = df.iloc[kept_idx].copy().reset_index(drop=True)
    out["messages"] = new_messages
    out["metadata"] = new_metadata
    if "images" in out.columns:
        # Dropping a turn drops no picture, so a filtered row would otherwise
        # keep images nothing references and leave its indices non-contiguous.
        # Compact BEFORE rebasing: rebase copies files positionally off this
        # column, so the two must see the same list. compact_row_images owns the
        # renumbering (and asserts the result is dense); rebase renumbers nothing.
        compacted_images, compacted_messages = [], []
        for row_images, row_messages in zip(out["images"], out["messages"], strict=True):
            imgs, msgs = compact_row_images(row_images, row_messages)
            compacted_images.append(imgs)
            compacted_messages.append(msgs)
        out["images"] = compacted_images
        out["messages"] = compacted_messages
        out["images"] = rebase_images_for_output(
            out,
            source_parquet=src,
            output_parquet=dst,
            image_path_root=output_root,
            dry_run=dry_run,
        )
    if not dry_run:  # dry-run: compute every tag + counter, but write nothing
        write_partition(out.to_dict("records"), dst)
    return (tot_strip, tot_drop, tot_term, tot_collapse, tot_final_norm, reason_counts, True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--log-root", required=True, help="input rollout log-root")
    ap.add_argument(
        "--out",
        default=None,
        help="output (annotated) log-root (required unless --dry-run)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="compute + report the clean/tagged counts WITHOUT writing --out: "
        "'if I annotated now, how many would be tagged?'. --out is ignored.",
    )
    ap.add_argument(
        "--actions",
        default=None,
        help="comma-separated action names to strip (default: screenshot,wait)",
    )
    ap.add_argument(
        "--strip-noop-save",
        action="store_true",
        help="ABLATION ONLY, not recommended: strip bare Ctrl+S. Saving is a good "
        "generalizable habit, redundant only because this env auto-saves — "
        "stripping it overfits the student. Export saves are preserved even when on.",
    )
    ap.add_argument(
        "--drop-failed",
        action="store_true",
        help="NO-OP (kept for CLI compatibility): reward is not a gate; threshold "
        "metadata.others.episode_return downstream instead",
    )
    ap.add_argument(
        "--drop-loops",
        action="store_true",
        help="tag exclude_reason=footgun:loop on >=3 consecutive identical actions (stall)",
    )
    ap.add_argument(
        "--drop-undo-storm",
        action="store_true",
        help="tag exclude_reason=footgun:undo_storm on >= --undo-storm-min total Ctrl+Z",
    )
    ap.add_argument("--undo-storm-min", type=int, default=4)
    ap.add_argument(
        "--drop-no-submit",
        action="store_true",
        help="tag exclude_reason=footgun:no_submit when no canonical terminate/response "
        "action or content-only completion text was issued",
    )
    ap.add_argument(
        "--ensure-terminate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="opt-in: append an explicit terminate(success) to a final content-only "
        "assistant turn and add the terminate schema to metadata. Default preserves "
        "GPT-style text-only finals.",
    )
    ap.add_argument(
        "--collapse-reasoning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="flatten each assistant turn's inline_reasoning into a single-line "
        "paragraph (ON by default; GPT-5.5 emits a bold header + multi-line body, "
        "the student is trained to produce a one-line Thought)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing non-empty --out root. Without this, filter "
        "requires a fresh output root so stale trajectory.parquet files "
        "cannot survive a rerun.",
    )
    ap.add_argument(
        "--jobs", type=int, default=16,
        help="worker processes (default 16). Each trajectory is independent, so "
             "this scales nearly linearly; use 1 to debug a traceback.",
    )
    args = ap.parse_args()

    if not args.dry_run and not args.out:
        ap.error("--out is required unless --dry-run")
    noop = frozenset(a.strip() for a in args.actions.split(",")) if args.actions \
        else frozenset(DEFAULT_NOOP_ACTIONS)
    src_root = Path(args.log_root)
    out_root = Path(args.out) if args.out else src_root  # dry-run never writes; dst path is unused
    traj_files = sorted(src_root.rglob("trajectory.parquet"))
    if not traj_files:
        raise SystemExit(f"no trajectory.parquet under {src_root}")
    if not args.dry_run:
        try:
            prepare_output_dir(
                out_root,
                overwrite=args.overwrite,
                label="filter output root",
                protected_roots=(src_root,),
            )
        except (FileExistsError, ValueError) as e:
            raise SystemExit(str(e)) from e

    if args.dry_run:
        print("=== DRY RUN — computing clean/tagged stats, writing nothing ===")
    print(f"stripping {sorted(noop)}{' + bare Ctrl+S' if args.strip_noop_save else ''} from "
          f"{len(traj_files)} trajectories | drop_failed={args.drop_failed} "
          f"drop_loops={args.drop_loops} drop_undo_storm={args.drop_undo_storm} "
          f"drop_no_submit={args.drop_no_submit} ensure_terminate={args.ensure_terminate} "
          f"collapse_reasoning={args.collapse_reasoning}")
    ts = td = tterm = tcollapse = tfinal = nw = 0
    reason_counts: Counter[str] = Counter()
    tasks = [
        (src, out_root / src.relative_to(src_root), noop, args.strip_noop_save,
         out_root, args.drop_loops, args.drop_undo_storm, args.drop_no_submit,
         args.undo_storm_min, args.ensure_terminate, args.collapse_reasoning,
         args.dry_run)
        for src in traj_files
    ]
    if args.jobs > 1:
        with mp.Pool(args.jobs) as pool:
            results = pool.imap_unordered(_process_one, tasks, chunksize=8)
            results = list(results)
    else:
        results = [_process_one(t) for t in tasks]

    for src, (ns, nd, nterm, ncol, nfinal, npolicy, wrote) in results:
        dst = out_root / src.relative_to(src_root)
        ts += ns
        td += nd
        tterm += nterm
        tcollapse += ncol
        tfinal += nfinal
        reason_counts.update(npolicy)
        if wrote:
            nw += 1
            if not args.dry_run:
                sidecar = src.parent / "summary.json"
                if sidecar.exists():
                    shutil.copy2(sidecar, dst.parent / "summary.json")

    total = reason_counts["_total"]
    annotated = reason_counts["_trajectories"]
    dropped_optenv = reason_counts["_dropped_optenv"]
    dropped_oob = reason_counts["_dropped_oob"]
    clean = total - annotated
    verb = "WOULD be" if args.dry_run else ""
    dest = "(dry run — nothing written)" if args.dry_run else f"→ {out_root}"
    print(
        f"done (ANNOTATE mode — HARD-DROPPED {dropped_optenv} trajectories that typed "
        f"/opt/env/, {dropped_oob} with OOB coordinates, "
        f"{reason_counts['_dropped_undeclared_tool']} with an undeclared tool call and "
        f"{reason_counts['_dropped_invalid_action']} with an invalid action name; "
        f"all {total} trajectories "
        "kept after hard drops): "
        f"stripped {ts} no-op actions, dropped {td} no-op-only turns, injected terminate "
        f"into {tterm} trajectories, collapsed {tcollapse} inline_reasoning blocks, "
        f"normalized {tfinal} content-only final turns to text 'Done.' {dest} — "
        f"{clean} clean (no exclude_reason) + {annotated} {verb} tagged  "
        f"= {100*clean//max(total,1)}% clean"
    )
    # Bookkeeping keys are underscore-prefixed and real exclude_reasons never are,
    # so the convention filters them -- a hard-drop counter listed here would read as
    # a tag on rows still in the output, when those rows were removed.
    print("exclude_reason tag counts: " + ", ".join(
        f"{reason}={count}" for reason, count in sorted(reason_counts.items())
        if not reason.startswith("_")
    ))


if __name__ == "__main__":
    main()
