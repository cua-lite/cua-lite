"""WebGym data-collection filter — the SINGLE filtering step before staging/SFT.

Dedicated to the webgym GPT→small-VLM distillation pipeline (a standalone dev
script, not an importable module). ALL trajectory filtering lives here, so it's
explicit in one place — ``lite.data.hf.stage`` no longer applies any default
filter (its default behavior is now "stage everything"). Jobs:
  (1) drop env-agnostic no-op actions (``screenshot``/``wait``) from each turn;
  (2) ``--drop-failed`` — drop unsuccessful trajectories (``episode_return < 1.0``),
      the success filter that used to be ``stage``'s hidden default;
  (3) opt-in drop of whole footgun trajectories that hurt the student
      (F1/F2/F3/F6 + ``serp_only``/``search_flail``);
  (4) ``--drop-captcha`` — drop trajectories that hit a bot-verification / captcha
      wall (mostly redundant with ``--drop-failed`` — blocked pages usually score
      0 — but catches the ~8% of successes that detoured through a verification page).
  (5) ``--drop-unsubmitted`` — drop trajectories whose final answer was never
      submitted via a top-level ``response`` tool call with non-empty text. The
      answer must flow through ``response`` for the grader to see it; an
      unsubmitted trajectory scores 0 and teaches the student to end without
      answering. (Redundant with ``--drop-failed`` on the success set — these
      always score 0 — but an explicit guard for staging without ``--drop-failed``.)
  (6) ``--drop-illposed-task`` — drop trajectories whose TASK instruction is a
      degenerate/underspecified template with unfilled placeholders ("...a specific
      keyword ... within a specific category ... for a specific date range"). These
      are data-generation bugs, not policy failures — the model's only correct move
      is to refuse, so they teach nothing. Best applied as a PRE-ROLLOUT task filter
      to avoid wasting rollouts; here it also drops any that slipped through.

  NOTE on web search (validated on r1 data): from a datacenter egress IP the search
  ENGINES (DuckDuckGo / Google / Bing) intermittently IP-block the search REQUEST
  (the results page errors with "anomaly"/reCAPTCHA even though the homepage loads),
  while individual content sites vary. We deliberately KEEP *productive* search
  (search -> click a result -> read; 9-34% of successes) and only drop the
  degenerate ``serp_only`` (search, never click through). A blanket
  ``--drop-search-goto`` is available but NOT recommended for 8B/32B — it kills the
  learnable research skill. The durable fix for search flakiness is infra (clean
  egress IP / proxy), not filtering.

Some teachers (notably GPT computer-use) emit actions that are genuine no-ops in
*any* env and only waste a step:

  * ``screenshot`` — the gym auto-captures an observation after every step, so an
    explicit screenshot does nothing.
  * ``wait`` — advances no env state.

Training a student to imitate these teaches it to waste steps. This tool removes
them from the recorded CUA-lite ``tool_calls`` and drops any assistant turn that
becomes empty. In current ``role:"tool"`` logs, only the orphaned tool results
for that dropped turn are removed; in rows without ``role:"tool"`` results, the
paired preceding user observation is removed. The cleaned rollout log-root is
otherwise written in the same shape that ``export_sft`` consumes.

The default set is deliberately env-agnostic (only universally-useless actions).
Env-specific no-ops are NOT stripped by default — pass ``--actions`` to add them
when you know the target env treats them as no-ops. On webgym those are the
canonical desktop actions its action translator cannot execute
(``cursor_position``/``hold_key``/``key_down``/``key_up``/``mouse_down``/
``mouse_up``; see ``quality_check._NOOP``).

``--actions`` matches on the action NAME only. Argument-qualified variants are
therefore NOT expressible: a double click is canonical ``click`` with ``clicks=2``,
and ``--actions click`` would strip every click. If you ever need to strip one of
those, add an argument predicate here rather than widening ``--actions``.

Dropping a no-op-only turn is safe: a no-op doesn't change env state, so the next
observation already matches what followed it. The image that turn showed is then
referenced by nothing, so the output row is COMPACTED — the orphan is dropped and
the surviving indices are renumbered ``0..N-1`` by ``compact_row_images``
(``devs/data/utils.py``), which
rewrites the ``images`` array and every ``{"type":"image","index": N}`` part in
one step and asserts the result is dense. (It used to be left in place as
"harmless"; it is harmless to ``export_sft``, which resolves sparse indices
correctly, but it publishes a picture no message shows.) When a leading
preceding-observation user message is dropped, the latest observation image is
excluded from carried content; text, metadata, and earlier indexed reference
image parts are preserved.

Run (host):
    python devs/data/webgym/filter.py \
        --log-root .logs/rollout/gpt_webgym_collect \
        --out      .logs/rollout/gpt_webgym_collect_clean \
        --drop-failed --drop-loops --drop-serp-only \
        --drop-captcha --drop-unsubmitted --drop-illposed-task
    # then stage the --out root (stage no longer filters): see
    #   devs/data/webgym/AGENTS.md (Stage, Upload Transport, And Download) /
    #   docs/examples/rollout_to_hf.md.
    # tests: uv run pytest devs/data/webgym/tests/test_webgym_filter.py

Default stripped set is the env-agnostic ``screenshot,wait`` pair; override with
``--actions a,b,c`` when you intentionally want to include webgym-specific
unsupported actions.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from lite.core.tools.action_space import (
    LITE_ACTION_BATCH_TOOL_NAMES,
    action_coordinate_arguments_out_of_range,
)
from lite.core.tools.calls import tool_call_id, tool_call_name
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
# Depth is per-file: this file is ``<repo>/devs/data/webgym/filter.py``, so the
# root is ``parents[3]``.
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

# Env-agnostic no-ops: useless in any gym (auto-screenshot every step; wait
# advances no state). Env-specific no-ops are opt-in via --actions.
DEFAULT_NOOP_ACTIONS = ("screenshot", "wait")

# Any run of whitespace (newlines included) → used to flatten inline_reasoning to one line.
_WS_RUN = re.compile(r"\s+")


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


def _iter_top_level_calls(message: dict) -> list[tuple[str, dict[str, Any]]]:
    """Top-level canonical calls; standalone extras must stay here."""
    return [(tool_call_name(tc), _args_of(tc)) for tc in (message.get("tool_calls") or [])]


def has_oob_coordinate(messages: list[dict]) -> bool:
    """True if any tool-call carries a coordinate outside the normalized [0, 1000].

    Model edge over-prediction (a click predicted just past the screen) or screenshot-
    resolution corruption (#56). This is the rollout-pipeline analog of the preproc
    adapters' unconditional ``has_oob_coordinate`` drop: the WHOLE trajectory is
    dropped before staging (a single bad coordinate means the teacher mis-grounded).
    ``messages`` must already be ``to_plain``-ed (ndarray args → lists)."""
    for m in messages:
        if not isinstance(m, dict):
            continue
        for _, args in _iter_action_items(m):
            if action_coordinate_arguments_out_of_range(args):
                return True
    return False


def strip_noop_actions(
    messages: list[dict], noop: frozenset[str]
) -> tuple[list[dict], int, int]:
    """Return (cleaned_messages, n_actions_stripped, n_turns_dropped).

    ``messages`` is a ``[user, assistant, tool, assistant, tool, ...]`` sequence —
    the observation for turn N+1 is the ``role:"tool"`` result that FOLLOWS assistant
    turn N, paired by the assistant call's ``id`` and the result's ``tool_call_id``.
    For each assistant turn: drop no-op ``tool_calls``; if none remain, drop the
    turn AND the ``role:"tool"`` results carrying those ids.

    Rows without ``role:"tool"`` results put the observation in a ``role:"user"``
    message BEFORE the turn; for those the preceding user is popped instead and a
    leading goal is carried forward. The branch is selected by result layout — does
    the sequence contain a ``role:"tool"`` message — not by whether the assistant
    call has an id: preceding-observation rows can carry stamped ids too, so that
    test misroutes them.
    """
    # Caller passes already-plain messages (filter pipeline + tests build
    # plain dicts), and this fn never mutates input in place (it dict()-copies
    # any turn it edits), so a second to_plain() deep-copy here is pure waste.
    msgs = messages
    out: list[dict] = []
    n_stripped = n_dropped = 0
    # Non-observation content carried from a dropped LEADING user (the goal/
    # instruction, including indexed reference image parts):
    # when the trajectory opens with a no-op-only assistant turn, popping its
    # paired user obs would strand the goal (→ two consecutive user msgs, breaking
    # chat-template alternation). Instead we carry goal content forward and merge
    # it into the next surviving user observation.
    carried: list[dict] = []
    has_tool_results = any(m.get("role") == "tool" for m in msgs)
    # Assistant call ids whose turn was dropped; their role:"tool" results go too.
    orphaned_call_ids: set[str] = set()
    for m in msgs:
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
                    if action_name in noop:
                        n_stripped += 1
                    else:
                        kept_actions.append({"action": action_name, **action_args})
                if kept_actions:
                    kept.append(_with_args(tc, {**args, "actions": kept_actions}))
                elif has_tool_results and tool_call_id(tc):
                    orphaned_call_ids.add(tool_call_id(tc) or "")
                continue
            if name in noop:
                n_stripped += 1
                if has_tool_results and tool_call_id(tc):
                    orphaned_call_ids.add(tool_call_id(tc) or "")
                continue
            kept.append(tc)
        if not kept and tcs:
            # No-op-only turn → drop it and its paired preceding user obs so
            # alternation is preserved (the no-op didn't change state, so the next
            # obs already matches). If that user was the trajectory's only user so
            # far it carries the goal — don't strand it: carry its non-observation
            # parts forward to merge into the next user obs, and drop its
            # now-stale pre-no-op screenshot.
            if has_tool_results:
                # role:"tool" result layout: results follow, so keep the observation.
                orphaned_call_ids |= {
                    call_id for tc in tcs if (call_id := tool_call_id(tc))
                }
            elif out and out[-1].get("role") == "user":
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


# Search-ENGINE host markers for the F1 footgun-drop (a ``goto`` to a search engine teaches the
# distilled student the dead "address-bar / external-search" habit).
# NOTE: match search-engine *hosts* only — a site's OWN search endpoint (e.g.
# ``wyso.org/search?q=…``) is legitimate grounded navigation and must NOT be flagged.
_SEARCH_GOTO_MARKERS = (
    "google.", "bing.com", "duckduckgo.", "baidu.com", "search.yahoo.",
    "ecosia.org", "startpage.com", "brave.com/search", "search.brave.",
    "mojeek.", "yandex.",
)

# Bot-verification / captcha markers for the --drop-captcha footgun. webgym pages
# carry NO observation text (the screenshot is the only obs), so the block surfaces
# in the TEACHER's assistant text (action_description / inline_reasoning), e.g.
# "Click the human verification checkbox." Precise markers only (a bare "verify" /
# "robot" would false-positive on normal reasoning like "verify the recipe").
_CAPTCHA_MARKERS = re.compile(
    r"\b(re|h)?captcha\b|human verification|verify (you are|you'?re|that you are) (a )?human|"
    r"are you (a )?(human|robot)|i'?m not a robot|not a robot\b|cloudflare|just a moment|"
    r"attention required|unusual traffic|verifying you are( a)? human|press (and|&) hold",
    re.I,
)

# Ill-posed / underspecified TASK template: degenerate instructions whose slots were
# never filled (e.g. "...a specific keyword ... within a specific category ... for a
# specific date range ..."). The repeated "a specific <noun>" placeholder is the tell;
# require >=3 occurrences so a legitimate 1-2-slot task ("find a specific product with a
# specific feature") is NOT flagged. A data-generation bug, not a policy failure.
_ILLPOSED_RE = re.compile(r"\ba specific \w+", re.I)
_ILLPOSED_MIN = 3

# Minimal public-suffix heuristic for the F2 cross-domain drop: the *registered domain* is the last
# two labels, or the last three when the second-to-last label is a common second-level domain (so
# ``amazon.co.uk`` and ``foo.gov.au`` resolve to the brand, not the suffix). Good enough to tell a
# same-brand subdomain (apple.com → podcasts.apple.com, KEEP) from a different registered domain
# (openfoam.org → cfd.direct, DROP). We avoid a tldextract dependency on purpose — it's transitive.
_MULTI_SLDS = frozenset({"co", "com", "org", "net", "gov", "edu", "ac", "or", "ne", "go"})


def _registered_domain(host: str) -> str:
    host = (host or "").lower().split(":")[0].strip(".")
    if host.startswith("www."):
        host = host[4:]
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    if labels[-2] in _MULTI_SLDS:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _instruction_text(messages: list[dict]) -> str:
    """The task instruction = the text part of the FIRST user message (the goal +
    ``Initial website:`` line webgym injects). Empty if absent."""
    for m in messages:
        if m.get("role") != "user":
            continue
        for p in m.get("content") or []:
            if isinstance(p, dict) and p.get("type") == "text":
                return p.get("text", "") or ""
        return ""
    return ""


def _start_host(messages: list[dict]) -> str:
    """The task's start website host, parsed from the ``Initial website: <url>`` line that webgym
    injects into the first user instruction (present in 100% of trajectories; matches metadata)."""
    for m in messages:
        if m.get("role") != "user":
            continue
        for p in m.get("content") or []:
            if isinstance(p, dict) and p.get("type") == "text":
                hit = re.search(r"[Ii]nitial website:\s*(\S+)", p.get("text", "") or "")
                if hit:
                    return urlparse(hit.group(1)).netloc.replace("www.", "")
        return ""  # first user message had no Initial-website line → give up
    return ""


def _traj_footguns(
    messages: list[dict], drop_search_goto: bool, drop_loops: bool,
    drop_xdomain_goto: bool = False, drop_search_start: bool = False,
    drop_search_flail: bool = False, drop_serp_only: bool = False,
    drop_captcha: bool = False, drop_unsubmitted: bool = False,
    drop_illposed: bool = False,
) -> set[str]:
    """Footgun labels in a trajectory's assistant tool_calls — patterns that hurt the distilled
    student even when the trajectory SUCCEEDED (so the success filter misses them):

      * ``search_goto`` — a ``goto`` to a search engine (F1).
      * ``loop`` — **3+** consecutive identical ``(name, args)`` actions (F3, a genuine stall).
        (2 consecutive is NOT flagged — a double-click or two same-position scrolls in a long
        page-scan are legitimate, not footguns.)
      * ``xdomain_goto`` — a ``goto`` to a different **registered domain (eTLD+1)** than the task's
        start website (F2: memory-recalled / cross-brand URL, e.g. youtube.com → wikipedia.org).
        Same-brand subdomains and same-domain deep/API links are KEPT.
      * ``search_start`` — the task's **start website is a search engine** (F6: open-web research
        task that degenerates into multi-engine search; low-value for a grounded student).
      * ``search_flail`` — STRICT (2B): the trajectory **bounces search engines (>=2 distinct) or
        chains >=3 searches**. Too aggressive for 8B/32B research (it kills productive multi-step
        research); prefer ``serp_only`` for larger targets.
      * ``serp_only`` — the trajectory **searches but never clicks through to a real page**
        (snippet-scraping / SERP thrashing, e.g. 126677: same query on 3 engines, 0 clicks, then
        answers from snippets). Degenerate for ANY model size. Productive research (search → click
        a result → read, however many steps) is KEPT — that's the learnable complex-task skill.
      * ``captcha`` — the trajectory hit a **bot-verification / captcha wall** (the teacher's text
        shows it, e.g. "Click the human verification checkbox" / "cloudflare" / "just a moment").
        Teaches the student to click verification widgets / dead-end. Mostly redundant with the
        success filter (blocked pages usually score 0) but catches blocked *successes*.

      * ``unsubmitted`` — no top-level ``response`` tool call with non-empty text
        anywhere in the trajectory (answer never submitted to the grader).
      * ``illposed_task`` — the task instruction is a degenerate underspecified
        template (>=3 unfilled "a specific <noun>" placeholders).

    ``messages`` must already be ``to_plain``-ed (ndarray args → lists)."""
    found: set[str] = set()
    if drop_illposed and len(_ILLPOSED_RE.findall(_instruction_text(messages))) >= _ILLPOSED_MIN:
        found.add("illposed_task")
    if drop_unsubmitted:
        submitted = any(
            name == "response" and str(args.get("text", "") or "").strip()
            for m in messages if m.get("role") == "assistant"
            for name, args in _iter_top_level_calls(m)
        )
        if not submitted:
            found.add("unsubmitted")
    if drop_captcha:
        for m in messages:
            if m.get("role") != "assistant":
                continue
            for it in m.get("content") or []:
                t = it.get("text") if isinstance(it, dict) else None
                if t and _CAPTCHA_MARKERS.search(t):
                    found.add("captcha")
                    break
            if "captcha" in found:
                break
    start_host = _start_host(messages) if (drop_xdomain_goto or drop_search_start) else ""
    start_reg = _registered_domain(start_host)
    if drop_search_start and any(s in start_host for s in _SEARCH_GOTO_MARKERS):
        found.add("search_start")
    prev_key = None
    run = 1
    search_engines: set[str] = set()
    n_search = 0
    searched = False
    clicked_through = False
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for name, args in _iter_top_level_calls(m):
            if name != "goto":
                continue
            url = str(args.get("url", "") or "")
            search_hit = next((s for s in _SEARCH_GOTO_MARKERS if s in url), None)
            if search_hit:
                n_search += 1
                search_engines.add(search_hit)
                searched = True
            elif url and searched:
                clicked_through = True   # navigated to a real page after searching
            if drop_search_goto and search_hit:
                found.add("search_goto")
            if drop_xdomain_goto and start_reg:
                goto_reg = _registered_domain(urlparse(url).netloc)
                if goto_reg and goto_reg != start_reg:
                    found.add("xdomain_goto")
        for name, args in _iter_action_items(m):
            if name == "click" and searched:
                clicked_through = True   # followed a result after searching (productive research)
            key = (name, json.dumps(args, sort_keys=True, default=str))
            run = run + 1 if key == prev_key else 1
            if drop_loops and run >= 3:
                found.add("loop")
            prev_key = key
    if drop_search_flail and (len(search_engines) >= 2 or n_search >= 3):
        found.add("search_flail")
    if drop_serp_only and searched and not clicked_through:
        found.add("serp_only")
    return found


def _current_metadata(metadata: Any) -> dict:
    """Normalize metadata while keeping rollout facts under ``others``."""
    return coerce_meta(metadata)


def _episode_return(row: Any) -> float:
    """Trajectory reward from ``metadata.others.episode_return``."""
    ret = coerce_meta(row["metadata"]).get("others", {}).get("episode_return", 0.0)
    try:
        return float(ret)
    except (TypeError, ValueError):
        return 0.0


def _process_file(
    src: Path, dst: Path, noop: frozenset[str],
    output_root: Path,
    drop_search_goto: bool = False, drop_loops: bool = False,
    drop_xdomain_goto: bool = False, drop_search_start: bool = False,
    drop_search_flail: bool = False, drop_serp_only: bool = False,
    drop_captcha: bool = False, drop_failed: bool = False,
    drop_unsubmitted: bool = False, drop_illposed: bool = False,
    collapse_reasoning: bool = True,
) -> tuple[int, int, int, int, int, int, int, bool]:
    # (stripped, dropped, footgun, failed, oob, invalid_action, reasoning_collapsed,
    #  content_only_finals_normalized, wrote)
    """Returns (n_stripped, n_turns_dropped, n_footgun_trajs_dropped, n_failed_dropped,
    n_oob_dropped, n_invalid_action_dropped, n_reasoning_collapsed,
    n_content_only_finals_normalized, wrote_file).

    Trajectories that fail the success filter (``--drop-failed``: ``episode_return < 1.0``), carry
    an out-of-[0,1000] coordinate (always dropped, see ``has_oob_coordinate``), or are flagged by
    ``_traj_footguns`` are dropped entirely (not written) so they never reach ``export_sft``. If
    every trajectory in the file is dropped, nothing is written."""
    any_footgun_flag = (drop_search_goto or drop_loops or drop_xdomain_goto
                        or drop_search_start or drop_search_flail or drop_serp_only
                        or drop_captcha or drop_unsubmitted or drop_illposed)
    df = pd.read_parquet(src)
    total_stripped = total_dropped = total_footgun = total_failed = total_oob = total_collapse = 0
    total_invalid_action = 0
    total_final = 0
    keep_idx: list[int] = []
    new_messages = []
    new_metadata = []
    for i, (_, row) in enumerate(df.iterrows()):
        if drop_failed and _episode_return(row) < 1.0:
            total_failed += 1
            continue
        msgs = coerce_messages(row["messages"])
        # MUST run before every pass that walks the batch (has_oob_coordinate and
        # everything after it go through _iter_action_items, whose _action_name_args
        # RAISES on a child name outside the tool's action set). One such row would
        # otherwise abort the whole log-root and take every clean sibling with it.
        # Observed here as `mouse_wheel`/`mousewheel` where the action space says
        # `scroll`; the env rejected those steps at collection time anyway.
        if has_invalid_action_batch(msgs):
            total_invalid_action += 1
            continue
        # Always drop a trajectory with an out-of-[0,1000] coordinate (corruption /
        # edge over-prediction) — the rollout analog of preproc's has_oob_coordinate.
        if has_oob_coordinate(msgs):
            total_oob += 1
            continue
        if any_footgun_flag and _traj_footguns(
            msgs, drop_search_goto, drop_loops, drop_xdomain_goto, drop_search_start,
            drop_search_flail, drop_serp_only, drop_captcha, drop_unsubmitted,
            drop_illposed,
        ):
            total_footgun += 1
            continue
        cleaned, ns, nd = strip_noop_actions(msgs, noop)
        if collapse_reasoning:
            cleaned, nc = collapse_inline_reasoning(cleaned)
            total_collapse += nc
        # Unconditional, no flag — see messages.normalize_content_only_final. A
        # webgym answer is submitted through the ``response`` TOOL, so an answering
        # final turn has tool_calls and is untouched; only a no-tool-call final (an
        # unsubmitted/abandoned ending) is normalized to text "Done.".
        cleaned, nfin = normalize_content_only_final(cleaned)
        total_final += int(nfin)
        keep_idx.append(i)
        new_messages.append(cleaned)
        if "metadata" in df.columns:
            new_metadata.append(_current_metadata(row["metadata"]))
        total_stripped += ns
        total_dropped += nd
    if not keep_idx:
        return (total_stripped, total_dropped, total_footgun, total_failed, total_oob,
            total_invalid_action, total_collapse, total_final, False)
    out = df.iloc[keep_idx].copy()
    out["messages"] = new_messages
    if "metadata" in out.columns:
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
        )
    write_partition(out.to_dict("records"), dst)
    return (total_stripped, total_dropped, total_footgun, total_failed, total_oob,
            total_invalid_action, total_collapse, total_final, True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-root", required=True, help="input rollout log-root")
    ap.add_argument("--out", required=True, help="output (cleaned) log-root")
    ap.add_argument(
        "--actions", default=None,
        help="comma-separated action names to strip (default: screenshot,wait)",
    )
    ap.add_argument(
        "--drop-search-goto", action="store_true",
        help="F1: drop whole trajectories containing a goto to a search engine",
    )
    ap.add_argument(
        "--drop-loops", action="store_true",
        help="F3: drop whole trajectories with >=3 consecutive identical actions",
    )
    ap.add_argument(
        "--drop-xdomain-goto", action="store_true",
        help="F2: drop whole trajectories with a goto to a different registered domain (eTLD+1) "
             "than the task's start website (same-brand subdomains / same-domain deep links kept)",
    )
    ap.add_argument(
        "--drop-search-start", action="store_true",
        help="F6: drop whole trajectories whose start website is a search engine",
    )
    ap.add_argument(
        "--drop-search-flail", action="store_true",
        help="STRICT (2B target): drop trajectories that bounce search engines (>=2 distinct) or "
             "chain >=3 searches. Too aggressive for 8B/32B research — prefer --drop-serp-only.",
    )
    ap.add_argument(
        "--drop-serp-only", action="store_true",
        help="drop trajectories that search but NEVER click through to a real page (snippet-"
             "scraping / SERP thrashing) — degenerate for any model size. KEEPS productive "
             "research (search -> click a result -> read), the learnable complex-task skill.",
    )
    ap.add_argument(
        "--drop-failed", action="store_true",
        help="SUCCESS FILTER: drop trajectories with episode_return < 1.0 (the VLM-judge success "
             "predicate that used to be lite.data.hf.stage's hidden default). With this, stage "
             "stages everything and ALL filtering is explicit here.",
    )
    ap.add_argument(
        "--drop-captcha", action="store_true",
        help=(
            "drop trajectories that hit a bot-verification / captcha wall "
            "(teacher text shows it). Mostly redundant with --drop-failed "
            "(blocked pages usually score 0) but catches blocked successes."
        ),
    )
    ap.add_argument(
        "--drop-unsubmitted", action="store_true",
        help="drop trajectories whose answer was never submitted via a top-level non-empty "
             "`response` tool call. Redundant with --drop-failed (these score 0) but an explicit "
             "guard for staging without it.",
    )
    ap.add_argument(
        "--drop-illposed-task", action="store_true",
        help="drop trajectories whose TASK instruction is a degenerate underspecified template "
             "(>=3 unfilled 'a specific <noun>' placeholders) — a data-gen bug, not a policy "
             "failure. Best used as a pre-rollout task filter.",
    )
    ap.add_argument(
        "--collapse-reasoning", action=argparse.BooleanOptionalAction, default=True,
        help="flatten each assistant turn's inline_reasoning into a single-line paragraph "
             "(ON by default; GPT-5.5 emits a bold header + multi-line body, the student is "
             "trained to produce a one-line Thought)",
    )
    ap.add_argument(
        "--overwrite", action="store_true",
        help="replace an existing non-empty --out root. Without this, filter requires a fresh "
             "output root so stale trajectory.parquet files cannot survive a rerun.",
    )
    args = ap.parse_args()

    noop = frozenset(
        a.strip() for a in args.actions.split(",")
    ) if args.actions else frozenset(DEFAULT_NOOP_ACTIONS)

    src_root = Path(args.log_root)
    out_root = Path(args.out)
    traj_files = sorted(src_root.rglob("trajectory.parquet"))
    if not traj_files:
        raise SystemExit(f"no trajectory.parquet under {src_root}")
    try:
        prepare_output_dir(
            out_root,
            overwrite=args.overwrite,
            label="filter output root",
            protected_roots=(src_root,),
        )
    except (FileExistsError, ValueError) as e:
        raise SystemExit(str(e)) from e

    drop_on = (args.drop_search_goto or args.drop_loops or args.drop_xdomain_goto
               or args.drop_search_start or args.drop_search_flail or args.drop_serp_only
               or args.drop_captcha or args.drop_failed or args.drop_unsubmitted
               or args.drop_illposed_task)
    print(f"stripping {sorted(noop)} from {len(traj_files)} trajectories"
          + (f" | drop_failed={args.drop_failed} drop_search_goto={args.drop_search_goto}"
             f" drop_loops={args.drop_loops} drop_xdomain_goto={args.drop_xdomain_goto}"
             f" drop_search_start={args.drop_search_start}"
             f" drop_search_flail={args.drop_search_flail}"
             f" drop_serp_only={args.drop_serp_only} drop_captcha={args.drop_captcha}"
             f" drop_unsubmitted={args.drop_unsubmitted}"
             f" drop_illposed_task={args.drop_illposed_task}"
             if drop_on else ""))
    tot_stripped = tot_dropped = tot_footgun = tot_failed = tot_oob = tot_collapse = n_written = 0
    tot_invalid = 0
    tot_final = 0
    for src in traj_files:
        rel = src.relative_to(src_root)
        dst = out_root / rel
        ns, nd, nf, nfail, noob, ninval, ncol, nfin, wrote = _process_file(
            src, dst, noop, out_root, args.drop_search_goto, args.drop_loops,
            args.drop_xdomain_goto, args.drop_search_start, args.drop_search_flail,
            args.drop_serp_only, args.drop_captcha, args.drop_failed,
            args.drop_unsubmitted, args.drop_illposed_task, args.collapse_reasoning,
        )
        tot_stripped += ns
        tot_dropped += nd
        tot_footgun += nf
        tot_failed += nfail
        tot_oob += noob
        tot_invalid += ninval
        tot_collapse += ncol
        tot_final += nfin
        if wrote:
            n_written += 1
            # Carry summary.json (export_sft reads episode_return from the
            # parquet metadata, but keep the sample dir faithful for inspection).
            sidecar = src.parent / "summary.json"
            if sidecar.exists():
                shutil.copy2(sidecar, dst.parent / "summary.json")

    print(f"done: stripped {tot_stripped} no-op actions, dropped {tot_dropped} no-op-only turns, "
          f"dropped {tot_failed} failed + {tot_footgun} footgun + {tot_oob} OOB-coordinate "
          f"+ {tot_invalid} invalid-action "
          f"trajectories, collapsed {tot_collapse} inline_reasoning blocks, normalized "
          f"{tot_final} content-only final turns to text 'Done.' → {out_root} "
          f"({n_written}/{len(traj_files)} trajectories kept)")


if __name__ == "__main__":
    main()
