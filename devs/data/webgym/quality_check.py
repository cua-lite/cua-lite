"""WebGym distillation-data quality report — the canonical quality battery.

Companion to ``filter.py`` (same dir). On the SUCCESS demos (``episode_return>=1.0``
— what actually gets distilled) it reports, per difficulty tier:

  * action histogram + webgym no-op share (screenshot/wait + the desktop verbs
    webgym's action translator cannot execute — see ``_NOOP`` below);
  * ``goto`` breakdown — search-engine / same-domain / shallow-cross-domain /
    DEEP-cross-domain(url-guess) — plus concrete URL samples;
  * footgun prevalence surfaced by the full trajectory audit:
      - ``serp_only``      : searched but never clicked through (snippet-scrape; drop)
      - ``searched``       : used a web search engine (incl. PRODUCTIVE search — kept,
                             not itself a drop condition)
      - ``search_recover`` : searched, hit a block/error, recovered & still succeeded
                             (the fault-tolerance demo — a small fraction is fine)
      - ``captcha``        : hit a bot-verification / captcha wall surfaced in
                             teacher text
      - ``back_bounce``    : >=2 ``back`` (open→back→open oscillation)
      - ``scroll_hunt``    : >=6 consecutive scrolls (viewport hunting)
      - ``loop``           : >=3 consecutive IDENTICAL actions (full args — real stall)
      - ``deepx``          : guessed deep/constructed URLs
      - ``unsubmitted``    : answer never sent via a top-level non-empty ``response``
                             call (should be ~0 in successes)
  * action_description coverage (the SFT supervision target) + turn-length stats.

Why both action-level AND footgun stats: quality regressions (url-guessing, no-op
abuse, search thrashing distinct from productive search, captcha walls,
scroll/back flailing, unsubmitted answers) surface early without opening
trajectories. NOTE: this measures ACTION-PATTERN quality only; it cannot tell
whether an answer is truly grounded in a visited page (e.g. a search-card answer)
— that needs an LLM/VLM grounding judge.

Run (host):  # <commit> = pinned cua-lite commit; see devs/data/webgym/gpt5_5/AGENTS.md
    uv run python devs/data/webgym/quality_check.py .data/rollout/webgym/gpt5_5/<commit>/d7
    uv run python devs/data/webgym/quality_check.py .data/rollout/webgym/gpt5_5/<commit>   # all tiers
"""
from __future__ import annotations

import collections
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from lite.core.tools.action_space import LiteDesktopActionSet
from lite.data.staging import coerce_messages, coerce_meta

# Co-located with filter.py — import its shared trajectory helpers directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter import (  # noqa: E402
    _CAPTCHA_MARKERS,
    _ILLPOSED_MIN,
    _ILLPOSED_RE,
    _SEARCH_GOTO_MARKERS,
    _instruction_text,
    _iter_action_items,
    _iter_top_level_calls,
    _registered_domain,
    _start_host,
    _traj_footguns,
)

# Canonical desktop GUI actions webgym's action translator actually EXECUTES
# (``lite/gym/envs/webgym/main.py::_to_webgym_command``). Everything else in the
# canonical catalog hits its ``return None  # no-op or not supported`` branch, and
# ``wait`` is translated but advances no env state — so the complement is exactly
# the "wasted step" set: cursor_position, hold_key, key_down, key_up, mouse_down,
# mouse_up, screenshot, wait. Derived, not hand-listed, so a new action in
# lite/core/tools/action_space/base.py is flagged here until webgym learns to execute it.
_WEBGYM_EXECUTED = frozenset({"click", "type", "key", "scroll", "mouse_move", "drag"})
_NOOP = LiteDesktopActionSet.get_action_names() - _WEBGYM_EXECUTED

# Broader block/error signal for search_recover (a SUPERSET of filter's captcha
# markers — also catches the search-engine "anomaly"/"unexpected error" pages and
# HTTP 403/blocked that the teacher narrates after a failed search).
_BLOCK = re.compile(
    r"unexpected error|\banomaly\b|\b403\b|forbidden|been blocked|\bblocked\b|"
    r"did(n'?t| not) load|failed to load|access denied|rate.?limit", re.I)


def _success_trajs(root: Path):
    for f in sorted(root.rglob("trajectory.parquet")):
        df = pd.read_parquet(f)
        for _, row in df.iterrows():
            o = coerce_meta(row["metadata"]).get("others", {}) or {}
            if float(o.get("episode_return", 0) or 0) >= 1.0:
                yield coerce_messages(row["messages"]), o
        del df


def _teacher_text(msgs) -> str:
    out = []
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        for c in (m.get("content") or []):
            if isinstance(c, dict) and c.get("type") in ("action_description", "inline_reasoning"):
                out.append(c.get("text", "") or "")
    return "\n".join(out)


def report_tier(root: Path) -> None:
    acts = collections.Counter()
    goto_ddg = goto_same = goto_shlwx = goto_deepx = 0
    goto_examples: list[str] = []
    serp = loop = searched = search_recover = captcha = 0
    back_bounce = scroll_hunt = unsub = illposed = 0
    turns: list[int] = []
    steps = ad_empty = parallel_steps = 0
    n = 0
    for msgs, _o in _success_trajs(root):
        n += 1
        start_reg = _registered_domain(_start_host(msgs))
        ttext = _teacher_text(msgs)
        t = 0
        did_search = submitted = False
        back_n = scroll_run = scroll_max = 0
        for m in msgs:
            if m.get("role") != "assistant":
                continue
            t += 1
            steps += 1
            action_items = _iter_action_items(m)
            if len(action_items) > 1:
                parallel_steps += 1
            if not any(c.get("type") == "action_description" and (c.get("text") or "").strip()
                       for c in (m.get("content") or []) if isinstance(c, dict)):
                ad_empty += 1
            step_is_scroll = False
            for a, _ar in action_items:
                acts[a] += 1
                if a == "scroll":
                    step_is_scroll = True
            # Standalone tools (response / browser nav) live ONLY at the top level —
            # one nested inside an action-batch call is invalid data, not a submit/nav
            # (``_iter_action_items`` rejects it loudly), so never descend here.
            for a, ar in _iter_top_level_calls(m):
                if a == "back":
                    back_n += 1
                if a == "response" and str(ar.get("text", "") or "").strip():
                    submitted = True
                if a == "goto":
                    url = str(ar.get("url", ""))
                    srch = any(s in url for s in _SEARCH_GOTO_MARKERS)
                    reg = _registered_domain(urlparse(url).netloc)
                    deep = urlparse(url).path not in ("", "/")
                    x = bool(reg and start_reg and reg != start_reg)
                    if srch:
                        goto_ddg += 1
                        did_search = True
                    elif x and deep:
                        goto_deepx += 1
                    elif x:
                        goto_shlwx += 1
                    else:
                        goto_same += 1
                    if len(goto_examples) < 12:
                        goto_examples.append(url)
            scroll_run = scroll_run + 1 if step_is_scroll else 0
            scroll_max = max(scroll_max, scroll_run)
        turns.append(t)
        ff = _traj_footguns(msgs, False, True, drop_serp_only=True)
        if "serp_only" in ff:
            serp += 1
        if "loop" in ff:
            loop += 1
        if did_search:
            searched += 1
        if did_search and _BLOCK.search(ttext):
            search_recover += 1
        if _CAPTCHA_MARKERS.search(ttext):
            captcha += 1
        if back_n >= 2:
            back_bounce += 1
        if scroll_max >= 6:
            scroll_hunt += 1
        if not submitted:
            unsub += 1
        if len(_ILLPOSED_RE.findall(_instruction_text(msgs))) >= _ILLPOSED_MIN:
            illposed += 1

    if not n:
        print(f"  {root.name}: no success demos yet")
        return
    tot_act = sum(acts.values()) or 1
    noop_n = sum(v for k, v in acts.items() if k in _NOOP)
    st = steps or 1
    pc = lambda x: f"{x}({x/n*100:.0f}%)"  # noqa: E731
    print(f"  {root.name}: {n} success demos | turns mean={sum(turns)/n:.1f} max={max(turns)}")
    print("    actions(top): " + ", ".join(f"{k}={v}" for k, v in acts.most_common(8)))
    print(f"    NO-OP actions: {noop_n} ({noop_n/tot_act*100:.1f}% of actions)  "
          f"[{', '.join(f'{k}={acts[k]}' for k in sorted(_NOOP) if acts[k]) or 'none'}]")
    print(f"    goto: search_engine={goto_ddg} same-domain={goto_same} shallowX={goto_shlwx} "
          f"DEEP-X(url-guess)={goto_deepx}")
    print(f"    footguns: serp_only={pc(serp)} loop={pc(loop)} searched={pc(searched)} "
          f"search_recover={pc(search_recover)} captcha={pc(captcha)}")
    print(f"              back_bounce={pc(back_bounce)} scroll_hunt={pc(scroll_hunt)} "
          f"deepX={pc(goto_deepx)} unsubmitted={pc(unsub)} illposed={pc(illposed)}")
    print(f"    action_desc: empty={ad_empty} ({ad_empty/st*100:.1f}% of steps)  |  "
          f"parallel-tool steps: {parallel_steps} ({parallel_steps/st*100:.1f}%, compound — OK)")
    flags = []
    if serp:
        flags.append(f"⚠ serp_only={serp}")
    if unsub:
        flags.append(f"⚠ unsubmitted={unsub}")
    if search_recover / n > 0.15:
        flags.append(f"⚠ search_recover {search_recover/n*100:.0f}%")
    if captcha / n > 0.05:
        flags.append(f"⚠ captcha {captcha/n*100:.0f}%")
    if noop_n / tot_act > 0.15:
        flags.append(f"⚠ no-op {noop_n/tot_act*100:.0f}%")
    if ad_empty / st > 0.02:
        flags.append(f"⚠ action_desc empty {ad_empty/st*100:.0f}%")
    print(f"    FLAGS: {', '.join(flags) if flags else 'clean ✓'}")
    if goto_examples:
        print("    goto URLs (sample): " + " | ".join(goto_examples[:8]))


def main() -> None:
    base = Path(sys.argv[1])
    subtiers = [d for d in sorted(base.iterdir()) if d.is_dir()
                and list(d.rglob("trajectory.parquet"))] if base.is_dir() else []
    roots = subtiers if subtiers and base.name == "r1" else [base]
    for root in roots:
        report_tier(root)


if __name__ == "__main__":
    main()
