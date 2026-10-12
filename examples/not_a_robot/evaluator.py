"""Classify visible game headings and explicit access-control responses.

This module does not infer task success. The environment owns completion based
on an observed transition from its verified starting level.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

_LEVEL_HEADING = re.compile(r"^[ \t]*Level[ \t]+(\d+)[ \t]*:[ \t]*([^\r\n]*)$", re.I | re.M)


@dataclass(frozen=True)
class GameState:
    """Page classification with the visible evidence used to produce it."""

    level: int | None
    label: str | None
    status: Literal["in_progress", "access_blocked", "unrecognized"]
    evidence: str


def classify_page(
    *,
    url: str,
    title: str,
    text: str,
    http_status: int | None = None,
    cf_mitigated: str | None = None,
) -> GameState:
    """Recognize one visible Level 1..48 heading, giving access gates priority.

    ``text`` must come from rendered visible text rather than HTML or hidden
    game state. Robot/human wording alone is normal synthetic gameplay.
    """
    gate_evidence: str | None = None
    normalized_title = title.strip().casefold()
    normalized_text = text.casefold()
    if http_status == 403:
        gate_evidence = "Main document returned HTTP 403."
    elif cf_mitigated and cf_mitigated.strip().casefold() == "challenge":
        gate_evidence = "Main document returned cf-mitigated: challenge."
    elif normalized_title.rstrip(".!… ") == "just a moment":
        gate_evidence = f"Access-challenge page title: {title!r}."
    elif "cloudflare" in normalized_title and (
        "attention required" in normalized_title or "access denied" in normalized_title
    ):
        gate_evidence = f"Cloudflare access-gate title: {title!r}."
    elif urlsplit(url).path.startswith("/cdn-cgi/challenge-platform/"):
        gate_evidence = "The page URL is a Cloudflare challenge-platform endpoint."
    elif "cloudflare" in normalized_text and any(
        marker in normalized_text
        for marker in (
            "checking your browser before accessing",
            "performing security verification",
            "verify you are human",
            "sorry, you have been blocked",
        )
    ):
        gate_evidence = "Visible Cloudflare access-challenge text."
    if gate_evidence is not None:
        return GameState(None, None, "access_blocked", gate_evidence)

    headings = {
        (int(match.group(1)), match.group(2).strip() or None)
        for match in _LEVEL_HEADING.finditer(text.replace("\r\n", "\n").replace("\r", "\n"))
    }
    if not headings:
        return GameState(None, None, "unrecognized", "No visible Level N: heading was found.")
    if len(headings) != 1:
        return GameState(
            None, None, "unrecognized", "Multiple distinct level headings are visible."
        )
    level, label = next(iter(headings))
    if not 1 <= level <= 48:
        return GameState(None, None, "unrecognized", f"Visible level {level} is outside 1..48.")
    evidence = f"Visible heading: Level {level}: {label or ''}".rstrip()
    return GameState(level, label, "in_progress", evidence)
