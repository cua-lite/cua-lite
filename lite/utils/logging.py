"""Logging setup helper for CUA-lite scripts.

Single point of truth for the ``logging.basicConfig`` call shared across
all ``scripts/*.py`` entry points — format / default level changes happen
in one place.

Override the default via the ``LOG_LEVEL`` env var (12-factor convention):

    LOG_LEVEL=DEBUG uv run python scripts/rollout.py ...

Usage:
    from lite.utils.logging import setup_logging
    setup_logging()
"""

from __future__ import annotations

import logging
import os


def setup_logging() -> None:
    """Configure root logger with CUA-lite's canonical format + level.

    Default level is ``INFO``; override via ``LOG_LEVEL`` env var.
    """
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
