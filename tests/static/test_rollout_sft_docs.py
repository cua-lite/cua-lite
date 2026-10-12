"""Static guards for SFT rollout contract prose."""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SFT_ROLLOUT = REPO / "lite" / "train" / "rollout" / "sft.py"


def _module_docstring(path: Path) -> str:
    return ast.get_docstring(
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    ) or ""


def test_sft_docstring_does_not_claim_rollout_image_index_preflight() -> None:
    docstring = _module_docstring(SFT_ROLLOUT)
    normalized = " ".join(docstring.split())

    assert "rollout validates the count before segmenting" not in normalized
    assert "producer/export code owns validation and ordering" in normalized
