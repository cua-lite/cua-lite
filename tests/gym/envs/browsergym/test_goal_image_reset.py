"""BrowserGym reset owns env-side goal-image metadata.

The BrowserGym agent-extension goal-image tests remain at their existing path
until the extension-owner move is allowed. This file holds only the env-owned
static guard over ``lite/gym/envs/browsergym/main.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path


class TestGoalImagesAreResetOnly:
    """``BrowserGymEnv.reset`` attaches VWA goal image refs; ``step`` does not."""

    _MAIN = (
        Path(__file__).resolve().parents[4]
        / "lite" / "gym" / "envs" / "browsergym" / "main.py"
    )

    def _method(self, name: str) -> ast.AST:
        tree = ast.parse(self._MAIN.read_text())
        cls = next(
            n for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "BrowserGymEnv"
        )
        return next(
            n for n in cls.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
        )

    @staticmethod
    def _obs_metadata_kwargs(node: ast.AST) -> list[ast.AST | None]:
        out: list[ast.AST | None] = []
        for n in ast.walk(node):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "LiteEnvObservation"
            ):
                kw = {k.arg: k.value for k in n.keywords}
                out.append(kw.get("metadata"))
        return out

    @staticmethod
    def _names(node: ast.AST) -> set[str]:
        return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}

    @staticmethod
    def _is_metadata_passthrough(node: ast.AST | None) -> bool:
        if isinstance(node, ast.Name):
            return node.id == "metadata"
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or) and len(node.values) == 2:
            left, right = node.values
            return (
                isinstance(left, ast.Name)
                and left.id == "metadata"
                and isinstance(right, ast.Constant)
                and right.value is None
            )
        return False

    @staticmethod
    def _str_consts(node: ast.AST) -> set[str]:
        return {
            n.value for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        }

    def test_reset_attaches_goal_images_metadata(self):
        reset = self._method("reset")
        assert "_extract_goal_images_b64" in self._names(reset)
        assert "goal_images_b64" in self._str_consts(reset)
        metas = self._obs_metadata_kwargs(reset)
        assert any(self._is_metadata_passthrough(m) for m in metas), (
            "reset must pass the built goal-image metadata into its observation"
        )

    def test_step_emits_metadata_none_and_no_goal_images(self):
        step = self._method("step")
        assert "_extract_goal_images_b64" not in self._names(step)
        assert "goal_images_b64" not in self._str_consts(step)
        assert self._obs_metadata_kwargs(step) == []
