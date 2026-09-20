"""Import/API smoke tests for the ``lite.train.rollout`` package.

Guards the engine/segmenter split + the underscore-naming standard against
future drift: catches broken import paths, mis-placed symbols, and rename
mistakes (the exact failure class that a whole-module move introduces) without
needing a GPU or the env-server.

Naming standard enforced here: a module-level name in ``core.engine`` /
``core.segmenter`` is public (no leading ``_``) IFF it is imported by another
production module (a shim, or the sibling core module); everything else is
module-private. The public set below is the frozen contract — update it only
when the shim-facing API genuinely changes.

Run (inside the Slime container — imports slime):
    pytest tests/train/rollout/test_smoke.py
"""

from __future__ import annotations

import ast
import importlib
import inspect

import pytest

# Slime-Docker-only (the module docstring's own contract): the rollout core
# imports sglang_router/slime, which exist only inside the Slime container.
# On any host venv these are named skips, not failures.
pytest.importorskip("sglang_router", reason="Slime-Docker-only (sglang_router absent)")
pytest.importorskip("slime.rollout.base_types", reason="Slime-Docker-only (slime.rollout absent)")


def test_shim_entrypoints_resolve():
    """The dotted paths slime loads from scripts must resolve to callables."""
    for mod_name, attrs in {
        "lite.train.rollout.grpo": (
            "generate",
            "generate_rollout",
            "convert_samples_to_train_data",
        ),
        "lite.train.rollout.reinforce": (
            "generate",
            "generate_rollout",
            "convert_samples_to_train_data",
        ),
        "lite.train.rollout.dagger": (
            "generate",
            "generate_rollout",
            "convert_samples_to_train_data",
        ),
        "lite.train.rollout.sft": ("generate_rollout",),
    }.items():
        mod = importlib.import_module(mod_name)
        for a in attrs:
            assert callable(getattr(mod, a)), f"{mod_name}.{a} missing/not callable"


def test_core_reexports_the_shim_facing_api():
    """``core/__init__`` must expose exactly the shim-facing API — no more, no less."""
    core = importlib.import_module("lite.train.rollout.core")
    expected = {
        "generate", "generate_rollout", "bucket_trajectories", "flatten_and_align",
        "get_episode_returns", "safe_wandb_log", "dummy_sample",
        "synthetic_dummy_sample", "build_segment_samples",
    }
    assert set(core.__all__) == expected
    for name in expected:
        assert callable(getattr(core, name)), f"core.{name} missing/not callable"


def test_public_names_live_in_the_expected_module():
    """Driver/batch API lives in engine; the pure packing entry lives in segmenter."""
    engine = importlib.import_module("lite.train.rollout.core.engine")
    segmenter = importlib.import_module("lite.train.rollout.core.segmenter")
    for name in ("generate", "generate_rollout", "bucket_trajectories", "flatten_and_align",
                 "get_episode_returns", "safe_wandb_log", "dummy_sample", "synthetic_dummy_sample"):
        assert callable(getattr(engine, name)), f"core.engine.{name} missing"
    assert callable(segmenter.build_segment_samples)


def test_whitebox_internals_still_exist():
    """Private symbols the test suite reaches into (must survive renames)."""
    engine = importlib.import_module("lite.train.rollout.core.engine")
    for name in (
        "_active_envs", "_group_seeds", "_current_rollout_id", "_last_rollout_n_errored",
        "AbortError", "_empty_sample", "_classify_failure", "_count_empty_trajs",
        "_register_env", "_unregister_env", "_abort", "_generate_rollout_async",
        "_eval_rollout", "_safe_close_env", "_regroup_trajectories", "_filter_errored_rollouts",
    ):
        assert hasattr(engine, name), f"core.engine.{name} disappeared"
    segmenter = importlib.import_module("lite.train.rollout.core.segmenter")
    for name in ("_segment_steps", "_emit_rl_sample", "_log_lazy_mode_once"):
        assert hasattr(segmenter, name), f"core.segmenter.{name} disappeared"


def test_segmenter_is_a_leaf_no_engine_import():
    """segmenter must not depend on engine (keeps the pure packing path importable alone)."""
    import sys
    sys.modules.pop("lite.train.rollout.core.segmenter", None)
    seg = importlib.import_module("lite.train.rollout.core.segmenter")
    tree = ast.parse(inspect.getsource(seg))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported = {alias.name for alias in node.names}
            assert "engine" not in imported
            assert "lite.train.rollout.core.engine" not in imported
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported = {alias.name for alias in node.names}
            assert module != "lite.train.rollout.core.engine"
            if module == "lite.train.rollout.core" or (node.level and not module):
                assert "engine" not in imported
