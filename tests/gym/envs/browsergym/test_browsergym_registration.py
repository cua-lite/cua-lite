"""BrowserGym live MiniWoB and registry tests."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("browsergym.core", reason="browsergym not installed")

from lite.core.tools import make_tool_call
from lite.gym.envs.browsergym.main import (
    _BENCHMARK_VIEWPORTS,
    BrowserGymEnv,
)

# ---------------------------------------------------------------------------
# Live MiniWoB tests (require MINIWOB_URL)
# ---------------------------------------------------------------------------

_has_miniwob = os.environ.get("MINIWOB_URL") is not None


@pytest.mark.skipif(not _has_miniwob, reason="MINIWOB_URL not set")
class TestMiniWoBLive:
    """Integration tests against a real MiniWoB server."""

    def _make_live(
        self,
        task: str = "click-dialog",
        max_steps: int = 10,
        **kwargs: Any,
    ) -> BrowserGymEnv:
        import lite.gym as gym
        return gym.make(f"browsergym.miniwob@{task}", max_steps=max_steps, **kwargs)

    @pytest.mark.asyncio
    async def test_reset_returns_screenshot(self):
        env = self._make_live()
        try:
            obs = await env.reset()
            assert obs.image
            raw = obs.image
            assert raw[:4] == b"\x89PNG"
            assert obs.text
        finally:
            await env.close()

    @pytest.mark.asyncio
    async def test_fixed_seed_makes_task_reproducible(self):
        """Headline feature, end-to-end: the registration-time fixed seed makes a
        task reproduce the same instance every reset; a different seed selects a
        different variant. ``click-button`` randomizes its target label per seed,
        so ``_instruction`` is a faithful proxy for the instance."""
        import lite.gym as gym

        async def goal(**kw):
            with patch.dict(os.environ):
                os.environ.pop("CUA_LITE_ENV_SERVER_URL", None)
                env = gym.make("browsergym.miniwob@click-button", max_steps=2, **kw)
            try:
                await env.reset()
                return env._instruction.strip()
            finally:
                await env.close()

        a = await goal()        # registered fixed seed (_DEFAULT_TASK_SEED)
        b = await goal()        # same seed → must reproduce
        c = await goal(seed=7)  # different seed → must vary
        assert a == b, f"fixed seed not reproducible: {a!r} != {b!r}"
        assert a != c, f"seed had no effect: {a!r} == {c!r}"

    @pytest.mark.asyncio
    async def test_click_action(self):
        env = self._make_live()
        try:
            await env.reset()
            r = await env.step([
                make_tool_call("click", {"coordinate": [500, 500]}),
            ])
            assert r.results[0].images
            assert not r.terminated
        finally:
            await env.close()

    @pytest.mark.asyncio
    async def test_type_action(self):
        env = self._make_live("enter-text")
        try:
            await env.reset()
            r = await env.step([
                make_tool_call("click", {"coordinate": [500, 600]}),
                make_tool_call("type", {"text": "hello"}),
            ])
            assert r.results[0].images
        finally:
            await env.close()

    @pytest.mark.asyncio
    async def test_terminate(self):
        env = self._make_live(extra_tools=["terminate"])
        try:
            await env.reset()
            r = await env.step([
                make_tool_call("terminate", {"status": "success"}),
            ])
            assert r.terminated is True
            assert r.reward is not None
        finally:
            await env.close()

    @pytest.mark.asyncio
    async def test_response_terminates(self):
        env = self._make_live(extra_tools=["response"])
        try:
            await env.reset()
            r = await env.step([
                make_tool_call("response", {"text": "OK"}),
            ])
            assert r.terminated is True
        finally:
            await env.close()

    @pytest.mark.asyncio
    async def test_truncation_at_max_steps(self):
        env = self._make_live(max_steps=2)
        try:
            await env.reset()
            r1 = await env.step([
                make_tool_call("click", {"coordinate": [500, 500]}),
            ])
            assert not r1.truncated
            r2 = await env.step([
                make_tool_call("click", {"coordinate": [500, 500]}),
            ])
            assert r2.truncated
            assert not r2.terminated
        finally:
            await env.close()


# ---------------------------------------------------------------------------
# Task registration
# ---------------------------------------------------------------------------

class TestRegistration:

    def test_miniwob_tasks_registered(self):
        try:
            import browsergym.miniwob  # noqa: F401
        except ImportError:
            pytest.skip("browsergym.miniwob not installed")

        import lite.gym as gym
        result = gym.registry.task_ids("browsergym.miniwob")
        if isinstance(result, dict):
            task_ids = [tid for ids in result.values() for tid in ids]
        else:
            task_ids = result
        assert len(task_ids) > 100, f"Expected 100+ MiniWoB tasks, got {len(task_ids)}"
        assert "click-dialog" in task_ids

    def test_browser_registry_resolves_to_browser_action_space(self):
        from lite.agents.core.action_space import ActionSpaceRegistry
        from lite.agents.core.action_space.base import LiteBrowserActionSpace

        browser_inst = ActionSpaceRegistry.get("lite@browser")
        assert isinstance(browser_inst, LiteBrowserActionSpace)
        actions = LiteBrowserActionSpace.get_declared_action_schema_names()
        assert "back" not in actions
        assert "goto" not in actions
        assert "click" in actions

    def test_stale_web_registry_key_is_not_supported(self):
        from lite.agents.core.action_space import ActionSpaceRegistry

        with pytest.raises(KeyError, match="not found"):
            ActionSpaceRegistry.get("lite@web")

    def test_benchmark_viewports_complete(self):
        # Sanity: viewport map covers all benchmarks we register.
        for bench in ("miniwob", "webarena", "visualwebarena", "timewarp"):
            assert bench in _BENCHMARK_VIEWPORTS
