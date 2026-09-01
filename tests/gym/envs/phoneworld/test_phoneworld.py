from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from lite.core.tools import make_tool_call
from lite.gym.envs.phoneworld import main as M
from lite.gym.registry import _splits


def _fake(task_id: str, **kwargs) -> M.PhoneWorldEnv:
    config, _ = M._TASK_CONFIGS[task_id]
    return M._make_env(config=config, task_id=task_id, use_fake=True, **kwargs)


def test_manifest_and_global_budget():
    assert len(_splits["phoneworld"]["eval"]) == 120
    assert len(_splits["phoneworld"]["train"]) == 300
    assert len(M._TASK_CONFIGS) == 420
    assert len(M._APP_LABELS) == 34
    assert all("max_steps" not in task for _, task in M._TASK_CONFIGS.values())
    assert _fake("mjd_v3_001")._max_steps == 60


def test_manifest_canonicalizes_upstream_schema_once():
    train = {task["task_id"]: task for task in M._load_tasks()["train"]}
    eval_tasks = {task["task_id"]: task for task in M._load_tasks()["eval"]}

    assert "task_041" in train
    assert "keep_017" in train and "task_132" not in train
    assert eval_tasks["cross_v3_005"]["apps"] == ["mdouban", "mweread"]
    assert eval_tasks["cross_v3_005"]["verification"]["database_path"].startswith(
        "/data/data/com.phoneuse.mweread/"
    )
    assert train["nm_005_search_artist"]["verification"] == {
        "type": "answer_contains",
        "keywords": [],
        "min_length": 3,
    }
    for task in [*train.values(), *eval_tasks.values()]:
        verification = task["verification"]
        if verification["type"] == "sqlite":
            assert any(
                f"/com.phoneuse.{app}/" in verification["database_path"]
                for app in task["apps"]
            )


def test_rollout_configs_cover_gpt_and_qwen():
    root = Path(__file__).resolve().parents[4]
    for relative in (
        "scripts/configs/gpt/default/phoneworld.yaml",
        "scripts/configs/qwen3_vl/default/phoneworld.yaml",
        "scripts/configs/qwen3_vl/compact/phoneworld.yaml",
    ):
        assert (root / relative).is_file()


def test_verifier_compilation_and_answer_semantics():
    assert (
        M._count_sql(
            {
                "table": "user_notes",
                "rules": [{"field": "text", "operator": "contains", "value": "O'Reilly"}],
            }
        )
        == "SELECT COUNT(*) FROM user_notes WHERE text LIKE '%O''Reilly%'"
    )
    assert (
        M._count_sql(
            {
                "table": "orders",
                "rules": [{"field": "status", "operator": "LIKE", "value": "%paid%"}],
            }
        )
        == "SELECT COUNT(*) FROM orders WHERE status LIKE '%paid%'"
    )
    assert M._answer_matches({"keywords": ["Foo", "BAR"], "min_length": 3}, "foo bar")
    assert not M._answer_matches({"keywords": [], "min_length": 3}, "ab")
    with pytest.raises(ValueError):
        M._count_sql({"table": "x; DROP TABLE x", "rules": []})


@pytest.mark.asyncio
async def test_response_and_container_per_episode_reset():
    env = _fake("mjd_v3_001", extra_tools=["response"])
    try:
        await env.reset()
        first = env._env
        result = await env.step(
            [
                make_tool_call("response", {"text": "售价是8999元"}, call_id="answer"),
            ]
        )
        assert result.terminated and result.reward == 1.0

        await env.reset()
        assert first._closed
        assert env._env is not first
    finally:
        await env.close()


@pytest.mark.asyncio
async def test_sqlite_multi_check_and_mixed_are_conjunctive():
    task_id = "mbilibili_v3_003"
    env = _fake(task_id)

    class RPC:
        def __init__(self, counts):
            self.counts = iter(counts)

        def post(self, _path, body=None):
            return {"count": next(self.counts)}

    verifier = env._task_static["verification"]
    env._env = SimpleNamespace(interaction_cache="目标关键词", _rpc=RPC([1, 1]))
    expected_answer = verifier["answer_contains"]["keywords"]
    env._env.interaction_cache = " ".join(expected_answer)
    assert await env._evaluate_episode(True, False, __import__("asyncio").get_event_loop()) == 1.0

    env._env = SimpleNamespace(interaction_cache=" ".join(expected_answer), _rpc=RPC([1, 0]))
    assert await env._evaluate_episode(True, False, __import__("asyncio").get_event_loop()) == 0.0
