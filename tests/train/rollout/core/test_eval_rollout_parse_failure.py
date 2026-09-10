"""Eval-side keep/retry contract in ``_eval_rollout``.

The eval loop must draw the same line the train path draws: a trajectory is
*errored* when it produced no token-carrying turn, and nothing else. A model
parse failure carries tokens, so it is a scored measurement — never re-queued,
always in the eval denominator. Before this contract existed the eval loop also
excluded ``Status.FAILED``, which re-ran every parse failure up to
``max_eval_retries`` times and then dropped it from ``return_mean``.

Runs inside the Slime container: importing ``engine`` pulls the full training
stack (slime, ray, sglang_router), so it is auto-skipped on a plain dev host.
"""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest

pytest.importorskip("slime.utils", reason="slime not installed")
pytest.importorskip("slime.utils.data", reason="ray not installed")
pytest.importorskip(
    "lite.train.rollout.core.engine", reason="training stack not installed"
)

import slime.utils.data as slime_data
import slime.utils.processing_utils as slime_processing
from slime.utils.types import Sample

from lite.train.rollout.core import engine as _engine


def _turn(index: int, *, status: Sample.Status, tokens: list[int],
          episode_return: float) -> Sample:
    """One segment Sample as ``build_segment_samples`` would emit it."""
    return Sample(
        group_index=0,
        index=index,
        group_id=index,
        prompt="p",
        tokens=tokens,
        loss_mask=[1] * len(tokens),
        response_length=len(tokens),
        response="r",
        label=None,
        reward=episode_return,
        status=status,
        metadata={"others": {"episode_return": episode_return}},
    )


def _empty_turn(index: int, *, retryable: bool) -> Sample:
    """A token-less Sample as ``_empty_sample`` emits it on an env failure."""
    return _engine._empty_sample(
        Sample(group_index=0, index=index, prompt="p"),
        failure_reason="task_crash",
        retryable=retryable,
    )


@pytest.fixture
def eval_harness(monkeypatch):
    """Stub slime's dataset/tokenizer/wandb boundaries; record generate calls."""
    calls: list[int] = []
    logged: dict[str, float] = {}

    dataset_cfg = SimpleNamespace(
        name="ds", path="unused", input_key="prompt", label_key="label",
        metadata_key="metadata", tool_key=None, temperature=0.0, top_p=1.0,
        top_k=-1, max_response_len=8, n_samples_per_eval_prompt=1,
        inject_metadata=lambda md: dict(md or {}),
    )
    args = Namespace(
        eval_datasets=[dataset_cfg], hf_checkpoint="unused",
        eval_max_prompt_len=None, multimodal_keys=None,
        apply_chat_template=False, apply_chat_template_kwargs={},
        rollout_stop=None, rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False, max_eval_retries=3,
        # engine.py:1099 reads both unconditionally, outside any try.
        eval_reward_key=None, reward_key=None,
    )

    monkeypatch.setattr(slime_processing, "load_tokenizer", lambda *a, **k: None)
    monkeypatch.setattr(slime_processing, "load_processor", lambda *a, **k: None)
    monkeypatch.setattr(
        slime_data, "Dataset",
        lambda **kw: SimpleNamespace(
            samples=[Sample(group_index=0, index=0, prompt="p")]
        ),
    )
    monkeypatch.setattr(_engine, "safe_wandb_log", logged.update)

    def install(returns_per_call):
        async def fake_generate_and_rm(args, sample, **kwargs):
            calls.append(sample.index)
            i = min(len(calls) - 1, len(returns_per_call) - 1)
            return returns_per_call[i]
        monkeypatch.setattr(_engine, "generate_and_rm", fake_generate_and_rm)

    return SimpleNamespace(args=args, calls=calls, logged=logged, install=install)


async def test_parse_failure_is_scored_not_retried(eval_harness):
    """FAILED status + tokens: generated once, valid, its return is reported."""
    eval_harness.install([[_turn(0, status=Sample.Status.FAILED,
                                 tokens=[1, 2, 3], episode_return=0.0)]])

    out, _ = await _engine._eval_rollout(eval_harness.args, rollout_id=0)

    assert eval_harness.calls == [0], "a parse failure must not be re-queued"
    assert out.data["ds"]["rewards"] == [0.0]
    assert eval_harness.logged["eval/ds/n_trajs_valid"] == 1
    assert eval_harness.logged["eval/ds/n_trajs_errored"] == 0
    assert eval_harness.logged["eval/ds/n_trajs_retried"] == 0


async def test_parse_failure_that_solved_the_task_keeps_its_return(eval_harness):
    """The env's verdict decides the score, not the turn's status."""
    eval_harness.install([[_turn(0, status=Sample.Status.FAILED,
                                 tokens=[1, 2, 3], episode_return=1.0)]])

    out, _ = await _engine._eval_rollout(eval_harness.args, rollout_id=0)

    assert eval_harness.calls == [0]
    assert out.data["ds"]["rewards"] == [1.0]
    assert eval_harness.logged["eval/ds/return_mean"] == 1.0


async def test_tokenless_env_failure_is_still_retried(eval_harness):
    """The other half of the contract: no tokens at all is still errored."""
    eval_harness.install([[_empty_turn(0, retryable=True)]])

    out, _ = await _engine._eval_rollout(eval_harness.args, rollout_id=0)

    assert eval_harness.calls == [0, 0, 0, 0], "1 attempt + max_eval_retries=3"
    assert out.data["ds"]["rewards"] == [0.0]
    assert eval_harness.logged["eval/ds/n_trajs_errored"] == 1
    assert eval_harness.logged["eval/ds/n_trajs_valid"] == 0


async def test_nonretryable_env_failure_is_not_retried(eval_harness):
    """``is_retryable``'s cached verdict still gates the retry."""
    eval_harness.install([[_empty_turn(0, retryable=False)]])

    await _engine._eval_rollout(eval_harness.args, rollout_id=0)

    assert eval_harness.calls == [0]
    assert eval_harness.logged["eval/ds/n_trajs_errored"] == 1
