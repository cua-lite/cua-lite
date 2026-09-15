"""Real Stockfish GUI regressions with a separate privileged White-side oracle.

The oracle reconstructs only public reset/move events in an independent Chess
instance. It owns another genuine WASM Worker and chooses normal GUI inputs.
No game position, engine response, evaluator state or win event is injected.
All engine searches use real time; these are not screenshot-model benchmarks.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from contextlib import asynccontextmanager

import pytest

from examples.not_a_robot import local_tasks
from examples.not_a_robot.env import NotARobotEnv
from examples.not_a_robot.tests.test_local_tasks import click, gui
from examples.not_a_robot.tests.test_local_tasks import local_env as local_env

PUBLIC_POSITION = """async () => {
  const {Chess} = await import('/vendor/spatial/chess-1.4.0.js');
  const events = syntheticTask.snapshot().events;
  const reset = events.findLastIndex(event => event.kind === 'chess_reset');
  const assistance = events[reset].assistance_queens;
  const position = new Chess();
  for (const square of ['a2','h2','b2','g2','d2','e2'].slice(0, assistance)) {
    position.remove(square); position.put({type:'q', color:'w'}, square);
  }
  const moves = events.slice(reset + 1).filter(event => event.kind === 'chess_move');
  for (const event of moves) position.move(event.san);
  return {fen:position.fen(), turn:position.turn(), assistance,
    white_moves:moves.filter(event => event.side === 'w').length,
    legal:position.moves({verbose:true}).map(move => move.from + move.to + (move.promotion || '')),
    game_over:position.isGameOver(), checkmate:position.isCheckmate(),
    winner:position.isCheckmate() ? (position.turn() === 'b' ? 'w' : 'b') : null};
}"""


ORACLE_WORKER = r"""async () => {
  const worker = new Worker('/vendor/spatial/stockfish-17-lite-single.js');
  let pending = null;
  let readyResolve, readyReject;
  const ready = new Promise((resolve, reject) => { readyResolve=resolve; readyReject=reject; });
  const startupTimer = setTimeout(
    () => readyReject(new Error('Oracle UCI startup timed out')), 20000);
  worker.onerror = event => {
    const error = new Error(event.message || 'Oracle Worker failed');
    readyReject(error);
    if (pending) { clearTimeout(pending.timer); pending.reject(error); pending=null; }
  };
  worker.onmessage = event => {
    const line = String(event.data);
    if (line === 'uciok') {
      worker.postMessage('setoption name Skill Level value 20');
      worker.postMessage('isready');
    } else if (line === 'readyok') {
      clearTimeout(startupTimer); readyResolve();
    } else if (pending) {
      const variation = line.match(/ multipv (\d+) .*? pv ([a-h][1-8][a-h][1-8][qrbn]?)/);
      if (variation) pending.variations.set(Number(variation[1]), variation[2]);
      if (line.startsWith('bestmove ')) {
        const request = pending; pending=null; clearTimeout(request.timer);
        const worstIndex = Math.max(...request.variations.keys());
        const move = request.worst ? request.variations.get(worstIndex) : line.split(' ')[1];
        if (!move || move === '(none)') request.reject(new Error('Oracle returned no legal move'));
        else request.resolve(move);
      }
    }
  };
  worker.postMessage('uci');
  try { await ready; } catch (error) {
    clearTimeout(startupTimer); worker.terminate(); throw error;
  }
  return {
    worker,
    choose(fen, worst) {
      return new Promise((resolve, reject) => {
        const timer=setTimeout(() => {
          pending=null; reject(new Error('Oracle search timed out'));
        }, 10000);
        pending={resolve,reject,timer,worst,variations:new Map()};
        worker.postMessage('setoption name MultiPV value ' + (worst ? '256' : '1'));
        worker.postMessage('position fen ' + fen);
        worker.postMessage('go movetime 300');
      });
    },
    close() {
      clearTimeout(startupTimer);
      if (pending) {
        clearTimeout(pending.timer); pending.reject(new Error('Oracle closed')); pending=null;
      }
      worker.terminate();
    }
  };
}"""


@asynccontextmanager
async def chess_oracle(page):
    async with page.expect_worker() as created:
        handle = await page.evaluate_handle(ORACLE_WORKER)
    worker = await created.value
    closed = asyncio.get_running_loop().create_future()
    worker.on("close", lambda _: closed.set_result(None) if not closed.done() else None)
    try:
        yield handle
    finally:
        await handle.evaluate("oracle => oracle.close()")
        await asyncio.wait_for(closed, timeout=10)
        await handle.dispose()


async def play_oracle_turn(env, oracle, *, worst=False):
    page = env.unwrapped._page
    position = await page.evaluate(PUBLIC_POSITION)
    assert position["turn"] == "w" and not position["game_over"]
    move = await oracle.evaluate(
        "(oracle, request) => oracle.choose(request.fen, request.worst)",
        {"fen": position["fen"], "worst": worst},
    )
    assert move in position["legal"]
    # The game's source contract auto-promotes White to queen.
    assert len(move) == 4 or move[-1] == "q"
    await click(env, page.locator(f'[data-square="{move[:2]}"]'))
    await click(env, page.locator(f'[data-square="{move[2:4]}"]'))
    await (
        page.locator(".neal-spatial-status")
        .filter(has_text=re.compile(r"White to move\.|White wins|Black wins|[Dd]raw|Engine error"))
        .wait_for(timeout=15000)
    )
    status = await page.locator(".neal-spatial-status").inner_text()
    assert "Engine error" not in status
    return await page.evaluate(PUBLIC_POSITION)


@pytest.mark.live
async def test_stockfish_real_uci_ready_and_owned_worker_cleanup(local_env):
    env = await local_env("neal_44")
    page = env.unwrapped._page
    await page.get_by_text("White to move.", exact=True).wait_for(timeout=25000)
    events = await page.evaluate("syntheticTask.snapshot().events")
    ready = [event for event in events if event["kind"] == "chess_engine_ready"]
    assert len(ready) == 1
    assert ready[0]["skill_level"] == 10 and ready[0]["move_time_ms"] == 1000
    sent = [
        event["line"]
        for event in events
        if event["kind"] == "chess_engine_uci" and event["direction"] == "sent"
    ]
    received = [
        event["line"]
        for event in events
        if event["kind"] == "chess_engine_uci" and event["direction"] == "received"
    ]
    assert "uci" in sent and "setoption name Skill Level value 10" in sent
    assert "isready" in sent and "uciok" in received and "readyok" in received
    assert len(page.workers) == 1
    worker = page.workers[0]
    closed = asyncio.get_running_loop().create_future()
    worker.on("close", lambda _: closed.set_result(None) if not closed.done() else None)
    await env.close()
    await asyncio.wait_for(closed, timeout=10)
    assert page.is_closed()


@pytest.mark.live
async def test_stockfish_refresh_drains_real_old_bestmove_and_new_search_recovers(local_env):
    env = await local_env("neal_44", max_steps=30)
    page = env.unwrapped._page
    await page.get_by_text("White to move.", exact=True).wait_for(timeout=25000)
    worker = page.workers[0]
    await click(env, page.locator('[data-square="e2"]'))
    await click(env, page.locator('[data-square="e4"]'))
    assert await page.get_by_text("Black is thinking…", exact=True).is_visible()
    await click(env, page.get_by_role("button", name="Refresh challenge"))
    await click(env, page.get_by_role("button", name="Refresh challenge"))
    await page.get_by_text("White to move.", exact=True).wait_for(timeout=10000)
    await gui(env, [{"action": "wait", "duration": 1.2}])
    assert page.workers == [worker]
    events = await page.evaluate("syntheticTask.snapshot().events")
    assert sum(event["kind"] == "chess_engine_stale_result" for event in events) == 1
    reset = max(index for index, event in enumerate(events) if event["kind"] == "chess_reset")
    assert not any(event["kind"] == "chess_move" for event in events[reset + 1 :])
    position = await page.evaluate(PUBLIC_POSITION)
    assert position["white_moves"] == 0 and position["assistance"] == 0
    assert "White pawn" in await page.locator('[data-square="e2"]').get_attribute("aria-label")
    await click(env, page.locator('[data-square="d2"]'))
    await click(env, page.locator('[data-square="d4"]'))
    await page.get_by_text("White to move.", exact=True).wait_for(timeout=15000)
    events = await page.evaluate("syntheticTask.snapshot().events")
    moves = [event for event in events[reset + 1 :] if event["kind"] == "chess_move"]
    assert [event["side"] for event in moves] == ["w", "b"]
    assert moves[0]["from"] == "d2" and moves[0]["to"] == "d4"
    assert moves[1]["opponent"] == "stockfish_17_lite_wasm"
    sent = [
        event["line"]
        for event in events
        if event["kind"] == "chess_engine_uci" and event["direction"] == "sent"
    ]
    assert sent.count("go movetime 1000") == 2 and "stop" in sent
    assert "ucinewgame" not in sent


@pytest.mark.live
@pytest.mark.parametrize("failure_kind", ["missing_js", "missing_wasm", "corrupt_wasm"])
async def test_stockfish_dependency_failure_archives_infra_without_local_fallback(
    tmp_path, monkeypatch, failure_kind
):
    assets = dict(local_tasks.ASSETS)
    wasm_path = "/vendor/spatial/stockfish-17-lite-single.wasm"
    if failure_kind == "corrupt_wasm":
        # A deliberately invalid, separately owned fixture reaches the real
        # browser WASM loader. Never modify the original vendored dependency.
        invalid_wasm = tmp_path / "unit-test-invalid.wasm"
        invalid_wasm.write_bytes(b"unit-test invalid wasm\n")
        assets[wasm_path] = (str(invalid_wasm.resolve()), "application/wasm")
    else:
        extension = "js" if failure_kind == "missing_js" else "wasm"
        del assets[f"/vendor/spatial/stockfish-17-lite-single.{extension}"]
    monkeypatch.setattr(local_tasks, "ASSETS", assets)
    env = NotARobotEnv(
        mode="local",
        local_task="neal_44",
        target_level=None,
        artifact_root=str(tmp_path),
        browser_executable=os.environ.get("NEAL_BROWSER_EXECUTABLE"),
        cursor=False,
        max_steps=20,
    )
    try:
        try:
            await env.reset()
        except RuntimeError:
            # A real dependency failure may arrive before the first observation.
            assert env.outcome == "infra_error" and env.state.status == "infra_error"
        else:
            await (
                env._page.locator(".neal-spatial-status")
                .filter(has_text="Engine error")
                .wait_for(timeout=25000)
            )
            result = await gui(env, [{"action": "screenshot"}])
            assert result.truncated and not result.terminated
            assert result.info["outcome"] == "infra_error"
        assert env.state.status == "infra_error" and env.state.mistakes == 0
        assert env.state.reason
    finally:
        await env.close()
    manifest = json.loads((env.attempt_dir / "manifest.json").read_text())
    assert manifest["outcome"] == "infra_error"
    assert manifest["recording_complete"] and manifest["data"]["cleanup_complete"]
    events = [
        json.loads(line) for line in (env.attempt_dir / "events.jsonl").read_text().splitlines()
    ]
    game_events = [event["data"] for event in events if event["type"] == "game_event"]
    assert any(event["kind"] == "chess_engine_error" for event in game_events)
    assert not any(event["kind"] in {"chess_move", "chess_engine_ready"} for event in game_events)


@pytest.mark.live
async def test_stockfish_real_black_checkmate_rejects_verify_and_earns_assistance(local_env):
    env = await local_env("neal_44", max_steps=100, max_seconds=180)
    page = env.unwrapped._page
    await page.get_by_text("White to move.", exact=True).wait_for(timeout=25000)
    async with chess_oracle(page) as oracle:
        for _ in range(30):
            # Deliberately choose the genuine oracle's lowest-ranked legal White move.
            position = await play_oracle_turn(env, oracle, worst=True)
            if position["game_over"]:
                break
        assert position["checkmate"] and position["winner"] == "b"
    assert len(page.workers) == 1
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert not result.terminated and env.unwrapped.state.status == "in_progress"
    await click(env, page.get_by_role("button", name="Refresh challenge"))
    position = await page.evaluate(PUBLIC_POSITION)
    assert position["assistance"] == 1 and position["white_moves"] == 0
    assert "White queen" in await page.locator('[data-square="a2"]').get_attribute("aria-label")


@pytest.mark.live
async def test_stockfish_full_legal_white_win_with_earned_progressive_queens(local_env):
    env = await local_env("neal_44", max_steps=500, max_seconds=500)
    page = env.unwrapped._page
    await page.get_by_text("White to move.", exact=True).wait_for(timeout=25000)
    won = False
    async with chess_oracle(page) as oracle:
        for attempt in range(4):
            for _ in range(10 if attempt < 2 else 50):
                position = await play_oracle_turn(env, oracle)
                if position["game_over"]:
                    break
            if position["winner"] == "w":
                won = True
                break
            assert position["white_moves"] >= 10 or position["winner"] == "b"
            await click(env, page.get_by_role("button", name="Refresh challenge"))
            position = await page.evaluate(PUBLIC_POSITION)
            assert position["assistance"] == attempt + 1
            for square in ["a2", "h2", "b2", "g2"][: attempt + 1]:
                label = await page.locator(f'[data-square="{square}"]').get_attribute("aria-label")
                assert "White queen" in label
        assert won, "Independent real Stockfish White oracle did not win within bounded play"
    assert len(page.workers) == 1
    result = await click(env, page.get_by_role("button", name="Verify", exact=True))
    assert result.terminated and result.reward == 1
    assert env.unwrapped.state.reason == "white_wins_legal_chess_game"
