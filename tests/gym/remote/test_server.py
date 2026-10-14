"""Tests for ``lite.gym.remote.server`` — the env-server's auth +
bulk-cleanup + metrics endpoints.

All tests use FastAPI's ``TestClient`` against a freshly-built app —
no live env-server process, no Docker. Real envs are stubbed via a
mock ``gym.make`` that yields trivial async ``LiteBaseEnv`` instances,
so we exercise admission / cleanup / metrics paths without spinning
up containers.

Coverage matrix:
  - ``_make_bearer`` strict mode: right token / wrong token / no header
  - ``_make_bearer`` passthrough mode: client token → its hash; no
    header → "anonymous"; empty bearer → "anonymous"
  - ``DELETE /instances`` safety gate (400 vs 200) across the 8
    (session_id × env_id × force) combos
  - ``DELETE /instances`` token_hash isolation (alice can't kill bob's)
  - ``GET /metrics`` format + auth + aggregation across tenants
  - ``GET /envs``, ``GET /envs/{env_id}``, ``GET /envs/{env_id}/tasks``
    catalog endpoints + ``?expand=metadata`` map shape

Run::

    uv run pytest tests/gym/remote/test_server.py -v
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from gym.remote.conftest import (
    bearer_header as _h,
)
from gym.remote.conftest import (
    make_test_admission as _make_test_admission,
)

from lite.core.messages.final import (
    CONTENT_ONLY_FINAL_REASON,
    LOOP_DETECT_TERMINATE_REASON,
    make_no_tool_call_final_actions,
)
from lite.core.metadata import LiteCUAMetadata
from lite.core.tools import make_tool_call
from lite.core.tools.calls import (
    RUNTIME_INTERNAL_STOP_REASON_KEY,
    RUNTIME_RESULT_CALL_ID_KEY,
)
from lite.gym.base import LiteBaseEnv
from lite.gym.errors import EnvUnavailable
from lite.gym.remote.frame import FRAME_MAGIC, FRAME_VERSION, decode_step_result
from lite.gym.remote.scope import ServerScope
from lite.gym.remote.server import (
    State,
    _make_bearer,
    make_app,
)
from lite.gym.types import LiteEnvObservation, LiteEnvStepResult
from lite.gym.utils.feedback.ingress import (
    make_internal_terminate_action,
    prepare_env_tool_calls,
)
from lite.gym.utils.feedback.results import (
    build_tool_results_from_decisions,
    ordered_tool_call_ids,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dedicated_families(reg) -> dict:
    """Map every currently-registered (monkeypatched) fake-services env_id to DEDICATED.

    The reaper sweeps are gated by the declared family, and a
    services-registered env with no family trips the F3 validator. The dispatch tests below
    register fake ``EnvServices`` via ``monkeypatch.setattr(reg, "_services", {...})`` without
    a family, so they also monkeypatch ``reg._families`` to dedicated families.
    — DEDICATED ⇒ PER_INSTANCE (steady reconcile) + has_lifecycle (boot recover + shutdown),
    which is exactly the dispatch these tests assert."""
    from lite.gym.services import BackendFamily

    return dict.fromkeys(reg._services, BackendFamily.DEDICATED)


def _bare_state() -> State:
    """Minimal State for unit tests that only exercise auth / bearer
    plumbing — no admission, no env table mutations. Each ``_make_bearer``
    test needs one because the dependency now records into
    ``state.token_registry``; the small object is throwaway."""
    return State(
        admission=_make_test_admission(max_live_envs=10),
        idle_ttl_sec=60.0,
    )


class _NoopEnv(LiteBaseEnv):
    """Minimal env stub — no docker, no real work."""

    def _runtime_metadata(self) -> LiteCUAMetadata:
        return LiteCUAMetadata(dims=("browser", "use"))

    async def reset(self) -> LiteEnvObservation:
        return LiteEnvObservation(image=None, text="ok")

    async def step(self, actions: list) -> LiteEnvStepResult:
        return LiteEnvStepResult()

    async def close(self) -> None:
        pass


class _RecordingEnv(_NoopEnv):
    def __init__(self) -> None:
        self.seen_step_actions: list[list[Any]] = []

    async def step(self, actions: list) -> LiteEnvStepResult:
        self.seen_step_actions.append(actions)
        return LiteEnvStepResult()


class _IngressFeedbackEnv(_RecordingEnv):
    async def step(self, actions: list) -> LiteEnvStepResult:
        self.seen_step_actions.append(actions)
        ordered_call_ids = ordered_tool_call_ids(actions)
        _prepared, feedback = prepare_env_tool_calls(actions, self.metadata)
        return build_tool_results_from_decisions(
            LiteEnvStepResult(),
            ordered_call_ids=ordered_call_ids,
            continue_call_ids=ordered_call_ids,
            feedback=feedback,
        )


def _make_client(
    *,
    token: str | None = None,
    max_live_envs: int = 100,
    allowed_env_ids: set[str] | None = None,
):
    """Yield a TestClient + stubbed ``gym.make`` so POST /envs works
    without containers. Returns (client, patcher) so the caller can
    stop the patcher in teardown.
    """
    state = State(
        admission=_make_test_admission(max_live_envs=max_live_envs),
        idle_ttl_sec=3600.0,
        allowed_env_ids=allowed_env_ids,
    )
    app = make_app(state, token=token)
    # Bypass real env construction. Each create returns a fresh _NoopEnv.
    patcher = patch(
        "lite.gym.remote.server.gym.make",
        lambda env_key, **kw: _NoopEnv(),
    )
    patcher.start()
    return TestClient(app), patcher


def _make_recording_client(*, token: str = "zzh-dev"):
    created: list[_RecordingEnv] = []

    def _make_env(env_key: str, **kwargs: Any) -> _RecordingEnv:
        env = _RecordingEnv()
        created.append(env)
        return env

    state = State(
        admission=_make_test_admission(max_live_envs=10),
        idle_ttl_sec=3600.0,
    )
    app = make_app(state, token=token)
    patcher = patch("lite.gym.remote.server.gym.make", _make_env)
    patcher.start()
    return TestClient(app), patcher, created


def _make_ingress_feedback_client(*, token: str = "zzh-dev"):
    created: list[_IngressFeedbackEnv] = []

    def _make_env(env_key: str, **kwargs: Any) -> _IngressFeedbackEnv:
        env = _IngressFeedbackEnv()
        created.append(env)
        return env

    state = State(
        admission=_make_test_admission(max_live_envs=10),
        idle_ttl_sec=3600.0,
    )
    app = make_app(state, token=token)
    patcher = patch("lite.gym.remote.server.gym.make", _make_env)
    patcher.start()
    return TestClient(app), patcher, created


# ---------------------------------------------------------------------------
# _make_bearer
# ---------------------------------------------------------------------------


class TestMakeBearer:
    def test_strict_right_token_returns_hash(self):
        bearer = _make_bearer("hunter2", _bare_state())
        result = asyncio.run(bearer(authorization="Bearer hunter2"))
        assert result == hashlib.sha256(b"hunter2").hexdigest()[:6]

    def test_strict_wrong_token_raises_401(self):
        from fastapi import HTTPException

        bearer = _make_bearer("hunter2", _bare_state())
        with pytest.raises(HTTPException) as exc:
            asyncio.run(bearer(authorization="Bearer wrong"))
        assert exc.value.status_code == 401

    def test_strict_no_header_raises_401(self):
        from fastapi import HTTPException

        bearer = _make_bearer("hunter2", _bare_state())
        with pytest.raises(HTTPException) as exc:
            asyncio.run(bearer(authorization=None))
        assert exc.value.status_code == 401

    def test_passthrough_client_token_returns_its_hash(self):
        bearer = _make_bearer(None, _bare_state())
        result = asyncio.run(bearer(authorization="Bearer alice"))
        assert result == hashlib.sha256(b"alice").hexdigest()[:6]

    def test_passthrough_no_header_is_anonymous(self):
        bearer = _make_bearer(None, _bare_state())
        result = asyncio.run(bearer(authorization=None))
        assert result == "anonymous"

    def test_passthrough_empty_bearer_is_anonymous(self):
        bearer = _make_bearer(None, _bare_state())
        result = asyncio.run(bearer(authorization="Bearer "))
        assert result == "anonymous"

    def test_passthrough_malformed_header_is_anonymous(self):
        bearer = _make_bearer(None, _bare_state())
        result = asyncio.run(bearer(authorization="Basic xyz"))
        assert result == "anonymous"

    def test_passthrough_different_tokens_yield_different_hashes(self):
        bearer = _make_bearer(None, _bare_state())
        h_alice = asyncio.run(bearer(authorization="Bearer alice"))
        h_bob = asyncio.run(bearer(authorization="Bearer bob"))
        assert h_alice != h_bob
        assert h_alice == hashlib.sha256(b"alice").hexdigest()[:6]
        assert h_bob == hashlib.sha256(b"bob").hexdigest()[:6]


class TestHostStatusAuth:
    """section B6.4: ``/host_status`` is bearer-gated like ``/metrics`` — a strict-token
    server must 401 requests with no/wrong bearer and 200 the right one."""

    def test_host_status_bearer_gate(self):
        client, patcher = _make_client(token="strict_T")
        try:
            assert client.get("/host_status").status_code == 401
            assert client.get("/host_status", headers=_h("wrong")).status_code == 401
            r = client.get("/host_status", headers=_h("strict_T"))
            assert r.status_code == 200, r.text
            body = r.json()
            assert "cpu" in body and "memory" in body
            assert body["loop"] == {
                "stall_max_seconds": 0.0,
                "stall_max_at": None,
                "stall_max_60s": 0.0,
            }
            assert body["wire"] == {
                "frame_magic": FRAME_MAGIC,
                "frame_version": FRAME_VERSION,
            }
        finally:
            patcher.stop()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class TestState:
    # deleted cost_of / in_use_cost / pending_create_cost — those
    # tests are obsolete with the cost+budget model. Admission accounting
    # is now in AdmissionGate.in_flight, tested separately.

    def test_record_step_buckets_fast_calls(self):
        s = State(admission=_make_test_admission(max_live_envs=100), idle_ttl_sec=3600.0)
        s.record_step("x", duration_s=0.05, ok=True)  # <0.1s
        s.record_step("x", duration_s=0.3, ok=True)  # <0.5s
        s.record_step("x", duration_s=2.0, ok=True)  # <2.5s
        s.record_step("x", duration_s=100.0, ok=True)  # +Inf
        assert s.step_total["x"] == 4
        assert s.step_5xx_total.get("x", 0) == 0
        # bucket[0]=0.1s, bucket[1]=0.5s, bucket[3]=2.5s, last=+Inf
        b = s.step_duration_buckets["x"]
        assert b[0] == 1 and b[1] == 1 and b[3] == 1 and b[-1] == 1
        assert s.step_duration_sum["x"] == pytest.approx(0.05 + 0.3 + 2.0 + 100.0)

    def test_record_step_failures_counted_separately(self):
        s = State(admission=_make_test_admission(max_live_envs=100), idle_ttl_sec=3600.0)
        s.record_step("x", duration_s=0.1, ok=True)
        s.record_step("x", duration_s=0.1, ok=False)
        assert s.step_total["x"] == 2
        assert s.step_5xx_total["x"] == 1

    def test_record_reset_separate_from_step(self):
        s = State(admission=_make_test_admission(max_live_envs=100), idle_ttl_sec=3600.0)
        s.record_reset("x", 1.5, ok=True)
        s.record_reset("x", 12.0, ok=False)
        assert s.reset_total["x"] == 2
        assert s.reset_5xx_total["x"] == 1
        # Reset doesn't touch step counters.
        assert "x" not in s.step_total
        # Duration is bucketed into the reset histogram (previously reset
        # only had total/5xx counters, no latency observability).
        # Buckets are (0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0):
        assert s.reset_duration_buckets["x"][2] == 1  # 1.5s falls in (1.0, 5.0]
        assert s.reset_duration_buckets["x"][4] == 1  # 12.0s falls in (10.0, 30.0]
        assert abs(s.reset_duration_sum["x"] - 13.5) < 1e-9


# ---------------------------------------------------------------------------
# DELETE /instances — bulk-close safety gate matrix
# ---------------------------------------------------------------------------


def _create_env(
    client: TestClient,
    token: str,
    env_id: str,
    session_id: str,
    env_kwargs: dict[str, Any] | None = None,
) -> str:
    """Create one env via POST /instances, return its id."""
    r = client.post(
        "/instances",
        headers=_h(token),
        json={
            "env_key": f"{env_id}@t",
            "env_kwargs": env_kwargs or {},
            "session_id": session_id,
        },
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


class TestStepBeforeReset:
    """``/step`` before ``/reset`` is a client state-machine error → 409, not a
    500 from the env touching un-booted state. Regression guard for the
    server-robustness fix (found via test_live_stress_boot_path)."""

    def test_step_before_reset_returns_409_then_ok_after_reset(self):
        client, patcher = _make_client(token="zzh-dev")
        try:
            iid = _create_env(client, "zzh-dev", "lite.demo", "s1")
            # step before reset → clean 409 (not 500)
            r = client.post(f"/instances/{iid}/step", headers=_h("zzh-dev"), json={"actions": []})
            assert r.status_code == 409, r.text
            # section B7: the 409 carries a typed body the client reconstructs as
            # ProtocolMisuse (terminal — no pointless unknown→retry).
            assert r.json().get("error_type") == "ProtocolMisuse", r.text
            # reset, then step is allowed
            assert client.post(f"/instances/{iid}/reset", headers=_h("zzh-dev")).status_code == 200
            assert (
                client.post(
                    f"/instances/{iid}/step", headers=_h("zzh-dev"), json={"actions": []}
                ).status_code
                == 200
            )
        finally:
            patcher.stop()


class TestStepBodyValidation:
    """Remote /step request-body validation is a model-output boundary.

    Pairable argument/schema mistakes can still be env-owned feedback, but
    unpairable request envelopes stop at the remote boundary as a typed,
    terminal model-output error instead of env crashes or generic HTTP errors.
    """

    def _create_and_reset(self, client: TestClient) -> str:
        iid = _create_env(client, "zzh-dev", "lite.demo", "s1")
        assert (
            client.post(
                f"/instances/{iid}/reset",
                headers=_h("zzh-dev"),
            ).status_code
            == 200
        )
        return iid

    def _post_malformed_actions(self, actions: list[Any]):
        client, patcher, created = _make_recording_client()
        try:
            iid = self._create_and_reset(client)
            response = client.post(
                f"/instances/{iid}/step",
                headers=_h("zzh-dev"),
                json={"actions": actions},
            )
            return response, created[0]
        finally:
            patcher.stop()

    def test_non_object_step_action_returns_typed_model_output_error(self):
        response, env = self._post_malformed_actions(["not-a-tool-call"])

        assert response.status_code == 422, response.text
        payload = response.json()
        assert payload["error_type"] == "ModelOutputError"
        assert payload["kind"] == "malformed_step_request"
        assert "body.actions.0" in payload["what"]
        assert env.seen_step_actions == []

    @pytest.mark.parametrize(
        ("action", "expected"),
        [
            (
                {
                    "id": "provider_1",
                    "type": "function",
                    "function": {"name": "computer", "arguments": {}},
                    "index": 0,
                },
                "noncanonical outer keys ['index']",
            ),
            (
                {
                    "id": "call_nested",
                    "type": "function",
                    "function": {
                        "name": "computer",
                        "arguments": {},
                        "index": 0,
                    },
                },
                "function has noncanonical keys ['index']",
            ),
            (
                {
                    "_result_call_id": "call_private",
                    "type": "function",
                    "function": {"name": "computer", "arguments": {}},
                },
                "_result_call_id is reserved",
            ),
            (
                {
                    "tool_call_id": "call_legacy",
                    "type": "function",
                    "function": {"name": "computer", "arguments": {}},
                },
                "must use canonical 'id'",
            ),
            (
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "computer", "arguments": {}},
                },
                "id must be a non-empty string",
            ),
        ],
    )
    def test_malformed_dict_actions_return_typed_model_output_error(
        self,
        action: dict[str, Any],
        expected: str,
    ):
        response, env = self._post_malformed_actions([action])

        assert response.status_code == 422, response.text
        payload = response.json()
        assert payload["error_type"] == "ModelOutputError"
        assert payload["kind"] == "malformed_step_request"
        assert expected in payload["what"]
        assert "provider payload" not in payload["what"]
        assert payload["payload_metadata"]["action_index"] == 0
        assert payload["payload_metadata"]["action_keys"] == sorted(action.keys())
        if (
            "function" in action
            and isinstance(action["function"], dict)
            and "name" in action["function"]
        ):
            assert payload["payload_metadata"]["name"] == action["function"]["name"]
        assert env.seen_step_actions == []

    def test_missing_step_id_is_forwarded_for_direct_server_parity(self):
        action = make_tool_call("computer", {})
        response, env = self._post_malformed_actions([action])

        assert response.status_code == 200, response.text
        assert env.seen_step_actions == [[action]]

    @pytest.mark.parametrize(
        ("action", "expected"),
        [
            (
                {"type": "function", "function": {"arguments": {}}},
                "function.name must be a non-empty string",
            ),
            (
                {"type": "function", "function": {"name": "computer"}},
                "action.function missing arguments dict",
            ),
        ],
    )
    def test_missing_id_still_rejects_malformed_tool_envelope(
        self,
        action: dict[str, Any],
        expected: str,
    ):
        response, env = self._post_malformed_actions([action])

        assert response.status_code == 422, response.text
        payload = response.json()
        assert payload["error_type"] == "ModelOutputError"
        assert payload["kind"] == "malformed_step_request"
        assert expected in payload["what"]
        assert env.seen_step_actions == []

    def test_nonfinite_action_field_stays_out_of_the_422_envelope(self):
        """Starlette's ``JSONResponse`` hard-codes ``allow_nan=False``, so a bare
        ``NaN`` echoed back from the rejected action fails the response render
        itself — turning a typed 422 into a broken reply.
        """
        # Raw body via json.dumps: httpx's encoder is strict, but the stdlib
        # default (``allow_nan=True``) is not, so a third-party client really can
        # put a bare ``NaN`` on the wire.
        client, patcher, created = _make_recording_client()
        try:
            iid = self._create_and_reset(client)
            response = client.post(
                f"/instances/{iid}/step",
                headers={**_h("zzh-dev"), "content-type": "application/json"},
                content=json.dumps(
                    {
                        "actions": [
                            {
                                "role": "assistant",
                                "id": float("nan"),
                                "call_id": float("inf"),
                            }
                        ]
                    }
                ),
            )
            env = created[0]
        finally:
            patcher.stop()

        assert response.status_code == 422, response.text
        payload = response.json()
        assert payload["error_type"] == "ModelOutputError"
        assert payload["payload_metadata"]["id"] == {"type": "float"}
        assert payload["payload_metadata"]["call_id"] == {"type": "float"}
        assert env.seen_step_actions == []

    def test_duplicate_step_ids_return_typed_model_output_error(self):
        response, env = self._post_malformed_actions(
            [
                make_tool_call("computer", {}, call_id="dup"),
                make_tool_call("bash", {"command": "pwd"}, call_id="dup"),
            ]
        )

        assert response.status_code == 422, response.text
        payload = response.json()
        assert payload["error_type"] == "ModelOutputError"
        assert "duplicate call id 'dup'" in payload["what"]
        assert payload["payload_metadata"]["action_index"] == 1
        assert payload["payload_metadata"]["duplicate_of_action_index"] == 0
        assert payload["payload_metadata"]["id"] == "dup"
        assert env.seen_step_actions == []

    @pytest.mark.parametrize(
        ("action", "expected"),
        [
            (
                {
                    "id": "call_missing_name",
                    "type": "function",
                    "function": {"arguments": {}},
                },
                "tool_call.function.name must be a non-empty string",
            ),
            (
                {
                    "id": "call_missing_args",
                    "type": "function",
                    "function": {"name": "computer"},
                },
                "tool_call.function missing arguments dict",
            ),
            (
                {
                    "id": "call_bad_args",
                    "type": "function",
                    "function": {"name": "computer", "arguments": ["bad"]},
                },
                "tool_call.function.arguments must be an object, got list",
            ),
        ],
    )
    def test_pairable_malformed_call_reaches_env_feedback(
        self,
        action: dict[str, Any],
        expected: str,
    ):
        client, patcher, created = _make_ingress_feedback_client()
        try:
            iid = self._create_and_reset(client)
            response = client.post(
                f"/instances/{iid}/step",
                headers=_h("zzh-dev"),
                json={"actions": [action]},
            )

            assert response.status_code == 200, response.text
            assert created[0].seen_step_actions == [[action]]
            result = decode_step_result(response.content)
            assert len(result.results) == 1
            tool_result = result.results[0]
            assert tool_result.tool_call_id == action["id"]
            assert tool_result.error is not None and expected in tool_result.error
            assert tool_result.metadata and tool_result.metadata["is_error"] is True
        finally:
            patcher.stop()

    def test_content_only_internal_finish_without_call_id_still_reaches_env(self):
        client, patcher, created = _make_recording_client()
        try:
            iid = self._create_and_reset(client)
            actions = make_no_tool_call_final_actions("done")
            response = client.post(
                f"/instances/{iid}/step",
                headers=_h("zzh-dev"),
                json={"actions": actions},
            )

            assert response.status_code == 200, response.text
            assert created[0].seen_step_actions == [actions]
            # The sidecar is an ORDINARY dict key, so real /step JSON carries it.
            # Without it the env would classify this as a standalone ``response``
            # tool call instead of an internal finish.
            seen = created[0].seen_step_actions[0][0]
            assert seen[RUNTIME_INTERNAL_STOP_REASON_KEY] == CONTENT_ONLY_FINAL_REASON
        finally:
            patcher.stop()

    def test_loop_detect_terminate_routes_its_result_over_step_json(self):
        """Result routing must survive the wire, not just the direct env call.

        ``LoopDetectWrapper`` sits ABOVE the remote client, so its synthetic
        terminate reaches the server as ``/step`` JSON. If a sidecar were dropped
        there, the intercepted model call would silently lose its role:tool
        result and the stop reason would never reach the env.
        """
        client, patcher, created = _make_ingress_feedback_client()
        try:
            iid = self._create_and_reset(client)
            action = make_internal_terminate_action(result_call_id="call_0001")
            response = client.post(
                f"/instances/{iid}/step",
                headers=_h("zzh-dev"),
                json={"actions": [action]},
            )

            assert response.status_code == 200, response.text
            assert created[0].seen_step_actions == [[action]]
            seen = created[0].seen_step_actions[0][0]
            assert seen[RUNTIME_RESULT_CALL_ID_KEY] == "call_0001"
            assert seen[RUNTIME_INTERNAL_STOP_REASON_KEY] == LOOP_DETECT_TERMINATE_REASON
            assert "id" not in seen

            # The intercepted model call still gets its paired result.
            result = decode_step_result(response.content)
            assert [r.tool_call_id for r in result.results] == ["call_0001"]
        finally:
            patcher.stop()

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"actions": "x"},
            [],
        ],
    )
    def test_step_body_validation_errors_return_typed_model_output_error(self, body: Any):
        client, patcher, created = _make_recording_client()
        try:
            iid = self._create_and_reset(client)
            response = client.post(
                f"/instances/{iid}/step",
                headers=_h("zzh-dev"),
                json=body,
            )

            assert response.status_code == 422, response.text
            payload = response.json()
            assert payload["error_type"] == "ModelOutputError"
            assert payload["kind"] == "malformed_step_request"
            assert "malformed /step request body" in payload["what"]
            assert created[0].seen_step_actions == []
        finally:
            patcher.stop()

    def test_step_malformed_json_returns_typed_model_output_error(self):
        client, patcher, created = _make_recording_client()
        try:
            iid = self._create_and_reset(client)
            response = client.post(
                f"/instances/{iid}/step",
                headers={**_h("zzh-dev"), "Content-Type": "application/json"},
                data=b'{"actions": [',
            )

            assert response.status_code == 422, response.text
            payload = response.json()
            assert payload["error_type"] == "ModelOutputError"
            assert payload["kind"] == "malformed_step_request"
            assert "malformed /step request body" in payload["what"]
            assert created[0].seen_step_actions == []
        finally:
            patcher.stop()

    def test_non_step_validation_keeps_fastapi_shape(self):
        client, patcher = _make_client(token=None)
        try:
            r = client.post("/instances", json={"env_kwargs": {}})
            assert r.status_code == 422, r.text
            assert "error_type" not in r.json()
            assert "detail" in r.json()
        finally:
            patcher.stop()


class _FailingResetEnv(LiteBaseEnv):
    """Env whose reset() always raises a terminal (non-transient) error."""

    def _runtime_metadata(self) -> LiteCUAMetadata:
        return LiteCUAMetadata(dims=("browser", "use"))

    async def reset(self) -> LiteEnvObservation:
        raise ValueError("synthetic terminal reset failure")

    async def step(self, actions: list) -> LiteEnvStepResult:
        return LiteEnvStepResult()

    async def close(self) -> None:
        pass


class TestResetFailureReleasesSlot:
    """C2 regression: a terminal ``/reset`` failure must reclaim the admission
    slot + the ``state.envs`` entry immediately (mirroring DELETE), not leave it
    pinned until the idle-TTL reaper fires. Otherwise a client that abandons the
    instance after a reset error (crash / request timeout, i.e. no DELETE) pins
    capacity for ``idle_ttl_sec``. The probe uses ``max_live_envs=1`` so a leaked
    slot would make the follow-up create 503 instead of 201."""

    def test_terminal_reset_failure_frees_slot_immediately(self):
        state = State(
            admission=_make_test_admission(max_live_envs=1),
            idle_ttl_sec=3600.0,  # huge TTL: only the immediate-reclaim fix can free it
        )
        app = make_app(state, token="zzh-dev")
        patcher = patch(
            "lite.gym.remote.server.gym.make",
            lambda env_key, **kw: _FailingResetEnv(),
        )
        patcher.start()
        try:
            # raise_server_exceptions=False: let the unhandled ValueError become
            # a 500 response (as it would over a real socket) instead of
            # re-raising into the test — the handler's slot-reclaim runs either
            # way (it's before the re-raise), but we want to assert on state after.
            client = TestClient(app, raise_server_exceptions=False)
            iid = _create_env(client, "zzh-dev", "lite.demo", "s1")
            assert len(state.envs) == 1
            # terminal reset failure surfaces as 500 ...
            r = client.post(f"/instances/{iid}/reset", headers=_h("zzh-dev"))
            assert r.status_code == 500, r.text
            # ... with the section B7 typed envelope: the body names the raised
            # class (not FastAPI's opaque "Internal Server Error"), so the
            # client can reconstruct a typed error.
            assert r.json()["error_type"] == "ValueError", r.text
            # ... and the slot + entry are reclaimed synchronously (not idle-TTL).
            assert len(state.envs) == 0
            # capacity is free again despite max_live_envs=1 — the proof the
            # admission slot was released, not just the dict entry.
            r2 = client.post(
                "/instances",
                headers=_h("zzh-dev"),
                json={"env_key": "lite.demo@t", "env_kwargs": {}, "session_id": "s2"},
            )
            assert r2.status_code == 201, r2.text
        finally:
            patcher.stop()


class TestBulkDeleteMatrix:
    """The 7 alice scenarios + bob's isolation control, all in-process.

    Drives ``DELETE /instances?session_id=&env_id=&force=&dry_run=``
    through every combination of pinned/unpinned filters × force ×
    dry_run, asserting the wide-net guard kicks in only when neither
    session_id nor env_id is pinned and force isn't set.
    """

    def setup_method(self):
        self.client, self._patcher = _make_client(token=None)  # passthrough
        # alice has 4 envs across 2 sessions × 2 env_ids
        for sess in ("sA", "sB"):
            for env_id in ("x", "y"):
                _create_env(self.client, "alice", env_id, sess)
        # bob has 4 envs same shape
        for sess in ("sA", "sB"):
            for env_id in ("x", "y"):
                _create_env(self.client, "bob", env_id, sess)

    def teardown_method(self):
        self._patcher.stop()

    def _list(self, token: str) -> list[dict]:
        r = self.client.get("/instances", headers=_h(token))
        return r.json()["instances"]

    def test_initial_state_isolated(self):
        assert len(self._list("alice")) == 4
        assert len(self._list("bob")) == 4

    def test_case1_pinned_both_no_force_succeeds(self):
        # Pinned scope (session_id + env_id) — force not required.
        r = self.client.delete(
            "/instances",
            headers=_h("alice"),
            params={"session_id": "sA", "env_id": "x"},
        )
        assert r.status_code == 200
        assert len(r.json()["closed"]) == 1  # alice's x@sA only
        assert len(self._list("alice")) == 3
        assert len(self._list("bob")) == 4

    def test_case2_env_id_only_without_force_400(self):
        r = self.client.delete(
            "/instances",
            headers=_h("alice"),
            params={"env_id": "x"},
        )
        assert r.status_code == 400
        assert "force" in r.json()["detail"].lower()

    def test_case3_session_id_only_without_force_400(self):
        r = self.client.delete(
            "/instances",
            headers=_h("alice"),
            params={"session_id": "sA"},
        )
        assert r.status_code == 400

    def test_case4_empty_body_without_force_400(self):
        r = self.client.delete(
            "/instances",
            headers=_h("alice"),
        )
        assert r.status_code == 400

    def test_case5_env_id_only_with_force_kills_env_id_wide(self):
        # alice has 2 x-envs (x@sA + x@sB); force unlocks env_id-wide cleanup.
        r = self.client.delete(
            "/instances",
            headers=_h("alice"),
            params={"env_id": "x", "force": True},
        )
        assert r.status_code == 200
        assert len(r.json()["closed"]) == 2
        # alice: 2 y-envs left; bob still 4
        assert len(self._list("alice")) == 2
        assert len(self._list("bob")) == 4

    def test_case6_session_id_only_with_force_kills_session_wide(self):
        r = self.client.delete(
            "/instances",
            headers=_h("alice"),
            params={"session_id": "sA", "force": True},
        )
        assert r.status_code == 200
        assert len(r.json()["closed"]) == 2  # x@sA + y@sA
        assert len(self._list("alice")) == 2
        assert len(self._list("bob")) == 4

    def test_case7_force_only_kills_everything_under_token(self):
        # User kill switch.
        r = self.client.delete(
            "/instances",
            headers=_h("alice"),
            params={"force": True},
        )
        assert r.status_code == 200
        assert len(r.json()["closed"]) == 4
        assert len(self._list("alice")) == 0
        # CRITICAL: bob's envs survived alice's kill switch.
        assert len(self._list("bob")) == 4

    def test_token_hash_isolation_across_all_cleanup_variants(self):
        # Run all wide-net variants as alice; bob's envs never touched.
        for params in [
            {"env_id": "x", "force": True},
            {"session_id": "sA", "force": True},
            {"force": True},
        ]:
            r = self.client.delete(
                "/instances",
                headers=_h("alice"),
                params=params,
            )
            assert r.status_code == 200
        assert len(self._list("alice")) == 0
        assert len(self._list("bob")) == 4

    def test_dry_run_returns_would_close_without_deleting(self):
        # Dry-run on the user kill switch — should preview ALL 4 alice envs.
        r = self.client.delete(
            "/instances",
            headers=_h("alice"),
            params={"force": True, "dry_run": True},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["dry_run"] is True
        assert len(body["would_close"]) == 4
        assert "closed" not in body
        # CRITICAL: alice's envs still alive after dry-run.
        assert len(self._list("alice")) == 4
        assert len(self._list("bob")) == 4

    def test_dry_run_respects_scope_and_safety_gate(self):
        # Dry-run on a pinned scope — no force needed, returns the 1 it would hit.
        r = self.client.delete(
            "/instances",
            headers=_h("alice"),
            params={"session_id": "sA", "env_id": "x", "dry_run": True},
        )
        assert r.status_code == 200
        assert len(r.json()["would_close"]) == 1
        assert len(self._list("alice")) == 4  # still all 4
        # Dry-run on unpinned scope without force still fires the 400 gate
        # (because dry-run shouldn't bypass the explicit confirmation;
        # the operator should still SEE the safety prompt for unscoped).
        r = self.client.delete(
            "/instances",
            headers=_h("alice"),
            params={"env_id": "x", "dry_run": True},
        )
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def setup_method(self):
        self.client, self._patcher = _make_client(token=None, max_live_envs=50)

    def teardown_method(self):
        self._patcher.stop()

    def test_metrics_baseline_empty(self):
        r = self.client.get("/metrics", headers=_h("ops"))
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/plain")
        body = r.text
        assert "cua_lite_envs_total 0" in body
        # admission lines replace cost+budget.
        assert "cua_lite_admission_max_live_envs 50" in body
        assert "cua_lite_admission_in_flight 0" in body

    def test_metrics_aggregates_across_tenants(self):
        _create_env(self.client, "alice", "x", "s1")
        _create_env(self.client, "bob", "x", "s1")
        _create_env(self.client, "bob", "y", "s1")
        r = self.client.get("/metrics", headers=_h("ops"))
        body = r.text
        # ops sees ALL tenants' envs (operator view, not per-token-scoped).
        assert "cua_lite_envs_total 3" in body
        assert 'cua_lite_envs_by_id{env_id="x"} 2' in body
        assert 'cua_lite_envs_by_id{env_id="y"} 1' in body
        # in_flight, no cost units.
        assert "cua_lite_admission_in_flight 3" in body

    def test_metrics_strict_mode_rejects_wrong_token(self):
        client, patcher = _make_client(token="strict_T", max_live_envs=50)
        try:
            r = client.get("/metrics", headers=_h("alice"))
            assert r.status_code == 401
            r = client.get("/metrics", headers=_h("strict_T"))
            assert r.status_code == 200
        finally:
            patcher.stop()

    def test_metrics_counters_after_step(self):
        # Drive one env through reset + step + close; verify counters
        # appear in /metrics output.
        env_id = _create_env(self.client, "alice", "x", "s1")
        self.client.post(f"/instances/{env_id}/reset", headers=_h("alice"))
        self.client.post(
            f"/instances/{env_id}/step",
            headers=_h("alice"),
            json={"actions": []},
        )
        r = self.client.get("/metrics", headers=_h("ops"))
        body = r.text
        assert 'cua_lite_reset_total{env_id="x"} 1' in body
        assert 'cua_lite_step_total{env_id="x"} 1' in body
        # Histogram: at least the +Inf bucket should be present.
        assert 'cua_lite_step_duration_seconds_bucket{env_id="x",le="+Inf"} 1' in body
        assert 'cua_lite_step_duration_seconds_count{env_id="x"} 1' in body
        # No errors yet.
        assert "cua_lite_asgi_errors_total 0" in body
        # Ghost-env counter present (0 until the ghost reaper fires).
        assert "cua_lite_envs_dead_total 0" in body


class TestHttpKeepAlive:
    """``LiteEnvClient`` configures explicit httpx limits to keep
    connections warm under 32+ concurrent rollouts."""

    def test_client_has_explicit_limits(self, monkeypatch):
        # Lightweight smoke — just verify the httpx.Limits is set with
        # values comfortably above ENV_CONCURRENCY=32. Skip the eager
        # POST /instances (would fail on DNS resolution to "http://x");
        # this test isn't about the boot path.
        from lite.gym.remote.client import LiteEnvClient

        monkeypatch.setattr(LiteEnvClient, "_eager_create_sync", lambda self: None)
        c = LiteEnvClient(server_url="http://x", env_key="x@y", token="t")
        limits = c._client._transport._pool._max_connections
        keep = c._client._transport._pool._max_keepalive_connections
        assert limits >= 100
        assert keep >= 32


# ---------------------------------------------------------------------------
# Multi-tenant single-server (passthrough)
# ---------------------------------------------------------------------------


_DISPATCH_SCOPE = ServerScope(server_port=30100, token_hash="alice")


class TestStartupRecoveryDispatch:
    """env-server lifespan startup dispatches boot recovery via
    :func:`lite.gym.remote.recovery.recover_all` — the reconcile loop with an empty
    tracked set per env-id, which calls each env's ``EnvServices.live_ids`` then
    ``reap``. Tests verify the dispatcher iterates registered envs and reclaims
    orphans. Docker-side scoping is covered by ``tests/gym/remote/test_reaper.py``."""

    def test_dispatcher_calls_per_env_reap(self, monkeypatch):
        # ``from lite.gym import registry`` shadows the submodule with a
        # Registry instance; fetch the real module from sys.modules.
        import sys

        import lite.gym.registry  # noqa: F401 — populates sys.modules

        reg = sys.modules["lite.gym.registry"]
        import lite.gym.remote.recovery as recovery
        from lite.gym.services import EnvServices

        # Two fake env-id services objects. live_ids returns a (non-None) set so
        # reap runs at boot (in_use is empty since tracked is empty).
        called: list[tuple[str, frozenset, bool]] = []

        class _FakeSvc(EnvServices):
            def __init__(self, eid: str, n: int):
                self._eid, self._n = eid, n

            def live_ids(self, env_id, scope):
                return set()

            def reap(self, env_id, scope, in_use, *, boot=False):
                called.append((self._eid, frozenset(in_use), boot))
                return self._n

        monkeypatch.setattr(
            reg,
            "_specs",
            {
                "fake_a@t1": object(),
                "fake_b@t1": object(),
            },
        )
        monkeypatch.setattr(
            reg,
            "_services",
            {
                "fake_a": _FakeSvc("fake_a", 3),
                "fake_b": _FakeSvc("fake_b", 0),
            },
        )
        monkeypatch.setattr(reg, "_families", _dedicated_families(reg))
        # Suppress _import_all side effects (it'd re-import the real envs dir).
        monkeypatch.setattr(reg, "_import_all", lambda: None)

        total = recovery.recover_all(_DISPATCH_SCOPE)
        assert total == 3  # 3 + 0
        envs_called = {c[0] for c in called}
        assert envs_called == {"fake_a", "fake_b"}
        # Boot recovery: in_use is empty (no tracked instances) and the framework
        # supplies boot=True (recover_all is the one-shot boot path).
        for _eid, in_use, boot in called:
            assert in_use == frozenset()
            assert boot is True

    def test_dispatcher_skips_envs_without_services(self, monkeypatch):
        """Envs with no services object (e.g. osworld) are silently skipped."""
        import sys

        import lite.gym.registry  # noqa: F401 — populates sys.modules

        reg = sys.modules["lite.gym.registry"]
        import lite.gym.remote.recovery as recovery

        monkeypatch.setattr(reg, "_specs", {"fake_nohook@t1": object()})
        monkeypatch.setattr(reg, "_services", {})
        monkeypatch.setattr(reg, "_import_all", lambda: None)

        # Must not raise.
        total = recovery.recover_all(_DISPATCH_SCOPE)
        assert total == 0

    def test_dispatcher_swallows_reap_exceptions(self, monkeypatch):
        """One env's reap failure must not abort the others — failures
        log + continue."""
        import sys

        import lite.gym.registry  # noqa: F401 — populates sys.modules

        reg = sys.modules["lite.gym.registry"]
        import lite.gym.remote.recovery as recovery
        from lite.gym.services import EnvServices

        survivor_calls: list[int] = []

        class _Bad(EnvServices):
            def live_ids(self, env_id, scope):
                return set()

            def reap(self, env_id, scope, in_use, *, boot=False):
                raise RuntimeError("docker daemon unreachable")

        class _Good(EnvServices):
            def live_ids(self, env_id, scope):
                return set()

            def reap(self, env_id, scope, in_use, *, boot=False):
                survivor_calls.append(1)
                return 5

        monkeypatch.setattr(
            reg,
            "_specs",
            {
                "fake_bad@t1": object(),
                "fake_good@t1": object(),
            },
        )
        monkeypatch.setattr(
            reg,
            "_services",
            {
                "fake_bad": _Bad(),
                "fake_good": _Good(),
            },
        )
        monkeypatch.setattr(reg, "_families", _dedicated_families(reg))
        monkeypatch.setattr(reg, "_import_all", lambda: None)

        total = recovery.recover_all(_DISPATCH_SCOPE)
        assert survivor_calls == [1], "survivor reap must still run"
        assert total == 5  # only survivor's count contributes


class TestShutdownAllDispatch:
    """env-server lifespan finally dispatches per-env ``EnvServices.shutdown``
    via :func:`lite.gym.remote.recovery.shutdown_all` (deduped by ``id(svc)``)."""

    def test_shutdown_dispatcher_calls_shutdown(self, monkeypatch):
        import sys

        import lite.gym.registry  # noqa: F401

        reg = sys.modules["lite.gym.registry"]
        import lite.gym.remote.recovery as recovery
        from lite.gym.services import EnvServices

        called: list[tuple[str, object]] = []

        class _FakeSvc(EnvServices):
            def __init__(self, eid: str):
                self._eid = eid

            def shutdown(self, env_id, scope):
                called.append((self._eid, scope))

        monkeypatch.setattr(
            reg,
            "_specs",
            {
                "fake_a@t1": object(),
                "fake_b@t1": object(),
            },
        )
        monkeypatch.setattr(
            reg,
            "_services",
            {
                "fake_a": _FakeSvc("fake_a"),
                "fake_b": _FakeSvc("fake_b"),
            },
        )
        monkeypatch.setattr(reg, "_families", _dedicated_families(reg))
        monkeypatch.setattr(reg, "_import_all", lambda: None)

        n = recovery.shutdown_all(_DISPATCH_SCOPE)
        assert n == 2  # two distinct services dispatched
        assert {c[0] for c in called} == {"fake_a", "fake_b"}
        for _eid, scope in called:
            assert scope is _DISPATCH_SCOPE

    def test_shutdown_dispatcher_ignores_envs_without_services(self, monkeypatch):
        """Envs with no services object are skipped silently."""
        import sys

        import lite.gym.registry  # noqa: F401

        reg = sys.modules["lite.gym.registry"]
        import lite.gym.remote.recovery as recovery

        monkeypatch.setattr(reg, "_specs", {"fake_nohook@t1": object()})
        monkeypatch.setattr(reg, "_services", {})
        monkeypatch.setattr(reg, "_import_all", lambda: None)

        n = recovery.shutdown_all(_DISPATCH_SCOPE)
        assert n == 0

    def test_shutdown_dispatcher_continues_on_exception(self, monkeypatch):
        """One env's shutdown failure (e.g. pgrep timeout) must not
        block other envs' shutdown."""
        import sys

        import lite.gym.registry  # noqa: F401

        reg = sys.modules["lite.gym.registry"]
        import lite.gym.remote.recovery as recovery
        from lite.gym.services import EnvServices

        survivor_calls: list[int] = []

        class _Bad(EnvServices):
            def shutdown(self, env_id, scope):
                raise RuntimeError("kill -9 failed")

        class _Good(EnvServices):
            def shutdown(self, env_id, scope):
                survivor_calls.append(1)

        monkeypatch.setattr(
            reg,
            "_specs",
            {
                "fake_bad@t1": object(),
                "fake_good@t1": object(),
            },
        )
        monkeypatch.setattr(
            reg,
            "_services",
            {
                "fake_bad": _Bad(),
                "fake_good": _Good(),
            },
        )
        monkeypatch.setattr(reg, "_families", _dedicated_families(reg))
        monkeypatch.setattr(reg, "_import_all", lambda: None)

        n = recovery.shutdown_all(_DISPATCH_SCOPE)
        assert survivor_calls == [1]
        assert n == 1  # only survivor counted (failed one not incremented)


class TestReconcileAllDispatch:
    """env-server drift cycle dispatches per-env reconcile via
    :func:`lite.gym.remote.recovery.reconcile_all` — the steady-state path that runs
    every ~120 s. Mirrors the recover_all/shutdown_all skip + isolation tests."""

    @staticmethod
    def _empty_state_view():
        from lite.gym.types import StateView

        return StateView(by_env_id={}, snapshot_at=0.0)

    def test_reconcile_all_skips_envs_without_services(self, monkeypatch):
        """An env_id with no services object is skipped — empty result, no crash."""
        import sys

        import lite.gym.registry  # noqa: F401

        reg = sys.modules["lite.gym.registry"]
        import lite.gym.remote.recovery as recovery

        monkeypatch.setattr(reg, "_specs", {"fake_nohook@t1": object()})
        monkeypatch.setattr(reg, "_services", {})
        monkeypatch.setattr(reg, "_import_all", lambda: None)
        reports = recovery.reconcile_all(self._empty_state_view(), _DISPATCH_SCOPE)
        assert reports == []  # no EnvServices → nothing dispatched

    def test_reconcile_all_isolates_raising_env(self, monkeypatch):
        """One env's live_ids/reap raising must not abort the others' reconcile."""
        import sys

        import lite.gym.registry  # noqa: F401

        reg = sys.modules["lite.gym.registry"]
        import lite.gym.remote.recovery as recovery
        from lite.gym.services import EnvServices

        survivor: list[int] = []

        class _Bad(EnvServices):
            def live_ids(self, env_id, scope):
                raise RuntimeError("docker ps wedged")

        class _Good(EnvServices):
            def live_ids(self, env_id, scope):
                return set()

            def reap(self, env_id, scope, in_use, *, boot=False):
                survivor.append(1)
                return 0

        monkeypatch.setattr(
            reg,
            "_specs",
            {
                "fake_bad@t1": object(),
                "fake_good@t1": object(),
            },
        )
        monkeypatch.setattr(reg, "_services", {"fake_bad": _Bad(), "fake_good": _Good()})
        monkeypatch.setattr(reg, "_families", _dedicated_families(reg))
        monkeypatch.setattr(reg, "_import_all", lambda: None)
        reports = recovery.reconcile_all(self._empty_state_view(), _DISPATCH_SCOPE)
        assert survivor == [1], "survivor reconcile must run after the bad env raised"
        assert len(reports) == 2  # bad env contributes an empty ReapReport, not a crash


class TestFamilyGatingParity:
    """Recovery sweeps gate on the DECLARED family, not on
    ``isinstance(EnvServices)``. Pins the dispatch parity the all-DEDICATED fakes elsewhere
    miss — the dangerous one being **SINGLETON is recovered at boot but SKIPPED at steady
    reconcile** (admitting it would ``docker rm`` the shared backend mid-run). PURE/REMOTE
    are skipped everywhere; the F3 validator fails fast on a services env with no family."""

    @staticmethod
    def _empty_state_view():
        from lite.gym.types import StateView

        return StateView(by_env_id={}, snapshot_at=0.0)

    @staticmethod
    def _mixed(monkeypatch):
        """3 fakes spanning DEDICATED / SINGLETON / PURE, each recording its reap/shutdown."""
        import sys

        import lite.gym.registry  # noqa: F401

        reg = sys.modules["lite.gym.registry"]
        import lite.gym.remote.recovery as recovery
        from lite.gym.services import BackendFamily, EnvServices

        calls: list[tuple] = []

        class _Svc(EnvServices):
            def __init__(self, eid):
                self._eid = eid

            def live_ids(self, env_id, scope):
                return set()

            def reap(self, env_id, scope, in_use, *, boot=False):
                calls.append((self._eid, "reap", boot))
                return 0

            def shutdown(self, env_id, scope):
                calls.append((self._eid, "shutdown", None))

        monkeypatch.setattr(reg, "_specs", {})
        monkeypatch.setattr(
            reg,
            "_services",
            {
                "f_ded": _Svc("f_ded"),
                "f_sing": _Svc("f_sing"),
                "f_pure": _Svc("f_pure"),
            },
        )
        monkeypatch.setattr(
            reg,
            "_families",
            {
                "f_ded": BackendFamily.DEDICATED,
                "f_sing": BackendFamily.SINGLETON,
                "f_pure": BackendFamily.PURE,
            },
        )
        monkeypatch.setattr(reg, "_import_all", lambda: None)
        return reg, recovery, calls

    def test_steady_reconcile_reaps_only_dedicated(self, monkeypatch):
        reg, recovery, calls = self._mixed(monkeypatch)
        recovery.reconcile_all(self._empty_state_view(), _DISPATCH_SCOPE)
        reaped = {eid for eid, kind, _ in calls if kind == "reap"}
        assert reaped == {"f_ded"}, f"steady reconcile must reap ONLY DEDICATED, got {reaped}"

    def test_boot_recover_dedicated_and_singleton_not_pure(self, monkeypatch):
        reg, recovery, calls = self._mixed(monkeypatch)
        recovery.recover_all(_DISPATCH_SCOPE)
        booted = {eid for eid, kind, boot in calls if kind == "reap" and boot}
        assert booted == {"f_ded", "f_sing"}, f"boot recover = DEDICATED+SINGLETON, got {booted}"

    def test_shutdown_dedicated_and_singleton_not_pure(self, monkeypatch):
        reg, recovery, calls = self._mixed(monkeypatch)
        recovery.shutdown_all(_DISPATCH_SCOPE)
        shut = {eid for eid, kind, _ in calls if kind == "shutdown"}
        assert shut == {"f_ded", "f_sing"}, f"shutdown = DEDICATED+SINGLETON, got {shut}"

    def test_f3_validator_raises_on_services_without_family(self, monkeypatch):
        """F3: an env that registered EnvServices but no family fails fast at boot."""
        import sys

        import lite.gym.registry  # noqa: F401

        reg = sys.modules["lite.gym.registry"]
        import lite.gym.remote.recovery as recovery
        from lite.gym.services import EnvServices

        monkeypatch.setattr(reg, "_services", {"orphan_env": EnvServices()})
        monkeypatch.setattr(reg, "_families", {})
        monkeypatch.setattr(reg, "_import_all", lambda: None)
        with pytest.raises(RuntimeError, match="orphan_env"):
            recovery.recover_all(_DISPATCH_SCOPE)

    def test_f3_validator_passes_on_real_registry(self):
        """The real 13-env registry: every services-registered env declares a family."""
        import importlib

        reg = importlib.import_module("lite.gym.registry")
        import lite.gym.remote.recovery as recovery

        reg._import_all()
        recovery._validate_families_declared()  # must NOT raise


class TestHealthCheckDispatch:
    """``GET /envs/{id}`` availability routes through
    :func:`lite.gym.services.health_check`, which dispatches to the typed
    ``EnvServices.health`` and surfaces its error."""

    def test_health_check_propagates_deps_missing(self, monkeypatch):
        import sys

        import lite.gym.registry  # noqa: F401

        reg = sys.modules["lite.gym.registry"]
        from lite.gym.errors import EnvDepsMissingError
        from lite.gym.services import EnvServices, health_check

        class _Svc(EnvServices):
            def health(self, env_id):
                raise EnvDepsMissingError("deps missing", "run install.sh", "README.md")

        monkeypatch.setattr(reg, "_services", {"fake": _Svc()})
        with pytest.raises(EnvDepsMissingError):
            health_check("fake")

    def test_health_check_noop_without_services(self, monkeypatch):
        import sys

        import lite.gym.registry  # noqa: F401

        reg = sys.modules["lite.gym.registry"]
        from lite.gym.services import health_check

        monkeypatch.setattr(reg, "_services", {})
        health_check("fake")  # no services object → silent no-op, must not raise


class TestEnsureServicesDispatch:
    """First-use service startup routes through :func:`registry.ensure_services`,
    which dispatches to the typed ``EnvServices.ensure`` (in direct mode) and
    surfaces its error. (Parallel to TestHealthCheckDispatch.)"""

    def test_ensure_propagates_deps_missing(self, monkeypatch):
        import sys

        import lite.gym.registry  # noqa: F401

        reg = sys.modules["lite.gym.registry"]
        from lite.gym.errors import EnvDepsMissingError
        from lite.gym.services import EnvServices

        class _Svc(EnvServices):
            def ensure(self, env_id):
                raise EnvDepsMissingError("deps missing", "run install.sh", "README.md")

        monkeypatch.delenv("CUA_LITE_ENV_SERVER_URL", raising=False)  # direct mode
        monkeypatch.setattr(reg, "_services", {"fake": _Svc()})
        monkeypatch.setattr(reg, "_services_started", set())  # not yet started
        monkeypatch.setattr(reg, "_import_env", lambda *_args, **_kwargs: None)
        with pytest.raises(EnvDepsMissingError):
            reg.ensure_services("fake")

    def test_ensure_skipped_in_remote_mode(self, monkeypatch):
        """In remote-client mode the env-server owns startup — never dispatch."""
        import sys

        import lite.gym.registry  # noqa: F401

        reg = sys.modules["lite.gym.registry"]
        from lite.gym.services import EnvServices

        class _Svc(EnvServices):
            def ensure(self, env_id):
                raise AssertionError("ensure must NOT run in remote mode")

        monkeypatch.setenv("CUA_LITE_ENV_SERVER_URL", "http://x")
        monkeypatch.setattr(reg, "_services", {"fake": _Svc()})
        monkeypatch.setattr(reg, "_services_started", set())
        reg.ensure_services("fake")  # must return without dispatching


def test_register_services_rejects_non_envservices():
    """register_services fails fast on a non-EnvServices object — a mis-typed
    registration would otherwise be silently skipped everywhere (orphan leak)."""
    from lite.gym.services import register_services

    with pytest.raises(TypeError, match="EnvServices"):
        register_services("nonsense_env", object())


class _SlowCloseEnv(_NoopEnv):
    """Env whose ``close()`` sleeps to simulate a real ``docker rm -f``
    (which takes 1-3 s and serializes through the docker daemon).
    Lets tests verify that the DELETE handler doesn't block on close."""

    sleep_s: float = 0.3

    async def close(self) -> None:
        await asyncio.sleep(self.sleep_s)


class TestDetachedClose:
    """``DELETE /envs/<id>`` pops the env (releasing its L2 slot) inside
    the lock then spawns ``_close_quietly`` as a background task in
    ``state.pending_closes`` — HTTP response returns in <10 ms while
    ``docker rm -f`` runs detached.  Lifespan shutdown drains the set
    with a deadline so SIGTERM doesn't orphan the closes."""

    def _make(self) -> tuple[TestClient, State, object]:
        state = State(
            admission=_make_test_admission(max_live_envs=100),
            idle_ttl_sec=3600.0,
        )
        app = make_app(state, token=None)
        patcher = patch(
            "lite.gym.remote.server.gym.make",
            lambda env_key, **kw: _SlowCloseEnv(),
        )
        patcher.start()
        return TestClient(app), state, patcher

    def test_delete_returns_before_close_finishes(self):
        client, state, patcher = self._make()
        try:
            env_id = _create_env(client, "alice", "x", "s1")
            t0 = time.monotonic()
            r = client.delete(f"/instances/{env_id}", headers=_h("alice"))
            elapsed = time.monotonic() - t0
            assert r.status_code == 200
            # DELETE returns in <100 ms even though _SlowCloseEnv.close
            # sleeps 300 ms. The bg task is still running.
            assert elapsed < 0.15, (
                f"DELETE took {elapsed * 1000:.0f}ms; detached close should respond in <100ms"
            )
        finally:
            patcher.stop()

    def test_delete_releases_l2_slot_atomically(self):
        client, state, patcher = self._make()
        try:
            env_id = _create_env(client, "alice", "x", "s1")
            assert state.admission.in_flight == 1
            client.delete(f"/instances/{env_id}", headers=_h("alice"))
            # L2 slot released the moment state.envs.pop ran (inside lock),
            # which is before the bg task even starts.
            assert state.admission.in_flight == 0
            assert len(state.envs) == 0
        finally:
            patcher.stop()

    def test_delete_drains_bg_task_eventually(self):
        # Whether the bg task is mid-flight or already done by the
        # time the sync TestClient hands control back to the test
        # thread is asyncio-scheduling-dependent (the sync portal may
        # or may not yield enough cycles). What we DO need to
        # guarantee: after enough wall-clock, the done callback fires
        # and the set drains — no leaked tasks.
        client, state, patcher = self._make()
        try:
            env_id = _create_env(client, "alice", "x", "s1")
            client.delete(f"/instances/{env_id}", headers=_h("alice"))
            # Give the bg task wall-clock + a follow-up request to
            # drive the loop, then assert it drained.
            time.sleep(0.5)
            client.get("/instances", headers=_h("alice"))
            assert len(state.pending_closes) == 0
        finally:
            patcher.stop()

    def test_delete_idempotent_when_env_already_gone(self):
        # Same as before the patch: second DELETE returns 200 cleanly.
        client, state, patcher = self._make()
        try:
            env_id = _create_env(client, "alice", "x", "s1")
            r1 = client.delete(f"/instances/{env_id}", headers=_h("alice"))
            r2 = client.delete(f"/instances/{env_id}", headers=_h("alice"))
            assert r1.status_code == 200
            assert r2.status_code == 200
            # Only one bg task spawned (the second DELETE saw env=None).
            assert len(state.pending_closes) <= 1
        finally:
            patcher.stop()


class TestParallelBulkDelete:
    """``DELETE /instances`` closes victims in parallel via ``asyncio.gather``
    — matches the convention of ``_reap_idle`` and lifespan shutdown.
    Big when a 32+ stale-env wipe is needed; verifies wall-clock for a
    multi-env cleanup is bounded by max(close), not sum(close)."""

    def test_cleanup_parallelizes_slow_closes(self):
        state = State(admission=_make_test_admission(max_live_envs=100), idle_ttl_sec=3600.0)
        app = make_app(state, token=None)
        with patch(
            "lite.gym.remote.server.gym.make",
            lambda env_key, **kw: _SlowCloseEnv(),
        ):
            client = TestClient(app)
            # Spawn 8 envs; each close sleeps 0.3 s. Serial = ~2.4 s.
            # Parallel via gather = ~0.3 s.
            for _ in range(8):
                _create_env(client, "alice", "x", "s1")
            t0 = time.monotonic()
            r = client.delete(
                "/instances",
                headers=_h("alice"),
                params={"force": True},
            )
            elapsed = time.monotonic() - t0
            assert r.status_code == 200
            assert len(r.json()["closed"]) == 8
            # Allow generous margin (test runner jitter, asyncio
            # overhead). Serial would be ~2.4 s; we should be well
            # under that.
            assert elapsed < 1.0, (
                f"cleanup took {elapsed:.2f}s for 8 × 0.3s closes; expected parallel via gather"
            )


class TestPassthroughColocate:
    def setup_method(self):
        self.client, self._patcher = _make_client(token=None)

    def teardown_method(self):
        self._patcher.stop()

    def test_distinct_tokens_get_distinct_token_hashes(self):
        e_a = _create_env(self.client, "alice", "x", "s")
        e_b = _create_env(self.client, "bob", "x", "s")
        a = self.client.get("/instances", headers=_h("alice")).json()["instances"]
        b = self.client.get("/instances", headers=_h("bob")).json()["instances"]
        assert len(a) == 1 and a[0]["id"] == e_a
        assert len(b) == 1 and b[0]["id"] == e_b
        assert a[0]["token_hash"] != b[0]["token_hash"]

    def test_alice_cannot_delete_bobs_env(self):
        e_b = _create_env(self.client, "bob", "x", "s")
        # alice DELETE bob's env id → 403 owned by another bearer.
        r = self.client.delete(f"/instances/{e_b}", headers=_h("alice"))
        assert r.status_code == 403
        assert r.json()["error_type"] == "EnvAccessDenied"
        # bob's env still alive.
        b = self.client.get("/instances", headers=_h("bob")).json()["instances"]
        assert any(e["id"] == e_b for e in b)

    def test_anonymous_has_its_own_scope(self):
        # No Authorization header → identity "anonymous".
        # Note: TestClient sends no Authorization unless we add one.
        r = self.client.post(
            "/instances",
            json={"env_key": "x@t", "env_kwargs": {}, "session_id": "s"},
        )
        assert r.status_code == 201
        # anonymous can list its own
        body = self.client.get("/instances").json()["instances"]
        assert len(body) == 1
        assert body[0]["token_hash"] == "anonymous"


class TestAllowedEnvIds:
    """Verify --env-ids allow-list (None = accept all, set = restrict)."""

    def test_unset_accepts_any_env_id(self):
        client, patcher = _make_client(allowed_env_ids=None)
        try:
            r = client.post(
                "/instances",
                headers=_h("alice"),
                json={"env_key": "x@t", "env_kwargs": {}, "session_id": "s"},
            )
            assert r.status_code == 201
            r = client.post(
                "/instances",
                headers=_h("alice"),
                json={"env_key": "anything@t", "env_kwargs": {}, "session_id": "s"},
            )
            assert r.status_code == 201
        finally:
            patcher.stop()

    def test_allowed_env_id_admitted(self):
        client, patcher = _make_client(allowed_env_ids={"x", "y"})
        try:
            r = client.post(
                "/instances",
                headers=_h("alice"),
                json={"env_key": "x@t", "env_kwargs": {}, "session_id": "s"},
            )
            assert r.status_code == 201
        finally:
            patcher.stop()

    def test_disallowed_env_id_returns_typed_501_with_allowed_list(self):
        # An untyped body hits ``is_retryable``'s unknown→True default, so a
        # permanent misconfiguration would be retried to the caller's attempt cap.
        client, patcher = _make_client(allowed_env_ids={"x", "y"})
        try:
            r = client.post(
                "/instances",
                headers=_h("alice"),
                json={"env_key": "androidlab@bluecoins_1", "env_kwargs": {}, "session_id": "s"},
            )
            assert r.status_code == 501
            body = r.json()
            assert body["error_type"] == "EnvUnavailable"
            assert body["status_code"] == 501
            # Detail must name the offending env_id AND the allow-list,
            # so a misconfigured client can self-diagnose.
            detail = body["detail"]
            assert "androidlab" in detail
            assert "x" in detail and "y" in detail
            # The client reconstructs it as a TERMINAL typed error.
            from lite.gym.errors import is_retryable, lite_error_from_payload

            exc = lite_error_from_payload(body)
            assert isinstance(exc, EnvUnavailable) and not is_retryable(exc)
        finally:
            patcher.stop()

    def test_empty_allow_list_denies_all(self):
        client, patcher = _make_client(allowed_env_ids=set())
        try:
            r = client.post(
                "/instances",
                headers=_h("alice"),
                json={"env_key": "x@t", "env_kwargs": {}, "session_id": "s"},
            )
            assert r.status_code == 501
            assert r.json()["error_type"] == "EnvUnavailable"
            assert "<empty>" in r.json()["detail"]
        finally:
            patcher.stop()


class TestDepsMissingWireIdentity:
    """``EnvDepsMissingError`` must reach the client TYPED and TERMINAL from
    every served surface — including ``/reset`` and ``/step``, which have no
    catch of their own.

    It is an ``ImportError``, not a ``LiteGymError``, so the generic typed handler
    cannot see it; falling to the catch-all 500 makes the client wrap it as a
    retryable ``RemoteEnvError`` and burn the whole ``--max-attempts`` budget on
    a condition no retry can fix.

    It is reachable after create: ``lite.osworld``'s ``reset()`` re-runs the
    uncached ``_ensure_services`` every call, which raises for a stale image, a
    deleted qcow2, or a vanished ``/dev/kvm``.
    """

    def _client(self, *, raise_on: str):
        from lite.gym.errors import EnvDepsMissingError

        def _boom() -> None:
            raise EnvDepsMissingError(
                what="OSWorld image cua-lite/osworld:latest not built",
                install="uv run --no-sync bash lite/gym/envs/lite/osworld/scripts/install.sh",
                see="lite/gym/envs/lite/osworld/README.md",
            )

        class _DepsGoneEnv(_NoopEnv):
            async def reset(self) -> LiteEnvObservation:
                if raise_on == "reset":
                    _boom()
                return await super().reset()

            async def step(self, actions: list) -> LiteEnvStepResult:
                if raise_on == "step":
                    _boom()
                return await super().step(actions)

        state = State(
            admission=_make_test_admission(max_live_envs=10),
            idle_ttl_sec=3600.0,
        )
        app = make_app(state, token="zzh-dev")
        patcher = patch(
            "lite.gym.remote.server.gym.make",
            lambda env_key, **kw: _DepsGoneEnv(),
        )
        patcher.start()
        return TestClient(app), patcher, state

    @staticmethod
    def _assert_typed_501(r) -> None:
        from lite.gym.errors import is_retryable, lite_error_from_payload

        assert r.status_code == 501, r.text
        body = r.json()
        assert body["error_type"] == "EnvUnavailable"
        # server_message only — install/see are DIRECT MODE ONLY and must not
        # be shipped to a caller who cannot install on this machine.
        assert "install.sh" not in r.text and "README.md" not in r.text
        assert "not available on this server" in body["what"]
        exc = lite_error_from_payload(body)
        assert isinstance(exc, EnvUnavailable)
        assert is_retryable(exc) is False

    def test_reset_deps_missing_is_typed_501_not_catch_all_500(self):
        client, patcher, state = self._client(raise_on="reset")
        try:
            id_ = _create_env(client, "zzh-dev", "lite.osworld", "s")
            r = client.post(f"/instances/{id_}/reset", headers=_h("zzh-dev"))
            self._assert_typed_501(r)
            # The handler has only the request, so env_id is recovered from the
            # path via the live session — the operator needs it in the message.
            assert "lite.osworld" in r.json()["what"]
            # Terminal reset still reclaims the instance + its admission slot.
            assert id_ not in state.envs
        finally:
            patcher.stop()


# ---------------------------------------------------------------------------
# Env catalog endpoints — types tree
# ---------------------------------------------------------------------------


class TestEnvCatalog:
    """``GET /envs``, ``GET /envs/{env_id}``, ``GET /envs/{env_id}/tasks``
    — read-only registry surface. All routes share the same allow-list +
    block-list visibility rules from :func:`State`.
    """

    def _client_with_registry(
        self,
        *,
        registered: list[str],
        task_ids_by_env: dict[str, dict[str, list[str]]] | None = None,
        allowed: set[str] | None = None,
        blocked: set[str] | None = None,
    ) -> tuple[TestClient, Any]:
        """Build a TestClient with a stubbed registry.

        The catalog endpoints call ``gym.registry.task_ids(env_id)`` plus one
        of two membership accessors: ``registered_env_ids()`` (unscoped list)
        or ``is_known_env_id(env_id)`` (single-env probe, and the scoped list —
        it answers the same question without registering every other env).
        Patch all of them from the same ``registered`` list so the test
        controls what's "registered" without touching the real env
        directory walk.
        """
        state = State(
            admission=_make_test_admission(max_live_envs=100),
            idle_ttl_sec=3600.0,
            allowed_env_ids=allowed,
            blocked_env_ids=blocked or set(),
        )
        app = make_app(state, token=None)
        task_map = task_ids_by_env or {}

        def fake_task_ids(env_id: str) -> dict[str, list[str]]:
            if env_id not in task_map:
                raise KeyError(env_id)
            return task_map[env_id]

        p1 = patch(
            "lite.gym.remote.server.gym.registry.registered_env_ids",
            lambda: list(registered),
        )
        p1b = patch(
            "lite.gym.remote.server.gym.registry.is_known_env_id",
            lambda env_id: env_id in set(registered),
        )
        p2 = patch(
            "lite.gym.remote.server.gym.registry.task_ids",
            fake_task_ids,
        )
        p3 = patch(
            "lite.gym.remote.server.gym.registry.task_metadata",
            lambda env_id, tid: None,
        )
        # The catalog surface also runs the uniform health probe; a stubbed
        # registry must stub it too, or the test's outcome depends on whether
        # an EARLIER test in the same worker imported the real env module
        # (browsergym's real health() probes a live MiniWoB backend →
        # available:false on hosts without the stack — an order-dependent
        # flake observed under xdist).
        p4 = patch("lite.gym.services.health_check", lambda env_id: None)
        for p in (p1, p1b, p2, p3, p4):
            p.start()

        class _BundlePatcher:
            def stop(self):
                for p in (p1, p1b, p2, p3, p4):
                    p.stop()

        return TestClient(app), _BundlePatcher()

    def test_get_envs_bare_list_returns_registered_env_ids(self):
        client, patcher = self._client_with_registry(
            registered=["androidworld", "lite.osworld", "browsergym.miniwob"],
            task_ids_by_env={
                "androidworld": {"train": ["t1", "t2"], "eval": ["t3"]},
                "lite.osworld": {"train": []},
                "browsergym.miniwob": {"train": ["m1"]},
            },
        )
        try:
            r = client.get("/envs", headers=_h("alice"))
            assert r.status_code == 200
            assert sorted(r.json()) == sorted(
                ["androidworld", "lite.osworld", "browsergym.miniwob"]
            )
        finally:
            patcher.stop()

    def test_get_envs_expand_metadata_returns_full_map(self):
        # ?expand=metadata == old /envs/available shape.
        client, patcher = self._client_with_registry(
            registered=["androidworld", "lite.osworld"],
            task_ids_by_env={
                "androidworld": {"train": ["t1", "t2"], "eval": ["t3"]},
                "lite.osworld": {"train": ["o1"]},
            },
        )
        try:
            r = client.get(
                "/envs",
                params={"expand": "metadata"},
                headers=_h("alice"),
            )
            assert r.status_code == 200
            body = r.json()
            assert set(body.keys()) == {"androidworld", "lite.osworld"}
            assert body["androidworld"]["available"] is True
            assert body["androidworld"]["n_tasks"] == 3
            assert sorted(body["androidworld"]["splits"]) == ["eval", "train"]
            assert body["lite.osworld"]["n_tasks"] == 1
        finally:
            patcher.stop()

    def test_get_envs_unsupported_expand_400(self):
        client, patcher = self._client_with_registry(registered=["x"])
        try:
            r = client.get(
                "/envs",
                params={"expand": "weird"},
                headers=_h("alice"),
            )
            assert r.status_code == 400
            assert "expand" in r.json()["detail"]
        finally:
            patcher.stop()

    def test_get_envs_respects_allow_list(self):
        # Allow-list filters both /envs (bare) and /envs?expand=metadata.
        client, patcher = self._client_with_registry(
            registered=["x", "y", "z"],
            task_ids_by_env={
                "x": {"train": ["t"]},
                "y": {"train": ["t"]},
                "z": {"train": ["t"]},
            },
            allowed={"x", "y"},
        )
        try:
            assert sorted(client.get("/envs", headers=_h("a")).json()) == ["x", "y"]
            r = client.get(
                "/envs",
                params={"expand": "metadata"},
                headers=_h("a"),
            )
            assert set(r.json().keys()) == {"x", "y"}
        finally:
            patcher.stop()

    def test_get_envs_blocked_show_as_unavailable_in_expanded_view(self):
        # Blocked envs appear in the listing but with available=False so
        # clients can see them as known-but-unusable.
        client, patcher = self._client_with_registry(
            registered=["x", "y"],
            task_ids_by_env={"x": {"train": ["t"]}, "y": {"train": ["t"]}},
            blocked={"x"},
        )
        try:
            # Bare list still includes blocked.
            assert "x" in client.get("/envs", headers=_h("a")).json()
            r = client.get(
                "/envs",
                params={"expand": "metadata"},
                headers=_h("a"),
            )
            body = r.json()
            assert body["x"]["available"] is False
            assert "blocked" in body["x"]["error"].lower()
            assert body["y"]["available"] is True
        finally:
            patcher.stop()

    def test_get_one_env_returns_single_type_metadata(self):
        client, patcher = self._client_with_registry(
            registered=["androidworld"],
            task_ids_by_env={"androidworld": {"train": ["t1"], "eval": ["t2", "t3"]}},
        )
        try:
            r = client.get("/envs/androidworld", headers=_h("a"))
            assert r.status_code == 200
            body = r.json()
            assert body["available"] is True
            assert body["n_tasks"] == 3
            assert sorted(body["splits"]) == ["eval", "train"]
            # env_cost no longer exposed.
            assert "env_cost" not in body
        finally:
            patcher.stop()

    def test_get_one_env_404_when_not_registered(self):
        client, patcher = self._client_with_registry(registered=["x"])
        try:
            r = client.get("/envs/nope", headers=_h("a"))
            assert r.status_code == 404
            assert "not registered" in r.json()["detail"]
        finally:
            patcher.stop()

    def test_get_one_env_with_dotted_env_id(self):
        # ``browsergym.miniwob`` and ``lite.osworld`` contain dots —
        # FastAPI path params accept them natively, no encoding needed.
        client, patcher = self._client_with_registry(
            registered=["browsergym.miniwob"],
            task_ids_by_env={"browsergym.miniwob": {"train": ["m1"]}},
        )
        try:
            r = client.get("/envs/browsergym.miniwob", headers=_h("a"))
            assert r.status_code == 200
            assert r.json()["available"] is True
        finally:
            patcher.stop()

    def test_get_one_env_disallowed_returns_404(self):
        # Allow-list miss = 404 (not 403), matches the bare-list filter.
        client, patcher = self._client_with_registry(
            registered=["x", "y"],
            task_ids_by_env={"x": {"train": ["t"]}, "y": {"train": ["t"]}},
            allowed={"x"},
        )
        try:
            assert client.get("/envs/x", headers=_h("a")).status_code == 200
            assert client.get("/envs/y", headers=_h("a")).status_code == 404
        finally:
            patcher.stop()

    def test_get_one_env_blocked_returns_200_unavailable(self):
        # Blocked envs are reachable through GET /envs/{env_id} but
        # surface as available=false so the client can self-diagnose.
        client, patcher = self._client_with_registry(
            registered=["x"],
            task_ids_by_env={"x": {"train": ["t"]}},
            blocked={"x"},
        )
        try:
            r = client.get("/envs/x", headers=_h("a"))
            assert r.status_code == 200
            assert r.json()["available"] is False
            assert "blocked" in r.json()["error"].lower()
        finally:
            patcher.stop()

    def test_get_env_tasks_returns_splits_and_metadata(self):
        client, patcher = self._client_with_registry(
            registered=["androidworld"],
            task_ids_by_env={
                "androidworld": {"train": ["t1", "t2"], "eval": ["t3"]},
            },
        )
        try:
            r = client.get("/envs/androidworld/tasks", headers=_h("a"))
            assert r.status_code == 200
            body = r.json()
            assert body["splits"] == {"train": ["t1", "t2"], "eval": ["t3"]}
            # task_metadata patched to return None → empty metadata map.
            assert body["metadata"] == {}
        finally:
            patcher.stop()

    def test_get_env_tasks_blocked_returns_501(self):
        client, patcher = self._client_with_registry(
            registered=["lite.demo"],
            task_ids_by_env={"lite.demo": {"train": ["t"]}},
            blocked={"lite.demo"},
        )
        try:
            r = client.get("/envs/lite.demo/tasks", headers=_h("a"))
            assert r.status_code == 501
            detail = r.json()["detail"]
            assert "hard-blocked" in detail
            assert "direct mode" in detail
        finally:
            patcher.stop()

    def test_get_env_tasks_reports_server_config_separately(
        self,
        tmp_path,
        monkeypatch,
    ):
        from lite.gym.utils import config as env_config

        config_dir = tmp_path
        (config_dir / "configs").mkdir()
        (config_dir / "configs" / "default.yaml").write_text(
            """
env_var_prefix: MYENV
env_kwargs:
  max_steps: 9
make_kwargs:
  cursor: true
  step_timeout: 99.0
server_kwargs:
  memory_limit: "4GB"
  api_key: "server-secret"
  nested:
    access_token: "nested-secret"
    public: "ok"
""".lstrip()
        )
        env_config.load.cache_clear()
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("MYENV_API_KEY", "env-secret")
        # `{prefix}_EVAL_API_KEY_VAR` is judge-credential indirection: its
        # value NAMES the var actually holding the judge API key (the
        # convention webgym / online_mind2web / webharbor_webvoyager share).
        monkeypatch.setenv("MYENV_EVAL_API_KEY_VAR", "CUSTOM_JUDGE_KEY")
        monkeypatch.setenv("CUSTOM_JUDGE_KEY", "judge-secret")
        state = State(
            admission=_make_test_admission(max_live_envs=100),
            idle_ttl_sec=123.0,
            allowed_env_ids={"x"},
            blocked_env_ids=frozenset({"blocked"}),
            reset_jitter_sec=2.0,
            boot_jitter_sec=1.0,
            reset_concurrency=7,
            warm_singleton_enabled=True,
        )
        app = make_app(state, token="strict_T", admin_token="admin_T", port=30999)
        patches = [
            patch("lite.gym.remote.server.gym.registry.task_ids", lambda env_id: {"eval": ["t1"]}),
            patch("lite.gym.remote.server.gym.registry.task_metadata", lambda env_id, tid: None),
            patch(
                "lite.gym.remote.server.gym.registry.task_kwargs",
                lambda env_id, tid: {"step_timeout": 7.0},
            ),
            patch(
                "lite.gym.remote.server.gym.registry.env_make_kwargs",
                lambda env_id: {"cursor": False},
            ),
            patch(
                "lite.gym.remote.server.gym.registry.env_supported_kwargs",
                lambda env_id: ["seed"],
            ),
            patch(
                "lite.gym.remote.server._env_config_dir_for_env_id",
                lambda env_id: config_dir,
            ),
        ]
        for p in patches:
            p.start()
        try:
            r = TestClient(app).get("/envs/x/tasks", headers=_h("strict_T"))
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["kwargs"] == {"t1": {"step_timeout": 7.0}}
            assert body["env_make_kwargs"] == {"cursor": False}
            assert body["env_supported_kwargs"] == ["seed"]

            server_config = body["server_config"]
            assert server_config["server_kwargs"] == {
                "memory_limit": "4GB",
                "api_key": "<redacted>",
                "nested": {"access_token": "<redacted>", "public": "ok"},
            }
            assert "max_steps" not in server_config["server_kwargs"]
            assert "cursor" not in server_config["server_kwargs"]
            assert "server-secret" not in str(server_config)
            assert "nested-secret" not in str(server_config)
            assert "sk-test" not in str(server_config)
            assert "env-secret" not in str(server_config)

            assert server_config["provenance"]["available"] is True
            assert server_config["provenance"]["env_var_prefix"] == "MYENV"
            assert server_config["provenance"]["config_env_var"] == "MYENV_CONFIG"
            assert server_config["provenance"]["config_path"] == str(
                (config_dir / "configs" / "default.yaml").resolve()
            )
            assert server_config["server_process_env"]["OPENAI_API_KEY"] == {
                "set": True,
                "available": True,
            }
            assert server_config["server_process_env"]["MYENV_API_KEY"] == {
                "set": True,
                "available": True,
            }
            # The resolved indirection target shows up by its OWN name, not
            # the fixed indirection var's name.
            assert server_config["server_process_env"]["CUSTOM_JUDGE_KEY"] == {
                "set": True,
                "available": True,
            }
            assert "judge-secret" not in str(server_config)
            assert server_config["env_server"] == {
                "port": 30999,
                "auth_mode": "strict",
                "admin_auth_mode": "strict",
                "allowed_env_ids": ["x"],
                "blocked_env_ids": ["blocked"],
                "idle_ttl_sec": 123.0,
                "boot_jitter_sec": 1.0,
                "reset_jitter_sec": 2.0,
                "reset_concurrency": 7,
                "warm_singleton_enabled": True,
            }
        finally:
            for p in reversed(patches):
                p.stop()
            env_config.load.cache_clear()

    def test_get_env_tasks_404_when_not_registered(self):
        client, patcher = self._client_with_registry(
            registered=["x"],
            task_ids_by_env={"x": {"train": ["t"]}},
            allowed={"x"},  # only x is visible
        )
        try:
            r = client.get("/envs/missing/tasks", headers=_h("a"))
            assert r.status_code == 404
        finally:
            patcher.stop()

    def test_get_envs_requires_bearer_token_in_strict_mode(self):
        # Catalog routes are bearer-gated; in strict mode the wrong
        # token gets 401 like every other endpoint.
        state = State(
            admission=_make_test_admission(max_live_envs=100),
            idle_ttl_sec=3600.0,
        )
        app = make_app(state, token="strict_T")
        with patch(
            "lite.gym.remote.server.gym.registry.registered_env_ids",
            lambda: ["x"],
        ):
            client = TestClient(app)
            assert client.get("/envs", headers=_h("wrong")).status_code == 401
            assert client.get("/envs", headers=_h("strict_T")).status_code == 200


# ---------------------------------------------------------------------------
# DELETE /instances/{id} — response shape (ok + parked)
# ---------------------------------------------------------------------------


class TestGetSingleInstance:
    """``GET /instances/{id}`` mirrors one entry of the list endpoint
    — operators reach for it when they have an id from a log line and
    want its env_key / session_id / age without paginating."""

    def test_returns_session_summary(self):
        client, patcher = _make_client()
        try:
            env_kwargs = {"max_steps": 3, "cursor": False}
            id_ = _create_env(client, "alice", "x", "s1", env_kwargs=env_kwargs)
            r = client.get(f"/instances/{id_}", headers=_h("alice"))
            assert r.status_code == 200
            body = r.json()
            assert body["id"] == id_
            assert body["env_id"] == "x"
            assert body["session_id"] == "s1"
            assert body["env_key"] == "x@t"
            assert body["env_kwargs"] == env_kwargs
            assert "token_hash" not in body["env_kwargs"]
            assert "server_port" not in body["env_kwargs"]
            assert "session_id" not in body["env_kwargs"]
        finally:
            patcher.stop()

    def test_404_when_unknown_id(self):
        client, patcher = _make_client()
        try:
            r = client.get(
                "/instances/00000000000000000000000000000000",
                headers=_h("alice"),
            )
            assert r.status_code == 404
        finally:
            patcher.stop()

    def test_403_when_owned_by_another_token(self):
        client, patcher = _make_client(token=None)  # passthrough → per-token isolation
        try:
            id_ = _create_env(client, "alice", "x", "s1")
            # bob can't see alice's instance.
            r = client.get(f"/instances/{id_}", headers=_h("bob"))
            assert r.status_code == 403
            # TYPED, not FastAPI's bare {"detail": …}: an untyped body falls into
            # ``is_retryable``'s unknown→True default and gets re-attempted.
            from lite.gym.errors import is_retryable, lite_error_from_payload

            body = r.json()
            assert body["error_type"] == "EnvAccessDenied"
            assert "detail" in body  # human-readable key kept alongside
            assert is_retryable(lite_error_from_payload(body)) is False
        finally:
            patcher.stop()


class TestDeleteInstanceResponseShape:
    """DELETE /instances/{id} returns {ok: True}. R6 retired the
    legacy ``parked: bool`` field along with park-on-DELETE."""

    def test_destroy_path(self):
        client, patcher = _make_client()
        try:
            r = client.post(
                "/instances",
                headers=_h("alice"),
                json={"env_key": "x@t", "env_kwargs": {}, "session_id": "s"},
            )
            id_ = r.json()["id"]
            r = client.delete(f"/instances/{id_}", headers=_h("alice"))
            assert r.status_code == 200
            assert r.json() == {"ok": True}
        finally:
            patcher.stop()

    def test_idempotent_unknown_id(self):
        client, patcher = _make_client()
        try:
            r = client.delete(
                "/instances/00000000000000000000000000000000",
                headers=_h("alice"),
            )
            assert r.status_code == 200
            assert r.json() == {"ok": True}
        finally:
            patcher.stop()


class TestInFlightGuard:
    """section 7.G1: the idle reaper + bulk-DELETE never touch an env whose step/
    reset handler is currently awaited inside it."""

    def test_collect_stale_skips_in_flight(self):
        from types import SimpleNamespace

        from lite.gym.remote.server import _collect_stale

        now = 10_000.0
        old = now - 999.0  # far past any TTL
        stale_s = SimpleNamespace(token_hash="t", in_flight=0, last_active_at=old)
        busy_s = SimpleNamespace(token_hash="t", in_flight=1, last_active_at=old)
        legacy_unowned_s = SimpleNamespace(token_hash=None, in_flight=0, last_active_at=old)
        state = SimpleNamespace(
            idle_ttl_sec=60.0,
            envs={"stale": stale_s, "busy": busy_s, "legacy": legacy_unowned_s},
        )
        reaped = {id_ for id_, _, _ in _collect_stale(state, now)}
        assert reaped == {"stale", "legacy"}, (
            "in-flight entries must never be TTL-reaped; unowned legacy entries are not exempt"
        )

    def test_step_handler_tracks_in_flight(self):
        """The guard is live end-to-end: while env.step() is awaited, the
        session's in_flight is >0; it returns to 0 afterwards."""
        state = State(
            admission=_make_test_admission(max_live_envs=10),
            idle_ttl_sec=3600.0,
        )
        app = make_app(state, token="zzh-dev")
        seen: list[int] = []

        class _SpyEnv(_NoopEnv):
            async def step(self2, actions):
                (sess,) = state.envs.values()
                seen.append(sess.in_flight)
                return await super().step(actions)

        patcher = patch(
            "lite.gym.remote.server.gym.make",
            lambda env_key, **kw: _SpyEnv(),
        )
        patcher.start()
        try:
            client = TestClient(app)
            iid = _create_env(client, "zzh-dev", "lite.demo", "s1")
            assert client.post(f"/instances/{iid}/reset", headers=_h("zzh-dev")).status_code == 200
            r = client.post(f"/instances/{iid}/step", headers=_h("zzh-dev"), json={"actions": []})
            assert r.status_code == 200
            assert seen == [1], f"in_flight not held during the awaited step: {seen}"
            (sess,) = state.envs.values()
            assert sess.in_flight == 0, "in_flight must return to 0 after the call"
        finally:
            patcher.stop()

    def test_bulk_delete_skips_in_flight_and_reports_it(self):
        """section 7.G1 end-to-end on the bulk route: DELETE /instances must skip an
        env with a step/reset awaited inside it (reported in
        ``skipped_in_flight``, still tracked) while closing the idle one."""
        state = State(
            admission=_make_test_admission(max_live_envs=10),
            idle_ttl_sec=3600.0,
        )
        app = make_app(state, token="zzh-dev")
        patcher = patch(
            "lite.gym.remote.server.gym.make",
            lambda env_key, **kw: _NoopEnv(),
        )
        patcher.start()
        try:
            client = TestClient(app)
            busy_id = _create_env(client, "zzh-dev", "x", "s1")
            idle_id = _create_env(client, "zzh-dev", "x", "s1")
            state.envs[busy_id].in_flight = 1  # simulate an awaited step/reset
            # force=true satisfies the wide-net guard (unpinned scope).
            r = client.delete("/instances", headers=_h("zzh-dev"), params={"force": True})
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["closed"] == [idle_id]
            assert body["skipped_in_flight"] == [busy_id]
            # The busy env is still tracked (not torn down under a live op);
            # the idle one is gone.
            assert busy_id in state.envs
            assert idle_id not in state.envs
        finally:
            patcher.stop()
