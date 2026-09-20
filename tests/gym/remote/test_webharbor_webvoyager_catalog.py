"""Hermetic WebHarbor WebVoyager remote catalog coverage."""

from __future__ import annotations

from fastapi.testclient import TestClient
from gym.remote.conftest import bearer_header as _h
from gym.remote.conftest import make_test_admission

from lite.gym.remote.server import State, make_app

_ENV_ID = "webharbor.webvoyager"
_TASK_ID = "allrecipes.0"
_TOKEN = "webharbor-webvoyager-catalog-token"


def test_webharbor_webvoyager_tasks_catalog_serves_metadata_and_kwargs(monkeypatch):
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_URL", raising=False)
    monkeypatch.delenv("CUA_LITE_ENV_SERVER_TOKEN", raising=False)

    state = State(
        admission=make_test_admission(max_live_envs=4),
        idle_ttl_sec=3600.0,
        allowed_env_ids={_ENV_ID},
    )
    app = make_app(state, token=_TOKEN)
    client = TestClient(app)
    try:
        response = client.get(f"/envs/{_ENV_ID}/tasks", headers=_h(_TOKEN))
        assert response.status_code == 200, response.text
        body = response.json()

        assert _TASK_ID in body["splits"]["eval"]
        metadata = body["metadata"][_TASK_ID]
        assert metadata["metadata_kind"] == "cua"
        assert metadata["dims"] == ["browser", "use"]
        assert metadata["others"]["source"] == "webharbor"
        assert metadata["others"]["max_steps"] == 15
        assert body["env_make_kwargs"] == {"cursor": True}
        assert body["env_supported_kwargs"] == [
            "eval_config",
            "extra_tools",
            "fix_box_color",
            "include_page_context_text",
            "max_steps",
            "post_action_delay",
            "step_timeout",
            "use_som",
            "valid_actions",
            "viewport",
        ]
    finally:
        client.close()
