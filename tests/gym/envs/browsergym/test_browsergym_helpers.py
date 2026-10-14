"""Pure BrowserGym helper, screenshot, and goal/config extraction tests."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("browsergym.core", reason="browsergym not installed")

from lite.gym.envs.browsergym.main import (
    _encode_screenshot,
    _escape,
    _extract_goal_images_b64,
    _extract_instruction,
)
from tests.gym.envs.browsergym._support import (
    _make_fake,
    _png_bytes,
)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"action_subsets": ["bid"], "skip_dom_extraction": True},
        {"action_subsets": ["miniwob_all"], "skip_dom_extraction": True},
        {"use_som": True, "skip_dom_extraction": True},
        {"action_subsets": ["webarena"], "skip_dom_extraction": True},
        {"action_subsets": ["visualwebarena"], "skip_dom_extraction": True},
    ],
)
def test_skip_dom_extraction_rejects_dom_or_bid_configs(kwargs: dict[str, Any]):
    with pytest.raises(ValueError, match="skip_dom_extraction"):
        _make_fake(**kwargs)


def test_visualwebarena_mode_configs_lock_goal_mixed_and_som_surfaces():
    yaml = pytest.importorskip("yaml")
    root = Path(__file__).resolve().parents[4]
    config_paths = sorted(
        (root / "scripts" / "configs").glob("*/default/browsergym.visualwebarena/*.yaml")
    )
    checked = {path.name for path in config_paths}
    assert {"goal_image.yaml", "mixed.yaml", "som.yaml"} <= checked

    for path in config_paths:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        env_kwargs = data["env_kwargs"]
        extra_tools = set(env_kwargs.get("extra_tools") or [])
        rel = path.relative_to(root)
        agent_family = path.relative_to(root / "scripts" / "configs").parts[0]

        assert data["env_id"] == "browsergym.visualwebarena", rel
        assert {"response", "terminate"} <= extra_tools, rel
        if agent_family == "claude":
            assert data["agent_id"] == "claude", rel
        else:
            assert data["agent_id"] == "visualwebarena.goal_image", rel
            assert data["agent_kwargs"]["adapter_key"], rel

        if path.name == "goal_image.yaml":
            assert env_kwargs["use_screenshot"] is True, rel
            assert env_kwargs["use_ax_tree"] is False, rel
            assert env_kwargs["skip_dom_extraction"] is True, rel
            assert {"goto", "back"} <= extra_tools, rel
            assert "upload_file" not in extra_tools, rel
        elif path.name == "mixed.yaml":
            assert env_kwargs["use_screenshot"] is False, rel
            assert env_kwargs["use_ax_tree"] is True, rel
            assert env_kwargs["valid_actions"] == [], rel
            assert env_kwargs["action_subsets"] == ["visualwebarena"], rel
            assert {"click", "fill", "upload_file"} <= extra_tools, rel
        elif path.name == "som.yaml":
            assert env_kwargs["use_screenshot"] is True, rel
            assert env_kwargs["use_ax_tree"] is False, rel
            assert env_kwargs["use_som"] is True, rel
            assert env_kwargs["skip_dom_extraction"] is False, rel
            assert env_kwargs["valid_actions"] == [], rel
            assert env_kwargs["action_subsets"] == ["visualwebarena"], rel
            assert {"click", "fill", "upload_file"} <= extra_tools, rel
# ---------------------------------------------------------------------------
# String / quoting helpers
# ---------------------------------------------------------------------------

class TestEscapeHelper:
    """``_escape`` makes a string safe to embed inside a double-quoted Python literal."""

    def test_quotes(self):
        assert _escape('say "hi"') == 'say \\"hi\\"'

    def test_newlines(self):
        assert _escape("line1\nline2") == "line1\\nline2"

    def test_carriage_return(self):
        assert _escape("a\rb") == "a\\rb"

    def test_backslash(self):
        assert _escape("a\\b") == "a\\\\b"

    def test_combined(self):
        assert _escape('a\\b"c\nd') == 'a\\\\b\\"c\\nd'

    def test_empty(self):
        assert _escape("") == ""


# ---------------------------------------------------------------------------
# Screenshot encoding
# ---------------------------------------------------------------------------

class TestEncodeScreenshot:
    """``_encode_screenshot`` accepts numpy arrays / bytes; raises otherwise.

    It returns RAW PNG bytes, never base64 — ``LiteEnvObservation.image`` and
    ``LiteToolResult.images`` carry bytes in memory, and base64 is applied only
    at the model-adapter boundary (see ``lite/gym/types.py``).
    """

    def test_numpy_array(self):
        import numpy as np
        arr = np.full((8, 8, 3), 128, dtype=np.uint8)
        raw = _encode_screenshot(arr)
        assert isinstance(raw, bytes)
        assert raw[:4] == b"\x89PNG"

    def test_bytes_passthrough(self):
        png = _png_bytes()
        raw = _encode_screenshot(png)
        # Already-PNG bytes pass through unchanged (no re-encode, no base64).
        assert raw == png

    def test_none_raises(self):
        with pytest.raises(RuntimeError, match="None screenshot"):
            _encode_screenshot(None)

    def test_unexpected_type_raises(self):
        with pytest.raises(RuntimeError, match="Unexpected screenshot type"):
            _encode_screenshot(12345)


# ---------------------------------------------------------------------------
# Goal extraction
# ---------------------------------------------------------------------------

class TestExtractInstruction:

    def test_string_goal(self):
        assert _extract_instruction({"goal": "find the cheapest hat"}) == "find the cheapest hat"

    def test_goal_object_text_only(self):
        obs = {"goal": "", "goal_object": [
            {"type": "text", "text": "Part 1"},
            {"type": "text", "text": "Part 2"},
        ]}
        assert _extract_instruction(obs) == "Part 1\nPart 2"

    def test_goal_object_with_image(self):
        # Multimodal goals keep the human instruction and drop BrowserGym's
        # image carrier text, because the image itself is transported in
        # metadata and rendered by the VWA goal-image agent.
        obs = {"goal_object": [
            {"type": "text", "text": "Find this product:"},
            {"type": "text", "text": "Input image 1/1 below"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,XXX"}},
        ]}
        assert _extract_instruction(obs) == "Find this product:"

    def test_goal_object_with_image_ignores_legacy_goal_warning(self):
        obs = {
            "goal": (
                "Find this product:\n"
                "Input image 1/1 below (local path: PosixPath('/tmp/input.png'), "
                "url: 'http://localhost:8677/input.png')\n"
                "WARNING: This goal cannot be converted to a text-only goal format. "
                "Use the new goal format instead (\"goal_object\" field). Any agent "
                "reading this should abort immediately."
            ),
            "goal_object": [
                {"type": "text", "text": "Find this product:"},
                {
                    "type": "text",
                    "text": (
                        "Input image 1/1 below (local path: PosixPath('/tmp/input.png'), "
                        "url: 'http://localhost:8677/input.png')"
                    ),
                },
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,XXX"}},
            ],
        }
        assert _extract_instruction(obs) == "Find this product:"

    def test_goal_object_image_only_does_not_fall_back_to_legacy_warning(self):
        obs = {
            "goal": (
                "Input image 1/1 below (local path: PosixPath('/tmp/input.png'), "
                "url: 'http://localhost:8677/input.png')\n"
                "WARNING: This goal cannot be converted to a text-only goal format. "
                "Use the new goal format instead (\"goal_object\" field). Any agent "
                "reading this should abort immediately."
            ),
            "goal_object": [
                {
                    "type": "text",
                    "text": (
                        "Input image 1/1 below (local path: PosixPath('/tmp/input.png'), "
                        "url: 'http://localhost:8677/input.png')"
                    ),
                },
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,XXX"}},
            ],
        }
        assert _extract_instruction(obs) == ""

    def test_empty(self):
        assert _extract_instruction({}) == ""

    def test_string_goal_takes_precedence_for_text_only_goal_object(self):
        obs = {"goal": "primary", "goal_object": [{"type": "text", "text": "ignored"}]}
        assert _extract_instruction(obs) == "primary"


class TestExtractGoalImage:
    """``_extract_goal_images_b64`` pulls ALL data-URI images from a VWA
    goal_object, keeping the base64 body the reset metadata carries."""

    def test_single_image(self):
        b64 = base64.b64encode(_png_bytes()).decode()
        obs = {"goal_object": [
            {"type": "text", "text": "find this:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}
        assert _extract_goal_images_b64(obs) == [b64]

    def test_no_image(self):
        obs = {"goal_object": [{"type": "text", "text": "hi"}]}
        assert _extract_goal_images_b64(obs) == []

    def test_no_goal_object(self):
        assert _extract_goal_images_b64({}) == []

    def test_goal_object_none(self):
        assert _extract_goal_images_b64({"goal_object": None}) == []

    def test_all_of_multiple_images(self):
        # A multi-image goal transports every image, in goal_object order.
        first = base64.b64encode(b"FIRST").decode()
        second = base64.b64encode(b"SECOND").decode()
        obs = {"goal_object": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{first}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{second}"}},
        ]}
        assert _extract_goal_images_b64(obs) == [first, second]

    def test_skips_non_data_uri(self):
        good = base64.b64encode(b"GOOD").decode()
        obs = {"goal_object": [
            {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{good}"}},
        ]}
        assert _extract_goal_images_b64(obs) == [good]

    def test_skips_data_uri_without_comma(self):
        # Pathological data URI with no comma: skipped, not crashed.
        obs = {"goal_object": [
            {"type": "image_url", "image_url": {"url": "data:image/pngBASE64STUFF"}},
        ]}
        assert _extract_goal_images_b64(obs) == []

    def test_skips_non_dict_entries(self):
        xx = base64.b64encode(b"XX").decode()
        # goal_object can contain stray strings; must not crash.
        obs = {
            "goal_object": [
                "stray",
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{xx}"},
                },
            ]
        }
        assert _extract_goal_images_b64(obs) == [xx]
