"""Malformed grounding-point coordinates must become paired model feedback.

These fixtures construct the env classes directly and seed the screenshot bytes
instead of calling ``reset()``, so they do not need ScreenSpot-Pro or OSWorld-G
data files to be present.

This file stays shallow because it asserts the shared pure-grounding feedback
contract across ``screenspot_pro`` and ``osworld_g``; each test name keeps the
env-specific half explicit.

Run:
    uv run pytest tests/gym/envs/test_grounding_malformed_coordinates.py -q
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lite.core.tools import make_tool_call
from lite.core.tools.results import project_tool_result_text

NO_BACKEND_LEAK_STRINGS = (
    "xdotool",
    "pyautogui",
    "pynput",
    "Playwright",
    "unsupported operand",
    "TypeError:",
    "ValueError:",
    "RuntimeError:",
    "NoneType",
)


def _assert_no_backend_leaks(text: str | None) -> None:
    assert text is not None
    for leak in NO_BACKEND_LEAK_STRINGS:
        assert leak not in text


MALFORMED_POINT_ARGUMENTS = [
    {"coordinate": "abc"},
    {"coordinate": [None, None]},
    {"coordinate": [float("nan"), 1]},
    {"coordinate": [float("inf"), 1]},
    {"coordinate": [1, float("-inf")]},
    {},
    {"coordinate": None},
    {"coordinate": []},
    {"coordinate": [1]},
]


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", MALFORMED_POINT_ARGUMENTS)
async def test_screenspot_pro_malformed_point_coordinate_is_paired_feedback(
    arguments: dict[str, Any],
) -> None:
    from lite.gym.envs.screenspot_pro.main import ScreenSpotProEnv

    env = ScreenSpotProEnv(
        annotation={
            "img_filename": "unused.png",
            "instruction": "click the target",
            "bbox": [10, 10, 20, 20],
            "img_size": [100, 100],
        },
        images_dir=Path("/tmp"),
    )
    env._screenshot = b"shot"

    result = await env.step([
        make_tool_call("point", arguments, call_id="bad"),
    ])

    assert result.terminated is True
    assert result.truncated is False
    assert result.results[0].tool_call_id == "bad"
    assert result.results[0].text is None
    assert result.results[0].images[-1] == b"shot"
    assert "invalid arguments for point:" in (result.results[0].error or "")
    assert result.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_screenspot_pro_malformed_point_python_detail_is_not_model_visible() -> None:
    from lite.gym.envs.screenspot_pro.main import ScreenSpotProEnv

    env = ScreenSpotProEnv(
        annotation={
            "img_filename": "unused.png",
            "instruction": "click the target",
            "bbox": [10, 10, 20, 20],
            "img_size": [100, 100],
        },
        images_dir=Path("/tmp"),
    )
    env._screenshot = b"shot"

    result = await env.step([
        make_tool_call("point", {"coordinate": [None, None]}, call_id="bad"),
    ])

    visible_error = result.results[0].error
    assert visible_error == (
        "invalid arguments for point: coordinate values must be finite numbers"
    )
    _assert_no_backend_leaks(visible_error)
    _assert_no_backend_leaks(
        project_tool_result_text(result.results[0].text, visible_error)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", MALFORMED_POINT_ARGUMENTS)
async def test_screenspot_pro_malformed_point_without_call_id_does_not_poison_reward(
    arguments: dict[str, Any],
) -> None:
    from lite.gym.envs.screenspot_pro.main import ScreenSpotProEnv

    env = ScreenSpotProEnv(
        annotation={
            "img_filename": "unused.png",
            "instruction": "click the target",
            "bbox": [10, 10, 20, 20],
            "img_size": [100, 100],
        },
        images_dir=Path("/tmp"),
    )
    env._screenshot = b"shot"

    result = await env.step([
        make_tool_call("point", {"coordinate": [150, 150]}, call_id="good"),
        make_tool_call("point", arguments),
    ])

    assert result.reward == 1.0
    assert result.terminated is True
    assert result.truncated is False


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", MALFORMED_POINT_ARGUMENTS)
async def test_screenspot_pro_wrong_action_wrapper_does_not_poison_valid_sibling_reward(
    arguments: dict[str, Any],
) -> None:
    from lite.gym.envs.screenspot_pro.main import ScreenSpotProEnv

    env = ScreenSpotProEnv(
        annotation={
            "img_filename": "unused.png",
            "instruction": "click the target",
            "bbox": [10, 10, 20, 20],
            "img_size": [100, 100],
        },
        images_dir=Path("/tmp"),
    )
    env._screenshot = b"shot"

    result = await env.step([
        make_tool_call("point", {"coordinate": [150, 150]}, call_id="good"),
        make_tool_call(
            "computer",
            {"actions": [{"action": "point", **arguments}]},
            call_id="bad_batch",
        ),
    ])

    assert result.reward == 1.0
    assert result.terminated is True
    assert result.truncated is False
    assert len(result.results) == 1
    by_id = {res.tool_call_id: res for res in result.results}
    assert set(by_id) == {"bad_batch"}
    assert by_id["bad_batch"].text is None
    assert by_id["bad_batch"].images[-1] == b"shot"
    assert by_id["bad_batch"].error == (
        "invalid action: computer; choose an available action for this task"
    )
    assert by_id["bad_batch"].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_screenspot_pro_direct_valid_actions_empty_rejects_point() -> None:
    from lite.gym.envs.screenspot_pro.main import ScreenSpotProEnv

    env = ScreenSpotProEnv(
        annotation={
            "img_filename": "unused.png",
            "instruction": "click the target",
            "bbox": [10, 10, 20, 20],
            "img_size": [100, 100],
        },
        images_dir=Path("/tmp"),
        valid_actions=[],
    )
    env._screenshot = b"shot"

    result = await env.step([
        make_tool_call("point", {"coordinate": [150, 150]}, call_id="blocked"),
    ])

    assert result.reward == 0.0
    assert result.terminated is True
    assert result.results[0].tool_call_id == "blocked"
    assert result.results[0].images[-1] == b"shot"
    assert result.results[0].error == (
        "invalid action: point; choose an available action for this task"
    )
    assert result.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_screenspot_pro_bbox_tool_is_rejected_with_current_image() -> None:
    from lite.gym.envs.screenspot_pro.main import ScreenSpotProEnv

    env = ScreenSpotProEnv(
        annotation={
            "img_filename": "unused.png",
            "instruction": "click the target",
            "bbox": [10, 10, 20, 20],
            "img_size": [100, 100],
        },
        images_dir=Path("/tmp"),
    )
    env._screenshot = b"shot"

    result = await env.step([
        make_tool_call("bbox", {"coordinate": [None, None, None, None]}, call_id="bbox"),
    ])

    assert result.reward == 0.0
    assert result.terminated is True
    assert result.results[0].tool_call_id == "bbox"
    assert result.results[0].images[-1] == b"shot"
    assert result.results[0].text is None
    assert result.results[0].error == (
        "invalid action: bbox; choose an available action for this task"
    )
    assert result.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_screenspot_pro_unknown_tool_is_error_only_feedback() -> None:
    from lite.gym.envs.screenspot_pro.main import ScreenSpotProEnv

    env = ScreenSpotProEnv(
        annotation={
            "img_filename": "unused.png",
            "instruction": "click the target",
            "bbox": [10, 10, 20, 20],
            "img_size": [100, 100],
        },
        images_dir=Path("/tmp"),
    )
    env._screenshot = b"shot"

    result = await env.step([
        make_tool_call("frobnicate", {}, call_id="foo"),
    ])

    assert result.terminated is True
    assert result.truncated is False
    assert result.results[0].tool_call_id == "foo"
    assert result.results[0].images == []
    assert result.results[0].text is None
    assert result.results[0].error == "unknown tool: frobnicate"
    assert result.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_screenspot_pro_inactive_response_returns_error_only_feedback() -> None:
    from lite.gym.envs.screenspot_pro.main import ScreenSpotProEnv

    env = ScreenSpotProEnv(
        annotation={
            "img_filename": "unused.png",
            "instruction": "click the target",
            "bbox": [10, 10, 20, 20],
            "img_size": [100, 100],
        },
        images_dir=Path("/tmp"),
    )
    env._screenshot = b"shot"

    result = await env.step([
        make_tool_call("response", {"text": "Done."}, call_id="resp"),
    ])

    assert result.terminated is True
    assert result.truncated is False
    assert result.results[0].tool_call_id == "resp"
    assert result.results[0].images == []
    assert result.results[0].text is None
    assert result.results[0].error == "response is not available in this task."
    assert result.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_screenspot_pro_multiple_points_score_the_first_valid_answer() -> None:
    from lite.gym.envs.screenspot_pro.main import ScreenSpotProEnv

    annotation = {
        "img_filename": "unused.png",
        "instruction": "click the target",
        "bbox": [10, 10, 20, 20],
        "img_size": [100, 100],
    }
    env = ScreenSpotProEnv(annotation=annotation, images_dir=Path("/tmp"))
    env._screenshot = b"shot"

    result = await env.step([
        make_tool_call("point", {"coordinate": [150, 150]}, call_id="inside"),
        make_tool_call("point", {"coordinate": [900, 900]}, call_id="outside"),
    ])

    assert result.reward == 1.0
    assert result.terminated is True
    assert result.truncated is False
    assert result.info["annotation"] == annotation
    assert result.info["executed_actions"] == [
        {"call": "point", "args": {"x": 15.0, "y": 15.0}},
        {"call": "point", "args": {"x": 90.0, "y": 90.0}},
    ]
    assert result.results == []


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", MALFORMED_POINT_ARGUMENTS)
async def test_osworld_g_malformed_point_coordinate_is_paired_feedback(
    arguments: dict[str, Any],
) -> None:
    from lite.gym.envs.osworld_g.main import OSWorldGEnv

    annotation = {
        "instruction": "click the target",
        "image_path": "unused.png",
        "image_size": [100, 100],
        "box_type": "bbox",
        "box_coordinates": [10, 10, 20, 20],
        "GUI_types": [],
    }
    env = OSWorldGEnv(
        annotation_original=annotation,
        annotation_refined=annotation,
        images_dir=Path("/tmp"),
        task_id="fixture",
    )
    env._screenshot = b"shot"

    result = await env.step([
        make_tool_call("point", arguments, call_id="bad"),
    ])

    assert result.terminated is True
    assert result.truncated is False
    assert result.results[0].tool_call_id == "bad"
    assert result.results[0].text is None
    assert result.results[0].images[-1] == b"shot"
    assert "invalid arguments for point:" in (result.results[0].error or "")
    assert result.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_osworld_g_malformed_point_python_detail_is_not_model_visible() -> None:
    from lite.gym.envs.osworld_g.main import OSWorldGEnv

    annotation = {
        "instruction": "click the target",
        "image_path": "unused.png",
        "image_size": [100, 100],
        "box_type": "bbox",
        "box_coordinates": [10, 10, 20, 20],
        "GUI_types": [],
    }
    env = OSWorldGEnv(
        annotation_original=annotation,
        annotation_refined=annotation,
        images_dir=Path("/tmp"),
        task_id="fixture",
    )
    env._screenshot = b"shot"

    result = await env.step([
        make_tool_call("point", {"coordinate": [None, None]}, call_id="bad"),
    ])

    visible_error = result.results[0].error
    assert visible_error == (
        "invalid arguments for point: coordinate values must be finite numbers"
    )
    _assert_no_backend_leaks(visible_error)
    _assert_no_backend_leaks(
        project_tool_result_text(result.results[0].text, visible_error)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", MALFORMED_POINT_ARGUMENTS)
async def test_osworld_g_malformed_point_without_call_id_does_not_poison_reward(
    arguments: dict[str, Any],
) -> None:
    from lite.gym.envs.osworld_g.main import OSWorldGEnv

    annotation = {
        "instruction": "click the target",
        "image_path": "unused.png",
        "image_size": [100, 100],
        "box_type": "bbox",
        "box_coordinates": [10, 10, 20, 20],
        "GUI_types": [],
    }
    env = OSWorldGEnv(
        annotation_original=annotation,
        annotation_refined=annotation,
        images_dir=Path("/tmp"),
        task_id="fixture",
    )
    env._screenshot = b"shot"

    result = await env.step([
        make_tool_call("point", {"coordinate": [150, 150]}, call_id="good"),
        make_tool_call("point", arguments),
    ])

    assert result.reward == 1.0
    assert result.terminated is True
    assert result.truncated is False


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", MALFORMED_POINT_ARGUMENTS)
async def test_osworld_g_wrong_action_wrapper_does_not_poison_valid_sibling_reward(
    arguments: dict[str, Any],
) -> None:
    from lite.gym.envs.osworld_g.main import OSWorldGEnv

    annotation = {
        "instruction": "click the target",
        "image_path": "unused.png",
        "image_size": [100, 100],
        "box_type": "bbox",
        "box_coordinates": [10, 10, 20, 20],
        "GUI_types": [],
    }
    env = OSWorldGEnv(
        annotation_original=annotation,
        annotation_refined=annotation,
        images_dir=Path("/tmp"),
        task_id="fixture",
    )
    env._screenshot = b"shot"

    result = await env.step([
        make_tool_call("point", {"coordinate": [150, 150]}, call_id="good"),
        make_tool_call(
            "computer",
            {"actions": [{"action": "point", **arguments}]},
            call_id="bad_batch",
        ),
    ])

    assert result.reward == 1.0
    assert result.terminated is True
    assert result.truncated is False
    assert len(result.results) == 1
    by_id = {res.tool_call_id: res for res in result.results}
    assert set(by_id) == {"bad_batch"}
    assert by_id["bad_batch"].text is None
    assert by_id["bad_batch"].images[-1] == b"shot"
    assert by_id["bad_batch"].error == (
        "invalid action: computer; choose an available action for this task"
    )
    assert by_id["bad_batch"].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_osworld_g_direct_valid_actions_empty_rejects_point() -> None:
    from lite.gym.envs.osworld_g.main import OSWorldGEnv

    annotation = {
        "instruction": "click the target",
        "image_path": "unused.png",
        "image_size": [100, 100],
        "box_type": "bbox",
        "box_coordinates": [10, 10, 20, 20],
        "GUI_types": [],
    }
    env = OSWorldGEnv(
        annotation_original=annotation,
        annotation_refined=annotation,
        images_dir=Path("/tmp"),
        task_id="fixture",
        valid_actions=[],
    )
    env._screenshot = b"shot"

    result = await env.step([
        make_tool_call("point", {"coordinate": [150, 150]}, call_id="blocked"),
    ])

    assert result.reward == 0.0
    assert result.terminated is True
    assert result.results[0].tool_call_id == "blocked"
    assert result.results[0].images[-1] == b"shot"
    assert result.results[0].error == (
        "invalid action: point; choose an available action for this task"
    )
    assert result.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_osworld_g_bbox_tool_is_rejected_with_current_image() -> None:
    from lite.gym.envs.osworld_g.main import OSWorldGEnv

    annotation = {
        "instruction": "click the target",
        "image_path": "unused.png",
        "image_size": [100, 100],
        "box_type": "bbox",
        "box_coordinates": [10, 10, 20, 20],
        "GUI_types": [],
    }
    env = OSWorldGEnv(
        annotation_original=annotation,
        annotation_refined=annotation,
        images_dir=Path("/tmp"),
        task_id="fixture",
    )
    env._screenshot = b"shot"

    result = await env.step([
        make_tool_call("bbox", {"coordinate": [None, None, None, None]}, call_id="bbox"),
    ])

    assert result.reward == 0.0
    assert result.terminated is True
    assert result.results[0].tool_call_id == "bbox"
    assert result.results[0].images[-1] == b"shot"
    assert result.results[0].text is None
    assert result.results[0].error == (
        "invalid action: bbox; choose an available action for this task"
    )
    assert result.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_osworld_g_unknown_tool_is_error_only_feedback() -> None:
    from lite.gym.envs.osworld_g.main import OSWorldGEnv

    annotation = {
        "instruction": "click the target",
        "image_path": "unused.png",
        "image_size": [100, 100],
        "box_type": "bbox",
        "box_coordinates": [10, 10, 20, 20],
        "GUI_types": [],
    }
    env = OSWorldGEnv(
        annotation_original=annotation,
        annotation_refined=annotation,
        images_dir=Path("/tmp"),
        task_id="fixture",
    )
    env._screenshot = b"shot"

    result = await env.step([
        make_tool_call("frobnicate", {}, call_id="foo"),
    ])

    assert result.terminated is True
    assert result.truncated is False
    assert result.results[0].tool_call_id == "foo"
    assert result.results[0].images == []
    assert result.results[0].text is None
    assert result.results[0].error == "unknown tool: frobnicate"
    assert result.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_osworld_g_inactive_report_infeasible_returns_error_only_feedback() -> None:
    from lite.gym.envs.osworld_g.main import OSWorldGEnv

    annotation = {
        "instruction": "click the target",
        "image_path": "unused.png",
        "image_size": [100, 100],
        "box_type": "bbox",
        "box_coordinates": [10, 10, 20, 20],
        "GUI_types": [],
    }
    env = OSWorldGEnv(
        annotation_original=annotation,
        annotation_refined=annotation,
        images_dir=Path("/tmp"),
        task_id="fixture",
    )
    env._screenshot = b"shot"

    result = await env.step([
        make_tool_call("report_infeasible", {"reason": "not visible"}, call_id="refuse"),
    ])

    assert result.terminated is True
    assert result.truncated is False
    assert result.results[0].tool_call_id == "refuse"
    assert result.results[0].images == []
    assert result.results[0].text is None
    assert result.results[0].error == "report_infeasible is not available in this task."
    assert result.results[0].metadata == {"is_error": True}


@pytest.mark.asyncio
async def test_osworld_g_multiple_points_score_the_first_valid_answer() -> None:
    from lite.gym.envs.osworld_g.main import OSWorldGEnv

    annotation = {
        "instruction": "click the target",
        "image_path": "unused.png",
        "image_size": [100, 100],
        "box_type": "bbox",
        "box_coordinates": [10, 10, 10, 10],
        "GUI_types": [],
    }
    env = OSWorldGEnv(
        annotation_original=annotation,
        annotation_refined=annotation,
        images_dir=Path("/tmp"),
        task_id="fixture",
    )
    env._screenshot = b"shot"

    result = await env.step([
        make_tool_call("point", {"coordinate": [150, 150]}, call_id="inside"),
        make_tool_call("point", {"coordinate": [900, 900]}, call_id="outside"),
    ])

    assert result.reward == 1.0
    assert result.terminated is True
    assert result.truncated is False
    assert result.info["annotation"] == annotation
    assert result.info["executed_actions"] == [
        {"call": "point", "args": {"x": 15.0, "y": 15.0}},
        {"call": "point", "args": {"x": 90.0, "y": 90.0}},
    ]
    assert result.results == []
