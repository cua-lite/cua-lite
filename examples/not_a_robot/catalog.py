"""User-requested 48-level inventory; only level one has an independent start."""

from __future__ import annotations

LEVELS = tuple(
    {
        "task_id": f"level_{level:03d}",
        "level_id": level,
        "title": "Checkbox" if level == 1 else None,
        "independent_start_implemented": level == 1,
        "live_verified": False,
    }
    for level in range(1, 49)
)

GAME_URL = "https://neal.fun/not-a-robot/"
VALID_ACTIONS = [
    "click",
    "type",
    "key",
    "scroll",
    "drag",
    "wait",
    "screenshot",
    "mouse_move",
    "mouse_down",
    "mouse_up",
    "key_down",
    "key_up",
    "hold_key",
]
