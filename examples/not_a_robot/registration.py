"""Register locally authored tasks plus the existing Neal and fixture modes."""

from lite.gym.registry import register, registry
from lite.gym.services import BackendFamily, EnvServices, register_family, register_services

from .env import LIVE_ENVS, NotARobotEnv
from .local_tasks import LOCAL_TASKS


class NealServices(EnvServices):
    """Direct-mode lifecycle declaration; browsers are owned by their envs."""

    def live_ids(self, env_id, scope):
        return set(LIVE_ENVS)


for env_id, task_id, target in (
    ("not_a_robot", "level_001", 1),
    ("not_a_robot_campaign", "full_game", None),
):
    register_family(env_id, BackendFamily.DEDICATED)
    register_services(env_id, NealServices())
    registry.set_env_supported_kwargs(
        env_id,
        {
            "mode",
            "artifact_root",
            "browser_executable",
            "display_resolution",
            "headless",
            "max_steps",
            "max_seconds",
            "post_action_delay",
            "cursor",
            "extra_tools",
        },
    )
    register(
        f"{env_id}@{task_id}",
        NotARobotEnv,
        split="eval",
        metadata=NotARobotEnv.task_metadata(target),
        target_level=target,
    )


register_family("visual_tasks", BackendFamily.DEDICATED)
register_services("visual_tasks", NealServices())
registry.set_env_supported_kwargs(
    "visual_tasks",
    {
        "artifact_root",
        "browser_executable",
        "display_resolution",
        "headless",
        "max_steps",
        "max_seconds",
        "post_action_delay",
        "cursor",
        "extra_tools",
        "seed",
    },
)
for task_id in LOCAL_TASKS:
    register(
        f"visual_tasks@{task_id}",
        NotARobotEnv,
        split="eval",
        metadata=NotARobotEnv.task_metadata(None, "local", local_task=task_id),
        mode="local",
        target_level=None,
        local_task=task_id,
    )
