"""PhoneWorld gym environment on cua-lite's Android emulator runtime."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

from lite.core.metadata import LiteCUAMetadata
from lite.core.tools.extra_tools import make_open_app_tool
from lite.core.tools.schemas import BaseTools
from lite.gym.envs.androidworld.main import (
    _EXECUTOR,
    AndroidWorldEnv,
    AndroidWorldTaskConfig,
)
from lite.gym.registry import register, registry
from lite.gym.remote.reaper import ContainerServices
from lite.gym.services import BackendFamily, register_family, register_services
from lite.gym.types import LiteExecutedAction
from lite.gym.utils import config as env_config
from lite.gym.utils.backend.freshness import image_for
from lite.gym.utils.feedback.surface import (
    android_supported_actions,
    resolve_extra_tools,
    resolve_schema_valid_actions,
    resolve_valid_actions,
)
from lite.gym.utils.server.health import CachedEnvDepsHealthCheck

ENV_DIR = str(Path(__file__).parent)
CFG = env_config.load(ENV_DIR)
_IMAGE = CFG.env_kwargs["image"]
_MAX_STEPS = CFG.env_kwargs["max_steps"]
_POST_ACTION_DELAY = CFG.env_kwargs["post_action_delay"]
_OBSERVATION_TEXT = CFG.env_kwargs["observation_text"]
_SEED = CFG.env_kwargs["seed"]
_VALID_ACTIONS = CFG.env_kwargs.get("valid_actions")
_EXTRA_TOOLS = CFG.env_kwargs["extra_tools"]
_MAX_RESETS_PER_CONTAINER = CFG.server_kwargs["max_resets_per_container"]

_APP_LABELS = {
    "m12306": "12306",
    "malipay": "支付宝",
    "mbeike": "贝壳找房",
    "mbilibili": "哔哩哔哩",
    "mctrip": "携程旅行",
    "mdewu": "得物",
    "mdianping": "大众点评",
    "mdidi": "滴滴出行",
    "mdouban": "豆瓣",
    "mdouyin": "抖音",
    "meleme": "饿了么",
    "mfanqie": "番茄免费小说",
    "mgaode": "高德地图",
    "mhema": "盒马",
    "miqiyi": "爱奇艺",
    "mjd": "京东",
    "mkeep": "Keep",
    "mkfc": "肯德基",
    "mmcdonalds": "麦当劳",
    "mmeituan": "美团",
    "mmeituan_waimai": "美团外卖",
    "mnetease_music": "网易云音乐",
    "mqq": "QQ",
    "mqqmusic": "QQ音乐",
    "mtaobao": "淘宝",
    "mtengxunshipin": "腾讯视频",
    "mths": "同花顺",
    "mtoutiao": "今日头条",
    "mweibo": "微博",
    "mweread": "微信读书",
    "mxianyu": "闲鱼",
    "mxiaohongshu": "小红书",
    "mximalaya": "喜马拉雅",
    "mzhihu": "知乎",
}
_APP_PACKAGES = {
    **{app: f"com.phoneuse.{app}" for app in _APP_LABELS},
    **{label: f"com.phoneuse.{app}" for app, label in _APP_LABELS.items()},
}
_APP_CATALOG = list(_APP_LABELS.values()) + list(_APP_LABELS)
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_OPS = {"==": "=", ">": ">", "<": "<", ">=": ">=", "<=": "<=", "LIKE": "LIKE"}
_SUPPORTED_ACTIONS = android_supported_actions()


class PhoneWorldTools(BaseTools):
    _SCHEMAS: ClassVar[dict[str, dict[str, Any]]] = {
        "open_app": make_open_app_tool(_APP_CATALOG),
    }


class PhoneWorldEnv(AndroidWorldEnv):
    """Static PhoneWorld tasks with container-per-episode reset semantics."""

    ENV_ID = "phoneworld"
    EXTRA_TOOLS = PhoneWorldTools

    def __init__(
        self,
        *,
        config: AndroidWorldTaskConfig,
        image: str = _IMAGE,
        task_id: str = "",
        max_steps: int = _MAX_STEPS,
        post_action_delay: float = _POST_ACTION_DELAY,
        observation_text: str = _OBSERVATION_TEXT,
        seed: int | None = _SEED,
        valid_actions: list[str] | None = _VALID_ACTIONS,
        extra_tools: list[str] | None = _EXTRA_TOOLS,
    ) -> None:
        super().__init__(
            config=config,
            image=image,
            task_id=task_id,
            max_steps=max_steps,
            post_action_delay=post_action_delay,
            observation_text=observation_text,
            seed=seed,
            valid_actions=valid_actions,
            extra_tools=extra_tools,
        )
        self._max_resets_per_container = _MAX_RESETS_PER_CONTAINER

    @staticmethod
    def _task_metadata(config_metadata: dict[str, Any]) -> LiteCUAMetadata:
        return LiteCUAMetadata(
            dims=("mobile", "use"),
            extra_tool_schemas=resolve_extra_tools(
                _EXTRA_TOOLS,
                tools=PhoneWorldTools,
                env_name="phoneworld",
            ),
            valid_actions=resolve_schema_valid_actions(
                _VALID_ACTIONS,
                env_name="phoneworld",
                platform="mobile",
                supported_actions=_SUPPORTED_ACTIONS,
            ),
            others={**config_metadata, "apps": list(_APP_CATALOG)},
        )

    def bind(
        self,
        task_id: str = "",
        *,
        max_steps: int = _MAX_STEPS,
        post_action_delay: float = _POST_ACTION_DELAY,
        observation_text: str = _OBSERVATION_TEXT,
        seed: int | None = _SEED,
        valid_actions: list[str] | None = _VALID_ACTIONS,
        extra_tools: list[str] | None = _EXTRA_TOOLS,
    ) -> None:
        if observation_text not in {"none", "a11y:pixel", "a11y:norm"}:
            raise ValueError(f"unknown observation_text={observation_text!r}")
        if task_id:
            config, task = _TASK_CONFIGS.get(task_id, (None, None))
            if config is None:
                raise ValueError(f"unknown phoneworld task_id={task_id!r}")
            self._config = dataclasses.replace(
                config,
                metadata={**config.metadata},
                emulator_setup=self._config.emulator_setup,
                freeze_datetime=self._config.freeze_datetime,
                use_fake=self._config.use_fake,
                avd_name=self._config.avd_name,
            )
            self._task_static = task
            self._instruction = task["goal"]
        else:
            self._task_static = {}
            self._instruction = (
                self._config.instruction_template or "Interact with the Android device."
            )
        self._task_id = task_id
        self._task_class_name = None
        self._task_class = None
        self._task = None
        self._seed = seed
        self._max_steps = max_steps
        self._post_action_delay = post_action_delay
        self._observation_text = observation_text
        self._valid_actions = resolve_valid_actions(
            valid_actions,
            env_name="phoneworld",
            platform="mobile",
        )
        self._schema_valid_actions = resolve_schema_valid_actions(
            valid_actions,
            env_name="phoneworld",
            platform="mobile",
            supported_actions=_SUPPORTED_ACTIONS,
        )
        self._extra_tool_schemas = type(self).extra_tool_schemas(extra_tools)
        self._step_count = 0
        self._terminated = False

    async def init_task(self) -> None:
        self._step_count = 0
        self._terminated = False
        if self._env is not None:
            self._env.interaction_cache = ""

    async def _dispatch_action(self, name: str, args: dict[str, Any]) -> list[LiteExecutedAction]:
        if self._config.use_fake:
            return await super()._dispatch_action(name, args)
        loop = asyncio.get_event_loop()
        if name == "type":
            text = args.get("text", "")
            if not isinstance(text, str):
                raise ValueError("type.text must be a string")
            await loop.run_in_executor(
                _EXECUTOR,
                partial(self._env._rpc.post, "/env/type_b64", body={"text": text}),
            )
            return [{"call": "adb.input_text_b64", "args": {"text": text}}]
        if name == "open_app":
            app_name = args.get("app_name", "")
            package = _APP_PACKAGES.get(app_name)
            if package is None:
                raise ValueError(f"unknown PhoneWorld app {app_name!r}")
            await loop.run_in_executor(
                _EXECUTOR,
                partial(self._env._rpc.post, "/env/launch_package", body={"package": package}),
            )
            return [{"call": "adb.launch_package", "args": {"package": package}}]
        return await super()._dispatch_action(name, args)

    async def _evaluate_episode(
        self,
        terminated: bool,
        truncated: bool,
        loop: asyncio.AbstractEventLoop,
    ) -> float | None:
        if not (terminated or truncated):
            return None
        verifier = self._task_static["verification"]
        answer = self._env.interaction_cache or ""
        if verifier["type"] == "answer_contains":
            return float(_answer_matches(verifier, answer))
        results = []
        for check in verifier["checks"]:
            sql = _count_sql(check)
            result = await loop.run_in_executor(
                _EXECUTOR,
                partial(
                    self._env._rpc.post,
                    "/env/sqlite_count",
                    body={"database_path": verifier["database_path"], "sql": sql},
                ),
            )
            results.append(int(result["count"]) >= check["count_min"])
        ok = all(results)
        if "answer_contains" in verifier:
            ok = ok and _answer_matches(verifier["answer_contains"], answer)
        return float(ok)


def _sql_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _count_sql(check: dict[str, Any]) -> str:
    if not _IDENT.fullmatch(check["table"]):
        raise ValueError("invalid verifier table")
    conditions = []
    for rule in check["rules"]:
        field, operator, value = rule["field"], rule["operator"], rule["value"]
        if not _IDENT.fullmatch(field):
            raise ValueError("invalid verifier field")
        if operator == "contains":
            conditions.append(f"{field} LIKE {_sql_literal('%' + str(value) + '%')}")
        elif operator in _OPS:
            conditions.append(f"{field} {_OPS[operator]} {_sql_literal(value)}")
        else:
            raise ValueError(f"invalid verifier operator {operator!r}")
    where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    return f"SELECT COUNT(*) FROM {check['table']}{where}"


def _answer_matches(verifier: dict[str, Any], answer: str) -> bool:
    stripped = answer.strip()
    if len(stripped) < verifier.get("min_length", 0):
        return False
    keywords = verifier.get("keywords", [])
    return all(str(keyword).lower() in answer.lower() for keyword in keywords) and (
        bool(stripped) or bool(keywords) or verifier.get("min_length", 0) > 0
    )


def _load_tasks() -> dict[str, list[dict[str, Any]]]:
    with (Path(__file__).parent / "data" / "tasks.json").open() as file:
        return json.load(file)["splits"]


def _make_env(*, config: AndroidWorldTaskConfig, **kwargs: Any) -> PhoneWorldEnv:
    fields = AndroidWorldTaskConfig.__dataclass_fields__
    overrides = {key: value for key, value in kwargs.items() if key in fields}
    config = dataclasses.replace(config, metadata={**config.metadata}, **overrides)
    return PhoneWorldEnv(config=config, **{k: v for k, v in kwargs.items() if k not in fields})


_TASK_CONFIGS: dict[str, tuple[AndroidWorldTaskConfig, dict[str, Any]]] = {}
registry.set_env_make_kwargs("phoneworld", CFG.make_kwargs)
for _split, _tasks in _load_tasks().items():
    for _task in _tasks:
        _task_id = _task["task_id"]
        _others = {
            "app": _task["apps"][-1],
            "task_apps": list(_task["apps"]),
            "difficulty": _task["difficulty"],
            "verification_type": _task["verification"]["type"],
        }
        _config = AndroidWorldTaskConfig(
            instruction_template=_task["goal"],
            metadata=_others,
        )
        _TASK_CONFIGS[_task_id] = (_config, _task)
        register(
            key=f"phoneworld@{_task_id}",
            entry_point=partial(_make_env, config=_config, task_id=_task_id),
            split=_split,
            metadata=PhoneWorldEnv._task_metadata(_others),
        )


def _ensure_services(_env_id: str) -> None:
    from lite.gym.utils.backend.docker import require_image_present

    require_image_present(image_for("phoneworld", tag=_IMAGE))


_HEALTH_CHECK = CachedEnvDepsHealthCheck(_ensure_services)


class PhoneWorldServices(ContainerServices):
    def ensure(self, env_id: str) -> None:
        _ensure_services(env_id)

    def health(self, env_id: str) -> None:
        _HEALTH_CHECK(env_id)


register_services("phoneworld", PhoneWorldServices())
register_family("phoneworld", BackendFamily.DEDICATED)
