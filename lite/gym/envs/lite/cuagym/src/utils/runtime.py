"""Registration-free runtime configuration shared by rollout and validation.

Also owns the in-guest LLM-judge shim both backends install before running a
bundle's ``reward.py`` — see :data:`REWARD_JUDGE_MODULE` and
:func:`install_reward_judge`.

Run: not directly — imported by ``src/browser/scripts.py`` and
``src/desktop/scripts.py``. Smoke-test the resolved judge host-side with::

    uv run python -c "from lite.gym.envs.lite.cuagym.src.utils.runtime import \
redacted_judge_config; print(redacted_judge_config())"
"""

from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path
from typing import Any

from lite.core.tools import make_tool_call
from lite.gym.envs.lite.cuagym.src.utils.container import write_text
from lite.gym.errors import CuaGymTaskError
from lite.gym.sandbox.base import SandboxBaseEnv
from lite.gym.utils import config as env_config

ENV_DIR = Path(__file__).resolve().parents[2]
CFG = env_config.load(str(ENV_DIR))
ENV_KWARGS = CFG.env_kwargs
MAKE_KWARGS = CFG.make_kwargs
STEP_TIMEOUT = float(MAKE_KWARGS.get("step_timeout", 120.0))
ENV_PY = "/opt/env/venv/bin/python"
UNO_PY = "/opt/env/uno-venv/bin/python"
# The inherited GNOME marker appears before the child image's startup additions
# finish. Wait for a CUAGym marker so setup cannot launch an app in that gap.
READY_MARKER = "/tmp/lite-cuagym-ready"

_computer = ENV_KWARGS["computer"]
COMPUTER_CONFIG: dict[str, Any] = {
    "image": _computer["image"],
    "display": (
        f"{_computer['display_resolution'][0]}"
        f"x{_computer['display_resolution'][1]}"
    ),
    "memory": _computer["memory"],
    "cpu": _computer["cpu"],
    "timeout": _computer["timeout"],
}

DESKTOP_HEALTH_COMMAND = (
    "pgrep -x gnome-shell >/dev/null "
    "&& ! pgrep -f '[g]nome-session-failed' >/dev/null "
    "&& echo OK || echo DOWN"
)
GNOME_FAILURE_COMMAND = (
    "if [ -s /tmp/gnome-failed ]; then cat /tmp/gnome-failed; "
    "else true; fi"
)


async def desktop_alive(computer: Any) -> bool:
    """Run the same post-reset desktop health check for rollout and validation."""
    result = await computer.interface.run_command(DESKTOP_HEALTH_COMMAND)
    return (getattr(result, "stdout", "") or "").strip() == "OK"


async def desktop_failure_diagnostic(computer: Any) -> str | None:
    """Return a retryable desktop failure diagnostic, if one is present."""
    marker = await computer.interface.run_command(GNOME_FAILURE_COMMAND)
    detail = (getattr(marker, "stdout", "") or "").strip()
    if detail:
        return detail
    if not await desktop_alive(computer):
        return "post-reset GNOME desktop health check returned DOWN"
    return None


async def validate_pre_setup_runtime(computer: Any) -> None:
    """Require a live desktop before task setup starts."""
    if not await desktop_alive(computer):
        raise RuntimeError("pre-setup GNOME desktop health check returned DOWN")


async def validate_post_setup_runtime(
    computer: Any,
    *,
    screenshot: Any | None = None,
) -> None:
    """Exercise reset's screenshot channel and post-reset desktop health contract."""
    if screenshot is None:
        screenshot = await computer.interface.screenshot()
    if screenshot in (None, b"", ""):
        raise RuntimeError("reset screenshot channel returned an empty image")
    if not await desktop_alive(computer):
        raise RuntimeError("post-reset GNOME desktop health check returned DOWN")


async def window_ids(computer: Any) -> set[str]:
    """Return current X11 window IDs without depending on shell window names."""
    # Every other X-touching command here names DISPLAY explicitly; this one used to
    # ride whatever the image exported, which the transport deliberately does NOT
    # promise ("DISPLAY/HOME ride the image env", exec_stdio/server.py). Silence made
    # that dangerous rather than merely implicit: `2>/dev/null` dropped wmctrl's
    # "cannot open display", and the pipe's exit code is awk's, so a broken X
    # connection returned an EMPTY SET -- indistinguishable from "no windows yet".
    # Every caller reads that as a missing window, which is a hard `no_task_window`
    # setup failure for the browser and document branches.
    result = await computer.interface.run_command("DISPLAY=:0 wmctrl -l")
    if getattr(result, "returncode", 0) != 0:
        raise CuaGymTaskError(
            f"lite.cuagym cannot list windows: {(result.stderr or '').strip()[-200:]}",
            phase="setup",
            kind="command_failed",
        )
    return {
        line.split(None, 1)[0]
        for line in (result.stdout or "").splitlines()
        if line.strip()
    }


async def wait_for_new_window(
    computer: Any,
    existing: set[str],
    *,
    timeout: int = 25,
) -> bool:
    """Wait for a GUI launch to add a window, then allow it to paint."""
    for _ in range(timeout):
        if await window_ids(computer) - existing:
            await asyncio.sleep(2)
            return True
        await asyncio.sleep(1)
    return False


async def evaluate_terminal_step(
    task: Any,
    computer: Any,
    evaluate_final_fn: Any,
    *,
    timeout: float = STEP_TIMEOUT,
) -> float:
    """Run the inherited rollout terminal-step path under one step deadline."""
    env = object.__new__(SandboxBaseEnv)
    env._computer = computer
    env._post_action_delay = 0.0
    env._step_count = 0
    env._max_steps = None
    env._evaluate_step_fn = None
    env._evaluate_final_fn = evaluate_final_fn
    env._task = task
    env._debug = False
    result = await asyncio.wait_for(
        SandboxBaseEnv.step(
            env,
            [make_tool_call("terminate", {})],
        ),
        timeout=max(0.0, timeout),
    )
    if not result.terminated or result.truncated:
        raise RuntimeError("rollout terminal step did not terminate cleanly")
    if result.reward is None:
        raise RuntimeError("rollout terminal step returned no reward")
    return float(result.reward)


# --------------------------------------------------------------------------- #
# LLM-judge shim (`/tmp/reward_judge.py`).
# --------------------------------------------------------------------------- #
#: Where upstream expects the judge. 46 pinned bundles (38 live: 36 of the 1261
#: live browser tasks, 2 of the 9138 live desktop tasks) do
#: ``sys.path.insert(0, '/tmp'); from reward_judge import call_llm_judge``.
#: Nothing ever wrote it, and EVERY importer guards the import, so the component
#: silently scored 0 — indistinguishable from the agent not earning those points.
REWARD_JUDGE_MODULE = "/tmp/reward_judge.py"

#: Substituted with the host-resolved judge config when the module is installed.
#: Same trick as ``browser/scripts.py::_materialize`` — a placeholder, not an f-string,
#: so the module body below stays readable, valid Python and greppable.
_JUDGE_CONFIG_PLACEHOLDER = "__CUA_GYM_JUDGE_CONFIG__"

#: The judge module, verbatim, as it lands in the guest.
#:
#: API derived by an exhaustive AST census of every ``*.py`` in the 10910 pinned
#: bundles, not by guesswork: **52 call sites in 46 files, all three arguments
#: passed BY KEYWORD at every site, zero positional args, zero ``**kwargs``
#: splats and zero extra kwargs.** So the signature is exactly these three names
#: and there is nothing for a ``**kwargs`` sponge to absorb (contrast
#: ``query_vlm``-style sponge in a sibling env, whose census found 11 such sites).
#: The return value is consumed as a FLOAT in [0, 1] and multiplied by the
#: component's weight (``total_score += 0.15 * llm_score``) or compared against a
#: threshold (``if llm_score >= 0.5``), so the contract is a fraction of credit.
_REWARD_JUDGE_SOURCE = '''"""LLM judge for CUA-Gym reward scripts — installed by lite.cuagym at /tmp.

Upstream bundles do ``sys.path.insert(0, '/tmp')`` then
``from reward_judge import call_llm_judge`` and score
``weight * call_llm_judge(task_instruction=…, success_criteria=…, state_excerpt=…)``.

Configuration is resolved on the HOST and baked into CONFIG below; nothing here
reads the guest environment. See ``src/utils/runtime.py::judge_config``.
"""

from __future__ import annotations

import json
import random
import re
import time
import urllib.error
import urllib.request

CONFIG = __CUA_GYM_JUDGE_CONFIG__


class JudgeUnavailable(RuntimeError):
    """The judge could not be reached / did not answer at all.

    Raised — never swallowed into a 0.0 — because all 46 live call sites wrap the
    call in ``except`` precisely to fall back on a deterministic keyword check.
    Returning 0.0 on a provider outage would walk straight past that handler and
    silently delete the consolation credit those branches grant.
    """


_SYSTEM = (
    "You are a strict grader for a computer-use benchmark. You are given a task "
    "instruction, the success criteria for one scored component, and an excerpt of "
    "the resulting application state. Judge ONLY the stated criteria. "
    'Reply with JSON and nothing else: {"score": <number between 0 and 1>, '
    '"reason": "<one sentence>"}. Use 1 for fully met, 0 for not met, and a '
    "fraction for partial credit."
)

_USER = (
    "TASK INSTRUCTION:\\n{task_instruction}\\n\\n"
    "SUCCESS CRITERIA:\\n{success_criteria}\\n\\n"
    "OBSERVED STATE:\\n{state_excerpt}\\n"
)

#: Score shapes a real model emits when it ignores the JSON contract, each paired
#: with the denominator its number is out of (None = read it from group 2).
_PATTERNS = (
    (re.compile(r"(?:score|rating|grade)[\\"\\']?\\s*[:=]\\s*([01](?:\\.\\d+)?)", re.I), 1.0),
    (re.compile(r"\\b(\\d{1,3}(?:\\.\\d+)?)\\s*%"), 100.0),
    (re.compile(r"\\b(\\d{1,3}(?:\\.\\d+)?)\\s*/\\s*(1|1\\.0|5|10|100)\\b"), None),
    (re.compile(r"^\\s*([01](?:\\.\\d+)?)\\s*$"), 1.0),
)


def _clamp(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def _score_from(text: str):
    """Read a 0..1 score out of a model reply, or return None."""
    if not text:
        return None
    try:
        payload = json.loads(text)
    except ValueError:
        match = re.search(r"\\{[\\s\\S]*\\}", text)
        try:
            payload = json.loads(match.group()) if match else None
        except ValueError:
            payload = None
    if isinstance(payload, dict):
        for key in ("score", "rating", "grade", "value"):
            raw = payload.get(key)
            if isinstance(raw, bool):
                return 1.0 if raw else 0.0
            if isinstance(raw, (int, float)):
                return _clamp(float(raw) / 100.0 if float(raw) > 1.0 else float(raw))
        for key in ("passed", "success", "correct", "answer"):
            if isinstance(payload.get(key), bool):
                return 1.0 if payload[key] else 0.0
    for pattern, denominator in _PATTERNS:
        match = pattern.search(text)
        if match:
            scale = float(match.group(2)) if denominator is None else denominator
            return _clamp(float(match.group(1)) / scale)
    low = text.strip().lower()
    if low.startswith("yes") or low.startswith("true") or low.startswith("pass"):
        return 1.0
    if low.startswith("no") or low.startswith("false") or low.startswith("fail"):
        return 0.0
    return None


def _endpoint() -> str:
    base = (CONFIG["base_url"] or "https://api.openai.com/v1").rstrip("/")
    return base if base.endswith("/chat/completions") else base + "/chat/completions"


def _redact(text: str) -> str:
    redacted = str(text)
    key = CONFIG.get("api_key")
    if key:
        redacted = redacted.replace(str(key), "<redacted>")
    return re.sub(
        r"(?i)(api[_-]?key|authorization|bearer)\\s*[:=]\\s*['\\"]?[^'\\",}\\s]+",
        r"\\1=<redacted>",
        redacted,
    )


def _ask(prompt: str) -> str:
    body = json.dumps({
        "model": CONFIG["model"],
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_completion_tokens": 512,
    }).encode()
    headers = {"Content-Type": "application/json"}
    if CONFIG["api_key"]:
        headers["Authorization"] = "Bearer " + CONFIG["api_key"]
    request = urllib.request.Request(_endpoint(), data=body, headers=headers)
    last = None
    for attempt in range(CONFIG["max_retries"]):
        try:
            with urllib.request.urlopen(request, timeout=CONFIG["timeout"]) as response:
                payload = json.loads(response.read().decode("utf-8", "replace"))
            return payload["choices"][0]["message"]["content"] or ""
        except Exception as exc:  # noqa: BLE001 - bounded provider retry
            last = exc
            if attempt + 1 < CONFIG["max_retries"]:
                delay = min(8, 2 ** attempt)
                time.sleep(delay * (0.75 + random.random() * 0.5))
    raise JudgeUnavailable("%s: %s" % (type(last).__name__, _redact(last)))


def call_llm_judge(task_instruction: str, success_criteria: str,
                   state_excerpt: str) -> float:
    """Score one rubric component in [0.0, 1.0].

    A PROVIDER failure raises ``JudgeUnavailable``; an UNREADABLE REPLY scores
    0.0. The two are different answers: the judge that never answered must reach
    the caller's ``except`` (its deterministic fallback), while a judge that DID
    answer something we cannot read has awarded no evidence of quality, and
    routing that to the fallback would hand out points the judge never endorsed.
    """
    reply = _ask(_USER.format(
        task_instruction=task_instruction,
        success_criteria=success_criteria,
        state_excerpt=state_excerpt,
    ))
    score = _score_from(reply)
    if score is None:
        print("WARN: reward_judge could not read a score from the reply: %r" % reply[:200])
        return 0.0
    return score
'''


#: The judge model, from ``configs/default.yaml`` (``env_kwargs.judge.model``) — a
#: declared default rather than a literal buried here, so it sits with the env's other
#: knobs and can be reviewed and overridden the same way.
_DEFAULT_JUDGE_MODEL: str = str(ENV_KWARGS["judge"]["model"])

#: Judge settings, most specific first. The `VLM_*` spelling is an ALIAS, kept so
#: one deployment can configure this judge and another env's with the same
#: exports; it is a name coincidence, not a code dependency.
_JUDGE_ENV_PREFIXES = ("LITE_CUAGYM_JUDGE_", "VLM_")


def _judge_env(suffix: str, default: Any) -> Any:
    for prefix in _JUDGE_ENV_PREFIXES:
        value = os.environ.get(prefix + suffix)
        if value:
            return value
    return default


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def judge_config() -> dict[str, Any]:
    """Resolve the judge call HOST-side; a model id is the only required input.

    A MODEL ID is the only thing an operator must supply, and it already defaults
    to :data:`_DEFAULT_JUDGE_MODEL`. Everything else is optional::

        # Default route: nothing judge-specific.
        export OPENAI_API_KEY=...
        # Optional custom endpoint.
        export OPENAI_BASE_URL=...

        # A different judge model, and/or an endpoint just for the judge.
        export LITE_CUAGYM_JUDGE_MODEL=Qwen/Qwen3-VL-8B-Instruct
        export LITE_CUAGYM_JUDGE_BASE_URL=...
        export LITE_CUAGYM_JUDGE_API_KEY=...

    ``VLM_MODEL`` / ``VLM_BASE_URL`` / ``VLM_API_KEY`` are honoured as aliases so a
    deployment that already exports them for another env's judge configures this
    one too — but purely by NAME. lite.cuagym imports nothing from a sibling env;
    this resolver is self-contained.

    Why the wire protocol and not litellm, which is what a host-side judge in this
    repo would use: a CUA-Gym ``reward.py`` runs INSIDE the task container, and the
    pinned image's ``/opt/env/venv`` has no litellm and no openai SDK (only
    ``requests`` / ``urllib3`` / ``certifi``; the image is hash-gated, so that
    cannot be fixed from here). The guest therefore speaks the one protocol it can
    speak unaided — OpenAI-compatible ``/chat/completions`` over stdlib ``urllib``.
    That is not a backend SWITCH: there is exactly one backend, no per-provider key
    table and no ``*_BACKEND`` knob. A non-OpenAI provider is reached the way
    litellm itself documents it — point the base URL at a litellm proxy.
    """
    model = _optional_text(_judge_env("MODEL", _DEFAULT_JUDGE_MODEL))
    if model is None:
        raise ValueError(
            "CUAGym judge model is empty; set env_kwargs.judge.model or "
            "LITE_CUAGYM_JUDGE_MODEL"
        )
    model = model.removeprefix("openai/")
    if not model:
        raise ValueError("CUAGym judge model is empty after removing provider prefix")
    try:
        max_retries = int(_judge_env("MAX_RETRIES", 3))
    except (TypeError, ValueError) as exc:
        raise ValueError("LITE_CUAGYM_JUDGE_MAX_RETRIES must be a positive integer") from exc
    if max_retries < 1:
        raise ValueError("LITE_CUAGYM_JUDGE_MAX_RETRIES must be a positive integer")
    try:
        timeout = float(_judge_env("TIMEOUT", 180.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("LITE_CUAGYM_JUDGE_TIMEOUT must be a positive finite number") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("LITE_CUAGYM_JUDGE_TIMEOUT must be a positive finite number")
    return {
        # An `openai/`-prefixed id means "OpenAI-compatible" and travels under its
        # bare name; any other prefix is passed through for a gateway to route.
        "model": model,
        # Never invent a value: an unset base URL/key stays None, and the guest
        # falls back to the public endpoint / an unauthenticated request rather
        # than sending an empty string that would override a working default.
        "base_url": _optional_text(
            _judge_env("BASE_URL", None) or os.environ.get("OPENAI_BASE_URL")
        ),
        "api_key": _optional_text(
            _judge_env("API_KEY", None) or os.environ.get("OPENAI_API_KEY")
        ),
        "max_retries": max_retries,
        "timeout": timeout,
    }


def redacted_judge_config() -> dict[str, Any]:
    """Return the resolved judge config with credentials masked for smoke tests."""
    config = dict(judge_config())
    if config.get("api_key"):
        config["api_key"] = "<redacted>"
    return config


def reward_judge_source(config: dict[str, Any] | None = None) -> str:
    """The guest module with the host-resolved config baked in.

    ``repr`` and not ``json.dumps``: the placeholder sits in PYTHON source, and
    JSON's ``null`` (an unset ``base_url``/``api_key`` — the common case, since
    both are optional) is a NameError there.
    """
    return _REWARD_JUDGE_SOURCE.replace(
        _JUDGE_CONFIG_PLACEHOLDER,
        repr(judge_config() if config is None else config),
    )


def bundle_needs_judge(source: str) -> bool:
    """True when a bundle's reward script imports the judge.

    Every one of the 46 pinned importers spells the module name, whether it uses
    ``from reward_judge import …`` or ``importlib.import_module('reward_judge')``.
    Gating on this keeps the other 10864 rewards byte-for-byte unchanged.
    """
    return "reward_judge" in source


async def install_reward_judge(computer: Any) -> None:
    """Write the judge module where the bundle's ``reward.py`` will import it.

    Installed at EVALUATION time, after the episode has ended, so the resolved
    credentials never exist in a container the agent is still driving.
    """
    await write_text(computer, reward_judge_source(), REWARD_JUDGE_MODULE)
