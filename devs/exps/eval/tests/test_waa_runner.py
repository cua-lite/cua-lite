"""Exercise WAA launch arguments and campaign reuse without starting services."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
COMMIT_DIR = "2026-09-11T10-00_abcdef1"


@pytest.fixture
def runner(tmp_path):
    for relative in (
        "devs/exps/eval/waa/run.sh",
        "devs/exps/eval/utils/runtime_mode.sh",
        "devs/exps/eval/utils/campaign_dir.sh",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    configs = tmp_path / "scripts/configs/qwen3_5/default"
    configs.mkdir(parents=True)
    for name in ("waa.yaml", "waa.benchmark.yaml"):
        (configs / name).write_text("env_id: waa\n")
    other_family = tmp_path / "scripts/configs/qwen3_8/default"
    other_family.mkdir(parents=True)
    (other_family / "waa.yaml").write_text("env_id: waa\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    commands = {
        "git": """#!/usr/bin/env bash
case "$1" in
  status) printf '%s' "${MOCK_DIRTY:-}" ;;
  rev-parse) echo abcdef1 ;;
  log) if [[ "$*" == *--date=* ]]; then echo 2026-09-11T10-00; fi ;;
  *) exit 1 ;;
esac
""",
        "uv": """#!/usr/bin/env bash
exec "$TEST_PYTHON" -c 'import json, os, sys
print("MOCK_ROLLOUT=" + json.dumps({"argv": sys.argv[1:], "env": {
    k: os.environ.get(k) for k in ["HF_HUB_OFFLINE",
    "CUA_LITE_DOCKER_CREATE_CONCURRENCY", "CUA_LITE_EVAL_RUNTIME_MODE",
    "CUA_LITE_ENV_SERVER_URL", "CUA_LITE_ENV_SERVER_TOKEN"]}}))' "$@"
""",
        "curl": """#!/usr/bin/env bash
if [ "${MOCK_PROBE_FAIL:-0}" = 1 ]; then exit 7; fi
echo '{"available": true}'
""",
    }
    for name, text in commands.items():
        executable = bin_dir / name
        executable.write_text(text)
        executable.chmod(0o755)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("EVAL_", "CUA_LITE_", "CUDA_", "SESSION_ID"))
    }
    env.update(
        PATH=f"{bin_dir}:{env['PATH']}",
        TEST_PYTHON=sys.executable,
        EVAL_ALLOW_DIRECT="1",
        CUDA_VISIBLE_DEVICES="2,3",
        CUA_LITE_ENV_SERVER_URL="http://unused.invalid",
        CUA_LITE_ENV_SERVER_TOKEN="unused-test-token",
    )

    def launch(model="Qwen/Qwen3.5-27B", config="waa.yaml", **overrides):
        return subprocess.run(
            [
                "bash",
                str(tmp_path / "devs/exps/eval/waa/run.sh"),
                model,
                str(configs / config),
            ],
            env=env | overrides,
            cwd=tmp_path,
            text=True,
            capture_output=True,
            timeout=15,
        )

    return tmp_path, launch


def _launch_data(proc):
    assert proc.returncode == 0, proc.stderr
    line = next(line for line in proc.stdout.splitlines() if line.startswith("MOCK_ROLLOUT="))
    data = json.loads(line.removeprefix("MOCK_ROLLOUT="))
    argv = data["argv"]
    data["options"] = {
        flag: argv[index + 1]
        for index, flag in enumerate(argv[:-1])
        if flag.startswith("--") and flag != "--debug"
    }
    return data


def test_standard_filter_defaults_resume_and_config_isolation(runner):
    root, launch = runner
    first = _launch_data(launch())
    opts = first["options"]
    log_root = Path(opts["--log-root"])
    assert log_root == (
        root / ".exps/eval/waa" / COMMIT_DIR / "run_0/Qwen_Qwen3.5-27B__qwen3_5__default__waa"
    )
    assert _launch_data(launch())["options"]["--log-root"] == str(log_root)
    assert opts["--splits"] == "eval"
    assert opts["--filter"] == "lambda m: not m.others.get('exclude_reason')"
    assert opts["--concurrency"] == "30"
    assert opts["--max-attempts"] == "3" and opts["--min-valid-frac"] == "1"
    assert opts["--save-video"] == "false" and "--debug" in first["argv"]
    assert "--engine-kwargs" not in opts
    assert first["env"] == {
        "HF_HUB_OFFLINE": "1",
        "CUA_LITE_DOCKER_CREATE_CONCURRENCY": "10",
        "CUA_LITE_EVAL_RUNTIME_MODE": "direct",
        "CUA_LITE_ENV_SERVER_URL": None,
        "CUA_LITE_ENV_SERVER_TOKEN": None,
    }
    newer = log_root.parent.parent / "run_12_repeat"
    newer.mkdir()
    (log_root.parent.parent / "run_9").mkdir()
    resumed = _launch_data(launch())["options"]["--log-root"]
    assert Path(resumed).parent == newer
    variant = _launch_data(launch(config="waa.benchmark.yaml"))["options"]["--log-root"]
    assert Path(variant).parent == newer and variant != resumed
    other_family = _launch_data(launch(config="../../qwen3_8/default/waa.yaml"))["options"][
        "--log-root"
    ]
    assert Path(other_family).parent == newer and other_family != resumed
    explicit = _launch_data(launch(EVAL_RUN_ID="run_0"))["options"]["--log-root"]
    assert explicit == str(log_root)


def test_local_checkpoint_and_serving_overrides(runner):
    _, launch = runner
    engine = '{"tp_size": 1, "context_length": 32768, "max_running_requests": 30}'
    opts = _launch_data(launch(EVAL_ENGINE_KWARGS=engine))["options"]
    assert opts["--model-id"] == "Qwen/Qwen3.5-27B"
    assert "--model-path" not in opts
    assert opts["--engine-kwargs"] == engine
    opts = _launch_data(
        launch(EVAL_MODEL_PATH="/models/checkpoint with spaces", EVAL_CONCURRENCY="60")
    )["options"]
    assert opts["--model-path"] == "/models/checkpoint with spaces"
    assert opts["--concurrency"] == "60"


@pytest.mark.parametrize("model", ["gpt-5.5", "claude-opus-4-8", "gemini-3.6-flash"])
def test_api_model_does_not_require_gpus(runner, model):
    _, launch = runner
    opts = _launch_data(launch(model=model, CUDA_VISIBLE_DEVICES=""))["options"]
    assert opts["--model-id"] == model
    assert "--engine-kwargs" not in opts


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"EVAL_CONCURRENCY": "61"}, "integer from 1 to 60"),
        ({"EVAL_CONCURRENCY": "0"}, "integer from 1 to 60"),
        ({"EVAL_RUN_ID": "run_0/../../outside"}, "run_<N>[_<label>]"),
        ({"MOCK_DIRTY": " M scripts/configs/qwen3_5/default/waa.yaml"}, "commit first"),
        ({"EVAL_ALLOW_DIRECT": "0", "MOCK_PROBE_FAIL": "1"}, "env-server probe failed"),
    ],
)
def test_preflight_rejects_invalid_or_unpinned_runs(runner, overrides, error):
    _, launch = runner
    proc = launch(**overrides)
    assert proc.returncode != 0
    assert error in proc.stderr
    assert "MOCK_ROLLOUT=" not in proc.stdout
