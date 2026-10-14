from __future__ import annotations

import ast
import importlib.util
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from lite.core.utils.filters import parse_filter
from lite.infer.cli import make_infer_parser

ROOT = Path(__file__).resolve().parents[2]

COMMAND_DOC_PATHS = (
    "devs/agents/api/AGENTS.md",
    "devs/agents/local/AGENTS.md",
    "devs/agents/local/fara.md",
    "devs/agents/local/mai_ui.md",
    "devs/agents/local/step_gui.md",
    "devs/data/lite.cuagym/AGENTS.md",
    "devs/data/lite.cuagym/gpt5_5/AGENTS.md",
    "devs/data/lite.cuagym/qwen3_8_27b/AGENTS.md",
    "devs/data/lite.cuaworld/AGENTS.md",
    "devs/data/lite.cuaworld/gpt5_5/AGENTS.md",
    "devs/data/lite.cuaworld/qwen3_8_27b/AGENTS.md",
    "devs/data/lite.osworld/AGENTS.md",
    "devs/data/lite.osworld/gpt5_5/AGENTS.md",
    "devs/data/lite.osworld/qwen3_8_27b/AGENTS.md",
    "devs/data/lite.scalecua/AGENTS.md",
    "devs/data/lite.scalecua/gpt5_5/AGENTS.md",
    "devs/data/lite.scalecua/qwen3_8_27b/AGENTS.md",
    "devs/data/webgym/AGENTS.md",
    "devs/data/webgym/gpt5_5/AGENTS.md",
    "devs/envs/AGENTS.md",
    "devs/envs/lite.cuagym/AGENTS.md",
    "devs/envs/lite.osworld/synth/vlc.md",
    "devs/envs/lite.osworld/validate/rollout/plan.md",
    "devs/envs/lite.scalecua/scalecua.md",
    "devs/envs/lite.scalecua/validate/checklist.md",
    "devs/envs/waa.md",
    "lite/data/preproc/AGENTS.md",
    "lite/data/preproc/opencua/AGENTS.md",
    "lite/gym/envs/waa/README.md",
)

EXTERNALLY_COVERED_DOCS = {
    "lite/gym/envs/browsergym/README.md": (
        "tests/gym/envs/browsergym/test_browsergym_isolation.py::"
        "test_browsergym_readme_documents_cache_and_start_exports"
    ),
    "lite/gym/envs/webgym/README.md": (
        "tests/gym/envs/webgym/test_webgym.py::"
        "test_readme_documents_setup_health_and_lifecycle_smoke"
    ),
}

INTENTIONAL_COMMAND_DOC_SKIPS = {
    "devs/envs/lite.scalecua/gaps.md": "historical gap notes, not operator commands",
}

_BASH_BLOCK_RE = re.compile(r"```bash\n(.*?)\n```", re.S)
_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_HEREDOC_RE = re.compile(r"<<-?[\"']?([A-Za-z_][A-Za-z0-9_]*)[\"']?")
_SHELL_COMMAND_STARTS = {
    "bash",
    "cd",
    "echo",
    "env",
    "find",
    "git",
    "nohup",
    "pytest",
    "python",
    "python3",
    "rg",
    "sh",
    "test",
    "uv",
}


@dataclass(frozen=True)
class DocCommand:
    doc: str
    block_idx: int
    command_idx: int
    text: str
    tokens: tuple[str, ...]


def _logical_commands(block: str) -> list[str]:
    commands: list[str] = []
    pending: list[str] = []
    heredoc_end: str | None = None
    awaiting_quoted_continuation = False
    for raw in block.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if heredoc_end is not None:
            if stripped == heredoc_end:
                heredoc_end = None
            continue
        if not stripped or stripped.startswith("#"):
            continue
        if awaiting_quoted_continuation and _looks_like_new_shell_command(stripped):
            pending_text = " ".join(pending)
            raise ValueError(
                f"unterminated quoted shell command before {stripped!r}: {pending_text!r}"
            )
        continued = line.endswith("\\")
        if continued:
            line = line[:-1].rstrip()
        pending.append(line.strip())
        heredoc = _HEREDOC_RE.search(line)
        if heredoc is not None:
            commands.append(" ".join(pending))
            pending = []
            heredoc_end = heredoc.group(1)
            awaiting_quoted_continuation = False
            continue
        if continued:
            continue
        command = " ".join(pending)
        try:
            shlex.split(command, comments=True)
        except ValueError as exc:
            if "No closing quotation" in str(exc):
                awaiting_quoted_continuation = True
                continue
            raise
        commands.append(command)
        pending = []
        awaiting_quoted_continuation = False
    if pending:
        commands.append(" ".join(pending))
    return commands


def _looks_like_new_shell_command(stripped: str) -> bool:
    first = stripped.split(None, 1)[0].rstrip("\"'")
    return first in _SHELL_COMMAND_STARTS or bool(_ENV_ASSIGNMENT_RE.match(first))


def _doc_commands(paths: tuple[str, ...] = COMMAND_DOC_PATHS) -> list[DocCommand]:
    commands: list[DocCommand] = []
    for rel in paths:
        text = (ROOT / rel).read_text()
        for block_idx, block in enumerate(_BASH_BLOCK_RE.findall(text)):
            for command_idx, command_text in enumerate(_logical_commands(block)):
                normalized = _normalize_placeholders(command_text)
                tokens = shlex.split(normalized, comments=True)
                if tokens and tokens[-1] == "&":
                    tokens = tokens[:-1]
                if tokens:
                    commands.append(
                        DocCommand(
                            doc=rel,
                            block_idx=block_idx,
                            command_idx=command_idx,
                            text=command_text,
                            tokens=tuple(tokens),
                        )
                    )
    return commands


def _normalize_placeholders(command: str) -> str:
    agent = "qwen3_vl" if "<AgentName>" in command else "gpt"
    replacements = {
        "<model_id>": "gpt-5.5",
        "<AgentName>": "Qwen/Qwen3-VL-8B-Instruct",
        "<ENV>": "lite.osworld",
        "<env>": "lite.osworld",
        "<agent>": agent,
        "<PORT>": "30100",
        "<TOKEN>": "token",
        "<T>": "token",
        "<task_id>": "task_id",
        "<raw-log-root>": "/tmp/raw-log-root",
        "<annotated-log-root>": "/tmp/annotated-log-root",
        "<frozen-audited-prompt-data.parquet>": "/tmp/prompt.parquet",
        "<software>": "pymol",
        "<plat>": "ubuntu",
        "<task_type>": "use",
        "<Name>": "Lite.OSWorld",
        "<name>": "lite-osworld",
        "$SW": "pymol",
        "$SUB": "synth",
        "$COMMIT": "abc1234",
        "$TRAIN_N": "16139",
        "$CUAGYM_INPUT": "/tmp/prompt.parquet",
        "${CUAGYM_INPUT%.parquet}": "/tmp/prompt",
        "$PLAT": "desktop",
        "<GPU_A>": "0",
        "<GPU_B>": "1",
        "$SGLANG_URL": "http://127.0.0.1:30000",
        "$D": "7",          # webgym difficulty tier
        "$N": "2000",       # webgym per-tier attempt budget
        "$REC": "/tmp/reconstructed",
        "$CUAWORLD_CONCURRENCY": "24",
        "$LOG_ROOT": ".logs/rollout/test",
        "$RESUME_ROOT": ".logs/rollout/test",
        "$ENV_SERVER_PORT": "30100",
        "$ENV_SERVER_TOKEN": "token",
        "$HF_ORG": "cua-lite",
        "$READBACK_ROOT": "/tmp/readback",
        "${READBACK_ROOT}": "/tmp/readback",
        "${CUA_LITE_DATASETS_ROOT}": "/tmp/datasets",
    }
    command = command.replace("$TASK_FILTER", "lambda m: not m.others.get('exclude_reason')")
    command = command.replace("$PWD", str(ROOT))
    for old, new in replacements.items():
        command = command.replace(old, new)
    return command


def _tokens_after(tokens: tuple[str, ...], marker: tuple[str, ...]) -> tuple[str, ...] | None:
    for idx in range(len(tokens) - len(marker) + 1):
        if tokens[idx : idx + len(marker)] == marker:
            rest = list(tokens[idx + len(marker) :])
            for stop in (">", "2>&1", "&", "done"):
                if stop in rest:
                    rest = rest[: rest.index(stop)]
            return tuple(rest)
    return None


def _strip_env_prefix(tokens: tuple[str, ...]) -> tuple[dict[str, str], tuple[str, ...]]:
    env: dict[str, str] = {}
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token == "env":
            idx += 1
            while idx < len(tokens) and tokens[idx] == "-u":
                idx += 2
            continue
        if token == "nohup":
            idx += 1
            continue
        if _ENV_ASSIGNMENT_RE.match(token):
            name, value = token.split("=", 1)
            env[name] = value
            idx += 1
            continue
        break
    return env, tokens[idx:]


def _help_text(kind: str, target: str) -> str:
    if kind == "module":
        argv = [sys.executable, "-m", target, "--help"]
    else:
        argv = [sys.executable, str(ROOT / target), "--help"]
    proc = subprocess.run(
        argv,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    return proc.stdout


def _parse_script_args(script: str, args: tuple[str, ...]):
    spec = importlib.util.spec_from_file_location(
        f"_command_smoke_{Path(script).stem}", ROOT / script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old_argv = sys.argv
    try:
        sys.argv = [script, *args]
        return module._parse_args()
    finally:
        sys.argv = old_argv


def _flags(args: tuple[str, ...]) -> set[str]:
    flags: set[str] = set()
    for arg in args:
        if arg.startswith("--"):
            flags.add(arg.split("=", 1)[0])
        elif re.fullmatch(r"-[A-Za-z]", arg):
            flags.add(arg)
    return flags


def _arg_value(args: tuple[str, ...], flag: str) -> str | None:
    for idx, arg in enumerate(args):
        if arg == flag and idx + 1 < len(args):
            return args[idx + 1]
    return None


def _assert_json_dict(args, attr: str) -> None:
    value = getattr(args, attr)
    if value is not None:
        assert isinstance(value, dict)


def test_command_doc_manifest_matches_current_f11_scope() -> None:
    for rel in COMMAND_DOC_PATHS:
        text = (ROOT / rel).read_text()
        markers = ("uv run", "python ", "pytest", "bash ", "scripts/")
        assert any(marker in text for marker in markers), rel

    for rel, node in EXTERNALLY_COVERED_DOCS.items():
        assert (ROOT / rel).exists()
        assert (ROOT / node.split("::", 1)[0]).exists()

    for rel, reason in INTENTIONAL_COMMAND_DOC_SKIPS.items():
        assert (ROOT / rel).exists()
        assert reason

    assert "tests.md" not in COMMAND_DOC_PATHS
    assert "devs/envs/lite.scalecua/gaps.md" not in COMMAND_DOC_PATHS


def test_multiline_quoted_shell_commands_are_retained() -> None:
    commands = _doc_commands()
    command_texts = {(command.doc, command.text) for command in commands}

    assert any(
        doc == "lite/data/preproc/AGENTS.md"
        and text.startswith('uv run python -c "')
        and "load_dataset" in text
        for doc, text in command_texts
    )
    assert any(
        doc == "lite/data/preproc/opencua/AGENTS.md"
        and text.startswith('uv run python -c "')
        and "pyarrow.parquet" in text
        for doc, text in command_texts
    )
    assert any(
        doc == "devs/envs/lite.osworld/validate/rollout/plan.md"
        and 'find "$LOG_ROOT" -name summary.json' in text
        and 'python3 -c "' in text
        for doc, text in command_texts
    )


def test_unterminated_quoted_shell_command_cannot_swallow_next_command() -> None:
    with pytest.raises(ValueError, match="unterminated quoted shell command"):
        _logical_commands(
            'uv run python -c "unterminated\n'
            'echo ok"\n'
            "uv run python scripts/rollout.py --model-id gpt-5.5 --env-id lite.osworld"
        )


def test_documented_rollout_commands_parse_without_running() -> None:
    commands = _doc_commands()
    rollout_commands = [
        command
        for command in commands
        if _tokens_after(command.tokens, ("uv", "run", "python", "scripts/rollout.py"))
    ]
    assert rollout_commands
    parser = make_infer_parser()
    parsed_docs = set()

    # Mirrors ``LiteCUAMetadata``: ``dims`` is the (platform, task_type) routing
    # pair, and ``others`` is the domain catch-all. Documented filters slice on
    # both, so a stub missing either would let a broken filter through.
    sample = SimpleNamespace(
        dims=("desktop", "use"),
        platform="desktop",
        task_type="use",
        others={
            "exclude_reason": None,
            "episode_return": 1.0,
            "domain": "chrome",
            "sites": [],
            "task_id": "some_task_0001",
        },
    )
    for command in rollout_commands:
        args = _tokens_after(command.tokens, ("uv", "run", "python", "scripts/rollout.py"))
        assert args is not None
        parsed = parser.parse_args(list(args))
        parsed_docs.add(command.doc)
        assert parsed.model_id
        assert parsed.env_id
        if parsed.config_path:
            assert (ROOT / parsed.config_path).exists(), command.text
        for attr in (
            "env_kwargs",
            "agent_kwargs",
            "sampling_kwargs",
            "engine_kwargs",
            "api_kwargs",
        ):
            _assert_json_dict(parsed, attr)
        if parsed.filter_expr:
            assert isinstance(parse_filter(parsed.filter_expr)(sample), bool)
        # ``--prompt-data`` selects tasks from a frozen parquet, so the registry
        # selectors do not apply and ``_resolve_run_tasks`` raises on the pair.
        # argparse accepts them together, so only this check catches a doc that
        # combines them.
        if parsed.prompt_data:
            assert parsed.filter_expr is None, (
                f"{command.doc}: --filter is rejected with --prompt-data "
                f"(lite/infer/rollout.py); filter the frozen parquet instead\n"
                f"{command.text}"
            )
            assert parsed.splits is None, (
                f"{command.doc}: --splits is rejected with --prompt-data\n"
                f"{command.text}"
            )
        # A repeated ``--filter`` is last-wins, so a doc that appends a second one
        # silently drops the first (typically the ``exclude_reason`` gate).
        assert list(args).count("--filter") <= 1, (
            f"{command.doc}: two --filter flags — argparse keeps only the last, "
            f"so the earlier gate is silently dropped\n{command.text}"
        )

    assert "devs/agents/local/fara.md" in parsed_docs
    assert "devs/envs/lite.osworld/validate/rollout/plan.md" in parsed_docs


def test_documented_server_commands_match_cli_help() -> None:
    commands = _doc_commands()
    serve_env = []
    serve_sglang = []
    for command in commands:
        _env, tokens = _strip_env_prefix(command.tokens)
        env_args = _tokens_after(tokens, ("uv", "run", "python", "scripts/serve_env.py"))
        sglang_args = _tokens_after(tokens, ("uv", "run", "python", "scripts/serve_sglang.py"))
        if env_args is not None:
            serve_env.append(env_args)
        if sglang_args is not None:
            serve_sglang.append(sglang_args)

    assert serve_env
    assert serve_sglang
    serve_env_help = _help_text("script", "scripts/serve_env.py")
    for flag in (
        "--port",
        "--env-ids",
        "--token",
        "--idle-ttl-sec",
        "--max-live-envs",
        "--warm-singleton",
        "--timeout-keep-alive",
    ):
        assert flag in serve_env_help
    for args in serve_env:
        _parse_script_args("scripts/serve_env.py", args)

    serve_sglang_help = _help_text("script", "scripts/serve_sglang.py")
    for flag in ("--model-id", "--model-path", "--port", "--host", "--engine-kwargs"):
        assert flag in serve_sglang_help
    for args in serve_sglang:
        _parse_script_args("scripts/serve_sglang.py", args)


@pytest.mark.parametrize(
    "script",
    (
        "lite/gym/envs/lite/cuagym/scripts/install.sh",
        "lite/gym/envs/lite/cuaworld/scripts/install.sh",
        "lite/gym/envs/lite/osworld/scripts/install.sh",
        "lite/gym/envs/lite/scalecua/scripts/install.sh",
        "lite/gym/envs/waa/scripts/install.sh",
        "lite/gym/envs/waa/scripts/cleanup.sh",
        "lite/gym/envs/waa/scripts/uninstall.sh",
        "lite/gym/envs/waa/scripts/utils/prepare_image.sh",
        "lite/gym/envs/lite/scalecua/scripts/utils/tasks.sh",
    ),
)
def test_documented_shell_scripts_exist_and_have_valid_syntax(script: str) -> None:
    path = ROOT / script
    assert path.exists()
    subprocess.run(["bash", "-n", str(path)], cwd=ROOT, check=True, timeout=30)


@pytest.mark.parametrize(
    ("kind", "target", "flags"),
    (
        ("script", "devs/data/lite.osworld/filter.py", ("--log-root", "--out")),
        ("script", "devs/envs/lite.osworld/measure_gap.py", ("--domain",)),
        (
            "script",
            "devs/envs/lite.osworld/validate/rollout/replay_trajectory.py",
            (
                "--rollout",
                "--config-path",
                "--env-kwargs",
                "--max-turns",
                "--inter-turn-sleep",
                "--no-wait",
            ),
        ),
        (
            "script",
            "devs/envs/lite.osworld/validate/oracle/validate.py",
            ("--fixtures", "--filter"),
        ),
        ("script", "devs/envs/lite.scalecua/validate/static.py", ()),
        (
            "script",
            "devs/envs/lite.scalecua/validate/oracle/coverage_inventory.py",
            ("--catalog-dir", "--splits", "--output", "--markdown"),
        ),
        (
            "script",
            "devs/envs/lite.scalecua/validate/oracle/verified_inventory.py",
            ("--catalog-dir", "--splits", "--output", "--markdown", "--require-catalog-complete"),
        ),
        (
            "script",
            "devs/envs/lite.scalecua/validate/oracle/select_fixtures.py",
            ("--catalog-dir", "--output"),
        ),
        (
            "script",
            "lite/gym/envs/waa/scripts/utils/prepare_snapshot.py",
            ("--base-disk", "--assets-dir", "--out"),
        ),
        ("script", "lite/gym/envs/waa/scripts/utils/sync_tasks.py", ("--waa-root",)),
        (
            "script",
            "lite/gym/envs/waa/scripts/utils/smoke_test.py",
            ("--assets-dir",),
        ),
        (
            "script",
            "lite/gym/envs/waa/scripts/utils/verify_all_tasks.py",
            ("--base-disk", "--concurrency", "--attempts", "--resume"),
        ),
        (
            "module",
            "lite.data.hf.stage",
            ("--log-roots", "--name", "--description", "--config-names"),
        ),
        ("module", "lite.data.hf.upload", ("--org", "--private", "--dry-run", "--tag")),
        ("module", "lite.data.hf.download", ("--org", "--revision", "--out")),
        (
            "module",
            "lite.data.hf.unstage",
            ("--dataset", "--log-root", "--splits", "--config-names"),
        ),
        (
            "module",
            "lite.train.export.export_sft",
            (
                "--config",
                "--agent-id",
                "--model-id",
                "--data-paths",
                "--image-root",
                "--head",
                "--filter",
                "--num-proc",
                "-o",
            ),
        ),
    ),
)
def test_documented_python_entrypoints_expose_help(
    kind: str, target: str, flags: tuple[str, ...]
) -> None:
    help_text = _help_text(kind, target)
    for flag in flags:
        assert flag in help_text


@pytest.mark.parametrize(
    "targets",
    (
        ("devs/data/lite.osworld/tests/test_lite_osworld_filter.py",),
        (
            "tests/gym/envs/lite/cuagym/test_cuagym.py",
            "tests/gym/envs/lite/cuagym/test_cuagym_image_contract.py",
        ),
        ("tests/gym/envs/lite/cuagym/test_cuagym_parity.py",),
        (
            "tests/gym/envs/waa/test_waa.py::test_action_translation_coerces_drag_duration",
            "tests/gym/envs/waa/test_waa.py::test_action_translation_rejects_bad_model_durations",
            "tests/gym/matrix/test_agent_env_pair_matrix.py",
        ),
        ("tests/gym/envs/lite/scalecua", "tests/agents/test_registration_complete.py"),
        (
            "tests/data/preproc/cagui/test_preproc_cagui_understanding_entrypoint.py",
            "tests/data/preproc/cagui/test_preproc_cagui_tool_io_contract.py::test_cagui_post_action_screenshot_is_tool",
            "tests/data/preproc/cagui/test_preproc_cagui_tool_io_contract.py::test_cagui_status_task_complete_becomes_done",
            "tests/data/preproc/cagui/test_preproc_cagui_tool_io_contract.py::test_cagui_status_task_impossible_moves_to_metadata_others",
        ),
        (
            "tests/data/preproc/opencua/test_preproc_opencua_win_mac_entrypoint.py",
            "tests/data/preproc/opencua/test_preproc_opencua_tool_io_contract.py::test_opencua_terminal_only_terminate_becomes_done_and_tool_result",
            "tests/data/preproc/opencua/test_preproc_opencua_tool_io_contract.py::test_opencua_non_terminating_final_keeps_final_action",
            "tests/data/preproc/opencua/test_preproc_opencua_tool_io_contract.py::test_opencua_failure_terminate_moves_to_metadata_others",
            "tests/data/preproc/opencua/test_preproc_opencua_tool_io_contract.py::test_opencua_iter_examples_head_stops_at_trajectory_boundary",
        ),
    ),
)
def test_documented_pytest_targets_collect(targets: tuple[str, ...]) -> None:
    marker_args = ["-m", "live"] if any("test_cuagym_parity.py" in t for t in targets) else []
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            *marker_args,
            *targets,
        ],
        cwd=ROOT,
        check=True,
        timeout=60,
    )


def test_documented_paths_configs_and_filters_resolve() -> None:
    config_paths = {
        "scripts/configs/gpt/default/lite.osworld.yaml",
        "scripts/configs/gpt/recipes/collect/lite.osworld.yaml",
        "scripts/configs/gpt/default/lite.cuaworld.yaml",
        "scripts/configs/gpt/recipes/collect/lite.cuaworld.yaml",
        "scripts/configs/gpt/recipes/collect/lite.cuagym.yaml",
        "scripts/configs/gpt/recipes/collect/lite.scalecua.yaml",
        "scripts/configs/qwen3_vl/default/lite.osworld.yaml",
        "scripts/configs/qwen3_5/default/lite.cuagym.yaml",
        "scripts/configs/qwen3_5/default/lite.cuaworld.yaml",
        "scripts/configs/qwen3_5/default/lite.osworld.yaml",
        "scripts/configs/mai_ui/default/androidworld.yaml",
        "scripts/configs/step_gui/default/androidworld.yaml",
    }
    for rel in config_paths:
        assert (ROOT / rel).exists(), rel

    fara_defaults = {
        path.as_posix() for path in (ROOT / "scripts/configs/fara/default").rglob("*.yaml")
    }
    for suffix in (
        "webgym.yaml",
        "online_mind2web.yaml",
        "osworld_g.yaml",
        "screenspot_pro.yaml",
        "browsergym.miniwob/default.yaml",
        "browsergym.webarena/default.yaml",
        "browsergym.visualwebarena/goal_image.yaml",
        "webharbor.webvoyager/default.yaml",
        "webharbor.webvoyager/som.yaml",
    ):
        assert any(path.endswith(suffix) for path in fara_defaults), suffix

    metadata = SimpleNamespace(
        others={
            "exclude_reason": None,
            "episode_return": 1.0,
            "domain": "chrome",
            "sites": [],
        }
    )
    for expr in (
        "lambda m: not m.others.get('exclude_reason')",
        "lambda m: not m.others.get('exclude_reason') and "
        "(m.others.get('episode_return') or 0) > 0.5",
        "lambda m: m.others.get('domain') == 'chrome'",
        "lambda m: 'map' not in m.others.get('sites', [])",
    ):
        assert isinstance(parse_filter(expr)(metadata), bool)


def test_waa_readme_command_examples_are_smoked() -> None:
    text = (ROOT / "lite/gym/envs/waa/README.md").read_text()
    snippets = (
        "uv run --no-sync bash lite/gym/envs/waa/scripts/install.sh",
        "uv run --no-sync bash lite/gym/envs/waa/scripts/install.sh rebuild",
        "uv run --no-sync bash lite/gym/envs/waa/scripts/install.sh pull",
        "uv run --no-sync bash lite/gym/envs/waa/scripts/install.sh status",
        "uv run python scripts/serve_env.py   # serves all envs on :30100",
        "uv run python scripts/serve_env.py --env-ids waa",
        "uv run --no-sync bash lite/gym/envs/waa/scripts/cleanup.sh",
        "uv run --no-sync bash lite/gym/envs/waa/scripts/uninstall.sh",
        "uv run python lite/gym/envs/waa/scripts/utils/prepare_snapshot.py",
        "bash lite/gym/envs/waa/scripts/utils/prepare_image.sh",
        "lite/gym/envs/waa/scripts/utils/sync_tasks.py --waa-root ../WindowsAgentArena",
        "gym.registry.task_ids('waa', split='eval')",
        "gym.registry.task_ids('waa', split='eval_noctxt')",
        "WAA_DOCKER=1 uv run pytest tests/gym/envs/waa/test_waa.py -m live -n0",
        "uv run python lite/gym/envs/waa/scripts/utils/smoke_test.py",
        "uv run python lite/gym/envs/waa/scripts/utils/verify_all_tasks.py",
    )
    for snippet in snippets:
        assert snippet in text


def test_waa_direct_and_server_smoke_snippets_are_parseable_python() -> None:
    text = (ROOT / "devs/envs/waa.md").read_text()
    bodies = re.findall(r"<<'PY'\n(.*?)\nPY", text, flags=re.S)
    assert len(bodies) == 2
    for body in bodies:
        ast.parse(body)
        assert 'gym.registry.task_ids("waa", split="eval")' in body
        assert 'gym.make(f"waa@{task_id}", max_steps=5)' in body
        assert 'make_tool_call("computer"' in body
        assert '"action": "wait"' in body
        assert '"action": "hold_key"' in body
        assert '"action": "drag"' in body
        assert "call_waa_action_smoke" in body
        assert "await env.close()" in body

def test_documented_code_fences_are_balanced() -> None:
    """An unclosed ```fence swallows the prose after it into the code block, and
    the shell comments that follow render as headings. Both are invisible to the
    command extractor, which reads fenced blocks only — so nothing else catches
    it. Cheap to check, and it has bitten this manifest before."""
    unbalanced = []
    for rel in COMMAND_DOC_PATHS:
        text = (ROOT / rel).read_text()
        opens = sum(1 for line in text.splitlines() if line.startswith("```"))
        if opens % 2:
            unbalanced.append(f"{rel} ({opens} fence markers)")
    assert not unbalanced, "unclosed code fence:\n  " + "\n  ".join(unbalanced)
