"""Static gates for rollout source config surfaces.

These tests deliberately avoid GPU servers, Docker, and real env-server calls.
They pin the checked-in rollout rows and tool-surface YAML shape consumed by
``scripts/rollout.py`` and the train launchers.
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

from lite.core import LiteCUAMetadata
from lite.core.tools.extra_tools import LiteBrowserNavToolSet, LiteFinishToolSet
from lite.utils.config import load_config
from lite.utils.path import project_root

ROOT = project_root()


def _cfg(rel: str) -> dict:
    return load_config(ROOT / rel)


def _agent_kwargs(cfg: dict) -> dict:
    return cfg.get("agent_kwargs") or {}


def _protocol_kwargs(cfg: dict) -> dict:
    return _agent_kwargs(cfg).get("protocol_kwargs") or {}


def _script_config_paths() -> list[Path]:
    return sorted((ROOT / "scripts/configs").rglob("*.yaml"))


def _browser_rollout_configs() -> list[tuple[str, dict]]:
    configs: list[tuple[str, dict]] = []
    for path in _script_config_paths():
        cfg = load_config(path)
        env_id = cfg.get("env_id")
        if (
            isinstance(env_id, str)
            and (
                env_id.startswith("browsergym.")
                or env_id in {"online_mind2web", "webharbor.webvoyager"}
            )
        ):
            configs.append((str(path.relative_to(ROOT)), cfg))
    assert configs, "no browser rollout configs found"
    return configs


def _is_browser_vision_coord_default(rel: str, cfg: dict) -> bool:
    path = Path(rel)
    if path.name not in {"default.yaml", "goal_image.yaml", "online_mind2web.yaml"}:
        return False

    env_id = cfg["env_id"]
    env_kwargs = cfg.get("env_kwargs") or {}
    if env_id == "online_mind2web":
        return True
    if env_id == "webharbor.webvoyager":
        return env_kwargs.get("use_som") is not True
    if env_id.startswith("browsergym."):
        return (
            env_kwargs.get("use_screenshot", True) is True
            and env_kwargs.get("use_ax_tree", False) is False
            and env_kwargs.get("use_som", False) is False
            and "coord" in set(env_kwargs.get("action_subsets", ("coord",)))
        )
    return False


def _is_browser_explicit_text_or_som(rel: str, cfg: dict) -> bool:
    path = Path(rel)
    env_kwargs = cfg.get("env_kwargs") or {}
    return (
        path.name in {"text_only.yaml", "mixed.yaml", "som.yaml"}
        or env_kwargs.get("use_ax_tree") is True
        or env_kwargs.get("use_som") is True
    )


def test_browser_vision_coord_defaults_disable_page_context_text_at_env_boundary() -> None:
    configs = [
        (rel, cfg) for rel, cfg in _browser_rollout_configs()
        if _is_browser_vision_coord_default(rel, cfg)
    ]
    assert configs, "no browser vision+coord default configs found"

    for rel, cfg in configs:
        env_kwargs = cfg["env_kwargs"]

        assert env_kwargs["include_page_context_text"] is False, rel

        if cfg["env_id"].startswith("browsergym."):
            assert env_kwargs["use_screenshot"] is True, rel
            assert env_kwargs["use_ax_tree"] is False, rel


def test_browser_rollout_configs_do_not_disable_role_tool_text_in_protocol() -> None:
    for rel, cfg in _browser_rollout_configs():
        assert _protocol_kwargs(cfg).get("keep_text_with_images") is not False, rel


def test_browser_explicit_text_and_som_modes_keep_page_context_available() -> None:
    configs = [
        (rel, cfg) for rel, cfg in _browser_rollout_configs()
        if _is_browser_explicit_text_or_som(rel, cfg)
    ]
    assert configs, "no explicit browser text/SoM configs found"

    for rel, cfg in configs:
        env_kwargs = cfg["env_kwargs"]
        assert env_kwargs["include_page_context_text"] is True, rel


def test_lite_osworld_extra_tool_rollout_gates() -> None:
    qwen_config_paths = [
        "scripts/configs/qwen3_vl/default/lite.osworld.yaml",
        "scripts/configs/qwen3_vl/default/lite.osworld.bash.yaml",
        "scripts/configs/qwen3_vl/compact/lite.osworld.yaml",
        "scripts/configs/qwen3_5/default/lite.osworld.yaml",
        "scripts/configs/qwen3_5/default/lite.osworld.bash.yaml",
        "scripts/configs/qwen3_5/compact/lite.osworld.yaml",
    ]
    for rel in qwen_config_paths:
        env_kwargs = _cfg(rel)["env_kwargs"]
        assert "extra_tools" in env_kwargs, rel
        assert {"response", "terminate"} <= set(env_kwargs["extra_tools"]), rel

    for rel in (
        "scripts/configs/qwen3_vl/default/lite.osworld.bash.yaml",
        "scripts/configs/qwen3_5/default/lite.osworld.bash.yaml",
    ):
        assert "bash" in _cfg(rel)["env_kwargs"]["extra_tools"], rel

    gpt_env_kwargs = _cfg("scripts/configs/gpt/default/lite.osworld.yaml")["env_kwargs"]
    assert not (set(gpt_env_kwargs.get("extra_tools") or []) & {"response", "terminate"})


def test_lite_osworld_rollout_configs_pin_static_contract() -> None:
    gpt = _cfg("scripts/configs/gpt/default/lite.osworld.yaml")
    qwen3_vl_default = _cfg("scripts/configs/qwen3_vl/default/lite.osworld.yaml")
    qwen3_vl_compact = _cfg("scripts/configs/qwen3_vl/compact/lite.osworld.yaml")
    qwen3_5_default = _cfg("scripts/configs/qwen3_5/default/lite.osworld.yaml")
    qwen3_5_compact = _cfg("scripts/configs/qwen3_5/compact/lite.osworld.yaml")

    configs = [
        gpt,
        qwen3_vl_default,
        qwen3_vl_compact,
        qwen3_5_default,
        qwen3_5_compact,
    ]
    assert {cfg["env_id"] for cfg in configs} == {"lite.osworld"}
    assert {cfg["env_kwargs"]["max_steps"] for cfg in configs} == {30}

    assert gpt["agent_id"] == "gpt"
    assert _agent_kwargs(gpt)["api_kwargs"]["max_output_tokens"] == 4096
    assert "resolution" not in _agent_kwargs(gpt)

    assert qwen3_vl_default["agent_id"] == "qwen3_vl"
    assert _agent_kwargs(qwen3_vl_default)["sampling_kwargs"]["temperature"] == 0.0
    assert "resolution" not in _agent_kwargs(qwen3_vl_default)
    assert _protocol_kwargs(qwen3_vl_default) == {}

    assert _agent_kwargs(qwen3_vl_compact)["resolution"] == [1280, 720]
    assert _protocol_kwargs(qwen3_vl_compact)["full_history_size"] == 1

    assert qwen3_5_default["agent_id"] == "qwen3_5"
    assert _agent_kwargs(qwen3_5_default)["sampling_kwargs"]["temperature"] == 0.0
    assert "resolution" not in _agent_kwargs(qwen3_5_default)
    assert _protocol_kwargs(qwen3_5_default) == {}

    assert _agent_kwargs(qwen3_5_compact)["resolution"] == [1280, 720]
    assert _protocol_kwargs(qwen3_5_compact)["history_n"] == 1



_ENVS_DIR = ROOT / "lite/gym/envs"
_FINISH_TOOL_NAMES = LiteFinishToolSet.get_tool_names()


def _env_main_py(env_id: str) -> Path | None:
    """``env_id`` → its ``main.py``, using the registry's own resolution rule.

    ``lite.gym.registry._import_env_locked``: dotted names map to nested
    directories (``lite.osworld`` → ``envs/lite/osworld/main.py``), falling back
    to the umbrella parent when the sub-env has no directory of its own
    (``browsergym.miniwob`` → ``envs/browsergym/main.py``).
    """
    parts = env_id.split(".")
    while parts:
        candidate = _ENVS_DIR.joinpath(*parts) / "main.py"
        if candidate.is_file():
            return candidate
        parts = parts[:-1]
    return None


def _literal_task_type(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Attribute) and node.attr == "value":
        return _literal_task_type(node.value)
    if isinstance(node, ast.Attribute):
        return getattr(LiteCUAMetadata.TaskType, node.attr).value
    return None


def _declared_task_types(main_py: Path) -> set[str]:
    """Every literal CUA task type the env module hands to ``LiteCUAMetadata``.

    Read from source (``ast``) rather than by importing: most envs raise
    ``EnvDepsMissingError`` at import when their heavy backend (browsergym,
    android_world, …) isn't installed, and this gate must cover every
    committed row regardless of which extras the test venv has.
    """
    declared: set[str] = set()
    for node in ast.walk(ast.parse(main_py.read_text())):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != "dims":
                continue
            value = kw.value
            if not isinstance(value, (ast.Tuple, ast.List)) or len(value.elts) != 2:
                continue
            task_type = _literal_task_type(value.elts[1])
            if task_type is not None:
                declared.add(task_type)
    return declared


def _is_multi_step_env(main_py: Path) -> bool:
    """True when the env serves ``task_type="use"`` tasks (multi-step episodes).

    An env that never names ``task_type`` inherits ``LiteCUAMetadata``'s default,
    which IS ``use`` — so "declares nothing" must read as multi-step, not as
    "skip me". Only envs that exclusively declare ``grounding.*`` /
    ``understanding`` (single-turn scoring, no episode to end) are exempt.
    """
    declared = _declared_task_types(main_py)
    return not declared or LiteCUAMetadata.TaskType.USE.value in declared


def test_every_multi_step_config_row_resolves_a_terminal_channel() -> None:
    """A ``use`` row must have SOME way for the agent to end the episode.

    Two legitimate terminal channels exist, and a row must resolve one:

    1. A canonical finish tool (``response`` / ``terminate``) in
       ``env_kwargs.extra_tools`` — the advertised, model-visible channel.
    2. The pure-text final: ``tool_calls == []`` plus visible text, which the
       agent loop turns into an INTERNAL finish call
       (``make_no_tool_call_final_actions``). That call carries no persisted ``id``,
       is not model-emitted, and is handled below the advertised tool surface. A row
       that advertises NO standalone tool surface at all (``extra_tools``
       omitted or ``[]``) is deliberately driving on this channel — e.g. the
       GPT/Claude computer-use rows for ``lite.osworld`` / ``osworld`` /
       ``cua.bench``, whose native loops naturally end a turn with prose.

    The broken shape this gate exists for is the one in between: a row that DOES
    advertise a standalone tool surface (so the model answers in tool calls) but
    omits every finish tool. A model-emitted ``terminate``/``response`` is then
    env-owned unsupported feedback, and the episode can only end by max-step
    truncation. Both MiniWoB ``text_only``
    families shipped in exactly that state.

    Derived, not enumerated: rows come from the ``scripts/configs`` tree and
    ``use``-ness from the env module's own ``task_type``, so a new bench/agent
    directory is covered the day it lands.
    """
    multi_step: dict[str, bool] = {}
    offenders: list[str] = []
    checked = 0

    for path in sorted((ROOT / "scripts/configs").rglob("*.yaml")):
        cfg = load_config(path)
        env_id = cfg.get("env_id")
        if not env_id:
            continue  # recipe fragment (e.g. sft/default.yaml), not a rollout row
        if env_id not in multi_step:
            main_py = _env_main_py(env_id)
            assert main_py is not None, f"{path}: env_id {env_id!r} has no env module"
            multi_step[env_id] = _is_multi_step_env(main_py)
        if not multi_step[env_id]:
            continue  # grounding / understanding — single-step, nothing to end

        checked += 1
        extra_tools = (cfg.get("env_kwargs") or {}).get("extra_tools") or []
        if extra_tools and not set(extra_tools) & _FINISH_TOOL_NAMES:
            offenders.append(
                f"{path.relative_to(ROOT)} (env_id={env_id}, extra_tools={extra_tools})"
            )

    assert checked > 50, f"gate walked too few rows ({checked}) — path/env resolution broke"
    assert not offenders, (
        "multi-step (task_type=use) rows advertise a standalone tool surface with no "
        "finish tool, so a model-emitted terminate/response is env-owned "
        "unsupported feedback and the episode can only end by truncation:\n  "
        + "\n  ".join(offenders)
    )


_AST_CACHE: dict[Path, ast.AST] = {}


def _env_source_files(env_id: str) -> list[Path]:
    """The env's action/eval source: its ``main.py`` + its in-container server.

    Container-backed envs (mobilegym, webharbor.webvoyager, online_mind2web)
    execute the agent's tool calls inside ``docker/server.py``, so the
    ``response`` branch lives THERE, not in the host ``main.py``. Reading only
    ``main.py`` would silently classify them as having no answer channel.
    """
    main_py = _env_main_py(env_id)
    if main_py is None:
        return []
    files = [main_py]
    server = main_py.parent / "docker" / "server.py"
    if server.is_file():
        files.append(server)
    return files


def _parsed(path: Path) -> ast.AST:
    if path not in _AST_CACHE:
        _AST_CACHE[path] = ast.parse(path.read_text())
    return _AST_CACHE[path]


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _functions(tree: ast.AST) -> list[ast.AST]:
    return [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _branches_on_response(func: ast.AST) -> bool:
    """True when the function dispatches on the canonical ``response`` call."""
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Eq)
            and any(
                isinstance(c, ast.Constant) and c.value == "response"
                for c in node.comparators
            )
        ):
            return True
    return False


def _text_payload_reads(func: ast.AST) -> list[ast.AST]:
    """Every read of the call's ``"text"`` argument (``args.get("text")`` / ``args["text"]``)."""
    reads: list[ast.AST] = []
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "text"
        ):
            reads.append(node)
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == "text"
        ):
            reads.append(node)
    return reads


def _is_infeasible_marker_scan(read: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """True when the text read is only tested for an ``[infeasible]`` marker.

    ``osworld`` / ``osworld_2`` / ``waa`` all spell it
    ``"[infeasible]" in str(args.get("text", "")).lower()`` — the string is
    scanned for a flag and then DISCARDED, so nothing about it reaches reward.
    Walk up through the ``str(...)`` / ``.lower()`` wrappers to the enclosing
    ``in`` comparison; anything else (an assignment, a call argument, an f-string)
    means the payload itself is being consumed.
    """
    node = read
    while True:
        parent = parents.get(node)
        if parent is None:
            return False
        if isinstance(parent, ast.Compare):
            return (
                any(isinstance(op, (ast.In, ast.NotIn)) for op in parent.ops)
                and isinstance(parent.left, ast.Constant)
                and isinstance(parent.left.value, str)
                and "infeasible" in parent.left.value.lower()
            )
        if not isinstance(parent, (ast.Call, ast.Attribute)):
            return False
        node = parent


def _forwards_response_text(tree: ast.AST) -> bool:
    """True when the module consumes a ``response`` call's TEXT as data."""
    parents = _parent_map(tree)
    for func in _functions(tree):
        if not _branches_on_response(func):
            continue
        for read in _text_payload_reads(func):
            if not _is_infeasible_marker_scan(read, parents):
                return True
    return False


def _answer_eval_sub_envs(tree: ast.AST) -> set[str]:
    """Sub-benchmark names an umbrella module ties to reference-ANSWER eval.

    One module can serve several leaf ``env_id``s whose reward paths differ, and
    the shared ``response`` handler cannot tell them apart: ``browsergym``'s
    ``main.py`` maps ``response`` → ``send_msg_to_user`` for MiniWoB, WebArena
    and VisualWebArena alike, yet only the latter two score a string match
    against that message (MiniWoB's reward comes from the DOM episode).

    The module says so itself: ``_wa_vwa_task_facts`` reads the WA/VWA eval
    spec's ``reference_answers`` key (the ground-truth ANSWER strings) and is
    gated on ``benchmark not in ("webarena", "visualwebarena")``. So: any
    function that touches ``reference_answers`` scopes the answer channel to the
    benchmark names it compares against. Empty set → the module makes no such
    distinction and every leaf it serves inherits the module's classification.
    """
    scoped: set[str] = set()
    for func in _functions(tree):
        constants = {
            n.value for n in ast.walk(func)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        }
        if "reference_answers" not in constants:
            continue
        for node in ast.walk(func):
            if not isinstance(node, ast.Compare):
                continue
            if not any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops):
                continue
            for comparator in node.comparators:
                if not isinstance(comparator, (ast.Tuple, ast.List, ast.Set)):
                    continue
                if comparator.elts and all(
                    isinstance(e, ast.Constant) and isinstance(e.value, str)
                    for e in comparator.elts
                ):
                    scoped.update(e.value for e in comparator.elts)
    return scoped


def _is_answer_requiring(env_id: str) -> bool:
    """True when the env's reward depends on a string the agent returns.

    Derived from the env's own source, never from a config list:

    1. Does the action pipeline consume the ``response`` payload as DATA?
       (``_forwards_response_text`` — androidworld's ``interaction_cache``,
       mobileworld's ``json_action.answer``, mobilegym's ``ActionType.ANSWER``,
       webvoyager's ``inst.answer``, online_mind2web's ``ANSWER:`` action
       string, webgym's ``_agent_response``, browsergym's
       ``send_msg_to_user(text)``.) Envs that only scan the text for an
       ``[infeasible]`` marker (osworld / osworld_2 / waa) and envs with no
       ``response`` branch at all (lite.*, cua.bench, osworld_g, screenspot_pro)
       fail here.
    2. For an umbrella module serving several leaves, is THIS leaf one of the
       ones the module ties to reference-answer eval? (``_answer_eval_sub_envs``
       — keeps browsergym.webarena / .visualwebarena, drops browsergym.miniwob.)
    """
    files = _env_source_files(env_id)
    if not any(_forwards_response_text(_parsed(f)) for f in files):
        return False
    scoped: set[str] = set()
    for f in files:
        scoped |= _answer_eval_sub_envs(_parsed(f))
    if not scoped:
        return True
    return bool(set(env_id.split(".")) & scoped)


def _has_answer_gated_eval(env_id: str) -> bool:
    """True when EVERY task of the env is scored through the agent's answer.

    The QA-purity question ("is this benchmark 100% QA?") is a property of the
    task DATA, which is not always in the repo (webgym's tasks come from a
    HuggingFace dataset). What IS in the source — and is the property that
    actually matters — is whether the env's evaluator refuses to score without
    an agent answer. An env that branches its eval on the stored agent answer
    has no reward-bearing non-answer exit, so ``response`` doubles as the
    "done" signal and a separate ``terminate`` is not needed:

      * ``webgym``     — ``main.py``: ``if self._agent_response:`` →
        ``_vlm_evaluate()``, else ``_blocking_only_evaluate()`` (reward 0).
        Confirmed against the task data: the 1167 shipped tasks carry NO
        ``reference_answer``/``definite_answer``; the rubric judge is fed the
        agent's final response as ``action={"key": "answer", ...}``.
      * ``webharbor.webvoyager`` — ``main.py``: ``if not self._agent_response:``
        → ``eval_skip_reason="no_agent_response"``, reward ``None``. Confirmed
        against ``data/tasks.json``: 643/643 tasks ship a non-empty
        ``reference_answer`` — a genuinely 100%-QA benchmark.

    Everything else (``online_mind2web``'s WebJudge over screenshots + action
    strings, browsergym's benchmark reward, androidworld/androidlab/mobile*'s
    state checks) scores state or trajectory, so its non-QA tasks need
    ``terminate`` to say "done"/"can't" without burning to ``max_steps``.

    ``mobilegym``'s ``self._answer_fields`` is deliberately NOT matched (the
    attribute must name the AGENT's answer): it is an answer-SHEET flag on the
    task, not a record of what the agent returned.
    """
    main_py = _env_main_py(env_id)
    if main_py is None:
        return False
    for node in ast.walk(_parsed(main_py)):
        test = getattr(node, "test", None) if isinstance(
            node, (ast.If, ast.IfExp, ast.While)
        ) else None
        if test is None:
            continue
        for sub in ast.walk(test):
            if not isinstance(sub, ast.Attribute):
                continue
            attr = sub.attr.lower()
            if "agent" in attr and ("answer" in attr or "response" in attr):
                return True
    return False


def _multi_step_rollout_rows() -> list[tuple[Path, str, list[str]]]:
    """``(config path, env_id, extra_tools)`` for every committed multi-step row."""
    multi_step: dict[str, bool] = {}
    rows: list[tuple[Path, str, list[str]]] = []
    for path in sorted((ROOT / "scripts/configs").rglob("*.yaml")):
        cfg = load_config(path)
        env_id = cfg.get("env_id")
        if not env_id:
            continue  # recipe fragment (e.g. sft/default.yaml), not a rollout row
        if env_id not in multi_step:
            main_py = _env_main_py(env_id)
            assert main_py is not None, f"{path}: env_id {env_id!r} has no env module"
            multi_step[env_id] = _is_multi_step_env(main_py)
        if not multi_step[env_id]:
            continue  # grounding / understanding — single-step, nothing to end
        extra_tools = (cfg.get("env_kwargs") or {}).get("extra_tools") or []
        rows.append((path, env_id, list(extra_tools)))
    return rows


def test_every_answer_requiring_config_row_advertises_response() -> None:
    """An answer-scored env MUST advertise ``response``, or its reward is unreachable.

    ``response`` is the only canonical call an env turns into an ANSWER
    (browsergym: ``send_msg_to_user(text)``, which WA/VWA string-match against;
    webgym/webvoyager/online_mind2web/androidworld/…: stored as the agent's
    answer). ``terminate`` is not a substitute — browsergym degrades it to
    ``send_msg_to_user(reason or "Task completed")``, i.e. a status, not an
    answer.

    Omitting it hides the structured answer tool from the model-visible schema.
    Content-only final prose has a runtime ``response(text=...)`` backstop, but
    answer-scored rows still need the explicit `response` tool so a model can
    submit answers through the normal tool surface.
    """
    answer_requiring: dict[str, bool] = {}
    offenders: list[str] = []
    checked = 0

    for path, env_id, extra_tools in _multi_step_rollout_rows():
        if env_id not in answer_requiring:
            answer_requiring[env_id] = _is_answer_requiring(env_id)
        if not answer_requiring[env_id]:
            continue
        checked += 1
        if "response" not in extra_tools:
            offenders.append(
                f"{path.relative_to(ROOT)} (env_id={env_id}, extra_tools={extra_tools})"
            )

    assert checked > 50, f"gate walked too few rows ({checked}) — path/env resolution broke"
    assert not offenders, (
        "rows on answer-scored envs omit `response` from env_kwargs.extra_tools, so the "
        "model-visible structured answer tool is absent:\n  "
        + "\n  ".join(offenders)
    )


def test_every_non_qa_config_row_advertises_terminate() -> None:
    """A row on an env that is not answer-gated MUST advertise ``terminate``.

    ``response`` alone is enough only when every task is scored through the
    answer (``_has_answer_gated_eval``: webgym, webharbor.webvoyager). Anywhere
    else the benchmark has tasks with nothing to return — Online-Mind2Web's 300
    tasks ship no reference answer and are mostly imperatives, scored by
    WebJudge on screenshots + action strings — and such a task still needs to
    say "done" or "I can't", or it can only burn to ``max_steps``.

    Same carve-out as the terminal-channel gate: a row that advertises NO
    standalone tool surface (``extra_tools`` omitted or ``[]``) is deliberately
    on the pure-text-final channel, whose internal ``response`` is runtime-only
    and not part of the model-visible schema.
    """
    answer_gated: dict[str, bool] = {}
    offenders: list[str] = []
    checked = 0

    for path, env_id, extra_tools in _multi_step_rollout_rows():
        if not extra_tools:
            continue  # pure-text-final channel — internal response, no schema needed
        if env_id not in answer_gated:
            answer_gated[env_id] = _has_answer_gated_eval(env_id)
        if answer_gated[env_id]:
            continue  # 100% answer-scored — `response` IS the done signal
        checked += 1
        if "terminate" not in extra_tools:
            offenders.append(
                f"{path.relative_to(ROOT)} (env_id={env_id}, extra_tools={extra_tools})"
            )

    assert checked > 50, f"gate walked too few rows ({checked}) — path/env resolution broke"
    assert not offenders, (
        "rows on envs that are NOT scored purely through the agent's answer omit "
        "`terminate`, so a task with no answer to return can only end by max-step "
        "truncation:\n  " + "\n  ".join(offenders)
    )


def test_lite_osworld_bash_config_is_a_live_row() -> None:
    """The bash row exists and every ``lite.osworld.bash.yaml`` really enables bash.

    This used to assert the file was ABSENT: ``extra_tools: ["bash"]`` did not
    resolve into ``metadata.extra_tool_schemas``, so the yaml would have been
    dead on startup. That blocker is gone (``SandboxBaseEnv`` resolves the
    canonical ``bash`` schema and merges it in ``_runtime_metadata``), so the
    gate flips: a checked-in bash row must actually request bash, on an env that
    can honor it, with a terminal channel. Resolution itself is covered
    end-to-end by ``tests/gym/sandbox/test_bash_extra_tool.py``.
    """
    search_roots = [
        ROOT / "scripts/configs",
        ROOT / "examples",
    ]
    matches = [p for root in search_roots for p in root.glob("**/lite.osworld.bash.yaml")]
    assert matches, "no lite.osworld bash rollout row is checked in"
    for path in matches:
        cfg = load_config(path)
        extra_tools = (cfg.get("env_kwargs") or {}).get("extra_tools") or []
        assert cfg["env_id"] == "lite.osworld", path
        assert "bash" in extra_tools, f"{path}: bash row does not request bash"
        # Advertising a standalone surface costs the pure-text-final channel.
        assert set(extra_tools) & _FINISH_TOOL_NAMES, path


# ---------------------------------------------------------------------------
# tool I/O config drift guard — valid_actions is GUI-inner only
# ---------------------------------------------------------------------------

_NON_GUI_VALID_ACTIONS = {
    # Finish/intrinsic.
    "response",
    "terminate",
    "done",
    "finish",
    "submit",
    # Browser/mobile/env extra tools.
    "goto",
    "back",
    "forward",
    "go_back",
    "navigate",
    "open_app",
    "ask_user",
    "web_search",
    "pause_and_memorize_fact",
}

_AGENT_CONFIG_ROOTS = (
    Path("scripts/configs"),
    Path("examples/lite/v1/configs"),
    Path("devs/exps/train/desktop/configs"),
    Path("devs/exps/train/mobile/configs"),
)

_TOOL_SURFACE_AGENT_KWARGS = {"extra_tools", "extra_tool_schemas", "valid_actions", "others"}
_TOOL_SURFACE_SOURCE_KEYS = _TOOL_SURFACE_AGENT_KWARGS
_NAV_TOOL_NAMES = LiteBrowserNavToolSet.get_tool_names() | {
    # Env/backend aliases that must also stay out of valid_actions.
    "go_back",
    "go_forward",
    "navigate",
    "refresh",
    "scroll_to",
    "tab_focus",
    "tab_close",
}

#: The ``mobile.use`` campaign, the desktop set's sibling. Only the image budget
#: varies -- one resolution, and no turn-capped `h1` arm -- so the profile names carry
#: just that. `open_app` joins the finish pair for the same reason the pair is there at
#: all: mobilegym advertises it as a standalone extra tool, and an undeclared one is
#: NOOPed, so a profile that omitted it would be trained and scored on a surface that
#: cannot launch an app.
_MOBILE_USE_RECIPE_CONFIGS = {
    f"devs/exps/train/mobile/configs/qwen3_5/mobile.use.{_variant}.yaml"
    for _variant in ("i1", "i1.reasoning", "i4", "i4.reasoning")
}

_EXPECTED_OPEN_APP_CONFIGS = {
    "scripts/configs/claude/default/androidlab.yaml",
    "scripts/configs/claude/default/androidworld.yaml",
    "scripts/configs/claude/default/mobilegym.yaml",
    "scripts/configs/claude/default/mobileworld.yaml",
    "scripts/configs/gpt/default/androidlab.yaml",
    "scripts/configs/gpt/default/androidworld.yaml",
    "scripts/configs/gpt/default/mobilegym.yaml",
    "scripts/configs/gpt/default/mobileworld.yaml",
    "scripts/configs/gpt/recipes/collect/mobilegym.yaml",
    "scripts/configs/mai_ui/compact/androidworld.yaml",
    "scripts/configs/mai_ui/compact/mobilegym.yaml",
    "scripts/configs/mai_ui/default/androidlab.yaml",
    "scripts/configs/mai_ui/default/androidworld.yaml",
    "scripts/configs/mai_ui/default/mobilegym.yaml",
    "scripts/configs/mai_ui/default/mobileworld.yaml",
    "scripts/configs/qwen3_5/compact/androidworld.yaml",
    "scripts/configs/qwen3_8/compact/androidworld.yaml",
    "scripts/configs/qwen3_5/compact/mobilegym.yaml",
    "scripts/configs/qwen3_8/compact/mobilegym.yaml",
    "scripts/configs/qwen3_5/default/androidlab.yaml",
    "scripts/configs/qwen3_8/default/androidlab.yaml",
    "scripts/configs/qwen3_5/default/androidworld.yaml",
    "scripts/configs/qwen3_8/default/androidworld.yaml",
    "scripts/configs/qwen3_5/default/mobilegym.yaml",
    "scripts/configs/qwen3_8/default/mobilegym.yaml",
    "scripts/configs/qwen3_5/default/mobileworld.yaml",
    "scripts/configs/qwen3_8/default/mobileworld.yaml",
    "scripts/configs/qwen3_vl/compact/androidworld.yaml",
    "scripts/configs/qwen3_vl/compact/mobilegym.yaml",
    "scripts/configs/qwen3_vl/default/androidlab.yaml",
    "scripts/configs/qwen3_vl/default/androidworld.yaml",
    "scripts/configs/qwen3_vl/default/mobilegym.yaml",
    "scripts/configs/qwen3_vl/default/mobileworld.yaml",
    "scripts/configs/qwen3_vl/recipes/dagger/mobilegym.yaml",
    "scripts/configs/step_gui/default/androidlab.yaml",
    "scripts/configs/step_gui/default/androidworld.yaml",
    "scripts/configs/step_gui/default/mobilegym.yaml",
    "scripts/configs/step_gui/default/mobileworld.yaml",
    "scripts/configs/ui_venus_2/default/androidlab.yaml",
    "scripts/configs/ui_venus_2/default/androidworld.yaml",
    "scripts/configs/ui_venus_2/default/mobilegym.yaml",
    "scripts/configs/ui_venus_2/default/mobileworld.yaml",
} | _MOBILE_USE_RECIPE_CONFIGS

_MOBILE_ANSWER_FINISH_CONFIGS = {
    "scripts/configs/claude/default/androidlab.yaml",
    "scripts/configs/claude/default/androidworld.yaml",
    "scripts/configs/claude/default/mobilegym.yaml",
    "scripts/configs/claude/default/mobileworld.yaml",
    "scripts/configs/gpt/default/androidlab.yaml",
    "scripts/configs/gpt/default/androidworld.yaml",
    "scripts/configs/gpt/default/mobilegym.yaml",
    "scripts/configs/gpt/default/mobileworld.yaml",
    "scripts/configs/mai_ui/compact/androidworld.yaml",
    "scripts/configs/mai_ui/compact/mobilegym.yaml",
    "scripts/configs/mai_ui/default/androidlab.yaml",
    "scripts/configs/mai_ui/default/androidworld.yaml",
    "scripts/configs/mai_ui/default/mobilegym.yaml",
    "scripts/configs/mai_ui/default/mobileworld.yaml",
    "scripts/configs/qwen3_5/compact/androidworld.yaml",
    "scripts/configs/qwen3_5/compact/mobilegym.yaml",
    "scripts/configs/qwen3_5/default/androidlab.yaml",
    "scripts/configs/qwen3_5/default/androidworld.yaml",
    "scripts/configs/qwen3_5/default/mobilegym.yaml",
    "scripts/configs/qwen3_5/default/mobileworld.yaml",
    "scripts/configs/qwen3_vl/compact/androidworld.yaml",
    "scripts/configs/qwen3_vl/compact/mobilegym.yaml",
    "scripts/configs/qwen3_vl/default/androidlab.yaml",
    "scripts/configs/qwen3_vl/default/androidworld.yaml",
    "scripts/configs/qwen3_vl/default/mobilegym.yaml",
    "scripts/configs/qwen3_vl/default/mobileworld.yaml",
    "scripts/configs/qwen3_vl/recipes/dagger/mobilegym.yaml",
    "scripts/configs/step_gui/default/androidlab.yaml",
    "scripts/configs/step_gui/default/androidworld.yaml",
    "scripts/configs/step_gui/default/mobilegym.yaml",
    "scripts/configs/step_gui/default/mobileworld.yaml",
    "scripts/configs/ui_tars/default/androidlab.yaml",
    "scripts/configs/ui_tars/default/androidworld.yaml",
    "scripts/configs/ui_tars/default/mobilegym.yaml",
    "scripts/configs/ui_tars/default/mobileworld.yaml",
    "scripts/configs/ui_tars_15_v1/default/androidlab.yaml",
    "scripts/configs/ui_tars_15_v1/default/androidworld.yaml",
    "scripts/configs/ui_tars_15_v1/default/mobilegym.yaml",
    "scripts/configs/ui_tars_15_v1/default/mobileworld.yaml",
    "scripts/configs/ui_venus_2/default/androidlab.yaml",
    "scripts/configs/ui_venus_2/default/androidworld.yaml",
    "scripts/configs/ui_venus_2/default/mobilegym.yaml",
    "scripts/configs/ui_venus_2/default/mobileworld.yaml",
}

_EXPECTED_ASK_USER_CONFIGS = {
    "scripts/configs/claude/default/mobileworld.yaml",
    "scripts/configs/gpt/default/mobileworld.yaml",
    "scripts/configs/qwen3_5/default/mobileworld.yaml",
    "scripts/configs/qwen3_8/default/mobileworld.yaml",
    "scripts/configs/qwen3_vl/default/mobileworld.yaml",
}

_BROWSERGYM_RESPONSE_TERMINATE_NAV_CONFIGS = {
    "scripts/configs/claude/default/browsergym.miniwob/default.yaml",
    "scripts/configs/claude/default/browsergym.webarena/default.yaml",
    "scripts/configs/gpt/default/browsergym.miniwob/default.yaml",
    "scripts/configs/gpt/default/browsergym.webarena/default.yaml",
    "scripts/configs/qwen3_5/default/browsergym.miniwob/default.yaml",
    "scripts/configs/qwen3_5/default/browsergym.visualwebarena/goal_image.yaml",
    "scripts/configs/qwen3_5/default/browsergym.visualwebarena/mixed.yaml",
    "scripts/configs/qwen3_5/default/browsergym.visualwebarena/som.yaml",
    "scripts/configs/qwen3_5/default/browsergym.webarena/default.yaml",
    "scripts/configs/qwen3_5/default/browsergym.webarena/som.yaml",
    "scripts/configs/qwen3_5/default/browsergym.webarena/text_only.yaml",
    "scripts/configs/qwen3_vl/default/browsergym.miniwob/default.yaml",
    "scripts/configs/qwen3_vl/default/browsergym.visualwebarena/goal_image.yaml",
    "scripts/configs/qwen3_vl/default/browsergym.visualwebarena/mixed.yaml",
    "scripts/configs/qwen3_vl/default/browsergym.visualwebarena/som.yaml",
    "scripts/configs/qwen3_vl/default/browsergym.webarena/default.yaml",
    "scripts/configs/qwen3_vl/default/browsergym.webarena/som.yaml",
    "scripts/configs/qwen3_vl/default/browsergym.webarena/text_only.yaml",
}

# MiniWoB text+bid rows: `action_subsets: ["bid", "chat", "infeas"]` surfaces
# `response`/`terminate` but no nav (MiniWoB is single-page, no goto/tabs), so
# these are finish-only — kept out of _EXPECTED_NAV_CONFIGS.
_BROWSERGYM_BID_RESPONSE_TERMINATE_CONFIGS = {
    "scripts/configs/qwen3_5/default/browsergym.miniwob/text_only.yaml",
    "scripts/configs/qwen3_vl/default/browsergym.miniwob/text_only.yaml",
}

_EXPECTED_NAV_CONFIGS = {
    "scripts/configs/gpt/default/webgym.yaml",
    "scripts/configs/fara/default/browsergym.miniwob/default.yaml",
    "scripts/configs/fara/default/browsergym.visualwebarena/goal_image.yaml",
    "scripts/configs/fara/default/browsergym.webarena/default.yaml",
    "scripts/configs/fara/default/online_mind2web.yaml",
    "scripts/configs/fara/default/webgym.yaml",
    "scripts/configs/fara/default/webharbor.webvoyager/default.yaml",
    "scripts/configs/fara/default/webharbor.webvoyager/som.yaml",
    "scripts/configs/gpt/default/online_mind2web.yaml",
    "scripts/configs/gpt/default/webharbor.webvoyager/default.yaml",
    "scripts/configs/gpt/recipes/collect/webgym.yaml",
    "scripts/configs/qwen3_5/compact/webgym.yaml",
    "scripts/configs/qwen3_8/compact/webgym.yaml",
    "scripts/configs/qwen3_5/default/online_mind2web.yaml",
    "scripts/configs/qwen3_8/default/online_mind2web.yaml",
    "scripts/configs/qwen3_5/default/webgym.yaml",
    "scripts/configs/qwen3_8/default/webgym.yaml",
    "scripts/configs/qwen3_5/default/webharbor.webvoyager/default.yaml",
    "scripts/configs/qwen3_8/default/webharbor.webvoyager/default.yaml",
    "scripts/configs/qwen3_5/default/webharbor.webvoyager/som.yaml",
    "scripts/configs/qwen3_8/default/webharbor.webvoyager/som.yaml",
    "scripts/configs/qwen3_vl/compact/webgym.yaml",
    "scripts/configs/qwen3_vl/default/online_mind2web.yaml",
    "scripts/configs/qwen3_vl/default/webgym.yaml",
    "scripts/configs/qwen3_vl/default/webharbor.webvoyager/default.yaml",
    "scripts/configs/qwen3_vl/default/webharbor.webvoyager/som.yaml",
    "scripts/configs/ui_venus_2/default/webharbor.webvoyager/default.yaml",
    "scripts/configs/qwen3_8/default/browsergym.miniwob/default.yaml",
    "scripts/configs/qwen3_8/default/browsergym.visualwebarena/goal_image.yaml",
    "scripts/configs/qwen3_8/default/browsergym.visualwebarena/mixed.yaml",
    "scripts/configs/qwen3_8/default/browsergym.visualwebarena/som.yaml",
    "scripts/configs/qwen3_8/default/browsergym.webarena/default.yaml",
    "scripts/configs/qwen3_8/default/browsergym.webarena/som.yaml",
    "scripts/configs/qwen3_8/default/browsergym.webarena/text_only.yaml",
} | _BROWSERGYM_RESPONSE_TERMINATE_NAV_CONFIGS

#: The ``desktop.use`` teacher-comparison recipe, kept in two places: the public
#: walkthrough under ``examples/`` and the campaign copy under ``devs/exps/train/``
#: that must stand on its own. Every one declares ``extra_tools: [response,
#: terminate]`` -- required, not cosmetic: an undeclared finish tool is classified
#: "inactive" and NOOPed (``standalone_tool_call_feedback_with_reason``,
#: ``lite/gym/utils/feedback/ingress.py``, applied in ``lite/gym/sandbox/base.py``),
#: and the rendered action enum omits it -- so the model both cannot end an episode
#: and is evaluated on a surface its SFT data did not have.
_DESKTOP_USE_RECIPE_CONFIGS = {
    f"examples/lite/v1/configs/qwen3_5/desktop.use.{_variant}.yaml"
    for _variant in ("compact", "compact.reasoning", "default", "default.reasoning")
} | {
    # The campaign copy names its screenshot profiles by what they are (`lowr`/`highr`,
    # `i4`/`i1`/`h1`) and carries the one-image variants the public walkthrough does not.
    # Every profile has a `.reasoning` twin (`enable_thinking` alone), and the twin must
    # declare the same extra_tools: it is trained and scored on the same surface.
    f"devs/exps/train/desktop/configs/qwen3_5/desktop.use.{_variant}.yaml"
    for _variant in (
        "default", "default.reasoning",
        "lowr.i4", "lowr.i4.reasoning",
        "lowr.h1", "lowr.h1.reasoning",
        "lowr.i1", "lowr.i1.reasoning",
        "highr.h1", "highr.h1.reasoning",
        "highr.i1", "highr.i1.reasoning",
    )
}

_EXPECTED_RESPONSE_CONFIGS = {
    "scripts/configs/gpt/default/webgym.yaml",

    # The fara browsergym rows all select ``response`` so a tool-call-free
    # final turn is scored through ``make_no_tool_call_final_actions`` instead
    # of degrading to a bare ``terminate``; miniwob was the one row that had
    # drifted out of the set.
    "scripts/configs/fara/default/browsergym.miniwob/default.yaml",
    "scripts/configs/fara/default/browsergym.visualwebarena/goal_image.yaml",
    "scripts/configs/fara/default/browsergym.webarena/default.yaml",
    "scripts/configs/fara/default/webgym.yaml",
    "scripts/configs/fara/default/online_mind2web.yaml",
    "scripts/configs/fara/default/webharbor.webvoyager/default.yaml",
    "scripts/configs/fara/default/webharbor.webvoyager/som.yaml",
    "scripts/configs/gpt/default/online_mind2web.yaml",
    "scripts/configs/gpt/default/webharbor.webvoyager/default.yaml",
    "scripts/configs/gpt/recipes/collect/mobilegym.yaml",
    "scripts/configs/gpt/recipes/collect/webgym.yaml",
    "scripts/configs/qwen3_5/compact/webgym.yaml",
    "scripts/configs/qwen3_8/compact/webgym.yaml",
    # Qwen's native `answer` action lowers to canonical `response`; every
    # qwen lite.osworld row that advertises standalone finish tools must carry
    # both response and terminate.
    "scripts/configs/qwen3_5/compact/lite.osworld.yaml",
    "scripts/configs/qwen3_8/compact/lite.osworld.yaml",
    "scripts/configs/qwen3_5/default/lite.osworld.bash.yaml",
    "scripts/configs/qwen3_8/default/lite.osworld.bash.yaml",
    "scripts/configs/qwen3_5/default/lite.osworld.yaml",
    "scripts/configs/qwen3_8/default/lite.osworld.yaml",
    "scripts/configs/qwen3_5/default/online_mind2web.yaml",
    "scripts/configs/qwen3_8/default/online_mind2web.yaml",
    "scripts/configs/qwen3_5/default/webgym.yaml",
    "scripts/configs/qwen3_8/default/webgym.yaml",
    "scripts/configs/qwen3_5/default/webharbor.webvoyager/default.yaml",
    "scripts/configs/qwen3_8/default/webharbor.webvoyager/default.yaml",
    "scripts/configs/qwen3_5/default/webharbor.webvoyager/som.yaml",
    "scripts/configs/qwen3_8/default/webharbor.webvoyager/som.yaml",
    "scripts/configs/qwen3_vl/compact/webgym.yaml",
    "scripts/configs/ui_venus_2/default/webharbor.webvoyager/default.yaml",
    "scripts/configs/ui_venus_2/default/lite.osworld.yaml",
    "scripts/configs/ui_venus_2/default/osworld.yaml",
    "scripts/configs/qwen3_vl/compact/lite.osworld.yaml",
    "scripts/configs/qwen3_vl/default/lite.osworld.bash.yaml",
    "scripts/configs/qwen3_vl/default/lite.osworld.yaml",
    "scripts/configs/qwen3_vl/default/online_mind2web.yaml",
    "scripts/configs/qwen3_vl/default/webgym.yaml",
    "scripts/configs/qwen3_vl/default/webharbor.webvoyager/default.yaml",
    "scripts/configs/qwen3_vl/default/webharbor.webvoyager/som.yaml",
    "scripts/configs/qwen3_8/compact/androidworld.yaml",
    "scripts/configs/qwen3_8/compact/mobilegym.yaml",
    "scripts/configs/qwen3_8/default/androidlab.yaml",
    "scripts/configs/qwen3_8/default/androidworld.yaml",
    "scripts/configs/qwen3_8/default/browsergym.miniwob/default.yaml",
    "scripts/configs/qwen3_8/default/browsergym.miniwob/text_only.yaml",
    "scripts/configs/qwen3_8/default/browsergym.visualwebarena/goal_image.yaml",
    "scripts/configs/qwen3_8/default/browsergym.visualwebarena/mixed.yaml",
    "scripts/configs/qwen3_8/default/browsergym.visualwebarena/som.yaml",
    "scripts/configs/qwen3_8/default/browsergym.webarena/default.yaml",
    "scripts/configs/qwen3_8/default/browsergym.webarena/som.yaml",
    "scripts/configs/qwen3_8/default/browsergym.webarena/text_only.yaml",
    "scripts/configs/qwen3_8/default/mobilegym.yaml",
    "scripts/configs/qwen3_8/default/mobileworld.yaml",
} | (_BROWSERGYM_RESPONSE_TERMINATE_NAV_CONFIGS | _MOBILE_ANSWER_FINISH_CONFIGS
     | _BROWSERGYM_BID_RESPONSE_TERMINATE_CONFIGS | _DESKTOP_USE_RECIPE_CONFIGS
     | _MOBILE_USE_RECIPE_CONFIGS)

_EXPECTED_TERMINATE_CONFIGS = {
    "scripts/configs/evocua/default/waa.yaml",
    "scripts/configs/ui_tars_15_v1/default/waa.yaml",
    "scripts/configs/ui_venus_2/default/waa.yaml",
    "scripts/configs/gpt/recipes/collect/mobilegym.yaml",
    "scripts/configs/gpt/default/online_mind2web.yaml",
    "scripts/configs/qwen3_5/default/online_mind2web.yaml",
    "scripts/configs/qwen3_8/default/online_mind2web.yaml",
    "scripts/configs/qwen3_vl/default/online_mind2web.yaml",
    "scripts/configs/fara/default/browsergym.miniwob/default.yaml",
    "scripts/configs/fara/default/browsergym.visualwebarena/goal_image.yaml",
    "scripts/configs/fara/default/browsergym.webarena/default.yaml",
    "scripts/configs/fara/default/online_mind2web.yaml",
    "scripts/configs/fara/default/webgym.yaml",
    "scripts/configs/fara/default/webharbor.webvoyager/default.yaml",
    "scripts/configs/fara/default/webharbor.webvoyager/som.yaml",
    # The bash rows: advertising `bash` costs the pure-text-final channel, so
    # each must carry an explicit terminal tool (see their header notes). The two
    # qwen rows also exercise the image-budget history change — a bash turn
    # returns text and no screenshot, so it spends none of `image_max` /
    # `full_history_size` and its output rides into the summary instead of being
    # dropped with the window.
    "scripts/configs/gpt/default/lite.osworld.bash.yaml",
    "scripts/configs/qwen3_5/default/lite.osworld.bash.yaml",
    "scripts/configs/qwen3_8/default/lite.osworld.bash.yaml",
    "scripts/configs/qwen3_vl/default/lite.osworld.bash.yaml",
    "scripts/configs/evocua/default/lite.osworld.yaml",
    "scripts/configs/evocua/default/osworld.yaml",
    "scripts/configs/ui_venus_2/default/lite.osworld.yaml",
    "scripts/configs/ui_venus_2/default/osworld.yaml",
    "scripts/configs/qwen3_5/compact/lite.osworld.yaml",
    "scripts/configs/qwen3_8/compact/lite.osworld.yaml",
    "scripts/configs/qwen3_5/default/cua.bench/basic.yaml",
    "scripts/configs/qwen3_8/default/cua.bench/basic.yaml",
    "scripts/configs/qwen3_5/default/cua.bench/kicad.yaml",
    "scripts/configs/qwen3_8/default/cua.bench/kicad.yaml",
    "scripts/configs/qwen3_5/default/cua.bench/workflows.yaml",
    "scripts/configs/qwen3_8/default/cua.bench/workflows.yaml",
    "scripts/configs/qwen3_5/default/lite.cuagym.yaml",
    "scripts/configs/qwen3_8/default/lite.cuagym.yaml",
    "scripts/configs/qwen3_5/default/lite.cuaworld.yaml",
    "scripts/configs/qwen3_8/default/lite.cuaworld.yaml",
    "scripts/configs/qwen3_5/default/lite.osworld.yaml",
    "scripts/configs/qwen3_8/default/lite.osworld.yaml",
    "scripts/configs/qwen3_5/default/osworld.yaml",
    "scripts/configs/qwen3_8/default/osworld.yaml",
    "scripts/configs/qwen3_5/default/waa.yaml",
    "scripts/configs/qwen3_8/default/waa.yaml",
    "scripts/configs/qwen3_vl/compact/lite.osworld.yaml",
    "scripts/configs/qwen3_vl/default/cua.bench/basic.yaml",
    "scripts/configs/qwen3_vl/default/cua.bench/kicad.yaml",
    "scripts/configs/qwen3_vl/default/cua.bench/workflows.yaml",
    "scripts/configs/qwen3_vl/default/lite.osworld.yaml",
    "scripts/configs/qwen3_vl/default/osworld.yaml",
    "scripts/configs/qwen3_vl/default/osworld_2.yaml",
    "scripts/configs/qwen3_vl/default/waa.yaml",
    "scripts/configs/ui_tars/default/lite.osworld.yaml",
    "scripts/configs/ui_tars/default/osworld.yaml",
    "scripts/configs/ui_tars_15_v1/default/lite.osworld.yaml",
    "scripts/configs/ui_tars_15_v1/default/osworld.yaml",
    "scripts/configs/qwen3_8/compact/androidworld.yaml",
    "scripts/configs/qwen3_8/compact/mobilegym.yaml",
    "scripts/configs/qwen3_8/default/androidlab.yaml",
    "scripts/configs/qwen3_8/default/androidworld.yaml",
    "scripts/configs/qwen3_8/default/browsergym.miniwob/default.yaml",
    "scripts/configs/qwen3_8/default/browsergym.miniwob/text_only.yaml",
    "scripts/configs/qwen3_8/default/browsergym.visualwebarena/goal_image.yaml",
    "scripts/configs/qwen3_8/default/browsergym.visualwebarena/mixed.yaml",
    "scripts/configs/qwen3_8/default/browsergym.visualwebarena/som.yaml",
    "scripts/configs/qwen3_8/default/browsergym.webarena/default.yaml",
    "scripts/configs/qwen3_8/default/browsergym.webarena/som.yaml",
    "scripts/configs/qwen3_8/default/browsergym.webarena/text_only.yaml",
    "scripts/configs/qwen3_8/default/mobilegym.yaml",
    "scripts/configs/qwen3_8/default/mobileworld.yaml",
} | (_BROWSERGYM_RESPONSE_TERMINATE_NAV_CONFIGS | _MOBILE_ANSWER_FINISH_CONFIGS
     | _BROWSERGYM_BID_RESPONSE_TERMINATE_CONFIGS | _DESKTOP_USE_RECIPE_CONFIGS
     | _MOBILE_USE_RECIPE_CONFIGS)

_REQUIRED_RESPONSE_EXTRA_TOOL_CONFIGS = {
    "scripts/configs/fara/default/webgym.yaml",
    "scripts/configs/fara/default/online_mind2web.yaml",
    "scripts/configs/fara/default/webharbor.webvoyager/default.yaml",
    "scripts/configs/fara/default/webharbor.webvoyager/som.yaml",
    "scripts/configs/gpt/default/online_mind2web.yaml",
    "scripts/configs/gpt/default/webharbor.webvoyager/default.yaml",
    "scripts/configs/gpt/recipes/collect/webgym.yaml",
    "scripts/configs/qwen3_5/compact/webgym.yaml",
    "scripts/configs/qwen3_5/default/online_mind2web.yaml",
    "scripts/configs/qwen3_5/default/webgym.yaml",
    "scripts/configs/qwen3_5/default/webharbor.webvoyager/default.yaml",
    "scripts/configs/qwen3_5/default/webharbor.webvoyager/som.yaml",
    "scripts/configs/qwen3_vl/compact/webgym.yaml",
    "scripts/configs/qwen3_vl/default/online_mind2web.yaml",
    "scripts/configs/qwen3_vl/default/webgym.yaml",
    "scripts/configs/qwen3_vl/default/webharbor.webvoyager/default.yaml",
    "scripts/configs/qwen3_vl/default/webharbor.webvoyager/som.yaml",
}

_TARGET_WEBHARBOR_ONLINE_CONFIGS = {
    "scripts/configs/fara/default/online_mind2web.yaml",
    "scripts/configs/fara/default/webharbor.webvoyager/default.yaml",
    "scripts/configs/fara/default/webharbor.webvoyager/som.yaml",
    "scripts/configs/gpt/default/online_mind2web.yaml",
    "scripts/configs/gpt/default/webharbor.webvoyager/default.yaml",
    "scripts/configs/qwen3_5/default/online_mind2web.yaml",
    "scripts/configs/qwen3_5/default/webharbor.webvoyager/default.yaml",
    "scripts/configs/qwen3_5/default/webharbor.webvoyager/som.yaml",
    "scripts/configs/qwen3_vl/default/online_mind2web.yaml",
    "scripts/configs/qwen3_vl/default/webharbor.webvoyager/default.yaml",
    "scripts/configs/qwen3_vl/default/webharbor.webvoyager/som.yaml",
}


def _tool_io_source_yamls() -> list[Path]:
    root = project_root()
    paths: list[Path] = []
    for rel_root in _AGENT_CONFIG_ROOTS:
        base = root / rel_root
        if not base.exists():
            continue
        paths.extend(
            p for p in base.rglob("*.yaml")
            if "build/lib" not in p.as_posix() and "/.tmp/" not in p.as_posix()
        )
    env_root = root / "lite" / "gym" / "envs"
    if env_root.exists():
        paths.extend(env_root.glob("**/configs/default.yaml"))
    return sorted(set(paths))


def _agent_config_yamls() -> list[Path]:
    root = project_root()
    paths: list[Path] = []
    missing: list[Path] = []
    for rel_root in _AGENT_CONFIG_ROOTS:
        base = root / rel_root
        if not base.exists():
            missing.append(rel_root)
            continue
        paths.extend(sorted(base.rglob("*.yaml")))
    assert not missing, f"missing agent config roots: {missing}"
    assert paths, "no agent config yamls found"
    return paths


def _walk_tool_surface_keys(node, path: str = ""):
    if isinstance(node, dict):
        for key, value in node.items():
            next_path = f"{path}.{key}" if path else str(key)
            if key in _TOOL_SURFACE_SOURCE_KEYS:
                yield next_path, key, value
            yield from _walk_tool_surface_keys(value, next_path)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _walk_tool_surface_keys(value, f"{path}[{i}]")


def _source_yaml_rel(path: Path) -> str:
    return path.relative_to(project_root()).as_posix()


def _load_yaml_mapping(path: Path) -> dict:
    data = yaml.safe_load(path.read_text()) or {}
    assert isinstance(data, dict), f"{_source_yaml_rel(path)} must be a mapping"
    return data


def _env_extra_tools(rel: str) -> list:
    path = project_root() / rel
    data = _load_yaml_mapping(path)
    env_kwargs = data.get("env_kwargs") or {}
    assert isinstance(env_kwargs, dict), f"{rel}:env_kwargs must be a mapping"
    extra_tools = env_kwargs.get("extra_tools") or []
    assert isinstance(extra_tools, list), f"{rel}:env_kwargs.extra_tools must be a list"
    return extra_tools


def _set_drift(actual: set[str], expected: set[str]) -> str:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    return f"missing={missing}\nextra={extra}\nactual={sorted(actual)}"


def _walk_valid_actions(node, path: str = ""):
    if isinstance(node, dict):
        for key, value in node.items():
            next_path = f"{path}.{key}" if path else str(key)
            if key == "valid_actions":
                yield next_path, value
            yield from _walk_valid_actions(value, next_path)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _walk_valid_actions(value, f"{path}[{i}]")


def test_tool_io_yaml_valid_actions_are_gui_inner_only():
    """tool I/O narrows ``valid_actions`` to GUI inner actions only.

    Finish actions are intrinsic and nav/mobile/text tools belong in
    ``extra_tools`` or family-native wire, never in source YAML valid_actions.
    ``valid_actions: []`` remains valid and means "drop native GUI".
    """
    offenders: list[str] = []
    for path in _tool_io_source_yamls():
        data = yaml.safe_load(path.read_text()) or {}
        rel = path.relative_to(project_root())
        for key_path, value in _walk_valid_actions(data):
            if value in (None, []):
                continue
            if not isinstance(value, list):
                offenders.append(f"{rel}:{key_path} is {type(value).__name__}, expected list|null")
                continue
            bad = sorted(set(value) & _NON_GUI_VALID_ACTIONS)
            if bad:
                offenders.append(f"{rel}:{key_path} contains non-GUI {bad}")

    assert not offenders, "valid_actions must be GUI-inner only:\n" + "\n".join(offenders)


def test_tool_io_source_yaml_tool_surface_keys_are_env_kwargs_only():
    """Source YAML may select tool surface only through env_kwargs.

    ``agent_kwargs`` is model/runtime plumbing. It must never inject
    ``extra_tools``, ``valid_actions``, ``others``, or persisted
    ``extra_tool_schemas``.
    """
    offenders: list[str] = []
    allowed_source_paths = {"env_kwargs.extra_tools", "env_kwargs.valid_actions"}

    for path in _tool_io_source_yamls():
        rel = _source_yaml_rel(path)
        data = _load_yaml_mapping(path)
        agent_kwargs = data.get("agent_kwargs") or {}
        env_kwargs = data.get("env_kwargs") or {}

        if not isinstance(agent_kwargs, dict):
            offenders.append(
                f"{rel}:agent_kwargs is {type(agent_kwargs).__name__}, expected mapping"
            )
            continue
        if not isinstance(env_kwargs, dict):
            offenders.append(f"{rel}:env_kwargs is {type(env_kwargs).__name__}, expected mapping")
            continue

        for key_path, key, value in _walk_tool_surface_keys(data):
            if key_path in allowed_source_paths:
                if value is not None and not isinstance(value, list):
                    offenders.append(
                        f"{rel}:{key_path} is {type(value).__name__}, expected list|null"
                    )
                continue
            offenders.append(
                f"{rel}:{key_path} uses tool-surface key {key!r} outside "
                "top-level env_kwargs selector"
            )

    assert not offenders, "tool-surface YAML keys must be env_kwargs-only:\n" + "\n".join(offenders)


def test_tool_io_default_agent_yaml_extra_tool_matrix_is_explicit_and_classified():
    """Classify the current default agent x env tool surface.

    This is intentionally a source-YAML gate, not a resolved-metadata test:
    source configs decide ``extra_tools`` names; env resolution later turns
    those names into persisted ``metadata.extra_tool_schemas``.
    """
    response_configs: set[str] = set()
    terminate_configs: set[str] = set()
    open_app_configs: set[str] = set()
    ask_user_configs: set[str] = set()
    nav_configs: set[str] = set()
    valid_action_offenders: list[str] = []

    for path in _agent_config_yamls():
        rel = _source_yaml_rel(path)
        data = _load_yaml_mapping(path)
        env_kwargs = data.get("env_kwargs") or {}
        if not isinstance(env_kwargs, dict):
            continue

        extra_tools = env_kwargs.get("extra_tools") or []
        for tool in extra_tools:
            if tool == "response":
                response_configs.add(rel)
            if tool == "terminate":
                terminate_configs.add(rel)
            if tool == "open_app":
                open_app_configs.add(rel)
            if tool == "ask_user":
                ask_user_configs.add(rel)
            if tool in _NAV_TOOL_NAMES:
                nav_configs.add(rel)

        valid_actions = env_kwargs.get("valid_actions")
        if valid_actions:
            bad = sorted(
                set(valid_actions)
                & (_FINISH_TOOL_NAMES | _NAV_TOOL_NAMES | {"open_app", "ask_user"})
            )
            if bad:
                valid_action_offenders.append(f"{rel}:env_kwargs.valid_actions contains {bad}")

    assert response_configs == _EXPECTED_RESPONSE_CONFIGS, _set_drift(
        response_configs, _EXPECTED_RESPONSE_CONFIGS
    )
    assert terminate_configs == _EXPECTED_TERMINATE_CONFIGS, _set_drift(
        terminate_configs, _EXPECTED_TERMINATE_CONFIGS
    )
    assert open_app_configs == _EXPECTED_OPEN_APP_CONFIGS, _set_drift(
        open_app_configs, _EXPECTED_OPEN_APP_CONFIGS
    )
    assert ask_user_configs == _EXPECTED_ASK_USER_CONFIGS, _set_drift(
        ask_user_configs, _EXPECTED_ASK_USER_CONFIGS
    )
    assert nav_configs == _EXPECTED_NAV_CONFIGS, _set_drift(nav_configs, _EXPECTED_NAV_CONFIGS)
    assert not valid_action_offenders, (
        "finish/nav/open_app/ask_user must not be hidden in valid_actions:\n"
        + "\n".join(valid_action_offenders)
    )


def test_tool_io_browsergym_yaml_does_not_rely_on_implicit_extra_tools():
    offenders: list[str] = []
    for path in _agent_config_yamls():
        rel = _source_yaml_rel(path)
        data = _load_yaml_mapping(path)
        if not str(data.get("env_id") or "").startswith("browsergym."):
            continue
        env_kwargs = data.get("env_kwargs") or {}
        if not isinstance(env_kwargs, dict):
            continue
        if "extra_tools" not in env_kwargs:
            offenders.append(f"{rel}:env_kwargs.extra_tools omitted")
        elif env_kwargs["extra_tools"] is None:
            offenders.append(f"{rel}:env_kwargs.extra_tools is null")

    assert not offenders, (
        "browsergym source YAML must spell extra_tools explicitly:\n" + "\n".join(offenders)
    )


def test_tool_io_text_answer_rows_select_response_extra_tool():
    for rel in sorted(_REQUIRED_RESPONSE_EXTRA_TOOL_CONFIGS):
        assert "response" in _env_extra_tools(rel), rel


def test_tool_io_target_webharbor_online_rows_do_not_select_canonical_done():
    for rel in sorted(_TARGET_WEBHARBOR_ONLINE_CONFIGS):
        assert "done" not in _env_extra_tools(rel), rel


def test_tool_io_target_webharbor_online_rows_keep_finish_nav_out_of_valid_actions():
    forbidden = _FINISH_TOOL_NAMES | _NAV_TOOL_NAMES | {"done", "finish", "submit"}
    for rel in sorted(_TARGET_WEBHARBOR_ONLINE_CONFIGS):
        data = _load_yaml_mapping(project_root() / rel)
        valid_actions = (data.get("env_kwargs") or {}).get("valid_actions") or []
        assert not (set(valid_actions) & forbidden), rel


def test_tool_io_mobile_answer_rows_select_finish_extra_tools():
    for rel in sorted(_MOBILE_ANSWER_FINISH_CONFIGS):
        extras = _env_extra_tools(rel)
        assert "response" in extras, rel
        assert "terminate" in extras, rel
