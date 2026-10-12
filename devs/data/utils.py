"""Row helpers used only by the ``devs/`` data tooling, not by ``lite/``.

Split out of ``lite/data/staging.py``: both functions here have ZERO consumers
under ``lite/`` and are called only by the dev-side annotation/migration
scripts, so they live on the dev side of the boundary. Everything they build on
— ``coerce_image_paths``, ``to_plain``, ``resolve_artifact_path`` — stays in
``lite.data.staging`` and is imported back from there; ``devs`` → ``lite`` is
the normal direction.

Why ``devs/data/`` and not a neutral ``devs/utils.py``: the two PERMANENT
consumers are the annotation filters, ``devs/data/lite.osworld/filter.py`` and
``devs/data/webgym/filter.py``, both under this directory. The third consumer,
``devs/migration/upgrade.py``, reaches across subtrees — that is expected, not a
smell, because migration is temporary: ``devs/migration/AGENTS.md`` calls itself
"a **one-time repair of old ``Lite.*`` rollout inputs**, not a general data
pipeline". Shaping the layout around a subtree scheduled for deletion would be
backwards; when migration goes, this module is already where its real owners
live and nothing has to move a second time.

``devs`` is NOT an installed package (``pyproject.toml`` ships ``lite*`` only),
so an importer has to put the repo root on ``sys.path`` first — see the
``_REPO_ROOT`` bootstrap in ``devs/data/lite.osworld/filter.py``,
``devs/data/webgym/filter.py`` and ``devs/migration/upgrade.py``.

Use:
    _REPO_ROOT = Path(__file__).resolve().parents[<depth to repo root>]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from devs.data.utils import (
        _args_of,
        _action_name_args,
        _iter_action_items,
        _with_args,
        carry_content_without_observation_images,
        compact_row_images,
        rebase_images_for_output,
    )

Tests: uv run pytest tests/data/staging/test_staging_contract.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from lite.core.tools.action_space import (
    LITE_ACTION_BATCH_TOOL_NAMES,
    LITE_VALID_ACTION_NAMES,
    lite_action_names_for_action_batch_tool,
)
from lite.core.tools.calls import (
    make_tool_call,
    tool_call_arguments,
    tool_call_id,
    tool_call_name,
)

# ``_coerce_storage_image_index`` is PRIVATE and imported across the devs → lite
# boundary DELIBERATELY (owner-approved). ``devs/`` is developer tooling, not
# shipped code — ``pyproject.toml`` ships ``lite*`` only — so the "no private
# cross-boundary import" rule, which is about coupling INSIDE the shipped
# library, does not bind it. Both alternatives are worse: a local copy would be
# a SECOND spelling of the storage index-coercion rule (exactly the defect class
# this split exists to remove), and promoting it public would widen ``lite``'s
# API surface for a dev-only caller. If ``lite`` ever renames or drops it, this
# module fails loudly at import with an ``ImportError`` — that tripwire is the
# point, not an accident.
from lite.data.staging import (
    _coerce_storage_image_index,
    coerce_image_paths,
    resolve_artifact_path,
    to_plain,
)


def _args_of(tc: dict) -> dict[str, Any]:
    args = tool_call_arguments(tc)
    if not isinstance(args, dict):
        raise ValueError("canonical tool_call.function.arguments must be a dict")
    return args


def _with_args(tc: dict, args: dict[str, Any]) -> dict:
    """Return canonical ``tc`` with replacement function arguments."""
    return make_tool_call(tool_call_name(tc), args, call_id=tool_call_id(tc))


def _action_name_args(
    action: Any,
    action_batch_tool_name: str,
) -> tuple[str, dict[str, Any]]:
    """Split an action-batch ``actions[]`` entry and reject non-GUI children."""
    name = action["action"]
    if name not in (
        lite_action_names_for_action_batch_tool(action_batch_tool_name) or frozenset()
    ):
        raise ValueError(
            f"{name!r} is not valid inside {action_batch_tool_name}; standalone tools "
            f"must not be nested inside action-batch actions"
        )
    # Parquet list-of-struct rows can carry sibling action fields as nulls.
    # Filters must not republish those union placeholders as child arguments.
    args = {k: v for k, v in action.items() if k != "action" and v is not None}
    return name, args


def has_invalid_action_batch(messages: list[dict]) -> bool:
    """True if an action-batch child names an action that does not exist.

    The model sometimes invents an action name (``terminal``, ``select_all``,
    ``mouse_wheel``) or lets raw wire text land in it (``<parameter=action>\nkey``).
    The batch tool itself is schema-free, so ``has_undeclared_tool_call`` cannot see
    inside it, and staging rejects the row -- ``computer.actions cannot contain
    terminal``.

    Every filter that walks a batch through :func:`_iter_action_items` must call this
    FIRST and drop the row: ``_action_name_args`` RAISES on an unknown child name, so
    one such row aborts the whole log-root before any later pass can remove it, taking
    every clean sibling with it.

    Only the NAME is checked. Validating the child ARGUMENTS here would make this a
    second row validator against a contract the filter has not applied yet: it runs
    before normalisation, so an argument shape this pass is about to fix would be read
    as publish-invalid and the row deleted instead of repaired.
    """
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if fn.get("name") not in LITE_ACTION_BATCH_TOOL_NAMES:
                continue
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (TypeError, ValueError):
                    return True
            for act in (args.get("actions") or []) if isinstance(args, dict) else []:
                if not isinstance(act, dict):
                    return True
                name = act.get("action")
                if not isinstance(name, str) or name not in LITE_VALID_ACTION_NAMES:
                    return True
    return False


def _iter_action_items(message: dict) -> list[tuple[str, dict[str, Any]]]:
    """Action stream from standalone calls or action-batch calls."""
    items: list[tuple[str, dict[str, Any]]] = []
    for tc in message.get("tool_calls") or []:
        name = tool_call_name(tc)
        args = _args_of(tc)
        actions = args.get("actions")
        if name in LITE_ACTION_BATCH_TOOL_NAMES and isinstance(actions, list):
            items.extend(_action_name_args(action, name) for action in actions)
        else:
            items.append((name, args))
    return items


def _indexed_image_part(part: Any) -> bool:
    return isinstance(part, dict) and part.get("type") == "image" and "index" in part


def carry_content_without_observation_images(content: Any) -> list[dict]:
    """Carry authored content from a popped user turn, dropping its observation.

    Dev-side filters use this only when converting a preceding-observation
    layout: a no-op-only assistant turn is removed, so the user observation it
    answered is superseded by the next user observation. Reference/task images
    must already be represented as ordinary indexed image parts in canonical
    message content. Legacy image-selection metadata is intentionally ignored here;
    content image parts are the only image-reference source.
    """
    if not isinstance(content, list):
        return []

    image_positions = [
        pos for pos, part in enumerate(content)
        if _indexed_image_part(part)
    ]
    observation_pos = image_positions[-1] if image_positions else None

    carried: list[dict] = []
    for pos, part in enumerate(content):
        if isinstance(part, dict) and part.get("type") == "image":
            if pos == observation_pos:
                continue
            if not _indexed_image_part(part):
                continue
        carried.append(part)
    return carried


def compact_row_images(images, messages) -> tuple[list[str], list]:
    """Drop images no message references, and renumber the survivors 0..N-1.

    THE ONE PLACE THAT MAY RENUMBER AN IMAGE INDEX. Call it where a whole row is
    in hand -- a producer that has just decided which messages survive -- and
    never incrementally: the image list and every ``{"type":"image","index": N}``
    part must be rewritten in the SAME step, or a surviving message keeps a
    number that now addresses a different picture. That failure is silent (the
    model sees a screenshot that does not match its label, nothing raises), which
    is why this function asserts its own postcondition instead of trusting it.

    Why it is needed at all: a filter drops a STEP, not a picture, so without
    this the row keeps every image while the surviving messages reference a
    non-contiguous subset. That is internally consistent -- ``export_sft``
    resolves it correctly -- but it publishes orphan images that no message
    shows, which is confusing in a dataset viewer and pure waste on disk
    `[measured across published cua-lite datasets: 2,736 of 74,540 images
    (3.7%) were orphans, in 1,414 of 18,012 rows; every affected dataset came
    from the rollout+filter path, and every preproc dataset had zero]`.

    Returns ``(images, messages)`` with orphans removed and indices contiguous.
    Rows whose references are already ``0..N-1`` are returned unchanged, so this
    is a no-op on data that never lost a step.
    """
    paths = coerce_image_paths(images)
    plain = to_plain(messages)
    if not isinstance(plain, list) or not paths:
        return paths, plain

    referenced: list[int] = []
    seen: set[int] = set()
    for message in plain:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not _indexed_image_part(part):
                continue
            index = _coerce_storage_image_index(part["index"])
            if isinstance(index, int) and index not in seen:
                seen.add(index)
                referenced.append(index)

    if not referenced:
        # No image parts at all: this row's images are addressed some other way
        # (or it has none). Compacting to [] would destroy them; leave it alone.
        return paths, plain
    if any(i < 0 or i >= len(paths) for i in referenced):
        raise ValueError(
            f"image index out of range before compaction: referenced={sorted(referenced)} "
            f"but the row holds {len(paths)} images"
        )
    if sorted(referenced) == list(range(len(paths))):
        return paths, plain  # already dense -- nothing to do

    remap = {old: new for new, old in enumerate(sorted(referenced))}
    new_paths = [paths[old] for old in sorted(referenced)]
    new_messages = []
    for message in plain:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            new_messages.append(message)
            continue
        msg = dict(message)
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image" and "index" in part:
                index = _coerce_storage_image_index(part["index"])
                if isinstance(index, int):
                    part = {**part, "index": remap[index]}
            parts.append(part)
        msg["content"] = parts
        new_messages.append(msg)

    # Postcondition, asserted rather than assumed: the whole point of this
    # function is that no index may survive pointing at the wrong picture.
    # Covers the canonical message-content image references; metadata keys are
    # not a second source of image ownership for dev/data filtering.
    after = {
        _coerce_storage_image_index(part["index"])
        for message in new_messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict) and part.get("type") == "image" and "index" in part
    }
    if after != set(range(len(new_paths))):
        raise AssertionError(
            f"compaction left a non-dense index set: {sorted(after)} over "
            f"{len(new_paths)} images (remap={remap})"
        )
    return new_paths, new_messages


def rebase_images_for_output(
    rows,
    *,
    source_parquet: Path,
    output_parquet: Path,
    image_path_root: Path | None = None,
    dry_run: bool = False,
) -> list[list[str]]:
    """Copy row images beside an output parquet and rewrite row image paths.

    Preserves row image ORDER and therefore every persisted
    ``{"type":"image","index": N}`` binding -- it renumbers nothing. Callers that
    dropped messages must run :func:`compact_row_images` FIRST, so this receives
    an already-dense row; splitting the two keeps "copy the files" and "renumber
    the indices" from ever half-happening. It only makes a filtered log-root
    self-contained so later ``stage`` calls do not depend on the raw log-root.
    When ``image_path_root`` is provided, returned paths are relative to that
    root so direct raw export can use ``--image-root <root>``.
    """
    output_parquet = Path(output_parquet)
    path_root = Path(image_path_root).resolve() if image_path_root is not None else None
    rebased: list[list[str]] = []
    multi_row = len(rows) > 1
    for row_idx, raw_images in enumerate(rows["images"]):
        rel_dir = Path("images")
        if multi_row:
            rel_dir = rel_dir / f"row_{row_idx:04d}"
        abs_dir = output_parquet.parent / rel_dir
        row_out: list[str] = []
        for image_idx, image_path in enumerate(coerce_image_paths(raw_images)):
            source = resolve_artifact_path(image_path, anchor_path=source_parquet)
            rel_path = rel_dir / f"{image_idx:06d}{source.suffix or '.png'}"
            if not dry_run:
                abs_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, output_parquet.parent / rel_path)
            if path_root is None:
                stored_path = rel_path
            else:
                stored_path = (output_parquet.parent / rel_path).resolve().relative_to(path_root)
            row_out.append(str(stored_path))
        rebased.append(row_out)
    return rebased
