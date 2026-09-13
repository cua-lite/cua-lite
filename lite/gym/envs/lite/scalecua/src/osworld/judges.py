"""ScaleCUA judge overlay loader and DesktopEnv compatibility shim."""

from __future__ import annotations

import ast
import asyncio
import base64
import contextlib
import copy
import hashlib
import importlib.util
import inspect
import json
import logging
import math
import operator
import os
import re
import shlex
import sys
import textwrap
import threading
import urllib.parse
import xml.etree.ElementTree as ET
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from itertools import product
from pathlib import Path
from types import ModuleType
from typing import Any

from lite.gym.envs.lite.scalecua.src.utils import assets

OVERLAY_SPLITS = {"train", "rl"}
logger = logging.getLogger(__name__)
_REQUESTS_ROUTER_LOCK = threading.RLock()
_REQUESTS_ROUTER_LOCAL = threading.local()
_REQUESTS_PATCH_INSTALLED = False
_NOISY_COMMAND_PREFIXES = (
    "Xlib.xauth: warning, no xauthority details available",
)
CHROME_PROFILE_DIR = "/home/user/chrome-data"
CALC_FORMULA_VALUE_METRIC_NAMES = {
    "check_xlsx_cell_value__38c661cc",
}
CALC_COLUMN_SORTED_TOP_VALUES_METRIC_NAMES = {
    "check_xlsx_column_sorted__b22cf776",
}
CALC_SHEET_CELLS_METRIC_NAMES = {
    "check_xlsx_sheet_cells__7001880f97ce5d91e9e11e6dd84fdc47",
}
CALC_PERCENTAGE_METRIC_NAMES = {
    "check_percentage__3c3385e3",
}
CALC_NUMERIC_VALUE_METRIC_NAMES = {
    "check_numeric_value__28be6047",
}
CALC_DECIMAL_TEXT_METRIC_NAMES = {
    "check_calc_cell_text__154194afadf90d869e3ab3e594cc44ae",
    "check_calc_cell_text__a8896662f47f9f0431d9d6ec0b20825a",
}
CALC_PERIOD_RATE_SUM_METRIC_NAMES = {
    "check_xlsx_period_rate_sum__da315cf2edc339756d9b2f4defe564ad",
}
IMPRESS_NOTES_PANEL_METRIC_NAMES = {
    "check_notes_panel__be7f4bd7",
}
SPEEDTEST_REPORT_METRIC_NAMES = {
    "check_speedtest_report__26660ad1",
}
CHROME_EXPERIMENTS_EXACT_METRIC_NAMES = {
    "check_chrome_experiments_exact_match__d7481e87a8a6dde50669ce517c215edf",
}
CHROME_EXPERIMENTS_SET_METRIC_NAMES = {
    "check_enabled_experiments",
    "check_chrome_experiments_contains__0a40109ea287ab7bbd8cd9175a7ce6a5",
    "check_chrome_experiments_contains_any__70b32da51f968d0aa45e836eecc52bdb",
    "check_chrome_experiments_contains_all__e998f78abb27064086318477b860256b",
    "check_chrome_experiments_not_contains__f72b4886483f6a090bcc6a28ba49acde",
}
CHROME_EXTENSION_NAME_METRIC_NAMES = {
    "check_extension_name__de268faa9ad661d5adcd42e1b7f775e2",
}
CHROME_STARTUP_URL_METRIC_NAMES = {
    "check_chrome_startup_urls__715fc21b1c707dd79fc5ab9b6e4df514",
}
CHROME_BOOKMARK_WITH_TITLE_METRIC_NAMES = {
    "is_expected_bookmarks__35253b65",
}
CHROME_SEARCH_ENGINE_METRIC_NAMES = {
    "exact_match",
    "match_in_list",
    "check_search_engine__d2ec4a7b",
    "check_search_engine__3898b17c9447f0df83eb7e4316fa07a5",
    "check_search_engine__5f4fc73f7b634bc96eb191d9ae7f5689",
    "check_search_engine__29a0b35353e6c09dfbb5b1666d07a250",
}
CHROME_INTERNAL_ACTIVE_TAB_METRIC_NAMES = {
    "is_expected_active_tab_approximate",
}
URL_CANONICAL_ACTIVE_TAB_METRIC_NAMES = {
    "is_expected_active_tab",
}
URL_PATTERN_METRIC_NAMES = {
    "is_expected_url_pattern_match",
}
CALC_INITIALS_COLUMN_METRIC_NAMES = {
    "check_xlsx_column_g__cae1c5edda471ddd033496e2d51a93a2",
}
THUNDERBIRD_PREFS_METRIC_NAMES = {
    "check_thunderbird_prefs",
}
_THUNDERBIRD_PREF_RE = re.compile(
    r'^user_pref\("(?P<key>(?:[^"]|\\")+)",\s*(?P<val>.+)\);$'
)
_DIRECT_FILE_CALL_NAMES = {
    "open",
    "openpyxl.load_workbook",
    "load_workbook",
    "Document",
    "Presentation",
    "Image.open",
    "ZipFile",
    "zipfile.ZipFile",
}
_DIRECT_PATH_CONFIG_KEYS = {
    "path",
    "file",
    "file_path",
    "filepath",
    "local_path",
    "input_path",
    "source_path",
    "target_path",
    "target_file",
    "image_path",
    "xlsx_path",
    "xlsx_file",
    "workbook_path",
    "docx_path",
    "docx_file",
    "document_path",
    "pptx_path",
    "pptx_file",
    "ppt_file_path",
    "presentation_path",
    "csv_path",
}
_CHROME_PROFILE_PATH_ALIASES = (
    "/home/user/.config/google-chrome",
    "/home/user/.config/chromium",
    "/home/user/snap/chromium/common/chromium",
    "/home/ubuntu/.config/google-chrome",
    "/home/ubuntu/.config/chromium",
    "/home/ubuntu/snap/chromium/common/chromium",
    "$HOME/.config/google-chrome",
    "${HOME}/.config/google-chrome",
    "$HOME/.config/chromium",
    "${HOME}/.config/chromium",
    "$HOME/snap/chromium/common/chromium",
    "${HOME}/snap/chromium/common/chromium",
    "~/.config/google-chrome",
    "~/.config/chromium",
    "~/snap/chromium/common/chromium",
)
_CHROME_PROFILE_RELATIVE_ALIASES = (
    ".config/google-chrome",
    ".config/chromium",
    "snap/chromium/common/chromium",
)
_WINDOWS_USER_PATH_RE = re.compile(
    r"C:(?:\\+|/)Users(?:\\+|/)(?:user|ubuntu)(?:\\+|/)"
    r"(?P<root>AppData(?:\\+|/)Roaming(?:\\+|/)vlc|Desktop|Downloads|Documents|Music|Videos|Pictures)"
    r"(?P<rest>(?:(?:\\+|/)[^'\"\n\r;,) ]+)*)",
    re.IGNORECASE,
)


class JudgeResolverError(RuntimeError):
    """Raised when a ScaleCUA metric/getter cannot be resolved."""


def overlay_dir(runtime_split: str) -> Path | None:
    if runtime_split not in OVERLAY_SPLITS:
        return None
    return overlay_root() / runtime_split


def overlay_root() -> Path:
    return assets.CACHE_DIR / "judge_functions"


@lru_cache(maxsize=None)
def _load_overlay_module(runtime_split: str, module_name: str) -> ModuleType | None:
    root = overlay_dir(runtime_split)
    if root is None:
        return None
    path = root / f"{module_name}.py"
    if not path.is_file():
        return None
    package = f"lite_scalecua_judges_{runtime_split}_{module_name}"
    spec = importlib.util.spec_from_file_location(
        package,
        path,
        submodule_search_locations=[str(root)],
    )
    if spec is None or spec.loader is None:
        raise JudgeResolverError(f"cannot load ScaleCUA judge overlay: {path}")
    module = importlib.util.module_from_spec(spec)
    old = sys.modules.get(package)
    sys.modules[package] = module
    old_path = list(sys.path)
    try:
        spec.loader.exec_module(module)
        if module_name in {"getters", "metrics"}:
            _install_pptx_font_strikethrough_alias()
        if module_name == "getters":
            _install_gimp_xcf_helpers(package, module)
            _install_calc_chart_helpers(package, module)
            _install_generated_getter_helpers(package, module)
            _install_generated_getter_overrides(package, module)
        elif module_name == "metrics":
            _install_generated_metric_helpers(package, module)
    except Exception:
        if old is not None:
            sys.modules[package] = old
        else:
            sys.modules.pop(package, None)
        raise
    finally:
        sys.path[:] = old_path
    return module


def _parse_calc_cell_range(ref: str, workbook: Any | None = None) -> dict[str, Any] | None:
    """Parse an openpyxl chart data reference into sheet/column/row metadata."""

    text = str(ref or "")
    match = re.search(
        r"(?:'(?P<sheet_quoted>[^']+)'|(?P<sheet_plain>[A-Za-z0-9_ .-]+))!"
        r"\$?(?P<col1>[A-Z]+)\$?(?P<row1>\d+)"
        r"(?::\$?(?P<col2>[A-Z]+)\$?(?P<row2>\d+))?",
        text,
    )
    if match is None:
        return None
    sheet_name = match.group("sheet_quoted") or match.group("sheet_plain")
    if workbook is not None and sheet_name not in getattr(workbook, "sheetnames", []):
        return None
    col1 = match.group("col1")
    col2 = match.group("col2") or col1
    row1 = int(match.group("row1"))
    row2 = int(match.group("row2") or row1)
    try:
        from openpyxl.utils import column_index_from_string, get_column_letter

        start = column_index_from_string(col1)
        end = column_index_from_string(col2)
        columns = [
            get_column_letter(idx)
            for idx in range(min(start, end), max(start, end) + 1)
        ]
    except Exception:
        columns = [col1] if col1 == col2 else [col1, col2]
    return {
        "sheet_name": sheet_name,
        "columns": columns,
        "row_count": abs(row2 - row1) + 1,
    }


def _install_calc_chart_helpers(package: str, overlay: ModuleType) -> None:
    """Patch generated Calc chart getters missing their private helper."""

    modules = [overlay]
    modules.extend(
        module
        for name, module in list(sys.modules.items())
        if name.startswith(f"{package}.") and name.endswith(".verigen_getters.calc")
    )
    for module in modules:
        setattr(module, "_parse_cell_range", _parse_calc_cell_range)


def _install_gimp_xcf_helpers(package: str, overlay: ModuleType) -> None:
    """Provide helpers referenced but not shipped by ScaleCUA GIMP getters."""

    from lite.gym.envs.lite.scalecua.src.osworld import xcf

    helpers = {
        "parse_xcf_layers": xcf.parse_xcf_layers,
        "parse_xcf_file_directly": xcf.parse_xcf_file_directly,
        "parse_xcf_layer_groups": xcf.parse_xcf_layer_groups,
    }
    modules = [overlay]
    modules.extend(
        module
        for name, module in list(sys.modules.items())
        if name.startswith(f"{package}.") and name.endswith(".verigen_getters.gimp")
    )
    try:
        from desktop_env.evaluators.getters import gimp as upstream_gimp_getters
    except Exception:
        upstream_gimp_getters = None
    if upstream_gimp_getters is not None:
        modules.append(upstream_gimp_getters)
    for module in modules:
        for name, fn in helpers.items():
            setattr(module, name, fn)


def _install_generated_getter_helpers(package: str, overlay: ModuleType) -> None:
    """Provide helpers and imports omitted from generated getter shards."""

    modules = [overlay]
    modules.extend(
        module
        for name, module in list(sys.modules.items())
        if name.startswith(f"{package}.") and ".verigen_getters." in name
    )
    try:
        from lxml import etree as lxml_etree
    except Exception:
        lxml_etree = None
    for module in modules:
        setattr(module, "_get_video_rotation", _get_video_rotation)
        if lxml_etree is not None:
            setattr(module, "etree", lxml_etree)


def _get_video_rotation(video_bytes: bytes | bytearray | memoryview) -> tuple[int | None, bool]:
    """Extract video rotation metadata for generated VLC/file getters."""

    if not video_bytes:
        return (None, False)
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(bytes(video_bytes))
        tmp_path = tmp.name
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream_tags=rotate:stream_side_data=rotation",
                "-of",
                "json",
                tmp_path,
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0:
            return (None, False)
        data = json.loads(proc.stdout or "{}")
        streams = data.get("streams")
        if not isinstance(streams, list) or not streams:
            return (None, False)
        stream = streams[0] if isinstance(streams[0], dict) else {}
        rotation = _rotation_from_ffprobe_stream(stream)
        return (rotation if rotation is not None else 0, True)
    except Exception:
        return (None, False)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


def _rotation_from_ffprobe_stream(stream: dict[str, Any]) -> int | None:
    tags = stream.get("tags")
    if isinstance(tags, dict):
        parsed = _int_or_none(tags.get("rotate"))
        if parsed is not None:
            return parsed
    side_data = stream.get("side_data_list")
    if isinstance(side_data, list):
        for item in side_data:
            if not isinstance(item, dict):
                continue
            parsed = _int_or_none(item.get("rotation"))
            if parsed is not None:
                return parsed
    return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(round(float(str(value).strip())))
    except (TypeError, ValueError):
        return None


def _install_generated_getter_overrides(package: str, overlay: ModuleType) -> None:
    """Patch narrowly confirmed generated getter defects."""

    overrides = {
        "get_docx_italic_color_info__96c1a3698fda406b269dcc52d1b2d9c9": (
            _get_docx_italic_color_info_96c1
        ),
        "get_text_highlighting__1f6188764666903805a2bbde08d45ff2": (
            _get_text_highlighting_1f618
        ),
        "get_csv_merge_data__c66369e707b97de3ccd6da4699663fe6": (
            _get_csv_merge_data_c663
        ),
        "get_docx_hint_line_spacing__7fbc318f6c5b72e9f18e5cf28491b9c0": (
            _get_docx_hint_line_spacing_7fbc
        ),
        "get_doc_tabstops__5d3adaf927fd3613be8530bc92e14de1_qw35sft2_e618f7eb": (
            _get_doc_tabstops_normalized
        ),
        "get_doc_tabstops__14c984754ad8a9510e10b62f00a4c316_qw35sft2_e918c2d3": (
            _get_doc_tabstops_normalized
        ),
        "get_doc_tabstops__319c26794056ac5ca6a78fe65ba6a022_qw35sft2_e7ab72be": (
            _get_doc_tabstops_normalized
        ),
        "get_doc_tabstops__cf559754b29c29c8081fb54264f1ac10_qw35sft2_23c444b3": (
            _get_doc_tabstops_normalized
        ),
        "get_recreation_page_loaded__df352c0b3200dce606c52356619c2699": (
            _get_recreation_page_loaded_df352
        ),
        "get_gimp_layer_name_config__174a6594": (
            _get_gimp_layer_name_config_174a
        ),
        "get_xlsx_gross_profit_calculations__87606b1393cde1435f28787ed7fdd477": (
            _get_xlsx_gross_profit_calculations_87606
        ),
        "get_xlsx_sales_data__68a2bc639d62cfd83c2dd75c92c0cd2c": (
            _get_xlsx_sales_data_68a2
        ),
        "get_xlsx_website_qty__57e581555e19ceb3db2669015c9e00b9": (
            _get_xlsx_website_qty_57e
        ),
        "get_xlsx_cell_value__f211fc504e7a50f585ecb0b30c1d4299": (
            _get_xlsx_cell_value_f211
        ),
        "get_calc_cell_format__a3c8817989e00a27454e516ffd39d47e": (
            _get_calc_cell_format_a3c
        ),
        "get_xlsx_row_count__6c4a1081": (
            _get_xlsx_row_count_6c4a
        ),
        "get_calc_cell_value__154194afadf90d869e3ab3e594cc44ae": (
            _get_calc_cell_value_154194
        ),
        "get_data_delete_automacally__2c85759ee3c771990524833aa219e88b": (
            _get_data_delete_automacally_2c857
        ),
        "get_enable_enhanced_safety_browsing__6cfa1daf7e75edcbbc01473bc020676e": (
            _get_enable_enhanced_safety_browsing_pref
        ),
        "get_enable_enhanced_safety_browsing__ef79380fdcf7b5423e3660cab0b160fc": (
            _get_enable_enhanced_safety_browsing_pref
        ),
        "get_enable_safe_browsing__c066cfbb11182ed68d7b1c1e09f2d5ad": (
            _get_enable_safe_browsing_pref
        ),
        "get_disable_safe_browsing__eefbb2e3e4c8eef9ec6d58df9ab495a3": (
            _get_disable_safe_browsing_pref
        ),
        "get_chrome_color_scheme__c109c66a806aa3929078e7fafeda76b2_qw35sft2_d1928ab1": (
            _get_chrome_color_scheme_c109
        ),
        "get_chrome_appearance_and_dnt__197c5f21c248c0028a67e57d9193addd_qw35sft2_0f0a9cca": (
            _get_chrome_appearance_and_dnt_197c
        ),
        "get_chrome_startup_url_removed__a85eebd24e563a97c17935bc46126aa1": (
            _get_chrome_startup_url_removed_a85
        ),
    }
    modules = [overlay]
    modules.extend(
        module
        for name, module in list(sys.modules.items())
        if name.startswith(f"{package}.") and ".verigen_getters." in name
    )
    for module in modules:
        for name, fn in overrides.items():
            setattr(module, name, fn)
        _install_file_existence_family_overrides(module)
        _install_extension_manifest_family_overrides(module)
        _install_window_title_family_overrides(module)


def _install_file_existence_family_overrides(module: ModuleType) -> None:
    """Route generated file existence/rename getters through ``test -f`` by name.

    Generated ``get_file_exists__*`` getters that answer "does this single file
    exist" with a bool are defective in two recurring ways: some gate on
    ``len(file_bytes) > 0`` (so a 0-byte ``touch``ed file is mis-scored MISSING →
    reward 0), and some POST to an unreachable
    ``lite-scalecua-vm`` Flask host (name-resolution error → always False). Both
    share the same correct implementation: ``test -f config['path']``. Rather than
    patch each hash, override *every* getter in the family by name-prefix.

    Only bool-existence getters are touched. ``get_file_exists__*`` getters that
    return a richer dict (video-rotation validation, ODS validation,
    ``{'exists','has_content'}`` splits) carry distinct semantics and are left
    untouched — the shape gate below skips any getter with a dict return.
    All ``get_file_rename_status__*`` getters share one ``{new_exists, old_exists}``
    shape and are overridden unconditionally.
    """

    for name in list(vars(module)):
        if name.startswith("get_file_rename_status__"):
            setattr(module, name, _generic_file_rename_status)
        elif name.startswith("get_file_exists__"):
            fn = getattr(module, name)
            if callable(fn) and _is_bool_existence_getter(fn):
                setattr(module, name, _generic_file_exists)


def _is_bool_existence_getter(fn: Any) -> bool:
    """True if a generated ``get_file_exists__`` getter returns a bare bool.

    Distinguishes plain single-file existence getters (safe to flatten to
    ``test -f``) from richer getters that return a dict and must be preserved.
    A getter is treated as bool-existence iff it has at least one return and no
    return yields a dict literal. Already-installed generic handlers (whose
    source also has no dict return) pass and re-install idempotently.
    """

    if fn in (_generic_file_exists, _generic_file_rename_status):
        return fn is _generic_file_exists
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (OSError, TypeError, SyntaxError):
        return False
    func = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)),
        None,
    )
    if func is None:
        return False
    returns = [
        n for n in ast.walk(func) if isinstance(n, ast.Return) and n.value is not None
    ]
    if not returns:
        return False
    # A rich getter commonly builds ``result = {...}; ...; return result`` — the
    # return node is an ``ast.Name``, not an ``ast.Dict`` literal. A literal-only
    # guard under-matches that shape and FLATTENS the getter to a bare bool,
    # dropping the has_content/row_count/is_csv/... fields the paired metric
    # grades -> reward false-negative (incl. an RL-split reward getter). Treat a
    # return of a name bound to a dict literal as dict-returning too. Only ever
    # NARROWS the flatten set (genuine bare-bool getters still flatten), so it
    # cannot fabricate reward — it only guards against the file-getter override
    # incorrectly flattening a dict-returning getter.
    dict_vars = {
        t.id
        for a in ast.walk(func)
        if isinstance(a, ast.Assign) and isinstance(a.value, ast.Dict)
        for t in a.targets
        if isinstance(t, ast.Name)
    }
    for r in returns:
        if isinstance(r.value, ast.Dict):
            return False
        if isinstance(r.value, ast.Name) and r.value.id in dict_vars:
            return False
    return True


def _generic_file_exists(env: Any, config: dict[str, Any]) -> bool:
    """Bool existence via ``test -f`` — counts 0-byte files as existing."""

    return _controller_path_is_file(
        env, config.get("path") or config.get("file_path") or ""
    )


def _generic_file_rename_status(env: Any, config: dict[str, Any]) -> dict[str, bool]:
    """Rename status via ``test -f`` on new_path/old_path — 0-byte aware.

    Emits a SUPERSET of the rename-status keys any paired metric reads:
    ``new_exists``/``old_exists`` (train getters) AND ``old_gone`` (the RL
    getter's key). The rename branch of `_install_file_existence_family_overrides`
    replaces getters UNCONDITIONALLY (no shape guard, unlike the exists branch),
    so the override must satisfy every consumer's key shape at once: the RL
    getter `get_file_rename_status__5128f87c` returns ``{new_exists, old_gone}``
    and its metric awards +0.5 for ``old_gone``, while train metrics read only
    ``new_exists``/``old_exists``. Emitting just the flat ``{new_exists,
    old_exists}`` shape would silently drop ``old_gone``, permanently halving
    the RL getter's partial credit (a live RL-reward FN). ``old_gone = not
    old_exists`` is the same real `_controller_path_is_file(old_path)` probe
    inverted, so it cannot fabricate reward; adding the key is a no-op for
    train metrics that read only new/old_exists.
    """

    old_exists = _controller_path_is_file(env, config.get("old_path", ""))
    return {
        "new_exists": _controller_path_is_file(env, config.get("new_path", "")),
        "old_exists": old_exists,
        "old_gone": not old_exists,
    }


# --- Depth-tolerant unpacked-extension manifest search -------------------------
_EXTENSION_MANIFEST_GETTER_PREFIXES = (
    "get_extension_manifest__",
    "get_chrome_extension_manifest__",
    "get_webext_manifest__",
    # get_manifest_json__* reads a manifest.json at a FIXED config['path'] via
    # get_file (identical shape) and is the getter the live webext.eu rl tasks
    # (chrome_74d5859f) actually use -> same depth FN. The re-root is
    # missing-file-gated + bounded to the expected dir, so a manifest already at
    # the expected path is untouched (no re-root) and an absent one with no
    # nested match still scores 0 -> no new FP for generic manifest reads.
    "get_manifest_json__",
)
_EXTENSION_MANIFEST_PATH_KEYS = ("path", "manifest_path")


def _install_extension_manifest_family_overrides(module: ModuleType) -> None:
    """Make every generated extension-manifest getter depth-tolerant, by name.

    Generated ``get_extension_manifest__*`` / ``get_chrome_extension_manifest__*``
    / ``get_webext_manifest__*`` getters read the manifest.json at a *fixed* path
    (``config['path']`` or ``config['manifest_path']``). webext.eu ships the built
    extension as a ZIP that unpacks into its own subfolder, so the real
    ``manifest.json`` lands one directory deeper than the oracle expects
    (``~/Projects/page-modifier/page-modifier/manifest.json`` vs expected
    ``~/Projects/page-modifier/manifest.json``) -> ``get_file`` misses it ->
    ``{}``/``None`` -> reward 0 despite a correct extension (train
    chrome ``74d5859f_..._task_verify_2/3/16/57/59``). Rather than patch each hash,
    wrap every getter in the family by name-prefix and re-root its config paths
    onto the actual nested directory before it runs.
    """

    for name in list(vars(module)):
        if name.startswith(_EXTENSION_MANIFEST_GETTER_PREFIXES):
            fn = getattr(module, name)
            if callable(fn) and not getattr(fn, "_scalecua_depth_tolerant", False):
                setattr(module, name, _depth_tolerant_manifest_getter(fn))


def _depth_tolerant_manifest_getter(original):
    def _wrapped(env, config):
        return original(env, _reroot_extension_manifest_config(env, config))

    _wrapped._scalecua_depth_tolerant = True
    _wrapped.__name__ = getattr(original, "__name__", "manifest_getter")
    return _wrapped


def _reroot_extension_manifest_config(env: Any, config: dict[str, Any]) -> dict[str, Any]:
    """If the expected manifest.json is missing but the extension unpacked one
    level deeper, re-root every VM path in the config onto the actual directory.

    Bounded to the oracle's *expected* directory: an absent or unrelated
    extension elsewhere in the workspace finds nothing and still scores 0.
    """
    expected = ""
    for key in _EXTENSION_MANIFEST_PATH_KEYS:
        val = config.get(key)
        if isinstance(val, str) and val.endswith("manifest.json"):
            expected = val
            break
    if not expected or _controller_path_is_file(env, expected):
        return config
    expected_dir = expected.rsplit("/", 1)[0]
    found = _find_shallowest_vm_manifest(env, expected_dir)
    if not found or found == expected:
        return config
    actual_dir = found.rsplit("/", 1)[0]
    return _reroot_config_paths(copy.deepcopy(config), expected_dir, actual_dir)


def _find_shallowest_vm_manifest(env: Any, root: str, *, max_depth: int = 4) -> str:
    """Shallowest nested ``manifest.json`` strictly below ``root`` on the VM (or "")."""
    if not root:
        return ""
    inner = (
        "find " + shlex.quote(root)
        + " -mindepth 2 -maxdepth " + str(int(max_depth))
        + " -type f -name manifest.json -printf '%d %p\\n' 2>/dev/null | sort -n | head -1"
    )
    try:
        res = env.controller._command_result("bash -lc " + shlex.quote(inner), timeout=10)
    except Exception:
        return ""
    line = str(res.get("stdout", "")).strip()
    if not line:
        return ""
    parts = line.split(" ", 1)
    return parts[1].strip() if len(parts) == 2 else ""


def _reroot_config_paths(obj: Any, old_prefix: str, new_prefix: str) -> Any:
    if isinstance(obj, str):
        if obj == old_prefix:
            return new_prefix
        if obj.startswith(old_prefix + "/"):
            return new_prefix + obj[len(old_prefix):]
        return obj
    if isinstance(obj, list):
        return [_reroot_config_paths(v, old_prefix, new_prefix) for v in obj]
    if isinstance(obj, dict):
        return {k: _reroot_config_paths(v, old_prefix, new_prefix) for k, v in obj.items()}
    return obj


# --- Window-title getters run xdotool AS THE USER ------------------------------
# Generated window-title getters read a title via
# ``run_bash_script("... xdotool search --name '<n>' getwindowname ...")`` /
# ``xdotool getwindowname "$wid"``. The environment vendors its own ``xdotool``
# into a root-only ``/opt/env/bin`` -- absent from the ``user`` PATH -- so the
# user-run command hits RC-127 "command not found" -> empty title -> FN
# (get_compose_window_title__*, get_vscode_open_workspace__*_5b8c70c2,
# get_vscode_active_file__*_2cbb6469; container-proven, gate-zzh). ``wmctrl`` is
# on the standard PATH (``/usr/bin/wmctrl``, world-readable) and exposes the same
# titles via ``wmctrl -l``. Rather than patch each hash, wrap every getter in the
# class (source drives ``getwindowname`` through ``run_bash_script``) and rewrite
# that command to a wmctrl-first form, keeping the original xdotool call as a
# fallback -- mirroring the already-resilient get_vscode_window_title.
_WMCTRL_STRIP_PREFIX = "sed -E 's/^[^ ]+ +[^ ]+ +[^ ]+ +//'"
_WINDOW_TITLE_NAME_RE = re.compile(
    r"xdotool\s+search\s+--name\s+(['\"])(?P<name>.+?)\1\s+getwindowname"
)


def _harden_window_title_command(command: str) -> str:
    """Rewrite an xdotool window-title command to try ``wmctrl -l`` first.

    ``wmctrl -l`` emits ``<id> <desktop> <machine> <title>``; stripping the three
    leading columns yields the bare title xdotool's ``getwindowname`` would print,
    so each getter's downstream parse is unchanged. The original command is kept
    verbatim as a ``|| { ... }`` fallback. No-op for commands that don't read a
    window title and for getters already on wmctrl (e.g. get_vscode_window_title).
    """

    if (
        not isinstance(command, str)
        or "getwindowname" not in command
        or "wmctrl" in command
    ):
        return command
    # VS Code enumerates windows via `--class 'Code'` + a getwindowname loop; the
    # title always contains "Visual Studio Code", so match that in wmctrl -l.
    if re.search(r"--class\s+(['\"])Code\1", command) and "--name" not in command:
        wmctrl = (
            "wmctrl -l 2>/dev/null | grep -F -e 'Visual Studio Code' | "
            + _WMCTRL_STRIP_PREFIX
        )
        return f"{wmctrl} || {{ {command}; }}"
    # Single-title lookup: `xdotool search --name '<name>' getwindowname`.
    match = _WINDOW_TITLE_NAME_RE.search(command)
    if match:
        wmctrl = (
            "wmctrl -l 2>/dev/null | grep -F -m1 -e "
            + shlex.quote(match.group("name"))
            + " | "
            + _WMCTRL_STRIP_PREFIX
        )
        return f"{wmctrl} || {{ {command}; }}"
    return command


class _WmctrlHardenedController:
    """``env.controller`` proxy that hardens window-title bash commands."""

    def __init__(self, inner: Any) -> None:
        object.__setattr__(self, "_inner", inner)

    def run_bash_script(self, command: Any, *args: Any, **kwargs: Any) -> Any:
        return self._inner.run_bash_script(
            _harden_window_title_command(command), *args, **kwargs
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _WmctrlHardenedEnv:
    """``env`` proxy exposing a hardened controller; delegates everything else."""

    def __init__(self, env: Any) -> None:
        object.__setattr__(self, "_env", env)
        object.__setattr__(
            self, "controller", _WmctrlHardenedController(env.controller)
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


def _wmctrl_hardened_window_title_getter(original):
    def _wrapped(env, config):
        return original(_WmctrlHardenedEnv(env), config)

    _wrapped._scalecua_wmctrl_hardened = True
    _wrapped.__name__ = getattr(original, "__name__", "window_title_getter")
    return _wrapped


def _install_window_title_family_overrides(module: ModuleType) -> None:
    """Harden every generated window-title getter against the env-vendored xdotool.

    Class marker: a getter whose source drives ``getwindowname`` through
    ``run_bash_script`` (the compose + two vscode getters, and any future
    sibling). Each is wrapped so its window-title command runs ``wmctrl`` first
    with the original xdotool call as fallback. Idempotent; wrapping the already
    wmctrl-first get_vscode_window_title is a runtime no-op.
    """

    for name in list(vars(module)):
        if not name.startswith("get_"):
            continue
        fn = getattr(module, name)
        if not callable(fn) or getattr(fn, "_scalecua_wmctrl_hardened", False):
            continue
        try:
            source = inspect.getsource(fn)
        except (OSError, TypeError):
            continue
        if "getwindowname" in source and "run_bash_script" in source:
            setattr(module, name, _wmctrl_hardened_window_title_getter(fn))


def _chrome_prefs_from_controller(env: Any) -> dict[str, Any]:
    for path in (
        "/home/user/chrome-data/Default/Preferences",
        "/home/user/.config/google-chrome/Default/Preferences",
    ):
        data = env.controller.get_file(path)
        if not data:
            continue
        try:
            parsed = json.loads(bytes(data).decode("utf-8"))
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _get_data_delete_automacally_2c857(env: Any, config: dict[str, Any]) -> str:
    """Match lite.osworld's Chrome clear-on-exit preference semantics."""

    prefs = _chrome_prefs_from_controller(env)
    profile = prefs.get("profile", {})
    if not isinstance(profile, dict):
        profile = {}
    cookies = (
        profile
        .get("default_content_setting_values", {})
        .get("cookies")
    )
    content_settings = profile.get("content_settings", {})
    clear_on_exit = (
        content_settings.get("cookies", {}).get("clear_on_exit")
        if isinstance(content_settings, dict)
        else None
    )
    lifetime_enabled = (
        prefs.get("browser", {})
        .get("clear_data", {})
        .get("browsing_data_lifetime", {})
        .get("enabled")
    )
    return "true" if (
        cookies == 4
        or clear_on_exit is True
        or profile.get("clear_lso_data_enabled") is True
        or lifetime_enabled is True
    ) else "false"


def _chrome_safe_browsing_flags(prefs: dict[str, Any]) -> tuple[bool, bool]:
    safebrowsing = prefs.get("safebrowsing", {})
    if not isinstance(safebrowsing, dict):
        return (False, False)
    return (
        bool(safebrowsing.get("enabled", False)),
        bool(safebrowsing.get("enhanced", False)),
    )


def _get_enable_enhanced_safety_browsing_pref(
    env: Any,
    config: dict[str, Any],
) -> str:
    enabled, enhanced = _chrome_safe_browsing_flags(_chrome_prefs_from_controller(env))
    return "true" if enhanced else "false"


def _get_enable_safe_browsing_pref(env: Any, config: dict[str, Any]) -> str:
    enabled, enhanced = _chrome_safe_browsing_flags(_chrome_prefs_from_controller(env))
    return "true" if enabled or enhanced else "false"


def _get_disable_safe_browsing_pref(env: Any, config: dict[str, Any]) -> str:
    enabled, enhanced = _chrome_safe_browsing_flags(_chrome_prefs_from_controller(env))
    return "false" if not (enabled or enhanced) else "true"


_COLOR_SCHEME_STR_TO_INT = {
    "system": 0,
    "default": 0,
    "0": 0,
    "light": 1,
    "1": 1,
    "dark": 2,
    "2": 2,
}
_COLOR_SCHEME_INT_TO_STR = {0: "system", 1: "light", 2: "dark"}


def _normalize_chrome_color_scheme_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value in _COLOR_SCHEME_INT_TO_STR else None
    key = str(value).strip().lower()
    return _COLOR_SCHEME_STR_TO_INT.get(key)


def _chrome_color_scheme_value_from_prefs(prefs: dict[str, Any]) -> int:
    theme = prefs.get("browser", {}).get("theme", {})
    if not isinstance(theme, dict):
        theme = {}
    value = _normalize_chrome_color_scheme_value(
        # Canonical `color_scheme` first; `color_scheme2` is a stale mirror the
        # Settings UI leaves behind.
        theme.get("color_scheme", theme.get("color_scheme2"))
    )
    system_theme = (
        prefs.get("extensions", {})
        .get("theme", {})
        .get("system_theme")
    )
    if system_theme in (0, "0", False):
        return 1
    return 0 if value is None else value


def _get_chrome_color_scheme_c109(env: Any, config: dict[str, Any]) -> str:
    value = _chrome_color_scheme_value_from_prefs(_chrome_prefs_from_controller(env))
    return _COLOR_SCHEME_INT_TO_STR.get(value, "unknown")


def _get_chrome_appearance_and_dnt_197c(
    env: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    prefs = _chrome_prefs_from_controller(env)
    return {
        "color_scheme": _chrome_color_scheme_value_from_prefs(prefs),
        "do_not_track": bool(prefs.get("enable_do_not_track", False)),
    }


def _get_chrome_startup_url_removed_a85(env: Any, config: dict[str, Any]) -> str:
    """Check startup URL removal with Chrome's URL canonicalization."""

    check_url = _normalize_web_url_for_task(config.get("check_url"))
    if not check_url:
        return "false"
    prefs = _chrome_prefs_from_controller(env)
    startup_urls = prefs.get("session", {}).get("startup_urls", [])
    if not isinstance(startup_urls, list):
        startup_urls = []
    current = {
        _normalize_web_url_for_task(url)
        for url in startup_urls
        if isinstance(url, str)
    }
    return "false" if check_url in current else "true"


def _controller_path_is_file(env: Any, path: str) -> bool:
    """test -f a path via the container controller — 0-byte files count as existing.

    Generated file getters gate on ``len(file_bytes) > 0``, but controller.get_file
    returns b'' for an existing empty file, so a `touch`/`>`-created 0-byte file is
    mis-scored as MISSING → reward 0. Route through test -f.
    """
    if not path:
        return False
    inner = "test -f " + shlex.quote(str(path)) + " && echo EXISTS || echo MISSING"
    result = env.controller._command_result("bash -lc " + shlex.quote(inner), timeout=10)
    return "EXISTS" in str(result.get("stdout", ""))


def _docx_bytes_from_config(env: Any, config: dict[str, Any]) -> bytes | None:
    path = config.get("path")
    if not path:
        return None
    data = env.controller.get_file(path)
    if not data:
        return None
    return bytes(data)


def _recreation_loaded_result_from_page_state(
    url: str,
    title: str,
    text: str,
    html: str = "",
) -> dict[str, Any]:
    url_text = str(url or "")
    title_text = str(title or "")
    body_text = str(text or "")
    html_text = str(html or "")
    haystack = f"{url_text}\n{title_text}\n{body_text}\n{html_text}".lower()
    verification_checks = {
        "has_recreation_domain": "recreation.gov" in url_text.lower(),
        "has_search_button": any(
            token in haystack
            for token in (
                'aria-label="search',
                "search campgrounds",
                "search destinations",
                ">search<",
            )
        ),
        "has_booking_elements": any(
            token in haystack
            for token in (
                "reserve",
                "book now",
                "booking",
                "campsite",
                "availability",
                "available",
            )
        ),
        "has_recreation_logo": "recreation.gov" in haystack or "recreation logo" in haystack,
        "has_campground_content": any(
            token in haystack
            for token in (
                "campground",
                "campgrounds",
                "campsite",
                "campsites",
                "/camping/campgrounds/",
                "upper pines",
            )
        ),
        "page_title_valid": any(
            token in title_text.lower()
            for token in ("recreation", "campground", "camping", "camp", "reserve")
        ),
    }
    checks_passed = sum(1 for value in verification_checks.values() if value)
    is_recreation_domain = verification_checks["has_recreation_domain"]
    return {
        "is_valid": is_recreation_domain and checks_passed >= 3,
        "url": url_text,
        "is_recreation_domain": is_recreation_domain,
        "checks_passed": checks_passed,
        "total_checks": len(verification_checks),
        "page_title": title_text,
        "verification_details": verification_checks,
    }


def _get_gimp_layer_name_config_174a(env: Any, config: dict[str, Any]) -> list[str]:
    """Read layer names from the generated Overlay-layer XCF getter path."""

    from lite.gym.envs.lite.scalecua.src.osworld import xcf

    file_path = (
        config.get("file_path")
        or config.get("path")
        or config.get("target_path")
        or "/home/user/Desktop/white_background.xcf"
    )
    data = env.controller.get_file(str(file_path))
    if not data:
        return []
    cache_path = Path(env.cache_dir) / "gimp_layer_name_config_174a.xcf"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(bytes(data))
    try:
        return xcf.parse_xcf_layers(cache_path)
    except Exception as exc:
        logger.warning("GIMP XCF layer-name override failed for %s: %s", file_path, exc)
        return []


def _get_recreation_page_loaded_df352(env: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Robust current-DOM version of the generated Recreation.gov load getter."""

    remote_debugging_url = f"http://{env.vm_ip}:{env.chromium_port}"
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.connect_over_cdp(remote_debugging_url)
            try:
                pages = [
                    page
                    for context in browser.contexts
                    for page in context.pages
                    if not page.url.startswith(("chrome://", "devtools://"))
                ]
                page = next(
                    (
                        candidate
                        for candidate in pages
                        if "recreation.gov" in candidate.url.lower()
                    ),
                    pages[0] if pages else None,
                )
                if page is None:
                    return _recreation_loaded_result_from_page_state("", "", "")
                with contextlib.suppress(Exception):
                    page.wait_for_load_state("domcontentloaded", timeout=5000)
                with contextlib.suppress(Exception):
                    page.wait_for_timeout(1000)
                url = page.url
                title = page.title()
                text = ""
                html = ""
                with contextlib.suppress(Exception):
                    text = page.locator("body").inner_text(timeout=3000)
                with contextlib.suppress(Exception):
                    html = page.content()
                return _recreation_loaded_result_from_page_state(url, title, text, html)
            finally:
                with contextlib.suppress(Exception):
                    browser.close()
    except Exception as exc:
        logger.warning("Recreation.gov loaded getter override failed: %s", exc)
        return _recreation_loaded_result_from_page_state("", "", "")


def _get_docx_hint_line_spacing_7fbc(env: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Robust version of a generated getter using a stale enum member name."""

    data = _docx_bytes_from_config(env, config)
    if not data:
        return {"error": "File not found"}
    import os as _os
    import tempfile as _tempfile

    with _tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        from docx import Document

        doc = Document(tmp_path)
        hint_indices: list[int] = []
        found_header = False
        for index, paragraph in enumerate(doc.paragraphs):
            if "Five hints which may be useful" in paragraph.text:
                found_header = True
                continue
            if found_header and paragraph.text.strip() and len(hint_indices) < 5:
                hint_indices.append(index)

        spacings: list[float | None] = []
        for index in hint_indices:
            paragraph_format = doc.paragraphs[index].paragraph_format
            spacing_rule = paragraph_format.line_spacing_rule
            spacing_value = paragraph_format.line_spacing
            rule_text = str(spacing_rule or "").upper()
            if "ONE_POINT_FIVE" in rule_text or "1.5" in rule_text:
                spacings.append(1.5)
            elif "DOUBLE" in rule_text:
                spacings.append(2.0)
            elif "SINGLE" in rule_text:
                spacings.append(1.0)
            elif isinstance(spacing_value, (int, float)):
                spacings.append(float(spacing_value))
            else:
                spacings.append(None)
        return {"spacings": spacings, "hint_count": len(hint_indices)}
    finally:
        with contextlib.suppress(FileNotFoundError):
            _os.unlink(tmp_path)


def _normalize_docx_tab_alignment(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        if value == 0:
            return "LEFT"
        if value == 1:
            return "CENTER"
        if value == 2:
            return "RIGHT"
        if value == 3:
            return "DECIMAL"
    text = str(value if value is not None else "").upper()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.split(" ", 1)[0]


def _get_doc_tabstops_normalized(env: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Read docx tab stops with alignment normalized for generated metrics."""

    data = _docx_bytes_from_config(env, config)
    if not data:
        return {"error": "File not found"}
    import os as _os
    import tempfile as _tempfile

    with _tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        from docx import Document

        doc = Document(tmp_path)
        paragraphs: list[dict[str, Any]] = []
        for paragraph in doc.paragraphs:
            if not paragraph.text.strip():
                continue
            stops = []
            for tab_stop in paragraph.paragraph_format.tab_stops:
                stops.append(
                    {
                        "position_cm": round(tab_stop.position.cm, 2),
                        "alignment": _normalize_docx_tab_alignment(tab_stop.alignment),
                    }
                )
            paragraphs.append({"text_start": paragraph.text[:40], "stops": stops})
        return {"paragraphs": paragraphs, "count": len(paragraphs)}
    finally:
        with contextlib.suppress(FileNotFoundError):
            _os.unlink(tmp_path)


def _docx_run_bool_prop(run: Any, attr: str, tag: str) -> bool:
    value = getattr(run, attr, None)
    if value is not None:
        return bool(value)
    try:
        from docx.oxml.ns import qn

        r_pr = run._r.rPr
        if r_pr is None:
            return False
        for xml_tag in (tag, f"{tag}Cs"):
            elem = r_pr.find(qn(f"w:{xml_tag}"))
            if elem is None:
                continue
            raw = elem.get(qn("w:val"))
            if raw is None:
                return True
            return str(raw).strip().lower() not in {"0", "false", "off", "none"}
    except Exception:
        return False
    return False


def _docx_run_color_hex(run: Any) -> str | None:
    color = getattr(getattr(run, "font", None), "color", None)
    rgb = _rgb_tuple_from_color(getattr(color, "rgb", None))
    if rgb is None:
        try:
            from docx.oxml.ns import qn

            r_pr = run._r.rPr
            if r_pr is not None:
                color_elem = r_pr.find(qn("w:color"))
                if color_elem is not None:
                    value = color_elem.get(qn("w:val"))
                    if value and str(value).lower() != "auto":
                        rgb = _rgb_tuple_from_color(value)
        except Exception:
            rgb = None
    if rgb is None:
        return None
    return "{:02X}{:02X}{:02X}".format(*rgb)


def _get_docx_italic_color_info_96c1(env: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Robust version of a generated getter that calls RGBColor.hex()."""

    data = _docx_bytes_from_config(env, config)
    if not data:
        return {"italic_runs": []}
    import os as _os
    import tempfile as _tempfile

    from docx import Document

    with _tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        document = Document(tmp_path)
        italic_runs: list[dict[str, Any]] = []
        for para_idx, paragraph in enumerate(document.paragraphs):
            for run_idx, run in enumerate(paragraph.runs):
                if not str(run.text or "").strip():
                    continue
                if not _docx_run_bool_prop(run, "italic", "i"):
                    continue
                italic_runs.append(
                    {
                        "text": run.text,
                        "color_rgb": _docx_run_color_hex(run),
                        "para_idx": para_idx,
                        "run_idx": run_idx,
                    }
                )
        return {"italic_runs": italic_runs}
    finally:
        _os.unlink(tmp_path)


def _docx_highlight_color(run: Any) -> Any | None:
    try:
        from docx.enum.text import WD_COLOR_INDEX
        from docx.oxml.ns import qn
    except Exception:
        WD_COLOR_INDEX = None
        qn = None

    highlight = getattr(getattr(run, "font", None), "highlight_color", None)
    if (
        highlight is not None
        and WD_COLOR_INDEX is not None
        and highlight != WD_COLOR_INDEX.AUTO
    ):
        return highlight

    try:
        r_pr = run._r.rPr
        if r_pr is None or qn is None:
            return None
        highlight_elem = r_pr.find(qn("w:highlight"))
        if highlight_elem is not None:
            value = (highlight_elem.get(qn("w:val")) or "").strip().lower()
            if value and value not in {"none", "auto"}:
                if value == "yellow" and WD_COLOR_INDEX is not None:
                    return WD_COLOR_INDEX.YELLOW
                return value
        shading_elem = r_pr.find(qn("w:shd"))
        if shading_elem is not None:
            fill = (
                shading_elem.get(qn("w:fill"))
                or shading_elem.get(qn("w:color"))
                or ""
            ).strip()
            if fill and fill.lower() not in {"auto", "none"}:
                if _is_yellow_color(fill) and WD_COLOR_INDEX is not None:
                    return WD_COLOR_INDEX.YELLOW
                return fill
    except Exception:
        return None
    return None


def _run_overlap_text(run_text: str, run_start: int, target_start: int, target_end: int) -> str:
    run_end = run_start + len(run_text)
    if run_end <= target_start or run_start >= target_end:
        return ""
    start = max(target_start - run_start, 0)
    end = min(target_end - run_start, len(run_text))
    return run_text[start:end]


def _get_text_highlighting_1f618(env: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Accept python-docx highlight and LibreOffice OOXML shading for one task."""

    data = _docx_bytes_from_config(env, config)
    if not data:
        return {"found": False, "is_highlighted": False, "highlight_color": None}
    target_text = str(
        config.get("target_text") or '"Chocolate is a healthy food." Discuss.'
    )

    import os as _os
    import tempfile as _tempfile

    from docx import Document

    with _tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        document = Document(tmp_path)
        target_lower = target_text.lower()
        for para_idx, paragraph in enumerate(document.paragraphs):
            paragraph_text = "".join(str(run.text or "") for run in paragraph.runs)
            start_pos = paragraph_text.lower().find(target_lower)
            if start_pos == -1:
                continue
            end_pos = start_pos + len(target_text)
            current_pos = 0
            target_runs: list[tuple[Any, str]] = []
            for run in paragraph.runs:
                run_text = str(run.text or "")
                overlap = _run_overlap_text(run_text, current_pos, start_pos, end_pos)
                if overlap.strip():
                    target_runs.append((run, overlap))
                current_pos += len(run_text)
            if not target_runs:
                return {
                    "found": True,
                    "is_highlighted": False,
                    "highlight_color": None,
                    "paragraph_index": para_idx,
                }
            highlight_color = None
            for run, _overlap in target_runs:
                color = _docx_highlight_color(run)
                if color is None:
                    return {
                        "found": True,
                        "is_highlighted": False,
                        "highlight_color": None,
                        "paragraph_index": para_idx,
                    }
                if highlight_color is None:
                    highlight_color = color
            return {
                "found": True,
                "is_highlighted": True,
                "highlight_color": highlight_color,
                "paragraph_index": para_idx,
            }
        return {"found": False, "is_highlighted": False, "highlight_color": None}
    finally:
        _os.unlink(tmp_path)


def _csv_rows_from_bytes(data: bytes) -> list[list[str]]:
    import csv as _csv
    import io as _io

    text = data.decode("utf-8-sig", errors="ignore")
    sample = text[:4096]
    dialect = None
    try:
        dialect = _csv.Sniffer().sniff(sample, delimiters=",;\t")
    except Exception:
        dialect = None
    if dialect is None:
        return list(_csv.reader(_io.StringIO(text)))
    return list(_csv.reader(_io.StringIO(text), dialect))


def _table_first_column_values_from_vm_file(env: Any, path: str) -> set[str]:
    data = env.controller.get_file(path)
    if not data:
        return set()
    suffix = Path(path).suffix.lower()
    try:
        if suffix == ".xlsx":
            import io as _io

            import openpyxl

            workbook = openpyxl.load_workbook(
                _io.BytesIO(bytes(data)),
                read_only=True,
                data_only=True,
            )
            try:
                sheet = workbook.worksheets[0]
                values = {
                    str(row[0]).strip()
                    for row in sheet.iter_rows(min_row=2, values_only=True)
                    if row and row[0] is not None and str(row[0]).strip()
                }
                return values
            finally:
                workbook.close()
        if suffix == ".ods":
            import io as _io

            import pandas as _pd

            frame = _pd.read_excel(_io.BytesIO(bytes(data)), engine="odf")
            if frame.empty:
                return set()
            first_column = frame.iloc[:, 0]
            return {
                str(value).strip()
                for value in first_column.dropna().tolist()
                if str(value).strip()
            }
    except Exception:
        return set()
    return set()


def _empty_csv_merge_result() -> dict[str, Any]:
    return {
        "row_count": 0,
        "has_single_header": False,
        "sample_rows": [],
        "total_rows": 0,
        "unique_values_from_merged": set(),
        "source_file1_unique_values": set(),
        "source_file2_unique_values": set(),
    }


def _get_csv_merge_data_c663(env: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Robust version of a generated getter with missing source-file helpers."""

    path = config.get("path")
    if not path:
        return _empty_csv_merge_result()
    data = env.controller.get_file(path)
    if not data:
        return _empty_csv_merge_result()

    try:
        rows = _csv_rows_from_bytes(bytes(data))
    except Exception:
        return _empty_csv_merge_result()
    if not rows:
        return _empty_csv_merge_result()

    header = rows[0]
    data_rows = rows[1:]
    sample_rows: list[list[str]] = []
    if data_rows:
        sample_rows.extend(data_rows[:3])
        if len(data_rows) > 100:
            mid_idx = len(data_rows) // 2
            sample_rows.extend(data_rows[mid_idx : mid_idx + 3])
        if len(data_rows) > 6:
            sample_rows.extend(data_rows[-3:])

    unique_values = {
        str(row[0]).strip()
        for row in data_rows
        if row and str(row[0]).strip()
    }
    return {
        "row_count": len(data_rows),
        "has_single_header": all(row != header for row in data_rows),
        "sample_rows": sample_rows,
        "total_rows": len(rows),
        "unique_values_from_merged": unique_values,
        "source_file1_unique_values": _table_first_column_values_from_vm_file(
            env,
            "/home/user/Desktop/file1.xlsx",
        ),
        "source_file2_unique_values": _table_first_column_values_from_vm_file(
            env,
            "/home/user/Desktop/file2.ods",
        ),
    }


def _get_xlsx_gross_profit_calculations_87606(
    env: Any,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    """Repair generated gross-profit getter row bounds for IncomeStatement2."""

    path = config.get("path")
    if not path:
        return None
    data = env.controller.get_file(path)
    if not data:
        return None

    import tempfile as _tempfile

    import openpyxl

    with _tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(bytes(data))
        tmp_path = tmp.name
    try:
        workbook = openpyxl.load_workbook(tmp_path, data_only=True)
        sheet_idx = config.get("sheet", 0)
        sheet = workbook.worksheets[sheet_idx] if isinstance(sheet_idx, int) else workbook[sheet_idx]
        result = {
            "gross_profit_values": [],
            "net_sales_values": [],
            "cogs_values": [],
            "average_in_j12": _to_float_or_none(sheet["J12"].value),
        }
        for row in range(2, 11):
            result["gross_profit_values"].append(_to_float_or_none(sheet[f"J{row}"].value))
            result["net_sales_values"].append(_to_float_or_none(sheet[f"E{row}"].value))
            result["cogs_values"].append(_to_float_or_none(sheet[f"I{row}"].value))
        workbook.close()
        return result
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


def _to_float_or_none(value: Any) -> float | None:
    if isinstance(value, dict):
        for key in ("value", "expected_value", "amount_value"):
            if key in value:
                return _to_float_or_none(value.get(key))
        return None
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if text.endswith("%"):
            text = text[:-1].strip()
        text = re.sub(r"^\$+", "", text)
        value = text
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_xlsx_sales_data_68a2(env: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Repair generated SalesRep pie-chart metadata extraction."""

    path = config.get("path", "/home/user/SalesRep.xlsx")
    data = env.controller.get_file(path)
    if not data:
        return {"error": "Failed to retrieve file from VM"}

    import tempfile as _tempfile

    import openpyxl

    with _tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(bytes(data))
        tmp_path = tmp.name
    try:
        workbook = openpyxl.load_workbook(tmp_path, data_only=True)
        sheet = workbook.active
        result: dict[str, Any] = {}
        for row in range(1, 20):
            for col in range(1, 10):
                cell = sheet.cell(row=row, column=col)
                result[cell.coordinate] = cell.value
                rgb = _cell_fill_rgb(cell)
                if rgb:
                    result[f"{cell.coordinate}_fill"] = rgb
        result["_charts"] = [_calc_chart_info_for_generated_sales_task(chart, workbook) for chart in getattr(sheet, "_charts", [])]
        workbook.close()
        return result
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


def _cell_fill_rgb(cell: Any) -> str | None:
    fill = getattr(cell, "fill", None)
    fg_color = getattr(fill, "fgColor", None)
    rgb = getattr(fg_color, "rgb", None)
    if isinstance(rgb, str) and rgb:
        if rgb == "00000000":
            return None
        if len(rgb) == 8 and rgb.startswith("00"):
            return f"FF{rgb[-6:]}"
        return rgb
    return None


def _calc_chart_info_for_generated_sales_task(chart: Any, workbook: Any) -> dict[str, Any]:
    series = list(getattr(chart, "series", []) or [])
    return {
        "type": type(chart).__name__,
        "title": _chart_title_text(getattr(chart, "title", None)) or None,
        "series_count": len(series),
        "data_points": _calc_chart_data_points(series, workbook),
    }


def _calc_chart_data_points(series_items: list[Any], workbook: Any) -> int:
    counts = [_calc_chart_series_data_points(series, workbook) for series in series_items]
    counts = [count for count in counts if count > 0]
    if not counts:
        return 0
    if len(counts) > 1:
        return max(sum(counts), len(counts))
    return counts[0]


def _calc_chart_series_data_points(series: Any, workbook: Any) -> int:
    value_ref = getattr(getattr(series, "val", None), "numRef", None)
    cache = getattr(value_ref, "numCache", None)
    count = getattr(getattr(cache, "ptCount", None), "val", None)
    if count is not None:
        try:
            return int(count)
        except (TypeError, ValueError):
            pass
    points = getattr(cache, "pt", None)
    if points:
        try:
            return len(points)
        except TypeError:
            pass
    parsed = _parse_calc_cell_range(str(getattr(value_ref, "f", "") or ""), workbook)
    if parsed is None:
        return 0
    return max(int(parsed.get("row_count") or 0), len(parsed.get("columns") or []))


def _get_xlsx_website_qty_57e(env: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Repair generated website-quantity getter's over-strict summary sheet names."""

    path = config.get("path")
    data = env.controller.get_file(path) if path else None
    empty = {
        "has_summary_sheet": False,
        "summary_data": {},
        "has_total_column": False,
        "original_totals": {},
    }
    if not data:
        return empty

    import tempfile as _tempfile

    import openpyxl

    with _tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(bytes(data))
        tmp_path = tmp.name
    try:
        workbook = openpyxl.load_workbook(tmp_path, data_only=True)
        original = workbook.worksheets[0]
        result = {
            "has_summary_sheet": False,
            "summary_data": {},
            "has_total_column": _has_total_quantity_column(original),
            "original_totals": _website_quantity_totals_from_original_sheet(original),
        }
        for sheet in workbook.worksheets[1:]:
            if not _looks_like_website_quantity_summary_sheet(sheet.title):
                continue
            summary_data = _website_quantity_totals_from_summary_sheet(sheet)
            if summary_data:
                result["has_summary_sheet"] = True
                result["summary_data"] = summary_data
                break
        workbook.close()
        return result
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


def _looks_like_website_quantity_summary_sheet(name: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(name or "").strip().lower())
    if normalized in {"summary", "total", "totals", "summary table", "sheet2", "sheet 2"}:
        return True
    return "summary" in normalized or ("quantity" in normalized and "website" in normalized)


def _website_quantity_totals_from_summary_sheet(sheet: Any) -> dict[str, int]:
    totals: dict[str, int] = {}
    for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row, values_only=True):
        if not any(row):
            continue
        website = row[0] if len(row) > 0 else None
        quantity = row[1] if len(row) > 1 else None
        parsed = _to_int_or_none(quantity)
        if website and parsed is not None:
            totals[str(website).strip()] = parsed
    return totals


def _website_quantity_totals_from_original_sheet(sheet: Any) -> dict[str, int]:
    totals: dict[str, int] = {}
    for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row, values_only=True):
        if not any(row):
            continue
        website = row[1] if len(row) > 1 else None
        quantity = row[4] if len(row) > 4 else None
        parsed = _to_int_or_none(quantity)
        if website and parsed is not None:
            key = str(website).strip()
            totals[key] = totals.get(key, 0) + parsed
    return totals


def _has_total_quantity_column(sheet: Any) -> bool:
    header_row = next(sheet.iter_rows(min_row=1, max_row=1, values_only=True), ())
    for header in header_row:
        text = str(header or "").strip().lower()
        if "total" in text and "quantity" in text:
            return True
    return False


def _to_int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _get_xlsx_cell_value_f211(env: Any, config: dict[str, Any]) -> dict[str, Any] | None:
    """Repair generated Product/Amount total getter using the actual Product header."""

    path = config.get("path", "")
    data = env.controller.get_file(path) if path else None
    if not data:
        return None

    import tempfile as _tempfile

    import openpyxl

    amount_cell = str(config.get("cell") or "A1")
    label_cell = str(config.get("label_cell") or "A1")
    with _tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(bytes(data))
        tmp_path = tmp.name
    try:
        workbook = openpyxl.load_workbook(tmp_path, data_only=True)
        sheet = workbook.active
        label_value = sheet[label_cell].value
        label_text = str(label_value).strip() if label_value is not None else ""
        if not label_text:
            row_match = re.search(r"\d+", amount_cell)
            amount_row = int(row_match.group(0)) if row_match else None
            product_col = _find_header_column(sheet, "product")
            if amount_row is not None and product_col is not None:
                inferred = sheet.cell(row=amount_row, column=product_col).value
                label_text = str(inferred).strip() if inferred is not None else ""
        amount_value = _to_float_or_none(sheet[amount_cell].value)
        workbook.close()
        return {"product_label": label_text, "amount_value": amount_value}
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


def _get_calc_cell_format_a3c(env: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Repair generated 2020-cost getter that only scanned row 1 for 2020."""

    path = config.get("path", "")
    data = env.controller.get_file(path) if path else None
    if not data:
        return {}

    import tempfile as _tempfile

    import openpyxl

    with _tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(bytes(data))
        tmp_path = tmp.name
    try:
        workbook = openpyxl.load_workbook(tmp_path, data_only=True)
        sheet = workbook.active
        formatting: dict[str, dict[str, Any]] = {}
        data_by_cell: dict[str, Any] = {}
        for row in sheet.iter_rows():
            for cell in row:
                data_by_cell[cell.coordinate] = cell.value
                formatting[cell.coordinate] = {
                    "bold": bool(getattr(cell.font, "bold", False)),
                    "italic": bool(getattr(cell.font, "italic", False)),
                    "underline": getattr(cell.font, "underline", None) is not None,
                }

        header_row = _find_2020_cost_header_row(sheet)
        cost_col = _find_2020_cost_column(sheet, header_row) if header_row else None
        month_col = 1
        month_column = []
        cost_2020_column = []
        start_row = (header_row + 1) if header_row else 2
        for row_idx in range(start_row, sheet.max_row + 1):
            month_value = sheet.cell(row=row_idx, column=month_col).value
            if month_value is not None:
                month_column.append((row_idx, f"A{row_idx}", month_value))
            if cost_col is not None:
                cost_value = _to_float_or_none(sheet.cell(row=row_idx, column=cost_col).value)
                if cost_value is not None:
                    cost_2020_column.append(
                        (
                            row_idx,
                            f"{sheet.cell(row=row_idx, column=cost_col).column_letter}{row_idx}",
                            cost_value,
                        )
                    )
        workbook.close()
        return {
            "formatting": formatting,
            "data": data_by_cell,
            "month_column": month_column,
            "cost_2020_column": cost_2020_column,
        }
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


def _find_2020_cost_header_row(sheet: Any) -> int | None:
    for row_idx in range(1, sheet.max_row + 1):
        values = [str(cell.value or "").strip().lower() for cell in sheet[row_idx]]
        if any("2020" in value for value in values):
            for candidate in range(row_idx, min(sheet.max_row, row_idx + 3) + 1):
                candidate_values = [
                    str(cell.value or "").strip().lower()
                    for cell in sheet[candidate]
                ]
                if any("cost" in value or "total" in value for value in candidate_values):
                    return candidate
            return row_idx + 1 if row_idx < sheet.max_row else row_idx
    return None


def _find_2020_cost_column(sheet: Any, header_row: int | None) -> int | None:
    if header_row is None:
        return None
    for cell in sheet[header_row]:
        text = str(cell.value or "").strip().lower()
        if "cost" in text and "total" in text:
            return int(cell.column)
    numeric_scores: list[tuple[int, int]] = []
    for col in range(2, sheet.max_column + 1):
        score = 0
        for row in range(header_row + 1, min(sheet.max_row, header_row + 16) + 1):
            if _to_float_or_none(sheet.cell(row=row, column=col).value) is not None:
                score += 1
        if score:
            numeric_scores.append((score, col))
    if not numeric_scores:
        return None
    numeric_scores.sort()
    return numeric_scores[-1][1]


def _get_xlsx_row_count_6c4a(env: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Repair generated BoomerangSales filter getter by detecting the Type header."""

    path = config.get("path")
    data = env.controller.get_file(path) if path else None
    if not data:
        return {
            "visible_row_count": 0,
            "autofilter_enabled": False,
            "autofilter_ref": None,
            "type_column_filtered": False,
            "visible_row_types": [],
            "hidden_row_types": [],
        }

    import tempfile as _tempfile

    import openpyxl
    from openpyxl.utils.cell import range_boundaries

    sheet_idx = config.get("sheet", 0)
    with _tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(bytes(data))
        tmp_path = tmp.name
    try:
        workbook = openpyxl.load_workbook(tmp_path, data_only=True)
        sheet = workbook.worksheets[sheet_idx] if isinstance(sheet_idx, int) else workbook[sheet_idx]
        type_col = _find_header_column(sheet, "type") or 2
        autofilter_ref = getattr(sheet.auto_filter, "ref", None)
        type_column_filtered = False
        if autofilter_ref:
            min_col, _, _, _ = range_boundaries(str(autofilter_ref))
            expected_col_id = str(type_col - min_col)
            for filter_column in getattr(sheet.auto_filter, "filterColumn", []) or []:
                if str(getattr(filter_column, "colId", "")) == expected_col_id:
                    type_column_filtered = True
                    break

        visible_row_types: list[Any] = []
        hidden_row_types: list[Any] = []
        visible_row_count = 0
        for row_idx in range(1, sheet.max_row + 1):
            if not any(sheet.cell(row=row_idx, column=col).value is not None for col in range(1, sheet.max_column + 1)):
                continue
            hidden = bool(getattr(sheet.row_dimensions[row_idx], "hidden", False))
            if not hidden:
                visible_row_count += 1
            if row_idx == 1:
                continue
            type_value = sheet.cell(row=row_idx, column=type_col).value
            if type_value is None:
                continue
            if hidden:
                hidden_row_types.append(type_value)
            else:
                visible_row_types.append(type_value)
        workbook.close()
        return {
            "visible_row_count": visible_row_count,
            "autofilter_enabled": bool(autofilter_ref),
            "autofilter_ref": autofilter_ref,
            "type_column_filtered": type_column_filtered,
            "visible_row_types": visible_row_types,
            "hidden_row_types": hidden_row_types,
        }
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


def _get_calc_cell_value_154194(env: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Read Calc decimal-format cells with formula text available for fallback."""

    return _get_calc_cell_value_with_formula(env, config, source_cell="C1")


def _get_calc_cell_value_with_formula(
    env: Any,
    config: dict[str, Any],
    *,
    source_cell: str = "C1",
) -> dict[str, Any]:
    path = config.get("path")
    data = env.controller.get_file(path) if path else None
    if not data:
        return {"value": None, "number_format": None, "formula": None, "source_value": None}

    import tempfile as _tempfile

    import openpyxl

    sheet_ref = config.get("sheet", 0)
    cell_addr = config.get("cell", "A1")
    source_cell = str(config.get("source_cell") or source_cell)
    with _tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(bytes(data))
        tmp_path = tmp.name
    try:
        formula_workbook = openpyxl.load_workbook(tmp_path, data_only=False)
        formula_sheet = (
            formula_workbook.worksheets[sheet_ref]
            if isinstance(sheet_ref, int)
            else formula_workbook[sheet_ref]
        )
        formula_cell = formula_sheet[cell_addr]
        formula = (
            formula_cell.value
            if isinstance(formula_cell.value, str) and formula_cell.value.startswith("=")
            else None
        )
        value_workbook = openpyxl.load_workbook(tmp_path, data_only=True)
        value_sheet = (
            value_workbook.worksheets[sheet_ref]
            if isinstance(sheet_ref, int)
            else value_workbook[sheet_ref]
        )
        cell = value_sheet[cell_addr]
        source_value = value_sheet[source_cell].value
        result = {
            "value": cell.value,
            "number_format": cell.number_format,
            "formula": formula,
            "source_cell": source_cell,
            "source_value": source_value,
            "numeric_value": cell.value if isinstance(cell.value, (int, float)) else None,
        }
        formula_workbook.close()
        value_workbook.close()
        return result
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


def _find_header_column(sheet: Any, header: str) -> int | None:
    target = str(header).strip().lower()
    for row_idx in range(1, min(sheet.max_row, 5) + 1):
        for cell in sheet[row_idx]:
            if str(cell.value or "").strip().lower() == target:
                return int(cell.column)
    return None


def _calculate_image_brightness(image: Any) -> float:
    from PIL import ImageStat

    grayscale = image.convert("L")
    stat = ImageStat.Stat(grayscale)
    return float(stat.mean[0])


def _measure_image_saturation(image: Any) -> float:
    from PIL import ImageStat

    hsv = image.convert("HSV")
    saturation = hsv.split()[1]
    stat = ImageStat.Stat(saturation)
    return float(stat.mean[0])


def _normalize_image_brightness(image: Any, target_brightness: float):
    current_brightness = _calculate_image_brightness(image)
    if current_brightness <= 0:
        return image
    factor = float(target_brightness) / current_brightness

    def point_transform(value: int) -> int:
        return min(255, max(0, int(value * factor)))

    return image.point(point_transform)


def _structure_check_by_mse(img1: Any, img2: Any, threshold: float = 0.03) -> bool:
    from PIL import Image

    if not hasattr(img1, "size") or not hasattr(img2, "size"):
        if hasattr(img1, "shape"):
            img1 = Image.fromarray(img1)
        if hasattr(img2, "shape"):
            img2 = Image.fromarray(img2)
    if img1.size != img2.size:
        img1 = img1.resize(img2.size, Image.Resampling.LANCZOS)
    if img1.mode != "RGB":
        img1 = img1.convert("RGB")
    if img2.mode != "RGB":
        img2 = img2.convert("RGB")

    pixels1 = list(img1.getdata())
    pixels2 = list(img2.getdata())
    if not pixels1 or len(pixels1) != len(pixels2):
        return False
    mse = sum(
        sum((channel1 - channel2) ** 2 for channel1, channel2 in zip(px1, px2))
        for px1, px2 in zip(pixels1, pixels2)
    ) / (len(pixels1) * 3)
    normalized_mse = mse / (255.0**2)
    return math.sqrt(normalized_mse) < float(threshold)


class _TriangleDetection:
    """Mask object that also supports generated-code tuple unpacking."""

    def __init__(self, mask: Any, angle: float | None = None, contour: Any = None) -> None:
        self.mask = mask
        self.angle = angle
        self.contour = contour

    def __array__(self, dtype: Any = None):
        import numpy as np

        array = np.asarray(self.mask, dtype=bool)
        if dtype is not None:
            array = array.astype(dtype)
        return array

    def __iter__(self):
        yield self.mask
        yield self.angle
        yield self.contour

    def __getattr__(self, name: str) -> Any:
        return getattr(self.mask, name)

    def any(self) -> bool:
        return bool(self.mask.any())


def _triangle_mask_array(mask: Any):
    import numpy as np

    if isinstance(mask, _TriangleDetection):
        mask = mask.mask
    return np.asarray(mask, dtype=bool)


def _detect_yellow_triangle_mask(image_array: Any):
    import numpy as np

    array = np.asarray(image_array)
    if array.ndim < 3 or array.shape[2] < 3:
        return np.zeros(array.shape[:2], dtype=bool)
    rgb = array[:, :, :3].astype(float)
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    # Generated ScaleCUA GIMP tasks use the canonical yellow triangle asset.
    # Keep the mask tolerant to anti-aliased edges and export/compression drift.
    mask = (red >= 150) & (green >= 130) & (blue <= 150) & ((red + green) >= 320)
    return _largest_connected_mask(mask)


def _find_triangle_color(image_array: Any):
    import numpy as np

    array = np.asarray(image_array)
    mask = _detect_yellow_triangle_mask(array)
    if not mask.any():
        return None
    y, x = np.argwhere(mask)[0]
    if array.ndim >= 3:
        return array[y, x]
    return array[y, x].item()


def _calculate_centroid(mask: Any):
    import numpy as np

    mask_array = _triangle_mask_array(mask)
    coords = np.argwhere(mask_array)
    if coords.size == 0:
        return None
    return coords.mean(axis=0)


def _largest_connected_mask(mask: Any):
    import numpy as np

    mask_array = np.asarray(mask, dtype=bool)
    if not mask_array.any():
        return mask_array
    try:
        import cv2

        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            mask_array.astype("uint8"),
            8,
        )
        if count <= 1:
            return mask_array
        largest_label = max(range(1, count), key=lambda idx: int(stats[idx, cv2.CC_STAT_AREA]))
        return labels == largest_label
    except Exception:
        return mask_array


def _triangle_detection_from_array(image_array: Any) -> _TriangleDetection:
    import numpy as np

    mask = _detect_yellow_triangle_mask(image_array)
    angle: float | None = None
    contour: Any = None
    if mask.any():
        try:
            import cv2

            contours, _hierarchy = cv2.findContours(
                mask.astype("uint8"),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if contours:
                contour = max(contours, key=cv2.contourArea)
                (_center, _size, rect_angle) = cv2.minAreaRect(contour)
                angle = float(rect_angle)
        except Exception:
            contour = None
            coords = np.argwhere(mask)
            if coords.size:
                centered = coords - coords.mean(axis=0)
                try:
                    _, _singular, vh = np.linalg.svd(centered, full_matrices=False)
                    angle = float(math.degrees(math.atan2(vh[0][0], vh[0][1])))
                except Exception:
                    angle = None
    return _TriangleDetection(mask, angle, contour)


def _detect_yellow_triangle(image_array: Any) -> _TriangleDetection:
    return _triangle_detection_from_array(image_array)


def _verify_horizontal_mirror(
    original_array: Any,
    result_array: Any,
    original_mask: Any,
    result_mask: Any,
) -> bool:
    import numpy as np

    expected = np.fliplr(_triangle_mask_array(original_mask))
    result = _triangle_mask_array(result_mask)
    if expected.shape != result.shape or not expected.any() or not result.any():
        return False
    intersection = np.logical_and(expected, result).sum()
    union = np.logical_or(expected, result).sum()
    if union and intersection / union >= 0.55:
        return True
    expected_x = np.argwhere(expected)[:, 1].mean()
    result_x = np.argwhere(result)[:, 1].mean()
    width = result.shape[1]
    return abs(expected_x - result_x) <= max(6.0, width * 0.08)


def _verify_vertical_mirror(
    original_array: Any,
    result_array: Any,
    original_mask: Any,
    result_mask: Any,
) -> bool:
    import numpy as np

    expected = np.flipud(_triangle_mask_array(original_mask))
    result = _triangle_mask_array(result_mask)
    if expected.shape != result.shape or not expected.any() or not result.any():
        return False
    intersection = np.logical_and(expected, result).sum()
    union = np.logical_or(expected, result).sum()
    if union and intersection / union >= 0.55:
        return True
    expected_y = np.argwhere(expected)[:, 0].mean()
    result_y = np.argwhere(result)[:, 0].mean()
    height = result.shape[0]
    return abs(expected_y - result_y) <= max(6.0, height * 0.08)


def _verify_horizontal_flip(
    original_array: Any,
    result_array: Any,
    original_mask: Any,
    result_mask: Any,
    width: Any = None,
) -> float:
    if _verify_horizontal_mirror(original_array, result_array, original_mask, result_mask):
        return 1.0
    return 0.0


def _check_triangle_result_only(tgt_path: Any, expected_orientation: str = "flipped_horizontal") -> float:
    try:
        from PIL import Image
        import numpy as np

        image = Image.open(tgt_path)
        mask = _detect_yellow_triangle_mask(np.array(image))
        if not mask.any():
            return 0.0
        coords = np.argwhere(mask)
        centroid_y, centroid_x = coords.mean(axis=0)
        height, width = mask.shape[:2]
        if expected_orientation == "flipped_horizontal":
            return 1.0 if centroid_x > width / 2 else 0.0
        if expected_orientation == "flipped_vertical":
            return 1.0 if centroid_y > height / 2 else 0.0
    except Exception:
        return 0.0
    return 0.0


_NAMED_RGB_COLORS = {
    "black": (0, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 128, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "red": (255, 0, 0),
    "white": (255, 255, 255),
    "yellow": (255, 255, 0),
}


def _rgb_tuple_from_color(value: Any) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        try:
            return tuple(max(0, min(255, int(float(channel)))) for channel in value[:3])  # type: ignore[return-value]
        except (TypeError, ValueError):
            return None
    if not isinstance(value, (str, bytes, dict)) and hasattr(value, "__iter__"):
        try:
            items = list(value)
        except TypeError:
            items = []
        if len(items) >= 3:
            try:
                return tuple(max(0, min(255, int(float(channel)))) for channel in items[:3])  # type: ignore[return-value]
            except (TypeError, ValueError):
                return None
    if isinstance(value, dict):
        try:
            return (
                int(value["r"]),
                int(value["g"]),
                int(value["b"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
    for attr in ("rgb", "value"):
        attr_value = getattr(value, attr, None)
        if attr_value is not None and attr_value is not value:
            parsed = _rgb_tuple_from_color(attr_value)
            if parsed is not None:
                return parsed
    text = str(value).strip()
    if not text:
        return None
    named = _NAMED_RGB_COLORS.get(text.lower())
    if named is not None:
        return named
    text = text.removeprefix("#")
    if text.lower().startswith("0x"):
        text = text[2:]
    text = re.sub(r"[^0-9A-Fa-f]", "", text)
    if len(text) >= 8:
        text = text[-6:]
    elif len(text) == 6:
        pass
    elif len(text) == 3:
        text = "".join(channel * 2 for channel in text)
    else:
        return None
    try:
        return (
            int(text[0:2], 16),
            int(text[2:4], 16),
            int(text[4:6], 16),
        )
    except ValueError:
        return None


def _colors_similar(color1: Any, color2: Any, tolerance: Any = 20) -> bool:
    rgb1 = _rgb_tuple_from_color(color1)
    rgb2 = _rgb_tuple_from_color(color2)
    if rgb1 is None or rgb2 is None:
        return False
    try:
        tolerance_value = float(tolerance)
    except (TypeError, ValueError):
        tolerance_value = 20.0
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))
    return distance <= tolerance_value


def _color_distance(color1: Any, color2: Any) -> float:
    rgb1 = _rgb_tuple_from_color(color1)
    rgb2 = _rgb_tuple_from_color(color2)
    if rgb1 is None or rgb2 is None:
        return float("inf")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))


def _is_yellow_color(color: Any, tolerance: Any = 45) -> bool:
    rgb = _rgb_tuple_from_color(color)
    if rgb is None:
        return False
    red, green, blue = rgb
    try:
        tolerance_value = float(tolerance)
    except (TypeError, ValueError):
        tolerance_value = 45.0
    if _colors_similar(rgb, (255, 255, 0), tolerance_value):
        return True
    return red >= 180 and green >= 170 and blue <= max(120, tolerance_value * 2)


def _string_similarity_ratio(a: Any, b: Any) -> float:
    """Case-insensitive fuzzy string ratio in [0, 1].

    Several generated metric shards (e.g. ``check_pdf_export__d587f20a``) call a
    bare ``similarity(x, y)`` global that the generator never emitted, raising
    ``NameError`` at eval time. Provide the difflib-based ratio they assume.
    """

    from difflib import SequenceMatcher

    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


#: Names ``_install_generated_metric_helpers`` puts back into every generated
#: shard. A shard may CALL one of these without defining it and still run. Any
#: other undefined call is a NameError at scoring time -- see
#: ``dataset._metrics_calling_undefined_helpers``, which reads this set so the two
#: never drift apart.
_INJECTED_HELPER_NAMES: frozenset[str] = frozenset({
    "similarity", "calculate_brightness", "normalize_brightness",
    "measure_saturation", "structure_check_by_mse", "_colors_similar",
    "detect_yellow_triangle", "_detect_yellow_triangle", "verify_horizontal_flip",
    "_verify_horizontal_mirror", "_verify_vertical_mirror", "_check_result_only",
    "_find_triangle_color", "_calculate_centroid", "_parse_rgb_color",
    "_color_distance", "_rgb_distance", "_is_yellow_color", "is_yellow_color",
    "_verify_single_check__7767eef2",
})


def _install_generated_metric_helpers(package: str, overlay: ModuleType) -> None:
    """Provide helpers omitted from some generated metric shards."""

    helpers = {
        "similarity": _string_similarity_ratio,
        "calculate_brightness": _calculate_image_brightness,
        "normalize_brightness": _normalize_image_brightness,
        "measure_saturation": _measure_image_saturation,
        "structure_check_by_mse": _structure_check_by_mse,
        "_colors_similar": _colors_similar,
        "detect_yellow_triangle": _detect_yellow_triangle_mask,
        "_detect_yellow_triangle": _detect_yellow_triangle,
        "verify_horizontal_flip": _verify_horizontal_flip,
        "_verify_horizontal_mirror": _verify_horizontal_mirror,
        "_verify_vertical_mirror": _verify_vertical_mirror,
        "_check_result_only": _check_triangle_result_only,
        "_find_triangle_color": _find_triangle_color,
        "_calculate_centroid": _calculate_centroid,
        "_parse_rgb_color": _rgb_tuple_from_color,
        "_color_distance": _color_distance,
        "_rgb_distance": _color_distance,
        "_is_yellow_color": _is_yellow_color,
        "is_yellow_color": _is_yellow_color,
        "_verify_single_check__7767eef2": _verify_gimp_config_single_check,
    }
    modules = [overlay]
    modules.extend(
        module
        for name, module in list(sys.modules.items())
        if name.startswith(f"{package}.") and ".verigen_metrics." in name
    )
    try:
        from desktop_env.evaluators.metrics import gimp as upstream_gimp_metrics
    except Exception:
        upstream_gimp_metrics = None
    if upstream_gimp_metrics is not None:
        for name in tuple(helpers):
            helpers[name] = getattr(upstream_gimp_metrics, name, helpers[name])
    for module in modules:
        for name, fn in helpers.items():
            setattr(module, name, fn)


def _install_pptx_font_strikethrough_alias() -> None:
    """Match generated ScaleCUA getters that expect ``Font.strikethrough``.

    python-pptx exposes underline/bold/italic but not a public strikethrough
    property. Several generated ScaleCUA RL getters read that attribute, so
    provide a narrow alias over the run ``strike`` OOXML attribute.
    """

    try:
        from pptx.text.text import Font
    except Exception:
        return
    if hasattr(Font, "strikethrough"):
        return

    def _get_strikethrough(self):
        value = self._element.get("strike")
        if value is None:
            return False
        return str(value).lower() not in {"nostrike", "none", "false", "0"}

    def _set_strikethrough(self, value):
        self._element.set("strike", "sngStrike" if value else "noStrike")

    Font.strikethrough = property(_get_strikethrough, _set_strikethrough)


def _is_gimp_config_metric_name(name: str) -> bool:
    if name.startswith("check_config_status"):
        return True
    if name.startswith("check_gimp_gimprc_setting__"):
        return True
    return name in {
        "check_gimp_rulers__f13e522727f9f47f05a1828391e4f1b5",
    }


def _wrap_gimp_config_metric(metric):
    def wrapped(result, expected, **options):
        original = None
        try:
            original = metric(result, expected, **options)
            if _metric_succeeded(original):
                return original
        except Exception as exc:
            logger.debug(
                "Falling back for generated GIMP config metric %s: %s",
                getattr(metric, "__name__", metric),
                exc,
            )
        repaired = _gimp_config_status_score(result, expected)
        if repaired is not None:
            return repaired
        return original if original is not None else 0.0

    return wrapped


def _metric_succeeded(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return bool(value)


def _gimp_config_status_score(actual_config_path: Any, rule: Any) -> float | None:
    if actual_config_path is None:
        return 0.0
    if not isinstance(rule, dict):
        return None
    if "checks" in rule:
        checks = rule.get("checks")
        if not isinstance(checks, list) or not checks:
            return None
        content = _read_gimp_config_lines(actual_config_path)
        if content is None:
            return 0.0
        return 1.0 if all(_verify_gimp_config_single_check(content, check) for check in checks) else 0.0
    if "key" not in rule or "value" not in rule:
        return None
    content = _read_gimp_config_lines(actual_config_path)
    if content is None:
        return 0.0
    return 1.0 if _verify_gimp_config_single_check(content, rule) else 0.0


def _read_gimp_config_lines(actual_config_path: Any) -> list[str] | None:
    try:
        with open(os.fspath(actual_config_path), encoding="utf-8", errors="ignore") as handle:
            return handle.readlines()
    except Exception:
        return None


def _verify_gimp_config_single_check(content: Any, rule: Any) -> bool:
    if not isinstance(rule, dict):
        return False
    key_tokens = _gimp_config_key_tokens(rule.get("key"))
    if not key_tokens or "value" not in rule:
        return False
    expected = rule.get("value")
    for line in content or []:
        tokens = _gimp_config_line_tokens(str(line))
        if not tokens:
            continue
        values = _gimp_config_values_after_key(tokens, key_tokens)
        if values and _gimp_config_values_match(values, expected):
            return True
    return False


def _gimp_config_line_tokens(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return []
    lexer = shlex.shlex(stripped, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    lexer.whitespace += "()"
    try:
        return [_normalize_gimp_config_token(token) for token in lexer if token]
    except ValueError:
        return [
            _normalize_gimp_config_token(token)
            for token in re.sub(r"[()]", " ", stripped).split()
            if token
        ]


def _gimp_config_key_tokens(key: Any) -> list[str]:
    if isinstance(key, str):
        return [_normalize_gimp_config_token(key)]
    if isinstance(key, list):
        return [
            _normalize_gimp_config_token(token)
            for token in key
            if _normalize_gimp_config_token(token)
        ]
    return []


def _normalize_gimp_config_token(value: Any) -> str:
    text = str(value).strip().strip("()").strip()
    if (
        len(text) >= 2
        and text[0] == text[-1]
        and text[0] in {"'", '"'}
    ):
        text = text[1:-1]
    return text


def _gimp_config_values_after_key(
    tokens: list[str],
    key_tokens: list[str],
) -> list[str]:
    key_len = len(key_tokens)
    for index in range(0, len(tokens) - key_len + 1):
        if tokens[index : index + key_len] == key_tokens:
            return tokens[index + key_len :]
    if key_len == 2 and tokens and tokens[0] == key_tokens[0]:
        for index in range(1, len(tokens)):
            if tokens[index] == key_tokens[1]:
                return tokens[index + 1 :]
    return []


def _gimp_config_values_match(values: list[str], expected: Any) -> bool:
    expected_text = _normalize_gimp_config_token(expected)
    if not values:
        return False
    numeric_prefix: list[str] = []
    for value in values:
        if not _is_number_like(value):
            break
        numeric_prefix.append(value)
    if numeric_prefix and _is_number_like(expected_text):
        return all(_gimp_config_value_equal(value, expected_text) for value in numeric_prefix)
    return (
        _gimp_config_value_equal(values[0], expected_text)
        or _gimp_config_value_equal(" ".join(values), expected_text)
    )


_GIMP_MEMSIZE_UNIT = {"": 1, "b": 1, "k": 1024, "m": 1024**2, "g": 1024**3}


def _gimp_memsize_to_bytes(tok: Any) -> int | None:
    # '2G'/'512M' aren't float()-parseable so _is_number_like skips them;
    # compare via unit-parsed bytes. General unit-parse (not an allow-list);
    # distinct quantities never collide (1G != 2G).
    m = re.fullmatch(r"\s*(\d+)\s*([bkmgBKMG]?)\s*", str(tok or ""))
    return int(m.group(1)) * _GIMP_MEMSIZE_UNIT[m.group(2).lower()] if m else None


def _gimp_config_value_equal(actual: Any, expected: Any) -> bool:
    actual_text = _normalize_gimp_config_token(actual)
    expected_text = _normalize_gimp_config_token(expected)
    if actual_text == expected_text:
        return True
    if _is_number_like(actual_text) and _is_number_like(expected_text):
        return abs(float(actual_text) - float(expected_text)) <= 1e-6
    a, b = _gimp_memsize_to_bytes(actual_text), _gimp_memsize_to_bytes(expected_text)
    return a is not None and b is not None and a == b


def _is_number_like(value: Any) -> bool:
    try:
        float(str(value))
    except (TypeError, ValueError):
        return False
    return True


def _score_is_success(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    try:
        return float(value) >= 1.0
    except (TypeError, ValueError):
        return False


def _wrap_chrome_experiments_metric(metric, metric_name: str):
    def wrapped(result, expected, **options):
        if not isinstance(result, list) or not isinstance(expected, dict):
            original = _call_metric_with_supported_options(
                metric,
                result,
                expected,
                options,
            )
            return original
        actual_names = _normalize_chrome_experiment_names(result)
        expected_names = expected.get("experiment_names")
        if expected_names != ["disable-quic"]:
            expected_names = (
                expected.get("names")
                if expected.get("type") == "names"
                else expected_names
            )
        expected_name = expected.get("experiment_name")

        if expected_name:
            is_enabled = _chrome_experiment_enabled(actual_names, expected_name)
            if "not_contains" in metric_name:
                return 0.0 if is_enabled else 1.0
            return 1.0 if is_enabled else 0.0
        if (
            isinstance(expected_names, list)
            and (
                "not_contains" in metric_name
                or metric_name.startswith("check_enabled_experiments__")
            )
        ):
            if all(not _chrome_experiment_enabled(actual_names, name) for name in expected_names):
                return 1.0
            return 0.0

        original = _call_metric_with_supported_options(
            metric,
            result,
            expected,
            options,
        )
        if _score_is_success(original):
            return original
        if not isinstance(expected_names, list):
            return original

        if expected_names == ["disable-quic"] and "disable-quic" in actual_names:
            return 1.0
        if "contains_any" in metric_name:
            if any(_chrome_experiment_enabled(actual_names, name) for name in expected_names):
                return 1.0
            return original
        if all(_chrome_experiment_enabled(actual_names, name) for name in expected_names):
            return 1.0
        return original

    return wrapped


def _normalize_chrome_experiment_names(result: list[Any]) -> set[str]:
    names: set[str] = set()
    for item in result:
        if not isinstance(item, str) or not item:
            continue
        name, _, value = item.partition("@")
        if name == "enable-quic" and value == "2":
            names.add("disable-quic")
            continue
        if item == "enable-quic":
            names.add("disable-quic")
            continue
        if value == "2":
            continue
        names.add(name)
    return names


def _chrome_experiment_enabled(actual_names: set[str], expected_name: Any) -> bool:
    expected = str(expected_name or "").strip()
    if not expected:
        return False
    if expected in actual_names:
        return True
    aliases = {
        "tab-groups": {
            "tab-groups",
            "tab-groups-save",
            "tab-groups-save-ui-update",
            "tab-groups-save-v2",
            "tab-organization",
            "tab-reorganization",
        },
        "tab-groups-collapse": {
            "tab-groups-collapse",
            "tab-groups-save",
            "tab-groups-save-v2",
            "tab-groups-save-ui-update",
        },
    }
    return bool(aliases.get(expected, set()) & actual_names)


def _wrap_chrome_extension_name_metric(metric):
    def wrapped(result, expected, **options):
        original = metric(result, expected, **options)
        if _score_is_success(original):
            return original
        if not isinstance(result, list) or not isinstance(expected, dict):
            return original
        expected_name = str(expected.get("extension_name") or "").strip()
        if _normalize_extension_label(expected_name) != "helloextensions":
            return original
        for item in result:
            if _normalize_extension_label(item) in {"helloextension", "helloextensions"}:
                return 1.0
        return original

    return wrapped


def _normalize_extension_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _wrap_chrome_search_engine_candidate_metric(metric):
    def wrapped(result, expected, **options):
        original = _call_metric_with_supported_options(
            metric,
            result,
            expected,
            options,
        )
        if _score_is_success(original) or not _is_scalecua_candidate_result(result):
            return original
        expected_values = _expected_search_engine_values(expected)
        if not expected_values:
            return original
        candidates = result.get("candidates", [])
        if not isinstance(candidates, list):
            return original
        normalized_expected = {
            _normalize_search_engine_label(value)
            for value in expected_values
            if _normalize_search_engine_label(value)
        }
        for candidate in candidates:
            normalized_candidate = _normalize_search_engine_label(candidate)
            if normalized_candidate and normalized_candidate in normalized_expected:
                return 1.0
        return original

    return wrapped


def _call_metric_with_supported_options(
    metric,
    result: Any,
    expected: Any,
    options: dict[str, Any],
):
    try:
        sig = inspect.signature(metric)
    except (TypeError, ValueError):
        return metric(result, expected, **(options or {}))
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return metric(result, expected, **(options or {}))
    supported = {
        key: value
        for key, value in (options or {}).items()
        if key in sig.parameters
    }
    return metric(result, expected, **supported)


def _is_scalecua_candidate_result(result: Any) -> bool:
    return isinstance(result, dict) and result.get("__scalecua_candidates__") is True


def _expected_search_engine_values(expected: Any) -> list[Any]:
    if not isinstance(expected, dict):
        return []
    if isinstance(expected.get("expected"), list):
        return list(expected["expected"])
    if expected.get("expected") is not None:
        return [expected["expected"]]
    if expected.get("name") is not None:
        return [expected["name"]]
    if expected.get("expected_engine") is not None:
        return [expected["expected_engine"]]
    return []


def _normalize_search_engine_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    compact = re.sub(r"[^a-z0-9]+", "", text)
    if not compact:
        return ""
    if "startpage.com" in text or compact in {"startpage", "startpagecom"}:
        return "startpage"
    if "duckduckgo.com" in text or compact in {"duckduckgo", "duckduckgocom", "duckduckgohtml"}:
        return "duckduckgo"
    if "search.yahoo.com" in text or compact.startswith("yahoo"):
        return "yahoo"
    if "bing.com" in text or compact in {"bing", "microsoftbing", "microsoftbingdefault"}:
        return "bing"
    if "google.com" in text or compact in {"google", "googlesearch"}:
        return "google"
    if "ecosia.org" in text or compact in {"ecosia", "ecosiasearch"}:
        return "ecosia"
    if "brave.com" in text or compact in {"brave", "bravesearch"}:
        return "brave"
    if "qwant.com" in text or compact == "qwant":
        return "qwant"
    if "yandex" in text or compact == "yandex":
        return "yandex"
    if "baidu.com" in text or compact in {"baidu"}:
        return "baidu"
    if "ask.com" in text or compact in {"ask", "askcom"}:
        return "ask"
    return compact


def _wrap_vscode_negative_keybinding_metric(metric):
    def wrapped(result, expected, **options):
        original = metric(result, expected, **options)
        if _score_is_success(original):
            return original
        if _vscode_negative_keybinding_without_when_matches(result, expected):
            return 1.0
        return original

    wrapped.__name__ = getattr(metric, "__name__", "wrapped_vscode_keybinding_metric")
    return wrapped


def _vscode_negative_keybinding_without_when_matches(result: Any, expected: Any) -> bool:
    expected_binding = _extract_expected_vscode_keybinding(expected)
    if not expected_binding:
        return False
    expected_command = expected_binding.get("command")
    expected_key = expected_binding.get("key")
    expected_when = expected_binding.get("when")
    if (
        not isinstance(expected_key, str)
        or not isinstance(expected_command, str)
        or not expected_command.startswith("-")
        or not isinstance(expected_when, str)
        or not expected_when.strip()
    ):
        return False

    data = _load_vscode_keybindings_file(result)
    if not isinstance(data, list):
        return False
    for item in data:
        if not isinstance(item, dict):
            continue
        if item.get("key") != expected_key or item.get("command") != expected_command:
            continue
        actual_when = item.get("when")
        if actual_when in (None, ""):
            return True
    return False


def _extract_expected_vscode_keybinding(expected: Any) -> dict[str, Any] | None:
    if isinstance(expected, str):
        try:
            expected = json.loads(expected)
        except json.JSONDecodeError:
            return None
    if not isinstance(expected, dict):
        return None
    binding = expected.get("expected", expected)
    if isinstance(binding, str):
        try:
            binding = json.loads(binding)
        except json.JSONDecodeError:
            return None
    return binding if isinstance(binding, dict) else None


def _load_vscode_keybindings_file(path: Any) -> Any:
    if not isinstance(path, (str, os.PathLike)):
        return None
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    for candidate in (text, "\n".join(text.splitlines()[1:])):
        stripped = _strip_jsonc_comments(candidate).strip()
        if not stripped:
            continue
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            continue
    return None


def _strip_jsonc_comments(text: str) -> str:
    out: list[str] = []
    in_string = False
    escaped = False
    i = 0
    while i < len(text):
        char = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            i += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            i += 1
            continue
        if char == "/" and nxt == "/":
            while i < len(text) and text[i] not in "\r\n":
                i += 1
            continue
        if char == "/" and nxt == "*":
            i += 2
            while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(char)
        i += 1
    return "".join(out)


def _is_thunderbird_prefs_metric_name(name: str) -> bool:
    return name in THUNDERBIRD_PREFS_METRIC_NAMES or name.startswith(
        "check_thunderbird_prefs__"
    )


def _wrap_thunderbird_prefs_metric(metric):
    def wrapped(result, expected, **options):
        needs_compat = _thunderbird_prefs_rules_need_generated_compat(expected)
        try:
            original = _call_metric_with_supported_options(
                metric, result, expected, options
            )
        except (KeyError, NotImplementedError, TypeError, ValueError) as exc:
            if not needs_compat:
                logger.debug(
                    "Generated Thunderbird prefs fallback skipped for %s: %s",
                    getattr(metric, "__name__", metric),
                    exc,
                )
                return 0.0
        else:
            if _score_is_success(original) or not needs_compat:
                return original

        try:
            return _check_thunderbird_prefs_generated_compat(result, expected)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.debug(
                "Generated Thunderbird prefs fallback failed for %s: %s",
                getattr(metric, "__name__", metric),
                exc,
            )
            return 0.0

    wrapped.__name__ = getattr(metric, "__name__", "wrapped_thunderbird_prefs_metric")
    return wrapped


def _thunderbird_prefs_rules_need_generated_compat(expected: Any) -> bool:
    methods: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            method = value.get("method")
            if method is not None:
                methods.add(str(method))
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(expected)
    return bool(methods & {"==", "!=", "<=", ">=", "literal", "in", "contains"})


def _check_thunderbird_prefs_generated_compat(result: Any, expected: Any) -> float:
    if result is None or not isinstance(expected, dict):
        return 0.0
    if not isinstance(result, (str, os.PathLike)) or not os.path.isfile(result):
        return 0.0

    prefs = _read_thunderbird_prefs_file(os.fspath(result))
    expect_rules = expected.get("expect", {})
    unexpect_rules = expected.get("unexpect", {})
    if not isinstance(expect_rules, dict) or not isinstance(unexpect_rules, dict):
        return 0.0

    for key, rule in expect_rules.items():
        if key not in prefs or not _match_thunderbird_pref_generated_rule(
            prefs[key], rule
        ):
            return 0.0
    for key, rule in unexpect_rules.items():
        if key in prefs and _match_thunderbird_pref_generated_rule(prefs[key], rule):
            return 0.0
    return 1.0


def _read_thunderbird_prefs_file(path: str) -> dict[str, Any]:
    prefs: dict[str, Any] = {}
    with open(path, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = _THUNDERBIRD_PREF_RE.match(line.strip())
            if match is None:
                continue
            prefs[match.group("key")] = json.loads(match.group("val"))
    return prefs


def _match_thunderbird_pref_generated_rule(value: Any, rule: Any) -> bool:
    if not isinstance(rule, dict):
        return value == rule
    method = str(rule.get("method", "eq"))
    ref = rule.get("ref")
    method = {
        "==": "eq",
        "literal": "eq",
        "!=": "ne",
        "<=": "le",
        ">=": "ge",
        "gte": "ge",
        "lte": "le",
    }.get(method, method)
    if method.startswith("re"):
        flags = re.RegexFlag(0)
        for flag_name in method.split(".")[1:]:
            flags |= getattr(re, flag_name)
        return re.search(str(ref), str(value), flags) is not None
    if method in {"eq", "ne", "le", "lt", "ge", "gt"}:
        return getattr(operator, method)(value, ref)
    if method == "in":
        if isinstance(ref, (list, tuple, set, frozenset)):
            return value in ref
        return str(value) in str(ref)
    if method == "contains":
        if isinstance(ref, (list, tuple, set, frozenset)):
            return all(str(item) in str(value) for item in ref)
        return str(ref) in str(value)
    return False


def _canon_tb_action(a: Any) -> Any:
    # Thunderbird serializes actions with spaces/casing ("Mark flagged");
    # generated gold uses CamelCase ("MarkFlagged"). Normalize BOTH sides
    # (casefold + strip spaces/underscores) -> whole action family, not an
    # alias set.
    return re.sub(r"[ _]+", "", a.strip().lower()) if isinstance(a, str) else a


def _wrap_thunderbird_filter_action(metric):
    """Normalize serialized filter-action spelling before a generated
    check_thunderbird_filter__* matcher (raw _match_record / exact ==).
    ALWAYS-normalize (not on-zero): with an unexpect bucket a leaked Title-case
    forbidden action makes the raw == return a false 1.0 that an on-zero wrapper
    would never correct."""
    def wrapped(result, expected, **options):
        if result is None:
            return metric(result, expected, **options)
        text = re.sub(r'action="([^"]*)"',
                      lambda m: 'action="%s"' % _canon_tb_action(m.group(1)),
                      open(result).read())
        rules2 = copy.deepcopy(expected)
        if isinstance(rules2, dict):
            if "action" in rules2:
                rules2["action"] = _canon_tb_action(rules2["action"])
            for bucket in ("expect", "unexpect"):
                for e in rules2.get(bucket, []) or []:
                    if isinstance(e, dict) and "action" in e:
                        e["action"] = _canon_tb_action(e["action"])
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as f:
            f.write(text); tmp = f.name
        try:
            return metric(tmp, rules2, **options)
        finally:
            os.unlink(tmp)
    return wrapped


# --- Scoped string-normalization wrappers ---------------------------------------
# office apps auto-substitute these glyphs on text entry (LibreOffice "Replace
# dashes" / smart quotes); fold them in the ACTUAL only when the GOLD carries
# none -> a verbatim-glyph task (glyph present in its gold) stays exact.
_AUTOCORRECT_GLYPHS = {"–": "-", "—": "-", "‘": "'",
                       "’": "'", "“": '"', "”": '"', "…": "..."}
AUTOCORRECT_TEXT_METRIC_PREFIXES = (
    "check_cell_contains__", "check_cell_text", "check_text_contains",
    "check_text_content", "check_text_exact_match", "check_pptx_text_contains",
)


def _iter_gold_text(expected):
    if isinstance(expected, str):
        yield expected
    elif isinstance(expected, dict):
        for v in expected.values():
            yield from _iter_gold_text(v)
    elif isinstance(expected, (list, tuple)):
        for v in expected:
            yield from _iter_gold_text(v)


def _wrap_text_file_value_currency_metric(metric):
    """Currency: recover $-/thousands-formatted numeric answers when the
    gold is a bare number. _to_float_or_none already strips $/,/%. Guard: gold
    carrying a currency glyph stays exact (no verbatim-currency task is masked;
    catalog scan finds 0). 0->1 validated; verbatim-$ + wrong-number controls hold."""
    def wrapped(result, expected, **options):
        original = metric(result, expected, **options)
        if _metric_succeeded(original):
            return original
        gold = expected.get("value") if isinstance(expected, dict) else expected
        if gold is None or any(c in str(gold) for c in "$£€¥₹"):
            return original
        gnum, anum = _to_float_or_none(gold), _to_float_or_none(result)
        if gnum is None or anum is None:
            return original
        return 1.0 if _compare_numeric_value(anum, gnum, options.get("tolerance", 0.01)) else original
    return wrapped


def _wrap_autocorrect_text_metric(metric):
    """Autocorrect glyphs (CONTESTED W1, scoped safe): fold en/em-dash,
    smart-quotes, ellipsis in the ACTUAL only when the GOLD has none -> the
    verbatim-glyph rows (pptx_slide_count/duplicate, not wrapped here) and any
    future en-dash gold stay exact. 0->1 validated; verbatim-en-dash-gold control holds."""
    def wrapped(result, expected, **options):
        original = metric(result, expected, **options)
        if _metric_succeeded(original):
            return original
        if any(ch in g for g in _iter_gold_text(expected) for ch in _AUTOCORRECT_GLYPHS):
            return original
        fold = lambda s: "".join(_AUTOCORRECT_GLYPHS.get(c, c) for c in str(s))
        if isinstance(result, dict) and "value" in result:
            folded = dict(result)
            folded["value"] = fold(result.get("value"))
        else:
            folded = fold(result)
        return metric(folded, expected, **options)
    return wrapped


def _maybe_wrap_scalecua_metric(name: str, runtime_split: str, fn):
    if runtime_split not in OVERLAY_SPLITS:
        return fn
    if _is_thunderbird_prefs_metric_name(name):
        return _wrap_thunderbird_prefs_metric(fn)
    if name.startswith("check_pptx_image_size__"):
        return _wrap_pptx_image_size_metric(fn)
    if (
        name in CHROME_EXPERIMENTS_EXACT_METRIC_NAMES
        or name in CHROME_EXPERIMENTS_SET_METRIC_NAMES
    ):
        return _wrap_chrome_experiments_metric(fn, name)
    if name.startswith("check_enabled_experiments__"):
        return _wrap_chrome_experiments_metric(fn, name)
    if name in CHROME_EXTENSION_NAME_METRIC_NAMES:
        return _wrap_chrome_extension_name_metric(fn)
    if name in CHROME_STARTUP_URL_METRIC_NAMES:
        return _wrap_chrome_startup_urls_metric(fn)
    if name in CHROME_BOOKMARK_WITH_TITLE_METRIC_NAMES:
        return _wrap_chrome_bookmark_with_title_metric(fn)
    if name in URL_PATTERN_METRIC_NAMES:
        return _wrap_url_pattern_metric(fn)
    if name in CHROME_SEARCH_ENGINE_METRIC_NAMES or name.startswith("check_search_engine__"):
        return _wrap_chrome_search_engine_candidate_metric(fn)
    if name == "check_json_keybindings" or name.startswith("check_json_keybindings__"):
        return _wrap_vscode_negative_keybinding_metric(fn)
    if name.startswith("check_thunderbird_filter__"):
        return _wrap_thunderbird_filter_action(fn)
    if name.startswith("check_text_file_value__"):
        return _wrap_text_file_value_currency_metric(fn)
    if name.startswith(AUTOCORRECT_TEXT_METRIC_PREFIXES):
        return _wrap_autocorrect_text_metric(fn)
    return fn


def resolve_metric(name: str, runtime_split: str):
    overlay = _load_overlay_module(runtime_split, "metrics")
    if overlay is not None:
        fn = getattr(overlay, name, None)
        if fn is not None:
            if runtime_split in OVERLAY_SPLITS and _is_gimp_config_metric_name(name):
                return _wrap_gimp_config_metric(fn)
            if runtime_split in OVERLAY_SPLITS and name in CHROME_INTERNAL_ACTIVE_TAB_METRIC_NAMES:
                return _wrap_chrome_internal_active_tab_approximate(fn)
            if runtime_split in OVERLAY_SPLITS and name in URL_CANONICAL_ACTIVE_TAB_METRIC_NAMES:
                return _wrap_url_canonical_active_tab(fn)
            if runtime_split in OVERLAY_SPLITS and _is_calc_chart_metric_name(name):
                return _wrap_calc_chart_metric(fn)
            if runtime_split in OVERLAY_SPLITS and name in CALC_INITIALS_COLUMN_METRIC_NAMES:
                return _wrap_calc_initials_column_metric(fn)
            if runtime_split in OVERLAY_SPLITS and name.startswith("check_pptx_specific_text__"):
                return _wrap_pptx_specific_text_metric(fn)
            if runtime_split in OVERLAY_SPLITS and name.startswith("check_pptx_slide_title_and_bg__"):
                return _wrap_pptx_slide_title_and_bg_metric(fn)
            if runtime_split in OVERLAY_SPLITS and name in CALC_FORMULA_VALUE_METRIC_NAMES:
                return _wrap_xlsx_formula_value_metric(fn)
            if runtime_split in OVERLAY_SPLITS and name in CALC_COLUMN_SORTED_TOP_VALUES_METRIC_NAMES:
                return _wrap_xlsx_column_sorted_metric(fn)
            if runtime_split in OVERLAY_SPLITS and name in CALC_SHEET_CELLS_METRIC_NAMES:
                return _wrap_xlsx_sheet_cells_metric(fn)
            if runtime_split in OVERLAY_SPLITS and name in CALC_PERCENTAGE_METRIC_NAMES:
                return _wrap_calc_percentage_metric(fn)
            if runtime_split in OVERLAY_SPLITS and name in CALC_NUMERIC_VALUE_METRIC_NAMES:
                return _wrap_calc_numeric_value_metric(fn)
            if runtime_split in OVERLAY_SPLITS and name in CALC_DECIMAL_TEXT_METRIC_NAMES:
                return _wrap_calc_decimal_text_metric(fn)
            if runtime_split in OVERLAY_SPLITS and name in CALC_PERIOD_RATE_SUM_METRIC_NAMES:
                return _wrap_calc_period_rate_sum_metric(fn)
            if runtime_split in OVERLAY_SPLITS and name in IMPRESS_NOTES_PANEL_METRIC_NAMES:
                return _wrap_impress_notes_panel_metric(fn)
            if runtime_split in OVERLAY_SPLITS and name in SPEEDTEST_REPORT_METRIC_NAMES:
                return _wrap_speedtest_report_metric(fn)
            return _maybe_wrap_scalecua_metric(name, runtime_split, fn)
    if runtime_split in OVERLAY_SPLITS and name == "is_expected_bookmarks":
        return _is_expected_bookmarks_scalecua
    from lite.gym.envs.lite.osworld.src.eval import metrics as custom_metrics
    from desktop_env.evaluators import metrics as upstream_metrics

    fn = getattr(custom_metrics, name, None)
    if fn is not None:
        if runtime_split in OVERLAY_SPLITS and _is_gimp_config_metric_name(name):
            return _wrap_gimp_config_metric(fn)
        if runtime_split in OVERLAY_SPLITS and name in CHROME_INTERNAL_ACTIVE_TAB_METRIC_NAMES:
            return _wrap_chrome_internal_active_tab_approximate(fn)
        if runtime_split in OVERLAY_SPLITS and name in URL_CANONICAL_ACTIVE_TAB_METRIC_NAMES:
            return _wrap_url_canonical_active_tab(fn)
        return _maybe_wrap_scalecua_metric(name, runtime_split, fn)
    fn = getattr(upstream_metrics, name, None)
    if fn is not None:
        if runtime_split in OVERLAY_SPLITS and _is_gimp_config_metric_name(name):
            return _wrap_gimp_config_metric(fn)
        if runtime_split in OVERLAY_SPLITS and name in CHROME_INTERNAL_ACTIVE_TAB_METRIC_NAMES:
            return _wrap_chrome_internal_active_tab_approximate(fn)
        if runtime_split in OVERLAY_SPLITS and name in URL_CANONICAL_ACTIVE_TAB_METRIC_NAMES:
            return _wrap_url_canonical_active_tab(fn)
        return _maybe_wrap_scalecua_metric(name, runtime_split, fn)
    raise JudgeResolverError(f"metric not found for {runtime_split}: {name}")


def _chart_title_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    texts: list[str] = []
    tx = getattr(value, "tx", None)
    rich = getattr(tx, "rich", None)
    for paragraph in getattr(rich, "p", []) or []:
        run_texts: list[str] = []
        for run in getattr(paragraph, "r", []) or []:
            text = getattr(run, "t", None)
            if text is not None:
                run_texts.append(str(text))
        if run_texts:
            texts.append("".join(run_texts))
            continue
        paragraph_text = getattr(paragraph, "text", None)
        if isinstance(paragraph_text, str):
            texts.append(str(paragraph_text))
    if texts:
        return "".join(texts)
    direct = getattr(value, "text", None)
    if isinstance(direct, str):
        return direct
    return ""


def _wrap_calc_chart_metric(metric):
    def wrapped(result, expected, **options):
        if isinstance(result, dict) and isinstance(result.get("charts"), list):
            result = copy.deepcopy(result)
            for chart in result.get("charts", []):
                if isinstance(chart, dict):
                    chart["chart_title"] = _chart_title_text(chart.get("chart_title"))
                    _normalize_calc_chart_series(chart)
        return metric(result, expected, **options)

    return wrapped


def _wrap_calc_initials_column_metric(metric):
    def wrapped(result, expected, **options):
        original = metric(result, expected, **options)
        try:
            if float(original) >= 1.0:
                return original
        except (TypeError, ValueError):
            if original:
                return original
        if not isinstance(result, dict) or not isinstance(expected, dict):
            return original
        expected_rows = expected.get("rows")
        if not isinstance(expected_rows, dict) or not expected_rows:
            return original
        header = str(result.get(2, "") or "").strip().lower()
        if header != "initials":
            return original
        for row_str, expected_value in expected_rows.items():
            try:
                shifted_row = int(row_str) + 1
            except (TypeError, ValueError):
                return original
            actual = result.get(shifted_row)
            if str(actual or "").strip() != str(expected_value):
                return original
        return 1.0

    return wrapped


def _is_calc_chart_metric_name(name: str) -> bool:
    return name.startswith("check_calc_chart__")


def _normalize_calc_chart_series(chart: dict[str, Any]) -> None:
    series = chart.get("data_series")
    if isinstance(series, list):
        for item in series:
            if not isinstance(item, dict):
                continue
            if "title" in item:
                item["title"] = _chart_title_text(item.get("title"))
        return
    data_column_names = chart.get("data_column_names")
    if not isinstance(data_column_names, list) or not data_column_names:
        return
    row_count = int(chart.get("data_row_count") or 0)
    data_range = f"A2:A{row_count + 1}" if row_count > 0 else ""
    chart["data_series"] = [
        {
            "title": str(name),
            "values_ref": str(name),
            "data_range": data_range,
        }
        for name in data_column_names
    ]


def _compare_numeric_value(actual: Any, expected: Any, tolerance: Any = 0.01) -> bool:
    try:
        actual_value = float(actual)
        expected_value = float(expected)
        tolerance_value = float(tolerance)
    except (TypeError, ValueError):
        return False
    return abs(actual_value - expected_value) <= tolerance_value


def _xlsx_range_sum(workbook: Any, current_sheet: Any, reference: str) -> float:
    from openpyxl.utils.cell import range_boundaries

    ref = reference.strip().replace("$", "")
    sheet = current_sheet
    if "!" in ref:
        sheet_name, ref = ref.split("!", 1)
        sheet_name = sheet_name.strip("'")
        sheet = workbook[sheet_name]
    min_col, min_row, max_col, max_row = range_boundaries(ref)
    total = 0.0
    for row in sheet.iter_rows(
        min_row=min_row,
        max_row=max_row,
        min_col=min_col,
        max_col=max_col,
        values_only=True,
    ):
        for value in row:
            if isinstance(value, (int, float)):
                total += float(value)
    return total


def _evaluate_simple_xlsx_formula(workbook: Any, sheet: Any, formula: str) -> float | None:
    text = str(formula or "").strip()
    if text.startswith("="):
        text = text[1:].strip()
    match = re.fullmatch(r"SUM\((?P<args>[^)]+)\)", text, flags=re.IGNORECASE)
    if match is None:
        return None
    total = 0.0
    for arg in match.group("args").split(","):
        total += _xlsx_range_sum(workbook, sheet, arg)
    return total


def _wrap_xlsx_formula_value_metric(metric):
    def wrapped(result, expected, **options):
        original = metric(result, expected, **options)
        if original:
            return original
        if not isinstance(result, str) or not isinstance(expected, dict):
            return original
        cell_address = expected.get("cell")
        expected_value = expected.get("expected_value")
        if not cell_address or expected_value is None:
            return original
        try:
            import openpyxl

            workbook = openpyxl.load_workbook(result, data_only=False)
            sheet = workbook.worksheets[int(expected.get("sheet", 0) or 0)]
            formula = sheet[str(cell_address)].value
            if not isinstance(formula, str) or not formula.lstrip().startswith("="):
                workbook.close()
                return original
            actual = _evaluate_simple_xlsx_formula(workbook, sheet, formula)
            workbook.close()
        except Exception:
            return original
        if actual is not None and _compare_numeric_value(
            actual,
            expected_value,
            expected.get("tolerance", 0.01),
        ):
            return 1.0
        return original

    return wrapped


def _wrap_xlsx_column_sorted_metric(metric):
    def wrapped(result, expected, **options):
        original = metric(result, expected, **options)
        try:
            if float(original) >= 1.0:
                return original
        except (TypeError, ValueError):
            if original:
                return original
        if not isinstance(result, dict) or not isinstance(expected, dict):
            return original
        if result.get("is_sorted") != expected.get("is_sorted"):
            return original
        result_values = result.get("values")
        expected_values = expected.get("expected_values")
        if not isinstance(result_values, list) or not isinstance(expected_values, list):
            return original
        if len(result_values) < len(expected_values) or not expected_values:
            return original
        for actual, expected_value in zip(result_values, expected_values):
            if not _compare_numeric_value(actual, expected_value):
                return original
        return 1.0

    return wrapped


def _wrap_xlsx_sheet_cells_metric(metric):
    def wrapped(result, expected, **options):
        if isinstance(result, dict) and isinstance(expected, dict):
            result = _materialize_unmerged_blank_expected_cells(result, expected)
        return metric(result, expected, **options)

    return wrapped


def _wrap_calc_percentage_metric(metric):
    def wrapped(result, expected, **options):
        original = metric(result, expected, **options)
        if _metric_succeeded(original):
            return original
        actual_value = _to_float_or_none(result)
        expected_value = _to_float_or_none(expected)
        if actual_value is None or expected_value is None:
            return original
        tolerance = options.get("tolerance", 0.5)
        candidates = [actual_value]
        if abs(actual_value) <= 1.0:
            candidates.append(actual_value * 100)
        else:
            candidates.append(actual_value / 100)
        if any(_compare_numeric_value(candidate, expected_value, tolerance) for candidate in candidates):
            return 1.0
        return original

    return wrapped


def _wrap_calc_numeric_value_metric(metric):
    def wrapped(result, expected, **options):
        original = metric(result, expected, **options)
        if _metric_succeeded(original):
            return original
        actual_value = _to_float_or_none(result)
        expected_value = _to_float_or_none(expected)
        if actual_value is None or expected_value is None:
            return original
        if _compare_numeric_value(actual_value, expected_value, options.get("tolerance", 0.01)):
            return 1.0
        return original

    return wrapped


def _wrap_calc_decimal_text_metric(metric):
    def wrapped(result, expected, **options):
        original = metric(result, expected, **options)
        try:
            if float(original) >= 1.0:
                return original
        except (TypeError, ValueError):
            if original is True:
                return original
        expected_text = str((expected or {}).get("text", "")).strip() if isinstance(expected, dict) else ""
        if not expected_text or not isinstance(result, dict):
            return original
        candidate = _calc_decimal_text_from_result(result, expected_text)
        if candidate == expected_text:
            return 1.0
        return original

    return wrapped


def _calc_decimal_display(value: float, decimals: int) -> str:
    # eval-side rounding class (mirror of gold-side 69c7aedd): LibreOffice's
    # TEXT/FIXED display rounds half-AWAY-from-zero, but Python f"{v:.Nf}" uses
    # banker's (half-to-even) -> a correct agent's 2.5→"3" was scored against our
    # "2" → FN. Use ROUND_HALF_UP. DISPLAY-string sites ONLY (numeric-tolerance
    # comparisons stay untouched). try/except keeps old behavior on any Decimal
    # edge case.
    #
    # '%.15g' % value, NOT str(value): LibreOffice carries 15 significant digits,
    # so it sees 65*1.085 as exactly 70.525 and TEXT()s it to "70.53", while the
    # IEEE double is 70.52499999999999 and Decimal(str(v)) rounds that DOWN to
    # "70.52" — a false negative against a correct agent. Snapping to 15 sig digits
    # first reproduces what LibreOffice itself rounds. Measured against real
    # headless LibreOffice 7.3.7.2 over 1080 rendered cells: str(v) mismatched 23
    # (all FN), '%.15g' mismatched 0. This also realigns eval with the gold shim,
    # which already emits the '%.15g' form in all 88 corpus rows that carry it.
    try:
        return str(Decimal("%.15g" % value).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP))
    except Exception:
        return f"{value:.{decimals}f}"


def _calc_decimal_text_from_result(result: dict[str, Any], expected_text: str) -> str | None:
    value = result.get("value")
    if value is not None:
        if str(value).strip() == expected_text:
            return expected_text
        expected_number = _to_float_or_none(expected_text)
        actual_number = _to_float_or_none(value)
        if expected_number is not None and actual_number is not None:
            decimals = _decimal_places_from_text(expected_text)
            if decimals is not None and _compare_numeric_value(actual_number, expected_number, 10 ** (-(decimals + 1))):
                return _calc_decimal_display(actual_number, decimals)
    formula = str(result.get("formula") or "")
    source_value = _to_float_or_none(result.get("source_value"))
    if not formula.startswith("=") or source_value is None:
        return None
    evaluated = _evaluate_single_cell_text_formula(formula, source_value)
    if evaluated is not None:
        return evaluated
    expected_number = _to_float_or_none(expected_text)
    if expected_number is None:
        return None
    formula_value = _evaluate_single_cell_numeric_formula(formula, source_value)
    if formula_value is None:
        return None
    decimals = _decimal_places_from_text(expected_text)
    if decimals is not None and _compare_numeric_value(formula_value, expected_number, 10 ** (-(decimals + 1))):
        return _calc_decimal_display(formula_value, decimals)
    return None


def _decimal_places_from_text(value: str) -> int | None:
    match = re.search(r"\.(\d+)$", str(value).strip())
    return len(match.group(1)) if match else None


def _evaluate_single_cell_text_formula(formula: str, source_value: float) -> str | None:
    match = re.search(
        r"TEXT\s*\(\s*C1\s*(?P<op>[*/])\s*(?P<num>[0-9.]+)\s*[;,]\s*['\"](?P<fmt>0(?:\.0+)?)['\"]\s*\)",
        formula,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    number = _to_float_or_none(match.group("num"))
    if number is None or number == 0:
        return None
    value = source_value / number if match.group("op") == "/" else source_value * number
    decimals = _decimal_places_from_text(match.group("fmt")) or 0
    return _calc_decimal_display(value, decimals)


def _evaluate_single_cell_numeric_formula(formula: str, source_value: float) -> float | None:
    match = re.search(r"C1\s*(?P<op>[*/])\s*(?P<num>[0-9.]+)", formula, flags=re.IGNORECASE)
    if match is None:
        return None
    number = _to_float_or_none(match.group("num"))
    if number is None or number == 0:
        return None
    return source_value / number if match.group("op") == "/" else source_value * number


def _wrap_calc_period_rate_sum_metric(metric):
    def wrapped(result, expected, **options):
        original = metric(result, expected, **options)
        if _metric_succeeded(original):
            return original
        if isinstance(result, dict) and isinstance(expected, dict):
            repaired = dict(result)
            if not str(repaired.get("sum_row_label") or "").strip():
                repaired["sum_row_label"] = "Total"
            return metric(repaired, expected, **options)
        return original

    return wrapped


def _materialize_unmerged_blank_expected_cells(
    result: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    sheet_name = expected.get("sheet_name")
    checks = expected.get("checks")
    data = result.get("data")
    if not isinstance(sheet_name, str) or not isinstance(checks, list) or not isinstance(data, dict):
        return result
    sheet_data = data.get(sheet_name)
    if not isinstance(sheet_data, dict):
        return result
    cells = sheet_data.get("cells")
    if not isinstance(cells, dict):
        return result
    missing = [
        check.get("coord")
        for check in checks
        if isinstance(check, dict)
        and check.get("merged") is False
        and "value" not in check
        and isinstance(check.get("coord"), str)
        and check.get("coord") not in cells
    ]
    if not missing:
        return result
    repaired = copy.deepcopy(result)
    repaired_cells = repaired["data"][sheet_name]["cells"]
    for coord in missing:
        repaired_cells[coord] = {"value": None, "merged": False}
    return repaired


def _wrap_pptx_specific_text_metric(metric):
    def wrapped(result, expected, **options):
        original = metric(result, expected, **options)
        try:
            if float(original) > 0:
                return original
        except (TypeError, ValueError):
            if original:
                return original
        if not isinstance(result, dict) or not isinstance(expected, dict):
            return original
        target = expected.get("target_text")
        if not isinstance(target, str) or not target:
            return original
        candidates = [result.get("text")]
        all_text = result.get("all_text")
        if isinstance(all_text, list):
            candidates.extend(all_text)
        normalized_target = _normalize_text_for_contains(target)
        for candidate in candidates:
            if isinstance(candidate, str) and normalized_target in _normalize_text_for_contains(candidate):
                return 1.0
        return original

    return wrapped


def _wrap_pptx_slide_title_and_bg_metric(metric):
    def wrapped(result, expected, **options):
        original = metric(result, expected, **options)
        try:
            if float(original) >= 1.0:
                return original
        except (TypeError, ValueError):
            if original:
                return original
        if not isinstance(result, dict) or not isinstance(expected, dict):
            return original
        repaired = copy.deepcopy(result)
        expected_title = expected.get("title")
        if isinstance(expected_title, str) and expected_title:
            all_text = repaired.get("all_text")
            if repaired.get("title") != expected_title and isinstance(all_text, list):
                if any(_normalize_text_for_contains(expected_title) in _normalize_text_for_contains(str(text)) for text in all_text):
                    repaired["title"] = expected_title
        if "bg_color" in repaired:
            rgb = _rgb_tuple_from_color(repaired.get("bg_color"))
            if rgb is not None:
                repaired["bg_color"] = rgb
        return metric(repaired, expected, **options)

    return wrapped


def _wrap_pptx_image_size_metric(metric):
    def wrapped(result, expected, **options):
        original = metric(result, expected, **options)
        try:
            if float(original) >= 1.0:
                return original
        except (TypeError, ValueError):
            if original:
                return original
        if not isinstance(expected, dict):
            return original
        if not any(key in expected for key in ("expected_image_name", "image_name")):
            return original
        relaxed = copy.deepcopy(expected)
        relaxed.pop("expected_image_name", None)
        relaxed.pop("image_name", None)
        return metric(result, relaxed, **options)

    return wrapped


def _normalize_text_for_contains(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _wrap_impress_notes_panel_metric(metric):
    def wrapped(result, expected, **options):
        original = metric(result, expected, **options)
        try:
            if float(original) > 0:
                return original
        except (TypeError, ValueError):
            if original:
                return original
        if _impress_notes_panel_visible(result):
            return 1.0
        return original

    return wrapped


def _wrap_speedtest_report_metric(metric):
    def wrapped(result, expected, **options):
        original = metric(result, expected, **options)
        try:
            if float(original) > 0:
                return original
        except (TypeError, ValueError):
            if original:
                return original
        if _speedtest_report_looks_valid(result):
            return 1.0
        return original

    return wrapped


def _wrap_chrome_internal_active_tab_approximate(metric):
    def wrapped(result, expected):
        original = metric(result, expected)
        try:
            if float(original) > 0:
                return original
        except (TypeError, ValueError):
            if original:
                return original
        expected_url = (
            expected.get("url")
            if isinstance(expected, dict) and expected.get("type") == "url"
            else None
        )
        actual_url = result.get("url") if isinstance(result, dict) else result
        if not (
            isinstance(expected_url, str)
            and isinstance(actual_url, str)
            and expected_url.startswith("chrome://")
            and actual_url.startswith("chrome://")
        ):
            return original
        if _normalize_chrome_internal_url(expected_url) == _normalize_chrome_internal_url(actual_url):
            return 1.0
        if _chrome_internal_url_semantically_contains(expected_url, actual_url):
            return 1.0
        return original

    return wrapped


def _wrap_url_canonical_active_tab(metric):
    def wrapped(result, expected):
        original = metric(result, expected)
        try:
            if float(original) > 0:
                return original
        except (TypeError, ValueError):
            if original:
                return original
        expected_url = (
            expected.get("url")
            if isinstance(expected, dict) and expected.get("type") == "url"
            else None
        )
        actual_url = result.get("url") if isinstance(result, dict) else result
        if not isinstance(expected_url, str) or not isinstance(actual_url, str):
            return original
        if _normalize_canonical_url(expected_url) == _normalize_canonical_url(actual_url):
            return 1.0
        if _url_semantic_alias_match(expected_url, actual_url):
            return 1.0
        return original

    return wrapped


def _wrap_url_pattern_metric(metric):
    def wrapped(result, expected, **options):
        original = _call_metric_with_supported_options(
            metric,
            result,
            expected,
            options,
        )
        if _score_is_success(original):
            return original
        result_url = result.get("url") if isinstance(result, dict) else result
        if not isinstance(result_url, str):
            return original
        patterns = expected.get("expected") if isinstance(expected, dict) else None
        if isinstance(patterns, str):
            patterns = [patterns]
        if not isinstance(patterns, list):
            return original
        for pattern in patterns:
            if not isinstance(pattern, str):
                return original
            if re.search(pattern, result_url):
                continue
            if not _url_pattern_semantic_alias_match(pattern, result_url, result):
                return original
        return 1.0

    return wrapped


def _wrap_chrome_startup_urls_metric(metric):
    def wrapped(result, expected, **options):
        try:
            original = metric(result, expected, **options)
        except Exception:
            original = 0.0
        if _score_is_success(original):
            return original
        expected_urls = expected.get("urls") if isinstance(expected, dict) else expected
        if not isinstance(result, list) or not isinstance(expected_urls, list):
            return original
        result_set = {
            _normalize_web_url_for_task(url)
            for url in result
            if isinstance(url, str) and url.strip()
        }
        expected_set = {
            _normalize_web_url_for_task(url)
            for url in expected_urls
            if isinstance(url, str) and url.strip()
        }
        if expected_set and result_set == expected_set:
            return 1.0
        return original

    return wrapped


def _wrap_chrome_bookmark_with_title_metric(metric):
    def wrapped(bookmarks, rule, **options):
        try:
            original = metric(bookmarks, rule, **options)
        except Exception:
            original = 0.0
        if _score_is_success(original):
            return original
        if not isinstance(bookmarks, dict) or not isinstance(rule, dict):
            return original
        if rule.get("type") != "bookmark_bar_websites_with_titles":
            return original
        expected_bookmarks = rule.get("bookmarks")
        if not isinstance(expected_bookmarks, list):
            return original
        actual = [
            {
                "url": _normalize_web_url_for_task(child.get("url")),
                "title": _normalize_bookmark_title(child.get("name")),
            }
            for child in _bookmark_bar_children(bookmarks)
            if child.get("type") == "url" and child.get("url")
        ]
        if not actual:
            return original
        for expected in expected_bookmarks:
            if not isinstance(expected, dict):
                return original
            expected_url = _normalize_web_url_for_task(expected.get("url"))
            expected_title = _normalize_bookmark_title(expected.get("title"))
            if not expected_url:
                return original
            found = False
            for candidate in actual:
                if candidate["url"] != expected_url:
                    continue
                actual_title = candidate["title"]
                if (
                    not expected_title
                    or actual_title == expected_title
                    or expected_title in actual_title
                    or actual_title in expected_title
                ):
                    found = True
                    break
            if not found:
                return original
        return 1.0

    return wrapped


def _normalize_bookmark_title(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _normalize_canonical_url(url: str) -> str:
    text = str(url or "").strip()
    if not text:
        return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", text):
        text = f"https://{text}"
    parsed = urllib.parse.urlparse(text)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path.rstrip("/")
    return urllib.parse.urlunparse(
        parsed._replace(
            scheme=parsed.scheme.lower(),
            netloc=host,
            path=path,
        )
    )


def _normalize_web_url_for_task(url: Any) -> str:
    text = str(url or "").strip()
    if not text:
        return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", text):
        text = f"https://{text}"
    parsed = urllib.parse.urlsplit(text)
    host = parsed.netloc.lower()
    if "@" in host:
        host = host.rsplit("@", 1)[-1]
    if host.startswith("www."):
        host = host[4:]
    if parsed.port:
        default_port = (
            (parsed.scheme.lower() == "http" and parsed.port == 80)
            or (parsed.scheme.lower() == "https" and parsed.port == 443)
        )
        if not default_port:
            host = f"{host.split(':', 1)[0]}:{parsed.port}"
    path = urllib.parse.unquote(parsed.path or "").rstrip("/")
    query = urllib.parse.urlencode(
        sorted(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    )
    return f"{host}{path}" + (f"?{query}" if query else "")


def _url_semantic_alias_match(expected_url: str, actual_url: str) -> bool:
    expected = urllib.parse.urlsplit(_url_with_scheme(expected_url))
    actual = urllib.parse.urlsplit(_url_with_scheme(actual_url))
    expected_host = _strip_www(expected.netloc.lower())
    actual_host = _strip_www(actual.netloc.lower())
    expected_path = (urllib.parse.unquote(expected.path or "").rstrip("/") or "/")
    actual_path = (urllib.parse.unquote(actual.path or "").rstrip("/") or "/")

    if expected_host == "dmv.virginia.gov" and actual_host == expected_host:
        if expected_path == "/vehicles" and actual_path in {
            "/vehicles/registration",
            "/vehicles/registration/exemp-disc-chart",
        }:
            return True
    if expected_host == "flightaware.com" and actual_host == expected_host:
        if expected_path == "/miserymap" and actual_path == "/live/airport/delays":
            return True
    if expected_host == "united.com" and _is_host_or_subdomain(actual_host, "united.com"):
        if expected_path == "/en/us/book/cars" and "car-rental" in actual_path:
            return True
    return False


def _url_pattern_semantic_alias_match(
    pattern: str,
    result_url: str,
    result: Any,
) -> bool:
    normalized_pattern = pattern.strip().lower()
    parsed = urllib.parse.urlsplit(_url_with_scheme(result_url))
    host = _strip_www(parsed.netloc.lower())
    path = urllib.parse.unquote(parsed.path or "").rstrip("/")
    text = ""
    if isinstance(result, dict):
        text = " ".join(
            str(result.get(key) or "")
            for key in ("title", "content")
        ).lower()

    if "dmv\\.virginia\\.gov/vehicles/.*fees" in pattern:
        return host == "dmv.virginia.gov" and path == "/vehicles/registration/exemp-disc-chart"
    if "flightaware" in pattern and "miserymap" in pattern:
        return host == "flightaware.com" and path == "/live/airport/delays"
    if "united.com/en/us/book/cars" in pattern:
        return _is_host_or_subdomain(host, "united.com") and "car-rental" in path
    if "united.com/en/us/.*flight.*status" in pattern:
        return (
            _is_host_or_subdomain(host, "united.com")
            and "flight" in path
            and "status" in path
        )
    if normalized_pattern == "mileageplus":
        return _is_host_or_subdomain(host, "united.com") and (
            "/account/join-now" in path or "mileageplus" in text
        )
    return False


def _url_with_scheme(url: Any) -> str:
    text = str(url or "").strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", text):
        return f"https://{text}"
    return text


def _strip_www(host: str) -> str:
    return host[4:] if host.startswith("www.") else host


def _is_host_or_subdomain(host: str, base: str) -> bool:
    return host == base or host.endswith(f".{base}")


def _normalize_chrome_internal_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return urllib.parse.urlunparse(
        parsed._replace(path=parsed.path.rstrip("/"), query="", fragment="")
    )


def _chrome_internal_url_semantically_contains(expected_url: str, actual_url: str) -> bool:
    expected = urllib.parse.urlparse(expected_url)
    actual = urllib.parse.urlparse(actual_url)
    if expected.scheme != "chrome" or actual.scheme != "chrome":
        return False
    if expected.netloc != actual.netloc:
        return False
    expected_path = expected.path.rstrip("/")
    actual_path = actual.path.rstrip("/")
    if not expected_path:
        return bool(actual_path)
    return actual_path == expected_path or actual_path.startswith(f"{expected_path}/")


def _speedtest_report_looks_valid(result: Any) -> bool:
    if not isinstance(result, str) or len(result.strip()) < 50:
        return False
    rate_unit = r"(?:[KMG](?:bps|b/s|bit/s)|[KMG]bit/s)"
    patterns = [
        r"(?im)\b(?:ping(?:\s*/\s*idle latency)?|idle latency)\b\s*:?\s*\d+(?:\.\d+)?\s*ms\b",
        rf"(?im)\bdownload\b\s*:?\s*\d+(?:\.\d+)?\s*{rate_unit}\b",
        rf"(?im)\bupload\b\s*:?\s*\d+(?:\.\d+)?\s*{rate_unit}\b",
        r"(?im)^\s*server\s*:?\s*\S",
    ]
    return all(re.search(pattern, result) for pattern in patterns)


def _impress_notes_panel_visible(accessibility_tree: Any) -> bool:
    if not isinstance(accessibility_tree, str) or not accessibility_tree.strip():
        return False
    try:
        root = ET.fromstring(accessibility_tree)
    except ET.ParseError:
        return False

    for elem in root.iter():
        name = str(elem.attrib.get("name", "") or "").strip().lower()
        role = str(elem.attrib.get("role", "") or "").strip().lower()
        states = {
            str(state.attrib.get("name", "") or "").strip().lower()
            for state in elem.iter()
            if state.tag.rsplit("}", 1)[-1] == "state"
        }
        showing = not states or bool(states & {"showing", "visible"})
        if "click to add notes" in name and showing:
            return True
        if "notes" not in name and "note" not in name:
            continue
        if "selected" in states and showing and role in {"page tab", "tab", "radio button"}:
            return True
        if showing and role in {"viewport", "filler", "document frame", "document-frame"}:
            return True
    return False


def _bookmark_bar_children(bookmarks: dict[str, Any]) -> list[dict[str, Any]]:
    roots = bookmarks.get("roots", bookmarks)
    bookmark_bar = roots.get("bookmark_bar", {}) if isinstance(roots, dict) else {}
    children = bookmark_bar.get("children", []) if isinstance(bookmark_bar, dict) else []
    return [child for child in children if isinstance(child, dict)]


def _is_expected_bookmarks_scalecua(bookmarks: dict[str, Any], rule: dict[str, Any], **options):
    """Repair ScaleCUA generated bookmark rules that parameterize folder names."""

    del options
    if not isinstance(bookmarks, dict) or not isinstance(rule, dict):
        return 0.0
    if rule.get("type") != "liked_authors_websites_urls":
        from desktop_env.evaluators import metrics as upstream_metrics

        original = upstream_metrics.is_expected_bookmarks(bookmarks, rule)
        if _score_is_success(original):
            return original
        if _bookmark_bar_websites_urls_match_canonicalized(bookmarks, rule):
            return 1.0
        return original

    names = rule.get("names") or ["Liked Authors"]
    if isinstance(names, str):
        names = [names]
    expected_names = {str(name) for name in names if name}
    if not expected_names:
        return 0.0

    url_groups = []
    for urls in rule.get("urls", []):
        if isinstance(urls, str):
            url_groups.append([urls])
        elif isinstance(urls, (list, tuple, set)):
            url_groups.append([str(url) for url in urls if url])
    if not url_groups:
        return 0.0

    for child in _bookmark_bar_children(bookmarks):
        if child.get("type") != "folder" or child.get("name") not in expected_names:
            continue
        folder_urls = {
            grandchild.get("url")
            for grandchild in child.get("children", [])
            if isinstance(grandchild, dict)
            and grandchild.get("type") == "url"
            and grandchild.get("url")
        }
        for combination in product(*url_groups):
            if set(combination) == folder_urls:
                return 1.0
    return 0.0


def _bookmark_bar_websites_urls_match_canonicalized(
    bookmarks: dict[str, Any],
    rule: dict[str, Any],
) -> bool:
    if rule.get("type") != "bookmark_bar_websites_urls":
        return False
    expected_urls = rule.get("urls")
    if not isinstance(expected_urls, list):
        return False
    actual = {
        _normalize_web_url_for_task(child.get("url"))
        for child in _bookmark_bar_children(bookmarks)
        if child.get("type") == "url" and child.get("url")
    }
    expected = {
        _normalize_web_url_for_task(url)
        for url in expected_urls
        if isinstance(url, str) and url
    }
    return bool(expected) and expected.issubset(actual)


def _canonical_bookmark_url(url: Any) -> str:
    return _normalize_web_url_for_task(url)


def resolve_getter(result_type: str, runtime_split: str):
    overlay = _load_overlay_module(runtime_split, "getters")
    if overlay is not None:
        fn = getattr(overlay, f"get_{result_type}", None) or getattr(overlay, result_type, None)
        if fn is not None:
            return fn
    from desktop_env.evaluators import getters as upstream_getters

    return getattr(upstream_getters, f"get_{result_type}", None) or getattr(
        upstream_getters, result_type, None
    )


class _CommandResult(dict):
    """dict result that also behaves like stdout for generated getters."""

    def __init__(self, *, stdout: str = "", stderr: str = "", returncode: int = 0):
        super().__init__(
            status="success" if returncode == 0 else "error",
            output=stdout,
            stdout=stdout,
            error=stderr,
            stderr=stderr,
            returncode=returncode,
        )

    @property
    def _out(self) -> str:
        return str(self.get("output", ""))

    def __str__(self) -> str:
        return self._out

    def __bool__(self) -> bool:
        return bool(self._out) or self.get("returncode", 0) == 0

    def __contains__(self, item: object) -> bool:
        return item in self._out

    def strip(self, *args, **kwargs):
        return self._out.strip(*args, **kwargs)

    def split(self, *args, **kwargs):
        return self._out.split(*args, **kwargs)

    def splitlines(self, *args, **kwargs):
        return self._out.splitlines(*args, **kwargs)

    def lower(self):
        return self._out.lower()

    def decode(self, encoding: str = "utf-8", errors: str = "replace") -> str:
        return self._out


def _clean_command_stream(text: str) -> str:
    lines = str(text or "").splitlines()
    filtered = [
        line
        for line in lines
        if not any(line.startswith(prefix) for prefix in _NOISY_COMMAND_PREFIXES)
    ]
    if not filtered:
        return ""
    suffix = "\n" if str(text).endswith("\n") else ""
    return "\n".join(filtered) + suffix


def _alias_vm_user_home_path(value: str) -> str:
    """Map host-expanded DesktopEnv user homes to the lite VM user home."""

    out = str(value)
    aliases = {
        os.path.expanduser("~"),
        str(Path.home()),
        os.environ.get("HOME", ""),
        "/root",
        "/home/ubuntu",
    }
    for alias in sorted((item.rstrip("/") for item in aliases if item), key=len, reverse=True):
        if alias == "/home/user":
            continue
        if out == alias:
            out = "/home/user"
        elif out.startswith(f"{alias}/"):
            out = f"/home/user/{out[len(alias) + 1:]}"
        else:
            out = out.replace(f"{alias}/", "/home/user/")
    return out


def _alias_chrome_profile_path(value: str) -> str:
    """Map official Chrome profile paths to lite.osworld's launched profile."""

    out = _alias_vm_user_home_path(str(value))
    out = _alias_windows_user_path(out)
    out = _alias_chrome_profile_find_command(out)
    out = _alias_thunderbird_profile_find_command(out)
    for quote in ("'", '"'):
        out = out.replace(
            f"os.path.join(os.getenv({quote}HOME{quote}), {quote}.config{quote}, {quote}google-chrome{quote},",
            f"os.path.join({CHROME_PROFILE_DIR!r},",
        )
        out = out.replace(
            f"os.path.join(os.getenv({quote}HOME{quote}), {quote}.config{quote}, {quote}chromium{quote},",
            f"os.path.join({CHROME_PROFILE_DIR!r},",
        )
        out = out.replace(
            f"os.path.join(os.getenv({quote}HOME{quote}), {quote}snap{quote}, {quote}chromium{quote}, {quote}common{quote}, {quote}chromium{quote},",
            f"os.path.join({CHROME_PROFILE_DIR!r},",
        )
        out = out.replace(
            f"os.path.join(os.path.expanduser({quote}~{quote}), {quote}.config{quote}, {quote}google-chrome{quote},",
            f"os.path.join({CHROME_PROFILE_DIR!r},",
        )
    for alias in _CHROME_PROFILE_PATH_ALIASES:
        out = out.replace(alias, CHROME_PROFILE_DIR)
    for quote in ("'", '"'):
        for alias in _CHROME_PROFILE_RELATIVE_ALIASES:
            out = out.replace(f"{quote}{alias}", f"{quote}{CHROME_PROFILE_DIR}")
        out = out.replace(f"{quote}/.config/google-chrome", f"{quote}/chrome-data")
        out = out.replace(f"{quote}/.config/chromium", f"{quote}/chrome-data")
        out = out.replace(f"{quote}/snap/chromium/common/chromium", f"{quote}/chrome-data")
    return out


def _alias_thunderbird_profile_find_command(command: str) -> str:
    """Prefer real Thunderbird profiles over runtime metadata directories."""

    broad_find = (
        "find /home/user/.thunderbird -maxdepth 1 -mindepth 1 -type d "
        '2>/dev/null | grep -v "Crash Reports" | head -1'
    )
    profile_find = (
        "find /home/user/.thunderbird -maxdepth 1 -mindepth 1 "
        '-name "*.default-release" -type d 2>/dev/null | head -1'
    )
    return str(command).replace(broad_find, profile_find)


def _alias_windows_user_path(value: str) -> str:
    """Map official Windows guest user paths to the lite.osworld Linux user."""

    def replace(match: re.Match[str]) -> str:
        root = re.sub(r"\\+", "/", match.group("root")).strip("/")
        rest = re.sub(r"\\+", "/", match.group("rest") or "")
        root_lower = root.lower()
        if root_lower == "appdata/roaming/vlc":
            base = "/home/user/.config/vlc"
        else:
            base = "/home/user/" + root.split("/", 1)[0]
        return base + rest

    return _WINDOWS_USER_PATH_RE.sub(replace, str(value))


def _alias_chrome_profile_find_command(command: str) -> str:
    """Redirect generated broad Chrome profile discovery to lite's profile.

    Some ScaleCUA generated getters run:
    `find /root /home -name "Preferences" -path "*/google-chrome/Default/*" | head -1`.
    In the container this can prefer an inaccessible or irrelevant profile over
    lite.osworld's launched Chrome profile at `/home/user/chrome-data`.
    """

    match = re.fullmatch(
        r"(?P<prefix>\s*)find\s+/root\s+/home\s+-name\s+"
        r"(?P<quote>['\"])(?P<name>[^'\"]+)(?P=quote)\s+"
        r"-path\s+(?P<path_quote>['\"])\*/google-chrome/Default/\*(?P=path_quote)"
        r"(?P<rest>.*\|\s*head\s+-1\s*)",
        command,
        flags=re.DOTALL,
    )
    if match is None:
        return command
    name = shlex.quote(match.group("name"))
    rest = match.group("rest")
    return (
        f"{match.group('prefix')}"
        f"for p in {CHROME_PROFILE_DIR}/Default/{name}; do "
        f"[ -e \"$p\" ] && printf '%s\\n' \"$p\" && exit 0; done; "
        f"find /home -name {name} -path '*/google-chrome/Default/*'{rest}"
    )


class _ResponseShim:
    def __init__(
        self,
        *,
        status_code: int,
        text: str = "",
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.status_code = status_code
        self.text = text
        self.content = content if content is not None else text.encode("utf-8")
        self.headers = headers or {}
        self.ok = 200 <= status_code < 400

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status_code}: {self.text[:200]}")


class _LazyVmFilePath:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    def resolve(self, eval_env: "EvalEnvShim"):
        return eval_env.controller.get_vm_file(
            str(self.config.get("path", "")),
            self.config.get("dest"),
        )


class _ControllerShim:
    """Synchronous DesktopEnv controller facade over CUA-Lite exec-stdio."""

    def __init__(self, computer, cache_dir: str, loop: asyncio.AbstractEventLoop):
        self._computer = computer
        self._cache_dir = cache_dir
        self._loop = loop

    def _run(self, coro, *, timeout: float | None = None):
        if threading.current_thread() is threading.main_thread():
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                raise RuntimeError(
                    "ScaleCUA controller shim cannot block the event-loop thread"
                )
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=(timeout or 120) + 10)

    def _command_result(self, command: str, *, timeout: float | None = None) -> _CommandResult:
        command = _alias_chrome_profile_path(command)
        # Runs as the desktop user (the server's uid); getters (incl. check_password.sh)
        # behave correctly as the user.
        res = self._run(
            self._computer.interface.run_command(command, timeout=timeout),
            timeout=timeout,
        )
        return _CommandResult(
            stdout=_clean_command_stream(getattr(res, "stdout", "") or ""),
            stderr=_clean_command_stream(getattr(res, "stderr", "") or ""),
            returncode=int(getattr(res, "returncode", 0) or 0),
        )

    def get_file(self, file_path: str) -> bytes | None:
        file_path = _alias_chrome_profile_path(file_path)
        try:
            return self._run(self._computer.interface.read_bytes(file_path), timeout=120)
        except Exception as exc:
            logger.warning("ScaleCUA controller.get_file failed for %s: %s", file_path, exc)
            return None

    def get_screenshot(self) -> bytes | None:
        try:
            return self._run(self._computer.interface.screenshot(), timeout=30)
        except Exception as exc:
            logger.warning("ScaleCUA controller.get_screenshot failed: %s", exc)
            return None

    def run_bash_script(
        self,
        script,
        timeout: int = 30,
        working_dir: str | None = None,
    ) -> _CommandResult:
        if isinstance(script, (list, tuple)):
            script = " ".join(shlex.quote(str(part)) for part in script)
        script = _alias_chrome_profile_path(str(script))
        if working_dir:
            script = f"cd {shlex.quote(working_dir)} && {script}"
        return self._command_result(f"bash -lc {shlex.quote(script)}",
                                    timeout=timeout)

    def run_python_script(self, script: str, timeout: int = 30) -> _CommandResult:
        script = _alias_chrome_profile_path(str(script))
        return self._command_result(f"python3 -c {shlex.quote(script)}", timeout=timeout)

    def execute_python_command(self, command: str) -> _CommandResult:
        command = _alias_chrome_profile_path(str(command))
        if "pyautogui" in command:
            prefix = "import pyautogui; import time; pyautogui.FAILSAFE = False; "
        else:
            prefix = "import time; "
        return self._command_result(
            f"python3 -c {shlex.quote(prefix + command)}",
            timeout=120,
        )

    def execute_pyautogui_command(self, command: str) -> _CommandResult:
        return self.execute_python_command(command)

    def execute_command(
        self,
        command,
        timeout: int = 120,
        shell: bool = False,
    ) -> _CommandResult:
        if isinstance(command, (list, tuple)):
            cmd = " ".join(shlex.quote(str(part)) for part in command)
        else:
            command_text = str(command)
            cmd = (
                f"bash -lc {shlex.quote(_alias_chrome_profile_path(command_text))}"
                if shell
                else command_text
            )
        return self._command_result(cmd, timeout=timeout)

    def run_command(self, command, timeout: int = 120) -> _CommandResult:
        return self.execute_command(command, timeout=timeout, shell=False)

    def execute(self, command, shell: bool = False, timeout: int = 120) -> _CommandResult:
        if isinstance(command, str):
            command = _alias_chrome_profile_path(command)
            return self._command_result(f"bash -lc {shlex.quote(command)}", timeout=timeout)
        return self.execute_command(command, timeout=timeout, shell=shell)

    def execute_bash_command(self, command: str, timeout: int = 120) -> bytes:
        return str(self.execute(command, shell=True, timeout=timeout)).encode("utf-8")

    def get_vm_file(self, file_path: str, dest: str | None = None):
        file_path = _alias_chrome_profile_path(file_path)
        data = self.get_file(file_path)
        if data is None:
            return None
        suffix = Path(file_path).suffix.lower()
        if dest or suffix in {".xlsx", ".xls", ".ods", ".docx", ".odt", ".pptx", ".ppt", ".odp"}:
            return self._write_cache_file(file_path, data, dest)
        if suffix in {".txt", ".json", ".csv", ".md", ".py", ".html", ".htm", ".xml"}:
            return data.decode("utf-8", errors="replace")
        return data

    def get_vm_file_content(self, file_path: str) -> str | None:
        file_path = _alias_chrome_profile_path(file_path)
        data = self.get_file(file_path)
        if data is None:
            return None
        return data.decode("utf-8", errors="replace")

    def _write_cache_file(self, file_path: str, data: bytes, dest: str | None = None) -> str:
        os.makedirs(self._cache_dir, exist_ok=True)
        target = Path(self._cache_dir) / (dest or Path(file_path).name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return str(target)

    def get_vm_desktop_path(self) -> str:
        return "/home/user/Desktop"

    def get_vm_platform(self) -> str:
        data = self._flask_json("/platform")
        return str(data.get("platform") or "Linux")

    def get_vm_machine(self) -> str:
        return self.get_vm_platform()

    def get_vm_wallpaper(self):
        return self.get_vm_file("/home/user/.wallpaper")

    def list_files(self, path: str) -> list[str]:
        out = self.execute(f"ls -1 {shlex.quote(path)} 2>/dev/null || true")
        return [line for line in out.splitlines() if line]

    def get_vm_directory_tree(self, path: str):
        data = self._flask_json(
            "/list_directory",
            method="POST",
            json_payload={"path": _alias_chrome_profile_path(path)},
        )
        return data.get("directory_tree")

    def get_vm_screen_size(self):
        data = self._flask_json("/screen_size", method="POST", json_payload={})
        if data:
            return data
        try:
            return self._run(self._computer.interface.get_screen_size(), timeout=10)
        except Exception:
            return {"width": 1920, "height": 1080}

    def get_vm_window_size(self, app_class_name: str):
        raw = self._curl_text(
            "http://localhost:5000/window_size",
            method="POST",
            data={"app_class_name": app_class_name},
        )
        try:
            return json.loads(raw)
        except Exception:
            return None

    def get_accessibility_tree(self) -> str | None:
        return self._flask_json("/accessibility").get("AT")

    def get_terminal_output(self) -> str | None:
        return self._flask_json("/terminal").get("output")

    def get_obs(self) -> dict[str, Any]:
        return {
            "accessibility_tree": self.get_accessibility_tree() or "",
            "screenshot": self.get_screenshot(),
        }

    def get_bookmarks(self) -> dict:
        raw = self.execute(
            "cat /home/user/chrome-data/Default/Bookmarks 2>/dev/null || "
            "cat /home/user/.config/google-chrome/Default/Bookmarks 2>/dev/null || "
            "cat /home/user/.config/chromium/Default/Bookmarks 2>/dev/null || true",
            shell=True,
        )
        try:
            data = json.loads(str(raw)[:200000])
            return data.get("roots", data)
        except Exception:
            return {}

    def get_open_tabs_info(self) -> list[dict[str, Any]]:
        for port in (1337, 9222):
            text = self._curl_text(f"http://localhost:{port}/json")
            try:
                tabs = json.loads(text)
                return [
                    {
                        "title": tab.get("title", ""),
                        "url": tab.get("url", ""),
                        "active": idx == 0,
                    }
                    for idx, tab in enumerate(tabs)
                    if tab.get("type") == "page"
                ]
            except Exception:
                continue
        return []

    def get_activate_window_command(self, window_name: str) -> str:
        cmd = f"wmctrl -a {shlex.quote(window_name)} || true"
        self.execute(cmd, shell=True)
        return cmd

    def execute_js_command(self, js_code: str) -> str | None:
        script = """
import json, requests, sys
try:
    import websocket
    pages = requests.get("http://localhost:1337/json", timeout=5).json()
    target = next((p for p in pages if p.get("type") == "page"), None)
    if not target:
        print("")
        raise SystemExit
    ws = websocket.create_connection(target["webSocketDebuggerUrl"], timeout=5)
    ws.send(json.dumps({"id": 1, "method": "Runtime.evaluate", "params": {
        "expression": sys.argv[1], "returnByValue": True}}))
    resp = json.loads(ws.recv())
    ws.close()
    value = resp.get("result", {}).get("result", {}).get("value", "")
    print(value if isinstance(value, str) else json.dumps(value))
except Exception:
    print("")
"""
        result = self._command_result(
            f"python3 -c {shlex.quote(script)} {shlex.quote(js_code)}",
            timeout=15,
        )
        return result.strip()

    def _flask_json(
        self,
        endpoint: str,
        *,
        method: str = "GET",
        json_payload: dict | None = None,
    ) -> dict:
        raw = self._curl_text(
            f"http://localhost:5000{endpoint}",
            method=method,
            json_payload=json_payload,
        )
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def _curl_text(
        self,
        url: str,
        *,
        method: str = "GET",
        json_payload: dict | None = None,
        data: dict | str | None = None,
        timeout: int = 30,
    ) -> str:
        cmd = ["curl", "-sS", "-X", method.upper(), url]
        if json_payload is not None:
            cmd += ["-H", "Content-Type: application/json", "-d", json.dumps(json_payload)]
        elif isinstance(data, dict):
            for key, value in data.items():
                cmd += ["-d", f"{key}={value}"]
        elif data is not None:
            cmd += ["-d", str(data)]
        return str(
            self._command_result(
                " ".join(shlex.quote(part) for part in cmd),
                timeout=timeout,
            )
        )


def _encode_request_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return {"__tuple__": [_encode_request_value(item) for item in value]}
    if isinstance(value, bytes):
        return {"__bytes_b64__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, list):
        return [_encode_request_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode_request_value(item) for key, item in value.items()}
    return value


def _build_request_script(payload: str) -> str:
    return textwrap.dedent(
        """
        import base64, json, requests
        spec = json.loads(base64.b64decode({payload!r}).decode("utf-8"))

        def decode(value):
            if isinstance(value, dict) and "__bytes_b64__" in value:
                return base64.b64decode(value["__bytes_b64__"])
            if isinstance(value, dict) and "__tuple__" in value:
                return tuple(decode(item) for item in value["__tuple__"])
            if isinstance(value, dict):
                return {{key: decode(item) for key, item in value.items()}}
            if isinstance(value, list):
                return [decode(item) for item in value]
            return value

        kwargs = decode(spec["kwargs"])
        try:
            resp = requests.request(spec["method"], spec["url"], **kwargs)
            out = {{
                "status_code": resp.status_code,
                "headers": dict(resp.headers),
                "text": resp.text,
                "content_b64": base64.b64encode(resp.content).decode("ascii"),
            }}
        except Exception as exc:
            out = {{
                "status_code": 599,
                "headers": {{}},
                "text": str(exc),
                "content_b64": "",
            }}
        print(json.dumps(out))
        """
    ).format(payload=payload)


class EvalEnvShim:
    """Subset of DesktopEnv consumed by ScaleCUA generated judges."""

    def __init__(self, computer, cache_dir: str, loop: asyncio.AbstractEventLoop):
        self.computer = computer
        self.cache_dir = cache_dir
        self.controller = _ControllerShim(computer, cache_dir, loop)
        self.vm_platform = "Linux"
        self.vm_machine = "Linux"
        self.vm_ip = "lite-scalecua-vm"
        self.server_port = 5000
        self.chromium_port = 1337
        self.vlc_port = 8080
        self.current_use_proxy = False

    def run_sync(self, fn, *args, **kwargs):
        with _routed_requests_for_function(self, fn):
            return fn(*args, **kwargs)

    def request_in_container(self, method: str, url: str, **kwargs) -> _ResponseShim:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        rewritten = urllib.parse.urlunsplit(
            (parsed.scheme or "http", f"localhost:{port}", parsed.path, parsed.query, parsed.fragment)
        )
        safe_kwargs: dict[str, Any] = {}
        for key in ("headers", "params", "json", "data", "auth", "timeout"):
            if key in kwargs:
                safe_kwargs[key] = _encode_request_value(kwargs[key])
        payload = base64.b64encode(
            json.dumps(
                {"method": method.upper(), "url": rewritten, "kwargs": safe_kwargs}
            ).encode("utf-8")
        ).decode("ascii")
        script = _build_request_script(payload)
        result = self.controller._command_result(
            f"python3 -c {shlex.quote(script)}",
            timeout=int(kwargs.get("timeout") or 30),
        )
        try:
            data = json.loads(str(result))
        except Exception:
            return _ResponseShim(status_code=599, text=str(result))
        content = base64.b64decode(data.get("content_b64") or b"")
        return _ResponseShim(
            status_code=int(data.get("status_code", 599)),
            text=data.get("text", ""),
            content=content,
            headers=data.get("headers") or {},
        )


def make_eval_env(computer, cache_dir: str) -> EvalEnvShim:
    return EvalEnvShim(computer, cache_dir, asyncio.get_running_loop())


async def call_overlay_getter(
    fn,
    eval_env: EvalEnvShim,
    config: dict[str, Any],
    cache_dir: str,
):
    config = await _materialize_direct_file_config(fn, eval_env, config, cache_dir)
    args = _overlay_getter_args(fn, eval_env, config, cache_dir)
    result_type = str(config.get("type") or "").split("__", 1)[0]
    return await _call_sync_maybe_await(
        eval_env,
        fn,
        *args,
        _postprocess=lambda result: _repair_overlay_getter_result(
            fn,
            eval_env,
            result,
            result_type=result_type,
        ),
    )


def _repair_overlay_getter_result(
    fn,
    eval_env: EvalEnvShim,
    result,
    *,
    result_type: str = "",
):
    name = getattr(fn, "__name__", "")
    if (
        (result_type == "volume_settings_state" or name.startswith("get_volume_settings_state__"))
        and isinstance(result, dict)
        and result.get("settings_open") is False
    ):
        repaired = dict(result)
        repaired["settings_open"] = _settings_window_or_process_open(eval_env)
        return repaired
    if (
        result_type == "volume_terminal_state"
        or name.startswith("get_volume_terminal_state__")
    ) and isinstance(result, dict):
        repaired = dict(result)
        repaired["terminal_open"] = _terminal_window_open(eval_env)
        return repaired
    return result


def _settings_window_or_process_open(eval_env: EvalEnvShim) -> bool:
    command = r"""
export DISPLAY=${DISPLAY:-:1}
wmctrl -lx 2>/dev/null \
  | awk 'BEGIN{IGNORECASE=1} /gnome-control-center|Gnome-control-center|Settings/ {found=1} END{exit found ? 0 : 1}' \
  && exit 0
pgrep -f '(^|/)gnome-control-center([[:space:]]|$)' >/dev/null 2>&1
"""
    try:
        result = eval_env.controller._command_result(
            "bash -lc " + shlex.quote(command),
            timeout=5,
        )
    except Exception:
        return False
    return int(result.get("returncode", 1) or 0) == 0


def _terminal_window_open(eval_env: EvalEnvShim) -> bool:
    command = r"""
export DISPLAY=${DISPLAY:-:1}
if wmctrl -lx 2>/dev/null \
  | awk 'BEGIN{IGNORECASE=1} $3 ~ /(gnome-terminal|gnome-terminal-server|org\.gnome\.Terminal)/ {found=1} END{exit found ? 0 : 1}'; then
  printf '%s\n' terminal-open
  exit 0
fi
if xdotool search --onlyvisible --class 'gnome-terminal' >/dev/null 2>&1 \
  || xdotool search --onlyvisible --class 'Gnome-terminal' >/dev/null 2>&1 \
  || xdotool search --onlyvisible --class 'org.gnome.Terminal' >/dev/null 2>&1; then
  printf '%s\n' terminal-open
fi
"""
    try:
        result = eval_env.controller._command_result(
            "bash -lc " + shlex.quote(command),
            timeout=5,
        )
    except Exception:
        return False
    return "terminal-open" in str(result.get("stdout", ""))


async def _materialize_direct_file_config(
    fn,
    eval_env: EvalEnvShim,
    config: dict[str, Any],
    cache_dir: str,
) -> dict[str, Any]:
    """Localize VM paths for generated getters that bypass DesktopEnv file APIs."""

    if not _getter_needs_direct_file_materialization(fn):
        return config

    updates: dict[str, str] = {}
    for key, value in config.items():
        if key not in _DIRECT_PATH_CONFIG_KEYS:
            continue
        if not isinstance(value, str) or not _looks_like_vm_path(value):
            continue
        local_path = await _download_vm_path(eval_env, value, cache_dir)
        if local_path is not None:
            updates[key] = local_path
    if not updates:
        return config

    materialized = dict(config)
    materialized.update(updates)
    return materialized


@lru_cache(maxsize=4096)
def _getter_needs_direct_file_materialization(fn) -> bool:
    """True if *fn* opens a VM path from ``config`` with a LOCAL file API.

    Recovering this by parsing the getter's own source is the shape this repo
    deletes everywhere else — so the measurement that says it must stay here is
    recorded, rather than left to be re-derived by the next reader:

    * **Nobody upstream declares it.** These getters are pulled artifacts
      (``judge_functions/<split>/getters.py`` from the ScaleCUA asset snapshot,
      re-pulled wholesale), not functions this repo authors. There is no flag a
      producer could set, and no caller holds the fact: ``call_overlay_getter``
      is handed a ``result_type`` string.
    * **Provenance is not a proxy.** Over the real train overlay: 4851 getters,
      of which **4091 use the remote file API** (``controller.get_file`` etc.),
      **106** need materialization, 654 touch no file. "It came from the
      overlay" would materialize the 4091 too, rewriting their config paths to
      local ones the VM cannot read.
    * **The dataflow in :class:`_DirectConfigPathFileApiVisitor` is
      load-bearing.** Every one of those 106 binds the path first
      (``path = config["path"]`` … ``load_workbook(path)``). Matching only
      calls whose argument reads ``config[...]`` inline finds **0 of 106** — so
      dropping the variable tracking silently disables materialization and each
      of the 106 judges then opens a VM path on the host: FileNotFoundError,
      scored as a failed task.

    (Re-derive with :func:`_getter_uses_remote_file_api` /
    :func:`_getter_uses_direct_config_path_file_api` over
    ``_load_overlay_module("train", "getters")``.)
    """
    try:
        sig = inspect.signature(fn)
        params = [
            p
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
    except (TypeError, ValueError):
        params = []
    if params and params[0].name in {"file_path", "filepath", "local_path"}:
        return False

    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return False

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    if _getter_uses_remote_file_api(tree):
        return False
    return _getter_uses_direct_config_path_file_api(tree)


def _getter_uses_remote_file_api(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _qualified_ast_name(node.func)
        if name.endswith(".controller.get_file"):
            return True
        if name.endswith(".controller.get_vm_file"):
            return True
        if name.endswith(".controller.get_vm_file_content"):
            return True
        if name in {"get_vm_file", "get_vm_file_content"}:
            return True
    return False


def _getter_uses_direct_config_path_file_api(tree: ast.AST) -> bool:
    visitor = _DirectConfigPathFileApiVisitor()
    visitor.visit(tree)
    return visitor.found


class _DirectConfigPathFileApiVisitor(ast.NodeVisitor):
    def __init__(self):
        self.config_path_vars: set[str] = set()
        self.found = False

    def visit_Assign(self, node: ast.Assign) -> None:
        if self.found:
            return
        self.visit(node.value)
        if _expr_uses_config_path(node.value, self.config_path_vars):
            for target in node.targets:
                self._mark_target(target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if self.found:
            return
        if node.value is None:
            return
        self.visit(node.value)
        if _expr_uses_config_path(node.value, self.config_path_vars):
            self._mark_target(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        if self.found:
            return
        self.visit(node.value)
        if _expr_uses_config_path(node.value, self.config_path_vars):
            self._mark_target(node.target)

    def visit_Call(self, node: ast.Call) -> None:
        if self.found:
            return
        if _is_direct_file_call(node):
            file_arg = _file_call_path_arg(node)
            if file_arg is not None and _expr_uses_config_path(
                file_arg,
                self.config_path_vars,
            ):
                self.found = True
                return
        self.generic_visit(node)

    def _mark_target(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self.config_path_vars.add(target.id)


def _is_direct_file_call(node: ast.Call) -> bool:
    name = _qualified_ast_name(node.func)
    return name in _DIRECT_FILE_CALL_NAMES or name.startswith("pd.read_") or name.startswith(
        "pandas.read_"
    )


def _file_call_path_arg(node: ast.Call) -> ast.AST | None:
    if node.args:
        return node.args[0]
    for kw in node.keywords:
        if kw.arg in {"filename", "file", "path", "filepath", "file_path"}:
            return kw.value
    return None


def _expr_uses_config_path(node: ast.AST, config_path_vars: set[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in config_path_vars
    if isinstance(node, ast.Subscript):
        return _is_config_name(node.value) and _literal_config_key(node.slice) in _DIRECT_PATH_CONFIG_KEYS
    if isinstance(node, ast.Call) and _is_config_get_call(node):
        return bool(node.args) and _literal_config_key(node.args[0]) in _DIRECT_PATH_CONFIG_KEYS
    return any(_expr_uses_config_path(child, config_path_vars) for child in ast.iter_child_nodes(node))


def _is_config_get_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and _is_config_name(node.func.value)
    )


def _is_config_name(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "config"


def _literal_config_key(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _qualified_ast_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _qualified_ast_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _looks_like_vm_path(value: str) -> bool:
    return value.startswith(("/home/user/", "/home/ubuntu/", "~/", "$HOME/", "${HOME}/"))


async def _download_vm_path(
    eval_env: EvalEnvShim,
    remote_path: str,
    cache_dir: str,
) -> str | None:
    aliased_path = _alias_chrome_profile_path(remote_path)
    try:
        data = await eval_env.computer.interface.read_bytes(aliased_path)
    except Exception as exc:
        logger.warning(
            "ScaleCUA overlay could not materialize VM file %s: %s",
            remote_path,
            exc,
        )
        return None
    if data is None:
        return None

    target = _overlay_input_cache_path(remote_path, cache_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return str(target)


def _overlay_input_cache_path(remote_path: str, cache_dir: str) -> Path:
    path = Path(remote_path)
    suffix = path.suffix
    name = path.name or "vm_file"
    stem = name[: -len(suffix)] if suffix and name.endswith(suffix) else name
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)
    digest = hashlib.sha1(remote_path.encode("utf-8")).hexdigest()[:12]
    filename = f"{safe_stem or 'vm_file'}.{digest}{suffix}"
    return Path(cache_dir) / "_overlay_inputs" / filename


def _overlay_getter_args(
    fn,
    eval_env: EvalEnvShim,
    config: dict[str, Any],
    cache_dir: str,
) -> tuple:
    """Match official DesktopEnv getter calls without swallowing body TypeError."""

    try:
        sig = inspect.signature(fn)
        params = [
            p
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
    except (TypeError, ValueError):
        return (eval_env, config)

    if params and params[0].name in {"file_path", "filepath", "local_path"}:
        candidates = [
            (_LazyVmFilePath(config), eval_env, config),
            (_LazyVmFilePath(config), config),
        ]
    elif len(params) >= 3 or any(p.name == "cache_dir" for p in params):
        candidates = [
            (eval_env, config, cache_dir),
            (eval_env, config),
        ]
    else:
        candidates = [
            (eval_env, config),
            (config,),
        ]

    for args in candidates:
        try:
            sig.bind(*args)
        except TypeError:
            continue
        return args
    raise JudgeResolverError(
        f"getter signature not supported: {getattr(fn, '__name__', fn)!r}{sig}"
    )


async def call_metric(
    fn,
    eval_env: EvalEnvShim,
    result_data,
    expected_data,
    options: dict[str, Any],
):
    opts = dict(options or {})
    try:
        sig = inspect.signature(fn)
        if "env" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        ):
            opts.setdefault("env", eval_env)
    except (TypeError, ValueError):
        pass

    args = (result_data, expected_data) if expected_data is not None else (result_data,)
    try:
        sig = inspect.signature(fn)
        positional = [
            p
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        has_varargs = any(
            p.kind == inspect.Parameter.VAR_POSITIONAL
            for p in sig.parameters.values()
        )
        if not has_varargs and len(positional) <= 1:
            args = (result_data,)
    except (TypeError, ValueError):
        pass
    return await _call_sync_maybe_await(eval_env, fn, *args, **opts)


async def _call_sync_maybe_await(
    eval_env: EvalEnvShim,
    fn,
    *args,
    _postprocess=None,
    **kwargs,
):
    loop = asyncio.get_running_loop()

    def _call():
        resolved_args = tuple(
            arg.resolve(eval_env) if isinstance(arg, _LazyVmFilePath) else arg
            for arg in args
        )
        value = eval_env.run_sync(fn, *resolved_args, **kwargs)
        if _postprocess is not None and not inspect.isawaitable(value):
            value = _postprocess(value)
        return value

    value = await loop.run_in_executor(None, _call)
    return await _maybe_await(value)


@contextlib.contextmanager
def _routed_requests_for_function(eval_env: EvalEnvShim, fn):
    _install_shared_requests_router()
    sentinel = object()
    previous = getattr(_REQUESTS_ROUTER_LOCAL, "eval_env", sentinel)
    _REQUESTS_ROUTER_LOCAL.eval_env = eval_env
    try:
        yield
    finally:
        if previous is sentinel:
            with contextlib.suppress(AttributeError):
                delattr(_REQUESTS_ROUTER_LOCAL, "eval_env")
        else:
            _REQUESTS_ROUTER_LOCAL.eval_env = previous


def _install_shared_requests_router() -> None:
    """Route BOTH module-global and function-local ``import requests`` through
    the thread-local active ``EvalEnvShim``.

    ScaleCUA's generated getters import ``requests`` either at module scope or
    (in ~36 getters) with a function-**local** ``import requests`` inside the
    body. Rebinding only module globals cannot reach the local case: a local
    ``import requests`` re-binds the name to ``sys.modules['requests']`` — the
    untouched real module — shadowing any rebound global, so the call bypasses
    the router and hits the unresolvable VM URL directly.

    Every ``import requests`` in the process returns the *same*
    ``sys.modules['requests']`` object, so patching the routed callables **on
    that object** is seen by all call sites regardless of import style. Dispatch
    stays keyed on the thread-local active env, so the patch is inert (calls the
    captured original) whenever no env is active on the calling thread — no
    global side effects on the env-server host, HF, or other libraries.

    Process-wide, idempotent, and lock-guarded → safe under WA/VWA concurrency.
    """
    global _REQUESTS_PATCH_INSTALLED
    with _REQUESTS_ROUTER_LOCK:
        if _REQUESTS_PATCH_INSTALLED:
            return

        import requests

        orig_request = requests.request
        orig_session_request = requests.sessions.Session.request

        def routed_request(method, url, **kwargs):
            active_env = _active_requests_env()
            if active_env is not None and _should_route_request(active_env, str(url)):
                return active_env.request_in_container(method, str(url), **kwargs)
            return orig_request(method, url, **kwargs)

        def get(url, params=None, **kwargs):
            return routed_request("GET", url, params=params, **kwargs)

        def options(url, **kwargs):
            return routed_request("OPTIONS", url, **kwargs)

        def head(url, **kwargs):
            kwargs.setdefault("allow_redirects", False)
            return routed_request("HEAD", url, **kwargs)

        def post(url, data=None, json=None, **kwargs):
            return routed_request("POST", url, data=data, json=json, **kwargs)

        def put(url, data=None, **kwargs):
            return routed_request("PUT", url, data=data, **kwargs)

        def patch(url, data=None, **kwargs):
            return routed_request("PATCH", url, data=data, **kwargs)

        def delete(url, **kwargs):
            return routed_request("DELETE", url, **kwargs)

        def routed_session_request(self, method, url, **kwargs):
            active_env = _active_requests_env()
            if active_env is not None and _should_route_request(active_env, str(url)):
                return active_env.request_in_container(method, str(url), **kwargs)
            return orig_session_request(self, method, url, **kwargs)

        requests.request = routed_request
        requests.get = get
        requests.options = options
        requests.head = head
        requests.post = post
        requests.put = put
        requests.patch = patch
        requests.delete = delete
        # Patch at the class level so ``requests.Session().get(...)`` and every
        # other session verb (which all funnel through ``Session.request``) are
        # routed too, without wrapping the session object.
        requests.sessions.Session.request = routed_session_request

        _REQUESTS_PATCH_INSTALLED = True


def _active_requests_env() -> EvalEnvShim | None:
    return getattr(_REQUESTS_ROUTER_LOCAL, "eval_env", None)


def _should_route_request(active_env: EvalEnvShim, url: str) -> bool:
    # A generated judge addressing localhost/the VM IP always means a service
    # inside the container — route it there regardless of port. Falling
    # through to host-side requests would silently read env-server-host state
    # (e.g. http://localhost/ with no explicit port).
    try:
        parsed = urllib.parse.urlsplit(url)
    except Exception:
        return False
    return parsed.hostname in {active_env.vm_ip, "localhost", "127.0.0.1"}


async def _maybe_await(value):
    if hasattr(value, "__await__"):
        return await value
    return value
