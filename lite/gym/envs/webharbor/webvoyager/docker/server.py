"""In-container FastAPI RPC server for cua-lite/webharbor.webvoyager.

The upstream WebVoyager runner is a single Selenium script that mixes browser
lifecycle, model calls, action execution, screenshots, and logging. In cua-lite
SINGLETON the model lives outside the env, so this server keeps only the browser
task lifecycle inside the shared container:

  GET  /healthz  -> backend health and active session counts
  POST /reset    -> create a Selenium Chrome session, open the task URL
  POST /step     -> execute a batch of Lite browser coordinate actions
  POST /close    -> quit the Selenium session

The host-side env should be a thin JSON client. Payloads are plain primitives:
screenshot_b64 strings, flags, and small metadata dicts.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

try:
    from selenium import webdriver
    from selenium.common.exceptions import TimeoutException
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
except ModuleNotFoundError:  # Host-side unit tests import pure helpers only.
    class _MissingKeys:
        ALT = "alt"
        ARROW_DOWN = "arrowdown"
        ARROW_LEFT = "arrowleft"
        ARROW_RIGHT = "arrowright"
        ARROW_UP = "arrowup"
        BACKSPACE = "backspace"
        COMMAND = "cmd"
        CONTROL = "ctrl"
        DELETE = "delete"
        END = "end"
        ENTER = "enter"
        ESCAPE = "escape"
        HOME = "home"
        PAGE_DOWN = "pagedown"
        PAGE_UP = "pageup"
        SHIFT = "shift"
        SPACE = "space"
        TAB = "tab"

    webdriver = None  # type: ignore[assignment]
    TimeoutException = RuntimeError  # type: ignore[assignment]
    Service = None  # type: ignore[assignment]
    ActionChains = None  # type: ignore[assignment]
    By = None  # type: ignore[assignment]
    Keys = _MissingKeys  # type: ignore[assignment]

try:
    from utils import get_web_element_rect
except ImportError:  # Host-side unit tests do not have the container utils path.
    def get_web_element_rect(*_args: Any, **_kwargs: Any) -> tuple[list[Any], list[Any], str]:
        raise RuntimeError("webharbor docker utils module is unavailable")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("webharbor.webvoyager-server")

_CAPTURE_CURSOR_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAACgAAAA8CAYAAAAUufjgAAAFNklEQVR4nM3ZX0xTVxzA8e+5LVNIFxK53YNtjS7sYWabMTxhtvCiWdXGBUURcNnjMvcAj0tU/G8EAwjCeHBP4PayPSgSgpl/XgjZonGG7E+WrGStYTNLmtRIKfTPPXvA2/TPRXvb29JfcpLb29/vnE9P77m3txcppZBS2qnQUAAlmUyOSSlt640xDCmlTUopE4nEdxWJ1IEVi0wHViQyG1hxSCNgRSHXAlYMMmuRyOvXr1cWMhu4efNmOTY2VjlII6DL5ZLj4+OVgcwGulwu6XK5pNvtrgxkNtDj8aTali1b5I0bN9YVmfMjQQiR8frEiRMAdHR0AGCz2dqSySRSyk+FEMl1BwKcPHkSIQTt7e1lR+YF1JFA2ZF5AwFOnToFlBdpCgjQ3d2NEIK2trayIHOAiqK8tuj06dMIITh69GjJkaZnMB0JlBxZMBDgzJkzQGmRRQEBzp49ixCC1tbWkiCLBupIoCRIS4AA586dQwjBkSNHLEVmAIUQea3iteL8+fMAliItm0E9Lly4gBCCw4cPW4K0HKgjAUuQJQECXLx4ESEELS0tRSFLBtSRQFHIkgIBLl26BBSOLOhabDYuX76MEIJDhw6ZRpZ8BrORBw8eNIUsG1BHAqaQZQUC9PT0APkjyw4E6O3tRQhBc3Pza5HrAtSRwGuRlq3i6upqbt26hcPhKKh+LWSORghRUFteXmZqaqpgXBZyXP9zwDSwrq6OhoYGw/cmJiaQUlqKNHUMqqrK0NAQ4XCY48eP57wfCAR4/PgxDQ0NANy9e3diz5493+fp+hV4kr0zb6CqqgwODuLxePB4PNTX1+P3+3Pybt68mQI2NTU1ulyu7oWFhURWWhWQPZAAaoHn6Tvz+oqdTmcKp0dzc7Nh7uzsLKFQaFVRVeWcn5/fDvyW1Z4Avxi0DJwhUFGUjGaEA9i9ezcOhyMnX9M0bt++ncqz2WxfZI9hJl45g6qqcvXqVdxuNwCapiUjkchzgJqaGrxer+EsTk1NkUwmdeBHL168eM9yoKqqDAwMZOCGh4eHFxYWevVcn89nCAyFQszMzKT63LBhw5eWAlVVpb+/PwfX2dnZ43Q6RzRNiwBs27aNnTt3GiLTv2a73d4hpSzoDJ4DVFWVvr4+I1wv8O+mTZueLy8vf6vnHzhwwBA4NzdHMBjUP/Sb8Xj8s6KBQgiuXLlihOsD/tHzpJQj+vauXbuoq6vLAQJMTk6m+rbb7Z9bAtRXq6ZpyWvXro10dnb2A0/T8xwOx9zS0tJPLwfG5/PlrGZFUbh37x7RaFTv+/1oNNpUFFAPHdfV1dUPBI1yhBBD+rbX68Vut+fM4tLSEg8ePEjVVFVVmT7l5ADTcINAYK3C6urqH+LxeAhWj9vGxkbDY3F6ejpVY7PZPjG7WDKAabghYP5VhUKIeCwW+0Z/vX//fkOg3+8nEEh9zo0rKysfFgSUUibScLkXWaNiRRmRUiYBduzYgcfjMUTqq/nlOG8VAoxPT09/1dXV9XW+OICampqni4uLP8LqAjM65SiKwtatW1M1Qoj/zAJlLBZr37dv3yjwp5liAEVRBvTtvXv3ZvxWVBSFY8eOpV/Ho5FIZNbsGEVHOBz+WX9UpmmafPTokbxz544MBoMZj9FisVh/2XEADx8+fHtxcTEkXxHPnj2bk1JuXBcgQGtrq8/v9z/JhiUSiZX79+9P1tfXby+kXyvvMd8APm5paXnX6/W+U1tbuzEQCIRGR0d/9/v9fwAzgOkbFqtvgt8APgDcrN5OhIG/gL8L7fB/g30TfzONgi4AAAAASUVORK5CYII="
)
_CAPTURE_CURSOR_SRC = f"data:image/png;base64,{_CAPTURE_CURSOR_B64}"
_CAPTURE_CURSOR_WIDTH = 16
_CAPTURE_CURSOR_HEIGHT = 24

# Keep in sync with host-side ``MODEL_ACTION_ERROR_TYPES``. This file runs as an
# in-container script and must not import the host ``lite.gym`` package.
_MODEL_ACTION_ERROR_TYPES = (ValueError, TypeError, IndexError, KeyError)
_DEFAULT_MODEL_DURATION_CAP_SECONDS = 30.0
_MODEL_DURATION_CAPS_SECONDS = {
    "wait": 30.0,
    "hold_key": 5.0,
    "long_press": 5.0,
    "swipe": 5.0,
    "drag": 5.0,
}


def _coerce_bounded_seconds(
    value: Any,
    *,
    label: str,
    minimum: float = 0.0,
    maximum: float = _DEFAULT_MODEL_DURATION_CAP_SECONDS,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number, got bool")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise ValueError(f"{label} must be a finite number")
        try:
            number = float(raw)
        except ValueError as e:
            raise ValueError(f"{label} must be a finite number") from e
    else:
        raise ValueError(f"{label} must be a finite number, got {type(value).__name__}")
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if number < minimum:
        raise ValueError(f"{label} must be >= {minimum:g}")
    if number > maximum:
        raise ValueError(f"{label} must be <= {maximum:g}")
    return number


def _coerce_model_duration(
    value: Any,
    *,
    action_name: str,
    field: str = "duration",
    minimum: float = 0.0,
) -> float:
    maximum = _MODEL_DURATION_CAPS_SECONDS.get(
        action_name,
        _DEFAULT_MODEL_DURATION_CAP_SECONDS,
    )
    return _coerce_bounded_seconds(
        value,
        label=f"{action_name}.{field}",
        minimum=minimum,
        maximum=maximum,
    )


def _coerce_cursor_bool(value: Any, *, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"cursor must be boolean-like, got {value!r}")
    if isinstance(value, float):
        if value in (0.0, 1.0):
            return bool(value)
        raise ValueError(f"cursor must be boolean-like, got {value!r}")
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"", "0", "false", "no", "off"}:
            return False
    raise ValueError(f"cursor must be boolean-like, got {value!r}")


def _read_flat_action(action: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    name = str(action["name"])
    args = action["arguments"]
    if not isinstance(args, dict):
        raise TypeError(f"Lite tool_call.arguments must be a dict, got {type(args).__name__}")
    return name, args


def _append_action_error(
    action_errors: list[dict[str, Any]],
    *,
    index: int,
    kind: str,
    name: str,
    error: Any,
    action: Any = None,
    message: str | None = None,
) -> None:
    detail = str(error)
    record = {
        "index": index,
        "name": name,
        "error": detail,
        "message": message or f"{name}: {detail}",
        "kind": kind,
    }
    if isinstance(action, dict) and action.get("call_id"):
        record["call_id"] = action["call_id"]
    if action is not None:
        record["action"] = action
    action_errors.append(record)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # _startup_impl is defined later in the module; name lookup happens at call
    # time (startup), after the module is fully loaded.
    await _startup_impl()
    yield


app = FastAPI(title="cua-lite webharbor.webvoyager in-container server", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Config. The host forwards yaml/env choices into the container via docker run
# -e WEBHARBOR_WEBVOYAGER_*; literal defaults here are fallbacks only.
# ---------------------------------------------------------------------------
_CHROME_BIN = os.environ.get("CHROME_BIN", "/usr/bin/chromium")
_CHROMEDRIVER_BIN = os.environ.get("CHROMEDRIVER_BIN", "/usr/bin/chromedriver")
_HEADLESS = os.environ.get("WEBHARBOR_WEBVOYAGER_HEADLESS", "1") not in {"0", "false", "False"}
_WINDOW_W = int(os.environ.get("WEBHARBOR_WEBVOYAGER_VIEWPORT_W", "1280"))
_WINDOW_H = int(os.environ.get("WEBHARBOR_WEBVOYAGER_VIEWPORT_H", "720"))
# Cursor origin at session start, in the canonical [0, 1000] normalized space:
# the viewport CENTRE. A real desktop warps the pointer to screen centre at
# session start and the browser inherits it, so the turn-0 frame must show the
# cursor there. Seeding it (rather than leaving last_cursor None) is what keeps
# the FIRST observation shaped like every later one — a None seed made
# _show_capture_cursor return False and shipped a cursor-less turn-0 frame.
_CURSOR_ORIGIN_NORM: tuple[int, int] = (500, 500)
_PAGE_LOAD_TIMEOUT_S = float(os.environ.get("WEBHARBOR_WEBVOYAGER_PAGE_LOAD_TIMEOUT_S", "15"))
_SCRIPT_TIMEOUT_S = float(os.environ.get("WEBHARBOR_WEBVOYAGER_SCRIPT_TIMEOUT_S", "10"))
_RESET_SETTLE_S = float(os.environ.get("WEBHARBOR_WEBVOYAGER_RESET_SETTLE_S", "5"))
_POST_ACTION_DELAY_S = float(os.environ.get("WEBHARBOR_WEBVOYAGER_POST_ACTION_DELAY_S", "0.5"))
_INSTANCE_TTL_S = float(os.environ.get("WEBHARBOR_WEBVOYAGER_INSTANCE_TTL_S", "600"))
_REAPER_INTERVAL_S = float(os.environ.get("WEBHARBOR_WEBVOYAGER_REAPER_INTERVAL_S", "30"))
_MAX_INSTANCES = int(
    os.environ.get("WEBHARBOR_WEBVOYAGER_MAX_INSTANCES")
    or os.environ.get("WEBHARBOR_WEBVOYAGER_INSTANCES", "16")
)
_FIX_BOX_COLOR = os.environ.get("WEBHARBOR_WEBVOYAGER_FIX_BOX_COLOR", "1") not in {"0", "false", "False"}
_BASE_DOWNLOAD_DIR = Path(os.environ.get("WEBHARBOR_WEBVOYAGER_DOWNLOAD_DIR", "/tmp/webharbor.webvoyager-downloads"))
_WEBSYN_REQUIRED = os.environ.get("WEBHARBOR_WEBVOYAGER_REQUIRE_WEBSYN", "1") not in {"0", "false", "False"}
_WEBSYN_CONTROL_URL = os.environ.get("WEBSYN_CONTROL_URL", "http://127.0.0.1:8101")


@dataclass
class _Instance:
    driver: webdriver.Chrome
    task_id: str
    instruction: str
    start_url: str
    download_dir: Path
    max_steps: int
    fix_box_color: bool = True
    use_som: bool = False
    step_count: int = 0
    answer: str | None = None
    last_active: float = field(default_factory=time.monotonic)
    last_element: Any | None = None
    # Last pointer px (viewport space); drag(start_coordinate=None) and the
    # capture overlay use it. _reset_sync always seeds it to the viewport centre
    # (_CURSOR_ORIGIN_NORM), so the None default never survives a live instance.
    last_cursor: tuple[int, int] | None = None
    web_eles: list[Any] = field(default_factory=list)
    download_files: set[str] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)


_instances: dict[str, _Instance] = {}
_instances_lock = threading.Lock()
_capacity_sem = threading.BoundedSemaphore(_MAX_INSTANCES)


# ---------------------------------------------------------------------------
# Selenium / WebVoyager helpers
# ---------------------------------------------------------------------------
def _chrome_options(download_dir: Path, width: int, height: int) -> webdriver.ChromeOptions:
    options = webdriver.ChromeOptions()
    options.binary_location = _CHROME_BIN
    options.page_load_strategy = "eager"

    if _HEADLESS:
        options.add_argument("--headless=new")
    options.add_argument(f"--window-size={width},{height}")
    options.add_argument("--force-device-scale-factor=1")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-setuid-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-background-networking")
    options.add_argument("--disable-features=Translate,AutomationControlled")
    options.add_argument("--no-first-run")
    options.add_argument("--no-default-browser-check")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    )
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_experimental_option(
        "prefs",
        {
            "download.default_directory": str(download_dir),
            "plugins.always_open_pdf_externally": True,
            "download.prompt_for_download": False,
            "profile.default_content_setting_values.notifications": 2,
        },
    )
    return options


def _new_driver(download_dir: Path, width: int, height: int) -> webdriver.Chrome:
    download_dir.mkdir(parents=True, exist_ok=True)
    service = Service(_CHROMEDRIVER_BIN)
    driver = webdriver.Chrome(service=service, options=_chrome_options(download_dir, width, height))
    driver.set_window_size(width, height)
    driver.set_page_load_timeout(_PAGE_LOAD_TIMEOUT_S)
    driver.set_script_timeout(_SCRIPT_TIMEOUT_S)
    return driver


def _safe_get(driver: webdriver.Chrome, url: str) -> None:
    try:
        driver.get(url)
    except TimeoutException:
        logger.warning("navigation timed out after %.1fs; stopping load: %s", _PAGE_LOAD_TIMEOUT_S, url)
        try:
            driver.execute_script("window.stop();")
        except Exception:
            pass


def _install_spacebar_guard(driver: webdriver.Chrome) -> None:
    try:
        driver.execute_script(
            """
            window.onkeydown = function(e) {
              if (e.keyCode == 32 && e.target.type != 'text' &&
                  e.target.type != 'textarea' && e.target.type != 'search') {
                e.preventDefault();
              }
            };
            """
        )
    except Exception:
        pass


def _initial_page_setup(driver: webdriver.Chrome, url: str) -> None:
    _safe_get(driver, url)
    try:
        driver.find_element(By.TAG_NAME, "body").click()
    except Exception:
        pass
    _install_spacebar_guard(driver)
    time.sleep(_RESET_SETTLE_S)


def _remove_rects(driver: webdriver.Chrome, rects: list[Any]) -> None:
    for rect in rects or []:
        try:
            driver.execute_script("arguments[0].remove()", rect)
        except Exception:
            pass


def _show_capture_cursor(driver: webdriver.Chrome, cursor: tuple[int, int] | None) -> bool:
    if cursor is None:
        return False
    x, y = cursor
    try:
        driver.execute_script(
            """
            let el = document.getElementById('cua-lite-capture-cursor');
            if (!el) {
                el = document.createElement('img');
                el.id = 'cua-lite-capture-cursor';
                el.style.position = 'fixed';
                el.style.zIndex = '2147483647';
                el.style.pointerEvents = 'none';
                el.style.transform = 'translate(0, 0)';
                document.documentElement.appendChild(el);
            }
            el.src = arguments[2];
            el.style.width = arguments[3] + 'px';
            el.style.height = arguments[4] + 'px';
            el.style.left = arguments[0] + 'px';
            el.style.top = arguments[1] + 'px';
            """,
            x,
            y,
            _CAPTURE_CURSOR_SRC,
            _CAPTURE_CURSOR_WIDTH,
            _CAPTURE_CURSOR_HEIGHT,
        )
        return True
    except Exception as e:
        logger.warning(
            "cursor capture overlay failed for %s: %s",
            _safe_attr(driver, "current_url"),
            e,
        )
        return False


def _hide_capture_cursor(driver: webdriver.Chrome) -> None:
    try:
        driver.execute_script(
            """
            const cursor = document.getElementById('cua-lite-capture-cursor');
            if (cursor) cursor.remove();
            """
        )
    except Exception:
        pass


def _capture_frame_b64(inst: _Instance, *, cursor: bool = True) -> str:
    """One frame (base64 PNG) of the CURRENT viewport, cursor overlay included.

    Owns the frame and nothing else. The per-action frames of a batch use it
    directly, deliberately NOT :func:`_screenshot_observation`: that one rewrites
    ``inst.web_eles``, so relabelling between two actions of one batch would make
    a later ``click_elem``/``fill``/``input`` resolve ``[N]`` against an element
    list the model never saw.
    """
    driver = inst.driver
    cursor_visible = _show_capture_cursor(driver, inst.last_cursor if cursor else None)
    try:
        png = driver.get_screenshot_as_png()
    finally:
        if cursor_visible:
            _hide_capture_cursor(driver)
    return base64.b64encode(png).decode("ascii")


def _screenshot_observation(inst: _Instance, *, cursor: bool = True) -> dict[str, Any]:
    """Mirror WebVoyager's observation step: mark interactive elements, capture
    the screenshot, then remove the overlays before the next action."""
    driver = inst.driver
    rects: list[Any] = []
    web_eles: list[Any] = []
    web_text = ""
    # Set-of-Marks overlay is opt-in (use_som). Coord mode (default) gets a PLAIN
    # screenshot — no [N] boxes drawn, no element list. SoM mode draws the dashed
    # [N] boxes + returns web_eles/web_text so index actions can resolve [N].
    if inst.use_som:
        try:
            rects, web_eles, web_text = get_web_element_rect(driver, fix_color=inst.fix_box_color)
        except Exception as e:
            logger.warning("get_web_element_rect failed for %s: %s", inst.task_id, e)

    # Store element list so index-based actions (click_elem, fill) can resolve [N].
    inst.web_eles = web_eles

    if web_eles:
        try:
            centers = _elem_centers_normalized(driver, web_eles)
            web_text = _enrich_web_text(web_text, centers)
        except Exception as e:
            logger.warning("coord enrichment failed for %s: %s", inst.task_id, e)

    # Enrich empty input labels with placeholder text (e.g. date fields in booking forms)
    if web_eles and web_text:
        try:
            placeholders = driver.execute_script(
                "return arguments[0].map(function(el) {"
                "  return (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') ?"
                "    (el.placeholder || el.getAttribute('placeholder') || '') : '';"
                "});",
                web_eles,
            )
            parts = web_text.split("\t")
            enriched = []
            for entry in parts:
                if not entry.startswith("["):
                    enriched.append(entry)
                    continue
                close = entry.find("]:")
                if close == -1:
                    enriched.append(entry)
                    continue
                try:
                    idx = int(entry[1:close])
                except ValueError:
                    enriched.append(entry)
                    continue
                if idx < len(placeholders) and placeholders[idx]:
                    ph = placeholders[idx]
                    entry = entry.replace('<input> ""', f'<input> "{ph}"', 1)
                    entry = entry.replace('<textarea> ""', f'<textarea> "{ph}"', 1)
                enriched.append(entry)
            web_text = "\t".join(enriched)
        except Exception as e:
            logger.warning("placeholder enrichment failed for %s: %s", inst.task_id, e)

    # Extract visible non-interactive text (article body, abstracts, recipe content, etc.)
    try:
        page_text = driver.execute_script("""
        (function() {
            // Priority 1: semantic article containers (BBC news, news articles, blogs)
            var articleSels = [
                'article', '[role="article"]', 'main article',
                '.article-body', '.story-body', '.post-body', '.entry-content',
                '[class*="article-body"]', '[class*="story-body"]'
            ];
            for (var s = 0; s < articleSels.length; s++) {
                var els = Array.from(document.querySelectorAll(articleSels[s]));
                for (var i = 0; i < els.length; i++) {
                    var t = (els[i].innerText || '').trim().replace(/\\s+/g, ' ');
                    if (t.length > 100) return t.slice(0, 2000);
                }
            }
            // Priority 2: academic abstract (arxiv, papers)
            var abstractSels = ['.abstract', '#abstract', '[class*="abstract"]', 'blockquote.abstract'];
            for (var s = 0; s < abstractSels.length; s++) {
                var els = Array.from(document.querySelectorAll(abstractSels[s]));
                for (var i = 0; i < els.length; i++) {
                    var t = (els[i].innerText || '').trim().replace(/\\s+/g, ' ');
                    if (t.length > 30) return t.slice(0, 2000);
                }
            }
            // Priority 3: visible <p> elements in viewport (misc content)
            var els = Array.from(document.querySelectorAll('p'));
            var seen = new Set();
            var parts = [];
            var total = 0;
            for (var i = 0; i < els.length; i++) {
                var el = els[i];
                var r = el.getBoundingClientRect();
                if (r.width === 0 || r.height === 0) continue;
                if (r.top > window.innerHeight || r.bottom < 0) continue;
                var parent = el.parentElement;
                var pTag = parent ? (parent.tagName || '').toLowerCase() : '';
                if (pTag === 'nav' || pTag === 'header' || pTag === 'footer') continue;
                var t = (el.innerText || el.textContent || '').trim().replace(/\\s+/g, ' ');
                if (!t || t.length < 15 || seen.has(t)) continue;
                seen.add(t);
                parts.push(t);
                total += t.length;
                if (total > 2000) break;
            }
            return parts.join('\\n');
        })()
        """)
        if page_text and page_text.strip():
            web_text = (web_text + "\n\n[PAGE TEXT]\n" + page_text[:2000]).strip() if web_text else ("[PAGE TEXT]\n" + page_text[:2000])
    except Exception as e:
        logger.warning("page text extraction failed for %s: %s", inst.task_id, e)

    # Inject step counter so model knows how many turns remain. ``step_count``
    # is the host's turn budget (0 at reset, then the host's count for the turn
    # just executed), so the NEXT turn the model gets is ``step_count + 1``.
    turns_used = inst.step_count
    turns_left = inst.max_steps - turns_used
    step_prefix = f"[Turn {turns_used + 1}/{inst.max_steps}]"
    if turns_left <= 4:
        step_prefix += f" LAST {turns_left} TURNS - call response(text) NOW with what you found."
    web_text = (step_prefix + "\n" + web_text).strip() if web_text else step_prefix

    try:
        screenshot_b64 = _capture_frame_b64(inst, cursor=cursor)
    finally:
        _remove_rects(driver, rects)
    return {
        "screenshot_b64": screenshot_b64,
        "web_text": web_text,
        "url": _safe_attr(driver, "current_url"),
        "title": _safe_attr(driver, "title"),
    }


def _safe_attr(driver: webdriver.Chrome, name: str) -> str:
    try:
        return str(getattr(driver, name))
    except Exception:
        return ""


def _norm_coord_to_viewport(driver: webdriver.Chrome, coordinate: list[int] | tuple[int, int]) -> tuple[int, int]:
    x_norm = float(coordinate[0])
    y_norm = float(coordinate[1])
    if not math.isfinite(x_norm) or not math.isfinite(y_norm):
        raise ValueError(f"coordinate values must be finite numbers: {coordinate!r}")
    width, height = driver.execute_script("return [window.innerWidth, window.innerHeight];")
    x = max(0, min(int(width) - 1, round(int(width) * x_norm / 1000.0)))
    y = max(0, min(int(height) - 1, round(int(height) * y_norm / 1000.0)))
    return x, y


def _elem_centers_normalized(driver: webdriver.Chrome, web_eles: list[Any]) -> list[tuple[int, int]]:
    """Return 0–1000 normalized (cx, cy) for each element in web_eles, one JS round-trip."""
    if not web_eles:
        return []
    width, height = driver.execute_script("return [window.innerWidth, window.innerHeight];")
    centers_px: list = driver.execute_script(
        "return arguments[0].map(function(el) {"
        "  var r = el.getBoundingClientRect();"
        "  return [r.left + r.width / 2, r.top + r.height / 2];"
        "});",
        web_eles,
    )
    result = []
    for cx_px, cy_px in centers_px:
        cx = max(0, min(999, round(cx_px / width * 1000)))
        cy = max(0, min(999, round(cy_px / height * 1000)))
        result.append((cx, cy))
    return result


def _enrich_web_text(web_text: str, centers: list[tuple[int, int]]) -> str:
    """Append @ (cx, cy) to each [N]: entry in web_text using the 0–1000 coordinate space.

    Input:  "[0]: <button> \"Search\";\t[1]: \"Buy now\";"
    Output: "[0]: <button> \"Search\" @ (412, 80);\t[1]: \"Buy now\" @ (300, 150);"

    Entries whose label index exceeds len(centers) are passed through unchanged.
    """
    if not web_text or not centers:
        return web_text
    parts = web_text.split("\t")
    result = []
    for entry in parts:
        if not entry.startswith("["):
            result.append(entry)
            continue
        close = entry.find("]:")
        if close == -1:
            result.append(entry)
            continue
        try:
            idx = int(entry[1:close])
        except ValueError:
            result.append(entry)
            continue
        if idx >= len(centers):
            result.append(entry)
            continue
        cx, cy = centers[idx]
        if entry.endswith(";"):
            result.append(f"{entry[:-1]} @ ({cx}, {cy});")
        else:
            result.append(f"{entry} @ ({cx}, {cy})")
    return "\t".join(result)


def _element_from_coordinate(driver: webdriver.Chrome, coordinate: list[int] | tuple[int, int]):
    x, y = _norm_coord_to_viewport(driver, coordinate)
    return driver.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", x, y)


def _element_center_viewport(driver: webdriver.Chrome, element: Any) -> tuple[int, int] | None:
    try:
        cx, cy = driver.execute_script(
            "const r = arguments[0].getBoundingClientRect();"
            "return [Math.round(r.left + r.width / 2), Math.round(r.top + r.height / 2)];",
            element,
        )
        return int(cx), int(cy)
    except Exception:
        return None


def _active_or_last_element(inst: _Instance):
    try:
        active = inst.driver.switch_to.active_element
        tag = (active.tag_name or "").lower()
        if tag not in {"html", "body"}:
            return active
    except Exception:
        pass
    return inst.last_element


def _exec_action_click(inst: _Instance, element: Any) -> None:
    driver = inst.driver
    driver.execute_script("arguments[0].setAttribute('target', '_self')", element)
    element.click()
    inst.last_element = element
    center = _element_center_viewport(driver, element)
    if center is not None:
        inst.last_cursor = center
    time.sleep(3)


def _exec_action_type(inst: _Instance, element: Any, text: str, *, press_enter: bool) -> str:
    driver = inst.driver
    warn_obs = ""

    try:
        ele_tag_name = (element.tag_name or "").lower()
        ele_type = element.get_attribute("type")
        if ((ele_tag_name not in {"input", "textarea"})
                or (ele_tag_name == "input" and ele_type not in {"text", "search", "password", "email", "tel"})):
            warn_obs = (
                "note: The web element you're trying to type may not be a textbox, "
                f"and its tag name is <{element.tag_name}>, type is {ele_type}."
            )
    except Exception:
        pass

    try:
        element.clear()
        element.send_keys(Keys.CONTROL + "a")
        element.send_keys(" ")
        element.send_keys(Keys.BACKSPACE)
    except Exception:
        pass

    actions = ActionChains(driver)
    actions.click(element).perform()
    actions.pause(1)
    _install_spacebar_guard(driver)
    actions.send_keys(text)
    actions.pause(2)
    if press_enter:
        actions.send_keys(Keys.ENTER)
    actions.perform()
    inst.last_element = element
    center = _element_center_viewport(driver, element)
    if center is not None:
        inst.last_cursor = center
    time.sleep(3 if not press_enter else 10)
    return warn_obs


_KEY_MAP = {
    "alt": Keys.ALT,
    "arrowdown": Keys.ARROW_DOWN,
    "arrowleft": Keys.ARROW_LEFT,
    "arrowright": Keys.ARROW_RIGHT,
    "arrowup": Keys.ARROW_UP,
    "backspace": Keys.BACKSPACE,
    "cmd": Keys.COMMAND,
    "ctrl": Keys.CONTROL,
    "control": Keys.CONTROL,
    "delete": Keys.DELETE,
    "end": Keys.END,
    "enter": Keys.ENTER,
    "esc": Keys.ESCAPE,
    "escape": Keys.ESCAPE,
    "home": Keys.HOME,
    "option": Keys.ALT,
    "pagedown": Keys.PAGE_DOWN,
    "pageup": Keys.PAGE_UP,
    "shift": Keys.SHIFT,
    "space": Keys.SPACE,
    "tab": Keys.TAB,
    "win": Keys.COMMAND,
}


def _selenium_key(key: str) -> str:
    return _KEY_MAP.get(str(key).lower(), str(key))


def _model_key_list(value: Any, *, action_name: str, allow_empty: bool = True) -> list[str]:
    label = f"{action_name}.keys"
    if isinstance(value, str):
        raise ValueError(f"{label} must be a list of strings, not a string")
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list of strings")
    keys = list(value)
    if not keys and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    for index, key in enumerate(keys):
        if not isinstance(key, str):
            raise ValueError(f"{label}[{index}] must be a string")
    return keys


def _download_delta(inst: _Instance) -> list[str]:
    try:
        current = {
            p.name for p in inst.download_dir.iterdir()
            if p.is_file() and not p.name.endswith(".crdownload")
        }
    except FileNotFoundError:
        current = set()
    delta = sorted(current - inst.download_files)
    inst.download_files = current
    return delta


def _close_instance(iid: str) -> bool:
    with _instances_lock:
        inst = _instances.pop(iid, None)
    if inst is None:
        return False
    with inst.lock:
        try:
            inst.driver.quit()
        except Exception as e:
            logger.warning("driver quit failed for %s: %s", iid, e)
        try:
            shutil.rmtree(inst.download_dir, ignore_errors=True)
        except Exception:
            pass
    try:
        _capacity_sem.release()
    except ValueError:
        pass
    return True


def _websyn_health() -> dict[str, Any]:
    if not _WEBSYN_REQUIRED:
        return {"required": False, "ok": True}
    url = _WEBSYN_CONTROL_URL.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            data = json.loads(r.read())
        return {
            "required": True,
            "ok": bool(data.get("ok")),
            "sites": data.get("sites", {}),
        }
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {
            "required": True,
            "ok": False,
            "error": str(e),
            "url": url,
        }


# ---------------------------------------------------------------------------
# Action translation
# ---------------------------------------------------------------------------
def _execute_action(inst: _Instance, name: str, args: dict[str, Any]) -> dict[str, Any]:
    driver = inst.driver
    out: dict[str, Any] = {"call": name, "args": args}

    def _elem_by_index(action_name: str) -> Any:
        raw = args.get("index", -1)
        index = int(raw[0] if isinstance(raw, list) else raw)
        n = len(inst.web_eles)
        if index < 0 or index >= n:
            raise ValueError(f"{action_name}: index {index} out of range (0-{n - 1})")
        return inst.web_eles[index]

    if name == "click":
        if "index" in args:
            element = _elem_by_index("click")
        else:
            coordinate = args.get("coordinate")
            if coordinate is None:
                raise ValueError("click requires coordinate or index")
            inst.last_cursor = _norm_coord_to_viewport(driver, coordinate)
            element = _element_from_coordinate(driver, coordinate)
            if element is None:
                raise ValueError(f"no element at coordinate {coordinate}")
        _exec_action_click(inst, element)

    elif name == "input":
        text = str(args.get("text", ""))
        warn = _exec_action_type(inst, _elem_by_index("input"), text, press_enter=False)
        if warn:
            out["warning"] = warn

    elif name == "type":
        text = str(args.get("text", ""))
        element = _active_or_last_element(inst)
        if element is None:
            # Nothing focused. This is not a corner: ``_initial_page_setup``
            # clicks ``<body>`` at reset, so the FIRST ``type`` of every episode
            # lands here. It therefore has to match the focused branch on all
            # three counts, not just deliver the keystrokes:
            #   * ``press_enter`` -- the host strips the canonical trailing
            #     newline in ``_to_container_action`` (``webvoyager/main.py``)
            #     and hands the submit over as this flag, so ignoring it loses
            #     the Enter with nothing left in ``text`` to carry it;
            #   * the spacebar guard -- a navigation wipes the reset-time
            #     handler, and without it a space typed at a body-focused page
            #     scrolls instead of typing;
            #   * the settle -- a submit navigates, so it needs the same longer
            #     wait ``_exec_action_type`` uses or the screenshot catches the
            #     page mid-navigation.
            press_enter = bool(args.get("press_enter", False))
            _install_spacebar_guard(driver)
            actions = ActionChains(driver).send_keys(text)
            if press_enter:
                actions.pause(2)
                actions.send_keys(Keys.ENTER)
            actions.perform()
            time.sleep(10 if press_enter else 3)
        else:
            # Absent ``press_enter`` means TYPE ONLY, and the error is
            # asymmetric: a missing Enter costs one turn, a spurious Enter
            # irreversibly submits a form or navigates away. The flag is this
            # container's own wire, not the model's -- canonical ``type`` takes
            # only ``text``, and the host derives the flag from a trailing
            # newline in ``_to_container_action`` (``webvoyager/main.py``). The
            # model spells Enter as that newline or as a separate
            # ``key(["enter"])``.
            warn = _exec_action_type(inst, element, text, press_enter=bool(args.get("press_enter", False)))
            if warn:
                out["warning"] = warn

    elif name == "scroll":
        if "down" in args or "pages" in args:
            down = bool(args.get("down", True))
            pages = abs(float(args.get("pages", 1.0) or 1.0))
            viewport_h = driver.execute_script(
                "return window.innerHeight || document.documentElement.clientHeight || 768;"
            )
            pixels = max(1.0, pages) * float(viewport_h or 768)
            driver.execute_script("window.scrollBy(0, arguments[0]);", pixels if down else -pixels)
        else:
            direction = str(args.get("direction", "down")).lower()
            amount = int(args.get("amount", 3) or 3)
            coordinate = args.get("coordinate")
            if coordinate is not None:
                try:
                    inst.last_cursor = _norm_coord_to_viewport(driver, coordinate)
                    element = _element_from_coordinate(driver, coordinate)
                    if element is not None:
                        driver.execute_script("arguments[0].focus();", element)
                        inst.last_element = element
                except Exception:
                    pass
            sign = -1 if direction in {"up", "left"} else 1
            axis = "x" if direction in {"left", "right"} else "y"
            pixels = max(1, amount) * 100
            if axis == "x":
                driver.execute_script("window.scrollBy(arguments[0], 0);", sign * pixels)
            else:
                driver.execute_script("window.scrollBy(0, arguments[0]);", sign * pixels)
        time.sleep(3)

    elif name == "key":
        keys = [
            _selenium_key(k)
            for k in _model_key_list(args.get("keys", []), action_name=name)
        ]
        if keys:
            actions = ActionChains(driver)
            for key in keys[:-1]:
                actions.key_down(key)
            actions.send_keys(keys[-1])
            for key in reversed(keys[:-1]):
                actions.key_up(key)
            actions.perform()
        time.sleep(1)

    elif name == "key_down":
        actions = ActionChains(driver)
        for key in _model_key_list(args.get("keys", []), action_name=name):
            actions.key_down(_selenium_key(key))
        actions.perform()

    elif name == "key_up":
        actions = ActionChains(driver)
        for key in _model_key_list(args.get("keys", []), action_name=name):
            actions.key_up(_selenium_key(key))
        actions.perform()

    elif name == "wait":
        time.sleep(
            _coerce_model_duration(
                args.get("duration", 1.0),
                action_name="wait",
            )
        )

    elif name == "screenshot":
        pass

    elif name in {"back", "goback", "go_back"}:
        driver.back()
        time.sleep(2)

    elif name in {"goto", "navigate"}:
        url = args.get("url")
        if not url:
            raise ValueError(f"{name} requires url")
        _safe_get(driver, str(url))
        time.sleep(2)

    elif name == "google":
        _safe_get(driver, "https://www.google.com/")
        time.sleep(2)

    elif name == "drag":
        start = args.get("start_coordinate")
        end = args.get("coordinate")
        if end is None:
            raise ValueError("drag requires coordinate")
        if start is None:
            if inst.last_cursor is None:
                raise ValueError("drag requires start_coordinate or a tracked cursor")
            start = inst.last_cursor
        sx, sy = _norm_coord_to_viewport(driver, start)
        ex, ey = _norm_coord_to_viewport(driver, end)
        ActionChains(driver).move_by_offset(sx, sy).click_and_hold().move_by_offset(ex - sx, ey - sy).release().perform()
        inst.last_cursor = (ex, ey)
        time.sleep(1)

    elif name == "mouse_move":
        coordinate = args.get("coordinate")
        if coordinate is None:
            raise ValueError("mouse_move requires coordinate")
        inst.last_cursor = _norm_coord_to_viewport(driver, coordinate)
        element = _element_from_coordinate(driver, coordinate)
        if element is not None:
            ActionChains(driver).move_to_element(element).perform()
            inst.last_element = element

    elif name == "click_elem":
        _exec_action_click(inst, _elem_by_index("click_elem"))

    elif name == "fill":
        text = str(args.get("text", ""))
        press_enter = bool(args.get("press_enter", False))
        warn = _exec_action_type(inst, _elem_by_index("fill"), text, press_enter=press_enter)
        if warn:
            out["warning"] = warn

    else:
        logger.warning("unknown webharbor.webvoyager action: %s(%s)", name, args)
        out["warning"] = f"unknown action {name}"

    return out


# ---------------------------------------------------------------------------
# Lifecycle background task
# ---------------------------------------------------------------------------
async def _idle_reaper_loop() -> None:
    while True:
        await asyncio.sleep(_REAPER_INTERVAL_S)
        now = time.monotonic()
        stale: list[str] = []
        with _instances_lock:
            for iid, inst in _instances.items():
                # Skip instances with an in-flight step (lock held): reaping one
                # mid-step would quit the driver out from under it (C2).
                if inst.lock.locked():
                    continue
                if now - inst.last_active > _INSTANCE_TTL_S:
                    stale.append(iid)
        for iid in stale:
            logger.warning("reaping idle webharbor.webvoyager instance %s", iid)
            await asyncio.to_thread(_close_instance, iid)


async def _startup_impl() -> None:
    _BASE_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.create_task(_idle_reaper_loop())
    logger.info("webharbor.webvoyager server up: max_instances=%d viewport=%dx%d chrome=%s",
        _MAX_INSTANCES, _WINDOW_W, _WINDOW_H, _CHROME_BIN,)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    # ``ok`` is RPC LIVENESS — if we answered this request, the server is up. We
    # deliberately do NOT fold in the WebSyn mirror health: a transient 3s mirror
    # blip would otherwise flip ok=False and make the host judge the whole env
    # down (host health() raises -> every rollout fails), when a mirror issue is
    # per-task and recoverable. Mirror health is surfaced separately under
    # ``webharbor`` for observability (C4).
    return {
        "ok": True,
        "instances": len(_instances),
        "max_instances": _MAX_INSTANCES,
        "viewport": [_WINDOW_W, _WINDOW_H],
        "webharbor": _websyn_health(),
    }


def _reset_sync(body: dict[str, Any]) -> dict[str, Any]:
    # Parse the body BEFORE acquiring a capacity permit, so a malformed body
    # raises without leaking the semaphore (it's released only on close / the
    # build except below).
    task_id = str(body.get("task_id") or body.get("upstream_id") or uuid.uuid4().hex)
    instruction = str(body.get("instruction") or body.get("ques") or "")
    # ``web`` is the upstream WebVoyager raw URL field; Lite hosts send ``start_url``.
    start_url = str(body.get("start_url") or body.get("web") or body.get("url") or "https://www.google.com/")
    max_steps = int(body.get("max_steps") or 15)
    width = int(body.get("window_width") or body.get("viewport", [_WINDOW_W, _WINDOW_H])[0])
    height = int(body.get("window_height") or body.get("viewport", [_WINDOW_W, _WINDOW_H])[1])
    fix_box_color = bool(body.get("fix_box_color", _FIX_BOX_COLOR))
    use_som = bool(body.get("use_som", False))
    cursor = _coerce_cursor_bool(body.get("cursor"), default=True)

    if not _capacity_sem.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="webharbor.webvoyager pool full")

    iid = uuid.uuid4().hex
    download_dir = _BASE_DOWNLOAD_DIR / iid
    driver = None
    try:
        driver = _new_driver(download_dir, width, height)
        _initial_page_setup(driver, start_url)
        inst = _Instance(
            driver=driver,
            task_id=task_id,
            instruction=instruction,
            start_url=start_url,
            download_dir=download_dir,
            max_steps=max_steps,
            fix_box_color=fix_box_color,
            use_som=use_som,
            download_files=set(),
            last_cursor=_norm_coord_to_viewport(driver, _CURSOR_ORIGIN_NORM),
        )
        obs = _screenshot_observation(inst, cursor=cursor)
    except Exception:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass
        shutil.rmtree(download_dir, ignore_errors=True)
        _capacity_sem.release()
        raise

    with _instances_lock:
        _instances[iid] = inst

    return {
        "instance_id": iid,
        "screenshot_b64": obs["screenshot_b64"],
        "instruction": instruction,
        "max_steps": max_steps,
        "web_text": obs.get("web_text", ""),
        "url": obs.get("url", ""),
        "title": obs.get("title", ""),
    }


@app.post("/reset")
async def reset(request: Request) -> dict[str, Any]:
    body = await request.json()
    try:
        return await asyncio.to_thread(_reset_sync, body)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("reset failed")
        raise HTTPException(status_code=500, detail=f"reset failed: {e}") from e


def _step_sync(body: dict[str, Any]) -> dict[str, Any]:
    iid = body["instance_id"]
    with _instances_lock:
        inst = _instances.get(iid)
    if inst is None:
        raise HTTPException(status_code=404, detail=f"unknown instance {iid}")

    cursor = _coerce_cursor_bool(body.get("cursor"), default=True)
    terminated = False
    executed: list[dict[str, Any]] = []
    # One result frame per EXECUTED action, in action order. The host ships the
    # whole batch here, so this loop -- not the host -- is the only place that
    # can see the page between two actions of one batch. These are plain frames:
    # the turn's SoM overlay / web_text / url still come from the single
    # observation taken after the loop, which also replaces the last frame.
    frames_b64: list[str] = []
    # ``terminate``/``response`` execute but never drive the page, so they break
    # out of the loop owing a frame. The post-loop observation becomes theirs by
    # EXTENDING rather than superseding -- otherwise it would overwrite the
    # previous action's frame and a terminating batch would report one frame for
    # two executed actions.
    finish_owes_frame = False
    errors: list[str] = []
    action_errors: list[dict[str, Any]] = []
    # ``actions`` is a list by the time it gets here: :class:`StepBody` is the
    # request model, so FastAPI answers a non-list with a 422 before this runs.
    # Element shape is NOT checked there on purpose -- a bad element is the
    # MODEL's arguments and belongs in ``action_errors``; the list-ness is the
    # HOST client's serialization and never was a model error.
    actions = body["actions"]

    with inst.lock:
        # Re-check membership now that we hold inst.lock: the reaper pops under
        # _instances_lock then blocks on inst.lock before quit(); if it already
        # reaped this iid, fail cleanly with 404 instead of acting on a quit
        # driver (-> 500). Closes the get-then-lock vs pop-then-lock race (C2).
        with _instances_lock:
            if _instances.get(iid) is not inst:
                raise HTTPException(status_code=404, detail=f"instance {iid} was reaped")
        inst.last_active = time.monotonic()
        for action_index, action in enumerate(actions):
            try:
                name, args = _read_flat_action(action)
            except _MODEL_ACTION_ERROR_TYPES as e:
                logger.warning("invalid action envelope for %s: %s", iid, e)
                name = str(action.get("name", "<invalid>")) if isinstance(action, dict) else "<invalid>"
                errors.append(f"{name}: {e}")
                _append_action_error(
                    action_errors,
                    index=action_index,
                    kind="model_action",
                    name=name,
                    error=e,
                    action=action,
                )
                continue

            if name == "terminate":
                terminated = True
                executed.append({"call": "terminate", "args": args})
                finish_owes_frame = True
                break

            if name == "response":
                inst.answer = str(args.get("text", ""))
                terminated = True
                executed.append({"call": "response", "args": args})
                finish_owes_frame = True
                break

            try:
                executed.append(_execute_action(inst, name, args))
            except _MODEL_ACTION_ERROR_TYPES as e:
                logger.warning("action failed for %s: %s(%s): %s", iid, name, args, e)
                errors.append(f"{name}: {e}")
                _append_action_error(
                    action_errors,
                    index=action_index,
                    kind="model_action",
                    name=name,
                    error=e,
                    action=action,
                )
            except Exception:
                logger.exception("backend action failed for %s: %s(%s)", iid, name, args)
                raise
            else:
                # Success only, and OUTSIDE the try: a driver failure while
                # capturing is not the action's fault and must not be logged as
                # one.
                frames_b64.append(_capture_frame_b64(inst, cursor=cursor))

        if not terminated:
            raw_post_action_delay = body.get("post_action_delay", _POST_ACTION_DELAY_S)
            if raw_post_action_delay is None:
                raw_post_action_delay = 0.0
            post_action_delay = _coerce_bounded_seconds(
                raw_post_action_delay,
                label="post_action_delay",
            )
            if post_action_delay > 0:
                time.sleep(post_action_delay)

        # The host owns the count and ships it (main.py step); it is the side
        # that also sees the turns whose calls were all rejected client-side
        # and never reach us. A private ``+= 1`` here could only ever fall
        # behind it -- which it did, under-reporting truncation and inflating
        # the ``[Turn n/max]`` banner by one turn per rejected turn.
        inst.step_count = int(body["step_count"])
        truncated = not terminated and inst.step_count >= inst.max_steps
        downloads = _download_delta(inst)
        obs = _screenshot_observation(inst, cursor=cursor)
        # This observation IS the last executed action's frame. Normally that
        # action already appended its own, so re-taking it after the settle
        # delay (with the SoM overlay) supersedes; when the batch broke on a finish action, or
        # when nothing executed at all, no frame is owed to anyone yet and this
        # one extends instead. Either way the count equals executed actions.
        if frames_b64 and not finish_owes_frame:
            frames_b64[-1] = obs["screenshot_b64"]
        else:
            frames_b64.append(obs["screenshot_b64"])
        # Heartbeat at step END too: a single step slower than the TTL must not
        # look idle to the reaper (C2). The reaper also skips locked instances.
        inst.last_active = time.monotonic()

    return {
        # ONE frame per executed action, in action order. The model-visible
        # frame is the last entry; the host owns that selection.
        "screenshots_b64": frames_b64,
        "reward": None,
        "terminated": terminated,
        "truncated": truncated,
        "executed": executed,
        "errors": errors,
        "action_errors": action_errors,
        "downloads": downloads,
        "web_text": obs.get("web_text", ""),
        "url": obs.get("url", ""),
        "title": obs.get("title", ""),
        "answer": inst.answer,
    }


class StepBody(BaseModel):
    """The ``POST /step`` request, typed so a malformed one cannot be blamed on
    the model.

    ``actions`` used to arrive as ``Any``: a non-list was recorded as an
    ``action_errors`` entry with ``kind="model_action"`` and returned **200**, so
    the host told the model *"your arguments were invalid"* about a body the host
    client itself serialized (``main.py`` builds
    ``actions_to_send: list[LiteToolCall]``), and nothing retried. There is no
    infra member of ``ActionErrorKind`` and adding one would extend a model-blame
    vocabulary to cover an infra fault — so the shape is made unrepresentable
    instead: FastAPI validates it and answers 422.

    ``extra="allow"`` keeps the remaining fields on their existing coercers
    (``cursor`` and ``post_action_delay`` accept boolean-like / numeric-like wire
    values by design); only the field whose type was load-bearing is declared.
    """

    model_config = ConfigDict(extra="allow")

    instance_id: str
    actions: list[Any] = Field(default_factory=list)


@app.post("/step")
async def step(body_model: StepBody) -> dict[str, Any]:
    body = body_model.model_dump()
    try:
        return await asyncio.to_thread(_step_sync, body)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("step failed")
        raise HTTPException(status_code=500, detail=f"step failed: {e}") from e


@app.post("/close")
async def close(request: Request) -> dict[str, Any]:
    body = await request.json()
    iid = body.get("instance_id", "")
    closed = await asyncio.to_thread(_close_instance, iid)
    return {"ok": True, "closed": closed}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("WEBHARBOR_WEBVOYAGER_RPC_PORT", "8000")))
