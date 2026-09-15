"""Local task catalog, status contract and owned loopback-only static server.

Preview: ``python -m examples.not_a_robot.local_tasks --port 8765``.
Only bundled files and verified private reference assets are served. No file
listing is exposed. This is not a production multi-user server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from .reference_assets import REFERENCE_HASHES, REFERENCE_ROOT

ASSET_ROOT = Path(__file__).with_name("local")
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/game.js": ("game.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/tasks.json": ("tasks.json", "application/json"),
    "/neal.js": ("neal.js", "text/javascript; charset=utf-8"),
    "/neal.css": ("neal.css", "text/css; charset=utf-8"),
    "/neal_grids.js": ("neal_grids.js", "text/javascript; charset=utf-8"),
    "/neal_boards.js": ("neal_boards.js", "text/javascript; charset=utf-8"),
    "/neal_forms.js": ("neal_forms.js", "text/javascript; charset=utf-8"),
    "/neal_replacements.js": ("neal_replacements.js", "text/javascript; charset=utf-8"),
    "/neal_ducks.js": ("neal_ducks.js", "text/javascript; charset=utf-8"),
    "/neal_camera.js": ("neal_camera.js", "text/javascript; charset=utf-8"),
}
ASSETS.update(
    {
        f"/reference_assets/{name}": (
            f"reference_assets/{name}",
            {".webp": "image/webp", ".png": "image/png", ".jpg": "image/jpeg"}[Path(name).suffix],
        )
        for name in REFERENCE_HASHES
    }
)
CATALOG = json.loads((ASSET_ROOT / "tasks.json").read_text())
LOCAL_TASKS = {task["id"]: task for task in CATALOG["tasks"]}


def task_reference(task_id: str, reference_instance: str = "default") -> dict | None:
    """Resolve one captured reference without changing another instance's rules."""
    task = LOCAL_TASKS[task_id]
    if reference_instance == "default":
        return task.get("reference")
    variants = task.get("reference_variants", {})
    if reference_instance not in variants:
        raise ValueError(f"Unknown reference_instance {reference_instance!r} for {task_id}")
    return variants[reference_instance]


@dataclass(frozen=True)
class LocalTaskState:
    """Evaluator-only status, with no target coordinates or solution values."""

    task_id: str
    label: str
    version: str
    seed: int
    status: Literal["in_progress", "success", "failure"]
    mistakes: int
    progress: int
    reason: str
    elapsed_ms: int
    reference_instance: str = "default"


class LocalTaskServer:
    """Own an ephemeral localhost listener and its serving thread."""

    def __init__(self, port: int = 0, *, require_reference: bool = False):
        if require_reference and not REFERENCE_ROOT.is_dir():
            raise FileNotFoundError(
                "Reference artwork is not imported. Run python -m "
                "examples.not_a_robot.reference_assets /path/to/reference.zip first."
            )
        assets = {
            path: ((ASSET_ROOT / filename).read_bytes(), mime)
            for path, (filename, mime) in ASSETS.items()
            if not filename.startswith("reference_assets/") or (ASSET_ROOT / filename).is_file()
        }
        for name, expected in REFERENCE_HASHES.items():
            asset = assets.get(f"/reference_assets/{name}")
            if asset is not None and hashlib.sha256(asset[0]).hexdigest() != expected:
                raise ValueError(f"Private reference asset hash mismatch: {name}")
        self.asset_hashes = {
            ASSETS[path][0]: hashlib.sha256(content).hexdigest()
            for path, (content, _) in assets.items()
        }

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                asset = assets.get(urlsplit(self.path).path)
                if asset is None:
                    self.send_error(404)
                    return
                content, mime = asset
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'none'; script-src 'self'; style-src 'self'; "
                    "connect-src 'self'; img-src 'self'; base-uri 'none'; "
                    "form-action 'none'; frame-ancestors 'none'",
                )
                self.end_headers()
                self.wfile.write(content)

            def log_message(self, format, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.origin = f"http://127.0.0.1:{self._server.server_port}"
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        self._thread.start()

    def close(self):
        """Stop only this server and release its bound port."""
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = LocalTaskServer(args.port)
    print(f"Visual Tasks: {server.origin}/ (Ctrl+C to stop)", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


if __name__ == "__main__":
    main()
