"""Append-only, single-writer archives for the Not a Robot experiment.

``root`` names a new attempt directory, not a shared run directory. Only a
verified ``manifest.json`` marks recording complete; task success is separate.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class EventRecorder:
    """Preserve every emitted event and screenshot in an immutable attempt."""

    def __init__(self, root: Path, metadata: dict):
        self.metadata = json.loads(json.dumps(metadata, allow_nan=False))
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "images").mkdir()
        self._events = (self.root / "events.jsonl").open("x", encoding="utf-8")
        self._sequence = 0
        self._started = time.monotonic()
        self.emit("attempt_start", metadata=self.metadata)

    def emit(self, event_type: str, **data: Any) -> dict:
        """Append and sync a JSON event without truncating model output."""
        if self._events.closed:
            raise ValueError("Attempt recorder is closed")
        event = {
            "schema_version": 1,
            "sequence": self._sequence + 1,
            "type": event_type,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.monotonic() - self._started,
            "data": data,
        }
        line = json.dumps(event, ensure_ascii=False, allow_nan=False)
        self._events.write(line + "\n")
        self._events.flush()
        os.fsync(self._events.fileno())
        self._sequence += 1
        return event

    def image(self, png: bytes, **metadata: Any) -> dict:
        """Store one content-addressed PNG and emit its observation reference."""
        if self._events.closed:
            raise ValueError("Attempt recorder is closed")
        digest = hashlib.sha256(png).hexdigest()
        relative = f"images/{digest}.png"
        path = self.root / relative
        if not path.exists():
            with path.open("xb") as image_file:
                image_file.write(png)
                image_file.flush()
                os.fsync(image_file.fileno())
        reference = {**metadata, "path": relative, "sha256": digest, "bytes": len(png)}
        self.emit("observation", image=reference)
        return reference

    def finalize(self, outcome: str, **data: Any) -> dict:
        """Verify the archive and atomically publish its completion manifest.

        Missing/corrupt artifacts raise and leave no manifest. A failed task may
        still have a complete recording. This method closes the writer even if
        verification fails; a retry must use a new attempt directory.
        """
        self.emit("attempt_end", outcome=outcome, **data)
        self.close()

        event_hash = hashlib.sha256()
        event_bytes = 0
        event_count = 0
        images: dict[str, dict] = {}
        with (self.root / "events.jsonl").open("rb") as events:
            for event_count, line in enumerate(events, start=1):
                event_hash.update(line)
                event_bytes += len(line)
                event = json.loads(line)
                if event["sequence"] != event_count or not line.endswith(b"\n"):
                    raise ValueError(f"Invalid event sequence at record {event_count}")
                if event["type"] != "observation":
                    continue
                reference = event["data"]["image"]
                digest = reference["sha256"]
                relative = reference["path"]
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)
                    or relative != f"images/{digest}.png"
                ):
                    raise ValueError("Invalid observation image reference")
                if relative not in images:
                    content = (self.root / relative).read_bytes()
                    if hashlib.sha256(content).hexdigest() != digest:
                        raise ValueError(f"Image hash mismatch: {relative}")
                    images[relative] = {
                        "path": relative,
                        "sha256": digest,
                        "bytes": len(content),
                    }
                if reference["bytes"] != images[relative]["bytes"]:
                    raise ValueError(f"Image size mismatch: {relative}")
        if event_count != self._sequence:
            raise ValueError("Event archive is incomplete")

        manifest = {
            "schema_version": 1,
            "metadata": self.metadata,
            "outcome": outcome,
            "data": data,
            "recording_complete": True,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "events": {
                "path": "events.jsonl",
                "count": event_count,
                "sha256": event_hash.hexdigest(),
                "bytes": event_bytes,
            },
            "images": list(images.values()),
        }
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=self.root, prefix=".manifest-", delete=False
        ) as manifest_file:
            json.dump(manifest, manifest_file, ensure_ascii=False, allow_nan=False, indent=2)
            manifest_file.write("\n")
            manifest_file.flush()
            os.fsync(manifest_file.fileno())
        os.replace(manifest_file.name, self.root / "manifest.json")
        for directory in (self.root / "images", self.root):
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return manifest

    def close(self) -> None:
        """Close an interrupted attempt without claiming recording completeness."""
        if not self._events.closed:
            self._events.flush()
            os.fsync(self._events.fileno())
            self._events.close()
