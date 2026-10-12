"""Artifact-integrity checks that need no browser, model, or third-party packages."""

from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from examples.not_a_robot.recorder import EventRecorder

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aN1sAAAAASUVORK5CYII="
)


class TestEventRecorder(unittest.TestCase):
    def test_complete_archive_preserves_decisions_and_deduplicates_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "attempt_1"
            recorder = EventRecorder(root, {"task_id": "level_001"})
            response = "完整模型输出\n" * 2000
            recorder.emit("model_decision", response=response)
            first = recorder.image(PNG, observation_id="before")
            second = recorder.image(PNG, observation_id="after")
            manifest = recorder.finalize("success", terminal_observation=second)

            self.assertTrue(manifest["recording_complete"])
            self.assertEqual(manifest["outcome"], "success")
            self.assertEqual(first["path"], second["path"])
            self.assertEqual(len(manifest["images"]), 1)
            self.assertEqual(len(list((root / "images").iterdir())), 1)
            self.assertEqual(json.loads((root / "manifest.json").read_text()), manifest)
            content = (root / "events.jsonl").read_bytes()
            events = [json.loads(line) for line in content.splitlines()]
            self.assertEqual(events[1]["data"]["response"], response)
            self.assertEqual([event["sequence"] for event in events], list(range(1, 6)))
            self.assertEqual(manifest["events"]["sha256"], hashlib.sha256(content).hexdigest())
            self.assertEqual(manifest["events"]["bytes"], len(content))
            elapsed = [event["elapsed_seconds"] for event in events]
            self.assertEqual(elapsed, sorted(elapsed))
            self.assertTrue(all(event["timestamp_utc"].endswith("+00:00") for event in events))
            with self.assertRaises(ValueError):
                recorder.emit("too_late")

    def test_failed_task_can_have_complete_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "attempt_1"
            recorder = EventRecorder(root, {})
            recorder.image(PNG)
            recorder.emit("error", phase="action", message="Target disappeared")
            manifest = recorder.finalize("failure", reason="action_error")
            self.assertTrue(manifest["recording_complete"])
            self.assertEqual(manifest["outcome"], "failure")

    def test_existing_attempt_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "attempt_1"
            recorder = EventRecorder(root, {})
            recorder.close()
            original = (root / "events.jsonl").read_bytes()
            with self.assertRaises(FileExistsError):
                EventRecorder(root, {"replacement": True})
            self.assertEqual((root / "events.jsonl").read_bytes(), original)

    def test_tampered_or_missing_image_prevents_completion(self):
        for remove in (False, True):
            with self.subTest(remove=remove), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "attempt_1"
                recorder = EventRecorder(root, {})
                reference = recorder.image(PNG)
                image_path = root / reference["path"]
                if remove:
                    image_path.unlink()
                else:
                    image_path.write_bytes(b"corrupt screenshot")
                with self.assertRaises((ValueError, FileNotFoundError)):
                    recorder.finalize("success")
                self.assertFalse((root / "manifest.json").exists())

    def test_invalid_event_sequence_prevents_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "attempt_1"
            recorder = EventRecorder(root, {})
            recorder.emit("model_decision", action="click")
            events_path = root / "events.jsonl"
            lines = events_path.read_text().splitlines()
            event = json.loads(lines[0])
            event["sequence"] = 9
            lines[0] = json.dumps(event)
            events_path.write_text("\n".join(lines) + "\n")
            with self.assertRaises(ValueError):
                recorder.finalize("success")
            self.assertFalse((root / "manifest.json").exists())

    def test_close_preserves_events_without_claiming_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "attempt_1"
            recorder = EventRecorder(root, {})
            recorder.emit("error", phase="browser", message="Cancelled")
            recorder.close()
            recorder.close()
            self.assertFalse((root / "manifest.json").exists())
            events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
            self.assertEqual(events[-1]["type"], "error")
            with self.assertRaises(ValueError):
                recorder.image(PNG)


if __name__ == "__main__":
    unittest.main()
