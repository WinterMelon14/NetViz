import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from desktop.trace_worker import trace_success_for_transport


def payload_with_padding(size: int) -> dict:
    return {
        "model_name": "TransportModel",
        "graph": {"nodes": [], "edges": []},
        "padding": "x" * size,
    }


class TraceTransportTests(unittest.TestCase):
    def test_small_payload_is_returned_inline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "trace.json"
            result = trace_success_for_transport("small", payload_with_padding(8), str(output_path))

            self.assertEqual(result["trace"]["transfer"], "inline")
            self.assertFalse(output_path.exists())

    def test_large_payload_is_written_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir, patch("desktop.trace_worker.MAX_INLINE_TRACE_BYTES", 32):
            output_path = Path(temp_dir) / "trace.json"
            payload = payload_with_padding(64)
            result = trace_success_for_transport("large", payload, str(output_path))

            self.assertEqual(result["trace"]["transfer"], "file")
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), payload)
            self.assertFalse(output_path.with_suffix(".pending").exists())

    def test_payload_above_file_limit_returns_structured_error(self):
        with patch("desktop.trace_worker.MAX_INLINE_TRACE_BYTES", 8), patch("desktop.trace_worker.MAX_TRACE_FILE_BYTES", 16):
            result = trace_success_for_transport("huge", payload_with_padding(32), "unused.json")

            self.assertEqual(result["error"]["code"], "trace_too_large")


if __name__ == "__main__":
    unittest.main()
