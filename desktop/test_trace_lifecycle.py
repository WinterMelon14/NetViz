import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

from desktop.host import DesktopTraceApi, TraceRunManager
from desktop.trace_protocol import PROTOCOL_VERSION


def success_message(run_id: str) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "type": "success",
        "run_id": run_id,
        "trace": {
            "transfer": "inline",
            "payload": {"graph": {"nodes": [], "edges": []}},
        },
        "warnings": [],
    }


def worker_command(code: str):
    return lambda request_path: [sys.executable, "-c", code, str(request_path)]


def wait_for_active_run(manager: TraceRunManager) -> None:
    deadline = time.time() + 2
    while time.time() < deadline:
        if manager._active_runs:
            return
        time.sleep(0.01)
    raise AssertionError("Trace run did not become active.")


class TraceRunManagerTests(unittest.TestCase):
    def test_success_result_is_returned(self):
        code = (
            "import json, sys; "
            "request=json.load(open(sys.argv[1], encoding='utf-8')); "
            f"print(json.dumps({success_message('RUN_ID')!r} | {{'run_id': request['run_id']}}))"
        )
        manager = TraceRunManager(worker_command_factory=worker_command(code))

        result = manager.run_known_model_trace("run-success")

        self.assertEqual(result["type"], "success")
        self.assertEqual(result["run_id"], "run-success")

    def test_malformed_protocol_json_becomes_structured_error(self):
        manager = TraceRunManager(worker_command_factory=worker_command("print('not json')"))

        result = manager.run_known_model_trace("run-malformed")

        self.assertEqual(result["type"], "error")
        self.assertEqual(result["error"]["code"], "worker_protocol_error")
        self.assertIn("stdout", result["error"]["details"])

    def test_unsupported_protocol_version_becomes_structured_error(self):
        code = "import json; print(json.dumps({'protocol_version': 999, 'type': 'success', 'run_id': 'run-version'}))"
        manager = TraceRunManager(worker_command_factory=worker_command(code))

        result = manager.run_known_model_trace("run-version")

        self.assertEqual(result["error"]["code"], "worker_protocol_error")
        self.assertIn("protocol_version", result["error"]["details"])

    def test_mismatched_run_id_becomes_structured_error(self):
        code = (
            "import json; "
            f"print(json.dumps({success_message('other-run')!r}))"
        )
        manager = TraceRunManager(worker_command_factory=worker_command(code))

        result = manager.run_known_model_trace("run-expected")

        self.assertEqual(result["error"]["code"], "worker_protocol_error")
        self.assertEqual(result["error"]["details"]["actual_run_id"], "other-run")

    def test_worker_non_zero_exit_becomes_structured_error(self):
        manager = TraceRunManager(worker_command_factory=worker_command("import sys; print('partial'); sys.exit(7)"))

        result = manager.run_known_model_trace("run-crash")

        self.assertEqual(result["error"]["code"], "worker_crashed")
        self.assertEqual(result["error"]["details"]["exit_code"], 7)

    def test_worker_exit_without_output_becomes_structured_error(self):
        manager = TraceRunManager(worker_command_factory=worker_command(""))

        result = manager.run_known_model_trace("run-empty")

        self.assertEqual(result["error"]["code"], "worker_protocol_error")
        self.assertIn("without writing", result["error"]["message"])

    def test_stderr_is_captured_as_technical_details(self):
        code = (
            "import sys; "
            "print('diagnostic detail', file=sys.stderr); "
            "print('not json')"
        )
        manager = TraceRunManager(worker_command_factory=worker_command(code))

        result = manager.run_known_model_trace("run-stderr")

        self.assertEqual(result["type"], "error")
        self.assertIn("diagnostic detail", result["error"]["details"]["stderr"])

    def test_timeout_terminates_worker(self):
        manager = TraceRunManager(
            timeout_seconds=0.1,
            worker_command_factory=worker_command("import time; time.sleep(10)"),
        )

        result = manager.run_known_model_trace("run-timeout")

        self.assertEqual(result["error"]["code"], "timeout")

    def test_cancellation_targets_matching_run(self):
        manager = TraceRunManager(worker_command_factory=worker_command("import time; time.sleep(10)"))
        thread_result: dict[str, Any] = {}

        thread = threading.Thread(target=lambda: thread_result.update(manager.run_known_model_trace("run-cancel")))
        thread.start()
        wait_for_active_run(manager)

        cancel_result = manager.cancel_trace("run-cancel")
        thread.join(timeout=3)

        self.assertEqual(cancel_result["error"]["code"], "cancelled")
        self.assertEqual(thread_result["error"]["code"], "cancelled")

    def test_duplicate_run_request_is_rejected(self):
        manager = TraceRunManager(worker_command_factory=worker_command("import time; time.sleep(10)"))
        thread = threading.Thread(target=lambda: manager.run_known_model_trace("run-active"))
        thread.start()
        wait_for_active_run(manager)

        result = manager.run_known_model_trace("run-duplicate")
        manager.cancel_trace("run-active")
        thread.join(timeout=3)

        self.assertEqual(result["error"]["code"], "duplicate_run_request")

    def test_temp_files_are_cleaned_up(self):
        code = (
            "import json, sys; "
            "request=json.load(open(sys.argv[1], encoding='utf-8')); "
            f"print(json.dumps({success_message('RUN_ID')!r} | {{'run_id': request['run_id']}}))"
        )
        with tempfile.TemporaryDirectory() as temp_root:
            manager = TraceRunManager(worker_command_factory=worker_command(code), temp_root=Path(temp_root))

            result = manager.run_known_model_trace("run-cleanup")

            self.assertEqual(result["type"], "success")
            self.assertEqual(list(Path(temp_root).iterdir()), [])

    def test_multiple_result_messages_become_structured_error(self):
        code = "print('{\"protocol_version\": 1}'); print('{\"protocol_version\": 1}')"
        manager = TraceRunManager(worker_command_factory=worker_command(code))

        result = manager.run_known_model_trace("run-multiple")

        self.assertEqual(result["error"]["code"], "worker_protocol_error")
        self.assertIn("multiple", result["error"]["title"].lower())

    def test_source_inspection_bridge_accepts_valid_request(self):
        api = DesktopTraceApi(manager=TraceRunManager(worker_command_factory=worker_command("")))

        result = api.inspectModelSource({
            "sourceText": "import torch\nclass Demo(torch.nn.Module):\n    def forward(self, x): return x\n"
        })

        self.assertTrue(result["ok"])
        self.assertEqual(result["candidates"][0]["className"], "Demo")

    def test_source_inspection_bridge_rejects_invalid_request_shape(self):
        api = DesktopTraceApi(manager=TraceRunManager(worker_command_factory=worker_command("")))

        result = api.inspectModelSource({"sourceText": 123})

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "source_protocol_error")


if __name__ == "__main__":
    unittest.main()
