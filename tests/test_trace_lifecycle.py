import hashlib
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from desktop.host import TraceRunManager
from desktop.trace_protocol import MAX_DIAGNOSTIC_BYTES, MAX_PROTOCOL_OUTPUT_BYTES, PROTOCOL_VERSION


FIXTURE_MODEL = Path(__file__).resolve().parent / "fixtures" / "user_models" / "valid_model.py"


def bridge_request(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "source": {
            "file_path": str(FIXTURE_MODEL),
            "class_name": "UserModel",
            "content_sha256": hashlib.sha256(FIXTURE_MODEL.read_bytes()).hexdigest(),
        },
        "constructor": {"args": [], "kwargs": {}},
        "inputs": [{
            "kind": "tensor",
            "parameter_name": "x",
            "shape": [1, 4],
            "dtype": "float32",
            "generator": "random_normal",
        }],
    }


def success_message(run_id: str) -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "type": "success",
        "run_id": run_id,
        "trace": {
            "transfer": "inline",
            "payload": {"model_name": "TestModel", "graph": {"nodes": [], "edges": []}},
        },
        "warnings": [],
    }


def worker_command(code: str):
    return lambda request_path: [sys.executable, "-c", code, str(request_path)]


def wait_for_active_run(manager: TraceRunManager) -> None:
    deadline = time.time() + 3
    while time.time() < deadline:
        if manager._active_runs:
            return
        time.sleep(0.01)
    raise AssertionError("Trace run did not become active.")


class TraceRunManagerTests(unittest.TestCase):
    def test_success_result_is_returned(self):
        code = (
            "import json, sys; request=json.load(open(sys.argv[1], encoding='utf-8')); "
            f"print(json.dumps({success_message('RUN_ID')!r} | {{'run_id': request['run_id']}}))"
        )
        result = TraceRunManager(worker_command_factory=worker_command(code)).run_user_trace(bridge_request("success"))
        self.assertEqual(result["type"], "success")

    def test_malformed_or_missing_protocol_becomes_structured_error(self):
        for code in ("print('not json')", ""):
            with self.subTest(code=code):
                result = TraceRunManager(worker_command_factory=worker_command(code)).run_user_trace(bridge_request("protocol"))
                self.assertEqual(result["error"]["code"], "worker_protocol_error")

    def test_unsupported_version_and_stale_run_are_rejected(self):
        messages = [
            {**success_message("version"), "protocol_version": 999},
            success_message("other-run"),
        ]
        for index, message in enumerate(messages):
            with self.subTest(index=index):
                code = f"import json; print(json.dumps({message!r}))"
                result = TraceRunManager(worker_command_factory=worker_command(code)).run_user_trace(bridge_request("version"))
                self.assertEqual(result["error"]["code"], "worker_protocol_error")

    def test_worker_nonzero_exit_and_stderr_are_reported(self):
        code = "import sys; print('diagnostic detail', file=sys.stderr); print('partial'); sys.exit(7)"
        result = TraceRunManager(worker_command_factory=worker_command(code)).run_user_trace(bridge_request("crash"))
        self.assertEqual(result["error"]["code"], "worker_crashed")
        self.assertEqual(result["error"]["details"]["exit_code"], 7)
        self.assertIn("diagnostic detail", result["error"]["details"]["stderr"])

    def test_timeout_terminates_worker(self):
        manager = TraceRunManager(timeout_seconds=0.1, worker_command_factory=worker_command("import time; time.sleep(10)"))
        result = manager.run_user_trace(bridge_request("timeout"))
        self.assertEqual(result["error"]["code"], "timeout")

    def test_cancellation_is_targeted_and_idempotent(self):
        manager = TraceRunManager(worker_command_factory=worker_command("import time; time.sleep(10)"))
        result: dict = {}
        thread = threading.Thread(target=lambda: result.update(manager.run_user_trace(bridge_request("cancel"))))
        thread.start()
        wait_for_active_run(manager)
        first = manager.cancel_trace("cancel")
        thread.join(timeout=4)
        second = manager.cancel_trace("cancel")
        self.assertFalse(thread.is_alive())
        self.assertEqual(first["error"]["code"], "cancelled")
        self.assertEqual(result["error"]["code"], "cancelled")
        self.assertEqual(second["error"]["code"], "cancelled")

    def test_duplicate_request_is_rejected(self):
        manager = TraceRunManager(worker_command_factory=worker_command("import time; time.sleep(10)"))
        thread = threading.Thread(target=lambda: manager.run_user_trace(bridge_request("active")))
        thread.start()
        wait_for_active_run(manager)
        duplicate = manager.run_user_trace(bridge_request("duplicate"))
        manager.cancel_trace("active")
        thread.join(timeout=4)
        self.assertEqual(duplicate["error"]["code"], "duplicate_run_request")

    def test_failed_start_releases_reservation(self):
        def fail(_path: Path):
            raise RuntimeError("startup failed")

        manager = TraceRunManager(worker_command_factory=fail)
        self.assertEqual(manager.run_user_trace(bridge_request("first"))["error"]["code"], "worker_start_failed")
        self.assertEqual(manager.run_user_trace(bridge_request("second"))["error"]["code"], "worker_start_failed")
        self.assertEqual(manager._active_runs, {})

    def test_file_transfer_is_consumed_once_and_cleaned_up(self):
        code = (
            "import json, pathlib, sys; request=json.load(open(sys.argv[1], encoding='utf-8')); "
            "payload={'model_name':'LargeModel','graph':{'nodes':[],'edges':[]}}; "
            "text=json.dumps(payload,separators=(',',':')); path=pathlib.Path(request['output_path']); "
            "path.write_text(text,encoding='utf-8'); "
            "print(json.dumps({'protocol_version':1,'type':'success','run_id':request['run_id'],"
            "'trace':{'transfer':'file','path':str(path),'size_bytes':len(text.encode('utf-8'))},'warnings':[]}))"
        )
        with tempfile.TemporaryDirectory() as temp_root:
            manager = TraceRunManager(worker_command_factory=worker_command(code), temp_root=Path(temp_root))
            result = manager.run_user_trace(bridge_request("file"))
            consumed = manager.consume_trace_file("file", result["trace"]["path"])
            consumed_again = manager.consume_trace_file("file", result["trace"]["path"])
            self.assertTrue(consumed["ok"])
            self.assertEqual(consumed["payload"]["model_name"], "LargeModel")
            self.assertEqual(consumed_again["error"]["code"], "trace_file_unavailable")
            self.assertEqual(list(Path(temp_root).iterdir()), [])

    def test_protocol_and_diagnostic_output_are_bounded(self):
        protocol = TraceRunManager(worker_command_factory=worker_command(f"print('x' * {MAX_PROTOCOL_OUTPUT_BYTES + 1})"))
        result = protocol.run_user_trace(bridge_request("large-protocol"))
        self.assertEqual(result["error"]["code"], "worker_protocol_error")

        code = f"import sys; print('d' * {MAX_DIAGNOSTIC_BYTES + 100}, file=sys.stderr); print('not json')"
        diagnostic = TraceRunManager(worker_command_factory=worker_command(code)).run_user_trace(bridge_request("large-diagnostic"))
        self.assertIn("[diagnostic output truncated]", diagnostic["error"]["details"]["stderr"])


if __name__ == "__main__":
    unittest.main()
