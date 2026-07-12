import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

from desktop.host import DesktopTraceApi, TraceRunManager
from desktop.trace_protocol import MAX_DIAGNOSTIC_BYTES, MAX_PROTOCOL_OUTPUT_BYTES, PROTOCOL_VERSION

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def success_message(run_id: str) -> dict[str, Any]:
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
    deadline = time.time() + 2
    while time.time() < deadline:
        if manager._active_runs:
            return
        time.sleep(0.01)
    raise AssertionError("Trace run did not become active.")


class TraceRunManagerTests(unittest.TestCase):
    def test_shared_success_fixture_matches_host_protocol(self):
        fixture = json.loads((FIXTURE_ROOT / "trace_protocol_success.json").read_text(encoding="utf-8"))
        code = f"import json; print(json.dumps({fixture!r}))"
        manager = TraceRunManager(worker_command_factory=worker_command(code))

        result = manager.run_known_model_trace("fixture-run")

        self.assertEqual(result, fixture)

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

    def test_simultaneous_requests_launch_only_one_worker(self):
        launch_count = 0
        launch_lock = threading.Lock()
        start_barrier = threading.Barrier(3)

        def counted_worker(request_path: Path):
            nonlocal launch_count
            with launch_lock:
                launch_count += 1
            return worker_command("import time; time.sleep(10)")(request_path)

        manager = TraceRunManager(worker_command_factory=counted_worker)
        results: list[dict[str, Any]] = []

        def run(run_id: str):
            start_barrier.wait()
            results.append(manager.run_known_model_trace(run_id))

        threads = [threading.Thread(target=run, args=(f"run-{index}",)) for index in range(2)]
        for thread in threads:
            thread.start()
        start_barrier.wait()
        wait_for_active_run(manager)
        active_run_id = next(iter(manager._active_runs))
        manager.cancel_trace(active_run_id)
        for thread in threads:
            thread.join(timeout=3)

        self.assertEqual(launch_count, 1)
        self.assertEqual(sum(result["error"]["code"] == "duplicate_run_request" for result in results), 1)

    def test_failed_startup_releases_reservation(self):
        attempts = 0

        def failing_worker(_request_path: Path):
            nonlocal attempts
            attempts += 1
            raise RuntimeError("startup failed")

        manager = TraceRunManager(worker_command_factory=failing_worker)

        first = manager.run_known_model_trace("run-start-failure")
        second = manager.run_known_model_trace("run-start-retry")

        self.assertEqual(first["error"]["code"], "worker_start_failed")
        self.assertEqual(second["error"]["code"], "worker_start_failed")
        self.assertEqual(attempts, 2)
        self.assertEqual(manager._active_runs, {})

    def test_cancel_starting_reservation_prevents_worker_launch(self):
        setup_started = threading.Event()
        allow_setup = threading.Event()
        launch_count = 0
        original_temp_directory = tempfile.TemporaryDirectory

        class BlockingTempDirectory:
            def __init__(self, *args, **kwargs):
                self._temp_dir = original_temp_directory(*args, **kwargs)
                self.name = self._temp_dir.name
                setup_started.set()
                allow_setup.wait(timeout=2)

            def cleanup(self):
                self._temp_dir.cleanup()

        def counted_worker(request_path: Path):
            nonlocal launch_count
            launch_count += 1
            return worker_command("")(request_path)

        manager = TraceRunManager(worker_command_factory=counted_worker)
        result: dict[str, Any] = {}
        tempfile.TemporaryDirectory = BlockingTempDirectory
        try:
            thread = threading.Thread(target=lambda: result.update(manager.run_known_model_trace("run-starting")))
            thread.start()
            self.assertTrue(setup_started.wait(timeout=2))
            cancel_result = manager.cancel_trace("run-starting")
            allow_setup.set()
            thread.join(timeout=3)
        finally:
            tempfile.TemporaryDirectory = original_temp_directory

        self.assertEqual(cancel_result["error"]["code"], "cancelled")
        self.assertEqual(result["error"]["code"], "cancelled")
        self.assertEqual(launch_count, 0)

    def test_protocol_rejects_unsupported_transfer(self):
        message = success_message("run-transfer")
        message["trace"] = {"transfer": "socket"}
        code = f"import json; print(json.dumps({message!r}))"
        manager = TraceRunManager(worker_command_factory=worker_command(code))

        result = manager.run_known_model_trace("run-transfer")

        self.assertEqual(result["error"]["code"], "worker_protocol_error")
        self.assertIn("unsupported transfer", result["error"]["title"].lower())

    def test_protocol_rejects_incomplete_inline_and_file_transfers(self):
        for transfer in ({"transfer": "inline"}, {"transfer": "file"}):
            with self.subTest(transfer=transfer["transfer"]):
                message = success_message(f"run-{transfer['transfer']}")
                message["trace"] = transfer
                code = f"import json; print(json.dumps({message!r}))"
                manager = TraceRunManager(worker_command_factory=worker_command(code))

                result = manager.run_known_model_trace(message["run_id"])

                self.assertEqual(result["error"]["code"], "worker_protocol_error")

    def test_protocol_rejects_non_string_warning(self):
        message = success_message("run-warning")
        message["warnings"] = [123]
        code = f"import json; print(json.dumps({message!r}))"
        manager = TraceRunManager(worker_command_factory=worker_command(code))

        result = manager.run_known_model_trace("run-warning")

        self.assertEqual(result["error"]["code"], "worker_protocol_error")

    def test_file_transfer_is_consumed_once_and_cleaned_up(self):
        code = (
            "import json, pathlib, sys; "
            "request=json.load(open(sys.argv[1], encoding='utf-8')); "
            "payload={'model_name':'LargeModel','graph':{'nodes':[],'edges':[]}}; "
            "text=json.dumps(payload, separators=(',', ':')); "
            "pathlib.Path(request['output_path']).write_text(text, encoding='utf-8'); "
            "print(json.dumps({'protocol_version':1,'type':'success','run_id':request['run_id'],"
            "'trace':{'transfer':'file','path':request['output_path'],'size_bytes':len(text.encode('utf-8'))},"
            "'warnings':[]}))"
        )
        with tempfile.TemporaryDirectory() as temp_root:
            manager = TraceRunManager(worker_command_factory=worker_command(code), temp_root=Path(temp_root))
            result = manager.run_known_model_trace("run-file")
            trace_path = result["trace"]["path"]

            consumed = manager.consume_trace_file("run-file", trace_path)
            consumed_again = manager.consume_trace_file("run-file", trace_path)

            self.assertTrue(consumed["ok"])
            self.assertEqual(consumed["payload"]["model_name"], "LargeModel")
            self.assertEqual(consumed_again["error"]["code"], "trace_file_unavailable")
            self.assertEqual(list(Path(temp_root).iterdir()), [])

    def test_protocol_output_is_bounded(self):
        code = f"print('x' * {MAX_PROTOCOL_OUTPUT_BYTES + 1})"
        manager = TraceRunManager(worker_command_factory=worker_command(code))

        result = manager.run_known_model_trace("run-large-protocol")

        self.assertEqual(result["error"]["code"], "worker_protocol_error")
        self.assertIn("too large", result["error"]["title"].lower())

    def test_diagnostic_output_is_truncated(self):
        code = (
            f"import sys; print('d' * {MAX_DIAGNOSTIC_BYTES + 100}, file=sys.stderr); "
            "print('not json')"
        )
        manager = TraceRunManager(worker_command_factory=worker_command(code))

        result = manager.run_known_model_trace("run-large-diagnostic")

        stderr = result["error"]["details"]["stderr"]
        self.assertIn("[diagnostic output truncated]", stderr)
        self.assertLess(len(stderr), MAX_DIAGNOSTIC_BYTES + 100)

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
