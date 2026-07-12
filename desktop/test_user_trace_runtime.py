import json
import hashlib
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from desktop.host import DesktopTraceApi, TraceRunManager
from desktop.selected_files import SelectedPythonFiles
from desktop.trace_worker import run_trace
from desktop.user_model_runtime import (
    UserTraceRuntimeError,
    build_tensor_inputs,
    instantiate_model,
    load_user_module,
    sanitize_user_source,
)


VALID_MODEL_SOURCE = """\
import torch

class UserModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)
"""
FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "user_models"


class UserTraceRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source_path = self.root / "user_model.py"
        self.source_path.write_text(VALID_MODEL_SOURCE, encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def worker_request(self, run_id="user-runtime"):
        return {
            "protocol_version": 1,
            "run_id": run_id,
            "source": {
                "file_path": str(self.source_path),
                "class_name": "UserModel",
                "content_sha256": hashlib.sha256(self.source_path.read_bytes()).hexdigest(),
            },
            "constructor": {"args": [], "kwargs": {}},
            "inputs": [{
                "kind": "tensor",
                "shape": [1, 4],
                "dtype": "float32",
                "generator": "random_normal",
            }],
            "output_path": str(self.root / "result.json"),
        }

    def bridge_request(self, run_id="user-host"):
        request = self.worker_request(run_id)
        request.pop("protocol_version")
        request.pop("output_path")
        return request

    def fixture_bridge_request(self, fixture_name: str, run_id: str, class_name: str = "UserModel"):
        path = FIXTURE_ROOT / fixture_name
        return {
            "run_id": run_id,
            "source": {
                "file_path": str(path),
                "class_name": class_name,
                "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
            "constructor": {"args": [], "kwargs": {}},
            "inputs": [{
                "kind": "tensor",
                "shape": [1, 4],
                "dtype": "float32",
                "generator": "random_normal",
            }],
        }

    def write_request(self, request):
        path = self.root / "request.json"
        path.write_text(json.dumps(request), encoding="utf-8")
        return path

    def test_module_names_are_unique_per_load(self):
        digest = hashlib.sha256(self.source_path.read_bytes()).hexdigest()
        first = load_user_module(str(self.source_path), "same-run", digest)
        second = load_user_module(str(self.source_path), "same-run", digest)
        self.assertNotEqual(first.__name__, second.__name__)

    def test_import_failure_is_structured(self):
        self.source_path.write_text("def broken(:\n", encoding="utf-8")
        with self.assertRaises(UserTraceRuntimeError) as raised:
            load_user_module(str(self.source_path), "import-failure", hashlib.sha256(self.source_path.read_bytes()).hexdigest())
        self.assertEqual(raised.exception.code, "module_import_failed")
        self.assertEqual(raised.exception.stage, "module_import")

    def test_worker_rejects_source_changed_after_request_creation(self):
        request = self.worker_request("source-race")
        sentinel = self.root / "executed.txt"
        self.source_path.write_text(
            f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('executed')\n",
            encoding="utf-8",
        )
        result = run_trace(str(self.write_request(request)))
        self.assertEqual(result["error"]["code"], "source_changed")
        self.assertFalse(sentinel.exists())

    def test_model_resolution_and_construction_failures_are_distinct(self):
        module = load_user_module(str(self.source_path), "resolution", hashlib.sha256(self.source_path.read_bytes()).hexdigest())
        with self.assertRaises(UserTraceRuntimeError) as missing:
            instantiate_model(module, "Missing", [], {})
        self.assertEqual(missing.exception.code, "model_class_not_found")

        self.source_path.write_text("UserModel = 42\n", encoding="utf-8")
        module = load_user_module(str(self.source_path), "not-class", hashlib.sha256(self.source_path.read_bytes()).hexdigest())
        with self.assertRaises(UserTraceRuntimeError) as not_class:
            instantiate_model(module, "UserModel", [], {})
        self.assertEqual(not_class.exception.code, "model_class_invalid")

        self.source_path.write_text("class UserModel: pass\n", encoding="utf-8")
        module = load_user_module(str(self.source_path), "not-module", hashlib.sha256(self.source_path.read_bytes()).hexdigest())
        with self.assertRaises(UserTraceRuntimeError) as not_module:
            instantiate_model(module, "UserModel", [], {})
        self.assertEqual(not_module.exception.code, "model_instance_invalid")

        self.source_path.write_text(
            "import torch\nclass UserModel(torch.nn.Module):\n"
            "    def __init__(self): raise RuntimeError('construction broke')\n",
            encoding="utf-8",
        )
        module = load_user_module(str(self.source_path), "construction", hashlib.sha256(self.source_path.read_bytes()).hexdigest())
        with self.assertRaises(UserTraceRuntimeError) as construction:
            instantiate_model(module, "UserModel", [], {})
        self.assertEqual(construction.exception.code, "model_construction_failed")

    def test_tensor_input_has_exact_shape_dtype_and_device(self):
        tensors = build_tensor_inputs(self.worker_request()["inputs"])
        self.assertEqual(tuple(tensors[0].shape), (1, 4))
        self.assertEqual(tensors[0].dtype, torch.float32)
        self.assertEqual(tensors[0].device.type, "cpu")

    def test_self_contained_file_produces_inline_trace(self):
        result = run_trace(str(self.write_request(self.worker_request())))
        self.assertEqual(result["type"], "success")
        self.assertEqual(result["trace"]["transfer"], "inline")
        self.assertEqual(result["trace"]["payload"]["model_name"], "UserModel")

    def test_worker_suppresses_unrelated_top_level_calls(self):
        sentinel = self.root / "top-level-executed.txt"
        self.source_path.write_text(
            '"""Model fixture docstring."""\n'
            "from pathlib import Path\n"
            "import torch\n"
            "CONFIG = 2\n"
            "def scale(x): return x * CONFIG\n"
            "class UserModel(torch.nn.Module):\n"
            "    def forward(self, x): return scale(x)\n"
            "class Unrelated(torch.nn.Module): pass\n"
            "model = Unrelated(d_model=4)\n"
            f"Path({str(sentinel)!r}).write_text('executed')\n",
            encoding="utf-8",
        )

        result = run_trace(str(self.write_request(self.worker_request("sanitized-source"))))

        self.assertEqual(result["type"], "success")
        self.assertFalse(sentinel.exists())
        self.assertEqual(list(self.root.glob("sanitized-model-*.py")), [])

    def test_sanitizer_preserves_lines_and_non_call_configuration(self):
        source = (
            '"""docs"""\n'
            "VALUE = 4\n"
            "created = object()\n"
            "print('side effect')\n"
            "class Model: pass\n"
        )

        sanitized = sanitize_user_source(source)

        self.assertEqual(len(sanitized.splitlines()), len(source.splitlines()))
        self.assertIn("VALUE = 4", sanitized)
        self.assertIn("class Model: pass", sanitized)
        self.assertNotIn("object()", sanitized)
        self.assertNotIn("print(", sanitized)

    def test_self_contained_file_uses_file_backed_transport(self):
        with patch("desktop.trace_worker.MAX_INLINE_TRACE_BYTES", 1):
            result = run_trace(str(self.write_request(self.worker_request("file-user"))))
        self.assertEqual(result["type"], "success")
        self.assertEqual(result["trace"]["transfer"], "file")
        self.assertTrue(Path(result["trace"]["path"]).is_file())

    def test_host_runs_user_trace_and_survives_worker_crash(self):
        manager = TraceRunManager(timeout_seconds=20)
        success = manager.run_user_trace(self.bridge_request())
        self.assertEqual(success["type"], "success")

        self.source_path.write_text(
            "import os\n"
            "def crash_worker(selected_class): os._exit(9)\n"
            "@crash_worker\n"
            "class UserModel: pass\n",
            encoding="utf-8",
        )
        crashed = manager.run_user_trace(self.bridge_request("crash-user"))
        self.assertEqual(crashed["error"]["code"], "worker_crashed")

        recovered = manager.run_user_trace(self.fixture_bridge_request("valid_model.py", "after-user-crash"))
        self.assertEqual(recovered["type"], "success")

    def test_host_overwrites_untrusted_output_path(self):
        attacker_path = self.root / "attacker-output.json"
        request = self.bridge_request("host-output")
        request["output_path"] = str(attacker_path)

        result = TraceRunManager().run_user_trace(request)

        self.assertEqual(result["type"], "success")
        self.assertFalse(attacker_path.exists())

    def test_selected_handle_bridge_runs_trace(self):
        selected_files = SelectedPythonFiles(picker=lambda: str(self.source_path))
        descriptor = selected_files.select()["selected"]
        inspection = selected_files.inspect(descriptor["selectionId"])
        api = DesktopTraceApi(manager=TraceRunManager(), selected_files=selected_files)
        request = self.bridge_request("selected-bridge")
        request["source"] = {
            "selection_id": descriptor["selectionId"],
            "class_name": "UserModel",
            "content_sha256": inspection["sourceIdentity"]["contentSha256"],
        }

        result = api.runSelectedUserTrace(request)

        self.assertEqual(result["type"], "success")
        self.assertEqual(result["trace"]["payload"]["model_name"], "UserModel")

    def test_literal_constructor_arguments_are_applied(self):
        self.source_path.write_text(
            "import torch\nclass UserModel(torch.nn.Module):\n"
            "    def __init__(self, width, *, config):\n        super().__init__()\n        self.width = width\n        self.config = config\n"
            "    def forward(self, x): return x + self.width if self.config['enabled'] else x\n",
            encoding="utf-8",
        )
        request = self.worker_request("literal-constructor")
        request["constructor"] = {"args": [2], "kwargs": {"config": {"enabled": True}}}
        result = run_trace(str(self.write_request(request)))
        self.assertEqual(result["type"], "success")

    def test_cancellation_works_during_user_execution_stages(self):
        stage_sources = {
            "import": "import time\ntime.sleep(10)\n" + VALID_MODEL_SOURCE,
            "construction": (
                "import time\nimport torch\nclass UserModel(torch.nn.Module):\n"
                "    def __init__(self):\n        super().__init__()\n        time.sleep(10)\n"
                "    def forward(self, x): return x\n"
            ),
            "forward": (
                "import time\nimport torch\nclass UserModel(torch.nn.Module):\n"
                "    def forward(self, x):\n        time.sleep(10)\n        return x\n"
            ),
        }
        for stage, source in stage_sources.items():
            with self.subTest(stage=stage):
                self.source_path.write_text(source, encoding="utf-8")
                manager = TraceRunManager(timeout_seconds=20)
                run_id = f"cancel-{stage}"
                result: dict = {}
                thread = threading.Thread(target=lambda: result.update(manager.run_user_trace(self.bridge_request(run_id))))
                thread.start()
                deadline = time.time() + 3
                while time.time() < deadline and not manager._active_runs:
                    time.sleep(0.01)
                self.assertIn(run_id, manager._active_runs)
                cancellation = manager.cancel_trace(run_id)
                thread.join(timeout=4)
                self.assertFalse(thread.is_alive())
                self.assertEqual(cancellation["error"]["code"], "cancelled")
                self.assertEqual(result["error"]["code"], "cancelled")

    def test_worker_error_preserves_failure_stage(self):
        self.source_path.write_text("import tensor_trace_missing_test_module\n", encoding="utf-8")
        result = run_trace(str(self.write_request(self.worker_request())))
        self.assertEqual(result["error"]["code"], "module_import_failed")
        self.assertEqual(result["error"]["stage"], "module_import")

    def test_failure_stage_fixtures_are_stable_and_host_recovers(self):
        cases = [
            ("import_failure.py", "module_import_failed"),
            ("missing_class.py", "model_class_not_found"),
            ("wrong_object_type.py", "model_class_invalid"),
            ("constructor_failure.py", "model_construction_failed"),
            ("forward_failure.py", "trace_execution_failed"),
            ("excessive_stderr.py", "module_import_failed"),
        ]
        with tempfile.TemporaryDirectory() as temp_root:
            manager = TraceRunManager(timeout_seconds=20, temp_root=Path(temp_root))
            for index, (fixture_name, error_code) in enumerate(cases):
                with self.subTest(fixture=fixture_name):
                    result = manager.run_user_trace(self.fixture_bridge_request(fixture_name, f"fixture-{index}"))
                    self.assertEqual(result["error"]["code"], error_code)
                    self.assertEqual(manager._active_runs, {})
                    self.assertEqual(list(Path(temp_root).iterdir()), [])
                    recovered = manager.run_user_trace(self.fixture_bridge_request("valid_model.py", f"recover-{index}"))
                    self.assertEqual(recovered["type"], "success")

    def test_sleeping_fixture_times_out_and_oversized_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_root:
            manager = TraceRunManager(timeout_seconds=0.1, temp_root=Path(temp_root))
            timed_out = manager.run_user_trace(self.fixture_bridge_request("sleeping_forward.py", "sleep-fixture"))
            self.assertEqual(timed_out["error"]["code"], "timeout")
            oversized = self.fixture_bridge_request("valid_model.py", "oversized-fixture")
            oversized["inputs"][0]["shape"] = [16_777_217]
            rejected = TraceRunManager(timeout_seconds=20, temp_root=Path(temp_root)).run_user_trace(oversized)
            self.assertEqual(rejected["error"]["code"], "user_trace_request_invalid")
            self.assertEqual(list(Path(temp_root).iterdir()), [])

    def test_large_trace_fixture_uses_file_transport(self):
        manager = TraceRunManager(timeout_seconds=30)
        result = manager.run_user_trace(self.fixture_bridge_request("large_trace.py", "large-fixture"))
        self.assertEqual(result["type"], "success")
        self.assertEqual(result["trace"]["transfer"], "file")
        consumed = manager.consume_trace_file(result["run_id"], result["trace"]["path"])
        self.assertTrue(consumed["ok"])


if __name__ == "__main__":
    unittest.main()
