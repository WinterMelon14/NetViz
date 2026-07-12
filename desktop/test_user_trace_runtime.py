import json
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
            "source": {"file_path": str(self.source_path), "class_name": "UserModel"},
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

    def write_request(self, request):
        path = self.root / "request.json"
        path.write_text(json.dumps(request), encoding="utf-8")
        return path

    def test_module_names_are_unique_per_load(self):
        first = load_user_module(str(self.source_path), "same-run")
        second = load_user_module(str(self.source_path), "same-run")
        self.assertNotEqual(first.__name__, second.__name__)

    def test_import_failure_is_structured(self):
        self.source_path.write_text("def broken(:\n", encoding="utf-8")
        with self.assertRaises(UserTraceRuntimeError) as raised:
            load_user_module(str(self.source_path), "import-failure")
        self.assertEqual(raised.exception.code, "module_import_failed")
        self.assertEqual(raised.exception.stage, "module_import")

    def test_model_resolution_and_construction_failures_are_distinct(self):
        module = load_user_module(str(self.source_path), "resolution")
        with self.assertRaises(UserTraceRuntimeError) as missing:
            instantiate_model(module, "Missing")
        self.assertEqual(missing.exception.code, "model_class_not_found")

        self.source_path.write_text("UserModel = 42\n", encoding="utf-8")
        module = load_user_module(str(self.source_path), "not-class")
        with self.assertRaises(UserTraceRuntimeError) as not_class:
            instantiate_model(module, "UserModel")
        self.assertEqual(not_class.exception.code, "model_class_invalid")

        self.source_path.write_text("class UserModel: pass\n", encoding="utf-8")
        module = load_user_module(str(self.source_path), "not-module")
        with self.assertRaises(UserTraceRuntimeError) as not_module:
            instantiate_model(module, "UserModel")
        self.assertEqual(not_module.exception.code, "model_instance_invalid")

        self.source_path.write_text(
            "import torch\nclass UserModel(torch.nn.Module):\n"
            "    def __init__(self): raise RuntimeError('construction broke')\n",
            encoding="utf-8",
        )
        module = load_user_module(str(self.source_path), "construction")
        with self.assertRaises(UserTraceRuntimeError) as construction:
            instantiate_model(module, "UserModel")
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

        self.source_path.write_text("import os\nos._exit(9)\n", encoding="utf-8")
        crashed = manager.run_user_trace(self.bridge_request("crash-user"))
        self.assertEqual(crashed["error"]["code"], "worker_crashed")

        recovered = manager.run_known_model_trace("after-user-crash")
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
        api = DesktopTraceApi(manager=TraceRunManager(), selected_files=selected_files)
        request = self.bridge_request("selected-bridge")
        request["source"] = {
            "selection_id": descriptor["selectionId"],
            "class_name": "UserModel",
        }

        result = api.runSelectedUserTrace(request)

        self.assertEqual(result["type"], "success")
        self.assertEqual(result["trace"]["payload"]["model_name"], "UserModel")

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
        self.source_path.write_text("raise RuntimeError('import broke')\n", encoding="utf-8")
        result = run_trace(str(self.write_request(self.worker_request())))
        self.assertEqual(result["error"]["code"], "module_import_failed")
        self.assertEqual(result["error"]["stage"], "module_import")


if __name__ == "__main__":
    unittest.main()
