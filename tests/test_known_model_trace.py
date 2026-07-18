import unittest
import hashlib
from pathlib import Path
import torch.fx as fx
import torch
from unittest.mock import patch

from desktop.host import TraceRunManager
from util.summary import model_summary
from util.Interpreter import SummaryInterpreter


class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


def known_model_input():
    return torch.randn(1, 4)


class KnownModelTraceTests(unittest.TestCase):
    def test_real_desktop_worker_returns_user_model_trace(self):
        model_path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "user_models" / "valid_model.py"
        result = TraceRunManager().run_user_trace({
            "run_id": "integration-user-model",
            "source": {
                "file_path": str(model_path),
                "class_name": "UserModel",
                "content_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            },
            "constructor": {"args": [], "kwargs": {}},
            "inputs": [{
                "kind": "tensor",
                "parameter_name": "x",
                "shape": [1, 4],
                "dtype": "float32",
                "generator": "random_normal",
            }],
        })

        self.assertEqual(result["type"], "success")
        self.assertEqual(result["trace"]["transfer"], "inline")
        self.assertEqual(result["trace"]["payload"]["model_name"], "UserModel")

    def test_known_model_produces_labeled_connected_trace(self):
        payload = model_summary(TestModel(), known_model_input())

        nodes = {node["id"]: node for node in payload["graph"]["nodes"]}
        self.assertEqual(payload["model_name"], "TestModel")
        self.assertEqual(nodes["x"]["label"], "x")
        self.assertEqual(nodes["linear"]["label"], "Linear")
        self.assertEqual(nodes["linear"]["module"]["type"], "Linear")
        self.assertEqual(nodes["linear"]["params"]["count"], 10)
        self.assertEqual(
            payload["graph"]["edges"],
            [{
                "id": "x:0->linear:0",
                "source": "x",
                "source_output": 0,
                "target": "linear",
                "target_input": 0,
            }],
        )

    def test_desktop_trace_executes_model_once_without_shape_propagation(self):
        model = TestModel()
        execution_count = 0

        def count_execution(_module, _args, _output):
            nonlocal execution_count
            execution_count += 1

        model.linear.register_forward_hook(count_execution)
        model_summary(model, known_model_input(), run_shape_prop=False)

        self.assertEqual(execution_count, 1)

    def test_shape_propagation_failure_emits_warning(self):
        with patch("util.summary.ShapeProp.propagate", side_effect=RuntimeError("shape failure")):
            with self.assertWarnsRegex(RuntimeWarning, "Shape propagation failed"):
                payload = model_summary(TestModel(), known_model_input(), run_shape_prop=True)

        self.assertEqual(payload["model_name"], "TestModel")

    def test_interpreter_does_not_retain_raw_node_values(self):
        interpreter = SummaryInterpreter(fx.symbolic_trace(TestModel()))
        payload = model_summary(TestModel(), known_model_input(), run_shape_prop=False)

        self.assertFalse(hasattr(interpreter, "node_values"))
        self.assertEqual(payload["graph"]["nodes"][1]["outputs"][0]["shape"], [1, 2])

    def test_deep_model_activation_cache_reduction_measurement(self):
        model = torch.nn.Sequential(*[torch.nn.Linear(256, 256) for _ in range(64)])
        payload = model_summary(model, torch.randn(1, 256), run_shape_prop=False)
        legacy_retained_bytes = sum(
            output.get("memory", {}).get("num_bytes", 0)
            for node in payload["graph"]["nodes"]
            for output in node["outputs"]
        )

        self.assertEqual(legacy_retained_bytes, 65 * 1024)
        self.assertFalse(hasattr(SummaryInterpreter(fx.symbolic_trace(model)), "node_values"))


if __name__ == "__main__":
    unittest.main()
