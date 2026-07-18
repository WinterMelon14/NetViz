import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from desktop.trace_worker import run_trace
from desktop.user_model_runtime import build_structured_inputs


class StructuredInputExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source_path = self.root / "model.py"

    def tearDown(self):
        self.temp_dir.cleanup()

    def request(self, run_id: str, source: str) -> dict:
        self.source_path.write_text(source, encoding="utf-8")
        return {
            "protocol_version": 1,
            "input_schema_version": 2,
            "run_id": run_id,
            "source": {
                "file_path": str(self.source_path),
                "class_name": "UserModel",
                "content_sha256": hashlib.sha256(self.source_path.read_bytes()).hexdigest(),
            },
            "constructor": {"args": [], "kwargs": {}},
            "args": [],
            "kwargs": {},
            "input_provider": None,
            "output_path": str(self.root / "result.json"),
        }

    def run_request(self, request: dict) -> dict:
        path = self.root / "request.json"
        path.write_text(json.dumps(request), encoding="utf-8")
        return run_trace(str(path))

    def test_constructs_nested_values_and_kwargs(self):
        args, kwargs, diagnostics = build_structured_inputs(
            [{"kind": "tuple", "items": [{"kind": "integer", "value": 3}, {"kind": "none"}]}],
            {"mask": {"kind": "tensor", "shape": [2, 2], "dtype": "int64", "generator": "random_integer", "integer_max_exclusive": 2}},
        )
        self.assertEqual(args, [(3, None)])
        self.assertEqual(tuple(kwargs["mask"].shape), (2, 2))
        self.assertEqual(diagnostics[0]["parameter_name"], "kwargs.mask")

    def test_executes_keyword_only_and_nested_inputs(self):
        request = self.request("structured-runtime", (
            "import torch\nclass UserModel(torch.nn.Module):\n"
            "    def forward(self, pair, *, mask, scale=1.0):\n"
            "        left, right = pair\n"
            "        return (left + right) * mask.float() * scale\n"
        ))
        request["args"] = [{"kind": "tuple", "items": [
            {"kind": "tensor", "shape": [1, 4], "dtype": "float32", "generator": "random_normal"},
            {"kind": "tensor", "shape": [1, 4], "dtype": "float32", "generator": "random_normal"},
        ]}]
        request["kwargs"] = {
            "mask": {"kind": "tensor", "shape": [1, 4], "dtype": "int64", "generator": "random_integer", "integer_max_exclusive": 2},
            "scale": {"kind": "float", "value": 0.5},
        }
        self.assertEqual(self.run_request(request)["type"], "success")

    def test_provider_accepts_legacy_tensor_sequence(self):
        request = self.request("provider-legacy", (
            "import torch\ndef netviz_example_inputs(): return (torch.ones(1, 4),)\n"
            "class UserModel(torch.nn.Module):\n    def forward(self, x): return x.relu()\n"
        ))
        request["input_provider"] = {"function_name": "netviz_example_inputs", "parameter_names": ["x"]}
        self.assertEqual(self.run_request(request)["type"], "success")

    def test_provider_requires_explicit_wrapper_for_kwargs(self):
        request = self.request("provider-structured", (
            "import torch\ndef netviz_example_inputs():\n"
            "    return {'args': ((torch.ones(1, 4), torch.ones(1, 4)),), 'kwargs': {'scale': 2.0}}\n"
            "class UserModel(torch.nn.Module):\n"
            "    def forward(self, pair, *, scale): return (pair[0] + pair[1]) * scale\n"
        ))
        request["input_provider"] = {"function_name": "netviz_example_inputs", "parameter_names": ["pair"]}
        self.assertEqual(self.run_request(request)["type"], "success")

        request = self.request("provider-ambiguous", (
            "import torch\ndef netviz_example_inputs(): return {'x': torch.ones(1, 4)}\n"
            "class UserModel(torch.nn.Module):\n    def forward(self, x): return x\n"
        ))
        request["input_provider"] = {"function_name": "netviz_example_inputs", "parameter_names": ["x"]}
        self.assertEqual(self.run_request(request)["error"]["code"], "example_input_invalid")


if __name__ == "__main__":
    unittest.main()
