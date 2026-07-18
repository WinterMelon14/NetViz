import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from desktop.trace_worker import run_trace
from desktop.user_trace_request import (
    MAX_TENSOR_ELEMENTS,
    UserTraceRequestError,
    validate_user_trace_request,
)


class UserTraceRequestTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source_path = self.root / "model.py"
        self.source_path.write_text("class Model: pass\n", encoding="utf-8")
        self.output_path = self.root / "result.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def request(self):
        return {
            "protocol_version": 1,
            "run_id": "request-test",
            "source": {
                "file_path": str(self.source_path),
                "class_name": "Model",
                "content_sha256": hashlib.sha256(self.source_path.read_bytes()).hexdigest(),
            },
            "constructor": {"args": [], "kwargs": {}},
            "inputs": [{
                "kind": "tensor",
                "parameter_name": "x",
                "shape": [1, 3, 16, 16],
                "dtype": "float32",
                "generator": "random_normal",
            }],
            "output_path": str(self.output_path),
        }

    def assert_invalid(self, request, path):
        with self.assertRaises(UserTraceRequestError) as raised:
            validate_user_trace_request(request, expected_output_path=self.output_path)
        self.assertEqual(raised.exception.path, path)

    def test_valid_request_is_normalized(self):
        result = validate_user_trace_request(self.request(), expected_output_path=self.output_path)
        self.assertEqual(result["input_schema_version"], 1)
        self.assertEqual(result["inputs"][0]["shape"], [1, 3, 16, 16])

    def test_explicit_v1_and_v2_are_never_inferred_from_shape(self):
        request = self.request()
        request["input_schema_version"] = 1
        self.assertEqual(validate_user_trace_request(request, expected_output_path=self.output_path)["input_schema_version"], 1)
        request = self.request()
        request.pop("inputs")
        request.update({"args": [], "kwargs": {}})
        self.assert_invalid(request, "request.args")
        request["input_schema_version"] = 2
        result = validate_user_trace_request(request, expected_output_path=self.output_path)
        self.assertEqual(result["input_schema_version"], 2)
        request["inputs"] = []
        self.assert_invalid(request, "request.inputs")

    def test_schema_v2_accepts_nested_args_kwargs_and_preserves_containers(self):
        request = self.request()
        request.pop("inputs")
        request.update({
            "input_schema_version": 2,
            "args": [{"kind": "tuple", "items": [
                {"kind": "tensor", "shape": [2, 3], "dtype": "float32", "generator": "random_normal"},
                {"kind": "integer", "value": 4},
            ]}],
            "kwargs": {"options": {"kind": "dict", "entries": [
                {"key": "enabled", "value": {"kind": "boolean", "value": True}},
                {"key": "name", "value": {"kind": "string", "value": "demo"}},
                {"key": "missing", "value": {"kind": "none"}},
            ]}},
        })
        result = validate_user_trace_request(request, expected_output_path=self.output_path)
        self.assertEqual(result["args"][0]["kind"], "tuple")
        self.assertEqual(result["kwargs"]["options"]["entries"][2]["value"], {"kind": "none"})

    def test_shared_schema_v2_call_fixture_is_valid(self):
        fixture_path = Path(__file__).parent / "fixtures" / "structured_input_call_v2.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        request = self.request()
        request.pop("inputs")
        request.update(fixture)
        result = validate_user_trace_request(request, expected_output_path=self.output_path)
        self.assertEqual(result["args"], fixture["args"])
        self.assertEqual(result["kwargs"], fixture["kwargs"])

    def test_schema_v2_rejects_nonfinite_scalars_and_reports_nested_paths(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                request = self.request()
                request.pop("inputs")
                request.update({"input_schema_version": 2, "args": [], "kwargs": {"scale": {"kind": "float", "value": value}}})
                self.assert_invalid(request, "kwargs.scale.value")
        request = self.request()
        request.pop("inputs")
        request.update({"input_schema_version": 2, "args": [], "kwargs": {"config": {"kind": "dict", "entries": [
            {"key": "mask", "value": {"kind": "tensor", "shape": [0], "dtype": "int64", "generator": "random_integer"}},
        ]}}})
        self.assert_invalid(request, "kwargs.config.entries[0].value.shape[0]")

    def test_schema_v2_rejects_duplicate_dictionary_keys_and_provider_values(self):
        request = self.request()
        request.pop("inputs")
        request.update({"input_schema_version": 2, "args": [{"kind": "dict", "entries": [
            {"key": "x", "value": {"kind": "integer", "value": 1}},
            {"key": "x", "value": {"kind": "integer", "value": 2}},
        ]}], "kwargs": {}})
        self.assert_invalid(request, "args[0].entries[1].key")
        request = self.request()
        request.pop("inputs")
        request.update({"input_schema_version": 2, "args": [{"kind": "integer", "value": 1}], "kwargs": {}, "input_provider": {"function_name": "netviz_example_inputs", "parameter_names": []}})
        self.assert_invalid(request, "args")

    def test_accepts_bounded_int64_random_inputs(self):
        request = self.request()
        request["inputs"][0].update({
            "dtype": "int64",
            "generator": "random_integer",
            "integer_max_exclusive": 17,
        })
        result = validate_user_trace_request(request, expected_output_path=self.output_path)
        self.assertEqual(result["inputs"][0]["integer_max_exclusive"], 17)

        request["inputs"][0]["integer_max_exclusive"] = 0
        self.assert_invalid(request, "inputs[0].integer_max_exclusive")

    def test_rejects_protocol_run_source_and_unknown_fields(self):
        cases = [
            ("protocol_version", 2, "protocol_version"),
            ("run_id", "", "run_id"),
        ]
        for field, value, path in cases:
            with self.subTest(path=path):
                request = self.request()
                request[field] = value
                self.assert_invalid(request, path)
        request = self.request()
        request["executable"] = "model()"
        self.assert_invalid(request, "request.executable")

    def test_profile_request_is_explicit_and_bounded(self):
        request = self.request()
        request["trace_mode"] = "profile"
        request["profile"] = {"warmup_runs": 1, "measurement_runs": 3, "percentiles": [50, 90]}
        result = validate_user_trace_request(request, expected_output_path=self.output_path)
        self.assertEqual(result["trace_mode"], "profile")
        self.assertEqual(result["profile"]["measurement_runs"], 3)

        request = self.request()
        request["profile"] = {"measurement_runs": 3}
        self.assert_invalid(request, "profile")

        request = self.request()
        request["trace_mode"] = "profile"
        request["profile"] = {"measurement_runs": 0}
        self.assert_invalid(request, "profile.measurement_runs")

    def test_rejects_invalid_source_path_and_class(self):
        request = self.request()
        request["source"]["file_path"] = str(self.root / "missing.py")
        self.assert_invalid(request, "source.file_path")
        request = self.request()
        request["source"]["class_name"] = "Model()"
        self.assert_invalid(request, "source.class_name")

    def test_accepts_json_safe_constructor_values(self):
        request = self.request()
        request["constructor"] = {
            "args": [None, True, 3, 1.5, "name", [1, 2]],
            "kwargs": {"config": {"enabled": False}},
        }
        result = validate_user_trace_request(request, expected_output_path=self.output_path)
        self.assertEqual(result["constructor"], request["constructor"])

    def test_rejects_non_json_constructor_values_and_excessive_depth(self):
        request = self.request()
        request["constructor"]["args"] = [{"bad": {1, 2}}]
        self.assert_invalid(request, "constructor.args[0].bad")
        request = self.request()
        nested = None
        for _ in range(9):
            nested = [nested]
        request["constructor"]["args"] = [nested]
        self.assert_invalid(request, "constructor.args[0][0][0][0][0][0][0][0][0]")

    def test_accepts_zero_and_multiple_inputs_and_rejects_excessive_count(self):
        request = self.request()
        request["inputs"] = []
        self.assertEqual(validate_user_trace_request(request, expected_output_path=self.output_path)["inputs"], [])
        request = self.request()
        request["inputs"] = request["inputs"] * 2
        request["inputs"][1] = {**request["inputs"][1], "parameter_name": "y"}
        self.assertEqual(len(validate_user_trace_request(request, expected_output_path=self.output_path)["inputs"]), 2)
        request["inputs"] = request["inputs"] * 5
        self.assert_invalid(request, "inputs")

    def test_rejects_dimensions_dtype_and_generator(self):
        request = self.request()
        request = self.request()
        request["inputs"][0]["shape"] = [1] * 9
        self.assert_invalid(request, "inputs[0].shape")
        request = self.request()
        request["inputs"][0]["dtype"] = "float64"
        self.assert_invalid(request, "inputs[0].dtype")
        request = self.request()
        request["inputs"][0]["generator"] = "zeros"
        self.assert_invalid(request, "inputs[0].generator")

    def test_rejects_combined_input_allocation(self):
        request = self.request()
        request["inputs"] = [
            {**request["inputs"][0], "parameter_name": "left", "shape": [10_000_000]},
            {**request["inputs"][0], "parameter_name": "right", "shape": [10_000_000]},
        ]
        self.assert_invalid(request, "inputs")

    def test_rejects_negative_non_integer_and_oversized_dimensions(self):
        for dimension in (-1, 0, 1.5, True):
            with self.subTest(dimension=dimension):
                request = self.request()
                request["inputs"][0]["shape"] = [dimension]
                self.assert_invalid(request, "inputs[0].shape[0]")
        request = self.request()
        request["inputs"][0]["shape"] = [MAX_TENSOR_ELEMENTS + 1]
        self.assert_invalid(request, "inputs[0].shape")

    def test_rejects_non_host_output_path(self):
        request = self.request()
        request["output_path"] = str(self.root / "attacker-selected.json")
        self.assert_invalid(request, "output_path")

    def test_invalid_request_does_not_import_user_code(self):
        sentinel = self.root / "imported.txt"
        self.source_path.write_text(
            f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('imported')\n",
            encoding="utf-8",
        )
        request = self.request()
        request["inputs"][0]["shape"] = [-1]
        request_path = self.root / "request.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")

        result = run_trace(str(request_path))

        self.assertEqual(result["error"]["code"], "user_trace_request_invalid")
        self.assertFalse(sentinel.exists())

    def test_project_context_supports_package_relative_imports_and_change_detection(self):
        package = self.root / "pkg"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        layer_path = package / "layers.py"
        model_path = package / "model.py"
        layer_path.write_text("import torch\nclass Block(torch.nn.Module):\n    def forward(self, x): return x.relu()\n", encoding="utf-8")
        model_path.write_text(
            "import torch\nfrom .layers import Block\n"
            "class UserModel(torch.nn.Module):\n"
            "    def __init__(self):\n        super().__init__()\n        self.block = Block()\n"
            "    def forward(self, x): return self.block(x)\n",
            encoding="utf-8",
        )
        request = self.request()
        request["source"] = {
            "file_path": str(model_path),
            "class_name": "UserModel",
            "content_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        }
        request["project_context"] = {
            "project_root": str(self.root),
            "working_directory": str(self.root),
            "entry_relative_path": "pkg/model.py",
            "local_modules": [{
                "path": "pkg/layers.py",
                "content_sha256": hashlib.sha256(layer_path.read_bytes()).hexdigest(),
                "size_bytes": layer_path.stat().st_size,
                "exists": True,
            }],
            "resources": [],
        }

        request_path = self.root / "request.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        result = run_trace(str(request_path))

        self.assertEqual(result["type"], "success")

        layer_path.write_text(layer_path.read_text(encoding="utf-8") + "\nVALUE = 1\n", encoding="utf-8")
        request["run_id"] = "project-changed"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        changed = run_trace(str(request_path))
        self.assertEqual(changed["error"]["code"], "project_module_changed")

    def test_worker_profile_mode_returns_profiling_payload(self):
        self.source_path.write_text(
            "import torch\nclass Model(torch.nn.Module):\n"
            "    def __init__(self):\n        super().__init__()\n        self.linear = torch.nn.Linear(4, 2)\n"
            "    def forward(self, x): return self.linear(x).relu()\n",
            encoding="utf-8",
        )
        request = self.request()
        request["source"]["content_sha256"] = hashlib.sha256(self.source_path.read_bytes()).hexdigest()
        request["source"]["class_name"] = "Model"
        request["inputs"][0]["shape"] = [1, 4]
        request["trace_mode"] = "profile"
        request["profile"] = {"warmup_runs": 0, "measurement_runs": 2, "percentiles": [50, 95]}
        request_path = self.root / "request.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")

        result = run_trace(str(request_path))

        self.assertEqual(result["type"], "success")
        profiling = result["trace"]["payload"]["profiling"]
        self.assertEqual(profiling["config"]["measurement_runs"], 2)
        self.assertTrue(profiling["expensive_operations"])

    def test_runtime_sources_do_not_call_eval_or_exec_builtins(self):
        desktop_root = Path(__file__).resolve().parents[1] / "desktop"
        for filename in ("user_trace_request.py", "user_model_runtime.py", "structured_inputs.py", "trace_worker.py"):
            tree = ast.parse((desktop_root / filename).read_text(encoding="utf-8"))
            forbidden = [
                node.func.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"eval", "exec"}
            ]
            self.assertEqual(forbidden, [], filename)


if __name__ == "__main__":
    unittest.main()
