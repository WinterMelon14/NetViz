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
        self.assertEqual(result["inputs"][0]["shape"], [1, 3, 16, 16])

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

    def test_rejects_input_count_dimensions_dtype_and_generator(self):
        request = self.request()
        request["inputs"] = []
        self.assert_invalid(request, "inputs")
        request = self.request()
        request["inputs"][0]["shape"] = [1] * 9
        self.assert_invalid(request, "inputs[0].shape")
        request = self.request()
        request["inputs"][0]["dtype"] = "float64"
        self.assert_invalid(request, "inputs[0].dtype")
        request = self.request()
        request["inputs"][0]["generator"] = "zeros"
        self.assert_invalid(request, "inputs[0].generator")

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

    def test_runtime_sources_do_not_call_eval_or_exec_builtins(self):
        for filename in ("user_trace_request.py", "user_model_runtime.py", "trace_worker.py"):
            tree = ast.parse((Path(__file__).with_name(filename)).read_text(encoding="utf-8"))
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
