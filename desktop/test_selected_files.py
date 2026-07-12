import tempfile
import unittest
from pathlib import Path

from desktop.selected_files import MAX_SOURCE_FILE_BYTES, SelectedPythonFiles


class SelectedPythonFilesTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source = self.root / "model.py"
        self.source.write_text(
            "import torch\nclass Model(torch.nn.Module):\n    def forward(self, x): return x\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_picker_cancellation_is_normal(self):
        result = SelectedPythonFiles(picker=lambda: None).select()
        self.assertEqual(result, {"ok": True, "selected": None})

    def test_selection_returns_opaque_normalized_descriptor(self):
        result = SelectedPythonFiles(picker=lambda: str(self.source)).select()
        descriptor = result["selected"]
        self.assertTrue(result["ok"])
        self.assertEqual(descriptor["fileName"], "model.py")
        self.assertEqual(descriptor["sizeBytes"], self.source.stat().st_size)
        self.assertNotIn("path", descriptor)
        self.assertNotIn(str(self.root), str(descriptor))

    def test_non_python_file_is_rejected(self):
        text_file = self.root / "model.txt"
        text_file.write_text("text", encoding="utf-8")
        result = SelectedPythonFiles(picker=lambda: str(text_file)).select()
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "selected_file_invalid")

    def test_selected_file_is_inspected_without_importing(self):
        sentinel = self.root / "imported.txt"
        self.source.write_text(
            f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('imported')\n"
            "import torch\nclass Model(torch.nn.Module):\n    def forward(self, x): return x\n",
            encoding="utf-8",
        )
        files = SelectedPythonFiles(picker=lambda: str(self.source))
        descriptor = files.select()["selected"]
        result = files.inspect(descriptor["selectionId"])
        self.assertTrue(result["ok"])
        self.assertEqual(result["candidates"][0]["className"], "Model")
        self.assertFalse(sentinel.exists())

    def test_oversized_file_is_rejected_before_read(self):
        self.source.write_bytes(b"x" * (MAX_SOURCE_FILE_BYTES + 1))
        files = SelectedPythonFiles(picker=lambda: str(self.source))
        descriptor = files.select()["selected"]
        result = files.inspect(descriptor["selectionId"])
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "source_too_large")

    def test_syntax_and_missing_handle_failures_are_structured(self):
        self.source.write_text("def broken(:\n", encoding="utf-8")
        files = SelectedPythonFiles(picker=lambda: str(self.source))
        descriptor = files.select()["selected"]
        syntax = files.inspect(descriptor["selectionId"])
        missing = files.inspect("not-selected")
        self.assertEqual(syntax["error"]["code"], "source_syntax_error")
        self.assertEqual(missing["error"]["code"], "source_inspection_failed")

    def test_trace_request_resolves_only_registered_handle(self):
        files = SelectedPythonFiles(picker=lambda: str(self.source))
        descriptor = files.select()["selected"]
        request = {
            "run_id": "selected-run",
            "source": {"selection_id": descriptor["selectionId"], "class_name": "Model"},
            "constructor": {"args": [], "kwargs": {}},
            "inputs": [],
        }
        resolved = files.trace_request(request)
        self.assertEqual(resolved["source"]["file_path"], str(self.source.resolve()))
        request["source"]["selection_id"] = str(self.source)
        rejected = files.trace_request(request)
        self.assertEqual(rejected["error"]["code"], "selected_file_unavailable")


if __name__ == "__main__":
    unittest.main()
