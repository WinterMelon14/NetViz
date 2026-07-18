import tempfile
import unittest
from pathlib import Path

from desktop.source_handles import SourceHandles
from desktop.user_trace_constants import MAX_SOURCE_FILE_BYTES


class SourceHandlesTests(unittest.TestCase):
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
        result = SourceHandles(picker=lambda: None).select()
        self.assertEqual(result, {"ok": True, "selected": None})

    def test_selection_returns_opaque_normalized_descriptor(self):
        result = SourceHandles(picker=lambda: str(self.source)).select()
        descriptor = result["selected"]
        self.assertTrue(result["ok"])
        self.assertEqual(descriptor["displayName"], "model.py")
        self.assertEqual(descriptor["kind"], "file")
        self.assertEqual(descriptor["sizeBytes"], self.source.stat().st_size)
        self.assertNotIn("path", descriptor)
        self.assertNotIn(str(self.root), str(descriptor))

    def test_non_python_file_is_rejected(self):
        text_file = self.root / "model.txt"
        text_file.write_text("text", encoding="utf-8")
        result = SourceHandles(picker=lambda: str(text_file)).select()
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "selected_file_invalid")

    def test_selected_file_is_inspected_without_importing(self):
        sentinel = self.root / "imported.txt"
        self.source.write_text(
            f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('imported')\n"
            "import torch\nclass Model(torch.nn.Module):\n    def forward(self, x): return x\n",
            encoding="utf-8",
        )
        files = SourceHandles(picker=lambda: str(self.source))
        descriptor = files.select()["selected"]
        result = files.inspect(descriptor["sourceId"])
        self.assertTrue(result["ok"])
        self.assertEqual(result["candidates"][0]["className"], "Model")
        self.assertFalse(sentinel.exists())

    def test_oversized_file_is_rejected_before_read(self):
        self.source.write_bytes(b"x" * (MAX_SOURCE_FILE_BYTES + 1))
        files = SourceHandles(picker=lambda: str(self.source))
        descriptor = files.select()["selected"]
        result = files.inspect(descriptor["sourceId"])
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "source_too_large")

    def test_syntax_and_missing_handle_failures_are_structured(self):
        self.source.write_text("def broken(:\n", encoding="utf-8")
        files = SourceHandles(picker=lambda: str(self.source))
        descriptor = files.select()["selected"]
        syntax = files.inspect(descriptor["sourceId"])
        missing = files.inspect("not-selected")
        self.assertEqual(syntax["error"]["code"], "source_syntax_error")
        self.assertEqual(missing["error"]["code"], "source_inspection_failed")

    def test_trace_request_resolves_only_registered_handle(self):
        files = SourceHandles(picker=lambda: str(self.source))
        descriptor = files.select()["selected"]
        inspection = files.inspect(descriptor["sourceId"])
        request = {
            "run_id": "selected-run",
            "source": {
                "source_id": descriptor["sourceId"],
                "class_name": "Model",
                "content_sha256": inspection["sourceIdentity"]["contentSha256"],
            },
            "constructor": {"args": [], "kwargs": {}},
            "inputs": [],
        }
        resolved = files.trace_request(request)
        self.assertEqual(resolved["source"]["file_path"], str(self.source.resolve()))
        request["source"]["source_id"] = str(self.source)
        rejected = files.trace_request(request)
        self.assertEqual(rejected["error"]["code"], "source_unavailable")

    def test_changed_source_requires_reinspection(self):
        files = SourceHandles(picker=lambda: str(self.source))
        descriptor = files.select()["selected"]
        inspection = files.inspect(descriptor["sourceId"])
        self.source.write_text("class Different: pass\n", encoding="utf-8")
        result = files.trace_request({
            "run_id": "changed-source",
            "source": {
                "source_id": descriptor["sourceId"],
                "class_name": "Model",
                "content_sha256": inspection["sourceIdentity"]["contentSha256"],
            },
            "constructor": {"args": [], "kwargs": {}},
            "inputs": [],
        })
        self.assertEqual(result["error"]["code"], "source_changed")

    def test_missing_selected_file_is_reported_clearly(self):
        files = SourceHandles(picker=lambda: str(self.source))
        descriptor = files.select()["selected"]
        self.source.unlink()
        result = files.inspect(descriptor["sourceId"])
        self.assertEqual(result["error"]["code"], "source_inspection_failed")
        self.assertIn("no longer exists", result["error"]["message"])

    def test_inline_registration_and_inspection_do_not_execute_source(self):
        sentinel = self.root / "inline-imported.txt"
        source = (
            f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('imported')\n"
            "import torch\nclass Model(torch.nn.Module):\n    def forward(self, x): return x\n"
        )
        files = SourceHandles(temp_root=self.root / "handles")

        registered = files.register_inline({"sourceText": source})
        descriptor = registered["source"]
        inspected = files.inspect(descriptor["sourceId"])

        self.assertEqual(descriptor["kind"], "inline")
        self.assertEqual(descriptor["displayName"], "pasted_model.py")
        self.assertEqual(inspected["candidates"][0]["className"], "Model")
        self.assertFalse(sentinel.exists())

    def test_inline_registration_rejects_empty_nul_and_oversized_source(self):
        files = SourceHandles(temp_root=self.root / "handles")

        self.assertEqual(files.register_inline("source")["error"]["code"], "inline_source_invalid")
        self.assertEqual(files.register_inline({"sourceText": "x", "unknown": True})["error"]["code"], "inline_source_invalid")
        self.assertEqual(files.register_inline({"sourceText": " "})["error"]["code"], "inline_source_empty")
        self.assertEqual(files.register_inline({"sourceText": "x\x00y"})["error"]["code"], "inline_source_invalid")
        self.assertEqual(files.register_inline({"sourceText": "x" * (MAX_SOURCE_FILE_BYTES + 1)})["error"]["code"], "source_too_large")

    def test_replacing_releasing_and_closing_inline_sources_clean_up_files(self):
        handles_root = self.root / "handles"
        files = SourceHandles(temp_root=handles_root)
        first = files.register_inline({"sourceText": "class First: pass\n"})["source"]
        first_path = next(handles_root.iterdir())

        second = files.register_inline({"sourceText": "class Second: pass\n"})["source"]
        second_path = next(handles_root.iterdir())
        self.assertFalse(first_path.exists())
        self.assertNotEqual(first["sourceId"], second["sourceId"])

        self.assertTrue(files.release(second["sourceId"])["released"])
        self.assertFalse(second_path.exists())
        third = files.register_inline({"sourceText": "class Third: pass\n"})["source"]
        self.assertTrue(list(handles_root.iterdir()))
        files.close()
        self.assertFalse(list(handles_root.iterdir()))
        self.assertEqual(files.inspect(third["sourceId"])["error"]["code"], "source_inspection_failed")

    def test_inline_identity_is_exact_and_required_for_trace_resolution(self):
        files = SourceHandles(temp_root=self.root / "handles")
        source = "import torch\r\nclass Model(torch.nn.Module):\r\n    def forward(self, x): return x\r\n"
        descriptor = files.register_inline({"sourceText": source})["source"]
        before_inspection = files.trace_request({
            "run_id": "inline-before-inspection",
            "source": {
                "source_id": descriptor["sourceId"],
                "class_name": "Model",
                "content_sha256": "0" * 64,
            },
            "constructor": {"args": [], "kwargs": {}},
            "inputs": [],
        })
        inspection = files.inspect(descriptor["sourceId"])
        request = {
            "run_id": "inline-run",
            "source": {
                "source_id": descriptor["sourceId"],
                "class_name": "Model",
                "content_sha256": inspection["sourceIdentity"]["contentSha256"],
            },
            "constructor": {"args": [], "kwargs": {}},
            "inputs": [],
        }

        resolved = files.trace_request(request)

        self.assertEqual(before_inspection["error"]["code"], "source_reinspection_required")
        self.assertEqual(Path(resolved["source"]["file_path"]).read_bytes(), source.encode("utf-8"))
        self.assertEqual(resolved["source"]["content_sha256"], inspection["sourceIdentity"]["contentSha256"])

    def test_replacing_inline_source_invalidates_the_previous_handle(self):
        files = SourceHandles(temp_root=self.root / "handles")
        first = files.register_inline({"sourceText": "import torch\nclass First(torch.nn.Module):\n    def forward(self, x): return x\n"})["source"]
        first_inspection = files.inspect(first["sourceId"])
        files.register_inline({"sourceText": "import torch\nclass Second(torch.nn.Module):\n    def forward(self, x): return x\n"})

        rejected = files.trace_request({
            "run_id": "replaced-inline",
            "source": {
                "source_id": first["sourceId"],
                "class_name": "First",
                "content_sha256": first_inspection["sourceIdentity"]["contentSha256"],
            },
            "constructor": {"args": [], "kwargs": {}},
            "inputs": [],
        })

        self.assertEqual(rejected["error"]["code"], "source_unavailable")


if __name__ == "__main__":
    unittest.main()
