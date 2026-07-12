import hashlib
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from desktop.source_inspection import inspect_model_source, inspection_error
from desktop.trace_protocol import trace_error
from desktop.user_trace_constants import MAX_SOURCE_FILE_BYTES

PythonFilePicker = Callable[[], str | None]


@dataclass(frozen=True)
class InspectedSource:
    content_sha256: str
    size_bytes: int


def native_python_file_picker() -> str | None:
    import webview

    if not webview.windows:
        return None
    selected = webview.windows[0].create_file_dialog(
        webview.FileDialog.OPEN,
        allow_multiple=False,
        file_types=("Python files (*.py)",),
    )
    return selected[0] if selected else None


class SelectedPythonFiles:
    def __init__(self, picker: PythonFilePicker = native_python_file_picker):
        self._picker = picker
        self._paths: dict[str, Path] = {}
        self._inspected: dict[str, InspectedSource] = {}

    def select(self) -> dict[str, Any]:
        try:
            selected_path = self._picker()
        except Exception as exc:
            return {
                "ok": False,
                "error": {
                    "code": "file_picker_failed",
                    "title": "Python file could not be selected",
                    "message": str(exc),
                    "stage": "file_selection",
                },
            }
        if not selected_path:
            return {"ok": True, "selected": None}

        try:
            path = self._validate_path(Path(selected_path))
        except ValueError as exc:
            return {
                "ok": False,
                "error": {
                    "code": "selected_file_invalid",
                    "title": "Selected file is not supported",
                    "message": str(exc),
                    "stage": "file_selection",
                },
            }

        selection_id = uuid4().hex
        self._paths[selection_id] = path
        return {"ok": True, "selected": self._descriptor(selection_id, path)}

    def inspect(self, selection_id: Any) -> dict[str, Any]:
        try:
            path = self._resolve(selection_id)
            size_bytes = path.stat().st_size
            if size_bytes > MAX_SOURCE_FILE_BYTES:
                return inspection_error(
                    "source_too_large",
                    "Source file is too large",
                    f"Source inspection is limited to {MAX_SOURCE_FILE_BYTES} bytes.",
                    {"maxBytes": MAX_SOURCE_FILE_BYTES, "actualBytes": size_bytes},
                )
            source_bytes = path.read_bytes()
            source_text = source_bytes.decode("utf-8")
        except (OSError, UnicodeError, ValueError) as exc:
            return inspection_error(
                "source_inspection_failed",
                "Source file could not be read",
                str(exc),
            )
        result = inspect_model_source(source_text)
        if result.get("ok") is True:
            identity = InspectedSource(hashlib.sha256(source_bytes).hexdigest(), len(source_bytes))
            self._inspected[selection_id] = identity
            result = dict(result)
            result["sourceIdentity"] = {
                "contentSha256": identity.content_sha256,
                "sizeBytes": identity.size_bytes,
            }
        else:
            self._inspected.pop(selection_id, None)
        return result

    def trace_request(self, request: Any) -> dict[str, Any] | dict[str, object]:
        if not isinstance(request, dict):
            return trace_error(
                None,
                "selected_trace_request_invalid",
                "Selected trace request is invalid",
                "The selected trace request must be an object.",
                "host_selection",
            )
        source = request.get("source")
        if not isinstance(source, dict):
            return trace_error(
                request.get("run_id"),
                "selected_trace_request_invalid",
                "Selected trace request is invalid",
                "The request must include a selected source descriptor.",
                "host_selection",
            )
        try:
            path = self._resolve(source.get("selection_id"))
        except ValueError as exc:
            return trace_error(
                request.get("run_id"),
                "selected_file_unavailable",
                "Selected Python file is unavailable",
                str(exc),
                "host_selection",
            )
        identity = self._inspected.get(source.get("selection_id"))
        requested_sha256 = source.get("content_sha256")
        if identity is None or requested_sha256 != identity.content_sha256:
            return trace_error(
                request.get("run_id"),
                "source_reinspection_required",
                "Model source must be inspected again",
                "The trace request does not match the latest inspected file contents.",
                "host_selection",
            )
        try:
            actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            return trace_error(
                request.get("run_id"),
                "selected_file_unavailable",
                "Selected Python file is unavailable",
                str(exc),
                "host_selection",
            )
        if actual_sha256 != identity.content_sha256:
            self._inspected.pop(source.get("selection_id"), None)
            return trace_error(
                request.get("run_id"),
                "source_changed",
                "Model source changed",
                "The Python file differs from the inspected version. Inspect it again before tracing.",
                "host_selection",
            )

        resolved = dict(request)
        resolved["source"] = {
            "file_path": str(path),
            "class_name": source.get("class_name"),
            "content_sha256": identity.content_sha256,
        }
        return resolved

    def _resolve(self, selection_id: Any) -> Path:
        if not isinstance(selection_id, str) or selection_id not in self._paths:
            raise ValueError("The selected file handle is missing, invalid, or expired.")
        return self._validate_path(self._paths[selection_id])

    @staticmethod
    def _validate_path(path: Path) -> Path:
        if path.suffix.lower() != ".py":
            raise ValueError("Only .py files can be selected.")
        try:
            file_stat = path.stat()
        except OSError as exc:
            raise ValueError("The selected file no longer exists.") from exc
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("The selected path must be a regular Python file.")
        return path.resolve()

    @staticmethod
    def _descriptor(selection_id: str, path: Path) -> dict[str, Any]:
        return {
            "selectionId": selection_id,
            "fileName": path.name,
            "sizeBytes": path.stat().st_size,
        }
