import hashlib
import stat
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from desktop.source_inspection import inspect_model_source, inspection_error
from desktop.trace_protocol import trace_error
from desktop.user_trace_constants import MAX_SOURCE_CHARS, MAX_SOURCE_DISPLAY_NAME_CHARS, MAX_SOURCE_FILE_BYTES

PythonFilePicker = Callable[[], str | None]


def _relative_project_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _frontend_project_files(project_context: Any, field: str) -> list[dict[str, Any]]:
    if not isinstance(project_context, dict) or not isinstance(project_context.get(field), list):
        return []
    items = []
    for item in project_context[field]:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            continue
        descriptor = {
            "path": item["path"],
            "exists": item.get("exists") is True,
        }
        if isinstance(item.get("contentSha256"), str):
            descriptor["content_sha256"] = item["contentSha256"]
        if isinstance(item.get("sizeBytes"), int):
            descriptor["size_bytes"] = item["sizeBytes"]
        items.append(descriptor)
    return items


@dataclass(frozen=True)
class SourceHandle:
    id: str
    kind: str
    display_name: str
    size_bytes: int
    source_path: Path
    content_sha256: str | None = None
    project_root: Path | None = None


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


class SourceHandles:
    def __init__(
        self,
        picker: PythonFilePicker = native_python_file_picker,
        temp_root: Path | None = None,
    ):
        self._picker = picker
        self._temp_directory = tempfile.TemporaryDirectory(prefix="netviz-sources-") if temp_root is None else None
        self._temp_root = Path(self._temp_directory.name) if self._temp_directory else Path(temp_root)
        self._temp_root.mkdir(parents=True, exist_ok=True)
        self._handles: dict[str, SourceHandle] = {}
        self._inspected: dict[str, str] = {}
        self._lock = threading.RLock()

    def __del__(self):
        self.close()

    def select(self) -> dict[str, Any]:
        try:
            selected_path = self._picker()
        except Exception as exc:
            return self._error("file_picker_failed", "Python file could not be selected", str(exc), "file_selection")
        if not selected_path:
            return {"ok": True, "selected": None}

        try:
            path = self._validate_file_path(Path(selected_path))
        except ValueError as exc:
            return self._error("selected_file_invalid", "Selected file is not supported", str(exc), "file_selection")

        handle = SourceHandle(uuid4().hex, "file", path.name, path.stat().st_size, path, project_root=path.parent)
        with self._lock:
            self._handles[handle.id] = handle
        return {"ok": True, "selected": self._descriptor(handle)}

    def register_inline(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, dict):
            return self._error("inline_source_invalid", "Pasted source is invalid", "The registration request must be an object.", "source_registration")
        unknown = sorted(set(request) - {"sourceText", "displayName"})
        if unknown:
            return self._error("inline_source_invalid", "Pasted source is invalid", f"{unknown[0]} is not a supported field.", "source_registration")
        source_text = request.get("sourceText")
        if not isinstance(source_text, str) or not source_text.strip():
            return self._error("inline_source_empty", "Pasted source is empty", "Enter Python source before inspecting it.", "source_registration")
        if "\x00" in source_text:
            return self._error("inline_source_invalid", "Pasted source is invalid", "Python source cannot contain NUL characters.", "source_registration")
        if len(source_text) > MAX_SOURCE_CHARS:
            return self._source_too_large(len(source_text), None)
        try:
            source_bytes = source_text.encode("utf-8")
        except UnicodeEncodeError as exc:
            return self._error("inline_source_invalid", "Pasted source is invalid", str(exc), "source_registration")
        if len(source_bytes) > MAX_SOURCE_FILE_BYTES:
            return self._source_too_large(len(source_text), len(source_bytes))

        display_name = request.get("displayName", "pasted_model.py")
        if not isinstance(display_name, str) or not display_name.strip():
            display_name = "pasted_model.py"
        display_name = Path(display_name).name
        if not display_name.lower().endswith(".py"):
            display_name += ".py"
        display_name = display_name[:MAX_SOURCE_DISPLAY_NAME_CHARS]

        handle_id = uuid4().hex
        path = self._temp_root / f"{handle_id}.py"
        try:
            path.write_bytes(source_bytes)
        except OSError as exc:
            return self._error("inline_source_write_failed", "Pasted source could not be stored", str(exc), "source_registration")
        handle = SourceHandle(
            handle_id,
            "inline",
            display_name,
            len(source_bytes),
            path,
            hashlib.sha256(source_bytes).hexdigest(),
            path.parent,
        )
        with self._lock:
            previous_inline_ids = [source_id for source_id, item in self._handles.items() if item.kind == "inline"]
            for source_id in previous_inline_ids:
                self._release_locked(source_id)
            self._handles[handle.id] = handle
        return {"ok": True, "source": self._descriptor(handle)}

    def inspect(self, source_id: Any) -> dict[str, Any]:
        try:
            handle = self._resolve(source_id)
            source_bytes = handle.source_path.read_bytes()
            if len(source_bytes) > MAX_SOURCE_FILE_BYTES:
                return inspection_error(
                    "source_too_large",
                    "Source file is too large",
                    f"Source inspection is limited to {MAX_SOURCE_FILE_BYTES} bytes.",
                    {"maxBytes": MAX_SOURCE_FILE_BYTES, "actualBytes": len(source_bytes)},
                )
            source_text = source_bytes.decode("utf-8")
        except (OSError, UnicodeError, ValueError) as exc:
            return inspection_error("source_inspection_failed", "Source could not be read", str(exc))

        result = inspect_model_source(source_text, handle.source_path, handle.project_root)
        if result.get("ok") is not True:
            with self._lock:
                self._inspected.pop(source_id, None)
                current = self._handles.get(source_id)
                if current:
                    self._handles[source_id] = SourceHandle(
                        current.id, current.kind, current.display_name, current.size_bytes, current.source_path, None, current.project_root
                    )
            return result

        identity = hashlib.sha256(source_bytes).hexdigest()
        inspected_handle = SourceHandle(
            handle.id, handle.kind, handle.display_name, len(source_bytes), handle.source_path, identity, handle.project_root
        )
        with self._lock:
            self._handles[handle.id] = inspected_handle
            self._inspected[handle.id] = identity
        normalized = dict(result)
        normalized["sourceIdentity"] = {"contentSha256": identity, "sizeBytes": len(source_bytes)}
        return normalized

    def trace_request(self, request: Any) -> dict[str, Any]:
        if not isinstance(request, dict):
            return trace_error(None, "source_trace_request_invalid", "Trace request is invalid", "The trace request must be an object.", "host_source")
        source = request.get("source")
        if not isinstance(source, dict):
            return trace_error(request.get("run_id"), "source_trace_request_invalid", "Trace request is invalid", "The request must include a source descriptor.", "host_source")
        source_id = source.get("source_id")
        try:
            handle = self._resolve(source_id)
        except ValueError as exc:
            return trace_error(request.get("run_id"), "source_unavailable", "Python source is unavailable", str(exc), "host_source")
        inspected_sha256 = self._inspected.get(handle.id)
        requested_sha256 = source.get("content_sha256")
        if inspected_sha256 is None or requested_sha256 != inspected_sha256:
            return trace_error(request.get("run_id"), "source_reinspection_required", "Model source must be inspected again", "The trace request does not match the latest inspected source.", "host_source")
        try:
            actual_sha256 = hashlib.sha256(handle.source_path.read_bytes()).hexdigest()
        except OSError as exc:
            return trace_error(request.get("run_id"), "source_unavailable", "Python source is unavailable", str(exc), "host_source")
        if actual_sha256 != inspected_sha256:
            with self._lock:
                self._inspected.pop(handle.id, None)
                current = self._handles.get(handle.id)
                if current:
                    self._handles[handle.id] = SourceHandle(current.id, current.kind, current.display_name, current.size_bytes, current.source_path, None)
            return trace_error(request.get("run_id"), "source_changed", "Model source changed", "The Python source differs from the inspected version. Inspect it again before tracing.", "host_source")

        resolved = dict(request)
        resolved["source"] = {
            "file_path": str(handle.source_path),
            "class_name": source.get("class_name"),
            "content_sha256": inspected_sha256,
        }
        project_context = request.get("project_context")
        if handle.project_root is not None:
            local_modules = _frontend_project_files(project_context, "localModules")
            resources = _frontend_project_files(project_context, "resources")
            resolved["project_context"] = {
                "project_root": str(handle.project_root),
                "working_directory": str(handle.project_root),
                "entry_relative_path": _relative_project_path(handle.source_path, handle.project_root),
                "local_modules": local_modules,
                "resources": resources,
            }
        return resolved

    def release(self, source_id: Any) -> dict[str, Any]:
        if not isinstance(source_id, str):
            return self._error("source_release_invalid", "Source could not be released", "The source ID must be a string.", "source_release")
        with self._lock:
            released = self._release_locked(source_id)
        return {"ok": True, "released": released}

    def display_name(self, source_id: Any) -> str | None:
        with self._lock:
            handle = self._handles.get(source_id) if isinstance(source_id, str) else None
        return handle.display_name if handle else None

    def close(self) -> None:
        with self._lock:
            for source_id in list(self._handles):
                self._release_locked(source_id)
            if self._temp_directory:
                self._temp_directory.cleanup()
                self._temp_directory = None

    def _resolve(self, source_id: Any) -> SourceHandle:
        with self._lock:
            handle = self._handles.get(source_id) if isinstance(source_id, str) else None
        if handle is None:
            raise ValueError("The source handle is missing, invalid, or expired.")
        if handle.kind == "file":
            path = self._validate_file_path(handle.source_path)
            return SourceHandle(handle.id, handle.kind, handle.display_name, path.stat().st_size, path, handle.content_sha256, handle.project_root or path.parent)
        try:
            file_stat = handle.source_path.stat()
        except OSError as exc:
            raise ValueError("The pasted source is no longer available.") from exc
        if not stat.S_ISREG(file_stat.st_mode) or handle.source_path.parent.resolve() != self._temp_root.resolve():
            raise ValueError("The pasted source path is invalid.")
        return handle

    def _release_locked(self, source_id: str) -> bool:
        handle = self._handles.pop(source_id, None)
        self._inspected.pop(source_id, None)
        if handle is None:
            return False
        if handle.kind == "inline":
            try:
                handle.source_path.unlink(missing_ok=True)
            except OSError:
                self._handles[source_id] = handle
                raise
        return True

    @staticmethod
    def _validate_file_path(path: Path) -> Path:
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
    def _descriptor(handle: SourceHandle) -> dict[str, Any]:
        descriptor = {"sourceId": handle.id, "kind": handle.kind, "displayName": handle.display_name, "sizeBytes": handle.size_bytes}
        if handle.project_root is not None:
            descriptor["projectRootDisplay"] = handle.project_root.name or str(handle.project_root)
        return descriptor

    @staticmethod
    def _error(code: str, title: str, message: str, stage: str) -> dict[str, Any]:
        return {"ok": False, "error": {"code": code, "title": title, "message": message, "stage": stage}}

    @staticmethod
    def _source_too_large(char_count: int, byte_count: int | None) -> dict[str, Any]:
        details = {"maxChars": MAX_SOURCE_CHARS, "maxBytes": MAX_SOURCE_FILE_BYTES, "actualChars": char_count}
        if byte_count is not None:
            details["actualBytes"] = byte_count
        result = SourceHandles._error("source_too_large", "Pasted source is too large", f"Pasted source is limited to {MAX_SOURCE_CHARS} characters and {MAX_SOURCE_FILE_BYTES} UTF-8 bytes.", "source_registration")
        result["error"]["details"] = details
        return result
