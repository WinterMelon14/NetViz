import ast
import importlib.util
import hashlib
import inspect
import sys
import os
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any
from uuid import uuid4

from desktop.structured_inputs import StructuredInputError, construct_structured_call, validate_provider_result


class UserTraceRuntimeError(RuntimeError):
    def __init__(
        self,
        code: str,
        title: str,
        message: str,
        stage: str,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.title = title
        self.message = message
        self.stage = stage
        self.details = details or {}


def _contains_call(node: ast.AST) -> bool:
    return any(isinstance(child, ast.Call) for child in ast.walk(node))


def _keep_top_level_statement(node: ast.stmt, index: int) -> bool:
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return True
    if isinstance(node, ast.Expr):
        return index == 0 and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        value = node.value
        return value is not None and not _contains_call(value)
    return False


def sanitize_user_source(source_text: str) -> str:
    """Suppress unrelated top-level execution while preserving source line numbers."""
    tree = ast.parse(source_text)
    lines = source_text.splitlines(keepends=True)
    for index, node in enumerate(tree.body):
        if _keep_top_level_statement(node, index):
            continue
        start = node.lineno - 1
        end = (node.end_lineno or node.lineno) - 1
        first_ending = "\r\n" if lines[start].endswith("\r\n") else "\n" if lines[start].endswith("\n") else ""
        lines[start] = f"pass{first_ending}"
        for line_index in range(start + 1, end + 1):
            ending = "\r\n" if lines[line_index].endswith("\r\n") else "\n" if lines[line_index].endswith("\n") else ""
            lines[line_index] = ending
    return "".join(lines)


def _verify_source_hash(path: Path, expected_sha256: str, message: str) -> None:
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise UserTraceRuntimeError(
            "source_changed",
            "Model source changed",
            message,
            "source_identity",
            {"expected_sha256": expected_sha256, "actual_sha256": actual_sha256},
        )


def verify_project_files(project_context: dict[str, Any]) -> None:
    project_root = Path(project_context["project_root"])
    for collection, code, title in (
        ("local_modules", "project_module_changed", "Local project module changed"),
        ("resources", "project_resource_changed", "Project resource changed"),
    ):
        for descriptor in project_context.get(collection, []):
            path = (project_root / descriptor["path"]).resolve()
            if not descriptor.get("exists"):
                if not path.is_file():
                    raise UserTraceRuntimeError(
                        "project_resource_missing" if collection == "resources" else "project_module_missing",
                        "Declared project file is missing",
                        f"{descriptor['path']} was not found under the selected project root.",
                        "project_identity",
                        {"path": descriptor["path"]},
                    )
                continue
            actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_sha256 != descriptor["content_sha256"]:
                raise UserTraceRuntimeError(
                    code,
                    title,
                    f"{descriptor['path']} changed after inspection. Inspect the project again before tracing.",
                    "project_identity",
                    {"path": descriptor["path"], "expected_sha256": descriptor["content_sha256"], "actual_sha256": actual_sha256},
                )


@contextmanager
def project_execution_context(project_context: dict[str, Any]):
    project_root = str(Path(project_context["project_root"]))
    working_directory = Path(project_context["working_directory"])
    previous_cwd = Path.cwd()
    previous_sys_path = list(sys.path)
    try:
        os.chdir(working_directory)
        sys.path = [project_root, *[item for item in sys.path if item != project_root]]
        yield
    finally:
        sys.path = previous_sys_path
        os.chdir(previous_cwd)


@contextmanager
def load_sanitized_user_module(file_path: str, run_id: str, expected_sha256: str, working_directory: Path):
    source_path = Path(file_path)
    sanitized_path = working_directory / f"sanitized-model-{uuid4().hex}.py"

    def remove_sanitized_source() -> None:
        try:
            sanitized_path.unlink(missing_ok=True)
        except OSError as exc:
            print(f"Could not remove sanitized model source {sanitized_path}: {exc}", file=sys.stderr)

    try:
        _verify_source_hash(
            source_path,
            expected_sha256,
            "The Python file differs from the inspected version. Inspect it again before tracing.",
        )
        sanitized_path.write_text(sanitize_user_source(source_path.read_text(encoding="utf-8")), encoding="utf-8")
        module = load_user_module(str(sanitized_path), run_id, hashlib.sha256(sanitized_path.read_bytes()).hexdigest())
        _verify_source_hash(
            source_path,
            expected_sha256,
            "The Python file changed while it was being imported. Inspect it again before tracing.",
        )
    except UserTraceRuntimeError:
        remove_sanitized_source()
        raise
    except BaseException as exc:
        remove_sanitized_source()
        raise UserTraceRuntimeError(
            "module_import_failed",
            "Model file could not be imported",
            str(exc) or type(exc).__name__,
            "module_import",
            {"file_path": str(source_path)},
        ) from exc

    try:
        yield module
    finally:
        remove_sanitized_source()


@contextmanager
def load_project_user_module(file_path: str, run_id: str, expected_sha256: str, scratch_directory: Path, project_context: dict[str, Any]):
    source_path = Path(file_path).resolve()
    sanitized_path = scratch_directory / f"sanitized-model-{uuid4().hex}.py"

    def remove_sanitized_source() -> None:
        try:
            sanitized_path.unlink(missing_ok=True)
        except OSError as exc:
            print(f"Could not remove sanitized model source {sanitized_path}: {exc}", file=sys.stderr)

    try:
        _verify_source_hash(
            source_path,
            expected_sha256,
            "The Python file differs from the inspected version. Inspect it again before tracing.",
        )
        verify_project_files(project_context)
        sanitized_path.write_text(sanitize_user_source(source_path.read_text(encoding="utf-8")), encoding="utf-8")
        with project_execution_context(project_context):
            module = load_user_module(
                str(sanitized_path),
                run_id,
                hashlib.sha256(sanitized_path.read_bytes()).hexdigest(),
                module_name_from_project_entry(project_context["entry_relative_path"], run_id),
            )
        _verify_source_hash(
            source_path,
            expected_sha256,
            "The Python file changed while it was being imported. Inspect it again before tracing.",
        )
        verify_project_files(project_context)
    except UserTraceRuntimeError:
        remove_sanitized_source()
        raise
    except BaseException as exc:
        remove_sanitized_source()
        raise UserTraceRuntimeError(
            "module_import_failed",
            "Model file could not be imported",
            str(exc) or type(exc).__name__,
            "module_import",
            {"file_path": str(source_path), "project_root": project_context.get("project_root")},
        ) from exc

    try:
        yield module
    finally:
        remove_sanitized_source()


def load_user_module(file_path: str, run_id: str, expected_sha256: str, module_name: str | None = None) -> ModuleType:
    path = Path(file_path)
    import_name = module_name or f"tensor_trace_user_{run_id.replace('-', '_')}_{uuid4().hex}"
    try:
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise UserTraceRuntimeError(
                "source_changed",
                "Model source changed",
                "The Python file differs from the inspected version. Inspect it again before tracing.",
                "source_identity",
                {"expected_sha256": expected_sha256, "actual_sha256": actual_sha256},
            )
        spec = importlib.util.spec_from_file_location(import_name, path)
        if spec is None or spec.loader is None:
            raise ImportError("Python could not create a module specification for this file.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[import_name] = module
        spec.loader.exec_module(module)
        final_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if final_sha256 != expected_sha256:
            raise UserTraceRuntimeError(
                "source_changed",
                "Model source changed",
                "The Python file changed while it was being imported. Inspect it again before tracing.",
                "source_identity",
                {"expected_sha256": expected_sha256, "actual_sha256": final_sha256},
            )
        return module
    except UserTraceRuntimeError:
        raise
    except BaseException as exc:
        raise UserTraceRuntimeError(
            "module_import_failed",
            "Model file could not be imported",
            str(exc) or type(exc).__name__,
            "module_import",
            {"file_path": str(path)},
        ) from exc


def instantiate_model(module: ModuleType, class_name: str, args: list[Any], kwargs: dict[str, Any]):
    import torch

    if not hasattr(module, class_name):
        raise UserTraceRuntimeError(
            "model_class_not_found",
            "Model class was not found",
            f'The imported file does not define "{class_name}".',
            "model_resolution",
            {"class_name": class_name},
        )
    selected = getattr(module, class_name)
    if not inspect.isclass(selected):
        raise UserTraceRuntimeError(
            "model_class_invalid",
            "Selected model is not a class",
            f'"{class_name}" is not a class.',
            "model_resolution",
            {"class_name": class_name},
        )
    try:
        model = selected(*args, **kwargs)
    except BaseException as exc:
        raise UserTraceRuntimeError(
            "model_construction_failed",
            "Model could not be constructed",
            str(exc) or type(exc).__name__,
            "model_construction",
            {"class_name": class_name},
        ) from exc
    if not isinstance(model, torch.nn.Module):
        raise UserTraceRuntimeError(
            "model_instance_invalid",
            "Constructed object is not a PyTorch module",
            f'"{class_name}" did not construct a torch.nn.Module instance.',
            "model_construction",
            {"class_name": class_name},
        )
    return model


def build_tensor_inputs(input_specs: list[dict[str, Any]]):
    import torch

    tensors = []
    try:
        for input_spec in input_specs:
            if input_spec["dtype"] == "int64":
                tensors.append(torch.randint(0, input_spec["integer_max_exclusive"], tuple(input_spec["shape"]), dtype=torch.int64, device="cpu"))
            else:
                tensors.append(torch.randn(tuple(input_spec["shape"]), dtype=torch.float32, device="cpu"))
    except BaseException as exc:
        raise UserTraceRuntimeError(
            "input_construction_failed",
            "Trace input could not be constructed",
            str(exc) or type(exc).__name__,
            "input_construction",
        ) from exc
    return tensors


def build_structured_inputs(args: list[dict[str, Any]], kwargs: dict[str, dict[str, Any]]):
    try:
        return construct_structured_call(args, kwargs)
    except BaseException as exc:
        raise UserTraceRuntimeError(
            "input_construction_failed",
            "Trace input could not be constructed",
            str(exc) or type(exc).__name__,
            "input_construction",
        ) from exc


def module_name_from_project_entry(entry_relative_path: str, run_id: str) -> str:
    path = Path(entry_relative_path)
    if path.name == "__init__.py":
        parts = path.with_suffix("").parts[:-1]
    else:
        parts = path.with_suffix("").parts
    if parts and all(part.isidentifier() for part in parts):
        return ".".join(parts)
    return f"tensor_trace_user_{run_id.replace('-', '_')}_{uuid4().hex}"


def build_provider_inputs(module: ModuleType, provider: dict[str, Any]):
    function = getattr(module, provider["function_name"], None)
    if not callable(function):
        raise UserTraceRuntimeError("example_input_provider_not_found", "Example input provider was not found", "The inspected netviz_example_inputs function is unavailable.", "example_input_construction")
    try:
        values = function()
    except BaseException as exc:
        raise UserTraceRuntimeError("example_input_construction_failed", "Example inputs could not be created", str(exc) or type(exc).__name__, "example_input_construction") from exc
    try:
        return validate_provider_result(values, provider["parameter_names"])
    except StructuredInputError as exc:
        raise UserTraceRuntimeError(
            "example_input_invalid",
            "Example inputs are not supported",
            str(exc),
            "example_input_construction",
            {"path": exc.path},
        ) from exc
