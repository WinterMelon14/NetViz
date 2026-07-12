import importlib.util
import hashlib
import inspect
from pathlib import Path
from types import ModuleType
from typing import Any
from uuid import uuid4


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


def load_user_module(file_path: str, run_id: str, expected_sha256: str) -> ModuleType:
    path = Path(file_path)
    module_name = f"tensor_trace_user_{run_id.replace('-', '_')}_{uuid4().hex}"
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
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError("Python could not create a module specification for this file.")
        module = importlib.util.module_from_spec(spec)
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
            tensors.append(torch.randn(tuple(input_spec["shape"]), dtype=torch.float32, device="cpu"))
    except BaseException as exc:
        raise UserTraceRuntimeError(
            "input_construction_failed",
            "Trace input could not be constructed",
            str(exc) or type(exc).__name__,
            "input_construction",
        ) from exc
    return tensors
