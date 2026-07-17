import math
import json
import stat
from pathlib import Path
from typing import Any

from desktop.trace_protocol import PROTOCOL_VERSION
from desktop.structured_inputs import StructuredInputError, validate_structured_call
from desktop.user_trace_constants import (
    DEFAULT_INTEGER_MAX_EXCLUSIVE,
    MAX_CONSTRUCTOR_LITERAL_DEPTH,
    MAX_CONSTRUCTOR_LITERAL_VALUES,
    MAX_CONSTRUCTOR_SERIALIZED_BYTES,
    MAX_CONSTRUCTOR_STRING_CHARS,
    MAX_TENSOR_DIMENSIONS,
    MAX_TENSOR_ELEMENTS,
    MAX_TOTAL_INPUT_BYTES,
    MAX_USER_INPUTS,
    SUPPORTED_TENSOR_DTYPES,
    SUPPORTED_TENSOR_GENERATORS,
    TENSOR_DTYPE_BYTES,
)


class UserTraceRequestError(ValueError):
    def __init__(self, path: str, message: str):
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise UserTraceRequestError(path, "must be an object")
    return value


def _exact_fields(value: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise UserTraceRequestError(f"{path}.{unknown[0]}", "is not a supported request field")


def _non_empty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise UserTraceRequestError(path, "must be a non-empty string")
    return value


def _validate_literal(value: Any, path: str, depth: int, counter: list[int]) -> Any:
    if depth > MAX_CONSTRUCTOR_LITERAL_DEPTH:
        raise UserTraceRequestError(path, f"exceeds maximum depth {MAX_CONSTRUCTOR_LITERAL_DEPTH}")
    counter[0] += 1
    if counter[0] > MAX_CONSTRUCTOR_LITERAL_VALUES:
        raise UserTraceRequestError(path, f"exceeds maximum value count {MAX_CONSTRUCTOR_LITERAL_VALUES}")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise UserTraceRequestError(path, "must be a finite number")
        return value
    if isinstance(value, str):
        if len(value) > MAX_CONSTRUCTOR_STRING_CHARS:
            raise UserTraceRequestError(path, f"exceeds maximum string length {MAX_CONSTRUCTOR_STRING_CHARS}")
        return value
    if isinstance(value, list):
        return [_validate_literal(item, f"{path}[{index}]", depth + 1, counter) for index, item in enumerate(value)]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise UserTraceRequestError(path, "dictionary keys must be strings")
            if len(key) > MAX_CONSTRUCTOR_STRING_CHARS:
                raise UserTraceRequestError(path, f"dictionary key exceeds maximum string length {MAX_CONSTRUCTOR_STRING_CHARS}")
            normalized[key] = _validate_literal(item, f"{path}.{key}", depth + 1, counter)
        return normalized
    raise UserTraceRequestError(path, "must be null, a boolean, finite number, string, list, or dictionary")


def _validate_constructor(value: Any) -> dict[str, Any]:
    constructor = _object(value, "constructor")
    _exact_fields(constructor, {"args", "kwargs"}, "constructor")
    args = constructor.get("args")
    kwargs = constructor.get("kwargs")
    if not isinstance(args, list):
        raise UserTraceRequestError("constructor.args", "must be an array")
    if not isinstance(kwargs, dict):
        raise UserTraceRequestError("constructor.kwargs", "must be an object")
    counter = [0]
    normalized_kwargs: dict[str, Any] = {}
    for key, item in kwargs.items():
        if not isinstance(key, str) or not key.isidentifier():
            raise UserTraceRequestError("constructor.kwargs", "keys must be Python identifiers")
        normalized_kwargs[key] = _validate_literal(item, f"constructor.kwargs.{key}", 1, counter)
    normalized = {
        "args": [_validate_literal(item, f"constructor.args[{index}]", 1, counter) for index, item in enumerate(args)],
        "kwargs": normalized_kwargs,
    }
    serialized_bytes = len(json.dumps(normalized, separators=(",", ":")).encode("utf-8"))
    if serialized_bytes > MAX_CONSTRUCTOR_SERIALIZED_BYTES:
        raise UserTraceRequestError("constructor", f"exceeds maximum serialized size {MAX_CONSTRUCTOR_SERIALIZED_BYTES} bytes")
    return normalized


def validate_user_trace_request(
    value: Any,
    *,
    expected_output_path: Path | None = None,
) -> dict[str, Any]:
    request = _object(value, "request")
    input_schema_version = request.get("input_schema_version", 1)
    if input_schema_version not in {1, 2}:
        raise UserTraceRequestError("input_schema_version", "must equal 1 or 2")
    input_fields = {"inputs"} if input_schema_version == 1 else {"args", "kwargs"}
    _exact_fields(request, {"protocol_version", "input_schema_version", "run_id", "source", "constructor", "input_provider", "output_path", "project_context"} | input_fields, "request")

    if request.get("protocol_version") != PROTOCOL_VERSION:
        raise UserTraceRequestError("protocol_version", f"must equal {PROTOCOL_VERSION}")
    run_id = _non_empty_string(request.get("run_id"), "run_id")

    source = _object(request.get("source"), "source")
    _exact_fields(source, {"file_path", "class_name", "content_sha256"}, "source")
    file_path_text = _non_empty_string(source.get("file_path"), "source.file_path")
    class_name = _non_empty_string(source.get("class_name"), "source.class_name")
    content_sha256 = _non_empty_string(source.get("content_sha256"), "source.content_sha256")
    if len(content_sha256) != 64 or any(character not in "0123456789abcdef" for character in content_sha256):
        raise UserTraceRequestError("source.content_sha256", "must be a lowercase SHA-256 digest")
    if not class_name.isidentifier():
        raise UserTraceRequestError("source.class_name", "must be a Python identifier")
    file_path = Path(file_path_text)
    if file_path.suffix.lower() != ".py":
        raise UserTraceRequestError("source.file_path", "must reference a .py file")
    try:
        file_stat = file_path.stat()
    except OSError as exc:
        raise UserTraceRequestError("source.file_path", "must reference an existing regular file") from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise UserTraceRequestError("source.file_path", "must reference an existing regular file")
    project_context = _validate_project_context(request.get("project_context"), file_path)

    constructor = _validate_constructor(request.get("constructor"))

    provider = request.get("input_provider")
    normalized_provider = None
    if provider is not None:
        provider = _object(provider, "input_provider")
        _exact_fields(provider, {"function_name", "parameter_names"}, "input_provider")
        if provider.get("function_name") != "netviz_example_inputs":
            raise UserTraceRequestError("input_provider.function_name", "must equal 'netviz_example_inputs'")
        parameter_names = provider.get("parameter_names")
        if not isinstance(parameter_names, list) or len(parameter_names) > MAX_USER_INPUTS or any(not isinstance(name, str) or not name.isidentifier() for name in parameter_names):
            raise UserTraceRequestError("input_provider.parameter_names", f"must contain at most {MAX_USER_INPUTS} Python identifiers")
        normalized_provider = {"function_name": "netviz_example_inputs", "parameter_names": list(parameter_names)}

    inputs = request.get("inputs") if input_schema_version == 1 else []
    if not isinstance(inputs, list):
        raise UserTraceRequestError("inputs", "must be an array")
    if len(inputs) > MAX_USER_INPUTS:
        raise UserTraceRequestError("inputs", f"must contain at most {MAX_USER_INPUTS} tensor inputs")
    if normalized_provider is not None and inputs:
        raise UserTraceRequestError("inputs", "must be empty when input_provider is configured")

    total_bytes = 0
    normalized_inputs: list[dict[str, Any]] = []
    for index, raw_input in enumerate(inputs):
        path = f"inputs[{index}]"
        input_spec = _object(raw_input, path)
        _exact_fields(input_spec, {"kind", "parameter_name", "shape", "dtype", "generator", "integer_max_exclusive"}, path)
        parameter_name = _non_empty_string(input_spec.get("parameter_name"), f"{path}.parameter_name")
        if not parameter_name.isidentifier():
            raise UserTraceRequestError(f"{path}.parameter_name", "must be a Python identifier")
        if input_spec.get("kind") != "tensor":
            raise UserTraceRequestError(f"{path}.kind", "must equal 'tensor'")
        shape = input_spec.get("shape")
        if not isinstance(shape, list):
            raise UserTraceRequestError(f"{path}.shape", "must be an array")
        if len(shape) > MAX_TENSOR_DIMENSIONS:
            raise UserTraceRequestError(
                f"{path}.shape",
                f"must contain at most {MAX_TENSOR_DIMENSIONS} dimensions",
            )
        for dimension_index, dimension in enumerate(shape):
            if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
                raise UserTraceRequestError(
                    f"{path}.shape[{dimension_index}]",
                    "must be an integer of at least 1",
                )
        element_count = math.prod(shape)
        if element_count > MAX_TENSOR_ELEMENTS:
            raise UserTraceRequestError(
                f"{path}.shape",
                f"contains {element_count} elements; maximum is {MAX_TENSOR_ELEMENTS}",
            )
        dtype = input_spec.get("dtype")
        if dtype not in SUPPORTED_TENSOR_DTYPES:
            raise UserTraceRequestError(f"{path}.dtype", f"must be one of {sorted(SUPPORTED_TENSOR_DTYPES)}")
        generator = input_spec.get("generator")
        if generator not in SUPPORTED_TENSOR_GENERATORS:
            raise UserTraceRequestError(f"{path}.generator", f"must be one of {sorted(SUPPORTED_TENSOR_GENERATORS)}")
        if dtype == "float32" and generator != "random_normal":
            raise UserTraceRequestError(f"{path}.generator", "must equal 'random_normal' for float32")
        if dtype == "int64" and generator != "random_integer":
            raise UserTraceRequestError(f"{path}.generator", "must equal 'random_integer' for int64")
        integer_max = input_spec.get("integer_max_exclusive", DEFAULT_INTEGER_MAX_EXCLUSIVE)
        if dtype == "int64" and (isinstance(integer_max, bool) or not isinstance(integer_max, int) or integer_max < 1):
            raise UserTraceRequestError(f"{path}.integer_max_exclusive", "must be an integer of at least 1")
        if dtype == "float32" and "integer_max_exclusive" in input_spec:
            raise UserTraceRequestError(f"{path}.integer_max_exclusive", "is only supported for int64")
        total_bytes += element_count * TENSOR_DTYPE_BYTES[dtype]
        normalized_inputs.append({
            "kind": "tensor",
            "parameter_name": parameter_name,
            "shape": list(shape),
            "dtype": dtype,
            "generator": generator,
            **({"integer_max_exclusive": integer_max} if dtype == "int64" else {}),
        })

    if total_bytes > MAX_TOTAL_INPUT_BYTES:
        raise UserTraceRequestError(
            "inputs",
            f"require {total_bytes} bytes; maximum is {MAX_TOTAL_INPUT_BYTES}",
        )

    normalized_args: list[dict[str, Any]] = []
    normalized_kwargs: dict[str, dict[str, Any]] = {}
    if input_schema_version == 2:
        try:
            normalized_args, normalized_kwargs = validate_structured_call(request.get("args"), request.get("kwargs"))
        except StructuredInputError as exc:
            raise UserTraceRequestError(exc.path, exc.message) from exc
        if normalized_provider is not None and (normalized_args or normalized_kwargs):
            raise UserTraceRequestError("args", "args and kwargs must be empty when input_provider is configured")

    output_path_text = _non_empty_string(request.get("output_path"), "output_path")
    output_path = Path(output_path_text)
    if expected_output_path is not None and output_path != expected_output_path:
        raise UserTraceRequestError("output_path", "does not match the host-controlled result path")

    return {
        "protocol_version": PROTOCOL_VERSION,
        "input_schema_version": input_schema_version,
        "run_id": run_id,
        "source": {"file_path": str(file_path), "class_name": class_name, "content_sha256": content_sha256},
        "project_context": project_context,
        "constructor": constructor,
        "inputs": normalized_inputs,
        "args": normalized_args,
        "kwargs": normalized_kwargs,
        "input_provider": normalized_provider,
        "output_path": str(output_path),
    }


def _validate_project_context(value: Any, source_path: Path) -> dict[str, Any]:
    if value is None:
        root = source_path.parent.resolve()
        return {
            "project_root": str(root),
            "working_directory": str(root),
            "entry_relative_path": source_path.name,
            "local_modules": [],
            "resources": [],
        }
    context = _object(value, "project_context")
    _exact_fields(context, {"project_root", "working_directory", "entry_relative_path", "local_modules", "resources"}, "project_context")
    project_root = Path(_non_empty_string(context.get("project_root"), "project_context.project_root")).resolve()
    working_directory = Path(_non_empty_string(context.get("working_directory"), "project_context.working_directory")).resolve()
    try:
        root_stat = project_root.stat()
        working_directory.relative_to(project_root)
        source_path.resolve().relative_to(project_root)
    except (OSError, ValueError) as exc:
        raise UserTraceRequestError("project_context.project_root", "must contain the source file and working directory") from exc
    if not stat.S_ISDIR(root_stat.st_mode):
        raise UserTraceRequestError("project_context.project_root", "must reference an existing directory")
    if not working_directory.is_dir():
        raise UserTraceRequestError("project_context.working_directory", "must reference an existing directory")
    entry_relative_path = _non_empty_string(context.get("entry_relative_path"), "project_context.entry_relative_path")
    if (project_root / entry_relative_path).resolve() != source_path.resolve():
        raise UserTraceRequestError("project_context.entry_relative_path", "must match source.file_path")

    return {
        "project_root": str(project_root),
        "working_directory": str(working_directory),
        "entry_relative_path": entry_relative_path,
        "local_modules": _validate_project_files(context.get("local_modules"), project_root, "project_context.local_modules"),
        "resources": _validate_project_files(context.get("resources"), project_root, "project_context.resources"),
    }


def _validate_project_files(value: Any, project_root: Path, path: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise UserTraceRequestError(path, "must be an array")
    normalized = []
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        descriptor = _object(item, item_path)
        _exact_fields(descriptor, {"path", "content_sha256", "size_bytes", "exists"}, item_path)
        relative = _non_empty_string(descriptor.get("path"), f"{item_path}.path")
        resolved = (project_root / relative).resolve()
        try:
            resolved.relative_to(project_root)
        except ValueError as exc:
            raise UserTraceRequestError(f"{item_path}.path", "must stay inside project_root") from exc
        exists = descriptor.get("exists")
        if not isinstance(exists, bool):
            raise UserTraceRequestError(f"{item_path}.exists", "must be a boolean")
        digest = descriptor.get("content_sha256")
        if exists:
            if not resolved.is_file():
                raise UserTraceRequestError(f"{item_path}.path", "must reference an existing file")
            if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise UserTraceRequestError(f"{item_path}.content_sha256", "must be a lowercase SHA-256 digest")
            size_bytes = descriptor.get("size_bytes")
            if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
                raise UserTraceRequestError(f"{item_path}.size_bytes", "must be a non-negative integer")
            normalized.append({"path": relative, "content_sha256": digest, "size_bytes": size_bytes, "exists": True})
        else:
            normalized.append({"path": relative, "exists": False})
    return normalized
