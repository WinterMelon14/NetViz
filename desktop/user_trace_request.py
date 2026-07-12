import math
import stat
from pathlib import Path
from typing import Any

from desktop.trace_protocol import PROTOCOL_VERSION

MAX_USER_INPUTS = 1
MAX_TENSOR_DIMENSIONS = 8
MAX_TENSOR_ELEMENTS = 16_777_216
MAX_TOTAL_INPUT_BYTES = 64 * 1024 * 1024
FLOAT32_BYTES = 4
SUPPORTED_TENSOR_DTYPES = frozenset({"float32"})
SUPPORTED_TENSOR_GENERATORS = frozenset({"random_normal"})


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


def validate_user_trace_request(
    value: Any,
    *,
    expected_output_path: Path | None = None,
) -> dict[str, Any]:
    request = _object(value, "request")
    _exact_fields(
        request,
        {"protocol_version", "run_id", "source", "constructor", "inputs", "output_path"},
        "request",
    )

    if request.get("protocol_version") != PROTOCOL_VERSION:
        raise UserTraceRequestError("protocol_version", f"must equal {PROTOCOL_VERSION}")
    run_id = _non_empty_string(request.get("run_id"), "run_id")

    source = _object(request.get("source"), "source")
    _exact_fields(source, {"file_path", "class_name"}, "source")
    file_path_text = _non_empty_string(source.get("file_path"), "source.file_path")
    class_name = _non_empty_string(source.get("class_name"), "source.class_name")
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

    constructor = _object(request.get("constructor"), "constructor")
    _exact_fields(constructor, {"args", "kwargs"}, "constructor")
    if constructor.get("args") != []:
        raise UserTraceRequestError("constructor.args", "must be empty in protocol version 1")
    if constructor.get("kwargs") != {}:
        raise UserTraceRequestError("constructor.kwargs", "must be empty in protocol version 1")

    inputs = request.get("inputs")
    if not isinstance(inputs, list):
        raise UserTraceRequestError("inputs", "must be an array")
    if len(inputs) != MAX_USER_INPUTS:
        raise UserTraceRequestError("inputs", f"must contain exactly {MAX_USER_INPUTS} tensor input")

    total_bytes = 0
    normalized_inputs: list[dict[str, Any]] = []
    for index, raw_input in enumerate(inputs):
        path = f"inputs[{index}]"
        input_spec = _object(raw_input, path)
        _exact_fields(input_spec, {"kind", "shape", "dtype", "generator"}, path)
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
            raise UserTraceRequestError(f"{path}.dtype", "must equal 'float32'")
        generator = input_spec.get("generator")
        if generator not in SUPPORTED_TENSOR_GENERATORS:
            raise UserTraceRequestError(f"{path}.generator", "must equal 'random_normal'")
        total_bytes += element_count * FLOAT32_BYTES
        normalized_inputs.append({
            "kind": "tensor",
            "shape": list(shape),
            "dtype": dtype,
            "generator": generator,
        })

    if total_bytes > MAX_TOTAL_INPUT_BYTES:
        raise UserTraceRequestError(
            "inputs",
            f"require {total_bytes} bytes; maximum is {MAX_TOTAL_INPUT_BYTES}",
        )

    output_path_text = _non_empty_string(request.get("output_path"), "output_path")
    output_path = Path(output_path_text)
    if expected_output_path is not None and output_path != expected_output_path:
        raise UserTraceRequestError("output_path", "does not match the host-controlled result path")

    return {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": run_id,
        "source": {"file_path": str(file_path), "class_name": class_name},
        "constructor": {"args": [], "kwargs": {}},
        "inputs": normalized_inputs,
        "output_path": str(output_path),
    }
