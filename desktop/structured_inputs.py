"""Validation and construction for bounded schema-v2 forward arguments."""

import json
import math
from dataclasses import dataclass
from typing import Any

from desktop.user_trace_constants import (
    DEFAULT_INTEGER_MAX_EXCLUSIVE,
    MAX_INPUT_SERIALIZED_BYTES,
    MAX_INPUT_STRING_CHARS,
    MAX_STRUCTURED_CONTAINER_ITEMS,
    MAX_STRUCTURED_INPUT_DEPTH,
    MAX_STRUCTURED_INPUT_VALUES,
    MAX_TENSOR_DIMENSIONS,
    MAX_TENSOR_ELEMENTS,
    MAX_TOTAL_INPUT_BYTES,
    MAX_USER_INPUTS,
    SUPPORTED_TENSOR_DTYPES,
    SUPPORTED_TENSOR_GENERATORS,
    TENSOR_DTYPE_BYTES,
)


class StructuredInputError(ValueError):
    def __init__(self, path: str, message: str):
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message


@dataclass
class InputBudget:
    values: int = 0
    tensors: int = 0
    tensor_bytes: int = 0


def validate_structured_call(args: Any, kwargs: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not isinstance(args, list):
        raise StructuredInputError("args", "must be an array")
    if not isinstance(kwargs, dict):
        raise StructuredInputError("kwargs", "must be an object")
    if len(args) + len(kwargs) > MAX_USER_INPUTS:
        raise StructuredInputError("args", f"args and kwargs may contain at most {MAX_USER_INPUTS} top-level values")
    for key in kwargs:
        if not isinstance(key, str) or not key.isidentifier():
            raise StructuredInputError("kwargs", "keys must be Python identifiers")

    budget = InputBudget()
    normalized_args = [_validate_spec(item, f"args[{index}]", 1, budget) for index, item in enumerate(args)]
    normalized_kwargs = {key: _validate_spec(item, f"kwargs.{key}", 1, budget) for key, item in kwargs.items()}
    serialized_size = len(json.dumps({"args": normalized_args, "kwargs": normalized_kwargs}, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    if serialized_size > MAX_INPUT_SERIALIZED_BYTES:
        raise StructuredInputError("args", f"structured inputs exceed {MAX_INPUT_SERIALIZED_BYTES} serialized bytes")
    return normalized_args, normalized_kwargs


def _validate_spec(value: Any, path: str, depth: int, budget: InputBudget) -> dict[str, Any]:
    if depth > MAX_STRUCTURED_INPUT_DEPTH:
        raise StructuredInputError(path, f"exceeds maximum depth {MAX_STRUCTURED_INPUT_DEPTH}")
    budget.values += 1
    if budget.values > MAX_STRUCTURED_INPUT_VALUES:
        raise StructuredInputError(path, f"exceeds maximum value count {MAX_STRUCTURED_INPUT_VALUES}")
    if not isinstance(value, dict):
        raise StructuredInputError(path, "must be a tagged input object")
    kind = value.get("kind")
    if kind == "tensor":
        return _validate_tensor_spec(value, path, budget)
    if kind == "none":
        _exact_fields(value, {"kind"}, path)
        return {"kind": "none"}
    if kind == "boolean":
        _exact_fields(value, {"kind", "value"}, path)
        if not isinstance(value.get("value"), bool):
            raise StructuredInputError(f"{path}.value", "must be a boolean")
        return {"kind": "boolean", "value": value["value"]}
    if kind == "integer":
        _exact_fields(value, {"kind", "value"}, path)
        number = value.get("value")
        if isinstance(number, bool) or not isinstance(number, int):
            raise StructuredInputError(f"{path}.value", "must be an integer")
        return {"kind": "integer", "value": number}
    if kind == "float":
        _exact_fields(value, {"kind", "value"}, path)
        number = value.get("value")
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise StructuredInputError(f"{path}.value", "must be a finite number")
        return {"kind": "float", "value": float(number)}
    if kind == "string":
        _exact_fields(value, {"kind", "value"}, path)
        text = value.get("value")
        if not isinstance(text, str) or len(text) > MAX_INPUT_STRING_CHARS:
            raise StructuredInputError(f"{path}.value", f"must be a string of at most {MAX_INPUT_STRING_CHARS} characters")
        return {"kind": "string", "value": text}
    if kind in {"list", "tuple"}:
        _exact_fields(value, {"kind", "items"}, path)
        items = value.get("items")
        if not isinstance(items, list) or len(items) > MAX_STRUCTURED_CONTAINER_ITEMS:
            raise StructuredInputError(f"{path}.items", f"must be an array of at most {MAX_STRUCTURED_CONTAINER_ITEMS} values")
        return {"kind": kind, "items": [_validate_spec(item, f"{path}.items[{index}]", depth + 1, budget) for index, item in enumerate(items)]}
    if kind == "dict":
        _exact_fields(value, {"kind", "entries"}, path)
        entries = value.get("entries")
        if not isinstance(entries, list) or len(entries) > MAX_STRUCTURED_CONTAINER_ITEMS:
            raise StructuredInputError(f"{path}.entries", f"must be an array of at most {MAX_STRUCTURED_CONTAINER_ITEMS} entries")
        normalized_entries = []
        keys: set[str] = set()
        for index, entry in enumerate(entries):
            entry_path = f"{path}.entries[{index}]"
            if not isinstance(entry, dict):
                raise StructuredInputError(entry_path, "must be an object")
            _exact_fields(entry, {"key", "value"}, entry_path)
            key = entry.get("key")
            if not isinstance(key, str) or len(key) > MAX_INPUT_STRING_CHARS:
                raise StructuredInputError(f"{entry_path}.key", f"must be a string of at most {MAX_INPUT_STRING_CHARS} characters")
            if key in keys:
                raise StructuredInputError(f"{entry_path}.key", "duplicates an earlier dictionary key")
            keys.add(key)
            normalized_entries.append({"key": key, "value": _validate_spec(entry.get("value"), f"{entry_path}.value", depth + 1, budget)})
        return {"kind": "dict", "entries": normalized_entries}
    raise StructuredInputError(f"{path}.kind", "is not a supported input kind")


def _validate_tensor_spec(value: dict[str, Any], path: str, budget: InputBudget) -> dict[str, Any]:
    _exact_fields(value, {"kind", "shape", "dtype", "generator", "integer_max_exclusive"}, path)
    shape = value.get("shape")
    if not isinstance(shape, list) or len(shape) > MAX_TENSOR_DIMENSIONS:
        raise StructuredInputError(f"{path}.shape", f"must be an array of at most {MAX_TENSOR_DIMENSIONS} dimensions")
    for index, dimension in enumerate(shape):
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise StructuredInputError(f"{path}.shape[{index}]", "must be an integer of at least 1")
    element_count = math.prod(shape)
    if element_count > MAX_TENSOR_ELEMENTS:
        raise StructuredInputError(f"{path}.shape", f"contains {element_count} elements; maximum is {MAX_TENSOR_ELEMENTS}")
    dtype = value.get("dtype")
    generator = value.get("generator")
    if dtype not in SUPPORTED_TENSOR_DTYPES:
        raise StructuredInputError(f"{path}.dtype", f"must be one of {sorted(SUPPORTED_TENSOR_DTYPES)}")
    if generator not in SUPPORTED_TENSOR_GENERATORS:
        raise StructuredInputError(f"{path}.generator", f"must be one of {sorted(SUPPORTED_TENSOR_GENERATORS)}")
    expected_generator = "random_integer" if dtype == "int64" else "random_normal"
    if generator != expected_generator:
        raise StructuredInputError(f"{path}.generator", f"must equal {expected_generator!r} for {dtype}")
    integer_max = value.get("integer_max_exclusive", DEFAULT_INTEGER_MAX_EXCLUSIVE)
    if dtype == "int64" and (isinstance(integer_max, bool) or not isinstance(integer_max, int) or integer_max < 1):
        raise StructuredInputError(f"{path}.integer_max_exclusive", "must be an integer of at least 1")
    if dtype != "int64" and "integer_max_exclusive" in value:
        raise StructuredInputError(f"{path}.integer_max_exclusive", "is only supported for int64")
    budget.tensors += 1
    if budget.tensors > MAX_USER_INPUTS:
        raise StructuredInputError(path, f"structured inputs may contain at most {MAX_USER_INPUTS} tensors")
    tensor_bytes = element_count * TENSOR_DTYPE_BYTES[dtype]
    budget.tensor_bytes += tensor_bytes
    if budget.tensor_bytes > MAX_TOTAL_INPUT_BYTES:
        raise StructuredInputError(path, f"tensor allocations exceed {MAX_TOTAL_INPUT_BYTES} bytes")
    return {"kind": "tensor", "shape": list(shape), "dtype": dtype, "generator": generator, **({"integer_max_exclusive": integer_max} if dtype == "int64" else {})}


def construct_structured_call(args: list[dict[str, Any]], kwargs: dict[str, dict[str, Any]]) -> tuple[list[Any], dict[str, Any], list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    built_args = [_construct_spec(item, f"args[{index}]", diagnostics) for index, item in enumerate(args)]
    built_kwargs = {key: _construct_spec(item, f"kwargs.{key}", diagnostics) for key, item in kwargs.items()}
    return built_args, built_kwargs, diagnostics


def _construct_spec(spec: dict[str, Any], path: str, diagnostics: list[dict[str, Any]]) -> Any:
    kind = spec["kind"]
    if kind == "tensor":
        import torch
        if spec["dtype"] == "int64":
            tensor = torch.randint(0, spec["integer_max_exclusive"], tuple(spec["shape"]), dtype=torch.int64, device="cpu")
        else:
            tensor = torch.randn(tuple(spec["shape"]), dtype=torch.float32, device="cpu")
        diagnostics.append(_tensor_diagnostic(path, tensor, spec["generator"]))
        return tensor
    if kind == "none":
        return None
    if kind in {"boolean", "integer", "float", "string"}:
        return spec["value"]
    if kind == "list":
        return [_construct_spec(item, f"{path}.items[{index}]", diagnostics) for index, item in enumerate(spec["items"])]
    if kind == "tuple":
        return tuple(_construct_spec(item, f"{path}.items[{index}]", diagnostics) for index, item in enumerate(spec["items"]))
    if kind == "dict":
        return {entry["key"]: _construct_spec(entry["value"], f"{path}.{entry['key']}", diagnostics) for entry in spec["entries"]}
    raise RuntimeError(f"Validated structured input has unsupported kind {kind!r}")


def validate_provider_result(value: Any, parameter_names: list[str]) -> tuple[list[Any], dict[str, Any], list[dict[str, Any]]]:
    if isinstance(value, (tuple, list)):
        if len(value) > MAX_USER_INPUTS or not all(_is_tensor(item) for item in value):
            raise StructuredInputError("provider", f"legacy provider results must contain at most {MAX_USER_INPUTS} tensors")
        args, kwargs = list(value), {}
    elif isinstance(value, dict) and set(value) == {"args", "kwargs"}:
        args, kwargs = value["args"], value["kwargs"]
        if not isinstance(args, (tuple, list)):
            raise StructuredInputError("provider.args", "must be a tuple or list")
        if not isinstance(kwargs, dict):
            raise StructuredInputError("provider.kwargs", "must be a dictionary")
        if len(args) + len(kwargs) > MAX_USER_INPUTS:
            raise StructuredInputError("provider", f"args and kwargs may contain at most {MAX_USER_INPUTS} top-level values")
        for key in kwargs:
            if not isinstance(key, str) or not key.isidentifier():
                raise StructuredInputError("provider.kwargs", "keys must be Python identifiers")
        args = list(args)
    else:
        raise StructuredInputError("provider", "must return a legacy tensor tuple/list or an explicit {'args': ..., 'kwargs': ...} wrapper")
    budget = InputBudget()
    diagnostics: list[dict[str, Any]] = []
    for index, item in enumerate(args):
        label = parameter_names[index] if index < len(parameter_names) else f"args[{index}]"
        _validate_provider_value(item, f"provider.args[{index}]", 1, budget, diagnostics, label)
    for key, item in kwargs.items():
        _validate_provider_value(item, f"provider.kwargs.{key}", 1, budget, diagnostics, key)
    return args, kwargs, diagnostics


def _validate_provider_value(value: Any, path: str, depth: int, budget: InputBudget, diagnostics: list[dict[str, Any]], label: str) -> None:
    if depth > MAX_STRUCTURED_INPUT_DEPTH:
        raise StructuredInputError(path, f"exceeds maximum depth {MAX_STRUCTURED_INPUT_DEPTH}")
    budget.values += 1
    if budget.values > MAX_STRUCTURED_INPUT_VALUES:
        raise StructuredInputError(path, f"exceeds maximum value count {MAX_STRUCTURED_INPUT_VALUES}")
    if _is_tensor(value):
        if value.device.type != "cpu" or str(value.dtype) not in {"torch.float32", "torch.int64"}:
            raise StructuredInputError(path, "tensors must be CPU float32 or int64")
        if value.numel() > MAX_TENSOR_ELEMENTS:
            raise StructuredInputError(path, f"tensor exceeds {MAX_TENSOR_ELEMENTS} elements")
        budget.tensors += 1
        budget.tensor_bytes += value.numel() * value.element_size()
        if budget.tensors > MAX_USER_INPUTS or budget.tensor_bytes > MAX_TOTAL_INPUT_BYTES:
            raise StructuredInputError(path, "provider tensors exceed the configured count or allocation limit")
        diagnostics.append(_tensor_diagnostic(label, value, "provider"))
        return
    if value is None or isinstance(value, bool) or (isinstance(value, int) and not isinstance(value, bool)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StructuredInputError(path, "must be a finite number")
        return
    if isinstance(value, str):
        if len(value) > MAX_INPUT_STRING_CHARS:
            raise StructuredInputError(path, f"string exceeds {MAX_INPUT_STRING_CHARS} characters")
        return
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_STRUCTURED_CONTAINER_ITEMS:
            raise StructuredInputError(path, f"container exceeds {MAX_STRUCTURED_CONTAINER_ITEMS} items")
        for index, item in enumerate(value):
            _validate_provider_value(item, f"{path}[{index}]", depth + 1, budget, diagnostics, label)
        return
    if isinstance(value, dict):
        if len(value) > MAX_STRUCTURED_CONTAINER_ITEMS:
            raise StructuredInputError(path, f"dictionary exceeds {MAX_STRUCTURED_CONTAINER_ITEMS} items")
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > MAX_INPUT_STRING_CHARS:
                raise StructuredInputError(path, "nested dictionary keys must be bounded strings")
            _validate_provider_value(item, f"{path}.{key}", depth + 1, budget, diagnostics, label)
        return
    raise StructuredInputError(path, f"contains unsupported value type {type(value).__name__}")


def _tensor_diagnostic(parameter_name: str, tensor: Any, generator: str) -> dict[str, Any]:
    return {
        "parameter_name": parameter_name,
        "shape": list(tensor.shape),
        "dtype": "int64" if str(tensor.dtype) == "torch.int64" else "float32",
        "generator": generator,
        "estimated_bytes": tensor.numel() * tensor.element_size(),
    }


def _is_tensor(value: Any) -> bool:
    try:
        import torch
        return isinstance(value, torch.Tensor)
    except ImportError:
        return False


def _exact_fields(value: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise StructuredInputError(f"{path}.{unknown[0]}", "is not a supported field")
