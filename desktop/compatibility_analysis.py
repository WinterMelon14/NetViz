"""Conservative AST-only compatibility analysis for one model candidate."""

import ast
import sys
from typing import Any

from desktop.compatibility_schema import finding, source_evidence, validate_compatibility_report
from desktop.user_trace_constants import (
    COMPATIBILITY_REPORT_VERSION,
    DEFAULT_TRACE_TIMEOUT_SECONDS,
    MAX_TENSOR_DIMENSIONS,
    MAX_TENSOR_ELEMENTS,
    MAX_TOTAL_INPUT_BYTES,
    MAX_USER_INPUTS,
    SUPPORTED_TENSOR_DTYPES,
    TRACE_DEVICE_TYPE,
    TRACE_MODEL_MODE,
)


def build_compatibility_report(module: ast.Module, class_node: ast.ClassDef, candidate: dict[str, Any]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    findings.extend(_class_and_signature_findings(class_node, candidate))
    findings.extend(_input_findings(candidate))
    findings.extend(_import_findings(module))
    findings.extend(_resource_findings(module))
    findings.extend(_runtime_findings())
    findings.extend(_fx_findings(class_node, candidate))
    report = {
        "schemaVersion": COMPATIBILITY_REPORT_VERSION,
        "className": class_node.name,
        "findings": findings,
        "tracingOutcome": {
            "status": "unknown",
            "statement": "Symbolic tracing outcome is unknown until the model is executed.",
        },
    }
    return validate_compatibility_report(report)


def _class_and_signature_findings(class_node: ast.ClassDef, candidate: dict[str, Any]) -> list[dict[str, Any]]:
    confidence = candidate["confidence"]
    items = [finding(
        "class",
        "model_class_confirmed" if confidence in {"confirmed", "likely"} else "model_class_candidate_uncertain",
        "supported" if confidence in {"confirmed", "likely"} else "unknown",
        "Model class identified" if confidence in {"confirmed", "likely"} else "Model class requires confirmation",
        f"Static inspection classified {class_node.name} with {confidence} confidence.",
        "source",
        source_evidence(f"class {class_node.name}", class_node.lineno),
    )]
    items.extend(_parameter_findings("constructor", candidate["constructor"]["parameters"]))
    forward = candidate.get("forward")
    if not forward:
        items.append(finding("forward", "forward_missing", "unsupported", "Forward method not found", "The selected class does not declare a forward method that NetViz can inspect.", "source", source_evidence(f"class {class_node.name}", class_node.lineno), "Add a forward method or select another class."))
        return items
    if forward.get("isAsync"):
        items.append(finding("forward", "forward_async", "unsupported", "Async forward is unsupported", "The current trace runtime invokes a synchronous forward method.", "source", source_evidence("async forward", forward["lineNumber"]), "Use a synchronous forward method for tracing."))
    else:
        items.append(finding("forward", "forward_sync", "supported", "Forward method found", "A synchronous forward method is available for configuration.", "source", source_evidence("forward", forward["lineNumber"])))
    items.extend(_parameter_findings("forward", forward["parameters"]))
    for key, code, title in (("hasVarArgs", "forward_varargs", "Variadic positional inputs"), ("hasVarKwargs", "forward_varkwargs", "Variadic keyword inputs")):
        if forward.get(key):
            items.append(finding("forward", code, "unknown", title, "Static inspection cannot determine which variadic values a representative run requires.", "source", source_evidence(title, forward["lineNumber"]), "Use an explicit forward signature for the current input editor."))
    return items


def _parameter_findings(category: str, parameters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for parameter in parameters:
        name = parameter["name"]
        target = {"kind": f"{category}_parameter", "name": name}
        line = parameter.get("lineNumber")
        if not parameter["required"]:
            items.append(finding(category, f"{category}_parameter_optional", "supported", f"{name} may be omitted", "The declaration provides a default, so NetViz may leave this parameter out of the call.", "source", source_evidence(parameter.get("declaration", name), line), target=target))
        else:
            items.append(finding(category, f"{category}_parameter_configuration_required", "configuration_required", f"Configure {name}", "A representative value must be confirmed before execution.", "source", source_evidence(parameter.get("declaration", name), line), "Complete the matching configuration field.", target))
    return items


def _input_findings(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    forward = candidate.get("forward") or {}
    suggestions = {item["parameterName"]: item for item in forward.get("inputSuggestions", [])}
    items = []
    for parameter in forward.get("parameters", []):
        if not parameter["required"]:
            continue
        name = parameter["name"]
        target = {"kind": "forward_parameter", "name": name}
        suggestion = suggestions.get(name)
        if suggestion:
            shape = suggestion["shapeTemplate"]
            unresolved = [index for index, dimension in enumerate(shape) if dimension is None]
            items.append(finding(
                "input", "input_shape_partially_known" if unresolved else "input_shape_known",
                "unknown" if unresolved else "supported",
                f"{name} shape is {'partially known' if unresolved else 'suggested'}",
                f"Suggested shape: {shape}; unresolved dimensions: {unresolved or 'none'}.",
                "heuristic", source_evidence("; ".join(suggestion["evidence"])),
                "Confirm every representative dimension before execution." if unresolved else None, target,
            ))
            dtype = suggestion.get("dtypeCategory")
            items.append(finding(
                "input", "input_dtype_suggested" if dtype else "input_dtype_unresolved",
                "supported" if dtype in {"floating", "integer"} else "unknown",
                f"{name} dtype {'suggested' if dtype else 'is unresolved'}",
                f"Static consumer analysis suggests {dtype}." if dtype else "Static inspection cannot determine a reliable dtype.",
                "heuristic", source_evidence("; ".join(suggestion["evidence"])),
                "Confirm the representative dtype before execution.", target,
            ))
        else:
            items.extend([
                finding("input", "input_shape_unresolved", "unknown", f"{name} shape is unresolved", "Arbitrary tensor shapes cannot be derived reliably from model source alone.", "heuristic", source_evidence(parameter.get("declaration", name), parameter.get("lineNumber")), "Confirm a representative shape.", target),
                finding("input", "input_dtype_unresolved", "unknown", f"{name} dtype is unresolved", "No reliable static dtype suggestion was found.", "heuristic", source_evidence(parameter.get("declaration", name), parameter.get("lineNumber")), "Confirm a representative dtype.", target),
            ])
    return items


def _import_findings(module: ast.Module) -> list[dict[str, Any]]:
    items = []
    standard = getattr(sys, "stdlib_module_names", frozenset())
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imports = [(alias.name, 0) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imports = [(node.module or ".", node.level)]
        else:
            continue
        for name, level in imports:
            root = name.split(".")[0]
            if level:
                code, status, title = "import_relative_local", "unsupported", "Relative local import"
                explanation = "The current single-file worker does not establish package context for relative imports."
                remediation = "Project-aware imports are planned for R2."
            elif root in standard:
                code, status, title = "import_standard_library", "supported", "Standard-library import"
                explanation, remediation = f"{name} is available from the Python standard library.", None
            elif root in {"torch", "torchvision"}:
                code, status, title = "import_runtime_dependency", "supported", "PyTorch runtime import"
                explanation, remediation = f"{name} is expected in the selected tracing environment.", None
            else:
                code, status, title = "import_resolution_unknown", "unknown", "Import availability is unknown"
                explanation, remediation = f"Static inspection cannot determine whether {name} is a local or installed dependency.", "Ensure the dependency is available in the tracing environment."
            items.append(finding("import", code, status, title, explanation, "source", source_evidence(ast.unparse(node), node.lineno), remediation))
    return _dedupe(items)


def _resource_findings(module: ast.Module) -> list[dict[str, Any]]:
    items = []
    resource_calls = {"open", "torch.load", "load", "from_pretrained", "Path"}
    for node in ast.walk(module):
        if not isinstance(node, ast.Call) or not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
            continue
        call_name = ast.unparse(node.func)
        if call_name.split(".")[-1] not in resource_calls:
            continue
        path = node.args[0].value
        items.append(finding("resource", "resource_reference_likely", "unknown", "Likely external resource", f"The call {call_name} references {path!r}; static inspection cannot confirm when or how it is used.", "heuristic", source_evidence(ast.unparse(node), node.lineno), "Confirm that the resource is available from the worker's current directory."))
    return _dedupe(items)


def _runtime_findings() -> list[dict[str, Any]]:
    constraints = [
        ("runtime_device_cpu", "CPU execution", f"Generated representative tensors use {TRACE_DEVICE_TYPE}."),
        ("runtime_mode_eval", "Evaluation mode", f"The traced graph runs in {TRACE_MODEL_MODE} mode."),
        ("runtime_dtypes", "Supported generated dtypes", f"Generated tensors support {', '.join(sorted(SUPPORTED_TENSOR_DTYPES))}."),
        ("runtime_input_limits", "Input allocation limits", f"At most {MAX_USER_INPUTS} tensors, {MAX_TENSOR_DIMENSIONS} dimensions, {MAX_TENSOR_ELEMENTS:,} elements per tensor, and {MAX_TOTAL_INPUT_BYTES:,} total bytes."),
        ("runtime_timeout", "Trace timeout", f"The worker timeout is {DEFAULT_TRACE_TIMEOUT_SECONDS} seconds."),
    ]
    return [finding("runtime", code, "supported", title, explanation, "runtime", [{"kind": "runtime", "text": explanation}]) for code, title, explanation in constraints]


def _fx_findings(class_node: ast.ClassDef, candidate: dict[str, Any]) -> list[dict[str, Any]]:
    forward = next((node for node in class_node.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "forward"), None)
    if forward is None:
        return []
    parameter_names = {item["name"] for item in (candidate.get("forward") or {}).get("parameters", [])}
    items = []
    for node in ast.walk(forward):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "item":
            items.append(_fx_warning("fx_tensor_item", "Tensor value extraction", ".item() may require a concrete tensor value during symbolic tracing.", node))
        if isinstance(node, (ast.If, ast.While)) and _references_names(node.test, parameter_names):
            items.append(_fx_warning("fx_tensor_dependent_branch", "Potential tensor-dependent control flow", "A branch condition references a forward parameter and may not be symbolically traceable.", node.test))
        if isinstance(node, (ast.For, ast.comprehension)) and _references_names(node.iter, parameter_names):
            items.append(_fx_warning("fx_tensor_iteration", "Potential tensor iteration", "Iteration over a value derived from a forward parameter may require concrete tensor data.", node.iter))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr" and node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == "self":
            items.append(_fx_warning("fx_dynamic_module_selection", "Dynamic module selection", "Dynamic getattr on the model may hide the module target from symbolic tracing.", node))
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and any(_is_self_attribute(target) for target in _assignment_targets(node)):
            items.append(_fx_warning("fx_forward_module_mutation", "Model mutation during forward", "Assigning model state during forward can distort or prevent symbolic tracing.", node))
    return _dedupe(items)


def _fx_warning(code: str, title: str, explanation: str, node: ast.AST) -> dict[str, Any]:
    return finding("fx", code, "unknown", title, explanation, "heuristic", source_evidence(ast.unparse(node), getattr(node, "lineno", None)), "Run tracing to determine whether this construct is supported for the selected inputs.")


def _references_names(node: ast.AST, names: set[str]) -> bool:
    return any(isinstance(child, ast.Name) and child.id in names for child in ast.walk(node))


def _assignment_targets(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.Assign):
        return list(node.targets)
    target = getattr(node, "target", None)
    return [target] if isinstance(target, ast.AST) else []


def _is_self_attribute(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self"


def _dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, int | None, str]] = set()
    result = []
    for item in items:
        evidence = item["evidence"][0]
        key = (item["code"], evidence.get("lineNumber"), evidence["text"])
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result
