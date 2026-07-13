import ast
from typing import Any, Literal
from desktop.user_trace_constants import MAX_SOURCE_CHARS

InspectionErrorCode = Literal[
    "source_empty",
    "source_too_large",
    "source_syntax_error",
    "source_inspection_failed",
    "source_protocol_error",
]


def inspection_error(
    code: InspectionErrorCode,
    title: str,
    message: str,
    details: dict[str, Any] | None = None,
    traceback_text: str | None = None,
) -> dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "code": code,
            "title": title,
            "message": message,
            "stage": "source_inspection",
            "details": details or {},
            "traceback": traceback_text,
        },
    }


def inspect_model_source_request(request: Any) -> dict[str, Any]:
    if not isinstance(request, dict):
        return inspection_error(
            "source_protocol_error",
            "Invalid source inspection request",
            "The source inspection request must be an object.",
        )

    source_text = request.get("sourceText")
    if not isinstance(source_text, str):
        return inspection_error(
            "source_protocol_error",
            "Invalid source inspection request",
            "The source inspection request must include sourceText as a string.",
        )

    return inspect_model_source(source_text)


def inspect_model_source(source_text: str) -> dict[str, Any]:
    if not source_text.strip():
        return inspection_error(
            "source_empty",
            "Source is empty",
            "Paste Python source before inspecting model classes.",
        )

    if len(source_text) > MAX_SOURCE_CHARS:
        return inspection_error(
            "source_too_large",
            "Source is too large",
            f"Source inspection is limited to {MAX_SOURCE_CHARS} characters.",
            {"maxChars": MAX_SOURCE_CHARS, "actualChars": len(source_text)},
        )

    try:
        module = ast.parse(source_text)
    except SyntaxError as exc:
        return inspection_error(
            "source_syntax_error",
            "Python syntax error",
            exc.msg,
            {
                "line": exc.lineno,
                "column": exc.offset,
                "endLine": exc.end_lineno,
                "endColumn": exc.end_offset,
                "sourceLine": exc.text.rstrip("\n") if exc.text else None,
                "parserMessage": exc.msg,
            },
        )

    try:
        aliases = inspect_import_aliases(module)
        candidates: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        for class_node, is_nested in iter_class_definitions(module):
            candidate = inspect_class(class_node, aliases)
            if candidate:
                candidates.append(candidate)
                if is_nested:
                    warnings.append({
                        "code": "nested_class",
                        "message": f"{class_node.name} is nested inside another definition.",
                        "lineNumber": class_node.lineno,
                    })
                forward = candidate.get("forward")
                if isinstance(forward, dict) and forward.get("isAsync"):
                    warnings.append({
                        "code": "async_forward",
                        "message": f"{class_node.name}.forward is async and cannot be traced as a normal forward method.",
                        "lineNumber": forward.get("lineNumber"),
                    })

        return {"ok": True, "candidates": candidates, "warnings": warnings}
    except Exception as exc:
        return inspection_error(
            "source_inspection_failed",
            "Source inspection failed",
            str(exc),
        )


def inspect_import_aliases(module: ast.Module) -> dict[str, set[str]]:
    aliases = {
        "torch": set[str](),
        "torch_nn": set[str](),
        "module": set[str](),
    }

    for node in module.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name
                if alias.name == "torch":
                    aliases["torch"].add(local_name)
                elif alias.name == "torch.nn":
                    aliases["torch_nn"].add(local_name)
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if module_name == "torch":
                for alias in node.names:
                    if alias.name == "nn":
                        aliases["torch_nn"].add(alias.asname or alias.name)
            elif module_name == "torch.nn":
                for alias in node.names:
                    if alias.name == "Module":
                        aliases["module"].add(alias.asname or alias.name)

    return aliases


def iter_class_definitions(module: ast.Module) -> list[tuple[ast.ClassDef, bool]]:
    classes: list[tuple[ast.ClassDef, bool]] = []

    def visit(node: ast.AST, inside_class_or_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                classes.append((child, inside_class_or_function))
                visit(child, True)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, True)
            else:
                visit(child, inside_class_or_function)

    visit(module, False)
    return classes


def inspect_class(class_node: ast.ClassDef, aliases: dict[str, set[str]]) -> dict[str, Any] | None:
    base_texts = [safe_unparse(base) for base in class_node.bases]
    confidence = candidate_confidence(class_node, aliases)
    if confidence is None:
        return None

    return {
        "className": class_node.name,
        "lineNumber": class_node.lineno,
        "bases": base_texts,
        "confidence": confidence,
        "constructor": inspect_constructor(class_node),
        "forward": inspect_forward(class_node),
    }


def candidate_confidence(class_node: ast.ClassDef, aliases: dict[str, set[str]]) -> str | None:
    base_texts = [safe_unparse(base) for base in class_node.bases]
    for base_text in base_texts:
        for torch_alias in aliases["torch"]:
            if base_text == f"{torch_alias}.nn.Module":
                return "confirmed"
        for nn_alias in aliases["torch_nn"]:
            if base_text == f"{nn_alias}.Module":
                return "confirmed"
        if base_text in aliases["module"]:
            return "likely"

    if find_method(class_node, "forward") or model_like_name(class_node.name):
        return "possible"

    return None


def model_like_name(class_name: str) -> bool:
    lowered = class_name.lower()
    return any(token in lowered for token in ("model", "module", "net"))


def inspect_constructor(class_node: ast.ClassDef) -> dict[str, Any]:
    init_method = find_method(class_node, "__init__")
    if init_method is None:
        return {
            "kind": "implicit",
            "supportsNoArgumentConstruction": True,
            "parameters": [],
        }

    parameters, _, _ = inspect_parameters(init_method.args)
    required_parameters = [parameter for parameter in parameters if parameter["required"]]
    return {
        "kind": "explicit",
        "supportsNoArgumentConstruction": len(required_parameters) == 0,
        "parameters": parameters,
    }


def inspect_forward(class_node: ast.ClassDef) -> dict[str, Any] | None:
    forward = find_method(class_node, "forward")
    if forward is None:
        return None

    parameters, has_varargs, has_kwargs = inspect_parameters(forward.args)
    return {
        "parameters": parameters,
        "hasVarArgs": has_varargs,
        "hasVarKwargs": has_kwargs,
        "varArgName": forward.args.vararg.arg if forward.args.vararg else None,
        "varKwargName": forward.args.kwarg.arg if forward.args.kwarg else None,
        "lineNumber": forward.lineno,
        "isAsync": isinstance(forward, ast.AsyncFunctionDef),
    }


def find_method(class_node: ast.ClassDef, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name:
            return item
    return None


def inspect_parameters(args: ast.arguments) -> tuple[list[dict[str, Any]], bool, bool]:
    positional_args = list(args.posonlyargs) + list(args.args)
    positional_defaults = [None] * (len(positional_args) - len(args.defaults)) + list(args.defaults)
    parameters: list[dict[str, Any]] = []

    for index, (argument, default) in enumerate(zip(positional_args, positional_defaults)):
        if index == 0 and argument.arg == "self":
            continue
        position = "positional_only" if index < len(args.posonlyargs) else "positional_or_keyword"
        parameters.append(parameter_result(argument, position, default))

    for argument, default in zip(args.kwonlyargs, args.kw_defaults):
        parameters.append(parameter_result(argument, "keyword_only", default))

    return parameters, args.vararg is not None, args.kwarg is not None


def parameter_result(argument: ast.arg, position: str, default: ast.expr | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": argument.arg,
        "position": position,
        "required": default is None,
    }
    if argument.annotation is not None:
        result["annotationText"] = safe_unparse(argument.annotation)
    if default is not None:
        literal = safe_literal(default)
        if literal["isLiteral"]:
            result["defaultValue"] = literal["value"]
        else:
            result["defaultDisplay"] = safe_unparse(default)
    return result


def safe_literal(node: ast.AST) -> dict[str, Any]:
    if isinstance(node, ast.Constant) and (
        node.value is None
        or isinstance(node.value, (bool, int, float, str))
    ):
        return {"isLiteral": True, "value": node.value}

    if isinstance(node, (ast.List, ast.Tuple)):
        values = [safe_literal(item) for item in node.elts]
        if all(item["isLiteral"] for item in values):
            return {"isLiteral": True, "value": [item["value"] for item in values]}

    if isinstance(node, ast.Dict):
        keys = [safe_literal(key) for key in node.keys]
        values = [safe_literal(item) for item in node.values]
        if all(key["isLiteral"] and isinstance(key["value"], str) for key in keys) and all(item["isLiteral"] for item in values):
            return {
                "isLiteral": True,
                "value": {key["value"]: item["value"] for key, item in zip(keys, values)},
            }

    return {"isLiteral": False}


def safe_unparse(node: ast.AST) -> str:
    return ast.unparse(node)
