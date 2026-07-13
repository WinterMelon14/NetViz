"""Static, conservative representative-input suggestions from model source AST."""

import ast
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModuleSpec:
    kind: str
    args: list[ast.expr]
    kwargs: dict[str, ast.expr]
    path: str
    first_child: "ModuleSpec | None" = None


def inspect_input_suggestions(
    class_node: ast.ClassDef,
    parameter_names: list[str],
    aliases: dict[str, Any],
) -> list[dict[str, Any]]:
    forward = _find_method(class_node, "forward")
    if forward is None:
        return []
    modules = _inspect_modules(class_node, aliases)
    analyzer = _ForwardConsumerAnalyzer(parameter_names, modules)
    analyzer.visit_statements(forward.body)
    return [analyzer.suggestions[name] for name in parameter_names if name in analyzer.suggestions]


def _inspect_modules(class_node: ast.ClassDef, aliases: dict[str, Any]) -> dict[str, ModuleSpec]:
    init_method = _find_method(class_node, "__init__")
    if init_method is None:
        return {}
    defaults = _parameter_defaults(init_method)
    modules: dict[str, ModuleSpec] = {}
    for statement in init_method.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target, value = statement.targets[0], statement.value
        elif isinstance(statement, ast.AnnAssign):
            target, value = statement.target, statement.value
        path = _self_attribute(target)
        if path and isinstance(value, ast.Call):
            spec = _module_spec(value, f"self.{path}", aliases, defaults)
            if spec:
                modules[path] = spec
    return modules


def _module_spec(
    call: ast.Call,
    path: str,
    aliases: dict[str, Any],
    defaults: dict[str, Any],
) -> ModuleSpec | None:
    kind = _module_kind(call.func, aliases)
    if kind is None:
        return None
    kwargs = {item.arg: item.value for item in call.keywords if item.arg}
    first_child = None
    if kind == "Sequential":
        for child in call.args:
            if isinstance(child, ast.Call):
                first_child = _module_spec(child, f"{path}[0]", aliases, defaults)
                if first_child:
                    break
    normalized_args = [_replace_default_names(item, defaults) for item in call.args]
    normalized_kwargs = {name: _replace_default_names(item, defaults) for name, item in kwargs.items()}
    return ModuleSpec(kind, normalized_args, normalized_kwargs, path, first_child)


def _module_kind(function: ast.expr, aliases: dict[str, Any]) -> str | None:
    supported = {
        "Linear", "Conv1d", "Conv2d", "Conv3d", "BatchNorm1d", "BatchNorm2d",
        "BatchNorm3d", "LayerNorm", "Embedding", "LSTM", "GRU",
        "MultiheadAttention", "Sequential",
    }
    text = ast.unparse(function)
    direct_layers = aliases.get("torch_layers", {})
    if text in direct_layers and direct_layers[text] in supported:
        return direct_layers[text]
    for torch_alias in aliases.get("torch", set()):
        prefix = f"{torch_alias}.nn."
        if text.startswith(prefix) and text[len(prefix):] in supported:
            return text[len(prefix):]
    for nn_alias in aliases.get("torch_nn", set()):
        prefix = f"{nn_alias}."
        if text.startswith(prefix) and text[len(prefix):] in supported:
            return text[len(prefix):]
    return None


class _ForwardConsumerAnalyzer:
    def __init__(self, parameters: list[str], modules: dict[str, ModuleSpec]):
        self.aliases = {name: {name} for name in parameters}
        self.modules = modules
        self.suggestions: dict[str, dict[str, Any]] = {}
        self.resolved: set[str] = set()
        self.branch_depth = 0

    def visit_statements(self, statements: list[ast.stmt]) -> None:
        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(statement, ast.Assign):
                self.visit_expr(statement.value)
                roots = self.roots(statement.value)
                for target in statement.targets:
                    if isinstance(target, ast.Name) and roots:
                        self.aliases[target.id] = roots
            elif isinstance(statement, ast.AnnAssign) and statement.value:
                self.visit_expr(statement.value)
                roots = self.roots(statement.value)
                if isinstance(statement.target, ast.Name) and roots:
                    self.aliases[statement.target.id] = roots
            elif isinstance(statement, ast.If):
                self.visit_expr(statement.test)
                self.branch_depth += 1
                self.visit_statements(statement.body)
                self.visit_statements(statement.orelse)
                self.branch_depth -= 1
            elif isinstance(statement, (ast.For, ast.While, ast.With, ast.Try)):
                self.branch_depth += 1
                for child in ast.iter_child_nodes(statement):
                    if isinstance(child, ast.expr):
                        self.visit_expr(child)
                    elif isinstance(child, ast.stmt):
                        self.visit_statements([child])
                self.branch_depth -= 1
            else:
                for child in ast.iter_child_nodes(statement):
                    if isinstance(child, ast.expr):
                        self.visit_expr(child)

    def visit_expr(self, expression: ast.expr) -> None:
        if isinstance(expression, ast.Call):
            for argument in expression.args:
                self.visit_expr(argument)
            for keyword in expression.keywords:
                self.visit_expr(keyword.value)
            roots = set().union(
                *(self.roots(item) for item in expression.args),
                *(self.roots(item.value) for item in expression.keywords),
            )
            roots.update(self.roots(expression.func))
            unresolved = roots - self.resolved
            if not unresolved:
                return
            module_path = _self_call_path(expression.func)
            module = self.modules.get(module_path) if module_path else None
            if module and module.kind == "Sequential":
                module = module.first_child
            for parameter in unresolved:
                self.resolved.add(parameter)
                if self.branch_depth or module is None:
                    continue
                suggestion = _suggestion_for(module, parameter)
                if suggestion:
                    self.suggestions[parameter] = suggestion
            return
        for child in ast.iter_child_nodes(expression):
            if isinstance(child, ast.expr):
                self.visit_expr(child)

    def roots(self, expression: ast.AST | None) -> set[str]:
        if isinstance(expression, ast.Name):
            return set(self.aliases.get(expression.id, set()))
        roots: set[str] = set()
        if expression is not None:
            for child in ast.iter_child_nodes(expression):
                roots.update(self.roots(child))
        return roots


def _suggestion_for(module: ModuleSpec, parameter: str) -> dict[str, Any] | None:
    first = _literal(module.args[0]) if module.args else None
    base: dict[str, Any] = {
        "parameterName": parameter,
        "confidence": "high",
        "consumerPath": module.path,
        "evidence": [],
    }
    kind = module.kind
    if kind == "Linear":
        base.update(shapeTemplate=[1, first if isinstance(first, int) else None], dimensionSources=["default", "inferred" if isinstance(first, int) else "unknown"])
        base["evidence"] = [f"{module.path} resolves to Linear and consumes this parameter first.", f"Linear expects {first} final features." if isinstance(first, int) else "Linear's final feature size is not a source literal."]
    elif kind in {"Conv1d", "Conv2d", "Conv3d"}:
        rank = int(kind[4]) + 2
        base.update(shapeTemplate=[1, first if isinstance(first, int) else None] + [None] * (rank - 2), dimensionSources=["default", "inferred" if isinstance(first, int) else "unknown"] + ["unknown"] * (rank - 2))
        base["evidence"] = [f"{module.path} resolves to {kind} and consumes this parameter first.", f"{kind} expects {first} input channels." if isinstance(first, int) else f"{kind}'s input channel count is not a source literal."]
        if kind == "Conv2d":
            base["presetKind"] = "image"
    elif kind in {"BatchNorm1d", "BatchNorm2d", "BatchNorm3d"}:
        spatial = int(kind[-2])
        rank = spatial + 2
        base.update(shapeTemplate=[1, first if isinstance(first, int) else None] + [None] * (rank - 2), dimensionSources=["default", "inferred" if isinstance(first, int) else "unknown"] + ["unknown"] * (rank - 2), confidence="medium")
        base["evidence"] = [f"{module.path} resolves to {kind}.", f"{kind} is configured for {first} features/channels." if isinstance(first, int) else "Its feature/channel count is not a source literal."]
    elif kind == "LayerNorm":
        normalized_literal = first if isinstance(first, (list, tuple)) else [first] if isinstance(first, int) else []
        normalized = list(normalized_literal)
        base.update(shapeTemplate=[1] + normalized, dimensionSources=["default"] + ["inferred"] * len(normalized))
        base["evidence"] = [f"{module.path} resolves to LayerNorm and consumes this parameter first.", f"LayerNorm normalizes trailing dimensions {normalized}."]
    elif kind == "Embedding":
        vocabulary = first if isinstance(first, int) else None
        base.update(shapeTemplate=[1, None], dimensionSources=["default", "unknown"], dtypeCategory="integer", presetKind="sequence")
        if vocabulary:
            base["integerRange"] = {"min": 0, "maxExclusive": vocabulary}
        base["evidence"] = [f"{module.path} resolves to Embedding and consumes this parameter first.", f"Token IDs must be integers from 0 through {vocabulary - 1}." if vocabulary else "Embedding requires integer token IDs; the vocabulary bound is unknown."]
    elif kind in {"LSTM", "GRU", "MultiheadAttention"}:
        batch_first = _literal(module.kwargs.get("batch_first")) is True
        feature = first if isinstance(first, int) else None
        shape = [1, None, feature] if batch_first else [None, 1, feature]
        sources = ["default", "unknown", "inferred" if feature is not None else "unknown"] if batch_first else ["unknown", "default", "inferred" if feature is not None else "unknown"]
        base.update(shapeTemplate=shape, dimensionSources=sources, presetKind="sequence", confidence="low" if kind == "MultiheadAttention" else "medium")
        base["evidence"] = [f"{module.path} resolves to {kind}.", f"The configured feature size is {feature}; batch_first is {batch_first}." if feature is not None else f"The feature size is not a source literal; batch_first is {batch_first}."]
    else:
        return None
    return base


def _parameter_defaults(method: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    args = list(method.args.posonlyargs) + list(method.args.args)
    defaults = [None] * (len(args) - len(method.args.defaults)) + list(method.args.defaults)
    return {argument.arg: _literal(default) for argument, default in zip(args, defaults) if default is not None}


def _replace_default_names(expression: ast.expr, defaults: dict[str, Any]) -> ast.expr:
    if isinstance(expression, ast.Name) and expression.id in defaults and defaults[expression.id] is not None:
        value = defaults[expression.id]
        return ast.Constant(value) if not isinstance(value, (list, tuple)) else ast.List(elts=[ast.Constant(item) for item in value])
    return expression


def _literal(expression: ast.AST | None) -> Any:
    if expression is None:
        return None
    try:
        value = ast.literal_eval(expression)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, (bool, int, list, tuple)) else None


def _self_attribute(expression: ast.expr | None) -> str | None:
    if isinstance(expression, ast.Attribute) and isinstance(expression.value, ast.Name) and expression.value.id == "self":
        return expression.attr
    return None


def _self_call_path(expression: ast.expr) -> str | None:
    return _self_attribute(expression)


def _find_method(class_node: ast.ClassDef, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    return next((item for item in class_node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name), None)
