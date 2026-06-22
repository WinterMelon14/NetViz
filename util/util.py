from typing import Any
import torch
import torch.fx as fx

# ============================================================
# Small utilities
# ============================================================

def human_bytes(num_bytes: int | None) -> str | None:
    if num_bytes is None:
        return None

    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)

    for unit in units:
        if abs(value) < 1024.0:
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.2f} {unit}"
        value /= 1024.0

    return f"{value:.2f} PB"


def tensor_num_bytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def jsonable(x: Any):
    if isinstance(x, fx.Node):
        return x.name

    if isinstance(x, torch.Size):
        return list(x)

    if isinstance(x, torch.dtype):
        return str(x)

    if isinstance(x, torch.device):
        return str(x)

    if isinstance(x, slice):
        return {
            "start": jsonable(x.start),
            "stop": jsonable(x.stop),
            "step": jsonable(x.step),
        }

    if isinstance(x, (str, int, float, bool)) or x is None:
        return x

    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]

    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}

    if hasattr(x, "__name__"):
        return x.__name__

    return str(x)


def target_name(target):
    if isinstance(target, str):
        return target
    if hasattr(target, "__name__"):
        return target.__name__
    return str(target)


def node_kind(node: fx.Node):
    if node.op == "placeholder":
        return "input"
    if node.op == "call_module":
        return "module"
    if node.op == "call_function":
        return "function"
    if node.op == "call_method":
        return "method"
    if node.op == "get_attr":
        return "attribute"
    if node.op == "output":
        return "output"
    return node.op
