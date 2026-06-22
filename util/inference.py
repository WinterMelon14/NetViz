import torch
from typing import Any

# ============================================================
# Role inference
# ============================================================

def infer_constant_role(value: Any, *, arg_name: str | None = None):
    if arg_name in {"dim", "axis"}:
        return "dim"

    if arg_name in {"size", "shape"}:
        return "shape"

    if isinstance(value, torch.Size):
        return "shape"

    if isinstance(value, (int, float, bool)):
        return "scalar"

    if isinstance(value, (tuple, list)):
        if all(isinstance(v, int) for v in value):
            return "shape"

    return "constant"


def is_tensor_list_value(value: Any):
    return (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and all(isinstance(v, torch.Tensor) for v in value)
    )


