import torch
from typing import Any 
from util.tensor import tensor_value_record
from util.util import jsonable
# ============================================================
# Output flattening
# ============================================================

def flatten_outputs(value: Any, *, max_preview_items: int):
    outputs = []

    def visit(v):
        idx = len(outputs)

        if isinstance(v, torch.Tensor):
            outputs.append(tensor_value_record(v, idx, max_preview_items=max_preview_items))
        elif isinstance(v, (list, tuple)):
            for item in v:
                visit(item)
        elif isinstance(v, dict):
            for item in v.values():
                visit(item)
        else:
            outputs.append({
                "index": idx,
                "role": "constant",
                "value": jsonable(v),
            })

    visit(value)
    return outputs
