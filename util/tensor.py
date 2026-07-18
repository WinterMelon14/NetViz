import torch
from util.util import tensor_num_bytes, human_bytes
# ============================================================
# Tensor summaries
# ============================================================

def tensor_preview(t: torch.Tensor, max_items=8):
    with torch.no_grad():
        td = t.detach().cpu()
        flat = td.flatten()

        if flat.numel() == 0:
            return []

        preview = flat[:max_items]
        if preview.dtype == torch.bool:
            return preview.to(torch.int64).tolist()

        return preview.tolist()


def tensor_stats(t: torch.Tensor):
    with torch.no_grad():
        td = t.detach().cpu()
        flat = td.flatten()

        if flat.numel() == 0:
            return {
                "numel": 0,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "zeros_pct": None,
                "has_nan": False,
                "has_inf": False,
            }

        stats = {
            "numel": int(flat.numel()),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "mean": float(flat.float().mean()),
            "std": float(flat.float().std()) if flat.numel() > 1 else 0.0,
            "zeros_pct": 100.0* float((flat == 0).float().mean()),
        }

        if torch.is_floating_point(flat):
            stats["has_nan"] = bool(torch.isnan(flat).any())
            stats["has_inf"] = bool(torch.isinf(flat).any())
        else:
            stats["has_nan"] = False
            stats["has_inf"] = False

        return stats


def tensor_value_record(t: torch.Tensor, index: int, max_preview_items=8):
    num_bytes = tensor_num_bytes(t)

    return {
        "index": index,
        "role": "tensor",
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "preview": tensor_preview(t, max_items=max_preview_items),
        "summary": tensor_stats(t),
        "memory": {
            "num_bytes": num_bytes,
            "human": human_bytes(num_bytes),
        },
    }
