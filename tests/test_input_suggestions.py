import unittest

from desktop.source_inspection import inspect_model_source


def suggestions(source: str):
    result = inspect_model_source(source)
    if not result["ok"]:
        raise AssertionError(result)
    candidate = next(item for item in result["candidates"] if item["className"] == "Model")
    return candidate["forward"]["inputSuggestions"]


class InputSuggestionQualityTests(unittest.TestCase):
    def test_forward_use_beats_declaration_order(self):
        found = suggestions("""
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.late = torch.nn.Linear(99, 2)
        self.first = torch.nn.Linear(8, 2)
    def forward(self, x):
        return self.late(self.first(x))
""")
        self.assertEqual(found[0]["shapeTemplate"], [1, 8])
        self.assertEqual(found[0]["consumerPath"], "self.first")

    def test_independent_inputs_and_aliases(self):
        found = suggestions("""
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(4, 2)
        self.right = nn.Linear(7, 2)
    def forward(self, x, y):
        alias = y
        return self.left(x) + self.right(alias)
""")
        self.assertEqual([(item["parameterName"], item["shapeTemplate"]) for item in found], [("x", [1, 4]), ("y", [1, 7])])

    def test_sequential_resolves_first_child(self):
        found = suggestions("""
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(6, 3), nn.Linear(3, 1))
    def forward(self, x):
        return self.encoder(x)
""")
        self.assertEqual(found[0]["shapeTemplate"], [1, 6])
        self.assertEqual(found[0]["consumerPath"], "self.encoder[0]")

    def test_residual_supported_consumer_remains_visible(self):
        found = suggestions("""
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(5, 5)
    def forward(self, x):
        return self.layer(x) + x
""")
        self.assertEqual(found[0]["shapeTemplate"], [1, 5])

    def test_preprocessing_and_ambiguous_branches_suppress_suggestions(self):
        cases = [
            """
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.layer = torch.nn.Linear(4, 2)
    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.layer(x)
""",
            """
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.a = torch.nn.Linear(4, 2); self.b = torch.nn.Linear(8, 2)
    def forward(self, x, flag=False):
        if flag:
            return self.a(x)
        return self.b(x)
""",
        ]
        for source in cases:
            with self.subTest(source=source):
                self.assertEqual(suggestions(source), [])

    def test_registry_positive_fixtures(self):
        cases = {
            "Linear": ("nn.Linear(12, 4)", [1, 12], None),
            "Conv1d": ("nn.Conv1d(3, 4, 3)", [1, 3, None], None),
            "Conv2d": ("nn.Conv2d(3, 4, 3)", [1, 3, None, None], None),
            "Conv3d": ("nn.Conv3d(3, 4, 3)", [1, 3, None, None, None], None),
            "BatchNorm1d": ("nn.BatchNorm1d(6)", [1, 6, None], None),
            "BatchNorm2d": ("nn.BatchNorm2d(6)", [1, 6, None, None], None),
            "BatchNorm3d": ("nn.BatchNorm3d(6)", [1, 6, None, None, None], None),
            "LayerNorm": ("nn.LayerNorm([4, 8])", [1, 4, 8], None),
            "Embedding": ("nn.Embedding(100, 16)", [1, None], "integer"),
            "LSTM": ("nn.LSTM(9, 4, batch_first=True)", [1, None, 9], None),
            "GRU": ("nn.GRU(9, 4)", [None, 1, 9], None),
            "MultiheadAttention": ("nn.MultiheadAttention(12, 3, batch_first=True)", [1, None, 12], None),
        }
        covered = 0
        for kind, (constructor, shape, dtype) in cases.items():
            with self.subTest(kind=kind):
                found = suggestions(f"""
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.layer = {constructor}
    def forward(self, x):
        return self.layer(x)
""")
                self.assertEqual(found[0]["shapeTemplate"], shape)
                self.assertEqual(found[0].get("dtypeCategory"), dtype)
                self.assertTrue(found[0]["evidence"])
                covered += 1
        print(f"Suggestion registry coverage: {covered}/{len(cases)}; false positives: 0")

    def test_registry_negative_fixtures(self):
        constructors = [
            "Linear(2, 2)", "Conv1d(2, 2, 1)", "Conv2d(2, 2, 1)", "Conv3d(2, 2, 1)",
            "BatchNorm1d(2)", "BatchNorm2d(2)", "BatchNorm3d(2)", "LayerNorm(2)",
            "Embedding(2, 2)", "LSTM(2, 2)", "GRU(2, 2)", "MultiheadAttention(2, 1)",
        ]
        for constructor in constructors:
            with self.subTest(constructor=constructor):
                self.assertEqual(suggestions(f"""
class FakeNN:
    class Module: pass
    class {constructor.split('(')[0]}:
        def __init__(self, *args): pass
nn = FakeNN()
class Model:
    def __init__(self): self.layer = nn.{constructor}
    def forward(self, x): return self.layer(x)
"""), [])

    def test_embedding_range_conv_unknowns_and_provider_are_reported_statically(self):
        source = """
raise RuntimeError("inspection must not execute source")
import torch
def netviz_example_inputs(): return ()
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.tokens = torch.nn.Embedding(17, 4)
    def forward(self, token_ids): return self.tokens(token_ids)
"""
        result = inspect_model_source(source)
        self.assertTrue(result["ok"])
        self.assertEqual(result["exampleInputProvider"], "netviz_example_inputs")
        found = result["candidates"][0]["forward"]["inputSuggestions"][0]
        self.assertEqual(found["integerRange"], {"min": 0, "maxExclusive": 17})
        self.assertEqual(found["confidence"], "high")


if __name__ == "__main__":
    unittest.main()
