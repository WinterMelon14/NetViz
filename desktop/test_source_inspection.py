import unittest

from desktop.source_inspection import MAX_SOURCE_CHARS, inspect_model_source, inspect_model_source_request


def candidates(source: str):
    result = inspect_model_source(source)
    if not result["ok"]:
        raise AssertionError(result)
    return result["candidates"]


class SourceInspectionTests(unittest.TestCase):
    def test_empty_source(self):
        result = inspect_model_source("   ")

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "source_empty")

    def test_source_too_large(self):
        result = inspect_model_source("x" * (MAX_SOURCE_CHARS + 1))

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "source_too_large")

    def test_valid_torch_nn_module_subclass(self):
        found = candidates("""
import torch
class MyModel(torch.nn.Module):
    def forward(self, x):
        return x
""")

        self.assertEqual(found[0]["className"], "MyModel")
        self.assertEqual(found[0]["confidence"], "confirmed")

    def test_torch_nn_alias(self):
        found = candidates("""
import torch.nn as nn
class AliasModel(nn.Module):
    def forward(self, x):
        return x
""")

        self.assertEqual(found[0]["confidence"], "confirmed")

    def test_from_torch_import_nn(self):
        found = candidates("""
from torch import nn
class ImportModel(nn.Module):
    def forward(self, x):
        return x
""")

        self.assertEqual(found[0]["confidence"], "confirmed")

    def test_from_torch_nn_import_module(self):
        found = candidates("""
from torch.nn import Module
class DirectModel(Module):
    def forward(self, x):
        return x
""")

        self.assertEqual(found[0]["confidence"], "likely")

    def test_multiple_model_classes(self):
        found = candidates("""
import torch
class FirstModel(torch.nn.Module):
    def forward(self, x): return x
class SecondModel(torch.nn.Module):
    def forward(self, x): return x
""")

        self.assertEqual([candidate["className"] for candidate in found], ["FirstModel", "SecondModel"])

    def test_class_without_forward(self):
        found = candidates("""
import torch
class NoForward(torch.nn.Module):
    pass
""")

        self.assertIsNone(found[0]["forward"])

    def test_required_constructor_arguments(self):
        found = candidates("""
import torch
class NeedsWidth(torch.nn.Module):
    def __init__(self, width):
        pass
    def forward(self, x): return x
""")

        self.assertFalse(found[0]["constructor"]["supportsNoArgumentConstruction"])

    def test_optional_constructor_arguments(self):
        found = candidates("""
import torch
class OptionalWidth(torch.nn.Module):
    def __init__(self, width=4):
        pass
    def forward(self, x): return x
""")

        self.assertTrue(found[0]["constructor"]["supportsNoArgumentConstruction"])
        self.assertEqual(found[0]["constructor"]["parameters"][0]["defaultValue"], 4)

    def test_required_forward_parameter(self):
        found = candidates("""
import torch
class ForwardRequired(torch.nn.Module):
    def forward(self, x):
        return x
""")

        parameter = found[0]["forward"]["parameters"][0]
        self.assertEqual(parameter["name"], "x")
        self.assertTrue(parameter["required"])

    def test_optional_forward_parameter(self):
        found = candidates("""
import torch
class ForwardOptional(torch.nn.Module):
    def forward(self, x=None):
        return x
""")

        parameter = found[0]["forward"]["parameters"][0]
        self.assertFalse(parameter["required"])
        self.assertIsNone(parameter["defaultValue"])

    def test_keyword_only_parameters(self):
        found = candidates("""
import torch
class KwOnly(torch.nn.Module):
    def forward(self, x, *, mask=True):
        return x
""")

        parameter = found[0]["forward"]["parameters"][1]
        self.assertEqual(parameter["position"], "keyword_only")
        self.assertEqual(parameter["defaultValue"], True)

    def test_positional_only_parameters(self):
        found = candidates("""
import torch
class PosOnly(torch.nn.Module):
    def forward(self, x, /, y):
        return x
""")

        self.assertEqual(found[0]["forward"]["parameters"][0]["position"], "positional_only")
        self.assertEqual(found[0]["forward"]["parameters"][1]["position"], "positional_or_keyword")

    def test_varargs_and_kwargs(self):
        found = candidates("""
import torch
class Variadic(torch.nn.Module):
    def forward(self, x, *args, **kwargs):
        return x
""")

        forward = found[0]["forward"]
        self.assertTrue(forward["hasVarArgs"])
        self.assertTrue(forward["hasVarKwargs"])
        self.assertEqual(forward["varArgName"], "args")
        self.assertEqual(forward["varKwargName"], "kwargs")

    def test_type_annotations(self):
        found = candidates("""
import torch
class Annotated(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
""")

        self.assertEqual(found[0]["forward"]["parameters"][0]["annotationText"], "torch.Tensor")

    def test_literal_defaults(self):
        found = candidates("""
import torch
class Defaults(torch.nn.Module):
    def forward(self, x, dims=(1, 2), names=['a', 'b']):
        return x
""")

        params = found[0]["forward"]["parameters"]
        self.assertEqual(params[1]["defaultValue"], [1, 2])
        self.assertEqual(params[2]["defaultValue"], ["a", "b"])

    def test_non_literal_defaults(self):
        found = candidates("""
import torch
class NonLiteral(torch.nn.Module):
    def forward(self, x, device=torch.device('cpu')):
        return x
""")

        parameter = found[0]["forward"]["parameters"][1]
        self.assertNotIn("defaultValue", parameter)
        self.assertEqual(parameter["defaultDisplay"], "torch.device('cpu')")

    def test_syntax_error(self):
        result = inspect_model_source("class Broken(:\n    pass\n")

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "source_syntax_error")
        self.assertEqual(result["error"]["details"]["line"], 1)

    def test_no_candidate_models(self):
        found = candidates("""
class PlainThing:
    pass
""")

        self.assertEqual(found, [])

    def test_nested_class_definitions(self):
        result = inspect_model_source("""
import torch
class Outer:
    class InnerModel(torch.nn.Module):
        def forward(self, x): return x
""")

        self.assertTrue(result["ok"])
        self.assertEqual(result["candidates"][0]["className"], "InnerModel")
        self.assertEqual(result["warnings"][0]["code"], "nested_class")

    def test_decorated_methods(self):
        found = candidates("""
import torch
class Decorated(torch.nn.Module):
    @staticmethod
    def helper():
        return None
    @custom_decorator
    def forward(self, x):
        return x
""")

        self.assertEqual(found[0]["forward"]["parameters"][0]["name"], "x")

    def test_async_forward_reported_honestly(self):
        result = inspect_model_source("""
import torch
class AsyncModel(torch.nn.Module):
    async def forward(self, x):
        return x
""")

        self.assertTrue(result["ok"])
        self.assertTrue(result["candidates"][0]["forward"]["isAsync"])
        self.assertEqual(result["warnings"][0]["code"], "async_forward")

    def test_invalid_request_shape(self):
        result = inspect_model_source_request({"sourceText": 123})

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "source_protocol_error")


if __name__ == "__main__":
    unittest.main()
