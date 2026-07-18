import unittest
import json
import tempfile
from pathlib import Path

from desktop.compatibility_schema import CompatibilityReportError, validate_compatibility_report
from desktop.source_inspection import MAX_SOURCE_CHARS, inspect_model_source, inspect_model_source_request


def candidates(source: str):
    result = inspect_model_source(source)
    if not result["ok"]:
        raise AssertionError(result)
    return result["candidates"]


class SourceInspectionTests(unittest.TestCase):
    def test_detects_example_provider_and_linear_input_suggestion(self):
        result = inspect_model_source("""
import torch
def netviz_example_inputs(): return (torch.randn(1, 12),)
class Demo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(12, 4)
    def forward(self, x): return self.linear(x)
""")
        self.assertEqual(result["exampleInputProvider"], "netviz_example_inputs")
        suggestion = result["candidates"][0]["forward"]["inputSuggestions"][0]
        self.assertEqual(suggestion["shapeTemplate"], [1, 12])
        self.assertTrue(any("12 final features" in evidence for evidence in suggestion["evidence"]))

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
        self.assertEqual(found[0]["constructor"]["parameters"][0]["typeText"], "int")

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

    def test_compatibility_contract_fixture_and_versions(self):
        fixture_path = Path(__file__).parent / "fixtures" / "compatibility_report_v1.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        self.assertIs(validate_compatibility_report(fixture), fixture)
        with self.assertRaises(CompatibilityReportError):
            validate_compatibility_report({**fixture, "schemaVersion": 2})
        malformed = {**fixture, "findings": [{**fixture["findings"][0], "status": "maybe"}]}
        with self.assertRaises(CompatibilityReportError):
            validate_compatibility_report(malformed)

    def test_report_covers_parameter_categories_without_execution(self):
        result = inspect_model_source('''
raise RuntimeError("static inspection executed source")
import torch
class SignatureModel(torch.nn.Module):
    def __init__(self, width: int, expansion=2, *, label="demo"):
        super().__init__()
    def forward(self, x: torch.Tensor, /, y=None, *extras, mask, flag=True, **options):
        return x
''')
        self.assertTrue(result["ok"])
        candidate = result["candidates"][0]
        constructor = candidate["constructor"]["parameters"]
        forward = candidate["forward"]["parameters"]
        self.assertEqual([item["name"] for item in constructor], ["width", "expansion", "label"])
        self.assertEqual(constructor[0]["compatibilityStatus"], "configuration_required")
        self.assertTrue(constructor[1]["omittable"])
        self.assertEqual(forward[0]["position"], "positional_only")
        self.assertEqual(next(item for item in forward if item["name"] == "mask")["compatibilityStatus"], "configuration_required")
        codes = {item["code"] for item in candidate["compatibilityReport"]["findings"]}
        self.assertIn("forward_varargs", codes)
        self.assertIn("forward_varkwargs", codes)
        self.assertNotIn("forward_required_keyword_only", codes)

    def test_report_keeps_tuple_defaults_and_partial_input_facts(self):
        result = inspect_model_source('''
import torch
class PartialModel(torch.nn.Module):
    def __init__(self, dims=(2, 3)):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3)
    def forward(self, image):
        return self.conv(image)
''')
        candidate = result["candidates"][0]
        self.assertEqual(candidate["constructor"]["parameters"][0]["defaultKind"], "tuple")
        findings = candidate["compatibilityReport"]["findings"]
        shape = next(item for item in findings if item["code"] == "input_shape_partially_known")
        dtype = next(item for item in findings if item["category"] == "input" and item["code"] == "input_dtype_unresolved")
        self.assertEqual(shape["status"], "unknown")
        self.assertEqual(dtype["status"], "unknown")

    def test_report_inventories_imports_resources_and_runtime_constraints(self):
        result = inspect_model_source('''
import json
import torch
import optional_dependency as optional
from .layers import Block
class ResourceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.load("weights.pt")
    def forward(self, x): return x
''')
        findings = result["candidates"][0]["compatibilityReport"]["findings"]
        by_code = {item["code"] for item in findings}
        self.assertIn("import_standard_library", by_code)
        self.assertIn("import_runtime_dependency", by_code)
        self.assertIn("import_runtime_dependency_unavailable", by_code)
        self.assertIn("import_relative_local", by_code)
        self.assertIn("resource_reference_likely", by_code)
        self.assertIn("runtime_device_cpu", by_code)
        self.assertIn("runtime_timeout", by_code)

    def test_project_context_discovers_local_modules_and_resources(self):
        with tempfile.TemporaryDirectory() as temp_root:
            root = Path(temp_root)
            (root / "layers.py").write_text("class Block: pass\n", encoding="utf-8")
            (root / "weights.pt").write_bytes(b"checkpoint")
            source_path = root / "model.py"
            source = '''
import torch
from layers import Block
class ResourceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.load("weights.pt")
    def forward(self, x): return x
'''
            source_path.write_text(source, encoding="utf-8")

            result = inspect_model_source(source, source_path, root)

        self.assertTrue(result["ok"])
        self.assertEqual(result["projectContext"]["entryRelativePath"], "model.py")
        self.assertEqual(result["projectContext"]["localModules"][0]["path"], "layers.py")
        self.assertEqual(result["projectContext"]["resources"][0]["path"], "weights.pt")
        findings = result["candidates"][0]["compatibilityReport"]["findings"]
        codes = {item["code"] for item in findings}
        self.assertIn("import_local_project", codes)
        self.assertIn("resource_declared", codes)

    def test_report_flags_fx_hazards_as_unknown(self):
        result = inspect_model_source('''
import torch
class HazardModel(torch.nn.Module):
    def forward(self, x):
        self.last = x
        if x.item() > 0:
            return x
        for item in x:
            x = item
        return getattr(self, "layer")(x)
''')
        findings = result["candidates"][0]["compatibilityReport"]["findings"]
        hazards = [item for item in findings if item["category"] == "fx"]
        self.assertEqual({item["status"] for item in hazards}, {"unknown"})
        self.assertTrue({"fx_tensor_item", "fx_tensor_dependent_branch", "fx_tensor_iteration", "fx_dynamic_module_selection", "fx_forward_module_mutation"}.issubset({item["code"] for item in hazards}))
        self.assertEqual(result["candidates"][0]["compatibilityReport"]["tracingOutcome"]["status"], "unknown")


if __name__ == "__main__":
    unittest.main()
