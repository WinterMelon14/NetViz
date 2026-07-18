import unittest
from pathlib import Path

from scripts.audit_bundle import forbidden_archive_modules, is_forbidden_release_path


class BundleAuditTests(unittest.TestCase):
    def test_netviz_development_files_are_rejected(self):
        rejected = (
            "tests/test_runtime.py",
            "_internal/docs/TRACE_SCHEMA.md",
            "_internal/frontend/src/main.tsx",
            "_internal/frontend/package.json",
            "_internal/desktop/host.py",
            "_internal/contract.py",
        )
        for path in rejected:
            with self.subTest(path=path):
                self.assertTrue(is_forbidden_release_path(Path(path)))

    def test_dependency_tests_sources_and_licenses_are_not_netviz_violations(self):
        allowed = (
            "_internal/numpy/_core/tests/test_api.py",
            "_internal/torch/fx/passes/tests/test_pass_manager.py",
            "_internal/torch/distributed/_composable/contract.py",
            "_internal/numpy-2.4.6.dist-info/licenses/numpy/_core/src/highway/LICENSE",
        )
        for path in allowed:
            with self.subTest(path=path):
                self.assertFalse(is_forbidden_release_path(Path(path)))

    def test_archive_matching_is_exact_for_application_modules(self):
        output = """
        contract
        torch.distributed._composable.contract
        networkx.algorithms.minors.contraction
        transformers.models.auto
        """
        self.assertEqual(forbidden_archive_modules(output), ["contract", "transformers.models.auto"])


if __name__ == "__main__":
    unittest.main()
