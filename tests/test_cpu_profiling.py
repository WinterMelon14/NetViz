import unittest

import torch

from util.summary import model_summary


class CPUProfilingTests(unittest.TestCase):
    def test_ordinary_trace_has_no_profiling_payload(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
        payload = model_summary(model, torch.randn(1, 4), run_shape_prop=False)

        self.assertNotIn("profiling", payload)
        self.assertTrue(all("profile" not in node for node in payload["graph"]["nodes"]))

    def test_profile_payload_records_samples_and_critical_path(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
        payload = model_summary(
            model,
            torch.randn(1, 4),
            run_shape_prop=False,
            profile_config={"warmup_runs": 1, "measurement_runs": 3, "percentiles": [50, 90]},
        )

        profiling = payload["profiling"]
        self.assertEqual(profiling["schemaVersion"], 1)
        self.assertEqual(profiling["config"]["warmup_runs"], 1)
        self.assertEqual(profiling["config"]["measurement_runs"], 3)
        observed = [node for node in profiling["nodes"] if node["sample_count"]]
        self.assertTrue(observed)
        self.assertTrue(all(node["sample_count"] == 3 for node in observed))
        self.assertTrue(all(node["median_ms"] >= 0 for node in observed))
        self.assertTrue(profiling["critical_path"]["node_ids"])
        self.assertGreaterEqual(profiling["critical_path"]["total_ms"], 0)


if __name__ == "__main__":
    unittest.main()
