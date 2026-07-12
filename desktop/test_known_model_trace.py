import unittest

from desktop.known_model import TestModel, known_model_input
from util.summary import model_summary


class KnownModelTraceTests(unittest.TestCase):
    def test_known_model_produces_labeled_connected_trace(self):
        payload = model_summary(TestModel(), known_model_input())

        nodes = {node["id"]: node for node in payload["graph"]["nodes"]}
        self.assertEqual(payload["model_name"], "TestModel")
        self.assertEqual(nodes["x"]["label"], "x")
        self.assertEqual(nodes["linear"]["label"], "Linear")
        self.assertEqual(nodes["linear"]["module"]["type"], "Linear")
        self.assertEqual(nodes["linear"]["params"]["count"], 10)
        self.assertEqual(
            payload["graph"]["edges"],
            [{
                "id": "x:0->linear:0",
                "source": "x",
                "source_output": 0,
                "target": "linear",
                "target_input": 0,
            }],
        )


if __name__ == "__main__":
    unittest.main()
