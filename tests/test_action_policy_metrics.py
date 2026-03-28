import unittest

from train_agent.eval.action_policy_metrics import compute_action_policy_metrics


class ActionPolicyMetricsTest(unittest.TestCase):
    def test_compute_action_policy_metrics_returns_per_action_scores(self):
        metrics = compute_action_policy_metrics(
            logits=[
                [3.0, 1.0, 0.0],
                [0.2, 2.1, 0.1],
                [0.1, 1.0, 2.5],
                [2.1, 0.5, 0.4],
            ],
            labels=[0, 1, 2, 1],
            label_names=["quote_evidence", "search", "stop"],
        )
        self.assertAlmostEqual(metrics["accuracy"], 0.75, places=6)
        self.assertIn("macro_f1", metrics)
        self.assertEqual(metrics["confusion_matrix"][1][0], 1)
        self.assertIn("search", metrics["per_class"])
        self.assertGreater(metrics["per_class"]["stop"]["f1"], 0.0)
