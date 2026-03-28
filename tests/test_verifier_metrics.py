import unittest

from train_agent.eval.verifier_metrics import compute_verifier_metrics


class VerifierMetricsTest(unittest.TestCase):
    def test_compute_verifier_metrics_reports_classification_and_ranking(self):
        logits = [
            [0.1, 0.3, 4.2],
            [0.2, 3.6, 0.1],
            [4.5, 0.2, 0.1],
            [0.1, 2.8, 0.1],
        ]
        labels = [2, 1, 0, 1]
        label_names = ["CONTRADICT", "NEUTRAL", "SUPPORT"]
        group_ids = ["claim-a", "claim-a", "claim-b", "claim-b"]

        metrics = compute_verifier_metrics(
            logits=logits,
            labels=labels,
            label_names=label_names,
            group_ids=group_ids,
        )

        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0)
        self.assertAlmostEqual(metrics["mrr"], 1.0)
        self.assertAlmostEqual(metrics["recall@1"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], [[1, 0, 0], [0, 2, 0], [0, 0, 1]])
        self.assertAlmostEqual(metrics["per_class"]["CONTRADICT"]["precision"], 1.0)
        self.assertAlmostEqual(metrics["per_class"]["NEUTRAL"]["recall"], 1.0)
        self.assertAlmostEqual(metrics["per_class"]["SUPPORT"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
