import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from train_agent.scripts import eval_action_policy_predictions


class DummyPolicy:
    def __init__(self, *_args, **_kwargs):
        self.label_names = ["quote_evidence", "search", "stop"]

    def predict_logits(self, texts):
        self.seen_texts = list(texts)
        return [
            [0.1, 2.5, 0.3],
            [3.0, 0.2, 0.1],
            [0.2, 0.3, 2.2],
        ]


class DummyPolicyWithMistake:
    def __init__(self, *_args, **_kwargs):
        self.label_names = ["quote_evidence", "search", "stop"]

    def predict_logits(self, texts):
        self.seen_texts = list(texts)
        return [
            [0.1, 2.5, 0.3],
            [0.1, 2.2, 0.1],
            [0.2, 0.3, 2.2],
        ]


class ActionPolicyEvalScriptTest(unittest.TestCase):
    def test_evaluate_action_policy_file_reports_metrics_and_distributions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            eval_file = root / "eval.jsonl"
            rows = [
                {
                    "trajectory_id": "traj-1",
                    "step_id": 0,
                    "task": "next_action_classification",
                    "text": "Need more evidence search",
                    "label": "search",
                    "label_text": "{\"action_type\": \"search\"}",
                },
                {
                    "trajectory_id": "traj-1",
                    "step_id": 1,
                    "task": "next_action_classification",
                    "text": "Quote the strongest snippet",
                    "label": "quote_evidence",
                    "label_text": "{\"action_type\": \"quote_evidence\"}",
                },
                {
                    "trajectory_id": "traj-1",
                    "step_id": 2,
                    "task": "next_action_classification",
                    "text": "Evidence is sufficient stop",
                    "label": "stop",
                    "label_text": "{\"action_type\": \"stop\"}",
                },
            ]
            eval_file.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            with patch("train_agent.scripts.eval_action_policy_predictions.FrozenActionPolicy", DummyPolicy):
                summary = eval_action_policy_predictions.evaluate_action_policy_file(
                    model_dir=root / "model",
                    eval_file=eval_file,
                    max_length=256,
                    batch_size=8,
                    attn_implementation="sdpa",
                )

        self.assertEqual(summary["num_examples"], 3)
        self.assertEqual(summary["label_names"], ["quote_evidence", "search", "stop"])
        self.assertAlmostEqual(summary["accuracy"], 1.0, places=6)
        self.assertAlmostEqual(summary["macro_f1"], 1.0, places=6)
        self.assertEqual(summary["gold_distribution"]["search"], 1)
        self.assertEqual(summary["prediction_distribution"]["quote_evidence"], 1)
        self.assertEqual(summary["confusion_matrix"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_evaluate_action_policy_file_can_return_prediction_and_error_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            eval_file = root / "eval.jsonl"
            rows = [
                {
                    "trajectory_id": "traj-1",
                    "step_id": 0,
                    "task": "next_action_classification",
                    "text": "Need more evidence search",
                    "label": "search",
                    "label_text": "{\"action_type\": \"search\"}",
                },
                {
                    "trajectory_id": "traj-1",
                    "step_id": 1,
                    "task": "next_action_classification",
                    "text": "Quote the strongest snippet",
                    "label": "quote_evidence",
                    "label_text": "{\"action_type\": \"quote_evidence\"}",
                },
                {
                    "trajectory_id": "traj-1",
                    "step_id": 2,
                    "task": "next_action_classification",
                    "text": "Evidence is sufficient stop",
                    "label": "stop",
                    "label_text": "{\"action_type\": \"stop\"}",
                },
            ]
            eval_file.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            with patch("train_agent.scripts.eval_action_policy_predictions.FrozenActionPolicy", DummyPolicyWithMistake):
                summary = eval_action_policy_predictions.evaluate_action_policy_file(
                    model_dir=root / "model",
                    eval_file=eval_file,
                    max_length=256,
                    batch_size=8,
                    attn_implementation="sdpa",
                    include_predictions=True,
                )

        self.assertEqual(len(summary["prediction_rows"]), 3)
        self.assertEqual(len(summary["error_rows"]), 1)
        self.assertFalse(summary["prediction_rows"][1]["is_correct"])
        self.assertEqual(summary["error_rows"][0]["gold_label"], "quote_evidence")
        self.assertEqual(summary["error_rows"][0]["predicted_label"], "search")


if __name__ == "__main__":
    unittest.main()
