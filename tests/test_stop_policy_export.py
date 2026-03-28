import unittest

from train_agent.scripts.export_scifact_stop_policy_data import (
    convert_action_record_to_stop_record,
    summarize_stop_records,
)


class StopPolicyExportTest(unittest.TestCase):
    def test_convert_action_record_to_stop_record_maps_stop_action_to_yes(self):
        record = {
            "trajectory_id": "traj-1",
            "step_id": 2,
            "task": "next_action_classification",
            "text": "state text",
            "label": "stop",
            "label_text": '{"action_type": "stop"}',
            "metadata": {"termination_reason": "success_stop"},
        }
        converted = convert_action_record_to_stop_record(record)
        self.assertEqual(converted["task"], "stop_policy_classification")
        self.assertEqual(converted["label"], "yes")
        self.assertIn('"should_stop": "yes"', converted["label_text"])

    def test_summarize_stop_records_counts_yes_and_no_labels(self):
        records = [
            {"label": "no"},
            {"label": "no"},
            {"label": "yes"},
        ]
        summary = summarize_stop_records(records, episodes=2)
        self.assertEqual(summary["num_examples"], 3)
        self.assertEqual(summary["label_counts"]["no"], 2)
        self.assertEqual(summary["label_counts"]["yes"], 1)
        self.assertAlmostEqual(summary["label_distribution"]["yes"], 1 / 3, places=6)


if __name__ == "__main__":
    unittest.main()
