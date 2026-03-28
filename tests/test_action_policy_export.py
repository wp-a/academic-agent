import unittest

from train_agent.scripts.export_scifact_action_policy_data import sanitize_action_record, summarize_action_records


class ActionPolicyExportTest(unittest.TestCase):
    def test_sanitize_action_record_drops_metadata(self):
        record = {
            "trajectory_id": "traj-1",
            "step_id": 0,
            "task": "next_action_classification",
            "text": "state text",
            "label": "search",
            "label_text": "{\"action_type\": \"search\"}",
            "metadata": {"reward": 0.05},
        }
        clean = sanitize_action_record(record)
        self.assertEqual(sorted(clean.keys()), ["label", "label_text", "step_id", "task", "text", "trajectory_id"])

    def test_summarize_action_records_counts_actions(self):
        records = [
            {"label": "search"},
            {"label": "quote_evidence"},
            {"label": "stop"},
            {"label": "search"},
        ]
        summary = summarize_action_records(records, episodes=2)
        self.assertEqual(summary["num_examples"], 4)
        self.assertEqual(summary["action_counts"]["search"], 2)
        self.assertAlmostEqual(summary["average_steps"], 2.0, places=6)
