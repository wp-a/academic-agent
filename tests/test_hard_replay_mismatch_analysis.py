import importlib
import json
import tempfile
import unittest
from pathlib import Path


def _write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


class HardReplayMismatchAnalysisTest(unittest.TestCase):
    def test_analyze_mismatch_files_summarizes_failure_buckets(self):
        module = importlib.import_module("train_agent.scripts.analyze_hard_replay_mismatches")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            diagnostics_path = tmpdir_path / "diagnostics.jsonl"
            action_path = tmpdir_path / "off_policy_action.jsonl"
            stop_path = tmpdir_path / "off_policy_stop.jsonl"

            diagnostics_records = [
                {
                    "episode_id": "ep-stop-early",
                    "claim": "Claim 1",
                    "label_hint": "CONTRADICT",
                    "reference_policy_type": "conservative",
                    "post_quote_search_budget": 1,
                    "num_steps": 5,
                    "num_mismatches": 1,
                    "mismatch_step_indices": [4],
                    "steps": [
                        {
                            "step_index": 4,
                            "reference_action": "quote_evidence",
                            "predicted_action": "stop",
                            "action_match": False,
                            "stop_policy_should_stop": True,
                            "suppressed_stop": False,
                            "revealed_evidence": [{"doc_id": "doc-1"}],
                            "quoted_evidence": [],
                            "done": True,
                            "info": {"success_stop": False, "early_stop": True},
                        }
                    ],
                },
                {
                    "episode_id": "ep-oversearch",
                    "claim": "Claim 2",
                    "label_hint": "CONTRADICT",
                    "reference_policy_type": "conservative",
                    "post_quote_search_budget": 1,
                    "num_steps": 5,
                    "num_mismatches": 1,
                    "mismatch_step_indices": [3],
                    "steps": [
                        {
                            "step_index": 3,
                            "reference_action": "stop",
                            "predicted_action": "search",
                            "action_match": False,
                            "stop_policy_should_stop": False,
                            "suppressed_stop": False,
                            "revealed_evidence": [{"doc_id": "doc-2"}],
                            "quoted_evidence": [{"doc_id": "doc-2"}],
                            "done": False,
                            "info": {},
                        }
                    ],
                },
            ]
            _write_jsonl(diagnostics_path, diagnostics_records)

            action_records = [
                {
                    "trajectory_id": "ep-stop-early",
                    "step_id": 4,
                    "label": "quote_evidence",
                    "metadata": {
                        "episode_id": "ep-stop-early",
                        "student_action": "stop",
                        "reference_action": "quote_evidence",
                        "is_first_off_policy_step": True,
                        "stop_policy_should_stop": True,
                    },
                },
                {
                    "trajectory_id": "ep-oversearch",
                    "step_id": 3,
                    "label": "stop",
                    "metadata": {
                        "episode_id": "ep-oversearch",
                        "student_action": "search",
                        "reference_action": "stop",
                        "is_first_off_policy_step": True,
                        "stop_policy_should_stop": False,
                    },
                },
                {
                    "trajectory_id": "ep-oversearch",
                    "step_id": 4,
                    "label": "stop",
                    "metadata": {
                        "episode_id": "ep-oversearch",
                        "student_action": "stop",
                        "reference_action": "stop",
                        "is_first_off_policy_step": False,
                        "stop_policy_should_stop": True,
                    },
                },
            ]
            _write_jsonl(action_path, action_records)

            stop_records = [
                {"trajectory_id": "ep-stop-early", "step_id": 4, "label": "no", "metadata": {"episode_id": "ep-stop-early"}},
                {"trajectory_id": "ep-oversearch", "step_id": 3, "label": "yes", "metadata": {"episode_id": "ep-oversearch"}},
                {"trajectory_id": "ep-oversearch", "step_id": 4, "label": "yes", "metadata": {"episode_id": "ep-oversearch"}},
            ]
            _write_jsonl(stop_path, stop_records)

            summary = module.analyze_mismatch_files(
                diagnostics_path=diagnostics_path,
                off_policy_action_path=action_path,
                off_policy_stop_path=stop_path,
            )

            self.assertEqual(summary["mismatch_episode_count"], 2)
            self.assertEqual(summary["mismatch_step_count"], 2)
            self.assertEqual(summary["bucket_counts"]["premature_stop_after_evidence"], 1)
            self.assertEqual(summary["bucket_counts"]["oversearch_after_quote"], 1)
            self.assertEqual(summary["first_mismatch_action_pairs"]["quote_evidence->stop"], 1)
            self.assertEqual(summary["first_mismatch_action_pairs"]["stop->search"], 1)
            self.assertEqual(summary["off_policy_action_example_count"], 3)
            self.assertEqual(summary["off_policy_stop_example_count"], 3)
            self.assertEqual(summary["episodes"][0]["episode_id"], "ep-oversearch")
            self.assertEqual(summary["episodes"][1]["episode_id"], "ep-stop-early")


if __name__ == "__main__":
    unittest.main()
