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


class MergeRelabelIntoTrainsetTest(unittest.TestCase):
    def test_merge_jsonl_files_prefers_relabel_records_on_duplicate_keys(self):
        module = importlib.import_module("train_agent.scripts.merge_relabel_into_trainset")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            base_path = tmpdir_path / "base.jsonl"
            relabel_path = tmpdir_path / "relabel.jsonl"
            output_path = tmpdir_path / "merged.jsonl"

            _write_jsonl(
                base_path,
                [
                    {
                        "trajectory_id": "ep-1",
                        "step_id": 0,
                        "task": "next_action_classification",
                        "text": "base-a",
                        "label": "search",
                        "label_text": "{\"action_type\": \"search\"}",
                    },
                    {
                        "trajectory_id": "ep-2",
                        "step_id": 1,
                        "task": "next_action_classification",
                        "text": "base-b",
                        "label": "stop",
                        "label_text": "{\"action_type\": \"stop\"}",
                    },
                ],
            )
            _write_jsonl(
                relabel_path,
                [
                    {
                        "trajectory_id": "ep-2",
                        "step_id": 1,
                        "task": "next_action_classification",
                        "text": "relabel-b",
                        "label": "quote_evidence",
                        "label_text": "{\"action_type\": \"quote_evidence\"}",
                        "metadata": {"source": "hard_off_policy_relabel_v1"},
                    },
                    {
                        "trajectory_id": "ep-3",
                        "step_id": 0,
                        "task": "next_action_classification",
                        "text": "relabel-c",
                        "label": "stop",
                        "label_text": "{\"action_type\": \"stop\"}",
                        "metadata": {"source": "hard_off_policy_relabel_v1"},
                    },
                ],
            )

            summary = module.merge_jsonl_files(
                base_path=base_path,
                relabel_path=relabel_path,
                output_path=output_path,
            )

            merged = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["base_records"], 2)
            self.assertEqual(summary["relabel_records"], 2)
            self.assertEqual(summary["merged_records"], 3)
            self.assertEqual(summary["overridden_records"], 1)
            self.assertEqual([record["trajectory_id"] for record in merged], ["ep-1", "ep-2", "ep-3"])
            self.assertEqual(merged[1]["label"], "quote_evidence")
            self.assertEqual(merged[1]["metadata"]["source"], "hard_off_policy_relabel_v1")

    def test_merge_jsonl_files_supports_stop_policy_records_independently(self):
        module = importlib.import_module("train_agent.scripts.merge_relabel_into_trainset")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            base_path = tmpdir_path / "base_stop.jsonl"
            relabel_path = tmpdir_path / "relabel_stop.jsonl"
            output_path = tmpdir_path / "merged_stop.jsonl"

            _write_jsonl(
                base_path,
                [
                    {
                        "trajectory_id": "ep-1",
                        "step_id": 0,
                        "task": "stop_policy_classification",
                        "text": "base-stop-a",
                        "label": "no",
                        "label_text": "{\"should_stop\": \"no\", \"reason\": \"continue_after_search\"}",
                    }
                ],
            )
            _write_jsonl(
                relabel_path,
                [
                    {
                        "trajectory_id": "ep-1",
                        "step_id": 0,
                        "task": "stop_policy_classification",
                        "text": "relabel-stop-a",
                        "label": "yes",
                        "label_text": "{\"should_stop\": \"yes\", \"reason\": \"sufficient_quoted_evidence\"}",
                        "metadata": {"source": "hard_off_policy_relabel_v1"},
                    }
                ],
            )

            summary = module.merge_jsonl_files(
                base_path=base_path,
                relabel_path=relabel_path,
                output_path=output_path,
            )

            merged = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["merged_records"], 1)
            self.assertEqual(summary["overridden_records"], 1)
            self.assertEqual(merged[0]["task"], "stop_policy_classification")
            self.assertEqual(merged[0]["label"], "yes")
            self.assertEqual(json.loads(merged[0]["label_text"])["reason"], "sufficient_quoted_evidence")


if __name__ == "__main__":
    unittest.main()
