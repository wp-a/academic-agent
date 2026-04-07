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


class MixRelabelDatasetTest(unittest.TestCase):
    def test_build_mixed_dataset_excludes_uncertain_skip_by_default(self):
        module = importlib.import_module("train_agent.scripts.build_mixed_trainset")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            base_path = tmpdir_path / "base.jsonl"
            relabel_path = tmpdir_path / "relabel.jsonl"
            output_path = tmpdir_path / "mixed.jsonl"

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
                    }
                ],
            )
            _write_jsonl(
                relabel_path,
                [
                    {
                        "trajectory_id": "ep-2",
                        "step_id": 0,
                        "task": "next_action_classification",
                        "text": "relabel-good",
                        "label": "quote_evidence",
                        "label_text": "{\"action_type\": \"quote_evidence\"}",
                        "metadata": {
                            "source": "hard_off_policy_relabel_v1",
                            "relabel_decision_type": "correct_reference",
                        },
                    },
                    {
                        "trajectory_id": "ep-3",
                        "step_id": 0,
                        "task": "next_action_classification",
                        "text": "relabel-uncertain",
                        "label": "stop",
                        "label_text": "{\"action_type\": \"stop\"}",
                        "metadata": {
                            "source": "hard_off_policy_relabel_v1",
                            "relabel_decision_type": "uncertain_skip",
                        },
                    },
                ],
            )

            summary = module.build_mixed_dataset(
                base_path=base_path,
                relabel_path=relabel_path,
                output_path=output_path,
            )

            mixed = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(summary["base_records"], 1)
            self.assertEqual(summary["relabel_records"], 2)
            self.assertEqual(summary["excluded_uncertain_skip_records"], 1)
            self.assertEqual(summary["mixed_records"], 2)
            self.assertEqual([record["trajectory_id"] for record in mixed], ["ep-1", "ep-2"])
            self.assertEqual(sorted(mixed[1].keys()), ["label", "label_text", "step_id", "task", "text", "trajectory_id"])

    def test_build_dagger_recipe_writes_fixed_action_and_stop_train_mixed_paths(self):
        module = importlib.import_module("train_agent.scripts.build_mixed_trainset")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            base_dir = tmpdir_path / "base"
            relabel_dir = tmpdir_path / "relabel"
            output_dir = tmpdir_path / "mixed"
            base_dir.mkdir(parents=True, exist_ok=True)
            relabel_dir.mkdir(parents=True, exist_ok=True)

            _write_jsonl(
                base_dir / "scifact_hard_action_policy_train.jsonl",
                [
                    {
                        "trajectory_id": "ep-a-base",
                        "step_id": 0,
                        "task": "next_action_classification",
                        "text": "action-base",
                        "label": "search",
                        "label_text": "{\"action_type\": \"search\"}",
                    }
                ],
            )
            _write_jsonl(
                base_dir / "scifact_hard_stop_policy_train.jsonl",
                [
                    {
                        "trajectory_id": "ep-s-base",
                        "step_id": 0,
                        "task": "stop_policy_classification",
                        "text": "stop-base",
                        "label": "no",
                        "label_text": "{\"should_stop\": \"no\", \"reason\": \"continue_after_search\"}",
                    }
                ],
            )
            _write_jsonl(
                relabel_dir / "off_policy_action_relabel.jsonl",
                [
                    {
                        "trajectory_id": "ep-a-new",
                        "step_id": 1,
                        "task": "next_action_classification",
                        "text": "action-new",
                        "label": "quote_evidence",
                        "label_text": "{\"action_type\": \"quote_evidence\"}",
                        "metadata": {"relabel_decision_type": "correct_reference"},
                    }
                ],
            )
            _write_jsonl(
                relabel_dir / "off_policy_stop_relabel.jsonl",
                [
                    {
                        "trajectory_id": "ep-s-new",
                        "step_id": 1,
                        "task": "stop_policy_classification",
                        "text": "stop-new",
                        "label": "yes",
                        "label_text": "{\"should_stop\": \"yes\", \"reason\": \"sufficient_quoted_evidence\"}",
                        "metadata": {"relabel_decision_type": "correct_reference"},
                    }
                ],
            )

            summary = module.build_scifact_hard_dagger_recipe(
                base_dir=base_dir,
                relabel_dir=relabel_dir,
                output_dir=output_dir,
            )

            action_output_path = output_dir / "scifact_hard_action_policy_train_mixed.jsonl"
            stop_output_path = output_dir / "scifact_hard_stop_policy_train_mixed.jsonl"
            self.assertTrue(action_output_path.exists())
            self.assertTrue(stop_output_path.exists())
            self.assertEqual(summary["action"]["output_path"], str(action_output_path))
            self.assertEqual(summary["stop"]["output_path"], str(stop_output_path))

            action_rows = [json.loads(line) for line in action_output_path.read_text(encoding="utf-8").splitlines()]
            stop_rows = [json.loads(line) for line in stop_output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["trajectory_id"] for row in action_rows], ["ep-a-base", "ep-a-new"])
            self.assertEqual([row["trajectory_id"] for row in stop_rows], ["ep-s-base", "ep-s-new"])


if __name__ == "__main__":
    unittest.main()
