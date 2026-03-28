import argparse
import json
import tempfile
import unittest
from pathlib import Path

from train_agent.trainers import train_action_policy


class ActionPolicyTrainerTest(unittest.TestCase):
    def test_smoke_trainer_writes_eval_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_file = root / "train.jsonl"
            eval_file = root / "eval.jsonl"
            output_dir = root / "outputs"
            train_rows = [
                {
                    "trajectory_id": "traj-1",
                    "step_id": 0,
                    "task": "next_action_classification",
                    "text": "State says search",
                    "label": "search",
                    "label_text": "{\"action_type\": \"search\"}",
                },
                {
                    "trajectory_id": "traj-1",
                    "step_id": 1,
                    "task": "next_action_classification",
                    "text": "State says quote",
                    "label": "quote_evidence",
                    "label_text": "{\"action_type\": \"quote_evidence\"}",
                },
            ]
            eval_rows = [
                {
                    "trajectory_id": "traj-2",
                    "step_id": 0,
                    "task": "next_action_classification",
                    "text": "State says stop",
                    "label": "stop",
                    "label_text": "{\"action_type\": \"stop\"}",
                }
            ]
            train_file.write_text("\n".join(json.dumps(row) for row in train_rows) + "\n", encoding="utf-8")
            eval_file.write_text("\n".join(json.dumps(row) for row in eval_rows) + "\n", encoding="utf-8")

            args = argparse.Namespace(
                train_file=train_file,
                eval_file=eval_file,
                output_dir=output_dir,
                model_name_or_path="",
                max_length=64,
                learning_rate=1e-3,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=1,
                logging_steps=1,
                eval_steps=1,
                save_steps=1,
                max_steps=1,
                smoke_test=False,
                attn_implementation="sdpa",
                use_lora=False,
                lora_r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                lora_target_modules="q_proj,k_proj,v_proj,o_proj",
                lora_modules_to_save="score",
                gradient_checkpointing=False,
                trust_remote_code=False,
                seed=7,
            )
            train_action_policy.run_training(args)
            metrics = json.loads((output_dir / "eval_metrics.json").read_text(encoding="utf-8"))
            self.assertIn("accuracy", metrics)
            self.assertIn("macro_f1", metrics)
            self.assertIn("confusion_matrix", metrics)
