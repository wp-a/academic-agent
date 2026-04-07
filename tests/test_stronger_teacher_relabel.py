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


class StrongerTeacherRelabelTest(unittest.TestCase):
    def test_build_relabels_from_files_supports_llm_api_teacher_backend(self):
        module = importlib.import_module("train_agent.scripts.build_stronger_teacher_relabels")

        calls = []

        def fake_llm_api_teacher_decision(**kwargs):
            calls.append(kwargs)
            return {
                "action_type": "stop",
                "should_stop": "yes",
                "stop_reason": "sufficient_quoted_evidence",
                "confidence": "high",
                "rationale_short": "LLM teacher says the quoted evidence is sufficient.",
                "decision_type": "override_reference",
            }

        module._llm_api_teacher_decision = fake_llm_api_teacher_decision

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            diagnostics_path = tmpdir_path / "diagnostics.jsonl"
            action_path = tmpdir_path / "off_policy_action.jsonl"
            stop_path = tmpdir_path / "off_policy_stop.jsonl"
            output_dir = tmpdir_path / "relabel_v1"

            _write_jsonl(
                diagnostics_path,
                [
                    {
                        "episode_id": "ep-llm",
                        "claim": "Claim llm",
                        "label_hint": "SUPPORT",
                        "reference_policy_type": "conservative",
                        "post_quote_search_budget": 1,
                        "num_steps": 4,
                        "num_mismatches": 1,
                        "mismatch_step_indices": [2],
                        "steps": [
                            {
                                "step_index": 2,
                                "reference_action": "search",
                                "predicted_action": "quote_evidence",
                                "action_match": False,
                                "stop_policy_should_stop": False,
                                "suppressed_stop": False,
                                "revealed_evidence": [{"doc_id": "doc-1"}],
                                "quoted_evidence": [{"doc_id": "doc-1"}],
                                "done": False,
                                "info": {},
                            }
                        ],
                    }
                ],
            )
            _write_jsonl(
                action_path,
                [
                    {
                        "trajectory_id": "ep-llm",
                        "step_id": 2,
                        "task": "next_action_classification",
                        "text": "state-llm",
                        "label": "search",
                        "label_text": "{\"action_type\": \"search\"}",
                        "metadata": {
                            "episode_id": "ep-llm",
                            "student_action": "quote_evidence",
                            "reference_action": "search",
                            "is_first_off_policy_step": True,
                            "reference_policy_type": "conservative",
                            "post_quote_search_budget": 1,
                            "used_stop_policy": True,
                            "stop_policy_should_stop": False,
                            "suppressed_stop": False,
                        },
                    }
                ],
            )
            _write_jsonl(
                stop_path,
                [
                    {
                        "trajectory_id": "ep-llm",
                        "step_id": 2,
                        "task": "stop_policy_classification",
                        "text": "state-llm",
                        "label": "no",
                        "label_text": "{\"should_stop\": \"no\", \"reason\": \"continue_after_search\"}",
                        "metadata": {"episode_id": "ep-llm"},
                    }
                ],
            )

            summary = module.build_relabels_from_files(
                diagnostics_path=diagnostics_path,
                off_policy_action_path=action_path,
                off_policy_stop_path=stop_path,
                output_dir=output_dir,
                dataset="scifact",
                split="validation",
                teacher_backend="llm_api",
                teacher_type="llm_stronger_teacher_v1",
                teacher_version="prompt-v1",
                teacher_model_name="test-llm-model",
            )

            self.assertEqual(summary["teacher_backend"], "llm_api")
            self.assertEqual(summary["teacher_model_name"], "test-llm-model")
            self.assertEqual(summary["decision_type_distribution"]["override_reference"], 1)
            self.assertEqual(summary["decision_type_episode_ids"]["override_reference"], ["ep-llm"])
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["teacher_model_name"], "test-llm-model")

            action_out = [json.loads(line) for line in (output_dir / "off_policy_action_relabel.jsonl").read_text(encoding="utf-8").splitlines()]
            stop_out = [json.loads(line) for line in (output_dir / "off_policy_stop_relabel.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(action_out[0]["label"], "stop")
            self.assertEqual(action_out[0]["metadata"]["teacher_backend"], "llm_api")
            self.assertEqual(action_out[0]["metadata"]["relabel_decision_type"], "override_reference")
            self.assertEqual(stop_out[0]["label"], "yes")
            self.assertEqual(json.loads(stop_out[0]["label_text"])["reason"], "sufficient_quoted_evidence")

    def test_build_relabels_from_files_exports_training_ready_action_and_stop_jsonl(self):
        module = importlib.import_module("train_agent.scripts.build_stronger_teacher_relabels")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            diagnostics_path = tmpdir_path / "diagnostics.jsonl"
            action_path = tmpdir_path / "off_policy_action.jsonl"
            stop_path = tmpdir_path / "off_policy_stop.jsonl"
            output_dir = tmpdir_path / "relabel_v1"

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
                    "task": "next_action_classification",
                    "text": "state-a",
                    "label": "quote_evidence",
                    "label_text": "{\"action_type\": \"quote_evidence\"}",
                    "metadata": {
                        "episode_id": "ep-stop-early",
                        "student_action": "stop",
                        "reference_action": "quote_evidence",
                        "is_first_off_policy_step": True,
                        "reference_policy_type": "conservative",
                        "post_quote_search_budget": 1,
                        "used_stop_policy": True,
                        "stop_policy_should_stop": True,
                        "suppressed_stop": False,
                    },
                },
                {
                    "trajectory_id": "ep-oversearch",
                    "step_id": 3,
                    "task": "next_action_classification",
                    "text": "state-b",
                    "label": "stop",
                    "label_text": "{\"action_type\": \"stop\"}",
                    "metadata": {
                        "episode_id": "ep-oversearch",
                        "student_action": "search",
                        "reference_action": "stop",
                        "is_first_off_policy_step": True,
                        "reference_policy_type": "conservative",
                        "post_quote_search_budget": 1,
                        "used_stop_policy": True,
                        "stop_policy_should_stop": False,
                        "suppressed_stop": False,
                    },
                },
                {
                    "trajectory_id": "ep-oversearch",
                    "step_id": 4,
                    "task": "next_action_classification",
                    "text": "state-c",
                    "label": "stop",
                    "label_text": "{\"action_type\": \"stop\"}",
                    "metadata": {
                        "episode_id": "ep-oversearch",
                        "student_action": "stop",
                        "reference_action": "stop",
                        "is_first_off_policy_step": False,
                        "reference_policy_type": "conservative",
                        "post_quote_search_budget": 1,
                        "used_stop_policy": True,
                        "stop_policy_should_stop": True,
                        "suppressed_stop": False,
                    },
                },
            ]
            _write_jsonl(action_path, action_records)

            stop_records = [
                {
                    "trajectory_id": "ep-stop-early",
                    "step_id": 4,
                    "task": "stop_policy_classification",
                    "text": "state-a",
                    "label": "no",
                    "label_text": "{\"should_stop\": \"no\", \"reason\": \"continue_after_quote_evidence\"}",
                    "metadata": {"episode_id": "ep-stop-early"},
                },
                {
                    "trajectory_id": "ep-oversearch",
                    "step_id": 3,
                    "task": "stop_policy_classification",
                    "text": "state-b",
                    "label": "yes",
                    "label_text": "{\"should_stop\": \"yes\", \"reason\": \"chosen_stop\"}",
                    "metadata": {"episode_id": "ep-oversearch"},
                },
                {
                    "trajectory_id": "ep-oversearch",
                    "step_id": 4,
                    "task": "stop_policy_classification",
                    "text": "state-c",
                    "label": "yes",
                    "label_text": "{\"should_stop\": \"yes\", \"reason\": \"chosen_stop\"}",
                    "metadata": {"episode_id": "ep-oversearch"},
                },
            ]
            _write_jsonl(stop_path, stop_records)

            summary = module.build_relabels_from_files(
                diagnostics_path=diagnostics_path,
                off_policy_action_path=action_path,
                off_policy_stop_path=stop_path,
                output_dir=output_dir,
                dataset="scifact",
                split="validation",
                teacher_type="rule_based_stronger_teacher_v1",
                teacher_version="v1",
            )

            self.assertEqual(summary["episodes_relabeled"], 2)
            self.assertEqual(summary["teacher_backend"], "rule_based")
            self.assertEqual(summary["action_records_relabeled"], 3)
            self.assertEqual(summary["stop_records_relabeled"], 3)
            self.assertEqual(summary["bucket_distribution"]["premature_stop_after_evidence"], 1)
            self.assertEqual(summary["bucket_distribution"]["oversearch_after_quote"], 1)
            self.assertEqual(summary["decision_type_distribution"]["correct_reference"], 3)
            self.assertEqual(
                summary["decision_type_episode_ids"]["correct_reference"],
                ["ep-oversearch", "ep-stop-early"],
            )

            action_out = [json.loads(line) for line in (output_dir / "off_policy_action_relabel.jsonl").read_text(encoding="utf-8").splitlines()]
            stop_out = [json.loads(line) for line in (output_dir / "off_policy_stop_relabel.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(action_out), 3)
            self.assertEqual(len(stop_out), 3)

            self.assertEqual(action_out[0]["label"], "quote_evidence")
            self.assertEqual(action_out[0]["metadata"]["failure_bucket"], "premature_stop_after_evidence")
            self.assertEqual(action_out[0]["metadata"]["teacher_label_stop"], "no")
            self.assertEqual(action_out[0]["metadata"]["teacher_stop_reason"], "need_quote_before_stop")
            self.assertEqual(action_out[0]["metadata"]["teacher_type"], "rule_based_stronger_teacher_v1")
            self.assertEqual(action_out[0]["metadata"]["teacher_version"], "v1")

            self.assertEqual(action_out[1]["label"], "stop")
            self.assertEqual(action_out[1]["metadata"]["failure_bucket"], "oversearch_after_quote")
            self.assertEqual(action_out[1]["metadata"]["teacher_label_stop"], "yes")
            self.assertEqual(action_out[1]["metadata"]["teacher_stop_reason"], "sufficient_quoted_evidence")

            self.assertEqual(stop_out[0]["label"], "no")
            self.assertEqual(json.loads(stop_out[0]["label_text"])["reason"], "need_quote_before_stop")
            self.assertEqual(stop_out[1]["label"], "yes")
            self.assertEqual(json.loads(stop_out[1]["label_text"])["reason"], "sufficient_quoted_evidence")

    def test_build_relabels_from_files_routes_low_confidence_records_to_uncertain_skip_outputs(self):
        module = importlib.import_module("train_agent.scripts.build_stronger_teacher_relabels")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            diagnostics_path = tmpdir_path / "diagnostics.jsonl"
            action_path = tmpdir_path / "off_policy_action.jsonl"
            stop_path = tmpdir_path / "off_policy_stop.jsonl"
            output_dir = tmpdir_path / "relabel_v1"

            _write_jsonl(
                diagnostics_path,
                [
                    {
                        "episode_id": "ep-uncertain",
                        "claim": "Claim 3",
                        "label_hint": "SUPPORT",
                        "reference_policy_type": "conservative",
                        "post_quote_search_budget": 1,
                        "num_steps": 4,
                        "num_mismatches": 1,
                        "mismatch_step_indices": [2],
                        "steps": [
                            {
                                "step_index": 2,
                                "reference_action": "search",
                                "predicted_action": "quote_evidence",
                                "action_match": False,
                                "stop_policy_should_stop": False,
                                "suppressed_stop": False,
                                "revealed_evidence": [],
                                "quoted_evidence": [],
                                "done": False,
                                "info": {},
                            }
                        ],
                    }
                ],
            )
            _write_jsonl(
                action_path,
                [
                    {
                        "trajectory_id": "ep-uncertain",
                        "step_id": 2,
                        "task": "next_action_classification",
                        "text": "state-u",
                        "label": "search",
                        "label_text": "{\"action_type\": \"search\"}",
                        "metadata": {
                            "episode_id": "ep-uncertain",
                            "student_action": "quote_evidence",
                            "reference_action": "search",
                            "is_first_off_policy_step": True,
                            "reference_policy_type": "conservative",
                            "post_quote_search_budget": 1,
                            "used_stop_policy": True,
                            "stop_policy_should_stop": False,
                            "suppressed_stop": False,
                        },
                    }
                ],
            )
            _write_jsonl(
                stop_path,
                [
                    {
                        "trajectory_id": "ep-uncertain",
                        "step_id": 2,
                        "task": "stop_policy_classification",
                        "text": "state-u",
                        "label": "no",
                        "label_text": "{\"should_stop\": \"no\", \"reason\": \"continue_after_search\"}",
                        "metadata": {"episode_id": "ep-uncertain"},
                    }
                ],
            )

            summary = module.build_relabels_from_files(
                diagnostics_path=diagnostics_path,
                off_policy_action_path=action_path,
                off_policy_stop_path=stop_path,
                output_dir=output_dir,
                dataset="scifact",
                split="validation",
                teacher_type="rule_based_stronger_teacher_v1",
                teacher_version="v1",
                minimum_teacher_confidence="high",
            )

            self.assertEqual(summary["episodes_relabeled"], 1)
            self.assertEqual(summary["teacher_backend"], "rule_based")
            self.assertEqual(summary["action_records_relabeled"], 0)
            self.assertEqual(summary["stop_records_relabeled"], 0)
            self.assertEqual(summary["uncertain_skip_action_records"], 1)
            self.assertEqual(summary["uncertain_skip_stop_records"], 1)
            self.assertEqual(summary["decision_type_distribution"]["uncertain_skip"], 1)
            self.assertEqual(summary["decision_type_episode_ids"]["uncertain_skip"], ["ep-uncertain"])

            action_out_path = output_dir / "off_policy_action_relabel.jsonl"
            stop_out_path = output_dir / "off_policy_stop_relabel.jsonl"
            uncertain_action_path = output_dir / "off_policy_action_uncertain_skip.jsonl"
            uncertain_stop_path = output_dir / "off_policy_stop_uncertain_skip.jsonl"

            self.assertEqual(action_out_path.read_text(encoding="utf-8").strip(), "")
            self.assertEqual(stop_out_path.read_text(encoding="utf-8").strip(), "")

            uncertain_action = [json.loads(line) for line in uncertain_action_path.read_text(encoding="utf-8").splitlines()]
            uncertain_stop = [json.loads(line) for line in uncertain_stop_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(uncertain_action[0]["metadata"]["relabel_decision_type"], "uncertain_skip")
            self.assertEqual(uncertain_action[0]["metadata"]["teacher_confidence"], "medium")
            self.assertEqual(uncertain_stop[0]["metadata"]["relabel_decision_type"], "uncertain_skip")


if __name__ == "__main__":
    unittest.main()
