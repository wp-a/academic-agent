import json
import tempfile
import unittest
from pathlib import Path

from train_agent.scripts.export_scifact_decomposed_verifier_data import export_scifact_decomposed_split


class SciFactDecomposedVerifierExportTest(unittest.TestCase):
    def test_export_scifact_split_writes_relevance_and_stance_jsonl(self):
        rows = [
            {
                "id": 1,
                "claim": "Tree canopy reduces urban heat.",
                "label": "SUPPORT",
                "evidence": {
                    101: {
                        "label": "SUPPORT",
                        "sentence_ids": [0],
                    }
                },
                "cited_doc_ids": [101, 202],
            },
            {
                "id": 2,
                "claim": "Aspirin cures all cancers.",
                "label": "NOT_ENOUGH_INFO",
                "evidence": {},
                "cited_doc_ids": [303],
            },
        ]
        corpus_by_doc_id = {
            "101": {"sentences": ["Tree canopy lowers local temperature.", "Background sentence."]},
            "202": {"sentences": ["Unrelated distractor sentence."]},
            "303": {"sentences": ["No direct evidence is available."]},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            relevance_path = root / "scifact_relevance_train.jsonl"
            stance_path = root / "scifact_stance_train.jsonl"
            summary = export_scifact_decomposed_split(
                rows=rows,
                corpus_by_doc_id=corpus_by_doc_id,
                relevance_output_path=relevance_path,
                stance_output_path=stance_path,
            )

            self.assertEqual(summary["relevance"]["num_examples"], 4)
            self.assertEqual(summary["relevance"]["label_counts"]["RELEVANT"], 1)
            self.assertEqual(summary["relevance"]["label_counts"]["NEUTRAL"], 3)
            self.assertEqual(summary["stance"]["num_examples"], 1)
            self.assertEqual(summary["stance"]["label_counts"]["SUPPORT"], 1)

            relevance_rows = [json.loads(line) for line in relevance_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            stance_rows = [json.loads(line) for line in stance_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual({row["label"] for row in relevance_rows}, {"RELEVANT", "NEUTRAL"})
            self.assertEqual([row["label"] for row in stance_rows], ["SUPPORT"])

    def test_export_scifact_split_can_limit_relevance_to_hard_negatives(self):
        rows = [
            {
                "id": 1,
                "claim": "Tree canopy reduces urban heat.",
                "label": "SUPPORT",
                "evidence": {
                    101: {
                        "label": "SUPPORT",
                        "sentence_ids": [0],
                    }
                },
                "cited_doc_ids": [101, 202, 303],
            }
        ]
        corpus_by_doc_id = {
            "101": {"sentences": ["Tree canopy lowers local temperature.", "Bird migration sentence."]},
            "202": {"sentences": ["Urban heat is reduced by tree cover in cities."]},
            "303": {"sentences": ["Bananas grow in tropical regions."]},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            relevance_path = root / "scifact_relevance_train.jsonl"
            stance_path = root / "scifact_stance_train.jsonl"
            summary = export_scifact_decomposed_split(
                rows=rows,
                corpus_by_doc_id=corpus_by_doc_id,
                relevance_output_path=relevance_path,
                stance_output_path=stance_path,
                relevance_hard_negatives_per_positive=1,
                relevance_random_negatives_per_positive=0,
            )

            self.assertEqual(summary["relevance"]["num_examples"], 2)
            relevance_rows = [json.loads(line) for line in relevance_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(
                {(row["doc_id"], row["sentence_id"], row["label"]) for row in relevance_rows},
                {
                    ("101", 0, "RELEVANT"),
                    ("202", 0, "NEUTRAL"),
                },
            )


if __name__ == "__main__":
    unittest.main()
