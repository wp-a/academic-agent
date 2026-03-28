import json
import tempfile
import unittest
from pathlib import Path

from train_agent.scripts.export_scifact_verifier_data import export_scifact_verifier_split


class SciFactVerifierExportTest(unittest.TestCase):
    def test_export_scifact_split_writes_jsonl_and_label_counts(self):
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
            output_path = Path(tmpdir) / "scifact_train.jsonl"
            summary = export_scifact_verifier_split(
                rows=rows,
                corpus_by_doc_id=corpus_by_doc_id,
                output_path=output_path,
            )

            self.assertEqual(summary["num_examples"], 4)
            self.assertEqual(summary["label_counts"]["SUPPORT"], 1)
            self.assertEqual(summary["label_counts"]["NEUTRAL"], 3)

            exported = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(exported), 4)
            self.assertEqual(exported[0]["dataset"], "scifact")
            self.assertIn(exported[0]["label"], {"SUPPORT", "NEUTRAL"})
            self.assertIn("evidence_text", exported[0])


if __name__ == "__main__":
    unittest.main()
