import unittest

from train_agent.data.adapters.fever import build_fever_restricted_episode
from train_agent.data.adapters.hover import build_hover_verifier_examples
from train_agent.data.adapters.scifact import (
    build_scifact_relevance_examples,
    build_scifact_stance_examples,
    build_scifact_verifier_examples,
)


class VerifierAdapterTest(unittest.TestCase):
    def test_scifact_adapter_emits_support_and_neutral_examples(self):
        row = {
            "id": 7,
            "claim": "Tree canopy reduces urban heat.",
            "label": "SUPPORT",
            "documents": [
                {
                    "doc_id": "doc-support",
                    "sentences": [
                        "Tree canopy lowers local surface temperature.",
                        "Trees also affect air quality.",
                    ],
                },
                {
                    "doc_id": "doc-distractor",
                    "sentences": [
                        "This sentence is unrelated to the claim.",
                    ],
                },
            ],
            "evidence": [
                {
                    "doc_id": "doc-support",
                    "sentence_ids": [0],
                    "label": "SUPPORT",
                }
            ],
        }

        examples = build_scifact_verifier_examples(row)
        triples = {(item.doc_id, item.sentence_id, item.label) for item in examples}
        self.assertIn(("doc-support", 0, "SUPPORT"), triples)
        self.assertIn(("doc-distractor", 0, "NEUTRAL"), triples)

    def test_scifact_relevance_adapter_marks_gold_evidence_as_relevant(self):
        row = {
            "id": 8,
            "claim": "Tree canopy reduces urban heat.",
            "label": "SUPPORT",
            "documents": [
                {
                    "doc_id": "doc-support",
                    "sentences": [
                        "Tree canopy lowers local surface temperature.",
                        "Trees also affect air quality.",
                    ],
                },
                {
                    "doc_id": "doc-distractor",
                    "sentences": [
                        "This sentence is unrelated to the claim.",
                    ],
                },
            ],
            "evidence": [
                {
                    "doc_id": "doc-support",
                    "sentence_ids": [0],
                    "label": "SUPPORT",
                }
            ],
        }

        examples = build_scifact_relevance_examples(row)
        triples = {(item.doc_id, item.sentence_id, item.label) for item in examples}
        self.assertIn(("doc-support", 0, "RELEVANT"), triples)
        self.assertIn(("doc-support", 1, "NEUTRAL"), triples)
        self.assertIn(("doc-distractor", 0, "NEUTRAL"), triples)

    def test_scifact_relevance_adapter_can_keep_only_hard_negative_sentences(self):
        row = {
            "id": 81,
            "claim": "Tree canopy reduces urban heat.",
            "label": "SUPPORT",
            "documents": [
                {
                    "doc_id": "doc-support",
                    "sentences": [
                        "Tree canopy lowers local surface temperature.",
                        "Background sentence about birds.",
                    ],
                },
                {
                    "doc_id": "doc-hard",
                    "sentences": [
                        "Urban heat is reduced by tree cover in cities.",
                    ],
                },
                {
                    "doc_id": "doc-easy",
                    "sentences": [
                        "Bananas grow in tropical regions.",
                    ],
                },
            ],
            "evidence": [
                {
                    "doc_id": "doc-support",
                    "sentence_ids": [0],
                    "label": "SUPPORT",
                }
            ],
        }

        examples = build_scifact_relevance_examples(
            row,
            max_hard_negatives_per_positive=1,
            max_random_negatives_per_positive=0,
        )
        triples = {(item.doc_id, item.sentence_id, item.label) for item in examples}
        self.assertEqual(
            triples,
            {
                ("doc-support", 0, "RELEVANT"),
                ("doc-hard", 0, "NEUTRAL"),
            },
        )

    def test_scifact_stance_adapter_keeps_only_relevant_support_or_contradict(self):
        row = {
            "id": 9,
            "claim": "The treatment is ineffective.",
            "label": "REFUTES",
            "documents": [
                {
                    "doc_id": "doc-refute",
                    "sentences": [
                        "The treatment showed no effect in the trial.",
                        "Background sentence.",
                    ],
                },
                {
                    "doc_id": "doc-distractor",
                    "sentences": [
                        "Unrelated sentence.",
                    ],
                },
            ],
            "evidence": [
                {
                    "doc_id": "doc-refute",
                    "sentence_ids": [0],
                    "label": "CONTRADICT",
                }
            ],
        }

        examples = build_scifact_stance_examples(row)
        triples = [(item.doc_id, item.sentence_id, item.label) for item in examples]
        self.assertEqual(triples, [("doc-refute", 0, "CONTRADICT")])

    def test_fever_adapter_builds_restricted_episode_with_gold_doc_in_pool(self):
        row = {
            "id": 12,
            "claim": "The Nile is in South America.",
            "label": "REFUTES",
            "documents": [
                {"doc_id": "Nile", "sentences": ["The Nile is a major north-flowing river in Africa."]},
                {"doc_id": "Amazon", "sentences": ["The Amazon River is in South America."]},
            ],
            "evidence_sets": [
                [{"doc_id": "Nile", "sentence_id": 0}],
            ],
        }

        episode = build_fever_restricted_episode(row, max_steps=5)
        self.assertEqual(episode.label_hint, "CONTRADICT")
        self.assertIn("Nile", episode.doc_pool)
        self.assertEqual(episode.gold_evidence[0].doc_id, "Nile")
        self.assertIn("Africa", episode.document_contents["Nile"])

    def test_hover_adapter_marks_multihop_support_sentences_as_positive(self):
        row = {
            "id": "hover-3",
            "claim": "Claim requiring two supporting hops.",
            "label": "SUPPORTED",
            "documents": [
                {"doc_id": "DocOne", "sentences": ["First hop support sentence."]},
                {"doc_id": "DocTwo", "sentences": ["Background.", "Second hop support sentence."]},
            ],
            "supporting_facts": [
                {"doc_id": "DocOne", "sentence_id": 0},
                {"doc_id": "DocTwo", "sentence_id": 1},
            ],
        }

        examples = build_hover_verifier_examples(row)
        positives = {
            (item.doc_id, item.sentence_id, item.label)
            for item in examples
            if item.label != "NEUTRAL"
        }
        self.assertEqual(
            positives,
            {
                ("DocOne", 0, "SUPPORT"),
                ("DocTwo", 1, "SUPPORT"),
            },
        )


if __name__ == "__main__":
    unittest.main()
