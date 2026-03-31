import unittest

from train_agent.rl.restricted_retrieval import RestrictedEvidence, RestrictedRetrievalEpisode
from train_agent.scripts.eval_action_policy_offline_replay import (
    evaluate_policy_on_episodes,
    maybe_augment_episode_for_hard_eval,
)


class DummyVerifier:
    def score_documents(self, claim, documents):
        return {doc_id: 1.0 if doc_id == "doc-good" else 0.0 for doc_id in documents}

    def score_document_sentences(self, claim, documents, *, aggregation="max", aggregation_top_k=3):
        return {doc_id: 1.0 if doc_id == "doc-good" else 0.0 for doc_id in documents}


class DummyPolicy:
    def predict_action(self, text: str) -> str:
        if "History:\nNo previous steps." in text:
            return "search"
        if "Action: search" in text and "Action: quote_evidence" not in text:
            return "quote_evidence"
        return "stop"


class DummyJointActionPolicy:
    label_names = ["quote_evidence", "search", "stop"]

    def predict_logits(self, texts):
        logits = []
        for text in texts:
            if "History:\nNo previous steps." in text:
                logits.append([0.1, 3.0, 0.2])
            elif "Action: search" in text and "Action: quote_evidence" not in text:
                logits.append([4.0, 0.1, 8.0])
            else:
                logits.append([0.1, 0.2, 6.0])
        return logits


class DummyStopPolicy:
    def predict_should_stop(self, text: str) -> bool:
        return "Action: quote_evidence" in text


def build_single_doc_episode() -> RestrictedRetrievalEpisode:
    return RestrictedRetrievalEpisode(
        episode_id="ep-1",
        claim="Claim",
        label_hint="SUPPORT",
        doc_pool=["doc-good"],
        gold_evidence=[RestrictedEvidence(doc_id="doc-good", sentence_ids=[0], stance="SUPPORT", snippet="Evidence")],
        document_contents={"doc-good": "Evidence sentence."},
        document_sentences={"doc-good": ["Evidence sentence."]},
        max_steps=4,
    )


def build_distractor_episode() -> RestrictedRetrievalEpisode:
    return RestrictedRetrievalEpisode(
        episode_id="ep-hard",
        claim="Claim",
        label_hint="SUPPORT",
        doc_pool=["doc-good", "doc-hard"],
        gold_evidence=[RestrictedEvidence(doc_id="doc-good", sentence_ids=[0], stance="SUPPORT", snippet="Evidence")],
        document_contents={
            "doc-good": "Evidence sentence.",
            "doc-hard": "Distractor sentence.",
        },
        document_sentences={
            "doc-good": ["Evidence sentence."],
            "doc-hard": ["Distractor sentence."],
        },
        max_steps=5,
    )


class ActionPolicyOfflineEvalTest(unittest.TestCase):
    def test_maybe_augment_episode_for_hard_eval_appends_lexical_distractors(self):
        episode = RestrictedRetrievalEpisode(
            episode_id="ep-1",
            claim="tree canopy lowers summer temperature in cities",
            label_hint="SUPPORT",
            doc_pool=["doc-good"],
            gold_evidence=[
                RestrictedEvidence(
                    doc_id="doc-good",
                    sentence_ids=[0],
                    stance="SUPPORT",
                    snippet="Tree canopy lowers urban temperature.",
                )
            ],
            document_contents={"doc-good": "Tree canopy lowers urban temperature."},
            document_sentences={"doc-good": ["Tree canopy lowers urban temperature."]},
            max_steps=5,
        )

        augmented = maybe_augment_episode_for_hard_eval(
            episode,
            corpus_text_by_doc={
                "doc-good": "Tree canopy lowers urban temperature.",
                "doc-hard": "Urban tree canopy overlap with summer temperature patterns.",
                "doc-easy": "Protein folding in yeast cells.",
            },
            corpus_sentences_by_doc={
                "doc-good": ["Tree canopy lowers urban temperature."],
                "doc-hard": ["Urban tree canopy overlap with summer temperature patterns."],
                "doc-easy": ["Protein folding in yeast cells."],
            },
            num_distractor_docs=1,
        )

        self.assertEqual(augmented.doc_pool, ["doc-good", "doc-hard"])
        self.assertIn("doc-hard", augmented.document_contents)
        self.assertEqual(augmented.gold_evidence[0].doc_id, "doc-good")

    def test_evaluate_policy_on_episodes_reports_episode_metrics(self):
        summary = evaluate_policy_on_episodes(
            [build_single_doc_episode()],
            verifier=DummyVerifier(),
            action_policy=DummyPolicy(),
            doc_aggregation="full_document",
            aggregation_top_k=3,
        )
        self.assertAlmostEqual(summary["action_agreement"], 1.0, places=6)
        self.assertAlmostEqual(summary["average_steps"], 3.0, places=6)
        self.assertAlmostEqual(summary["stop_precision"], 1.0, places=6)
        self.assertAlmostEqual(summary["stop_recall"], 1.0, places=6)
        self.assertAlmostEqual(summary["quote_evidence_hit_rate"], 1.0, places=6)

    def test_evaluate_policy_on_episodes_can_gate_stop_with_stop_policy(self):
        summary = evaluate_policy_on_episodes(
            [build_single_doc_episode()],
            verifier=DummyVerifier(),
            action_policy=DummyJointActionPolicy(),
            stop_policy=DummyStopPolicy(),
            doc_aggregation="full_document",
            aggregation_top_k=3,
        )
        self.assertTrue(summary["used_stop_policy"])
        self.assertEqual(summary["stop_policy_yes_count"], 1)
        self.assertEqual(summary["stop_policy_no_count"], 2)
        self.assertEqual(summary["suppressed_stop_count"], 1)
        self.assertAlmostEqual(summary["action_agreement"], 1.0, places=6)
        self.assertAlmostEqual(summary["stop_precision"], 1.0, places=6)
        self.assertAlmostEqual(summary["stop_recall"], 1.0, places=6)

    def test_evaluate_policy_on_episodes_supports_conservative_reference_policy(self):
        summary = evaluate_policy_on_episodes(
            [build_distractor_episode()],
            verifier=DummyVerifier(),
            action_policy=DummyPolicy(),
            reference_policy_type="conservative",
            post_quote_search_budget=1,
            doc_aggregation="full_document",
            aggregation_top_k=3,
        )
        self.assertEqual(summary["reference_policy_type"], "conservative")
        self.assertEqual(summary["post_quote_search_budget"], 1)
        self.assertAlmostEqual(summary["action_agreement"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(summary["stop_recall"], 0.0, places=6)

    def test_conservative_reference_differs_from_weak_on_distractor_episode(self):
        weak_summary = evaluate_policy_on_episodes(
            [build_distractor_episode()],
            verifier=DummyVerifier(),
            action_policy=DummyPolicy(),
            reference_policy_type="weak",
            post_quote_search_budget=1,
            doc_aggregation="full_document",
            aggregation_top_k=3,
        )
        conservative_summary = evaluate_policy_on_episodes(
            [build_distractor_episode()],
            verifier=DummyVerifier(),
            action_policy=DummyPolicy(),
            reference_policy_type="conservative",
            post_quote_search_budget=1,
            doc_aggregation="full_document",
            aggregation_top_k=3,
        )
        self.assertAlmostEqual(weak_summary["action_agreement"], 1.0, places=6)
        self.assertLess(conservative_summary["action_agreement"], weak_summary["action_agreement"])
        self.assertNotEqual(weak_summary["stop_recall"], conservative_summary["stop_recall"])


if __name__ == "__main__":
    unittest.main()
