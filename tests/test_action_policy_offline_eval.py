import unittest

from train_agent.rl.restricted_retrieval import RestrictedEvidence, RestrictedRetrievalEpisode
from train_agent.scripts.eval_action_policy_offline_replay import evaluate_policy_on_episodes


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


class ActionPolicyOfflineEvalTest(unittest.TestCase):
    def test_evaluate_policy_on_episodes_reports_episode_metrics(self):
        episode = RestrictedRetrievalEpisode(
            episode_id="ep-1",
            claim="Claim",
            label_hint="SUPPORT",
            doc_pool=["doc-good"],
            gold_evidence=[RestrictedEvidence(doc_id="doc-good", sentence_ids=[0], stance="SUPPORT", snippet="Evidence")],
            document_contents={"doc-good": "Evidence sentence."},
            document_sentences={"doc-good": ["Evidence sentence."]},
            max_steps=4,
        )
        summary = evaluate_policy_on_episodes(
            [episode],
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
