import unittest

from train_agent.eval.restricted_ranking import evaluate_restricted_ranking_episodes
from train_agent.rl.restricted_retrieval import RestrictedEvidence, RestrictedRetrievalEpisode


def build_episode(episode_id, gold_doc_id):
    return RestrictedRetrievalEpisode(
        episode_id=episode_id,
        claim=f"Claim for {episode_id}",
        label_hint="SUPPORT",
        doc_pool=["doc-a", "doc-b"],
        gold_evidence=[
            RestrictedEvidence(
                doc_id=gold_doc_id,
                sentence_ids=[0],
                stance="SUPPORT",
                snippet="Gold evidence.",
            )
        ],
        document_contents={
            "doc-a": "weak background evidence",
            "doc-b": "strong support evidence",
        },
        max_steps=4,
    )


class FakeFrozenVerifier:
    def __init__(self, preferred_by_claim):
        self.preferred_by_claim = preferred_by_claim

    def score_documents(self, claim, documents):
        preferred = self.preferred_by_claim[claim]
        return {doc_id: 0.9 if doc_id == preferred else 0.2 for doc_id in documents}


class RestrictedRankingEvalTest(unittest.TestCase):
    def test_evaluate_restricted_ranking_reports_doc_metrics(self):
        episodes = [
            build_episode("ep-1", "doc-b"),
            build_episode("ep-2", "doc-a"),
        ]
        verifier = FakeFrozenVerifier(
            {
                "Claim for ep-1": "doc-b",
                "Claim for ep-2": "doc-b",
            }
        )

        metrics = evaluate_restricted_ranking_episodes(episodes, verifier)

        self.assertEqual(metrics["episodes"], 2)
        self.assertEqual(metrics["positive_episodes"], 2)
        self.assertAlmostEqual(metrics["doc_mrr"], 0.75)
        self.assertAlmostEqual(metrics["doc_recall@1"], 0.5)
        self.assertAlmostEqual(metrics["doc_recall@3"], 1.0)


if __name__ == "__main__":
    unittest.main()
