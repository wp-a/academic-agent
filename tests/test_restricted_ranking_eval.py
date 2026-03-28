import math
import unittest

from train_agent.eval.restricted_ranking import evaluate_restricted_ranking_episodes, rank_episode_documents
from train_agent.rl.restricted_retrieval import RestrictedEvidence, RestrictedRetrievalEpisode


def build_episode(episode_id, gold_doc_id):
    document_sentences = {
        "doc-a": ["weak background evidence"],
        "doc-b": ["strong support evidence"],
    }
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
        document_contents={doc_id: " ".join(sentences) for doc_id, sentences in document_sentences.items()},
        document_sentences=document_sentences,
        max_steps=4,
    )


def build_multi_sentence_episode():
    document_sentences = {
        "doc-a": ["a strong sentence", "a weak sentence"],
        "doc-b": ["b medium one", "b medium two", "b medium three"],
    }
    return RestrictedRetrievalEpisode(
        episode_id="ep-agg",
        claim="Claim for ep-agg",
        label_hint="SUPPORT",
        doc_pool=["doc-a", "doc-b"],
        gold_evidence=[
            RestrictedEvidence(
                doc_id="doc-a",
                sentence_ids=[0],
                stance="SUPPORT",
                snippet="Gold evidence.",
            )
        ],
        document_contents={doc_id: " ".join(sentences) for doc_id, sentences in document_sentences.items()},
        document_sentences=document_sentences,
        max_steps=4,
    )


class FakeFrozenVerifier:
    def __init__(self, preferred_by_claim=None, sentence_scores_by_claim=None):
        self.preferred_by_claim = preferred_by_claim or {}
        self.sentence_scores_by_claim = sentence_scores_by_claim or {}

    def score_documents(self, claim, documents):
        preferred = self.preferred_by_claim[claim]
        return {doc_id: 0.9 if doc_id == preferred else 0.2 for doc_id in documents}

    def score_document_sentences(self, claim, documents, *, aggregation="max", aggregation_top_k=3):
        by_doc = self.sentence_scores_by_claim[claim]
        results = {}
        for doc_id in documents:
            ranked = sorted(by_doc[doc_id], reverse=True)
            if aggregation == "max":
                score = ranked[0]
            elif aggregation == "top_2_mean":
                top_scores = ranked[:2]
                score = sum(top_scores) / len(top_scores)
            elif aggregation == "top_k_weighted_mean":
                top_scores = ranked[: max(1, aggregation_top_k)]
                weights = list(range(len(top_scores), 0, -1))
                score = sum(value * weight for value, weight in zip(top_scores, weights)) / sum(weights)
            elif aggregation == "logsumexp":
                score = math.log(sum(math.exp(value) for value in ranked))
            else:
                raise ValueError(aggregation)
            results[doc_id] = float(score)
        return results


class RestrictedRankingEvalTest(unittest.TestCase):
    def test_evaluate_restricted_ranking_reports_doc_metrics_and_hits(self):
        episodes = [
            build_episode("ep-1", "doc-b"),
            build_episode("ep-2", "doc-a"),
        ]
        verifier = FakeFrozenVerifier(
            preferred_by_claim={
                "Claim for ep-1": "doc-b",
                "Claim for ep-2": "doc-b",
            }
        )

        metrics = evaluate_restricted_ranking_episodes(episodes, verifier)

        self.assertEqual(metrics["episodes"], 2)
        self.assertEqual(metrics["positive_episodes"], 2)
        self.assertEqual(metrics["doc_hits@1"], 1)
        self.assertEqual(metrics["doc_hits@3"], 2)
        self.assertEqual(metrics["doc_misses@1"], 1)
        self.assertEqual(metrics["doc_misses@3"], 0)
        self.assertAlmostEqual(metrics["doc_mrr"], 0.75)
        self.assertAlmostEqual(metrics["doc_recall@1"], 0.5)
        self.assertAlmostEqual(metrics["doc_recall@3"], 1.0)

    def test_evaluate_restricted_ranking_can_bootstrap(self):
        episodes = [
            build_episode("ep-1", "doc-b"),
            build_episode("ep-2", "doc-a"),
            build_episode("ep-3", "doc-b"),
        ]
        verifier = FakeFrozenVerifier(
            preferred_by_claim={
                "Claim for ep-1": "doc-b",
                "Claim for ep-2": "doc-b",
                "Claim for ep-3": "doc-a",
            }
        )

        metrics = evaluate_restricted_ranking_episodes(
            episodes,
            verifier,
            bootstrap_samples=200,
            bootstrap_seed=13,
        )

        self.assertIn("bootstrap", metrics)
        self.assertEqual(metrics["bootstrap"]["samples"], 200)
        self.assertEqual(metrics["bootstrap"]["seed"], 13)
        self.assertEqual(len(metrics["bootstrap"]["doc_mrr_ci95"]), 2)
        self.assertEqual(len(metrics["bootstrap"]["doc_recall@1_ci95"]), 2)
        self.assertEqual(len(metrics["bootstrap"]["doc_recall@3_ci95"]), 2)
        self.assertGreaterEqual(metrics["bootstrap"]["doc_recall@3_ci95"][0], 0.0)
        self.assertLessEqual(metrics["bootstrap"]["doc_recall@3_ci95"][1], 1.0)

    def test_rank_episode_documents_supports_sentence_aggregation_modes(self):
        episode = build_multi_sentence_episode()
        verifier = FakeFrozenVerifier(
            sentence_scores_by_claim={
                "Claim for ep-agg": {
                    "doc-a": [0.9, 0.1],
                    "doc-b": [0.6, 0.6, 0.6],
                }
            }
        )

        max_ranking = rank_episode_documents(episode, verifier, doc_aggregation="max")
        top2_ranking = rank_episode_documents(episode, verifier, doc_aggregation="top_2_mean")
        weighted_ranking = rank_episode_documents(
            episode,
            verifier,
            doc_aggregation="top_k_weighted_mean",
            aggregation_top_k=3,
        )
        logsumexp_ranking = rank_episode_documents(episode, verifier, doc_aggregation="logsumexp")

        self.assertEqual(max_ranking[0]["doc_id"], "doc-a")
        self.assertEqual(top2_ranking[0]["doc_id"], "doc-b")
        self.assertEqual(weighted_ranking[0]["doc_id"], "doc-a")
        self.assertEqual(logsumexp_ranking[0]["doc_id"], "doc-b")


if __name__ == "__main__":
    unittest.main()
