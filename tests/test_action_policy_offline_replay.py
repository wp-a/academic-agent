import unittest

from train_agent.rl.restricted_retrieval import RestrictedEvidence, RestrictedRetrievalEpisode
from train_agent.scripts.export_scifact_frozen_verifier_replay import (
    WeakCoupledReplayPolicy,
    replay_episode_to_action_examples,
    summarize_replay_records,
)


class FakeFrozenVerifier:
    def score_documents(self, claim, documents):
        del claim
        return {
            doc_id: 0.95 if "Best support sentence" in text else 0.15
            for doc_id, text in documents.items()
        }

    def score_document_sentences(self, claim, documents, *, aggregation="max", aggregation_top_k=3):
        del claim, aggregation, aggregation_top_k
        return {
            doc_id: 0.95 if any("Best support sentence" in sentence for sentence in sentences) else 0.15
            for doc_id, sentences in documents.items()
        }


def build_episode(max_steps: int = 4) -> RestrictedRetrievalEpisode:
    return RestrictedRetrievalEpisode(
        episode_id="episode-1",
        claim="Example support claim",
        label_hint="SUPPORT",
        doc_pool=["doc-a", "doc-b"],
        gold_evidence=[
            RestrictedEvidence(
                doc_id="doc-b",
                sentence_ids=[0],
                stance="SUPPORT",
                snippet="Best support sentence.",
            )
        ],
        document_contents={
            "doc-a": "Background context only.",
            "doc-b": "Best support sentence.",
        },
        document_sentences={
            "doc-a": ["Background context only."],
            "doc-b": ["Best support sentence."],
        },
        max_steps=max_steps,
    )


class ActionPolicyOfflineReplayTest(unittest.TestCase):
    def test_replay_episode_exports_search_quote_stop_examples(self):
        records = replay_episode_to_action_examples(
            build_episode(),
            frozen_verifier=FakeFrozenVerifier(),
            policy=WeakCoupledReplayPolicy(),
        )

        self.assertEqual([record["label"] for record in records], ["search", "quote_evidence", "stop"])
        self.assertIn("Verifier Summary", records[1]["text"])
        self.assertIn("doc-b", records[1]["text"])
        self.assertTrue(records[-1]["metadata"]["success_stop"])

    def test_replay_summary_reports_success_rates(self):
        records = replay_episode_to_action_examples(
            build_episode(),
            frozen_verifier=FakeFrozenVerifier(),
            policy=WeakCoupledReplayPolicy(),
        )
        summary = summarize_replay_records([records])

        self.assertEqual(summary["episodes"], 1)
        self.assertEqual(summary["success_rate"], 1.0)
        self.assertEqual(summary["average_steps"], 3.0)
        self.assertAlmostEqual(summary["action_distribution"]["search"], 1 / 3, places=6)


if __name__ == "__main__":
    unittest.main()
