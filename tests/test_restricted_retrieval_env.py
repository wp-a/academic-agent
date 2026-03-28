import unittest

from train_agent.rl.restricted_retrieval import RestrictedEvidence, RestrictedRetrievalEnv, RestrictedRetrievalEpisode


def build_support_episode(max_steps: int = 4) -> RestrictedRetrievalEpisode:
    return RestrictedRetrievalEpisode(
        episode_id="episode-support",
        claim="Example support claim",
        label_hint="SUPPORT",
        doc_pool=["doc-a", "doc-b"],
        gold_evidence=[
            RestrictedEvidence(
                doc_id="doc-a",
                sentence_ids=[2],
                stance="SUPPORT",
                snippet="Example support sentence.",
            )
        ],
        max_steps=max_steps,
    )


def build_unknown_episode(max_steps: int = 4) -> RestrictedRetrievalEpisode:
    return RestrictedRetrievalEpisode(
        episode_id="episode-unknown",
        claim="Example unknown claim",
        label_hint="UNKNOWN",
        doc_pool=["doc-z"],
        gold_evidence=[],
        max_steps=max_steps,
    )


def build_ranked_episode(max_steps: int = 4) -> RestrictedRetrievalEpisode:
    return RestrictedRetrievalEpisode(
        episode_id="episode-ranked",
        claim="Example ranked claim",
        label_hint="SUPPORT",
        doc_pool=["doc-a", "doc-b"],
        gold_evidence=[
            RestrictedEvidence(
                doc_id="doc-b",
                sentence_ids=[1],
                stance="SUPPORT",
                snippet="Best support sentence.",
            )
        ],
        document_contents={
            "doc-a": "Background sentence with little direct evidence.",
            "doc-b": "Best support sentence with direct evidence for the claim.",
        },
        max_steps=max_steps,
    )


class FakeFrozenVerifier:
    def score_documents(self, claim, documents):
        del claim
        return {
            doc_id: 0.95 if "Best support sentence" in text else 0.15
            for doc_id, text in documents.items()
        }


class RestrictedRetrievalEnvTest(unittest.TestCase):
    def test_state_text_hides_label_hint(self):
        env = RestrictedRetrievalEnv(build_support_episode())
        state = env.reset()
        self.assertNotIn("SUPPORT", state.to_text())
        self.assertNotIn("label hint", state.to_text().lower())

    def test_search_reveals_new_evidence_in_deterministic_order(self):
        env = RestrictedRetrievalEnv(build_support_episode())
        state = env.reset()
        self.assertEqual(state.revealed_docs, [])
        result = env.step("search")
        self.assertAlmostEqual(result.reward, 0.05)
        self.assertFalse(result.done)
        self.assertEqual(result.state.revealed_docs, ["doc-a"])
        self.assertEqual(len(result.state.revealed_evidence), 1)
        self.assertEqual(result.state.revealed_evidence[0].doc_id, "doc-a")

    def test_stop_before_quote_is_early_stop_and_penalized(self):
        env = RestrictedRetrievalEnv(build_support_episode())
        env.reset()
        result = env.step("stop")
        self.assertTrue(result.done)
        self.assertAlmostEqual(result.reward, -1.05)
        self.assertTrue(result.info["early_stop"])
        self.assertFalse(result.info["success_stop"])

    def test_search_quote_stop_happy_path(self):
        env = RestrictedRetrievalEnv(build_support_episode())
        env.reset()
        search_result = env.step("search")
        quote_result = env.step("quote_evidence")
        stop_result = env.step("stop")
        self.assertAlmostEqual(search_result.reward, 0.05)
        self.assertAlmostEqual(quote_result.reward, 0.95)
        self.assertAlmostEqual(stop_result.reward, 0.95)
        self.assertTrue(stop_result.done)
        self.assertTrue(stop_result.info["success_stop"])
        self.assertEqual(len(stop_result.state.quoted_evidence), 1)

    def test_unknown_stop_without_gold_evidence_is_success(self):
        env = RestrictedRetrievalEnv(build_unknown_episode())
        env.reset()
        env.step("search")
        result = env.step("stop")
        self.assertTrue(result.done)
        self.assertFalse(result.info.get("early_stop", False))
        self.assertTrue(result.info["success_stop"])
        self.assertAlmostEqual(result.reward, 0.95)

    def test_second_quote_without_new_evidence_is_invalid(self):
        env = RestrictedRetrievalEnv(build_support_episode())
        env.reset()
        env.step("search")
        env.step("quote_evidence")
        result = env.step("quote_evidence")
        self.assertFalse(result.done)
        self.assertAlmostEqual(result.reward, -0.55)
        self.assertEqual(result.info["invalid_action"], "quote_evidence")

    def test_max_steps_terminates_episode(self):
        env = RestrictedRetrievalEnv(build_support_episode(max_steps=2))
        env.reset()
        first = env.step("search")
        second = env.step("search")
        self.assertFalse(first.done)
        self.assertTrue(second.done)
        self.assertEqual(second.info["termination_reason"], "max_steps")
        self.assertAlmostEqual(second.reward, -0.25)

    def test_search_uses_frozen_verifier_to_choose_best_doc(self):
        env = RestrictedRetrievalEnv(build_ranked_episode(), frozen_verifier=FakeFrozenVerifier())
        state = env.reset()
        self.assertEqual(state.revealed_docs, [])
        result = env.step("search")
        self.assertEqual(result.state.revealed_docs, ["doc-b"])
        self.assertEqual(result.info["verifier_top_doc"], "doc-b")
        self.assertGreater(result.state.verifier_scores["doc-b"], result.state.verifier_scores["doc-a"])

    def test_state_text_reports_verifier_summary_without_label_hint(self):
        env = RestrictedRetrievalEnv(build_ranked_episode(), frozen_verifier=FakeFrozenVerifier())
        env.reset()
        result = env.step("search")
        rendered = result.state.to_text()
        self.assertIn("Verifier Summary", rendered)
        self.assertIn("doc-b", rendered)
        self.assertNotIn("SUPPORT", rendered)


if __name__ == "__main__":
    unittest.main()
