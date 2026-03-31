import unittest

from train_agent.data.adapters.scifact_hard import (
    augment_episode_with_lexical_distractors,
    select_lexical_hard_distractors,
)
from train_agent.rl.restricted_retrieval import RestrictedEvidence, RestrictedRetrievalEpisode, RestrictedRetrievalState
from train_agent.scripts.export_scifact_hard_replay_data import ConservativeReplayPolicy


class SciFactHardReplayTest(unittest.TestCase):
    def test_select_lexical_hard_distractors_prefers_overlap_and_excludes_gold_docs(self):
        claim = "tree canopy lowers summer temperature in cities"
        corpus_text_by_doc = {
            "gold-doc": "tree canopy lowers surface temperature in urban blocks",
            "hard-doc-1": "urban tree canopy and summer heat island temperature patterns",
            "hard-doc-2": "city summer temperature depends on canopy cover and density",
            "easy-doc": "protein folding in yeast cells",
        }
        selected = select_lexical_hard_distractors(
            claim=claim,
            corpus_text_by_doc=corpus_text_by_doc,
            excluded_doc_ids={"gold-doc"},
            num_distractor_docs=2,
        )
        self.assertEqual(selected, ["hard-doc-1", "hard-doc-2"])

    def test_augment_episode_with_lexical_distractors_appends_new_docs(self):
        episode = RestrictedRetrievalEpisode(
            episode_id="ep-1",
            claim="tree canopy lowers summer temperature in cities",
            label_hint="SUPPORT",
            doc_pool=["gold-doc"],
            gold_evidence=[
                RestrictedEvidence(
                    doc_id="gold-doc",
                    sentence_ids=[0],
                    stance="SUPPORT",
                    snippet="tree canopy lowers surface temperature",
                )
            ],
            document_contents={"gold-doc": "tree canopy lowers surface temperature"},
            document_sentences={"gold-doc": ["tree canopy lowers surface temperature"]},
            max_steps=5,
        )
        augmented = augment_episode_with_lexical_distractors(
            episode=episode,
            corpus_text_by_doc={
                "gold-doc": "tree canopy lowers surface temperature",
                "hard-doc": "urban tree canopy and summer temperature overlap",
                "easy-doc": "protein folding in yeast cells",
            },
            corpus_sentences_by_doc={
                "gold-doc": ["tree canopy lowers surface temperature"],
                "hard-doc": ["urban tree canopy and summer temperature overlap"],
                "easy-doc": ["protein folding in yeast cells"],
            },
            num_distractor_docs=1,
        )
        self.assertEqual(augmented.doc_pool, ["gold-doc", "hard-doc"])
        self.assertIn("hard-doc", augmented.document_contents)
        self.assertEqual(augmented.gold_evidence[0].doc_id, "gold-doc")

    def test_conservative_replay_policy_searches_once_after_first_quote(self):
        policy = ConservativeReplayPolicy(gold_doc_ids={"gold-doc"}, post_quote_search_budget=1)
        state = RestrictedRetrievalState(
            claim="Claim",
            doc_pool=["gold-doc", "hard-doc"],
            revealed_docs=["gold-doc"],
            revealed_evidence=[
                RestrictedEvidence(doc_id="gold-doc", sentence_ids=[0], stance="SUPPORT", snippet="evidence")
            ],
            quoted_evidence=[
                RestrictedEvidence(doc_id="gold-doc", sentence_ids=[0], stance="SUPPORT", snippet="evidence")
            ],
            step_index=2,
            max_steps=5,
        )
        self.assertEqual(policy.choose_action(state), "search")


if __name__ == "__main__":
    unittest.main()
