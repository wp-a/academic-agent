from __future__ import annotations

import re
from typing import Mapping, Sequence

from train_agent.rl.restricted_retrieval import RestrictedRetrievalEpisode

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _text_terms(text: str) -> set[str]:
    return {match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text) if len(match.group(0)) >= 3}


def select_lexical_hard_distractors(
    *,
    claim: str,
    corpus_text_by_doc: Mapping[str, str],
    excluded_doc_ids: set[str],
    num_distractor_docs: int,
) -> list[str]:
    if num_distractor_docs <= 0:
        return []
    claim_terms = _text_terms(claim)
    ranked = []
    for doc_id, text in corpus_text_by_doc.items():
        if doc_id in excluded_doc_ids:
            continue
        doc_terms = _text_terms(str(text))
        overlap = claim_terms & doc_terms
        overlap_count = len(overlap)
        precision = overlap_count / max(len(doc_terms), 1)
        ranked.append((overlap_count, precision, doc_id))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    selected = [doc_id for overlap_count, _precision, doc_id in ranked if overlap_count > 0][:num_distractor_docs]
    if len(selected) < num_distractor_docs:
        fallbacks = [doc_id for _overlap_count, _precision, doc_id in ranked if doc_id not in selected]
        selected.extend(fallbacks[: max(0, num_distractor_docs - len(selected))])
    return selected


def augment_episode_with_lexical_distractors(
    *,
    episode: RestrictedRetrievalEpisode,
    corpus_text_by_doc: Mapping[str, str],
    corpus_sentences_by_doc: Mapping[str, Sequence[str]],
    num_distractor_docs: int,
) -> RestrictedRetrievalEpisode:
    selected = select_lexical_hard_distractors(
        claim=episode.claim,
        corpus_text_by_doc=corpus_text_by_doc,
        excluded_doc_ids=set(episode.doc_pool),
        num_distractor_docs=num_distractor_docs,
    )
    if not selected:
        return episode
    document_contents = dict(episode.document_contents)
    document_sentences = {doc_id: list(sentences) for doc_id, sentences in episode.document_sentences.items()}
    for doc_id in selected:
        document_contents[doc_id] = str(corpus_text_by_doc.get(doc_id, ""))
        document_sentences[doc_id] = [str(sentence) for sentence in corpus_sentences_by_doc.get(doc_id, [])]
    return RestrictedRetrievalEpisode(
        episode_id=episode.episode_id,
        claim=episode.claim,
        label_hint=episode.label_hint,
        doc_pool=list(episode.doc_pool) + [doc_id for doc_id in selected if doc_id not in episode.doc_pool],
        gold_evidence=list(episode.gold_evidence),
        document_contents=document_contents,
        document_sentences=document_sentences,
        max_steps=episode.max_steps,
    )
