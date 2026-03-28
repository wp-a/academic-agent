from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

from train_agent.data.adapters.common import build_document_map, build_restricted_episode, build_verifier_examples, normalize_verifier_label
from train_agent.data.schemas import VerifierExample
from train_agent.rl.restricted_retrieval import RestrictedRetrievalEpisode


def _hover_positive_labels(row: Mapping[str, object]) -> Dict[Tuple[str, int], str]:
    normalized_label = normalize_verifier_label(row.get("label"))
    if normalized_label == "NEUTRAL":
        return {}
    positives: Dict[Tuple[str, int], str] = {}
    supporting_facts = row.get("supporting_facts") or []
    for item in supporting_facts:
        if not isinstance(item, Mapping):
            continue
        doc_id = item.get("doc_id") or item.get("title")
        sentence_id = item.get("sentence_id")
        if doc_id is None or sentence_id is None:
            continue
        positives[(str(doc_id), int(sentence_id))] = normalized_label
    return positives


def build_hover_verifier_examples(
    row: Mapping[str, object],
    corpus: Optional[Mapping[str, object]] = None,
) -> list[VerifierExample]:
    document_map = build_document_map(row, corpus=corpus)
    return build_verifier_examples(
        dataset="hover",
        sample_id=str(row.get("id") or row.get("claim_id") or "unknown"),
        claim=str(row.get("claim") or ""),
        document_map=document_map,
        positive_labels=_hover_positive_labels(row),
    )


def build_hover_restricted_episode(
    row: Mapping[str, object],
    corpus: Optional[Mapping[str, object]] = None,
    max_steps: int = 4,
) -> RestrictedRetrievalEpisode:
    document_map = build_document_map(row, corpus=corpus)
    return build_restricted_episode(
        episode_prefix="hover",
        sample_id=str(row.get("id") or row.get("claim_id") or "unknown"),
        claim=str(row.get("claim") or ""),
        raw_label=row.get("label"),
        document_map=document_map,
        positive_labels=_hover_positive_labels(row),
        max_steps=max_steps,
    )
