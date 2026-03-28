from __future__ import annotations

import hashlib
import re
from typing import Dict, Mapping, Optional, Tuple

from train_agent.data.adapters.common import (
    build_document_map,
    build_restricted_episode,
    build_verifier_examples,
    normalize_verifier_label,
)
from train_agent.data.schemas import VerifierExample
from train_agent.rl.restricted_retrieval import RestrictedRetrievalEpisode

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _scifact_positive_labels(row: Mapping[str, object]) -> Dict[Tuple[str, int], str]:
    positives: Dict[Tuple[str, int], str] = {}
    evidence = row.get("evidence", [])
    if isinstance(evidence, Mapping):
        for raw_doc_id, payload in evidence.items():
            if not isinstance(payload, Mapping):
                continue
            label = normalize_verifier_label(payload.get("label") or row.get("label"))
            for sentence_id in payload.get("sentence_ids", []):
                positives[(str(raw_doc_id), int(sentence_id))] = label
    elif isinstance(evidence, list):
        for item in evidence:
            if not isinstance(item, Mapping):
                continue
            doc_id = item.get("doc_id") or item.get("evidence_doc_id")
            if doc_id is None:
                continue
            label = normalize_verifier_label(item.get("label") or item.get("evidence_label") or row.get("label"))
            sentence_ids = item.get("sentence_ids") or item.get("evidence_sentences") or []
            for sentence_id in sentence_ids:
                positives[(str(doc_id), int(sentence_id))] = label
    if not positives and row.get("evidence_doc_id") and row.get("evidence_sentences"):
        label = normalize_verifier_label(row.get("evidence_label") or row.get("label"))
        for sentence_id in row.get("evidence_sentences", []):
            positives[(str(row["evidence_doc_id"]), int(sentence_id))] = label
    return positives


def _text_terms(text: str) -> set[str]:
    return {match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text) if len(match.group(0)) >= 3}


def _hard_negative_score(claim: str, sentence: str) -> Tuple[int, float]:
    claim_terms = _text_terms(claim)
    sentence_terms = _text_terms(sentence)
    overlap = claim_terms & sentence_terms
    precision = len(overlap) / max(len(sentence_terms), 1)
    return len(overlap), precision


def _stable_example_tiebreak(example: VerifierExample) -> str:
    digest = hashlib.md5(example.example_id.encode("utf-8")).hexdigest()
    return digest


def _select_relevance_examples(
    *,
    examples: list[VerifierExample],
    claim: str,
    max_hard_negatives_per_positive: int,
    max_random_negatives_per_positive: int,
) -> list[VerifierExample]:
    if max_hard_negatives_per_positive <= 0 and max_random_negatives_per_positive <= 0:
        return examples

    positives = [example for example in examples if example.label == "RELEVANT"]
    negatives = [example for example in examples if example.label != "RELEVANT"]
    if not negatives:
        return positives

    multiplier = max(len(positives), 1)
    hard_budget = max_hard_negatives_per_positive * multiplier
    random_budget = max_random_negatives_per_positive * multiplier

    ranked_negatives = sorted(
        negatives,
        key=lambda example: (
            -_hard_negative_score(claim, example.evidence_text)[0],
            -_hard_negative_score(claim, example.evidence_text)[1],
            example.doc_id,
            example.sentence_id,
        ),
    )
    selected_hard = ranked_negatives[:hard_budget] if hard_budget > 0 else []
    selected_ids = {example.example_id for example in selected_hard}
    remaining_negatives = [example for example in ranked_negatives if example.example_id not in selected_ids]
    selected_random = sorted(remaining_negatives, key=_stable_example_tiebreak)[:random_budget] if random_budget > 0 else []

    selected_examples = positives + selected_hard + selected_random
    return sorted(selected_examples, key=lambda example: (example.doc_id, example.sentence_id, example.label))


def build_scifact_verifier_examples(
    row: Mapping[str, object],
    corpus: Optional[Mapping[str, object]] = None,
) -> list[VerifierExample]:
    document_map = build_document_map(row, corpus=corpus)
    return build_verifier_examples(
        dataset="scifact",
        sample_id=str(row.get("id") or row.get("claim_id") or "unknown"),
        claim=str(row.get("claim") or ""),
        document_map=document_map,
        positive_labels=_scifact_positive_labels(row),
    )


def build_scifact_relevance_examples(
    row: Mapping[str, object],
    corpus: Optional[Mapping[str, object]] = None,
    *,
    max_hard_negatives_per_positive: int = 0,
    max_random_negatives_per_positive: int = 0,
) -> list[VerifierExample]:
    document_map = build_document_map(row, corpus=corpus)
    claim = str(row.get("claim") or "")
    relevance_labels = {key: "RELEVANT" for key in _scifact_positive_labels(row).keys()}
    examples = build_verifier_examples(
        dataset="scifact",
        sample_id=str(row.get("id") or row.get("claim_id") or "unknown"),
        claim=claim,
        document_map=document_map,
        positive_labels=relevance_labels,
    )
    return _select_relevance_examples(
        examples=examples,
        claim=claim,
        max_hard_negatives_per_positive=max_hard_negatives_per_positive,
        max_random_negatives_per_positive=max_random_negatives_per_positive,
    )


def build_scifact_stance_examples(
    row: Mapping[str, object],
    corpus: Optional[Mapping[str, object]] = None,
) -> list[VerifierExample]:
    document_map = build_document_map(row, corpus=corpus)
    sample_id = str(row.get("id") or row.get("claim_id") or "unknown")
    claim = str(row.get("claim") or "")
    examples: list[VerifierExample] = []
    for (doc_id, sentence_id), label in _scifact_positive_labels(row).items():
        if label == "NEUTRAL":
            continue
        sentences = list(document_map.get(doc_id, []))
        if not 0 <= sentence_id < len(sentences):
            continue
        examples.append(
            VerifierExample(
                example_id=f"scifact-{sample_id}-{doc_id}-{sentence_id}",
                sample_id=sample_id,
                dataset="scifact",
                group_id=sample_id,
                claim=claim,
                evidence_text=str(sentences[sentence_id]),
                doc_id=str(doc_id),
                sentence_id=int(sentence_id),
                label=label,
            )
        )
    return examples


def build_scifact_restricted_episode(
    row: Mapping[str, object],
    corpus: Optional[Mapping[str, object]] = None,
    max_steps: int = 4,
) -> RestrictedRetrievalEpisode:
    document_map = build_document_map(row, corpus=corpus)
    return build_restricted_episode(
        episode_prefix="scifact",
        sample_id=str(row.get("id") or row.get("claim_id") or "unknown"),
        claim=str(row.get("claim") or ""),
        raw_label=row.get("label") or row.get("evidence_label"),
        document_map=document_map,
        positive_labels=_scifact_positive_labels(row),
        max_steps=max_steps,
    )
