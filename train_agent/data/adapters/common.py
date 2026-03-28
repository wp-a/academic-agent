from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from train_agent.data.schemas import VerifierExample
from train_agent.rl.restricted_retrieval import RestrictedEvidence, RestrictedRetrievalEpisode


NEGATIVE_VERIFIER_LABELS = {"NEUTRAL", "NOT_ENOUGH_INFO", "UNKNOWN"}
LABEL_ALIASES = {
    "support": "SUPPORT",
    "supports": "SUPPORT",
    "supported": "SUPPORT",
    "entails": "SUPPORT",
    "contradict": "CONTRADICT",
    "contradiction": "CONTRADICT",
    "contradicts": "CONTRADICT",
    "refute": "CONTRADICT",
    "refutes": "CONTRADICT",
    "refuted": "CONTRADICT",
    "not enough info": "NEUTRAL",
    "not_enough_info": "NEUTRAL",
    "nei": "NEUTRAL",
    "neutral": "NEUTRAL",
    "unknown": "NEUTRAL",
}


def normalize_verifier_label(label: object, default: str = "NEUTRAL") -> str:
    if label is None:
        return default
    normalized = str(label).strip().lower().replace("-", " ").replace("_", " ")
    normalized = " ".join(normalized.split())
    if not normalized:
        return default
    return LABEL_ALIASES.get(normalized, str(label).strip().upper())


def build_document_map(row: Mapping[str, object], corpus: Optional[Mapping[str, object]] = None) -> Dict[str, List[str]]:
    documents = row.get("documents")
    if isinstance(documents, list):
        result: Dict[str, List[str]] = {}
        for item in documents:
            if not isinstance(item, Mapping):
                continue
            doc_id = str(item.get("doc_id") or item.get("title") or item.get("id") or "")
            if not doc_id:
                continue
            sentences = item.get("sentences") or item.get("abstract") or item.get("lines") or []
            result[doc_id] = [str(sentence) for sentence in sentences]
        if result:
            return result
    if isinstance(documents, Mapping):
        result = {}
        for raw_doc_id, payload in documents.items():
            doc_id = str(raw_doc_id)
            if isinstance(payload, Mapping):
                sentences = payload.get("sentences") or payload.get("abstract") or payload.get("lines") or []
            else:
                sentences = payload if isinstance(payload, Sequence) else []
            result[doc_id] = [str(sentence) for sentence in sentences]
        if result:
            return result

    result = {}
    if corpus is None:
        return result

    candidate_ids: List[str] = []
    for key in ("cited_doc_ids", "doc_pool", "retrieved_doc_ids"):
        values = row.get(key)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            candidate_ids.extend(str(value) for value in values)
    for key in ("evidence", "supporting_facts", "evidence_sets"):
        values = row.get(key)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            for item in values:
                if isinstance(item, Mapping):
                    doc_id = item.get("doc_id") or item.get("title")
                    if doc_id is not None:
                        candidate_ids.append(str(doc_id))
                elif isinstance(item, Sequence):
                    for nested in item:
                        if isinstance(nested, Mapping):
                            doc_id = nested.get("doc_id") or nested.get("title")
                            if doc_id is not None:
                                candidate_ids.append(str(doc_id))
    for doc_id in dict.fromkeys(candidate_ids):
        payload = corpus.get(doc_id)
        if payload is None:
            continue
        if isinstance(payload, Mapping):
            sentences = payload.get("sentences") or payload.get("abstract") or payload.get("lines") or []
        else:
            sentences = payload if isinstance(payload, Sequence) else []
        result[doc_id] = [str(sentence) for sentence in sentences]
    return result


def build_document_contents(document_map: Mapping[str, Sequence[str]]) -> Dict[str, str]:
    return {
        doc_id: " ".join(sentence.strip() for sentence in sentences if sentence).strip()
        for doc_id, sentences in document_map.items()
    }


def sentence_text(document_map: Mapping[str, Sequence[str]], doc_id: str, sentence_id: int) -> str:
    sentences = list(document_map.get(doc_id, []))
    if 0 <= sentence_id < len(sentences):
        return str(sentences[sentence_id])
    return ""


def build_verifier_examples(
    *,
    dataset: str,
    sample_id: str,
    claim: str,
    document_map: Mapping[str, Sequence[str]],
    positive_labels: Mapping[Tuple[str, int], str],
) -> List[VerifierExample]:
    examples: List[VerifierExample] = []
    group_id = str(sample_id)
    for doc_id, sentences in document_map.items():
        for sentence_id, text in enumerate(sentences):
            label = positive_labels.get((doc_id, sentence_id), "NEUTRAL")
            examples.append(
                VerifierExample(
                    example_id=f"{dataset}-{sample_id}-{doc_id}-{sentence_id}",
                    sample_id=str(sample_id),
                    dataset=dataset,
                    group_id=group_id,
                    claim=claim,
                    evidence_text=str(text),
                    doc_id=str(doc_id),
                    sentence_id=int(sentence_id),
                    label=label,
                )
            )
    return examples


def build_restricted_episode(
    *,
    episode_prefix: str,
    sample_id: str,
    claim: str,
    raw_label: object,
    document_map: Mapping[str, Sequence[str]],
    positive_labels: Mapping[Tuple[str, int], str],
    max_steps: int,
) -> RestrictedRetrievalEpisode:
    gold_evidence = []
    for (doc_id, sentence_id), stance in positive_labels.items():
        gold_evidence.append(
            RestrictedEvidence(
                doc_id=doc_id,
                sentence_ids=[int(sentence_id)],
                stance=stance,
                snippet=sentence_text(document_map, doc_id, sentence_id),
            )
        )
    label_hint = normalize_verifier_label(raw_label)
    if not gold_evidence:
        label_hint = "UNKNOWN"
    return RestrictedRetrievalEpisode(
        episode_id=f"{episode_prefix}-{sample_id}",
        claim=str(claim),
        label_hint=label_hint,
        doc_pool=list(document_map.keys()),
        gold_evidence=gold_evidence,
        document_contents=build_document_contents(document_map),
        max_steps=max_steps,
    )
