from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


ACTION_SPACE = ["search", "quote_evidence", "stop"]


@dataclass(frozen=True)
class RestrictedEvidence:
    doc_id: str
    sentence_ids: List[int]
    stance: str
    snippet: str = ""


@dataclass(frozen=True)
class RestrictedRetrievalEpisode:
    episode_id: str
    claim: str
    label_hint: str
    doc_pool: List[str]
    gold_evidence: List[RestrictedEvidence]
    document_contents: Dict[str, str] = field(default_factory=dict)
    document_sentences: Dict[str, List[str]] = field(default_factory=dict)
    max_steps: int = 4


@dataclass
class RestrictedRetrievalState:
    claim: str
    doc_pool: List[str]
    document_contents: Dict[str, str] = field(default_factory=dict)
    document_sentences: Dict[str, List[str]] = field(default_factory=dict)
    revealed_docs: List[str] = field(default_factory=list)
    revealed_evidence: List[RestrictedEvidence] = field(default_factory=list)
    quoted_evidence: List[RestrictedEvidence] = field(default_factory=list)
    verifier_scores: Dict[str, float] = field(default_factory=dict)
    history: List[str] = field(default_factory=list)
    step_index: int = 0
    max_steps: int = 4

    def to_text(self) -> str:
        history = "\n".join(self.history) if self.history else "No previous steps."
        evidence_lines = []
        for item in self.revealed_evidence:
            snippet = item.snippet if item.snippet else "N/A"
            sentence_ids = ",".join(str(idx) for idx in item.sentence_ids) or "N/A"
            evidence_lines.append(
                f"doc_id={item.doc_id} stance={item.stance.lower()} sentence_ids={sentence_ids} snippet={snippet}"
            )
        evidence_text = "\n".join(evidence_lines) if evidence_lines else "No evidence recorded."
        verifier_lines = []
        if self.verifier_scores:
            ranked = sorted(self.verifier_scores.items(), key=lambda item: item[1], reverse=True)
            verifier_lines = [f"doc_id={doc_id} score={score:.4f}" for doc_id, score in ranked[:3]]
        verifier_text = "\n".join(verifier_lines) if verifier_lines else "No verifier scores recorded."
        return (
            "System:\n"
            "You are an evidence-seeking agent. Choose the next action that improves evidence coverage.\n\n"
            f"Claim:\n{self.claim}\n\n"
            f"Current Observation:\nRevealed {len(self.revealed_docs)} of {len(self.doc_pool)} documents.\n\n"
            f"History:\n{history}\n\n"
            f"Verifier Summary:\n{verifier_text}\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            "Action Space:\n" + ", ".join(ACTION_SPACE) + "\n\n"
            "Return one JSON action."
        )


@dataclass
class StepResult:
    state: RestrictedRetrievalState
    reward: float
    done: bool
    info: Dict[str, object] = field(default_factory=dict)


class FrozenVerifierProtocol:
    def score_documents(self, claim: str, documents: Dict[str, str]) -> Dict[str, float]:
        raise NotImplementedError

    def score_document_sentences(
        self,
        claim: str,
        documents: Dict[str, List[str]],
        *,
        aggregation: str = "max",
        aggregation_top_k: int = 3,
    ) -> Dict[str, float]:
        raise NotImplementedError


def build_scifact_episode(row: Dict[str, object], max_steps: int = 4) -> RestrictedRetrievalEpisode:
    cited_doc_ids = [str(doc_id) for doc_id in row.get("cited_doc_ids", [])]
    evidence_doc_id = str(row.get("evidence_doc_id", "") or "")
    evidence_sentences = [int(idx) for idx in row.get("evidence_sentences", [])]
    evidence_label = str(row.get("evidence_label", "") or "")
    label_hint = evidence_label if evidence_label else "UNKNOWN"

    document_contents = {}
    document_sentences = {}
    if isinstance(row.get("documents"), list):
        for item in row["documents"]:
            if not isinstance(item, dict):
                continue
            doc_id = str(item.get("doc_id") or "")
            if not doc_id:
                continue
            sentences = [str(sentence) for sentence in item.get("sentences", [])]
            document_sentences[doc_id] = sentences
            document_contents[doc_id] = " ".join(sentences)

    doc_pool = cited_doc_ids.copy()
    if evidence_doc_id and evidence_doc_id not in doc_pool:
        doc_pool.insert(0, evidence_doc_id)
    if not doc_pool and evidence_doc_id:
        doc_pool = [evidence_doc_id]

    gold_evidence: List[RestrictedEvidence] = []
    if evidence_doc_id and evidence_sentences:
        gold_evidence.append(
            RestrictedEvidence(
                doc_id=evidence_doc_id,
                sentence_ids=evidence_sentences,
                stance=label_hint,
                snippet="",
            )
        )

    return RestrictedRetrievalEpisode(
        episode_id="scifact-{}".format(row["id"]),
        claim=str(row["claim"]),
        label_hint=label_hint,
        doc_pool=doc_pool,
        gold_evidence=gold_evidence,
        document_contents=document_contents,
        document_sentences=document_sentences,
        max_steps=max_steps,
    )


class RestrictedRetrievalEnv:
    def __init__(
        self,
        episode: RestrictedRetrievalEpisode,
        frozen_verifier: Optional[FrozenVerifierProtocol] = None,
        *,
        doc_aggregation: str = "full_document",
        aggregation_top_k: int = 3,
    ):
        self.episode = episode
        self.frozen_verifier = frozen_verifier
        self.doc_aggregation = doc_aggregation
        self.aggregation_top_k = max(1, aggregation_top_k)
        self._gold_by_doc = {item.doc_id: item for item in episode.gold_evidence}
        self._state: RestrictedRetrievalState | None = None
        self._done = False

    def reset(self) -> RestrictedRetrievalState:
        self._done = False
        self._state = RestrictedRetrievalState(
            claim=self.episode.claim,
            doc_pool=self.episode.doc_pool.copy(),
            document_contents=self.episode.document_contents.copy(),
            document_sentences={doc_id: list(sentences) for doc_id, sentences in self.episode.document_sentences.items()},
            step_index=0,
            max_steps=self.episode.max_steps,
        )
        return self._state

    @property
    def state(self) -> RestrictedRetrievalState:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        return self._state

    def step(self, action_type: str) -> StepResult:
        if self._done:
            raise RuntimeError("Episode already finished.")
        if action_type not in ACTION_SPACE:
            raise ValueError(f"Unsupported action_type: {action_type}")

        state = self.state
        state.step_index += 1
        reward = -0.05
        info: Dict[str, object] = {
            "action_type": action_type,
            "label_hint": self.episode.label_hint,
        }

        if action_type == "search":
            reward += self._apply_search(state, info)
        elif action_type == "quote_evidence":
            reward += self._apply_quote(state, info)
        else:
            reward += self._apply_stop(state, info)

        done = bool(info.get("termination_reason"))
        if not done and state.step_index >= state.max_steps:
            done = True
            reward += 0.0 if self._is_success_stop(state) else -0.1
            info["termination_reason"] = "max_steps"
            info["success_stop"] = self._is_success_stop(state)

        self._done = done
        return StepResult(state=state, reward=round(reward, 4), done=done, info=info)

    def _rank_unrevealed_docs(self, state: RestrictedRetrievalState):
        unrevealed_docs = [doc_id for doc_id in state.doc_pool if doc_id not in state.revealed_docs]
        if not unrevealed_docs:
            return unrevealed_docs, {}
        if self.frozen_verifier is None:
            return unrevealed_docs, {}

        if self.doc_aggregation != "full_document" and state.document_sentences:
            documents = {doc_id: list(state.document_sentences.get(doc_id, [])) for doc_id in unrevealed_docs}
            scores = {
                doc_id: float(score)
                for doc_id, score in self.frozen_verifier.score_document_sentences(
                    state.claim,
                    documents,
                    aggregation=self.doc_aggregation,
                    aggregation_top_k=self.aggregation_top_k,
                ).items()
                if doc_id in documents
            }
        else:
            documents = {doc_id: state.document_contents.get(doc_id, "") for doc_id in unrevealed_docs}
            scores = {
                doc_id: float(score)
                for doc_id, score in self.frozen_verifier.score_documents(state.claim, documents).items()
                if doc_id in documents
            }

        for doc_id in unrevealed_docs:
            scores.setdefault(doc_id, 0.0)
        state.verifier_scores.update(scores)
        ranked_docs = sorted(
            unrevealed_docs,
            key=lambda doc_id: (scores.get(doc_id, 0.0), -state.doc_pool.index(doc_id)),
            reverse=True,
        )
        return ranked_docs, scores

    def _apply_search(self, state: RestrictedRetrievalState, info: Dict[str, object]) -> float:
        ranked_docs, scores = self._rank_unrevealed_docs(state)
        unrevealed_docs = ranked_docs or [doc_id for doc_id in state.doc_pool if doc_id not in state.revealed_docs]
        if not unrevealed_docs:
            info["invalid_action"] = "search"
            info["search_miss"] = True
            return -0.1

        doc_id = unrevealed_docs[0]
        if scores:
            info["verifier_top_doc"] = doc_id
            info["verifier_ranking"] = [
                {"doc_id": candidate_id, "score": round(scores[candidate_id], 6)}
                for candidate_id in unrevealed_docs
            ]
        state.revealed_docs.append(doc_id)
        state.history.append(f"Action: search | doc_id={doc_id}")

        evidence = self._gold_by_doc.get(doc_id)
        if evidence is None:
            state.history.append(f"Observation: no evidence found in doc {doc_id}")
            info["search_miss"] = True
            return -0.1

        state.revealed_evidence.append(evidence)
        state.history.append(
            f"Observation: evidence available in doc {doc_id} at sentences {evidence.sentence_ids}"
        )
        info["search_miss"] = False
        return 0.1

    def _apply_quote(self, state: RestrictedRetrievalState, info: Dict[str, object]) -> float:
        quoted_doc_ids = {quoted.doc_id for quoted in state.quoted_evidence}
        remaining = [item for item in state.revealed_evidence if item.doc_id not in quoted_doc_ids]
        if not remaining:
            info["invalid_action"] = "quote_evidence"
            return -0.5

        evidence = remaining[0]
        state.quoted_evidence.append(evidence)
        state.history.append(
            f"Action: quote_evidence | doc_id={evidence.doc_id} | sentence_ids={evidence.sentence_ids}"
        )
        info["quoted_stance"] = evidence.stance
        return 1.0

    def _apply_stop(self, state: RestrictedRetrievalState, info: Dict[str, object]) -> float:
        state.history.append("Action: stop")
        info["termination_reason"] = "stopped"
        success = self._is_success_stop(state)
        info["success_stop"] = success
        if self.episode.label_hint != "UNKNOWN" and not state.quoted_evidence:
            info["early_stop"] = True
        return 1.0 if success else -1.0

    def _is_success_stop(self, state: RestrictedRetrievalState) -> bool:
        if self.episode.label_hint == "UNKNOWN":
            return len(self.episode.gold_evidence) == 0 and len(state.quoted_evidence) == 0
        if not state.quoted_evidence:
            return False
        gold_pairs = {(item.doc_id, tuple(item.sentence_ids), item.stance) for item in self.episode.gold_evidence}
        quoted_pairs = {(item.doc_id, tuple(item.sentence_ids), item.stance) for item in state.quoted_evidence}
        return len(gold_pairs & quoted_pairs) > 0
