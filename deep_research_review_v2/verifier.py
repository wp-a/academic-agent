from __future__ import annotations

from dataclasses import dataclass
from typing import List

from deep_research_review_v2.trajectory import EvidenceItem


@dataclass
class VerificationResult:
    support_score: float
    contradiction_score: float
    sufficiency_score: float
    notes: List[str]


def heuristic_verify(evidence: List[EvidenceItem]) -> VerificationResult:
    support = sum(item.score for item in evidence if item.stance == "support")
    contradict = sum(item.score for item in evidence if item.stance == "contradict")
    sufficiency = min(1.0, len(evidence) / 4.0)
    notes = []
    if support == 0 and contradict == 0:
        notes.append("No evidence attached.")
    if support and contradict:
        notes.append("Mixed evidence found; prefer another search or graph update step.")
    if sufficiency < 0.75:
        notes.append("Evidence record is still sparse.")
    return VerificationResult(
        support_score=support,
        contradiction_score=contradict,
        sufficiency_score=sufficiency,
        notes=notes,
    )
