from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class ClaimRecord:
    sample_id: str
    dataset: str
    claim: str
    hypothesis: str = ""
    context: Dict[str, str] = field(default_factory=dict)
    documents: List[Dict[str, str]] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class EvidenceRecord:
    doc_id: str
    snippet: str
    stance: str
    score: float = 1.0
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class TrajectoryExample:
    trajectory_id: str
    step_id: int
    task: str
    prompt: str
    response: str
    metadata: Optional[Dict[str, str]] = None


@dataclass
class VerifierExample:
    example_id: str
    sample_id: str
    dataset: str
    group_id: str
    claim: str
    evidence_text: str
    doc_id: str
    sentence_id: int
    label: str
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
