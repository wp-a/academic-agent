from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from train_agent.data.schemas import EvidenceRecord


DEFAULT_ACTION_SPACE = [
    "search",
    "open_document",
    "quote_evidence",
    "update_graph",
    "ask_followup",
    "stop",
]


@dataclass
class AgentState:
    claim: str
    hypothesis: str
    observation: str
    history: List[str] = field(default_factory=list)
    evidence: List[EvidenceRecord] = field(default_factory=list)
    graph_summary: Dict[str, List[str]] = field(default_factory=dict)
    candidate_actions: List[str] = field(default_factory=lambda: DEFAULT_ACTION_SPACE.copy())
    budget: Dict[str, int] = field(default_factory=dict)


@dataclass
class AgentActionLabel:
    action_type: str
    argument: str
    rationale: str


@dataclass
class ActionPolicyExample:
    trajectory_id: str
    step_id: int
    state: AgentState
    label: AgentActionLabel

    def to_prompt(self) -> str:
        history = "\n".join(self.state.history) if self.state.history else "No previous steps."
        evidence_lines = []
        for item in self.state.evidence:
            evidence_lines.append(
                f"doc_id={item.doc_id} stance={item.stance} score={item.score:.2f} snippet={item.snippet}"
            )
        evidence_text = "\n".join(evidence_lines) if evidence_lines else "No evidence recorded."
        action_space = ", ".join(self.state.candidate_actions)
        return (
            "System:\n"
            "You are an evidence-seeking agent. Choose the next action that improves evidence coverage.\n\n"
            f"Claim:\n{self.state.claim}\n\n"
            f"Hypothesis:\n{self.state.hypothesis}\n\n"
            f"Current Observation:\n{self.state.observation}\n\n"
            f"History:\n{history}\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            f"Action Space:\n{action_space}\n\n"
            "Return one JSON action."
        )
