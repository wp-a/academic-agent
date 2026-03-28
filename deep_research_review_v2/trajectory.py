from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


ACTION_SPACE = [
    "search",
    "open_document",
    "quote_evidence",
    "update_graph",
    "ask_followup",
    "stop",
]


@dataclass
class EvidenceItem:
    doc_id: str
    snippet: str
    stance: str
    score: float = 1.0
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class AgentAction:
    action_type: str
    argument: str
    rationale: str
    evidence: List[EvidenceItem] = field(default_factory=list)
    graph_update: Dict[str, List[str]] = field(default_factory=dict)
    reward_hint: Optional[float] = None


@dataclass
class TrajectoryStep:
    step_id: int
    observation: str
    candidate_actions: List[str]
    action: AgentAction
    should_stop: bool
    stop_reason: str


@dataclass
class ClaimTrajectory:
    trajectory_id: str
    claim: str
    hypothesis: str
    label: str
    metadata: Dict[str, str] = field(default_factory=dict)
    steps: List[TrajectoryStep] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, payload: Dict) -> "ClaimTrajectory":
        steps: List[TrajectoryStep] = []
        for raw_step in payload.get("steps", []):
            action_payload = raw_step["action"]
            evidence = [EvidenceItem(**item) for item in action_payload.get("evidence", [])]
            action = AgentAction(
                action_type=action_payload["action_type"],
                argument=action_payload["argument"],
                rationale=action_payload["rationale"],
                evidence=evidence,
                graph_update=action_payload.get("graph_update", {}),
                reward_hint=action_payload.get("reward_hint"),
            )
            steps.append(
                TrajectoryStep(
                    step_id=raw_step["step_id"],
                    observation=raw_step["observation"],
                    candidate_actions=raw_step.get("candidate_actions", ACTION_SPACE),
                    action=action,
                    should_stop=raw_step["should_stop"],
                    stop_reason=raw_step.get("stop_reason", ""),
                )
            )
        return cls(
            trajectory_id=payload["trajectory_id"],
            claim=payload["claim"],
            hypothesis=payload["hypothesis"],
            label=payload["label"],
            metadata=payload.get("metadata", {}),
            steps=steps,
            created_at=payload.get("created_at", datetime.utcnow().isoformat()),
        )


class TrajectoryRecorder:
    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, trajectory: ClaimTrajectory) -> None:
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(trajectory.to_json())
            handle.write("\n")

    def extend(self, trajectories: Iterable[ClaimTrajectory]) -> None:
        with self.output_path.open("a", encoding="utf-8") as handle:
            for trajectory in trajectories:
                handle.write(trajectory.to_json())
                handle.write("\n")


def load_trajectories(path: Path) -> List[ClaimTrajectory]:
    records: List[ClaimTrajectory] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(ClaimTrajectory.from_dict(json.loads(line)))
    return records
