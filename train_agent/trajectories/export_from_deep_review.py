from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from deep_research_review_v2.trajectory import load_trajectories
from train_agent.data.schemas import EvidenceRecord
from train_agent.trajectories.state_action_schema import AgentActionLabel, AgentState, ActionPolicyExample


def build_history(trajectory, step_idx: int) -> List[str]:
    history = []
    for step in trajectory.steps[:step_idx]:
        history.append(f"Observation: {step.observation}")
        history.append(
            f"Action: {step.action.action_type} | argument={step.action.argument} | rationale={step.action.rationale}"
        )
    return history


def build_evidence(trajectory, step_idx: int) -> List[EvidenceRecord]:
    records: List[EvidenceRecord] = []
    for step in trajectory.steps[: step_idx + 1]:
        for item in step.action.evidence:
            records.append(
                EvidenceRecord(
                    doc_id=item.doc_id,
                    snippet=item.snippet,
                    stance=item.stance,
                    score=item.score,
                    metadata=item.metadata,
                )
            )
    return records


def export_action_examples(input_path: Path, output_path: Path) -> int:
    trajectories = load_trajectories(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            for step_idx, step in enumerate(trajectory.steps):
                state = AgentState(
                    claim=trajectory.claim,
                    hypothesis=trajectory.hypothesis,
                    observation=step.observation,
                    history=build_history(trajectory, step_idx),
                    evidence=build_evidence(trajectory, step_idx),
                    graph_summary=step.action.graph_update,
                    candidate_actions=step.candidate_actions,
                    budget={"step_index": step_idx, "num_steps": len(trajectory.steps)},
                )
                example = ActionPolicyExample(
                    trajectory_id=trajectory.trajectory_id,
                    step_id=step.step_id,
                    state=state,
                    label=AgentActionLabel(
                        action_type=step.action.action_type,
                        argument=step.action.argument,
                        rationale=step.action.rationale,
                    ),
                )
                row = {
                    "trajectory_id": example.trajectory_id,
                    "step_id": example.step_id,
                    "task": "next_action_classification",
                    "text": example.to_prompt(),
                    "label": example.label.action_type,
                    "label_text": json.dumps(example.label.__dict__, ensure_ascii=False),
                }
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
                count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = export_action_examples(args.input, args.output)
    print(json.dumps({"output": str(args.output), "num_examples": count}))


if __name__ == "__main__":
    main()
