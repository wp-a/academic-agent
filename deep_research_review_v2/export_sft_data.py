from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from deep_research_review_v2.trajectory import ACTION_SPACE, ClaimTrajectory, load_trajectories


SYSTEM_PROMPT = (
    "You are an evidence-seeking research agent. "
    "Work on the claim by searching for supporting and contradicting evidence, "
    "update the evidence graph when needed, and stop only when the current record is sufficient."
)


def build_context(trajectory: ClaimTrajectory, current_step_idx: int) -> str:
    history_lines: List[str] = []
    for step in trajectory.steps[:current_step_idx]:
        history_lines.append(f"Observation: {step.observation}")
        history_lines.append(
            "Action: "
            f"{step.action.action_type} | argument={step.action.argument} | rationale={step.action.rationale}"
        )
        if step.action.evidence:
            for item in step.action.evidence:
                history_lines.append(
                    f"Evidence: doc_id={item.doc_id} stance={item.stance} score={item.score:.2f} snippet={item.snippet}"
                )
        if step.action.graph_update:
            history_lines.append(f"GraphUpdate: {json.dumps(step.action.graph_update, ensure_ascii=False)}")
    return "\n".join(history_lines) if history_lines else "No previous steps."


def next_action_target(trajectory: ClaimTrajectory, step_idx: int) -> str:
    step = trajectory.steps[step_idx]
    return json.dumps(
        {
            "action_type": step.action.action_type,
            "argument": step.action.argument,
            "rationale": step.action.rationale,
        },
        ensure_ascii=False,
    )


def stopping_target(trajectory: ClaimTrajectory, step_idx: int) -> str:
    step = trajectory.steps[step_idx]
    return json.dumps(
        {
            "should_stop": "yes" if step.should_stop else "no",
            "reason": step.stop_reason,
        },
        ensure_ascii=False,
    )


def build_prompt(task: str, trajectory: ClaimTrajectory, step_idx: int) -> str:
    step = trajectory.steps[step_idx]
    instruction = (
        "Decide the single best next action from the allowed action space."
        if task == "next_action"
        else "Decide whether the agent should stop now."
    )
    return (
        f"System:\n{SYSTEM_PROMPT}\n\n"
        f"Task:\n{instruction}\n\n"
        f"Claim:\n{trajectory.claim}\n\n"
        f"Hypothesis:\n{trajectory.hypothesis}\n\n"
        f"Current Observation:\n{step.observation}\n\n"
        f"Action Space:\n{', '.join(step.candidate_actions or ACTION_SPACE)}\n\n"
        f"History:\n{build_context(trajectory, step_idx)}\n\n"
        f"Answer in JSON."
    )


def export_examples(
    trajectories: List[ClaimTrajectory],
    task: str,
) -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []
    for trajectory in trajectories:
        for step_idx in range(len(trajectory.steps)):
            prompt = build_prompt(task=task, trajectory=trajectory, step_idx=step_idx)
            target = (
                next_action_target(trajectory, step_idx)
                if task == "next_action"
                else stopping_target(trajectory, step_idx)
            )
            examples.append(
                {
                    "task": task,
                    "trajectory_id": trajectory.trajectory_id,
                    "step_id": trajectory.steps[step_idx].step_id,
                    "prompt": prompt,
                    "response": target,
                }
            )
    return examples


def split_train_eval(examples: List[Dict[str, str]], eval_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    eval_count = max(1, int(len(shuffled) * eval_ratio)) if shuffled else 0
    eval_examples = shuffled[:eval_count]
    train_examples = shuffled[eval_count:]
    return train_examples, eval_examples


def dump_jsonl(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectories = load_trajectories(args.input)
    if not trajectories:
        raise ValueError(f"No trajectories found in {args.input}")
    for task in ("next_action", "stopping"):
        examples = export_examples(trajectories, task=task)
        train_records, eval_records = split_train_eval(examples, args.eval_ratio, args.seed)
        dump_jsonl(train_records, args.output_dir / f"{task}_train.jsonl")
        dump_jsonl(eval_records, args.output_dir / f"{task}_eval.jsonl")
        stats = {
            "task": task,
            "num_examples": len(examples),
            "num_train": len(train_records),
            "num_eval": len(eval_records),
        }
        dump_jsonl([stats], args.output_dir / f"{task}_stats.jsonl")
        print(json.dumps(stats))


if __name__ == "__main__":
    main()
