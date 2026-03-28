from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from train_agent.eval.action_policy_metrics import compute_action_policy_metrics
from train_agent.models.action_policy import FrozenActionPolicy


def load_eval_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_prediction_rows(
    *,
    rows: List[Dict[str, object]],
    logits: List[List[float]],
    gold_labels: List[int],
    pred_ids: List[int],
    label_names: List[str],
) -> List[Dict[str, object]]:
    prediction_rows: List[Dict[str, object]] = []
    for row, row_logits, gold_id, pred_id in zip(rows, logits, gold_labels, pred_ids):
        prediction_rows.append(
            {
                "trajectory_id": str(row["trajectory_id"]),
                "step_id": int(row["step_id"]),
                "task": str(row["task"]),
                "text": str(row["text"]),
                "gold_label": label_names[gold_id],
                "predicted_label": label_names[pred_id],
                "is_correct": bool(gold_id == pred_id),
                "scores": {
                    label_names[idx]: round(float(score), 6) for idx, score in enumerate(row_logits)
                },
            }
        )
    return prediction_rows


def evaluate_action_policy_file(
    *,
    model_dir: Path,
    eval_file: Path,
    max_length: int,
    batch_size: int,
    attn_implementation: str,
    include_predictions: bool = False,
) -> Dict[str, object]:
    rows = load_eval_rows(eval_file)
    if not rows:
        raise ValueError(f"No evaluation examples found in {eval_file}")

    policy = FrozenActionPolicy(
        model_dir,
        max_length=max_length,
        batch_size=batch_size,
        attn_implementation=attn_implementation,
    )
    label_to_id = {label: idx for idx, label in enumerate(policy.label_names)}
    texts = [str(row["text"]) for row in rows]
    gold_labels: List[int] = []
    for row in rows:
        label = str(row["label"])
        if label not in label_to_id:
            raise ValueError(f"Unknown label '{label}' not present in model label_names.json")
        gold_labels.append(label_to_id[label])

    logits = policy.predict_logits(texts)
    pred_ids = [max(range(len(row)), key=lambda idx: row[idx]) for row in logits]
    metrics = compute_action_policy_metrics(logits=logits, labels=gold_labels, label_names=policy.label_names)

    gold_distribution = Counter(policy.label_names[label_id] for label_id in gold_labels)
    prediction_distribution = Counter(policy.label_names[label_id] for label_id in pred_ids)
    summary: Dict[str, object] = {
        "model_dir": str(model_dir),
        "eval_file": str(eval_file),
        "num_examples": len(rows),
        "label_names": list(policy.label_names),
        "gold_distribution": {label: int(gold_distribution.get(label, 0)) for label in policy.label_names},
        "prediction_distribution": {
            label: int(prediction_distribution.get(label, 0)) for label in policy.label_names
        },
    }
    summary.update(metrics)
    if include_predictions:
        prediction_rows = build_prediction_rows(
            rows=rows,
            logits=logits,
            gold_labels=gold_labels,
            pred_ids=pred_ids,
            label_names=list(policy.label_names),
        )
        summary["prediction_rows"] = prediction_rows
        summary["error_rows"] = [row for row in prediction_rows if not row["is_correct"]]
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--eval_file", type=Path, required=True)
    parser.add_argument("--output_path", type=Path)
    parser.add_argument("--predictions_output_path", type=Path)
    parser.add_argument("--errors_output_path", type=Path)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--attn_implementation", default="sdpa")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_predictions = args.predictions_output_path is not None or args.errors_output_path is not None
    summary = evaluate_action_policy_file(
        model_dir=args.model_dir,
        eval_file=args.eval_file,
        max_length=args.max_length,
        batch_size=args.batch_size,
        attn_implementation=args.attn_implementation,
        include_predictions=include_predictions,
    )
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.predictions_output_path is not None:
        args.predictions_output_path.parent.mkdir(parents=True, exist_ok=True)
        args.predictions_output_path.write_text(
            json.dumps(summary["prediction_rows"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.errors_output_path is not None:
        args.errors_output_path.parent.mkdir(parents=True, exist_ok=True)
        args.errors_output_path.write_text(
            json.dumps(summary["error_rows"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
