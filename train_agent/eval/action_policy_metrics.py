from __future__ import annotations

from typing import Dict, Sequence


def _argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=lambda idx: values[idx])


def _confusion_matrix(predictions: Sequence[int], labels: Sequence[int], num_labels: int):
    matrix = [[0 for _ in range(num_labels)] for _ in range(num_labels)]
    for gold, pred in zip(labels, predictions):
        matrix[int(gold)][int(pred)] += 1
    return matrix


def _per_class_metrics(matrix, label_names: Sequence[str]) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for label_idx, label_name in enumerate(label_names):
        tp = int(matrix[label_idx][label_idx])
        fp = sum(int(matrix[row_idx][label_idx]) for row_idx in range(len(label_names)) if row_idx != label_idx)
        fn = sum(int(matrix[label_idx][col_idx]) for col_idx in range(len(label_names)) if col_idx != label_idx)
        support = sum(int(value) for value in matrix[label_idx])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2.0 * precision * recall) / (precision + recall) if precision + recall else 0.0
        metrics[str(label_name)] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }
    return metrics


def compute_action_policy_metrics(*, logits: Sequence[Sequence[float]], labels: Sequence[int], label_names: Sequence[str]) -> Dict[str, object]:
    predictions = [_argmax(row) for row in logits]
    accuracy = sum(int(pred == gold) for pred, gold in zip(predictions, labels)) / max(len(labels), 1)
    confusion_matrix = _confusion_matrix(predictions, labels, len(label_names))
    per_class = _per_class_metrics(confusion_matrix, label_names)
    macro_f1 = sum(float(metrics["f1"]) for metrics in per_class.values()) / max(len(per_class), 1)
    return {
        "accuracy": round(accuracy, 6),
        "macro_f1": round(macro_f1, 6),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
    }
