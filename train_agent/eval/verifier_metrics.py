from __future__ import annotations

import math
from typing import Dict, List, Sequence

from train_agent.data.adapters.common import NEGATIVE_VERIFIER_LABELS


def _softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    denom = sum(exps) or 1.0
    return [value / denom for value in exps]


def _argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=lambda idx: values[idx])


def _confusion_matrix(predictions: Sequence[int], labels: Sequence[int], num_labels: int) -> List[List[int]]:
    matrix = [[0 for _ in range(num_labels)] for _ in range(num_labels)]
    for gold, pred in zip(labels, predictions):
        matrix[int(gold)][int(pred)] += 1
    return matrix


def _per_class_metrics(matrix: Sequence[Sequence[int]], label_names: Sequence[str]) -> Dict[str, Dict[str, float]]:
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
            'precision': round(precision, 6),
            'recall': round(recall, 6),
            'f1': round(f1, 6),
            'support': support,
        }
    return metrics


def compute_verifier_metrics(
    *,
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    label_names: Sequence[str],
    group_ids: Sequence[str],
) -> Dict[str, object]:
    predictions = [_argmax(row) for row in logits]
    accuracy = sum(int(pred == gold) for pred, gold in zip(predictions, labels)) / max(len(labels), 1)
    confusion_matrix = _confusion_matrix(predictions, labels, len(label_names))
    per_class = _per_class_metrics(confusion_matrix, label_names)
    macro_f1 = sum(float(metrics['f1']) for metrics in per_class.values()) / max(len(per_class), 1)

    positive_indices = [
        idx for idx, name in enumerate(label_names) if str(name).upper() not in NEGATIVE_VERIFIER_LABELS
    ]
    grouped_rows: Dict[str, List[Dict[str, float]]] = {}
    for group_id, row_logits, gold_label in zip(group_ids, logits, labels):
        probs = _softmax([float(value) for value in row_logits])
        grouped_rows.setdefault(str(group_id), []).append(
            {
                'positive_score': sum(probs[idx] for idx in positive_indices),
                'is_positive': int(gold_label in positive_indices),
            }
        )

    mrr_total = 0.0
    recall_at_1 = 0.0
    recall_at_3 = 0.0
    positive_groups = 0
    for rows in grouped_rows.values():
        if not any(item['is_positive'] for item in rows):
            continue
        positive_groups += 1
        ranked = sorted(rows, key=lambda item: item['positive_score'], reverse=True)
        first_positive_rank = None
        for rank, item in enumerate(ranked, start=1):
            if item['is_positive']:
                first_positive_rank = rank
                break
        if first_positive_rank is None:
            continue
        mrr_total += 1.0 / first_positive_rank
        recall_at_1 += 1.0 if first_positive_rank <= 1 else 0.0
        recall_at_3 += 1.0 if first_positive_rank <= 3 else 0.0

    divisor = max(positive_groups, 1)
    return {
        'accuracy': round(accuracy, 6),
        'macro_f1': round(macro_f1, 6),
        'mrr': round(mrr_total / divisor, 6),
        'recall@1': round(recall_at_1 / divisor, 6),
        'recall@3': round(recall_at_3 / divisor, 6),
        'positive_groups': float(positive_groups),
        'per_class': per_class,
        'confusion_matrix': confusion_matrix,
    }
