from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _normalize_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    return bool(value)


def _get_first_mismatch_step(record: Dict[str, Any]) -> Dict[str, Any]:
    mismatch_indices = list(record.get("mismatch_step_indices") or [])
    target_index = mismatch_indices[0] if mismatch_indices else None
    for step in record.get("steps", []):
        if target_index is not None and step.get("step_index") == target_index:
            return step
    for step in record.get("steps", []):
        if not bool(step.get("action_match", True)):
            return step
    raise ValueError(f"Record for episode {record.get('episode_id')} does not contain a mismatch step.")


def classify_failure_bucket(step: Dict[str, Any]) -> str:
    reference_action = str(step.get("reference_action") or "")
    predicted_action = str(step.get("predicted_action") or "")
    revealed_evidence = list(step.get("revealed_evidence") or [])
    quoted_evidence = list(step.get("quoted_evidence") or [])

    if reference_action == "quote_evidence" and predicted_action == "stop":
        if revealed_evidence and not quoted_evidence:
            return "premature_stop_after_evidence"
        return "premature_stop_instead_of_quote"

    if reference_action == "stop" and predicted_action == "search":
        if quoted_evidence:
            return "oversearch_after_quote"
        return "search_instead_of_stop"

    if reference_action == "stop" and predicted_action == "quote_evidence":
        return "late_quote_instead_of_stop"

    if reference_action == "search" and predicted_action == "stop":
        return "premature_stop_before_search"

    return "other_mismatch"


def infer_error_source(step: Dict[str, Any], off_policy_action_record: Optional[Dict[str, Any]]) -> str:
    reference_action = str(step.get("reference_action") or "")
    predicted_action = str(step.get("predicted_action") or "")
    metadata = (off_policy_action_record or {}).get("metadata", {}) or {}
    used_stop_policy = bool(metadata.get("used_stop_policy"))
    stop_policy_should_stop = _normalize_bool(metadata.get("stop_policy_should_stop"))

    if reference_action == "stop" and predicted_action != "stop":
        if used_stop_policy and stop_policy_should_stop is False:
            return "stop_policy_false_negative"
        return "action_policy_non_stop_when_should_stop"

    if reference_action != "stop" and predicted_action == "stop":
        if used_stop_policy and stop_policy_should_stop is True:
            return "stop_policy_false_positive"
        return "action_policy_stop_when_should_continue"

    return "action_policy_label_error"


def _index_off_policy_records(records: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    indexed: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for record in records:
        metadata = record.get("metadata", {}) or {}
        episode_id = str(metadata.get("episode_id") or record.get("trajectory_id") or "")
        step_id = int(record.get("step_id", 0))
        indexed[(episode_id, step_id)] = record
    return indexed


def analyze_mismatch_records(
    diagnostics_records: List[Dict[str, Any]],
    *,
    off_policy_action_records: Optional[List[Dict[str, Any]]] = None,
    off_policy_stop_records: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    action_records = off_policy_action_records or []
    stop_records = off_policy_stop_records or []
    action_index = _index_off_policy_records(action_records)
    stop_count_by_episode: Dict[str, int] = defaultdict(int)
    action_count_by_episode: Dict[str, int] = defaultdict(int)
    for record in action_records:
        metadata = record.get("metadata", {}) or {}
        episode_id = str(metadata.get("episode_id") or record.get("trajectory_id") or "")
        action_count_by_episode[episode_id] += 1
    for record in stop_records:
        metadata = record.get("metadata", {}) or {}
        episode_id = str(metadata.get("episode_id") or record.get("trajectory_id") or "")
        stop_count_by_episode[episode_id] += 1

    bucket_counts: Counter = Counter()
    error_source_counts: Counter = Counter()
    first_mismatch_action_pairs: Counter = Counter()
    stop_policy_decision_counts: Counter = Counter()
    episodes: List[Dict[str, Any]] = []
    total_mismatch_steps = 0

    for record in diagnostics_records:
        first_mismatch_step = _get_first_mismatch_step(record)
        episode_id = str(record.get("episode_id") or "")
        step_index = int(first_mismatch_step.get("step_index", 0))
        off_policy_action = action_index.get((episode_id, step_index))
        bucket = classify_failure_bucket(first_mismatch_step)
        error_source = infer_error_source(first_mismatch_step, off_policy_action)
        reference_action = str(first_mismatch_step.get("reference_action") or "")
        predicted_action = str(first_mismatch_step.get("predicted_action") or "")
        action_pair = f"{reference_action}->{predicted_action}"
        stop_policy_should_stop = _normalize_bool(
            (off_policy_action or {}).get("metadata", {}).get(
                "stop_policy_should_stop",
                first_mismatch_step.get("stop_policy_should_stop"),
            )
        )

        bucket_counts[bucket] += 1
        error_source_counts[error_source] += 1
        first_mismatch_action_pairs[action_pair] += 1
        stop_policy_decision_counts[str(stop_policy_should_stop)] += 1
        total_mismatch_steps += int(record.get("num_mismatches", len(record.get("mismatch_step_indices") or [])))

        episodes.append(
            {
                "episode_id": episode_id,
                "claim": str(record.get("claim") or ""),
                "label_hint": str(record.get("label_hint") or ""),
                "bucket": bucket,
                "error_source": error_source,
                "first_mismatch_step_index": step_index,
                "first_reference_action": reference_action,
                "first_predicted_action": predicted_action,
                "stop_policy_should_stop": stop_policy_should_stop,
                "num_mismatches": int(record.get("num_mismatches", 0)),
                "off_policy_action_examples": action_count_by_episode.get(episode_id, 0),
                "off_policy_stop_examples": stop_count_by_episode.get(episode_id, 0),
                "reference_policy_type": str(record.get("reference_policy_type") or ""),
                "post_quote_search_budget": int(record.get("post_quote_search_budget", 0)),
            }
        )

    episodes.sort(key=lambda item: item["episode_id"])
    return {
        "mismatch_episode_count": len(diagnostics_records),
        "mismatch_step_count": total_mismatch_steps,
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "error_source_counts": dict(sorted(error_source_counts.items())),
        "first_mismatch_action_pairs": dict(sorted(first_mismatch_action_pairs.items())),
        "first_mismatch_stop_policy_should_stop": dict(sorted(stop_policy_decision_counts.items())),
        "off_policy_action_example_count": len(action_records),
        "off_policy_stop_example_count": len(stop_records),
        "episodes": episodes,
    }


def analyze_mismatch_files(
    *,
    diagnostics_path: Path,
    off_policy_action_path: Optional[Path] = None,
    off_policy_stop_path: Optional[Path] = None,
) -> Dict[str, Any]:
    diagnostics_records = load_jsonl(diagnostics_path)
    off_policy_action_records = load_jsonl(off_policy_action_path) if off_policy_action_path is not None else []
    off_policy_stop_records = load_jsonl(off_policy_stop_path) if off_policy_stop_path is not None else []
    summary = analyze_mismatch_records(
        diagnostics_records,
        off_policy_action_records=off_policy_action_records,
        off_policy_stop_records=off_policy_stop_records,
    )
    summary.update(
        {
            "diagnostics_path": str(diagnostics_path),
            "off_policy_action_path": str(off_policy_action_path) if off_policy_action_path is not None else "",
            "off_policy_stop_path": str(off_policy_stop_path) if off_policy_stop_path is not None else "",
        }
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics_path", type=Path, required=True)
    parser.add_argument("--off_policy_action_path", type=Path)
    parser.add_argument("--off_policy_stop_path", type=Path)
    parser.add_argument("--output_path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze_mismatch_files(
        diagnostics_path=args.diagnostics_path,
        off_policy_action_path=args.off_policy_action_path,
        off_policy_stop_path=args.off_policy_stop_path,
    )
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
