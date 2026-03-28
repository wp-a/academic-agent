from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

from datasets import load_dataset

from train_agent.data.adapters.scifact import build_scifact_restricted_episode
from train_agent.models.verifier import FrozenSequenceVerifier
from train_agent.scripts.export_scifact_frozen_verifier_replay import (
    WeakCoupledReplayPolicy,
    build_scifact_corpus_map,
    replay_episode_to_action_examples,
)


ACTION_LABELS = ["quote_evidence", "search", "stop"]


def sanitize_action_record(record: Dict[str, object]) -> Dict[str, object]:
    return {
        "trajectory_id": str(record["trajectory_id"]),
        "step_id": int(record["step_id"]),
        "task": str(record["task"]),
        "text": str(record["text"]),
        "label": str(record["label"]),
        "label_text": str(record["label_text"]),
    }


def summarize_action_records(records: Sequence[Dict[str, object]], episodes: int) -> Dict[str, object]:
    action_counts: Counter = Counter(str(record["label"]) for record in records)
    total_actions = sum(action_counts.values())
    return {
        "episodes": episodes,
        "num_examples": total_actions,
        "average_steps": round(total_actions / max(episodes, 1), 6),
        "label_names": ACTION_LABELS,
        "action_counts": {label: int(action_counts.get(label, 0)) for label in ACTION_LABELS},
        "action_distribution": {
            label: round(action_counts.get(label, 0) / max(total_actions, 1), 6) for label in ACTION_LABELS
        },
    }


def export_split(
    *,
    split: str,
    verifier: FrozenSequenceVerifier,
    corpus_map: Dict[str, object],
    output_path: Path,
    max_steps: int,
    doc_aggregation: str,
    aggregation_top_k: int,
    trust_remote_code: bool,
) -> Dict[str, object]:
    claims = load_dataset("allenai/scifact", "claims", split=split, trust_remote_code=trust_remote_code)
    policy = WeakCoupledReplayPolicy()
    clean_records: List[Dict[str, object]] = []
    episode_count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in claims:
            episode = build_scifact_restricted_episode(dict(row), corpus=corpus_map, max_steps=max_steps)
            if not episode.gold_evidence:
                continue
            records = replay_episode_to_action_examples(
                episode,
                frozen_verifier=verifier,
                policy=policy,
                doc_aggregation=doc_aggregation,
                aggregation_top_k=aggregation_top_k,
            )
            clean = [sanitize_action_record(record) for record in records]
            clean_records.extend(clean)
            for record in clean:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
            episode_count += 1
    summary = summarize_action_records(clean_records, episode_count)
    summary.update(
        {
            "split": split,
            "output_jsonl": str(output_path),
            "doc_aggregation": doc_aggregation,
            "aggregation_top_k": aggregation_top_k,
            "max_steps": max_steps,
        }
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verifier_model_name_or_path", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--doc_aggregation", default="full_document")
    parser.add_argument("--aggregation_top_k", type=int, default=3)
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_dataset = load_dataset("allenai/scifact", "corpus", trust_remote_code=args.trust_remote_code)
    corpus = corpus_dataset[list(corpus_dataset.keys())[0]]
    corpus_map = build_scifact_corpus_map(corpus)
    verifier = FrozenSequenceVerifier(
        args.verifier_model_name_or_path,
        attn_implementation=args.attn_implementation,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: Dict[str, Dict[str, object]] = {}
    for split in ["train", "validation"]:
        output_path = args.output_dir / f"scifact_action_policy_{split}.jsonl"
        summary = export_split(
            split=split,
            verifier=verifier,
            corpus_map=corpus_map,
            output_path=output_path,
            max_steps=args.max_steps,
            doc_aggregation=args.doc_aggregation,
            aggregation_top_k=args.aggregation_top_k,
            trust_remote_code=args.trust_remote_code,
        )
        summaries[split] = summary
        (args.output_dir / f"scifact_action_policy_{split}_stats.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    export_summary = {
        "verifier_model_name_or_path": args.verifier_model_name_or_path,
        "doc_aggregation": args.doc_aggregation,
        "aggregation_top_k": args.aggregation_top_k,
        "label_names": ACTION_LABELS,
        "splits": summaries,
    }
    (args.output_dir / "export_summary.json").write_text(
        json.dumps(export_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(export_summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
