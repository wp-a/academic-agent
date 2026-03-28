from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Mapping

from datasets import load_dataset

from train_agent.data.adapters.scifact import (
    build_scifact_relevance_examples,
    build_scifact_stance_examples,
)
from train_agent.scripts.export_scifact_verifier_data import build_scifact_corpus_map


SPLIT_NAMES = ("train", "validation", "test")


def _write_examples(
    *,
    rows: Iterable[Mapping[str, object]],
    corpus_by_doc_id: Mapping[str, object],
    output_path: Path,
    build_examples: Callable[..., object],
    build_example_kwargs: Mapping[str, object] | None = None,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_counts: Counter = Counter()
    num_examples = 0
    build_example_kwargs = dict(build_example_kwargs or {})
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            examples = build_examples(row, corpus=corpus_by_doc_id, **build_example_kwargs)
            for example in examples:
                payload = example.to_dict()
                payload.pop("metadata", None)
                handle.write(json.dumps(payload, ensure_ascii=False))
                handle.write("\n")
                label_counts[payload["label"]] += 1
                num_examples += 1
    return {
        "output_path": str(output_path),
        "num_examples": num_examples,
        "label_counts": dict(sorted(label_counts.items())),
    }


def export_scifact_decomposed_split(
    *,
    rows: Iterable[Mapping[str, object]],
    corpus_by_doc_id: Mapping[str, object],
    relevance_output_path: Path,
    stance_output_path: Path,
    relevance_hard_negatives_per_positive: int = 0,
    relevance_random_negatives_per_positive: int = 0,
):
    relevance_kwargs = {
        "max_hard_negatives_per_positive": relevance_hard_negatives_per_positive,
        "max_random_negatives_per_positive": relevance_random_negatives_per_positive,
    }
    return {
        "relevance": _write_examples(
            rows=rows,
            corpus_by_doc_id=corpus_by_doc_id,
            output_path=relevance_output_path,
            build_examples=build_scifact_relevance_examples,
            build_example_kwargs=relevance_kwargs,
        ),
        "stance": _write_examples(
            rows=rows,
            corpus_by_doc_id=corpus_by_doc_id,
            output_path=stance_output_path,
            build_examples=build_scifact_stance_examples,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dataset_name", default="allenai/scifact")
    parser.add_argument("--claims_config", default="claims")
    parser.add_argument("--corpus_config", default="corpus")
    parser.add_argument("--relevance_hard_negatives_per_positive", type=int, default=0)
    parser.add_argument("--relevance_random_negatives_per_positive", type=int, default=0)
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    claims_dataset = load_dataset(
        args.dataset_name,
        args.claims_config,
        trust_remote_code=args.trust_remote_code,
    )
    corpus_dataset = load_dataset(
        args.dataset_name,
        args.corpus_config,
        trust_remote_code=args.trust_remote_code,
    )
    corpus_rows = corpus_dataset[list(corpus_dataset.keys())[0]] if hasattr(corpus_dataset, "keys") else corpus_dataset
    corpus_by_doc_id = build_scifact_corpus_map(corpus_rows)

    split_summaries = {}
    for split_name in SPLIT_NAMES:
        split_summaries[split_name] = export_scifact_decomposed_split(
            rows=claims_dataset[split_name],
            corpus_by_doc_id=corpus_by_doc_id,
            relevance_output_path=args.output_dir / f"scifact_relevance_{split_name}.jsonl",
            stance_output_path=args.output_dir / f"scifact_stance_{split_name}.jsonl",
            relevance_hard_negatives_per_positive=args.relevance_hard_negatives_per_positive,
            relevance_random_negatives_per_positive=args.relevance_random_negatives_per_positive,
        )

    stats_path = args.output_dir / "scifact_decomposed_verifier_stats.json"
    stats_payload = {
        "task": "decomposed_verifier",
        "stages": {
            "relevance": {
                "labels": ["NEUTRAL", "RELEVANT"],
                "hard_negatives_per_positive": args.relevance_hard_negatives_per_positive,
                "random_negatives_per_positive": args.relevance_random_negatives_per_positive,
            },
            "stance": {
                "labels": ["CONTRADICT", "SUPPORT"],
                "train_on_relevant_only": True,
            },
        },
        "splits": split_summaries,
        "stats_path": str(stats_path),
    }
    stats_path.write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats_payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
