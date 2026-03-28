from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Mapping

from datasets import load_dataset

from train_agent.data.adapters.scifact import build_scifact_verifier_examples


SPLIT_NAMES = ("train", "validation", "test")


def build_scifact_corpus_map(corpus_rows: Iterable[Mapping[str, object]]) -> Dict[str, Dict[str, object]]:
    corpus_map: Dict[str, Dict[str, object]] = {}
    for row in corpus_rows:
        doc_id = row.get("doc_id") or row.get("id")
        if doc_id is None:
            continue
        abstract = row.get("abstract") or row.get("sentences") or row.get("lines") or []
        if isinstance(abstract, str):
            abstract = [line for line in abstract.split("\n") if line.strip()]
        corpus_map[str(doc_id)] = {
            "sentences": [str(sentence) for sentence in abstract],
            "title": str(row.get("title") or ""),
        }
    return corpus_map


def export_scifact_verifier_split(*, rows: Iterable[Mapping[str, object]], corpus_by_doc_id: Mapping[str, object], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_counts: Counter = Counter()
    num_examples = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            examples = build_scifact_verifier_examples(row, corpus=corpus_by_doc_id)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dataset_name", default="allenai/scifact")
    parser.add_argument("--claims_config", default="claims")
    parser.add_argument("--corpus_config", default="corpus")
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
        split_rows = claims_dataset[split_name]
        output_path = args.output_dir / f"scifact_verifier_{split_name}.jsonl"
        split_summaries[split_name] = export_scifact_verifier_split(
            rows=split_rows,
            corpus_by_doc_id=corpus_by_doc_id,
            output_path=output_path,
        )

    stats_path = args.output_dir / "scifact_verifier_stats.json"
    stats_payload = {
        "task": "verifier_relevance_and_stance",
        "labels": ["CONTRADICT", "NEUTRAL", "SUPPORT"],
        "sufficiency_target": False,
        "splits": split_summaries,
        "stats_path": str(stats_path),
    }
    stats_path.write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats_payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
