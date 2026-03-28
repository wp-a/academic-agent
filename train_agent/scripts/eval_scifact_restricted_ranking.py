from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from datasets import load_dataset

from train_agent.data.adapters.scifact import build_scifact_restricted_episode
from train_agent.eval.restricted_ranking import evaluate_restricted_ranking_episodes
from train_agent.models.verifier import FrozenSequenceVerifier


def build_scifact_corpus_map(corpus_rows) -> Dict[str, object]:
    corpus_map: Dict[str, object] = {}
    for row in corpus_rows:
        doc_id = row.get('doc_id') or row.get('id')
        if doc_id is None:
            continue
        corpus_map[str(doc_id)] = dict(row)
    return corpus_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--output_path', type=Path, required=True)
    parser.add_argument('--split', default='validation')
    parser.add_argument('--max_examples', type=int, default=0)
    parser.add_argument('--max_steps', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--attn_implementation', default='sdpa')
    parser.add_argument('--trust_remote_code', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    claims = load_dataset('allenai/scifact', 'claims', split=args.split, trust_remote_code=args.trust_remote_code)
    corpus_dataset = load_dataset('allenai/scifact', 'corpus', trust_remote_code=args.trust_remote_code)
    corpus = corpus_dataset[list(corpus_dataset.keys())[0]]
    corpus_map = build_scifact_corpus_map(corpus)

    episodes = []
    for row in claims:
        episode = build_scifact_restricted_episode(dict(row), corpus=corpus_map, max_steps=args.max_steps)
        if not episode.gold_evidence:
            continue
        episodes.append(episode)
        if args.max_examples and len(episodes) >= args.max_examples:
            break

    verifier = FrozenSequenceVerifier(
        args.model_name_or_path,
        attn_implementation=args.attn_implementation,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    metrics = evaluate_restricted_ranking_episodes(episodes, verifier)
    metrics.update(
        {
            'split': args.split,
            'model_name_or_path': args.model_name_or_path,
            'max_examples': args.max_examples,
        }
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == '__main__':
    main()
