from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from datasets import load_dataset

from train_agent.data.adapters.scifact import build_scifact_restricted_episode
from train_agent.models.verifier import FrozenSequenceVerifier
from train_agent.rl.restricted_retrieval import FrozenVerifierProtocol, RestrictedRetrievalEnv, RestrictedRetrievalEpisode


class WeakCoupledReplayPolicy:
    def choose_action(self, state) -> str:
        if state.quoted_evidence:
            return "stop"
        if len(state.revealed_evidence) > len(state.quoted_evidence):
            return "quote_evidence"
        if len(state.revealed_docs) < len(state.doc_pool):
            return "search"
        if state.revealed_evidence:
            return "quote_evidence"
        return "stop"


def replay_episode_to_action_examples(
    episode: RestrictedRetrievalEpisode,
    *,
    frozen_verifier: FrozenVerifierProtocol,
    policy: Optional[WeakCoupledReplayPolicy] = None,
    doc_aggregation: str = "full_document",
    aggregation_top_k: int = 3,
) -> List[Dict[str, object]]:
    policy = policy or WeakCoupledReplayPolicy()
    env = RestrictedRetrievalEnv(
        episode,
        frozen_verifier=frozen_verifier,
        doc_aggregation=doc_aggregation,
        aggregation_top_k=aggregation_top_k,
    )
    state = env.reset()
    records: List[Dict[str, object]] = []
    done = False
    step_id = 0
    while not done:
        state_text = state.to_text()
        action = policy.choose_action(state)
        result = env.step(action)
        records.append(
            {
                "trajectory_id": episode.episode_id,
                "step_id": step_id,
                "task": "next_action_classification",
                "text": state_text,
                "label": action,
                "label_text": json.dumps({"action_type": action}, ensure_ascii=False),
                "metadata": {
                    "reward": float(result.reward),
                    "done": bool(result.done),
                    "success_stop": bool(result.info.get("success_stop", False)),
                    "early_stop": bool(result.info.get("early_stop", False)),
                    "termination_reason": str(result.info.get("termination_reason", "")),
                    "verifier_top_doc": str(result.info.get("verifier_top_doc", "")),
                    "revealed_docs": list(result.state.revealed_docs),
                    "quoted_docs": [item.doc_id for item in result.state.quoted_evidence],
                },
            }
        )
        state = result.state
        done = result.done
        step_id += 1
    return records


def summarize_replay_records(episode_records: Sequence[Sequence[Dict[str, object]]]) -> Dict[str, object]:
    episodes = len(episode_records)
    total_steps = 0
    successes = 0
    early_stops = 0
    top_doc_hits = 0
    top_doc_decisions = 0
    action_counts: Counter = Counter()
    for records in episode_records:
        if not records:
            continue
        total_steps += len(records)
        last_metadata = records[-1].get("metadata", {})
        successes += int(bool(last_metadata.get("success_stop", False)))
        early_stops += int(bool(last_metadata.get("early_stop", False)))
        for record in records:
            action = str(record.get("label", ""))
            if action:
                action_counts[action] += 1
            metadata = record.get("metadata", {})
            if action == "search":
                top_doc = str(metadata.get("verifier_top_doc", "") or "")
                top_doc_decisions += 1
                quoted_docs = metadata.get("quoted_docs", [])
                revealed_docs = metadata.get("revealed_docs", [])
                if top_doc and (top_doc in revealed_docs or top_doc in quoted_docs):
                    top_doc_hits += 1
    total_actions = sum(action_counts.values())
    return {
        "episodes": episodes,
        "average_steps": round(total_steps / max(episodes, 1), 6),
        "success_rate": round(successes / max(episodes, 1), 6),
        "early_stop_rate": round(early_stops / max(episodes, 1), 6),
        "action_distribution": {
            action: round(count / total_actions, 6) for action, count in sorted(action_counts.items())
        } if total_actions else {},
        "verifier_top_doc_hit_rate": round(top_doc_hits / max(top_doc_decisions, 1), 6),
    }


def build_scifact_corpus_map(corpus_rows) -> Dict[str, object]:
    corpus_map: Dict[str, object] = {}
    for row in corpus_rows:
        doc_id = row.get("doc_id") or row.get("id")
        if doc_id is None:
            continue
        corpus_map[str(doc_id)] = dict(row)
    return corpus_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--summary_path", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max_examples", type=int, default=0)
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
    claims = load_dataset("allenai/scifact", "claims", split=args.split, trust_remote_code=args.trust_remote_code)
    corpus_dataset = load_dataset("allenai/scifact", "corpus", trust_remote_code=args.trust_remote_code)
    corpus = corpus_dataset[list(corpus_dataset.keys())[0]]
    corpus_map = build_scifact_corpus_map(corpus)

    verifier = FrozenSequenceVerifier(
        args.model_name_or_path,
        attn_implementation=args.attn_implementation,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    policy = WeakCoupledReplayPolicy()
    episode_records: List[List[Dict[str, object]]] = []
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in claims:
            episode = build_scifact_restricted_episode(dict(row), corpus=corpus_map, max_steps=args.max_steps)
            if not episode.gold_evidence:
                continue
            records = replay_episode_to_action_examples(
                episode,
                frozen_verifier=verifier,
                policy=policy,
                doc_aggregation=args.doc_aggregation,
                aggregation_top_k=args.aggregation_top_k,
            )
            episode_records.append(records)
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
            if args.max_examples and len(episode_records) >= args.max_examples:
                break

    summary = summarize_replay_records(episode_records)
    summary.update(
        {
            "split": args.split,
            "model_name_or_path": args.model_name_or_path,
            "doc_aggregation": args.doc_aggregation,
            "aggregation_top_k": args.aggregation_top_k,
            "num_examples": sum(len(records) for records in episode_records),
            "output_jsonl": str(args.output_jsonl),
        }
    )
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
