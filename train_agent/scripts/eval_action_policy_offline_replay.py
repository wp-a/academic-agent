from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from datasets import load_dataset

from train_agent.data.adapters.scifact import build_scifact_restricted_episode
from train_agent.models.action_policy import FrozenActionPolicy
from train_agent.models.verifier import FrozenSequenceVerifier
from train_agent.rl.restricted_retrieval import RestrictedRetrievalEpisode, RestrictedRetrievalEnv
from train_agent.scripts.export_scifact_frozen_verifier_replay import WeakCoupledReplayPolicy, build_scifact_corpus_map


def evaluate_policy_on_episodes(
    episodes: Sequence[RestrictedRetrievalEpisode],
    *,
    verifier: FrozenSequenceVerifier,
    action_policy: FrozenActionPolicy,
    reference_policy: Optional[WeakCoupledReplayPolicy] = None,
    doc_aggregation: str = "full_document",
    aggregation_top_k: int = 3,
) -> Dict[str, object]:
    reference_policy = reference_policy or WeakCoupledReplayPolicy()
    episode_count = 0
    total_steps = 0
    agreement = 0
    predicted_stop = 0
    reference_stop = 0
    correct_stop = 0
    predicted_quote = 0
    quote_hits = 0
    successes = 0
    early_stops = 0
    action_counts: Counter = Counter()

    for episode in episodes:
        env = RestrictedRetrievalEnv(
            episode,
            frozen_verifier=verifier,
            doc_aggregation=doc_aggregation,
            aggregation_top_k=aggregation_top_k,
        )
        state = env.reset()
        done = False
        episode_count += 1
        while not done:
            reference_action = reference_policy.choose_action(state)
            predicted_action = action_policy.predict_action(state.to_text())
            total_steps += 1
            action_counts[predicted_action] += 1
            agreement += int(predicted_action == reference_action)
            if reference_action == "stop":
                reference_stop += 1
            if predicted_action == "stop":
                predicted_stop += 1
                correct_stop += int(reference_action == "stop")
            result = env.step(predicted_action)
            if predicted_action == "quote_evidence":
                predicted_quote += 1
                quote_hits += int(result.info.get("invalid_action") != "quote_evidence")
            if result.done:
                successes += int(bool(result.info.get("success_stop", False)))
                early_stops += int(bool(result.info.get("early_stop", False)))
            state = result.state
            done = result.done

    return {
        "episodes": episode_count,
        "num_actions": total_steps,
        "action_agreement": round(agreement / max(total_steps, 1), 6),
        "average_steps": round(total_steps / max(episode_count, 1), 6),
        "stop_precision": round(correct_stop / max(predicted_stop, 1), 6),
        "stop_recall": round(correct_stop / max(reference_stop, 1), 6),
        "quote_evidence_hit_rate": round(quote_hits / max(predicted_quote, 1), 6),
        "success_rate": round(successes / max(episode_count, 1), 6),
        "early_stop_rate": round(early_stops / max(episode_count, 1), 6),
        "action_distribution": {
            action: round(count / max(total_steps, 1), 6) for action, count in sorted(action_counts.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_model_dir", type=Path, required=True)
    parser.add_argument("--verifier_model_name_or_path", required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max_steps", type=int, default=4)
    parser.add_argument("--policy_max_length", type=int, default=768)
    parser.add_argument("--policy_batch_size", type=int, default=8)
    parser.add_argument("--verifier_max_length", type=int, default=384)
    parser.add_argument("--verifier_batch_size", type=int, default=8)
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

    episodes: List[RestrictedRetrievalEpisode] = []
    for row in claims:
        episode = build_scifact_restricted_episode(dict(row), corpus=corpus_map, max_steps=args.max_steps)
        if episode.gold_evidence:
            episodes.append(episode)

    verifier = FrozenSequenceVerifier(
        args.verifier_model_name_or_path,
        attn_implementation=args.attn_implementation,
        max_length=args.verifier_max_length,
        batch_size=args.verifier_batch_size,
    )
    action_policy = FrozenActionPolicy(
        args.policy_model_dir,
        max_length=args.policy_max_length,
        batch_size=args.policy_batch_size,
        attn_implementation=args.attn_implementation,
    )
    summary = evaluate_policy_on_episodes(
        episodes,
        verifier=verifier,
        action_policy=action_policy,
        doc_aggregation=args.doc_aggregation,
        aggregation_top_k=args.aggregation_top_k,
    )
    summary.update(
        {
            "split": args.split,
            "policy_model_dir": str(args.policy_model_dir),
            "verifier_model_name_or_path": args.verifier_model_name_or_path,
            "doc_aggregation": args.doc_aggregation,
            "aggregation_top_k": args.aggregation_top_k,
        }
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
