from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Dict, List

from datasets import load_dataset

from train_agent.rl import RestrictedRetrievalEnv, build_scifact_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=4)
    return parser.parse_args()


def choose_rule_action(state) -> str:
    if not state.revealed_docs or (
        not state.revealed_evidence and len(state.revealed_docs) < len(state.doc_pool)
    ):
        return "search"
    if state.revealed_evidence and not state.quoted_evidence:
        return "quote_evidence"
    return "stop"


def run_episode(row: Dict[str, object], max_steps: int) -> Dict[str, object]:
    episode = build_scifact_episode(row, max_steps=max_steps)
    env = RestrictedRetrievalEnv(episode)
    state = env.reset()
    trace: List[Dict[str, object]] = []
    total_reward = 0.0
    action_counter: Counter = Counter()
    invalid_quote_count = 0
    search_miss_count = 0
    early_stop = False
    success = False

    done = False
    while not done:
        action = choose_rule_action(state)
        action_counter[action] += 1
        result = env.step(action)
        total_reward += result.reward
        if result.info.get("invalid_action") == "quote_evidence":
            invalid_quote_count += 1
        if action == "search" and result.info.get("search_miss"):
            search_miss_count += 1
        if result.info.get("early_stop"):
            early_stop = True
        if result.done:
            success = bool(result.info.get("success_stop", False))
        trace.append(
            {
                "step_index": result.state.step_index,
                "action": action,
                "reward": result.reward,
                "done": result.done,
                "revealed_docs": result.state.revealed_docs.copy(),
                "quoted_docs": [item.doc_id for item in result.state.quoted_evidence],
                "info": result.info,
            }
        )
        state = result.state
        done = result.done

    return {
        "episode_id": episode.episode_id,
        "claim": episode.claim,
        "label_hint": episode.label_hint,
        "doc_pool_size": len(episode.doc_pool),
        "gold_evidence_docs": [item.doc_id for item in episode.gold_evidence],
        "total_reward": round(total_reward, 4),
        "num_steps": len(trace),
        "success": success,
        "early_stop": early_stop,
        "invalid_quote_count": invalid_quote_count,
        "search_miss_count": search_miss_count,
        "action_counts": dict(action_counter),
        "trace": trace,
        "final_state_text": state.to_text(),
    }


def summarize(episodes: List[Dict[str, object]]) -> Dict[str, object]:
    action_distribution: Counter = Counter()
    total_return = 0.0
    total_steps = 0
    successes = 0
    early_stops = 0
    invalid_quotes = 0
    total_searches = 0
    search_misses = 0

    for episode in episodes:
        total_return += episode["total_reward"]
        total_steps += episode["num_steps"]
        successes += int(episode["success"])
        early_stops += int(episode["early_stop"])
        invalid_quotes += episode["invalid_quote_count"]
        search_misses += episode["search_miss_count"]
        for action, count in episode["action_counts"].items():
            action_distribution[action] += count
            if action == "search":
                total_searches += count

    num_episodes = max(len(episodes), 1)
    total_actions = sum(action_distribution.values())
    return {
        "success_rate": round(successes / num_episodes, 6),
        "average_return": round(total_return / num_episodes, 6),
        "average_steps": round(total_steps / num_episodes, 6),
        "action_distribution": {
            action: round(count / total_actions, 6) for action, count in sorted(action_distribution.items())
        } if total_actions else {},
        "early_stop_rate": round(early_stops / num_episodes, 6),
        "invalid_quote_rate": round(invalid_quotes / num_episodes, 6),
        "search_miss_rate": round(search_misses / total_searches, 6) if total_searches else 0.0,
    }


def main() -> None:
    args = parse_args()
    dataset = load_dataset("allenai/scifact", "claims", trust_remote_code=True)[args.split]
    episodes = [run_episode(dataset[idx], max_steps=args.max_steps) for idx in range(min(args.limit, len(dataset)))]
    metrics = summarize(episodes)
    print(json.dumps({"split": args.split, "num_episodes": len(episodes), "metrics": metrics}, ensure_ascii=False))
    for item in episodes:
        print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
