from __future__ import annotations

import random
from typing import Dict, List, Sequence, Set, Tuple

from train_agent.rl.restricted_retrieval import FrozenVerifierProtocol, RestrictedRetrievalEnv, RestrictedRetrievalEpisode


DOCUMENT_AGGREGATIONS = (
    "full_document",
    "max",
    "top_2_mean",
    "top_k_weighted_mean",
    "logsumexp",
)


def rank_episode_documents(
    episode: RestrictedRetrievalEpisode,
    frozen_verifier: FrozenVerifierProtocol,
    *,
    doc_aggregation: str = "full_document",
    aggregation_top_k: int = 3,
) -> List[Dict[str, float]]:
    env = RestrictedRetrievalEnv(
        episode,
        frozen_verifier=frozen_verifier,
        doc_aggregation=doc_aggregation,
        aggregation_top_k=aggregation_top_k,
    )
    env.reset()
    result = env.step("search")
    ranking = result.info.get("verifier_ranking")
    if isinstance(ranking, list) and ranking:
        return [
            {
                "doc_id": str(item["doc_id"]),
                "score": float(item["score"]),
            }
            for item in ranking
        ]
    return [
        {
            "doc_id": str(doc_id),
            "score": 0.0,
        }
        for doc_id in episode.doc_pool
    ]


def _collect_episode_ranking_rows(
    episodes: Sequence[RestrictedRetrievalEpisode],
    frozen_verifier: FrozenVerifierProtocol,
    *,
    doc_aggregation: str,
    aggregation_top_k: int,
) -> Tuple[List[Dict[str, float]], int]:
    rows: List[Dict[str, float]] = []
    total_doc_pool = 0
    for episode in episodes:
        total_doc_pool += len(episode.doc_pool)
        gold_doc_ids: Set[str] = {item.doc_id for item in episode.gold_evidence}
        if not gold_doc_ids:
            continue
        ranking = rank_episode_documents(
            episode,
            frozen_verifier,
            doc_aggregation=doc_aggregation,
            aggregation_top_k=aggregation_top_k,
        )
        first_positive_rank = None
        for rank, item in enumerate(ranking, start=1):
            if str(item["doc_id"]) in gold_doc_ids:
                first_positive_rank = rank
                break
        rows.append(
            {
                "episode_id": str(episode.episode_id),
                "reciprocal_rank": 0.0 if first_positive_rank is None else 1.0 / first_positive_rank,
                "hit@1": 1.0 if first_positive_rank is not None and first_positive_rank <= 1 else 0.0,
                "hit@3": 1.0 if first_positive_rank is not None and first_positive_rank <= 3 else 0.0,
            }
        )
    return rows, total_doc_pool


def _bootstrap_mean_ci(values: Sequence[float], *, bootstrap_samples: int, seed: int) -> List[float]:
    if not values or bootstrap_samples <= 0:
        return []
    rng = random.Random(seed)
    sample_size = len(values)
    means: List[float] = []
    for _ in range(int(bootstrap_samples)):
        sampled = [values[rng.randrange(sample_size)] for _ in range(sample_size)]
        means.append(sum(sampled) / sample_size)
    means.sort()
    low_index = max(0, int(0.025 * bootstrap_samples) - 1)
    high_index = min(bootstrap_samples - 1, int(0.975 * bootstrap_samples) - 1)
    return [round(means[low_index], 6), round(means[high_index], 6)]


def evaluate_restricted_ranking_episodes(
    episodes: Sequence[RestrictedRetrievalEpisode],
    frozen_verifier: FrozenVerifierProtocol,
    *,
    doc_aggregation: str = "full_document",
    aggregation_top_k: int = 3,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 7,
) -> Dict[str, object]:
    rows, total_doc_pool = _collect_episode_ranking_rows(
        episodes,
        frozen_verifier,
        doc_aggregation=doc_aggregation,
        aggregation_top_k=aggregation_top_k,
    )
    positive_episodes = len(rows)
    divisor = max(positive_episodes, 1)
    episode_count = len(episodes)
    reciprocal_ranks = [float(row["reciprocal_rank"]) for row in rows]
    hits_at_1 = [float(row["hit@1"]) for row in rows]
    hits_at_3 = [float(row["hit@3"]) for row in rows]
    metrics: Dict[str, object] = {
        "episodes": episode_count,
        "positive_episodes": positive_episodes,
        "avg_doc_pool_size": round(total_doc_pool / max(episode_count, 1), 6),
        "doc_mrr": round(sum(reciprocal_ranks) / divisor, 6),
        "doc_recall@1": round(sum(hits_at_1) / divisor, 6),
        "doc_recall@3": round(sum(hits_at_3) / divisor, 6),
        "doc_hits@1": int(sum(hits_at_1)),
        "doc_hits@3": int(sum(hits_at_3)),
        "doc_misses@1": int(positive_episodes - sum(hits_at_1)),
        "doc_misses@3": int(positive_episodes - sum(hits_at_3)),
    }
    if bootstrap_samples > 0 and rows:
        metrics["bootstrap"] = {
            "samples": int(bootstrap_samples),
            "seed": int(bootstrap_seed),
            "doc_mrr_ci95": _bootstrap_mean_ci(reciprocal_ranks, bootstrap_samples=bootstrap_samples, seed=bootstrap_seed),
            "doc_recall@1_ci95": _bootstrap_mean_ci(hits_at_1, bootstrap_samples=bootstrap_samples, seed=bootstrap_seed + 1),
            "doc_recall@3_ci95": _bootstrap_mean_ci(hits_at_3, bootstrap_samples=bootstrap_samples, seed=bootstrap_seed + 2),
        }
    return metrics
