from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set

from train_agent.rl.restricted_retrieval import FrozenVerifierProtocol, RestrictedRetrievalEnv, RestrictedRetrievalEpisode


def rank_episode_documents(
    episode: RestrictedRetrievalEpisode,
    frozen_verifier: FrozenVerifierProtocol,
) -> List[Dict[str, float]]:
    env = RestrictedRetrievalEnv(episode, frozen_verifier=frozen_verifier)
    env.reset()
    result = env.step('search')
    ranking = result.info.get('verifier_ranking')
    if isinstance(ranking, list) and ranking:
        return [
            {
                'doc_id': str(item['doc_id']),
                'score': float(item['score']),
            }
            for item in ranking
        ]
    return [
        {
            'doc_id': str(doc_id),
            'score': 0.0,
        }
        for doc_id in episode.doc_pool
    ]


def evaluate_restricted_ranking_episodes(
    episodes: Sequence[RestrictedRetrievalEpisode],
    frozen_verifier: FrozenVerifierProtocol,
) -> Dict[str, float]:
    mrr_total = 0.0
    recall_at_1 = 0.0
    recall_at_3 = 0.0
    positive_episodes = 0
    total_doc_pool = 0

    for episode in episodes:
        total_doc_pool += len(episode.doc_pool)
        gold_doc_ids: Set[str] = {item.doc_id for item in episode.gold_evidence}
        if not gold_doc_ids:
            continue
        positive_episodes += 1
        ranking = rank_episode_documents(episode, frozen_verifier)
        first_positive_rank = None
        for rank, item in enumerate(ranking, start=1):
            if str(item['doc_id']) in gold_doc_ids:
                first_positive_rank = rank
                break
        if first_positive_rank is None:
            continue
        mrr_total += 1.0 / first_positive_rank
        recall_at_1 += 1.0 if first_positive_rank <= 1 else 0.0
        recall_at_3 += 1.0 if first_positive_rank <= 3 else 0.0

    divisor = max(positive_episodes, 1)
    episode_count = len(episodes)
    return {
        'episodes': episode_count,
        'positive_episodes': positive_episodes,
        'avg_doc_pool_size': round(total_doc_pool / max(episode_count, 1), 6),
        'doc_mrr': round(mrr_total / divisor, 6),
        'doc_recall@1': round(recall_at_1 / divisor, 6),
        'doc_recall@3': round(recall_at_3 / divisor, 6),
    }
