from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datasets import load_dataset

from train_agent.data.adapters.scifact import build_scifact_restricted_episode
from train_agent.data.adapters.scifact_hard import augment_episode_with_lexical_distractors
from train_agent.models.action_policy import FrozenActionPolicy
from train_agent.models.stop_policy import FrozenStopPolicy
from train_agent.models.verifier import FrozenSequenceVerifier
from train_agent.rl.restricted_retrieval import RestrictedEvidence, RestrictedRetrievalEpisode, RestrictedRetrievalEnv
from train_agent.scripts.export_scifact_frozen_verifier_replay import WeakCoupledReplayPolicy, build_scifact_corpus_map
from train_agent.scripts.export_scifact_hard_replay_data import ConservativeReplayPolicy
from train_agent.scripts.export_scifact_stop_policy_data import convert_action_record_to_stop_record


def build_corpus_text_by_doc(corpus_map: Dict[str, object]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for doc_id, payload in corpus_map.items():
        abstract = payload.get("abstract") or payload.get("sentences") or payload.get("lines") or []
        result[str(doc_id)] = " ".join(str(sentence) for sentence in abstract)
    return result


def build_corpus_sentences_by_doc(corpus_map: Dict[str, object]) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for doc_id, payload in corpus_map.items():
        abstract = payload.get("abstract") or payload.get("sentences") or payload.get("lines") or []
        result[str(doc_id)] = [str(sentence) for sentence in abstract]
    return result


def maybe_augment_episode_for_hard_eval(
    episode: RestrictedRetrievalEpisode,
    *,
    corpus_text_by_doc: Dict[str, str],
    corpus_sentences_by_doc: Dict[str, List[str]],
    num_distractor_docs: int,
) -> RestrictedRetrievalEpisode:
    if num_distractor_docs <= 0:
        return episode
    return augment_episode_with_lexical_distractors(
        episode=episode,
        corpus_text_by_doc=corpus_text_by_doc,
        corpus_sentences_by_doc=corpus_sentences_by_doc,
        num_distractor_docs=num_distractor_docs,
    )


def choose_action_with_optional_stop_suppression(action_policy, state_text: str) -> Tuple[str, bool]:
    if hasattr(action_policy, "predict_logits") and hasattr(action_policy, "label_names"):
        logits = action_policy.predict_logits([state_text])[0]
        label_names = list(action_policy.label_names)
        ranked_indices = sorted(range(len(logits)), key=lambda idx: logits[idx], reverse=True)
        top_label = label_names[ranked_indices[0]]
        if top_label != "stop":
            return top_label, False
        for idx in ranked_indices[1:]:
            label = label_names[idx]
            if label != "stop":
                return label, True
        raise ValueError("Action policy has no non-stop label to fall back to.")

    predicted_action = action_policy.predict_action(state_text)
    if predicted_action == "stop":
        raise ValueError("Action policy predicted stop but does not expose logits for non-stop fallback.")
    return predicted_action, False


def _build_reference_policy(
    episode: RestrictedRetrievalEpisode,
    *,
    reference_policy_type: str,
    post_quote_search_budget: int,
):
    if reference_policy_type == "conservative":
        return ConservativeReplayPolicy(
            gold_doc_ids={item.doc_id for item in episode.gold_evidence},
            post_quote_search_budget=post_quote_search_budget,
        )
    if reference_policy_type == "weak":
        return WeakCoupledReplayPolicy()
    raise ValueError(f"Unsupported reference_policy_type: {reference_policy_type}")


def _serialize_evidence(items: Sequence[RestrictedEvidence]) -> List[Dict[str, object]]:
    return [
        {
            "doc_id": item.doc_id,
            "sentence_ids": list(item.sentence_ids),
            "stance": item.stance,
            "snippet": item.snippet,
        }
        for item in items
    ]


def _serialize_verifier_scores(scores: Dict[str, float]) -> List[Dict[str, object]]:
    return [
        {"doc_id": doc_id, "score": round(float(score), 6)}
        for doc_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]


def _build_episode_diagnostics_record(
    episode: RestrictedRetrievalEpisode,
    *,
    reference_policy_type: str,
    post_quote_search_budget: int,
    mismatch_step_indices: List[int],
    step_records: List[Dict[str, Any]],
) -> Dict[str, object]:
    return {
        "episode_id": episode.episode_id,
        "claim": episode.claim,
        "label_hint": episode.label_hint,
        "doc_pool": list(episode.doc_pool),
        "gold_evidence": _serialize_evidence(episode.gold_evidence),
        "reference_policy_type": reference_policy_type,
        "post_quote_search_budget": post_quote_search_budget,
        "num_steps": len(step_records),
        "num_mismatches": len(mismatch_step_indices),
        "mismatch_step_indices": mismatch_step_indices,
        "steps": step_records,
    }


def _build_off_policy_action_record(
    episode: RestrictedRetrievalEpisode,
    *,
    step_index: int,
    state_text: str,
    student_action: str,
    reference_action: str,
    is_first_off_policy_step: bool,
    reference_policy_type: str,
    post_quote_search_budget: int,
    used_stop_policy: bool,
    stop_policy_should_stop: Optional[bool],
    suppressed_stop: bool,
) -> Dict[str, object]:
    return {
        "trajectory_id": episode.episode_id,
        "step_id": int(step_index),
        "task": "next_action_classification",
        "text": state_text,
        "label": reference_action,
        "label_text": json.dumps({"action_type": reference_action}, ensure_ascii=False),
        "metadata": {
            "episode_id": episode.episode_id,
            "student_action": student_action,
            "reference_action": reference_action,
            "is_first_off_policy_step": is_first_off_policy_step,
            "reference_policy_type": reference_policy_type,
            "post_quote_search_budget": post_quote_search_budget,
            "used_stop_policy": used_stop_policy,
            "stop_policy_should_stop": stop_policy_should_stop,
            "suppressed_stop": suppressed_stop,
        },
    }


def _build_off_policy_stop_record(action_record: Dict[str, object]) -> Dict[str, object]:
    stop_record = convert_action_record_to_stop_record(action_record)
    metadata = action_record.get("metadata")
    if metadata:
        stop_record["metadata"] = dict(metadata)
    return stop_record


def evaluate_policy_on_episodes(
    episodes: Sequence[RestrictedRetrievalEpisode],
    *,
    verifier: FrozenSequenceVerifier,
    action_policy: FrozenActionPolicy,
    stop_policy: Optional[FrozenStopPolicy] = None,
    reference_policy_type: str = "weak",
    post_quote_search_budget: int = 1,
    doc_aggregation: str = "full_document",
    aggregation_top_k: int = 3,
    diagnostics_output_path: Optional[Path] = None,
    off_policy_action_output_path: Optional[Path] = None,
    off_policy_stop_output_path: Optional[Path] = None,
) -> Dict[str, object]:
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
    stop_policy_yes_count = 0
    stop_policy_no_count = 0
    suppressed_stop_count = 0
    mismatch_episode_count = 0
    mismatch_step_count = 0
    off_policy_episode_count = 0
    action_counts: Counter = Counter()
    mismatch_episode_records: List[Dict[str, object]] = []
    off_policy_action_records: List[Dict[str, object]] = []
    off_policy_stop_records: List[Dict[str, object]] = []
    capture_off_policy = off_policy_action_output_path is not None or off_policy_stop_output_path is not None

    for episode in episodes:
        reference_policy = _build_reference_policy(
            episode,
            reference_policy_type=reference_policy_type,
            post_quote_search_budget=post_quote_search_budget,
        )
        env = RestrictedRetrievalEnv(
            episode,
            frozen_verifier=verifier,
            doc_aggregation=doc_aggregation,
            aggregation_top_k=aggregation_top_k,
        )
        state = env.reset()
        done = False
        episode_count += 1
        episode_step_records: List[Dict[str, Any]] = []
        episode_mismatch_step_indices: List[int] = []
        off_policy_started = False
        off_policy_counted = False
        while not done:
            state_text = state.to_text()
            reference_action = reference_policy.choose_action(state)
            stop_policy_decision: Optional[bool] = None
            suppressed_stop = False
            if stop_policy is not None:
                stop_policy_decision = bool(stop_policy.predict_should_stop(state_text))
                if stop_policy_decision:
                    predicted_action = "stop"
                    stop_policy_yes_count += 1
                else:
                    predicted_action, suppressed_stop = choose_action_with_optional_stop_suppression(action_policy, state_text)
                    stop_policy_no_count += 1
                    suppressed_stop_count += int(suppressed_stop)
            else:
                predicted_action = action_policy.predict_action(state_text)

            is_first_off_policy_step = False
            if predicted_action != reference_action:
                episode_mismatch_step_indices.append(state.step_index)
                if not off_policy_started:
                    off_policy_started = True
                    is_first_off_policy_step = True

            if capture_off_policy and off_policy_started:
                if not off_policy_counted:
                    off_policy_episode_count += 1
                    off_policy_counted = True
                action_record = _build_off_policy_action_record(
                    episode,
                    step_index=state.step_index,
                    state_text=state_text,
                    student_action=predicted_action,
                    reference_action=reference_action,
                    is_first_off_policy_step=is_first_off_policy_step,
                    reference_policy_type=reference_policy_type,
                    post_quote_search_budget=post_quote_search_budget,
                    used_stop_policy=bool(stop_policy is not None),
                    stop_policy_should_stop=stop_policy_decision,
                    suppressed_stop=suppressed_stop,
                )
                if off_policy_action_output_path is not None:
                    off_policy_action_records.append(action_record)
                if off_policy_stop_output_path is not None:
                    off_policy_stop_records.append(_build_off_policy_stop_record(action_record))

            step_record: Dict[str, Any] = {
                "step_index": state.step_index,
                "state_text": state_text,
                "reference_action": reference_action,
                "predicted_action": predicted_action,
                "action_match": predicted_action == reference_action,
                "stop_policy_should_stop": stop_policy_decision,
                "suppressed_stop": suppressed_stop,
                "revealed_docs": list(state.revealed_docs),
                "revealed_evidence": _serialize_evidence(state.revealed_evidence),
                "quoted_evidence": _serialize_evidence(state.quoted_evidence),
                "verifier_scores": _serialize_verifier_scores(state.verifier_scores),
            }

            total_steps += 1
            action_counts[predicted_action] += 1
            agreement += int(predicted_action == reference_action)
            if reference_action == "stop":
                reference_stop += 1
            if predicted_action == "stop":
                predicted_stop += 1
                correct_stop += int(reference_action == "stop")
            result = env.step(predicted_action)
            step_record.update(
                {
                    "reward": float(result.reward),
                    "done": bool(result.done),
                    "info": dict(result.info),
                    "next_revealed_docs": list(result.state.revealed_docs),
                    "next_revealed_evidence": _serialize_evidence(result.state.revealed_evidence),
                    "next_quoted_evidence": _serialize_evidence(result.state.quoted_evidence),
                }
            )
            episode_step_records.append(step_record)
            if predicted_action == "quote_evidence":
                predicted_quote += 1
                quote_hits += int(result.info.get("invalid_action") != "quote_evidence")
            if result.done:
                successes += int(bool(result.info.get("success_stop", False)))
                early_stops += int(bool(result.info.get("early_stop", False)))
            state = result.state
            done = result.done

        if episode_mismatch_step_indices:
            mismatch_episode_count += 1
            mismatch_step_count += len(episode_mismatch_step_indices)
            mismatch_episode_records.append(
                _build_episode_diagnostics_record(
                    episode,
                    reference_policy_type=reference_policy_type,
                    post_quote_search_budget=post_quote_search_budget,
                    mismatch_step_indices=episode_mismatch_step_indices,
                    step_records=episode_step_records,
                )
            )

    if diagnostics_output_path is not None:
        diagnostics_output_path.parent.mkdir(parents=True, exist_ok=True)
        with diagnostics_output_path.open("w", encoding="utf-8") as handle:
            for record in mismatch_episode_records:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")

    if off_policy_action_output_path is not None:
        off_policy_action_output_path.parent.mkdir(parents=True, exist_ok=True)
        with off_policy_action_output_path.open("w", encoding="utf-8") as handle:
            for record in off_policy_action_records:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")

    if off_policy_stop_output_path is not None:
        off_policy_stop_output_path.parent.mkdir(parents=True, exist_ok=True)
        with off_policy_stop_output_path.open("w", encoding="utf-8") as handle:
            for record in off_policy_stop_records:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")

    summary = {
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
        "used_stop_policy": bool(stop_policy is not None),
        "stop_policy_yes_count": stop_policy_yes_count,
        "stop_policy_no_count": stop_policy_no_count,
        "suppressed_stop_count": suppressed_stop_count,
        "reference_policy_type": reference_policy_type,
        "post_quote_search_budget": post_quote_search_budget,
        "mismatch_episode_count": mismatch_episode_count,
        "mismatch_step_count": mismatch_step_count,
        "diagnostics_output_path": str(diagnostics_output_path) if diagnostics_output_path is not None else "",
        "off_policy_episode_count": off_policy_episode_count,
        "off_policy_action_examples": len(off_policy_action_records),
        "off_policy_stop_examples": len(off_policy_stop_records),
        "off_policy_action_output_path": str(off_policy_action_output_path) if off_policy_action_output_path is not None else "",
        "off_policy_stop_output_path": str(off_policy_stop_output_path) if off_policy_stop_output_path is not None else "",
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_model_dir", type=Path, required=True)
    parser.add_argument("--stop_model_dir", type=Path)
    parser.add_argument("--verifier_model_name_or_path", required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--diagnostics_output_path", type=Path)
    parser.add_argument("--off_policy_action_output_path", type=Path)
    parser.add_argument("--off_policy_stop_output_path", type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max_steps", type=int, default=4)
    parser.add_argument("--policy_max_length", type=int, default=768)
    parser.add_argument("--policy_batch_size", type=int, default=8)
    parser.add_argument("--stop_max_length", type=int, default=512)
    parser.add_argument("--stop_batch_size", type=int, default=8)
    parser.add_argument("--verifier_max_length", type=int, default=384)
    parser.add_argument("--verifier_batch_size", type=int, default=8)
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--doc_aggregation", default="full_document")
    parser.add_argument("--aggregation_top_k", type=int, default=3)
    parser.add_argument("--num_distractor_docs", type=int, default=0)
    parser.add_argument("--reference_policy_type", choices=["weak", "conservative"], default="weak")
    parser.add_argument("--post_quote_search_budget", type=int, default=1)
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    claims = load_dataset("allenai/scifact", "claims", split=args.split, trust_remote_code=args.trust_remote_code)
    corpus_dataset = load_dataset("allenai/scifact", "corpus", trust_remote_code=args.trust_remote_code)
    corpus = corpus_dataset[list(corpus_dataset.keys())[0]]
    corpus_map = build_scifact_corpus_map(corpus)

    corpus_text_by_doc = build_corpus_text_by_doc(corpus_map) if args.num_distractor_docs > 0 else {}
    corpus_sentences_by_doc = build_corpus_sentences_by_doc(corpus_map) if args.num_distractor_docs > 0 else {}

    episodes: List[RestrictedRetrievalEpisode] = []
    for row in claims:
        episode = build_scifact_restricted_episode(dict(row), corpus=corpus_map, max_steps=args.max_steps)
        if not episode.gold_evidence:
            continue
        episode = maybe_augment_episode_for_hard_eval(
            episode,
            corpus_text_by_doc=corpus_text_by_doc,
            corpus_sentences_by_doc=corpus_sentences_by_doc,
            num_distractor_docs=args.num_distractor_docs,
        )
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
    stop_policy = None
    if args.stop_model_dir is not None:
        stop_policy = FrozenStopPolicy(
            args.stop_model_dir,
            max_length=args.stop_max_length,
            batch_size=args.stop_batch_size,
            attn_implementation=args.attn_implementation,
        )
    summary = evaluate_policy_on_episodes(
        episodes,
        verifier=verifier,
        action_policy=action_policy,
        stop_policy=stop_policy,
        reference_policy_type=args.reference_policy_type,
        post_quote_search_budget=args.post_quote_search_budget,
        doc_aggregation=args.doc_aggregation,
        aggregation_top_k=args.aggregation_top_k,
        diagnostics_output_path=args.diagnostics_output_path,
        off_policy_action_output_path=args.off_policy_action_output_path,
        off_policy_stop_output_path=args.off_policy_stop_output_path,
    )
    summary.update(
        {
            "split": args.split,
            "policy_model_dir": str(args.policy_model_dir),
            "stop_model_dir": str(args.stop_model_dir) if args.stop_model_dir is not None else "",
            "verifier_model_name_or_path": args.verifier_model_name_or_path,
            "doc_aggregation": args.doc_aggregation,
            "aggregation_top_k": args.aggregation_top_k,
            "num_distractor_docs": args.num_distractor_docs,
            "reference_policy_type": args.reference_policy_type,
            "post_quote_search_budget": args.post_quote_search_budget,
            "diagnostics_output_path": str(args.diagnostics_output_path) if args.diagnostics_output_path is not None else "",
            "off_policy_action_output_path": str(args.off_policy_action_output_path) if args.off_policy_action_output_path is not None else "",
            "off_policy_stop_output_path": str(args.off_policy_stop_output_path) if args.off_policy_stop_output_path is not None else "",
        }
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
