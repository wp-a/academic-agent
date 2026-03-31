from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

from datasets import load_dataset

from train_agent.data.adapters.scifact import build_scifact_restricted_episode
from train_agent.data.adapters.scifact_hard import augment_episode_with_lexical_distractors
from train_agent.models.verifier import FrozenSequenceVerifier
from train_agent.scripts.export_scifact_action_policy_data import sanitize_action_record, summarize_action_records
from train_agent.scripts.export_scifact_frozen_verifier_replay import build_scifact_corpus_map, replay_episode_to_action_examples
from train_agent.scripts.export_scifact_stop_policy_data import convert_action_record_to_stop_record, summarize_stop_records


class ConservativeReplayPolicy:
    def __init__(self, *, gold_doc_ids: set[str], post_quote_search_budget: int = 1):
        self.gold_doc_ids = set(gold_doc_ids)
        self.post_quote_search_budget = max(0, int(post_quote_search_budget))

    def choose_action(self, state) -> str:
        revealed_gold = {item.doc_id for item in state.revealed_evidence if item.doc_id in self.gold_doc_ids}
        quoted_gold = {item.doc_id for item in state.quoted_evidence if item.doc_id in self.gold_doc_ids}
        if revealed_gold - quoted_gold:
            return "quote_evidence"
        target_revealed_docs = min(len(state.doc_pool), max(1, len(self.gold_doc_ids)) + self.post_quote_search_budget)
        if quoted_gold and len(state.revealed_docs) < target_revealed_docs and len(state.revealed_docs) < len(state.doc_pool):
            return "search"
        if quoted_gold:
            return "stop"
        if len(state.revealed_docs) < len(state.doc_pool):
            return "search"
        return "stop"


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


def export_split(
    *,
    split: str,
    verifier: FrozenSequenceVerifier,
    corpus_map: Dict[str, object],
    corpus_text_by_doc: Dict[str, str],
    corpus_sentences_by_doc: Dict[str, List[str]],
    output_dir: Path,
    max_steps: int,
    doc_aggregation: str,
    aggregation_top_k: int,
    trust_remote_code: bool,
    num_distractor_docs: int,
    post_quote_search_budget: int,
) -> Dict[str, object]:
    claims = load_dataset("allenai/scifact", "claims", split=split, trust_remote_code=trust_remote_code)
    action_records: List[Dict[str, object]] = []
    stop_records: List[Dict[str, object]] = []
    episode_count = 0
    action_output_path = output_dir / f"scifact_hard_action_policy_{split}.jsonl"
    stop_output_path = output_dir / f"scifact_hard_stop_policy_{split}.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    with action_output_path.open("w", encoding="utf-8") as action_handle, stop_output_path.open("w", encoding="utf-8") as stop_handle:
        for row in claims:
            base_episode = build_scifact_restricted_episode(dict(row), corpus=corpus_map, max_steps=max_steps)
            if not base_episode.gold_evidence:
                continue
            episode = augment_episode_with_lexical_distractors(
                episode=base_episode,
                corpus_text_by_doc=corpus_text_by_doc,
                corpus_sentences_by_doc=corpus_sentences_by_doc,
                num_distractor_docs=num_distractor_docs,
            )
            policy = ConservativeReplayPolicy(
                gold_doc_ids={item.doc_id for item in episode.gold_evidence},
                post_quote_search_budget=post_quote_search_budget,
            )
            raw_action_records = replay_episode_to_action_examples(
                episode,
                frozen_verifier=verifier,
                policy=policy,
                doc_aggregation=doc_aggregation,
                aggregation_top_k=aggregation_top_k,
            )
            clean_action_records = [sanitize_action_record(record) for record in raw_action_records]
            action_records.extend(clean_action_records)
            for record in clean_action_records:
                action_handle.write(json.dumps(record, ensure_ascii=False))
                action_handle.write("\n")

            clean_stop_records = [convert_action_record_to_stop_record(record) for record in raw_action_records]
            stop_records.extend(clean_stop_records)
            for record in clean_stop_records:
                stop_handle.write(json.dumps(record, ensure_ascii=False))
                stop_handle.write("\n")
            episode_count += 1

    action_summary = summarize_action_records(action_records, episode_count)
    stop_summary = summarize_stop_records(stop_records, episode_count)
    action_summary.update(
        {
            "split": split,
            "output_jsonl": str(action_output_path),
            "doc_aggregation": doc_aggregation,
            "aggregation_top_k": aggregation_top_k,
            "max_steps": max_steps,
            "num_distractor_docs": num_distractor_docs,
            "post_quote_search_budget": post_quote_search_budget,
        }
    )
    stop_summary.update(
        {
            "split": split,
            "output_jsonl": str(stop_output_path),
            "doc_aggregation": doc_aggregation,
            "aggregation_top_k": aggregation_top_k,
            "max_steps": max_steps,
            "num_distractor_docs": num_distractor_docs,
            "post_quote_search_budget": post_quote_search_budget,
        }
    )
    return {
        "action": action_summary,
        "stop": stop_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verifier_model_name_or_path", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--num_distractor_docs", type=int, default=3)
    parser.add_argument("--post_quote_search_budget", type=int, default=1)
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
    corpus_text_by_doc = build_corpus_text_by_doc(corpus_map)
    corpus_sentences_by_doc = build_corpus_sentences_by_doc(corpus_map)
    verifier = FrozenSequenceVerifier(
        args.verifier_model_name_or_path,
        attn_implementation=args.attn_implementation,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: Dict[str, Dict[str, object]] = {}
    for split in ["train", "validation"]:
        summary = export_split(
            split=split,
            verifier=verifier,
            corpus_map=corpus_map,
            corpus_text_by_doc=corpus_text_by_doc,
            corpus_sentences_by_doc=corpus_sentences_by_doc,
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            doc_aggregation=args.doc_aggregation,
            aggregation_top_k=args.aggregation_top_k,
            trust_remote_code=args.trust_remote_code,
            num_distractor_docs=args.num_distractor_docs,
            post_quote_search_budget=args.post_quote_search_budget,
        )
        summaries[split] = summary
        (args.output_dir / f"scifact_hard_action_policy_{split}_stats.json").write_text(
            json.dumps(summary["action"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (args.output_dir / f"scifact_hard_stop_policy_{split}_stats.json").write_text(
            json.dumps(summary["stop"], ensure_ascii=False, indent=2), encoding="utf-8"
        )

    export_summary = {
        "verifier_model_name_or_path": args.verifier_model_name_or_path,
        "doc_aggregation": args.doc_aggregation,
        "aggregation_top_k": args.aggregation_top_k,
        "max_steps": args.max_steps,
        "num_distractor_docs": args.num_distractor_docs,
        "post_quote_search_budget": args.post_quote_search_budget,
        "splits": summaries,
    }
    (args.output_dir / "export_summary.json").write_text(
        json.dumps(export_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(export_summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
