from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from train_agent.data.adapters.scifact import build_scifact_restricted_episode
from train_agent.data.adapters.scifact_hard import augment_episode_with_lexical_distractors
from train_agent.models.action_policy import FrozenActionPolicy
from train_agent.models.stop_policy import FrozenStopPolicy
from train_agent.models.verifier import FrozenSequenceVerifier
from train_agent.rl.restricted_retrieval import RestrictedRetrievalEpisode
from train_agent.scripts.build_mixed_trainset import build_scifact_hard_dagger_recipe
from train_agent.scripts.build_stronger_teacher_relabels import build_relabels_from_files
from train_agent.scripts.eval_action_policy_offline_replay import (
    build_corpus_sentences_by_doc,
    build_corpus_text_by_doc,
    evaluate_policy_on_episodes,
)
from train_agent.scripts.export_scifact_frozen_verifier_replay import build_scifact_corpus_map
from train_agent.trainers.train_action_policy import run_training


JOINT_COMPARE_METRICS = (
    'action_agreement',
    'stop_recall',
    'success_rate',
    'early_stop_rate',
    'quote_evidence_hit_rate',
)
STEP_COMPARE_METRICS = ('accuracy', 'macro_f1')
PRESET_EXPORT_RELABEL_MIX_ONLY = 'export_relabel_mix_only'
SUPPORTED_PRESETS = {PRESET_EXPORT_RELABEL_MIX_ONLY}


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    if not getattr(args, 'preset', None):
        return args
    if args.preset == PRESET_EXPORT_RELABEL_MIX_ONLY:
        args.skip_smoke_compare = True
        return args
    raise ValueError(f'Unsupported preset: {args.preset}')


def build_round_paths(output_root: Path) -> Dict[str, Path]:
    return {
        'output_root': output_root,
        'off_policy_dir': output_root / 'off_policy',
        'relabel_dir': output_root / 'relabel',
        'mixed_train_dir': output_root / 'mixed_train',
        'smoke_compare_dir': output_root / 'smoke_compare',
        'round_summary_path': output_root / 'round_summary.json',
    }


def _round_delta(base_value: Any, mixed_value: Any) -> float:
    return round(float(mixed_value) - float(base_value), 6)


def summarize_joint_comparison(*, base: Dict[str, Any], mixed: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'base': {metric: base[metric] for metric in JOINT_COMPARE_METRICS},
        'mixed': {metric: mixed[metric] for metric in JOINT_COMPARE_METRICS},
        'delta': {metric: _round_delta(base[metric], mixed[metric]) for metric in JOINT_COMPARE_METRICS},
    }


def summarize_step_comparison(*, base: Dict[str, Any], mixed: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'base': {metric: base[metric] for metric in STEP_COMPARE_METRICS},
        'mixed': {metric: mixed[metric] for metric in STEP_COMPARE_METRICS},
        'delta': {metric: _round_delta(base[metric], mixed[metric]) for metric in STEP_COMPARE_METRICS},
    }


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def build_scifact_hard_episodes(
    *,
    split: str,
    max_steps: int,
    num_distractor_docs: int,
    trust_remote_code: bool,
    episode_limit: Optional[int],
) -> List[RestrictedRetrievalEpisode]:
    claims = load_dataset('allenai/scifact', 'claims', split=split, trust_remote_code=trust_remote_code)
    corpus_dataset = load_dataset('allenai/scifact', 'corpus', trust_remote_code=trust_remote_code)
    corpus = corpus_dataset[list(corpus_dataset.keys())[0]]
    corpus_map = build_scifact_corpus_map(corpus)
    corpus_text_by_doc = build_corpus_text_by_doc(corpus_map) if num_distractor_docs > 0 else {}
    corpus_sentences_by_doc = build_corpus_sentences_by_doc(corpus_map) if num_distractor_docs > 0 else {}

    episodes: List[RestrictedRetrievalEpisode] = []
    for row in claims:
        episode = build_scifact_restricted_episode(dict(row), corpus=corpus_map, max_steps=max_steps)
        if not episode.gold_evidence:
            continue
        if num_distractor_docs > 0:
            episode = augment_episode_with_lexical_distractors(
                episode=episode,
                corpus_text_by_doc=corpus_text_by_doc,
                corpus_sentences_by_doc=corpus_sentences_by_doc,
                num_distractor_docs=num_distractor_docs,
            )
        episodes.append(episode)
        if episode_limit is not None and len(episodes) >= episode_limit:
            break
    return episodes


def export_off_policy_states(*, episodes: List[RestrictedRetrievalEpisode], args: argparse.Namespace, paths: Dict[str, Path]) -> Dict[str, Any]:
    paths['off_policy_dir'].mkdir(parents=True, exist_ok=True)
    summary_path = paths['off_policy_dir'] / 'offline_replay_train_hard.json'
    diagnostics_path = paths['off_policy_dir'] / 'offline_replay_train_hard_mismatch_episodes.jsonl'
    action_path = paths['off_policy_dir'] / 'off_policy_action_train_hard.jsonl'
    stop_path = paths['off_policy_dir'] / 'off_policy_stop_train_hard.jsonl'

    verifier = FrozenSequenceVerifier(
        args.verifier_model_name_or_path,
        attn_implementation=args.attn_implementation,
        max_length=args.verifier_max_length,
        batch_size=args.verifier_batch_size,
    )
    action_policy = FrozenActionPolicy(
        Path(args.student_policy_model_dir),
        max_length=args.source_policy_max_length,
        batch_size=args.source_policy_batch_size,
        attn_implementation=args.attn_implementation,
    )
    stop_policy = FrozenStopPolicy(
        Path(args.student_stop_model_dir),
        max_length=args.source_stop_max_length,
        batch_size=args.source_stop_batch_size,
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
        diagnostics_output_path=diagnostics_path,
        off_policy_action_output_path=action_path,
        off_policy_stop_output_path=stop_path,
    )
    summary.update(
        {
            'split': args.train_split,
            'episodes_built': len(episodes),
            'episode_limit': args.train_episode_limit,
            'summary_path': str(summary_path),
            'diagnostics_output_path': str(diagnostics_path),
            'off_policy_action_output_path': str(action_path),
            'off_policy_stop_output_path': str(stop_path),
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def _build_training_namespace(*, train_file: Path, eval_file: Path, output_dir: Path, args: argparse.Namespace) -> Namespace:
    return Namespace(
        train_file=train_file,
        eval_file=eval_file,
        output_dir=output_dir,
        model_name_or_path=args.smoke_init_model_name_or_path,
        max_length=args.smoke_max_length,
        learning_rate=args.smoke_learning_rate,
        num_train_epochs=3,
        per_device_train_batch_size=args.smoke_train_batch_size,
        per_device_eval_batch_size=args.smoke_eval_batch_size,
        gradient_accumulation_steps=1,
        logging_steps=args.smoke_logging_steps,
        eval_steps=args.smoke_eval_steps,
        save_steps=args.smoke_save_steps,
        max_steps=args.smoke_max_steps,
        smoke_test=False,
        attn_implementation=args.attn_implementation,
        use_lora=False,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        lora_target_modules='q_proj,k_proj,v_proj,o_proj',
        lora_modules_to_save='score',
        gradient_checkpointing=False,
        trust_remote_code=False,
        seed=args.seed,
    )


def train_smoke_models(*, paths: Dict[str, Path], args: argparse.Namespace) -> Dict[str, Any]:
    smoke_dir = paths['smoke_compare_dir']
    smoke_dir.mkdir(parents=True, exist_ok=True)

    action_base_dir = smoke_dir / 'action_base'
    action_mixed_dir = smoke_dir / 'action_mixed'
    stop_base_dir = smoke_dir / 'stop_base'
    stop_mixed_dir = smoke_dir / 'stop_mixed'

    run_training(
        _build_training_namespace(
            train_file=Path(args.base_train_dir) / 'scifact_hard_action_policy_train.jsonl',
            eval_file=Path(args.base_train_dir) / 'scifact_hard_action_policy_validation.jsonl',
            output_dir=action_base_dir,
            args=args,
        )
    )
    run_training(
        _build_training_namespace(
            train_file=paths['mixed_train_dir'] / 'scifact_hard_action_policy_train_mixed.jsonl',
            eval_file=Path(args.base_train_dir) / 'scifact_hard_action_policy_validation.jsonl',
            output_dir=action_mixed_dir,
            args=args,
        )
    )
    run_training(
        _build_training_namespace(
            train_file=Path(args.base_train_dir) / 'scifact_hard_stop_policy_train.jsonl',
            eval_file=Path(args.base_train_dir) / 'scifact_hard_stop_policy_validation.jsonl',
            output_dir=stop_base_dir,
            args=args,
        )
    )
    run_training(
        _build_training_namespace(
            train_file=paths['mixed_train_dir'] / 'scifact_hard_stop_policy_train_mixed.jsonl',
            eval_file=Path(args.base_train_dir) / 'scifact_hard_stop_policy_validation.jsonl',
            output_dir=stop_mixed_dir,
            args=args,
        )
    )

    action_base_eval = _load_json(action_base_dir / 'eval_metrics.json')
    action_mixed_eval = _load_json(action_mixed_dir / 'eval_metrics.json')
    stop_base_eval = _load_json(stop_base_dir / 'eval_metrics.json')
    stop_mixed_eval = _load_json(stop_mixed_dir / 'eval_metrics.json')

    return {
        'action_base_dir': str(action_base_dir),
        'action_mixed_dir': str(action_mixed_dir),
        'stop_base_dir': str(stop_base_dir),
        'stop_mixed_dir': str(stop_mixed_dir),
        'action_step_level': summarize_step_comparison(base=action_base_eval, mixed=action_mixed_eval),
        'stop_step_level': summarize_step_comparison(base=stop_base_eval, mixed=stop_mixed_eval),
    }


def run_joint_eval(
    *,
    episodes: List[RestrictedRetrievalEpisode],
    policy_model_dir: Path,
    stop_model_dir: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    verifier = FrozenSequenceVerifier(
        args.verifier_model_name_or_path,
        attn_implementation=args.attn_implementation,
        max_length=args.verifier_max_length,
        batch_size=args.verifier_batch_size,
    )
    action_policy = FrozenActionPolicy(
        policy_model_dir,
        max_length=args.smoke_max_length,
        batch_size=args.smoke_eval_batch_size,
        attn_implementation=args.attn_implementation,
    )
    stop_policy = FrozenStopPolicy(
        stop_model_dir,
        max_length=args.smoke_max_length,
        batch_size=args.smoke_eval_batch_size,
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
    )
    summary.update(
        {
            'split': args.validation_split,
            'policy_model_dir': str(policy_model_dir),
            'stop_model_dir': str(stop_model_dir),
            'output_path': str(output_path),
        }
    )
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_root', type=Path, required=True)
    parser.add_argument('--preset', choices=sorted(SUPPORTED_PRESETS))
    parser.add_argument('--student_policy_model_dir', required=True)
    parser.add_argument('--student_stop_model_dir', required=True)
    parser.add_argument('--verifier_model_name_or_path', required=True)
    parser.add_argument('--base_train_dir', default='data/processed/scifact_hard_replay_v1')
    parser.add_argument('--teacher_backend', default='rule_based')
    parser.add_argument('--teacher_type', default='rule_based_stronger_teacher_v1')
    parser.add_argument('--teacher_version', default='v1')
    parser.add_argument('--minimum_teacher_confidence', default='high')
    parser.add_argument('--train_split', default='train')
    parser.add_argument('--validation_split', default='validation')
    parser.add_argument('--train_episode_limit', type=int)
    parser.add_argument('--validation_episode_limit', type=int)
    parser.add_argument('--max_steps', type=int, default=5)
    parser.add_argument('--num_distractor_docs', type=int, default=3)
    parser.add_argument('--reference_policy_type', choices=['weak', 'conservative'], default='conservative')
    parser.add_argument('--post_quote_search_budget', type=int, default=1)
    parser.add_argument('--doc_aggregation', default='full_document')
    parser.add_argument('--aggregation_top_k', type=int, default=3)
    parser.add_argument('--attn_implementation', default='sdpa')
    parser.add_argument('--verifier_max_length', type=int, default=384)
    parser.add_argument('--verifier_batch_size', type=int, default=8)
    parser.add_argument('--source_policy_max_length', type=int, default=512)
    parser.add_argument('--source_policy_batch_size', type=int, default=8)
    parser.add_argument('--source_stop_max_length', type=int, default=512)
    parser.add_argument('--source_stop_batch_size', type=int, default=8)
    parser.add_argument('--smoke_init_model_name_or_path', default='outputs/action_policy_scifact_bert_tiny_v1')
    parser.add_argument('--smoke_max_length', type=int, default=256)
    parser.add_argument('--smoke_max_steps', type=int, default=200)
    parser.add_argument('--smoke_train_batch_size', type=int, default=16)
    parser.add_argument('--smoke_eval_batch_size', type=int, default=32)
    parser.add_argument('--smoke_learning_rate', type=float, default=2e-4)
    parser.add_argument('--smoke_logging_steps', type=int, default=20)
    parser.add_argument('--smoke_eval_steps', type=int, default=50)
    parser.add_argument('--smoke_save_steps', type=int, default=200)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--trust_remote_code', action='store_true')
    parser.add_argument('--skip_smoke_compare', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = apply_preset(parse_args())
    paths = build_round_paths(args.output_root)
    for key in ('output_root', 'off_policy_dir', 'relabel_dir', 'mixed_train_dir', 'smoke_compare_dir'):
        paths[key].mkdir(parents=True, exist_ok=True)

    train_episodes = build_scifact_hard_episodes(
        split=args.train_split,
        max_steps=args.max_steps,
        num_distractor_docs=args.num_distractor_docs,
        trust_remote_code=args.trust_remote_code,
        episode_limit=args.train_episode_limit,
    )
    off_policy_summary = export_off_policy_states(episodes=train_episodes, args=args, paths=paths)
    relabel_summary = build_relabels_from_files(
        diagnostics_path=paths['off_policy_dir'] / 'offline_replay_train_hard_mismatch_episodes.jsonl',
        off_policy_action_path=paths['off_policy_dir'] / 'off_policy_action_train_hard.jsonl',
        off_policy_stop_path=paths['off_policy_dir'] / 'off_policy_stop_train_hard.jsonl',
        output_dir=paths['relabel_dir'],
        split=args.train_split,
        teacher_backend=args.teacher_backend,
        teacher_type=args.teacher_type,
        teacher_version=args.teacher_version,
        minimum_teacher_confidence=args.minimum_teacher_confidence,
    )
    mixed_train_summary = build_scifact_hard_dagger_recipe(
        base_dir=Path(args.base_train_dir),
        relabel_dir=paths['relabel_dir'],
        output_dir=paths['mixed_train_dir'],
        include_uncertain_skip=False,
    )

    round_summary: Dict[str, Any] = {
        'output_root': str(paths['output_root']),
        'preset': args.preset,
        'off_policy_export': off_policy_summary,
        'relabel': relabel_summary,
        'mixed_train': mixed_train_summary,
    }

    if not args.skip_smoke_compare:
        validation_episodes = build_scifact_hard_episodes(
            split=args.validation_split,
            max_steps=args.max_steps,
            num_distractor_docs=args.num_distractor_docs,
            trust_remote_code=args.trust_remote_code,
            episode_limit=args.validation_episode_limit,
        )
        smoke_compare_summary = train_smoke_models(paths=paths, args=args)
        base_joint = run_joint_eval(
            episodes=validation_episodes,
            policy_model_dir=paths['smoke_compare_dir'] / 'action_base',
            stop_model_dir=paths['smoke_compare_dir'] / 'stop_base',
            output_path=paths['smoke_compare_dir'] / 'joint_offline_replay_validation_hard_base.json',
            args=args,
        )
        mixed_joint = run_joint_eval(
            episodes=validation_episodes,
            policy_model_dir=paths['smoke_compare_dir'] / 'action_mixed',
            stop_model_dir=paths['smoke_compare_dir'] / 'stop_mixed',
            output_path=paths['smoke_compare_dir'] / 'joint_offline_replay_validation_hard_mixed.json',
            args=args,
        )
        smoke_compare_summary['joint_offline_replay'] = summarize_joint_comparison(base=base_joint, mixed=mixed_joint)
        smoke_compare_summary['base_joint_output_path'] = str(paths['smoke_compare_dir'] / 'joint_offline_replay_validation_hard_base.json')
        smoke_compare_summary['mixed_joint_output_path'] = str(paths['smoke_compare_dir'] / 'joint_offline_replay_validation_hard_mixed.json')
        round_summary['smoke_compare'] = smoke_compare_summary

    paths['round_summary_path'].write_text(json.dumps(round_summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(round_summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
