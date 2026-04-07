from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple
from urllib import error, request

from train_agent.scripts.analyze_hard_replay_mismatches import analyze_mismatch_files, load_jsonl


SOURCE_NAME = 'hard_off_policy_relabel_v1'
_CONFIDENCE_RANK = {'low': 0, 'medium': 1, 'high': 2}
SUPPORTED_TEACHER_BACKENDS = {'rule_based', 'llm_api'}
SUPPORTED_ACTION_TYPES = {'search', 'quote_evidence', 'stop'}
SUPPORTED_STOP_LABELS = {'yes', 'no'}
SUPPORTED_DECISION_TYPES = {'correct_reference', 'override_reference', 'uncertain_skip'}
DEFAULT_LLM_API_BASE = 'https://api.openai.com/v1'
DEFAULT_LLM_API_KEY_ENV = 'OPENAI_API_KEY'
DEFAULT_LLM_MODEL_ENV = 'LLM_ANNOTATOR_MODEL'
DEFAULT_LLM_TIMEOUT_SECONDS = 60


def _rule_based_teacher_decision(bucket: str, reference_action: str) -> Dict[str, str]:
    if bucket == 'premature_stop_after_evidence':
        return {
            'action_type': 'quote_evidence',
            'should_stop': 'no',
            'stop_reason': 'need_quote_before_stop',
            'confidence': 'high',
            'rationale_short': 'Gold evidence is revealed but has not been quoted yet.',
            'decision_type': 'correct_reference',
        }
    if bucket == 'oversearch_after_quote':
        return {
            'action_type': 'stop',
            'should_stop': 'yes',
            'stop_reason': 'sufficient_quoted_evidence',
            'confidence': 'high',
            'rationale_short': 'Gold evidence has already been quoted, so more search is redundant.',
            'decision_type': 'correct_reference',
        }
    should_stop = 'yes' if reference_action == 'stop' else 'no'
    stop_reason = 'chosen_stop' if should_stop == 'yes' else f'continue_after_{reference_action}'
    return {
        'action_type': reference_action,
        'should_stop': should_stop,
        'stop_reason': stop_reason,
        'confidence': 'medium',
        'rationale_short': 'Fallback to the current reference policy label.',
        'decision_type': 'correct_reference',
    }


def _extract_json_object(text: str) -> Dict[str, Any]:
    payload = str(text).strip()
    try:
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = payload.find('{')
    end = payload.rfind('}')
    if start == -1 or end == -1 or end <= start:
        raise ValueError('No JSON object found in LLM response.')
    parsed = json.loads(payload[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError('LLM response JSON is not an object.')
    return parsed


def _normalize_teacher_decision(raw: Dict[str, Any], *, reference_action: str) -> Dict[str, str]:
    action_type = str(raw.get('action_type') or reference_action or 'search').strip().lower()
    if action_type not in SUPPORTED_ACTION_TYPES:
        action_type = reference_action if reference_action in SUPPORTED_ACTION_TYPES else 'search'

    should_stop = str(raw.get('should_stop') or ('yes' if action_type == 'stop' else 'no')).strip().lower()
    if should_stop not in SUPPORTED_STOP_LABELS:
        should_stop = 'yes' if action_type == 'stop' else 'no'

    stop_reason = str(raw.get('stop_reason') or ('chosen_stop' if should_stop == 'yes' else f'continue_after_{action_type}')).strip()
    confidence = str(raw.get('confidence') or 'low').strip().lower()
    if confidence not in _CONFIDENCE_RANK:
        confidence = 'low'

    decision_type = str(raw.get('decision_type') or '').strip().lower()
    if decision_type not in SUPPORTED_DECISION_TYPES:
        if confidence == 'low':
            decision_type = 'uncertain_skip'
        elif action_type == reference_action:
            decision_type = 'correct_reference'
        else:
            decision_type = 'override_reference'

    rationale_short = str(raw.get('rationale_short') or '').strip()
    if not rationale_short:
        rationale_short = 'LLM teacher returned a normalized decision without additional explanation.'

    return {
        'action_type': action_type,
        'should_stop': should_stop,
        'stop_reason': stop_reason,
        'confidence': confidence,
        'rationale_short': rationale_short,
        'decision_type': decision_type,
    }


def _build_llm_teacher_prompt(
    *,
    action_record: Dict[str, Any],
    episode_summary: Dict[str, Any],
    reference_action: str,
) -> str:
    metadata = dict(action_record.get('metadata', {}) or {})
    payload = {
        'episode_id': metadata.get('episode_id') or action_record.get('trajectory_id'),
        'failure_bucket': episode_summary.get('bucket'),
        'reference_action': reference_action,
        'student_action': metadata.get('student_action'),
        'stop_policy_should_stop': metadata.get('stop_policy_should_stop'),
        'is_first_off_policy_step': metadata.get('is_first_off_policy_step'),
        'post_quote_search_budget': metadata.get('post_quote_search_budget'),
        'state_text': action_record.get('text'),
    }
    return (
        'You are a stronger teacher for an evidence-seeking agent in a restricted evidence environment. '\
        'Read the state and produce one JSON object with keys '\
        'action_type, should_stop, stop_reason, confidence, decision_type, rationale_short. '\
        'Allowed action_type: search, quote_evidence, stop. '\
        'Allowed should_stop: yes, no. '\
        'Allowed confidence: low, medium, high. '\
        'Allowed decision_type: correct_reference, override_reference, uncertain_skip. '\
        'Prefer uncertain_skip if the state is ambiguous or underspecified. '\
        'Do not output markdown.\n\n'
        f'{json.dumps(payload, ensure_ascii=False, indent=2)}'
    )


def _extract_chat_content(response_payload: Dict[str, Any]) -> str:
    choices = response_payload.get('choices') or []
    if not choices:
        raise ValueError('LLM response missing choices.')
    message = choices[0].get('message') or {}
    content = message.get('content')
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                parts.append(str(item.get('text') or ''))
        if parts:
            return ''.join(parts)
    raise ValueError('LLM response missing message.content text.')


def _llm_api_teacher_decision(
    *,
    bucket: str,
    reference_action: str,
    action_record: Dict[str, Any],
    episode_summary: Dict[str, Any],
    teacher_type: str,
    teacher_version: str,
    teacher_model_name: Optional[str],
    teacher_api_base: Optional[str],
    teacher_api_key_env: str,
    teacher_timeout_seconds: int,
) -> Dict[str, str]:
    model_name = str(teacher_model_name or os.environ.get(DEFAULT_LLM_MODEL_ENV, '')).strip()
    if not model_name:
        raise ValueError('teacher_model_name is required for teacher_backend=llm_api')

    api_key = str(os.environ.get(teacher_api_key_env, '')).strip()
    if not api_key:
        raise ValueError(f'{teacher_api_key_env} is required for teacher_backend=llm_api')

    api_base = str(teacher_api_base or os.environ.get('OPENAI_BASE_URL') or DEFAULT_LLM_API_BASE).rstrip('/')
    prompt = _build_llm_teacher_prompt(
        action_record=action_record,
        episode_summary=episode_summary,
        reference_action=reference_action,
    )
    request_payload = {
        'model': model_name,
        'messages': [
            {
                'role': 'system',
                'content': (
                    'You are a careful stronger teacher for a modular evidence-seeking agent. '\
                    'Return a single JSON object and no extra prose.'
                ),
            },
            {'role': 'user', 'content': prompt},
        ],
        'temperature': 0.0,
        'response_format': {'type': 'json_object'},
    }
    http_request = request.Request(
        url=f'{api_base}/chat/completions',
        data=json.dumps(request_payload).encode('utf-8'),
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        },
        method='POST',
    )
    try:
        with request.urlopen(http_request, timeout=teacher_timeout_seconds) as response:
            response_payload = json.loads(response.read().decode('utf-8'))
        content = _extract_chat_content(response_payload)
        raw_decision = _extract_json_object(content)
        return _normalize_teacher_decision(raw_decision, reference_action=reference_action)
    except ValueError:
        raise
    except Exception as exc:  # pragma: no cover - network path is hard to unit test directly.
        return {
            'action_type': reference_action,
            'should_stop': 'yes' if reference_action == 'stop' else 'no',
            'stop_reason': 'llm_backend_error',
            'confidence': 'low',
            'rationale_short': f'LLM backend error: {exc.__class__.__name__}',
            'decision_type': 'uncertain_skip',
        }


def _resolve_teacher_decision(
    *,
    teacher_backend: str,
    bucket: str,
    reference_action: str,
    action_record: Dict[str, Any],
    episode_summary: Dict[str, Any],
    teacher_type: str,
    teacher_version: str,
    teacher_model_name: Optional[str],
    teacher_api_base: Optional[str],
    teacher_api_key_env: str,
    teacher_timeout_seconds: int,
) -> Dict[str, str]:
    normalized_backend = str(teacher_backend).strip().lower()
    if normalized_backend not in SUPPORTED_TEACHER_BACKENDS:
        raise ValueError(f'Unsupported teacher_backend: {teacher_backend}')
    if normalized_backend == 'rule_based':
        return _rule_based_teacher_decision(bucket=bucket, reference_action=reference_action)
    if normalized_backend == 'llm_api':
        return _llm_api_teacher_decision(
            bucket=bucket,
            reference_action=reference_action,
            action_record=action_record,
            episode_summary=episode_summary,
            teacher_type=teacher_type,
            teacher_version=teacher_version,
            teacher_model_name=teacher_model_name,
            teacher_api_base=teacher_api_base,
            teacher_api_key_env=teacher_api_key_env,
            teacher_timeout_seconds=teacher_timeout_seconds,
        )
    raise ValueError(f'Unsupported teacher_backend: {teacher_backend}')


def _index_records(records: List[Dict[str, Any]]) -> Dict[Tuple[str, int, str], Dict[str, Any]]:
    indexed: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for record in records:
        key = (str(record['trajectory_id']), int(record['step_id']), str(record['task']))
        indexed[key] = record
    return indexed


def _episode_summary_map(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(item['episode_id']): item for item in summary.get('episodes', [])}


def _normalize_minimum_teacher_confidence(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in _CONFIDENCE_RANK:
        raise ValueError(f'Unsupported minimum_teacher_confidence: {value}')
    return normalized


def _should_route_to_uncertain_skip(teacher: Dict[str, str], minimum_teacher_confidence: str) -> bool:
    confidence = str(teacher.get('confidence') or 'low').lower()
    teacher_rank = _CONFIDENCE_RANK.get(confidence, 0)
    minimum_rank = _CONFIDENCE_RANK[minimum_teacher_confidence]
    return teacher_rank < minimum_rank


def _build_action_relabel_record(
    record: Dict[str, Any],
    *,
    episode_summary: Dict[str, Any],
    dataset: str,
    split: str,
    teacher_backend: str,
    teacher_type: str,
    teacher_version: str,
    teacher_model_name: Optional[str],
    teacher_api_base: Optional[str],
    teacher_api_key_env: str,
    teacher_timeout_seconds: int,
    original_record_path: str,
    minimum_teacher_confidence: str,
) -> Dict[str, Any]:
    metadata = dict(record.get('metadata', {}) or {})
    teacher = _resolve_teacher_decision(
        teacher_backend=teacher_backend,
        bucket=str(episode_summary.get('bucket') or ''),
        reference_action=str(metadata.get('reference_action') or record.get('label') or ''),
        action_record=record,
        episode_summary=episode_summary,
        teacher_type=teacher_type,
        teacher_version=teacher_version,
        teacher_model_name=teacher_model_name,
        teacher_api_base=teacher_api_base,
        teacher_api_key_env=teacher_api_key_env,
        teacher_timeout_seconds=teacher_timeout_seconds,
    )
    if _should_route_to_uncertain_skip(teacher, minimum_teacher_confidence):
        teacher = dict(teacher)
        teacher['decision_type'] = 'uncertain_skip'
    metadata.update(
        {
            'dataset': dataset,
            'split': split,
            'source': SOURCE_NAME,
            'failure_bucket': str(episode_summary.get('bucket') or ''),
            'teacher_backend': teacher_backend,
            'teacher_type': teacher_type,
            'teacher_version': teacher_version,
            'teacher_model_name': teacher_model_name,
            'teacher_api_base': teacher_api_base,
            'teacher_api_key_env': teacher_api_key_env,
            'teacher_label_action': teacher['action_type'],
            'teacher_label_stop': teacher['should_stop'],
            'teacher_stop_reason': teacher['stop_reason'],
            'teacher_confidence': teacher['confidence'],
            'teacher_rationale_short': teacher['rationale_short'],
            'relabel_decision_type': teacher['decision_type'],
            'minimum_teacher_confidence': minimum_teacher_confidence,
            'original_record_path': original_record_path,
        }
    )
    return {
        'trajectory_id': str(record['trajectory_id']),
        'step_id': int(record['step_id']),
        'task': str(record['task']),
        'text': str(record['text']),
        'label': teacher['action_type'],
        'label_text': json.dumps({'action_type': teacher['action_type']}, ensure_ascii=False),
        'metadata': metadata,
    }


def _build_stop_relabel_record(
    action_relabel_record: Dict[str, Any],
    *,
    original_stop_record: Optional[Dict[str, Any]],
    original_record_path: str,
) -> Dict[str, Any]:
    metadata = dict(action_relabel_record.get('metadata', {}) or {})
    metadata['original_stop_record_path'] = original_record_path
    label = str(metadata.get('teacher_label_stop') or 'no')
    reason = str(metadata.get('teacher_stop_reason') or 'uncertain')
    text = str(original_stop_record.get('text')) if original_stop_record is not None else str(action_relabel_record['text'])
    return {
        'trajectory_id': str(action_relabel_record['trajectory_id']),
        'step_id': int(action_relabel_record['step_id']),
        'task': 'stop_policy_classification',
        'text': text,
        'label': label,
        'label_text': json.dumps({'should_stop': label, 'reason': reason}, ensure_ascii=False),
        'metadata': metadata,
    }


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write('\n')



def build_relabels_from_files(
    *,
    diagnostics_path: Path,
    off_policy_action_path: Path,
    off_policy_stop_path: Path,
    output_dir: Path,
    dataset: str = 'scifact',
    split: str = 'validation',
    teacher_backend: str = 'rule_based',
    teacher_type: str = 'rule_based_stronger_teacher_v1',
    teacher_version: str = 'v1',
    teacher_model_name: Optional[str] = None,
    teacher_api_base: Optional[str] = None,
    teacher_api_key_env: str = DEFAULT_LLM_API_KEY_ENV,
    teacher_timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS,
    minimum_teacher_confidence: str = 'medium',
) -> Dict[str, Any]:
    minimum_teacher_confidence = _normalize_minimum_teacher_confidence(minimum_teacher_confidence)
    teacher_backend = str(teacher_backend).strip().lower()
    if teacher_backend not in SUPPORTED_TEACHER_BACKENDS:
        raise ValueError(f'Unsupported teacher_backend: {teacher_backend}')
    normalized_teacher_model_name = str(teacher_model_name).strip() if teacher_model_name else None
    normalized_teacher_api_base = str(teacher_api_base).strip().rstrip('/') if teacher_api_base else None
    analysis_summary = analyze_mismatch_files(
        diagnostics_path=diagnostics_path,
        off_policy_action_path=off_policy_action_path,
        off_policy_stop_path=off_policy_stop_path,
    )
    episode_summary_by_id = _episode_summary_map(analysis_summary)
    action_records = load_jsonl(off_policy_action_path)
    stop_records = load_jsonl(off_policy_stop_path)
    stop_index = _index_records(stop_records)

    action_relabels: List[Dict[str, Any]] = []
    stop_relabels: List[Dict[str, Any]] = []
    uncertain_action_records: List[Dict[str, Any]] = []
    uncertain_stop_records: List[Dict[str, Any]] = []
    decision_type_counts: Counter = Counter()
    decision_type_episode_ids: DefaultDict[str, set] = defaultdict(set)

    for action_record in action_records:
        episode_id = str((action_record.get('metadata', {}) or {}).get('episode_id') or action_record.get('trajectory_id') or '')
        episode_summary = episode_summary_by_id[episode_id]
        action_relabel = _build_action_relabel_record(
            action_record,
            episode_summary=episode_summary,
            dataset=dataset,
            split=split,
            teacher_backend=teacher_backend,
            teacher_type=teacher_type,
            teacher_version=teacher_version,
            teacher_model_name=normalized_teacher_model_name,
            teacher_api_base=normalized_teacher_api_base,
            teacher_api_key_env=teacher_api_key_env,
            teacher_timeout_seconds=int(teacher_timeout_seconds),
            original_record_path=str(off_policy_action_path),
            minimum_teacher_confidence=minimum_teacher_confidence,
        )
        stop_key = (str(action_record['trajectory_id']), int(action_record['step_id']), 'stop_policy_classification')
        original_stop_record = stop_index.get(stop_key)
        stop_relabel = _build_stop_relabel_record(
            action_relabel,
            original_stop_record=original_stop_record,
            original_record_path=str(off_policy_stop_path),
        )
        decision_type = action_relabel['metadata']['relabel_decision_type']
        decision_type_counts[decision_type] += 1
        decision_type_episode_ids[decision_type].add(episode_id)
        if decision_type == 'uncertain_skip':
            uncertain_action_records.append(action_relabel)
            uncertain_stop_records.append(stop_relabel)
            continue
        action_relabels.append(action_relabel)
        stop_relabels.append(stop_relabel)

    output_dir.mkdir(parents=True, exist_ok=True)
    action_output_path = output_dir / 'off_policy_action_relabel.jsonl'
    stop_output_path = output_dir / 'off_policy_stop_relabel.jsonl'
    uncertain_action_output_path = output_dir / 'off_policy_action_uncertain_skip.jsonl'
    uncertain_stop_output_path = output_dir / 'off_policy_stop_uncertain_skip.jsonl'
    summary_output_path = output_dir / 'relabel_summary.json'
    _write_jsonl(action_output_path, action_relabels)
    _write_jsonl(stop_output_path, stop_relabels)
    _write_jsonl(uncertain_action_output_path, uncertain_action_records)
    _write_jsonl(uncertain_stop_output_path, uncertain_stop_records)

    summary = {
        'dataset': dataset,
        'split': split,
        'teacher_backend': teacher_backend,
        'teacher_type': teacher_type,
        'teacher_version': teacher_version,
        'teacher_model_name': normalized_teacher_model_name,
        'teacher_api_base': normalized_teacher_api_base,
        'teacher_api_key_env': teacher_api_key_env,
        'teacher_timeout_seconds': int(teacher_timeout_seconds),
        'minimum_teacher_confidence': minimum_teacher_confidence,
        'episodes_relabeled': len(episode_summary_by_id),
        'action_records_relabeled': len(action_relabels),
        'stop_records_relabeled': len(stop_relabels),
        'uncertain_skip_action_records': len(uncertain_action_records),
        'uncertain_skip_stop_records': len(uncertain_stop_records),
        'bucket_distribution': dict(sorted(Counter(item['bucket'] for item in episode_summary_by_id.values()).items())),
        'decision_type_distribution': dict(sorted(decision_type_counts.items())),
        'decision_type_episode_ids': {
            key: sorted(value) for key, value in sorted(decision_type_episode_ids.items())
        },
        'diagnostics_path': str(diagnostics_path),
        'off_policy_action_path': str(off_policy_action_path),
        'off_policy_stop_path': str(off_policy_stop_path),
        'action_output_path': str(action_output_path),
        'stop_output_path': str(stop_output_path),
        'uncertain_action_output_path': str(uncertain_action_output_path),
        'uncertain_stop_output_path': str(uncertain_stop_output_path),
    }
    summary_output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--diagnostics_path', type=Path, required=True)
    parser.add_argument('--off_policy_action_path', type=Path, required=True)
    parser.add_argument('--off_policy_stop_path', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--dataset', default='scifact')
    parser.add_argument('--split', default='validation')
    parser.add_argument('--teacher_backend', default='rule_based', choices=sorted(SUPPORTED_TEACHER_BACKENDS))
    parser.add_argument('--teacher_type', default='rule_based_stronger_teacher_v1')
    parser.add_argument('--teacher_version', default='v1')
    parser.add_argument('--teacher_model_name')
    parser.add_argument('--teacher_api_base')
    parser.add_argument('--teacher_api_key_env', default=DEFAULT_LLM_API_KEY_ENV)
    parser.add_argument('--teacher_timeout_seconds', type=int, default=DEFAULT_LLM_TIMEOUT_SECONDS)
    parser.add_argument('--minimum_teacher_confidence', default='medium')
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    summary = build_relabels_from_files(
        diagnostics_path=args.diagnostics_path,
        off_policy_action_path=args.off_policy_action_path,
        off_policy_stop_path=args.off_policy_stop_path,
        output_dir=args.output_dir,
        dataset=args.dataset,
        split=args.split,
        teacher_backend=args.teacher_backend,
        teacher_type=args.teacher_type,
        teacher_version=args.teacher_version,
        teacher_model_name=args.teacher_model_name,
        teacher_api_base=args.teacher_api_base,
        teacher_api_key_env=args.teacher_api_key_env,
        teacher_timeout_seconds=args.teacher_timeout_seconds,
        minimum_teacher_confidence=args.minimum_teacher_confidence,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
