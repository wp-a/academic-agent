from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


ACTION_BASE_FILENAME = 'scifact_hard_action_policy_train.jsonl'
STOP_BASE_FILENAME = 'scifact_hard_stop_policy_train.jsonl'
ACTION_RELABEL_FILENAME = 'off_policy_action_relabel.jsonl'
STOP_RELABEL_FILENAME = 'off_policy_stop_relabel.jsonl'
ACTION_MIXED_FILENAME = 'scifact_hard_action_policy_train_mixed.jsonl'
STOP_MIXED_FILENAME = 'scifact_hard_stop_policy_train_mixed.jsonl'
TRAIN_SCHEMA_FIELDS = ('trajectory_id', 'step_id', 'task', 'text', 'label', 'label_text')


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _record_key(record: Dict[str, Any]) -> Tuple[str, int, str]:
    return (str(record['trajectory_id']), int(record['step_id']), str(record['task']))


def _is_uncertain_skip(record: Dict[str, Any]) -> bool:
    metadata = dict(record.get('metadata', {}) or {})
    return str(metadata.get('relabel_decision_type') or '') == 'uncertain_skip'


def _normalize_training_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {field: record[field] for field in TRAIN_SCHEMA_FIELDS}


def build_mixed_dataset(
    *,
    base_path: Path,
    relabel_path: Path,
    output_path: Path,
    include_uncertain_skip: bool = False,
) -> Dict[str, Any]:
    base_records = [_normalize_training_record(record) for record in load_jsonl(base_path)]
    relabel_records = load_jsonl(relabel_path)

    filtered_relabel_records: List[Dict[str, Any]] = []
    excluded_uncertain_skip_records = 0
    for record in relabel_records:
        if not include_uncertain_skip and _is_uncertain_skip(record):
            excluded_uncertain_skip_records += 1
            continue
        filtered_relabel_records.append(_normalize_training_record(record))

    mixed_records = list(base_records)
    index_by_key = {_record_key(record): idx for idx, record in enumerate(mixed_records)}
    overridden_records = 0
    appended_records = 0

    for record in filtered_relabel_records:
        key = _record_key(record)
        if key in index_by_key:
            mixed_records[index_by_key[key]] = record
            overridden_records += 1
        else:
            index_by_key[key] = len(mixed_records)
            mixed_records.append(record)
            appended_records += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        for record in mixed_records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write('\n')

    return {
        'base_path': str(base_path),
        'relabel_path': str(relabel_path),
        'output_path': str(output_path),
        'include_uncertain_skip': include_uncertain_skip,
        'base_records': len(base_records),
        'relabel_records': len(relabel_records),
        'included_relabel_records': len(filtered_relabel_records),
        'excluded_uncertain_skip_records': excluded_uncertain_skip_records,
        'mixed_records': len(mixed_records),
        'overridden_records': overridden_records,
        'appended_records': appended_records,
    }


def build_scifact_hard_dagger_recipe(
    *,
    base_dir: Path,
    relabel_dir: Path,
    output_dir: Path,
    include_uncertain_skip: bool = False,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    action_summary = build_mixed_dataset(
        base_path=base_dir / ACTION_BASE_FILENAME,
        relabel_path=relabel_dir / ACTION_RELABEL_FILENAME,
        output_path=output_dir / ACTION_MIXED_FILENAME,
        include_uncertain_skip=include_uncertain_skip,
    )
    stop_summary = build_mixed_dataset(
        base_path=base_dir / STOP_BASE_FILENAME,
        relabel_path=relabel_dir / STOP_RELABEL_FILENAME,
        output_path=output_dir / STOP_MIXED_FILENAME,
        include_uncertain_skip=include_uncertain_skip,
    )
    summary = {
        'base_dir': str(base_dir),
        'relabel_dir': str(relabel_dir),
        'output_dir': str(output_dir),
        'include_uncertain_skip': include_uncertain_skip,
        'action': action_summary,
        'stop': stop_summary,
    }
    summary_path = output_dir / 'dagger_mix_summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    summary['summary_path'] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=Path)
    parser.add_argument('--relabel_path', type=Path)
    parser.add_argument('--output_path', type=Path)
    parser.add_argument('--base_dir', type=Path)
    parser.add_argument('--relabel_dir', type=Path)
    parser.add_argument('--output_dir', type=Path)
    parser.add_argument('--include_uncertain_skip', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.base_dir is not None or args.relabel_dir is not None or args.output_dir is not None:
        if args.base_dir is None or args.relabel_dir is None or args.output_dir is None:
            raise ValueError('--base_dir, --relabel_dir, and --output_dir must be provided together')
        summary = build_scifact_hard_dagger_recipe(
            base_dir=args.base_dir,
            relabel_dir=args.relabel_dir,
            output_dir=args.output_dir,
            include_uncertain_skip=bool(args.include_uncertain_skip),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.base_path is None or args.relabel_path is None or args.output_path is None:
        raise ValueError('--base_path, --relabel_path, and --output_path are required for single-file mixing')
    summary = build_mixed_dataset(
        base_path=args.base_path,
        relabel_path=args.relabel_path,
        output_path=args.output_path,
        include_uncertain_skip=bool(args.include_uncertain_skip),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
