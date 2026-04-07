from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _record_key(record: Dict[str, Any]) -> Tuple[str, int, str]:
    return (str(record["trajectory_id"]), int(record["step_id"]), str(record["task"]))


def merge_jsonl_files(*, base_path: Path, relabel_path: Path, output_path: Path) -> Dict[str, Any]:
    base_records = load_jsonl(base_path)
    relabel_records = load_jsonl(relabel_path)

    merged_records = list(base_records)
    index_by_key = {_record_key(record): idx for idx, record in enumerate(merged_records)}
    overridden_records = 0
    appended_records = 0

    for record in relabel_records:
        key = _record_key(record)
        if key in index_by_key:
            merged_records[index_by_key[key]] = record
            overridden_records += 1
        else:
            index_by_key[key] = len(merged_records)
            merged_records.append(record)
            appended_records += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in merged_records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")

    return {
        "base_path": str(base_path),
        "relabel_path": str(relabel_path),
        "output_path": str(output_path),
        "base_records": len(base_records),
        "relabel_records": len(relabel_records),
        "merged_records": len(merged_records),
        "overridden_records": overridden_records,
        "appended_records": appended_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=Path, required=True)
    parser.add_argument("--relabel_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = merge_jsonl_files(
        base_path=args.base_path,
        relabel_path=args.relabel_path,
        output_path=args.output_path,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
