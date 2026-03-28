from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_jsonl(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write('\n')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--train_output', type=Path, required=True)
    parser.add_argument('--eval_output', type=Path, required=True)
    parser.add_argument('--eval_ratio', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    if len(rows) < 2:
        raise ValueError('Need at least 2 examples to create train/eval split.')
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    eval_count = max(1, int(len(rows) * args.eval_ratio))
    eval_rows = rows[:eval_count]
    train_rows = rows[eval_count:]
    if not train_rows:
        train_rows = eval_rows[:1]
        eval_rows = eval_rows[1:]
    dump_jsonl(train_rows, args.train_output)
    dump_jsonl(eval_rows, args.eval_output)
    print(json.dumps({
        'input_examples': len(rows),
        'train_examples': len(train_rows),
        'eval_examples': len(eval_rows),
        'train_output': str(args.train_output),
        'eval_output': str(args.eval_output),
    }))


if __name__ == '__main__':
    main()
