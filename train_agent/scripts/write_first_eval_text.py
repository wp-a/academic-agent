from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=Path, required=True)
    parser.add_argument("--output_text", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_text.parent.mkdir(parents=True, exist_ok=True)
    with args.input_jsonl.open("r", encoding="utf-8") as handle:
        first = json.loads(next(handle))
    args.output_text.write_text(first["text"], encoding="utf-8")
    print(json.dumps({
        "trajectory_id": first["trajectory_id"],
        "step_id": first["step_id"],
        "output_text": str(args.output_text),
    }))


if __name__ == "__main__":
    main()
