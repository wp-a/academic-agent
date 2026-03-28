from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_agent.models.action_policy import FrozenActionPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--state_text", default="")
    parser.add_argument("--state_text_file", type=Path)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--attn_implementation", default="sdpa")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.state_text_file is not None:
        state_text = args.state_text_file.read_text(encoding="utf-8")
    elif args.state_text:
        state_text = args.state_text
    else:
        raise ValueError("Provide --state_text or --state_text_file.")

    policy = FrozenActionPolicy(
        args.model_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        attn_implementation=args.attn_implementation,
    )
    logits = policy.predict_logits([state_text])[0]
    pred_id = max(range(len(logits)), key=lambda idx: logits[idx])
    result = {
        "predicted_label_id": pred_id,
        "predicted_action_type": policy.label_names[pred_id],
        "scores": {policy.label_names[i]: round(float(logits[i]), 6) for i in range(len(logits))},
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
