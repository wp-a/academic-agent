from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--state_text", default="")
    parser.add_argument("--state_text_file", type=Path)
    parser.add_argument("--max_length", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.state_text_file is not None:
        state_text = args.state_text_file.read_text(encoding="utf-8")
    elif args.state_text:
        state_text = args.state_text
    else:
        raise ValueError("Provide --state_text or --state_text_file.")

    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    with (args.model_dir / "label_names.json").open("r", encoding="utf-8") as handle:
        label_names = json.load(handle)

    model.eval()
    encoded = tokenizer(
        state_text,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**encoded)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs).item())

    result = {
        "predicted_label_id": pred_id,
        "predicted_action_type": label_names[pred_id],
        "scores": {label_names[i]: round(float(probs[i].item()), 6) for i in range(len(label_names))},
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
