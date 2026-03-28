from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from train_agent.trainers.common import (
    build_pretrained_classifier,
    build_smoke_classifier,
    load_classification_datasets,
    set_runtime_env,
    tokenize_classification_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=Path, required=True)
    parser.add_argument("--eval_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name_or_path", default="")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_runtime_env()

    train_dataset, eval_dataset, label_names = load_classification_datasets(args.train_file, args.eval_file)
    if args.model_name_or_path:
        model, tokenizer = build_pretrained_classifier(args.model_name_or_path, num_labels=len(label_names))
        use_cpu = False
    else:
        model, tokenizer = build_smoke_classifier(train_dataset, eval_dataset, num_labels=len(label_names), max_length=args.max_length)
        use_cpu = True

    train_dataset = tokenize_classification_dataset(train_dataset, tokenizer, args.max_length)
    eval_dataset = tokenize_classification_dataset(eval_dataset, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        fp16=not use_cpu,
        bf16=False,
        no_cuda=use_cpu,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        save_total_limit=2,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    metrics = train_result.metrics
    with (args.output_dir / "training_args.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2, default=str)
    with (args.output_dir / "train_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, default=str)
    with (args.output_dir / "label_names.json").open("w", encoding="utf-8") as handle:
        json.dump(label_names, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"output_dir": str(args.output_dir), "label_names": label_names, "metrics": metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
