from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from train_agent.eval.action_policy_metrics import compute_action_policy_metrics
from train_agent.trainers.common import (
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
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--lora_modules_to_save", default="score")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def build_pretrained_classifier(model_name_or_path: str, num_labels: int, *, attn_implementation: str, trust_remote_code: bool):
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
    kwargs = {
        "config": config,
        "ignore_mismatched_sizes": True,
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, **kwargs)
    except (TypeError, ValueError):
        kwargs.pop("attn_implementation", None)
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, **kwargs)
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def maybe_apply_lora(model, args: argparse.Namespace):
    if not args.use_lora:
        return model
    target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    modules_to_save = [item.strip() for item in args.lora_modules_to_save.split(",") if item.strip()]
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save or None,
        bias="none",
    )
    return get_peft_model(model, peft_config)


def _to_list(values):
    return values.tolist() if hasattr(values, "tolist") else values


def build_metrics_fn(label_names: List[str]):
    def _compute_metrics(eval_prediction):
        logits = _to_list(eval_prediction.predictions)
        labels = _to_list(eval_prediction.label_ids)
        return compute_action_policy_metrics(logits=logits, labels=labels, label_names=label_names)

    return _compute_metrics


def build_training_args(args: argparse.Namespace, *, use_cpu: bool) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
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
        group_by_length=True,
        ddp_find_unused_parameters=False,
        seed=args.seed,
    )


def normalize_eval_metrics(metrics: dict) -> dict:
    return {(key[5:] if key.startswith("eval_") else key): value for key, value in metrics.items()}


def run_training(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_runtime_env()
    set_seed(args.seed)

    train_dataset, eval_dataset, label_names = load_classification_datasets(args.train_file, args.eval_file)
    if args.model_name_or_path:
        model, tokenizer = build_pretrained_classifier(
            args.model_name_or_path,
            num_labels=len(label_names),
            attn_implementation=args.attn_implementation,
            trust_remote_code=args.trust_remote_code,
        )
        use_cpu = False
    else:
        model, tokenizer = build_smoke_classifier(
            train_dataset,
            eval_dataset,
            num_labels=len(label_names),
            max_length=args.max_length,
        )
        use_cpu = True

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model = maybe_apply_lora(model, args)

    train_dataset = tokenize_classification_dataset(train_dataset, tokenizer, args.max_length)
    eval_dataset = tokenize_classification_dataset(eval_dataset, tokenizer, args.max_length)

    trainer = Trainer(
        model=model,
        args=build_training_args(args, use_cpu=use_cpu),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if not use_cpu else None),
        compute_metrics=build_metrics_fn(label_names),
    )
    train_result = trainer.train()
    raw_eval_metrics = trainer.evaluate()
    eval_metrics = normalize_eval_metrics(raw_eval_metrics)
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    with (args.output_dir / "training_args.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2, default=str)
    with (args.output_dir / "train_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(train_result.metrics, handle, ensure_ascii=False, indent=2, default=str)
    with (args.output_dir / "eval_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(eval_metrics, handle, ensure_ascii=False, indent=2, default=str)
    with (args.output_dir / "label_names.json").open("w", encoding="utf-8") as handle:
        json.dump(label_names, handle, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "label_names": label_names,
                "train_metrics": train_result.metrics,
                "eval_metrics": eval_metrics,
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
