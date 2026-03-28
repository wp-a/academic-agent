from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


PROMPT_TEMPLATE = "{prompt}\n\nResponse:\n{response}"


@dataclass
class SFTConfig:
    model_name_or_path: str
    train_file: Path
    eval_file: Path
    output_dir: Path
    max_length: int = 1024
    learning_rate: float = 2e-4
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    warmup_ratio: float = 0.03
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    fp16: bool = False


def parse_args() -> SFTConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["next_action", "stopping"], required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--train_file", type=Path, required=True)
    parser.add_argument("--eval_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    return SFTConfig(
        model_name_or_path=args.model_name_or_path,
        train_file=args.train_file,
        eval_file=args.eval_file,
        output_dir=args.output_dir,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        warmup_ratio=args.warmup_ratio,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        fp16=args.fp16,
    )


def format_example(row: Dict[str, str]) -> Dict[str, str]:
    return {"text": PROMPT_TEMPLATE.format(prompt=row["prompt"], response=row["response"])}


def tokenize_dataset(dataset, tokenizer, max_length: int):
    def _tokenize(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        encoded["labels"] = [ids[:] for ids in encoded["input_ids"]]
        return encoded

    return dataset.map(_tokenize, batched=True, remove_columns=dataset.column_names)


def build_model_and_tokenizer(config: SFTConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def main() -> None:
    config = parse_args()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    raw_train = load_dataset("json", data_files=str(config.train_file), split="train").map(format_example)
    raw_eval = load_dataset("json", data_files=str(config.eval_file), split="train").map(format_example)
    model, tokenizer = build_model_and_tokenizer(config)
    train_dataset = tokenize_dataset(raw_train, tokenizer, config.max_length)
    eval_dataset = tokenize_dataset(raw_eval, tokenizer, config.max_length)

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        logging_steps=config.logging_steps,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        warmup_ratio=config.warmup_ratio,
        fp16=config.fp16,
        bf16=False,
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to=[],
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    with (config.output_dir / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.__dict__, handle, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
