from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import ClassLabel, Features, Value, load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    PreTrainedTokenizerFast,
)

CLASSIFICATION_SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def set_runtime_env() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def read_action_labels(path: Path) -> List[str]:
    labels = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            labels.add(row["label"])
    return sorted(labels)


def classification_features(label_names: List[str]) -> Features:
    return Features(
        {
            "trajectory_id": Value("string"),
            "step_id": Value("int64"),
            "task": Value("string"),
            "text": Value("string"),
            "label": ClassLabel(names=label_names),
            "label_text": Value("string"),
        }
    )


def load_classification_datasets(train_file: Path, eval_file: Path):
    label_names = sorted(set(read_action_labels(train_file) + read_action_labels(eval_file)))
    features = classification_features(label_names)
    train_dataset = load_dataset("json", data_files=str(train_file), split="train", features=features)
    eval_dataset = load_dataset("json", data_files=str(eval_file), split="train", features=features)
    return train_dataset, eval_dataset, label_names


def build_smoke_classifier_tokenizer(train_dataset, eval_dataset) -> PreTrainedTokenizerFast:
    texts = list(train_dataset["text"]) + list(eval_dataset["text"])
    backend = Tokenizer(WordLevel(unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=CLASSIFICATION_SPECIAL_TOKENS)
    backend.train_from_iterator(texts, trainer=trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    return tokenizer


def tokenize_classification_dataset(dataset, tokenizer, max_length: int):
    def _tokenize(batch: Dict[str, List]):
        encoded = tokenizer(batch["text"], truncation=True, max_length=max_length, padding=False)
        encoded["labels"] = batch["label"]
        return encoded

    return dataset.map(_tokenize, batched=True, remove_columns=dataset.column_names)


def build_smoke_classifier(train_dataset, eval_dataset, num_labels: int, max_length: int) -> Tuple[BertForSequenceClassification, PreTrainedTokenizerFast]:
    tokenizer = build_smoke_classifier_tokenizer(train_dataset, eval_dataset)
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=max_length + 2,
        num_labels=num_labels,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = BertForSequenceClassification(config)
    return model, tokenizer


def build_pretrained_classifier(model_name_or_path: str, num_labels: int):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    return model, tokenizer
