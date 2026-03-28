from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from peft import AutoPeftModelForSequenceClassification
except ImportError:  # pragma: no cover
    AutoPeftModelForSequenceClassification = None


class FrozenActionPolicy:
    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        max_length: int = 256,
        batch_size: int = 8,
        attn_implementation: str = "sdpa",
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.max_length = max_length
        self.batch_size = max(1, batch_size)
        self.label_names = self._load_label_names(self.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
        self.model = self._load_model(self.model_name_or_path, attn_implementation=attn_implementation)
        if getattr(self.model.config, "pad_token_id", None) is None and self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @staticmethod
    def _load_label_names(model_name_or_path: str) -> List[str]:
        label_path = Path(model_name_or_path) / "label_names.json"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label_names.json under {model_name_or_path}")
        with label_path.open("r", encoding="utf-8") as handle:
            return list(json.load(handle))

    @staticmethod
    def _is_peft_adapter(model_name_or_path: str) -> bool:
        return (Path(model_name_or_path) / "adapter_config.json").exists()

    def _load_model(self, model_name_or_path: str, *, attn_implementation: str):
        model_kwargs = {}
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        if self._is_peft_adapter(model_name_or_path) and AutoPeftModelForSequenceClassification is not None:
            loader = AutoPeftModelForSequenceClassification.from_pretrained
            model_kwargs["num_labels"] = len(self.label_names)
            model_kwargs["ignore_mismatched_sizes"] = True
        else:
            loader = AutoModelForSequenceClassification.from_pretrained
        try:
            return loader(model_name_or_path, **model_kwargs)
        except (TypeError, ValueError):
            model_kwargs.pop("attn_implementation", None)
            return loader(model_name_or_path, **model_kwargs)

    def predict_logits(self, texts: Sequence[str]) -> List[List[float]]:
        all_logits: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            encoded = self.tokenizer(
                batch,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = self.model(**encoded)
            logits = outputs.logits.detach().cpu().tolist()
            all_logits.extend([[float(value) for value in row] for row in logits])
        return all_logits

    def predict_action_ids(self, texts: Sequence[str]) -> List[int]:
        logits = self.predict_logits(texts)
        return [max(range(len(row)), key=lambda idx: row[idx]) for row in logits]

    def predict_actions(self, texts: Sequence[str]) -> List[str]:
        return [self.label_names[idx] for idx in self.predict_action_ids(texts)]

    def predict_action(self, text: str) -> str:
        return self.predict_actions([text])[0]
