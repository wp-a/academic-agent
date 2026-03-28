from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
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

from train_agent.data.adapters.common import NEGATIVE_VERIFIER_LABELS
from train_agent.trainers.common import CLASSIFICATION_SPECIAL_TOKENS


def build_smoke_verifier_tokenizer(train_dataset, eval_dataset) -> PreTrainedTokenizerFast:
    texts: List[str] = []
    for dataset in (train_dataset, eval_dataset):
        texts.extend(str(item) for item in dataset["claim"])
        texts.extend(str(item) for item in dataset["evidence_text"])
    backend = Tokenizer(WordLevel(unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=CLASSIFICATION_SPECIAL_TOKENS)
    backend.train_from_iterator(texts, trainer=trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


def _configure_label_maps(model, label_names: List[str]) -> None:
    model.config.label2id = {label: idx for idx, label in enumerate(label_names)}
    model.config.id2label = {idx: label for idx, label in enumerate(label_names)}


def _load_sequence_classifier(model_name_or_path: str, kwargs: Dict[str, object]):
    load_kwargs = dict(kwargs)
    try:
        return AutoModelForSequenceClassification.from_pretrained(model_name_or_path, **load_kwargs)
    except TypeError:
        load_kwargs.pop("attn_implementation", None)
        return AutoModelForSequenceClassification.from_pretrained(model_name_or_path, **load_kwargs)
    except ValueError as exc:
        message = str(exc)
        if load_kwargs.get("attn_implementation") == "sdpa" and "does not support an attention implementation through torch.nn.functional.scaled_dot_product_attention yet" in message:
            load_kwargs["attn_implementation"] = "eager"
            return AutoModelForSequenceClassification.from_pretrained(model_name_or_path, **load_kwargs)
        raise


def build_smoke_verifier(train_dataset, eval_dataset, num_labels: int, max_length: int) -> Tuple[BertForSequenceClassification, PreTrainedTokenizerFast]:
    tokenizer = build_smoke_verifier_tokenizer(train_dataset, eval_dataset)
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


def build_pretrained_verifier(model_name_or_path: str, num_labels: int, label_names: List[str], attn_implementation: str = "sdpa"):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    kwargs: Dict[str, object] = {
        "num_labels": num_labels,
        "ignore_mismatched_sizes": True,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = _load_sequence_classifier(model_name_or_path, kwargs)
    _configure_label_maps(model, label_names)
    return model, tokenizer


class FrozenSequenceVerifier:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        attn_implementation: str = "sdpa",
        max_length: int = 384,
        batch_size: int = 8,
        device: Optional[str] = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        kwargs: Dict[str, object] = {}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.model = _load_sequence_classifier(model_name_or_path, kwargs)
        self.model.eval()
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.positive_indices = self._resolve_positive_indices()

    def _resolve_positive_indices(self) -> List[int]:
        label_names: List[str] = []
        label2id = getattr(self.model.config, "label2id", {}) or {}
        if label2id:
            label_names = ["" for _ in range(len(label2id))]
            for label_name, label_idx in label2id.items():
                label_names[int(label_idx)] = str(label_name)
        else:
            id2label = getattr(self.model.config, "id2label", {}) or {}
            if id2label:
                label_names = ["" for _ in range(len(id2label))]
                for label_idx, label_name in id2label.items():
                    label_names[int(label_idx)] = str(label_name)
        positive_indices = [
            idx for idx, label_name in enumerate(label_names) if label_name.upper() not in NEGATIVE_VERIFIER_LABELS
        ]
        if positive_indices:
            return positive_indices
        return list(range(int(self.model.config.num_labels)))

    @torch.inference_mode()
    def _score_texts(self, claim: str, texts: Sequence[str]) -> List[float]:
        if not texts:
            return []
        scores: List[float] = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = list(texts[start : start + self.batch_size])
            encoded = self.tokenizer(
                [claim] * len(batch_texts),
                batch_texts,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            logits = self.model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)
            positive_scores = probabilities[:, self.positive_indices].sum(dim=-1)
            scores.extend(float(score) for score in positive_scores.tolist())
        return scores

    def _aggregate_sentence_scores(
        self,
        sentence_scores: Sequence[float],
        *,
        aggregation: str,
        aggregation_top_k: int,
    ) -> float:
        ranked_scores = sorted((float(score) for score in sentence_scores), reverse=True)
        if not ranked_scores:
            return 0.0
        if aggregation == "max":
            return ranked_scores[0]
        if aggregation == "top_2_mean":
            top_scores = ranked_scores[:2]
            return sum(top_scores) / len(top_scores)
        if aggregation == "top_k_weighted_mean":
            top_scores = ranked_scores[: max(1, aggregation_top_k)]
            weights = list(range(len(top_scores), 0, -1))
            return sum(value * weight for value, weight in zip(top_scores, weights)) / sum(weights)
        if aggregation == "logsumexp":
            return float(torch.logsumexp(torch.tensor(ranked_scores, dtype=torch.float32), dim=0).item())
        raise ValueError(f"Unsupported aggregation: {aggregation}")

    @torch.inference_mode()
    def score_documents(self, claim: str, documents: Dict[str, str]) -> Dict[str, float]:
        items = list(documents.items())
        if not items:
            return {}
        scores = self._score_texts(claim, [text for _, text in items])
        return {str(doc_id): float(score) for (doc_id, _), score in zip(items, scores)}

    @torch.inference_mode()
    def score_document_sentences(
        self,
        claim: str,
        documents: Dict[str, List[str]],
        *,
        aggregation: str = "max",
        aggregation_top_k: int = 3,
    ) -> Dict[str, float]:
        flattened: List[Tuple[str, str]] = []
        sentence_scores_by_doc: Dict[str, List[float]] = {str(doc_id): [] for doc_id in documents}
        for doc_id, sentences in documents.items():
            for sentence in sentences:
                text = str(sentence).strip()
                if text:
                    flattened.append((str(doc_id), text))
        if flattened:
            scores = self._score_texts(claim, [text for _, text in flattened])
            for (doc_id, _), score in zip(flattened, scores):
                sentence_scores_by_doc[str(doc_id)].append(float(score))
        return {
            str(doc_id): self._aggregate_sentence_scores(
                sentence_scores_by_doc.get(str(doc_id), []),
                aggregation=aggregation,
                aggregation_top_k=aggregation_top_k,
            )
            for doc_id in documents
        }
