from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from datasets import ClassLabel, Dataset, Features, Value, load_dataset
from torch.utils.data import WeightedRandomSampler
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from train_agent.data.adapters.common import NEGATIVE_VERIFIER_LABELS
from train_agent.eval.verifier_metrics import compute_verifier_metrics
from train_agent.models.verifier import build_pretrained_verifier, build_smoke_verifier
from train_agent.trainers.common import set_runtime_env

MODEL_INPUT_KEYS = ('input_ids', 'attention_mask', 'token_type_ids')


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights: Optional[List[float]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = None
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        loss, outputs = compute_classification_loss(model, inputs, class_weights=self.class_weights)
        if return_outputs:
            return loss, outputs
        return loss


class WeightedSamplingTrainer(Trainer):
    def __init__(self, *args, sample_weights: Optional[List[float]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_weights = sample_weights

    def _get_train_sampler(self):
        if self.sample_weights is None:
            return super()._get_train_sampler()
        if self.args.world_size > 1:
            raise ValueError('class_balance=sampler currently supports single-process training only')
        return WeightedRandomSampler(
            weights=torch.tensor(self.sample_weights, dtype=torch.double),
            num_samples=len(self.sample_weights),
            replacement=True,
        )


class PairwiseRankingTrainer(Trainer):
    def __init__(
        self,
        *args,
        train_data_collator,
        eval_data_collator,
        positive_label_indices: Sequence[int],
        negative_label_indices: Sequence[int],
        **kwargs,
    ):
        self.train_data_collator = train_data_collator
        self.eval_data_collator = eval_data_collator
        super().__init__(*args, data_collator=train_data_collator, **kwargs)
        self.positive_label_indices = [int(index) for index in positive_label_indices]
        self.negative_label_indices = [int(index) for index in negative_label_indices]

    def _with_data_collator(self, collator, loader_fn, *args, **kwargs):
        previous_collator = self.data_collator
        self.data_collator = collator
        try:
            return loader_fn(*args, **kwargs)
        finally:
            self.data_collator = previous_collator

    def get_train_dataloader(self):
        return self._with_data_collator(self.train_data_collator, super().get_train_dataloader)

    def get_eval_dataloader(self, eval_dataset=None):
        return self._with_data_collator(self.eval_data_collator, super().get_eval_dataloader, eval_dataset)

    def get_test_dataloader(self, test_dataset):
        return self._with_data_collator(self.eval_data_collator, super().get_test_dataloader, test_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        if 'positive_input_ids' in inputs:
            positive_outputs = model(**select_model_inputs(inputs, prefix='positive_'))
            negative_outputs = model(**select_model_inputs(inputs, prefix='negative_'))
            loss = compute_pairwise_ranking_loss(
                positive_logits=positive_outputs.get('logits'),
                negative_logits=negative_outputs.get('logits'),
                positive_label_indices=self.positive_label_indices,
                negative_label_indices=self.negative_label_indices,
            )
            if return_outputs:
                return loss, {
                    'positive_logits': positive_outputs.get('logits'),
                    'negative_logits': negative_outputs.get('logits'),
                }
            return loss

        loss, outputs = compute_classification_loss(model, inputs)
        if return_outputs:
            return loss, outputs
        return loss


class PairwiseDataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        positive_features = []
        negative_features = []
        lengths = []
        for feature in features:
            positive_feature = {
                key[len('positive_'):]: value
                for key, value in feature.items()
                if key.startswith('positive_') and key[len('positive_'):] in MODEL_INPUT_KEYS
            }
            negative_feature = {
                key[len('negative_'):]: value
                for key, value in feature.items()
                if key.startswith('negative_') and key[len('negative_'):] in MODEL_INPUT_KEYS
            }
            positive_features.append(positive_feature)
            negative_features.append(negative_feature)
            lengths.append(int(feature.get('length', 0)))

        positive_batch = self.tokenizer.pad(
            positive_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        negative_batch = self.tokenizer.pad(
            negative_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch = {f'positive_{key}': value for key, value in positive_batch.items()}
        batch.update({f'negative_{key}': value for key, value in negative_batch.items()})
        batch['length'] = torch.tensor(lengths, dtype=torch.long)
        return batch


def read_verifier_labels(path: Path):
    labels = set()
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            labels.add(row['label'])
    return sorted(labels)


def verifier_features(label_names):
    return Features(
        {
            'example_id': Value('string'),
            'sample_id': Value('string'),
            'dataset': Value('string'),
            'group_id': Value('string'),
            'claim': Value('string'),
            'evidence_text': Value('string'),
            'doc_id': Value('string'),
            'sentence_id': Value('int64'),
            'label': ClassLabel(names=label_names),
        }
    )


def load_verifier_datasets(train_file: Path, eval_file: Path):
    label_names = sorted(set(read_verifier_labels(train_file) + read_verifier_labels(eval_file)))
    features = verifier_features(label_names)
    train_dataset = load_dataset('json', data_files=str(train_file), split='train', features=features)
    eval_dataset = load_dataset('json', data_files=str(eval_file), split='train', features=features)
    return train_dataset, eval_dataset, label_names


def tokenize_verifier_dataset(dataset, tokenizer, max_length: int):
    def _tokenize(batch):
        encoded = tokenizer(
            batch['claim'],
            batch['evidence_text'],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        encoded['labels'] = batch['label']
        encoded['length'] = [len(input_ids) for input_ids in encoded['input_ids']]
        return encoded

    return dataset.map(_tokenize, batched=True, remove_columns=dataset.column_names)


def resolve_positive_negative_label_indices(label_names: Sequence[str]) -> Tuple[List[int], List[int]]:
    positive_indices = [
        index for index, label_name in enumerate(label_names) if str(label_name).upper() not in NEGATIVE_VERIFIER_LABELS
    ]
    negative_indices = [index for index in range(len(label_names)) if index not in positive_indices]
    return positive_indices, negative_indices


def _build_document_pairwise_examples(dataset, positive_label_indices: Sequence[int]):
    grouped_documents: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in dataset:
        group_id = str(row['group_id'])
        doc_id = str(row['doc_id'])
        documents = grouped_documents.setdefault(group_id, {})
        document = documents.setdefault(
            doc_id,
            {
                'claim': str(row['claim']),
                'doc_id': doc_id,
                'sentences': {},
                'is_positive': False,
            },
        )
        document['sentences'][int(row['sentence_id'])] = str(row['evidence_text'])
        if int(row['label']) in positive_label_indices:
            document['is_positive'] = True

    pair_examples = []
    positive_docs = 0
    negative_docs = 0
    groups_with_pairs = 0
    for group_id in sorted(grouped_documents):
        aggregated_documents = []
        for doc_id in sorted(grouped_documents[group_id]):
            document = grouped_documents[group_id][doc_id]
            ordered_sentences = [
                document['sentences'][sentence_id]
                for sentence_id in sorted(document['sentences'])
            ]
            aggregated_documents.append(
                {
                    'group_id': group_id,
                    'claim': document['claim'],
                    'doc_id': doc_id,
                    'text': '\n'.join(ordered_sentences),
                    'is_positive': bool(document['is_positive']),
                }
            )
        positives = [document for document in aggregated_documents if document['is_positive']]
        negatives = [document for document in aggregated_documents if not document['is_positive']]
        positive_docs += len(positives)
        negative_docs += len(negatives)
        if not positives or not negatives:
            continue
        groups_with_pairs += 1
        for positive_document in positives:
            for negative_document in negatives:
                pair_examples.append(
                    {
                        'pair_id': f"{group_id}:{positive_document['doc_id']}>{negative_document['doc_id']}",
                        'group_id': group_id,
                        'claim': positive_document['claim'],
                        'positive_text': positive_document['text'],
                        'negative_text': negative_document['text'],
                    }
                )

    return pair_examples, {
        'pairwise_level': 'document',
        'num_pairs': len(pair_examples),
        'num_groups_with_pairs': groups_with_pairs,
        'num_positive_examples': positive_docs,
        'num_negative_examples': negative_docs,
    }


def _build_sentence_pairwise_examples(dataset, positive_label_indices: Sequence[int]):
    grouped_examples: Dict[str, Dict[str, object]] = {}
    for row in dataset:
        group_id = str(row['group_id'])
        group = grouped_examples.setdefault(
            group_id,
            {
                'claim': str(row['claim']),
                'positives': [],
                'negatives': [],
            },
        )
        example = {
            'example_id': str(row['example_id']),
            'text': str(row['evidence_text']),
        }
        if int(row['label']) in positive_label_indices:
            group['positives'].append(example)
        else:
            group['negatives'].append(example)

    pair_examples = []
    positive_examples = 0
    negative_examples = 0
    groups_with_pairs = 0
    for group_id in sorted(grouped_examples):
        group = grouped_examples[group_id]
        positives = list(group['positives'])
        negatives = list(group['negatives'])
        positive_examples += len(positives)
        negative_examples += len(negatives)
        if not positives or not negatives:
            continue
        groups_with_pairs += 1
        for positive_example in positives:
            for negative_example in negatives:
                pair_examples.append(
                    {
                        'pair_id': f"{group_id}:{positive_example['example_id']}>{negative_example['example_id']}",
                        'group_id': group_id,
                        'claim': str(group['claim']),
                        'positive_text': positive_example['text'],
                        'negative_text': negative_example['text'],
                    }
                )

    return pair_examples, {
        'pairwise_level': 'sentence',
        'num_pairs': len(pair_examples),
        'num_groups_with_pairs': groups_with_pairs,
        'num_positive_examples': positive_examples,
        'num_negative_examples': negative_examples,
    }


def build_pairwise_ranking_examples(dataset, label_names: Sequence[str], pairwise_level: str = 'sentence'):
    positive_label_indices, negative_label_indices = resolve_positive_negative_label_indices(label_names)
    if len(positive_label_indices) != 1 or len(negative_label_indices) != 1:
        raise ValueError('pairwise ranking objective currently expects exactly one positive and one negative relevance label')
    if pairwise_level == 'document':
        return _build_document_pairwise_examples(dataset, positive_label_indices)
    if pairwise_level == 'sentence':
        return _build_sentence_pairwise_examples(dataset, positive_label_indices)
    raise ValueError(f'unsupported pairwise_level: {pairwise_level}')


def tokenize_pairwise_ranking_dataset(
    dataset,
    tokenizer,
    max_length: int,
    label_names: Sequence[str],
    pairwise_level: str = 'sentence',
):
    pair_examples, stats = build_pairwise_ranking_examples(dataset, label_names, pairwise_level=pairwise_level)
    if not pair_examples:
        raise ValueError('pairwise ranking objective requires at least one positive/negative pair')
    pair_dataset = Dataset.from_list(pair_examples)

    def _tokenize(batch):
        positive_encoded = tokenizer(
            batch['claim'],
            batch['positive_text'],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        negative_encoded = tokenizer(
            batch['claim'],
            batch['negative_text'],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        encoded = {f'positive_{key}': value for key, value in positive_encoded.items()}
        encoded.update({f'negative_{key}': value for key, value in negative_encoded.items()})
        encoded['length'] = [
            max(len(positive_ids), len(negative_ids))
            for positive_ids, negative_ids in zip(positive_encoded['input_ids'], negative_encoded['input_ids'])
        ]
        return encoded

    tokenized = pair_dataset.map(_tokenize, batched=True, remove_columns=pair_dataset.column_names)
    return tokenized, stats


def compute_balanced_class_weights(labels, num_labels: int) -> List[float]:
    counts = [0 for _ in range(num_labels)]
    for label in labels:
        counts[int(label)] += 1
    total = sum(counts)
    if total == 0 or num_labels == 0:
        return [1.0 for _ in range(num_labels)]
    weights = []
    for count in counts:
        weight = total / (num_labels * count) if count else 0.0
        weights.append(weight)
    mean_weight = sum(weights) / len(weights) if weights else 1.0
    if mean_weight == 0:
        return [1.0 for _ in range(num_labels)]
    return [round(weight / mean_weight, 6) for weight in weights]


def compute_example_sampling_weights(labels, num_labels: int) -> List[float]:
    class_weights = compute_balanced_class_weights(labels, num_labels)
    return [class_weights[int(label)] for label in labels]


def select_model_inputs(inputs, prefix: str = '') -> Dict[str, torch.Tensor]:
    model_inputs: Dict[str, torch.Tensor] = {}
    for key in MODEL_INPUT_KEYS:
        prefixed_key = f'{prefix}{key}'
        if prefixed_key in inputs:
            model_inputs[key] = inputs[prefixed_key]
    return model_inputs


def compute_classification_loss(model, inputs, class_weights: Optional[torch.Tensor] = None):
    labels = inputs['labels']
    outputs = model(**select_model_inputs(inputs))
    logits = outputs.get('logits')
    weights = class_weights.to(logits.device) if class_weights is not None else None
    loss = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
    return loss, outputs


def compute_positive_scores(
    logits: torch.Tensor,
    positive_label_indices: Sequence[int],
    negative_label_indices: Sequence[int],
) -> torch.Tensor:
    positive_scores = logits[:, list(positive_label_indices)].mean(dim=-1)
    if negative_label_indices:
        negative_scores = logits[:, list(negative_label_indices)].mean(dim=-1)
        return positive_scores - negative_scores
    return positive_scores


def compute_pairwise_ranking_loss(
    *,
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    positive_label_indices: Sequence[int],
    negative_label_indices: Sequence[int],
) -> torch.Tensor:
    positive_scores = compute_positive_scores(positive_logits, positive_label_indices, negative_label_indices)
    negative_scores = compute_positive_scores(negative_logits, positive_label_indices, negative_label_indices)
    margins = positive_scores - negative_scores
    return torch.nn.functional.softplus(-margins).mean()


def build_training_args(args: argparse.Namespace, use_cpu: bool) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        evaluation_strategy='steps',
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_strategy='steps',
        fp16=not use_cpu,
        bf16=False,
        no_cuda=use_cpu,
        dataloader_num_workers=0,
        remove_unused_columns=args.training_objective != 'pairwise',
        save_total_limit=2,
        report_to=[],
        group_by_length=args.class_balance != 'sampler',
        length_column_name='length',
        ddp_find_unused_parameters=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=Path, required=True)
    parser.add_argument('--eval_file', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--model_name_or_path', default='')
    parser.add_argument('--max_length', type=int, default=384)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--per_device_train_batch_size', type=int, default=8)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=8)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--eval_steps', type=int, default=50)
    parser.add_argument('--save_steps', type=int, default=50)
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--attn_implementation', default='sdpa')
    parser.add_argument('--class_balance', choices=['none', 'loss', 'sampler'], default='none')
    parser.add_argument('--training_objective', choices=['classification', 'pairwise'], default='classification')
    parser.add_argument('--pairwise_level', choices=['sentence', 'document'], default='sentence')
    parser.add_argument('--smoke_test', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_runtime_env()

    if args.training_objective == 'pairwise' and args.class_balance != 'none':
        raise ValueError('pairwise ranking objective currently requires class_balance=none')

    train_dataset_raw, eval_dataset_raw, label_names = load_verifier_datasets(args.train_file, args.eval_file)
    positive_label_indices, negative_label_indices = resolve_positive_negative_label_indices(label_names)
    if args.model_name_or_path:
        model, tokenizer = build_pretrained_verifier(
            model_name_or_path=args.model_name_or_path,
            num_labels=len(label_names),
            label_names=label_names,
            attn_implementation=args.attn_implementation,
        )
        use_cpu = False
    else:
        model, tokenizer = build_smoke_verifier(
            train_dataset=train_dataset_raw,
            eval_dataset=eval_dataset_raw,
            num_labels=len(label_names),
            max_length=args.max_length,
        )
        use_cpu = True

    pairwise_stats = None
    if args.training_objective == 'pairwise':
        train_dataset, pairwise_stats = tokenize_pairwise_ranking_dataset(
            train_dataset_raw,
            tokenizer,
            args.max_length,
            label_names,
            pairwise_level=args.pairwise_level,
        )
    else:
        train_dataset = tokenize_verifier_dataset(train_dataset_raw, tokenizer, args.max_length)
    eval_dataset = tokenize_verifier_dataset(eval_dataset_raw, tokenizer, args.max_length)
    training_args = build_training_args(args, use_cpu=use_cpu)

    labels = list(train_dataset_raw['label'])
    class_weights = None
    sample_weights = None
    if args.class_balance == 'loss':
        class_weights = compute_balanced_class_weights(labels, len(label_names))
        with (args.output_dir / 'class_weights.json').open('w', encoding='utf-8') as handle:
            json.dump(
                {
                    'class_balance': args.class_balance,
                    'label_names': label_names,
                    'class_weights': class_weights,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
    elif args.class_balance == 'sampler':
        sample_weights = compute_example_sampling_weights(labels, len(label_names))
        with (args.output_dir / 'sampling_weights.json').open('w', encoding='utf-8') as handle:
            json.dump(
                {
                    'class_balance': args.class_balance,
                    'label_names': label_names,
                    'class_weights': compute_balanced_class_weights(labels, len(label_names)),
                    'num_examples': len(sample_weights),
                    'weight_min': min(sample_weights) if sample_weights else 0.0,
                    'weight_max': max(sample_weights) if sample_weights else 0.0,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

    if args.training_objective == 'pairwise':
        train_data_collator = PairwiseDataCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
        eval_data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
        trainer = PairwiseRankingTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            train_data_collator=train_data_collator,
            eval_data_collator=eval_data_collator,
            positive_label_indices=positive_label_indices,
            negative_label_indices=negative_label_indices,
        )
    else:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
        if class_weights is not None:
            trainer = WeightedLossTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                class_weights=class_weights,
            )
        elif sample_weights is not None:
            trainer = WeightedSamplingTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                sample_weights=sample_weights,
            )
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )

    train_result = trainer.train()
    trainer.save_model()

    prediction_output = trainer.predict(eval_dataset)
    logits = prediction_output.predictions[0] if isinstance(prediction_output.predictions, tuple) else prediction_output.predictions
    eval_metrics = compute_verifier_metrics(
        logits=logits,
        labels=prediction_output.label_ids,
        label_names=label_names,
        group_ids=list(eval_dataset_raw['group_id']),
    )
    train_metrics = train_result.metrics
    if not trainer.is_world_process_zero():
        return

    tokenizer.save_pretrained(args.output_dir)
    with (args.output_dir / 'training_args.json').open('w', encoding='utf-8') as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2, default=str)
    with (args.output_dir / 'train_metrics.json').open('w', encoding='utf-8') as handle:
        json.dump(train_metrics, handle, ensure_ascii=False, indent=2, default=str)
    with (args.output_dir / 'eval_metrics.json').open('w', encoding='utf-8') as handle:
        json.dump(eval_metrics, handle, ensure_ascii=False, indent=2, default=str)
    with (args.output_dir / 'label_names.json').open('w', encoding='utf-8') as handle:
        json.dump(label_names, handle, ensure_ascii=False, indent=2)
    if pairwise_stats is not None:
        with (args.output_dir / 'pairwise_stats.json').open('w', encoding='utf-8') as handle:
            json.dump(pairwise_stats, handle, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {
                'output_dir': str(args.output_dir),
                'label_names': label_names,
                'class_balance': args.class_balance,
                'training_objective': args.training_objective,
                'pairwise_level': args.pairwise_level,
                'class_weights': class_weights,
                'sample_weights': {
                    'num_examples': len(sample_weights),
                    'weight_min': min(sample_weights),
                    'weight_max': max(sample_weights),
                } if sample_weights is not None else None,
                'pairwise_stats': pairwise_stats,
                'train_metrics': train_metrics,
                'eval_metrics': eval_metrics,
            },
            ensure_ascii=False,
        )
    )


if __name__ == '__main__':
    main()
