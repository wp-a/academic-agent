import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from datasets import Dataset
from transformers import DataCollatorWithPadding, TrainingArguments

from train_agent.models.verifier import build_pretrained_verifier, build_smoke_verifier
from train_agent.trainers.train_verifier import (
    PairwiseDataCollator,
    PairwiseRankingTrainer,
    build_pairwise_ranking_examples,
    build_training_args,
    compute_balanced_class_weights,
    compute_example_sampling_weights,
    compute_pairwise_ranking_loss,
    load_verifier_datasets,
    tokenize_pairwise_ranking_dataset,
    tokenize_verifier_dataset,
)


class VerifierTrainerTest(unittest.TestCase):
    def test_load_and_tokenize_verifier_datasets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_file = root / 'train.jsonl'
            eval_file = root / 'eval.jsonl'
            train_rows = [
                {
                    'example_id': 'ex-1',
                    'sample_id': 'sample-1',
                    'dataset': 'scifact',
                    'group_id': 'sample-1',
                    'claim': 'Claim A',
                    'evidence_text': 'Support sentence A',
                    'doc_id': 'doc-a',
                    'sentence_id': 0,
                    'label': 'SUPPORT',
                },
                {
                    'example_id': 'ex-2',
                    'sample_id': 'sample-1',
                    'dataset': 'scifact',
                    'group_id': 'sample-1',
                    'claim': 'Claim A',
                    'evidence_text': 'Distractor sentence',
                    'doc_id': 'doc-b',
                    'sentence_id': 0,
                    'label': 'NEUTRAL',
                },
            ]
            eval_rows = [
                {
                    'example_id': 'ex-3',
                    'sample_id': 'sample-2',
                    'dataset': 'fever',
                    'group_id': 'sample-2',
                    'claim': 'Claim B',
                    'evidence_text': 'Refuting sentence B',
                    'doc_id': 'doc-c',
                    'sentence_id': 1,
                    'label': 'CONTRADICT',
                }
            ]
            train_file.write_text(
                '\n'.join(json.dumps(row, ensure_ascii=False) for row in train_rows) + '\n',
                encoding='utf-8',
            )
            eval_file.write_text(
                '\n'.join(json.dumps(row, ensure_ascii=False) for row in eval_rows) + '\n',
                encoding='utf-8',
            )

            train_dataset, eval_dataset, label_names = load_verifier_datasets(train_file, eval_file)
            self.assertEqual(label_names, ['CONTRADICT', 'NEUTRAL', 'SUPPORT'])
            self.assertEqual(train_dataset[0]['claim'], 'Claim A')
            self.assertIn('evidence_text', train_dataset.column_names)

            _, tokenizer = build_smoke_verifier(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                num_labels=len(label_names),
                max_length=64,
            )
            tokenized = tokenize_verifier_dataset(train_dataset, tokenizer, max_length=64)
            self.assertIn('input_ids', tokenized.column_names)
            self.assertIn('labels', tokenized.column_names)

    def test_build_pretrained_verifier_falls_back_to_eager_when_sdpa_is_unsupported(self):
        class FakeTokenizer:
            def __init__(self):
                self.pad_token = None
                self.eos_token = '[EOS]'
                self.unk_token = '[UNK]'

        class FakeModel:
            def __init__(self):
                self.config = SimpleNamespace(label2id={}, id2label={})

        fake_tokenizer = FakeTokenizer()
        fake_model = FakeModel()
        calls = []

        def fake_from_pretrained(*args, **kwargs):
            calls.append(kwargs.get('attn_implementation'))
            if kwargs.get('attn_implementation') == 'sdpa':
                raise ValueError('does not support an attention implementation through torch.nn.functional.scaled_dot_product_attention yet')
            return fake_model

        with patch('train_agent.models.verifier.AutoTokenizer.from_pretrained', return_value=fake_tokenizer):
            with patch('train_agent.models.verifier.AutoModelForSequenceClassification.from_pretrained', side_effect=fake_from_pretrained):
                model, tokenizer = build_pretrained_verifier(
                    model_name_or_path='microsoft/deberta-v3-large',
                    num_labels=3,
                    label_names=['CONTRADICT', 'NEUTRAL', 'SUPPORT'],
                    attn_implementation='sdpa',
                )

        self.assertIs(model, fake_model)
        self.assertIs(tokenizer, fake_tokenizer)
        self.assertEqual(fake_tokenizer.pad_token, '[EOS]')
        self.assertEqual(calls, ['sdpa', 'eager'])
        self.assertEqual(model.config.label2id['SUPPORT'], 2)

    def test_compute_balanced_class_weights_upweights_minority_labels(self):
        weights = compute_balanced_class_weights([0, 1, 1, 1, 2, 2], num_labels=3)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[2], weights[1])
        self.assertAlmostEqual(sum(weights) / len(weights), 1.0, places=6)

    def test_compute_example_sampling_weights_upweights_minority_examples(self):
        weights = compute_example_sampling_weights([0, 1, 1, 1, 2, 2], num_labels=3)
        self.assertEqual(len(weights), 6)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[4], weights[1])

    def test_build_pairwise_ranking_examples_defaults_to_sentence_level(self):
        dataset = Dataset.from_list(
            [
                {
                    'example_id': 'ex-1',
                    'sample_id': 'sample-1',
                    'dataset': 'scifact',
                    'group_id': 'sample-1',
                    'claim': 'Claim A',
                    'evidence_text': 'Relevant first sentence.',
                    'doc_id': 'doc-a',
                    'sentence_id': 0,
                    'label': 1,
                },
                {
                    'example_id': 'ex-2',
                    'sample_id': 'sample-1',
                    'dataset': 'scifact',
                    'group_id': 'sample-1',
                    'claim': 'Claim A',
                    'evidence_text': 'Neutral same doc sentence.',
                    'doc_id': 'doc-a',
                    'sentence_id': 1,
                    'label': 0,
                },
                {
                    'example_id': 'ex-3',
                    'sample_id': 'sample-1',
                    'dataset': 'scifact',
                    'group_id': 'sample-1',
                    'claim': 'Claim A',
                    'evidence_text': 'Neutral sentence B.',
                    'doc_id': 'doc-b',
                    'sentence_id': 0,
                    'label': 0,
                },
            ]
        )

        pairs, stats = build_pairwise_ranking_examples(dataset, ['NEUTRAL', 'RELEVANT'])
        self.assertEqual(stats['pairwise_level'], 'sentence')
        self.assertEqual(stats['num_pairs'], 2)
        self.assertEqual(stats['num_groups_with_pairs'], 1)
        self.assertEqual(stats['num_positive_examples'], 1)
        self.assertEqual(stats['num_negative_examples'], 2)
        self.assertEqual(pairs[0]['positive_text'], 'Relevant first sentence.')
        self.assertTrue(pairs[0]['pair_id'].startswith('sample-1:ex-1>'))

    def test_build_pairwise_ranking_examples_can_aggregate_documents(self):
        dataset = Dataset.from_list(
            [
                {
                    'example_id': 'ex-1',
                    'sample_id': 'sample-1',
                    'dataset': 'scifact',
                    'group_id': 'sample-1',
                    'claim': 'Claim A',
                    'evidence_text': 'Relevant first sentence.',
                    'doc_id': 'doc-a',
                    'sentence_id': 0,
                    'label': 1,
                },
                {
                    'example_id': 'ex-2',
                    'sample_id': 'sample-1',
                    'dataset': 'scifact',
                    'group_id': 'sample-1',
                    'claim': 'Claim A',
                    'evidence_text': 'Relevant second sentence.',
                    'doc_id': 'doc-a',
                    'sentence_id': 1,
                    'label': 0,
                },
                {
                    'example_id': 'ex-3',
                    'sample_id': 'sample-1',
                    'dataset': 'scifact',
                    'group_id': 'sample-1',
                    'claim': 'Claim A',
                    'evidence_text': 'Neutral sentence B.',
                    'doc_id': 'doc-b',
                    'sentence_id': 0,
                    'label': 0,
                },
            ]
        )

        pairs, stats = build_pairwise_ranking_examples(dataset, ['NEUTRAL', 'RELEVANT'], pairwise_level='document')
        self.assertEqual(stats['pairwise_level'], 'document')
        self.assertEqual(stats['num_pairs'], 1)
        self.assertEqual(stats['num_groups_with_pairs'], 1)
        self.assertEqual(pairs[0]['positive_text'], 'Relevant first sentence.\nRelevant second sentence.')

    def test_compute_pairwise_ranking_loss_prefers_correct_ordering(self):
        low_loss = compute_pairwise_ranking_loss(
            positive_logits=torch.tensor([[0.1, 2.0]], dtype=torch.float32),
            negative_logits=torch.tensor([[1.5, 0.2]], dtype=torch.float32),
            positive_label_indices=[1],
            negative_label_indices=[0],
        )
        high_loss = compute_pairwise_ranking_loss(
            positive_logits=torch.tensor([[1.5, 0.2]], dtype=torch.float32),
            negative_logits=torch.tensor([[0.1, 2.0]], dtype=torch.float32),
            positive_label_indices=[1],
            negative_label_indices=[0],
        )
        self.assertLess(float(low_loss), float(high_loss))

    def test_pairwise_trainer_uses_pointwise_eval_collator(self):
        train_raw = Dataset.from_list(
            [
                {
                    'example_id': 'train-pos',
                    'sample_id': 'sample-1',
                    'dataset': 'scifact',
                    'group_id': 'sample-1',
                    'claim': 'Claim A',
                    'evidence_text': 'Relevant evidence.',
                    'doc_id': 'doc-a',
                    'sentence_id': 0,
                    'label': 1,
                },
                {
                    'example_id': 'train-neg',
                    'sample_id': 'sample-1',
                    'dataset': 'scifact',
                    'group_id': 'sample-1',
                    'claim': 'Claim A',
                    'evidence_text': 'Distractor sentence.',
                    'doc_id': 'doc-b',
                    'sentence_id': 0,
                    'label': 0,
                },
            ]
        )
        eval_raw = Dataset.from_list(
            [
                {
                    'example_id': 'eval-pos',
                    'sample_id': 'sample-2',
                    'dataset': 'scifact',
                    'group_id': 'sample-2',
                    'claim': 'Claim B',
                    'evidence_text': 'Relevant eval evidence.',
                    'doc_id': 'doc-c',
                    'sentence_id': 0,
                    'label': 1,
                },
                {
                    'example_id': 'eval-neg',
                    'sample_id': 'sample-2',
                    'dataset': 'scifact',
                    'group_id': 'sample-2',
                    'claim': 'Claim B',
                    'evidence_text': 'Distractor eval sentence.',
                    'doc_id': 'doc-d',
                    'sentence_id': 0,
                    'label': 0,
                },
            ]
        )
        model, tokenizer = build_smoke_verifier(
            train_dataset=train_raw,
            eval_dataset=eval_raw,
            num_labels=2,
            max_length=64,
        )
        pairwise_train, _ = tokenize_pairwise_ranking_dataset(
            train_raw,
            tokenizer,
            64,
            ['NEUTRAL', 'RELEVANT'],
            pairwise_level='sentence',
        )
        pointwise_eval = tokenize_verifier_dataset(eval_raw, tokenizer, max_length=64)

        with tempfile.TemporaryDirectory() as tmpdir:
            training_args = TrainingArguments(
                output_dir=tmpdir,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                learning_rate=1e-3,
                num_train_epochs=1,
                max_steps=1,
                logging_steps=1,
                eval_steps=1,
                save_steps=1,
                evaluation_strategy='no',
                fp16=False,
                bf16=False,
                no_cuda=True,
                dataloader_num_workers=0,
                remove_unused_columns=False,
                report_to=[],
            )
            trainer = PairwiseRankingTrainer(
                model=model,
                args=training_args,
                train_dataset=pairwise_train,
                eval_dataset=pointwise_eval,
                tokenizer=tokenizer,
                train_data_collator=PairwiseDataCollator(tokenizer=tokenizer),
                eval_data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                positive_label_indices=[1],
                negative_label_indices=[0],
            )
            metrics = trainer.evaluate()

        self.assertIn('eval_loss', metrics)

    def test_build_training_args_disables_find_unused_parameters_for_gpu(self):
        args = argparse.Namespace(
            output_dir=Path('/tmp/verifier-test'),
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            learning_rate=2e-5,
            num_train_epochs=1,
            max_steps=-1,
            logging_steps=10,
            eval_steps=10,
            save_steps=10,
            class_balance='none',
            training_objective='classification',
        )
        training_args = build_training_args(args, use_cpu=False)
        self.assertFalse(training_args.ddp_find_unused_parameters)
        self.assertTrue(training_args.group_by_length)
        self.assertTrue(training_args.remove_unused_columns)

    def test_build_training_args_disables_length_grouping_for_sampler(self):
        args = argparse.Namespace(
            output_dir=Path('/tmp/verifier-test'),
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            learning_rate=2e-5,
            num_train_epochs=1,
            max_steps=-1,
            logging_steps=10,
            eval_steps=10,
            save_steps=10,
            class_balance='sampler',
            training_objective='classification',
        )
        training_args = build_training_args(args, use_cpu=False)
        self.assertFalse(training_args.group_by_length)
        self.assertTrue(training_args.remove_unused_columns)

    def test_build_training_args_keeps_pairwise_columns(self):
        args = argparse.Namespace(
            output_dir=Path('/tmp/verifier-test'),
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            learning_rate=2e-5,
            num_train_epochs=1,
            max_steps=-1,
            logging_steps=10,
            eval_steps=10,
            save_steps=10,
            class_balance='none',
            training_objective='pairwise',
        )
        training_args = build_training_args(args, use_cpu=False)
        self.assertFalse(training_args.remove_unused_columns)
        self.assertTrue(training_args.group_by_length)


if __name__ == '__main__':
    unittest.main()
