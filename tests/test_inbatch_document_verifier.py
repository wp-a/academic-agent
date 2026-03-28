import tempfile
import unittest

import torch
from datasets import Dataset
from transformers import DataCollatorWithPadding, TrainingArguments

from train_agent.models.verifier import build_smoke_verifier
from train_agent.trainers.train_verifier import (
    InBatchDocumentDataCollator,
    InBatchDocumentRankingTrainer,
    build_inbatch_document_ranking_dataset,
    build_inbatch_document_ranking_examples,
    compute_inbatch_document_ranking_loss,
    tokenize_verifier_dataset,
)


class InBatchDocumentVerifierTest(unittest.TestCase):
    def test_build_inbatch_document_ranking_examples_aggregates_positive_documents(self):
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
                    'evidence_text': 'Neutral other doc sentence.',
                    'doc_id': 'doc-b',
                    'sentence_id': 0,
                    'label': 0,
                },
                {
                    'example_id': 'ex-4',
                    'sample_id': 'sample-2',
                    'dataset': 'scifact',
                    'group_id': 'sample-2',
                    'claim': 'Claim B',
                    'evidence_text': 'Relevant sentence B.',
                    'doc_id': 'doc-c',
                    'sentence_id': 0,
                    'label': 1,
                },
            ]
        )

        examples, stats = build_inbatch_document_ranking_examples(dataset, ['NEUTRAL', 'RELEVANT'])

        self.assertEqual(stats['objective'], 'inbatch_document')
        self.assertEqual(stats['num_examples'], 2)
        self.assertEqual(stats['num_groups'], 2)
        self.assertEqual(stats['groups_with_multiple_positive_docs'], 0)
        self.assertEqual(examples[0]['document_text'], 'Relevant first sentence.\nNeutral same doc sentence.')

    def test_compute_inbatch_document_ranking_loss_prefers_matching_docs(self):
        positive_mask = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
        low_loss = compute_inbatch_document_ranking_loss(
            logits=torch.tensor(
                [
                    [[0.1, 2.0], [2.0, 0.1]],
                    [[2.0, 0.1], [0.1, 2.0]],
                ],
                dtype=torch.float32,
            ),
            positive_mask=positive_mask,
            positive_label_indices=[1],
            negative_label_indices=[0],
        )
        high_loss = compute_inbatch_document_ranking_loss(
            logits=torch.tensor(
                [
                    [[2.0, 0.1], [0.1, 2.0]],
                    [[0.1, 2.0], [2.0, 0.1]],
                ],
                dtype=torch.float32,
            ),
            positive_mask=positive_mask,
            positive_label_indices=[1],
            negative_label_indices=[0],
        )
        self.assertLess(float(low_loss), float(high_loss))

    def test_inbatch_document_dataset_adds_length(self):
        train_raw = Dataset.from_list(
            [
                {
                    'example_id': 'train-pos-a',
                    'sample_id': 'sample-1',
                    'dataset': 'scifact',
                    'group_id': 'sample-1',
                    'claim': 'Claim A',
                    'evidence_text': 'Relevant evidence A.',
                    'doc_id': 'doc-a',
                    'sentence_id': 0,
                    'label': 1,
                },
                {
                    'example_id': 'train-neg-a',
                    'sample_id': 'sample-1',
                    'dataset': 'scifact',
                    'group_id': 'sample-1',
                    'claim': 'Claim A',
                    'evidence_text': 'Distractor A.',
                    'doc_id': 'doc-b',
                    'sentence_id': 0,
                    'label': 0,
                },
                {
                    'example_id': 'train-pos-b',
                    'sample_id': 'sample-2',
                    'dataset': 'scifact',
                    'group_id': 'sample-2',
                    'claim': 'Claim B',
                    'evidence_text': 'Relevant evidence B.',
                    'doc_id': 'doc-c',
                    'sentence_id': 0,
                    'label': 1,
                },
                {
                    'example_id': 'train-neg-b',
                    'sample_id': 'sample-2',
                    'dataset': 'scifact',
                    'group_id': 'sample-2',
                    'claim': 'Claim B',
                    'evidence_text': 'Distractor B.',
                    'doc_id': 'doc-d',
                    'sentence_id': 0,
                    'label': 0,
                },
            ]
        )
        eval_raw = Dataset.from_list(
            [
                {
                    'example_id': 'eval-pos',
                    'sample_id': 'sample-3',
                    'dataset': 'scifact',
                    'group_id': 'sample-3',
                    'claim': 'Claim C',
                    'evidence_text': 'Relevant eval evidence.',
                    'doc_id': 'doc-e',
                    'sentence_id': 0,
                    'label': 1,
                }
            ]
        )
        _, tokenizer = build_smoke_verifier(
            train_dataset=train_raw,
            eval_dataset=eval_raw,
            num_labels=2,
            max_length=64,
        )
        train_dataset, stats = build_inbatch_document_ranking_dataset(
            train_raw,
            tokenizer,
            64,
            ['NEUTRAL', 'RELEVANT'],
        )

        self.assertEqual(stats['num_examples'], 2)
        self.assertIn('document_text', train_dataset.column_names)
        self.assertIn('length', train_dataset.column_names)

    def test_inbatch_document_trainer_uses_pointwise_eval_collator(self):
        train_raw = Dataset.from_list(
            [
                {
                    'example_id': 'train-pos-a',
                    'sample_id': 'sample-1',
                    'dataset': 'scifact',
                    'group_id': 'sample-1',
                    'claim': 'Claim A',
                    'evidence_text': 'Relevant evidence A.',
                    'doc_id': 'doc-a',
                    'sentence_id': 0,
                    'label': 1,
                },
                {
                    'example_id': 'train-neg-a',
                    'sample_id': 'sample-1',
                    'dataset': 'scifact',
                    'group_id': 'sample-1',
                    'claim': 'Claim A',
                    'evidence_text': 'Distractor A.',
                    'doc_id': 'doc-b',
                    'sentence_id': 0,
                    'label': 0,
                },
                {
                    'example_id': 'train-pos-b',
                    'sample_id': 'sample-2',
                    'dataset': 'scifact',
                    'group_id': 'sample-2',
                    'claim': 'Claim B',
                    'evidence_text': 'Relevant evidence B.',
                    'doc_id': 'doc-c',
                    'sentence_id': 0,
                    'label': 1,
                },
                {
                    'example_id': 'train-neg-b',
                    'sample_id': 'sample-2',
                    'dataset': 'scifact',
                    'group_id': 'sample-2',
                    'claim': 'Claim B',
                    'evidence_text': 'Distractor B.',
                    'doc_id': 'doc-d',
                    'sentence_id': 0,
                    'label': 0,
                },
            ]
        )
        eval_raw = Dataset.from_list(
            [
                {
                    'example_id': 'eval-pos',
                    'sample_id': 'sample-3',
                    'dataset': 'scifact',
                    'group_id': 'sample-3',
                    'claim': 'Claim C',
                    'evidence_text': 'Relevant eval evidence.',
                    'doc_id': 'doc-e',
                    'sentence_id': 0,
                    'label': 1,
                },
                {
                    'example_id': 'eval-neg',
                    'sample_id': 'sample-3',
                    'dataset': 'scifact',
                    'group_id': 'sample-3',
                    'claim': 'Claim C',
                    'evidence_text': 'Distractor eval evidence.',
                    'doc_id': 'doc-f',
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
        inbatch_train, stats = build_inbatch_document_ranking_dataset(
            train_raw,
            tokenizer,
            64,
            ['NEUTRAL', 'RELEVANT'],
        )
        pointwise_eval = tokenize_verifier_dataset(eval_raw, tokenizer, max_length=64)

        with tempfile.TemporaryDirectory() as tmpdir:
            training_args = TrainingArguments(
                output_dir=tmpdir,
                per_device_train_batch_size=2,
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
            trainer = InBatchDocumentRankingTrainer(
                model=model,
                args=training_args,
                train_dataset=inbatch_train,
                eval_dataset=pointwise_eval,
                tokenizer=tokenizer,
                train_data_collator=InBatchDocumentDataCollator(tokenizer=tokenizer, max_length=64),
                eval_data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                positive_label_indices=[1],
                negative_label_indices=[0],
            )
            metrics = trainer.evaluate()

        self.assertEqual(stats['num_examples'], 2)
        self.assertIn('eval_loss', metrics)


if __name__ == '__main__':
    unittest.main()
