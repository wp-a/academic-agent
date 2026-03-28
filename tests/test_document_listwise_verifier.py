import tempfile
import unittest

import torch
from datasets import Dataset
from transformers import DataCollatorWithPadding, TrainingArguments

from train_agent.models.verifier import build_smoke_verifier
from train_agent.trainers.train_verifier import (
    DocumentListwiseDataCollator,
    DocumentListwiseRankingTrainer,
    build_document_listwise_ranking_dataset,
    build_document_listwise_ranking_examples,
    compute_document_listwise_ranking_loss,
    tokenize_verifier_dataset,
)


class DocumentListwiseVerifierTest(unittest.TestCase):
    def test_build_document_listwise_ranking_examples_keeps_mixed_groups(self):
        dataset = Dataset.from_list(
            [
                {
                    "example_id": "ex-1",
                    "sample_id": "sample-1",
                    "dataset": "scifact",
                    "group_id": "sample-1",
                    "claim": "Claim A",
                    "evidence_text": "Relevant sentence A1.",
                    "doc_id": "doc-a",
                    "sentence_id": 0,
                    "label": 1,
                },
                {
                    "example_id": "ex-2",
                    "sample_id": "sample-1",
                    "dataset": "scifact",
                    "group_id": "sample-1",
                    "claim": "Claim A",
                    "evidence_text": "Neutral sentence A2.",
                    "doc_id": "doc-b",
                    "sentence_id": 0,
                    "label": 0,
                },
                {
                    "example_id": "ex-3",
                    "sample_id": "sample-2",
                    "dataset": "scifact",
                    "group_id": "sample-2",
                    "claim": "Claim B",
                    "evidence_text": "Relevant sentence B1.",
                    "doc_id": "doc-c",
                    "sentence_id": 0,
                    "label": 1,
                },
            ]
        )

        groups, stats = build_document_listwise_ranking_examples(dataset, ["NEUTRAL", "RELEVANT"])

        self.assertEqual(stats["objective"], "document_listwise")
        self.assertEqual(stats["num_groups"], 1)
        self.assertEqual(stats["num_docs"], 2)
        self.assertEqual(groups[0]["claim"], "Claim A")
        self.assertEqual(groups[0]["documents"][0]["doc_id"], "doc-a")
        self.assertTrue(groups[0]["documents"][0]["is_positive"])

    def test_compute_document_listwise_ranking_loss_prefers_positive_docs_above_negatives(self):
        positive_mask = torch.tensor([[True, False]], dtype=torch.bool)
        doc_mask = torch.tensor([[True, True]], dtype=torch.bool)
        low_loss = compute_document_listwise_ranking_loss(
            logits=torch.tensor([[[0.1, 2.0], [2.0, 0.1]]], dtype=torch.float32),
            positive_mask=positive_mask,
            doc_mask=doc_mask,
            positive_label_indices=[1],
            negative_label_indices=[0],
        )
        high_loss = compute_document_listwise_ranking_loss(
            logits=torch.tensor([[[2.0, 0.1], [0.1, 2.0]]], dtype=torch.float32),
            positive_mask=positive_mask,
            doc_mask=doc_mask,
            positive_label_indices=[1],
            negative_label_indices=[0],
        )
        self.assertLess(float(low_loss), float(high_loss))

    def test_document_listwise_dataset_adds_lengths(self):
        train_raw = Dataset.from_list(
            [
                {
                    "example_id": "ex-1",
                    "sample_id": "sample-1",
                    "dataset": "scifact",
                    "group_id": "sample-1",
                    "claim": "Claim A",
                    "evidence_text": "Relevant sentence A1.",
                    "doc_id": "doc-a",
                    "sentence_id": 0,
                    "label": 1,
                },
                {
                    "example_id": "ex-2",
                    "sample_id": "sample-1",
                    "dataset": "scifact",
                    "group_id": "sample-1",
                    "claim": "Claim A",
                    "evidence_text": "Neutral sentence A2.",
                    "doc_id": "doc-b",
                    "sentence_id": 0,
                    "label": 0,
                },
            ]
        )
        eval_raw = Dataset.from_list(
            [
                {
                    "example_id": "eval-1",
                    "sample_id": "sample-9",
                    "dataset": "scifact",
                    "group_id": "sample-9",
                    "claim": "Claim Z",
                    "evidence_text": "Neutral eval.",
                    "doc_id": "doc-z",
                    "sentence_id": 0,
                    "label": 0,
                }
            ]
        )
        _, tokenizer = build_smoke_verifier(
            train_dataset=train_raw,
            eval_dataset=eval_raw,
            num_labels=2,
            max_length=64,
        )

        listwise_dataset, stats = build_document_listwise_ranking_dataset(
            train_raw,
            tokenizer,
            64,
            ["NEUTRAL", "RELEVANT"],
        )

        self.assertEqual(stats["num_groups"], 1)
        self.assertIn("documents", listwise_dataset.column_names)
        self.assertIn("length", listwise_dataset.column_names)

    def test_document_listwise_trainer_uses_pointwise_eval_collator(self):
        train_raw = Dataset.from_list(
            [
                {
                    "example_id": "ex-1",
                    "sample_id": "sample-1",
                    "dataset": "scifact",
                    "group_id": "sample-1",
                    "claim": "Claim A",
                    "evidence_text": "Relevant sentence A1.",
                    "doc_id": "doc-a",
                    "sentence_id": 0,
                    "label": 1,
                },
                {
                    "example_id": "ex-2",
                    "sample_id": "sample-1",
                    "dataset": "scifact",
                    "group_id": "sample-1",
                    "claim": "Claim A",
                    "evidence_text": "Neutral sentence A2.",
                    "doc_id": "doc-b",
                    "sentence_id": 0,
                    "label": 0,
                },
                {
                    "example_id": "ex-3",
                    "sample_id": "sample-2",
                    "dataset": "scifact",
                    "group_id": "sample-2",
                    "claim": "Claim B",
                    "evidence_text": "Relevant sentence B1.",
                    "doc_id": "doc-c",
                    "sentence_id": 0,
                    "label": 1,
                },
                {
                    "example_id": "ex-4",
                    "sample_id": "sample-2",
                    "dataset": "scifact",
                    "group_id": "sample-2",
                    "claim": "Claim B",
                    "evidence_text": "Neutral sentence B2.",
                    "doc_id": "doc-d",
                    "sentence_id": 0,
                    "label": 0,
                },
            ]
        )
        eval_raw = Dataset.from_list(
            [
                {
                    "example_id": "eval-pos",
                    "sample_id": "sample-3",
                    "dataset": "scifact",
                    "group_id": "sample-3",
                    "claim": "Claim C",
                    "evidence_text": "Relevant eval sentence.",
                    "doc_id": "doc-e",
                    "sentence_id": 0,
                    "label": 1,
                },
                {
                    "example_id": "eval-neg",
                    "sample_id": "sample-3",
                    "dataset": "scifact",
                    "group_id": "sample-3",
                    "claim": "Claim C",
                    "evidence_text": "Neutral eval sentence.",
                    "doc_id": "doc-f",
                    "sentence_id": 0,
                    "label": 0,
                },
            ]
        )
        model, tokenizer = build_smoke_verifier(
            train_dataset=train_raw,
            eval_dataset=eval_raw,
            num_labels=2,
            max_length=64,
        )
        listwise_train, _ = build_document_listwise_ranking_dataset(
            train_raw,
            tokenizer,
            64,
            ["NEUTRAL", "RELEVANT"],
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
                evaluation_strategy="no",
                fp16=False,
                bf16=False,
                no_cuda=True,
                dataloader_num_workers=0,
                remove_unused_columns=False,
                report_to=[],
            )
            trainer = DocumentListwiseRankingTrainer(
                model=model,
                args=training_args,
                train_dataset=listwise_train,
                eval_dataset=pointwise_eval,
                tokenizer=tokenizer,
                train_data_collator=DocumentListwiseDataCollator(tokenizer=tokenizer, max_length=64),
                eval_data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                positive_label_indices=[1],
                negative_label_indices=[0],
            )
            metrics = trainer.evaluate()

        self.assertIn("eval_loss", metrics)


if __name__ == "__main__":
    unittest.main()
