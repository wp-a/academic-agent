import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from train_agent.models.action_policy import FrozenActionPolicy


class DummyTokenizer:
    pad_token = None
    eos_token = "<eos>"
    unk_token = "<unk>"
    pad_token_id = 7

    def __call__(self, texts, truncation, max_length, padding, return_tensors):
        raise AssertionError("tokenization should not run in this loader test")


class DummyModel:
    def __init__(self):
        self.config = SimpleNamespace(pad_token_id=None, num_labels=3)

    def eval(self):
        return self

    def to(self, device):
        return self


class ActionPolicyModelTest(unittest.TestCase):
    def test_peft_loader_uses_action_label_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "adapter_config.json").write_text("{}", encoding="utf-8")
            (root / "label_names.json").write_text(json.dumps(["quote_evidence", "search", "stop"]), encoding="utf-8")

            loader_calls = {}

            def fake_loader(model_name_or_path, **kwargs):
                loader_calls["path"] = model_name_or_path
                loader_calls["kwargs"] = kwargs
                return DummyModel()

            with patch("train_agent.models.action_policy.AutoTokenizer.from_pretrained", return_value=DummyTokenizer()), patch(
                "train_agent.models.action_policy.AutoPeftModelForSequenceClassification.from_pretrained",
                side_effect=fake_loader,
            ):
                policy = FrozenActionPolicy(root)

            self.assertEqual(loader_calls["path"], str(root))
            self.assertEqual(loader_calls["kwargs"]["num_labels"], 3)
            self.assertTrue(loader_calls["kwargs"]["ignore_mismatched_sizes"])
            self.assertEqual(policy.model.config.pad_token_id, 7)

    def test_loader_can_fall_back_to_parent_label_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_dir = root / "checkpoint-10"
            checkpoint_dir.mkdir()
            (root / "label_names.json").write_text(
                json.dumps(["quote_evidence", "search", "stop"]),
                encoding="utf-8",
            )

            loader_calls = {}

            def fake_loader(model_name_or_path, **kwargs):
                loader_calls["path"] = model_name_or_path
                loader_calls["kwargs"] = kwargs
                return DummyModel()

            with patch("train_agent.models.action_policy.AutoTokenizer.from_pretrained", return_value=DummyTokenizer()), patch(
                "train_agent.models.action_policy.AutoModelForSequenceClassification.from_pretrained",
                side_effect=fake_loader,
            ):
                policy = FrozenActionPolicy(checkpoint_dir)

        self.assertEqual(loader_calls["path"], str(checkpoint_dir))
        self.assertEqual(policy.label_names, ["quote_evidence", "search", "stop"])


if __name__ == "__main__":
    unittest.main()
