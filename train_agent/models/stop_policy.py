from __future__ import annotations

from pathlib import Path

from train_agent.models.action_policy import FrozenActionPolicy


class FrozenStopPolicy:
    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        max_length: int = 512,
        batch_size: int = 8,
        attn_implementation: str = "sdpa",
    ) -> None:
        self.policy = FrozenActionPolicy(
            model_name_or_path,
            max_length=max_length,
            batch_size=batch_size,
            attn_implementation=attn_implementation,
        )
        label_names = set(self.policy.label_names)
        if "yes" not in label_names or "no" not in label_names:
            raise ValueError(
                f"Stop policy expects 'yes'/'no' labels, got {self.policy.label_names}"
            )

    @property
    def label_names(self):
        return self.policy.label_names

    def predict_label(self, text: str) -> str:
        return self.policy.predict_action(text)

    def predict_should_stop(self, text: str) -> bool:
        return self.predict_label(text) == "yes"
