from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised only without training deps
    torch = None

from .tokenizer import TokenizedSample


@dataclass
class PolicyBatch:
    input_ids: "torch.LongTensor"
    attention_mask: "torch.BoolTensor"
    choice_positions: "torch.LongTensor"
    action_positions: "torch.LongTensor"
    action_mask: "torch.BoolTensor"
    labels: "torch.LongTensor"
    decision_types: list[str]


class PolicyCollator:
    """Pad tokenized policy samples into tensors for dynamic action scoring."""

    def __init__(self, *, pad_id: int, max_length: int = 512) -> None:
        if torch is None:
            raise RuntimeError("PolicyCollator requires torch; install the training extra")
        if max_length < 1:
            raise ValueError("max_length must be positive")
        self.pad_id = pad_id
        self.max_length = max_length

    def __call__(self, samples: Sequence[TokenizedSample]) -> PolicyBatch:
        if not samples:
            raise ValueError("cannot collate an empty batch")
        max_seq_len = max(len(sample.input_ids) for sample in samples)
        if max_seq_len > self.max_length:
            raise ValueError(f"sample length {max_seq_len} exceeds max_length {self.max_length}")
        max_actions = max(len(sample.action_positions) for sample in samples)
        if max_actions < 1:
            raise ValueError("samples must contain at least one legal action")

        batch_size = len(samples)
        input_ids = torch.full((batch_size, max_seq_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
        choice_positions = torch.empty(batch_size, dtype=torch.long)
        action_positions = torch.zeros((batch_size, max_actions), dtype=torch.long)
        action_mask = torch.zeros((batch_size, max_actions), dtype=torch.bool)
        labels = torch.empty(batch_size, dtype=torch.long)
        decision_types: list[str] = []

        for row, sample in enumerate(samples):
            seq_len = len(sample.input_ids)
            action_count = len(sample.action_positions)
            if not (0 <= sample.label < action_count):
                raise ValueError(f"label {sample.label} is outside {action_count} actions")
            input_ids[row, :seq_len] = torch.tensor(sample.input_ids, dtype=torch.long)
            attention_mask[row, :seq_len] = True
            choice_positions[row] = sample.choice_position
            action_positions[row, :action_count] = torch.tensor(sample.action_positions, dtype=torch.long)
            action_mask[row, :action_count] = True
            labels[row] = sample.label
            decision_types.append(sample.decision_type)

        return PolicyBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            choice_positions=choice_positions,
            action_positions=action_positions,
            action_mask=action_mask,
            labels=labels,
            decision_types=decision_types,
        )
