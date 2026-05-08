from __future__ import annotations

import unittest

from mjgpt_training.samples import build_policy_sample
from mjgpt_training.tokenizer import build_vocab, encode_sample

from test_training_samples import make_record

try:
    import torch

    from mjgpt_training.collator import PolicyCollator
    from mjgpt_training.model import MahjongPolicyModel, ModelConfig
except ModuleNotFoundError:  # pragma: no cover - depends on optional training extra
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class CollatorModelTests(unittest.TestCase):
    def test_collator_pads_variable_length_and_actions(self) -> None:
        sample_a = build_policy_sample(make_record("A0"))
        record_b = make_record("A1")
        record_b["state_text"] = record_b["state_text"].replace("<BOS>", "<BOS>\n<ROUND> E1")
        record_b["legal_actions"].append({"id": "A2", "kind": "PASS", "text": "PASS"})
        record_b["state_text"] = record_b["state_text"].replace(
            "</LEGAL_ACTIONS>",
            "<A2> PASS </A2>\n</LEGAL_ACTIONS>",
        )
        sample_b = build_policy_sample(record_b)
        vocab = build_vocab([sample_a, sample_b])
        encoded = [encode_sample(sample, vocab) for sample in (sample_a, sample_b)]

        batch = PolicyCollator(pad_id=vocab.pad_id)(encoded)
        self.assertEqual(batch.input_ids.shape[0], 2)
        self.assertEqual(batch.action_positions.shape, (2, 3))
        self.assertFalse(batch.action_mask[0, 2])
        self.assertTrue(batch.action_mask[1, 2])

    def test_tiny_model_forward_backward(self) -> None:
        sample = build_policy_sample(make_record("A1"))
        vocab = build_vocab([sample])
        encoded = encode_sample(sample, vocab)
        batch = PolicyCollator(pad_id=vocab.pad_id)([encoded, encoded])
        config = ModelConfig(
            vocab_size=len(vocab.id_to_token),
            max_position_embeddings=128,
            n_layers=1,
            n_heads=2,
            hidden_size=32,
            intermediate_size=64,
            scorer_hidden_size=32,
            dropout=0.0,
        )
        model = MahjongPolicyModel(config)
        output = model(
            batch.input_ids,
            batch.attention_mask,
            batch.choice_positions,
            batch.action_positions,
            batch.action_mask,
            batch.labels,
        )
        self.assertEqual(output.logits.shape, (2, 2))
        self.assertIsNotNone(output.loss)
        output.loss.backward()


if __name__ == "__main__":
    unittest.main()
