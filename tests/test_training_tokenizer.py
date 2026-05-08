from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mjgpt_training.samples import SampleFormatError, build_policy_sample
from mjgpt_training.tokenizer import MahjongVocab, build_vocab, encode_sample, tokenize_text

from test_training_samples import make_record


class TokenizerTests(unittest.TestCase):
    def test_tokenize_preserves_tiles_and_action_boundaries(self) -> None:
        tokens = tokenize_text("<A0> DISCARD 5mr HAND </A0>\n<CHOICE>")
        self.assertEqual(tokens, ["<A0>", "DISCARD", "5mr", "HAND", "</A0>", "<CHOICE>"])

    def test_build_vocab_and_encode_sample(self) -> None:
        sample = build_policy_sample(make_record())
        vocab = build_vocab([sample])
        encoded = encode_sample(sample, vocab)
        self.assertEqual(encoded.label, 1)
        self.assertEqual(len(encoded.action_positions), 2)
        self.assertEqual(vocab.decode_ids([encoded.input_ids[encoded.choice_position]])[0], "<CHOICE>")
        action_tokens = vocab.decode_ids(encoded.input_ids[i] for i in encoded.action_positions)
        self.assertEqual(action_tokens, ["</A0>", "</A1>"])

    def test_vocab_round_trip_json(self) -> None:
        sample = build_policy_sample(make_record())
        vocab = build_vocab([sample])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vocab.json"
            vocab.save(path)
            loaded = MahjongVocab.load(path)
        self.assertEqual(loaded.token_to_id, vocab.token_to_id)
        self.assertEqual(loaded.id_to_token, vocab.id_to_token)

    def test_encode_rejects_missing_action_end(self) -> None:
        sample = build_policy_sample(make_record())
        bad_sample = type(sample)(
            input_text=sample.input_text.replace("</A1>", ""),
            label=sample.label,
            legal_action_count=sample.legal_action_count,
            decision_type=sample.decision_type,
            source=sample.source,
        )
        vocab = build_vocab([bad_sample])
        with self.assertRaises(SampleFormatError):
            encode_sample(bad_sample, vocab)


if __name__ == "__main__":
    unittest.main()
