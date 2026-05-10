from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mjgpt_training.samples import PolicySample, SampleFormatError, build_policy_sample
from mjgpt_training.tokenizer import (
    SCORE_BUCKET_MODE,
    MahjongVocab,
    build_fixed_vocab,
    build_vocab,
    encode_sample,
    score_bucket_token,
    tokenize_text,
)

from test_training_samples import make_record


class TokenizerTests(unittest.TestCase):
    def test_tokenize_preserves_tiles_and_action_boundaries(self) -> None:
        tokens = tokenize_text("<A0> DISCARD 5mr HAND </A0>\n<CHOICE>")
        self.assertEqual(tokens, ["<A0>", "DISCARD", "5mr", "HAND", "</A0>", "<CHOICE>"])

    def test_score_bucket_tokenize_only_normalizes_score_lines(self) -> None:
        text = "\n".join(
            [
                "<SCORES>",
                "SELF 25000",
                "SHIMO -100",
                "",
                "<SELF_HAND>",
                "1m 2m 3m",
                "<LEGAL_ACTIONS>",
                "<A10> DISCARD 2m HAND </A10>",
            ]
        )
        tokens = tokenize_text(text, score_mode=SCORE_BUCKET_MODE)
        self.assertIn("<SCORE_25000_29999>", tokens)
        self.assertIn("<SCORE_NEG>", tokens)
        self.assertIn("2m", tokens)
        self.assertIn("<A10>", tokens)
        self.assertNotIn("25000", tokens)

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

    def test_fixed_vocab_round_trip_json_preserves_tokenization(self) -> None:
        vocab = build_fixed_vocab()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vocab.json"
            vocab.save(path)
            loaded = MahjongVocab.load(path)
        self.assertEqual(loaded.score_mode(), SCORE_BUCKET_MODE)
        self.assertEqual(loaded.token_to_id, vocab.token_to_id)

    def test_fixed_vocab_contains_expected_stable_tokens(self) -> None:
        vocab = build_fixed_vocab()
        for token in ("<SCORE_25000_29999>", "5mr", "<A63>", "</A63>", "SELF_TURN_AFTER_DRAW", "TSUMOGIRI"):
            self.assertIn(token, vocab.token_to_id)
        self.assertIn("100", vocab.token_to_id)
        self.assertNotIn("101", vocab.token_to_id)

    def test_fixed_vocab_rejects_excessive_action_count(self) -> None:
        with self.assertRaises(ValueError):
            build_fixed_vocab(max_actions=100_000)

    def test_fixed_vocab_encodes_sample_with_score_buckets(self) -> None:
        sample = PolicySample(
            input_text="\n".join(
                [
                    "<BOS>",
                    "<SCORES>",
                    "SELF 25000",
                    "SHIMO 24000",
                    "TOIMEN 30000",
                    "KAMI 21000",
                    "<LEGAL_ACTIONS>",
                    "<A0> DISCARD 5mr HAND </A0>",
                    "<A1> DISCARD 2m HAND </A1>",
                    "</LEGAL_ACTIONS>",
                    "<CHOICE>",
                ]
            ),
            label=0,
            legal_action_count=2,
            decision_type="SELF_TURN_AFTER_DRAW",
        )
        vocab = build_fixed_vocab()
        encoded = encode_sample(sample, vocab)
        decoded = vocab.decode_ids(encoded.input_ids)
        self.assertIn(score_bucket_token(25000), decoded)
        self.assertNotIn("25000", decoded)
        self.assertNotIn(vocab.unk_token, decoded)

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
