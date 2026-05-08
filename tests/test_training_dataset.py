from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from mjgpt_training.dataset import JsonlPolicyDataset, MjsonStreamingPolicyDataset, iter_jsonl_policy_samples, iter_mjson_policy_samples
from mjgpt_training.tokenizer import build_vocab

from test_training_samples import make_record


class TrainingDatasetTests(unittest.TestCase):
    def test_jsonl_policy_samples_support_plain_and_gzip(self) -> None:
        record = make_record()
        with tempfile.TemporaryDirectory() as tmp:
            plain = Path(tmp) / "samples.jsonl"
            gz = Path(tmp) / "samples.jsonl.gz"
            plain.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
            with gzip.open(gz, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            samples = list(iter_jsonl_policy_samples([plain, gz]))
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].label, 1)

    def test_jsonl_dataset_yields_tokenized_samples(self) -> None:
        record = make_record()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.jsonl"
            path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
            samples = list(iter_jsonl_policy_samples([path]))
            vocab = build_vocab(samples)
            encoded = list(JsonlPolicyDataset([path], vocab=vocab))
        self.assertEqual(len(encoded), 1)
        self.assertEqual(encoded[0].label, 1)
        self.assertEqual(len(encoded[0].action_positions), 2)

    def test_mjson_streaming_dataset_yields_samples(self) -> None:
        root = Path("data-draft")
        if not root.exists():
            self.skipTest("data-draft not present")
        samples = list(iter_mjson_policy_samples([root], limit_files=1, limit_records=8))
        self.assertGreater(len(samples), 0)
        vocab = build_vocab(samples)
        encoded = list(MjsonStreamingPolicyDataset([root], vocab=vocab, limit_files=1, limit_records=8))
        self.assertGreater(len(encoded), 0)
        self.assertGreaterEqual(len(encoded[0].action_positions), 1)


if __name__ == "__main__":
    unittest.main()
