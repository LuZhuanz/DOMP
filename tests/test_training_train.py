from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mjgpt_training.train import TrainConfig, train


class MinimalTrainTests(unittest.TestCase):
    def test_streaming_mjson_train_writes_outputs(self) -> None:
        root = Path("data-draft")
        if not root.exists():
            self.skipTest("data-draft not present")
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "train"
            result = train(
                TrainConfig(
                    inputs=[root],
                    data_format="mjson",
                    output_dir=output_dir,
                    model_size="debug",
                    batch_size=2,
                    max_steps=1,
                    limit_files=1,
                    limit_records=8,
                    device="cpu",
                    log_every=1,
                )
            )
            self.assertEqual(result.steps, 1)
            self.assertEqual(result.samples, 2)
            self.assertTrue((output_dir / "model.pt").exists())
            self.assertTrue((output_dir / "optimizer.pt").exists())
            self.assertTrue((output_dir / "vocab.json").exists())
            self.assertTrue((output_dir / "model_config.json").exists())
            state = json.loads((output_dir / "train_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["steps"], 1)
            self.assertEqual(state["samples"], 2)
            self.assertTrue((output_dir / "train.log.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
