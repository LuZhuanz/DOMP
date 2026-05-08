from __future__ import annotations

import unittest

from mjgpt_training.samples import SampleFormatError, build_policy_sample


def make_record(choice_id: str = "A1") -> dict:
    return {
        "source_file": "sample.mjson",
        "decision_index": 3,
        "decision_type": "SELF_TURN_AFTER_DRAW",
        "state_text": "\n".join(
            [
                "<BOS>",
                "<LEGAL_ACTIONS>",
                "<A0> DISCARD 1m HAND </A0>",
                "<A1> DISCARD 2m HAND </A1>",
                "</LEGAL_ACTIONS>",
                "",
                "<CHOICE>",
                f"<{choice_id}>",
                "</CHOICE>",
                "",
                "<EXECUTE>",
                "DISCARD 2m HAND",
                "</EXECUTE>",
                "<EOS>",
            ]
        ),
        "legal_actions": [
            {"id": "A0", "kind": "DISCARD", "text": "DISCARD 1m HAND"},
            {"id": "A1", "kind": "DISCARD", "text": "DISCARD 2m HAND"},
        ],
        "choice_id": choice_id,
        "execute": "DISCARD 2m HAND",
        "validation_flags": [],
    }


class PolicySampleTests(unittest.TestCase):
    def test_build_policy_sample_removes_choice_answer_and_execute(self) -> None:
        sample = build_policy_sample(make_record())
        self.assertEqual(sample.label, 1)
        self.assertEqual(sample.legal_action_count, 2)
        self.assertEqual(sample.decision_type, "SELF_TURN_AFTER_DRAW")
        self.assertTrue(sample.input_text.endswith("<CHOICE>"))
        self.assertIn("<A1> DISCARD 2m HAND </A1>", sample.input_text)
        self.assertNotIn("</CHOICE>", sample.input_text)
        self.assertNotIn("<EXECUTE>", sample.input_text)
        self.assertNotIn("DISCARD 2m HAND\n</EXECUTE>", sample.input_text)

    def test_rejects_choice_id_outside_legal_actions(self) -> None:
        with self.assertRaises(SampleFormatError):
            build_policy_sample(make_record("A2"))

    def test_rejects_missing_choice(self) -> None:
        record = make_record()
        record["state_text"] = record["state_text"].replace("<CHOICE>", "<NO_CHOICE>")
        with self.assertRaises(SampleFormatError):
            build_policy_sample(record)


if __name__ == "__main__":
    unittest.main()
