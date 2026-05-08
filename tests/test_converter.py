from __future__ import annotations

import unittest
from pathlib import Path

from mjgpt_converter.agari import is_complete_hand, is_tenpai
from mjgpt_converter.converter import ConvertReport, convert_file
from mjgpt_converter.io import read_events
from mjgpt_converter.tiles import normalize_tile, sort_tiles


class TileTests(unittest.TestCase):
    def test_normalize_honors_and_sort_red_fives(self) -> None:
        self.assertEqual(normalize_tile("E"), "1z")
        self.assertEqual(normalize_tile("C"), "7z")
        self.assertEqual(sort_tiles(["5pr", "4p", "5p", "1m"]), ["1m", "4p", "5p", "5pr"])


class AgariTests(unittest.TestCase):
    def test_complete_and_tenpai_shape(self) -> None:
        hand = ["1m", "1m", "1m", "2m", "3m", "4m", "5p", "6p", "7p", "2s", "3s", "4s", "7z", "7z"]
        self.assertTrue(is_complete_hand(hand))
        tenpai = hand[:-1]
        self.assertTrue(is_tenpai(tenpai))


class ConvertSmokeTests(unittest.TestCase):
    def test_sample_file_replays_without_invalid_actions(self) -> None:
        path = Path("data-draft/2024010100gm-00a9-0000-0d9240dd.mjson")
        if not path.exists():
            self.skipTest("sample mjson not present")
        report = ConvertReport()
        records = convert_file(path, read_events(path), report)
        self.assertGreater(len(records), 0)
        self.assertEqual(report.invalid_decisions, 0)
        self.assertIn("<LEGAL_ACTIONS>", records[0]["state_text"])


if __name__ == "__main__":
    unittest.main()
