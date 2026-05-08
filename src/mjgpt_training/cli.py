from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Iterable

from .samples import PolicySample, build_policy_sample
from .tokenizer import MahjongVocab, build_vocab


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mjgpt-train")
    sub = parser.add_subparsers(dest="cmd", required=True)

    vocab_p = sub.add_parser("build-vocab", help="Build a word-level vocab from converter JSONL.")
    vocab_p.add_argument("input", type=Path, nargs="+")
    vocab_p.add_argument("--out", type=Path, required=True)
    vocab_p.add_argument("--min-freq", type=int, default=1)
    vocab_p.add_argument("--max-size", type=int, default=None)
    vocab_p.add_argument("--limit-records", type=int, default=None)

    args = parser.parse_args(argv)
    if args.cmd == "build-vocab":
        samples = _iter_policy_samples(args.input, limit_records=args.limit_records)
        vocab = build_vocab(samples, min_freq=args.min_freq, max_size=args.max_size)
        vocab.save(args.out)
        print(f"wrote {len(vocab.id_to_token)} tokens to {args.out}")


def _iter_policy_samples(paths: list[Path], *, limit_records: int | None = None) -> Iterable[PolicySample]:
    count = 0
    for path in paths:
        for record in _iter_jsonl_records(path):
            yield build_policy_sample(record)
            count += 1
            if limit_records is not None and count >= limit_records:
                return


def _iter_jsonl_records(path: Path) -> Iterable[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
