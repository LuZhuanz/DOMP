from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import iter_policy_samples
from .tokenizer import RAW_SCORE_MODE, SCORE_BUCKET_MODE, build_fixed_vocab, build_vocab
from .train import TrainConfig, train


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mjgpt-train")
    sub = parser.add_subparsers(dest="cmd", required=True)

    vocab_p = sub.add_parser("build-vocab", help="Build a word-level vocab from JSONL or streaming mjson.")
    vocab_p.add_argument("input", type=Path, nargs="+")
    vocab_p.add_argument("--data-format", choices=["jsonl", "mjson"], default="jsonl")
    vocab_p.add_argument("--out", type=Path, required=True)
    vocab_p.add_argument("--min-freq", type=int, default=1)
    vocab_p.add_argument("--max-size", type=int, default=None)
    vocab_p.add_argument("--score-mode", choices=[RAW_SCORE_MODE, SCORE_BUCKET_MODE], default=RAW_SCORE_MODE)
    vocab_p.add_argument("--limit-files", type=int, default=None)
    vocab_p.add_argument("--limit-records", type=int, default=None)
    vocab_p.add_argument("--shuffle-files", action="store_true")
    vocab_p.add_argument("--seed", type=int, default=42)
    vocab_p.add_argument("--no-strict", dest="strict", action="store_false")

    fixed_vocab_p = sub.add_parser("build-fixed-vocab", help="Build a stable v1 text vocab without scanning data.")
    fixed_vocab_p.add_argument("--out", type=Path, required=True)
    fixed_vocab_p.add_argument("--max-actions", type=int, default=64)
    fixed_vocab_p.add_argument("--small-int-max", type=int, default=100)

    train_p = sub.add_parser("train", help="Run minimal supervised policy training.")
    train_p.add_argument("input", type=Path, nargs="+")
    train_p.add_argument("--data-format", choices=["jsonl", "mjson"], default="mjson")
    train_p.add_argument("--output-dir", type=Path, required=True)
    train_p.add_argument("--vocab", type=Path, default=None)
    train_p.add_argument("--model-size", choices=["debug", "tiny", "small", "base"], default="debug")
    train_p.add_argument("--max-length", type=int, default=512)
    train_p.add_argument("--batch-size", type=int, default=4)
    train_p.add_argument("--max-steps", type=int, default=20)
    train_p.add_argument("--lr", type=float, default=3e-4)
    train_p.add_argument("--weight-decay", type=float, default=0.1)
    train_p.add_argument("--grad-clip", type=float, default=1.0)
    train_p.add_argument("--num-workers", type=int, default=0)
    train_p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--log-every", type=int, default=1)
    train_p.add_argument("--limit-files", type=int, default=None)
    train_p.add_argument("--limit-records", type=int, default=None)
    train_p.add_argument("--shuffle-files", action="store_true")
    train_p.add_argument("--no-strict", dest="strict", action="store_false")
    train_p.add_argument("--val-input", type=Path, nargs="*", default=None)
    train_p.add_argument("--val-manifest", type=Path, default=None)
    train_p.add_argument("--val-ratio", type=float, default=0.0)
    train_p.add_argument("--val-batches", type=int, default=20)
    train_p.add_argument("--eval-every", type=int, default=0)

    args = parser.parse_args(argv)
    if args.cmd == "build-vocab":
        samples = iter_policy_samples(
            args.input,
            data_format=args.data_format,
            limit_files=args.limit_files,
            limit_records=args.limit_records,
            strict=args.strict,
            shuffle_files=args.shuffle_files,
            seed=args.seed,
        )
        vocab = build_vocab(samples, min_freq=args.min_freq, max_size=args.max_size, score_mode=args.score_mode)
        vocab.save(args.out)
        print(f"wrote {len(vocab.id_to_token)} tokens to {args.out}")
        return

    if args.cmd == "build-fixed-vocab":
        vocab = build_fixed_vocab(max_actions=args.max_actions, small_int_max=args.small_int_max)
        vocab.save(args.out)
        print(f"wrote {len(vocab.id_to_token)} tokens to {args.out}")
        return

    result = train(
        TrainConfig(
            inputs=args.input,
            data_format=args.data_format,
            output_dir=args.output_dir,
            vocab_path=args.vocab,
            model_size=args.model_size,
            max_length=args.max_length,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            num_workers=args.num_workers,
            device=args.device,
            seed=args.seed,
            log_every=args.log_every,
            limit_files=args.limit_files,
            limit_records=args.limit_records,
            strict=args.strict,
            shuffle_files=args.shuffle_files,
            val_inputs=args.val_input,
            val_manifest=args.val_manifest,
            val_ratio=args.val_ratio,
            val_batches=args.val_batches,
            eval_every=args.eval_every,
        )
    )
    print(
        f"trained {result.steps} steps on {result.samples} samples; "
        f"final_loss={result.final_loss}; output_dir={result.output_dir}"
    )
