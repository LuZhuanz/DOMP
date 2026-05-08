from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader

from .collator import PolicyBatch, PolicyCollator
from .dataset import JsonlPolicyDataset, MjsonStreamingPolicyDataset, iter_policy_samples
from .model import MahjongPolicyModel, ModelConfig
from .tokenizer import MahjongVocab, build_vocab


DataFormat = Literal["jsonl", "mjson"]


@dataclass(frozen=True)
class TrainConfig:
    inputs: list[Path]
    data_format: DataFormat
    output_dir: Path
    vocab_path: Path | None = None
    model_size: str = "debug"
    max_length: int = 512
    batch_size: int = 4
    max_steps: int = 20
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    num_workers: int = 0
    device: str = "auto"
    seed: int = 42
    log_every: int = 1
    limit_files: int | None = None
    limit_records: int | None = None
    strict: bool = True
    shuffle_files: bool = False


@dataclass(frozen=True)
class TrainResult:
    steps: int
    samples: int
    final_loss: float | None
    final_accuracy_top1: float | None
    output_dir: Path


def train(config: TrainConfig) -> TrainResult:
    """Run a minimal supervised policy training loop."""
    if config.max_steps < 1:
        raise ValueError("max_steps must be positive")
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")

    _set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(config.device)
    vocab = _load_or_build_vocab(config)
    vocab.save(config.output_dir / "vocab.json")

    dataset = _build_dataset(config, vocab)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=PolicyCollator(pad_id=vocab.pad_id, max_length=config.max_length),
        num_workers=config.num_workers,
    )
    model_config = ModelConfig.preset(
        config.model_size,
        vocab_size=len(vocab.id_to_token),
        max_position_embeddings=config.max_length,
    )
    model = MahjongPolicyModel(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    log_path = config.output_dir / "train.log.jsonl"
    steps = 0
    samples_seen = 0
    final_loss: float | None = None
    final_accuracy: float | None = None
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        for batch in loader:
            if steps >= config.max_steps:
                break
            batch = _batch_to_device(batch, device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch.input_ids,
                batch.attention_mask,
                batch.choice_positions,
                batch.action_positions,
                batch.action_mask,
                batch.labels,
            )
            assert output.loss is not None
            output.loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            steps += 1
            batch_size = int(batch.labels.shape[0])
            samples_seen += batch_size
            final_loss = float(output.loss.detach().cpu())
            final_accuracy = _accuracy_top1(output.logits.detach(), batch.labels)
            if steps % config.log_every == 0 or steps == 1:
                elapsed = max(time.perf_counter() - start, 1e-9)
                event = {
                    "step": steps,
                    "samples": samples_seen,
                    "loss": final_loss,
                    "accuracy_top1": final_accuracy,
                    "samples_per_sec": samples_seen / elapsed,
                    "device": str(device),
                }
                log_file.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
                log_file.flush()
                print(
                    f"step={steps} samples={samples_seen} "
                    f"loss={final_loss:.4f} acc={final_accuracy:.3f}"
                )

    _save_outputs(config, model_config, model, optimizer, steps, samples_seen, final_loss, final_accuracy)
    return TrainResult(
        steps=steps,
        samples=samples_seen,
        final_loss=final_loss,
        final_accuracy_top1=final_accuracy,
        output_dir=config.output_dir,
    )


def _load_or_build_vocab(config: TrainConfig) -> MahjongVocab:
    if config.vocab_path is not None:
        return MahjongVocab.load(config.vocab_path)
    samples = iter_policy_samples(
        config.inputs,
        data_format=config.data_format,
        limit_files=config.limit_files,
        limit_records=config.limit_records,
        strict=config.strict,
        shuffle_files=False,
        seed=config.seed,
    )
    return build_vocab(samples)


def _build_dataset(config: TrainConfig, vocab: MahjongVocab):
    if config.data_format == "jsonl":
        return JsonlPolicyDataset(config.inputs, vocab=vocab, limit_records=config.limit_records)
    if config.data_format == "mjson":
        return MjsonStreamingPolicyDataset(
            config.inputs,
            vocab=vocab,
            limit_files=config.limit_files,
            limit_records=config.limit_records,
            strict=config.strict,
            shuffle_files=config.shuffle_files,
            seed=config.seed,
        )
    raise ValueError(f"unknown data_format: {config.data_format}")


def _batch_to_device(batch: PolicyBatch, device: torch.device) -> PolicyBatch:
    return PolicyBatch(
        input_ids=batch.input_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
        choice_positions=batch.choice_positions.to(device),
        action_positions=batch.action_positions.to(device),
        action_mask=batch.action_mask.to(device),
        labels=batch.labels.to(device),
        decision_types=batch.decision_types,
    )


def _accuracy_top1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=-1)
    return float((predictions == labels).float().mean().cpu())


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    return torch.device(device)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_outputs(
    config: TrainConfig,
    model_config: ModelConfig,
    model: MahjongPolicyModel,
    optimizer: torch.optim.Optimizer,
    steps: int,
    samples_seen: int,
    final_loss: float | None,
    final_accuracy: float | None,
) -> None:
    torch.save(model.state_dict(), config.output_dir / "model.pt")
    torch.save(optimizer.state_dict(), config.output_dir / "optimizer.pt")
    (config.output_dir / "model_config.json").write_text(
        json.dumps(asdict(model_config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    train_state = {
        "steps": steps,
        "samples": samples_seen,
        "final_loss": final_loss,
        "final_accuracy_top1": final_accuracy,
        "train_config": _jsonable_train_config(config),
    }
    (config.output_dir / "train_state.json").write_text(
        json.dumps(train_state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _jsonable_train_config(config: TrainConfig) -> dict:
    data = asdict(config)
    data["inputs"] = [str(path) for path in config.inputs]
    data["output_dir"] = str(config.output_dir)
    data["vocab_path"] = str(config.vocab_path) if config.vocab_path is not None else None
    return data
