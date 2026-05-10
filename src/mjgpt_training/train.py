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
from .dataset import JsonlPolicyDataset, MjsonStreamingPolicyDataset, _collect_mjson_files, iter_policy_samples
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
    val_inputs: list[Path] | None = None
    val_manifest: Path | None = None
    val_ratio: float = 0.0
    val_batches: int = 20
    eval_every: int = 0
    save_every: int = 0
    resume: bool = False


@dataclass(frozen=True)
class TrainResult:
    steps: int
    samples: int
    final_loss: float | None
    final_accuracy_top1: float | None
    final_val_loss: float | None
    final_val_accuracy_top1: float | None
    output_dir: Path


def train(config: TrainConfig) -> TrainResult:
    """Run a minimal supervised policy training loop."""
    if config.max_steps < 1:
        raise ValueError("max_steps must be positive")
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not (0.0 <= config.val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0, 1)")
    if config.val_batches < 1:
        raise ValueError("val_batches must be positive")
    if config.eval_every < 0:
        raise ValueError("eval_every must be non-negative")
    if config.save_every < 0:
        raise ValueError("save_every must be non-negative")

    _set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(config.device)
    vocab = _load_or_build_vocab(config)
    vocab.save(config.output_dir / "vocab.json")

    train_inputs, val_inputs = _resolve_train_val_inputs(config)
    dataset = _build_dataset(config, vocab, train_inputs, limit_files=None if val_inputs else config.limit_files)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=PolicyCollator(pad_id=vocab.pad_id, max_length=config.max_length),
        num_workers=config.num_workers,
    )
    val_loader = _build_val_loader(config, vocab, val_inputs)
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
    final_val_loss: float | None = None
    final_val_accuracy: float | None = None
    start = time.perf_counter()

    if config.resume:
        steps, samples_seen = _try_resume(config, model, optimizer)
        start -= samples_seen / max(1, steps) * (time.perf_counter() - start)  # rough offset

    log_mode = "a" if config.resume else "w"
    with log_path.open(log_mode, encoding="utf-8") as log_file:
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
                should_eval = val_loader is not None and (
                    steps == 1 or steps % (config.eval_every or config.log_every) == 0
                )
                if should_eval:
                    final_val_loss, final_val_accuracy = _evaluate(model, val_loader, device, config.val_batches)
                event = {
                    "step": steps,
                    "samples": samples_seen,
                    "loss": final_loss,
                    "accuracy_top1": final_accuracy,
                    "samples_per_sec": samples_seen / elapsed,
                    "device": str(device),
                }
                if final_val_loss is not None and should_eval:
                    event["val_loss"] = final_val_loss
                    event["val_accuracy_top1"] = final_val_accuracy
                log_file.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
                log_file.flush()
                message = f"step={steps} samples={samples_seen} loss={final_loss:.4f} acc={final_accuracy:.3f}"
                if final_val_loss is not None and should_eval:
                    message += f" val_loss={final_val_loss:.4f} val_acc={final_val_accuracy:.3f}"
                print(message)

            if config.save_every > 0 and steps % config.save_every == 0:
                _save_checkpoint(config, model_config, model, optimizer, steps, samples_seen)

    _save_outputs(
        config,
        model_config,
        model,
        optimizer,
        steps,
        samples_seen,
        final_loss,
        final_accuracy,
        final_val_loss,
        final_val_accuracy,
    )
    return TrainResult(
        steps=steps,
        samples=samples_seen,
        final_loss=final_loss,
        final_accuracy_top1=final_accuracy,
        final_val_loss=final_val_loss,
        final_val_accuracy_top1=final_val_accuracy,
        output_dir=config.output_dir,
    )


def _try_resume(config: TrainConfig, model: MahjongPolicyModel, optimizer: torch.optim.Optimizer) -> tuple[int, int]:
    """Load checkpoint and return resumed step/sample counts."""
    model_path = config.output_dir / "model.pt"
    optimizer_path = config.output_dir / "optimizer.pt"
    state_path = config.output_dir / "train_state.json"
    if not model_path.exists():
        raise FileNotFoundError(f"resume requested but checkpoint not found: {model_path}")
    if not optimizer_path.exists():
        raise FileNotFoundError(f"resume requested but optimizer state not found: {optimizer_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"resume requested but train_state not found: {state_path}")

    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu", weights_only=True))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    steps = int(state.get("steps", 0))
    samples = int(state.get("samples", 0))
    print(f"resumed from step={steps} samples={samples}")
    return steps, samples


def _save_checkpoint(
    config: TrainConfig,
    model_config: ModelConfig,
    model: MahjongPolicyModel,
    optimizer: torch.optim.Optimizer,
    steps: int,
    samples_seen: int,
) -> None:
    """Save a periodic checkpoint without overwriting the final outputs."""
    ckpt_dir = config.output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    prefix = f"step_{steps}"
    torch.save(model.state_dict(), ckpt_dir / f"{prefix}_model.pt")
    torch.save(optimizer.state_dict(), ckpt_dir / f"{prefix}_optimizer.pt")
    (ckpt_dir / f"{prefix}_model_config.json").write_text(
        json.dumps(asdict(model_config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    train_state = {
        "steps": steps,
        "samples": samples_seen,
        "train_config": _jsonable_train_config(config),
    }
    (ckpt_dir / f"{prefix}_train_state.json").write_text(
        json.dumps(train_state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved checkpoint to {ckpt_dir}/{prefix}_*")


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


def _build_dataset(
    config: TrainConfig,
    vocab: MahjongVocab,
    inputs: list[Path],
    *,
    limit_files: int | None,
):
    if config.data_format == "jsonl":
        return JsonlPolicyDataset(inputs, vocab=vocab, limit_records=config.limit_records)
    if config.data_format == "mjson":
        return MjsonStreamingPolicyDataset(
            inputs,
            vocab=vocab,
            limit_files=limit_files,
            limit_records=config.limit_records,
            strict=config.strict,
            shuffle_files=config.shuffle_files,
            seed=config.seed,
        )
    raise ValueError(f"unknown data_format: {config.data_format}")


def _build_val_loader(config: TrainConfig, vocab: MahjongVocab, val_inputs: list[Path] | None) -> DataLoader | None:
    if val_inputs is None or len(val_inputs) == 0:
        return None
    dataset = _build_dataset(config, vocab, val_inputs, limit_files=None)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=PolicyCollator(pad_id=vocab.pad_id, max_length=config.max_length),
        # Validation is intentionally bounded by val_batches; extra workers add overhead for small eval slices.
        num_workers=0,
    )


def _resolve_train_val_inputs(config: TrainConfig) -> tuple[list[Path], list[Path] | None]:
    if config.val_ratio == 0.0 and not config.val_inputs and config.val_manifest is None:
        return list(config.inputs), None
    if config.data_format != "mjson":
        raise ValueError("validation split currently supports mjson inputs only")

    all_files = _collect_mjson_files(
        config.inputs,
        limit_files=config.limit_files,
        shuffle_files=config.shuffle_files,
        seed=config.seed,
    )
    val_files = _resolve_val_files(config, all_files)
    val_set = {path.resolve() for path in val_files}
    train_files = [path for path in all_files if path.resolve() not in val_set]
    if not train_files:
        raise ValueError("validation split left no training files")
    if not val_files:
        raise ValueError("validation split produced no validation files")
    _write_file_list(config.output_dir / "validation_files.txt", val_files)
    return train_files, val_files


def _resolve_val_files(config: TrainConfig, all_files: list[Path]) -> list[Path]:
    if config.val_manifest is not None:
        return _read_file_list(config.val_manifest)
    if config.val_inputs:
        return _collect_mjson_files(config.val_inputs)
    if config.val_ratio <= 0.0:
        return []
    if len(all_files) < 2:
        raise ValueError("val_ratio requires at least two mjson files")
    rng = random.Random(config.seed)
    shuffled = list(all_files)
    rng.shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * config.val_ratio))
    val_count = min(val_count, len(shuffled) - 1)
    return sorted(shuffled[:val_count])


def _read_file_list(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(path)
    files = [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not files:
        raise ValueError(f"validation manifest is empty: {path}")
    return files


def _write_file_list(path: Path, files: list[Path]) -> None:
    path.write_text("".join(f"{file}\n" for file in files), encoding="utf-8")


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


@torch.no_grad()
def _evaluate(
    model: MahjongPolicyModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    try:
        losses: list[float] = []
        accuracies: list[float] = []
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = _batch_to_device(batch, device)
            output = model(
                batch.input_ids,
                batch.attention_mask,
                batch.choice_positions,
                batch.action_positions,
                batch.action_mask,
                batch.labels,
            )
            assert output.loss is not None
            losses.append(float(output.loss.detach().cpu()))
            accuracies.append(_accuracy_top1(output.logits.detach(), batch.labels))
        if not losses:
            raise RuntimeError("validation loader produced no batches")
        return sum(losses) / len(losses), sum(accuracies) / len(accuracies)
    finally:
        if was_training:
            model.train()


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
    final_val_loss: float | None,
    final_val_accuracy: float | None,
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
        "final_val_loss": final_val_loss,
        "final_val_accuracy_top1": final_val_accuracy,
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
    data["val_inputs"] = [str(path) for path in config.val_inputs] if config.val_inputs is not None else None
    data["val_manifest"] = str(config.val_manifest) if config.val_manifest is not None else None
    return data
