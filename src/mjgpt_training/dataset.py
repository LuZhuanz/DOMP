from __future__ import annotations

import gzip
import json
import random
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from torch.utils.data import IterableDataset, get_worker_info

from mjgpt_converter.agari import clear_agari_caches
from mjgpt_converter.converter import ConvertReport, convert_file
from mjgpt_converter.io import iter_mjson_files, read_events

from .samples import PolicySample, SampleFormatError, build_policy_sample
from .tokenizer import MahjongVocab, TokenizedSample, encode_sample


class DatasetFormatError(ValueError):
    """Raised when a training dataset input cannot be decoded."""


def iter_jsonl_records(paths: Sequence[Path]) -> Iterator[dict]:
    """Yield converter records from plain or gzip JSONL files."""
    for path in paths:
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
                    raise DatasetFormatError(f"{path}:{line_number}: invalid JSON") from exc


def iter_jsonl_policy_samples(paths: Sequence[Path], *, limit_records: int | None = None) -> Iterator[PolicySample]:
    """Yield leakage-safe policy samples from converter JSONL files."""
    count = 0
    for record in iter_jsonl_records(paths):
        yield build_policy_sample(record)
        count += 1
        if limit_records is not None and count >= limit_records:
            return


def iter_mjson_policy_samples(
    roots: Sequence[Path],
    *,
    limit_files: int | None = None,
    limit_records: int | None = None,
    strict: bool = True,
    shuffle_files: bool = False,
    seed: int = 0,
) -> Iterator[PolicySample]:
    """Convert mjson files one by one and yield policy samples without writing JSONL."""
    files = _collect_mjson_files(roots, limit_files=limit_files, shuffle_files=shuffle_files, seed=seed)
    yield from _iter_mjson_policy_samples_from_files(files, limit_records=limit_records, strict=strict)


def _collect_mjson_files(
    roots: Sequence[Path],
    *,
    limit_files: int | None = None,
    shuffle_files: bool = False,
    seed: int = 0,
) -> list[Path]:
    if limit_files is not None and limit_files < 1:
        raise ValueError("limit_files must be positive")
    files: list[Path] = []
    for root in roots:
        files.extend(iter_mjson_files(root))
    files = sorted(dict.fromkeys(files))
    if shuffle_files:
        rng = random.Random(seed)
        rng.shuffle(files)
    if limit_files is not None:
        files = files[:limit_files]
    return files


def _iter_mjson_policy_samples_from_files(
    files: Sequence[Path],
    *,
    limit_records: int | None = None,
    strict: bool = True,
) -> Iterator[PolicySample]:
    if limit_records is not None and limit_records < 1:
        raise ValueError("limit_records must be positive")
    emitted = 0
    report = ConvertReport()
    for file in files:
        report.files += 1
        before_errors = len(report.errors)
        try:
            records = convert_file(file, read_events(file), report)
        except Exception as exc:
            if strict:
                raise RuntimeError(f"{file}: conversion failed: {exc!r}") from exc
            continue
        finally:
            clear_agari_caches()

        if strict and len(report.errors) > before_errors:
            latest = report.errors[before_errors]
            raise RuntimeError(f"{file}: conversion reported error: {latest!r}")
        for record in records:
            try:
                yield build_policy_sample(record)
            except SampleFormatError:
                if strict:
                    raise
                continue
            emitted += 1
            if limit_records is not None and emitted >= limit_records:
                return


class JsonlPolicyDataset(IterableDataset):
    """IterableDataset for converter JSONL records."""

    def __init__(self, paths: Sequence[Path], *, vocab: MahjongVocab, limit_records: int | None = None) -> None:
        self.paths = list(paths)
        self.vocab = vocab
        self.limit_records = limit_records

    def __iter__(self) -> Iterator[TokenizedSample]:
        for sample in iter_jsonl_policy_samples(self.paths, limit_records=self.limit_records):
            yield encode_sample(sample, self.vocab)


class MjsonStreamingPolicyDataset(IterableDataset):
    """IterableDataset that converts mjson logs on the fly into tokenized samples."""

    def __init__(
        self,
        roots: Sequence[Path],
        *,
        vocab: MahjongVocab,
        limit_files: int | None = None,
        limit_records: int | None = None,
        strict: bool = True,
        shuffle_files: bool = False,
        seed: int = 0,
    ) -> None:
        self.roots = list(roots)
        self.vocab = vocab
        self.limit_files = limit_files
        self.limit_records = limit_records
        self.strict = strict
        self.shuffle_files = shuffle_files
        self.seed = seed

    def __iter__(self) -> Iterator[TokenizedSample]:
        files = _collect_mjson_files(
            self.roots,
            limit_files=self.limit_files,
            shuffle_files=self.shuffle_files,
            seed=self.seed,
        )
        worker = get_worker_info()
        if worker is not None:
            files = files[worker.id :: worker.num_workers]
        for sample in _iter_mjson_policy_samples_from_files(
            files,
            limit_records=self.limit_records,
            strict=self.strict,
        ):
            yield encode_sample(sample, self.vocab)


def iter_policy_samples(
    paths: Sequence[Path],
    *,
    data_format: str,
    limit_files: int | None = None,
    limit_records: int | None = None,
    strict: bool = True,
    shuffle_files: bool = False,
    seed: int = 0,
) -> Iterable[PolicySample]:
    """Yield policy samples for vocab building from either JSONL or mjson inputs."""
    if data_format == "jsonl":
        return iter_jsonl_policy_samples(paths, limit_records=limit_records)
    if data_format == "mjson":
        return iter_mjson_policy_samples(
            paths,
            limit_files=limit_files,
            limit_records=limit_records,
            strict=strict,
            shuffle_files=shuffle_files,
            seed=seed,
        )
    raise ValueError(f"unknown data_format: {data_format}")
