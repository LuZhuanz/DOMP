from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .samples import PolicySample, SampleFormatError


ACTION_END_RE = re.compile(r"</A\d+>")


@dataclass
class MahjongVocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    bos_token: str = "<BOS>"
    choice_token: str = "<CHOICE>"

    @classmethod
    def from_tokens(
        cls,
        tokens: Iterable[str],
        *,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        bos_token: str = "<BOS>",
        choice_token: str = "<CHOICE>",
    ) -> MahjongVocab:
        ordered: list[str] = []
        seen: set[str] = set()
        for token in (pad_token, unk_token, bos_token, choice_token, *tokens):
            if token not in seen:
                ordered.append(token)
                seen.add(token)
        return cls(
            token_to_id={token: i for i, token in enumerate(ordered)},
            id_to_token=ordered,
            pad_token=pad_token,
            unk_token=unk_token,
            bos_token=bos_token,
            choice_token=choice_token,
        )

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    def encode_tokens(self, tokens: Iterable[str]) -> list[int]:
        unk = self.unk_id
        return [self.token_to_id.get(token, unk) for token in tokens]

    def decode_ids(self, ids: Iterable[int]) -> list[str]:
        return [self.id_to_token[i] if 0 <= i < len(self.id_to_token) else self.unk_token for i in ids]

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "token_to_id": self.token_to_id,
            "special_tokens": {
                "pad": self.pad_token,
                "unk": self.unk_token,
                "bos": self.bos_token,
                "choice": self.choice_token,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> MahjongVocab:
        token_to_id = data.get("token_to_id")
        if not isinstance(token_to_id, dict):
            raise ValueError("vocab JSON must contain token_to_id")
        special = data.get("special_tokens", {})
        ordered = sorted(token_to_id.items(), key=lambda item: item[1])
        if [idx for _, idx in ordered] != list(range(len(ordered))):
            raise ValueError("token ids must be contiguous from 0")
        return cls(
            token_to_id=dict(token_to_id),
            id_to_token=[token for token, _ in ordered],
            pad_token=special.get("pad", "<PAD>"),
            unk_token=special.get("unk", "<UNK>"),
            bos_token=special.get("bos", "<BOS>"),
            choice_token=special.get("choice", "<CHOICE>"),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> MahjongVocab:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class TokenizedSample:
    input_ids: list[int]
    choice_position: int
    action_positions: list[int]
    label: int
    decision_type: str
    source: str | None = None


def tokenize_text(text: str) -> list[str]:
    """Tokenize converter text using the v1 whitespace tokenization contract."""
    return text.split()


def build_vocab(
    samples: Iterable[PolicySample],
    *,
    min_freq: int = 1,
    max_size: int | None = None,
) -> MahjongVocab:
    """Build a word-level vocabulary from leakage-safe policy samples."""
    if min_freq < 1:
        raise ValueError("min_freq must be positive")
    if max_size is not None and max_size < 4:
        raise ValueError("max_size must leave room for special tokens")

    counts: Counter[str] = Counter()
    for sample in samples:
        counts.update(tokenize_text(sample.input_text))

    candidates = [(token, count) for token, count in counts.items() if count >= min_freq]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    if max_size is not None:
        candidates = candidates[: max_size - 4]
    return MahjongVocab.from_tokens(token for token, _ in candidates)


def encode_sample(sample: PolicySample, vocab: MahjongVocab) -> TokenizedSample:
    """Encode one policy sample and extract choice/action positions."""
    tokens = tokenize_text(sample.input_text)
    if tokens.count(vocab.choice_token) != 1:
        raise SampleFormatError("sample must contain exactly one <CHOICE> token")
    choice_position = tokens.index(vocab.choice_token)
    action_positions = [i for i, token in enumerate(tokens) if ACTION_END_RE.fullmatch(token)]
    if len(action_positions) != sample.legal_action_count:
        raise SampleFormatError(
            f"found {len(action_positions)} action end tokens, expected {sample.legal_action_count}"
        )
    if not (0 <= sample.label < len(action_positions)):
        raise SampleFormatError(f"label {sample.label} is outside {len(action_positions)} actions")
    return TokenizedSample(
        input_ids=vocab.encode_tokens(tokens),
        choice_position=choice_position,
        action_positions=action_positions,
        label=sample.label,
        decision_type=sample.decision_type,
        source=sample.source,
    )
