from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from mjgpt_converter.tiles import REL_NAMES, WIND_NAMES, all_tile_bases

from .samples import PolicySample, SampleFormatError


ACTION_END_RE = re.compile(r"</A\d+>")
INT_RE = re.compile(r"-?\d+")
RAW_SCORE_MODE = "raw"
SCORE_BUCKET_MODE = "score_bucket"
SCORE_BUCKET_STEP = 5_000
SCORE_BUCKET_MAX = 60_000
DEFAULT_MAX_ACTIONS = 64
MAX_ACTIONS_LIMIT = 1_024
DEFAULT_SMALL_INT_MAX = 100


@dataclass
class MahjongVocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    bos_token: str = "<BOS>"
    choice_token: str = "<CHOICE>"
    tokenization: dict[str, object] | None = None

    @classmethod
    def from_tokens(
        cls,
        tokens: Iterable[str],
        *,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        bos_token: str = "<BOS>",
        choice_token: str = "<CHOICE>",
        tokenization: dict[str, object] | None = None,
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
            tokenization=tokenization,
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

    def score_mode(self) -> str:
        if not self.tokenization:
            return RAW_SCORE_MODE
        mode = self.tokenization.get("score_mode", RAW_SCORE_MODE)
        if not isinstance(mode, str):
            return RAW_SCORE_MODE
        return mode

    def to_dict(self) -> dict:
        data = {
            "version": 1,
            "token_to_id": self.token_to_id,
            "special_tokens": {
                "pad": self.pad_token,
                "unk": self.unk_token,
                "bos": self.bos_token,
                "choice": self.choice_token,
            },
        }
        if self.tokenization:
            data["tokenization"] = self.tokenization
        return data

    @classmethod
    def from_dict(cls, data: dict) -> MahjongVocab:
        token_to_id = data.get("token_to_id")
        if not isinstance(token_to_id, dict):
            raise ValueError("vocab JSON must contain token_to_id")
        special = data.get("special_tokens", {})
        tokenization = data.get("tokenization")
        if tokenization is not None and not isinstance(tokenization, dict):
            raise ValueError("tokenization must be an object")
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
            tokenization=tokenization,
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


def tokenize_text(text: str, *, score_mode: str = RAW_SCORE_MODE) -> list[str]:
    """Tokenize converter text using the v1 whitespace tokenization contract."""
    if score_mode == RAW_SCORE_MODE:
        return text.split()
    if score_mode != SCORE_BUCKET_MODE:
        raise ValueError(f"unknown score_mode: {score_mode}")

    tokens: list[str] = []
    in_scores = False
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "<SCORES>":
            in_scores = True
            tokens.extend(parts)
            continue
        if in_scores and parts[0].startswith("<"):
            in_scores = False
        if in_scores and len(parts) == 2 and parts[0] in REL_NAMES and INT_RE.fullmatch(parts[1]):
            tokens.extend([parts[0], score_bucket_token(int(parts[1]))])
            continue
        tokens.extend(parts)
    return tokens


def score_bucket_token(score: int) -> str:
    if score < 0:
        return "<SCORE_NEG>"
    if score >= SCORE_BUCKET_MAX:
        return f"<SCORE_{SCORE_BUCKET_MAX:05d}_PLUS>"
    start = (score // SCORE_BUCKET_STEP) * SCORE_BUCKET_STEP
    end = start + SCORE_BUCKET_STEP - 1
    return f"<SCORE_{start:05d}_{end:05d}>"


def build_vocab(
    samples: Iterable[PolicySample],
    *,
    min_freq: int = 1,
    max_size: int | None = None,
    score_mode: str = RAW_SCORE_MODE,
) -> MahjongVocab:
    """Build a word-level vocabulary from leakage-safe policy samples."""
    if min_freq < 1:
        raise ValueError("min_freq must be positive")
    if max_size is not None and max_size < 4:
        raise ValueError("max_size must leave room for special tokens")

    counts: Counter[str] = Counter()
    for sample in samples:
        counts.update(tokenize_text(sample.input_text, score_mode=score_mode))

    candidates = [(token, count) for token, count in counts.items() if count >= min_freq]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    if max_size is not None:
        candidates = candidates[: max_size - 4]
    return MahjongVocab.from_tokens(
        (token for token, _ in candidates),
        tokenization={"score_mode": score_mode} if score_mode != RAW_SCORE_MODE else None,
    )


def build_fixed_vocab(
    *,
    max_actions: int = DEFAULT_MAX_ACTIONS,
    small_int_max: int = DEFAULT_SMALL_INT_MAX,
) -> MahjongVocab:
    """Build a stable v1 text vocab without scanning training data."""
    if max_actions < 1:
        raise ValueError("max_actions must be positive")
    if max_actions > MAX_ACTIONS_LIMIT:
        raise ValueError(f"max_actions must be <= {MAX_ACTIONS_LIMIT}")
    if small_int_max < 1:
        raise ValueError("small_int_max must be positive")

    tokens: list[str] = []
    tokens.extend(_structure_tokens())
    tokens.extend(_semantic_tokens())
    tokens.extend(_tile_tokens())
    tokens.extend(_action_boundary_tokens(max_actions))
    tokens.extend(str(i) for i in range(small_int_max + 1))
    tokens.extend(_score_bucket_tokens())
    return MahjongVocab.from_tokens(tokens, tokenization={"score_mode": SCORE_BUCKET_MODE})


def encode_sample(sample: PolicySample, vocab: MahjongVocab) -> TokenizedSample:
    """Encode one policy sample and extract choice/action positions."""
    tokens = tokenize_text(sample.input_text, score_mode=vocab.score_mode())
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


def _structure_tokens() -> list[str]:
    return [
        "<ROUND>",
        "<HONBA>",
        "<KYOTAKU>",
        "<BAKAZE>",
        "<DEALER>",
        "<WALL_LEFT>",
        "<DORA_INDICATORS>",
        "<TN>",
        "<WINDS>",
        "<SCORES>",
        "<PLAYER_STATES>",
        "<SELF_HAND>",
        "<DRAW>",
        "<MELDS>",
        "<RIVERS>",
        "<DECISION>",
        "</DECISION>",
        "<LEGAL_ACTIONS>",
        "</LEGAL_ACTIONS>",
        "</CHOICE>",
        "<EXECUTE>",
        "</EXECUTE>",
        "<EOS>",
    ]


def _semantic_tokens() -> list[str]:
    tokens = [
        "ACTOR",
        "ANKAN",
        "CALLED_BY_KAMI",
        "CALLED_BY_SELF",
        "CALLED_BY_SHIMO",
        "CALLED_BY_TOIMEN",
        "CHI",
        "DISCARD",
        "EMPTY",
        "FROM_KAMI",
        "FROM_SELF",
        "FROM_SHIMO",
        "FROM_TOIMEN",
        "HAND",
        "IPPATSU",
        "KAKAN",
        "KAMI",
        "KYUUSHU_KYUUHAI",
        "MENZEN",
        "MINKAN",
        "NO_IPPATSU",
        "NO_RIICHI",
        "OPEN",
        "PASS",
        "PON",
        "POST_RIICHI",
        "REACTION_TO_DISCARD",
        "RIICHI",
        "RIICHI_DECL",
        "RON",
        "SELF",
        "SELF_TURN_AFTER_DRAW",
        "SELF_TURN_AFTER_MELD",
        "SHIMO",
        "TOIMEN",
        "TRIGGER_ACTOR",
        "TRIGGER_TILE",
        "TSUMO",
        "TSUMOGIRI",
        "TYPE",
    ]
    tokens.extend(WIND_NAMES)
    tokens.extend(f"{wind}{kyoku}" for wind in WIND_NAMES for kyoku in range(1, 5))
    return tokens


def _tile_tokens() -> list[str]:
    return [*all_tile_bases(), "5mr", "5pr", "5sr"]


def _action_boundary_tokens(max_actions: int) -> list[str]:
    tokens: list[str] = []
    for i in range(max_actions):
        tokens.extend([f"<A{i}>", f"</A{i}>"])
    return tokens


def _score_bucket_tokens() -> list[str]:
    tokens = ["<SCORE_NEG>"]
    for start in range(0, SCORE_BUCKET_MAX, SCORE_BUCKET_STEP):
        end = start + SCORE_BUCKET_STEP - 1
        tokens.append(f"<SCORE_{start:05d}_{end:05d}>")
    tokens.append(f"<SCORE_{SCORE_BUCKET_MAX:05d}_PLUS>")
    return tokens
