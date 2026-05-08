from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


CHOICE_RE = re.compile(r"^A(\d+)$")


class SampleFormatError(ValueError):
    """Raised when a converter record cannot be used as a policy sample."""


@dataclass(frozen=True)
class PolicySample:
    input_text: str
    label: int
    legal_action_count: int
    decision_type: str
    source: str | None = None


def build_policy_sample(record: Mapping[str, Any], *, source: str | None = None) -> PolicySample:
    """Build a leakage-safe supervised policy sample from one converter record."""
    try:
        state_text = _require_str(record, "state_text")
        choice_id = _require_str(record, "choice_id")
        decision_type = _require_str(record, "decision_type")
    except KeyError as exc:
        raise SampleFormatError(f"missing required field: {exc.args[0]}") from exc

    legal_actions = record.get("legal_actions")
    if not isinstance(legal_actions, list):
        raise SampleFormatError("legal_actions must be a list")
    legal_action_count = len(legal_actions)
    if legal_action_count < 1:
        raise SampleFormatError("legal_actions must not be empty")

    label = _parse_choice_id(choice_id)
    if label >= legal_action_count:
        raise SampleFormatError(f"choice_id {choice_id!r} is outside {legal_action_count} legal actions")

    _validate_action_rows(legal_actions)
    input_text = _strip_answer_from_state_text(state_text)
    if "<EXECUTE>" in input_text or "</EXECUTE>" in input_text:
        raise SampleFormatError("input_text still contains execute block")

    if source is None:
        source = _default_source(record)
    return PolicySample(
        input_text=input_text,
        label=label,
        legal_action_count=legal_action_count,
        decision_type=decision_type,
        source=source,
    )


def _require_str(record: Mapping[str, Any], key: str) -> str:
    value = record[key]
    if not isinstance(value, str) or not value:
        raise SampleFormatError(f"{key} must be a non-empty string")
    return value


def _parse_choice_id(choice_id: str) -> int:
    match = CHOICE_RE.fullmatch(choice_id)
    if match is None:
        raise SampleFormatError(f"choice_id must look like A0, got {choice_id!r}")
    return int(match.group(1))


def _strip_answer_from_state_text(state_text: str) -> str:
    if state_text.count("<CHOICE>") != 1:
        raise SampleFormatError("state_text must contain exactly one <CHOICE>")
    before_choice, _after_choice = state_text.split("<CHOICE>", 1)
    input_text = before_choice.rstrip() + "\n<CHOICE>"
    if "<LEGAL_ACTIONS>" not in input_text or "</LEGAL_ACTIONS>" not in input_text:
        raise SampleFormatError("input_text must contain a complete legal action block")
    if input_text.count("<LEGAL_ACTIONS>") != 1 or input_text.count("</LEGAL_ACTIONS>") != 1:
        raise SampleFormatError("input_text must contain exactly one legal action block")
    return input_text


def _validate_action_rows(legal_actions: list[Any]) -> None:
    for i, row in enumerate(legal_actions):
        if not isinstance(row, Mapping):
            raise SampleFormatError(f"legal_actions[{i}] must be an object")
        expected_id = f"A{i}"
        if row.get("id") != expected_id:
            raise SampleFormatError(f"legal_actions[{i}].id must be {expected_id!r}")
        if not isinstance(row.get("text"), str) or not row.get("text"):
            raise SampleFormatError(f"legal_actions[{i}].text must be a non-empty string")


def _default_source(record: Mapping[str, Any]) -> str | None:
    source_file = record.get("source_file")
    decision_index = record.get("decision_index")
    if source_file is None:
        return None
    if decision_index is None:
        return str(source_file)
    return f"{source_file}#{decision_index}"
