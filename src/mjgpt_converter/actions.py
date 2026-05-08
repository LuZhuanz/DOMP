from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LegalAction:
    kind: str
    text: str


def assign_ids(actions: list[LegalAction]) -> list[tuple[str, LegalAction]]:
    return [(f"A{i}", action) for i, action in enumerate(actions)]
