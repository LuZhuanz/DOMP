from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable


def iter_mjson_files(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"mjson input path does not exist: {path}")
    if path.is_file():
        return [path]
    files = sorted(path.rglob("*.mjson"))
    if not files:
        raise FileNotFoundError(f"no .mjson files found under: {path}")
    return files


def open_mjson(path: Path):
    with path.open("rb") as raw:
        magic = raw.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def read_events(path: Path) -> list[dict]:
    events: list[dict] = []
    with open_mjson(path) as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            event = json.loads(line)
            event["_line"] = line_no
            events.append(event)
    return events


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "wt", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count
