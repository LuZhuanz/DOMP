from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from .agari import clear_agari_caches
from .converter import ConvertReport, convert_file, inspect_paths
from .io import iter_mjson_files, read_events


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mjgpt-convert")
    sub = parser.add_subparsers(dest="cmd", required=True)

    inspect_p = sub.add_parser("inspect", help="Inspect mjson event coverage.")
    inspect_p.add_argument("input", type=Path)

    convert_p = sub.add_parser("convert", help="Convert mjson logs to long decision JSONL.")
    convert_p.add_argument("input", type=Path)
    convert_p.add_argument("--out", type=Path, default=Path("out/v0/decisions.long.jsonl"))
    convert_p.add_argument("--report", type=Path, default=Path("out/v0/report.json"))
    convert_p.add_argument("--limit-files", type=int, default=None)

    validate_p = sub.add_parser("validate", help="Run conversion and write only the validation report.")
    validate_p.add_argument("input", type=Path)
    validate_p.add_argument("--report", type=Path, default=Path("out/v0/report.json"))
    validate_p.add_argument("--limit-files", type=int, default=None)

    args = parser.parse_args(argv)
    if args.cmd == "inspect":
        print(json.dumps(inspect_paths(args.input), ensure_ascii=False, indent=2))
        return

    if args.cmd == "convert":
        count, report = _stream_convert(args.input, args.out, limit_files=args.limit_files)
        _write_report(args.report, report.as_dict() | {"records_written": count})
        print(f"wrote {count} records to {args.out}")
        print(f"wrote report to {args.report}")
        return

    count, report = _stream_validate(args.input, limit_files=args.limit_files)
    _write_report(args.report, report.as_dict() | {"records_scanned": count})
    print(f"validated {count} records")
    print(f"wrote report to {args.report}")


def _stream_convert(input_path: Path, out_path: Path, limit_files: int | None = None) -> tuple[int, ConvertReport]:
    report = ConvertReport()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    opener = gzip.open if out_path.suffix == ".gz" else Path.open
    with opener(out_path, "wt", encoding="utf-8") as out:
        for file in _selected_files(input_path, limit_files):
            report.files += 1
            try:
                records = convert_file(file, read_events(file), report)
                for record in records:
                    out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    count += 1
                clear_agari_caches()
            except Exception as exc:
                report.errors.append({"file": str(file), "error": repr(exc), "scope": "file"})
    return count, report


def _stream_validate(input_path: Path, limit_files: int | None = None) -> tuple[int, ConvertReport]:
    report = ConvertReport()
    count = 0
    for file in _selected_files(input_path, limit_files):
        report.files += 1
        try:
            count += len(convert_file(file, read_events(file), report))
            clear_agari_caches()
        except Exception as exc:
            report.errors.append({"file": str(file), "error": repr(exc), "scope": "file"})
    return count, report


def _selected_files(input_path: Path, limit_files: int | None) -> list[Path]:
    files = iter_mjson_files(input_path)
    if limit_files is None:
        return files
    if limit_files < 1:
        raise ValueError("--limit-files must be positive")
    return files[:limit_files]


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
