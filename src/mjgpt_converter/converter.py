from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .actions import LegalAction
from .encoder import encode_state
from .io import iter_mjson_files, read_events
from .rules import format_ankan_action, meld_discard_actions, reaction_actions, self_turn_actions
from .state import GameState
from .tiles import from_rel, normalize_tile, sort_tiles, tile_base


REACTION_TYPES = {"chi", "pon", "daiminkan", "hora"}


@dataclass
class ConvertReport:
    files: int = 0
    events: Counter[str] = field(default_factory=Counter)
    decisions: Counter[str] = field(default_factory=Counter)
    invalid_decisions: int = 0
    errors: list[dict] = field(default_factory=list)
    unknown_events: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict:
        return {
            "files": self.files,
            "events": dict(self.events),
            "decisions": dict(self.decisions),
            "invalid_decisions": self.invalid_decisions,
            "unknown_events": dict(self.unknown_events),
            "errors": self.errors[:500],
            "notes": [
                "Agari legality is shape-based in v0; yaku and furiten are not fully enforced.",
                "Rule-profile differences are surfaced as actual_added_to_legal validation flags.",
            ],
        }


def inspect_paths(path: Path) -> dict:
    files = iter_mjson_files(path)
    counts: Counter[str] = Counter()
    keys: dict[str, set[str]] = defaultdict(set)
    bad: list[dict] = []
    for file in files:
        try:
            for event in read_events(file):
                typ = event.get("type", "UNKNOWN")
                counts[typ] += 1
                keys[typ].update(k for k in event if not k.startswith("_"))
        except Exception as exc:  # pragma: no cover - diagnostics path
            bad.append({"file": str(file), "error": repr(exc)})
    return {
        "files": len(files),
        "events": dict(counts),
        "event_keys": {k: sorted(v) for k, v in sorted(keys.items())},
        "bad_files": bad,
    }


def convert_paths(path: Path) -> tuple[list[dict], ConvertReport]:
    records: list[dict] = []
    report = ConvertReport()
    for file in iter_mjson_files(path):
        report.files += 1
        try:
            events = read_events(file)
            records.extend(convert_file(file, events, report))
        except Exception as exc:
            report.errors.append({"file": str(file), "error": repr(exc), "scope": "file"})
    return records, report


def convert_file(path: Path, events: list[dict], report: ConvertReport) -> list[dict]:
    """Replay one mjson event stream and return auditable decision records."""
    state = GameState()
    game_id = path.stem
    records: list[dict] = []
    decision_index = 0
    for i, event in enumerate(events):
        typ = event.get("type")
        report.events[typ] += 1
        try:
            if typ == "start_game":
                state.start_game(event, game_id)
            elif typ == "start_kyoku":
                state.start_kyoku(event)
            elif typ == "tsumo":
                state.on_tsumo(event["actor"], event["pai"])
            elif typ == "reach":
                next_dahai = _find_next(events, i + 1, {"dahai"})
                if next_dahai is not None:
                    decision_index += 1
                    records.append(
                        _make_self_sample(
                            state,
                            event["actor"],
                            "SELF_TURN_AFTER_DRAW",
                            _discard_execute(next_dahai, riichi=True),
                            path,
                            event,
                            decision_index,
                            report,
                        )
                    )
                state.on_reach(event["actor"])
            elif typ == "reach_accepted":
                state.on_reach_accepted(event["actor"])
            elif typ == "dahai":
                actor = event["actor"]
                if not state.players[actor].pending_reach:
                    decision_index += 1
                    decision_type = "SELF_TURN_AFTER_DRAW" if state.players[actor].draw else "SELF_TURN_AFTER_MELD"
                    actions = self_turn_actions(state, actor) if state.players[actor].draw else meld_discard_actions(state, actor)
                    records.append(
                        _make_sample(
                            state,
                            actor,
                            decision_type,
                            actions,
                            _discard_execute(event, riichi=False),
                            path,
                            event,
                            decision_index,
                            report,
                        )
                    )
                discarded = state.on_dahai(actor, event["pai"], event["tsumogiri"])
                decision_index = _emit_reaction_samples(
                    records,
                    state,
                    path,
                    events,
                    i,
                    discarded,
                    decision_index,
                    report,
                )
            elif typ == "chi":
                state.on_chi(event["actor"], event["target"], event["pai"], event["consumed"])
            elif typ == "pon":
                state.on_pon(event["actor"], event["target"], event["pai"], event["consumed"])
            elif typ == "daiminkan":
                state.on_daiminkan(event["actor"], event["target"], event["pai"], event["consumed"])
            elif typ == "ankan":
                decision_index += 1
                execute = format_ankan_action(state.players[event["actor"]].decision_tiles, event["consumed"][0])
                records.append(
                    _make_self_sample(
                        state,
                        event["actor"],
                        "SELF_TURN_AFTER_DRAW",
                        execute,
                        path,
                        event,
                        decision_index,
                        report,
                    )
                )
                state.on_ankan(event["actor"], event["consumed"])
            elif typ == "kakan":
                decision_index += 1
                tile = normalize_tile(event.get("pai") or event.get("consumed", [""])[-1])
                execute = f"KAKAN {tile_base(tile)}"
                records.append(
                    _make_self_sample(
                        state,
                        event["actor"],
                        "SELF_TURN_AFTER_DRAW",
                        execute,
                        path,
                        event,
                        decision_index,
                        report,
                    )
                )
                state.on_kakan(event["actor"], tile)
            elif typ == "dora":
                state.add_dora(event["dora_marker"])
            elif typ == "hora":
                if event["actor"] == event["target"] and state.players[event["actor"]].draw is not None:
                    decision_index += 1
                    records.append(
                        _make_self_sample(
                            state,
                            event["actor"],
                            "SELF_TURN_AFTER_DRAW",
                            "TSUMO",
                            path,
                            event,
                            decision_index,
                            report,
                        )
                    )
            elif typ in {"ryukyoku", "end_kyoku", "end_game"}:
                pass
            else:
                report.unknown_events[typ] += 1
        except Exception as exc:
            report.errors.append(
                {
                    "file": str(path),
                    "line": event.get("_line"),
                    "event": {k: v for k, v in event.items() if not k.startswith("_")},
                    "error": repr(exc),
                }
            )
    return records


def _find_next(events: list[dict], start: int, types: set[str]) -> dict | None:
    for event in events[start:]:
        if event.get("type") in types:
            return event
        if event.get("type") in {"end_kyoku", "start_kyoku", "end_game"}:
            return None
    return None


def _discard_execute(event: dict, riichi: bool) -> str:
    tile = normalize_tile(event["pai"])
    source = "TSUMOGIRI" if event["tsumogiri"] else "HAND"
    prefix = "RIICHI " if riichi else ""
    return f"{prefix}DISCARD {tile} {source}"


def _emit_reaction_samples(
    records: list[dict],
    state: GameState,
    path: Path,
    events: list[dict],
    event_index: int,
    discarded: str,
    decision_index: int,
    report: ConvertReport,
) -> int:
    target = events[event_index]["actor"]
    actual_by_actor = _actual_reactions(events, event_index + 1, target)
    for actor in ((target + 1) % 4, (target + 2) % 4, (target + 3) % 4):
        actions = reaction_actions(state, actor, target, discarded)
        if len(actions) <= 1 and actor not in actual_by_actor:
            continue
        execute = actual_by_actor.get(actor, "PASS")
        decision_index += 1
        records.append(
            _make_sample(
                state,
                actor,
                "REACTION_TO_DISCARD",
                actions,
                execute,
                path,
                events[event_index],
                decision_index,
                report,
                trigger_actor=target,
                trigger_tile=discarded,
            )
        )
    return decision_index


def _actual_reactions(events: list[dict], start: int, target: int) -> dict[int, str]:
    actual: dict[int, str] = {}
    for event in events[start:]:
        typ = event.get("type")
        if typ == "reach_accepted":
            continue
        if typ not in REACTION_TYPES:
            break
        if event.get("target") != target:
            break
        actor = event["actor"]
        if typ == "hora":
            actual[actor] = "RON"
        elif typ == "chi":
            tiles = sort_tiles([normalize_tile(event["pai"]), *[normalize_tile(t) for t in event["consumed"]]])
            actual[actor] = "CHI " + " ".join(tiles) + f" {from_rel(target, actor)}"
        elif typ == "pon":
            tiles = sort_tiles([normalize_tile(event["pai"]), *[normalize_tile(t) for t in event["consumed"]]])
            actual[actor] = "PON " + " ".join(tiles) + f" {from_rel(target, actor)}"
        elif typ == "daiminkan":
            tiles = sort_tiles([normalize_tile(event["pai"]), *[normalize_tile(t) for t in event["consumed"]]])
            actual[actor] = "MINKAN " + " ".join(tiles) + f" {from_rel(target, actor)}"
    return actual


def _make_self_sample(
    state: GameState,
    actor: int,
    decision_type: str,
    execute: str,
    path: Path,
    event: dict,
    decision_index: int,
    report: ConvertReport,
) -> dict:
    return _make_sample(
        state,
        actor,
        decision_type,
        self_turn_actions(state, actor),
        execute,
        path,
        event,
        decision_index,
        report,
    )


def _make_sample(
    state: GameState,
    actor: int,
    decision_type: str,
    actions: list[LegalAction],
    execute: str,
    path: Path,
    event: dict,
    decision_index: int,
    report: ConvertReport,
    trigger_actor: int | None = None,
    trigger_tile: str | None = None,
) -> dict:
    report.decisions[decision_type] += 1
    validation_flags: list[str] = []
    action_texts = [a.text for a in actions]
    if execute not in action_texts:
        validation_flags.append("actual_added_to_legal")
        report.invalid_decisions += 1
        actions = list(actions) + [LegalAction(_kind_from_execute(execute), execute)]
        action_texts.append(execute)
        report.errors.append(
            {
                "file": str(path),
                "line": event.get("_line"),
                "decision_type": decision_type,
                "execute": execute,
                "error": "actual action not generated by legal action rules",
            }
        )
    choice_id = f"A{action_texts.index(execute)}"
    state_text, action_rows = encode_state(
        state,
        actor,
        decision_type,
        actions,
        choice_id,
        execute,
        trigger_actor=trigger_actor,
        trigger_tile=trigger_tile,
    )
    return {
        "game_id": state.game_id,
        "source_file": str(path),
        "source_line": event.get("_line"),
        "kyoku_index": state.kyoku_index,
        "decision_index": decision_index,
        "actor": actor,
        "decision_type": decision_type,
        "state_text": state_text,
        "legal_actions": action_rows,
        "choice_id": choice_id,
        "execute": execute,
        "validation_flags": validation_flags,
    }


def _kind_from_execute(execute: str) -> str:
    if execute.startswith("RIICHI"):
        return "RIICHI"
    return execute.split()[0]
