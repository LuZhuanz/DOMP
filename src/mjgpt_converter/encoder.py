from __future__ import annotations

from .actions import LegalAction, assign_ids
from .state import GameState
from .tiles import player_order, rel_name, sort_tiles


def encode_state(
    state: GameState,
    actor: int,
    decision_type: str,
    legal_actions: list[LegalAction],
    choice_id: str,
    execute: str,
    trigger_actor: int | None = None,
    trigger_tile: str | None = None,
) -> tuple[str, list[dict]]:
    """Encode a visible decision state and numbered legal action table."""
    numbered = assign_ids(legal_actions)
    lines: list[str] = [
        "<BOS>",
        f"<ROUND> {state.round_name()}",
        f"<HONBA> {state.honba}",
        f"<KYOTAKU> {state.kyotaku}",
        f"<BAKAZE> {state.bakaze}",
        f"<DEALER> {rel_name(state.oya, actor)}",
        f"<WALL_LEFT> {state.wall_left}",
        "<DORA_INDICATORS> " + " ".join(state.dora_markers),
        f"<TN> {state.turn_number()}",
        "",
        "<WINDS>",
        *state.wind_lines(actor),
        "",
        "<SCORES>",
        *state.score_lines(actor),
        "",
        "<PLAYER_STATES>",
        *state.player_state_lines(actor),
        "",
        "<SELF_HAND>",
        " ".join(sort_tiles(state.players[actor].hand)) or "EMPTY",
        "",
        "<DRAW>",
        state.players[actor].draw or "EMPTY",
        "",
        "<MELDS>",
    ]
    lines.extend(state.visible_meld_line(p, actor) for p in player_order(actor))
    lines.extend(["", "<RIVERS>"])
    lines.extend(state.river_line(p, actor) for p in player_order(actor))
    lines.extend(["", "<DECISION>", f"TYPE {decision_type}", "ACTOR SELF"])
    if trigger_actor is not None and trigger_tile is not None:
        lines.append(f"TRIGGER_ACTOR {rel_name(trigger_actor, actor)}")
        lines.append(f"TRIGGER_TILE {trigger_tile}")
    lines.extend(["", "<LEGAL_ACTIONS>"])
    action_rows: list[dict] = []
    for action_id, action in numbered:
        lines.append(f"<{action_id}> {action.text} </{action_id}>")
        action_rows.append({"id": action_id, "kind": action.kind, "text": action.text})
    lines.extend(
        [
            "</LEGAL_ACTIONS>",
            "",
            "<CHOICE>",
            f"<{choice_id}>",
            "</CHOICE>",
            "",
            "<EXECUTE>",
            execute,
            "</EXECUTE>",
            "",
            "</DECISION>",
            "<EOS>",
        ]
    )
    return "\n".join(lines), action_rows
