from __future__ import annotations

from collections import Counter

from .actions import LegalAction
from .agari import is_complete_hand, is_tenpai, terminal_honor_count
from .state import GameState
from .tiles import (
    base_counter,
    from_rel,
    normalize_tile,
    physical_choices,
    sort_tiles,
    tile_base,
    tile_index,
)


def _discard_actions(state: GameState, actor: int, allow_riichi: bool = True) -> list[LegalAction]:
    player = state.players[actor]
    actions: list[LegalAction] = []
    if player.riichi:
        if player.draw is not None:
            actions.append(LegalAction("DISCARD", f"DISCARD {player.draw} TSUMOGIRI"))
        return actions

    seen: set[tuple[str, str]] = set()
    if player.draw is not None:
        key = (player.draw, "TSUMOGIRI")
        seen.add(key)
        actions.append(LegalAction("DISCARD", f"DISCARD {player.draw} TSUMOGIRI"))
    for tile in sort_tiles(player.hand):
        key = (tile, "HAND")
        if key in seen:
            continue
        seen.add(key)
        actions.append(LegalAction("DISCARD", f"DISCARD {tile} HAND"))

    if allow_riichi and can_riichi(state, actor):
        riichi_actions: list[LegalAction] = []
        for action in actions:
            discard_tile = action.text.split()[1]
            source = action.text.split()[2]
            tiles_after = list(player.decision_tiles)
            remove_one_base(tiles_after, discard_tile)
            if is_tenpai(tiles_after, player.open_meld_count):
                riichi_actions.append(LegalAction("RIICHI", f"RIICHI DISCARD {discard_tile} {source}"))
        actions = riichi_actions + actions
    return actions


def remove_one_base(tiles: list[str], tile: str) -> None:
    # Hands are tiny, but using pop(i) makes the physical removal explicit.
    base = tile_base(tile)
    for i, candidate in enumerate(tiles):
        if tile_base(candidate) == base:
            tiles.pop(i)
            return
    raise ValueError(f"tile {tile} not found")


def can_riichi(state: GameState, actor: int) -> bool:
    player = state.players[actor]
    return (
        player.draw is not None
        and player.is_menzen
        and not player.riichi
        and state.scores[actor] >= 1000
        and state.wall_left >= 4
    )


def kan_actions(state: GameState, actor: int) -> list[LegalAction]:
    player = state.players[actor]
    actions: list[LegalAction] = []
    counts = base_counter(player.decision_tiles)
    for base, count in sorted(counts.items(), key=lambda item: tile_index(item[0])):
        if count >= 4:
            actions.append(LegalAction("ANKAN", format_ankan_action(player.decision_tiles, base)))
    for meld in player.melds:
        if meld.kind != "PON":
            continue
        base = tile_base(meld.tiles[0])
        if any(tile_base(t) == base for t in player.decision_tiles):
            actions.append(LegalAction("KAKAN", f"KAKAN {base}"))
    return actions


def self_turn_actions(state: GameState, actor: int) -> list[LegalAction]:
    """Return legal actions for a player decision after drawing or rinshan draw."""
    player = state.players[actor]
    actions: list[LegalAction] = []
    if player.draw is not None and is_complete_hand(player.decision_tiles, player.open_meld_count):
        actions.append(LegalAction("TSUMO", "TSUMO"))
    if is_kyuushu_kyuuhai(state, actor):
        actions.append(LegalAction("KYUUSHU_KYUUHAI", "KYUUSHU_KYUUHAI"))
    actions.extend(kan_actions(state, actor))
    actions.extend(_discard_actions(state, actor, allow_riichi=True))
    return _dedupe_actions(actions)


def meld_discard_actions(state: GameState, actor: int) -> list[LegalAction]:
    return _dedupe_actions(_discard_actions(state, actor, allow_riichi=False))


def is_kyuushu_kyuuhai(state: GameState, actor: int) -> bool:
    player = state.players[actor]
    if state.any_call or player.river or player.draw is None:
        return False
    return terminal_honor_count(player.decision_tiles) >= 9


def reaction_actions(state: GameState, actor: int, target: int, tile: str) -> list[LegalAction]:
    """Return legal PASS/claim actions for actor after target discards tile."""
    player = state.players[actor]
    tile = normalize_tile(tile)
    actions = [LegalAction("PASS", "PASS")]
    if is_complete_hand(player.hand + [tile], player.open_meld_count):
        actions.append(LegalAction("RON", "RON"))
    if not player.riichi:
        actions.extend(chi_actions(state, actor, target, tile))
        actions.extend(pon_actions(state, actor, target, tile))
        actions.extend(minkan_actions(state, actor, target, tile))
    return _dedupe_actions(actions)


def chi_actions(state: GameState, actor: int, target: int, tile: str) -> list[LegalAction]:
    if (target + 1) % 4 != actor:
        return []
    base = tile_base(tile)
    idx = tile_index(base)
    if idx >= 27:
        return []
    number = idx % 9 + 1
    suit = base[1]
    hand = state.players[actor].hand
    actions: list[LegalAction] = []
    for start in (number - 2, number - 1, number):
        if start < 1 or start + 2 > 9:
            continue
        seq = [f"{start}{suit}", f"{start + 1}{suit}", f"{start + 2}{suit}"]
        needed = [b for b in seq if b != base]
        if len(needed) != 2:
            continue
        first_choices = physical_choices(hand, needed[0], 1)
        second_choices = physical_choices(hand, needed[1], 1)
        for first in first_choices:
            for second in second_choices:
                consumed = list(first + second)
                if len(consumed) != 2 or Counter(tile_base(t) for t in consumed) != Counter(needed):
                    continue
                meld_tiles = sort_tiles(consumed + [tile])
                actions.append(
                    LegalAction("CHI", "CHI " + " ".join(meld_tiles) + f" {from_rel(target, actor)}")
                )
    return actions


def pon_actions(state: GameState, actor: int, target: int, tile: str) -> list[LegalAction]:
    base = tile_base(tile)
    actions: list[LegalAction] = []
    for consumed in physical_choices(state.players[actor].hand, base, 2):
        meld_tiles = sort_tiles(list(consumed) + [tile])
        actions.append(LegalAction("PON", "PON " + " ".join(meld_tiles) + f" {from_rel(target, actor)}"))
    return actions


def minkan_actions(state: GameState, actor: int, target: int, tile: str) -> list[LegalAction]:
    base = tile_base(tile)
    actions: list[LegalAction] = []
    for consumed in physical_choices(state.players[actor].hand, base, 3):
        meld_tiles = sort_tiles(list(consumed) + [tile])
        actions.append(
            LegalAction("MINKAN", "MINKAN " + " ".join(meld_tiles) + f" {from_rel(target, actor)}")
        )
    return actions


def format_ankan_action(tiles: list[str], tile: str) -> str:
    """Build the canonical ANKAN action text from physical visible self tiles."""
    base = tile_base(tile)
    kan_tiles = sort_tiles([t for t in tiles if tile_base(t) == base])[:4]
    if len(kan_tiles) != 4:
        raise ValueError(f"cannot form ANKAN {base} from {tiles}")
    return "ANKAN " + " ".join(kan_tiles)


def _dedupe_actions(actions: list[LegalAction]) -> list[LegalAction]:
    seen: set[str] = set()
    out: list[LegalAction] = []
    order = {
        "PASS": 0,
        "TSUMO": 1,
        "RON": 2,
        "KYUUSHU_KYUUHAI": 3,
        "ANKAN": 4,
        "KAKAN": 5,
        "MINKAN": 6,
        "RIICHI": 7,
        "DISCARD": 8,
        "CHI": 9,
        "PON": 10,
    }
    for action in sorted(actions, key=lambda a: (order.get(a.kind, 99), a.text)):
        if action.text not in seen:
            seen.add(action.text)
            out.append(action)
    return out
