from __future__ import annotations

from dataclasses import dataclass, field

from .tiles import (
    called_by_rel,
    from_rel,
    normalize_tile,
    normalize_tiles,
    player_order,
    rel_name,
    remove_many,
    remove_one,
    seat_wind,
    sort_tiles,
    tile_base,
)


@dataclass
class RiverDiscard:
    tile: str
    source: str
    riichi_decl: bool = False
    post_riichi: bool = False
    called_by: int | None = None


@dataclass
class Meld:
    kind: str
    tiles: list[str]
    source: int | None = None
    opened: bool = True


@dataclass
class PlayerState:
    hand: list[str] = field(default_factory=list)
    draw: str | None = None
    melds: list[Meld] = field(default_factory=list)
    river: list[RiverDiscard] = field(default_factory=list)
    riichi: bool = False
    ippatsu: bool = False
    pending_reach: bool = False

    @property
    def closed_tiles(self) -> list[str]:
        return sort_tiles(self.hand)

    @property
    def decision_tiles(self) -> list[str]:
        tiles = list(self.hand)
        if self.draw is not None:
            tiles.append(self.draw)
        return tiles

    @property
    def open_meld_count(self) -> int:
        return len(self.melds)

    @property
    def is_menzen(self) -> bool:
        return not any(m.opened for m in self.melds)


@dataclass
class GameState:
    game_id: str = ""
    names: list[str] = field(default_factory=list)
    aka_flag: bool = True
    bakaze: str = "E"
    kyoku: int = 1
    honba: int = 0
    kyotaku: int = 0
    oya: int = 0
    scores: list[int] = field(default_factory=lambda: [25000, 25000, 25000, 25000])
    dora_markers: list[str] = field(default_factory=list)
    players: list[PlayerState] = field(default_factory=lambda: [PlayerState() for _ in range(4)])
    wall_left: int = 70
    any_call: bool = False
    last_discard_actor: int | None = None
    last_discard_tile: str | None = None
    kyoku_index: int = 0

    def start_game(self, event: dict, game_id: str) -> None:
        self.game_id = game_id
        self.names = event.get("names", [])
        self.aka_flag = bool(event.get("aka_flag", True))

    def start_kyoku(self, event: dict) -> None:
        self.kyoku_index += 1
        self.bakaze = event["bakaze"]
        self.kyoku = int(event["kyoku"])
        self.honba = int(event["honba"])
        self.kyotaku = int(event["kyotaku"])
        self.oya = int(event["oya"])
        self.scores = [int(s) for s in event["scores"]]
        self.dora_markers = [normalize_tile(event["dora_marker"])]
        self.wall_left = 70
        self.any_call = False
        self.last_discard_actor = None
        self.last_discard_tile = None
        self.players = [PlayerState(hand=sort_tiles(normalize_tiles(h))) for h in event["tehais"]]

    def add_dora(self, tile: str) -> None:
        self.dora_markers.append(normalize_tile(tile))

    def on_tsumo(self, actor: int, tile: str) -> None:
        self.players[actor].draw = normalize_tile(tile)
        self.wall_left = max(0, self.wall_left - 1)

    def on_reach(self, actor: int) -> None:
        self.players[actor].pending_reach = True

    def on_reach_accepted(self, actor: int) -> None:
        player = self.players[actor]
        player.pending_reach = False
        player.riichi = True
        player.ippatsu = True
        self.scores[actor] -= 1000
        self.kyotaku += 1

    def clear_ippatsu_on_call(self) -> None:
        for player in self.players:
            player.ippatsu = False

    def on_dahai(self, actor: int, tile: str, tsumogiri: bool) -> str:
        """Apply a discard and append the corresponding public river entry."""
        player = self.players[actor]
        tile = normalize_tile(tile)
        source = "TSUMOGIRI" if tsumogiri else "HAND"
        removed = self._remove_discarded_tile(player, tile, tsumogiri)
        river_tile = normalize_tile(removed)
        riichi_decl = player.pending_reach
        post_riichi = player.riichi and not riichi_decl
        player.river.append(
            RiverDiscard(
                tile=river_tile,
                source=source,
                riichi_decl=riichi_decl,
                post_riichi=post_riichi,
            )
        )
        if player.riichi and post_riichi:
            player.ippatsu = False
        self.last_discard_actor = actor
        self.last_discard_tile = river_tile
        return river_tile

    def _remove_discarded_tile(self, player: PlayerState, tile: str, tsumogiri: bool) -> str:
        if self._is_current_draw(player, tile):
            if tsumogiri or tile not in player.hand:
                return self._discard_current_draw(player)
        return self._discard_from_closed_hand(player, tile, keep_current_draw=not tsumogiri)

    def _is_current_draw(self, player: PlayerState, tile: str) -> bool:
        return player.draw is not None and normalize_tile(player.draw) == tile

    def _discard_current_draw(self, player: PlayerState) -> str:
        assert player.draw is not None
        removed = player.draw
        player.draw = None
        return removed

    def _discard_from_closed_hand(self, player: PlayerState, tile: str, keep_current_draw: bool) -> str:
        removed = remove_one(player.hand, tile)
        if keep_current_draw and player.draw is not None:
            player.hand.append(player.draw)
            player.hand = sort_tiles(player.hand)
            player.draw = None
        return removed

    def mark_called(self, caller: int, target: int) -> None:
        if self.players[target].river:
            self.players[target].river[-1].called_by = caller

    def on_chi(self, actor: int, target: int, tile: str, consumed: list[str]) -> None:
        tile = normalize_tile(tile)
        removed = remove_many(self.players[actor].hand, normalize_tiles(consumed))
        meld_tiles = sort_tiles(removed + [tile])
        self.players[actor].melds.append(Meld("CHI", meld_tiles, source=target, opened=True))
        self.mark_called(actor, target)
        self.any_call = True
        self.clear_ippatsu_on_call()

    def on_pon(self, actor: int, target: int, tile: str, consumed: list[str]) -> None:
        tile = normalize_tile(tile)
        removed = remove_many(self.players[actor].hand, normalize_tiles(consumed))
        meld_tiles = sort_tiles(removed + [tile])
        self.players[actor].melds.append(Meld("PON", meld_tiles, source=target, opened=True))
        self.mark_called(actor, target)
        self.any_call = True
        self.clear_ippatsu_on_call()

    def on_daiminkan(self, actor: int, target: int, tile: str, consumed: list[str]) -> None:
        tile = normalize_tile(tile)
        removed = remove_many(self.players[actor].hand, normalize_tiles(consumed))
        meld_tiles = sort_tiles(removed + [tile])
        self.players[actor].melds.append(Meld("MINKAN", meld_tiles, source=target, opened=True))
        self.mark_called(actor, target)
        self.any_call = True
        self.clear_ippatsu_on_call()

    def on_ankan(self, actor: int, consumed: list[str]) -> None:
        player = self.players[actor]
        removed = remove_many(player.decision_tiles, normalize_tiles(consumed))
        # remove_many operated on a temporary list, so remove from actual zones.
        for tile in removed:
            if player.draw is not None and normalize_tile(player.draw) == tile:
                player.draw = None
            else:
                remove_one(player.hand, tile)
        if player.draw is not None:
            player.hand.append(player.draw)
            player.hand = sort_tiles(player.hand)
            player.draw = None
        player.melds.append(Meld("ANKAN", sort_tiles(removed), source=None, opened=False))

    def on_kakan(self, actor: int, tile: str) -> None:
        player = self.players[actor]
        tile = normalize_tile(tile)
        removed = tile
        if player.draw is not None and normalize_tile(player.draw) == tile:
            player.draw = None
        else:
            removed = remove_one(player.hand, tile)
        if player.draw is not None:
            player.hand.append(player.draw)
            player.hand = sort_tiles(player.hand)
            player.draw = None
        for meld in player.melds:
            if meld.kind == "PON" and len(meld.tiles) == 3 and all(tile_base(t) == tile_base(tile) for t in meld.tiles):
                meld.kind = "KAKAN"
                meld.tiles = sort_tiles(meld.tiles + [removed])
                return
        player.melds.append(Meld("KAKAN", [removed], source=None, opened=True))

    def round_name(self) -> str:
        return f"{self.bakaze}{self.kyoku}"

    def turn_number(self) -> int:
        return max(1, (70 - self.wall_left) // 4 + 1)

    def visible_meld_line(self, owner: int, perspective: int) -> str:
        label = rel_name(owner, perspective)
        melds = self.players[owner].melds
        if not melds:
            return f"{label} EMPTY"
        parts = [label]
        for meld in melds:
            parts.extend([meld.kind, *sort_tiles(meld.tiles)])
            if meld.source is not None:
                parts.append(from_rel(meld.source, perspective))
        return " ".join(parts)

    def river_line(self, owner: int, perspective: int) -> str:
        label = rel_name(owner, perspective)
        river = self.players[owner].river
        if not river:
            return f"{label} EMPTY"
        parts = [label]
        for discard in river:
            parts.extend([discard.tile, discard.source])
            if discard.riichi_decl:
                parts.append("RIICHI_DECL")
            if discard.post_riichi:
                parts.append("POST_RIICHI")
            if discard.called_by is not None:
                parts.append(called_by_rel(discard.called_by, perspective))
        return " ".join(parts)

    def wind_lines(self, perspective: int) -> list[str]:
        return [f"{rel_name(p, perspective)} {seat_wind(p, self.oya)}" for p in player_order(perspective)]

    def score_lines(self, perspective: int) -> list[str]:
        return [f"{rel_name(p, perspective)} {self.scores[p]}" for p in player_order(perspective)]

    def player_state_lines(self, perspective: int) -> list[str]:
        lines: list[str] = []
        for p in player_order(perspective):
            player = self.players[p]
            lines.append(
                " ".join(
                    [
                        rel_name(p, perspective),
                        "MENZEN" if player.is_menzen else "OPEN",
                        "RIICHI" if player.riichi else "NO_RIICHI",
                        "IPPATSU" if player.ippatsu else "NO_IPPATSU",
                    ]
                )
            )
        return lines
