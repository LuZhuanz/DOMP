from __future__ import annotations

from collections import Counter
from itertools import combinations

HONOR_MAP = {
    "E": "1z",
    "S": "2z",
    "W": "3z",
    "N": "4z",
    "P": "5z",
    "F": "6z",
    "C": "7z",
}

WIND_NAMES = ["E", "S", "W", "N"]
REL_NAMES = ["SELF", "SHIMO", "TOIMEN", "KAMI"]


def normalize_tile(tile: str) -> str:
    return HONOR_MAP.get(tile, tile)


def normalize_tiles(tiles: list[str]) -> list[str]:
    return [normalize_tile(t) for t in tiles]


def tile_base(tile: str) -> str:
    tile = normalize_tile(tile)
    if tile in {"5mr", "5pr", "5sr"}:
        return tile[:2]
    return tile


def tile_index(tile: str) -> int:
    tile = tile_base(tile)
    n = int(tile[0])
    suit = tile[1]
    if suit == "m":
        return n - 1
    if suit == "p":
        return 9 + n - 1
    if suit == "s":
        return 18 + n - 1
    if suit == "z":
        return 27 + n - 1
    raise ValueError(f"unknown tile: {tile}")


def index_tile(index: int) -> str:
    if 0 <= index < 9:
        return f"{index + 1}m"
    if 9 <= index < 18:
        return f"{index - 8}p"
    if 18 <= index < 27:
        return f"{index - 17}s"
    if 27 <= index < 34:
        return f"{index - 26}z"
    raise ValueError(f"unknown tile index: {index}")


def tile_sort_key(tile: str) -> tuple[int, int, str]:
    tile = normalize_tile(tile)
    base = tile_base(tile)
    red_rank = 1 if tile.endswith("r") else 0
    return (tile_index(base), red_rank, tile)


def sort_tiles(tiles: list[str]) -> list[str]:
    return sorted((normalize_tile(t) for t in tiles), key=tile_sort_key)


def counts34(tiles: list[str]) -> list[int]:
    counts = [0] * 34
    for tile in tiles:
        counts[tile_index(tile)] += 1
    return counts


def base_counter(tiles: list[str]) -> Counter[str]:
    return Counter(tile_base(t) for t in tiles)


def remove_one(tiles: list[str], tile: str) -> str:
    """Remove one matching physical tile from tiles and return the removed tile.

    Exact red/plain identity is preferred. If the requested tile is a base tile and
    only a red five is present, the red tile is removed because it represents the
    same physical base for rule purposes.
    """
    tile = normalize_tile(tile)
    if tile in tiles:
        tiles.remove(tile)
        return tile
    base = tile_base(tile)
    for candidate in list(tiles):
        if tile_base(candidate) == base:
            tiles.remove(candidate)
            return candidate
    raise ValueError(f"tile {tile} not in hand {tiles}")


def remove_many(tiles: list[str], consumed: list[str]) -> list[str]:
    removed: list[str] = []
    for tile in consumed:
        removed.append(remove_one(tiles, normalize_tile(tile)))
    return removed


def physical_choices(hand: list[str], base: str, need: int) -> list[tuple[str, ...]]:
    matches = [t for t in hand if tile_base(t) == tile_base(base)]
    unique: set[tuple[str, ...]] = set()
    for combo in combinations(range(len(matches)), need):
        unique.add(tuple(sort_tiles([matches[i] for i in combo])))
    return sorted(unique, key=lambda c: [tile_sort_key(t) for t in c])


def all_tile_bases() -> list[str]:
    return [index_tile(i) for i in range(34)]


def rel_name(target: int, perspective: int) -> str:
    return REL_NAMES[(target - perspective) % 4]


def from_rel(target: int, perspective: int) -> str:
    return f"FROM_{rel_name(target, perspective)}"


def called_by_rel(target: int, perspective: int) -> str:
    return f"CALLED_BY_{rel_name(target, perspective)}"


def player_order(perspective: int) -> list[int]:
    return [(perspective + offset) % 4 for offset in range(4)]


def seat_wind(player: int, dealer: int) -> str:
    return WIND_NAMES[(player - dealer) % 4]
