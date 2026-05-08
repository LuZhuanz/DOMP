from __future__ import annotations

from functools import lru_cache

from .tiles import all_tile_bases, counts34, tile_index

TERMINAL_HONOR_INDICES = set([0, 8, 9, 17, 18, 26] + list(range(27, 34)))


def _can_make_sets(counts: tuple[int, ...]) -> bool:
    try:
        first = next(i for i, c in enumerate(counts) if c)
    except StopIteration:
        return True

    work = list(counts)
    if work[first] >= 3:
        work[first] -= 3
        if _can_make_sets_cached(tuple(work)):
            return True
        work[first] += 3

    suit_start = (first // 9) * 9
    if first < 27 and first + 2 < suit_start + 9:
        if work[first + 1] and work[first + 2]:
            work[first] -= 1
            work[first + 1] -= 1
            work[first + 2] -= 1
            if _can_make_sets_cached(tuple(work)):
                return True
    return False


@lru_cache(maxsize=200_000)
def _can_make_sets_cached(counts: tuple[int, ...]) -> bool:
    return _can_make_sets(counts)


def is_standard_win(counts: list[int], open_melds: int = 0) -> bool:
    return _is_standard_win_cached(tuple(counts), open_melds)


@lru_cache(maxsize=100_000)
def _is_standard_win_cached(counts_tuple: tuple[int, ...], open_melds: int = 0) -> bool:
    counts = list(counts_tuple)
    needed_sets = 4 - open_melds
    if needed_sets < 0:
        return False
    if sum(counts) != needed_sets * 3 + 2:
        return False
    for i, c in enumerate(counts):
        if c >= 2:
            work = counts[:]
            work[i] -= 2
            if _can_make_sets_cached(tuple(work)):
                return True
    return False


def is_chiitoi(counts: list[int], open_melds: int = 0) -> bool:
    return open_melds == 0 and sum(counts) == 14 and sum(1 for c in counts if c == 2) == 7


def is_kokushi(counts: list[int], open_melds: int = 0) -> bool:
    if open_melds != 0 or sum(counts) != 14:
        return False
    return all(counts[i] >= 1 for i in TERMINAL_HONOR_INDICES) and any(
        counts[i] >= 2 for i in TERMINAL_HONOR_INDICES
    )


def is_complete_hand(tiles: list[str], open_melds: int = 0) -> bool:
    return _is_complete_counts_cached(tuple(counts34(tiles)), open_melds)


def is_tenpai(tiles: list[str], open_melds: int = 0) -> bool:
    return _is_tenpai_counts_cached(tuple(counts34(tiles)), open_melds)


@lru_cache(maxsize=100_000)
def _is_complete_counts_cached(counts_tuple: tuple[int, ...], open_melds: int = 0) -> bool:
    counts = list(counts_tuple)
    return (
        is_standard_win(counts, open_melds)
        or is_chiitoi(counts, open_melds)
        or is_kokushi(counts, open_melds)
    )


@lru_cache(maxsize=100_000)
def _is_tenpai_counts_cached(counts_tuple: tuple[int, ...], open_melds: int = 0) -> bool:
    counts = list(counts_tuple)
    for base in all_tile_bases():
        idx = tile_index(base)
        if counts[idx] >= 4:
            continue
        counts[idx] += 1
        if _is_complete_counts_cached(tuple(counts), open_melds):
            counts[idx] -= 1
            return True
        counts[idx] -= 1
    return False


def terminal_honor_count(tiles: list[str]) -> int:
    return len({tile_index(t) for t in tiles if tile_index(t) in TERMINAL_HONOR_INDICES})


def clear_agari_caches() -> None:
    _can_make_sets_cached.cache_clear()
    _is_standard_win_cached.cache_clear()
    _is_complete_counts_cached.cache_clear()
    _is_tenpai_counts_cached.cache_clear()
