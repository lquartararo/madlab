"""
Bracket utility functions: fold/unfold index permutations, bracket I/O, and visualization.

The NCAA bracket has 64 teams and 63 games across 6 rounds.
Teams are stored in seed order; internally simulations use "matchup order"
(achieved via fold/unfold permutations).
"""

from __future__ import annotations

import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    plt = None
    _MPL_AVAILABLE = False
import numpy as np


# ---------------------------------------------------------------------------
# Core fold/unfold index functions
# ---------------------------------------------------------------------------

def _fold_indices(length: int, block_size: int) -> np.ndarray:
    """
    Return permutation indices p such that x[p] == fold(x, block_size).

    fold reorders: first block, last block, second block, second-to-last, ...
    """
    n_blocks = length // block_size
    if length % block_size != 0:
        raise ValueError(f"length={length} not divisible by block_size={block_size}")
    if n_blocks % 2 != 0:
        raise ValueError(f"n_blocks={n_blocks} must be even")

    order = np.empty(n_blocks, dtype=np.intp)
    lo, hi, out = 0, n_blocks - 1, 0
    while lo <= hi:
        order[out] = lo
        order[out + 1] = hi
        lo += 1; hi -= 1; out += 2
    return (order[:, None] * block_size + np.arange(block_size)).ravel()


def _unfold_indices(length: int, block_size: int) -> np.ndarray:
    """
    Return permutation indices p such that x[p] == unfold(x, block_size).

    unfold is the inverse of fold: odd-indexed blocks first, then even-indexed in reverse.
    Matches R: c(seq(1, n-1, 2), seq(n, 2, -2)).
    """
    n_blocks = length // block_size
    if length % block_size != 0:
        raise ValueError(f"length={length} not divisible by block_size={block_size}")
    if n_blocks % 2 != 0:
        raise ValueError(f"n_blocks={n_blocks} must be even")

    # R: c(seq(1, n-1, 2), seq(n, 2, -2)) -- 1-based
    evens_1based = np.arange(1, n_blocks, 2)    # [1, 3, 5, ...]
    odds_1based = np.arange(n_blocks, 1, -2)    # [n, n-2, ...]
    order = np.concatenate([evens_1based, odds_1based]) - 1  # to 0-based
    return (order[:, None] * block_size + np.arange(block_size)).ravel()


def _compose(length: int, fn, block_sizes: list[int]) -> np.ndarray:
    """
    Compose a sequence of permutations on an array of `length`.

    For f1, f2, f3 applied sequentially to array x:
        y = x[p1][p2][p3] = x[p1[p2[p3]]]
    We compute: start with p1, then repeatedly apply: idx = idx[pi]
    """
    idx = fn(length, block_sizes[0])
    for bs in block_sizes[1:]:
        idx = idx[fn(length, bs)]
    return idx


# ---------------------------------------------------------------------------
# Precomputed permutation constants
# ---------------------------------------------------------------------------

# FOLD_TO_MATCHUP: seed-order -> matchup-order for 64-team bracket
# Applies fold(1), fold(2), fold(4), fold(8), fold(16), fold(32) sequentially
FOLD_TO_MATCHUP: np.ndarray = _compose(64, _fold_indices, [1, 2, 4, 8, 16, 32])

# UNTANGLE[r]: converts matchup-order round-r outcomes back to seed order
# Mirrors the R untangling.indices
UNTANGLE: dict[int, np.ndarray] = {
    1: _compose(32, _unfold_indices, [16, 8, 4, 2, 1]),
    2: _compose(16, _unfold_indices, [8, 4, 2, 1]),
    3: _compose(8, _unfold_indices, [4, 2, 1]),
    4: _compose(4, _unfold_indices, [2, 1]),
    5: _compose(2, _unfold_indices, [1]),
    6: np.array([0], dtype=np.intp),
}

# Round label for each of the 63 rows in a filled bracket
ROUND_OF_ROW: np.ndarray = np.array(
    [1] * 32 + [2] * 16 + [3] * 8 + [4] * 4 + [5, 5] + [6],
    dtype=np.int8,
)

# Reorder from "pool/ESPN reading order" to internal seed order
# (from R input.bracket.filled; 1-based in R, converted to 0-based here)
_INPUT_REORDER_1BASED = [
    1, 17, 25, 9, 13, 29, 21, 5, 7, 23, 31, 15, 11, 27, 19, 3, 4,
    20, 28, 12, 16, 32, 24, 8, 6, 22, 30, 14, 10, 26, 18, 2, 33, 41, 45, 37,
    39, 47, 43, 35, 36, 44, 48, 40, 38, 46, 42, 34, 49, 53, 55, 51, 52, 56,
    54, 50, 57, 59, 60, 58, 61, 62, 63,
]
INPUT_REORDER: np.ndarray = np.array(_INPUT_REORDER_1BASED, dtype=np.intp) - 1


# ---------------------------------------------------------------------------
# Data loading / saving helpers
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"


def load_bracket(league: str, year: int) -> list[str]:
    """Load a 64-team bracket vector from bundled data."""
    path = DATA_DIR / f"bracket.{league}.{year}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No bracket data for {league}/{year}. "
            f"Run prep_data(year={year}, league='{league}') first."
        )
    with open(path) as f:
        return json.load(f)


def load_teams(league: str) -> "pl.DataFrame":
    """Load the teams DataFrame (id, name, name_pop, name_538) for a league."""
    import polars as pl
    return pl.read_parquet(DATA_DIR / f"teams.{league}.parquet")


def load_games(league: str, year: int) -> "pl.DataFrame":
    """Load game results DataFrame for a league/year."""
    import polars as pl
    path = DATA_DIR / f"games.{league}.{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No game data for {league}/{year}. "
            f"Run prep_data(year={year}, league='{league}') first."
        )
    return pl.read_parquet(path)


def load_pred(source: str, league: str, year: int) -> "pl.DataFrame":
    """Load prediction DataFrame for a given source/league/year."""
    import polars as pl
    path = DATA_DIR / f"pred.{source}.{league}.{year}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No prediction data for source={source}, {league}/{year}. "
            f"Available sources: pop, 538, kenpom."
        )
    return pl.read_parquet(path)


def save_bracket(bracket: list[str], league: str, year: int) -> None:
    """Persist a bracket list to the data directory."""
    path = DATA_DIR / f"bracket.{league}.{year}.json"
    with open(path, "w") as f:
        json.dump(bracket, f)


def save_games(df: "pl.DataFrame", league: str, year: int) -> None:
    df.write_parquet(DATA_DIR / f"games.{league}.{year}.parquet")


def save_pred(df: "pl.DataFrame", source: str, league: str, year: int) -> None:
    df.write_parquet(DATA_DIR / f"pred.{source}.{league}.{year}.parquet")


# ---------------------------------------------------------------------------
# Bracket utilities
# ---------------------------------------------------------------------------

def input_bracket_filled(picks: list[str]) -> list[str]:
    """
    Convert a bracket in pool/ESPN reading order (63 picks) to internal seed order.
    See R input.bracket.filled for the derivation of the reorder vector.
    """
    if len(picks) != 63:
        raise ValueError("picks must have length 63")
    return np.array(picks)[INPUT_REORDER].tolist()


# ---------------------------------------------------------------------------
# Display-order helpers (used by the web server for JSON API)
# ---------------------------------------------------------------------------

def bracket_display_slots(bracket_empty: list[str], league: str) -> list[dict]:
    """Return 64 initial bracket slots in display (matchup) order with name/seed."""
    import polars as pl
    teams_df = load_teams(league)
    id_to_name: dict[str, str] = dict(
        zip(teams_df["id"].cast(str).to_list(), teams_df["name"].to_list())
    )

    def resolve(team_id: str) -> str:
        if not team_id:
            return "TBD"
        return "/".join(id_to_name.get(p, p) for p in team_id.split("/"))

    seeds = list(range(1, 17)) * 4
    ids_m   = np.array(bracket_empty)[FOLD_TO_MATCHUP].tolist()
    seeds_m = np.array(seeds, dtype=int)[FOLD_TO_MATCHUP].tolist()

    return [
        {"id": tid, "name": resolve(tid), "seed": int(s)}
        for s, tid in zip(seeds_m, ids_m)
    ]


def picks_display_order(picks: list[str], bracket_empty: list[str], league: str) -> list[dict]:
    """Return 63 picks in display (matchup) order with name/seed metadata."""
    import polars as pl
    teams_df = load_teams(league)
    id_to_name: dict[str, str] = dict(
        zip(teams_df["id"].cast(str).to_list(), teams_df["name"].to_list())
    )

    def resolve(team_id: str) -> str:
        return "/".join(id_to_name.get(p, p) for p in str(team_id).split("/"))

    seeds = list(range(1, 17)) * 4
    seed_map: dict[str, int] = {}
    for i, t in enumerate(bracket_empty):
        for part in t.split("/"):
            seed_map[part] = seeds[i]

    fold_seqs = {1: [1,2,4,8,16], 2: [1,2,4,8], 3: [1,2,4], 4: [1,2], 5: [1]}
    picks_arr = np.array(picks, dtype=object)
    for r, block_sizes in fold_seqs.items():
        mask = np.where(ROUND_OF_ROW == r)[0]
        idx  = _compose(len(mask), _fold_indices, block_sizes)
        picks_arr[mask] = picks_arr[mask][idx]

    return [
        {
            "id": str(tid),
            "name": resolve(str(tid)),
            "seed": seed_map.get(str(tid), seed_map.get(str(tid).split("/")[0], 0)),
        }
        for tid in picks_arr
    ]


# ---------------------------------------------------------------------------
# Bracket visualization
# ---------------------------------------------------------------------------

def draw_bracket(
    bracket_empty: list[str],
    bracket_filled: list[str] | None = None,
    league: str = "men",
    text_size: float = 6.5,
    figsize: tuple[float, float] = (24, 14),
    dpi: int = 100,
):
    """
    Draw the NCAA tournament bracket using matplotlib.

    Parameters
    ----------
    bracket_empty  : list of 64 team IDs in seed order
    bracket_filled : optional list of 63 game outcome team IDs
    league         : "men" or "women"
    text_size      : font size for team labels
    figsize        : figure size in inches

    Returns
    -------
    matplotlib Figure
    """
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib is required for draw_bracket(). Install it with: pip install matplotlib")
    teams_df = load_teams(league)
    id_to_name: dict[str, str] = dict(
        zip(teams_df["id"].cast(str).to_list(), teams_df["name"].to_list())
    )

    def id_to_label(team_id: str | None) -> str:
        if not team_id:
            return "TBD"
        parts = team_id.split("/")
        return "/".join(id_to_name.get(p, p) for p in parts)

    # Seeds 1..16 repeated 4 times (groups of 4 for each quadrant)
    seeds = list(range(1, 17)) * 4

    # Convert bracket_empty to display names with seed prefix, then fold to matchup order
    empty_labels = [f"{seeds[i]} {id_to_label(t)}" for i, t in enumerate(bracket_empty)]
    empty_matchup = np.array(empty_labels)[FOLD_TO_MATCHUP].tolist()

    # Build seed map for team IDs
    seed_map: dict[str, int] = {}
    for i, t in enumerate(bracket_empty):
        for part in t.split("/"):
            seed_map[part] = seeds[i]

    filled_matchup: list[str] | None = None
    if bracket_filled is not None:
        if len(bracket_filled) != 63:
            raise ValueError("bracket_filled must have length 63")
        filled_labels = [
            f"{seed_map.get(t, '?')} {id_to_label(t)}" for t in bracket_filled
        ]
        # Convert each round from seed order to display (matchup) order using fold.
        # Mirrors R: bracket.filled[round==r] %>% fold(1) %>% fold(2) %>% ...
        # Round 1 (32): fold(1,2,4,8,16); Round 2 (16): fold(1,2,4,8); etc.
        fold_seqs = {1: [1,2,4,8,16], 2: [1,2,4,8], 3: [1,2,4], 4: [1,2], 5: [1]}
        filled_arr = np.array(filled_labels)
        for r, block_sizes in fold_seqs.items():
            mask = np.where(ROUND_OF_ROW == r)[0]
            n = len(mask)
            idx = _compose(n, _fold_indices, block_sizes)
            filled_arr[mask] = filled_arr[mask][idx]
        filled_matchup = filled_arr.tolist()

    # x, y coordinate arrays matching R draw.bracket exactly.
    # Layout: left-R1 (32), right-R1 (32), left-R2 (16), right-R2 (16), ...
    # First 64 positions = initial team slots; positions 64-126 = game outcome slots.
    def ys_for_n(n: int) -> np.ndarray:
        return np.linspace(1 - 1 / (2 * n), 1 / (2 * n), n)

    x = np.array([
        *[-6.0] * 32, *[6.0] * 32,       # R1: left, right  (initial teams 0-63)
        *[-5.0] * 16, *[5.0] * 16,       # R2: left, right
        *[-4.0] * 8,  *[4.0] * 8,        # R3
        *[-3.0] * 4,  *[3.0] * 4,        # R4
        -2.0, -2.0, 2.0, 2.0,            # R5
        -1.0, 1.0, 0.0,                  # R6 (semis) + champion
    ])
    y = np.concatenate([
        np.tile(ys_for_n(32), 2),        # R1 left + right
        np.tile(ys_for_n(16), 2),        # R2
        np.tile(ys_for_n(8),  2),        # R3
        np.tile(ys_for_n(4),  2),        # R4
        [3/4, 1/4, 3/4, 1/4],           # R5
        [3/5, 2/5, 1/2],                 # R6 + champion
    ])

    # Design-system colours (hex approximations of the OKLCH tokens)
    C_LINE   = "#ddd8d0"   # --rule:   oklch(90% 0.010 80)  warm gray
    C_INK    = "#1e1e2e"   # --ink:    oklch(16% 0.018 265) near-black, cool tint
    C_MUTED  = "#7a7a8e"   # --ink-3:  oklch(64% 0.010 265) secondary text
    C_ACCENT = "#d45f00"   # --accent: oklch(62% 0.22  38)  orange

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_visible(False)
    ax.set_xlim(-7.5, 7.5)
    ax.set_ylim(-0.01, 1.01)
    ax.axis("off")

    # Horizontal slot lines
    for xi, yi in zip(x, y):
        ax.plot([xi - 0.5, xi + 0.5], [yi, yi], color=C_LINE, lw=0.8, solid_capstyle="round")

    # Vertical connecting lines
    block_sizes = [32, 32, 16, 16, 8, 8, 4, 4, 2, 2, 1, 1, 1]
    start = 0
    for bsize in block_sizes:
        end = start + bsize
        for i in range(start, end - 1, 2):
            vx = x[i] + (1 if x[i] < 0 else -1) * 0.5
            ax.plot([vx, vx], [y[i], y[i + 1]], color=C_LINE, lw=0.8, solid_capstyle="round")
        start = end

    # Empty bracket labels: seed in muted, name in ink
    font_kw = dict(va="bottom", ha="left", clip_on=True,
                   fontsize=text_size, fontfamily="DejaVu Sans")
    for i, label in enumerate(empty_matchup):
        seed, _, name = label.partition(" ")
        ax.text(x[i] - 0.46, y[i] + 0.005, seed + " ",
                color=C_MUTED, **font_kw)
        ax.text(x[i] - 0.46 + 0.18, y[i] + 0.005, name,
                color=C_INK, **font_kw)

    # Filled bracket labels: accent color, slightly bolder
    if filled_matchup is not None:
        n_empty = len(empty_matchup)
        font_filled = dict(va="bottom", ha="left", clip_on=True,
                           fontsize=text_size, fontfamily="DejaVu Sans",
                           fontweight="semibold")
        for i, label in enumerate(filled_matchup):
            seed, _, name = label.partition(" ")
            xi, yi = x[n_empty + i], y[n_empty + i]
            ax.text(xi - 0.46, yi + 0.005, seed + " ",
                    color=C_MUTED, **font_filled)
            ax.text(xi - 0.46 + 0.18, yi + 0.005, name,
                    color=C_ACCENT, **font_filled)

    fig.tight_layout(pad=0.5)
    return fig
