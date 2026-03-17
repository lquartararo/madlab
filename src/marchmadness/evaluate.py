"""
Bracket scoring and evaluation.

score_bracket() is fully vectorized with NumPy (no Python loop over simulations).
test_bracket() evaluates a bracket against simulated pools and outcomes.
"""

from __future__ import annotations

import numpy as np

from .bracket import ROUND_OF_ROW
from .simulate import sim_bracket, CURRENT_YEAR


def score_bracket(
    bracket_empty: list[str],
    bracket_picks: np.ndarray,
    bracket_outcome: np.ndarray,
    bonus_round: list[float] | np.ndarray | None = None,
    bonus_seed: list[float] | np.ndarray | None = None,
    bonus_combine: str = "add",
) -> np.ndarray:
    """
    Score one or more brackets against one or more outcomes.

    Parameters
    ----------
    bracket_empty   : list of 64 team IDs in seed order
    bracket_picks   : (63,) or (63, n_brackets) array of team ID strings
    bracket_outcome : (63,) or (63, n_sims) array of team ID strings
    bonus_round     : length-6 vector of points per correct pick per round
    bonus_seed      : length-16 vector of bonus points by winner's seed
    bonus_combine   : "add" or "multiply"

    Returns
    -------
    scores : np.ndarray of shape (n_brackets, n_sims) or scalar depending on inputs
    """
    if bonus_round is None:
        bonus_round = np.array([1, 2, 4, 8, 16, 32], dtype=float)
    if bonus_seed is None:
        bonus_seed = np.zeros(16, dtype=float)

    bonus_round = np.asarray(bonus_round, dtype=float)
    bonus_seed = np.asarray(bonus_seed, dtype=float)

    if len(bonus_round) != 6:
        raise ValueError("bonus_round must have length 6")
    if len(bonus_seed) != 16:
        raise ValueError("bonus_seed must have length 16")

    # Normalize to (63, n) column-major layout.
    # np.atleast_2d adds a leading dim, giving (1, 63) for a 1D input — wrong.
    # We always want (63, 1) for a single bracket/outcome.
    picks = np.asarray(bracket_picks)
    picks_1d = picks.ndim == 1
    if picks.ndim == 1:
        picks = picks[:, None]          # (63,) → (63, 1)

    outcome = np.asarray(bracket_outcome)
    outcome_1d = outcome.ndim == 1
    if outcome.ndim == 1:
        outcome = outcome[:, None]      # (63,) → (63, 1)

    if picks.shape[0] != 63:
        raise ValueError(f"picks first dimension must be 63, got {picks.shape[0]}")
    if outcome.shape[0] != 63:
        raise ValueError(f"outcome first dimension must be 63, got {outcome.shape[0]}")

    # Per-row round bonus (shape 63)
    row_round_bonus = bonus_round[ROUND_OF_ROW - 1]  # (63,)

    # Seed of each team in the empty bracket (1..16 repeated 4x)
    seeds_64 = np.tile(np.arange(1, 17), 4)  # (64,)
    seed_map: dict[str, int] = {t: int(seeds_64[i]) for i, t in enumerate(bracket_empty)}

    # Seed bonus for picks: shape (63, n_brackets)
    if picks.ndim == 1:
        picks = picks[:, None]
    picks_seed_bonus = np.vectorize(lambda t: bonus_seed[seed_map.get(t, 1) - 1])(picks)  # (63, n_brackets)

    # Points per correct pick: (63, n_brackets)
    if bonus_combine == "add":
        points = row_round_bonus[:, None] + picks_seed_bonus  # broadcast over brackets
    elif bonus_combine == "multiply":
        points = row_round_bonus[:, None] * picks_seed_bonus
    else:
        raise ValueError(f"bonus_combine must be 'add' or 'multiply', got {bonus_combine!r}")

    # Score: sum of points where pick == outcome, fully vectorized
    # picks:   (63, n_brackets)
    # outcome: (63, n_sims)
    # result:  (n_brackets, n_sims)
    n_brackets = picks.shape[1]
    n_sims = outcome.shape[1]

    # Expand for broadcasting: picks (63, n_brackets, 1), outcome (63, 1, n_sims)
    correct = picks[:, :, None] == outcome[:, None, :]  # (63, n_brackets, n_sims)
    # points (63, n_brackets) -> (63, n_brackets, 1)
    scores = (correct * points[:, :, None]).sum(axis=0)  # (n_brackets, n_sims)

    if picks_1d and outcome_1d:
        return float(scores[0, 0])
    if picks_1d:
        return scores[0, :]  # (n_sims,)
    if outcome_1d:
        return scores[:, 0]  # (n_brackets,)
    return scores


def test_bracket(
    bracket_empty: list[str],
    bracket_picks: list[str],
    prob_matrix: np.ndarray | None = None,
    prob_source: str = "pop",
    pool_source: str = "pop",
    league: str = "men",
    year: int = CURRENT_YEAR,
    pool_bias: list[str] | None = None,
    pool_size: int = 30,
    num_sims: int = 1000,
    bonus_round: list[float] | None = None,
    bonus_seed: list[float] | None = None,
    bonus_combine: str = "add",
    print_progress: bool = True,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Test a bracket by simulating many pools.

    Parameters
    ----------
    bracket_empty : list of 64 team IDs
    bracket_picks : list of 63 team IDs (your bracket)
    (all other parameters match find_bracket / sim_bracket semantics)

    Returns
    -------
    dict with keys:
      "score"      : np.ndarray of shape (num_sims,) - your score per sim
      "percentile" : np.ndarray of shape (num_sims,) - your percentile per sim
      "win"        : np.ndarray of shape (num_sims,) bool - did you win per sim
    """
    if len(bracket_empty) != 64:
        raise ValueError("bracket_empty must have length 64")
    if len(bracket_picks) != 63:
        raise ValueError("bracket_picks must have length 63")
    if pool_size < 2:
        raise ValueError("pool_size must be at least 2")

    if rng is None:
        rng = np.random.default_rng()

    if bonus_round is None:
        bonus_round = [1, 2, 4, 8, 16, 32]
    if bonus_seed is None:
        bonus_seed = [0] * 16

    if print_progress:
        print(f"Testing your bracket ...")
        print(f"  Simulating {num_sims} pools of size {pool_size} ...")

    pool = sim_bracket(
        bracket_empty, prob_matrix=None, prob_source=pool_source,
        league=league, year=year, home_teams=pool_bias,
        num_reps=num_sims * pool_size, rng=rng,
    )
    outcome = sim_bracket(
        bracket_empty, prob_matrix=prob_matrix, prob_source=prob_source,
        league=league, year=year, num_reps=num_sims, rng=rng,
    )

    if print_progress:
        print(f"  Scoring {num_sims * (1 + pool_size)} brackets ...")

    picks_arr = np.array(bracket_picks)[:, None]  # (63, 1)

    # Score your bracket: (1, num_sims)
    my_score = score_bracket(
        bracket_empty, picks_arr, outcome,
        bonus_round=bonus_round, bonus_seed=bonus_seed, bonus_combine=bonus_combine,
    )  # (1, num_sims)

    # Score pool brackets: pool is (63, num_sims*pool_size)
    # We need to score each pool sim's pool_size brackets against the corresponding outcome
    pool_scores = np.empty((pool_size, num_sims))
    for i in range(num_sims):
        pool_slice = pool[:, i * pool_size:(i + 1) * pool_size]  # (63, pool_size)
        s = score_bracket(
            bracket_empty, pool_slice, outcome[:, i],
            bonus_round=bonus_round, bonus_seed=bonus_seed, bonus_combine=bonus_combine,
        )  # (pool_size, 1) or (pool_size,)
        pool_scores[:, i] = np.squeeze(s)

    # Combine: (pool_size+1, num_sims)
    all_scores = np.vstack([my_score, pool_scores])  # (pool_size+1, num_sims)

    # Rank within each simulation (higher = better)
    my_rank = (all_scores[0] >= all_scores).sum(axis=0)  # count of entries <= mine
    percentile = my_rank / (pool_size + 1)
    win = all_scores[0] == all_scores.max(axis=0)

    return {
        "score": all_scores[0],
        "percentile": percentile,
        "win": win,
    }
