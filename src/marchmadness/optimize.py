"""
Bracket optimization: find_bracket()

Generates num_candidates random brackets and evaluates each against num_sims
simulated pools. Picks the best candidate by the chosen criterion.

Key optimization: scoring is fully vectorized across all candidates × simulations
using NumPy broadcasting (no Python loop over simulations).
"""

from __future__ import annotations

import numpy as np

from .bracket import ROUND_OF_ROW
from .simulate import sim_bracket, CURRENT_YEAR
from .evaluate import score_bracket


def find_bracket(
    bracket_empty: list[str],
    prob_matrix: np.ndarray | None = None,
    prob_source: str = "pop",
    pool_source: str = "pop",
    league: str = "men",
    year: int = CURRENT_YEAR,
    pool_bias: list[str] | None = None,
    num_candidates: int = 100,
    num_sims: int = 1000,
    criterion: str = "percentile",
    pool_size: int = 30,
    bonus_round: list[float] | None = None,
    bonus_seed: list[float] | None = None,
    bonus_combine: str = "add",
    print_progress: bool = True,
    on_progress: object = None,
    rng: np.random.Generator | None = None,
) -> list[str]:
    """
    Find a good bracket by evaluating random candidates against simulated pools.

    Parameters
    ----------
    bracket_empty   : list of 64 team IDs in seed order
    prob_matrix     : optional _ProbMatrix from bradley_terry(); overrides prob_source
    prob_source     : source for candidate bracket generation and outcome simulation
    pool_source     : source for opponent bracket simulation
    league          : "men" | "women"
    year            : tournament year
    pool_bias       : team names to apply home bias (pool_source="pop" only)
    num_candidates  : number of candidate brackets to evaluate
    num_sims        : number of pool simulations per candidate
    criterion       : "percentile" | "score" | "win"
    pool_size       : number of pool opponents (excluding you)
    bonus_round     : length-6 scoring vector
    bonus_seed      : length-16 seed-bonus vector
    bonus_combine   : "add" | "multiply"
    print_progress  : print progress to stdout
    rng             : numpy random Generator

    Returns
    -------
    list[str] of 63 team IDs (the best bracket found)
    """
    if num_candidates < 2:
        raise ValueError("num_candidates must be at least 2")
    if num_sims < 2:
        raise ValueError("num_sims must be at least 2")
    if pool_size < 1:
        raise ValueError("pool_size must be at least 1")
    if criterion not in ("percentile", "score", "win"):
        raise ValueError(f"criterion must be 'percentile', 'score', or 'win'")
    if isinstance(prob_source, str) and prob_source == "kenpom" and (league == "women" or year < 2018):
        raise ValueError("kenpom prob_source is only available for men's 2018+")
    if isinstance(pool_source, str) and pool_source == "kenpom" and (league == "women" or year < 2018):
        raise ValueError("kenpom pool_source is only available for men's 2018+")

    if rng is None:
        rng = np.random.default_rng()
    if bonus_round is None:
        bonus_round = [1, 2, 4, 8, 16, 32]
    if bonus_seed is None:
        bonus_seed = [0] * 16

    # 1. Generate candidate brackets (63, num_candidates)
    candidates = sim_bracket(
        bracket_empty, prob_matrix=prob_matrix, prob_source=prob_source,
        league=league, year=year, num_reps=num_candidates, rng=rng,
    )

    def _emit(msg: str) -> None:
        if print_progress:
            print(msg)
        if on_progress:
            on_progress(msg)

    _emit(f"Generating {num_candidates} candidate brackets…")

    # 2. Simulate pool opponents: (63, num_sims * pool_size)
    pool = sim_bracket(
        bracket_empty, prob_source=pool_source,
        league=league, year=year, home_teams=pool_bias,
        num_reps=num_sims * pool_size, rng=rng,
    )

    # 3. Simulate tournament outcomes: (63, num_sims)
    outcomes = sim_bracket(
        bracket_empty, prob_matrix=prob_matrix, prob_source=prob_source,
        league=league, year=year, num_reps=num_sims, rng=rng,
    )

    _emit(f"Simulating {num_sims} pools of {pool_size} opponents…")
    _emit(f"Scoring {num_sims * (num_candidates + pool_size):,} brackets…")

    # 4. Vectorized scoring: candidates (63, num_candidates) vs outcomes (63, num_sims)
    # -> candidate_scores: (num_candidates, num_sims)
    candidate_scores = score_bracket(
        bracket_empty, candidates, outcomes,
        bonus_round=bonus_round, bonus_seed=bonus_seed, bonus_combine=bonus_combine,
    )  # (num_candidates, num_sims)

    # 5. Score pool brackets per simulation
    # pool: (63, num_sims * pool_size) -> we need (pool_size, num_sims)
    pool_scores = np.empty((pool_size, num_sims))
    for i in range(num_sims):
        pool_slice = pool[:, i * pool_size:(i + 1) * pool_size]
        s = score_bracket(
            bracket_empty, pool_slice, outcomes[:, i],
            bonus_round=bonus_round, bonus_seed=bonus_seed, bonus_combine=bonus_combine,
        )
        pool_scores[:, i] = np.squeeze(s) if s.ndim > 1 else s

    # 6. Select best candidate by criterion
    # all_scores shape: (pool_size + num_candidates, num_sims)
    all_scores = np.vstack([pool_scores, candidate_scores])

    if criterion == "percentile":
        # For each candidate, compute mean percentile rank across simulations
        n_total = all_scores.shape[0]
        # ranks[i, j] = rank of all_scores[i, j] within column j (1-based, ties = max)
        # Efficient: argsort twice
        ranks = np.empty_like(all_scores, dtype=float)
        for j in range(num_sims):
            col = all_scores[:, j]
            order = np.argsort(col)
            ranks_col = np.empty(n_total)
            ranks_col[order] = np.arange(1, n_total + 1)
            # ties: use max rank
            for k in range(n_total):
                ranks_col[k] = (col <= col[k]).sum()
            ranks[:, j] = ranks_col
        percentiles = ranks[pool_size:] / n_total  # (num_candidates, num_sims)
        best_idx = int(np.argmax(percentiles.mean(axis=1)))

    elif criterion == "score":
        best_idx = int(np.argmax(candidate_scores.mean(axis=1)))

    elif criterion == "win":
        pool_max = pool_scores.max(axis=0)  # (num_sims,)
        wins = candidate_scores >= pool_max[None, :]  # (num_candidates, num_sims)
        best_idx = int(np.argmax(wins.mean(axis=1)))

    return candidates[:, best_idx].tolist()
