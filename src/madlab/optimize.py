"""
Bracket optimization: find_bracket()

Generates num_candidates random brackets and evaluates each against num_sims
simulated pools. Picks the best candidate by the chosen criterion.

Key optimization: scoring is fully vectorized across all candidates × simulations
using NumPy broadcasting (no Python loop over simulations).
"""

from __future__ import annotations

import time

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

    def _emit(msg: str) -> None:
        if print_progress:
            print(msg)
        if on_progress:
            on_progress(msg)

    # 1. Generate candidate brackets (63, num_candidates)
    _emit(f"Generating {num_candidates} candidate brackets…")
    candidates = sim_bracket(
        bracket_empty, prob_matrix=prob_matrix, prob_source=prob_source,
        league=league, year=year, num_reps=num_candidates, rng=rng,
    )

    # 2. Determine chunk size to cap peak memory at ~400 MB.
    #    Each pool bracket = 63 team-ID strings ≈ 100 bytes in numpy unicode.
    _BUDGET_BYTES = 400 * 1024 * 1024
    _bytes_per_bracket = 63 * 100
    _max_pool_per_chunk = max(1, _BUDGET_BYTES // (_bytes_per_bracket * pool_size))
    sim_chunk = min(num_sims, max(1, _max_pool_per_chunk))

    _emit(f"Simulating {num_sims:,} tournaments…")

    # Accumulators
    candidate_scores = np.zeros((num_candidates, num_sims))
    pool_max = np.full(num_sims, -np.inf)
    pool_all = np.zeros((pool_size, num_sims))  # only needed for percentile

    emit_every = max(1, num_sims // 50)   # ~50 progress updates total
    scoring_start = time.perf_counter()

    def _fmt_remaining(done: int) -> str:
        elapsed = time.perf_counter() - scoring_start
        if done < 2 or elapsed < 0.5:
            return ""
        rate = done / elapsed
        secs = int((num_sims - done) / rate)
        if secs < 60:
            return f" · ~{secs}s remaining"
        return f" · ~{secs // 60}m {secs % 60}s remaining"

    for start in range(0, num_sims, sim_chunk):
        n = min(sim_chunk, num_sims - start)

        outcomes_chunk = sim_bracket(
            bracket_empty, prob_matrix=prob_matrix, prob_source=prob_source,
            league=league, year=year, num_reps=n, rng=rng,
        )
        candidate_scores[:, start:start + n] = score_bracket(
            bracket_empty, candidates, outcomes_chunk,
            bonus_round=bonus_round, bonus_seed=bonus_seed, bonus_combine=bonus_combine,
        )
        pool_chunk = sim_bracket(
            bracket_empty, prob_source=pool_source,
            league=league, year=year, home_teams=pool_bias,
            num_reps=n * pool_size, rng=rng,
        )

        for j in range(n):
            pool_j = pool_chunk[:, j * pool_size:(j + 1) * pool_size]
            s = score_bracket(
                bracket_empty, pool_j, outcomes_chunk[:, j],
                bonus_round=bonus_round, bonus_seed=bonus_seed, bonus_combine=bonus_combine,
            )
            s_flat = np.squeeze(s) if s.ndim > 1 else s
            pool_max[start + j] = float(np.max(s_flat))
            if criterion == "percentile":
                pool_all[:, start + j] = s_flat

            done = start + j + 1
            if done % emit_every == 0 or done == num_sims:
                _emit(f"Simulated {done:,}/{num_sims:,} tournaments{_fmt_remaining(done)}")

        del pool_chunk, outcomes_chunk

    _emit(f"Selecting best bracket…")

    # 3. Select best candidate by criterion
    if criterion == "percentile":
        all_scores = np.vstack([pool_all, candidate_scores])  # (pool_size + num_candidates, num_sims)
        n_total = all_scores.shape[0]
        ranks = np.empty_like(all_scores, dtype=float)
        for j in range(num_sims):
            col = all_scores[:, j]
            ranks[:, j] = np.array([(col <= v).sum() for v in col], dtype=float)
        percentiles = ranks[pool_size:] / n_total
        best_idx = int(np.argmax(percentiles.mean(axis=1)))

    elif criterion == "score":
        best_idx = int(np.argmax(candidate_scores.mean(axis=1)))

    elif criterion == "win":
        wins = candidate_scores >= pool_max[None, :]
        best_idx = int(np.argmax(wins.mean(axis=1)))

    return candidates[:, best_idx].tolist()
