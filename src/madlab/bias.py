"""
Home-team bias adjustment for population pick distributions.

Adds +3/4 to the log-odds of the conditional round-win probability for
specified teams, then re-normalises column totals. See R add.home.bias for
the original derivation (CBS Sports / Brad Null, 2016).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from .bracket import load_pred


def add_home_bias(
    teams: list[str],
    league: str = "men",
    year: int = 2025,
    bias_logodds: float = 0.75,
) -> pl.DataFrame:
    """
    Adjust population pick probabilities to reflect home-team bias.

    Parameters
    ----------
    teams        : list of team names matching the 'name' column of pred.pop.*.*
    league       : "men" | "women"
    year         : tournament year
    bias_logodds : log-odds increase for home teams (default 3/4 per R implementation)

    Returns
    -------
    DataFrame matching pred.pop structure with adjusted probabilities.
    """
    prob_df = load_pred("pop", league, year)
    round_cols = ["round1", "round2", "round3", "round4", "round5", "round6"]

    name_col = "name"
    if name_col not in prob_df.columns:
        raise ValueError("pred.pop DataFrame must have a 'name' column")

    prob_names = prob_df[name_col].to_list()
    missing = [t for t in teams if t not in prob_names]
    if missing:
        raise ValueError(f"Teams not found in pred.pop.{league}.{year}: {missing}")

    prob_np = prob_df.select(round_cols).to_numpy().astype(float)  # (64, 6)
    n_teams = prob_np.shape[0]

    # Conditional probabilities: P(win round r | won round r-1)
    cond = np.empty_like(prob_np)
    cond[:, 0] = prob_np[:, 0]  # R1: no conditioning
    for r in range(1, 6):
        with np.errstate(invalid="ignore", divide="ignore"):
            cond[:, r] = np.where(prob_np[:, r - 1] > 0, prob_np[:, r] / prob_np[:, r - 1], 0.0)

    # Convert to log-odds, apply bias, convert back
    with np.errstate(divide="ignore", invalid="ignore"):
        lo = np.log(cond / (1 - cond))

    home_mask = np.array([n in teams for n in prob_names])
    lo_bias = lo.copy()
    lo_bias[home_mask] += bias_logodds

    # Sigmoid back to probabilities
    cond_bias = 1 / (1 + np.exp(-lo_bias))
    # Restore NaN/inf cases (0 or 1 probs became NaN through log-odds)
    not_finite = ~np.isfinite(lo)
    cond_bias[not_finite] = cond[not_finite]

    # Cumulative probabilities
    prob_bias = np.empty_like(prob_np)
    prob_bias[:, 0] = cond_bias[:, 0]
    for r in range(1, 6):
        prob_bias[:, r] = prob_bias[:, r - 1] * cond_bias[:, r]

    # Re-normalise so each round's column sums to the expected number of winners
    expected_winners = 2 ** np.arange(5, -1, -1).astype(float)  # [32, 16, 8, 4, 2, 1]
    col_sums = prob_bias.sum(axis=0)
    col_sums = np.where(col_sums > 0, col_sums, 1.0)
    prob_bias = prob_bias * (expected_winners / col_sums)

    # Rebuild DataFrame
    result = prob_df.clone()
    for i, col in enumerate(round_cols):
        result = result.with_columns(pl.Series(col, prob_bias[:, i]))

    return result
