"""
Bradley-Terry model for estimating pairwise win probabilities from game scores.

Improvements over R version:
- sklearn RidgeCV uses efficient LOO-CV via SVD (faster than glmnet's 100-lambda grid)
- Optional margin-of-victory capping to reduce outlier blowout influence
- Optional reduced OT penalty (OT games set to a small margin vs. hard 0)
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl


def _ridge_cv(X: np.ndarray, y: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Ridge regression with LOO-CV via SVD. Replicates sklearn RidgeCV behaviour."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    Uy = U.T @ y
    s2 = s ** 2
    best_alpha, best_loss = alphas[0], np.inf
    for alpha in alphas:
        d = s2 / (s2 + alpha)
        y_hat = U @ (d * Uy)
        h = np.sum(U ** 2 * d, axis=1)
        loo = (y - y_hat) / (1 - h)
        loss = np.mean(loo ** 2)
        if loss < best_loss:
            best_loss, best_alpha = loss, alpha
    return Vt.T @ ((s / (s2 + best_alpha)) * Uy)


def bradley_terry(
    games: pl.DataFrame,
    mov_cap: float | None = 25.0,
    ot_margin: float = 0.0,
) -> np.ndarray:
    """
    Fit a Bradley-Terry model on game score data and return a win-probability matrix.

    Parameters
    ----------
    games : DataFrame with columns game_id, home_id, away_id, home_score, away_score,
            neutral (0/1), ot (string, empty if regulation)
    mov_cap : optional cap on margin of victory (e.g., 25 points). Reduces outlier
              influence from blowouts. Set to None to disable.
    ot_margin : score differential to assign OT games instead of exactly 0.
                0.0 (default) preserves the R behaviour; a small value like 0.5
                can be used to retain some information.

    Returns
    -------
    prob_matrix : np.ndarray of shape (n_teams, n_teams)
        prob_matrix[i, j] = P(team i beats team j on a neutral site).
        Row/column ordering matches sorted unique team IDs; use team_ids attribute.

    Attributes
    ----------
    The returned array has two extra attributes attached:
      - prob_matrix.team_ids  : sorted list of team ID strings
      - prob_matrix.sigma     : estimated score std-dev (for diagnostics)
    """
    required = {"game_id", "home_id", "away_id", "home_score", "away_score", "neutral", "ot"}
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"games DataFrame missing columns: {missing}")

    # Drop games involving non-D1 teams (marked 'NA' in the R data)
    g = games.filter((pl.col("home_id") != "NA") & (pl.col("away_id") != "NA"))

    home_ids = g["home_id"].cast(str).to_numpy()
    away_ids = g["away_id"].cast(str).to_numpy()
    neutral = g["neutral"].cast(float).to_numpy().astype(bool)
    ot = g["ot"].cast(str).to_numpy()

    home_scores = g["home_score"].cast(float).to_numpy()
    away_scores = g["away_score"].cast(float).to_numpy()

    # Build team index
    all_ids = np.unique(np.concatenate([home_ids, away_ids]))
    team_to_idx = {t: i for i, t in enumerate(all_ids)}
    n_teams = len(all_ids)
    n_games = len(g)

    # Build sparse design matrix:
    #   column 0: home-court indicator (1 if not neutral)
    #   columns 1..n_teams: team indicators (+1 home, -1 away)
    n_cols = 1 + n_teams
    home_indicator_rows = np.where(~neutral)[0]
    team_rows = np.concatenate([np.arange(n_games), np.arange(n_games)])
    home_col_indices = np.array([team_to_idx[t] + 1 for t in home_ids])
    away_col_indices = np.array([team_to_idx[t] + 1 for t in away_ids])
    team_cols = np.concatenate([home_col_indices, away_col_indices])
    team_vals = np.concatenate([np.ones(n_games), -np.ones(n_games)])

    X = np.zeros((n_games, n_cols), dtype=np.float64)
    X[home_indicator_rows, 0] = 1.0
    X[team_rows, team_cols] = team_vals

    # Build response: home score minus away score
    y = home_scores - away_scores
    # OT games: use ot_margin (default 0 = R behavior)
    ot_mask = ot != ""
    y[ot_mask] = ot_margin

    # Optional MoV cap
    if mov_cap is not None:
        y = np.clip(y, -mov_cap, mov_cap)

    # Fit ridge regression with LOO-CV via SVD (pure numpy, replicates sklearn RidgeCV)
    alphas = np.exp(np.linspace(np.log(n_games * np.exp(5)), np.log(n_games * np.exp(-10)), 100))
    beta = _ridge_cv(X, y, alphas)

    # Estimate score differential std-dev
    y_pred = X @ beta
    sigma = float(np.sqrt(np.mean((y - y_pred) ** 2)))

    # Build point-spread matrix: beta[i] - beta[j] for all pairs
    team_betas = beta[1:]
    point_spread = team_betas[:, None] - team_betas[None, :]  # shape (n_teams, n_teams)

    # Convert to win probabilities via normal CDF: 0.5 * erfc(-x / (sigma * sqrt2))
    _erfc = np.frompyfunc(math.erfc, 1, 1)
    prob_matrix = (0.5 * _erfc(-point_spread / (sigma * math.sqrt(2)))).astype(float)

    # Attach metadata as custom attributes (numpy structured approach)
    prob_matrix = prob_matrix.view(np.ndarray)
    prob_matrix.flags.writeable = True

    # Return as a subclass that carries team_ids
    result = _ProbMatrix(prob_matrix)
    result.team_ids = all_ids.tolist()
    result.sigma = sigma
    return result


class _ProbMatrix(np.ndarray):
    """np.ndarray subclass that carries team_ids and sigma as attributes."""
    team_ids: list[str]
    sigma: float

    def __new__(cls, arr):
        return np.asarray(arr).view(cls)

    def __array_finalize__(self, obj):
        self.team_ids = getattr(obj, "team_ids", [])
        self.sigma = getattr(obj, "sigma", float("nan"))
