"""
Bradley-Terry model for estimating pairwise win probabilities from game scores.

Improvements over R version:
- sklearn RidgeCV uses efficient LOO-CV via SVD (faster than glmnet's 100-lambda grid)
- Optional margin-of-victory capping to reduce outlier blowout influence
- Optional reduced OT penalty (OT games set to a small margin vs. hard 0)
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import sparse, stats
from sklearn.linear_model import RidgeCV


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

    X = sparse.csr_matrix(
        (
            np.concatenate([np.ones(len(home_indicator_rows)), team_vals]),
            (
                np.concatenate([home_indicator_rows, team_rows]),
                np.concatenate([np.zeros(len(home_indicator_rows), dtype=int), team_cols]),
            ),
        ),
        shape=(n_games, n_cols),
    )

    # Build response: home score minus away score
    y = home_scores - away_scores
    # OT games: use ot_margin (default 0 = R behavior)
    ot_mask = ot != ""
    y[ot_mask] = ot_margin

    # Optional MoV cap
    if mov_cap is not None:
        y = np.clip(y, -mov_cap, mov_cap)

    # Fit ridge regression with LOO-CV (faster than 100-lambda glmnet grid)
    # alphas correspond to lambda*n in sklearn vs lambda in R glmnet
    alphas = np.exp(np.linspace(np.log(n_games * np.exp(5)), np.log(n_games * np.exp(-10)), 100))
    ridge = RidgeCV(alphas=alphas, fit_intercept=False)
    ridge.fit(X, y)
    beta = ridge.coef_  # [home_advantage, beta_team_0, ..., beta_team_{n-1}]

    # Estimate score differential std-dev
    y_pred = X @ beta
    sigma = float(np.sqrt(np.mean((y - y_pred) ** 2)))

    # Build point-spread matrix: beta[i] - beta[j] for all pairs
    team_betas = beta[1:]
    point_spread = team_betas[:, None] - team_betas[None, :]  # shape (n_teams, n_teams)

    # Convert to win probabilities via normal CDF
    prob_matrix = stats.norm.cdf(point_spread, scale=sigma)

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
