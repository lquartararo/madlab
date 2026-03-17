"""
Tournament simulation engine.

sim_bracket() is the main entry point. Internally dispatches to:
  - _sim_bracket_matrix()  when a prob_matrix is provided
  - _sim_bracket_source()  when using a named probability source (pop / 538 / kenpom)

Both variants work with integer team indices for inner-loop speed, converting
back to team ID strings only when storing results. All num_reps replications
are processed simultaneously via vectorized NumPy operations.
"""

from __future__ import annotations

import numpy as np

from .bracket import FOLD_TO_MATCHUP, ROUND_OF_ROW, UNTANGLE, load_pred, load_teams

CURRENT_YEAR = 2026


def sim_bracket(
    bracket_empty: list[str],
    prob_matrix: np.ndarray | None = None,
    prob_source: str = "pop",
    league: str = "men",
    year: int = CURRENT_YEAR,
    home_teams: list[str] | None = None,
    num_reps: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Simulate the full tournament bracket.

    Parameters
    ----------
    bracket_empty : list[str] of 64 team IDs in seed order
    prob_matrix   : optional _ProbMatrix from bradley_terry(); if given, overrides prob_source
    prob_source   : "pop" | "538" | "kenpom"  (ignored when prob_matrix is set)
    league        : "men" | "women"
    year          : tournament year
    home_teams    : list of team names to apply home-bias adjustment (pop source only)
    num_reps      : number of simulations
    rng           : numpy random Generator; if None, uses default_rng()

    Returns
    -------
    outcome : np.ndarray of shape (63, num_reps), dtype=object (team ID strings)
        Each column is one simulation. Rows follow seed order within each round:
        rows 0-31 = R1 winners, 32-47 = R2 winners, ..., row 62 = champion.
    """
    if len(bracket_empty) != 64:
        raise ValueError("bracket_empty must have length 64")
    if rng is None:
        rng = np.random.default_rng()

    outcome = np.empty((63, num_reps), dtype=object)
    # Convert bracket to matchup order
    teams_matchup = np.array(bracket_empty)[FOLD_TO_MATCHUP].tolist()

    if prob_matrix is not None:
        return _sim_bracket_matrix(prob_matrix, league, num_reps, outcome, teams_matchup, rng)
    else:
        return _sim_bracket_source(prob_source, league, year, home_teams, num_reps, outcome,
                                   teams_matchup, rng)


# ---------------------------------------------------------------------------
# Internal: probability-matrix path
# ---------------------------------------------------------------------------

def _sim_bracket_matrix(
    prob_matrix: np.ndarray,
    league: str,
    num_reps: int,
    outcome: np.ndarray,
    teams_matchup: list[str],
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate using a pairwise win-probability matrix."""

    # Build prob matrix lookup using integer indices for speed
    if hasattr(prob_matrix, "team_ids"):
        pm_ids = prob_matrix.team_ids
    else:
        raise ValueError("prob_matrix must be a _ProbMatrix with team_ids attribute")

    id_to_idx: dict[str, int] = {t: i for i, t in enumerate(pm_ids)}

    # Extend matrix for First Four composite teams (e.g., "171/264")
    pm_ext = np.array(prob_matrix, dtype=float)
    composite_ids = list({t for t in teams_matchup if "/" in t})
    composite_idx: dict[str, int] = {}

    for composite in composite_ids:
        a, b = composite.split("/")
        if a not in id_to_idx or b not in id_to_idx:
            raise ValueError(f"prob_matrix missing team ID: {a} or {b}")
        ia, ib = id_to_idx[a], id_to_idx[b]
        p_a = pm_ext[ia, ib]  # P(a beats b)
        # Composite row/col as weighted average
        new_row = p_a * pm_ext[ia] + (1 - p_a) * pm_ext[ib]
        new_col = p_a * pm_ext[:, ia] + (1 - p_a) * pm_ext[:, ib]
        n = pm_ext.shape[0]
        new_pm = np.empty((n + 1, n + 1), dtype=float)
        new_pm[:n, :n] = pm_ext
        new_pm[n, :n] = new_row
        new_pm[:n, n] = new_col
        new_pm[n, n] = 0.5
        pm_ext = new_pm
        new_idx = n
        id_to_idx[composite] = new_idx

    # Check all bracket teams are in the matrix
    missing = [t for t in teams_matchup if t not in id_to_idx]
    if missing:
        raise ValueError(f"prob_matrix missing rows for teams: {missing[:5]}")

    # Convert teams_matchup to integer indices
    matchup_idx = np.array([id_to_idx[t] for t in teams_matchup], dtype=np.intp)

    # Replicate across all simulations
    teams_remaining = np.tile(matchup_idx, num_reps)  # (64 * num_reps,)

    for r in range(1, 7):
        n_games = 2 ** (6 - r) * num_reps
        # Each pair of consecutive entries is a matchup
        idx_a = teams_remaining[0::2]  # shape: (n_games,)
        idx_b = teams_remaining[1::2]

        # Win probabilities for each game
        win_probs = pm_ext[idx_a, idx_b]

        # Draw winners
        draws = rng.random(n_games)
        winner_indices = np.where(draws < win_probs, idx_a, idx_b)
        teams_remaining = winner_indices

        # Convert back to string IDs and store in seed order.
        # winners layout: [rep0_game0..game(n-1), rep1_game0..game(n-1), ...]
        # reshape(num_reps, n_round_games).T gives (n_round_games, num_reps)
        # so column j = rep j's winners in matchup order → UNTANGLE → seed order.
        n_round_games = 2 ** (6 - r)
        round_results_idx = winner_indices.reshape(num_reps, n_round_games).T
        # Convert indices back to team IDs
        idx_to_id = {v: k for k, v in id_to_idx.items()}
        round_results_str = np.vectorize(idx_to_id.__getitem__)(round_results_idx)
        row_mask = np.where(ROUND_OF_ROW == r)[0]
        outcome[row_mask, :] = round_results_str[UNTANGLE[r], :]

    return outcome


# ---------------------------------------------------------------------------
# Internal: named-source path
# ---------------------------------------------------------------------------

def _sim_bracket_source(
    prob_source: str,
    league: str,
    year: int,
    home_teams: list[str] | None,
    num_reps: int,
    outcome: np.ndarray,
    teams_matchup: list[str],
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate using round-advancement probability tables (pop / 538 / kenpom)."""
    import polars as pl
    if isinstance(prob_source, str) and prob_source == "kenpom" and (league == "women" or year < 2018):
        raise ValueError("kenpom source is only available for men's tournaments 2018+")

    if isinstance(prob_source, pl.DataFrame):
        # Passed directly as a DataFrame — use as-is
        prob_df = prob_source
    elif isinstance(prob_source, str) and prob_source == "pop" and home_teams:
        from .bias import add_home_bias
        prob_df = add_home_bias(home_teams, league=league, year=year)
    else:
        prob_df = load_pred(prob_source, league, year)

    round_cols = ["round1", "round2", "round3", "round4", "round5", "round6"]
    # Determine ID column
    id_col = "team_id" if "team_id" in prob_df.columns else "name"

    prob_np = prob_df.select(round_cols).to_numpy(allow_copy=True).astype(float)  # (64, 6)
    id_list = prob_df[id_col].cast(str).to_list()
    prob_lookup: dict[str, np.ndarray] = {tid: prob_np[i] for i, tid in enumerate(id_list)}

    # Handle First Four composites.
    # If pred_pop already contains the composite key (e.g. "2378/47"), use it directly.
    # Otherwise, build it as a weighted average of the two component teams.
    for t in set(teams_matchup):
        if "/" in t and t not in prob_lookup:
            a, b = t.split("/")
            pa = prob_lookup.get(a)
            pb = prob_lookup.get(b)
            if pa is None or pb is None:
                raise ValueError(f"No predictions for {a} or {b} (needed for composite {t})")
            p_a = pa[0]
            prob_lookup[t] = p_a * pa + (1 - p_a) * pb

    missing = [t for t in teams_matchup if t not in prob_lookup and "/" not in t]
    if missing:
        raise ValueError(f"No predictions from source for teams: {missing[:5]}. Is year correct?")

    # Derived marginal probabilities: P(advance in round r and NOT in round r+1)
    def cumulative_to_marginal(p6: np.ndarray) -> np.ndarray:
        """P(win round r, lose round r+1) for r=1..5, plus P(win championship)."""
        diffs = np.diff(p6)  # [p2-p1, p3-p2, ...] -- these are negative
        return np.concatenate([-diffs, [p6[-1]]])  # each element >= 0

    marginal: dict[str, np.ndarray] = {t: cumulative_to_marginal(p) for t, p in prob_lookup.items()}

    # Tournament structure: 64 teams -> 6 rounds
    # In each round r, teams are grouped into 2^(6-r) slots; one winner per slot per sim
    # The bracket in matchup order: teams_matchup[2k] plays teams_matchup[2k+1] in R1
    # Winners of R1 play each other in R2, etc.

    # Group sizes in the original 64 by round
    for r in range(1, 7):
        n_slots = 2 ** (6 - r)
        group_size = 64 // n_slots  # original teams per slot

        round_outcomes = np.empty((n_slots, num_reps), dtype=object)

        for slot in range(n_slots):
            group_teams = teams_matchup[slot * group_size: (slot + 1) * group_size]
            probs = np.array([marginal[t][r - 1] for t in group_teams], dtype=float)

            # Normalize (handle zero or negative values from floating point)
            probs = np.clip(probs, 0, None)
            total = probs.sum()
            if total <= 0:
                probs = np.ones(len(group_teams)) / len(group_teams)
            else:
                probs /= total

            choices = rng.choice(len(group_teams), size=num_reps, p=probs, replace=True)
            round_outcomes[slot, :] = np.array(group_teams)[choices]

        # Store in seed order
        row_mask = np.where(ROUND_OF_ROW == r)[0]
        outcome[row_mask, :] = round_outcomes[UNTANGLE[r], :]

    # Enforce consistency: if team T wins round r, they must have won all previous rounds
    # For each round r >= 2, find the round-1 through r-1 entries for each winner and overwrite
    for r in range(2, 7):
        n_slots_r = 2 ** (6 - r)
        group_size_r = 64 // n_slots_r
        row_mask_r = np.where(ROUND_OF_ROW == r)[0]

        for slot in range(n_slots_r):
            winners = outcome[row_mask_r[slot], :]  # (num_reps,) - winner of this slot

            for prev_r in range(1, r):
                n_slots_prev = 2 ** (6 - prev_r)
                group_size_prev = 64 // n_slots_prev
                row_mask_prev = np.where(ROUND_OF_ROW == prev_r)[0]

                # Which sub-slot of prev_r does this slot correspond to?
                # slot of round r -> maps to slot*ratio .. (slot+1)*ratio-1 slots in prev_r
                ratio = n_slots_prev // n_slots_r
                for sub in range(ratio):
                    prev_slot = slot * ratio + sub
                    prev_group = teams_matchup[
                        prev_slot * group_size_prev: (prev_slot + 1) * group_size_prev
                    ]
                    # Only enforce if winner is in this prev group
                    in_group = np.isin(winners, prev_group)
                    if in_group.any():
                        outcome[row_mask_prev[prev_slot], in_group] = winners[in_group]

    return outcome
