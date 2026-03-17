"""Tests for bracket utilities (fold/unfold, data loading, scoring)."""

import json
from pathlib import Path

import numpy as np
import pytest

from marchmadness.bracket import (
    _fold_indices,
    _unfold_indices,
    FOLD_TO_MATCHUP,
    UNTANGLE,
    ROUND_OF_ROW,
    load_bracket,
    input_bracket_filled,
)
from marchmadness.evaluate import score_bracket


def test_fold_unfold_roundtrip():
    """fold then unfold with block_size=1 should recover original order."""
    x = np.arange(64)
    folded = x[_fold_indices(64, 1)]
    unfolded = folded[_unfold_indices(64, 1)]
    np.testing.assert_array_equal(x, unfolded)


def test_fold_unfold_small():
    """Verify fold on a small example: fold([1,2,3,4], block_size=1) = [1,4,2,3]."""
    x = np.array([1, 2, 3, 4])
    folded = x[_fold_indices(4, 1)]
    np.testing.assert_array_equal(folded, [1, 4, 2, 3])


def test_fold_to_matchup_length():
    assert len(FOLD_TO_MATCHUP) == 64


def test_untangle_lengths():
    expected = {1: 32, 2: 16, 3: 8, 4: 4, 5: 2, 6: 1}
    for r, n in expected.items():
        assert len(UNTANGLE[r]) == n, f"Round {r} untangle length mismatch"


def test_round_of_row():
    assert len(ROUND_OF_ROW) == 63
    assert sum(ROUND_OF_ROW == 1) == 32
    assert sum(ROUND_OF_ROW == 2) == 16
    assert sum(ROUND_OF_ROW == 3) == 8
    assert sum(ROUND_OF_ROW == 4) == 4
    assert sum(ROUND_OF_ROW == 5) == 2
    assert sum(ROUND_OF_ROW == 6) == 1


def test_load_bracket():
    bracket = load_bracket("men", 2025)
    assert len(bracket) == 64
    assert all(isinstance(t, str) for t in bracket)


def test_input_bracket_filled_length():
    picks = [str(i) for i in range(63)]
    result = input_bracket_filled(picks)
    assert len(result) == 63


def test_input_bracket_filled_reorder():
    """Should be a permutation of the input."""
    picks = [str(i) for i in range(63)]
    result = input_bracket_filled(picks)
    assert sorted(result) == sorted(picks)


def test_score_bracket_all_correct():
    """If picks == outcome, score should equal sum of round bonuses."""
    bracket_empty = load_bracket("men", 2025)
    # Make up 63 "team IDs" that are just the bracket entries
    # Use the first 63 unique entries of the bracket (will have duplicates in outcome)
    picks = np.array(["T" + str(i) for i in range(63)])
    outcome = picks.copy()
    bonus_round = np.array([1, 2, 4, 8, 16, 32], dtype=float)
    bonus_seed = np.zeros(16)

    # Use the real bracket_empty for seed lookup
    # Override picks to match actual team IDs
    b = bracket_empty
    # Simulate: all top seeds win every game
    # In round 1 rows (0-31): winner is seed 1 of each matchup
    # For test: just check that if picks == outcome, score sums correctly
    fake_bracket_empty = ["T" + str(i) for i in range(64)]
    fake_picks = np.array(["T" + str(i) for i in range(63)])
    fake_outcome = fake_picks.copy()

    expected = float(sum(
        bonus_round[r - 1] * (2 ** (6 - r))
        for r in range(1, 7)
    ))
    score = score_bracket(
        fake_bracket_empty, fake_picks, fake_outcome,
        bonus_round=bonus_round.tolist(), bonus_seed=bonus_seed.tolist()
    )
    assert abs(float(score) - expected) < 1e-9


def test_score_bracket_none_correct():
    """If no picks match outcome, score should be 0."""
    fake_bracket_empty = ["T" + str(i) for i in range(64)]
    picks = np.array(["T" + str(i) for i in range(63)])
    outcome = np.array(["X" + str(i) for i in range(63)])
    score = score_bracket(fake_bracket_empty, picks, outcome)
    assert float(score) == 0.0


def test_score_bracket_multiple_sims():
    """score_bracket with 2D outcome returns (n_brackets, n_sims) shape."""
    fake_bracket_empty = ["T" + str(i) for i in range(64)]
    picks = np.array(["T" + str(i) for i in range(63)])[:, None]  # (63, 1)
    outcome = np.stack([picks[:, 0], picks[:, 0]], axis=1)  # (63, 2) - both correct
    scores = score_bracket(fake_bracket_empty, picks, outcome)
    assert scores.shape == (1, 2)
    assert scores[0, 0] == scores[0, 1]
