"""Tests for the Bradley-Terry model."""

import numpy as np
import pytest

from marchmadness.bracket import load_games, load_bracket
from marchmadness.model import bradley_terry


@pytest.fixture(scope="module")
def games_2025():
    return load_games("men", 2025)


@pytest.fixture(scope="module")
def prob_matrix_2025(games_2025):
    return bradley_terry(games_2025)


def test_prob_matrix_shape(prob_matrix_2025, games_2025):
    """Matrix should be square with one row/col per team."""
    n = prob_matrix_2025.shape[0]
    assert prob_matrix_2025.shape == (n, n)
    assert n > 100  # should have many teams


def test_prob_matrix_diagonal(prob_matrix_2025):
    """P(team beats itself) should be 0.5."""
    diag = np.diag(prob_matrix_2025)
    np.testing.assert_allclose(diag, 0.5, atol=0.01)


def test_prob_matrix_antisymmetry(prob_matrix_2025):
    """P(i beats j) + P(j beats i) should be ~1."""
    total = prob_matrix_2025 + prob_matrix_2025.T
    np.testing.assert_allclose(total, 1.0, atol=1e-10)


def test_prob_matrix_range(prob_matrix_2025):
    """All probabilities should be in [0, 1]."""
    assert np.all(prob_matrix_2025 >= 0)
    assert np.all(prob_matrix_2025 <= 1)


def test_team_ids_attribute(prob_matrix_2025, games_2025):
    """team_ids should match teams in games."""
    assert hasattr(prob_matrix_2025, "team_ids")
    assert len(prob_matrix_2025.team_ids) == prob_matrix_2025.shape[0]


def test_sigma_positive(prob_matrix_2025):
    """Estimated sigma should be positive."""
    assert prob_matrix_2025.sigma > 0


def test_mov_cap_option(games_2025):
    """MoV cap should produce a valid matrix."""
    pm = bradley_terry(games_2025, mov_cap=20.0)
    assert pm.shape[0] > 0
    np.testing.assert_allclose(np.diag(pm), 0.5, atol=0.01)


def test_no_mov_cap(games_2025):
    """No MoV cap (None) should also work."""
    pm = bradley_terry(games_2025, mov_cap=None)
    assert pm.shape[0] > 0
