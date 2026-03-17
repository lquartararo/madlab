"""Tests for bracket simulation."""

import numpy as np
import pytest

from marchmadness.bracket import load_bracket
from marchmadness.simulate import sim_bracket
from marchmadness.model import bradley_terry
from marchmadness.bracket import load_games


@pytest.fixture(scope="module")
def bracket_2025():
    return load_bracket("men", 2025)


@pytest.fixture(scope="module")
def prob_matrix_2025():
    games = load_games("men", 2025)
    return bradley_terry(games)


def test_sim_bracket_shape_single(bracket_2025, prob_matrix_2025):
    """Single simulation should return (63, 1) array."""
    rng = np.random.default_rng(42)
    out = sim_bracket(bracket_2025, prob_matrix=prob_matrix_2025, num_reps=1, rng=rng)
    assert out.shape == (63, 1)


def test_sim_bracket_shape_multi(bracket_2025, prob_matrix_2025):
    """Multiple simulations should return (63, N) array."""
    rng = np.random.default_rng(42)
    out = sim_bracket(bracket_2025, prob_matrix=prob_matrix_2025, num_reps=10, rng=rng)
    assert out.shape == (63, 10)


def test_sim_bracket_teams_valid(bracket_2025, prob_matrix_2025):
    """All simulated team IDs should be in the original bracket."""
    rng = np.random.default_rng(42)
    out = sim_bracket(bracket_2025, prob_matrix=prob_matrix_2025, num_reps=5, rng=rng)
    flat_teams = set(out.ravel())
    # Extract base team IDs (remove First Four composite separators)
    valid_ids = set()
    for t in bracket_2025:
        for part in t.split("/"):
            valid_ids.add(part)
    assert flat_teams.issubset(valid_ids)


def test_sim_bracket_champion_in_bracket(bracket_2025, prob_matrix_2025):
    """The champion (last row) should be one of the bracket teams."""
    rng = np.random.default_rng(99)
    out = sim_bracket(bracket_2025, prob_matrix=prob_matrix_2025, num_reps=20, rng=rng)
    valid_ids = set()
    for t in bracket_2025:
        for part in t.split("/"):
            valid_ids.add(part)
    champions = set(out[62, :])
    assert champions.issubset(valid_ids)


def test_sim_bracket_pop_source(bracket_2025):
    """Simulation with pop source should run without error."""
    rng = np.random.default_rng(7)
    out = sim_bracket(
        bracket_2025, prob_source="pop", league="men", year=2025, num_reps=5, rng=rng
    )
    assert out.shape == (63, 5)


def test_sim_bracket_reproducible(bracket_2025, prob_matrix_2025):
    """Same seed should produce same result."""
    out1 = sim_bracket(bracket_2025, prob_matrix=prob_matrix_2025, num_reps=3,
                        rng=np.random.default_rng(123))
    out2 = sim_bracket(bracket_2025, prob_matrix=prob_matrix_2025, num_reps=3,
                        rng=np.random.default_rng(123))
    np.testing.assert_array_equal(out1, out2)
