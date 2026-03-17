"""
madlab - A numerical computing environment for March Madness.

Port of the R package mRchmadness by Eli Shayer and Scott Powers.

Quick start
-----------
>>> from marchmadness.bracket import load_bracket, load_games
>>> from marchmadness.model import bradley_terry
>>> from marchmadness.optimize import find_bracket
>>> from marchmadness.evaluate import test_bracket

>>> games = load_games("men", 2025)
>>> prob_matrix = bradley_terry(games)
>>> bracket_empty = load_bracket("men", 2025)
>>> my_bracket = find_bracket(bracket_empty, prob_matrix=prob_matrix, league="men")
>>> results = test_bracket(bracket_empty, my_bracket, prob_matrix=prob_matrix, league="men")
"""

from .bracket import (
    load_bracket,
    load_games,
    load_pred,
    load_teams,
    save_bracket,
    save_games,
    save_pred,
    draw_bracket,
    input_bracket_filled,
    FOLD_TO_MATCHUP,
    UNTANGLE,
    ROUND_OF_ROW,
)
from .model import bradley_terry
from .simulate import sim_bracket, CURRENT_YEAR
from .evaluate import score_bracket, test_bracket
from .optimize import find_bracket
from .bias import add_home_bias
from .scrape import (
    scrape_game_results,
    scrape_population_distribution,
    prep_data,
    get_challenge_id,
    register_challenge_id,
    validate_bracket,
    bracket_set_first_four,
)

__version__ = "1.2026.0"
__all__ = [
    "load_bracket", "load_games", "load_pred", "load_teams",
    "save_bracket", "save_games", "save_pred",
    "draw_bracket", "input_bracket_filled",
    "bradley_terry",
    "sim_bracket",
    "score_bracket", "test_bracket",
    "find_bracket",
    "add_home_bias",
    "scrape_game_results", "scrape_population_distribution",
    "prep_data", "get_challenge_id", "register_challenge_id",
    "validate_bracket", "bracket_set_first_four",
    "CURRENT_YEAR",
]
