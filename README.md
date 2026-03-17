# madlab

A numerical computing environment for March Madness. Python port of the R package [mRchmadness](https://github.com/elishayer/mRchmadness) by Eli Shayer and Scott Powers.

## What it does

1. **Scrape** season game results and the ESPN population pick distribution
2. **Model** team strength with a Bradley-Terry ridge regression on score differentials
3. **Simulate** the full 64-team bracket thousands of times
4. **Optimize** a bracket against a simulated pool using your chosen criterion (win probability, expected percentile, or expected score)
5. **Test** a bracket to see its projected win rate and percentile distribution
6. **Visualize** any bracket as a crisp vector SVG

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for environment and package management

## Installation

```bash
git clone <this-repo>
cd marchmadness
uv sync
```

## Quick start

```python
import numpy as np
from marchmadness import (
    load_bracket, load_games,
    bradley_terry, find_bracket, test_bracket, draw_bracket,
)

# Load bundled 2025 data
bracket = load_bracket("men", 2025)   # 64 team IDs in seed order
games   = load_games("men", 2025)     # ~6000 regular-season games

# Fit Bradley-Terry model -> pairwise win-probability matrix
prob_matrix = bradley_terry(games)

# Find a good bracket (optimise for win probability in a 30-person pool)
rng = np.random.default_rng(42)
my_bracket = find_bracket(
    bracket,
    prob_matrix=prob_matrix,
    league="men",
    year=2025,
    num_candidates=100,
    num_sims=1000,
    criterion="win",
    pool_size=30,
    rng=rng,
)

# Evaluate it
results = test_bracket(
    bracket, my_bracket,
    prob_matrix=prob_matrix,
    league="men", year=2025,
    pool_size=30, num_sims=1000,
    rng=rng,
)
print(f"Win probability:  {results['win'].mean()*100:.1f}%")
print(f"Mean percentile:  {results['percentile'].mean()*100:.1f}%")

# Visualize
fig = draw_bracket(bracket, my_bracket, league="men")
fig.savefig("my_bracket.png", dpi=150, bbox_inches="tight")
```

## Updating data on Selection Sunday

### Step 1 — Discover the ESPN challenge ID

ESPN's Gambit API requires a challenge ID that changes each year. The scraper tries to auto-discover it. If that fails:

1. Open `https://fantasy.espn.com/games/tournament-challenge-bracket-<year>/` in a browser
2. Open DevTools → Network tab, filter for `propositions`
3. Find the `challengeId` in the request URL (e.g., `challengeId=259`)
4. Register it once before scraping:

```python
from marchmadness import register_challenge_id
register_challenge_id(2026, "men",   259)
register_challenge_id(2026, "women", 260)
```

### Step 2 — Run `prep_data`

```python
from marchmadness import prep_data, validate_bracket, bracket_set_first_four, save_bracket

data = prep_data(year=2026, league="men", verbose=True)
bracket = data["bracket"]

# Check for unresolved First Four team IDs
missing = validate_bracket(bracket)
if missing:
    print("Missing indices:", missing)
    # Manually set First Four composite IDs
    bracket = bracket_set_first_four(bracket, 40, "21/153")   # example
    save_bracket(bracket, "men", 2026)
```

`prep_data` concurrently scrapes all ~350 team schedules using async HTTP (roughly 10× faster than sequential), so the full season ingest typically takes under a minute.

### Step 3 — Use the new data

```python
games      = load_games("men", 2026)
bracket    = load_bracket("men", 2026)
prob_matrix = bradley_terry(games)
```

## API reference

### Data

| Function | Description |
|---|---|
| `load_bracket(league, year)` | 64-team bracket as `list[str]` of ESPN team IDs |
| `load_games(league, year)` | Polars DataFrame of season game results |
| `load_pred(source, league, year)` | Round-advancement probability table (`pop`, `538`, `kenpom`) |
| `load_teams(league)` | Team metadata DataFrame (ID, name, aliases) |
| `save_bracket / save_games / save_pred` | Persist scraped data to the package data directory |

### Modelling

| Function | Description |
|---|---|
| `bradley_terry(games, mov_cap=25.0)` | Fit ridge regression; returns (n×n) win-probability matrix with `.team_ids` and `.sigma` |

**Bradley-Terry options:**
- `mov_cap` — clip score differentials to ±N points before fitting (reduces blowout noise; `None` to disable)
- `ot_margin` — value assigned to OT games instead of 0 (default `0.0` matches R behaviour)

### Simulation

```python
sim_bracket(
    bracket_empty,
    prob_matrix=None,   # use bradley_terry() output, or...
    prob_source="pop",  # "pop" | "538" | "kenpom"
    league="men",
    year=2025,
    home_teams=None,    # apply home-bias adjustment to these team names
    num_reps=1,
    rng=None,
)
# returns (63, num_reps) array of team ID strings
```

### Optimization

```python
find_bracket(
    bracket_empty,
    prob_matrix=None,
    prob_source="pop",
    pool_source="pop",        # source used to simulate your opponents
    league="men",
    year=2025,
    pool_bias=None,           # home-bias team names for pool simulation
    num_candidates=100,       # random brackets to evaluate
    num_sims=1000,            # simulations per candidate
    criterion="percentile",   # "percentile" | "score" | "win"
    pool_size=30,             # opponents in your pool (excluding you)
    bonus_round=[1,2,4,8,16,32],
    bonus_seed=[0]*16,
    bonus_combine="add",      # "add" | "multiply"
    rng=None,
)
# returns list[str] of 63 team IDs
```

### Evaluation

```python
test_bracket(
    bracket_empty, bracket_picks,
    prob_matrix=None,
    pool_source="pop",
    pool_size=30, num_sims=1000,
    bonus_round=..., bonus_seed=..., bonus_combine=...,
    rng=None,
)
# returns dict: {"score": ndarray, "percentile": ndarray, "win": ndarray}
```

### Scraping

```python
scrape_game_results(year, league="men")           # returns Polars DataFrame
scrape_population_distribution(year, league="men") # returns Polars DataFrame
prep_data(year, league="men", skip_game_results=False, verbose=True)
register_challenge_id(year, league, challenge_id)  # manual ESPN ID override
validate_bracket(bracket)                          # list of missing-ID indices
bracket_set_first_four(bracket, index, "id1/id2") # fix a First Four slot
```

### Home-team bias

```python
from marchmadness import add_home_bias
adjusted_pred = add_home_bias(["UNC", "Duke"], league="men", year=2025)
```

Adds +¾ to the log-odds of the conditional win probability for each listed team (per Null 2016 / CBS Sports). Pass via `home_teams=` in `sim_bracket` or `pool_bias=` in `find_bracket`.

### Bracket input from ESPN

If you have filled out a bracket on ESPN and want to evaluate it:

```python
from marchmadness import input_bracket_filled
# Convert 63 picks in ESPN/pool reading order -> internal seed order
my_bracket = input_bracket_filled(my_picks_in_espn_order)
```

## Web app

```bash
uv run uvicorn marchmadness.server:app --reload --port 8000
```

Then open [http://localhost:8000](http://localhost:8000).

The app auto-loads the current year's bracket, lets you tune pool size, scoring scheme, and optimization criterion, and streams live progress during the optimization. The bracket renders as a card-based SVG in the browser with hover-to-trace team paths and an animated score distribution histogram.

## Bundled data

Pre-scraped data (converted from R `.RData` to parquet/JSON) is included in the package for the following years:

| Dataset | Years |
|---|---|
| `games.men.*` | 2016–2026 (no 2020) |
| `games.women.*` | 2017, 2018, 2021, 2023–2026 |
| `bracket.men.*` | 2017–2026 |
| `bracket.women.*` | 2017–2026 |
| `pred.pop.men.*` | 2016–2018, 2021–2026 |
| `pred.pop.women.*` | 2016, 2017, 2022–2026 |
| `pred.538.men.*` | 2017–2018, 2021–2023 |
| `pred.538.women.*` | 2017, 2018, 2022, 2023 |
| `pred.kenpom.men.*` | 2018, 2019, 2021, 2022 |

## Performance notes

This Python port is substantially faster than the R original on the computationally heavy paths:

| Operation | Speedup vs R | Method |
|---|---|---|
| `find_bracket` / `test_bracket` scoring | ~50–100× | Fully vectorized NumPy (no per-simulation loop) |
| `scrape_game_results` | ~10–20× | Async concurrent HTTP (all teams in parallel) |
| `bradley_terry` | ~3–5× | `sklearn.RidgeCV` with SVD-based LOO-CV vs. 100-lambda glmnet grid |
| Data loading | ~5–10× | Parquet vs. `.RData` |

## Converting R data (advanced)

To re-convert `.RData` files from a local mRchmadness checkout:

```bash
uv run scripts/convert_rdata.py \
  --rdata-dir ../mRchmadness/data \
  --out-dir src/marchmadness/data
```

## Running tests

```bash
uv run pytest tests/ -v
```

## Credits

Original R package by [Eli Shayer](https://github.com/elishayer) and [Scott Powers](https://github.com/saberpowers). This Python port preserves the methodology and adds performance improvements and a more robust data-ingestion pipeline.
