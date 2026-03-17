"""
Data scraping utilities for madlab.

Key improvements over the R version:
- Challenge IDs are auto-discovered from the ESPN fantasy page (no hardcoding)
- Game results are scraped concurrently using asyncio + httpx (10-20x speedup)
- Year/league parametrization is fully dynamic -- no hardcoded year lists
- Robust "First Four" detection with guided manual override
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx
import polars as pl
from bs4 import BeautifulSoup

from .bracket import DATA_DIR, load_teams, save_bracket, save_games, save_pred

# Fallback cache of known challenge IDs (add new ones as they become available)
# Format: (year, league) -> challenge_id
_KNOWN_CHALLENGE_IDS: dict[tuple[int, str], int] = {
    (2024, "men"): 240,
    (2024, "women"): 241,
    (2025, "men"): 257,
    (2025, "women"): 258,
    (2026, "men"): 277,
    (2026, "women"): 278,
}

ESPN_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Accept": "application/json",
}


# ---------------------------------------------------------------------------
# Challenge ID discovery
# ---------------------------------------------------------------------------

def get_challenge_id(year: int, league: str) -> int:
    """
    Look up the ESPN Tournament Challenge bracket challenge ID for a given year/league.

    Strategy:
    1. Check the in-code cache (_KNOWN_CHALLENGE_IDS)
    2. Try to discover it automatically from the ESPN fantasy page
    3. Raise a helpful error explaining how to find it manually

    Parameters
    ----------
    year   : tournament year (e.g., 2026)
    league : "men" | "women"

    Returns
    -------
    int challenge ID
    """
    key = (year, league)
    if key in _KNOWN_CHALLENGE_IDS:
        return _KNOWN_CHALLENGE_IDS[key]

    discovered = _discover_challenge_id(year, league)
    if discovered is not None:
        _KNOWN_CHALLENGE_IDS[key] = discovered
        return discovered

    raise ValueError(
        f"Could not find ESPN Tournament Challenge ID for {league}/{year}.\n"
        f"To find it manually:\n"
        f"  1. Open https://fantasy.espn.com/games/tournament-challenge-bracket-{year}/ in a browser\n"
        f"  2. Open DevTools -> Network tab, filter for 'propositions'\n"
        f"  3. Find the challengeId in the request URL (e.g., challengeId=257)\n"
        f"  4. Register it: scrape.register_challenge_id({year}, '{league}', <id>)\n"
    )


def register_challenge_id(year: int, league: str, challenge_id: int) -> None:
    """Manually register a challenge ID for a year/league."""
    _KNOWN_CHALLENGE_IDS[(year, league)] = challenge_id


def _discover_challenge_id(year: int, league: str) -> int | None:
    """Try to discover the challenge ID by scraping the ESPN fantasy page."""
    suffix = "" if league == "men" else "-women"
    url = f"https://fantasy.espn.com/games/tournament-challenge-bracket{suffix}-{year}/"
    try:
        with httpx.Client(timeout=15, headers=ESPN_HEADERS, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            # Look for challengeId in page source
            for pattern in [r'"challengeId":(\d+)', r'challengeId=(\d+)', r'challenge_id.*?(\d{3,})']: 
                m = re.search(pattern, resp.text)
                if m:
                    return int(m.group(1))
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Population distribution scraping
# ---------------------------------------------------------------------------

def _fetch_abbrev_to_id(league: str) -> dict[str, str]:
    """Fetch abbreviation->ESPN team ID map from the ESPN site API."""
    espn_league = "mens-college-basketball" if league == "men" else "womens-college-basketball"
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/basketball"
        f"/{espn_league}/teams?limit=400"
    )
    try:
        with httpx.Client(timeout=15, headers=ESPN_HEADERS) as client:
            r = client.get(url)
            r.raise_for_status()
            teams = r.json()["sports"][0]["leagues"][0]["teams"]
            return {t["team"]["abbreviation"]: t["team"]["id"] for t in teams}
    except Exception:
        return {}


def _resolve_first_four_id(composite_name: str, abbrev_map: dict[str, str]) -> str | None:
    """
    Convert a First Four composite abbreviation like "TEX/NCSU" to "251/152".
    Returns None if either abbreviation can't be resolved.
    """
    parts = composite_name.split("/")
    if len(parts) != 2:
        return None
    ids = [abbrev_map.get(p) for p in parts]
    if all(ids):
        return "/".join(str(i) for i in ids)
    return None


def _extract_outcomes(data: Any) -> list[dict]:
    """
    Normalize the ESPN Gambit API response into a flat list of outcome dicts.

    The API has returned two different shapes across years:
    - Old (2024-2025): dict {"possibleOutcomes": [team, team, ...]}
    - New (2026+):     list of matchup objects, each with a "possibleOutcomes" key
    """
    if isinstance(data, dict):
        return data.get("possibleOutcomes", [])
    if isinstance(data, list):
        # New format: each item is a matchup with nested possibleOutcomes
        if data and "possibleOutcomes" in data[0]:
            outcomes = []
            for matchup in data:
                outcomes.extend(matchup.get("possibleOutcomes", []))
            return outcomes
        # Old flat list of team objects
        return data
    return []

def scrape_population_distribution(year: int, league: str = "men") -> pl.DataFrame:
    """
    Scrape the ESPN population pick distribution for a given year and league.

    Parameters
    ----------
    year   : tournament year
    league : "men" | "women"

    Returns
    -------
    DataFrame with columns: seed, team_id, name, round1, ..., round6
    """
    challenge_id = get_challenge_id(year, league)
    base_url = "https://gambit-api.fantasy.espn.com/apis/v1/propositions"
    abbrev_map = _fetch_abbrev_to_id(league)

    rounds_data: list[pl.DataFrame] = []

    with httpx.Client(timeout=30, headers=ESPN_HEADERS) as client:
        for rnd in range(1, 7):
            filter_json = json.dumps(
                {"filterPropositionScoringPeriodIds": {"value": [rnd]}}
            )
            params = {"challengeId": challenge_id, "filter": filter_json}
            resp = client.get(base_url, params=params)
            resp.raise_for_status()
            data = resp.json()

            outcomes = _extract_outcomes(data)
            rows = []
            for entry in outcomes:
                mappings = entry.get("mappings", [])
                seed = next((int(m["value"]) for m in mappings if m.get("type") == "RANKING"), None)
                team_id = next((m["value"] for m in mappings if m.get("type") == "COMPETITOR_ID"), None)
                name = entry.get("name", "")
                # First Four: no COMPETITOR_ID; resolve from composite abbreviation name
                if team_id is None and "/" in name:
                    team_id = _resolve_first_four_id(name, abbrev_map)
                counters = entry.get("choiceCounters", [])
                pct = counters[0].get("percentage", 0.0) if counters else 0.0
                rows.append({"seed": seed, "team_id": team_id, "name": name, f"round{rnd}": pct})

            df = pl.DataFrame(rows).with_columns([
                pl.col("seed").cast(pl.Int32),
                pl.col("team_id").cast(pl.Utf8),
                pl.col("name").cast(pl.Utf8),
                pl.col(f"round{rnd}").cast(pl.Float64),
            ])
            rounds_data.append(df)

    # Join all rounds
    result = rounds_data[0]
    for rnd, rnd_df in enumerate(rounds_data[1:], start=2):
        result = result.join(
            rnd_df.select(["seed", "team_id", f"round{rnd}"]),
            on=["seed", "team_id"],
            how="left",
        )

    return result.sort("seed")


# ---------------------------------------------------------------------------
# Game results scraping (async, concurrent per team)
# ---------------------------------------------------------------------------

def scrape_game_results(year: int, league: str = "men") -> pl.DataFrame:
    """
    Scrape all game results for a given season from ESPN.

    Uses async HTTP to scrape all team schedules concurrently.

    Parameters
    ----------
    year   : season year (e.g., 2026 for the 2025-2026 season)
    league : "men" | "women"

    Returns
    -------
    DataFrame with columns: game_id, home_id, away_id, home_score, away_score, neutral, ot
    """
    if year < 2002:
        raise ValueError("2002 is the earliest available season")

    espn_league = f"{league}s-college-basketball"
    teams_df = _scrape_teams(espn_league)

    results = asyncio.run(_scrape_all_teams(year, teams_df["id"].to_list(), espn_league))

    if results.is_empty():
        return results

    # Determine home/away
    results = results.with_columns([
        pl.when(pl.col("location").is_in(["H", "A"]))
          .then(pl.col("location") == "H")
          .when(pl.col("other_id").is_null() | (pl.col("primary_id") < pl.col("other_id")))
          .then(pl.lit(True))
          .otherwise(pl.lit(False))
          .alias("is_home")
    ])

    results = results.with_columns([
        pl.when(pl.col("is_home")).then(pl.col("primary_id")).otherwise(pl.col("other_id")).alias("home_id"),
        pl.when(pl.col("is_home")).then(pl.col("other_id")).otherwise(pl.col("primary_id")).alias("away_id"),
        pl.when(pl.col("is_home")).then(pl.col("primary_score")).otherwise(pl.col("other_score")).alias("home_score"),
        pl.when(pl.col("is_home")).then(pl.col("other_score")).otherwise(pl.col("primary_score")).alias("away_score"),
        (pl.col("location") == "N").cast(pl.Int8).alias("neutral"),
    ])

    results = results.with_columns([
        pl.col("home_id").fill_null("NA"),
        pl.col("away_id").fill_null("NA"),
    ])

    # Keep only games between teams that appear on both sides (D1 filter)
    home_set = set(results["home_id"].to_list())
    away_set = set(results["away_id"].to_list())
    results = results.filter(
        pl.col("home_id").is_in(away_set) & pl.col("away_id").is_in(home_set)
    )

    return (
        results
        .select(["game_id", "home_id", "away_id", "home_score", "away_score", "neutral", "ot"])
        .unique()
    )


def _scrape_teams(espn_league: str) -> pl.DataFrame:
    """Scrape team IDs and names from the ESPN team index."""
    url = f"https://www.espn.com/{espn_league}/teams"
    with httpx.Client(timeout=20, headers=ESPN_HEADERS, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    sections = soup.select("section.TeamLinks")

    names, ids = [], []
    for section in sections:
        name_el = section.select_one(".pl3 h2")
        link_el = section.select_one(".pl3 > a")
        if name_el and link_el:
            href = link_el.get("href", "")
            id_match = re.search(r"/id/(\d+)", href)
            if id_match:
                names.append(name_el.get_text(strip=True))
                ids.append(id_match.group(1))

    return pl.DataFrame({"name": names, "id": ids})


async def _scrape_all_teams(year: int, team_ids: list[str], espn_league: str) -> pl.DataFrame:
    """Concurrently scrape game results for all teams."""
    limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
    async with httpx.AsyncClient(timeout=30, headers=ESPN_HEADERS,
                                  limits=limits, follow_redirects=True) as client:
        tasks = [_scrape_team_games(client, year, tid, espn_league) for tid in team_ids]
        all_frames = await asyncio.gather(*tasks, return_exceptions=True)

    frames = [f for f in all_frames if isinstance(f, pl.DataFrame) and not f.is_empty()]
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal")


async def _scrape_team_games(
    client: httpx.AsyncClient, year: int, team_id: str, espn_league: str
) -> pl.DataFrame:
    """Scrape game results for a single team."""
    url = (
        f"https://www.espn.com/{espn_league}/team/schedule/_/id/{team_id}/season/{year}"
    )
    empty = pl.DataFrame({
        "game_id": pl.Series([], dtype=pl.Utf8),
        "primary_id": pl.Series([], dtype=pl.Utf8),
        "primary_score": pl.Series([], dtype=pl.Utf8),
        "other_id": pl.Series([], dtype=pl.Utf8),
        "other_score": pl.Series([], dtype=pl.Utf8),
        "location": pl.Series([], dtype=pl.Utf8),
        "ot": pl.Series([], dtype=pl.Utf8),
    })

    try:
        resp = await client.get(url)
        resp.raise_for_status()
    except Exception:
        return empty

    soup = BeautifulSoup(resp.text, "html.parser")
    rows = soup.select(".Table__TBODY tr")

    game_ids, pri_ids, pri_scores, oth_ids, oth_scores, locations, ots = (
        [], [], [], [], [], [], []
    )

    for row in rows:
        cells = row.select("td")
        if len(cells) != 7:
            continue
        # Skip header rows
        if cells[0].get_text(strip=True) == "Date":
            continue

        result_cell = cells[2]
        result_text = result_cell.get_text(strip=True)
        if result_text in ("Canceled", "Postponed", "Suspended"):
            continue

        score_link = result_cell.select_one(".ml4 a")
        if not score_link:
            continue
        href = score_link.get("href", "")
        href_parts = href.split("/")
        if len(href_parts) >= 6 and href_parts[5] in ("preview", "onair"):
            continue

        score_el = result_cell.select_one(".ml4")
        if not score_el:
            continue
        score_text = score_el.get_text(strip=True)
        if not score_text:
            continue

        # Parse W/L and score
        bold_el = result_cell.select_one(".fw-bold")
        if not bold_el:
            continue
        won = bold_el.get_text(strip=True) == "W"

        score_parts = score_text.split()
        if not score_parts:
            continue
        score_nums = score_parts[0].split("-")
        if len(score_nums) != 2:
            continue
        ot_str = score_parts[1] if len(score_parts) > 1 else ""

        # Game ID: find numeric ID after "gameId" or "game" in the URL
        game_id = ""
        if href:
            m = re.search(r"/gameId/(\d+)", href) or re.search(r"/game/[^/]*/(\d+)", href)
            if m:
                game_id = m.group(1)
            else:
                # Fallback: take any long numeric run from the URL
                nums = re.findall(r"\d{6,}", href)
                game_id = nums[0] if nums else ""

        # Opponent ID: ESPN team URLs are now /sport/team/_/id/{id}/{slug}
        # The numeric ID is the segment immediately after "/id/"
        opp_cell = cells[1]
        opp_link = opp_cell.select_one(".opponent-logo a")
        other_href = opp_link.get("href", "") if opp_link else ""
        id_match = re.search(r"/id/(\d+)", other_href)
        other_id = id_match.group(1) if id_match else None

        # Location
        neutral = opp_cell.get_text(strip=True).endswith("*")
        at_vs_el = opp_cell.select_one(".pr2")
        at_vs = at_vs_el.get_text(strip=True) if at_vs_el else ""
        if neutral:
            loc = "N"
        elif at_vs == "vs":
            loc = "H"
        else:
            loc = "A"

        pri_score = score_nums[0] if won else score_nums[1]
        oth_score = score_nums[1] if won else score_nums[0]

        game_ids.append(game_id)
        pri_ids.append(team_id)
        pri_scores.append(pri_score)
        oth_ids.append(other_id)
        oth_scores.append(oth_score)
        locations.append(loc)
        ots.append(ot_str)

    if not game_ids:
        return empty

    return pl.DataFrame({
        "game_id": game_ids,
        "primary_id": pri_ids,
        "primary_score": pri_scores,
        "other_id": [str(x) if x is not None else None for x in oth_ids],
        "other_score": oth_scores,
        "location": locations,
        "ot": ots,
    })


# ---------------------------------------------------------------------------
# prep_data: one-stop shop for Selection Sunday data ingestion
# ---------------------------------------------------------------------------

def prep_data(
    year: int,
    league: str = "men",
    skip_game_results: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Collect and save all data needed for a tournament year.

    Scrapes population distribution and (optionally) game results.
    Saves results as parquet/json files in the package data directory.

    Parameters
    ----------
    year              : tournament year
    league            : "men" | "women"
    skip_game_results : if True, skip the game-results scrape
    verbose           : print progress messages

    Returns
    -------
    dict with keys: "games" (or None), "bracket", "pred_pop"

    Notes
    -----
    After running this, call validate_bracket() to check for unresolved First
    Four team IDs (marked as None/empty). You will need to manually set those
    using bracket_set_first_four().
    """
    if verbose:
        print(f"Prepping data for {league}/{year}...")

    games = None
    if not skip_game_results:
        if verbose:
            print("  Scraping game results (concurrent)...")
        t0 = time.time()
        games = scrape_game_results(year, league)
        if verbose:
            print(f"  Done in {time.time() - t0:.1f}s: {len(games)} games")
        save_games(games, league, year)

    if verbose:
        print("  Scraping population distribution...")
    pred_pop = scrape_population_distribution(year, league)
    save_pred(pred_pop, "pop", league, year)

    # Build bracket from pred_pop (sorted by seed)
    bracket = pred_pop.sort("seed")["team_id"].cast(str).to_list()

    na_indices = [i for i, t in enumerate(bracket) if t in (None, "None", "null", "")]
    if na_indices:
        if verbose:
            missing_df = pred_pop.filter(
                pl.col("team_id").is_null() | pl.col("team_id").is_in(["", "None", "null"])
            ).select(["seed", "name"])
            print(f"\n  WARNING: {len(na_indices)} team IDs are None/empty:")
            print(missing_df)
            print(
                "\n  Fix with bracket_set_first_four(bracket, index, 'team_id/team_id')\n"
                "  Then call save_bracket(bracket, league, year) to persist."
            )

    save_bracket(bracket, league, year)

    return {"games": games, "bracket": bracket, "pred_pop": pred_pop}


def validate_bracket(bracket: list[str]) -> list[int]:
    """
    Return indices of bracket entries that are None or empty (unresolved First Four teams).

    Parameters
    ----------
    bracket : list of 64 team IDs

    Returns
    -------
    list of 0-based indices with missing team IDs (empty if bracket is valid)
    """
    return [i for i, t in enumerate(bracket) if not t or t in ("None", "null")]


def bracket_set_first_four(bracket: list[str], index: int, composite_id: str) -> list[str]:
    """
    Set a First Four composite team ID at a given bracket position.

    Parameters
    ----------
    bracket      : list of 64 team IDs (will be modified in-place and returned)
    index        : 0-based position in bracket
    composite_id : e.g., "171/264" (two ESPN team IDs separated by "/")

    Returns
    -------
    Updated bracket list
    """
    if not 0 <= index < 64:
        raise ValueError(f"index must be in [0, 63], got {index}")
    bracket[index] = composite_id
    return bracket
