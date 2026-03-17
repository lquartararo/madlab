"""
FastAPI web server — replaces Streamlit webapp.

Run with:
    uv run uvicorn madlab.server:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import io
import json
import re
from pathlib import Path
from typing import AsyncIterator

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

try:
    from .bracket import (load_bracket, draw_bracket,
                          bracket_display_slots, picks_display_order)
    from .model import bradley_terry
    from .optimize import find_bracket
    from .evaluate import test_bracket
    from .simulate import CURRENT_YEAR
    from .scrape import prep_data
except ImportError:
    from madlab.bracket import (load_bracket, draw_bracket,
                                      bracket_display_slots, picks_display_order)
    from madlab.model import bradley_terry
    from madlab.optimize import find_bracket
    from madlab.evaluate import test_bracket
    from madlab.simulate import CURRENT_YEAR
    from madlab.scrape import prep_data

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = FastAPI(title="madlab")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"


def _fig_to_svg(fig: plt.Figure) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True)
    svg = buf.getvalue()
    svg = re.sub(r"<\?xml[^>]*\?>", "", svg)
    svg = re.sub(r"<!DOCTYPE[^>]*>", "", svg)
    svg = svg.strip()
    svg = re.sub(r'width="[^"]*"', 'width="100%"', svg, count=1)
    svg = re.sub(r'\s*height="[^"]*"', '', svg, count=1)
    return svg


def _load_games_safe(league: str, year: int):
    try:
        from madlab.bracket import load_games, DATA_DIR
        import polars as pl
        path = DATA_DIR / f"games.{league}.{year}.parquet"
        if not path.exists():
            return None
        return pl.read_parquet(path)
    except Exception:
        return None


SCORING_OPTIONS = {
    "traditional": {"bonus_round": [1, 2, 4, 8, 16, 32], "bonus_seed": [0] * 16, "bonus_combine": "add"},
    "upset":        {"bonus_round": [1, 1, 1, 1, 1,  1],  "bonus_seed": list(range(15, -1, -1)), "bonus_combine": "add"},
    "seed":         {"bonus_round": [1, 2, 4, 8, 16, 32], "bonus_seed": list(range(15, -1, -1)), "bonus_combine": "add"},
}

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    html = _TEMPLATE_PATH.read_text()
    return HTMLResponse(html)


@app.get("/api/bracket/json")
async def bracket_json(league: str = Query("men"), year: int = Query(CURRENT_YEAR)):
    def _compute():
        bracket = load_bracket(league, year)
        return bracket_display_slots(bracket, league)

    slots = await asyncio.to_thread(_compute)
    return {"league": league, "year": year, "slots": slots}


@app.get("/api/bracket/svg")
async def bracket_svg(league: str = Query("men"), year: int = Query(CURRENT_YEAR)):
    def _render():
        bracket = load_bracket(league, year)
        fig = draw_bracket(bracket, league=league)
        svg = _fig_to_svg(fig)
        plt.close(fig)
        return svg

    svg = await asyncio.to_thread(_render)
    return {"svg": svg}


class FindRequest(BaseModel):
    league: str = "men"
    year: int = CURRENT_YEAR
    pool_size: int = 30
    num_candidates: int = 500
    num_sims: int = 2000
    num_test: int = 1000
    criterion: str = "win"
    scoring: str = "traditional"


@app.post("/api/find")
async def find(req: FindRequest):
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()  # capture before entering thread

    def _on_progress(msg: str) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, msg)

    async def _run() -> None:
        try:
            def _work():
                bracket = load_bracket(req.league, req.year)
                games = _load_games_safe(req.league, req.year)
                if games is None:
                    raise RuntimeError(f"No game data for {req.league} {req.year}. Run prep_data() first.")

                scoring = SCORING_OPTIONS[req.scoring]
                rng = np.random.default_rng()

                _on_progress("Fitting Bradley-Terry model…")
                pm = bradley_terry(games)

                picks = find_bracket(
                    bracket_empty=bracket,
                    prob_matrix=pm,
                    pool_source="pop",
                    league=req.league,
                    year=req.year,
                    num_candidates=req.num_candidates,
                    num_sims=req.num_sims,
                    criterion=req.criterion,
                    pool_size=req.pool_size,
                    print_progress=False,
                    on_progress=_on_progress,
                    rng=rng,
                    **scoring,
                )

                _on_progress(f"Testing bracket ({req.num_test:,} simulations)…")
                results = test_bracket(
                    bracket_empty=bracket,
                    bracket_picks=picks,
                    prob_matrix=pm,
                    pool_source="pop",
                    league=req.league,
                    year=req.year,
                    pool_size=req.pool_size,
                    num_sims=req.num_test,
                    print_progress=False,
                    rng=rng,
                    **scoring,
                )

                _on_progress("Rendering bracket…")
                picks_data = picks_display_order(picks, bracket, req.league)

                stats = {
                    "win_prob": float(results["win"].mean() * 100),
                    "mean_percentile": float(results["percentile"].mean() * 100),
                    "mean_score": float(results["score"].mean()),
                }
                return {
                    "stats": stats,
                    "picks": picks_data,
                    "scores": results["score"].tolist(),
                    "percentiles": results["percentile"].tolist(),
                }

            result = await asyncio.to_thread(_work)
            await queue.put(json.dumps({"type": "result", **result}))
        except Exception as e:
            await queue.put(json.dumps({"type": "error", "message": str(e)}))
        finally:
            await queue.put(None)  # sentinel

    async def _stream() -> AsyncIterator[str]:
        task = asyncio.create_task(_run())
        while True:
            item = await queue.get()
            if item is None:
                break
            try:
                data = json.loads(item)
                yield f"data: {json.dumps(data)}\n\n"
            except json.JSONDecodeError:
                yield f"data: {json.dumps({'type': 'progress', 'message': item})}\n\n"
        await task

    return StreamingResponse(_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
