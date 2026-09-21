"""Heisman odds for the season in progress, from what the site already knows.

Nothing here is fetched that the site does not already have except the players'
season-to-date totals: three calls. The team side - ratings, records, schedule - comes
from the same payload the rankings pages are built from, and the season simulation
runs on it exactly as the simulation page's does.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..export import simdata
from ..gameday import features as gd_features
from ..ingest import cfbd, players, registry
from . import calibrate, data, final_model, forecast, funnel, project, snapshots

RATIOS = Path(__file__).resolve().parents[3] / "data" / "heisman_ratios.json"


def load_settings(path: Path = RATIOS) -> dict:
    """How the backtest chose to project (last-season prior, team link) and the finalist calibration curve."""
    saved = json.loads(path.read_text())
    return {"variant": saved.get("variant") or {}, "finalistMap": saved.get("finalistMap")}


def load_ratios(week: int, path: Path = RATIOS) -> np.ndarray:
    """History's ratios for the latest snapshot week that is not after ``week``."""
    by_week = json.loads(path.read_text())["byWeek"]
    usable = [int(w) for w in by_week if int(w) <= max(week, project.SNAPSHOT_WEEKS[0])]
    return np.asarray(by_week[str(max(usable))], dtype=float)


def state_from(payload: dict) -> pd.DataFrame:
    """The standings the candidates are judged against, from the site's own ratings."""
    frame = pd.DataFrame(payload["teams"]).set_index("team")
    state = pd.DataFrame(index=frame.index)
    state["power"] = frame["power"]
    state["resume"] = frame["resume"]
    state["power_rank"] = frame["rank"]
    state["rank"] = gd_features.committee_score_rank(frame["power"].to_numpy(), frame["resume"].to_numpy())
    state["wins"], state["losses"] = frame["wins"], frame["losses"]
    state["games"] = frame["wins"] + frame["losses"]
    return state


def current_odds(payload: dict, *, sims: int = 10_000, seed: int = 2026) -> dict | None:
    """Each candidate's chance of winning the Heisman and of reaching New York, or None if there is no model."""
    model = final_model.load_model()
    if model is None or not RATIOS.exists():
        return None
    year = int(payload["season"])
    table = players.weekly_table(year, None)
    if table.empty:
        return None
    table = table.assign(team=[registry.resolve(t, t) for t in table["team"]])
    state = state_from(payload)
    voting = data.load_voting()
    settings = load_settings()
    setting = settings["variant"]
    pool = snapshots.candidates(table, state, final_model.previous_finalists(voting, year),
                                snapshots.last_rates(data.load_players(), year) if setting.get("last") else None)

    teams = [{"team": t["team"], "power": t["power"], "conference": t["conference"]} for t in payload["teams"]]
    conference = {t["team"]: t["conference"] for t in teams}
    schedule = simdata._prepare(cfbd.games(year, completed_only=False))
    schedule, entries = simdata.championships(schedule, conference)
    runs = forecast.simulate_teams(teams, schedule, home_field=float(payload["homeField"]), sims=sims, seed=seed,
                                   championships=entries or None)
    left = forecast.remaining_games(schedule, runs["names"])
    result = forecast.odds(pool, runs, left, load_ratios(int(payload["week"])), model, seed=seed, link=float(setting.get("link") or 0.0),
                           variant={"last": setting.get("last", 0.0), "shrink": setting.get("shrink", project.SHRINK_GAMES)})
    if settings["finalistMap"]:
        result["finalist"] = calibrate.apply(settings["finalistMap"], result["finalist"])
    result = result.merge(pool[["player", "team", "pass_yds", "pass_td", "rush_yds", "rush_td", "rec_yds", "rec_td"]],
                          on=["player", "team"], how="left")
    return {"season": year, "week": int(payload["week"]), "sims": sims, "candidates": len(pool), "odds": result}


def defenders_to_watch(payload: dict, *, n: int = 3) -> list[dict]:
    """The leading defensive players on the teams voters are watching.

    Not modelled and not given odds: nobody who was mainly a defender has won since 1997, and the few who
    reached the ceremony are too few to fit. But a defender is the single likeliest source of a finalist
    the model has not heard of, so the page names the ones to keep an eye on.
    """
    year = int(payload["season"])
    rows: dict[tuple[str, str], dict] = {}
    for category in ("defensive", "interceptions"):
        for r in players.category_rows(year, category, end_week=None):
            column = players.COLUMNS.get((r["category"], r["statType"]))
            if column is None:
                continue
            row = rows.setdefault((str(r["playerId"]), r["team"]), {"player": r["player"], "team": r["team"],
                                                                     "position": r.get("position"), **{c: 0.0 for c in snapshots.DEFENSIVE}})
            try:
                row[column] = float(r["stat"])
            except (TypeError, ValueError):
                pass
    if not rows:
        return []
    frame = pd.DataFrame(list(rows.values()))
    frame["team"] = [registry.resolve(t, t) for t in frame["team"]]
    ranks = {t["team"]: t["rank"] for t in payload["teams"]}
    frame = frame[frame["team"].map(ranks).le(funnel.TEAM_RANK_CUTOFF)]
    frame["score"] = frame["def_tot"] + 4 * frame["def_sacks"] + 3 * frame["def_tfl"] + 8 * frame["def_int"] + 3 * frame["def_pd"]
    top = frame.nlargest(n, "score")
    return [{"player": r.player, "team": r.team, "position": r.position, "tackles": int(r.def_tot), "sacks": r.def_sacks,
             "interceptions": int(r.def_int)} for r in top.itertuples()]
