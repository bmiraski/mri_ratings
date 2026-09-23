"""The rooting guide: this week's games a team is not playing in that move its playoff odds most, and who to cheer.

A team page already says what its own games are worth. This is everyone else's games: for each one still to kick
off this week, the season simulation is run twice more on the baseline's seed - once with the home side forced to
win, once with the visitor - and every other team's playoff, conference-title and bye odds are compared between
the two. Common random numbers: the two runs share every draw but that one game's sign, so the difference is that
game's effect and not noise, and not the leakage of reading it off the baseline, where the runs in which an
underdog won are also the runs in which it was drawn stronger for the rest of its season.

About two simulations per game - some 140 on a busy Saturday, half a second each.

Written to ``site/data/rooting.json`` and kept by week in ``rooting_history.json``, for a later "did it pay off".
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import pandas as pd

from ..sim import season
from . import simdata

TOP = 5
MIN_DELTA = 0.005
BAND = (0.005, 0.995)
SIMS = season.DEFAULT_SIMS
KEYS = (("playoff", "playoff"), ("conferenceTitle", "confTitle"), ("bye", "bye"))


def _odds(result: dict) -> dict[str, dict]:
    return {n: {k: o[k] for k, _ in KEYS} for n, o in result["teams"].items()}


def guide(games: list[dict], home_runs: dict[int, dict], away_runs: dict[int, dict], baseline: dict,
          win_prob: dict[int, float]) -> dict[str, dict]:
    """Each team's top games from the forced pairs. ``games`` carry ``gameId``, ``home``, ``away``, ``kickoff``."""
    out = {}
    for team, base in baseline.items():
        rows = []
        for g in games:
            if team in (g["home"], g["away"]):
                continue                                  # its own games are on its page already
            if_home, if_away = home_runs[g["gameId"]][team], away_runs[g["gameId"]][team]
            delta = {short: if_home[key] - if_away[key] for key, short in KEYS}
            rows.append((g, delta, if_home, if_away))

        def entry(g, delta, if_home, if_away, key, short):
            root_home = delta[short] > 0
            side, p_side = (g["home"], win_prob[g["gameId"]]) if root_home else (g["away"], 1 - win_prob[g["gameId"]])
            good, bad = (if_home, if_away) if root_home else (if_away, if_home)
            return {"gameId": g["gameId"], "home": g["home"], "away": g["away"], "kickoff": g["kickoff"],
                    "neutral": g["neutral"], "rootFor": side, "rootForWinProb": round(p_side, 3),
                    "upsetNeeded": p_side < 0.5,
                    "ifRoot": round(good[key], 4), "ifNot": round(bad[key], 4), "delta": round(abs(delta[short]), 4),
                    "playoffDelta": round(abs(delta["playoff"]), 4), "confTitleDelta": round(abs(delta["confTitle"]), 4),
                    "byeDelta": round(abs(delta["bye"]), 4)}

        in_band = BAND[0] <= base["playoff"] <= BAND[1]
        chosen, basis = [], None
        if in_band:
            ranked = sorted(rows, key=lambda r: -abs(r[1]["playoff"]))
            chosen = [entry(*r, "playoff", "playoff") for r in ranked if abs(r[1]["playoff"]) >= MIN_DELTA][:TOP]
            basis = "playoff" if chosen else None
        if in_band and not chosen:
            # Mid-pack in October nothing else moves a playoff chance, but a conference race can still turn on
            # someone else's Saturday.
            ranked = sorted(rows, key=lambda r: -abs(r[1]["confTitle"]))
            chosen = [entry(*r, "conferenceTitle", "confTitle") for r in ranked if abs(r[1]["confTitle"]) >= MIN_DELTA][:TOP]
            basis = "conferenceTitle" if chosen else None
        if base["playoff"] > BAND[1]:
            message = "Your fate is in your own hands this week."
        elif not chosen:
            message = "Nothing this weekend moves the needle."
        else:
            message = None
        out[team] = {"playoff": base["playoff"], "conferenceTitle": base["conferenceTitle"], "basis": basis,
                     "games": chosen if basis else [], "message": message}
    return out


def build(year: int, payload: dict, out_path: Path, history_path: Path, *, now: dt.datetime | None = None,
          sims: int = SIMS, schedule: pd.DataFrame | None = None) -> dict | None:
    """Run the forced pairs for this week's games not yet kicked off; write the guide and its history."""
    from ..ingest import cfbd

    started = time.monotonic()
    now = now or dt.datetime.now(dt.timezone.utc)
    teams = [{"team": t["team"], "power": t["power"], "conference": t["conference"]} for t in payload["teams"]]
    conference = {t["team"]: t["conference"] for t in teams}
    full = simdata._prepare(schedule if schedule is not None else cfbd.games(year, completed_only=False))
    regular = full[full["season_type"] == "regular"] if "season_type" in full else full
    unplayed = regular[~regular["played"]]
    if unplayed.empty:
        return None                 # the field is set: nothing left to root for, as the simulation page goes quiet
    week = int(unplayed["week"].min())
    sched, title_games = simdata.championships(full, conference)
    kick = pd.to_datetime(unplayed["start_date"], utc=True, errors="coerce")
    this_week = unplayed[(unplayed["week"] == week) & (kick > pd.Timestamp(now))
                         & (unplayed["team1"].isin(conference) | unplayed["team2"].isin(conference))]
    if this_week.empty:
        return None

    home_field = float(payload["homeField"])
    power = {t["team"]: float(t["power"]) for t in teams}
    floor = min(power.values()) - 8.0
    run = lambda forced=None: season.simulate(teams, sched, home_field=home_field, sims=sims,           # noqa: E731
                                              championships=title_games or None, forced=forced)
    baseline = _odds(run())
    games, home_runs, away_runs, win_prob = [], {}, {}, {}
    for g in this_week.itertuples():
        gid = int(g.game_id)
        games.append({"gameId": gid, "home": g.team2, "away": g.team1, "neutral": bool(g.neutral),
                      "kickoff": g.start_date})
        home_runs[gid] = _odds(run({gid: "home"}))
        away_runs[gid] = _odds(run({gid: "away"}))
        margin = power.get(g.team2, floor) - power.get(g.team1, floor) + (0.0 if g.neutral else home_field)
        win_prob[gid] = _win_prob(margin)
    teams_out = guide(games, home_runs, away_runs, baseline, win_prob)
    elapsed = time.monotonic() - started
    result = {"season": year, "week": week, "generated": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "sims": sims,
              "games": len(games), "seconds": round(elapsed, 1), "teams": teams_out}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1))
    history = json.loads(history_path.read_text()) if history_path.exists() else {}
    history.setdefault(str(year), {})[str(week)] = {"generated": result["generated"], "teams": {
        n: t["games"] for n, t in teams_out.items() if t["games"]}}
    history_path.write_text(json.dumps(history, indent=1))
    return result


def _win_prob(margin: float) -> float:
    """The home side's chance, on the betting board's spread - the number every other page shows beside a game."""
    from scipy.stats import norm

    from ..betting import board
    return float(norm.cdf(margin / board.SIGMA))
