"""The week's slate: every game, what the model makes of it, and what rides on it.

For each game: kickoff, the model's line and win probability, the market's number
beside it, whether the disagreement is large enough for the board to flag, and
how much the result moves each team's playoff chance - the last of those comes
straight from the season simulation, which can say what a Saturday game does to
a team's December.

Games already played this week are shown with their result and the model's
pre-week call, so the page is a record as the week goes on and not just a
preview. The model's number for a played game is the one it had *entering the
week*, not the current rating, which has already learned from the result.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..betting import board as board_module
from ..betting import lines as lines_module
from ..ingest import cfbd, registry
from ..ratings import mri2, priors

SIGMA = board_module.SIGMA
WATCH = 8

try:
    from zoneinfo import ZoneInfo
    EASTERN = ZoneInfo("America/New_York")
except Exception:  # noqa: BLE001 - a runner without tz data still gets a page
    EASTERN = dt.timezone(dt.timedelta(hours=-4))


def _eastern(iso: str | None) -> dt.datetime | None:
    if not iso:
        return None
    return pd.Timestamp(iso).to_pydatetime().astimezone(EASTERN)


def _canonical(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("team1", "team2"):
        frame[column] = [registry.resolve(n, n) for n in frame[column]]
    return frame


def _swing(entry: dict | None) -> dict | None:
    """The side of the game with more to lose, as its own small record."""
    if not entry:
        return None
    best = None
    for side in ("home", "away"):
        odds = entry.get(side)
        if not odds:
            continue
        swing = odds["ifWin"] - odds["ifLose"]
        if best is None or swing > best["swing"]:
            best = {"side": side, "ifWin": odds["ifWin"], "ifLose": odds["ifLose"], "swing": swing}
    return best


def _stake(entry: dict | None, home: str, away: str, sim: dict | None) -> dict | None:
    """The team with more riding on a game: its playoff chance if it loses and if it wins, and where it stands now.

    The current chance is what makes the other two readable. "20% -> 60%" looks like a team swinging between
    two extremes; "42% now, 20% in a loss, 60% in a win" shows it is in the middle of that range.
    """
    best = _swing(entry)
    if not best:
        return None
    team = home if best["side"] == "home" else away
    out = {**{k: (round(float(v), 4) if k != "side" else v) for k, v in best.items()}, "team": team}
    now = ((sim or {}).get("teams") or {}).get(team, {}).get("playoff")
    if now is not None:
        out["now"] = round(float(now), 4)
    return out


def build(year: int, payload: dict, board: dict, sim: dict | None, weekly: pd.DataFrame, *,
          now: dt.datetime | None = None) -> dict | None:
    schedule = _canonical(cfbd.games(year, completed_only=False))
    unplayed = schedule[~schedule["played"]]
    if unplayed.empty:
        return None
    week = int(unplayed["week"].min())
    games = schedule[schedule["week"] == week].copy()

    teams = {t["team"]: t for t in payload["teams"]}
    power = {n: float(t["power"]) for n, t in teams.items()}
    replacement = min(power.values()) - 8.0
    home_field = float(payload["homeField"])

    # What the model thought entering the week, for games that are over.
    if week > 1 and not weekly[weekly["week"] == week - 1].empty:
        entering = weekly[weekly["week"] == week - 1]
        prior_power = entering.set_index("team")["power"].to_dict()
        prior_hf = float(entering["home_field"].iloc[0])
    else:
        previous = board_module._previous_season(year)
        every = sorted(set(schedule["team1"]) | set(schedule["team2"]))
        prior_power = priors.for_season(year, previous, every, list(teams)).to_dict()
        prior_hf = mri2.DEFAULT_HOME_FIELD_PRIOR

    book = lines_module.preferred_lines(year)
    book = book.drop_duplicates("game_id").set_index("game_id") if not book.empty else pd.DataFrame()
    flagged = {(g["home"], g["away"]) for g in board.get("flagged", [])}
    board_rows = {(g["home"], g["away"]): g for g in board.get("games", [])}
    leverage = (sim or {}).get("leverage", {})

    rows, fcs_rows = [], []
    for game in games.itertuples():
        home, away = game.team2, game.team1
        if home not in teams and away not in teams:
            continue
        both = home in teams and away in teams
        when = _eastern(game.start_date)
        tbd = bool(getattr(game, "start_time_tbd", False))

        if game.played:
            rating = lambda t: float(prior_power.get(t, replacement))          # noqa: E731
            edge_home = 0.0 if game.neutral else prior_hf
        else:
            rating = lambda t: float(power.get(t, replacement))                # noqa: E731
            edge_home = 0.0 if game.neutral else home_field
        predicted = rating(home) - rating(away) + edge_home

        entry = {
            "id": int(game.game_id), "week": week,
            "date": when.strftime("%Y-%m-%d") if when else None,
            "dateLabel": when.strftime("%a, %b %-d") if when else "Date to be set",
            "time": "TBD" if tbd or not when else when.strftime("%-I:%M %p").replace(" ", "\u202f") + " ET",
            "sort": when.strftime("%H%M") if when and not tbd else "9999",
            "home": home, "away": away, "neutral": bool(game.neutral),
            "predicted": round(predicted, 1),
            "homeWinProbability": round(float(norm.cdf(predicted / SIGMA)), 3),
            "played": bool(game.played),
        }
        if not both:
            fcs_rows.append(entry)
            continue

        line = book.loc[game.game_id] if isinstance(book, pd.DataFrame) and game.game_id in book.index else None
        if line is not None:
            entry["market"] = round(float(line["market"]), 1) if pd.notna(line["market"]) else None
            entry["open"] = round(float(line["market_open"]), 1) if pd.notna(line.get("market_open")) else None
            entry["total"] = float(line["total"]) if pd.notna(line.get("total")) else None
        if not game.played:
            row = board_rows.get((home, away))
            if row and row.get("edge") is not None:
                entry["edge"] = row["edge"]
                entry["flagged"] = (home, away) in flagged
            stake = _stake(leverage.get(str(int(game.game_id))), home, away, sim)
            if stake:
                entry["stake"] = stake
        else:
            actual = float(game.pts2 - game.pts1)
            entry["result"] = {
                "homeScore": int(game.pts2), "awayScore": int(game.pts1),
                "modelCorrect": bool(np.sign(predicted) == np.sign(actual)),
                "modelError": round(abs(predicted - actual), 1),
                "marketError": round(abs(entry["market"] - actual), 1) if entry.get("market") is not None else None,
            }
        rows.append(entry)

    upcoming = [g for g in rows if not g["played"]]
    done = [g for g in rows if g["played"]]

    def order(g):
        rank = lambda t: teams[t]["rank"] if t in teams else 999            # noqa: E731
        return (g["date"] or "9999", g["sort"], min(rank(g["home"]), rank(g["away"])))

    upcoming.sort(key=order)
    done.sort(key=order)
    fcs_rows.sort(key=order)

    watch = sorted(
        (g for g in upcoming if g.get("stake")),
        key=lambda g: -g["stake"]["swing"],
    )[:WATCH]

    days: list[dict] = []
    for g in upcoming:
        if not days or days[-1]["date"] != g["date"]:
            days.append({"date": g["date"], "label": g["dateLabel"], "games": []})
        days[-1]["games"].append(g)

    return {
        "week": week,
        "days": days,
        "results": done,
        "fcs": fcs_rows,
        "watch": [g["id"] for g in watch],
        "flagged": sum(bool(g.get("flagged")) for g in upcoming),
        "games": len(rows) + len(fcs_rows),
    }
