"""The model's record, in public.

Two records, and the difference between them is the point.

*The season so far, reconstructed.* For every game already played, what would
the model have said, using only the ratings that existed before that week? That
is honest about information - nothing from the week it is predicting leaks in -
but it was computed after the fact, and a record computed after the fact is a
backtest however carefully it is done. It is labelled as one.

*The forward log.* From the day it started, every game the board flags is written
down before kickoff - model line, the market's number at that moment, the side -
and never edited. Results are filled in afterwards; the pick is not. Only this
one can be called a track record, and for a long time it will be too short to
mean anything. It is published anyway, including the losing weeks, because a
record that begins on the day it is started is the only kind that cannot be
improved in hindsight.

Grading follows the betting page: the pick is a disagreement with the market,
measured against the number at the time it was made, at standard -110 prices.
Closing line value is the honest early signal - it says whether the market moved
toward the pick - and converges in a few hundred bets where win-loss needs
thousands.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..ingest import cfbd, registry
from ..ratings import mri2
from . import board as board_module
from . import lines as lines_module

WIN_PAYOUT = 100 / 110          # a -110 winner returns 0.909 units per unit staked
MIN_EDGE = board_module.MIN_EDGE_TO_SHOW
SIGMA = board_module.SIGMA

# The 2003-2019 walk-forward result MRI 2.0 was validated on, for scale.
REFERENCE = {"seasons": "2003\u20132019", "accuracy": 0.738, "mae": 13.0}


def _canonical(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("team1", "team2"):
        frame[column] = [registry.resolve(n, n) for n in frame[column]]
    return frame


def grade(side: str, home_margin: float, line: float) -> str:
    """Win, loss or push for a pick against a number.

    ``line`` is the market's expected home margin, so the home side covers when
    the home team beats it and the away side when it does not.
    """
    if home_margin == line:
        return "push"
    covered_home = home_margin > line
    return "win" if covered_home == (side == "home") else "loss"


def units(result: str) -> float:
    return WIN_PAYOUT if result == "win" else (-1.0 if result == "loss" else 0.0)


def closing_value(side: str, taken: float, closing: float) -> float:
    """Points the market moved toward the pick after it was made."""
    move = closing - taken
    return move if side == "home" else -move


def reconstruct(year: int, payload: dict, weekly: pd.DataFrame) -> dict:
    """What the model would have said before each game already played."""
    schedule = _canonical(cfbd.games(year, completed_only=False))
    played = schedule[schedule["played"]].copy()
    fbs = {t["team"] for t in payload["teams"]}
    played = played[played["team1"].isin(fbs) & played["team2"].isin(fbs)]

    book = lines_module.preferred_lines(year)
    book = book.drop_duplicates("game_id").set_index("game_id") if not book.empty else pd.DataFrame()

    previous = board_module._previous_season(year)
    known = set(previous.index) if previous is not None else set()
    every_team = sorted(set(schedule["team1"]) | set(schedule["team2"]))
    preseason = mri2.build_prior(previous, every_team, centre_teams=sorted(fbs))

    rows = []
    for row in played.itertuples():
        week = int(row.week)
        if week > 1:
            rated = weekly[weekly["week"] == week - 1]
            if rated.empty:
                continue
            power = rated.set_index("team")["power"]
            home_field = float(rated["home_field"].iloc[0])
        else:
            power, home_field = preseason, mri2.DEFAULT_HOME_FIELD_PRIOR

        predicted = float(power.get(row.team2, np.nan)) - float(power.get(row.team1, np.nan)) \
            + (0.0 if row.neutral else home_field)
        if not np.isfinite(predicted):
            continue
        actual = float(row.pts2 - row.pts1)

        before = played[played["week"] < week]
        seen = pd.concat([before["team1"], before["team2"]]).value_counts()
        thin = [t for t in (row.team1, row.team2) if t not in known and int(seen.get(t, 0)) < 4]

        market = opening = None
        if not isinstance(book, pd.DataFrame) or book.empty or row.game_id not in book.index:
            pass
        else:
            line = book.loc[row.game_id]
            market = float(line["market"]) if pd.notna(line["market"]) else None
            opening = float(line["market_open"]) if pd.notna(line.get("market_open")) else None

        rows.append({
            "week": week, "game_id": int(row.game_id), "home": row.team2, "away": row.team1,
            "predicted": predicted, "actual": actual, "market": market, "open": opening,
            "confident": not thin,
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"games": 0}

    frame["correct"] = (np.sign(frame["predicted"]) == np.sign(frame["actual"])) | (frame["actual"] == 0)
    frame["model_error"] = (frame["predicted"] - frame["actual"]).abs()
    priced = frame[frame["market"].notna()].copy()
    priced["market_error"] = (priced["market"] - priced["actual"]).abs()
    priced["market_correct"] = np.sign(priced["market"]) == np.sign(priced["actual"])

    diff = priced["model_error"] - priced["market_error"]
    summary = {
        "games": int(len(frame)),
        "accuracy": round(float(frame["correct"].mean()), 4),
        "mae": round(float(frame["model_error"].mean()), 2),
        "priced": int(len(priced)),
        "modelMaePriced": round(float(priced["model_error"].mean()), 2) if len(priced) else None,
        "marketMae": round(float(priced["market_error"].mean()), 2) if len(priced) else None,
        "marketAccuracy": round(float(priced["market_correct"].mean()), 4) if len(priced) else None,
        "modelAccuracyPriced": round(float(priced["correct"].mean()), 4) if len(priced) else None,
        "maeGap": round(float(diff.mean()), 2) if len(priced) else None,
        "maeGapError": round(float(1.96 * diff.std(ddof=1) / np.sqrt(len(priced))), 2)
        if len(priced) > 2 else None,
        # Actual margin regressed on the predicted one. 1.0 means the lines are the
        # right size; below 1.0 means they are too extreme - a favourite laying
        # more points than it wins by, on average.
        "slope": _slope(frame["predicted"], frame["actual"]),
        "marketSlope": _slope(priced["market"], priced["actual"]) if len(priced) else None,
    }

    # Disagreements with the opening number, the board's own rule.
    bets = []
    for row in priced[priced["open"].notna() & priced["confident"]].itertuples():
        edge = row.predicted - row.open
        if abs(edge) < MIN_EDGE:
            continue
        side = "home" if edge > 0 else "away"
        result = grade(side, row.actual, row.open)
        bets.append({
            "week": row.week, "home": row.home, "away": row.away, "side": side,
            "team": row.home if side == "home" else row.away,
            "predicted": round(row.predicted, 1), "open": round(row.open, 1),
            "close": round(row.market, 1), "edge": round(edge, 1),
            "actual": int(row.actual), "result": result, "units": round(units(result), 3),
            "clv": round(closing_value(side, row.open, row.market), 1),
        })
    decided = [b for b in bets if b["result"] != "push"]
    summary["bets"] = {
        "count": len(bets),
        "wins": sum(b["result"] == "win" for b in bets),
        "losses": sum(b["result"] == "loss" for b in bets),
        "pushes": sum(b["result"] == "push" for b in bets),
        "ats": round(sum(b["result"] == "win" for b in decided) / len(decided), 4) if decided else None,
        "units": round(sum(b["units"] for b in bets), 2),
        "clv": round(float(np.mean([b["clv"] for b in bets])), 2) if bets else None,
    }

    weeks = []
    for week, g in frame.groupby("week"):
        p = priced[priced["week"] == week]
        wb = [b for b in bets if b["week"] == week]
        weeks.append({
            "week": int(week), "games": int(len(g)),
            "accuracy": round(float(g["correct"].mean()), 4),
            "mae": round(float(g["model_error"].mean()), 2),
            "marketMae": round(float(p["market_error"].mean()), 2) if len(p) else None,
            "bets": len(wb),
            "record": f"{sum(b['result'] == 'win' for b in wb)}-{sum(b['result'] == 'loss' for b in wb)}"
                      + (f"-{sum(b['result'] == 'push' for b in wb)}" if any(b['result'] == 'push' for b in wb) else ""),
            "units": round(sum(b["units"] for b in wb), 2),
        })
    return {"summary": summary, "weeks": weeks, "bets": sorted(bets, key=lambda b: (b["week"], -abs(b["edge"])))}


def update_log(year: int, board: dict, path: Path, *, now: dt.datetime | None = None) -> dict:
    """Record the flagged games that have not kicked off, and grade those that have.

    A pick already in the file is never rewritten: its model line, its number and
    its side stay exactly as they were when it was written. Only the result
    fields are added, once, after the game.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    log = json.loads(path.read_text()) if path.exists() else {
        "season": year, "started": now.date().isoformat(), "picks": []}
    known = {p["game_id"] for p in log["picks"]}

    schedule = _canonical(cfbd.games(year, completed_only=False)).set_index("game_id")
    for game in board.get("flagged", []):
        gid = _game_id(schedule, game)
        if gid is None or gid in known:
            continue
        kickoff = schedule.loc[gid, "start_date"]
        if kickoff and pd.Timestamp(kickoff) <= pd.Timestamp(now):
            continue                       # a pick made after kickoff is not a pick
        taken = game["market"] if game["market"] is not None else game["marketOpen"]
        if taken is None:
            continue
        log["picks"].append({
            "game_id": int(gid), "week": game["week"], "home": game["home"], "away": game["away"],
            "neutral": game["neutral"], "kickoff": kickoff,
            "side": "home" if game["edge"] > 0 else "away",
            "predicted": game["predicted"], "open": game["marketOpen"], "taken": taken,
            "edge": game["edge"], "loggedAt": now.strftime("%Y-%m-%dT%H:%MZ"),
        })
        known.add(gid)

    book = lines_module.preferred_lines(year)
    book = book.drop_duplicates("game_id").set_index("game_id") if not book.empty else pd.DataFrame()
    for pick in log["picks"]:
        if "result" in pick:
            continue
        gid = pick["game_id"]
        if gid not in schedule.index or not bool(schedule.loc[gid, "played"]):
            continue
        actual = float(schedule.loc[gid, "pts2"] - schedule.loc[gid, "pts1"])
        pick["actual"] = int(actual)
        pick["result"] = grade(pick["side"], actual, pick["taken"])
        pick["units"] = round(units(pick["result"]), 3)
        if isinstance(book, pd.DataFrame) and gid in book.index and pd.notna(book.loc[gid, "market"]):
            pick["close"] = round(float(book.loc[gid, "market"]), 1)
            pick["clv"] = round(closing_value(pick["side"], pick["taken"], pick["close"]), 1)

    text = json.dumps(log, indent=1, sort_keys=True)
    if not path.exists() or path.read_text() != text:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return summarize(log)


def summarize(log: dict) -> dict:
    graded = [p for p in log["picks"] if "result" in p]
    decided = [p for p in graded if p["result"] != "push"]
    with_close = [p for p in graded if "clv" in p]
    return {
        "started": log["started"],
        # Keys sorted, so the output is the same whether the pick was just made
        # or read back from the file - otherwise every run re-orders the published
        # record and commits a diff that says nothing.
        "picks": [dict(sorted(p.items())) for p in
                  sorted(log["picks"], key=lambda p: (p["week"], -abs(p["edge"]), p["game_id"]))],
        "logged": len(log["picks"]),
        "graded": len(graded),
        "wins": sum(p["result"] == "win" for p in graded),
        "losses": sum(p["result"] == "loss" for p in graded),
        "pushes": sum(p["result"] == "push" for p in graded),
        "ats": round(sum(p["result"] == "win" for p in decided) / len(decided), 4) if decided else None,
        "units": round(sum(p["units"] for p in graded), 2),
        "clv": round(float(np.mean([p["clv"] for p in with_close])), 2) if with_close else None,
    }


def _slope(predicted: pd.Series, actual: pd.Series) -> float | None:
    x = predicted - predicted.mean()
    denominator = float((x * x).sum())
    return round(float((x * (actual - actual.mean())).sum() / denominator), 2) if denominator else None


def _game_id(schedule: pd.DataFrame, game: dict) -> int | None:
    match = schedule[(schedule["team2"] == game["home"]) & (schedule["team1"] == game["away"])
                     & (schedule["week"] == game["week"])]
    return int(match.index[0]) if len(match) else None


def build(year: int, payload: dict, weekly: pd.DataFrame, board: dict, log_path: Path, *,
          now: dt.datetime | None = None) -> dict:
    return {
        "reference": REFERENCE,
        "reconstructed": reconstruct(year, payload, weekly),
        "forward": update_log(year, board, log_path, now=now),
    }
