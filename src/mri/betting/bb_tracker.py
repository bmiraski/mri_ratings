"""Two basketball betting habits, tracked in the open from the first game of the season.

*Underdog picks.* The model has the team the market makes the underdog winning
outright. The bet is that underdog, with the points.

*The fade list.* Teams that keep failing to cover: ten or more graded games this
season and covering 35% or fewer of them. The bet is their opponent. (Chicago
State was the example that started it: the market never seemed to get their
number right.)

Neither has made money in the backtest (``scripts/backtest_bb_strategies.py``,
whose numbers the betting page shows above this record). They are tracked
because they are how these games have actually been bet, and a forward record
is the honest way to find out whether that has changed.

These are rule outputs, not bets. A person reviews each one - injuries, lineups,
anything the numbers cannot see - before any money goes down, and the pages say
so beside every list. The log records every pick the rule makes whether or not it
was bet, so the record measures the rule itself, not those human decisions.

Every pick is logged the morning of the game at the line then available, graded
against the result and the closing line afterwards, and never edited. A game
both strategies like is logged under each, since each record has to stand on
its own.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

WIN_UNITS = 100 / 110            # at -110
FADE_MIN_GAMES = 10
FADE_MAX_COVER = 0.35
STRATEGIES = ("dog", "fade")


def fade_list(results: pd.DataFrame) -> dict[str, dict]:
    """Teams covering FADE_MAX_COVER or less after FADE_MIN_GAMES graded games.

    ``results`` holds this season's played games with a market line: ``home``,
    ``away``, ``market`` (the market's expected home margin) and ``actual``
    (the home margin). Pushes do not count toward either side of the record.
    """
    record: dict[str, list[float]] = {}
    for r in results.itertuples():
        margin = r.actual - r.market
        if margin == 0:
            continue
        record.setdefault(r.home, []).append(margin)
        record.setdefault(r.away, []).append(-margin)
    out = {}
    for team, margins in record.items():
        covers = sum(m > 0 for m in margins)
        if len(margins) >= FADE_MIN_GAMES and covers / len(margins) <= FADE_MAX_COVER:
            out[team] = {"covers": covers, "misses": len(margins) - covers,
                         "coverRate": round(covers / len(margins), 3),
                         "againstSpread": round(float(np.mean(margins)), 1)}
    return dict(sorted(out.items(), key=lambda kv: (kv[1]["coverRate"], kv[1]["againstSpread"])))


def dog_pick(game: dict) -> str | None:
    """The market underdog, when the model has them winning outright; else None."""
    market, predicted = game.get("market"), game.get("predicted")
    if market is None or predicted is None or market == 0 or predicted == 0:
        return None
    if np.sign(predicted) == np.sign(market):
        return None
    return "home" if market < 0 else "away"


def fade_picks(game: dict, fades: dict) -> list[tuple[str, str]]:
    """(side to bet, team being faded) for each faded team in the game - unless both are, which is no bet."""
    home_faded, away_faded = game["home"] in fades, game["away"] in fades
    if home_faded and away_faded:
        return []
    if home_faded:
        return [("away", game["home"])]
    if away_faded:
        return [("home", game["away"])]
    return []


def tags(game: dict, fades: dict) -> dict:
    """What the slate shows beside a game: which strategies it falls under."""
    out = {}
    side = dog_pick(game)
    if side:
        out["dog"] = game[side]
    faded = [team for _, team in fade_picks(game, fades)]
    if faded:
        out["fade"] = faded
    return out


def _grade(pick: dict, actual: float, close: float | None) -> dict:
    home_margin_vs_line = actual - pick["taken"]
    edge = home_margin_vs_line if pick["side"] == "home" else -home_margin_vs_line
    result = "push" if edge == 0 else "win" if edge > 0 else "loss"
    units = {"win": WIN_UNITS, "loss": -1.0, "push": 0.0}[result]
    graded = {**pick, "actual": actual, "result": result, "units": round(units, 3)}
    if close is not None:
        # Closing line value in points: how far the line moved toward the side taken after it was logged.
        graded["close"] = close
        graded["clv"] = round((close - pick["taken"]) if pick["side"] == "home" else (pick["taken"] - close), 1)
    return graded


def update(picks_path: Path, today_games: list[dict], fades: dict, finals: dict[int, float],
           closes: dict[int, float], *, logged_at: str) -> dict:
    """Log today's picks, grade anything now final, and return the whole record.

    ``today_games`` are board rows (id, home, away, predicted, market, day) for games not yet played.
    ``finals`` maps game id to the home margin; ``closes`` maps game id to the closing market line.
    """
    book = json.loads(picks_path.read_text()) if picks_path.exists() else {"picks": []}
    picks = book["picks"]
    logged = {(p["strategy"], p["game_id"], p["side"]) for p in picks}

    for game in today_games:
        if game.get("market") is None:
            continue
        base = {"game_id": game["id"], "day": game["day"], "home": game["home"], "away": game["away"],
                "predicted": game["predicted"], "taken": game["market"], "loggedAt": logged_at}
        candidates = []
        side = dog_pick(game)
        if side:
            candidates.append({**base, "strategy": "dog", "side": side, "team": game[side]})
        for side, faded in fade_picks(game, fades):
            candidates.append({**base, "strategy": "fade", "side": side, "team": game[side], "faded": faded,
                               "fadedRecord": f"{fades[faded]['covers']}-{fades[faded]['misses']}"})
        for pick in candidates:
            if (pick["strategy"], pick["game_id"], pick["side"]) not in logged:
                picks.append(pick)
                logged.add((pick["strategy"], pick["game_id"], pick["side"]))

    for i, pick in enumerate(picks):
        if "result" not in pick and pick["game_id"] in finals:
            picks[i] = _grade(pick, finals[pick["game_id"]], closes.get(pick["game_id"]))

    book["picks"] = picks
    picks_path.parent.mkdir(parents=True, exist_ok=True)
    picks_path.write_text(json.dumps(book, indent=1))
    return summary(picks)


def summary(picks: list[dict]) -> dict:
    out = {}
    for strategy in STRATEGIES:
        mine = [p for p in picks if p["strategy"] == strategy]
        graded = [p for p in mine if "result" in p]
        wins = sum(p["result"] == "win" for p in graded)
        losses = sum(p["result"] == "loss" for p in graded)
        clv = [p["clv"] for p in graded if p.get("clv") is not None]
        out[strategy] = {
            "logged": len(mine), "graded": len(graded), "wins": wins, "losses": losses,
            "pushes": len(graded) - wins - losses,
            "ats": round(wins / (wins + losses), 4) if wins + losses else None,
            "units": round(sum(p["units"] for p in graded), 2),
            "clv": round(float(np.mean(clv)), 2) if clv else None,
            "recent": sorted(graded, key=lambda p: p["day"], reverse=True)[:15],
        }
    return out


def build(season: int, board: dict, picks_path: Path, *, now: dt.datetime | None = None) -> dict:
    """The fade list, today's picks and the record, from the season's schedule and lines."""
    from ..ingest import bb_registry, cbbd
    from . import bb_lines

    now = now or dt.datetime.now(dt.timezone.utc)
    games = cbbd.games(season, completed_only=False)
    for column in ("team1", "team2"):
        games[column] = [bb_registry.resolve(n, n, season=season) for n in games[column]]
    played = games[games["played"]]
    book = bb_lines.book_lines(season)
    book = book.drop_duplicates("game_id").set_index("game_id") if not book.empty else pd.DataFrame()

    results = pd.DataFrame([
        {"home": g.team2, "away": g.team1, "actual": float(g.pts2 - g.pts1), "market": float(book.loc[g.game_id, "market"])}
        for g in played.itertuples() if not book.empty and g.game_id in book.index
        and pd.notna(book.loc[g.game_id, "market"])
    ], columns=["home", "away", "actual", "market"])
    fades = fade_list(results)

    finals = {int(g.game_id): float(g.pts2 - g.pts1) for g in played.itertuples()}
    closes = {int(i): float(v) for i, v in book["market"].items() if pd.notna(v)} if not book.empty else {}
    today = now.astimezone(_eastern()).date().isoformat()
    upcoming = [g for g in board.get("games", []) if _eastern_day(g) == today and g["id"] not in finals]
    record = update(picks_path, upcoming, fades, finals, closes, logged_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"))
    return {
        "season": season,
        "rules": {"fadeMinGames": FADE_MIN_GAMES, "fadeMaxCover": FADE_MAX_COVER},
        "fades": fades,
        "today": today,
        "record": record,
        "todayPicks": {
            "dog": [{**g, "pick": g[dog_pick(g)], "side": dog_pick(g)} for g in upcoming if dog_pick(g)],
            "fade": [{**g, "pick": g[side], "side": side, "faded": team}
                     for g in upcoming for side, team in fade_picks(g, fades)],
        },
    }


def _eastern():
    from ..export.live import EASTERN
    return EASTERN


def _eastern_day(game: dict) -> str | None:
    start = game.get("start")
    if not start:
        return game.get("day")
    return pd.Timestamp(start).tz_convert(_eastern()).date().isoformat()
