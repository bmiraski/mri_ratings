"""Today's basketball slate: every Division I game, what the model makes of it, and what rides on it.

Basketball plays nightly, so the slate is a day rather than football's week: the
page is today's games, with tomorrow's alongside, rebuilt every morning. The
morning build first saves yesterday's page with its finals (``snapshot``).

What rides on a game depends on the calendar. From Christmas, when bracketology
starts, it is the NCAA Tournament: each side's chance of making the field if it
wins and if it loses, read off the joint simulation, and the key games are the
ones that move a bid most. Before that there is nothing to measure it against,
and the key games are simply the best ones - two good teams, a close line.

The games each betting habit would take are tagged here too (``bb_tracker``), so
the slate and the betting page agree about them.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from ..betting import bb_tracker
from .live import EASTERN

WATCH = 10
KEY_SWING = 0.05
BEST = 8


def _eastern(start: str) -> dt.datetime:
    return pd.Timestamp(start).tz_convert(EASTERN).to_pydatetime()


def _stake(entry: dict | None, game: dict, bracket: dict | None) -> dict | None:
    """The side with more riding on the game, in football's shape: its bid chance with a loss, now, and with a win."""
    if not entry:
        return None
    now = {t["team"]: t["pField"] for t in (bracket or {}).get("teams", [])}
    best = None
    for side in ("home", "away"):
        odds = entry.get(side)
        if not odds:
            continue
        swing = odds["ifWin"] - odds["ifLose"]
        if best is None or swing > best["swing"]:
            best = {"side": side, "team": game[side], "ifWin": odds["ifWin"], "ifLose": odds["ifLose"],
                    "swing": round(swing, 4)}
    if best and best["team"] in now:
        best["now"] = now[best["team"]]
    return best


def _quality(game: dict, power: dict) -> float:
    """How good a game should be: the weaker side's rating, less a little for every point of expected margin."""
    return min(power.get(game["home"], -50.0), power.get(game["away"], -50.0)) - 0.6 * abs(game["predicted"])


def build(season: int, payload: dict, board: dict, tracker: dict | None, *,
          now: dt.datetime | None = None) -> dict | None:
    now = now or dt.datetime.now(dt.timezone.utc)
    today = now.astimezone(EASTERN).date()
    teams = {t["team"]: t for t in payload["teams"]}
    power = {n: float(t["power"]) for n, t in teams.items()}
    bracket = payload.get("bracketology")
    live_bracket = bracket if bracket and not bracket.get("frozen") else None
    leverage = (live_bracket or {}).get("leverage") or {}
    fades = (tracker or {}).get("fades") or {}

    days: dict[dt.date, list[dict]] = {}
    for g in board.get("games", []):
        if not g.get("start"):
            continue
        when = _eastern(g["start"])
        if when.date() not in (today, today + dt.timedelta(days=1)):
            continue
        entry = {
            "id": g["id"], "date": when.date().isoformat(), "dateLabel": when.strftime("%a, %b %-d"),
            "time": when.strftime("%-I:%M %p").replace(" ", " ") + " ET", "sort": when.strftime("%H%M"),
            "home": g["home"], "away": g["away"], "neutral": g["neutral"],
            "conferences": sorted({teams.get(g["home"], {}).get("conference"), teams.get(g["away"], {}).get("conference")} - {None}),
            "predicted": g["predicted"], "homeWinProbability": g["winProbability"],
            "market": g.get("market"), "open": g.get("marketOpen"), "edge": g.get("edge"), "played": False,
            "tracked": bb_tracker.tags(g, fades),
        }
        stake = _stake(leverage.get(str(g["id"])), entry, live_bracket)
        if stake:
            entry["stake"] = stake
        days.setdefault(when.date(), []).append(entry)
    if not days.get(today) and not days.get(today + dt.timedelta(days=1)):
        return None

    rank = lambda t: teams.get(t, {}).get("rank", 999)                         # noqa: E731
    for games in days.values():
        games.sort(key=lambda g: (g["sort"], min(rank(g["home"]), rank(g["away"]))))
    todays = days.get(today, [])

    staked = sorted((g for g in todays if (g.get("stake") or {}).get("swing", 0) >= KEY_SWING),
                    key=lambda g: -g["stake"]["swing"])[:WATCH]
    if staked:
        watch, watch_kind = [g["id"] for g in staked], "stakes"
    else:
        watch = [g["id"] for g in sorted(todays, key=lambda g: -_quality(g, power))[:BEST]]
        watch_kind = "best"

    label = lambda d: d.strftime("%a, %b %-d")                                 # noqa: E731
    return {
        "season": season,
        "date": today.isoformat(),
        "dateLabel": today.strftime("%A, %B %-d"),
        "days": [{"date": d.isoformat(), "label": label(d), "games": days[d]} for d in sorted(days)],
        "watch": watch,
        "watchKind": watch_kind,
        "results": [],
        "fcs": [],
        "games": len(todays),
        "fades": fades,
    }


def archive_path(root: Path, day: str) -> Path:
    return root / "slate" / f"{day}.json"


def snapshot(root: Path, new_slate: dict | None, finals_for_day) -> Path | None:
    """Save the outgoing day's slate with its finals when this build is about to replace it.

    ``root`` is docs/basketball. ``finals_for_day(date)`` gives ``{game_id: (home_points, away_points)}``.
    Only today's games are kept: tomorrow's were a preview and will be on tomorrow's page.
    """
    from .slatearchive import freeze

    path = root / "slate.json"
    if not path.exists():
        return None
    old = json.loads(path.read_text())
    if not old.get("date") or (new_slate and new_slate.get("date") == old["date"]):
        return None
    out = archive_path(root, old["date"])
    if out.exists():
        return None
    old = {**old, "days": [d for d in old["days"] if d["date"] == old["date"]]}
    site_path = root / "basketball.json"
    site = json.loads(site_path.read_text()) if site_path.exists() else {}
    record = freeze(old, site, finals_for_day(old["date"]),
                    saved=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    return out
