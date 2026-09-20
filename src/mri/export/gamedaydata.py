"""The College GameDay page's data: what is confirmed, and what the model expects.

Two runs of the same forecast, kept apart on purpose. The weeks ESPN has not
announced are the forecast. The weeks it has announced but that have not been
played are scored against what the model would have said, which is the one honest
in-season test this page gets: those announcements were not used to fit anything.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from ..gameday import forecast, history
from ..ingest import cfbd, registry
from ..sim import season

# How many games to list for a week. Always at least MIN_SHOWN; past that, as many as it takes to
# account for TARGET of the probability, up to MAX_SHOWN. A week with one obvious game lists six;
# a week where the chances are spread thinly (Rivalry Week) lists more, and says how many it left out.
MIN_SHOWN = 6
MAX_SHOWN = 15
TARGET = 0.85


def shown(entries: list[dict]) -> list[dict]:
    """The games worth listing, most likely first."""
    ordered = sorted(entries, key=lambda e: -e["probability"])
    total, count = 0.0, 0
    for e in ordered:
        if count >= MAX_SHOWN or (count >= MIN_SHOWN and total >= TARGET):
            break
        total += e["probability"]
        count += 1
    return ordered[:count]


def _prepare(schedule: pd.DataFrame) -> pd.DataFrame:
    frame = schedule.copy()
    for column in ("team1", "team2"):
        frame[column] = [registry.resolve(n, n) for n in frame[column]]
    if "conference_game" not in frame.columns:
        frame["conference_game"] = frame["conf1"] == frame["conf2"]
    return frame


def _week_dates(schedule: pd.DataFrame) -> dict[int, dt.date]:
    """The Saturday of each week, from when its games are played."""
    starts = pd.to_datetime(schedule["start_date"], utc=True).dt.tz_convert("America/New_York")
    median = starts.groupby(schedule["week"]).median()
    return {int(w): d.date() for w, d in median.items()}


def _context(year: int, data: dict, names: set[str]) -> dict:
    """What the model knows about each team before the season: last year's rank, and its GameDay brand."""
    from . import sitedata  # noqa: PLC0415

    previous = sitedata._prior_for(year)
    last_rank = {}
    if previous is not None:
        fbs = previous[previous.index.isin(names)]
        last_rank = {t: float(r) for t, r in fbs.rank(ascending=False).items()}
    brand = {registry.resolve(t, t): v for t, v in history.appearance_rates(data, year).items()}
    return {"lastRank": last_rank, "brand": brand}


def _label(day: dt.date) -> str:
    return day.strftime("%b %-d")


def build(year: int, payload: dict, *, sims: int = forecast.DEFAULT_SIMS) -> dict | None:
    model = forecast.load_model()
    if model is None:
        return None
    data = history.load()
    schedule = _prepare(cfbd.games(year, completed_only=False))
    teams = [{"team": t["team"], "power": t["power"], "conference": t["conference"]} for t in payload["teams"]]
    names = {t["team"] for t in teams}

    announced = []
    for a in data.get("announced2026", []):
        announced.append({**a, "teams": [registry.resolve(n, n) for n in a["teams"]],
                          "host": registry.resolve(a["host"], a["host"]) if a.get("host") else None})
    done = {a["week"] for a in announced}
    unplayed = schedule[~schedule["played"]]
    if unplayed.empty:
        return None
    first_open = int(unplayed["week"].min())
    last = season.CHAMPIONSHIP_WEEK
    dates = _week_dates(schedule)
    dates.setdefault(last, dates.get(last - 1, dates[max(dates)]) + dt.timedelta(days=7))

    ahead = [w for w in range(max(done, default=0) + 1, last + 1)]
    checked = [w for w in sorted(done) if w >= first_open]
    home_field = float(payload["homeField"])
    context = _context(year, data, names)
    forecast_ahead = forecast.forecast(teams, schedule, home_field=home_field, model=model, weeks=ahead, sims=sims, context=context)
    forecast_check = (forecast.forecast(teams, schedule, home_field=home_field, model=model, weeks=checked, sims=sims, context=context)
                      if checked else {"weeks": {}})

    def entry(e: dict, w: int) -> dict:
        if e["kind"] == "championship":
            return {"kind": "championship", "conference": e["conference"], "probability": e["probability"],
                    "rankHome": e["expectedRankHome"], "rankAway": e["expectedRankAway"],
                    "bothTop10": e["bothTop10"], "bothUnbeaten": e["bothUnbeaten"]}
        venue = schedule.loc[schedule["game_id"] == e["gameId"], "venue"]
        return {"kind": "game", "home": e["home"], "away": e["away"], "neutral": e["neutral"],
                "venue": (venue.iloc[0] if len(venue) else None), "probability": e["probability"],
                "rankHome": e["expectedRankHome"], "rankAway": e["expectedRankAway"],
                "bothTop10": e["bothTop10"], "bothUnbeaten": e["bothUnbeaten"]}

    weeks = []
    for w in ahead:
        block = forecast_ahead["weeks"].get(w)
        if not block:
            continue
        listed = shown(block["games"])
        weeks.append({"week": w, "date": _label(dates[w]), "iso": dates[w].isoformat(),
                      "championship": w == last, "other": block["other"],
                      "games": [entry(e, w) for e in listed],
                      "covered": round(sum(e["probability"] for e in listed), 4),
                      "omitted": len(block["games"]) - len(listed)})

    check = []
    for a in announced:
        w = a["week"]
        block = forecast_check["weeks"].get(w)
        if not block:
            continue
        pair = set(a["teams"])
        ranked = sorted(block["games"], key=lambda e: -e["probability"])
        hit = next((i for i, e in enumerate(ranked) if {e.get("home"), e.get("away")} == pair), None)
        check.append({"week": w, "date": a["date"], "teams": a["teams"], "host": a["host"],
                      "probability": ranked[hit]["probability"] if hit is not None else None,
                      "rank": hit + 1 if hit is not None else None,
                      "favourite": {"home": ranked[0].get("home"), "away": ranked[0].get("away"),
                                    "probability": ranked[0]["probability"]}})

    sites = sorted(
        ({"team": t, **v} for t, v in forecast_ahead["teams"].items() if t in names and v["hostsAtLeastOnce"] >= 0.02),
        key=lambda s: -s["hostsAtLeastOnce"])[:14]

    return {
        "season": year,
        "week": int(payload["week"]),
        "sims": forecast_ahead["sims"],
        "announced": announced,
        "weeks": weeks,
        "check": check,
        "sites": sites,
        "armyNavy": model["armyNavy"],
        "model": {"features": model["features"], "coefficients": model["coefficients"], "weeks": model["weeks"],
                  "fitted": model["fitted"], "otherRate": model["otherRate"]},
    }
