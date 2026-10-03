"""Rebuild the current season's slate archives for weeks the site was not yet saving live.

``slatearchive.snapshot`` freezes a week the first time a build finds the slate has moved on, so a week
the site never saw (the season's first two, before the archive existed) has no page, and a week first
saved late (week 4, saved after most of its games were final) kept those games in ``results`` as a
compact table instead of the full rows every other archive has.

This rebuilds a week the way ``slate.build`` priced it: ratings as they stood entering the week (the
preseason prior for week 1, the previous week's snapshot after that), the market line where the feed
has one, and each team's rank from those same ratings. The result is the same record ``snapshot`` would
have written, with every game in ``days``, and is written once - an existing file is never replaced
unless ``--merge-results`` is given for a week saved live with a ``results`` block.

Run:  PYTHONPATH=src python3 scripts/backfill_season_slates.py 1 2
      PYTHONPATH=src python3 scripts/backfill_season_slates.py --merge-results 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

from mri.betting import board as board_module  # noqa: E402
from mri.betting import lines as lines_module  # noqa: E402
from mri.export import sitedata, slate, slatearchive  # noqa: E402
from mri.ingest import cfbd  # noqa: E402
from mri.ratings import mri2, priors  # noqa: E402

SEASON = 2026
DOCS = ROOT / "docs"
SITE_JSON = ROOT / "site" / "data" / "site.json"


def entering_ratings(schedule: pd.DataFrame, weekly: pd.DataFrame, fbs: list[str], week: int):
    """(power by team, home field) as the model had them entering ``week``: the same rule as ``slate.build``."""
    if week > 1 and not weekly[weekly["week"] == week - 1].empty:
        rated = weekly[weekly["week"] == week - 1]
        return rated.set_index("team")["power"].to_dict(), float(rated["home_field"].iloc[0])
    every = sorted(set(schedule["team1"]) | set(schedule["team2"]))
    prior = priors.for_season(SEASON, board_module._previous_season(SEASON), every, fbs)
    return prior.to_dict(), mri2.DEFAULT_HOME_FIELD_PRIOR


def build_week(week: int, schedule: pd.DataFrame, weekly: pd.DataFrame, teams_payload: list[dict]) -> dict:
    fbs = [t["team"] for t in teams_payload]
    power, home_field = entering_ratings(schedule, weekly, fbs, week)
    replacement = min(power[t] for t in fbs if t in power) - 8.0
    book = lines_module.preferred_lines(SEASON)
    book = book.drop_duplicates("game_id").set_index("game_id") if not book.empty else pd.DataFrame()

    games = schedule[schedule["block"] == week]
    entries, fcs_entries, finals = [], [], {}
    for g in games.itertuples():
        home, away = g.team2, g.team1
        if home not in fbs and away not in fbs:
            continue
        when = slate._eastern(g.start_date)
        tbd = bool(getattr(g, "start_time_tbd", False))
        predicted = (power.get(home, replacement) - power.get(away, replacement)
                     + (0.0 if g.neutral else home_field))
        entry = {
            "id": int(g.game_id), "week": week,
            "date": when.strftime("%Y-%m-%d") if when else None,
            "dateLabel": when.strftime("%a, %b %-d") if when else "Date to be set",
            "time": "TBD" if tbd or not when else when.strftime("%-I:%M %p").replace(" ", " ") + " ET",
            "sort": when.strftime("%H%M") if when and not tbd else "9999",
            "home": home, "away": away, "neutral": bool(g.neutral),
            "predicted": round(predicted, 1),
            "homeWinProbability": round(float(norm.cdf(predicted / slate.SIGMA)), 3),
            "played": False,
        }
        if home in fbs and away in fbs:
            if isinstance(book, pd.DataFrame) and g.game_id in book.index:
                line = book.loc[g.game_id]
                entry["market"] = round(float(line["market"]), 1) if pd.notna(line["market"]) else None
                entry["open"] = round(float(line["market_open"]), 1) if pd.notna(line.get("market_open")) else None
                entry["total"] = float(line["total"]) if pd.notna(line.get("total")) else None
            entries.append(entry)
            finals[str(int(g.game_id))] = {"home": int(g.pts2), "away": int(g.pts1)}
        else:
            fcs_entries.append(entry)

    order = lambda e: (e["date"] or "9999", e["sort"])  # noqa: E731
    ordered = sorted(entries, key=order)
    days: list[dict] = []
    for e in ordered:
        if not days or days[-1]["date"] != e["date"]:
            days.append({"date": e["date"], "label": e["dateLabel"], "games": []})
        days[-1]["games"].append(e)

    field = sorted(fbs, key=lambda t: -power.get(t, replacement))
    ranks = {t: i for i, t in enumerate(field, start=1)}
    return {
        "season": SEASON, "week": week, "days": days, "results": [], "fcs": sorted(fcs_entries, key=order),
        "watch": [], "flagged": 0, "games": len(entries) + len(fcs_entries),
        "saved": f"{SEASON}-week-{week}-reconstructed",
        "finals": finals,
        "teams": slatearchive.historical_teams(teams_payload, ranks),
    }


def merge_results(record: dict) -> dict:
    """Fold a late-saved week's ``results`` back into ``days`` so every game is a full row with its final.

    The lines a played game carries (model entering the week, market, opener, total) are already on the
    result entry; what a pre-kickoff row has and these do not is the board's edge and the playoff stake,
    which were never captured for a game already over by the time the page was saved.
    """
    finals = dict(record["finals"])
    games = [g for d in record["days"] for g in d["games"]]
    for g in record["results"]:
        res = g["result"]
        finals[str(g["id"])] = {"home": res["homeScore"], "away": res["awayScore"]}
        games.append({k: v for k, v in g.items() if k != "result"} | {"played": False})
    games.sort(key=lambda g: (g["date"] or "9999", g["sort"]))
    days: list[dict] = []
    for g in games:
        if not days or days[-1]["date"] != g["date"]:
            days.append({"date": g["date"], "label": g["dateLabel"], "games": []})
        days[-1]["games"].append(g)
    return {**record, "days": days, "results": [], "finals": finals}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("weeks", type=int, nargs="+")
    parser.add_argument("--merge-results", action="store_true")
    args = parser.parse_args()

    if args.merge_results:
        for week in args.weeks:
            path = slatearchive.archive_path(DOCS, SEASON, week)
            record = json.loads(path.read_text())
            path.write_text(json.dumps(merge_results(record), indent=1))
            print(f"merged results into days for week {week}")
        return

    schedule = slate._canonical(cfbd.games(SEASON, completed_only=False))
    schedule["block"] = cfbd.sequence(schedule)
    weekly = sitedata.weekly_ratings(SEASON)
    teams_payload = json.loads(SITE_JSON.read_text())["teams"]
    for week in args.weeks:
        path = slatearchive.archive_path(DOCS, SEASON, week)
        if path.exists():
            print(f"week {week}: {path.name} exists, leaving it alone")
            continue
        record = build_week(week, schedule, weekly, teams_payload)
        path.write_text(json.dumps(record, indent=1))
        print(f"week {week}: wrote {path.name} ({record['games']} games)")


if __name__ == "__main__":
    main()
