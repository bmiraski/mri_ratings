"""The Heisman page's data: this week's odds, and how they have moved.

The odds come from ``mri.heisman.live`` (the season simulation, the projection and
the final-vote model). This module turns them into what a page can show, keeps the
week-by-week history that movement is measured against, and stops updating when
the ballots close: after the deadline the honest thing to show is the last odds
before the voters decided, labelled as such, not odds recomputed from a season the
voters have already judged.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

from ..heisman import data, live

MARKET = Path(__file__).resolve().parents[3] / "data" / "heisman_market.json"
MARKET_STALE_DAYS = 10
FINALIST_ROWS = 8

SHOWN = 15
KEPT = 30                 # players whose odds are saved each week, for movement
POSITIONS = {"QB": "QB", "RB": "RB", "REC": "WR/TE"}


def _load(path: Path, year: int) -> dict:
    if path.exists():
        history = json.loads(path.read_text())
        if history.get("season") == year:
            return history
    return {"season": year, "weeks": {}}


def _save(path: Path, history: dict) -> None:
    path.write_text(json.dumps(history, sort_keys=True, separators=(",", ":")) + "\n")


def _key(player: str, team: str) -> str:
    return f"{player}|{team}"


def _num(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def market_for(rows: list[dict], today: dt.date, path: Path | None = None) -> dict | None:
    """Sportsbook odds for the players on the page, if a recent enough snapshot has been saved.

    The market is a benchmark, never an input: it is shown beside the model so a reader can see
    where they disagree, and nothing here changes a probability. Stale odds are worse than none, so a file
    older than ten days is ignored.
    """
    path = path or MARKET
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    if (today - dt.date.fromisoformat(saved["asOf"])).days > MARKET_STALE_DAYS:
        return None
    wanted = {data.normalize(k): v for k, v in saved["odds"].items()}
    matched = 0
    for r in rows:
        odds = wanted.get(data.normalize(r["player"]))
        if odds is None:
            r["market"] = None
            continue
        matched += 1
        r["market"] = {"odds": odds, "implied": round(100 / (odds + 100) if odds > 0 else -odds / (-odds + 100), 4)}
    if matched < 2:
        for r in rows:
            r["market"] = None
        return None
    return {"asOf": saved["asOf"], "source": saved["source"], "note": saved["note"]}


def defender_rate(voting: dict) -> dict:
    """How often a defender has been invited to the ceremony, from the voting record itself."""
    seasons = sorted(int(y) for y in voting["seasons"] if int(y) >= 2012)
    hit = {y: [f["player"] for f in voting["seasons"][str(y)]["finalists"] if f.get("defender") and f.get("finalist") is not False]
           for y in seasons}
    years = [y for y, names in hit.items() if names]
    return {"seasons": len(seasons), "count": len(years), "years": years, "names": [n for y in years for n in hit[y]]}


def build(year: int, payload: dict, history_path: Path, *, today: dt.date | None = None, odds_fn=None,
          sims: int = 10_000, defenders_fn=None) -> dict | None:
    """The page's data, or None if there is nothing honest to show."""
    today = today or dt.date.today()
    voting = data.load_voting()
    dates = data.key_dates(voting, year)            # None for a season whose calendar has not been added yet
    history = _load(history_path, year)
    closed = bool(dates) and today >= dt.date.fromisoformat(dates["votingDeadline"])
    week = int(payload["week"])

    if closed:
        # The ballots are in. Show the last odds recorded before they closed, unchanged.
        if not history["weeks"]:
            return None
        week = max(int(w) for w in history["weeks"])
        frozen = history["weeks"][str(week)]
        return {**frozen["page"], "closed": True, "updated": frozen["page"]["updated"], "dates": dates}

    result = (odds_fn or live.current_odds)(payload, sims=sims)
    if not result:
        return None
    odds = result["odds"]
    teams = {t["team"]: t for t in payload["teams"]}

    earlier = [int(w) for w in history["weeks"] if int(w) < week]
    before = history["weeks"][str(max(earlier))]["odds"] if earlier else {}
    before_rank = {k: i + 1 for i, k in enumerate(sorted(before, key=lambda k: -before[k][0]))}

    rows = []
    for i, r in odds.head(SHOWN).iterrows():
        key = _key(r["player"], r["team"])
        yards = float(r["pass_yds"] + r["rush_yds"] + r["rec_yds"])
        score = yards + 20 * float(r["pass_td"] + r["rush_td"] + r["rec_td"])
        prior = before.get(key)
        rows.append({
            "player": r["player"], "team": r["team"], "position": POSITIONS[r["group"]], "group": r["group"],
            "win": round(float(r["win"]), 4), "finalist": round(float(r["finalist"]), 4),
            "change": round(float(r["win"]) - prior[0], 4) if prior else None,
            "rankChange": (before_rank[key] - (i + 1)) if key in before_rank else None,
            "line": {"passYds": int(r["pass_yds"]), "passTd": int(r["pass_td"]), "rushYds": int(r["rush_yds"]),
                     "rushTd": int(r["rush_td"]), "recYds": int(r["rec_yds"]), "recTd": int(r["rec_td"])},
            "pace": int(round(yards * float(r["projected"]) / score, -1)) if score > 0 else None,
            "teamRank": int(teams[r["team"]]["rank"]) if r["team"] in teams else None,
            "teamRecord": f"{teams[r['team']]['wins']}\u2013{teams[r['team']]['losses']}" if r["team"] in teams else None,
            "teamTop4": round(float(r["team_top4"]), 3),
            "winIfTop4": _num(r["win_if_top4"]), "winIfNot": _num(r["win_if_not"]),
        })
    shown_win = sum(r["win"] for r in rows)
    by_position = {POSITIONS[g]: round(float(v), 3) for g, v in odds.groupby("group")["win"].sum().items()}
    by_team: dict[str, dict] = {}
    for r in rows:
        by_team.setdefault(r["team"], {"player": r["player"], "win": r["win"], "position": r["position"]})

    market = market_for(rows, today)
    try:
        watch = (defenders_fn or live.defenders_to_watch)(payload)
    except Exception:  # noqa: BLE001 - a list of names to watch is not worth the page
        watch = []
    reach = [{"player": r["player"], "team": r["team"], "position": POSITIONS[r["group"]], "finalist": round(float(r["finalist"]), 4),
              "win": round(float(r["win"]), 4)} for _, r in odds.sort_values("finalist", ascending=False).head(FINALIST_ROWS).iterrows()]
    voted = {int(y): next(f["player"] for f in s["finalists"] if f["finish"] == 1) for y, s in voting["seasons"].items()}
    page = {
        "season": year, "week": week, "sims": result["sims"], "candidates": result["candidates"], "updated": today.isoformat(),
        "closed": False, "dates": dates, "players": rows, "rest": round(max(0.0, 1.0 - shown_win), 4),
        "byPosition": by_position, "byTeam": by_team, "winners": {str(y): n for y, n in voted.items()},
        "reach": reach, "expectedFinalists": round(float(odds["finalist"].sum()), 2), "defenders": {**defender_rate(voting), "watch": watch},
        "market": market,
    }
    history["weeks"][str(week)] = {
        "odds": {_key(r["player"], r["team"]): [round(float(r["win"]), 4), round(float(r["finalist"]), 4)]
                 for _, r in odds.head(KEPT).iterrows()},
        "page": page,
    }
    _save(history_path, history)
    return page
