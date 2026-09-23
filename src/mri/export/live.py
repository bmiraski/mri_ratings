"""Live scores for the slate: what the scoreboard says, and what the model now makes of it.

A GitHub Action runs this every fifteen minutes through the season, but it only
spends an API call while one of the slate's games is on: from shortly before the
first kickoff of the day until every game that has started is final. Outside
that window it reads two local files and exits.

Its output is ``docs/live.json``, which the slate page fetches on its own. The
page is rebuilt once a day; the scores are not part of it.

The live win chance is the model's, not the feed's, so it agrees with the
pregame number beside it: the final margin is the current lead plus the model's
predicted margin for the time that is left, with the spread of outcomes shrinking
as the clock runs down. At kickoff it is exactly the pregame chance.

Standard library only. The Action runs on the runner's own Python without
installing the rating stack, which keeps a fifteen-minute job to a few seconds.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
    EASTERN = ZoneInfo("America/New_York")
except Exception:  # noqa: BLE001 - a runner without tz data still gets scores
    EASTERN = dt.timezone(dt.timedelta(hours=-4))

SCOREBOARD = "https://api.collegefootballdata.com/scoreboard"
SIGMA = 16.5                 # the betting board's; a test keeps the two in step
REGULATION = 3600.0
OVERTIME_FRACTION = 0.05     # an overtime period is worth about three minutes of regulation
LEAD_IN = dt.timedelta(minutes=10)
LONGEST_GAME = dt.timedelta(hours=5)

# An upset is brewing, from the second half on, when either:
BIG_UNDERDOG = 0.25          # a team the model gave this chance or less is leading, or
RANKED = 25                  # a team ranked this high is trailing one that is not.


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def seconds_left(period: int | None, clock: str | None) -> float:
    """Regulation seconds remaining; zero once overtime starts."""
    if not period:
        return REGULATION
    if period > 4:
        return 0.0
    try:
        parts = [int(p) for p in (clock or "0:00").split(":")]
    except ValueError:
        parts = [0]
    in_period = sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
    return max(0.0, (4 - period) * 900.0 + min(in_period, 900))


def win_probability(predicted: float, home_lead: float, left: float, *, overtime: bool = False) -> float:
    """The home side's chance: current lead plus the model's margin for the time that is left."""
    fraction = OVERTIME_FRACTION if overtime else left / REGULATION
    if fraction <= 0:
        return 1.0 if home_lead > 0 else 0.0 if home_lead < 0 else 0.5
    return _phi((home_lead + predicted * fraction) / (SIGMA * math.sqrt(fraction)))


def status_label(status: str, period: int | None, clock: str | None) -> str:
    if status == "completed":
        return "Final" if (period or 4) <= 4 else ("Final/OT" if period == 5 else f"Final/{period - 4}OT")
    if status != "in_progress" or not period:
        return ""
    if period > 4:
        return "OT" if period == 5 else f"{period - 4}OT"
    if period == 2 and seconds_left(period, clock) == 1800:
        return "Half"
    return f"Q{period} {clock or ''}".strip()


def upset(game: dict, home: int, away: int, period: int | None, final: bool, ranks: dict[str, int]) -> str | None:
    """Why this game is an upset in the making (or, once final, an upset), or None.

    From the second half on only: an early field goal is not a story. The two
    rules are independent, and either is enough.
    """
    if not final and (period or 0) < 3:
        return None
    lead = home - away
    if lead == 0:
        return None
    leader, trailer = (game["home"], game["away"]) if lead > 0 else (game["away"], game["home"])
    leader_pregame = game["homeWinProbability"] if lead > 0 else 1 - game["homeWinProbability"]
    if leader_pregame <= BIG_UNDERDOG:
        chance = f"{leader_pregame:.0%}" if leader_pregame >= 0.01 else "under 1%"
        verb = "won" if final else "leads"
        return f"{leader} {verb}; the model gave them {chance} before kickoff"
    trailer_rank, leader_rank = ranks.get(trailer, 999), ranks.get(leader, 999)
    if trailer_rank <= RANKED < leader_rank:
        verb = "lost to" if final else "trails"
        return f"#{trailer_rank} {trailer} {verb} unranked {leader}"
    return None


def kickoff(game: dict) -> dt.datetime | None:
    """Kickoff in Eastern time from the slate's date and HHMM sort key; TBD games count from noon."""
    if not game.get("date"):
        return None
    day = dt.date.fromisoformat(game["date"])
    hhmm = "1200" if game.get("sort") in (None, "9999") else game["sort"]
    return dt.datetime.combine(day, dt.time(int(hhmm[:2]), int(hhmm[2:])), tzinfo=EASTERN)


def slate_games(slate: dict) -> list[dict]:
    return [g for day in slate.get("days", []) for g in day["games"]]


def should_poll(slate: dict, live: dict | None, now: dt.datetime) -> bool:
    """True while any slate game is about to start or has started and is not yet final."""
    done = {gid for gid, g in ((live or {}).get("games") or {}).items() if g.get("status") == "completed"} \
        if (live or {}).get("week") == slate.get("week") else set()
    for game in slate_games(slate):
        start = kickoff(game)
        if start is None or str(game["id"]) in done:
            continue
        if start - LEAD_IN <= now <= start + LONGEST_GAME:
            return True
    return False


def merge(slate: dict, scoreboard: list[dict], previous: dict | None, ranks: dict[str, int],
          now: dt.datetime) -> dict:
    """Fold a scoreboard reading into the week's live record.

    The record keeps every game seen this week, so Thursday's finals are still in
    it on Saturday and the archive can be written from it on Sunday.
    """
    same_week = (previous or {}).get("week") == slate["week"] and (previous or {}).get("season") == slate.get("season")
    games = dict((previous or {}).get("games") or {}) if same_week else {}
    ours = {g["id"]: g for g in slate_games(slate)}
    for entry in scoreboard:
        game = ours.get(entry.get("id"))
        status = entry.get("status")
        if game is None or status not in ("in_progress", "completed"):
            continue
        home = entry.get("homeTeam", {}).get("points")
        away = entry.get("awayTeam", {}).get("points")
        if home is None or away is None:
            continue
        period, clock = entry.get("period"), entry.get("clock")
        final = status == "completed"
        left = 0.0 if final else seconds_left(period, clock)
        chance = win_probability(game["predicted"], home - away, left,
                                 overtime=not final and (period or 0) > 4)
        possession = entry.get("possession")
        games[str(game["id"])] = {
            "status": status,
            "label": status_label(status, period, clock),
            "period": period,
            "clock": clock,
            "home": int(home), "away": int(away),
            "possession": possession if possession in ("home", "away") else None,
            "homeWinProbability": round(chance, 3),
            "upset": upset(game, int(home), int(away), period, final, ranks),
        }
    return {
        "season": slate.get("season"),
        "week": slate["week"],
        "updated": now.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "updatedLabel": now.astimezone(EASTERN).strftime("%-I:%M %p ET"),
        "games": games,
    }


def fetch_scoreboard(key: str) -> list[dict]:
    request = urllib.request.Request(f"{SCOREBOARD}?classification=fbs", headers={
        "Authorization": f"Bearer {key}", "Accept": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 3:
                raise
            time.sleep(2 ** attempt)
    return []


def run(docs: Path, *, now: dt.datetime | None = None, key: str | None = None, fetch=fetch_scoreboard) -> str:
    """One tick of the live job. Returns what it did, for the Action's log."""
    now = now or dt.datetime.now(dt.timezone.utc)
    slate_path, live_path = docs / "slate.json", docs / "live.json"
    if not slate_path.exists():
        return "no slate"
    slate = json.loads(slate_path.read_text())
    site = json.loads((docs / "site.json").read_text()) if (docs / "site.json").exists() else {}
    slate.setdefault("season", site.get("season"))
    previous = json.loads(live_path.read_text()) if live_path.exists() else None
    if not should_poll(slate, previous, now):
        return "no games on"
    key = key or os.environ.get("CFBD_API_KEY", "").strip()
    if not key:
        return "no API key"
    ranks = {t["team"]: t["rank"] for t in site.get("teams", [])}
    live = merge(slate, fetch(key), previous, ranks, now)
    live_path.write_text(json.dumps(live, indent=1))
    on = sum(g["status"] == "in_progress" for g in live["games"].values())
    return f"{on} in progress, {len(live['games'])} tracked this week"
