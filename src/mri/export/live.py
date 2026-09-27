"""Live scores for the slate: what the scoreboard says, and what the model now makes of it.

A GitHub Action runs this every fifteen minutes through both seasons, but it only
spends an API call while one of a slate's games is on: from shortly before the
first tip or kickoff until every game that has started is final. Outside that
window it reads two local files and exits.

Its output is ``live.json`` beside each sport's slate page, which the page fetches
on its own. The pages are rebuilt once a day; the scores are not part of them.

The live win chance is the model's, not the feed's, so it agrees with the
pregame number beside it: the final margin is the current lead plus the model's
predicted margin for the time that is left, with the spread of outcomes shrinking
as the clock runs down. At the start it is exactly the pregame chance.

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
from dataclasses import dataclass
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
    EASTERN = ZoneInfo("America/New_York")
except Exception:  # noqa: BLE001 - a runner without tz data still gets scores
    EASTERN = dt.timezone(dt.timedelta(hours=-4))


@dataclass(frozen=True)
class Sport:
    name: str
    scoreboard: str
    periods: int                 # regulation periods: quarters or halves
    period_seconds: float
    sigma: float                 # each betting board's single-game spread; tests keep them in step
    second_half: int             # the first period of the second half, when upset alerts start
    longest: dt.timedelta        # after this long, a game that has not reported final is left alone
    overtime_fraction: float     # an overtime period as a share of regulation, for the live chance

    @property
    def regulation(self) -> float:
        return self.periods * self.period_seconds


FOOTBALL = Sport("football", "https://api.collegefootballdata.com/scoreboard?classification=fbs",
                 4, 900.0, 16.5, 3, dt.timedelta(hours=5), 0.05)
BASKETBALL = Sport("basketball", "https://api.collegebasketballdata.com/scoreboard",
                   2, 1200.0, 11.0, 2, dt.timedelta(hours=3), 0.125)

SCOREBOARD = FOOTBALL.scoreboard
SIGMA = FOOTBALL.sigma
REGULATION = FOOTBALL.regulation
OVERTIME_FRACTION = FOOTBALL.overtime_fraction
LEAD_IN = dt.timedelta(minutes=10)
LONGEST_GAME = FOOTBALL.longest
FINAL = ("completed", "final")   # football's word, and basketball's

# An upset is brewing, from the second half on, when either:
BIG_UNDERDOG = 0.25          # a team the model gave this chance or less is leading, or
RANKED = 25                  # a team ranked this high is trailing one that is not.


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def seconds_left(period: int | None, clock: str | None, *, sport: Sport = FOOTBALL) -> float:
    """Regulation seconds remaining; zero once overtime starts."""
    if not period:
        return sport.regulation
    if period > sport.periods:
        return 0.0
    try:
        parts = [int(float(p)) for p in (clock or "0:00").split(":")]
    except ValueError:
        parts = [0]
    in_period = sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
    return max(0.0, (sport.periods - period) * sport.period_seconds + min(in_period, sport.period_seconds))


def win_probability(predicted: float, home_lead: float, left: float, *, overtime: bool = False,
                    sport: Sport = FOOTBALL) -> float:
    """The home side's chance: current lead plus the model's margin for the time that is left."""
    fraction = sport.overtime_fraction if overtime else left / sport.regulation
    if fraction <= 0:
        return 1.0 if home_lead > 0 else 0.0 if home_lead < 0 else 0.5
    return _phi((home_lead + predicted * fraction) / (sport.sigma * math.sqrt(fraction)))


def status_label(status: str, period: int | None, clock: str | None, *, sport: Sport = FOOTBALL) -> str:
    regulation = sport.periods
    if status in FINAL:
        extra = (period or regulation) - regulation
        return "Final" if extra <= 0 else ("Final/OT" if extra == 1 else f"Final/{extra}OT")
    if status != "in_progress" or not period:
        return ""
    if period > regulation:
        extra = period - regulation
        return "OT" if extra == 1 else f"{extra}OT"
    if seconds_left(period, clock, sport=sport) == sport.regulation / 2 and period == regulation // 2:
        return "Half"
    if sport.periods == 2:
        return f"{'1st' if period == 1 else '2nd'} {clock or ''}".strip()
    return f"Q{period} {clock or ''}".strip()


def upset(game: dict, home: int, away: int, period: int | None, final: bool, ranks: dict[str, int],
          *, sport: Sport = FOOTBALL) -> str | None:
    """Why this game is an upset in the making (or, once final, an upset), or None.

    From the second half on only: an early run is not a story. The two rules are
    independent, and either is enough.
    """
    if not final and (period or 0) < sport.second_half:
        return None
    lead = home - away
    if lead == 0:
        return None
    leader, trailer = (game["home"], game["away"]) if lead > 0 else (game["away"], game["home"])
    leader_pregame = game["homeWinProbability"] if lead > 0 else 1 - game["homeWinProbability"]
    start = "tip-off" if sport is BASKETBALL else "kickoff"
    if leader_pregame <= BIG_UNDERDOG:
        chance = f"{leader_pregame:.0%}" if leader_pregame >= 0.01 else "under 1%"
        verb = "won" if final else "leads"
        return f"{leader} {verb}; the model gave them {chance} before {start}"
    trailer_rank, leader_rank = ranks.get(trailer, 999), ranks.get(leader, 999)
    if trailer_rank <= RANKED < leader_rank:
        verb = "lost to" if final else "trails"
        return f"#{trailer_rank} {trailer} {verb} unranked {leader}"
    return None


def kickoff(game: dict) -> dt.datetime | None:
    """Start time in Eastern time from the slate's date and HHMM sort key; TBD games count from noon."""
    if not game.get("date"):
        return None
    day = dt.date.fromisoformat(game["date"])
    hhmm = "1200" if game.get("sort") in (None, "9999") else game["sort"]
    return dt.datetime.combine(day, dt.time(int(hhmm[:2]), int(hhmm[2:])), tzinfo=EASTERN)


def slate_games(slate: dict) -> list[dict]:
    return [g for day in slate.get("days", []) for g in day["games"]]


def slate_key(slate: dict) -> str:
    """What a live record belongs to: a football week, or a basketball day."""
    if slate.get("week") is not None:
        return f"{slate.get('season')}-w{slate['week']}"
    return str(slate.get("date"))


def _same(slate: dict, live: dict | None) -> bool:
    if not live:
        return False
    if live.get("key") is not None:
        return live["key"] == slate_key(slate)
    return live.get("week") == slate.get("week") and live.get("season") == slate.get("season")


def should_poll(slate: dict, live: dict | None, now: dt.datetime, *, sport: Sport = FOOTBALL) -> bool:
    """True while any slate game is about to start or has started and is not yet final."""
    done = {gid for gid, g in (live or {}).get("games", {}).items() if g.get("status") in FINAL} \
        if _same(slate, live) else set()
    for game in slate_games(slate):
        start = kickoff(game)
        if start is None or str(game["id"]) in done:
            continue
        if start - LEAD_IN <= now <= start + sport.longest:
            return True
    return False


def next_window(slate: dict, live: dict | None, now: dt.datetime, *, sport: Sport = FOOTBALL) -> dt.datetime | None:
    """When polling is next worth doing: ``now`` if a game is on, else the soonest lead-in, else None."""
    if should_poll(slate, live, now, sport=sport):
        return now
    done = {gid for gid, g in (live or {}).get("games", {}).items() if g.get("status") in FINAL} \
        if _same(slate, live) else set()
    upcoming = [start - LEAD_IN for game in slate_games(slate)
                if (start := kickoff(game)) is not None and str(game["id"]) not in done and start - LEAD_IN > now]
    return min(upcoming, default=None)


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def window(root: Path, *, now: dt.datetime, sport: Sport = FOOTBALL) -> dt.datetime | None:
    """``next_window`` for one sport's folder, read from its slate.json and live.json."""
    slate = _read(root / "slate.json")
    if slate is None:
        return None
    slate.setdefault("season", (_read(root / _site_file(sport)) or {}).get("season"))
    return next_window(slate, _read(root / "live.json"), now, sport=sport)


def _site_file(sport: Sport) -> str:
    return "basketball.json" if sport is BASKETBALL else "site.json"


def merge(slate: dict, scoreboard: list[dict], previous: dict | None, ranks: dict[str, int],
          now: dt.datetime, *, sport: Sport = FOOTBALL) -> dict:
    """Fold a scoreboard reading into the live record for this slate.

    The record keeps every game seen for the slate - a football week, a basketball
    day - so earlier finals stay in it as later games are played.
    """
    games = dict((previous or {}).get("games") or {}) if _same(slate, previous) else {}
    ours = {g["id"]: g for g in slate_games(slate)}
    for entry in scoreboard:
        game = ours.get(entry.get("id"))
        status = entry.get("status")
        if game is None or status not in ("in_progress",) + FINAL:
            continue
        home = (entry.get("homeTeam") or {}).get("points")
        away = (entry.get("awayTeam") or {}).get("points")
        if home is None or away is None:
            continue
        period, clock = entry.get("period"), entry.get("clock")
        final = status in FINAL
        left = 0.0 if final else seconds_left(period, clock, sport=sport)
        chance = win_probability(game["predicted"], home - away, left, sport=sport,
                                 overtime=not final and (period or 0) > sport.periods)
        possession = entry.get("possession")
        games[str(game["id"])] = {
            "status": "completed" if final else status,
            "label": status_label(status, period, clock, sport=sport),
            "period": period,
            "clock": clock,
            "home": int(home), "away": int(away),
            "possession": possession if possession in ("home", "away") else None,
            "homeWinProbability": round(chance, 3),
            "upset": upset(game, int(home), int(away), period, final, ranks, sport=sport),
        }
    return {
        "key": slate_key(slate),
        "season": slate.get("season"),
        "week": slate.get("week"),
        "date": slate.get("date"),
        "updated": now.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "updatedLabel": now.astimezone(EASTERN).strftime("%-I:%M %p ET"),
        "games": games,
    }


def fetch_scoreboard(key: str, sport: Sport = FOOTBALL) -> list[dict]:
    request = urllib.request.Request(sport.scoreboard, headers={
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


def run(root: Path, *, now: dt.datetime | None = None, key: str | None = None, fetch=None,
        sport: Sport = FOOTBALL) -> str:
    """One tick of the live job for one sport. ``root`` is that sport's folder: docs/, or docs/basketball/."""
    now = now or dt.datetime.now(dt.timezone.utc)
    slate_path, live_path = root / "slate.json", root / "live.json"
    if not slate_path.exists():
        return "no slate"
    slate = json.loads(slate_path.read_text())
    site_file = root / _site_file(sport)
    site = json.loads(site_file.read_text()) if site_file.exists() else {}
    slate.setdefault("season", site.get("season"))
    previous = json.loads(live_path.read_text()) if live_path.exists() else None
    if not should_poll(slate, previous, now, sport=sport):
        return "no games on"
    key = key or os.environ.get("CFBD_API_KEY", "").strip()
    if not key:
        return "no API key"
    ranks = {t["team"]: t["rank"] for t in site.get("teams", [])}
    board = (fetch or (lambda k: fetch_scoreboard(k, sport)))(key)
    live = merge(slate, board, previous, ranks, now, sport=sport)
    live_path.write_text(json.dumps(live, indent=1))
    on = sum(g["status"] == "in_progress" for g in live["games"].values())
    return f"{on} in progress, {len(live['games'])} tracked"
