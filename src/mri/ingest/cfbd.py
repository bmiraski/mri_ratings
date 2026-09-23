"""Client for the College Football Data API, with a cache that never forgets.

The free tier allows 1,000 calls a month, which is plenty if nothing is ever
fetched twice. Every response is written to ``data/raw`` keyed by endpoint and
parameters, and finished weeks are served from disk forever after. Only the
current week is ever re-fetched.

Normalization matches the Excel archive so both sources feed the same rating
code: ``team1`` is the visitor and ``team2`` the host, which is the convention
every workbook used.

One real improvement over the archive: the API names FCS opponents individually
and flags each team's classification, so MRI 2.0 can rate Mercer and North
Dakota State as the different propositions they are. MRI Classic still pools
them into "Non D1A", because that is what MRI Classic did.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.collegefootballdata.com"
ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = ROOT / "data" / "raw"
POOLED_FCS = "Non D1A"

RETRIES = 6
MAX_BACKOFF = 60.0
FBS = "fbs"


# Cache files this process has already re-fetched. A build asks for the current
# season's games from several places, and one fresh copy per run is all it needs
# - without this a refresh costs one API call per caller instead of one per run.
_FETCHED: set[Path] = set()


def current_season(today: dt.date | None = None) -> int:
    """The football season in progress: the calendar year from June onward, and
    the previous one during the January playoffs and the offseason's start."""
    today = today or dt.date.today()
    return today.year if today.month >= 6 else today.year - 1


class CfbdError(RuntimeError):
    pass


class QuotaExceeded(CfbdError):
    """The monthly call allowance is gone. Distinct from a passing throttle:
    nothing that waits will help, so callers that can degrade should."""


def _api_key() -> str:
    key = os.environ.get("CFBD_API_KEY")
    if key:
        return key.strip()
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("CFBD_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise CfbdError(
        "No CFBD API key. Set CFBD_API_KEY or put it in .env at the repo root. "
        "Free keys: https://collegefootballdata.com/key"
    )


def _cache_path(endpoint: str, params: dict) -> Path:
    """A readable, and above all unique, filename for a response.

    The signature is trimmed to keep filenames manageable, and the trim used to
    be the whole of it - which silently made two different requests share a
    cache file whenever they agreed on everything up to the cut. The basketball
    windows are exactly that shape: sorted alphabetically, ``endDateRange`` and
    ``season`` come first and use up the budget, so every window ending on the
    same date collided no matter where it started.

    Monthly windows all end on different dates, so nothing was wrong until
    something tried to split one - and then a half-month request quietly
    returned the whole month's cached, truncated payload, and the split looked
    like it had made matters worse rather than exposing a cache bug.

    A digest is appended whenever the signature had to be cut, which keeps every
    untruncated name exactly as it was and makes the rest unique.
    """
    stem = endpoint.strip("/").replace("/", "_")
    if params:
        digest = hashlib.sha1(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()[:10]
        signature = "_".join(f"{k}{v}" for k, v in sorted(params.items()) if k != "year")
        signature = f"{signature[:40]}_{digest}" if len(signature) > 40 else (signature or digest)
        year = params.get("year", "")
        stem = f"{stem}_{year}_{signature}"
    return CACHE_DIR / f"{stem}.json"


def request(endpoint: str, *, refresh: bool = False, **params):
    """Fetch an endpoint, using the on-disk cache unless ``refresh``."""
    params = {k: v for k, v in params.items() if v is not None}
    path = _cache_path(endpoint, params)

    if refresh and path in _FETCHED:
        refresh = False

    if path.exists() and not refresh:
        return json.loads(path.read_text())

    # A refresh is an optimization - "this week's numbers may have moved" - not
    # a requirement. When the allowance is gone, the choice is between last
    # week's cached copy and no page at all, and the cached copy wins every
    # time. Only a genuinely uncached request has nothing to fall back to.
    stale = json.loads(path.read_text()) if path.exists() else None

    try:
        key = _api_key()
    except CfbdError:
        # Same principle as a spent quota: a refresh is an optimization. With a
        # cached copy in hand and no way to ask for a newer one, the copy is the
        # answer. This is what lets the test step run without a key - it has the
        # committed cache and no business fetching anything.
        if stale is not None:
            return stale
        raise

    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    # 1s, 2s, 4s over three attempts was fine when a build made eight calls. The
    # basketball paths now fetch a season a window at a time and split the busy
    # ones, so a run can make hundreds - and a real rate-limit window outlasts
    # seven seconds every time. Retry-After is honoured when the server sends
    # one, because guessing is strictly worse than being told.
    for attempt in range(RETRIES):
        response = requests.get(f"{BASE_URL}{endpoint}", params=params, headers=headers, timeout=60)
        if response.status_code == 429:
            # Two different things arrive as 429. A throttle clears in seconds
            # and is worth waiting out; a spent monthly quota does not clear
            # until the month does, and retrying it six times just means six
            # more refusals and two minutes of nothing. The body says which.
            if "quota" in response.text.lower():
                if stale is not None:
                    print(f"  quota spent; serving cached {endpoint}")
                    return stale
                raise QuotaExceeded(
                    f"{endpoint}: monthly API call quota exhausted. Cached data "
                    "still works; anything uncached has to wait for the quota to "
                    "reset, or a higher tier at collegefootballdata.com/upgrade."
                )
            wait = response.headers.get("Retry-After")
            delay = float(wait) if wait and wait.isdigit() else min(2 ** attempt, MAX_BACKOFF)
            time.sleep(delay)
            continue
        if not response.ok:
            raise CfbdError(f"{endpoint} {params} -> {response.status_code}: {response.text[:200]}")
        payload = response.json()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        _FETCHED.add(path)
        return payload
    raise CfbdError(f"{endpoint} rate limited after {RETRIES} attempts")


def talent(year: int) -> pd.Series:
    """The 247Sports talent composite for each team: roster quality, accumulated
    over recruiting classes. Published before the season and fixed after it, so it
    is cached for good. Available from 2015; teams the composite does not cover
    (the service academies in some years) are simply absent."""
    rows = request("/talent", year=year)
    return pd.Series({r["team"]: float(r["talent"]) for r in rows if r.get("talent") is not None},
                     dtype=float, name="talent")


def returning(year: int) -> pd.DataFrame:
    """Returning production: the share of last season's production - measured in
    predicted points added, and in usage - that is still on the roster. Fixed
    before the season, so cached for good. Available from 2014.

    Columns ``percent_ppa`` and ``usage`` are shares in [0, 1]; a team that lost
    its passer, its rushers and its receivers to the portal is near zero.
    """
    rows = request("/player/returning", year=year)
    frame = pd.DataFrame(
        [{"team": r["team"], "percent_ppa": r.get("percentPPA"), "usage": r.get("usage")} for r in rows]
    )
    if frame.empty:
        return pd.DataFrame(columns=["percent_ppa", "usage"])
    return frame.set_index("team").astype(float).clip(lower=0.0, upper=1.0)


def games(
    year: int,
    *,
    season_type: str = "both",
    refresh: bool | None = None,
    completed_only: bool = True,
) -> pd.DataFrame:
    """All games for a season, normalized to the archive's column names.

    ``refresh=None`` means "if the season is still being played". A finished
    season is served from cache forever; the one in progress is re-fetched once
    per run, because the list of which games are final is exactly the thing
    that changes. Serving it from cache is how a completed game stayed
    "scheduled" - and out of the ratings - for days.

    ``completed_only`` keeps the rating path honest - a scheduled game carries
    no result. Pass False for the site's remaining-schedule view, where unplayed
    games are exactly the point.
    """
    if refresh is None:
        refresh = year == current_season()
    raw = request("/games", year=year, seasonType=season_type, refresh=refresh)
    rows = []
    for game in raw:
        played = bool(game.get("completed")) and game.get("homePoints") is not None \
            and game.get("awayPoints") is not None
        if completed_only and not played:
            continue
        rows.append(
            {
                "game_id": game["id"],
                "season": game["season"],
                "week": game["week"],
                "season_type": game["seasonType"],
                "start_date": game.get("startDate"),
                # The feed puts a placeholder time on games whose kickoff has not
                # been set, so a start time is only real when this is False.
                "start_time_tbd": bool(game.get("startTimeTBD")),
                "venue": game.get("venue"),
                "team1": game["awayTeam"],
                "team2": game["homeTeam"],
                "played": played,
                "pts1": float(game["awayPoints"]) if played else None,
                "pts2": float(game["homePoints"]) if played else None,
                "win1": (1.0 if game["awayPoints"] > game["homePoints"] else 0.0) if played else 0.0,
                "win2": (1.0 if game["homePoints"] > game["awayPoints"] else 0.0) if played else 0.0,
                "neutral": bool(game.get("neutralSite")),
                "class1": (game.get("awayClassification") or "").lower(),
                "class2": (game.get("homeClassification") or "").lower(),
                "conf1": game.get("awayConference"),
                "conf2": game.get("homeConference"),
                # Counts toward the conference standings. Not the same as two
                # teams sharing a conference: the Pac-12's week-13 flex games
                # are between members and do not count.
                "conference_game": bool(game.get("conferenceGame")),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # A game between two teams neither of which is FBS is not our business.
    keep = (frame["class1"] == FBS) | (frame["class2"] == FBS)
    return frame[keep].sort_values(["week", "game_id"]).reset_index(drop=True)


def team_box_scores(year: int, week: int, *, season_type: str = "regular", refresh: bool = False):
    """Per-team rushing, passing and turnovers for one week.

    Needed only by MRI Classic; MRI 2.0 runs on scores alone.
    """
    raw = request(
        "/games/teams", year=year, week=week, seasonType=season_type, refresh=refresh
    )
    rows = []
    for game in raw:
        for team in game.get("teams", []):
            stats = {s["category"]: s["stat"] for s in team.get("stats", [])}
            rows.append(
                {
                    "game_id": game["id"],
                    "team": team["team"],
                    "home_away": team.get("homeAway"),
                    "rush": _number(stats.get("rushingYards")),
                    "pass": _number(stats.get("netPassingYards")),
                    "turnovers": _number(stats.get("turnovers")),
                }
            )
    return pd.DataFrame(rows)


def lines(year: int, week: int, *, season_type: str = "regular", provider: str = "DraftKings",
          refresh: bool = False) -> pd.DataFrame:
    """Betting lines for one week, one book.

    ``spread_open`` matters as much as ``spread``: a model like this has a far
    better chance against an opening number than against a closing one.
    """
    raw = request("/lines", year=year, week=week, seasonType=season_type, refresh=refresh)
    rows = []
    for game in raw:
        for line in game.get("lines", []):
            if provider and line.get("provider") != provider:
                continue
            rows.append(
                {
                    "game_id": game["id"],
                    "season": game["season"],
                    "week": game["week"],
                    "team1": game["awayTeam"],
                    "team2": game["homeTeam"],
                    "pts1": game.get("awayScore"),
                    "pts2": game.get("homeScore"),
                    "provider": line.get("provider"),
                    # CFBD quotes the spread from the home team's perspective.
                    "spread": _number(line.get("spread")),
                    "spread_open": _number(line.get("spreadOpen")),
                    "over_under": _number(line.get("overUnder")),
                    "home_moneyline": _number(line.get("homeMoneyline")),
                    "away_moneyline": _number(line.get("awayMoneyline")),
                }
            )
    return pd.DataFrame(rows)


def fbs_teams(year: int, *, refresh: bool = False) -> pd.DataFrame:
    """Team metadata: conference, colors and logos for the site."""
    raw = request("/teams/fbs", year=year, refresh=refresh)
    return pd.DataFrame(
        [
            {
                "team_id": t["id"],
                "team": t["school"],
                "mascot": t.get("mascot"),
                "abbreviation": t.get("abbreviation"),
                "conference": t.get("conference"),
                "color": t.get("color"),
                "alt_color": t.get("alternateColor"),
                "logo": (t.get("logos") or [None])[0],
                "logo_dark": (t.get("logos") or [None, None])[-1],
            }
            for t in raw
        ]
    )


def venues(*, refresh: bool = False) -> pd.DataFrame:
    """Every stadium the API knows: where it is, how high, how big, and whether
    it has a roof. One call covers every season, so it is cached for good."""
    raw = request("/venues", refresh=refresh)
    rows = []
    for v in raw:
        rows.append(
            {
                "venue_id": v["id"],
                "name": v.get("name"),
                "city": v.get("city"),
                "state": v.get("state"),
                "latitude": _number(v.get("latitude")),
                "longitude": _number(v.get("longitude")),
                "elevation": _number(v.get("elevation")),
                "capacity": _number(v.get("capacity")),
                "dome": bool(v.get("dome")),
                "grass": bool(v.get("grass")),
                "timezone": v.get("timezone"),
            }
        )
    return pd.DataFrame(rows)


def team_homes(*, refresh: bool = False) -> pd.DataFrame:
    """Each team's listed home stadium, FCS included.

    The games feed only shows an FCS team at home when it hosts an FBS side,
    which some never do. This is where their travel starts from."""
    raw = request("/teams", refresh=refresh)
    rows = []
    for t in raw:
        loc = t.get("location") or {}
        if loc.get("id") is None:
            continue
        rows.append({"team": t["school"], "team_id": t["id"], "venue_id": loc["id"],
                     "classification": t.get("classification")})
    return pd.DataFrame(rows)


def calendar(year: int, *, refresh: bool = False) -> pd.DataFrame:
    return pd.DataFrame(request("/calendar", year=year, refresh=refresh))


def pool_non_fbs(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse every non-FBS opponent into one team, the way Classic expects."""
    frame = frame.copy()
    frame.loc[frame["class1"] != FBS, "team1"] = POOLED_FCS
    frame.loc[frame["class2"] != FBS, "team2"] = POOLED_FCS
    return frame


def _number(value):
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None
