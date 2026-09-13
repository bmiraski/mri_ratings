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
FBS = "fbs"


class CfbdError(RuntimeError):
    pass


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
    stem = endpoint.strip("/").replace("/", "_")
    if params:
        digest = hashlib.sha1(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()[:10]
        signature = "_".join(f"{k}{v}" for k, v in sorted(params.items()) if k != "year")
        signature = signature[:40] or digest
        year = params.get("year", "")
        stem = f"{stem}_{year}_{signature}"
    return CACHE_DIR / f"{stem}.json"


def request(endpoint: str, *, refresh: bool = False, **params):
    """Fetch an endpoint, using the on-disk cache unless ``refresh``."""
    params = {k: v for k, v in params.items() if v is not None}
    path = _cache_path(endpoint, params)

    if path.exists() and not refresh:
        return json.loads(path.read_text())

    headers = {"Authorization": f"Bearer {_api_key()}", "Accept": "application/json"}
    for attempt in range(3):
        response = requests.get(f"{BASE_URL}{endpoint}", params=params, headers=headers, timeout=60)
        if response.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        if not response.ok:
            raise CfbdError(f"{endpoint} {params} -> {response.status_code}: {response.text[:200]}")
        payload = response.json()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        return payload
    raise CfbdError(f"{endpoint} rate limited after three attempts")


def games(
    year: int,
    *,
    season_type: str = "both",
    refresh: bool = False,
    completed_only: bool = True,
) -> pd.DataFrame:
    """All games for a season, normalized to the archive's column names.

    ``completed_only`` keeps the rating path honest - a scheduled game carries
    no result. Pass False for the site's remaining-schedule view, where unplayed
    games are exactly the point.
    """
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
