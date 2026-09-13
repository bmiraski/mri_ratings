"""Client for the College Basketball Data API.

Same operator as the football API, and the same key works for both, but the
shape of the problem is different in three ways that matter:

*Scale.* 364 D1 teams against football's 138, and roughly 6,000 games a season
against 900. The ratings maths is unbothered - a 365-column normal-equations
solve is trivial - but every name has to match, and there are two and a half
times as many chances to get one wrong.

*Pagination.* The games endpoint caps at 3,000 rows and ignores page, skip and
offset; only ``startDateRange``/``endDateRange`` narrows it. So a season is
fetched a month at a time, which also makes the cache naturally incremental:
finished months never need re-fetching.

*Season labels.* ``season=2025`` means the 2024-25 season, labelled "20242025".
The year is the one the season ends in, which matches Ben's own file naming
(MRIBasketball201920 is season 2020).

Normalization matches the football path and the workbooks: ``team1`` is the
visitor, ``team2`` the host. Verified against the 2019-20 workbook, where the
visitor wins 35.3% and loses by 5.2 on average.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from . import cfbd

BASE_URL = "https://api.collegebasketballdata.com"
PAGE_CAP = 3000

# A season runs November through early April; the API labels it by the year it
# ends in. Windows are monthly, which keeps every request well under the cap.
SEASON_MONTHS = [(11, 1), (12, 1), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2)]


def request(endpoint: str, *, refresh: bool = False, **params):
    """Same caching request path as football, pointed at the basketball host."""
    original = cfbd.BASE_URL
    cfbd.BASE_URL = BASE_URL
    try:
        return cfbd.request(endpoint, refresh=refresh, **params)
    finally:
        cfbd.BASE_URL = original


def _windows(season: int) -> list[tuple[str, str]]:
    """Month-long date ranges covering one season."""
    out = []
    for month, offset in SEASON_MONTHS:
        year = season - 1 if offset == 1 else season
        start = dt.date(year, month, 1)
        end = (start.replace(day=28) + dt.timedelta(days=8)).replace(day=1)
        out.append((start.isoformat(), end.isoformat()))
    return out


def games(season: int, *, refresh_last: bool = True, completed_only: bool = True) -> pd.DataFrame:
    """Every game of a season, normalized to the archive's column names.

    ``refresh_last`` re-fetches the final window, which is the only one whose
    results can still change. Earlier months come from cache.
    """
    windows = _windows(season)
    today = dt.date.today()
    rows, seen = [], set()

    for index, (start, end) in enumerate(windows):
        # A window already in the past cannot change; only refresh a live one.
        live = dt.date.fromisoformat(end) >= today
        payload = request(
            "/games",
            season=season,
            startDateRange=start,
            endDateRange=end,
            refresh=refresh_last and live,
        )
        if len(payload) >= PAGE_CAP:
            print(f"  warning: {start}..{end} hit the {PAGE_CAP}-row cap; games may be missing")

        for game in payload:
            if game["id"] in seen:
                continue
            seen.add(game["id"])
            played = game.get("homePoints") is not None and game.get("awayPoints") is not None
            if completed_only and not played:
                continue
            rows.append(
                {
                    "game_id": game["id"],
                    "season": game["season"],
                    "season_type": game.get("seasonType"),
                    "start_date": game.get("startDate"),
                    "team1": game["awayTeam"],
                    "team2": game["homeTeam"],
                    "played": played,
                    "pts1": float(game["awayPoints"]) if played else None,
                    "pts2": float(game["homePoints"]) if played else None,
                    "win1": (1.0 if game["awayPoints"] > game["homePoints"] else 0.0) if played else 0.0,
                    "win2": (1.0 if game["homePoints"] > game["awayPoints"] else 0.0) if played else 0.0,
                    "neutral": bool(game.get("neutralSite")),
                    "conf1": game.get("awayConference"),
                    "conf2": game.get("homeConference"),
                    "tournament": game.get("tournament"),
                    "game_type": game.get("gameType"),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("start_date").reset_index(drop=True)


def teams(season: int, *, refresh: bool = False) -> pd.DataFrame:
    """D1 team metadata: conference, colours, abbreviation."""
    raw = request("/teams", season=season, refresh=refresh)
    return pd.DataFrame(
        [
            {
                "team_id": t["id"],
                "team": t["school"],
                "display": t.get("displayName"),
                "abbreviation": t.get("abbreviation"),
                "mascot": t.get("mascot"),
                "conference": t.get("conference"),
                "color": t.get("primaryColor"),
                "alt_color": t.get("secondaryColor"),
            }
            for t in raw
        ]
    )


def team_box_scores(season: int, *, refresh_last: bool = True) -> pd.DataFrame:
    """Per-team rebounds and turnovers, which is what MRI Basketball needs.

    Football's Classic wanted rushing and passing; basketball's wants rebound
    differential and turnover differential, so this is the equivalent call.
    """
    rows = []
    today = dt.date.today()
    for start, end in _windows(season):
        live = dt.date.fromisoformat(end) >= today
        payload = request(
            "/games/teams",
            season=season,
            startDateRange=start,
            endDateRange=end,
            refresh=refresh_last and live,
        )
        for entry in payload:
            for team in entry.get("teams", []):
                stats = team.get("stats", team)
                rows.append(
                    {
                        "game_id": entry.get("gameId", entry.get("id")),
                        "team": team.get("team"),
                        "home_away": "home" if team.get("isHome") else "away",
                        "rebounds": _number(stats.get("totalRebounds") or stats.get("rebounds")),
                        "turnovers": _number(stats.get("turnovers")),
                    }
                )
    return pd.DataFrame(rows)


def _number(value):
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        value = value.get("total")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
