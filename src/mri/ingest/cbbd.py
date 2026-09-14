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


def _is_final(game: dict) -> bool:
    """Whether a game was actually played.

    This is not the obvious test, and the obvious test is wrong. The football
    API returns null points for a game that has not happened; this one returns
    zero. So ``homePoints is not None`` accepts every cancelled, postponed and
    not-yet-played game and records it as a 0-0 tie.

    That is not a hypothetical. It put 1,465 phantom ties into the first build
    of the chain - 992 of them in 2020-21, where COVID cancelled 719 games and
    another 275 were never rescheduled. Each one entered the solve as real
    evidence that two teams are exactly equal.

    ``status`` carries the answer cleanly: every 0-0 row in every season checked
    is cancelled, postponed or scheduled, and every final has real points. The
    points test is kept as a fallback in case the field ever goes missing, but
    it is no longer the primary.
    """
    status = (game.get("status") or "").lower()
    if status:
        return status == "final"
    points = (game.get("homePoints"), game.get("awayPoints"))
    return all(p is not None for p in points) and points != (0, 0)


def _windows(season: int) -> list[tuple[str, str]]:
    """Month-long date ranges covering one season."""
    out = []
    for month, offset in SEASON_MONTHS:
        year = season - 1 if offset == 1 else season
        start = dt.date(year, month, 1)
        end = (start.replace(day=28) + dt.timedelta(days=8)).replace(day=1)
        out.append((start.isoformat(), end.isoformat()))
    return out


# Days after a window closes during which its results can still change: a game
# that tipped near midnight, or one finalized late.
GRACE = dt.timedelta(days=2)


def _is_live(start: str, end: str, today: dt.date) -> bool:
    """Whether a window's contents can still change, and so is worth re-fetching.

    The first rule here was "end is not in the past", which is true of every
    future month as well as the current one. In September that made all seven
    windows of the coming season look live, and a daily job spent seven API calls
    a day re-reading a schedule that had not changed. A window that has not begun
    cannot have results, so only one window is ever live: the one we are in.

    A window with no cache at all is fetched regardless - that check lives in the
    request layer, not here - so a month is always read once before it starts and
    re-read while it is running.

    The known gap: a game postponed out of one window and replayed inside an
    earlier, already-closed one is not picked up until something forces a
    refresh. Rare, and the alternative is re-fetching the whole season daily.
    """
    return dt.date.fromisoformat(start) <= today <= dt.date.fromisoformat(end) + GRACE


def games(season: int, *, refresh_last: bool = True, completed_only: bool = True) -> pd.DataFrame:
    """Every game of a season, normalized to the archive's column names.

    ``refresh_last`` re-fetches the window we are currently inside, which is the
    only one whose results can still change. Every other month comes from cache.
    """
    windows = _windows(season)
    today = dt.date.today()
    rows, seen = [], set()

    for start, end in windows:
        payload = request(
            "/games",
            season=season,
            startDateRange=start,
            endDateRange=end,
            refresh=refresh_last and _is_live(start, end, today),
        )
        if len(payload) >= PAGE_CAP:
            print(f"  warning: {start}..{end} hit the {PAGE_CAP}-row cap; games may be missing")

        for game in payload:
            if game["id"] in seen:
                continue
            seen.add(game["id"])
            played = _is_final(game)
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
                    "status": game.get("status"),
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
                # ESPN's id for the school, which is what its logo CDN is keyed
                # by. The basketball feed carries no logo URL of its own.
                "source_id": t.get("sourceId"),
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

    The shape is not football's and the first version of this assumed it was.
    Football nests both sides under ``teams``; this endpoint returns one flat
    row per team-game with ``teamStats`` and ``opponentStats`` beside each
    other, and the counts live one level down again - ``rebounds.total``,
    ``turnovers.total``, not bare integers. Asking for the football shape
    produced an empty frame and no error at all, which is the failure worth
    naming: every caller saw "this season has no box scores" and believed it.
    """
    rows = []
    today = dt.date.today()
    for start, end in _windows(season):
        payload = request(
            "/games/teams",
            season=season,
            startDateRange=start,
            endDateRange=end,
            refresh=refresh_last and _is_live(start, end, today),
        )
        for entry in payload:
            stats = entry.get("teamStats") or {}
            rows.append(
                {
                    "game_id": entry.get("gameId"),
                    "season": entry.get("season"),
                    "start_date": entry.get("startDate"),
                    "team": entry.get("team"),
                    "opponent": entry.get("opponent"),
                    "home_away": "home" if entry.get("isHome") else "away",
                    "neutral": bool(entry.get("neutralSite")),
                    "points": _number(stats.get("points")),
                    "rebounds": _number(stats.get("rebounds")),
                    "turnovers": _number(stats.get("turnovers")),
                }
            )
    return pd.DataFrame(rows)


def _number(value):
    """Unwrap the counts, which arrive as objects rather than integers.

    ``{"offensive": 11, "defensive": 25, "total": 36}`` for rebounds,
    ``{"total": 6, "teamTotal": 1}`` for turnovers. "total" is the one the
    formula wants in both cases.
    """
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        value = value.get("total")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def classic_table(season: int, *, refresh_last: bool = True) -> pd.DataFrame:
    """A game log in the archive's shape, so MRI Basketball Classic can run on it.

    Classic needs rebounds and turnovers for both sides of every game, which the
    games feed does not carry and the box-score feed does. This joins them into
    the exact columns the 2019-20 workbook used - team1/team2, pts, reb, to, win
    - with team1 the visitor, so the ported formula runs on current seasons the
    same way it runs on the archive.

    Games whose box score is missing a count are dropped rather than defaulted.
    A zero would be read as "grabbed no rebounds", which is a result rather than
    an absence, and Classic would rate it as one.

    Only games the games feed calls final are kept, and that filter is doing real
    work: the box-score feed emits rows for games that never happened, with zeros
    throughout and nothing to mark them. Delaware at Towson on 2022-01-28 is one -
    "scheduled" in the games feed, a 0-0 final here. Classic reads a 0-0 game as a
    loss for both sides, so a single phantom row moved Towson's rating by 1.7
    points and nine other teams' through the opponent chain. This is the same
    trap as the one in ``games`` wearing different clothes: a feed that answers
    with zeros instead of nulls, and a caller that believes it.
    """
    box = team_box_scores(season, refresh_last=refresh_last)
    if box.empty:
        return box

    real = set(games(season, refresh_last=refresh_last)["game_id"])
    box = box[box["game_id"].isin(real)]
    box = box.dropna(subset=["points", "rebounds", "turnovers"])
    rows = []
    for game_id, pair in box.groupby("game_id"):
        if len(pair) != 2:
            continue
        # team1 is the visitor, matching the workbooks. On a neutral court the
        # feed still marks one side home; the ordering is then arbitrary and
        # harmless, since Classic has no home-court term.
        away = pair[pair["home_away"] == "away"]
        home = pair[pair["home_away"] == "home"]
        if len(away) != 1 or len(home) != 1:
            away, home = pair.iloc[[0]], pair.iloc[[1]]
        a, h = away.iloc[0], home.iloc[0]
        rows.append(
            {
                "game_id": game_id,
                "season": a["season"],
                "start_date": a["start_date"],
                "team1": a["team"], "team2": h["team"],
                "pts1": float(a["points"]), "pts2": float(h["points"]),
                "reb1": float(a["rebounds"]), "reb2": float(h["rebounds"]),
                "to1": float(a["turnovers"]), "to2": float(h["turnovers"]),
                "win1": 1.0 if a["points"] > h["points"] else 0.0,
                "win2": 1.0 if h["points"] > a["points"] else 0.0,
                "neutral": bool(a["neutral"]),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("start_date").reset_index(drop=True) if not frame.empty else frame
