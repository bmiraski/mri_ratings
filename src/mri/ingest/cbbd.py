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

    ``status`` carries the answer cleanly in the seasons the chain rates: every
    0-0 row there is cancelled, postponed or scheduled, and every final has real
    points. It does not carry it in the older ones. 2011-12 has 76 rows and
    2012-13 has 60 that say "final" and 0-0 in the same breath, and trusting
    status alone let every one of them back in - as a loss for both sides, which
    is how Classic reads a tie.

    So the score is checked as well as the status, and equal scores are rejected
    rather than only 0-0. Basketball plays overtime until somebody wins; a game
    in this feed with the same number on both sides did not happen, whatever
    that number is.
    """
    points = (game.get("homePoints"), game.get("awayPoints"))
    if any(p is None for p in points) or points[0] == points[1]:
        return False
    status = (game.get("status") or "").lower()
    return status == "final" if status else True


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


def _split_to_fit(season: int, refresh_last: bool, today: dt.date,
                  depth: int = 4) -> list[tuple[str, str]]:
    """Month windows, halved until each fits under the row cap.

    This endpoint returns one row per team per game - two rows a game - so it
    reaches the 3,000-row cap in a busy month where the games feed, at one row
    each, never comes close. The cap is silent: the response just stops, and
    nothing distinguishes a month with exactly 3,000 rows from a month that had
    more.

    It was not hypothetical. January 2016 and both November and January of
    2025-26 came back at exactly 3,000, and the basketball Classic game log for
    2025-26 was 121 games short as a result - which means the Excel workbook
    built from it was too, agreeing perfectly with a Python calculation that was
    reading the same truncated data.

    So each window is fetched, and split in half and refetched whenever it comes
    back at the cap. Bounded depth, because a window that still overflows when
    it is under two days wide is a different problem and should be visible
    rather than recursed into forever.
    """
    out = []
    pending = [(start, end, depth) for start, end in _windows(season)]
    while pending:
        start, end, budget = pending.pop(0)
        payload = request(
            "/games/teams",
            season=season,
            startDateRange=start,
            endDateRange=end,
            refresh=refresh_last and _is_live(start, end, today),
        )
        first, last = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
        span = (last - first).days
        if len(payload) < PAGE_CAP or span <= 1 or budget <= 0:
            if len(payload) >= PAGE_CAP:
                print(f"  warning: {start}..{end} still at the {PAGE_CAP}-row cap")
            out.append((start, end))
            continue
        middle = (first + dt.timedelta(days=span // 2)).isoformat()
        pending = [(start, middle, budget - 1), (middle, end, budget - 1)] + pending
    return out


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
    for start, end in _split_to_fit(season, refresh_last, today):
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

    The games feed is the spine and the box scores are joined onto it, rather
    than the other way round, because the two do not cover the same games. The
    box-score feed is missing about one game in eight in 2004-05 and 2011-12,
    one in twenty in the two seasons after 2004-05, and under one in a hundred
    everywhere since 2007-08. Building the log from the box scores alone dropped
    those games entirely, which cost a typical 2011-12 team four games off its
    record - a team that went 32-7 was published as 28-6. Building it from the
    games feed keeps the result, the schedule and the game credit, and leaves
    the rebound and turnover counts empty for the games that have none.

    Empty, not zero. A zero would be read as "grabbed no rebounds", which is a
    result rather than an absence, and Classic would rate it as one; blanks are
    excluded from the per-game statistics by their own denominator instead (see
    ``classic.compute``).

    Only games the games feed calls final are kept, and that filter is doing real
    work: the box-score feed emits rows for games that never happened, with zeros
    throughout and nothing to mark them. Delaware at Towson on 2022-01-28 is one -
    "scheduled" in the games feed, a 0-0 final here. Classic reads a 0-0 game as a
    loss for both sides, so a single phantom row moved Towson's rating by 1.7
    points and nine other teams' through the opponent chain. This is the same
    trap as the one in ``games`` wearing different clothes: a feed that answers
    with zeros instead of nulls, and a caller that believes it.
    """
    feed = games(season, refresh_last=refresh_last)
    if feed.empty:
        return pd.DataFrame()
    feed = feed[feed["played"]]
    if feed.empty:
        return pd.DataFrame()

    box = team_box_scores(season, refresh_last=refresh_last)
    stats: dict[object, dict[str, dict]] = {}
    if not box.empty:
        box = box.dropna(subset=["rebounds", "turnovers"])
        for game_id, pair in box.groupby("game_id"):
            if len(pair) != 2:
                continue
            by_team = {str(row["team"]): row for _, row in pair.iterrows()}
            if len(by_team) == 2:
                stats[game_id] = by_team

    def counts(game_id, team: str) -> tuple[float | None, float | None]:
        side = stats.get(game_id, {}).get(str(team))
        if side is None:
            return None, None
        return float(side["rebounds"]), float(side["turnovers"])

    rows = []
    for game in feed.itertuples():
        # team1 is the visitor, matching the workbooks. On a neutral court the
        # feed still marks one side home; the ordering is then arbitrary and
        # harmless, since Classic has no home-court term.
        reb1, to1 = counts(game.game_id, game.team1)
        reb2, to2 = counts(game.game_id, game.team2)
        if (reb1 is None) != (reb2 is None):
            # One side of a game and not the other is not a usable half-game.
            reb1 = reb2 = to1 = to2 = None
        rows.append(
            {
                "game_id": game.game_id,
                "season": game.season,
                "start_date": game.start_date,
                "team1": game.team1, "team2": game.team2,
                "pts1": float(game.pts1), "pts2": float(game.pts2),
                "reb1": reb1, "reb2": reb2,
                "to1": to1, "to2": to2,
                "win1": float(game.win1), "win2": float(game.win2),
                "neutral": bool(game.neutral),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("start_date").reset_index(drop=True) if not frame.empty else frame


def _win_shares(value) -> float | None:
    """Win shares arrive as ``{"offensive", "defensive", "total", "totalPer40"}``."""
    if isinstance(value, dict):
        value = value.get("total")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def player_seasons(season: int, *, refresh: bool = False) -> pd.DataFrame:
    """One row per player per team per season: minutes, win shares and the rest.

    A single call returns the whole division - about ten thousand rows - so a
    season costs one request, not one per team. Fixed once the season is over.
    """
    rows = []
    for r in request("/stats/player/season", season=season, refresh=refresh):
        rows.append({
            "season": season,
            "athlete_id": r.get("athleteId"),
            "name": r.get("name"),
            "team": r.get("team"),
            "conference": r.get("conference"),
            "games": r.get("games"),
            "starts": r.get("starts"),
            "minutes": r.get("minutes"),
            "win_shares": _win_shares(r.get("winShares")),
            "net_rating": r.get("netRating"),
            "usage": r.get("usage"),
            "porpag": r.get("PORPAG"),
        })
    frame = pd.DataFrame(rows)
    return frame.dropna(subset=["athlete_id", "team"]) if not frame.empty else frame


def rosters(season: int, *, refresh: bool = False) -> pd.DataFrame:
    """Who is on each team's roster for a season: (team, athlete_id, name).

    Not populated for a coming season until the schools post their rosters, and
    the feed says nothing then - an empty ``players`` list, not an error - so an
    empty frame from this function means "not yet", never "nobody".
    """
    rows = []
    for team in request("/teams/roster", season=season, refresh=refresh):
        for p in team.get("players") or []:
            rows.append({"season": season, "team": team.get("team"), "conference": team.get("conference"),
                         "athlete_id": p.get("id"), "name": p.get("name")})
    return pd.DataFrame(rows, columns=["season", "team", "conference", "athlete_id", "name"])


def recruits(year: int, *, refresh: bool = False) -> pd.DataFrame:
    """A recruiting class: each commit's stars, rating and school.

    ``year`` is the class - the year the players graduate high school - so the
    class of 2026 plays in the 2026-27 season, which this project calls 2027.
    """
    rows = []
    for r in request("/recruiting/players", year=year, refresh=refresh):
        committed = r.get("committedTo") or {}
        rows.append({"year": year, "name": r.get("name"), "stars": r.get("stars"), "rating": r.get("rating"),
                     "team": committed.get("name"), "position": r.get("position")})
    return pd.DataFrame(rows, columns=["year", "name", "stars", "rating", "team", "position"])


def draft_picks(year: int, *, refresh: bool = False) -> pd.DataFrame:
    """NBA draft picks for a year, with the athlete id that links them to a college roster."""
    rows = [{"year": year, "athlete_id": r.get("athleteId"), "overall": r.get("overall"), "name": r.get("name"),
             "college_id": r.get("sourceTeamCollegeId")}
            for r in request("/draft/picks", year=year, refresh=refresh)]
    return pd.DataFrame(rows, columns=["year", "athlete_id", "overall", "name", "college_id"])
