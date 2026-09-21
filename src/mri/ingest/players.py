"""Player-level football data, for the Heisman work.

The API's design does most of the work here: one request returns the whole
division. Every FBS passer of a season is one call of about a megabyte and a
half, not one call per player, so "every player" is five calls a season for the
counting stats and one each for efficiency (PPA) and usage.

Three things to know about what comes back.

*Coverage begins around 2009.* 2005 has six quarterbacks in it. 2008 has 288 and
2009 has 446, which is the division, so the usable history for anything built on
player stats is 2009 on. (Advanced numbers - PPA and usage - start in 2013.)

*Counting stats are regular-season only, because the request says so.* The
Heisman is voted on before the bowls and the playoff, and ``seasonType=regular``
is what makes a season's totals the totals the voters saw.

*PPA is not.* Its endpoint takes no season type, so a season's PPA includes its
bowl and playoff games. For a player it is a small distortion; for the handful who
had a monster bowl game it is not nothing, and anything that uses it should know.

Rows come back one per player per statistic; ``season_table`` makes them one per
player per team, wide.
"""

from __future__ import annotations

import pandas as pd

from . import cfbd

# (category, statType) -> column. Everything else the feed returns is ignored.
COLUMNS = {
    ("passing", "ATT"): "pass_att", ("passing", "COMPLETIONS"): "pass_cmp", ("passing", "YDS"): "pass_yds",
    ("passing", "TD"): "pass_td", ("passing", "INT"): "pass_int",
    ("rushing", "CAR"): "rush_car", ("rushing", "YDS"): "rush_yds", ("rushing", "TD"): "rush_td",
    ("receiving", "REC"): "rec_rec", ("receiving", "YDS"): "rec_yds", ("receiving", "TD"): "rec_td",
    ("defensive", "TOT"): "def_tot", ("defensive", "SOLO"): "def_solo", ("defensive", "SACKS"): "def_sacks",
    ("defensive", "TFL"): "def_tfl", ("defensive", "PD"): "def_pd", ("defensive", "TD"): "def_td",
    ("interceptions", "INT"): "def_int",
}
CATEGORIES = ("passing", "rushing", "receiving", "defensive", "interceptions")
FIRST_SEASON = 2009          # the first season with the whole division in the feed
FIRST_ADVANCED = 2013        # PPA and usage begin here

# A player is worth keeping if he did enough of something for a voter to have noticed. The
# thresholds are far below anything a Heisman candidate reaches; they exist to keep the table
# to a few thousand rows a season and not the ten thousand names the feed carries.
KEEP = {"pass_att": 100, "rush_car": 75, "rec_rec": 30, "def_tot": 50, "def_sacks": 6, "def_tfl": 10, "def_int": 3}


def _refresh(year: int, refresh: bool | None) -> bool:
    return (year == cfbd.current_season()) if refresh is None else refresh


def category_rows(year: int, category: str, *, refresh: bool | None = None, end_week: int | None = None) -> list[dict]:
    """One category for the whole division. ``end_week`` gives the totals through that week."""
    return cfbd.request("/stats/player/season", year=year, category=category, seasonType="regular",
                        endWeek=end_week, refresh=_refresh(year, refresh) if end_week is None else False)


def ppa_rows(year: int, *, refresh: bool | None = None) -> list[dict]:
    return cfbd.request("/ppa/players/season", year=year, refresh=_refresh(year, refresh))


def usage_rows(year: int, *, refresh: bool | None = None) -> list[dict]:
    return cfbd.request("/player/usage", year=year, refresh=_refresh(year, refresh))


def season_table(year: int, *, refresh: bool | None = None, keep_all: bool = False) -> pd.DataFrame:
    """One row per player per team for a season: counting stats, and PPA and usage where they exist."""
    stats: dict[tuple[str, str], dict] = {}
    for category in CATEGORIES:
        for r in category_rows(year, category, refresh=refresh):
            column = COLUMNS.get((r["category"], r["statType"]))
            if column is None:
                continue
            key = (str(r["playerId"]), r["team"])
            row = stats.setdefault(key, {"season": year, "player_id": key[0], "player": r["player"], "team": r["team"],
                                         "conference": r.get("conference"), "position": r.get("position")})
            try:
                row[column] = float(r["stat"])
            except (TypeError, ValueError):
                continue
    frame = pd.DataFrame(list(stats.values()))
    if frame.empty:
        return frame
    for column in COLUMNS.values():
        if column not in frame:
            frame[column] = 0.0
    frame[list(COLUMNS.values())] = frame[list(COLUMNS.values())].fillna(0.0)

    if year >= FIRST_ADVANCED:
        ppa = {(str(r["id"]), r["team"]): r for r in ppa_rows(year, refresh=refresh)}
        use = {(str(r["id"]), r["team"]): r for r in usage_rows(year, refresh=refresh)}
        keys = list(zip(frame["player_id"], frame["team"]))
        frame["ppa_avg"] = [(ppa.get(k) or {}).get("averagePPA", {}).get("all") for k in keys]
        frame["ppa_total"] = [(ppa.get(k) or {}).get("totalPPA", {}).get("all") for k in keys]
        frame["ppa_pass_total"] = [(ppa.get(k) or {}).get("totalPPA", {}).get("pass") for k in keys]
        frame["ppa_rush_total"] = [(ppa.get(k) or {}).get("totalPPA", {}).get("rush") for k in keys]
        frame["usage"] = [(use.get(k) or {}).get("usage", {}).get("overall") for k in keys]
        # A player the stat feeds miss but PPA has (a two-way or trick-play player) is rarely a candidate
        # for anything; the counting-stat rows are the spine and PPA is joined onto them.

    # From 2022 the feed carries FCS players too, which doubles the season and is not who the
    # Heisman is voted among. Every season is cut to the FBS teams of that season, so the
    # seasons are comparable.
    fbs = set(cfbd.fbs_teams(year)["team"])
    frame = frame[frame["team"].isin(fbs)].reset_index(drop=True)

    if keep_all:
        return frame
    mask = pd.Series(False, index=frame.index)
    for column, floor in KEEP.items():
        mask |= frame[column] >= floor
    return frame[mask].reset_index(drop=True)


OFFENSE = ("passing", "rushing", "receiving")
# Below this a player has not done enough of anything by mid-season for a voter to have noticed,
# and the table stays a few thousand rows a snapshot.
WEEKLY_KEEP = {"pass_att": 30, "rush_car": 25, "rec_rec": 10}


def weekly_table(year: int, week: int | None) -> pd.DataFrame:
    """Offensive totals through a given week: one row per notable FBS player.

    ``week=None`` is the season to date, re-fetched once per run for the season in progress.

    Three calls, and no defence: the snapshots exist to score offensive candidates in the middle of
    a season, and the defenders who have ever mattered are not among them.
    """
    stats: dict[tuple[str, str], dict] = {}
    for category in OFFENSE:
        for r in category_rows(year, category, end_week=week):
            column = COLUMNS.get((r["category"], r["statType"]))
            if column is None:
                continue
            key = (str(r["playerId"]), r["team"])
            row = stats.setdefault(key, {"season": year, "week": week, "player_id": key[0], "player": r["player"],
                                         "team": r["team"], "position": r.get("position")})
            try:
                row[column] = float(r["stat"])
            except (TypeError, ValueError):
                continue
    frame = pd.DataFrame(list(stats.values()))
    if frame.empty:
        return frame
    columns = [c for (cat, _), c in COLUMNS.items() if cat in OFFENSE]
    for column in columns:
        if column not in frame:
            frame[column] = 0.0
    frame[columns] = frame[columns].fillna(0.0)
    fbs = set(cfbd.fbs_teams(year)["team"])
    frame = frame[frame["team"].isin(fbs)]
    mask = pd.Series(False, index=frame.index)
    for column, floor in WEEKLY_KEEP.items():
        mask |= frame[column] >= floor
    return frame[mask].reset_index(drop=True)
