"""Assemble the table MRI Classic needs from the API.

MRI 2.0 runs on scores alone, so the games feed is enough for it. Classic also
wants rushing yards, net passing yards and turnovers per side, and those live in
the box-score endpoint, which is scoped one week at a time.

The output matches the Excel archive's column names exactly, so the same
``ratings.classic`` code runs over either source without knowing the difference.
Non-FBS opponents are pooled back into "Non D1A" here, because that is what
Classic did and Classic is meant to stay frozen. MRI 2.0 keeps them separate.
"""

from __future__ import annotations

import pandas as pd

from . import cfbd, registry

STAT_COLUMNS = ["rush1", "rush2", "pass1", "pass2", "to1", "to2"]


def season_weeks(year: int) -> list[tuple[str, int]]:
    """Weeks with completed games, as (season_type, week) pairs."""
    games = cfbd.games(year)
    if games.empty:
        return []
    pairs = games[["season_type", "week"]].drop_duplicates()
    return [(row.season_type, int(row.week)) for row in pairs.itertuples()]


def classic_table(year: int, *, refresh_last_week: bool = True) -> pd.DataFrame:
    """Games for one season with the per-side stats Classic needs.

    ``refresh_last_week`` re-fetches the most recent week, which is the only one
    whose box scores can still change. Everything earlier is served from cache.
    """
    games = cfbd.games(year)
    if games.empty:
        return games

    weeks = season_weeks(year)
    latest = weeks[-1] if weeks else None

    frames = []
    for season_type, week in weeks:
        refresh = refresh_last_week and (season_type, week) == latest
        frames.append(
            cfbd.team_box_scores(year, week, season_type=season_type, refresh=refresh)
        )
    box = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if box.empty:
        return games.assign(**{c: 0.0 for c in STAT_COLUMNS})

    away = box[box["home_away"] == "away"].set_index("game_id")
    home = box[box["home_away"] == "home"].set_index("game_id")

    merged = games.copy()
    merged["rush1"] = merged["game_id"].map(away["rush"])
    merged["pass1"] = merged["game_id"].map(away["pass"])
    merged["to1"] = merged["game_id"].map(away["turnovers"])
    merged["rush2"] = merged["game_id"].map(home["rush"])
    merged["pass2"] = merged["game_id"].map(home["pass"])
    merged["to2"] = merged["game_id"].map(home["turnovers"])

    missing = merged[STAT_COLUMNS].isna().all(axis=1).sum()
    if missing:
        print(f"  note: {missing} of {len(merged)} games have no box score; stats zero-filled")
    merged[STAT_COLUMNS] = merged[STAT_COLUMNS].fillna(0.0)

    for column in ("team1", "team2"):
        merged[column] = [registry.resolve(n, n) for n in merged[column]]
    return cfbd.pool_non_fbs(merged)
