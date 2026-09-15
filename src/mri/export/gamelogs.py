"""Game logs for past seasons: one page per team per season.

The team page carries a season-by-season summary. This is the layer under it -
every game a team played in a given year, on its own page, so a single season is
linkable rather than buried in a long accordion on a page about something else.

What each era can show differs, and the pages say so rather than presenting a
uniform table with quietly empty columns:

*2020 onward* has a rated opponent for every game, so each row carries the
margin the ratings imply and how far the result beat it - the same "performance"
number the live season shows. That is the interesting column and it exists only
where MRI 2.0 rated the season.

*2003-2019 football* comes from the workbook game logs, which pooled every
non-FBS opponent into "Non D1A". About one row in nine names no real opponent as
a result. Nothing can recover it from this source, so those rows say "non-FBS
opponent" rather than printing a placeholder as though it were a school.

*The basketball workbook seasons* carry no dates at all, and neither do the
football workbooks before 2018 - the date column exists and is empty for every
row of 2003-2017. Those games are listed in workbook order rather than
chronologically, and the page says so instead of implying a sequence it cannot
support.

The expected margin uses each season's *final* ratings, not the ratings as they
stood on the day. That is a deliberate and visible difference from the live
season, which prices each week from the weeks before it. For a finished season
the final rating is the better estimate of how good a team actually was, and
hindsight is the point of an archive; the live page is where a no-lookahead
number belongs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PARQUET = Path(__file__).resolve().parents[3] / "data" / "parquet"
ARCHIVE_BB = Path(__file__).resolve().parents[3] / "data" / "archive-bb"

POOLED = "Non D1A"


def _read(name: str) -> pd.DataFrame:
    path = PARQUET / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _rows_from(games: pd.DataFrame, canonical, power: dict, home_field: float,
               *, dated: bool) -> dict[str, list[dict]]:
    """Turn a season's games into per-team rows, both sides of every game."""
    out: dict[str, list[dict]] = {}
    if games.empty:
        return out

    neutral = games["neutral"] if "neutral" in games else pd.Series(False, index=games.index)
    for order, (_, game) in enumerate(games.iterrows()):
        away, home = canonical(game["team1"]), canonical(game["team2"])
        if pd.isna(game.get("pts1")) or pd.isna(game.get("pts2")):
            continue
        away_pts, home_pts = float(game["pts1"]), float(game["pts2"])
        is_neutral = bool(neutral.get(game.name, False))
        edge = 0.0 if is_neutral else home_field

        for team, opponent, site, scored, allowed, expected in (
            (home, away, "n" if is_neutral else "vs", home_pts, away_pts,
             (power.get(home), power.get(away), edge)),
            (away, home, "n" if is_neutral else "at", away_pts, home_pts,
             (power.get(away), power.get(home), -edge if not is_neutral else 0.0)),
        ):
            mine, theirs, venue = expected
            margin = scored - allowed
            row = {
                "order": order,
                "opponent": opponent,
                "pooled": opponent == POOLED,
                "site": site,
                "scored": int(scored),
                "allowed": int(allowed),
                "won": margin > 0,
                "margin": int(margin),
                "expected": None,
                "performance": None,
            }
            if mine is not None and theirs is not None:
                predicted = mine - theirs + venue
                row["expected"] = round(predicted, 1)
                row["performance"] = round(margin - predicted, 1)
            when = str(game.get("start_date") or game.get("date") or "")[:10]
            row["when"] = when if dated and when and when != "None" else None
            out.setdefault(team, []).append(row)

    # Chronological where dates exist, workbook order where they do not. The
    # feed does not hand these back in order - a 2025 log arrived with a bowl
    # game sitting between week one and week sixteen.
    for rows in out.values():
        rows.sort(key=lambda r: (r["when"] or "", r["order"]))
    return out


def football(current: int) -> dict[int, dict[str, list[dict]]]:
    """{season: {team: [game rows]}} for every completed football season."""
    from ..ingest import registry

    def canonical(name):
        return registry.resolve(name, str(name))

    logs: dict[int, dict[str, list[dict]]] = {}

    modern_games = _read("current_games")
    modern_ratings = _read("current_ratings")
    if not modern_games.empty:
        for season, chunk in modern_games.groupby("season"):
            season = int(season)
            if season >= current:
                continue
            rated = modern_ratings[modern_ratings["season"] == season]
            power = {canonical(r["team"]): float(r["power"]) for _, r in rated.iterrows()}
            logs[season] = _rows_from(chunk, canonical, power, _home_field(season), dated=True)

    archive = _read("archive_games")
    if not archive.empty:
        for season, chunk in archive.groupby("season"):
            season = int(season)
            if season >= current:
                continue
            # No MRI 2.0 for these years by design, so no expected column.
            logs.setdefault(season, {}).update(
                _rows_from(chunk, canonical, {}, 0.0, dated=True)
            )
    return logs


def basketball(current: int | None) -> dict[int, dict[str, list[dict]]]:
    """{season: {team: [game rows]}} for every completed basketball season."""
    from ..ingest import bb_registry as registry

    logs: dict[int, dict[str, list[dict]]] = {}

    games = _read("bb_games")
    ratings = _read("bb_ratings")
    if not games.empty:
        for season, chunk in games.groupby("season"):
            season = int(season)
            if current is not None and season >= current:
                continue

            def canonical(name, _s=season):
                return registry.resolve(name, str(name), season=_s)

            rated = ratings[ratings["season"] == season]
            power = {canonical(r["team"]): float(r["power"]) for _, r in rated.iterrows()}
            logs[season] = _rows_from(chunk, canonical, power, 2.9, dated=True)

    # 2004-05 to 2017-18, apart from the seasons Ben published himself, are
    # computed from the API rather than a workbook, so their game logs come from
    # the same place - and unlike the workbook seasons they carry dates.
    computed = _read("bb_classic")
    if not computed.empty:
        from ..ingest import bb_gamelog

        power_by_season = {}
        for season in sorted(int(x) for x in computed["season"].unique()):
            if season in logs or (current is not None and season >= current):
                continue

            def canonical(name, _s=season):
                return registry.resolve(name, str(name), season=_s)

            table = bb_gamelog.for_season(season)
            if table.empty:
                continue
            # No MRI 2.0 for these years, so no expected margin, exactly as for
            # football's Classic seasons.
            logs[season] = _rows_from(table, canonical, power_by_season, 0.0, dated=True)

    if ARCHIVE_BB.exists():
        from ..ingest.archive import read_basketball_archive

        for season, data in read_basketball_archive(ARCHIVE_BB).items():
            season = int(season)
            if season in logs or (current is not None and season >= current):
                continue

            def canonical(name, _s=season):
                return registry.resolve(name, str(name), season=_s)

            # These workbooks carry no dates, so the rows keep workbook order.
            logs[season] = _rows_from(data.games, canonical, {}, 0.0, dated=False)
    return logs


def _home_field(season: int) -> float:
    """Home field for a completed season, as the site's own ratings estimated it.

    Not looked up per season because the stored tables do not carry it; 2.5 is
    where the football fits have sat since 2020 and the number only shifts a
    row's expected margin by a point either way. The live pages use the fitted
    value; this is an archive of finished games.
    """
    return 2.5


def prune(logs: dict[int, dict[str, list[dict]]], history: dict[str, list[dict]]
          ) -> dict[int, dict[str, list[dict]]]:
    """Keep only team-seasons the archive actually rates.

    Without this, every FCS opponent that ever appeared on a schedule gets a
    page - thousands of them, for teams the site does not rank and has no page
    for. A game log is only reachable from a season the team was rated in.
    """
    rated = {(row["season"], team) for team, rows in history.items() for row in rows}
    return {
        season: {team: rows for team, rows in teams.items() if (season, team) in rated}
        for season, teams in logs.items()
    }
