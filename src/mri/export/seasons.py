"""The season archive: every year the system has, in the rating that produced it.

The honest difficulty here is that the archive spans two different ratings. MRI
Classic ran from 2000 and its numbers are cumulative points on a scale where a
good season is 130 and a great one is 150. MRI 2.0 answers a different question
in a different unit - points against an average team - where a great season is
+35. They are not comparable and nothing here pretends otherwise: each season is
labelled with the rating that produced it, and no page puts the two in one
column or one chart.

Retro-rating the old seasons with MRI 2.0 would have made one continuous scale
and was rejected. The archived game logs pool every non-FBS opponent into a
single "Non D1A" team, which is precisely the distortion MRI 2.0 exists to
remove; rating those seasons with it would produce numbers that look like the
modern ones, sit on the same axis, and mean something different again. A visible
seam is better than an invisible one.

What is here:

    football     2003-2019  MRI Classic, recomputed from the workbook game logs
                            and validated against what Ben published at the time
    football     2020 onward
                            MRI 2.0, chained season to season
    basketball   2012-13, 2018-19, 2019-20
                            MRI Basketball Classic, as published
    basketball   2020-21 onward
                            MRI 2.0, chained from 2019-20

Not 2017-18: that workbook is a December snapshot, nobody past twelve games.
The season in progress is excluded too - the rankings page is where a live
season belongs, and an archive holding a second, staler answer to the same
question is worse than one that waits.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PARQUET = Path(__file__).resolve().parents[3] / "data" / "parquet"
ARCHIVE_BB = Path(__file__).resolve().parents[3] / "data" / "archive-bb"

CLASSIC = "MRI Classic"
MODERN = "MRI 2.0"

# How many rows a season page shows. Entries carry every rated team, because
# the per-team history is an inversion of exactly this data and needs all of
# them; the cap is applied when the season page renders.
TOP = 40

# A workbook with a median of nine games played per team is a snapshot somebody
# saved in December, not a season. MRIBasketball201718.xlsx is exactly that -
# 1,521 games, nobody past twelve - and published as "2017-18 final" it would be
# a wrong answer sitting in an archive that exists to be a record. It still
# validates the formula, which is a different job.
MIN_MEDIAN_GAMES = 20


def _records(games: pd.DataFrame, season: int) -> dict[str, tuple[int, int]]:
    chunk = games[games["season"] == season]
    tally: dict[str, list[int]] = {}
    for row in chunk.itertuples():
        for team, won in ((row.team1, row.win1), (row.team2, row.win2)):
            entry = tally.setdefault(team, [0, 0])
            entry[0 if won == 1.0 else 1] += 1
    return {k: (v[0], v[1]) for k, v in tally.items()}


def _read(name: str) -> pd.DataFrame:
    path = PARQUET / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def football_seasons(current: int | None = None) -> list[dict]:
    """2003 through the last completed season, Classic then MRI 2.0.

    ``current`` is excluded while it is being played: the rankings page already
    shows it, and a second entry for a two-game season would be a different
    answer to the same question sitting in the place people go for the record.
    """
    from ..ingest import registry

    out = []

    classic = _read("archive_ratings")
    published = _read("archive_published")
    if not classic.empty:
        # The workbooks spell fourteen programs differently than the API does -
        # Cal, Central Florida, Mississippi, North Carolina State, Troy State and
        # nine more. Without resolving them each of those teams had two
        # histories: seventeen Classic seasons under the old name and six MRI
        # 2.0 seasons under the new one, neither aware of the other, and a team
        # page showing whichever half matched its own spelling. The registry
        # already held every mapping; nothing was asking it.
        def canonical(name: object) -> str:
            return registry.resolve(name, str(name))

        by_season = {
            int(s): {canonical(t): r for t, r in
                     c.set_index("team")["rank"].to_dict().items()}
            for s, c in published.groupby("season")
        } if not published.empty else {}
        for season, chunk in classic.groupby("season"):
            if current is not None and int(season) >= current:
                continue
            chunk = chunk.sort_values("rank")
            rows = [
                {
                    "rank": int(r["rank"]),
                    "team": canonical(r["team"]),
                    "wins": int(r["wins"]),
                    "losses": int(r["losses"]),
                    "rating": round(float(r["mri"]), 2),
                    "secondary": round(float(r["mri_per_game"]), 2),
                    "sosRank": int(r["sos_rank"]) if pd.notna(r["sos_rank"]) else None,
                }
                for _, r in chunk.iterrows()
            ]
            out.append(
                {
                    "season": int(season),
                    "label": str(int(season)),
                    "system": CLASSIC,
                    "ratingName": "MRI",
                    "secondaryName": "Per game",
                    "teams": rows,
                    "rated": len(rows),
                    "source": "recomputed",
                    # Whether the recomputation still agrees with the workbook.
                    "matchesPublished": _agrees(rows, by_season.get(int(season))),
                }
            )

    modern = _read("current_ratings")
    games = _read("current_games")
    if not modern.empty:
        for season, chunk in modern.groupby("season"):
            if current is not None and int(season) >= current:
                continue
            records = _records(games, int(season)) if not games.empty else {}
            # current_ratings rates every team that played, FCS included, so an
            # unfiltered table shows Idaho at #94 in a list of FBS teams. Ranks
            # are recomputed after the filter rather than left with gaps.
            chunk = chunk[chunk["team"].map(registry.is_fbs)].sort_values("rank")
            rows = [
                {
                    "rank": position,
                    "team": r["team"],
                    "wins": records.get(r["team"], (0, 0))[0],
                    "losses": records.get(r["team"], (0, 0))[1],
                    "rating": round(float(r["power"]), 2),
                    "secondary": round(float(r["resume"]), 2) if pd.notna(r["resume"]) else None,
                    "sosRank": None,
                }
                for position, (_, r) in enumerate(chunk.iterrows(), start=1)
            ]
            out.append(
                {
                    "season": int(season),
                    "label": str(int(season)),
                    "system": MODERN,
                    "ratingName": "Power",
                    "secondaryName": "Résumé",
                    "teams": rows,
                    "rated": len(rows),
                    "source": "computed",
                    "matchesPublished": None,
                }
            )

    return sorted(out, key=lambda s: -s["season"])


def basketball_seasons(current: int | None = None) -> list[dict]:
    """The published workbook seasons, then the rebuilt chain.

    ``current`` is excluded when it is still being played - the live pages
    already show it, and an archive entry for an unfinished season would be a
    second, quietly different answer to the same question.
    """
    from ..ingest import bb_registry as registry
    from .bb_sitedata import season_label

    out = []

    if ARCHIVE_BB.exists():
        from ..ingest.archive import read_basketball_archive

        for season, data in read_basketball_archive(ARCHIVE_BB).items():
            table = data.published.sort_values("rank")
            played = (table["wins"] + table["losses"]).median()
            if played < MIN_MEDIAN_GAMES:
                print(f"  archive: skipping {season} - median {played:.0f} games "
                      f"played, a mid-season snapshot rather than a season")
                continue
            # The same split football had, and worse: the workbooks spell more
            # than fifty of these programs differently than the API does, so
            # without resolving them a team's Classic seasons and its MRI 2.0
            # seasons file under two names and neither page knows about the
            # other. Departed programs keep their workbook spelling, which is
            # correct - there is no current name to resolve them to.
            rows = [
                {
                    "rank": int(r["rank"]),
                    "team": registry.resolve(r["team"], str(r["team"]), season=int(season)),
                    "wins": int(r["wins"]),
                    "losses": int(r["losses"]),
                    "rating": round(float(r["mri"]), 2),
                    "secondary": None,
                    "sosRank": None,
                }
                for _, r in table.iterrows()
            ]
            out.append(
                {
                    "season": int(season),
                    "label": season_label(int(season)),
                    "system": CLASSIC,
                    "ratingName": "MRI",
                    "secondaryName": None,
                    "teams": rows,
                    "rated": len(rows),
                    # Read straight out of the workbook, not recomputed. Saying
                    # otherwise would claim a verification that never happened
                    # here - the formula is checked by the test suite, which is
                    # a different statement than "this table was rebuilt".
                    "source": "published",
                    "matchesPublished": None,
                }
            )

    ratings = _read("bb_ratings")
    games = _read("bb_games")
    if not ratings.empty:
        for season, chunk in ratings.groupby("season"):
            season = int(season)
            if current is not None and season >= current:
                continue
            records = _records(games, season) if not games.empty else {}
            chunk = chunk[chunk["team"].map(lambda t: registry.is_d1(t, season=season))]
            chunk = chunk.sort_values("rank")
            rows = [
                {
                    "rank": position,
                    "team": r["team"],
                    "wins": records.get(r["team"], (0, 0))[0],
                    "losses": records.get(r["team"], (0, 0))[1],
                    "rating": round(float(r["power"]), 2),
                    "secondary": round(float(r["resume"]), 2) if pd.notna(r["resume"]) else None,
                    "sosRank": None,
                }
                # Re-ranked after the non-D1 rows are dropped, so the archive
                # does not show a #1 followed by a #3.
                for position, (_, r) in enumerate(chunk.iterrows(), start=1)
            ]
            out.append(
                {
                    "season": season,
                    "label": season_label(season),
                    "system": MODERN,
                    "ratingName": "Power",
                    "secondaryName": "Résumé",
                    "teams": rows,
                    "rated": len(rows),
                    "source": "computed",
                    "matchesPublished": None,
                }
            )

    return sorted(out, key=lambda s: -s["season"])


def _agrees(rows: list[dict], published: dict | None) -> bool | None:
    """Does the recomputation still put the same teams in the same places?

    The archive's whole claim to be a record rests on this, so it is carried
    into the payload and shown rather than assumed.
    """
    if not published or not rows:
        return None
    return all(
        published.get(row["team"]) == row["rank"]
        for row in rows[:25]
        if row["team"] in published
    )


def for_sport(sport: str, current: int | None = None) -> list[dict]:
    if sport == "basketball":
        return basketball_seasons(current)
    return football_seasons(current)


def team_history(entries: list[dict]) -> dict[str, list[dict]]:
    """Invert the season list: for each team, every season it was rated in.

    Ranks travel across the two ratings and the numbers do not. Both formulas
    rank within the same field, so "#4 in 2011" and "#4 in 2023" mean roughly
    the same thing even though 143.5 and +34.6 do not; the rating is carried
    with the name of the system that produced it so a reader is never invited to
    subtract one from the other.

    The field itself grew - 117 FBS teams in 2003, 138 now - so even a rank is a
    rank among more teams than it used to be. The row carries the size so the
    page can say so.
    """
    history: dict[str, list[dict]] = {}
    for entry in entries:
        for team in entry["teams"]:
            history.setdefault(team["team"], []).append(
                {
                    "season": entry["season"],
                    "label": entry["label"],
                    "system": entry["system"],
                    "ratingName": entry["ratingName"],
                    "rank": team["rank"],
                    "of": entry["rated"],
                    "wins": team["wins"],
                    "losses": team["losses"],
                    "rating": team["rating"],
                }
            )
    for rows in history.values():
        rows.sort(key=lambda r: -r["season"])
    return history
