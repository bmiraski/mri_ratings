"""The coach_season table: one row per coach per school per season.

Most of this is a straight pass-through of CFBD's own per-coach-season
numbers. The one real problem is a school that had two coaches in one
season - a mid-season firing or resignation - where CFBD gives each coach's
game count but not which games. ``attribute_games`` recovers that from the
school's own schedule: the coach who was already at the school entering the
season coaches the games at the front, and whoever is new to the school
coaches the games at the back, split at each coach's own reported count. That
is what a mid-season change actually looks like, and it also doubles as a
check that CFBD's per-coach counts really do add up to a played schedule.

``interim`` uses the same "new to the school" signal: within a split season,
a coach with no earlier season at that school is flagged interim for that
season only. It is a proxy, not a title CFBD provides - a coach hired
mid-season straight into the permanent job would be flagged the same way for
their first, partial season - but it is the only signal the data actually
carries, and it self-corrects the moment that coach has a full season of
their own (see the module tests for a worked example).
"""

from __future__ import annotations

import pandas as pd

from ..ingest import cfbd, registry
from . import ids

COLUMNS = ["coach_id", "coach_name", "school", "season", "games", "wins", "losses", "interim"]


def _canonical_schools(coach_rows: pd.DataFrame) -> pd.DataFrame:
    """Put ``school`` into registry spelling.

    CFBD is not internally consistent about it: ``/coaches`` calls Appalachian
    State "App State", but ``/games`` (what ``attribute_games`` matches a
    split season's schedule against) calls it "Appalachian State". Left
    unresolved, that mismatch would both fail every split-season game lookup
    for the affected schools and stop their rows from joining to
    ``coach_ratings_history.parquet`` by (school, season).
    """
    out = coach_rows.copy()
    out["school"] = [registry.resolve(s, s) for s in out["school"]]
    return out


def _order_games(school: str, season: int, games_fn=cfbd.games) -> pd.DataFrame:
    """A school's schedule for one season, in registry-canonical team names.

    ``cfbd.games`` returns CFBD's own raw spelling (e.g. "App State"), which
    is not always what ``registry.resolve`` calls the same team (e.g.
    "Appalachian State", what ``school`` and ``coach_ratings_history.parquet``
    both use) - so this resolves the schedule's own team columns the same
    way before matching, rather than asking every caller to know CFBD's raw
    spelling. ``games_fn`` takes a season and returns a games frame; tests
    inject a synthetic one instead of calling the real API.
    """
    games = games_fn(season)
    if games.empty:
        return games
    games = games.copy()
    for column in ("team1", "team2"):
        games[column] = [registry.resolve(name, name) for name in games[column]]
    at_school = games[(games["team1"] == school) | (games["team2"] == school)]
    return at_school.sort_values(["start_date", "game_id"])


def _is_continuing(coach_id_: str, school: str, season: int, coach_rows: pd.DataFrame) -> bool:
    """Whether this coach already had a season at this school before ``season``."""
    prior = coach_rows[
        (coach_rows["coach_id"] == coach_id_)
        & (coach_rows["school"] == school)
        & (coach_rows["season"] < season)
    ]
    return not prior.empty


def entry_order(coach_ids_here: list[str], school: str, season: int, coach_rows: pd.DataFrame) -> list[str]:
    """Coaches at one split school-season, ordered by when they first arrived at the school.

    Public because ``mri.coaches.metrics`` reuses this exact rule to decide
    which coach in a split needs a point-in-time ``power_end`` refit.
    """

    def anchor(cid: str):
        prior = coach_rows[
            (coach_rows["coach_id"] == cid) & (coach_rows["school"] == school) & (coach_rows["season"] <= season)
        ]
        return (int(prior["season"].min()), cid)

    return sorted(coach_ids_here, key=anchor)


def attribute_games(
    school: str, season: int, coaches_here: pd.DataFrame, coach_rows: pd.DataFrame, games_fn=cfbd.games
) -> pd.DataFrame:
    """Every game the school played in ``season``, with the coach_id who gets credit.

    ``coaches_here`` is the (more-than-one-row) coach_season slice for this
    school-season, each row carrying CFBD's own ``games`` count. Raises if
    the counts do not add up to the school's actual schedule that season -
    callers should catch this per school-season rather than let one bad row
    stop a whole build, since it means either the entry-order heuristic
    guessed wrong or the two data sources disagree.
    """
    order = entry_order(coaches_here["coach_id"].tolist(), school, season, coach_rows)
    schedule = _order_games(school, season, games_fn)
    counts = coaches_here.set_index("coach_id")["games"].to_dict()

    assignments = []
    pos = 0
    for cid in order:
        n = int(counts[cid])
        block = schedule.iloc[pos : pos + n]
        assignments.extend({"game_id": g, "coach_id": cid} for g in block["game_id"])
        pos += n
    if pos != len(schedule):
        raise ValueError(f"{school} {season}: coach games sum to {pos}, schedule has {len(schedule)} games")
    return pd.DataFrame(assignments, columns=["game_id", "coach_id"])


def build_coach_season(
    coach_rows: pd.DataFrame, *, validate: bool = True, games_fn=cfbd.games
) -> tuple[pd.DataFrame, list[dict]]:
    """The coach_season table, plus a list of school-seasons whose split-game counts did not validate.

    ``coach_rows`` is ``mri.ingest.cfbd.coaches`` with ``coach_id`` already
    assigned (``mri.coaches.ids.assign_ids``). Games, wins and losses are
    CFBD's own numbers, kept as reported.
    """
    problems: list[dict] = []
    interim_flags: dict[tuple[str, str, int], bool] = {}

    for (school, season), group in coach_rows.groupby(["school", "season"]):
        if len(group) < 2:
            continue
        season = int(season)
        for _, row in group.iterrows():
            interim_flags[(row["coach_id"], school, season)] = not _is_continuing(
                row["coach_id"], school, season, coach_rows
            )
        if not validate:
            continue
        try:
            attribute_games(school, season, group, coach_rows, games_fn)
        except Exception as exc:  # noqa: BLE001 - collected, not fatal to the build
            problems.append({"school": school, "season": season, "error": str(exc)})

    table = coach_rows.copy()
    table["interim"] = [
        interim_flags.get((r.coach_id, r.school, int(r.season)), False) for r in table.itertuples()
    ]
    return table[COLUMNS].sort_values(["coach_id", "season"]).reset_index(drop=True), problems


def build(min_year: int, max_year: int, *, validate: bool = True) -> dict:
    """Fetch, identify and validate: the coach_season table plus id collisions and split-season problems."""
    raw = _canonical_schools(cfbd.coaches(min_year, max_year))
    with_ids = ids.assign_ids(raw)
    collisions = ids.detect_collisions(with_ids)
    table, problems = build_coach_season(with_ids, validate=validate)
    return {"table": table, "collisions": collisions, "problems": problems}
