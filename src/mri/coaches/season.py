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

import json
from pathlib import Path

import pandas as pd

from ..ingest import cfbd, registry
from . import ids

COLUMNS = ["coach_id", "coach_name", "school", "season", "conference", "games", "wins", "losses", "ties", "interim"]
CORRECTIONS_PATH = Path(__file__).resolve().parents[3] / "data" / "coach_season_corrections.json"


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


def apply_corrections(coach_rows: pd.DataFrame, path: Path = CORRECTIONS_PATH) -> pd.DataFrame:
    """Hand-maintained fixes for known-bad CFBD /coaches records (data/coach_season_corrections.json).

    CFBD is occasionally just wrong about one field of one row - see that
    file's own note for how such a case gets found (a split-season
    validation failure whose game counts only add up once you swap in the
    right school for one side). Each correction is matched by coach name,
    season and the field's current value, so a correction that no longer
    matches anything - because CFBD fixed it upstream, or the season/name
    was mistyped here - is caught rather than silently doing nothing.
    """
    if not path.exists():
        return coach_rows
    corrections = json.loads(path.read_text()).get("corrections", [])
    out = coach_rows.copy()
    for c in corrections:
        mask = (
            (out["coach_name"] == c["coach_name"])
            & (out["season"] == c["season"])
            & (out[c["field"]] == c["from"])
        )
        if not mask.any():
            raise ValueError(
                f"coach_season_corrections.json: no row matches {c['coach_name']} {c['season']} "
                f"{c['field']}={c['from']!r} - has the underlying CFBD data changed?"
            )
        out.loc[mask, c["field"]] = c["to"]
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


def stint_start(coach_season: pd.DataFrame) -> pd.Series:
    """Each row's own stint's first season - not the coach's first season ever at
    this school. A coach who left and came back (Alvarez at Wisconsin, Petrino at
    Louisville, Rich Rodriguez and Schiano both returning to a school years after
    leaving it) starts a new stint wherever a gap breaks the run of consecutive
    seasons. Needs the coach's full, unfiltered history at that school to see the
    gap at all - a caller working from a filtered slice should compute this
    against the unfiltered table first and reindex onto the slice.
    """
    key = coach_season[["coach_id", "school", "season"]].sort_values(["coach_id", "school", "season"])
    prev_season = key.groupby(["coach_id", "school"])["season"].shift(1)
    new_stint = prev_season.isna() | (key["season"] != prev_season + 1)
    stint_number = new_stint.groupby([key["coach_id"], key["school"]]).cumsum()
    start = key.groupby(["coach_id", "school", stint_number])["season"].transform("min")
    return start.reindex(coach_season.index)


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


def _record(games: pd.DataFrame, school: str) -> tuple[int, int]:
    """Wins and losses for ``school`` within one set of its own games."""
    home = games[games["team2"] == school]
    away = games[games["team1"] == school]
    wins = int((home["pts2"] > home["pts1"]).sum() + (away["pts1"] > away["pts2"]).sum())
    return wins, int(len(games) - wins)


def conference_record(coach_season: pd.DataFrame, *, games_fn=cfbd.games) -> pd.DataFrame:
    """Conference wins and losses for every coach-season: ``coach_id, school, season,
    conf_wins, conf_losses``.

    A non-split season is a straight filter of that school's schedule to
    ``conference_game`` rows. A split season reuses ``attribute_games``'s
    existing game-to-coach assignment rather than a second attribution
    scheme - the same games, just also checked against ``conference_game``
    and tallied. A split whose game counts don't validate (already flagged
    elsewhere, e.g. by ``build_coach_season``'s own ``problems`` list) gets
    zero for every coach there rather than a guess.
    """
    rows = []
    for (school, season), group in coach_season.groupby(["school", "season"]):
        season = int(season)
        schedule = _order_games(school, season, games_fn)
        conf_games = schedule[schedule["conference_game"]] if not schedule.empty else schedule

        if len(group) < 2:
            wins, losses = _record(conf_games, school) if not schedule.empty else (0, 0)
            rows.append({"coach_id": group["coach_id"].iloc[0], "school": school, "season": season,
                         "conf_wins": wins, "conf_losses": losses})
            continue

        try:
            assignments = attribute_games(school, season, group, coach_season, games_fn)
        except Exception:  # noqa: BLE001 - a bad split gets zeros, not a guess
            for cid in group["coach_id"]:
                rows.append({"coach_id": cid, "school": school, "season": season, "conf_wins": 0, "conf_losses": 0})
            continue

        for cid, game_ids in assignments.groupby("coach_id")["game_id"]:
            wins, losses = _record(conf_games[conf_games["game_id"].isin(game_ids)], school)
            rows.append({"coach_id": cid, "school": school, "season": season, "conf_wins": wins, "conf_losses": losses})

    return pd.DataFrame(rows, columns=["coach_id", "school", "season", "conf_wins", "conf_losses"])


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
    raw = apply_corrections(_canonical_schools(cfbd.coaches(min_year, max_year)))
    with_ids = ids.assign_ids(raw)
    collisions = ids.detect_collisions(with_ids)
    table, problems = build_coach_season(with_ids, validate=validate)
    return {"table": table, "collisions": collisions, "problems": problems}
