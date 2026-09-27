"""The coaches feature's data: the index, per-coach career detail, the team-page
strip's career summaries, the method-page's model report, and this week's
in-season odds.

Deliberately unlisted for now (see build_site.py) - some of the departure
labels behind the hot-seat model were assembled and reviewed by hand, and
the plan is to watch this run for a few weeks before anything here is
linked from a page a visitor would actually find.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..coaches import inseason
from ..coaches.season import stint_start
from ..ingest import cfbd
from ..sim import season as sim_season
from . import simdata

DEFAULT_HISTORY_PATH = Path(__file__).resolve().parents[3] / "site" / "data" / "hotseat_history.json"
CAREER_COLUMNS = ["season", "school", "games", "wins", "losses", "interim", "power_end", "prior", "added", "vs_par", "resume"]


def _prepared_sim_inputs(current_season: int, payload: dict):
    """The same teams/schedule/championships prep mri.export.simdata and mri.export.rooting
    already do for their own simulate() calls - each consumer runs its own, rather than
    sharing one simulation's chunks across unrelated features."""
    teams = [{"team": t["team"], "power": t["power"], "conference": t["conference"]} for t in payload["teams"]]
    conference = {t["team"]: t["conference"] for t in teams}
    schedule = simdata._prepare(cfbd.games(current_season, completed_only=False))
    schedule, title_games = simdata.championships(schedule, conference)
    return teams, schedule, title_games or None


def _current_odds(current_season: int, payload: dict, coach_season: pd.DataFrame, model_path: Path,
                   history_path: Path, sims: int) -> dict:
    """This week's hot-seat odds, and last week's, for the change column - or {} if the
    model hasn't shipped, or there's no simulation to run this week (e.g. a finished season)."""
    if not model_path.exists() or not payload.get("sim"):
        return {"current": {}, "previous": {}}
    model = json.loads(model_path.read_text())
    if not model.get("passed"):
        return {"current": {}, "previous": {}}

    teams, schedule, championships = _prepared_sim_inputs(current_season, payload)
    week = int(payload["week"])
    history = inseason.weekly_odds(
        coach_season, current_season, week, teams, schedule, payload["homeField"], history_path,
        sims=sims, championships=championships, model_path=model_path,
    )
    return {"current": history["weeks"].get(str(week), {}), "previous": history["weeks"].get(str(week - 1), {})}


def _correct_current_season_records(coach_season: pd.DataFrame, current_season: int, payload: dict) -> pd.DataFrame:
    """CFBD's /coaches endpoint doesn't report a season's games/wins/losses until
    that season is over, so every coach_season row for the live, in-progress season
    still shows 0-0. Overwrite it with the real, already-computed record from
    payload["teams"] instead - but only where the season hasn't been split between
    a fired coach and an interim, since a split's rows must sum to the team's total
    rather than each carry it."""
    team_record = {t["team"]: (t["wins"], t["losses"]) for t in payload["teams"]}
    current_rows = coach_season[coach_season["season"] == current_season]
    unsplit_schools = set(current_rows.groupby("school").size().loc[lambda s: s == 1].index)
    corrected = coach_season.copy()
    mask = (corrected["season"] == current_season) & corrected["school"].isin(unsplit_schools)
    for idx in corrected.index[mask]:
        wins_losses = team_record.get(corrected.at[idx, "school"])
        if wins_losses is not None:
            corrected.at[idx, "wins"], corrected.at[idx, "losses"] = wins_losses
    return corrected


def _index_rows(coach_season: pd.DataFrame, current_season: int, odds: dict) -> list[dict]:
    active = coach_season[(~coach_season["interim"]) & (coach_season["season"] == current_season)].copy()
    active["stintStart"] = stint_start(coach_season).reindex(active.index)
    rows = []
    for row in active.itertuples():
        current, previous = odds["current"].get(row.coach_id), odds["previous"].get(row.coach_id)
        rows.append({
            "coachId": row.coach_id, "name": row.coach_name, "school": row.school,
            "tenureYear": int(row.season - row.stintStart + 1),
            "record": f"{int(row.wins)}-{int(row.losses)}",
            "added": None if pd.isna(row.added) else round(float(row.added), 1),
            "vsPar": None if pd.isna(row.vs_par) else round(float(row.vs_par), 1),
            "hotSeat": current, "hotSeatChange": round(current - previous, 4) if current is not None and previous is not None else None,
        })
    return rows


def _by_team(coach_season: pd.DataFrame) -> dict[str, list[dict]]:
    """One row per (coach, school) stint - a coach who left and came back later
    (Alvarez at Wisconsin, Petrino at Louisville, ...) gets a separate row for
    each stint rather than one row spanning the years they weren't there."""
    with_stints = coach_season.assign(_stintStart=stint_start(coach_season))
    by_team: dict[str, list[dict]] = {}
    grouped = with_stints.sort_values("season").groupby(["coach_id", "school", "_stintStart"])
    for (coach_id, school, _), group in grouped:
        group = group[~group["interim"]]
        if group.empty:
            continue
        seasons = sorted(group["season"].tolist())
        years = f"{seasons[0]}–{seasons[-1]}" if seasons[0] != seasons[-1] else str(seasons[0])
        mean_added = group["added"].mean()
        by_team.setdefault(school, []).append({
            "coachId": coach_id, "name": group["coach_name"].iloc[0], "years": years,
            "record": f"{int(group['wins'].sum())}-{int(group['losses'].sum())}",
            "meanAdded": None if pd.isna(mean_added) else round(float(mean_added), 1),
        })
    for school in by_team:
        by_team[school].sort(key=lambda r: r["years"])
    return by_team


def _career_table(group: pd.DataFrame) -> list[dict]:
    rows = []
    for row in group.sort_values("season").itertuples():
        rows.append({
            "season": int(row.season), "school": row.school, "interim": bool(row.interim),
            "record": f"{int(row.wins)}-{int(row.losses)}",
            "powerEnd": None if pd.isna(row.power_end) else round(float(row.power_end), 1),
            "prior": None if pd.isna(row.prior) else round(float(row.prior), 1),
            "added": None if pd.isna(row.added) else round(float(row.added), 1),
            "vsPar": None if pd.isna(row.vs_par) else round(float(row.vs_par), 1),
            "resume": None if pd.isna(row.resume) else round(float(row.resume), 2),
        })
    return rows


def _detail(coach_season: pd.DataFrame, coach_tenure: pd.DataFrame, current_season: int, odds: dict) -> dict[str, dict]:
    tenure_by_id = coach_tenure[coach_tenure["school"].isna()].set_index("coach_id")
    detail = {}
    for coach_id, group in coach_season.groupby("coach_id"):
        career = tenure_by_id.loc[coach_id] if coach_id in tenure_by_id.index else None
        is_current = bool((group["season"] == current_season).any() and not group[group["season"] == current_season]["interim"].all())
        detail[coach_id] = {
            "coachId": coach_id, "name": group["coach_name"].iloc[0], "seasons": _career_table(group),
            "meanAdded": None if career is None or pd.isna(career["mean_added"]) else round(float(career["mean_added"]), 1),
            "meanVsPar": None if career is None or pd.isna(career["mean_vs_par"]) else round(float(career["mean_vs_par"]), 1),
            "hotSeat": odds["current"].get(coach_id) if is_current else None,
        }
    return detail


def build(
    current_season: int, payload: dict, coach_season: pd.DataFrame, coach_tenure: pd.DataFrame, *,
    model_path: Path = inseason.DEFAULT_MODEL_PATH,
    history_path: Path = DEFAULT_HISTORY_PATH, sims: int = sim_season.DEFAULT_SIMS,
) -> dict:
    """Everything the coaches feature needs. In-season odds are computed only if a
    fitted, shipped model exists and this build has a simulation to run."""
    model = json.loads(model_path.read_text()) if model_path.exists() else None
    coach_season = _correct_current_season_records(coach_season, current_season, payload)
    odds = _current_odds(current_season, payload, coach_season, model_path, history_path, sims)
    return {
        "week": int(payload["week"]),
        "index": _index_rows(coach_season, current_season, odds),
        "byTeam": _by_team(coach_season),
        "detail": _detail(coach_season, coach_tenure, current_season, odds),
        "model": model,
    }
