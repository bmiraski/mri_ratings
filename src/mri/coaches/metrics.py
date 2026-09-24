"""Per-coach-season metrics: how much a coach's team beat its starting expectation,
and how that compares to the program's own normal level.

All metrics are on the MRI 2.0 Power scale. Every lookup goes through
``coach_ratings_history.parquet`` (``mri.ratings.history``) - the one table
with power, prior and résumé on one consistent scale across the archive/CFBD
boundary - so nothing here recomputes a rating except the one case that
table can't answer: a split season's departing coach, whose ``power_end`` is
a point-in-time refit through their last game rather than the season's final
number (``power_as_of``, ``split_season_power_end``).

``added`` for a coach's own later seasons is partly measured against their
own earlier work, because the prior is built from last season - that's why
``vs_par`` and ``vs_inherited`` exist alongside it, and why tenure summaries
rank by ``vs_par`` rather than ``added``.
"""

from __future__ import annotations

import pandas as pd

from ..ingest import registry
from ..ratings import history, mri2
from . import season

PROGRAM_PAR_SEASONS = 10
PROGRAM_PAR_MIN_SEASONS = 3

METRIC_COLUMNS = [
    "power_end", "prior", "added", "inherited", "vs_inherited", "program_par", "vs_par", "resume",
]


def _fbs_for(teams: list[str], season_: int) -> list[str]:
    """The anchor set for one season, matching how coach_ratings_history was built:
    ``was_fbs`` for the archive era, ``is_fbs`` for the bridge era (mri.ratings.history)."""
    if season_ < history.SEAM_YEAR:
        return [t for t in teams if registry.was_fbs(t, season_)]
    return [t for t in teams if registry.is_fbs(t)]


def _chronological(season_games: pd.DataFrame) -> pd.DataFrame:
    """A season's full game log, every team, in chronological order.

    Bridge-era rows (CFBD-derived, ``current_games.parquet``) are sorted
    explicitly by ``start_date``/``game_id``. Archive-era rows
    (``archive_games.parquet``) mostly have no usable date at all - only
    2018 and 2019 carry real ones - so they're left in their stored row
    order, the same assumption ``mri2.mark_postseason`` already leans on
    ("row order is chronological in every archived workbook").
    """
    if "start_date" in season_games.columns:
        return season_games.sort_values(["start_date", "game_id"]).reset_index(drop=True)
    return season_games.reset_index(drop=True)


def power_as_of(school: str, season_games: pd.DataFrame, games_through: int, *, prior: pd.Series, fbs: list[str]) -> float:
    """A team's MRI 2.0 Power refit on its league's schedule truncated to that team's ``games_through``-th game.

    ``season_games`` is the WHOLE season's games (every team, not just
    ``school``), the same source ``power_end`` normally comes from -
    truncating and refitting is the same trick ``sitedata.weekly_ratings``
    uses to show a rating as of a given week, cut here at an exact game
    count instead of a week boundary. ``games_through`` is 1-indexed and is
    exactly a coach's own ``games`` count from ``coach_season.parquet``, so
    it needs no join against CFBD's or the archive's own game identifiers -
    only how many of the school's games, in order, to include.
    """
    ordered = _chronological(season_games)
    at_school = ordered.index[(ordered["team1"] == school) | (ordered["team2"] == school)]
    if games_through > len(at_school):
        raise ValueError(f"{school}: asked for game {games_through}, but only {len(at_school)} were played")
    truncated = ordered.iloc[: at_school[games_through - 1] + 1]
    neutral = truncated["neutral"] if "neutral" in truncated.columns else None
    model = mri2.fit(
        truncated, prior=prior, neutral=neutral, anchor_teams=fbs, with_resume=False, with_efficiency=False
    )
    return float(model.power.get(school, float("nan")))


def split_season_power_end(
    coach_season: pd.DataFrame, ratings_history: pd.DataFrame, archive_games: pd.DataFrame, current_games: pd.DataFrame
) -> dict[tuple[str, str, int], float]:
    """``power_end`` overrides for a split season's non-final coach(es), keyed (coach_id, school, season).

    The coach who finishes the season isn't included here - their last game
    *is* the season's last game, so the season's already-persisted power is
    exactly their point-in-time rating; no refit needed. A split whose
    ``games`` counts don't add up to the schedule (the same handful phase 1
    already flags in ``season.build_coach_season``'s ``problems``) is skipped
    rather than raising, since a bad count there means the wrong cutoff, not
    a crash-worthy one.
    """
    overrides: dict[tuple[str, str, int], float] = {}

    for (school, season_), group in coach_season.groupby(["school", "season"]):
        if len(group) < 2:
            continue
        season_ = int(season_)
        games_source = archive_games if season_ < history.SEAM_YEAR else current_games
        season_games = games_source[games_source["season"] == season_]
        if season_games.empty:
            continue

        order = season.entry_order(group["coach_id"].tolist(), school, season_, coach_season)
        counts = group.set_index("coach_id")["games"]
        prior = ratings_history.loc[ratings_history["season"] == season_].set_index("team")["prior"]
        teams = sorted(set(season_games["team1"]) | set(season_games["team2"]))
        fbs = _fbs_for(teams, season_)

        cumulative = 0
        for i, coach_id in enumerate(order):
            cumulative += int(counts[coach_id])
            if i == len(order) - 1:
                continue  # finishes the season - no refit needed
            try:
                overrides[(coach_id, school, season_)] = power_as_of(
                    school, season_games, cumulative, prior=prior, fbs=fbs
                )
            except ValueError:
                continue  # counts don't add up for this split - already flagged elsewhere
    return overrides


def _program_par(school: str, stint_start: int, ratings_history: pd.DataFrame) -> float | None:
    """Mean Power over the up-to-10 seasons before ``stint_start``, or None with fewer than 3."""
    window = range(stint_start - PROGRAM_PAR_SEASONS, stint_start)
    rows = ratings_history[(ratings_history["team"] == school) & (ratings_history["season"].isin(window))]
    if len(rows) < PROGRAM_PAR_MIN_SEASONS:
        return None
    return float(rows["power"].mean())


def coach_season_metrics(
    coach_season: pd.DataFrame,
    ratings_history: pd.DataFrame,
    archive_games: pd.DataFrame,
    current_games: pd.DataFrame,
    *,
    overrides: dict[tuple[str, str, int], float] | None = None,
) -> pd.DataFrame:
    """``coach_season.parquet``, extended with every metric in the plan's §2 table.

    ``overrides`` is the split-season point-in-time refits from
    ``split_season_power_end``; computed fresh if not passed in (a caller
    that also wants to report how many there were should compute it once and
    pass it to both).
    """
    ratings = ratings_history.set_index(["team", "season"])
    if overrides is None:
        overrides = split_season_power_end(coach_season, ratings_history, archive_games, current_games)

    out = coach_season.copy()
    out["stint_start"] = out.groupby(["coach_id", "school"])["season"].transform("min")

    def power_end(row) -> float | None:
        override = overrides.get((row.coach_id, row.school, int(row.season)))
        if override is not None:
            return override
        return ratings["power"].get((row.school, int(row.season)))

    out["power_end"] = out.apply(power_end, axis=1)
    out["prior"] = out.apply(lambda r: ratings["prior"].get((r.school, int(r.season))), axis=1)
    out["added"] = out["power_end"] - out["prior"]
    out["inherited"] = out.apply(lambda r: ratings["power"].get((r.school, int(r.stint_start) - 1)), axis=1)
    out["vs_inherited"] = out["power_end"] - out["inherited"]
    out["program_par"] = out.apply(lambda r: _program_par(r.school, int(r.stint_start), ratings_history), axis=1)
    out["vs_par"] = out["power_end"] - out["program_par"]
    out["resume"] = out.apply(lambda r: ratings["resume"].get((r.school, int(r.season))), axis=1)

    return out.drop(columns="stint_start")


def _summarize(group: pd.DataFrame, coach_id: str, school: str | None) -> dict:
    ranked = group.dropna(subset=["vs_par"])
    best = ranked.loc[ranked["vs_par"].idxmax()] if not ranked.empty else None
    worst = ranked.loc[ranked["vs_par"].idxmin()] if not ranked.empty else None
    return {
        "coach_id": coach_id,
        "school": school,
        "seasons": len(group),
        "mean_added": group["added"].mean(),
        "mean_vs_par": group["vs_par"].mean(),
        "best_season": int(best["season"]) if best is not None else None,
        "best_added": float(best["added"]) if best is not None else None,
        "best_vs_par": float(best["vs_par"]) if best is not None else None,
        "worst_season": int(worst["season"]) if worst is not None else None,
        "worst_added": float(worst["added"]) if worst is not None else None,
        "worst_vs_par": float(worst["vs_par"]) if worst is not None else None,
    }


def tenure_summaries(metrics: pd.DataFrame) -> pd.DataFrame:
    """One row per coach (career, ``school`` null) and one row per (coach, school) stint.

    Best/worst season ranks by ``vs_par`` - see the module docstring for why,
    not by ``added`` - with both values shown for whichever season is picked.
    """
    rows = [_summarize(group, coach_id, None) for coach_id, group in metrics.groupby("coach_id")]
    rows += [
        _summarize(group, coach_id, school)
        for (coach_id, school), group in metrics.groupby(["coach_id", "school"])
    ]
    return pd.DataFrame(rows)


def build(
    coach_season: pd.DataFrame, ratings_history: pd.DataFrame, archive_games: pd.DataFrame, current_games: pd.DataFrame
) -> dict:
    overrides = split_season_power_end(coach_season, ratings_history, archive_games, current_games)
    table = coach_season_metrics(coach_season, ratings_history, archive_games, current_games, overrides=overrides)
    tenure = tenure_summaries(table)
    return {"metrics": table, "tenure": tenure, "refits": overrides}
