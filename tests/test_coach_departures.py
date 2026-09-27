"""Departure labeling: only a confirmed last season gets classified, the
program-vs-personal moved-up override, the added fallback for early years
without enough program_par history, and the split-season false-positive it
was built to fix."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mri.coaches import departures


def _rows(*records: dict) -> pd.DataFrame:
    base = {
        "coach_id": None, "coach_name": None, "school": None, "season": None,
        "games": 12, "wins": 6, "losses": 6, "interim": False,
        "power_end": 0.0, "prior": 0.0, "added": np.nan, "inherited": np.nan,
        "vs_inherited": np.nan, "program_par": np.nan, "vs_par": np.nan, "resume": 0.0,
    }
    return pd.DataFrame([{**base, **r} for r in records])


def test_a_confirmed_successor_makes_a_row_eligible_but_still_active_is_not() -> None:
    coach_season = _rows(
        {"coach_id": "a", "coach_name": "A", "school": "School X", "season": 2020},
        {"coach_id": "b", "coach_name": "B", "school": "School X", "season": 2021},  # A's successor
        {"coach_id": "b", "coach_name": "B", "school": "School X", "season": 2022},  # B still active
    )
    eligible = departures.eligible_departures(coach_season)
    assert set(zip(eligible["coach_id"], eligible["season"])) == {("a", 2020)}


def test_a_departing_coachs_own_split_season_does_not_flag_the_year_before_it() -> None:
    """The bug this was built to fix: an interim closing out someone's final season is a
    'different coach_id next year' too, but the coach was still there for the year before that."""
    coach_season = _rows(
        {"coach_id": "vet", "coach_name": "Vet", "school": "School X", "season": 2023},
        {"coach_id": "vet", "coach_name": "Vet", "school": "School X", "season": 2024},  # last full season
        {"coach_id": "vet", "coach_name": "Vet", "school": "School X", "season": 2025, "games": 12},
        {"coach_id": "interim", "coach_name": "Interim", "school": "School X", "season": 2025,
         "games": 1, "interim": True},
        {"coach_id": "successor", "coach_name": "Successor", "school": "School X", "season": 2026},
    )
    eligible = departures.eligible_departures(coach_season)
    assert set(zip(eligible["coach_id"], eligible["season"])) == {("vet", 2025)}


def test_interim_rows_are_never_eligible() -> None:
    coach_season = _rows(
        {"coach_id": "interim", "coach_name": "Interim", "school": "School X", "season": 2020, "interim": True},
        {"coach_id": "successor", "coach_name": "Successor", "school": "School X", "season": 2021},
    )
    assert departures.eligible_departures(coach_season).empty


def _classify_row(coach_season: pd.DataFrame, coach_id: str, season: int) -> dict:
    row = next(r for r in coach_season.itertuples() if r.coach_id == coach_id and r.season == season)
    return departures.classify(row, coach_season)


def test_below_par_with_no_next_job_is_fired() -> None:
    coach_season = _rows(
        {"coach_id": "a", "coach_name": "A", "school": "School X", "season": 2020, "vs_par": -5.0},
        {"coach_id": "b", "coach_name": "B", "school": "School X", "season": 2021},
    )
    result = _classify_row(coach_season, "a", 2020)
    assert result["label"] == departures.FIRED


def test_a_bigger_next_job_is_moved_up_even_with_a_below_par_final_season() -> None:
    coach_season = _rows(
        {"coach_id": "a", "coach_name": "A", "school": "Small School", "season": 2020,
         "vs_par": -8.0, "program_par": 0.0},
        {"coach_id": "b", "coach_name": "B", "school": "Small School", "season": 2021},
        {"coach_id": "a", "coach_name": "A", "school": "Big School", "season": 2021, "program_par": 20.0},
    )
    result = _classify_row(coach_season, "a", 2020)
    assert result["label"] == departures.MOVED_UP


def test_above_par_with_no_next_job_is_ambiguous_not_fired() -> None:
    coach_season = _rows(
        {"coach_id": "a", "coach_name": "A", "school": "School X", "season": 2020, "vs_par": +4.0},
        {"coach_id": "b", "coach_name": "B", "school": "School X", "season": 2021},
    )
    result = _classify_row(coach_season, "a", 2020)
    assert result["label"] == departures.AMBIGUOUS


def test_a_smaller_next_job_falls_through_to_the_same_split_as_no_job() -> None:
    coach_season = _rows(
        {"coach_id": "a", "coach_name": "A", "school": "Big School", "season": 2020,
         "vs_par": -8.0, "program_par": 20.0},
        {"coach_id": "b", "coach_name": "B", "school": "Big School", "season": 2021},
        {"coach_id": "a", "coach_name": "A", "school": "Small School", "season": 2021, "program_par": 0.0},
    )
    result = _classify_row(coach_season, "a", 2020)
    assert result["label"] == departures.FIRED  # below par, and the smaller job doesn't change that

    coach_season.loc[coach_season["coach_id"] == "a", "vs_par"] = +4.0  # same shape, above par instead
    result = _classify_row(coach_season, "a", 2020)
    assert result["label"] == departures.AMBIGUOUS


def test_missing_program_par_falls_back_to_added_instead_of_defaulting_to_fired() -> None:
    """The real bug this fixed: early-archive coaches with no program_par history at all -
    including well-known voluntary retirements - were silently defaulting to "fired"."""
    coach_season = _rows(
        {"coach_id": "a", "coach_name": "A", "school": "School X", "season": 2004,
         "vs_par": np.nan, "program_par": np.nan, "added": +2.0},
        {"coach_id": "b", "coach_name": "B", "school": "School X", "season": 2005},
    )
    result = _classify_row(coach_season, "a", 2004)
    assert result["label"] == departures.AMBIGUOUS  # above its own prior, no next job - not a confident firing
    assert "added" in result["reason"]

    coach_season.loc[coach_season["coach_id"] == "a", "added"] = -2.0
    result = _classify_row(coach_season, "a", 2004)
    assert result["label"] == departures.FIRED  # below its own prior - a real signal, not a guess


def test_no_signal_at_all_is_ambiguous() -> None:
    coach_season = _rows(
        {"coach_id": "a", "coach_name": "A", "school": "School X", "season": 2004,
         "vs_par": np.nan, "program_par": np.nan, "added": np.nan},
        {"coach_id": "b", "coach_name": "B", "school": "School X", "season": 2005},
    )
    result = _classify_row(coach_season, "a", 2004)
    assert result["label"] == departures.AMBIGUOUS


def test_build_only_labels_interim_coaches_via_someone_elses_row_never_their_own() -> None:
    coach_season = _rows(
        {"coach_id": "a", "coach_name": "A", "school": "School X", "season": 2020, "vs_par": -5.0},
        {"coach_id": "interim", "coach_name": "Interim", "school": "School X", "season": 2021, "interim": True},
        {"coach_id": "b", "coach_name": "B", "school": "School X", "season": 2022},
    )
    table = departures.build(coach_season)
    assert "interim" not in set(table["coach_id"])
    assert set(table["coach_id"]) == {"a"}
