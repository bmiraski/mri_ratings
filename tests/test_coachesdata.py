"""coachesdata._correct_current_season_records: CFBD's /coaches endpoint reports
0-0 for a season still in progress, so the live record comes from payload["teams"]
instead - except for a school whose season is split between a fired coach and an
interim, where each row's partial record must be left alone."""

from __future__ import annotations

import pandas as pd

from mri.export import coachesdata


def _row(coach_id: str, school: str, season: int, wins: int, losses: int, interim: bool = False) -> dict:
    return {"coach_id": coach_id, "coach_name": coach_id, "school": school, "season": season,
            "wins": wins, "losses": losses, "interim": interim}


def test_unsplit_current_season_takes_the_record_from_payload() -> None:
    coach_season = pd.DataFrame([
        _row("a", "School A", 2025, wins=10, losses=2),
        _row("a", "School A", 2026, wins=0, losses=0),
    ])
    payload = {"teams": [{"team": "School A", "wins": 4, "losses": 0}]}

    corrected = coachesdata._correct_current_season_records(coach_season, 2026, payload)

    current = corrected[(corrected["school"] == "School A") & (corrected["season"] == 2026)].iloc[0]
    assert (current["wins"], current["losses"]) == (4, 0)
    past = corrected[(corrected["school"] == "School A") & (corrected["season"] == 2025)].iloc[0]
    assert (past["wins"], past["losses"]) == (10, 2)  # untouched


def test_split_current_season_is_left_alone() -> None:
    coach_season = pd.DataFrame([
        _row("fired", "School B", 2026, wins=1, losses=3),
        _row("interim", "School B", 2026, wins=1, losses=1, interim=True),
    ])
    payload = {"teams": [{"team": "School B", "wins": 2, "losses": 4}]}

    corrected = coachesdata._correct_current_season_records(coach_season, 2026, payload)

    assert corrected.set_index("coach_id").loc["fired", ["wins", "losses"]].tolist() == [1, 3]
    assert corrected.set_index("coach_id").loc["interim", ["wins", "losses"]].tolist() == [1, 1]


def test_school_missing_from_payload_is_left_alone() -> None:
    coach_season = pd.DataFrame([_row("a", "School C", 2026, wins=0, losses=0)])
    payload = {"teams": [{"team": "Some Other School", "wins": 5, "losses": 0}]}

    corrected = coachesdata._correct_current_season_records(coach_season, 2026, payload)

    assert corrected.iloc[0][["wins", "losses"]].tolist() == [0, 0]


def _career_row(coach_id: str, school: str, season: int, wins: int, losses: int, added: float, interim: bool = False) -> dict:
    return {"coach_id": coach_id, "coach_name": coach_id, "school": school, "season": season,
            "wins": wins, "losses": losses, "added": added, "interim": interim}


def test_by_team_gives_a_returning_coach_a_separate_row_per_stint() -> None:
    """Petrino at Louisville: 2003-2006, then a decade away, then 2014-2018 - two
    rows, each with its own years/record/mean added, not one row spanning the gap."""
    coach_season = pd.DataFrame([
        _career_row("petrino", "Louisville", 2003, 9, 4, 3.0),
        _career_row("petrino", "Louisville", 2004, 11, 1, 18.0),
        _career_row("petrino", "Louisville", 2005, 9, 3, 4.0),
        _career_row("petrino", "Louisville", 2006, 12, 1, 9.0),
        _career_row("petrino", "Louisville", 2014, 9, 4, 2.0),
        _career_row("petrino", "Louisville", 2015, 8, 5, 0.0),
        _career_row("petrino", "Louisville", 2016, 9, 4, 10.0),
        _career_row("petrino", "Louisville", 2017, 8, 5, 2.0),
        _career_row("petrino", "Louisville", 2018, 2, 8, -17.0),
    ])

    by_team = coachesdata._by_team(coach_season)

    rows = by_team["Louisville"]
    assert len(rows) == 2
    assert rows[0]["years"] == "2003–2006"
    assert rows[0]["record"] == "41-9"
    assert rows[1]["years"] == "2014–2018"
    assert rows[1]["record"] == "36-26"


def test_by_team_keeps_one_row_for_a_continuous_tenure() -> None:
    coach_season = pd.DataFrame([
        _career_row("smart", "Georgia", 2016, 8, 5, 1.0),
        _career_row("smart", "Georgia", 2017, 12, 1, 9.0),
    ])

    rows = coachesdata._by_team(coach_season)["Georgia"]

    assert len(rows) == 1
    assert rows[0]["years"] == "2016–2017"
    assert rows[0]["record"] == "20-6"


def test_record_shows_ties_only_when_there_were_some() -> None:
    assert coachesdata._record(8, 3) == "8-3"
    assert coachesdata._record(8, 3, 0) == "8-3"
    assert coachesdata._record(8, 3, 1) == "8-3-1"
    assert coachesdata._record(8, 3, float("nan")) == "8-3"


def test_by_team_reaches_back_before_2003_and_sums_ties_into_the_stint_record() -> None:
    rows = [
        {**_row("old", "School C", s, wins=7, losses=3), "added": None, "ties": 1 if s == 1979 else 0}
        for s in (1978, 1979)
    ]
    entry = coachesdata._by_team(pd.DataFrame(rows))["School C"][0]
    assert entry["years"] == "1978–1979"
    assert entry["record"] == "14-6-1"
    assert entry["meanAdded"] is None
