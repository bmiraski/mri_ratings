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
