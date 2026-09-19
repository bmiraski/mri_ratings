"""Championship games must be neither double-counted nor missed."""

from __future__ import annotations

import pandas as pd

from mri.export import simdata, slate
from mri.sim import season

CONF = {"A1": "SEC", "A2": "SEC", "A3": "SEC", "B1": "Big Ten", "B2": "Big Ten"}


def frame(rows):
    return pd.DataFrame(
        [{"game_id": i, "week": w, "team1": a, "team2": h, "played": p,
          "pts1": 10.0 if p else float("nan"), "pts2": 20.0 if p else float("nan"),
          "neutral": True, "conference_game": c} for i, (w, a, h, p, c) in enumerate(rows)]
    )


def test_an_unplayed_title_game_leaves_the_schedule_and_becomes_an_entry() -> None:
    schedule = frame([(13, "A1", "A2", False, True), (season.CHAMPIONSHIP_WEEK, "A2", "A1", False, True)])
    kept, entries = simdata.championships(schedule, CONF)
    assert len(kept) == 1 and entries["SEC"]["a"] == "A1" and not entries["SEC"]["played"]


def test_a_played_title_game_stays_in_the_schedule() -> None:
    schedule = frame([(season.CHAMPIONSHIP_WEEK, "A2", "A1", True, True)])
    kept, entries = simdata.championships(schedule, CONF)
    assert len(kept) == 1
    assert entries["SEC"]["played"] and entries["SEC"]["a_won"] is True      # host scored 20 to 10


def test_two_conference_games_in_title_week_is_not_a_title_game() -> None:
    schedule = frame([(season.CHAMPIONSHIP_WEEK, "A2", "A1", False, True),
                      (season.CHAMPIONSHIP_WEEK, "A3", "A1", False, True)])
    kept, entries = simdata.championships(schedule, CONF)
    assert "SEC" not in entries and len(kept) == 2


def test_non_conference_games_in_title_week_are_ignored() -> None:
    schedule = frame([(season.CHAMPIONSHIP_WEEK, "A2", "B1", False, False)])
    kept, entries = simdata.championships(schedule, CONF)
    assert not entries and len(kept) == 1


def test_as_of_hides_later_results() -> None:
    schedule = frame([(1, "A1", "A2", True, True), (2, "A1", "A3", True, True)])
    then = simdata._as_of(schedule, 1)
    assert then["played"].tolist() == [True, False]
    assert then["pts1"].isna().tolist() == [False, True]


def test_kickoff_is_shown_in_eastern_time() -> None:
    when = slate._eastern("2026-09-19T23:30:00.000Z")
    assert (when.hour, when.minute) == (19, 30)
    assert slate._eastern(None) is None


def test_the_side_with_more_at_stake_is_reported() -> None:
    entry = {"home": {"ifWin": 0.5, "ifLose": 0.4}, "away": {"ifWin": 0.3, "ifLose": 0.05}, "swing": 0.25}
    stake = slate._swing(entry)
    assert stake["side"] == "away" and round(stake["swing"], 2) == 0.25
    assert slate._swing(None) is None
