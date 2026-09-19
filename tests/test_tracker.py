"""The track record is only worth publishing if it cannot be improved afterwards."""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from mri.betting import tracker


def test_grading_the_home_side() -> None:
    assert tracker.grade("home", 10, 7.5) == "win"
    assert tracker.grade("home", 7, 7.5) == "loss"
    assert tracker.grade("home", 7, 7) == "push"


def test_grading_the_away_side() -> None:
    assert tracker.grade("away", 7, 7.5) == "win"
    assert tracker.grade("away", 10, 7.5) == "loss"
    assert tracker.grade("away", -3, -3) == "push"


def test_units_at_minus_110() -> None:
    assert tracker.units("win") == pytest.approx(0.9091, abs=1e-4)
    assert tracker.units("loss") == -1.0
    assert tracker.units("push") == 0.0


def test_closing_line_value_follows_the_side_taken() -> None:
    # The market moved from home -3 to home -5: toward the home side.
    assert tracker.closing_value("home", 3.0, 5.0) == 2.0
    assert tracker.closing_value("away", 3.0, 5.0) == -2.0


def test_slope_of_a_perfect_and_an_overdone_line() -> None:
    actual = pd.Series([10.0, -4.0, 20.0, 0.0])
    assert tracker._slope(actual, actual) == 1.0
    assert tracker._slope(actual * 1.5, actual) == pytest.approx(1 / 1.5, abs=0.01)


# ---- the forward log

KICKOFF = "2026-09-26T20:00:00.000Z"
NOW = dt.datetime(2026, 9, 24, 11, 0, tzinfo=dt.timezone.utc)


def fake_schedule(played: bool = False, pts=(20.0, 27.0)):
    return pd.DataFrame([
        {"game_id": 1, "week": 4, "team1": "Away U", "team2": "Home U", "played": played,
         "pts1": pts[0] if played else None, "pts2": pts[1] if played else None,
         "start_date": KICKOFF, "neutral": False},
        {"game_id": 2, "week": 4, "team1": "Late A", "team2": "Late B", "played": False,
         "pts1": None, "pts2": None, "start_date": "2026-09-24T09:00:00.000Z", "neutral": False},
    ])


def board(edge=6.0, market=4.0):
    def game(home, away, edge):
        return {"week": 4, "home": home, "away": away, "neutral": False, "predicted": 10.0,
                "market": market, "marketOpen": market - 1, "edge": edge}
    return {"flagged": [game("Home U", "Away U", edge), game("Late B", "Late A", edge)]}


@pytest.fixture
def wire(monkeypatch):
    state = {"schedule": fake_schedule(), "book": pd.DataFrame()}
    monkeypatch.setattr(tracker.cfbd, "games", lambda *a, **k: state["schedule"])
    monkeypatch.setattr(tracker.lines_module, "preferred_lines", lambda *a, **k: state["book"])
    return state


def test_a_pick_is_logged_once_and_never_rewritten(wire, tmp_path) -> None:
    path = tmp_path / "picks.json"
    tracker.update_log(2026, board(edge=6.0, market=4.0), path, now=NOW)
    first = json.loads(path.read_text())["picks"][0]

    later = NOW + dt.timedelta(days=1)
    tracker.update_log(2026, board(edge=9.0, market=1.0), path, now=later)
    again = json.loads(path.read_text())["picks"]
    assert len(again) == 1
    assert again[0] == first, "a logged pick changed after it was written"
    assert first["taken"] == 4.0 and first["side"] == "home"


def test_a_game_already_kicked_off_is_not_a_pick(wire, tmp_path) -> None:
    path = tmp_path / "picks.json"
    tracker.update_log(2026, board(), path, now=NOW)
    ids = [p["game_id"] for p in json.loads(path.read_text())["picks"]]
    assert 2 not in ids and 1 in ids


def test_results_are_added_but_the_pick_is_not_touched(wire, tmp_path) -> None:
    path = tmp_path / "picks.json"
    tracker.update_log(2026, board(edge=6.0, market=4.0), path, now=NOW)
    before = json.loads(path.read_text())["picks"][0]

    wire["schedule"] = fake_schedule(played=True, pts=(20.0, 27.0))     # home wins by 7
    wire["book"] = pd.DataFrame([{"game_id": 1, "market": 6.5, "market_open": 3.0}])
    summary = tracker.update_log(2026, board(), path, now=NOW + dt.timedelta(days=3))

    after = json.loads(path.read_text())["picks"][0]
    for key, value in before.items():
        assert after[key] == value, key
    assert after["result"] == "win" and after["actual"] == 7        # 7 beats a taken line of 4
    assert after["close"] == 6.5 and after["clv"] == 2.5             # the market moved toward home
    assert summary["wins"] == 1 and summary["graded"] == 1 and summary["units"] == pytest.approx(0.91, abs=0.01)


def test_a_graded_pick_is_not_regraded(wire, tmp_path) -> None:
    path = tmp_path / "picks.json"
    tracker.update_log(2026, board(edge=6.0, market=4.0), path, now=NOW)
    wire["schedule"] = fake_schedule(played=True, pts=(20.0, 27.0))
    tracker.update_log(2026, board(), path, now=NOW + dt.timedelta(days=3))
    graded = json.loads(path.read_text())["picks"][0]
    wire["schedule"] = fake_schedule(played=True, pts=(50.0, 3.0))       # nonsense correction
    tracker.update_log(2026, board(), path, now=NOW + dt.timedelta(days=4))
    assert json.loads(path.read_text())["picks"][0] == graded


def test_rerunning_changes_nothing(wire, tmp_path) -> None:
    """A run with no news must not produce a diff: the log is committed every day."""
    path = tmp_path / "picks.json"
    first = tracker.update_log(2026, board(), path, now=NOW)
    text = path.read_text()
    second = tracker.update_log(2026, board(), path, now=NOW)
    assert second == first
    assert json.dumps(second, sort_keys=False) == json.dumps(first, sort_keys=False)
    assert path.read_text() == text
