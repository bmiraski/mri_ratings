"""In December the next slate is Army-Navy, then the bowls - never "week 1".

The feed numbers the postseason from week 1 again, so every page that picks the
upcoming week as the lowest unplayed ``week`` skipped Army-Navy (regular week 16)
for the bowls and called them week 1. They pick on ``cfbd.sequence`` blocks now,
the same numbering the weekly ratings use, and label the postseason block "Bowls".
"""

from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from mri.betting import board, tracker
from mri.export import gamedaydata, site, sitedata, slate, slatearchive
from mri.ingest import cfbd

POWER = {"Alabama": 20.0, "Georgia": 18.0, "Ohio State": 25.0, "Michigan": 10.0, "Army": -2.0, "Navy": 1.0}
BOWLS = cfbd.POSTSEASON_LABEL


def december(*, army_navy_played: bool = False, bowl_played: bool = False) -> pd.DataFrame:
    """Title week played; Army-Navy (week 16) and two bowls (the feed's week 1) to come.

    Game 1 is a genuine week-1 game with the same home and away as the first bowl,
    so anything matching on teams and raw week alone would find the wrong one.
    """
    rows = [
        (1, 1, "regular", "Ohio State", "Alabama", True, "2026-09-05T19:30:00.000Z"),
        (1501, 15, "regular", "Georgia", "Alabama", True, "2026-12-05T21:00:00.000Z"),
        (1502, 15, "regular", "Michigan", "Ohio State", True, "2026-12-05T17:00:00.000Z"),
        (1601, 16, "regular", "Army", "Navy", army_navy_played, "2026-12-12T20:00:00.000Z"),
        (9001, 1, "postseason", "Ohio State", "Alabama", bowl_played, "2026-12-19T20:00:00.000Z"),
        (9002, 1, "postseason", "Michigan", "Georgia", False, "2026-12-31T17:00:00.000Z"),
    ]
    return pd.DataFrame([
        {"game_id": gid, "season": 2026, "week": week, "season_type": kind, "team1": away, "team2": home,
         "played": played, "pts1": 17.0 if played else None, "pts2": 24.0 if played else None,
         "win1": 0.0, "win2": 1.0 if played else 0.0, "neutral": kind == "postseason",
         "start_date": start, "start_time_tbd": False, "venue": None,
         "conf1": None, "conf2": None, "conference_game": False, "class1": "fbs", "class2": "fbs"}
        for gid, week, kind, away, home, played, start in rows
    ])


@pytest.fixture
def feed(monkeypatch):
    state = {"schedule": december()}

    def games(year, *, completed_only=True, **_):
        frame = state["schedule"].copy()
        return frame[frame["played"]].reset_index(drop=True) if completed_only else frame

    for module in (board, tracker, slate, sitedata, gamedaydata):
        monkeypatch.setattr(module.cfbd, "games", games)
    monkeypatch.setattr(board.lines_module, "preferred_lines", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(board, "_previous_season", lambda year: pd.Series(POWER))
    return state


RATINGS = SimpleNamespace(power=pd.Series(POWER), home_field=2.5)


# ---- the betting board and the forward log

def test_the_board_takes_army_navy_before_the_bowls(feed) -> None:
    card = board.build_board(2026, RATINGS)
    assert card["week"] == 16 and "label" not in card
    assert [(g["away"], g["home"]) for g in card["games"]] == [("Army", "Navy")]


def test_the_bowls_are_the_block_after_the_last_regular_week(feed) -> None:
    feed["schedule"] = december(army_navy_played=True)
    card = board.build_board(2026, RATINGS)
    assert card["week"] == 17 and card["label"] == BOWLS
    assert {g["home"] for g in card["games"]} == {"Alabama", "Georgia"}
    assert all(g["week"] == 17 and g["label"] == BOWLS for g in card["games"])


def test_a_bowl_pick_is_logged_against_the_bowl_not_the_week_1_game(feed, tmp_path) -> None:
    feed["schedule"] = december(army_navy_played=True)
    bowl = {"week": 17, "label": BOWLS, "home": "Alabama", "away": "Ohio State", "neutral": True,
            "predicted": -5.0, "market": -1.0, "marketOpen": -1.0, "edge": -4.0}
    path = tmp_path / "picks.json"
    tracker.update_log(2026, {"flagged": [bowl]}, path, now=dt.datetime(2026, 12, 14, tzinfo=dt.timezone.utc))
    pick = json.loads(path.read_text())["picks"][0]
    assert pick["game_id"] == 9001
    assert pick["week"] == 17 and pick["label"] == BOWLS


def test_regular_season_picks_still_match_on_the_feeds_week(feed) -> None:
    # Picks logged before the board used blocks carry the feed's week; for the
    # regular season it is the same number.
    schedule = december().set_index("game_id")
    assert tracker._game_id(schedule, {"week": 1, "home": "Alabama", "away": "Ohio State"}) == 1
    assert tracker._game_id(schedule, {"week": 16, "home": "Navy", "away": "Army"}) == 1601


# ---- the slate page

def slate_payload():
    teams = [{"team": t, "power": p, "rank": i + 1} for i, (t, p) in enumerate(sorted(POWER.items(), key=lambda x: -x[1]))]
    return {"teams": teams, "homeField": 2.5}


def weekly_through(block: int, power: dict) -> pd.DataFrame:
    return pd.DataFrame({"team": list(power), "power": list(power.values()), "week": block, "home_field": 2.0})


def test_the_slate_is_army_navy_not_week_1_of_the_bowls(feed) -> None:
    page = slate.build(2026, slate_payload(), {}, None, weekly_through(15, POWER))
    assert page["week"] == 16 and "label" not in page
    ids = {g["id"] for day in page["days"] for g in day["games"]} | {g["id"] for g in page["results"]}
    assert ids == {1601}


def test_the_bowl_slate_is_labelled_and_played_bowls_use_the_last_regular_ratings(feed) -> None:
    feed["schedule"] = december(army_navy_played=True, bowl_played=True)
    entering = {**POWER, "Alabama": 0.0, "Ohio State": 30.0}
    weekly = pd.concat([weekly_through(15, POWER), weekly_through(16, entering)])
    page = slate.build(2026, slate_payload(), {}, None, weekly)
    assert page["week"] == 17 and page["label"] == BOWLS
    assert [g["id"] for g in page["results"]] == [9001]
    assert [g["id"] for day in page["days"] for g in day["games"]] == [9002]
    # Neutral site, entering-the-bowls ratings: Alabama 0 against Ohio State 30.
    assert page["results"][0]["predicted"] == -30.0


def test_archived_finals_are_the_blocks_not_the_feeds_week(feed) -> None:
    schedule = december(army_navy_played=True, bowl_played=True)
    assert set(slatearchive.finals(schedule, 17)) == {9001}
    assert set(slatearchive.finals(schedule, 1)) == {1}


# ---- team pages

def test_a_bowl_comes_last_on_a_team_page_and_is_labelled(feed) -> None:
    feed["schedule"] = december(army_navy_played=True, bowl_played=True)
    payload = {"teams": [{"team": t, "power": p, "rank": 1} for t, p in POWER.items()], "homeField": 2.5}
    details = sitedata.team_details(2026, payload)
    played = details["Alabama"]["played"]
    assert [g["week"] for g in played] == [1, 15, 17]
    assert played[-1]["label"] == BOWLS and "label" not in played[0]
    assert [g["week"] for g in details["Georgia"]["upcoming"]] == [17]


# ---- College GameDay

def test_gameday_forecasts_only_the_regular_season(feed, monkeypatch) -> None:
    seen = {}

    def fake_forecast(teams, schedule, *, weeks, **_):
        seen.setdefault("schedules", []).append(schedule)
        seen.setdefault("weeks", []).append(list(weeks))
        return {"sims": 1, "weeks": {}, "teams": {}}

    model = {"armyNavy": None, "features": [], "coefficients": [], "weeks": [], "fitted": None, "otherRate": 0.0}
    monkeypatch.setattr(gamedaydata.forecast, "load_model", lambda: model)
    monkeypatch.setattr(gamedaydata.forecast, "forecast", fake_forecast)
    monkeypatch.setattr(gamedaydata.history, "load", lambda: {"announced2026": [
        {"week": 15, "date": "2026-12-05", "teams": ["Alabama", "Georgia"], "host": "Georgia"}]})
    monkeypatch.setattr(gamedaydata, "_context", lambda *a: {})

    page = gamedaydata.build(2026, {"teams": [{"team": t, "power": p, "conference": "X"} for t, p in POWER.items()],
                                    "homeField": 2.5, "week": 15})
    assert page is not None
    assert all((s["season_type"] == "regular").all() for s in seen["schedules"])
    # Week 15's announcement has been played through; with the bowls taken out,
    # week 1 is not "still open" and nothing is re-checked.
    assert len(seen["weeks"]) == 1


# ---- the pages' labels

def test_labelled_weeks_read_as_their_label() -> None:
    assert site.week_name({"week": 7}) == "Week 7"
    assert site.week_name({"week": 17, "label": BOWLS}) == "Bowls"
    assert site.week_cell({"week": 7}) == "7"
    assert site.week_cell({"week": 17, "label": BOWLS}) == "Bowls"
