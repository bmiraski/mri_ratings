"""Live scores on the slate, and the weekly slate archive."""

from __future__ import annotations

import datetime as dt
import json

import pytest

from mri.export import live, slatearchive

EASTERN = live.EASTERN


def game(gid=1, *, home="Hosts", away="Visitors", predicted=7.0, p=0.66, date="2026-09-26", sort="1530"):
    return {"id": gid, "home": home, "away": away, "predicted": predicted, "homeWinProbability": p,
            "date": date, "sort": sort, "time": "3:30 PM ET", "neutral": False}


def board(gid=1, status="in_progress", home=14, away=10, period=3, clock="7:30", possession="home"):
    return {"id": gid, "status": status, "period": period, "clock": clock, "possession": possession,
            "homeTeam": {"points": home}, "awayTeam": {"points": away}}


def slate_of(*games, week=4, season=2026):
    return {"season": season, "week": week, "days": [{"date": "2026-09-26", "games": list(games)}]}


def at(hour, minute=0, day=26):
    return dt.datetime(2026, 9, day, hour, minute, tzinfo=EASTERN)


def test_the_live_model_uses_the_boards_spread() -> None:
    from mri.betting import board as board_module
    assert live.SIGMA == board_module.SIGMA


def test_clock_arithmetic() -> None:
    assert live.seconds_left(None, None) == 3600
    assert live.seconds_left(1, "15:00") == 3600
    assert live.seconds_left(3, "7:30") == 900 + 450
    assert live.seconds_left(4, "0:00") == 0
    assert live.seconds_left(5, "") == 0


def test_live_chance_starts_at_the_pregame_chance_and_settles_on_the_score() -> None:
    from statistics import NormalDist
    pregame = NormalDist().cdf(7.0 / live.SIGMA)
    assert live.win_probability(7.0, 0, 3600) == pytest.approx(pregame)
    assert live.win_probability(7.0, -10, 60) < 0.05                   # down ten with a minute left
    assert live.win_probability(-7.0, 3, 0) == 1.0                     # final, whatever the model thought
    assert live.win_probability(0.0, 0, 0) == 0.5
    assert 0.5 < live.win_probability(3.0, 0, 0, overtime=True) < live.win_probability(3.0, 0, 3600)


def test_status_labels() -> None:
    assert live.status_label("in_progress", 3, "7:30") == "Q3 7:30"
    assert live.status_label("in_progress", 2, "0:00") == "Half"
    assert live.status_label("in_progress", 6, "") == "2OT"
    assert live.status_label("completed", 4, "0:00") == "Final"
    assert live.status_label("completed", 5, "0:00") == "Final/OT"


def test_upset_rules() -> None:
    g = game(p=0.85)                                            # Visitors had 15%
    assert live.upset(g, 10, 14, 2, False, {}) is None          # first half: not yet a story
    assert "Visitors leads" in live.upset(g, 10, 14, 3, False, {})
    assert "Visitors won" in live.upset(g, 10, 14, 4, True, {})
    assert live.upset(g, 14, 10, 3, False, {}) is None          # favourite ahead
    even = game(p=0.55)                                         # no big underdog, but a ranked team trailing an unranked one
    assert live.upset(even, 10, 14, 3, False, {"Hosts": 12}) == "#12 Hosts trails unranked Visitors"
    assert live.upset(even, 10, 14, 3, False, {"Hosts": 12, "Visitors": 20}) is None
    assert live.upset(even, 14, 14, 4, False, {"Hosts": 12}) is None


def test_polls_only_around_the_slates_games() -> None:
    s = slate_of(game(sort="1530"))
    assert not live.should_poll(s, None, at(9))
    assert live.should_poll(s, None, at(15, 25))                 # ten minutes before kickoff
    assert live.should_poll(s, None, at(18))
    assert not live.should_poll(s, None, at(21))                 # five hours on, whatever happened
    done = {"season": 2026, "week": 4, "games": {"1": {"status": "completed"}}}
    assert not live.should_poll(s, done, at(18, 30))             # final: stop spending calls
    last_week = {**done, "week": 3}
    assert live.should_poll(s, last_week, at(18, 30))            # an old week's record says nothing about this one


def test_merge_keeps_the_week_and_starts_over_for_a_new_one() -> None:
    thursday = game(1, date="2026-09-24", sort="1930")
    saturday = game(2, p=0.9)
    s = slate_of(thursday, saturday)
    first = live.merge(s, [board(1, "completed", 30, 20, 4, "0:00")], None, {}, at(23, day=24))
    assert first["games"]["1"]["label"] == "Final" and first["games"]["1"]["homeWinProbability"] == 1.0
    second = live.merge(s, [board(2, home=7, away=17), board(99)], first, {}, at(17))
    assert set(second["games"]) == {"1", "2"}                    # Thursday's final is kept; unknown games are not
    assert second["games"]["2"]["upset"] and second["games"]["2"]["possession"] == "home"
    assert second["updatedLabel"] == "5:00 PM ET"
    fresh = live.merge(slate_of(game(3), week=5), [], second, {}, at(12, day=27))
    assert fresh["games"] == {} and fresh["week"] == 5


def test_run_spends_no_call_when_nothing_is_on(tmp_path) -> None:
    (tmp_path / "slate.json").write_text(json.dumps(slate_of(game())))
    (tmp_path / "site.json").write_text(json.dumps({"season": 2026, "teams": [{"team": "Hosts", "rank": 5}]}))
    calls = []

    def fetch(key):
        calls.append(key)
        return [board()]

    assert live.run(tmp_path, now=at(9), key="k", fetch=fetch) == "no games on" and not calls
    assert live.run(tmp_path, now=at(17), key="k", fetch=fetch).startswith("1 in progress")
    written = json.loads((tmp_path / "live.json").read_text())
    assert written["games"]["1"]["home"] == 14 and len(calls) == 1


def test_snapshot_saves_the_outgoing_week_once(tmp_path) -> None:
    old = {**slate_of(game(1), game(2)), "watch": [], "results": [], "fcs": [], "games": 2}
    (tmp_path / "slate.json").write_text(json.dumps(old))
    (tmp_path / "site.json").write_text('{"season": 2026, "teams": [{"team": "Hosts", "rank": 3, "abbreviation": "HOS",'
                                        ' "color": "#123456", "logo": "logos/h.png", "power": NaN}]}')
    finals = lambda week: {1: (21, 24), 2: (35, 3), 77: (1, 0)}                 # noqa: E731
    assert slatearchive.snapshot(tmp_path, {**old, "week": 4}, 2026, finals) is None   # same week: nothing to save
    path = slatearchive.snapshot(tmp_path, {**old, "week": 5}, 2026, finals)
    saved = json.loads(path.read_text())
    assert path.name == "2026-week-4.json"
    assert saved["finals"] == {"1": {"home": 21, "away": 24}, "2": {"home": 35, "away": 3}}
    assert saved["teams"]["Hosts"] == {"rank": 3, "abbreviation": "HOS", "color": "#123456", "logo": "logos/h.png"}
    assert slatearchive.snapshot(tmp_path, None, 2026, finals) is None           # already saved: never rewritten
