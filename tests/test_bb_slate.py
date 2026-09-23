"""The basketball slate, the two tracked betting habits, and the bid stakes behind the key games."""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from mri.betting import bb_tracker
from mri.export import bb_slate, live


def game(gid=1, home="Hosts", away="Visitors", predicted=-2.0, market=3.0, start="2026-02-28T19:00:00Z", **extra):
    return {"id": gid, "home": home, "away": away, "neutral": False, "predicted": predicted, "market": market,
            "marketOpen": None, "edge": None, "winProbability": 0.43, "start": start, "day": start[:10], **extra}


# ---- the habits

def test_underdog_pick_is_the_market_dog_the_model_has_winning() -> None:
    assert bb_tracker.dog_pick(game(predicted=-2.0, market=3.0)) == "away"    # market likes home, model likes away
    assert bb_tracker.dog_pick(game(predicted=2.0, market=-3.0)) == "home"
    assert bb_tracker.dog_pick(game(predicted=5.0, market=3.0)) is None       # they agree on the winner
    assert bb_tracker.dog_pick(game(market=None)) is None and bb_tracker.dog_pick(game(market=0.0)) is None


def test_fade_list_needs_ten_games_and_a_bad_cover_rate() -> None:
    rows = [{"home": "Chicago State", "away": f"Opp{i}", "market": -10.0, "actual": -20.0} for i in range(8)]
    rows += [{"home": "Chicago State", "away": "Opp8", "market": -10.0, "actual": 0.0},
             {"home": "Chicago State", "away": "Opp9", "market": -10.0, "actual": -10.0}]    # a push: not counted
    assert bb_tracker.fade_list(pd.DataFrame(rows)) == {}                                   # nine graded games
    rows.append({"home": "Chicago State", "away": "Opp10", "market": -10.0, "actual": -30.0})
    listed = bb_tracker.fade_list(pd.DataFrame(rows))
    assert list(listed) == ["Chicago State"] and listed["Chicago State"]["covers"] == 1
    assert listed["Chicago State"]["misses"] == 9 and listed["Chicago State"]["coverRate"] == 0.1


def test_fading_both_teams_is_no_bet() -> None:
    fades = {"Hosts": {}, "Visitors": {}}
    assert bb_tracker.fade_picks(game(), fades) == []
    assert bb_tracker.fade_picks(game(), {"Hosts": {}}) == [("away", "Hosts")]


def test_picks_are_logged_once_and_graded_with_closing_line_value(tmp_path) -> None:
    path = tmp_path / "picks.json"
    fades = {"Hosts": {"covers": 2, "misses": 9}}
    g = game(predicted=-2.0, market=3.0)                      # dog pick: away +3; fade Hosts: also away +3
    first = bb_tracker.update(path, [g], fades, {}, {}, logged_at="t0")
    again = bb_tracker.update(path, [{**g, "market": 1.0}], fades, {}, {}, logged_at="t1")
    assert first["dog"]["logged"] == again["dog"]["logged"] == 1 and again["fade"]["logged"] == 1
    assert json.loads(path.read_text())["picks"][0]["taken"] == 3.0          # the morning's line is kept, never edited
    # Home won by 2, so the away side at +3 covers. The line closed at home -4 (away +4): the market moved
    # away from the side taken, so the bet gave up a point of closing line value.
    done = bb_tracker.update(path, [], fades, {1: 2.0}, {1: 4.0}, logged_at="t2")
    assert done["dog"]["wins"] == 1 and done["dog"]["units"] == round(100 / 110, 2)
    assert done["dog"]["clv"] == -1.0
    push = bb_tracker._grade({"side": "home", "taken": 3.0}, 3.0, None)
    assert push["result"] == "push" and push["units"] == 0.0


# ---- the slate

def payload(bracket=None):
    teams = [{"team": n, "rank": r, "power": p, "conference": c, "abbreviation": n[:4].upper(), "color": "#123456"}
             for n, r, p, c in (("Hosts", 12, 20.0, "Big East"), ("Visitors", 40, 12.0, "Big East"),
                                ("Other", 80, 2.0, "MAAC"), ("Rival", 90, 1.0, "MAAC"))]
    return {"sport": "basketball", "season": 2026, "teams": teams, "bracketology": bracket}


def test_slate_is_today_and_tomorrow_with_best_games_before_bracketology() -> None:
    board = {"games": [game(1), game(2, "Other", "Rival", 1.0, 1.5, "2026-02-28T23:00:00Z"),
                       game(3, start="2026-03-01T20:00:00Z"), game(4, start="2026-03-05T20:00:00Z")]}
    now = dt.datetime(2026, 2, 28, 11, tzinfo=dt.timezone.utc)
    slate = bb_slate.build(2026, payload(), board, {"fades": {}}, now=now)
    assert slate["date"] == "2026-02-28" and [d["date"] for d in slate["days"]] == ["2026-02-28", "2026-03-01"]
    assert slate["games"] == 2 and slate["watchKind"] == "best" and slate["watch"][0] == 1   # the better game first
    first = slate["days"][0]["games"][0]
    assert first["time"] == "2:00 PM ET" and first["tracked"] == {"dog": "Visitors"}
    assert first["conferences"] == ["Big East"]


def test_key_games_switch_to_bid_stakes_once_bracketology_runs() -> None:
    bracket = {"frozen": False, "teams": [{"team": "Hosts", "pField": 0.55}],
               "leverage": {"1": {"home": {"ifWin": 0.7, "ifLose": 0.35}, "away": {"ifWin": 0.1, "ifLose": 0.05},
                                  "swing": 0.35}}}
    now = dt.datetime(2026, 2, 28, 11, tzinfo=dt.timezone.utc)
    slate = bb_slate.build(2026, payload(bracket), {"games": [game(1)]}, None, now=now)
    g = slate["days"][0]["games"][0]
    assert slate["watchKind"] == "stakes" and slate["watch"] == [1]
    assert g["stake"] == {"side": "home", "team": "Hosts", "ifWin": 0.7, "ifLose": 0.35, "swing": 0.35, "now": 0.55}


def test_yesterdays_slate_is_saved_once_with_only_its_own_games(tmp_path) -> None:
    old = {"season": 2026, "date": "2026-02-28", "watch": [], "games": 1,
           "days": [{"date": "2026-02-28", "games": [{"id": 1}]}, {"date": "2026-03-01", "games": [{"id": 2}]}]}
    (tmp_path / "slate.json").write_text(json.dumps(old))
    (tmp_path / "basketball.json").write_text(json.dumps({"season": 2026, "teams": []}))
    finals = lambda day: {1: (70, 65), 2: (60, 61)}                                          # noqa: E731
    assert bb_slate.snapshot(tmp_path, {"date": "2026-02-28"}, finals) is None              # same day
    path = bb_slate.snapshot(tmp_path, {"date": "2026-03-01"}, finals)
    saved = json.loads(path.read_text())
    assert path.name == "2026-02-28.json" and [d["date"] for d in saved["days"]] == ["2026-02-28"]
    assert saved["finals"] == {"1": {"home": 70, "away": 65}}
    assert bb_slate.snapshot(tmp_path, None, finals) is None


# ---- live basketball and the bid leverage

def test_basketball_live_uses_halves_and_its_own_spread() -> None:
    from mri.betting import bb_board
    bb = live.BASKETBALL
    assert bb.sigma == bb_board.SIGMA
    assert live.seconds_left(1, "20:00", sport=bb) == 2400 and live.seconds_left(2, "5:00", sport=bb) == 300
    assert live.status_label("in_progress", 1, "0:00", sport=bb) == "Half"
    assert live.status_label("in_progress", 2, "4:12", sport=bb) == "2nd 4:12"
    assert live.status_label("final", 3, "0:00", sport=bb) == "Final/OT"
    g = {"home": "Hosts", "away": "Visitors", "homeWinProbability": 0.9}
    assert live.upset(g, 30, 35, 1, False, {}, sport=bb) is None                  # first half
    assert "before tip-off" in live.upset(g, 50, 55, 2, False, {}, sport=bb)


def test_basketball_live_record_is_keyed_by_the_day() -> None:
    slate = {"season": 2026, "date": "2026-02-28", "days": [{"games": [
        {"id": 7, "home": "Hosts", "away": "Visitors", "predicted": 4.0, "homeWinProbability": 0.64,
         "date": "2026-02-28", "sort": "1900"}]}]}
    now = dt.datetime(2026, 2, 28, 19, 30, tzinfo=live.EASTERN)
    board = [{"id": 7, "status": "final", "period": 2, "clock": "0:00", "homeTeam": {"points": 70}, "awayTeam": {"points": 60}}]
    rec = live.merge(slate, board, None, {}, now, sport=live.BASKETBALL)
    assert rec["key"] == "2026-02-28" and rec["games"]["7"]["status"] == "completed"
    assert not live.should_poll(slate, rec, now, sport=live.BASKETBALL)


def test_bid_leverage_reads_the_worlds_where_each_side_wins() -> None:
    from mri.bracket import joint

    future = pd.DataFrame({"game_id": [11, 12], "team1": ["A", "C"], "team2": ["B", "D"],
                           "start_date": ["2026-02-28T19:00Z", "2026-03-09T19:00Z"]})
    S = 200
    home_won = np.zeros((2, S))
    home_won[0, :100] = 1                                  # B wins game 11 in the first 100 worlds
    field = np.zeros((4, S), dtype=bool)
    field[1, :80] = True                                   # B makes the field in 80 of its 100 wins...
    field[1, 100:120] = True                               # ...and 20 of its 100 losses
    lev = joint._leverage(future, home_won, field, {"A": 0, "B": 1, "C": 2, "D": 3}, "2026-03-01")
    assert list(lev) == ["11"]                             # game 12 is past the window
    assert lev["11"]["home"] == {"ifWin": 0.8, "ifLose": 0.2} and lev["11"]["swing"] == 0.6


# ---- which tournament a game is part of

def test_events_are_named_from_the_feed() -> None:
    ev = lambda **g: bb_slate.event({"start": "2026-03-19T16:00:00Z", **g})                     # noqa: E731
    ncaa = ev(notes="NCAA Men's Basketball Championship - South Region - Sweet 16", tournament="NCAA", seasonType="postseason")
    assert ncaa["kind"] == "ncaa" and ncaa["label"] == "NCAA Tournament · South · Sweet 16"
    assert ncaa["detail"] == "South · Sweet 16"
    final_four = ev(notes="NCAA Men's Basketball Championship - Final Four", tournament="NCAA", seasonType="postseason")
    assert final_four["label"] == "NCAA Tournament · Final Four" and final_four["region"] is None
    assert ev(notes="NIT - 2nd Round", tournament="NIT", seasonType="postseason")["label"] == "NIT · 2nd Round"
    crown = ev(notes="College Basketball Crown Championship Game", seasonType="postseason")
    assert crown["kind"] == "crown" and crown["label"] == "College Basketball Crown · Championship Game"
    # A conference tournament is named from the conference, not the sponsor in the note.
    big12 = ev(notes="Phillips 66 Big 12 Tournament - Semifinal", seasonType="regular", gameType="TRNMNT",
               homeConference="Big 12", awayConference="Big 12", start="2026-03-13T16:00:00Z")
    assert big12["kind"] == "conference" and big12["label"] == "Big 12 Tournament · Semifinal"
    mte = ev(notes="Player Era Festival", seasonType="regular", gameType="TRNMNT", homeConference="SEC",
             awayConference="Big East", start="2025-11-25T16:00:00Z")
    assert mte["kind"] == "event" and mte["label"] == "Player Era Festival"
    assert ev(notes=None, seasonType="regular", gameType="STD") is None


def test_tournament_days_are_grouped_by_tournament_with_seeds() -> None:
    from mri.export import site

    teams = [{"team": n, "rank": 30 + i, "power": 10.0, "conference": "SEC", "abbreviation": n[:3].upper(), "color": "#123456"}
             for i, n in enumerate(("Aa", "Bb", "Cc", "Dd", "Ee", "Ff"))]
    p = {"sport": "basketball", "season": 2026, "seasonLabel": "2025-26", "periodLabel": "Final", "teams": teams,
         "sports": ["football", "basketball"], "generated": "2026-03-19T11:00:00Z", "gamesRated": 1, "homeField": 3.0,
         "week": 20, "conferences": []}
    base = {"neutral": True, "predicted": 1.0, "market": None, "marketOpen": None, "edge": None, "winProbability": 0.54,
            "day": "2026-03-19"}
    games = [
        {**base, "id": 1, "home": "Aa", "away": "Bb", "start": "2026-03-19T17:00:00Z", "tournament": "NIT",
         "seasonType": "postseason", "notes": "NIT - 1st Round"},
        {**base, "id": 2, "home": "Cc", "away": "Dd", "start": "2026-03-19T20:00:00Z", "tournament": "NCAA",
         "seasonType": "postseason", "notes": "NCAA Men's Basketball Championship - East Region - 1st Round",
         "homeSeed": 1, "awaySeed": 16},
        {**base, "id": 3, "home": "Ee", "away": "Ff", "start": "2026-03-19T23:00:00Z", "seasonType": "regular", "gameType": "STD"},
    ]
    now = dt.datetime(2026, 3, 19, 11, tzinfo=dt.timezone.utc)
    p["slate"] = bb_slate.build(2026, p, {"games": games}, None, now=now)
    text = site.slate_page(p)
    heads = [h.split("</b>")[0] for h in text.split('<p class="slslothead"><b>')[1:]]
    assert heads == ["NCAA Tournament", "NIT", "Other games"]
    assert "East · 1st Round" in text and '<span class="slseed" title="Seed">16</span>' in text
    assert 'id="slevents"' in text and 'data-event="NCAA Tournament"' in text
    # The NCAA field is set, so there are no bids left to win or lose: no stake column at all.
    assert p["slate"]["bidStakes"] is False and ">Bid stake<" not in text and 'class="slst"' not in text
    assert "nostakes" in text


def test_bid_stakes_show_until_the_field_is_set() -> None:
    board = {"games": [game(1)]}
    now = dt.datetime(2026, 2, 28, 11, tzinfo=dt.timezone.utc)
    assert bb_slate.build(2026, payload(), board, None, now=now)["bidStakes"] is True
    frozen = {"frozen": True, "teams": [], "leverage": {}}
    assert bb_slate.build(2026, payload(frozen), board, None, now=now)["bidStakes"] is False
