"""The Hidden Heisman: the week's biggest game from outside the power conferences."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mri.export import hidden_heisman as hh


def line(player, team, opponent, week=3, **stats):
    base = {"season": 2026, "week": week, "game_id": 1, "player_id": player, "player": player, "team": team,
            "opponent": opponent, "points": 42, "opp_points": 17, **{k: 0.0 for k in hh.STATS}}
    return {**base, **{k: float(v) for k, v in stats.items()}}


CONF = {"Georgia": "SEC", "JMU": "Sun Belt", "Boise State": "Mountain West", "Notre Dame": "FBS Independents",
        "Army": "American", "Weak U": "Sun Belt", "Strong U": "Big Ten", "Avg U": "MAC"}
POWER = {"Strong U": 20.0, "Avg U": 0.0, "Weak U": -20.0, "Georgia": 25.0, "JMU": 2.0, "Boise State": 5.0,
         "Notre Dame": 15.0, "Army": -1.0}


def test_power_conferences_and_notre_dame_are_out_and_so_is_anyone_over_five_percent() -> None:
    rows = [line("Big SEC Guy", "Georgia", "Avg U", rush_yds=300, rush_td=5),
            line("Irish Guy", "Notre Dame", "Avg U", rush_yds=290, rush_td=5),
            line("Famous", "Boise State", "Avg U", rush_yds=280, rush_td=5),
            line("Hidden", "JMU", "Avg U", rush_yds=200, rush_td=3)]
    won = hh.pick(rows, CONF, 2026, POWER, {("Famous", "Boise State")}, np.array([1.0, 50.0]))
    assert won["player"] == "Hidden" and won["score"] == 38.0 and won["rank"] == 2        # one past game was bigger


def test_the_pac12_counts_as_power_only_through_2023() -> None:
    conf = {"Oregon": "Pac-12", "JMU": "Sun Belt", "Avg U": "MAC"}
    rows = [line("Duck", "Oregon", "Avg U", rush_yds=300, rush_td=4), line("Duke", "JMU", "Avg U", rush_yds=100)]
    assert hh.pick(rows, conf, 2023, {"Avg U": 0.0, "JMU": 0.0, "Oregon": 0.0}, set(), np.zeros(0))["player"] == "Duke"
    assert hh.pick(rows, conf, 2026, {"Avg U": 0.0, "JMU": 0.0, "Oregon": 0.0}, set(), np.zeros(0))["player"] == "Duck"


def test_a_good_opponent_counts_for_more_and_an_fcs_one_for_less() -> None:
    mean, sd = 0.0, 10.0
    assert hh.opponent_factor(0.0, mean, sd) == 1.0
    assert hh.opponent_factor(40.0, mean, sd) == hh.FACTOR_RANGE[1]            # capped
    assert hh.opponent_factor(None, mean, sd) == hh.FCS_FACTOR
    # The same line against a strong team beats a slightly bigger one against a weak team.
    rows = [line("Tough", "JMU", "Strong U", rush_yds=180, rush_td=2), line("Soft", "Army", "Weak U", rush_yds=195, rush_td=2)]
    assert hh.pick(rows, CONF, 2026, POWER, set(), np.zeros(0))["player"] == "Tough"


def test_facts_are_checked_against_the_whole_week() -> None:
    rows = [line("Hidden", "JMU", "Avg U", pass_yds=295, pass_td=2, rush_yds=153, rush_td=4),
            line("Big SEC Guy", "Georgia", "Avg U", pass_yds=400, pass_td=5)]
    facts = hh.pick(rows, CONF, 2026, POWER, set(), np.zeros(0))["facts"]
    # Most touchdowns and most total yards, both true; not the most passing yards, where Georgia had more.
    assert facts == ["6 total touchdowns, the most by any FBS player this week",
                     "448 total yards, the most by any FBS player this week"]


def test_each_finished_week_is_decided_once(tmp_path) -> None:
    payload = {"teams": [{"team": t, "conference": c, "power": POWER[t]} for t, c in CONF.items()],
               "heisman": {"players": [{"player": "Famous", "team": "Boise State", "win": 0.2}]}}
    schedule = pd.DataFrame({"week": [1, 2, 3], "season_type": "regular", "played": [True, True, False]})
    calls = []

    def fetch(year, week):
        calls.append(week)
        return [line(f"Star{week}", "JMU", "Avg U", week=week, rush_yds=150 + week, rush_td=2),
                line("Famous", "Boise State", "Avg U", week=week, rush_yds=400, rush_td=5)]

    saved = tmp_path / "hh.json"
    out = hh.build(2026, payload, pd.DataFrame(), saved, tmp_path, fetch=fetch, schedule=schedule)
    assert [w["week"] for w in out["winners"]] == [1, 2] and calls == [1, 2]       # week 3 is not over
    assert out["latest"]["player"] == "Star2"                                       # Famous is over 5%
    hh.build(2026, payload, pd.DataFrame(), saved, tmp_path, fetch=fetch, schedule=schedule)
    assert calls == [1, 2]                                                           # never fetched or decided again
    stored = json.loads(saved.read_text())["2026"]
    assert set(stored["weeks"]) == {"1", "2"} and "award" not in stored                # the season is not over


def test_the_season_award_waits_for_the_last_regular_season_game(tmp_path) -> None:
    payload = {"teams": [{"team": t, "conference": c, "power": POWER[t]} for t, c in CONF.items()],
               "heisman": {"players": [{"player": "Famous", "team": "Boise State", "win": 0.2}]}}

    def fetch(year, week):
        return [line("Steady", "JMU", "Avg U", week=week, rush_yds=140, rush_td=2),        # every week, never the best
                line("OneHit", "Army", "Avg U", week=week, rush_yds=300 if week == 1 else 20, rush_td=4 if week == 1 else 0),
                line("Famous", "Boise State", "Avg U", week=week, rush_yds=400, rush_td=5)]

    saved = tmp_path / "hh.json"
    open_season = pd.DataFrame({"week": [1, 2, 3], "season_type": "regular", "played": [True, True, False]})
    assert hh.build(2026, payload, pd.DataFrame(), saved, tmp_path, fetch=fetch, schedule=open_season)["award"] is None
    over = pd.DataFrame({"week": [1, 2, 3], "season_type": "regular", "played": True})
    out = hh.build(2026, payload, pd.DataFrame(), saved, tmp_path, fetch=fetch, schedule=over)
    award = out["award"]
    assert award["player"] == "Steady" and award["games"] == 3 and award["line"]["rush_yds"] == 420
    assert award["weeksWon"] == [2, 3]                  # OneHit took week 1; Famous is over 5% and never eligible
    assert out["awards"] == [award]


def test_the_season_award_shows_on_the_heisman_page_and_the_team_page() -> None:
    from mri.export import site

    team = {"team": "JMU", "conference": "Sun Belt"}
    award = {"season": 2026, "player": "Steady", "team": "JMU", "conference": "Sun Belt", "games": 12,
             "line": {"pass_yds": 0, "pass_td": 0, "rush_yds": 1800, "rush_td": 22, "rec_yds": 200, "rec_td": 2},
             "adjusted": 300.0, "weeksWon": [2, 9], "runnersUp": [], "heismanWin": None}
    weekly = {"season": 2026, "week": 2, "player": "Steady", "team": "JMU", "conference": "Sun Belt", "opponent": "Avg U",
              "opponentRank": 60, "points": 30, "oppPoints": 10, "line": award["line"], "score": 20, "factor": 1.0,
              "adjusted": 20, "rarity": 0.02, "rank": 2000, "of": 140000, "facts": [], "heismanWin": None}
    payload = {"season": 2026, "hiddenHeisman": {"winners": [weekly], "latest": weekly, "past": [], "award": award,
                                                  "awards": [award]}}
    card, _ = site._hidden_heisman_section(payload, {})
    assert "The Hidden Heisman &middot; 2026 season" in card and "weekly winner in weeks 2, 9" in card
    assert card.index("2026 season") < card.index("Week 2")                  # the season award leads
    assert "2026 season winner" in "".join(site._hidden_highlight(team, payload))
    assert ">Season<" in site._hidden_team_section(team, payload)
