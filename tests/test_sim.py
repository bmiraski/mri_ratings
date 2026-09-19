"""The season simulation must obey the rules of the game it is simulating.

These are invariants, not accuracy claims - accuracy is what
scripts/backtest_simulation.py is for. What can be checked cheaply and exactly is
that every run picks twelve teams, crowns one champion, gives each conference one
champion, and never lets a conference champion miss a field it is guaranteed
into. A bracket that sometimes seats eleven teams would still look plausible in a
table, which is why these are tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mri.sim import season

CONFERENCES = {"SEC": 8, "Big Ten": 8, "ACC": 8, "Big 12": 8, "Sun Belt": 8}


def league(seed: int = 0):
    """Forty-one teams: five conferences of eight and Notre Dame."""
    rng = np.random.default_rng(seed)
    teams = []
    for conference, size in CONFERENCES.items():
        for i in range(size):
            teams.append({"team": f"{conference} {i}", "power": float(rng.normal(0, 8)),
                          "conference": conference})
    teams.append({"team": "Notre Dame", "power": 12.0, "conference": "FBS Independent"})

    games = []
    for conference in CONFERENCES:
        members = [t["team"] for t in teams if t["conference"] == conference]
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                games.append((members[a], members[b], True))
    games.append(("Notre Dame", "SEC 0", False))
    frame = pd.DataFrame(
        [{"game_id": i, "week": 5, "team1": a, "team2": b, "played": False,
          "pts1": np.nan, "pts2": np.nan, "neutral": False, "conference_game": c}
         for i, (a, b, c) in enumerate(games)]
    )
    return teams, frame


@pytest.fixture(scope="module")
def result():
    teams, schedule = league()
    return teams, schedule, season.simulate(teams, schedule, home_field=3.0, sims=3000)


def test_every_run_picks_twelve_and_one_champion(result) -> None:
    _, _, out = result
    odds = out["teams"].values()
    assert sum(t["playoff"] for t in odds) == pytest.approx(season.FIELD, abs=5e-3)
    assert sum(t["title"] for t in odds) == pytest.approx(1.0, abs=5e-3)


def test_each_conference_with_a_title_game_has_one_champion(result) -> None:
    teams, _, out = result
    for conference in CONFERENCES:
        members = [t["team"] for t in teams if t["conference"] == conference]
        assert sum(out["teams"][m]["conferenceTitle"] for m in members) == pytest.approx(1.0, abs=5e-3)
        assert sum(out["teams"][m]["conferenceGame"] for m in members) == pytest.approx(2.0, abs=5e-3)


def test_the_bracket_only_narrows(result) -> None:
    _, _, out = result
    for name, t in out["teams"].items():
        chain = [t["playoff"], t["quarterfinal"], t["semifinal"], t["final"], t["title"]]
        assert chain == sorted(chain, reverse=True), name
        assert t["bye"] <= t["playoff"] + 1e-9, name


def test_a_power_conference_champion_is_always_in_the_field(result) -> None:
    teams, _, out = result
    for t in teams:
        if t["conference"] in season.POWER_FOUR:
            o = out["teams"][t["team"]]
            assert o["playoff"] >= o["conferenceTitle"] - 1e-9, t["team"]


def test_the_best_group_of_six_team_is_always_in(result) -> None:
    teams, _, out = result
    sun_belt = [t["team"] for t in teams if t["conference"] == "Sun Belt"]
    assert sum(out["teams"][m]["playoff"] for m in sun_belt) >= 1.0 - 1e-9


def test_stronger_teams_do_better(result) -> None:
    teams, _, out = result
    ordered = sorted((t for t in teams if t["conference"] == "SEC"), key=lambda t: t["power"])
    weakest, strongest = ordered[0]["team"], ordered[-1]["team"]
    assert out["teams"][strongest]["title"] > out["teams"][weakest]["title"]
    assert out["teams"][strongest]["conferenceTitle"] > out["teams"][weakest]["conferenceTitle"]


def test_same_inputs_give_the_same_page() -> None:
    teams, schedule = league()
    a = season.simulate(teams, schedule, home_field=3.0, sims=1500)
    b = season.simulate(teams, schedule, home_field=3.0, sims=1500)
    assert a == b


def test_winning_a_game_helps(result) -> None:
    teams, schedule, _ = result
    tracked = [int(g) for g in schedule["game_id"][:6]]
    out = season.simulate(teams, schedule, home_field=3.0, sims=4000, track=tracked)
    assert out["leverage"], "no game had both outcomes represented"
    for entry in out["leverage"].values():
        for side in ("home", "away"):
            if side in entry:
                assert entry[side]["ifWin"] >= entry[side]["ifLose"] - 0.02


def test_a_known_title_game_result_is_final() -> None:
    teams, schedule = league()
    a, b = "SEC 0", "SEC 1"
    out = season.simulate(
        teams, schedule, home_field=3.0, sims=1000,
        championships={"SEC": {"a": a, "b": b, "neutral": True, "played": True, "a_won": False}},
    )
    assert out["teams"][b]["conferenceTitle"] == 1.0
    assert out["teams"][a]["conferenceTitle"] == 0.0
    assert out["teams"][a]["conferenceGame"] == out["teams"][b]["conferenceGame"] == 1.0


def test_a_scheduled_title_game_keeps_its_participants() -> None:
    teams, schedule = league()
    out = season.simulate(
        teams, schedule, home_field=3.0, sims=1000,
        championships={"Big Ten": {"a": "Big Ten 2", "b": "Big Ten 5", "neutral": True, "played": False}},
    )
    assert out["teams"]["Big Ten 2"]["conferenceGame"] == 1.0
    assert out["teams"]["Big Ten 5"]["conferenceGame"] == 1.0
    others = [f"Big Ten {i}" for i in (0, 1, 3, 4, 6, 7)]
    assert all(out["teams"][o]["conferenceGame"] == 0.0 for o in others)


def test_rating_uncertainty_widens_the_odds_early() -> None:
    """With the error bars off, the favourite is more certain of everything."""
    teams, schedule = league()
    honest = season.simulate(teams, schedule, home_field=3.0, sims=4000)
    exact = season.simulate(teams, schedule, home_field=3.0, sims=4000, rules={"uncertainty": 0.0})
    best = max(teams, key=lambda t: t["power"])["team"]
    assert exact["teams"][best]["title"] >= honest["teams"][best]["title"] - 0.02
