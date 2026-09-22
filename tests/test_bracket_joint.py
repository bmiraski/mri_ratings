import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mri.bracket import atlarge, joint, seeding
from mri.bracket.template import Template

ROOT = Path(__file__).resolve().parents[1]
CONFS = [f"C{i}" for i in range(8)]
TEMPLATE = Template(tiers=(8, 4), seasons_seen=3, stable=True)


def _league(seed: int = 0):
    """Eight 12-team conferences: a single round robin in each, six non-conference games per team, all
    played. Team 0 of each conference is much the best, so its standings lead is never in doubt."""
    rng = np.random.default_rng(seed)
    teams = {f"{c}-T{j}": c for c in CONFS for j in range(12)}
    power = pd.Series({t: (25.0 if t.endswith("-T0") else 0.0) + rng.normal(0, 6) - 1.2 * int(t.split("T")[-1])
                       for t in teams})
    rows = []
    day = pd.Timestamp("2024-11-05")

    def game(a, b, conf_game, when):
        p_home = 1 / (1 + np.exp(-(power[b] - power[a] + 3) / 6))
        home_won = rng.random() < p_home
        if conf_game and (a.endswith("-T0") or b.endswith("-T0")):
            home_won = b.endswith("-T0")                       # the best team never drops a league game
        rows.append({"game_id": len(rows), "season": 2025, "season_type": "regular",
                     "start_date": when.strftime("%Y-%m-%dT19:00:00.000Z"), "team1": a, "team2": b, "played": True,
                     "pts1": 60.0 if home_won else 70.0, "pts2": 70.0 if home_won else 60.0,
                     "win1": 0.0 if home_won else 1.0, "win2": 1.0 if home_won else 0.0, "neutral": False,
                     "conf1": teams[a], "conf2": teams[b], "tournament": None, "game_type": "STD"})

    names = list(teams)
    for k, t in enumerate(names):
        others = [o for o in names if teams[o] != teams[t]]
        for o in rng.choice(others, 3, replace=False):
            game(t, o, False, day + pd.Timedelta(days=int(rng.integers(0, 40))))
    for c in CONFS:
        members = [t for t in names if teams[t] == c]
        for i, (a, b) in enumerate(itertools.combinations(members, 2)):
            game(a, b, True, pd.Timestamp("2025-01-02") + pd.Timedelta(days=i % 60))
    return pd.DataFrame(rows), power, teams


def _inputs(games, power, teams, **kw):
    beta = np.array(json.loads((ROOT / "data" / "atlarge_model.json").read_text())["coefficients"])
    base = dict(season=2025, games=games, power=power, home_field=3.0, resume_sigma=11.0, conference_of=teams,
                templates={c: TEMPLATE for c in CONFS}, beta=beta, fmt=seeding.FORMATS[68], sims=300, seed=1)
    base.update(kw)
    return joint.Inputs(**base)


@pytest.fixture(scope="module")
def league():
    return _league()


def test_every_world_fills_the_field_exactly(league):
    res = joint.run(_inputs(*league, as_of="2025-02-01"))
    assert res.teams["pField"].sum() == pytest.approx(68)
    assert res.teams["pAuto"].sum() == pytest.approx(len(CONFS))
    for odds in res.champions.values():
        assert sum(odds.values()) == pytest.approx(1)
    seed_cols = [f"seed{k}" for k in range(1, 17)]
    assert np.allclose(res.teams[seed_cols].sum(axis=1), res.teams["pField"])


def test_placeholder_gives_the_standings_leader_the_bid(league):
    res = joint.run(_inputs(*league))                         # whole regular season played, no tournament yet
    for c in CONFS:
        assert res.champions[c] == {f"{c}-T0": 1.0}
        assert res.conference_status[c] == "not started"


def test_model_mode_spreads_the_bid_but_favours_the_leader(league):
    res = joint.run(_inputs(*league, auto_mode={c: "model" for c in CONFS}))
    for c in CONFS:
        odds = res.champions[c]
        assert odds[f"{c}-T0"] == max(odds.values())
    assert sum(len(res.champions[c]) > 1 for c in CONFS) >= 4          # upsets happen in most leagues


def test_tournament_games_land_on_the_resumes(league):
    """With the regular season over, the only games left are the conference tournaments: a 12-team
    bracket is 11 games, two team-games each, in every world."""
    t = joint.run(_inputs(*league)).teams
    extra = t["projectedWins"] + t["projectedLosses"] - t["winsNow"] - t["lossesNow"]
    assert np.allclose(extra.groupby(t["conference"]).sum(), 22)
    assert (t["projectedWins"] >= t["winsNow"]).all()


def _tournament(games, conf, bracket, start="2025-03-05"):
    """Append conference-tournament games: ``bracket`` is a list of (away, home, home_won, day_offset)."""
    rows = []
    for k, (a, b, home_won, offset) in enumerate(bracket):
        rows.append({"game_id": 100000 + k, "season": 2025, "season_type": "regular",
                     "start_date": (pd.Timestamp(start) + pd.Timedelta(days=offset)).strftime("%Y-%m-%dT19:00:00.000Z"),
                     "team1": a, "team2": b, "played": True, "pts1": 60.0 if home_won else 70.0,
                     "pts2": 70.0 if home_won else 60.0, "win1": 0.0 if home_won else 1.0,
                     "win2": 1.0 if home_won else 0.0, "neutral": True, "conf1": conf, "conf2": conf,
                     "tournament": None, "game_type": "TRNMNT"})
    return pd.concat([games, pd.DataFrame(rows)], ignore_index=True)


def test_a_finished_tournament_overrides_the_placeholder(league):
    games, power, teams = league
    t = lambda j: f"C0-T{j}"  # noqa: E731
    # A three-game bracket won by T5: E entrants, E - 1 games, one team unbeaten, nothing left scheduled.
    bracket = [(t(5), t(0), False, 0), (t(2), t(1), False, 0), (t(5), t(2), False, 1)]
    res = joint.run(_inputs(_tournament(games, "C0", bracket), power, teams, templates={**{c: TEMPLATE for c in CONFS},
                            "C0": Template(tiers=(4,), seasons_seen=3, stable=True)}))
    assert res.conference_status["C0"] == "decided"
    assert res.champions["C0"] == {t(5): 1.0}


def test_a_stepladder_in_progress_is_not_mistaken_for_finished(league):
    games, power, teams = league
    t = lambda j: f"C0-T{j}"  # noqa: E731
    # Stepladder: 5 vs 4 has been played; the winner still has to beat the waiting top seeds. After the
    # first game exactly one entrant is unbeaten and E - 1 games are played - "finished" by the count alone.
    bracket = [(t(4), t(3), True, 0), (t(3), t(2), True, 2)]           # the second game is still to come...
    g = _tournament(games, "C0", bracket)
    g.loc[g["game_id"] == 100001, "played"] = False                    # ...so it's on the schedule, not played
    res = joint.run(_inputs(g, power, teams, as_of="2025-03-06",
                            templates={**{c: TEMPLATE for c in CONFS},
                                       "C0": Template(tiers=(2, 1, 1, 1), seasons_seen=3, stable=True)}))
    assert res.conference_status["C0"] == "underway"
    assert t(4) not in res.champions["C0"]                             # the loser is out
    assert len(res.champions["C0"]) > 1


def test_committee_noise_widens_the_bubble(league):
    sharp = joint.run(_inputs(*league))
    blurred = joint.run(_inputs(*league, committee_noise=1.25))
    between = lambda r: int(((r.teams["pField"] > 0.02) & (r.teams["pField"] < 0.98)).sum())  # noqa: E731
    assert between(blurred) > between(sharp)
    assert blurred.teams["pField"].sum() == pytest.approx(68)


@pytest.mark.parametrize("quad", [1, 2, 3, 4])
@pytest.mark.parametrize("road", [False, True])
@pytest.mark.parametrize("record", [(14, 14), (20, 10), (27, 4)])
def test_turning_a_loss_into_a_win_never_hurts_under_the_committed_model(quad, road, record):
    """The committed coefficients have two signs that read backwards on their own (road/neutral wins and
    bad-loss rate); what matters in a simulation is the whole score, so check that the net effect of
    winning a game instead of losing it is always a better (lower) score."""
    beta = np.array(json.loads((ROOT / "data" / "atlarge_model.json").read_text())["coefficients"])
    wins, losses = record
    base = {"wins": wins, "losses": losses + 1, "powerRank": 40, "resume": 2.0, "quad1_wins": 3, "quad1_losses": 4,
            "bad_losses": 2, "sos": 90, "road_neutral_wins": 5}
    lost = dict(base)
    if quad == 1:
        lost["quad1_losses"] += 1
    if quad >= 3:
        lost["bad_losses"] += 1
    won = dict(base, wins=wins + 1, losses=losses, resume=base["resume"] + 1.0)
    if quad == 1:
        won["quad1_wins"] += 1
    if road:
        won["road_neutral_wins"] += 1
    as_arrays = lambda d: {k: np.array([v], dtype=float) for k, v in d.items()}  # noqa: E731
    s_won = atlarge.score_columns(atlarge.columns(as_arrays(won)), beta)[0]
    s_lost = atlarge.score_columns(atlarge.columns(as_arrays(lost)), beta)[0]
    assert s_won < s_lost


def test_independents_get_no_automatic_bid(league):
    games, power, teams = league
    teams = dict(teams)
    for t in [n for n in teams if n.startswith("C7-")]:
        teams[t] = "Indep."
    g = games.copy()
    g.loc[g["conf1"] == "C7", "conf1"] = "Indep."
    g.loc[g["conf2"] == "C7", "conf2"] = "Indep."
    res = joint.run(_inputs(g, power, teams, templates={c: TEMPLATE for c in CONFS[:7]}))
    assert "Indep." not in res.champions
    assert res.teams["pAuto"].sum() == pytest.approx(7)
    assert res.teams["pField"].sum() == pytest.approx(68)
