"""The Heisman model, the projection, and the forecast that joins them to the season simulation.

Some of these are invariants (probabilities sum to one; a better player on a better team is
more likely to win), some pin the pieces that are easy to get subtly wrong (an early-season
quarterback is a quarterback; a projection is a rate times the games left), and the last group
checks that the committed backtest still says what the documentation says it does.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mri.gameday import model as choice
from mri.heisman import data, features, final_model, forecast, funnel, live, project, snapshots
from mri.ingest import players
from mri.sim import season

ROOT = Path(__file__).resolve().parents[1]
CONFERENCES = {"SEC": 8, "Big Ten": 8, "ACC": 8, "Big 12": 8, "Sun Belt": 8}


def league(seed: int = 0):
    """Forty-one teams (five conferences of eight and Notre Dame) with a conference-only schedule left to play."""
    rng = np.random.default_rng(seed)
    teams = [{"team": f"{c} {i}", "power": float(rng.normal(0, 8)), "conference": c}
             for c, size in CONFERENCES.items() for i in range(size)]
    teams.append({"team": "Notre Dame", "power": 12.0, "conference": "FBS Independent"})
    games = []
    for c in CONFERENCES:
        members = [t["team"] for t in teams if t["conference"] == c]
        games += [(members[a], members[b], True) for a in range(len(members)) for b in range(a + 1, len(members))]
    games.append(("Notre Dame", "SEC 0", False))
    frame = pd.DataFrame([{"game_id": i, "week": 5, "team1": a, "team2": b, "played": False, "pts1": np.nan, "pts2": np.nan,
                           "neutral": False, "conference_game": c} for i, (a, b, c) in enumerate(games)])
    return teams, frame


# ---- features

def test_features_describe_a_player_relative_to_the_field() -> None:
    score = np.array([4000.0, 3000.0, 1500.0, 500.0])
    group = np.array([0, 0, 1, 2])                      # QB, QB, RB, receiver
    X = features.matrix(score, group, np.array([1.0, 30.0, 5.0, 2.0]), np.array([0.0, 3.0, 1.0, 0.0]), np.zeros(4))
    col = {n: i for i, n in enumerate(features.NAMES)}
    assert np.exp(X[:, col["prod_rank"]]).round().tolist() == [1, 2, 3, 4]
    assert np.exp(X[:, col["group_rank"]]).round().tolist() == [1, 2, 1, 1]          # the RB is the best RB
    assert X[0, col["prod_z"]] > 0 > X[3, col["prod_z"]] and X[:, col["prod_z"]].mean() == pytest.approx(0, abs=1e-9)
    assert X[:, col["unbeaten"]].tolist() == [1, 0, 0, 1] and X[:, col["is_rb"]].tolist() == [0, 0, 1, 0]
    assert X[:, col["is_receiver"]].tolist() == [0, 0, 0, 1]


def test_features_work_on_many_simulated_seasons_at_once() -> None:
    rng = np.random.default_rng(0)
    score = rng.uniform(500, 4000, size=(50, 12))
    group = np.array([0] * 6 + [1] * 3 + [2] * 3)
    team_rank = rng.integers(1, 40, size=(50, 12))
    X = features.matrix(score, group, team_rank, 0.0, np.zeros(12))
    assert X.shape == (50, 12, len(features.NAMES))
    one = features.matrix(score[7], group, team_rank[7], 0.0, np.zeros(12))
    assert np.allclose(X[7], one)                                   # the batch is just each season on its own
    assert (np.exp(X[..., 0]).round().min(axis=-1) == 1).all()


# ---- the funnel's quarterback bar

def test_a_september_quarterback_is_still_a_quarterback() -> None:
    table = pd.DataFrame([
        {"player": "Passer", "team": "A", "pass_att": 90, "pass_yds": 1000, "pass_td": 9, "rush_car": 10, "rush_yds": 30, "rush_td": 0,
         "rec_rec": 0, "rec_yds": 0, "rec_td": 0, "def_tot": 0, "def_sacks": 0, "def_tfl": 0, "def_int": 0, "def_pd": 0}])
    assert funnel.classify(table)["group"].iloc[0] == "RB"                                       # 90 passes is not a season's worth
    assert funnel.classify(table, pd.Series([3.0]))["group"].iloc[0] == "QB"                    # it is three games' worth
    assert funnel.classify(table, pd.Series([12.0]))["group"].iloc[0] == "RB"


# ---- projection

def test_a_rate_is_shrunk_toward_the_group_in_proportion_to_how_few_games_it_rests_on() -> None:
    group = np.array([0, 0])
    per_game = np.array([60.0, 20.0])                   # the same two rates, seen after one game and after ten
    few = project.group_rates(per_game * 1, np.array([1.0, 1.0]), group)
    many = project.group_rates(per_game * 10, np.array([10.0, 10.0]), group)
    assert (few[0] - few[1]) == pytest.approx(40 / 3)                   # a third of the way from the group's 40 to his own rate
    assert (many[0] - many[1]) == pytest.approx(40 * 10 / 12)
    assert few.mean() == pytest.approx(40.0) and many.mean() == pytest.approx(40.0)     # shrinkage never moves the group mean


def test_ratios_measure_what_history_did_to_the_projection() -> None:
    current, games = np.array([1000.0, 1000.0, 1000.0]), np.array([4.0, 4.0, 4.0])
    r = project.ratios(current, games, np.zeros(3, dtype=int), np.array([8.0, 8.0, 0.0]), np.array([3000.0, 1000.0, 1500.0]))
    assert len(r) == 2                                                    # a player with no games left tells nothing
    assert r[0] == pytest.approx(2000 / (8 * 250)) and r[1] == 0.0        # 1.0 of projection; zero for the hurt player
    huge = project.ratios(np.array([100.0]), np.array([4.0]), np.zeros(1, dtype=int), np.array([8.0]), np.array([1e6]))
    assert huge[0] == project.RATIO_CAP


def test_a_projection_is_what_he_has_plus_a_rate_times_the_games_left() -> None:
    out = project.project(np.array([1000.0]), np.array([4.0]), np.zeros(1, dtype=int), np.array([[8.0], [9.0]]), np.array([[1.0], [0.0]]))
    assert out[:, 0].tolist() == [1000 + 8 * 250, 1000]                   # a ratio of zero is a season that ended


# ---- the final-vote model

def synthetic_seasons(n: int = 14, seed: int = 0) -> list[final_model.Season]:
    rng = np.random.default_rng(seed)
    out = []
    for year in range(n):
        G = 30
        score = rng.uniform(800, 4500, G)
        group = rng.integers(0, 3, G)
        rank = rng.permutation(np.arange(1, G + 1)).astype(float)
        X = features.matrix(score, group, rank, 0.0, np.zeros(G))
        utility = 1.5 * X[:, features.NAMES.index("prod_z")] - 1.2 * X[:, features.NAMES.index("team_rank")] + rng.gumbel(size=G)
        out.append(final_model.Season(year, pd.DataFrame({"team_rank": rank}), X, int(np.argmax(utility))))
    return out


def test_the_model_recovers_what_drives_the_winner_and_beats_chance() -> None:
    seasons = synthetic_seasons(60)
    cols = [features.NAMES.index(n) for n in ("prod_z", "team_rank")]
    beta = final_model.fit(seasons, cols, penalty=0.3)
    assert beta[0] > 0.8 and beta[1] < -0.6
    test = synthetic_seasons(30, seed=99)
    hits = np.mean([np.argmax(choice.probabilities(beta, s.X[:, cols])) == s.winner for s in test])
    assert hits > 0.2 and hits > 5 / 30                # thirty candidates a season: chance is one in thirty


def test_the_committed_model_points_the_right_way() -> None:
    m = final_model.load_model()
    coef = dict(zip(m["features"], m["coefficients"]))
    assert set(m["features"]) <= set(features.NAMES)
    assert coef["prod_z"] > 0 and coef["group_rank"] < 0 and coef["team_rank"] < 0
    assert m["seasons"] == 14 and m["fitted"] == "2012-2025"


def test_every_winner_is_a_candidate_in_every_season() -> None:
    from mri.heisman import teams
    from mri.ingest import cfbd

    voting, table = data.load_voting(), data.load_players()
    for year in range(2013, 2026):
        s = final_model.season(year, table, teams.team_state(cfbd.games(year)), voting)
        assert s.winner is not None, year
        assert len(s.pool) == 65 and s.X.shape == (65, len(features.NAMES))


# ---- the engine's observer

def test_watching_the_engine_changes_nothing_and_sees_every_run() -> None:
    teams, schedule = league()
    plain = season.simulate(teams, schedule, home_field=3.0, sims=1200, seed=5)
    seen = []
    watched = season.simulate(teams, schedule, home_field=3.0, sims=1200, seed=5, observe=seen.append)
    assert watched["teams"] == plain["teams"]
    rank = np.concatenate([c["rank"] for c in seen])
    assert rank.shape == (1200, len(teams))
    assert (np.sort(rank, axis=1) == np.arange(1, len(teams) + 1)).all()          # each run ranks every team once, 1 = best
    assert (np.concatenate([c["losses"] for c in seen]).min() >= 0)


# ---- the forecast

@pytest.fixture(scope="module")
def runs():
    teams, schedule = league()
    return teams, schedule, forecast.simulate_teams(teams, schedule, home_field=3.0, sims=2000)


def candidates(teams):
    # Conference members all have the same number of games left; Notre Dame has one, which would
    # hand its players a different remaining schedule and muddy what this is comparing.
    ranked = sorted((t for t in teams if t["team"] != "Notre Dame"), key=lambda t: -t["power"])
    best, weak = ranked[0]["team"], ranked[-1]["team"]
    return pd.DataFrame([
        {"player": "Star", "team": best, "group": "QB", "group_code": 0, "off_score": 2600.0, "team_games": 5.0, "prev_finalist": 0.0},
        {"player": "Good", "team": weak, "group": "QB", "group_code": 0, "off_score": 2600.0, "team_games": 5.0, "prev_finalist": 0.0},
        {"player": "Back", "team": best, "group": "RB", "group_code": 1, "off_score": 900.0, "team_games": 5.0, "prev_finalist": 0.0},
        {"player": "Wide", "team": weak, "group": "REC", "group_code": 2, "off_score": 600.0, "team_games": 5.0, "prev_finalist": 0.0},
    ])


MODEL = {"features": ["prod_rank", "group_rank", "prod_z", "team_rank"], "coefficients": [0.0, -1.7, 0.8, -1.4]}


def test_odds_are_probabilities_and_the_better_situation_wins_more(runs) -> None:
    teams, schedule, r = runs
    left = forecast.remaining_games(schedule, r["names"])
    ratios = np.array([0.5, 0.8, 1.0, 1.2])
    out = forecast.odds(candidates(teams), r, left, ratios, MODEL, seed=1)
    assert out["win"].sum() == pytest.approx(1.0, abs=1e-9)
    assert (out["finalist"] >= out["win"] - 1e-9).all() and out["finalist"].max() <= 1.0
    top = out.set_index("player")["win"]
    assert top["Star"] > top["Good"]                        # the same season on a far better team
    assert top["Star"] > top["Back"] and top["Star"] > top["Wide"]
    again = forecast.odds(candidates(teams), r, left, ratios, MODEL, seed=1)
    assert out["win"].tolist() == again["win"].tolist()


def test_a_team_with_no_games_left_and_no_title_game_projects_to_what_it_has() -> None:
    teams, schedule = league()
    schedule = schedule.assign(played=True, pts1=30.0, pts2=20.0)
    r = forecast.simulate_teams(teams, schedule, home_field=3.0, sims=200)
    left = forecast.remaining_games(schedule, r["names"])
    assert set(left.values()) == {0}
    out = forecast.odds(candidates(teams), r, left, np.array([1.0]), MODEL, seed=1)
    assert out["projected"].max() <= 2600.0 + 5 * (2600 / 5)      # at most one more game, the title game


# ---- the live path

def test_ratios_come_from_the_latest_snapshot_not_after_the_week_asked_about() -> None:
    by_week = json.loads(live.RATIOS.read_text())["byWeek"]
    assert list(live.load_ratios(4)) == by_week["3"] and list(live.load_ratios(1)) == by_week["3"]
    assert list(live.load_ratios(8)) == by_week["7"] and list(live.load_ratios(15)) == by_week["13"]


def test_the_sites_own_ratings_become_the_standings_the_candidates_are_judged_by() -> None:
    payload = {"teams": [{"team": "A", "power": 30.0, "resume": 2.0, "rank": 1, "wins": 4, "losses": 0},
                         {"team": "B", "power": 10.0, "resume": -1.0, "rank": 2, "wins": 2, "losses": 2}]}
    state = live.state_from(payload)
    assert state.loc["A", "rank"] == 1 and state.loc["B", "rank"] == 2
    assert state["games"].tolist() == [4, 4] and state["power_rank"].tolist() == [1, 2]


def test_weekly_totals_ask_the_feed_for_the_week_and_only_offence(monkeypatch) -> None:
    asked = []

    def rows(year, category, refresh=None, end_week=None):
        asked.append((category, end_week))
        stat = lambda kind, v: {"playerId": "1", "player": "P", "team": "Indiana", "position": "QB", "category": category,
                                "statType": kind, "stat": str(v)}
        return [stat("ATT", 200), stat("YDS", 2000)] if category == "passing" else []

    monkeypatch.setattr(players, "category_rows", rows)
    monkeypatch.setattr(players.cfbd, "fbs_teams", lambda year: pd.DataFrame({"team": ["Indiana"]}))
    table = players.weekly_table(2024, 7)
    assert asked == [("passing", 7), ("rushing", 7), ("receiving", 7)]
    assert table["pass_yds"].iloc[0] == 2000 and table["week"].iloc[0] == 7 and "def_tot" not in table.columns


# ---- what the documentation claims

def test_the_committed_final_vote_backtest_beats_the_simple_alternatives() -> None:
    b = json.loads((ROOT / "site" / "data" / "heisman_backtest.json").read_text())
    chosen = b["candidateSets"][b["chosen"]]
    assert chosen["top3"] >= 0.8 and chosen["top1"] >= 0.6
    assert chosen["top1"] > max(b["baselines"][k] for k in ("mostProductive", "mostProductiveOnATopFiveTeam", "bestOnTheBestTeam"))
    assert chosen["logLoss"] < b["uniformLogLoss"] - 2


def test_the_committed_forecast_backtest_beats_the_standings_and_is_calibrated() -> None:
    b = json.loads((ROOT / "site" / "data" / "heisman_forecast_backtest.json").read_text())
    better = sum(v["forecast"]["logLoss"] < v["current"]["logLoss"] for v in b["byWeek"].values())
    assert better == len(b["byWeek"])
    assert b["byWeek"]["13"]["forecast"]["top3"] >= 0.75 and b["byWeek"]["3"]["forecast"]["top3"] >= 0.5
    for bucket in b["calibration"]:
        if bucket["candidates"] >= 200:
            assert abs(bucket["predicted"] - bucket["observed"]) < 0.03, bucket
