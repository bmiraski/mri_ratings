"""The GameDay forecast must be a probability, and the history behind it must be right."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mri.export import gamedaydata, site
from mri.gameday import features, forecast, history, model

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "data" / "gameday_locations.json").read_text())
USED = ("best_rank", "worst_rank", "both_top10", "both_top25", "both_unbeaten", "losses")
MODEL = {"features": list(USED), "coefficients": [0.0, -1.0, 0.1, 0.9, 0.0, -1.0], "otherRate": 0.05,
         "armyNavy": {"estimate": 0.1, "lastFour": 0}}


# ---- features

def test_committee_rank_puts_the_better_team_first() -> None:
    power, resume = np.array([3.0, 1.0, 2.0]), np.array([0.0, 2.0, 1.0])
    assert features.committee_score_rank(power, resume).tolist() == [3, 1, 2]
    assert features.committee_score_rank(np.stack([power, power]), np.stack([resume, resume])).shape == (2, 3)


def test_feature_matrix_broadcasts_and_counts_what_it_says() -> None:
    x = features.matrix(rank_home=3, rank_away=[5, 40], losses_home=0, losses_away=[0, 2])
    assert x.shape == (2, len(features.NAMES))
    both10 = features.NAMES.index("both_top10")
    unbeaten = features.NAMES.index("both_unbeaten")
    losses = features.NAMES.index("losses")
    assert x[0, both10] == 1 and x[1, both10] == 0
    assert x[0, unbeaten] == 1 and x[1, unbeaten] == 0
    assert x[1, losses] == 2


def test_unranked_teams_are_capped_not_infinite() -> None:
    x = features.matrix(rank_home=500, rank_away=900, losses_home=1, losses_away=1)
    assert np.isfinite(x).all()
    assert x[features.NAMES.index("worst_rank")] == pytest.approx(np.log(features.RANK_CAP))


def test_a_neutral_site_counts_either_elite_conference() -> None:
    assert features.elite_conference(["ACC"], ["SEC"], [True])[0]
    assert not features.elite_conference(["ACC"], ["SEC"], [False])[0]


# ---- the choice model

def synthetic_weeks(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    sets = []
    for _ in range(n):
        rank_h, rank_a = rng.integers(1, 60, 30), rng.integers(1, 60, 30)
        lh, la = rng.integers(0, 4, 30), rng.integers(0, 4, 30)
        X = features.matrix(rank_home=rank_h, rank_away=rank_a, losses_home=lh, losses_away=la)
        cols = [features.NAMES.index(f) for f in USED]
        truth = np.array([0.0, -1.2, 0.0, 1.0, 0.0, -0.8])
        u = X[:, cols] @ truth + rng.gumbel(size=30)
        sets.append((X[:, cols], int(np.argmax(u))))
    return sets


def test_the_model_recovers_which_way_the_features_point() -> None:
    beta = model.fit(synthetic_weeks(150), penalty=1.0)
    assert beta[1] < -0.5          # the worse team's rank counts against a game
    assert beta[5] < -0.3          # so do losses
    assert beta[3] > 0.2           # and being in the top 25 helps


def test_probabilities_are_a_distribution_and_beat_a_coin_toss() -> None:
    sets = synthetic_weeks(120)
    beta = model.fit(sets, penalty=1.0)
    for X, _ in sets[:5]:
        p = model.probabilities(beta, X)
        assert p.sum() == pytest.approx(1.0) and (p >= 0).all()
    s = model.score(beta, sets)
    assert s["logLoss"] < s["baseLogLoss"] - 0.5 and s["top1"] > 0.15


# ---- the history file

def test_the_history_file_is_well_formed() -> None:
    assert "source" in DATA and DATA["seasons"]
    for year, stops in DATA["seasons"].items():
        dates = [s["date"] for s in stops]
        assert dates == sorted(dates), f"{year} is out of order"
        assert len(set(dates)) == len(dates)
        for s in stops:
            assert s["date"].startswith(year) or int(s["date"][:4]) == int(year), s
            assert s["kind"] in ("reg", "cg", "svc", "fcs")
            assert len(s["teams"]) == 2
            assert s["host"] is None or s["host"] in s["teams"], s


def test_every_fbs_stop_is_a_real_game_on_the_right_day_at_the_right_place() -> None:
    """The transcription is checked against the schedule, so a typo cannot hide."""
    from mri.ingest import cfbd

    problems = []
    for year, stops in DATA["seasons"].items():
        try:
            games = cfbd.games(int(year), completed_only=False)
        except Exception:  # noqa: BLE001 - no cache for that year on this machine
            continue
        starts = pd.to_datetime(games["start_date"], utc=True)
        for s in stops:
            if s["kind"] == "fcs":
                continue
            a, b = s["teams"]
            hit = games[((games["team1"] == a) & (games["team2"] == b)) | ((games["team1"] == b) & (games["team2"] == a))]
            hit = hit[(starts[hit.index] - pd.Timestamp(s["date"], tz="UTC")).abs() <= pd.Timedelta(days=3)]
            if len(hit) != 1:
                problems.append(f"{s['date']} {a} / {b}: {len(hit)} matching games")
            elif s["host"] and hit.iloc[0]["team2"] != s["host"]:
                problems.append(f"{s['date']} {a} / {b}: host {s['host']} but the schedule says {hit.iloc[0]['team2']}")
    assert not problems, problems


def test_the_announced_weeks_are_consecutive_and_have_hosts_in_the_game() -> None:
    weeks = [a["week"] for a in DATA["announced2026"]]
    assert weeks == list(range(1, len(weeks) + 1))
    for a in DATA["announced2026"]:
        assert a["host"] in a["teams"]


def test_recent_hosting_and_brand_are_counted_from_the_right_seasons() -> None:
    data = {"seasons": {"2013": [{"teams": ["A", "B"], "host": "A"}], "2014": [{"teams": ["A", "C"], "host": "C"}],
                        "2015": [{"teams": ["A", "D"], "host": "A"}]},
            "hostsOnly": {"2012": ["A", "Z"]}}
    assert history.hosts_before(data, 2016, span=3) == {"A": 2, "C": 1}
    assert history.hosts_before(data, 2014, span=3) == {"A": 2, "Z": 1}
    rates = history.appearance_rates(data, 2016)
    assert rates["A"] == pytest.approx(3 / 3) and rates["B"] == pytest.approx(1 / 3)


def test_a_stop_is_placed_in_the_week_of_its_game_or_of_its_date() -> None:
    games = pd.DataFrame({
        "week": [5, 5, 6], "team1": ["A", "C", "E"], "team2": ["B", "D", "F"],
        "start_date": ["2026-10-03T16:00:00Z", "2026-10-03T20:00:00Z", "2026-10-10T16:00:00Z"]})
    stops = [{"teams": ["A", "B"], "date": "2026-10-03"}, {"teams": ["X", "Y"], "date": "2026-10-10"}]
    assert history.stop_weeks(stops, games) == [5, 6]


# ---- the forecast

def league():
    rng = np.random.default_rng(3)
    teams, games = [], []
    for conference in ("SEC", "Big Ten", "ACC", "Big 12", "Sun Belt"):
        members = [f"{conference} {i}" for i in range(8)]
        teams += [{"team": m, "power": float(rng.normal(0, 8)), "conference": conference} for m in members]
        week = 4
        for a in range(8):
            for b in range(a + 1, 8):
                games.append((week, members[a], members[b], True))
                week = 4 + (week - 3) % 10
    frame = pd.DataFrame([{"game_id": i, "week": w, "team1": a, "team2": b, "played": False, "pts1": np.nan,
                           "pts2": np.nan, "neutral": False, "conference_game": c}
                          for i, (w, a, b, c) in enumerate(games)])
    return teams, frame


@pytest.fixture(scope="module")
def result():
    teams, schedule = league()
    return teams, schedule, forecast.forecast(teams, schedule, home_field=3.0, model=MODEL,
                                              weeks=[6, 8, 14], sims=1500)


def test_each_week_is_a_distribution_over_the_games_plus_somewhere_else(result) -> None:
    _, schedule, out = result
    for week in (6, 8):
        games = out["weeks"][week]["games"]
        assert sum(e["probability"] for e in games) == pytest.approx(1.0 - MODEL["otherRate"], abs=5e-3)
        assert len(games) == int(((schedule["week"] == week) & ~schedule["played"]).sum())


def test_the_championship_week_chooses_among_conference_title_games(result) -> None:
    _, _, out = result
    block = out["weeks"][14]["games"]
    assert {e["conference"] for e in block} == {"SEC", "Big Ten", "ACC", "Big 12", "Sun Belt"}
    assert sum(e["probability"] for e in block) == pytest.approx(1.0 - MODEL["otherRate"], abs=5e-3)


def test_better_teams_make_bigger_games(result) -> None:
    teams, _, out = result
    power = {t["team"]: t["power"] for t in teams}
    games = out["weeks"][6]["games"]
    combined = {(e["home"], e["away"]): power[e["home"]] + power[e["away"]] for e in games}
    top = max(games, key=lambda e: e["probability"])
    median = np.median(list(combined.values()))
    assert combined[(top["home"], top["away"])] > median


def test_hosting_is_no_likelier_than_appearing_and_all_are_probabilities(result) -> None:
    _, _, out = result
    for name, t in out["teams"].items():
        assert 0.0 <= t["hostsAtLeastOnce"] <= t["appearsAtLeastOnce"] + 1e-9 <= 1.0 + 1e-9, name


def test_the_same_inputs_give_the_same_forecast() -> None:
    teams, schedule = league()
    a = forecast.forecast(teams, schedule, home_field=3.0, model=MODEL, weeks=[6], sims=800)
    b = forecast.forecast(teams, schedule, home_field=3.0, model=MODEL, weeks=[6], sims=800)
    assert a == b


def test_the_committed_model_is_usable() -> None:
    m = forecast.load_model()
    assert m is not None
    assert len(m["coefficients"]) == len(m["features"])
    assert all(f in features.NAMES for f in m["features"])
    assert 0.0 < m["otherRate"] < 0.2 and 0.0 <= m["armyNavy"]["estimate"] <= 1.0


# ---- the page

def test_the_week_dates_come_from_when_games_are_played() -> None:
    games = pd.DataFrame({"week": [1, 1, 2], "start_date": ["2026-09-05T23:00:00Z", "2026-09-06T02:00:00Z",
                                                            "2026-09-12T23:00:00Z"]})
    dates = gamedaydata._week_dates(games)
    assert str(dates[1]) == "2026-09-05" and str(dates[2]) == "2026-09-12"
    assert gamedaydata._label(dates[1]) == "Sep 5"


def test_probabilities_read_without_false_precision() -> None:
    assert site._pct(0.0) == "&lt;0.1%" and site._pct(0.474) == "47%" and site._pct(0.032) == "3.2%"


def test_a_clear_week_lists_six_and_a_murky_one_lists_until_it_has_most_of_the_chance() -> None:
    clear = [{"probability": p} for p in (0.5, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.01, 0.01)]
    assert len(gamedaydata.shown(clear)) == gamedaydata.MIN_SHOWN

    murky = [{"probability": 0.03} for _ in range(30)]                # thin and flat
    listed = gamedaydata.shown(murky)
    assert len(listed) == gamedaydata.MAX_SHOWN                         # capped, not the 29 it would take

    middling = [{"probability": p} for p in (0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02)]
    listed = gamedaydata.shown(middling)
    assert gamedaydata.MIN_SHOWN < len(listed) < len(middling)
    assert sum(e["probability"] for e in listed) >= gamedaydata.TARGET - 0.005
    assert [e["probability"] for e in listed] == sorted((e["probability"] for e in listed), reverse=True)

