"""In-season hot-seat odds: run_features reconstructs the same numbers hotseat.dataset
would for a season that's already finished (checked directly against it), a coach
missing core history is excluded the same way, odds() averages correctly across
runs, and the weekly history file never rewrites a finished week."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mri.coaches import hotseat, inseason

EMPTY_DEPARTURES = pd.DataFrame(columns=["coach_id", "school", "season", "label"])


def _row(coach_id: str, school: str, season_: int, **overrides) -> dict:
    base = {
        "coach_id": coach_id, "coach_name": coach_id, "school": school, "season": season_,
        "conference": "SEC", "games": 12, "wins": 6, "losses": 6, "interim": False,
        "conf_wins": 4, "conf_losses": 4,
        "power_end": 0.0, "prior": 0.0, "added": 0.0, "inherited": 0.0,
        "vs_inherited": 0.0, "program_par": 0.0, "vs_par": 0.0, "resume": 0.0, "vs_talent": np.nan,
    }
    base.update(overrides)
    return base


def _schedule(school: str, opponent: str, n_conf: int, n_nonconf: int) -> pd.DataFrame:
    n = n_conf + n_nonconf
    return pd.DataFrame({
        "game_id": range(1, n + 1), "team1": [opponent] * n, "team2": [school] * n,
        "conference_game": [True] * n_conf + [False] * n_nonconf,
    })


def _single_run(school: str, *, est: float, wins: float, losses: float, conf_wins: float) -> dict:
    return {
        "index": {school: 0},
        "est": np.array([[est]]), "wins": np.array([[wins]]), "losses": np.array([[losses]]),
        "confWins": np.array([[conf_wins]]),
    }


def test_run_features_matches_hotseat_dataset_for_a_finished_season() -> None:
    """Feed run_features a single 'run' that reproduces exactly what really happened,
    and it should reconstruct the same feature values hotseat.dataset computes for
    that same, already-finished coach-season."""
    coach_season = pd.DataFrame([
        _row("a", "School A", 2010, vs_par=3.0),
        # vs_par/added/vs_inherited are pre-computed by phase 2's metrics pipeline in real
        # data (power_end - program_par, etc.) - set consistently here since hotseat.dataset
        # reads them as stored, not recomputed.
        _row("a", "School A", 2011, prior=2.0, program_par=1.0, inherited=-1.0,
             power_end=5.0, vs_par=4.0, added=3.0, vs_inherited=6.0,
             wins=8, losses=4, conf_wins=5, conf_losses=3, games=12),
        _row("filler", "School Z", 2012),  # pushes the dataset's "most recent season" boundary
    ])
    expected = hotseat.dataset(coach_season, EMPTY_DEPARTURES).set_index(["coach_id", "season"]).loc[("a", 2011)]

    runs = _single_run("School A", est=5.0, wins=8.0, losses=4.0, conf_wins=5.0)
    schedule = _schedule("School A", "Opponent", n_conf=8, n_nonconf=4)
    features = inseason.run_features(coach_season, 2011, runs, schedule)

    got = features["a"].iloc[0]
    for column in hotseat.FULL_COLUMNS:
        assert got[column] == pytest.approx(expected[column]), column


def test_run_features_resets_tenure_year_after_a_return_stint() -> None:
    """A coach who left a school and came back years later (Petrino at Louisville,
    Rich Rodriguez at West Virginia) is evaluated on tenure at the CURRENT stint,
    not the number of years since they first showed up at that school."""
    coach_season = pd.DataFrame([
        _row("a", "School A", 2003, program_par=np.nan),  # first stint - no signal to fix year1 either way
        _row("a", "School A", 2004, program_par=np.nan),
        _row("a", "School A", 2014,  # second stint, year 1 again, after a decade away
             prior=2.0, program_par=1.0, inherited=-1.0, power_end=5.0, vs_par=4.0, added=3.0, vs_inherited=6.0,
             wins=8, losses=4, conf_wins=5, conf_losses=3, games=12),
        _row("filler", "School Z", 2015),
    ])
    expected = hotseat.dataset(coach_season, EMPTY_DEPARTURES).set_index(["coach_id", "season"]).loc[("a", 2014)]
    assert expected["year1"] == 1.0  # confirms hotseat.dataset itself treats this as a fresh stint

    runs = _single_run("School A", est=5.0, wins=8.0, losses=4.0, conf_wins=5.0)
    schedule = _schedule("School A", "Opponent", n_conf=8, n_nonconf=4)
    features = inseason.run_features(coach_season, 2014, runs, schedule)

    got = features["a"].iloc[0]
    assert got["year1"] == 1.0
    assert got["year2"] == 0.0
    for column in hotseat.FULL_COLUMNS:
        assert got[column] == pytest.approx(expected[column]), column


def test_missing_program_par_or_inherited_excludes_the_coach() -> None:
    coach_season = pd.DataFrame([
        _row("a", "School A", 2011, program_par=np.nan),
        _row("b", "School B", 2011, inherited=np.nan),
        _row("c", "School C", 2011),
    ])
    runs = {
        "index": {"School A": 0, "School B": 1, "School C": 2},
        "est": np.zeros((1, 3)), "wins": np.zeros((1, 3)), "losses": np.zeros((1, 3)), "confWins": np.zeros((1, 3)),
    }
    schedule = pd.concat([_schedule(s, "Opponent", 4, 4) for s in ("School A", "School B", "School C")])
    features = inseason.run_features(coach_season, 2011, runs, schedule)
    assert set(features) == {"c"}


def test_odds_averages_across_runs_with_known_coefficients(tmp_path, monkeypatch) -> None:
    columns = ["vs_par"]
    model = {"features": columns, "coefficientsFull": {"vs_par": 1.0, "intercept": 0.0}}
    model_path = tmp_path / "hotseat_model.json"
    model_path.write_text(json.dumps(model))

    coach_season = pd.DataFrame([_row("a", "School A", 2011)])
    # two runs: vs_par = power_end - program_par = 2.0 and -2.0 -> sigmoid(2) and sigmoid(-2), averaged
    runs = {
        "index": {"School A": 0},
        "est": np.array([[2.0], [-2.0]]), "wins": np.array([[6.0], [6.0]]), "losses": np.array([[6.0], [6.0]]),
        "confWins": np.array([[3.0], [3.0]]),
    }
    schedule = _schedule("School A", "Opponent", 6, 6)

    monkeypatch.setattr(inseason, "simulate_teams", lambda *a, **k: runs)
    result = inseason.odds(coach_season, 2011, teams=[], schedule=schedule, home_field=0.0, model_path=model_path)

    expected = np.mean([1 / (1 + np.exp(-2.0)), 1 / (1 + np.exp(2.0))])
    assert result["a"] == pytest.approx(expected)


def test_weekly_history_never_rewrites_a_finished_week(tmp_path, monkeypatch) -> None:
    history_path = tmp_path / "hotseat_history.json"
    coach_season = pd.DataFrame([_row("a", "School A", 2011)])
    schedule = _schedule("School A", "Opponent", 6, 6)

    calls = {"n": 0}

    def fake_odds(*args, **kwargs):
        calls["n"] += 1
        return pd.Series({"a": 0.1 * calls["n"]})

    monkeypatch.setattr(inseason, "odds", fake_odds)

    h1 = inseason.weekly_odds(coach_season, 2011, 3, [], schedule, 0.0, history_path)
    assert h1["weeks"]["3"] == {"a": 0.1}

    h2 = inseason.weekly_odds(coach_season, 2011, 4, [], schedule, 0.0, history_path)
    assert h2["weeks"]["3"] == {"a": 0.1}   # week 3 untouched
    assert h2["weeks"]["4"] == {"a": 0.2}   # week 4 is the new current week

    h3 = inseason.weekly_odds(coach_season, 2011, 4, [], schedule, 0.0, history_path)
    assert h3["weeks"]["4"] == {"a": 0.3}   # the current week is always recomputed
