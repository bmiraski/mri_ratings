"""The §3 talent gate: does it actually detect a real talent/power_end relationship,
reject a fake one, and hold together for the edge cases (missing prior-season
talent, non-FBS teams) it's supposed to handle gracefully?"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mri.coaches import talent

SEASONS = range(2015, 2021)  # six seasons: enough for a real leave-one-out split
# Real, long-tenured FBS names, not synthetic ones - fbs_for filters teams through
# the real registry (was_fbs/is_fbs), so a made-up name would just be dropped.
TEAMS = ["Alabama", "Georgia", "Ohio State", "Texas", "Oklahoma", "Florida",
         "LSU", "Michigan", "Penn State", "Wisconsin", "Clemson", "Notre Dame"]


def _ratings_history(power_by_team_season: dict[tuple[str, int], float]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"team": t, "season": s, "power": p} for (t, s), p in power_by_team_season.items()]
    )


def _talent_fn(strength: dict[str, float]):
    """A synthetic talent_fn: every team's composite is fixed across seasons, scaled by ``strength``."""
    def fn(season_: int) -> pd.Series:
        return pd.Series(strength, dtype=float)
    return fn


def test_dataset_falls_back_to_zero_for_missing_prior_season_talent() -> None:
    """2015 is the composite's first year - there is no real 2014 talent to look up."""
    power = _ratings_history({(t, 2015): 0.0 for t in TEAMS})
    rng = np.random.default_rng(0)
    strength = {t: float(rng.normal()) for t in TEAMS}

    def talent_fn(season_: int) -> pd.Series:
        return pd.Series(strength, dtype=float) if season_ >= 2015 else pd.Series(dtype=float)

    data = talent.dataset(power, talent_fn=talent_fn)
    assert (data["talent_last"] == 0.0).all()
    assert not (data["talent"] == 0.0).all()  # the current-season z-scores are real, just this-season-only


def test_gate_passes_when_talent_genuinely_predicts_power_end() -> None:
    rng = np.random.default_rng(1)
    strength = {t: float(rng.normal()) for t in TEAMS}
    power_by = {}
    for s in SEASONS:
        for t in TEAMS:
            power_by[(t, s)] = 15.0 * strength[t] + rng.normal(scale=1.0)  # talent dominates, small noise
    ratings_history = _ratings_history(power_by)

    model = talent.fit(ratings_history, talent_fn=_talent_fn(strength))
    assert model["passed"] is True
    assert model["outOfSampleR2"] > 0.5
    assert model["teamSeasons"] == len(TEAMS) * len(SEASONS)


def test_gate_fails_when_power_end_is_unrelated_to_talent() -> None:
    rng = np.random.default_rng(2)
    strength = {t: float(rng.normal()) for t in TEAMS}
    power_by = {(t, s): float(rng.normal(scale=10.0)) for s in SEASONS for t in TEAMS}  # pure noise
    ratings_history = _ratings_history(power_by)

    model = talent.fit(ratings_history, talent_fn=_talent_fn(strength))
    assert model["passed"] is False
    assert model["outOfSampleR2"] < talent.GATE_R2


def test_vs_talent_is_the_residual_from_the_fitted_coefficients() -> None:
    rng = np.random.default_rng(3)
    strength = {t: float(rng.normal()) for t in TEAMS}
    power_by = {(t, s): 15.0 * strength[t] + rng.normal(scale=1.0) for s in SEASONS for t in TEAMS}
    ratings_history = _ratings_history(power_by)

    model = talent.fit(ratings_history, talent_fn=_talent_fn(strength))
    residual = talent.vs_talent(ratings_history, model, talent_fn=_talent_fn(strength))

    data = talent.dataset(ratings_history, talent_fn=_talent_fn(strength)).set_index(["team", "season"])
    c = model["coefficients"]
    for (team, season_), value in residual.items():
        row = data.loc[(team, season_)]
        predicted = c["intercept"] + c["talent"] * row["talent"] + c["talent_last"] * row["talent_last"]
        assert value == pytest.approx(row["power_end"] - predicted)


def test_pre_2015_seasons_are_excluded() -> None:
    ratings_history = _ratings_history({("Alabama", 2010): 5.0, ("Alabama", 2015): 5.0})
    data = talent.dataset(ratings_history, talent_fn=_talent_fn({"Alabama": 1.0}))
    assert list(data["season"]) == [2015]


def test_non_fbs_teams_are_excluded() -> None:
    """fbs_for filters through the real registry - a name it doesn't recognize as
    FBS never enters the dataset, even if it has a row in ratings_history."""
    ratings_history = _ratings_history({("Alabama", 2015): 5.0, ("Not A Real Team", 2015): 5.0})
    data = talent.dataset(ratings_history, talent_fn=_talent_fn({"Alabama": 1.0, "Not A Real Team": 1.0}))
    assert list(data["team"]) == ["Alabama"]
