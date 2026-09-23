"""Per-stadium home field: the geometry, the clocks, and the shrinkage."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from mri.ratings import hfa

UTC = dt.timezone.utc


def test_great_circle_distance_matches_a_known_pair() -> None:
    # JFK to LAX is the textbook 2,475 statute miles on the ellipsoid; a sphere
    # lands within half a percent of it, which is far inside what travel needs.
    # A degree of longitude on the equator is 69.1.
    assert hfa.great_circle_miles(40.6413, -73.7781, 33.9416, -118.4085) == pytest.approx(2475, rel=0.005)
    assert hfa.great_circle_miles(0, 0, 0, 1) == pytest.approx(69.09, abs=0.05)
    assert hfa.great_circle_miles(41.66, -91.55, 41.66, -91.55) == 0.0


def test_time_zones_follow_daylight_saving() -> None:
    september = dt.datetime(2019, 9, 7, 16, 0, tzinfo=UTC)   # noon Eastern, daylight time
    november = dt.datetime(2019, 11, 9, 17, 0, tzinfo=UTC)   # noon Eastern, standard time
    assert hfa.utc_offset_hours("America/New_York", september) == -4
    assert hfa.utc_offset_hours("America/New_York", november) == -5
    # Arizona never moves, so an Eastern trip is three zones in September and two in November.
    assert hfa.utc_offset_hours("America/Phoenix", september) == -7
    assert hfa.utc_offset_hours("America/Phoenix", november) == -7
    # Noon Eastern is 9am on a Pacific body clock in both months.
    assert hfa.local_hour("America/Los_Angeles", september) == 9
    assert hfa.local_hour("America/Los_Angeles", november) == 9
    assert hfa.local_hour("America/Phoenix", november) == 10


def _groups(effects, games, noise=16.0, seed=0):
    rng = np.random.default_rng(seed)
    values, groups = [], []
    for g, (effect, n) in enumerate(zip(effects, games)):
        values.extend(effect + rng.normal(0, noise, n))
        groups.extend([g] * n)
    return pd.Series(values), pd.Series(groups)


def test_shrinkage_pulls_small_samples_hardest() -> None:
    rng = np.random.default_rng(1)
    effects = list(rng.normal(0, 3.0, 200)) + [20.0, 20.0]
    games = [80] * 200 + [1, 80]
    values, groups = _groups(effects, games)
    table = hfa.shrink(values, groups)
    assert table.attrs["tau2"] > 0
    one, eighty = table.loc[200], table.loc[201]
    assert abs(one["effect"]) < 0.1 * abs(one["raw"])
    assert eighty["weight"] > 5 * one["weight"]
    assert abs(eighty["effect"]) > 0.4 * abs(eighty["raw"])


def test_no_detectable_spread_means_every_effect_is_zero() -> None:
    # Every group's mean is exactly zero: nothing to find, so tau^2 < 0.
    values = pd.Series([10.0, -10.0] * 300)
    groups = pd.Series(np.repeat(np.arange(30), 20))
    table = hfa.shrink(values, groups)
    assert table.attrs["tau2"] <= 0
    assert (table["effect"] == 0.0).all()


def test_neutral_sites_get_travel_but_no_stadium_effect() -> None:
    frame = pd.DataFrame({
        "hosted": [1.0, 0.0],
        "venue_id": [10, 10],
        "home_venue_h": [10, 20],
        "home_venue_a": [30, 30],
    })
    fits = {
        "venue": pd.DataFrame({"effect": [2.0]}, index=[10]),
        "road": pd.DataFrame({"effect": [0.0, 0.0]}, index=[20, 30]),
    }
    adj = hfa.stadium_adjustment(frame, fits, road=False)
    assert adj[0] == 2.0 and adj[1] == 0.0

    # Florida-Georgia in Jacksonville: neutral, and Georgia travels further.
    games = hfa.venue_games([2017])
    cocktail = games[(games["home"] == "Florida") & (games["away"] == "Georgia")].iloc[0]
    assert cocktail["hosted"] == 0
    assert cocktail["travel_away"] > cocktail["travel_home"] > 0
    assert cocktail["travel"] > 0
