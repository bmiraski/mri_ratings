"""The composite score: fit against a known ranking, and the field it selects
and seeds from that score."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mri.bracket import atlarge


def synthetic_pool(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    power = rng.uniform(-20, 20, n)
    quad1_wins = np.clip((power + 20) / 4, 0, None).round()
    quad1_losses = rng.integers(0, 3, n)
    return pd.DataFrame({
        "team": [f"Team{i}" for i in range(n)], "powerRank": pd.Series(power).rank(ascending=False, method="min"),
        "resume": power / 3, "quad1_wins": quad1_wins, "quad1_losses": quad1_losses,
        "wins": 20 + rng.integers(0, 12, n), "losses": rng.integers(2, 14, n),
        "bad_losses": rng.integers(0, 3, n), "sos": rng.uniform(1, 300, n), "road_neutral_wins": rng.integers(0, 10, n),
    })


def test_design_matrix_has_an_intercept_column_of_ones() -> None:
    X = atlarge.design(synthetic_pool(10))
    assert X.shape == (10, len(atlarge.FEATURES) + 1)
    assert (X[:, -1] == 1.0).all()


def test_fit_recovers_a_known_relationship_between_power_and_seed() -> None:
    pool = synthetic_pool(80, seed=1)
    pool["seed"] = np.clip(np.ceil(pool["powerRank"] / 5), 1, 16)          # seed is purely a function of power rank here
    beta = atlarge.fit([pool], ridge=0.5)
    scores = atlarge.score(pool, beta)
    assert scores.corr(pool["seed"]) > 0.9                                # the fitted score tracks the true seed closely
    # A team that is strictly better everywhere scores strictly better.
    better = pool.iloc[[0]].copy()
    worse = better.copy()
    worse["powerRank"] += 50
    worse["resume"] -= 10
    worse["quad1_wins"] = 0
    assert atlarge.score(worse, beta).iloc[0] > atlarge.score(better, beta).iloc[0]


def test_select_and_seed_keeps_every_auto_bid_and_fills_the_rest_by_score() -> None:
    pool = synthetic_pool(50, seed=2)
    beta = np.array([1.0, -1.0, -1.0, 0.5, 0.3, -0.05, 0.0])              # arbitrary but matches FEATURES + intercept order
    weakest_team = pool.loc[pool["powerRank"].idxmax(), "team"]           # the single worst team by power
    auto_bids = {"Weak Conference": weakest_team}
    field = atlarge.select_and_seed(pool, beta, auto_bids=auto_bids, field_size=20)
    assert weakest_team in set(field["team"])                            # in no matter how it scores
    assert field.loc[field["team"] == weakest_team, "bidType"].iloc[0] == "auto"
    assert len(field) == 20 and (field["bidType"] == "at-large").sum() == 19
    assert field["projectedSeed"].min() >= 1 and field["projectedSeed"].max() <= 16
    assert field["projectedSeed"].is_monotonic_increasing or (field.sort_values("compositeScore")["projectedSeed"].is_monotonic_increasing)


def test_a_bigger_field_is_just_a_longer_list_not_a_different_model() -> None:
    # The extra at-large slots this year are exactly this: same score, same coefficients, more teams taken.
    pool = synthetic_pool(90, seed=3)
    beta = atlarge.fit([pool.assign(seed=np.clip(np.ceil(pool["powerRank"] / 5), 1, 16))], ridge=1.0)
    small = atlarge.select_and_seed(pool, beta, auto_bids={}, field_size=68)
    big = atlarge.select_and_seed(pool, beta, auto_bids={}, field_size=76)
    assert set(small["team"]) <= set(big["team"])                        # everyone who got in before still gets in
    assert len(big) - len(small) == 8
