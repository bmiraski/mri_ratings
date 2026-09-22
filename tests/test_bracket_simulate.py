"""The Monte Carlo bracket: probabilities that sum to one, reward the better seed,
and reward it more the deeper its bye - checked against the actual shape of a
real bracket (the SEC's own 2025 tiers), not an invented one."""

from __future__ import annotations

import numpy as np
import pytest

from mri.bracket import simulate
from mri.bracket.template import Template

SEC_SHAPE = Template(tiers=(8, 4, 4), seasons_seen=1, stable=True)         # the real, checked 2025 shape
FLAT_16 = Template(tiers=(16,), seasons_seen=1, stable=True)


def seeds16() -> list[str]:
    return [f"S{i}" for i in range(1, 17)]


def power16(spread: float = 1.0) -> dict:
    return {f"S{i}": (16 - i) * spread for i in range(1, 17)}


def test_win_probability_is_a_proper_probability_and_symmetric() -> None:
    assert simulate.win_probability(np.array([0.0]))[0] == pytest.approx(0.5)
    assert simulate.win_probability(np.array([10.0]))[0] > 0.5
    a = simulate.win_probability(np.array([7.0]))[0]
    b = simulate.win_probability(np.array([-7.0]))[0]
    assert a + b == pytest.approx(1.0)
    assert simulate.win_probability(np.array([7.0]), home_edge=3.0)[0] > a       # home court only helps the host


def test_every_teams_chances_sum_to_one_and_are_deterministic_by_seed() -> None:
    res = simulate.simulate(SEC_SHAPE, seeds16(), power16(), sims=4000, seed=1)
    assert len(res) == 16
    assert sum(res.values()) == pytest.approx(1.0, abs=1e-9)
    again = simulate.simulate(SEC_SHAPE, seeds16(), power16(), sims=4000, seed=1)
    assert res == again
    different = simulate.simulate(SEC_SHAPE, seeds16(), power16(), sims=4000, seed=2)
    assert res != different


def test_a_rejects_the_wrong_number_of_seeds() -> None:
    with pytest.raises(ValueError):
        simulate.simulate(SEC_SHAPE, seeds16()[:15], power16(), sims=100)


def test_better_seeds_win_more_often_monotonically() -> None:
    res = simulate.simulate(SEC_SHAPE, seeds16(), power16(2.0), sims=9000, seed=3)
    ordered = [res[f"S{i}"] for i in range(1, 17)]
    assert all(ordered[i] >= ordered[i + 1] - 0.01 for i in range(len(ordered) - 1))
    assert res["S1"] == max(res.values())


def test_byes_are_worth_something_a_flat_field_would_not_give(seed=7) -> None:
    # With every seed exactly as strong as the last (no separation at all), the only thing that can
    # possibly favour S1 over S16 is the bracket shape itself - so any edge S1 shows here is purely
    # what its byes are worth, isolated from rating strength.
    flat_power = {f"S{i}": 0.0 for i in range(1, 17)}
    bye_shape = simulate.simulate(SEC_SHAPE, seeds16(), flat_power, sims=12000, seed=seed)
    no_bye_shape = simulate.simulate(FLAT_16, seeds16(), flat_power, sims=12000, seed=seed)
    assert bye_shape["S1"] > no_bye_shape["S1"] + 0.02


def test_campus_hosting_helps_whoever_the_bracket_makes_the_host() -> None:
    power = power16(1.0)
    no_edge = simulate.simulate(SEC_SHAPE, seeds16(), power, home_edge=0.0, sims=8000, seed=5)
    with_edge = simulate.simulate(SEC_SHAPE, seeds16(), power, home_edge=4.0, sims=8000, seed=5)
    assert with_edge["S1"] > no_edge["S1"]           # the better seed is the one who'd be hosting


def test_an_odd_field_lets_the_best_remaining_seed_sit_out_a_round() -> None:
    t = Template(tiers=(3,), seasons_seen=1, stable=True)
    res = simulate.simulate(t, ["A", "B", "C"], {"A": 5.0, "B": 0.0, "C": -5.0}, sims=8000, seed=1)
    assert sum(res.values()) == pytest.approx(1.0, abs=1e-9)
    assert res["A"] > res["B"] > res["C"]
