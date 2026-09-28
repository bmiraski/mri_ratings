"""Surprisal/entropy scoring and the per-week-of-season recalibration."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mri.ratings import chaos


def test_band_for_week_matches_the_spec_boundaries() -> None:
    assert chaos.band_for_week(1) == "early"
    assert chaos.band_for_week(3) == "early"
    assert chaos.band_for_week(4) == "mid"
    assert chaos.band_for_week(7) == "mid"
    assert chaos.band_for_week(8) == "late"
    assert chaos.band_for_week(16) == "late"  # a postseason block's high sequence number


def test_surprisal_and_entropy_match_a_hand_computed_three_game_week(monkeypatch) -> None:
    # Independent, hand-worked reference values (plain math.log, not chaos.py) for three games:
    # a comfortable favourite winning, a big upset, and a smaller upset the other direction.
    def hand_surprisal(p_winner):
        return -math.log(p_winner)

    def hand_entropy(p):
        return -(p * math.log(p) + (1 - p) * math.log(1 - p))

    def hand_variance(p):
        return p * (1 - p) * math.log(p / (1 - p)) ** 2

    games = pd.DataFrame([
        {"game_id": 1, "home": "A", "away": "B", "home_win_prob": 0.8,
         "home_won": True, "home_score": 30, "away_score": 10},
        {"game_id": 2, "home": "C", "away": "D", "home_win_prob": 0.3,
         "home_won": True, "home_score": 21, "away_score": 20},
        {"game_id": 3, "home": "E", "away": "F", "home_win_prob": 0.6,
         "home_won": False, "home_score": 17, "away_score": 24},
    ])

    winner_probs = [0.8, 0.3, 0.4]  # game 3's winner is the away side, at 1 - 0.6
    home_probs = [0.8, 0.3, 0.6]

    expected_surprisal = sum(hand_surprisal(p) for p in winner_probs)
    expected_entropy = sum(hand_entropy(p) for p in home_probs)
    expected_variance = sum(hand_variance(p) for p in home_probs)
    expected_z = (expected_surprisal - expected_entropy) / math.sqrt(expected_variance)

    # Exercise the primitives directly too, not just the aggregate.
    assert chaos.surprisal(0.8) == pytest.approx(hand_surprisal(0.8))
    assert chaos.entropy(0.3) == pytest.approx(hand_entropy(0.3))
    assert chaos.surprisal_variance(0.6) == pytest.approx(hand_variance(0.6))

    # A 3-game week is under the real MIN_GAMES floor (spec test 6 covers that gate separately);
    # lower it here so this fixture can exercise the full week_score aggregation.
    monkeypatch.setattr(chaos, "MIN_GAMES", 3)
    result = chaos.week_score(games)

    assert result["games"] == 3
    assert result["z"] == pytest.approx(expected_z)
    assert result["upsets"] == 2  # games 2 and 3 were both won by the underdog
    assert result["expectedUpsets"] == pytest.approx(0.2 + 0.3 + 0.4)
    # Shocks are ordered by ascending winner probability: game 2 (0.3), then game 3 (0.4).
    assert [s["gameId"] for s in result["shocks"]] == [2, 3, 1]
    assert result["shocks"][0]["winner"] == "C" and result["shocks"][0]["loser"] == "D"
    assert result["shocks"][1]["winner"] == "F" and result["shocks"][1]["loser"] == "E"


def test_week_score_marks_a_small_week_not_enough_games() -> None:
    games = pd.DataFrame([
        {"game_id": i, "home": "A", "away": "B", "home_win_prob": 0.6,
         "home_won": True, "home_score": 20, "away_score": 10}
        for i in range(6)
    ])
    result = chaos.week_score(games)
    assert result["games"] == 6
    assert result["z"] is None
    assert result["note"] == "not_enough_games"


def test_apply_recalibration_is_identity_without_a_calibration_file() -> None:
    assert chaos.apply_recalibration(0.734, week=2, calibration=None) == pytest.approx(0.734)


def test_apply_recalibration_uses_the_matching_band() -> None:
    calibration = {"bands": {"early": {"a": 1.0, "b": 0.0, "n": 100},
                              "mid": {"a": 2.0, "b": 0.5, "n": 100},
                              "late": {"a": 1.0, "b": 0.0, "n": 100}}}
    # Identity band: unchanged.
    assert chaos.apply_recalibration(0.7, week=2, calibration=calibration) == pytest.approx(0.7)
    # "mid" band actually transforms the probability.
    recalibrated = chaos.apply_recalibration(0.7, week=5, calibration=calibration)
    assert recalibrated != pytest.approx(0.7)


def _standardized_zs(n: int) -> np.ndarray:
    """A deterministic, symmetric sample with mean exactly 0 and std exactly 1 (ddof=1)."""
    raw = np.linspace(-2.5, 2.5, n)
    return (raw - raw.mean()) / raw.std(ddof=1)


def test_fit_recalibration_recovers_a_known_miscalibration() -> None:
    # Raw probabilities drawn from a true model, then deliberately over-confident (pushed toward
    # 0/1) to simulate the model's real early-season overconfidence - fit_recalibration should
    # pull them most of the way back.
    rng = np.random.default_rng(0)
    n = 4000
    true_p = rng.uniform(0.1, 0.9, n)
    home_won = rng.uniform(0, 1, n) < true_p
    overconfident_p = 1.0 / (1.0 + np.exp(-2.0 * np.log(true_p / (1 - true_p))))  # a=2 too sharp
    games = pd.DataFrame({
        "season": 2010, "week": np.tile([2, 5, 10], n // 3 + 1)[:n],
        "home_win_prob": overconfident_p, "home_won": home_won,
    })

    fit = chaos.fit_recalibration(games)
    for band in ("early", "mid", "late"):
        # The true relationship is a=0.5 (undoing the a=2 overconfidence); the fit should land
        # close to that in every band, each with a healthy sample size.
        assert fit["bands"][band]["a"] == pytest.approx(0.5, abs=0.15)
        assert fit["bands"][band]["n"] > 500


def test_self_check_passes_on_a_well_calibrated_synthetic_archive() -> None:
    weeks = [1, 4, 8]
    z = _standardized_zs(60)
    frame = pd.DataFrame({
        "week": np.tile(weeks, 20),
        "z": z,
        "burnIn": False,
    })
    report = chaos.self_check(frame)
    assert report["overall"]["mean"] == pytest.approx(0.0, abs=0.1)
    assert report["overall"]["std"] == pytest.approx(1.0, abs=0.1)
    for band in ("early", "mid", "late"):
        assert report["byBand"][band]["mean"] == pytest.approx(0.0, abs=0.25)
        assert report["byBand"][band]["std"] == pytest.approx(1.0, abs=0.25)


def test_self_check_flags_a_band_off_by_more_than_the_tolerance() -> None:
    early_z = _standardized_zs(40) + 0.6  # a real miscalibration: early weeks read too chaotic
    mid_z = _standardized_zs(40)
    late_z = _standardized_zs(40)
    frame = pd.DataFrame({
        "week": np.repeat([1, 5, 10], 40),
        "z": np.concatenate([early_z, mid_z, late_z]),
        "burnIn": False,
    })
    report = chaos.self_check(frame)
    assert abs(report["byBand"]["early"]["mean"]) > 0.25
    assert abs(report["byBand"]["mid"]["mean"]) < 0.25
    assert abs(report["byBand"]["late"]["mean"]) < 0.25


def test_self_check_excludes_burn_in_weeks() -> None:
    good = _standardized_zs(40)
    frame = pd.DataFrame({
        "week": [10] * 40 + [10] * 5,
        "z": np.concatenate([good, np.full(5, 50.0)]),  # burn-in weeks are wildly "chaotic"
        "burnIn": [False] * 40 + [True] * 5,
    })
    report = chaos.self_check(frame)
    assert report["overall"]["mean"] == pytest.approx(0.0, abs=0.1)


def test_percentile_rank_is_the_share_strictly_calmer() -> None:
    population = [-1.0, 0.0, 0.5, 1.0, 2.0]
    assert chaos.percentile_rank(1.5, population) == pytest.approx(80.0)
    assert chaos.percentile_rank(-2.0, population) == pytest.approx(0.0)
    assert chaos.percentile_rank(0.0, []) is None


def test_calibration_round_trips_through_disk(tmp_path) -> None:
    path = tmp_path / "chaos_calibration.json"
    calibration = {"bands": {"early": {"a": 1.1, "b": -0.1, "n": 500}}, "fittedThrough": {"season": 2025}}
    chaos.save_calibration(calibration, path)
    assert chaos.load_calibration(path) == calibration
    assert chaos.load_calibration(tmp_path / "missing.json") is None
