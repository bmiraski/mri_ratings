"""MRI 2.0 must stay sane, stay calibrated, and stay better than Classic.

The last of those is the point of the whole exercise, so it is asserted on the
seasons that took no part in hyperparameter tuning.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mri.ratings import backtest, mri2

ARCHIVE = Path(__file__).resolve().parents[1] / "data" / "parquet" / "archive_games.parquet"
HOLDOUT = list(range(2014, 2020))

pytestmark = pytest.mark.skipif(not ARCHIVE.exists(), reason="archive not built")


@pytest.fixture(scope="module")
def games() -> pd.DataFrame:
    return pd.read_parquet(ARCHIVE)


@pytest.fixture(scope="module")
def season_2019(games: pd.DataFrame) -> mri2.Ratings:
    return mri2.fit(games[games["season"] == 2019])


def test_compression_is_near_identity_for_normal_margins() -> None:
    for margin in (3, 7, 10, 14):
        assert abs(mri2.compress(margin) - margin) < 1.0


def test_compression_saturates_on_blowouts() -> None:
    assert mri2.compress(45) < 45
    assert mri2.compress(80) - mri2.compress(60) < 3.0
    assert mri2.compress(-45) == pytest.approx(-mri2.compress(45))


def test_ratings_are_centred_on_an_average_team(season_2019: mri2.Ratings) -> None:
    fbs = season_2019.power.drop(mri2.POOLED_FCS, errors="ignore")
    assert abs(fbs.mean()) < 2.0, "an average FBS team should rate near zero"


def test_power_spread_is_realistic(season_2019: mri2.Ratings) -> None:
    fbs = season_2019.power.drop(mri2.POOLED_FCS, errors="ignore")
    assert 15.0 < fbs.max() < 45.0, "best team should be 15-45 points above average"
    assert -45.0 < fbs.min() < -15.0


def test_home_field_is_plausible(season_2019: mri2.Ratings) -> None:
    assert 1.0 < season_2019.home_field < 5.0, "modern CFB home field is roughly 2-4 points"


def test_pooled_fcs_rates_near_the_bottom(season_2019: mri2.Ratings) -> None:
    """The pooled opponent is an average of every non-FBS team played, so it
    belongs near the bottom of FBS - but not necessarily beneath all of it. In
    2019 exactly one FBS team, 1-11 Massachusetts, rated below it, which is the
    right answer rather than a bug."""
    fbs = season_2019.power.drop(mri2.POOLED_FCS)
    percentile = (fbs < season_2019.power[mri2.POOLED_FCS]).mean()
    assert percentile < 0.05


def test_resume_rewards_the_undefeated_champion(season_2019: mri2.Ratings) -> None:
    """LSU went 15-0 in 2019 through the hardest schedule in the sport."""
    assert season_2019.resume.idxmax() == "LSU"


def test_power_and_resume_disagree_somewhere(season_2019: mri2.Ratings) -> None:
    """If the two numbers always agreed, only one of them would be needed."""
    table = season_2019.table()
    assert not table["rank"].equals(table["resume_rank"].astype(int))


def test_postseason_games_are_flagged_neutral(games: pd.DataFrame) -> None:
    season = games[games["season"] == 2019].reset_index(drop=True)
    neutral = mri2.mark_postseason(season)
    assert 40 <= neutral.sum() < 60, "40 bowls plus about ten title games"
    assert not neutral.iloc[:200].any(), "nothing early in the year is postseason"


def test_prediction_is_symmetric(season_2019: mri2.Ratings) -> None:
    home_view = season_2019.predict("Ohio State", "Michigan")
    away_view = season_2019.predict("Michigan", "Ohio State")
    assert home_view == pytest.approx(-away_view + 2 * season_2019.home_field)


def test_neutral_site_removes_home_field(season_2019: mri2.Ratings) -> None:
    edge = season_2019.predict("Clemson", "Alabama") - season_2019.predict(
        "Clemson", "Alabama", neutral=True
    )
    assert edge == pytest.approx(season_2019.home_field)


@pytest.mark.slow
def test_beats_classic_on_holdout_seasons(games: pd.DataFrame) -> None:
    """The reason MRI 2.0 exists. Measured only on seasons never tuned on."""
    frame = backtest.evaluate_archive(games, seasons=HOLDOUT)
    mri2_accuracy = frame["mri2_accuracy"].mean()
    classic_accuracy = frame["classic_accuracy"].mean()
    assert mri2_accuracy > classic_accuracy + 0.01, (
        f"MRI 2.0 {mri2_accuracy:.4f} vs Classic {classic_accuracy:.4f} - "
        "the rewrite has stopped paying for itself"
    )


@pytest.mark.slow
def test_margin_error_stays_under_fourteen_points(games: pd.DataFrame) -> None:
    frame = backtest.evaluate_archive(games, seasons=HOLDOUT, with_classic=False)
    assert frame["mri2_mae"].mean() < 14.0


@pytest.mark.slow
def test_win_probabilities_are_calibrated(games: pd.DataFrame) -> None:
    """A Brier score above 0.20 would mean the probabilities are barely better
    than guessing the home team every time."""
    frame = backtest.evaluate_archive(games, seasons=HOLDOUT, with_classic=False)
    assert frame["mri2_brier"].mean() < 0.20


def test_prior_shrinks_toward_the_middle() -> None:
    previous = pd.Series({"Good": 20.0, "Bad": -20.0, "Average": 0.0})
    prior = mri2.build_prior(previous, ["Good", "Bad", "Average", "Newcomer"], regression=0.5)
    assert prior["Good"] == pytest.approx(10.0)
    assert prior["Bad"] == pytest.approx(-10.0)
    assert prior["Newcomer"] < prior["Average"], "an unknown team starts below average"


def test_fit_rejects_an_empty_schedule() -> None:
    with pytest.raises(ValueError):
        mri2.fit(pd.DataFrame(columns=["team1", "team2", "pts1", "pts2", "win1", "win2"]))
