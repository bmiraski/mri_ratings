"""Walk-forward evaluation: does MRI 2.0 actually beat MRI Classic?

The only honest test of a rating is out-of-sample prediction. For each season we
fit both models on the games played up to a cutoff and score them on the games
that come next, which they have never seen. Row order in the archive is
chronological, so the prefix really is "the season so far".

Both models get the same games. MRI 2.0 additionally gets a prior built from the
previous season, which is not an unfair advantage but one of the things being
tested: whether carrying last year forward helps more than it misleads.

Metrics
-------
accuracy   share of games whose winner was picked correctly
mae        mean absolute error of the predicted margin, in points (2.0 only,
           since Classic's rating is an index and cannot predict a spread)
brier      calibration of the win probability, lower is better (2.0 only)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import classic, mri2


def _classic_picks(train: pd.DataFrame, test: pd.DataFrame, home_edge: float) -> np.ndarray:
    """Classic picks the higher rating, with a nudge for home field.

    Classic has no notion of points, so home field is expressed as a rating
    bonus scaled to the spread of the ratings themselves - the most generous
    reading of what the spreadsheet's Pickem sheet did.
    """
    ratings = classic.compute(train).set_index("team")["mri"]
    spread = ratings.std(ddof=1) or 1.0
    bonus = home_edge * spread / 10.0

    fallback = float(ratings.min()) - 10.0
    home = np.nan_to_num(ratings.reindex(test["team2"]).to_numpy(float), nan=fallback)
    away = np.nan_to_num(ratings.reindex(test["team1"]).to_numpy(float), nan=fallback)
    return (home + bonus) > away


def evaluate_season(
    games: pd.DataFrame,
    prior: pd.Series | None = None,
    *,
    cutoffs=(0.4, 0.5, 0.6, 0.7, 0.8),
    horizon: float = 0.1,
    with_classic: bool = True,
    **fit_kwargs,
) -> pd.DataFrame:
    """Score both models at several points in one season."""
    from scipy.stats import norm

    games = games.reset_index(drop=True)
    neutral = mri2.mark_postseason(games)
    n = len(games)
    rows = []

    for cut in cutoffs:
        split = int(n * cut)
        stop = min(n, int(n * (cut + horizon)))
        train, test = games.iloc[:split], games.iloc[split:stop]
        if len(test) < 10 or train.empty:
            continue

        model = mri2.fit(
            train,
            prior=prior,
            neutral=neutral.iloc[:split],
            with_resume=False,
            with_efficiency=False,
            **fit_kwargs,
        )

        test_neutral = neutral.iloc[split:stop].to_numpy(bool)
        home = np.nan_to_num(
            model.power.reindex(test["team2"]).to_numpy(float), nan=mri2.REPLACEMENT_PRIOR
        )
        away = np.nan_to_num(
            model.power.reindex(test["team1"]).to_numpy(float), nan=mri2.REPLACEMENT_PRIOR
        )

        predicted = home - away + np.where(test_neutral, 0.0, model.home_field)
        actual = test["pts2"].to_numpy(float) - test["pts1"].to_numpy(float)
        home_won = test["win2"].to_numpy(float) == 1.0
        probability = norm.cdf(predicted / max(model.sigma, 1e-6))

        rows.append(
            {
                "cutoff": cut,
                "train_games": len(train),
                "test_games": len(test),
                "mri2_accuracy": float(((predicted > 0) == home_won).mean()),
                "mri2_mae": float(np.abs(predicted - actual).mean()),
                "mri2_brier": float(np.mean((probability - home_won) ** 2)),
                "classic_accuracy": (
                    float((_classic_picks(train, test, model.home_field) == home_won).mean())
                    if with_classic
                    else float("nan")
                ),
                "home_field": model.home_field,
            }
        )
    return pd.DataFrame(rows)


def evaluate_archive(
    games: pd.DataFrame,
    *,
    use_prior: bool = True,
    prior_regression: float = mri2.DEFAULT_PRIOR_REGRESSION,
    seasons=None,
    with_classic: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Walk every season in order, carrying each season's ratings forward.

    ``seasons`` limits which seasons are *scored*; earlier ones are still fitted
    so the prior chain stays intact. That is what lets hyperparameters be tuned
    on one block of years and validated on another.
    """
    results = []
    previous: pd.Series | None = None
    scored = set(seasons) if seasons is not None else None

    for season in sorted(games["season"].unique()):
        season_games = games[games["season"] == season]
        teams = sorted(set(season_games["team1"]) | set(season_games["team2"]))
        prior = mri2.build_prior(previous, teams, prior_regression) if use_prior else None

        if scored is not None and season not in scored:
            previous = mri2.fit(
                season_games, prior=prior, with_resume=False, with_efficiency=False, **kwargs
            ).power
            continue

        frame = evaluate_season(season_games, prior=prior, with_classic=with_classic, **kwargs)
        if not frame.empty:
            results.append(frame.assign(season=season))

        previous = mri2.fit(
            season_games, prior=prior, with_resume=False, with_efficiency=False, **kwargs
        ).power

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def summarize(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "mri2_accuracy": frame["mri2_accuracy"].mean(),
            "classic_accuracy": frame["classic_accuracy"].mean(),
            "edge": frame["mri2_accuracy"].mean() - frame["classic_accuracy"].mean(),
            "mri2_mae": frame["mri2_mae"].mean(),
            "mri2_brier": frame["mri2_brier"].mean(),
            "home_field": frame["home_field"].mean(),
            "windows": len(frame),
        }
    )
