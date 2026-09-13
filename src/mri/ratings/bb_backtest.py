"""Walk-forward evaluation for basketball.

Same discipline as the football harness: to rate a game, the model may see only
games played before it. Two things differ.

Basketball games are dated rather than numbered by week, so the season is
ordered by tip-off and sliced by position. And neutral sites are flagged in the
data rather than inferred from games played, which is better - a December
tournament in the Bahamas is neutral in November, and the football rule of
"both teams have played twelve" would never catch it.

There is no Classic comparison here. MRI Basketball Classic needs rebounds and
turnovers per game, which the games feed does not carry, so the comparison
would cost a box-score pull for every season. The question this answers is
whether the parameters are right, not which rating wins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..ingest import bb_registry as registry
from . import mri2


def prepare(season: int) -> pd.DataFrame:
    """Completed games for a season, canonical names, chronological."""
    from ..ingest import cbbd

    games = cbbd.games(season)
    if games.empty:
        return games
    games = games.copy()
    for column in ("team1", "team2"):
        games[column] = [registry.resolve(n, n, season=season) for n in games[column]]
    return games.sort_values("start_date").reset_index(drop=True)


def fit_slice(
    games: pd.DataFrame,
    prior: pd.Series | None,
    profile: mri2.Profile = mri2.BASKETBALL_PROFILE,
    season: int | None = None,
    **overrides,
) -> mri2.Ratings:
    """Fit one slice with a profile's settings, overridable for tuning."""
    teams = sorted(set(games["team1"]) | set(games["team2"]))
    # Anchor and centre on D1 only. Non-D1 opponents appear in November and
    # would otherwise drag the scale, exactly as FCS teams did in football.
    lookup_season = season or registry.CURRENT_SEASON
    rated = [t for t in teams if registry.is_d1(t, season=lookup_season)]
    settings = {
        "compression": profile.compression,
        "ridge": profile.ridge,
        "home_field_prior": profile.home_field_prior,
    }
    settings.update(overrides)
    regression = settings.pop("prior_regression", profile.prior_regression)

    return mri2.fit(
        games,
        prior=mri2.build_prior(prior, teams, regression, centre_teams=rated or None),
        neutral=games["neutral"],
        anchor_teams=rated or None,
        with_resume=False,
        with_efficiency=False,
        **settings,
    )


def evaluate_season(
    games: pd.DataFrame,
    prior: pd.Series | None = None,
    *,
    season: int | None = None,
    cutoffs=(0.4, 0.5, 0.6, 0.7, 0.8),
    horizon: float = 0.1,
    profile: mri2.Profile = mri2.BASKETBALL_PROFILE,
    **overrides,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Score the model at several points in one season."""
    from scipy.stats import norm

    if games.empty:
        return pd.DataFrame(), prior

    n = len(games)
    rows = []
    final = prior

    for cut in cutoffs:
        split = int(n * cut)
        stop = min(n, int(n * (cut + horizon)))
        train, test = games.iloc[:split], games.iloc[split:stop]
        if len(test) < 50 or train.empty:
            continue

        model = fit_slice(train, prior, profile, season=season, **overrides)
        final = model.power
        replacement = float(model.power.min()) - 5.0

        home = np.nan_to_num(model.power.reindex(test["team2"]).to_numpy(float), nan=replacement)
        away = np.nan_to_num(model.power.reindex(test["team1"]).to_numpy(float), nan=replacement)
        predicted = home - away + np.where(test["neutral"].to_numpy(bool), 0.0, model.home_field)
        actual = (test["pts2"] - test["pts1"]).to_numpy(float)
        home_won = test["win2"].to_numpy(float) == 1.0
        probability = norm.cdf(predicted / max(model.sigma, 1e-6))

        rows.append(
            {
                "cutoff": cut,
                "train": len(train),
                "test": len(test),
                "accuracy": float(((predicted > 0) == home_won).mean()),
                "mae": float(np.abs(predicted - actual).mean()),
                "brier": float(np.mean((probability - home_won) ** 2)),
                "home_field": model.home_field,
            }
        )

    return pd.DataFrame(rows), final


def evaluate(
    seasons,
    *,
    warmup: pd.Series | None = None,
    scored=None,
    profile: mri2.Profile = mri2.BASKETBALL_PROFILE,
    **overrides,
) -> pd.DataFrame:
    """Walk seasons in order, carrying ratings forward between them."""
    prior = warmup
    wanted = set(scored) if scored is not None else None
    out = []

    for season in sorted(seasons):
        games = prepare(season)
        if games.empty:
            continue
        frame, prior = evaluate_season(
            games, prior, season=season, profile=profile, **overrides
        )
        if wanted is None or season in wanted:
            if not frame.empty:
                out.append(frame.assign(season=season))

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def summarize(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    return pd.Series(
        {
            "accuracy": frame["accuracy"].mean(),
            "mae": frame["mae"].mean(),
            "brier": frame["brier"].mean(),
            "home_field": frame["home_field"].mean(),
            "windows": len(frame),
        }
    )
