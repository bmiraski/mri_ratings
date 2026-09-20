"""The data behind the basketball prior: what to fit it to, and how to grade it.

Every season since 2010 is rated by the site's own method - each season's prior
the old rule applied to the previous season's final ratings - so the target and
the last-season feature are on the scale the rest of the pipeline uses. Each
Division I team that also played the year before is then one row: last season's
rating, what its roster turned out to be, and how good it was.

The roster for a season already played is everyone who played for the team that
season. That is a little kinder than a real preseason roster - a returning player
who never suits up looks like one who left - which is why the live model is scored
by walk-forward game predictions and not only by how well it fits.

Used by ``scripts/fit_bb_prior.py`` and ``scripts/backtest_bb_priors.py``. Nothing
on the build path imports it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..ingest import bb_registry as registry
from . import bb_priors, mri2

FIRST = 2010
LAST = registry.CURRENT_SEASON - 1          # the last season that has been played
YEARS = list(range(2012, LAST + 1))
GAMES = bb_priors.PARQUET / "bb_classic_games.parquet"


def season_games(season: int) -> pd.DataFrame:
    games = pd.read_parquet(GAMES)
    games = games[games["season"] == season].copy()
    for column in ("team1", "team2"):
        games[column] = [registry.resolve(n, n, season=season) for n in games[column]]
    return games.sort_values("start_date").reset_index(drop=True)


def division_one(season: int, games: pd.DataFrame) -> list[str]:
    teams = sorted(set(games["team1"]) | set(games["team2"]))
    return [t for t in teams if registry.is_d1(t, season=season)]


def fit_season(games: pd.DataFrame, prior: pd.Series | None, d1: list[str]) -> mri2.Ratings:
    p = mri2.BASKETBALL_PROFILE
    return mri2.fit(games, prior=prior, neutral=games["neutral"], anchor_teams=d1, compression=p.compression,
                    ridge=p.ridge, home_field_prior=p.home_field_prior, with_resume=False, with_efficiency=False)


def old_chain() -> dict[int, pd.Series]:
    """Final ratings for every season, each starting from the last under the old prior."""
    p = mri2.BASKETBALL_PROFILE
    chain, previous = {}, None
    for season in range(FIRST, LAST + 1):
        games = season_games(season)
        d1 = division_one(season, games)
        teams = sorted(set(games["team1"]) | set(games["team2"]))
        prior = mri2.build_prior(previous, teams, p.prior_regression, centre_teams=d1) if previous is not None else None
        ratings = fit_season(games, prior, d1)
        previous = ratings.power
        chain[season] = ratings.power[ratings.power.index.isin(d1)]
    return chain


def dataset(chain: dict[int, pd.Series]) -> pd.DataFrame:
    played, recruits, draft = bb_priors._tables()
    rows = []
    for season in YEARS:
        d1 = list(chain[season].index)
        roster = {t: set(g["athlete_id"]) for t, g in played[played["season"] == season].groupby("team_c")}
        feats = bb_priors.features(season, d1, played, recruits, draft, roster=roster)
        feats["y"] = [chain[season].get(t, np.nan) for t in feats.index]
        feats["last_season"] = [chain[season - 1].get(t, np.nan) for t in feats.index]
        feats["season"] = season
        rows.append(feats.reset_index())
    return pd.concat(rows, ignore_index=True)


ROSTER_FEATURES = ("ret_ws", "in_ws", "frosh")
NO_ROSTER_FEATURES = ("vet_min", "draft_min", "frosh")


def design(frame: pd.DataFrame, names) -> np.ndarray:
    return np.column_stack([np.ones(len(frame)), frame["last_season"]] + [frame[n] for n in names])


def fit_coefficients(frame: pd.DataFrame, names) -> dict[str, float]:
    beta = np.linalg.lstsq(design(frame, names), frame["y"], rcond=None)[0]
    return dict(zip(["intercept", "last_season", *names], (round(float(b), 4) for b in beta)))


def complete(frame: pd.DataFrame) -> pd.DataFrame:
    need = ["y", "last_season", *ROSTER_FEATURES, *NO_ROSTER_FEATURES]
    return frame.dropna(subset=need).reset_index(drop=True)


def model_from(frame: pd.DataFrame, tighten: float) -> dict:
    return {
        "roster": {"features": list(ROSTER_FEATURES), "coefficients": fit_coefficients(frame, ROSTER_FEATURES)},
        "noRoster": {"features": list(NO_ROSTER_FEATURES), "coefficients": fit_coefficients(frame, NO_ROSTER_FEATURES)},
        "tighten": tighten,
    }
