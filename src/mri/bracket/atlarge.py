"""Who gets the at-large bids, and everyone's seed: one composite score, fit
against the real seed line rather than a proxy for it.

The football committee proxy was calibrated against the CFP's top-25 poll,
because that's all a twelve-team field left to compare against. Basketball's
field is bigger and every team in it carries a real, exact seed number - 1
through 16, not a top-25 cutoff - so the fit target here is the actual number
the actual committee actually assigned, for fourteen real seasons. That is a
better thing to calibrate against than anything the football side had.

A team's score is a linear blend of what the committee says it weighs:
strength (power), résumé (wins above expected, and specifically wins in
Quadrant 1), how bad the losses were, and the schedule played. Fit once by
ordinary ridge regression against seed number, using every team that actually
made a historical field - auto bids included, since the same committee seeds
an automatic qualifier by the same yardstick it seeds anyone else. Rank order
does the rest: apply the fitted score to the *whole* division, not just the
historical field, and whoever ranks above the cutoff line is the at-large
field, whatever that line happens to be need up to (44 this year, 36 before it).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ("power_rank_log", "resume", "quad1_win_pct", "bad_loss_rate", "sos_rank_log", "road_neutral_wins")


def _safe_div(a, b):
    return np.where(b > 0, a / np.maximum(b, 1), 0.0)


def design(features: pd.DataFrame) -> np.ndarray:
    """The feature matrix a score is built from, plus an intercept column."""
    games = features["wins"] + features["losses"]
    quad1_games = features["quad1_wins"] + features["quad1_losses"]
    cols = {
        "power_rank_log": np.log(np.maximum(features["powerRank"], 1.0)),
        "resume": features["resume"].fillna(0.0),
        "quad1_win_pct": _safe_div(features["quad1_wins"], quad1_games),
        "bad_loss_rate": _safe_div(features["bad_losses"], games),
        "sos_rank_log": np.log(np.maximum(features["sos"], 1.0)),
        "road_neutral_wins": features["road_neutral_wins"].fillna(0.0),
    }
    X = np.column_stack([cols[f] for f in FEATURES])
    return np.column_stack([X, np.ones(len(features))])


def fit(seasons: list[pd.DataFrame], *, ridge: float = 3.0) -> np.ndarray:
    """Ridge regression of actual seed number on the design matrix, over every field team of every
    season passed in (each frame needs a ``seed`` column already joined on)."""
    all_features = pd.concat(seasons, ignore_index=True)
    X = design(all_features)
    y = all_features["seed"].to_numpy(dtype=float)
    penalty = np.eye(X.shape[1]) * ridge
    penalty[-1, -1] = 0.0                     # never shrink the intercept
    return np.linalg.solve(X.T @ X + penalty, X.T @ y)


def score(features: pd.DataFrame, beta: np.ndarray) -> pd.Series:
    """A lower score is a better team - the same direction as a seed number, so sorting ascending
    gives the field in the order the committee would rank it."""
    return pd.Series(design(features) @ beta, index=features.index)


def select_and_seed(features: pd.DataFrame, beta: np.ndarray, *, auto_bids: dict[str, str], field_size: int) -> pd.DataFrame:
    """The projected field: every automatic qualifier, plus enough of the best-scoring remaining
    teams to fill it out, seeded by the same score across the two groups together.

    ``auto_bids`` maps conference -> the team that has (or, live, is projected to have) the
    automatic bid; every value in it is in the field no matter its score.
    """
    f = features.copy()
    f["compositeScore"] = score(f, beta)
    autos = set(auto_bids.values())
    f["bidType"] = np.where(f["team"].isin(autos), "auto", "at-large")
    at_large_slots = field_size - len(autos)
    pool = f[f["bidType"] == "at-large"].nsmallest(at_large_slots, "compositeScore")
    projected = pd.concat([f[f["bidType"] == "auto"], pool]).sort_values("compositeScore").reset_index(drop=True)
    rank_position = np.arange(1, len(projected) + 1)
    projected["projectedSeed"] = np.minimum(np.ceil(rank_position / (field_size / 16)).astype(int), 16)
    return projected
