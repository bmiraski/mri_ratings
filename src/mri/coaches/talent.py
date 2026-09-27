"""The plan's §3 talent check: does a team's 247Sports talent composite - this
season's and last season's - explain ``power_end`` well enough to be worth
publishing as its own metric?

Regression, not a rating: two features (this season's talent z-score, last
season's) and an intercept, fit on every FBS team-season 2015+ (2015 is the
composite's first year - see ``mri.ingest.cfbd.talent``) with ``power_end``
(``coach_ratings_history.parquet``'s ``power`` column) as the target.
Z-scoring reuses ``priors.talent_scores`` - the same service-academy
handling (unmeasured, not zero-talent) and the same "missing means average"
fallback the preseason prior already relies on, so a team with no prior-
season talent (its first FBS year, or the composite's own first year, 2015)
just gets a neutral 0 for that feature rather than being dropped.

The gate is decided empirically, by leave-one-season-out cross-validation
(the same pattern ``scripts/fit_prior.py`` uses): out-of-sample R² has to
clear ``GATE_R2`` in ``fit()``'s own held-out check before ``vs_talent`` is
worth shipping as a metric or a future hot-seat-model feature. If it
doesn't, the plan says to drop it and note that on the method page - this
module still returns the fit either way, so the caller can report the
number either way rather than silently picking one path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..ingest import cfbd
from ..ratings import priors
from .metrics import fbs_for

START_YEAR = 2015
GATE_R2 = 0.10  # "a meaningful share of the variance" - see the module docstring
TERMS = ("intercept", "talent", "talent_last")


def dataset(ratings_history: pd.DataFrame, *, talent_fn=cfbd.talent) -> pd.DataFrame:
    """One row per FBS team-season, 2015+: ``power_end`` (``power``), this season's
    and last season's talent z-score.

    ``talent_fn`` takes a season and returns a raw talent Series (same shape
    as ``cfbd.talent``); tests inject a synthetic one instead of calling the
    real API.
    """
    seasons = sorted(int(s) for s in ratings_history["season"].unique() if s >= START_YEAR)
    talent_z: dict[int, pd.Series] = {}
    rows = []
    for season_ in seasons:
        teams = ratings_history.loc[ratings_history["season"] == season_, "team"].tolist()
        fbs = fbs_for(teams, season_)
        if season_ not in talent_z:
            talent_z[season_] = priors.talent_scores(talent_fn(season_), fbs)
        if season_ - 1 not in talent_z:
            talent_z[season_ - 1] = priors.talent_scores(talent_fn(season_ - 1), fbs)

        power = ratings_history.loc[ratings_history["season"] == season_].set_index("team")["power"]
        for team in fbs:
            rows.append({
                "season": season_, "team": team, "power_end": power.get(team),
                "talent": talent_z[season_].get(team, 0.0),
                "talent_last": talent_z[season_ - 1].get(team, 0.0),
            })
    return pd.DataFrame(rows).dropna(subset=["power_end"])


def design(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(frame)), frame["talent"], frame["talent_last"]])


def fit(ratings_history: pd.DataFrame, *, talent_fn=cfbd.talent) -> dict:
    """Leave-one-season-out cross-validated fit: coefficients, out-of-sample R², and the gate's verdict."""
    data = dataset(ratings_history, talent_fn=talent_fn)
    residual = np.zeros(len(data))
    for season_ in sorted(data["season"].unique()):
        train = data[data["season"] != season_]
        test_idx = data.index[data["season"] == season_]
        beta = np.linalg.lstsq(design(train), train["power_end"], rcond=None)[0]
        residual[data.index.get_indexer(test_idx)] = (
            data.loc[test_idx, "power_end"].to_numpy() - design(data.loc[test_idx]) @ beta
        )

    actual = data["power_end"].to_numpy()
    ss_res = float((residual ** 2).sum())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    beta = np.linalg.lstsq(design(data), data["power_end"], rcond=None)[0]
    coefficients = dict(zip(TERMS, (round(float(b), 4) for b in beta)))

    return {
        "seasons": [int(data["season"].min()), int(data["season"].max())],
        "teamSeasons": int(len(data)),
        "coefficients": coefficients,
        "outOfSampleR2": round(r2, 4),
        "outOfSampleRmse": round(float(np.sqrt((residual ** 2).mean())), 2),
        "gate": GATE_R2,
        "passed": bool(r2 >= GATE_R2),
    }


def vs_talent(ratings_history: pd.DataFrame, model: dict, *, talent_fn=cfbd.talent) -> pd.Series:
    """Every FBS team-season's residual (actual power_end minus what talent alone predicts),
    indexed like ``dataset`` - only meaningful to publish if ``model["passed"]``."""
    data = dataset(ratings_history, talent_fn=talent_fn)
    c = model["coefficients"]
    predicted = c["intercept"] + c["talent"] * data["talent"] + c["talent_last"] * data["talent_last"]
    return (data["power_end"] - predicted).rename("vs_talent").set_axis(
        pd.MultiIndex.from_frame(data[["team", "season"]])
    )
