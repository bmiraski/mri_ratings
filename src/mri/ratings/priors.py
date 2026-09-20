"""Preseason priors that know who is actually on the roster.

MRI 2.0 starts every season from a prior, and for a long time the prior was last
season's rating pulled 30% of the way to average - the same 30% for everyone. That
treats a team that returns its whole roster like a team that lost it to the
transfer portal, and it does badly by the second kind. North Texas and Iowa State
began 2026 with almost no returning production (under 3% of last year's) and were
rated as if they were the teams that had played the year before.

Two more things are known before a game is played, and both carry real
information about the season ahead:

*Talent* - the 247Sports composite of every player on the roster. It is roster
quality accumulated over recruiting classes, and it is the single best
predictor of a rating that last season's rating does not already contain.

*Returning production* - the share of last year's production still on the
roster, by predicted points added.

The prior is a regression of a season's final rating on last season's rating,
talent (as a z-score within FBS) and returning production, fitted on 2015-2025
(``scripts/fit_prior.py``, coefficients in ``data/prior_model.json``). Scored by
leaving each season out in turn it misses by 8.5 points where the old prior
missed by 9.1, and in the first three weeks of a season - when the prior is most
of what the ratings know - it takes about a point off the average error of a
game prediction (``scripts/backtest_priors.py``).

Two exceptions, both deliberate:

*The service academies.* Their recruits are not ranked the way everyone else's
are, and the composite calls Army, Navy and Air Force about two standard deviations
worse than average; on average they beat what that implied by about twenty points. With talent treated
as unmeasured for them the prior is off by under two.

*Teams that were not FBS last year* have no comparable last season, so they keep
the old prior rather than be extrapolated.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import mri2

MODEL_PATH = Path(__file__).resolve().parents[3] / "data" / "prior_model.json"
TALENT_UNMEASURED = ("Army", "Navy", "Air Force")
TERMS = ("intercept", "last_season", "talent", "returning")


def load_model(path: Path = MODEL_PATH) -> dict | None:
    """The fitted coefficients, or None if there are none to use."""
    if not path.exists():
        return None
    model = json.loads(path.read_text())
    return model if all(t in model["coefficients"] for t in TERMS) else None


def talent_scores(talent: pd.Series | None, fbs: list[str]) -> pd.Series:
    """Talent as a z-score within FBS, zero where it is missing or unmeasured."""
    if talent is None or talent.empty:
        return pd.Series(0.0, index=fbs)
    values = talent.reindex(fbs)
    z = ((values - values.mean()) / values.std()).fillna(0.0)
    z[[t for t in TALENT_UNMEASURED if t in z.index]] = 0.0
    return z


def returning_share(returning: pd.DataFrame | None, fbs: list[str]) -> pd.Series:
    """Returning production, with the FBS median standing in where it is missing."""
    if returning is None or returning.empty:
        return pd.Series(0.5, index=fbs)
    share = returning["percent_ppa"].reindex(fbs)
    return share.fillna(share.median() if share.notna().any() else 0.5)


def predict(model: dict, last_season, talent_z, returning) -> np.ndarray:
    c = model["coefficients"]
    return (c["intercept"] + c["last_season"] * np.asarray(last_season, dtype=float)
            + c["talent"] * np.asarray(talent_z, dtype=float)
            + c["returning"] * np.asarray(returning, dtype=float))


def preseason_prior(
    previous: pd.Series | None,
    teams: list[str],
    *,
    fbs: list[str],
    continuing: set[str],
    talent: pd.Series | None,
    returning: pd.DataFrame | None,
    model: dict | None,
    regression: float = mri2.DEFAULT_PRIOR_REGRESSION,
) -> pd.Series:
    """The prior for a season: the regression where it applies, the old rule elsewhere."""
    base = mri2.build_prior(previous, teams, regression, centre_teams=fbs)
    if model is None or previous is None or previous.empty:
        return base

    rated = [t for t in fbs if t in continuing and t in previous.index and t in base.index]
    if not rated:
        return base
    z = talent_scores(talent, fbs)
    share = returning_share(returning, fbs)
    out = base.copy()
    out.loc[rated] = predict(model, previous.reindex(rated), z.reindex(rated), share.reindex(rated))
    return out


@functools.lru_cache(maxsize=8)
def _inputs(year: int):
    """Everything a season's prior needs from the cache, read once per process."""
    from ..ingest import cfbd

    last = cfbd.games(year - 1)
    continuing = frozenset(last.loc[last["class1"] == "fbs", "team1"]) | frozenset(
        last.loc[last["class2"] == "fbs", "team2"])
    return continuing, cfbd.talent(year), cfbd.returning(year)


def for_season(year: int, previous: pd.Series | None, teams: list[str], fbs: list[str]) -> pd.Series:
    """What a build should use as the prior for ``year``.

    Loads the season's talent and returning production and last year's FBS
    membership from the cache. If any of it is missing the old prior is returned,
    so a gap in the data costs accuracy in September and never the build.
    """
    base = mri2.build_prior(previous, teams, centre_teams=fbs)
    model = load_model()
    if model is None or previous is None:
        return base
    try:
        continuing, talent, returning = _inputs(year)
    except Exception as exc:  # noqa: BLE001 - the old prior is a safe answer to any gap
        print(f"  prior for {year}: talent and returning production unavailable ({exc}); "
              "using last season only")
        return base
    return preseason_prior(previous, teams, fbs=fbs, continuing=set(continuing), talent=talent,
                           returning=returning, model=model)
