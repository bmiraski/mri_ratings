"""The data behind the preseason prior: what to fit it to.

Every season 2013-2025 is rated on its own with no prior, so the target does not
contain the thing being predicted. From 2015, the first season with talent data,
each FBS team that was also FBS the year before is one row. 2020 is left out as a
target: a season of a few games and cancelled schedules teaches nothing about a
normal one.

Used by ``scripts/fit_prior.py`` to fit the coefficients and by
``scripts/backtest_priors.py`` to grade them. Nothing on the build path imports it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..ingest import cfbd
from . import mri2, priors

YEARS = [y for y in range(2015, 2026) if y != 2020]


def season_ratings() -> tuple[dict[int, pd.Series], dict[int, list[str]]]:
    power, fbs = {}, {}
    for year in range(2013, 2026):
        games = cfbd.games(year)
        games = games[(games["season_type"] == "regular") & games["played"]].reset_index(drop=True)
        teams = sorted(set(games.loc[games["class1"] == "fbs", "team1"])
                       | set(games.loc[games["class2"] == "fbs", "team2"]))
        model = mri2.fit(games, prior=None, neutral=games["neutral"], anchor_teams=teams,
                         with_resume=False, with_efficiency=False)
        power[year] = model.power[model.power.index.isin(teams)]
        fbs[year] = teams
    return power, fbs


def dataset(power: dict[int, pd.Series], fbs: dict[int, list[str]]) -> pd.DataFrame:
    rows = []
    for year in sorted(set(YEARS)):
        teams = fbs[year]
        z = priors.talent_scores(cfbd.talent(year), teams)
        share = priors.returning_share(cfbd.returning(year), teams)
        for team in teams:
            if team not in fbs[year - 1]:
                continue                                  # not FBS last year: nothing to compare
            rows.append({"year": year, "team": team, "y": power[year][team],
                         "last_season": power[year - 1][team], "talent": z[team],
                         "returning": share[team]})
    return pd.DataFrame(rows)


def design(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(frame)), frame["last_season"], frame["talent"], frame["returning"]])


