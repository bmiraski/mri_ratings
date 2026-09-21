"""How good a season each team is having, as of any week.

The Heisman goes to a player on a team voters are watching, so every candidate
carries his team's standing: its rank by the same résumé-heavy blend the season
simulation uses for the committee, and its record.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..gameday import features as gd_features
from ..ingest import cfbd
from ..ratings import mri2


def fbs_of(games: pd.DataFrame) -> list[str]:
    return sorted(set(games.loc[games["class1"] == "fbs", "team1"]) | set(games.loc[games["class2"] == "fbs", "team2"]))


def team_state(games: pd.DataFrame, *, through_week: int | None = None, prior: pd.Series | None = None) -> pd.DataFrame:
    """Rank and record for every FBS team from the regular-season games played through a week.

    Fitted with no preseason prior for a finished season, because the state only
    describes teams that have played a full schedule. Early in a season pass ``prior``:
    three games say little, and a rank built from them alone is mostly noise. Columns: power, resume, rank (committee-style, 1 = best), power_rank,
    wins, losses, games.
    """
    g = games[(games["season_type"] == "regular") & games["played"]]
    if through_week is not None:
        g = g[g["week"] <= through_week]
    g = g.reset_index(drop=True)
    fbs = fbs_of(g)
    teams = sorted(set(g["team1"]) | set(g["team2"]))
    pr = mri2.build_prior(prior, teams, centre_teams=fbs) if prior is not None else None
    model = mri2.fit(g, prior=pr, neutral=g["neutral"], anchor_teams=fbs, with_efficiency=False)
    table = model.table().set_index("team")
    table = table[table.index.isin(fbs)].copy()
    table["rank"] = gd_features.committee_score_rank(table["power"].to_numpy(), table["resume"].to_numpy())
    table["power_rank"] = table["power"].rank(ascending=False, method="min").astype(int)
    wins, losses = {}, {}
    for r in g.itertuples():
        for team, won in ((r.team1, r.win1), (r.team2, r.win2)):
            (wins if won == 1.0 else losses)[team] = (wins if won == 1.0 else losses).get(team, 0) + 1
    table["wins"] = [wins.get(t, 0) for t in table.index]
    table["losses"] = [losses.get(t, 0) for t in table.index]
    table["games"] = table["wins"] + table["losses"]
    table.attrs["home_field"] = float(model.home_field)
    return table
