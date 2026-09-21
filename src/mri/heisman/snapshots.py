"""The candidates at any point of a season.

The same funnel the finished season uses, applied to season-to-date numbers and to
the team standings as of that week, so a candidate in Week 5 is chosen the way one
in December is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data, features, final_model, funnel

DEFENSIVE = ["def_tot", "def_solo", "def_sacks", "def_tfl", "def_pd", "def_td", "def_int"]


def candidates(table: pd.DataFrame, state: pd.DataFrame, prev: set[str]) -> pd.DataFrame:
    """The pool for a snapshot of offensive stats and a team state (power_rank, rank, losses, games)."""
    t = table.copy()
    for column in DEFENSIVE:
        if column not in t:
            t[column] = 0.0                     # weekly snapshots carry no defence; the funnel treats that as none
    pool = final_model.pool(t, state["power_rank"], state["rank"], state["losses"], prev, games=state["games"])
    pool["team_games"] = pool["team"].map(state["games"]).astype(float)
    return pool


def finish(pool: pd.DataFrame, final_table: pd.DataFrame) -> pd.Series:
    """Each candidate's finished-season production, by player id and team."""
    f = funnel.classify(final_table).set_index(["player_id", "team"])["off_score"]
    keys = list(zip(pool["player_id"], pool["team"]))
    return pd.Series([float(f.get(k, np.nan)) for k in keys], index=pool.index)
