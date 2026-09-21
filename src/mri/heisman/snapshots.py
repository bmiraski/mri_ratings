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


LAST_SEASON_GAMES = {2020: 10.0}          # the short season; every other one is about twelve and a half
DEFAULT_GAMES = 12.5


def last_rates(final_table: pd.DataFrame, year: int) -> dict[str, float]:
    """Each player's per-game production the season before ``year``, by player id.

    By id and not by team, so a transfer keeps his history. A player who did not play FBS ball last year
    (a freshman, a newcomer from another division) is simply absent.
    """
    last = funnel.classify(final_table[final_table["season"] == year - 1])
    games = LAST_SEASON_GAMES.get(year - 1, DEFAULT_GAMES)
    best = last.sort_values("off_score", ascending=False).drop_duplicates("player_id")
    return dict(zip(best["player_id"], best["off_score"] / games))


def candidates(table: pd.DataFrame, state: pd.DataFrame, prev: set[str], last: dict[str, float] | None = None) -> pd.DataFrame:
    """The pool for a snapshot of offensive stats and a team state (power_rank, rank, losses, games)."""
    t = table.copy()
    for column in DEFENSIVE:
        if column not in t:
            t[column] = 0.0                     # weekly snapshots carry no defence; the funnel treats that as none
    pool = final_model.pool(t, state["power_rank"], state["rank"], state["losses"], prev, games=state["games"])
    pool["team_games"] = pool["team"].map(state["games"]).astype(float)
    pool["last_rate"] = pool["player_id"].map(last).astype(float) if last is not None else np.nan
    return pool


def finish(pool: pd.DataFrame, final_table: pd.DataFrame) -> pd.Series:
    """Each candidate's finished-season production, by player id and team."""
    f = funnel.classify(final_table).set_index(["player_id", "team"])["off_score"]
    keys = list(zip(pool["player_id"], pool["team"]))
    return pd.Series([float(f.get(k, np.nan)) for k in keys], index=pool.index)
