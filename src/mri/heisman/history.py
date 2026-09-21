"""Every snapshot of every past season, ready to learn from.

For each season and snapshot week: the standings as they stood, the candidates, and -
because the season is over - what each of them went on to do. From that come the
projection ratios, and everything the backtest scores.
"""

from __future__ import annotations

import pandas as pd

from ..ingest import cfbd
from . import project, snapshots, teams


def prepare(seasons: list[int], weekly: pd.DataFrame, final_table: pd.DataFrame, games_all: dict, finals: dict,
            power_np: dict, previous, weeks=project.SNAPSHOT_WEEKS):
    """Standings and candidates at each snapshot, and the ratios that history supplies.

    ``previous(year)`` returns the set of last season's finalists' normalized names. Returns
    ``(prep, ratio_by)``: prep maps (season, week) to (state, pool); ratio_by maps week to
    {season: ratios}.
    """
    ratio_by: dict[int, dict[int, object]] = {w: {} for w in weeks}
    prep: dict[tuple[int, int], tuple] = {}
    for y in seasons:
        prior = power_np[y - 1]
        played = games_all[y][games_all[y]["played"]]
        for w in weeks:
            state = teams.team_state(played, through_week=w, prior=prior)
            table = weekly[(weekly["season"] == y) & (weekly["week"] == w)]
            pool = snapshots.candidates(table, state, previous(y))
            pool["season"] = y
            final = snapshots.finish(pool, final_table[final_table["season"] == y]).fillna(pool["off_score"])
            remaining = finals[y].loc[pool["team"], "games"].to_numpy() - pool["team_games"].to_numpy()
            ratio_by[w][y] = project.ratios(pool["off_score"], pool["team_games"], pool["group_code"], remaining, final)
            prep[(y, w)] = (state, pool)
    return prep, ratio_by
