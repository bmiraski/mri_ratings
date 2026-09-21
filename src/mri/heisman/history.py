"""Every snapshot of every past season, ready to learn from.

For each season and snapshot week: the standings as they stood, the candidates, and -
because the season is over - what each of them went on to do. From that come the
projection ratios, and everything the backtest scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..ingest import cfbd
from . import project, snapshots, teams


def prepare(seasons: list[int], weekly: pd.DataFrame, final_table: pd.DataFrame, games_all: dict, finals: dict,
            power_np: dict, previous, weeks=project.SNAPSHOT_WEEKS):
    """Standings and candidates at each snapshot, and what became of every candidate.

    ``previous(year)`` returns the set of last season's finalists' normalized names. Returns ``raw``,
    mapping (season, week) to a dict of the snapshot's ``state`` and ``pool`` and the arrays a projection
    is judged by: ``cur`` (production so far), ``games``, ``group``, ``remaining`` (games the team went on
    to play), ``final`` (what he finished with), ``last`` (his rate last season) and ``wins_left`` (how
    many of those games his team won).
    """
    raw: dict[tuple[int, int], dict] = {}
    for y in seasons:
        prior = power_np[y - 1]
        played = games_all[y][games_all[y]["played"]]
        last = snapshots.last_rates(final_table, y)
        for w in weeks:
            state = teams.team_state(played, through_week=w, prior=prior)
            table = weekly[(weekly["season"] == y) & (weekly["week"] == w)]
            pool = snapshots.candidates(table, state, previous(y), last)
            pool["season"] = y
            final = snapshots.finish(pool, final_table[final_table["season"] == y]).fillna(pool["off_score"])
            end = finals[y]
            raw[(y, w)] = {
                "state": state, "pool": pool, "cur": pool["off_score"].to_numpy(), "games": pool["team_games"].to_numpy(),
                "group": pool["group_code"].to_numpy(), "final": final.to_numpy(), "last": pool["last_rate"].to_numpy(),
                "remaining": end.loc[pool["team"], "games"].to_numpy() - pool["team_games"].to_numpy(),
                "wins_left": end.loc[pool["team"], "wins"].to_numpy() - state.loc[pool["team"], "wins"].to_numpy()}
    return raw


def ratio_pools(raw: dict, weeks=project.SNAPSHOT_WEEKS, variant: dict | None = None) -> dict[int, dict[int, np.ndarray]]:
    """The projection ratios of every season at every week, under one way of projecting."""
    out: dict[int, dict[int, np.ndarray]] = {w: {} for w in weeks}
    for (y, w), r in raw.items():
        out[w][y] = project.ratios(r["cur"], r["games"], r["group"], r["remaining"], r["final"], r["last"], variant)
    return out


def team_link(raw: dict, variant: dict | None = None) -> float:
    """How closely a candidate's finish tracks how his team did over the games left.

    The rank correlation between each candidate's ratio (what he did against his projection) and the
    share of his team's remaining games it won, pooled over every snapshot, turned into the correlation
    of the two normal scores that a copula would use.
    """
    from scipy.stats import spearmanr

    xs, ys = [], []
    for r in raw.values():
        projected = r["remaining"] * project.group_rates(r["cur"], r["games"], r["group"], r["last"], variant)
        ok = (r["remaining"] >= 1) & (projected > 0)
        xs.extend(np.clip((r["final"] - r["cur"])[ok] / projected[ok], 0, project.RATIO_CAP))
        ys.extend((r["wins_left"][ok] / r["remaining"][ok]))
    rho = spearmanr(xs, ys).statistic
    return float(2 * np.sin(np.pi * rho / 6))
