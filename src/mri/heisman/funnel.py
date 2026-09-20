"""Who could plausibly win: cutting a season's ten thousand players to about seventy.

The feed carries everyone, so this is arithmetic, not API calls. Three cuts:

*Position group.* Each player is a quarterback, a running back, a receiver or a
defender, by what he actually did that season and not by the position on the
roster - Travis Hunter and Jabrill Peppers both played on both sides of the ball.

*Production within the group.* A simple volume score (yards, plus twenty for every
touchdown; for defenders, tackles and the plays that change games). It is crude on
purpose: this stage only has to let the eventual winner through, and the model that
follows does the real ranking.

*A team that could be in the conversation.* Only teams in the top of our rankings,
because a voter's attention follows the team: in every season from 2013 to 2025 the
winner played for a top-ten team by our numbers.

Tested against history: the thirteen winners of 2013-2025 all survive it, along with
most of the other finalists (the ones it loses are efficiency passers and two-way
defenders, none of whom finished better than fourth).
"""

from __future__ import annotations

import pandas as pd

GROUP_SIZES = {"QB": 20, "RB": 20, "REC": 25, "DEF": 10}
TEAM_RANK_CUTOFF = 45


def classify(table: pd.DataFrame) -> pd.DataFrame:
    """Add total yards and touchdowns, the volume scores, and the position group."""
    t = table.copy()
    t["tot_yds"] = t["pass_yds"] + t["rush_yds"] + t["rec_yds"]
    t["tot_td"] = t["pass_td"] + t["rush_td"] + t["rec_td"]
    t["off_score"] = t["tot_yds"] + 20 * t["tot_td"]
    t["def_score"] = t["def_tot"] + 4 * t["def_sacks"] + 3 * t["def_tfl"] + 8 * t["def_int"] + 3 * t["def_pd"]

    def group(r) -> str:
        if r["pass_att"] >= 100 and r["pass_att"] >= r["rush_car"]:
            return "QB"
        if r["def_score"] >= 60 and r["off_score"] < 400:
            return "DEF"
        return "REC" if r["rec_rec"] > r["rush_car"] else "RB"

    t["group"] = t.apply(group, axis=1)
    return t


def candidates(season_table: pd.DataFrame, team_rank: pd.Series, *, sizes: dict | None = None,
               cutoff: int = TEAM_RANK_CUTOFF) -> pd.DataFrame:
    """The players worth scoring in one season, best of each group first.

    ``team_rank`` is each team's rank in our ratings (1 = best), as of the moment being asked about.
    """
    sizes = sizes or GROUP_SIZES
    t = classify(season_table)
    t = t.assign(team_rank=t["team"].map(team_rank))
    kept = []
    for group, size in sizes.items():
        score = "def_score" if group == "DEF" else "off_score"
        pool = t[(t["group"] == group) & (t["team_rank"] <= cutoff)]
        kept.append(pool.nlargest(size, score))
    return pd.concat(kept)
