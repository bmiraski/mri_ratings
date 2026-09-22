"""A team's résumé, the way the committee actually talks about one: not just a
rating, but who it played, where, and how it did against them.

The committee has used the NET's rank since 2018-19 to sort opponents into four
quadrants and judges a team by its record within each. We have no historical NET
to build that from, so our own power rank stands in for it - the two systems
agree on the shape of the field far more than they disagree, and it is the same
substitution the football committee proxy already makes.

Everything here is "as of Selection Sunday": every regular-season game,
conference tournaments included (they are labelled ``season_type == "regular"``
in this feed, the same as a November game - only the actual NCAA tournament,
which hasn't happened yet, is excluded). A team's own power rank for locating
its games in the quadrant grid is computed once, over the same window, so a
team is never quietly using its own tournament run to inflate the very
opponents its resume is judged against; each team's rank is only ever used to
place *other* teams' games, and by the last day of the sort every team has been
both a subject and an opponent in the same single, self-consistent ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..ingest import bb_registry

# (rank ceiling, quadrant) - checked in order; whatever's left over is Quadrant 4.
QUAD_THRESHOLDS = {
    "home": ((30, 1), (75, 2), (160, 3)),
    "neutral": ((50, 1), (100, 2), (200, 3)),
    "away": ((75, 1), (135, 2), (240, 3)),
}


def quadrant(opponent_rank: float, location: str) -> int:
    """1-4, the way the committee's own sheet buckets a game: same opponent rank, different
    quadrant depending on whether the team travelled, hosted, or met on a neutral floor."""
    for ceiling, quad in QUAD_THRESHOLDS[location]:
        if opponent_rank <= ceiling:
            return quad
    return 4


def selection_sunday_games(season: int, games: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every game a Selection Sunday committee would have seen: the whole regular season,
    conference tournaments included, nothing past it."""
    from ..ingest import cbbd

    g = games if games is not None else cbbd.games(season)
    return g[g["season_type"] == "regular"]


def team_features(season: int, ratings, games: pd.DataFrame, conference_of: dict[str, str],
                  conference_champions: dict[str, str]) -> pd.DataFrame:
    """One row per team that played a Selection Sunday game: record, quadrant record, schedule
    strength, résumé, and whether it's the guaranteed automatic bid for its conference.

    ``ratings`` is a full :class:`mri.ratings.mri2.Ratings` (power *and* résumé), fit with
    ``with_resume=True``, over every team this season - an opponent needs a rank too, whether
    or not it ever makes a bracket.
    """
    power = ratings.power
    resume_of = ratings.resume if ratings.resume is not None else pd.Series(dtype=float)
    rank = power.rank(ascending=False, method="min")
    won_auto = {v for v in conference_champions.values()}
    rows: dict[str, dict] = {}

    def bucket(team: str) -> dict:
        return rows.setdefault(team, {
            "team": team, "wins": 0, "losses": 0, "road_wins": 0, "neutral_wins": 0,
            "quad_wins": {1: 0, 2: 0, 3: 0, 4: 0}, "quad_losses": {1: 0, 2: 0, 3: 0, 4: 0},
            "opponent_ranks": [],
        })

    for r in games.itertuples():
        for team, opp, won, hosted in ((r.team1, r.team2, r.win1 == 1.0, False), (r.team2, r.team1, r.win2 == 1.0, True)):
            if opp not in rank.index or team not in rank.index:
                continue                # a team outside this ranking (or an opponent outside it) tells a resume nothing
            row = bucket(team)
            location = "neutral" if r.neutral else ("home" if hosted else "away")
            q = quadrant(rank[opp], location)
            row["opponent_ranks"].append(rank[opp])
            if won:
                row["wins"] += 1
                row["quad_wins"][q] += 1
                if location == "away":
                    row["road_wins"] += 1
                elif location == "neutral":
                    row["neutral_wins"] += 1
            else:
                row["losses"] += 1
                row["quad_losses"][q] += 1

    out = []
    for team, row in rows.items():
        out.append({
            "team": team, "season": season, "conference": conference_of.get(team),
            "wins": row["wins"], "losses": row["losses"],
            "road_neutral_wins": row["road_wins"] + row["neutral_wins"],
            "quad1_wins": row["quad_wins"][1], "quad1_losses": row["quad_losses"][1],
            "quad2_wins": row["quad_wins"][2], "quad2_losses": row["quad_losses"][2],
            "quad3_losses": row["quad_losses"][3], "quad4_losses": row["quad_losses"][4],
            "bad_losses": row["quad_losses"][3] + row["quad_losses"][4],
            "quality_wins": row["quad_wins"][1],
            "sos": float(np.mean(row["opponent_ranks"])) if row["opponent_ranks"] else np.nan,
            "power": power.get(team, np.nan), "powerRank": rank.get(team, np.nan),
            "resume": resume_of.get(team, np.nan), "isAutoBid": team in won_auto,
        })
    return pd.DataFrame(out)
