"""One season's tournament, ground-truthed: who got in, who was an automatic bid,
and what conference-standings seed order the placeholder and the simulation both
need.

The committee's own seeding uses tiebreakers (head-to-head, results against
common opponents) this data can't fully reconstruct; ranking by conference
win percentage, with ties broken by overall record, is the approximation used
everywhere a conference's seed order is needed. It is exactly the ordering the
placeholder itself uses, so the two are apples to apples.
"""

from __future__ import annotations

import pandas as pd

from ..ingest import bb_bracket, bb_registry, cbbd
from ..ratings import mri2


def _canonical(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("team1", "team2"):
        frame[column] = [bb_registry.resolve(n, n, season=season) for n in frame[column]]
    return frame


def team_power(season: int, *, through: str | None = None, games: pd.DataFrame | None = None) -> tuple[pd.Series, float]:
    """Power ratings as of a point in the season, and the home-court edge fit alongside them.

    Regular-season games only: the conference and NCAA tournaments are exactly what this
    is trying to predict, so neither can be in the games the rating is fit from. The
    home-court number is this same season's regular season, not a fixed constant - it
    only matters to the simulation for a conference whose tournament is played on campus
    rather than at one neutral site, but every conference gets it from the same fit.
    """
    g = games if games is not None else cbbd.games(season)
    g = _canonical(g, season)
    g = g[g["season_type"] == "regular"]
    if through:
        g = g[g["start_date"] < through]
    if g.empty:
        return pd.Series(dtype=float), 0.0
    teams = sorted(set(g["team1"]) | set(g["team2"]))
    d1 = [t for t in teams if bb_registry.is_d1(t, season=season)]
    profile = mri2.BASKETBALL_PROFILE
    fitted = mri2.fit(g, neutral=g["neutral"], anchor_teams=d1 or None, compression=profile.compression,
                      ridge=profile.ridge, home_field_prior=profile.home_field_prior, with_resume=False,
                      with_efficiency=False)
    return fitted.power, fitted.home_field


def conference_games(season: int, games: pd.DataFrame | None = None) -> pd.DataFrame:
    """Regular-season, non-tournament conference games, for standings."""
    g = games if games is not None else cbbd.games(season)
    return g[(g["season_type"] == "regular") & (g["game_type"] == "STD") & (g["conf1"] == g["conf2"]) & g["conf1"].notna()]


def standings(season: int, conference: str, games: pd.DataFrame | None = None) -> list[str]:
    """A conference's teams, best record first: the seeding both the placeholder and the simulation use."""
    g = conference_games(season, games)
    g = g[(g["conf1"] == conference)]
    wins: dict[str, int] = {}
    losses: dict[str, int] = {}
    for r in g.itertuples():
        for team, won in ((r.team1, r.win1), (r.team2, r.win2)):
            (wins if won == 1.0 else losses)[team] = (wins if won == 1.0 else losses).get(team, 0) + 1
    teams = set(wins) | set(losses)
    return sorted(teams, key=lambda t: (-(wins.get(t, 0) / max(wins.get(t, 0) + losses.get(t, 0), 1)), -wins.get(t, 0)))


def auto_bid_winners(season: int, conferences: list[str] | None = None) -> dict[str, str]:
    """Each conference's automatic bid: the team that won its tournament, from the games themselves."""
    conf_games = bb_bracket.conference_tournaments(season)
    wanted = conferences or (sorted(conf_games["conference"].unique()) if not conf_games.empty else [])
    out = {}
    for conf in wanted:
        champ = bb_bracket.champion(conf_games[conf_games["conference"] == conf])
        if champ:
            out[conf] = champ
    return out


def field(season: int) -> pd.DataFrame:
    """The actual NCAA field: one row per team, its best (lowest) seed, its region, and whether it got
    there as an automatic bid or an at-large.

    A team that played in more than one region across seasons isn't a concern here - one row per team per
    *this* season is all a single season's field needs, and every team appears in exactly one region.
    """
    ncaa = bb_bracket.ncaa_tournament(season)
    if ncaa.empty:
        return ncaa
    rows: dict[str, dict] = {}
    for r in ncaa.itertuples():
        for team, seed, conf, opp_conf in ((r.team1, r.seed1, r.conf1, r.conf2), (r.team2, r.seed2, r.conf2, r.conf1)):
            if pd.isna(seed):
                continue
            row = rows.setdefault(team, {"team": team, "seed": int(seed), "region": r.region, "conference": conf})
            row["seed"] = min(row["seed"], int(seed))                # a team's seed is fixed; a First Four loser's seed still counts
    autos = auto_bid_winners(season)
    winner_by_conf = {v: k for k, v in autos.items()}                # a conference sends at most one champion
    for team, row in rows.items():
        row["bidType"] = "auto" if winner_by_conf.get(team) == row["conference"] else "at-large"
    return pd.DataFrame(rows.values())

