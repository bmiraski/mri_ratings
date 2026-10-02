"""A small, deterministic season for tests that need a real solve but not real data.

It carries everything the decomposition has to get right: neutral-site games, an
outsider (non-anchor) opponent, the pooled non-FBS team, and a team that has a
prior but has played nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mri.ratings import mri2

FBS = [f"Team {c}" for c in "ABCDEFGHIJKL"]
OUTSIDERS = ["Outsider One", "Outsider Two"]
IDLE = "Team Idle"  # has a prior, plays no games: w = 0


def synthetic_games(seed: int = 7, rounds: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    strength = {t: float(rng.normal(0, 9)) for t in FBS}
    strength.update({o: -24.0 for o in OUTSIDERS})
    strength[mri2.POOLED_FCS] = -26.0
    rows = []
    for rnd in range(rounds):
        order = list(rng.permutation(FBS))
        for i in range(0, len(order), 2):
            away, home = order[i], order[i + 1]
            rows.append((away, home, bool(rng.random() < 0.2)))
        # one outsider visit per round, the first via the pooled stand-in
        rows.append(((OUTSIDERS + [mri2.POOLED_FCS])[rnd % 3], FBS[rnd % len(FBS)], False))
    games = []
    for away, home, neutral in rows:
        edge = strength[home] - strength[away] + (0.0 if neutral else 3.0) + rng.normal(0, 13)
        margin = int(round(edge))
        p_home = max(margin, 0) + 14
        p_away = max(-margin, 0) + 14
        games.append(
            {"team1": away, "team2": home, "pts1": p_away, "pts2": p_home,
             "win1": float(p_away > p_home), "win2": float(p_home > p_away), "neutral": neutral}
        )
    prior = pd.Series({t: strength.get(t, 0.0) * 0.6 + float(rng.normal(0, 3)) for t in FBS + [IDLE]})
    return pd.DataFrame(games), prior


def synthetic_payload(rounds: int = 5, sport: str = "football") -> dict:
    """A minimal site payload whose team details carry a real decomposition.

    Built the way the site builds it: fit the games, then ``decomposition.attach``
    onto per-team details, with Power rounded to two decimals as the payload does.
    """
    from mri.export import decomposition

    games, prior = synthetic_games(rounds=rounds)
    model = mri2.fit(games, prior=prior, neutral=games["neutral"], anchor_teams=FBS)
    games = games.assign(game_id=range(1, len(games) + 1))
    # `fit` was run on the frame without ids; the ids ride along for the join only.
    shares = decomposition.core.decompose(model, games)

    ranked = [t for t in model.power.sort_values(ascending=False).index if t in FBS]
    teams = [
        {"team": t, "rank": i + 1, "power": round(float(model.power[t]), 2),
         "resume": 0.0, "resumeRank": i + 1, "wins": 0, "losses": 0, "confWins": 0, "confLosses": 0,
         "conference": "Test", "color": "#336699", "altColor": "#999999", "logo": None,
         "abbreviation": None, "previousRank": None, "movement": 0, "trajectory": [], "rankHistory": [],
         "classicRank": None, "classicRating": None, "sosRank": None, "roster": None}
        for i, t in enumerate(ranked)
    ]
    details = {t["team"]: {"played": [], "upcoming": []} for t in teams}
    for k, g in games.iterrows():
        for team, opp, site, sign in ((g.team2, g.team1, "n" if g.neutral else "vs", 1),
                                      (g.team1, g.team2, "n" if g.neutral else "at", -1)):
            if team not in details:
                continue
            margin = int(sign * (g.pts2 - g.pts1))
            details[team]["played"].append({
                "week": k // 6 + 1, "opponent": opp, "opponentRank": None, "opponentPower": 0.0,
                "site": site, "expected": 0.0, "scored": int(g.pts2 if sign > 0 else g.pts1),
                "allowed": int(g.pts1 if sign > 0 else g.pts2), "won": margin > 0, "margin": margin,
                "performance": float(margin), "_gid": int(g.game_id),
            })
    for d in details.values():
        d.update({"remainingDifficulty": None, "playedDifficulty": 0.0, "bestWin": None, "worstLoss": None})
    power = {t["team"]: t["power"] for t in teams}
    decomposition.attach(details, shares, power)
    return {"season": 2026, "week": rounds, "weeks": list(range(1, rounds + 1)), "sport": sport,
            "homeField": round(model.home_field, 2), "gamesRated": len(games), "teams": teams, "details": details, "conferences": [],
            "generated": "2026-10-01T00:00:00", "sports": [sport]}
