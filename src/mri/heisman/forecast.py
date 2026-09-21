"""Heisman odds for the season in progress.

Play the rest of the season thousands of times (the same engine as the season
simulation, watched run by run), project every candidate's finished stat line from
what he has done and what history says happens next, score the field with the
final-vote model, and count who wins. Each run is a whole imagined season: a team
that loses twice in December is a different Heisman argument from one that goes
unbeaten, and the run knows which it is.

What is treated as independent, and should be said: a player's output and his team's
results are drawn separately. A quarterback's big games and his team's wins really are
linked, and a model that ties them would say the best teams' stars are a little more
likely still. The backtest is where that would show up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..sim import season
from . import features, project


def simulate_teams(teams: list[dict], schedule: pd.DataFrame, *, home_field: float, sims: int, seed: int = season.DEFAULT_SEED,
                   rules: dict | None = None, championships: dict | None = None) -> dict:
    """Play out the rest of the season and keep each run's finish for every team."""
    chunks: list[dict] = []
    season.simulate(teams, schedule, home_field=home_field, sims=sims, seed=seed, rules=rules, championships=championships,
                    observe=lambda c: chunks.append({k: v for k, v in c.items()}))
    names = [t["team"] for t in teams]
    return {"names": names, "index": {n: i for i, n in enumerate(names)},
            "rank": np.concatenate([c["rank"] for c in chunks]).astype(float),
            "made_cg": np.concatenate([c["made_cg"] for c in chunks]).astype(float)}


def remaining_games(schedule: pd.DataFrame, names: list[str], last_regular_week: int = 13) -> dict[str, int]:
    """Regular-season games each team still has to play, championship game aside."""
    left = schedule[(~schedule["played"]) & (schedule["week"] <= last_regular_week)]
    counts: dict[str, int] = {}
    for team in list(left["team1"]) + list(left["team2"]):
        counts[team] = counts.get(team, 0) + 1
    return {n: counts.get(n, 0) for n in names}


def odds(pool: pd.DataFrame, runs: dict, left: dict[str, int], ratio_pool: np.ndarray, model: dict, *,
         seed: int = 2026, finalist_draws: int = 3) -> pd.DataFrame:
    """Each candidate's chance to win, and to finish in the top four, across the simulated seasons.

    ``pool`` needs player, team, group_code, off_score (season to date), team_games and prev_finalist.
    """
    rng = np.random.default_rng(seed)
    S = runs["rank"].shape[0]
    G = len(pool)
    team_idx = np.array([runs["index"][t] for t in pool["team"]])
    cols = [features.NAMES.index(n) for n in model["features"]]
    beta = np.asarray(model["coefficients"], dtype=float)

    base_left = np.array([left.get(t, 0) for t in pool["team"]], dtype=float)
    remaining = base_left[None, :] + runs["made_cg"][:, team_idx]                          # (S, G)
    draws = rng.choice(ratio_pool, size=(S, G))
    final = project.project(pool["off_score"].to_numpy(), pool["team_games"].to_numpy(), pool["group_code"].to_numpy(),
                            remaining, draws)
    team_rank = runs["rank"][:, team_idx]
    X = features.matrix(final, pool["group_code"].to_numpy(), team_rank, 0.0, pool["prev_finalist"].to_numpy())[..., cols]
    u = X @ beta
    u -= u.max(axis=1, keepdims=True)
    p = np.exp(u)
    p /= p.sum(axis=1, keepdims=True)

    top4 = np.zeros(G)
    for _ in range(finalist_draws):
        gumbel = -np.log(-np.log(rng.random((S, G))))
        order = np.argsort(-(u + gumbel), axis=1)[:, :4]
        top4 += np.bincount(order.ravel(), minlength=G)
    out = pool[["player", "team", "group"]].copy()
    out["win"] = p.mean(axis=0)
    out["finalist"] = top4 / (S * finalist_draws)
    out["projected"] = final.mean(axis=0)
    return out.sort_values("win", ascending=False).reset_index(drop=True)
