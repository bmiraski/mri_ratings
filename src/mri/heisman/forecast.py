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

MIN_RUNS = 30          # fewer simulated seasons than this and a conditional chance is noise


def simulate_teams(teams: list[dict], schedule: pd.DataFrame, *, home_field: float, sims: int, seed: int = season.DEFAULT_SEED,
                   rules: dict | None = None, championships: dict | None = None) -> dict:
    """Play out the rest of the season and keep each run's finish for every team."""
    chunks: list[dict] = []
    season.simulate(teams, schedule, home_field=home_field, sims=sims, seed=seed, rules=rules, championships=championships,
                    observe=lambda c: chunks.append({k: v for k, v in c.items()}))
    names = [t["team"] for t in teams]
    return {"names": names, "index": {n: i for i, n in enumerate(names)},
            "rank": np.concatenate([c["rank"] for c in chunks]).astype(float),
            "made_cg": np.concatenate([c["made_cg"] for c in chunks]).astype(float),
            "win_z": _standardize(np.concatenate([c["wins"] for c in chunks]).astype(float))}


def _standardize(wins: np.ndarray) -> np.ndarray:
    """How good a season each simulated one was for each team, against that team's own other simulated seasons."""
    spread = wins.std(axis=0, keepdims=True)
    return np.where(spread > 0, (wins - wins.mean(axis=0, keepdims=True)) / np.where(spread > 0, spread, 1.0), 0.0)


def remaining_games(schedule: pd.DataFrame, names: list[str], last_regular_week: int = 13) -> dict[str, int]:
    """Regular-season games each team still has to play, championship game aside."""
    left = schedule[(~schedule["played"]) & (schedule["week"] <= last_regular_week)]
    counts: dict[str, int] = {}
    for team in list(left["team1"]) + list(left["team2"]):
        counts[team] = counts.get(team, 0) + 1
    return {n: counts.get(n, 0) for n in names}


def draws_for(ratio_pool: np.ndarray, shape: tuple[int, int], rng: np.random.Generator, *, link: float = 0.0,
              team_z: np.ndarray | None = None) -> np.ndarray:
    """Projection ratios for every simulated season and candidate.

    Independent draws from history's ratios, unless ``link`` (a correlation) says a player's season
    and his team's move together: then each draw is the ratio at the quantile a shared normal score
    picks out, the score being part his team's simulated fortunes (a good team's quarterback has
    the more points to score) and part his own luck.
    """
    if link <= 0 or team_z is None:
        return rng.choice(ratio_pool, size=shape)
    from scipy.special import ndtr

    score = np.sqrt(link) * team_z + np.sqrt(1.0 - link) * rng.standard_normal(shape)
    ordered = np.sort(ratio_pool)
    return ordered[np.minimum((ndtr(score) * len(ordered)).astype(int), len(ordered) - 1)]


def odds(pool: pd.DataFrame, runs: dict, left: dict[str, int], ratio_pool: np.ndarray, model: dict, *,
         seed: int = 2026, finalist_draws: int = 3, link: float = 0.0, variant: dict | None = None) -> pd.DataFrame:
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
    draws = draws_for(ratio_pool, (S, G), rng, link=link, team_z=runs["win_z"][:, team_idx])
    last = pool["last_rate"].to_numpy() if "last_rate" in pool else None
    final = project.project(pool["off_score"].to_numpy(), pool["team_games"].to_numpy(), pool["group_code"].to_numpy(),
                            remaining, draws, last, variant)
    team_rank = runs["rank"][:, team_idx]
    X = features.matrix(final, pool["group_code"].to_numpy(), team_rank, 0.0, pool["prev_finalist"].to_numpy())[..., cols]
    u = X @ beta
    u -= u.max(axis=1, keepdims=True)
    p = np.exp(u)
    p /= p.sum(axis=1, keepdims=True)

    reached = np.zeros(G)
    for _ in range(finalist_draws):
        gumbel = -np.log(-np.log(rng.random((S, G))))
        order = np.argsort(-(u + gumbel), axis=1)[:, :4]
        reached += np.bincount(order.ravel(), minlength=G)
    out = pool[["player", "team", "group"]].copy()
    out["win"] = p.mean(axis=0)
    out["finalist"] = reached / (S * finalist_draws)
    out["projected"] = final.mean(axis=0)

    # What has to happen: how his chances differ between the seasons in which his team finishes among the
    # top four and the ones in which it does not. Left empty when a case is too rare to say anything about.
    elite = team_rank <= 4
    with_, without = elite.sum(axis=0), (~elite).sum(axis=0)
    out["team_top4"] = elite.mean(axis=0)
    out["win_if_top4"] = np.where(with_ >= MIN_RUNS, (p * elite).sum(axis=0) / np.maximum(with_, 1), np.nan)
    out["win_if_not"] = np.where(without >= MIN_RUNS, (p * ~elite).sum(axis=0) / np.maximum(without, 1), np.nan)
    return out.sort_values("win", ascending=False).reset_index(drop=True)
