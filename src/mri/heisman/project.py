"""From a player's season so far to a distribution over his finished season.

A quarterback with 1,200 yards through four games is not on pace for 4,800, and
the reasons are all things a projection has to price: the hot start regresses, an
injury ends a season, a blowout sits the starter, a bye week is not a game. Rather
than model each, the projection asks history what actually happened.

Take every candidate at the same point of every other season. Project him as
his rate so far (shrunk toward his position group's, in proportion to how few games
that rate rests on) times the games his team has left. Divide what he really added
by that projection. Those ratios - mostly a little under one, sometimes zero when a
player was hurt, occasionally above one - are the distribution. A candidate is then
projected by drawing ratios from it. Injuries, regression and garbage time are in it
because they were in the seasons it came from, in the proportions they came.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SHRINK_GAMES = 2.0          # a rate is worth this many games of the group's average
RATIO_CAP = 3.0             # a stat line more than three times its projection is a data problem
SNAPSHOT_WEEKS = (3, 5, 7, 9, 11, 13)


NO_LAST = {"last": 0.0, "shrink": SHRINK_GAMES}


def group_rates(current: np.ndarray, games: np.ndarray, group: np.ndarray, last: np.ndarray | None = None,
                variant: dict | None = None) -> np.ndarray:
    """Each player's shrunk per-game rate.

    A rate is pulled toward what a player of his kind does. Early in a season that is the position group's
    average; if he played last season it can instead be partly his own rate then, since a quarterback who
    threw for 4,000 yards last year is better evidence about his September than three games are.
    ``variant`` says how much (``last``, a weight from 0 to 1) and how many games a rate is worth
    (``shrink``). ``last`` is his per-game production last season, NaN where he has none.
    """
    v = {**NO_LAST, **(variant or {})}
    per_game = current / np.maximum(games, 1.0)
    prior = np.empty_like(per_game)
    for g in np.unique(group):
        prior[group == g] = per_game[group == g].mean()
    if last is not None and v["last"] > 0:
        last = np.asarray(last, dtype=float)
        prior = np.where(np.isnan(last), prior, v["last"] * last + (1.0 - v["last"]) * prior)
    weight = games / (games + v["shrink"])
    return weight * per_game + (1.0 - weight) * prior


def ratios(current, games, group, remaining, final, last=None, variant=None) -> np.ndarray:
    """What history did to the projection of each candidate in one snapshot."""
    current, games, remaining, final = (np.asarray(a, dtype=float) for a in (current, games, remaining, final))
    projected = remaining * group_rates(current, games, np.asarray(group), last, variant)
    ok = (remaining >= 1) & (projected > 0)
    return np.clip((final - current)[ok] / projected[ok], 0.0, RATIO_CAP)


def project(current, games, group, remaining, ratio_draws, last=None, variant=None) -> np.ndarray:
    """Finished-season production for every draw: (draws, candidates).

    ``remaining`` may differ by draw (a team plays a championship game in some of the
    simulated seasons and not in others), so it is (draws, candidates) or broadcasts to it.
    """
    rate = group_rates(np.asarray(current, dtype=float), np.asarray(games, dtype=float), np.asarray(group), last, variant)
    return np.asarray(current, dtype=float) + np.asarray(remaining, dtype=float) * rate * ratio_draws
