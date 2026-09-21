"""What a Heisman voter's ballot looks like, as numbers.

Two rules govern the design, and both come from how few Heismans there are.

*Everything is relative to the field that year.* A 4,000-yard season wins in one
year and is fourth in the next, so a player is described by where he stands among
the season's candidates - his rank, and how far above the field's average he is -
and never by a raw total. The model then sees the same kind of number in 2013 and
in 2026, however the game changes.

*Very few features.* Fourteen usable winners cannot support a long list. What is
left is what voters say they weigh and what the record shows they do: how much the
player produced, how good and how unbeaten his team was, his position, and whether
he was already a finalist last year.

Everything works on arrays with a run dimension in front, so the same code scores
one finished season and ten thousand simulated ones.
"""

from __future__ import annotations

import numpy as np

NAMES = (
    "prod_rank",        # log of rank by production among the season's candidates
    "group_rank",       # log of rank within his own position group
    "prod_z",           # how far above the field's average he is, in standard deviations
    "team_rank",        # log of his team's rank among all FBS teams, résumé-heavy
    "losses",           # his team's losses
    "unbeaten",         # his team has not lost
    "is_rb",
    "is_receiver",
    "prev_finalist",    # was a Heisman finalist last season
)
GROUPS = {"QB": 0, "RB": 1, "REC": 2}


def _rank_descending(x: np.ndarray) -> np.ndarray:
    """1 = largest, along the last axis."""
    order = np.argsort(-x, axis=-1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(1, x.shape[-1] + 1).reshape((1,) * (x.ndim - 1) + (-1,)), axis=-1)
    return rank


def matrix(score, group, team_rank, losses, prev_finalist) -> np.ndarray:
    """Features for every candidate.

    ``score`` is production, shape (..., G); ``group`` (G,) holds 0/1/2 for QB/RB/REC;
    ``team_rank`` and ``losses`` broadcast against ``score``; ``prev_finalist`` is (G,).
    Returns (..., G, len(NAMES)).
    """
    score = np.asarray(score, dtype=float)
    group = np.asarray(group)
    shape = score.shape
    prod_rank = _rank_descending(score)
    group_rank = np.empty(shape, dtype=float)
    for g in np.unique(group):
        cols = np.flatnonzero(group == g)
        group_rank[..., cols] = _rank_descending(score[..., cols])
    spread = score.std(axis=-1, keepdims=True)
    z = (score - score.mean(axis=-1, keepdims=True)) / np.where(spread > 0, spread, 1.0)
    team_rank = np.broadcast_to(np.asarray(team_rank, dtype=float), shape)
    losses = np.broadcast_to(np.asarray(losses, dtype=float), shape)
    columns = (
        np.log(prod_rank),
        np.log(group_rank),
        z,
        np.log(np.maximum(team_rank, 1.0)),
        np.minimum(losses, 6.0),
        (losses == 0).astype(float),
        np.broadcast_to((group == GROUPS["RB"]).astype(float), shape),
        np.broadcast_to((group == GROUPS["REC"]).astype(float), shape),
        np.broadcast_to(np.asarray(prev_finalist, dtype=float), shape),
    )
    return np.stack(columns, axis=-1)
