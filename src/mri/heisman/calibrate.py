"""Turning a forecast's probabilities into ones that have come true that often.

The finalist odds run high for everyone below the favorites: a candidate given 9% reached
the ceremony about 5% of the time. Some of that is structural - the four places the forecast
hands out include defenders, who are not in the pool, so the players in it are competing for
fewer than four - and some is the model's. Either way the cure is the same and it is
empirical: bin the backtest's predictions, see how often each bin came true, and draw the curve
through them. Monotone, so a higher forecast never maps to a lower chance.
"""

from __future__ import annotations

import numpy as np

EDGES = (0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.0001)


def fit_map(predicted, observed, edges=EDGES, min_count: int = 40) -> list[list[float]]:
    """Knots (predicted, observed) of a monotone curve from 0 to 1 through the bins' averages."""
    predicted, observed = np.asarray(predicted, float), np.asarray(observed, float)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (predicted >= lo) & (predicted < hi)
        if m.sum() >= min_count:
            bins.append([predicted[m].mean(), observed[m].mean(), float(m.sum())])
    # Pool adjacent violators: merge neighbours until the observed rate never falls.
    merged: list[list[float]] = []
    for x, y, n in bins:
        merged.append([x, y, n])
        while len(merged) > 1 and merged[-2][1] > merged[-1][1]:
            (x1, y1, n1), (x2, y2, n2) = merged[-2], merged[-1]
            merged[-2:] = [[(x1 * n1 + x2 * n2) / (n1 + n2), (y1 * n1 + y2 * n2) / (n1 + n2), n1 + n2]]
    return [[0.0, 0.0]] + [[round(x, 4), round(y, 4)] for x, y, _ in merged] + [[1.0, 1.0]]


def apply(knots, p) -> np.ndarray:
    """Read probabilities off the curve, interpolating between its knots."""
    knots = np.asarray(knots, float)
    return np.interp(np.asarray(p, float), knots[:, 0], knots[:, 1])
