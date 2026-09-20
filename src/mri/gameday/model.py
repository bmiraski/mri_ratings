"""A conditional logit: given the week's games, which one does GameDay pick?

Each game gets a score - a weighted sum of its features - and the probability of
being chosen is that score's share of the week's total, the same softmax that
underlies a multinomial choice. It is the right model for the question because
GameDay picks exactly one game a week, so what matters is how a game compares with
the others that week, not how big it looks in absolute terms.

Ridge-penalised, because a hundred and thirty weeks is not many for fourteen
features, with the penalty chosen by leaving whole seasons out.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _nll(beta, sets, penalty):
    total = penalty * float(beta @ beta)
    grad = 2.0 * penalty * beta
    for X, pick in sets:
        u = X @ beta
        u -= u.max()
        p = np.exp(u)
        p /= p.sum()
        total -= np.log(max(p[pick], 1e-12))
        grad += X.T @ p - X[pick]
    return total, grad


def fit(sets: list[tuple[np.ndarray, int]], penalty: float = 1.0) -> np.ndarray:
    n = sets[0][0].shape[1]
    mean = np.mean([X.mean(axis=0) for X, _ in sets], axis=0)
    res = minimize(_nll, np.zeros(n), args=(sets, penalty), jac=True, method="L-BFGS-B")
    return res.x


def probabilities(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    u = X @ beta
    u = u - u.max(axis=-1, keepdims=True)
    p = np.exp(u)
    return p / p.sum(axis=-1, keepdims=True)


def score(beta: np.ndarray, sets) -> dict:
    """Top-1 and top-3 hit rates, and the mean log-probability of the actual pick."""
    top1 = top3 = 0
    log = 0.0
    base = 0.0
    for X, pick in sets:
        p = probabilities(beta, X)
        order = np.argsort(-p)
        top1 += int(order[0] == pick)
        top3 += int(pick in order[:3])
        log += np.log(max(p[pick], 1e-12))
        base += np.log(1.0 / len(p))
    n = len(sets)
    return {"weeks": n, "top1": top1 / n, "top3": top3 / n, "logLoss": -log / n, "baseLogLoss": -base / n}
