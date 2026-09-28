"""How surprising a week's results were, scored against the model rather than the calendar.

A plain upset count treats a 5% underdog's win the same as a 45% underdog's - both are "an
upset" - which buries the thing worth measuring. Surprisal fixes that: a winner's pregame win
probability ``p`` costs ``-ln(p)``, so a 5% shock costs far more than a coin flip going the "wrong"
way. A week's total surprisal is compared against what the model itself expected (its entropy),
and the gap is standardised into a ``z`` score so weeks with different game counts and different
average favourite strength are still comparable. The published number is that ``z``'s percentile
against every other archived week - "wilder than 91% of weeks since 1978".

This only works if the probabilities being scored are honest. The model's own early-season lines
run overconfident (``scripts/backtest_priors.py``), and a systematically overconfident line makes
every early week look artificially chaotic - the model being wrong, not the world. ``fit_recalibration``
launders that out with a small per-week-of-season logistic recalibration, fit once on the full walk-
forward archive and cached in ``CALIBRATION_PATH`` (the same "fit offline, load read-only" pattern
``priors.py`` uses for ``prior_model.json``) rather than refit on every build. ``self_check`` is the
executable form of "check that it worked": a properly recalibrated archive has mean ``z`` near 0 and
standard deviation near 1, both overall and within each band.

1978-1980 (``history.BURN_IN_SEASONS``) are cold-started at a flat, uninformative prior, so their
probabilities are known to be too extreme in a way that has nothing to do with that era's football -
exactly the contamination the calibration guard exists to prevent. Every caller here (fitting,
self-check, ranking) is expected to filter those seasons out first; nothing in this module drops
them automatically, so a caller that forgets to filter fails loudly by producing visibly bad numbers
rather than silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CALIBRATION_PATH = ROOT / "data" / "chaos_calibration.json"

# Week-of-season bands for the §2 recalibration, matching the spec's "weeks 1-3, 4-7, 8+".
# A postseason block's week number (``cfbd.sequence()``'s "last regular week + 1") always lands
# in "late", which is exactly right - conference championship week and bowls are not early-season
# small-sample weeks.
_EARLY = range(1, 4)
_MID = range(4, 8)

MIN_GAMES = 10  # skip any week with fewer games than this - the spec's example is week 0
EPS = 1e-4  # clip probabilities away from the 0/1 edges, where ln blows up


def band_for_week(week: int) -> str:
    """Which recalibration/scoring band a week falls in: "early" (1-3), "mid" (4-7), else "late"."""
    if week in _EARLY:
        return "early"
    if week in _MID:
        return "mid"
    return "late"


def _clip(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def surprisal(p_winner) -> np.ndarray:
    """``-ln(p)`` for the probability the model gave the team that actually won."""
    return -np.log(_clip(p_winner))


def entropy(p) -> np.ndarray:
    """A game's expected surprisal: ``-[p ln p + (1-p) ln(1-p)]``. Symmetric in ``p``/``1-p``,
    so it does not matter whether ``p`` is the home probability or the winner's."""
    p = _clip(p)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def surprisal_variance(p) -> np.ndarray:
    """``p(1-p)*(ln(p/(1-p)))**2`` - also symmetric in ``p``/``1-p`` (squaring cancels the sign
    flip from swapping ``p`` for ``1-p``), so the home probability can be used directly."""
    p = _clip(p)
    return p * (1.0 - p) * np.log(p / (1.0 - p)) ** 2


def game_terms(p_home: float, home_won: bool) -> dict:
    """The per-game pieces ``week_score`` sums: surprisal and entropy from the winner's and the
    home side's probability respectively, this game's variance contribution, and the two
    probabilities a shock listing wants (the winner's, and the underdog's)."""
    p_home = float(p_home)
    p_winner = p_home if home_won else 1.0 - p_home
    return {
        "surprisal": float(surprisal(p_winner)),
        "entropy": float(entropy(p_home)),
        "variance": float(surprisal_variance(p_home)),
        "winnerProb": float(p_winner),
        "underdogProb": float(min(p_home, 1.0 - p_home)),
    }


def week_score(games: pd.DataFrame) -> dict:
    """Score one week of completed games.

    ``games`` needs columns ``game_id, home, away, home_win_prob`` (already recalibrated),
    ``home_won, home_score, away_score``. Returns games/upsets/expectedUpsets/z plus the 3
    biggest shocks (lowest winner probability). A week under ``MIN_GAMES`` is returned with
    ``z: None`` and ``note: "not_enough_games"`` rather than being scored - the caller still
    freezes this entry (it is a real, final answer: "not enough games to rate"), it just never
    enters a ranking population.
    """
    n = len(games)
    if n < MIN_GAMES:
        return {"games": n, "upsets": None, "expectedUpsets": None, "z": None,
                "shocks": [], "note": "not_enough_games"}

    p_home = games["home_win_prob"].to_numpy(float)
    home_won = games["home_won"].to_numpy(bool)
    p_winner = np.where(home_won, p_home, 1.0 - p_home)
    underdog_prob = np.minimum(p_home, 1.0 - p_home)
    upset = p_winner < 0.5

    s = surprisal(p_winner)
    h = entropy(p_home)
    v = surprisal_variance(p_home)
    sum_var = float(v.sum())
    z = float((s - h).sum() / np.sqrt(sum_var)) if sum_var > 0 else 0.0

    shock_order = np.argsort(p_winner)[:3]
    shocks = []
    for i in shock_order:
        row = games.iloc[int(i)]
        home_score, away_score = int(row["home_score"]), int(row["away_score"])
        if home_won[i]:
            winner, loser, winner_score, loser_score = row["home"], row["away"], home_score, away_score
        else:
            winner, loser, winner_score, loser_score = row["away"], row["home"], away_score, home_score
        shocks.append({
            "gameId": int(row["game_id"]), "winner": winner, "loser": loser,
            "pregameWinProb": float(p_winner[i]),
            "winnerScore": winner_score, "loserScore": loser_score,
            "homeScore": home_score, "awayScore": away_score,
        })

    return {
        "games": n,
        "upsets": int(upset.sum()),
        "expectedUpsets": float(underdog_prob.sum()),
        "z": z,
        "shocks": shocks,
    }


def _fit_logistic_2param(x: np.ndarray, y: np.ndarray, *, iterations: int = 50,
                          ridge: float = 1e-6) -> tuple[float, float]:
    """2-parameter logistic regression ``p = sigmoid(a*x + b)`` by Newton's method.

    Starting at ``a=1, b=0`` - "the raw probability is already calibrated" - so a band with
    plenty of games and nothing wrong with it converges in a step or two. ``ridge`` only guards
    the Hessian's conditioning on a degenerate band; every real band here has thousands of games,
    so it never meaningfully shrinks the fit.
    """
    a, b = 1.0, 0.0
    for _ in range(iterations):
        z = a * x + b
        p = 1.0 / (1.0 + np.exp(-z))
        w = p * (1.0 - p)
        grad = np.array([np.sum((y - p) * x), np.sum(y - p)])
        hess = np.array([[-np.sum(w * x * x) - ridge, -np.sum(w * x)],
                          [-np.sum(w * x), -np.sum(w) - ridge]])
        delta = np.linalg.solve(hess, -grad)
        a, b = a + delta[0], b + delta[1]
        if np.abs(delta).max() < 1e-10:
            break
    return float(a), float(b)


def fit_recalibration(games: pd.DataFrame) -> dict:
    """Fit the §2 per-band logistic recalibration: ``p' = sigmoid(a*logit(p) + b)``.

    ``games`` needs columns ``season, week, home_win_prob`` (raw, pre-recalibration walk-forward
    probability) and ``home_won``. The caller is responsible for filtering out burn-in seasons
    first (see the module docstring) - fitting on them would let three seasons of known-bad,
    cold-start probabilities distort the coefficients applied to the other 45.
    """
    p = _clip(games["home_win_prob"])
    logit_p = np.log(p / (1.0 - p))
    y = games["home_won"].to_numpy(float)
    band = games["week"].astype(int).map(band_for_week).to_numpy()

    bands = {}
    for name in ("early", "mid", "late"):
        mask = band == name
        a, b = _fit_logistic_2param(logit_p[mask], y[mask])
        bands[name] = {"a": a, "b": b, "n": int(mask.sum())}

    return {"bands": bands,
            "fittedThrough": {"season": int(games["season"].max()), "week": int(games["week"].max())}}


def apply_recalibration(p_home: float, week: int, calibration: dict | None) -> float:
    """Recalibrate one raw home win probability. Identity if ``calibration`` is absent (isolation:
    a build with no calibration file yet still scores weeks, just uncalibrated)."""
    if calibration is None:
        return float(p_home)
    params = calibration["bands"].get(band_for_week(week))
    if params is None:
        return float(p_home)
    p = _clip(p_home)
    logit_p = np.log(p / (1.0 - p))
    z = params["a"] * logit_p + params["b"]
    return float(1.0 / (1.0 + np.exp(-z)))


def load_calibration(path: Path = CALIBRATION_PATH) -> dict | None:
    """The fitted recalibration, or None if there is none yet to use."""
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_calibration(calibration: dict, path: Path = CALIBRATION_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n")


def self_check(scored_weeks: pd.DataFrame) -> dict:
    """§2.5's check that the recalibration worked: mean ``z`` near 0, std near 1, overall and
    within each week-of-season band. ``scored_weeks`` needs columns ``week, z, burnIn`` - burn-in
    rows are excluded here too, same reasoning as ``fit_recalibration``."""
    live = scored_weeks[~scored_weeks["burnIn"] & scored_weeks["z"].notna()]
    overall = {"mean": float(live["z"].mean()), "std": float(live["z"].std(ddof=1))}

    band = live["week"].astype(int).map(band_for_week)
    by_band = {}
    for name in ("early", "mid", "late"):
        sub = live.loc[band == name, "z"]
        by_band[name] = {"mean": float(sub.mean()) if len(sub) else None,
                          "std": float(sub.std(ddof=1)) if len(sub) > 1 else None}
    return {"overall": overall, "byBand": by_band}


def percentile_rank(z: float, population: list[float]) -> float | None:
    """The share of ``population`` strictly calmer than ``z`` - "wilder than N% of weeks".
    None if the population is empty (nothing to rank against yet)."""
    if not population:
        return None
    return float((np.asarray(population, dtype=float) < z).mean() * 100.0)
