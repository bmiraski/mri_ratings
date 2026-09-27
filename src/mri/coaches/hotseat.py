"""The plan's §4 hot-seat model: P(fired) at the end of a season, from information
available at that season's end.

The label is 1 if a season is a confirmed departure (mri.coaches.departures
- someone else really did take over the following season) classified
``fired_or_pushed_out``, 0 for every other coach-season: still employed
next year, or a departure classified ``moved_up``/``retired_or_other``.
That is what makes this "P(fired)" and not "P(this is a departure)" - most
training rows are simply a coach who kept their job.

A ridge-penalized logistic regression, hand-rolled via
``scipy.optimize.minimize`` with its own gradient - the same shape as
``mri.gameday.model``'s conditional logit and ``mri.heisman.final_model``'s,
just a binary logit instead of a softmax. About 350 positives over twenty
-odd seasons is "keep it small," in the plan's own words, and this project
has never used scikit-learn; there is no reason to start for something
this size.

Known blind spots - print these on the eventual page, not just here:

* Buyouts are the biggest confounder and are not in any data source this
  pipeline has access to. A coach a program can't afford to fire stays
  employed on outcomes this model would call a firing offense elsewhere.
* Athletic-director turnover. A new AD often means a new coach regardless
  of the old one's record.
* Scandals. Recruiting violations, off-field conduct - the record on paper
  can be excellent and the coach still gone.
* A coach leaving on their own for a better job is a real, common outcome
  this model was never asked to predict - that is what the ``moved_up``
  label already separates out, not a defect in P(fired) missing it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata

from ..sim.season import POWER_FOUR
from .season import stint_start as _stint_start

FIRST_LABELED_SEASON = 2004  # matches mri.coaches.departures's own cutoff
MIN_TRAIN_SEASONS = 6        # seasons of history before the first held-out evaluation
DEFAULT_PENALTY = 4.0

BASELINE_COLUMNS = ["win_pct", "year1", "year2", "year3"]
FULL_COLUMNS = [
    "vs_par", "vs_par_lag", "added", "vs_inherited", "win_pct", "conf_win_pct",
    "year1", "year2", "year3", "power_conference", "vs_talent",
]

TALENT_MODEL_PATH = Path(__file__).resolve().parents[3] / "data" / "talent_model.json"


def full_columns(talent_model_path: Path = TALENT_MODEL_PATH) -> list[str]:
    """``FULL_COLUMNS``, dropping ``vs_talent`` if the §3 gate didn't actually pass.

    Checked at runtime rather than assumed, so a future re-fit of the talent
    gate that comes back negative doesn't silently keep training on a
    feature the plan said to drop.
    """
    if talent_model_path.exists() and json.loads(talent_model_path.read_text()).get("passed"):
        return list(FULL_COLUMNS)
    return [c for c in FULL_COLUMNS if c != "vs_talent"]


def dataset(coach_season: pd.DataFrame, departures: pd.DataFrame) -> pd.DataFrame:
    """One row per training example: every feature, plus the label.

    Every non-interim coach-season from ``FIRST_LABELED_SEASON`` through the
    second-to-last season on record is a row (the last season's outcome
    isn't confirmed yet - the same reasoning ``mri.coaches.departures.
    eligible_departures`` already uses for "last season," extended to every
    season here). Rows missing ``vs_par``, ``vs_inherited`` or ``added`` are
    dropped - there simply isn't enough program history to say anything,
    the same "null, not a guess" rule phase 2's metrics already follow.
    ``vs_talent`` is the one feature that's imputed rather than dropped-for
    (0, i.e. no signal) when missing, since the plan marks it a bonus
    feature conditional on the talent gate, not a core one - requiring it
    would silently throw out every pre-2015 season.
    """
    last_confirmed = int(coach_season["season"].max()) - 1
    rows = coach_season[
        (~coach_season["interim"])
        & (coach_season["season"] >= FIRST_LABELED_SEASON)
        & (coach_season["season"] <= last_confirmed)
        & (coach_season["games"] > 0)
    ].copy()

    # stint_start needs the coach's full history at the school, not just the
    # post-FIRST_LABELED_SEASON slice, so it's computed against the unfiltered table.
    rows["stint_start"] = _stint_start(coach_season).reindex(rows.index)
    rows["tenure_year"] = rows["season"] - rows["stint_start"] + 1
    rows["year1"] = (rows["tenure_year"] == 1).astype(float)
    rows["year2"] = (rows["tenure_year"] == 2).astype(float)
    rows["year3"] = (rows["tenure_year"] == 3).astype(float)

    rows["win_pct"] = rows["wins"] / rows["games"]
    conf_games = rows["conf_wins"] + rows["conf_losses"]
    rows["conf_win_pct"] = np.where(conf_games > 0, rows["conf_wins"] / conf_games.replace(0, np.nan), rows["win_pct"])

    rows["power_conference"] = rows["conference"].isin(POWER_FOUR).astype(float)

    by_key = coach_season.set_index(["coach_id", "school", "season"])["vs_par"]
    lag_index = pd.MultiIndex.from_arrays([rows["coach_id"], rows["school"], rows["season"] - 1])
    rows["vs_par_lag"] = by_key.reindex(lag_index).fillna(0.0).to_numpy()

    rows["vs_talent"] = rows["vs_talent"].fillna(0.0)

    key_cols = ["coach_id", "school", "season"]
    dep = departures.set_index(key_cols)["label"] if not departures.empty else pd.Series(dtype=object)
    matched = dep.reindex(pd.MultiIndex.from_frame(rows[key_cols]))
    rows["label"] = (matched.to_numpy() == "fired_or_pushed_out").astype(float)

    rows = rows.dropna(subset=["vs_par", "vs_inherited", "added"])
    return rows.reset_index(drop=True)


def design(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """The design matrix for ``columns``, with an intercept column appended last."""
    X = np.column_stack([frame[c].to_numpy(dtype=float) for c in columns]) if columns else np.empty((len(frame), 0))
    return np.column_stack([X, np.ones(len(frame))])


def _nll(beta: np.ndarray, X: np.ndarray, y: np.ndarray, penalty: float):
    z = X @ beta
    p = 1.0 / (1.0 + np.exp(-z))
    eps = 1e-12
    nll = -float(np.sum(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)))
    reg = np.copy(beta)
    reg[-1] = 0.0  # never penalize the intercept
    nll += penalty * float(reg @ reg)
    grad = X.T @ (p - y) + 2.0 * penalty * reg
    return nll, grad


def fit(frame: pd.DataFrame, columns: list[str], *, penalty: float = DEFAULT_PENALTY) -> np.ndarray:
    """Ridge-penalized logistic regression: coefficients for ``columns``, intercept last."""
    X = design(frame, columns)
    y = frame["label"].to_numpy(dtype=float)
    result = minimize(_nll, np.zeros(X.shape[1]), args=(X, y, penalty), jac=True, method="L-BFGS-B")
    return result.x


def predict(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(X @ beta)))


def walk_forward(
    data: pd.DataFrame, columns: list[str], *, min_train_seasons: int = MIN_TRAIN_SEASONS, penalty: float = DEFAULT_PENALTY
) -> pd.DataFrame:
    """Out-of-sample P(fired) for every season past the first ``min_train_seasons``,
    fit on strictly earlier seasons each time - a held-out season's own rows are
    never in its own training fold."""
    seasons = sorted(data["season"].unique())
    rows = []
    for season_ in seasons[min_train_seasons:]:
        train = data[data["season"] < season_]
        test = data[data["season"] == season_]
        if train.empty or test.empty:
            continue
        beta = fit(train, columns, penalty=penalty)
        p = predict(beta, design(test, columns))
        rows.append(pd.DataFrame({
            "season": season_, "coach_id": test["coach_id"].to_numpy(), "school": test["school"].to_numpy(),
            "label": test["label"].to_numpy(), "p": p,
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["season", "coach_id", "school", "label", "p"]
    )


def auc(y, p) -> float:
    """Rank-based AUC (the Mann-Whitney U statistic, normalized) - no sklearn needed."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(p)
    r_pos = ranks[y == 1].sum()
    u = r_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def brier(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    """Predicted vs. actual fired rate by decile, pooled across every held-out season -
    any one season has too few positives to bucket on its own."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    rows = []
    for idx in np.array_split(order, bins):
        if len(idx) == 0:
            continue
        rows.append({"n": len(idx), "predicted": float(p[idx].mean()), "actual": float(y[idx].mean())})
    return pd.DataFrame(rows)


def evaluate(data: pd.DataFrame, *, full_cols: list[str] | None = None) -> dict:
    """Walk-forward the full model and the baseline, and decide whether the full model ships.

    Ship bar, exactly as the plan states it: beats the baseline on Brier
    score in most held-out seasons. If it doesn't, ``passed`` is False and
    the caller's job is to say so plainly, not to ship a probability column
    anyway.
    """
    full_cols = full_cols if full_cols is not None else full_columns()
    full = walk_forward(data, full_cols)
    base = walk_forward(data, BASELINE_COLUMNS)
    merged = full.merge(base, on=["season", "coach_id", "school", "label"], suffixes=("_full", "_base"))

    per_season = []
    for season_ in sorted(merged["season"].unique()):
        sub = merged[merged["season"] == season_]
        b_full, b_base = brier(sub["label"], sub["p_full"]), brier(sub["label"], sub["p_base"])
        per_season.append({
            "season": int(season_), "n": int(len(sub)), "positives": int(sub["label"].sum()),
            "brierFull": round(b_full, 4), "brierBaseline": round(b_base, 4), "fullBetter": bool(b_full < b_base),
        })

    wins = sum(1 for s in per_season if s["fullBetter"])
    passed = wins > len(per_season) / 2 if per_season else False

    return {
        "features": full_cols,
        "minTrainSeasons": MIN_TRAIN_SEASONS,
        "penalty": DEFAULT_PENALTY,
        "trainSeasons": [int(data["season"].min()), int(data["season"].max())],
        "teamSeasons": int(len(data)),
        "positives": int(data["label"].sum()),
        "perSeason": per_season,
        "seasonsFullBetter": wins,
        "seasonsTotal": len(per_season),
        "aucFull": round(auc(merged["label"], merged["p_full"]), 4),
        "aucBaseline": round(auc(merged["label"], merged["p_base"]), 4),
        "brierFull": round(brier(merged["label"], merged["p_full"]), 4),
        "brierBaseline": round(brier(merged["label"], merged["p_base"]), 4),
        "calibration": calibration_table(merged["label"], merged["p_full"]).to_dict(orient="records"),
        "passed": bool(passed),
        "coefficientsFull": dict(zip(full_cols + ["intercept"], (round(float(b), 4) for b in fit(data, full_cols)))),
        "coefficientsBaseline": dict(
            zip(BASELINE_COLUMNS + ["intercept"], (round(float(b), 4) for b in fit(data, BASELINE_COLUMNS)))
        ),
    }
