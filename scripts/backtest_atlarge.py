"""Fit the at-large and seeding model, and grade it leave-one-season-out.

Two questions, and the ground truth answers both exactly - a real 68-team field with
real seed numbers each season, not a proxy for one:

1. Given the actual automatic bid winners, does ranking the rest of Division I by
   composite score reproduce the actual at-large field?
2. Applied to the whole field (auto and at-large together), how close does the
   composite score's rank order come to the real seed line?

Each season's fit uses coefficients learned from the *other* thirteen, exactly as
the football and Heisman committee proxies are graded.

Run:  PYTHONPATH=src python3 scripts/backtest_atlarge.py
Writes data/atlarge_model.json and site/data/atlarge_backtest.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.bracket import atlarge  # noqa: E402

RIDGE = 3.0


def load() -> pd.DataFrame:
    t = pd.read_parquet(ROOT / "data" / "parquet" / "atlarge_history.parquet")
    return t[t["conference"].notna()]                    # a team the registry can't place isn't scoreable


def loso(table: pd.DataFrame) -> dict:
    seasons = sorted(table["season"].unique())
    seed_errors, ranks, selection_hits, selection_totals = [], [], [], []
    per_season = {}
    for season in seasons:
        train = [table[table["season"] == s][table[table["season"] == s]["seed"].notna()] for s in seasons if s != season]
        beta = atlarge.fit(train, ridge=RIDGE)

        test_all = table[table["season"] == season]
        field = test_all[test_all["seed"].notna()]
        if field.empty:
            continue
        auto_bids = dict(zip(field[field["bidType"] == "auto"]["conference"], field[field["bidType"] == "auto"]["team"]))
        projected = atlarge.select_and_seed(test_all, beta, auto_bids=auto_bids, field_size=len(field))

        actual_at_large = set(field[field["bidType"] == "at-large"]["team"])
        projected_at_large = set(projected[projected["bidType"] == "at-large"]["team"])
        hit = len(actual_at_large & projected_at_large)
        selection_hits.append(hit)
        selection_totals.append(len(actual_at_large))

        # The seed line's own accuracy is a different question from selection, and asked of every
        # real field team directly - a team the model itself would not have picked as an at-large
        # still has a real seed to compare a score against; restricting to the model's own selection
        # would silently drop every year's real surprises instead of grading against them.
        field_scores = atlarge.score(field, beta)
        field_rank_position = field_scores.rank(method="min")
        projected_seed = np.minimum(np.ceil(field_rank_position / (len(field) / 16)).astype(int), 16)
        err = (projected_seed - field["seed"]).abs()
        seed_errors.extend(err.tolist())
        rho = spearmanr(field["seed"], field_scores).statistic     # both "lower is better": should correlate positively
        ranks.append(rho)
        per_season[int(season)] = {
            "fieldSize": len(field), "atLargeHitRate": round(hit / len(actual_at_large), 3),
            "seedMAE": round(float(err.mean()), 2), "seedSpearman": round(float(rho), 3),
        }
    return {
        "seasons": len(per_season), "meanSeedMAE": round(float(np.mean(seed_errors)), 3),
        "meanSeedSpearman": round(float(np.mean(ranks)), 3),
        "atLargeHitRate": round(sum(selection_hits) / sum(selection_totals), 3),
        "atLargeHits": sum(selection_hits), "atLargeSlots": sum(selection_totals), "bySeason": per_season,
    }


def baseline_power_only(table: pd.DataFrame) -> dict:
    """The simplest alternative: rank purely by power, nothing else. What the composite score has to beat."""
    seasons = sorted(table["season"].unique())
    hits = totals = 0
    for season in seasons:
        test_all = table[table["season"] == season]
        field = test_all[test_all["seed"].notna()]
        if field.empty:
            continue
        auto_bids = dict(zip(field[field["bidType"] == "auto"]["conference"], field[field["bidType"] == "auto"]["team"]))
        autos = set(auto_bids.values())
        actual_at_large = set(field[field["bidType"] == "at-large"]["team"])
        pool = test_all[~test_all["team"].isin(autos)].nsmallest(len(actual_at_large), "powerRank")
        hits += len(actual_at_large & set(pool["team"]))
        totals += len(actual_at_large)
    return {"atLargeHitRate": round(hits / totals, 3)}


def main() -> None:
    table = load()
    result = loso(table)
    base = baseline_power_only(table)
    print(f"seasons graded: {result['seasons']}")
    print(f"at-large hit rate: {result['atLargeHitRate']:.1%} ({result['atLargeHits']} of {result['atLargeSlots']})"
          f"  |  power-rank-only baseline: {base['atLargeHitRate']:.1%}")
    print(f"seed MAE: {result['meanSeedMAE']:.2f}  |  seed rank correlation: {result['meanSeedSpearman']:.3f}")
    for season, s in sorted(result["bySeason"].items()):
        print(f"  {season}: at-large hit {s['atLargeHitRate']:.0%}  seed MAE {s['seedMAE']:.2f}  rho {s['seedSpearman']:.2f}")

    full_fit = [table[table["season"] == s][table[table["season"] == s]["seed"].notna()] for s in table["season"].unique()]
    beta = atlarge.fit(full_fit, ridge=RIDGE)
    model = {"features": list(atlarge.FEATURES) + ["intercept"], "coefficients": [round(float(b), 4) for b in beta], "ridge": RIDGE,
            "fitted": f"{min(table['season'])}-{max(table['season'])}", "seasons": result["seasons"]}
    (ROOT / "data" / "atlarge_model.json").write_text(json.dumps(model, indent=2) + "\n")

    summary = {**result, "powerOnlyBaseline": base}
    (ROOT / "site" / "data" / "atlarge_backtest.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
