"""Choose the basketball profile's parameters, honestly.

Same discipline as the football tuner: search on one block of seasons, report
on another that took no part in the search. Basketball needs its own numbers
because its margins are tighter than football's (a standard deviation near 14
against 16.5), its home advantage is different, and it has 365 teams rather
than 138 for the prior to shrink toward.

Objective is mean absolute error of the predicted margin. Accuracy is too blunt
to tune on and MAE is what any downstream spread work depends on.

Run:  PYTHONPATH=src python3 scripts/tune_basketball.py
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.ratings import bb_backtest as bb  # noqa: E402

CHAIN = range(2021, 2027)
TUNE = [2022, 2023, 2024]
HOLDOUT = [2025, 2026]

GRID = {
    "compression": [14.0, 20.0, 26.0, 34.0, 1e6],
    "ridge": [2.0, 4.0, 8.0, 15.0],
    "prior_regression": [0.2, 0.35, 0.5, 0.7],
}


def score(seasons, **params) -> dict:
    frame = bb.evaluate(CHAIN, scored=seasons, **params)
    return {
        "mae": frame["mae"].mean(),
        "accuracy": frame["accuracy"].mean(),
        "brier": frame["brier"].mean(),
        "home_field": frame["home_field"].mean(),
    }


def main() -> None:
    combos = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
    print(f"searching {len(combos)} combinations on {TUNE}")

    started = time.time()
    rows = []
    for index, params in enumerate(combos, 1):
        rows.append({**params, **score(TUNE, **params)})
        if index % 10 == 0:
            print(f"  {index}/{len(combos)}  ({time.time() - started:.0f}s)")

    results = pd.DataFrame(rows).sort_values("mae")
    print("\nbest eight on the tuning seasons:")
    print(results.head(8).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best = {k: results.iloc[0][k] for k in GRID}
    print(f"\nchosen: {best}")

    print(f"\nholdout {HOLDOUT}, never searched:")
    comparison = pd.DataFrame(
        {"tuned": score(HOLDOUT, **best), "untuned defaults": score(HOLDOUT)}
    )
    print(comparison.to_string(float_format=lambda v: f"{v:.4f}"))

    results.to_csv(ROOT / "data" / "tuning_basketball.csv", index=False)
    print(f"\nfull grid written ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
