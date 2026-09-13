"""Choose MRI 2.0's hyperparameters honestly.

Three numbers govern the model: how hard ratings are shrunk toward the prior
(``ridge``), where blowout margins stop paying (``compression``), and how much
of last season is carried into this one (``prior_regression``).

They are picked on one block of seasons and reported on another that took no
part in the search, because a grid search scored on the same years it was tuned
on will always look better than the model really is.

Objective is mean absolute error of the predicted margin. Accuracy is too blunt
to tune on - most games have an obvious favourite, so it barely moves - and MAE
is what the betting module ultimately depends on.

Run:  PYTHONPATH=src python3 scripts/tune_mri2.py
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.ratings import backtest  # noqa: E402

TUNE_SEASONS = range(2003, 2014)
HOLDOUT_SEASONS = range(2014, 2020)

GRID = {
    "ridge": [2.0, 4.0, 6.0, 9.0, 14.0, 20.0, 30.0, 45.0],
    "compression": [14.0, 20.0, 28.0, 40.0, 1e6],
    "prior_regression": [0.15, 0.3, 0.45, 0.6, 0.8],
}


def score(games: pd.DataFrame, seasons, **params) -> dict:
    frame = backtest.evaluate_archive(
        games,
        seasons=list(seasons),
        with_classic=False,
        prior_regression=params["prior_regression"],
        ridge=params["ridge"],
        compression=params["compression"],
    )
    return {
        "mae": frame["mri2_mae"].mean(),
        "accuracy": frame["mri2_accuracy"].mean(),
        "brier": frame["mri2_brier"].mean(),
        "classic_accuracy": frame["classic_accuracy"].mean(),
    }


def main() -> None:
    games = pd.read_parquet(ROOT / "data" / "parquet" / "archive_games.parquet")
    combos = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
    print(f"searching {len(combos)} combinations on {TUNE_SEASONS.start}-{TUNE_SEASONS.stop - 1}")

    started = time.time()
    rows = []
    for i, params in enumerate(combos, 1):
        rows.append({**params, **score(games, TUNE_SEASONS, **params)})
        if i % 40 == 0:
            print(f"  {i}/{len(combos)}  ({time.time() - started:.0f}s)")

    results = pd.DataFrame(rows).sort_values("mae")
    print("\nbest ten on the tuning seasons:")
    print(results.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    best = results.iloc[0]
    params = {k: best[k] for k in GRID}
    print(f"\nchosen: {params}")

    print(f"\nholdout {HOLDOUT_SEASONS.start}-{HOLDOUT_SEASONS.stop - 1}, never searched:")
    tuned = score(games, HOLDOUT_SEASONS, **params)
    default = score(
        games,
        HOLDOUT_SEASONS,
        ridge=9.0,
        compression=28.0,
        prior_regression=0.35,
    )
    comparison = pd.DataFrame({"tuned": tuned, "untuned defaults": default})
    print(comparison.to_string(float_format=lambda v: f"{v:.4f}"))

    results.to_csv(ROOT / "data" / "tuning_results.csv", index=False)
    print(f"\nfull grid written to data/tuning_results.csv ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
