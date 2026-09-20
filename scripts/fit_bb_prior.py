"""Fit the basketball preseason prior: a season's rating from last season's, and the roster.

Reports, so the choice can be seen and not just trusted, the out-of-sample error
of a season's final rating from leaving each season out in turn: the old prior
(65% of last season), and the regression with only what is known before rosters
post, and with a roster.

Run:  PYTHONPATH=src python3 scripts/fit_bb_prior.py
Writes data/bb_prior_model.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.ratings import bb_prior_fit as fit, mri2  # noqa: E402

TIGHTEN = 0.70        # chosen by scripts/backtest_bb_priors.py; see the note in bb_priors.py


def loyo(frame, names):
    error = np.zeros(len(frame))
    for season in frame["season"].unique():
        train, test = frame[frame["season"] != season], frame[frame["season"] == season]
        beta = np.linalg.lstsq(fit.design(train, names), train["y"], rcond=None)[0]
        error[test.index] = test["y"] - fit.design(test, names) @ beta
    return error


def main() -> None:
    chain = fit.old_chain()
    data = fit.complete(fit.dataset(chain))
    print(f"{len(data)} team-seasons over {data['season'].nunique()} seasons")

    old = data["y"] - (1 - mri2.BASKETBALL_PROFILE.prior_regression) * data["last_season"]
    low = data["ret_ws"] < 0.3
    results = {"old": old, "last season only": loyo(data, ()), "before rosters": loyo(data, fit.NO_ROSTER_FEATURES),
               "with a roster": loyo(data, fit.ROSTER_FEATURES)}
    rmse = {}
    print(f"\nout-of-sample error of a season's final rating, leaving each season out   all teams   teams that kept <30% of last year's win shares (n={low.sum()})")
    for name, e in results.items():
        rmse[name] = {"all": round(float(np.sqrt((e ** 2).mean())), 2), "lowReturning": round(float(np.sqrt((e[low] ** 2).mean())), 2)}
        print(f"  {name:20s}{rmse[name]['all']:8.2f}{rmse[name]['lowReturning']:14.2f}")

    model = fit.model_from(data, TIGHTEN)
    model.update({"fitted": f"{fit.YEARS[0]}-{fit.YEARS[-1]}", "teamSeasons": int(len(data)), "outOfSampleRmse": rmse})
    print(json.dumps(model["roster"]), "\n", json.dumps(model["noRoster"]))
    (ROOT / "data" / "bb_prior_model.json").write_text(json.dumps(model, indent=2) + "\n")
    print("wrote data/bb_prior_model.json")


if __name__ == "__main__":
    main()
