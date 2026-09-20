"""Fit the preseason prior: a season's rating from last season's, talent and returning production.

Every season 2013-2025 is rated on its own, with no prior, so the target does not
contain the thing being predicted. Each season from 2015 (the first with talent
data) is then one row per FBS team that was also FBS the year before. 2020 is
left out as a target: a season of a few games and cancelled schedules teaches
nothing about a normal one.

Reported, so the choice can be seen and not just trusted: the out-of-sample error
from leaving each season out in turn, against the old prior (0.7 x last season),
for all teams and for the ones that lost nearly everyone.

Run:  PYTHONPATH=src python3 scripts/fit_prior.py
Writes data/prior_model.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.ingest import cfbd  # noqa: E402
from mri.ratings import mri2, priors  # noqa: E402
from mri.ratings.prior_fit import YEARS, dataset, design, season_ratings  # noqa: E402


def main() -> None:
    power, fbs = season_ratings()
    data = dataset(power, fbs)
    print(f"{len(data)} team-seasons, {data['year'].nunique()} seasons")

    errors = np.zeros(len(data))
    for year in data["year"].unique():
        train, test = data[data["year"] != year], data[data["year"] == year]
        beta = np.linalg.lstsq(design(train), train["y"], rcond=None)[0]
        errors[test.index] = test["y"] - design(test) @ beta
    old = data["y"] - (1 - mri2.DEFAULT_PRIOR_REGRESSION) * data["last_season"]

    low = data["returning"] < 0.10
    print(f"\nout-of-sample error, leaving each season out          all teams   <10% returning (n={low.sum()})")
    print(f"  old prior: {1 - mri2.DEFAULT_PRIOR_REGRESSION:.0%} of last season, rest to average      "
          f"{np.sqrt((old ** 2).mean()):5.2f}          bias {old[low].mean():+.1f}")
    print(f"  talent and returning production                     {np.sqrt((errors ** 2).mean()):5.2f}"
          f"          bias {errors[low].mean():+.1f}")

    beta = np.linalg.lstsq(design(data), data["y"], rcond=None)[0]
    coefficients = dict(zip(priors.TERMS, (round(float(b), 4) for b in beta)))
    print("\ncoefficients:", coefficients)

    # Talent against results, for the team pages: what a roster's talent alone is worth.
    measured = data[~data["team"].isin(priors.TALENT_UNMEASURED)]
    slope, intercept = np.polyfit(measured["talent"], measured["y"], 1)
    residual = measured["y"] - (intercept + slope * measured["talent"])
    # Does beating your roster one year predict beating it the next?
    later = measured.assign(residual=residual)[["team", "year", "residual"]].assign(year=lambda f: f["year"] - 1)
    paired = measured.assign(residual=residual).merge(later, on=["team", "year"], suffixes=("", "_next"))
    talent_fit = {
        "intercept": round(float(intercept), 3), "slope": round(float(slope), 3),
        "r2": round(float(np.corrcoef(measured["talent"], measured["y"])[0, 1] ** 2), 3),
        "residualSd": round(float(residual.std()), 2),
        "persistence": round(float(paired["residual"].corr(paired["residual_next"])), 3),
    }
    print("talent alone:", talent_fit)

    # The service academies, scored against what their composite talent would imply.
    raw_z = {}
    for year in sorted(set(YEARS)):
        z = priors.talent_scores(cfbd.talent(year), fbs[year])
        # talent_scores zeroes the academies; recompute their raw z for the record
        values = cfbd.talent(year).reindex(fbs[year])
        raw_z[year] = (values - values.mean()) / values.std()
    academy = data[data["team"].isin(priors.TALENT_UNMEASURED)]
    academy_z = np.array([raw_z[r.year][r.team] for r in academy.itertuples()])
    academy_gap = academy["y"].to_numpy() - (intercept + slope * np.nan_to_num(academy_z, nan=0.0))
    valid = ~np.isnan(academy_z)
    academies = {"meanZ": round(float(np.nanmean(academy_z)), 2),
                 "gap": round(float(academy_gap[valid].mean()), 1),
                 # ...and after the prior has also counted last season's rating
                 "gapWithPrior": round(float(errors[academy.index].mean()), 1),
                 "teamSeasons": int(valid.sum())}
    print("academies:", academies)

    out = {
        "fitted": f"{YEARS[0]}-{YEARS[-1]}, 2020 excluded",
        "teamSeasons": int(len(data)),
        "coefficients": coefficients,
        "outOfSampleRmse": {"model": round(float(np.sqrt((errors ** 2).mean())), 2),
                            "old": round(float(np.sqrt((old ** 2).mean())), 2)},
        "talentFit": talent_fit,
        "academies": academies,
        "lowReturning": {"share": 0.10, "teamSeasons": int(low.sum()),
                         "oldBias": round(float(old[low].mean()), 1), "newBias": round(float(errors[low].mean()), 1)},
        "talentUnmeasured": list(priors.TALENT_UNMEASURED),
    }
    (ROOT / "data" / "prior_model.json").write_text(json.dumps(out, indent=2) + "\n")
    print("wrote data/prior_model.json")


if __name__ == "__main__":
    main()
