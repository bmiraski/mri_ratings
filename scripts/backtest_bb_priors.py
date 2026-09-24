"""Does the roster-aware prior predict basketball games better than the old one?

Every season 2014-2026 is walked in order (2021, the COVID season, is chained but
not scored). Within a season, each slice of games is predicted using only the games
before it: the first 5% from the prior alone, then 5-10%, 10-20%, 20-40% and the
rest, each from a fit to everything earlier. Three priors are compared, each
chained properly - a season starts from the previous season's final ratings under
the same method:

  old              last season, 35% of the way to average (what the site used)
  before rosters   the regression using only what is known before rosters post
  with a roster    the regression with returning and incoming win shares

The new priors' coefficients are fitted leaving the predicted season out.

Run:  PYTHONPATH=src python3 scripts/backtest_bb_priors.py
Writes site/data/bb_prior_backtest.json, which the method page reads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.ratings import bb_prior_fit as fit, bb_priors, mri2  # noqa: E402

BINS = ((0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 1.0))
KINDS = ("old", "before rosters", "with a roster")
SCORED = [y for y in range(2014, 2027) if y != 2021]


def main() -> None:
    chain = fit.old_chain()
    data = fit.complete(fit.dataset(chain))
    played, recruits, draft = bb_priors._tables()
    profile = mri2.BASKETBALL_PROFILE

    state = {k: chain[2012] for k in KINDS}
    rows = []
    for season in range(2013, fit.LAST + 1):
        games = fit.season_games(season)
        d1 = fit.division_one(season, games)
        teams = sorted(set(games["team1"]) | set(games["team2"]))
        n = len(games)
        model = fit.model_from(data[data["season"] != season], bb_priors.load_model()["tighten"])
        roster = {t: set(g["athlete_id"]) for t, g in played[played["season"] == season].groupby("team_c")}
        with_roster = bb_priors.features(season, d1, played, recruits, draft, roster=roster)
        without = bb_priors.features(season, d1, played, recruits, draft, roster=None)

        for kind in KINDS:
            previous = state[kind]
            if kind == "old":
                prior = mri2.build_prior(previous, teams, profile.prior_regression, centre_teams=d1,
                                             outsiders_to_replacement=False)
            else:
                prior = bb_priors.preseason_prior(previous, teams, d1, with_roster if kind == "with a roster" else without, model)
            for lo, hi in BINS:
                a, b = int(n * lo), int(n * hi)
                test = games.iloc[a:b]
                if season in SCORED and len(test):
                    if a == 0:
                        power, home = prior, profile.home_field_prior
                    else:
                        ratings = fit.fit_season(games.iloc[:a], prior, d1)
                        power, home = ratings.power, ratings.home_field
                    floor = float(power.min()) - 5
                    h = np.nan_to_num(power.reindex(test["team2"]).to_numpy(float), nan=floor)
                    v = np.nan_to_num(power.reindex(test["team1"]).to_numpy(float), nan=floor)
                    predicted = h - v + np.where(test["neutral"].to_numpy(bool), 0.0, home)
                    actual = (test["pts2"] - test["pts1"]).to_numpy(float)
                    rows.append(pd.DataFrame({"season": season, "kind": kind, "bin": f"{lo:.0%}-{hi:.0%}", "lo": lo,
                                              "error": np.abs(predicted - actual),
                                              "correct": (predicted > 0) == (actual > 0)}))
            state[kind] = fit.fit_season(games, prior, d1).power
        print(f"  {season} done", flush=True)

    frame = pd.concat(rows, ignore_index=True)
    out = {"seasons": [SCORED[0], SCORED[-1]], "excluded": [2021], "tighten": bb_priors.load_model()["tighten"], "bins": {}}
    for (lo, hi) in BINS:
        label = f"{lo:.0%}-{hi:.0%}"
        entry = {}
        for kind in KINDS:
            s = frame[(frame["bin"] == label) & (frame["kind"] == kind)].reset_index(drop=True)
            entry[kind] = {"games": int(len(s)), "mae": round(float(s["error"].mean()), 2),
                           "accuracy": round(float(s["correct"].mean()), 4)}
        base = frame[(frame["bin"] == label) & (frame["kind"] == "old")]["error"].reset_index(drop=True)
        for kind in ("before rosters", "with a roster"):
            new = frame[(frame["bin"] == label) & (frame["kind"] == kind)]["error"].reset_index(drop=True)
            gain = base - new
            entry[kind]["gain"] = round(float(gain.mean()), 2)
            entry[kind]["gainError"] = round(float(1.96 * gain.std(ddof=1) / np.sqrt(len(gain))), 2)
        out["bins"][label] = entry
        print(f"{label:9s} old {entry['old']['mae']:6.2f} | before rosters {entry['before rosters']['mae']:6.2f} "
              f"({entry['before rosters']['gain']:+.2f}) | with a roster {entry['with a roster']['mae']:6.2f} "
              f"({entry['with a roster']['gain']:+.2f} +/- {entry['with a roster']['gainError']:.2f})")
    (ROOT / "site" / "data" / "bb_prior_backtest.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
