"""Fit the College GameDay choice model and grade it.

For each week from 2014 to 2025 (2020 aside) in which GameDay went to an FBS game,
the model is asked to pick that game out of every FBS game that week, using only
ratings from before the week. It is scored by leaving each season out in turn.

The features and their fate are reported, including the ones that did not survive:
whether GameDay has already been to a place this season, and how often it has
hosted lately, are the rules of thumb everyone repeats, and neither improves the
prediction once the rankings are known.

Run:  PYTHONPATH=src python3 scripts/fit_gameday.py
Writes data/gameday_model.json and site/data/gameday_backtest.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.gameday import features, history, model  # noqa: E402

STAKES = ("best_rank", "worst_rank", "both_top10", "both_top25", "both_unbeaten", "losses")
USED = STAKES + ("last_rank_best", "last_rank_worst", "brand")
PENALTY = 3.0
CANDIDATE_SETS = {
    "rankings and records only": STAKES,
    "+ how the teams ranked last season and how often GameDay has wanted them (chosen)": USED,
    "+ closeness of the game": USED + ("closeness",),
    "+ SEC / Big Ten host": USED + ("elite_conference",),
    "+ already visited this season": USED + ("repeat_home", "repeat_away"),
    "+ hosted lately": USED + ("host_recent",),
    "+ played each other at this point in the last two years": USED + ("rivalry",),
}


def loyo(weeks, names, penalty=PENALTY):
    cols = [features.NAMES.index(n) for n in names]
    years = sorted({w.year for w in weeks})
    total = {"weeks": 0, "top1": 0.0, "top3": 0.0, "log": 0.0}
    calibration = []
    for year in years:
        train = [(w.X[:, cols], w.pick) for w in weeks if w.year != year]
        beta = model.fit(train, penalty)
        for w in (w for w in weeks if w.year == year):
            p = model.probabilities(beta, w.X[:, cols])
            order = np.argsort(-p)
            total["weeks"] += 1
            total["top1"] += int(order[0] == w.pick)
            total["top3"] += int(w.pick in order[:3])
            total["log"] -= np.log(max(p[w.pick], 1e-12))
            calibration.extend((float(pi), int(i == w.pick)) for i, pi in enumerate(p))
    n = total["weeks"]
    return {"top1": total["top1"] / n, "top3": total["top3"] / n, "logLoss": total["log"] / n}, np.array(calibration)


def main() -> None:
    data = history.load()
    weeks = history.build(data=data)
    n = len(weeks)
    uniform = float(np.mean([np.log(len(w.games)) for w in weeks]))
    heuristic = [np.argsort(np.exp(w.X[:, 0]) + np.exp(w.X[:, 1])) for w in weeks]
    baseline = {
        "uniformTop1": float(np.mean([1 / len(w.games) for w in weeks])),
        "uniformLogLoss": uniform,
        "biggestRanksTop1": float(np.mean([h[0] == w.pick for h, w in zip(heuristic, weeks)])),
        "biggestRanksTop3": float(np.mean([w.pick in h[:3] for h, w in zip(heuristic, weeks)])),
    }

    comparison = {}
    for label, names in CANDIDATE_SETS.items():
        result, calib = loyo(weeks, names)
        comparison[label] = {k: round(float(v), 3) for k, v in result.items()}
        if names == USED:
            chosen_calibration = calib
        print(f"{label:40s} top-1 {result['top1']:.3f}  top-3 {result['top3']:.3f}  log loss {result['logLoss']:.3f}")

    bins = []
    for lo, hi in [(0, .02), (.02, .05), (.05, .15), (.15, .30), (.30, .50), (.50, 1.01)]:
        m = (chosen_calibration[:, 0] >= lo) & (chosen_calibration[:, 0] < hi)
        bins.append({"range": f"{lo:.0%}-{min(hi, 1):.0%}", "games": int(m.sum()),
                     "predicted": round(float(chosen_calibration[m, 0].mean()), 3),
                     "observed": round(float(chosen_calibration[m, 1].mean()), 3)})

    cols = [features.NAMES.index(x) for x in USED]
    beta = model.fit([(w.X[:, cols], w.pick) for w in weeks], PENALTY)

    # Weeks GameDay left the FBS schedule, and Army-Navy, as plain rates.
    seasons = [y for y in data["seasons"] if 2014 <= int(y) <= 2025 and y != "2020"]
    stops = [s for y in seasons for s in data["seasons"][y]]
    # FCS and off-schedule stops from about week 5 on, to match the weeks the model is fitted on.
    off = sum(s["kind"] == "fcs" and s["date"] >= f"{s['date'][:4]}-09-25" for s in stops)
    other_rate = off / (n + off)
    svc_years = {y: any(s["kind"] == "svc" for s in data["seasons"][y]) for y in sorted(data["seasons"]) if int(y) >= 2014}
    recent = [svc_years[y] for y in sorted(svc_years)][-4:]

    out = {
        "features": list(USED),
        "coefficients": [round(float(b), 4) for b in beta],
        "ridge": PENALTY,
        "fitted": f"{history.YEARS[0]}-{history.YEARS[-1]}, 2020 excluded; weeks {history.FIRST_WEEK} on",
        "weeks": n,
        "otherRate": round(float(other_rate), 4),
        "armyNavy": {"visitsSince2014": int(sum(svc_years.values())), "seasons": len(svc_years),
                     "lastFour": int(sum(recent)), "estimate": round((sum(recent) + 0.5) / (len(recent) + 1), 3)},
    }
    (ROOT / "data" / "gameday_model.json").write_text(json.dumps(out, indent=2) + "\n")

    summary = {
        "weeks": n, "seasons": [history.YEARS[0], history.YEARS[-1]], "meanCandidates": round(float(np.mean([len(w.games) for w in weeks])), 1),
        "baseline": {k: round(v, 3) for k, v in baseline.items()},
        "candidateSets": comparison, "chosen": next(k for k in CANDIDATE_SETS if k.endswith("(chosen)")),
        "calibration": bins,
        "coefficients": dict(zip(USED, out["coefficients"])),
    }
    (ROOT / "site" / "data" / "gameday_backtest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("baseline",)}, indent=1))
    print("coefficients:", summary["coefficients"])
    print("other-rate", out["otherRate"], "| army-navy", out["armyNavy"])


if __name__ == "__main__":
    main()
