"""Were the simulation's probabilities any good? Check them against history.

For every season 2014-2025 (2020 was not a full schedule), the simulation is run
as it would have been on the morning of weeks 3, 6, 9 and 12 - using only the
ratings and the schedule known then - and each team's chance of finishing in the
committee's top 12 is compared to whether it did.

That target is not the same as making the field: the four conference champions
and the top Group of Six team get in regardless, and the format changed in 2024.
It is the part of the question that does not depend on which year's rules
applied, which makes it the fairest single thing to grade across a decade.

The comparison that matters is the same run with the rating error switched off.
A simulation that treats its ratings as exact is confident, and in Week 3 it is
wrong about how confident it should be.

Run:  PYTHONPATH=src python3 scripts/backtest_simulation.py
Writes site/data/sim_backtest.json, which the site's method page reads.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.ingest import cfbd  # noqa: E402
from mri.ratings import mri2  # noqa: E402
from mri.sim import season  # noqa: E402

YEARS = [y for y in range(2014, 2026) if y != 2020]
WEEKS = (3, 6, 9, 12)
SIMS = 3_000
NOISE_GRID = (0.0, 0.2, 0.4, 0.6)

CONFERENCE_NAMES = {
    "Mid-American": "MAC",
    "American Athletic": "American",
    "FBS Independents": "FBS Independent",
}


def main() -> None:
    cfp = json.loads((ROOT / "data" / "cfp_final_rankings.json").read_text())

    # The prior chain: each season starts from the last one's final ratings.
    seasons, previous = {}, None
    for year in range(YEARS[0] - 1, YEARS[-1] + 1):
        games = cfbd.games(year)
        games = games[games["season_type"] == "regular"].reset_index(drop=True)
        fbs = sorted(set(games.loc[games["class1"] == "fbs", "team1"])
                     | set(games.loc[games["class2"] == "fbs", "team2"]))
        teams = sorted(set(games["team1"]) | set(games["team2"]))
        prior = mri2.build_prior(previous, teams, centre_teams=fbs) if previous is not None else None
        model = mri2.fit(games, prior=prior, neutral=games["neutral"], anchor_teams=fbs,
                         with_resume=False, with_efficiency=False)
        seasons[year] = (fbs, prior)
        previous = model.power

    rows = []
    for year in YEARS:
        raw = cfbd.games(year, completed_only=False)
        # Games that were never played (cancelled, or a score the feed never got)
        # carry no result. Treating one as played on the strength of its week put
        # a NaN into the fit, which turned every rating in 2024 into NaN and every
        # team's top-12 chance into 100% - so only real results are schedule.
        raw = raw[raw["played"] & (raw["season_type"] == "regular") & (raw["week"] <= 13)]
        raw = raw.reset_index(drop=True)
        fbs, prior = seasons[year]
        conf = _conferences(raw, fbs)
        actual = {r[1] for r in cfp[str(year)]["ranks"][:12]}

        for week in WEEKS:
            so_far = raw[raw["week"] <= week]
            teams = sorted(set(so_far["team1"]) | set(so_far["team2"]))
            model = mri2.fit(
                so_far,
                prior=mri2.build_prior(prior_of(year, seasons), teams, centre_teams=fbs)
                if year > YEARS[0] - 1 else None,
                neutral=so_far["neutral"], anchor_teams=fbs,
                with_resume=False, with_efficiency=False,
            )
            field = [{"team": t, "power": float(model.power.get(t, -20.0)), "conference": conf[t]}
                     for t in fbs]
            schedule = raw.assign(played=raw["week"] <= week)
            schedule.loc[~schedule["played"], ["pts1", "pts2"]] = np.nan
            for noise in NOISE_GRID:
                for name, uncertainty in (("model", 1.0), ("exact", 0.0)):
                    result = season.simulate(
                        field, schedule, home_field=model.home_field, sims=SIMS,
                        seed=season.DEFAULT_SEED,
                        rules={"uncertainty": uncertainty, "committee_noise": noise},
                    )["teams"]
                    for t in fbs:
                        rows.append({"year": year, "week": week, "variant": f"{name}@{noise}",
                                     "team": t, "p": result[t]["top12"], "hit": int(t in actual)})
        print(f"  {year} done")

    frame = pd.DataFrame(rows)
    base = frame["hit"].mean()
    brier = {v: float(((g["p"] - g["hit"]) ** 2).mean()) for v, g in frame.groupby("variant")}
    best_noise = min(NOISE_GRID, key=lambda n: brier[f"model@{n}"])

    def block(name: str) -> dict:
        group = frame[frame["variant"] == f"{name}@{best_noise}"]
        return {
            "brier": round(brier[f"{name}@{best_noise}"], 4),
            "byWeek": {int(w): round(float(((g["p"] - g["hit"]) ** 2).mean()), 4)
                       for w, g in group.groupby("week")},
            "calibration": _bins(group),
            "week3": _bins(group[group["week"] == 3]),
        }

    summary = {
        "seasons": [YEARS[0], YEARS[-1]], "weeks": list(WEEKS), "sims": SIMS,
        "target": "finishes in the committee's final top 12",
        "committeeNoise": best_noise,
        "noiseSweep": {str(n): round(brier[f"model@{n}"], 4) for n in NOISE_GRID},
        "baseRate": round(float(base), 4),
        "baseBrier": round(float(base * (1 - base)), 4),
        "variants": {"model": block("model"), "exact": block("exact")},
    }
    out = ROOT / "site" / "data" / "sim_backtest.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "variants"}, indent=1))
    for name, d in summary["variants"].items():
        print(name, d["brier"], d["byWeek"])


def prior_of(year, seasons):
    return seasons[year][1]


def _conferences(raw: pd.DataFrame, fbs: list[str]) -> dict[str, str]:
    votes: dict[str, Counter] = {t: Counter() for t in fbs}
    for r in raw.itertuples():
        if r.team1 in votes and r.conf1:
            votes[r.team1][r.conf1] += 1
        if r.team2 in votes and r.conf2:
            votes[r.team2][r.conf2] += 1
    out = {}
    for t, c in votes.items():
        name = c.most_common(1)[0][0] if c else "FBS Independent"
        out[t] = CONFERENCE_NAMES.get(name, name)
    return out


def _bins(group: pd.DataFrame) -> list[dict]:
    edges = [0, 0.02, 0.10, 0.30, 0.50, 0.70, 0.90, 1.0001]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        g = group[(group["p"] >= lo) & (group["p"] < hi)]
        if len(g):
            out.append({"range": f"{lo:.0%}-{min(hi, 1):.0%}", "teams": int(len(g)),
                        "predicted": round(float(g["p"].mean()), 3),
                        "observed": round(float(g["hit"].mean()), 3)})
    return out


if __name__ == "__main__":
    main()
