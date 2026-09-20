"""If this page had existed at the end of Week 3, how often would it have been right?

The choice model is graded on weeks where it was handed the *actual* rankings. This
is the harder, honest test: standing at the end of Week 3 of each season 2015-2025
(2020 aside), with only the ratings and results known then, simulate the rest of the
season and forecast every GameDay stop from Week 8 through the championship week.
Then look at where GameDay actually went.

Run:  PYTHONPATH=src python3 scripts/backtest_gameday.py
Writes site/data/gameday_forecast_backtest.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.gameday import forecast, history  # noqa: E402
from mri.ingest import cfbd  # noqa: E402
from mri.ratings import mri2, prior_fit  # noqa: E402

AS_OF = 3
FIRST, SIMS = 8, 2_000
NAMES = {"Mid-American": "MAC", "American Athletic": "American", "FBS Independents": "FBS Independent"}


def main() -> None:
    data = history.load()
    model = forecast.load_model()
    power_np, fbs_np = prior_fit.season_ratings()
    rows = []
    for year in [y for y in range(2015, 2026) if y != 2020]:
        games = cfbd.games(year, completed_only=False)
        games = games[games["season_type"] == "regular"].reset_index(drop=True)
        fbs = set(fbs_np[year])
        stops = data["seasons"][str(year)]
        weeks = history.stop_weeks(stops, games)
        cg_week = next((w for s, w in zip(stops, weeks) if s["kind"] == "cg"), None)
        if cg_week is None:
            continue

        so_far = games[(games["week"] <= AS_OF) & games["played"]]
        teams = sorted(set(games["team1"]) | set(games["team2"]))
        prior = mri2.build_prior(power_np[year - 1], teams, centre_teams=sorted(fbs))
        fitted = mri2.fit(so_far, prior=prior, neutral=so_far["neutral"], anchor_teams=sorted(fbs),
                          with_resume=False, with_efficiency=False)
        votes = {t: {} for t in fbs}
        for r in games.itertuples():
            for t, c in ((r.team1, r.conf1), (r.team2, r.conf2)):
                if t in votes and c:
                    votes[t][c] = votes[t].get(c, 0) + 1
        conf = {t: NAMES.get(max(v, key=v.get), max(v, key=v.get)) if v else "FBS Independent" for t, v in votes.items()}
        field = [{"team": t, "power": float(fitted.power.get(t, -20.0)), "conference": conf[t]} for t in sorted(fbs)]

        schedule = games.assign(played=games["played"] & (games["week"] <= AS_OF))
        schedule.loc[~schedule["played"], ["pts1", "pts2"]] = np.nan
        target = [w for w in sorted(set(weeks)) if FIRST <= w <= cg_week]
        context = {"lastRank": {t: float(r) for t, r in power_np[year - 1].rank(ascending=False).items()},
                   "brand": history.appearance_rates(data, year)}
        result = forecast.forecast(field, schedule, home_field=fitted.home_field, model=model, weeks=target,
                                   championship_week=cg_week, sims=SIMS, context=context)

        for stop, week in zip(stops, weeks):
            if week is None or week not in result["weeks"] or stop["kind"] not in ("reg", "cg"):
                continue
            entries = result["weeks"][week]["games"]
            a, b = stop["teams"]
            if stop["kind"] == "cg":
                mine = [e for e in entries if e["conference"] == conf.get(a)]
            else:
                mine = [e for e in entries if {e["home"], e["away"]} == {a, b}]
            if not mine:
                continue
            p = mine[0]["probability"]
            rank = 1 + sum(e["probability"] > p for e in entries)
            top = max(entries, key=lambda e: e["probability"])
            rows.append({"year": year, "week": week, "p": p, "rank": rank, "n": len(entries),
                         "kind": stop["kind"], "top": top["probability"],
                         "pick": " vs ".join(stop["teams"]),
                         "favourite": top.get("conference") or f"{top['away']} at {top['home']}"})
        print(f"  {year} done")

    frame = pd.DataFrame(rows)
    out = {
        "asOf": AS_OF, "firstWeek": FIRST, "sims": SIMS, "stops": int(len(frame)),
        "seasons": [int(frame["year"].min()), int(frame["year"].max())],
        "top1": round(float((frame["rank"] == 1).mean()), 3),
        "top3": round(float((frame["rank"] <= 3).mean()), 3),
        "top5": round(float((frame["rank"] <= 5).mean()), 3),
        "meanProbability": round(float(frame["p"].mean()), 3),
        "logLoss": round(float(-np.log(frame["p"].clip(lower=1e-4)).mean()), 3),
        "uniformLogLoss": round(float(np.log(frame["n"]).mean()), 3),
        "byWeek": {int(w): {"stops": int(len(g)), "top1": round(float((g["rank"] == 1).mean()), 3),
                            "top3": round(float((g["rank"] <= 3).mean()), 3)} for w, g in frame.groupby("week")},
    }
    frame.to_csv("/tmp/gd/forecast_stops.csv", index=False)
    (ROOT / "site" / "data" / "gameday_forecast_backtest.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "byWeek"}, indent=1))
    print("by week:", out["byWeek"])


if __name__ == "__main__":
    main()
