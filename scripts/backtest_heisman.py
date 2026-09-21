"""If this odds engine had existed, how would it have done?

For every season from 2013 to the latest one in the voting file, and each snapshot week (3, 5, 7, 9, 11 and 13), stand at that
point knowing only what was known then: the season's stats to date, the standings to date,
the ratings to date. Play out the rest of the season, project every candidate, score the
field, and see where the eventual winner sat.

Everything learned from other seasons - the final-vote coefficients, the projection ratios -
is learned leaving the predicted season out.

Compared against the same model applied to the standings as they stood, with no
projection and no simulated season. What the engine adds over that is the point of it.

Several ways of projecting are tried side by side on the same simulated seasons: independent
draws (the original), a link between a player's season and his team's, and last season's rate
as the prior a player's rate is pulled toward. The one with the lowest log loss across the six
weeks wins if it is clearly better; otherwise the original stays. The choice is written to
data/heisman_ratios.json, which the live forecast reads.

Run:  PYTHONPATH=src python3 scripts/backtest_heisman.py
Resumable (it saves as it goes). Writes site/data/heisman_forecast_backtest.json.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.gameday import model as choice  # noqa: E402
from mri.heisman import calibrate, data, features, final_model, forecast, history, project, teams  # noqa: E402
from mri.ingest import cfbd  # noqa: E402
from mri.ratings import prior_fit  # noqa: E402

LAST = data.last_season()
SEASONS = list(range(2013, LAST + 1))
WEEKS = project.SNAPSHOT_WEEKS
SIMS = 1000
BUDGET = 250
CACHE = Path("/tmp/heisman_backtest_cache2.pkl")
NAMES = {"Mid-American": "MAC", "American Athletic": "American", "FBS Independents": "FBS Independent"}
MIN_GAIN = 0.02             # log loss a variant must gain over the original to replace it

VARIANTS = {
    "original": {"variant": None, "link": False},
    "team link": {"variant": None, "link": True},
    "last season (weight .5, 2 games)": {"variant": {"last": 0.5, "shrink": 2.0}, "link": False},
    "last season (weight .7, 4 games)": {"variant": {"last": 0.7, "shrink": 4.0}, "link": False},
    "last season (weight .5) + link": {"variant": {"last": 0.5, "shrink": 2.0}, "link": True},
    "last season (weight .7, 4 games) + link": {"variant": {"last": 0.7, "shrink": 4.0}, "link": True},
}


def conferences(games: pd.DataFrame, fbs: list[str]) -> dict[str, str]:
    votes: dict[str, dict] = {t: {} for t in fbs}
    for r in games.itertuples():
        for t, c in ((r.team1, r.conf1), (r.team2, r.conf2)):
            if t in votes and c:
                votes[t][c] = votes[t].get(c, 0) + 1
    out = {}
    for t, v in votes.items():
        name = max(v, key=v.get) if v else "FBS Independent"
        out[t] = NAMES.get(name, name)
    return out


def summarize(rows: list[dict]) -> dict:
    found = [r for r in rows if r]
    ranks = [r["rank"] for r in found]
    return {"top1": round(float(np.mean([x == 1 for x in ranks])), 3), "top3": round(float(np.mean([x <= 3 for x in ranks])), 3),
            "top4": round(float(np.mean([x <= 4 for x in ranks])), 3), "meanWinnerRank": round(float(np.mean(ranks)), 2),
            "winnerProbability": round(float(np.mean([r["p"] for r in found])), 3),
            "logLoss": round(float(-np.mean([np.log(max(r["p"], 1e-3)) for r in found])), 3), "seasons": len(found)}


def main() -> None:
    voting, final_table = data.load_voting(), data.load_players()
    weekly = pd.read_parquet(ROOT / "data" / "parquet" / "player_weekly.parquet")
    power_np, _ = prior_fit.season_ratings(LAST)
    model_meta = final_model.load_model()
    cols = [features.NAMES.index(n) for n in model_meta["features"]]

    games_all = {y: cfbd.games(y, completed_only=False) for y in SEASONS}
    finals = {y: teams.team_state(cfbd.games(y)) for y in range(final_model.FIRST_SEASON, LAST + 1)}
    seasons = [final_model.season(y, final_table, finals[y], voting) for y in range(final_model.FIRST_SEASON, LAST + 1)]
    power_np[2012] = finals[2012]["power"]           # 2013's prior is 2012's ratings

    print("building snapshots", flush=True)
    raw = history.prepare(SEASONS, weekly, final_table, games_all, finals, power_np,
                          lambda y: final_model.previous_finalists(voting, y))
    pools = {name: history.ratio_pools(raw, WEEKS, v["variant"]) for name, v in VARIANTS.items()}
    link = history.team_link(raw)
    print(f"team link (correlation of a candidate's finish with his team's): {link:.2f}", flush=True)

    done = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else {}
    start = time.time()
    for y in SEASONS:
        winner = next(f for f in voting["seasons"][str(y)]["finalists"] if f["finish"] == 1)
        invited = [f for f in voting["seasons"][str(y)]["finalists"] if f.get("finalist") is not False]
        beta = choice.fit([(s.X[:, cols], s.winner) for s in seasons if s.year != y], model_meta["ridge"])
        model = {"features": model_meta["features"], "coefficients": list(beta)}
        g = games_all[y]
        regular = g[(g["season_type"] == "regular") & (g["week"] <= 13)].reset_index(drop=True)
        for w in WEEKS:
            if (y, w) in done:
                continue
            if time.time() - start > BUDGET:
                CACHE.write_bytes(pickle.dumps(done))
                print("budget reached; rerun to continue", flush=True)
                return
            state, pool = raw[(y, w)]["state"], raw[(y, w)]["pool"]
            fbs = list(state.index)
            conf = conferences(g, fbs)
            field = [{"team": t, "power": float(state.loc[t, "power"]), "conference": conf[t]} for t in fbs]
            hidden = regular.assign(played=regular["played"] & (regular["week"] <= w))
            hidden.loc[~hidden["played"], ["pts1", "pts2"]] = np.nan
            runs = forecast.simulate_teams(field, hidden, home_field=state.attrs["home_field"], sims=SIMS)
            left = forecast.remaining_games(hidden, fbs)

            keyed = pool.assign(season=y)
            hit = data.resolve(keyed, y, winner["player"], winner["school"])
            winner_row = int(hit.index[0]) if len(hit) else None
            invited_rows = {int(i) for f in invited for i in data.resolve(keyed, y, f["player"], f["school"]).index}

            def locate(frame, order_key):
                if winner_row is None:
                    return None
                i = int(frame.index[frame["_row"] == winner_row][0])
                return {"p": float(frame["win"][i]), "rank": i + 1, "top4": bool(i < 4)}

            entry = {"n": len(pool), "variants": {}}
            for name, v in VARIANTS.items():
                ratio_pool = np.concatenate([a for k, a in pools[name][w].items() if k != y])
                pool_r = pool.reset_index(drop=True).assign(_row=lambda d: d.index)
                result = forecast.odds(pool_r, runs, left, ratio_pool, model, link=link if v["link"] else 0.0, variant=v["variant"])
                result = result.merge(pool_r[["player", "team", "_row"]], on=["player", "team"], how="left")
                entry["variants"][name] = {
                    "loc": locate(result, None), "leader": (result["player"][0], float(result["win"][0])),
                    "all": [(float(a), float(b), int(r == winner_row), int(r in invited_rows))
                            for a, b, r in zip(result["win"], result["finalist"], result["_row"])]}

            X0 = features.matrix(pool["off_score"].to_numpy(), pool["group_code"].to_numpy(), pool["team_rank"].to_numpy(),
                                 0.0, pool["prev_finalist"].to_numpy())[:, cols]
            base = pool[["player", "team"]].reset_index(drop=True).assign(win=choice.probabilities(beta, X0), _row=lambda d: d.index)
            base = base.sort_values("win", ascending=False).reset_index(drop=True)
            entry["current"] = locate(base, None)
            done[(y, w)] = entry
        print(f"  {y} done", flush=True)
    CACHE.write_bytes(pickle.dumps(done))

    comparison = {}
    for name in VARIANTS:
        by_week = {str(w): summarize([done[(y, w)]["variants"][name]["loc"] for y in SEASONS]) for w in WEEKS}
        comparison[name] = {"byWeek": by_week, "meanLogLoss": round(float(np.mean([v["logLoss"] for v in by_week.values()])), 3),
                            "earlyLogLoss": round(float(np.mean([by_week[k]["logLoss"] for k in ("3", "5")])), 3),
                            "meanTop3": round(float(np.mean([v["top3"] for v in by_week.values()])), 3)}
        print(f"{name:42s} log loss {comparison[name]['meanLogLoss']:.3f}  early (wk 3-5) {comparison[name]['earlyLogLoss']:.3f}  top-3 {comparison[name]['meanTop3']:.2f}")
    best = min(comparison, key=lambda k: comparison[k]["meanLogLoss"])
    chosen = best if comparison["original"]["meanLogLoss"] - comparison[best]["meanLogLoss"] >= MIN_GAIN else "original"
    print("chosen:", chosen)

    v = VARIANTS[chosen]
    ratios_out = {str(w): [round(float(x), 3) for x in np.concatenate(list(history.ratio_pools(raw, WEEKS, v["variant"])[w].values()))]
                  for w in WEEKS}
    setting = {"last": (v["variant"] or {}).get("last", 0.0), "shrink": (v["variant"] or {}).get("shrink", project.SHRINK_GAMES),
               "link": round(link, 3) if v["link"] else 0.0}
    # Finalist odds run high below the favorites. Draw the curve that makes them come true, and grade it
    # honestly: for each season the curve is fitted on the other twelve and applied to that one.
    rows = [(y, *r) for w in WEEKS for y in SEASONS for r in done[(y, w)]["variants"][chosen]["all"]]
    table = np.array(rows, dtype=float)                # year, win, finalist, is winner, is finalist
    finalist_map = calibrate.fit_map(table[:, 2], table[:, 4])
    raw_brier, cal_brier = [], []
    for y in SEASONS:
        train, test = table[table[:, 0] != y], table[table[:, 0] == y]
        knots = calibrate.fit_map(train[:, 2], train[:, 4])
        raw_brier.append(np.mean((test[:, 2] - test[:, 4]) ** 2))
        cal_brier.append(np.mean((calibrate.apply(knots, test[:, 2]) - test[:, 4]) ** 2))
    print(f"finalist Brier: raw {np.mean(raw_brier):.4f} -> calibrated {np.mean(cal_brier):.4f} (each season fitted on the others)")
    (ROOT / "data" / "heisman_ratios.json").write_text(json.dumps(
        {"seasons": [SEASONS[0], SEASONS[-1]], "capped": project.RATIO_CAP, "variant": setting, "finalistMap": finalist_map,
         "byWeek": ratios_out}) + "\n")

    summary = {"seasons": [SEASONS[0], SEASONS[-1]], "sims": SIMS, "chosen": chosen, "setting": setting,
               "teamLink": round(link, 3), "variants": {k: {"meanLogLoss": c["meanLogLoss"], "earlyLogLoss": c["earlyLogLoss"], "meanTop3": c["meanTop3"]}
                                                      for k, c in comparison.items()}, "byWeek": {}, "calibration": [], "finalistCalibration": [],
               "finalistBrier": {"raw": round(float(np.mean(raw_brier)), 5), "calibrated": round(float(np.mean(cal_brier)), 5)}}
    for w in WEEKS:
        summary["byWeek"][str(w)] = {"forecast": comparison[chosen]["byWeek"][str(w)],
                                     "current": summarize([done[(y, w)]["current"] for y in SEASONS])}
    allrows = np.array([r for w in WEEKS for y in SEASONS for r in done[(y, w)]["variants"][chosen]["all"]])
    for lo, hi in [(0, .02), (.02, .05), (.05, .15), (.15, .30), (.30, .50), (.50, 1.01)]:
        m = (allrows[:, 0] >= lo) & (allrows[:, 0] < hi)
        if m.sum():
            summary["calibration"].append({"range": f"{lo:.0%}-{min(hi, 1):.0%}", "candidates": int(m.sum()),
                                           "predicted": round(float(allrows[m, 0].mean()), 3), "observed": round(float(allrows[m, 2].mean()), 3)})
    for lo, hi in [(0, .05), (.05, .15), (.15, .30), (.30, .50), (.50, .75), (.75, 1.01)]:
        m = (allrows[:, 1] >= lo) & (allrows[:, 1] < hi)
        if m.sum():
            summary["finalistCalibration"].append({"range": f"{lo:.0%}-{min(hi, 1):.0%}", "candidates": int(m.sum()),
                                                   "predicted": round(float(allrows[m, 1].mean()), 3), "observed": round(float(allrows[m, 3].mean()), 3)})
    summary["byWeekDetail"] = {str(w): {str(y): {"rank": (done[(y, w)]["variants"][chosen]["loc"] or {}).get("rank"),
                                                "p": round((done[(y, w)]["variants"][chosen]["loc"] or {"p": 0})["p"], 3),
                                                "leader": done[(y, w)]["variants"][chosen]["leader"][0]} for y in SEASONS} for w in WEEKS}
    (ROOT / "site" / "data" / "heisman_forecast_backtest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"finalistCalibration": summary["finalistCalibration"]}, indent=1))


if __name__ == "__main__":
    main()
