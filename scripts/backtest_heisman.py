"""If this odds engine had existed, how would it have done?

For every season 2013-2025 and each snapshot week (3, 5, 7, 9, 11 and 13), stand at that
point knowing only what was known then: the season's stats to date, the standings to date,
the ratings to date. Play out the rest of the season, project every candidate, score the
field, and see where the eventual winner sat.

Everything learned from other seasons - the final-vote coefficients, the projection ratios -
is learned leaving the predicted season out.

Compared against the same model applied to the standings as they stood, with no
projection and no simulated season. What the engine adds over that is the point of it.

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
from mri.heisman import data, features, final_model, forecast, history, project, teams  # noqa: E402
from mri.ingest import cfbd  # noqa: E402
from mri.ratings import prior_fit  # noqa: E402

SEASONS = list(range(2013, 2026))
WEEKS = project.SNAPSHOT_WEEKS
SIMS = 1000
BUDGET = 250
CACHE = Path("/tmp/heisman_backtest_cache.pkl")
NAMES = {"Mid-American": "MAC", "American Athletic": "American", "FBS Independents": "FBS Independent"}


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


def main() -> None:
    voting, final_table, weekly = data.load_voting(), data.load_players(), pd.read_parquet(ROOT / "data" / "parquet" / "player_weekly.parquet")
    power_np, _ = prior_fit.season_ratings()
    model_meta = final_model.load_model()
    cols = [features.NAMES.index(n) for n in model_meta["features"]]

    games_all = {y: cfbd.games(y, completed_only=False) for y in SEASONS}
    finals = {y: teams.team_state(cfbd.games(y)) for y in range(final_model.FIRST_SEASON, 2026)}
    seasons = [final_model.season(y, final_table, finals[y], voting) for y in range(final_model.FIRST_SEASON, 2026)]
    power_np[2012] = finals[2012]["power"]           # 2013's prior is 2012's ratings

    # Projection ratios: what became of every candidate at each snapshot, in every season.
    print("building projection ratios", flush=True)
    prep, ratio_by = history.prepare(SEASONS, weekly, final_table, games_all, finals, power_np,
                                     lambda y: final_model.previous_finalists(voting, y))
    ratios_out = {str(w): [round(float(x), 3) for x in np.concatenate(list(ratio_by[w].values()))] for w in WEEKS}
    (ROOT / "data" / "heisman_ratios.json").write_text(json.dumps(
        {"seasons": [SEASONS[0], SEASONS[-1]], "capped": project.RATIO_CAP, "byWeek": ratios_out}) + "\n")

    done = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else {}
    start = time.time()
    for y in SEASONS:
        winner = next(f for f in voting["seasons"][str(y)]["finalists"] if f["finish"] == 1)
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
            state, pool = prep[(y, w)]
            fbs = list(state.index)
            conf = conferences(g, fbs)
            field = [{"team": t, "power": float(state.loc[t, "power"]), "conference": conf[t]} for t in fbs]
            hidden = regular.assign(played=regular["played"] & (regular["week"] <= w))
            hidden.loc[~hidden["played"], ["pts1", "pts2"]] = np.nan
            runs = forecast.simulate_teams(field, hidden, home_field=state.attrs["home_field"], sims=SIMS)
            left = forecast.remaining_games(hidden, fbs)
            ratio_pool = np.concatenate([v for k, v in ratio_by[w].items() if k != y])
            result = forecast.odds(pool, runs, left, ratio_pool, model)

            X0 = features.matrix(pool["off_score"].to_numpy(), pool["group_code"].to_numpy(), pool["team_rank"].to_numpy(),
                                 0.0, pool["prev_finalist"].to_numpy())[:, cols]
            p0 = choice.probabilities(beta, X0)
            base = pool[["player", "team"]].assign(win=p0).sort_values("win", ascending=False).reset_index(drop=True)

            def locate(frame):
                hit = data.resolve(frame.assign(season=y), y, winner["player"], winner["school"])
                if hit.empty:
                    return None
                i = int(hit.index[0])
                return {"p": float(frame["win"][i]), "rank": i + 1, "top4": bool(i < 4),
                        "finalist": float(frame["finalist"][i]) if "finalist" in frame else None}

            done[(y, w)] = {"forecast": locate(result), "current": locate(base), "n": len(pool),
                            "all": [(float(a), int(b)) for a, b in zip(result["win"], (result["player"] == winner["player"]).astype(int))],
                            "leader": (result["player"][0], float(result["win"][0]))}
        print(f"  {y} done", flush=True)
    CACHE.write_bytes(pickle.dumps(done))

    summary = {"seasons": [SEASONS[0], SEASONS[-1]], "sims": SIMS, "byWeek": {}, "calibration": []}
    pooled = {w: [] for w in WEEKS}
    for w in WEEKS:
        rows = [(y, done[(y, w)]) for y in SEASONS]
        for kind in ("forecast", "current"):
            found = [r[kind] for _, r in rows if r[kind]]
            ranks = [r["rank"] for r in found]
            summary["byWeek"].setdefault(str(w), {})[kind] = {
                "top1": round(float(np.mean([x == 1 for x in ranks])), 3), "top3": round(float(np.mean([x <= 3 for x in ranks])), 3),
                "top4": round(float(np.mean([x <= 4 for x in ranks])), 3), "meanWinnerRank": round(float(np.mean(ranks)), 2),
                "winnerProbability": round(float(np.mean([r["p"] for r in found])), 3),
                "logLoss": round(float(-np.mean([np.log(max(r["p"], 1e-3)) for r in found])), 3), "seasons": len(found)}
        for _, r in rows:
            pooled[w].extend(r["all"])
    allp = np.array([x for w in WEEKS for x in pooled[w]])
    for lo, hi in [(0, .02), (.02, .05), (.05, .15), (.15, .30), (.30, .50), (.50, 1.01)]:
        m = (allp[:, 0] >= lo) & (allp[:, 0] < hi)
        if m.sum():
            summary["calibration"].append({"range": f"{lo:.0%}-{min(hi, 1):.0%}", "candidates": int(m.sum()),
                                           "predicted": round(float(allp[m, 0].mean()), 3), "observed": round(float(allp[m, 1].mean()), 3)})
    summary["byWeekDetail"] = {str(w): {str(y): {"rank": done[(y, w)]["forecast"]["rank"] if done[(y, w)]["forecast"] else None,
                                                "p": round(done[(y, w)]["forecast"]["p"], 3) if done[(y, w)]["forecast"] else None,
                                                "leader": done[(y, w)]["leader"][0]} for y in SEASONS} for w in WEEKS}
    (ROOT / "site" / "data" / "heisman_forecast_backtest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"byWeek": summary["byWeek"], "calibration": summary["calibration"]}, indent=1))


if __name__ == "__main__":
    main()
