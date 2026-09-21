"""Fit the final-vote model and grade it on seasons it has not seen.

For each season from 2012 to the latest one in the voting file, the candidates are the funnel's pool at the end of the
regular season, described by where each stands among them and how good his team is.
The model is fitted to the other thirteen seasons and asked who won this one.

Alongside it, the simple things a fan would say: the best passer on the best team;
the most productive player; the most productive player on a top-five team. If the
model cannot beat those it is not worth having.

Run:  PYTHONPATH=src python3 scripts/fit_heisman.py
Writes data/heisman_model.json and site/data/heisman_backtest.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.gameday import model as choice  # noqa: E402
from mri.heisman import data, features, final_model, teams  # noqa: E402
from mri.ingest import cfbd  # noqa: E402

USED = ("prod_rank", "group_rank", "prod_z", "team_rank")
PENALTY = 0.3
# label -> (features from the standard matrix, extra columns computed here)
CANDIDATE_SETS = {
    "production only": (("prod_rank", "group_rank", "prod_z"), ()),
    "team only": (("team_rank", "losses", "unbeaten"), ()),
    "production and team (chosen)": (USED, ()),
    "+ record (losses, unbeaten)": (USED + ("losses", "unbeaten"), ()),
    "+ position and last year's finalists": (USED + ("is_rb", "is_receiver", "prev_finalist"), ()),
    "+ late-season production (weeks 10-13)": (USED, ("late_z",)),
    "+ production against top-25 opponents": (USED, ("sig_z", "sig_pg_z")),
}
TOP = 25                # an opponent is "good" if it finished in the top 25 of our ratings


def _z(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() or 1.0)


def extras(seasons, finals, weekly, games) -> None:
    """Attach two families of extra columns to each season's pool.

    ``late_z``: what he did after week 9, standardized against the field. ``sig_z`` and ``sig_pg_z``: how much
    he produced against the teams that finished in the top 25, in total and per such game.
    """
    from mri.heisman import funnel

    zeros = {c: 0.0 for c in ["def_tot", "def_solo", "def_sacks", "def_tfl", "def_pd", "def_td", "def_int"]}
    for s in seasons:
        w9 = funnel.classify(weekly[(weekly["season"] == s.year) & (weekly["week"] == 9)].assign(**zeros))
        at9 = w9.set_index(["player_id", "team"])["off_score"]
        before = np.array([float(at9.get((a, b), 0.0)) for a, b in zip(s.pool["player_id"], s.pool["team"])])
        cols = {"late_z": _z(np.maximum(s.pool["off_score"].to_numpy() - before, 0.0))}
        if games is not None:
            good = set(finals[s.year].index[finals[s.year]["power_rank"] <= TOP])
            g = games[(games["season"] == s.year) & games["opponent"].isin(good)].copy()
            g["score"] = g["pass_yds"] + g["rush_yds"] + g["rec_yds"] + 20 * (g["pass_td"] + g["rush_td"] + g["rec_td"])
            total = g.groupby(["player_id", "team"])["score"].sum()
            count = g.groupby(["player_id", "team"])["score"].size()
            keys = list(zip(s.pool["player_id"], s.pool["team"]))
            sig = np.array([float(total.get(k, 0.0)) for k in keys])
            n = np.array([float(count.get(k, 0.0)) for k in keys])
            cols["sig_z"] = _z(sig)
            cols["sig_pg_z"] = _z(np.where(n > 0, sig / np.maximum(n, 1), 0.0))
        s.extra = pd.DataFrame(cols)


def loyo(seasons, names, extra_names=(), penalty=PENALTY):
    cols = [features.NAMES.index(n) for n in names]

    def X(x):
        base = x.X[:, cols]
        return np.hstack([base, x.extra[list(extra_names)].to_numpy()]) if extra_names else base

    out = {"top1": 0, "top3": 0, "log": 0.0, "ranks": [], "p": [], "hit": [], "finalists": [0, 0]}
    for s in seasons:
        beta = choice.fit([(X(x), x.winner) for x in seasons if x.year != s.year], penalty)
        p = choice.probabilities(beta, X(s))
        order = np.argsort(-p)
        rank = int(np.where(order == s.winner)[0][0]) + 1
        out["ranks"].append(rank)
        out["top1"] += rank == 1
        out["top3"] += rank <= 3
        out["log"] -= np.log(max(p[s.winner], 1e-9))
        out["p"].extend(p.tolist())
        out["hit"].extend((np.arange(len(p)) == s.winner).astype(int).tolist())
    n = len(seasons)
    return {"top1": out["top1"] / n, "top3": out["top3"] / n, "logLoss": out["log"] / n,
            "meanWinnerRank": float(np.mean(out["ranks"])), "ranks": out["ranks"], "_p": out["p"], "_hit": out["hit"]}


def baselines(seasons):
    def rate(pick):
        return float(np.mean([pick(s) == s.winner for s in seasons]))

    def best_on_best(s):
        top = s.pool[s.pool["team_rank"] == s.pool["team_rank"].min()]
        return int(top["off_score"].idxmax())

    def top_five(s):
        pool = s.pool[s.pool["team_rank"] <= 5]
        return int(pool["off_score"].idxmax())

    return {"mostProductive": rate(lambda s: int(s.pool["off_score"].idxmax())),
            "mostProductiveOnATopFiveTeam": rate(top_five), "bestOnTheBestTeam": rate(best_on_best),
            "uniform": float(np.mean([1 / len(s.pool) for s in seasons]))}


def main() -> None:
    voting = data.load_voting()
    table = data.load_players()
    last = data.last_season(voting)
    states = {y: teams.team_state(cfbd.games(y)) for y in range(final_model.FIRST_SEASON, last + 1)}
    seasons = [final_model.season(y, table, states[y], voting) for y in range(final_model.FIRST_SEASON, last + 1)]
    games_path = ROOT / "data" / "parquet" / "player_games.parquet"
    extras(seasons, states, pd.read_parquet(ROOT / "data" / "parquet" / "player_weekly.parquet"),
           pd.read_parquet(games_path) if games_path.exists() else None)
    for s in seasons:
        assert s.winner is not None, f"{s.year}'s winner is not among the candidates"
    n = len(seasons)

    comparison = {}
    for label, (names, more) in CANDIDATE_SETS.items():
        if more and any(m not in seasons[0].extra.columns for m in more):
            continue                                   # the per-game table has not been built
        r = loyo(seasons, names, more)
        comparison[label] = {k: round(float(r[k]), 3) for k in ("top1", "top3", "logLoss", "meanWinnerRank")}
        print(f"{label:38s} top-1 {r['top1']:.2f}  top-3 {r['top3']:.2f}  mean rank {r['meanWinnerRank']:.1f}  log loss {r['logLoss']:.2f}")
        if label == "production and team (chosen)":
            chosen = r
    base = baselines(seasons)
    print("baselines:", {k: round(v, 3) for k, v in base.items()})

    p, hit = np.array(chosen["_p"]), np.array(chosen["_hit"])
    calibration = []
    for lo, hi in [(0, .02), (.02, .05), (.05, .15), (.15, .30), (.30, .50), (.50, 1.01)]:
        m = (p >= lo) & (p < hi)
        if m.sum():
            calibration.append({"range": f"{lo:.0%}-{min(hi, 1):.0%}", "candidates": int(m.sum()),
                                "predicted": round(float(p[m].mean()), 3), "observed": round(float(hit[m].mean()), 3)})

    cols = [features.NAMES.index(x) for x in USED]
    beta = choice.fit([(s.X[:, cols], s.winner) for s in seasons], PENALTY)
    coefficients = dict(zip(USED, (round(float(b), 4) for b in beta)))
    print("coefficients:", coefficients)

    model = {"features": list(USED), "coefficients": [coefficients[k] for k in USED], "ridge": PENALTY,
             "fitted": f"{seasons[0].year}-{seasons[-1].year}", "seasons": n, "candidatesPerSeason": len(seasons[0].pool),
             "groups": list(final_model.SIZES), "teamRankCutoff": 45}
    (ROOT / "data" / "heisman_model.json").write_text(json.dumps(model, indent=2) + "\n")

    worst = sorted(zip((s.year for s in seasons), chosen["ranks"]), key=lambda kv: -kv[1])[:3]
    summary = {"seasons": n, "years": [seasons[0].year, seasons[-1].year], "baselines": {k: round(v, 3) for k, v in base.items()},
               "uniformLogLoss": round(float(np.log(len(seasons[0].pool))), 3), "candidateSets": comparison,
               "chosen": "production and team (chosen)", "calibration": calibration,
               "winnerRanks": {str(s.year): r for s, r in zip(seasons, chosen["ranks"])},
               "hardestSeasons": [{"year": y, "winnerRank": r} for y, r in worst]}
    (ROOT / "site" / "data" / "heisman_backtest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("winner's rank by season:", summary["winnerRanks"])


if __name__ == "__main__":
    main()
