"""Backtest Phase 3: the joint simulation, graded against every real field 2011-2025.

For each season (2020 cancelled) and three points in it - February 1, the day the
first conference tournament tips off, and Selection Sunday itself - rate every team
on the games played so far, simulate the rest of the season thousands of times, and
compare what comes out to the field the committee actually picked:

* **Field odds** (every D1 team's chance to make it) - Brier score and log loss,
  next to the simplest honest alternative: "if the season ended today", with each
  conference's standings leader as its automatic bid. The simulation has to beat
  that to be worth its complexity.
* **The projected field** (one bracket: likeliest automatic bids, likeliest
  at-large teams) - how many of the real 68 it names.
* **Seed lines** - for real field teams the projection also has, how far its seed
  line is from the real one.
* **Region placement** - whether every projected bracket satisfies every
  committee principle :mod:`mri.bracket.regions` implements.

Two settings for the automatic bid, since whether a conference leaves the
placeholder is Ben's call: every conference on the placeholder, and every
conference the Phase 1 backtest flags as clearing both bars switched to the
model. The second is also run with ratings treated as exact (no per-world rating
error), to show what the error bars are worth.

The at-large score is fit leave-one-season-out, as in Phase 2 - a season's own
field never trains the coefficients it's graded with. The committee-noise size
(``committeeNoise`` in data/atlarge_model.json) is one number chosen across all
seasons at once, so it isn't held out the same way; it's a single parameter, and
the Selection Sunday grid in scripts/backtest_atlarge.py shows how flat the
optimum is. A conference's tournament
format is read from its three previous tournaments, as it would have to be live;
for the first seasons, with no earlier tournaments in the window, from that
season's own (a format is announced before the season, so this leaks structure,
not results).

Run:  PYTHONPATH=src python3 scripts/backtest_bracketology.py
Resumable (cache in /tmp); writes data/bracketology_backtest.json.
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

from mri.bracket import atlarge, history, joint, regions, seeding, template as template_mod  # noqa: E402
from mri.ingest import bb_bracket, bb_registry, cbbd  # noqa: E402

SEASONS = [y for y in range(2011, 2026) if y != 2020]
CHECKPOINTS = ("feb1", "confStart", "selectionSunday")
SIMS = 2000
LOOKBACK = 3
CACHE = Path("/tmp/bracketology_backtest_cache.pkl")
BUDGET = 240
EPS = 1e-4


def templates_for(season: int, conferences: set[str]) -> dict:
    prior = [y for y in SEASONS if season - LOOKBACK <= y < season]
    years = prior or [season]
    frames = [bb_bracket.conference_tournaments(y) for y in years]
    return {c: template_mod.infer([f[f["conference"] == c] for f in frames if not f.empty], min_seasons=1)
            for c in conferences}


def recommended_modes() -> dict[str, str]:
    path = ROOT / "data" / "bracket_autobid_backtest.json"
    by_conf = json.loads(path.read_text())["byConference"]
    return {c: ("model" if s.get("beatsPlaceholder") else "placeholder") for c, s in by_conf.items()}


def as_of_dates(season: int) -> dict[str, str | None]:
    conf_start = bb_bracket.conference_tournaments(season)["start_date"].min()[:10]
    return {"feb1": f"{season}-02-01", "confStart": conf_start, "selectionSunday": None}


def grade(res: joint.Result, fmt, actual: pd.DataFrame, meetings, universe_filter=None) -> dict:
    t = res.teams
    in_field = t["team"].isin(set(actual["team"])).astype(float)
    p = t["pField"].clip(EPS, 1 - EPS)
    projected, _ = joint.projected_field(res, fmt)
    placement = regions.place(projected, meetings)
    problems = regions.check(projected, placement, meetings)
    real = actual.set_index("team")["seed"]
    both = projected[projected["team"].isin(real.index)]
    seed_err = (both["seedLine"] - both["team"].map(real)).abs()
    exp_err = (t.set_index("team").reindex(real.index)["expectedSeed"] - real).abs().dropna()
    return {
        "brier": float(((t["pField"] - in_field) ** 2).sum()),       # summed over teams: comparable across seasons
        "logLoss": float(-(in_field * np.log(p) + (1 - in_field) * np.log(1 - p)).sum()),
        "fieldHits": int(len(set(projected["team"]) & set(actual["team"]))),
        "fieldSize": int(len(actual)),
        "seedMAE": float(seed_err.mean()), "seedExact": float((seed_err == 0).mean()),
        "expectedSeedMAE": float(exp_err.mean()) if len(exp_err) else None,
        "regionProblems": len(problems), "linesMoved": len(placement.moved),
        "calibration": list(zip(t["pField"].round(4).tolist(), in_field.astype(int).tolist())),
    }


def one_season(season: int, modes: dict[str, str]) -> dict:
    games = history._canonical(cbbd.games(season), season)
    actual = history.field(season)
    if actual.empty:
        return {}
    table = pd.read_parquet(ROOT / "data" / "parquet" / "atlarge_history.parquet")
    table = table[table["conference"].notna()]
    train = [table[(table["season"] == s) & table["seed"].notna()] for s in table["season"].unique() if s != season]
    beta = atlarge.fit(train)
    fmt = seeding.FORMATS[68]
    noise = json.loads((ROOT / "data" / "atlarge_model.json").read_text()).get("committeeNoise", 0.0)
    meetings = regions.meetings_from_games(games[games["season_type"] == "regular"])
    out = {}
    for name, as_of in as_of_dates(season).items():
        ratings = history.team_ratings(season, through=as_of, games=games, with_resume=True)
        conf_of = {t: bb_registry.conference_of(t, season=season) for t in ratings.power.index}
        conf_of = {t: c for t, c in conf_of.items() if c}
        templates = templates_for(season, set(conf_of.values()))
        common = dict(season=season, games=games, power=ratings.power, home_field=ratings.home_field,
                      resume_sigma=ratings.sigma, conference_of=conf_of, beta=beta, fmt=fmt, as_of=as_of,
                      seed=season, committee_noise=noise)
        row = {}
        for variant, auto_mode, uncertainty in (("placeholder", {}, joint.RATING_UNCERTAINTY),
                                                ("recommended", modes, joint.RATING_UNCERTAINTY),
                                                ("recommendedExact", modes, 0.0)):
            res = joint.run(joint.Inputs(templates=templates, auto_mode=auto_mode, sims=SIMS,
                                         rating_uncertainty=uncertainty, **common))
            row[variant] = grade(res, fmt, actual, meetings)
        # "If the season ended today": no games left to play, no conference tournaments, each
        # standings leader takes the automatic bid - Phase 2's own answer on the day, and the bar.
        today = games if as_of is None else games[games["start_date"] < as_of]
        res = joint.run(joint.Inputs(templates={}, sims=1, rating_uncertainty=0.0,
                                     **{**common, "games": today, "committee_noise": 0.0}))
        row["endedToday"] = grade(res, fmt, actual, meetings)
        out[name] = row
    return out


def summarize(done: dict) -> dict:
    summary = {}
    for cp in CHECKPOINTS:
        summary[cp] = {}
        for variant in ("placeholder", "recommended", "recommendedExact", "endedToday"):
            rows = [done[s][cp][variant] for s in done if done[s] and cp in done[s]]
            if not rows:
                continue
            calib = [pair for r in rows for pair in r["calibration"]]
            buckets = []
            for lo in np.arange(0, 1.0, 0.1):
                sel = [y for p, y in calib if lo < p <= lo + 0.1] if lo > 0 else [y for p, y in calib if 0.0 < p <= 0.1]
                ps = [p for p, y in calib if (lo < p <= lo + 0.1 if lo > 0 else 0.0 < p <= 0.1)]
                if sel:
                    buckets.append({"range": f"{lo:.1f}-{lo + 0.1:.1f}", "n": len(sel),
                                    "meanP": round(float(np.mean(ps)), 3), "madeField": round(float(np.mean(sel)), 3)})
            summary[cp][variant] = {
                "seasons": len(rows),
                "brierPerSeason": round(float(np.mean([r["brier"] for r in rows])), 2),
                "logLossPerSeason": round(float(np.mean([r["logLoss"] for r in rows])), 2),
                "fieldHitRate": round(sum(r["fieldHits"] for r in rows) / sum(r["fieldSize"] for r in rows), 3),
                "seedMAE": round(float(np.mean([r["seedMAE"] for r in rows])), 2),
                "seedExact": round(float(np.mean([r["seedExact"] for r in rows])), 3),
                "regionProblems": int(sum(r["regionProblems"] for r in rows)),
                "linesMoved": int(sum(r["linesMoved"] for r in rows)),
                "calibration": buckets,
            }
    return summary


def main() -> None:
    done: dict = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else {}
    modes = recommended_modes()
    start = time.time()
    for season in SEASONS:
        if season in done:
            continue
        if time.time() - start > BUDGET:
            CACHE.write_bytes(pickle.dumps(done))
            print("budget reached; rerun to continue")
            return
        print(f"season {season}...", flush=True)
        done[season] = one_season(season, modes)
        CACHE.write_bytes(pickle.dumps(done))

    summary = summarize(done)
    print()
    for cp in CHECKPOINTS:
        print(f"== {cp} ==")
        for variant, s in summary[cp].items():
            print(f"  {variant:12s} Brier {s['brierPerSeason']:6.2f}/season  log loss {s['logLossPerSeason']:6.2f}  "
                  f"field {s['fieldHitRate']:.1%}  seed MAE {s['seedMAE']:.2f} (exact {s['seedExact']:.0%})  "
                  f"region problems {s['regionProblems']}  moves {s['linesMoved']}")
    per_season = {str(s): {cp: {v: {k: x for k, x in r.items() if k != "calibration"} for v, r in row.items()}
                           for cp, row in rows.items()} for s, rows in done.items() if rows}
    out = {"seasons": SEASONS, "sims": SIMS, "summary": summary, "bySeason": per_season,
           "recommendedModel": sorted(c for c, m in modes.items() if m == "model")}
    (ROOT / "data" / "bracketology_backtest.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
