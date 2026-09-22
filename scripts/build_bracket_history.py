"""Backtest: could the conference-tournament simulation have called the automatic
bid, and does it beat the placeholder ("the #1 seed wins") that's live today?

For each conference in each season 2011-2025 (2020 cancelled): infer that
conference's bracket template from its three seasons *before* the one being
predicted (so nothing about the target season's own bracket leaks in), build
its regular-season standings and end-of-regular-season power ratings for the
target season itself, and simulate the bracket forward. Graded against what
actually happened, next to the placeholder, conference by conference - the
switch from placeholder to real model is a per-conference decision on these
numbers, not an automatic one.

Run:  PYTHONPATH=src python3 scripts/build_bracket_history.py
Resumable (saves its cache as it goes; API responses are cached separately by
the ingest layer, so a rerun after an interruption is cheap).
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

from mri.bracket import history, simulate, template as template_mod  # noqa: E402
from mri.ingest import bb_bracket, cbbd  # noqa: E402

SEASONS = [y for y in range(2011, 2026) if y != 2020]
TEMPLATE_LOOKBACK = 3
CACHE = Path("/tmp/bracket_history_cache.pkl")
BUDGET = 250
EPS = 1e-6


def infer_conference_template(conference: str, before_season: int) -> template_mod.Template | None:
    games = [bb_bracket.conference_tournaments(y) for y in SEASONS if before_season - TEMPLATE_LOOKBACK <= y < before_season]
    games = [g[g["conference"] == conference] for g in games if not g.empty]
    return template_mod.infer(games, min_seasons=1)


def log_loss(p: float) -> float:
    return -np.log(max(p, EPS))


def main() -> None:
    done: dict = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else {}
    start = time.time()
    for season in SEASONS:
        if season in done:
            continue
        if time.time() - start > BUDGET:
            print("budget reached; rerun to continue")
            CACHE.write_bytes(pickle.dumps(done))
            return
        print(f"season {season}...", flush=True)
        f = history.field(season)
        autos = dict(zip(f[f["bidType"] == "auto"]["conference"], f[f["bidType"] == "auto"]["team"]))
        if not autos:
            done[season] = {}
            continue
        games = cbbd.games(season)
        # Ratings must stop before any conference tournament game, or a conference's own bracket
        # partly rates the very teams whose path through it is being simulated (a hot upset winner's
        # rating would already reflect the upset). The earliest conference tournament tip-off of the
        # season is a safe, uniform cutoff for every conference at once.
        conf_start = bb_bracket.conference_tournaments(season)["start_date"].min()
        power, home_field = history.team_power(season, through=conf_start, games=games)
        season_rows = {}
        for conf, champ in autos.items():
            standings_order = history.standings(season, conf, games)
            tpl = infer_conference_template(conf, season)
            if tpl is None:
                season_rows[conf] = {"skipped": True, "reason": "no template", "fieldSize": len(standings_order)}
                continue
            # A good many conferences invite only their top K standings teams to the tournament (K =
            # the template's size), dropping the rest of the league. The top-K slice IS the field then,
            # not a mismatch to skip over - only flag when there still isn't a clean field to seed with.
            seeds = standings_order[:tpl.size] if len(standings_order) >= tpl.size else standings_order
            if tpl.size != len(seeds) or champ not in seeds:
                season_rows[conf] = {"skipped": True, "reason": "too few standings teams" if tpl.size != len(seeds) else
                                     "champion outside the inferred top-K (a seeding tiebreak likely put it just over the line)",
                                     "fieldSize": len(standings_order), "templateSize": tpl.size}
                continue
            rated = [power.get(t, np.nan) for t in seeds]
            if any(pd.isna(r) for r in rated):
                season_rows[conf] = {"skipped": True, "reason": "unrated team", "fieldSize": len(seeds)}
                continue
            home_edge = home_field if tpl.campus_hosted else 0.0
            probs = simulate.simulate(tpl, seeds, dict(zip(seeds, rated)), home_edge=home_edge, sims=8000, seed=season)
            season_rows[conf] = {
                "champion": champ, "seedOfChampion": seeds.index(champ) + 1, "fieldSize": len(seeds),
                "templateTiers": tpl.tiers, "templateStable": tpl.stable, "campusHosted": tpl.campus_hosted,
                "modelProbability": probs[champ], "topSeedIsChampion": seeds[0] == champ,
                "modelPickedRight": max(probs, key=probs.get) == champ,
            }
        done[season] = season_rows
        CACHE.write_bytes(pickle.dumps(done))
    CACHE.write_bytes(pickle.dumps(done))

    by_conf: dict[str, list[dict]] = {}
    for season, rows in done.items():
        for conf, r in rows.items():
            by_conf.setdefault(conf, []).append({"season": season, **r})

    # Log loss alone is a weak test here: the placeholder is a deterministic 100%-or-nothing call, so
    # any model that spreads probability at all beats it on log loss almost everywhere the top seed
    # doesn't win outright every single time - that isn't the same as the model actually calling more
    # winners correctly. Both numbers are kept, and the switch condition below asks for both: a real
    # log-loss margin AND a same-or-better hit rate on the model's own top pick.
    MIN_LOGLOSS_MARGIN = 0.3
    summary = {}
    for conf, rows in by_conf.items():
        graded = [r for r in rows if not r.get("skipped")]
        if not graded:
            summary[conf] = {"seasons": len(rows), "graded": 0}
            continue
        model_ll = np.mean([log_loss(r["modelProbability"]) for r in graded])
        placeholder_ll = np.mean([log_loss(1.0 if r["topSeedIsChampion"] else EPS) for r in graded])
        model_acc = np.mean([r["modelPickedRight"] for r in graded])
        top_seed_acc = np.mean([r["topSeedIsChampion"] for r in graded])
        summary[conf] = {
            "seasons": len(rows), "graded": len(graded),
            "modelLogLoss": round(float(model_ll), 3), "placeholderLogLoss": round(float(placeholder_ll), 3),
            "modelAccuracy": round(float(model_acc), 3), "placeholderAccuracy": round(float(top_seed_acc), 3),
            "avgFieldSize": round(float(np.mean([r["fieldSize"] for r in graded])), 1),
            "beatsPlaceholder": bool(placeholder_ll - model_ll >= MIN_LOGLOSS_MARGIN and model_acc >= top_seed_acc - EPS),
        }
    beats = sum(v.get("beatsPlaceholder", False) for v in summary.values())
    print(f"\n{beats} of {len(summary)} conferences: the simulation clears both bars (log loss and accuracy) against the placeholder")
    print("Sample sizes are small (n=2-12 graded seasons per conference) - read the pattern, not any one conference's exact number.\n")
    ranked = sorted(summary.items(), key=lambda kv: -(kv[1].get("modelAccuracy", 0) - kv[1].get("placeholderAccuracy", 0)))
    for conf, s in ranked:
        if s.get("graded"):
            flag = "switch candidate" if s["beatsPlaceholder"] else ("tie" if s["modelAccuracy"] == s["placeholderAccuracy"] else "keep placeholder")
            print(f"  {conf:12s} n={s['graded']:2d}  model right {s['modelAccuracy']:5.0%}  placeholder right {s['placeholderAccuracy']:5.0%}"
                  f"   |  log loss {s['modelLogLoss']:.2f} vs {s['placeholderLogLoss']:.2f}  [{flag}]")
        else:
            print(f"  {conf:12s} not enough seasons graded")

    out = {"seasons": SEASONS, "byConference": summary, "raw": {str(k): v for k, v in done.items()}}
    (ROOT / "data" / "bracket_autobid_backtest.json").write_text(json.dumps(out, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
