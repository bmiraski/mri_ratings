"""Does down-weighting old games lower walk-forward error? Football and basketball, separately.

Recency weighting is an option on the shared solver (``mri2.fit(recency_half_life=...)``),
so the same code serves both sports - but whether it helps, and at what half-life, is
decided per sport. The half-life is a fraction of the season, which is comparable across
a 12-game and a 30-game schedule and needs no dates.

Protocol, fixed in advance:

* half-life grid: none, 1.0, 0.6, 0.4, 0.25, 0.15 of a season;
* tuned *jointly* with the ridge (current value x0.5, x0.75, x1, x1.5), because weighting
  shrinks the effective sample and so strengthens the ridge relative to the data;
* objective is MAE of the predicted margin on the tuning block; accuracy and Brier ride along;
* the holdout is scored once, after the choice: the best recency setting on the tuning block
  against today's unweighted baseline. It is never searched.

Football tunes on 2003-2013 and reports 2014-2019; basketball tunes on 2022-2024 and
reports 2025-2026. Both walk the whole chain with the half-life applied, including the
end-of-season rating that primes the next season.

A third mode, ``basketball-extended``, is *supplementary*: the two-season basketball holdout
cannot settle much, so it scores the whole grid on 2009-2020 from the Classic games archive
(``bb_classic_games.parquet``; same API source, neutral-site flags from 2008 on). Those seasons are
earlier than the tuning block, took no part in the choice, and are reported as a second opinion -
not as a way to pick a different setting.

``football-modern`` is the same kind of second opinion for football: the whole grid on 2021-2025
from ``current_games.parquet`` (CFBD data with FCS opponents named individually; 2020 primes the
chain), a period the workbook archive does not cover. Row order there is chronological by week.

Run:  PYTHONPATH=src python3 scripts/tune_recency.py football
      PYTHONPATH=src python3 scripts/tune_recency.py basketball
      PYTHONPATH=src python3 scripts/tune_recency.py basketball-extended
      PYTHONPATH=src python3 scripts/tune_recency.py football-modern
"""

from __future__ import annotations

import functools
import itertools
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.ratings import backtest, bb_backtest as bb, mri2  # noqa: E402

warnings.simplefilter("ignore")  # early-season slices are legitimately disconnected

HALF_LIVES = [None, 1.0, 0.6, 0.4, 0.25, 0.15]
RIDGE_MULTIPLIERS = [0.5, 0.75, 1.0, 1.5]
CUTOFFS = (0.4, 0.5, 0.6, 0.7, 0.8)

FOOTBALL = {
    "tune": list(range(2003, 2014)),
    "holdout": list(range(2014, 2020)),
    "ridge": mri2.FOOTBALL_PROFILE.ridge,
}
BASKETBALL = {
    "chain": list(range(2021, 2027)),
    "tune": [2022, 2023, 2024],
    "holdout": [2025, 2026],
    "ridge": mri2.BASKETBALL_PROFILE.ridge,
}


def football_runner():
    games = pd.read_parquet(ROOT / "data" / "parquet" / "archive_games.parquet")

    def run(seasons, ridge, half_life) -> pd.DataFrame:
        frame = backtest.evaluate_archive(
            games, seasons=list(seasons), with_classic=False,
            ridge=ridge, recency_half_life=half_life,
        )
        return frame.rename(
            columns={"mri2_accuracy": "accuracy", "mri2_mae": "mae", "mri2_brier": "brier"}
        )

    return run


def basketball_runner(chain):
    # Parsing a season's cached JSON dominates the run time and never changes.
    prepare = functools.lru_cache(maxsize=None)(bb.prepare)
    bb.prepare = prepare

    def run(seasons, ridge, half_life) -> pd.DataFrame:
        return bb.evaluate(chain, scored=seasons, ridge=ridge, recency_half_life=half_life)

    return run


def extended_basketball() -> None:
    chain, scored = list(range(2008, 2021)), list(range(2009, 2021))
    classic = pd.read_parquet(ROOT / "data" / "parquet" / "bb_classic_games.parquet")
    cache: dict[int, pd.DataFrame] = {}

    def prepare(season: int) -> pd.DataFrame:
        if season not in cache:
            games = classic[classic["season"] == season].copy()
            for column in ("team1", "team2"):
                games[column] = [bb.registry.resolve(n, n, season=season) for n in games[column]]
            cache[season] = games.sort_values("start_date", kind="stable").reset_index(drop=True)
        return cache[season]

    bb.prepare = prepare
    started, frames = time.time(), []
    for i, (half_life, mult) in enumerate(itertools.product(HALF_LIVES, RIDGE_MULTIPLIERS), 1):
        ridge = BASKETBALL["ridge"] * mult
        frame = bb.evaluate(chain, scored=scored, ridge=ridge, recency_half_life=half_life)
        frames.append(frame.assign(half_life=half_life, ridge_mult=mult))
        print(f"  {i}/24  hl={half_life} ridge x{mult}  mae {frame['mae'].mean():.4f}  "
              f"({time.time() - started:.0f}s)", flush=True)
    long = pd.concat(frames, ignore_index=True)
    long.to_csv(ROOT / "data" / "recency_extended_basketball.csv", index=False)
    print(f"written data/recency_extended_basketball.csv ({time.time() - started:.0f}s)")


def modern_football() -> None:
    games = pd.read_parquet(ROOT / "data" / "parquet" / "current_games.parquet")
    games = games[(games["played"]) & (games["season"] <= 2025)].reset_index(drop=True)
    started, frames = time.time(), []
    for i, (half_life, mult) in enumerate(itertools.product(HALF_LIVES, RIDGE_MULTIPLIERS), 1):
        frame = backtest.evaluate_archive(
            games, seasons=list(range(2021, 2026)), with_classic=False,
            ridge=FOOTBALL["ridge"] * mult, recency_half_life=half_life,
        ).rename(columns={"mri2_accuracy": "accuracy", "mri2_mae": "mae", "mri2_brier": "brier"})
        frames.append(frame.assign(half_life=half_life, ridge_mult=mult))
        print(f"  {i}/24  hl={half_life} ridge x{mult}  mae {frame['mae'].mean():.4f}  "
              f"({time.time() - started:.0f}s)", flush=True)
    pd.concat(frames, ignore_index=True).to_csv(ROOT / "data" / "recency_modern_football.csv", index=False)


def summarize(frame: pd.DataFrame) -> dict:
    out = {
        "mae": frame["mae"].mean(),
        "accuracy": frame["accuracy"].mean(),
        "brier": frame["brier"].mean(),
    }
    for cut, part in frame.groupby("cutoff"):
        out[f"mae_c{int(round(cut * 100))}"] = part["mae"].mean()
    return out


def main(sport: str) -> None:
    if sport == "basketball-extended":
        return extended_basketball()
    if sport == "football-modern":
        return modern_football()
    if sport == "football":
        cfg, run = FOOTBALL, football_runner()
    elif sport == "basketball":
        cfg = BASKETBALL
        run = basketball_runner(cfg["chain"])
    else:
        raise SystemExit("usage: tune_recency.py football|basketball")

    base_ridge = cfg["ridge"]
    combos = list(itertools.product(HALF_LIVES, RIDGE_MULTIPLIERS))
    print(f"{sport}: {len(combos)} combinations on {cfg['tune']}")
    started = time.time()

    rows = []
    for i, (half_life, mult) in enumerate(combos, 1):
        ridge = base_ridge * mult
        frame = run(cfg["tune"], ridge, half_life)
        rows.append({"half_life": half_life, "ridge_mult": mult, "ridge": ridge, **summarize(frame)})
        print(f"  {i}/{len(combos)}  hl={half_life} ridge x{mult}  mae {rows[-1]['mae']:.4f}  "
              f"({time.time() - started:.0f}s)")

    grid = pd.DataFrame(rows)
    grid_path = ROOT / "data" / f"tuning_recency_{sport}.csv"
    grid.to_csv(grid_path, index=False)

    show = grid.assign(half_life=grid["half_life"].fillna(float("inf"))).sort_values("mae")
    print("\nbest eight on the tuning seasons (inf = unweighted):")
    print(show.head(8).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    base_tune = grid[(grid["half_life"].isna()) & (grid["ridge_mult"] == 1.0)].iloc[0]
    recent = grid[grid["half_life"].notna()].sort_values("mae").iloc[0]
    print(f"\nbaseline on tuning block (unweighted, ridge x1): MAE {base_tune['mae']:.4f}")
    print(f"best recency setting: half-life {recent['half_life']}, ridge x{recent['ridge_mult']}  "
          f"MAE {recent['mae']:.4f}  (tune gain {base_tune['mae'] - recent['mae']:+.4f})")
    overall = grid.sort_values("mae").iloc[0]
    if pd.isna(overall["half_life"]):
        print(f"note: overall best on the tuning block is unweighted at ridge x{overall['ridge_mult']}")

    print(f"\nholdout {cfg['holdout']}, scored once:")
    baseline = run(cfg["holdout"], base_ridge, None).assign(setting="baseline")
    chosen = run(
        cfg["holdout"], recent["ridge"], float(recent["half_life"])
    ).assign(setting=f"hl={recent['half_life']} ridge x{recent['ridge_mult']}")
    both = pd.concat([baseline, chosen], ignore_index=True)
    both.to_csv(ROOT / "data" / f"recency_holdout_{sport}.csv", index=False)
    print(pd.DataFrame({"baseline": summarize(baseline), "chosen": summarize(chosen)})
          .to_string(float_format=lambda v: f"{v:.4f}"))

    per_season = pd.DataFrame(
        {
            "baseline_mae": baseline.groupby("season")["mae"].mean(),
            "chosen_mae": chosen.groupby("season")["mae"].mean(),
        }
    )
    per_season["gain"] = per_season["baseline_mae"] - per_season["chosen_mae"]
    print("\nper season (gain > 0 = recency better):")
    print(per_season.to_string(float_format=lambda v: f"{v:+.4f}"))
    print(f"\ngrid written to {grid_path.relative_to(ROOT)} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
