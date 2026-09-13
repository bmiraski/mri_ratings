"""Rate the basketball seasons Ben missed, and the current one.

His workbooks stop at 2019-20. This chains MRI 2.0 forward from there through
2020-21 to 2025-26, so each season starts from a real prior rather than a
guess, and writes the ratings alongside football's.

Run:  PYTHONPATH=src python3 scripts/build_basketball.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.ingest import bb_registry as registry  # noqa: E402
from mri.ratings import bb_backtest as bb, mri2  # noqa: E402

WARMUP = 2020          # 2019-20, the last season Ben ran himself
SEASONS = range(2021, 2027)


def main() -> None:
    out = ROOT / "data" / "parquet"
    out.mkdir(parents=True, exist_ok=True)

    warmup = bb.prepare(WARMUP)
    prior = bb.fit_slice(warmup, None, season=WARMUP).power
    print(f"primed from {WARMUP - 1}-{str(WARMUP)[2:]} ({len(warmup)} games)")

    seasons, ratings = [], []
    for season in SEASONS:
        games = bb.prepare(season)
        if games.empty:
            print(f"  {season}: no games")
            continue

        teams = sorted(set(games["team1"]) | set(games["team2"]))
        rated = [t for t in teams if registry.is_d1(t, season=season)]
        model = mri2.fit(
            games,
            prior=mri2.build_prior(
                prior, teams, mri2.BASKETBALL_PROFILE.prior_regression, centre_teams=rated
            ),
            neutral=games["neutral"],
            anchor_teams=rated,
            compression=mri2.BASKETBALL_PROFILE.compression,
            ridge=mri2.BASKETBALL_PROFILE.ridge,
            home_field_prior=mri2.BASKETBALL_PROFILE.home_field_prior,
            with_efficiency=False,
        )

        table = model.table().assign(season=season)
        d1 = table[table["team"].map(lambda t: registry.is_d1(t, season=season))]
        best = d1.iloc[0]
        print(
            f"  {season - 1}-{str(season)[2:]}: {len(games):>5} games, {len(d1):>3} D1 teams, "
            f"HCA {model.home_field:4.2f}, top: {best['team']} ({best['power']:+.1f})"
        )

        seasons.append(games.assign(season=season))
        ratings.append(table)
        prior = model.power

    pd.concat(seasons, ignore_index=True).to_parquet(out / "bb_games.parquet")
    pd.concat(ratings, ignore_index=True).to_parquet(out / "bb_ratings.parquet")

    current = ratings[-1]
    current = current[current["team"].map(lambda t: registry.is_d1(t, season=2026))].head(20)
    print("\n2025-26 top 20 by Power:\n")
    print(
        current[["rank", "team", "power", "resume", "resume_rank", "games"]].to_string(
            index=False, float_format=lambda v: f"{v:7.2f}"
        )
    )


if __name__ == "__main__":
    main()
