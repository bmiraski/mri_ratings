"""Compute MRI Basketball Classic for the seasons Ben never ran.

His basketball workbooks cover 2012-13, 2018-19 and 2019-20 (2017-18 exists but
is a December snapshot, not a season). That leaves 2013-14 through 2017-18
missing entirely, and those years are now reachable: the API carries rebounds
and turnovers per game back to at least 2013-14, which is exactly what Classic
needs and what the games feed alone cannot provide.

These ratings are computed, not published. Ben's own workbook seasons are what
he put out at the time and the archive says so; these are what his formula says
about seasons he did not run it on. The distinction is carried into the payload
rather than left for a reader to infer, because "MRI Classic, 2015-16" reads
like a historical record either way.

Run:  PYTHONPATH=src python3 scripts/build_bb_classic.py [first] [last]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.ingest import bb_registry as registry, cbbd  # noqa: E402
from mri.ratings import classic  # noqa: E402

OUT = ROOT / "data" / "parquet" / "bb_classic.parquet"

# The gap between the 2012-13 workbook and the 2018-19 one.
DEFAULT_FIRST, DEFAULT_LAST = 2014, 2018


def compute(season: int) -> pd.DataFrame:
    games = cbbd.classic_table(season)
    if games.empty:
        return pd.DataFrame()

    games = games.copy()
    for column in ("team1", "team2"):
        games[column] = [
            registry.resolve(n, str(n), season=season) for n in games[column]
        ]
    # Division I only. Non-D1 opponents still contribute to opponent records -
    # that is what the pooled row did in football - but are not rated and take no
    # part in the z-score statistics.
    rated = sorted(
        {t for t in set(games["team1"]) | set(games["team2"])
         if registry.is_d1(t, season=season)}
    )
    table = classic.compute(games, rated, sport=classic.BASKETBALL)
    return table.assign(season=season)


def main() -> None:
    first = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FIRST
    last = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_LAST

    frames = []
    if OUT.exists():
        existing = pd.read_parquet(OUT)
        frames.append(existing[~existing["season"].between(first, last)])

    for season in range(first, last + 1):
        table = compute(season)
        if table.empty:
            print(f"  {season - 1}-{str(season)[2:]}: no usable games")
            continue
        best = table.iloc[0]
        print(
            f"  {season - 1}-{str(season)[2:]}: {len(table):>3} D1 teams, "
            f"top: {best['team']} ({best['mri']:.2f}, "
            f"{int(best['wins'])}-{int(best['losses'])})"
        )
        frames.append(table)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(OUT)
    seasons = sorted(combined["season"].unique())
    print(f"\nwrote {len(combined)} rows for seasons {seasons} to "
          f"{OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
