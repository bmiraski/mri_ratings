"""Compute MRI Basketball Classic for the seasons Ben never ran.

The basketball workbooks in this repository cover 2012-13, 2018-19 and 2019-20
(2017-18 exists but is a December snapshot, not a season). Everything from
2004-05 forward that those three do not cover is reachable from the API, which
carries rebounds and turnovers per game - exactly what Classic needs and what
the games feed alone cannot provide.

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

from mri.ingest import bb_gamelog, bb_registry as registry  # noqa: E402
from mri.ratings import classic  # noqa: E402

OUT = ROOT / "data" / "parquet" / "bb_classic.parquet"

# Everything the API can support that Ben's workbooks in this repo do not cover.
#
# The games feed reaches back to 2000-01, but Classic needs rebounds and
# turnovers, which live in the box scores, and those do not really start until
# 2004-05. The four seasons before it are not thin, they are empty: 2000-01 and
# 2001-02 return no box-score rows at all, 2002-03 returns one game and 2003-04
# twelve. A rating built on twelve games is not a worse rating, it is a
# different thing wearing the same name, so the floor is set where the data
# actually begins and the earlier years are left for the workbooks.
FIRST_WITH_BOX_SCORES = 2005
MIN_GAMES = 2000  # a real Division I season is 5,000-plus; this only catches ruins
DEFAULT_FIRST, DEFAULT_LAST = 2005, 2018

# Seasons Ben published himself. Computing these would silently replace his own
# ratings with a reconstruction of them, which is a worse answer even when the
# numbers agree.
PUBLISHED = {2013, 2019, 2020}


def compute(season: int) -> pd.DataFrame:
    games = bb_gamelog.for_season(season)
    if len(games) < MIN_GAMES:
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
        if season in PUBLISHED:
            print(f"  {season - 1}-{str(season)[2:]}: Ben's own workbook, left alone")
            continue
        table = compute(season)
        if table.empty:
            print(f"  {season - 1}-{str(season)[2:]}: too few box scores to rate")
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
