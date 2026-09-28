"""Pull 1978-2002 games from CFBD - the seasons before Ben's workbooks start.

MRI 2.0's historical backfill runs the chain back to 1978, the first season of
the Division I-A split. Ben's Excel archive only goes back to 2003, so those
25 seasons come from the API instead, the same source ``build_current.py``
already uses for 2020+: one call per season (``cfbd.games`` defaults to both
regular season and postseason), FCS opponents named individually rather than
pooled, and neutral sites flagged rather than inferred.

Division membership needs no hand-maintained file - checked directly against
the API for a spread of years (1978, 1985, 1993, 1998, 2002), ``/teams/fbs``
returns real, plausible rosters with no anachronistic teams, and
``registry.was_fbs``/``fbs_members`` already falls back to a team's own CFBD
name when the current registry doesn't recognize it (Brown, Colgate, and 29
other programs that were I-A in 1978 but have since left it). This script
only pulls
and caches the games; it does not fit anything - see ``mri.ratings.history``
for the rating walk that will consume ``pre2003_games.parquet``.

Run:  PYTHONPATH=src python3 scripts/build_pre2003_games.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.ingest import cfbd, registry  # noqa: E402

PRE2003_SEASONS = range(1978, 2003)


def canonical(frame: pd.DataFrame) -> pd.DataFrame:
    """Put every FBS name into its registry spelling; leave FCS/former-FBS names alone."""
    frame = frame.copy()
    for column in ("team1", "team2"):
        frame[column] = [registry.resolve(name, name) for name in frame[column]]
    return frame


def main() -> None:
    out = ROOT / "data" / "parquet"
    out.mkdir(parents=True, exist_ok=True)

    seasons = []
    previous_fbs = None
    for season in PRE2003_SEASONS:
        games = canonical(cfbd.games(season))
        if games.empty:
            print(f"  {season}: no games returned")
            continue

        fbs = registry.fbs_members(season)
        jump = f" ({len(fbs) - previous_fbs:+d})" if previous_fbs is not None and fbs else ""
        neutral = int(games["neutral"].sum())
        print(f"  {season}: {len(games):>4} games, {len(fbs):>3} FBS teams{jump}, {neutral} neutral-site")
        if fbs:
            previous_fbs = len(fbs)

        seasons.append(games)

    frame = pd.concat(seasons, ignore_index=True)
    path = out / "pre2003_games.parquet"
    frame.to_parquet(path)
    print(f"\nwrote {len(frame)} games, {PRE2003_SEASONS.start}-{PRE2003_SEASONS.stop - 1}, "
          f"to {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
