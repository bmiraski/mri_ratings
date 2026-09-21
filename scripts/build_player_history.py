"""Collect every notable player of every season since 2009 into one table.

Five calls a season for the counting stats (passing, rushing, receiving,
defensive, interceptions) and two more from 2013 for PPA and usage. The raw
responses are large - the defensive one alone is nine megabytes a season - and
are ignored by git; what is kept is ``data/parquet/player_seasons.parquet``, the
few thousand players a season anyone could have voted for.

Run:  PYTHONPATH=src python3 scripts/build_player_history.py [first_season]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.heisman import data  # noqa: E402
from mri.ingest import players  # noqa: E402


def main() -> None:
    first = int(sys.argv[1]) if len(sys.argv) > 1 else players.FIRST_SEASON
    last = data.last_season()                          # the last season whose Heisman has been recorded
    frames = []
    for year in range(first, last + 1):
        frame = players.season_table(year)
        frames.append(frame)
        print(f"  {year}: {len(frame):5d} players kept", flush=True)
    out = ROOT / "data" / "parquet" / "player_seasons.parquet"
    table = pd.concat(frames, ignore_index=True)
    table.to_parquet(out)
    print(f"{len(table):,} player-seasons, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
