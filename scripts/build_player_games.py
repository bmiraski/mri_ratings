"""Every notable offensive game of every season since 2012, one row per player per game.

For testing whether how a player did against good teams tells you anything the season's totals
do not. One call per season-week (about 2 MB each, ignored by git); the derived table is
``data/parquet/player_games.parquet``. Safe to interrupt and rerun.

Run:  PYTHONPATH=src python3 scripts/build_player_games.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.heisman import data  # noqa: E402
from mri.ingest import cfbd, players  # noqa: E402

SEASONS = range(2012, data.last_season() + 1)
WEEKS = range(1, 16)


def main() -> None:
    rows = []
    for year in SEASONS:
        before = len(rows)
        fbs = set(cfbd.fbs_teams(year)["team"])              # from 2022 the feed includes FCS games as well
        for week in WEEKS:
            rows.extend(r for r in players.game_rows(year, week) if r["team"] in fbs)
        print(f"  {year}: {len(rows) - before} player-games", flush=True)
    table = pd.DataFrame(rows)
    out = ROOT / "data" / "parquet" / "player_games.parquet"
    table.to_parquet(out)
    print(f"{len(table):,} player-games, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
