"""Collect the player-level basketball data a roster-aware prior is fitted on.

For every season since 2010: who played for which team, for how many minutes and
with how many win shares (one API call a season); each recruiting class and its
commitments; and each NBA draft, so departures can be told from transfers.

The raw feeds are large and ignored by git. What is kept is the derived tables -
a few megabytes for every season - in data/parquet.

Run:  PYTHONPATH=src python3 scripts/build_bb_players.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.ingest import bb_registry, cbbd  # noqa: E402

FIRST = 2010
LAST = bb_registry.CURRENT_SEASON - 1          # the last season that has been played


def main() -> None:
    out = ROOT / "data" / "parquet"
    players = pd.concat([cbbd.player_seasons(s) for s in range(FIRST, LAST + 1)], ignore_index=True)
    recruits = pd.concat([cbbd.recruits(y) for y in range(FIRST - 1, LAST + 1)], ignore_index=True)
    draft = pd.concat([cbbd.draft_picks(y) for y in range(FIRST, LAST + 1)], ignore_index=True)
    players.to_parquet(out / "bb_players.parquet")
    recruits.to_parquet(out / "bb_recruits.parquet")
    draft.to_parquet(out / "bb_draft.parquet")
    print(f"{len(players):,} player-seasons over {players['season'].nunique()} seasons; "
          f"{len(recruits):,} recruits; {len(draft):,} draft picks")


if __name__ == "__main__":
    main()
