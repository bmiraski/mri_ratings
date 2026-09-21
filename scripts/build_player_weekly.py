"""Season-to-date offensive stats at the weeks the Heisman backtest asks about.

Three calls (passing, rushing, receiving) for each season and snapshot week, using
the endpoint's ``endWeek`` to get totals through that week. Raw responses are cached
and ignored by git; what is kept is ``data/parquet/player_weekly.parquet``.

Run:  PYTHONPATH=src python3 scripts/build_player_weekly.py
Safe to interrupt and rerun: finished responses are cached.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.ingest import players  # noqa: E402

SEASONS = range(2012, 2026)
WEEKS = (3, 5, 7, 9, 11, 13)
BUDGET = 240          # seconds; the sandbox stops a command at 300


def main() -> None:
    out = ROOT / "data" / "parquet" / "player_weekly.parquet"
    have = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    done = set(zip(have["season"], have["week"])) if not have.empty else set()
    frames = [have] if not have.empty else []
    start = time.time()
    for year in SEASONS:
        for week in WEEKS:
            if (year, week) in done:
                continue
            if time.time() - start > BUDGET:
                break
            frames.append(players.weekly_table(year, week))
            done.add((year, week))
            print(f"  {year} week {week}: {len(frames[-1])} players", flush=True)
        else:
            continue
        break
    pd.concat(frames, ignore_index=True).to_parquet(out)
    total = len(SEASONS) * len(WEEKS)
    print(f"{len(done)} of {total} snapshots")


if __name__ == "__main__":
    main()
