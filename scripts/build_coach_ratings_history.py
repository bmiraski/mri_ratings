"""Persist MRI 2.0 end-of-season Power for every season, 2003-present, on one consistent scale.

See ``mri.ratings.history`` for what this does and why - the archive/CFBD
name-canonicalization fix, the anchoring choice, and the documented seam at
``history.SEAM_YEAR`` where the archive walk meets ``current_ratings.parquet``.

Run:  PYTHONPATH=src python3 scripts/build_coach_ratings_history.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.ratings import history  # noqa: E402


def main() -> None:
    parquet_dir = ROOT / "data" / "parquet"
    archive_games = pd.read_parquet(parquet_dir / "archive_games.parquet")
    current_ratings = pd.read_parquet(parquet_dir / "current_ratings.parquet")

    print(f"walking {history.ARCHIVE_SEASONS.start}-{history.ARCHIVE_SEASONS.stop - 1} from the archive...")
    canonical = history.canonical_games(archive_games)
    archive_history = history.archive_walk(canonical)

    gap = history.seam_gap(archive_history, canonical)
    print(f"  {history.SEAM_YEAR - 1} vs. build_current.py's no-prior prime: mean {gap.mean():+.2f}, "
          f"std {gap.std():.2f}, max abs {gap.abs().max():.2f} (see mri.ratings.history's docstring - "
          f"this is the expected seam at {history.SEAM_YEAR}, not a bug)")

    print(f"  bridging in {int(current_ratings['season'].min())}-{int(current_ratings['season'].max())} "
          f"from current_ratings.parquet")

    current = current_ratings[["team", "season", "power"]]
    full_history = pd.concat([archive_history, current], ignore_index=True).sort_values(["season", "team"])
    out_path = parquet_dir / "coach_ratings_history.parquet"
    full_history.to_parquet(out_path, index=False)

    seasons = sorted(full_history["season"].unique())
    print(f"\nwrote {len(full_history)} team-seasons, {seasons[0]}-{seasons[-1]}, to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
