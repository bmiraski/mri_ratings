"""Persist MRI 2.0 end-of-season Power, prior and résumé for every season,
2003-present, on one consistent scale.

See ``mri.ratings.history`` for what this does and why - the archive/CFBD
name-canonicalization fix, the anchoring choice, the documented seam at
``history.SEAM_YEAR`` where the archive walk meets ``current_ratings.parquet``,
and why the 2020+ prior is a replica of build_current.py's loop rather than a
change to that script.

Run:  PYTHONPATH=src python3 scripts/build_coach_ratings_history.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.ratings import history  # noqa: E402

COLUMNS = ["team", "season", "power", "prior", "resume"]


def main() -> None:
    parquet_dir = ROOT / "data" / "parquet"
    archive_games = pd.read_parquet(parquet_dir / "archive_games.parquet")
    current_games = pd.read_parquet(parquet_dir / "current_games.parquet")
    current_ratings = pd.read_parquet(parquet_dir / "current_ratings.parquet")

    print(f"walking {history.ARCHIVE_SEASONS.start}-{history.ARCHIVE_SEASONS.stop - 1} from the archive...")
    canonical = history.canonical_games(archive_games)
    archive_history = history.archive_walk(canonical)

    seam = history.seam_gap(archive_history, canonical)
    print(f"  {history.SEAM_YEAR - 1} vs. build_current.py's no-prior prime: mean {seam.mean():+.2f}, "
          f"std {seam.std():.2f}, max abs {seam.abs().max():.2f} (see mri.ratings.history's docstring - "
          f"this is the expected seam at {history.SEAM_YEAR}, not a bug)")

    print("  replicating build_current.py's 2020+ prior walk...")
    archive_2019 = canonical[canonical["season"] == history.SEAM_YEAR - 1]
    prior_walk = history.current_prior_walk(archive_2019, current_games)
    replica_gap = history.current_prior_walk_gap(prior_walk, current_ratings)
    print(f"  replica power vs. current_ratings.parquet: mean {replica_gap.mean():+.4f}, "
          f"max abs {replica_gap.abs().max():.4f} (should be ~0 - a real gap means the replica "
          f"has drifted from what build_current.py actually does)")

    current = current_ratings[["team", "season", "power", "resume"]].merge(
        prior_walk[["team", "season", "prior"]], on=["team", "season"], how="left"
    )
    missing_prior = current["prior"].isna().sum()
    if missing_prior:
        print(f"  {missing_prior} 2020+ team-season(s) have no matching prior from the replica walk")

    full_history = pd.concat(
        [archive_history[COLUMNS], current[COLUMNS]], ignore_index=True
    ).sort_values(["season", "team"])
    out_path = parquet_dir / "coach_ratings_history.parquet"
    full_history.to_parquet(out_path, index=False)

    seasons = sorted(full_history["season"].unique())
    print(f"\nwrote {len(full_history)} team-seasons, {seasons[0]}-{seasons[-1]}, to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
