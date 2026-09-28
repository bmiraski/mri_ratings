"""Persist MRI 2.0 end-of-season Power, prior and résumé for every season,
1978-present, on one consistent scale.

See ``mri.ratings.history`` for what this does and why - the CFBD/workbook/
current-era name-canonicalization, the anchoring choice, the two documented
seams (2003, where the CFBD-sourced backfill meets Ben's workbooks, and
``history.SEAM_YEAR``, where the archive walk meets ``current_ratings.parquet``),
the ``burn_in`` flag on 1978-1980, and why the 2020+ prior is a replica of
build_current.py's loop rather than a change to that script.

Run:  PYTHONPATH=src python3 scripts/build_coach_ratings_history.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.ratings import history  # noqa: E402

COLUMNS = ["team", "season", "power", "prior", "resume", "burn_in"]


def main() -> None:
    parquet_dir = ROOT / "data" / "parquet"
    pre2003_games = pd.read_parquet(parquet_dir / "pre2003_games.parquet")
    archive_games = pd.read_parquet(parquet_dir / "archive_games.parquet")
    current_games = pd.read_parquet(parquet_dir / "current_games.parquet")
    current_ratings = pd.read_parquet(parquet_dir / "current_ratings.parquet")

    print(f"walking {history.PRE2003_SEASONS.start}-{history.ARCHIVE_SEASONS.stop - 1} "
          f"({history.PRE2003_SEASONS.start}-{history.PRE2003_SEASONS.stop - 1} from CFBD, "
          f"{history.ARCHIVE_SEASONS.start}-{history.ARCHIVE_SEASONS.stop - 1} from the workbooks)...")
    canonical = history.combine_early_games(pre2003_games, archive_games)
    archive_history = history.archive_walk(canonical)

    early_seam = history.early_seam_gap(archive_history, canonical)
    print(f"  2003-2019 extended-chain Power vs. the old flat-2003-start walk: mean {early_seam.mean():+.2f}, "
          f"std {early_seam.std():.2f}, max abs {early_seam.abs().max():.2f} (expected - see "
          f"mri.ratings.history's docstring; the extended chain is now canonical)")

    burn_in_count = int(archive_history["burn_in"].sum())
    print(f"  {burn_in_count} team-seasons flagged burn_in "
          f"({history.BURN_IN_SEASONS.start}-{history.BURN_IN_SEASONS.stop - 1})")

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
    current["burn_in"] = False
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

    # A separate, site-facing table - see mri.ratings.history.is_scores_only's
    # docstring for why scores_only never touches Power or résumé, and keep
    # this schema isolated from coach_ratings_history.parquet's own consumers
    # rather than growing that file's column set underneath them.
    flags = history.incomplete_schedule_flags()
    mri2_history = full_history.copy()
    mri2_history["scores_only"] = mri2_history["season"].map(history.is_scores_only)
    mri2_history["incomplete_schedule"] = [
        (team, season) in flags for team, season in zip(mri2_history["team"], mri2_history["season"])
    ]
    mri2_path = parquet_dir / "mri2_history.parquet"
    mri2_history.to_parquet(mri2_path, index=False)
    flagged_count = int(mri2_history["incomplete_schedule"].sum())
    print(f"wrote {len(mri2_history)} team-seasons to {mri2_path.relative_to(ROOT)} "
          f"({flagged_count} flagged incomplete_schedule)")


if __name__ == "__main__":
    main()
