"""Add the coaches-hot-seat-plan's §2 metrics to coach_season.parquet, and write tenure summaries.

Run:  PYTHONPATH=src python3 scripts/build_coach_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.coaches import metrics  # noqa: E402
from mri.ratings import history  # noqa: E402


def main() -> None:
    parquet_dir = ROOT / "data" / "parquet"
    coach_season = pd.read_parquet(parquet_dir / "coach_season.parquet")
    ratings_history = pd.read_parquet(parquet_dir / "coach_ratings_history.parquet")
    archive_games = history.canonical_games(pd.read_parquet(parquet_dir / "archive_games.parquet"))
    current_games = pd.read_parquet(parquet_dir / "current_games.parquet")

    print(f"computing metrics for {len(coach_season)} coach-seasons...")
    result = metrics.build(coach_season, ratings_history, archive_games, current_games)
    table, tenure, refits = result["metrics"], result["tenure"], result["refits"]

    print(f"  {len(refits)} split-season departure(s) got a point-in-time power_end refit")

    for column in metrics.METRIC_COLUMNS:
        present = table[column].notna().sum()
        print(f"  {column:14} {present:>4}/{len(table)} rows ({present / len(table):.0%})")

    metrics_path = parquet_dir / "coach_season.parquet"
    table.to_parquet(metrics_path, index=False)
    print(f"\nwrote {len(table)} rows (with metrics) to {metrics_path.relative_to(ROOT)}")

    tenure_path = parquet_dir / "coach_tenure.parquet"
    tenure.to_parquet(tenure_path, index=False)
    career = tenure[tenure["school"].isna()]
    print(f"wrote {len(tenure)} rows ({len(career)} career, {len(tenure) - len(career)} per-stop) "
          f"to {tenure_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
