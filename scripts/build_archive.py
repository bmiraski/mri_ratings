"""Convert the Drive workbook archive into Parquet tables.

Writes three files under data/parquet/:

  archive_games.parquet     every game, 2003-2019, one row per game
  archive_ratings.parquet   MRI Classic recomputed for each season
  archive_published.parquet the ratings as originally published, for auditing

Run:  PYTHONPATH=src python3 scripts/build_archive.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.ingest.archive import read_archive  # noqa: E402
from mri.ratings import classic  # noqa: E402


def main() -> None:
    archive_dir = ROOT / "data" / "archive"
    out_dir = ROOT / "data" / "parquet"
    out_dir.mkdir(parents=True, exist_ok=True)

    seasons = read_archive(archive_dir)
    if not seasons:
        raise SystemExit(f"no workbooks found in {archive_dir}")

    all_games, all_ratings, all_published = [], [], []
    for year, season in seasons.items():
        games, teams = classic.canonicalize(season.games, season.teams)
        games = games.assign(season=year)
        all_games.append(games)

        ratings = classic.compute(season.games, season.teams).assign(season=year)
        all_ratings.append(ratings)
        all_published.append(season.published.assign(season=year))

        worst = (
            ratings.set_index("team")["mri"]
            .sub(season.published.set_index("team")["mri"])
            .abs()
            .max()
        )
        print(f"  {year}: {len(games):>4} games, {len(ratings):>3} teams, max deviation {worst:.2e}")

    pd.concat(all_games, ignore_index=True).to_parquet(out_dir / "archive_games.parquet")
    pd.concat(all_ratings, ignore_index=True).to_parquet(out_dir / "archive_ratings.parquet")
    pd.concat(all_published, ignore_index=True).to_parquet(out_dir / "archive_published.parquet")

    total = sum(len(g) for g in all_games)
    print(f"\nwrote {len(seasons)} seasons / {total} games to {out_dir}")


if __name__ == "__main__":
    main()
