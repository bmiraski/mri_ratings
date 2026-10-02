"""Convert the Drive workbook archive into Parquet tables.

Writes three files under data/parquet/:

  archive_games.parquet     every game, 2003-2019, one row per game, plus a ``neutral``
                            column (filled for 2004, 2005 and 2017; see ingest/archive_fixes) and
                            a ``classic_ready`` flag (False for added games that have no box score yet)
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

from mri.ingest import archive_fixes  # noqa: E402
from mri.ingest.archive import read_archive  # noqa: E402
from mri.ratings import classic  # noqa: E402


def main() -> None:
    archive_dir = ROOT / "data" / "archive"
    out_dir = ROOT / "data" / "parquet"
    out_dir.mkdir(parents=True, exist_ok=True)

    seasons = read_archive(archive_dir)
    if not seasons:
        raise SystemExit(f"no workbooks found in {archive_dir}")

    # Box scores for games added from CFBD are fetched once and kept in archive_games.parquet; a rebuild
    # without an API key reuses them instead of dropping them.
    existing = out_dir / "archive_games.parquet"
    known = pd.read_parquet(existing) if existing.exists() else None

    all_games, all_ratings, all_published = [], [], []
    for year, season in seasons.items():
        # Scores checked by hand against another source (data/archive_score_corrections.json).
        workbook = archive_fixes.apply_score_corrections(year, season.games)
        games, teams = classic.canonicalize(workbook, season.teams)
        games = games.assign(season=year)

        # Postseason games the workbook lacks (all of 2017's bowls). MRI 2.0 needs only the
        # scores; Classic also needs box scores, so it gets them only when they are in hand.
        extra = archive_fixes.missing_postseason(year, games, teams, known=known)
        classic_games = workbook
        if not extra.empty:
            print(f"  {year}: {len(extra)} postseason games missing from the workbook, added from CFBD")
            ready = archive_fixes.has_stats(extra)
            games["classic_ready"] = True
            games = pd.concat(
                [games, extra.drop(columns=["neutral", "game_id"]).assign(classic_ready=ready)],
                ignore_index=True,
            )
            if ready:
                classic_games = pd.concat(
                    [workbook, extra.drop(columns=["neutral", "game_id", "season"])], ignore_index=True
                )
            else:
                print(f"  {year}: Classic left WITHOUT these games (no box scores), and they are marked "
                      "classic_ready=False; re-run with CFBD_API_KEY set to include them")

        # Neutral-site flags from CFBD where the inferred rule is known to fail, and wherever
        # games were added. Left NaN elsewhere, which keeps the rule.
        games["neutral"] = pd.array([pd.NA] * len(games), dtype="boolean")
        if year in archive_fixes.NEUTRAL_FROM_CFBD or not extra.empty:
            games["neutral"] = archive_fixes.neutral_from_cfbd(year, games).astype("boolean").to_numpy()
        if "classic_ready" not in games.columns:
            games["classic_ready"] = True
        all_games.append(games)

        ratings = classic.compute(classic_games, season.teams).assign(season=year)
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
