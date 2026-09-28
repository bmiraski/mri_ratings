"""Walk-forward accuracy of the extended 1978-2019 chain, by era.

The backfill plan's own hypothesis: accuracy in the older eras should be
similar to or better than 2003-2019 (mismatches were bigger then), and a
sharp drop points to a data gap - a missing game, a wrong division call -
rather than a model failure. This scores exactly that, using the same
walk-forward method ``mri.ratings.backtest`` already validated MRI 2.0
with (``evaluate_season``, unmodified), but running its own outer season
loop rather than ``backtest.evaluate_archive``'s: that function doesn't
anchor between seasons, which is fine chaining the 17 unanchored seasons
(2003-2019) it was built for, but would let the rating scale drift over the
42 seasons this backfill chains (see ``mri2.fit``'s own docstring on what an
unanchored FCS pool does to it). Anchoring every season to its own FBS field
via ``registry.was_fbs`` - what ``mri.ratings.history.archive_walk`` already
does for the persisted ratings - keeps that drift from being mistaken for a
genuine accuracy problem in the older eras.

MRI Classic needs rushing/passing/turnover box scores, which CFBD's /games
endpoint (the source for 1978-2002, and for 2020+ - current_games.parquet
has never carried them either) doesn't return. So Classic is only scored for
2003-2019, which is workbook-sourced and always carried box scores; the two
CFBD eras are scored on MRI 2.0 alone. Burn-in seasons (1978-1980) are
walked, to keep the prior chain real, but never scored - they simply aren't
named in ``ERAS``.

Run:  PYTHONPATH=src python3 scripts/validate_backfill.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.ingest import registry  # noqa: E402
from mri.ratings import backtest, history, mri2  # noqa: E402

ERAS = {
    "1981-1990": range(1981, 1991),
    "1991-2002": range(1991, 2003),
    "2003-2019": range(2003, 2020),
}

# Unlike history.combine_early_games (which never needs box scores - the
# persisted walk always fits with with_efficiency=False), Classic needs its
# rushing/passing/turnover columns, so this keeps them: NaN for the pre2003
# rows, real for the workbook rows. Never mixed within one season's slice,
# since a season is 100% one source or the other.
_VALIDATION_COLUMNS = ["team1", "team2", "pts1", "pts2", "win1", "win2", "season", "neutral",
                       "rush1", "rush2", "pass1", "pass2", "to1", "to2"]


def combine_for_validation(pre2003_games: pd.DataFrame, archive_games: pd.DataFrame) -> pd.DataFrame:
    pre2003 = pre2003_games[[c for c in _VALIDATION_COLUMNS if c in pre2003_games.columns]]
    workbook = archive_games[[c for c in _VALIDATION_COLUMNS if c in archive_games.columns]]
    return history.canonical_games(pd.concat([pre2003, workbook], ignore_index=True))


def walk_forward_by_era(games: pd.DataFrame, eras: dict[str, range]) -> pd.DataFrame:
    """Anchored walk-forward accuracy/MAE/Brier for every season, tagged by era."""
    season_to_era = {s: name for name, seasons in eras.items() for s in seasons}
    rows = []
    previous: pd.Series | None = None
    for season in sorted(games["season"].unique()):
        season_games = games[games["season"] == season]
        teams = sorted(set(season_games["team1"]) | set(season_games["team2"]))
        fbs = [t for t in teams if registry.was_fbs(t, season)]
        prior = mri2.build_prior(previous, teams, centre_teams=fbs)

        era = season_to_era.get(season)
        if era is not None:
            scored = backtest.evaluate_season(
                season_games, prior=prior, with_classic=(era == "2003-2019"), anchor_teams=fbs
            )
            if not scored.empty:
                rows.append(scored.assign(season=season, era=era))

        previous = mri2.fit(
            season_games, prior=prior, anchor_teams=fbs, with_resume=False, with_efficiency=False
        ).power

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    parquet_dir = ROOT / "data" / "parquet"
    pre2003_games = pd.read_parquet(parquet_dir / "pre2003_games.parquet")
    archive_games = pd.read_parquet(parquet_dir / "archive_games.parquet")
    games = combine_for_validation(pre2003_games, archive_games)

    scored = walk_forward_by_era(games, ERAS)

    print("Walk-forward accuracy by era (out-of-sample, anchored to each season's FBS field):\n")
    for era in ERAS:
        block = scored[scored["era"] == era]
        if block.empty:
            print(f"  {era}: no scored games")
            continue
        summary = backtest.summarize(block)
        line = (f"  {era}: mri2 accuracy {summary['mri2_accuracy']:.1%}, "
                f"mae {summary['mri2_mae']:.2f}, brier {summary['mri2_brier']:.4f}, "
                f"home field {summary['home_field']:.2f} ({summary['windows']} windows)")
        if era == "2003-2019":
            line += f"\n    classic accuracy {summary['classic_accuracy']:.1%}, edge {summary['edge']:+.1%}"
        print(line)

    out_path = parquet_dir / "backfill_validation.parquet"
    scored.to_parquet(out_path, index=False)
    print(f"\nwrote {len(scored)} scored windows to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
