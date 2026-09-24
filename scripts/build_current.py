"""Bridge the archive to the present and rate the current season.

Ben's workbooks stop at 2019. To rate 2026 properly the model needs a prior,
and that prior should come from a rating chain rather than a guess, so this
pulls 2020-2026 from the API and walks MRI 2.0 forward season by season:
2019's final ratings prime 2020, whose final ratings prime 2021, and so on.

That costs one API call per season, because the games endpoint returns a whole
year at a time. Box scores and betting lines are week-scoped and much more
expensive, so they are left to the modules that actually need them.

Two things improve the moment the data comes from the API instead of the
workbooks: FCS opponents are named individually rather than pooled, and neutral
sites are flagged rather than inferred.

Run:  PYTHONPATH=src python3 scripts/build_current.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.ingest import cfbd, registry  # noqa: E402
from mri.ratings import mri2, priors  # noqa: E402

BRIDGE_SEASONS = range(2020, 2027)
CURRENT_SEASON = 2026


def canonical(frame: pd.DataFrame) -> pd.DataFrame:
    """Put every FBS name into its registry spelling; leave FCS names alone."""
    frame = frame.copy()
    for column in ("team1", "team2"):
        frame[column] = [registry.resolve(name, name) for name in frame[column]]
    return frame


def main() -> None:
    out = ROOT / "data" / "parquet"
    out.mkdir(parents=True, exist_ok=True)

    archive = pd.read_parquet(out / "archive_games.parquet")
    archive_2019 = canonical(archive[archive["season"] == 2019])
    previous = mri2.fit(
        archive_2019,
        anchor_teams=[t for t in set(archive_2019["team2"]) if registry.is_fbs(t)],
        with_resume=False,
        with_efficiency=False,
    ).power
    print(f"primed from the 2019 archive ({len(previous)} teams)")

    seasons, ratings = [], []
    for season in BRIDGE_SEASONS:
        games = canonical(cfbd.games(season))
        if games.empty:
            print(f"  {season}: no completed games yet")
            continue

        teams = sorted(set(games["team1"]) | set(games["team2"]))
        fbs_teams = [t for t in teams if registry.is_fbs(t)]
        prior = priors.for_season(season, previous, teams, fbs_teams)
        model = mri2.fit(
            games, prior=prior, neutral=games["neutral"], anchor_teams=fbs_teams
        )

        table = model.table().assign(season=season)
        fbs = table[table["team"].map(registry.is_fbs)]
        best = fbs.iloc[0]
        print(
            f"  {season}: {len(games):>4} games, {len(fbs):>3} FBS teams, "
            f"HFA {model.home_field:5.2f}, top: {best['team']} ({best['power']:+.1f})"
        )

        seasons.append(games.assign(season=season))
        ratings.append(table)
        previous = model.power

    pd.concat(seasons, ignore_index=True).to_parquet(out / "current_games.parquet")
    pd.concat(ratings, ignore_index=True).to_parquet(out / "current_ratings.parquet")

    current = ratings[-1]
    current = current[current["team"].map(registry.is_fbs)].head(25)
    print(f"\n{CURRENT_SEASON} top 25 by Power:\n")
    print(
        current[["rank", "team", "power", "resume", "resume_rank", "games"]].to_string(
            index=False, float_format=lambda v: f"{v:7.2f}"
        )
    )


if __name__ == "__main__":
    main()
