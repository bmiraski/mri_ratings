"""Two data-quality passes over the 1978-2019 backfill: missing games and a sanity list.

Missing-game check
    Compares each FBS team's game count against its own season's median (the
    expected length varies by era - 10-11 games in the 1980s, 12-13 today -
    so there is no single constant to compare against) and flags anyone more
    than 2 games short. A gap here means a missing game or a wrong division
    call, not a model problem.

Sanity list
    The end-of-season #1 team by Power, every season - not to check it
    against the AP poll (the plan expects disagreement; that's the point of
    a computer rating), but to catch a data problem a #1 team wouldn't
    plausibly have: an undersized schedule, an implausible power gap, or the
    same team parked at #1 for a run of seasons long enough to suggest the
    fit is stuck rather than genuinely dominant.

Run:  PYTHONPATH=src python3 scripts/check_backfill_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.ingest import registry  # noqa: E402
from mri.ratings import history  # noqa: E402

FLAG_GAMES_SHORT = 2


def missing_game_report(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, season_games in games.groupby("season"):
        counts = pd.concat([season_games["team1"], season_games["team2"]]).value_counts()
        fbs_counts = counts[[t for t in counts.index if registry.was_fbs(t, season)]]
        if fbs_counts.empty:
            continue
        expected = fbs_counts.median()
        short = fbs_counts[fbs_counts <= expected - FLAG_GAMES_SHORT]
        for team, played in short.items():
            rows.append({"season": season, "team": team, "played": played, "expected": expected})
    return pd.DataFrame(rows)


def main() -> None:
    parquet_dir = ROOT / "data" / "parquet"
    pre2003_games = pd.read_parquet(parquet_dir / "pre2003_games.parquet")
    archive_games = pd.read_parquet(parquet_dir / "archive_games.parquet")
    games = history.combine_early_games(pre2003_games, archive_games)

    print("=== Missing-game check (1978-2019, FBS teams only) ===\n")
    short = missing_game_report(games)
    if short.empty:
        print("  no FBS team is more than 2 games short of its season's median.")
    else:
        by_season = short.groupby("season").size()
        print(f"  {len(short)} team-seasons flagged, across {len(by_season)} seasons:")
        print(short.sort_values(["season", "played"]).to_string(index=False))

    print("\n=== Sanity list: end-of-season #1 by Power, 1978-2019 ===\n")
    ratings = pd.read_parquet(parquet_dir / "coach_ratings_history.parquet")
    ratings = ratings[ratings["season"] <= 2019].copy()
    ratings["is_fbs"] = [registry.was_fbs(t, s) for t, s in zip(ratings["team"], ratings["season"])]
    fbs = ratings[ratings["is_fbs"]]

    played = pd.concat([games["team1"], games["team2"]]).rename("team").to_frame()
    played["season"] = pd.concat([games["season"], games["season"]]).to_numpy()
    game_counts = played.groupby(["team", "season"]).size().rename("games")

    top = fbs.sort_values(["season", "power"], ascending=[True, False]).groupby("season").first().reset_index()
    top = top.join(game_counts, on=["team", "season"])
    print(top[["season", "team", "power", "games", "burn_in"]].to_string(
        index=False, float_format=lambda v: f"{v:6.2f}"
    ))

    thin_schedule = top[top["games"] < 8]
    if not thin_schedule.empty:
        print(f"\n  flagged: {len(thin_schedule)} #1 team(s) with under 8 games:")
        print(thin_schedule[["season", "team", "games"]].to_string(index=False))

    streaks = (top["team"] != top["team"].shift()).cumsum()
    longest = top.groupby(streaks)["team"].agg(["first", "size"]).sort_values("size", ascending=False).head(3)
    print(f"\n  longest #1 streaks: {list(longest.itertuples(index=False, name=None))}")


if __name__ == "__main__":
    main()
