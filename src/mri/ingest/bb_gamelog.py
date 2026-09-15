"""The basketball Classic game log, cached as a derived table.

MRI Basketball Classic needs rebounds and turnovers for every game, which means
the box-score endpoint, which means roughly 300MB of raw JSON for thirteen
seasons. That cache is deliberately not committed - it would more than double
the repository to spare a manual script some API calls.

The consequence was not obvious until the scheduled run hit it: anything that
needs a Classic game log and does not have that cache has to fetch it, and the
test step has no API key. So the workbook acceptance test - the one that proves
the spreadsheet reproduces the rating - could not run in CI at all, and failed
rather than skipping.

The fix is to commit what is actually wanted. The joined, normalized game log is
115KB a season against 23MB of JSON: 1.5MB for every season the project has, or
half a percent of the raw form. Tests and the workbook export read this; only a
refresh of a live season touches the API.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

STORE = Path(__file__).resolve().parents[3] / "data" / "parquet" / "bb_classic_games.parquet"


def load(season: int) -> pd.DataFrame:
    """One season from the committed store, or an empty frame."""
    if not STORE.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(STORE)
    return frame[frame["season"] == season].reset_index(drop=True)


def available() -> list[int]:
    if not STORE.exists():
        return []
    return sorted(int(s) for s in pd.read_parquet(STORE)["season"].unique())


def for_season(season: int, *, refresh: bool = False) -> pd.DataFrame:
    """A season's Classic game log, from the store when it is there.

    A season still being played is re-fetched, because its log grows every
    night. A finished one never changes and is served from the committed table,
    which is what lets the tests run without a key.
    """
    from . import cbbd
    from .bb_registry import CURRENT_SEASON

    if not refresh and season < CURRENT_SEASON:
        cached = load(season)
        if not cached.empty:
            return cached
    return cbbd.classic_table(season)


def rebuild(seasons, *, verbose: bool = True) -> pd.DataFrame:
    """Refetch the given seasons and rewrite the store, keeping the others."""
    from . import cbbd

    frames = []
    if STORE.exists():
        existing = pd.read_parquet(STORE)
        frames.append(existing[~existing["season"].isin(list(seasons))])

    for season in sorted(seasons):
        table = cbbd.classic_table(season)
        if table.empty:
            if verbose:
                print(f"  {season}: no usable games")
            continue
        if verbose:
            print(f"  {season}: {len(table)} games")
        frames.append(table)

    combined = pd.concat(frames, ignore_index=True).sort_values(["season", "start_date"])
    STORE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(STORE, index=False)
    return combined
