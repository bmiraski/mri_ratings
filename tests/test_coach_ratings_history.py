"""The persisted archive-era MRI 2.0 history must be the same chained walk-forward
that evaluate_archive validated, on a scale that holds still season to season."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mri.ingest import registry
from mri.ratings import backtest, history, mri2

ARCHIVE = Path(__file__).resolve().parents[1] / "data" / "parquet" / "archive_games.parquet"
pytestmark = pytest.mark.skipif(not ARCHIVE.exists(), reason="archive not built")

CHECK_SEASONS = (2018, 2019)  # which seasons get the (slower) per-team assertions


@pytest.fixture(scope="module")
def canonical_games() -> pd.DataFrame:
    return history.canonical_games(pd.read_parquet(ARCHIVE))


@pytest.fixture(scope="module")
def walk(canonical_games: pd.DataFrame) -> pd.DataFrame:
    return history.archive_walk(canonical_games)


def test_every_archive_season_is_present(walk: pd.DataFrame) -> None:
    assert sorted(walk["season"].unique()) == list(history.ARCHIVE_SEASONS)


def test_anchoring_holds_the_scale_still(walk: pd.DataFrame) -> None:
    """Each checked season's FBS field should average to ~zero, season after season."""
    for season in CHECK_SEASONS:
        rows = walk[walk["season"] == season]
        fbs = rows[rows["team"].map(lambda t: registry.was_fbs(t, season))]
        assert abs(fbs["power"].mean()) < 1.0, f"{season}: FBS field is not centred on zero"


def test_the_chain_is_really_chained(canonical_games: pd.DataFrame, walk: pd.DataFrame) -> None:
    """Refitting a season with the walk's own prior for it must reproduce that season's row."""
    second = CHECK_SEASONS[-1]
    first = second - 1
    previous = walk[walk["season"] == first].set_index("team")["power"]

    second_games = canonical_games[canonical_games["season"] == second]
    teams = sorted(set(second_games["team1"]) | set(second_games["team2"]))
    fbs = [t for t in teams if registry.was_fbs(t, second)]
    prior = mri2.build_prior(previous, teams, centre_teams=fbs)
    refit = mri2.fit(second_games, prior=prior, anchor_teams=fbs, with_resume=False, with_efficiency=False).power

    persisted = walk[walk["season"] == second].set_index("team")["power"]
    common = persisted.index.intersection(refit.index)
    assert (persisted[common] - refit[common]).abs().max() < 1e-6


def test_matches_evaluate_archives_own_team_set_per_season(canonical_games: pd.DataFrame, walk: pd.DataFrame) -> None:
    """Same seasons scored the validated way should see the same teams this walk sees."""
    scored = backtest.evaluate_archive(canonical_games, seasons=list(CHECK_SEASONS))
    assert not scored.empty
    for season in CHECK_SEASONS:
        walked_teams = set(walk.loc[walk["season"] == season, "team"])
        games = canonical_games[canonical_games["season"] == season]
        assert walked_teams == set(games["team1"]) | set(games["team2"])
