"""The 2003-2019 archive after repair: 2017's bowls are present and 2004/05 know their neutral sites."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mri.ingest import archive_fixes
from mri.ratings import backtest, mri2

ARCHIVE = Path(__file__).resolve().parents[1] / "data" / "parquet" / "archive_games.parquet"
pytestmark = pytest.mark.skipif(not ARCHIVE.exists(), reason="archive not built")


@pytest.fixture(scope="module")
def games() -> pd.DataFrame:
    return pd.read_parquet(ARCHIVE)


def test_2017_has_its_bowls_and_the_title_game(games: pd.DataFrame) -> None:
    season = games[games["season"] == 2017]
    assert len(season) == 834 + 40
    final = season[(season["team1"] == "Alabama") & (season["team2"] == "Georgia")]
    assert len(final) == 1 and (final["pts1"].iloc[0], final["pts2"].iloc[0]) == (26, 23)
    assert season.tail(40)["neutral"].mean() > 0.8


def test_no_archive_season_is_missing_postseason_games_any_more(games: pd.DataFrame) -> None:
    for season in (2003, 2004, 2005, 2016, 2017, 2018, 2019):
        assert archive_fixes.missing_postseason(
            season, games[games["season"] == season].reset_index(drop=True), with_box_scores=False
        ).empty, season


def test_added_games_use_workbook_spellings(games: pd.DataFrame) -> None:
    season = games[games["season"] == 2017]
    original = set(season.head(834)["team1"]) | set(season.head(834)["team2"])
    added = season.tail(40)
    assert set(added["team1"]) | set(added["team2"]) <= original


@pytest.mark.parametrize("year", [2004, 2005])
def test_bowls_are_neutral_in_the_years_the_rule_missed(games: pd.DataFrame, year: int) -> None:
    season = games[games["season"] == year].reset_index(drop=True)
    assert season["neutral"].notna().all()
    assert mri2.mark_postseason(season).sum() == 2           # the inferred rule, still broken here
    assert season["neutral"].sum() >= 28                      # CFBD knows all 28 bowls
    assert season["neutral"].iloc[-40:].sum() >= 28           # and they sit at the end


def test_other_seasons_keep_the_inferred_rule(games: pd.DataFrame) -> None:
    season = games[games["season"] == 2010].reset_index(drop=True)
    assert season["neutral"].isna().all()
    assert backtest.season_neutral(season).equals(mri2.mark_postseason(season))


def test_the_harness_uses_the_table_flags_when_complete(games: pd.DataFrame) -> None:
    season = games[games["season"] == 2004].reset_index(drop=True)
    assert backtest.season_neutral(season).sum() == season["neutral"].sum()
