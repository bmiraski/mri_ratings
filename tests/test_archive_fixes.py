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


# --- hand-checked score corrections ---------------------------------------------------------

CORRECTED = [
    (2003, "Arkansas State", "Mississippi", 0, 55),
    (2004, "Cincinnati", "East Carolina", 24, 19),
    (2004, "Arizona State", "Oregon", 28, 13),
    (2005, "Temple", "Bowling Green", 7, 69),
    (2009, "Southern Mississippi", "Kansas", 28, 35),
    (2010, "UCLA", "Oregon", 13, 60),
    # Confirmed correct in the workbook, so deliberately untouched:
    (2003, "Cal", "Kansas State", 28, 42),
    (2018, "Air Force", "Florida Atlantic", 27, 33),
    (2018, "Texas State", "Georgia State", 40, 31),
]


@pytest.mark.parametrize("season,away,home,away_pts,home_pts", CORRECTED)
def test_checked_scores_are_in_the_archive(games: pd.DataFrame, season, away, home, away_pts, home_pts) -> None:
    row = games[(games["season"] == season) & (games["team1"] == away) & (games["team2"] == home)]
    assert len(row) == 1
    assert (row["pts1"].iloc[0], row["pts2"].iloc[0]) == (away_pts, home_pts)


def _corrections(tmp_path, entry) -> object:
    import json

    path = tmp_path / "c.json"
    path.write_text(json.dumps({"corrections": [entry]}))
    return path


def _one_game(pts1: float, pts2: float) -> pd.DataFrame:
    return pd.DataFrame({"team1": ["A"], "team2": ["B"], "pts1": [pts1], "pts2": [pts2],
                         "win1": [float(pts1 > pts2)], "win2": [float(pts2 > pts1)]})


def test_a_correction_changes_only_its_game(tmp_path) -> None:
    path = _corrections(tmp_path, {"season": 2005, "team1": "a", "team2": "B", "old": [7, 70], "new": [7, 69]})
    fixed = archive_fixes.apply_score_corrections(2005, _one_game(7, 70), path)
    assert (fixed["pts1"].iloc[0], fixed["pts2"].iloc[0]) == (7, 69)
    untouched = archive_fixes.apply_score_corrections(2006, _one_game(7, 70), path)
    assert untouched["pts2"].iloc[0] == 70


def test_a_stale_correction_fails_loudly(tmp_path) -> None:
    path = _corrections(tmp_path, {"season": 2005, "team1": "A", "team2": "B", "old": [7, 70], "new": [7, 69]})
    with pytest.raises(ValueError, match="expected 1"):
        archive_fixes.apply_score_corrections(2005, _one_game(7, 71), path)


def test_a_correction_that_flips_the_winner_is_refused(tmp_path) -> None:
    path = _corrections(tmp_path, {"season": 2005, "team1": "A", "team2": "B", "old": [7, 70], "new": [70, 7]})
    with pytest.raises(ValueError, match="winner"):
        archive_fixes.apply_score_corrections(2005, _one_game(7, 70), path)


def test_2017_bowls_keep_their_box_scores(games: pd.DataFrame) -> None:
    bowls = games[(games["season"] == 2017)].tail(40)
    assert bowls["classic_ready"].all()
    assert bowls[["rush1", "rush2", "pass1", "pass2", "to1", "to2"]].notna().all().all()
