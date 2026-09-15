"""The exported basketball workbooks must be live, not dumps - and must
reproduce ``classic.compute(..., sport=classic.BASKETBALL)`` exactly.

Mirrors ``tests/test_workbook.py`` (the football version), with one addition
that matters more here than it does there: basketball keeps no pooled
non-rated team, so a non-D1 opponent has to be excluded from the ranking and
from the z-score pool individually, and that exclusion is worth pinning on
its own rather than trusting the football pattern transferred cleanly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

openpyxl = pytest.importorskip("openpyxl")

from mri.export import bb_workbook  # noqa: E402
from mri.ingest import bb_registry as registry  # noqa: E402
from mri.ratings import classic  # noqa: E402

EXPORTS = Path(__file__).resolve().parents[1] / "exports"
SEASON = 2026
NON_D1 = "Fictional State"  # not a real program, so the registry never lists it


@pytest.fixture(scope="module")
def sample_games() -> pd.DataFrame:
    """A tiny round-robin among three real D1 teams plus one non-D1 opponent.

    Real names (rather than bare letters, as the football fixture uses) so
    ``bb_registry.is_d1``/``conference_of`` answer for them without any
    monkeypatching - and so the non-D1 opponent is non-D1 for the ordinary
    reason a lookup finds nothing, not because a test rigged it that way.
    """
    return pd.DataFrame(
        {
            "team1": ["Kansas", "Gonzaga", "Duke", NON_D1, "Gonzaga"],
            "team2": ["Duke", "Kansas", "Gonzaga", "Duke", "Duke"],
            "pts1": [62.0, 70.0, 55.0, 40.0, 75.0],
            "pts2": [69.0, 65.0, 80.0, 90.0, 68.0],
            "reb1": [30.0, 35.0, 25.0, 18.0, 33.0],
            "reb2": [32.0, 28.0, 38.0, 42.0, 30.0],
            "to1": [12.0, 9.0, 14.0, 20.0, 10.0],
            "to2": [10.0, 11.0, 8.0, 6.0, 9.0],
            "win1": [0.0, 1.0, 0.0, 0.0, 1.0],
            "win2": [1.0, 0.0, 1.0, 1.0, 0.0],
            "neutral": [False] * 5,
            "start_date": ["2025-11-05T00:00:00.000Z"] * 5,
        }
    )


def test_writes_the_expected_sheets(sample_games, tmp_path) -> None:
    path = bb_workbook.write_season(SEASON, sample_games, None, tmp_path / "t.xlsx")
    book = openpyxl.load_workbook(path)
    assert {"About", "Games", "Team Data", "MRI"} <= set(book.sheetnames)


def test_games_sheet_keeps_the_specified_columns(sample_games, tmp_path) -> None:
    """An old workbook and a new one must line up without translation."""
    path = bb_workbook.write_season(SEASON, sample_games, None, tmp_path / "t.xlsx")
    sheet = openpyxl.load_workbook(path)["Games"]
    headers = [sheet.cell(1, c).value for c in range(1, 22)]
    assert headers == [
        "Team1", "Team2", "Pnt1", "Pnt2", "Reb1", "Reb2", "TO1", "TO2", "WT1", "WT2",
        "T1W", "T1L", "T2W", "T2L", "T1OW", "T1OL", "T2OW", "T2OL", "T1Point", "T2Point",
        "Date",
    ]


def test_team_data_sheet_keeps_the_specified_columns(sample_games, tmp_path) -> None:
    path = bb_workbook.write_season(SEASON, sample_games, None, tmp_path / "t.xlsx")
    sheet = openpyxl.load_workbook(path)["Team Data"]
    headers = [sheet.cell(1, c).value for c in range(1, 23)]
    assert headers == [
        "Team", "W", "L", "Opp W", "Opp L", "O2 W", "O2 L", "Pnts", "Rebs", "O Rebs",
        "TO's", "O TO's", "Away Win", "Away Loss", "Reb Diff", "RDPG", "TO Diff", "TODPG",
        "Conference", "MRI Score", "SOS", "SOS Rank",
    ]


def test_derived_columns_are_formulas_not_values(sample_games, tmp_path) -> None:
    """If these were values the workbook would be a dump, not a model."""
    path = bb_workbook.write_season(SEASON, sample_games, None, tmp_path / "t.xlsx")
    book = openpyxl.load_workbook(path)
    assert str(book["Games"].cell(2, 19).value).startswith("=")
    assert str(book["Team Data"].cell(2, 20).value).startswith("=")
    assert str(book["MRI"].cell(2, 6).value).startswith("=")


def test_named_ranges_exist(sample_games, tmp_path) -> None:
    """The MRI formula reads these; without them every rating is #NAME?."""
    path = bb_workbook.write_season(SEASON, sample_games, None, tmp_path / "t.xlsx")
    book = openpyxl.load_workbook(path)
    for name in ("RDAVG", "RebSD", "TODAVG", "TOSD"):
        assert name in book.defined_names


def test_non_d1_opponent_gets_no_rating(sample_games, tmp_path) -> None:
    """Basketball has no pooled row - a non-D1 opponent keeps its own name but
    is excluded from the rated field the same way football's single pooled
    row is: no MRI score, no SOS rank, no conference."""
    path = bb_workbook.write_season(SEASON, sample_games, None, tmp_path / "t.xlsx")
    sheet = openpyxl.load_workbook(path)["Team Data"]
    row = next(
        r for r in range(2, sheet.max_row + 1)
        if sheet.cell(r, 1).value == NON_D1
    )
    assert sheet.cell(row, 19).value is None  # Conference
    assert sheet.cell(row, 20).value is None  # MRI Score
    assert sheet.cell(row, 22).value is None  # SOS Rank
    # But its raw record is still populated, because a D1 opponent's own
    # lookups depend on it.
    assert str(sheet.cell(row, 2).value).startswith("=")  # W


def test_non_d1_opponent_is_absent_from_mri_sheet(sample_games, tmp_path) -> None:
    path = bb_workbook.write_season(SEASON, sample_games, None, tmp_path / "t.xlsx")
    sheet = openpyxl.load_workbook(path)["MRI"]
    teams = {sheet.cell(r, 2).value for r in range(2, sheet.max_row + 1)}
    assert NON_D1 not in teams
    assert teams == {"Kansas", "Gonzaga", "Duke"}


@pytest.mark.parametrize("path", sorted(EXPORTS.glob("MRIBasketball*.xlsx")))
def test_exported_workbook_matches_python(path: Path) -> None:
    """The sheet's own formulas must reproduce the Python rating.

    This is the acceptance criterion for the whole port: a clean
    recalculation only proves the formulas evaluate; this proves they compute
    the same thing ``classic.compute`` does, for every rated team.
    """
    from mri.ingest import bb_gamelog

    season = int("".join(c for c in path.stem if c.isdigit()))
    book = openpyxl.load_workbook(path, data_only=True)
    sheet = book["MRI"]

    from_excel = {
        sheet.cell(r, 2).value: float(sheet.cell(r, 6).value)
        for r in range(2, sheet.max_row + 1)
        if sheet.cell(r, 2).value and sheet.cell(r, 6).value is not None
    }
    assert from_excel, f"{path.name} has no computed ratings - was it recalculated?"

    # The same canonicalization write_season applies, so the Python side is
    # rating exactly the field the workbook rated.
    games = bb_gamelog.for_season(season)
    games = games.copy()
    for column in ("team1", "team2"):
        games[column] = [registry.resolve(n, n, season=season) for n in games[column]]
    all_teams = sorted(set(games["team1"]) | set(games["team2"]))
    d1 = [t for t in all_teams if registry.is_d1(t, season=season)]

    from_python = classic.compute(games, d1, sport=classic.BASKETBALL).set_index("team")["mri"].to_dict()

    shared = set(from_excel) & set(from_python)
    assert len(shared) > 300, f"{path.name}: only {len(shared)} teams matched"
    worst = max(abs(from_excel[t] - from_python[t]) for t in shared)
    assert worst < 1e-6, f"{path.name}: worst disagreement {worst}"
