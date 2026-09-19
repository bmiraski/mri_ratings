"""The exported workbooks must be live, not dumps.

The point of matching the original layout is that a corrected score re-rates
the season, the way Ben's own workbooks did. That only holds if the formulas
are real and correct, so these tests check the sheet's own arithmetic against
the Python implementation rather than trusting that it recalculated cleanly -
a formula with an off-by-one range evaluates perfectly and is still wrong.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

openpyxl = pytest.importorskip("openpyxl")

from mri.export import workbook  # noqa: E402
from mri.ratings import classic  # noqa: E402

EXPORTS = Path(__file__).resolve().parents[1] / "exports"


@pytest.fixture(scope="module")
def sample_games() -> pd.DataFrame:
    """A tiny round-robin plus a pooled non-FBS opponent."""
    return pd.DataFrame(
        {
            "team1": ["B", "C", "A", "Non D1A", "C"],
            "team2": ["A", "B", "C", "A", "A"],
            "pts1": [10.0, 20.0, 14.0, 7.0, 21.0],
            "pts2": [24.0, 17.0, 28.0, 45.0, 20.0],
            "rush1": [100.0, 150.0, 120.0, 80.0, 130.0],
            "rush2": [200.0, 140.0, 210.0, 260.0, 190.0],
            "pass1": [180.0, 220.0, 240.0, 90.0, 200.0],
            "pass2": [260.0, 190.0, 230.0, 300.0, 210.0],
            "to1": [2.0, 1.0, 3.0, 4.0, 1.0],
            "to2": [1.0, 2.0, 1.0, 0.0, 2.0],
            "win1": [0.0, 1.0, 0.0, 0.0, 1.0],
            "win2": [1.0, 0.0, 1.0, 1.0, 0.0],
        }
    )


def test_writes_the_expected_sheets(sample_games, tmp_path) -> None:
    path = workbook.write_season(2026, sample_games, None, tmp_path / "t.xlsx")
    book = openpyxl.load_workbook(path)
    assert {"About", "Games", "Team Data", "MRI"} <= set(book.sheetnames)


def test_games_sheet_keeps_the_original_columns(sample_games, tmp_path) -> None:
    """An old workbook and a new one must line up without translation."""
    path = workbook.write_season(2026, sample_games, None, tmp_path / "t.xlsx")
    sheet = openpyxl.load_workbook(path)["Games"]
    headers = [sheet.cell(1, c).value for c in range(1, 13)]
    assert headers == [
        "Team1", "Team2", "Pnt1", "Pnt2", "Rush1", "Rush2",
        "Pass1", "Pass2", "TO1", "TO2", "WT1", "WT2",
    ]


def test_derived_columns_are_formulas_not_values(sample_games, tmp_path) -> None:
    """If these were values the workbook would be a dump, not a model."""
    path = workbook.write_season(2026, sample_games, None, tmp_path / "t.xlsx")
    book = openpyxl.load_workbook(path)
    assert str(book["Games"].cell(2, 21).value).startswith("=")
    assert str(book["Team Data"].cell(2, 22).value).startswith("=")
    assert str(book["MRI"].cell(2, 5).value).startswith("=")


def test_named_ranges_exist(sample_games, tmp_path) -> None:
    """The MRI formula reads these; without them every rating is #NAME?."""
    path = workbook.write_season(2026, sample_games, None, tmp_path / "t.xlsx")
    book = openpyxl.load_workbook(path)
    for name in ("RMEAN", "RDEV", "PMEAN", "PDEV", "DMEAN", "DDEV", "TOMEAN", "TODEV"):
        assert name in book.defined_names


def test_pooled_row_gets_no_rating(sample_games, tmp_path) -> None:
    """Ranking the pooled team against a range it is excluded from is #N/A,
    which is why the originals left its rating cells empty."""
    path = workbook.write_season(2026, sample_games, None, tmp_path / "t.xlsx")
    sheet = openpyxl.load_workbook(path)["Team Data"]
    pooled = next(
        r for r in range(2, sheet.max_row + 1)
        if sheet.cell(r, 1).value == classic.POOLED_FCS
    )
    assert sheet.cell(pooled, 21).value is None  # SOS rank
    assert sheet.cell(pooled, 22).value is None  # MRI


@pytest.mark.parametrize("path", sorted(EXPORTS.glob("MRIFootball*.xlsx")))
def test_exported_workbook_matches_python(path: Path) -> None:
    """The sheet's own formulas must reproduce the Python rating.

    This is the test that matters: a clean recalculation only proves the
    formulas evaluate. This proves they compute the right thing.
    """
    from mri.ingest import boxscores

    year = int("".join(c for c in path.stem if c.isdigit()))
    book = openpyxl.load_workbook(path, data_only=True)
    sheet = book["MRI"]

    from_excel = {
        sheet.cell(r, 2).value: float(sheet.cell(r, 5).value)
        for r in range(2, sheet.max_row + 1)
        if sheet.cell(r, 2).value and sheet.cell(r, 5).value is not None
    }
    assert from_excel, f"{path.name} has no computed ratings - was it recalculated?"

    games = boxscores.classic_table(year)

    # The workbook for the season in progress is a snapshot: it is exported by
    # hand (recalculation needs LibreOffice), while the daily run keeps pulling
    # in new games. Once a game is played the two are computed from different
    # data and cannot agree, which is not a fault in the formulas - and if it
    # failed the gate, no daily refresh would ever be committed. A finished
    # season has no such excuse, so a mismatch there still fails.
    from mri.ingest import cfbd

    exported_games = book["Games"].max_row - 1  # header row
    if year == cfbd.current_season() and len(games) != exported_games:
        pytest.skip(
            f"{path.name} is a snapshot of {exported_games} games; "
            f"{len(games)} have been played since"
        )

    teams = sorted(set(games["team1"]) | set(games["team2"]))
    from_python = classic.compute(games, teams).set_index("team")["mri"].to_dict()

    shared = set(from_excel) & set(from_python)
    worst = max(abs(from_excel[t] - from_python[t]) for t in shared)
    assert worst < 1e-9, f"{path.name}: worst disagreement {worst}"
