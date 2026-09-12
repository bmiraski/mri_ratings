"""Read Ben's historical MRI Excel workbooks (2003-2019) into normalized tables.

Every workbook in the Drive archive shares the same layout:

  Games      one row per game, team1/team2 with points, rushing, passing,
             turnovers, win flags, and an Excel date serial in column W
  Team Data  one row per FBS team (plus the pooled "Non D1A" row last),
             with the MRI in column V
  MRI        the published ranking table

Only the Games sheet is treated as source of truth. Everything on Team Data
and MRI is derived, and we recompute it in ratings.classic so the Python
implementation can be checked against the workbook's own numbers.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import xlrd

POOLED_FCS = "Non D1A"

# Games sheet column indices (0-based).
_G = {
    "team1": 0,
    "team2": 1,
    "pts1": 2,
    "pts2": 3,
    "rush1": 4,
    "rush2": 5,
    "pass1": 6,
    "pass2": 7,
    "to1": 8,
    "to2": 9,
    "win1": 10,
    "win2": 11,
    "date": 22,
}

_NUMERIC = ["pts1", "pts2", "rush1", "rush2", "pass1", "pass2", "to1", "to2", "win1", "win2"]


@dataclass(frozen=True)
class ArchiveSeason:
    """One season parsed out of a workbook."""

    year: int
    path: Path
    games: pd.DataFrame
    teams: list[str]
    published: pd.DataFrame

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (
            f"ArchiveSeason({self.year}, games={len(self.games)}, "
            f"teams={len(self.teams)}, published={len(self.published)})"
        )


def year_from_path(path: Path | str) -> int:
    """Pull the season year out of a filename like MRIFootball2010_final.xls."""
    match = re.search(r"(19|20)\d{2}", Path(path).name)
    if not match:
        raise ValueError(f"no year in filename: {path}")
    return int(match.group(0))


def _num(cell, default=None):
    """Excel cells in these sheets are numbers, blanks, or stray strings."""
    value = cell.value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            try:
                return float(stripped)
            except ValueError:
                return default
    return default


def _serial_to_date(serial: float | None, datemode: int) -> _dt.date | None:
    if serial is None or serial <= 0:
        return None
    try:
        return _dt.date(*xlrd.xldate_as_tuple(serial, datemode)[:3])
    except Exception:
        return None


def read_games(book: xlrd.Book) -> pd.DataFrame:
    """Normalize the Games sheet into one row per game."""
    sheet = book.sheet_by_name("Games")
    rows = []
    for r in range(1, sheet.nrows):
        row = sheet.row(r)
        team1 = str(row[_G["team1"]].value).strip()
        team2 = str(row[_G["team2"]].value).strip()
        if not team1 or not team2:
            continue

        record = {"team1": team1, "team2": team2}
        for key in _NUMERIC:
            record[key] = _num(row[_G[key]])

        # A row with no win flag on either side is a scheduled-but-unplayed
        # game; the workbook carries those as blanks.
        if record["win1"] is None and record["win2"] is None:
            continue
        record["win1"] = record["win1"] or 0.0
        record["win2"] = record["win2"] or 0.0

        serial = _num(row[_G["date"]]) if len(row) > _G["date"] else None
        record["date"] = _serial_to_date(serial, book.datemode)
        rows.append(record)

    frame = pd.DataFrame(rows)
    for key in _NUMERIC:
        frame[key] = frame[key].fillna(0.0).astype(float)
    return frame


def read_team_list(book: xlrd.Book) -> list[str]:
    """Team Data column A, in sheet order. The pooled FCS row sorts last."""
    sheet = book.sheet_by_name("Team Data")
    teams = []
    for r in range(1, sheet.nrows):
        name = str(sheet.cell(r, 0).value).strip()
        if not name:
            break
        teams.append(name)
    return teams


def read_published_ratings(book: xlrd.Book) -> pd.DataFrame:
    """The MRI sheet's published ranking, used as the validation target."""
    sheet = book.sheet_by_name("MRI")
    rows = []
    for r in range(2, sheet.nrows):
        team = str(sheet.cell(r, 1).value).strip()
        if not team:
            continue
        rating = _num(sheet.cell(r, 4))
        if rating is None:
            continue
        rows.append(
            {
                "rank": _num(sheet.cell(r, 0)),
                "team": team,
                "wins": _num(sheet.cell(r, 2), 0.0),
                "losses": _num(sheet.cell(r, 3), 0.0),
                "mri": rating,
                "mri_per_game": _num(sheet.cell(r, 5)),
            }
        )
    return pd.DataFrame(rows)


def read_workbook(path: Path | str) -> ArchiveSeason:
    path = Path(path)
    book = xlrd.open_workbook(path)
    return ArchiveSeason(
        year=year_from_path(path),
        path=path,
        games=read_games(book),
        teams=read_team_list(book),
        published=read_published_ratings(book),
    )


def read_archive(directory: Path | str) -> dict[int, ArchiveSeason]:
    """Read every workbook in a directory, keyed by season year.

    Where a year has several files (2010 ships a regular-season and a final
    workbook), the one with the most games wins.
    """
    seasons: dict[int, ArchiveSeason] = {}
    for path in sorted(Path(directory).glob("*.xls")):
        try:
            season = read_workbook(path)
        except Exception as exc:  # pragma: no cover - corrupt file guard
            print(f"  skipped {path.name}: {exc}")
            continue
        existing = seasons.get(season.year)
        if existing is None or len(season.games) > len(existing.games):
            seasons[season.year] = season
    return dict(sorted(seasons.items()))
