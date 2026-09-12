"""MRI Classic must reproduce the original workbooks exactly.

This is the trust gate for the whole rebuild: if the Python port does not
return the same numbers Ben's spreadsheets published, nothing downstream can
be believed. Tolerance is floating-point only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mri.ingest.archive import read_archive
from mri.ratings import classic

ARCHIVE = Path(__file__).resolve().parents[1] / "data" / "archive"
TOLERANCE = 1e-9

_seasons = read_archive(ARCHIVE) if ARCHIVE.exists() else {}


@pytest.mark.skipif(not _seasons, reason="archive workbooks not present")
@pytest.mark.parametrize("year", sorted(_seasons))
def test_reproduces_published_ratings(year: int) -> None:
    season = _seasons[year]
    computed = classic.compute(season.games, season.teams).set_index("team")
    published = season.published.set_index("team")

    joined = published.join(computed[["mri", "wins", "losses"]], rsuffix="_calc", how="inner")
    assert len(joined) == len(published), (
        f"{year}: {len(published) - len(joined)} published teams missing from the "
        "computed table - likely a team-name spelling variant"
    )

    worst = (joined["mri_calc"] - joined["mri"]).abs().max()
    assert worst < TOLERANCE, f"{year}: largest rating deviation was {worst}"

    assert (joined["wins"] == joined["wins_calc"]).all(), f"{year}: win totals disagree"
    assert (joined["losses"] == joined["losses_calc"]).all(), f"{year}: loss totals disagree"


@pytest.mark.skipif(not _seasons, reason="archive workbooks not present")
@pytest.mark.parametrize("year", sorted(_seasons))
def test_ranking_order_matches(year: int) -> None:
    season = _seasons[year]
    computed = classic.compute(season.games, season.teams)
    published = season.published.sort_values("mri", ascending=False)

    assert computed["team"].tolist()[:25] == published["team"].tolist()[:25], (
        f"{year}: top 25 ordering differs from the published ranking"
    )


def test_case_insensitive_team_matching() -> None:
    """Excel's SUMIF ignores case; the 2009 log spells Boise State two ways."""
    games = pd.DataFrame(
        {
            "team1": ["Boise State", "Boise STate"],
            "team2": ["Idaho", "Nevada"],
            "pts1": [40.0, 30.0],
            "pts2": [10.0, 20.0],
            "rush1": [200.0, 150.0],
            "rush2": [100.0, 90.0],
            "pass1": [200.0, 250.0],
            "pass2": [150.0, 160.0],
            "to1": [1.0, 0.0],
            "to2": [2.0, 3.0],
            "win1": [1.0, 1.0],
            "win2": [0.0, 0.0],
        }
    )
    result = classic.compute(games, ["Boise State", "Idaho", "Nevada"])
    boise = result.set_index("team").loc["Boise State"]
    assert boise["wins"] == 2, "spelling variants must collapse into one team"
