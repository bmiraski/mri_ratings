"""MRI Basketball must reproduce its workbooks, and must not be football.

The two formulas share a skeleton, which makes it easy to port one and assume
the other. These tests pin the four places they actually differ, so a later
"simplification" that collapses them fails here rather than silently changing
every basketball rating.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mri.ratings import classic

openpyxl = pytest.importorskip("openpyxl")

from mri.ingest.archive import read_basketball_archive, year_from_path  # noqa: E402

ARCHIVE = Path(__file__).resolve().parents[1] / "data" / "archive-bb"
TOLERANCE = 1e-9

_seasons = read_basketball_archive(ARCHIVE) if ARCHIVE.exists() else {}


@pytest.mark.skipif(not _seasons, reason="basketball workbooks not present")
@pytest.mark.parametrize("year", sorted(_seasons))
def test_reproduces_published_ratings(year: int) -> None:
    """The gate: the port must match the workbook it replaces."""
    season = _seasons[year]
    computed = classic.compute(
        season.games, season.teams, sport=classic.BASKETBALL
    ).set_index("team")
    published = season.published.set_index("team")["mri"].to_dict()

    shared = set(computed.index) & set(published)
    assert len(shared) > 300, f"{year}: only {len(shared)} teams matched by name"

    worst = max(abs(computed.loc[t, "mri"] - published[t]) for t in shared)
    assert worst < TOLERANCE, f"{year}: worst deviation {worst}"


@pytest.mark.skipif(not _seasons, reason="basketball workbooks not present")
@pytest.mark.parametrize("year", sorted(_seasons))
def test_ranking_order_matches(year: int) -> None:
    season = _seasons[year]
    computed = classic.compute(season.games, season.teams, sport=classic.BASKETBALL)
    published = season.published.set_index("team")["mri"].to_dict()

    shared = set(published)
    ours = [t for t in computed["team"] if t in shared][:10]
    theirs = sorted(shared, key=lambda t: -published[t])[:10]
    assert ours == theirs, f"{year}: top ten differs"


def test_margin_cap_is_thirty_not_thirty_five() -> None:
    assert classic.BASKETBALL.margin_cap == 30.0
    assert classic.FOOTBALL.margin_cap == 35.0


def test_basketball_has_no_undefeated_fallback() -> None:
    """Football credits 0.1 against an unbeaten opponent; basketball does not.
    Copying one in would change every rating in a season where anyone runs
    the table."""
    assert classic.BASKETBALL.undefeated_fallback is None
    assert classic.FOOTBALL.undefeated_fallback == 0.1


def test_sports_weight_different_statistics() -> None:
    basketball = {name for name, _, _ in classic.BASKETBALL.components}
    football = {name for name, _, _ in classic.FOOTBALL.components}
    assert basketball == {"rebound_diff_per_game", "turnover_diff_per_game"}
    assert not basketball & football


def test_basketball_sos_removes_own_games() -> None:
    """The RPI adjustment: a team must not inflate its own strength of schedule
    by beating people, since each of its wins reappears as an opponent loss."""
    records = pd.DataFrame(
        {
            "wins": [10.0], "losses": [5.0],
            "opp_wins": [100.0], "opp_losses": [110.0],
            "opp_opp_wins": [1500.0], "opp_opp_losses": [1600.0],
            "opp_win_pct": [0.476], "opp_opp_win_pct": [0.484],
        },
        index=["A"],
    )
    adjusted = classic._sos_basketball(records)["A"]
    unadjusted = classic._sos_football(records)["A"]
    assert adjusted != pytest.approx(unadjusted)


def test_season_year_conventions() -> None:
    """Span filenames end the season; single-year basketball files start it."""
    assert year_from_path("MRIBasketball201920.xlsx") == 2020
    assert year_from_path("MRIBasketball201718.xlsx") == 2018
    assert year_from_path("MRIBasketball2012.xlsx") == 2013  # the 2012-13 season
    assert year_from_path("MRIFootball2018.xls") == 2018


def test_football_path_is_unchanged_by_the_refactor() -> None:
    """FOOTBALL is the default, so existing callers keep the frozen behaviour."""
    games = pd.DataFrame(
        {
            "team1": ["B"], "team2": ["A"],
            "pts1": [10.0], "pts2": [24.0],
            "rush1": [100.0], "rush2": [200.0],
            "pass1": [180.0], "pass2": [260.0],
            "to1": [2.0], "to2": [1.0],
            "win1": [0.0], "win2": [1.0],
        }
    )
    default = classic.compute(games, ["A", "B"])
    explicit = classic.compute(games, ["A", "B"], sport=classic.FOOTBALL)
    pd.testing.assert_frame_equal(default, explicit)
