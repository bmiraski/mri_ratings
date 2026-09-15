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


def _bb_games(reb_missing: bool) -> pd.DataFrame:
    """Two games for one team, optionally with the second box score absent."""
    return pd.DataFrame(
        {
            "team1": ["Duke", "Duke"],
            "team2": ["Kansas", "Kentucky"],
            "pts1": [80.0, 70.0],
            "pts2": [70.0, 60.0],
            "reb1": [40.0, None if reb_missing else 40.0],
            "reb2": [30.0, None if reb_missing else 30.0],
            "to1": [10.0, None if reb_missing else 10.0],
            "to2": [15.0, None if reb_missing else 15.0],
            "win1": [1.0, 1.0],
            "win2": [0.0, 0.0],
        }
    )


def test_missing_box_score_keeps_the_result_and_skips_the_statistics() -> None:
    """A game with no rebound counts still happened.

    The basketball feed is missing about one box score in eight for 2004-05 and
    2011-12. Building the log from the box scores alone dropped those games
    outright, which cost a typical 2011-12 team four games off its record. They
    have to count towards the record, the schedule and the game credit, and to
    be left out of the per-game statistics only - so the team's rebound margin
    per game is what its reported games say, not that number diluted by games
    nobody reported.
    """
    teams = ["Duke", "Kansas", "Kentucky"]
    full = classic.compute(_bb_games(False), teams, sport=classic.BASKETBALL)
    partial = classic.compute(_bb_games(True), teams, sport=classic.BASKETBALL)

    for table in (full, partial):
        duke = table.set_index("team").loc["Duke"]
        assert duke["wins"] == 2, "both games count towards the record"
        assert duke["games"] == 2

    assert full.set_index("team").loc["Duke"]["stat_games"] == 2
    assert partial.set_index("team").loc["Duke"]["stat_games"] == 1

    # +10 a game either way: one game of +10 over one counted game, not over two.
    assert full.set_index("team").loc["Duke"]["rebound_diff_per_game"] == 10.0
    assert partial.set_index("team").loc["Duke"]["rebound_diff_per_game"] == 10.0


@pytest.mark.skipif(not _seasons, reason="archive workbooks not present")
def test_football_counts_every_game_as_a_stat_game() -> None:
    """The stat denominator is a basketball concession, not a football change.

    Every row of the football archive carries rushing, passing and turnovers, so
    stat_games equals games played and the ratings are the ones the workbooks
    published - which the acceptance tests above check directly. This says why
    they still pass.
    """
    season = _seasons[sorted(_seasons)[0]]
    table = classic.compute(season.games, season.teams)
    assert (table["stat_games"] == table["games"]).all()
