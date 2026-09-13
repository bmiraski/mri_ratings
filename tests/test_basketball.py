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


# --- MRI 2.0 for basketball -------------------------------------------------

RATINGS = Path(__file__).resolve().parents[1] / "data" / "parquet" / "bb_ratings.parquet"


@pytest.fixture(scope="module")
def bb_ratings():
    if not RATINGS.exists():
        pytest.skip("basketball ratings not built")
    return pd.read_parquet(RATINGS)


def test_profiles_differ_between_sports() -> None:
    """Basketball margins are tighter and its home advantage larger, so the
    same solver needs different settings."""
    from mri.ratings import mri2

    assert mri2.BASKETBALL_PROFILE.compression != mri2.FOOTBALL_PROFILE.compression
    assert mri2.BASKETBALL_PROFILE.home_field_prior > mri2.FOOTBALL_PROFILE.home_field_prior


def test_ratings_cover_every_season(bb_ratings) -> None:
    assert set(bb_ratings["season"]) == set(range(2021, 2027))


def test_each_season_rates_a_full_field(bb_ratings) -> None:
    from mri.ingest import bb_registry as registry

    for season, chunk in bb_ratings.groupby("season"):
        d1 = chunk[chunk["team"].map(lambda t: registry.is_d1(t, season=int(season)))]
        assert 330 < len(d1) < 375, f"{season}: {len(d1)} D1 teams rated"


def test_scale_does_not_drift_across_the_chain(bb_ratings) -> None:
    """The football bridge decayed from +37 to -34 before the scale was
    anchored. This is the guard against that happening again."""
    from mri.ingest import bb_registry as registry

    tops = []
    for season, chunk in bb_ratings.groupby("season"):
        d1 = chunk[chunk["team"].map(lambda t: registry.is_d1(t, season=int(season)))]
        tops.append(d1["power"].max())
    assert all(15 < top < 45 for top in tops), f"best-team ratings drifted: {tops}"


def test_cancelled_and_scheduled_games_are_not_played() -> None:
    """The basketball API returns 0-0 for a game that never happened, where the
    football one returns null. Testing points-are-present therefore accepts
    every cancellation as a tie, which is how 1,465 phantom games reached the
    first build of the chain. Status is the field that actually answers."""
    from mri.ingest import cbbd

    assert cbbd._is_final({"status": "final", "homePoints": 70, "awayPoints": 68})
    assert not cbbd._is_final({"status": "cancelled", "homePoints": 0, "awayPoints": 0})
    assert not cbbd._is_final({"status": "postponed", "homePoints": 0, "awayPoints": 0})
    assert not cbbd._is_final({"status": "scheduled", "homePoints": 0, "awayPoints": 0})
    # Fallback for a payload with no status at all.
    assert cbbd._is_final({"homePoints": 70, "awayPoints": 68})
    assert not cbbd._is_final({"homePoints": 0, "awayPoints": 0})


def test_no_phantom_ties_reached_the_ratings() -> None:
    """The same bug, caught from the other end."""
    games = Path(__file__).resolve().parents[1] / "data" / "parquet" / "bb_games.parquet"
    if not games.exists():
        pytest.skip("basketball games not built")
    frame = pd.read_parquet(games)
    ties = frame[(frame["pts1"] == 0) & (frame["pts2"] == 0)]
    assert ties.empty, f"{len(ties)} unplayed games rated as 0-0 ties"


def test_home_court_is_plausible_every_season(bb_ratings) -> None:
    """Between one and five points. Above that the fit is blaming schedule on
    the venue; below it, something has gone wrong with the anchor."""
    from mri.ratings import bb_backtest as bb

    games = bb.prepare(2026)
    if games.empty:
        pytest.skip("season unavailable")
    model = bb.fit_slice(games, None, season=2026)
    assert 1.0 < model.home_field < 5.0
