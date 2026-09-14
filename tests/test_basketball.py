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


def test_only_the_current_window_is_refetched() -> None:
    """A daily job must not re-read months that cannot change.

    The original rule - "the window has not ended yet" - is true of every future
    month too, so in the off-season all seven windows of the coming season looked
    live and cost seven API calls a day to learn nothing.
    """
    import datetime as dt

    from mri.ingest import cbbd

    november = ("2026-11-01", "2026-12-01")
    december = ("2026-12-01", "2027-01-01")
    march = ("2027-03-01", "2027-04-01")

    # Mid-November: only November.
    today = dt.date(2026, 11, 15)
    assert cbbd._is_live(*november, today)
    assert not cbbd._is_live(*december, today)
    assert not cbbd._is_live(*march, today)

    # September, before anything has tipped: nothing is live.
    preseason = dt.date(2026, 9, 13)
    assert not any(cbbd._is_live(*w, preseason) for w in (november, december, march))

    # A window stays live briefly past its end, for late finals.
    assert cbbd._is_live(*november, dt.date(2026, 12, 2))
    assert not cbbd._is_live(*november, dt.date(2026, 12, 10))


def test_season_probe_is_cached() -> None:
    """Both build and build_full ask which season to show; each answer walks a
    season of date windows, so asking twice doubled every run's API calls."""
    from mri.export import bb_sitedata

    assert hasattr(bb_sitedata.latest_playing_season, "cache_info")


# --- betting ---------------------------------------------------------------

def test_market_sign_convention() -> None:
    """The spread is quoted from the home team's perspective and is negative
    when the home team is favoured, so the market's expected home margin is
    -spread. Getting this backwards produces a backtest that runs perfectly and
    reports the model as an inverse oracle."""
    import pandas as pd

    from mri.betting import bb_lines

    games = Path(__file__).resolve().parents[1] / "data" / "parquet" / "bb_priced_dk.parquet"
    if not games.exists():
        pytest.skip("priced games not built")
    frame = pd.read_parquet(games)
    # An efficient market's number should track the actual margin closely and
    # sit near it on average.
    assert frame["market"].corr(frame["actual"]) > 0.5
    assert abs(frame["market"].mean() - frame["actual"].mean()) < 2.0
    assert bb_lines.BOOK == "DraftKings"


def test_projection_services_are_never_treated_as_markets() -> None:
    """numberfire and teamrankings sit beside the sportsbooks in this feed and
    are model outputs. Scoring against them measures agreement with somebody
    else's projection and would report it as beating a market."""
    from mri.betting import bb_lines

    assert "numberfire" in bb_lines.NOT_MARKETS
    assert "teamrankings" in bb_lines.NOT_MARKETS
    frame = bb_lines.season_lines(2021)
    if frame.empty:
        pytest.skip("2020-21 lines unavailable")
    assert not frame["provider"].str.casefold().isin(bb_lines.NOT_MARKETS).any()


def test_backtest_corrects_for_the_number_of_buckets() -> None:
    """Eight buckets is eight chances at p < 0.05, so one usually takes it."""
    from mri.betting import bb_backtest as bb

    assert bb.COMPARISONS == len(bb.BUCKETS) - 1
    assert bb.COMPARISONS >= 8


def test_the_board_never_prices_a_game_it_has_already_seen() -> None:
    """The whole discipline in one assertion: a board built as of a date may
    only use games that finished before it."""
    import datetime as dt

    import pandas as pd

    from mri.betting import bb_board
    from mri.ingest import cbbd

    cut = dt.date(2026, 1, 15)
    real = cbbd.games

    def as_of(season, **kwargs):
        frame = real(season, completed_only=False).copy()
        day = pd.to_datetime(frame["start_date"], format="ISO8601", utc=True).dt.date
        frame.loc[day >= cut, ["played", "pts1", "pts2", "win1", "win2"]] = [
            False, None, None, 0.0, 0.0
        ]
        return frame[frame["played"]] if kwargs.get("completed_only", True) else frame

    cbbd.games = as_of
    try:
        board = bb_board.build_board(2026, today=cut)
    finally:
        cbbd.games = real

    if not board["games"]:
        pytest.skip("season data unavailable")
    days = {g["day"] for g in board["games"]}
    assert min(days) >= cut.isoformat(), "the board is pricing games already played"
    assert max(days) <= (cut + dt.timedelta(days=bb_board.HORIZON_DAYS)).isoformat()
    # Thin teams are excluded from the shown disagreements, never from the data.
    assert all(g["confident"] for g in board["disagreements"])


def test_the_prior_chain_heals_itself(tmp_path, monkeypatch) -> None:
    """bb_ratings.parquet is written by a script nothing runs automatically.
    The first season after nobody re-ran it, every team would have started from
    no prior at all - not an error, just a silently worse November."""
    from mri.export import bb_sitedata

    monkeypatch.setattr(bb_sitedata, "RATINGS", tmp_path / "absent.parquet")
    prior = bb_sitedata._prior_for(2026)
    if prior is None:
        pytest.skip("2024-25 games unavailable")
    assert len(prior) > 300
    assert prior.max() > 15, "a recomputed prior must be on the same scale"


def test_the_recomputed_prior_matches_the_stored_one(tmp_path, monkeypatch) -> None:
    """The fallback has to agree with the chain, or a missing file would
    silently move every rating."""
    import pandas as pd

    from mri.export import bb_sitedata

    if not bb_sitedata.RATINGS.exists():
        pytest.skip("basketball ratings not built")
    stored = bb_sitedata._prior_for(2026)
    monkeypatch.setattr(bb_sitedata, "RATINGS", tmp_path / "absent.parquet")
    recomputed = bb_sitedata._prior_for(2026)

    shared = stored.index.intersection(recomputed.index)
    assert len(shared) > 300
    # Identical, not merely close. The fallback reconstructs the same chain the
    # script wrote, which is the only version of this worth having: a fallback
    # that is approximately right moves every rating on the site the day the
    # file goes missing, and nothing says it happened.
    assert (stored[shared] - recomputed[shared]).abs().max() < 1e-9
    assert list(stored.nlargest(5).index) == list(recomputed.nlargest(5).index)


def test_the_box_score_feed_is_filtered_to_real_games() -> None:
    """The box-score endpoint emits rows for games that never happened, zeros
    throughout and nothing marking them - Delaware at Towson on 2022-01-28 is
    "scheduled" in the games feed and a 0-0 final here. Classic reads 0-0 as a
    loss for both sides, so one phantom row moved ten teams' ratings."""
    from mri.ingest import cbbd

    table = cbbd.classic_table(2022)
    if table.empty:
        pytest.skip("2021-22 box scores unavailable")
    ties = table[(table["pts1"] == 0) & (table["pts2"] == 0)]
    assert ties.empty, f"{len(ties)} unplayed games in the Classic game log"
    assert 22442 not in set(table["game_id"]), "the known phantom game is back"
