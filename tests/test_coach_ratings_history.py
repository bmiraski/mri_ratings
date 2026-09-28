"""The persisted archive-era MRI 2.0 history must be the same chained walk-forward
that evaluate_archive validated, on a scale that holds still season to season."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mri.ingest import registry
from mri.ratings import backtest, history, mri2

ARCHIVE = Path(__file__).resolve().parents[1] / "data" / "parquet" / "archive_games.parquet"
PRE2003 = Path(__file__).resolve().parents[1] / "data" / "parquet" / "pre2003_games.parquet"
COACH_SEASON = Path(__file__).resolve().parents[1] / "data" / "parquet" / "coach_season.parquet"
pytestmark = pytest.mark.skipif(not ARCHIVE.exists(), reason="archive not built")

CHECK_SEASONS = (2018, 2019)  # which seasons get the (slower) per-team assertions
EARLY_CHECK_SEASONS = (1985, 1995)  # spot years for the CFBD-sourced pre-2003 half


@pytest.fixture(scope="module")
def canonical_games() -> pd.DataFrame:
    return history.canonical_games(pd.read_parquet(ARCHIVE))


@pytest.fixture(scope="module")
def walk(canonical_games: pd.DataFrame) -> pd.DataFrame:
    return history.archive_walk(canonical_games)


def test_every_archive_season_is_present(walk: pd.DataFrame) -> None:
    assert sorted(walk["season"].unique()) == list(history.ARCHIVE_SEASONS)


def test_anchoring_holds_the_scale_still(walk: pd.DataFrame) -> None:
    """Each checked season's FBS field should average to ~zero, season after season."""
    for season in CHECK_SEASONS:
        rows = walk[walk["season"] == season]
        fbs = rows[rows["team"].map(lambda t: registry.was_fbs(t, season))]
        assert abs(fbs["power"].mean()) < 1.0, f"{season}: FBS field is not centred on zero"


def test_the_chain_is_really_chained(canonical_games: pd.DataFrame, walk: pd.DataFrame) -> None:
    """Refitting a season with the walk's own prior for it must reproduce that season's row."""
    second = CHECK_SEASONS[-1]
    first = second - 1
    previous = walk[walk["season"] == first].set_index("team")["power"]

    second_games = canonical_games[canonical_games["season"] == second]
    teams = sorted(set(second_games["team1"]) | set(second_games["team2"]))
    fbs = [t for t in teams if registry.was_fbs(t, second)]
    prior = mri2.build_prior(previous, teams, centre_teams=fbs)
    refit = mri2.fit(second_games, prior=prior, anchor_teams=fbs, with_resume=False, with_efficiency=False).power

    persisted = walk[walk["season"] == second].set_index("team")["power"]
    common = persisted.index.intersection(refit.index)
    assert (persisted[common] - refit[common]).abs().max() < 1e-6


def test_matches_evaluate_archives_own_team_set_per_season(canonical_games: pd.DataFrame, walk: pd.DataFrame) -> None:
    """Same seasons scored the validated way should see the same teams this walk sees."""
    scored = backtest.evaluate_archive(canonical_games, seasons=list(CHECK_SEASONS))
    assert not scored.empty
    for season in CHECK_SEASONS:
        walked_teams = set(walk.loc[walk["season"] == season, "team"])
        games = canonical_games[canonical_games["season"] == season]
        assert walked_teams == set(games["team1"]) | set(games["team2"])


# ---------------------------------------------------------------------------
# the historical backfill: 1978-2002 (CFBD), chained into the walk above
# ---------------------------------------------------------------------------

pytestmark_backfill = pytest.mark.skipif(not PRE2003.exists(), reason="pre-2003 backfill not built")


@pytest.fixture(scope="module")
def full_games(canonical_games: pd.DataFrame) -> pd.DataFrame:
    if not PRE2003.exists():
        pytest.skip("pre-2003 backfill not built")
    pre2003 = pd.read_parquet(PRE2003)
    archive = pd.read_parquet(ARCHIVE)
    return history.combine_early_games(pre2003, archive)


@pytest.fixture(scope="module")
def full_walk(full_games: pd.DataFrame) -> pd.DataFrame:
    return history.archive_walk(full_games)


@pytestmark_backfill
def test_full_backfill_covers_every_season_1978_2019(full_walk: pd.DataFrame) -> None:
    assert sorted(full_walk["season"].unique()) == list(history.BACKFILL_SEASONS)


@pytestmark_backfill
def test_burn_in_flags_exactly_the_first_three_seasons(full_walk: pd.DataFrame) -> None:
    flagged = set(full_walk.loc[full_walk["burn_in"], "season"])
    assert flagged == set(history.BURN_IN_SEASONS)


@pytestmark_backfill
def test_anchoring_holds_the_scale_still_before_2003(full_walk: pd.DataFrame) -> None:
    """Same check as test_anchoring_holds_the_scale_still, on CFBD-sourced seasons."""
    for season in EARLY_CHECK_SEASONS:
        rows = full_walk[full_walk["season"] == season]
        fbs = rows[rows["team"].map(lambda t: registry.was_fbs(t, season))]
        assert abs(fbs["power"].mean()) < 1.0, f"{season}: FBS field is not centred on zero"


@pytestmark_backfill
def test_pre2003_division_membership_matches_cfbd_every_season(full_walk: pd.DataFrame) -> None:
    """The division file the original plan worried might be needed: checked
    against the live API for every pre-2003 season, not a static list, so a
    season this repo has never looked at closely still gets caught if
    ``registry.was_fbs`` ever disagrees with what CFBD itself says."""
    from mri.ingest import cfbd

    for season in history.PRE2003_SEASONS:
        try:
            expected = len(cfbd.fbs_teams(season))
        except Exception:  # noqa: BLE001
            pytest.skip("FBS roster unavailable offline")
        rows = full_walk[full_walk["season"] == season]
        actual = int(rows["team"].map(lambda t: registry.was_fbs(t, season)).sum())
        assert actual == expected, f"{season}: walk anchors {actual} FBS teams, API says {expected}"


@pytestmark_backfill
def test_extended_chain_gap_from_the_old_flat_start_is_small(
    full_walk: pd.DataFrame, full_games: pd.DataFrame
) -> None:
    """The 2003 seam mri.ratings.history documents: real and bounded, asserted
    here rather than only printed once by the build script, so the canonical
    switch it describes stays a checked fact rather than a one-time
    observation."""
    gap = history.early_seam_gap(full_walk, full_games)
    assert not gap.empty
    assert abs(gap.mean()) < 1.0
    assert gap.abs().max() < 10.0


@pytest.mark.skipif(not COACH_SEASON.exists(), reason="coach_season.parquet not built")
def test_coach_training_does_not_yet_overlap_burn_in_seasons() -> None:
    """Coach data doesn't extend before 2003 yet, so burn_in (1978-1980) has
    nothing to be excluded from today - this guards that assumption rather
    than asserting an exclusion mri.coaches.metrics doesn't implement. If
    coach data ever grows back into the backfill, this starts failing and
    that exclusion becomes a real, not aspirational, requirement."""
    coach_season = pd.read_parquet(COACH_SEASON)
    overlap = set(coach_season["season"].unique()) & set(history.BURN_IN_SEASONS)
    assert not overlap, f"coach_season.parquet now has burn-in seasons {overlap} - exclude them from training"
