"""Regression: scripts/build_current.py primes 2020 from archive_games.parquet's 2019 rows, and
that prime must be keyed by registry-canonical team names.

The archive's own canonicalization (mri.ratings.classic.canonicalize) only
dedupes spellings within one workbook season - it never promised to agree
with registry.resolve, the spelling everything from 2020 onward uses. It
doesn't always agree: the archive calls California "Cal" in every season,
including 2019. Priming with the raw archive names (no registry.resolve) put
"Cal" in the primed Power series instead of "California" - so when the 2020
prior looked up "California", it missed and California started 2020 with no
real prior at all, silently, instead of its actual 2019 rating. Confirmed by
running the two versions against the live archive: California's primed
rating was -22.0 (mri2.REPLACEMENT_PRIOR, i.e. no prior found) unresolved,
and its real, far-from-replacement 2019 rating once resolved.

This is a smaller, targeted stand-in for build_current.py's own priming
block (see its ``canonical`` helper and the ``archive_2019`` line in
``main()``) rather than an import of the script itself, matching how the
rest of this suite tests the library modules a build script calls into
rather than the orchestration scripts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mri.ingest import registry
from mri.ratings import mri2

ARCHIVE = Path(__file__).resolve().parents[1] / "data" / "parquet" / "archive_games.parquet"
pytestmark = pytest.mark.skipif(not ARCHIVE.exists(), reason="archive not built")

BOUNDARY_SEASON = 2019
A_MISSPELLED_TEAM = ("Cal", "California")  # (the archive's spelling, the registry's)


@pytest.fixture(scope="module")
def boundary_games() -> pd.DataFrame:
    games = pd.read_parquet(ARCHIVE)
    return games[games["season"] == BOUNDARY_SEASON]


def _prime(games: pd.DataFrame) -> pd.Series:
    return mri2.fit(
        games,
        anchor_teams=[t for t in set(games["team2"]) if registry.is_fbs(t)],
        with_resume=False,
        with_efficiency=False,
    ).power


def test_unresolved_archive_names_lose_the_mismatched_teams_prior(boundary_games: pd.DataFrame) -> None:
    """Documents the bug: priming on raw archive spelling drops a team's real rating."""
    archive_name, registry_name = A_MISSPELLED_TEAM
    primed = _prime(boundary_games)
    assert archive_name in primed.index
    assert registry_name not in primed.index


def test_canonicalizing_first_keeps_every_teams_real_prior(boundary_games: pd.DataFrame) -> None:
    """The fix: resolve team names before priming, so a 2020 lookup by registry spelling succeeds."""
    archive_name, registry_name = A_MISSPELLED_TEAM
    resolved = boundary_games.copy()
    for column in ("team1", "team2"):
        resolved[column] = [registry.resolve(name, name) for name in resolved[column]]

    primed = _prime(resolved)
    assert registry_name in primed.index
    assert archive_name not in primed.index
    # A real, prior-informed rating - nowhere near "no prior found".
    assert primed[registry_name] > mri2.REPLACEMENT_PRIOR + 10
