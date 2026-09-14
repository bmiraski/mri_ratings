"""Basketball team identity, where a wrong alias merges two programs silently.

The football registry is small enough to eyeball. This one covers 365 teams, so
the guarantees are asserted instead: every alias points at a team that exists,
nothing in the archive goes unresolved, and the specific pairs a fuzzy matcher
wanted to merge stay separate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openpyxl")

from mri.ingest import bb_registry as registry  # noqa: E402
from mri.ingest.archive import read_basketball_archive  # noqa: E402

ARCHIVE = Path(__file__).resolve().parents[1] / "data" / "archive-bb"


def _offline(exc: Exception) -> bool:
    return "Proxy" in type(exc).__name__ or "Connection" in type(exc).__name__


@pytest.fixture(scope="module")
def roster():
    try:
        return registry.teams()
    except Exception as exc:  # pragma: no cover - offline guard
        if _offline(exc):
            pytest.skip("basketball API unreachable")
        raise


def test_roster_is_the_expected_size(roster) -> None:
    assert 330 < len(roster) < 400, f"{len(roster)} teams is not a plausible D1"


def test_every_alias_points_at_a_real_team(roster) -> None:
    """A typo here silently sends a team's games nowhere."""
    assert registry.validate_aliases() == []


@pytest.mark.parametrize(
    "historical,expected",
    [
        ("Mississippi", "Ole Miss"),
        ("North Carolina State", "NC State"),
        ("Nebraska Omaha", "Omaha"),
        ("IUPUI", "IU Indianapolis"),
        ("Texas Pan-American", "UT Rio Grande Valley"),
        ("Wisconsin-Green Bay", "Green Bay"),
        ("Cal State Sacramento", "Sacramento State"),
        ("LIU-Brooklyn", "Long Island University"),
    ],
)
def test_known_renames_resolve(roster, historical: str, expected: str) -> None:
    assert registry.resolve(historical) == expected


@pytest.mark.parametrize(
    "left,right",
    [
        ("Mississippi", "Mississippi State"),
        ("North Carolina State", "South Carolina State"),
        ("Nebraska Omaha", "Nebraska"),
        ("Louisiana-Lafayette", "Louisiana Tech"),
        ("Central Florida", "North Florida"),
    ],
)
def test_similar_names_stay_separate(roster, left: str, right: str) -> None:
    """Every one of these pairs is what fuzzy matching proposed merging, at
    scores between 0.71 and 0.90. They are different schools."""
    assert registry.resolve(left) != registry.resolve(right)


def test_departed_programs_do_not_resolve(roster) -> None:
    """A school that left D1 must resolve to nothing rather than to whichever
    surviving team has a similar name."""
    assert registry.resolve("Savannah State") is None
    assert registry.resolve("St. Francis (NY)") is None


@pytest.mark.skipif(not ARCHIVE.exists(), reason="basketball workbooks not present")
def test_no_archive_name_is_unresolved(roster) -> None:
    """An unresolved name is a team whose games get silently dropped."""
    seasons = read_basketball_archive(ARCHIVE)
    if not seasons:
        pytest.skip("no usable workbooks")
    names = set()
    for season in seasons.values():
        names |= set(season.teams)
    audit = registry.audit(names)
    unresolved = audit[audit["status"] == "UNRESOLVED"]["name"].tolist()
    assert not unresolved, f"unresolved: {unresolved}"


def test_conferences_are_plausible(roster) -> None:
    grouped = registry.conferences()
    assert 25 < len(grouped) < 40
    assert all(len(members) >= 4 for members in grouped.values())


def test_louisiana_monroe_resolves() -> None:
    """It sat in DEPARTED with the note "carried under a different spelling;
    resolved per season" - and nothing resolved it. The API calls it UL Monroe.
    A departed-list entry that is really a missing alias is worse than no entry,
    because it reads as a decision somebody made."""
    from mri.ingest import bb_registry as registry

    assert registry.resolve("Louisiana-Monroe", season=2026) == "UL Monroe"
    assert "Louisiana-Monroe" not in registry.DEPARTED
    # St. Francis (NY) really did leave - it dropped to Division III.
    assert "St. Francis (NY)" in registry.DEPARTED
    assert registry.resolve("St. Francis (NY)", season=2026) is None
