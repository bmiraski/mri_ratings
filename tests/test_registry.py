"""The registry is the single source of truth for team identity.

A silent alias failure costs a team half its schedule, so these tests guard the
counts and the historical mapping rather than trusting the JSON by eye.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mri.ingest import registry

EXPECTED_TOTAL = 138
EXPECTED_SIZES = {
    "ACC": 17,
    "Big Ten": 18,
    "Big 12": 16,
    "SEC": 16,
    "American": 14,
    "Conference USA": 10,
    "MAC": 13,
    "Mountain West": 10,
    "Pac-12": 8,
    "Sun Belt": 14,
    "Independent": 2,
}

ARCHIVE_GAMES = Path(__file__).resolve().parents[1] / "data" / "parquet" / "archive_games.parquet"

# Idaho dropped back to FCS after 2017; Non D1A is the pooled opponent that
# MRI Classic uses and MRI 2.0 replaces.
EXPECTED_UNRESOLVED = {"Idaho", "Non D1A"}


def test_team_count() -> None:
    assert len(registry.teams()) == EXPECTED_TOTAL


def test_conference_sizes() -> None:
    actual = {name: len(meta["teams"]) for name, meta in registry.conferences().items()}
    assert actual == EXPECTED_SIZES


def test_no_team_in_two_conferences() -> None:
    seen: dict[str, str] = {}
    for conference, meta in registry.conferences().items():
        for team in meta["teams"]:
            assert team not in seen, f"{team} is in both {seen.get(team)} and {conference}"
            seen[team] = conference


@pytest.mark.parametrize(
    "alias,canonical",
    [
        ("Central Florida", "UCF"),
        ("Connecticut", "UConn"),
        ("Miami", "Miami (FL)"),
        ("Miami (Ohio)", "Miami (OH)"),
        ("Mississippi", "Ole Miss"),
        ("Louisiana-Lafayette", "Louisiana"),
        ("Middle Tenn. St", "Middle Tennessee"),
        ("Troy State", "Troy"),
        ("  boston   COLLEGE ", "Boston College"),
    ],
)
def test_alias_resolution(alias: str, canonical: str) -> None:
    assert registry.resolve(alias) == canonical


def test_unknown_names_return_default() -> None:
    assert registry.resolve("Some FCS School") is None
    assert registry.resolve("Some FCS School", "FCS") == "FCS"


@pytest.mark.skipif(not ARCHIVE_GAMES.exists(), reason="archive not built")
def test_historical_names_all_map() -> None:
    games = pd.read_parquet(ARCHIVE_GAMES)
    names = set(games["team1"]) | set(games["team2"])
    assert set(registry.unresolved(names)) == EXPECTED_UNRESOLVED
