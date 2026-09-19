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


def test_fbs_membership_is_season_aware() -> None:
    """The hand-written registry describes 2026 and nothing else. The field was
    128 teams in 2020 and is 138 now: James Madison arrived in 2022, Sam Houston
    and Jacksonville State in 2023, Kennesaw State in 2024, Delaware in 2025,
    North Dakota State in 2026. Asking the current registry about 2020 put all
    of them in rankings for seasons they spent playing FCS football."""
    from mri.ingest import registry

    assert registry.was_fbs("Alabama", 2020)
    assert not registry.was_fbs("James Madison", 2020)
    assert registry.was_fbs("James Madison", 2022)
    assert not registry.was_fbs("Delaware", 2024)
    assert registry.was_fbs("Delaware", 2025)
    assert not registry.was_fbs("North Dakota State", 2025)

    # Idaho went the other way and left; it is in no season we rate.
    assert not registry.was_fbs("Idaho", 2020)

    # The field grows, and never shrinks across this range.
    sizes = [len(registry.fbs_members(y)) for y in range(2020, 2027)]
    assert sizes == sorted(sizes), sizes
    assert sizes[0] == 128 and sizes[-1] == 138, sizes


def test_season_aware_membership_resolves_aliases() -> None:
    """A season answered by the API and one answered by the registry have to
    agree on spelling, or a team drops out of its own history at the boundary."""
    from mri.ingest import registry

    for name in ("Cal", "Central Florida", "Mississippi", "North Carolina State"):
        canonical = registry.resolve(name)
        assert canonical and canonical != name
        assert registry.was_fbs(name, 2022) == registry.was_fbs(canonical, 2022)


def test_current_membership_is_unchanged_by_the_season_aware_path() -> None:
    """is_fbs() with no season still means "now", and the live site depends on
    that. was_fbs(None) must agree with it exactly."""
    from mri.ingest import registry

    for team in registry.teams():
        assert registry.is_fbs(team) == registry.was_fbs(team)
    assert registry.was_fbs("North Dakota State") is True


def test_a_cached_response_is_served_when_no_key_is_available() -> None:
    """The scheduled run's test step has no API key, on purpose - it has the
    committed cache and no business fetching anything. Before this, any test
    that touched a refreshed endpoint died on "No CFBD API key" rather than
    reading the copy sitting next to it, and took the whole gate down."""
    import json

    from mri.ingest import cfbd

    params = {"year": 2026, "week": 1, "seasonType": "regular"}
    path = cfbd._cache_path("/games/teams", params)
    if not path.exists():
        pytest.skip("football box scores not cached")

    import os
    saved, os.environ["CFBD_API_KEY"] = os.environ.pop("CFBD_API_KEY", None), ""
    env_file = cfbd.ROOT / ".env"
    hidden = env_file.with_suffix(".hidden") if env_file.exists() else None
    try:
        del os.environ["CFBD_API_KEY"]
        if hidden:
            env_file.rename(hidden)
        # refresh=True would normally force a fetch; with no key it must fall
        # back rather than raise.
        payload = cfbd.request("/games/teams", refresh=True, **params)
        assert payload == json.loads(path.read_text())
    finally:
        if hidden:
            hidden.rename(env_file)
        if saved is not None:
            os.environ["CFBD_API_KEY"] = saved


def test_the_season_in_progress_is_refreshed_and_finished_ones_are_not(monkeypatch) -> None:
    """The bug this guards: ``cfbd.games`` served the current season from cache
    forever, so a game that finished after the cache was seeded stayed
    "scheduled" and never reached the ratings. The daily run reported "no new
    games" for days while the API had them."""
    import datetime as dt

    from mri.ingest import cfbd

    assert cfbd.current_season(dt.date(2026, 9, 18)) == 2026
    assert cfbd.current_season(dt.date(2027, 1, 12)) == 2026  # playoffs

    seen = {}

    def fake_request(endpoint, *, refresh=False, **params):
        seen[params["year"]] = refresh
        return []

    monkeypatch.setattr(cfbd, "request", fake_request)
    monkeypatch.setattr(cfbd, "current_season", lambda today=None: 2026)
    cfbd.games(2026)
    cfbd.games(2019)
    assert seen == {2026: True, 2019: False}

    # An explicit choice still wins.
    cfbd.games(2026, refresh=False)
    assert seen[2026] is False


def test_a_refresh_is_fetched_once_per_run(monkeypatch, tmp_path) -> None:
    """Several callers ask for the same current-season file in one build. Each
    should not cost an API call."""
    import os

    from mri.ingest import cfbd

    monkeypatch.setattr(cfbd, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cfbd, "_FETCHED", set())
    monkeypatch.setenv("CFBD_API_KEY", "test-key")

    calls = []

    class Response:
        status_code = 200
        ok = True
        headers = {}
        text = ""

        def json(self):
            return [{"n": len(calls)}]

    def fake_get(*args, **kwargs):
        calls.append(args)
        return Response()

    monkeypatch.setattr(cfbd.requests, "get", fake_get)
    first = cfbd.request("/games", refresh=True, year=2026)
    second = cfbd.request("/games", refresh=True, year=2026)
    assert len(calls) == 1
    assert first == second
    assert os.path.exists(cfbd._cache_path("/games", {"year": 2026}))
