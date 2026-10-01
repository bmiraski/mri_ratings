"""The schedule-graph check: islands, zero-game teams, leaves, and a fit it never changes."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from mri.ratings import connectivity, mri2


def _games(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Visitor, host; the visitor always loses 10-20 so margins are well defined."""
    return pd.DataFrame(
        {
            "team1": [a for a, _ in pairs],
            "team2": [h for _, h in pairs],
            "pts1": 10.0,
            "pts2": 20.0,
            "win1": 0.0,
            "win2": 1.0,
        }
    )


TWO_ISLANDS = [("A", "B"), ("B", "C"), ("C", "A"), ("D", "E"), ("E", "F"), ("F", "D")]
CONNECTED = TWO_ISLANDS + [("C", "D")]


def test_two_islands_are_flagged() -> None:
    conn = connectivity.analyze(_games(TWO_ISLANDS), rated=list("ABCDEF"))
    assert not conn.ok
    assert conn.loose.components == 2
    assert conn.strict.components == 2
    assert len(conn.strict.orphans) == 3
    assert set(conn.strict.orphans["team"]) in ({"A", "B", "C"}, {"D", "E", "F"})
    assert (conn.strict.orphans["games"] == 2).all()


def test_connected_graph_passes() -> None:
    conn = connectivity.analyze(_games(CONNECTED), rated=list("ABCDEF"))
    assert conn.ok and conn.summary() == "connected"
    assert conn.loose.components == 1 and conn.strict.main_size == 6


def test_zero_game_team_is_reported_separately() -> None:
    conn = connectivity.analyze(_games(CONNECTED), rated=list("ABCDEF") + ["G"])
    assert conn.zero_game == ("G",)
    assert conn.strict.connected            # G is missing, not stranded
    assert not conn.ok
    assert "zero games" in conn.summary()


def test_expected_roster_catches_name_mismatches() -> None:
    games = _games([("A", "B"), ("B", "Miami (FL)")])
    conn = connectivity.analyze(games, rated=["A", "B"], expected=["A", "B", "Miami"])
    assert conn.zero_game == ("Miami",)


def test_leaf_opponent_is_not_flagged_when_strict() -> None:
    games = _games(CONNECTED + [("FCS Tech", "A")])
    conn = connectivity.analyze(games, rated=list("ABCDEF"))
    assert conn.loose.connected and conn.strict.connected and conn.ok


def test_rated_team_reachable_only_through_unrated_node_is_flagged_strict_only() -> None:
    # X and Y never meet; each only plays the unrated FCS side.
    games = _games(CONNECTED + [("X", "FCS Tech"), ("FCS Tech", "Y"), ("FCS Tech", "A")])
    conn = connectivity.analyze(games, rated=list("ABCDEF") + ["X", "Y"])
    assert conn.loose.connected
    assert set(conn.strict.orphans["team"]) == {"X", "Y"}
    assert not conn.ok


def test_assert_connected_raises_with_names() -> None:
    conn = connectivity.analyze(_games(TWO_ISLANDS), rated=list("ABCDEF"))
    with pytest.raises(connectivity.ConnectivityError, match="outside the main component"):
        connectivity.assert_connected(conn, "2026 week 5")
    connectivity.assert_connected(connectivity.analyze(_games(CONNECTED), rated=list("ABCDEF")))


def test_fit_warns_but_does_not_raise_and_stores_the_report() -> None:
    games = _games(TWO_ISLANDS)
    with pytest.warns(connectivity.ConnectivityWarning):
        ratings = mri2.fit(games, anchor_teams=list("ABCDEF"), with_resume=False, with_efficiency=False)
    assert not ratings.connectivity.ok


def test_fit_is_identical_with_the_check_on_or_off() -> None:
    games = _games(TWO_ISLANDS + [("C", "D"), ("A", "E")])
    kwargs = dict(ridge=3.0, anchor_teams=list("ABCDEF"), with_efficiency=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        on = mri2.fit(games, **kwargs)
    off = mri2.fit(games, check_connectivity=False, **kwargs)
    assert on.power.equals(off.power)
    assert on.home_field == off.home_field and on.sigma == off.sigma
    assert on.resume.equals(off.resume)
    assert off.connectivity is None and on.connectivity is not None
