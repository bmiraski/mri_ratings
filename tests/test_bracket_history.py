"""Ground truth for one season: conference standings, who won each conference's
bracket, and the labeled field - all built and checked without touching the API."""

from __future__ import annotations

import pandas as pd
import pytest

from mri.bracket import history


def std_game(a, b, a_won):
    return {"season_type": "regular", "game_type": "STD", "conf1": "Test", "conf2": "Test",
           "team1": a, "team2": b, "win1": 1.0 if a_won else 0.0, "win2": 0.0 if a_won else 1.0}


def test_standings_rank_by_conference_win_percentage() -> None:
    games = pd.DataFrame([
        std_game("Best", "Worst", True), std_game("Best", "Mid", True), std_game("Mid", "Worst", True),
        std_game("Worst", "Best", False),      # Best 2-0 vs a repeat; Worst 0-2
    ])
    order = history.standings(2025, "Test", games)
    assert order[0] == "Best" and order[-1] == "Worst"
    assert set(order) == {"Best", "Mid", "Worst"}


def test_standings_only_look_at_the_named_conferences_own_games() -> None:
    games = pd.DataFrame([
        std_game("A", "B", True),
        {"season_type": "regular", "game_type": "STD", "conf1": "Test", "conf2": "Other", "team1": "A", "team2": "C",
         "win1": 1.0, "win2": 0.0},                                                              # not a conference game
        {"season_type": "regular", "game_type": "TRNMNT", "conf1": "Test", "conf2": "Test", "team1": "A", "team2": "B",
         "win1": 0.0, "win2": 1.0},                                                              # the conference tournament, not standings
    ])
    order = history.standings(2025, "Test", games)
    assert set(order) == {"A", "B"} and "C" not in order


def test_auto_bid_winners_come_from_who_actually_won_each_bracket(monkeypatch) -> None:
    from mri.ingest import bb_bracket

    conf_games = pd.DataFrame([
        {"conference": "Alpha", "start_date": "2025-03-10", "team1": "A1", "team2": "A2", "win1": True},
        {"conference": "Beta", "start_date": "2025-03-09", "team1": "B1", "team2": "B2", "win1": False},
        {"conference": "Beta", "start_date": "2025-03-11", "team1": "B2", "team2": "B3", "win1": True},
    ])
    monkeypatch.setattr(bb_bracket, "conference_tournaments", lambda season, refresh=False: conf_games)
    winners = history.auto_bid_winners(2025)
    assert winners == {"Alpha": "A1", "Beta": "B2"}


def test_the_field_labels_conference_champions_as_auto_and_everyone_else_at_large(monkeypatch) -> None:
    from mri.ingest import bb_bracket

    ncaa = pd.DataFrame([
        {"team1": "Champ", "team2": "Rival", "seed1": 1, "seed2": 16, "region": "East", "conf1": "Alpha", "conf2": "Zeta"},
        {"team1": "Bubble", "team2": "Champ", "seed1": 10, "seed2": 1, "region": "East", "conf1": "Beta", "conf2": "Alpha"},
    ])
    monkeypatch.setattr(bb_bracket, "ncaa_tournament", lambda season, refresh=False: ncaa)
    monkeypatch.setattr(history, "auto_bid_winners", lambda season, conferences=None: {"Alpha": "Champ"})
    f = history.field(2025).set_index("team")
    assert f.loc["Champ", "bidType"] == "auto" and f.loc["Champ", "seed"] == 1
    assert f.loc["Bubble", "bidType"] == "at-large"
    assert f.loc["Rival", "bidType"] == "at-large"          # Zeta's own champion, if it has one, isn't Rival


def test_the_field_keeps_a_teams_best_seed_if_it_ever_shows_up_twice(monkeypatch) -> None:
    from mri.ingest import bb_bracket

    ncaa = pd.DataFrame([
        {"team1": "PlayIn", "team2": "Foe", "seed1": 16, "seed2": 16, "region": "West", "conf1": "Alpha", "conf2": "Beta"},
        {"team1": "PlayIn", "team2": "Other", "seed1": 16, "seed2": 1, "region": "West", "conf1": "Alpha", "conf2": "Gamma"},
    ])
    monkeypatch.setattr(bb_bracket, "ncaa_tournament", lambda season, refresh=False: ncaa)
    monkeypatch.setattr(history, "auto_bid_winners", lambda season, conferences=None: {})
    f = history.field(2025).set_index("team")
    assert f.loc["PlayIn", "seed"] == 16
