"""Parsing the postseason feed: rounds, regions, and who won a bracket."""

from __future__ import annotations

import pandas as pd
import pytest

from mri.ingest import bb_bracket as bb


def test_ncaa_notes_split_into_region_and_round() -> None:
    assert bb._parse_notes("Men's Basketball Championship - East Region - 1st Round") == ("East", "1st Round")
    assert bb._parse_notes("Men's Basketball Championship - Final Four") == (None, "Final Four")
    assert bb._parse_notes("Men's Basketball Championship - National Championship") == (None, "National Championship")
    assert bb._parse_notes(None) == (None, None)


def test_conference_notes_have_no_region_to_find() -> None:
    assert bb._parse_notes("ASUN Championship - 1st Round") == (None, "1st Round")
    assert bb._parse_notes("Northeast Conference Tournament - Semifinal") == (None, "Semifinal")


def test_champion_is_the_winner_of_the_latest_dated_game() -> None:
    games = pd.DataFrame([
        {"start_date": "2025-03-12T17:00:00.000Z", "team1": "A", "team2": "B", "win1": True},
        {"start_date": "2025-03-14T17:00:00.000Z", "team1": "C", "team2": "A", "win1": False},   # the final: A wins
    ])
    assert bb.champion(games) == "A"
    assert bb.champion(games.iloc[0:0]) is None
