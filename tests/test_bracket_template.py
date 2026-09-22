"""Bracket shape, learned from who played whom on which day - not from a name."""

from __future__ import annotations

import pandas as pd
import pytest

from mri.bracket import template as tpl


def game(date: str, a: str, b: str, *, neutral: bool = True) -> dict:
    return {"start_date": f"{date}T19:00:00.000Z", "team1": a, "team2": b, "win1": True, "neutral": neutral}


def test_a_flat_bracket_with_no_byes_is_one_tier() -> None:
    games = pd.DataFrame([game("2025-03-01", "A", "B"), game("2025-03-01", "C", "D"),
                          game("2025-03-03", "A", "C")])
    t = tpl._tiers_for_one_season(games)
    assert t == (4,)                       # all four teams appear on day one


def test_tiered_byes_are_read_off_the_calendar_not_a_label() -> None:
    # SEC-shaped: 8 play day one, 4 more join day two, the top 4 join day three - exactly what the real
    # 2025 SEC tournament did, whatever each conference happens to call these rounds.
    games = pd.DataFrame([
        game("2025-03-12", "L9", "L16"), game("2025-03-12", "L10", "L15"),
        game("2025-03-12", "L11", "L14"), game("2025-03-12", "L12", "L13"),
        game("2025-03-13", "L9", "S5"), game("2025-03-13", "L10", "S6"),
        game("2025-03-13", "L11", "S7"), game("2025-03-13", "L12", "S8"),
        game("2025-03-14", "L9", "S1"), game("2025-03-14", "L10", "S2"),
        game("2025-03-14", "L11", "S3"), game("2025-03-14", "L12", "S4"),
        game("2025-03-15", "L9", "L10"), game("2025-03-15", "L11", "L12"),
        game("2025-03-16", "L9", "L11"),
    ])
    assert tpl._tiers_for_one_season(games) == (8, 4, 4)


def test_late_night_games_are_grouped_by_the_us_arena_day_not_the_utc_date() -> None:
    # 11:45pm Eastern on March 12 is already March 13 in UTC. Both of these are "the same night" of round one.
    games = pd.DataFrame([
        {"start_date": "2025-03-12T22:00:00.000Z", "team1": "A", "team2": "B", "win1": True, "neutral": True},
        {"start_date": "2025-03-13T03:45:00.000Z", "team1": "C", "team2": "D", "win1": True, "neutral": True},
    ])
    assert tpl._tiers_for_one_season(games) == (4,)


def test_campus_hosted_is_read_from_the_neutral_flag() -> None:
    on_campus = pd.DataFrame([game("2025-03-01", "A", "B", neutral=False)])
    at_one_site = pd.DataFrame([game("2025-03-01", "A", "B", neutral=True)])
    assert tpl._campus_hosted(on_campus) is True
    assert tpl._campus_hosted(at_one_site) is False
    assert tpl._campus_hosted(on_campus.iloc[0:0]) is False


def test_infer_prefers_the_most_recent_season_and_flags_disagreement() -> None:
    old = pd.DataFrame([game("2022-03-01", "A", "B"), game("2022-03-01", "C", "D"), game("2022-03-03", "A", "C")])
    old2 = pd.DataFrame([game("2023-03-01", "A", "B"), game("2023-03-01", "C", "D"), game("2023-03-03", "A", "C")])
    grown = pd.DataFrame([  # the conference added two teams for this season
        game("2024-03-01", "A", "B"), game("2024-03-01", "C", "D"), game("2024-03-01", "E", "F"),
        game("2024-03-03", "A", "C"), game("2024-03-03", "E", "G"),
    ])
    t = tpl.infer([old, old2, grown])
    assert t.tiers == tpl._tiers_for_one_season(grown)   # the most recent shape, not a vote
    assert t.stable is False and t.seasons_seen == 3

    agree = tpl.infer([old, old2, old2])
    assert agree.stable is True


def test_infer_needs_at_least_the_minimum_seasons_requested() -> None:
    one = pd.DataFrame([game("2024-03-01", "A", "B")])
    assert tpl.infer([one], min_seasons=2) is None
    assert tpl.infer([one], min_seasons=1) is not None
    assert tpl.infer([], min_seasons=1) is None


def test_template_size_and_rounds() -> None:
    t = tpl.Template(tiers=(8, 4, 4), seasons_seen=3, stable=True)
    assert t.size == 16 and t.rounds == 3
