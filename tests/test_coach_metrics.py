"""Per-coach-season metrics: power_end (including the split-season point-in-time
refit), the prior/program-par lookups keyed by stint rather than by coach_id,
nullability where there isn't enough history, and tenure summaries."""

from __future__ import annotations

import pandas as pd
import pytest

from mri.coaches import metrics
from mri.ratings import mri2

EMPTY_GAMES = pd.DataFrame(columns=["team1", "team2", "season"])


def _coach_season_rows(*rows: tuple[str, str, int, int]) -> pd.DataFrame:
    """coach_id, school, season, games -> a minimal coach_season frame."""
    return pd.DataFrame(
        [
            {"coach_id": cid, "coach_name": cid, "school": school, "season": season_, "games": games_,
             "wins": 0, "losses": 0, "interim": False}
            for cid, school, season_, games_ in rows
        ]
    )


# ---------------------------------------------------------------------------
# power_as_of
# ---------------------------------------------------------------------------

def test_power_as_of_matches_a_direct_fit_on_the_truncated_games() -> None:
    games = pd.DataFrame([
        {"team1": "Away", "team2": "School", "pts1": 10, "pts2": 20, "season": 2020},   # School's 1st game
        {"team1": "School", "team2": "Rival", "pts1": 14, "pts2": 7, "season": 2020},   # School's 2nd game
        {"team1": "Other1", "team2": "Other2", "pts1": 21, "pts2": 17, "season": 2020},
        {"team1": "School", "team2": "Other1", "pts1": 30, "pts2": 3, "season": 2020},  # School's 3rd game
    ])
    prior = pd.Series(0.0, index=["School", "Away", "Rival", "Other1", "Other2"])
    fbs: list[str] = []

    got = metrics.power_as_of("School", games, 2, prior=prior, fbs=fbs)
    expected = mri2.fit(
        games.iloc[:2], prior=prior, anchor_teams=fbs, with_resume=False, with_efficiency=False
    ).power["School"]
    assert got == pytest.approx(expected)

    # and it's a real truncation, not just the full-season fit in disguise
    full = mri2.fit(games, prior=prior, anchor_teams=fbs, with_resume=False, with_efficiency=False).power["School"]
    assert got != pytest.approx(full)


def test_power_as_of_raises_when_asked_for_more_games_than_were_played() -> None:
    games = pd.DataFrame([{"team1": "Away", "team2": "School", "pts1": 10, "pts2": 20, "season": 2020}])
    with pytest.raises(ValueError, match="only 1 were played"):
        metrics.power_as_of("School", games, 2, prior=pd.Series(dtype=float), fbs=[])


# ---------------------------------------------------------------------------
# split-season power_end
# ---------------------------------------------------------------------------

def _split_season_games() -> pd.DataFrame:
    return pd.DataFrame([
        {"team1": "Opp1", "team2": "School", "pts1": 10, "pts2": 40, "season": 2016},  # old-coach, big win
        {"team1": "School", "team2": "Opp2", "pts1": 35, "pts2": 10, "season": 2016},  # old-coach, big win
        {"team1": "Opp3", "team2": "School", "pts1": 20, "pts2": 17, "season": 2016},  # new-coach
        {"team1": "School", "team2": "Opp4", "pts1": 14, "pts2": 21, "season": 2016},  # new-coach, a loss
    ])


def _flat_ratings(season_: int, teams: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{"team": t, "season": season_, "power": 0.0, "prior": 0.0, "resume": 0.0} for t in teams])


def test_split_season_power_end_only_overrides_the_departing_coach() -> None:
    season_games = _split_season_games()
    coach_season = _coach_season_rows(
        ("old-coach", "School", 2015, 12),   # a season at School before 2016 - the incumbent
        ("old-coach", "School", 2016, 2),
        ("new-coach", "School", 2016, 2),    # first appearance at School - the one who finishes the season
    )
    ratings_history = _flat_ratings(2016, ["School", "Opp1", "Opp2", "Opp3", "Opp4"])

    overrides = metrics.split_season_power_end(coach_season, ratings_history, season_games, EMPTY_GAMES)

    assert ("old-coach", "School", 2016) in overrides
    assert ("new-coach", "School", 2016) not in overrides

    prior = ratings_history.set_index("team")["prior"]
    expected = metrics.power_as_of("School", season_games, 2, prior=prior, fbs=[])
    assert overrides[("old-coach", "School", 2016)] == pytest.approx(expected)
    full = mri2.fit(season_games, prior=prior, anchor_teams=[], with_resume=False, with_efficiency=False).power["School"]
    assert overrides[("old-coach", "School", 2016)] != pytest.approx(full)


def test_coach_season_metrics_uses_the_override_for_the_departing_coach_only() -> None:
    season_games = _split_season_games()
    coach_season = _coach_season_rows(
        ("old-coach", "School", 2015, 12),
        ("old-coach", "School", 2016, 2),
        ("new-coach", "School", 2016, 2),
    )
    ratings_history = _flat_ratings(2016, ["School", "Opp1", "Opp2", "Opp3", "Opp4"])
    ratings_history = pd.concat([ratings_history, _flat_ratings(2015, ["School"])], ignore_index=True)

    table = metrics.coach_season_metrics(coach_season, ratings_history, season_games, EMPTY_GAMES)
    row_2016 = table[table["season"] == 2016].set_index("coach_id")
    full_season_power = ratings_history.set_index(["team", "season"])["power"][("School", 2016)]

    assert row_2016.loc["new-coach", "power_end"] == pytest.approx(full_season_power)
    assert row_2016.loc["old-coach", "power_end"] != pytest.approx(full_season_power)


# ---------------------------------------------------------------------------
# inherited / program_par: keyed by stint, and nullable
# ---------------------------------------------------------------------------

def test_inherited_and_program_par_are_keyed_by_stint_not_by_coach_id() -> None:
    coach_season = _coach_season_rows(
        ("wanderer-2000", "School A", 2005, 12),
        ("wanderer-2000", "School B", 2010, 12),
    )
    ratings_history = pd.concat([
        _flat_ratings_range("School A", range(1995, 2005), power=5.0),
        _flat_ratings_range("School B", range(2000, 2010), power=-3.0),
        _flat_ratings_range("School A", [2005], power=8.0),
        _flat_ratings_range("School B", [2010], power=2.0),
    ], ignore_index=True)

    table = metrics.coach_season_metrics(coach_season, ratings_history, EMPTY_GAMES, EMPTY_GAMES, overrides={})
    a = table[table["school"] == "School A"].iloc[0]
    b = table[table["school"] == "School B"].iloc[0]

    assert a["inherited"] == pytest.approx(5.0)
    assert b["inherited"] == pytest.approx(-3.0)
    assert a["inherited"] != b["inherited"]
    assert a["program_par"] == pytest.approx(5.0)
    assert b["program_par"] == pytest.approx(-3.0)


def _flat_ratings_range(team: str, seasons, power: float) -> pd.DataFrame:
    return pd.DataFrame([{"team": team, "season": s, "power": power, "prior": 0.0, "resume": 0.0} for s in seasons])


def test_program_par_needs_three_seasons_but_inherited_needs_only_one() -> None:
    coach_season = _coach_season_rows(("newish-2010", "School C", 2010, 12))
    ratings_history = pd.DataFrame([
        {"team": "School C", "season": 2008, "power": 1.0, "prior": 0.0, "resume": 0.0},
        {"team": "School C", "season": 2009, "power": 2.0, "prior": 0.0, "resume": 0.0},
        {"team": "School C", "season": 2010, "power": 3.0, "prior": 0.5, "resume": 0.0},
    ])
    row = metrics.coach_season_metrics(coach_season, ratings_history, EMPTY_GAMES, EMPTY_GAMES, overrides={}).iloc[0]

    assert row["inherited"] == pytest.approx(2.0)   # only the one season right before is needed
    assert pd.isna(row["program_par"])               # 2 prior seasons of history isn't the required 3
    assert pd.isna(row["vs_par"])


def test_a_coach_with_no_prior_history_gets_a_null_inherited() -> None:
    coach_season = _coach_season_rows(("brandnew-2010", "School D", 2010, 12))
    ratings_history = pd.DataFrame([{"team": "School D", "season": 2010, "power": 0.0, "prior": 0.0, "resume": 0.0}])
    row = metrics.coach_season_metrics(coach_season, ratings_history, EMPTY_GAMES, EMPTY_GAMES, overrides={}).iloc[0]
    assert pd.isna(row["inherited"])
    assert pd.isna(row["program_par"])


# ---------------------------------------------------------------------------
# tenure summaries
# ---------------------------------------------------------------------------

def test_tenure_summaries_pick_best_and_worst_by_vs_par() -> None:
    metrics_table = pd.DataFrame([
        {"coach_id": "c1", "school": "School E", "season": 2018, "added": 5.0, "vs_par": 3.0},
        {"coach_id": "c1", "school": "School E", "season": 2019, "added": -2.0, "vs_par": -6.0},
        {"coach_id": "c1", "school": "School E", "season": 2020, "added": 1.0, "vs_par": 0.5},
    ])
    tenure = metrics.tenure_summaries(metrics_table)
    career = tenure[tenure["school"].isna()].iloc[0]

    assert career["best_season"] == 2018
    assert career["best_vs_par"] == pytest.approx(3.0)
    assert career["best_added"] == pytest.approx(5.0)
    assert career["worst_season"] == 2019
    assert career["worst_vs_par"] == pytest.approx(-6.0)
    assert career["mean_added"] == pytest.approx((5.0 - 2.0 + 1.0) / 3)
    assert career["mean_vs_par"] == pytest.approx((3.0 - 6.0 + 0.5) / 3)
