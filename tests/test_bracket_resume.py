"""A team's résumé, built from its own games: quadrant record, road wins, schedule
strength - checked against a small synthetic season where the right answer is
known by construction."""

from __future__ import annotations

import pandas as pd
import pytest

from mri.bracket import resume


def test_quadrant_thresholds_move_with_where_the_game_was_played() -> None:
    # The same opponent (rank 60) is Quad 1 away, Quad 2 at home or on a neutral floor.
    assert resume.quadrant(60, "away") == 1
    assert resume.quadrant(60, "home") == 2
    assert resume.quadrant(60, "neutral") == 2
    assert resume.quadrant(1, "home") == 1
    assert resume.quadrant(500, "away") == 4


def game(a, b, a_won, *, neutral=False):
    return {"team1": a, "team2": b, "win1": 1.0 if a_won else 0.0, "win2": 0.0 if a_won else 1.0, "neutral": neutral}


class FakeRatings:
    def __init__(self, power, resume_series=None):
        self.power = power
        self.resume = resume_series


def test_team_features_reads_record_location_and_quadrant_off_the_schedule() -> None:
    # A big enough universe (300 filler teams) that "rank 2" and "rank 250" actually land in
    # different quadrants - with only a handful of teams every opponent is nominally "top something".
    filler = {f"Filler{i}": -100.0 - i for i in range(300)}          # ranks 5..304, worse than everyone below
    power = pd.Series({"Alpha": 30.0, "Elite": 25.0, "Patsy": -400.0, "Ambush": -420.0, **filler})
    games = pd.DataFrame([
        game("Patsy", "Alpha", False),                        # Alpha (team2/home) beats a Quad-4 patsy at home
        game("Alpha", "Elite", True),                          # Alpha (team1/away) wins on the road, a Quad-1 win
    ] + [{"team1": "Ambush", "team2": "Alpha", "win1": 1.0, "win2": 0.0, "neutral": False}])  # a home loss to a Quad-4 team
    ratings = FakeRatings(power, pd.Series({"Alpha": 5.0}))
    conf_of = {"Alpha": "Test", "Elite": "Test", "Patsy": "Other", "Ambush": "Other"}
    feats = resume.team_features(2025, ratings, games, conf_of, {"Test": "Alpha"})
    alpha = feats[feats.team == "Alpha"].iloc[0]
    assert alpha.wins == 2 and alpha.losses == 1
    assert alpha.quad1_wins == 1 and alpha.road_neutral_wins == 1                # the road win over Elite
    assert alpha.quad4_losses == 1 and alpha.bad_losses == 1                    # the home loss to Ambush (rank worst)
    assert bool(alpha.isAutoBid) is True and alpha.resume == 5.0
    assert bool(feats[feats.team == "Elite"].iloc[0].isAutoBid) is False


def test_only_games_between_two_ranked_teams_count() -> None:
    power = pd.Series({"Alpha": 10.0})            # "Ghost" has no rank at all - not in the power series
    games = pd.DataFrame([game("Alpha", "Ghost", True)])
    feats = resume.team_features(2025, FakeRatings(power), games, {"Alpha": "Test"}, {})
    assert feats.empty            # neither side of an unranked game gets a resume entry from it
