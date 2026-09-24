"""A snapshot of September must not know how the bowls went.

The feed numbers the postseason from week 1 again and serves it alongside the
regular season, so anything that walks forward on ``week`` alone mixes January
into the first snapshot of every finished season. Every walk-forward runs on
``cfbd.sequence`` instead, and the postseason is one snapshot numbered after the
last regular week and labelled "Bowls" on the site.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mri.betting import tracker
from mri.export import simdata, site, sitedata
from mri.ingest import boxscores, cfbd

TEAMS = ["Alabama", "Georgia", "Ohio State", "Michigan"]


def season(*, bowls_played: bool = True) -> pd.DataFrame:
    """Two regular-season weeks, then a bowl the feed calls week 1."""
    rows = [
        (101, 1, "regular", "Alabama", "Georgia", True),
        (102, 1, "regular", "Ohio State", "Michigan", True),
        (201, 2, "regular", "Georgia", "Ohio State", True),
        (202, 2, "regular", "Michigan", "Alabama", True),
        (901, 1, "postseason", "Alabama", "Ohio State", bowls_played),
    ]
    return pd.DataFrame([
        {"game_id": gid, "season": 2025, "week": week, "season_type": kind,
         "team1": away, "team2": home, "played": played,
         "pts1": 14.0 if played else None, "pts2": 28.0 if played else None,
         "win1": 0.0, "win2": 1.0 if played else 0.0, "neutral": kind == "postseason",
         "conf1": None, "conf2": None, "conference_game": False,
         "start_date": None, "class1": "fbs", "class2": "fbs"}
        for gid, week, kind, away, home, played in rows
    ])


def fake_games(frame):
    def games(year, *, completed_only=True, **_):
        return frame[frame["played"]].reset_index(drop=True) if completed_only else frame.copy()
    return games


def test_sequence_puts_the_postseason_after_the_last_regular_week() -> None:
    blocks = cfbd.sequence(season())
    assert blocks.tolist() == [1, 1, 2, 2, 3]


# ---- the site's weekly ratings

@pytest.fixture
def fitted(monkeypatch):
    """Run weekly_ratings on the fake season, recording what each fit saw."""
    seen: list[set[int]] = []

    class Model:
        home_field = 2.5

        def __init__(self, games):
            self.games = games

        def table(self):
            return pd.DataFrame({"team": TEAMS, "power": [4.0, 3.0, 2.0, 1.0],
                                 "resume": [0.0] * 4, "resume_rank": [1, 2, 3, 4]})

    def fit(games, **_):
        seen.append(set(games["game_id"]))
        return Model(games)

    monkeypatch.setattr(sitedata.cfbd, "games", fake_games(season()))
    monkeypatch.setattr(sitedata, "_prior_for", lambda year: None)
    monkeypatch.setattr(sitedata.priors, "for_season", lambda *a, **k: None)
    monkeypatch.setattr(sitedata.mri2, "fit", fit)
    return sitedata.weekly_ratings(2025), seen


def test_week_one_snapshot_of_a_finished_season_has_no_postseason_games(fitted) -> None:
    weekly, seen = fitted
    assert seen[0] == {101, 102}
    assert seen[1] == {101, 102, 201, 202}
    assert 901 not in seen[0] | seen[1]


def test_the_postseason_is_one_snapshot_after_the_regular_season(fitted) -> None:
    weekly, seen = fitted
    assert len(seen) == 3 and 901 in seen[2]
    flags = weekly.groupby("week")["postseason"].first()
    assert flags.to_dict() == {1: False, 2: False, 3: True}


def test_postseason_block_is_numbered_from_the_whole_schedule(monkeypatch) -> None:
    # A bowl played before the last regular week has a final must still be the
    # block after it, not share its number: the label cannot move later.
    frame = season()
    frame.loc[frame["game_id"] == 202, ["played", "pts1", "pts2"]] = [False, None, None]
    monkeypatch.setattr(sitedata.cfbd, "games", fake_games(frame))
    games = sitedata._sequenced(2025)
    assert games.set_index("game_id")["block"].to_dict() == {101: 1, 102: 1, 201: 2, 901: 3}


# ---- the betting record

def test_a_bowl_is_priced_from_the_last_regular_week_not_the_preseason(monkeypatch) -> None:
    monkeypatch.setattr(tracker.cfbd, "games", fake_games(season()))
    monkeypatch.setattr(tracker.lines_module, "preferred_lines", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(tracker.board_module, "_previous_season", lambda year: pd.Series(0.0, index=TEAMS))
    monkeypatch.setattr(tracker.priors, "for_season", lambda *a, **k: pd.Series(0.0, index=TEAMS))
    rated = {1: [1.0, 1.0, 1.0, 1.0], 2: [10.0, 0.0, 30.0, 0.0], 3: [99.0, 0.0, -99.0, 0.0]}
    weekly = pd.concat([pd.DataFrame({"team": TEAMS, "power": p, "week": w, "home_field": 2.0})
                        for w, p in rated.items()])

    record = tracker.reconstruct(2025, {"teams": [{"team": t} for t in TEAMS]}, weekly)

    weeks = {w["week"]: w for w in record["weeks"]}
    assert sorted(weeks) == [1, 2, 3]
    assert weeks[3]["games"] == 1 and weeks[3]["label"] == cfbd.POSTSEASON_LABEL
    assert "label" not in weeks[1]
    # Ohio State (30) hosts Alabama (10) at a neutral site: +20 on the week-2
    # ratings. The preseason prior would have said 0; the bowl's own snapshot +198.
    assert record["summary"]["games"] == 5
    bowl_error = weeks[3]["mae"]
    assert bowl_error == pytest.approx(abs(20.0 - 14.0))


# ---- the season simulation

def test_simulation_week_one_hides_the_bowls() -> None:
    schedule = simdata._prepare(season())
    then = simdata._as_of(schedule, 1)
    assert set(then.loc[then["played"], "game_id"]) == {101, 102}
    assert then.set_index("game_id").loc[901, ["pts1", "pts2"]].isna().all()


# ---- box scores

def test_box_score_weeks_are_in_the_order_played(monkeypatch) -> None:
    monkeypatch.setattr(boxscores.cfbd, "games", fake_games(season().sort_values(["week", "game_id"])))
    assert boxscores.season_weeks(2025) == [("regular", 1), ("regular", 2), ("postseason", 1)]


# ---- the pages

def test_the_postseason_is_labelled_not_numbered() -> None:
    assert site.week_heading(7) == "Wk 7"
    assert site.week_heading(cfbd.POSTSEASON_LABEL) == "Bowls"
    assert site.period_text({"week": 17, "periodLabel": "Bowls"}) == "Bowls"
