"""Historical slate pages show each team's rank as of that week, not today's."""

from __future__ import annotations

import json

import pandas as pd

from mri.export import slatearchive
from mri.ratings import hfa

TEAMS = [
    {"team": "A", "rank": 1, "abbreviation": "AA", "color": "#111", "logo": "a.png"},
    {"team": "B", "rank": 2, "abbreviation": "BB", "color": "#222", "logo": "b.png"},
    {"team": "Z", "rank": 3, "abbreviation": "ZZ", "color": "#333", "logo": "z.png"},   # not FBS that season
]


def test_teams_carry_the_week_rank_and_timeless_identity() -> None:
    teams = slatearchive.historical_teams(TEAMS, {"A": 7, "B": 1})
    assert teams["A"] == {"rank": 7, "abbreviation": "AA", "color": "#111", "logo": "a.png"}
    assert teams["B"]["rank"] == 1


def test_a_team_outside_that_seasons_field_gets_no_rank_at_all() -> None:
    assert "Z" not in slatearchive.historical_teams(TEAMS, {"A": 1, "B": 2})


def _write(path, saved, teams) -> None:
    path.write_text(json.dumps({"season": 1999, "week": 3, "saved": saved, "teams": teams,
                                "days": [{"games": [{"id": 1, "homeWinProbability": 0.61}]}],
                                "finals": {"1": {"home": 24, "away": 10}}}))


def test_refresh_rewrites_only_the_teams_of_reconstructed_archives(tmp_path) -> None:
    folder = tmp_path / "slate"
    folder.mkdir()
    stale = {"A": {"rank": 1}}
    _write(folder / "1999-week-3.json", "1999-week-3-historical-backfill", stale)
    live = folder / "2026-week-3.json"
    live.write_text(json.dumps({"season": 2026, "week": 3, "saved": "2026-09-20T12:00:00Z", "teams": stale}))

    changed = slatearchive.refresh_historical_ranks(tmp_path, {(1999, 3): {"A": 9, "B": 2}}, TEAMS)

    assert changed == 1
    data = json.loads((folder / "1999-week-3.json").read_text())
    assert data["teams"]["A"]["rank"] == 9 and "Z" not in data["teams"]
    assert data["days"][0]["games"][0]["homeWinProbability"] == 0.61      # frozen numbers untouched
    assert data["finals"] == {"1": {"home": 24, "away": 10}}
    assert json.loads(live.read_text())["teams"] == stale                  # live weeks are never rewritten
    assert slatearchive.refresh_historical_ranks(tmp_path, {(1999, 3): {"A": 9, "B": 2}}, TEAMS) == 0


def test_walk_forward_ranks_follow_the_ratings_at_the_start_of_each_week(monkeypatch) -> None:
    def game(game_id, week, away, home, pts1, pts2):
        return {"game_id": game_id, "week": week, "season_type": "regular", "team1": away, "team2": home,
                "pts1": pts1, "pts2": pts2, "neutral": False, "class1": "fbs", "class2": "fbs"}

    season = pd.DataFrame([
        game(1, 1, "A", "B", 3.0, 40.0),    # B routs A in week 1
        game(2, 1, "C", "D", 10.0, 10.5),
        game(3, 2, "A", "C", 0.0, 3.0),
        game(4, 2, "B", "D", 30.0, 3.0),
    ])
    monkeypatch.setattr(hfa.cfbd, "games", lambda year: pd.DataFrame() if year == 1977 else season)
    monkeypatch.setattr(hfa.registry, "was_fbs", lambda name, season=None: True)
    monkeypatch.setattr(hfa.registry, "is_fbs", lambda name: True)

    ranks: dict = {}
    hfa.walk_forward([1978], ranks=ranks)

    assert set(ranks) == {(1978, 1), (1978, 2)}
    assert set(ranks[(1978, 1)].values()) == {1, 2, 3, 4}         # a full ordering of the field
    assert ranks[(1978, 2)]["B"] == 1 and ranks[(1978, 2)]["A"] == 4   # week 2 has seen week 1; week 1 had not
