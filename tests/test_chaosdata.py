"""The Chaos Meter's pregame cache and frozen archive: write-once, and a final week never moves."""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from mri.export import chaosdata

KICKOFF = "2026-09-26T20:00:00.000Z"
NOW = dt.datetime(2026, 9, 24, 11, 0, tzinfo=dt.timezone.utc)


def fake_schedule() -> pd.DataFrame:
    return pd.DataFrame([
        {"game_id": 1, "week": 4, "team1": "Away U", "team2": "Home U",
         "start_date": KICKOFF, "neutral": False},
        {"game_id": 2, "week": 4, "team1": "Late A", "team2": "Late B",
         "start_date": "2026-09-24T09:00:00.000Z", "neutral": False},  # already kicked off at NOW
    ])


def slate_game(game_id: int, home_prob: float, played: bool = False) -> dict:
    return {"id": game_id, "homeWinProbability": home_prob, "played": played}


def slate(days_games=None, results=None, fcs=None) -> dict:
    return {"season": 2026, "week": 4,
            "days": [{"date": "2026-09-26", "label": "Sat", "games": days_games or []}],
            "results": results or [], "fcs": fcs or []}


@pytest.fixture
def wire(monkeypatch):
    state = {"schedule": fake_schedule()}
    monkeypatch.setattr(chaosdata.cfbd, "games", lambda *a, **k: state["schedule"])
    return state


# ---- freeze_pregame (spec tests 2, 3, 7)

def test_a_stored_pregame_probability_survives_a_changed_model(wire, tmp_path) -> None:
    path = tmp_path / "chaos_pregame.json"
    chaosdata.freeze_pregame(2026, slate(days_games=[slate_game(1, 0.62)]), path, now=NOW)
    first = json.loads(path.read_text())["games"]["1"]

    later = NOW + dt.timedelta(hours=1)
    chaosdata.freeze_pregame(2026, slate(days_games=[slate_game(1, 0.90)]), path, now=later)
    again = json.loads(path.read_text())["games"]["1"]
    assert again == first
    assert again["homeWinProbability"] == 0.62


def test_a_game_already_kicked_off_at_first_build_is_never_backfilled(wire, tmp_path) -> None:
    path = tmp_path / "chaos_pregame.json"
    # Game 2's 09:00 kickoff has already passed NOW (11:00) the very first time this runs.
    data = chaosdata.freeze_pregame(2026, slate(days_games=[slate_game(1, 0.5), slate_game(2, 0.5)]),
                                     path, now=NOW)
    assert "1" in data["games"] and "2" not in data["games"]

    # Never assigned one later either, once it might look like a game just not seen yet.
    chaosdata.freeze_pregame(2026, slate(days_games=[slate_game(1, 0.5), slate_game(2, 0.5)]),
                              path, now=NOW + dt.timedelta(days=1))
    assert "2" not in json.loads(path.read_text())["games"]


def test_a_played_game_is_never_assigned_a_probability(wire, tmp_path) -> None:
    path = tmp_path / "chaos_pregame.json"
    data = chaosdata.freeze_pregame(2026, slate(results=[slate_game(1, 0.5, played=True)]), path, now=NOW)
    assert "1" not in data["games"]


def test_freeze_pregame_does_nothing_when_slate_is_none(wire, tmp_path) -> None:
    path = tmp_path / "chaos_pregame.json"
    data = chaosdata.freeze_pregame(2026, None, path, now=NOW)
    assert data == {"season": 2026, "games": {}}
    assert not path.exists()


def test_pregame_cache_resets_when_the_season_rolls_over(wire, tmp_path) -> None:
    path = tmp_path / "chaos_pregame.json"
    chaosdata.freeze_pregame(2026, slate(days_games=[slate_game(1, 0.5)]), path, now=NOW)
    data = chaosdata.freeze_pregame(2027, slate(days_games=[slate_game(9, 0.5)]), path, now=NOW)
    assert data["season"] == 2027
    assert "1" not in data["games"] and "9" in data["games"]


def test_fcs_games_are_captured_too(wire, tmp_path) -> None:
    path = tmp_path / "chaos_pregame.json"
    data = chaosdata.freeze_pregame(2026, slate(fcs=[slate_game(1, 0.95)]), path, now=NOW)
    assert "1" in data["games"]


# ---- reconstruct_current_season (the launch catch-up fallback for a game the live cache never saw)

def test_reconstruct_current_season_matches_a_hand_computed_probability() -> None:
    from mri.betting import board
    from scipy.stats import norm

    schedule = pd.DataFrame([
        {"game_id": 1, "block": 2, "season_type": "regular", "team1": "Away", "team2": "Home",
         "played": True, "pts1": 10.0, "pts2": 24.0, "neutral": False},
    ])
    weekly = pd.DataFrame([
        {"week": 1, "team": "Home", "power": 5.0, "home_field": 2.5},
        {"week": 1, "team": "Away", "power": -3.0, "home_field": 2.5},
    ])
    result = chaosdata.reconstruct_current_season(2026, schedule, weekly)
    predicted = 5.0 - (-3.0) + 2.5
    assert result[1] == pytest.approx(float(norm.cdf(predicted / board.SIGMA)))


def test_reconstruct_current_season_prices_an_fcs_opponent_off_the_floor() -> None:
    from mri.betting import board
    from scipy.stats import norm

    schedule = pd.DataFrame([
        {"game_id": 1, "block": 2, "season_type": "regular", "team1": "Cupcake FCS", "team2": "Home",
         "played": True, "pts1": 3.0, "pts2": 45.0, "neutral": False},
    ])
    weekly = pd.DataFrame([
        {"week": 1, "team": "Home", "power": 10.0, "home_field": 2.5},
        {"week": 1, "team": "Someone Else", "power": -5.0, "home_field": 2.5},  # the field's floor
    ])
    result = chaosdata.reconstruct_current_season(2026, schedule, weekly)
    floor = -5.0 - 8.0  # Cupcake FCS never appears in `weekly` at all
    predicted = 10.0 - floor + 2.5
    assert result[1] == pytest.approx(float(norm.cdf(predicted / board.SIGMA)))


def test_reconstruct_current_season_skips_a_week_with_no_prior_ratings() -> None:
    schedule = pd.DataFrame([
        {"game_id": 1, "block": 5, "season_type": "regular", "team1": "A", "team2": "H",
         "played": True, "pts1": 10.0, "pts2": 20.0, "neutral": False},
    ])
    assert chaosdata.reconstruct_current_season(2026, schedule, pd.DataFrame()) == {}


def test_reconstruct_current_season_is_empty_when_nothing_has_been_played() -> None:
    schedule = pd.DataFrame([
        {"game_id": 1, "block": 1, "season_type": "regular", "team1": "A", "team2": "H",
         "played": False, "pts1": None, "pts2": None, "neutral": False},
    ])
    assert chaosdata.reconstruct_current_season(2026, schedule, pd.DataFrame()) == {}


def test_finalize_current_season_falls_back_to_reconstruction_when_pregame_cache_is_empty(tmp_path) -> None:
    """The launch scenario: a week finished before chaos_pregame.json ever captured anything for it.
    It should get a real score, not freeze forever as "not enough games"."""
    games = pd.DataFrame([
        {"game_id": i, "block": 4, "season_type": "regular", "team1": f"Away{i}", "team2": f"Home{i}",
         "played": True, "pts1": 10.0, "pts2": 24.0, "neutral": False}
        for i in range(1, 11)
    ])
    weekly = pd.DataFrame(
        [{"week": 3, "team": f"Home{i}", "power": 5.0, "home_field": 2.5} for i in range(1, 11)]
        + [{"week": 3, "team": f"Away{i}", "power": -5.0, "home_field": 2.5} for i in range(1, 11)]
    )
    history_path, sim_path = tmp_path / "chaos_history.json", tmp_path / "sim_history.json"
    data = chaosdata.finalize_current_season(2026, games, weekly, {"games": {}}, None,
                                              history_path, sim_path, now=NOW)
    entry = data["seasons"]["2026"]["weeks"]["4"]
    assert entry["missingPregame"] == 0
    assert entry["games"] == 10
    assert entry["reconstructed"] is True


# ---- finalize_current_season (spec tests 3, 4, 6) and fallout

def _ten_game_week(week: int = 4, prob: float = 0.7) -> tuple[pd.DataFrame, dict]:
    games = pd.DataFrame([
        {"game_id": i, "block": week, "season_type": "regular",
         "team1": f"Away{i}", "team2": f"Home{i}", "played": True, "pts1": 10.0, "pts2": 24.0,
         "neutral": False}
        for i in range(1, 11)
    ])
    pregame = {"season": 2026,
               "games": {str(i): {"homeWinProbability": prob, "loggedAt": "x"} for i in range(1, 11)}}
    return games, pregame


def test_finalizing_twice_does_not_change_an_already_final_week(tmp_path) -> None:
    games, pregame = _ten_game_week()
    history_path, sim_path = tmp_path / "chaos_history.json", tmp_path / "sim_history.json"

    chaosdata.finalize_current_season(2026, games, pd.DataFrame(), pregame, None, history_path, sim_path, now=NOW)
    first = json.loads(history_path.read_text())

    mutated_games = games.copy()
    mutated_games["pts2"] = 3.0  # would flip every result if the week were ever rescored
    mutated_pregame = {"season": 2026, "games": {k: {**v, "homeWinProbability": 0.1}
                                                  for k, v in pregame["games"].items()}}
    chaosdata.finalize_current_season(2026, mutated_games, pd.DataFrame(), mutated_pregame, None,
                                       history_path, sim_path, now=NOW)
    assert json.loads(history_path.read_text()) == first


def test_a_week_with_unplayed_games_is_not_finalized_yet(tmp_path) -> None:
    games = pd.DataFrame([{"game_id": 1, "block": 4, "season_type": "regular",
                            "team1": "A", "team2": "H", "played": False,
                            "pts1": None, "pts2": None}])
    history_path, sim_path = tmp_path / "chaos_history.json", tmp_path / "sim_history.json"
    data = chaosdata.finalize_current_season(2026, games, pd.DataFrame(), {"games": {}}, None,
                                              history_path, sim_path, now=NOW)
    assert "4" not in data["seasons"].get("2026", {}).get("weeks", {})


def test_a_week_under_ten_games_is_marked_not_enough_games(tmp_path) -> None:
    games, pregame = _ten_game_week(week=0)
    games = games.iloc[:6]  # under the MIN_GAMES floor
    pregame["games"] = {k: v for k, v in pregame["games"].items() if int(k) <= 6}
    history_path, sim_path = tmp_path / "chaos_history.json", tmp_path / "sim_history.json"
    data = chaosdata.finalize_current_season(2026, games, pd.DataFrame(), pregame, None, history_path, sim_path, now=NOW)
    entry = data["seasons"]["2026"]["weeks"]["0"]
    assert entry["z"] is None
    assert entry["note"] == "not_enough_games"
    assert entry["final"] is True


def test_missing_pregame_is_counted_not_silently_dropped(tmp_path) -> None:
    games, pregame = _ten_game_week()
    pregame["games"].pop("1")  # e.g. it kicked off before the feature's first build
    history_path, sim_path = tmp_path / "chaos_history.json", tmp_path / "sim_history.json"
    data = chaosdata.finalize_current_season(2026, games, pd.DataFrame(), pregame, None, history_path, sim_path, now=NOW)
    entry = data["seasons"]["2026"]["weeks"]["4"]
    assert entry["missingPregame"] == 1
    assert entry["games"] == 9


def test_burn_in_seasons_are_flagged_on_the_season_entry(tmp_path) -> None:
    games, pregame = _ten_game_week()
    history_path, sim_path = tmp_path / "chaos_history.json", tmp_path / "sim_history.json"
    data = chaosdata.finalize_current_season(1979, games, pd.DataFrame(), pregame, None, history_path, sim_path, now=NOW)
    assert data["seasons"]["1979"]["burnIn"] is True

    data2 = chaosdata.finalize_current_season(2026, games, pd.DataFrame(), pregame, None, history_path, sim_path, now=NOW)
    assert data2["seasons"]["2026"]["burnIn"] is False


def test_finalize_attaches_fallout_from_sim_history(tmp_path) -> None:
    games, pregame = _ten_game_week(week=4)
    sim_path = tmp_path / "sim_history.json"
    sim_path.write_text(json.dumps({"season": 2026, "weeks": {
        "3": {f"Home{i}": [0.5, 0.1, 0.1] for i in range(1, 11)},
        "4": {f"Home{i}": [0.6, 0.1, 0.1] for i in range(1, 11)},
    }}))
    history_path = tmp_path / "chaos_history.json"
    data = chaosdata.finalize_current_season(2026, games, pd.DataFrame(), pregame, None, history_path, sim_path, now=NOW)
    entry = data["seasons"]["2026"]["weeks"]["4"]
    assert entry["fallout"]["moved"] == pytest.approx(0.5)  # 10 teams each moved 0.1, /2


def test_fallout_for_week_computes_moved_and_top_movers() -> None:
    sim_history = {"weeks": {
        "3": {"A": [0.50, 0.1, 0.2], "B": [0.30, 0.05, 0.1], "C": [0.20, 0.02, 0.05]},
        "4": {"A": [0.70, 0.2, 0.3], "B": [0.10, 0.01, 0.02], "C": [0.20, 0.02, 0.05]},
    }}
    result = chaosdata.fallout_for_week(sim_history, 4)
    assert result["moved"] == pytest.approx((0.20 + 0.20) / 2)
    assert result["gained"][0]["team"] == "A"
    assert result["lost"][0]["team"] == "B"


def test_fallout_for_week_is_none_for_week_one_or_missing_history() -> None:
    assert chaosdata.fallout_for_week({"weeks": {"1": {}}}, 1) is None
    assert chaosdata.fallout_for_week({}, 5) is None


# ---- ranking population and percentile-at-freeze-time

def test_ranking_population_splits_bowls_from_regular_and_excludes_burn_in() -> None:
    history_data = {"seasons": {
        "1979": {"burnIn": True, "weeks": {"1": {"z": 9.0}}},  # excluded: burn-in
        "2020": {"burnIn": False, "weeks": {
            "1": {"z": 1.0}, "2": {"z": None},  # excluded: not scored
            "16": {"z": 2.0, "label": "Bowls"},
        }},
        "2021": {"burnIn": False, "weeks": {"1": {"z": 3.0}}},
    }}
    assert sorted(chaosdata.ranking_population(history_data, bowls=False)) == [1.0, 3.0]
    assert chaosdata.ranking_population(history_data, bowls=True) == [2.0]


def test_finalize_current_season_computes_percentile_against_the_archive_at_freeze_time(tmp_path) -> None:
    history_path, sim_path = tmp_path / "chaos_history.json", tmp_path / "sim_history.json"
    # Seed an existing archive of 4 calmer weeks (z=0) that a new, wilder week should out-rank.
    seeded = {"seasons": {"2020": {"burnIn": False, "weeks": {
        str(w): {"z": 0.0, "games": 20, "upsets": 2, "expectedUpsets": 2.0, "shocks": [],
                 "missingPregame": 0, "final": True, "reconstructed": True, "percentile": 50.0}
        for w in range(1, 5)
    }}}}
    chaosdata.save_history(history_path, seeded)

    games, pregame = _ten_game_week(week=4, prob=0.05)  # home was a heavy underdog in every game, and won
    data = chaosdata.finalize_current_season(2026, games, pd.DataFrame(), pregame, None, history_path, sim_path, now=NOW)
    new_entry = data["seasons"]["2026"]["weeks"]["4"]
    assert new_entry["percentile"] == pytest.approx(100.0)  # wilder than all 4 seeded weeks

    # And the seeded weeks' own stored percentiles were not touched by this later, wilder week.
    assert data["seasons"]["2020"]["weeks"]["1"]["percentile"] == 50.0


# ---- current_reading

def test_current_reading_shows_expected_upsets_before_any_games_played() -> None:
    games = pd.DataFrame([{"game_id": 1, "block": 5, "played": False, "pts1": None, "pts2": None,
                            "team1": "A", "team2": "H", "season_type": "regular"}])
    pregame = {"games": {"1": {"homeWinProbability": 0.3, "loggedAt": "x"}}}
    reading = chaosdata.current_reading(games, 5, pregame)
    assert reading["partial"] is True
    assert reading["gamesPlayed"] == 0
    assert reading["expectedUpsets"] == pytest.approx(0.3)


def test_current_reading_scores_games_played_so_far() -> None:
    games, pregame = _ten_game_week(week=6)
    games.loc[games["game_id"] > 3, "played"] = False
    games.loc[games["game_id"] > 3, ["pts1", "pts2"]] = None
    reading = chaosdata.current_reading(games, 6, pregame)
    assert reading["partial"] is True
    assert reading["gamesPlayed"] == 3


# ---- build (the orchestrating entry point build_site.py calls)

def _full_schedule(week: int = 4, played: bool = True) -> pd.DataFrame:
    return pd.DataFrame([
        {"game_id": i, "week": week, "season_type": "regular", "team1": f"Away{i}", "team2": f"Home{i}",
         "played": played, "pts1": 10.0, "pts2": 24.0, "start_date": KICKOFF, "neutral": False}
        for i in range(1, 11)
    ])


def test_build_returns_none_and_still_freezes_pregame_when_history_is_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(chaosdata.cfbd, "games", lambda *a, **k: _full_schedule(played=False))
    s = slate(days_games=[slate_game(i, 0.6) for i in range(1, 11)])
    s["week"] = 4
    result = chaosdata.build(2026, s, pd.DataFrame(), tmp_path)
    assert result is None
    assert (tmp_path / chaosdata.PREGAME_PATH_NAME).exists()


def test_build_freezes_the_current_week_once_it_is_final(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(chaosdata.cfbd, "games", lambda *a, **k: _full_schedule(played=True))
    chaosdata.save_history(tmp_path / chaosdata.HISTORY_PATH_NAME, {"seasons": {}})
    (tmp_path / chaosdata.PREGAME_PATH_NAME).write_text(json.dumps({
        "season": 2026,
        "games": {str(i): {"homeWinProbability": 0.6, "loggedAt": "x"} for i in range(1, 11)},
    }))
    s = slate(days_games=[])
    s["week"] = 4

    result = chaosdata.build(2026, s, pd.DataFrame(), tmp_path)
    assert result is not None
    entry = result["archive"]["seasons"]["2026"]["weeks"]["4"]
    assert entry["final"] is True
    assert entry["games"] == 10
    assert result["current"]["gamesPlayed"] == 10


# ---- isolation (spec test 7)

def test_finalize_current_season_with_no_sim_history_file_still_works(tmp_path) -> None:
    games, pregame = _ten_game_week()
    history_path = tmp_path / "chaos_history.json"
    sim_path = tmp_path / "sim_history.json"  # never written
    data = chaosdata.finalize_current_season(2026, games, pd.DataFrame(), pregame, None, history_path, sim_path, now=NOW)
    entry = data["seasons"]["2026"]["weeks"]["4"]
    assert "fallout" not in entry
