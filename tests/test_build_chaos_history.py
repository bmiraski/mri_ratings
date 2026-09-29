"""The historical slate backfill: real kickoff-time handling, day-bucketing, and write-once."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
bch = importlib.import_module("build_chaos_history")


def _week(week: int = 3, n: int = 12, *, fbs_both: bool = True, season_type: str = "regular",
          start_date: str = "2015-09-19T19:30:00.000Z", start_time_tbd: bool = False, offset: int = 0
          ) -> pd.DataFrame:
    return pd.DataFrame([
        {"game_id": 1000 + offset + i, "season": 1987, "week": week, "home_team": f"Home{i}",
         "away_team": f"Away{i}", "neutral": False, "fbs_both": fbs_both, "base_line": 6.0,
         "home_win_prob": 0.7, "season_type": season_type, "pts1": 10, "pts2": 24,
         "start_date": start_date, "start_time_tbd": start_time_tbd}
        for i in range(n)
    ])


# ---- _game_kickoff

def test_game_kickoff_formats_a_real_timestamp() -> None:
    date, label, time, sort = bch._game_kickoff("2015-09-19T19:30:00.000Z")
    assert date == "2015-09-19"
    assert label == "Sat, Sep 19"
    assert time.endswith("ET") and "TBD" not in time
    assert sort != "9999"


def test_game_kickoff_treats_midnight_utc_as_date_only() -> None:
    """CFBD's older seasons store a date-only kickoff as midnight UTC with nothing real behind it -
    this must not shift the calendar date back a day or print a fabricated time."""
    date, label, time, sort = bch._game_kickoff("1987-09-12T00:00:00.000Z")
    assert date == "1987-09-12"          # not shifted to 09-11 by an Eastern-time conversion
    assert label == "Sat, Sep 12"
    assert time == "TBD"
    assert sort == "9999"


def test_game_kickoff_handles_a_missing_date() -> None:
    assert bch._game_kickoff(None) == (None, "Date unknown", "TBD", "9999")
    assert bch._game_kickoff(float("nan")) == (None, "Date unknown", "TBD", "9999")


# ---- _slate_archive_games / _bucket_days

def test_slate_archive_games_splits_fbs_and_fcs() -> None:
    games = pd.concat([_week(n=2, fbs_both=True), _week(n=1, fbs_both=False, offset=100)], ignore_index=True)
    days, finals, fcs = bch._slate_archive_games(games)
    assert len(days) == 2
    assert len(fcs) == 1
    assert len(finals) == 3
    g = days[0]
    assert g["home"] == "Home0" and g["away"] == "Away0"
    assert g["homeWinProbability"] == pytest.approx(0.7)
    assert g["predicted"] == pytest.approx(6.0)
    assert g["played"] is True


def test_slate_archive_games_finals_match_pts() -> None:
    games = _week(n=1)
    _, finals, _ = bch._slate_archive_games(games)
    assert finals["1000"] == {"home": 24, "away": 10}


def test_bucket_days_groups_by_date_in_order() -> None:
    entries = [
        {"date": "2015-09-19", "dateLabel": "Sat, Sep 19", "sort": "1200", "id": 1},
        {"date": "2015-09-18", "dateLabel": "Fri, Sep 18", "sort": "2000", "id": 2},
        {"date": "2015-09-19", "dateLabel": "Sat, Sep 19", "sort": "1500", "id": 3},
    ]
    days = bch._bucket_days(entries)
    assert [d["date"] for d in days] == ["2015-09-18", "2015-09-19"]
    assert [g["id"] for g in days[1]["games"]] == [1, 3]  # sorted within the day by kickoff time


# ---- _slate_archive_record

def test_slate_archive_record_shape() -> None:
    games = _week(n=12)
    record = bch._slate_archive_record(1987, 3, games, [{"team": "Home0", "rank": 5, "color": "#fff"}])
    assert record["season"] == 1987 and record["week"] == 3
    assert "label" not in record
    assert record["watch"] == [] and record["results"] == []
    assert record["games"] == 12
    assert record["teams"]["Home0"]["rank"] == 5
    assert "Home1" not in record["teams"]  # not in the payload's team list - degrades to plain text


def test_slate_archive_record_labels_bowls() -> None:
    games = _week(n=12, season_type="postseason")
    record = bch._slate_archive_record(1987, 16, games, [])
    assert record["label"] == bch.cfbd.POSTSEASON_LABEL


# ---- _write_slate_archives

def test_write_slate_archives_writes_one_file_per_week(tmp_path) -> None:
    pregame = pd.concat([_week(week=3, n=12), _week(week=4, n=12, offset=100)], ignore_index=True)
    bch._write_slate_archives(pregame, tmp_path, [])
    assert (tmp_path / "slate" / "1987-week-3.json").exists()
    assert (tmp_path / "slate" / "1987-week-4.json").exists()


def test_write_slate_archives_skips_a_week_under_the_games_floor(tmp_path) -> None:
    pregame = _week(week=3, n=3)  # under chaos.MIN_GAMES
    bch._write_slate_archives(pregame, tmp_path, [])
    assert not (tmp_path / "slate" / "1987-week-3.json").exists()


def test_write_slate_archives_never_overwrites_an_existing_file(tmp_path) -> None:
    pregame = _week(week=3, n=12)
    bch._write_slate_archives(pregame, tmp_path, [])
    path = tmp_path / "slate" / "1987-week-3.json"
    before = path.read_text()

    changed = pregame.copy()
    changed["home_win_prob"] = 0.01
    bch._write_slate_archives(changed, tmp_path, [])
    assert path.read_text() == before
