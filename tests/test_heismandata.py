"""The Heisman page's data: history, movement, and the freeze when the ballots close."""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from mri.export import heismandata

PAYLOAD = {"week": 4, "teams": [{"team": "Alpha", "rank": 2, "wins": 4, "losses": 0}, {"team": "Beta", "rank": 30, "wins": 2, "losses": 2},
                                {"team": "Gamma", "rank": 9, "wins": 3, "losses": 1}]}


def odds(shift: float = 0.0) -> dict:
    rows = [("Ann Alpha", "Alpha", "QB", 0.30 + shift, 0.6, 1400, 12, 100, 1), ("Bob Beta", "Beta", "RB", 0.10 - shift, 0.3, 0, 0, 900, 8),
            ("Cy Gamma", "Gamma", "REC", 0.05, 0.1, 0, 0, 0, 0), ("Ann Alpha II", "Alpha", "QB", 0.02, 0.05, 900, 6, 0, 0)]
    frame = pd.DataFrame([{"player": n, "team": t, "group": g, "win": w, "finalist": f, "projected": 3000.0, "team_top4": 0.5,
                           "win_if_top4": 0.4 if t == "Alpha" else np.nan, "win_if_not": 0.1 if t == "Alpha" else np.nan,
                           "pass_yds": py, "pass_td": ptd, "rush_yds": ry, "rush_td": rtd, "rec_yds": 700 if g == "REC" else 0,
                           "rec_td": 5 if g == "REC" else 0}
                          for n, t, g, w, f, py, ptd, ry, rtd in rows])
    return {"season": 2026, "week": 4, "sims": 5000, "candidates": 65, "odds": frame}


def test_the_page_data_is_the_odds_made_readable(tmp_path) -> None:
    page = heismandata.build(2026, PAYLOAD, tmp_path / "h.json", today=dt.date(2026, 9, 27), odds_fn=lambda p, sims: odds())
    top = page["players"][0]
    assert (top["player"], top["team"], top["position"], top["win"]) == ("Ann Alpha", "Alpha", "QB", 0.3)
    assert top["teamRank"] == 2 and top["teamRecord"] == "4\u20130" and top["line"]["passYds"] == 1400
    assert top["winIfTop4"] == 0.4 and top["winIfNot"] == 0.1
    assert page["players"][1]["winIfTop4"] is None                      # a case too rare to say anything about is left empty
    assert top["pace"] == 1540 or top["pace"] > 0                       # yards, scaled by what the projection adds
    assert page["rest"] == pytest.approx(1 - sum(r["win"] for r in page["players"]), abs=1e-4)
    assert page["byPosition"] == {"QB": 0.32, "RB": 0.1, "WR/TE": 0.05}
    assert page["byTeam"]["Alpha"]["player"] == "Ann Alpha"             # a team's best candidate, not its second
    assert page["winners"]["2019"] == "Joe Burrow" and page["closed"] is False


def test_movement_is_measured_against_the_last_week_it_was_saved(tmp_path) -> None:
    path = tmp_path / "h.json"
    heismandata.build(2026, PAYLOAD, path, today=dt.date(2026, 9, 20), odds_fn=lambda p, sims: odds(0.0))
    PAYLOAD["week"] = 5
    try:
        page = heismandata.build(2026, PAYLOAD, path, today=dt.date(2026, 9, 27), odds_fn=lambda p, sims: odds(0.04))
    finally:
        PAYLOAD["week"] = 4
    rows = {r["player"]: r for r in page["players"]}
    assert rows["Ann Alpha"]["change"] == pytest.approx(0.04) and rows["Bob Beta"]["change"] == pytest.approx(-0.04)
    assert rows["Ann Alpha"]["rankChange"] == 0
    saved = json.loads(path.read_text())
    assert sorted(saved["weeks"]) == ["4", "5"]                        # both weeks are kept


def test_rerunning_a_week_replaces_it_instead_of_adding_to_it(tmp_path) -> None:
    path = tmp_path / "h.json"
    for shift in (0.0, 0.01):
        heismandata.build(2026, PAYLOAD, path, today=dt.date(2026, 9, 27), odds_fn=lambda p, sims, s=shift: odds(s))
    saved = json.loads(path.read_text())
    assert list(saved["weeks"]) == ["4"] and saved["weeks"]["4"]["odds"]["Ann Alpha|Alpha"][0] == 0.31


def test_a_new_season_does_not_inherit_last_seasons_history(tmp_path) -> None:
    path = tmp_path / "h.json"
    path.write_text(json.dumps({"season": 2025, "weeks": {"3": {"odds": {"Ann Alpha|Alpha": [0.9, 0.9]}, "page": {}}}}))
    page = heismandata.build(2026, PAYLOAD, path, today=dt.date(2026, 9, 27), odds_fn=lambda p, sims: odds())
    assert page["players"][0]["change"] is None


def test_nothing_is_shown_when_there_are_no_odds_to_show(tmp_path) -> None:
    assert heismandata.build(2026, PAYLOAD, tmp_path / "h.json", today=dt.date(2026, 9, 27), odds_fn=lambda p, sims: None) is None
    assert not (tmp_path / "h.json").exists()


def test_after_the_ballots_close_the_last_odds_are_shown_and_never_recomputed(tmp_path) -> None:
    path = tmp_path / "h.json"
    heismandata.build(2026, PAYLOAD, path, today=dt.date(2026, 12, 1), odds_fn=lambda p, sims: odds())

    def boom(payload, sims):
        raise AssertionError("odds were recomputed after the voting closed")

    page = heismandata.build(2026, PAYLOAD, path, today=dt.date(2026, 12, 8), odds_fn=boom)
    assert page["closed"] is True and page["players"][0]["player"] == "Ann Alpha" and page["updated"] == "2026-12-01"
    assert heismandata.build(2026, PAYLOAD, tmp_path / "none.json", today=dt.date(2026, 12, 8), odds_fn=boom) is None


# ---- the market benchmark, the finalist view, defenders

def market_file(tmp_path, as_of="2026-09-20", odds=None):
    path = tmp_path / "market.json"
    path.write_text(json.dumps({"asOf": as_of, "source": "A book", "note": "Includes the margin.",
                                "odds": odds or {"Ann Alpha": 310, "Bob Beta": -150}}))
    return path


def test_market_odds_become_implied_chances_for_the_players_on_the_page(tmp_path) -> None:
    rows = [{"player": "Ann Alpha"}, {"player": "Bob Beta"}, {"player": "Cy Gamma"}]
    info = heismandata.market_for(rows, dt.date(2026, 9, 22), market_file(tmp_path))
    assert info["asOf"] == "2026-09-20"
    assert rows[0]["market"] == {"odds": 310, "implied": pytest.approx(100 / 410, abs=1e-4)}
    assert rows[1]["market"]["implied"] == pytest.approx(0.6)          # a favorite at -150
    assert rows[2]["market"] is None


def test_stale_or_thin_market_data_is_not_shown(tmp_path) -> None:
    rows = [{"player": "Ann Alpha"}, {"player": "Bob Beta"}]
    assert heismandata.market_for(rows, dt.date(2026, 10, 5), market_file(tmp_path)) is None           # fifteen days old
    thin = [{"player": "Ann Alpha"}, {"player": "Cy Gamma"}]
    assert heismandata.market_for(thin, dt.date(2026, 9, 21), market_file(tmp_path)) is None           # only one match
    assert all(r["market"] is None for r in thin)
    assert heismandata.market_for(rows, dt.date(2026, 9, 21), tmp_path / "missing.json") is None


def test_the_defender_rate_comes_from_the_voting_record() -> None:
    from mri.heisman import data

    d = heismandata.defender_rate(data.load_voting())
    assert d["years"] == [2012, 2016, 2019, 2021] and d["count"] == 4 and d["seasons"] == 14
    assert "Chase Young" in d["names"] and "Jacob Rodriguez" not in d["names"]    # a top-ten finish is not an invitation


def test_the_page_data_carries_the_finalist_view_the_defenders_and_the_market(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(heismandata, "MARKET", market_file(tmp_path))
    page = heismandata.build(2026, PAYLOAD, tmp_path / "h.json", today=dt.date(2026, 9, 22), odds_fn=lambda p, sims: odds(),
                             defenders_fn=lambda payload: [{"player": "Dee Fender", "team": "Alpha", "position": "LB", "tackles": 30,
                                                            "sacks": 3.0, "interceptions": 1}])
    assert [r["player"] for r in page["reach"]][:2] == ["Ann Alpha", "Bob Beta"]        # most likely finalist first
    assert page["expectedFinalists"] == pytest.approx(sum(r["finalist"] for r in page["reach"]), abs=0.05)
    assert page["defenders"]["watch"][0]["player"] == "Dee Fender" and page["defenders"]["count"] == 4
    assert page["market"]["asOf"] == "2026-09-20" and page["players"][0]["market"]["odds"] == 310


def test_a_failing_defender_lookup_costs_the_page_only_the_names(tmp_path) -> None:
    def boom(payload):
        raise RuntimeError("the API was down")

    page = heismandata.build(2026, PAYLOAD, tmp_path / "h.json", today=dt.date(2026, 9, 22), odds_fn=lambda p, sims: odds(), defenders_fn=boom)
    assert page["defenders"]["watch"] == [] and page["players"]
