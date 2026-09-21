"""The Heisman record must be right, and every name in it must find its player.

Phase 0 of the Heisman work is data, and data is where a project like this quietly
goes wrong: a mistyped school makes a finalist vanish from a fit and nothing
complains. These tests are the complaint.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mri.heisman import data, funnel
from mri.ingest import players

VOTING = data.load_voting()

WINNERS = {
    2005: "Reggie Bush", 2006: "Troy Smith", 2007: "Tim Tebow", 2008: "Sam Bradford", 2009: "Mark Ingram",
    2010: "Cam Newton", 2011: "Robert Griffin III", 2012: "Johnny Manziel", 2013: "Jameis Winston",
    2014: "Marcus Mariota", 2015: "Derrick Henry", 2016: "Lamar Jackson", 2017: "Baker Mayfield",
    2018: "Kyler Murray", 2019: "Joe Burrow", 2020: "DeVonta Smith", 2021: "Bryce Young", 2022: "Caleb Williams",
    2023: "Jayden Daniels", 2024: "Travis Hunter", 2025: "Fernando Mendoza",
}

# Finalists the stat feed does not carry: 2009-2011 have holes, including Auburn's whole 2010 offense.
FEED_GAPS = {(2009, "Ndamukong Suh"), (2010, "Cam Newton"), (2010, "LaMichael James"), (2011, "Tyrann Mathieu")}


# ---- the voting file

def test_every_season_has_exactly_the_winner_it_should() -> None:
    assert sorted(int(y) for y in VOTING["seasons"]) == sorted(WINNERS)
    for year, name in WINNERS.items():
        first = VOTING["seasons"][str(year)]["finalists"][0]
        assert first["player"] == name and first["finish"] == 1 and first["points"] > 0, year


def test_finishes_and_points_agree_where_both_are_known() -> None:
    for year, season in VOTING["seasons"].items():
        known = [f for f in season["finalists"] if f["finish"] is not None and f["points"] is not None]
        known.sort(key=lambda f: f["finish"])
        points = [f["points"] for f in known]
        assert points == sorted(points, reverse=True), f"{year}: a lower finish has more points"
        finishes = [f["finish"] for f in season["finalists"] if f["finish"] is not None]
        assert len(finishes) == len(set(finishes)), f"{year}: two players share a finish"
        names = [f["player"] for f in season["finalists"]]
        assert len(names) == len(set(names)), f"{year}: a finalist is listed twice"


def test_points_are_possible_given_the_ballots() -> None:
    for year, season in VOTING["seasons"].items():
        if season.get("voters"):
            ceiling = 3 * season["voters"]
            assert all((f["points"] or 0) <= ceiling for f in season["finalists"]), year


def test_the_ceremony_dates_are_in_order() -> None:
    d = VOTING["keyDates2026"]
    assert d["ballotsDistributed"] < d["votingDeadline"] <= d["finalistsAnnounced"] < d["ceremony"]


# ---- names and schools

def test_names_are_compared_the_way_people_write_them() -> None:
    n = data.normalize
    assert n("C.J. Stroud") == n("CJ Stroud") == "cj stroud"
    assert n("Marvin Harrison Jr.") == "marvin harrison" == n("Marvin Harrison")
    assert n("Manti Te'o") == "manti teo" and n("Robert Griffin III") == "robert griffin"
    assert n("José Núñez") == "jose nunez"


def test_a_nickname_is_the_same_person_but_a_stranger_is_not() -> None:
    same = data.same_person
    assert same("cam ward", "cameron ward") and same("cameron ward", "cam ward")
    assert not same("cam ward", "cam wards") and not same("ed ward", "ted ward")
    assert not same("al smith", "alan jones") and not same("cam ward", "ward")


def test_resolve_stays_inside_the_school_and_the_season() -> None:
    table = pd.DataFrame([
        {"season": 2024, "team": "Miami", "player": "Cameron Ward", "player_id": "1"},
        {"season": 2024, "team": "Ole Miss", "player": "Cameron Ward", "player_id": "2"},
        {"season": 2023, "team": "Miami", "player": "Cameron Ward", "player_id": "3"},
    ])
    found = data.resolve(table, 2024, "Cam Ward", "Miami (Fla)")
    assert found["player_id"].tolist() == ["1"]
    assert data.resolve(table, 2024, "Cam Ward", "Mississippi")["player_id"].tolist() == ["2"]
    assert data.resolve(table, 2022, "Cam Ward", "Miami").empty


# ---- the joined record

PLAYERS = data.load_players()


def test_the_player_table_has_no_duplicates_and_covers_every_season_since_2009() -> None:
    assert not PLAYERS.duplicated(["season", "player_id", "team"]).any()
    assert sorted(PLAYERS["season"].unique()) == list(range(2009, 2026))
    assert set(players.COLUMNS.values()) <= set(PLAYERS.columns)


def test_every_finalist_since_2012_finds_exactly_one_player() -> None:
    problems = []
    for f in data.finalists(VOTING, first=2009):
        found = data.resolve(PLAYERS, f["season"], f["player"], f["school"])
        if (f["season"], f["player"]) in FEED_GAPS:
            if len(found):
                problems.append(f"{f['season']} {f['player']} is in the feed now: drop him from FEED_GAPS")
        elif len(found) != 1:
            problems.append(f"{f['season']} {f['player']} ({f['school']}): {len(found)} matches")
    assert not problems, problems


def test_every_winner_has_a_season_worth_voting_for() -> None:
    for f in data.finalists(VOTING, first=2012):
        if f["finish"] != 1:
            continue
        row = funnel.classify(data.resolve(PLAYERS, f["season"], f["player"], f["school"])).iloc[0]
        assert row["tot_yds"] >= 1000 or row["def_score"] >= 100, (f["season"], f["player"], row["tot_yds"])


def test_the_table_is_only_fbs_and_only_notable() -> None:
    per_season = PLAYERS.groupby("season")["team"].nunique()
    assert per_season.max() <= 140, "FCS teams are back in the table"
    tiny = PLAYERS[(PLAYERS[["pass_att", "rush_car", "rec_rec", "def_tot", "def_sacks", "def_tfl", "def_int"]] == 0).all(axis=1)]
    assert tiny.empty


# ---- the pivot

def fake_feed(monkeypatch, *, fbs=("Indiana", "Ohio State")):
    def stat(pid, name, team, pos, category, kind, value):
        return {"playerId": pid, "player": name, "team": team, "position": pos, "conference": "Big Ten",
                "category": category, "statType": kind, "stat": str(value)}

    rows = {
        "passing": [stat("1", "Star QB", "Indiana", "QB", "passing", "ATT", 300), stat("1", "Star QB", "Indiana", "QB", "passing", "YDS", 3500),
                    stat("1", "Star QB", "Indiana", "QB", "passing", "TD", 33),
                    stat("2", "Backup", "Indiana", "QB", "passing", "ATT", 12), stat("2", "Backup", "Indiana", "QB", "passing", "YDS", 80),
                    stat("9", "FCS QB", "Some FCS", "QB", "passing", "ATT", 400)],
        "rushing": [stat("3", "Runner", "Ohio State", "RB", "rushing", "CAR", 200), stat("3", "Runner", "Ohio State", "RB", "rushing", "YDS", 1100),
                    stat("3", "Runner", "Ohio State", "RB", "rushing", "LONG", 70)],
        "receiving": [], "defensive": [], "interceptions": [],
    }
    monkeypatch.setattr(players, "category_rows", lambda year, category, refresh=None: rows[category])
    monkeypatch.setattr(players, "ppa_rows", lambda year, refresh=None: [
        {"id": "1", "team": "Indiana", "averagePPA": {"all": 0.6}, "totalPPA": {"all": 180.0, "pass": 170.0, "rush": 10.0}}])
    monkeypatch.setattr(players, "usage_rows", lambda year, refresh=None: [
        {"id": "1", "team": "Indiana", "usage": {"overall": 0.4}}])
    monkeypatch.setattr(players.cfbd, "fbs_teams", lambda year: pd.DataFrame({"team": list(fbs)}))


def test_the_pivot_makes_one_wide_row_per_player_and_keeps_only_the_notable_fbs(monkeypatch) -> None:
    fake_feed(monkeypatch)
    table = players.season_table(2024)
    assert sorted(table["player"]) == ["Runner", "Star QB"]              # the backup and the FCS passer are gone
    qb = table[table["player"] == "Star QB"].iloc[0]
    assert (qb["pass_att"], qb["pass_yds"], qb["pass_td"], qb["rush_car"]) == (300, 3500, 33, 0)
    assert qb["ppa_total"] == 180.0 and qb["usage"] == 0.4 and qb["ppa_avg"] == 0.6
    everyone = players.season_table(2024, keep_all=True)
    assert "Backup" in set(everyone["player"]) and "FCS QB" not in set(everyone["player"])


def test_advanced_numbers_exist_only_from_2013(monkeypatch) -> None:
    fake_feed(monkeypatch)
    assert "ppa_total" not in players.season_table(2012).columns
    assert "ppa_total" in players.season_table(2013).columns


# ---- the funnel

def test_the_funnel_keeps_every_winner_since_2013_and_most_finalists() -> None:
    from mri.ratings import prior_fit

    power, _ = prior_fit.season_ratings()
    kept_finalists = total = 0
    for year in range(2013, 2026):
        rank = power[year].rank(ascending=False)
        pool = funnel.candidates(PLAYERS[PLAYERS["season"] == year], rank)
        assert 60 <= len(pool) <= 80, (year, len(pool))
        for f in VOTING["seasons"][str(year)]["finalists"]:
            if f.get("finalist") is False:
                continue
            found = data.resolve(PLAYERS, year, f["player"], f["school"])
            total += 1
            hit = bool(len(found) and found.index[0] in pool.index)
            kept_finalists += hit
            if f["finish"] == 1:
                assert hit, f"the {year} winner, {f['player']}, would not have been a candidate"
    assert kept_finalists / total >= 0.85, (kept_finalists, total)


def test_defenders_are_marked_in_the_voting_record() -> None:
    marked = [(int(y), f["player"]) for y, s in VOTING["seasons"].items() for f in s["finalists"] if f.get("defender")]
    assert sorted(marked) == [(2012, "Manti Te'o"), (2016, "Jabrill Peppers"), (2019, "Chase Young"), (2021, "Aidan Hutchinson"),
                              (2025, "Caleb Downs"), (2025, "Jacob Rodriguez")]
