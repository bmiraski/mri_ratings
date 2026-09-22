import numpy as np
import pandas as pd
import pytest

from mri.bracket import regions, seeding


def _seeded(conferences: dict[int, str] | None = None, size: int = 68) -> pd.DataFrame:
    """A field of distinct-conference teams T1..Tn in true-seed order, with chosen overrides."""
    fmt = seeding.FORMATS[size]
    autos = 31 if size == 68 else 32
    is_auto = np.array([False] * (size - autos) + [True] * autos)
    line, opening = seeding.lines(is_auto, fmt)
    conferences = conferences or {}
    return pd.DataFrame({
        "team": [f"T{i}" for i in range(1, size + 1)],
        "conference": [conferences.get(i, f"C{i}") for i in range(1, size + 1)],
        "trueSeed": np.arange(1, size + 1), "seedLine": line, "opening": opening,
        "bidType": np.where(is_auto, "auto", "at-large"),
    })


def test_earliest_meeting():
    assert regions.earliest_meeting(1, 16) == 1
    assert regions.earliest_meeting(1, 8) == 2
    assert regions.earliest_meeting(1, 5) == 3
    assert regions.earliest_meeting(1, 2) == 4
    assert regions.earliest_meeting(6, 3) == 2
    assert regions.earliest_meeting(5, 12) == 1


def test_no_conflicts_is_a_pure_s_curve():
    seeded = _seeded()
    pl = regions.place(seeded)
    region = pl.region_of
    assert [region[f"T{i}"] for i in range(1, 5)] == [1, 2, 3, 4]
    assert [region[f"T{i}"] for i in range(5, 9)] == [4, 3, 2, 1]        # best No. 2 with the weakest No. 1
    assert [region[f"T{i}"] for i in range(9, 13)] == [1, 2, 3, 4]
    totals = pl.region_totals(dict(zip(seeded.team, seeded.trueSeed)))
    assert max(totals.values()) - min(totals.values()) <= regions.BALANCE_LIMIT
    assert regions.check(seeded, pl) == [] and not pl.moved


def test_every_region_has_one_of_each_line():
    seeded = _seeded(size=76)
    pl = regions.place(seeded)
    table = pl.table(seeded)
    direct_or_game = table.drop_duplicates(["region", "seedLine", "openingGame"])
    for r in range(1, 5):
        lines = direct_or_game[direct_or_game.region == r]["seedLine"].value_counts()
        assert set(lines.index) == set(range(1, 17)) and (lines == 1).all()


def test_top_four_lines_keep_a_conferences_first_four_apart():
    # T1 (a No. 1) and T8 (the weakest No. 2) share a conference; the S-curve puts T8 in T1's region.
    seeded = _seeded({1: "X", 8: "X"})
    pl = regions.place(seeded)
    assert pl.region_of["T1"] != pl.region_of["T8"]
    assert regions.check(seeded, pl) == []


def test_fifth_team_on_the_top_four_lines_is_exempt():
    seeded = _seeded({1: "X", 2: "X", 3: "X", 4: "X", 5: "X"})
    pl = regions.place(seeded)
    assert len({pl.region_of[f"T{i}"] for i in range(1, 5)}) == 4
    assert regions.check(seeded, pl) == []


def test_played_three_times_means_not_before_the_regional_final():
    # T17 (a No. 5) and T44 (a No. 12 in the 68 field) - conference-mates who played three times.
    seeded = _seeded({17: "X", 44: "X"})
    meetings = {frozenset(("T17", "T44")): 3}
    pl = regions.place(seeded, meetings)
    if pl.region_of["T17"] == pl.region_of["T44"]:
        assert regions.earliest_meeting(pl.line_of["T17"], pl.line_of["T44"]) == 4
    assert regions.check(seeded, pl, meetings) == []


def test_played_once_only_rules_out_the_first_round():
    assert regions.required_round(1) == 2 and regions.required_round(0) == 2
    assert regions.required_round(2) == 3 and regions.required_round(3) == 4


def test_overall_five_not_in_overall_ones_region():
    seeded = _seeded()
    pl = regions.place(seeded)
    assert pl.region_of["T5"] != 1


def test_opening_round_conference_mates_are_split_when_possible():
    seeded = _seeded()
    or_teams = seeded[seeded.opening & (seeded.bidType == "at-large")]["team"].tolist()     # T34..T37 on line 11
    seeded.loc[seeded.team.isin(or_teams[:2]), "conference"] = "X"
    pl = regions.place(seeded)
    games = [set(u.teams) for u in pl.units if u.is_game]
    assert set(or_teams[:2]) not in games
    assert regions.check(seeded, pl) == []


def test_unavoidable_opening_round_rematch_is_reported_not_hidden():
    seeded = _seeded()
    or_teams = seeded[seeded.opening & (seeded.bidType == "at-large")]["team"].tolist()
    seeded.loc[seeded.team.isin(or_teams[:3]), "conference"] = "X"
    pl = regions.place(seeded)
    problems = regions.check(seeded, pl)
    assert len(problems) == 1 and "Opening Round" in problems[0]


def test_meetings_from_games():
    g = pd.DataFrame({"team1": ["A", "B", "A"], "team2": ["B", "A", "C"]})
    m = regions.meetings_from_games(g)
    assert m[frozenset(("A", "B"))] == 2 and m[frozenset(("A", "C"))] == 1


def test_big_conference_stress_leaves_only_unavoidable_problems():
    """Nine teams from one conference, every pair having played twice (so no two may share a pod), at
    random places on lines 1-12: whatever can be placed legally is, and anything left is the one case no
    placement fixes - three or more of a line's four Opening Round teams from the same conference."""
    import itertools

    rng = np.random.default_rng(0)
    for _ in range(60):
        picks = sorted(int(i) for i in rng.choice(np.arange(1, 49), 9, replace=False))
        seeded = _seeded({i: "X" for i in picks})
        meetings = {frozenset(p): 2 for p in itertools.combinations([f"T{i}" for i in picks], 2)}
        pl = regions.place(seeded, meetings)
        assert all("Opening Round" in p for p in regions.check(seeded, pl, meetings))
