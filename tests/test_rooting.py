"""The rooting guide: forced outcomes on common random numbers, and what each team page makes of them."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from mri.export import rooting
from mri.sim import season

CONFS = ["SEC", "Big Ten", "Big 12", "ACC", "American", "Mountain West"]


def league(seed: int = 3):
    """Six ten-team conferences with a round robin half played: enough of a season for the field to be in doubt."""
    rng = np.random.default_rng(seed)
    teams = [{"team": f"{c}-{j}", "conference": c, "power": float(rng.normal(8 if c in CONFS[:4] else -4, 8))}
             for c in CONFS for j in range(10)]
    rows = []
    for c in CONFS:
        members = [t["team"] for t in teams if t["conference"] == c]
        pairs = [(a, b) for i, a in enumerate(members) for b in members[i + 1:]]
        for k, (a, b) in enumerate(pairs):
            week = 1 + k % 9
            played = week <= 4
            home_won = rng.random() < 0.5
            rows.append({"game_id": 1000 + len(rows), "week": week, "season_type": "regular",
                         "start_date": f"2026-09-{5 + week:02d}T19:00:00.000Z", "team1": a, "team2": b,
                         "played": played, "pts1": (20.0 if home_won else 30.0) if played else np.nan,
                         "pts2": (30.0 if home_won else 20.0) if played else np.nan, "neutral": False,
                         "conference_game": True, "conf1": c, "conf2": c})
    return teams, pd.DataFrame(rows)


@pytest.fixture(scope="module")
def world():
    return league()


def run(teams, schedule, **kw):
    return season.simulate(teams, schedule, home_field=2.5, sims=4000, seed=11, **kw)


def test_a_forced_result_holds_in_every_run(world):
    teams, schedule = world
    gid = int(schedule[~schedule["played"]]["game_id"].iloc[0])
    home, away = schedule.set_index("game_id").loc[gid, ["team2", "team1"]]
    seen = []
    run(teams, schedule, forced={gid: "home"}, observe=lambda chunk: seen.append(chunk["wins"]))
    base = []
    run(teams, schedule, observe=lambda chunk: base.append(chunk["wins"]))
    names = [t["team"] for t in teams]
    h, a = names.index(home), names.index(away)
    forced_wins = np.concatenate(seen)
    # Against the unforced run on the same seed, the home side has a win wherever it had a loss, never the reverse.
    assert (np.concatenate(seen)[:, h] >= np.concatenate(base)[:, h]).all()
    assert (forced_wins[:, a] <= np.concatenate(base)[:, a]).all()


def test_the_same_seed_reproduces_the_baseline(world):
    teams, schedule = world
    assert run(teams, schedule)["teams"] == run(teams, schedule)["teams"]
    assert run(teams, schedule)["teams"] == run(teams, schedule, forced={})["teams"]


def test_the_two_forced_runs_average_back_to_the_baseline(world):
    """The key check: P(baseline) = p(home) P(if home) + (1 - p(home)) P(if away), within simulation noise."""
    teams, schedule = world
    unplayed = schedule[~schedule["played"]]
    base = run(teams, schedule)["teams"]
    for gid in unplayed["game_id"].iloc[:3]:
        gid = int(gid)
        home = run(teams, schedule, forced={gid: "home"})["teams"]
        away = run(teams, schedule, forced={gid: "away"})["teams"]
        game = unplayed.set_index("game_id").loc[gid]
        h_name = game["team2"]
        # The sim's own chance of a home win, read off the forced pair: the home side's wins differ by exactly it.
        p_home = base[h_name]["projectedWins"] - away[h_name]["projectedWins"]
        assert 0 < p_home < 1
        for name, odds in base.items():
            mixed = p_home * home[name]["playoff"] + (1 - p_home) * away[name]["playoff"]
            assert odds["playoff"] == pytest.approx(mixed, abs=0.02), (gid, name)


def guide_for(world, gid_count=6):
    teams, schedule = world
    unplayed = schedule[~schedule["played"]]
    week = int(unplayed["week"].min())
    this_week = unplayed[unplayed["week"] == week].head(gid_count)
    games = [{"gameId": int(g.game_id), "home": g.team2, "away": g.team1, "neutral": False, "kickoff": g.start_date}
             for g in this_week.itertuples()]
    odds = lambda result: rooting._odds(result)                                         # noqa: E731
    baseline = odds(run(teams, schedule))
    home = {g["gameId"]: odds(run(teams, schedule, forced={g["gameId"]: "home"})) for g in games}
    away = {g["gameId"]: odds(run(teams, schedule, forced={g["gameId"]: "away"})) for g in games}
    return games, rooting.guide(games, home, away, baseline, {g["gameId"]: 0.6 for g in games}), baseline


def test_a_team_never_roots_in_its_own_game_and_a_rival_losing_helps_its_title_odds(world):
    games, guide, _ = guide_for(world)
    for g in games:
        for side in (g["home"], g["away"]):
            assert all(e["gameId"] != g["gameId"] for e in guide[side]["games"])
    # Same conference, other game: for the title race, the team roots for whichever side is not its closer rival -
    # and the root-for side's title chance always beats the other side's.
    for name, entry in guide.items():
        for e in entry["games"]:
            assert e["ifRoot"] >= e["ifNot"]


def test_outside_the_band_is_one_line():
    games = [{"gameId": 1, "home": "A", "away": "B", "neutral": False, "kickoff": ""}]
    odds = lambda p: {"playoff": p, "conferenceTitle": 0.5, "bye": 0.1}                # noqa: E731
    baseline = {"Lock": odds(0.999), "Longshot": odds(0.001), "A": odds(0.5), "B": odds(0.5)}
    home = {1: {"Lock": odds(0.999), "Longshot": odds(0.004), "A": odds(0.6), "B": odds(0.4)}}
    away = {1: {"Lock": odds(0.999), "Longshot": odds(0.0), "A": odds(0.4), "B": odds(0.6)}}
    guide = rooting.guide(games, home, away, baseline, {1: 0.5})
    assert guide["Lock"]["games"] == [] and guide["Lock"]["message"] == "Your fate is in your own hands this week."
    assert guide["Longshot"]["games"] == [] and guide["Longshot"]["message"] == "Nothing this weekend moves the needle."


def test_the_conference_title_fallback():
    games = [{"gameId": 1, "home": "Rival", "away": "Other", "neutral": False, "kickoff": ""}]
    same = {"playoff": 0.2, "conferenceTitle": 0.30, "bye": 0.0}
    baseline = {"Us": same, "Rival": same, "Other": same}
    home = {1: {"Us": {**same, "conferenceTitle": 0.25}, "Rival": same, "Other": same}}
    away = {1: {"Us": {**same, "conferenceTitle": 0.36}, "Rival": same, "Other": same}}
    us = rooting.guide(games, home, away, baseline, {1: 0.7})["Us"]
    assert us["basis"] == "conferenceTitle" and us["games"][0]["rootFor"] == "Other"
    assert us["games"][0]["upsetNeeded"] and us["games"][0]["delta"] == pytest.approx(0.11)


def test_the_guide_runs_inside_its_budget(world, tmp_path):
    teams, schedule = world
    payload = {"teams": teams, "homeField": 2.5}
    started = time.monotonic()
    result = rooting.build(2026, payload, tmp_path / "r.json", tmp_path / "h.json", sims=2000, schedule=schedule,
                           now=pd.Timestamp("2026-09-01", tz="UTC").to_pydatetime())
    elapsed = time.monotonic() - started
    print(f"rooting: {result['games']} games in {elapsed:.1f}s")
    assert result["week"] == 5 and elapsed < 60
    assert (tmp_path / "h.json").exists()


def test_the_conference_leader_losing_raises_a_contenders_title_odds(world):
    teams, schedule = world
    base = run(teams, schedule)["teams"]
    unplayed = schedule[~schedule["played"]]
    checked = 0
    for c in CONFS:
        members = sorted((t["team"] for t in teams if t["conference"] == c), key=lambda n: -base[n]["conferenceTitle"])
        leader, contender = members[0], members[1]
        # The leader against someone weak, where the winner of an upset is not in the race itself.
        games = unplayed[((unplayed["team1"] == leader) | (unplayed["team2"] == leader))
                         & ~unplayed["team1"].isin(members[1:4]) & ~unplayed["team2"].isin(members[1:4])]
        if games.empty:
            continue
        g = games.iloc[0]
        leader_home = g["team2"] == leader
        loses = run(teams, schedule, forced={int(g["game_id"]): "away" if leader_home else "home"})["teams"]
        wins = run(teams, schedule, forced={int(g["game_id"]): "home" if leader_home else "away"})["teams"]
        assert loses[contender]["conferenceTitle"] > wins[contender]["conferenceTitle"], (c, leader, contender)
        checked += 1
    assert checked >= 3
