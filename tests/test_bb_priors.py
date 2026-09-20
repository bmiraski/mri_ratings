"""A basketball prior must move the way rosters do, and never break a build."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from mri.ratings import bb_priors, mri2

SEASON = 2025
D1 = ["Duke", "Kansas", "Kentucky", "Gonzaga"]
MODEL = {
    "roster": {"features": ["ret_ws", "in_ws", "frosh"],
               "coefficients": {"intercept": -5.0, "last_season": 0.8, "ret_ws": 8.0, "in_ws": 0.4, "frosh": 6.0}},
    "noRoster": {"features": ["vet_min", "draft_min", "frosh"],
                 "coefficients": {"intercept": 0.3, "last_season": 0.8, "vet_min": -3.0, "draft_min": -7.0, "frosh": 6.0}},
    "tighten": 0.7,
}


def raw_players() -> pd.DataFrame:
    """Last season (2024): Duke's minutes are split between a returner, a senior and a draftee."""
    rows = [
        # athlete, team, minutes, win shares, season
        (1, "Duke", 1000, 6.0, 2024), (2, "Duke", 800, 4.0, 2024), (3, "Duke", 600, 2.0, 2024),
        (4, "Kansas", 900, 5.0, 2024), (5, "Kansas", 900, 5.0, 2024),
        (6, "Kentucky", 700, 3.0, 2024), (7, "Gonzaga", 500, 2.0, 2024),
        # the transfer: played for Gonzaga in 2024, and will be on Duke's roster
        (8, "Gonzaga", 900, 7.0, 2024),
        # earlier seasons for player 2, so last year was his fourth
        (2, "Duke", 700, 3.0, 2023), (2, "Duke", 600, 2.0, 2022), (2, "Duke", 500, 1.0, 2021),
    ]
    return pd.DataFrame([{"season": s, "athlete_id": a, "team": t, "minutes": m, "win_shares": w, "name": f"P{a}"}
                         for a, t, m, w, s in rows])


def tables():
    recruits = pd.DataFrame([{"year": 2024, "name": "R", "stars": 5, "rating": 0.99, "team": "Kansas", "position": "G"},
                             {"year": 2024, "name": "S", "stars": 3, "rating": 0.75, "team": "Kansas", "position": "F"}])
    draft = pd.DataFrame([{"year": 2024, "athlete_id": 3, "overall": 12, "name": "P3", "college_id": 1}])
    return bb_priors._played(raw_players()), recruits, draft


def test_a_returning_share_counts_production_not_just_bodies() -> None:
    played, recruits, draft = tables()
    roster = {"Duke": {1, 2, 8, 20, 21, 22, 23, 24}}          # 1 and 2 stay, 3 leaves, 8 arrives from Gonzaga
    feats = bb_priors.features(SEASON, D1, played, recruits, draft, roster=roster)
    duke = feats.loc["Duke"]
    assert duke["has_roster"]
    assert duke["ret_ws"] == pytest.approx((6.0 + 4.0) / 12.0)       # 10 of Duke's 12 win shares stayed
    assert duke["in_ws"] == pytest.approx(7.0)                       # the newcomer's production elsewhere
    assert duke["vet_min"] == pytest.approx(800 / 2400)              # player 2 was in his fourth season
    assert duke["draft_min"] == pytest.approx(600 / 2400)            # player 3 was drafted
    assert feats.loc["Kansas", "frosh"] == pytest.approx(0.19)       # (0.99 - 0.8) counts, the 0.75 does not


def test_without_a_roster_the_roster_columns_are_empty_and_the_flag_is_off() -> None:
    played, recruits, draft = tables()
    feats = bb_priors.features(SEASON, D1, played, recruits, draft, roster=None)
    assert not feats["has_roster"].any() and feats["ret_ws"].isna().all()
    assert feats.loc["Duke", "vet_min"] == pytest.approx(800 / 2400)


def test_a_half_filled_roster_is_not_a_roster() -> None:
    played, recruits, draft = tables()
    feats = bb_priors.features(SEASON, D1, played, recruits, draft, roster={"Duke": {1, 2, 3}})
    assert not feats.loc["Duke", "has_roster"]


def test_each_team_gets_the_version_its_data_supports() -> None:
    row = pd.Series({"ret_ws": 0.8, "in_ws": 2.0, "frosh": 0.1, "vet_min": 0.2, "draft_min": 0.0, "has_roster": True})
    with_roster = bb_priors.predict(MODEL, 10.0, row)
    assert with_roster == pytest.approx(-5.0 + 0.8 * 10 + 8.0 * 0.8 + 0.4 * 2.0 + 6.0 * 0.1)
    no_roster = row.copy()
    no_roster["has_roster"] = False
    without = bb_priors.predict(MODEL, 10.0, no_roster)
    assert without == pytest.approx(0.3 + 0.8 * 10 + -3.0 * 0.2 + -7.0 * 0.0 + 6.0 * 0.1)
    assert bb_priors.predict(MODEL, float("nan"), row) is None
    missing = row.copy()
    missing["ret_ws"] = np.nan
    assert bb_priors.predict(MODEL, 10.0, missing) is None


def test_a_team_that_lost_its_roster_is_rated_below_one_that_kept_it() -> None:
    previous = pd.Series({t: 10.0 for t in D1})
    base = {"in_ws": 0.0, "frosh": 0.0, "vet_min": 0.0, "draft_min": 0.0, "has_roster": True}
    feats = pd.DataFrame({t: {**base, "ret_ws": r} for t, r in zip(D1, (0.1, 0.9, 0.5, 0.5))}).T
    out = bb_priors.preseason_prior(previous, D1, D1, feats, MODEL)
    assert out["Duke"] < out["Kansas"]


def test_the_level_is_kept_and_the_spread_is_pulled_in() -> None:
    previous = pd.Series({"Duke": 20.0, "Kansas": 10.0, "Kentucky": 0.0, "Gonzaga": -10.0})
    base = {"in_ws": 0.0, "frosh": 0.0, "vet_min": 0.0, "draft_min": 0.0, "has_roster": True, "ret_ws": 0.5}
    feats = pd.DataFrame({t: base for t in D1}).T
    old = mri2.build_prior(previous, D1, mri2.BASKETBALL_PROFILE.prior_regression, centre_teams=D1)
    new = bb_priors.preseason_prior(previous, D1, D1, feats, MODEL)
    assert new.mean() == pytest.approx(old.mean())                   # not a point off the old level
    assert new.std() == pytest.approx(0.7 * 0.8 * previous.std(), rel=1e-6)   # 0.8 on last season, then 70% of it


def test_teams_the_model_cannot_speak_for_keep_the_old_prior() -> None:
    previous = pd.Series({"Duke": 10.0, "Kansas": 5.0})
    feats = pd.DataFrame({"Duke": {"ret_ws": 0.5, "in_ws": 1.0, "frosh": 0.1, "vet_min": 0.1, "draft_min": 0.0, "has_roster": True}}).T
    old = mri2.build_prior(previous, ["Duke", "Kansas", "Newcomer"], mri2.BASKETBALL_PROFILE.prior_regression, centre_teams=D1)
    out = bb_priors.preseason_prior(previous, ["Duke", "Kansas", "Newcomer"], D1, feats, MODEL)
    assert out["Kansas"] == pytest.approx(old["Kansas"]) and out["Newcomer"] == pytest.approx(old["Newcomer"])


def test_no_model_or_no_last_season_is_the_old_rule() -> None:
    previous = pd.Series({t: 10.0 for t in D1})
    old = mri2.build_prior(previous, D1, mri2.BASKETBALL_PROFILE.prior_regression, centre_teams=D1)
    feats = pd.DataFrame(index=D1)
    assert bb_priors.preseason_prior(previous, D1, D1, feats, None).equals(old)
    assert bb_priors.preseason_prior(None, D1, D1, feats, MODEL).equals(
        mri2.build_prior(None, D1, mri2.BASKETBALL_PROFILE.prior_regression, centre_teams=D1))


# ---- where rosters come from

def test_a_played_season_takes_its_roster_from_the_stats_and_a_coming_one_from_the_feed(monkeypatch) -> None:
    played, recruits, draft = tables()
    monkeypatch.setattr(bb_priors, "_tables", lambda: (played, recruits, draft))
    assert bb_priors.rosters_for(2024, D1)["Duke"] == {1, 2, 3}

    calls = []

    def feed(season, refresh=False):
        assert refresh, "a coming season's roster must be re-fetched, not read from a stale cache"
        calls.append(season)
        return pd.DataFrame({"season": season, "team": ["Duke"] * 9, "conference": "ACC",
                             "athlete_id": list(range(30, 39)), "name": "x"})

    monkeypatch.setattr("mri.ingest.cbbd.rosters", feed)
    assert calls == [] and len(bb_priors.rosters_for(2027, D1)["Duke"]) == 9 and calls == [2027]


def test_rosters_not_posted_yet_mean_no_rosters_and_never_an_error(monkeypatch) -> None:
    played, recruits, draft = tables()
    monkeypatch.setattr(bb_priors, "_tables", lambda: (played, recruits, draft))
    monkeypatch.setattr("mri.ingest.cbbd.rosters", lambda season, refresh=False: pd.DataFrame(columns=["season", "team", "conference", "athlete_id", "name"]))
    assert bb_priors.rosters_for(2027, D1) is None

    def boom(season, refresh=False):
        raise RuntimeError("offline")

    monkeypatch.setattr("mri.ingest.cbbd.rosters", boom)
    assert bb_priors.rosters_for(2027, D1) is None


def test_status_says_which_version_is_in_use(monkeypatch) -> None:
    monkeypatch.setattr(bb_priors, "rosters_for", lambda season, d1: None)
    assert bb_priors.status(2027, D1)["mode"] == "before rosters"
    full = {t: set(range(10)) for t in D1}
    monkeypatch.setattr(bb_priors, "rosters_for", lambda season, d1: full)
    assert bb_priors.status(2027, D1) == {"season": 2027, "teamsWithRosters": 4, "teams": 4, "mode": "roster"}


def test_a_data_gap_costs_accuracy_and_never_the_build(monkeypatch) -> None:
    previous = pd.Series({t: 10.0 for t in D1})
    old = mri2.build_prior(previous, D1, mri2.BASKETBALL_PROFILE.prior_regression, centre_teams=D1)
    monkeypatch.setattr(bb_priors, "load_model", lambda *a, **k: None)
    assert bb_priors.for_season(2027, previous, D1, D1).equals(old)
    monkeypatch.setattr(bb_priors, "load_model", lambda *a, **k: MODEL)
    monkeypatch.setattr(bb_priors, "_tables", lambda: None)
    assert bb_priors.for_season(2027, previous, D1, D1).equals(old)


# ---- the committed files

def test_the_fitted_model_has_the_signs_the_story_needs() -> None:
    model = bb_priors.load_model()
    assert model is not None, "data/bb_prior_model.json is missing"
    c = model["roster"]["coefficients"]
    assert 0.3 < c["last_season"] < 1.1 and c["ret_ws"] > 0 and c["frosh"] > 0
    assert 0.3 < model["tighten"] < 1.0
    n = model["noRoster"]["coefficients"]
    assert n["draft_min"] < 0                                   # minutes headed for the draft are minutes lost
    r = model["outOfSampleRmse"]
    assert r["with a roster"]["all"] < r["before rosters"]["all"] < r["last season only"]["all"]


def test_the_backtest_file_says_the_new_prior_wins_where_it_claims_to() -> None:
    root = bb_priors.ROOT / "site" / "data" / "bb_prior_backtest.json"
    if not root.exists():
        pytest.skip("backtest not run")
    bt = json.loads(root.read_text())
    for label, e in bt["bins"].items():
        assert e["with a roster"]["mae"] <= e["old"]["mae"], label
        assert e["with a roster"]["gain"] > 0 and e["with a roster"]["gainError"] >= 0


# ---- the team page's roster lines

def test_roster_context_says_what_is_known_in_each_mode(monkeypatch) -> None:
    from mri.export import bb_sitedata

    played, recruits, draft = tables()
    monkeypatch.setattr(bb_priors, "_tables", lambda: (played, recruits, draft))

    roster = {"Duke": {1, 2, 8, 20, 21, 22, 23, 24}, "Kansas": {4, 5, 40, 41, 42, 43, 44, 45}}
    monkeypatch.setattr(bb_priors, "rosters_for", lambda season, d1: roster)
    ctx = bb_sitedata.roster_context(SEASON, D1)
    assert ctx["Duke"]["mode"] == "roster" and ctx["Duke"]["returning"] == pytest.approx(10 / 12, abs=1e-3)
    assert ctx["Duke"]["incoming"] == 7.0 and ctx["Duke"]["returningOf"] == 2
    assert ctx["Kansas"]["returning"] == 1.0 and ctx["Kansas"]["returningRank"] == 1
    assert ctx["Kansas"]["freshman"] == pytest.approx(0.19) and ctx["Kansas"]["freshmanRank"] == 1
    assert ctx["Kentucky"]["mode"] == "before rosters" and "returning" not in ctx["Kentucky"]
    assert ctx["Duke"]["veteranMinutes"] == pytest.approx(800 / 2400, abs=1e-3)

    monkeypatch.setattr(bb_priors, "rosters_for", lambda season, d1: None)
    none_posted = bb_sitedata.roster_context(SEASON, D1)
    assert all(e["mode"] == "before rosters" and "returning" not in e for e in none_posted.values())


def test_roster_context_is_empty_rather_than_fatal_without_data(monkeypatch) -> None:
    from mri.export import bb_sitedata

    monkeypatch.setattr(bb_priors, "_tables", lambda: None)
    assert bb_sitedata.roster_context(SEASON, D1) == {}
