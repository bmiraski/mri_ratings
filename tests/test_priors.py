"""A preseason prior must move the way rosters do, and never break a build."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from mri.export import sitedata
from mri.ratings import mri2, priors

MODEL = {"coefficients": {"intercept": -4.0, "last_season": 0.5, "talent": 4.0, "returning": 8.0}}
FBS = [f"Team {i}" for i in range(10)] + ["Army"]
PREVIOUS = pd.Series({t: 10.0 for t in FBS} | {"Some FCS": -25.0})


def talent(values: dict | None = None) -> pd.Series:
    base = {t: 700.0 + 20 * i for i, t in enumerate(FBS)}
    base.update(values or {})
    return pd.Series(base, dtype=float)


def returning(values: dict | None = None) -> pd.DataFrame:
    base = {t: 0.6 for t in FBS}
    base.update(values or {})
    return pd.DataFrame({"percent_ppa": pd.Series(base), "usage": pd.Series(base)})


def prior(**overrides) -> pd.Series:
    args = dict(previous=PREVIOUS, teams=FBS + ["Some FCS"], fbs=FBS, continuing=set(FBS),
                talent=talent(), returning=returning(), model=MODEL)
    args.update(overrides)
    previous, teams = args.pop("previous"), args.pop("teams")
    return priors.preseason_prior(previous, teams, **args)


def test_the_formula_is_the_regression() -> None:
    got = priors.predict(MODEL, [10.0], [1.0], [0.5])
    assert got[0] == pytest.approx(-4.0 + 0.5 * 10 + 4.0 * 1 + 8.0 * 0.5)


def test_a_team_that_lost_its_roster_is_rated_below_one_that_kept_it() -> None:
    out = prior(talent=talent({"Team 3": 800.0, "Team 4": 800.0}),
                returning=returning({"Team 3": 0.02, "Team 4": 0.95}))
    assert out["Team 3"] < out["Team 4"]
    # 0.93 of returning production at 8 points per whole share
    assert out["Team 4"] - out["Team 3"] == pytest.approx(8.0 * 0.93)


def test_more_talent_means_a_higher_prior() -> None:
    out = prior(talent=talent({"Team 3": 600.0, "Team 4": 1000.0}))
    assert out["Team 4"] > out["Team 3"]


def test_the_old_prior_is_the_fallback_for_everyone_the_model_does_not_cover() -> None:
    old = mri2.build_prior(PREVIOUS, FBS + ["Some FCS"], centre_teams=FBS)
    out = prior(continuing=set(FBS) - {"Team 2"})
    assert out["Team 2"] == pytest.approx(old["Team 2"])         # not FBS last year
    assert out["Some FCS"] == pytest.approx(old["Some FCS"])     # an FCS opponent
    assert out["Team 1"] != pytest.approx(old["Team 1"])


def test_without_a_model_or_a_last_season_nothing_changes() -> None:
    old = mri2.build_prior(PREVIOUS, FBS + ["Some FCS"], centre_teams=FBS)
    assert prior(model=None).equals(old)
    assert prior(previous=None).equals(mri2.build_prior(None, FBS + ["Some FCS"]))


def test_the_service_academies_talent_is_not_counted() -> None:
    cheap = prior(talent=talent({"Army": 300.0}))["Army"]
    rich = prior(talent=talent({"Army": 1300.0}))["Army"]
    assert cheap == pytest.approx(rich)


def test_talent_is_a_z_score_within_fbs_and_zero_when_unknown() -> None:
    z = priors.talent_scores(talent({"Army": 100.0}), FBS)
    others = z.drop("Army")
    assert z["Army"] == 0.0
    assert others.mean() == pytest.approx(0.0, abs=0.5)
    assert priors.talent_scores(None, FBS).eq(0.0).all()
    missing = talent().drop("Team 5")
    assert priors.talent_scores(missing, FBS)["Team 5"] == 0.0


def test_missing_returning_production_is_the_median_not_a_zero() -> None:
    r = returning({"Team 1": 0.2, "Team 2": 0.8}).drop("Team 3")
    share = priors.returning_share(r, FBS)
    assert share["Team 3"] == pytest.approx(r["percent_ppa"].median())
    assert priors.returning_share(None, FBS).eq(0.5).all()


def test_a_data_gap_costs_accuracy_and_never_the_build(monkeypatch) -> None:
    def boom(year):
        raise RuntimeError("no cache and no network")

    priors._inputs.cache_clear()
    monkeypatch.setattr(priors, "_inputs", boom)
    monkeypatch.setattr(priors, "load_model", lambda *a, **k: MODEL)
    out = priors.for_season(2026, PREVIOUS, FBS + ["Some FCS"], FBS)
    assert out.equals(mri2.build_prior(PREVIOUS, FBS + ["Some FCS"], centre_teams=FBS))


def test_the_fitted_model_has_the_signs_the_story_needs() -> None:
    model = priors.load_model()
    assert model is not None, "data/prior_model.json is missing"
    c = model["coefficients"]
    assert 0.2 < c["last_season"] < 0.9            # last season matters, and is not everything
    assert c["talent"] > 0 and c["returning"] > 0
    assert model["outOfSampleRmse"]["model"] < model["outOfSampleRmse"]["old"]


# ---- the team page's roster context

def test_roster_context_reports_what_talent_is_worth(monkeypatch) -> None:
    monkeypatch.setattr(sitedata.cfbd, "talent", lambda year: talent({"Team 9": 1500.0}))
    monkeypatch.setattr(sitedata.cfbd, "returning", lambda year: returning({"Team 2": 0.03}))
    monkeypatch.setattr(sitedata.priors, "load_model",
                        lambda *a, **k: {"talentFit": {"intercept": -1.0, "slope": 7.0}})
    teams = [{"team": t, "power": 5.0} for t in FBS]
    ctx = sitedata.roster_context(2026, teams)

    assert ctx["Team 9"]["talentRank"] == 1 and ctx["Team 9"]["talentOf"] == len(FBS) - 1
    assert ctx["Team 9"]["talentGap"] == pytest.approx(5.0 - ctx["Team 9"]["talentImplied"], abs=0.05)
    assert ctx["Team 2"]["returning"] == 0.03 and ctx["Team 2"]["returningRank"] == len(FBS)
    assert ctx["Army"]["talent"] is None and ctx["Army"]["talentNote"] == "unmeasured"


def test_roster_context_is_empty_rather_than_fatal_without_data(monkeypatch) -> None:
    def boom(year):
        raise RuntimeError("offline")

    monkeypatch.setattr(sitedata.cfbd, "talent", boom)
    assert sitedata.roster_context(2026, [{"team": "Team 0", "power": 0.0}]) == {}


def test_the_committed_model_file_is_valid_json() -> None:
    assert json.loads(priors.MODEL_PATH.read_text())["teamSeasons"] > 1000
