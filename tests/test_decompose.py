"""The rating decomposition must tie out exactly, and must refuse what it cannot explain."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from mri.ratings import decompose, mri2

# The shared synthetic season lives beside the tests but is not a package, and
# test_site's import audit treats any bare `import x` as a third-party dependency.
_spec = importlib.util.spec_from_file_location("synthetic", Path(__file__).with_name("synthetic.py"))
synthetic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(synthetic)
FBS, IDLE, OUTSIDERS, synthetic_games = (
    synthetic.FBS, synthetic.IDLE, synthetic.OUTSIDERS, synthetic.synthetic_games
)

GOLDEN = Path(__file__).parent / "golden" / "synthetic_fit.json"
ARCHIVE = Path(__file__).resolve().parents[1] / "data" / "parquet" / "archive_games.parquet"


@pytest.fixture(scope="module")
def season():
    games, prior = synthetic_games()
    model = mri2.fit(games, prior=prior, neutral=games["neutral"], anchor_teams=FBS + [IDLE])
    return games, prior, model


def test_default_fit_outputs_are_unchanged(season) -> None:
    """Exposing the internals is additive: every pre-existing output matches the golden file.

    Compared to 1e-9, not bitwise: the golden file was written on one machine and CI's
    LAPACK differs in the last digit. (Before/after on the same machine, the real fits
    were bit-identical.)
    """
    _, _, model = season
    golden = json.loads(GOLDEN.read_text())
    for name, got in (("power", model.power), ("games_played", model.games_played), ("resume", model.resume)):
        assert got.to_dict() == pytest.approx(golden[name], rel=1e-9, abs=1e-9), name
    assert model.home_field == pytest.approx(golden["home_field"], rel=1e-9)
    assert model.sigma == pytest.approx(golden["sigma"], rel=1e-9)


def test_identity_reconstructs_every_team(season) -> None:
    games, _, model = season
    shares = decompose.decompose(model, games)
    assert set(shares) == set(model.power.index)
    for team, share in shares.items():
        rebuilt = share.margin_term + share.opponent_term + share.prior_term
        assert abs(rebuilt - model.power[team]) < 1e-9
        assert abs(sum(g.contribution for g in share.games) + share.prior_term - model.power[team]) < 1e-9
        assert share.n == model.games_played[team] == len(share.games)


def test_synthetic_season_covers_the_awkward_cases(season) -> None:
    games, _, model = season
    assert games["neutral"].any(), "needs neutral-site games"
    shares = decompose.decompose(model, games)
    assert all(o in shares for o in OUTSIDERS + [mri2.POOLED_FCS])
    # An FBS team that hosted an outsider sees the outsider's solved rating, not a stand-in.
    host_games = [g for s in shares.values() for g in s.games if g.opponent in OUTSIDERS]
    assert host_games and all(g.opponent_rating == model.power[g.opponent] for g in host_games)


def test_solver_home_field_is_not_the_published_one(season) -> None:
    """The trap: the identity needs the solver's coefficient. Using the published one fails."""
    games, _, model = season
    it = model.internals
    assert it.raw_home_field != model.home_field
    wrong = dataclasses.replace(it, raw_home_field=model.home_field)
    with pytest.raises(decompose.DecompositionUnavailable, match="misses"):
        decompose.decompose(dataclasses.replace(model, internals=wrong), games)


def test_zero_games_means_the_prior_and_does_not_divide_by_zero() -> None:
    share = decompose._share("Idle", 0, 4.0, [], [], 7.5, [], 7.5)
    assert share.w == 0.0 and share.avg_margin_counted == 0.0 and share.avg_opponent == 0.0
    assert share.margin_term + share.opponent_term + share.prior_term == pytest.approx(7.5)
    assert decompose.weight(0, 0.0) == 0.0


def test_few_games_are_mostly_preseason() -> None:
    assert decompose.weight(2, 4.0) == pytest.approx(1 / 3)
    assert decompose.weight(3, 4.0) == pytest.approx(3 / 7)


@pytest.mark.parametrize("flag", ["recency_weighted", "pace_adjusted"])
def test_refuses_fits_the_identity_does_not_describe(season, flag) -> None:
    games, _, model = season
    changed = dataclasses.replace(model.internals, **{flag: True})
    with pytest.raises(decompose.DecompositionUnavailable, match="unweighted"):
        decompose.decompose(dataclasses.replace(model, internals=changed), games)


def test_refuses_a_fit_without_internals(season) -> None:
    games, _, model = season
    with pytest.raises(decompose.DecompositionUnavailable):
        decompose.decompose(dataclasses.replace(model, internals=None), games)


def test_refuses_games_that_are_not_the_fits(season) -> None:
    games, _, model = season
    with pytest.raises(decompose.DecompositionUnavailable):
        decompose.decompose(model, games.iloc[:-1])
    shuffled = games.sample(frac=1.0, random_state=1)
    with pytest.raises(decompose.DecompositionUnavailable):
        decompose.decompose(model, shuffled)


@pytest.mark.skipif(not ARCHIVE.exists(), reason="archive not built")
def test_identity_on_a_real_size_season() -> None:
    """A full 2019 season: hundreds of teams, real neutral-site bowls, pooled FCS."""
    archive = pd.read_parquet(ARCHIVE)
    games = archive[archive["season"] == 2019].reset_index(drop=True)
    model = mri2.fit(games, with_efficiency=False)
    shares = decompose.decompose(model, games)
    err = max(abs(s.margin_term + s.opponent_term + s.prior_term - model.power[t]) for t, s in shares.items())
    assert err < 1e-9
    assert len(shares) > 100


def test_a_real_recency_weighted_fit_is_refused(season) -> None:
    """Recency weighting is on main now: the fit itself must flag it, not just a hand-set flag."""
    games, prior, _ = season
    weighted = mri2.fit(games, prior=prior, neutral=games["neutral"], anchor_teams=FBS + [IDLE],
                        recency_half_life=0.5)
    assert weighted.internals.recency_weighted
    with pytest.raises(decompose.DecompositionUnavailable, match="unweighted"):
        decompose.decompose(weighted, games)
