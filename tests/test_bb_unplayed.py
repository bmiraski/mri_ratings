"""Early in a season a fit knows only the teams that have played; the rest must be rated at their prior."""
import pandas as pd

from mri.export import bb_sitedata
from mri.ratings import mri2


def _model() -> mri2.Ratings:
    index = pd.Index(["A", "B"])
    return mri2.Ratings(
        power=pd.Series([10.0, -2.0], index=index), home_field=3.5, sigma=11.0,
        games_played=pd.Series([4, 4], index=index), resume=pd.Series([1.0, -1.0], index=index),
    )


def test_unplayed_teams_are_rated_at_their_prior() -> None:
    prior = pd.Series({"A": 9.0, "B": -1.0, "C": 4.5, "D": -7.0})
    rated = bb_sitedata.with_unplayed(_model(), prior, ["A", "B", "C", "D"])
    assert rated.power["C"] == 4.5 and rated.power["D"] == -7.0
    assert rated.power["A"] == 10.0                      # a team with games keeps its fitted rating
    assert rated.games_played["C"] == 0
    table = rated.table()
    assert set(table["team"]) == {"A", "B", "C", "D"}
    assert rated.predict("C", "D", neutral=True) == 11.5   # not priced at a replacement-level floor


def test_nothing_missing_returns_the_model_unchanged() -> None:
    model = _model()
    assert bb_sitedata.with_unplayed(model, pd.Series({"A": 1.0, "B": 2.0}), ["A", "B"]) is model
