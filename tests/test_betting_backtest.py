"""The betting backtest may only train on games played before the one it prices."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from mri.betting import backtest
from mri.ingest import cfbd


def _season() -> pd.DataFrame:
    """Two regular weeks and a bowl the feed labels postseason week 1."""
    rows = [
        (1, 1, "regular", "A", "B"),
        (2, 1, "regular", "C", "D"),
        (3, 2, "regular", "A", "C"),
        (4, 2, "regular", "B", "D"),
        (5, 1, "postseason", "A", "D"),
    ]
    return pd.DataFrame(
        [
            {"game_id": g, "week": w, "season_type": t, "team1": a, "team2": h,
             "pts1": 10.0, "pts2": 20.0, "neutral": t == "postseason"}
            for g, w, t, a, h in rows
        ]
    )


def test_postseason_sorts_after_the_last_regular_week() -> None:
    assert cfbd.sequence(_season()).tolist() == [1, 1, 2, 2, 3]


def test_week_two_training_set_contains_no_postseason_games(monkeypatch) -> None:
    trained_on = []

    def fake_fit(train, **_):
        trained_on.append(train.copy())
        return SimpleNamespace(power=pd.Series(0.0, index=list("ABCD")), home_field=2.0)

    monkeypatch.setattr(backtest.cfbd, "games", lambda year: _season())
    monkeypatch.setattr(backtest.mri2, "fit", fake_fit)
    monkeypatch.setattr(backtest.mri2, "build_prior", lambda *a, **k: None)
    monkeypatch.setattr(backtest, "MIN_TRAIN_GAMES", 1)

    lines = pd.DataFrame({"game_id": [3, 4, 5], "market": [3.0, 3.0, 3.0], "market_open": [3.0] * 3})
    priced, _ = backtest.price_season(2019, lines, None)

    week_two = trained_on[0]
    assert set(week_two["game_id"]) == {1, 2}
    assert (week_two["season_type"] == "regular").all()
    # The bowl is priced, last, from the whole regular season.
    assert set(trained_on[-1]["game_id"]) == {1, 2, 3, 4}
    assert priced["game_id"].tolist() == [3, 4, 5]
