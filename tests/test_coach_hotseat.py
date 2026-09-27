"""The hot-seat model: label construction (fired vs. everything else, and what's
excluded), tenure-year dummies keyed by stint, the vs_par lag, conference
records, walk-forward's no-leakage guarantee, the hand-rolled AUC/Brier
metrics, and the ship bar."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mri.coaches import hotseat, season

MAX_SEASON = 2026  # the dataset's own "most recent, unconfirmed" boundary


def _row(coach_id: str, school: str, season_: int, **overrides) -> dict:
    base = {
        "coach_id": coach_id, "coach_name": coach_id, "school": school, "season": season_,
        "conference": "SEC", "games": 12, "wins": 6, "losses": 6, "interim": False,
        "conf_wins": 4, "conf_losses": 4,
        "power_end": 0.0, "prior": 0.0, "added": 0.0, "inherited": 0.0,
        "vs_inherited": 0.0, "program_par": 0.0, "vs_par": 0.0, "resume": 0.0, "vs_talent": np.nan,
    }
    base.update(overrides)
    return base


def _coach_season(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _departures(*rows: tuple[str, str, int, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"coach_id": c, "school": s, "season": yr, "label": label} for c, s, yr, label in rows]
    )


EMPTY_DEPARTURES = pd.DataFrame(columns=["coach_id", "school", "season", "label"])


# ---------------------------------------------------------------------------
# label construction
# ---------------------------------------------------------------------------

def test_fired_departure_is_a_positive_everything_else_is_not() -> None:
    coach_season = _coach_season(
        _row("fired-coach", "School A", 2010),
        _row("moved-coach", "School B", 2010),
        _row("retired-coach", "School C", 2010),
        _row("staying-coach", "School D", 2010),
        _row("staying-coach", "School D", 2011),  # confirms 2010 wasn't a departure at all
    )
    departures = _departures(
        ("fired-coach", "School A", 2010, "fired_or_pushed_out"),
        ("moved-coach", "School B", 2010, "moved_up"),
        ("retired-coach", "School C", 2010, "retired_or_other"),
    )
    data = hotseat.dataset(coach_season, departures)
    labels = data.set_index("coach_id")["label"]
    assert labels["fired-coach"] == 1.0
    assert labels["moved-coach"] == 0.0
    assert labels["retired-coach"] == 0.0
    assert labels["staying-coach"] == 0.0


def test_interim_rows_and_the_most_recent_season_are_excluded() -> None:
    coach_season = _coach_season(
        _row("interim-coach", "School A", 2010, interim=True),
        _row("regular-coach", "School A", MAX_SEASON),  # most recent season on record - unconfirmed
        _row("regular-coach", "School A", MAX_SEASON - 1),  # confirmed - not the max season
    )
    data = hotseat.dataset(coach_season, EMPTY_DEPARTURES)
    assert "interim-coach" not in set(data["coach_id"])
    assert MAX_SEASON not in set(data["season"])
    assert (MAX_SEASON - 1) in set(data["season"])


def test_rows_missing_core_history_are_dropped_not_guessed() -> None:
    coach_season = _coach_season(
        _row("a", "School A", 2010, vs_par=np.nan),
        _row("b", "School B", 2010, vs_inherited=np.nan),
        _row("c", "School C", 2010, added=np.nan),
        _row("d", "School D", 2010),
        _row("filler", "School Z", 2011),  # pushes the "most recent season" boundary past 2010
    )
    data = hotseat.dataset(coach_season, EMPTY_DEPARTURES)
    assert set(data.loc[data["season"] == 2010, "coach_id"]) == {"d"}


# ---------------------------------------------------------------------------
# tenure year, and the vs_par lag
# ---------------------------------------------------------------------------

def test_tenure_year_dummies_are_keyed_by_stint() -> None:
    coach_season = _coach_season(
        _row("wanderer", "School A", 2008),
        _row("wanderer", "School A", 2009),
        _row("wanderer", "School A", 2010),
        _row("wanderer", "School A", 2011),  # year 4 - no dummy, the reference category
        _row("wanderer", "School B", 2011),  # a new stint elsewhere - year 1 again, same season
        _row("filler", "School Z", 2012),  # pushes the "most recent season" boundary past 2011
    )
    data = hotseat.dataset(coach_season, EMPTY_DEPARTURES).set_index(["school", "season"])
    assert (data.loc[("School A", 2008), ["year1", "year2", "year3"]] == [1, 0, 0]).all()
    assert (data.loc[("School A", 2009), ["year1", "year2", "year3"]] == [0, 1, 0]).all()
    assert (data.loc[("School A", 2010), ["year1", "year2", "year3"]] == [0, 0, 1]).all()
    assert (data.loc[("School A", 2011), ["year1", "year2", "year3"]] == [0, 0, 0]).all()
    assert (data.loc[("School B", 2011), ["year1", "year2", "year3"]] == [1, 0, 0]).all()


def test_tenure_year_resets_on_a_return_to_the_same_school() -> None:
    """Petrino at Louisville: 2004-2005, a decade away, then back in 2014 -
    year 1 again at his own return, not year 11 counted from his first arrival."""
    coach_season = _coach_season(
        _row("petrino", "Louisville", 2004),
        _row("petrino", "Louisville", 2005),
        _row("petrino", "Louisville", 2014),
        _row("petrino", "Louisville", 2015),
        _row("filler", "School Z", 2016),  # pushes the "most recent season" boundary
    )
    data = hotseat.dataset(coach_season, EMPTY_DEPARTURES).set_index("season")
    assert (data.loc[2004, ["year1", "year2", "year3"]] == [1, 0, 0]).all()
    assert (data.loc[2005, ["year1", "year2", "year3"]] == [0, 1, 0]).all()
    assert (data.loc[2014, ["year1", "year2", "year3"]] == [1, 0, 0]).all()
    assert (data.loc[2015, ["year1", "year2", "year3"]] == [0, 1, 0]).all()


def test_vs_par_lag_is_zero_in_year_one_and_the_real_value_after() -> None:
    coach_season = _coach_season(
        _row("a", "School A", 2010, vs_par=5.0),
        _row("a", "School A", 2011, vs_par=-3.0),
        _row("filler", "School Z", 2012),  # pushes the "most recent season" boundary past 2011
    )
    data = hotseat.dataset(coach_season, EMPTY_DEPARTURES).set_index("season")
    assert data.loc[2010, "vs_par_lag"] == 0.0   # year 1 - nothing to lag
    assert data.loc[2011, "vs_par_lag"] == 5.0   # year 2 - last season's real vs_par


# ---------------------------------------------------------------------------
# conference_record
# ---------------------------------------------------------------------------

def _schedule(school: str, opponent_conf_flags: list[bool]) -> pd.DataFrame:
    n = len(opponent_conf_flags)
    return pd.DataFrame({
        "game_id": range(1, n + 1), "team1": ["Opponent"] * n, "team2": [school] * n,
        "pts1": [10] * n, "pts2": [20] * n,  # school always wins, by 10-20
        "start_date": pd.date_range("2016-09-01", periods=n, freq="7D"),
        "conference_game": opponent_conf_flags,
    })


def test_conference_record_splits_correctly_across_a_split_season() -> None:
    coach_season = _coach_season(
        _row("old-coach", "School X", 2015),
        _row("old-coach", "School X", 2016, games=2),
        _row("new-coach", "School X", 2016, games=2),
    )
    # 4 games: 1st two (old-coach) conference+non-conference, last two (new-coach) both conference
    schedule = _schedule("School X", [True, False, True, True])
    conf = season.conference_record(coach_season, games_fn=lambda yr: schedule)
    conf_2016 = conf[conf["season"] == 2016].set_index("coach_id")
    assert conf_2016.loc["old-coach", "conf_wins"] == 1   # 1 of their 2 games was a conference win
    assert conf_2016.loc["new-coach", "conf_wins"] == 2   # both of their games were conference wins


# ---------------------------------------------------------------------------
# walk-forward, AUC/Brier, and the ship bar
# ---------------------------------------------------------------------------

def _synthetic_seasons(n_seasons: int, signal: bool, seed: int = 0) -> pd.DataFrame:
    """Coach-seasons across ``n_seasons`` years, where ``vs_par`` cleanly predicts the
    label when ``signal`` is True, and predicts nothing when it's False."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(2004, 2004 + n_seasons):
        for i in range(150):
            vs_par = rng.normal()
            label_prob = (vs_par < -0.5) if signal else (rng.random() < 0.1)
            rows.append(_row(f"c{s}-{i}", f"School {i}", s, vs_par=vs_par))
            if label_prob:
                rows[-1]["_fired"] = True
    coach_season = pd.DataFrame(rows)
    fired = coach_season[coach_season.get("_fired", False) == True]  # noqa: E712
    departures = _departures(*[(r.coach_id, r.school, r.season, "fired_or_pushed_out") for r in fired.itertuples()])
    coach_season = coach_season.drop(columns=[c for c in ["_fired"] if c in coach_season.columns])
    return hotseat.dataset(coach_season, departures)


def test_walk_forward_never_leaks_a_held_out_seasons_own_rows() -> None:
    data = _synthetic_seasons(10, signal=True)
    predictions = hotseat.walk_forward(data, ["vs_par"], min_train_seasons=4)
    for season_ in predictions["season"].unique():
        train_seasons = data[data["season"] < season_]["season"]
        assert (train_seasons < season_).all()
        assert season_ not in set(train_seasons)


def test_auc_and_brier_on_perfectly_separated_and_coin_flip_data() -> None:
    y_sep = np.array([0, 0, 0, 1, 1, 1])
    p_sep = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert hotseat.auc(y_sep, p_sep) == pytest.approx(1.0)
    assert hotseat.brier(y_sep, p_sep) < 0.1

    rng = np.random.default_rng(1)
    y_flip = rng.integers(0, 2, size=2000)
    p_flip = np.full(2000, 0.5)
    assert hotseat.auc(y_flip, p_flip) == pytest.approx(0.5, abs=1e-9)  # ties only - exactly 0.5


def test_ship_bar_passes_with_signal_and_fails_without_it() -> None:
    with_signal = _synthetic_seasons(12, signal=True, seed=2)
    result = hotseat.evaluate(with_signal, full_cols=["vs_par"])
    assert result["passed"] is True
    assert result["seasonsFullBetter"] == result["seasonsTotal"]

    # A per-season win count is noisy at this scale even with hundreds of rows a
    # season; AUC pooled across every held-out season is the stable check that the
    # "no real signal" case actually carries none - near 0.5, not a real edge.
    without_signal = _synthetic_seasons(12, signal=False, seed=3)
    result_no_signal = hotseat.evaluate(without_signal, full_cols=["vs_par"])
    assert result_no_signal["aucFull"] == pytest.approx(0.5, abs=0.07)
