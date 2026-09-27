"""In-season hot-seat odds: run the season simulation, rebuild the fitted model's
feature row for every currently active coach *per run*, score each run, and
average - the same trick ``mri.heisman.forecast`` already uses for "what does
this player's Heisman case look like in each of ten thousand imagined
seasons," just with a binary logit instead of a softmax.

Everything that doesn't depend on how the rest of the season plays out - the
tenure-year dummies, ``vs_par``'s one-season lag, the power-conference flag,
``vs_talent`` - is already fixed and comes straight from ``coach_season.
parquet``. Three things do depend on the run, and only three: the season's
final Power (``power_end``, and everything derived from it - ``added``,
``vs_par``, ``vs_inherited``), the win percentage, and the conference win
percentage. ``mri.sim.season.simulate()``'s ``observe`` hook already exposes
per-run ``wins``/``losses``; ``est`` (final Power) and ``confWins`` are the two
fields added there for this feature specifically.

A coach missing ``program_par`` or ``inherited`` is excluded here exactly the
same way ``mri.coaches.hotseat.dataset`` excludes them from training - there
is no more evidence in-season than there was for that same coach-season once
it's finished, so the same "null, not a guess" rule applies.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..export import simdata
from ..sim import season
from . import hotseat

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[3] / "data" / "hotseat_model.json"


def simulate_teams(
    teams: list[dict], schedule: pd.DataFrame, *, home_field: float,
    sims: int = season.DEFAULT_SIMS, seed: int = season.DEFAULT_SEED, championships: dict | None = None,
) -> dict:
    """Play out the rest of the season and keep every run's wins, losses, final Power
    and conference wins - the same wrapper shape as ``mri.heisman.forecast.simulate_teams``."""
    chunks: list[dict] = []
    season.simulate(
        teams, schedule, home_field=home_field, sims=sims, seed=seed, championships=championships,
        observe=lambda c: chunks.append({k: v for k, v in c.items()}),
    )
    names = [t["team"] for t in teams]
    return {
        "names": names, "index": {n: i for i, n in enumerate(names)},
        "wins": np.concatenate([c["wins"] for c in chunks]).astype(float),
        "losses": np.concatenate([c["losses"] for c in chunks]).astype(float),
        "est": np.concatenate([c["est"] for c in chunks]).astype(float),
        "confWins": np.concatenate([c["confWins"] for c in chunks]).astype(float),
    }


def _conf_games_total(schedule: pd.DataFrame, team: str) -> int:
    """The full season's conference-game count for ``team`` - played and remaining,
    a fixed schedule fact, not something that varies run to run."""
    at_team = (schedule["team1"] == team) | (schedule["team2"] == team)
    return int((at_team & schedule["conference_game"]).sum())


def _active_coaches(coach_season: pd.DataFrame, current_season: int) -> pd.DataFrame:
    """Every non-interim coach with enough program history to score - the same
    exclusion `hotseat.dataset` applies to training rows."""
    active = coach_season[(~coach_season["interim"]) & (coach_season["season"] == current_season)]
    return active.dropna(subset=["program_par", "inherited", "prior"])


def run_features(coach_season: pd.DataFrame, current_season: int, runs: dict, schedule: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One feature frame per active coach, one row per simulated run.

    ``runs`` is ``simulate_teams()``'s output (or a synthetic stand-in shaped
    the same way, for tests - a single-row array reproduces a known outcome
    exactly, which is how this is checked against `hotseat.dataset`'s own
    numbers for a season that's already finished).
    """
    active = _active_coaches(coach_season, current_season)
    stint_start = coach_season.groupby(["coach_id", "school"])["season"].min()
    prior_vs_par = coach_season.set_index(["coach_id", "school", "season"])["vs_par"]

    out: dict[str, pd.DataFrame] = {}
    for row in active.itertuples():
        if row.school not in runs["index"]:
            continue
        idx = runs["index"][row.school]
        power_end = runs["est"][:, idx]
        wins, losses = runs["wins"][:, idx], runs["losses"][:, idx]
        conf_wins = runs["confWins"][:, idx]
        games = wins + losses
        conf_games_total = _conf_games_total(schedule, row.school)

        start = stint_start.get((row.coach_id, row.school), row.season)
        tenure_year = int(row.season) - int(start) + 1
        lag = prior_vs_par.get((row.coach_id, row.school, int(row.season) - 1), 0.0)
        lag = 0.0 if pd.isna(lag) else float(lag)

        win_pct = wins / np.maximum(games, 1.0)
        conf_win_pct = conf_wins / conf_games_total if conf_games_total > 0 else win_pct

        out[row.coach_id] = pd.DataFrame({
            "vs_par": power_end - row.program_par,
            "vs_par_lag": np.full_like(power_end, lag),
            "added": power_end - row.prior,
            "vs_inherited": power_end - row.inherited,
            "win_pct": win_pct,
            "conf_win_pct": conf_win_pct,
            "year1": np.full_like(power_end, float(tenure_year == 1)),
            "year2": np.full_like(power_end, float(tenure_year == 2)),
            "year3": np.full_like(power_end, float(tenure_year == 3)),
            "power_conference": np.full_like(power_end, float(row.conference in season.POWER_FOUR)),
            "vs_talent": np.full_like(power_end, 0.0 if pd.isna(row.vs_talent) else float(row.vs_talent)),
        })
    return out


def odds(
    coach_season: pd.DataFrame, current_season: int, teams: list[dict], schedule: pd.DataFrame, home_field: float, *,
    sims: int = season.DEFAULT_SIMS, seed: int = season.DEFAULT_SEED, championships: dict | None = None,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> pd.Series:
    """Every active coach's P(fired), averaged across simulated runs of the rest of the season."""
    model = json.loads(Path(model_path).read_text())
    columns, coefficients = model["features"], model["coefficientsFull"]
    beta = np.array([coefficients[c] for c in columns] + [coefficients["intercept"]])

    runs = simulate_teams(teams, schedule, home_field=home_field, sims=sims, seed=seed, championships=championships)
    features = run_features(coach_season, current_season, runs, schedule)

    results = {
        coach_id: float(hotseat.predict(beta, hotseat.design(frame, columns)).mean())
        for coach_id, frame in features.items()
    }
    return pd.Series(results, name="p_fired")


def _load_history(path: Path, season_: int) -> dict:
    if path.exists():
        history = json.loads(path.read_text())
        if history.get("season") == season_:
            return history
    return {"season": season_, "weeks": {}}


def _save_history(path: Path, history: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, sort_keys=True, separators=(",", ":")) + "\n")


def weekly_odds(
    coach_season: pd.DataFrame, current_season: int, current_week: int, teams: list[dict], schedule: pd.DataFrame,
    home_field: float, history_path: Path, **odds_kwargs,
) -> dict:
    """This week's odds, with every earlier finished week kept exactly as first computed -
    the same "only the current week is ever overwritten" rule as ``mri.export.simdata``'s
    ``sim_history.json``."""
    history = _load_history(history_path, current_season)
    current = odds(coach_season, current_season, teams, schedule, home_field, **odds_kwargs)
    history["weeks"][str(current_week)] = current.round(4).to_dict()
    _save_history(history_path, history)
    return history
