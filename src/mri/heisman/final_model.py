"""Who wins, given a finished regular season.

A conditional logit over the season's candidates, the same shape as the College
GameDay model: each player gets a score from his features, and his chance of
winning is his share of the field's total. It is the right model for a prize that
goes to exactly one person, because what matters is how a player compares with the
others that year, not how good he looks alone.

Trained on the winner alone, one observation a season. That is thin, so the model
is small and ridge-penalised, and it is graded by leaving whole seasons out.

Defenders are not candidates. Nobody who was mainly a defender has won since 1997,
and the two-way winner of 2024 played receiver enough to be a receiver here. The
handful who have finished in the top ten as defenders are the field's remainder,
and the page will say so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..gameday import model as choice
from . import data, features, funnel

MODEL_PATH = Path(__file__).resolve().parents[3] / "data" / "heisman_model.json"
SIZES = {k: v for k, v in funnel.GROUP_SIZES.items() if k != "DEF"}
FIRST_SEASON = 2012


@dataclass
class Season:
    year: int
    pool: pd.DataFrame          # one row per candidate
    X: np.ndarray               # (candidates, features)
    winner: int | None          # row of the winner in pool


def previous_finalists(voting: dict, year: int) -> set[str]:
    last = voting["seasons"].get(str(year - 1), {}).get("finalists", [])
    return {data.normalize(f["player"]) for f in last}


def pool(table: pd.DataFrame, power_rank: pd.Series, committee_rank: pd.Series, losses: pd.Series,
         prev: set[str], games: pd.Series | None = None) -> pd.DataFrame:
    """The candidates for a season, with everything the features need."""
    kept = funnel.candidates(table, power_rank, sizes=SIZES, games=games).copy()
    kept["team_rank"] = kept["team"].map(committee_rank).astype(float)
    kept["team_losses"] = kept["team"].map(losses).astype(float)
    kept["prev_finalist"] = kept["player"].map(lambda n: data.normalize(n) in prev).astype(float)
    kept["group_code"] = kept["group"].map(features.GROUPS)
    return kept.reset_index(drop=True)


def matrix(p: pd.DataFrame) -> np.ndarray:
    return features.matrix(p["off_score"].to_numpy(), p["group_code"].to_numpy(), p["team_rank"].to_numpy(),
                           p["team_losses"].to_numpy(), p["prev_finalist"].to_numpy())


def season(year: int, table: pd.DataFrame, state: pd.DataFrame, voting: dict) -> Season:
    """A finished season's pool and features, and who won it."""
    p = pool(table[table["season"] == year], state["power_rank"], state["rank"], state["losses"],
             previous_finalists(voting, year))
    winner = next(f for f in voting["seasons"][str(year)]["finalists"] if f["finish"] == 1)
    found = data.resolve(p, year, winner["player"], winner["school"])
    return Season(year, p, matrix(p), int(found.index[0]) if len(found) else None)


def fit(seasons: list[Season], columns: list[int] | None = None, penalty: float = 1.0) -> np.ndarray:
    sets = [(s.X if columns is None else s.X[:, columns], s.winner) for s in seasons if s.winner is not None]
    return choice.fit(sets, penalty)


def load_model(path: Path = MODEL_PATH) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None
