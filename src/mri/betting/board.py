"""This week's card: where the model disagrees with the market.

What the backtest found, and what this board is therefore for:

*Against closing lines the model loses.* 50.5% across 8,248 walk-forward bets,
against a 52.38% break-even. That is not a near miss, it is a verdict, and it
is what should be expected - a closing spread is the sharpest number in sports
and a rating built from scores is not going to out-argue it.

*Against opening lines there is a hint.* 53.2% across 1,936 bets, which sounds
better but carries p = 0.26 and does not survive correction for the number of
buckets examined.

*The one clean signal is closing line value.* Mean CLV rises monotonically with
the size of the disagreement - -0.02, +0.18, +0.46, +0.98, +2.51 points across
five ordered bins - and is positive in all three seasons. A dose-response
relationship across ordered bins is much harder to produce by chance than one
good bucket, and it says the market tends to move toward this model's side
after the opener.

That is a reason to keep measuring, not a reason to bet. So this board publishes
the disagreements and grades itself against the closing number, because CLV
converges in a few hundred bets where win-loss needs thousands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..ingest import cfbd, registry
from ..ratings import mri2
from . import lines as lines_module

MIN_EDGE_TO_SHOW = 3.0
SIGMA = 16.5


def build_board(year: int, ratings: mri2.Ratings | None = None, *, week: int | None = None) -> dict:
    """The upcoming week's games, model line beside market line."""
    schedule = _canonical(cfbd.games(year, completed_only=False))
    if schedule.empty:
        return {"week": None, "games": []}

    played = schedule[schedule["played"]]
    upcoming = schedule[~schedule["played"]]
    if upcoming.empty:
        return {"week": None, "games": []}

    target = week or int(upcoming["week"].min())
    slate = upcoming[upcoming["week"] == target]

    if ratings is None:
        ratings = _rate(year, played)

    book = lines_module.preferred_lines(year)
    book = book.set_index("game_id") if not book.empty else pd.DataFrame()

    replacement = float(ratings.power.min()) - 5.0
    from scipy.stats import norm

    # How much the model actually knows about each team. A program new to FBS
    # has no prior and a handful of games, so its rating is barely more than a
    # guess - and the biggest "edges" on an early-season card are exactly those
    # teams. Betting them is backing the model's ignorance, not its opinion.
    prior_power = _previous_season(year)
    known = set(prior_power.index) if prior_power is not None else set()
    appearances = pd.concat([played["team1"], played["team2"]]).value_counts()

    rows = []
    for row in slate.itertuples():
        if not (registry.is_fbs(row.team1) and registry.is_fbs(row.team2)):
            continue
        home = float(ratings.power.get(row.team2, replacement))
        away = float(ratings.power.get(row.team1, replacement))
        predicted = home - away + (0.0 if row.neutral else ratings.home_field)

        market = market_open = None
        source = None
        if not book.empty and row.game_id in book.index:
            line = book.loc[row.game_id]
            if isinstance(line, pd.DataFrame):
                line = line.iloc[0]
            market = float(line["market"]) if pd.notna(line["market"]) else None
            market_open = float(line["market_open"]) if pd.notna(line.get("market_open")) else None
            source = line.get("source")

        reference = market_open if market_open is not None else market
        edge = (predicted - reference) if reference is not None else None

        thin = [
            t for t in (row.team1, row.team2)
            if t not in known and int(appearances.get(t, 0)) < 4
        ]

        rows.append(
            {
                "week": int(row.week),
                "home": row.team2,
                "away": row.team1,
                "neutral": bool(row.neutral),
                "predicted": round(predicted, 1),
                "market": round(market, 1) if market is not None else None,
                "marketOpen": round(market_open, 1) if market_open is not None else None,
                "edge": round(edge, 1) if edge is not None else None,
                "side": (row.team2 if edge and edge > 0 else row.team1) if edge else None,
                "winProbability": round(float(norm.cdf(predicted / SIGMA)), 3),
                "source": source,
                "confident": not thin,
                "unknownTeams": thin,
            }
        )

    rows.sort(key=lambda r: -abs(r["edge"] or 0))
    flagged = [
        r for r in rows
        if r["edge"] is not None and abs(r["edge"]) >= MIN_EDGE_TO_SHOW and r["confident"]
    ]
    return {"week": target, "games": rows, "flagged": flagged}


def _rate(year: int, played: pd.DataFrame) -> mri2.Ratings:
    """Fit with the preseason prior, exactly as the rankings do.

    Fitting without it was the board's original bug. Two weeks into a season
    there is not enough evidence to separate 138 teams, so every rating
    collapses toward zero and the model cheerfully claims a 30-point
    disagreement with a market that is not wrong. The prior is what makes an
    early-season number mean anything, and the board has to use the same one
    the published ratings do or it is pricing a different model.
    """
    teams = sorted(set(played["team1"]) | set(played["team2"]))
    fbs = [t for t in teams if registry.is_fbs(t)]
    return mri2.fit(
        played,
        prior=mri2.build_prior(_previous_season(year), teams, centre_teams=fbs),
        neutral=played["neutral"],
        anchor_teams=fbs,
        with_resume=False,
        with_efficiency=False,
    )


def _previous_season(year: int) -> pd.Series | None:
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "data" / "parquet" / "current_ratings.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    previous = frame[frame["season"] == year - 1]
    return previous.set_index("team")["power"] if not previous.empty else None


def _canonical(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("team1", "team2"):
        frame[column] = [registry.resolve(n, n) for n in frame[column]]
    return frame
