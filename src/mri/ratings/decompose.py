"""Where a MRI 2.0 rating comes from, in the form the SRS article used.

``mri2.fit`` solves a ridge problem. For a team with ``n`` games the normal
equation is

    (n + lam) * r = sum_g [ m_g - s_g * h_g + r_opp(g) ] + lam * p

with ``r`` the solver's rating, ``m_g`` the compressed margin from the team's
side, ``s_g`` +1 at home and -1 away, ``h_g`` the solver's own home-field
coefficient (0 at a neutral site) and ``p`` the preseason prior. ``fit`` then
multiplies by ``scale`` and subtracts the anchor shift ``c``; both are linear, so
the published rating ``R`` satisfies the same equation in published units:

    R = [ sum_g ( scale * (m_g - s_g * h_g) + R_opp(g) ) + lam * P' ] / (n + lam)

    P' = scale * p - c

Read it as: **Power = w * (average margin counted + average opponent rating) +
(1 - w) * prior, with w = n / (n + lam)**. Each game contributes
``(scale * (m - s*h) + R_opp) / (n + lam)``, and those contributions plus the
prior's share sum to the rating exactly.

Two things to keep straight:

* ``h`` here is ``FitInternals.raw_home_field``, the coefficient the solver used.
  It is not ``Ratings.home_field``, which ``fit`` re-estimates from residuals
  after the solve and is what the site publishes.
* The opponent term is the opponent's *published* rating, including outsiders the
  site does not list (FCS in football, non-D1 in basketball). The solver rates
  those individually, so the identity needs their solved value, not a stand-in.

This is an explanation of the model's reasoning, not independent evidence that
the rating is right: the opponent ratings come out of the same solve.

It holds only for the plain solve. A recency-weighted or tempo-adjusted fit
changes the normal equations, so those are refused rather than explained wrongly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .mri2 import Ratings

# The identity is exact up to floating point; anything this large is a different
# fit from the one that produced the ratings, not rounding.
TOLERANCE = 1e-6


class DecompositionUnavailable(ValueError):
    """The fit cannot be explained by the identity (or was not given what it needs)."""


@dataclass(frozen=True)
class GameShare:
    game_id: object
    opponent: str
    margin_raw: float          # points, team's side
    margin_counted: float      # scale * (compressed margin - site * raw home field)
    opponent_rating: float     # opponent's published rating
    contribution: float        # (margin_counted + opponent_rating) / (n + lam)


@dataclass(frozen=True)
class TeamShare:
    team: str
    n: int
    ridge: float
    w: float                   # n / (n + lam): how much the games count
    avg_margin_counted: float
    avg_opponent: float
    prior: float               # P', in published units
    margin_term: float         # w * avg_margin_counted
    opponent_term: float       # w * avg_opponent
    prior_term: float          # (1 - w) * prior
    power: float
    games: list[GameShare] = field(default_factory=list)


def weight(n: int, ridge: float) -> float:
    """Share of the rating set by games rather than the prior. Zero games: zero."""
    return n / (n + ridge) if n + ridge > 0 else 0.0


def _share(team, n, ridge, margins, opponents, prior_published, games, power) -> TeamShare:
    w = weight(n, ridge)
    avg_margin = float(np.mean(margins)) if n else 0.0
    avg_opp = float(np.mean(opponents)) if n else 0.0
    return TeamShare(
        team=team,
        n=n,
        ridge=ridge,
        w=w,
        avg_margin_counted=avg_margin,
        avg_opponent=avg_opp,
        prior=prior_published,
        margin_term=w * avg_margin,
        opponent_term=w * avg_opp,
        prior_term=(1.0 - w) * prior_published,
        power=power,
        games=games,
    )


def decompose(ratings: Ratings, games: pd.DataFrame) -> dict[str, TeamShare]:
    """Split every rated team's power into margin, opponents and prior.

    ``games`` must be the table the fit was run on, in the same order.
    Raises ``DecompositionUnavailable`` if the fit cannot be explained this way,
    or if the identity does not reproduce ``ratings.power`` - which would mean
    these are not that fit's games.
    """
    it = ratings.internals
    if it is None:
        raise DecompositionUnavailable("this fit did not keep its solver internals")
    if it.recency_weighted or it.pace_adjusted:
        raise DecompositionUnavailable(
            "the decomposition identity holds only for the unweighted solve; "
            "this fit used recency weights or a pace adjustment"
        )
    games = games.reset_index(drop=True)
    if len(games) != len(it.margin):
        raise DecompositionUnavailable(
            f"{len(games)} games given, but the fit was run on {len(it.margin)}"
        )

    index = {team: i for i, team in enumerate(it.teams)}
    power = ratings.power.reindex(it.teams).to_numpy(float)
    home_edge = np.where(it.neutral, 0.0, it.raw_home_field)
    has_id = "game_id" in games.columns

    margins: list[list[float]] = [[] for _ in it.teams]
    opponents: list[list[float]] = [[] for _ in it.teams]
    shares: list[list[tuple]] = [[] for _ in it.teams]

    away_names = games["team1"].to_numpy()
    home_names = games["team2"].to_numpy()
    raw = (games["pts2"].to_numpy(float) - games["pts1"].to_numpy(float))
    for k in range(len(games)):
        gid = games["game_id"].iloc[k] if has_id else k
        away, home = away_names[k], home_names[k]
        ia, ib = index[away], index[home]
        # The host sees margin y and an edge of +h; the visitor sees -y and -h.
        for me, opp, sign, name in ((ib, ia, 1.0, away), (ia, ib, -1.0, home)):
            counted = it.scale * (sign * it.margin[k] - sign * home_edge[k])
            margins[me].append(counted)
            opponents[me].append(power[opp])
            shares[me].append((gid, name, sign * raw[k], counted, power[opp]))

    published_prior = it.scale * it.prior - it.anchor_shift
    out: dict[str, TeamShare] = {}
    worst = 0.0
    for i, team in enumerate(it.teams):
        n = len(margins[i])
        denom = n + it.ridge
        game_rows = [
            GameShare(g, opp, float(m_raw), float(counted), float(r_opp), float((counted + r_opp) / denom))
            for g, opp, m_raw, counted, r_opp in shares[i]
        ]
        share = _share(team, n, it.ridge, margins[i], opponents[i], float(published_prior[i]),
                       game_rows, float(power[i]))
        rebuilt = share.margin_term + share.opponent_term + share.prior_term
        worst = max(worst, abs(rebuilt - power[i]))
        out[team] = share

    if worst > TOLERANCE:
        raise DecompositionUnavailable(
            f"the identity misses the fitted ratings by {worst:.3g}; "
            "these games are not the ones the fit was run on"
        )
    return out
