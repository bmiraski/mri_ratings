"""What makes ESPN put College GameDay at a game, as numbers.

Everything here is computed the same way for a game played in 2016 and for one
that has not been scheduled to matter yet, from the two things this site already
knows about every team: how good it is and what it has done. Nothing is looked up
from an AP poll, because the AP poll of week 11 does not exist yet and ours does.

The features fall into four groups.

*Stakes.* How highly the two teams rank - by the same résumé-heavy blend the
season simulation uses to stand in for the selection committee - and whether they
are both unbeaten, both in the top 10, both in the top 25. GameDay goes to the game
the country will care about, and the country cares about the top of the rankings.

*The game.* How close the model thinks it is, and whether it is at a neutral site
(a conference championship, say) rather than a campus.

*The place.* Whether the host is in the SEC or Big Ten, and whether the visiting
team is better than the host (a top team going on the road is the classic GameDay
setup, and a top host is a better one).

*Wear.* GameDay does not like to return to the same place. How many times it has
already been to either team this season, and how often the host has hosted in
the last three seasons, which is where fatigue and brand pull in opposite
directions and the data decides which wins.
"""

from __future__ import annotations

import numpy as np

NAMES = (
    "best_rank",        # log of the better team's rank
    "worst_rank",       # log of the worse team's rank
    "both_top10",
    "both_top25",
    "both_unbeaten",
    "losses",           # combined losses, capped
    "closeness",        # how close the model line is, in 10-point units
    "neutral",
    "elite_conference",
    "host_better",      # the host outranks the visitor
    "repeat_home",      # times GameDay has already featured the host this season
    "repeat_away",      # ...and the visitor
    "host_recent",      # times the host has hosted in the previous three seasons
    "rivalry",          # they met at this point of the season, in each of the last two years
    "last_rank_best",   # log of the better team's rank at the end of last season
    "last_rank_worst",
    "brand",            # how often the two teams have been GameDay stops lately
)

RANK_CAP = 60
ELITE = ("SEC", "Big Ten")


def elite_conference(home_conf, away_conf, neutral) -> np.ndarray:
    home = np.isin(np.asarray(home_conf, dtype=object), ELITE)
    away = np.isin(np.asarray(away_conf, dtype=object), ELITE)
    return np.where(np.asarray(neutral, dtype=bool), home | away, home)


def matrix(*, rank_home, rank_away, losses_home, losses_away, spread=0.0, neutral=0.0, elite=0.0,
           repeat_home=0.0, repeat_away=0.0, host_recent=0.0, rivalry=0.0,
           last_rank_home=None, last_rank_away=None, brand_home=0.0, brand_away=0.0) -> np.ndarray:
    """Feature rows for any number of games. Every argument broadcasts."""
    lh_rank = np.minimum(np.asarray(RANK_CAP if last_rank_home is None else last_rank_home, dtype=float), RANK_CAP)
    la_rank = np.minimum(np.asarray(RANK_CAP if last_rank_away is None else last_rank_away, dtype=float), RANK_CAP)
    rh = np.minimum(np.asarray(rank_home, dtype=float), RANK_CAP)
    ra = np.minimum(np.asarray(rank_away, dtype=float), RANK_CAP)
    best, worst = np.minimum(rh, ra), np.maximum(rh, ra)
    lh, la = np.asarray(losses_home, dtype=float), np.asarray(losses_away, dtype=float)
    columns = (
        np.log(best),
        np.log(worst),
        (worst <= 10).astype(float),
        (worst <= 25).astype(float),
        ((lh == 0) & (la == 0)).astype(float),
        np.minimum(lh + la, 6.0),
        np.minimum(np.abs(np.asarray(spread, dtype=float)) / 10.0, 3.0),
        np.asarray(neutral, dtype=float),
        np.asarray(elite, dtype=float),
        (rh < ra).astype(float),
        np.minimum(np.asarray(repeat_home, dtype=float), 2.0),
        np.minimum(np.asarray(repeat_away, dtype=float), 2.0),
        np.minimum(np.asarray(host_recent, dtype=float), 3.0),
        np.asarray(rivalry, dtype=float),
        np.log(np.minimum(lh_rank, la_rank)),
        np.log(np.maximum(lh_rank, la_rank)),
        np.log1p(np.asarray(brand_home, dtype=float)) + np.log1p(np.asarray(brand_away, dtype=float)),
    )
    shape = np.broadcast(*columns).shape
    return np.stack([np.broadcast_to(c, shape) for c in columns], axis=-1)


def committee_score_rank(power: np.ndarray, resume: np.ndarray, weight: float = 0.30) -> np.ndarray:
    """1 = best. The same blend the season simulation uses for the committee."""
    def z(x):
        return (x - x.mean(axis=-1, keepdims=True)) / x.std(axis=-1, keepdims=True)

    score = weight * z(power) + (1.0 - weight) * z(resume)
    order = np.argsort(-score, axis=-1)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(1, score.shape[-1] + 1)[None, :] if score.ndim == 2
                      else np.arange(1, score.shape[-1] + 1), axis=-1)
    return rank
