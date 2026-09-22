"""Playing out a conference tournament, thousands of times.

Built from a :class:`~mri.bracket.template.Template` and a seed line (best team
first). Every round, the pool of teams still alive is sorted by seed and paired
end to end - the best remaining seed against the worst, the second-best against
the second-worst - and a new tier's worth of teams, if this round has one,
joins before the pairing happens. That reproduces the shape every conference's
bracket actually has: the top seeds' path gets easier the deeper their bye, and
whoever survives the bottom of the bracket keeps meeting better opponents.

**What this does not reproduce.** A true single-elimination bracket is drawn
once and stays fixed - two teams that were never going to meet before the final
still can't meet early just because of upsets elsewhere. Sorting and pairing by
whoever is left, round after round, is a reseeded bracket, and only some
conferences (the Horizon League among them) actually play theirs that way. For
a static bracket this is an approximation: it tends to matter more once one of
the very top seeds has already lost, when a true static bracket might not put
the other one in its path. Good enough to estimate who wins the auto bid; not
a substitute for knowing the real bracket once it's drawn.
"""

from __future__ import annotations

import numpy as np

from .template import Template

SIGMA = 11.0                 # points; the same single-game spread the basketball betting board uses


def win_probability(rating_diff: np.ndarray, home_edge: float = 0.0, sigma: float = SIGMA) -> np.ndarray:
    """Chance the first team wins, given its rating minus the opponent's (plus, if it hosts, home court)."""
    from scipy.stats import norm

    return norm.cdf((rating_diff + home_edge) / sigma)


def simulate(template: Template, seeds: list[str], power: dict, *, home_edge: float = 0.0, sims: int = 20_000,
            seed: int = 0) -> dict[str, float]:
    """Each team's chance of winning the automatic bid.

    ``seeds`` is every participant, best seed first - exactly ``template.size`` of
    them. ``power`` is a rating for each (higher is better; same units as the
    win-probability sigma above). Ties among the field for the same rating are
    fine; a real conference is never that close, but a simulated one occasionally is.
    """
    if len(seeds) != template.size:
        raise ValueError(f"{len(seeds)} seeds for a {template.size}-team template")
    rng = np.random.default_rng(seed)
    ratings = np.array([power[t] for t in seeds])
    wins = np.zeros(len(seeds))
    order = entry_order(template)
    for run in range(sims):
        wins[run_once(order, ratings, rng, home_edge)] += 1
    return {seeds[i]: float(wins[i] / sims) for i in range(len(seeds))}


def entry_order(template: Template) -> list[list[int]]:
    """Seed indexes (0 = best seed) grouped by the round they enter, worst group first.

    template.tiers[0] is the worst seeds, who enter round one; each later tier is a better group of
    seeds joining with a deeper bye. Building the list in that same order needs no reversal - the
    worst group is already first.
    """
    order: list[list[int]] = []
    cursor = template.size
    for count in template.tiers:
        order.append(list(range(cursor - count, cursor)))
        cursor -= count
    return order


def run_once(order: list[list[int]], ratings: np.ndarray, rng: np.random.Generator, home_edge: float = 0.0, *,
             prob=None, record: list | None = None) -> int:
    """Play one bracket through to a champion and return the champion's seed index.

    ``prob(a, b)``, if given, is the chance the better-rated ``a`` beats ``b`` - a precomputed lookup is
    far faster than a normal CDF per game when this runs thousands of times. ``record``, if given,
    collects every game as ``(winner, loser, better_rated)`` so a caller can put the games on a résumé.
    """
    alive = list(order[0])
    for tier in order[1:]:
        # One round happens as each later tier's byes join the survivors so far - this is why a
        # bracket with three tiers of byes plays five rounds to crown a champion from sixteen teams,
        # not the four a same-size bracket with no byes would need: the SEC's own tournament does
        # exactly this, and the real game log is what this was checked against.
        alive = _play_round(alive, ratings, rng, home_edge, prob=prob, record=record) + tier
    while len(alive) > 1:
        alive = _play_round(alive, ratings, rng, home_edge, prob=prob, record=record)
    return alive[0]


def _play_round(alive: list[int], ratings: np.ndarray, rng: np.random.Generator, home_edge: float, *,
                prob=None, record: list | None = None) -> list[int]:
    """One round: sort the pool by rating, best paired against worst, survivors returned in the same order."""
    ordered = sorted(alive, key=lambda i: -ratings[i])
    n = len(ordered)
    survivors = []
    for k in range(n // 2):
        a, b = ordered[k], ordered[n - 1 - k]
        p = prob(a, b) if prob is not None else win_probability(ratings[a] - ratings[b], home_edge)
        winner = a if rng.random() < p else b
        survivors.append(winner)
        if record is not None:
            record.append((winner, b if winner == a else a, a))
    if n % 2:
        survivors.append(ordered[n // 2])               # an odd pool: the middle seed sits out this round
    return survivors
