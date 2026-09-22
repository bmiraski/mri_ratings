"""Phase 3: the whole field at once, thousands of times.

Phase 1 answered "who wins each conference's automatic bid" and Phase 2 "given
the automatic bids, who gets the at-large spots and what seed is everyone" - each
on its own. They aren't independent. A bubble team's chances depend on how many
bid thieves win their conference tournaments that year; a conference favourite
who loses in its tournament final drops to an at-large résumé it may or may not
survive on. So each simulated world here plays out, in order:

1. **The rest of the regular season**, game by game, from current power ratings
   (the same single-game model as the betting board: normal, sigma 11 points,
   home court where there is one).
2. **Every conference's standings**, from that world's results, and **every
   conference tournament** in that world's seed order (Phase 1's bracket engine,
   one run per world), whose games then count on each team's résumé the way they
   do on the committee's sheet.
3. **The automatic bids**: the tournament winner, for conferences Ben has
   switched to the model; the standings leader, for conferences still on the
   placeholder (Phase 1's rule - see ``data/bracketology_settings.json``). A
   conference whose real tournament is already over has a real champion, and
   every world uses it whatever the setting.
4. **The at-large field and the true seed list**, from Phase 2's composite score
   applied to that world's résumés - plus that world's own draw of *committee
   noise*, since the score is a model of the committee and not the committee - and
   **seed lines** by the NCAA's own format (:mod:`mri.bracket.seeding` - the
   76-team Opening Round from 2027).

**Ratings aren't exact.** A February rating is an estimate from twenty-odd games, so
each world draws its own error for every team - shaped as in the football
simulation, the single-game spread over the square root of games played plus the
fit's prior weight, and scaled by :data:`RATING_UNCERTAINTY` from the backtest -
and plays that world's games on the perturbed ratings.
Without it, a team the model rates a point ahead of its league looks surer to win
the league than it is. Résumés still read the published ratings: the committee's
sheet doesn't know our error bars either.

**What stays fixed within a run.** Power ratings aren't refit inside each world
- the same simplification the football season simulation makes, and the right one
at this scale: a win moves a team's rating a fraction of a point, but it moves its
*résumé* by up to a full win, and résumé is what the committee reads. That's also
what makes thousands of worlds cheap: MRI 2.0's résumé is a sum over games of
(result - what an average team would have expected), so with power fixed every
résumé feature is a sum over games, and a whole batch of worlds is one sparse
matrix product rather than thousands of refits.

**What's approximate.** A conference tournament that's underway when this runs is
finished by reseeding whoever is still alive rather than following the real
bracket's remaining path; conference-tournament eligibility (a reclassifying
program, a postseason ban) isn't modelled; standings ties are broken by
conference wins and then at random, not by each league's own tiebreakers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm

from . import atlarge, resume as resume_mod, seeding, simulate
from .template import Template

MODES = ("placeholder", "model")
# Independents share a conference label in the feed and the registry in some seasons (2013, 2023-24),
# and two of them playing each other looks like a conference game - but there's no automatic bid for
# being the best independent. They're at-large candidates like anyone else.
NO_AUTO_BID = frozenset({"Indep.", "Independent", "Independents"})
# The ridge's evidence weight in games (MRI 2.0's basketball ridge), as in the football simulation: a
# team's rating error is the single-game spread over sqrt(games played + this).
PRIOR_GAMES = 8.0
# How much to scale that error. Football's formula (1.0) is the starting point; the February 1 backtest
# (scripts/backtest_bracketology.py, 2011-2025) scores 1.0 / 1.5 / 2.0 / 3.0 at Brier 26.75 / 26.68 /
# 26.63 / 26.64 per season, so 2.0 - past it, nothing changes. Some of what it's standing in for is that
# ratings don't update inside a world as its results come in.
RATING_UNCERTAINTY = 2.0


@dataclass
class Inputs:
    """Everything one run needs. ``games`` is the whole season's schedule (canonical team names, a
    ``played`` column); ``as_of`` splits it - results before that date, simulation on and after. Leave
    ``as_of`` as None live: every game that's been played counts, every one that hasn't is simulated."""

    season: int
    games: pd.DataFrame
    power: pd.Series
    home_field: float
    resume_sigma: float                              # the fit's own residual sigma, which MRI 2.0's résumé uses
    conference_of: dict[str, str]                    # the selection universe: D1 team -> its conference
    templates: dict[str, Template | None]
    beta: np.ndarray
    fmt: seeding.Format
    auto_mode: dict[str, str] = field(default_factory=dict)
    as_of: str | None = None
    sims: int = 2000
    seed: int = 0
    game_sigma: float = simulate.SIGMA
    committee_noise: float = 0.0                     # score units; see scripts/backtest_atlarge.py
    rating_uncertainty: float = RATING_UNCERTAINTY   # multiplier on rating error; 0 treats ratings as exact


@dataclass
class Result:
    teams: pd.DataFrame                              # one row per team in the universe
    champions: dict[str, dict[str, float]]           # conference -> team -> P(automatic bid)
    conference_status: dict[str, str]                # conference -> not started / underway / decided
    sims: int


def is_conference_tournament(g: pd.DataFrame) -> pd.Series:
    """A conference's own tournament: same-conference ``TRNMNT`` games in February or March, filed as
    ``regular`` (the NCAA tournament is the only thing this feed calls postseason)."""
    month = g["start_date"].str[5:7]
    return ((g["game_type"] == "TRNMNT") & (g["conf1"] == g["conf2"]) & g["conf1"].notna()
            & (g["season_type"] == "regular") & month.isin(["02", "03"]))


def split(inputs: Inputs) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """(results, games still to simulate, conference-tournament flag for the results)."""
    g = inputs.games[inputs.games["season_type"] == "regular"].copy()
    before = g["start_date"] < inputs.as_of if inputs.as_of else pd.Series(True, index=g.index)
    played = g["played"].astype(bool) & before
    rated = g["team1"].isin(inputs.power.index) & g["team2"].isin(inputs.power.index)
    results = g[played & rated]
    future = g[~played & ~is_conference_tournament(g) & rated]
    return results, future, is_conference_tournament(results)


class _Sides:
    """Every (team, game) pairing in a set of games, with what each one contributes to a résumé."""

    def __init__(self, games: pd.DataFrame, inputs: Inputs, rank: pd.Series, index: dict[str, int]):
        n = len(games)
        self.n_games = n
        power = inputs.power
        away, home = games["team1"].to_numpy(), games["team2"].to_numpy()
        neutral = games["neutral"].to_numpy(bool)
        edge = np.where(neutral, 0.0, inputs.home_field)
        p_home = power.reindex(home).to_numpy(float)
        p_away = power.reindex(away).to_numpy(float)
        # Chance the home side wins this game, for simulating it (the betting board's model)...
        self.p_home_wins = norm.cdf((p_home - p_away + edge) / inputs.game_sigma)
        # ...and what an average team would expect from each side of it, for the résumé (MRI 2.0's).
        exp_home = norm.cdf((0.0 - p_away + edge) / inputs.resume_sigma)
        exp_away = norm.cdf((0.0 - p_home - edge) / inputs.resume_sigma)

        self.rows = {}
        for side, team, opp, expected, loc in (
            ("home", home, away, exp_home, np.where(neutral, "neutral", "home")),
            ("away", away, home, exp_away, np.where(neutral, "neutral", "away")),
        ):
            idx = np.array([index.get(t, -1) for t in team])
            opp_rank = rank.reindex(opp).to_numpy(float)
            quad = np.array([resume_mod.quadrant(r, l) for r, l in zip(opp_rank, loc)]) if n else np.zeros(0, int)
            self.rows[side] = {
                "idx": idx, "expected": expected, "opp_rank": opp_rank, "quad": quad,
                "road_neutral": loc != "home", "game": np.arange(n),
            }

    def incidence(self, side: str, weight: np.ndarray, n_teams: int) -> sparse.csr_matrix:
        r = self.rows[side]
        keep = r["idx"] >= 0
        return sparse.csr_matrix((np.asarray(weight, float)[keep], (r["idx"][keep], r["game"][keep])),
                                 shape=(n_teams, self.n_games))


def _accumulate(features: dict[str, np.ndarray], sides: _Sides, home_won: np.ndarray, n: int) -> None:
    """Add a batch of games to the feature matrices. ``home_won`` is games x worlds (or games x 1 for results)."""
    home_won = np.asarray(home_won, dtype=float)
    away_won = 1.0 - home_won
    for side, won in (("home", home_won), ("away", away_won)):
        r = sides.rows[side]
        lost = 1.0 - won
        ones = np.ones(sides.n_games)
        M = sides.incidence(side, ones, n)
        features["wins"] += M @ won
        features["losses"] += M @ lost
        features["resume"] += M @ won - (sides.incidence(side, r["expected"], n) @ ones)[:, None]
        features["road_neutral_wins"] += sides.incidence(side, r["road_neutral"], n) @ won
        q1 = sides.incidence(side, r["quad"] == 1, n)
        features["quad1_wins"] += q1 @ won
        features["quad1_losses"] += q1 @ lost
        features["bad_losses"] += sides.incidence(side, r["quad"] >= 3, n) @ lost
        features["opp_rank_sum"] += (sides.incidence(side, np.nan_to_num(r["opp_rank"]), n) @ ones)[:, None]
        features["games"] += (M @ ones)[:, None]


def _standings_key(conf_wins: np.ndarray, conf_losses: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Higher is better: conference win percentage, then conference wins, then a coin flip."""
    pct = conf_wins / np.maximum(conf_wins + conf_losses, 1)
    return pct * 1e4 + conf_wins + rng.random(conf_wins.shape) * 0.5


def run(inputs: Inputs) -> Result:
    S = inputs.sims
    rng = np.random.default_rng(inputs.seed)
    rank = inputs.power.rank(ascending=False, method="min")
    universe = sorted(t for t, c in inputs.conference_of.items() if c and t in inputs.power.index)
    index = {t: i for i, t in enumerate(universe)}
    n = len(universe)

    results, future, ct_results = split(inputs)

    names = ("wins", "losses", "resume", "road_neutral_wins", "quad1_wins", "quad1_losses", "bad_losses",
             "opp_rank_sum", "games")
    features = {k: np.zeros((n, S)) for k in names}

    # Results: the same for every world.
    past = _Sides(results, inputs, rank, index)
    _accumulate(features, past, results["win2"].to_numpy(float)[:, None], n)
    wins_now, losses_now = features["wins"][:, 0].copy(), features["losses"][:, 0].copy()

    # Each world's rating error: one draw per team, per world.
    pos = {t: i for i, t in enumerate(inputs.power.index)}
    played_count = pd.concat([results["team1"], results["team2"]]).value_counts().reindex(inputs.power.index)
    sd = inputs.rating_uncertainty * inputs.game_sigma / np.sqrt(played_count.fillna(0).to_numpy(float) + PRIOR_GAMES)
    error = rng.normal(0.0, 1.0, (len(pos), S)) * sd[:, None]

    # The rest of the regular season, drawn once per world, on that world's ratings.
    upcoming = _Sides(future, inputs, rank, index)
    home_pos = np.array([pos[t] for t in future["team2"]], dtype=int)
    away_pos = np.array([pos[t] for t in future["team1"]], dtype=int)
    shift = (error[home_pos] - error[away_pos]) / inputs.game_sigma if len(future) else np.zeros((0, S))
    p_world = norm.cdf(norm.ppf(np.clip(upcoming.p_home_wins, 1e-12, 1 - 1e-12))[:, None] + shift)
    future_home_won = (rng.random((len(future), S)) < p_world).astype(float)
    _accumulate(features, upcoming, future_home_won, n)

    champions_idx, champion_counts, status = _conference_tournaments(
        inputs, results, ct_results, future, future_home_won, rank, index, features, rng, error, pos)

    # Score every world, then select and seed each one.
    cols = {
        "wins": features["wins"], "losses": features["losses"], "resume": features["resume"],
        "road_neutral_wins": features["road_neutral_wins"], "quad1_wins": features["quad1_wins"],
        "quad1_losses": features["quad1_losses"], "bad_losses": features["bad_losses"],
        "sos": features["opp_rank_sum"] / np.maximum(features["games"], 1),
        "powerRank": np.repeat(rank.reindex(universe).to_numpy(float)[:, None], S, axis=1),
    }
    score = atlarge.score_columns(atlarge.columns(cols), inputs.beta)
    mean_score = score.mean(axis=1)
    if inputs.committee_noise > 0:
        # The composite score models the committee; it isn't the committee. Each world gets its own
        # draw of everything the score can't see (sized from Phase 2's own misses), or every team
        # the score likes would come out a certainty and ~1 in 6 of those would miss.
        score = score + rng.normal(0.0, inputs.committee_noise, score.shape)

    in_field = np.zeros(n)
    auto = np.zeros(n)
    opening = np.zeros(n)
    line_counts = np.zeros((n, 17))
    overall_sum = np.zeros(n)
    fmt = inputs.fmt
    for s in range(S):
        autos = np.unique(champions_idx[:, s][champions_idx[:, s] >= 0])
        is_auto_team = np.zeros(n, dtype=bool)
        is_auto_team[autos] = True
        slots = fmt.size - len(autos)
        candidates = np.flatnonzero(~is_auto_team)
        pick = candidates[np.argpartition(score[candidates, s], slots - 1)[:slots]] if slots > 0 else candidates[:0]
        chosen = np.concatenate([autos, pick])
        chosen = chosen[np.argsort(score[chosen, s], kind="stable")]
        seed_line, plays_in = seeding.lines(is_auto_team[chosen], fmt)
        in_field[chosen] += 1
        auto[autos] += 1
        opening[chosen[plays_in]] += 1
        line_counts[chosen, seed_line] += 1
        overall_sum[chosen] += np.arange(1, len(chosen) + 1)

    table = pd.DataFrame({
        "team": universe,
        "conference": [inputs.conference_of[t] for t in universe],
        "power": inputs.power.reindex(universe).to_numpy(float),
        "powerRank": rank.reindex(universe).to_numpy(float),
        "pField": in_field / S, "pAuto": auto / S, "pAtLarge": (in_field - auto) / S, "pOpening": opening / S,
        "expectedOverall": np.where(in_field > 0, overall_sum / np.maximum(in_field, 1), np.nan),
        "meanScore": mean_score,
        "winsNow": wins_now, "lossesNow": losses_now,
        "projectedWins": features["wins"].mean(axis=1),
        "projectedLosses": features["losses"].mean(axis=1),
    })
    for line in range(1, 17):
        table[f"seed{line}"] = line_counts[:, line] / S
    with np.errstate(invalid="ignore"):
        table["expectedSeed"] = (line_counts[:, 1:] * np.arange(1, 17)).sum(axis=1) / np.where(in_field > 0, in_field, np.nan)
    champions = {c: {universe[i]: cnt / S for i, cnt in counts.items()} for c, counts in champion_counts.items()}
    return Result(teams=table.sort_values(["pField", "meanScore"], ascending=[False, True]).reset_index(drop=True),
                  champions=champions, conference_status=status, sims=S)


def _conference_tournaments(inputs, results, ct_results, future, future_home_won, rank, index, features, rng,
                            error, pos):
    """Standings, conference tournaments and automatic bids, world by world.

    Returns a (conferences x worlds) array of champion indexes into the universe (-1 when a
    conference's champion isn't in it), per-conference champion counts, and each conference's status.
    """
    S = inputs.sims
    g_all = pd.concat([results, future])
    conf_games = g_all[(g_all["game_type"] == "STD") & (g_all["conf1"] == g_all["conf2"]) & g_all["conf1"].notna()]
    conferences = sorted(c for c in conf_games["conf1"].unique() if c not in NO_AUTO_BID)
    champions_idx = np.full((len(conferences), S), -1, dtype=int)
    counts: dict[str, dict[int, int]] = {}
    status: dict[str, str] = {}
    future_pos = {gid: k for k, gid in enumerate(future["game_id"])}
    # Conference tournament games on the schedule that haven't counted yet (live: not played; in a
    # backtest: on or after the as-of date).
    g = inputs.games[inputs.games["season_type"] == "regular"]
    ct_all = g[is_conference_tournament(g)]
    pending = ct_all[~ct_all["game_id"].isin(results["game_id"])]
    remaining_ct = pending["conf1"].value_counts().to_dict()

    power = inputs.power
    rank_arr = rank

    for ci, conf in enumerate(conferences):
        cg = conf_games[conf_games["conf1"] == conf]
        members = sorted((set(cg["team1"]) | set(cg["team2"])) & set(power.index))
        m_idx = {t: i for i, t in enumerate(members)}
        wins = np.zeros((len(members), S))
        losses = np.zeros((len(members), S))
        for r in cg.itertuples():
            a, h = m_idx.get(r.team1), m_idx.get(r.team2)
            if a is None or h is None:
                continue
            k = future_pos.get(r.game_id)
            home_won = future_home_won[k] if k is not None else np.full(S, float(r.win2))
            wins[h] += home_won
            losses[h] += 1 - home_won
            wins[a] += 1 - home_won
            losses[a] += home_won
        key = _standings_key(wins, losses, rng)
        order = np.argsort(-key, axis=0)                          # members x worlds, best first

        tpl = inputs.templates.get(conf)
        played_ct = results[ct_results.reindex(results.index, fill_value=False) & (results["conf1"] == conf)]
        mode = inputs.auto_mode.get(conf, "placeholder")
        ratings = power.reindex(members).to_numpy(float)
        world_ratings = ratings[:, None] + error[[pos[m] for m in members]]          # members x worlds
        hosted = bool(tpl and tpl.campus_hosted)
        edge = inputs.home_field if hosted else 0.0
        # P[a, b, s]: in world s, the chance a beats b when a is the better-rated side there.
        P = norm.cdf((world_ratings[:, None, :] - world_ratings[None, :, :] + edge) / inputs.game_sigma)
        champs_local = np.empty(S, dtype=int)
        record_all: list[tuple[int, int, int, int]] = []           # (world, winner, loser, better-rated)

        if not played_ct.empty:
            losers = {r.team1 if r.win2 == 1.0 else r.team2 for r in played_ct.itertuples()}
            entrants = set(played_ct["team1"]) | set(played_ct["team2"])
            unbeaten = [t for t in entrants if t not in losers]
            size = tpl.size if tpl else len(members)
            final_order = [members[i] for i in order[:, 0]]       # standings are final once a tournament starts
            # Single elimination: E teams need exactly E - 1 games, so one unbeaten team after E - 1 games is
            # necessary for a finished tournament - but not sufficient: after every game of a stepladder
            # (the Southland's), exactly one entrant is unbeaten while the top seeds still wait. So it also
            # needs either the whole field to have entered, or no more of this conference's tournament
            # games on the schedule. Standings can't settle it alone: a reclassifying program can finish
            # in the top K and never be eligible to enter at all.
            still_scheduled = remaining_ct.get(conf, 0) > 0
            decided = (len(played_ct) == len(entrants) - 1 and len(unbeaten) == 1
                       and (len(entrants) >= size or not still_scheduled))
            waiting = [t for t in final_order if t not in entrants][:max(size - len(entrants), 0)]
            alive = [m_idx[t] for t in (unbeaten if decided else unbeaten + waiting) if t in m_idx]
            alive.sort(key=lambda i: final_order.index(members[i]))
            if decided:
                status[conf] = "decided"
                champs_local[:] = alive[0]
            else:
                status[conf] = "underway"
                pool = alive
                order_once = [list(range(len(pool)))]
                for s in range(S):
                    rec: list = []
                    w = simulate.run_once(order_once, world_ratings[pool, s], rng,
                                          prob=lambda a, b, s=s: P[pool[a], pool[b], s], record=rec)
                    champs_local[s] = pool[w]
                    record_all += [(s, pool[x], pool[y], pool[z]) for x, y, z in rec]
        elif tpl is None:
            status[conf] = "no tournament"
            champs_local[:] = order[0]
        else:
            status[conf] = "not started"
            size = min(tpl.size, len(members))
            bracket_order = simulate.entry_order(tpl) if size == tpl.size else [list(range(size))]
            for s in range(S):
                seeds = order[:size, s]
                rec: list = []
                w = simulate.run_once(bracket_order, world_ratings[seeds, s], rng,
                                      prob=lambda a, b, seeds=seeds, s=s: P[seeds[a], seeds[b], s], record=rec)
                champs_local[s] = seeds[0] if mode == "placeholder" else seeds[w]
                record_all += [(s, seeds[x], seeds[y], seeds[z]) for x, y, z in rec]

        if record_all:
            _add_tournament_games(features, record_all, members, ratings, hosted, inputs, rank_arr, index)
        uni = np.array([index.get(members[i], -1) for i in range(len(members))])
        champions_idx[ci] = uni[champs_local]
        vals, cnts = np.unique(champions_idx[ci][champions_idx[ci] >= 0], return_counts=True)
        counts[conf] = dict(zip(vals.tolist(), cnts.tolist()))
    return champions_idx, counts, status


def _add_tournament_games(features, record, members, ratings, hosted, inputs, rank, index) -> None:
    """Put one conference's simulated tournament games on each world's résumés."""
    rec = np.array(record)
    world, winner, loser, better = rec[:, 0], rec[:, 1], rec[:, 2], rec[:, 3]
    member_power = ratings
    member_rank = rank.reindex(members).to_numpy(float)
    uni = np.array([index.get(t, -1) for t in members])
    for team, opp, won in ((winner, loser, 1.0), (loser, winner, 0.0)):
        if hosted:
            loc = np.where(team == better, "home", "away")
        else:
            loc = np.full(len(team), "neutral")
        edge = np.where(loc == "home", inputs.home_field, np.where(loc == "away", -inputs.home_field, 0.0))
        expected = norm.cdf((0.0 - member_power[opp] + edge) / inputs.resume_sigma)
        opp_rank = member_rank[opp]
        quad = np.array([resume_mod.quadrant(r, l) for r, l in zip(opp_rank, loc)])
        rows = uni[team]
        keep = rows >= 0
        rows, w = rows[keep], world[keep]
        np.add.at(features["wins" if won else "losses"], (rows, w), 1.0)
        np.add.at(features["resume"], (rows, w), won - expected[keep])
        np.add.at(features["games"], (rows, w), 1.0)
        np.add.at(features["opp_rank_sum"], (rows, w), np.nan_to_num(opp_rank[keep]))
        if won:
            np.add.at(features["road_neutral_wins"], (rows, w), (loc[keep] != "home").astype(float))
            np.add.at(features["quad1_wins"], (rows, w), (quad[keep] == 1).astype(float))
        else:
            np.add.at(features["quad1_losses"], (rows, w), (quad[keep] == 1).astype(float))
            np.add.at(features["bad_losses"], (rows, w), (quad[keep] >= 3).astype(float))


def projected_field(result: Result, fmt: seeding.Format) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """One bracket to show: the most likely automatic bid from each conference, the at-large teams
    most often in the field, and the whole field ordered by average composite score across worlds -
    plus the bubble around the cutoff. Region placement runs on this (:func:`mri.bracket.regions.place`)."""
    t = result.teams.set_index("team")
    autos = []
    for conf, odds in result.champions.items():
        if odds:
            best = max(odds, key=lambda team: (odds[team], t.at[team, "pField"] if team in t.index else 0))
            if best in t.index:
                autos.append(best)
    autos = list(dict.fromkeys(autos))
    rest = t.drop(index=autos).sort_values(["pAtLarge", "meanScore"], ascending=[False, True])
    at_large = rest.head(fmt.size - len(autos))
    field_ = pd.concat([t.loc[autos].assign(bidType="auto"), at_large.assign(bidType="at-large")])
    field_ = field_.sort_values("meanScore").reset_index()
    field_["trueSeed"] = np.arange(1, len(field_) + 1)
    field_["seedLine"], field_["opening"] = seeding.lines(field_["bidType"].eq("auto").to_numpy(), fmt)
    out = rest.iloc[fmt.size - len(autos):]
    bubble = {
        "openingRoundAtLarge": field_[(field_["bidType"] == "at-large") & field_["opening"]]["team"].tolist(),
        "firstFourOut": out.head(4).index.tolist(),
        "nextFourOut": out.iloc[4:8].index.tolist(),
    }
    return field_, bubble
