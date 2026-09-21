"""Season simulation: play out the rest of the schedule thousands of times.

Every remaining game is drawn from the ratings, conference standings are
settled, the ten conference championship games are played, a committee ranking
is produced, the 12-team field is selected under the 2026-27 rules, and the
bracket is played. Counting what happened across the runs gives each team's
chance of winning its conference, making the field, earning a bye, and winning
the title. Nothing here is a forecast in the sense of a claim: it is the ratings
taken seriously, with their uncertainty included.

Three modelling choices decide whether the numbers are honest, and each is
stated rather than buried.

*Ratings are uncertain, and early in the season very.* A team's rating is an
estimate. Each run first draws every team's *true* strength around its rating,
with a spread that shrinks as games accumulate (``SIGMA_GAME / sqrt(games +
PRIOR_GAMES)``, the same evidence weighting the ridge fit uses). Without this,
Week 3 would say Indiana wins the title 40% of the time, because a rating with
no error bars is a certainty. Game noise is set to ``SIGMA_GAME`` = 15.5 so that
noise plus rating error lands on the 16.5-point spread of real margins.

*The committee is modelled, not asked.* The selection committee ranks by
résumé far more than by strength. Blending the two published MRI numbers - 70%
résumé, 30% power - recovers 10.8 of the committee's true top 12 and 3.7 of its
top 4 across 2014-2025, with the average team landing 1.5 places from its real
rank (``scripts/calibrate_committee.py``). It is a proxy and is described as one.

*Conference tiebreakers are simplified.* Ties in conference wins are broken by
coin flip. Real rules start with head-to-head, so a tied pair's odds are
slightly wrong; the top two of every conference are otherwise exact.

Selection follows the 2026-27 rules published by the CFP: the ACC, Big 12, Big
Ten and SEC champions are in; so is the highest-ranked team from the American,
Conference USA, MAC, Mountain West, Pac-12 and Sun Belt, champion or not; Notre
Dame is in if it is ranked in the top 12; the rest are the best remaining teams.
The 12 are seeded by ranking, the top four get byes, and 5-12 play at the higher
seed's home. Semifinal pairings follow the bracket: 1's side meets 4's side, and
2's meets 3's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr

POWER_FOUR = ("ACC", "Big 12", "Big Ten", "SEC")
GROUP_OF_SIX = ("American", "Conference USA", "MAC", "Mountain West", "Pac-12", "Sun Belt")
INDEPENDENT_AUTO = "Notre Dame"

# The higher seed hosts these conferences' championship games. The four power
# conferences and the MAC play theirs at a neutral site.
TOP_SEED_HOSTS = ("American", "Conference USA", "Mountain West", "Pac-12", "Sun Belt")

FIELD = 12
CHAMPIONSHIP_WEEK = 14             # the calendar's week for conference title games
COMMITTEE_POWER_WEIGHT = 0.30      # the rest is résumé; see calibrate_committee.py
# Even with perfect information the proxy misses about one of the committee's
# twelve, so a probability near 100% claims more than the proxy knows. Noise on
# the score, in z-score units, stands in for what the committee weighs that the
# ratings do not. Chosen by backtest_simulation.py, which sweeps it.
COMMITTEE_NOISE = 0.20
SIGMA_GAME = 15.5                  # intrinsic game noise, before rating error
SIGMA_RESUME = 16.5                # the spread résumé's "average team" is judged on
PRIOR_GAMES = 4.0                  # the ridge's evidence weight, in games
DEFAULT_SIMS = 10_000
DEFAULT_SEED = 2026
CHUNK = 2_000


def simulate(
    teams: list[dict],
    schedule: pd.DataFrame,
    *,
    home_field: float,
    sims: int = DEFAULT_SIMS,
    seed: int = DEFAULT_SEED,
    track: list[int] | None = None,
    championships: dict[str, dict] | None = None,
    rules: dict | None = None,
    observe=None,
) -> dict:
    """Simulate the rest of the season.

    ``teams``     dicts with ``team``, ``power`` and ``conference`` (FBS only).
    ``schedule``  every game of the season, played or not, with ``team1`` the
                  visitor and ``team2`` the host, ``played``, ``pts1``,
                  ``pts2``, ``neutral`` and ``conference_game``.
    ``track``     game ids whose effect on each side's playoff chance to
                  measure - the games worth explaining.

    ``championships``
                  conference title games whose participants are already known,
                  keyed by conference: ``{"a": home side, "b": other side,
                  "neutral": bool, "played": bool, "a_won": bool}``. Once they
                  are set the standings stop mattering for that conference.
                  A game already played stays in ``schedule``, where it counts
                  toward records and résumé like any other.
    ``observe``   a callable handed each chunk of runs as it finishes, for callers that
                  need more than the summary counts (the Heisman forecast needs each
                  team's finish in each run). It receives a dict of ``rank`` (1 = best,
                  the committee blend *without* the committee's noise), ``losses``,
                  ``wins`` and ``made_cg``, each shaped (runs, teams).
    ``rules``     overrides ``power_four``, ``group_of_six`` and ``top_seed_hosts``
                  (tuples of conference names), ``uncertainty`` (a multiplier on
                  rating error) and ``committee_noise``. Only the historical backtest needs this: the
                  conferences had other names, and other members.

    Deterministic for a given input and seed, so the page only changes when a
    result or a rating does.
    """
    rules = rules or {}
    p4 = tuple(rules.get("power_four", POWER_FOUR))
    g6 = tuple(rules.get("group_of_six", GROUP_OF_SIX))
    hosts = tuple(rules.get("top_seed_hosts", TOP_SEED_HOSTS))
    names = [t["team"] for t in teams]
    T = len(names)
    index = {n: i for i, n in enumerate(names)}
    conference = np.array([t["conference"] for t in teams])
    power = np.array([float(t["power"]) for t in teams] + [0.0])
    power[T] = power[:T].min() - 8.0            # a non-FBS opponent, as team pages price it
    fcs = T

    sched = schedule[
        schedule["team1"].isin(index) | schedule["team2"].isin(index)
    ].reset_index(drop=True)
    home = np.array([index.get(n, fcs) for n in sched["team2"]])
    away = np.array([index.get(n, fcs) for n in sched["team1"]])
    neutral = sched["neutral"].to_numpy(bool)
    played = sched["played"].to_numpy(bool)
    home_won = (sched["pts2"] > sched["pts1"]).to_numpy(bool)
    same_conf = np.array(
        [h < T and a < T and conference[h] == conference[a] for h, a in zip(home, away)]
    )
    conf_game = sched["conference_game"].to_numpy(bool) & same_conf

    G = len(sched)
    unplayed = np.flatnonzero(~played)
    U = len(unplayed)
    hf_all = np.where(neutral, 0.0, home_field)

    def incidence(side: np.ndarray, rows: np.ndarray, weight=None) -> np.ndarray:
        m = np.zeros((len(rows), T), dtype=np.float32)
        s = side[rows]
        ok = s < T
        w = np.ones(len(rows), dtype=np.float32) if weight is None else weight[rows].astype(np.float32)
        m[np.flatnonzero(ok), s[ok]] = w[ok]
        return m

    every = np.arange(G)
    H_all, A_all = incidence(home, every), incidence(away, every)
    H_u, A_u = incidence(home, unplayed), incidence(away, unplayed)
    H_uc = incidence(home, unplayed, conf_game)
    A_uc = incidence(away, unplayed, conf_game)

    # What has already happened, per team.
    def tally(mask: np.ndarray):
        h = mask & (home < T)
        a = mask & (away < T)
        wins = np.bincount(home[h & home_won], minlength=T + 1)[:T] \
            + np.bincount(away[a & ~home_won], minlength=T + 1)[:T]
        losses = np.bincount(home[h & ~home_won], minlength=T + 1)[:T] \
            + np.bincount(away[a & home_won], minlength=T + 1)[:T]
        return wins.astype(float), losses.astype(float)

    base_wins, base_losses = tally(played)
    base_conf_wins, _ = tally(played & conf_game)
    n_played = base_wins + base_losses
    remaining = (H_u.sum(axis=0) + A_u.sum(axis=0)).astype(float)
    # ``uncertainty`` scales the rating error. It exists so the backtest can show
    # what the error bars are worth by switching them off; the site never does.
    sd = float(rules.get("uncertainty", 1.0)) * SIGMA_GAME / np.sqrt(n_played + PRIOR_GAMES)

    members = {c: np.flatnonzero(conference == c) for c in sorted(set(conference)) if c in p4 + g6}
    is_p4 = np.isin(conference, p4)
    is_g6 = np.isin(conference, g6)
    nd = index.get(INDEPENDENT_AUTO)

    tracked = []
    if track:
        position = {int(sched.loc[g, "game_id"]): k for k, g in enumerate(unplayed)}
        for gid in track:
            k = position.get(int(gid))
            if k is None:
                continue
            g = unplayed[k]
            tracked.append((int(gid), k, int(home[g]), int(away[g])))
    lever = {gid: np.zeros(6) for gid, *_ in tracked}

    rng = np.random.default_rng(seed)
    acc = {key: np.zeros(T) for key in (
        "cg", "champ", "field", "bye", "quarter", "semi", "final", "title",
        "wins", "unbeaten", "top12")}
    total = 0

    remaining_sims = sims
    while remaining_sims > 0:
        B = min(CHUNK, remaining_sims)
        remaining_sims -= B
        total += B
        rows = np.arange(B)

        true = np.empty((B, T + 1))
        true[:, :T] = power[:T] + rng.standard_normal((B, T)) * sd
        true[:, T] = power[T]

        # ---- the rest of the regular season
        hu, au = home[unplayed], away[unplayed]
        edge_u = np.where(neutral[unplayed], 0.0, home_field)
        mu = true[:, hu] - true[:, au] + edge_u
        margin = mu + rng.standard_normal((B, U)) * SIGMA_GAME
        hw = margin > 0
        hwf = hw.astype(np.float32)
        expected = power[hu] - power[au] + edge_u
        resid = (margin - expected).astype(np.float32)

        sim_wins = hwf @ H_u + (1.0 - hwf) @ A_u
        conf_wins = base_conf_wins + hwf @ H_uc + (1.0 - hwf) @ A_uc
        reg_wins = base_wins + sim_wins
        reg_losses = base_losses + remaining - sim_wins
        res_sum = resid @ H_u - resid @ A_u

        # ---- standings and championship games
        champ = np.zeros((B, T), dtype=bool)
        made_cg = np.zeros((B, T), dtype=bool)
        cg_res = np.zeros((B, T), dtype=np.float32)
        already_played = np.zeros((B, T), dtype=bool)
        cg_rows = []           # (a, b, a_won, hfa) for the résumé
        for c, idx in members.items():
            if len(idx) < 2:
                continue
            known = (championships or {}).get(c)
            if known:
                ta = np.full(B, index[known["a"]])
                tb = np.full(B, index[known["b"]])
                hfa = 0.0 if known.get("neutral", True) else home_field
                if known.get("played"):
                    # Already in the ratings and in the résumé: only the title matters.
                    winner = np.where(known["a_won"], ta, tb)
                    champ[rows, winner] = True
                    made_cg[rows, ta] = True
                    made_cg[rows, tb] = True
                    already_played[rows, ta] = True
                    already_played[rows, tb] = True
                    continue
            else:
                key = conf_wins[:, idx] + rng.random((B, len(idx))) * 0.5
                top = np.argsort(-key, axis=1)[:, :2]
                ta, tb = idx[top[:, 0]], idx[top[:, 1]]
                hfa = home_field if c in hosts else 0.0
            m = true[rows, ta] - true[rows, tb] + hfa + rng.standard_normal(B) * SIGMA_GAME
            a_won = m > 0
            winner = np.where(a_won, ta, tb)
            champ[rows, winner] = True
            made_cg[rows, ta] = True
            made_cg[rows, tb] = True
            r = (m - (power[ta] - power[tb] + hfa)).astype(np.float32)
            cg_res[rows, ta] += r
            cg_res[rows, tb] -= r
            cg_rows.append((ta, tb, a_won, hfa))
        res_sum = res_sum + cg_res

        # ---- ratings after the games are played
        n_final = n_played + remaining + (made_cg & ~already_played)
        est = power[:T] + res_sum / (n_final + PRIOR_GAMES)
        est_ext = np.concatenate([est, np.full((B, 1), power[T])], axis=1)

        # ---- résumé: wins above an average team's, against the schedule played
        hw_all = np.empty((B, G), dtype=bool)
        hw_all[:, played] = home_won[played]
        hw_all[:, unplayed] = hw
        p_home = ndtr((0.0 - est_ext[:, away] + hf_all) / SIGMA_RESUME)
        p_away = ndtr((0.0 - est_ext[:, home] - hf_all) / SIGMA_RESUME)
        hwa = hw_all.astype(np.float32)
        resume = (hwa - p_home) @ H_all + ((1.0 - hwa) - p_away) @ A_all
        for ta, tb, a_won, hfa in cg_rows:
            pa = ndtr((0.0 - est_ext[rows, tb] + hfa) / SIGMA_RESUME)
            pb = ndtr((0.0 - est_ext[rows, ta] - hfa) / SIGMA_RESUME)
            resume[rows, ta] += a_won - pa
            resume[rows, tb] += (~a_won) - pb

        # ---- the committee's ranking, and the field
        def z(x):
            return (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)

        score = COMMITTEE_POWER_WEIGHT * z(est) + (1.0 - COMMITTEE_POWER_WEIGHT) * z(resume)
        if observe is not None:
            clean = np.argsort(-score, axis=1)
            clean_rank = np.empty_like(clean)
            clean_rank[rows[:, None], clean] = np.arange(1, T + 1)[None, :]
            observe({"rank": clean_rank, "losses": reg_losses.copy(), "wins": reg_wins.copy(), "made_cg": made_cg.copy()})
        score = score + float(rules.get("committee_noise", COMMITTEE_NOISE)) \
            * rng.standard_normal((B, T))
        order = np.argsort(-score, axis=1)
        rank = np.empty_like(order)
        rank[rows[:, None], order] = np.arange(T)[None, :]

        auto = champ & is_p4[None, :]
        g6_rank = np.where(is_g6[None, :], rank, T + 1)
        best_g6 = np.argmin(g6_rank, axis=1)
        auto[rows, best_g6] = True
        if nd is not None:
            auto[:, nd] |= rank[:, nd] < FIELD
        priority = np.where(auto, -1, rank)
        chosen = np.argsort(priority, axis=1, kind="stable")[:, :FIELD]
        chosen_rank = rank[rows[:, None], chosen]
        seeds = chosen[rows[:, None], np.argsort(chosen_rank, axis=1)]   # (B, 12), seed 1 first

        # ---- the bracket, played on the same true strengths
        def play(a, b, a_hosts):
            m = true[rows, a] - true[rows, b] + (home_field if a_hosts else 0.0) \
                + rng.standard_normal(B) * SIGMA_GAME
            return np.where(m > 0, a, b)

        s = [seeds[:, k] for k in range(FIELD)]
        w89, w710 = play(s[7], s[8], True), play(s[6], s[9], True)
        w611, w512 = play(s[5], s[10], True), play(s[4], s[11], True)
        q1, q2 = play(s[0], w89, False), play(s[3], w512, False)
        q3, q4 = play(s[2], w611, False), play(s[1], w710, False)
        sf_a, sf_b = play(q1, q2, False), play(q3, q4, False)
        title = play(sf_a, sf_b, False)

        def count(*groups):
            return np.bincount(np.concatenate(groups), minlength=T)[:T].astype(float)

        acc["cg"] += made_cg.sum(axis=0)
        acc["champ"] += champ.sum(axis=0)
        acc["field"] += count(seeds.ravel())
        acc["bye"] += count(*s[:4])
        acc["quarter"] += count(*s[:4], w89, w710, w611, w512)
        acc["semi"] += count(q1, q2, q3, q4)
        acc["final"] += count(sf_a, sf_b)
        acc["title"] += count(title)
        acc["wins"] += reg_wins.sum(axis=0)
        acc["unbeaten"] += (reg_losses == 0).sum(axis=0)
        acc["top12"] += (rank < FIELD).sum(axis=0)

        if tracked:
            in_field = np.zeros((B, T), dtype=bool)
            in_field[rows[:, None], seeds] = True
            for gid, k, h, a in tracked:
                won = hw[:, k]
                lev = lever[gid]
                lev[0] += won.sum()
                lev[1] += (~won).sum()
                if h < T:
                    lev[2] += in_field[won, h].sum()
                    lev[3] += in_field[~won, h].sum()
                if a < T:
                    lev[4] += in_field[won, a].sum()
                    lev[5] += in_field[~won, a].sum()

    out = {}
    for i, name in enumerate(names):
        out[name] = {
            "conferenceGame": round(acc["cg"][i] / total, 4),
            "conferenceTitle": round(acc["champ"][i] / total, 4),
            "playoff": round(acc["field"][i] / total, 4),
            "bye": round(acc["bye"][i] / total, 4),
            "quarterfinal": round(acc["quarter"][i] / total, 4),
            "semifinal": round(acc["semi"][i] / total, 4),
            "final": round(acc["final"][i] / total, 4),
            "title": round(acc["title"][i] / total, 4),
            "projectedWins": round(acc["wins"][i] / total, 2),
            "projectedLosses": round(base_wins[i] + base_losses[i] + remaining[i] - acc["wins"][i] / total, 2),
            "unbeaten": round(acc["unbeaten"][i] / total, 4),
            "top12": round(acc["top12"][i] / total, 4),
            "gamesLeft": int(remaining[i]),
        }

    leverage = {}
    for gid, k, h, a in tracked:
        n_home_won, n_home_lost, h_if_won, h_if_lost, a_if_h_won, a_if_h_lost = lever[gid]
        if n_home_won < 30 or n_home_lost < 30:
            continue                       # too lopsided to say what the other result would do
        entry = {}
        home_swing = away_swing = 0.0
        if h < T:
            entry["home"] = {"ifWin": round(h_if_won / n_home_won, 4),
                             "ifLose": round(h_if_lost / n_home_lost, 4)}
            home_swing = entry["home"]["ifWin"] - entry["home"]["ifLose"]
        if a < T:
            # The visitor wins when the host loses, so the columns swap.
            entry["away"] = {"ifWin": round(a_if_h_lost / n_home_lost, 4),
                             "ifLose": round(a_if_h_won / n_home_won, 4)}
            away_swing = entry["away"]["ifWin"] - entry["away"]["ifLose"]
        entry["swing"] = round(max(abs(home_swing), abs(away_swing)), 4)
        leverage[str(gid)] = entry

    return {
        "sims": total,
        "seed": seed,
        "teams": out,
        "leverage": leverage,
        "fieldSize": FIELD,
    }
