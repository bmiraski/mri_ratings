"""Where will GameDay be? Play the season out and ask the choice model each week.

The choice model knows which games look big *given the rankings that week*. The
rankings that week are what nobody knows, so this plays the rest of the season
thousands of times, works out each run's committee-style ranking and each team's
record as of every target week, and asks the choice model who it would pick in each
run. Averaging those answers gives each game's chance of being the one.

Because the model that survived testing has no memory - whether GameDay has
already been to a place turned out not to help predict where it goes next - the
weeks are independent given a simulated season, and "chance a team hosts at least
once between now and the end" is exact arithmetic on the weekly chances, not a
noisy count.

The conference championship week is treated as what it is: a choice among the ten
title games, which are only known once standings settle, so each run picks its own
ten. Army-Navy is not modelled as a game; GameDay was there every year from 2014 to
2021 and has not been since, and that is stated as a plain rate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from ..sim import season
from . import features

MODEL_PATH = Path(__file__).resolve().parents[3] / "data" / "gameday_model.json"
CHUNK = 1_000
DEFAULT_SIMS = 4_000


def load_model(path: Path = MODEL_PATH) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def columns(model: dict) -> list[int]:
    return [features.NAMES.index(n) for n in model["features"]]


def forecast(
    teams: list[dict],
    schedule: pd.DataFrame,
    *,
    home_field: float,
    model: dict,
    weeks: list[int],
    championship_week: int = season.CHAMPIONSHIP_WEEK,
    sims: int = DEFAULT_SIMS,
    seed: int = season.DEFAULT_SEED,
    context: dict | None = None,
) -> dict:
    """Each candidate game's chance of hosting GameDay, for each week asked about.

    ``schedule`` is every game of the season with ``game_id``, ``week``, ``team1``
    (visitor), ``team2`` (host), ``played``, ``pts1``, ``pts2``, ``neutral`` and
    ``conference_game``. ``teams`` are FBS teams with ``team``, ``power`` and
    ``conference``. ``context`` carries what is known about each team before the
    season: ``lastRank`` (its rank at the end of last season) and ``brand`` (how
    often GameDay has wanted it lately), each a dict by team name.
    """
    names = [t["team"] for t in teams]
    T = len(names)
    index = {n: i for i, n in enumerate(names)}
    conference = np.array([t["conference"] for t in teams])
    power = np.array([float(t["power"]) for t in teams] + [0.0])
    power[T] = power[:T].min() - 8.0
    fcs = T
    context = context or {}
    last_rank = np.array([float(context.get("lastRank", {}).get(n, features.RANK_CAP)) for n in names] + [features.RANK_CAP])
    brand = np.array([float(context.get("brand", {}).get(n, 0.0)) for n in names] + [0.0])
    beta = np.asarray(model["coefficients"], dtype=float)
    cols = columns(model)
    p_other = float(model.get("otherRate", 0.05))

    sched = schedule[schedule["team1"].isin(index) | schedule["team2"].isin(index)].reset_index(drop=True)
    home = np.array([index.get(n, fcs) for n in sched["team2"]])
    away = np.array([index.get(n, fcs) for n in sched["team1"]])
    neutral = sched["neutral"].to_numpy(bool)
    played = sched["played"].to_numpy(bool)
    week = sched["week"].to_numpy(int)
    home_won = (sched["pts2"] > sched["pts1"]).to_numpy(bool)
    same_conf = np.array([h < T and a < T and conference[h] == conference[a] for h, a in zip(home, away)])
    conf_game = sched["conference_game"].to_numpy(bool) & same_conf
    G = len(sched)
    unplayed = np.flatnonzero(~played)
    U = len(unplayed)
    hf_all = np.where(neutral, 0.0, home_field)

    def incidence(side, rows, weight=None):
        m = np.zeros((len(rows), T), dtype=np.float32)
        s = side[rows]
        ok = s < T
        w = np.ones(len(rows), dtype=np.float32) if weight is None else weight[rows].astype(np.float32)
        m[np.flatnonzero(ok), s[ok]] = w[ok]
        return m

    every = np.arange(G)
    H_all, A_all = incidence(home, every), incidence(away, every)
    H_u, A_u = incidence(home, unplayed), incidence(away, unplayed)
    H_uc, A_uc = incidence(home, unplayed, conf_game), incidence(away, unplayed, conf_game)

    def tally(mask):
        h, a = mask & (home < T), mask & (away < T)
        wins = np.bincount(home[h & home_won], minlength=T + 1)[:T] + np.bincount(away[a & ~home_won], minlength=T + 1)[:T]
        losses = np.bincount(home[h & ~home_won], minlength=T + 1)[:T] + np.bincount(away[a & home_won], minlength=T + 1)[:T]
        return wins.astype(float), losses.astype(float)

    base_wins, base_losses = tally(played)
    base_conf, _ = tally(played & conf_game)
    n_played = base_wins + base_losses
    sd = season.SIGMA_GAME / np.sqrt(n_played + season.PRIOR_GAMES)
    week_u = week[unplayed]
    members = {c: np.flatnonzero(conference == c)
               for c in sorted(set(conference)) if c in season.POWER_FOUR + season.GROUP_OF_SIX}
    is_fbs_game = (home < T) & (away < T)

    # Candidate games per target week (the championship week is built per run).
    candidates = {}
    for w in weeks:
        if w == championship_week:
            continue
        idx = np.flatnonzero((week == w) & ~played & is_fbs_game)
        candidates[w] = idx

    rng = np.random.default_rng(seed)
    acc = {w: {} for w in weeks}
    games_p = {w: np.zeros(len(candidates[w])) for w in candidates}
    games_rank = {w: np.zeros((len(candidates[w]), 2)) for w in candidates}
    games_big = {w: np.zeros((len(candidates[w]), 2)) for w in candidates}      # both top-10, both unbeaten
    cg_p = {c: 0.0 for c in members}
    cg_stats = {c: np.zeros(5) for c in members}                               # rank a, rank b, both top-10, unbeaten, appearance
    never_host = np.zeros(T)         # summed over runs: the chance, in each run, of never hosting
    never_appear = np.zeros(T)
    host_week = {w: np.zeros(T) for w in weeks}
    appear_week = {w: np.zeros(T) for w in weeks}
    total = 0

    remaining = sims
    while remaining > 0:
        B = min(CHUNK, remaining)
        remaining -= B
        total += B
        rows = np.arange(B)

        true = np.empty((B, T + 1))
        true[:, :T] = power[:T] + rng.standard_normal((B, T)) * sd
        true[:, T] = power[T]
        hu, au = home[unplayed], away[unplayed]
        edge_u = np.where(neutral[unplayed], 0.0, home_field)
        margin = true[:, hu] - true[:, au] + edge_u + rng.standard_normal((B, U)) * season.SIGMA_GAME
        hw = margin > 0
        resid = (margin - (power[hu] - power[au] + edge_u)).astype(np.float32)
        hwf = hw.astype(np.float32)
        hw_all = np.empty((B, G), dtype=bool)
        hw_all[:, played] = home_won[played]
        hw_all[:, unplayed] = hw

        run_no_host = np.zeros((B, T))
        run_no_appear = np.zeros((B, T))

        for w in weeks:
            before_u = week_u < w
            Hb, Ab = H_u * before_u[:, None], A_u * before_u[:, None]
            rem_before = Hb.sum(axis=0) + Ab.sum(axis=0)
            sim_wins = hwf @ Hb + (1.0 - hwf) @ Ab
            wins = base_wins + sim_wins
            losses = base_losses + rem_before - sim_wins
            res = resid @ Hb - resid @ Ab
            est = power[:T] + res / (n_played + rem_before + season.PRIOR_GAMES)
            est_ext = np.concatenate([est, np.full((B, 1), power[T])], axis=1)

            before_all = week < w
            p_home = ndtr((0.0 - est_ext[:, away] + hf_all) / season.SIGMA_RESUME)
            p_away = ndtr((0.0 - est_ext[:, home] - hf_all) / season.SIGMA_RESUME)
            hwa = hw_all.astype(np.float32)
            Hm, Am = H_all * before_all[:, None], A_all * before_all[:, None]
            resume = (hwa - p_home) @ Hm + ((1.0 - hwa) - p_away) @ Am
            rank = features.committee_score_rank(est, resume)

            if w == championship_week:
                conf_wins = base_conf + hwf @ (H_uc * before_u[:, None]) + (1.0 - hwf) @ (A_uc * before_u[:, None])
                tops, tabs = [], []
                for c, idx in members.items():
                    key = conf_wins[:, idx] + rng.random((B, len(idx))) * 0.5
                    top = np.argsort(-key, axis=1)[:, :2]
                    tops.append(idx[top[:, 0]])
                    tabs.append(idx[top[:, 1]])
                ta, tb = np.stack(tops, axis=1), np.stack(tabs, axis=1)          # (B, conferences)
                rh, ra = np.take_along_axis(rank, ta, 1), np.take_along_axis(rank, tb, 1)
                lh, la = np.take_along_axis(losses, ta, 1), np.take_along_axis(losses, tb, 1)
                X = features.matrix(rank_home=rh, rank_away=ra, losses_home=lh, losses_away=la,
                                    last_rank_home=last_rank[ta], last_rank_away=last_rank[tb],
                                    brand_home=brand[ta], brand_away=brand[tb])[..., cols]
                p = _softmax(X @ beta) * (1.0 - p_other)
                for k, c in enumerate(members):
                    cg_p[c] += p[:, k].sum()
                    best, worst = np.minimum(rh[:, k], ra[:, k]), np.maximum(rh[:, k], ra[:, k])
                    cg_stats[c] += [rh[:, k].sum(), ra[:, k].sum(), (worst <= 10).sum(),
                                    ((lh[:, k] == 0) & (la[:, k] == 0)).sum(), p[:, k].sum()]
                    for side in (ta[:, k], tb[:, k]):
                        appear_week[w][:] += np.bincount(side, weights=p[:, k], minlength=T)[:T]
                        run_no_appear[rows, side] += np.log1p(-np.minimum(p[:, k], 0.999999))
                continue

            idx = candidates[w]
            if not len(idx):
                continue
            h, a = home[idx], away[idx]
            X = features.matrix(rank_home=rank[:, h], rank_away=rank[:, a], losses_home=losses[:, h],
                                losses_away=losses[:, a], last_rank_home=last_rank[h], last_rank_away=last_rank[a],
                                brand_home=brand[h], brand_away=brand[a])[..., cols]
            p = _softmax(X @ beta) * (1.0 - p_other)
            games_p[w] += p.sum(axis=0)
            rh, ra = rank[:, h], rank[:, a]
            games_rank[w] += np.stack([rh.sum(axis=0), ra.sum(axis=0)], axis=1)
            worst = np.maximum(rh, ra)
            games_big[w] += np.stack([(worst <= 10).sum(axis=0),
                                      ((losses[:, h] == 0) & (losses[:, a] == 0)).sum(axis=0)], axis=1)
            for k in range(len(idx)):
                host_week[w][h[k]] += p[:, k].sum()
                run_no_host[rows, h[k]] += np.log1p(-np.minimum(p[:, k], 0.999999))
                for side in (h[k], a[k]):
                    if side < T:
                        appear_week[w][side] += p[:, k].sum()
                        run_no_appear[rows, side] += np.log1p(-np.minimum(p[:, k], 0.999999))

        never_host += np.exp(run_no_host).sum(axis=0)
        never_appear += np.exp(run_no_appear).sum(axis=0)

    out_weeks = {}
    for w in weeks:
        if w == championship_week:
            entries = []
            for c in members:
                s = cg_stats[c] / total
                entries.append({
                    "kind": "championship", "conference": c, "probability": round(float(cg_p[c] / total), 4),
                    "expectedRankHome": round(float(s[0]), 1), "expectedRankAway": round(float(s[1]), 1),
                    "bothTop10": round(float(s[2]), 3), "bothUnbeaten": round(float(s[3]), 3),
                })
            out_weeks[w] = {"games": sorted(entries, key=lambda e: -e["probability"]), "other": round(p_other, 4)}
            continue
        entries = []
        for k, g in enumerate(candidates[w]):
            entries.append({
                "kind": "game", "gameId": int(sched.loc[g, "game_id"]),
                "home": names[home[g]], "away": names[away[g]], "neutral": bool(neutral[g]),
                "probability": round(float(games_p[w][k] / total), 4),
                "expectedRankHome": round(float(games_rank[w][k, 0] / total), 1),
                "expectedRankAway": round(float(games_rank[w][k, 1] / total), 1),
                "bothTop10": round(float(games_big[w][k, 0] / total), 3),
                "bothUnbeaten": round(float(games_big[w][k, 1] / total), 3),
            })
        out_weeks[w] = {"games": sorted(entries, key=lambda e: -e["probability"]), "other": round(p_other, 4)}

    team_out = {
        n: {"hostsAtLeastOnce": round(float(1.0 - never_host[i] / total), 4),
            "appearsAtLeastOnce": round(float(1.0 - never_appear[i] / total), 4)}
        for i, n in enumerate(names)
    }
    return {"sims": total, "weeks": out_weeks, "teams": team_out}


def _softmax(u: np.ndarray) -> np.ndarray:
    u = u - u.max(axis=-1, keepdims=True)
    p = np.exp(u)
    return p / p.sum(axis=-1, keepdims=True)
