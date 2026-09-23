"""Where should an FCS team's prior regress to?

Every game of 2014-2025 with an FBS side is predicted walk-forward: ratings
refitted before each week on the games already played, with the same fit and
``priors.for_season`` chain as the weekly build. Two rules for the teams outside
the FBS field:

  old   regress toward the FBS average, like everyone else; a newcomer starts at
        70% of replacement level
  new   regress toward replacement level; a newcomer starts there

Scored on MAE and on log loss at the site's game sigma, with a paired t across
windows (each season cut into five chronological blocks), for all games,
FBS-vs-FBS games and FCS games separately.

Run:  PYTHONPATH=src python3 scripts/backtest_fcs_prior.py
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.betting import board  # noqa: E402
from mri.ingest import cfbd, registry  # noqa: E402
from mri.ratings import mri2, priors  # noqa: E402

WARMUP = 2013
SCORED = range(2014, 2026)
WINDOWS_PER_SEASON = 5


def canonical(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("team1", "team2"):
        frame[column] = [registry.resolve(n, n) for n in frame[column]]
    return frame


def sequence(games: pd.DataFrame) -> pd.Series:
    """Regular-season weeks in order, then the postseason as one block (the
    feed numbers postseason weeks from 1 again)."""
    regular = games["season_type"] == "regular"
    last = int(games.loc[regular, "week"].max()) if regular.any() else 0
    return pd.Series(np.where(regular, games["week"], last + 1), index=games.index)


def walk_forward() -> pd.DataFrame:
    first = canonical(cfbd.games(WARMUP - 1))
    fbs = [t for t in sorted(set(first["team1"]) | set(first["team2"])) if registry.is_fbs(t)]
    previous = mri2.fit(first, neutral=first["neutral"], anchor_teams=fbs,
                        with_resume=False, with_efficiency=False).power
    rows = []
    for season in range(WARMUP, SCORED[-1] + 1):
        games = canonical(cfbd.games(season)).reset_index(drop=True)
        games["seq"] = sequence(games)
        teams = sorted(set(games["team1"]) | set(games["team2"]))
        fbs = [t for t in teams if registry.is_fbs(t)]
        prior = priors.for_season(season, previous, teams, fbs)
        for block in sorted(games["seq"].unique()):
            train, target = games[games["seq"] < block], games[games["seq"] == block]
            if train.empty:
                power, home_field = prior, mri2.DEFAULT_HOME_FIELD_PRIOR
            else:
                model = mri2.fit(train, prior=prior, neutral=train["neutral"], anchor_teams=fbs,
                                 with_resume=False, with_efficiency=False)
                power, home_field = model.power.reindex(teams).fillna(prior), model.home_field
            for g in target.itertuples():
                line = power[g.team2] - power[g.team1] + (0.0 if g.neutral else home_field)
                rows.append((g.game_id, season, int(block), g.class1 == cfbd.FBS and g.class2 == cfbd.FBS,
                             float(line), float(g.pts2 - g.pts1)))
        previous = mri2.fit(games, prior=prior, neutral=games["neutral"], anchor_teams=fbs,
                            with_resume=False, with_efficiency=False).power
    frame = pd.DataFrame(rows, columns=["game_id", "season", "seq", "fbs_both", "line", "margin"])
    return frame[frame["season"].isin(SCORED)].set_index("game_id")


def run(outsiders_to_replacement: bool) -> pd.DataFrame:
    original = mri2.build_prior
    mri2.build_prior = functools.partial(original, outsiders_to_replacement=outsiders_to_replacement)
    try:
        frame = walk_forward()
    finally:
        mri2.build_prior = original
    p = np.clip(norm.cdf(frame["line"] / board.SIGMA), 1e-6, 1 - 1e-6)
    won = (frame["margin"] > 0).astype(float)
    frame["abs_err"] = (frame["margin"] - frame["line"]).abs()
    frame["log_loss"] = -(won * np.log(p) + (1 - won) * np.log(1 - p))
    order = frame.groupby("season")["seq"].rank(method="first")
    frame["window"] = np.ceil(order / frame.groupby("season")["seq"].transform("size")
                              * WINDOWS_PER_SEASON).astype(int)
    return frame


def paired_t(old: pd.DataFrame, new: pd.DataFrame, metric: str) -> float:
    """Positive when the new rule is better."""
    gain = (old.groupby(["season", "window"])[metric].mean()
            - new.groupby(["season", "window"])[metric].mean()).dropna()
    return float(gain.mean() / (gain.std(ddof=1) / np.sqrt(len(gain))))


def main() -> None:
    old, new = run(False), run(True)
    new = new.loc[old.index]
    scopes = {"all games": slice(None), "FBS vs FBS": old["fbs_both"], "FCS games": ~old["fbs_both"]}
    print(f"{'':12}{'games':>7}{'MAE old':>9}{'new':>7}{'LL old':>9}{'new':>8}"
          f"{'t(LL)':>7}{'t(MAE)':>8}{'seasons':>9}")
    for name, mask in scopes.items():
        o, n = old[mask], new[mask]
        better = int((o.groupby("season")["log_loss"].mean() > n.groupby("season")["log_loss"].mean()).sum())
        print(f"{name:12}{len(o):>7}{o['abs_err'].mean():>9.3f}{n['abs_err'].mean():>7.3f}"
              f"{o['log_loss'].mean():>9.4f}{n['log_loss'].mean():>8.4f}"
              f"{paired_t(o, n, 'log_loss'):>7.2f}{paired_t(o, n, 'abs_err'):>8.2f}"
              f"{better:>5}/{len(SCORED)}")
    fcs = ~old["fbs_both"]
    print(f"\nFCS games, mean of margin - line: old {(old['margin'] - old['line'])[fcs].mean():+.2f}, "
          f"new {(new['margin'] - new['line'])[fcs].mean():+.2f}")


if __name__ == "__main__":
    main()
