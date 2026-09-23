"""Have the two basketball betting habits made money? The backtest the betting page quotes.

*Underdog picks*: the model has the market underdog winning outright; bet that
underdog with the points. *The fade list*: bet against a team covering 35% or
fewer of its games after ten or more.

Both are graded on the walk-forward prices ``backtest_bb_betting.py`` saved -
the model as it stood the morning of each game - against ESPN Bet over four
seasons, with DraftKings (Ben's book, one season) as the cross-check. A team's
cover record for the fade list is built only from games before the day being
bet, the same no-peeking rule.

Several fade thresholds are reported so the chosen one is not the lucky one.
Break-even at -110 is 52.38%.

Run:  PYTHONPATH=src python3 scripts/backtest_bb_strategies.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.betting.backtest import BREAK_EVEN, _one_sided_p  # noqa: E402
from mri.betting.bb_tracker import FADE_MAX_COVER, FADE_MIN_GAMES, WIN_UNITS  # noqa: E402

BANDS = ((0, 3), (3, 6), (6, 10), (10, 99))
FADE_RULES = ((8, 0.35), (10, 0.30), (FADE_MIN_GAMES, FADE_MAX_COVER), (15, 0.35))


def record(won: pd.Series) -> dict:
    won = won.dropna().astype(int)
    n, w = len(won), int(won.sum())
    return {"bets": n, "wins": w, "losses": n - w,
            "ats": round(w / n, 4) if n else None,
            "units": round(w * WIN_UNITS - (n - w), 1),
            "p": round(float(_one_sided_p(w, n)), 4) if n else None}


def graded(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame[frame["market"] != 0].copy()
    d["home_cover"] = np.where(d["actual"] > d["market"], 1.0, np.where(d["actual"] < d["market"], 0.0, np.nan))
    return d.dropna(subset=["home_cover"]).sort_values("day")


def underdogs(d: pd.DataFrame) -> dict:
    picked = d[np.sign(d["predicted"]) != np.sign(d["market"])]
    dog_covered = pd.Series(np.where(picked["market"] < 0, picked["home_cover"], 1 - picked["home_cover"]),
                            index=picked.index)
    out = {"all": record(dog_covered),
           "bySeason": {str(s): record(dog_covered[g.index]) for s, g in picked.groupby("season")},
           "bySpread": []}
    for lo, hi in BANDS:
        band = picked[(picked["market"].abs() >= lo) & (picked["market"].abs() < hi)]
        out["bySpread"].append({"band": f"{lo}–{hi}" if hi < 99 else f"{lo}+", **record(dog_covered[band.index])})
    return out


def fades(d: pd.DataFrame) -> dict:
    """Each team's cover record before the day, then how betting against the listed ones went."""
    history: dict[str, list[int]] = {}
    rows = []
    for _, today in d.groupby("day", sort=True):
        for r in today.itertuples():
            for team, side in ((r.home, "home"), (r.away, "away")):
                past = history.get(team, [])
                rows.append({"idx": r.Index, "team": team, "side": side, "n": len(past),
                             "rate": float(np.mean(past)) if past else np.nan})
        for r in today.itertuples():
            history.setdefault(r.home, []).append(int(r.home_cover))
            history.setdefault(r.away, []).append(int(1 - r.home_cover))
    t = pd.DataFrame(rows).join(d[["home_cover", "season"]], on="idx")
    t["fade_won"] = np.where(t["side"] == "home", 1 - t["home_cover"], t["home_cover"])
    out = {"rules": []}
    for n_min, rate in FADE_RULES:
        chosen = t[(t["n"] >= n_min) & (t["rate"] <= rate)]
        entry = {"minGames": n_min, "maxCover": rate, **record(chosen["fade_won"])}
        if (n_min, rate) == (FADE_MIN_GAMES, FADE_MAX_COVER):
            out["all"] = entry
            out["bySeason"] = {str(s): record(g["fade_won"]) for s, g in chosen.groupby("season")}
        out["rules"].append(entry)
    return out


def main() -> None:
    out = {"breakEven": BREAK_EVEN}
    for book, name in (("espn", "ESPN Bet"), ("dk", "DraftKings")):
        path = ROOT / "data" / "parquet" / f"bb_priced_{book}.parquet"
        if not path.exists():
            print(f"{name}: no priced games - run scripts/backtest_bb_betting.py first")
            continue
        d = graded(pd.read_parquet(path))
        out[book] = {"book": name, "games": len(d), "seasons": sorted(int(s) for s in d["season"].unique()),
                     "underdogs": underdogs(d), "fades": fades(d)}
        u, f = out[book]["underdogs"]["all"], out[book]["fades"]["all"]
        print(f"{name}: underdog picks {u['wins']}-{u['losses']} ({u['ats']:.1%}, {u['units']:+.1f}u); "
              f"fades {f['wins']}-{f['losses']} ({f['ats']:.1%}, {f['units']:+.1f}u)")
    (ROOT / "site" / "data" / "bb_strategies.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
