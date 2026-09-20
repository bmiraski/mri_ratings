"""Does the roster-aware prior predict games better than the old one?

Every game of 2015-2025 (2020 aside) is predicted using only the earlier weeks of
its own season, once with the old prior (last season, 30% toward average) and once
with the talent-and-returning-production prior. Both chains are run properly: each
season's prior comes from the *previous season's final ratings under the same
method*, not from a shared shortcut, and the new prior's coefficients are fitted
with the season being predicted left out.

The first weeks are where it matters. By November the games have taught the model
what the prior was guessing; in September the prior is most of what it knows.

Run:  PYTHONPATH=src python3 scripts/backtest_priors.py
Writes site/data/prior_backtest.json, which the method page reads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.betting import lines as lines_module  # noqa: E402
from mri.ingest import cfbd  # noqa: E402
from mri.ratings import mri2, prior_fit, priors  # noqa: E402

SPANS = (("weeks 1-3", 1, 3), ("weeks 4-6", 4, 6), ("weeks 7-13", 7, 13), ("all", 1, 13))


def main() -> None:
    power_np, fbs_np = prior_fit.season_ratings()
    data = prior_fit.dataset(power_np, fbs_np)

    chain = {"old": power_np[2014], "new": power_np[2014]}
    rows = []
    for year in range(2015, 2026):
        games = cfbd.games(year)
        games = games[(games["season_type"] == "regular") & games["played"]].reset_index(drop=True)
        fbs = fbs_np[year]
        teams = sorted(set(games["team1"]) | set(games["team2"]))
        last = cfbd.games(year - 1)
        continuing = set(fbs_np[year - 1])

        train = data[data["year"] != year]
        beta = np.linalg.lstsq(prior_fit.design(train), train["y"], rcond=None)[0]
        model = {"coefficients": dict(zip(priors.TERMS, beta))}
        prior = {
            "old": mri2.build_prior(chain["old"], teams, centre_teams=fbs),
            "new": priors.preseason_prior(
                chain["new"], teams, fbs=fbs, continuing=continuing,
                talent=cfbd.talent(year), returning=cfbd.returning(year), model=model),
        }
        book = lines_module.preferred_lines(year)
        book = book.drop_duplicates("game_id").set_index("game_id") if not book.empty else pd.DataFrame()

        if year != 2020:
            for week in range(1, 14):
                so_far = games[games["week"] < week]
                target = games[(games["week"] == week) & games["team1"].isin(fbs) & games["team2"].isin(fbs)]
                if target.empty:
                    continue
                for name in ("old", "new"):
                    if so_far.empty:
                        power, home = prior[name], mri2.DEFAULT_HOME_FIELD_PRIOR
                    else:
                        fitted = mri2.fit(so_far, prior=prior[name], neutral=so_far["neutral"], anchor_teams=fbs,
                                          with_resume=False, with_efficiency=False)
                        power, home = fitted.power.reindex(teams).fillna(prior[name]), fitted.home_field
                    for g in target.itertuples():
                        market = book["market"].get(g.game_id, np.nan) if len(book) else np.nan
                        rows.append((year, week, name, g.game_id,
                                     power[g.team2] - power[g.team1] + (0 if g.neutral else home),
                                     g.pts2 - g.pts1, market))
        for name in ("old", "new"):
            final = mri2.fit(games, prior=prior[name], neutral=games["neutral"], anchor_teams=fbs,
                             with_resume=False, with_efficiency=False)
            chain[name] = final.power
        print(f"  {year} done")

    frame = pd.DataFrame(rows, columns=["year", "week", "prior", "gid", "pred", "actual", "market"])
    out = {"seasons": [2015, 2025], "excluded": [2020], "spans": {}}
    for label, lo, hi in SPANS:
        span = frame[frame["week"].between(lo, hi)]
        entry = {}
        for name in ("old", "new"):
            s = span[span["prior"] == name]
            x = s["pred"] - s["pred"].mean()
            priced = s[s["market"].notna()]
            entry[name] = {
                "games": int(len(s)),
                "mae": round(float((s["pred"] - s["actual"]).abs().mean()), 2),
                "accuracy": round(float((np.sign(s["pred"]) == np.sign(s["actual"])).mean()), 4),
                "slope": round(float((x * (s["actual"] - s["actual"].mean())).sum() / (x * x).sum()), 2),
                "marketMae": round(float((priced["market"] - priced["actual"]).abs().mean()), 2),
            }
        a = span[span["prior"] == "old"].reset_index(drop=True)
        b = span[span["prior"] == "new"].reset_index(drop=True)
        gain = (a["pred"] - a["actual"]).abs() - (b["pred"] - b["actual"]).abs()
        entry["maeGain"] = round(float(gain.mean()), 2)
        entry["maeGainError"] = round(float(1.96 * gain.std(ddof=1) / np.sqrt(len(gain))), 2)
        out["spans"][label] = entry
        print(f"{label:11s} old {entry['old']['mae']:5.2f}  new {entry['new']['mae']:5.2f}  "
              f"gain {entry['maeGain']:+.2f} +/- {entry['maeGainError']:.2f} | accuracy "
              f"{entry['old']['accuracy']:.1%} -> {entry['new']['accuracy']:.1%} | slope "
              f"{entry['old']['slope']:.2f} -> {entry['new']['slope']:.2f} | market {entry['old']['marketMae']:.2f}")
    (ROOT / "site" / "data" / "prior_backtest.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
