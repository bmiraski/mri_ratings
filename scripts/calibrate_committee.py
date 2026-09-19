"""How well does a blend of MRI Power and Résumé recover the committee's ranking?

The season simulation has to rank teams the way the CFP selection committee will,
and the committee's method is a judgement, not a formula. What can be measured
is how closely the two published MRI numbers track its final pre-bowl ranking
over the last twelve seasons, and which mix of them tracks it best.

The answer is résumé-heavy - about 70% résumé, 30% power - which is the committee
telling you what it values: what a team has done, more than how good it is.

``data/cfp_final_rankings.json`` holds each season's final pre-bowl committee
ranking, fetched week by week from CollegeFootballData's /rankings endpoint. Not
without the week: the season-wide response for older years returns polls that do
not belong to the week they are labelled with.

Run:  PYTHONPATH=src python3 scripts/calibrate_committee.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.ingest import cfbd  # noqa: E402
from mri.ratings import mri2  # noqa: E402

YEARS = range(2014, 2026)


def season_end_ratings() -> dict[int, pd.DataFrame]:
    """Power and résumé at the end of each regular season, championship games in."""
    finals, previous = {}, None
    for year in range(YEARS[0] - 1, YEARS[-1] + 1):
        games = cfbd.games(year)
        games = games[games["season_type"] == "regular"].reset_index(drop=True)
        fbs = sorted(set(games.loc[games["class1"] == "fbs", "team1"])
                     | set(games.loc[games["class2"] == "fbs", "team2"]))
        teams = sorted(set(games["team1"]) | set(games["team2"]))
        prior = mri2.build_prior(previous, teams, centre_teams=fbs) if previous is not None else None
        model = mri2.fit(games, prior=prior, neutral=games["neutral"], anchor_teams=fbs,
                         with_efficiency=False)
        table = model.table().set_index("team")
        finals[year] = table[table.index.isin(fbs)]
        previous = model.power
    return finals


def z(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std()


def main() -> None:
    committee = json.loads((ROOT / "data" / "cfp_final_rankings.json").read_text())
    finals = season_end_ratings()

    print("weight on power  top-12 hit  top-4 hit  rank error  spearman (top 25)")
    for weight in np.arange(0.0, 1.01, 0.1):
        rows = []
        for year in YEARS:
            table = finals[year]
            score = weight * z(table["power"]) + (1 - weight) * z(table["resume"])
            order = score.sort_values(ascending=False).index.tolist()
            actual = [r[1] for r in committee[str(year)]["ranks"]]
            position = {t: i + 1 for i, t in enumerate(order)}
            rows.append((
                len(set(actual[:12]) & set(order[:12])),
                len(set(actual[:4]) & set(order[:4])),
                np.mean([abs(position.get(t, 138) - (i + 1)) for i, t in enumerate(actual[:12])]),
                spearmanr([position.get(t, 138) for t in actual[:25]], range(1, 26)).statistic,
            ))
        a = np.mean(rows, axis=0)
        print(f"     {weight:.1f}           {a[0]:5.2f}       {a[1]:5.2f}       {a[2]:5.2f}      {a[3]:.3f}")


if __name__ == "__main__":
    main()
