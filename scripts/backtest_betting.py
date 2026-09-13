"""Run the betting backtest and save what the site needs to tell the truth.

Walk-forward across 2014-2025: to price week W the ratings see only weeks
before W, plus a prior from last season. That is what a bettor has on Friday.

Writes site/data/betting.json - the summary tables the site publishes, results
included when they are bad, because a betting page that only reports its good
weeks is worse than no betting page.

Run:  PYTHONPATH=src python3 scripts/backtest_betting.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.betting import backtest  # noqa: E402

SEASONS = range(2014, 2026)


def main() -> None:
    cache = ROOT / "data" / "parquet" / "betting_priced.parquet"
    if cache.exists():
        priced = pd.read_parquet(cache)
        print(f"loaded {len(priced)} priced games from cache")
    else:
        print("pricing seasons walk-forward...")
        priced = backtest.run(SEASONS)
        priced.to_parquet(cache)

    close = backtest.evaluate(priced, against="market")
    close_summary = backtest.summarize(close)

    opens = priced.dropna(subset=["market_open"])
    open_graded = backtest.evaluate(opens, against="market_open")
    open_summary = backtest.summarize(open_graded)

    clv_rows = []
    graded = open_graded[~open_graded["push"]]
    for low, high in [(0, 2), (2, 4), (4, 6), (6, 9), (9, 99)]:
        chunk = graded[(graded["edge"].abs() >= low) & (graded["edge"].abs() < high)]
        if len(chunk) < 40:
            continue
        clv_rows.append(
            {
                "edge": f"{low}-{high}" if high < 99 else f"{low}+",
                "bets": len(chunk),
                "clv": round(float(chunk["clv"].mean()), 2),
                "ats": round(float(chunk["won"].mean()), 4),
            }
        )

    by_season = []
    for season, chunk in graded.groupby("season"):
        wins, n = int(chunk["won"].sum()), len(chunk)
        by_season.append(
            {
                "season": int(season),
                "bets": n,
                "ats": round(wins / n, 4),
                "units": round(wins - (n - wins) * 1.1, 1),
                "clv": round(float(chunk["clv"].mean()), 2),
            }
        )

    payload = {
        "seasons": [int(priced["season"].min()), int(priced["season"].max())],
        "closing": {
            "bets": int(close_summary.iloc[-1]["bets"]),
            "ats": round(float(close_summary.iloc[-1]["ats"]), 4),
            "units": float(close_summary.iloc[-1]["units"]),
            "buckets": close_summary.to_dict("records"),
        },
        "opening": {
            "seasons": sorted(int(s) for s in opens["season"].unique()),
            "bets": int(open_summary.iloc[-1]["bets"]),
            "ats": round(float(open_summary.iloc[-1]["ats"]), 4),
            "units": float(open_summary.iloc[-1]["units"]),
            "pValue": round(float(open_summary.iloc[-1]["p_value"]), 4),
            "clv": round(float(graded["clv"].mean()), 3),
            "buckets": open_summary.to_dict("records"),
            "bySeason": by_season,
            "clvGradient": clv_rows,
        },
        "breakEven": backtest.BREAK_EVEN,
    }

    out = ROOT / "site" / "data" / "betting.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))

    print(f"\nvs closing: {payload['closing']['ats']:.4f} ATS on {payload['closing']['bets']} bets "
          f"({payload['closing']['units']:+.0f} units)")
    print(f"vs opening: {payload['opening']['ats']:.4f} ATS on {payload['opening']['bets']} bets "
          f"({payload['opening']['units']:+.0f} units, p={payload['opening']['pValue']})")
    print(f"mean CLV:   {payload['opening']['clv']:+.3f} points")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
