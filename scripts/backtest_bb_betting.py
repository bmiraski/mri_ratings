"""Does the basketball model beat the market?

Two runs, deliberately. ESPN Bet carries four seasons and is the only market in
this feed that can support a multi-season answer. DraftKings - the book Ben
bets - covers one season, so it is a cross-check: if the two disagree, the
disagreement is the finding.

Run:  PYTHONPATH=src python3 scripts/backtest_bb_betting.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.betting import bb_backtest as bb, bb_lines  # noqa: E402


def clean(records: list[dict]) -> list[dict]:
    """NaN is valid in Python's json and invalid to every browser that parses
    the file afterwards. Nulls instead."""
    return [
        {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in row.items()}
        for row in records
    ]


def headline(priced: pd.DataFrame, label: str) -> dict:
    """Why the answer is what it is: the market is a better predictor."""
    import numpy as np

    m = priced.dropna(subset=["market"])
    graded = bb.evaluate(m)
    graded = graded[~graded["push"]]
    seasons = {
        int(s): round(float(c["won"].mean()), 4) for s, c in graded.groupby("season")
    }
    return {
        "book": label,
        "games": int(len(m)),
        "modelMae": round(float(np.abs(m["predicted"] - m["actual"]).mean()), 2),
        "marketMae": round(float(np.abs(m["market"] - m["actual"]).mean()), 2),
        "modelWinner": round(float((np.sign(m["predicted"]) == np.sign(m["actual"])).mean()), 4),
        "marketWinner": round(float((np.sign(m["market"]) == np.sign(m["actual"])).mean()), 4),
        "correlation": round(float(np.corrcoef(m["predicted"], m["market"])[0, 1]), 3),
        "ats": round(float(graded["won"].mean()), 4),
        "atsBySeason": seasons,
        "breakEven": bb.BREAK_EVEN,
    }


def report(priced: pd.DataFrame, label: str, against: str = "market") -> pd.DataFrame:
    graded = bb.evaluate(priced, against=against)
    if graded.empty:
        print(f"\n{label}: nothing priced")
        return pd.DataFrame()
    table = bb.summarize(graded)
    print(f"\n{label} - {len(graded)} games, vs {against}")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return table


def main() -> None:
    out = {}

    print("ESPN Bet, four seasons:")
    espn = bb.run(bb_lines.available_seasons("ESPN Bet"), "ESPN Bet")
    if not espn.empty:
        espn.to_parquet(ROOT / "data" / "parquet" / "bb_priced_espn.parquet")
        out["espn_close"] = clean(report(espn, "ESPN Bet closing").to_dict("records"))
        out["espn_headline"] = headline(espn, "ESPN Bet")
        known = espn[espn["known"]]
        report(known, "ESPN Bet closing, both teams rated")
        if espn["market_open"].notna().any():
            out["espn_open"] = clean(
                report(espn, "ESPN Bet opening", "market_open").to_dict("records")
            )

    print("\nDraftKings, one season - cross-check only:")
    dk = bb.run(bb_lines.available_seasons("DraftKings"), "DraftKings")
    if not dk.empty:
        dk.to_parquet(ROOT / "data" / "parquet" / "bb_priced_dk.parquet")
        out["dk_close"] = clean(report(dk, "DraftKings closing").to_dict("records"))
        out["dk_open"] = clean(report(dk, "DraftKings opening", "market_open").to_dict("records"))
        out["dk_headline"] = headline(dk, "DraftKings")

    (ROOT / "site" / "data" / "bb_betting.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nwritten to site/data/bb_betting.json")


if __name__ == "__main__":
    main()
