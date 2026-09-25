"""Classify every confirmed coaching departure: fired, moved up, or ambiguous.

Ambiguous cases are printed in full for hand review - edit their "label" to
fired_or_pushed_out / moved_up / retired_or_other and "source" to "manual"
directly in data/coach_departures.json once reviewed.

Run:  PYTHONPATH=src python3 scripts/build_coach_departures.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.coaches import departures  # noqa: E402


def main() -> None:
    coach_season = pd.read_parquet(ROOT / "data" / "parquet" / "coach_season.parquet")

    table = departures.build(coach_season)
    print(f"{len(table)} confirmed departures, {coach_season['season'].min()}-{coach_season['season'].max()}")

    counts = table["label"].value_counts()
    for label in (departures.FIRED, departures.MOVED_UP, departures.AMBIGUOUS):
        print(f"  {label:20} {counts.get(label, 0)}")

    ambiguous = table[table["label"] == departures.AMBIGUOUS].sort_values(["season", "school"])
    print(f"\n{len(ambiguous)} ambiguous case(s) for hand review:")
    for row in ambiguous.itertuples():
        print(f"  {row.season} {row.school:24} {row.coach_name:22} {row.reason}")

    out = {
        "generated": dt.date.today().isoformat(),
        "departures": table.to_dict(orient="records"),
    }
    out_path = ROOT / "data" / "coach_departures.json"
    out_path.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(f"\nwrote {len(table)} departures to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
