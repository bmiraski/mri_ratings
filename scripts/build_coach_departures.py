"""Classify every confirmed coaching departure: fired, moved up, or ambiguous.

Ambiguous cases are printed in full for hand review - edit their "label" to
fired_or_pushed_out / moved_up / retired_or_other and "source" to "manual"
directly in data/coach_departures.json once reviewed. Rows already marked
"manual" are carried over unchanged on every rebuild.

Run:  PYTHONPATH=src python3 scripts/build_coach_departures.py
"""

from __future__ import annotations

import datetime as dt
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.coaches import departures  # noqa: E402


def _no_nan(value):
    """Recursively turn float NaN into None.

    ``departures.build()`` already writes ``None`` for a missing vs_par -
    but it returns a DataFrame, and pandas silently coerces ``None`` back to
    ``NaN`` in a float column, which ``to_dict(orient="records")`` then
    surfaces again. Left alone, ``json.dumps`` still writes it (as the bare,
    non-standard token ``NaN``, not valid JSON), which is exactly the kind
    of thing a stricter JSON tool - such as whatever a hand-edit was made
    with - can choke on.
    """
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _no_nan(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_no_nan(v) for v in value]
    return value


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

    out_path = ROOT / "data" / "coach_departures.json"
    existing = json.loads(out_path.read_text())["departures"] if out_path.exists() else []
    records, orphans = departures.keep_manual_labels(_no_nan(table.to_dict(orient="records")), existing)
    kept = sum(1 for r in records if r.get("source") == "manual")
    print(f"\nkept {kept} hand-reviewed label(s) from the existing file")
    for r in orphans:
        print(f"  WARNING: manual label no longer matches a departure, dropped: "
              f"{r['season']} {r['school']} {r['coach_name']} ({r['label']})")
    out = {"generated": dt.date.today().isoformat(), "departures": records}
    out_path.write_text(json.dumps(out, indent=2, default=str, allow_nan=False) + "\n")
    print(f"\nwrote {len(table)} departures to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
