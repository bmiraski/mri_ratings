"""Fit and walk-forward validate the hot-seat model: P(fired) at a season's end.

Ships only if it beats the win%+tenure baseline on Brier score in most
held-out seasons - see mri.coaches.hotseat for the method and the ship bar.
Writes data/hotseat_model.json either way, so a "didn't ship" run is still
inspectable rather than silently producing nothing.

Run:  PYTHONPATH=src python3 scripts/fit_hotseat.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.coaches import hotseat  # noqa: E402


def main() -> None:
    parquet_dir = ROOT / "data" / "parquet"
    coach_season = pd.read_parquet(parquet_dir / "coach_season.parquet")
    departures = pd.DataFrame(json.loads((ROOT / "data" / "coach_departures.json").read_text())["departures"])

    data = hotseat.dataset(coach_season, departures)
    print(f"{len(data)} training rows, {int(data['label'].sum())} positives, "
          f"{data['season'].min()}-{data['season'].max()}")
    print(f"features: {hotseat.full_columns()}")

    result = hotseat.evaluate(data)

    print(f"\nwalk-forward: {result['seasonsTotal']} held-out seasons "
          f"({result['trainSeasons'][0] + hotseat.MIN_TRAIN_SEASONS}-{result['trainSeasons'][1]})")
    for s in result["perSeason"]:
        mark = "full better" if s["fullBetter"] else "baseline better"
        print(f"  {s['season']}  n={s['n']:>3}  positives={s['positives']:>2}  "
              f"brier full={s['brierFull']:.4f} baseline={s['brierBaseline']:.4f}  ({mark})")

    print(f"\nfull model beat the baseline in {result['seasonsFullBetter']}/{result['seasonsTotal']} held-out seasons")
    print(f"AUC   full={result['aucFull']:.3f}  baseline={result['aucBaseline']:.3f}")
    print(f"Brier full={result['brierFull']:.4f}  baseline={result['brierBaseline']:.4f}")
    print("\ncalibration (predicted vs. actual fired rate, by decile):")
    for row in result["calibration"]:
        print(f"  n={row['n']:>3}  predicted={row['predicted']:.3f}  actual={row['actual']:.3f}")

    verdict = "SHIPS" if result["passed"] else "does NOT ship"
    print(f"\nhot-seat model {verdict} (ship bar: beat the baseline on Brier in most held-out seasons)")

    out = {"generated": dt.date.today().isoformat(), **result}
    out_path = ROOT / "data" / "hotseat_model.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
