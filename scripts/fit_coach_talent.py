"""The plan's §3 gate: does talent explain power_end well enough to publish as vs_talent?

Leave-one-season-out cross-validated regression of power_end on this
season's and last season's talent composite, 2015+. Reports the verdict; see
mri.coaches.talent for the method and the gate's threshold.

Run:  PYTHONPATH=src python3 scripts/fit_coach_talent.py
Writes data/talent_model.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.coaches import talent  # noqa: E402


def main() -> None:
    ratings_path = ROOT / "data" / "parquet" / "coach_ratings_history.parquet"
    ratings_history = pd.read_parquet(ratings_path)
    ratings_history = ratings_history.drop(columns="vs_talent", errors="ignore")  # a prior run's column, if any

    print(f"fitting on {talent.START_YEAR}+ FBS team-seasons...")
    model = talent.fit(ratings_history)

    print(f"  {model['teamSeasons']} team-seasons, {model['seasons'][0]}-{model['seasons'][1]}")
    print(f"  coefficients: {model['coefficients']}")
    print(f"  out-of-sample R² {model['outOfSampleR2']:.3f} (gate: {model['gate']}), "
          f"RMSE {model['outOfSampleRmse']:.2f}")
    verdict = "PASSES" if model["passed"] else "does not pass"
    print(f"\n  talent {verdict} the gate.")

    out_path = ROOT / "data" / "talent_model.json"
    out_path.write_text(json.dumps(model, indent=2) + "\n")
    print(f"wrote {out_path.relative_to(ROOT)}")

    if model["passed"]:
        residual = talent.vs_talent(ratings_history, model).reset_index()
        merged = ratings_history.merge(residual, on=["team", "season"], how="left")
        merged.to_parquet(ratings_path, index=False)
        print(f"  vs_talent added for {residual['vs_talent'].notna().sum()} team-seasons, "
              f"written to {ratings_path.relative_to(ROOT)}")
    else:
        print("  gate failed - vs_talent NOT added; note this on the method page")


if __name__ == "__main__":
    main()
