"""Build a bracketology projection by hand: site/data/bracketology.json, or anywhere with --out.

The daily site build does this itself (``mri.export.bracketdata``) inside its calendar window;
this script is for running it outside that - and for replaying any past date, which is how the
pages were designed before a single live game of the 76-team era had been played.

Run:  PYTHONPATH=src python3 scripts/build_bracketology.py
      [--season 2025 --as-of 2025-02-01 --out /tmp/x.json --force]   (replay any past date)
      [--field-size 76]                                                (...under the 2027 format)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.bracket import seeding  # noqa: E402
from mri.bracket.live import build, load_settings, season_for, selection_sunday, start_date  # noqa: E402

OUT = ROOT / "site" / "data" / "bracketology.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int)
    parser.add_argument("--as-of", help="replay a past date (YYYY-MM-DD); omit for live")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--sims", type=int)
    parser.add_argument("--force", action="store_true", help="ignore the Christmas / Selection Sunday gates")
    parser.add_argument("--field-size", type=int, choices=sorted(seeding.FORMATS),
                        help="replay a season under another format (e.g. 2025 as a 76-team field)")
    args = parser.parse_args()

    settings = load_settings()
    today = dt.date.fromisoformat(args.as_of) if args.as_of else dt.date.today()
    season = args.season or season_for(today)
    if not args.force:
        if today < start_date(season, settings):
            print(f"bracketology: not before {start_date(season, settings)}; nothing written")
            return
        if today > selection_sunday(season, settings) and args.out.exists():
            print("bracketology: frozen after Selection Sunday; keeping the last projection")
            return
    data = build(season, args.as_of, settings, args.sims or settings.get("sims", 5000), args.field_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=1) + "\n")
    in_field = [t for t in data["teams"] if t["pField"] >= 0.5]
    print(f"bracketology {season} as of {data['asOf']}: {len(data['teams'])} teams with a chance, "
          f"{len(in_field)} at 50%+; rule problems: {len(data['projected']['ruleProblems'])}")


if __name__ == "__main__":
    main()
