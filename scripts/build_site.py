"""Refresh the data and regenerate the site.

One entry point for the weekly run: pull whatever games are new, recompute both
ratings week by week, and write the static site.

Output goes to docs/, which is committed to the repo. GitHub Pages serves that
folder directly, so the host never needs to run a build or hold the API key -
whatever produces the files (this script, or the weekly Action) is the only
thing that needs credentials.

Run:  PYTHONPATH=src python3 scripts/build_site.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.export import logos, site, sitedata  # noqa: E402

SEASON = 2026


def main() -> None:
    data_dir = ROOT / "site" / "data"
    public = ROOT / "docs"

    print(f"building {SEASON}...")
    payload = sitedata.build_full(SEASON, data_dir)
    print(f"  week {payload['week']}, {payload['gamesRated']} games, {len(payload['teams'])} teams")

    summary = logos.cache_logos(payload, public)
    print(f"  logos: {summary['fetched']} fetched, {summary['cached']} cached, "
          f"{summary['failed']} failed")

    files = site.build(payload, public)
    print(f"  wrote {len(files)} files to {public.relative_to(ROOT)}")

    top = payload["teams"][0]
    print(f"  #1 {top['team']} ({top['power']:+.1f})")


if __name__ == "__main__":
    main()
