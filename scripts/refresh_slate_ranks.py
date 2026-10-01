"""Give every reconstructed slate archive the ranks its teams had at the time.

The historical slate pages (``docs/slate/<season>-week-<n>.json``, 1978 on) were written with each team's
*current* rank beside it, so a 1999 matchup listed Georgia at today's #1. ``hfa.walk_forward`` already
refits the ratings before every week to price that week's games; this reads the Power rank out of those
same fits and rewrites the ``teams`` of each reconstructed archive with it.

Touches nothing else in the files: the frozen pregame probabilities the chaos history scores against stay as
they are, and weeks saved live (2026) already hold the right ranks. Idempotent. The pages themselves are
re-rendered from this JSON by the next ``build_site.py``.

Run:  PYTHONPATH=src python3 scripts/refresh_slate_ranks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.export import slatearchive  # noqa: E402
from mri.ingest import cfbd  # noqa: E402
from mri.ratings import hfa  # noqa: E402

LAST_SEASON = cfbd.current_season() - 1  # the backfill covers completed seasons only
SEASONS = range(1978, LAST_SEASON + 1)


def main() -> None:
    site = ROOT / "site" / "data" / "site.json"
    teams_payload = json.loads(site.read_text())["teams"]
    ranks: dict = {}
    print(f"walking forward {SEASONS.start}-{SEASONS.stop - 1} for point-in-time ranks...")
    hfa.walk_forward(SEASONS, ranks=ranks)
    changed = slatearchive.refresh_historical_ranks(ROOT / "docs", ranks, teams_payload)
    print(f"rewrote the teams of {changed} historical slate archive(s)")


if __name__ == "__main__":
    main()
