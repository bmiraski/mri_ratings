"""One tick of the live-score job: update each sport's live.json while its slate's games are on.

Run every fifteen minutes by .github/workflows/live.yml. Spends an API call only
while a game is about to start or in progress; otherwise it exits having read
two local files. Standard library only, so the Action needs no installs.

Run:  PYTHONPATH=src python3 scripts/live_scores.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.export import live  # noqa: E402

if __name__ == "__main__":
    for sport, root in ((live.FOOTBALL, ROOT / "docs"), (live.BASKETBALL, ROOT / "docs" / "basketball")):
        try:
            print(f"{sport.name}: {live.run(root, sport=sport)}")
        except Exception as exc:  # noqa: BLE001 - one sport's failure must not cost the other its scores
            print(f"{sport.name}: failed - {exc}")
