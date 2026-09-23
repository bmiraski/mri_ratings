"""One tick of the live-score job: update docs/live.json while the slate's games are on.

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
    print(live.run(ROOT / "docs"))
