"""Export each basketball season as an Excel workbook in the shape of the
originals. See ``scripts/export_workbooks.py`` for the football version this
mirrors - ``recalculate()`` below is that script's, verbatim, because the
reason for recalculating (openpyxl saves formulas with no cached values) is
the same for both sports.

Basketball has no archive gap to fill: bb_ratings.parquet already covers
2021-2026 (the 2020-21 through 2025-26 seasons) end to end. This just gives
each of those seasons the same live-formula workbook football's seasons get.

Run:  PYTHONPATH=src python3 scripts/export_bb_workbooks.py [first] [last]
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.export import bb_workbook  # noqa: E402
from mri.ingest import bb_registry as registry  # noqa: E402
from mri.ingest import cbbd  # noqa: E402

DEFAULT_FIRST, DEFAULT_LAST = 2021, 2026
OUT_DIR = ROOT / "exports"
RECALC = Path("/root/.claude/skills/synced")


def modern_ratings(season: int) -> pd.DataFrame:
    """MRI 2.0 for a season, from the chain build_basketball.py produced."""
    path = ROOT / "data" / "parquet" / "bb_ratings.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    chunk = frame[frame["season"] == season]
    if chunk.empty:
        return pd.DataFrame()
    # The chain rates every team it saw, D1 and not; the workbook only ranks
    # the D1 field, so rank has to be recomputed after the non-D1 rows drop
    # out rather than reused as-is.
    chunk = chunk[chunk["team"].map(lambda t: registry.is_d1(t, season=season))].copy()
    chunk = chunk.sort_values("rank")
    chunk["rank"] = range(1, len(chunk) + 1)
    return chunk


def recalculate(path: Path) -> dict:
    """Compute every formula so the file is readable outside Excel."""
    script = next(RECALC.glob("*/xlsx/scripts/recalc.py"), None)
    if script is None:
        return {"status": "skipped", "reason": "recalc.py not found"}
    result = subprocess.run(
        [sys.executable, str(script), str(path), "120"],
        capture_output=True, text=True, timeout=300,
    )
    try:
        # The report is pretty-printed JSON across many lines, so parse the
        # whole of stdout rather than its last line.
        return json.loads(result.stdout.strip())
    except (ValueError, IndexError):
        return {"status": "unreadable", "stdout": result.stdout[-300:],
                "stderr": result.stderr[-300:]}


def main() -> None:
    first = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FIRST
    last = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_LAST

    OUT_DIR.mkdir(exist_ok=True)
    failures = []

    for season in range(first, last + 1):
        games = cbbd.classic_table(season)
        if games.empty:
            print(f"  {season}: no games")
            continue

        path = OUT_DIR / f"MRIBasketball{season}.xlsx"
        bb_workbook.write_season(season, games, modern_ratings(season), path)

        report = recalculate(path)
        status = report.get("status")
        size = path.stat().st_size / 1024
        detail = (f"{report.get('total_formulas', '?')} formulas, "
                  f"{report.get('total_errors', '?')} errors")
        print(f"  {season}: {len(games):>4} games -> {path.name} ({size:.0f} KB) [{status}: {detail}]")

        if status == "errors_found":
            failures.append((season, report.get("error_summary")))

    if failures:
        print("\nFORMULA ERRORS - do not ship these:")
        for season, summary in failures:
            print(f"  {season}: {summary}")
        raise SystemExit(1)
    print(f"\nwrote {last - first + 1} workbooks to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
