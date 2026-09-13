"""Export each season as an Excel workbook in the shape of the originals.

Fills the 2020-2025 gap, where Ben's own run stopped and no workbook exists,
and can re-export the archived seasons for a consistent set.

The Classic sheets are live formulas; the MRI 2.0 sheet is a snapshot, for
reasons the workbook's About sheet explains. Every file is recalculated with
LibreOffice before it is written out, because openpyxl saves formulas with no
cached values - unrecalculated, every formula cell reads as blank to pandas,
previewers, and anything else that is not Excel.

Run:  PYTHONPATH=src python3 scripts/export_workbooks.py [first] [last]
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.export import workbook  # noqa: E402
from mri.ingest import boxscores, registry  # noqa: E402
from mri.ratings import mri2  # noqa: E402

DEFAULT_FIRST, DEFAULT_LAST = 2020, 2026
OUT_DIR = ROOT / "exports"
RECALC = Path("/root/.claude/skills/synced")


def modern_ratings(year: int) -> pd.DataFrame:
    """MRI 2.0 for a season, from the chain build_current.py produced."""
    path = ROOT / "data" / "parquet" / "current_ratings.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    season = frame[frame["season"] == year]
    if season.empty:
        return pd.DataFrame()
    season = season[season["team"].map(registry.is_fbs)].copy()
    season["rank"] = range(1, len(season) + 1)
    return season


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

    for year in range(first, last + 1):
        games = boxscores.classic_table(year)
        if games.empty:
            print(f"  {year}: no games")
            continue

        path = OUT_DIR / f"MRIFootball{year}.xlsx"
        workbook.write_season(year, games, modern_ratings(year), path)

        report = recalculate(path)
        status = report.get("status")
        size = path.stat().st_size / 1024
        detail = (f"{report.get('total_formulas', '?')} formulas, "
                  f"{report.get('total_errors', '?')} errors")
        print(f"  {year}: {len(games):>4} games -> {path.name} ({size:.0f} KB) [{status}: {detail}]")

        if status == "errors_found":
            failures.append((year, report.get("error_summary")))

    if failures:
        print("\nFORMULA ERRORS - do not ship these:")
        for year, summary in failures:
            print(f"  {year}: {summary}")
        raise SystemExit(1)
    print(f"\nwrote {last - first + 1} workbooks to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
