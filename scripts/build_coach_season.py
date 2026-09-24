"""Build the coach_season table: every FBS head coach, every school, every season.

Pulls CFBD's ``/coaches`` endpoint (one ranged call covers the whole span),
assigns a stable ``coach_id`` (``mri.coaches.ids``), and validates that a
split school-season's coaches' game counts really do add up to a played
schedule (``mri.coaches.season``). Prints any id collisions or split-season
problems so they can be hand-fixed in ``data/coach_aliases.json`` rather than
silently written into the table.

Run:  PYTHONPATH=src python3 scripts/build_coach_season.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.coaches import season  # noqa: E402

MIN_YEAR = 2003
MAX_YEAR = 2026


def main() -> None:
    print(f"pulling CFBD coaches, {MIN_YEAR}-{MAX_YEAR}...")
    result = season.build(MIN_YEAR, MAX_YEAR)
    table, collisions, problems = result["table"], result["collisions"], result["problems"]

    print(f"  {len(table)} coach-seasons, {table['coach_id'].nunique()} coaches, "
          f"{table['school'].nunique()} schools")
    print(f"  {int(table['interim'].sum())} flagged interim")

    if not collisions.empty:
        print(f"\n  {len(collisions)} coach_id collision(s) - add a canonical spelling to "
              f"data/coach_aliases.json for each real second person:")
        for _, row in collisions.iterrows():
            print(f"    {row['coach_id']}: {row['names']} (keys {row['coach_keys']})")
    else:
        print("  no coach_id collisions")

    if problems:
        print(f"\n  {len(problems)} split-season(s) whose game counts did not validate:")
        for p in problems:
            print(f"    {p['school']} {p['season']}: {p['error']}")
    else:
        print("  every split season's game counts validated against the schedule")

    out_dir = ROOT / "data" / "parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "coach_season.parquet"
    table.to_parquet(out_path, index=False)
    print(f"\nwrote {len(table)} rows to {out_path.relative_to(ROOT)}")

    check_team_names(table, out_dir)


def check_team_names(table, parquet_dir: Path) -> None:
    """Flag any (school, season) that coach_ratings_history.parquet has no row for.

    A coach_season row that can't be matched to a rating by (school, season)
    means a future phase's join silently drops that coach-season - worth
    surfacing now rather than discovering it as a missing metric later.
    """
    import pandas as pd

    history_path = parquet_dir / "coach_ratings_history.parquet"
    if not history_path.exists():
        print("\n  coach_ratings_history.parquet not found - run build_coach_ratings_history.py "
              "first to check team-name coverage")
        return
    history = pd.read_parquet(history_path)
    known = set(zip(history["team"], history["season"]))
    missing = sorted(set(zip(table["school"], table["season"])) - known)
    if not missing:
        print("  every coach_season (school, season) has a matching rating")
        return
    print(f"\n  {len(missing)} (school, season) pair(s) with no matching row in "
          f"coach_ratings_history.parquet - likely a name mismatch between CFBD's /coaches "
          f"and the ratings pipeline:")
    for school, sn in missing[:20]:
        print(f"    {school!r} {sn}")


if __name__ == "__main__":
    main()
