"""Build the historical feature set the at-large model is fit and graded on.

For each season 2011-2025 (2020 cancelled): every D1 team's résumé as of
Selection Sunday - power, résumé, quadrant record, schedule strength - and the
real field it was measured against (who got in, as what kind of bid, at what
seed). Nothing here is Monte Carlo; it's the same rating fit Phase 1 uses, run
without the pre-tournament cutoff, since Selection Sunday comes after the
conference tournaments and should see them.

Run:  PYTHONPATH=src python3 scripts/build_atlarge_history.py
Resumable; season computations cache to /tmp so an interruption costs nothing.
Writes data/parquet/atlarge_history.parquet.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.bracket import history, resume  # noqa: E402
from mri.ingest import bb_registry, cbbd  # noqa: E402

SEASONS = [y for y in range(2011, 2026) if y != 2020]
CACHE = Path("/tmp/atlarge_history_cache.pkl")
BUDGET = 260


def build_one(season: int) -> pd.DataFrame:
    games = cbbd.games(season)
    ratings = history.team_ratings(season, games=games, with_resume=True)
    ss_games = resume.selection_sunday_games(season, games)
    conf_of = {t: bb_registry.conference_of(t, season=season) for t in ratings.power.index}
    champs = history.auto_bid_winners(season)
    feats = resume.team_features(season, ratings, ss_games, conf_of, champs)

    field = history.field(season)
    if field.empty:
        feats["seed"], feats["bidType"], feats["region"] = None, None, None
        return feats
    field = field.set_index("team")
    feats["seed"] = feats["team"].map(field["seed"])
    feats["bidType"] = feats["team"].map(field["bidType"])
    feats["region"] = feats["team"].map(field["region"])
    return feats


def main() -> None:
    done: dict[int, pd.DataFrame] = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else {}
    start = time.time()
    for season in SEASONS:
        if season in done:
            continue
        if time.time() - start > BUDGET:
            print("budget reached; rerun to continue")
            CACHE.write_bytes(pickle.dumps(done))
            return
        print(f"season {season}...", flush=True)
        done[season] = build_one(season)
        CACHE.write_bytes(pickle.dumps(done))

    table = pd.concat(done.values(), ignore_index=True)
    out = ROOT / "data" / "parquet" / "atlarge_history.parquet"
    table.to_parquet(out)
    fielded = table[table["seed"].notna()]
    print(f"{len(table):,} team-seasons, {len(fielded)} in an actual field, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
