"""Audit the schedule graph of every archived season, sliced through the season.

Writes ``notes/connectivity_audit.md``. For each season it asks, at many points
through the season, whether every rated team (FBS for football, D1 for
basketball) is in the main component of the rated-only game graph and whether
any team on that season's roster has zero games. That answers two questions:

* was a *finished* season ever disconnected, and who was stranded;
* how far into a season does the graph become, and *stay*, connected - which is
  what decides when a production build may fail loudly rather than warn.

Football slices: end of each week where the table has weeks (2020+ and the
pre-2003 CFBD seasons), otherwise every 1% of the season's rows - the 2003-2019
workbooks carry neither weeks nor dates. Basketball slices: end of each day.

Run:  PYTHONPATH=src python3 scripts/audit_connectivity.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.ingest import bb_registry, registry  # noqa: E402
from mri.ratings import connectivity  # noqa: E402

warnings.simplefilter("ignore")
PARQUET = ROOT / "data" / "parquet"
OUT = ROOT / "notes" / "connectivity_audit.md"
COMMENTARY = ROOT / "notes" / "connectivity_audit_commentary.md"
FRACTIONS = (0.05, 0.10, 0.15, 0.25, 0.40)
BB_CHECKPOINTS = {"Nov 15": (11, 15), "Nov 30": (11, 30), "Dec 15": (12, 15), "Dec 31": (12, 31)}


def canonical(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("team1", "team2"):
        frame[column] = [registry.resolve(n, n) for n in frame[column]]
    return frame


def football_seasons():
    """(label, season, games, rated, roster, boundaries) for every football season, oldest first."""
    pre = pd.read_parquet(PARQUET / "pre2003_games.parquet")
    arc = pd.read_parquet(PARQUET / "archive_games.parquet")
    cur = pd.read_parquet(PARQUET / "current_games.parquet")
    cur = cur[cur["played"]] if "played" in cur else cur
    sources = [("CFBD", pre), ("workbook", arc), ("CFBD", cur)]
    for source, frame in sources:
        for season, games in frame.groupby("season"):
            games = canonical(games).reset_index(drop=True)
            if source == "CFBD":
                games = games.sort_values(["week", "start_date"], kind="stable").reset_index(drop=True)
            members = registry.fbs_members(int(season))
            names = set(games["team1"]) | set(games["team2"])
            if members:
                rated = sorted(n for n in names if n in members)
                roster = sorted(members)
            else:  # no roster cached: fall back to the current registry
                rated = sorted(n for n in names if registry.was_fbs(n, int(season)))
                roster = rated
            n = len(games)
            ends = sorted({max(1, round(n * p / 100)) for p in range(1, 101)})
            if "week" in games.columns:
                weeks = games["week"].to_numpy()
                labels = [f"{round(100 * e / n)}% (wk{weeks[e - 1]})" for e in ends]
            else:
                labels = [f"{round(100 * e / n)}%" for e in ends]
            checkpoints = {f"{int(f * 100)}%": max(1, round(n * f)) for f in FRACTIONS}
            yield source, int(season), games, rated, roster, ends, labels, checkpoints


def basketball_seasons():
    classic = pd.read_parquet(PARQUET / "bb_classic_games.parquet")
    for season, games in classic.groupby("season"):
        roster_map = bb_registry.teams(int(season))
        games = games.copy()
        for column in ("team1", "team2"):
            games[column] = [bb_registry.resolve(n, n, season=int(season)) for n in games[column]]
        games = games.sort_values("start_date", kind="stable").reset_index(drop=True)
        names = set(games["team1"]) | set(games["team2"])
        rated = sorted(n for n in names if n in roster_map)
        day = pd.to_datetime(games["start_date"], utc=True).dt.strftime("%m-%d").to_numpy()
        # season runs Nov..Apr; sort key keeps Nov/Dec before Jan.
        key = pd.to_datetime(games["start_date"], utc=True).dt.strftime("%Y-%m-%d").to_numpy()
        ends = [int(np.flatnonzero(key <= d).max()) + 1 for d in sorted(set(key))]
        labels = sorted(set(key))
        checkpoints = {
            name: int(np.searchsorted(key, f"{season - 1 if month >= 11 else season}-{month:02d}-{day:02d}", side="right"))
            for name, (month, day) in BB_CHECKPOINTS.items()
        }
        yield "Classic/API", int(season), games, rated, sorted(roster_map), ends, labels, checkpoints


def scan(games, rated, roster, ends, labels):
    """Connectivity at the end of each slice; returns list of (label, rows, Connectivity)."""
    out = []
    for end, label in zip(ends, labels):
        out.append((label, end, connectivity.analyze(games.iloc[:end], rated=rated, expected=roster)))
    return out


def first_stable(results, ok=lambda c: c.ok):
    """First slice after which every later slice passes ``ok``, or None if the end does not."""
    if not results or not ok(results[-1][2]):
        return None
    first = len(results) - 1
    while first > 0 and ok(results[first - 1][2]):
        first -= 1
    return results[first][0], results[first][1]


def audit(seasons_iter, sport):
    rows, detail = [], []
    for source, season, games, rated, roster, ends, labels, checkpoints in seasons_iter:
        res = scan(games, rated, roster, ends, labels)
        final = res[-1][2]
        stable = first_stable(res)
        # Orphans only: a roster team with no games at all (a program not yet D1, a league that
        # cancelled its season) is a different question from a stranded one.
        stable_strict = first_stable(res, ok=lambda c: c.strict.connected)
        late = ""
        if stable_strict and stable_strict[1] > res[0][1]:
            before = next(c for lab, e, c in reversed(res) if e < stable_strict[1])
            late = ", ".join(before.strict.orphans["team"].head(3))
        row = {
            "season": season, "source": source, "games": len(games), "rated": len(rated),
            "loose_components_end": final.loose.components,
            "strict_orphans_end": 0 if final.strict is None else len(final.strict.orphans),
            "zero_game_end": len(final.zero_game),
            "stable_from": stable[0] if stable else "never",
            "orphan_free_from": stable_strict[0] if stable_strict else "never",
            "orphan_free_share": round(stable_strict[1] / len(games), 3) if stable_strict else None,
            "last_orphan_before": late,
        }
        for name, rows_in in checkpoints.items():
            usable = [c for _, e, c in res if e <= rows_in]
            c = usable[-1] if usable else res[0][2]
            row[f"orph@{name}"] = 0 if c.strict is None else len(c.strict.orphans)
            row[f"zero@{name}"] = len(c.zero_game)
        rows.append(row)
        print(f"{sport} {season}: {row['strict_orphans_end']} orphans, {row['zero_game_end']} zero-game at end; "
              f"orphan-free from {row['orphan_free_from']}")
        if final.strict is not None and (not final.strict.orphans.empty or final.zero_game):
            detail.append((season, final))
    return pd.DataFrame(rows), detail


def md_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in frame.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(lines)


def detail_block(detail) -> str:
    if not detail:
        return "_None. Every finished season is fully connected with no zero-game teams._\n"
    out = []
    for season, conn in detail:
        out.append(f"**{season}**")
        if conn.strict is not None and not conn.strict.orphans.empty:
            out.append("\n" + md_table(conn.strict.orphans) + "\n")
        if conn.zero_game:
            out.append(f"\nZero games: {', '.join(conn.zero_game)}\n")
    return "\n".join(out)


def main() -> None:
    fb, fb_detail = audit(football_seasons(), "football")
    bb, bb_detail = audit(basketball_seasons(), "basketball")
    OUT.parent.mkdir(exist_ok=True)
    body = [
        "# Connectivity audit\n",
        "_Tables generated by `scripts/audit_connectivity.py`; the reading below them is "
        "`notes/connectivity_audit_commentary.md`, written by hand and spliced in._\n",
        COMMENTARY.read_text() if COMMENTARY.exists() else "",
        "## Football\n",
        "Strict view = FBS teams only, games between two FBS teams. Roster = that season's CFBD FBS list "
        "(a rostered team with no games is *zero-game*). `stable_from` is the first slice after which every "
        "later slice is fully connected with no zero-game teams.\n",
        md_table(fb), "\n### Finished seasons with orphans or zero-game teams\n", detail_block(fb_detail),
        "\n## Basketball\n",
        "Strict view = D1 teams of that season's roster, games between two D1 teams. Slices are daily. "
        "Seasons 2005-2020 come from `bb_classic_games.parquet`, 2021-2026 from the same file (identical to "
        "`bb_games.parquet` there).\n",
        md_table(bb), "\n### Finished seasons with orphans or zero-game teams\n", detail_block(bb_detail),
    ]
    OUT.write_text("\n".join(body))
    fb.to_csv(ROOT / "data" / "connectivity_audit_football.csv", index=False)
    bb.to_csv(ROOT / "data" / "connectivity_audit_basketball.csv", index=False)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
