"""Repairs to the 2003-2019 workbook archive, taken from the CFBD games cache.

Two defects in Ben's workbooks, both found by comparing them with CFBD:

*Missing postseason games.* The 2017 workbook stops at the end of the regular
season: all 40 bowl and playoff games are absent. Every other season matches
CFBD's postseason exactly (2016 included - the three games that first looked
missing were an FCS playoff game and two orientation mismatches). The
detection below is generic, so a future workbook with the same gap is caught
the same way.

*Neutral sites in 2004 and 2005.* The workbooks have no site column, so
``mri2.mark_postseason`` infers postseason games as "both teams have already
played twelve". Teams played eleven in those years, so it flags 2 games in each
season against 28 bowls, and every bowl is rated as a home game for the
second-listed team. CFBD carries the real flag.

Both fixes need only scores, which is why they come from the games cache and
work offline. The first also needs box scores for MRI Classic (rushing and
passing yards, turnovers), which are one API call per week; ``missing_postseason``
attaches them when they are cached or fetchable and leaves them NaN otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..ratings import classic, mri2
from . import cfbd, registry

# Seasons whose neutral flags come from CFBD. Elsewhere the rule works (it agrees
# with CFBD's postseason flag to within a handful of games a year) and changing it
# would move numbers that were tuned and validated with it.
NEUTRAL_FROM_CFBD = (2004, 2005)

STAT_COLUMNS = ["rush1", "rush2", "pass1", "pass2", "to1", "to2"]

CORRECTIONS_PATH = Path(__file__).resolve().parents[3] / "data" / "archive_score_corrections.json"


def apply_score_corrections(season: int, games: pd.DataFrame, path: Path = CORRECTIONS_PATH) -> pd.DataFrame:
    """Fix workbook scores that were checked by hand against another source.

    Each entry names the game, the score the workbook holds (``old``) and the right one (``new``). An entry
    that does not match exactly one row raises, so a stale or mistyped correction cannot quietly do nothing.
    ``games`` is the raw workbook table (visitor in ``team1``); win flags are untouched, which is right
    because no correction changes a winner (asserted).
    """
    if not path.exists():
        return games
    games = games.copy()
    key1 = games["team1"].map(classic._match_key)
    key2 = games["team2"].map(classic._match_key)
    for entry in json.loads(path.read_text())["corrections"]:
        if entry["season"] != season:
            continue
        hit = (
            (key1 == classic._match_key(entry["team1"])) & (key2 == classic._match_key(entry["team2"]))
            & (games["pts1"] == entry["old"][0]) & (games["pts2"] == entry["old"][1])
        )
        if hit.sum() != 1:
            raise ValueError(f"score correction matches {int(hit.sum())} rows, expected 1: {entry}")
        old_winner = entry["old"][0] > entry["old"][1]
        if old_winner != (entry["new"][0] > entry["new"][1]):
            raise ValueError(f"score correction changes the winner, which win1/win2 would not follow: {entry}")
        games.loc[hit, ["pts1", "pts2"]] = entry["new"]
    return games


def _canon(name: object) -> str:
    return registry.resolve(name, str(name))


def _key(team1: object, team2: object, pts1: float, pts2: float):
    """Unordered pair plus the sorted score: robust to home/away disagreements."""
    return (frozenset((_canon(team1), _canon(team2))), tuple(sorted((int(pts1), int(pts2)))))


def _completed(season: int) -> pd.DataFrame:
    games = cfbd.games(season)
    return games[games["played"]].reset_index(drop=True) if not games.empty else games


def neutral_from_cfbd(season: int, games: pd.DataFrame) -> pd.Series:
    """CFBD's neutral-site flag for each archive game, with the inferred rule where unmatched.

    Rows are matched on the unordered team pair and the score. Anything CFBD does
    not have (games against pooled "Non D1A" opponents, mostly) falls back to
    ``mri2.mark_postseason``, so the result is defined for every row.
    """
    games = games.reset_index(drop=True)
    flags = mri2.mark_postseason(games).to_numpy(bool).copy()
    cf = _completed(season)
    lookup: dict = {}
    for row in cf.itertuples():
        lookup.setdefault(_key(row.team1, row.team2, row.pts1, row.pts2), bool(row.neutral))
    for i, row in enumerate(games.itertuples()):
        hit = lookup.get(_key(row.team1, row.team2, row.pts1, row.pts2))
        if hit is not None:
            flags[i] = hit
    return pd.Series(flags, index=games.index, name="neutral")


def match_rate(season: int, games: pd.DataFrame) -> float:
    cf = _completed(season)
    have = {_key(r.team1, r.team2, r.pts1, r.pts2) for r in cf.itertuples()}
    keys = [_key(r.team1, r.team2, r.pts1, r.pts2) for r in games.itertuples()]
    return float(np.mean([k in have for k in keys]))


def _archive_spelling(games: pd.DataFrame, teams: list[str] | None) -> dict[str, str]:
    """Registry spelling -> the workbook's own spelling for every team it names."""
    names = set(games["team1"]) | set(games["team2"]) | set(teams or [])
    return {_canon(n): n for n in names}


def missing_postseason(
    season: int, games: pd.DataFrame, teams: list[str] | None = None, *, with_box_scores: bool = True,
    known: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Postseason games CFBD has and the workbook lacks, in the workbook's own columns.

    A game counts when at least one side was FBS that season and no workbook game
    has the same pair and score. Names are returned in the workbook's spelling so
    MRI Classic's roster lookups still hit; a non-FBS side is pooled to "Non D1A".
    Returns an empty frame when nothing is missing. Raises if a team cannot be
    placed, because a silently dropped game is the failure this exists to prevent.
    """
    cf = _completed(season)
    if cf.empty:
        return pd.DataFrame()
    post = cf[cf["season_type"] == "postseason"]
    members = registry.fbs_members(season)
    have = {_key(r.team1, r.team2, r.pts1, r.pts2) for r in games.itertuples()}

    spell = _archive_spelling(games, teams)
    rows = []
    for g in post.sort_values("start_date", kind="stable").itertuples():
        away, home = _canon(g.team1), _canon(g.team2)
        if away not in members and home not in members:
            continue  # FCS playoff
        if _key(g.team1, g.team2, g.pts1, g.pts2) in have:
            continue
        sides = []
        for name in (away, home):
            if name not in members:
                sides.append(registry.POOLED_FCS)
            elif name in spell:
                sides.append(spell[name])
            else:
                raise ValueError(f"{season} postseason: cannot place {name!r} in the workbook roster")
        rows.append(
            {
                "team1": sides[0], "team2": sides[1], "pts1": g.pts1, "pts2": g.pts2,
                "win1": g.win1, "win2": g.win2,
                "date": pd.Timestamp(g.start_date).date(), "season": season,
                "neutral": bool(g.neutral), "game_id": g.game_id,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for column in STAT_COLUMNS:
        out[column] = np.nan
    if with_box_scores:
        out = _attach_box_scores(season, out)
    if known is not None and not has_stats(out):
        out = _reuse_known_stats(season, out, known)
    return out


def _reuse_known_stats(season: int, extra: pd.DataFrame, known: pd.DataFrame) -> pd.DataFrame:
    """Fill missing box-score stats from a previously built archive table.

    Lets the archive be rebuilt offline: box scores are fetched once (they need an API key) and live
    in ``archive_games.parquet`` afterwards, so a rebuild without a key keeps them.
    """
    prior = known[(known["season"] == season) & known[STAT_COLUMNS].notna().all(axis=1)]
    if prior.empty:
        return extra
    lookup = {_key(r.team1, r.team2, r.pts1, r.pts2): r for r in prior.itertuples()}
    extra = extra.copy()
    filled = 0
    for i, row in extra.iterrows():
        hit = lookup.get(_key(row["team1"], row["team2"], row["pts1"], row["pts2"]))
        if hit is not None and pd.isna(row[STAT_COLUMNS]).all():
            for column in STAT_COLUMNS:
                extra.at[i, column] = getattr(hit, column)
            filled += 1
    if filled:
        print(f"  {season}: reused box scores for {filled} games from the existing archive_games.parquet")
    return extra


def _attach_box_scores(season: int, extra: pd.DataFrame) -> pd.DataFrame:
    """Fill rushing, passing and turnovers from CFBD's postseason box scores, if obtainable."""
    try:
        box = cfbd.team_box_scores(season, 1, season_type="postseason")
    except cfbd.CfbdError as exc:
        print(f"  note: no box scores for the {season} postseason ({exc}); Classic stats left empty")
        return extra
    except Exception as exc:  # noqa: BLE001 - network or quota; the scores-only rows are still useful
        print(f"  note: could not fetch {season} postseason box scores ({exc}); Classic stats left empty")
        return extra
    if box.empty:
        return extra
    away = box[box["home_away"] == "away"].set_index("game_id")
    home = box[box["home_away"] == "home"].set_index("game_id")
    extra = extra.copy()
    extra["rush1"] = extra["game_id"].map(away["rush"])
    extra["pass1"] = extra["game_id"].map(away["pass"])
    extra["to1"] = extra["game_id"].map(away["turnovers"])
    extra["rush2"] = extra["game_id"].map(home["rush"])
    extra["pass2"] = extra["game_id"].map(home["pass"])
    extra["to2"] = extra["game_id"].map(home["turnovers"])
    return extra


def has_stats(extra: pd.DataFrame) -> bool:
    return not extra.empty and bool(extra[STAT_COLUMNS].notna().all().all())
