"""MRI Classic - faithful Python ports of the original Excel formulas.

This module is deliberately not improved. It exists to reproduce the workbooks
exactly so that (a) the historical rankings stay comparable and (b) there is a
known-good baseline to measure MRI 2.0 against. Every quirk is preserved.

Both sports share a skeleton - win percentage, opponents, opponents' opponents,
summed game credit, then z-scored statistical components - but the details
differ enough that treating basketball as football with renamed columns would
be wrong in four places. The ``Sport`` config below holds the differences.

FOOTBALL, from Team Data column V:

    MRI = 25*Win% + 25*OppWin% + 10*OppOppWin% + sum(game credit)
        + 5*z(RushYds/G) + 5*z(PassYds/G)
        + 7*-z(TotalYdsAllowed/G) + 3*z(TurnoverMargin)

    win  ->  min( 35, margin) * OppWin%  * OppOppWin%
    loss ->  max(-35, margin) * OppLoss% * OppOppLoss%   (0.1 if opp unbeaten)

BASKETBALL, from the 2019-20 workbook:

    MRI = 25*Win% + 25*OppWin% + 10*OppOppWin% + sum(game credit)
        + 10*z(ReboundDiff/G) + 6*z(TurnoverDiff/G)

    win  ->  min( 30, margin) * OppWin%  * OppOppWin%
    loss ->  max(-30, margin) * OppLoss% * OppOppLoss%   (no fallback)

The differences that matter: the cap is 30 not 35; there is no undefeated-
opponent fallback; the turnover term is per game where football's is a raw
total; and strength of schedule uses an RPI-style adjustment that removes the
team's own games from its opponents' records, which the football sheet never
did. Football pools non-FBS opponents into one "Non D1A" row; basketball drops
non-D1 games entirely, which is why its workbooks carry no pooled team.

z-scores use the sample standard deviation (Excel STDEV) over rated teams only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from typing import Callable

MARGIN_CAP = 35.0
UNDEFEATED_OPPONENT_FALLBACK = 0.1

WEIGHTS = {
    "win_pct": 25.0,
    "opp_win_pct": 25.0,
    "opp_opp_win_pct": 10.0,
    "rush": 5.0,
    "pass": 5.0,
    "defense": 7.0,
    "turnovers": 3.0,
}

POOLED_FCS = "Non D1A"


@dataclass(frozen=True)
class Sport:
    """What differs between the football and basketball workbooks."""

    name: str
    margin_cap: float
    undefeated_fallback: float | None
    # long-form column -> (team1 source, team2 source); the reverse pairing is
    # generated automatically for the opponent's side.
    stat_pairs: dict[str, tuple[str, str]]
    # Adds the z-scored component columns to the aggregated frame.
    derive: Callable[["pd.DataFrame", "pd.Series"], None]
    # (column, weight, sign) - sign -1 for "lower is better".
    components: tuple[tuple[str, float, int], ...]
    # Strength of schedule, which the two sports compute differently.
    sos: Callable[["pd.DataFrame"], "pd.Series"]


def _derive_football(records: pd.DataFrame, played: pd.Series) -> None:
    records["rush_per_game"] = _safe_pct(records["rush_for"], played)
    records["pass_per_game"] = _safe_pct(records["pass_for"], played)
    records["yards_allowed_per_game"] = _safe_pct(
        records["rush_against"] + records["pass_against"], played
    )
    records["turnover_margin"] = records["to_forced"] - records["to_committed"]


def _derive_basketball(records: pd.DataFrame, played: pd.Series) -> None:
    records["rebound_diff_per_game"] = _safe_pct(
        records["reb_for"] - records["reb_against"], played
    )
    records["turnover_diff_per_game"] = _safe_pct(
        records["to_forced"] - records["to_committed"], played
    )


def _sos_football(records: pd.DataFrame) -> pd.Series:
    """Opponents' win percentage times their opponents'. Unadjusted."""
    return records["opp_win_pct"] * records["opp_opp_win_pct"]


def _sos_basketball(records: pd.DataFrame) -> pd.Series:
    """RPI-style, with the team's own games removed from its opponents' records.

    From Team Data columns W, X and Y of the basketball workbooks. Without the
    adjustment a team inflates its own strength of schedule by beating people:
    every win it records shows up again as an opponent win.
    """
    own_games = records["wins"] + records["losses"]
    opp_numerator = records["opp_wins"] - records["losses"]
    opp_denominator = records["opp_wins"] + records["opp_losses"] - own_games
    opp = _safe_pct(opp_numerator, opp_denominator)

    opp2_numerator = records["opp_opp_wins"] - own_games * records["wins"]
    opp2_denominator = records["opp_opp_wins"] + records["opp_opp_losses"] - own_games ** 2
    opp2 = _safe_pct(opp2_numerator, opp2_denominator)
    return pd.Series(opp * opp2, index=records.index)


FOOTBALL = Sport(
    name="football",
    margin_cap=35.0,
    undefeated_fallback=0.1,
    stat_pairs={
        "rush_for": ("rush1", "rush2"),
        "rush_against": ("rush2", "rush1"),
        "pass_for": ("pass1", "pass2"),
        "pass_against": ("pass2", "pass1"),
        "to_committed": ("to1", "to2"),
        "to_forced": ("to2", "to1"),
    },
    derive=_derive_football,
    components=(
        ("rush_per_game", 5.0, 1),
        ("pass_per_game", 5.0, 1),
        ("yards_allowed_per_game", 7.0, -1),
        ("turnover_margin", 3.0, 1),
    ),
    sos=_sos_football,
)

BASKETBALL = Sport(
    name="basketball",
    margin_cap=30.0,
    undefeated_fallback=None,
    stat_pairs={
        "reb_for": ("reb1", "reb2"),
        "reb_against": ("reb2", "reb1"),
        "to_committed": ("to1", "to2"),
        "to_forced": ("to2", "to1"),
    },
    derive=_derive_basketball,
    components=(
        ("rebound_diff_per_game", 10.0, 1),
        ("turnover_diff_per_game", 6.0, 1),
    ),
    sos=_sos_basketball,
)


def _safe_pct(numerator: pd.Series | np.ndarray, denominator: pd.Series | np.ndarray):
    """Excel returns a division error on 0-0 teams; we return 0 and move on."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.zeros_like(numerator, dtype=float)
    nonzero = denominator != 0
    out[nonzero] = numerator[nonzero] / denominator[nonzero]
    return out


def _match_key(name: object) -> str:
    """Excel's SUMIF and VLOOKUP match case-insensitively and ignore runs of
    whitespace. The archive relies on that: 2009 spells Boise State two ways
    ("Boise State" and "Boise STate") and the workbook still totals them as one
    team. A case-sensitive groupby would silently split them, so every lookup
    goes through this key.
    """
    return " ".join(str(name).split()).casefold()


def canonicalize(games: pd.DataFrame, teams: list[str] | None) -> tuple[pd.DataFrame, list[str] | None]:
    """Collapse spelling variants onto one canonical name per team.

    The canonical spelling is the one from the Team Data roster when the team
    appears there, otherwise the first spelling seen in the game log.
    """
    canonical: dict[str, str] = {}
    for name in teams or []:
        canonical.setdefault(_match_key(name), " ".join(str(name).split()))
    for name in pd.concat([games["team1"], games["team2"]], ignore_index=True):
        canonical.setdefault(_match_key(name), " ".join(str(name).split()))

    games = games.copy()
    for column in ("team1", "team2"):
        games[column] = games[column].map(lambda n: canonical[_match_key(n)])

    if teams is not None:
        seen: dict[str, None] = {}
        for name in teams:
            seen.setdefault(canonical[_match_key(name)], None)
        teams = list(seen)
    return games, teams


def _long_form(games: pd.DataFrame, sport: Sport) -> pd.DataFrame:
    """One row per team per game, so aggregation is a single groupby."""
    home = {
        "team": games["team1"], "opponent": games["team2"],
        "points_for": games["pts1"], "points_against": games["pts2"],
        "won": games["win1"],
    }
    away = {
        "team": games["team2"], "opponent": games["team1"],
        "points_for": games["pts2"], "points_against": games["pts1"],
        "won": games["win2"],
    }
    for column, (first, second) in sport.stat_pairs.items():
        home[column] = games[first]
        away[column] = games[second]

    long = pd.concat(
        [pd.DataFrame(home), pd.DataFrame(away)], ignore_index=True
    )
    long["lost"] = 1.0 - long["won"]
    long["margin"] = long["points_for"] - long["points_against"]
    return long


def compute(
    games: pd.DataFrame,
    teams: list[str] | None = None,
    sport: Sport = FOOTBALL,
) -> pd.DataFrame:
    """Compute MRI Classic for every team in ``games``.

    Parameters
    ----------
    games
        Normalized game table with the archive's column names.
    teams
        The rated team list. Teams outside it (football's pooled FCS row) still
        contribute to opponent records but are excluded from the z-score
        statistics and from the returned rankings.
    sport
        ``FOOTBALL`` or ``BASKETBALL``. The two formulas share a skeleton and
        differ in the margin cap, the undefeated-opponent fallback, the
        statistical components and the strength-of-schedule definition.
    """
    games, teams = canonicalize(games, teams)
    long = _long_form(games, sport)

    # --- Pass 1: raw records -------------------------------------------------
    aggregations = {"wins": ("won", "sum"), "losses": ("lost", "sum")}
    for column in sport.stat_pairs:
        aggregations[column] = (column, "sum")
    records = long.groupby("team").agg(**aggregations)
    if teams is not None:
        records = records.reindex(records.index.union(teams)).fillna(0.0)

    wins = records["wins"]
    losses = records["losses"]

    # --- Pass 2: opponents' records ------------------------------------------
    long["opp_wins"] = long["opponent"].map(wins).fillna(0.0)
    long["opp_losses"] = long["opponent"].map(losses).fillna(0.0)
    opp = long.groupby("team").agg(
        opp_wins=("opp_wins", "sum"),
        opp_losses=("opp_losses", "sum"),
    )
    records = records.join(opp).fillna(0.0)

    # --- Pass 3: opponents' opponents' records -------------------------------
    long["opp_opp_wins"] = long["opponent"].map(records["opp_wins"]).fillna(0.0)
    long["opp_opp_losses"] = long["opponent"].map(records["opp_losses"]).fillna(0.0)
    opp2 = long.groupby("team").agg(
        opp_opp_wins=("opp_opp_wins", "sum"),
        opp_opp_losses=("opp_opp_losses", "sum"),
    )
    records = records.join(opp2).fillna(0.0)

    # --- Per-game points ------------------------------------------------------
    long["game_points"] = _game_points(long, sport)
    records = records.join(
        long.groupby("team")["game_points"].sum().rename("game_points")
    ).fillna(0.0)

    # --- Rates ---------------------------------------------------------------
    played = records["wins"] + records["losses"]
    records["games"] = played
    records["win_pct"] = _safe_pct(records["wins"], played)
    records["opp_win_pct"] = _safe_pct(
        records["opp_wins"], records["opp_wins"] + records["opp_losses"]
    )
    records["opp_opp_win_pct"] = _safe_pct(
        records["opp_opp_wins"], records["opp_opp_wins"] + records["opp_opp_losses"]
    )
    records["sos"] = sport.sos(records)

    sport.derive(records, played)

    # --- z-scores over rated teams only --------------------------------------
    fbs = _fbs_mask(records, teams)

    def z(column: str) -> pd.Series:
        pool = records.loc[fbs, column]
        mean, dev = pool.mean(), pool.std(ddof=1)
        if not dev:
            return pd.Series(0.0, index=records.index)
        return (records[column] - mean) / dev

    # --- The rating ----------------------------------------------------------
    records["mri"] = (
        WEIGHTS["win_pct"] * records["win_pct"]
        + WEIGHTS["opp_win_pct"] * records["opp_win_pct"]
        + WEIGHTS["opp_opp_win_pct"] * records["opp_opp_win_pct"]
        + records["game_points"]
    )
    for column, weight, sign in sport.components:
        contribution = z(column) * (weight * sign)
        records[f"z_{column}"] = contribution
        records["mri"] = records["mri"] + contribution
    records["mri_per_game"] = _safe_pct(records["mri"], played)

    result = records.loc[fbs].copy()
    result = result.sort_values("mri", ascending=False)
    result.insert(0, "rank", range(1, len(result) + 1))
    result.insert(1, "sos_rank", result["sos"].rank(ascending=False, method="min").astype(int))
    return result.reset_index().rename(columns={"index": "team"})


def _game_points(long: pd.DataFrame, sport: Sport = FOOTBALL) -> np.ndarray:
    """Opponent-weighted credit for each game, per the Games sheet."""
    opp_w = long["opp_wins"].to_numpy(float)
    opp_l = long["opp_losses"].to_numpy(float)
    opp2_w = long["opp_opp_wins"].to_numpy(float)
    opp2_l = long["opp_opp_losses"].to_numpy(float)
    margin = long["margin"].to_numpy(float)
    won = long["won"].to_numpy(float) == 1.0

    opp_games = opp_w + opp_l
    opp2_games = opp2_w + opp2_l

    opp_win_pct = np.divide(opp_w, opp_games, out=np.zeros_like(opp_w), where=opp_games != 0)
    opp2_win_pct = np.divide(opp2_w, opp2_games, out=np.zeros_like(opp2_w), where=opp2_games != 0)
    # The football sheet substitutes 0.1 when the opponent has no losses, so a
    # loss to an undefeated team still costs something rather than nothing. The
    # basketball sheet has no such fallback, and copying one in would change
    # every rating in a season where anyone runs the table.
    opp_loss_pct = np.divide(
        opp_l, opp_games, out=np.zeros_like(opp_l), where=opp_games != 0
    )
    if sport.undefeated_fallback is not None:
        opp_loss_pct = np.where(opp_l == 0, sport.undefeated_fallback, opp_loss_pct)
    opp2_loss_pct = np.divide(
        opp2_l, opp2_games, out=np.zeros_like(opp2_l), where=opp2_games != 0
    )

    win_points = np.minimum(sport.margin_cap, margin) * opp_win_pct * opp2_win_pct
    loss_points = np.maximum(-sport.margin_cap, margin) * opp_loss_pct * opp2_loss_pct
    return np.where(won, win_points, loss_points)


def _fbs_mask(records: pd.DataFrame, teams: list[str] | None) -> pd.Series:
    if teams is None:
        return records.index != POOLED_FCS
    fbs_teams = [t for t in teams if t != POOLED_FCS]
    return records.index.isin(fbs_teams)
