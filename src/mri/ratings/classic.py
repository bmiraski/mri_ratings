"""MRI Classic - a faithful Python port of the 2018 Excel formula.

This module is deliberately not improved. It exists to reproduce the workbook
exactly so that (a) the historical rankings stay comparable and (b) there is a
known-good baseline to measure MRI 2.0 against. Every quirk is preserved,
including the pooled FCS opponent, the +/-35 margin cap, and the 0.1 fallback
when an opponent is undefeated.

The formula, from Team Data column V:

    MRI = 25 * WinPct
        + 25 * OppWinPct
        + 10 * OppOppWinPct
        + sum(game points)
        +  5 * z(RushYds/G)
        +  5 * z(PassYds/G)
        +  7 * -z(TotalYdsAllowed/G)
        +  3 * z(TurnoverMargin)

Game points, from Games columns U and V:

    win  ->  min( 35, margin) * OppWinPct  * OppOppWinPct
    loss ->  max(-35, margin) * OppLossPct * OppOppLossPct

z-scores use the sample standard deviation (Excel STDEV) over FBS teams only;
the pooled FCS row is excluded from the statistics and from the rankings, but
its win-loss record still feeds every opponent lookup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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


def _long_form(games: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game, so aggregation is a single groupby."""
    side1 = pd.DataFrame(
        {
            "team": games["team1"],
            "opponent": games["team2"],
            "points_for": games["pts1"],
            "points_against": games["pts2"],
            "rush_for": games["rush1"],
            "rush_against": games["rush2"],
            "pass_for": games["pass1"],
            "pass_against": games["pass2"],
            "to_committed": games["to1"],
            "to_forced": games["to2"],
            "won": games["win1"],
        }
    )
    side2 = pd.DataFrame(
        {
            "team": games["team2"],
            "opponent": games["team1"],
            "points_for": games["pts2"],
            "points_against": games["pts1"],
            "rush_for": games["rush2"],
            "rush_against": games["rush1"],
            "pass_for": games["pass2"],
            "pass_against": games["pass1"],
            "to_committed": games["to2"],
            "to_forced": games["to1"],
            "won": games["win2"],
        }
    )
    long = pd.concat([side1, side2], ignore_index=True)
    long["lost"] = 1.0 - long["won"]
    long["margin"] = long["points_for"] - long["points_against"]
    return long


def compute(games: pd.DataFrame, teams: list[str] | None = None) -> pd.DataFrame:
    """Compute MRI Classic for every team in ``games``.

    Parameters
    ----------
    games
        Normalized game table from ``mri.ingest.archive.read_games``.
    teams
        FBS team list. Teams outside this list (the pooled FCS row) still
        contribute to opponent records but are excluded from the z-score
        statistics and from the returned rankings.
    """
    games, teams = canonicalize(games, teams)
    long = _long_form(games)

    # --- Pass 1: raw records -------------------------------------------------
    records = long.groupby("team").agg(
        wins=("won", "sum"),
        losses=("lost", "sum"),
        rush_yards=("rush_for", "sum"),
        rush_allowed=("rush_against", "sum"),
        pass_yards=("pass_for", "sum"),
        pass_allowed=("pass_against", "sum"),
        to_committed=("to_committed", "sum"),
        to_forced=("to_forced", "sum"),
    )
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
    long["game_points"] = _game_points(long)
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
    records["sos"] = records["opp_win_pct"] * records["opp_opp_win_pct"]

    records["rush_per_game"] = _safe_pct(records["rush_yards"], played)
    records["pass_per_game"] = _safe_pct(records["pass_yards"], played)
    records["yards_allowed_per_game"] = _safe_pct(
        records["rush_allowed"] + records["pass_allowed"], played
    )
    records["turnover_margin"] = records["to_forced"] - records["to_committed"]

    # --- z-scores over FBS only ----------------------------------------------
    fbs = _fbs_mask(records, teams)
    stats = {}
    for column in (
        "rush_per_game",
        "pass_per_game",
        "yards_allowed_per_game",
        "turnover_margin",
    ):
        pool = records.loc[fbs, column]
        stats[column] = (pool.mean(), pool.std(ddof=1))

    def z(column: str) -> pd.Series:
        mean, dev = stats[column]
        if not dev:
            return pd.Series(0.0, index=records.index)
        return (records[column] - mean) / dev

    records["z_rush"] = z("rush_per_game")
    records["z_pass"] = z("pass_per_game")
    records["z_defense"] = -z("yards_allowed_per_game")
    records["z_turnovers"] = z("turnover_margin")

    # --- The rating ----------------------------------------------------------
    records["mri"] = (
        WEIGHTS["win_pct"] * records["win_pct"]
        + WEIGHTS["opp_win_pct"] * records["opp_win_pct"]
        + WEIGHTS["opp_opp_win_pct"] * records["opp_opp_win_pct"]
        + records["game_points"]
        + WEIGHTS["rush"] * records["z_rush"]
        + WEIGHTS["pass"] * records["z_pass"]
        + WEIGHTS["defense"] * records["z_defense"]
        + WEIGHTS["turnovers"] * records["z_turnovers"]
    )
    records["mri_per_game"] = _safe_pct(records["mri"], played)

    result = records.loc[fbs].copy()
    result = result.sort_values("mri", ascending=False)
    result.insert(0, "rank", range(1, len(result) + 1))
    result.insert(1, "sos_rank", result["sos"].rank(ascending=False, method="min").astype(int))
    return result.reset_index().rename(columns={"index": "team"})


def _game_points(long: pd.DataFrame) -> np.ndarray:
    """Opponent-weighted credit for each game, per Games!U and Games!V."""
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
    # Excel substitutes 0.1 when the opponent has no losses, so a loss to an
    # undefeated team still costs something rather than nothing.
    opp_loss_pct = np.where(
        opp_l == 0,
        UNDEFEATED_OPPONENT_FALLBACK,
        np.divide(opp_l, opp_games, out=np.zeros_like(opp_l), where=opp_games != 0),
    )
    opp2_loss_pct = np.divide(
        opp2_l, opp2_games, out=np.zeros_like(opp2_l), where=opp2_games != 0
    )

    win_points = np.minimum(MARGIN_CAP, margin) * opp_win_pct * opp2_win_pct
    loss_points = np.maximum(-MARGIN_CAP, margin) * opp_loss_pct * opp2_loss_pct
    return np.where(won, win_points, loss_points)


def _fbs_mask(records: pd.DataFrame, teams: list[str] | None) -> pd.Series:
    if teams is None:
        return records.index != POOLED_FCS
    fbs_teams = [t for t in teams if t != POOLED_FCS]
    return records.index.isin(fbs_teams)
