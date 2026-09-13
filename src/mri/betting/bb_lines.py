"""Basketball betting lines, normalized.

The football module's two traps are both present here, unchanged.

*Provider names are not stable.* DraftKings is spelled "Draft Kings" in this
feed - every one of the 4,516 DraftKings lines in 2025-26 - so an exact match on
"DraftKings" finds nothing at all and reports no error while doing it. The alias
table is shared with football rather than copied, because a book that gets a new
spelling should only have to be fixed once.

*The spread is quoted from the home team's perspective and is negative when the
home team is favoured.* The market's expected home margin is therefore
``-spread``. Verified on 4,514 DraftKings lines from 2025-26: correlation with
actual home margin is +0.62 using ``-spread`` and -0.62 using ``spread``; the
home side covers 49.0% of the time and the mean market number (+4.32) sits just
above the mean actual margin (+4.05), which is what an efficient market that
also collects vig looks like.

Coverage is the thing to understand before designing any test here, and it is
not what a single season suggests:

    numberfire      2020-21             NOT A MARKET - a projection service
    teamrankings    2020-21 to 2022-23  NOT A MARKET - a projection service
    consensus       2020-21 to 2022-23
    ESPN Bet        2022-23 onward, ~5,400 games a season; openers from 2024-25
    Bovada          2025-26 only
    DraftKings      2025-26 only, 4,503 games

Two consequences, both of which constrain what can honestly be claimed.

DraftKings - the book Ben actually bets - covers one season. That is a real
sample for bucketing by edge size, but one season cannot tell an edge apart from
a season, so it can support a cross-check and not a finding.

numberfire and teamrankings are model outputs, not sportsbooks. Backtesting
against them would measure agreement with somebody else's projection and report
it as market edge. They are excluded by name rather than by judgement at the
call site, because the failure is silent and the numbers look fine.

One structural difference from football: there are no week numbers, so lines are
fetched by date window like games, and joined to games by id.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from ..ingest import bb_registry as registry, cbbd
from .lines import PROVIDER_ALIASES, normalize_provider  # noqa: F401 - shared table

# The book Ben actually bets into. Kept as a name rather than inlined so a
# backtest and a board cannot quietly disagree about which market they mean.
BOOK = "DraftKings"

# The longest-running real sportsbook in this feed, and so the only one that can
# carry a multi-season backtest. DraftKings prices the live board.
HISTORY_BOOK = "ESPN Bet"

# Projection services that the feed lists alongside sportsbooks. A model scored
# against these is being scored against another model, and would report the
# agreement as though it were an edge over a market.
NOT_MARKETS = {"numberfire", "teamrankings"}

# Where each real market actually exists, so a backtest asks only for what is
# there rather than silently returning an empty frame.
FIRST_SEASON = {"DraftKings": 2026, "Bovada": 2026, "ESPN Bet": 2023, "consensus": 2021}


def season_lines(year: int, *, refresh_last: bool = True) -> pd.DataFrame:
    """Every line for a season, one row per game per provider.

    ``market`` is the market's expected home margin - the sign already flipped -
    so it compares directly against a model's predicted margin. ``market_open``
    is the same for the opening number, which is where football's backtest
    suggested any edge would have to live.
    """
    today = dt.date.today()
    rows = []
    for start, end in cbbd._windows(year):
        payload = cbbd.request(
            "/lines",
            season=year,
            startDateRange=start,
            endDateRange=end,
            refresh=refresh_last and cbbd._is_live(start, end, today),
        )
        for game in payload:
            home = registry.resolve(game.get("homeTeam"), game.get("homeTeam"), season=year)
            away = registry.resolve(game.get("awayTeam"), game.get("awayTeam"), season=year)
            for line in game.get("lines") or []:
                spread = _number(line.get("spread"))
                if spread is None:
                    continue
                opening = _number(line.get("spreadOpen"))
                rows.append(
                    {
                        "game_id": game["gameId"],
                        "season": game["season"],
                        "start_date": game.get("startDate"),
                        "season_type": game.get("seasonType"),
                        "team1": away,
                        "team2": home,
                        "pts1": game.get("awayScore"),
                        "pts2": game.get("homeScore"),
                        "provider": normalize_provider(line.get("provider")),
                        # Negated: the market's expected home margin.
                        "market": -spread,
                        "market_open": -opening if opening is not None else None,
                        "total": _number(line.get("overUnder")),
                    }
                )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["actual"] = frame["pts2"] - frame["pts1"]
    # Projection services are dropped here rather than at the call site: a
    # backtest that picked one up would run perfectly and measure the wrong
    # thing.
    frame = frame[~frame["provider"].str.casefold().isin(NOT_MARKETS)]
    # One row per game per provider; duplicates appear when a book is listed
    # twice under different spellings.
    frame = frame.drop_duplicates(subset=["game_id", "provider"])
    return frame.sort_values("start_date").reset_index(drop=True)


def book_lines(year: int, provider: str = BOOK, *, refresh_last: bool = True) -> pd.DataFrame:
    """One book's lines for a season.

    No fallback to another market when the chosen one is missing. Football falls
    back to a consensus number to keep its long history usable, but here the two
    books barely overlap in time, so a silent substitution would produce a
    backtest whose early seasons and late seasons are measuring different
    markets while presenting one number.
    """
    frame = season_lines(year, refresh_last=refresh_last)
    if frame.empty:
        return frame
    return frame[frame["provider"] == provider].reset_index(drop=True)


def available_seasons(provider: str) -> list[int]:
    """The seasons a market can actually be asked about."""
    from ..ingest.bb_registry import CURRENT_SEASON

    first = FIRST_SEASON.get(provider)
    return list(range(first, CURRENT_SEASON)) if first else []


def _number(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
