"""Betting lines, normalized.

Two traps in this data, both of which silently shrink or corrupt a backtest:

*Provider names are not stable.* DraftKings appears as both "DraftKings" and
"Draft Kings" - in 2026 the spaced spelling is 290 of 683 lines, so an exact
match drops fewer than half the games without any error.

*The spread is quoted from the home team's perspective and is negative when the
home team is favoured.* The market's expected home margin is therefore
``-spread``. Verified against 811 games of 2024 DraftKings lines: correlation
with actual home margin is +0.66 using ``-spread`` and -0.66 using ``spread``,
and the home side covers 49.9% of the time, which is what an efficient market
should look like.

Availability is worth knowing before designing any test:

    consensus     2013-2025, no opening line ever
    Bovada        opening lines from 2022
    DraftKings    2023 onward only
"""

from __future__ import annotations

import pandas as pd

from ..ingest import cfbd, registry

# Every spelling a book has appeared under, mapped to one name.
PROVIDER_ALIASES = {
    "draftkings": "DraftKings",
    "draft kings": "DraftKings",
    "espn bet": "ESPN Bet",
    "espnbet": "ESPN Bet",
    "william hill (new jersey)": "William Hill",
    "caesars": "Caesars",
    "bovada": "Bovada",
    "consensus": "consensus",
    "teamrankings": "teamrankings",
    "numberfire": "numberfire",
}

# Where each book's numbers actually exist, so a backtest asks for what is there.
FIRST_SEASON = {"DraftKings": 2023, "Bovada": 2021, "ESPN Bet": 2023, "consensus": 2013}


def normalize_provider(name: object) -> str:
    key = " ".join(str(name).split()).casefold()
    return PROVIDER_ALIASES.get(key, str(name).strip())


def season_lines(year: int, *, refresh: bool | None = None) -> pd.DataFrame:
    """Every line for a season, one row per game per provider.

    ``market`` is the market's expected home margin - the sign already flipped -
    so it compares directly against a model's predicted margin.

    ``refresh=None`` re-fetches the season in progress, once per run. Lines are
    the most perishable thing this site publishes - they move daily - and they
    were being served from the first copy cached, so a board built on Friday was
    pricing games off whatever the market said whenever the file was seeded.
    Finished seasons stay cached forever.
    """
    if refresh is None:
        refresh = year == cfbd.current_season()
    raw = cfbd.request("/lines", year=year, refresh=refresh)
    rows = []
    for game in raw:
        home = registry.resolve(game.get("homeTeam"), game.get("homeTeam"))
        away = registry.resolve(game.get("awayTeam"), game.get("awayTeam"))
        for line in game.get("lines", []):
            spread = _number(line.get("spread"))
            if spread is None:
                continue
            opening = _number(line.get("spreadOpen"))
            rows.append(
                {
                    "game_id": game["id"],
                    "season": game["season"],
                    "week": game["week"],
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
    # One row per game per provider; duplicates happen when a book is listed
    # twice under different spellings.
    return frame.drop_duplicates(subset=["game_id", "provider"]).reset_index(drop=True)


def preferred_lines(year: int, provider: str = "DraftKings", fallback: str = "consensus",
                    *, refresh: bool | None = None) -> pd.DataFrame:
    """The chosen book where it exists, a fallback where it does not.

    DraftKings only starts in 2023, so a backtest that insists on it has three
    seasons to work with. Falling back to the consensus number keeps the long
    history usable while the recent seasons still reflect the book actually
    being bet into.
    """
    frame = season_lines(year, refresh=refresh)
    if frame.empty:
        return frame

    chosen = frame[frame["provider"] == provider]
    if len(chosen) >= 100:
        return chosen.assign(source=provider)

    backup = frame[frame["provider"] == fallback]
    return backup.assign(source=fallback)


def _number(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
