"""How a conference's own tournament is shaped, learned from its games rather than
hand-documented.

Conferences vary in how many teams they invite and how many rounds of bye the
top seeds get, and more than one has changed its format within the last decade
(the MAC and the Southland both have, within this system's own training window).
A hand-written table of 32 formats would need re-checking every year and would
silently go stale the year it wasn't. Instead: group a conference's tournament
games by the calendar date they were actually played — the earliest date is
round one, whoever the conference calls it — and count how many teams are new to
the bracket at each successive date. A team appearing for the first time at the
third date group had a bye through the first two rounds; how many teams debut at
each round *is* the conference's bye structure, and it falls out of the schedule
without needing to know anyone's seed.

The single assumption this leans on, never checked and never wrong in practice:
whichever teams get the deepest byes are the better regular-season seeds. A
bracket has never been built the other way round.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import pandas as pd

EASTERN = ZoneInfo("America/New_York")


def _game_day(iso: str) -> dt.date:
    """The calendar date a game belongs to in US arena time, not the UTC date the API stores.

    A tip-off at 11:30pm Eastern is still stored a few hours into the next UTC date,
    which would otherwise split one night's round of games across two "rounds."
    """
    return pd.Timestamp(iso).to_pydatetime().astimezone(EASTERN).date()


@dataclass
class Template:
    """``tiers[0]`` is how many teams play in the very first round (no bye);
    ``tiers[i]`` for i > 0 is how many teams debut at that round instead, each
    with i rounds of bye. ``sum(tiers)`` is the field size.

    ``campus_hosted`` is True when the conference plays some rounds on the higher
    seed's home floor rather than at one neutral site throughout - which needs a
    home-court adjustment in the simulation, and which is the more common format
    among one-bid conferences (the NEC and plenty of others do this; every power
    conference plays its whole event at one arena)."""

    tiers: tuple[int, ...]
    seasons_seen: int
    stable: bool                # every season examined agreed on this shape
    campus_hosted: bool = False

    @property
    def size(self) -> int:
        return sum(self.tiers)

    @property
    def rounds(self) -> int:
        return len(self.tiers)


def _tiers_for_one_season(games: pd.DataFrame) -> tuple[int, ...] | None:
    """One season's bracket shape, from its own games alone."""
    if games.empty:
        return None
    days = games["start_date"].map(_game_day)
    by_date = sorted(days.unique())
    seen: set[str] = set()
    tiers: list[int] = []
    for date in by_date:
        round_games = games[days == date]
        teams = set(round_games["team1"]) | set(round_games["team2"])
        debut = teams - seen
        if debut:
            tiers.append(len(debut))
        seen |= teams
    return tuple(tiers) if tiers else None


def _campus_hosted(games: pd.DataFrame) -> bool:
    """Whether any of this bracket's games were played somewhere other than one shared neutral site."""
    return bool((~games["neutral"]).any()) if "neutral" in games.columns and not games.empty else False


def infer(seasons_of_games: list[pd.DataFrame], *, min_seasons: int = 1) -> Template | None:
    """A conference's current template, from its own tournament in each of several recent seasons.

    ``seasons_of_games`` is oldest first. The most recent season's shape is what is
    returned — a conference that just changed its field size or bye count (the SEC
    and Big 12 both have, growing with realignment) is using the new shape now, not
    whatever a vote across its history would say. ``stable`` reports whether every
    season examined agreed, which is the signal worth a human's attention: an
    unstable read is either a real, recent change (trust it) or too few
    tournament-only games in the window to read cleanly (check it).
    """
    shapes = [t for t in (_tiers_for_one_season(g) for g in seasons_of_games) if t]
    if len(shapes) < min_seasons:
        return None
    hosted = [_campus_hosted(g) for g, t in zip(seasons_of_games, (_tiers_for_one_season(g) for g in seasons_of_games)) if t]
    return Template(tiers=shapes[-1], seasons_seen=len(shapes), stable=(len(set(shapes)) == 1),
                    campus_hosted=hosted[-1] if hosted else False)
