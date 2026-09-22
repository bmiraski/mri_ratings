"""The NCAA tournament and conference tournament games, from the postseason feed.

The regular games ingest already pulls every postseason game byte for byte, but
drops two columns on the way in that this work needs: the seed (only the NCAA
tournament carries one) and ``gameNotes``, which is the only place the round and
region live. Kept separate from :mod:`mri.ingest.cbbd` rather than added to it,
because nothing else needs these columns and every rating computation would
otherwise carry them for no reason.

Two feeds, not one, because they arrive differently:

*The NCAA tournament* is ``seasonType=postseason``, ``tournament="NCAA"``, and
every game carries both teams' seeds. ``gameNotes`` reads
``"Men's Basketball Championship - {Region} Region - {Round}"`` for every round
except the Final Four and the championship game, which drop the region.

*A conference tournament* is ``seasonType=regular``, ``gameType="TRNMNT"``,
``conferenceGame=True``. Its ``gameNotes`` reads ``"{Conference} Championship -
{Round}"`` — but the round vocabulary is not consistent across conferences
("Quarterfinal" in one, "2nd Round" in another for the structurally same stage),
so the round each game belongs to is recovered from its calendar date instead of
its label: the earliest date a conference's bracket has games is round one,
whatever either conference calls it. Seed numbers are never given for these
games at all; a conference's own seeding has to come from its standings.
"""

from __future__ import annotations

import re

import pandas as pd

from . import cfbd
from .cbbd import BASE_URL, request

# The prefix varies ("Men's Basketball Championship", "SEC Tournament", "ASUN Championship", ...) and
# isn't needed for anything here, so the pattern only pins down the region (if there is one) and the
# round at the end, whatever the event itself is called.
_REGION_ROUND = re.compile(r"^.+? - (?:(?P<region>[^-]+) Region - )?(?P<round>[^-]+)$")


def _parse_notes(notes: str | None) -> tuple[str | None, str | None]:
    if not notes:
        return None, None
    m = _REGION_ROUND.match(notes)
    return (m.group("region"), m.group("round")) if m else (None, notes)


def _window_games(season: int, season_type: str, start: str, end: str, *, refresh: bool) -> list[dict]:
    original = cfbd.BASE_URL
    cfbd.BASE_URL = BASE_URL
    try:
        return request("/games", season=season, seasonType=season_type, startDateRange=start, endDateRange=end, refresh=refresh)
    finally:
        cfbd.BASE_URL = original


def ncaa_tournament(season: int, *, refresh: bool = False) -> pd.DataFrame:
    """Every NCAA tournament game of a season: both teams' seeds, the round, the region.

    The tournament always falls in March and early April of the year the season is
    named for (season 2025 is the 2024-25 season, and its tournament is March 2025).
    """
    raw = _window_games(season, "postseason", f"{season}-03-01", f"{season}-04-15", refresh=refresh)
    rows = []
    for g in raw:
        if g.get("tournament") != "NCAA":
            continue
        region, rnd = _parse_notes(g.get("gameNotes"))
        rows.append({
            "season": season, "game_id": g["id"], "start_date": g.get("startDate"), "region": region, "round": rnd,
            "team1": g["awayTeam"], "team2": g["homeTeam"], "seed1": g.get("awaySeed"), "seed2": g.get("homeSeed"),
            "conf1": g.get("awayConference"), "conf2": g.get("homeConference"),
            "pts1": g.get("awayPoints"), "pts2": g.get("homePoints"), "win1": bool(g.get("awayWinner")),
        })
    frame = pd.DataFrame(rows)
    return frame.sort_values("start_date").reset_index(drop=True) if not frame.empty else frame


def conference_tournaments(season: int, *, refresh: bool = False) -> pd.DataFrame:
    """Every conference tournament game of a season, across every conference.

    Conference tournaments run from late February through mid-March; the window is
    wide on purpose; a handful of one-bid conferences finish as early as the second
    week of March in a normal year and the earliest in recent seasons (Southland,
    2013 rule change onward) has started the last week of February.
    """
    raw = _window_games(season, "regular", f"{season}-02-20", f"{season}-03-20", refresh=refresh)
    rows = []
    for g in raw:
        if g.get("gameType") != "TRNMNT" or not g.get("conferenceGame"):
            continue
        conf = g.get("awayConference") or g.get("homeConference")
        if g.get("awayConference") != g.get("homeConference"):
            continue                          # a conference's own bracket only, never a possible cross-conference oddity
        _, rnd = _parse_notes(g.get("gameNotes"))
        rows.append({
            "season": season, "conference": conf, "game_id": g["id"], "start_date": g.get("startDate"), "round_label": rnd,
            "team1": g["awayTeam"], "team2": g["homeTeam"], "pts1": g.get("awayPoints"), "pts2": g.get("homePoints"),
            "win1": bool(g.get("awayWinner")), "neutral": bool(g.get("neutralSite")),
        })
    frame = pd.DataFrame(rows)
    return frame.sort_values("start_date").reset_index(drop=True) if not frame.empty else frame


def champion(games: pd.DataFrame) -> str | None:
    """The team that won a single conference's bracket: the winner of its latest-dated game."""
    if games.empty:
        return None
    last = games.sort_values("start_date").iloc[-1]
    return last["team1"] if last["win1"] else last["team2"]
