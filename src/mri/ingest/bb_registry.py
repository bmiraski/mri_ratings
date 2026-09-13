"""Team identity for basketball: 365 D1 programs, and the names they used to have.

Football's registry is a hand-written file because 138 teams and 10 conferences
fit in one's head. Basketball has 365 teams across 31 conferences, so the roster
and conference assignments come from the API, which is authoritative and updates
itself. Only the part that needs human judgement is written by hand: what a
program used to be called.

That part is written by hand deliberately. Fuzzy matching proposes plausible and
wrong answers here - on this archive it offered Mississippi -> Mississippi State
(0.79), North Carolina State -> South Carolina State (0.90), Nebraska Omaha ->
Nebraska (0.73) and St. Francis (NY) -> St. Francis (PA) (0.85). Every one is a
different school, and accepting any of them would merge two programs' records
without raising an error. So the matcher is a proposal generator; the mapping
below was checked one line at a time, and every target is asserted to exist in
the API roster.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

CURRENT_SEASON = 2027  # the 2026-27 season

# Historical workbook spelling -> current API name.
ALIASES = {
    # Style and abbreviation
    "Albany": "UAlbany",
    "American": "American University",
    "Appalachian State": "App State",
    "Arkansas-Little Rock": "Little Rock",
    "Cal": "California",
    "Central Conn St.": "Central Connecticut",
    "Central Florida": "UCF",
    "Connecticut": "UConn",
    "Hawaii": "Hawai'i",
    "Illinois-Chicago": "UIC",
    "Loyola (MD)": "Loyola Maryland",
    "Md.-Baltimore Co.": "UMBC",
    "Md.-Eastern Shore": "Maryland Eastern Shore",
    "Miami (Ohio)": "Miami (OH)",
    "Middle Tenn. St": "Middle Tennessee",
    "Mississippi": "Ole Miss",
    "Nicholls State": "Nicholls",
    "McNeese State": "McNeese",
    "North Carolina State": "NC State",
    "Prairie View": "Prairie View A&M",
    "Seattle": "Seattle U",
    "Southeastern Louisiana": "SE Louisiana",
    "Southern Mississippi": "Southern Miss",
    "St. Joseph's": "Saint Joseph's",
    "St. Mary's": "Saint Mary's",
    "St. Peter's": "Saint Peter's",
    "Stephen Austin": "Stephen F. Austin",
    "Tennessee-Martin": "UT Martin",
    "Texas AMCC": "Texas A&M-Corpus Christi",
    "Texas Arlington": "UT Arlington",
    "Texas San Antonio": "UTSA",
    "UCSB": "UC Santa Barbara",
    "USC Upstate": "South Carolina Upstate",
    "Utah Valley State": "Utah Valley",
    "Virginia Commonwealth": "VCU",
    "William and Mary": "William & Mary",
    # Programs that dropped a geographic qualifier
    "Cal State Sacramento": "Sacramento State",
    "Nebraska Omaha": "Omaha",
    "UM-Kansas City": "Kansas City",
    "Wisconsin-Green Bay": "Green Bay",
    "Wisconsin-Milwaukee": "Milwaukee",
    "Louisiana-Lafayette": "Louisiana",
    # Actual renames
    "Detroit": "Detroit Mercy",
    "Houston Baptist": "Houston Christian",
    "IUPUI": "IU Indianapolis",
    "IUPUFW": "Purdue Fort Wayne",
    "LIU-Brooklyn": "Long Island University",
    "Texas Pan-American": "UT Rio Grande Valley",
}

# Programs in the archive that have since left Division I. They are not errors
# and must not be fuzzy-matched onto a surviving school with a similar name.
#
# Membership is season-dependent, which is the thing to keep hold of: the API
# roster changes every year, so a historical name should be resolved against the
# season being rated rather than against whatever is current. St. Francis (PA)
# is the live example - present in 2024-25 and 2025-26, gone from 2026-27.
DEPARTED = {
    "Hartford": "left D1 after 2022-23",
    "Savannah State": "left D1 after 2018-19",
    "St. Francis (NY)": "dropped to D3 in 2023; a different school from St. Francis (PA)",
    "St. Francis (PA)": "leaves D1 after 2025-26",
    "Louisiana-Monroe": "carried under a different spelling; resolved per season",
}


@dataclass(frozen=True)
class Team:
    name: str
    conference: str
    abbreviation: str | None
    color: str | None


def _key(name: object) -> str:
    """Case, accents and punctuation all vary between sources."""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = "".join(c if c.isalnum() or c.isspace() else " " for c in text)
    return " ".join(text.split()).casefold()


@lru_cache(maxsize=4)
def teams(season: int = CURRENT_SEASON) -> dict[str, Team]:
    """The D1 roster for a season, from the API."""
    from . import cbbd

    frame = cbbd.teams(season)
    return {
        row.team: Team(
            name=row.team,
            conference=row.conference or "Independent",
            abbreviation=row.abbreviation,
            color=row.color,
        )
        for row in frame.itertuples()
    }


@lru_cache(maxsize=4)
def _index(season: int = CURRENT_SEASON) -> dict[str, str]:
    roster = teams(season)
    index = {_key(name): name for name in roster}
    for alias, canonical in ALIASES.items():
        if canonical in roster:
            index.setdefault(_key(alias), canonical)
    return index


def resolve(name: object, default: str | None = None, season: int = CURRENT_SEASON) -> str | None:
    return _index(season).get(_key(name), default)


def is_d1(name: object, season: int = CURRENT_SEASON) -> bool:
    return resolve(name, season=season) is not None


def conference_of(name: object, season: int = CURRENT_SEASON) -> str | None:
    canonical = resolve(name, season=season)
    return teams(season)[canonical].conference if canonical else None


def conferences(season: int = CURRENT_SEASON) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for team in teams(season).values():
        grouped.setdefault(team.conference, []).append(team.name)
    return {k: sorted(v) for k, v in sorted(grouped.items())}


def audit(names, season: int = CURRENT_SEASON) -> pd.DataFrame:
    """What a set of names resolves to, and what is left over.

    Run this against every archive season before trusting a rating: an
    unresolved name is a team whose games are about to be silently dropped.
    """
    rows = []
    for name in sorted({str(n) for n in names}):
        canonical = resolve(name, season=season)
        rows.append(
            {
                "name": name,
                "resolves_to": canonical,
                "status": (
                    "ok" if canonical
                    else "departed" if name in DEPARTED
                    else "UNRESOLVED"
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_aliases(season: int = CURRENT_SEASON) -> list[str]:
    """Every alias must point at a real team. Returns the broken ones."""
    roster = teams(season)
    return [
        f"{alias} -> {canonical}"
        for alias, canonical in ALIASES.items()
        if canonical not in roster
    ]
