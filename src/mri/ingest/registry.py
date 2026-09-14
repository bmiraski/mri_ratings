"""The FBS team registry: canonical names, conferences, and alias resolution.

Team naming is the quiet source of bugs in a project like this. The 2003-2019
workbooks, the CFBD API, and sportsbooks all spell teams differently, and a
mismatch shows up as a team that silently loses half its schedule. Everything
that touches a team name goes through ``resolve`` so there is exactly one
canonical spelling per program.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
import unicodedata
from functools import lru_cache
from pathlib import Path

_REGISTRY_PATH = Path(__file__).with_name("teams_2026.json")

POOLED_FCS = "Non D1A"


@dataclass(frozen=True)
class Team:
    name: str
    conference: str
    conference_display: str
    tier: str


def _key(name: object) -> str:
    """Normalize a team name down to something that matches across sources.

    Excel compared text case- and whitespace-insensitively, which the archive
    depends on. The API adds two more wrinkles: diacritics ("San Jose State")
    and punctuation ("Hawai'i", "Middle Tenn. St"), so both are stripped. What
    survives is letters, digits and single spaces.
    """
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = "".join(c if c.isalnum() or c.isspace() else " " for c in text)
    return " ".join(text.split()).casefold()


@lru_cache(maxsize=1)
def _load() -> dict:
    with _REGISTRY_PATH.open() as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def teams() -> dict[str, Team]:
    """Canonical name -> Team, for the current season."""
    registry = _load()
    out: dict[str, Team] = {}
    for conference, meta in registry["conferences"].items():
        for name in meta["teams"]:
            out[name] = Team(
                name=name,
                conference=conference,
                conference_display=meta["display"],
                tier=meta["tier"],
            )
    return out


@lru_cache(maxsize=1)
def _alias_index() -> dict[str, str]:
    index = {_key(name): name for name in teams()}
    for alias, canonical in _load()["aliases"].items():
        index.setdefault(_key(alias), canonical)
    return index


def resolve(name: object, default: str | None = None) -> str | None:
    """Map any known spelling onto the canonical name.

    Returns ``default`` for names the registry does not know, which is the
    normal case for FCS opponents and for programs that have left FBS.
    """
    return _alias_index().get(_key(name), default)


def conference_of(name: object) -> str | None:
    canonical = resolve(name)
    return teams()[canonical].conference if canonical else None


def conferences() -> dict[str, dict]:
    return _load()["conferences"]


def is_fbs(name: object) -> bool:
    """Whether a team is FBS *now*, per the hand-written 2026 registry.

    This is the right question for the live season and the wrong one for
    history. Use ``was_fbs`` when the season matters.
    """
    return resolve(name) is not None


# The registry is a hand-written file for the current season, which is fine for
# 138 teams that fit in one's head - and no use at all for a 23-season archive.
# Membership moved underneath it: 128 FBS teams in 2020, 138 in 2026. James
# Madison arrived in 2022, Kennesaw State and Delaware in 2025, North Dakota
# State in 2026, and Idaho left before any of it.
#
# Asking the current registry about 2020 therefore put teams in that season's
# rankings that were playing FCS football at the time. The API knows the roster
# for each year and is the authority here, exactly as it is for basketball;
# names come back through ``resolve`` so a season answered by the API and a
# season answered by the registry agree on spelling.
@lru_cache(maxsize=32)
def fbs_members(season: int) -> frozenset[str]:
    """The canonical names of every FBS team in a given season."""
    from . import cfbd

    try:
        frame = cfbd.fbs_teams(season)
    except Exception:  # noqa: BLE001 - an archive page is not worth a crash
        return frozenset()
    return frozenset(resolve(n, str(n)) for n in frame["team"])


def was_fbs(name: object, season: int | None = None) -> bool:
    """Whether a team was FBS in a season. Falls back to the current registry.

    The fallback matters: with no network and no cache the archive should show
    a slightly-too-generous field rather than an empty one.
    """
    if season is None:
        return is_fbs(name)
    members = fbs_members(season)
    if not members:
        return is_fbs(name)
    return resolve(name, str(name)) in members


def unresolved(names) -> list[str]:
    """Names that the registry cannot place - useful as an ingest guard."""
    seen: dict[str, None] = {}
    for name in names:
        if not is_fbs(name):
            seen.setdefault(" ".join(str(name).split()), None)
    return sorted(seen)
