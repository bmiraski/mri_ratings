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
    """Match the way Excel compares text: case- and whitespace-insensitive."""
    return " ".join(str(name).split()).casefold()


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
    return resolve(name) is not None


def unresolved(names) -> list[str]:
    """Names that the registry cannot place - useful as an ingest guard."""
    seen: dict[str, None] = {}
    for name in names:
        if not is_fbs(name):
            seen.setdefault(" ".join(str(name).split()), None)
    return sorted(seen)
