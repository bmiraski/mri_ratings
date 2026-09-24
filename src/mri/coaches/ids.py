"""Coach identity: a stable, human-readable id from a name that CFBD does not stably identify.

CFBD's own coach ``id`` is only a join key within one API response (see
``mri.ingest.cfbd.coaches``); asking again, or for a different year range,
can renumber it. The id used everywhere else in this project is built from
the coach's name and the year of their first FBS hire instead, e.g.
``nick-saban-1990`` - readable, and stable regardless of how many times or in
what order CFBD is asked.

Two things can go wrong with a name-based id, and both are handled by a
hand-maintained alias file rather than by cleverness in code:

* The same coach's name is spelled differently in different records. Loading
  ``data/coach_aliases.json`` and mapping a raw spelling to a canonical one
  before the id is built fixes this.
* Two different real coaches share a name and a first-hire year closely
  enough to collide on the slug. ``detect_collisions`` finds these - it
  cannot resolve them - so they can be hand-checked and, if genuinely two
  people, given a disambiguating alias entry.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

ALIASES_PATH = Path(__file__).resolve().parents[3] / "data" / "coach_aliases.json"


def _slug(text: str) -> str:
    # Matches ``mri.export.site.slug``'s regex; duplicated rather than
    # imported so this data-layer module does not depend on the site renderer.
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


@lru_cache(maxsize=1)
def _aliases() -> dict[str, str]:
    if not ALIASES_PATH.exists():
        return {}
    return json.loads(ALIASES_PATH.read_text()).get("aliases", {})


def canonical_name(raw_name: str) -> str:
    """The name to build an id from, after applying hand-maintained aliases."""
    return _aliases().get(raw_name, raw_name)


def first_hire_year(hire_date: str | None, seasons: pd.Series) -> int:
    """The year to anchor a coach's id to: their hire date's year if it parses, else their earliest season."""
    if hire_date:
        try:
            return int(str(hire_date)[:4])
        except ValueError:
            pass
    return int(seasons.min())


def coach_id(raw_name: str, hire_year: int) -> str:
    return f"{_slug(canonical_name(raw_name))}-{hire_year}"


def assign_ids(coach_rows: pd.DataFrame) -> pd.DataFrame:
    """Add a ``coach_id`` column to a flattened coach-season frame, one id per ``coach_key``."""
    if coach_rows.empty:
        return coach_rows.assign(coach_id=pd.Series(dtype=str))

    per_coach = coach_rows.groupby("coach_key").agg(
        coach_name=("coach_name", "first"), hire_date=("hire_date", "first")
    )
    seasons_by_key = coach_rows.groupby("coach_key")["season"]
    ids = {
        key: coach_id(row["coach_name"], first_hire_year(row["hire_date"], seasons_by_key.get_group(key)))
        for key, row in per_coach.iterrows()
    }
    out = coach_rows.copy()
    out["coach_id"] = out["coach_key"].map(ids)
    return out


def detect_collisions(coach_rows: pd.DataFrame) -> pd.DataFrame:
    """Coach ids that map to more than one CFBD ``coach_key`` - candidates for a hand-added alias.

    Takes a frame that already has ``coach_id`` (from ``assign_ids``).
    Returns one row per colliding id, listing the distinct keys and raw names
    involved so a human can tell whether it is one coach spelled two ways (an
    alias fix) or two different people (their ids need to be told apart, for
    example by hand-editing one's alias to include a middle initial).
    """
    if coach_rows.empty:
        return pd.DataFrame(columns=["coach_id", "coach_keys", "names"])
    by_id = coach_rows.groupby("coach_id")["coach_key"].nunique()
    colliding = by_id[by_id > 1].index
    rows = []
    for cid in colliding:
        subset = coach_rows[coach_rows["coach_id"] == cid]
        rows.append(
            {
                "coach_id": cid,
                "coach_keys": sorted(subset["coach_key"].unique().tolist()),
                "names": sorted(subset["coach_name"].unique().tolist()),
            }
        )
    return pd.DataFrame(rows, columns=["coach_id", "coach_keys", "names"])
