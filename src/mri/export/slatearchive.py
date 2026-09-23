"""Past weeks' slates, kept as they stood before the Sunday refresh replaced them.

The slate page moves on to the next week the first time a build finds every game
of the current one played - Sunday morning, in practice. Just before it does,
``snapshot`` saves the outgoing page's data to ``docs/slate/<season>-week-<n>.json``
with what it needs to be drawn again later: the final scores, and each team's
rank, colour and logo as they were that week rather than as they are now.

The saved file is written once and never touched again. The page drawn from it is
re-rendered on every build, so a change to the slate page reaches past weeks too.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

TEAM_FIELDS = ("rank", "abbreviation", "color", "logo")


def archive_path(docs: Path, season: int, week: int) -> Path:
    return docs / "slate" / f"{season}-week-{week}.json"


def _load(path: Path) -> dict | None:
    # Python's json reads the bare NaN the site payload can carry, which is what this needs.
    return json.loads(path.read_text()) if path.exists() else None


def freeze(slate: dict, site: dict, finals: dict[int, tuple[int, int]], *, saved: str) -> dict:
    """The saved record: the slate as it was, the week's teams as they were, and the final scores."""
    season = slate.get("season") or site.get("season")
    ids = {g["id"] for day in slate["days"] for g in day["games"]}
    teams = {t["team"]: {k: t.get(k) for k in TEAM_FIELDS} for t in site.get("teams", [])}
    return {
        **slate,
        "season": season,
        "saved": saved,
        "teams": teams,
        "finals": {str(i): {"home": h, "away": a} for i, (h, a) in finals.items() if i in ids},
    }


def snapshot(docs: Path, new_slate: dict | None, season: int, finals_for_week) -> Path | None:
    """Save the outgoing week if this build is about to replace it. Returns the file written, if any.

    ``finals_for_week(week)`` gives ``{game_id: (home_points, away_points)}`` for a finished week; the
    caller has the season's games loaded already.
    """
    old = _load(docs / "slate.json")
    if not old:
        return None
    site = _load(docs / "site.json") or {}
    old_season = old.get("season") or site.get("season") or season
    if new_slate and new_slate["week"] == old["week"] and (new_slate.get("season") or season) == old_season:
        return None
    path = archive_path(docs, old_season, old["week"])
    if path.exists():
        return None
    record = freeze({**old, "season": old_season}, site, finals_for_week(old["week"]),
                    saved=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1))
    return path
