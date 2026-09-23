"""The bracketology pages' data: today's projection, how it has moved, and - once the real bracket
is out - how it did.

Three states, from the calendar in ``data/bracketology_settings.json``:

* **Before Christmas: nothing.** Too few games for a field to mean much, so there is no page and no
  nav link, the same way the football simulation disappears out of season.
* **Christmas to Selection Sunday: live.** The joint simulation runs on every build. Each day's
  odds are saved (compactly) so the page can show a week's movement, which is the comparison
  that means something in a sport that plays every night.
* **After Selection Sunday: frozen.** The last projection made before the committee's reveal is
  kept exactly as it was, and once the real field can be read from the tournament feed, set
  beside it: who we had right, who we missed, and how far the seed lines were off. A projection
  recomputed after the bracket is public would be marking our own homework.

The page disappears again when the season rolls over in July.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from ..bracket import live

HISTORY_DAYS = 7            # movement is measured against the projection about a week earlier


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _save(path: Path, data: dict, *, compact: bool = False) -> None:
    path.write_text(json.dumps(data, separators=(",", ":"), sort_keys=True) + "\n" if compact
                    else json.dumps(data, indent=1) + "\n")


def movement(data: dict, history: dict, today: dt.date) -> str | None:
    """Attach each team's change since about a week ago, in place. Returns the date compared to."""
    days = sorted(d for d in history.get("days", {}) if d < today.isoformat())
    if not days:
        return None
    target = (today - dt.timedelta(days=HISTORY_DAYS)).isoformat()
    earlier = [d for d in days if d <= target]
    base = earlier[-1] if earlier else days[0]
    before = history["days"][base]
    for t in data["teams"]:
        prior = before.get(t["team"])
        if prior is None:
            t["change"] = round(t["pField"], 4)                 # new to the list: from (about) zero
            t["seedChange"] = None
            continue
        t["change"] = round(t["pField"] - prior[0], 4)
        t["seedChange"] = (round(prior[1] - t["expectedSeed"], 2)
                           if prior[1] is not None and t["expectedSeed"] is not None else None)
    return base


def compare(data: dict, actual) -> dict | None:
    """The frozen projection against the field the committee picked."""
    if actual is None or actual.empty:
        return None
    real = actual.set_index("team")
    projected = {f["team"]: f for f in data["projected"]["field"]}
    named = sorted(set(projected) & set(real.index))
    rows = []
    for team in sorted(set(projected) | set(real.index), key=lambda t: (real["seed"].get(t, 99), t)):
        p, in_real = projected.get(team), team in real.index
        rows.append({"team": team, "projected": p["seedLine"] if p else None,
                     "actual": int(real.at[team, "seed"]) if in_real else None,
                     "region": real.at[team, "region"] if in_real else None,
                     "bidType": real.at[team, "bidType"] if in_real else (p["bidType"] if p else None)})
    both = [r for r in rows if r["projected"] and r["actual"]]
    errors = [abs(r["projected"] - r["actual"]) for r in both]
    at_large_real = set(real.index[real["bidType"] == "at-large"])
    at_large_proj = {t for t, f in projected.items() if f["bidType"] == "at-large"}
    return {
        "fieldSize": int(len(real)), "named": len(named),
        "atLarge": {"named": len(at_large_real & at_large_proj), "of": len(at_large_real)},
        "missedOut": sorted(set(projected) - set(real.index)),     # projected in, left out
        "missedIn": sorted(set(real.index) - set(projected)),      # left out of the projection, picked
        "seedExact": sum(e == 0 for e in errors), "seedWithinOne": sum(e <= 1 for e in errors),
        "seedCompared": len(errors), "seedMAE": round(sum(errors) / len(errors), 2) if errors else None,
        "rows": rows,
    }


def _actual_field(season: int):
    """The real field, re-read from the feed until it's complete (it fills in as games are scheduled)."""
    from ..bracket import history
    from ..ingest import bb_bracket

    bb_bracket.ncaa_tournament(season, refresh=True)
    return history.field(season)


def build(payload: dict, data_dir: Path, *, today: dt.date | None = None, as_of: str | None = None,
          season: int | None = None, field_size: int | None = None, sims: int | None = None,
          settings: dict | None = None, actual_fn=None) -> dict | None:
    """The page data, or None when there should be no page.

    ``today`` and ``as_of`` replay a past date (the pages were designed that way, on 2024-25 as a
    76-team field); left alone, this is the live, daily build.
    """
    settings = settings or live.load_settings()
    today = today or dt.date.today()
    season = season or live.season_for(today)
    saved_path, history_path = data_dir / "bracketology.json", data_dir / "bracketology_history.json"
    start, sunday = live.start_date(season, settings), live.selection_sunday(season, settings)

    if today < start:
        return None
    if today > sunday:
        saved = _load(saved_path)
        if saved.get("season") != season:
            return None
        saved["frozen"] = True
        result = saved.get("result")
        if not result or result["fieldSize"] < saved["fieldSize"]:
            try:
                result = compare(saved, (actual_fn or _actual_field)(season))
            except Exception as exc:  # noqa: BLE001 - the frozen projection is still worth showing
                print(f"  bracketology: real field not readable yet ({exc})")
            if result:
                saved["result"] = result
                _save(saved_path, saved)
        return saved

    data = live.build(season, as_of, settings, sims or settings.get("sims", 5000), field_size)
    history = _load(history_path)
    if history.get("season") != season:
        history = {"season": season, "days": {}}
    data["comparedTo"] = movement(data, history, today)
    data.update({"updated": today.isoformat(), "selectionSunday": sunday.isoformat(), "frozen": False,
                 "startsOn": start.isoformat()})
    history["days"][today.isoformat()] = {t["team"]: [t["pField"], t["expectedSeed"]] for t in data["teams"]}
    _save(history_path, history, compact=True)
    _save(saved_path, data)
    return data
