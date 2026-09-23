"""The season-simulation page's data: run it, compare it with last week, keep it.

Two things beyond running the simulation are here because the page would be
dishonest without them.

*It remembers.* Each week's odds are saved, so the page can show what moved and
by how much. A week that is over is never recomputed: its numbers stay as they
were published, and only the week in progress is overwritten. A page that
quietly re-derived its own history with today's ratings would be showing what
the model wishes it had said.

*It backfills once.* The first run has no last week to compare against, so it
reconstructs earlier weeks from the ratings and the schedule as they stood then -
results after that week are hidden, not just ignored - and saves them as
history. That is a reconstruction, and the saved file says so.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..ingest import cfbd, registry
from ..sim import season


def _prepare(schedule: pd.DataFrame) -> pd.DataFrame:
    frame = schedule.copy()
    for column in ("team1", "team2"):
        frame[column] = [registry.resolve(n, n) for n in frame[column]]
    if "conference_game" not in frame.columns:
        frame["conference_game"] = frame["conf1"] == frame["conf2"]
    return frame


def championships(schedule: pd.DataFrame, conference: dict[str, str]):
    """Conference title games, once the calendar has them.

    Returns the schedule with any unplayed title game taken out - the simulation
    plays those itself - and a description of each, keyed by conference. Played
    ones stay in the schedule, where they count like any other result.

    A game is a title game if it is a conference game in the championship week.
    Two of those in one conference means the week is not what was assumed, and
    nothing is treated as a title game rather than guessing which is.
    """
    in_week = schedule[
        (schedule["week"] == season.CHAMPIONSHIP_WEEK) & schedule["conference_game"]
    ]
    entries: dict[str, dict] = {}
    seen: dict[str, int] = {}
    drop = []
    for row in in_week.itertuples():
        c = conference.get(row.team2)
        if not c or conference.get(row.team1) != c:
            continue
        seen[c] = seen.get(c, 0) + 1
        entries[c] = {
            "game_id": int(row.game_id),
            "a": row.team2, "b": row.team1, "neutral": bool(row.neutral),
            "played": bool(row.played),
            "a_won": bool(row.pts2 > row.pts1) if row.played else False,
        }
        if not row.played:
            drop.append(row.Index)
    for c, n in seen.items():
        if n > 1:
            entries.pop(c, None)
    drop = [i for i in drop if conference.get(schedule.loc[i, "team2"]) in entries]
    return schedule.drop(index=drop), entries


def _as_of(schedule: pd.DataFrame, week: int) -> pd.DataFrame:
    """The schedule as it stood at the end of ``week``: later results hidden."""
    frame = schedule.copy()
    frame["played"] = frame["played"] & (frame["week"] <= week)
    frame.loc[~frame["played"], ["pts1", "pts2"]] = float("nan")
    return frame


def _run(teams, schedule, home_field, conference, *, sims, track=None):
    schedule, entries = championships(schedule, conference)
    return season.simulate(
        teams, schedule, home_field=home_field, sims=sims, track=track,
        championships=entries or None,
    )


def build(year: int, payload: dict, weekly: pd.DataFrame, history_path: Path, *,
          sims: int = season.DEFAULT_SIMS) -> dict:
    teams = [{"team": t["team"], "power": t["power"], "conference": t["conference"]}
             for t in payload["teams"]]
    conference = {t["team"]: t["conference"] for t in teams}
    schedule = _prepare(cfbd.games(year, completed_only=False))
    latest = int(payload["week"])

    unplayed = schedule[~schedule["played"]]
    slate_week = int(unplayed["week"].min()) if not unplayed.empty else None
    track = []
    if slate_week is not None:
        slate = unplayed[unplayed["week"] == slate_week]
        track = [int(g) for g, a, h in zip(slate["game_id"], slate["team1"], slate["team2"])
                 if a in conference or h in conference]

    now = _run(teams, schedule, payload["homeField"], conference, sims=sims, track=track)

    history = _load(history_path, year)
    for week in range(1, latest):
        if str(week) in history["weeks"]:
            continue
        rated = weekly[weekly["week"] == week]
        if rated.empty:
            continue
        power = rated.set_index("team")["power"]
        then = [{**t, "power": float(power.get(t["team"], t["power"]))} for t in teams]
        result = _run(then, _as_of(schedule, week), float(rated["home_field"].iloc[0]),
                      conference, sims=sims)
        history["weeks"][str(week)] = _compact(result)
        history["reconstructed"] = sorted(set(history.get("reconstructed", [])) | {week})
    history["weeks"][str(latest)] = _compact(now)
    _save(history_path, history)

    before = history["weeks"].get(str(latest - 1), {})
    teams_out = {}
    for name, odds in now["teams"].items():
        prev = before.get(name)
        teams_out[name] = {
            **odds,
            "playoffChange": round(odds["playoff"] - prev[0], 4) if prev else None,
            "titleChange": round(odds["title"] - prev[1], 4) if prev else None,
        }
    return {
        "season": year,
        "week": latest,
        "slateWeek": slate_week,
        "sims": now["sims"],
        "fieldSize": now["fieldSize"],
        "teams": teams_out,
        "leverage": now["leverage"],
        "hasHistory": bool(before),
        "reconstructedWeeks": history.get("reconstructed", []),
    }


def _compact(result: dict) -> dict:
    return {t: [v["playoff"], v["title"], v["conferenceTitle"]] for t, v in result["teams"].items()}


def _load(path: Path, year: int) -> dict:
    if path.exists():
        data = json.loads(path.read_text())
        if data.get("season") == year:
            return data
    return {"season": year, "weeks": {}}


def _save(path: Path, history: dict) -> None:
    text = json.dumps(history, sort_keys=True, separators=(",", ":"))
    if not path.exists() or path.read_text() != text:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
