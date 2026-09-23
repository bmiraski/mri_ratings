"""The Hidden Heisman: each week's biggest game from a player outside the power conferences.

A Heisman page tradition from the old site, brought back. Every finished week, one player from the Group of
Five or an independent (not Notre Dame) whose stat line was the wildest of the week - and who the Heisman
odds are not already talking about: anyone above ``MAX_HEISMAN`` that week is left out, so the award goes to
someone voters are overlooking.

*The score* is yards and touchdowns, with passing yards worth less than rushing and receiving because there
are more of them: pass yards / 25 + 4 per passing touchdown + rush and receiving yards / 10 + 6 per touchdown.
A dual-threat game counts in full.

*The opponent* moves it: the score is scaled by how good the other team was entering the week, by MRI power
(MRI rates teams, not defences, so this stands in for "a good defence"). An average FBS opponent leaves it
alone; a top one is worth up to 30% more, a weak one or an FCS team as much as 15% less. Most of these games
are not against great teams, which is the point - it breaks ties toward the one that was.

*Rarity* is the raw stat line ranked against every FBS player-game since 2012 (``player_games.parquet``).

Each week is decided once, the first build after all its games are final, and saved to
``site/data/hidden_heisman.json``; later rating changes never rewrite a past winner. One API call per week.

*The season award* goes, once every regular-season game is final (conference championships included), to the
best non-power player of the whole year that the odds never took seriously: the most opponent-scaled points
summed over all his games, with the same exclusions as the weekly award, checked against the odds at the time.
Each week keeps its top ``KEEP_PER_WEEK`` non-power lines for this, which a season-long leader always makes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MAX_HEISMAN = 0.05
POWER = frozenset({"SEC", "Big Ten", "Big 12", "ACC"})
POWER_THROUGH_2023 = POWER | {"Pac-12"}          # the old Pac-12; the rebuilt one from 2026 is not a power league
POWER_INDEPENDENT = frozenset({"Notre Dame"})
FACTOR_PER_SD = 0.15
FACTOR_RANGE = (0.85, 1.30)
FCS_FACTOR = FACTOR_RANGE[0]
STATS = ("pass_yds", "pass_td", "rush_yds", "rush_td", "rec_yds", "rec_td")
KEEP_PER_WEEK = 120


def power_conferences(season: int) -> frozenset:
    return POWER_THROUGH_2023 if season <= 2023 else POWER


def score(row) -> float:
    g = row.get if isinstance(row, dict) else row.__getitem__
    return (g("pass_yds") / 25 + 4 * g("pass_td") + g("rush_yds") / 10 + 6 * g("rush_td")
            + g("rec_yds") / 10 + 6 * g("rec_td"))


def opponent_factor(opponent_power: float | None, mean: float, sd: float) -> float:
    if opponent_power is None or not np.isfinite(opponent_power) or sd <= 0:
        return FCS_FACTOR
    return float(np.clip(1 + FACTOR_PER_SD * (opponent_power - mean) / sd, *FACTOR_RANGE))


def _facts(winner: dict, week_rows: list[dict]) -> list[str]:
    """Things that are true of the winner's game against every FBS player that week, checked, not asserted."""
    facts = []
    tds = lambda r: r["pass_td"] + r["rush_td"] + r["rec_td"]                        # noqa: E731
    yards = lambda r: r["pass_yds"] + r["rush_yds"] + r["rec_yds"]                  # noqa: E731
    for label, value in (("total touchdowns", tds), ("total yards", yards),
                         ("rushing yards", lambda r: r["rush_yds"]), ("receiving yards", lambda r: r["rec_yds"]),
                         ("passing yards", lambda r: r["pass_yds"])):
        mine = value(winner)
        if mine <= 0:
            continue
        best = max(value(r) for r in week_rows)
        if mine >= best:
            ties = sum(value(r) == best for r in week_rows)
            number = f"{int(mine)} {label}"
            facts.append(f"{number}, the most by any FBS player this week" if ties == 1
                         else f"{number}, tied for the most by any FBS player this week")
        if len(facts) == 2:
            break
    return facts


def scored(week_rows: list[dict], conference: dict[str, str], season: int, opponent_power: dict[str, float],
           excluded: set[tuple[str, str]]) -> list[tuple[float, float, float, dict]]:
    """(opponent-scaled score, raw score, factor, row) for every eligible line, best first."""
    power_confs = power_conferences(season)
    values = np.array(list(opponent_power.values()), float)
    mean, sd = (float(values.mean()), float(values.std())) if len(values) else (0.0, 0.0)
    out = []
    for row in week_rows:
        team = row["team"]
        if conference.get(team) in power_confs or team in POWER_INDEPENDENT or (row["player"], team) in excluded:
            continue
        raw = score(row)
        factor = opponent_factor(opponent_power.get(row["opponent"]), mean, sd)
        out.append((raw * factor, raw, factor, row))
    return sorted(out, key=lambda x: -x[0])


def pick(week_rows: list[dict], conference: dict[str, str], season: int, opponent_power: dict[str, float],
         excluded: set[tuple[str, str]], history: np.ndarray) -> dict | None:
    """The week's winner from its FBS player lines, or None when nobody outside the power leagues qualifies."""
    ranked = scored(week_rows, conference, season, opponent_power, excluded)
    if not ranked:
        return None
    adjusted, raw, factor, row = ranked[0]
    ahead = int((history > raw).sum())
    opp_rank = sorted(opponent_power, key=lambda t: -opponent_power[t]).index(row["opponent"]) + 1 \
        if row["opponent"] in opponent_power else None
    return {
        "season": season, "week": int(row["week"]), "player": row["player"], "team": row["team"],
        "conference": conference.get(row["team"]), "opponent": row["opponent"], "opponentRank": opp_rank,
        "points": row.get("points"), "oppPoints": row.get("opp_points"),
        "line": {k: int(row[k]) for k in STATS},
        "score": round(raw, 1), "factor": round(factor, 3), "adjusted": round(adjusted, 1),
        "rarity": round(float((history >= raw).mean()), 5) if len(history) else None,
        "rank": ahead + 1, "of": int(len(history)),
        "facts": _facts(row, week_rows),
    }


def _history_scores(root: Path) -> np.ndarray:
    path = root / "data" / "parquet" / "player_games.parquet"
    if not path.exists():
        return np.zeros(0)
    table = pd.read_parquet(path)
    return (table["pass_yds"] / 25 + 4 * table["pass_td"] + table["rush_yds"] / 10 + 6 * table["rush_td"]
            + table["rec_yds"] / 10 + 6 * table["rec_td"]).to_numpy(float)


def season_award(season: dict, conference: dict[str, str], excluded: set[tuple[str, str]], year: int) -> dict | None:
    """The year's best non-power player the odds never took seriously: most opponent-scaled points over the season."""
    power_confs = power_conferences(year)
    totals: dict[tuple[str, str], dict] = {}
    for week, rows in season.get("candidates", {}).items():
        for r in rows:
            key = (r["player"], r["team"])
            if conference.get(r["team"]) in power_confs or r["team"] in POWER_INDEPENDENT or key in excluded:
                continue
            t = totals.setdefault(key, {"player": r["player"], "team": r["team"], "adjusted": 0.0, "games": 0,
                                        "line": {k: 0 for k in STATS}})
            t["adjusted"] += r["adjusted"]
            t["games"] += 1
            for k in STATS:
                t["line"][k] += r["line"][k]
    if not totals:
        return None
    best = max(totals.values(), key=lambda t: t["adjusted"])
    weeks_won = [int(w) for w, v in season.get("weeks", {}).items() if (v["player"], v["team"]) == (best["player"], best["team"])]
    runner = sorted((t for t in totals.values() if t is not best), key=lambda t: -t["adjusted"])[:4]
    return {"season": year, "player": best["player"], "team": best["team"], "conference": conference.get(best["team"]),
            "line": best["line"], "games": best["games"], "adjusted": round(best["adjusted"], 1),
            "weeksWon": sorted(weeks_won),
            "runnersUp": [{"player": t["player"], "team": t["team"], "adjusted": round(t["adjusted"], 1)} for t in runner]}


def build(year: int, payload: dict, weekly: pd.DataFrame, saved_path: Path, root: Path, *, fetch=None,
          schedule: pd.DataFrame | None = None) -> dict:
    """Decide every finished week that has no winner yet, and the season once it is over; save; return them."""
    from ..ingest import cfbd, players, registry

    fetch = fetch or players.game_rows
    saved = json.loads(saved_path.read_text()) if saved_path.exists() else {}
    season = saved.setdefault(str(year), {})
    season.setdefault("weeks", {})
    season.setdefault("candidates", {})

    if schedule is None:
        schedule = cfbd.games(year, completed_only=False)
    regular = schedule[schedule["season_type"] == "regular"] if "season_type" in schedule else schedule
    weeks = sorted(int(w) for w in regular["week"].unique())
    # Finished: every game played - or a later week already under way, so a cancelled game cannot hold one open.
    started = [w for w in weeks if regular[regular["week"] == w]["played"].any()]
    finished = [w for w in weeks if regular[regular["week"] == w]["played"].all() or any(s > w for s in started)]

    teams = {t["team"]: t for t in payload["teams"]}
    conference = {n: t["conference"] for n, t in teams.items()}
    heisman = payload.get("heisman") or {}
    excluded = {(p["player"], p["team"]) for p in heisman.get("players", []) if p.get("win", 0) > MAX_HEISMAN}
    history = _history_scores(root)

    def win_chance(player, team):
        return next((p["win"] for p in heisman.get("players", []) if (p["player"], p["team"]) == (player, team)), None)

    for week in finished:
        if str(week) in season["weeks"]:
            continue
        rows = [dict(r, team=registry.resolve(r["team"], r["team"]),
                     opponent=registry.resolve(r["opponent"], r["opponent"]) if r.get("opponent") else None)
                for r in fetch(year, week)]
        rows = [r for r in rows if r["team"] in teams]
        if not rows:
            continue
        entering = weekly[weekly["week"] == week - 1] if not weekly.empty else weekly
        opponent_power = (entering.set_index("team")["power"].astype(float).to_dict() if not entering.empty
                          else {n: float(t["power"]) for n, t in teams.items()})
        winner = pick(rows, conference, year, opponent_power, excluded, history)
        if winner:
            winner["heismanWin"] = win_chance(winner["player"], winner["team"])
            season["weeks"][str(week)] = winner
        # Every non-power line worth keeping, for the season award - before the weekly exclusions, which are
        # checked again, against the odds then, when the season is decided.
        season["candidates"][str(week)] = [
            {"player": r["player"], "team": r["team"], "adjusted": round(adj, 2), "line": {k: int(r[k]) for k in STATS}}
            for adj, _, _, r in scored(rows, conference, year, opponent_power, set())[:KEEP_PER_WEEK]]

    over = bool(weeks) and set(weeks) <= set(finished) and len(started) == len(weeks)
    if over and "award" not in season:
        award = season_award(season, conference, excluded, year)
        if award:
            award["heismanWin"] = win_chance(award["player"], award["team"])
            season["award"] = award

    saved_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path.write_text(json.dumps(saved, indent=1))
    winners = [season["weeks"][w] for w in sorted(season["weeks"], key=int)]
    past = [w for s, other in saved.items() if s != str(year) for w in other.get("weeks", {}).values()]
    awards = [other["award"] for s, other in saved.items() if other.get("award")]
    return {"season": year, "winners": winners, "latest": winners[-1] if winners else None, "past": past,
            "award": season.get("award"), "awards": awards}
