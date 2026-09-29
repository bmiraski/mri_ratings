"""The Chaos Meter's data: freeze each week's pregame odds, score the week once it is final, and
never revise either.

Two things here follow patterns this codebase already has, for the same reason in both cases:
a number computed after the fact from hindsight is not the number that was on the board when it
mattered.

*The pregame cache* (``chaos_pregame.json``) is ``tracker.update_log``'s forward-log pattern:
a game's stored probability is written once, before kickoff, and a game whose kickoff has already
passed when this first sees it is never assigned one there after the fact.

*The frozen archive* (``chaos_history.json``) is ``simdata.build``'s "a week that is over is never
recomputed" pattern: once a week's games are all played it is scored once, marked ``final``, and
left alone forever - including its percentile, which drifts as more weeks are archived and would
otherwise quietly rewrite history every time a wilder week showed up later.

A game the pregame cache never saw is not simply dropped, though. ``reconstruct_current_season``
walk-forwards this season's own ratings the same way ``tracker.reconstruct`` does for the public
record page - as of the end of the previous week, never touching the game's own result - so a week
that finished before this feature's own launch still gets scored honestly instead of freezing
forever as "not enough games." A game only ends up genuinely missing (counted in a week's
``missingPregame``) when even that reconstruction has nothing to go on, which in practice means the
previous week's ratings themselves are not available.

Fallout - how much a week reshuffled the playoff picture - is not computed here at all. It is
already sitting in ``sim_history.json``'s week-over-week playoff odds; ``fallout_for_week`` just
reads the delta.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
from scipy.stats import norm

from ..betting import board as board_module
from ..ingest import cfbd, registry
from ..ratings import chaos, history, mri2, priors

PREGAME_PATH_NAME = "chaos_pregame.json"
HISTORY_PATH_NAME = "chaos_history.json"


def _canonical(frame: pd.DataFrame) -> pd.DataFrame:
    """CFBD's feed names a team however it likes ("Miami"); the ratings this module joins against
    are keyed by the registry's canonical spelling ("Miami (FL)") - the same mismatch ``tracker.py``,
    ``slate.py``, ``simdata.py`` and ``hfa.py`` all guard against on their own schedule input, and
    this module must too, or a real favourite whose raw name doesn't match its rated one prices as
    a replacement-level unknown instead of itself."""
    frame = frame.copy()
    for column in ("team1", "team2"):
        frame[column] = [registry.resolve(n, n) for n in frame[column]]
    return frame


def _candidate_games(slate: dict) -> dict[int, dict]:
    """Every in-scope game on the slate - both-FBS, played or not, plus FCS opponents - keyed by
    its real game id, deduplicated the way a game could otherwise appear in both ``days`` and
    ``results`` across two different builds of the same week."""
    candidates: dict[int, dict] = {}
    for day in slate.get("days", []):
        for game in day.get("games", []):
            candidates[game["id"]] = game
    for game in slate.get("results", []):
        candidates[game["id"]] = game
    for game in slate.get("fcs", []):
        candidates[game["id"]] = game
    return candidates


def freeze_pregame(year: int, slate: dict | None, path: Path, *, now: dt.datetime | None = None) -> dict:
    """Store each in-scope game's pregame win probability once, before kickoff, never again.

    A game already stored keeps that value forever, even if the model's since changed its mind.
    A game whose kickoff has already passed the first time this runs is never assigned one after
    the fact - it just never enters ``games``, and a caller scoring the week counts it as missing
    rather than pricing it with hindsight.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    data = json.loads(path.read_text()) if path.exists() else {}
    if data.get("season") != year:
        data = {"season": year, "games": {}}

    if not slate:
        return data

    schedule = cfbd.games(year, completed_only=False).set_index("game_id")
    for gid, game in _candidate_games(slate).items():
        key = str(gid)
        if key in data["games"]:
            continue
        if game.get("played"):
            continue
        kickoff = schedule.loc[gid, "start_date"] if gid in schedule.index else None
        if kickoff and pd.Timestamp(kickoff) <= pd.Timestamp(now):
            continue
        data["games"][key] = {"homeWinProbability": game["homeWinProbability"],
                               "loggedAt": now.strftime("%Y-%m-%dT%H:%MZ")}

    text = json.dumps(data, indent=1, sort_keys=True)
    if not path.exists() or path.read_text() != text:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return data


def fallout_for_week(sim_history: dict, week: int) -> dict | None:
    """How much a week reshuffled the playoff picture, straight from ``sim_history.json``'s
    already-computed week-over-week odds. None if either week is missing (week 1, or no
    simulation history yet)."""
    weeks = sim_history.get("weeks", {})
    this_week, prior_week = weeks.get(str(week)), weeks.get(str(week - 1))
    if this_week is None or prior_week is None:
        return None

    deltas = {team: values[0] - prior_week[team][0]
              for team, values in this_week.items() if team in prior_week}
    moved = sum(abs(d) for d in deltas.values()) / 2.0
    ranked = sorted(deltas.items(), key=lambda kv: kv[1])
    return {
        "moved": round(moved, 4),
        "gained": [{"team": t, "change": round(d, 4)} for t, d in reversed(ranked[-3:]) if d > 0],
        "lost": [{"team": t, "change": round(d, 4)} for t, d in ranked[:3] if d < 0],
    }


def load_history(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"seasons": {}}


def save_history(path: Path, data: dict) -> None:
    text = json.dumps(data, indent=1, sort_keys=True)
    if not path.exists() or path.read_text() != text:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def reconstruct_current_season(year: int, schedule: pd.DataFrame, weekly: pd.DataFrame) -> dict[int, float]:
    """Every played game's raw pregame home win probability, walked forward from ratings as of the
    end of the previous week - the same idea as ``tracker.reconstruct`` - falling back to the
    preseason prior for any team missing from that snapshot.

    That fallback matters more than it looks: ``weekly`` only carries a team once it has actually
    played, so a real FBS team on a bye in week 1 is just as "missing" from a week-2 snapshot as a
    true FCS opponent is. Treating both cases the same way - pricing the missing side at the worst
    rated FBS team's floor, the way a genuine FCS opponent is priced elsewhere on the site - turned
    a 21.5-point favorite on a bye into a near-certain underdog the first time this ran for real.
    ``priors.for_season``'s prior already knows the difference (a real, regressed prior for an FBS
    team, replacement level only for a team the prior doesn't recognize at all), so it is the right
    fallback for every team, not just week 1's.
    """
    schedule = _canonical(schedule)
    played = schedule[schedule["played"]]
    if played.empty:
        return {}
    teams = sorted(set(schedule["team1"]) | set(schedule["team2"]))
    fbs = sorted(t for t in teams if registry.is_fbs(t))
    preseason = priors.for_season(year, board_module._previous_season(year), teams, fbs)

    out: dict[int, float] = {}
    for week, games in played.groupby("block"):
        week = int(week)
        if week > 1:
            rated = weekly[weekly["week"] == week - 1] if "week" in weekly.columns else pd.DataFrame()
            power = rated.set_index("team")["power"] if not rated.empty else pd.Series(dtype=float)
            home_field = float(rated["home_field"].iloc[0]) if not rated.empty else mri2.DEFAULT_HOME_FIELD_PRIOR
        else:
            power, home_field = pd.Series(dtype=float), mri2.DEFAULT_HOME_FIELD_PRIOR

        def rating(team: str) -> float:
            if team in power.index:
                return float(power[team])
            return float(preseason.get(team, mri2.REPLACEMENT_PRIOR))

        for g in games.itertuples():
            predicted = rating(g.team2) - rating(g.team1) + (0.0 if g.neutral else home_field)
            out[int(g.game_id)] = float(norm.cdf(predicted / board_module.SIGMA))
    return out


def archived_slate_pregame(docs_dir: Path, year: int, week: int) -> dict[int, float]:
    """Every game's genuinely frozen-at-the-time pregame win probability from the slate archive
    (``docs/slate/<year>-week-<n>.json``), if one exists for this week.

    This is the authoritative source for a week that finished before ``chaos_pregame.json`` ever
    captured it - it is the exact number the Slate page itself showed before kickoff, saved once by
    ``slatearchive.snapshot`` and never touched again, not a number recomputed from today's ratings.
    That distinction is not academic: CFBD's feed for a past week can shift slightly as it is
    re-fetched (a late-arriving box score, a corrected classification), which nudges a from-scratch
    walk-forward recomputation away from what the model actually said at the time - the same "wilder
    than X% of weeks" a reader already saw on the archived page must still say wilder than X%, not a
    different number recomputed after the fact. Falls back to nothing (an empty dict, letting the
    caller reach for ``reconstruct_current_season`` instead) for a week the archive never saved,
    e.g. before the archive feature had a week to freeze.
    """
    from . import slatearchive

    path = slatearchive.archive_path(docs_dir, year, week)
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out: dict[int, float] = {}
    for day in data.get("days", []):
        for g in day["games"]:
            out[g["id"]] = g["homeWinProbability"]
    for g in data.get("results", []):
        out[g["id"]] = g["homeWinProbability"]
    return out


def _score_games(games: pd.DataFrame, pregame: dict, calibration: dict | None, week: int,
                  archived: dict[int, float] | None = None,
                  reconstructed: dict[int, float] | None = None) -> tuple[dict, int, bool]:
    """Score one week's completed games against their stored pregame probabilities.

    Falls back, in order, to ``archived`` (the slate archive's own frozen number for the week, when
    one exists) and then ``reconstructed`` (a from-scratch walk-forward, for a week the archive
    never saved). Returns the ``chaos.week_score`` result, how many completed games had no
    probability from any source, and whether the week needed either fallback for any of its games.
    """
    stored = pregame.get("games", {})
    archived = archived or {}
    reconstructed = reconstructed or {}
    rows, missing, used_fallback = [], 0, False
    for g in games.itertuples():
        entry = stored.get(str(g.game_id))
        if entry is not None:
            p_raw = entry["homeWinProbability"]
        elif g.game_id in archived:
            p_raw = archived[g.game_id]
            used_fallback = True
        elif g.game_id in reconstructed:
            p_raw = reconstructed[g.game_id]
            used_fallback = True
        else:
            missing += 1
            continue
        p_home = chaos.apply_recalibration(p_raw, week, calibration)
        rows.append({"game_id": g.game_id, "home": g.team2, "away": g.team1,
                      "home_win_prob": p_home, "home_won": bool(g.pts2 > g.pts1),
                      "home_score": g.pts2, "away_score": g.pts1})
    scored = chaos.week_score(pd.DataFrame(rows)) if rows else {
        "games": 0, "upsets": None, "expectedUpsets": None, "z": None, "shocks": [],
        "note": "not_enough_games"}
    return scored, missing, used_fallback


def ranking_population(history_data: dict, *, bowls: bool) -> list[float]:
    """Every already-final, non-burn-in week's ``z``, split into the pool a regular-season or
    championship-week entry ranks against and the pool a "Bowls" entry ranks against instead
    (spec §3/§5 - a bowl season is only ever compared to other bowl seasons)."""
    population = []
    for season_entry in history_data.get("seasons", {}).values():
        if season_entry.get("burnIn"):
            continue
        for entry in season_entry["weeks"].values():
            if entry.get("z") is None:
                continue
            if (entry.get("label") == cfbd.POSTSEASON_LABEL) == bowls:
                population.append(entry["z"])
    return population


def finalize_current_season(year: int, weekly_games: pd.DataFrame, weekly: pd.DataFrame, pregame: dict,
                             calibration: dict | None, history_path: Path, sim_history_path: Path,
                             *, docs_dir: Path | None = None, now: dt.datetime | None = None) -> dict:
    """Score and freeze every week of ``year`` that is now fully complete and not already final.

    ``weekly_games`` is the season's full schedule (``cfbd.games(year, completed_only=False)``,
    with a ``block`` column from ``cfbd.sequence``); ``weekly`` is the week-by-week ratings table
    (``sitedata.weekly_ratings``). For a game the live pregame cache never saw, ``docs_dir`` (the
    site's published output root) is checked first for that week's own frozen slate archive - the
    genuinely-before-kickoff number, preferred over ``reconstruct_current_season``'s from-scratch
    recomputation, which can drift as the underlying schedule data is refreshed. A week already
    frozen is never touched again, whatever ``pregame``/``calibration`` say now - the same guarantee
    ``simdata.build`` gives ``sim_history.json``. Its percentile is computed once, against the
    archive as it stood at that moment, and stored - never recomputed as later, wilder weeks are
    added (spec §5: "percentiles drift, and the archive shouldn't").
    """
    weekly_games = _canonical(weekly_games)
    data = load_history(history_path)
    season_key = str(year)
    burn_in = year in history.BURN_IN_SEASONS
    season_entry = data["seasons"].setdefault(season_key, {"burnIn": burn_in, "weeks": {}})
    sim_history = json.loads(sim_history_path.read_text()) if sim_history_path.exists() else {}
    reconstructed = reconstruct_current_season(year, weekly_games, weekly)

    for week, games in weekly_games.groupby("block"):
        week = int(week)
        week_key = str(week)
        if week_key in season_entry["weeks"]:
            continue
        if not games["played"].all():
            continue  # not final yet - leave it for a future build

        archived = archived_slate_pregame(docs_dir, year, week) if docs_dir else {}
        scored, missing, used_fallback = _score_games(games, pregame, calibration, week, archived, reconstructed)
        entry = {**scored, "missingPregame": missing, "final": True, "reconstructed": used_fallback}
        is_bowls = (games["season_type"] != "regular").any()
        if is_bowls:
            entry["label"] = cfbd.POSTSEASON_LABEL
        fallout = fallout_for_week(sim_history, week)
        if fallout is not None:
            entry["fallout"] = fallout

        if burn_in or entry["z"] is None:
            entry["percentile"] = None
        else:
            population = ranking_population(data, bowls=is_bowls)
            percentile = chaos.percentile_rank(entry["z"], population)
            entry["percentile"] = round(percentile, 2) if percentile is not None else None

        season_entry["weeks"][week_key] = entry

    save_history(history_path, data)
    return data


def build(year: int, slate: dict | None, weekly: pd.DataFrame, data_dir: Path,
          docs_dir: Path | None = None) -> dict | None:
    """Everything the site needs for the Chaos Meter this build.

    ``weekly`` is ``sitedata.weekly_ratings(year)`` - already computed once for the rest of the
    build - passed through to ``reconstruct_current_season``'s fallback. ``docs_dir`` is the site's
    published output root (``docs/``), where ``finalize_current_season`` looks for each week's own
    frozen slate archive before falling back to reconstruction.

    Always freezes this week's pregame odds first - that costs nothing and should never wait on
    the historical archive existing. Returns None (and does nothing else) if ``chaos_history.json``
    doesn't exist yet - the one-off backfill, ``scripts/build_chaos_history.py``, hasn't been run -
    so the site still builds without a Chaos panel or ``chaos.html``, exactly like every other
    optional feature here when its data is missing.
    """
    pregame = freeze_pregame(year, slate, data_dir / PREGAME_PATH_NAME)

    history_path = data_dir / HISTORY_PATH_NAME
    if not history_path.exists():
        return None

    calibration = chaos.load_calibration()
    schedule = cfbd.games(year, completed_only=False).copy()
    schedule["block"] = cfbd.sequence(schedule)

    archive = finalize_current_season(year, schedule, weekly, pregame, calibration,
                                       history_path, data_dir / "sim_history.json", docs_dir=docs_dir)

    current = current_reading(schedule, int(slate["week"]), pregame) if slate else None
    return {"season": year, "current": current, "archive": archive}


def current_reading(games: pd.DataFrame, week: int, pregame: dict) -> dict | None:
    """The in-progress week's live "so far" reading for the Slate panel - recomputed every build,
    never written to ``chaos_history.json``. None if the week's games haven't been played at all
    yet (nothing "so far" to show beyond the pregame expectations, which the panel gets from the
    slate directly)."""
    games = _canonical(games)
    this_week = games[games["block"] == week]
    played = this_week[this_week["played"]]
    stored = pregame.get("games", {})
    expected_upsets = sum(
        min(p, 1 - p) for p in (
            e["homeWinProbability"] for gid, e in stored.items()
            if int(gid) in set(this_week["game_id"])
        )
    )
    if played.empty:
        return {"partial": True, "gamesPlayed": 0, "expectedUpsets": round(expected_upsets, 2)}

    scored, missing, _ = _score_games(played, pregame, None, week)
    return {"partial": True, "gamesPlayed": len(played), "missingPregame": missing,
            "expectedUpsets": round(expected_upsets, 2), **scored}
