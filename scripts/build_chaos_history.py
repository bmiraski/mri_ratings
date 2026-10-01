"""One-off historical backfill for the Chaos Meter: fit the recalibration and freeze every week.

Every completed season from 1978 (the first the rating chain now reaches, per the MRI 2.0
historical backfill) through last season is walked forward with ``hfa.walk_forward`` - the same
per-game, per-block-refit engine the per-stadium home-field work already validated - to get each
game's *pregame* win probability, honestly reconstructed from only what was known before it was
played. Those raw probabilities are then recalibrated per week-of-season band (``chaos.
fit_recalibration``) and every week is scored and frozen into ``site/data/chaos_history.json``.

This is not part of the nightly build. A full walk-forward refit over 48 seasons is not nightly-
build weight, and more importantly a frozen week's ``z``/``percentile`` must never silently drift
because a nightly refit nudged the recalibration by noise - the same reason ``data/
prior_model.json`` is fit by ``scripts/fit_prior.py`` once, not every night. Re-run this only to
deliberately refresh the recalibration (e.g. after several more seasons accumulate).

1978-1980 (``history.BURN_IN_SEASONS``) are cold-started at a flat prior and are known to run too
extreme, so they are scored and archived like any other week but excluded from the recalibration
fit, the self-check, and both ranking populations - see ``chaos.py``'s module docstring.

Refuses to write ``chaos_history.json`` if the self-check doesn't pass (mean z within 0.1 of 0,
std within 0.1 of 1 overall; each week-of-season band within 0.25) - this is the executable form
of the spec's "check that it worked."

Run:  PYTHONPATH=src python3 scripts/build_chaos_history.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import json  # noqa: E402

from mri.betting import board  # noqa: E402
from mri.export import chaosdata, slatearchive  # noqa: E402
from mri.export.slate import _eastern  # noqa: E402
from mri.ingest import cfbd  # noqa: E402
from mri.ratings import chaos, hfa, history  # noqa: E402

LAST_SEASON = cfbd.current_season() - 1  # the backfill covers completed seasons only
SEASONS = range(1978, LAST_SEASON + 1)

DOCS_DIR = ROOT / "docs"
SITE_JSON = ROOT / "site" / "data" / "site.json"

# The live site never converts a margin to a probability with a block's own fitted sigma - slate.py,
# board.py and tracker.py all use this fixed constant instead. That turns out to matter here: a
# thin training week (many teams with a single game so far) lets mri2.fit's per-block sigma come
# out implausibly small - a handful of points instead of ~16 - which turns an ordinary result into
# an astronomical surprisal once norm.cdf divides by it. Using the same fixed constant the rest of
# the codebase already trusts sidesteps that instability entirely, and keeps the walk-forward
# reconstruction consistent with how a probability is priced everywhere else.
SIGMA = board.SIGMA

TOLERANCE_OVERALL = 0.1
TOLERANCE_BAND = 0.25

HISTORY_PATH = ROOT / "site" / "data" / chaosdata.HISTORY_PATH_NAME


def _with_results(pregame: pd.DataFrame) -> pd.DataFrame:
    """Join each game's real season type, final score and kickoff time back onto ``walk_forward``'s
    predictions - it returns the pregame numbers a rating needs, not the outcome and kickoff a
    Chaos score and a historical slate page both need. The kickoff fields only end up used by the
    slate archive (see ``_write_slate_archives``); carrying them here costs nothing and keeps this
    one join the single source both consumers read from, rather than a second walk-forward run."""
    frames = []
    for season, group in pregame.groupby("season"):
        schedule = cfbd.games(int(season)).set_index("game_id")
        frames.append(group.join(
            schedule[["season_type", "pts1", "pts2", "start_date", "start_time_tbd"]], on="game_id"))
    return pd.concat(frames, ignore_index=True)


def _game_kickoff(start_date) -> tuple[str | None, str, str, str]:
    """(date, dateLabel, time, sort) for one game.

    Older CFBD seasons store a date-only kickoff as midnight UTC with no real time behind it -
    every game in a week can share the exact same instant. Converting that through slate.py's
    Eastern-time helper would both shift the calendar date back a day (midnight UTC is the
    previous evening in Eastern) and print a specific kickoff time that was never real. Detected
    by the UTC time itself being exactly midnight, and read straight off the UTC date instead.
    """
    if start_date is None or pd.isna(start_date):
        return None, "Date unknown", "TBD", "9999"
    ts = pd.Timestamp(start_date)
    if ts.hour == 0 and ts.minute == 0 and ts.second == 0:
        return ts.strftime("%Y-%m-%d"), ts.strftime("%a, %b %-d"), "TBD", "9999"
    when = _eastern(start_date)
    return (when.strftime("%Y-%m-%d"), when.strftime("%a, %b %-d"),
            when.strftime("%-I:%M %p").replace(" ", " ") + " ET", when.strftime("%H%M"))


def _slate_archive_games(games: pd.DataFrame) -> tuple[list[dict], dict[str, dict], list[dict]]:
    """One week's games as (FBS-vs-FBS game dicts for ``days``, ``finals`` keyed by game id as a
    string, FBS-vs-FCS game dicts for ``fcs``) - the same split ``slate.build()`` makes for the
    live week, and the same reason: an FCS opponent's game is real and counted, but is never one of
    the "this week's ranked matchups" cards.
    """
    day_games, fcs_games = [], []
    finals: dict[str, dict] = {}
    for g in games.itertuples():
        date, date_label, time, sort = _game_kickoff(g.start_date)
        entry = {
            "id": int(g.game_id), "week": int(g.week),
            "date": date, "dateLabel": date_label, "time": time, "sort": sort,
            "home": g.home_team, "away": g.away_team, "neutral": bool(g.neutral),
            "predicted": round(float(g.base_line), 1),
            "homeWinProbability": round(float(g.home_win_prob), 3),
            "played": True,
        }
        finals[str(int(g.game_id))] = {"home": int(g.pts2), "away": int(g.pts1)}
        (day_games if g.fbs_both else fcs_games).append(entry)
    return day_games, finals, fcs_games


def _bucket_days(entries: list[dict]) -> list[dict]:
    """Group games by date, in order - the same bucketing ``slate.build()`` does for the live
    week's ``days``, reused here rather than reimplemented."""
    days: list[dict] = []
    for g in sorted(entries, key=lambda g: (g["date"] or "9999", g["sort"])):
        if not days or days[-1]["date"] != g["date"]:
            days.append({"date": g["date"], "label": g["dateLabel"], "games": []})
        days[-1]["games"].append(g)
    return days


def _slate_archive_record(season: int, week: int, games: pd.DataFrame, teams_payload: list[dict],
                          ranks: dict | None = None) -> dict:
    day_entries, finals, fcs_entries = _slate_archive_games(games)
    is_bowls = bool((games["season_type"] != "regular").any())
    return {
        "season": season, "week": week,
        **({"label": cfbd.POSTSEASON_LABEL} if is_bowls else {}),
        "days": _bucket_days(day_entries), "results": [], "fcs": fcs_entries,
        "watch": [], "flagged": 0, "games": len(day_entries) + len(fcs_entries),
        "saved": f"{season}-week-{week}-{slatearchive.HISTORICAL_SUFFIX}",
        "finals": finals,
        # Ranks as of that week, not today's: see slatearchive.historical_teams.
        "teams": slatearchive.historical_teams(teams_payload, (ranks or {}).get((season, week), {})),
    }


def _write_slate_archives(pregame_scored: pd.DataFrame, docs_dir: Path, teams_payload: list[dict],
                          ranks: dict | None = None) -> None:
    """Write every week's historical slate archive, write-once, same as the live season's own
    ``slatearchive.snapshot`` - a week already on disk is never touched again. ``slate_archives()``
    (``site.py``) picks these up and renders them to HTML on the site's next normal build; nothing
    here needs to render HTML itself."""
    written = 0
    for (season, week), games in pregame_scored.groupby(["season", "week"]):
        season, week = int(season), int(week)
        if len(games) < chaos.MIN_GAMES:
            continue  # nothing worth a page for - matches the "not enough games" scoring rule
        path = slatearchive.archive_path(docs_dir, season, week)
        if path.exists():
            continue
        record = _slate_archive_record(season, week, games, teams_payload, ranks)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1))
        written += 1
    print(f"wrote {written} historical slate archive(s) to {docs_dir / 'slate'}")


def main() -> None:
    print(f"walking forward {SEASONS.start}-{SEASONS.stop - 1} ({len(SEASONS)} seasons)...")
    ranks: dict = {}
    pregame = hfa.walk_forward(SEASONS, ranks=ranks)
    pregame = _with_results(pregame).rename(columns={"seq": "week"})
    pregame["home_win_prob_raw"] = norm.cdf(pregame["base_line"] / SIGMA)
    pregame["home_won"] = pregame["pts2"] > pregame["pts1"]
    pregame["burn_in"] = pregame["season"].isin(history.BURN_IN_SEASONS)

    print(f"fitting recalibration on {(~pregame['burn_in']).sum()} non-burn-in games "
          f"({pregame['burn_in'].sum()} burn-in games excluded)...")
    fit_frame = pregame.loc[~pregame["burn_in"], ["season", "week", "home_win_prob_raw", "home_won"]] \
        .rename(columns={"home_win_prob_raw": "home_win_prob"})
    calibration = chaos.fit_recalibration(fit_frame)
    for band, params in calibration["bands"].items():
        print(f"  {band}: a={params['a']:.3f} b={params['b']:.3f} (n={params['n']})")

    pregame["home_win_prob"] = [chaos.apply_recalibration(p, int(w), calibration)
                                 for p, w in zip(pregame["home_win_prob_raw"], pregame["week"])]

    # Load rather than start fresh: this script only ever computes 1978-LAST_SEASON, but the live
    # site's own current-season weeks (finalize_current_season, a different write path) already
    # live in this same file. Starting from {} would silently erase them on save - every week in
    # SEASONS is about to be recomputed and overwritten anyway, so loading first costs nothing and
    # only protects seasons this script has no business touching.
    history_data = chaosdata.load_history(HISTORY_PATH)
    scored_rows = []
    for (season, week), games in pregame.groupby(["season", "week"]):
        season, week = int(season), int(week)
        games = games.rename(columns={"home_team": "home", "away_team": "away",
                                       "pts2": "home_score", "pts1": "away_score"})
        result = chaos.week_score(games)
        is_bowls = bool((games["season_type"] != "regular").any())
        entry = {**result, "missingPregame": 0, "final": True, "reconstructed": True}
        if is_bowls:
            entry["label"] = cfbd.POSTSEASON_LABEL

        season_entry = history_data["seasons"].setdefault(
            str(season), {"burnIn": season in history.BURN_IN_SEASONS, "weeks": {}})
        season_entry["weeks"][str(week)] = entry

        if result["z"] is not None:
            scored_rows.append({"season": season, "week": week, "z": result["z"],
                                 "burnIn": season in history.BURN_IN_SEASONS})

    scored = pd.DataFrame(scored_rows)
    report = chaos.self_check(scored)
    print(f"self-check: overall mean={report['overall']['mean']:.3f} std={report['overall']['std']:.3f}")
    for band, stats in report["byBand"].items():
        if stats["mean"] is not None:
            print(f"  {band}: mean={stats['mean']:.3f} std={stats['std']:.3f}")

    failures = []
    if abs(report["overall"]["mean"]) > TOLERANCE_OVERALL or abs(report["overall"]["std"] - 1) > TOLERANCE_OVERALL:
        failures.append("overall")
    for band, stats in report["byBand"].items():
        if stats["mean"] is None:
            continue
        if abs(stats["mean"]) > TOLERANCE_BAND or abs(stats["std"] - 1) > TOLERANCE_BAND:
            failures.append(band)
    if failures:
        print(f"self-check FAILED for: {', '.join(failures)} - refusing to write chaos_history.json")
        raise SystemExit(1)

    calibration["selfCheck"] = report
    chaos.save_calibration(calibration)
    print(f"wrote {chaos.CALIBRATION_PATH}")

    # Percentiles: a regular-season/championship week ranks against every other one; a "Bowls"
    # block ranks only against past bowl seasons (spec §3/§5). Burn-in weeks are never ranked.
    # Precomputed once - unlike the live, incremental freeze, every week here is known up front.
    #
    # Scoped to SEASONS (1978-LAST_SEASON) on both sides - the population these rank against, and
    # which weeks get re-stamped - because history_data may also carry the live site's own current
    # season, frozen incrementally by finalize_current_season with its own "percentile as of the
    # moment it froze" guarantee. This script re-running to refresh the historical recalibration
    # must not silently move a live week's already-frozen percentile, or drag a season this script
    # has never scored into what a historical week ranks against either.
    backfill_only = {"seasons": {k: v for k, v in history_data["seasons"].items() if int(k) in SEASONS}}
    regular_pop = chaosdata.ranking_population(backfill_only, bowls=False)
    bowls_pop = chaosdata.ranking_population(backfill_only, bowls=True)
    for season_key, season_entry in history_data["seasons"].items():
        if int(season_key) not in SEASONS:
            continue
        for entry in season_entry["weeks"].values():
            if season_entry["burnIn"] or entry["z"] is None:
                entry["percentile"] = None
                continue
            population = bowls_pop if entry.get("label") == cfbd.POSTSEASON_LABEL else regular_pop
            percentile = chaos.percentile_rank(entry["z"], population)
            entry["percentile"] = round(percentile, 2) if percentile is not None else None

    history_data["calibration"] = {"bands": calibration["bands"], "fittedThrough": calibration["fittedThrough"]}
    chaosdata.save_history(HISTORY_PATH, history_data)
    print(f"wrote {HISTORY_PATH} ({len(scored_rows)} scored weeks, "
          f"{sum(len(s['weeks']) for s in history_data['seasons'].values())} weeks total)")

    teams_payload = json.loads(SITE_JSON.read_text())["teams"] if SITE_JSON.exists() else []
    _write_slate_archives(pregame, DOCS_DIR, teams_payload, ranks)


if __name__ == "__main__":
    main()
