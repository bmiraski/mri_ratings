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

from mri.betting import board  # noqa: E402
from mri.export import chaosdata  # noqa: E402
from mri.ingest import cfbd  # noqa: E402
from mri.ratings import chaos, hfa, history  # noqa: E402

LAST_SEASON = cfbd.current_season() - 1  # the backfill covers completed seasons only
SEASONS = range(1978, LAST_SEASON + 1)

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
    """Join each game's real season type and final score back onto ``walk_forward``'s predictions -
    it returns the pregame numbers a rating needs, not the outcome a Chaos score needs."""
    frames = []
    for season, group in pregame.groupby("season"):
        schedule = cfbd.games(int(season)).set_index("game_id")
        frames.append(group.join(schedule[["season_type", "pts1", "pts2"]], on="game_id"))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    print(f"walking forward {SEASONS.start}-{SEASONS.stop - 1} ({len(SEASONS)} seasons)...")
    pregame = hfa.walk_forward(SEASONS)
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

    history_data = {"seasons": {}}
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
    regular_pop = chaosdata.ranking_population(history_data, bowls=False)
    bowls_pop = chaosdata.ranking_population(history_data, bowls=True)
    for season_entry in history_data["seasons"].values():
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


if __name__ == "__main__":
    main()
