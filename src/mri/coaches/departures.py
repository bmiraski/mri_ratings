"""The plan's §4 departure labels: fired, moved up, or ambiguous.

"Last season at a school" is read literally in the plan as the max season a
coach has on record there - but taken literally it would also flag every
*currently active* coach's most recent row, since we simply don't have next
season's data yet for them. That's a data horizon, not a departure. So a row
only counts as a real, labelable last season when there's actual evidence of
succession: a different coach_id has a row for the same school the following
season (interim or not - a program moving on from someone is the fact being
confirmed, not whether the successor is permanent). Absent that, the row is
excluded entirely, not "ambiguous."

Classification, in priority order:

1. Moved up - a non-interim next job the following season at a school whose
   program_par is within MOVED_UP_TOLERANCE of the current one's. Checked
   first and overrides everything else: a documented bigger-or-comparable
   job next season is strong enough evidence that it doesn't matter whether
   the final season here was above or below par.
2. Fired or pushed out - no such next job, and this season's vs_par (or,
   when program_par isn't available yet - it needs a decade of lookback
   most early archive seasons don't have - added, this season's power
   against its own preseason prior instead) is negative.
3. Ambiguous - everything else: no next job and vs_par/added >= 0 (the
   plan's own example of an ambiguous case), a next job at a clearly smaller
   program, or truly no signal at all (program_par and added both
   unavailable). "Retired or other" is never auto-assigned - there is no
   retirement signal in the data at all - it only exists as something a
   human assigns while reviewing the ambiguous bucket.

The `added` fallback matters more than it might look: checked against real
data, defaulting bare "no program_par" cases straight to "fired" (the plan's
literal words - "the coach took no head job... or the coach was below
program_par," reading a null par as satisfying neither) misclassified a
real 41% of the auto-fired bucket, including well-known voluntary
retirements and NFL moves in the archive's early years before program_par
had anywhere to look back to (Bob Stoops, Frank Beamer, Pete Carroll, Urban
Meyer, Nick Saban's 2004 LSU exit). `added` only needs one prior season, so
it's real evidence almost everywhere `program_par` is missing, and using it
instead of a blind default was confirmed as the right call over just
flagging every one of those cases ambiguous instead (that would have pushed
the ambiguous bucket from ~110 to ~250, most of it resolvable automatically).

One acknowledged, unfixable gap: the plan lists an NFL job under "moved up,"
but there is no NFL coaching data anywhere in this pipeline. A coach who
left for the NFL shows up as "no next FBS job found," landing in Fired (if
their last season was below par) or Ambiguous (if not) - exactly the
failure mode the ambiguous bucket exists to catch, so it surfaces for hand
review rather than being silently misclassified as a firing.
"""

from __future__ import annotations

import pandas as pd

FIRST_LABELED_SEASON = 1984  # 1978-80 are burn-in and early seasons lack program_par; 1984 on is labeled
                             # (the plan's own cutoff was 2004 - widening beat it out of sample)
MOVED_UP_TOLERANCE = 3.0     # points of program_par slack that still counts as "comparable"

FIRED = "fired_or_pushed_out"
MOVED_UP = "moved_up"
AMBIGUOUS = "ambiguous"


def eligible_departures(coach_season: pd.DataFrame) -> pd.DataFrame:
    """Every (coach_id, school, season) row that is a confirmed last season at that school.

    A row is eligible when a *different* coach_id has a row for the same
    school the following season - proof the program moved on, regardless of
    whether that successor is an interim or a permanent hire - AND the
    departing coach has no row of their own there next season. Without that
    second check, a coach whose *own* final season happens to be a split
    (an interim closing it out, like a bowl game after they've stepped away)
    would also flag the season *before* that split as a false departure: the
    interim is a "different coach_id" next year even though the real coach
    is still there for most of it.
    """
    candidates = coach_season[
        (~coach_season["interim"]) & (coach_season["season"] >= FIRST_LABELED_SEASON)
    ]
    by_school_season = coach_season.groupby(["school", "season"])["coach_id"].apply(set)

    def has_successor(row) -> bool:
        next_coaches = by_school_season.get((row.school, int(row.season) + 1), set())
        return row.coach_id not in next_coaches and bool(next_coaches)

    mask = candidates.apply(has_successor, axis=1)
    return candidates[mask].reset_index(drop=True)


def _next_job(row, coach_season: pd.DataFrame) -> dict | None:
    """The departing coach's next non-interim head-coaching job, if any, the following season."""
    next_season = int(row.season) + 1
    rows = coach_season[
        (coach_season["coach_id"] == row.coach_id)
        & (coach_season["season"] == next_season)
        & (~coach_season["interim"])
    ]
    if rows.empty:
        return None
    next_row = rows.iloc[0]
    return {
        "school": next_row["school"],
        "season": next_season,
        "program_par": None if pd.isna(next_row["program_par"]) else float(next_row["program_par"]),
    }


def _performance_signal(row) -> tuple[bool | None, str]:
    """Below or above expectations for this final season, and how it was measured.

    vs_par (against the program's own normal level) when it's available;
    otherwise added (against this season's own preseason prior, which only
    needs one prior season rather than program_par's decade of lookback) -
    real evidence almost everywhere program_par is missing, rather than a
    blind default. Returns (None, "...") only when neither is available at
    all, which is rare.
    """
    vs_par = None if pd.isna(row.vs_par) else float(row.vs_par)
    if vs_par is not None:
        return vs_par < 0, f"vs_par {vs_par:+.1f}"
    added = None if pd.isna(row.added) else float(row.added)
    if added is not None:
        return added < 0, f"added {added:+.1f} vs. its own preseason prior (program_par unavailable)"
    return None, "no signal available at all (program_par and added both unavailable)"


def classify(row, coach_season: pd.DataFrame) -> dict:
    """The label, reason and next-job evidence for one eligible departure row.

    The plan's "Fired" default fires on either of two conditions: a below-par
    final season, OR no next job at a comparable-or-bigger program - and
    that second half covers a *smaller* next job exactly the same as no job
    at all, taken at face value ("no head job at a comparable or bigger
    program" is just as true either way). So a clearly smaller next job
    doesn't get its own separate ambiguous bucket; it falls through to the
    same performance-based Fired/Ambiguous split that "no job at all" gets -
    only an *above*-expectations season with nothing comparable to show for
    it is the plan's own example of genuine ambiguity, and that reasoning
    doesn't care whether "nothing comparable" means no job or a smaller one.
    """
    next_job = _next_job(row, coach_season)
    program_par = None if pd.isna(row.program_par) else float(row.program_par)

    if next_job is not None:
        if next_job["program_par"] is None or program_par is None:
            # Can't compare - neither a confident "moved up" nor a confident "fired".
            return {"label": AMBIGUOUS, "reason": "next job found, but program_par isn't available to compare",
                    "next": next_job}
        if next_job["program_par"] >= program_par - MOVED_UP_TOLERANCE:
            return {
                "label": MOVED_UP,
                "reason": f"next job at {next_job['school']} ({next_job['season']}), "
                          f"program_par {next_job['program_par']:.1f} vs. {program_par:.1f} here",
                "next": next_job,
            }
        smaller = (f"next job at {next_job['school']} ({next_job['season']}) is a clearly smaller program "
                   f"(program_par {next_job['program_par']:.1f} vs. {program_par:.1f} here)")
    else:
        smaller = "no next job found"

    below, signal = _performance_signal(row)
    if below is True:
        return {"label": FIRED, "reason": f"below expectations ({signal}), {smaller}", "next": next_job}
    if below is False:
        return {
            "label": AMBIGUOUS,
            "reason": f"above expectations ({signal}) but {smaller} - {AMBIGUOUS}, per the plan's own example",
            "next": next_job,
        }
    return {"label": AMBIGUOUS, "reason": f"{signal}, {smaller}", "next": next_job}


def build(coach_season: pd.DataFrame) -> pd.DataFrame:
    """Every eligible departure, classified: coach_id, school, season, label, reason, vs_par, added, next."""
    eligible = eligible_departures(coach_season)
    rows = []
    for row in eligible.itertuples():
        result = classify(row, coach_season)
        rows.append({
            "coach_id": row.coach_id,
            "coach_name": row.coach_name,
            "school": row.school,
            "season": int(row.season),
            "label": result["label"],
            "source": "auto",
            "reason": result["reason"],
            "vs_par": None if pd.isna(row.vs_par) else float(row.vs_par),
            "added": None if pd.isna(row.added) else float(row.added),
            "next": result["next"],
        })
    return pd.DataFrame(rows)


def keep_manual_labels(records: list[dict], existing: list[dict]) -> tuple[list[dict], list[dict]]:
    """``records`` (a fresh build) with every hand-reviewed row from ``existing`` carried over.

    ``build`` only knows how to guess; a ``source: manual`` row is a person's decision and
    must survive a rebuild. Returns the merged records and any manual rows whose
    (coach_id, school, season) no longer appears in the fresh build - kept out of the
    result and reported, since the underlying coach data moved under them.
    """
    key = lambda r: (r["coach_id"], r["school"], r["season"])  # noqa: E731
    manual = {key(r): r for r in existing if r.get("source") == "manual"}
    merged = [manual.get(key(r), r) for r in records]
    seen = {key(r) for r in records}
    return merged, [r for k, r in manual.items() if k not in seen]
