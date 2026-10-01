## Summary

**Was disconnection ever a real problem? Barely, and never in a way that touched published ratings.**
Across 49 football seasons (1978-2026) and 22 basketball seasons (2005-2026), exactly one *finished*
season has a strict orphan (football 2020, one team). Every other finished season's rated graph is a
single connected component. Zero-game roster teams turn up in a handful of seasons and are all
accounted for below. Nothing here suggests the ridge solve has been quietly mis-rating a
conference-sized island.

| | Finding | Real? |
|---|---|---|
| Football 2020 | **New Mexico State** is a strict orphan: two games, both against FCS opponents (Tarleton State and Utah Tech), played in Feb-Mar 2021, after the fall season. It sits at its prior. | Yes, and correct to flag |
| Football 2004 | Florida Atlantic and Florida A&M are on CFBD's FBS list for 2004 but have no games in the workbook under any spelling; FAU's first workbook games are 2005 (11 of them). | Roster-source artifact; not traced further |
| Basketball 2021 | Nine roster teams with zero games: the eight Ivy League schools (whose season was cancelled) and Maryland Eastern Shore (reason not checked). | Yes, a genuine absence |
| Basketball 2005-13 | Abilene Christian is on the API roster with zero games; its first games in the data are 2013-14. | Roster-source artifact |
| Football 2026 (in progress) | 9 FBS teams are outside the main component after Week 3: Coastal Carolina, Delaware, NC State, Vanderbilt, Virginia, West Virginia (one 6-team island), Houston, Oregon State, Texas Tech (a 3-team island). Each has played three games and none against the main body. | Early-season; expected to clear. Watch it. |

**No registry name mismatch was found** in any archive season: the checks that would surface one (a
rostered team with zero games, or a rated team with games but no rated opponent) only fire on the
cases above, none of which is a spelling problem. That is the useful negative result: the registry
and the feeds agree.

The loose view differs from the strict view only in two places: 2020 and 2026 (the same teams), and
basketball 2013 and 2014, where two non-D1 programs (Faith Baptist Bible and Grinnell; Crossroads College and Grinnell) form a two-node island of their own. That is
irrelevant to ratings, and is why the strict view exists.

## How long until a season connects

*Football.* From 2021 to 2025 every FBS team is connected to the main body by Week 3 and stays that
way (27-31% of the games). For the 2003-2019 workbooks, which have no weeks, the graph is orphan-free
from 24% to 32% of the games (about Week 4-5 of 14). The last team to join the main body varies by
year (Hawaii in 2000, 2001 and 2004; Pittsburgh in 1998; different teams otherwise).

*Basketball.* For 2014-2026 the rated graph is connected by Nov 16-27 and stays so (the latest is
Nov 27, 2013-14). But 2005-2013 had D1 teams that opened against non-D1 opponents for weeks - Alabama
A&M until Dec 18, 2005; New Orleans until Dec 15, 2010; Texas State until Dec 3, 2011 - and those
look the same as a real disconnect on any day before their first D1 game. That is why the policy
below uses Dec 15, not Nov 30.

## Policy implemented

* `mri2.fit` warns (`ConnectivityWarning`) and stores `ratings.connectivity`. It never raises: Week 1
  of every season is disconnected and the walk-forward fits thousands of slices.
* `scripts/build_current.py` and `scripts/build_basketball.py` fail the build for the **current season
  only**, once it is far enough along: football once Week 5 games exist; basketball once the latest
  game is on or after Dec 15. Finished seasons print a warning. They cannot fail, because 2020 and
  2021 have permanent, legitimate gaps.
* `MRI_CONNECTIVITY_ALLOW="Team A,Team B"` waves named teams through, for a team that truly has no games.
* `mri2_history.parquet` takes its 2020+ rows from `current_ratings.parquet`, so it is covered by the
  `build_current.py` check; its archive seasons are finished and only warn.
* As of 2026-10-01 the football check is **not yet enforced** (only Weeks 1-3 are in the data), so the
  nine 2026 orphans above print a warning and the build passes. If they are still there once Week 5
  games arrive, the build will fail and that is the intended behaviour.

## Notes on method

Slices are every 1% of a season's games (football) or every day (basketball), in row order. Row order
is chronological at week granularity in every source (see `notes/recency_weighting.md`), which is all
this audit needs. "Strict" counts only games between two rated teams; "zero-game" is a roster team
absent from the slice. The `orph@` and `zero@` columns are the picture at fixed checkpoints, and
`last_orphan_before` names who was last to join the main body.

