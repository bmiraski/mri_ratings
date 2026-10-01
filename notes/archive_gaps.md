# Workbook vs CFBD: the unexplained gaps

Follow-up to the 2003-2019 archive repair. The "10-38 regular-season games a year missing" figure in the
earlier notes was an artifact of my own check: it matched on team names, so every game against a pooled
"Non D1A" opponent looked unmatched. Matched properly (FBS side plus the score, the non-FBS side pooled), the
workbooks and CFBD agree almost exactly. The method: for each season, compare CFBD's completed games that touch
an FBS team (FBS as CFBD lists it for that season) with the workbook's rows, as multisets of (team pair,
sorted score).

| Season | Workbook rows | CFBD FBS games | Workbook-only | CFBD-only |
|---|---|---|---|---|
| 2003, 2018 | 771, 884 | 771, 884 | 2 | 2 |
| 2004 | 707 | 730 | 12 | 35 |
| 2005, 2009, 2010 | 718, 808, 808 | same | 1 | 1 |
| 2007 | 798 | 792 | 6 | 0 |
| 2008 | 804 | 805 | 0 | 1 |
| 2013 | 855 | 848 | 7 | 0 |
| every other season | | | 0 | 0 |

Every residual falls into one of three causes. **No games are missing from any workbook.**

## 1. Transitional and mislisted FBS teams (convention, not error): 2004, 2007, 2013

The workbooks pool a team as "Non D1A" in its first or transitional year; CFBD's FBS list for that season includes it.

* **2004:** Florida Atlantic and Florida A&M are on CFBD's FBS list. The workbook files their games under "Non D1A".
  That is the right call (FAMU is FCS, and FAU played its first FBS season in 2005), and it explains the two
  zero-game teams in `connectivity_audit.md`. Of the 35 CFBD-only rows, 33 involve FAU or FAMU (mostly games
  against other FCS teams, which a workbook of FBS schedules would not hold anyway); the other two are score
  disagreements (section 2). Of the 12 workbook-only rows, 10 are FAU/FAMU games against FBS teams, filed as
  "Non D1A", and the same two are score disagreements.
* **2007:** Western Kentucky's 6 games; **2013:** Old Dominion's 7. Both were transitional programs.

No action: the workbook treatment is deliberate and consistent.

## 2. Score disagreements (9 games): resolved

The workbook and CFBD differ on nine games, and the maintainer confirmed the correct score for each. Three
were already right in the workbook; six are corrected in `data/archive_score_corrections.json`, which
`scripts/build_archive.py` applies before Classic and MRI 2.0 see the games (an entry that no longer matches
exactly one row, or that would change a winner, fails the build).

| Season | Game | Workbook had | Now | Verdict |
|---|---|---|---|---|
| 2003 | Cal at Kansas State | 28-42 | 28-42 | workbook was right (CFBD has 7-10) |
| 2003 | Arkansas State at Ole Miss | 0-41 | 0-55 | corrected |
| 2004 | Cincinnati v East Carolina | 30-19 | 24-19 | corrected |
| 2004 | Arizona State v Oregon | 28-10 | 28-13 | corrected |
| 2005 | Temple at Bowling Green | 7-70 | 7-69 | corrected |
| 2009 | Southern Miss at Kansas | 29-35 | 28-35 | corrected |
| 2010 | UCLA at Oregon | 16-60 | 13-60 | corrected |
| 2018 | Air Force v FAU | 27-33 | 27-33 | workbook was right (CFBD has 26) |
| 2018 | Texas State v Georgia State | 40-31 | 40-31 | workbook was right (CFBD has 30) |

Effect. Classic caps margins at 35, so only three corrections move it: 2004 (four teams, up to 1.19 index
points; five rank positions shift) and 2009 (two teams, up to 0.37; four rank positions shift). 2003, 2005 and
2010 are unchanged in Classic because the corrected margins are over the cap either way. MRI 2.0 compresses
rather than caps, so it moves a little in every season from 2003 to 2013, by at most 0.49 points on one team
(2004), through the priors. Classic's 2004 and 2009 now differ from Ben's published figures for those teams;
`archive_published.parquet` is untouched.

## 3. A duplicate in the CFBD feed: 2008

App State at LSU appears twice in CFBD's 2008 games under two game ids. The workbook has it once (as "Non D1A"
13, LSU 41). Nothing is wrong in the workbook. `current_games` and `pre2003_games` have no duplicate games, so
the production tables are unaffected; only a raw 2008 CFBD pull would double-count it.

## Conclusion

The workbooks are complete, and the six score errors in them are corrected.
