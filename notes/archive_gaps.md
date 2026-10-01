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

Every residual falls into one of four causes. **No games are missing from any workbook.**

## 1. Transitional and mislisted FBS teams (convention, not error): 2004, 2007, 2013

The workbooks pool a team as "Non D1A" in its first or transitional year; CFBD's FBS list for that season includes it.

* **2004:** Florida Atlantic and Florida A&M are on CFBD's FBS list. The workbook files their games under "Non D1A".
  That is the right call (FAMU is FCS, and FAU played its first FBS season in 2005), and it explains the two
  zero-game teams in `connectivity_audit.md`. About 24 CFBD-only rows and 10 workbook-only rows are these games
  seen from the two sides.
* **2007:** Western Kentucky's 6 games; **2013:** Old Dominion's 7. Both were transitional programs.

No action: the workbook treatment is deliberate and consistent.

## 2. Score disagreements (8 games): the workbook and CFBD differ and I cannot say which is right

| Season | Game | Workbook | CFBD |
|---|---|---|---|
| 2003 | Cal at Kansas State | Cal 28, KSU 42 | Cal 7, KSU 10 (dated Aug 23) |
| 2003 | Arkansas State at Ole Miss | 0-41 | 0-55 |
| 2004 | Cincinnati v East Carolina | 30-19 | 24-19 |
| 2004 | Arizona State v Oregon | 28-10 | 28-13 |
| 2005 | Temple at Bowling Green | 7-70 | 7-69 |
| 2009 | Southern Miss at Kansas | 29-35 | 28-35 |
| 2010 | UCLA at Oregon | 16-60 | 13-60 |
| 2018 | Air Force v FAU; Texas State v Georgia State | 27-33; 40-31 | 26-33; 40-30 |

My recollection is that CFBD is right for UCLA-Oregon (60-13) and the workbook for Cal-Kansas State (42-28),
but I did not check any of these against an outside source. Each is a one-to-three-point difference except the
two 2003 games. They are worth correcting only with a source in hand.

## 3. A duplicate in the CFBD feed: 2008

App State at LSU appears twice in CFBD's 2008 games under two game ids. The workbook has it once (as "Non D1A"
13, LSU 41). Nothing is wrong in the workbook. `current_games` and `pre2003_games` have no duplicate games, so
the production tables are unaffected; only a raw 2008 CFBD pull would double-count it.

## Conclusion

The workbooks are complete. The only real defects are the eight score disagreements, and I have not changed any
of them.
