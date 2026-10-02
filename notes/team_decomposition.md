# "How this rating is built" on team pages

Branch `team-decomposition`. Brief: `~/Downloads/team-decomposition-brief.md`.
Scope: current-season football and basketball team pages. Archive and historical
pages (`team_season_page`, `gamelogs.py`) are out: pre-2020 seasons used Classic,
which has no ridge decomposition, and 2020+ archive fits would need a prior-chain
refit per season.

## The identity

`mri2.fit` solves a ridge problem. For a team with `n` games:

    (n + λ) · r = Σ_g [ m_g − s_g·h_g + r_opp(g) ] + λ · p

`fit` then rescales by `scale` and subtracts the anchor shift `c`, both linear, so
in published units:

    Power = [ Σ_g ( scale·(m_g − s_g·h_g) + R_opp(g) ) + λ·P' ] / (n + λ),   P' = scale·p − c

i.e. **Power = w·(avg margin counted + avg opponent) + (1 − w)·prior, w = n/(n+λ)**.
Each game adds `(scale·(m − s·h) + R_opp)/(n + λ)`; those plus the prior's share sum
to Power exactly. λ is 4 for football and 8 for basketball.

Code: `src/mri/ratings/decompose.py` (core, sport-agnostic),
`src/mri/export/decomposition.py` (payload shaping), `site.py` (page).

## Step 0 gate

Reconstructed every rated team's `R_i` from the identity on the latest real fit of
each sport, using only the solver's own quantities.

| | teams | games | max abs error |
|---|---|---|---|
| Football 2026, week 4 | 233 (138 FBS + 95 FCS) | 331 | **6.4e-14** |
| Basketball 2025-26, final week | 727 (365 D1 + 362 non-D1) | 6,300 | **1.0e-13** |

No fudge term, no "other adjustments". The core function also re-checks the
identity on every call (tolerance 1e-6) and raises if the games it is given are not
the ones the fit was run on.

## Traps

1. **Home field.** The identity needs the solver's coefficient, not the published
   one. They are far apart: football week 4 is 1.68 solver vs 2.80 published;
   basketball is 3.22 vs 2.57. `fit` now keeps `raw_home_field`. A test shows that
   using the published value makes the identity fail.
2. **Internals exposed, additively.** `Ratings.internals` (a frozen `FitInternals`):
   teams, prior vector, ridge, compression, scale, anchor shift, raw home field,
   compressed margins, neutral flags. Nothing else in `fit` changed. Proof: a golden
   file for a synthetic fit (`tests/golden/synthetic_fit.json`, generated from the
   unmodified code) and a before/after comparison of the real football and
   basketball fits, `assert_series_equal(check_exact=True)`: bit-identical.
3. **Recency weights and `pace_adjust` are not on main.** The brief says `fit` has
   both; on `main` it has neither. Recency weighting (`recency_half_life`,
   `check_connectivity`) lives only on the unmerged `tempo-adjustment` branch;
   `pace_adjust` and `tests/test_tempo.py` are not on any branch I could find. So I
   built on main and gave `FitInternals` two flags, `recency_weighted` and
   `pace_adjusted` (both False). `decompose` raises `DecompositionUnavailable` when
   either is set (tested). **Whoever merges either feature must set the flag in
   `fit`.** Until they do, the guard cannot fire. Separately, the 1e-6 identity
   check would catch a weighted fit anyway, since the reconstruction would miss.
4. **Same fit as the published power.** Nothing post-processes `power` after the
   fit: `weekly_ratings` filters to FBS (D1 for basketball), ranks, and `build`
   rounds to 2 decimals. But the final model was never exposed, so
   `weekly_ratings` / `build` take an optional `sink` that receives the last week's
   `(model, games)`; `build_full` hands it to `team_details`. The decomposition
   explains that object, not a refit. The payload's rounded power is used as the
   check: if parts miss it by more than 0.006 for a team, that team gets no section.
5. **Outsiders.** The opponent term has to be the solver's value for FCS / non-D1
   opponents. In football the pooled `Non D1A` team is not held fixed: only its
   *prior* is set to replacement; it is solved like any team, so the identity holds
   unmodified. The site payload only lists FBS/D1 teams, and the Results page used a
   stand-in (`min(listed power) − 8`) for everyone else. **That stand-in disagrees
   with the solver** (football: FCS opponents solve to −38.7 on average against the
   stand-in's −33.3). See "Decision for review" below.
6. **Zero games.** `w = 0`, `Power = P'`, no division by zero; the page says "no
   games yet, so this rating is the preseason prior alone". A team with zero games
   cannot appear in a fit (the team list comes from the games), so on the live site
   this only arises if a payload carries one. Covered by unit and page tests.

## Decision for review: "Schedule faced" moved

The brief requires the section's average opponent to equal "Schedule faced". The two
could not agree while "Schedule faced" used the stand-in for outsiders and the
identity needs the solved value. I changed `playedDifficulty` (only when a
decomposition exists) to the solver's average opponent. Nothing else moved: Expected,
Perf, best win / worst loss and "Schedule ahead" still use the old per-game
`opponentPower`.

Effect on the 2026 build I tested: football, 115 of 138 teams' "Schedule faced"
changes (up to 6.6 points, mean −1.25: schedules with FCS games get easier on the
page). Basketball: 247 of 365 change, max 1.4. This is a displayed number, not a
rating. **If you would rather leave it alone, the alternative is to show the old
stand-in in the section plus an explicit "outsider correction" line; I chose not to
because the brief rules out an unexplained term.** The one-line revert is the
`detail["playedDifficulty"] = ...` assignment in `decomposition.attach`.

"Schedule ahead" still uses the stand-in, and now sits next to a "Schedule faced"
computed differently. A follow-up could move it to solved ratings too.

## Rounding

Displayed parts sum to displayed Power by largest-remainder rounding at render time
(`site._round_to_total`), targeting the Power tile's own formatting:

* the three parts in the section sum to the Power tile;
* the per-game Adds column plus the prior row sums to the Power tile, and the prior
  row is the same number as the section's.

Per-game figures are therefore rounded "so the column adds up", which can nudge a
single game by 0.1 from its hover arithmetic (the hover shows two decimals). Tested
on every team of a synthetic payload at 2 and 5 rounds, and checked on **all 503
teams of the real 2026 football and basketball payloads: 0 mismatches**.

## Size

Football `site.json` (week 4 of 2026, 138 teams, 331 games): 650,851 → 749,388 bytes,
**+98.5 KB (+15.1%)**. It grows with games played, roughly linear, so expect about
+250 to 300 KB by season's end. Per game: `counted`, `oppRating`, `adds`; per team a
`decomposition` block.

Basketball details are not written to `bb.json` (they are rendered into pages only),
so the cost there is page weight. Football team pages: +3.0 KB each (1.12 → 1.54 MB
total, 138 pages). Basketball: +7.8 KB each (5.76 → 8.62 MB total, 365 pages), mostly
the per-game hover text. Cheapest saving if wanted: build the hover in a few lines of
JS from the numbers already in the row.

## What the pages look like

Football (Ohio State, week 4) and basketball (Duke, final) at 1200px; Ohio State at
375px; a synthetic 2-game page. Not committed (screenshots are in the session).

* **Desktop.** Under the tiles and highlight cards: "How this rating is built", the
  formula in one sentence with the real numbers ("After 4 games, w is 50%, so the
  preseason prior is 50% of this rating — mostly preseason, for now"), then a small
  table (Margin counted +29.7 avg × 50% = +14.8; Opponents faced −1.7 avg × 50% =
  −0.9; Preseason prior +34.1 × 50% = +17.1; Power +31.0) and the caveat that the
  opponent ratings come from the same model. Results gains an **Adds** column; a
  footer shows the prior row and Power, so the column visibly sums. Hover an Adds
  figure: "Margin +53 counts as +45.4. Opponent rated −25.3. (+45.4 −25.3) ÷ (4 + 4)
  ≈ +2.51".
* **Phone (375px).** The Results table first overflowed (Adds off-screen, Result
  wrapping). Tightened cell padding and header type at ≤520px and kept the score on
  one line; it now fits (scrollWidth = clientWidth) with Adds visible. The three-part
  table fits too, with numbers kept on one line.
* **Early season** (synthetic, 2 games): w = 33%, "the preseason prior is 67% of
  this rating — mostly preseason, for now"; parts +7.3, −1.6, +7.2 = +12.9, and the
  Adds column sums the same.

## Things that read confusingly

* **Adds looks small for big wins.** Ohio State's 56–3 win adds +2.5 while its Perf is
  −6.1. That is correct (a game's share is divided by n + λ, and the opponent term
  pulls against a weak opponent), but a reader may expect a blowout to "add" a lot.
  The Results hint now says a game can beat expectation and add little.
* **The prior is on the rating scale, not last year's number.** +34.1 for Ohio State
  is `scale·p − c` (football `scale` is about 1.37 this early), not last season's
  published rating. The hover says so, but it may still surprise.
* **Counted margin can exceed the raw margin early in a season.** The scale factor
  is above 1 while shrinkage is heavy, so a 53-point win counts as +45.4 (compression
  takes it down, the scale puts it back up). The "a 56-pt win counts as about 30"
  story is true for later in the season, not for week 4.
* The Results hint is now a long paragraph. It could move into a collapsible.
* A `cursor: help` was added to hover cells only (`.adds`, `.buildtable`), not
  sitewide.

## Left for later

* Set `recency_weighted` / `pace_adjusted` in `fit` when those features merge.
* "Schedule ahead" on the solved-rating basis, if you want the two schedule
  figures to be the same kind of number.
* The page-level test over the published payload is skipped until the daily build
  writes decompositions into `site/data/site.json`.

## Verification

* `pytest -m "not slow"`: 754 pass (+1 skipped here for the published payload), 3
  deselected, after removing a test-helper import that `test_site`'s import audit
  flagged.
* Not run: the daily build itself, and nothing in `docs/` or `data/parquet` was
  regenerated or committed. Renders went to a scratch directory.
