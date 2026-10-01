# Recency weighting: results

**Recommendation: do not ship, for either sport.** Down-weighting older games did not lower
walk-forward error in football or basketball, at any half-life, on any block I looked at. The code is
in and off by default; `FOOTBALL_PROFILE` and `BASKETBALL_PROFILE` are unchanged.

| | Best recency setting on the tuning block | Holdout MAE gain (needs ≥ 0.05) | Seasons improved (needs ≥ 2/3) | Accuracy / Brier not worse | Verdict |
|---|---|---|---|---|---|
| Football (tune 2003-13, hold 2014-19) | half-life 1.0, ridge ×0.75 | **+0.037** ✗ | 4 of 6 ✓ | accuracy −0.27 pt ✗, Brier −0.0006 ✓ | **Don't ship** |
| Basketball (tune 2022-24, hold 2025-26) | half-life 1.0, ridge ×0.75 | **−0.014** (worse) ✗ | 0 of 2 ✗ | accuracy +0.14 pt, Brier −0.0001 | **Don't ship** |

Gain is baseline MAE minus recency MAE, so positive is better. The tuning-block gains were +0.011
(football) and +0.004 (basketball): same sign as the football holdout, opposite to basketball's, and
both far inside noise (the basketball tuner already established that 0.025 points is noise).

Basketball's two-season holdout is too short to settle anything alone, so I added two second opinions
(below). They agree with it, so I call basketball "doesn't help" rather than "inconclusive".

## What was built

`mri2.fit(recency_half_life=None, season_games=None)`. Default `None` is the old solver. Age is
`(last row in slice − this row) / season_games`, weight `0.5 ** (age / half_life)`, so the half-life is
a fraction of the season. `season_games` is the full season's length; both walk-forward harnesses pass
it, so a half-life means the same at the 40% and the 80% cutoff. It is also passed through the whole
chain, including the end-of-season fit that primes next year.

Everything downstream of the solve, deliberately:

| Quantity | Treatment | Why |
|---|---|---|
| Ridge solve | weighted (`XᵀWX + P`, `XᵀWy + P·target`) | the feature |
| `scale` rescale | weighted regression of margin on edge | it corrects shrinkage bias in the weighted estimator, so it should read the same evidence |
| Home-field mean | weighted mean, prior kept at full strength against the *weighted* count | so weighting leaves the prior relatively heavier, the same effect as on the team ridge |
| `sigma` | reliability-weighted variance (reduces to ddof=1 for equal weights) | it prices future games, which look like recent ones |
| **Résumé** | **unweighted**, from a second solve | "what has this team earned over the full season". Power is "how good now". This costs one extra solve only when recency is on and `with_resume` is true |
| Efficiency (`adj_offense/defense`) | unweighted | descriptive, not in scope |

I did not ablate the weighted-versus-unweighted choice for `scale`, home field and `sigma`. Given the
results it would not change the conclusion.

**Default path is unchanged.** I checked bit for bit (exact `==` on power, home field, sigma, résumé and
efficiency) against the previous `mri2.py` over 32 fits: three football seasons at three cutoffs and
three parameter sets, a 2024 CFBD season, and basketball at three cutoffs. The walk-forward baseline
reproduces to four decimals before and after. `tests/test_recency.py` pins a golden snapshot from the
old solver, shows that flat weights reduce to the unweighted solve, that résumé ignores recency, and
that the harness passes the half-life through.

**Row order.** The solver assumes chronological rows. Verified: 2018-19 carry dates and have no
backward step (2018 has 38 undated rows, one contiguous block at 96-100% of the season, and the dated rows
on either side are in order). The undated 2003-2017
seasons are consistent with it: the first game flagged postseason sits at 93-95% of the season in all but
three. `current_games` and `pre2003_games` are chronological by *week* but not by start time within a
week (in `current_games` 41% of adjacent pairs step backward in start time, in `pre2003_games` 4%; 62 of those steps are by more than a week), so within-week
order there is not reliable. A week is about 7% of a season; the shortest half-life tested is 15%.

## Baselines (recorded 2026-10-01, before any change)

Football, 2003-2019, all five cutoffs (0.4-0.8), horizon 0.1:

| Block | Accuracy | MAE | Brier |
|---|---|---|---|
| All 17 seasons | 73.61% | 12.956 | 0.1772 |
| Tune 2003-13 | 73.73% | 12.692 | 0.1761 |
| Holdout 2014-19 | 73.39% | 13.440 | 0.1792 |

Against the older figures (73.8% / 13.0 / 0.177), MAE and Brier are the same to rounding and accuracy has
drifted down 0.2 points. Basketball (`BASKETBALL_PROFILE`, chain 2021-2026):

| Block | Accuracy | MAE | Brier |
|---|---|---|---|
| Tune 2022-24 | 70.35% | 9.1414 | 0.1918 |
| Holdout 2025-26 | 70.16% | 9.3018 | 0.1922 |

Per cutoff and per season are in `data/recency_baseline_football.csv` and
`data/recency_baseline_basketball.csv`. Football baseline MAE by cutoff, tune/holdout: 0.4: 13.09 /
13.63, 0.5: 12.14 / 13.96, 0.6: 12.60 / 13.30, 0.7: 12.91 / 12.78, 0.8: 12.71 / 13.53.

## Grids (tuning block, MAE; `data/tuning_recency_*.csv`)

Football, 2003-2013. Rows are half-life as a fraction of the season (∞ is unweighted); columns are ridge
× the profile's 4.0.

| half-life | ×0.5 | ×0.75 | ×1.0 | ×1.5 |
|---|---|---|---|---|
| 0.15 | 13.091 | 13.231 | 13.401 | 13.710 |
| 0.25 | 12.813 | 12.885 | 12.995 | 13.240 |
| 0.40 | 12.722 | 12.745 | 12.818 | 13.009 |
| 0.60 | 12.704 | 12.698 | 12.747 | 12.903 |
| 1.00 | 12.711 | **12.681** | 12.710 | 12.834 |
| ∞ | 12.754 | 12.695 | **12.692 (baseline)** | 12.765 |

Basketball, 2022-2024 (ridge ×8.0):

| half-life | ×0.5 | ×0.75 | ×1.0 | ×1.5 |
|---|---|---|---|---|
| 0.15 | 9.296 | 9.333 | 9.385 | 9.487 |
| 0.25 | 9.194 | 9.215 | 9.256 | 9.340 |
| 0.40 | 9.156 | 9.160 | 9.186 | 9.261 |
| 0.60 | 9.149 | 9.143 | 9.159 | 9.220 |
| 1.00 | 9.150 | **9.137** | 9.146 | 9.193 |
| ∞ | 9.157 | 9.142 | **9.141 (baseline)** | 9.169 |

The shape is the same in both sports. Error rises monotonically as the half-life shrinks; at the
long end it is flat. Lowering the ridge as the half-life shortens claws back some of the loss
(weighting does shrink the effective sample, so this is the effect the brief anticipated), but never to
the baseline. The "winner" in each sport (1.0, ×0.75) is barely weighted at all (a game one full
season old counts half) and sits within 0.004 to 0.011 of the baseline. Accuracy and Brier tell the same
story in the CSVs.

## Holdout, scored once

Football, 2014-2019: baseline (unweighted, ridge 4.0) vs half-life 1.0 with ridge 3.0.

| | Baseline | Recency | Change |
|---|---|---|---|
| Accuracy | 73.39% | 73.12% | −0.27 pt |
| MAE | 13.440 | 13.403 | −0.037 |
| Brier | 0.1792 | 0.1786 | −0.0006 |

By cutoff (MAE change, negative is better): 0.4 −0.013, 0.5 −0.049, 0.6 −0.099, 0.7 −0.045, 0.8 +0.023.
By season (MAE change): 2014 +0.032, 2015 +0.021, 2016 −0.068, 2017 −0.062, 2018 −0.071, 2019 −0.072.
By season (accuracy change): 2014 −0.7 pt, 2015 −0.5, 2016 −0.5, 2017 +0.5, 2018 0.0, 2019 −0.5.

Fails the 0.05 bar and the accuracy condition. The gain also appears only in 2016-19 (2014 and 2015 got worse) and the
tuning block saw just +0.011, so it looks more like a late-sample quirk than a signal. The check on 2021-25
below does not support it.

Basketball, 2025-2026: baseline vs half-life 1.0 with ridge 6.0.

| | Baseline | Recency | Change |
|---|---|---|---|
| Accuracy | 70.16% | 70.30% | +0.14 pt |
| MAE | 9.3018 | 9.3153 | +0.0135 (worse) |
| Brier | 0.1922 | 0.1921 | −0.0001 |

MAE change by cutoff: 0.4 +0.006, 0.5 +0.019, 0.6 +0.024, 0.7 +0.016, 0.8 +0.002. By season: 2025
+0.010, 2026 +0.017, so it is worse in both seasons and in 6 of 10 season-cutoff cells. Per-cutoff and per-season
detail for both sports is in `data/recency_holdout_football.csv` and `data/recency_holdout_basketball.csv`.

**Early versus late.** The hypothesis that recency hurts early (few games, weights shrink the effective
sample) and helps late is not borne out. In football the gain, such as it is, sits at the 0.5-0.7
cutoffs and vanishes at 0.8. In basketball recency is never better than baseline at any cutoff.

## Second opinions (post hoc, not used to choose anything)

These exist because two basketball seasons cannot support a decision. They are blocks the choice never
saw, and they run the whole grid, not just the chosen setting.

**Can basketball be extended? Yes.** `bb_classic_games.parquet` holds 22 seasons (2005-2026) from the
same API, with no ties. Neutral-site flags are missing before 2008 (0% flagged), then about 10-14%. I
used 2009-2020, chain from 2008, a 12-season block (`scripts/tune_recency.py basketball-extended`,
`data/recency_extended_basketball.csv`). Season 2020 ends at the March shutdown.

| half-life (ridge ×1.0) | MAE | Seasons better than baseline (of 12) |
|---|---|---|
| ∞ (baseline) | 8.678 | |
| 1.0 | 8.680 | 5 |
| 0.6 | 8.691 | 4 |
| 0.4 | 8.713 | 2 |
| 0.25 | 8.775 | 0 |
| 0.15 | 8.918 | 0 |

The best single cell in the whole grid is the unweighted baseline (8.678). The nominal winner from the
tuning block (1.0, ×0.75) scores 8.679, which is 0.0007 worse than baseline and better in 6 seasons of 12. Shorter half-lives are
strictly worse and never win a season. Taking this together with the holdout, basketball recency
weighting does not help.

**Football 2021-2025** (`football-modern`, `data/recency_modern_football.csv`): `current_games.parquet`,
a period and a data source the workbook archive does not cover. Baseline 12.987; half-life 1.0 ridge ×0.75
12.979; 1.0 ×1.0 13.035; 0.6 13.078; 0.4 13.150; 0.25 13.324; 0.15 13.685. Again no gain, and a clean
monotone penalty for shorter half-lives. Football's apparent holdout gain does not persist.

## Why it probably doesn't work here

Margins have a standard deviation of 14-16 points per game, and a team's true strength moves slowly
over a season relative to that noise. Each game dropped from the effective sample costs variance, and the
drift it removes is small. This is consistent with the shape of the grids (flat at long half-lives, steeply
worse at short ones) but I did not test it directly. A recency effect concentrated in particular events
(an injury to a quarterback, a mid-season coaching change) would not show up in an aggregate weight like
this one, and I did not test that either.

## Things noticed on the way

* Basketball's carry-forward prior comes from `evaluate_season`'s last cutoff fit (0.8 of the season),
  not an end-of-season fit. That is existing behaviour, unchanged, and it means the basketball chain is
  evaluated with a slightly stale prior. A half-life reaches it only through that fit.
* Bowls, checked against the cached CFBD games: the 2017 workbook is missing **all 42** FBS-vs-FBS
  bowl and playoff games (it ends at the regular season, 834 games), and 2016 is missing 3 of 41. 2004
  and 2005 are *not* missing bowls (all 28 are present); `mark_postseason` simply fails to flag them
  because teams had played only 11 games by then, so they are treated as home games for the second-listed
  team rather than neutral. This affects `mri2_history` and any archive-based result for those seasons.
* If production ever uses a half-life, `season_games` must be the scheduled season length, not the
  games played so far. Defaulting to the slice length makes the half-life shrink and grow with the
  calendar.

## Reproduce

```
PYTHONPATH=src python3 scripts/tune_recency.py football
PYTHONPATH=src python3 scripts/tune_recency.py basketball
PYTHONPATH=src python3 scripts/tune_recency.py basketball-extended   # supplementary
PYTHONPATH=src python3 scripts/tune_recency.py football-modern       # supplementary
```
