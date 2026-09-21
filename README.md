# MRI Ratings

A computer rating system for college football, rebuilt. Originally run as an
Excel workbook from 2000 through 2019; this is the Python version.

## Status

| Piece | State |
|---|---|
| **MRI Classic** | Done. The original spreadsheet formula, ported and verified to reproduce all 17 archived seasons exactly. |
| **Team registry** | Done. 138 FBS teams for 2026 with alias resolution back to the historical workbooks. |
| **MRI 2.0** | Done and validated. Beats Classic by 1.7 points of straight-up accuracy across 2003-2019. |
| **CFBD ingest** | Done. 2020-2026 bridged from the API, cached on disk. |
| **Rankings site** | Done. 157 pages in `docs/`, served by GitHub Pages at mri.mira.ski. |
| **Weekly refresh** | Done. GitHub Action rebuilds and commits three times a week. |
| **Betting module** | Done. Verdict: it does not beat closing lines. See `docs/betting.html`. |
| **Excel export** | Done. `exports/` carries a live workbook per season, 2020-2026. |

## The two ratings

**MRI Classic** is the 2018 formula, frozen:

```
MRI = 25 x Win%  +  25 x OppWin%  +  10 x OppOppWin%  +  sum(game points)
    +  5 x z(RushYds/G)  +  5 x z(PassYds/G)
    +  7 x -z(YdsAllowed/G)  +  3 x z(TurnoverMargin)
```

where each game contributes `min(35, margin) x OppWin% x OppOppWin%` for a win
and `max(-35, margin) x OppLoss% x OppOppLoss%` for a loss.

It is kept unchanged as a baseline and for continuity with the published
historical rankings. Its known limits — one-level-deep strength of schedule,
a pooled FCS opponent, raw yardage that rewards tempo, no prior and no
home-field term — are what MRI 2.0 addresses.

**MRI 2.0** turns every game into one equation,

```
margin  =  rating_home - rating_away + home_field
```

and solves the whole season at once, so a team's rating depends on its
opponents' ratings, which depend on theirs. Strength of schedule stops being a
separate statistic and becomes a property of the solution. Ratings are shrunk
toward a prior carried from last season, which is what makes a Week 3 ranking
publishable instead of noise.

It publishes two numbers rather than one:

| | Question | Units | Use |
|---|---|---|---|
| **Power** | How good is this team? | points vs an average team | predictions, spreads |
| **Résumé** | What has it earned? | wins above what an average team would manage against the same schedule | rankings, playoff arguments |

2019 is the clean illustration: Ohio State rates the higher Power (+34.3), but
LSU carries the better Résumé (+9.1 wins), and LSU is the team that went 15-0
and won the title.

### Does it actually work?

Walk-forward across all 17 seasons, fitting on the games played so far and
scoring the games that come next:

| | MRI 2.0 | MRI Classic |
|---|---|---|
| straight-up accuracy | **73.8%** | 71.9% |
| margin error (MAE) | 13.0 pts | n/a - Classic has no point scale |
| Brier score | 0.177 | n/a |

A +1.9 point edge, winning 13 of 17 seasons, paired t = 4.1 over 85 windows.
The gap is widest early in the year (+3.1 points at the 40% mark), which is the
prior doing its job.

Hyperparameters were searched on 2003-2013 and the margin is reported on
2014-2019, which took no part in the search: +2.1 points there.

## Validation

`tests/test_classic.py` requires the Python port to reproduce every rating
published in the 2003-2019 workbooks. Current worst-case deviation across all
17 seasons is ~5e-13, with identical top-25 ordering in every year.

One subtlety worth knowing: Excel's `SUMIF` and `VLOOKUP` match text
case-insensitively, and the archive depends on it — the 2009 log spells Boise
State two different ways and the spreadsheet still totalled it as one team. All
team lookups go through a normalized match key so spelling variants collapse the
way they always did.

## Layout

```
src/mri/ingest      archive reader, team registry, CFBD client
src/mri/ratings     classic.py (frozen), mri2.py (in progress)
src/mri/betting     model line vs market, backtest, tracker.py (the public record)
src/mri/sim         season simulation: conference races, committee, 12-team field, bracket
src/mri/export      site + xlsx output
scripts/            build entry points
data/archive/       the original .xls workbooks, 2003-2019
data/parquet/       normalized game logs and recomputed ratings
tests/              validation gates
```

## Running

```bash
pip install -r requirements.txt
PYTHONPATH=src python3 scripts/build_archive.py   # workbooks -> parquet
PYTHONPATH=src python3 -m pytest tests/ -q
```

### Preseason priors

MRI 2.0 starts each season from a prior. `mri/ratings/priors.py` builds it as a regression on
last season's rating, roster talent (the 247 composite) and returning production, with the
service academies' talent treated as unmeasured and teams new to FBS keeping the old rule.
Coefficients live in `data/prior_model.json`. To refit after a season ends:

```bash
PYTHONPATH=src python3 scripts/fit_prior.py        # refit; prints out-of-sample error
PYTHONPATH=src python3 scripts/backtest_priors.py  # grade it game by game; feeds the method page
PYTHONPATH=src python3 scripts/build_current.py    # rebuild the season-to-season chain
```

Talent and returning production are fetched once per season and cached in `data/raw`. If either
is missing the old prior is used and the build carries on.

### College GameDay forecast

`gameday.html` guesses where ESPN's College GameDay will be for the weeks not yet announced.
`src/mri/gameday/` holds a choice model (which game each week does the show pick, given the
rankings that week?) fitted on 2014-2025 stops in `data/gameday_locations.json`, and a forecast
that plays out the season and asks the model each run. To keep it current:

- **Each Monday**, add the week's announced stop to `announced2026` in `data/gameday_locations.json`
  (`week`, `date`, `city`, `teams`, `host`; names as CollegeFootballData spells them).
- **After each season**, add the year's stops to `seasons` in the same file, then
  `PYTHONPATH=src python3 scripts/fit_gameday.py` and `scripts/backtest_gameday.py`.

The history was transcribed from NCAA.com's list of GameDay locations; a test checks every entry against
the schedule (right teams, right day, right host), so a typo fails the build.

### Basketball preseason priors

Basketball's prior (`mri/ratings/bb_priors.py`) is a regression on last season's rating and the roster:
returning win shares, incoming win shares (transfers) and the freshman class. There are two versions. Until
the schools post a season's rosters the API returns none, and each team gets the version that needs only
last season, the draft and the recruiting class; teams switch to the roster version one at a time as their
rosters appear, with no action needed. `data/bb_prior_model.json` holds the coefficients.

After each season ends (and `CURRENT_SEASON` in `bb_registry.py` is bumped):

```bash
PYTHONPATH=src python3 scripts/build_bb_players.py   # player stats, recruits, draft -> data/parquet
PYTHONPATH=src python3 scripts/fit_bb_prior.py       # refit the coefficients
PYTHONPATH=src python3 scripts/backtest_bb_priors.py # grade it game by game; feeds the method page
PYTHONPATH=src python3 scripts/build_basketball.py   # rebuild the season-to-season chain
```

### Heisman odds (model and forecast built; page next)

**The record.** `data/heisman_voting.json` holds every finalist 2005-2025 (the winner is certain; the Trust lists the
others in an order that is sometimes alphabetical, so a finish is recorded only when a result was published).
`data/parquet/player_seasons.parquet` is every notable FBS player since 2009 (regular-season counting stats;
the feed has holes in 2009-2011, so everything is fitted on 2012 on). `data/parquet/player_weekly.parquet` is
season-to-date offensive totals at weeks 3, 5, 7, 9, 11 and 13 of every season, for the backtest.

**The final-vote model** (`mri/heisman/final_model.py`, `scripts/fit_heisman.py`). A conditional logit over each
season's ~65 candidates (top 20 QBs, 20 RBs and 25 receivers on top-45 teams; defenders are not candidates - nobody
who was mainly a defender has won since 1997). Four features, all relative to that year's field: rank by
production, rank within his position, how far above the field's average he is, and his team's résumé-heavy rank.
Efficiency (PPA), record, position and last year's finalist were tried and did not help on fourteen seasons, so
they are out. Leaving each season out in turn, it names the winner 64% of the time and has him in its top three
86% of the time; "the most productive player on a top-five team" manages 50%.

**The forecast** (`mri/heisman/forecast.py`, `project.py`, `live.py`). Play the rest of the season 10,000 times with the
season engine (which now takes an optional `observe` callback for each run's finish), project each candidate's
finished stat line as his shrunk rate times the games his team has left times a ratio drawn from what actually
happened to candidates at that point of every other season (`data/heisman_ratios.json`; injuries and regression are in
it), score the field, and count. Player output and team results are drawn independently - the known simplification.
`scripts/backtest_heisman.py` grades it on 2013-2025 at six points in each season, leaving the season out.

- A single hand-edit is needed each December: add the new season's finalists (and any published points) to
  `data/heisman_voting.json`, then rerun the four scripts below.
- Each year: `build_player_history.py` (about 90 calls), `build_player_weekly.py` (about 250; resumable),
  `fit_heisman.py`, `backtest_heisman.py`. The raw responses are large and ignored by git.

### Season simulation, slate and record

`scripts/build_site.py` also builds three football pages, each isolated so a failure
leaves the rankings published and the page off rather than stale:

- **Simulation** (`simulation.html`) - `mri/sim/season.py` plays the rest of the schedule
  10,000 times. `scripts/calibrate_committee.py` chose the committee proxy (70% Resume,
  30% Power) against `data/cfp_final_rankings.json`; `scripts/backtest_simulation.py`
  grades the probabilities against history and writes `site/data/sim_backtest.json`, which
  the method page reads. Re-run both by hand when the model changes. Weekly odds are
  kept in `site/data/sim_history.json`; a finished week is never recomputed.
- **Slate** (`slate.html`) - every game this week with the model's line, the market's, and
  the playoff stakes from the simulation.
- **Record** (on `betting.html`) - a walk-forward reconstruction of the season, and an
  append-only forward log in `site/data/picks.json`. A logged pick is never rewritten; only
  its result is added after the game. Do not edit that file by hand.

The rankings endpoint of the CFBD API must be called with a `week`: without one, older
seasons return polls that do not belong to the week they are labelled with.

## Data

Historical game logs come from the original workbooks in `data/archive/`.
Current-season data comes from the [College Football Data API](https://collegefootballdata.com).


## Deployment

The site is a folder of files. `docs/` is committed to the repo and GitHub Pages
serves it directly, so the host never runs a build and never holds a credential.

    Settings -> Pages -> Deploy from a branch -> main / docs

`docs/CNAME` is written by the build rather than by hand: GitHub creates that
file when you set a custom domain, and since the generator rewrites the whole
output directory it would otherwise be deleted on the next build and quietly
take the domain down. Change the domain in `CUSTOM_DOMAIN` in
`src/mri/export/site.py`.

`docs/.nojekyll` stops GitHub running the output through Jekyll, which would
skip anything whose name starts with an underscore.

### Weekly refresh

`.github/workflows/weekly.yml` rebuilds the ratings Monday, Thursday and
Saturday, runs the tests, and commits `docs/` if anything changed. It needs one
repository secret:

    Settings -> Secrets and variables -> Actions -> New repository secret
    CFBD_API_KEY = your key from collegefootballdata.com/key

Locally, put the same key in `.env` at the repo root (gitignored).


## Excel export

`scripts/export_workbooks.py` writes one workbook per season into `exports/`,
laid out like the 2003-2019 originals so an old file and a new one sit side by
side without translation. 2020-2025 are the seasons that were never run.

The Classic sheets are **live**: Games holds the results, Team Data aggregates
them with the original SUMIF web and named ranges, and the MRI column is a
formula over those aggregates. Correct a score and the season re-rates.

The MRI 2.0 sheet is a **snapshot**. That rating solves every game in the season
at once as a regularized least-squares system, which a spreadsheet cannot
express, so it is written as values and the About sheet says so.

Every export is recalculated with LibreOffice and then checked against the
Python implementation - the sheet's own formulas must reproduce `classic.compute`
to within 1e-9. A clean recalculation only proves the formulas evaluate; that
check proves they compute the right thing.
