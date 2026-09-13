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
src/mri/betting     model line vs market, backtest
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
