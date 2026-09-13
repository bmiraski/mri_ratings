# MRI Ratings

A computer rating system for college football, rebuilt. Originally run as an
Excel workbook from 2000 through 2019; this is the Python version.

## Status

| Piece | State |
|---|---|
| **MRI Classic** | Done. The original spreadsheet formula, ported and verified to reproduce all 17 archived seasons exactly. |
| **Team registry** | Done. 138 FBS teams for 2026 with alias resolution back to the historical workbooks. |
| **MRI 2.0** | Done and validated. Beats Classic by 1.9 points of straight-up accuracy across 2003-2019. |
| **CFBD ingest** | Blocked: `api.collegefootballdata.com` is not on the sandbox egress allowlist. |
| **Rankings site** | Not started. |
| **Betting module** | Not started. |

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
