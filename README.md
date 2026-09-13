# MRI Ratings

A computer rating system for college football, rebuilt. Originally run as an
Excel workbook from 2000 through 2019; this is the Python version.

## Status

| Piece | State |
|---|---|
| **MRI Classic** | Done. The original spreadsheet formula, ported and verified to reproduce all 17 archived seasons exactly. |
| **Team registry** | Done. 138 FBS teams for 2026 with alias resolution back to the historical workbooks. |
| **CFBD ingest** | In progress. |
| **MRI 2.0** | Not started. |
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

**MRI 2.0** (in design) solves ratings and schedule strength simultaneously
via a ridge-regularized fit on game margins, and splits the output into a
*Résumé* rating for ranking and a *Power* rating, in points, for prediction.

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
