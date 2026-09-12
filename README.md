# MRI

Rebuild of the MRI computer rating system for college football (basketball to follow).

## Status

- **MRI Classic** — the 2003-2019 Excel formula, ported to Python and verified to
  reproduce all 17 archived seasons exactly (max deviation ~5e-13, top-25 order
  identical in every year).
- **MRI 2.0** — not yet built.

## Layout

    src/mri/ingest      data readers (Excel archive, CFBD API)
    src/mri/ratings     classic.py (frozen), mri2.py (in progress)
    src/mri/betting     model line vs market, backtest
    src/mri/export      site + xlsx output
    scripts/            build entry points
    tests/              validation gates

## Running

    PYTHONPATH=src python3 scripts/build_archive.py
    PYTHONPATH=src python3 -m pytest tests/ -q
