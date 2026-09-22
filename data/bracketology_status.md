# Bracketology — status

*Phase 1 (automatic bids) built 2026-09-22. See README's "NCAA Tournament
bracketology" section for the full write-up and current backtest numbers, and
`claude_bracketology-plan.md` in the project for the original plan.*

- `mri/ingest/bb_bracket.py` — NCAA and conference tournament games (seeds,
  rounds, regions) from the postseason feed.
- `mri/bracket/template.py` — a conference's bracket shape (byes, campus-hosted
  or not), inferred from its own games' calendar dates.
- `mri/bracket/simulate.py` — the Monte Carlo bracket engine.
- `mri/bracket/history.py` — ground truth (field, seeds, auto-vs-at-large,
  standings, power ratings) for any past season.
- `scripts/build_bracket_history.py` — the backtest; writes
  `data/bracket_autobid_backtest.json`.

**Not started:** Phase 2 (at-large selection and seeding), Phase 3 (joint
simulation, region placement), Phase 4 (the pages).
