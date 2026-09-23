# Bracketology — status

*All four phases built 2026-09-22: automatic bids, at-large and seeding, the
joint simulation with seed lines and regions, and the pages. First live season:
2026-27, from Christmas.
See README's "NCAA Tournament bracketology" section for the full write-up and
current backtest numbers, and `claude_bracketology-plan.md` in the project for
the original plan.*

- `mri/ingest/bb_bracket.py` — NCAA and conference tournament games (seeds,
  rounds, regions) from the postseason feed.
- `mri/bracket/template.py` — a conference's bracket shape (byes, campus-hosted
  or not), inferred from its own games' calendar dates.
- `mri/bracket/simulate.py` — the Monte Carlo bracket engine.
- `mri/bracket/history.py` — ground truth (field, seeds, auto-vs-at-large,
  standings, power ratings) for any past season.
- `scripts/build_bracket_history.py` — the backtest; writes
  `data/bracket_autobid_backtest.json`.
- `mri/bracket/resume.py` — a team's Selection Sunday résumé (record, Quad 1-4,
  schedule strength).
- `mri/bracket/atlarge.py` — the composite score: at-large selection and
  seeding, fit against the real historical seed line.
- `scripts/build_atlarge_history.py`, `scripts/backtest_atlarge.py` — feature
  history and the fit/backtest; write `data/parquet/atlarge_history.parquet`,
  `data/atlarge_model.json`, `site/data/atlarge_backtest.json`.

- `mri/bracket/seeding.py` — true seed list to seed lines, 68- and 76-team
  formats (the 2027 Opening Round).
- `mri/bracket/regions.py` — region placement by the committee's published
  bracketing principles.
- `mri/bracket/joint.py` — the joint simulation: rest of season, standings,
  conference tournaments, automatic bids, at-large field and seeds, per world.
- `scripts/backtest_bracketology.py` — graded against every real field
  2011-2025 at three checkpoints; writes `data/bracketology_backtest.json`.
- `scripts/build_bracketology.py` — live projection to
  `site/data/bracketology.json` (Christmas to Selection Sunday).
- `data/bracketology_settings.json` — every conference simulates its
  tournament (the placeholder is retired; `autoBid` can bring it back for one
  conference); season-keyed tournament-format overrides for leagues realignment
  reshaped (2027: Pac-12, Mountain West, SWAC, Summit announced; WCC, UAC, ASUN
  provisional - replace when their brackets are published); calendar.

- `mri/bracket/live.py` — the live projection, shared by the daily build and
  `scripts/build_bracketology.py` (by hand / replays).
- `mri/export/bracketdata.py` — calendar gates, day-by-day history for movement,
  the Selection Sunday freeze and the comparison with the real field.
- `mri/export/bracketpages.py` — the seed list and bracket pages, and the team
  page line.

**To watch in the first live weeks:** the 76-team seed-line seam (an at-large
team can land on 13 below weaker Opening Round 11s/12s); the WCC, UAC and ASUN
provisional formats; the build log's `format warnings`.
