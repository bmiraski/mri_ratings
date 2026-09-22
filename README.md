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

### NCAA Tournament bracketology (Phases 1-3 built; Phase 4, the pages, not started)

Two-part model, in the plan Ben and Claude wrote together
(`claude_bracketology-plan.md` in the project): first, who wins each conference's
automatic bid; then the at-large field and everyone's seed; then both at once,
simulated forward, with the field placed into a bracket. All three are built and
backtested below.

**Why a conference's own tournament isn't hand-documented.** Byes, reseeding, and
how many teams even get invited vary conference to conference and change over
time - the SEC and Big 12 both changed shape mid-decade just from realignment
adding teams. `mri/bracket/template.py` infers each conference's current bracket
shape from its own tournament games: group them by the US-arena calendar day
(not the UTC date the API stores, which splits a late night's games across two
days if not converted), and count how many teams are new to the bracket at each
successive day. A team appearing for the first time on day three had a bye
through the first two rounds; how many teams debut at each stage *is* the
conference's bye structure, and it falls out of the schedule without needing
anyone's seed. Checked against seven real conferences' documented formats
(SEC, A-10, Southland, Horizon, Big 12, MAC, Ivy) and matched all of them,
including catching the SEC's and Big 12's real membership growth as a shape
change rather than noise.

**Seeding a conference's own bracket** (`mri/bracket/history.py`) is an
approximation: teams ranked by conference win percentage, ties broken by
win count and then arbitrarily - the committee's real tiebreakers (head-to-head,
common opponents) aren't reconstructable from this data. Checked by hand
against the real 2025 SEC standings: it gets the field and the top of the
order right and can misorder a true tie, exactly as expected.

**The simulation** (`mri/bracket/simulate.py`) plays each conference's bracket
forward: every round, the survivors are sorted by rating and paired end to end,
best against worst, with each new tier's byes folding in as their round
arrives. This is a reseeded bracket by construction - some conferences (the
Horizon League among them) actually play theirs that way, but a true static
bracket is fixed from the draw and doesn't rebuild itself around upsets. Good
enough to estimate who wins the auto bid; not a substitute for the real bracket
once it's drawn. Home-court is applied for conferences that play on campus
rather than at one neutral site (detected the same way, from the game data).

**The backtest** (`scripts/build_bracket_history.py`) grades the simulation
against the placeholder that's live today (the #1 regular-season standings team
wins the auto bid) for every conference, 2011-2025 (2020 cancelled), using
ratings cut off *before* that conference's own tournament and each conference's
own actual bracket that year. That cutoff matters and is worth stating plainly:
an earlier version of this backtest fit ratings on the whole regular season,
which in this feed's own labelling includes conference tournament games
(`season_type == "regular"` covers both) - so a team's rating was quietly
absorbing its own tournament run before that same run got re-simulated to
"predict" it. Checked directly: 2025 Florida, which won both its conference
tournament and the national title, rated 20.1 with the leak and 18.3 without
it. Fixed by cutting every team's rating off at the season's earliest
conference-tournament tip-off, and every number below is from that corrected
run. Log loss alone is still a weak test here - the placeholder is a
deterministic 100%-or-nothing call, so almost any real probability model beats
it on log loss whenever the top seed doesn't win outright every time. The
switch condition asks for both: a real log-loss margin (>= 0.3) *and* the
simulation's own top pick being right at least as often as the placeholder's.

Current read (all 32 conferences, n=6-13 graded seasons each - small samples,
read the pattern and not any one conference's exact number):

- **19 of 32 conferences clear both bars** - WCC (92% right vs. the
  placeholder's 67%), Southland (82% vs. 55%), Patriot (77% vs. 54%) and UAC
  (64% vs. 36%) are the clearest; several others clear it by a much thinner
  margin (Big South, Sun Belt, CAA, Big East, ASUN and American all tie the
  placeholder's own hit rate and pass only on log loss).
- **13 don't**: A-10, ACC, Am. East, Big 12, Big Sky, Horizon, Ivy, MEAC,
  Mountain West, NEC, Pac-12, SoCon, Summit - the placeholder currently calls
  more of these right. Several power conferences (ACC, Big 12, Pac-12) are in
  this group, which is exactly the kind of thing the leak was masking.
- The corrected picture is honestly weaker than the leaked one (which showed
  24 of 32 clearing both bars) - a reminder that the earlier, better-looking
  numbers were partly measuring the model's ability to see results it was
  about to be asked to predict, not real skill.
- Per Ben's call: whether a given conference actually switches off the
  placeholder is a decision made from these numbers, not automatic.

Rerun with `PYTHONPATH=src python3 scripts/build_bracket_history.py`
(resumable; results cache to `/tmp/bracket_history_cache.pkl` and the summary
writes to `data/bracket_autobid_backtest.json`).

#### Phase 2 — at-large selection and seeding

**A team's résumé** (`mri/bracket/resume.py`) as of Selection Sunday: record,
quadrant record (Quad 1-4, the committee's own NET-era buckets - our power rank
stands in for NET, which has no public history to fit against), road/neutral
wins, and schedule strength. Selection Sunday comes after the conference
tournaments, so those games are included in both the ratings and the record;
only the NCAA tournament itself (which hasn't happened yet at any point this
asks about) is excluded.

**The composite score** (`mri/bracket/atlarge.py`) is a ridge regression of six
features - power rank, résumé, Quad 1 win rate, bad-loss rate, schedule
strength, road/neutral win count - against the *actual* historical seed number,
fit once over 2011-2025 (2020 cancelled; 2017 is missing two teams from a small
gap in the API's own tournament data for that one season). A lower score is a
better team; ranking by it selects the at-large field (whoever scores best among
the teams that didn't already win an automatic bid) and seeds the whole field,
automatic bids included - the same committee seeds an automatic qualifier by the
same yardstick as anyone else. Filling the extra eight at-large slots the 2027
expansion adds is exactly this: the same score, the same coefficients, a longer
list - not a retrained model.

Leave-one-season-out, against the real field, every season:

| | This model | Power rank alone |
|---|---|---|
| At-large field, correctly identified | **84.5%** (436 of 516) | 71.5% |
| Seed number, mean absolute error | **1.20** seed lines | — |
| Seed number, rank correlation | **0.93** | — |

`bad_loss_rate`'s fitted coefficient doesn't point the way the committee's own
stated principles would suggest (more bad losses should hurt, not help), and
neither does `road_neutral_wins`' (more road wins should help) - most likely
both collinear with résumé and schedule strength once those are also in the fit,
and worth a second look rather than something hand-tuned away against the data's
own signal. What matters once the score drives a simulation is the net effect of
a result, and that's pinned by a test: under the committed coefficients, winning
a game instead of losing it always improves a team's score, whatever the quadrant
or site (`tests/test_bracket_joint.py`).

**Committee noise.** The score is a model of the committee, not the committee:
treated as exact, it makes every team it likes a certainty, and on Selection
Sunday about one in six of those missed. `scripts/backtest_atlarge.py` now also
fits how much to blur it - a random shift to every team's score in every
simulated world, `committeeNoise` in `data/atlarge_model.json`, in score units
(roughly seed lines) - as the size that best predicts the real at-large field on
Selection Sunday, leave-one-season-out. It lands at 1.25, and cuts that day's
at-large log loss from 105 to 28 per season.

Rerun with:

```bash
PYTHONPATH=src python3 scripts/build_atlarge_history.py   # historical features -> data/parquet/atlarge_history.parquet
PYTHONPATH=src python3 scripts/backtest_atlarge.py        # fit + noise + grade -> data/atlarge_model.json, site/data/atlarge_backtest.json
```

#### Phase 3 — the whole field, simulated

**The joint simulation** (`mri/bracket/joint.py`). Phases 1 and 2 aren't
independent - a bubble team's chances depend on how many bid thieves win their
conference tournaments that year - so each simulated world plays out, in order:
the rest of the regular season, game by game; every conference's standings and
its tournament (Phase 1's bracket engine, one run per world, the games landing on
each team's résumé the way they do on the committee's sheet); the automatic bids
(tournament winner for conferences switched to the model, standings leader for
those still on the placeholder, the real champion once a tournament is actually
over); then the at-large field and seed lines from Phase 2's score, with that
world's own draw of committee noise. Each world also draws its own rating error
per team (shaped as in the football simulation, the single-game spread over the
square root of games played plus the fit's prior weight; scaled 2x, which the
February backtest preferred), since a February rating is an estimate.

Power isn't refit inside each world. That's the same simplification the football
simulation makes, and it's what makes this cheap: MRI 2.0's résumé is a sum over
games of (result minus what an average team would have expected), so with power
fixed, every résumé feature is a sum over games, and thousands of worlds are one
sparse matrix product - about a second per thousand worlds for a full season.

**Seed lines** (`mri/bracket/seeding.py`) follow the NCAA's announced 2027 format
to the letter: the 12 lowest-ranked at-large teams play in as four No. 11s and
eight No. 12s, the 12 lowest-ranked automatic qualifiers as four No. 15s and
eight No. 16s. So line 12 is entirely Opening Round at-large teams, and a
mid-major champion who used to be a 12 is a 13 now. One seam in the new format
the first real bracket will settle: an at-large team good enough to skip the
Opening Round but ranked below more than ten automatic qualifiers becomes a direct
No. 13, *below* weaker at-large teams playing in as 11s and 12s. Replaying
2024-25 under the new format does this to one team (UConn). The committee might
instead drop a strong champion to 13; nothing published says which, so this
follows the rule as written until a real bracket shows otherwise. History is
graded in the 68-team format, with the First Four at-large pair on line 11.

**Regions** (`mri/bracket/regions.py`) follow the committee's published
bracketing principles directly: No. 1 seeds' regions fix the semifinals (overall
No. 1's region meets No. 4's); later lines go along the S-curve; a conference's
first four teams on the top four lines go to four different regions; conference
mates who played three or more times can't meet before a regional final, twice
not before a regional semifinal, once not in the first round (game counts from
the actual schedule); overall No. 5 stays out of overall No. 1's region; a team
may move one line to make it all work; and the top four lines' true-seed totals
are balanced to within six points. Placement is greedy along the S-curve and
then repaired by trading places (same line, or one line away) wherever a rule is
still broken - the greedy pass alone got cornered by the 2012 Big East's nine
bids. What it doesn't do is geography - the overall
No. 1 picks its region and teams are kept near home, none of which is in the
principles as a formula - so regions are numbered by their No. 1 seed rather than
named, and a team's *region* is an approximation even where its seed line and
path are rule-exact. When a rule can't be met (three of the four at-large Opening
Round teams from one conference makes a same-conference Opening Round game
unavoidable), it's reported in `ruleProblems`, not hidden. Opening Round
pairings put neighbours on the true seed list together; the NCAA publishes no
pairing rule, so that's an assumption.

**The backtest** (`scripts/backtest_bracketology.py`): every season 2011-2025
(2020 cancelled), from three points - February 1, the day the first conference
tournament tips off, and Selection Sunday - against the field the committee
actually picked. Phase 2's coefficients are leave-one-season-out; conference
tournament formats are read from each conference's three previous tournaments.
"Model" below switches the 19 conferences Phase 1 flags to the tournament
simulation and leaves the other 13 on the placeholder; the bar to clear is "if
the season ended today" (games to date, standings leaders take the automatic
bids - Phase 2's answer on the day).

| Per season | Feb 1 | Conf. tournaments start | Selection Sunday |
|---|---|---|---|
| Brier score, simulation (model) | **26.6** | **24.0** | **8.5** |
| Brier score, simulation (all placeholder) | 28.1 | 29.0 | 8.5 |
| Brier score, "if the season ended today" | 44.1 | 36.6 | 11.6 |
| Log loss, simulation (model) vs. ended today | **90** vs. 406 | **107** vs. 338 | **31** vs. 107 |
| Projected field: real teams named | 71.6% | 74.6% | 91.3% |
| Seed line error, real field teams | 1.77 | 1.41 | 1.18 |

Read with three caveats. First, the 19-conference "model" setting was picked by
Phase 1's backtest on these same seasons, so its edge over the placeholder here
is somewhat flattering. Second, calibration: on Selection Sunday it's good in
every band; when the conference tournaments start it's good except at the top,
where 99% has meant 91% - and nearly all of that is the placeholder itself: a
one-bid league's standings leader is a certainty by rule, and in a check across
seven of these seasons those leaders missed the field 27% of the time (everyone
else at 90%+: 4%). On February 1, the 70-90%
band runs about ten points optimistic (75% has meant 64%, 85% has meant 77%);
read early-season bubble odds in that band a little down. Third, region
placement: of 42 projected brackets, 3 have a flagged rule problem, all the
unavoidable Opening Round case above.

**Live** (`scripts/build_bracketology.py` -> `site/data/bracketology.json`):
every team's odds of the field, the automatic bid and each seed line; each
conference's automatic-bid odds and status (not started / underway / decided);
one projected bracket, placed into regions, with the bubble (the Opening Round
at-large teams, first four out, next four out). Nothing is written before
Christmas, and the file is frozen from Selection Sunday on. Settings live in
`data/bracketology_settings.json`: every conference is on the **placeholder**
until Ben switches it, one line per conference; `formatOverride` gives a
conference a hand-written tournament shape when its last three tournaments no
longer describe it (the rebuilt Pac-12, whose last tournament was the old
12-team league, is the obvious candidate before March). Replay any past date
with `--season 2025 --as-of 2025-02-01 --force --out /tmp/x.json`, and add
`--field-size 76` to see it under the 2027 format.

```bash
PYTHONPATH=src python3 scripts/backtest_bracketology.py   # resumable; -> data/bracketology_backtest.json
PYTHONPATH=src python3 scripts/build_bracketology.py      # live; no-op outside Christmas..Selection Sunday
```

**Not started:** Phase 4 (the pages - list/S-curve view and bracket view - and
wiring `build_bracketology.py` into the daily build).

### Heisman odds

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

**The page** (`docs/heisman.html`, `mri/export/heismandata.py`, `heisman_page` in `site.py`). Built with every site
build, isolated like the other football extras: if anything fails, the page and its nav link disappear and the
rankings still publish. It costs three API calls (season-to-date passing, rushing and receiving) and about 25
seconds. `site/data/heisman_history.json` keeps one snapshot per week (rebuilding a week replaces it) and is what the
"Wk" column measures against. Each team's page names its top candidate. **After the voting deadline (Dec 7) the page
stops recomputing** and shows the last odds recorded before the ballots closed, labelled as such.

**Things that were tried and did not help** (each tested the way everything else is, and left out; the numbers are on the page):
a link between a player's finish and his team's simulated results (real, correlation about 0.20, but no gain in log loss);
last season's rate as the prior a player's early rate is pulled toward (worse); late-season production and production
against top-25 opponents as features of the final-vote model (both worse). The backtest picks between projection variants
by log loss and keeps the original unless another gains at least 0.02, so this can change if a later season tips it.
`data/parquet/player_games.parquet` (`scripts/build_player_games.py`, one call per season-week) exists for the last of those.

**Finalist odds** are calibrated with a monotone curve fitted in the backtest (`mri/heisman/calibrate.py`,
`finalistMap` in `data/heisman_ratios.json`); on held-out seasons it trims the error only slightly, mostly by lowering the
mid-sized chances that ran high. Defenders are still not scored: the page says a defender has been invited in 4 of the last
14 seasons (flagged `defender` in the voting file) and names the leading ones, from two extra API calls (defensive and interceptions).

**Market benchmark.** `data/heisman_market.json` holds sportsbook odds for whoever you type in, with an `asOf` date. The
page shows them beside the model when at least two players match and the file is under ten days old, and never uses them in a number.
Updating it is optional and by hand.

- **Each December, one hand-edit:** add the new season's finalists (and any published points) to `data/heisman_voting.json`
  under `seasons`, and next season's ballot dates under `keyDates` (dates are per season; a season with none never freezes).
  The latest season in that file is what everything else follows: `data.last_season()` sets how far the fitting scripts, the
  backtest and the player-table builders run, so refitting is running them, with no edits to code. The committed model and its
  test say so if the voting file is ahead of them.
- To refit (optional, a small change: one more season in fourteen): `build_player_history.py` (about 90 calls), `build_player_weekly.py`
  (about 250; resumable), `fit_heisman.py`, `backtest_heisman.py`. The raw responses are large and ignored by git.

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
