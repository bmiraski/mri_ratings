"""MRI 2.0 end-of-season Power for every season, 2003-present, on one consistent scale.

``current_ratings.parquet`` (``scripts/build_current.py``) only covers 2020+.
The 2003-2019 walk-forward that validated MRI 2.0
(``mri.ratings.backtest.evaluate_archive``) computes an end-of-season Power
rating for every archive season too - it is the ``previous`` variable carried
between seasons - but only uses it to prime the next season and score
accuracy; it is never kept. Anything that wants a team's rating history
across the archive/CFBD boundary (coach metrics, first) needs it kept, so
``archive_walk`` walks the archive the same way and returns every season's
ratings, and ``build`` splices that onto ``current_ratings.parquet`` for a
single 2003-present table.

Two differences from ``evaluate_archive``, both deliberate:

* That walk does not pass ``anchor_teams`` to ``mri2.fit``, which is fine for
  its pairwise accuracy scoring but not here - without an anchor the rating
  scale drifts season to season as FCS opponents enter the pool (see
  ``mri2.fit``'s own docstring). This walk anchors every season to that
  season's FBS field via ``registry.was_fbs``.
* It resolves every team name through ``registry.resolve`` before fitting
  anything. The archive's own canonicalization (``mri.ratings.classic.
  canonicalize``) only dedupes spellings *within* one workbook season - it
  never promised to agree with the registry's CFBD-era spelling, and it
  doesn't always (the archive calls California "Cal" in every season, for
  one). Left unresolved, a team like that gets two disconnected entries -
  its archive-spelled seasons and its registry-spelled ones never joining up
  as "the same team" for a coach's tenure.

Hyperparameters are left at ``mri2``'s module defaults throughout, matching
both ``evaluate_archive`` and ``build_current.py``.

There is a real seam at ``SEAM_YEAR``, and it is worth understanding rather
than hiding. ``build_current.py`` primes 2020 by fitting 2019 with NO prior
at all - every team starts at zero for that one fit - while this walk
carries a prior all the way from 2003, matching what ``evaluate_archive``
actually validated. A prior-informed fit lets perennial top and bottom
programs sit at their true extremes instead of being shrunk toward average;
checked against each other, 2019 team ratings can differ by several points
for a team like Alabama or UTEP (see ``seam_gap``). The archive side here is
the more accurate methodology and is kept as-is; 2020+ is spliced in
unchanged from ``current_ratings.parquet`` so every published Power number
still matches the live site. Anything computing a metric like a coach's
"change since hire" across a tenure that crosses ``SEAM_YEAR`` should not
treat that one step as clean signal the way every other season-to-season
step is.
"""

from __future__ import annotations

import pandas as pd

from ..ingest import registry
from . import mri2, priors

ARCHIVE_SEASONS = range(2003, 2020)
SEAM_YEAR = 2020  # first season sourced from current_ratings.parquet instead of the archive walk


def canonical_games(frame: pd.DataFrame) -> pd.DataFrame:
    """Put every team name in ``team1``/``team2`` into its registry spelling."""
    frame = frame.copy()
    for column in ("team1", "team2"):
        frame[column] = [registry.resolve(name, name) for name in frame[column]]
    return frame


def archive_walk(archive_games: pd.DataFrame, *, seasons=ARCHIVE_SEASONS) -> pd.DataFrame:
    """Anchored MRI 2.0 Power, prior and résumé for every team, every archive season, chained forward.

    ``archive_games`` must already be name-canonicalized (``canonical_games``).
    ``resume`` (wins above an average team's, against the schedule played) is
    a pure addition to what phase 1 computed - ``with_resume=True`` doesn't
    touch the ridge solve, so every ``power`` value is unchanged.
    """
    rows = []
    previous: pd.Series | None = None
    for season in seasons:
        season_games = archive_games[archive_games["season"] == season]
        if season_games.empty:
            continue
        teams = sorted(set(season_games["team1"]) | set(season_games["team2"]))
        fbs = [t for t in teams if registry.was_fbs(t, season)]
        prior = mri2.build_prior(previous, teams, centre_teams=fbs)
        model = mri2.fit(season_games, prior=prior, anchor_teams=fbs, with_resume=True, with_efficiency=False)
        rows.append(pd.DataFrame({
            "team": model.power.index,
            "season": season,
            "power": model.power.values,
            "prior": prior.reindex(model.power.index).values,
            "resume": model.resume.reindex(model.power.index).values,
        }))
        previous = model.power
    columns = ["team", "season", "power", "prior", "resume"]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)


def seam_gap(archive_history: pd.DataFrame, archive_games: pd.DataFrame) -> pd.Series:
    """Gap between the walk's SEAM_YEAR-1 row and build_current.py's own standalone (no-prior) prime of it.

    ``archive_games`` must already be name-canonicalized. Real, not noise -
    see the module docstring - returned as the full per-team difference so a
    caller can report more than just the worst case.
    """
    boundary = SEAM_YEAR - 1
    at_boundary = archive_games[archive_games["season"] == boundary]
    reprime = mri2.fit(
        at_boundary,
        anchor_teams=[t for t in set(at_boundary["team2"]) if registry.is_fbs(t)],
        with_resume=False,
        with_efficiency=False,
    ).power
    mine = archive_history[archive_history["season"] == boundary].set_index("team")["power"]
    common = mine.index.intersection(reprime.index)
    return (mine[common] - reprime[common]).rename("gap")


def current_prior_walk(archive_2019: pd.DataFrame, current_games: pd.DataFrame) -> pd.DataFrame:
    """The preseason prior for every team, every 2020+ season - the one thing
    ``build_current.py``'s own loop computes and never keeps.

    Deliberately a replica of that loop (canonicalized 2019 prime,
    ``priors.for_season``, ``mri2.fit`` with ``anchor_teams`` via
    ``registry.is_fbs`` - not the more historically-correct ``was_fbs`` this
    module uses for the archive, because faithfully reproducing what
    production actually did is the point) rather than a change to
    ``build_current.py`` itself: this whole feature stays isolated from the
    live ratings pipeline, the same way every other site feature does.
    ``current_games`` is ``current_games.parquet`` as-is (already
    canonicalized by ``build_current.py``); ``archive_2019`` must already be
    canonicalized (``canonical_games``).

    Returns ``team, season, prior, power`` - ``power`` is this replica's own
    recomputation, kept so a caller can check it against
    ``current_ratings.parquet``'s actual numbers (see
    ``current_prior_walk_gap``) rather than trusting the replica blindly.
    """
    previous = mri2.fit(
        archive_2019,
        anchor_teams=[t for t in set(archive_2019["team2"]) if registry.is_fbs(t)],
        with_resume=False,
        with_efficiency=False,
    ).power

    rows = []
    for season in sorted(current_games["season"].unique()):
        games = current_games[current_games["season"] == season]
        teams = sorted(set(games["team1"]) | set(games["team2"]))
        fbs = [t for t in teams if registry.is_fbs(t)]
        prior = priors.for_season(int(season), previous, teams, fbs)
        model = mri2.fit(games, prior=prior, neutral=games["neutral"], anchor_teams=fbs)
        rows.append(pd.DataFrame({
            "team": prior.index, "season": season,
            "prior": prior.values, "power": prior.index.map(model.power),
        }))
        previous = model.power
    columns = ["team", "season", "prior", "power"]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)


def current_prior_walk_gap(replica: pd.DataFrame, current_ratings: pd.DataFrame) -> pd.Series:
    """Per-team-season gap between the replica's own power and current_ratings.parquet's.

    Should be ~0 everywhere; a real gap means build_current.py's methodology
    has drifted from what this replica assumes and ``current_prior_walk``
    needs updating to match.
    """
    mine = replica.set_index(["team", "season"])["power"]
    theirs = current_ratings.set_index(["team", "season"])["power"]
    common = mine.index.intersection(theirs.index)
    return (mine[common] - theirs[common]).rename("gap")


def build(archive_games: pd.DataFrame, current_games: pd.DataFrame, current_ratings: pd.DataFrame) -> pd.DataFrame:
    """The full 2003-present history: the anchored archive walk spliced onto the live ratings,
    with a preseason ``prior`` alongside ``power`` and ``resume`` throughout.

    ``archive_games`` and ``current_games`` are read straight from their
    parquet files (not yet canonicalized for the archive side - this does
    it; ``current_games`` already is, by ``build_current.py``).
    ``current_ratings`` is current_ratings.parquet as-is.
    """
    archive_games = canonical_games(archive_games)
    archive_history = archive_walk(archive_games)

    archive_2019 = archive_games[archive_games["season"] == SEAM_YEAR - 1]
    prior_walk = current_prior_walk(archive_2019, current_games)
    current = current_ratings[["team", "season", "power", "resume"]].merge(
        prior_walk[["team", "season", "prior"]], on=["team", "season"], how="left"
    )
    if not current.empty:
        assert int(current["season"].min()) == SEAM_YEAR, "current_ratings.parquet no longer starts at SEAM_YEAR"

    columns = ["team", "season", "power", "prior", "resume"]
    return pd.concat([archive_history[columns], current[columns]], ignore_index=True).sort_values(["season", "team"])
