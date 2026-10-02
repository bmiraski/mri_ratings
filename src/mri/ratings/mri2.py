"""MRI 2.0 - opponent strength solved simultaneously rather than approximated.

MRI Classic measures schedule strength as "my opponents' win percentage, times
their opponents' win percentage". That treats a 9-3 Sun Belt team and a 9-3 SEC
team as equal tests, because win percentage does not know who anyone played.

MRI 2.0 replaces that with a single simultaneous fit. Every game is one
equation,

    compressed_margin  =  rating_home - rating_away + home_field

and all equations are solved at once, so a team's rating depends on its
opponents' ratings, which depend on theirs, all the way down. Strength of
schedule stops being a separate statistic and becomes a property of the
solution.

Three further departures from Classic:

*Margin is compressed, not capped.* Classic truncates at +/-35, so a 34-point
win and a 60-point win are worth the same. Here margin passes through
``k * tanh(margin / k)``, which is nearly the identity inside about two
touchdowns and flattens beyond it. Ratings stay denominated in points.

*Ratings are pulled toward a prior.* The ridge penalty shrinks each team toward
a preseason estimate. With two games played the prior dominates; by midseason
the schedule does. This is what makes a Week 3 ranking publishable, and it falls
out of one parameter rather than a hand-tuned weekly schedule.

*Two numbers, not one.* ``power`` answers "how good is this team" and is
denominated in points, so the gap between two teams is a predicted spread.
``resume`` answers "what has this team earned", measured as wins above what an
average team would have managed against the same schedule at the same sites.
Classic conflated these, which is why it was hard to bet with.

Home and away come from the game table's column order: in every archived
workbook, team1 is the visitor.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import connectivity as _connectivity

# Defaults tuned against the 2003-2019 archive; see scripts/tune_mri2.py.
DEFAULT_COMPRESSION = 40.0
DEFAULT_RIDGE = 4.0
DEFAULT_PRIOR_REGRESSION = 0.30
DEFAULT_MARGIN_SIGMA = 16.0
REPLACEMENT_PRIOR = -22.0
POOLED_FCS = "Non D1A"

# Home field is barely identifiable from two weeks of games, and early in a
# season the fit will happily blame a 40-point win over an FCS team on playing
# at home. Anchoring it to a mild prior keeps it honest until the schedule can
# speak for itself.
DEFAULT_HOME_FIELD_PRIOR = 3.0
DEFAULT_HOME_FIELD_RIDGE = 60.0


@dataclass(frozen=True)
class Profile:
    """Per-sport settings for the same solver.

    The method is identical for both sports - every game is one equation and
    the season is solved at once. What differs is scale. Basketball margins are
    tighter (a standard deviation near 14 against football's 16.5), home
    advantage is larger, there are 365 teams rather than 138, and a non-D1
    opponent sits much further below the floor than an FCS one does.
    """

    name: str
    compression: float
    ridge: float
    prior_regression: float
    home_field_prior: float
    replacement_prior: float


FOOTBALL_PROFILE = Profile(
    name="football",
    compression=DEFAULT_COMPRESSION,
    ridge=DEFAULT_RIDGE,
    prior_regression=DEFAULT_PRIOR_REGRESSION,
    home_field_prior=DEFAULT_HOME_FIELD_PRIOR,
    replacement_prior=REPLACEMENT_PRIOR,
)

# Searched by scripts/tune_basketball.py over 80 combinations, tuned on
# 2021-22 to 2023-24 and reported on 2024-25 and 2025-26. Re-run after the
# phantom-tie fix, on clean data.
#
# The search found nothing, and found nothing twice. On the holdout the best
# searched settings score MAE 9.2771 against 9.3018 for these - a fortieth of a
# point - while scoring *worse* on accuracy (70.03% against 70.16%) and on
# Brier. That is noise, not an improvement.
#
# It is a real finding about the sport rather than a failed exercise:
# basketball plays ~6,000 games among 365 teams, so by the time 40% of a season
# is gone there is enough evidence per team that the prior and the ridge barely
# matter. Football, with 900 games among 138 teams, is far more sensitive -
# the same search there bought a full point of MAE.
#
# One reason to prefer these over the searched winner beyond the tie: the
# searched settings imply a home court of 2.41 points, while the seasons
# themselves come in at 2.7 to 3.3. The number that matches the world wins a
# tie against the number that matches the objective by a fortieth of a point.
BASKETBALL_PROFILE = Profile(
    name="basketball",
    compression=34.0,
    ridge=8.0,
    prior_regression=0.35,
    home_field_prior=3.5,
    replacement_prior=-20.0,
)


@dataclass(frozen=True)
class FitInternals:
    """What the solver used, kept so a rating can be explained after the fact.

    Nothing here feeds back into any published number. ``raw_home_field`` is the
    solver's own coefficient, which is not the ``home_field`` on ``Ratings``: that
    one is re-estimated from residuals after the solve. The normal equations
    (``ratings.decompose``) need the first.

    ``recency_weighted`` and ``pace_adjusted`` exist so that a fit using either
    can say so; the decomposition's identity holds only for the unweighted solve
    and refuses otherwise.
    """

    teams: list[str]
    prior: np.ndarray        # per team, in solver units
    ridge: float
    compression: float
    scale: float
    anchor_shift: float      # subtracted from scaled ratings; 0 with no anchor
    raw_home_field: float
    margin: np.ndarray       # compressed margin per game, host's side
    neutral: np.ndarray      # bool per game
    recency_weighted: bool = False
    pace_adjusted: bool = False


@dataclass
class Ratings:
    """The fitted model for one season (or one slice of one)."""

    power: pd.Series
    home_field: float
    sigma: float
    games_played: pd.Series
    resume: pd.Series = field(default=None)
    adj_offense: pd.Series = field(default=None)
    adj_defense: pd.Series = field(default=None)
    internals: FitInternals | None = field(default=None, repr=False)
    connectivity: _connectivity.Connectivity | None = field(default=None, repr=False)

    def table(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "power": self.power,
                "resume": self.resume,
                "games": self.games_played,
                "adj_offense": self.adj_offense,
                "adj_defense": self.adj_defense,
            }
        )
        frame = frame.sort_values("power", ascending=False)
        frame.insert(0, "rank", range(1, len(frame) + 1))
        if frame["resume"].notna().any():
            frame.insert(
                1, "resume_rank", frame["resume"].rank(ascending=False, method="min").astype("Int64")
            )
        return frame.rename_axis("team").reset_index()

    def predict(self, home: str, away: str, neutral: bool = False) -> float:
        """Predicted margin from the home team's perspective, in points."""
        edge = self.power.get(home, REPLACEMENT_PRIOR) - self.power.get(away, REPLACEMENT_PRIOR)
        return edge + (0.0 if neutral else self.home_field)

    def win_probability(self, home: str, away: str, neutral: bool = False) -> float:
        from scipy.stats import norm

        return float(norm.cdf(self.predict(home, away, neutral) / self.sigma))


def compress(margin, k: float = DEFAULT_COMPRESSION):
    """Diminishing returns on blowouts, near-linear for normal scores.

    ``k * tanh(margin / k)`` is within a point of the identity out to about 20
    points and saturates near +/-k, so running up the score stops paying long
    before it becomes free.
    """
    margin = np.asarray(margin, dtype=float)
    return k * np.tanh(margin / k)


def mark_postseason(games: pd.DataFrame) -> pd.Series:
    """Flag bowl and playoff games without needing a date column.

    A game is postseason when both teams have already played twelve others.
    Row order is chronological in every archived workbook, which is enough; only
    2018 and 2019 carry real dates. Checked against those two, the rule catches
    all 40 of 2019's bowls plus the ten conference championship games, which are
    played at neutral sites too.

    Postseason games are then treated as neutral, which the data supports:
    across 2019 the visitor's average margin is -4.6 in the regular season and
    -1.5 afterwards.
    """
    counts: dict[str, int] = {}
    flags = []
    for away, home in zip(games["team1"], games["team2"]):
        flags.append(counts.get(away, 0) >= 12 and counts.get(home, 0) >= 12)
        counts[away] = counts.get(away, 0) + 1
        counts[home] = counts.get(home, 0) + 1
    return pd.Series(flags, index=games.index, name="neutral")


def _design(games: pd.DataFrame, teams: list[str], neutral: np.ndarray):
    """Build the least-squares system: one row per game, one column per team."""
    index = {team: i for i, team in enumerate(teams)}
    n_games, n_teams = len(games), len(teams)

    X = np.zeros((n_games, n_teams + 1))
    rows = np.arange(n_games)
    X[rows, [index[t] for t in games["team2"]]] = 1.0   # home
    X[rows, [index[t] for t in games["team1"]]] -= 1.0  # away
    X[:, -1] = np.where(neutral, 0.0, 1.0)              # home-field term
    return X


def fit(
    games: pd.DataFrame,
    prior: pd.Series | None = None,
    *,
    compression: float = DEFAULT_COMPRESSION,
    ridge: float = DEFAULT_RIDGE,
    neutral: pd.Series | np.ndarray | None = None,
    anchor_teams: list[str] | None = None,
    home_field_prior: float = DEFAULT_HOME_FIELD_PRIOR,
    home_field_ridge: float = DEFAULT_HOME_FIELD_RIDGE,
    with_resume: bool = True,
    with_efficiency: bool = True,
    recency_half_life: float | None = None,
    season_games: int | None = None,
    check_connectivity: bool = True,
) -> Ratings:
    """Solve every game at once for a power rating in points.

    Parameters
    ----------
    games
        Normalized game table. ``team1`` is the visitor, ``team2`` the host.
    prior
        Preseason rating per team. Teams absent from it start at replacement
        level, which is where an FCS opponent belongs.
    ridge
        Strength of the pull toward ``prior``. Larger means a team needs more
        evidence to move. Because the penalty is fixed while the number of
        equations per team grows, shrinkage fades over a season on its own.
    anchor_teams
        Teams whose average rating is pinned to zero, normally the FBS field.
        Ratings are only determined up to a constant, so without an anchor the
        scale is free to wander - and it does, badly, once hundreds of
        individually-named FCS opponents enter the pool and drag the centre of
        mass down with them. Anchoring changes no prediction, since a uniform
        shift cancels in every rating difference, but it keeps "zero" meaning
        "an average FBS team" from one season to the next.
    recency_half_life
        Optional recency weighting for ``power``: the age, as a *fraction of the
        season*, at which a game counts half as much as the latest one. ``None``
        (the default) is the unweighted fit, bit for bit. Age is read from row
        position, not dates - the archive has none for most seasons - so the
        rows must be chronological, as the walk-forward already assumes.
        ``season_games`` is the length of the full season the slice belongs to
        (default: the slice itself); a walk-forward slice must pass it so a
        half-life means the same thing at every cutoff.

        Weighted: the solve, the ``scale`` rescale, the home-field mean and
        ``sigma``. All four describe "how good is this team now" and are used
        to predict games after the slice, so they should read the evidence the
        same way. The home-field mean keeps its ridge prior at full strength
        against the *weighted* game count, so weighting leaves that prior
        relatively heavier - the same effect it has on the team ridge.

        Not weighted: ``resume`` ("what has this team earned over the whole
        season"), which comes from a second, unweighted solve, and the
        yardage-efficiency layer.
    check_connectivity
        Store a ``Connectivity`` report on the result and warn (never raise) if
        a rated team is stranded; see ``mri.ratings.connectivity``.
    """
    if games.empty:
        raise ValueError("no games to fit")

    games = games.reset_index(drop=True)
    if neutral is None:
        neutral = mark_postseason(games)
    neutral = np.asarray(neutral, dtype=bool)

    teams = sorted(set(games["team1"]) | set(games["team2"]))
    X = _design(games, teams, neutral)
    y = compress(games["pts2"].to_numpy(float) - games["pts1"].to_numpy(float), compression)

    # Absent a prior, every team starts at average - zero - so the solved scale
    # is centred on an average FBS team and a rating reads directly as points
    # better or worse than that. Only teams the prior does not know (a new FCS
    # opponent, say) start at replacement level.
    prior_vector = np.zeros(len(teams), dtype=float)
    if prior is not None:
        for i, team in enumerate(teams):
            value = prior.get(team) if team in prior.index else None
            prior_vector[i] = (
                float(value) if value is not None and np.isfinite(value) else REPLACEMENT_PRIOR
            )
    # The pooled non-FBS opponent is never a real team; hold it at replacement.
    if POOLED_FCS in teams:
        prior_vector[teams.index(POOLED_FCS)] = REPLACEMENT_PRIOR

    weights = None
    if recency_half_life is not None:
        weights = recency_weights(len(games), recency_half_life, season_games)

    solve_args = (games, X, y, neutral, teams, prior_vector)
    solve_kwargs = dict(
        ridge=ridge,
        home_field_prior=home_field_prior,
        home_field_ridge=home_field_ridge,
        anchor_teams=anchor_teams,
    )
    power, home_field, sigma, hosted, solved = _solve(*solve_args, weights=weights, **solve_kwargs)
    if weights is None or not with_resume:
        resume_basis = (power, home_field, sigma)
    else:
        # Résumé is "what has this team earned over the whole season", so it is
        # judged against an unweighted fit of the same games. Power is the
        # "how good now" number; the two answer different questions.
        plain = _solve(*solve_args, weights=None, **solve_kwargs)
        resume_basis = plain[:3]

    played = pd.concat([games["team1"], games["team2"]]).value_counts()
    played = played.reindex(teams).fillna(0).astype(int)

    ratings = Ratings(power=power, home_field=home_field, sigma=sigma, games_played=played)
    scale, anchor_shift, raw_home_field = solved
    ratings.internals = FitInternals(
        teams=teams,
        prior=prior_vector,
        ridge=float(ridge),
        compression=float(compression),
        scale=scale,
        anchor_shift=anchor_shift,
        raw_home_field=raw_home_field,
        margin=y,
        neutral=neutral,
        recency_weighted=weights is not None,
    )

    if check_connectivity:
        ratings.connectivity = _connectivity.analyze(games, rated=anchor_teams)
        if not ratings.connectivity.ok:
            warnings.warn(ratings.connectivity.summary(), _connectivity.ConnectivityWarning,
                          stacklevel=2)

    if with_resume:
        ratings.resume = wins_above_expected(games, *resume_basis, neutral)

    # Yardage lives in box scores, which the API serves a week at a time and the
    # games feed omits entirely. The power rating never needed it, so when it is
    # absent the efficiency layer is simply skipped rather than treated as an
    # error - scores alone are enough to rate a season.
    has_yardage = {"rush1", "rush2", "pass1", "pass2"} <= set(games.columns)
    if with_efficiency and has_yardage:
        offense, defense = _fit_efficiency(games, teams, X[:, :-1], ridge)
        ratings.adj_offense, ratings.adj_defense = offense, defense
    return ratings


def recency_weights(n_games: int, half_life: float, season_games: int | None = None) -> np.ndarray:
    """Weight per game, 0.5 ** (age / half_life), age a fraction of the season.

    Age is the game's row distance from the slice's last game, divided by the
    length of the full season. Rows must be chronological.
    """
    if half_life <= 0:
        raise ValueError("recency_half_life must be positive")
    season = season_games or n_games
    age = (n_games - 1 - np.arange(n_games)) / float(season)
    return 0.5 ** (age / half_life)


def _solve(
    games, X, y, neutral, teams, prior_vector, *,
    ridge, home_field_prior, home_field_ridge, anchor_teams, weights,
):
    """One ridge solve plus everything downstream of it that depends on the weights.

    ``weights=None`` is the original unweighted arithmetic, expression for
    expression; the weighted branches reduce to it when every weight is one.
    """
    penalty = np.full(len(teams) + 1, ridge)
    penalty[-1] = home_field_ridge
    target = np.append(prior_vector, home_field_prior)

    if weights is None:
        gram = X.T @ X + np.diag(penalty)
        rhs = X.T @ y + penalty * target
    else:
        gram = X.T @ (weights[:, None] * X) + np.diag(penalty)
        rhs = X.T @ (weights * y) + penalty * target
    solution = np.linalg.solve(gram, rhs)

    raw = solution[:-1]
    raw_home_field = solution[-1]

    # The fit runs on compressed margins and is shrunk toward the prior, so its
    # output is systematically smaller than real scoreboard margins. Rescale by
    # regressing actual margin on predicted edge, which puts `power` back into
    # honest points: the gap between two teams is a predicted spread, not an
    # index. Without this the model would quietly under-price every favourite.
    actual = games["pts2"].to_numpy(float) - games["pts1"].to_numpy(float)
    edge = X[:, :-1] @ raw + X[:, -1] * raw_home_field
    if weights is None:
        scale = float(actual @ edge / (edge @ edge)) if edge @ edge > 0 else 1.0
    else:
        denominator = float(edge @ (weights * edge))
        scale = float(actual @ (weights * edge) / denominator) if denominator > 0 else 1.0
    scale = float(np.clip(scale, 0.5, 4.0))

    power = pd.Series(raw * scale, index=teams, name="power")

    # Home field is deliberately NOT rescaled with the ratings. Its prior is
    # expressed in real points, and early in a season - when shrinkage makes the
    # fitted edges small and the scale factor correspondingly large - multiplying
    # it through inflates a 3-point effect into eight or more. Instead it is
    # re-estimated on the scaled ratings as a shrunken mean of what the home
    # side actually scored beyond the rating difference.
    residual_margin = actual - (X[:, :-1] @ raw) * scale
    hosted = ~np.asarray(neutral, dtype=bool)

    # Only games between teams we actually know anything about get a say. A
    # September blowout of an FCS visitor whose rating is pinned near its prior
    # leaves a huge positive residual that has nothing to do with home field,
    # and in a two-week-old season those games are half the schedule.
    if anchor_teams:
        known = set(anchor_teams)
        measurable = hosted & np.array(
            [a in known and b in known for a, b in zip(games["team1"], games["team2"])]
        )
        if measurable.sum() < 20:
            measurable = hosted
    else:
        measurable = hosted

    if weights is None:
        home_field = float(
            (residual_margin[measurable].sum() + home_field_ridge * home_field_prior)
            / (measurable.sum() + home_field_ridge)
        )
    else:
        w_home = weights[measurable]
        home_field = float(
            ((w_home * residual_margin[measurable]).sum() + home_field_ridge * home_field_prior)
            / (w_home.sum() + home_field_ridge)
        )

    anchor_shift = 0.0
    if anchor_teams:
        present = [t for t in anchor_teams if t in power.index]
        if present:
            anchor_shift = float(power[present].mean())
            power = power - anchor_shift

    residuals = residual_margin - np.where(hosted, home_field, 0.0)  # noqa: E501
    # A one-game training set (some early CFBD seasons open with a single nationally-televised
    # game before the rest of the week kicks off) makes ddof=1 undefined, and numpy returns NaN -
    # which is truthy, so ``nan or DEFAULT`` would return the NaN rather than falling back.
    if weights is None:
        fitted_sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float("nan")
    elif len(residuals) > 1:
        # Reliability-weighted variance; equals the ddof=1 estimate when weights are equal.
        v1, v2 = weights.sum(), (weights**2).sum()
        mean = (weights * residuals).sum() / v1
        fitted_sigma = float(
            np.sqrt((weights * (residuals - mean) ** 2).sum() / (v1 - v2 / v1))
        )
    else:
        fitted_sigma = float("nan")
    sigma = fitted_sigma if fitted_sigma > 0 else DEFAULT_MARGIN_SIGMA

    return power, home_field, sigma, hosted, (scale, anchor_shift, float(raw_home_field))


def wins_above_expected(
    games: pd.DataFrame,
    power: pd.Series,
    home_field: float,
    sigma: float,
    neutral: np.ndarray,
) -> pd.Series:
    """How many more games a team won than an average team would have.

    This is the résumé number. It asks, for each game on the schedule, what the
    chance was that a league-average team would have won it at that site, and
    compares the total to what the team actually did. A 10-2 record against a
    brutal schedule scores higher than 10-2 against nobody, which is the whole
    point, and the units are wins rather than an index nobody can interpret.
    """
    from scipy.stats import norm

    away = games["team1"].to_numpy()
    home = games["team2"].to_numpy()
    opp_power_for_home = power.reindex(away).to_numpy(float)
    opp_power_for_away = power.reindex(home).to_numpy(float)
    edge = np.where(neutral, 0.0, home_field)

    # An average team (rating 0) playing this opponent, at this site.
    expected_home = norm.cdf((0.0 - opp_power_for_home + edge) / sigma)
    expected_away = norm.cdf((0.0 - opp_power_for_away - edge) / sigma)

    actual_home = games["win2"].to_numpy(float)
    actual_away = games["win1"].to_numpy(float)

    tally = pd.concat(
        [
            pd.Series(actual_home - expected_home, index=home),
            pd.Series(actual_away - expected_away, index=away),
        ]
    )
    return tally.groupby(level=0).sum().reindex(power.index).rename("resume")


def _fit_efficiency(games: pd.DataFrame, teams: list[str], X_teams: np.ndarray, ridge: float):
    """Opponent-adjusted yards per game, on offense and defense.

    Classic used raw yardage z-scores, which reward tempo and reward playing
    nobody. Solving offense and defense jointly removes the schedule from both:
    a unit is measured against what its opponents normally allow.
    """
    away_yards = games["rush1"].to_numpy(float) + games["pass1"].to_numpy(float)
    home_yards = games["rush2"].to_numpy(float) + games["pass2"].to_numpy(float)

    index = {team: i for i, team in enumerate(teams)}
    n = len(teams)
    rows = np.arange(len(games))

    # Two observations per game: home offense vs away defense, and the reverse.
    A = np.zeros((2 * len(games), 2 * n))
    A[rows, [index[t] for t in games["team2"]]] = 1.0
    A[rows, [n + index[t] for t in games["team1"]]] = -1.0
    A[rows + len(games), [index[t] for t in games["team1"]]] = 1.0
    A[rows + len(games), [n + index[t] for t in games["team2"]]] = -1.0

    observed = np.concatenate([home_yards, away_yards])
    mean = observed.mean()
    b = observed - mean

    gram = A.T @ A + np.eye(2 * n) * ridge
    solution = np.linalg.solve(gram, A.T @ b)

    offense = pd.Series(solution[:n] + mean, index=teams, name="adj_offense")
    defense = pd.Series(mean - solution[n:], index=teams, name="adj_defense")
    return offense, defense


def build_prior(
    previous: pd.Series | None,
    teams: list[str],
    regression: float = DEFAULT_PRIOR_REGRESSION,
    centre_teams: list[str] | None = None,
    *,
    outsiders_to_replacement: bool = True,
) -> pd.Series:
    """Carry last season's ratings forward, regressed toward the mean.

    Rosters turn over, so last year's number is informative but stale.
    ``regression`` is the share pulled back toward average: 0 trusts last season
    completely, 1 discards it.

    ``centre_teams`` says which teams define "average". This matters more than
    it looks: API data names every FCS opponent individually, so a plain median
    over all teams is a median over mostly-FCS teams, and regressing toward it
    drags the entire FBS field down a little more each season.

    Teams outside ``centre_teams`` - FCS opponents - regress toward replacement
    level instead, and a newcomer starts there. Pulling them toward the FBS
    average lifted every FCS prior by about seven points a season, and one or
    two games a year against FBS sides never pulled them back down: walked
    forward over 2014-2025, FBS hosts beat FCS visitors by 9.7 points more than
    predicted. Regressed toward replacement, that bias is under a point, and
    FBS-vs-FBS lines improve too, because FBS teams stop being credited for
    routine blowouts (``scripts/backtest_fcs_prior.py``).
    ``outsiders_to_replacement=False`` keeps the old rule, which basketball
    still uses until it is tested there.
    """
    if previous is None or previous.empty:
        return pd.Series(0.0, index=teams)

    pool = previous
    if centre_teams:
        present = [t for t in centre_teams if t in previous.index]
        if present:
            pool = previous[present]
    centre = pool.median()

    floor = REPLACEMENT_PRIOR + centre
    carried = previous.reindex(teams)
    prior = (1.0 - regression) * carried.fillna(floor) + regression * centre
    if centre_teams and outsiders_to_replacement:
        outside = ~prior.index.isin(centre_teams)
        prior[outside] = ((1.0 - regression) * carried[outside] + regression * floor).fillna(floor)
    return prior
