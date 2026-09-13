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

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Defaults tuned against the 2003-2019 archive; see scripts/tune_mri2.py.
DEFAULT_COMPRESSION = 40.0
DEFAULT_RIDGE = 4.0
DEFAULT_PRIOR_REGRESSION = 0.30
DEFAULT_MARGIN_SIGMA = 16.0
REPLACEMENT_PRIOR = -22.0
POOLED_FCS = "Non D1A"


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
    with_resume: bool = True,
    with_efficiency: bool = True,
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

    penalty = np.full(len(teams) + 1, ridge)
    penalty[-1] = 0.0  # let home-field advantage float free
    target = np.append(prior_vector, 0.0)

    gram = X.T @ X + np.diag(penalty)
    rhs = X.T @ y + penalty * target
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
    scale = float(actual @ edge / (edge @ edge)) if edge @ edge > 0 else 1.0
    scale = float(np.clip(scale, 0.5, 4.0))

    power = pd.Series(raw * scale, index=teams, name="power")
    home_field = float(raw_home_field * scale)

    residuals = actual - edge * scale
    sigma = float(np.std(residuals, ddof=1)) or DEFAULT_MARGIN_SIGMA

    played = pd.concat([games["team1"], games["team2"]]).value_counts()
    played = played.reindex(teams).fillna(0).astype(int)

    ratings = Ratings(power=power, home_field=home_field, sigma=sigma, games_played=played)

    if with_resume:
        ratings.resume = wins_above_expected(games, power, home_field, sigma, neutral)
    if with_efficiency:
        offense, defense = _fit_efficiency(games, teams, X[:, :-1], ridge)
        ratings.adj_offense, ratings.adj_defense = offense, defense
    return ratings


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
) -> pd.Series:
    """Carry last season's ratings forward, regressed toward the mean.

    Rosters turn over, so last year's number is informative but stale.
    ``regression`` is the share pulled back toward average: 0 trusts last season
    completely, 1 discards it.
    """
    if previous is None or previous.empty:
        return pd.Series(0.0, index=teams)
    carried = previous.reindex(teams)
    centre = previous.median()
    carried = carried.fillna(REPLACEMENT_PRIOR)
    return (1.0 - regression) * carried + regression * centre
