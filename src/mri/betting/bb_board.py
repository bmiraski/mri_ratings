"""Tonight's card: where the model disagrees with the market.

What the backtest found, stated before anything else on this page is read:

*The market is simply better than the model.* Over 16,955 walk-forward games
against ESPN Bet, the model's mean absolute error is 9.40 points and the
market's is 8.86. On Ben's own book over 2025-26 it is 9.52 against 8.81. The
market also calls more winners outright. The two numbers correlate at 0.90, so
where they disagree it is usually the model that is wrong.

*Against closing lines the model loses, in every season tested.* 50.7% ATS
across 16,954 graded bets against ESPN Bet - 51.6%, 50.1%, 50.1%, 51.9% by
season - against a 52.38% break-even. On DraftKings it is 48.3%, which is not
a near miss either.

*The one bucket that clears significance does not survive a look at it.*
Disagreements of 15 or more points went 54-25, p = 0.003 even after correction.
There are 79 of them in 16,954 games, 42 in the first season tested, and the
games themselves are November mismatches against opponents the model has barely
seen - one of them has the model favouring Mississippi Valley State by four
where the market had Hawai'i by 25.5. It won that bet while being wrong by 22
points. That bucket is measuring the model's own failures, and it does not
replicate on DraftKings (10 bets, p = 0.44).

*The faint positive is closing line value.* Betting the opener, mean CLV rises
with the size of the disagreement and is positive overall (+0.20 points). That
is worth continuing to measure and is not worth money.

So there is no basketball betting board in the sense of a card to bet. This is
an analysis view: it shows where the model and the market disagree, and it says
on its face that the disagreements have not made money. Nothing here is
filtered to look like a recommendation, which is why there is no flagged list
the way football has one.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from ..export import bb_sitedata
from ..ingest import bb_registry as registry, cbbd
from ..ratings import bb_priors, mri2
from . import bb_lines

# Below this the disagreement is inside the noise of a 9.4-point model and not
# worth a reader's attention either way.
MIN_EDGE_TO_SHOW = 3.0
SIGMA = 11.0

# How far ahead to look. Basketball schedules thinly beyond a couple of days and
# lines are not posted much earlier than that.
HORIZON_DAYS = 3


def build_board(season: int | None = None, ratings: mri2.Ratings | None = None,
                *, today: dt.date | None = None) -> dict:
    """The next few days of games, model line beside market line."""
    from scipy.stats import norm

    season = season or bb_sitedata.latest_playing_season()
    today = today or dt.date.today()

    schedule = _canonical(cbbd.games(season, completed_only=False), season)
    if schedule.empty:
        return _empty(season)

    dates = pd.to_datetime(schedule["start_date"], format="ISO8601", utc=True).dt.date
    schedule = schedule.assign(day=dates)
    played = schedule[schedule["played"]]
    slate = schedule[
        (~schedule["played"])
        & (schedule["day"] >= today)
        & (schedule["day"] <= today + dt.timedelta(days=HORIZON_DAYS))
    ]
    if slate.empty or played.empty:
        return _empty(season)

    if ratings is None:
        ratings = _rate(season, played)

    book = bb_lines.book_lines(season)
    book = book.drop_duplicates(subset="game_id").set_index("game_id") if not book.empty else pd.DataFrame()

    replacement = float(ratings.power.min()) - 5.0
    prior = bb_sitedata._prior_for(season)
    known = set(prior.index) if prior is not None else set()
    appearances = pd.concat([played["team1"], played["team2"]]).value_counts()

    rows = []
    for row in slate.itertuples():
        if not (registry.is_d1(row.team1, season=season) and registry.is_d1(row.team2, season=season)):
            continue
        home = float(ratings.power.get(row.team2, replacement))
        away = float(ratings.power.get(row.team1, replacement))
        predicted = home - away + (0.0 if row.neutral else ratings.home_field)

        market = market_open = None
        if not book.empty and row.game_id in book.index:
            line = book.loc[row.game_id]
            market = float(line["market"]) if pd.notna(line["market"]) else None
            market_open = float(line["market_open"]) if pd.notna(line.get("market_open")) else None

        reference = market_open if market_open is not None else market
        edge = (predicted - reference) if reference is not None else None

        # A team with no prior and almost no games is a guess wearing a rating,
        # and the largest disagreements in November are exactly those teams.
        thin = [
            t for t in (row.team1, row.team2)
            if t not in known and int(appearances.get(t, 0)) < 5
        ]

        rows.append(
            {
                "id": int(row.game_id),
                "start": row.start_date,
                # What the slate needs to say which tournament a game belongs to.
                "notes": _text(getattr(row, "notes", None)),
                "tournament": _text(getattr(row, "tournament", None)),
                "seasonType": _text(getattr(row, "season_type", None)),
                "gameType": _text(getattr(row, "game_type", None)),
                "homeConference": _text(getattr(row, "conf2", None)),
                "awayConference": _text(getattr(row, "conf1", None)),
                "homeSeed": _seed(getattr(row, "seed2", None)),
                "awaySeed": _seed(getattr(row, "seed1", None)),
                "day": row.day.isoformat(),
                "home": row.team2,
                "away": row.team1,
                "neutral": bool(row.neutral),
                "predicted": round(predicted, 1),
                "market": round(market, 1) if market is not None else None,
                "marketOpen": round(market_open, 1) if market_open is not None else None,
                "edge": round(edge, 1) if edge is not None else None,
                "side": (row.team2 if edge and edge > 0 else row.team1) if edge else None,
                "winProbability": round(float(norm.cdf(predicted / SIGMA)), 3),
                "confident": not thin,
                "unknownTeams": thin,
            }
        )

    rows.sort(key=lambda r: -abs(r["edge"] or 0))
    disagreements = [
        r for r in rows
        if r["edge"] is not None and abs(r["edge"]) >= MIN_EDGE_TO_SHOW and r["confident"]
    ]
    return {
        "season": season,
        "seasonLabel": bb_sitedata.season_label(season),
        "from": today.isoformat(),
        "to": (today + dt.timedelta(days=HORIZON_DAYS)).isoformat(),
        "games": rows,
        # Named for what it is. Football calls its equivalent "flagged", which
        # reads as a recommendation; nothing here has earned that word.
        "disagreements": disagreements,
        "priced": sum(1 for r in rows if r["market"] is not None),
    }


def _text(value) -> str | None:
    return value if isinstance(value, str) and value else None


def _seed(value) -> int | None:
    try:
        return int(value) if value is not None and not pd.isna(value) else None
    except (TypeError, ValueError):
        return None


def _empty(season: int) -> dict:
    return {
        "season": season,
        "seasonLabel": bb_sitedata.season_label(season),
        "from": None,
        "to": None,
        "games": [],
        "disagreements": [],
        "priced": 0,
    }


def _rate(season: int, played: pd.DataFrame) -> mri2.Ratings:
    """Fit with the preseason prior, exactly as the rankings do.

    Fitting without it was the football board's original bug: early in a season
    there is not enough evidence to separate the field, every rating collapses
    toward zero, and the model cheerfully claims a thirty-point disagreement
    with a market that is not wrong. The board must use the same prior the
    published ratings use or it is pricing a different model than the one on
    the rest of the site.
    """
    profile = mri2.BASKETBALL_PROFILE
    teams = sorted(set(played["team1"]) | set(played["team2"]))
    d1 = [t for t in teams if registry.is_d1(t, season=season)]
    return mri2.fit(
        played,
        prior=bb_priors.for_season(season, bb_sitedata._prior_for(season), teams, d1),
        neutral=played["neutral"],
        anchor_teams=d1 or None,
        compression=profile.compression,
        ridge=profile.ridge,
        home_field_prior=profile.home_field_prior,
        with_resume=False,
        with_efficiency=False,
    )


def _canonical(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("team1", "team2"):
        frame[column] = [registry.resolve(n, n, season=season) for n in frame[column]]
    return frame
