"""Does the basketball model beat the market? Measured strictly, with no peeking.

Same rule as football, translated to a sport that plays every night: to price a
game tipping on day D, the ratings may see only games that finished before day
D, plus a prior carried from last season. That is what a bettor has in the
morning. Football could slice by week because a week is the unit it plays in;
here the slice is the calendar day, and the model is refit for every day that
has a priced game on it - roughly 150 refits a season.

Slicing coarser would be cheaper and would also cheat. A model refit weekly and
used to price Saturday's games has seen Thursday's results when it prices them.

Break-even at -110 is 52.38%. Anything below that loses money however good the
accuracy looks, and a point or two above it on a few hundred bets is noise.
Eight edge buckets get tested, so a p-value has to clear a Bonferroni-corrected
bar rather than a bare 0.05 - test enough buckets and one clears 0.05 by
construction.

What the data will support, and what it will not: ESPN Bet covers four seasons
and 17,783 priced games, which is enough to bucket. DraftKings - Ben's book -
covers one season. One season cannot tell an edge from a season, so the
DraftKings number is a cross-check on the ESPN Bet finding and never the
finding itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..ingest import bb_registry as registry, cbbd
from ..ratings import mri2
from .backtest import BREAK_EVEN, _one_sided_p, evaluate  # noqa: F401 - shared grading

# Below this the fit is mostly prior, which is honest - a bettor in November
# really is working off last season - but under a few hundred games the solve is
# describing noise rather than shrinking toward the prior in any useful way.
MIN_TRAIN_GAMES = 300

# Buckets tested, and therefore the multiple-comparison correction that applies
# to every p-value reported against them.
BUCKETS = (0, 1, 2, 3, 5, 7, 10, 15, 100)
COMPARISONS = len(BUCKETS) - 1


def _canonical(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("team1", "team2"):
        frame[column] = [registry.resolve(n, n, season=season) for n in frame[column]]
    return frame


def _days(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["start_date"], format="ISO8601", utc=True).dt.date


def price_season(
    year: int,
    lines: pd.DataFrame,
    prior: pd.Series | None,
    **fit_kwargs,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Walk a season day by day, pricing each day from earlier days only."""
    games = _canonical(cbbd.games(year), year)
    if games.empty or lines.empty:
        return pd.DataFrame(), prior

    games = games.assign(day=_days(games)).sort_values("start_date")
    priced_ids = set(lines["game_id"])
    book = lines.drop_duplicates(subset="game_id").set_index("game_id")

    profile = mri2.BASKETBALL_PROFILE
    out = []
    final_power = prior

    for day, today in games.groupby("day", sort=True):
        if not priced_ids & set(today["game_id"]):
            continue
        train = games[games["day"] < day]
        if len(train) < MIN_TRAIN_GAMES:
            continue

        teams = sorted(set(train["team1"]) | set(train["team2"]))
        d1 = [t for t in teams if registry.is_d1(t, season=year)]
        model = mri2.fit(
            train,
            prior=mri2.build_prior(prior, teams, profile.prior_regression, centre_teams=d1 or None,
                                   outsiders_to_replacement=False),
            neutral=train["neutral"],
            anchor_teams=d1 or None,
            compression=profile.compression,
            ridge=profile.ridge,
            home_field_prior=profile.home_field_prior,
            with_resume=False,
            with_efficiency=False,
            **fit_kwargs,
        )
        final_power = model.power
        replacement = float(model.power.min()) - 5.0

        for row in today.itertuples():
            if row.game_id not in book.index:
                continue
            line = book.loc[row.game_id]
            home = float(model.power.get(row.team2, replacement))
            away = float(model.power.get(row.team1, replacement))
            predicted = home - away + (0.0 if row.neutral else model.home_field)

            out.append(
                {
                    "season": year,
                    "day": day,
                    "game_id": row.game_id,
                    "home": row.team2,
                    "away": row.team1,
                    # A team the fit has never seen is priced off the floor,
                    # which is a guess rather than a rating. Flagged so the
                    # grading can be run with and without them.
                    "known": row.team2 in model.power.index and row.team1 in model.power.index,
                    "predicted": predicted,
                    "market": float(line["market"]),
                    "market_open": (
                        float(line["market_open"]) if pd.notna(line.get("market_open")) else np.nan
                    ),
                    "actual": float(row.pts2) - float(row.pts1),
                    "provider": line.get("provider"),
                }
            )

    return pd.DataFrame(out), final_power


def summarize(frame: pd.DataFrame, buckets=BUCKETS) -> pd.DataFrame:
    """ATS record by how large the disagreement was, honestly corrected.

    Football's version reports a bare one-sided p-value per bucket. With eight
    buckets that is eight chances to clear 0.05, so one of them usually does.
    ``significant`` here is the corrected verdict; ``p_value`` is left raw so the
    correction is visible rather than baked in.
    """
    if frame.empty:
        return pd.DataFrame()

    graded = frame[~frame["push"]]
    rows = []

    def record(label, chunk):
        if chunk.empty:
            return
        wins, n = int(chunk["won"].sum()), len(chunk)
        rate = wins / n
        p = _one_sided_p(wins, n)
        rows.append(
            {
                "edge": label,
                "bets": n,
                "won": wins,
                "ats": rate,
                "vs_breakeven": rate - BREAK_EVEN,
                "units": round(wins - (n - wins) * 1.1, 1),
                "p_value": p,
                "significant": bool(p < 0.05 / COMPARISONS),
                "clv": round(chunk["clv"].mean(), 2) if "clv" in chunk else np.nan,
            }
        )

    for low, high in zip(buckets, buckets[1:]):
        label = f"{low}-{high}" if high < 100 else f"{low}+"
        record(label, graded[(graded["edge"].abs() >= low) & (graded["edge"].abs() < high)])
    record("ALL", graded)
    return pd.DataFrame(rows)


def run(seasons, provider: str, **fit_kwargs) -> pd.DataFrame:
    """Price every season in order, carrying ratings forward between them."""
    from . import bb_lines

    prior = _warmup(min(seasons))
    out = []
    for season in sorted(seasons):
        book = bb_lines.book_lines(season, provider)
        if book.empty:
            print(f"  {season}: no {provider} lines")
            continue
        priced, prior = price_season(season, book, prior, **fit_kwargs)
        if not priced.empty:
            print(f"  {season}: {len(priced):>5} games priced against {provider}")
            out.append(priced)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _warmup(first: int) -> pd.Series | None:
    """Prime from the season before the first one tested, so November has a prior."""
    games = _canonical(cbbd.games(first - 1), first - 1)
    if games.empty:
        return None
    teams = sorted(set(games["team1"]) | set(games["team2"]))
    d1 = [t for t in teams if registry.is_d1(t, season=first - 1)]
    profile = mri2.BASKETBALL_PROFILE
    return mri2.fit(
        games,
        neutral=games["neutral"],
        anchor_teams=d1 or None,
        compression=profile.compression,
        ridge=profile.ridge,
        home_field_prior=profile.home_field_prior,
        with_resume=False,
        with_efficiency=False,
    ).power
