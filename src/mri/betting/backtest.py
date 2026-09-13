"""Does the model beat the market? Measured strictly, with no peeking.

The rule that makes this honest: to price a game in week W, the ratings may see
only games from weeks before W, plus a prior carried from last season. That is
what a bettor actually has on Friday. A backtest that fits on the full season
and then "predicts" week 4 will look brilliant and mean nothing.

Break-even at -110 is 52.38%. Anything below that is a losing strategy no
matter how good the accuracy looks, and a result a point or two above it on a
few hundred bets is noise. The bar for calling something real here is
deliberately high, because the cost of a false positive is Ben's money.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..ingest import cfbd, registry
from ..ratings import mri2

BREAK_EVEN = 0.5238  # -110 juice
MIN_TRAIN_GAMES = 60


def _canonical(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("team1", "team2"):
        frame[column] = [registry.resolve(n, n) for n in frame[column]]
    return frame


def price_season(
    year: int,
    lines: pd.DataFrame,
    prior: pd.Series | None,
    **fit_kwargs,
) -> tuple[pd.DataFrame, pd.Series]:
    """Walk a season week by week, pricing each week from prior weeks only."""
    games = _canonical(cfbd.games(year))
    if games.empty or lines.empty:
        return pd.DataFrame(), prior

    lines = lines.set_index("game_id")
    priced = []
    final_power = prior

    for week in sorted(games["week"].unique()):
        train = games[games["week"] < week]
        if len(train) < MIN_TRAIN_GAMES:
            continue

        teams = sorted(set(train["team1"]) | set(train["team2"]))
        fbs = [t for t in teams if registry.is_fbs(t)]
        model = mri2.fit(
            train,
            prior=mri2.build_prior(prior, teams, centre_teams=fbs),
            neutral=train["neutral"],
            anchor_teams=fbs,
            with_resume=False,
            with_efficiency=False,
            **fit_kwargs,
        )
        final_power = model.power
        replacement = float(model.power.min()) - 5.0

        week_games = games[games["week"] == week]
        for row in week_games.itertuples():
            if row.game_id not in lines.index:
                continue
            line = lines.loc[row.game_id]
            if isinstance(line, pd.DataFrame):
                line = line.iloc[0]

            home = float(model.power.get(row.team2, replacement))
            away = float(model.power.get(row.team1, replacement))
            edge_site = 0.0 if row.neutral else model.home_field
            predicted = home - away + edge_site
            actual = float(row.pts2) - float(row.pts1)

            priced.append(
                {
                    "season": year,
                    "week": int(week),
                    "game_id": row.game_id,
                    "home": row.team2,
                    "away": row.team1,
                    "predicted": predicted,
                    "market": float(line["market"]),
                    "market_open": (
                        float(line["market_open"]) if pd.notna(line.get("market_open")) else np.nan
                    ),
                    "actual": actual,
                    "source": line.get("source", line.get("provider")),
                }
            )

    return pd.DataFrame(priced), final_power


def evaluate(priced: pd.DataFrame, *, against: str = "market") -> pd.DataFrame:
    """Score each bet the model would have made against a given number."""
    frame = priced.dropna(subset=[against]).copy()
    if frame.empty:
        return frame

    frame["edge"] = frame["predicted"] - frame[against]
    frame["pick_home"] = frame["edge"] > 0

    # Did the side we backed cover? A push is neither won nor lost.
    margin_vs_line = frame["actual"] - frame[against]
    frame["push"] = margin_vs_line == 0
    frame["won"] = np.where(frame["pick_home"], margin_vs_line > 0, margin_vs_line < 0)
    frame.loc[frame["push"], "won"] = False

    # Closing line value: did betting the open beat where the number landed?
    if against == "market_open" and "market" in frame:
        moved = frame["market"] - frame["market_open"]
        frame["clv"] = np.where(frame["pick_home"], moved, -moved)
    return frame


def summarize(frame: pd.DataFrame, buckets=(0, 1, 2, 3, 5, 7, 10, 100)) -> pd.DataFrame:
    """ATS record by how large the disagreement was."""
    if frame.empty:
        return pd.DataFrame()

    graded = frame[~frame["push"]]
    rows = []
    for low, high in zip(buckets, buckets[1:]):
        chunk = graded[(graded["edge"].abs() >= low) & (graded["edge"].abs() < high)]
        if chunk.empty:
            continue
        wins = int(chunk["won"].sum())
        n = len(chunk)
        rate = wins / n
        rows.append(
            {
                "edge": f"{low}-{high}" if high < 100 else f"{low}+",
                "bets": n,
                "won": wins,
                "ats": rate,
                "vs_breakeven": rate - BREAK_EVEN,
                "units": round(wins - (n - wins) * 1.1, 1),
                "p_value": _one_sided_p(wins, n),
                "clv": round(chunk["clv"].mean(), 2) if "clv" in chunk else np.nan,
            }
        )

    total = graded
    if not total.empty:
        wins, n = int(total["won"].sum()), len(total)
        rows.append(
            {
                "edge": "ALL",
                "bets": n,
                "won": wins,
                "ats": wins / n,
                "vs_breakeven": wins / n - BREAK_EVEN,
                "units": round(wins - (n - wins) * 1.1, 1),
                "p_value": _one_sided_p(wins, n),
                "clv": round(total["clv"].mean(), 2) if "clv" in total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _one_sided_p(wins: int, n: int) -> float:
    """Probability of doing this well or better by chance at break-even.

    Against 52.38%, not 50%: beating a coin flip is not the achievement, beating
    the juice is.
    """
    from scipy.stats import binomtest

    if n == 0:
        return float("nan")
    return float(binomtest(wins, n, BREAK_EVEN, alternative="greater").pvalue)


def run(seasons, provider: str = "DraftKings", **fit_kwargs) -> pd.DataFrame:
    """Price every season in order, carrying ratings forward between them."""
    from . import lines as lines_module

    prior = None
    # Prime from the season before the first one tested, so week 1 has a prior.
    first = min(seasons)
    try:
        warmup = _canonical(cfbd.games(first - 1))
        if not warmup.empty:
            teams = sorted(set(warmup["team1"]) | set(warmup["team2"]))
            fbs = [t for t in teams if registry.is_fbs(t)]
            prior = mri2.fit(
                warmup, neutral=warmup["neutral"], anchor_teams=fbs,
                with_resume=False, with_efficiency=False,
            ).power
    except Exception as exc:  # noqa: BLE001
        print(f"  no warmup season available: {exc}")

    out = []
    for season in sorted(seasons):
        book = lines_module.preferred_lines(season, provider=provider)
        if book.empty:
            print(f"  {season}: no lines")
            continue
        priced, prior = price_season(season, book, prior, **fit_kwargs)
        if not priced.empty:
            source = priced["source"].iloc[0]
            print(f"  {season}: {len(priced):>4} games priced against {source}")
            out.append(priced)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
