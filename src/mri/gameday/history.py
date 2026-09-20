"""Every weekly choice ESPN made from 2014 to 2025, laid out as a choice problem.

For each week, the games GameDay could have gone to (every FBS-vs-FBS game that
week) and the one it did. The features of each candidate are computed from ratings
that existed *before that week*, so nothing about the outcome leaks into the
question of whether the game looked big.

Weeks that are not a choice among FBS games are left out of the fit and counted
separately: the FCS and off-schedule stops, and Army-Navy, which GameDay attended
every year from 2014 to 2021 and none since.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..ingest import cfbd
from ..ratings import mri2, prior_fit
from . import features

DATA = Path(__file__).resolve().parents[3] / "data" / "gameday_locations.json"
FIRST_WEEK = 5              # earlier weeks are openers and neutral-site kickoffs, a different question
YEARS = [y for y in range(2014, 2026) if y != 2020]


@dataclass
class Week:
    year: int
    week: int
    games: pd.DataFrame          # the candidates, with team1 (away) and team2 (home)
    X: np.ndarray                # (n_candidates, n_features)
    pick: int                    # index of the game GameDay chose


def load(path: Path = DATA) -> dict:
    return json.loads(path.read_text())


def stop_weeks(stops: list[dict], games: pd.DataFrame) -> list[int | None]:
    """The week of each stop: from the game it went to, or from its date when it went elsewhere."""
    starts = pd.to_datetime(games["start_date"], utc=True)
    by_week = starts.groupby(games["week"]).median()
    out = []
    for s in stops:
        a, b = s["teams"]
        match = games[((games["team1"] == a) & (games["team2"] == b)) | ((games["team1"] == b) & (games["team2"] == a))]
        date = pd.Timestamp(s["date"], tz="UTC")
        if len(match):
            near = match.iloc[(pd.to_datetime(match["start_date"], utc=True) - date).abs().argmin()]
            out.append(int(near["week"]))
        else:
            out.append(int((by_week - date).abs().idxmin()))
    return out


def hosts_before(data: dict, year: int, span: int = 3) -> dict[str, int]:
    """How often each school hosted in the ``span`` seasons before ``year``."""
    counts: dict[str, int] = {}
    for y in range(year - span, year):
        for s in data["seasons"].get(str(y), []):
            if s["host"]:
                counts[s["host"]] = counts.get(s["host"], 0) + 1
        for h in data.get("hostsOnly", {}).get(str(y), []):
            counts[h] = counts.get(h, 0) + 1
    return counts


def appearance_rates(data: dict, year: int, span: int = 5) -> dict[str, float]:
    """GameDay stops per season, either side of the game, over the ``span`` seasons before ``year``.

    Only seasons with full records count (2013 on); a team's "brand" is how often
    ESPN has wanted it lately, whatever it was doing that week.
    """
    seasons = [y for y in range(year - span, year) if str(y) in data["seasons"]]
    counts: dict[str, int] = {}
    for y in seasons:
        for s in data["seasons"][str(y)]:
            for t in s["teams"]:
                counts[t] = counts.get(t, 0) + 1
    return {t: c / max(len(seasons), 1) for t, c in counts.items()}


def build(years=YEARS, data: dict | None = None) -> list[Week]:
    data = data or load()
    power_np, fbs_np = prior_fit.season_ratings()
    all_games = {y: cfbd.games(y) for y in range(min(years) - 2, max(years) + 1)}
    weeks: list[Week] = []

    for year in years:
        games = all_games[year]
        games = games[games["season_type"] == "regular"].reset_index(drop=True)
        fbs = set(fbs_np[year])
        stops = data["seasons"].get(str(year), [])
        s_weeks = stop_weeks(stops, games)
        recent = hosts_before(data, year)
        brand = appearance_rates(data, year)
        last_rank = power_np[year - 1].rank(ascending=False)
        meetings = {y: {(frozenset((r.team1, r.team2)), int(r.week)) for r in all_games[y].itertuples()}
                    for y in (year - 1, year - 2) if y in all_games}
        teams = sorted(set(games["team1"]) | set(games["team2"]))
        prior = mri2.build_prior(power_np[year - 1], teams, centre_teams=sorted(fbs))

        for week in sorted(games["week"].unique()):
            if week < FIRST_WEEK:
                continue
            wk_stops = [(s, w) for s, w in zip(stops, s_weeks) if w == week]
            if not wk_stops or any(s["kind"] in ("svc", "fcs") for s, _ in wk_stops):
                continue                        # not a choice among FBS games
            pick = wk_stops[0][0]

            so_far = games[games["week"] < week]
            model = mri2.fit(so_far, prior=prior, neutral=so_far["neutral"], anchor_teams=sorted(fbs),
                             with_efficiency=False)
            table = model.table()
            table = table[table["team"].isin(fbs)].reset_index(drop=True)
            table["cr"] = features.committee_score_rank(table["power"].to_numpy(), table["resume"].to_numpy())
            rank = table.set_index("team")["cr"]
            power = table.set_index("team")["power"]

            cand = games[(games["week"] == week) & games["team1"].isin(fbs) & games["team2"].isin(fbs)].reset_index(drop=True)
            if cand.empty:
                continue
            losses = {}
            for r in so_far.itertuples():
                loser = r.team1 if r.pts1 < r.pts2 else r.team2
                losses[loser] = losses.get(loser, 0) + 1
            seen: dict[str, int] = {}
            for s, w in zip(stops, s_weeks):
                if w is not None and w < week:
                    for t in s["teams"]:
                        seen[t] = seen.get(t, 0) + 1

            home, away = cand["team2"], cand["team1"]
            spread = (home.map(power) - away.map(power)).to_numpy() + np.where(cand["neutral"], 0.0, model.home_field)
            rival = np.array([
                all(any(p == frozenset((h, a)) and abs(w - week) <= 1 for p, w in meetings[y]) for y in meetings)
                and len(meetings) == 2 for h, a in zip(home, away)])
            X = features.matrix(
                rank_home=home.map(rank).fillna(features.RANK_CAP).to_numpy(),
                rank_away=away.map(rank).fillna(features.RANK_CAP).to_numpy(),
                losses_home=home.map(losses).fillna(0).to_numpy(),
                losses_away=away.map(losses).fillna(0).to_numpy(),
                spread=spread, neutral=cand["neutral"].to_numpy(),
                elite=features.elite_conference(cand["conf2"], cand["conf1"], cand["neutral"].to_numpy()),
                repeat_home=home.map(seen).fillna(0).to_numpy(),
                repeat_away=away.map(seen).fillna(0).to_numpy(),
                host_recent=home.map(recent).fillna(0).to_numpy(),
                rivalry=rival,
                last_rank_home=home.map(last_rank).fillna(features.RANK_CAP).to_numpy(),
                last_rank_away=away.map(last_rank).fillna(features.RANK_CAP).to_numpy(),
                brand_home=home.map(brand).fillna(0).to_numpy(),
                brand_away=away.map(brand).fillna(0).to_numpy(),
            )
            a, b = pick["teams"]
            hit = cand.index[((cand["team1"] == a) & (cand["team2"] == b)) | ((cand["team1"] == b) & (cand["team2"] == a))]
            if len(hit) != 1:
                continue
            weeks.append(Week(year, int(week), cand, X, int(hit[0])))
        print(f"  {year}: {sum(1 for w in weeks if w.year == year)} weeks")
    return weeks
