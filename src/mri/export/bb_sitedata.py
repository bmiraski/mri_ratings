"""Build the JSON the basketball rankings site renders from.

The payload has the same shape as football's, because the site generator is
shared and the alternative - a second set of templates - would drift within a
month. Where the two sports differ, they differ in the payload rather than in
the renderer.

Three differences are real and worth naming.

*There is no week number.* Football hands you one; basketball is dated. Weeks
here are counted from the Monday of the season's first game, which gives the
same weekly snapshot, movement and archive the football site has, without
pretending the API supplied something it did not.

*There is no Classic column.* MRI Basketball Classic needs rebounds and
turnovers per game, which the games feed does not carry. Publishing it would
cost a box-score pull per season for a comparison nobody asked for, so the
basketball site ships one rating and says so.

*The season may be over.* Basketball runs November to April, so for half the
year the honest thing to show is the completed season marked final, not an
empty page for a season that has not started. The season shown is the most
recent one with games in it.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from ..ingest import bb_registry as registry, cbbd
from ..ratings import mri2
from . import common

RATINGS = Path(__file__).resolve().parents[3] / "data" / "parquet" / "bb_ratings.parquet"

# Team colours come back from the API without the hash, and a few are missing.
FALLBACK = "#444444"
FALLBACK_ALT = "#888888"


def season_label(season: int) -> str:
    """2026 is the 2025-26 season, which is what a reader wants to see."""
    return f"{season - 1}–{str(season)[2:]}"


@lru_cache(maxsize=1)
def latest_playing_season(limit: int = 3) -> int:
    """The most recent season with completed games in it.

    In September nothing has been played yet, so this returns last season and
    the site marks it final. In December it returns the live one.

    Cached because both ``build`` and ``build_full`` ask, and each answer walks
    a season's worth of date windows. Uncached it doubled the API calls of every
    run for a value that cannot change mid-build.
    """
    from ..ingest.bb_registry import CURRENT_SEASON

    for season in range(CURRENT_SEASON, CURRENT_SEASON - limit, -1):
        if not cbbd.games(season).empty:
            return season
    raise ValueError("no season with completed games")


def _canonical(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("team1", "team2"):
        frame[column] = [registry.resolve(n, n, season=season) for n in frame[column]]
    return frame


def _weeks(frame: pd.DataFrame) -> pd.Series:
    """Week index counted from the Monday on or before the first game."""
    dates = pd.to_datetime(frame["start_date"], format="ISO8601", utc=True)
    first = dates.min().normalize()
    start = first - pd.Timedelta(days=int(first.dayofweek))
    return ((dates - start).dt.days // 7 + 1).astype(int)


def _prior_for(season: int) -> pd.Series | None:
    """The previous season's final ratings from the chained build."""
    if not RATINGS.exists():
        return None
    frame = pd.read_parquet(RATINGS)
    previous = frame[frame["season"] == season - 1]
    return None if previous.empty else previous.set_index("team")["power"]


def weekly_ratings(season: int) -> pd.DataFrame:
    """MRI 2.0 as of the end of each week of the season."""
    games = _canonical(cbbd.games(season), season)
    if games.empty:
        return pd.DataFrame()
    games = games.assign(week=_weeks(games))

    profile = mri2.BASKETBALL_PROFILE
    prior = _prior_for(season)
    frames = []

    for week in sorted(games["week"].unique()):
        so_far = games[games["week"] <= week]
        teams = sorted(set(so_far["team1"]) | set(so_far["team2"]))
        d1 = [t for t in teams if registry.is_d1(t, season=season)]
        # Under about a hundred games the solve is describing noise, and a
        # published week-one ranking of 350 teams off three days of play is
        # worse than no ranking at all.
        if len(so_far) < 100 or not d1:
            continue

        model = mri2.fit(
            so_far,
            prior=mri2.build_prior(prior, teams, profile.prior_regression, centre_teams=d1),
            neutral=so_far["neutral"],
            anchor_teams=d1,
            compression=profile.compression,
            ridge=profile.ridge,
            home_field_prior=profile.home_field_prior,
            with_efficiency=False,
        )
        table = model.table()
        table = table[table["team"].map(lambda t: registry.is_d1(t, season=season))].copy()
        table["rank"] = range(1, len(table) + 1)
        frames.append(table.assign(week=int(week), home_field=model.home_field))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def team_identities(season: int) -> dict[str, dict]:
    """Conference and colours per team.

    No logos. CFBD's CDN carries football schools only, so 227 of the 365 would
    404; the design's colour mark covers every team instead of most of them.
    """
    frame = cbbd.teams(season)
    out = {}
    for row in frame.itertuples():
        out[row.team] = {
            "conference": row.conference or "Independent",
            "color": _hex(row.color, FALLBACK),
            "altColor": _hex(row.alt_color, FALLBACK_ALT),
            "abbreviation": row.abbreviation,
            "logo": None,
        }
    return out


def _records(games: pd.DataFrame, season: int) -> dict[str, tuple[int, int]]:
    tally: dict[str, list[int]] = {}
    for row in games.itertuples():
        for team, won in ((row.team1, row.win1), (row.team2, row.win2)):
            if not registry.is_d1(team, season=season):
                continue
            entry = tally.setdefault(team, [0, 0])
            entry[0 if won == 1.0 else 1] += 1
    return {k: (v[0], v[1]) for k, v in tally.items()}


def build(season: int, out_dir: Path) -> dict:
    """Write bb.json and return the payload."""
    modern = weekly_ratings(season)
    if modern.empty:
        raise ValueError(f"no completed games for {season_label(season)}")

    games = _canonical(cbbd.games(season), season)
    identities = team_identities(season)

    latest = int(modern["week"].max())
    current = modern[modern["week"] == latest].set_index("team")
    earlier = sorted(w for w in modern["week"].unique() if w < latest)
    previous = modern[modern["week"] == earlier[-1]].set_index("team") if earlier else None

    trajectory = {
        team: group.sort_values("week")["power"].round(2).tolist()
        for team, group in modern.groupby("team")
    }
    rank_history = {
        team: group.sort_values("week")["rank"].astype(int).tolist()
        for team, group in modern.groupby("team")
    }

    records = _records(games, season)
    teams_payload = []
    for team, row in current.iterrows():
        identity = identities.get(team, {})
        last = (
            int(previous.loc[team, "rank"])
            if previous is not None and team in previous.index
            else None
        )
        teams_payload.append(
            {
                "team": team,
                "rank": int(row["rank"]),
                "power": round(float(row["power"]), 2),
                "resume": round(float(row["resume"]), 2) if pd.notna(row["resume"]) else None,
                "resumeRank": int(row["resume_rank"]) if pd.notna(row["resume_rank"]) else None,
                "wins": records.get(team, (0, 0))[0],
                "losses": records.get(team, (0, 0))[1],
                "conference": identity.get("conference")
                or registry.conference_of(team, season=season)
                or "Independent",
                "color": identity.get("color", FALLBACK),
                "altColor": identity.get("altColor", FALLBACK_ALT),
                "logo": None,
                "abbreviation": identity.get("abbreviation"),
                "previousRank": last,
                "movement": (last - int(row["rank"])) if last else 0,
                "trajectory": trajectory.get(team, []),
                "rankHistory": rank_history.get(team, []),
                "classicRank": None,
                "classicRating": None,
                "sosRank": None,
            }
        )

    conferences = common.conference_strength(teams_payload)

    finished = season < latest_playing_season() or _season_over(games)
    payload = {
        "sport": "basketball",
        "season": season,
        "seasonLabel": season_label(season),
        "week": latest,
        "weeks": [int(w) for w in sorted(modern["week"].unique())],
        "periodLabel": "Final" if finished else f"Week {latest}",
        "finished": finished,
        "generated": pd.Timestamp.utcnow().isoformat(),
        "homeField": round(float(current["home_field"].iloc[0]), 2),
        "gamesRated": int(len(games)),
        "teams": sorted(teams_payload, key=lambda t: t["rank"]),
        "conferences": conferences,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bb.json").write_text(json.dumps(payload, indent=2))
    return payload


def _season_over(games: pd.DataFrame) -> bool:
    """A season is done when its last game is more than a fortnight behind us."""
    if games.empty:
        return True
    last = pd.to_datetime(games["start_date"], format="ISO8601", utc=True).max()
    return (pd.Timestamp.utcnow().tz_localize(None) - last.tz_localize(None)).days > 14


def team_details(season: int, payload: dict) -> dict:
    """Per-team schedule with each game measured against expectation."""
    from scipy.stats import norm

    schedule = _canonical(cbbd.games(season, completed_only=False), season)
    schedule = schedule.assign(week=_weeks(schedule))
    power = {t["team"]: t["power"] for t in payload["teams"]}
    ranks = {t["team"]: t["rank"] for t in payload["teams"]}
    home_field = payload["homeField"]
    sigma = 11.0  # basketball margins are tighter than football's 16.5
    replacement = min(power.values()) - 8 if power else -30.0

    def rating(team: str) -> float:
        return power.get(team, replacement)

    details = {t["team"]: {"played": [], "upcoming": []} for t in payload["teams"]}

    for row in schedule.itertuples():
        edge = 0.0 if row.neutral else home_field
        expected_home = rating(row.team2) - rating(row.team1) + edge

        for team, opponent, site, expected in (
            (row.team2, row.team1, "n" if row.neutral else "vs", expected_home),
            (row.team1, row.team2, "n" if row.neutral else "at", -expected_home),
        ):
            if team not in details:
                continue
            entry = {
                "week": int(row.week),
                "opponent": opponent,
                "opponentRank": ranks.get(opponent),
                "opponentPower": round(rating(opponent), 1),
                "site": site,
                "expected": round(expected, 1),
            }
            if row.played:
                scored = row.pts2 if team == row.team2 else row.pts1
                allowed = row.pts1 if team == row.team2 else row.pts2
                margin = scored - allowed
                entry.update(
                    {
                        "scored": int(scored),
                        "allowed": int(allowed),
                        "won": margin > 0,
                        "margin": int(margin),
                        "performance": round(margin - expected, 1),
                    }
                )
                details[team]["played"].append(entry)
            else:
                entry["winProbability"] = round(float(norm.cdf(expected / sigma)), 3)
                details[team]["upcoming"].append(entry)

    for detail in details.values():
        detail["played"].sort(key=lambda g: g["week"])
        detail["upcoming"].sort(key=lambda g: g["week"])
        played, remaining = detail["played"], detail["upcoming"]
        detail["remainingDifficulty"] = (
            round(sum(g["opponentPower"] for g in remaining) / len(remaining), 1)
            if remaining else None
        )
        detail["playedDifficulty"] = (
            round(sum(g["opponentPower"] for g in played) / len(played), 1) if played else None
        )
        detail["bestWin"] = max(
            (g for g in played if g["won"]), key=lambda g: g["opponentPower"], default=None
        )
        detail["worstLoss"] = min(
            (g for g in played if not g["won"]), key=lambda g: g["opponentPower"], default=None
        )
    return details


def build_full(season: int | None, out_dir: Path) -> dict:
    """The payload with the per-team detail the team pages need.

    The detail is returned but not written. For 365 teams it is 4MB, it is
    rendered into the team pages anyway, and bb.json is committed - so writing it
    would put a 4MB churning blob in the repository to serve no reader.
    """
    season = season or latest_playing_season()
    payload = build(season, out_dir)
    payload["details"] = team_details(season, payload)
    return payload


def _hex(value, fallback: str) -> str:
    if not value or not isinstance(value, str):
        return fallback
    value = value.strip()
    if not value.startswith("#"):
        value = "#" + value
    return value if len(value) == 7 else fallback
