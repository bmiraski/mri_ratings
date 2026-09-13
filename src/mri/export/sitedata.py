"""Build the JSON the rankings site renders from.

Ratings are recomputed as of the end of each week, not just once at the end, so
the site can show movement, a trajectory, and an archive of what the ranking
said at the time. That week-by-week recomputation is also what makes the site
honest: a ranking that quietly rewrites its own history is not a record.

Both ratings ship. MRI 2.0 leads; MRI Classic runs beside it, frozen, so the two
can be compared week to week.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from ..ingest import boxscores, cfbd, registry
from ..ratings import classic, mri2


@dataclass
class TeamIdentity:
    team: str
    conference: str
    color: str
    alt_color: str
    logo: str | None
    abbreviation: str | None


def team_identities(year: int) -> dict[str, TeamIdentity]:
    """Conference, colors and logo per team, keyed by canonical name."""
    frame = cfbd.fbs_teams(year)
    out: dict[str, TeamIdentity] = {}
    for row in frame.itertuples():
        name = registry.resolve(row.team)
        if not name:
            continue
        meta = registry.teams()[name]
        out[name] = TeamIdentity(
            team=name,
            conference=meta.conference_display,
            color=_hex(row.color, "#444444"),
            alt_color=_hex(row.alt_color, "#888888"),
            logo=row.logo,
            abbreviation=row.abbreviation,
        )
    return out


def weekly_ratings(year: int) -> pd.DataFrame:
    """MRI 2.0 as of the end of each completed week."""
    games = _canonical(cfbd.games(year))
    if games.empty:
        return pd.DataFrame()

    prior = _prior_for(year)
    weeks = sorted(games["week"].unique())
    frames = []

    for week in weeks:
        so_far = games[games["week"] <= week]
        teams = sorted(set(so_far["team1"]) | set(so_far["team2"]))
        fbs = [t for t in teams if registry.is_fbs(t)]
        model = mri2.fit(
            so_far,
            prior=mri2.build_prior(prior, teams, centre_teams=fbs),
            neutral=so_far["neutral"],
            anchor_teams=fbs,
        )
        table = model.table()
        table = table[table["team"].map(registry.is_fbs)].copy()
        table["rank"] = range(1, len(table) + 1)
        frames.append(table.assign(week=int(week), home_field=model.home_field))

    return pd.concat(frames, ignore_index=True)


def weekly_classic(year: int) -> pd.DataFrame:
    """MRI Classic as of the end of each completed week, for comparison."""
    table = boxscores.classic_table(year)
    if table.empty:
        return pd.DataFrame()

    frames = []
    for week in sorted(table["week"].unique()):
        so_far = table[table["week"] <= week]
        teams = sorted({t for t in set(so_far["team1"]) | set(so_far["team2"])})
        result = classic.compute(so_far, teams)
        result = result[result["team"] != classic.POOLED_FCS].copy()
        result["rank"] = range(1, len(result) + 1)
        frames.append(result.assign(week=int(week)))
    return pd.concat(frames, ignore_index=True)


def build(year: int, out_dir: Path) -> dict:
    """Write site.json and return the payload."""
    identities = team_identities(year)
    modern = weekly_ratings(year)
    legacy = weekly_classic(year)
    games = _canonical(cfbd.games(year))

    if modern.empty:
        raise ValueError(f"no completed games for {year}")

    latest_week = int(modern["week"].max())
    current = modern[modern["week"] == latest_week].set_index("team")
    previous = (
        modern[modern["week"] == latest_week - 1].set_index("team")
        if latest_week > 1
        else None
    )
    classic_now = (
        legacy[legacy["week"] == latest_week].set_index("team")
        if not legacy.empty
        else None
    )

    trajectory = {
        team: group.sort_values("week")["power"].round(2).tolist()
        for team, group in modern.groupby("team")
    }
    rank_history = {
        team: group.sort_values("week")["rank"].astype(int).tolist()
        for team, group in modern.groupby("team")
    }

    records = _records(games)
    teams_payload = []
    for team, row in current.iterrows():
        identity = identities.get(team)
        last_rank = int(previous.loc[team, "rank"]) if previous is not None and team in previous.index else None
        teams_payload.append(
            {
                "team": team,
                "rank": int(row["rank"]),
                "power": round(float(row["power"]), 2),
                "resume": round(float(row["resume"]), 2) if pd.notna(row["resume"]) else None,
                "resumeRank": int(row["resume_rank"]) if pd.notna(row["resume_rank"]) else None,
                "wins": records.get(team, (0, 0))[0],
                "losses": records.get(team, (0, 0))[1],
                "conference": identity.conference if identity else registry.conference_of(team),
                "color": identity.color if identity else "#444444",
                "altColor": identity.alt_color if identity else "#888888",
                "logo": identity.logo if identity else None,
                "abbreviation": identity.abbreviation if identity else None,
                "previousRank": last_rank,
                "movement": (last_rank - int(row["rank"])) if last_rank else 0,
                "trajectory": trajectory.get(team, []),
                "rankHistory": rank_history.get(team, []),
                "classicRank": (
                    int(classic_now.loc[team, "rank"])
                    if classic_now is not None and team in classic_now.index
                    else None
                ),
                "classicRating": (
                    round(float(classic_now.loc[team, "mri"]), 2)
                    if classic_now is not None and team in classic_now.index
                    else None
                ),
                "sosRank": (
                    int(classic_now.loc[team, "sos_rank"])
                    if classic_now is not None and team in classic_now.index
                    else None
                ),
            }
        )

    conferences = (
        pd.DataFrame(teams_payload)
        .groupby("conference")["power"]
        .agg(["mean", "count", "max"])
        .sort_values("mean", ascending=False)
        .round(2)
        .reset_index()
        .to_dict("records")
    )

    payload = {
        "season": year,
        "week": latest_week,
        "generated": pd.Timestamp.utcnow().isoformat(),
        "homeField": round(float(current["home_field"].iloc[0]), 2),
        "gamesRated": int(len(games)),
        "teams": sorted(teams_payload, key=lambda t: t["rank"]),
        "conferences": conferences,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "site.json").write_text(json.dumps(payload, indent=2))
    return payload


def _records(games: pd.DataFrame) -> dict[str, tuple[int, int]]:
    tally: dict[str, list[int]] = {}
    for row in games.itertuples():
        for team, won in ((row.team1, row.win1), (row.team2, row.win2)):
            if not registry.is_fbs(team):
                continue
            entry = tally.setdefault(team, [0, 0])
            entry[0 if won == 1.0 else 1] += 1
    return {k: (v[0], v[1]) for k, v in tally.items()}


def _canonical(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("team1", "team2"):
        frame[column] = [registry.resolve(n, n) for n in frame[column]]
    return frame


def _prior_for(year: int) -> pd.Series | None:
    """Last season's final ratings, from the chain build_current.py wrote."""
    path = Path(__file__).resolve().parents[3] / "data" / "parquet" / "current_ratings.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    previous = frame[frame["season"] == year - 1]
    if previous.empty:
        return None
    return previous.set_index("team")["power"]


def _hex(value, fallback: str) -> str:
    if not value or not isinstance(value, str):
        return fallback
    value = value.strip()
    if not value.startswith("#"):
        value = "#" + value
    return value if len(value) == 7 else fallback
