"""Basketball preseason priors that know who is on the roster.

MRI 2.0 starts every season from a prior, and basketball's was last season's
rating pulled 35% of the way to average for every team. That misses the one thing
that turns over faster in college basketball than anywhere else: the roster. A
team can lose all five starters to graduation, the draft and the transfer portal
and be rated in November as the team that played last March.

Three things are known before a game is played, and carry real information:

*Returning production* - the share of last season's win shares still on the
roster, with a returning starter counted for what he produced and not merely for
having stayed.

*Incoming production* - what the roster's newcomers did last season somewhere
else. The transfer portal made this as large a part of a roster as the returning
players, and it is the piece a returning-minutes number cannot see.

*The freshman class* - how highly the incoming recruits were rated.

Rosters for a coming season are posted late, and the feed says nothing until then
- an empty list, not an error - so there are two versions of the prior, and a
team gets the one its data supports:

*With a roster*: returning win shares, incoming win shares and the freshman class.
*Before one*: only what the previous season and the draft already imply - how many
of last year's minutes belong to players in their fourth year or later or headed
to the NBA draft, and the freshman class. That gets about half the improvement.

The regression is on last season's rating and these features, fitted to how good
the team turned out to be (``scripts/fit_bb_prior.py``, coefficients in
``data/bb_prior_model.json``). Two adjustments, both found by walk-forward
testing and not by argument:

*The level is kept where the old prior had it.* A regression fitted to a rating
scale can sit a point or two off the level the rest of the pipeline uses, and that
offset lands on every Division I team's games against non-Division I opponents,
which are most of the games in November.

*The spread is pulled in to 70%.* The solver already weighs the prior against the
games at a fixed ratio tuned for the old prior's spread, so a more informative
prior needs to be a little more cautious in scale, not less. At full spread the
new prior is worse than the old one; at 70% it is better in every part of the
season; at 50% it starts to give the gain back.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..ingest import bb_registry as registry
from . import mri2

ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "data" / "bb_prior_model.json"
PARQUET = ROOT / "data" / "parquet"
MIN_ROSTER = 8               # a posted roster this long is a roster; shorter is a list still being filled
VETERAN_SEASONS = 4          # a fourth Division I season, counting last year's, is likely a last one
TEAM_MINUTES = 8000.0        # about what a team's players play in a season, to put incoming minutes on a scale


def load_model(path: Path = MODEL_PATH) -> dict | None:
    if not path.exists():
        return None
    model = json.loads(path.read_text())
    return model if {"roster", "noRoster", "tighten"} <= set(model) else None


@functools.lru_cache(maxsize=1)
def _tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """The played-player table (canonical names, one row per player-season), recruits and draft picks."""
    try:
        players = pd.read_parquet(PARQUET / "bb_players.parquet")
        recruits = pd.read_parquet(PARQUET / "bb_recruits.parquet")
        draft = pd.read_parquet(PARQUET / "bb_draft.parquet")
    except FileNotFoundError:
        return None
    return _played(players), recruits, draft


def _played(players: pd.DataFrame) -> pd.DataFrame:
    """One row per player per season, canonical team names, production floored at zero."""
    frame = players[players["minutes"].fillna(0) > 0].copy()
    frame["ws"] = frame["win_shares"].fillna(0).clip(lower=0)
    frame["team_c"] = [registry.resolve(t, t, season=int(s)) for t, s in zip(frame["team"], frame["season"])]
    return frame.sort_values("minutes", ascending=False).drop_duplicates(["season", "athlete_id"])


def features(season: int, d1: list[str], played: pd.DataFrame, recruits: pd.DataFrame, draft: pd.DataFrame,
             roster: dict[str, set] | None = None) -> pd.DataFrame:
    """What is known about each team's roster before ``season``, one row per team.

    ``roster`` maps a team to the athlete ids on its roster for ``season``. Where a
    team has none (or too few), the roster-dependent columns are NaN and the team
    is handled by the version of the model that does not need them.
    """
    prev = played[played["season"] == season - 1]
    first = played.groupby("athlete_id")["season"].min()
    by_id = prev.set_index("athlete_id")
    team_minutes = prev.groupby("team_c")["minutes"].sum()
    team_ws = prev.groupby("team_c")["ws"].sum()
    drafted = set(draft.loc[draft["year"] == season - 1, "athlete_id"].dropna())
    class_ = recruits[recruits["year"] == season - 1].copy()
    class_["team_c"] = [registry.resolve(t, t, season=season) for t in class_["team"]]

    rows = []
    for team in d1:
        last = prev[prev["team_c"] == team]
        minutes = team_minutes.get(team, 0.0)
        ws = team_ws.get(team, 0.0)
        veterans = last[(season - 1 - last["athlete_id"].map(first)) >= VETERAN_SEASONS - 1]
        gone = last[last["athlete_id"].isin(drafted)]
        commits = class_[class_["team_c"] == team]
        row = {
            "team": team,
            "vet_min": veterans["minutes"].sum() / minutes if minutes > 0 else np.nan,
            "draft_min": gone["minutes"].sum() / minutes if minutes > 0 else np.nan,
            "frosh": float((commits["rating"].fillna(0.8) - 0.8).clip(lower=0).sum()),
            "ret_ws": np.nan, "in_ws": np.nan, "has_roster": False,
        }
        ids = (roster or {}).get(team)
        if ids is not None and len(ids) >= MIN_ROSTER and ws > 0:
            kept = last[last["athlete_id"].isin(ids)]
            arrivals = [i for i in ids if i in by_id.index and by_id.loc[i, "team_c"] != team]
            row["ret_ws"] = kept["ws"].sum() / ws
            row["in_ws"] = float(by_id.loc[arrivals, "ws"].sum()) if arrivals else 0.0
            row["has_roster"] = True
        rows.append(row)
    return pd.DataFrame(rows).set_index("team")


def predict(model: dict, previous: float, row: pd.Series) -> float | None:
    """The regression's answer for one team, or None if its data cannot support one."""
    version = model["roster"] if row["has_roster"] else model["noRoster"]
    c = version["coefficients"]
    values = [row[f] for f in version["features"]]
    if any(pd.isna(v) for v in values) or pd.isna(previous):
        return None
    return float(c["intercept"] + c["last_season"] * previous + sum(c[f] * row[f] for f in version["features"]))


def preseason_prior(previous: pd.Series | None, teams: list[str], d1: list[str], feats: pd.DataFrame | None,
                    model: dict | None) -> pd.Series:
    """The regression's prior where it applies and the old rule everywhere else."""
    profile = mri2.BASKETBALL_PROFILE
    old = mri2.build_prior(previous, teams, profile.prior_regression, centre_teams=d1 or None,
                           outsiders_to_replacement=False)
    if model is None or feats is None or previous is None or previous.empty:
        return old
    replaced, values = [], []
    for team in d1:
        if team in feats.index and team in previous.index:
            value = predict(model, float(previous[team]), feats.loc[team])
            if value is not None:
                replaced.append(team)
                values.append(value)
    if not replaced:
        return old
    new = np.array(values)
    out = old.copy()
    out.loc[replaced] = (new - new.mean()) * float(model["tighten"]) + old[replaced].to_numpy().mean()
    return out


def rosters_for(season: int, d1: list[str]) -> dict[str, set] | None:
    """Rosters for a season, or None if none are to be had.

    A season already played has its roster in the player stats (everyone who played
    for a team). A coming season has one only once the schools post it; until then
    the feed answers with empty lists and this returns None.
    """
    tables = _tables()
    if tables is not None:
        cur = tables[0][tables[0]["season"] == season]
        if not cur.empty:
            return {t: set(g["athlete_id"]) for t, g in cur.groupby("team_c")}
    try:
        from ..ingest import cbbd

        # Refreshed once per run: rosters fill in through the fall, and a cached
        # empty list must not be mistaken for a roster that never arrives.
        frame = cbbd.rosters(season, refresh=True)
    except Exception:  # noqa: BLE001 - a missing roster costs accuracy, never the build
        return None
    if frame.empty:
        return None
    frame["team_c"] = [registry.resolve(t, t, season=season) for t in frame["team"]]
    posted = {t: set(g["athlete_id"].dropna()) for t, g in frame.groupby("team_c") if t}
    return posted if sum(len(v) >= MIN_ROSTER for v in posted.values()) else None


def for_season(season: int, previous: pd.Series | None, teams: list[str], d1: list[str]) -> pd.Series:
    """What a build should use as the prior for ``season``.

    Falls back to the old prior if there is no model, no player data, or no last
    season to build from, so a gap in the data costs accuracy and not the build.
    """
    model = load_model()
    tables = _tables()
    if model is None or tables is None or previous is None:
        return mri2.build_prior(previous, teams, mri2.BASKETBALL_PROFILE.prior_regression, centre_teams=d1 or None,
                               outsiders_to_replacement=False)
    feats = features(season, d1, *tables, roster=rosters_for(season, d1))
    return preseason_prior(previous, teams, d1, feats, model)


def status(season: int, d1: list[str]) -> dict:
    """Whether the season's rosters are posted yet, for the method page to say so."""
    roster = rosters_for(season, d1) or {}
    ready = sum(1 for t in d1 if len(roster.get(t, ())) >= MIN_ROSTER)
    return {"season": season, "teamsWithRosters": ready, "teams": len(d1),
            "mode": "roster" if ready >= 0.6 * max(len(d1), 1) else "before rosters"}
