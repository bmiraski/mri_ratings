"""Home field, per stadium and per trip.

MRI 2.0 gives every home team the same few points. This module asks whether
that number should move with the stadium (some crowds are louder than others)
and with the circumstances of the trip: altitude, miles travelled, time zones
crossed, and kickoffs that land in the visitor's morning.

It is a two-stage fit on out-of-sample residuals, so nothing here touches the
rating solve:

1. Walk every season forward. Each game is predicted from ratings fitted only
   on the games before it, and the residual is what the scoreboard said beyond
   ``power_home - power_away + home_field``.
2. Regress those residuals on the situational features (ordinary least squares,
   robust standard errors).
3. What is left is grouped by stadium and shrunk toward zero by empirical Bayes.
   A stadium with 80 home games has a standard error near two points on its own
   mean, while the real spread between stadiums is likely nearer one, so most
   stadiums end up close to average. That is the expected result, not a failure.

The 2003-2019 workbooks carry no venues, so this runs on the CFBD games feed
(2013 onward), where every game names its stadium.

Every situational feature is a *difference between the two teams*: the
visitor's miles minus the host's, the visitor's climb minus the host's. At a
true home game the host's share is zero and it reduces to the plain visitor
number. At a neutral site it is what makes Florida-Georgia in Jacksonville, or
a bowl two hours from one campus, a little less neutral - and it means the
arbitrary "home" label on a neutral game changes nothing.
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..ingest import cfbd, registry
from . import mri2, priors

EARTH_RADIUS_MILES = 3958.8
METERS_TO_FEET = 3.28084
EARLY_KICKOFF_HOUR = 11      # before 11:00 on the visitor's body clock
LATE_KICKOFF_HOUR = 22       # 10pm or later on it
COVID_SEASON = 2020
TREND_ORIGIN = 2019

# The handful of stadiums the API is missing something for, filled from public
# records. Two of the API's coordinates are simply wrong (Orlando's points at
# Naples, Wrigley's at the Loop), so these replace rather than fill.
VENUE_PATCHES = {
    538: dict(latitude=33.8644, longitude=-118.2611, elevation_ft=40, timezone="America/Los_Angeles"),
    3494: dict(elevation_ft=460, timezone="America/Chicago"),
    4205: dict(elevation_ft=20, timezone="America/Los_Angeles"),
    4250: dict(latitude=41.9484, longitude=-87.6553, elevation_ft=595, timezone="America/Chicago"),
    4737: dict(latitude=53.3607, longitude=-6.2511, timezone="Europe/Dublin"),
    4779: dict(timezone="America/Nassau"),
    5403: dict(latitude=28.5411, longitude=-81.3893, elevation_ft=95, timezone="America/New_York"),
    5455: dict(latitude=33.1106, longitude=-96.8297, elevation_ft=690, timezone="America/Chicago"),
    7065: dict(elevation_ft=100, timezone="America/Los_Angeles"),
}

# The situational terms, in the order they are reported. Scales are chosen so a
# coefficient reads as points per natural unit: per 1,000 feet, per 1,000 miles,
# per time zone, per body-clock game.
FEATURES = ["hosted", "climb", "travel", "tz_east", "tz_west", "body_clock", "late_clock",
            "crowd", "trend", "covid_2020"]


# --------------------------------------------------------------------------- geometry

def great_circle_miles(lat1, lon1, lat2, lon2):
    """Haversine distance in statute miles. Vectorised."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def utc_offset_hours(timezone: str, when: dt.datetime) -> float:
    """Hours ahead of UTC in ``timezone`` at the instant ``when`` (aware, UTC).
    Daylight saving is the reason this takes a date: Arizona and Hawaii sit
    still while everyone else moves in November."""
    return when.astimezone(ZoneInfo(timezone)).utcoffset().total_seconds() / 3600.0


def local_hour(timezone: str, when: dt.datetime) -> float:
    local = when.astimezone(ZoneInfo(timezone))
    return local.hour + local.minute / 60.0


# --------------------------------------------------------------------------- venues

@lru_cache(maxsize=1)
def venue_table() -> pd.DataFrame:
    """Stadiums with elevation in feet and a time zone for every one we use.

    The API gives elevation in metres, and leaves elevation or time zone blank
    for about half its venues - almost all of them high-school fields and
    stadiums no FBS game has visited. The rest are patched by hand above, then
    anything still blank borrows from the nearest stadium that has it.
    """
    frame = cfbd.venues().set_index("venue_id")
    frame["elevation_ft"] = frame["elevation"] * METERS_TO_FEET
    for vid, patch in VENUE_PATCHES.items():
        if vid in frame.index:
            for key, value in patch.items():
                frame.loc[vid, key] = value

    known = frame.dropna(subset=["latitude", "longitude"])
    for column in ("elevation_ft", "timezone"):
        donors = known.dropna(subset=[column])
        for vid in known.index[known[column].isna()]:
            d = great_circle_miles(known.at[vid, "latitude"], known.at[vid, "longitude"],
                                   donors["latitude"], donors["longitude"])
            frame.loc[vid, column] = donors[column].iloc[int(np.argmin(d))]
    return frame


def _raw_games(seasons) -> pd.DataFrame:
    """Every completed game with an FBS side, straight from the cached feed, in
    the API's own team names (which is what the stadium lookups are keyed on)."""
    rows = []
    for year in seasons:
        for g in cfbd.request("/games", year=year, seasonType="both"):
            if "fbs" not in (g.get("homeClassification"), g.get("awayClassification")):
                continue
            if not g.get("completed") or g.get("homePoints") is None or g.get("awayPoints") is None:
                continue
            rows.append(
                {
                    "game_id": g["id"],
                    "season": g["season"],
                    "season_type": g["seasonType"],
                    "week": g["week"],
                    "start_date": g.get("startDate"),
                    "start_time_tbd": bool(g.get("startTimeTBD")),
                    "venue_id": g.get("venueId"),
                    "home": g["homeTeam"],
                    "away": g["awayTeam"],
                    "neutral": bool(g.get("neutralSite")),
                    "conference_game": bool(g.get("conferenceGame")),
                    "attendance": g.get("attendance"),
                    "home_margin": float(g["homePoints"]) - float(g["awayPoints"]),
                }
            )
    return pd.DataFrame(rows)


def home_venues(raw: pd.DataFrame) -> pd.DataFrame:
    """Where each team plays its home games, season by season.

    The venue that hosted the majority of the team's non-neutral home games that
    year - so a stadium move, a renovation year spent elsewhere, or 2020's
    relocations are all handled by construction. A team with no home game in the
    feed that season (an FCS visitor, usually) takes the stadium it used most
    across all seasons, then the one the API lists for it.
    """
    hosted = raw[~raw["neutral"] & raw["venue_id"].notna()]
    counts = hosted.groupby(["home", "season", "venue_id"]).size().rename("n").reset_index()
    per_season = (counts.sort_values(["home", "season", "n", "venue_id"], ascending=[True, True, False, True])
                  .drop_duplicates(["home", "season"]).rename(columns={"home": "team"}))
    per_season["source"] = "season"

    overall = (counts.groupby(["home", "venue_id"])["n"].sum().reset_index()
               .sort_values(["home", "n", "venue_id"], ascending=[True, False, True])
               .drop_duplicates("home").set_index("home")["venue_id"])
    listed = cfbd.team_homes().drop_duplicates("team").set_index("team")["venue_id"]

    have = set(zip(per_season["team"], per_season["season"]))
    extra = []
    for season in sorted(raw["season"].unique()):
        chunk = raw[raw["season"] == season]
        for team in sorted(set(chunk["home"]) | set(chunk["away"])):
            if (team, season) in have:
                continue
            if team in overall.index:
                extra.append((team, season, overall[team], "career"))
            elif team in listed.index:
                extra.append((team, season, listed[team], "listed"))
    extra = pd.DataFrame(extra, columns=["team", "season", "venue_id", "source"])
    out = pd.concat([per_season[["team", "season", "venue_id", "source"]], extra], ignore_index=True)
    out["venue_id"] = out["venue_id"].astype(int)
    return out


# --------------------------------------------------------------------------- features

def venue_games(seasons) -> pd.DataFrame:
    """One row per game: the stadium, both teams' homes, and every §2 feature.

    Everything is from the listed home team's perspective and, apart from the
    stadium itself, is the visitor's number minus the host's.
    """
    raw = _raw_games(seasons)
    homes = home_venues(raw).set_index(["team", "season"])["venue_id"]
    venues = venue_table()

    def home_of(team, season):
        return homes.get((team, season), np.nan)

    raw["home_venue_h"] = [home_of(t, s) for t, s in zip(raw["home"], raw["season"])]
    raw["home_venue_a"] = [home_of(t, s) for t, s in zip(raw["away"], raw["season"])]
    raw["venue_imputed"] = raw["venue_id"].isna()
    raw.loc[raw["venue_imputed"], "venue_id"] = raw.loc[raw["venue_imputed"], "home_venue_h"]
    raw = raw.dropna(subset=["venue_id"]).copy()
    raw["venue_id"] = raw["venue_id"].astype(int)

    def attr(ids, column):
        return venues[column].reindex(ids).to_numpy()

    v, h, a = raw["venue_id"].to_numpy(), raw["home_venue_h"].to_numpy(), raw["home_venue_a"].to_numpy()
    lat_v, lon_v = attr(v, "latitude").astype(float), attr(v, "longitude").astype(float)
    travel_h = great_circle_miles(attr(h, "latitude"), attr(h, "longitude"), lat_v, lon_v)
    travel_a = great_circle_miles(attr(a, "latitude"), attr(a, "longitude"), lat_v, lon_v)
    elev_v = attr(v, "elevation_ft").astype(float)
    climb_h = np.clip(elev_v - attr(h, "elevation_ft").astype(float), 0, None)
    climb_a = np.clip(elev_v - attr(a, "elevation_ft").astype(float), 0, None)

    tz_v, tz_h, tz_a = attr(v, "timezone"), attr(h, "timezone"), attr(a, "timezone")
    shift_h, shift_a, early_h, early_a, late_h, late_a = ([] for _ in range(6))
    for when, tbd, zv, zh, za in zip(raw["start_date"], raw["start_time_tbd"], tz_v, tz_h, tz_a):
        kickoff = pd.Timestamp(when).to_pydatetime()
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=dt.timezone.utc)
        off_v = utc_offset_hours(zv, kickoff)
        shift_h.append(off_v - utc_offset_hours(zh, kickoff) if isinstance(zh, str) else 0.0)
        shift_a.append(off_v - utc_offset_hours(za, kickoff) if isinstance(za, str) else 0.0)
        # A kickoff the feed has not pinned down carries a placeholder time, and a
        # placeholder says nothing about anyone's body clock.
        hour_h = local_hour(zh, kickoff) if isinstance(zh, str) and not tbd else np.nan
        hour_a = local_hour(za, kickoff) if isinstance(za, str) and not tbd else np.nan
        early_h.append(float(hour_h < EARLY_KICKOFF_HOUR))
        early_a.append(float(hour_a < EARLY_KICKOFF_HOUR))
        late_h.append(float(hour_h >= LATE_KICKOFF_HOUR))
        late_a.append(float(hour_a >= LATE_KICKOFF_HOUR))
    shift_h, shift_a = np.array(shift_h), np.array(shift_a)

    hosted = ~raw["neutral"].to_numpy()
    capacity = attr(v, "capacity").astype(float)
    fill = raw["attendance"].astype(float).to_numpy() / np.where(capacity > 0, capacity, np.nan)
    raw["fill"] = np.where(hosted & (fill > 0), np.clip(fill, 0, 1.3), np.nan)

    out = raw.assign(
        travel_home=travel_h, travel_away=travel_a,
        elev_venue=elev_v, climb_home=climb_h, climb_away=climb_a,
        tz_shift_home=shift_h, tz_shift_away=shift_a,
        hosted=hosted.astype(float),
        climb=(climb_a - climb_h) / 1000.0,
        travel=(travel_a - travel_h) / 1000.0,
        # Hours the visitor's clock jumped forward (flying east) or back (west),
        # net of the host's own trip.
        tz_east=np.clip(shift_a, 0, None) - np.clip(shift_h, 0, None),
        tz_west=np.clip(-shift_a, 0, None) - np.clip(-shift_h, 0, None),
        body_clock=np.array(early_a) - np.array(early_h),
        late_clock=np.array(late_a) - np.array(late_h),
        trend=hosted * (raw["season"].to_numpy() - TREND_ORIGIN) / 10.0,
        covid_2020=(hosted & (raw["season"].to_numpy() == COVID_SEASON)).astype(float),
    )
    out["crowd"] = _lagged_crowd(out)
    # A neutral game with no venue on file cannot borrow the listed host's
    # stadium the way a home game can - that would put Florida-Georgia in
    # Gainesville. Its trip is unknown, so it contributes nothing.
    unknown_trip = (out["venue_imputed"] & (out["hosted"] == 0)).to_numpy()
    for column in ("climb", "travel", "tz_east", "tz_west", "body_clock", "late_clock"):
        out[column] = np.where(unknown_trip, 0.0, out[column].fillna(0.0))
    return out.reset_index(drop=True)


def _lagged_crowd(frame: pd.DataFrame) -> np.ndarray:
    """How full this stadium usually is, from seasons before this one.

    Not the game's own attendance: that is only known once the gates close, and
    it follows the result as much as it causes it (nobody drives to watch a
    2-8 team). The stadium's fill in earlier seasons is what a Friday forecast
    can know. 2020 is left out of the history - empty stands were policy, not
    demand. Centred, and zero at neutral sites.
    """
    hist = frame[(frame["hosted"] == 1) & frame["fill"].notna() & (frame["season"] != COVID_SEASON)]
    by = hist.groupby(["venue_id", "season"])["fill"].agg(["sum", "count"]).reset_index()
    values = np.full(len(frame), np.nan)
    for i, (vid, season, hosted) in enumerate(zip(frame["venue_id"], frame["season"], frame["hosted"])):
        if not hosted:
            continue
        past = by[(by["venue_id"] == vid) & (by["season"] < season)]
        if past["count"].sum() >= 3:
            values[i] = past["sum"].sum() / past["count"].sum()
    centre = np.nanmean(values) if np.isfinite(values).any() else 0.0
    return np.where(frame["hosted"].to_numpy() == 1, np.nan_to_num(values - centre, nan=0.0), 0.0)


# --------------------------------------------------------------------------- stage 1

def walk_forward(seasons, warmup: int | None = None) -> pd.DataFrame:
    """Pregame predictions for every game, from ratings that had not seen it.

    Same fit and prior as the weekly build (``priors.for_season`` chained from
    the season before), refitted before each week on the games already played.
    Week 1 is priced from the prior alone.
    """
    seasons = sorted(seasons)
    warmup = warmup or seasons[0] - 1

    def canonical(frame):
        frame = frame.copy()
        for column in ("team1", "team2"):
            frame[column] = [registry.resolve(n, n) for n in frame[column]]
        return frame

    first = canonical(cfbd.games(warmup))
    fbs = [t for t in sorted(set(first["team1"]) | set(first["team2"])) if registry.is_fbs(t)]
    previous = mri2.fit(first, neutral=first["neutral"], anchor_teams=fbs,
                        with_resume=False, with_efficiency=False).power

    rows = []
    for season in seasons:
        games = canonical(cfbd.games(season)).reset_index(drop=True)
        games["seq"] = cfbd.sequence(games)
        teams = sorted(set(games["team1"]) | set(games["team2"]))
        fbs = [t for t in teams if registry.is_fbs(t)]
        prior = priors.for_season(season, previous, teams, fbs)

        for block in sorted(games["seq"].unique()):
            train = games[games["seq"] < block]
            target = games[games["seq"] == block]
            if train.empty:
                power, home_field, sigma = prior, mri2.DEFAULT_HOME_FIELD_PRIOR, mri2.DEFAULT_MARGIN_SIGMA
            else:
                model = mri2.fit(train, prior=prior, neutral=train["neutral"], anchor_teams=fbs,
                                 with_resume=False, with_efficiency=False)
                power = model.power.reindex(teams).fillna(prior)
                home_field, sigma = model.home_field, model.sigma
            for g in target.itertuples():
                rows.append((g.game_id, season, int(block), g.team2, g.team1, bool(g.neutral),
                             g.class2 == cfbd.FBS and g.class1 == cfbd.FBS,
                             float(power[g.team2] - power[g.team1]), home_field, sigma,
                             float(g.pts2 - g.pts1)))

        previous = mri2.fit(games, prior=prior, neutral=games["neutral"], anchor_teams=fbs,
                            with_resume=False, with_efficiency=False).power
    frame = pd.DataFrame(rows, columns=["game_id", "season", "seq", "home_team", "away_team", "neutral",
                                        "fbs_both", "diff", "hfa_base", "sigma", "margin"])
    frame["base_line"] = frame["diff"] + np.where(frame["neutral"], 0.0, frame["hfa_base"])
    frame["residual"] = frame["margin"] - frame["base_line"]
    return frame


# --------------------------------------------------------------------------- stages 2 and 3

def fit_situational(frame: pd.DataFrame, features=FEATURES) -> dict:
    """OLS of the residual on the situational terms, with robust (HC1) errors.

    ``hosted`` is the intercept for true home games; there is no global one,
    because at a neutral site the "home" label is arbitrary and every other term
    flips sign with it. A feature that never varies in the data (2020, before
    2020 happened) is left out and given a coefficient of zero.
    """
    X = frame[list(features)].astype(float)
    live = [c for c in X.columns if X[c].abs().sum() > 0 and X[c].nunique() > 1 or c == "hosted"]
    A, y = X[live].to_numpy(), frame["residual"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    # White's sandwich with the HC1 small-sample factor.
    n, p = A.shape
    bread = np.linalg.inv(A.T @ A)
    meat = (A * ((y - A @ beta) ** 2)[:, None]).T @ A
    cov = bread @ meat @ bread * n / (n - p)
    params = dict(zip(live, beta))
    errors = dict(zip(live, np.sqrt(np.diag(cov))))
    coef = {c: float(params.get(c, 0.0)) for c in features}
    se = {c: float(errors.get(c, np.nan)) for c in features}
    return {"coef": coef, "se": se, "n": int(n), "live": live}


def situational(frame: pd.DataFrame, fit: dict) -> np.ndarray:
    return sum(frame[c].astype(float).to_numpy() * v for c, v in fit["coef"].items())


def shrink(values: pd.Series, groups: pd.Series, *, min_games_for_tau: int = 5) -> pd.DataFrame:
    """Empirical-Bayes means per group: ``n / (n + k) * mean``, ``k = sigma^2 / tau^2``.

    ``tau^2`` - the real spread between groups - is the variance of the group
    means less the average sampling variance, estimated on groups with enough
    games for their mean to be worth anything. If that comes out at or below
    zero the honest answer is that no difference is detectable, and every
    effect is zero.
    """
    frame = pd.DataFrame({"v": values.to_numpy(float), "g": groups.to_numpy()})
    stats = frame.groupby("g")["v"].agg(["mean", "count"])
    sigma2 = float(frame["v"].var(ddof=1))
    big = stats[stats["count"] >= min_games_for_tau]
    tau2 = float(big["mean"].var(ddof=1) - (sigma2 / big["count"]).mean()) if len(big) > 2 else 0.0

    if tau2 <= 0:
        out = stats.assign(effect=0.0, se=0.0, weight=0.0)
    else:
        k = sigma2 / tau2
        weight = stats["count"] / (stats["count"] + k)
        # Posterior sd under the normal-normal model: sqrt(1 / (n/sigma^2 + 1/tau^2)).
        se = np.sqrt(1.0 / (stats["count"] / sigma2 + 1.0 / tau2))
        out = stats.assign(effect=weight * stats["mean"], se=se, weight=weight)
    out.attrs.update(tau2=tau2, sigma2=sigma2, k=(sigma2 / tau2 if tau2 > 0 else np.inf))
    return out.rename(columns={"count": "n", "mean": "raw"})


def fit_stadiums(frame: pd.DataFrame, sit_fit: dict | None) -> dict:
    """Stage 3 on what the situational terms leave behind.

    ``venue``: hosted games grouped by stadium - the plan's estimate.
    ``road``: each team's games away from home (road and neutral), grouped by
    its home stadium. A team with a big home edge has part of it baked into its
    rating, which then runs hot on the road; this is the other half of the same
    effect, and adding it back is what variant (b2) tests.
    """
    rest = frame["residual"] - (situational(frame, sit_fit) if sit_fit else 0.0)
    hosted = frame["hosted"] == 1
    venue = shrink(rest[hosted], frame.loc[hosted, "venue_id"])

    # Road residuals from the travelling team's own perspective.
    away_side = pd.DataFrame({"v": -rest, "g": frame["home_venue_a"]})
    neutral_home = pd.DataFrame({"v": rest[~hosted], "g": frame.loc[~hosted, "home_venue_h"]})
    road_rows = pd.concat([away_side, neutral_home], ignore_index=True).dropna()
    road = shrink(road_rows["v"], road_rows["g"])
    return {"venue": venue, "road": road}


def stadium_adjustment(frame: pd.DataFrame, fits: dict, *, road: bool) -> np.ndarray:
    venue_eff = fits["venue"]["effect"]
    adj = np.where(frame["hosted"] == 1, frame["venue_id"].map(venue_eff).fillna(0.0), 0.0)
    if road:
        road_eff = fits["road"]["effect"]
        a = frame["home_venue_a"].map(road_eff).fillna(0.0).to_numpy()
        h = frame["home_venue_h"].map(road_eff).fillna(0.0).to_numpy()
        adj = adj - a + np.where(frame["hosted"] == 1, 0.0, h)
    return adj


def home_edge_contrast(frame: pd.DataFrame, sit_fit: dict | None) -> pd.DataFrame:
    """The number for the page: how much better each team plays at its own
    stadium than on the road, beyond what the average home field and the trip
    already explain. Home residual minus road residual, per tenant stadium,
    shrunk the same way - which makes the rating's own absorption of half the
    effect cancel out."""
    rest = frame["residual"] - (situational(frame, sit_fit) if sit_fit else 0.0)
    hosted = (frame["hosted"] == 1) & (frame["venue_id"] == frame["home_venue_h"])
    home = pd.DataFrame({"g": frame.loc[hosted, "venue_id"], "v": rest[hosted]})
    road = pd.concat([
        pd.DataFrame({"g": frame["home_venue_a"], "v": -rest}),
        pd.DataFrame({"g": frame.loc[frame["hosted"] == 0, "home_venue_h"], "v": rest[frame["hosted"] == 0]}),
    ]).dropna()
    h = home.groupby("g")["v"].agg(["mean", "count"])
    r = road.groupby("g")["v"].agg(["mean", "count"])
    both = h.join(r, lsuffix="_home", rsuffix="_road", how="inner")
    sigma2 = float(pd.concat([home["v"], road["v"]]).var(ddof=1))
    both["raw"] = both["mean_home"] - both["mean_road"]
    both["samp"] = sigma2 * (1 / both["count_home"] + 1 / both["count_road"])
    big = both[both["count_home"] >= 10]
    tau2 = float(big["raw"].var(ddof=1) - big["samp"].mean())
    if tau2 <= 0:
        both["effect"], both["se"], both["weight"] = 0.0, 0.0, 0.0
    else:
        both["weight"] = tau2 / (tau2 + both["samp"])
        both["effect"] = both["weight"] * both["raw"]
        both["se"] = np.sqrt(1 / (1 / both["samp"] + 1 / tau2))
    both.attrs.update(tau2=tau2, sigma2=sigma2)
    return both

