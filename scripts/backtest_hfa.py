"""Does home field vary by stadium or by trip, enough to predict better?

Every FBS-vs-FBS game of 2016-2025 is predicted from walk-forward ratings (only
earlier weeks of its own season, plus the prior), and the home-field layer on
top is fitted only on seasons before the one being scored. Variants:

  baseline  the single home-field number the model uses today
  (a)       + situational terms: altitude, travel, time zones, body clock, crowd
  (b)       + per-stadium effects, empirical-Bayes shrunk
  (b2)      (b) plus the road half of each team's home edge
  (c)       (a) and (b) together
  (c2)      (a) and (b2) together

Pass rule, the bar MRI 2.0 cleared: better log loss in most held-out seasons and
a paired t > 2 across windows (each season cut into five chronological blocks).

Writes data/hfa_backtest.json (the full report), data/parquet/venue_game.parquet,
and site/data/homefield.json - the "Home field" page, which describes the past
and changes no line.

Run:  PYTHONPATH=src python3 scripts/backtest_hfa.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.betting import board  # noqa: E402
from mri.ingest import cfbd  # noqa: E402
from mri.ratings import hfa, mri2  # noqa: E402

SEASONS = range(2013, 2026)
SCORED = range(2016, 2026)
WINDOWS_PER_SEASON = 5
Z80 = 1.2816
VARIANTS = ["a", "b", "b2", "c", "c2"]
# The trip and the stands: every situational term except the intercept and the
# two that describe the season rather than the stadium.
CONDITIONS = ["climb", "travel", "tz_east", "tz_west", "body_clock", "late_clock", "crowd"]


def score(frame: pd.DataFrame, line: np.ndarray) -> pd.DataFrame:
    # The site's own game sigma, not the fit's: the fit reports its in-sample
    # spread (~12.8), while pregame errors run at ~16.3, and scoring with the
    # smaller number would punish any variant for spreading the lines at all.
    p = np.clip(norm.cdf(line / board.SIGMA), 1e-6, 1 - 1e-6)
    won = (frame["margin"].to_numpy() > 0).astype(float)
    return pd.DataFrame({
        "abs_err": np.abs(frame["margin"].to_numpy() - line),
        "log_loss": -(won * np.log(p) + (1 - won) * np.log(1 - p)),
        "brier": (p - won) ** 2,
    }, index=frame.index)


def lines_for(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    sit = hfa.fit_situational(train)
    sit_adj = hfa.situational(test, sit)
    plain = hfa.fit_stadiums(train, None)
    after = hfa.fit_stadiums(train, sit)
    base = test["base_line"].to_numpy()
    return {
        "baseline": base,
        "a": base + sit_adj,
        "b": base + hfa.stadium_adjustment(test, plain, road=False),
        "b2": base + hfa.stadium_adjustment(test, plain, road=True),
        "c": base + sit_adj + hfa.stadium_adjustment(test, after, road=False),
        "c2": base + sit_adj + hfa.stadium_adjustment(test, after, road=True),
    }


def summarize(scored: pd.DataFrame, mask=None) -> dict:
    frame = scored if mask is None else scored[mask]
    out = {"games": int(len(frame) // (len(VARIANTS) + 1))}
    for name, chunk in frame.groupby("variant"):
        out[name] = {m: round(float(chunk[m].mean()), 5) for m in ("abs_err", "log_loss", "brier")}
    return out


def paired(scored: pd.DataFrame, variant: str) -> dict:
    """Per-window and per-season log-loss gains of ``variant`` over baseline."""
    wide = scored.pivot_table(index=["season", "window"], columns="variant", values="log_loss", aggfunc="mean")
    gain = wide["baseline"] - wide[variant]
    t = float(gain.mean() / (gain.std(ddof=1) / np.sqrt(len(gain))))
    by_season = scored.pivot_table(index="season", columns="variant", values="log_loss", aggfunc="mean")
    season_gain = by_season["baseline"] - by_season[variant]
    mae = scored.pivot_table(index=["season", "window"], columns="variant", values="abs_err", aggfunc="mean")
    mae_gain = mae["baseline"] - mae[variant]
    passes = bool((season_gain > 0).sum() > len(season_gain) / 2 and t > 2)
    return {
        "windows": int(len(gain)),
        "logLossGain": round(float(gain.mean()), 6),
        "t": round(t, 2),
        "seasonsBetter": int((season_gain > 0).sum()),
        "seasons": int(len(season_gain)),
        "maeGain": round(float(mae_gain.mean()), 4),
        "maeT": round(float(mae_gain.mean() / (mae_gain.std(ddof=1) / np.sqrt(len(mae_gain)))), 2),
        "bySeason": {int(s): round(float(v), 5) for s, v in season_gain.items()},
        "passes": passes,
    }


def season_home_field() -> dict[int, float]:
    """Each season's own home field, fitted on its FBS-vs-FBS games with the
    home-field prior all but switched off.

    This, not the ``covid_2020`` coefficient, is the crowd experiment: the
    residuals the coefficient sees are measured against each season's in-season
    home field, which already absorbed most of 2020's drop.
    """
    out = {}
    for year in SEASONS:
        g = cfbd.games(year)
        g = g[(g["class1"] == cfbd.FBS) & (g["class2"] == cfbd.FBS)].reset_index(drop=True)
        out[int(year)] = round(season_fit_home_field(g), 2)
    return out


def season_fit_home_field(g: pd.DataFrame) -> float:
    teams = sorted(set(g["team1"]) | set(g["team2"]))
    return float(mri2.fit(g, neutral=g["neutral"], anchor_teams=teams, home_field_ridge=1.0,
                          with_resume=False, with_efficiency=False).home_field)


def crowd_2020_error(reps: int = 200, seed: int = 2020) -> float:
    """Bootstrap standard error of 2020's own home field, games resampled."""
    g = cfbd.games(hfa.COVID_SEASON)
    g = g[(g["class1"] == cfbd.FBS) & (g["class2"] == cfbd.FBS)].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    draws = [season_fit_home_field(g.iloc[rng.integers(0, len(g), len(g))].reset_index(drop=True))
             for _ in range(reps)]
    return float(np.std(draws, ddof=1))


def page_data(data: pd.DataFrame, sit: dict, edge: pd.DataFrame, report: dict) -> dict:
    """What the Home field page shows: one row per stadium an FBS team calls home now.

    ``adjustment`` is what a typical visit there adds to the league's home field -
    the trip and the stands (fitted situational terms, averaged over the games the
    stadium actually hosted, relative to the average stadium) plus the stadium's
    own shrunk edge. The page adds the model's current home field on top.
    """
    from mri.ingest import registry

    venues = hfa.venue_table()
    everything = hfa.venue_games(range(min(SEASONS), max(SEASONS) + 2))
    hosted_all = everything[everything["hosted"] == 1]

    coef = sit["coef"]
    data = data.assign(conditions=sum(data[c] * coef[c] for c in CONDITIONS))
    hosted = data[(data["hosted"] == 1) & (data["venue_id"] == data["home_venue_h"])]
    centre = float(hosted["conditions"].mean())
    trip = hosted.groupby("venue_id")["conditions"].agg(["mean", "size"])

    rows = []
    for team in sorted(registry.teams()):
        mine = hosted_all[hosted_all["home"].map(lambda n: registry.resolve(n, n)) == team]
        if mine.empty:
            continue
        season = int(mine["season"].max())
        vid = int(mine[mine["season"] == season]["venue_id"].mode().iloc[0])
        if vid not in trip.index or vid not in edge.index or trip.at[vid, "size"] < 10:
            continue
        at_home = mine[mine["venue_id"] == vid]
        won = int((at_home["home_margin"] > 0).sum())
        e, v = edge.loc[vid], venues.loc[vid]
        lo, hi = float(e["effect"] - Z80 * e["se"]), float(e["effect"] + Z80 * e["se"])
        conditions = float(trip.at[vid, "mean"] - centre)
        rows.append({
            "team": team, "venue": v["name"], "city": v["city"], "state": v["state"],
            "adjustment": round(conditions + float(e["effect"]), 2),
            "conditions": round(conditions, 2),
            "stadium": round(float(e["effect"]), 2),
            "stadiumInterval": [round(lo, 2), round(hi, 2)],
            "excludesZero": bool(lo > 0 or hi < 0),
            "elevation": int(round(float(v["elevation_ft"]), -1)),
            "capacity": int(v["capacity"]) if v["capacity"] == v["capacity"] and v["capacity"] > 0 else None,
            "dome": bool(v["dome"]),
            "homeWins": won, "homeLosses": int(len(at_home) - won),
            "games": int(trip.at[vid, "size"]),
        })
    rows.sort(key=lambda r: -r["adjustment"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    def term(name, scale=1.0):
        c = report["situational"][name]
        return {"points": round(c["coef"] * scale, 2), "interval80": [round(x * scale, 2) for x in c["interval80"]]}

    return {
        "seasons": [min(SEASONS), max(SEASONS)],    # what the fit saw; home records run to today
        "games": report["games"],
        "stadiums": rows,
        "anyExcludesZero": any(r["excludesZero"] for r in rows),
        "stadiumSpread": round(float(np.sqrt(max(report["homeEdge"]["tau2"], 0.0))), 2),
        "travels": {
            "altitude": term("climb"),            # per 1,000 feet the visitor climbs
            "travel": term("travel"),             # per 1,000 extra miles
            "bodyClock": term("body_clock"),      # kickoff before 11am on the visitor's clock
            "crowd2020": {"points": report["crowd2020"],
                          "interval80": [round(report["crowd2020"] - Z80 * report["crowd2020Error"], 2),
                                         round(report["crowd2020"] + Z80 * report["crowd2020Error"], 2)]},
        },
        "trendPerDecade": report["trendPerDecade"],
        "gate": {v: {k: report["tests"][v][k] for k in ("t", "seasonsBetter", "seasons", "passes")}
                 for v in VARIANTS},
        "shipped": any(report["tests"][v]["passes"] for v in VARIANTS),
    }


def main() -> None:
    venue_game = hfa.venue_games(SEASONS)
    venue_game.drop(columns=["start_date"]).to_parquet(ROOT / "data" / "parquet" / "venue_game.parquet")
    pregame = hfa.walk_forward(SEASONS)

    keep = ["game_id", "venue_id", "home_venue_h", "home_venue_a", "venue_imputed", *hfa.FEATURES]
    data = pregame.merge(venue_game[keep], on="game_id", how="inner")
    fcs = data[~data["fbs_both"]]
    data = data[data["fbs_both"]].reset_index(drop=True)
    print(f"{len(data)} FBS-vs-FBS games with venues; {len(fcs)} FCS games set aside")

    rows = []
    for season in SCORED:
        train, test = data[data["season"] < season], data[data["season"] == season].copy()
        order = test["seq"].rank(method="first")
        test["window"] = np.ceil(order / len(test) * WINDOWS_PER_SEASON).astype(int)
        for name, line in lines_for(train, test).items():
            rows.append(score(test, line).assign(variant=name, season=season, window=test["window"],
                                                 hosted=test["hosted"].to_numpy()))
        print(f"  scored {season}")
    scored = pd.concat(rows, ignore_index=True)

    tests = {v: paired(scored, v) for v in VARIANTS}
    for v, r in tests.items():
        print(f"  ({v:2s}) log loss gain {r['logLossGain']:+.5f}  t={r['t']:+.2f}  "
              f"better in {r['seasonsBetter']}/{r['seasons']} seasons  MAE gain {r['maeGain']:+.3f} "
              f"(t={r['maeT']:+.2f})  {'PASS' if r['passes'] else 'fail'}")

    # Coefficients and stadium effects on everything, for the report and the page.
    sit = hfa.fit_situational(data)
    stadiums = hfa.fit_stadiums(data, sit)
    edge = hfa.home_edge_contrast(data, sit)
    venues = hfa.venue_table()

    def interval(c, s):
        return [round(c - Z80 * s, 3), round(c + Z80 * s, 3)] if np.isfinite(s) else None

    def stadium_rows(table):
        out = []
        for vid, r in table.sort_values("effect", ascending=False).iterrows():
            v = venues.loc[int(vid)] if int(vid) in venues.index else None
            out.append({
                "venueId": int(vid), "name": None if v is None else v["name"],
                "effect": round(float(r["effect"]), 3), "raw": round(float(r["raw"]), 3),
                "interval80": interval(float(r["effect"]), float(r["se"])),
                "games": int(r["n"] if "n" in r else r["count_home"]),
            })
        return out

    by_season = season_home_field()
    others = [v for y, v in by_season.items() if y != hfa.COVID_SEASON]
    years = [y for y in by_season if y != hfa.COVID_SEASON]

    payload = {
        "seasons": [min(SEASONS), max(SEASONS)],
        "seasonHomeField": by_season,
        "crowd2020": round(by_season[hfa.COVID_SEASON] - float(np.mean(others)), 2),
        "crowd2020Error": round(crowd_2020_error(), 2),
        "trendPerDecade": round(float(np.polyfit(years, others, 1)[0] * 10), 2),
        "scored": [min(SCORED), max(SCORED)],
        "games": int(len(data)),
        "fcsSetAside": {
            "games": int(len(fcs)),
            "meanResidual": round(float(fcs["residual"].mean()), 2),
            "note": "FBS-vs-FCS games are left out: the pregame line under-rates the FBS side by this much "
                    "on average, which would swamp every home-field term.",
        },
        "baseResidual": {
            "hosted": round(float(data.loc[data["hosted"] == 1, "residual"].mean()), 3),
            "neutral": round(float(data.loc[data["hosted"] == 0, "residual"].mean()), 3),
            "hfaBaseMean": round(float(data.loc[data["hosted"] == 1, "hfa_base"].mean()), 3),
        },
        "situational": {
            c: {"coef": round(sit["coef"][c], 3), "se": round(sit["se"][c], 3),
                "interval80": interval(sit["coef"][c], sit["se"][c])}
            for c in hfa.FEATURES
        },
        "stadiums": {
            "tau2": round(stadiums["venue"].attrs["tau2"], 3),
            "sigma2": round(stadiums["venue"].attrs["sigma2"], 2),
            "k": round(stadiums["venue"].attrs["k"], 1) if np.isfinite(stadiums["venue"].attrs["k"]) else None,
            "detectable": bool(stadiums["venue"].attrs["tau2"] > 0),
            "roadTau2": round(stadiums["road"].attrs["tau2"], 3),
            "venue": stadium_rows(stadiums["venue"]),
        },
        "homeEdge": {
            "tau2": round(edge.attrs["tau2"], 3),
            "detectable": bool(edge.attrs["tau2"] > 0),
            "venues": stadium_rows(edge.rename(columns={"count_home": "n"})),
        },
        "variants": {
            "overall": summarize(scored),
            "hosted": summarize(scored, scored["hosted"] == 1),
            "neutral": summarize(scored, scored["hosted"] == 0),
        },
        "tests": tests,
    }
    out = ROOT / "data" / "hfa_backtest.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")

    page = page_data(data, sit, edge, payload)
    out = ROOT / "site" / "data" / "homefield.json"
    out.write_text(json.dumps(page, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)}: {len(page['stadiums'])} stadiums, "
          f"{'some' if page['anyExcludesZero'] else 'none'} clear of zero")


if __name__ == "__main__":
    main()
