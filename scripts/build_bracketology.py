"""Build the live bracketology projection: site/data/bracketology.json.

Runs the Phase 3 joint simulation for the current men's basketball season and
writes everything the bracketology pages (Phase 4) will need - every team's odds of
making the field and of each seed line, each conference's automatic-bid odds, and
one projected bracket placed into regions by the committee's own principles.

Calendar gates (from data/bracketology_settings.json):
* nothing before Christmas (``startsOn``) - too few games for a field to mean much;
* frozen from Selection Sunday on: once the real bracket is out, the last
  projection is kept as it was, to set beside what actually happened.

Run:  PYTHONPATH=src python3 scripts/build_bracketology.py
      [--season 2025 --as-of 2025-02-01 --out /tmp/x.json --force]   (replay any past date)
      [--field-size 76]                                                (...under the 2027 format)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.bracket import history, joint, regions, seeding, template as template_mod  # noqa: E402
from mri.ingest import bb_bracket, bb_registry, cbbd  # noqa: E402

SETTINGS = ROOT / "data" / "bracketology_settings.json"
OUT = ROOT / "site" / "data" / "bracketology.json"
LOOKBACK = 3


def season_for(today: dt.date) -> int:
    """The men's basketball season in progress, named for the year it ends in (2026-27 is 2027)."""
    return today.year + 1 if today.month >= 7 else today.year


def selection_sunday(season: int, settings: dict) -> dt.date:
    """From the settings if listed; otherwise the Sunday on or after March 11, which has been right
    every year this system covers."""
    listed = settings.get("selectionSunday", {}).get(str(season))
    if listed:
        return dt.date.fromisoformat(listed)
    march11 = dt.date(season, 3, 11)
    return march11 + dt.timedelta(days=(6 - march11.weekday()) % 7)


def start_date(season: int, settings: dict) -> dt.date:
    month, day = (int(x) for x in settings.get("startsOn", "12-25").split("-"))
    return dt.date(season - 1, month, day)


def templates_for(season: int, conferences: set[str], overrides: dict) -> tuple[dict, dict]:
    """Each conference's tournament shape, and where it came from: this season's entry in
    ``formatOverride`` if there is one ("announced" or "provisional"), otherwise inferred from the
    conference's last three tournaments ("inferred")."""
    frames = [bb_bracket.conference_tournaments(y) for y in range(season - LOOKBACK, season) if y != 2020]
    this_season = overrides.get(str(season), {})
    out, source = {}, {}
    for conf in conferences:
        if conf in this_season:
            o = this_season[conf]
            out[conf] = template_mod.Template(tiers=tuple(o["tiers"]), seasons_seen=0, stable=True,
                                              campus_hosted=bool(o.get("campusHosted", False)))
            source[conf] = "announced" if o.get("announced") else "provisional"
            continue
        out[conf] = template_mod.infer([f[f["conference"] == conf] for f in frames if not f.empty], min_seasons=1)
        source[conf] = "inferred"
    return out, source


def format_warnings(templates: dict, source: dict, members: dict[str, int]) -> list[str]:
    """Inferred formats that can't be right any more: a bracket bigger than the league now is. (One
    smaller than the league is fine - plenty of conferences invite only their top K.)"""
    out = []
    for conf, tpl in sorted(templates.items()):
        if source.get(conf) == "inferred" and tpl is not None and tpl.size > members.get(conf, 0):
            out.append(f"{conf}: last tournaments had {tpl.size} teams, league now has {members.get(conf, 0)} - "
                       f"add a formatOverride for this season")
        if tpl is None:
            out.append(f"{conf}: no recent tournament to read a format from - add a formatOverride")
    return out


def _round(x, digits=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), digits)


def build(season: int, as_of: str | None, settings: dict, sims: int, field_size: int | None = None) -> dict:
    live = as_of is None
    games = cbbd.games(season, completed_only=False) if live else cbbd.games(season)
    games = history._canonical(games, season)
    if "played" not in games.columns:
        games["played"] = True

    previous = None
    if live:
        from mri.export import bb_sitedata

        previous = bb_sitedata._prior_for(season)
    played = games[games["played"].astype(bool)]
    ratings = history.team_ratings(season, through=as_of, games=played, with_resume=True, previous=previous)
    conf_of = {t: bb_registry.conference_of(t, season=season) for t in ratings.power.index}
    conf_of = {t: c for t, c in conf_of.items() if c}
    conferences = set(conf_of.values())
    templates, source = templates_for(season, conferences, settings.get("formatOverride", {}))
    members = pd.Series(list(conf_of.values())).value_counts().to_dict()
    warnings = format_warnings(templates, source, members)
    for w in warnings:
        print(f"  format warning - {w}")
    model = json.loads((ROOT / "data" / "atlarge_model.json").read_text())
    beta = np.array(model["coefficients"])
    fmt = seeding.FORMATS[field_size or seeding.field_size(season)]
    modes = {c: settings.get("autoBid", {}).get(c, joint.DEFAULT_MODE) for c in conferences}

    res = joint.run(joint.Inputs(season=season, games=games, power=ratings.power, home_field=ratings.home_field,
                                 resume_sigma=ratings.sigma, conference_of=conf_of, templates=templates, beta=beta,
                                 fmt=fmt, auto_mode=modes, as_of=as_of, sims=sims, seed=0,
                                 committee_noise=model.get("committeeNoise", 0.0)))
    projected, bubble = joint.projected_field(res, fmt)
    meetings = regions.meetings_from_games(games[games["season_type"] == "regular"])
    placement = regions.place(projected, meetings)
    bracket = placement.table(projected)

    teams = res.teams[res.teams["pField"] >= 0.005]
    return {
        "season": season, "asOf": as_of or dt.date.today().isoformat(), "sims": sims,
        "committeeNoise": model.get("committeeNoise", 0.0),
        "formatWarnings": warnings,
        "fieldSize": fmt.size, "automaticBids": sum(1 for c in res.champions if res.champions[c]),
        "conferences": {
            c: {"mode": modes[c], "status": res.conference_status.get(c),
                "tournamentFormat": list(templates[c].tiers) if templates.get(c) else None,
                "formatSource": source.get(c),
                "odds": [{"team": t, "p": _round(p)} for t, p in sorted(res.champions.get(c, {}).items(),
                                                                          key=lambda kv: -kv[1])[:6]]}
            for c in sorted(conferences)
        },
        "teams": [{
            "team": r.team, "conference": r.conference, "power": _round(r.power, 2), "powerRank": int(r.powerRank),
            "record": f"{int(r.winsNow)}-{int(r.lossesNow)}",
            "projectedRecord": f"{r.projectedWins:.1f}-{r.projectedLosses:.1f}",
            "pField": _round(r.pField), "pAuto": _round(r.pAuto), "pAtLarge": _round(r.pAtLarge),
            "pOpening": _round(r.pOpening), "expectedSeed": _round(r.expectedSeed, 2),
            "seedOdds": [_round(getattr(r, f"seed{k}")) for k in range(1, 17)],
        } for r in teams.itertuples()],
        "projected": {
            "field": [{
                "team": r.team, "conference": r.conference, "bidType": r.bidType, "trueSeed": int(r.trueSeed),
                "seedLine": int(r.seedLine), "openingRound": bool(r.opening), "region": int(r.region),
                "openingGame": r.openingGame if isinstance(r.openingGame, str) else None,
                "movedFrom": None if pd.isna(r.movedFrom) else int(r.movedFrom),
            } for r in bracket.itertuples()],
            "regionOrder": list(regions.BRACKET_ORDER),
            "semifinals": [list(p) for p in regions.SEMIFINALS],
            "regionTotals": placement.region_totals(dict(zip(projected["team"], projected["trueSeed"]))),
            "ruleProblems": regions.check(projected, placement, meetings),
            "bubble": bubble,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int)
    parser.add_argument("--as-of", help="replay a past date (YYYY-MM-DD); omit for live")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--sims", type=int)
    parser.add_argument("--force", action="store_true", help="ignore the Christmas / Selection Sunday gates")
    parser.add_argument("--field-size", type=int, choices=sorted(seeding.FORMATS),
                        help="replay a season under another format (e.g. 2025 as a 76-team field)")
    args = parser.parse_args()

    settings = json.loads(SETTINGS.read_text())
    today = dt.date.fromisoformat(args.as_of) if args.as_of else dt.date.today()
    season = args.season or season_for(today)
    if not args.force:
        if today < start_date(season, settings):
            print(f"bracketology: not before {start_date(season, settings)}; nothing written")
            return
        if today > selection_sunday(season, settings) and args.out.exists():
            print("bracketology: frozen after Selection Sunday; keeping the last projection")
            return
    data = build(season, args.as_of, settings, args.sims or settings.get("sims", 5000), args.field_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=1) + "\n")
    in_field = [t for t in data["teams"] if t["pField"] >= 0.5]
    print(f"bracketology {season} as of {data['asOf']}: {len(data['teams'])} teams with a chance, "
          f"{len(in_field)} at 50%+; rule problems: {len(data['projected']['ruleProblems'])}")


if __name__ == "__main__":
    main()
