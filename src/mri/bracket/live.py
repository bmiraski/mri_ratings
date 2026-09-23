"""The live bracketology projection: the Phase 3 joint simulation for a season, as data a page can show.

Shared by the site build (``mri.export.bracketdata``, daily) and ``scripts/build_bracketology.py``
(by hand, and for replaying any past date). Everything about *which* projection to make - the
season, the tournament formats, the automatic-bid mode, the calendar - is read from
``data/bracketology_settings.json``.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..ingest import bb_bracket, bb_registry, cbbd
from . import history, joint, regions, seeding, template as template_mod

ROOT = Path(__file__).resolve().parents[3]
SETTINGS = ROOT / "data" / "bracketology_settings.json"
LOOKBACK = 3


# How far ahead the slate's bid stakes are computed: today's games and the next two days'.
LEVERAGE_DAYS = 3


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
        from ..export import bb_sitedata

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
                                 committee_noise=model.get("committeeNoise", 0.0),
                                 leverage_through=((dt.date.fromisoformat(as_of[:10]) if as_of else dt.date.today())
                                                   + dt.timedelta(days=LEVERAGE_DAYS)).isoformat()))
    projected, bubble = joint.projected_field(res, fmt)
    meetings = regions.meetings_from_games(games[games["season_type"] == "regular"])
    placement = regions.place(projected, meetings)
    bracket = placement.table(projected)

    teams = res.teams[res.teams["pField"] >= 0.005]
    return {
        "season": season, "asOf": as_of or dt.date.today().isoformat(), "sims": sims,
        "committeeNoise": model.get("committeeNoise", 0.0),
        "formatWarnings": warnings,
        "leverage": res.leverage,
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




def load_settings() -> dict:
    return json.loads(SETTINGS.read_text())
