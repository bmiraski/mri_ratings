"""The basketball preseason rankings page: its data, its page, and the snapshot that outlives the preseason."""

from __future__ import annotations

import json

import pandas as pd

from mri.export import bb_preseason, site
from mri.ratings import bb_priors


def team(name, rank, conference="Big East", **extra):
    return {"team": name, "rank": rank, "power": 10.0, "conference": conference, "abbreviation": name[:3].upper(),
            "color": "#123456", "logo": None, **extra}


def payload(preseason=None):
    return {"sport": "basketball", "season": 2026, "seasonLabel": "2025-26", "periodLabel": "Final", "finished": True,
            "week": 23, "weeks": [23], "teams": [team("Hosts", 1), team("Visitors", 2, "MAAC")], "conferences": [],
            "sports": ["football", "basketball"], "generated": "2026-09-20T00:00:00Z", "gamesRated": 100,
            "homeField": 3.0, "preseason": preseason}


def table(season=2027):
    rows = [
        {"team": "Visitors", "rank": 1, "power": 12.5, "conference": "MAAC", "color": "#123456", "logo": None, "lastRank": 2,
         "lastPower": 9.0, "hasPage": True, "mode": "roster", "returning": 0.41, "returningRank": 30, "incoming": 7.5,
         "freshman": 0.6, "freshmanRank": 12},
        {"team": "Hosts", "rank": 2, "power": 8.0, "conference": "Big East", "color": "#123456", "logo": None, "lastRank": 1,
         "lastPower": 11.0, "hasPage": True, "mode": "before rosters", "returning": None, "returningRank": None,
         "incoming": None, "freshman": None, "freshmanRank": None},
        {"team": "Newcomer", "rank": 3, "power": -4.0, "conference": "Big East", "color": "#123456", "logo": None,
         "lastRank": None, "lastPower": None, "hasPage": False, "mode": None, "returning": None, "returningRank": None,
         "incoming": None, "freshman": None, "freshmanRank": None},
    ]
    return {"season": season, "seasonLabel": "2026-27", "lastSeasonLabel": "2025-26", "mode": "roster",
            "teamsWithRosters": 1, "teams": rows}


def test_the_page_ranks_by_the_prior_and_says_who_moved_and_why() -> None:
    html = bb_preseason.preseason_page(payload(table()))
    assert "2026–27 preseason rankings" in html or "2026-27 preseason rankings" in html
    first, second = html.index("Visitors"), html.index("Hosts")
    assert first < second                                              # ranked by the prior, not last season's order
    assert "41%" in html and "7.5" in html                              # returning share and incoming win shares
    assert 'class="muted">New</span>' in html                           # a team with no last season
    assert 'data-conf="MAAC"' in html and '<option value="MAAC">' in html


def test_a_team_without_a_page_is_not_linked() -> None:
    html = bb_preseason.preseason_page(payload(table()))
    assert 'href="team/newcomer.html"' not in html and 'href="team/visitors.html"' in html


def test_the_nav_the_rankings_page_and_the_method_page_point_to_it() -> None:
    p = payload(table())
    p["teams"] = [team("Hosts", 1, wins=0, losses=0, resume=0.0, resumeRank=1, movement=0, trajectory=[0, 0],
                       previousRank=None)]
    assert 'href="preseason.html">Preseason' in site.page("t", "", p)
    assert 'href="preseason.html">See the 2026-27 preseason rankings' in site.rankings_page(p)
    assert 'href="preseason.html">preseason rankings' in site._bb_prior_section(
        {**p, "priorModel": bb_priors.load_model(), "priorState": {"season": 2027, "mode": "roster", "teamsWithRosters": 1, "teams": 1}})
    assert "preseason.html" not in site.page("t", "", payload(None))


def test_once_the_season_is_under_way_the_note_goes_but_the_saved_page_stays(tmp_path) -> None:
    saved = table(2027)
    (tmp_path / "preseason.json").write_text(json.dumps(saved))
    assert bb_preseason.saved(tmp_path, 2027) == saved
    assert bb_preseason.saved(tmp_path, 2026) is None                   # not this season's snapshot
    assert bb_preseason.saved(tmp_path / "missing", 2027) is None
    live = {**payload(saved), "season": 2027, "finished": False}
    live["teams"] = [team("Hosts", 1, wins=0, losses=0, resume=0.0, resumeRank=1, movement=0, trajectory=[0, 0], previousRank=None)]
    assert "starts soon" not in site.rankings_page(live)               # the banner is for the weeks before the season


def test_build_ranks_every_team_by_the_prior(monkeypatch) -> None:
    from mri.betting import bb_board
    from mri.export import bb_sitedata
    from mri.ingest import bb_registry

    class Team:
        def __init__(self, conference, color=None):
            self.conference, self.color = conference, color

    monkeypatch.setattr(bb_board, "preseason_power", lambda season: pd.Series({"Hosts": 3.0, "Visitors": 9.0, "Outsider": 99.0}))
    monkeypatch.setattr(bb_registry, "teams", lambda season=None: {"Hosts": Team("Big East"), "Visitors": Team("MAAC")})
    monkeypatch.setattr(bb_sitedata, "roster_context", lambda season, names: {
        "Visitors": {"mode": "roster", "returning": 0.5, "returningRank": 1, "incoming": 2.0}})
    data = bb_preseason.build(2027, payload())
    assert [t["team"] for t in data["teams"]] == ["Visitors", "Hosts"]          # Outsider is not Division I
    assert data["teams"][0]["lastRank"] == 2 and data["teams"][1]["lastRank"] == 1
    assert data["teamsWithRosters"] == 1 and data["mode"] == "before rosters"   # 1 of 2 is under 60%


def test_a_build_writes_the_page_and_its_data_but_keeps_the_table_out_of_the_published_json(tmp_path) -> None:
    import pytest
    from pathlib import Path

    data = Path(__file__).resolve().parents[1] / "site" / "data" / "bb.json"
    if not data.exists():
        pytest.skip("basketball site data not built")
    p = json.loads(data.read_text())
    p["details"], p["sports"] = {}, ["football", "basketball"]
    p["preseason"] = table(p["season"] + 1)
    p["preseason"]["teams"][0]["hasPage"] = False                      # no real team pages are needed to render
    site.build(p, tmp_path, publish_details=False)
    out = tmp_path / "basketball"
    assert "preseason rankings" in (out / "preseason.html").read_text()
    assert json.loads((out / "preseason.json").read_text())["season"] == p["season"] + 1
    assert "preseason" not in json.loads((out / "basketball.json").read_text())
    assert 'href="preseason.html">Preseason' in (out / "index.html").read_text()
