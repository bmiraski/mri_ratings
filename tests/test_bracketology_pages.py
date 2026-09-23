"""The bracketology pages and the data layer behind them: calendar gates, movement, the Selection
Sunday freeze, and pages that render, link and read correctly.

The projection itself is tested elsewhere (``test_bracket_joint``); here a synthetic one is built
from the real basketball site data's teams - a 76-team field seeded and placed by the real
``seeding`` and ``regions`` code - so the pages see exactly the shape the live build produces.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mri.bracket import regions, seeding
from mri.export import bracketdata, bracketpages, site

BB_DATA = Path(__file__).resolve().parents[1] / "site" / "data" / "bb.json"
SETTINGS = {"startsOn": "12-25", "sims": 100, "selectionSunday": {"2027": "2027-03-14"}, "autoBid": {}, "formatOverride": {}}


def _projection(teams: list[dict], field_size: int = 76) -> dict:
    """A projection in live.build's shape: the best 44 at-large plus 32 'champions' below them."""
    ranked = sorted(teams, key=lambda t: -t["power"])[:field_size + 20]
    field = ranked[:field_size]
    autos = 32 if field_size == 76 else 31
    frame = pd.DataFrame({
        "team": [t["team"] for t in field], "conference": [t["conference"] for t in field],
        "bidType": ["at-large"] * (field_size - autos) + ["auto"] * autos,
    })
    frame["trueSeed"] = np.arange(1, field_size + 1)
    frame["seedLine"], frame["opening"] = seeding.lines(frame["bidType"].eq("auto").to_numpy(), seeding.FORMATS[field_size])
    placement = regions.place(frame)
    table = placement.table(frame)
    out_of_field = [t["team"] for t in ranked[field_size:]]
    rows = []
    for i, t in enumerate(ranked):
        p = max(0.02, 1.0 - i / (field_size + 20))
        line = min(16, i // 4 + 1)
        odds = [0.0] * 16
        odds[line - 1] = p
        rows.append({"team": t["team"], "conference": t["conference"], "power": t["power"], "powerRank": i + 1,
                     "record": "20-5", "projectedRecord": "25.0-6.0", "pField": round(p, 4),
                     "pAuto": 0.3 if i >= field_size - autos else 0.0, "pAtLarge": round(p * 0.9, 4),
                     "pOpening": 0.0, "expectedSeed": float(line), "seedOdds": odds, "change": 0.02, "seedChange": 0.0})
    return {
        "season": 2027, "asOf": "2027-02-01", "sims": 100, "committeeNoise": 1.25, "formatWarnings": [],
        "fieldSize": field_size, "automaticBids": autos,
        "conferences": {"SEC": {"mode": "model", "status": "not started", "tournamentFormat": [4, 6, 6], "formatSource": "inferred",
                                "odds": [{"team": rows[0]["team"], "p": 0.4}, {"team": rows[1]["team"], "p": 0.3}]},
                        "WCC": {"mode": "model", "status": "decided", "tournamentFormat": [4, 2, 2, 2], "formatSource": "provisional",
                                "odds": [{"team": rows[5]["team"], "p": 1.0}]}},
        "teams": rows,
        "projected": {
            "field": [{"team": r.team, "conference": r.conference, "bidType": r.bidType, "trueSeed": int(r.trueSeed),
                       "seedLine": int(r.seedLine), "openingRound": bool(r.opening), "region": int(r.region),
                       "openingGame": r.openingGame if isinstance(r.openingGame, str) else None, "movedFrom": None}
                      for r in table.itertuples()],
            "regionOrder": list(regions.BRACKET_ORDER), "semifinals": [list(p) for p in regions.SEMIFINALS],
            "regionTotals": {str(k): v for k, v in placement.region_totals(dict(zip(frame.team, frame.trueSeed))).items()},
            "ruleProblems": [],
            "bubble": {"openingRoundAtLarge": list(table[(table.bidType == "at-large") & table.opening]["team"]),
                       "firstFourOut": out_of_field[:4], "nextFourOut": out_of_field[4:8]},
        },
        "comparedTo": "2027-01-25", "updated": "2027-02-01", "selectionSunday": "2027-03-14", "frozen": False,
        "startsOn": "2026-12-25",
    }


@pytest.fixture(scope="module")
def bb_payload() -> dict:
    if not BB_DATA.exists():
        pytest.skip("basketball site data not built")
    payload = json.loads(BB_DATA.read_text())
    payload["details"] = {}
    payload["sports"] = ["football", "basketball"]
    payload["bracketology"] = _projection(payload["teams"])
    payload["bracketologyBacktest"] = json.loads(
        (Path(__file__).resolve().parents[1] / "data" / "bracketology_backtest.json").read_text())
    return payload


@pytest.fixture(scope="module")
def built(bb_payload, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("bb")
    site.build(bb_payload, out, publish_details=False)
    return out / "basketball"


def test_pages_and_nav_link_exist(built):
    assert (built / "bracketology.html").exists() and (built / "bracket.html").exists()
    assert 'href="bracketology.html">Bracketology</a>' in (built / "index.html").read_text()


def test_no_nav_link_without_data(bb_payload, tmp_path):
    payload = {k: v for k, v in bb_payload.items() if k != "bracketology"}
    site.build(payload, tmp_path, publish_details=False)
    assert "Bracketology" not in (tmp_path / "basketball" / "index.html").read_text()
    assert not (tmp_path / "basketball" / "bracketology.html").exists()


def test_links_resolve(built):
    broken = []
    for name in ("bracketology.html", "bracket.html"):
        page = built / name
        for href in re.findall(r'(?:href|src)="([^"]+)"', page.read_text()):
            if href.startswith(("http://", "https://", "#", "data:")) or href == "../index.html":
                continue                       # the football site, via the sport switch, isn't built here
            if not (page.parent / urllib.parse.unquote(href)).resolve().exists():
                broken.append(f"{name} -> {href}")
    assert not broken, broken[:10]


def test_view_toggle_marks_only_the_current_view(built):
    for name in ("bracketology.html", "bracket.html"):
        toggle = re.search(r'<div class="views".*?</div>', (built / name).read_text(), re.S).group(0)
        assert toggle.count("class=on") == 1 and f'href="{name}" class=on' in toggle


def test_seed_list_has_every_team_once(bb_payload, built):
    text = (built / "bracketology.html").read_text()
    table = re.search(r'<table class="slate scurve">.*?</table>', text, re.S).group(0)
    for f in bb_payload["bracketology"]["projected"]["field"]:
        assert table.count(f'>{site.esc(f["team"])}</a>') == 1, f["team"]
    assert table.count('class="lineband"') == 16


def test_bracket_has_every_first_round_game(built):
    text = (built / "bracket.html").read_text()
    for r in range(1, 5):
        region = re.search(rf'<section class="region" id="region-{r}">.*?</section>', text, re.S).group(0)
        assert region.count('<div class="game">') == 8
    opening = re.search(r"<h2>Opening Round</h2>.*?</table>", text, re.S).group(0)
    assert opening.count('<td class="orgame">') == 12          # 76-team format: 12 Opening Round games


def test_team_page_line(bb_payload, built):
    top = bb_payload["bracketology"]["projected"]["field"][0]["team"]
    line = bracketpages.team_line(top, bb_payload)
    assert "projected No. 1 seed" in line and "../bracketology.html" in line
    outsider = bb_payload["bracketology"]["projected"]["bubble"]["firstFourOut"][0]
    assert "first four out" in bracketpages.team_line(outsider, bb_payload)


def test_frozen_page_shows_the_result(bb_payload, tmp_path):
    frozen = dict(bb_payload["bracketology"], frozen=True)
    field = frozen["projected"]["field"]
    actual = pd.DataFrame({"team": [f["team"] for f in field[:-2]] + ["Somebody Else", "Another"],
                           "seed": [f["seedLine"] for f in field[:-2]] + [16, 16],
                           "region": ["East"] * len(field), "bidType": [f["bidType"] for f in field]})
    frozen["result"] = bracketdata.compare(frozen, actual)
    assert frozen["result"]["named"] == len(field) - 2
    assert frozen["result"]["seedExact"] == len(field) - 2
    text = bracketpages.list_page({**bb_payload, "bracketology": frozen})
    assert "Frozen." in text and "How it did" in text and "Somebody Else" in text


def test_pct_never_rounds_up_to_certain():
    assert site._pct(0.995) == "&gt;99%" and site._pct(0.9949) == "99%"


# --------------------------------------------------------------------------- data layer

def test_nothing_before_christmas(tmp_path):
    assert bracketdata.build({}, tmp_path, today=dt.date(2026, 12, 24), season=2027, settings=SETTINGS) is None
    assert not list(tmp_path.iterdir())


def test_live_run_saves_and_measures_a_weeks_movement(tmp_path, monkeypatch, bb_payload):
    calls = []

    def fake_build(season, as_of, settings, sims, field_size=None):
        calls.append(as_of)
        data = _projection(bb_payload["teams"])
        bump = 0.1 * len(calls)
        for t in data["teams"]:
            t["pField"] = round(min(1.0, t["pField"] * 0.5 + bump), 4)
        for key in ("comparedTo", "updated", "frozen"):
            data.pop(key)
        return data

    monkeypatch.setattr(bracketdata.live, "build", fake_build)
    days = [dt.date(2027, 1, 1) + dt.timedelta(days=k) for k in range(9)]
    for day in days:
        out = bracketdata.build({}, tmp_path, today=day, season=2027, settings=SETTINGS)
    assert out["comparedTo"] == "2027-01-02"             # a week before Jan 9, the newest day at least that old
    assert all(abs(t["change"] - 0.7) < 1e-3 for t in out["teams"] if t["pField"] < 1.0)
    history = json.loads((tmp_path / "bracketology_history.json").read_text())
    assert len(history["days"]) == 9
    assert json.loads((tmp_path / "bracketology.json").read_text())["updated"] == "2027-01-09"


def test_frozen_after_selection_sunday(tmp_path, bb_payload):
    saved = _projection(bb_payload["teams"])
    (tmp_path / "bracketology.json").write_text(json.dumps(saved))
    field = saved["projected"]["field"]
    actual = pd.DataFrame({"team": [f["team"] for f in field], "seed": [f["seedLine"] for f in field],
                           "region": ["East"] * len(field), "bidType": [f["bidType"] for f in field]})
    out = bracketdata.build({}, tmp_path, today=dt.date(2027, 3, 16), season=2027, settings=SETTINGS,
                            actual_fn=lambda season: actual)
    assert out["frozen"] and out["result"]["named"] == len(field) and out["updated"] == saved["updated"]

    def boom(season):
        raise AssertionError("a complete result is never re-read")

    again = bracketdata.build({}, tmp_path, today=dt.date(2027, 3, 20), season=2027, settings=SETTINGS, actual_fn=boom)
    assert again["result"]["named"] == len(field)


def test_frozen_with_no_saved_projection_shows_nothing(tmp_path):
    assert bracketdata.build({}, tmp_path, today=dt.date(2027, 3, 20), season=2027, settings=SETTINGS) is None


def test_team_page_tile_carries_the_tournament_chance(bb_payload, built):
    from mri.export import site

    top = bb_payload["bracketology"]["projected"]["field"][0]["team"]
    team = next(t for t in bb_payload["teams"] if t["team"] == top)
    text = site.team_page(team, bb_payload)
    assert "NCAA Tournament" in text and "projected No. 1 seed" in text and "../bracketology.html" in text
    assert "Playoff chance" not in text
