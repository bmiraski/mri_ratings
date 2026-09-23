"""The live build's settings handling: season-keyed format overrides and the stale-format warning.

The tournament feed is stubbed so nothing here touches the API.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from mri.bracket.template import Template

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def build():
    from mri.bracket import live

    return live


def _no_tournaments(season, **_):
    return pd.DataFrame(columns=["conference", "start_date", "team1", "team2", "neutral"])


def test_override_applies_only_to_its_own_season(build, monkeypatch):
    monkeypatch.setattr(build.bb_bracket, "conference_tournaments", _no_tournaments)
    overrides = {"2027": {"X": {"tiers": [2, 3, 2, 2], "announced": True}, "Y": {"tiers": [8], "announced": False}}}
    tpl, source = build.templates_for(2027, {"X", "Y", "Z"}, overrides)
    assert tpl["X"].tiers == (2, 3, 2, 2) and source["X"] == "announced"
    assert tpl["Y"].tiers == (8,) and source["Y"] == "provisional"
    assert source["Z"] == "inferred"
    tpl, source = build.templates_for(2028, {"X"}, overrides)
    assert source["X"] == "inferred"                    # a 2027 override never leaks into 2028


def test_warns_when_an_inferred_bracket_outgrows_the_league(build):
    templates = {"Shrunk": Template((8, 4), 3, True), "TopK": Template((4, 2, 2), 3, True),
                 "Fixed": Template((2, 7), 0, True), "Unknown": None}
    source = {"Shrunk": "inferred", "TopK": "inferred", "Fixed": "provisional", "Unknown": "inferred"}
    members = {"Shrunk": 9, "TopK": 12, "Fixed": 9, "Unknown": 10}
    warnings = build.format_warnings(templates, source, members)
    assert len(warnings) == 2
    assert any(w.startswith("Shrunk") for w in warnings) and any(w.startswith("Unknown") for w in warnings)


def test_shipped_settings_are_all_model_and_every_override_is_well_formed():
    settings = json.loads((ROOT / "data" / "bracketology_settings.json").read_text())
    assert all(mode == "model" for mode in settings.get("autoBid", {}).values())
    for season, confs in settings["formatOverride"].items():
        assert season.isdigit()
        for conf, o in confs.items():
            assert all(isinstance(t, int) and t > 0 for t in o["tiers"]), conf
            assert isinstance(o["announced"], bool), conf
