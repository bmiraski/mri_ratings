"""The "How this rating is built" section: it must tie out to what the page prints."""

from __future__ import annotations

import copy
import importlib.util
import json
import re
from pathlib import Path

import pytest

from mri.export import decomposition, site

# The shared synthetic season lives beside the tests but is not a package, and
# test_site's import audit treats any bare `import x` as a third-party dependency.
_spec = importlib.util.spec_from_file_location("synthetic", Path(__file__).with_name("synthetic.py"))
synthetic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(synthetic)
synthetic_payload = synthetic.synthetic_payload

REAL = Path(__file__).resolve().parents[1] / "site" / "data" / "site.json"


def _tenths(text: str) -> float:
    return float(text.replace("−", "-"))


def _section(html: str) -> str:
    return html.split('<section class="build">')[1].split("</section>")[0]


def _build_table_adds(html: str) -> list[float]:
    body = _section(html).split("<tbody>")[1].split("</tbody>")[0]
    return [_tenths(v) for v in re.findall(r'<td class="num">([+\-−][\d.]+)</td></tr>', body)]


def _printed_power(html: str) -> float:
    return _tenths(re.search(r'statl">Power</span><span class="statv">([+\-][\d.]+)<', html).group(1))


def _results_adds(html: str) -> list[float]:
    return [_tenths(v) for v in re.findall(r'<td class="num adds"[^>]*>([+\-][\d.]+)</td>', html)]


def _results_prior(html: str) -> float:
    return _tenths(re.search(r"Preseason prior \([^)]*\)</td>\s*<td class=\"num\">([+\-][\d.]+)<", html).group(1))


@pytest.fixture(scope="module", params=[2, 5])
def payload(request) -> dict:
    return synthetic_payload(rounds=request.param)


def test_every_team_renders_the_section_and_its_parts_sum_to_the_printed_power(payload) -> None:
    for team in payload["teams"]:
        html = site.team_page(team, payload)
        assert "How this rating is built" in html
        margin, opponents, prior = _build_table_adds(html)
        assert round(margin + opponents + prior, 1) == _printed_power(html), team["team"]


def test_the_adds_column_plus_the_prior_sums_to_power(payload) -> None:
    for team in payload["teams"]:
        html = site.team_page(team, payload)
        adds = _results_adds(html)
        assert len(adds) == len(payload["details"][team["team"]]["played"])
        assert round(sum(adds) + _results_prior(html), 1) == _printed_power(html), team["team"]
        # the prior is one number on the page, not two
        assert _results_prior(html) == _build_table_adds(html)[2]


def test_schedule_faced_is_the_sections_average_opponent(payload) -> None:
    for team in payload["teams"]:
        html = site.team_page(team, payload)
        faced = _tenths(re.search(r'Schedule faced</span><span class="hlv">([+\-][\d.]+) avg', html).group(1))
        shown = _tenths(re.search(r'Opponents faced</td><td class="num">([+\-][\d.]+) avg', html).group(1))
        assert faced == shown, team["team"]


def test_early_season_reads_as_mostly_preseason() -> None:
    early = synthetic_payload(rounds=2)
    html = site.team_page(early["teams"][0], early)
    assert "mostly preseason" in html
    assert re.search(r"preseason prior is <strong>(5\d|6\d)%</strong>", html)
    late = synthetic_payload(rounds=5)
    assert "mostly preseason" not in site.team_page(late["teams"][0], late)


def test_perf_and_adds_are_explained_apart(payload) -> None:
    html = site.team_page(payload["teams"][0], payload)
    assert "<strong>Perf</strong>" in html and "<strong>Adds</strong>" in html
    assert "not independent proof" in html
    assert 'title="Margin' in html  # the hover: raw margin, counted margin, the arithmetic


def test_section_and_column_are_dropped_when_there_is_no_decomposition(payload) -> None:
    bare = copy.deepcopy(payload)
    for detail in bare["details"].values():
        detail.pop("decomposition", None)
    html = site.team_page(bare["teams"][0], bare)
    assert "How this rating is built" not in html
    assert ">Adds<" not in html and "<strong>Adds</strong>" not in html
    assert "<tfoot>" not in html
    assert "<strong>Perf</strong>" in html


def test_a_team_with_no_games_is_its_prior() -> None:
    p = synthetic_payload(rounds=2)
    team = p["teams"][0]
    p["details"][team["team"]] = {
        "played": [], "upcoming": [], "playedDifficulty": None, "remainingDifficulty": None,
        "bestWin": None, "worstLoss": None,
        "decomposition": {"n": 0, "ridge": 4.0, "w": 0.0, "avgMargin": 0.0, "avgOpponent": 0.0,
                          "prior": team["power"], "marginTerm": 0.0, "opponentTerm": 0.0,
                          "priorTerm": team["power"]},
    }
    html = site.team_page(team, p)
    assert "no games yet" in html and "preseason prior alone" in html
    assert "nan" not in re.findall(r">([^<]*)<", _section(html)).__str__().lower()
    assert _build_table_adds(html) == [round(team["power"], 1)]
    assert "Margin counted" not in html and "&times; 100%" in html


def test_largest_remainder_rounding() -> None:
    assert sum(site._round_to_total([1.04, 1.04, 1.04], 3.1)) == pytest.approx(3.1)
    assert site._round_to_total([14.85, -0.871, 17.062], 31.0) == [14.8, -0.9, 17.1]
    tied = site._round_to_total([-0.26, -0.26, -0.26], -0.8)  # equal remainders: which one gives is arbitrary
    assert sum(tied) == pytest.approx(-0.8) and sorted(tied) == pytest.approx([-0.3, -0.3, -0.2])
    assert all(abs(r - v) < 0.1 + 1e-9 for r, v in zip(site._round_to_total([2.46, 3.51, 4.03], 10.0), [2.46, 3.51, 4.03]))
    assert site._round_to_total([], 0.0) == []


def test_a_decomposition_that_misses_the_printed_power_is_not_published() -> None:
    p = synthetic_payload(rounds=3)
    team = p["teams"][0]
    detail = p["details"][team["team"]]
    block = detail["decomposition"]
    assert decomposition.team_block.__name__  # sanity
    from mri.ratings import decompose as core

    share = core.TeamShare(team["team"], block["n"], 4.0, block["w"], block["avgMargin"], block["avgOpponent"],
                           block["prior"], block["marginTerm"], block["opponentTerm"], block["priorTerm"] + 0.5, 0.0)
    assert decomposition.team_block(share, team["power"], block["n"]) is None
    assert decomposition.team_block(share, team["power"] + 0.5, block["n"] + 1) is None  # wrong game count


def test_the_private_game_key_never_reaches_the_payload(payload) -> None:
    assert "_gid" not in json.dumps(payload["details"])


@pytest.mark.skipif(
    not REAL.exists() or "decomposition" not in next(iter(json.loads(REAL.read_text()).get("details", {"": {}}).values()), {}),
    reason="the published payload predates this feature; the daily build adds it",
)
def test_every_team_in_the_published_payload_ties_out() -> None:
    p = json.loads(REAL.read_text())
    for team in p["teams"]:
        detail = p["details"][team["team"]]
        parts = site.build_parts(team, detail)
        assert round(parts["margin"] + parts["opponents"] + parts["prior"], 1) == parts["power"]
        if detail["played"]:
            assert round(sum(parts["adds"]) + parts["prior"], 1) == parts["power"]
        assert detail["playedDifficulty"] == detail["decomposition"]["avgOpponent"]
