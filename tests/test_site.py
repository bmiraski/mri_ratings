"""The generated site must be complete, linked, and safe to publish.

These are cheap structural guarantees - every team reachable, every link
resolving, nothing unescaped - so a broken build fails here rather than on the
live site.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from mri.export import site

DATA = Path(__file__).resolve().parents[1] / "site" / "data" / "site.json"

pytestmark = pytest.mark.skipif(not DATA.exists(), reason="site data not built")


@pytest.fixture(scope="module")
def payload() -> dict:
    return json.loads(DATA.read_text())


@pytest.fixture(scope="module")
def built(payload, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("site")
    site.build(payload, out)
    return out


def test_slugs_are_unique(payload) -> None:
    """Two teams sharing a slug would silently overwrite one another's page."""
    slugs = [site.slug(t["team"]) for t in payload["teams"]]
    assert len(set(slugs)) == len(slugs)


def test_every_team_has_a_page(payload, built) -> None:
    for team in payload["teams"]:
        path = built / "team" / f"{site.slug(team['team'])}.html"
        assert path.exists(), f"missing page for {team['team']}"


def test_every_conference_has_a_page(payload, built) -> None:
    for conference in payload["conferences"]:
        path = built / "conference" / f"{site.slug(conference['conference'])}.html"
        assert path.exists(), f"missing page for {conference['conference']}"


def test_core_pages_exist(built) -> None:
    for name in ("index.html", "conferences.html", "archive.html", "method.html", "styles.css"):
        assert (built / name).exists(), f"missing {name}"


def test_internal_links_all_resolve(built) -> None:
    """Walk every page and follow every relative link."""
    broken = []
    for page in built.rglob("*.html"):
        for href in re.findall(r'href="([^"]+)"', page.read_text()):
            if href.startswith(("http://", "https://", "#", "mailto:")):
                continue
            target = (page.parent / href).resolve()
            if not target.exists():
                broken.append(f"{page.relative_to(built)} -> {href}")
    assert not broken, "broken links: " + ", ".join(broken[:10])


def test_team_names_are_escaped(built) -> None:
    """Texas A&M is the canary: a raw ampersand means nothing is being escaped."""
    text = (built / "index.html").read_text()
    assert "Texas A&amp;M" in text
    assert "Texas A&M<" not in text


def test_pages_declare_a_viewport(built) -> None:
    for name in ("index.html", "method.html"):
        assert 'name="viewport"' in (built / name).read_text()


def test_sparkline_suppressed_below_three_points() -> None:
    assert "<path" not in site.sparkline([1.0, 2.0])
    assert "<path" in site.sparkline([1.0, 2.0, 3.0])


def test_sparkline_handles_a_flat_line() -> None:
    """A team whose rating never moved must not divide by zero."""
    assert "<path" in site.sparkline([5.0, 5.0, 5.0])


def test_every_team_row_links_to_its_page(payload, built) -> None:
    index = (built / "index.html").read_text()
    for team in payload["teams"][:25]:
        assert f'team/{site.slug(team["team"])}.html' in index


def test_logo_paths_are_depth_correct(payload, built) -> None:
    """A cached logo referenced from team/ must climb out of the directory,
    or every team page shows a colour chip instead."""
    cached = [t for t in payload["teams"] if not str(t.get("logo", "")).startswith("http")]
    if not cached:
        pytest.skip("logos not cached in this build")
    team = cached[0]
    text = (built / "team" / f"{site.slug(team['team'])}.html").read_text()
    assert f'src="../{team["logo"]}"' in text


def test_cached_logo_files_exist(payload, built) -> None:
    import shutil
    source = Path(__file__).resolve().parents[1] / "docs" / "logos"
    if not source.exists():
        pytest.skip("logos not cached")
    shutil.copytree(source, built / "logos", dirs_exist_ok=True)
    missing = [
        t["team"] for t in payload["teams"]
        if not str(t.get("logo", "")).startswith("http") and not (built / t["logo"]).exists()
    ]
    assert not missing, f"missing logo files for {missing[:5]}"


def test_build_is_idempotent(payload, tmp_path) -> None:
    """Two builds of unchanged data must produce byte-identical output.

    The footer timestamp is rendered into every page, so without this the
    scheduled job commits 157 files every run whether or not a game was
    played - which buries real changes in noise.
    """
    import copy

    first = copy.deepcopy(payload)
    site.build(first, tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.html")}

    second = copy.deepcopy(payload)
    second["generated"] = "2099-01-01T00:00:00"
    site.build(second, tmp_path)
    after = {p.name: p.read_bytes() for p in tmp_path.glob("*.html")}

    assert before == after, "a clock tick should not change the output"


def test_new_data_does_change_the_output(payload, tmp_path) -> None:
    """The flip side: a real change must not be suppressed."""
    import copy

    first = copy.deepcopy(payload)
    site.build(first, tmp_path)
    before = (tmp_path / "index.html").read_bytes()

    second = copy.deepcopy(payload)
    second["generated"] = "2099-01-01T00:00:00"
    second["teams"][0]["power"] += 5.0
    site.build(second, tmp_path)

    assert (tmp_path / "index.html").read_bytes() != before


def test_digest_ignores_only_the_timestamp(payload) -> None:
    import copy

    a = copy.deepcopy(payload)
    b = copy.deepcopy(payload)
    b["generated"] = "2099-01-01T00:00:00"
    assert site.content_digest(a) == site.content_digest(b)

    b["week"] = 99
    assert site.content_digest(a) != site.content_digest(b)


# --- two sports under one roof ---------------------------------------------

BB_DATA = Path(__file__).resolve().parents[1] / "site" / "data" / "bb.json"


@pytest.fixture(scope="module")
def bb_payload() -> dict:
    """bb.json plus the detail it deliberately does not carry.

    The per-team detail is 4MB for 365 teams and is rendered into the team
    pages, so it is never written to disk. The pages still need it, so the test
    rebuilds it from the cached game data the same way the real build does."""
    if not BB_DATA.exists():
        pytest.skip("basketball site data not built")
    payload = json.loads(BB_DATA.read_text())
    from mri.export import bb_sitedata
    payload["details"] = bb_sitedata.team_details(payload["season"], payload)
    betting = BB_DATA.parent / "bb_betting.json"
    if betting.exists():
        from mri.betting import bb_board
        payload["betting"] = json.loads(betting.read_text())
        payload["board"] = bb_board.build_board(payload["season"])
    return payload


@pytest.fixture(scope="module")
def both(payload, bb_payload, tmp_path_factory) -> Path:
    """Both sports rendered into one docs/ root, as the real build does."""
    out = tmp_path_factory.mktemp("both")
    sports = ["football", "basketball"]
    site.build({**payload, "sports": sports}, out)
    site.build({**bb_payload, "sports": sports}, out, publish_details=False)
    return out


def test_football_keeps_the_root(both) -> None:
    """Every URL published so far is a root URL. Moving football into a
    subdirectory would break all of them, so it stays put."""
    assert (both / "index.html").exists()
    assert (both / "team").is_dir()
    assert (both / "basketball" / "index.html").exists()


def test_one_stylesheet_for_the_whole_site(both) -> None:
    """Two copies would drift. Both sports link back to the root one."""
    assert (both / "styles.css").exists()
    assert not (both / "basketball" / "styles.css").exists()
    deep = (both / "basketball" / "team").glob("*.html")
    assert 'href="../../styles.css"' in next(deep).read_text()


def test_links_resolve_across_both_sports(both) -> None:
    import urllib.parse

    broken = []
    for page in both.rglob("*.html"):
        for href in re.findall(r'(?:href|src)="([^"]+)"', page.read_text()):
            if href.startswith(("http://", "https://", "#", "mailto:", "data:")):
                continue
            if not (page.parent / urllib.parse.unquote(href)).resolve().exists():
                broken.append(f"{page.relative_to(both)} -> {href}")
    assert not broken, broken[:10]


def test_each_sport_switch_points_at_the_other(both) -> None:
    """The switch is the only cross-sport link, and it is on every page. A
    depth mistake here breaks hundreds of pages at once, silently."""
    seen = 0
    for page in both.rglob("*.html"):
        html = page.read_text()
        assert 'class="sports"' in html, page
        assert html.count('aria-current="page"') == 1, page
        seen += 1
    assert seen > 500


def test_a_solo_build_shows_no_switch(payload, tmp_path) -> None:
    """A build that publishes one sport must not put a header link on every
    page pointing at a directory it did not create."""
    site.build(payload, tmp_path)
    html = (tmp_path / "index.html").read_text()
    assert 'class="sports"' not in html
    assert "basketball/index.html" not in html


def test_basketball_publishes_no_classic_column(bb_payload, both) -> None:
    """Basketball has one rating. A Classic column would be an empty promise."""
    index = (both / "basketball" / "index.html").read_text()
    assert "MRI Classic" not in index
    assert "Where Classic disagrees" not in index


def test_basketball_json_omits_the_bulky_detail(both) -> None:
    """4MB recommitted weekly, for something already rendered into every team
    page."""
    published = json.loads((both / "basketball" / "basketball.json").read_text())
    assert "details" not in published
    assert published["teams"]


def test_sports_do_not_share_a_json_file(both) -> None:
    assert (both / "site.json").exists()
    assert (both / "basketball" / "basketball.json").exists()


def test_basketball_build_is_idempotent(bb_payload, tmp_path) -> None:
    """The weekly Action commits whatever changed. A build that rewrites 400
    files for a timestamp commits 400 files every week."""
    site.build(bb_payload, tmp_path, publish_details=False)
    first = {p: p.read_bytes() for p in sorted(tmp_path.rglob("*")) if p.is_file()}
    site.build(bb_payload, tmp_path, publish_details=False)
    second = {p: p.read_bytes() for p in sorted(tmp_path.rglob("*")) if p.is_file()}
    assert first == second


def test_hidden_actually_hides(built) -> None:
    """The conference filter sets el.hidden on the rows it wants gone.

    The browser's own stylesheet hides [hidden] with display:none, but that is a
    UA rule and any author rule setting display beats it. .row is display:flex,
    so for as long as this rule was missing the filter marked 122 of 138 rows
    hidden and every one of them stayed on screen - the dropdown did nothing at
    all, silently, in production.

    Anything the site hides with the attribute depends on this one rule, so it
    is asserted rather than trusted.
    """
    css = (built / "styles.css").read_text()
    assert re.search(r"\[hidden\]\s*\{[^}]*display:\s*none\s*!important", css), (
        "the stylesheet must force [hidden] to display:none, or every "
        "display-setting class silently defeats it"
    )


# --- conference strength ----------------------------------------------------

def test_conference_strength_is_top_weighted_not_a_plain_mean() -> None:
    """A deep league with a weak tail should not lose to a small clean one on
    the strength of its floor. Each next-best team counts less."""
    from mri.export import common

    deep = [{"conference": "Deep", "power": p} for p in (30, 28, 26, 24, 5, 4, 3, 2)]
    flat = [{"conference": "Flat", "power": p} for p in (16, 16, 16, 16, 16, 16, 16, 16)]
    rows = {r["conference"]: r for r in common.conference_strength(deep + flat)}

    assert rows["Deep"]["mean"] < rows["Flat"]["mean"], "the plain means must disagree"
    assert rows["Deep"]["strength"] > rows["Flat"]["strength"]


def test_conference_strength_does_not_reward_being_small() -> None:
    """Truncating a conference to its best teams must not raise its rating -
    that is the failure mode of a top-N measure, and why this decays instead."""
    from mri.export import common

    full = [{"conference": "A", "power": p} for p in (30, 28, 26, 24, 22, 20)]
    trimmed = full[:4]
    a = common.conference_strength(full)[0]["strength"]
    b = common.conference_strength(trimmed)[0]["strength"]
    assert b > a, "a shorter, equally strong-at-the-top league should score higher here"
    assert abs(b - a) < 3.0, "but only slightly - not by discarding its tail wholesale"


def test_groups_too_small_to_be_leagues_are_not_ranked() -> None:
    """Football's two independents came third on this measure. They are
    arithmetically strong and they are not a conference."""
    from mri.export import common

    teams = [{"conference": "Big", "power": p} for p in (20, 19, 18, 17, 16, 15)]
    teams += [{"conference": "Independent", "power": p} for p in (34, 30)]
    rows = common.conference_strength(teams)

    assert rows[0]["conference"] == "Big"
    assert rows[0]["ranked"]
    assert not rows[-1]["ranked"]
    assert rows[-1]["conference"] == "Independent"


def test_unranked_groups_stay_off_the_strength_panel(built) -> None:
    """They keep a page and a filter entry; they just do not appear in a
    ranking of conferences."""
    index = (built / "index.html").read_text()
    panel = index.split("Conference strength", 1)[1].split("</section>", 1)[0]
    assert "FBS Independent" not in panel
    assert (built / "conference" / "fbs-independent.html").exists()
    assert 'value="FBS Independent"' in index  # still in the filter


def test_basketball_betting_page_leads_with_the_verdict(both) -> None:
    """A betting page that opens with picks and hides its record is a tout
    sheet. This one has no picks to open with, and must say so first."""
    page = both / "basketball" / "betting.html"
    if not page.exists():
        pytest.skip("basketball betting data not built")
    html = page.read_text()
    verdict = html.index("does not beat the market")
    assert verdict < html.index("Where they disagree now")
    # Football's word for its filtered list. Nothing here has earned it.
    assert "flagged" not in html.lower()


def test_the_intermediate_json_is_idempotent_too(tmp_path) -> None:
    """docs/ stopped churning on a timestamp a while ago; site/data/ did not,
    and once it joined the committed paths every scheduled run produced a
    two-file diff containing nothing but a new clock reading."""
    from mri.export import common

    payload = {"season": 2026, "teams": [{"team": "A", "power": 1.0}],
               "generated": "2026-01-01T00:00:00+00:00"}
    path = tmp_path / "site.json"
    common.settle_timestamp(payload, path)
    path.write_text(json.dumps(payload))

    again = dict(payload, generated="2026-06-01T12:00:00+00:00")
    common.settle_timestamp(again, path)
    assert again["generated"] == "2026-01-01T00:00:00+00:00"

    # Real ratings change: the caller's fresh stamp is kept, not overwritten.
    # (settle_timestamp only ever carries an old stamp forward; it relies on the
    # caller having set "now" first, which every build does.)
    changed = dict(again, teams=[{"team": "A", "power": 2.0}],
                   generated="2026-06-01T12:00:00+00:00")
    common.settle_timestamp(changed, path)
    assert changed["generated"] == "2026-06-01T12:00:00+00:00"


def test_every_third_party_import_is_declared() -> None:
    """openpyxl was imported by two modules and listed in no requirements file.
    In CI that is not a skipped test, it is three files failing to collect, so
    the scheduled run's whole test gate fell over before it ran anything."""
    import ast
    import sys

    root = Path(__file__).resolve().parents[1]
    declared = {
        line.split(">=")[0].split("==")[0].strip().lower()
        for line in (root / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    stdlib = set(sys.stdlib_module_names)

    imported = set()
    for folder in ("src", "scripts", "tests"):
        for file in (root / folder).rglob("*.py"):
            for node in ast.walk(ast.parse(file.read_text())):
                if isinstance(node, ast.Import):
                    imported.update(a.name.split(".")[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    imported.add(node.module.split(".")[0])

    missing = sorted(
        m for m in imported
        if m not in stdlib and m != "mri" and m.lower() not in declared
    )
    assert not missing, f"imported but not in requirements.txt: {missing}"
