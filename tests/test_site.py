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
    # A handful of packages install under a different name than they import as.
    aliases = {"PIL": "pillow", "yaml": "pyyaml", "dateutil": "python-dateutil",
               "bs4": "beautifulsoup4", "sklearn": "scikit-learn"}

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
        if m not in stdlib and m != "mri"
        and aliases.get(m, m).lower() not in declared
    )
    assert not missing, f"imported but not in requirements.txt: {missing}"


def test_basketball_logos_are_cached_not_hotlinked() -> None:
    """365 remote images is 365 requests to somebody else's CDN on every page
    load, and a site that breaks when they reorganise it.

    Asserted against the published JSON rather than the intermediate: logos are
    cached between the two, so site/data/ holds the remote URLs by design and
    docs/ holds what readers actually get."""
    published = Path(__file__).resolve().parents[1] / "docs" / "basketball" / "basketball.json"
    if not published.exists():
        pytest.skip("site not built")
    teams = json.loads(published.read_text())["teams"]
    remote = [t for t in teams if (t.get("logo") or "").startswith("http")]
    assert not remote, f"{len(remote)} logos still point at a remote host"
    have = [t for t in teams if t.get("logo")]
    assert len(have) > 350, f"only {len(have)} of {len(teams)} teams have a logo"


def test_cached_logos_are_small_enough_to_commit() -> None:
    """ESPN serves 500px and ignores size hints. Stored as they arrive, 365 of
    them is 23MB in the repository for marks drawn at 46px."""
    folder = Path(__file__).resolve().parents[1] / "docs" / "basketball" / "logos"
    if not folder.exists():
        pytest.skip("logos not cached")
    sizes = [f.stat().st_size for f in folder.iterdir() if f.is_file()]
    assert sizes
    assert max(sizes) < 60_000, f"largest cached logo is {max(sizes)} bytes"
    assert sum(sizes) < 12_000_000, f"logo cache is {sum(sizes) / 1e6:.1f}MB"


def test_a_failed_logo_falls_back_to_the_colour_chip() -> None:
    """A logo left pointing at a URL that just failed is a broken image on
    every page that draws the team."""
    from mri.export import logos

    payload = {"teams": [{"team": "Nowhere", "logo": "https://example.invalid/x.png",
                          "color": "#123456"}]}
    summary = logos.cache_logos(payload, Path("/tmp/mri-logo-test"))
    assert summary["failed"] == 1
    assert payload["teams"][0]["logo"] is None


# --- season archive ---------------------------------------------------------

def test_the_archive_excludes_the_season_being_played() -> None:
    """A live season belongs on the rankings page. An archive entry for it is a
    second, staler answer to the same question, in the place people go for the
    record."""
    from mri.export import seasons

    entries = seasons.football_seasons(current=2026)
    assert entries
    assert max(s["season"] for s in entries) < 2026


def test_the_archive_excludes_mid_season_workbooks() -> None:
    """MRIBasketball201718.xlsx is a December snapshot - 1,521 games, nobody
    past twelve. Published as '2017-18 final' it is simply a wrong answer."""
    from mri.export import seasons

    entries = seasons.basketball_seasons()
    assert entries, "basketball archive is empty"
    # 2017-18 is in the archive now, but as a full season computed from the API
    # rather than the December workbook - the invariant is that no published
    # season is a snapshot, not that this particular year is absent.
    for entry in entries:
        best = entry["teams"][0]
        assert best["wins"] + best["losses"] >= 20, (
            f"{entry['label']}: leader played {best['wins'] + best['losses']} games"
        )
    snapshot = next(e for e in entries if e["season"] == 2018)
    assert snapshot["source"] == "computed"
    assert snapshot["teams"][0]["wins"] >= 30, "2017-18 is the snapshot again"


def test_archive_seasons_rank_from_one_with_no_gaps() -> None:
    """current_ratings rates FCS teams too, so an unfiltered football table
    showed Idaho at #94 among FBS teams and ranks with holes in them."""
    from mri.export import seasons

    for entry in seasons.football_seasons(current=2026) + seasons.basketball_seasons():
        ranks = [t["rank"] for t in entry["teams"]]
        assert ranks == list(range(1, len(ranks) + 1)), f"{entry['label']}: {ranks[:8]}"


def test_the_two_ratings_are_never_mixed_in_one_table() -> None:
    """Classic counts cumulative points where 150 is great; MRI 2.0 counts
    points against an average team where +35 is. One sorted column of both
    would invite a comparison that does not exist.

    Read from docs/ rather than a temp build: the archive is assembled by the
    build script, not by sitedata, so the intermediate the fixture uses never
    carries it."""
    page = Path(__file__).resolve().parents[1] / "docs" / "seasons.html"
    if not page.exists():
        pytest.skip("site not built")
    html = page.read_text()
    # Each system heads its own table.
    assert html.count("<table>") >= 2
    assert "MRI Classic" in html and "MRI 2.0" in html


def test_each_season_claims_only_the_provenance_it_has() -> None:
    """Football's Classic seasons are recomputed from the game logs and checked
    against what was published. Basketball's are read straight out of the
    workbook. Claiming the second is the first would be inventing a
    verification."""
    from mri.export import seasons

    for entry in seasons.football_seasons(current=2026):
        if entry["system"] == seasons.CLASSIC:
            assert entry["source"] == "recomputed"
    # Basketball Classic is now a mix: the workbook years are Ben's published
    # ratings; 2013-14 to 2017-18 are years he never ran, computed here. Neither
    # may claim to be the other.
    workbooks = {2013, 2019, 2020}
    for entry in seasons.basketball_seasons():
        if entry["system"] != seasons.CLASSIC:
            continue
        expected = "published" if entry["season"] in workbooks else "computed"
        assert entry["source"] == expected, f"{entry['label']}: {entry['source']}"
        assert not entry["matchesPublished"]


def test_renamed_programs_have_one_history_not_two() -> None:
    """The workbooks and the API spell many programs differently - Cal,
    Central Florida, Mississippi, North Carolina State, Troy State and ten more
    in football; over fifty in basketball. Unresolved, each of those teams had
    two histories that did not know about each other, and a team page showed
    whichever half matched its own spelling.

    The registries already held every mapping. Nothing was asking them."""
    from mri.export import seasons
    from mri.ingest import registry

    history = seasons.team_history(seasons.football_seasons(current=2026))
    stale = [name for name in history if registry.resolve(name, name) != name]
    assert not stale, f"histories still filed under superseded names: {stale}"

    for old in ("Cal", "Central Florida", "Mississippi", "North Carolina State",
                "Troy State", "Miami (Ohio)", "Southern Mississippi"):
        assert old not in history, f"{old} still has its own history"

    # And the joined result actually spans both ratings.
    for team in ("California", "UCF", "Ole Miss", "NC State", "Troy"):
        systems = {r["system"] for r in history[team]}
        assert systems == {"MRI Classic", "MRI 2.0"}, f"{team}: {systems}"


def test_basketball_histories_are_joined_too() -> None:
    from mri.export import seasons

    history = seasons.team_history(seasons.basketball_seasons(current=2027))
    spanning = sum(1 for rows in history.values()
                   if len({r["system"] for r in rows}) == 2)
    assert spanning > 300, f"only {spanning} teams span both ratings"
    for old in ("Cal", "Central Florida", "Connecticut", "Louisiana-Monroe"):
        assert old not in history, f"{old} still has its own history"


def test_history_rows_carry_the_field_size() -> None:
    """A rank travels between the two ratings; the field it was a rank of grew
    from 117 teams to 138, so the row has to say."""
    from mri.export import seasons

    for rows in seasons.team_history(seasons.football_seasons(current=2026)).values():
        for row in rows:
            assert row["of"] >= row["rank"] > 0


def test_archive_field_size_matches_the_real_one_each_season() -> None:
    """The strongest available check: the number of teams the archive rates in
    a season should equal the number the API says were FBS that year."""
    from mri.export import seasons
    from mri.ingest import cfbd

    for entry in seasons.football_seasons(current=2026):
        if entry["system"] != "MRI 2.0":
            continue
        try:
            actual = len(cfbd.fbs_teams(entry["season"]))
        except Exception:  # noqa: BLE001
            pytest.skip("FBS roster unavailable offline")
        assert entry["rated"] == actual, (
            f"{entry['season']}: archive rates {entry['rated']}, API says {actual}"
        )


def test_no_team_is_rated_before_it_joined_fbs() -> None:
    from mri.export import seasons
    from mri.ingest import registry

    for entry in seasons.football_seasons(current=2026):
        if entry["system"] != "MRI 2.0":
            continue
        for team in entry["teams"]:
            assert registry.was_fbs(team["team"], entry["season"]), (
                f"{team['team']} rated in {entry['season']} but was not FBS"
            )


# --- per-season game logs ---------------------------------------------------

def test_game_logs_exist_only_for_seasons_a_team_was_rated() -> None:
    """Without pruning, every FCS team that ever appeared on a schedule gets a
    page - thousands of them, for teams the site does not rank."""
    from mri.export import gamelogs, seasons

    history = seasons.team_history(seasons.football_seasons(current=2026))
    logs = gamelogs.prune(gamelogs.football(2026), history)
    rated = {(row["season"], team) for team, rows in history.items() for row in rows}
    for season, teams in logs.items():
        for team in teams:
            assert (season, team) in rated, f"{team} has a {season} log but no {season} rating"


def test_game_logs_are_chronological_where_dates_exist() -> None:
    """The feed does not return them in order: a 2025 log arrived with a bowl
    game sitting between week one and week sixteen."""
    from mri.export import gamelogs, seasons

    history = seasons.team_history(seasons.football_seasons(current=2026))
    logs = gamelogs.prune(gamelogs.football(2026), history)
    for teams in logs.values():
        for rows in teams.values():
            dated = [r["when"] for r in rows if r["when"]]
            assert dated == sorted(dated)


def test_pre_2020_logs_carry_no_expected_margin() -> None:
    """MRI 2.0 never rated those seasons, and a column of dashes that looks
    like a missing value is worse than a stated absence."""
    from mri.export import gamelogs, seasons

    history = seasons.team_history(seasons.football_seasons(current=2026))
    logs = gamelogs.prune(gamelogs.football(2026), history)
    for rows in logs[2011].values():
        assert all(r["expected"] is None for r in rows)
    assert any(r["expected"] is not None
               for rows in logs[2025].values() for r in rows)


def test_pooled_opponents_are_flagged_not_printed_as_a_school() -> None:
    """The workbooks pooled every non-FBS opponent into "Non D1A". Roughly one
    pre-2020 row in nine, and it is not the name of a team."""
    from mri.export import gamelogs, seasons

    history = seasons.team_history(seasons.football_seasons(current=2026))
    logs = gamelogs.prune(gamelogs.football(2026), history)
    pooled = [r for rows in logs[2011].values() for r in rows if r["pooled"]]
    assert pooled, "expected some pooled opponents in a 2011 log"

    page = Path(__file__).resolve().parents[1] / "docs" / "team" / "alabama" / "2011.html"
    if page.exists():
        html = page.read_text()
        assert "Non D1A" not in html
        assert "non-FBS opponent" in html


def test_a_game_log_page_states_its_season_record(built) -> None:
    """The page has to agree with the summary row that links to it."""
    from mri.export import gamelogs, seasons

    history = seasons.team_history(seasons.football_seasons(current=2026))
    logs = gamelogs.prune(gamelogs.football(2026), history)
    for team, rows in list(history.items())[:20]:
        for row in rows:
            log = (logs.get(row["season"]) or {}).get(team)
            if not log:
                continue
            wins = sum(1 for g in log if g["won"])
            assert wins == row["wins"], f"{team} {row['season']}: {wins} vs {row['wins']}"


# --- champions and the rank chart -------------------------------------------

def test_every_completed_season_has_exactly_one_champion() -> None:
    """A title is a finishing position, so each season awards one and only one.
    Two would mean the rank recomputation left a duplicate; none would mean it
    started from zero somewhere."""
    from mri.export import seasons

    for entries in (seasons.football_seasons(current=2026),
                    seasons.basketball_seasons(current=2027)):
        for entry in entries:
            firsts = [t for t in entry["teams"] if t["rank"] == 1]
            assert len(firsts) == 1, f"{entry['label']}: {len(firsts)} teams at #1"


def test_titles_come_only_from_finished_seasons() -> None:
    """Leading in week three is not winning anything. The archive holds only
    completed seasons, and the banner reads from the archive - this asserts that
    the live season cannot leak into it."""
    from mri.export import seasons, site

    entries = seasons.football_seasons(current=2026)
    payload = {"history": seasons.team_history(entries)}
    champions = {t for t in payload["history"] if site.titles_of(t, payload)}
    assert champions
    for team in champions:
        for row in site.titles_of(team, payload):
            assert row["season"] < 2026, f"{team} credited with the live season"

    # Alabama's six is the checkable case.
    assert len(site.titles_of("Alabama", payload)) == 6


def test_the_banner_marks_champions_and_only_champions() -> None:
    """Read from docs/ rather than a temp build: the archive is assembled by the
    build script, so the intermediate the fixture uses carries no history and
    every page would look title-less."""
    docs = Path(__file__).resolve().parents[1] / "docs" / "team"
    if not (docs / "alabama.html").exists():
        pytest.skip("site not built")

    won = (docs / "alabama.html").read_text()
    assert "MRI Champion" in won
    assert "6 football titles" in won.lower()
    for year in ("2009", "2011", "2020"):
        assert year in won

    for name in ("vanderbilt", "kansas-state", "syracuse"):
        page = docs / f"{name}.html"
        if page.exists():
            assert "MRI Champion" not in page.read_text(), f"{name} has no titles"


def test_the_rank_chart_never_crops_a_season() -> None:
    """The axis is scaled to the team rather than the field, which is only
    honest if it still contains every point."""
    import re

    from mri.export import seasons, site

    entries = seasons.football_seasons(current=2026)
    history = seasons.team_history(entries)
    drawn = 0
    for rows in history.values():
        svg = site._rank_chart(rows)
        if not svg:
            continue
        drawn += 1
        axis_max = int(re.search(r'axis running to\s*(\d+)', svg).group(1))
        assert axis_max >= max(r["rank"] for r in rows)
        assert axis_max >= 25, "the floor stops one-place wobbles being drawn as drama"
        # One dot per season, gold only for the wins.
        assert svg.count("<circle") == len(rows)
        assert svg.count('class="champ"') == sum(1 for r in rows if r["rank"] == 1)
    assert drawn > 100


def test_the_rank_chart_is_suppressed_when_there_is_no_shape() -> None:
    """Same rule the sparklines follow: two points always draw a full-slope
    line whatever the underlying change."""
    from mri.export import site

    row = {"season": 2024, "label": "2024", "system": "MRI 2.0", "ratingName": "Power",
           "rank": 4, "of": 134, "wins": 10, "losses": 3, "rating": 20.0}
    assert site._rank_chart([row]) == ""
    assert site._rank_chart([row, dict(row, season=2025)]) == ""
    assert site._rank_chart([row, dict(row, season=2025), dict(row, season=2023)])


def test_brand_assets_are_published(built: Path) -> None:
    """The header, the favicon and the share card all point at real files.

    They are copied out of site/assets/web rather than generated here, so the
    failure mode is a page referencing a file nobody copied - which a browser
    reports as a missing logo and nothing else notices.
    """
    assets = built / "assets"
    for name in ("mri-lockup.png", "mri-lockup-ink.png", "favicon.ico",
                 "icon-192.png", "apple-touch-icon.png", "mri-card.png"):
        assert (assets / name).exists(), f"{name} was not published"
    assert (built / "favicon.ico").exists(), "browsers ask for /favicon.ico by habit"


def test_the_light_theme_gets_its_own_wordmark(built: Path) -> None:
    """The wordmark is white, so the same file on a light background is blank.

    <picture> swaps it for the ink version under prefers-color-scheme: light.
    If that source ever goes missing the page still renders - with an invisible
    logo for every reader whose system is set to light.
    """
    page = (built / "index.html").read_text()
    assert 'media="(prefers-color-scheme: light)"' in page
    assert "mri-lockup-ink.png" in page

    from PIL import Image
    import numpy as np

    def ink(name: str) -> float:
        art = np.array(Image.open(built / "assets" / name).convert("RGBA"))
        # The wordmark half only; the icon is a red field in both files.
        right = art[:, art.shape[1] // 2:, :]
        visible = right[..., 3] > 128
        return float(right[..., :3][visible].mean())

    assert ink("mri-lockup.png") > 200, "the dark-theme wordmark should be near white"
    assert ink("mri-lockup-ink.png") < 60, "the light-theme wordmark should be near black"


def test_the_share_card_is_not_transparent(built: Path) -> None:
    """Link previews composite onto a background of their own choosing, which
    for a white wordmark on transparency is usually white on white."""
    from PIL import Image

    card = Image.open(built / "assets" / "mri-card.png")
    assert card.mode in ("RGB", "P"), f"share card carries an alpha channel ({card.mode})"
    assert card.size == (1200, 630)


# ---- season simulation, slate and public record

BACKTEST = Path(__file__).resolve().parents[1] / "site" / "data" / "sim_backtest.json"


@pytest.fixture(scope="module")
def extended(payload) -> dict:
    """The payload with the three football extras, filled with plausible numbers."""
    p = json.loads(json.dumps(payload))
    teams = p["teams"]
    odds = {
        t["team"]: {
            "conferenceGame": 0.1, "conferenceTitle": 0.05 + 0.5 / t["rank"], "playoff": 0.9 / t["rank"],
            "bye": 0.4 / t["rank"], "quarterfinal": 0.5 / t["rank"], "semifinal": 0.3 / t["rank"],
            "final": 0.2 / t["rank"], "title": 0.1 / t["rank"], "projectedWins": 9.0, "projectedLosses": 3.0,
            "unbeaten": 0.01, "top12": 0.1, "gamesLeft": 10, "playoffChange": 0.012, "titleChange": -0.001,
        }
        for t in teams
    }
    p["sim"] = {"season": p["season"], "week": p["week"], "slateWeek": p["week"], "sims": 10000,
                "fieldSize": 12, "teams": odds, "leverage": {}, "hasHistory": True,
                "reconstructedWeeks": [1]}
    home, away = teams[0]["team"], teams[1]["team"]
    game = {"id": 1, "week": p["week"], "date": "2026-09-26", "dateLabel": "Sat, Sep 26", "time": "7:30 PM ET",
            "sort": "1930", "home": home, "away": away, "neutral": False, "predicted": 7.4,
            "homeWinProbability": 0.7, "played": False, "market": 6.5, "open": 5.5, "total": 51.5,
            "edge": 1.9, "flagged": False,
            "stake": {"side": "home", "ifWin": 0.6, "ifLose": 0.2, "swing": 0.4, "team": home}}
    done = {**game, "id": 2, "played": True, "dateLabel": "Thu, Sep 24",
            "result": {"homeScore": 27, "awayScore": 13, "modelCorrect": True, "modelError": 6.6,
                       "marketError": 7.5}}
    fcs = {"id": 3, "week": p["week"], "date": "2026-09-26", "dateLabel": "Sat, Sep 26", "time": "TBD",
           "sort": "9999", "home": home, "away": "Some FCS", "neutral": False, "predicted": 30.0,
           "homeWinProbability": 0.99, "played": False}
    p["slate"] = {"week": p["week"], "days": [{"date": "2026-09-26", "label": "Sat, Sep 26", "games": [game]}],
                  "results": [done], "fcs": [fcs], "watch": [1], "flagged": 0, "games": 3}
    summary = {"games": 100, "accuracy": 0.8, "mae": 14.0, "priced": 100, "modelMaePriced": 14.0,
               "marketMae": 12.0, "marketAccuracy": 0.82, "modelAccuracyPriced": 0.8, "maeGap": 2.0,
               "maeGapError": 1.4, "slope": 0.73, "marketSlope": 0.93,
               "bets": {"count": 10, "wins": 4, "losses": 6, "pushes": 0, "ats": 0.4, "units": -2.4, "clv": 0.3}}
    p["record"] = {
        "reference": {"seasons": "2003–2019", "accuracy": 0.738, "mae": 13.0},
        "reconstructed": {"summary": summary, "bets": [], "weeks": [
            {"week": 1, "games": 50, "accuracy": 0.8, "mae": 15.0, "marketMae": 12.0, "bets": 5,
             "record": "2-3", "units": -1.2}]},
        "forward": {"started": "2026-09-19", "logged": 1, "graded": 0, "wins": 0, "losses": 0, "pushes": 0,
                    "ats": None, "units": 0, "clv": None,
                    "picks": [{"game_id": 1, "week": 4, "home": home, "away": away, "neutral": False,
                               "side": "home", "predicted": 12.0, "taken": 6.5, "edge": 5.5,
                               "kickoff": "2026-09-26T23:30:00.000Z", "loggedAt": "2026-09-24T11:00Z"}]},
    }
    if BACKTEST.exists():
        p["simBacktest"] = json.loads(BACKTEST.read_text())
    return p


@pytest.fixture(scope="module")
def built_extended(extended, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("site_extended")
    site.build(extended, out)
    return out


def test_new_pages_exist_and_are_linked(built_extended) -> None:
    for name in ("simulation.html", "slate.html", "simulation.json", "slate.json", "record.json"):
        assert (built_extended / name).exists(), name
    index = (built_extended / "index.html").read_text()
    assert 'href="simulation.html"' in index and 'href="slate.html"' in index


def test_links_still_resolve_with_the_new_pages(built_extended) -> None:
    broken = []
    for page in built_extended.rglob("*.html"):
        for href in re.findall(r'href="([^"]+)"', page.read_text()):
            if href.startswith(("http://", "https://", "#", "mailto:")):
                continue
            if not (page.parent / href.split("#")[0]).resolve().exists():
                broken.append(f"{page.relative_to(built_extended)} -> {href}")
    assert not broken, broken[:10]


def test_nav_promises_only_what_the_build_produced(built) -> None:
    """The stock payload has no simulation, so the header must not link to one."""
    text = (built / "index.html").read_text()
    assert "simulation.html" not in text and "slate.html" not in text
    assert not (built / "simulation.html").exists()


def test_the_bulky_extras_stay_out_of_site_json(built_extended) -> None:
    published = json.loads((built_extended / "site.json").read_text())
    for key in ("sim", "slate", "record", "simBacktest"):
        assert key not in published


def test_simulation_page_lists_every_team_escaped(built_extended, extended) -> None:
    text = (built_extended / "simulation.html").read_text()
    assert text.count("<tr data-conf=") == len(extended["teams"])
    assert "Texas A&amp;M" in text and "Texas A&M<" not in text
    assert "Season simulation" in text


def test_slate_page_shows_each_section(built_extended) -> None:
    text = (built_extended / "slate.html").read_text()
    for needle in ("Most riding on it", "Already played this week", "Against FCS opponents", "Some FCS"):
        assert needle in text, needle


def test_betting_page_carries_the_record_and_says_what_it_is(extended) -> None:
    root = Path(__file__).resolve().parents[1] / "site" / "data"
    if not (root / "betting.json").exists() or not (root / "board.json").exists():
        pytest.skip("no betting data built")
    betting = json.loads((root / "betting.json").read_text())
    board = json.loads((root / "board.json").read_text())
    text = site.betting_page(extended, betting, board)
    assert "The season, reconstructed" in text and "The forward log" in text
    assert "backtest however carefully" in text
    # The stock page, without a record, still renders.
    bare = {k: v for k, v in extended.items() if k != "record"}
    assert "The forward log" not in site.betting_page(bare, betting, board)


def test_record_section_reports_the_model_losing_when_it_is() -> None:
    record = {
        "reference": {"seasons": "2003", "accuracy": 0.738, "mae": 13.0},
        "reconstructed": {"summary": {
            "games": 40, "accuracy": 0.75, "mae": 15.0, "modelMaePriced": 15.0, "marketMae": 12.0,
            "marketAccuracy": 0.8, "maeGap": 3.0, "maeGapError": 1.0, "slope": 0.7, "marketSlope": 0.95,
            "bets": {"count": 5, "wins": 1, "losses": 4, "pushes": 0, "ats": 0.2, "units": -2.6, "clv": None}},
            "weeks": [], "bets": []},
        "forward": {"started": "2026-09-19", "logged": 0, "graded": 0, "wins": 0, "losses": 0, "pushes": 0,
                    "ats": None, "units": 0, "clv": None, "picks": []},
    }
    text = site._record_section(record)
    assert "3.0 points worse than the market" in text
    assert "Nothing logged yet" in text


def test_probabilities_are_not_shown_with_false_precision() -> None:
    assert site._pct(0.0) == "&lt;0.1%"
    assert site._pct(0.0432) == "4.3%"
    assert site._pct(0.437) == "44%"
    assert site._pct(0.9993) == "&gt;99%"
    assert site._pct(0.031, signed=True) == "+3.1"
    assert site._pct(-0.0004, signed=True) == "0"


def test_method_page_reports_the_simulation_check(built_extended) -> None:
    text = (built_extended / "method.html").read_text()
    assert 'id="simulation"' in text
    if BACKTEST.exists():
        assert "Brier score" in text


# ---- roster context on team pages, and the priors section of the method page

def test_team_page_shows_talent_against_results_and_returning_production(extended) -> None:
    p = json.loads(json.dumps(extended))
    team = p["teams"][0]
    team["roster"] = {"talent": 1003.7, "talentRank": 1, "talentOf": 127, "talentImplied": 17.0,
                      "talentGap": 16.6, "returning": 0.687, "returningRank": 25, "returningOf": 128}
    text = site.team_page(team, p)
    assert "Roster talent" in text and "#1 of 127" in text and "beating it" in text
    assert "69% of last year" in text and "#25 of 128" in text
    assert "counted against its preseason rating" not in text

    team["roster"]["returning"], team["roster"]["talentGap"] = 0.02, 1.0
    low = site.team_page(team, p)
    assert "counted against its preseason rating" in low and "in line with it" in low


def test_team_page_says_the_academies_are_not_comparable(extended) -> None:
    p = json.loads(json.dumps(extended))
    team = p["teams"][0]
    team["roster"] = {"talent": None, "talentNote": "unmeasured", "returning": None}
    text = site.team_page(team, p)
    assert "not comparable" in text and "Returning production" not in text


def test_team_page_without_roster_data_still_renders(extended) -> None:
    p = json.loads(json.dumps(extended))
    team = p["teams"][0]
    team["roster"] = None
    assert "Roster talent" not in site.team_page(team, p)


def test_method_page_explains_the_prior_and_grades_it(extended, tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    model = root / "data" / "prior_model.json"
    if not model.exists():
        pytest.skip("no prior model fitted")
    p = json.loads(json.dumps(extended))
    p["priorModel"] = json.loads(model.read_text())
    backtest = root / "site" / "data" / "prior_backtest.json"
    if backtest.exists():
        p["priorBacktest"] = json.loads(backtest.read_text())
    site.build(p, tmp_path)
    text = (tmp_path / "method.html").read_text()
    assert 'id="priors"' in text and "returning production" in text.lower()
    if backtest.exists():
        assert "Old prior miss" in text
    bare = {k: v for k, v in p.items() if k not in ("priorModel", "priorBacktest")}
    assert 'id="priors"' not in site.method_page(bare)


# ---- the GameDay page

@pytest.fixture(scope="module")
def with_gameday(extended) -> dict:
    p = json.loads(json.dumps(extended))
    a, b, c = (t["team"] for t in p["teams"][:3])
    game = {"kind": "game", "home": a, "away": b, "neutral": False, "venue": "Some Stadium", "probability": 0.42,
            "rankHome": 4.0, "rankAway": 9.0, "bothTop10": 0.55, "bothUnbeaten": 0.2}
    cg = {"kind": "championship", "conference": "SEC", "probability": 0.5, "rankHome": 3.0, "rankAway": 8.0,
          "bothTop10": 0.9, "bothUnbeaten": 0.1}
    p["gameday"] = {
        "season": 2026, "week": 3, "sims": 4000,
        "announced": [{"week": 1, "date": "2026-09-05", "city": "Baton Rouge, LA", "teams": [a, b], "host": a},
                      {"week": 2, "date": "2026-09-12", "city": "Austin, TX", "teams": [c, b], "host": c}],
        "weeks": [{"week": 8, "date": "Oct 24", "iso": "2026-10-24", "championship": False, "other": 0.05,
                   "games": [game], "covered": 0.42, "omitted": 9},
                  {"week": 14, "date": "Dec 5", "iso": "2026-12-05", "championship": True, "other": 0.05,
                   "games": [cg], "covered": 0.5}],
        "check": [{"week": 1, "date": "2026-09-05", "teams": [a, b], "host": a, "probability": 0.02, "rank": 6,
                   "favourite": {"home": c, "away": b, "probability": 0.3}}],
        "sites": [{"team": a, "hostsAtLeastOnce": 0.6, "appearsAtLeastOnce": 0.7}],
        "armyNavy": {"estimate": 0.1, "lastFour": 0, "visitsSince2014": 8, "seasons": 12},
        "model": {"features": ["best_rank", "worst_rank", "both_top10", "both_top25", "both_unbeaten", "losses"],
                  "coefficients": [-0.02, -0.98, 0.14, 0.94, 0.04, -0.97], "weeks": 107, "fitted": "x",
                  "otherRate": 0.053},
    }
    p["gamedayBacktest"] = {
        "choice": {"meanCandidates": 52, "chosen": "c", "candidateSets": {"c": {"top1": 0.64, "top3": 0.89}},
                   "baseline": {"uniformTop1": 0.024}},
        "forecast": {"stops": 69, "top1": 0.41, "top3": 0.64, "top5": 0.77}}
    return p


def test_the_gameday_page_and_its_link_exist_only_when_built(with_gameday, extended, tmp_path) -> None:
    site.build(with_gameday, tmp_path / "a")
    site.build(extended, tmp_path / "b")
    assert (tmp_path / "a" / "gameday.html").exists() and (tmp_path / "a" / "gameday.json").exists()
    assert 'href="gameday.html"' in (tmp_path / "a" / "index.html").read_text()
    assert not (tmp_path / "b" / "gameday.html").exists()
    assert "gameday.html" not in (tmp_path / "b" / "index.html").read_text()
    published = json.loads((tmp_path / "a" / "site.json").read_text())
    assert "gameday" not in published and "gamedayBacktest" not in published


def test_the_gameday_page_says_what_is_confirmed_and_what_is_a_guess(with_gameday) -> None:
    text = site.gameday_page(with_gameday)
    for needle in ("Where will College GameDay be?", "Confirmed", "The forecast", "SEC championship game",
                   "Who is likely to host", "Army", "How it works, and how well"):
        assert needle in text.replace("&ndash;", "-") or needle in text, needle
    assert "through Week 2 are known" in text          # derived from the announced list, not written in
    assert "42%" in text and "Some Stadium" in text
    assert "9 of them" in text                        # the games it left out are counted, not hidden
    assert "Week 1 is the current example" in text     # the announced week the model ranked sixth


def test_the_gameday_page_links_only_to_pages_that_exist(with_gameday) -> None:
    without_sim = {k: v for k, v in with_gameday.items() if k != "sim"}
    assert 'href="simulation.html"' not in site.gameday_page(without_sim)
    assert 'href="simulation.html"' in site.gameday_page(with_gameday)
