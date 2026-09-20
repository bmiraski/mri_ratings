"""Static site generator for mri.mira.ski.

Renders the whole site from site.json: rankings, a page per team, a page per
conference, methodology, and a weekly archive. No framework and no build step -
the output is plain HTML and one stylesheet, which is all a rankings site needs
and which will still work in five years.

Design follows the "Console" direction: dark by default, card-led rows, and the
disagreement between MRI 2.0 and MRI Classic surfaced rather than buried, since
that gap is the most interesting thing the system produces.

Team color is identity, never encoding. 138 brand colors were not chosen to be
told apart from each other and several pairs are indistinguishable under common
colorblindness, so every value is carried by position and text; color is a rule
beside the name.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..betting.board import MIN_EDGE_TO_SHOW as MIN_EDGE
from ..sim import season as sim_season
from . import common

# The brand crimson, from the logo. It is the site's accent in both themes at
# Ben's direction. Worth knowing what that costs: against the dark background it
# measures 2.3:1, where 3:1 is the floor for a graphic element and 4.5:1 for
# text, so the sparklines and conference bars are dimmer in dark mode than the
# blue they replaced. It reads as one brand with the logo, which is the trade.
BRAND = "#AB011B"
SERIES = BRAND
SERIES_LIGHT = BRAND

# The row sparklines are the exception. Every ranking row already carries its
# team's colour as a stripe down its left edge, and a crimson trace beside 138
# of those is one more colour competing rather than a brand doing brand work.
# The ink colour steps out of the way and follows the theme, so it is white on
# dark and near-black on light rather than literally white in both.
TREND = "var(--primary)"

# The custom domain. Written into the output as a CNAME file on every build:
# GitHub Pages puts that file in the repo when you set the domain in Settings,
# and since this generator rewrites the whole output directory it would
# otherwise be deleted on the next build and quietly take the domain down.
CUSTOM_DOMAIN = "mri.mira.ski"

# Absolute, because a link preview is fetched by a scraper that has no page to
# resolve a relative path against.
SITE_URL = f"https://{CUSTOM_DOMAIN}/"


# --------------------------------------------------------------------------
# per-sport chrome
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Chrome:
    """What differs between the two sports' sites.

    The pages themselves are shared. Keeping a second set of templates for
    basketball would mean every fix to a table or a chart had to be made twice,
    and within a month one of them would be stale. So the sports differ in this
    object and in their payloads, and nowhere else.

    ``home`` is the sport's directory under docs/. Football keeps the root so
    that every URL published so far still resolves; basketball is a subtree.
    """

    sport: str
    title: str          # the switch label
    home: str           # "" for football, "basketball/" for basketball
    noun: str           # "college football"
    field: str          # what "average" means: FBS, Division I
    venue: str          # home field / home court
    outsider: str       # tag for an opponent outside the rated field
    source_name: str
    source_url: str
    with_classic: bool
    history: str        # the era line in the footer


CHROME = {
    "football": Chrome(
        sport="football", title="Football", home="", noun="college football",
        field="FBS", venue="home field", outsider="FCS",
        source_name="CollegeFootballData", source_url="https://collegefootballdata.com",
        with_classic=True, history="run 2000&ndash;2019 and rebuilt for {season}",
    ),
    "basketball": Chrome(
        sport="basketball", title="Basketball", home="basketball/",
        noun="college basketball", field="Division I", venue="home court",
        outsider="non-D1",
        source_name="CollegeBasketballData", source_url="https://collegebasketballdata.com",
        # Fixed, not {season}: the chain starts at 2020-21 whatever season the
        # page happens to be showing.
        with_classic=False,
        history="run through 2019&ndash;20 and rebuilt from 2020&ndash;21",
    ),
}


def chrome_for(payload: dict) -> Chrome:
    return CHROME[payload.get("sport", "football")]


def season_text(payload: dict) -> str:
    """'2026' for football, '2025-26' for basketball."""
    return str(payload.get("seasonLabel") or payload["season"])


def period_text(payload: dict) -> str:
    """'Week 7', or 'Final' once a basketball season has finished."""
    return payload.get("periodLabel") or f"Week {payload['week']}"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def esc(value) -> str:
    return html.escape(str(value), quote=True)


# --------------------------------------------------------------------------
# small chart primitives
# --------------------------------------------------------------------------

def sparkline(values, width=58, height=22, color=TREND) -> str:
    """2px trajectory with a ringed end marker, per the mark specs.

    Suppressed below three points. With only two, every sparkline is a
    full-slope diagonal whatever the underlying change, because each is
    normalized to its own range - a team that slipped a tenth of a point looks
    identical to one that moved five. The movement chip already carries
    direction honestly; a shape earns its place once there is a shape.
    """
    values = [v for v in (values or []) if v is not None]
    if len(values) < 3:
        return f'<svg width="{width}" height="{height}" aria-hidden="true"></svg>'
    low, high = min(values), max(values)
    span = (high - low) or 1.0
    step = width / (len(values) - 1)
    points = [(i * step, height - 3 - ((v - low) / span) * (height - 6)) for i, v in enumerate(values)]
    path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points))
    ex, ey = points[-1]
    return (
        f'<svg class="spark" width="{width}" height="{height}" viewBox="0 0 {width} {height}" aria-hidden="true">'
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="3" fill="{color}" stroke="var(--surface)" stroke-width="2"/></svg>'
    )


def movement_chip(value) -> str:
    if not value:
        return '<span class="mv flat" title="No change">–</span>'
    arrow = "▲" if value > 0 else "▼"
    kind = "up" if value > 0 else "down"
    label = f"Up {value}" if value > 0 else f"Down {abs(value)}"
    return f'<span class="mv {kind}" title="{label}">{arrow}{abs(value)}</span>'


def identity_mark(team: dict, size: int = 20, depth: int = 0) -> str:
    """Logo when we have one, a color chip when we don't. Never color alone -
    the team name is always beside it.

    ``depth`` prefixes the cached logo path for pages in a subdirectory. Cached
    paths are site-relative rather than absolute so the output also works when
    opened straight off disk.
    """
    if team.get("logo"):
        source = team["logo"]
        if not source.startswith(("http://", "https://")):
            source = "../" * depth + source
        fallback = (
            "this.replaceWith(Object.assign(document.createElement('span'),"
            f"{{className:'chip',style:'background:{esc(team['color'])}'}}))"
        )
        return (
            f'<img class="logo" src="{esc(source)}" alt="" width="{size}" height="{size}"'
            f' loading="lazy" onerror="{esc(fallback)}">'
        )
    # The chip takes the same size as the logo it stands in for. Football rarely
    # reaches this branch, so the fixed 20px went unnoticed; basketball uses it
    # for all 365 teams, where a 20px mark beside a 46px heading reads as a bug.
    return (
        f'<span class="chip" style="background:{esc(team["color"])};'
        f'width:{size}px;height:{size}px;border-radius:{max(3, size // 7)}px"></span>'
    )


# --------------------------------------------------------------------------
# page chrome
# --------------------------------------------------------------------------

def page(title: str, body: str, payload: dict, *, depth: int = 0, description: str = "") -> str:
    chrome = chrome_for(payload)
    up = "../" * depth
    # Two levels of relative path, because there are now two roots. ``up`` walks
    # back to the sport's own index; ``docs`` walks back one further to the site
    # root, which is where the other sport lives.
    docs = up + ("../" if chrome.home else "")

    # The betting page exists only once a backtest has been run, so the nav must
    # not promise it unconditionally - a dead link in the header on every page
    # is a worse failure than a missing section.
    betting_link = (
        f'\n      <a href="{up}betting.html">Betting</a>'
        if payload.get("betting") and payload.get("board") else ""
    )
    seasons_link = f'\n      <a href="{up}seasons.html">Seasons</a>' if payload.get("seasons") else ""
    # Football only, and only when the build produced them: the same rule as the
    # betting link, for the same reason.
    slate_link = f'\n      <a href="{up}slate.html">Slate</a>' if payload.get("slate") else ""
    sim_link = f'\n      <a href="{up}simulation.html">Simulation</a>' if payload.get("sim") else ""
    if payload.get("gameday"):
        sim_link += f'\n      <a href="{up}gameday.html">GameDay</a>'
    # The switch offers only sports this build actually published. The same rule
    # as the betting link above: a header link to a directory that does not exist
    # is a dead link on every page of the site, which is worse than no switch.
    published = payload.get("sports") or [chrome.sport]
    current = ' aria-current="page" class="on"'
    switch = "" if len(published) < 2 else '<div class="sports">{}</div>'.format("".join(
        '<a href="{}{}index.html"{}>{}</a>'.format(
            docs, CHROME[name].home,
            current if name == chrome.sport else "", CHROME[name].title
        )
        for name in published if name in CHROME
    ))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{esc(description or f'A computer rating system for {chrome.noun}.')}">
<link rel="stylesheet" href="{docs}styles.css">
<link rel="icon" href="{docs}favicon.ico" sizes="any">
<link rel="icon" type="image/png" href="{docs}assets/icon-192.png" sizes="192x192">
<link rel="apple-touch-icon" href="{docs}assets/apple-touch-icon.png">
<meta name="theme-color" content="{BRAND}">
<meta property="og:title" content="{esc(title)}">
<meta property="og:description" content="{esc(description or f'A computer rating system for {chrome.noun}.')}">
<meta property="og:type" content="website">
<meta property="og:image" content="{SITE_URL}assets/mri-card.png">
<meta name="twitter:card" content="summary_large_image">
</head>
<body>
<header class="site">
  <div class="wrap bar">
    <a class="mark" href="{up}index.html" aria-label="The MRI, home">
      <picture>
        <source media="(prefers-color-scheme: light)" srcset="{docs}assets/mri-lockup-ink.png">
        <img src="{docs}assets/mri-lockup.png" alt="The MRI" width="129" height="38">
      </picture>
    </a>
    {switch}
    <nav>
      <a href="{up}index.html">Rankings</a>
      <a href="{up}conferences.html">Conferences</a>{slate_link}{sim_link}
{betting_link}
      <a href="{up}archive.html">Archive</a>{seasons_link}
      <a href="{up}method.html">Method</a>
    </nav>
    <div class="stamp">{esc(season_text(payload))} &middot; {esc(period_text(payload))}</div>
  </div>
</header>
<main class="wrap">{body}</main>
<footer class="site">
  <div class="wrap">
    <p>MRI &mdash; a computer rating system for {chrome.noun},
    {chrome.history.format(season=season_text(payload))}. Game data from
    <a href="{chrome.source_url}">{chrome.source_name}</a>.</p>
    <p class="muted">Ratings last changed {esc(payload['generated'][:16].replace('T', ' '))} UTC
    &middot; {payload['gamesRated']} games rated &middot; {chrome.venue} {payload['homeField']:.1f} pts</p>
  </div>
</footer>
</body>
</html>"""


# --------------------------------------------------------------------------
# rankings
# --------------------------------------------------------------------------

def rankings_page(payload: dict) -> str:
    chrome = chrome_for(payload)
    teams = payload["teams"]
    top = teams[0]

    rows = []
    for team in teams:
        gap = (team["classicRank"] - team["rank"]) if team.get("classicRank") else None
        classic_note = (
            f'<span class="sub-item" title="MRI Classic rank">C #{team["classicRank"]}</span>'
            if team.get("classicRank") else ""
        )
        rows.append(f"""
      <li class="row" style="--team:{esc(team['color'])}"
          data-conf="{esc(team['conference'])}"
          data-power="{team['power']}" data-resume="{team['resume'] or 0}"
          data-classic="{team.get('classicRank') or 999}" data-rank="{team['rank']}">
        <span class="rk">{team['rank']}</span>
        <span class="mid">
          <span class="top">{identity_mark(team)}<a class="nm" href="team/{slug(team['team'])}.html">{esc(team['team'])}</a>{movement_chip(team['movement'])}</span>
          <span class="sub">{esc(team['conference'])} <span class="sub-item">{team['wins']}&ndash;{team['losses']}</span> {classic_note}{_title_chip(team, payload)}</span>
        </span>
        <span class="sp">{sparkline(team['trajectory'])}</span>
        <span class="vals"><span class="pw">{team['power']:+.1f}</span><span class="rs">r&eacute;s {team['resume']:+.2f}</span></span>
      </li>""")

    disagreements = sorted(
        [t for t in teams if t.get("classicRank")],
        key=lambda t: -abs(t["classicRank"] - t["rank"]),
    )[:5]
    gap_rows = "".join(
        f"""<li><a class="gnm" href="team/{slug(t['team'])}.html">{esc(t['team'])}</a>
        <span class="gv">#{t['rank']} <em>vs</em> #{t['classicRank']}</span></li>"""
        for t in disagreements
    )
    # Basketball publishes one rating, so the comparison panel and the sort that
    # depends on it are dropped rather than rendered empty.
    classic_panel = f"""
      <section class="panel">
        <p class="ptitle">Where Classic disagrees</p>
        <ul class="list">{gap_rows}</ul>
        <p class="note">MRI 2.0 rank versus the frozen 2018 formula. Classic has no
        preseason prior and weights raw win percentage heavily, so in September an
        unbeaten team that has played nobody rides high.</p>
      </section>
""" if chrome.with_classic and gap_rows else ""
    classic_option = (
        '\n            <option value="classic">By MRI Classic</option>'
        if chrome.with_classic else ""
    )

    # Where the panel Classic would have filled goes to the argument the two
    # published numbers are having with each other: who the model rates far above
    # what their record has earned, and who has earned more than they look.
    gap_panel = "" if chrome.with_classic else _resume_gap_panel(teams)

    movers = sorted(teams, key=lambda t: -abs(t["movement"]))[:5]
    mover_rows = "".join(
        f"""<li><a class="gnm" href="team/{slug(t['team'])}.html">{esc(t['team'])}</a>
        <span class="gv">{movement_chip(t['movement'])} to #{t['rank']}</span></li>"""
        for t in movers if t["movement"]
    ) or '<li class="empty">No movement &mdash; the ranking is unchanged from last week.</li>'
    # In a finished season the last week's movement is a ±1 shuffle at the bottom
    # of the table. Presenting that as news would be filling a box for its own
    # sake, so the panel goes away once the season is over.
    movers_panel = "" if payload.get("finished") else f"""
      <section class="panel">
        <p class="ptitle">Biggest movers</p>
        <ul class="list">{mover_rows}</ul>
      </section>
"""

    ranked = [c for c in payload["conferences"] if c.get("ranked", True)]
    conferences = "".join(_conference_bar(c, ranked) for c in ranked)
    options = "".join(
        f'<option value="{esc(c["conference"])}">{esc(c["conference"])}</option>'
        for c in payload["conferences"]
    )

    body = f"""
  <div class="grid">
    <section class="panel">
      <div class="panelhead">
        <h1 class="ptitle">Power rating</h1>
        <div class="controls">
          <label class="sr">Conference<select id="conf"><option value="">All conferences</option>{options}</select></label>
          <label class="sr">Sort<select id="sort">
            <option value="rank">By power</option>
            <option value="resume">By r&eacute;sum&eacute;</option>{classic_option}
          </select></label>
        </div>
      </div>
      <p class="hint">Points against an average {chrome.field} team &mdash; the gap between two
      rows is a predicted spread. <strong>R&eacute;sum&eacute;</strong> is wins above what an
      average team would manage against the same schedule.</p>
      <ul id="list">{''.join(rows)}</ul>
      <p class="empty" id="none" hidden>No teams match that filter.</p>
    </section>

    <aside>
      <section class="panel">
        <p class="ptitle">Number one</p>
        <div class="hero">
          <div class="heronum">{top['power']:+.0f}</div>
          <div>
            <div class="heronm"><a href="team/{slug(top['team'])}.html">{esc(top['team'])}</a></div>
            <div class="herosub">{top['wins']}&ndash;{top['losses']} &middot; r&eacute;sum&eacute; #{top['resumeRank']}</div>
          </div>
        </div>
      </section>

{classic_panel}{gap_panel}{movers_panel}
      <section class="panel">
        <p class="ptitle">Conference strength</p>
        <ul class="conf">{conferences}</ul>
        <p class="note">A top-weighted average of member ratings &mdash; each
        next-best team counts 10% less than the one above it, so the measure is
        about quality at the top without ignoring what is underneath, and does
        not reward a conference merely for being small.</p>
      </section>
    </aside>
  </div>
<script>
(function () {{
  var list = document.getElementById('list');
  var items = Array.prototype.slice.call(list.children);
  var conf = document.getElementById('conf');
  var sort = document.getElementById('sort');
  var none = document.getElementById('none');
  function apply() {{
    var want = conf.value, key = sort.value, shown = 0;
    var ordered = items.slice().sort(function (a, b) {{
      if (key === 'resume') return b.dataset.resume - a.dataset.resume;
      if (key === 'classic') return a.dataset.classic - b.dataset.classic;
      return a.dataset.rank - b.dataset.rank;
    }});
    ordered.forEach(function (el) {{
      var ok = !want || el.dataset.conf === want;
      el.hidden = !ok;
      if (ok) shown++;
      list.appendChild(el);
    }});
    none.hidden = shown > 0;
  }}
  conf.addEventListener('change', apply);
  sort.addEventListener('change', apply);
}})();
</script>"""
    return page(f"MRI — {season_text(payload)} {chrome.noun} rankings", body, payload,
                description=f"Opponent-adjusted {chrome.noun} ratings, updated weekly.")


def _resume_gap_panel(teams: list[dict]) -> str:
    """The widest disagreements between what a team is and what it has earned.

    Power and Résumé are published as separate numbers precisely because they
    answer different questions, and the places they disagree most are the whole
    reason for keeping them apart: a team the model rates twenty spots above its
    record, or one whose record flatters it. Only teams inside the top hundred
    are eligible - a rank gap of forty means nothing at #300, where the ratings
    are a tenth of a point apart.
    """
    # One direction only. A list headed "better than their record" that also
    # contains teams whose record flatters them is a heading that lies about half
    # its own rows, so this keeps only the teams the model rates above what they
    # have earned, and the note explains which way to read it.
    eligible = [
        t for t in teams
        if t.get("resumeRank") and t["rank"] <= 100 and t["resumeRank"] > t["rank"]
    ]
    ranked = sorted(eligible, key=lambda t: -(t["resumeRank"] - t["rank"]))[:5]
    if not ranked:
        return ""
    rows = "".join(
        f"""<li><a class="gnm" href="team/{slug(t['team'])}.html">{esc(t['team'])}</a>
        <span class="gv">#{t['rank']} <em>vs</em> #{t['resumeRank']}</span></li>"""
        for t in ranked
    )
    return f"""
      <section class="panel">
        <p class="ptitle">Better than their record</p>
        <ul class="list">{rows}</ul>
        <p class="note">Power rank, then r&eacute;sum&eacute; rank. These teams have played
        better than their wins show &mdash; close losses to good teams, or a schedule
        that gave them nothing to bank. It is the gap the two published numbers exist
        to make visible.</p>
      </section>
"""


def _conference_bar(conference: dict, among: list[dict]) -> str:
    value = conference.get("strength", conference["mean"])
    widest = max(abs(c.get("strength", c["mean"])) for c in among) or 1
    width = abs(value) / widest * 100
    negative = value < 0
    return (
        f'<li><a class="confnm" href="conference/{slug(conference["conference"])}.html">'
        f'{esc(conference["conference"])}</a>'
        f'<span class="track"><span class="confbar{" neg" if negative else ""}" '
        f'style="width:{width:.0f}%"></span></span>'
        f'<span class="confv">{value:+.1f}</span></li>'
    )


# --------------------------------------------------------------------------
# team pages
# --------------------------------------------------------------------------

def _trend_stat(team: dict) -> str:
    """A trend tile once there is a trend; until then, say where the team moved.

    An empty box that promises a chart and shows nothing is worse than a
    different fact, and in the season's first weeks there is no shape to plot.
    """
    points = [v for v in (team.get("trajectory") or []) if v is not None]
    if len(points) >= 3:
        return (
            '<div class="stat"><span class="statl">Trend</span>'
            f'<span class="statv sparkbig">{sparkline(points, 90, 34)}</span>'
            '<span class="statn">rating by week</span></div>'
        )
    previous = team.get("previousRank")
    if previous:
        change = previous - team["rank"]
        word = "up" if change > 0 else ("down" if change < 0 else "unchanged")
        value = f"{'+' if change > 0 else ''}{change}" if change else "&mdash;"
        return (
            '<div class="stat"><span class="statl">Movement</span>'
            f'<span class="statv">{value}</span>'
            f'<span class="statn">{word} from #{previous} last week</span></div>'
        )
    return (
        '<div class="stat"><span class="statl">Trend</span>'
        '<span class="statv">&mdash;</span>'
        '<span class="statn">available from week 3</span></div>'
    )


def team_season_page(team: dict, entry: dict, rows: list[dict], payload: dict) -> str:
    """One team, one past season, every game.

    Linkable on its own, which is the reason it is a page rather than an
    accordion on the team page: a single season is a thing people send each
    other.
    """
    chrome = chrome_for(payload)
    lookup = {t["team"] for t in payload["teams"]}
    has_expected = any(r["expected"] is not None for r in rows)
    dated = any(r["when"] for r in rows)

    def opponent(row: dict) -> str:
        if row["pooled"]:
            return f'<span class="fcs">non-{chrome.field} opponent</span>'
        name = row["opponent"]
        if name in lookup:
            return f'<a href="../{slug(name)}.html">{esc(name)}</a>'
        return esc(name)

    body_rows = "".join(f"""
      <tr>
        {f'<td class="wk">{esc(r["when"][5:]) if r["when"] else "&ndash;"}</td>' if dated else ''}
        <td class="site">{'at' if r['site'] == 'at' else ('vs' if r['site'] == 'vs' else 'N')}</td>
        <td class="opp">{opponent(r)}</td>
        <td class="res"><span class="{'w' if r['won'] else 'l'}">{'W' if r['won'] else 'L'}</span> {r['scored']}&ndash;{r['allowed']}</td>
        {f'<td class="num">{r["expected"]:+.1f}</td><td class="num perf {"over" if r["performance"] > 0 else "under"}">{r["performance"]:+.1f}</td>' if has_expected and r["expected"] is not None else ('<td class="num muted">&ndash;</td><td class="num muted">&ndash;</td>' if has_expected else '')}
      </tr>""" for r in rows)

    wins = sum(1 for r in rows if r["won"])
    note = ""
    if not has_expected:
        note = (f"""<p class="note">No expected-margin column for this season:
        MRI 2.0 rates {chrome.noun} from 2020 onward, and these games were rated by
        the original formula, which scores a result rather than predicting one.</p>""")
    if not dated:
        note += ("""<p class="note">The workbook for this season carries no dates, so
        games are listed in the order it recorded them rather than chronologically.</p>""")
    elif any(r["when"] is None for r in rows):
        missing = sum(1 for r in rows if r["when"] is None)
        note += (f"""<p class="note">{missing} of these games carry no date in the
        source workbook; they sort to the top rather than into the season.</p>""")

    header = (('<th>Date</th>' if dated else '') + '<th></th><th>Opponent</th><th>Result</th>'
              + ('<th class="num">Expected</th><th class="num">Perf</th>' if has_expected else ''))

    body = f"""
  <h1>{esc(team['team'])} &middot; {esc(entry['label'])}</h1>
  <p class="teamsub">{wins}&ndash;{len(rows) - wins} &middot; #{entry['rank']} of {entry['of']}
  &middot; {esc(entry['ratingName'])} {entry['rating']:,.2f} &middot; {esc(entry['system'])}</p>
  <div class="tablewrap"><table>
    <thead><tr>{header}</tr></thead>
    <tbody>{body_rows}</tbody>
  </table></div>
  {note}
  <p class="hint"><a href="../{slug(team['team'])}.html">{esc(team['team'])}</a>
  &middot; <a href="../../season/{entry['season']}.html">{esc(entry['label'])} rankings</a></p>"""
    return page(f"{team['team']} {entry['label']} — MRI {chrome.noun}", body, payload, depth=2,
                description=f"Every {entry['label']} game for {team['team']}, "
                            "with the margin the ratings implied.")


def titles_of(team: str, payload: dict) -> list[dict]:
    """The seasons a team finished the year rated first.

    Finished, not led: the archive only ever holds completed seasons, so a team
    sitting top in week three of a live season is nowhere near this list. That
    is the whole point of a banner - it has to be earned all the way through.
    """
    rows = (payload.get("history") or {}).get(team) or []
    return [r for r in rows if r["rank"] == 1]


TROPHY = (
    '<svg class="trophy" viewBox="0 0 24 24" width="{size}" height="{size}" aria-hidden="true">'
    '<path fill="currentColor" d="M18 4V2H6v2H2v3a5 5 0 0 0 4.1 4.9A6 6 0 0 0 11 16.9V19H7v3h10v-3h-4'
    'v-2.1a6 6 0 0 0 4.9-5A5 5 0 0 0 22 7V4h-4zM4 7V6h2v3.8A3 3 0 0 1 4 7zm16 0a3 3 0 0 1-2 2.8V6h2v1z"/>'
    "</svg>"
)


def _title_chip(team: dict, payload: dict) -> str:
    """A quiet marker in the rankings list. Sparse by nature - fourteen teams in
    football, nine in basketball - so it marks a row rather than decorating it."""
    won = titles_of(team["team"], payload)
    if not won:
        return ""
    label = f"{len(won)} MRI title{'' if len(won) == 1 else 's'}"
    return (f'<span class="sub-item wonchip" title="{esc(label)}">'
            f'{TROPHY.format(size=11)}{len(won)}</span>')


def _titles_banner(team: dict, payload: dict) -> str:
    """Flags fly forever."""
    won = titles_of(team["team"], payload)
    if not won:
        return ""
    chrome = chrome_for(payload)
    years = ", ".join(
        f'<a href="{slug(team["team"])}/{r["season"]}.html">{esc(r["label"])}</a>'
        if (payload.get("gamelogs") or {}).get(r["season"], {}).get(team["team"])
        else esc(r["label"])
        for r in sorted(won, key=lambda r: r["season"])
    )
    plural = "" if len(won) == 1 else "s"
    return f"""
    <div class="titles">
      {TROPHY.format(size=26)}
      <div>
        <span class="titlesl">MRI Champion &middot; {len(won)} {chrome.noun.split()[-1]} title{plural}</span>
        <span class="titlesy">{years}</span>
      </div>
    </div>"""


def _rank_chart(rows: list[dict]) -> str:
    """Rank by season, drawn on the one axis the two ratings share.

    The ratings themselves cannot be plotted together and this is the reason
    this chart exists at all: Classic counts cumulative points where a great
    season is 150, MRI 2.0 counts points against an average team where a great
    season is +35, and a line joining them would be a lie with a trend in it.
    Rank is different - both formulas rank within the same field - so the line
    is honest even across the seam, which is drawn in rather than hidden.

    Inverted, because first belongs at the top. The axis runs to the largest
    field the team ever played in, so the field growing from 117 teams to 138
    is visible rather than silently rescaling every year.
    """
    if len(rows) < 3:
        return ""

    points = sorted(rows, key=lambda r: r["season"])
    seasons = [r["season"] for r in points]
    # The axis runs to this team's own worst finish, rounded up, not to the size
    # of the field. Against 365 basketball teams a full-field axis makes every
    # line a flat squiggle in the top eighth of an empty box; against 138 it
    # still wastes half the plot for anyone decent. The floor of 25 stops a
    # permanently-elite team's one-place wobbles from being drawn as drama - the
    # same reason the sparklines refuse to draw two points.
    deepest = max(r["rank"] for r in points)
    worst = next((step for step in (25, 50, 75, 100, 150, 200, 250, 300, 400)
                  if step >= deepest), deepest)
    lo, hi = min(seasons), max(seasons)
    span = (hi - lo) or 1

    W, H = 720, 190
    L, R, T, B = 34, 12, 14, 26
    px = lambda s: L + (s - lo) / span * (W - L - R)
    py = lambda rank: T + (rank - 1) / max(worst - 1, 1) * (H - T - B)

    path = " ".join(
        f"{'M' if i == 0 else 'L'}{px(r['season']):.1f},{py(r['rank']):.1f}"
        for i, r in enumerate(points)
    )

    # The seam between the two ratings, where there is one.
    seam = ""
    for earlier, later in zip(points, points[1:]):
        if earlier["system"] != later["system"]:
            x = (px(earlier["season"]) + px(later["season"])) / 2
            # Labels sit on the baseline, not at the top where the line lives.
            seam = (
                f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{H - B}" '
                f'stroke="var(--axis)" stroke-width="1" stroke-dasharray="3 3"/>'
                f'<text x="{x - 5:.1f}" y="{H - B - 5}" text-anchor="end" class="ct">Classic</text>'
                f'<text x="{x + 5:.1f}" y="{H - B - 5}" text-anchor="start" class="ct">MRI 2.0</text>'
            )
            break

    # A top-25 reference line, which is the threshold anyone reading this cares
    # about, drawn only when the team's axis actually reaches it.
    top25 = ""
    if worst > 30:
        y = py(25)
        top25 = (f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" stroke="var(--grid)" '
                 f'stroke-width="1"/><text x="{L - 5}" y="{y + 3:.1f}" text-anchor="end" class="ct">25</text>')

    dots = "".join(
        f'<circle cx="{px(r["season"]):.1f}" cy="{py(r["rank"]):.1f}" '
        f'r="{4 if r["rank"] == 1 else 2.6}" '
        f'class="{"champ" if r["rank"] == 1 else "pt"}">'
        f'<title>{esc(r["label"])}: #{r["rank"]} of {r["of"]} ({r["wins"]}-{r["losses"]})</title>'
        f"</circle>"
        for r in points
    )

    first, last = points[0], points[-1]
    return f"""
      <div class="rankchart">
        <svg viewBox="0 0 {W} {H}" role="img"
             aria-label="Finishing rank by season, best at the top.">
          {top25}{seam}
          <line x1="{L}" y1="{T}" x2="{L}" y2="{H - B}" stroke="var(--axis)" stroke-width="1"/>
          <text x="{L - 5}" y="{T + 4}" text-anchor="end" class="ct">1</text>
          <text x="{L - 5}" y="{H - B}" text-anchor="end" class="ct">{worst}</text>
          <path d="{path}" fill="none" stroke="var(--series)" stroke-width="2"
                stroke-linejoin="round" stroke-linecap="round"/>
          {dots}
          <text x="{px(first['season']):.1f}" y="{H - 8}" text-anchor="start" class="ct">{esc(first['label'])}</text>
          <text x="{px(last['season']):.1f}" y="{H - 8}" text-anchor="end" class="ct">{esc(last['label'])}</text>
        </svg>
        <p class="note">Finishing rank, best at the top, on an axis running to
        {worst} &mdash; this team's own range, not the size of the field. Each row
        below says what its rank was out of. Hover a point for the season.</p>
      </div>"""


def _history_section(team: dict, payload: dict) -> str:
    """Every season this team has been rated, newest first.

    Rank is the column that travels. Both formulas rank within the same field,
    so #4 in 2011 and #4 in 2023 mean approximately the same thing, while 143.50
    and +34.60 do not - one counts cumulative points and the other counts points
    against an average team. So the rating is shown with the name of the system
    beside it and never in a chart that would imply a line between them.

    The field grew too, from 117 teams to 138, which is why each row says what
    it was a rank of.
    """
    rows = (payload.get("history") or {}).get(team["team"]) or []
    if len(rows) < 2:
        return ""

    chrome = chrome_for(payload)
    logs = payload.get("gamelogs") or {}
    have_log = {r["season"] for r in rows
                if (logs.get(r["season"]) or {}).get(team["team"])}

    def season_link(r: dict) -> str:
        if r["season"] in have_log:
            return f'<a href="{slug(team["team"])}/{r["season"]}.html">{esc(r["label"])}</a>'
        return f'<a href="../season/{r["season"]}.html">{esc(r["label"])}</a>'

    body = "".join(f"""
      <tr{' class="wonit"' if r['rank'] == 1 else ''}>
        <td class="rk">{season_link(r)}</td>
        <td class="num">{TROPHY.format(size=13) if r['rank'] == 1 else ''}<strong>{r['rank']}</strong><span class="of"> of {r['of']}</span></td>
        <td class="rec">{r['wins']}&ndash;{r['losses']}</td>
        <td class="num">{r['rating']:,.2f}</td>
        <td class="muted sysname">{esc(r['ratingName'])}</td>
      </tr>""" for r in rows)

    best = min(rows, key=lambda r: r["rank"])
    systems = {r["system"] for r in rows}
    caveat = (
        " Two ratings appear here and their numbers are not comparable; the rank is."
        if len(systems) > 1 else ""
    )
    return f"""
    <section>
      <h2>Season by season</h2>
      <p class="hint">Best finish: <strong>#{best['rank']}</strong> in {esc(best['label'])}.
      {len(rows)} rated seasons.{caveat}
      {"Season links open that year's game log." if have_log else ""}</p>
      {_rank_chart(rows)}
      <div class="tablewrap"><table>
        <thead><tr><th>Season</th><th class="num">Rank</th><th>Rec</th>
        <th class="num">Rating</th><th>&nbsp;</th></tr></thead>
        <tbody>{body}</tbody>
      </table></div>
    </section>"""


def team_page(team: dict, payload: dict) -> str:
    chrome = chrome_for(payload)
    detail = payload["details"].get(team["team"], {"played": [], "upcoming": []})
    lookup = {t["team"]: t for t in payload["teams"]}

    def opponent_link(name: str) -> str:
        if name in lookup:
            return f'<a href="{slug(name)}.html">{esc(name)}</a>'
        return f'{esc(name)} <span class="fcs">{chrome.outsider}</span>'

    played = "".join(f"""
      <tr>
        <td class="wk">{g['week']}</td>
        <td class="site">{'at' if g['site'] == 'at' else ('vs' if g['site'] == 'vs' else 'N')}</td>
        <td class="opp">{opponent_link(g['opponent'])}</td>
        <td class="res"><span class="{'w' if g['won'] else 'l'}">{'W' if g['won'] else 'L'}</span> {g['scored']}&ndash;{g['allowed']}</td>
        <td class="num">{g['expected']:+.1f}</td>
        <td class="num perf {'over' if g['performance'] > 0 else 'under'}">{g['performance']:+.1f}</td>
      </tr>""" for g in detail["played"]) or '<tr><td colspan="6" class="empty">No games played yet.</td></tr>'

    upcoming = "".join(f"""
      <tr>
        <td class="wk">{g['week']}</td>
        <td class="site">{'at' if g['site'] == 'at' else ('vs' if g['site'] == 'vs' else 'N')}</td>
        <td class="opp">{opponent_link(g['opponent'])}</td>
        <td class="num">{g['expected']:+.1f}</td>
        <td class="num">{g['winProbability']:.0%}</td>
      </tr>""" for g in detail["upcoming"][:8]) or '<tr><td colspan="5" class="empty">Season complete.</td></tr>'

    best = detail.get("bestWin")
    worst = detail.get("worstLoss")
    highlights = []
    if best:
        highlights.append(f'<div class="hl"><span class="hll">Best win</span><span class="hlv">{esc(best["opponent"])} ({best["scored"]}&ndash;{best["allowed"]})</span></div>')
    if worst:
        highlights.append(f'<div class="hl"><span class="hll">Worst loss</span><span class="hlv">{esc(worst["opponent"])} ({worst["scored"]}&ndash;{worst["allowed"]})</span></div>')
    if detail.get("playedDifficulty") is not None:
        highlights.append(f'<div class="hl"><span class="hll">Schedule faced</span><span class="hlv">{detail["playedDifficulty"]:+.1f} avg opponent</span></div>')
    if detail.get("remainingDifficulty") is not None:
        highlights.append(f'<div class="hl"><span class="hll">Schedule ahead</span><span class="hlv">{detail["remainingDifficulty"]:+.1f} avg opponent</span></div>')

    roster = team.get("roster") or {}
    if roster.get("mode"):
        # Basketball: what is known about the roster, and how the preseason rating used it.
        if roster["mode"] == "roster" and roster.get("returning") is not None:
            text = f'{roster["returning"]:.0%} of last season\'s win shares &middot; #{roster["returningRank"]} of {roster["returningOf"]}'
            if roster["returning"] < 0.30:
                text += " &middot; counted against its preseason rating"
            highlights.append(f'<div class="hl" title="The share of last season\'s win shares produced by players still on the roster."><span class="hll">Returning production</span><span class="hlv">{text}</span></div>')
            newcomers = roster["incoming"]
            highlights.append(f'<div class="hl" title="What this roster\'s newcomers produced last season at other schools: the transfer portal."><span class="hll">Incoming production</span><span class="hlv">{newcomers:.1f} win shares from other schools\' players</span></div>')
        elif roster.get("veteranMinutes") is not None:
            highlights.append(f'<div class="hl" title="Rosters for this season are not posted yet, so the preseason rating leans on who was likely to leave: players in a fourth season or later, and players who were drafted."><span class="hll">Last season\'s minutes</span><span class="hlv">{roster["veteranMinutes"]:.0%} from fourth-year-plus players &middot; {roster["draftMinutes"]:.0%} from players drafted</span></div>')
        if roster.get("freshman"):
            highlights.append(f'<div class="hl" title="How highly the incoming recruits were rated, summed over the class."><span class="hll">Recruiting class</span><span class="hlv">#{roster["freshmanRank"]} in the country</span></div>')
    elif roster.get("talent") is not None:
        text = f'#{roster["talentRank"]} of {roster["talentOf"]}'
        if "talentGap" in roster:
            gap = roster["talentGap"]
            verdict = ("in line with it" if abs(gap) < 5
                       else "beating it" if gap > 0 else "short of it")
            text += (f' &middot; worth {roster["talentImplied"]:+.1f}, rated {team["power"]:+.1f}: {verdict}')
        highlights.append(f'<div class="hl" title="The 247Sports talent composite: roster quality built up over recruiting classes. It explains about a third of the variation in ratings, so a gap of a few points means little."><span class="hll">Roster talent</span><span class="hlv">{text}</span></div>')
    elif roster.get("talentNote"):
        highlights.append('<div class="hl"><span class="hll">Roster talent</span><span class="hlv">not comparable &mdash; recruiting rankings do not measure the service academies</span></div>')
    if not roster.get("mode") and roster.get("returning") is not None:
        text = f'{roster["returning"]:.0%} of last year\'s &middot; #{roster["returningRank"]} of {roster["returningOf"]}'
        if roster["returning"] < 0.25:
            text += ' &middot; counted against its preseason rating'
        highlights.append(f'<div class="hl" title="The share of last season\'s production, by predicted points added, still on the roster."><span class="hll">Returning production</span><span class="hlv">{text}</span></div>')

    body = f"""
  <article class="teampage" style="--team:{esc(team['color'])}">
    {_titles_banner(team, payload)}
    <div class="teamhead">
      {identity_mark(team, 46, depth=1)}
      <div>
        <h1>{esc(team['team'])}</h1>
        <p class="teamsub">{esc(team['conference'])} &middot; {team['wins']}&ndash;{team['losses']}
        {f"&middot; MRI Classic #{team['classicRank']}" if team.get('classicRank') else ""}</p>
      </div>
    </div>

    <div class="stats">
      <div class="stat"><span class="statl">Power</span><span class="statv">{team['power']:+.1f}</span><span class="statn">#{team['rank']} of {len(payload['teams'])}</span></div>
      <div class="stat"><span class="statl">R&eacute;sum&eacute;</span><span class="statv">{team['resume']:+.2f}</span><span class="statn">#{team['resumeRank']} &middot; wins above average</span></div>
      {_trend_stat(team)}
    </div>

    <div class="highlights">{''.join(highlights)}</div>

    <section>
      <h2>Results</h2>
      <p class="hint"><strong>Expected</strong> is the margin the model's ratings imply for that
      matchup and site. <strong>Perf</strong> is how far the actual margin beat it &mdash; the
      number that moves a rating, rather than the win or loss alone.</p>
      <div class="tablewrap"><table>
        <thead><tr><th>Wk</th><th></th><th>Opponent</th><th>Result</th>
        <th class="num">Expected</th><th class="num">Perf</th></tr></thead>
        <tbody>{played}</tbody>
      </table></div>
    </section>

    <section>
      <h2>Remaining schedule</h2>
      <div class="tablewrap"><table>
        <thead><tr><th>Wk</th><th></th><th>Opponent</th>
        <th class="num">Projected</th><th class="num">Win prob</th></tr></thead>
        <tbody>{upcoming}</tbody>
      </table></div>
    </section>
{_history_section(team, payload)}
  </article>"""
    return page(f"{team['team']} — MRI {season_text(payload)}", body, payload, depth=1,
                description=f"{team['team']} MRI rating, schedule and game-by-game performance.")


# --------------------------------------------------------------------------
# conference pages
# --------------------------------------------------------------------------

def conference_page(name: str, payload: dict) -> str:
    chrome = chrome_for(payload)
    members = [t for t in payload["teams"] if t["conference"] == name]
    classic_cell = (
        lambda t: f"""
        <td class="num muted">{t.get('classicRank') or '&ndash;'}</td>"""
    ) if chrome.with_classic else (lambda t: "")
    classic_head = '<th class="num">Classic</th>' if chrome.with_classic else ""
    rows = "".join(f"""
      <tr style="--team:{esc(t['color'])}">
        <td class="rk">{t['rank']}</td>
        <td class="tm"><span class="rule"></span>{identity_mark(t, 18, depth=1)}<a href="../team/{slug(t['team'])}.html">{esc(t['team'])}</a></td>
        <td class="rec">{t['wins']}&ndash;{t['losses']}</td>
        <td class="num">{t['power']:+.1f}</td>
        <td class="num">{t['resume']:+.2f}</td>{classic_cell(t)}
      </tr>""" for t in members)

    entry = next((c for c in payload["conferences"] if c["conference"] == name), {})
    mean = sum(t["power"] for t in members) / len(members) if members else 0
    strength = entry.get("strength", mean)
    rank = next(
        (i + 1 for i, c in enumerate(payload["conferences"]) if c["conference"] == name), None
    )
    body = f"""
  <h1>{esc(name)}</h1>
  <p class="teamsub">{len(members)} teams &middot; strength {strength:+.1f}
  {f"&middot; #{rank} of {len(payload['conferences'])}" if rank else ""}
  &middot; mean power {mean:+.1f}</p>
  <div class="tablewrap"><table class="conftable">
    <thead><tr><th>#</th><th>Team</th><th>Rec</th><th class="num">Power</th>
    <th class="num">R&eacute;sum&eacute;</th>{classic_head}</tr></thead>
    <tbody>{rows}</tbody>
  </table></div>"""
    return page(f"{name} — MRI {season_text(payload)}", body, payload, depth=1,
                description=f"{name} teams ranked by MRI power rating.")


def conferences_index(payload: dict) -> str:
    def card(c):
        return f"""
    <a class="confcard" href="conference/{slug(c['conference'])}.html">
      <span class="ccname">{esc(c['conference'])}</span>
      <span class="ccmean">{c.get('strength', c['mean']):+.1f}</span>
      <span class="ccmeta">{int(c['count'])} teams &middot; best {c['max']:+.1f}
      &middot; mean {c['mean']:+.1f}</span>
    </a>"""

    ranked = [c for c in payload["conferences"] if c.get("ranked", True)]
    unranked = [c for c in payload["conferences"] if not c.get("ranked", True)]
    cards = "".join(card(c) for c in ranked)
    loose = f"""
  <h2>Not conferences</h2>
  <p class="hint">Fewer than {common.MIN_RANKED} teams, so these are listed rather than
  ranked. Independents are the absence of a conference, and a two-team average of them
  is not a statement about any league.</p>
  <div class="confgrid">{''.join(card(c) for c in unranked)}</div>""" if unranked else ""
    body = f"""
  <h1>Conferences</h1>
  <p class="hint">Ranked by a top-weighted average of member ratings: each next-best
  team in a conference counts 10% less than the one above it. That measures quality
  at the top while still counting depth, and unlike a plain mean it does not punish
  an eighteen-team league for its tail. The plain mean is shown beside it.</p>
  <div class="confgrid">{cards}</div>{loose}"""
    return page(f"Conferences — MRI {season_text(payload)}", body, payload)


# --------------------------------------------------------------------------
# archive & methodology
# --------------------------------------------------------------------------

def archive_page(payload: dict) -> str:
    weeks = max(len(t["rankHistory"]) for t in payload["teams"])
    # Basketball's first rated week is not week one - a ranking of 365 teams off
    # three days of November play is noise, so those weeks are never published
    # and the column headings have to say which weeks these actually are.
    labels = payload.get("weeks") or list(range(1, weeks + 1))
    header = "".join(f"<th>Wk {labels[w] if w < len(labels) else w + 1}</th>" for w in range(weeks))
    rows = []
    for spot in range(min(25, len(payload["teams"]))):
        cells = []
        for week in range(weeks):
            at_rank = [t for t in payload["teams"] if len(t["rankHistory"]) > week
                       and t["rankHistory"][week] == spot + 1]
            name = at_rank[0]["team"] if at_rank else "—"
            link = (f'<a href="team/{slug(name)}.html">{esc(name)}</a>'
                    if at_rank else '<span class="muted">—</span>')
            cells.append(f"<td>{link}</td>")
        rows.append(f'<tr><td class="rk">{spot + 1}</td>{"".join(cells)}</tr>')

    body = f"""
  <h1>Archive</h1>
  <p class="hint">What the ranking said at the end of each week. Ratings are recomputed
  from scratch each week rather than nudged, but the record of what they said is kept
  &mdash; a ranking that quietly rewrites its own history is not a record.</p>
  <div class="tablewrap"><table class="archive">
    <thead><tr><th>#</th>{header}</tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table></div>"""
    return page(f"Archive — MRI {season_text(payload)}", body, payload)


def seasons_index(payload: dict) -> str:
    """Every season the system has, newest first.

    Deliberately two tables rather than one. The archive spans two ratings on
    two scales - Classic counts cumulative points where 150 is a great season,
    MRI 2.0 counts points against an average team where +35 is - and a single
    sorted column would invite a comparison that is not available. The seam is
    shown rather than smoothed.
    """
    chrome = chrome_for(payload)
    entries = payload.get("seasons") or []
    if not entries:
        return ""

    def table(rows, note):
        if not rows:
            return ""
        second = rows[0].get("secondaryName")
        body = "".join(f"""
      <tr>
        <td class="rk"><a href="season/{s['season']}.html">{esc(s['label'])}</a></td>
        <td class="opp">{esc(s['teams'][0]['team'])}</td>
        <td class="rec">{s['teams'][0]['wins']}&ndash;{s['teams'][0]['losses']}</td>
        <td class="num">{s['teams'][0]['rating']:,.2f}</td>
        {f'<td class="num muted">{s["teams"][0]["secondary"]:,.2f}</td>' if second and s["teams"][0].get("secondary") is not None else ('<td class="num muted">&ndash;</td>' if second else '')}
        <td class="num muted">{s['rated']}</td>
      </tr>""" for s in rows if s["teams"])
        return f"""
  <h2>{esc(rows[0]['system'])}</h2>
  <p class="hint">{note}</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Season</th><th>Number one</th><th>Rec</th>
    <th class="num">{esc(rows[0]['ratingName'])}</th>
    {f'<th class="num">{esc(second)}</th>' if second else ''}
    <th class="num">Teams rated</th></tr></thead>
    <tbody>{body}</tbody>
  </table></div>"""

    modern = [s for s in entries if s["system"] == "MRI 2.0"]
    classic = [s for s in entries if s["system"] == "MRI Classic"]
    verified = [s for s in classic if s.get("matchesPublished")]
    # The cumulative caveat applies to both sports; the 2017 example does not,
    # and a note explaining a football workbook on the basketball page is just a
    # wrong sentence in an archive that exists to be trusted.
    published = [s for s in classic if s.get("source") == "published"]
    computed = [s for s in classic if s.get("source") == "computed"]
    if verified:
        provenance = ("recomputed from the game logs in Ben's own workbooks; "
                      f"{len(verified)} of these reproduce the published top 25 exactly")
    elif published and computed:
        provenance = (
            f"{len(published)} of these seasons are exactly as Ben published them; "
            f"the other {len(computed)} are years he never ran, with the same formula "
            "applied to them here"
        )
    elif computed:
        provenance = "seasons Ben never ran, with his formula applied to them here"
    else:
        provenance = "exactly as Ben published it at the time"
    example = (
        " 2017's workbook stops before the bowls, which is most of why its number "
        "looks small."
        if chrome.sport == "football" else ""
    )
    classic_note = (
        f"The original formula, {provenance}. Classic totals are cumulative, so a "
        f"season with more games scores higher for that reason alone.{example}"
        + (" The per-game column is the comparable one."
           if any(s.get("secondaryName") for s in classic) else "")
    )

    body = f"""
  <h1>Seasons</h1>
  <p class="hint">Every season MRI has rated, and what it said when the season was
  over. Two ratings, two scales, kept apart on purpose: Classic counts cumulative
  points, where a great season is around 150; MRI 2.0 counts points against an
  average {chrome.field} team, where a great season is around +35. A number from one
  cannot be read against a number from the other.</p>
  {table(modern, "Points against an average team &mdash; the current rating.")}
  {table(classic, classic_note)}"""
    return page(f"Seasons — MRI {chrome.noun}", body, payload,
                description=f"Every season of MRI {chrome.noun} ratings, back to the beginning.")


def season_page(entry: dict, payload: dict) -> str:
    """One season's final ranking."""
    chrome = chrome_for(payload)
    lookup = {t["team"]: t for t in payload["teams"]}

    def name(team: str) -> str:
        # Linked only where the team still exists in the current field; an
        # archive is full of programs that have since moved or folded.
        if team in lookup:
            return f'<a href="../team/{slug(team)}.html">{esc(team)}</a>'
        return esc(team)

    secondary = entry.get("secondaryName")
    rows = "".join(f"""
      <tr>
        <td class="rk">{t['rank']}</td>
        <td class="opp">{name(t['team'])}</td>
        <td class="rec">{t['wins']}&ndash;{t['losses']}</td>
        <td class="num">{t['rating']:,.2f}</td>
        {f'<td class="num muted">{t["secondary"]:,.2f}</td>' if secondary and t.get("secondary") is not None else ('<td class="num muted">&ndash;</td>' if secondary else '')}
      </tr>""" for t in entry["teams"])

    checked = ""
    if entry.get("matchesPublished"):
        checked = ("""
  <p class="note">This table was recomputed from the season's game log rather than
  copied from the workbook, and it reproduces the published top 25 exactly.</p>""")
    elif entry.get("source") == "published":
        checked = ("""
  <p class="note">Taken directly from the workbook Ben published at the time, not
  recomputed.</p>""")
    elif entry.get("source") == "computed" and entry["system"] == "MRI Classic":
        checked = ("""
  <p class="note">Ben never ran this season. These are his original formula's
  ratings computed here from the season's box scores, so they are what MRI Classic
  says about the year rather than a record of what it said at the time.</p>""")

    body = f"""
  <h1>{esc(entry['label'])}</h1>
  <p class="teamsub">{esc(entry['system'])} &middot; {entry['rated']} teams rated
  &middot; showing the top {len(entry['teams'])}</p>
  <div class="tablewrap"><table>
    <thead><tr><th>#</th><th>Team</th><th>Rec</th>
    <th class="num">{esc(entry['ratingName'])}</th>
    {f'<th class="num">{esc(secondary)}</th>' if secondary else ''}</tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
  {checked}
  <p class="hint"><a href="../seasons.html">All seasons</a></p>"""
    return page(f"{entry['label']} — MRI {chrome.noun}", body, payload, depth=1,
                description=f"Final MRI {chrome.noun} ratings for {entry['label']}.")


def betting_page(payload: dict, betting: dict, board: dict) -> str:
    """The betting page, led by the verdict rather than the card.

    A betting page that opens with picks and hides its record is a tout sheet.
    This one opens with the finding that the model loses against closing lines,
    because that is the most important true thing about it.
    """
    closing, opening = betting["closing"], betting["opening"]
    record_html = _record_section(payload["record"]) if payload.get("record") else ""

    def bucket_rows(rows, clv=False):
        out = []
        for r in rows:
            highlight = ' class="total"' if r["edge"] == "ALL" else ""
            beat = float(r["ats"]) >= betting["breakEven"]
            out.append(f"""<tr{highlight}>
              <td>{esc(r['edge'])}</td>
              <td class="num">{int(r['bets'])}</td>
              <td class="num {'over' if beat else 'under'}">{float(r['ats']):.1%}</td>
              <td class="num {'over' if float(r['units']) > 0 else 'under'}">{float(r['units']):+.1f}</td>
              {f'<td class="num">{float(r["clv"]):+.2f}</td>' if clv and r.get('clv') == r.get('clv') else ('<td class="num muted">&ndash;</td>' if clv else '')}
            </tr>""")
        return "".join(out)

    gradient = "".join(f"""<tr>
        <td>{esc(r['edge'])} pts</td><td class="num">{r['bets']}</td>
        <td class="num {'over' if r['clv'] > 0 else 'under'}">{r['clv']:+.2f}</td>
        <td class="num">{r['ats']:.1%}</td></tr>""" for r in opening["clvGradient"])

    seasons = "".join(f"""<tr><td>{r['season']}</td><td class="num">{r['bets']}</td>
        <td class="num">{r['ats']:.1%}</td>
        <td class="num {'over' if r['units'] > 0 else 'under'}">{r['units']:+.1f}</td>
        <td class="num {'over' if r['clv'] > 0 else 'under'}">{r['clv']:+.2f}</td></tr>"""
        for r in opening["bySeason"])

    flagged = board.get("flagged", [])
    card = "".join(f"""<tr>
        <td class="wk">{g['week']}</td>
        <td class="opp">{esc(g['away'])} {'vs' if g['neutral'] else 'at'} {esc(g['home'])}</td>
        <td class="num">{g['predicted']:+.1f}</td>
        <td class="num">{(g['marketOpen'] if g['marketOpen'] is not None else g['market']):+.1f}</td>
        <td class="num perf {'over' if g['edge'] > 0 else 'under'}">{g['edge']:+.1f}</td>
        <td>{esc(g['side'])}</td></tr>""" for g in flagged[:15]) or \
        '<tr><td colspan="6" class="empty">No games on the current card.</td></tr>'

    set_aside = [g for g in board.get("games", []) if not g.get("confident")]
    aside_note = ""
    if set_aside:
        names = ", ".join(sorted({t for g in set_aside for t in g["unknownTeams"]}))
        aside_note = (f'<p class="note">{len(set_aside)} game(s) set aside because the model has '
                      f'no prior season for one side: {esc(names)}. An early-season "edge" against '
                      f'a team the model has never rated is ignorance, not an opinion.</p>')

    body = f"""
  <article class="prose">
  <h1>Betting</h1>

  <div class="verdict">
    <p class="vlead">This model does not beat closing lines.</p>
    <p>Across <strong>{closing['bets']:,} walk-forward bets</strong> from
    {betting['seasons'][0]}&ndash;{betting['seasons'][1]}, it went
    <strong>{closing['ats']:.1%}</strong> against the market's number. Break-even at
    &minus;110 is {betting['breakEven']:.2%}. That is <strong>{closing['units']:+.0f} units</strong>
    &mdash; not a near miss, a verdict.</p>
  </div>

  <p>That is the expected result and worth stating first. A closing spread is the
  sharpest number in sports, and a rating built from final scores is not going to
  out-argue thousands of people with money at stake. Anyone whose model beats closing
  lines by a point and a half is either wrong about their backtest or should not be
  publishing it.</p>

  <h2>Against the number the market opens at</h2>
  <p>Openers are a different question, because they are posted before the market has
  digested anything. Here the picture is better but not conclusive:
  <strong>{opening['ats']:.1%}</strong> over {opening['bets']:,} bets
  ({opening['units']:+.0f} units) across {opening['seasons'][0]}&ndash;{opening['seasons'][-1]},
  the only seasons for which opening lines exist.</p>
  <p>That carries <strong>p = {opening['pValue']}</strong>. It is not significant, and the
  best individual bucket does not survive correction for the number of buckets examined.
  Treat it as a hint.</p>

  <table class="compare wide">
    <thead><tr><th>Disagreement</th><th class="num">Bets</th><th class="num">ATS</th>
    <th class="num">Units</th><th class="num">CLV</th></tr></thead>
    <tbody>{bucket_rows(opening['buckets'], clv=True)}</tbody>
  </table>

  <h2>The one clean signal</h2>
  <p>Closing line value &mdash; whether the market moves toward your side after you bet
  &mdash; is the honest early indicator, because it converges in hundreds of bets where
  win-loss records need thousands. It rises steadily with the size of the disagreement:</p>

  <table class="compare wide">
    <thead><tr><th>Disagreement</th><th class="num">Bets</th><th class="num">Mean CLV</th>
    <th class="num">ATS</th></tr></thead>
    <tbody>{gradient}</tbody>
  </table>

  <p>A dose-response pattern across ordered bins is much harder to produce by chance than
  one good bucket, and it is positive in all three seasons. It says the market tends to
  drift toward this model's side after the opener &mdash; which is what genuine information
  looks like, and is the reason to keep measuring rather than to start betting.</p>

  <table class="compare wide">
    <thead><tr><th>Season</th><th class="num">Bets</th><th class="num">ATS</th>
    <th class="num">Units</th><th class="num">CLV</th></tr></thead>
    <tbody>{seasons}</tbody>
  </table>

  <h2>How much more would it take to know</h2>
  <p>To confirm a true 53.5% edge at conventional power would take roughly
  <strong>12,000 bets</strong> &mdash; about seventeen seasons of betting every game. Even a
  56% edge would need close to two full seasons. The honest conclusion is that this
  question cannot be settled from history. It can only be settled forward, which is what
  the card below is for.</p>

{record_html}
  <h2>This week</h2>
  <p class="note"><strong>Tracked, not recommended.</strong> These are the largest
  disagreements with the opening number. They are published so the record accumulates in
  public, including the losing weeks.</p>
  {aside_note}
  <div class="tablewrap"><table>
    <thead><tr><th>Wk</th><th>Game</th><th class="num">Model</th><th class="num">Open</th>
    <th class="num">Edge</th><th>Side</th></tr></thead>
    <tbody>{card}</tbody>
  </table></div>
  </article>"""
    return page(f"Betting — MRI {payload['season']}", body, payload,
                description="What the model says about the market, and how badly it has done.")



# --------------------------------------------------------------------------
# season simulation, weekly slate, and the public record
# --------------------------------------------------------------------------

def _pct(value, *, signed: bool = False) -> str:
    """A probability as people read it: no false precision at either end."""
    if value is None:
        return "&ndash;"
    v = float(value)
    if signed:
        points = v * 100
        if abs(points) < 0.05:
            return "0"
        return f"{points:+.0f}" if abs(points) >= 10 else f"{points:+.1f}"
    if v < 0.0005:
        return "&lt;0.1%"
    if v < 0.10:
        return f"{v:.1%}"
    if v > 0.995:
        return "&gt;99%"
    return f"{v:.0%}"


def _record_text(wins, losses) -> str:
    return f"{int(wins)}&ndash;{int(losses)}"


_SIM_SCRIPT = '''
<script>
(function () {
  var body = document.querySelector('#simtable tbody');
  var items = Array.prototype.slice.call(body.children);
  var conf = document.getElementById('conf'), sort = document.getElementById('sort');
  var live = document.getElementById('live'), none = document.getElementById('none');
  function apply() {
    var key = sort.value, want = conf.value, shown = 0;
    var ordered = items.slice().sort(function (a, b) {
      if (key === 'rank') return a.dataset.rank - b.dataset.rank;
      return b.dataset[key] - a.dataset[key];
    });
    ordered.forEach(function (row, i) {
      var chance = parseFloat(row.dataset.field) > 0.0005 || parseFloat(row.dataset.champ) > 0.005;
      var ok = (!want || row.dataset.conf === want) && (!live.checked || chance);
      row.hidden = !ok;
      if (ok) { shown++; row.firstElementChild.textContent = shown; }
      body.appendChild(row);
    });
    none.hidden = shown > 0;
  }
  conf.addEventListener('change', apply);
  sort.addEventListener('change', apply);
  live.addEventListener('change', apply);
  apply();
})();
</script>'''


def simulation_page(payload: dict) -> str:
    sim = payload["sim"]
    odds = sim["teams"]
    teams = {t["team"]: t for t in payload["teams"]}
    details = payload.get("details") or {}
    n = sim["sims"]

    # Where each team's remaining schedule sits: the average power of the
    # opponents still to play, ranked hardest first.
    remaining = {t: d.get("remainingDifficulty") for t, d in details.items()
                 if d.get("remainingDifficulty") is not None}
    hardest = {t: i + 1 for i, t in enumerate(sorted(remaining, key=lambda t: -remaining[t]))}

    ordered = sorted(odds, key=lambda t: (-odds[t]["playoff"], -odds[t]["title"], teams[t]["rank"]))
    rows = []
    for position, name in enumerate(ordered, 1):
        o, team = odds[name], teams[name]
        change = o.get("playoffChange")
        chip = ""
        if change is not None and abs(change) >= 0.0005:
            kind = "over" if change > 0 else "under"
            chip = f'<span class="{kind}">{_pct(change, signed=True)}</span>'
        elif change is not None:
            chip = '<span class="muted">0</span>'
        difficulty = remaining.get(name)
        rows.append(f"""<tr data-conf="{esc(team['conference'])}" data-field="{o['playoff']}"
            data-title="{o['title']}" data-champ="{o['conferenceTitle']}" data-wins="{o['projectedWins']}"
            data-rank="{team['rank']}" data-left="{difficulty if difficulty is not None else -99}">
          <td class="rk">{position}</td>
          <td class="opp"><span class="nmcell">{identity_mark(team, 18)}<a href="team/{slug(name)}.html">{esc(name)}</a></span>
            <span class="muted sub">{esc(team['conference'])}</span></td>
          <td class="num">{_record_text(team['wins'], team['losses'])}</td>
          <td class="num">{o['projectedWins']:.1f}&ndash;{o['projectedLosses']:.1f}</td>
          <td class="num">{o['gamesLeft']}</td>
          <td class="num" title="{('Hardest remaining schedule: #%d of %d' % (hardest[name], len(hardest))) if name in hardest else ''}">{f'{difficulty:+.1f}' if difficulty is not None else '&ndash;'}</td>
          <td class="num">{_pct(o['conferenceTitle']) if team['conference'] not in ('FBS Independent', 'Independent') else '&ndash;'}</td>
          <td class="num prob"><span class="pbar" style="width:{o['playoff'] * 64:.0f}px"></span>{_pct(o['playoff'])}</td>
          <td class="num">{_pct(o['bye'])}</td>
          <td class="num prob"><span class="pbar alt" style="width:{min(o['title'] * 4, 1) * 64:.0f}px"></span>{_pct(o['title'])}</td>
          <td class="num">{chip}</td>
        </tr>""")

    conferences = []
    for c in sorted({t["conference"] for t in teams.values()}):
        members = sorted((t for t in teams.values() if t["conference"] == c),
                         key=lambda t: -odds[t["team"]]["conferenceTitle"])
        if c in ("FBS Independent", "Independent"):
            continue
        top = members[:3]
        line = " &middot; ".join(
            f'<a href="team/{slug(t["team"])}.html">{esc(t["team"])}</a> {_pct(odds[t["team"]]["conferenceTitle"])}'
            for t in top)
        conferences.append(f"""
    <div class="confcard">
      <span class="ccname">{esc(c)}</span>
      <span class="ccmeta">{line}</span>
    </div>""")

    movers = ""
    if sim.get("hasHistory"):
        changed = [t for t in ordered if odds[t].get("playoffChange") is not None]
        up = sorted(changed, key=lambda t: -odds[t]["playoffChange"])[:5]
        down = sorted(changed, key=lambda t: odds[t]["playoffChange"])[:5]

        def mover_rows(names):
            return "".join(
                f"""<li><a class="gnm" href="team/{slug(t)}.html">{esc(t)}</a>
                <span class="gv">{_pct(odds[t]['playoff'])} <em>{_pct(odds[t]['playoffChange'], signed=True)}</em></span></li>"""
                for t in names if abs(odds[t]["playoffChange"]) >= 0.0005
            ) or '<li class="empty">No movement yet.</li>'

        movers = f"""
  <div class="grid two">
    <section class="panel"><p class="ptitle">Gaining since last week</p><ul class="list">{mover_rows(up)}</ul></section>
    <section class="panel"><p class="ptitle">Losing since last week</p><ul class="list">{mover_rows(down)}</ul></section>
  </div>"""

    options = "".join(f'<option value="{esc(c["conference"])}">{esc(c["conference"])}</option>'
                      for c in payload["conferences"])
    backtest = payload.get("simBacktest")
    check = ""
    if backtest:
        model = backtest["variants"]["model"]
        check = (f"""<p class="note">Checked against history: run as of weeks {', '.join(str(w) for w in backtest['weeks'][:-1])}
        and {backtest['weeks'][-1]} of {backtest['seasons'][0]}&ndash;{backtest['seasons'][1]}, its chance of a
        top-12 finish scored a Brier of {model['brier']:.3f} against {backtest['baseBrier']:.3f} for
        knowing nothing but the base rate. <a href="method.html#simulation">How it is built and how it did.</a></p>""")

    body = f"""
  <article class="prose wide">
  <h1>Season simulation</h1>
  <p class="lead">The rest of the {payload['season']} schedule played out {n:,} times from the ratings
  as of {esc(period_text(payload))}: standings settled, the ten conference championship games played, a
  committee ranking produced, the 12-team field picked under this year's rules, and the bracket
  played through. The columns are how often each thing happened.</p>
  <p class="hint"><strong>Playoff</strong> is making the field &mdash; the four power-conference
  champions, the best team from the other six conferences, Notre Dame if it is ranked in the top 12,
  and the highest-ranked rest. <strong>Bye</strong> is a top-four seed. <strong>Title</strong> is winning
  the whole thing. Ratings are treated as estimates, not facts, so early in the season the range is wide.</p>
  {check}

  <section class="panel">
    <div class="panelhead">
      <p class="ptitle">Every team</p>
      <div class="controls">
        <label class="sr">Conference<select id="conf"><option value="">All conferences</option>{options}</select></label>
        <label class="sr">Sort<select id="sort">
          <option value="field">By playoff chance</option>
          <option value="title">By title chance</option>
          <option value="champ">By conference title</option>
          <option value="wins">By projected wins</option>
          <option value="left">By hardest schedule left</option>
          <option value="rank">By power</option>
        </select></label>
        <label class="chk"><input type="checkbox" id="live" checked> Only teams with a chance</label>
      </div>
    </div>
    <div class="tablewrap"><table id="simtable" class="simtable">
      <thead><tr><th class="rk">#</th><th>Team</th><th class="num">Rec.</th><th class="num" title="Projected final regular-season record">Proj.</th>
      <th class="num" title="Games remaining">Left</th><th class="num" title="Average power of the opponents still to play">Sched. left</th>
      <th class="num">Conf. title</th><th class="num">Playoff</th><th class="num">Bye</th><th class="num">Title</th>
      <th class="num" title="Change in playoff chance since last week, in points">&Delta;</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table></div>
    <p class="empty" id="none" hidden>No teams match that filter.</p>
  </section>
{movers}
  <h2>Conference races</h2>
  <div class="confgrid">{''.join(conferences)}</div>

  <h2>What this does not know</h2>
  <p>It knows scores, not injuries: a starting quarterback lost next week changes a team's
  chances and this page will not see it until the results show it. Conference tiebreakers are
  simplified to a coin flip after conference wins, so a two-way tie's odds are slightly off. And
  the committee is modelled from what it has valued in past years &mdash; résumé far more than
  strength &mdash; not asked.</p>
  <p class="muted">Deterministic for a given set of results and ratings: the same inputs give the
  same page, so it changes when something happened and not otherwise.</p>
  </article>"""
    body += _SIM_SCRIPT
    return page(f"Season simulation — MRI {season_text(payload)}", body, payload,
                description="Playoff, bye and title odds from simulating the rest of the season.")


def _favorite(predicted: float | None, home: str, away: str) -> str:
    """'Texas Tech −8.1' from a home-perspective margin."""
    if predicted is None:
        return "&ndash;"
    if abs(predicted) < 0.05:
        return "Pick'em"
    team = home if predicted > 0 else away
    return f"{esc(team)}&nbsp;&minus;{abs(predicted):.1f}"


def _matchup(g: dict, teams: dict) -> str:
    def side(name):
        team = teams.get(name)
        rank = f'<span class="rkchip">#{team["rank"]}</span> ' if team and team["rank"] <= 25 else ""
        mark = identity_mark(team, 16) if team else ""
        label = (f'<a href="team/{slug(name)}.html">{esc(name)}</a>' if team else esc(name))
        return f'<span class="nmcell">{mark}{rank}{label}</span>'
    joiner = "vs" if g["neutral"] else "at"
    return f'<span class="mu">{side(g["away"])} <span class="muted">{joiner}</span> {side(g["home"])}</span>'


def slate_page(payload: dict) -> str:
    slate = payload["slate"]
    teams = {t["team"]: t for t in payload["teams"]}
    by_id = {g["id"]: g for d in slate["days"] for g in d["games"]}
    # Links only to pages this build wrote, the same rule as the header.
    betting_ref = ('<a href="betting.html">betting page</a>'
                   if payload.get("betting") and payload.get("board") else "betting board")
    sim_ref = ('<a href="simulation.html">season simulation</a>'
               if payload.get("sim") else "season simulation")

    def market_cell(g):
        if g.get("market") is None:
            return '<span class="muted">&ndash;</span>'
        moved = ""
        if g.get("open") is not None and abs(g["open"] - g["market"]) >= 0.5:
            moved = f' <span class="muted" title="Where the line opened">opened {_favorite(g["open"], g["home"], g["away"])}</span>'
        return _favorite(g["market"], g["home"], g["away"]) + moved

    def stake_cell(g):
        s = g.get("stake")
        if not s or s["swing"] < 0.02:
            return '<span class="muted">&ndash;</span>'
        return (f'{esc(s["team"])} <span class="muted">{_pct(s["ifLose"])} &rarr;</span> '
                f'<strong>{_pct(s["ifWin"])}</strong>')

    def upcoming_row(g):
        favourite_p = g["homeWinProbability"] if g["predicted"] >= 0 else 1 - g["homeWinProbability"]
        edge = g.get("edge")
        edge_cell = "&ndash;" if edge is None else f"{edge:+.1f}"
        flag = ' class="flagged"' if g.get("flagged") else ""
        return f"""<tr{flag}>
          <td class="wk">{g['time']}</td>
          <td class="opp">{_matchup(g, teams)}</td>
          <td class="num">{_favorite(g['predicted'], g['home'], g['away'])}</td>
          <td class="num">{_pct(favourite_p)}</td>
          <td class="num">{market_cell(g)}</td>
          <td class="num perf {'over' if (edge or 0) > 0 else 'under'}">{edge_cell}{' &#9873;' if g.get('flagged') else ''}</td>
          <td>{stake_cell(g)}</td>
        </tr>"""

    head = """<thead><tr><th>Time</th><th>Game</th><th class="num">Model</th><th class="num">Win</th>
      <th class="num">Market</th><th class="num" title="Model minus the opening number">Edge</th>
      <th title="The team with the most to lose: its playoff chance if it loses, then if it wins">Playoff stake</th></tr></thead>"""

    days = "".join(f"""
  <h2>{esc(d['label'])}</h2>
  <div class="tablewrap"><table class="slate">{head}<tbody>{''.join(upcoming_row(g) for g in d['games'])}</tbody></table></div>"""
                   for d in slate["days"])

    watch = "".join(f"""<li>{_matchup(by_id[i], teams)}
        <span class="gv">{by_id[i]['dateLabel']} &middot; {stake_cell(by_id[i])}</span></li>"""
                    for i in slate["watch"] if i in by_id)
    watch_panel = f"""
  <section class="panel">
    <p class="ptitle">Most riding on it</p>
    <ul class="list watch">{watch}</ul>
    <p class="note">Games ranked by how much the result moves one team's playoff chance, from the
    {sim_ref}. The first figure is that team's chance if it loses;
    the second, if it wins.</p>
  </section>""" if watch else ""

    def result_row(g):
        r = g["result"]
        winner_home = r["homeScore"] > r["awayScore"]
        score = (f'{esc(g["away"])} {r["awayScore"]}, {esc(g["home"])} {r["homeScore"]}')
        mark = "&#10003;" if r["modelCorrect"] else "&#10007;"
        cls = "over" if r["modelCorrect"] else "under"
        market = f'{r["marketError"]:.1f}' if r.get("marketError") is not None else "&ndash;"
        return f"""<tr><td class="wk">{esc(g['dateLabel'])}</td><td class="opp">{score}</td>
          <td class="num">{_favorite(g['predicted'], g['home'], g['away'])}</td>
          <td class="num {cls}">{mark}</td><td class="num">{r['modelError']:.1f}</td><td class="num">{market}</td></tr>"""

    results = ""
    if slate["results"]:
        results = f"""
  <h2>Already played this week</h2>
  <div class="tablewrap"><table class="slate">
    <thead><tr><th>Day</th><th>Final</th><th class="num">Model (entering the week)</th><th class="num">Called it</th>
    <th class="num">Model miss</th><th class="num">Market miss</th></tr></thead>
    <tbody>{''.join(result_row(g) for g in slate['results'])}</tbody></table></div>"""

    fcs = ""
    if slate["fcs"]:
        rows = "".join(f"""<tr><td class="wk">{esc(g['dateLabel'])} &middot; {g['time']}</td>
          <td class="opp">{_matchup(g, teams)}</td>
          <td class="num">{_favorite(g['predicted'], g['home'], g['away'])}</td>
          <td class="num">{_pct(g['homeWinProbability'] if g['predicted'] >= 0 else 1 - g['homeWinProbability'])}</td></tr>"""
                       for g in slate["fcs"])
        fcs = f"""
  <h2>Against FCS opponents</h2>
  <div class="tablewrap"><table class="slate"><thead><tr><th>When</th><th>Game</th>
    <th class="num">Model</th><th class="num">Win</th></tr></thead><tbody>{rows}</tbody></table></div>"""

    body = f"""
  <article class="prose wide">
  <h1>Week {slate['week']} slate</h1>
  <p class="lead">{slate['games']} games with an FBS team. The model's line and win chance for each,
  the market's number beside it (DraftKings), and what the result does to the playoff picture.</p>
  <p class="hint"><strong>Model</strong> is the favourite and the margin the ratings predict.
  <strong>Edge</strong> is the model minus the opening number, in points; a flag means it is large enough
  that the {betting_ref} tracks it. These are tracked, not recommended:
  see how that has gone on the same page.</p>
{watch_panel}{days}{results}{fcs}
  </article>"""
    return page(f"Week {slate['week']} slate — MRI {season_text(payload)}", body, payload,
                description="Every game this week: model line, market line, and what rides on it.")


def _record_section(record: dict) -> str:
    """The public track record, for the betting page."""
    rec, fwd, ref = record["reconstructed"], record["forward"], record["reference"]
    if not rec.get("summary"):
        return ""
    s = rec["summary"]
    b = s["bets"]

    def signed(value, digits=1):
        return f"{value:+.{digits}f}"

    gap = s.get("maeGap")
    gap_text = ""
    if gap is not None and s.get("maeGapError") is not None:
        worse = gap > 0
        gap_text = (f"That is {abs(gap):.1f} points {'worse' if worse else 'better'} than the market "
                    f"(&plusmn;{s['maeGapError']:.1f}).")

    weeks = "".join(f"""<tr><td>{w['week']}</td><td class="num">{w['games']}</td>
        <td class="num">{w['accuracy']:.0%}</td><td class="num">{w['mae']:.1f}</td>
        <td class="num">{f"{w['marketMae']:.1f}" if w.get('marketMae') is not None else '&ndash;'}</td>
        <td class="num">{w['record']}</td>
        <td class="num {'over' if w['units'] > 0 else 'under' if w['units'] < 0 else ''}">{w['units']:+.1f}</td></tr>"""
                    for w in rec["weeks"])

    earlier = (f" {fwd['earlier']} of the picks were logged before the preseason prior began to use roster talent "
               "and returning production, so they were made by the earlier model; they stay as written."
               if fwd.get("earlier") else "")
    if fwd["logged"]:
        picks = "".join(f"""<tr><td class="wk">{p['week']}</td>
          <td class="opp">{esc(p['away'])} {'vs' if p['neutral'] else 'at'} {esc(p['home'])}</td>
          <td>{esc(p['home'] if p['side'] == 'home' else p['away'])}</td>
          <td class="num">{p['predicted']:+.1f}</td><td class="num">{p['taken']:+.1f}</td>
          <td class="num">{p['edge']:+.1f}</td>
          <td class="num">{esc(p['result']) if 'result' in p else '<span class="muted">pending</span>'}</td>
          <td class="num">{f"{p['clv']:+.1f}" if 'clv' in p else '<span class="muted">&ndash;</span>'}</td></tr>"""
                        for p in fwd["picks"][:40])
        more = (f'<p class="muted">Showing 40 of {fwd["logged"]}.</p>' if fwd["logged"] > 40 else "")
        forward = f"""
  <h3>The forward log</h3>
  <p>Started {esc(fwd['started'])}. Each flagged game is written down before kickoff &mdash; model line,
  the market's number at that moment, the side &mdash; and never edited afterwards. Graded against the number
  taken, at &minus;110.{earlier}</p>
  <p><strong>{fwd['wins']}&ndash;{fwd['losses']}{f"&ndash;{fwd['pushes']}" if fwd['pushes'] else ''}</strong>
  on {fwd['graded']} graded of {fwd['logged']} logged
  ({signed(fwd['units'])} units{f", mean closing line value {signed(fwd['clv'])}" if fwd['clv'] is not None else ''}).</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Wk</th><th>Game</th><th>Side</th><th class="num" title="Model's margin for the home team">Model (home)</th><th class="num" title="Market's expected home margin when logged">Line (home)</th>
    <th class="num">Edge</th><th class="num">Result</th><th class="num">CLV</th></tr></thead>
    <tbody>{picks}</tbody></table></div>{more}"""
    else:
        forward = f"""
  <h3>The forward log</h3>
  <p>Started {esc(fwd['started'])}. Nothing logged yet: it records each flagged game before kickoff and
  never edits it afterwards.</p>"""

    slope_text = ""
    if s.get("slope") is not None and s.get("marketSlope") is not None:
        slope_text = (f"""<tr><td>Size of the lines</td><td class="num">{s['slope']:.2f}</td>
          <td class="num">{s['marketSlope']:.2f}</td></tr>""")

    return f"""
  <h2>The record so far</h2>
  <p>Two records, and only one of them is a track record.</p>

  <h3>The season, reconstructed</h3>
  <p>For every game already played, what the model would have said using only the ratings from before that
  week. Nothing from the week it is predicting leaks in, but it was computed afterwards, and a record computed
  afterwards is a backtest however carefully it is done.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>{s['games']} games</th><th class="num">MRI 2.0</th><th class="num">Market</th></tr></thead>
    <tbody>
      <tr><td>Picked the winner</td><td class="num">{s['accuracy']:.1%}</td><td class="num">{f"{s['marketAccuracy']:.1%}" if s.get('marketAccuracy') is not None else '&ndash;'}</td></tr>
      <tr><td>Average miss, points</td><td class="num">{s['modelMaePriced'] if s.get('modelMaePriced') is not None else s['mae']}</td><td class="num">{s['marketMae'] if s.get('marketMae') is not None else '&ndash;'}</td></tr>{slope_text}
    </tbody>
  </table></div>
  <p>{gap_text} Over the seventeen archived seasons the model averaged {ref['mae']:.1f} points and
  picked {ref['accuracy']:.1%} winners; a September sample has far less to go on, because the ratings
  lean on last year until this year's results replace them. A size of 1.00 means the lines are the right
  size; below it, they are too extreme &mdash; a favourite laying more points than it wins by.</p>

  <p>Where the model disagreed with the opening number by {MIN_EDGE:g} points or more, the board's own rule:
  <strong>{b['wins']}&ndash;{b['losses']}{f"&ndash;{b['pushes']}" if b['pushes'] else ''}</strong> against the number
  ({f"{b['ats']:.1%}" if b['ats'] is not None else '&ndash;'}; break-even is 52.4%), {signed(b['units'])} units,
  mean closing line value {signed(b['clv']) if b['clv'] is not None else '&ndash;'}.
  That is {b['count']} bets, far too few to say whether the disagreements mean anything.</p>

  <div class="tablewrap"><table>
    <thead><tr><th>Week</th><th class="num">Games</th><th class="num">Winners</th><th class="num">Miss</th>
    <th class="num">Market miss</th><th class="num">Bets</th><th class="num">Units</th></tr></thead>
    <tbody>{weeks}</tbody>
  </table></div>
{forward}"""





def _prior_method_section(payload: dict) -> str:
    model = payload.get("priorModel")
    if not model:
        return ""
    c, fit = model["coefficients"], model.get("talentFit") or {}
    low, acad = model["lowReturning"], model["academies"]
    rmse = model.get("outOfSampleRmse") or {}
    bt = payload.get("priorBacktest")
    table = ""
    if bt:
        rows = "".join(
            f"""<tr><td>{esc(label)}</td><td class="num">{d['old']['games']:,}</td>
            <td class="num">{d['old']['mae']:.2f}</td><td class="num">{d['new']['mae']:.2f}</td>
            <td class="num">{d['maeGain']:+.2f} &plusmn;{d['maeGainError']:.2f}</td>
            <td class="num">{d['old']['marketMae']:.2f}</td></tr>"""
            for label, d in bt["spans"].items())
        table = f"""
  <p>Every game of {bt['seasons'][0]}&ndash;{bt['seasons'][1]} (2020 aside) predicted from only the earlier weeks of its own
  season, once from each prior. Both chains run properly, each season starting from the previous season's ratings under the
  same method, with the new prior's coefficients fitted leaving the predicted season out.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Games in</th><th class="num">Predicted</th><th class="num">Old prior miss</th>
    <th class="num">New prior miss</th><th class="num">Gain, points</th><th class="num">Market miss</th></tr></thead>
    <tbody>{rows}</tbody></table></div>
  <p>The gain is largest in the first three weeks, where the prior is most of what the ratings know, and fades as results
  replace it. The market still wins every row.</p>"""
    persistence = (f" The gap is not just noise: a team that beat its talent one year did so the next with a correlation of "
                   f"{fit['persistence']:.2f}.") if fit.get("persistence") is not None else ""
    return f"""
  <h2 id="priors">Preseason priors</h2>
  <p>Every season starts from a prior, and for a long time the prior was last season's rating pulled 30% toward
  average for everyone. That treats a team that returns its whole roster like one that lost it to the transfer portal.
  It is now a regression of a season's final rating on three things known before a game is played:</p>
  <p>last season's rating, counted at {c['last_season']:.2f};
  the roster's <strong>talent</strong> (the 247Sports composite, as a z-score within FBS), worth {c['talent']:+.1f} points
  per standard deviation; and <strong>returning production</strong> (the share of last year's production, by predicted points
  added, still on the roster), worth {c['returning']:+.1f} points from none to all of it.</p>
  <p>Fitted on {esc(model['fitted'])} ({model['teamSeasons']:,} team-seasons). Scored by leaving each season out in turn it misses
  by {rmse.get('model', 0):.1f} points where the old prior missed by {rmse.get('old', 0):.1f}. For the teams that returned under
  {low['share']:.0%} of their production ({low['teamSeasons']} team-seasons) the old prior over-rated them by
  {abs(low['oldBias']):.1f} points on average, because a team like that falls about that far from last season. The new prior still
  over-rates them by about {abs(low['newBias']):.0f}.</p>{table}
  <p><strong>Exceptions, on purpose.</strong> The service academies' recruits are not ranked the way everyone else's are, and
  the composite calls Army, Navy and Air Force about {abs(acad['meanZ']):.1f} standard deviations worse than average; across
  {acad['teamSeasons']} team-seasons they beat what that implied by {acad['gap']:.0f} points on average. With talent treated as
  unmeasured for them the prior is off by {abs(acad['gapWithPrior']):.1f}. Teams that were not FBS last year have
  no comparable last season and keep the old prior.</p>
  <p><strong>Talent and results.</strong> Talent alone explains about {fit.get('r2', 0):.0%} of the variation in a season's rating.
  Each team page shows what its roster's talent alone would predict next to what it is actually rated.{persistence}
  In testing, most of the improvement came from talent; returning production adds a smaller one on average, and matters most
  for the few teams that lose nearly everyone.</p>"""

def _sim_method_section(payload: dict) -> str:
    """The method page's account of the simulation, with its own report card."""
    if not payload.get("sim"):
        return ""            # no simulation page this build, so nothing to explain or link to
    weight = sim_season.COMMITTEE_POWER_WEIGHT
    backtest = payload.get("simBacktest")
    card = ""
    if backtest:
        model, exact = backtest["variants"]["model"], backtest["variants"]["exact"]
        bins = "".join(
            f"""<tr><td>{esc(b['range'])}</td><td class="num">{b['teams']:,}</td>
            <td class="num">{b['predicted']:.0%}</td><td class="num">{b['observed']:.0%}</td></tr>"""
            for b in model["calibration"] if b["teams"] >= 50)
        card = f"""
  <p>Graded the honest way: run as it would have been on the morning of weeks
  {', '.join(str(w) for w in backtest['weeks'][:-1])} and {backtest['weeks'][-1]} of every season from
  {backtest['seasons'][0]} to {backtest['seasons'][1]} (2020 excluded), using only what was known then, and
  compared with whether each team finished in the committee's top 12. Each row groups the teams the simulation
  gave that chance; the last columns are how often they did.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Predicted</th><th class="num">Team-seasons</th><th class="num">Average predicted</th>
    <th class="num">Actually did</th></tr></thead>
    <tbody>{bins}</tbody>
  </table></div>
  <p>Brier score {model['brier']:.3f}, against {backtest['baseBrier']:.3f} for a forecast that knows only that
  12 of about 130 teams make it. In Week 3 alone it was {model['byWeek'].get('3', model['byWeek'].get(3, 0)):.3f}.
  Treating the ratings as exact, with no error bars, scored {exact['brier']:.3f} overall &mdash; nearly the
  same, which is an honest finding: the uncertainty matters most in September and matters little by November.</p>"""
    return f"""
  <h2 id="simulation">Simulation and the playoff</h2>
  <p>The <a href="simulation.html">simulation page</a> plays the rest of the season {sim_season.DEFAULT_SIMS:,}
  times. Each run first draws every team's true strength around its rating, with a spread that shrinks as games
  are played, then plays the remaining games, settles conference standings, plays the ten conference
  championship games, ranks the teams the way the selection committee would, selects the 12-team field, and plays the
  bracket. Counting the runs gives each chance.</p>
  <p><strong>The committee is modelled, not asked.</strong> It ranks by r&eacute;sum&eacute; far more than by
  strength. A blend of {1 - weight:.0%} r&eacute;sum&eacute; and {weight:.0%} power recovers 10.8 of the
  committee's true top 12 and 3.7 of its top 4 across 2014&ndash;2025, with the average team landing 1.5 places
  from its real rank. A little noise is added on top, because even at its best the blend misses about one team in
  twelve. The field follows this year's rules: the ACC, Big 12, Big Ten and SEC champions are in; so is the
  highest-ranked team from the American, Conference USA, MAC, Mountain West, Pac-12 and Sun Belt; Notre Dame is in if it is
  ranked in the top 12; the remaining places go to the highest-ranked teams left.</p>
  <p><strong>Simplifications, stated.</strong> Ties in conference wins are broken by coin flip, not head-to-head.
  Injuries are invisible until they show up in results. The committee blend was chosen using the same twelve
  seasons it is reported on.</p>{card}"""


def gameday_page(payload: dict) -> str:
    g = payload["gameday"]
    teams = {t["team"]: t for t in payload["teams"]}
    bt = payload.get("gamedayBacktest") or {}
    choice, ahead = bt.get("choice"), bt.get("forecast")

    def chip(name: str) -> str:
        team = teams.get(name)
        if not team:
            return esc(name)
        return f'<span class="nmcell">{identity_mark(team, 16)}<a href="team/{slug(name)}.html">{esc(name)}</a></span>'

    def matchup(away: str, home: str, neutral: bool) -> str:
        return f'<span class="mu">{chip(away)} <span class="muted">{"vs" if neutral else "at"}</span> {chip(home)}</span>'

    confirmed = "".join(f"""<tr><td class="wk">{a['week']}</td><td class="wk">{esc(dt_label(a['date']))}</td>
          <td class="opp">{matchup(a['teams'][0] if a['teams'][1] == a['host'] else a['teams'][1], a['host'], False) if a.get('host') else ' vs '.join(chip(t) for t in a['teams'])}</td>
          <td>{esc(a['city'])}</td></tr>""" for a in g["announced"])

    def why(e: dict) -> str:
        bits = [f"usually ranked around #{e['rankAway']:.0f} and #{e['rankHome']:.0f}"]
        if e["bothTop10"] >= 0.05:
            bits.append(f"both top 10 in {e['bothTop10']:.0%} of seasons")
        if e["bothUnbeaten"] >= 0.05:
            bits.append(f"both unbeaten in {e['bothUnbeaten']:.0%}")
        return " &middot; ".join(bits)

    def forecast_block(w: dict) -> str:
        rows = []
        for e in w["games"]:
            if e["kind"] == "championship":
                game = f'<strong>{esc(e["conference"])} championship game</strong>'
                site = '<span class="muted">wherever it is played</span>'
            else:
                game = matchup(e["away"], e["home"], e["neutral"])
                site = esc(e["venue"] or e["home"])
            rows.append(f"""<tr><td class="num prob"><span class="pbar" style="width:{min(e['probability'] * 170, 170):.0f}px"></span>{_pct(e['probability'])}</td>
              <td class="opp">{game}</td><td>{site}</td><td class="muted small">{why(e)}</td></tr>""")
        left = max(0.0, 1.0 - w["other"] - w["covered"])
        rows.append(f"""<tr class="rest"><td class="num">{_pct(left)}</td><td class="opp muted">Every other game{f" ({w['omitted']} of them)" if w.get('omitted') else ""}</td><td></td><td></td></tr>
          <tr class="rest"><td class="num">{_pct(w['other'])}</td><td class="opp muted">Somewhere off the schedule &mdash; an FCS game, a surprise</td><td></td><td></td></tr>""")
        title = "Conference championship week" if w["championship"] else "Week " + str(w["week"])
        return f"""
  <section class="panel gd">
    <p class="ptitle">{title} &middot; {esc(w['date'])}</p>
    <div class="tablewrap"><table class="slate">
      <thead><tr><th class="num">Chance</th><th>Game</th><th>Site</th><th>Why</th></tr></thead>
      <tbody>{''.join(rows)}</tbody></table></div>
  </section>"""

    blocks = "".join(forecast_block(w) for w in g["weeks"])
    sim_ref = '<a href="simulation.html">season simulation</a>' if payload.get("sim") else "season simulation"

    sites = "".join(f"""<tr><td class="opp">{chip(s['team'])}</td>
          <td class="num prob"><span class="pbar" style="width:{s['hostsAtLeastOnce'] * 110:.0f}px"></span>{_pct(s['hostsAtLeastOnce'])}</td>
          <td class="num">{_pct(s['appearsAtLeastOnce'])}</td></tr>""" for s in g["sites"])

    an = g["armyNavy"]
    last_known = max((a["week"] for a in g["announced"]), default=0)
    visits: dict[str, int] = {}
    for a in g["announced"]:
        for t in a["teams"]:
            visits[t] = visits.get(t, 0) + 1
    repeats = [t for t, n in visits.items() if n > 1]
    repeat_text = (f" &mdash; it has already been back to {', '.join(esc(t) for t in repeats[:-1])}{' and ' if len(repeats) > 1 else ''}{esc(repeats[-1])} this year"
                   if repeats else "")
    misses = [c["week"] for c in g["check"] if c["rank"] is None or c["rank"] > 3]
    if misses:
        miss_text = ("Week " if len(misses) == 1 else "Weeks ") + " and ".join(str(w) for w in misses) + (
            " is the current example." if len(misses) == 1 else " are the current examples.")
    else:
        miss_text = "None of the announced weeks so far has been a surprise to it."
    check_rows = "".join(f"""<tr><td class="wk">{c['week']}</td><td class="opp">{' at '.join(chip(t) for t in ([x for x in c['teams'] if x != c['host']] + [c['host']]))}</td>
          <td class="num">{_pct(c['probability']) if c['probability'] is not None else '&ndash;'}</td>
          <td class="num">{('#' + str(c['rank'])) if c['rank'] else '&ndash;'}</td>
          <td>{chip(c['favourite']['away'])} <span class="muted">at</span> {chip(c['favourite']['home'])} <span class="muted">({_pct(c['favourite']['probability'])})</span></td></tr>"""
                           for c in g["check"] if c["host"])

    import math

    coef = dict(zip(g["model"]["features"], g["model"]["coefficients"]))
    halving = math.exp(coef["worst_rank"] * math.log(2))
    both25 = math.exp(coef["both_top25"])
    loss_cut = 1.0 - math.exp(coef["losses"])
    grade = ""
    if choice and ahead:
        grade = f"""
  <p>Two tests, both on 2014&ndash;2025 (2020 aside). <strong>Given the real rankings</strong> for each week, the model picks
  the actual GameDay game out of about {choice['meanCandidates']:.0f} candidates first {choice['candidateSets'][choice['chosen']]['top1']:.0%} of the
  time and in its top three {choice['candidateSets'][choice['chosen']]['top3']:.0%} of the time, leaving each season out in turn (a coin toss
  among the games would be {choice['baseline']['uniformTop1']:.0%}). <strong>The harder test</strong> is this page's job: stand at the end of Week 3, simulate the rest of the
  season, and forecast every stop from Week 8 to the championship game. Across {ahead['stops']} stops, the real site was the model's first choice {ahead['top1']:.0%} of the time,
  in its top three {ahead['top3']:.0%}, and in its top five {ahead['top5']:.0%}.</p>"""

    body = f"""
  <article class="prose wide">
  <h1>Where will College GameDay be?</h1>
  <p class="lead">A guess, for fun. ESPN announces each week's location the Monday before, so as of Week {g['week']} the show's stops through Week {last_known} are known.
  Here is the rest, from what has won the show's attention in past seasons and what the season simulation says about who will be ranked where.
  Not an official anything, and not a ranking.</p>

  <h2>Confirmed</h2>
  <div class="tablewrap"><table class="slate"><thead><tr><th>Wk</th><th>Date</th><th>Game</th><th>Site</th></tr></thead>
    <tbody>{confirmed}</tbody></table></div>

  <h2>The forecast</h2>
  <p class="hint">Each row is a game's chance of being the one, out of every game that week, averaged over {g['sims']:,} simulated
  seasons. The rankings in those seasons come from the same résumé-heavy blend the {sim_ref}
  uses for the committee. About {g['model']['otherRate']:.0%} of the time, every week, the show goes somewhere that is not an FBS game at all.</p>
  {blocks}

  <h2>Who is likely to host from here</h2>
  <div class="tablewrap"><table><thead><tr><th>Team</th><th class="num">Hosts at least once, Weeks 8&ndash;14</th>
    <th class="num">Plays in a stop at least once</th></tr></thead><tbody>{sites}</tbody></table></div>

  <h2>Army&ndash;Navy</h2>
  <p>GameDay went to Army&ndash;Navy every year from 2014 to 2021 and has not since. We give it about {_pct(an['estimate'])}
  for the December game, weighted toward the last four years, in which it went {an['lastFour']} times.</p>

  <h2>How it works, and how well</h2>
  <p>The model is a choice among the week's games. Each game gets a score from a handful of things about the two teams, and its chance is its
  share of the week's total. The big ones: how highly the <em>worse</em> of the two teams ranks, where each doubling of its rank roughly
  {'halves' if 0.4 < halving < 0.6 else 'cuts'} a game's chances (to {halving:.0%} of what they were); whether both teams are in the top 25, which makes a game about
  {both25:.1f} times as likely; and losses, each of which, between the two teams, cuts its chances by about {loss_cut:.0%}. Smaller: how the teams ranked last season, and
  how often GameDay has wanted them lately, which is a fair definition of a brand.</p>
  <p>What did not help, once the rankings were known: whether GameDay had already been to the host this season, how recently the host had hosted, how
  close the game is expected to be, and whether the host is in the SEC or Big Ten. The rule of thumb that the show does not come back to the same place is not visible
  in the picks{repeat_text}.</p>{grade}
  <p><strong>Where it has been wrong:</strong> games that are big for reasons the ratings cannot see. {miss_text}</p>
  <div class="tablewrap"><table><thead><tr><th>Wk</th><th>Announced</th><th class="num">Our chance</th><th class="num">Our rank</th><th>Our first choice</th></tr></thead>
    <tbody>{check_rows}</tbody></table></div>
  <p class="muted">Those weeks were announced before this page existed and were not used to fit anything, which makes them the one live test. Early-season stops
  lean on preseason reputation more than the model does; it was built for Week 8 on. Past locations from NCAA.com's history of the show; announcements from ESPN.</p>
  </article>"""
    return page(f"College GameDay forecast — MRI {season_text(payload)}", body, payload,
                description="Where will ESPN's College GameDay be? A forecast for the weeks not yet announced.")


def dt_label(iso: str) -> str:
    import datetime as _dt

    return _dt.date.fromisoformat(iso).strftime("%b %-d")

def method_page(payload: dict) -> str:
    if chrome_for(payload).sport == "basketball":
        return bb_method_page(payload)
    body = f"""
  <article class="prose">
  <h1>Method</h1>

  <p>MRI is a computer rating system for college football. It ran as an Excel
  workbook from 2000 to 2019 and was rebuilt in {payload['season']}. Two ratings are
  published: the original formula, unchanged, and its replacement.</p>

  <h2>MRI Classic</h2>
  <p>The 2018 formula, frozen. Each game earns a team credit weighted by how good
  its opponent was:</p>
  <pre>win   →  min( 35, margin) × OppWin%  × OppOppWin%
loss  →  max(−35, margin) × OppLoss% × OppOppLoss%</pre>
  <p>and the season rating adds those up alongside win percentage and four
  statistical z-scores:</p>
  <pre>MRI = 25 × Win%  +  25 × OppWin%  +  10 × OppOppWin%  +  Σ game credit
    +  5 × z(RushYds/G)  +  5 × z(PassYds/G)
    +  7 × −z(YdsAllowed/G)  +  3 × z(TurnoverMargin)</pre>
  <p>The Python port reproduces every rating published across the 2003&ndash;2019
  workbooks to within 5&times;10<sup>−13</sup>, with identical top-25 ordering in all
  seventeen seasons. It is kept for continuity, and because a baseline you cannot
  beat is a baseline worth keeping.</p>

  <h2>MRI 2.0</h2>
  <p>Classic measures schedule strength as opponents' win percentage times their
  opponents' win percentage. That treats a 9&ndash;3 Sun Belt team and a 9&ndash;3 SEC team as
  equal tests, because win percentage does not know who anyone played.</p>
  <p>MRI 2.0 turns every game into one equation &mdash;</p>
  <pre>margin  =  rating(home)  −  rating(away)  +  home field</pre>
  <p>&mdash; and solves the whole season at once. A team's rating then depends on its
  opponents' ratings, which depend on theirs, all the way down. Strength of schedule
  stops being a separate statistic and becomes a property of the solution.</p>
  <p>Four further differences:</p>
  <ul>
    <li><strong>Margin is compressed, not capped.</strong> Classic truncates at ±35, so a
    34-point win and a 60-point win score the same. Here margin passes through a curve
    that is near-identity inside two touchdowns and flattens beyond it.</li>
    <li><strong>Ratings are pulled toward a prior</strong> carried from last season, so the
    first month is signal rather than noise. The pull fades on its own as games accumulate.</li>
    <li><strong>Home field is estimated</strong> from the games themselves rather than assumed,
    and neutral sites are excluded. It currently sits at {payload['homeField']:.1f} points.</li>
    <li><strong>FCS opponents are rated individually.</strong> Classic pooled every non-FBS
    team into a single 7&ndash;105 entity, so beating North Dakota State and beating Mercer
    scored identically.</li>
  </ul>

  <h2>Two numbers</h2>
  <p><strong>Power</strong> answers how good a team is, in points against an average FBS
  team, so the gap between two teams is a predicted spread. <strong>R&eacute;sum&eacute;</strong>
  answers what a team has earned, measured as wins above what an average team would have
  managed against the same schedule at the same sites. Classic conflated these, which is
  why it was hard to bet with.</p>

  <h2>Does it work</h2>
  <p>Tested by walking forward through all seventeen archived seasons: fit on the games
  played so far, score the games that come next, never on data it has seen.</p>
  <table class="compare">
    <thead><tr><th></th><th>MRI 2.0</th><th>MRI Classic</th></tr></thead>
    <tbody>
      <tr><td>Straight-up accuracy</td><td><strong>73.6%</strong></td><td>72.0%</td></tr>
      <tr><td>Margin error</td><td>13.0 pts</td><td>n/a</td></tr>
      <tr><td>Brier score</td><td>0.177</td><td>n/a</td></tr>
    </tbody>
  </table>
  <p>MRI 2.0 wins 13 of 17 seasons. Its hyperparameters were searched on 2003&ndash;2013
  and the margin is reported on 2014&ndash;2019, which took no part in the search.</p>

{_prior_method_section(payload)}
{_sim_method_section(payload)}
  <h2>What it cannot do</h2>
  <p>A margin error near 13 points is roughly where closing betting spreads sit. That is
  the honest signal that this model should not be expected to beat a closing line. If it
  has an edge anywhere it is against opening numbers, and that claim will be backtested
  and reported here before it is acted on &mdash; including if the answer is no.</p>
  </article>"""
    return page(f"Method — MRI {season_text(payload)}", body, payload,
                description="How the MRI rating system works, and how well it does.")


def bb_betting_page(payload: dict, betting: dict, board: dict) -> str:
    """The basketball analysis view, led by the verdict rather than the card.

    Football's betting page opens with its finding and then shows a flagged
    list, because football's backtest found a live signal in closing line value.
    Basketball's found nothing at all, so there is no flagged list here and
    nothing on the page is filtered to resemble one. The disagreements are shown
    because they are interesting; the page says plainly that they have not made
    money, above the table rather than below it.
    """
    espn = betting.get("espn_headline") or {}
    dk = betting.get("dk_headline") or {}
    break_even = espn.get("breakEven", 0.5238)

    def bucket_rows(records, with_clv=False):
        out = []
        for row in records or []:
            if row["edge"] == "ALL":
                continue
            clv = f"{row['clv']:+.2f}" if with_clv and row.get("clv") is not None else "&ndash;"
            out.append(f"""
      <tr>
        <td>{esc(row['edge'])}</td>
        <td class="num">{row['bets']}</td>
        <td class="num {'over' if row['ats'] >= break_even else 'under'}">{row['ats']:.1%}</td>
        <td class="num">{row['units']:+.1f}</td>
        <td class="num muted">{clv}</td>
      </tr>""")
        return "".join(out)

    seasons = espn.get("atsBySeason") or {}
    season_rows = "".join(
        f"<tr><td>{esc(k)}</td><td class=\"num {'over' if v >= break_even else 'under'}\">{v:.1%}</td></tr>"
        for k, v in sorted(seasons.items())
    )

    games = board.get("disagreements") or []
    if games:
        card = "".join(f"""
      <tr>
        <td class="wk">{esc(g['day'][5:])}</td>
        <td class="opp">{esc(g['away'])} {'vs' if g['neutral'] else 'at'} {esc(g['home'])}</td>
        <td class="num">{g['predicted']:+.1f}</td>
        <td class="num">{g['market']:+.1f}</td>
        <td class="num perf {'over' if g['edge'] > 0 else 'under'}">{g['edge']:+.1f}</td>
        <td class="opp">{esc(g['side'])}</td>
      </tr>""" for g in games[:25])
        card_block = f"""
  <div class="tablewrap"><table>
    <thead><tr><th>Date</th><th>Game</th><th class="num">Model</th><th class="num">Market</th>
    <th class="num">Gap</th><th>Model's side</th></tr></thead>
    <tbody>{card}</tbody>
  </table></div>
  <p class="note">{board['priced']} of the next {len(board['games'])} games carry a DraftKings
  number. Games involving a team the model has barely seen are left out entirely, because
  the largest gaps in November are the model's ignorance rather than its opinion.</p>"""
    else:
        card_block = """
  <p class="empty">No games in the next few days. This fills in during the season.</p>"""

    body = f"""
  <article class="prose">
  <h1>Model versus market</h1>

  <p class="lede"><strong>The model does not beat the market, and this page is not a
  card to bet.</strong> It is published because the comparison is interesting and
  because a system that only shows you its good weeks is not telling you anything.</p>

  <h2>The market is the better predictor</h2>
  <table class="compare">
    <thead><tr><th></th><th>MRI 2.0</th><th>Market</th></tr></thead>
    <tbody>
      <tr><td>Mean margin error, {espn.get('games', 0):,} games (ESPN Bet)</td>
        <td>{espn.get('modelMae', 0):.2f} pts</td><td><strong>{espn.get('marketMae', 0):.2f} pts</strong></td></tr>
      <tr><td>Mean margin error, {dk.get('games', 0):,} games (DraftKings)</td>
        <td>{dk.get('modelMae', 0):.2f} pts</td><td><strong>{dk.get('marketMae', 0):.2f} pts</strong></td></tr>
      <tr><td>Picks the winner outright</td>
        <td>{espn.get('modelWinner', 0):.1%}</td><td><strong>{espn.get('marketWinner', 0):.1%}</strong></td></tr>
    </tbody>
  </table>
  <p>The two numbers correlate at {espn.get('correlation', 0):.2f}, so most of the time they
  agree. Where they disagree it is usually the model that is wrong, and half a point of
  margin error is the whole reason there is nothing to bet here.</p>

  <h2>Against the closing line, by season</h2>
  <div class="tablewrap"><table>
    <thead><tr><th>Season</th><th class="num">ATS</th></tr></thead>
    <tbody>{season_rows}</tbody>
  </table></div>
  <p>Break-even at &minus;110 is {break_even:.2%}. Four seasons, never close, no trend.
  On DraftKings over 2025&ndash;26 it is {dk.get('ats', 0):.1%} across
  {dk.get('games', 0):,} games, which is worse still.</p>

  <h2>By size of disagreement</h2>
  <p class="hint">If the model knew something the market did not, the buckets where it
  disagrees most should be the ones that win. Against ESPN Bet's closing number:</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Gap</th><th class="num">Bets</th><th class="num">ATS</th>
    <th class="num">Units</th><th class="num">CLV</th></tr></thead>
    <tbody>{bucket_rows(betting.get('espn_close'))}</tbody>
  </table></div>
  <p>One bucket clears significance: gaps of 15 points or more went 54&ndash;25.
  There are 79 of those in 16,954 games, 42 of them in the first season tested, and
  reading them is the end of the story &mdash; they are November mismatches against
  opponents the model has barely seen. In one, the model favoured Mississippi Valley
  State by four where the market had Hawai'i by 25.5. It won the bet while being wrong
  by 22 points. The bucket is measuring the model's own failures, and it does not
  reappear on DraftKings.</p>

  <h2>The one faint signal</h2>
  <p>Betting the opening number rather than the close, mean closing line value is
  positive and grows with the size of the disagreement &mdash; the market tends to drift
  toward this model's side after the opener. That is a reason to keep measuring, not a
  reason to bet, and it is the same shape of hint football produced.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Gap</th><th class="num">Bets</th><th class="num">ATS</th>
    <th class="num">Units</th><th class="num">CLV</th></tr></thead>
    <tbody>{bucket_rows(betting.get('dk_open'), with_clv=True)}</tbody>
  </table></div>
  <p class="note">DraftKings openers, 2025&ndash;26.</p>

  <h2>Where they disagree now</h2>
  {card_block}
  </article>"""
    return page(f"Model versus market — MRI basketball {season_text(payload)}", body, payload,
                description="Where the MRI basketball model disagrees with the market, "
                            "and why that has not made money.")



def _bb_prior_section(payload: dict) -> str:
    model = payload.get("priorModel")
    if not model:
        return ""
    backtest = payload.get("priorBacktest")
    state = payload.get("priorState") or {}
    rmse = model.get("outOfSampleRmse") or {}
    table = ""
    if backtest:
        rows = "".join(
            f"""<tr><td>{esc(label)}</td><td class="num">{e['old']['games']:,}</td><td class="num">{e['old']['mae']:.2f}</td>
            <td class="num">{e['before rosters']['mae']:.2f}</td><td class="num">{e['with a roster']['mae']:.2f}</td>
            <td class="num">{e['with a roster']['gain']:+.2f} &plusmn;{e['with a roster']['gainError']:.2f}</td></tr>"""
            for label, e in backtest["bins"].items())
        table = f"""
  <p>Every game of {backtest['seasons'][0]}&ndash;{backtest['seasons'][1]} (2021, the COVID season, chained but not scored), predicted from only the earlier
  games of its own season, in slices of the season (the average miss, in points, of each prior; the last column is what a roster gains over the old rule). Each prior is chained properly, starting every season from the previous season's final ratings under the same
  method, and the new priors' coefficients are fitted leaving the predicted season out.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Season so far</th><th class="num">Games</th><th class="num" title="Average miss, in points, of the old prior">Old</th>
    <th class="num" title="Average miss with the version that needs no roster">Before rosters</th>
    <th class="num" title="Average miss with returning and incoming production">With roster</th>
    <th class="num" title="Points gained with a roster, and the margin of error">Gain</th></tr></thead>
    <tbody>{rows}</tbody></table></div>
  <p>The gain is real and modest, and it fades. Basketball plays about 6,000 games among 365 teams, so by the time a fifth of the season is gone the games have
  taught the model most of what the prior was guessing. Football, with a fraction of the games, keeps the benefit longer.</p>"""
    now = ""
    if state:
        season_label = f"{state['season'] - 1}&ndash;{str(state['season'])[2:]}"
        if state.get("mode") == "roster":
            now = (f"Rosters for {season_label} are posted for {state['teamsWithRosters']} of {state['teams']} teams, "
                   "so this season's preseason ratings use the roster.")
        else:
            now = (f"Rosters for {season_label} have not been posted yet, so the preseason ratings use the version "
                   "that does not need them. The site switches to the roster version by itself as the schools post their rosters, one team at a time.")
    return f"""
  <h2 id="priors">Preseason priors</h2>
  <p>A season starts from a prior, and basketball's was last season's rating pulled 35% of the way to average for everyone. Rosters turn over faster in college
  basketball than anywhere else, so that treats a team that lost its whole rotation like one that kept it. The prior is now a regression on last season's rating
  and what is known about the roster: <strong>returning production</strong> (the share of last season's win shares still on the roster), <strong>incoming
  production</strong> (what the newcomers did last season somewhere else, which is the transfer portal) and the <strong>freshman class</strong>.</p>
  <p>Rosters for a coming season are posted late. Before they are, the prior uses only what last season and the draft already imply: how many of last year's
  minutes belong to players in their fourth year or later or headed to the NBA draft, and the freshman class. {now}</p>
  <p>Scored on how good teams turned out to be, leaving each season out in turn, a season's final rating is missed by {rmse.get('old', {}).get('all', 0):.1f} points
  under the old rule, {rmse.get('before rosters', {}).get('all', 0):.1f} with the before-rosters version and {rmse.get('with a roster', {}).get('all', 0):.1f} with a roster. Part of the gap
  to the old rule is only its weight on last season being tuned for something else; the fair comparison is a last-season-only refit at
  {rmse.get('last season only', {}).get('all', 0):.1f}. The game-by-game test below is the one to trust.</p>{table}
  <p><strong>Two adjustments, found by testing and not by argument.</strong> The new prior keeps the division's level where the old one had it, because a
  level that is a point off lands on every game against a non-Division I opponent, which is most of November. And its spread is pulled in to
  {model['tighten']:.0%}: at full spread it is worse than the old prior, at that setting it is better in every part of the season, and at 50% it starts to give the gain back.</p>"""

def bb_method_page(payload: dict) -> str:
    """The basketball method page.

    Shorter than football's on purpose. Football has seventeen validated
    seasons, a frozen formula to compare against and a betting backtest behind
    it. Basketball has four validated workbooks and six chained seasons, and
    saying more than that would be borrowing football's credibility.
    """
    body = f"""
  <article class="prose">
  <h1>Method</h1>

  <p>MRI is a computer rating system. Ben ran it as an Excel workbook for college
  football and college basketball; the basketball workbooks stop at 2019&ndash;20.
  This is the same idea rebuilt, and carried forward through the seasons in
  between.</p>

  <h2>The rating</h2>
  <p>Every game becomes one equation &mdash;</p>
  <pre>margin  =  rating(home)  −  rating(away)  +  home court</pre>
  <p>&mdash; and the whole season is solved at once. A team's rating depends on its
  opponents' ratings, which depend on theirs, all the way down, so strength of
  schedule is not a separate statistic bolted on afterwards. It is what the
  solution is made of.</p>
  <ul>
    <li><strong>Margin is compressed.</strong> A 30-point win is worth more than a
    20-point win and not very much more, through a curve that is near-identity
    inside a normal result and flattens past it. Nothing is gained by running up
    a score.</li>
    <li><strong>Ratings are pulled toward a preseason estimate</strong> built from last
    season's rating and who is on the roster, so November means something. The pull
    fades on its own as games accumulate, and it is gone well before conference play.</li>
    <li><strong>Home court is estimated</strong> from the games rather than assumed,
    and neutral sites are excluded. It sits at {payload['homeField']:.1f} points here, and has
    run between 2.7 and 3.3 across the six rebuilt seasons.</li>
    <li><strong>Non-Division-I opponents are rated individually</strong> rather than
    pooled, so a November exhibition against a good one is not the same result as
    a November exhibition against a bad one.</li>
  </ul>

  <h2>Two numbers</h2>
  <p><strong>Power</strong> answers how good a team is, in points against an average
  Division I team, so the gap between two teams is a predicted spread.
  <strong>R&eacute;sum&eacute;</strong> answers what a team has earned, measured as wins above
  what an average team would have managed against the same schedule at the same
  venues. They are different questions and they are published separately.</p>

{_bb_prior_section(payload)}
  <h2>What is and is not established</h2>
  <p>The original formula was ported and checked against the surviving workbooks:
  it reproduces every published rating in 2012&ndash;13, 2017&ndash;18, 2018&ndash;19
  and 2019&ndash;20 to floating-point tolerance. That is the part that is proven.</p>
  <p>Ratings since then are a chain: 2019&ndash;20 primes 2020&ndash;21, which primes
  the next, through to now. Walking forward within each season &mdash; fit on the
  games played, score the games that come next &mdash; the model calls about 70% of
  games correctly with a mean margin error near 9.3 points.</p>
  <p>Its settings were searched over 80 combinations and the search found nothing
  that beat the starting guesses on seasons it had not seen. That is reported here
  because it is true: at this many games per team the knobs stop mattering, and a
  tuning exercise that claims a win it did not get is worth less than one that
  admits it.</p>

  <h2>What it cannot do</h2>
  <p>A margin error near 9.3 points is about where college basketball closing spreads
  sit. As with football, that is the honest signal that this should not be expected
  to beat a closing line. There is no betting page for basketball and there will not
  be one unless a backtest earns it.</p>
  </article>"""
    return page(f"Method — MRI basketball {season_text(payload)}", body, payload,
                description="How the MRI basketball rating works, and what it has been shown to do.")


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def content_digest(payload: dict) -> str:
    """Fingerprint everything about the payload except when it was built."""
    material = {k: v for k, v in payload.items() if k not in {"generated", "digest"}}
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def _settle_timestamp(payload: dict, out_dir: Path, name: str = "site.json") -> None:
    """Keep the previous timestamp when nothing but the clock has moved.

    The footer stamp is rendered into all 157 pages, so a build that changed
    nothing else still produced a 157-file diff and a commit every time the
    scheduled job ran. Carrying the old timestamp forward when the fingerprint
    matches makes the build idempotent - and makes the stamp mean "when these
    ratings last changed", which is the more useful claim anyway.
    """
    payload["digest"] = content_digest(payload)
    existing = out_dir / name
    if not existing.exists():
        return
    try:
        previous = json.loads(existing.read_text())
    except (OSError, json.JSONDecodeError):
        return
    if previous.get("digest") == payload["digest"] and previous.get("generated"):
        payload["generated"] = previous["generated"]


BRAND_WEB = Path(__file__).resolve().parents[3] / "site" / "assets" / "web"


def _copy_brand(site_root: Path) -> list[Path]:
    """Put the logo, icons and share card where the pages reference them.

    Copied, not generated: ``scripts/build_brand_assets.py`` derives these from
    the delivered art and commits the result, so the site build moves bytes and
    stays byte-identical run to run rather than depending on whichever Pillow
    happens to be installed to encode a PNG the same way twice.

    The favicon goes to the root as well as to assets/, because browsers and
    feed readers ask for /favicon.ico by habit whatever the page declares.
    """
    if not BRAND_WEB.is_dir():
        return []
    assets = site_root / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    out = []
    for source in sorted(BRAND_WEB.iterdir()):
        if source.suffix.lower() not in (".png", ".ico"):
            continue
        for target in ([assets / source.name, site_root / source.name]
                       if source.name == "favicon.ico" else [assets / source.name]):
            data = source.read_bytes()
            # Only write when it differs, so an unchanged logo does not restamp
            # the file and show up as a change in the daily commit.
            if not target.exists() or target.read_bytes() != data:
                target.write_bytes(data)
            out.append(target)
    return out


def build(payload: dict, site_root: Path, *, publish_details: bool = True) -> list[Path]:
    """Render one sport's pages.

    ``site_root`` is docs/ itself, not the sport's folder - the stylesheet, the
    CNAME and the Jekyll opt-out belong to the site rather than to either sport,
    and writing two copies of the stylesheet is how they drift apart. The sport's
    own pages go under ``site_root / chrome.home``.

    ``publish_details`` controls whether the per-team schedule detail is written
    into the published JSON. It is already rendered into every team page, and for
    basketball it is 4MB that would be recommitted on every weekly run, so the
    basketball build leaves it out.
    """
    chrome = chrome_for(payload)
    out_dir = site_root / chrome.home if chrome.home else site_root
    json_name = f"{chrome.sport}.json" if chrome.home else "site.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    _settle_timestamp(payload, out_dir, json_name)
    (out_dir / "team").mkdir(exist_ok=True)
    (out_dir / "conference").mkdir(exist_ok=True)

    written = []

    def write(path: Path, content: str):
        path.write_text(content)
        written.append(path)

    # Site-wide, written once at the root whichever sport is building. Without
    # .nojekyll, GitHub Pages runs the output through Jekyll, which skips files
    # and directories whose names begin with an underscore.
    site_root.mkdir(parents=True, exist_ok=True)
    write(site_root / ".nojekyll", "")
    if CUSTOM_DOMAIN:
        write(site_root / "CNAME", CUSTOM_DOMAIN + "\n")
    write(site_root / "styles.css", STYLES)
    written.extend(_copy_brand(site_root))

    write(out_dir / "index.html", rankings_page(payload))
    write(out_dir / "conferences.html", conferences_index(payload))
    write(out_dir / "archive.html", archive_page(payload))
    write(out_dir / "method.html", method_page(payload))
    if payload.get("betting") and payload.get("board"):
        renderer = bb_betting_page if chrome.sport == "basketball" else betting_page
        write(out_dir / "betting.html", renderer(payload, payload["betting"], payload["board"]))

    # Football only, and only when the build produced them.
    for key, name, renderer in (("sim", "simulation", simulation_page), ("slate", "slate", slate_page),
                                ("gameday", "gameday", gameday_page)):
        if payload.get(key):
            write(out_dir / f"{name}.html", renderer(payload))
            write(out_dir / f"{name}.json", json.dumps(payload[key], indent=2))
    if payload.get("record"):
        write(out_dir / "record.json", json.dumps(payload["record"], indent=2))

    if payload.get("seasons"):
        (out_dir / "season").mkdir(exist_ok=True)
        write(out_dir / "seasons.html", seasons_index(payload))
        for entry in payload["seasons"]:
            write(out_dir / "season" / f"{entry['season']}.html", season_page(entry, payload))

    # The season tables are rendered into their own pages; carrying them in the
    # published JSON as well would roughly double it for no reader.
    drop = {"seasons", "history", "gamelogs", "sim", "slate", "record", "simBacktest",
            "priorModel", "priorBacktest", "priorState", "gameday", "gamedayBacktest"} \
        | (set() if publish_details else {"details"})
    published = {k: v for k, v in payload.items() if k not in drop}
    write(out_dir / json_name, json.dumps(published, indent=2))

    logs = payload.get("gamelogs") or {}
    history = payload.get("history") or {}
    for team in payload["teams"]:
        write(out_dir / "team" / f"{slug(team['team'])}.html", team_page(team, payload))
        seasons_with_log = [
            row for row in history.get(team["team"], [])
            if (logs.get(row["season"]) or {}).get(team["team"])
        ]
        if seasons_with_log:
            folder = out_dir / "team" / slug(team["team"])
            folder.mkdir(exist_ok=True)
            for row in seasons_with_log:
                write(folder / f"{row['season']}.html",
                      team_season_page(team, row, logs[row["season"]][team["team"]], payload))
    for conference in payload["conferences"]:
        name = conference["conference"]
        write(out_dir / "conference" / f"{slug(name)}.html", conference_page(name, payload))

    return written


STYLES = """
:root {
  --surface:#1a1a19; --plane:#0d0d0d; --primary:#ffffff; --secondary:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835;
  --up:#0ca30c; --down:#d03b3b; --series:#AB011B;
  color-scheme: dark;
}
@media (prefers-color-scheme: light) {
  :root:not([data-theme="dark"]) {
    --surface:#fcfcfb; --plane:#f9f9f7; --primary:#0b0b0b; --secondary:#52514e;
    --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
    --up:#006300; --down:#d03b3b; --series:#AB011B;
    color-scheme: light;
  }
}
* { box-sizing: border-box; }
/* The browser hides [hidden] with display:none from its own stylesheet, which
   any author rule setting display beats. .row is display:flex, so every row the
   conference filter hid stayed on screen and the filter did nothing at all.
   This is the rule that makes the attribute mean what it says. */
[hidden] { display: none !important; }
html { -webkit-text-size-adjust: 100%; }
body { margin:0; background:var(--plane); color:var(--primary);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif; line-height:1.5; }
a { color:inherit; }
.wrap { max-width:1080px; margin:0 auto; padding-left:18px; padding-right:18px; }

header.site { border-bottom:1px solid var(--grid); background:var(--surface); }
.bar { display:flex; align-items:center; gap:20px; flex-wrap:wrap; padding-block:14px; }
/* The mark is the lockup image now. Height is fixed and width follows, so the
   art keeps its proportions whatever the file turns out to be; the width and
   height attributes on the img are only there to reserve the box before it
   loads and stop the header jumping. The wordmark is white, so the light theme
   gets a recoloured file through <picture> rather than the same file on a
   background it disappears into. */
.mark { display:inline-flex; align-items:center; text-decoration:none; }
.mark img { height:38px; width:auto; display:block; }
/* The sport switch sits beside the wordmark rather than inside the nav, because
   it changes which site you are on and the nav changes which page. Segmented so
   it reads as a choice between two, with the current one filled rather than
   merely coloured - colour alone would not survive a greyscale print or a
   colourblind reader. */
.sports { display:flex; border:1px solid var(--grid); border-radius:999px; overflow:hidden; }
.sports a { font-size:12px; font-weight:600; padding:4px 12px; text-decoration:none;
  color:var(--muted); white-space:nowrap; }
.sports a:hover { color:var(--primary); }
.sports a.on { background:var(--primary); color:var(--surface); }
header nav { display:flex; gap:16px; flex:1; flex-wrap:wrap; }
header nav a { font-size:13px; color:var(--secondary); text-decoration:none; }
header nav a:hover { color:var(--primary); }
.stamp { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:0.08em; }

main { padding-block:26px 40px; }
.grid { display:grid; grid-template-columns:1.55fr 1fr; gap:18px; align-items:start; }
/* Grid children default to min-width:auto, so one wide row drags the whole
   track past the viewport. This is the fix for that. */
.grid > * { min-width: 0; }
.panel { background:var(--surface); border:1px solid var(--grid); border-radius:12px;
  padding:16px; margin-bottom:18px; }
aside .panel:last-child { margin-bottom:0; }
.panelhead { display:flex; align-items:center; gap:12px; flex-wrap:wrap; justify-content:space-between; }
.ptitle { font-size:11px; text-transform:uppercase; letter-spacing:0.1em;
  color:var(--muted); margin:0 0 10px; font-weight:600; }
.panelhead .ptitle { margin-bottom:0; }
.controls { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
.controls select { background:var(--plane); color:var(--primary); border:1px solid var(--axis);
  border-radius:7px; padding:5px 8px; font-size:12px; font-family:inherit; max-width:100%; }
.sr { display:flex; align-items:center; gap:6px; font-size:10px; text-transform:uppercase;
  letter-spacing:0.09em; color:var(--muted); font-weight:600; }
.hint { font-size:12.5px; color:var(--secondary); margin:10px 0 14px; }
.note { font-size:12px; color:var(--secondary); margin:10px 0 0; }
.empty { font-size:13px; color:var(--muted); padding:10px 0; }

ul { list-style:none; margin:0; padding:0; }
.row { display:flex; align-items:center; gap:11px; padding:8px 10px; border-radius:8px;
  border-left:3px solid var(--team); }
.row:hover { background:color-mix(in oklab, var(--team) 10%, transparent); }
.rk { font-size:13px; font-weight:700; width:26px; color:var(--secondary);
  font-variant-numeric:tabular-nums; flex:none; }
.mid { flex:1; min-width:0; }
.top { display:flex; align-items:center; gap:7px; }
.logo { flex:none; object-fit:contain; }
.chip { width:20px; height:20px; border-radius:3px; flex:none; display:inline-block; }
.nm { font-weight:600; font-size:14px; text-decoration:none; white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis; }
.nm:hover { text-decoration:underline; }
.sub { display:block; font-size:11px; color:var(--muted); margin-top:2px; }
.sub-item { margin-left:6px; }
.mv { font-size:10px; font-variant-numeric:tabular-nums; flex:none; }
.mv.up { color:var(--up); } .mv.down { color:var(--down); } .mv.flat { color:var(--muted); }
.sp { flex:none; }
.vals { text-align:right; flex:none; min-width:66px; display:flex; flex-direction:column; }
.pw { font-size:15px; font-weight:700; font-variant-numeric:tabular-nums; }
.rs { font-size:11px; color:var(--muted); font-variant-numeric:tabular-nums; }

.hero { display:flex; align-items:center; gap:14px; }
.heronum { font-size:48px; font-weight:800; line-height:1; letter-spacing:-0.03em; }
.heronm { font-size:15px; font-weight:600; }
.heronm a { text-decoration:none; }
.herosub { font-size:12px; color:var(--secondary); }

.list li { display:flex; justify-content:space-between; gap:10px; padding:7px 0;
  border-bottom:1px solid var(--grid); font-size:13px; }
.list li:last-child { border-bottom:none; }
.gnm { font-weight:600; text-decoration:none; }
.gnm:hover { text-decoration:underline; }
.gv { color:var(--secondary); font-variant-numeric:tabular-nums; white-space:nowrap; }
.gv em { font-style:normal; color:var(--muted); }

.conf li { display:flex; align-items:center; gap:9px; padding:5px 0; font-size:12px; }
.confnm { width:98px; color:var(--secondary); flex:none; text-decoration:none; }
.confnm:hover { color:var(--primary); }
.track { flex:1; height:8px; background:var(--grid); border-radius:4px; overflow:hidden; }
.confbar { display:block; height:8px; background:var(--series); border-radius:0 4px 4px 0; }
.confbar.neg { background:var(--muted); }
.confv { color:var(--muted); font-variant-numeric:tabular-nums; flex:none; width:40px; text-align:right; }

h1 { font-size:28px; letter-spacing:-0.02em; margin:0 0 6px; }
h2 { font-size:17px; margin:26px 0 8px; }
.teamhead { display:flex; align-items:center; gap:14px; padding-bottom:14px;
  border-bottom:3px solid var(--team); margin-bottom:18px; }
.teamsub { color:var(--secondary); font-size:13px; margin:0 0 16px; }
.stats { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:16px; }
.stat { background:var(--surface); border:1px solid var(--grid); border-radius:10px; padding:12px; }
.statl { display:block; font-size:10px; text-transform:uppercase; letter-spacing:0.09em; color:var(--muted); }
.statv { display:block; font-size:26px; font-weight:700; line-height:1.25; }
.statn { display:block; font-size:11px; color:var(--muted); }
.sparkbig { line-height:1; padding:4px 0; }
.highlights { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:10px; margin-bottom:8px; }
.hl { background:var(--surface); border:1px solid var(--grid); border-radius:9px; padding:9px 11px; }
.hll { display:block; font-size:10px; text-transform:uppercase; letter-spacing:0.09em; color:var(--muted); }
.hlv { font-size:13px; font-weight:600; }

.tablewrap { overflow-x:auto; border:1px solid var(--grid); border-radius:10px; background:var(--surface); }
table { width:100%; border-collapse:collapse; font-size:13.5px; }
thead th { text-align:left; font-size:10px; letter-spacing:0.09em; text-transform:uppercase;
  color:var(--muted); font-weight:600; padding:10px 10px 8px; border-bottom:1px solid var(--axis); white-space:nowrap; }
tbody td { padding:8px 10px; border-bottom:1px solid var(--grid); }
tbody tr:last-child td { border-bottom:none; }
.num { text-align:right; font-variant-numeric:tabular-nums; }
th.num { text-align:right; }
.wk, .rec { color:var(--secondary); font-variant-numeric:tabular-nums; }
.site { color:var(--muted); font-size:12px; }
.opp a { text-decoration:none; } .opp a:hover { text-decoration:underline; }
.fcs { font-size:10px; color:var(--muted); border:1px solid var(--axis); border-radius:3px; padding:0 4px; }
.res .w { color:var(--up); font-weight:700; } .res .l { color:var(--down); font-weight:700; }
.perf.over { color:var(--up); } .perf.under { color:var(--down); }
.muted { color:var(--muted); }
.conftable .tm { display:flex; align-items:center; gap:8px; }
.conftable .rule { width:3px; height:18px; border-radius:1px; background:var(--team); flex:none; }
.conftable .tm a { text-decoration:none; } .conftable .tm a:hover { text-decoration:underline; }
.archive td { font-size:12px; white-space:nowrap; }
.archive a { text-decoration:none; }

.confgrid { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; margin-top:16px; }
.confcard { background:var(--surface); border:1px solid var(--grid); border-radius:11px;
  padding:14px; text-decoration:none; display:block; }
.confcard:hover { border-color:var(--series); }
.ccname { display:block; font-weight:600; font-size:14px; }
.ccmean { display:block; font-size:24px; font-weight:700; font-variant-numeric:tabular-nums; }
.ccmeta { display:block; font-size:11px; color:var(--muted); }

.prose { max-width:70ch; }
.prose p, .prose li { color:var(--secondary); font-size:14.5px; }
.prose strong { color:var(--primary); }
.prose pre { background:var(--surface); border:1px solid var(--grid); border-radius:9px;
  padding:12px 14px; overflow-x:auto; font-size:12.5px; color:var(--primary); }
.compare.wide thead, .compare.wide tbody { min-width:430px; }
.compare { background:var(--surface); border:1px solid var(--grid); border-radius:10px;
  margin:14px 0; display:block; overflow-x:auto; max-width:100%; }
.compare thead, .compare tbody { display:table; width:100%; table-layout:fixed; }
.compare td { color:var(--secondary); } .compare td strong { color:var(--primary); }
/* Numbers sit right; their headings were still sitting left, so neither column
   lined up with the thing it labelled. */
.compare th:not(:first-child), .compare td:not(:first-child) { text-align:right; }
/* Season labels are two-part for basketball ("2025-26") and were breaking over
   two lines in a narrow first column. */
td.rk { white-space:nowrap; }
.of { color:var(--muted); font-weight:400; font-size:11px; }
/* Champion marks. Gold reads as "won something" without needing a legend, and
   every use of it is beside a number or a year, never carrying meaning alone. */
.trophy { color:#c8961e; vertical-align:-2px; margin-right:4px; }
.titles { display:flex; gap:12px; align-items:center; background:var(--surface);
  border:1px solid var(--grid); border-left:3px solid #c8961e; border-radius:10px;
  padding:12px 16px; margin-bottom:18px; }
.titles > div { display:flex; flex-direction:column; gap:2px; min-width:0; }
.titlesl { font-size:12px; font-weight:700; text-transform:uppercase;
  letter-spacing:0.08em; color:#c8961e; }
.titlesy { font-size:13px; color:var(--secondary); }
.titlesy a { color:var(--secondary); }
.wonchip { color:#c8961e; font-weight:600; }
tr.wonit td { background:color-mix(in srgb, #c8961e 7%, transparent); }
.rankchart { margin-top:14px; }
.rankchart svg { width:100%; height:auto; display:block; }
.rankchart .ct { font-size:10px; fill:var(--muted); }
.rankchart .pt { fill:var(--series); }
.rankchart .champ { fill:#c8961e; stroke:var(--surface); stroke-width:1.5; }
.sysname { font-size:11px; white-space:nowrap; }
.compare tr.total td { color:var(--primary); font-weight:700; border-top:1px solid var(--axis); }
.verdict { border-left:3px solid var(--down); background:var(--surface); border-radius:0 10px 10px 0;
  padding:14px 16px; margin:18px 0; }
.vlead { font-size:18px; font-weight:700; color:var(--primary); margin:0 0 8px; }
/* Specific enough to beat `.compare td`. Colour is supplementary here - every
   value carries its own sign and the break-even figure is stated in the text -
   so this is emphasis, not the encoding. */
.over, td.over, .compare td.over { color:var(--up); }
.under, td.under, .compare td.under { color:var(--down); }

footer.site { border-top:1px solid var(--grid); background:var(--surface);
  padding-block:20px; font-size:12px; color:var(--secondary); }
footer.site p { margin:0 0 6px; }
footer.site .muted { color:var(--muted); }

@media (max-width:860px) { .grid { grid-template-columns:1fr; } }
@media (max-width:620px) {
  .bar { gap:10px 14px; }
  /* Full width so the mark takes its own line above the nav, as the wordmark
     did - but the image inside must not stretch with it, so it stays auto and
     shrinks a little for the narrower bar. */
  .mark { width:100%; }
  .mark img { height:32px; }
  header nav { flex:1 1 auto; gap:14px; }
  .stamp { flex:none; }
  .panelhead { align-items:flex-start; flex-direction:column; gap:10px; }
}
@media (max-width:520px) {
  .stats { grid-template-columns:1fr; }
  .sp { display:none; }
  h1 { font-size:23px; }
  .vals { min-width:60px; }
  .confnm { width:80px; }
}
/* season simulation, slate, record */
.prose.wide { max-width:none; }
.prose h3 { font-size:15px; margin:22px 0 4px; }
.lead { font-size:15.5px; color:var(--secondary); max-width:78ch; }
.grid.two { grid-template-columns:1fr 1fr; }
.prob { white-space:nowrap; }
.pbar { display:inline-block; height:7px; border-radius:2px; background:var(--series); margin-right:7px;
  vertical-align:middle; min-width:1px; }
.pbar.alt { background:var(--secondary); }
.nmcell { display:inline-flex; align-items:center; gap:6px; }
.nmcell a { text-decoration:none; } .nmcell a:hover { text-decoration:underline; }
.sub { font-size:11px; margin-left:6px; }
.rkchip { font-size:11px; color:var(--muted); font-variant-numeric:tabular-nums; }
.chk { display:inline-flex; align-items:center; gap:6px; font-size:12px; color:var(--secondary); }
.simtable td.rk, .slate td.wk, .slate td.perf { white-space:nowrap; }
table.slate tr.flagged td { background:color-mix(in srgb, var(--series) 9%, transparent); }
.confcard a { text-decoration:none; } .confcard a:hover { text-decoration:underline; }
.confcard .ccmeta { font-size:12.5px; color:var(--secondary); line-height:1.7; margin-top:6px; }
table.slate tr.rest td { border-top:none; font-size:12.5px; }
.small { font-size:12px; line-height:1.5; }
.panel.gd { margin-top:14px; }
.watch li { flex-direction:column; align-items:flex-start; gap:3px; }
.mu { display:inline-flex; align-items:center; gap:6px; flex-wrap:wrap; }
@media (max-width:900px) { .grid.two { grid-template-columns:1fr; } }
"""
