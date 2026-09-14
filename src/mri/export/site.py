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

from . import common

SERIES = "#3987e5"
SERIES_LIGHT = "#2a78d6"

# The custom domain. Written into the output as a CNAME file on every build:
# GitHub Pages puts that file in the repo when you set the domain in Settings,
# and since this generator rewrites the whole output directory it would
# otherwise be deleted on the next build and quietly take the domain down.
CUSTOM_DOMAIN = "mri.mira.ski"


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

def sparkline(values, width=58, height=22, color=SERIES) -> str:
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
</head>
<body>
<header class="site">
  <div class="wrap bar">
    <a class="mark" href="{up}index.html">The <em>MRI</em></a>
    {switch}
    <nav>
      <a href="{up}index.html">Rankings</a>
      <a href="{up}conferences.html">Conferences</a>
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
          <span class="sub">{esc(team['conference'])} <span class="sub-item">{team['wins']}&ndash;{team['losses']}</span> {classic_note}</span>
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
    body = "".join(f"""
      <tr>
        <td class="rk"><a href="../season/{r['season']}.html">{esc(r['label'])}</a></td>
        <td class="num"><strong>{r['rank']}</strong><span class="of"> of {r['of']}</span></td>
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
      {len(rows)} rated seasons.{caveat}</p>
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

    body = f"""
  <article class="teampage" style="--team:{esc(team['color'])}">
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
    provenance = (
        "recomputed from the game logs in Ben's own workbooks; "
        f"{len(verified)} of these reproduce the published top 25 exactly"
        if verified else "exactly as Ben published it at the time"
    )
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
    <li><strong>Ratings are pulled toward last season's</strong> so November means
    something. The pull fades on its own as games accumulate, and it is gone
    well before conference play.</li>
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

    write(out_dir / "index.html", rankings_page(payload))
    write(out_dir / "conferences.html", conferences_index(payload))
    write(out_dir / "archive.html", archive_page(payload))
    write(out_dir / "method.html", method_page(payload))
    if payload.get("betting") and payload.get("board"):
        renderer = bb_betting_page if chrome.sport == "basketball" else betting_page
        write(out_dir / "betting.html", renderer(payload, payload["betting"], payload["board"]))

    if payload.get("seasons"):
        (out_dir / "season").mkdir(exist_ok=True)
        write(out_dir / "seasons.html", seasons_index(payload))
        for entry in payload["seasons"]:
            write(out_dir / "season" / f"{entry['season']}.html", season_page(entry, payload))

    # The season tables are rendered into their own pages; carrying them in the
    # published JSON as well would roughly double it for no reader.
    drop = {"seasons", "history"} | (set() if publish_details else {"details"})
    published = {k: v for k, v in payload.items() if k not in drop}
    write(out_dir / json_name, json.dumps(published, indent=2))

    for team in payload["teams"]:
        write(out_dir / "team" / f"{slug(team['team'])}.html", team_page(team, payload))
    for conference in payload["conferences"]:
        name = conference["conference"]
        write(out_dir / "conference" / f"{slug(name)}.html", conference_page(name, payload))

    return written


STYLES = """
:root {
  --surface:#1a1a19; --plane:#0d0d0d; --primary:#ffffff; --secondary:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835;
  --up:#0ca30c; --down:#d03b3b; --series:#3987e5;
  color-scheme: dark;
}
@media (prefers-color-scheme: light) {
  :root:not([data-theme="dark"]) {
    --surface:#fcfcfb; --plane:#f9f9f7; --primary:#0b0b0b; --secondary:#52514e;
    --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
    --up:#006300; --down:#d03b3b; --series:#2a78d6;
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
.mark { font-size:20px; font-weight:800; letter-spacing:-0.02em; text-decoration:none; }
.mark em { font-style:normal; color:var(--series); }
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
.compare thead, .compare tbody { display:table; width:100%; }
.compare td { color:var(--secondary); } .compare td strong { color:var(--primary); }
/* Numbers sit right; their headings were still sitting left, so neither column
   lined up with the thing it labelled. */
.compare th:not(:first-child), .compare td:not(:first-child) { text-align:right; }
/* Season labels are two-part for basketball ("2025-26") and were breaking over
   two lines in a narrow first column. */
td.rk { white-space:nowrap; }
.of { color:var(--muted); font-weight:400; font-size:11px; }
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
  .mark { width:100%; }
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
"""
