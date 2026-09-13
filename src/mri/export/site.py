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

import html
import json
import re
from pathlib import Path

SERIES = "#3987e5"
SERIES_LIGHT = "#2a78d6"


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


def identity_mark(team: dict, size: int = 20) -> str:
    """Logo when we have one, a color chip when we don't. Never color alone -
    the team name is always beside it."""
    if team.get("logo"):
        fallback = (
            "this.replaceWith(Object.assign(document.createElement('span'),"
            f"{{className:'chip',style:'background:{esc(team['color'])}'}}))"
        )
        return (
            f'<img class="logo" src="{esc(team["logo"])}" alt="" width="{size}" height="{size}"'
            f' loading="lazy" onerror="{esc(fallback)}">'
        )
    return f'<span class="chip" style="background:{esc(team["color"])}"></span>'


# --------------------------------------------------------------------------
# page chrome
# --------------------------------------------------------------------------

def page(title: str, body: str, payload: dict, *, depth: int = 0, description: str = "") -> str:
    up = "../" * depth
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{esc(description or 'A computer rating system for college football.')}">
<link rel="stylesheet" href="{up}styles.css">
</head>
<body>
<header class="site">
  <div class="wrap bar">
    <a class="mark" href="{up}index.html">The <em>MRI</em></a>
    <nav>
      <a href="{up}index.html">Rankings</a>
      <a href="{up}conferences.html">Conferences</a>
      <a href="{up}archive.html">Archive</a>
      <a href="{up}method.html">Method</a>
    </nav>
    <div class="stamp">{payload['season']} &middot; Week {payload['week']}</div>
  </div>
</header>
<main class="wrap">{body}</main>
<footer class="site">
  <div class="wrap">
    <p>MRI &mdash; a computer rating system for college football, run 2000&ndash;2019
    and rebuilt for {payload['season']}. Game data from
    <a href="https://collegefootballdata.com">CollegeFootballData</a>.</p>
    <p class="muted">Ratings generated {esc(payload['generated'][:16].replace('T', ' '))} UTC
    &middot; {payload['gamesRated']} games rated &middot; home field {payload['homeField']:.1f} pts</p>
  </div>
</footer>
</body>
</html>"""


# --------------------------------------------------------------------------
# rankings
# --------------------------------------------------------------------------

def rankings_page(payload: dict) -> str:
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

    movers = sorted(teams, key=lambda t: -abs(t["movement"]))[:5]
    mover_rows = "".join(
        f"""<li><a class="gnm" href="team/{slug(t['team'])}.html">{esc(t['team'])}</a>
        <span class="gv">{movement_chip(t['movement'])} to #{t['rank']}</span></li>"""
        for t in movers if t["movement"]
    ) or '<li class="empty">No movement yet &mdash; this is the first rated week.</li>'

    conferences = "".join(_conference_bar(c, payload) for c in payload["conferences"])
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
            <option value="resume">By r&eacute;sum&eacute;</option>
            <option value="classic">By MRI Classic</option>
          </select></label>
        </div>
      </div>
      <p class="hint">Points against an average FBS team &mdash; the gap between two
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

      <section class="panel">
        <p class="ptitle">Where Classic disagrees</p>
        <ul class="list">{gap_rows}</ul>
        <p class="note">MRI 2.0 rank versus the frozen 2018 formula. Classic has no
        preseason prior and weights raw win percentage heavily, so in September an
        unbeaten team that has played nobody rides high.</p>
      </section>

      <section class="panel">
        <p class="ptitle">Biggest movers</p>
        <ul class="list">{mover_rows}</ul>
      </section>

      <section class="panel">
        <p class="ptitle">Conference strength</p>
        <ul class="conf">{conferences}</ul>
        <p class="note">Mean power rating of member teams.</p>
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
    return page(f"MRI — {payload['season']} college football rankings", body, payload,
                description="Opponent-adjusted college football ratings, updated weekly.")


def _conference_bar(conference: dict, payload: dict) -> str:
    widest = max(abs(c["mean"]) for c in payload["conferences"]) or 1
    width = abs(conference["mean"]) / widest * 100
    negative = conference["mean"] < 0
    return (
        f'<li><a class="confnm" href="conference/{slug(conference["conference"])}.html">'
        f'{esc(conference["conference"])}</a>'
        f'<span class="track"><span class="confbar{" neg" if negative else ""}" '
        f'style="width:{width:.0f}%"></span></span>'
        f'<span class="confv">{conference["mean"]:+.1f}</span></li>'
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


def team_page(team: dict, payload: dict) -> str:
    detail = payload["details"].get(team["team"], {"played": [], "upcoming": []})
    lookup = {t["team"]: t for t in payload["teams"]}

    def opponent_link(name: str) -> str:
        if name in lookup:
            return f'<a href="{slug(name)}.html">{esc(name)}</a>'
        return f'{esc(name)} <span class="fcs">FCS</span>'

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
      {identity_mark(team, 46)}
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
  </article>"""
    return page(f"{team['team']} — MRI {payload['season']}", body, payload, depth=1,
                description=f"{team['team']} MRI rating, schedule and game-by-game performance.")


# --------------------------------------------------------------------------
# conference pages
# --------------------------------------------------------------------------

def conference_page(name: str, payload: dict) -> str:
    members = [t for t in payload["teams"] if t["conference"] == name]
    rows = "".join(f"""
      <tr style="--team:{esc(t['color'])}">
        <td class="rk">{t['rank']}</td>
        <td class="tm"><span class="rule"></span>{identity_mark(t, 18)}<a href="../team/{slug(t['team'])}.html">{esc(t['team'])}</a></td>
        <td class="rec">{t['wins']}&ndash;{t['losses']}</td>
        <td class="num">{t['power']:+.1f}</td>
        <td class="num">{t['resume']:+.2f}</td>
        <td class="num muted">{t.get('classicRank') or '&ndash;'}</td>
      </tr>""" for t in members)

    mean = sum(t["power"] for t in members) / len(members) if members else 0
    body = f"""
  <h1>{esc(name)}</h1>
  <p class="teamsub">{len(members)} teams &middot; mean power {mean:+.1f}</p>
  <div class="tablewrap"><table class="conftable">
    <thead><tr><th>#</th><th>Team</th><th>Rec</th><th class="num">Power</th>
    <th class="num">R&eacute;sum&eacute;</th><th class="num">Classic</th></tr></thead>
    <tbody>{rows}</tbody>
  </table></div>"""
    return page(f"{name} — MRI {payload['season']}", body, payload, depth=1,
                description=f"{name} teams ranked by MRI power rating.")


def conferences_index(payload: dict) -> str:
    cards = "".join(f"""
    <a class="confcard" href="conference/{slug(c['conference'])}.html">
      <span class="ccname">{esc(c['conference'])}</span>
      <span class="ccmean">{c['mean']:+.1f}</span>
      <span class="ccmeta">{int(c['count'])} teams &middot; best {c['max']:+.1f}</span>
    </a>""" for c in payload["conferences"])
    body = f"""
  <h1>Conferences</h1>
  <p class="hint">Ranked by the mean power rating of member teams.</p>
  <div class="confgrid">{cards}</div>"""
    return page(f"Conferences — MRI {payload['season']}", body, payload)


# --------------------------------------------------------------------------
# archive & methodology
# --------------------------------------------------------------------------

def archive_page(payload: dict) -> str:
    weeks = max(len(t["rankHistory"]) for t in payload["teams"])
    header = "".join(f"<th>Wk {w + 1}</th>" for w in range(weeks))
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
    return page(f"Archive — MRI {payload['season']}", body, payload)


def method_page(payload: dict) -> str:
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
    return page(f"Method — MRI {payload['season']}", body, payload,
                description="How the MRI rating system works, and how well it does.")


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def build(payload: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "team").mkdir(exist_ok=True)
    (out_dir / "conference").mkdir(exist_ok=True)

    written = []

    def write(path: Path, content: str):
        path.write_text(content)
        written.append(path)

    write(out_dir / "styles.css", STYLES)
    write(out_dir / "index.html", rankings_page(payload))
    write(out_dir / "conferences.html", conferences_index(payload))
    write(out_dir / "archive.html", archive_page(payload))
    write(out_dir / "method.html", method_page(payload))
    write(out_dir / "site.json", json.dumps(payload, indent=2))

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
html { -webkit-text-size-adjust: 100%; }
body { margin:0; background:var(--plane); color:var(--primary);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif; line-height:1.5; }
a { color:inherit; }
.wrap { max-width:1080px; margin:0 auto; padding-left:18px; padding-right:18px; }

header.site { border-bottom:1px solid var(--grid); background:var(--surface); }
.bar { display:flex; align-items:center; gap:20px; flex-wrap:wrap; padding-block:14px; }
.mark { font-size:20px; font-weight:800; letter-spacing:-0.02em; text-decoration:none; }
.mark em { font-style:normal; color:var(--series); }
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
.compare { background:var(--surface); border:1px solid var(--grid); border-radius:10px; margin:14px 0; }
.compare td { color:var(--secondary); } .compare td strong { color:var(--primary); }

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
