"""The basketball preseason rankings: every team ranked by the prior its season will start from.

Until a game of the coming season is final the site's own ratings are last season's, so the
roster-aware prior (``mri.ratings.bb_priors``) has nowhere to show. This page is where it does: each
team's preseason rating, where it finished last season, and the roster facts behind the move.

The data is saved to ``docs/basketball/preseason.json`` and the page is drawn from that file on every
build. Once the season starts the build stops computing it and the saved file stands as the season's
preseason snapshot, so the page keeps saying what the model thought before a game was played.
"""

from __future__ import annotations

import json
from pathlib import Path

from .site import _CHAOS_SORT_SCRIPT, esc, identity_mark, movement_chip, page, slug


def build(season: int, payload: dict) -> dict:
    """The preseason table for ``season``, from the prior and the site's last season of ratings."""
    from ..betting import bb_board
    from ..ingest import bb_registry as registry
    from . import bb_sitedata

    power = bb_board.preseason_power(season)
    d1 = registry.teams(season)
    names = [t for t in power.sort_values(ascending=False).index if t in d1]
    last = {t["team"]: t for t in payload["teams"]}
    roster = bb_sitedata.roster_context(season, names)

    teams = []
    for rank, name in enumerate(names, 1):
        before, known = last.get(name, {}), d1[name]
        context = roster.get(name) or {}
        teams.append({
            "team": name,
            "rank": rank,
            "power": round(float(power[name]), 2),
            "conference": known.conference or before.get("conference") or "Independent",
            "color": before.get("color") or known.color or bb_sitedata.FALLBACK,
            "logo": before.get("logo"),
            "lastRank": before.get("rank"),
            "lastPower": before.get("power"),
            "hasPage": name in last,
            "mode": context.get("mode"),
            "returning": context.get("returning"),
            "returningRank": context.get("returningRank"),
            "incoming": context.get("incoming"),
            "freshman": context.get("freshman"),
            "freshmanRank": context.get("freshmanRank"),
        })
    with_roster = sum(t["mode"] == "roster" for t in teams)
    return {
        "season": season,
        "seasonLabel": bb_sitedata.season_label(season),
        "lastSeasonLabel": payload.get("seasonLabel"),
        "mode": "roster" if with_roster >= 0.6 * max(len(teams), 1) else "before rosters",
        "teamsWithRosters": with_roster,
        "teams": teams,
    }


def saved(root: Path, season: int) -> dict | None:
    """The preseason table written for ``season`` by an earlier build, if there is one."""
    path = root / "preseason.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data if data.get("season") == season else None


def _roster_cells(t: dict) -> tuple[str, str, str]:
    if t.get("returning") is None:
        none = '<span class="muted">&ndash;</span>'
        return none, none, _freshman(t)
    returning = (f'<span title="Ranked {t["returningRank"]} of teams with posted rosters">{t["returning"] * 100:.0f}%</span>')
    return returning, f'{t["incoming"]:.1f}', _freshman(t)


def _freshman(t: dict) -> str:
    if not t.get("freshman"):
        return '<span class="muted">&ndash;</span>'
    return f'<span title="Class ranked {t["freshmanRank"]}">{t["freshman"]:.2f}</span>'


def preseason_page(payload: dict) -> str:
    data = payload["preseason"]
    teams = data["teams"]
    rows = []
    for t in teams:
        move = (t["lastRank"] - t["rank"]) if t.get("lastRank") else 0
        name = esc(t["team"])
        link = f'<a href="team/{slug(t["team"])}.html">{name}</a>' if t.get("hasPage") else name
        returning, incoming, freshman = _roster_cells(t)
        last = f'#{t["lastRank"]}' if t.get("lastRank") else '<span class="muted">New</span>'
        rows.append(f"""
      <tr data-conf="{esc(t['conference'])}" data-rank="{t['rank']}" data-team="{name}" data-power="{t['power']}"
        data-last="{t.get('lastRank') or 999}" data-move="{move}" data-returning="{t.get('returning') if t.get('returning') is not None else -1}"
        data-incoming="{t.get('incoming') if t.get('incoming') is not None else -1}" data-freshman="{t.get('freshman') or 0}">
        <td class="num">{t['rank']}</td>
        <td class="opp"><span class="nmcell">{identity_mark(t)}{link}</span></td>
        <td class="small">{esc(t['conference'])}</td>
        <td class="num">{t['power']:+.1f}</td>
        <td class="num">{last}</td>
        <td>{movement_chip(move)}</td>
        <td class="num">{returning}</td>
        <td class="num">{incoming}</td>
        <td class="num">{freshman}</td></tr>""")

    conferences = sorted({t["conference"] for t in teams})
    options = "".join(f'<option value="{esc(c)}">{esc(c)}</option>' for c in conferences)
    model = payload.get("priorModel") or {}
    miss = (model.get("outOfSampleRmse") or {}).get("with a roster", {}).get("all")
    old_miss = (model.get("outOfSampleRmse") or {}).get("old", {}).get("all")
    accuracy = (f" Out of sample, with a roster it misses a team&rsquo;s final rating by about {miss:.1f} points, against {old_miss:.1f} "
                "for the old rule of last season&rsquo;s rating pulled toward average.") if miss and old_miss else ""
    if data["mode"] == "roster":
        coverage = (f"Rosters are posted for {data['teamsWithRosters']} of {len(teams)} teams, so these ratings use them. "
                    "Where a school has not cleaned up its list the page can still read some departed players as returning.")
    else:
        coverage = ("Rosters have not been posted for most teams, so these ratings use only what last season and the draft imply. "
                    "The page is rebuilt as schools post theirs.")

    body = f"""
  <article class="prose wide" id="preseason">
  <h1>{esc(data['seasonLabel'])} preseason rankings</h1>
  <p class="lead">Where the model starts every team, before a game of the season has been played.</p>
  <p class="hint">Each team&rsquo;s <strong>Rating</strong> is the prior its season starts from: last season&rsquo;s rating, adjusted for
  <strong>Returning</strong> (the share of last season&rsquo;s win shares still on the roster), <strong>Incoming</strong> (what the
  newcomers produced last season at other schools, in win shares) and the <strong>Freshman</strong> class (rated recruits above a
  baseline). The games then move each rating away from it, quickly: by a fifth of the season the games have taught the model most
  of what the prior was guessing.{accuracy}</p>
  <p class="hint">{coverage} The prior is deliberately cautious in scale, so these ratings are closer together than they will be
  by March. The <a href="method.html#priors">method page</a> has the detail.</p>
  <p class="filters"><label>Conference <select id="pconf"><option value="">All</option>{options}</select></label>
  <span class="muted small">Click a column to sort.</span></p>
  <div class="tablewrap"><table class="sortable">
    <thead><tr>
      <th class="num" data-key="rank">#</th><th data-key="team">Team</th><th data-key="conf">Conf</th>
      <th class="num" data-key="power" title="Points better than an average team, before any game is played">Rating</th>
      <th class="num" data-key="last" title="Where the team finished {esc(str(data.get('lastSeasonLabel') or 'last season'))}">Last yr</th>
      <th data-key="move" title="Places gained or lost from last season&rsquo;s finish">Move</th>
      <th class="num" data-key="returning" title="Share of last season&rsquo;s win shares on the roster">Returning</th>
      <th class="num" data-key="incoming" title="Win shares the newcomers produced last season elsewhere">Incoming</th>
      <th class="num" data-key="freshman" title="Rated recruits above a baseline, summed">Freshman</th>
    </tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table></div>
  </article>{_CHAOS_SORT_SCRIPT}
<script>
document.getElementById('pconf').addEventListener('change', function () {{
  var want = this.value;
  document.querySelectorAll('#preseason tbody tr').forEach(function (row) {{
    row.hidden = !!want && row.getAttribute('data-conf') !== want;
  }});
}});
</script>"""
    return page(f"{data['seasonLabel']} preseason rankings — MRI Basketball", body, payload,
                description="Every Division I team ranked by the prior its season starts from: last season's rating, adjusted for the roster.")
