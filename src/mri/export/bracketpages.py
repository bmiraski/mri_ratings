"""The bracketology pages: the field as a seed list (the S-curve), and the same field as a bracket.

Two views of one projection, linked to each other. The list is where the numbers live - every
team's chance to make the field, its likely seed line and how both moved this week - and it is
the page to read on a phone. The bracket is the shape people actually argue about: who opens
against whom, and which region is hardest. Both are static HTML; the bracket's region tabs on
small screens are radio buttons and CSS, not script.

Everything here renders ``payload["bracketology"]`` from :mod:`mri.export.bracketdata`.
"""

from __future__ import annotations

from math import erf, sqrt

from ..bracket.simulate import SIGMA
from .site import _pct, dt_label, esc, identity_mark, page, season_text, slug

LINE_NAMES = {1: "No. 1 seeds", 2: "No. 2 seeds", 3: "No. 3 seeds", 4: "No. 4 seeds"}
BUBBLE_FLOOR = 0.05          # "also in the running": anyone outside the projection with at least this chance
BUBBLE_ROWS = 16


def _win(a: float, b: float) -> float:
    """Chance ``a`` beats ``b`` on a neutral floor: the betting board's single-game model."""
    return 0.5 * (1 + erf((a - b) / (SIGMA * sqrt(2))))


def _date(iso: str | None) -> str:
    return dt_label(iso) if iso else ""


def _team_bits(payload: dict):
    teams = {t["team"]: t for t in payload["teams"]}

    def link(name: str) -> str:
        return f'<a href="team/{slug(name)}.html">{esc(name)}</a>' if name in teams else esc(name)

    def mark(name: str, size: int = 20) -> str:
        return identity_mark(teams[name], size) if name in teams else ""

    return teams, link, mark


def _move(row: dict) -> str:
    change = row.get("change")
    if change is None:
        return '<span class="mv flat" title="New this week">new</span>'
    if abs(change) < 0.005:
        return '<span class="mv flat" title="No change in chance to make the field">&ndash;</span>'
    kind, arrow = ("up", "&#9650;") if change > 0 else ("down", "&#9660;")
    return (f'<span class="mv {kind}" title="Change in chance to make the field since last week, in points">'
            f'{arrow}{abs(change) * 100:.0f}</span>')


def _seed_range(row: dict) -> str:
    """Likeliest seed line, and the range covering the middle 80% of the worlds it's in the field."""
    odds = row.get("seedOdds") or []
    total = sum(o or 0 for o in odds)
    if total <= 0:
        return '<span class="muted">&ndash;</span>'
    cum, lo, hi = 0.0, None, None
    for line, p in enumerate(odds, 1):
        cum += (p or 0) / total
        if lo is None and cum >= 0.10:
            lo = line
        if hi is None and cum >= 0.90:
            hi = line
    likeliest = max(range(16), key=lambda i: odds[i] or 0) + 1
    span = f'<span class="muted"> ({lo}&ndash;{hi})</span>' if lo != hi else ""
    return f"<strong>{likeliest}</strong>{span}"


def _toggle(current: str) -> str:
    opts = (("bracketology.html", "Seed list"), ("bracket.html", "Bracket"))
    return '<div class="views" role="navigation" aria-label="Bracketology views">' + "".join(
        f'<a href="{href}"{" class=on aria-current=page" if href == current + ".html" else ""}>{label}</a>'
        for href, label in opts) + "</div>"


def _status(b: dict, *, arrows: bool = True) -> str:
    if b.get("frozen"):
        return (f'<p class="hint"><strong>Frozen.</strong> This is the last projection made before the bracket was revealed on '
                f'Selection Sunday ({esc(_date(b["selectionSunday"]))}), as it stood on {esc(_date(b["updated"]))}. '
                f'It is not updated from here: a projection redrawn after the real bracket is public would only be copying it.</p>')
    compared = f' Arrows are the change since {esc(_date(b["comparedTo"]))}.' if b.get("comparedTo") and arrows else ""
    return (f'<p class="hint">Updated {esc(_date(b["updated"]))} from {b["sims"]:,} simulated finishes to the season. '
            f'Selection Sunday is {esc(_date(b["selectionSunday"]))}.{compared}</p>')


def _result_panel(b: dict, link, mark) -> str:
    r = b.get("result")
    if not r:
        if b.get("frozen"):
            return '<p class="hint">The real field will be set beside this projection once the tournament feed has it.</p>'
        return ""
    missed = ""
    if r["missedOut"] or r["missedIn"]:
        out = ", ".join(link(t) for t in r["missedOut"]) or "none"
        inn = ", ".join(link(t) for t in r["missedIn"]) or "none"
        missed = (f'<p class="small"><strong>Projected in, left out:</strong> {out}.<br>'
                  f'<strong>Picked, but not projected:</strong> {inn}.</p>')
    rows = "".join(
        f'<tr><td class="opp"><span class="nmcell">{mark(x["team"], 18)}{link(x["team"])}</span></td>'
        f'<td class="num">{x["projected"] or "&ndash;"}</td><td class="num">{x["actual"] or "&ndash;"}</td>'
        f'<td class="num">{"" if not (x["projected"] and x["actual"]) else (x["projected"] - x["actual"]) or "&check;"}</td></tr>'
        for x in r["rows"])
    return f"""
  <section class="panel verdictpanel">
    <p class="ptitle">How it did</p>
    <div class="scoreline">
      <div><span class="big">{r['named']}<span class="of">/{r['fieldSize']}</span></span><span class="lbl">teams in the field named</span></div>
      <div><span class="big">{r['atLarge']['named']}<span class="of">/{r['atLarge']['of']}</span></span><span class="lbl">at-large teams named</span></div>
      <div><span class="big">{r['seedExact']}<span class="of">/{r['seedCompared']}</span></span><span class="lbl">on the exact seed line</span></div>
      <div><span class="big">{r['seedWithinOne']}<span class="of">/{r['seedCompared']}</span></span><span class="lbl">within one line</span></div>
    </div>
    {missed}
    <details><summary>Every team: projected against actual seed</summary>
      <div class="tablewrap"><table class="slate"><thead><tr><th>Team</th><th class="num">Projected</th><th class="num">Actual</th><th class="num">Off by</th></tr></thead>
      <tbody>{rows}</tbody></table></div></details>
  </section>"""


def _bubble_panel(b: dict, by_team: dict, link, mark) -> str:
    bubble = b["projected"]["bubble"]
    field = {f["team"]: f for f in b["projected"]["field"]}

    def item(name: str, tag: str = "") -> str:
        t = by_team.get(name, {})
        return (f'<li><span class="nmcell">{mark(name, 18)}{link(name)}</span>'
                f'<span class="gv">{tag}{_pct(t.get("pAtLarge"))}</span></li>')

    last_in = bubble["openingRoundAtLarge"]
    last_in_items = "".join(item(n, f'<em>No. {field[n]["seedLine"]}</em> ' if n in field else "") for n in last_in)
    label = f"Last {len(last_in)} in" + (" &middot; Opening Round" if last_in else "")
    return f"""
  <section class="panel">
    <p class="ptitle">Bubble watch</p>
    <div class="bubblegrid">
      <div><h3>{label}</h3><ul class="list">{last_in_items}</ul></div>
      <div><h3>First four out</h3><ul class="list">{''.join(item(n) for n in bubble['firstFourOut'])}</ul>
        <h3>Next four out</h3><ul class="list">{''.join(item(n) for n in bubble['nextFourOut'])}</ul></div>
    </div>
    <p class="note">The chance beside each team is how often it earned an <em>at-large</em> bid across every simulated finish (a team that might also win its
    conference tournament has a better chance of making the field than this); the grouping is this one projected bracket. In the {b['fieldSize']}-team field the
    {len(last_in)} lowest at-large teams play in the Opening Round.</p>
  </section>"""


def _conference_cards(b: dict, link) -> str:
    status_text = {"not started": "Tournament not started", "underway": "Tournament underway",
                   "decided": "Champion decided", "no tournament": "No tournament"}
    cards = []
    for conf, c in sorted(b["conferences"].items()):
        odds = c.get("odds") or []
        if not odds:
            continue
        lines = "".join(f'<li><span>{link(o["team"])}</span><span class="gv">'
                        f'<span class="pbar" style="width:{max(o["p"] * 70, 1):.0f}px"></span>{_pct(o["p"])}</span></li>'
                        for o in odds[:3])
        more = sum(o["p"] for o in odds[3:])
        rest = f'<li class="muted small"><span>Everyone else</span><span class="gv">{_pct(max(0.0, 1 - sum(o["p"] for o in odds[:3])))}</span></li>' \
            if len(odds) > 3 or more else ""
        fmt = ""
        if c.get("formatSource") == "provisional":
            fmt = ' <span class="tag" title="This conference has not published its tournament bracket yet; the format here is an assumption.">format assumed</span>'
        elif c.get("mode") == "placeholder":
            fmt = ' <span class="tag" title="Set by hand: the standings leader takes the bid.">standings leader</span>'
        cards.append(f'<div class="confcard autobid"><span class="ccname">{esc(conf)}{fmt}</span>'
                     f'<span class="ccmeta">{status_text.get(c.get("status"), "")}</span><ul class="list">{lines}{rest}</ul></div>')
    return "".join(cards)


def list_page(payload: dict) -> str:
    b = payload["bracketology"]
    by_team = {t["team"]: t for t in b["teams"]}
    _, link, mark = _team_bits(payload)
    field = b["projected"]["field"]
    semis = b["projected"]["semifinals"]
    one_seed = {f["region"]: f["team"] for f in field if f["seedLine"] == 1}

    rows, line_seen = [], None
    # By seed line, then true seed: in the 76-team format an Opening Round at-large team can rank above
    # direct teams on lower lines (see mri.bracket.seeding), so pure true-seed order would repeat lines.
    for f in sorted(field, key=lambda f: (f["seedLine"], f["trueSeed"])):
        t = by_team.get(f["team"], {})
        if f["seedLine"] != line_seen:
            line_seen = f["seedLine"]
            name = LINE_NAMES.get(line_seen, f"No. {line_seen} seeds")
            rows.append(f'<tr class="lineband"><td colspan="7">{name}</td></tr>')
        bid = "Auto" if f["bidType"] == "auto" else "At-large"
        if f["bidType"] == "auto":
            bid += f' <span class="muted small" title="Chance it wins the conference tournament">({_pct(t.get("pAuto"))})</span>'
        opening = ' <span class="tag or" title="Plays in the Opening Round for this seed line">Opening Rd</span>' if f["openingRound"] else ""
        moved = (f' <span class="tag" title="Moved from line {f["movedFrom"]} to keep conference teams apart, as the committee\'s rules allow">moved</span>'
                 if f.get("movedFrom") else "")
        rows.append(f"""<tr>
      <td class="wk">{f['trueSeed']}</td>
      <td class="opp"><span class="nmcell">{mark(f['team'])}<strong>{link(f['team'])}</strong></span>
        <div class="muted small">{esc(f['conference'])} &middot; {esc(t.get('record', ''))} &middot; MRI #{t.get('powerRank', '')}</div></td>
      <td class="small">{bid}{opening}{moved}</td>
      <td class="num small">{esc(one_seed.get(f['region'], ''))}</td>
      <td class="num prob"><span class="pbar" style="width:{min((t.get('pField') or 0) * 90, 90):.0f}px"></span>{_pct(t.get('pField'))}</td>
      <td>{_move(t)}</td>
      <td class="num">{_seed_range(t)}</td></tr>""")

    in_field = {f["team"] for f in field}
    outside = [t for t in b["teams"] if t["team"] not in in_field and t["pField"] >= BUBBLE_FLOOR]
    outside = sorted(outside, key=lambda t: -t["pField"])[:BUBBLE_ROWS]
    outside_rows = "".join(f"""<tr><td class="opp"><span class="nmcell">{mark(t['team'], 18)}{link(t['team'])}</span>
        <span class="muted small"> {esc(t['conference'])} &middot; {esc(t['record'])}</span></td>
      <td class="num prob"><span class="pbar" style="width:{t['pField'] * 90:.0f}px"></span>{_pct(t['pField'])}</td>
      <td class="num small">{_pct(t['pAuto']) if t['pAuto'] >= 0.005 else '&ndash;'}</td>
      <td>{_move(t)}</td></tr>""" for t in outside)

    bt = payload.get("bracketologyBacktest") or {}
    summary = (bt.get("summary") or {})
    grade = ""
    if summary:
        def cell(cp, key, fmt):
            s = (summary.get(cp) or {}).get("allModel") or {}
            return fmt.format(s[key]) if key in s else "&ndash;"
        ended = lambda cp: ((summary.get(cp) or {}).get("endedToday") or {}).get("brierPerSeason")  # noqa: E731
        grade = f"""
  <h2>How it has done</h2>
  <p>Every season from 2011 to 2025 (2020 was cancelled), rerun from three points in the season using only what was known then, against the field the committee
  actually picked. The comparison is the plain alternative: rank the teams as they stand, with each conference's standings leader taking its bid.</p>
  <div class="tablewrap"><table class="slate"><thead><tr><th></th><th class="num">February 1</th><th class="num">Conf. tournaments start</th><th class="num">Selection Sunday</th></tr></thead><tbody>
    <tr><td class="opp">Real field teams named</td><td class="num">{cell('feb1', 'fieldHitRate', '{:.0%}')}</td><td class="num">{cell('confStart', 'fieldHitRate', '{:.0%}')}</td><td class="num">{cell('selectionSunday', 'fieldHitRate', '{:.0%}')}</td></tr>
    <tr><td class="opp">Seed line, average miss</td><td class="num">{cell('feb1', 'seedMAE', '{:.1f}')}</td><td class="num">{cell('confStart', 'seedMAE', '{:.1f}')}</td><td class="num">{cell('selectionSunday', 'seedMAE', '{:.1f}')}</td></tr>
    <tr><td class="opp">Error in the chances (Brier, lower is better)</td><td class="num">{cell('feb1', 'brierPerSeason', '{:.1f}')}</td><td class="num">{cell('confStart', 'brierPerSeason', '{:.1f}')}</td><td class="num">{cell('selectionSunday', 'brierPerSeason', '{:.1f}')}</td></tr>
    <tr><td class="opp muted">&hellip; ranking the teams as they stand</td><td class="num muted">{ended('feb1') or 0:.1f}</td><td class="num muted">{ended('confStart') or 0:.1f}</td><td class="num muted">{ended('selectionSunday') or 0:.1f}</td></tr>
  </tbody></table></div>
  <p class="hint">Where it says 25%, teams like that have made the field about a quarter of the time; in early February the 70&ndash;90% range has run a few points
  high. History is graded in the old 68-team format - this is the first season of 76.</p>"""

    body = f"""
  <article class="prose wide bracketology">
  <div class="titlebar"><h1>Bracketology</h1>{_toggle('bracketology')}</div>
  <p class="lead">Who makes the {b['fieldSize']}-team NCAA Tournament and where they are seeded, from simulating the rest of the season - every
  remaining game and every conference tournament - thousands of times and picking the field the way the committee does.</p>
  {_status(b)}
  {_result_panel(b, link, mark)}
  {_bubble_panel(b, by_team, link, mark)}
  <h2>The projected field</h2>
  <p class="hint">By seed line; <strong>#</strong> is the committee-style overall rank, 1 the top overall seed. <strong>Region</strong> is named for its No. 1 seed; regions
  {semis[0][0]} and {semis[0][1]} meet in one national semifinal, {semis[1][0]} and {semis[1][1]} in the other. <strong>Field</strong> is the chance to make the
  tournament at all; <strong>Seed</strong> is the likeliest line, with the range covering most of the rest.</p>
  <div class="tablewrap"><table class="slate scurve"><thead><tr><th>#</th><th>Team</th><th>Bid</th><th class="num">Region</th>
    <th class="num">Field</th><th title="Change since last week">Wk</th><th class="num">Seed</th></tr></thead>
    <tbody>{''.join(rows)}</tbody></table></div>

  <h2>Also in the running</h2>
  <p class="hint">Teams outside the projected field with at least a {BUBBLE_FLOOR:.0%} chance. <strong>Auto</strong> is the chance of stealing a bid by winning the conference tournament.</p>
  <div class="tablewrap"><table class="slate"><thead><tr><th>Team</th><th class="num">Field</th><th class="num">Auto</th><th>Wk</th></tr></thead>
    <tbody>{outside_rows or '<tr><td colspan="4" class="empty">Nobody else above the line.</td></tr>'}</tbody></table></div>

  <h2>Automatic bids</h2>
  <p class="hint">Each conference's champion goes to the tournament. These are the chances of winning it, from simulating the rest of the league
  season and then the conference's own tournament bracket.</p>
  <div class="confgrid">{_conference_cards(b, link)}</div>

  <h2>How it works</h2>
  <p>Each simulated finish plays out the rest of the regular season game by game, then every conference's standings and its tournament in that
  world's seed order; those games land on each team's r&eacute;sum&eacute; the way they land on the committee's. The at-large teams and every seed come from a
  score fit to fourteen years of the committee's real seed lines: strength, r&eacute;sum&eacute; (wins beyond what an average team would manage on the same
  schedule), results against the best opponents, bad losses, schedule strength and road wins. Each world also blurs that score by about a seed line,
  because the score is a model of the committee and not the committee, and blurs every team's rating by how little we can know it yet.</p>
  <p><strong>Seed lines</strong> follow the NCAA's 2027 format: the twelve lowest at-large teams play in the Opening Round as No. 11s and No. 12s, and the twelve
  lowest conference champions as No. 15s and No. 16s. <strong>Regions</strong> follow the committee's published rules for keeping conference teams apart and the
  regions balanced, but not travel: the committee's geography is its own call, so treat the region as a guess and the seed line as the projection.</p>
  {grade}
  <p class="hint"><a href="method.html">Method: how the ratings work</a>.</p>
  </article>"""
    return page(f"Bracketology — MRI {season_text(payload)}", body, payload,
                description="The projected NCAA Tournament field and seed list, from a season simulation graded on every tournament since 2011.")


def _slot(field_by_region_line: dict, region: int, line: int) -> list[dict]:
    return field_by_region_line.get((region, line), [])


def bracket_page(payload: dict) -> str:
    b = payload["bracketology"]
    by_team = {t["team"]: t for t in b["teams"]}
    _, link, mark = _team_bits(payload)
    field = b["projected"]["field"]
    order = b["projected"]["regionOrder"]
    semis = b["projected"]["semifinals"]
    power = {t["team"]: t["power"] for t in b["teams"]}

    cell: dict[tuple[int, int], list[dict]] = {}
    for f in field:
        cell.setdefault((f["region"], f["seedLine"]), []).append(f)
    one_seed = {r: (cell.get((r, 1)) or [{}])[0].get("team", "") for r in range(1, 5)}

    def reach(slot: list[dict]) -> dict[str, float]:
        """Chance each team fills this slot: 1 for a direct team, its Opening Round game odds otherwise."""
        if len(slot) == 1:
            return {slot[0]["team"]: 1.0}
        a, c = slot[0]["team"], slot[1]["team"]
        p = _win(power.get(a, 0), power.get(c, 0))
        return {a: p, c: 1 - p}

    def side(slot: list[dict], opp: list[dict]) -> str:
        if not slot:
            return '<div class="tm empty">&ndash;</div>'
        mine, theirs = reach(slot), reach(opp) if opp else {}
        win = sum(pm * po * _win(power.get(m, 0), power.get(o, 0))
                  for m, pm in mine.items() for o, po in theirs.items()) if theirs else None
        pct = f'<span class="wp" title="Chance to win this game">{_pct(win)}</span>' if win is not None else ""
        if len(slot) == 1:
            f = slot[0]
            return (f'<div class="tm"><span class="sd">{f["seedLine"]}</span>{mark(f["team"], 18)}'
                    f'<span class="tn">{link(f["team"])}</span>{pct}</div>')
        names = " / ".join(link(f["team"]) for f in slot)
        return (f'<div class="tm or"><span class="sd">{slot[0]["seedLine"]}</span>'
                f'<span class="tn" title="Opening Round winner">{names}</span>{pct}</div>')

    def region_html(r: int) -> str:
        games = []
        for k in range(0, 16, 2):
            top, bottom = _slot(cell, r, order[k]), _slot(cell, r, order[k + 1])
            games.append(f'<div class="game">{side(top, bottom)}{side(bottom, top)}</div>')
        pods = "".join(f'<div class="pod">{games[i]}{games[i + 1]}</div>' for i in range(0, 8, 2))
        total = (b["projected"].get("regionTotals") or {}).get(str(r))
        weight = f'<span class="muted small" title="Sum of the true seeds on its top four lines; lower is a stronger region">top-four total {total}</span>' if total else ""
        return (f'<section class="region" id="region-{r}"><div class="regionhead"><h3>Region {r}</h3>'
                f'<span class="small">{esc(one_seed.get(r, ""))}&rsquo;s region</span>{weight}</div>{pods}</section>')

    opening = [f for f in field if f["openingRound"]]
    games_or: dict[str, list[dict]] = {}
    for f in opening:
        games_or.setdefault(f["openingGame"] or f["team"], []).append(f)
    or_rows = []
    for key, pair in sorted(games_or.items(), key=lambda kv: (kv[1][0]["seedLine"], kv[1][0]["trueSeed"])):
        if len(pair) != 2:
            continue
        a, c = pair
        p = _win(power.get(a["team"], 0), power.get(c["team"], 0))
        opp = [x for x in _slot(cell, a["region"], 17 - a["seedLine"])]
        vs = f' &rarr; No. {17 - a["seedLine"]} {link(opp[0]["team"])}' if opp else ""
        kind = "at-large" if a["bidType"] == "at-large" else "champions"
        pair_html = "".join(f'<div class="orteam"><span class="nmcell">{mark(x["team"], 18)}{link(x["team"])}</span>'
                            f'<span class="wp" title="Chance to win this game">{_pct(q)}</span></div>'
                            for x, q in ((a, p), (c, 1 - p)))
        or_rows.append(f"""<tr><td class="wk">{a['seedLine']}<div class="muted small">{kind}</div></td>
          <td class="orgame">{pair_html}</td>
          <td class="small">Region {a['region']}{vs}</td></tr>""")

    tabs = "".join(f'<input type="radio" name="rg" id="rg{r}"{" checked" if r == 1 else ""}><label for="rg{r}">{esc(one_seed.get(r, f"Region {r}"))}</label>'
                   for r in range(1, 5))
    problems = b["projected"].get("ruleProblems") or []
    problem_note = (f'<p class="hint">One bracketing rule couldn\'t be met this time: {esc(problems[0])}.</p>' if problems else "")
    body = f"""
  <article class="prose wide bracketology">
  <div class="titlebar"><h1>Bracketology</h1>{_toggle('bracket')}</div>
  <p class="lead">The projected field as a bracket: the likeliest champion from each conference, the at-large teams most often in the field, seeded
  and placed by the committee's own bracketing rules.</p>
  {_status(b, arrows=False)}
  <p class="hint">Percentages are each team's chance to win that first game, from the ratings. Regions are named for their No. 1 seed; Region {semis[0][0]}
  meets Region {semis[0][1]} in the national semifinals and Region {semis[1][0]} meets Region {semis[1][1]}. Placement follows the committee's published rules
  but not its travel decisions, so read the region as a guess and the matchup as the projection.</p>
  {problem_note}
  <h2>Opening Round</h2>
  <div class="tablewrap"><table class="slate"><thead><tr><th>Seed</th><th>Game</th><th>Winner plays</th></tr></thead>
    <tbody>{''.join(or_rows)}</tbody></table></div>
  <p class="hint">Who plays in is set by the NCAA's rule; who plays whom isn't published, so neighbours on the seed list are paired here.</p>
  <h2>First round</h2>
  <div class="regiontabs">{tabs}
    <div class="regions">{''.join(region_html(r) for r in range(1, 5))}</div>
  </div>
  </article>"""
    return page(f"Bracket — MRI {season_text(payload)}", body, payload,
                description="The projected NCAA Tournament bracket: every first-round matchup and the Opening Round.")


def team_line(team: str, payload: dict) -> str | None:
    """The one-line summary for a basketball team page, or None if the team isn't close to the field."""
    b = payload.get("bracketology")
    if not b:
        return None
    row = next((t for t in b["teams"] if t["team"] == team), None)
    if not row or row["pField"] < 0.01:
        return None
    projected = next((f for f in b["projected"]["field"] if f["team"] == team), None)
    if projected:
        where = f'projected No. {projected["seedLine"]} seed'
        if projected["openingRound"]:
            where += " (Opening Round)"
    elif team in (b["projected"]["bubble"].get("firstFourOut") or []):
        where = "first four out"
    else:
        where = "outside the projected field"
    verb = "was" if b.get("frozen") else ""
    chance = f'{_pct(row["pField"])} to make the field'
    return (f'<div class="hl" title="From the bracketology page{", as frozen on Selection Sunday" if verb else ""}."><span class="hll">Bracketology</span>'
            f'<span class="hlv"><a href="../bracketology.html">{where}</a> &middot; {chance}</span></div>')


STYLES = """
/* bracketology */
.titlebar { display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap; }
.views { display:flex; border:1px solid var(--grid); border-radius:999px; overflow:hidden; }
.views a { font-size:12px; font-weight:600; padding:5px 12px; text-decoration:none; color:var(--muted); }
.views a:hover { color:var(--primary); }
.views a.on { background:var(--primary); color:var(--surface); }
.tag { font-size:10px; color:var(--muted); border:1px solid var(--axis); border-radius:3px; padding:0 4px; white-space:nowrap;
  font-weight:500; text-transform:none; letter-spacing:0; }
.tag.or { color:var(--primary); border-color:var(--series); }
table.scurve tr.lineband td { font-size:10px; text-transform:uppercase; letter-spacing:0.09em; color:var(--muted);
  font-weight:600; background:var(--plane); padding-top:12px; }
.bubblegrid { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
.bubblegrid > * { min-width:0; }
.bubblegrid h3 { font-size:11px; text-transform:uppercase; letter-spacing:0.08em; color:var(--secondary); margin:0 0 4px; }
.bubblegrid h3 + ul + h3, .bubblegrid ul + h3 { margin-top:14px; }
.bubblegrid .list li { align-items:center; }
.confcard.autobid .ccname { display:flex; align-items:center; gap:6px; flex-wrap:wrap; }
.confcard.autobid .list li { padding:4px 0; font-size:12.5px; }
.confcard.autobid .list .pbar { height:6px; }
.verdictpanel { border-left:3px solid #c8961e; }
.scoreline { display:grid; grid-template-columns:repeat(4, 1fr); gap:10px; margin-bottom:8px; }
.scoreline .big { display:block; font-size:26px; font-weight:700; font-variant-numeric:tabular-nums; }
.scoreline .lbl { font-size:11px; color:var(--muted); }
.regiontabs > input { position:absolute; opacity:0; pointer-events:none; }
.regiontabs > label { display:none; }
.regions { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
.region { background:var(--surface); border:1px solid var(--grid); border-radius:12px; padding:12px 12px 4px; min-width:0; }
.regionhead { display:flex; align-items:baseline; gap:8px; flex-wrap:wrap; margin-bottom:8px; }
.regionhead h3 { margin:0; font-size:14px; }
.regionhead .muted { margin-left:auto; }
.pod { border-left:2px solid var(--axis); padding-left:8px; margin-bottom:10px; }
.game { border:1px solid var(--grid); border-radius:8px; margin-bottom:5px; background:var(--plane); }
.tm { display:flex; align-items:center; gap:7px; padding:5px 8px; font-size:13px; min-width:0; }
.tm + .tm { border-top:1px solid var(--grid); }
.tm .sd { width:18px; text-align:right; color:var(--muted); font-size:11px; font-variant-numeric:tabular-nums; flex:none; }
.tm .tn { flex:1; min-width:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.tm .tn a { text-decoration:none; } .tm .tn a:hover { text-decoration:underline; }
.tm.or .tn { font-size:12px; color:var(--secondary); }
.orgame { min-width:0; }
.orteam { display:flex; align-items:center; justify-content:space-between; gap:10px; padding:1px 0; }
.orteam .wp, .tm .wp { font-size:11px; color:var(--muted); font-variant-numeric:tabular-nums; flex:none; }
@media (max-width:760px) {
  .bubblegrid { grid-template-columns:1fr; }
  .scoreline { grid-template-columns:1fr 1fr; }
  .regiontabs > label { display:inline-block; font-size:12px; font-weight:600; padding:5px 10px; margin:0 4px 10px 0;
    border:1px solid var(--grid); border-radius:999px; color:var(--muted); cursor:pointer; }
  .regions { grid-template-columns:1fr; }
  .region { display:none; }
  #rg1:checked ~ .regions #region-1, #rg2:checked ~ .regions #region-2,
  #rg3:checked ~ .regions #region-3, #rg4:checked ~ .regions #region-4 { display:block; }
  #rg1:checked + label, #rg2:checked + label, #rg3:checked + label, #rg4:checked + label {
    background:var(--primary); color:var(--surface); border-color:var(--primary); }
  table.scurve td:nth-child(4), table.scurve th:nth-child(4) { display:none; }
}
"""
