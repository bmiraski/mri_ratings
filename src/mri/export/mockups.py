"""Two design directions for the rankings page, rendered from real data.

Both read the same site.json, so what you are comparing is the design and not
the numbers. Team color is used as identity - a rule, a chip - never as the
encoding of a value, since 138 brand colors were not chosen to be
distinguishable from one another. Values are carried by position and text.

Direction A, "Broadsheet": a sports page. Dense, typographic, quiet. Rank and
numbers do the work; color is a hairline.

Direction B, "Console": an analytics product. Dark, card-led, trajectory and
disagreement surfaced up front.
"""

from __future__ import annotations

import json
from pathlib import Path

# Validated chart tokens (see the dataviz reference palette).
INK = {
    "surface": "#fcfcfb",
    "plane": "#f9f9f7",
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "up": "#006300",
    "down": "#d03b3b",
    "series": "#2a78d6",
}
INK_DARK = {
    "surface": "#1a1a19",
    "plane": "#0d0d0d",
    "primary": "#ffffff",
    "secondary": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "up": "#0ca30c",
    "down": "#d03b3b",
    "series": "#3987e5",
}


def sparkline(values: list[float], width: int = 64, height: int = 20, color: str = "#2a78d6") -> str:
    """A 2px trajectory line with an end marker, per the mark specs."""
    if not values or len(values) < 2:
        return f'<svg width="{width}" height="{height}" aria-hidden="true"></svg>'
    low, high = min(values), max(values)
    span = (high - low) or 1.0
    step = width / (len(values) - 1)
    points = [
        (i * step, height - 3 - ((v - low) / span) * (height - 6)) for i, v in enumerate(values)
    ]
    path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points))
    end_x, end_y = points[-1]
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'aria-hidden="true" class="spark">'
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="3" fill="{color}" '
        f'stroke="var(--surface)" stroke-width="2"/>'
        f"</svg>"
    )


def _movement(value: int) -> str:
    if not value:
        return '<span class="mv flat">-</span>'
    arrow = "▲" if value > 0 else "▼"
    kind = "up" if value > 0 else "down"
    return f'<span class="mv {kind}">{arrow}{abs(value)}</span>'


def _identity(team: dict, size: int = 22) -> str:
    """Logo where available, a color chip where not. Never color alone."""
    if team.get("logo"):
        return (
            f'<img class="logo" src="{team["logo"]}" alt="" width="{size}" height="{size}" '
            f'loading="lazy" onerror="this.replaceWith(Object.assign(document.createElement(\'span\'),'
            f'{{className:\'chip\',style:\'background:{team["color"]}\'}}))">'
        )
    return f'<span class="chip" style="background:{team["color"]}"></span>'


def broadsheet(payload: dict) -> str:
    rows = []
    for team in payload["teams"][:40]:
        disagreement = ""
        if team.get("classicRank"):
            gap = team["classicRank"] - team["rank"]
            if abs(gap) >= 15:
                disagreement = f'<span class="flag" title="MRI Classic has this team at #{team["classicRank"]}">Classic #{team["classicRank"]}</span>'
        rows.append(f"""
      <tr>
        <td class="rk">{team['rank']}</td>
        <td class="mvc">{_movement(team['movement'])}</td>
        <td class="tm"><span class="rule" style="background:{team['color']}"></span>{_identity(team)}<span class="nm">{team['team']}</span>{disagreement}</td>
        <td class="cf">{team['conference']}</td>
        <td class="rec">{team['wins']}-{team['losses']}</td>
        <td class="num pw">{team['power']:+.1f}</td>
        <td class="num rs">{team['resume']:+.2f}</td>
        <td class="num rsr">{team['resumeRank']}</td>
        <td class="sp">{sparkline(team['trajectory'], color=INK['series'])}</td>
      </tr>""")

    return f"""<title>MRI — Broadsheet</title>
<style>
:root {{
  --surface:{INK['surface']}; --plane:{INK['plane']}; --primary:{INK['primary']};
  --secondary:{INK['secondary']}; --muted:{INK['muted']}; --grid:{INK['grid']};
  --axis:{INK['axis']}; --up:{INK['up']}; --down:{INK['down']};
  color-scheme: light;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --surface:{INK_DARK['surface']}; --plane:{INK_DARK['plane']}; --primary:{INK_DARK['primary']};
    --secondary:{INK_DARK['secondary']}; --muted:{INK_DARK['muted']}; --grid:{INK_DARK['grid']};
    --axis:{INK_DARK['axis']}; --up:{INK_DARK['up']}; --down:{INK_DARK['down']};
    color-scheme: dark;
  }}
}}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--plane); color:var(--primary);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif; }}
.wrap {{ max-width:1000px; margin:0 auto; padding-block:32px; padding-left:20px; padding-right:20px; }}
header {{ border-bottom:3px solid var(--primary); padding-bottom:10px; margin-bottom:4px; }}
h1 {{ margin:0; font-size:38px; letter-spacing:-0.03em; font-weight:800; }}
h1 span {{ font-weight:300; color:var(--secondary); }}
.kicker {{ display:flex; flex-wrap:wrap; gap:18px; padding:10px 0 22px;
  border-bottom:1px solid var(--grid); font-size:12px; color:var(--secondary);
  text-transform:uppercase; letter-spacing:0.08em; }}
.kicker b {{ color:var(--primary); font-variant-numeric:tabular-nums; }}
.lede {{ font-size:15px; color:var(--secondary); line-height:1.55; margin:20px 0 26px; max-width:62ch; }}
.tablewrap {{ overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
thead th {{ text-align:left; font-size:10px; letter-spacing:0.1em; text-transform:uppercase;
  color:var(--muted); font-weight:600; padding:0 8px 8px; border-bottom:1px solid var(--axis); white-space:nowrap; }}
tbody td {{ padding:7px 8px; border-bottom:1px solid var(--grid); vertical-align:middle; }}
tbody tr:hover {{ background:var(--surface); }}
.rk {{ font-size:16px; font-weight:700; width:34px; font-variant-numeric:tabular-nums; }}
.mvc {{ width:38px; }}
.mv {{ font-size:11px; font-variant-numeric:tabular-nums; }}
.mv.up {{ color:var(--up); }} .mv.down {{ color:var(--down); }} .mv.flat {{ color:var(--muted); }}
.tm {{ display:flex; align-items:center; gap:9px; min-width:220px; }}
.rule {{ width:3px; height:22px; border-radius:1px; flex:none; }}
.logo {{ flex:none; object-fit:contain; }}
.chip {{ width:22px; height:22px; border-radius:3px; flex:none; display:inline-block; }}
.nm {{ font-weight:600; white-space:nowrap; }}
.flag {{ font-size:10px; color:var(--secondary); border:1px solid var(--axis);
  border-radius:3px; padding:1px 5px; white-space:nowrap; }}
.cf {{ color:var(--secondary); font-size:12px; white-space:nowrap; }}
.rec, .num {{ font-variant-numeric:tabular-nums; }}
.rec {{ color:var(--secondary); }}
.num {{ text-align:right; }}
.pw {{ font-weight:700; }}
.rs {{ color:var(--secondary); }}
.rsr {{ color:var(--muted); font-size:12px; }}
.sp {{ width:70px; }}
footer {{ margin-top:26px; font-size:12px; color:var(--muted); line-height:1.6; }}
@media (max-width:640px) {{ h1 {{ font-size:27px; }} .cf, .rsr {{ display:none; }} }}
</style>
<div class="wrap">
  <header><h1>The MRI <span>/ college football</span></h1></header>
  <div class="kicker">
    <div>Week <b>{payload['week']}</b></div>
    <div>Season <b>{payload['season']}</b></div>
    <div>Games rated <b>{payload['gamesRated']}</b></div>
    <div>Home field <b>{payload['homeField']:.1f}</b></div>
  </div>
  <p class="lede">Power is points against an average FBS team, so the gap between
  two rows is a predicted spread. Résumé is wins above what an average team would
  have managed against the same schedule. Through two weeks Power is still mostly
  preseason prior &mdash; Résumé is the column reflecting what has actually happened.</p>
  <div class="tablewrap">
  <table>
    <thead><tr>
      <th>#</th><th></th><th>Team</th><th>Conf</th><th>Rec</th>
      <th style="text-align:right">Power</th><th style="text-align:right">Résumé</th>
      <th style="text-align:right">R#</th><th>Trend</th>
    </tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  </div>
  <footer>Tagged rows are where MRI Classic disagrees by 15 places or more.</footer>
</div>"""


def console(payload: dict) -> str:
    top = payload["teams"][0]
    disagreements = sorted(
        [t for t in payload["teams"] if t.get("classicRank")],
        key=lambda t: -abs(t["classicRank"] - t["rank"]),
    )[:4]

    cards = []
    for team in payload["teams"][:24]:
        gap = (team["classicRank"] - team["rank"]) if team.get("classicRank") else None
        cards.append(f"""
      <li class="row" style="--team:{team['color']}">
        <div class="rk">{team['rank']}</div>
        <div class="mid">
          <div class="top">{_identity(team, 20)}<span class="nm">{team['team']}</span>
            {_movement(team['movement'])}</div>
          <div class="sub">{team['conference']} &middot; {team['wins']}-{team['losses']}
            {f'&middot; Classic #{team["classicRank"]}' if team.get('classicRank') else ''}</div>
        </div>
        <div class="spark">{sparkline(team['trajectory'], 58, 22, INK_DARK['series'])}</div>
        <div class="vals">
          <div class="pw">{team['power']:+.1f}</div>
          <div class="rs">res {team['resume']:+.2f}</div>
        </div>
      </li>""")

    gaps = "".join(
        f"""<li><span class="gnm">{t['team']}</span>
        <span class="gv">#{t['rank']} <em>vs</em> #{t['classicRank']}</span></li>"""
        for t in disagreements
    )

    return f"""<title>MRI — Console</title>
<style>
:root {{
  --surface:{INK_DARK['surface']}; --plane:{INK_DARK['plane']}; --primary:{INK_DARK['primary']};
  --secondary:{INK_DARK['secondary']}; --muted:{INK_DARK['muted']}; --grid:{INK_DARK['grid']};
  --up:{INK_DARK['up']}; --down:{INK_DARK['down']}; --series:{INK_DARK['series']};
  color-scheme: dark;
}}
@media (prefers-color-scheme: light) {{
  :root:not([data-theme="dark"]) {{
    --surface:{INK['surface']}; --plane:{INK['plane']}; --primary:{INK['primary']};
    --secondary:{INK['secondary']}; --muted:{INK['muted']}; --grid:{INK['grid']};
    --up:{INK['up']}; --down:{INK['down']}; --series:{INK['series']};
    color-scheme: light;
  }}
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--plane); color:var(--primary);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif; }}
.wrap {{ max-width:1060px; margin:0 auto; padding-block:28px; padding-left:18px; padding-right:18px; }}
.masthead {{ display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; margin-bottom:20px; }}
.mark {{ font-size:24px; font-weight:800; letter-spacing:-0.02em; }}
.mark em {{ font-style:normal; color:var(--series); }}
.meta {{ font-size:12px; color:var(--muted); letter-spacing:0.06em; text-transform:uppercase; }}
.grid {{ display:grid; grid-template-columns:1.55fr 1fr; gap:18px; align-items:start; }}
.panel {{ background:var(--surface); border:1px solid var(--grid); border-radius:12px; padding:16px; }}
.ptitle {{ font-size:11px; text-transform:uppercase; letter-spacing:0.1em;
  color:var(--muted); margin:0 0 12px; font-weight:600; }}
.hero {{ display:flex; align-items:center; gap:14px; margin-bottom:6px; }}
.heronum {{ font-size:48px; font-weight:800; line-height:1; letter-spacing:-0.03em; }}
.heronm {{ font-size:15px; font-weight:600; }}
.herosub {{ font-size:12px; color:var(--secondary); }}
ul {{ list-style:none; margin:0; padding:0; }}
.row {{ display:flex; align-items:center; gap:11px; padding:8px 10px; border-radius:8px;
  border-left:3px solid var(--team); }}
.row:hover {{ background:color-mix(in oklab, var(--team) 9%, transparent); }}
.rk {{ font-size:13px; font-weight:700; width:22px; color:var(--secondary);
  font-variant-numeric:tabular-nums; flex:none; }}
.mid {{ flex:1; min-width:0; }}
.top {{ display:flex; align-items:center; gap:7px; }}
.logo {{ flex:none; object-fit:contain; }}
.chip {{ width:20px; height:20px; border-radius:3px; flex:none; display:inline-block; }}
.nm {{ font-weight:600; font-size:14px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.sub {{ font-size:11px; color:var(--muted); margin-top:2px; }}
.mv {{ font-size:10px; font-variant-numeric:tabular-nums; }}
.mv.up {{ color:var(--up); }} .mv.down {{ color:var(--down); }} .mv.flat {{ color:var(--muted); }}
.spark {{ flex:none; }}
.vals {{ text-align:right; flex:none; min-width:64px; }}
.pw {{ font-size:15px; font-weight:700; font-variant-numeric:tabular-nums; }}
.rs {{ font-size:11px; color:var(--muted); font-variant-numeric:tabular-nums; }}
.gap li {{ display:flex; justify-content:space-between; gap:10px; padding:7px 0;
  border-bottom:1px solid var(--grid); font-size:13px; }}
.gap li:last-child {{ border-bottom:none; }}
.gnm {{ font-weight:600; }}
.gv {{ color:var(--secondary); font-variant-numeric:tabular-nums; }}
.gv em {{ font-style:normal; color:var(--muted); }}
.note {{ font-size:12px; color:var(--secondary); line-height:1.55; margin:10px 0 0; }}
.conf li {{ display:flex; align-items:center; gap:8px; padding:5px 0; font-size:12px; }}
.confbar {{ height:8px; border-radius:0 4px 4px 0; background:var(--series); flex:none; }}
.confnm {{ width:92px; color:var(--secondary); flex:none; }}
.confv {{ color:var(--muted); font-variant-numeric:tabular-nums; }}
@media (max-width:820px) {{ .grid {{ grid-template-columns:1fr; }} }}
</style>
<div class="wrap">
  <div class="masthead">
    <div class="mark">The <em>MRI</em></div>
    <div class="meta">{payload['season']} &middot; Week {payload['week']} &middot;
      {payload['gamesRated']} games &middot; HFA {payload['homeField']:.1f}</div>
  </div>
  <div class="grid">
    <div class="panel">
      <p class="ptitle">Power rating &mdash; points vs an average FBS team</p>
      <ul>{''.join(cards)}</ul>
    </div>
    <div>
      <div class="panel" style="margin-bottom:18px">
        <p class="ptitle">Number one</p>
        <div class="hero">
          <div class="heronum">{top['power']:+.0f}</div>
          <div><div class="heronm">{top['team']}</div>
          <div class="herosub">{top['wins']}-{top['losses']} &middot; résumé #{top['resumeRank']}</div></div>
        </div>
      </div>
      <div class="panel" style="margin-bottom:18px">
        <p class="ptitle">Where Classic disagrees</p>
        <ul class="gap">{gaps}</ul>
        <p class="note">MRI 2.0 rank versus the 2018 formula. Classic has no prior
        and weights raw win percentage heavily, so in September an unbeaten team
        that has played nobody rides high.</p>
      </div>
      <div class="panel">
        <p class="ptitle">Conference strength</p>
        <ul class="conf">{_conf_bars(payload)}</ul>
      </div>
    </div>
  </div>
</div>"""


def _conf_bars(payload: dict) -> str:
    rows = payload["conferences"]
    widest = max(abs(c["mean"]) for c in rows) or 1
    out = []
    for conference in rows:
        width = abs(conference["mean"]) / widest * 108
        out.append(
            f'<li><span class="confnm">{conference["conference"]}</span>'
            f'<span class="confbar" style="width:{width:.0f}px"></span>'
            f'<span class="confv">{conference["mean"]:+.1f}</span></li>'
        )
    return "".join(out)


def write_mockups(payload: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for name, render in (("direction-a-broadsheet", broadsheet), ("direction-b-console", console)):
        path = out_dir / f"{name}.html"
        path.write_text(render(payload))
        files.append(path)
    return files


def load(path: Path) -> dict:
    return json.loads(path.read_text())
