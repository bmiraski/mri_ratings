"""Refresh the data and regenerate the site.

One entry point for the weekly run: pull whatever games are new, recompute both
ratings week by week, and write the static site.

Output goes to docs/, which is committed to the repo. GitHub Pages serves that
folder directly, so the host never needs to run a build or hold the API key -
whatever produces the files (this script, or the weekly Action) is the only
thing that needs credentials.

Run:  PYTHONPATH=src python3 scripts/build_site.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import json  # noqa: E402

from mri.betting import board  # noqa: E402
from mri.export import logos, site, sitedata  # noqa: E402

SEASON = 2026


def main() -> None:
    data_dir = ROOT / "site" / "data"
    public = ROOT / "docs"

    print(f"building {SEASON}...")
    payload = sitedata.build_full(SEASON, data_dir)

    # The betting page only appears once there is a backtest to be honest about.
    betting_path = data_dir / "betting.json"
    if betting_path.exists():
        payload["betting"] = json.loads(betting_path.read_text())
        payload["board"] = board.build_board(SEASON)
        print(f"  board: week {payload['board']['week']}, "
              f"{len(payload['board']['flagged'])} flagged")
    print(f"  week {payload['week']}, {payload['gamesRated']} games, {len(payload['teams'])} teams")

    summary = logos.cache_logos(payload, public)
    print(f"  logos: {summary['fetched']} fetched, {summary['cached']} cached, "
          f"{summary['failed']} failed")

    from mri.export import seasons as season_archive

    payload["seasons"] = season_archive.football_seasons(current=SEASON)
    payload["history"] = season_archive.team_history(payload["seasons"])
    from mri.export import gamelogs
    payload["gamelogs"] = gamelogs.prune(gamelogs.football(SEASON), payload["history"])
    print(f"  archive: {len(payload['seasons'])} past seasons, "
          f"history for {len(payload['history'])} teams")

    basketball = basketball_payload(data_dir)

    # Both payloads are prepared before either renders, because the sport switch
    # in the header must only offer a sport that this build actually published.
    # Rendering football first would mean its 158 pages carry a link to a
    # basketball directory that the next step might fail to create.
    sports = ["football"] + (["basketball"] if basketball else [])
    payload["sports"] = sports

    files = site.build(payload, public)
    print(f"  wrote {len(files)} files to {public.relative_to(ROOT)}")
    top = payload["teams"][0]
    print(f"  #1 {top['team']} ({top['power']:+.1f})")

    if basketball:
        basketball["sports"] = sports
        basketball["seasons"] = season_archive.basketball_seasons(
            current=basketball["season"] if not basketball.get("finished") else None
        )
        basketball["history"] = season_archive.team_history(basketball["seasons"])
        basketball["gamelogs"] = gamelogs.prune(
            gamelogs.basketball(
                basketball["season"] if not basketball.get("finished") else None
            ),
            basketball["history"],
        )
        print(f"  archive: {len(basketball['seasons'])} past basketball seasons, "
              f"history for {len(basketball['history'])} teams")
        # The basketball page appears only once there is a backtest to be honest
        # about, the same condition football's uses.
        bb_betting = data_dir / "bb_betting.json"
        if bb_betting.exists():
            from mri.betting import bb_board
            basketball["betting"] = json.loads(bb_betting.read_text())
            basketball["board"] = bb_board.build_board(basketball["season"])
            print(f"  board: {basketball['board']['priced']} priced, "
                  f"{len(basketball['board']['disagreements'])} disagreements")
        print(f"building basketball {basketball['seasonLabel']} "
              f"({basketball['periodLabel']})...")
        # Cached into the basketball subtree rather than shared with football's.
        # The stored path is relative to the sport's own root, which is what
        # every page's depth is counted from - one shared directory would need
        # each mark to know which sport it was being drawn for.
        summary = logos.cache_logos(basketball, public / "basketball")
        print(f"  logos: {summary['fetched']} fetched, {summary['cached']} cached, "
              f"{summary['failed']} failed")
        print(f"  {basketball['gamesRated']} games, {len(basketball['teams'])} teams, "
              f"home court {basketball['homeField']:.2f}")
        # The per-team detail is 4MB and is already rendered into every team page.
        files = site.build(basketball, public, publish_details=False)
        top = basketball["teams"][0]
        print(f"  wrote {len(files)} files - #1 {top['team']} ({top['power']:+.1f})")


def basketball_payload(data_dir):
    """The basketball data, or None.

    Kept in the same entry point so one weekly run refreshes both sports and they
    cannot drift apart, but isolated: basketball is out of season for half the
    football year, and a failure there must not take the football site down with
    it. Returning None rather than raising is what lets the switch disappear
    instead of pointing at nothing.
    """
    from mri.export import bb_sitedata

    try:
        return bb_sitedata.build_full(None, data_dir)
    except Exception as exc:  # noqa: BLE001 - one sport never blocks the other
        print(f"  basketball skipped: {exc}")
        return None


if __name__ == "__main__":
    main()
