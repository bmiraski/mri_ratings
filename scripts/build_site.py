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

from mri.betting import board, tracker  # noqa: E402
from mri.export import gamedaydata, heismandata, logos, simdata, site, sitedata, slate, slatearchive  # noqa: E402
from mri.ratings import priors  # noqa: E402

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

    add_football_extras(payload, data_dir)

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
        add_bracketology(basketball, data_dir)
        add_basketball_slate(basketball, data_dir, public / "basketball")
        # The per-team detail is 4MB and is already rendered into every team page.
        files = site.build(basketball, public, publish_details=False)
        top = basketball["teams"][0]
        print(f"  wrote {len(files)} files - #1 {top['team']} ({top['power']:+.1f})")


def add_football_extras(payload: dict, data_dir: Path) -> None:
    """The season simulation, this week's slate and the public record.

    Each is isolated the way basketball is, and for the same reason: they sit on
    top of the rankings, and a failure in one must not stop the rankings being
    published. What fails is left off - its nav link and page disappear - rather
    than rendered from stale numbers.
    """
    weekly = sitedata.weekly_ratings(SEASON)

    model = priors.load_model()
    if model:
        payload["priorModel"] = model
        backtest = data_dir / "prior_backtest.json"
        if backtest.exists():
            payload["priorBacktest"] = json.loads(backtest.read_text())

    sim = None
    try:
        sim = simdata.build(SEASON, payload, weekly, data_dir / "sim_history.json")
        payload["sim"] = sim
        top = max(sim["teams"].items(), key=lambda kv: kv[1]["title"])
        print(f"  simulation: {sim['sims']:,} runs, most likely champion {top[0]} "
              f"({top[1]['title']:.1%})")
        backtest = data_dir / "sim_backtest.json"
        if backtest.exists():
            payload["simBacktest"] = json.loads(backtest.read_text())
    except Exception as exc:  # noqa: BLE001
        print(f"  simulation skipped: {exc}")

    try:
        payload["slate"] = slate.build(SEASON, payload, payload.get("board") or {}, sim, weekly)
        if payload["slate"]:
            print(f"  slate: week {payload['slate']['week']}, {payload['slate']['games']} games")
    except Exception as exc:  # noqa: BLE001
        print(f"  slate skipped: {exc}")

    # Before this build overwrites docs/slate.json, keep the outgoing week if the slate has moved on.
    try:
        saved = slatearchive.snapshot(ROOT / "docs", payload.get("slate"), SEASON, _finals)
        if saved:
            print(f"  slate archive: saved {saved.relative_to(ROOT)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  slate archive skipped: {exc}")

    try:
        payload["gameday"] = gamedaydata.build(SEASON, payload)
        if payload["gameday"]:
            bt = {}
            for key, name in (("choice", "gameday_backtest.json"), ("forecast", "gameday_forecast_backtest.json")):
                path = data_dir / name
                if path.exists():
                    bt[key] = json.loads(path.read_text())
            payload["gamedayBacktest"] = bt
            print(f"  gameday: {len(payload['gameday']['weeks'])} weeks forecast")
    except Exception as exc:  # noqa: BLE001
        print(f"  gameday skipped: {exc}")

    try:
        payload["heisman"] = heismandata.build(SEASON, payload, data_dir / "heisman_history.json")
        if payload["heisman"]:
            payload["heismanBacktest"] = {
                key: json.loads((data_dir / name).read_text())
                for key, name in (("final", "heisman_backtest.json"), ("forecast", "heisman_forecast_backtest.json"))
                if (data_dir / name).exists()}
            top = payload["heisman"]["players"][0]
            print(f"  heisman: week {payload['heisman']['week']}, {top['player']} ({top['team']}) leads at {top['win']:.1%}"
                  f"{' [voting closed]' if payload['heisman']['closed'] else ''}")
    except Exception as exc:  # noqa: BLE001
        print(f"  heisman skipped: {exc}")

    # After the odds, which it needs: anyone the model gives more than 5% is not hidden.
    try:
        from mri.export import hidden_heisman
        payload["hiddenHeisman"] = hidden_heisman.build(SEASON, payload, weekly, data_dir / "hidden_heisman.json", ROOT)
        latest = payload["hiddenHeisman"]["latest"]
        if latest:
            print(f"  hidden heisman: week {latest['week']}, {latest['player']} ({latest['team']})")
    except Exception as exc:  # noqa: BLE001
        print(f"  hidden heisman skipped: {exc}")

    if payload.get("board"):
        try:
            payload["record"] = tracker.build(
                SEASON, payload, weekly, payload["board"], data_dir / "picks.json")
            fwd = payload["record"]["forward"]
            print(f"  record: {payload['record']['reconstructed'].get('summary', {}).get('games', 0)} "
                  f"games reconstructed, {fwd['logged']} picks logged")
        except Exception as exc:  # noqa: BLE001
            print(f"  record skipped: {exc}")


def _finals(week: int) -> dict[int, tuple[int, int]]:
    """Final scores for a week of this season, keyed by game id: (home points, away points)."""
    from mri.ingest import cfbd

    played = cfbd.games(SEASON)
    played = played[played["week"] == week]
    return {int(g.game_id): (int(g.pts2), int(g.pts1)) for g in played.itertuples()}


def add_basketball_slate(basketball: dict, data_dir: Path, root: Path) -> None:
    """Today's slate, the two tracked betting habits, and yesterday's slate saved with its finals.

    After bracketology, whose leverage is the slate's bid stakes from Christmas on. Isolated like the
    rest: a failure here leaves the slate and the tracker off rather than the basketball site.
    """
    from mri.betting import bb_tracker
    from mri.export import bb_slate

    board = basketball.get("board") or {}
    season = board.get("season") or basketball["season"]
    strategies = data_dir / "bb_strategies.json"
    if strategies.exists():
        basketball["bbStrategies"] = json.loads(strategies.read_text())
    try:
        picks = data_dir / "bb_picks.json"
        if board.get("games") or picks.exists():
            basketball["bbTracker"] = bb_tracker.build(season, board, picks)
            record = basketball["bbTracker"]["record"]
            print(f"  tracker: {len(basketball['bbTracker']['fades'])} on the fade list; "
                  f"dog picks {record['dog']['wins']}-{record['dog']['losses']}, "
                  f"fades {record['fade']['wins']}-{record['fade']['losses']}")
    except Exception as exc:  # noqa: BLE001
        print(f"  tracker skipped: {exc}")
    try:
        slate_data = bb_slate.build(season, basketball, board, basketball.get("bbTracker"))
        saved = bb_slate.snapshot(root, slate_data, lambda day: _bb_finals(season, day))
        if saved:
            print(f"  slate archive: saved {saved.relative_to(ROOT)}")
        basketball["slate"] = slate_data
        if slate_data:
            print(f"  slate: {slate_data['date']}, {slate_data['games']} games, key games by {slate_data['watchKind']}")
    except Exception as exc:  # noqa: BLE001
        print(f"  slate skipped: {exc}")


def _bb_finals(season: int, day: str) -> dict[int, tuple[int, int]]:
    """Final scores for one Eastern-time day of basketball: game id -> (home points, away points)."""
    import pandas as pd

    from mri.export.live import EASTERN
    from mri.ingest import cbbd

    played = cbbd.games(season)
    when = pd.to_datetime(played["start_date"], utc=True).dt.tz_convert(EASTERN).dt.date.astype(str)
    played = played[when == day]
    return {int(g.game_id): (int(g.pts2), int(g.pts1)) for g in played.itertuples()}


def add_bracketology(basketball: dict, data_dir: Path) -> None:
    """The NCAA Tournament projection, Christmas to the following July.

    Isolated like everything else on top of the rankings: if it fails, the pages and nav link are
    left off rather than the basketball site. Outside its calendar window it returns nothing and
    there is nothing to show.
    """
    from mri.export import bracketdata

    try:
        data = bracketdata.build(basketball, data_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"  bracketology skipped: {exc}")
        return
    if not data:
        return
    basketball["bracketology"] = data
    backtest = ROOT / "data" / "bracketology_backtest.json"
    if backtest.exists():
        basketball["bracketologyBacktest"] = json.loads(backtest.read_text())
    field = data["projected"]["field"]
    ones = ", ".join(f["team"] for f in field if f["seedLine"] == 1)
    print(f"  bracketology: {'frozen' if data.get('frozen') else 'live'}, No. 1 seeds {ones}"
          + (f"; format warnings: {len(data.get('formatWarnings') or [])}" if data.get("formatWarnings") else ""))


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
        payload = bb_sitedata.build_full(None, data_dir)
    except Exception as exc:  # noqa: BLE001 - one sport never blocks the other
        print(f"  basketball skipped: {exc}")
        return None

    # The method page explains the preseason prior; it is not published data.
    try:
        from mri.ingest import bb_registry
        from mri.ratings import bb_priors

        model = bb_priors.load_model()
        if model:
            payload["priorModel"] = model
            backtest = data_dir / "bb_prior_backtest.json"
            if backtest.exists():
                payload["priorBacktest"] = json.loads(backtest.read_text())
            payload["priorState"] = bb_priors.status(bb_registry.CURRENT_SEASON, list(bb_registry.teams()))
            print(f"  basketball prior: {payload['priorState']['mode']} "
                  f"({payload['priorState']['teamsWithRosters']} of {payload['priorState']['teams']} rosters)")
    except Exception as exc:  # noqa: BLE001
        print(f"  basketball prior note skipped: {exc}")
    return payload


if __name__ == "__main__":
    main()
