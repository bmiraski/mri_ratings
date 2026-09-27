"""The live-score job: update each sport's live.json while its slate's games are on.

Two modes:

    PYTHONPATH=src python3 scripts/live_scores.py            one tick, no commit
    PYTHONPATH=src python3 scripts/live_scores.py --watch    what the Action runs

``--watch`` ticks, commits and pushes any change, and keeps going every five
minutes for as long as a game is on - so one run covers a whole game window and
the job no longer depends on GitHub starting a scheduled run every fifteen
minutes, which it does not (on this repo it started about one in twenty). If
nothing is on but a kickoff or tip is close, it waits for it rather than leaving
the next run to catch it. It stops when every game that started is final and
nothing starts soon, or when its time budget is spent, well inside the job's
six-hour limit; the next dispatch picks up from there.

Spends an API call only while a game is about to start or in progress. Standard
library only, so the Action needs no installs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mri.export import live  # noqa: E402

SPORTS = ((live.FOOTBALL, ROOT / "docs"), (live.BASKETBALL, ROOT / "docs" / "basketball"))
LIVE_FILES = ("docs/live.json", "docs/basketball/live.json")

INTERVAL = dt.timedelta(minutes=5)       # between scoreboard reads while a game is on
HORIZON = dt.timedelta(minutes=20)       # wait for a kickoff this close rather than exit
BUDGET = dt.timedelta(hours=5, minutes=30)


def tick(now: dt.datetime | None = None) -> None:
    for sport, root in SPORTS:
        try:
            print(f"{sport.name}: {live.run(root, sport=sport, now=now)}", flush=True)
        except Exception as exc:  # noqa: BLE001 - one sport's failure must not cost the other its scores
            print(f"{sport.name}: failed - {exc}", flush=True)


def next_window(now: dt.datetime) -> dt.datetime | None:
    windows = [w for sport, root in SPORTS if (w := live.window(root, now=now, sport=sport)) is not None]
    return min(windows, default=None)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)


def commit(now: dt.datetime) -> bool:
    """Commit and push whichever live.json changed. The daily build never touches one, so a rebase is clean."""
    present = [f for f in LIVE_FILES if (ROOT / f).exists()]
    if not present:
        return False
    _git("add", *present)
    if _git("diff", "--staged", "--quiet").returncode == 0:
        return False
    _git("commit", "-m", f"Live scores ({now.astimezone(dt.timezone.utc):%Y-%m-%d %H:%M})")
    for attempt in range(3):
        if _git("pull", "--rebase", "--quiet").returncode == 0 and _git("push", "--quiet").returncode == 0:
            print("pushed", flush=True)
            return True
        time.sleep(5 * (attempt + 1))
    raise SystemExit("push failed three times")


def watch(*, clock=lambda: dt.datetime.now(dt.timezone.utc), sleep=time.sleep, run_tick=tick,
          find_window=next_window, save=commit, pull=lambda: _git("pull", "--rebase", "--quiet"),
          interval: dt.timedelta = INTERVAL, horizon: dt.timedelta = HORIZON,
          budget: dt.timedelta = BUDGET) -> str:
    """Tick, save and sleep while a game is on; wait out a close kickoff; stop otherwise."""
    started = clock()
    while True:
        now = clock()
        pull()                                   # the daily build may have moved the slate on
        opens = find_window(now)
        if opens is None or opens - now > horizon:
            return "nothing on"
        if opens > now:
            if opens - started > budget:
                return "out of time"
            print(f"waiting for {opens:%H:%M} UTC", flush=True)
            sleep((opens - now).total_seconds())
            continue
        run_tick(now)
        save(now)
        if clock() + interval - started > budget:
            return "out of time"
        sleep(interval.total_seconds())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--watch", action="store_true", help="loop through the game window, committing each change")
    if parser.parse_args().watch:
        print(watch(), flush=True)
    else:
        tick()
