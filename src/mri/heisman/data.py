"""The Heisman voting record, and finding its players in the stats.

The voting file names people and schools the way the Heisman Trust does; the stat
feed names them the way the data provider does. Joining the two is the whole
difficulty, and it is a real one: "C.J. Stroud" is "CJ Stroud", "Miami (Fla)" is
"Miami", "Mississippi" is "Ole Miss", and a suffix appears or does not.

Matching is by normalized name within a school, so two players of the same name at
different schools cannot be confused, and a name that matches nothing is reported
and never guessed at. ``resolve`` returns the matching stat rows; an empty result
is a fact about the data (a season before coverage, a spelling nobody has mapped)
that a test looks at, not an error the code swallows.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
VOTING = ROOT / "data" / "heisman_voting.json"
PLAYERS = ROOT / "data" / "parquet" / "player_seasons.parquet"

# The Trust's school names -> the stat feed's.
SCHOOLS = {
    "Miami": "Miami", "Miami (Fla)": "Miami", "Mississippi": "Ole Miss", "Pitt": "Pittsburgh", "Hawaii": "Hawai'i",
    "USC": "USC", "Southern California": "USC",
}

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv)\b\.?$")


def normalize(name: str) -> str:
    """Lower case, no accents, no punctuation, no generational suffix."""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"[.'’,\-]", "", text).strip()
    return _SUFFIX.sub("", text).strip()


def load_voting(path: Path = VOTING) -> dict:
    return json.loads(path.read_text())


def last_season(voting: dict | None = None) -> int:
    """The latest season whose Heisman has been awarded and recorded: what every fit and backtest runs through.

    Adding a season's finalists to the voting file is what moves this, and with it the whole pipeline: the
    fitting scripts, the backtest and the yearly rebuilds of the player tables all read it, so a refit is a
    matter of running them and not of editing them.
    """
    voting = voting or load_voting()
    return max(int(y) for y in voting["seasons"])


def key_dates(voting: dict, year: int) -> dict | None:
    """The season's ballot dates (ballots out, deadline, finalists announced, ceremony), if they are known."""
    return (voting.get("keyDates") or {}).get(str(year))


def load_players(path: Path = PLAYERS) -> pd.DataFrame:
    return pd.read_parquet(path)


def same_person(a: str, b: str) -> bool:
    """Whether two normalized names are the same player, allowing a nickname.

    "cam ward" is "cameron ward": same surname, and one first name begins the other
    (at least three letters of it). Within a single school in a single season that is
    safe; across a whole table it would not be.
    """
    if a == b:
        return True
    fa, _, la = a.rpartition(" ")
    fb, _, lb = b.rpartition(" ")
    if la != lb or not fa or not fb:
        return False
    short, long = sorted((fa, fb), key=len)
    return len(short) >= 3 and long.startswith(short)


def resolve(players: pd.DataFrame, season: int, name: str, school: str) -> pd.DataFrame:
    """The stat rows of one player in one season."""
    team = SCHOOLS.get(school, school)
    rows = players[(players["season"] == season) & (players["team"] == team)]
    wanted = normalize(name)
    return rows[rows["player"].map(lambda n: same_person(normalize(n), wanted))]


def finalists(voting: dict, first: int = 0) -> list[dict]:
    """Every listed finisher, with his season, flattened."""
    return [{"season": int(y), **f} for y, s in sorted(voting["seasons"].items()) if int(y) >= first for f in s["finalists"]]
