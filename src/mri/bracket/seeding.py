"""From a true seed list to seed lines: which team is a 5, which is a 12, and who
plays its way in.

The committee ranks the whole field 1 to N (the "true seed list" - N is 76 from
2027, 68 before it) and then assigns seed lines. For most of the list that is
four teams to a line, in order. The exceptions are the Opening Round (the First
Four, before 2027), which the NCAA defines by bid type rather than by rank:

*From 2027 (76 teams):* the 12 lowest-ranked at-large teams play in, as four
No. 11s and eight No. 12s, and the 12 lowest-ranked automatic qualifiers play in,
as four No. 15s and eight No. 16s. Every other team goes straight to the
64-team bracket. So line 12 is entirely at-large Opening Round teams, and a
mid-major champion who would have been a 12 under the old format is a 13 now -
there is no direct No. 12 slot left for it. That is a real consequence of the new
format, not a quirk of this code.

**An open question the first 76-team bracket will settle.** With only two direct
No. 11 slots and no direct No. 12s, an at-large team good enough to skip the
Opening Round can still rank below more than ten automatic qualifiers - and then
the strict reading above makes it a direct No. 13, a line *below* weaker at-large
teams playing in as 11s and 12s. Replaying 2024-25 under the new format does this
to one team (UConn). The committee might instead drop a strong champion to 13 and
keep the at-large team on 11 (its principles allow moving a team a line, two in
"extraordinary circumstances"). Nothing published says which; this follows the
announced rule to the letter until a real bracket shows otherwise.

*Through 2026 (68 teams):* the 4 lowest-ranked at-large teams and the 4 lowest-
ranked automatic qualifiers played in. The committee put the at-large pair on
whichever line they fell on - usually 11, occasionally 10 or 12 - and this puts
them on 11, which is the modern norm. Only historical grading uses this format.

Source: NCAA, "Division I Men's Basketball Championship - Selections"
(bracketing principles, updated for the 76-team field), and the NCAA's
May 2026 announcement of the expanded bracket.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Format:
    """``direct`` is how many teams go straight onto each seed line; ``opening_at_large`` and
    ``opening_auto`` are how many Opening Round *teams* land on each line (two per game, so a
    line with eight Opening Round teams has four games feeding four bracket slots)."""

    size: int
    direct: dict[int, int]
    opening_at_large: dict[int, int]
    opening_auto: dict[int, int]

    @property
    def opening_at_large_count(self) -> int:
        return sum(self.opening_at_large.values())

    @property
    def opening_auto_count(self) -> int:
        return sum(self.opening_auto.values())

    def direct_slots(self) -> np.ndarray:
        """The seed line of every direct slot, best first: [1,1,1,1,2,2,2,2,...]."""
        return np.array([line for line in sorted(self.direct) for _ in range(self.direct[line])], dtype=int)


_TOP_TEN = {line: 4 for line in range(1, 11)}

FORMATS = {
    76: Format(size=76, direct={**_TOP_TEN, 11: 2, 13: 4, 14: 4, 15: 2},
               opening_at_large={11: 4, 12: 8}, opening_auto={15: 4, 16: 8}),
    68: Format(size=68, direct={**_TOP_TEN, 11: 2, 12: 4, 13: 4, 14: 4, 15: 4, 16: 2},
               opening_at_large={11: 4}, opening_auto={16: 4}),
}


def field_size(season: int) -> int:
    """76 from the 2026-27 season (the March 2027 tournament) on, 68 before it."""
    return 76 if season >= 2027 else 68


def lines(is_auto: np.ndarray, fmt: Format) -> tuple[np.ndarray, np.ndarray]:
    """Seed line and Opening Round flag for a field already sorted best-first by true seed.

    ``is_auto`` is a boolean per team, in true-seed order. Returns ``(line, opening)`` in that
    same order. Fast enough to call once per simulated world.
    """
    is_auto = np.asarray(is_auto, dtype=bool)
    n = len(is_auto)
    if n != fmt.size:
        raise ValueError(f"{n} teams for a {fmt.size}-team format")
    line = np.zeros(n, dtype=int)
    opening = np.zeros(n, dtype=bool)

    at_large_pos = np.flatnonzero(~is_auto)
    auto_pos = np.flatnonzero(is_auto)
    if len(at_large_pos) < fmt.opening_at_large_count or len(auto_pos) < fmt.opening_auto_count:
        raise ValueError("too few teams of one bid type to fill the Opening Round")

    for positions, spec in ((at_large_pos[len(at_large_pos) - fmt.opening_at_large_count:], fmt.opening_at_large),
                            (auto_pos[len(auto_pos) - fmt.opening_auto_count:], fmt.opening_auto)):
        cursor = 0
        for seed_line in sorted(spec):                 # best of the Opening Round group takes the better line
            take = positions[cursor:cursor + spec[seed_line]]
            line[take] = seed_line
            opening[take] = True
            cursor += spec[seed_line]

    line[~opening] = fmt.direct_slots()
    return line, opening


def opening_round_games(teams_on_line: list[str]) -> list[tuple[str, str]]:
    """Pair one line's Opening Round teams into games, adjacent on the true seed list (the best
    two meet, then the next two). The NCAA's principles name *who* plays in but publish no rule for
    the exact pairings, so this is an assumption, and labelled as one wherever it's shown."""
    return [(teams_on_line[i], teams_on_line[i + 1]) for i in range(0, len(teams_on_line) - 1, 2)]
