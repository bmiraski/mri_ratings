"""Placing a seeded field into four regions, by the committee's own published rules.

The committee's principles (NCAA, "Division I Men's Basketball Championship -
Selections", updated for the 76-team field) are specific enough to implement
directly:

1. The four No. 1 seeds go to four regions, and those regions fix the national
   semifinals: overall No. 1's region meets overall No. 4's, No. 2's meets No. 3's.
2. Every later line is placed in true-seed order along an S-curve - the best No. 2
   seed joins the *weakest* No. 1's region, and so on - so the top four lines of
   every region add up to about the same total. The rules ask for no more than a
   six-point spread between regions' top-four-line totals.
3. The first four teams from one conference on the top four lines go to four
   different regions.
4. Two teams from the same conference that played **three or more** times
   (regular season and conference tournament) can't meet before a regional final;
   **twice**, not before a regional semifinal; **once or never**, not in the first
   round. Overall No. 5 shouldn't land in overall No. 1's region.
5. A team may move up or down one seed line to make all of that work.

Placement is greedy along the S-curve, one line at a time, then *repaired*: any
team still breaking a rule tries trading places with another slot on its line,
then with a direct team one line away, keeping any trade that leaves fewer rules
broken. The greedy pass alone gets cornered by a conference with many teams in
the field (the 2012 Big East had nine); with the repair, every rule that can be
met is - in stress tests of nine-team conferences whose teams all met twice, the
only thing ever left is a same-conference Opening Round game when three of a
line's four Opening Round teams share a conference, which no placement fixes.

What this does *not* do is geography. The committee lets the overall No. 1 seed
pick its region and first-weekend site, and keeps teams near home where it can;
none of that is in the principles as a formula, and none of it is attempted here.
So regions are numbered by their No. 1 seed (Region 1 is the overall No. 1's),
not named, and the region a given team lands in is an approximation of the
committee's - its seed line and its path, which the rules above pin down, are the
parts worth reading closely.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import pandas as pd

# Where each seed sits in a 16-team region, top to bottom: 1 plays 16, winner meets the 8/9 winner, etc.
BRACKET_ORDER = (1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15)
_POSITION = {seed: i for i, seed in enumerate(BRACKET_ORDER)}
SEMIFINALS = ((1, 4), (2, 3))
BALANCE_LIMIT = 6


def earliest_meeting(line_a: int, line_b: int) -> int:
    """The earliest round two teams in the same region can meet: 1 (first round) to 4 (regional final)."""
    pa, pb = _POSITION[line_a], _POSITION[line_b]
    for rnd, block in ((1, 2), (2, 4), (3, 8)):
        if pa // block == pb // block:
            return rnd
    return 4


def required_round(times_played: int) -> int:
    """How late two conference-mates must be kept apart, from how often they've already played."""
    if times_played >= 3:
        return 4
    if times_played == 2:
        return 3
    return 2


@dataclass
class Unit:
    """One slot on a seed line: a team, or an Opening Round game (two teams, one slot)."""

    line: int
    teams: tuple[str, ...]
    true_seed: int                     # the better true seed in the unit

    @property
    def is_game(self) -> bool:
        return len(self.teams) == 2


@dataclass
class Placement:
    region_of: dict[str, int]
    line_of: dict[str, int]
    units: list[Unit]
    moved: dict[str, int] = field(default_factory=dict)          # team -> the line it was moved *from*
    violations: list[str] = field(default_factory=list)

    def table(self, seeded: pd.DataFrame) -> pd.DataFrame:
        out = seeded.copy()
        out["region"] = out["team"].map(self.region_of)
        out["seedLine"] = out["team"].map(self.line_of)
        out["movedFrom"] = out["team"].map(self.moved)
        game_of = {t: f"{u.line}-{'-'.join(u.teams)}" for u in self.units if u.is_game for t in u.teams}
        out["openingGame"] = out["team"].map(game_of)
        return out

    def region_totals(self, true_seed: dict[str, int], through_line: int = 4) -> dict[int, int]:
        totals = {r: 0 for r in range(1, 5)}
        for team, region in self.region_of.items():
            if self.line_of[team] <= through_line:
                totals[region] += true_seed[team]
        return totals


class _Checker:
    """Everything the rules need to know about a pair of teams, precomputed once."""

    def __init__(self, seeded: pd.DataFrame, meetings: dict[frozenset, int]):
        self.conf = dict(zip(seeded["team"], seeded["conference"]))
        self.true_seed = dict(zip(seeded["team"], seeded["trueSeed"]))
        self.meetings = meetings
        top = seeded[seeded["seedLine"] <= 4].sort_values("trueSeed")
        # The first four of any conference on the top four lines; a fifth or later is exempt from rule 3.
        self.top_four_rule: set[str] = set()
        for _, group in top.groupby("conference"):
            self.top_four_rule |= set(group["team"].head(4))
        ordered = seeded.sort_values("trueSeed")["team"].tolist()
        self.overall = {i + 1: t for i, t in enumerate(ordered[:5])}
        self.same_game: set[frozenset] = set()        # Opening Round opponents, filled in once games are paired

    def pair_problem(self, a: str, line_a: int, b: str, line_b: int) -> str | None:
        ca, cb = self.conf.get(a), self.conf.get(b)
        if not ca or ca != cb:
            return None
        if frozenset((a, b)) in self.same_game:
            return (f"{a} and {b} ({ca}) meet in the Opening Round - unavoidable when one conference has "
                    f"more than half of a line's Opening Round teams")
        if line_a <= 4 and line_b <= 4 and a in self.top_four_rule and b in self.top_four_rule:
            return f"{a} and {b} ({ca}) share a region on the top four lines"
        played = self.meetings.get(frozenset((a, b)), 0)
        if earliest_meeting(line_a, line_b) < required_round(played):
            return f"{a} and {b} ({ca}, played {played}x) could meet in round {earliest_meeting(line_a, line_b)}"
        return None

    def unit_problems(self, unit: Unit, region: int, placed: dict[int, list[tuple[str, int]]]) -> list[str]:
        problems = []
        for a in unit.teams:
            for b, line_b in placed[region]:
                p = self.pair_problem(a, unit.line, b, line_b)
                if p:
                    problems.append(p)
            if unit.line == 2 and a == self.overall.get(5) and region == 1:
                problems.append(f"overall No. 5 {a} in overall No. 1's region")
        return problems


def _units(seeded: pd.DataFrame, checker: _Checker) -> dict[int, list[Unit]]:
    by_line: dict[int, list[Unit]] = {}
    for line, group in seeded.sort_values("trueSeed").groupby("seedLine"):
        direct = group[~group["opening"]]
        units = [Unit(int(line), (r.team,), int(r.trueSeed)) for r in direct.itertuples()]
        playing_in = group[group["opening"]]["team"].tolist()
        pairs = [(playing_in[i], playing_in[i + 1]) for i in range(0, len(playing_in) - 1, 2)]
        pairs = _untangle_pairs(pairs, checker)
        units += [Unit(int(line), p, min(checker.true_seed[p[0]], checker.true_seed[p[1]])) for p in pairs]
        checker.same_game |= {frozenset(p) for p in pairs}
        by_line[int(line)] = sorted(units, key=lambda u: u.true_seed)
    return by_line


def _untangle_pairs(pairs: list[tuple[str, str]], checker: _Checker) -> list[tuple[str, str]]:
    """Two conference-mates shouldn't open against each other in the Opening Round; swap partners with
    the neighbouring game when they would."""
    pairs = list(pairs)
    for i, (a, b) in enumerate(pairs):
        if checker.conf.get(a) and checker.conf.get(a) == checker.conf.get(b):
            for j in range(len(pairs)):
                if j == i:
                    continue
                c, d = pairs[j]
                if checker.conf.get(a) != checker.conf.get(d) and checker.conf.get(c) != checker.conf.get(b):
                    pairs[i], pairs[j] = (a, d), (c, b)
                    break
    return pairs


def _serpentine(line: int) -> list[int]:
    """The S-curve's default region order for a line: best team of the line first."""
    return [1, 2, 3, 4] if line % 2 == 1 else [4, 3, 2, 1]


def _try_line(units: list[Unit], placed, checker) -> tuple[list[int], list[str]] | None:
    """The valid assignment of this line's units to regions closest to the S-curve, or None."""
    default = _serpentine(units[0].line)[:len(units)]
    options = sorted(itertools.permutations([1, 2, 3, 4], len(units)),
                     key=lambda perm: sum(abs(a - b) for a, b in zip(perm, default)))
    for perm in options:
        if all(not checker.unit_problems(u, r, placed) for u, r in zip(units, perm)):
            return list(perm), []
    return None


def place(seeded: pd.DataFrame, meetings: dict[frozenset, int] | None = None) -> Placement:
    """Place a seeded field into four regions.

    ``seeded`` needs ``team``, ``conference``, ``trueSeed`` (1 = best), ``seedLine`` and ``opening``
    (from :func:`mri.bracket.seeding.lines`). ``meetings`` maps a pair of teams to how many times
    they've played this season, conference tournament included.
    """
    meetings = meetings or {}
    checker = _Checker(seeded, meetings)
    by_line = _units(seeded, checker)
    placed: dict[int, list[tuple[str, int]]] = {r: [] for r in range(1, 5)}
    region_of: dict[str, int] = {}
    line_of: dict[str, int] = {}
    moved: dict[str, int] = {}
    violations: list[str] = []
    all_units: list[Unit] = []

    for line in sorted(by_line):
        units = by_line[line]
        choice = _try_line(units, placed, checker)
        if choice is None and line < 16 and line + 1 in by_line:
            choice = _swap_with_next_line(line, by_line, placed, checker, moved)
            units = by_line[line]
        if choice is None:
            # Nothing valid even with a one-line move: take the S-curve and say so, rather than
            # silently bending a rule. Rare enough in practice that a human should look at it.
            perm = _serpentine(line)[:len(units)]
            for u, r in zip(units, perm):
                violations += checker.unit_problems(u, r, placed)
        else:
            perm = choice[0]
        for u, r in zip(units, perm):
            for t in u.teams:
                region_of[t] = r
                line_of[t] = u.line
                placed[r].append((t, u.line))
        all_units += units

    placement = Placement(region_of, line_of, all_units, moved, violations)
    _repair(placement, checker)
    _balance(placement, checker, by_line)
    return placement


def _problem_teams(placement: Placement, checker: _Checker) -> list[str]:
    """Every team involved in a broken rule, most-involved first."""
    counts: dict[str, int] = {}
    by_region: dict[int, list[str]] = {r: [] for r in range(1, 5)}
    for t, r in placement.region_of.items():
        by_region[r].append(t)
    for teams in by_region.values():
        for a, b in itertools.combinations(teams, 2):
            if checker.pair_problem(a, placement.line_of[a], b, placement.line_of[b]):
                if frozenset((a, b)) in checker.same_game:
                    continue                                   # an Opening Round pairing: no swap fixes it
                counts[a] = counts.get(a, 0) + 1
                counts[b] = counts.get(b, 0) + 1
    return sorted(counts, key=lambda t: -counts[t])


def _repair(placement: Placement, checker: _Checker, max_rounds: int = 60) -> None:
    """Greedy placement looks one line ahead; a conference with many teams in the field (the 2012 Big
    East had nine) can still leave it cornered several lines later. So: for each team still breaking a
    rule, try trading places with another slot on its own line in a different region, then with a
    direct team one line up or down (rule 5) - taking any trade that leaves fewer rules broken."""
    unit_of = {t: u for u in placement.units for t in u.teams}

    def count() -> int:
        return len(bracket_problems(placement, checker))

    def region_swap(u: Unit, v: Unit) -> None:
        ru, rv = placement.region_of[u.teams[0]], placement.region_of[v.teams[0]]
        for t in u.teams:
            placement.region_of[t] = rv
        for t in v.teams:
            placement.region_of[t] = ru

    def line_swap(u: Unit, v: Unit) -> None:
        a, b = u.teams[0], v.teams[0]
        placement.region_of[a], placement.region_of[b] = placement.region_of[b], placement.region_of[a]
        placement.line_of[a], placement.line_of[b] = placement.line_of[b], placement.line_of[a]
        u.line, v.line = v.line, u.line

    current = count()
    for _ in range(max_rounds):
        if current == 0:
            return
        improved = False
        for team in _problem_teams(placement, checker):
            u = unit_of[team]
            same_line = [v for v in placement.units if v is not u and v.line == u.line
                         and placement.region_of[v.teams[0]] != placement.region_of[team]]
            for v in same_line:
                region_swap(u, v)
                after = count()
                if after < current:
                    current, improved = after, True
                    break
                region_swap(u, v)
            if improved:
                break
            if u.is_game or u.line == 1:
                continue
            original = u.line
            for v in [v for v in placement.units if not v.is_game and abs(v.line - u.line) == 1 and v.line != 1]:
                line_swap(u, v)
                after = count()
                if after < current:
                    current, improved = after, True
                    for moved_team, from_line in ((u.teams[0], original), (v.teams[0], u.line)):
                        placement.moved.setdefault(moved_team, from_line)
                    break
                line_swap(u, v)
            if improved:
                break
        if not improved:
            return


def _swap_with_next_line(line, by_line, placed, checker, moved):
    """Rule 5: trade one direct team on this line for one on the next, and try again."""
    here, below = by_line[line], by_line[line + 1]
    candidates = []
    for i, u in enumerate(here):
        for j, v in enumerate(below):
            if u.is_game or v.is_game or line == 1:
                continue
            candidates.append((abs(u.true_seed - v.true_seed), i, j))
    for _, i, j in sorted(candidates):
        u, v = here[i], below[j]
        new_here = sorted(here[:i] + here[i + 1:] + [Unit(line, v.teams, v.true_seed)], key=lambda x: x.true_seed)
        choice = _try_line(new_here, placed, checker)
        if choice is not None:
            by_line[line] = new_here
            by_line[line + 1] = sorted(below[:j] + below[j + 1:] + [Unit(line + 1, u.teams, u.true_seed)],
                                       key=lambda x: x.true_seed)
            moved[v.teams[0]] = line + 1
            moved[u.teams[0]] = line
            return choice
    return None


def _balance(placement: Placement, checker: _Checker, by_line) -> None:
    """Swap same-line teams between regions on lines 2-4 while that narrows the top-four-line spread
    and breaks no rule - the committee's "no more than six points" check, done greedily."""
    def spread() -> int:
        totals = placement.region_totals(checker.true_seed)
        return max(totals.values()) - min(totals.values())

    improved = True
    while improved and spread() > BALANCE_LIMIT:
        improved = False
        for line in (2, 3, 4):
            teams = [t for t, ln in placement.line_of.items() if ln == line]
            for a, b in itertools.combinations(teams, 2):
                before = spread()
                ra, rb = placement.region_of[a], placement.region_of[b]
                placement.region_of[a], placement.region_of[b] = rb, ra
                if spread() < before and not bracket_problems(placement, checker):
                    improved = True
                else:
                    placement.region_of[a], placement.region_of[b] = ra, rb


def bracket_problems(placement: Placement, checker: _Checker) -> list[str]:
    """Every rule the finished bracket breaks (an empty list is a valid bracket)."""
    problems = []
    by_region: dict[int, list[str]] = {r: [] for r in range(1, 5)}
    for t, r in placement.region_of.items():
        by_region[r].append(t)
    for teams in by_region.values():
        for a, b in itertools.combinations(teams, 2):
            p = checker.pair_problem(a, placement.line_of[a], b, placement.line_of[b])
            if p:
                problems.append(p)
    five = checker.overall.get(5)
    if five and placement.region_of.get(five) == 1 and placement.line_of.get(five) == 2:
        problems.append(f"overall No. 5 {five} in overall No. 1's region")
    return problems


def check(seeded: pd.DataFrame, placement: Placement, meetings: dict[frozenset, int] | None = None) -> list[str]:
    """Public validity check, for tests and the backtest: every rule the placed bracket breaks."""
    checker = _Checker(seeded, meetings or {})
    checker.same_game = {frozenset(u.teams) for u in placement.units if u.is_game}
    return bracket_problems(placement, checker)


def meetings_from_games(games: pd.DataFrame) -> dict[frozenset, int]:
    """How many times each pair of teams played in a set of games."""
    counts: dict[frozenset, int] = {}
    for a, b in zip(games["team1"], games["team2"]):
        key = frozenset((a, b))
        counts[key] = counts.get(key, 0) + 1
    return counts
