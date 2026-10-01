"""Is the schedule graph connected? The ridge solve will not tell you.

Every rating in MRI 2.0 is anchored to the others through the games they played.
A team whose games link it to nobody in the main body of the field has nothing
to be measured against, and the ridge penalty quietly leaves it at its prior:
the solve succeeds, the table prints, and one team (or one conference-sized
island) is simply not rated. Two failures look like this and are worth telling
apart:

*An island.* Teams that play only each other, or that reach the rest of the
field only through unrated nodes. Real in basketball (small conferences with
thin cross-conference schedules) and in football now that FCS opponents are
named individually.

*A team with zero games.* Almost always a name mismatch - the registry's
spelling and the feed's do not agree, so the team "silently loses half its
schedule" (see the note atop ``ingest/registry.py``). It never enters the
solve at all, which is a different failure from sitting at the prior.

Two views are kept distinct:

``loose``  every team in the table. A one-game FCS or non-D1 opponent is a leaf
           attached by its single game; that is connected, not a disconnect.
``strict`` only the rated field (FBS for football, D1 for basketball), counting
           only games between two rated teams. A rated team whose only path to
           the rest runs through unrated nodes shows up here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


class ConnectivityWarning(UserWarning):
    """A slice's schedule graph has teams outside its main component."""


class ConnectivityError(RuntimeError):
    """Raised by production builds when a rated team is disconnected."""


_ORPHAN_COLUMNS = ["team", "games", "component"]


@dataclass(frozen=True)
class View:
    """Components of one graph: how many, how big the main one is, who is outside it."""

    nodes: int
    components: int
    main_size: int
    orphans: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=_ORPHAN_COLUMNS))

    @property
    def connected(self) -> bool:
        return self.orphans.empty


@dataclass(frozen=True)
class Connectivity:
    loose: View
    strict: View | None = None
    zero_game: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """No rated team is stranded and none is missing. Falls back to the loose view
        when no rated field was given."""
        view = self.strict if self.strict is not None else self.loose
        return view.connected and not self.zero_game

    def summary(self) -> str:
        parts = []
        view = self.strict if self.strict is not None else self.loose
        label = "rated" if self.strict is not None else "all"
        if not view.connected:
            names = ", ".join(view.orphans["team"].head(8))
            more = len(view.orphans) - 8
            parts.append(
                f"{len(view.orphans)} {label} team(s) outside the main component "
                f"({view.components} components, main = {view.main_size}): {names}"
                + (f", +{more} more" if more > 0 else "")
            )
        if self.zero_game:
            names = ", ".join(self.zero_game[:8])
            more = len(self.zero_game) - 8
            parts.append(
                f"{len(self.zero_game)} team(s) with zero games: {names}"
                + (f", +{more} more" if more > 0 else "")
            )
        return "; ".join(parts) if parts else "connected"


def _components(
    away: np.ndarray, home: np.ndarray, nodes: list[str], games_by_team: pd.Series
) -> View:
    index = {team: i for i, team in enumerate(nodes)}
    n = len(nodes)
    if n == 0:
        return View(nodes=0, components=0, main_size=0)
    rows = np.fromiter((index[t] for t in away), dtype=int, count=len(away))
    cols = np.fromiter((index[t] for t in home), dtype=int, count=len(home))
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    count, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels, minlength=count)
    main = int(np.argmax(sizes))
    outside = np.flatnonzero(labels != main)
    orphans = pd.DataFrame(
        {
            "team": [nodes[i] for i in outside],
            "games": [int(games_by_team.get(nodes[i], 0)) for i in outside],
            "component": [int(labels[i]) for i in outside],
        },
        columns=_ORPHAN_COLUMNS,
    ).sort_values(["component", "team"], ignore_index=True)
    return View(nodes=n, components=int(count), main_size=int(sizes[main]), orphans=orphans)


def analyze(
    games: pd.DataFrame,
    rated=None,
    expected=None,
) -> Connectivity:
    """Connected components of the game graph, loose and (given ``rated``) strict.

    Parameters
    ----------
    games
        Any table with ``team1`` and ``team2``.
    rated
        The rated field. Enables the strict view, which keeps only games between
        two rated teams. Rated teams with no games are reported as zero-game, not
        as orphans.
    expected
        Teams that ought to appear in ``games`` (a season's registry roster).
        Anything missing is reported as zero-game. Defaults to ``rated``.
    """
    away = games["team1"].to_numpy()
    home = games["team2"].to_numpy()
    played = pd.concat([games["team1"], games["team2"]]).value_counts()

    loose = _components(away, home, sorted(played.index), played)

    strict = None
    rated_set = set(rated) if rated is not None else None
    if rated_set is not None:
        both = np.array([a in rated_set and h in rated_set for a, h in zip(away, home)], dtype=bool)
        nodes = sorted(t for t in rated_set if t in played.index)
        strict = _components(away[both], home[both], nodes, played)

    wanted = set(expected) if expected is not None else (rated_set or set())
    zero = tuple(sorted(t for t in wanted if t not in played.index))
    return Connectivity(loose=loose, strict=strict, zero_game=zero)


def assert_connected(conn: Connectivity, label: str = "") -> None:
    """Raise ``ConnectivityError`` unless every rated team is in the main component."""
    if not conn.ok:
        prefix = f"{label}: " if label else ""
        raise ConnectivityError(prefix + conn.summary())
