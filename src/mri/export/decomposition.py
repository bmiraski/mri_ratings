"""The per-team "how this rating is built" payload, shared by football and basketball.

The numbers come from ``ratings.decompose``; this module only shapes them for the
page and refuses to hand over anything that does not tie out to the Power the
page prints.
"""

from __future__ import annotations

from ..ratings import decompose as core

# The published power is rounded to 2 decimals before the page sees it, so the
# parts (summed unrounded) can be that much off it and no more.
POWER_TOLERANCE = 0.006


def shares_for(fit: tuple | None) -> dict[str, core.TeamShare] | None:
    """Decompose the ``(model, games)`` fit that published the ratings, if it can be.

    A fit that cannot be explained (or no fit at all) yields None and the pages
    omit the section; a failure here must never cost the build.
    """
    if fit is None:
        return None
    model, games = fit
    try:
        return core.decompose(model, games)
    except core.DecompositionUnavailable as exc:
        print(f"  rating decomposition skipped: {exc}")
        return None


def share_index(shares: dict[str, core.TeamShare]) -> dict[tuple[str, object], core.GameShare]:
    """(team, game id) -> that game's share of the team's rating."""
    return {(team, g.game_id): g for team, share in shares.items() for g in share.games}


def game_fields(share: core.GameShare) -> dict:
    """Added to a played-game entry. The raw margin is already there as ``margin``."""
    return {
        "counted": round(share.margin_counted, 1),
        "oppRating": round(share.opponent_rating, 1),
        "adds": round(share.contribution, 2),
    }


def team_block(share: core.TeamShare, published_power: float, played: int) -> dict | None:
    """The team-level block, or None when it would not tie out to what is printed."""
    if share.n != played:
        return None
    if abs(share.margin_term + share.opponent_term + share.prior_term - published_power) > POWER_TOLERANCE:
        return None
    return {
        "n": share.n,
        "ridge": share.ridge,
        "w": round(share.w, 4),
        "avgMargin": round(share.avg_margin_counted, 1),
        "avgOpponent": round(share.avg_opponent, 1),
        "prior": round(share.prior, 2),
        "marginTerm": round(share.margin_term, 3),
        "opponentTerm": round(share.opponent_term, 3),
        "priorTerm": round(share.prior_term, 3),
    }


def attach(
    details: dict[str, dict],
    shares: dict[str, core.TeamShare] | None,
    power: dict[str, float],
) -> int:
    """Add ``decomposition`` to every team whose parts tie out. Returns how many got one.

    Played entries carry a private ``_gid`` while the schedule is walked; it is
    always removed here, whether or not there is anything to attach.

    ``playedDifficulty`` - the page's "Schedule faced" - is reset to the same
    average opponent the section shows, taken from the same solved ratings, so the
    two figures cannot disagree. The page's old stand-in value for outsiders
    (a floor under the listed teams) is a display convention; the solver rates
    each of them individually and the identity needs that value.
    """
    index = share_index(shares) if shares else {}
    attached = 0
    for team, detail in details.items():
        gids = [g.pop("_gid", None) for g in detail["played"]]
        share = shares.get(team) if shares else None
        if share is None or team not in power:
            continue
        rows = [index.get((team, gid)) for gid in gids]
        block = team_block(share, power[team], len(detail["played"]))
        if block is None or any(r is None for r in rows):
            continue
        for entry, row in zip(detail["played"], rows):
            entry.update(game_fields(row))
        detail["decomposition"] = block
        detail["playedDifficulty"] = block["avgOpponent"] if block["n"] else None
        attached += 1
    return attached
