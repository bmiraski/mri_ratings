"""Shared payload pieces that both sports compute the same way."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def content_digest(payload: dict) -> str:
    """Fingerprint everything about a payload except when it was built."""
    material = {k: v for k, v in payload.items() if k not in {"generated", "digest"}}
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def settle_timestamp(payload: dict, path: Path) -> dict:
    """Carry the previous timestamp forward when nothing else has changed.

    The published pages already do this, which is what stopped a scheduled run
    committing 157 files for a clock tick. The intermediate JSON did not, so
    once it joined the list of committed paths every daily run produced a
    two-file diff whose entire content was a new timestamp - the same churn,
    reintroduced one directory over.

    Applying it here rather than at the writer also keeps the two in agreement:
    the stamp now means "when these ratings last changed" in both places.
    """
    payload["digest"] = content_digest(payload)
    if not path.exists():
        return payload
    try:
        previous = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return payload
    if previous.get("digest") == payload["digest"] and previous.get("generated"):
        payload["generated"] = previous["generated"]
    return payload

# Each next-best team in a conference counts 10% less than the one above it.
#
# The plain mean answers "how deep is this league", which is a real question but
# not the one the panel is read as asking. It also penalises an eighteen-team
# conference for its tail while rewarding a sixteen-team one for having no bad
# teams, so in 2025-26 it put the SEC clear of the Big Ten on the strength of a
# high floor rather than a strong top.
#
# Going further the other way is worse: at 0.75 the measure is effectively "mean
# of the top half" and a league's back end stops mattering at all, which makes a
# top-heavy six-team conference look like a national power.
#
# 0.90 sits where the three strongest leagues of 2025-26 come out within 0.4 of
# each other, which is the honest answer for that season - they were not
# separable - while the leagues below them still sort cleanly.
DECAY = 0.90

# A group this small is not a league and must not be ranked as one. Football's
# two independents came out third on this measure, which is arithmetically true
# and says nothing about a conference, because they are not in one. They keep
# their page and their place in the filter; they just do not appear in a ranking
# of conference strength.
MIN_RANKED = 4


def conference_strength(teams: list[dict], decay: float = DECAY) -> list[dict]:
    """Rank conferences by a top-weighted average of member ratings.

    Weights decay by rank within the conference, so the measure is about quality
    at the top without ignoring what is underneath it. Independent of conference
    size, which a plain mean is not.
    """
    grouped: dict[str, list[float]] = {}
    for team in teams:
        grouped.setdefault(team["conference"], []).append(float(team["power"]))

    rows = []
    for conference, powers in grouped.items():
        ordered = sorted(powers, reverse=True)
        weights = [decay ** i for i in range(len(ordered))]
        strength = sum(p * w for p, w in zip(ordered, weights)) / sum(weights)
        rows.append(
            {
                "conference": conference,
                "strength": round(strength, 2),
                "mean": round(sum(ordered) / len(ordered), 2),
                "count": len(ordered),
                "max": round(ordered[0], 2),
                "ranked": len(ordered) >= MIN_RANKED,
            }
        )

    # Unranked groups sort last whatever their number, so nothing that is not a
    # conference can lead a table of conferences.
    return (
        pd.DataFrame(rows)
        .sort_values(["ranked", "strength"], ascending=[False, False])
        .reset_index(drop=True)
        .to_dict("records")
    )
