"""Coach identity and split-season attribution: the two places CFBD's /coaches
data needs help before it can be trusted as one row per coach per school per
season. Games are injected via ``games_fn`` rather than fetched, so none of
this touches the network."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from mri.coaches import ids, season


# ---------------------------------------------------------------------------
# coach_id and aliases
# ---------------------------------------------------------------------------

def test_coach_id_is_a_slug_of_name_and_first_hire_year() -> None:
    assert ids.coach_id("Nick Saban", 1990) == "nick-saban-1990"
    assert ids.coach_id("Bill O'Brien", 2012) == "bill-o-brien-2012"


def test_first_hire_year_prefers_hire_date_over_earliest_season() -> None:
    # A coach hired in December 1990 for the 1991 season - hire_date's year should win.
    assert ids.first_hire_year("1990-12-04T00:00:00.000Z", pd.Series([1991, 1992, 1993])) == 1990
    # No hire_date at all - fall back to the earliest season actually seen.
    assert ids.first_hire_year(None, pd.Series([1991, 1992])) == 1991
    assert ids.first_hire_year("not-a-date", pd.Series([1991, 1992])) == 1991


def test_an_alias_resolves_a_name_variant_to_the_same_id(tmp_path, monkeypatch) -> None:
    aliases_path = tmp_path / "coach_aliases.json"
    aliases_path.write_text(json.dumps({"aliases": {"Nick Saban Jr.": "Nick Saban"}}))
    monkeypatch.setattr(ids, "ALIASES_PATH", aliases_path)
    ids._aliases.cache_clear()

    assert ids.coach_id("Nick Saban Jr.", 1990) == ids.coach_id("Nick Saban", 1990)
    ids._aliases.cache_clear()  # leave the cache clean for later tests


def _coach_rows(*rows: tuple[str, str, str, str, int]) -> pd.DataFrame:
    """coach_key, coach_name, hire_date, school, season -> a minimal coach_rows frame."""
    return pd.DataFrame(
        [
            {
                "coach_key": key, "coach_name": name, "hire_date": hire, "school": school, "season": season_,
                "conference": "Conf", "games": 12, "wins": 6, "losses": 6, "ties": 0, "srs": 0.0, "sp_overall": 0.0,
            }
            for key, name, hire, school, season_ in rows
        ]
    )


def test_assign_ids_gives_one_id_per_coach_key() -> None:
    raw = _coach_rows(
        (1, "Vince Veteran", "2010-12-01", "School X", 2015),
        (1, "Vince Veteran", "2010-12-01", "School X", 2016),
        (2, "Ned Newcomer", "2016-11-15", "School X", 2016),
    )
    out = ids.assign_ids(raw)
    assert out.loc[out["coach_key"] == 1, "coach_id"].nunique() == 1
    assert out.loc[out["coach_key"] == 1, "coach_id"].iloc[0] == "vince-veteran-2010"
    assert out.loc[out["coach_key"] == 2, "coach_id"].iloc[0] == "ned-newcomer-2016"


def test_detect_collisions_flags_two_keys_sharing_an_id() -> None:
    raw = _coach_rows(
        (1, "John Smith", "2005-01-01", "School X", 2005),
        (2, "John Smith", "2005-06-01", "School Y", 2005),  # a different real person, same name+year
    )
    out = ids.assign_ids(raw)
    collisions = ids.detect_collisions(out)
    assert len(collisions) == 1
    assert set(collisions.iloc[0]["coach_keys"]) == {1, 2}


def test_no_collisions_among_genuinely_different_ids() -> None:
    raw = _coach_rows(
        (1, "Vince Veteran", "2010-12-01", "School X", 2015),
        (2, "Ned Newcomer", "2016-11-15", "School X", 2016),
    )
    out = ids.assign_ids(raw)
    assert ids.detect_collisions(out).empty


# ---------------------------------------------------------------------------
# split-season game attribution
# ---------------------------------------------------------------------------

def _schedule(school: str, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": range(1, n + 1),
            "team1": ["Opponent"] * n,
            "team2": [school] * n,
            "start_date": pd.date_range("2016-09-01", periods=n, freq="7D"),
        }
    )


def test_split_season_attributes_every_game_to_exactly_one_coach() -> None:
    """A fired incumbent's games come first, an interim's come last - matching a real mid-season change."""
    raw = _coach_rows(
        (1, "Vince Veteran", "2010-12-01", "School X", 2015),
        (1, "Vince Veteran", "2010-12-01", "School X", 2016),
        (2, "Ned Newcomer", "2016-11-15", "School X", 2016),
    )
    with_ids = ids.assign_ids(raw)
    with_ids.loc[(with_ids["coach_key"] == 1) & (with_ids["season"] == 2016), "games"] = 11
    with_ids.loc[with_ids["coach_key"] == 2, "games"] = 1

    coaches_here = with_ids[with_ids["season"] == 2016]
    assignments = season.attribute_games(
        "School X", 2016, coaches_here, with_ids, games_fn=lambda yr: _schedule("School X", 12)
    )

    assert len(assignments) == 12
    vince_id, ned_id = "vince-veteran-2010", "ned-newcomer-2016"
    assert (assignments["coach_id"] == vince_id).sum() == 11
    assert (assignments["coach_id"] == ned_id).sum() == 1
    # date order: the incumbent's games are the front of the schedule, the interim's the back
    ordered = assignments.merge(_schedule("School X", 12), on="game_id").sort_values("start_date")
    assert list(ordered["coach_id"])[:11] == [vince_id] * 11
    assert list(ordered["coach_id"])[11:] == [ned_id]


def test_split_season_raises_when_counts_do_not_match_the_schedule() -> None:
    raw = _coach_rows(
        (1, "Vince Veteran", "2010-12-01", "School X", 2016),
        (2, "Ned Newcomer", "2016-11-15", "School X", 2016),
    )
    with_ids = ids.assign_ids(raw)
    with_ids["games"] = 6  # 6 + 6 = 12, but the "schedule" below only has 10 games

    with pytest.raises(ValueError, match="coach games sum to"):
        season.attribute_games("School X", 2016, with_ids, with_ids, games_fn=lambda yr: _schedule("School X", 10))


# ---------------------------------------------------------------------------
# interim flag, and build_coach_season end to end
# ---------------------------------------------------------------------------

def test_interim_flags_the_newcomer_in_a_split_season_and_self_corrects() -> None:
    raw = _coach_rows(
        (1, "Vince Veteran", "2010-12-01", "School X", 2015),
        (1, "Vince Veteran", "2010-12-01", "School X", 2016),
        (2, "Ned Newcomer", "2016-11-15", "School X", 2016),
        (2, "Ned Newcomer", "2016-11-15", "School X", 2017),  # now the full-time coach
    )
    with_ids = ids.assign_ids(raw)
    with_ids.loc[(with_ids["coach_key"] == 1) & (with_ids["season"] == 2016), "games"] = 11
    with_ids.loc[(with_ids["coach_key"] == 2) & (with_ids["season"] == 2016), "games"] = 1

    table, problems = season.build_coach_season(
        with_ids, games_fn=lambda yr: _schedule("School X", 12)
    )

    assert problems == []
    vince_id, ned_id = "vince-veteran-2010", "ned-newcomer-2016"
    row = lambda cid, yr: bool(table[(table["coach_id"] == cid) & (table["season"] == yr)].iloc[0]["interim"])  # noqa: E731
    assert row(vince_id, 2015) is False
    assert row(vince_id, 2016) is False   # the incumbent who got fired is not "interim"
    assert row(ned_id, 2016) is True      # new to the school, mid-season - the proxy's best guess
    assert row(ned_id, 2017) is False     # a full season of his own: no longer flagged
