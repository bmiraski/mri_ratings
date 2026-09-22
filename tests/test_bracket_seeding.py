import numpy as np
import pytest

from mri.bracket import seeding


def _field(n_auto_top: int, fmt):
    """A field in true-seed order: ``n_auto_top`` strong automatic qualifiers near the top, then at-large
    teams, then the rest of the automatic qualifiers at the bottom - the usual shape."""
    autos = 32 if fmt.size == 76 else 31
    at_large = fmt.size - autos
    return np.array([True] * n_auto_top + [False] * at_large + [True] * (autos - n_auto_top))


def test_field_size_switches_in_2027():
    assert seeding.field_size(2026) == 68
    assert seeding.field_size(2027) == 76


@pytest.mark.parametrize("size", [68, 76])
def test_every_team_gets_a_line_and_lines_have_the_right_counts(size):
    fmt = seeding.FORMATS[size]
    line, opening = seeding.lines(_field(6, fmt), fmt)
    counts = {k: int((line == k).sum()) for k in range(1, 17)}
    assert sum(counts.values()) == size
    assert opening.sum() == fmt.opening_at_large_count + fmt.opening_auto_count
    if size == 76:
        assert counts == {**{k: 4 for k in range(1, 11)}, 11: 6, 12: 8, 13: 4, 14: 4, 15: 6, 16: 8}
    else:
        assert counts == {**{k: 4 for k in range(1, 17)}, 11: 6, 16: 6}


def test_76_team_opening_round_is_the_twelve_lowest_of_each_bid_type():
    fmt = seeding.FORMATS[76]
    is_auto = _field(6, fmt)
    line, opening = seeding.lines(is_auto, fmt)
    at_large = np.flatnonzero(~is_auto)
    autos = np.flatnonzero(is_auto)
    assert set(np.flatnonzero(opening & ~is_auto)) == set(at_large[-12:])
    assert set(np.flatnonzero(opening & is_auto)) == set(autos[-12:])
    # the better four of each group take the better line
    assert list(line[at_large[-12:]]) == [11] * 4 + [12] * 8
    assert list(line[autos[-12:]]) == [15] * 4 + [16] * 8


def test_76_team_line_12_is_all_opening_round_at_large():
    """A consequence of the new format worth pinning down: no champion can be a direct No. 12 any more."""
    fmt = seeding.FORMATS[76]
    is_auto = _field(10, fmt)
    line, opening = seeding.lines(is_auto, fmt)
    assert all(opening[line == 12]) and not any(is_auto[line == 12])


def test_wrong_size_is_refused():
    with pytest.raises(ValueError):
        seeding.lines(np.array([True] * 10), seeding.FORMATS[68])


def test_opening_round_games_pair_neighbours():
    assert seeding.opening_round_games(["a", "b", "c", "d"]) == [("a", "b"), ("c", "d")]
