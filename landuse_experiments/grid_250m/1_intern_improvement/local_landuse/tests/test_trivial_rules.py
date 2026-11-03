import numpy as np
import pytest

from local_landuse.trivial_rules import (build_rules, likely_destinations, modal_class,
                                         neighbourhood_majority)

CLASSES = np.array([8, 9, 10])


def history(rows=400, seed=0):
    rng = np.random.default_rng(seed)
    maps = {2001: rng.choice(CLASSES, rows)}
    for year in (2005, 2010, 2015):
        previous = maps[max(y for y in maps)]
        move = rng.random(rows) < .25
        maps[year] = np.where(move, rng.choice(CLASSES, rows), previous)
    return maps


def test_neighbourhood_majority_uses_the_window_not_the_opposite_edge():
    grid = np.array([[9, 9, 10], [9, 10, 10], [8, 10, 10]])
    majority = neighbourhood_majority(grid, CLASSES)
    assert majority[0, 0] == 9  # 9,9,9,10 in its corner window
    assert majority[0, 2] == 10  # right column; wrapping round would pull in the 8 and extra 9s
    assert majority.shape == grid.shape


def test_excluding_the_pixel_changes_only_that_vote():
    grid = np.array([[9, 9, 9], [9, 10, 9], [9, 9, 9]])
    assert neighbourhood_majority(grid, CLASSES)[1, 1] == 9
    assert neighbourhood_majority(grid, CLASSES, include_self=False)[1, 1] == 9
    lonely = np.array([[10, 10, 9], [10, 9, 9], [9, 9, 9]])
    assert neighbourhood_majority(lonely, CLASSES, include_self=False)[0, 0] == 10


def test_neighbourhood_majority_keeps_a_uniform_grid_unchanged():
    grid = np.full((5, 5), 9)
    assert np.array_equal(neighbourhood_majority(grid, CLASSES), grid)


def test_modal_class_takes_the_most_frequent_label():
    sequence = np.array([[9, 9, 10, 9], [8, 10, 10, 10]])
    assert modal_class(sequence).tolist() == [9, 10]


def test_likely_destination_ignores_unchanged_pixels():
    start = np.array([9, 9, 9, 9])
    end = np.array([9, 9, 10, 10])
    assert likely_destinations(start, end, CLASSES)[9] == 10


def test_rules_only_touch_pixels_they_claim_to_touch():
    maps = history()
    majority = np.full(len(maps[2015]), 8)
    rules = build_rules(maps, majority, CLASSES)
    unchanged = maps[2015] == maps[2010]
    for name in ("Revert to the 2010 class if changed 2010->2015",
                 "Previous changers take the majority class of their eight 2015 neighbours"):
        assert np.array_equal(rules[name][unchanged], maps[2015][unchanged])
    assert np.array_equal(rules["No change"], maps[2015])


def test_rules_never_read_labels_after_2015():
    maps = history()
    majority = np.full(len(maps[2015]), 9)
    first = build_rules(maps, majority, CLASSES)
    polluted = dict(maps)
    polluted[2020] = np.full(len(maps[2015]), 10)
    second = build_rules(polluted, majority, CLASSES)
    assert set(first) == set(second)
    for name, predicted in first.items():
        assert np.array_equal(predicted, second[name]), name


def test_reversion_rule_restores_the_previous_label():
    maps = {2001: np.array([9]), 2005: np.array([9]), 2010: np.array([9]), 2015: np.array([10])}
    rules = build_rules(maps, np.array([8]), CLASSES)
    assert rules["Revert to the 2010 class if changed 2010->2015"].tolist() == [9]


def test_unknown_class_in_the_vocabulary_does_not_break_destinations():
    assert likely_destinations(np.array([9]), np.array([9]), CLASSES) == {}
    with pytest.raises(ValueError):
        modal_class(np.empty((0, 0)))
