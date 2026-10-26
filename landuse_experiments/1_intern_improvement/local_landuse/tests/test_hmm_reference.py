import numpy as np
import pytest

from local_landuse.hmm_reference import (emission_matrix, fit_from_stack, fit_parameters, smooth_labels,
                                         swap_shape, viterbi)

PRIOR = np.array([.5, .5])
STICKY = np.array([[.95, .05], [.05, .95]])
SHAPE = np.array([[0., 1.], [1., 0.]])


def simulate(accuracy, rows=20000, steps=4, k=3, persistence=.9, seed=0):
    rng = np.random.default_rng(seed)
    true = np.empty((rows, steps), dtype=np.int64)
    true[:, 0] = rng.integers(0, k, rows)
    for t in range(1, steps):
        move = rng.random(rows) > persistence
        true[:, t] = np.where(move, (true[:, t - 1] + rng.integers(1, k, rows)) % k, true[:, t - 1])
    wrong = rng.random((rows, steps)) > accuracy
    return np.where(wrong, (true + rng.integers(1, k, (rows, steps))) % k, true)


def test_a_single_snapshot_flip_is_smoothed_away():
    path = viterbi(np.array([[0, 0, 0, 1, 0, 0]]), PRIOR, STICKY, emission_matrix(.8, SHAPE))
    assert path.tolist() == [[0, 0, 0, 0, 0, 0]]


def test_a_sustained_change_is_kept():
    observed = np.array([[0, 0, 0, 1, 1, 1]])
    assert np.array_equal(viterbi(observed, PRIOR, STICKY, emission_matrix(.8, SHAPE)), observed)


def test_perfect_labels_are_returned_unchanged():
    rng = np.random.default_rng(1)
    observed = rng.integers(0, 3, (500, 6))
    transition = np.full((3, 3), 1 / 3)
    shape = swap_shape(observed, np.ones(len(observed)), 3)
    assert np.array_equal(viterbi(observed, np.full(3, 1 / 3), transition, emission_matrix(1., shape)), observed)


def test_probability_matrices_are_normalised():
    observed = simulate(.8, rows=3000)
    unique, counts = np.unique(observed, axis=0, return_counts=True)
    fitted = fit_parameters(unique, counts, 3)
    assert fitted["transition"].sum(axis=1) == pytest.approx(np.ones(3))
    assert emission_matrix(fitted["accuracy"], fitted["shape"]).sum(axis=1) == pytest.approx(np.ones(3))
    assert fitted["prior"].sum() == pytest.approx(1.)


def test_noisier_labels_give_a_lower_estimated_accuracy():
    clean, noisy = [simulate(a, seed=4) for a in (.98, .7)]
    estimates = [fit_parameters(*np.unique(s, axis=0, return_counts=True), 3)["accuracy"] for s in (clean, noisy)]
    assert estimates[0] > estimates[1]


def test_parameters_never_read_labels_after_2015():
    classes = np.array([8, 9, 10])
    stack = classes[simulate(.8, rows=5000, steps=6, seed=2)]
    altered = stack.copy()
    altered[:, 4:] = classes[np.random.default_rng(9).integers(0, 3, (len(stack), 2))]
    first, second = fit_from_stack(stack, classes), fit_from_stack(altered, classes)
    assert first["accuracy"] == second["accuracy"]
    assert np.array_equal(first["transition"], second["transition"])


def test_unique_sequence_smoothing_matches_per_pixel_viterbi():
    classes = np.array([8, 9, 10])
    stack = classes[simulate(.8, rows=2000, steps=6, seed=5)]
    fitted = fit_from_stack(stack, classes)
    direct = viterbi(np.searchsorted(classes, stack), fitted["prior"], fitted["transition"],
                     emission_matrix(fitted["accuracy"], fitted["shape"]))
    assert np.array_equal(smooth_labels(stack, classes, fitted), classes[direct])


def test_unknown_classes_are_rejected():
    fitted = dict(prior=PRIOR, transition=STICKY, shape=SHAPE, accuracy=.8)
    with pytest.raises(ValueError, match="outside the vocabulary"):
        smooth_labels(np.array([[1, 2, 7]]), np.array([1, 2]), fitted)
