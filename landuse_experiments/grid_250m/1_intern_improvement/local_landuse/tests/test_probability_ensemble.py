import numpy as np
import pytest

from local_landuse.advanced import apply_decision, compact_metrics, tune_decision
from local_landuse.probability_ensemble import build_provider, greedy_selection

CLASSES = np.array([1, 2, 3])


def surfaces(seed=0, members=4, rows=4000):
    """Members that each see a noisy view of the same truth, plus one useless member."""
    rng = np.random.default_rng(seed)
    source = rng.choice(CLASSES, rows)
    observed = source.copy()
    flip = rng.random(rows) < .2
    observed[flip] = rng.choice(CLASSES, flip.sum())
    truth = np.zeros((rows, len(CLASSES)), np.float32)
    truth[np.arange(rows), np.searchsorted(CLASSES, observed)] = 1.
    out = {}
    for i in range(members):
        noise = rng.random((rows, len(CLASSES))).astype(np.float32)
        values = .45 * truth + .55 * noise
        out[f"member_{i}"] = values / values.sum(axis=1, keepdims=True)
    noise = rng.random((rows, len(CLASSES))).astype(np.float32)
    out["useless"] = noise / noise.sum(axis=1, keepdims=True)
    return out, source, observed


def single_fom(values, source, observed):
    policy, row, _ = tune_decision(values, CLASSES, source, observed)
    return row["figure_of_merit_pct"]


def test_weights_are_normalised_selection_counts():
    values, source, observed = surfaces()
    weights, _, history, _ = greedy_selection(values, CLASSES, source, observed, rounds=6)
    assert weights and sum(weights.values()) == pytest.approx(1.)
    assert set(weights) <= set(values)
    assert len(history) <= 6


def test_validation_score_improves_at_every_accepted_step():
    values, source, observed = surfaces(seed=3)
    _, _, history, _ = greedy_selection(values, CLASSES, source, observed, rounds=8)
    scores = [step["figure_of_merit_pct"] for step in history]
    assert scores == sorted(scores) and len(set(scores)) == len(scores)


def test_ensemble_is_never_worse_than_its_best_single_member():
    values, source, observed = surfaces(seed=7)
    _, policy, _, averaged = greedy_selection(values, CLASSES, source, observed, rounds=8)
    decided = apply_decision(averaged, CLASSES, source, policy)
    ensemble = compact_metrics(source, observed, decided, decided != source)["figure_of_merit_pct"]
    best = max(single_fom(v, source, observed) for v in values.values())
    assert ensemble >= best - 1e-9


def test_a_pure_noise_member_does_not_dominate_the_ensemble():
    values, source, observed = surfaces(seed=11)
    weights, _, _, _ = greedy_selection(values, CLASSES, source, observed, rounds=8)
    assert weights.get("useless", 0.) < .5


def test_greedy_selection_needs_at_least_one_member():
    with pytest.raises(ValueError):
        greedy_selection({}, CLASSES, np.array([1]), np.array([1]))


def test_unknown_provider_kind_is_rejected():
    with pytest.raises(ValueError, match="Unknown provider kind"):
        build_provider("not_a_provider", "phenology", "spectral", "reference.tif")
