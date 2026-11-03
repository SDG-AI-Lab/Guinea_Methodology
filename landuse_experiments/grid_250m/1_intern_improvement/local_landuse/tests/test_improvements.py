import copy

import numpy as np
import pytest
import rasterio

from local_landuse.improvements import (CANDIDATES, allocate_probabilities, audit_labels,
                                        build_features, choose_candidate, run_improvements)
from local_landuse.models import DestinationScorer, Settings, allocate_transitions, feature_stack


def test_baseline_features_match_controlled_and_history_uses_only_past():
    current = np.array([[1, 1, 2], [2, 2, 1], [1, 2, 2]])
    previous = np.ones_like(current)
    covariates = {"height": np.arange(9).reshape(3, 3)}
    indices = np.array([8, 1, 4, 2])
    full, old_names = feature_stack(current, covariates, [1, 2])
    X, names = build_features(current, covariates, [1, 2], indices)
    assert names == old_names
    np.testing.assert_array_equal(X, full.reshape(9, -1)[indices])
    history, history_names = build_features(current, covariates, [1, 2], indices,
                                            previous=previous, history=True, multiscale=True)
    np.testing.assert_array_equal(history[:, history_names.index("changed_since_previous")],
                                  (current != previous).ravel()[indices])
    assert np.isfinite(history).all()


def test_cached_allocator_matches_existing_and_abstention_is_separate():
    start = np.array([[1, 1, 1, 2, 2, 2]])
    X = np.array([[[.2], [.8], [.5], [.3], [.9], [.1]]])
    class Model:
        classes_ = np.array([1, 2])

        def predict_proba(self, X):
            return np.column_stack([1 - X[:, 0], X[:, 0]])

    model = Model()
    counts = {(1, 2): 2, (2, 1): 1}
    probs = model.predict_proba(X.reshape(6, 1))
    new, _ = allocate_probabilities(start, np.arange(6), probs, [1, 2], counts, 6)
    old, _ = allocate_transitions({key: DestinationScorer(model, key[1]) for key in counts},
                                  counts, start, X, calibration_years=5, forecast_years=5)
    np.testing.assert_array_equal(new, old)
    gated, ledger = allocate_probabilities(start, np.arange(6), probs, [1, 2], counts, 6,
                                           minimum_probability=.99)
    np.testing.assert_array_equal(gated, start)
    assert all(row["allocated_pixels"] == 0 for row in ledger)


def test_candidate_selection_rejects_final_test_scores_and_audit_does_not_edit():
    with pytest.raises(ValueError, match="validation"):
        choose_candidate([{"phase": "final", "name": "x", "figure_of_merit_pct": 100}])
    a = np.array([[1, 1, 2]])
    b = np.array([[2, 1, 2]])
    c = a.copy()
    result = audit_labels({2005: a, 2010: b, 2015: c})
    assert result["triplets"][0]["returned_to_initial_class_pixels"] == 1
    assert not result["label_edits"]
    np.testing.assert_array_equal(b, [[2, 1, 2]])


def write_tif(path, a):
    with rasterio.open(path, "w", driver="GTiff", width=a.shape[1], height=a.shape[0],
                       count=1, dtype=a.dtype, crs="EPSG:4326",
                       transform=rasterio.Affine(.01, 0, -13, 0, -.01, 10)) as dst:
        dst.write(a, 1)


def test_final_labels_cannot_change_selection_or_predictions(tmp_path):
    rng = np.random.default_rng(17)
    x = rng.random((30, 30)).astype(np.float32)
    for year, threshold in [(2001, .3), (2005, .4), (2010, .55), (2015, .7), (2020, .8)]:
        write_tif(tmp_path / f"Guinea_LandCover_{year}.tif", np.where(x < threshold, 1, 2).astype(np.uint8))
    write_tif(tmp_path / "Guinea_SRTM_DEM_Slope_Aspect.tif", x)
    specs = copy.deepcopy(CANDIDATES[:2])
    for s in specs:
        s["params"]["n_estimators"] = 3
    options = dict(data_dir=tmp_path, candidates=specs, max_samples=800, block_size=5,
                    jobs=1, min_change_pixels=1, make_plot=False)
    a = run_improvements(output_dir=tmp_path / "a", **options)
    write_tif(tmp_path / "Guinea_LandCover_2020.tif", np.full((30, 30), 99, dtype=np.uint8))
    b = run_improvements(output_dir=tmp_path / "b", **options)
    assert a["selection"] == b["selection"]
    for name in [s["name"] for s in specs] + ["selected_with_allocation_change"]:
        with rasterio.open(tmp_path / "a" / f"{name}.tif") as first, \
                rasterio.open(tmp_path / "b" / f"{name}.tif") as second:
            np.testing.assert_array_equal(first.read(), second.read())
    assert a["results"]["rf_baseline"] != b["results"]["rf_baseline"]
