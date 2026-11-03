"""Offline tests of model semantics, temporal separation and raster contracts."""

import sys

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from local_landuse.comparison import evaluate, read_raster, run_comparison
from local_landuse.models import (NODATA, Settings, allocate_transitions,
                                 feature_stack, predict_multiclass, train_multiclass,
                                 sample_training_indices, train_shared_transitions)


def test_multiclass_learns_persistence_and_multiple_destinations():
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 1, (30, 30)).astype(np.float32)
    start = np.ones(x.shape, dtype=np.int32)
    end = np.where(x < .33, 1, np.where(x < .66, 2, 3)).astype(np.int32)
    features, _ = feature_stack(start, {"x": x}, [1, 2, 3])
    settings = Settings(n_estimators=20, min_samples_leaf=1, n_jobs=1)
    model, info = train_multiclass(start, end, features, settings=settings, max_samples=700)
    # Independent locations, same underlying decision rule.
    test_x = rng.uniform(0, 1, (15, 15)).astype(np.float32)
    test_start = np.ones(test_x.shape, dtype=np.int32)
    truth = np.where(test_x < .33, 1, np.where(test_x < .66, 2, 3))
    test_features, _ = feature_stack(test_start, {"x": test_x}, [1, 2, 3])
    prediction, confidence = predict_multiclass(model, test_start, test_features, batch_size=13)
    assert np.mean(prediction == truth) > .95
    assert set(np.unique(prediction)) == {1, 2, 3}
    assert np.all((confidence >= 0) & (confidence <= 1))
    assert np.allclose(model.predict_proba(test_features.reshape(-1, 7)).sum(axis=1), 1)
    again, info2 = train_multiclass(start, end, features, settings=settings, max_samples=700)
    assert info == info2
    assert np.array_equal(model.predict(test_features.reshape(-1, 7)),
                          again.predict(test_features.reshape(-1, 7)))


class ScoreModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        return np.column_stack([1 - X[:, 0], X[:, 0]])


def test_allocation_uses_horizon_excludes_invalid_and_prevents_double_change():
    start = np.array([[1, 1, 1, 1, 2]], dtype=np.int32)
    features = np.array([[[.9], [.8], [np.nan], [.2], [.7]]], dtype=np.float32)
    # Largest demand first; an allocated 1->2 pixel cannot then become 3.
    models = {(1, 2): ScoreModel(), (2, 3): ScoreModel()}
    result, info = allocate_transitions(models, {(1, 2): 4, (2, 3): 2}, start, features,
                                        calibration_years=4, forecast_years=2)
    assert result.tolist() == [[2, 2, NODATA, 1, 3]]
    assert [row["requested_pixels"] for row in info] == [2, 1]


def test_neighborhood_excludes_nodata_and_is_based_on_start_map():
    start = np.array([[1, NODATA], [1, 2]], dtype=np.int32)
    features, names = feature_stack(start, {}, [1, 2])
    assert features[0, 0, names.index("fraction_class_1_3x3")] == pytest.approx(2 / 3)
    assert features[0, 0, names.index("fraction_class_2_3x3")] == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="odd"):
        feature_stack(start, {}, [1, 2], neighborhood=2)


def test_metrics_distinguish_missed_change_wrong_destination_and_persistence():
    start = np.array([1, 1, 1, 1])
    observed = np.array([2, 2, 1, 1])
    prediction = np.array([2, 3, 2, 1])
    score = evaluate(start, observed, prediction, np.ones(4, dtype=bool))["summary"]
    assert score["figure_of_merit_pct"] == pytest.approx(100 / 3)
    assert score["overall_accuracy"] == .5
    assert score["wrong_destination_pixels"] == 1
    persistence = evaluate(start, observed, start, np.ones(4, dtype=bool))["summary"]
    assert persistence["figure_of_merit_pct"] == 0
    assert persistence["missed_change_pixels"] == 2


def write_fixture(path, array, transform=None):
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None]
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[1], width=array.shape[2],
                       count=len(array), dtype=array.dtype, crs="EPSG:4326",
                       transform=transform or from_origin(-13, 10, .01, .01)) as dst:
        dst.write(array)


def test_temporal_runner_never_uses_test_labels_to_train_or_predict(tmp_path):
    rng = np.random.default_rng(8)
    x = rng.uniform(0, 1, (20, 20)).astype(np.float32)
    start = np.where(x < .5, 1, 2).astype(np.uint8)
    middle = np.where(x < .6, 1, 2).astype(np.uint8)
    future = np.where(x < .7, 1, 2).astype(np.uint8)
    for year, values in zip((2010, 2015, 2020), (start, middle, future)):
        write_fixture(tmp_path / f"Guinea_LandCover_{year}.tif", values)
    write_fixture(tmp_path / "Guinea_SRTM_DEM_Slope_Aspect.tif", x)
    kwargs = dict(data_dir=tmp_path, settings=Settings(n_estimators=5, n_jobs=1),
                  max_samples=300, min_change_pixels=10, make_plot=False)
    first = run_comparison(output_dir=tmp_path / "a", **kwargs)
    # Alter ONLY future labels, including a previously unseen destination class.
    write_fixture(tmp_path / "Guinea_LandCover_2020.tif", np.full((20, 20), 99, dtype=np.uint8))
    second = run_comparison(output_dir=tmp_path / "b", **kwargs)
    for name in ("multiclass_rf", "transition_rf_allocation"):
        with rasterio.open(tmp_path / "a" / f"{name}.tif") as a, \
                rasterio.open(tmp_path / "b" / f"{name}.tif") as b:
            assert np.array_equal(a.read(), b.read())
    assert first["training"] == second["training"]
    assert first["sampling"] == "shared"
    assert first["shared_sample_sha256"] == first["training"]["sample_sha256"]
    assert np.array_equal(np.load(tmp_path / "a" / "shared_training_indices.npy"),
                          np.load(tmp_path / "b" / "shared_training_indices.npy"))
    assert second["evaluation_classes_absent_from_training"] == [99]
    assert first["results"]["multiclass_rf"]["summary"] != second["results"]["multiclass_rf"]["summary"]
    with pytest.raises(ValueError, match="not empty"):
        run_comparison(output_dir=tmp_path / "a", **kwargs)
    with pytest.raises(ValueError, match="equal"):
        run_comparison(output_dir=tmp_path / "c", years=(2010, 2015, 2024), **kwargs)
    assert "ee" not in sys.modules


def test_every_forest_receives_identical_rows_without_balancing(monkeypatch):
    # An imbalanced sample with persistence, competing destinations and sources.
    start = np.array([1] * 80 + [2] * 40, dtype=np.int32).reshape(10, 12)
    end = start.copy()
    end.ravel()[:15] = 2
    end.ravel()[15:25] = 3
    end.ravel()[80:90] = 1
    features, _ = feature_stack(start, {"pixel_id": np.arange(120).reshape(10, 12)}, [1, 2, 3])
    chosen = sample_training_indices(start, end, features, max_samples=120, seed=42)
    fits = []

    class RecordingForest:
        def fit(self, X, y):
            # No **kwargs: test also fails if unexpected sample weighting is used.
            fits.append((X.copy(), y.copy()))
            return self

    monkeypatch.setattr(Settings, "forest", lambda self: RecordingForest())
    _, multi_info = train_multiclass(start, end, features, training_indices=chosen)
    models, demands, rows = train_shared_transitions(start, end, features, chosen, min_change_pixels=1)
    assert len(models) == 3
    for X, _ in fits[1:]:
        np.testing.assert_array_equal(X, fits[0][0])
    for row in rows:
        assert row["training_pixels"] == 120
        assert row["sample_sha256"] == multi_info["sample_sha256"]
    # All other destinations AND other source classes are negative examples.
    for (source, destination), (_, y) in zip(models, fits[1:]):
        expected = ((start.ravel()[chosen] == source) & (end.ravel()[chosen] == destination))
        np.testing.assert_array_equal(y, expected)
    assert demands == {(1, 2): 15, (1, 3): 10, (2, 1): 10}


def test_shared_demand_does_not_read_unsampled_transition_labels():
    start = np.ones((10, 10), dtype=np.int32)
    end = start.copy()
    end.ravel()[:10] = 2
    features, _ = feature_stack(start, {"x": np.arange(100).reshape(10, 10)}, [1, 2])
    chosen = np.arange(50)
    kwargs = {"settings": Settings(n_estimators=3, n_jobs=1), "min_change_pixels": 1}
    first_models, first_demands, first_rows = train_shared_transitions(start, end, features, chosen, **kwargs)
    end.ravel()[50:] = 99
    second_models, second_demands, second_rows = train_shared_transitions(start, end, features, chosen, **kwargs)
    multiclass, _ = train_multiclass(start, end, features, training_indices=chosen,
                                     settings=kwargs["settings"])
    for multi_rows, binary_rows in zip(multiclass.estimators_samples_,
                                       second_models[(1, 2)].estimators_samples_):
        np.testing.assert_array_equal(multi_rows, binary_rows)
    assert first_demands == second_demands == {(1, 2): 20}
    assert first_rows == second_rows
    np.testing.assert_array_equal(first_models[(1, 2)].predict(features.reshape(100, -1)),
                                  second_models[(1, 2)].predict(features.reshape(100, -1)))
    with pytest.raises(ValueError, match="unique"):
        train_multiclass(start, end, features, training_indices=[0, 0])


def test_raster_loading_masks_zero_and_rejects_shifted_grid(tmp_path):
    write_fixture(tmp_path / "a.tif", np.array([[0, 1], [2, 255]], dtype=np.uint8))
    values, profile = read_raster(tmp_path / "a.tif", categorical=True)
    assert values.tolist() == [[NODATA, 1], [2, NODATA]]
    write_fixture(tmp_path / "shifted.tif", np.ones((2, 2), dtype=np.uint8),
                  transform=from_origin(-12, 10, .01, .01))
    with pytest.raises(ValueError, match="Grid mismatch"):
        read_raster(tmp_path / "shifted.tif", profile, categorical=True)
