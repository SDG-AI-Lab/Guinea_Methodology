import numpy as np
import pytest
import rasterio

from local_landuse.controlled import block_mask, run_controlled_comparison
from local_landuse.models import DestinationScorer, Settings, spatial_split


def test_original_block_split_is_shared_and_disjoint():
    indices = np.arange(400 * 400)
    train, test, train_blocks, test_blocks = spatial_split(indices, (400, 400))
    assert len(test_blocks) == 5  # ceil(0.3 * 16 blocks), not 30% of individual pixels
    assert not np.intersect1d(train_blocks, test_blocks).size
    assert not np.intersect1d(train, test).size
    assert np.array_equal(np.sort(np.concatenate([train, test])), indices)
    heldout = block_mask((400, 400), test_blocks)
    assert heldout.ravel()[test].all()
    assert not heldout.ravel()[train].any()
    with pytest.raises(ValueError, match="two sampled"):
        spatial_split(np.arange(10), (10, 10))


def test_multiclass_score_adapter_uses_class_codes_not_column_positions():
    class Multi:
        classes_ = np.array([2, 7, 17])

        def predict_proba(self, X):
            return np.tile([.1, .7, .2], (len(X), 1))

    scorer = DestinationScorer(Multi(), 7)
    np.testing.assert_allclose(scorer.predict_proba(np.zeros((3, 2))), [[.3, .7]] * 3)
    np.testing.assert_array_equal(scorer.classes_, [0, 1])
    with pytest.raises(ValueError):
        DestinationScorer(Multi(), 99).predict_proba(np.zeros((1, 2)))


def write_tif(path, a):
    with rasterio.open(path, "w", driver="GTiff", width=a.shape[1], height=a.shape[0],
                       count=1, dtype=a.dtype, crs="EPSG:4326",
                       transform=rasterio.Affine(.01, 0, -13, 0, -.01, 10)) as dst:
        dst.write(a, 1)


def test_controlled_run_freezes_samples_blocks_demands_and_heldout_labels(tmp_path):
    rng = np.random.default_rng(7)
    x = rng.random((40, 40)).astype(np.float32)
    start = np.where(x < .5, 1, 2).astype(np.uint8)
    middle = np.where(x < .65, 1, np.where(x > .85, 3, 2)).astype(np.uint8)
    future = np.where(x < .7, 1, np.where(x > .8, 3, 2)).astype(np.uint8)
    for year, array in zip((2010, 2015, 2020), (start, middle, future)):
        write_tif(tmp_path / f"Guinea_LandCover_{year}.tif", array)
    write_tif(tmp_path / "Guinea_SRTM_DEM_Slope_Aspect.tif", x)
    kwargs = dict(data_dir=tmp_path, settings=Settings(n_estimators=3, n_jobs=1), max_samples=1000,
                  min_change_pixels=1, block_size=8, make_plot=False)
    a = run_controlled_comparison(output_dir=tmp_path / "a", **kwargs)
    assert all(a["audit"].values())
    assert a["split"]["train_pixels"] + a["split"]["test_pixels"] == 1000
    for scope in ("spatial_holdout", "temporal_holdout"):
        assert a[scope]["identical_allocation_counts"]
        results = a[scope]["results"]
        assert results["binary_rf_allocation"]["summary"]["predicted_change_pixels"] == \
            results["multiclass_rf_allocation"]["summary"]["predicted_change_pixels"]
        assert (tmp_path / "a" / scope / "binary_rf_allocation_allocation.csv").read_bytes() == \
            (tmp_path / "a" / scope / "multiclass_rf_allocation_allocation.csv").read_bytes()
    # Alter future evaluation labels: neither spatial nor temporal predictions change.
    write_tif(tmp_path / "Guinea_LandCover_2020.tif", np.full_like(future, 99))
    b = run_controlled_comparison(output_dir=tmp_path / "b", **kwargs)
    for scope in ("spatial_holdout", "temporal_holdout"):
        for model in ("binary_rf_allocation", "multiclass_rf_allocation"):
            with rasterio.open(tmp_path / "a" / scope / f"{model}.tif") as first, \
                    rasterio.open(tmp_path / "b" / scope / f"{model}.tif") as second:
                np.testing.assert_array_equal(first.read(), second.read())
    # Alter calibration destination labels ONLY in held-out blocks. Training,
    # spatial demand and spatial prediction must remain unchanged. Temporal
    # inputs legitimately change, so temporal predictions are not compared here.
    heldout = block_mask(start.shape, np.load(tmp_path / "a" / "test_blocks.npy"), 8)
    middle[heldout] = 99
    write_tif(tmp_path / "Guinea_LandCover_2015.tif", middle)
    c = run_controlled_comparison(output_dir=tmp_path / "c", **kwargs)
    assert a["train_sample_sha256"] == b["train_sample_sha256"] == c["train_sample_sha256"]
    assert (tmp_path / "a" / "spatial_holdout" / "shared_demands.csv").read_bytes() == \
        (tmp_path / "c" / "spatial_holdout" / "shared_demands.csv").read_bytes()
    for model in ("binary_rf_allocation", "multiclass_rf_allocation"):
        with rasterio.open(tmp_path / "a" / "spatial_holdout" / f"{model}.tif") as first, \
                rasterio.open(tmp_path / "c" / "spatial_holdout" / f"{model}.tif") as second:
            np.testing.assert_array_equal(first.read(), second.read())
