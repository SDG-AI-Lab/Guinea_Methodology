import numpy as np
import pytest

from local_landuse.class_typicality import ClassTypicality, class_descriptions, log_likelihoods


class FakePhenology:
    shape = (20, 20)
    years = (2010, 2015)


class FakeBase:
    """Two separable classes plus a handful of pixels sitting between them."""

    manifests = {"fake": True}
    phenology = FakePhenology()

    def __init__(self):
        rng = np.random.default_rng(0)
        self.values = {}
        for year in (2010, 2015):
            grid = rng.normal(0, .05, (20, 20)).astype(np.float32)
            grid[:10] += 1.
            grid[10:] -= 1.
            grid[9, :4] = 0.        # ambiguous pixels labelled as the upper class
            self.values[year] = grid

    def annual_grids(self, year):
        grid = self.values[year]
        return {key: (grid, np.ones((20, 20), bool))
                for key in ("ndvi_annual", "ndvi_wet", "ndvi_dry",
                            "evi_annual", "evi_wet", "evi_dry")}

    def __call__(self, year, indices):
        return np.zeros((len(indices), 1), np.float32), ["base"]


@pytest.fixture
def provider(tmp_path):
    import rasterio
    labels = np.where(np.arange(400).reshape(20, 20) < 200, 1, 2).astype(np.uint8)
    profile = dict(driver="GTiff", height=20, width=20, count=1, dtype="uint8",
                   crs="EPSG:4326", transform=rasterio.transform.from_origin(0, 0, 1, 1))
    for year in (2005, 2010, 2015):
        with rasterio.open(tmp_path / f"Guinea_LandCover_{year}.tif", "w", **profile) as dst:
            dst.write(labels, 1)
    return ClassTypicality(FakeBase(), data_dir=tmp_path, window=3, minimum=20)


def test_class_descriptions_skips_classes_without_enough_reference_pixels():
    features = np.zeros((10, 2), np.float32)
    labels = np.array([1] * 9 + [2])
    means, _, priors = class_descriptions(features, np.ones(10, bool), labels,
                                          np.array([1, 2]), minimum=5)
    assert set(means) == {1} and priors[1] == pytest.approx(1.)


def test_log_likelihood_is_highest_at_the_class_mean():
    means = {1: np.zeros(2), 2: np.full(2, 5.)}
    variances = {1: np.ones(2), 2: np.ones(2)}
    scores = log_likelihoods(np.array([[0., 0.], [5., 5.]], np.float32), means, variances,
                             {1: .5, 2: .5}, np.array([1, 2]))
    assert scores[0, 0] > scores[0, 1] and scores[1, 1] > scores[1, 0]


def test_ambiguous_pixels_look_less_typical_of_their_own_class(provider):
    features = provider._year_features(2015)
    flat = np.arange(400).reshape(20, 20)
    ambiguous, typical = flat[9, :4], flat[0]
    assert features["own_loglik"][ambiguous].max() < features["own_loglik"][typical].min()
    assert features["margin"][ambiguous].max() < features["margin"][typical].min()
    assert (features["best_other_class"][ambiguous] == 2).all()


def test_features_never_read_labels_or_observations_after_the_starting_year(provider, tmp_path):
    import rasterio
    indices = np.arange(400)
    before, names = provider(2015, indices)
    assert any(name.startswith("typicality_current_") for name in names)
    corrupted = np.full((20, 20), 2, np.uint8)
    with rasterio.open(tmp_path / "Guinea_LandCover_2020.tif", "w", driver="GTiff", height=20,
                       width=20, count=1, dtype="uint8", crs="EPSG:4326",
                       transform=rasterio.transform.from_origin(0, 0, 1, 1)) as dst:
        dst.write(corrupted, 1)
    fresh = ClassTypicality(FakeBase(), data_dir=tmp_path, window=3, minimum=20)
    after, _ = fresh(2015, indices)
    assert np.array_equal(before, after)


def test_change_columns_are_zero_when_no_earlier_year_is_available(provider):
    matrix, names = provider(2010, np.arange(400))
    changes = [i for i, name in enumerate(names) if name.startswith("typicality_change_")]
    assert changes and not matrix[:, changes].any()
