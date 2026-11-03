import json

import numpy as np
import pytest
import rasterio

from local_landuse.fire_models import UNBURNED_HORIZON, FireContext, HistoricalFire

SHAPE = (8, 8)
PROFILE = dict(driver="GTiff", height=SHAPE[0], width=SHAPE[1], count=3, dtype="float32",
               nodata=np.nan, crs="EPSG:4326",
               transform=rasterio.transform.from_origin(0, 0, 1, 1))


def write_year(root, year, burned, day=None, mapped=None):
    directory = root / str(year)
    directory.mkdir(parents=True, exist_ok=True)
    stack = np.stack([burned.astype(np.float32),
                      (np.zeros(SHAPE) if day is None else day).astype(np.float32),
                      (np.ones(SHAPE) if mapped is None else mapped).astype(np.float32)])
    with rasterio.open(directory / "annual_fire.tif", "w", **PROFILE) as dst:
        dst.write(stack)
    (directory / "manifest.json").write_text(json.dumps(dict(
        year=year, features=["burned_fraction", "first_burn_day", "mapped_fraction"])))


class FakeBase:
    manifests = {"fake": True}

    def __call__(self, year, indices):
        return np.zeros((len(indices), 1), np.float32), ["base"]


@pytest.fixture
def fire(tmp_path):
    reference = tmp_path / "reference.tif"
    with rasterio.open(reference, "w", **dict(PROFILE, count=1, dtype="uint8", nodata=0)) as dst:
        dst.write(np.ones(SHAPE, np.uint8), 1)
    for year in range(2006, 2016):
        burned = np.zeros(SHAPE)
        burned[0] = 1.                      # row 0 burns every year
        if year >= 2014:
            burned[1] = 1.                  # row 1 burns only in the last two years
        write_year(tmp_path / "fire", year, burned, day=np.full(SHAPE, 40. if year % 2 else 300.))
    return HistoricalFire(tmp_path / "fire", reference)


def test_years_are_discovered_from_acquired_directories(fire):
    assert fire.years == tuple(range(2006, 2016))


def test_missing_requested_year_is_rejected(fire, tmp_path):
    with pytest.raises(ValueError, match="Missing acquired burned-area years"):
        HistoricalFire(tmp_path / "fire", tmp_path / "reference.tif", years=(2005, 2010))


def test_history_never_reads_years_after_the_starting_year(fire):
    years, burned, _, _ = fire.history(2010)
    assert years.max() == 2010 and len(years) == 5 and burned.shape[0] == 5


def test_burn_frequency_and_recency_match_the_written_history(fire):
    columns = FireContext(FakeBase(), fire).columns(2015)
    assert columns["fire_last_10_burn_count"][0].tolist() == [10] * 8
    assert columns["fire_last_10_burn_count"][1].tolist() == [2] * 8
    assert columns["fire_years_since_last_burn"][0].tolist() == [0] * 8
    assert columns["fire_years_since_last_burn"][2].tolist() == [UNBURNED_HORIZON] * 8
    assert columns["fire_ever_burned"][2].tolist() == [0] * 8


def test_a_pixel_that_stopped_burning_shows_a_negative_frequency_trend(fire, tmp_path):
    for year in range(2006, 2016):
        burned = np.zeros(SHAPE)
        if year <= 2010:
            burned[3] = 1.
        write_year(tmp_path / "cooling", year, burned)
    cooling = HistoricalFire(tmp_path / "cooling", tmp_path / "reference.tif")
    columns = FireContext(FakeBase(), cooling).columns(2015)
    assert (columns["fire_frequency_trend"][3] < 0).all()
    assert columns["fire_years_since_last_burn"][3].tolist() == [5] * 8


def test_unmapped_years_are_supported_not_treated_as_unburned(fire, tmp_path):
    for year in range(2006, 2016):
        burned = np.where(year >= 2013, np.ones(SHAPE), np.full(SHAPE, np.nan))
        write_year(tmp_path / "partial", year, burned,
                   mapped=np.ones(SHAPE) if year >= 2013 else np.zeros(SHAPE))
    partial = HistoricalFire(tmp_path / "partial", tmp_path / "reference.tif")
    columns = FireContext(FakeBase(), partial).columns(2015)
    assert (columns["fire_last_10_observed_years"] == 3).all()
    assert (columns["fire_last_10_burned_share"] == 1.).all()


def test_provider_appends_named_columns_to_its_base(fire):
    matrix, names = FireContext(FakeBase(), fire)(2015, np.arange(64))
    assert names[0] == "base" and len(names) == matrix.shape[1]
    assert all(name.startswith("fire_") for name in names[1:])
