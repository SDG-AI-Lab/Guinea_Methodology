import numpy as np

from local_landuse.modis_phenology import TARGET_DAYS, valid_qa
from local_landuse.phenology_models import trajectory_statistics
from local_landuse.vegetation_context import masked_moments


def test_modis_quality_rejects_cloud_aerosol_snow_shadow():
    qa=np.array([0,1,2,3,3<<6,1<<8,1<<10,1<<14,1<<15,15<<2],dtype=np.uint16)
    np.testing.assert_array_equal(valid_qa(qa),[1,1,0,0,0,0,0,0,0,0])


def test_seasonality_is_finite_with_missing_periods_and_recovers_shape():
    phase=2*np.pi*np.asarray(TARGET_DAYS)/365.25
    values=np.column_stack([.5+.2*np.cos(phase),np.full(12,np.nan),np.full(12,.7)]).astype(np.float32)
    statistics=trajectory_statistics(values)
    assert all(np.isfinite(x).all() for x in statistics.values())
    assert abs(statistics['harmonic_amplitude'][0]-.4)<.02
    assert statistics['valid_periods'][1]==0
    assert statistics['harmonic_amplitude'][1]==0
    assert abs(statistics['mean'][2]-.7)<1e-6


def test_neighborhood_moments_exclude_missing_values_and_preserve_support():
    values = np.full((5, 5), .6, dtype=np.float32)
    values[2, 2] = np.nan
    mean, std, support = masked_moments(values, np.isfinite(values), 3)
    assert np.isfinite(mean).all() and np.isfinite(std).all()
    assert abs(mean[2, 2]-.6) < 1e-6
    assert std[2, 2] < .001
    assert abs(support[2, 2]-8/9) < 1e-6
    empty = masked_moments(values, np.zeros_like(values, dtype=bool), 3)
    assert all(np.all(grid == 0) for grid in empty)
