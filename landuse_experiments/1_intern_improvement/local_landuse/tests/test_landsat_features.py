import json

import numpy as np
import pytest
import rasterio

from local_landuse.landsat_features import FEATURE_NAMES, clear_native, spectral_indices
from local_landuse.spectral_models import HistoricalSpectra


def test_native_qa_and_spectral_indices():
    qa=np.array([64,192,65,66,68,72,80,96,0],dtype=np.uint16)
    saturation=np.zeros_like(qa)
    np.testing.assert_array_equal(clear_native(qa,saturation),[1,1,0,0,0,0,0,0,0])
    saturation[0]=1
    assert not clear_native(qa,saturation)[0]
    cube=spectral_indices(np.array([[[.2]],[[.6]],[[.3]],[[.1]]],dtype=np.float32))
    np.testing.assert_allclose(cube[4:,0,0],[.5,1/3,5/7],rtol=1e-6)
    assert np.isnan(spectral_indices(np.zeros((4,1,1),dtype=np.float32))[4:]).all()


def fixture_spectra(root, year, profile, names, values, coverage=1.):
    directory=root/str(year);directory.mkdir()
    with rasterio.open(directory/'landsat_dry_season.tif','w',**dict(profile,count=len(names),dtype='float32',nodata=np.nan)) as dst:
        dst.write(values)
        for i,name in enumerate(names,1):dst.set_band_description(i,name)
    (directory/'manifest.json').write_text(json.dumps(dict(year=year,
        date_window=[f'{year}-01-01',f'{year}-04-30'],features=names,land_coverage_fraction=coverage)))


def test_spectra_keep_missing_pixels_and_use_only_current_and_past(tmp_path):
    names=['mean_'+n for n in FEATURE_NAMES]+['std_'+n for n in FEATURE_NAMES]+['clear_observation_count']
    profile=dict(driver='GTiff',height=3,width=3,count=1,dtype='uint8',crs='EPSG:4326',
                  transform=rasterio.Affine(.01,0,-13,0,-.01,10))
    reference=tmp_path/'reference.tif'
    with rasterio.open(reference,'w',**profile) as dst:dst.write(np.ones((3,3),dtype=np.uint8),1)
    for year,value in [(2005,.2),(2010,.4),(2015,.8)]:
        values=np.full((15,3,3),value,dtype=np.float32)
        values[:,0,0]=np.nan
        fixture_spectra(tmp_path,year,profile,names,values)
    provider=HistoricalSpectra(tmp_path,reference)
    X,columns=provider(2010,np.arange(9))
    assert X.shape[0]==9 and np.isfinite(X).all()
    assert X[0,columns.index('landsat_current_mean_ndvi_missing')]==1
    assert X[1,columns.index('landsat_annual_trend_ndvi')]==pytest.approx(.04)
    assert 2015 not in provider.cache
    earlier,earlier_names=provider(2005,np.arange(9))
    assert earlier_names==columns
    assert np.all(earlier[:,columns.index('landsat_previous_mean_ndvi_missing')]==1)
    manifest=tmp_path/'2010/manifest.json'
    data=json.loads(manifest.read_text());data['date_window'][1]='2015-01-01';manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='after'):
        HistoricalSpectra(tmp_path,reference)
def test_native_qa_average_does_not_mix_gap_or_cloud_values():
    from rasterio.transform import from_origin
    from local_landuse.landsat_features import average_clear_native
    import numpy as np
    reference = dict(height=1, width=1, transform=from_origin(0, 2, 2, 2), crs="EPSG:4326")
    values = np.array([[.2, .4], [10., 20.]], dtype=np.float32)
    clear = np.array([[1, 1], [0, 0]], dtype=np.float32)
    actual = average_clear_native(values, clear, from_origin(0, 2, 1, 1), "EPSG:4326", reference)
    np.testing.assert_allclose(actual, [[.3]], atol=1e-6)
    empty = average_clear_native(values, np.zeros_like(clear), from_origin(0, 2, 1, 1), "EPSG:4326", reference)
    assert np.isnan(empty).all()

