import copy

import numpy as np
import rasterio

from local_landuse.models import NODATA
from local_landuse.temporal_memory import CANDIDATES, memory_features, run_memory
from local_landuse.source_ensemble import run_ensemble


def write_tif(path, a):
    with rasterio.open(path, 'w', driver='GTiff', width=a.shape[1], height=a.shape[0],
                       count=1, dtype=a.dtype, crs='EPSG:4326',
                       transform=rasterio.Affine(.01, 0, -13, 0, -.01, 10)) as dst:
        dst.write(a, 1)


def test_memory_missing_older_and_reversal_are_distinct():
    current = np.array([[1, 2], [1, 2]])
    previous = np.array([[2, 1], [1, 2]])
    older = np.array([[1, 1], [NODATA, 2]])
    X, names = memory_features(current, previous, older, {'height':np.ones((2,2))},
                               [1,2],np.arange(4),[(1,2),(2,1)],older_lag=4)
    np.testing.assert_array_equal(X[:, names.index('returned_to_older_class')], [1,0,0,0])
    np.testing.assert_array_equal(X[:, names.index('older_history_available')], [1,1,0,1])
    np.testing.assert_array_equal(X[:, names.index('observed_stable_duration_lower_bound')], [0,0,5,9])
    assert np.isfinite(X).all()


def test_memory_pooled_run_never_uses_final_labels(tmp_path):
    rng=np.random.default_rng(17)
    x=rng.random((30,30)).astype(np.float32)
    for year, threshold in [(2001,.3),(2005,.4),(2010,.55),(2015,.7),(2020,.8)]:
        write_tif(tmp_path/f'Guinea_LandCover_{year}.tif',np.where(x<threshold,1,2).astype(np.uint8))
    write_tif(tmp_path/'Guinea_SRTM_DEM_Slope_Aspect.tif',x)
    specs=copy.deepcopy([CANDIDATES[0],CANDIDATES[-1]])
    specs[-1]['expanded']=True
    for spec in specs:spec['trees']=3
    options=dict(data_dir=tmp_path,candidates=specs,max_samples=800,block_size=5,jobs=1)
    first=run_memory(tmp_path/'first',**options)
    first_ensemble=run_ensemble([tmp_path/'first'],tmp_path/'first_ensemble',data_dir=tmp_path)
    write_tif(tmp_path/'Guinea_LandCover_2020.tif',np.full((30,30),99,dtype=np.uint8))
    second=run_memory(tmp_path/'second',**options)
    second_ensemble=run_ensemble([tmp_path/'first'],tmp_path/'second_ensemble',data_dir=tmp_path)
    assert first_ensemble['policy']==second_ensemble['policy']
    assert first_ensemble['selected_model']==second_ensemble['selected_model']
    with rasterio.open(tmp_path/'first_ensemble/final_source_ensemble.tif') as a, \
            rasterio.open(tmp_path/'second_ensemble/final_source_ensemble.tif') as b:
        np.testing.assert_array_equal(a.read(),b.read())
    assert first['selected_model']==second['selected_model']
    assert first['manifest']==second['manifest']
    expanded=np.load(tmp_path/'first/expanded_train_indices.npy')
    groups=(expanded//30//5)*1000+(expanded%30//5)
    assert set(groups)<=set(np.load(tmp_path/'first/train_blocks.npy'))
    assert not set(groups)&set(np.load(tmp_path/'first/test_blocks.npy'))
    for spec in specs:
        for suffix in ('','_fixed_demand'):
            with rasterio.open(tmp_path/f"first/final_{spec['name']}{suffix}.tif") as a, \
                    rasterio.open(tmp_path/f"second/final_{spec['name']}{suffix}.tif") as b:
                np.testing.assert_array_equal(a.read(),b.read())
