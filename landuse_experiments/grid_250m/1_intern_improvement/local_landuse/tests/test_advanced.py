import copy

import numpy as np
import pytest
import rasterio
from sklearn.ensemble import RandomForestClassifier

from local_landuse.advanced import (CANDIDATES, ChangeModel, apply_decision,
    compact_metrics, decision_scores, landscape_features, run_advanced, tune_decision)
from local_landuse.comparison import evaluate
from local_landuse.models import NODATA


def test_destination_sensitive_fom_and_decision_exclude_persistence():
    source = np.array([1, 1, 1, 1, 2])
    observed = np.array([2, 3, 2, 1, 2])
    dest = np.array([2, 2, 2, 2, 1])
    change = np.array([True, True, False, True, False])
    result = compact_metrics(source, observed, dest, change)
    assert result['figure_of_merit_pct'] == 25
    full = evaluate(source, observed, np.where(change, dest, source), np.ones(5, bool))['summary']
    for key in result:
        assert result[key] == full[key]
    probabilities = np.array([[.7, .2, .1], [.05, .9, .05]])
    destination, score = decision_scores(probabilities, np.array([1, 2, 3]), np.array([1, 2]), 'odds')
    np.testing.assert_array_equal(destination, [2, 1])
    np.testing.assert_allclose(score, [.2/.7, .05/.9])
    with pytest.raises(ValueError, match='Source class'):
        decision_scores(probabilities, np.array([1, 2, 3]), np.array([4, 2]), 'odds')


def test_two_stage_probabilities_are_normalized_and_constant_targets_work():
    rng = np.random.default_rng(8)
    X = rng.random((200, 3))
    source = np.where(X[:, 0] > .5, 1, 2)
    target = np.where(X[:, 1] > .65, 3 - source, source)
    spec = copy.deepcopy(CANDIDATES[8])
    spec['trees'] = 3
    spec['weight'] = 4
    for y in (target, source, np.ones(200, dtype=int)):
        model = ChangeModel().fit(spec, X, source, y, np.array([1, 2]), jobs=1)
        p = model.predict(X, source, batch_size=37)
        assert np.isfinite(p).all()
        assert np.all((p >= 0) & (p <= 1))
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-6)


def test_unweighted_rf_preserves_original_bootstrap_draws():
    rng = np.random.default_rng(4)
    X = rng.random((200, 4))
    source = rng.choice([1, 2], 200)
    target = rng.choice([1, 2], 200)
    spec = dict(CANDIDATES[0], trees=3)
    old = RandomForestClassifier(n_estimators=3, max_depth=20, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=1).fit(X, target)
    new = ChangeModel().fit(spec, X, source, target, np.array([1, 2]), jobs=1)
    np.testing.assert_array_equal(new.main.model.predict_proba(X), old.predict_proba(X))


def test_geometry_is_finite_with_absent_classes_and_missing_neighbors():
    current = np.array([[NODATA, 1, 2], [1, 1, 2], [1, 2, 2]])
    previous = current.copy()
    terrain = np.arange(9, dtype=float).reshape(3, 3)
    terrain[0, 0] = np.nan
    X, names = landscape_features(current, previous, {'terrain': terrain},
                                  [1, 2, 3], np.arange(1, 9))
    assert np.isfinite(X).all()
    assert len(set(names)) == X.shape[1]
    assert 'current_log_patch_area_pixels' in names


def test_action_model_maps_relative_actions_to_real_classes():
    spec = dict(CANDIDATES[0], structure='action', trees=3)
    X = np.arange(20).reshape(10, 2)
    source = np.full(10, 2)
    previous = np.full(10, 1)
    for target in (source, previous, np.full(10, 3)):
        model = ChangeModel().fit(spec, X, source, target, np.array([1, 2, 3]),
                                  jobs=1, previous=previous)
        p = model.predict(X, source, previous=previous)
        np.testing.assert_allclose(p.sum(axis=1), 1)
        np.testing.assert_array_equal(model.classes_[p.argmax(1)], target)
    with pytest.raises(ValueError, match='previous'):
        model.predict(X, source)


def test_threshold_search_and_frozen_policy_agree():
    source = np.ones(200, dtype=int)
    observed = np.where(np.arange(200) > 120, 2, 1)
    p = np.column_stack([np.linspace(.99, .01, 200), np.linspace(.01, .99, 200)])
    policy, best, _ = tune_decision(p, np.array([1, 2]), source, observed)
    prediction = apply_decision(p, np.array([1, 2]), source, policy)
    result = evaluate(source, observed, prediction, np.ones(200, bool))['summary']
    assert best['figure_of_merit_pct'] == result['figure_of_merit_pct']
    assert result['figure_of_merit_pct'] > 90


def write_tif(path, values):
    with rasterio.open(path, 'w', driver='GTiff', width=values.shape[1], height=values.shape[0],
                       count=1, dtype=values.dtype, crs='EPSG:4326',
                       transform=rasterio.Affine(.01, 0, -13, 0, -.01, 10)) as dst:
        dst.write(values, 1)


def test_advanced_final_labels_do_not_select_train_or_predict(tmp_path):
    rng = np.random.default_rng(17)
    x = rng.random((30, 30)).astype(np.float32)
    for year, threshold in [(2001, .3), (2005, .4), (2010, .55), (2015, .7), (2020, .8)]:
        write_tif(tmp_path / f'Guinea_LandCover_{year}.tif', np.where(x < threshold, 1, 2).astype(np.uint8))
    write_tif(tmp_path / 'Guinea_SRTM_DEM_Slope_Aspect.tif', x)
    spec = copy.deepcopy(CANDIDATES[0])
    spec['trees'] = 3
    larger = dict(spec, name='expanded_rf', expanded=True)
    action = dict(spec, name='action_rf', structure='action')
    adapted = dict(spec, name='adapted_rf', spectral=True, adapt=True)
    requested_years=[]
    def dated_features(year,indices):
        requested_years.append(year)
        return ((indices/900+(year-2000)*.01).astype(np.float32)[:,None], ['modis_current_ndvi_mean'])
    options = dict(data_dir=tmp_path, candidates=[spec, larger, action, adapted], max_samples=500,
                   expanded_samples=800, block_size=5, jobs=1, make_plot=False,
                   feature_provider=dated_features)
    first = run_advanced(output_dir=tmp_path / 'first', **options)
    write_tif(tmp_path / 'Guinea_LandCover_2020.tif', np.full((30, 30), 99, dtype=np.uint8))
    second = run_advanced(output_dir=tmp_path / 'second', **options)
    assert first['selection'] == second['selection']
    assert first['expanded_sample_sha256'] == second['expanded_sample_sha256']
    assert set(requested_years)=={2005,2010,2015}
    for phase in ('validation','final'):
        np.testing.assert_array_equal(np.load(tmp_path/f'first/{phase}_adapted_rf_sample_weights.npy'),
                                      np.load(tmp_path/f'second/{phase}_adapted_rf_sample_weights.npy'))
    expanded = np.load(tmp_path / 'first/expanded_train_indices.npy')
    blocks = (expanded // 30 // 5) * 1000 + (expanded % 30 // 5)
    assert set(blocks) <= set(np.load(tmp_path / 'first/train_blocks.npy'))
    assert not set(blocks) & set(np.load(tmp_path / 'first/test_blocks.npy'))
    for name in [spec['name'], larger['name'], action['name'], adapted['name']]:
        for suffix in ('', '_fixed_demand'):
            with rasterio.open(tmp_path / f'first/final_{name}{suffix}.tif') as a, \
                    rasterio.open(tmp_path / f'second/final_{name}{suffix}.tif') as b:
                np.testing.assert_array_equal(a.read(), b.read())
