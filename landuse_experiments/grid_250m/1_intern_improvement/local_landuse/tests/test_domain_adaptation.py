import numpy as np
import pytest

from local_landuse.advanced import ChangeModel
from local_landuse.domain_adaptation import domain_feature_columns, estimate_density_weights


def test_domain_weights_favor_conditions_represented_at_forecast_start():
    rng = np.random.default_rng(9)
    historical = rng.normal(size=(600, 1)).astype(np.float32)
    starting = (historical + 1).astype(np.float32)
    groups = np.repeat(np.arange(12), 50)
    weights, report = estimate_density_weights(historical, starting, ['modis_current_ndvi_mean'],
        groups, jobs=1, trees=40)
    assert np.isfinite(weights).all() and np.all(weights > 0)
    assert abs(weights.mean()-1) < 1e-6
    assert weights[historical[:, 0] > .8].mean() > weights[historical[:, 0] < -.8].mean()
    assert report['oof_domain_auc'] > .65
    assert report['outcome_labels_used'] is False
    np.testing.assert_array_equal(domain_feature_columns(['future_class','previous_class_1',
        'modis_current_ndvi_mean','modis_current_ndvi_missing']), [2])


def test_external_weights_affect_change_detector_and_neutral_weights_preserve_rf():
    X = np.zeros((100, 2), dtype=np.float32)
    source = np.ones(100, dtype=int)
    target = source.copy(); target[:10] = 2
    spec = dict(family='rf', structure='two_stage', trees=30, depth=3, weight=1.)
    baseline = ChangeModel().fit(spec, X, source, target, [1, 2], jobs=1)
    neutral = ChangeModel().fit(spec, X, source, target, [1, 2], jobs=1, sample_weight=np.ones(100))
    np.testing.assert_array_equal(baseline.predict(X,source),neutral.predict(X,source))
    weighted = ChangeModel().fit(spec, X, source, target, [1, 2], jobs=1,
                                sample_weight=np.where(target!=source,20.,1.))
    assert weighted.predict(X,source)[0,1] > baseline.predict(X,source)[0,1]+.2
    with pytest.raises(ValueError,match='positive'):
        ChangeModel().fit(spec,X,source,target,[1,2],sample_weight=np.zeros(100))


def test_catboost_supports_action_and_destination_probabilities():
    pytest.importorskip('catboost')
    rng = np.random.default_rng(4)
    X = rng.normal(size=(90, 3)).astype(np.float32)
    source = np.tile([1, 2, 3], 30)
    previous = np.roll(source, 1)
    target = np.where(X[:, 0] > 0, previous, source)
    for structure in ('joint', 'two_stage', 'action'):
        model = ChangeModel().fit(dict(family='catboost',structure=structure,trees=3,depth=2,weight=1.),
            X, source, target, [1,2,3], jobs=1, previous=previous)
        p = model.predict(X,source,previous=previous)
        assert np.isfinite(p).all() and np.all(p >= 0)
        np.testing.assert_allclose(p.sum(axis=1),1.,atol=1e-6)
