import numpy as np

from local_landuse.annual_vegetation import annual_features, history_statistics


def test_annual_history_separates_trend_outlier_and_missingness():
    values=np.column_stack([.2+.02*np.arange(5),[.2,.22,.24,.26,.9],[np.nan,np.nan,.4,np.nan,np.nan]]).astype(np.float32)
    stats=history_statistics(values)
    np.testing.assert_allclose(stats['linear_slope'][0],.02,atol=1e-6)
    np.testing.assert_allclose(stats['robust_slope'][:2],.02,atol=1e-6)
    assert stats['current_anomaly'][1]>.6
    assert stats['linear_slope'][2]==0 and stats['valid_years'][2]==1
    assert all(np.isfinite(v).all() for v in stats.values())


def test_annual_features_request_only_five_past_and_starting_years():
    requests=[]
    def read(year,indices):
        requests.append(year)
        return np.full((24,len(indices)),.3+.01*(year-2000),dtype=np.float32)
    X,names=annual_features(2010,np.array([0,1]),read)
    assert requests==[2010,2009,2008,2007,2006]
    assert len(names)==len(set(names))==X.shape[1]
    assert np.isfinite(X).all()
    np.testing.assert_allclose(X[:,names.index('annual_history_ndvi_harmonic_mean_robust_slope')],.01,atol=1e-5)
